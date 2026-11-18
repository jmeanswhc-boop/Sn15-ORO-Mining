"""Raikou: buys for one shopper inside an ORO store world.

Built on KYO v6 (public ORO agent code, race 10 winner); this fork adds the
parts listed below.

A light model walks the store with the world's own tools. Code keeps the
ledger (items, needs, budget, listings seen), plans the questions to the
shopper, and stops wasted steps. A heavy model reads the request once and
checks the cart before an order and the evidence before a decline.

Added in this fork:
- store replies that write listings as text ('key: value | ...' fields, facts as
  'name=value; ...', race 15) are read into records once for every reader; the walker
  is still shown the store's own text;
- a clock: the episode is wrapped up from 200 s, no agent turn starts that would end
  past 252 s, the best cart is then checked out, helper calls that can wait are
  skipped late and late calls get a shorter timeout (race 10 lost runs to the 300 s limit);
- one run key pays for every task, and a race run's key stops at a set spend: every
  model call stops at a per-task spend limit, the late checks and the answer readings
  take the light model first once a task has spent a share, and the tool loop stops
  earlier (race 14: both race runs reached the key's limit and lost their last tasks);
- the store's keyword search is learned as any-word or all-word matching;
- cart replies are read back by shape: the store's listed lines replace Raikou's
  own record, and whether an add puts one unit, sets the count or adds it is learned;
  a line short of its item's quantity holds the checkout once, and the code's own
  cart calls (and a forced checkout) top short lines up; a reply that a note wrapped
  under a random key is read through the wrap (race 11 wrapped cart replies this way);
- when the shopper lowers the budget, code lists each item whose cart line no longer
  fits under the room left (the category listing, or a search for the category's
  name), opens the cheapest rows and names them before any ending without a purchase
  (race 11: a budget cut came at the first checkout in 30% of episodes);
- under a cheapest pick, the store's category listing (named once by the light model
  from the schemas) lists each cart line's category under its price, so the last
  check weighs listings keyword searches missed;
- before an ending without a purchase, code searches each item by a wanted value and
  opens new candidates; a forced finish with an empty cart is filled from review
  picks no check turned down;
- a checkout of an empty or short cart is held, and a commit before enough listings
  are open waits (the questions are chosen from opened listings);
- the planner sees how each key's values split the in-stock, in-budget listings, keys
  with one value on every listing are left out, and an answer with no usable
  question falls back to the listings' differing keys;
- a reading of the request that names no checkout tool is followed by one call that
  only maps the tools to their jobs;
- planner and tracing prompt changes (skip keys every listing shares, ask the key
  that names the product's own kind, give search words for unseen chain links);
- the pointer chase is bounded (pages, store calls, its own time share, no late
  start), searches only a shop or brand name the texts give (never a reader's own
  wording), and leaves a requirement that points at the listing chosen
  for another requested item to that item's pick (race 13: unbounded chases ran 45
  episodes out of time);
- every message goes to questions, and one message names up to eight properties of one
  item (race 14: the shopper answers whichever named property it holds a preference
  on, so one-property questions found it half as often); messages go one at a time and
  each answer is read before the next. Each queued property carries the odds that it
  holds the item's unwritten preference (the planner's rank at first); a message that
  drew nothing lowers the odds of what it named by how surely each place in the list was
  weighed (race 15: the 1st always, 2nd to 4th about nine times in ten, later ones far
  less often), so the next message about the item leads with the likeliest properties
  left; after a preference the properties named before it settle and the ones after it
  stay in front; an answer that did not take the message in moves the next message to
  another item (race 15: the same item puzzled the shopper again 100 times in 103,
  another item about three times in four), and items take turns: ones that have not
  puzzled the shopper first, then ones whose preference has not come out, then the one
  asked least;
- pointer chains are followed code-led (race 13: every agent opened the first named
  listing, few got further): a light reading names each link's shop, kind and details;
  code searches that shop by name, ranks its rows by the details, opens the best few
  (a second result page when the first shows no sign of them) and keeps the page that
  carries every detail in its attributes or its own sentences, a light reading of the
  page confirms it and gives the value, a pairing rule for it, or the next link;
- items that must work with another item's pick are paired by their codes: one light
  reading lists each row's own code, what it fits (ranges and exceptions too), the part
  its list goes on at and notes about other parts; code follows those parts, works out
  which listing works with which and the cheapest sets inside the budget, names them
  to the agent, holds a checkout whose lines do not work together and names working
  sets before an ending without a purchase;
- questions: a key on which every other listing shares the value of the listing that
  would be bought now cannot change the pick and goes last in the fallback; with items
  tied to another item's pick, the anchor item's keys are asked first (race 13: 20 of
  23 such tasks held the unwritten preference on the anchor).
"""
from __future__ import annotations

import difflib
import json
import random
import re
import time
from typing import Any

from src.agent.proxy_client import ProxyClient

LIGHT = "anthropic/claude-haiku-5.5"
HEAVY = "anthropic/claude-sonnet-5.5"
SPARE = "openai/gpt-6-luna"

LOOP_CALLS = 42
ALL_CALLS = 68
# spend per task: one run key pays for all of a run's tasks and stops at a set limit (a race run's limit divided by
# its tasks is about 0.055), so a task stays well under that on average
SPEND_SOFT = 0.040          # helper readings and checks that can wait stop here
SPEND_HARD = 0.058          # the tool loop stops here
SPEND_LIGHT = 0.030         # above this the late checks and the answer readings try the light model first
SPEND_STOP = 0.068          # no model call goes out above this: the code finishes the task on its own
GATE_ROOM = 0.012           # a step the request asks for at the order or the ending may still spend this much above the stop
PACK = 8                    # properties one message to the shopper names at most
# race 15, messages naming several properties of one item, a preference found on the same task in another episode:
# how often the shopper weighed it by its place in the list (1st 59 of 59, 2nd 40 of 44, 3rd 17 of 23, 4th 14 of 15,
# 5th 5 of 11, 6th 2 of 5, 7th 0 of 3), and how often the planner's ranks held it (1st and 2nd about half the time)
HEARD = (1.0, 0.9, 0.8, 0.8, 0.5, 0.4, 0.2, 0.1)
RANK_ODDS = (0.26, 0.22, 0.12, 0.10, 0.07, 0.05, 0.04, 0.035)
# waits before resending a failed model call; an episode has 300 s in all, so at most 45 s go to waiting
WAITS = (4, 10, 20, 35)
WAIT_TOTAL = 45
SHOW_CHARS = 450
OBS_LIMIT = 15000
HISTORY_LIMIT = 150000
GATE_ROUNDS = 3             # store calls the process gate may make before one checkout or ending
AUTO_PAGES = 6              # further search pages Raikou reads itself for a cheapest, most or least pick
CHEAPER_OPENS = 3           # cheaper rows of a cart line's kind whose attributes Raikou reads before a cheapest checkout
WRAP_AT = 200.0             # seconds into the episode: from here the agent is told to check out now
FINISH_AT = 252.0           # the loop stops here and the best cart is checked out (an episode has 300 s)
LATE_AT = 236.0             # after this, helper calls that can wait are skipped
SPEND_FINAL = 0.064         # the last cart check before a checkout still runs up to this spend
REF_STEPS = 16              # store calls the code's own chase of pointed-at listings may make in all
REF_HOPS = 6                # pages one pointer requirement's chase may follow
REF_OPENS = 3               # rows a chase opens to find the one listing a link describes
REF_READS = 14              # light readings of requirements and pages the chases may make in all
PAIR_ROWS = 30              # listing rows the compatibility reading looks at (12 a reading)
PAIR_SEARCHES = 3           # searches for listings that hold the rest of a compatibility list
CHASE_SECS = 90.0           # seconds one chase may take in all
CHASE_START = 170.0         # seconds: no chase starts later (the questions and the cart still need the time)
CHASE_STOP = 195.0          # seconds: no chase round starts later
SWEEP_STEPS = 4             # store calls the code's own search before an ending without a purchase may make

# one attempt per call (think() resends): a stuck call costs 50 s, not minutes, inside the 300 s
_llm = ProxyClient(timeout=50, max_retries=1, retry_delay=1)
# late in the episode a call gets less time: one started at the last moment still ends before 300 s
_llm_late = ProxyClient(timeout=25, max_retries=1, retry_delay=1)
# seconds since an arbitrary point: only differences are used (time spent in the episode), never the time of day
_now = time.monotonic
_env = ProxyClient(timeout=95, max_retries=3, retry_delay=1)


GUIDE = """You are Raikou, a careful buyer working for one shopper in an online store. You can act only by calling the store's tools. The episode is graded 1 or 0: it earns 1 only when the finish is exactly right. A purchase must contain every requested item at the requested quantity, each meeting every requirement in force when you check out, inside the budget; or, when the store truly has nothing that meets everything, the episode must end with the no-purchase tool in the way the request asks.

HOW THE STORE WORKS
- Every tool call is one step and steps are limited; thinking costs nothing. Tool names, argument names and result shapes are particular to this store: read each tool's description and schema and use the exact argument names and allowed values.
- Search tools can be narrow: a category filter, a keyword mode that may only read the first words of the query, a page size of a few rows, a token or offset for the next page. Compare the reported total with the rows you have seen and page on (with the exact token or offset the store gives) when the answer depends on seeing all of them. Keyword queries work best short and value-first (the wanted value plus the product noun). A filtered search that returns no rows usually means the filter value or the filter parameter is wrong: retry with the label exactly as result rows show it, or with another filter parameter.
- Search rows are summaries. Stores split a listing's record differently: one view can show its attributes (properties), another only its offer and description, and a side-by-side view may carry either. The ledger names the tool that shows attributes once Raikou knows it. Judge every requirement on a property from those attributes, so read them for each listing you may buy; one call that shows several listings saves steps when it shows what you need. Never open a listing again when nothing has changed since you read it.
- Questions to the shopper are handled for you by Raikou's interviewer, so you never message the shopper yourself. Its answers arrive in the conversation and are added to the ledger.
- The request can also ask for steps along the way: an approval, a message, something to check or read first. The ledger lists them. Raikou carries out the ones due just before the order or the ending itself when you call the checkout or the ending tool, so never wait for those and never end without a purchase because of them; do any others with the store's tools (product questions go through Raikou's interviewer).

WHAT COUNTS AS A FIT
- The kind of product is decided by the listing's category label: only the label that matches the requested item as a whole counts, never a neighbouring one.
- A requirement on a property is met when the listing's attribute for that property states the value. Key spelling does not matter (spaces, underscores, camel case, prefixes), and one element of a comma or slash list is enough. Values match by their words: case, singular or plural and plain synonyms are fine, and a value that contains the required value's words counts ("deep teal" meets teal, "extra roomy fit" meets roomy). A value with different words does not count (an abbreviation or code is not the word it stands for, and "teal green" is not "sea green"). Units are converted before comparing numbers, limits include the edge value, and a range must lie wholly inside a limit.
- An attribute whose key names the requirement's property (under that name or a plain synonym) decides: if it says something else, the listing fails whatever other keys or the title say. A key for a different property, even a related one (a handle colour is not the body's colour), decides nothing about the requirement. Raikou adds [evidence] lines after opened listings with the keys whose names match a requirement.
- When the listing has no attribute for that property, the wanted value stated under another attribute (for example a list of styles or features that includes it) or in the title or description counts. A number of another kind does not (a weight is not a capacity), nor a bare figure whose text does not say what it measures.
- When the category label itself states a requirement's value (a season or audience in the label), that requirement is met by the category.
- A requirement can point at another listing instead of giving a value ("the same X as the listing sold by <shop> with <property>", "within N of the X on the page of ..."). Resolve it first: find that listing (search its category, brand, shop or words), open its page in full with the single-item or side-by-side tool (a search row is not enough), and when that page refers you to a further listing for the property, find and open that one the same way; the last page gives the value. The order counts only when every page of that chain was opened. When the wanted value is anything but that other listing's value, the rule excludes only listings sharing that exact value, and never end without a purchase because of it alone.
- When a listing you are about to buy sends you to another page for more about it, look at that page before you buy.
- When one requested item must fit another (an accessory for a device), choose the two together from the listings' own words (a list of what it fits, a model or series code that both pages carry). When no accessory that meets its own requirements fits the device you picked, switch to another device that meets its requirements and that such an accessory fits; never end without a purchase while that switch is untried.
- Respect exclusions and closed lists. Wishes ("ideally", "would be nice") are tie-breakers only.
- Never buy an item that the store shows as sold out; the total (price times quantity, summed over every line) may not exceed the budget; equal is fine.

CHOOSING
- When the request asks for the cheapest (lowest total, smallest price), the order must hold the lowest-priced listing among ALL listings that fit. Find the whole fitting set (page through it: result rows are often in no price order, and Raikou reads the next pages of a short result itself), check candidates in ascending price, and take the first that fully fits. When a search takes a maximum price, use it: list the category below the price of your current best pick, so only cheaper candidates are left to check.
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
"links": a list of {"item": the number of a requested item (counting the items list from 1) that must work with, fit or connect to the listing chosen for another requested item (by model, part or connector), "on": that other item's number}, one entry per such pair; [] when no item depends on another item's pick that way.
"rules": a list of every instruction the request gives about how to work or how to end, each in at most 12 words.
"wishes": a list of things the shopper would merely like.
"ask_first": a list of things the request explicitly says to check with the shopper before choosing (empty when none).
"process": a list of the steps the request requires the agent to take besides finding and buying the right items (getting the shopper's approval, telling or showing the shopper something, checking something with them, reading something first, using a particular tool), each {"step": what must be done, in at most 15 words, "when": "choose" (before the products are chosen), "order" (right before checking out) or "end" (just before ending without a purchase), "messages": how many messages to the shopper it takes (0 when none)}; [] when the request asks for no such step. Requirements on the items, the budget, stock, the limits on messages and steps, and how to end when nothing fits are not steps.
"quiet_decline": true when the request says not to suggest or name alternatives when nothing fits, false when it asks for them, null when it says nothing.
"roles": an object naming the exact tool for each job, or null when no tool does it: "search", "open" (shows one listing's attributes or properties; when two single-listing tools exist, the one whose description promises attributes, properties or specifications), "offer" (shows one listing's current offer, such as its price and stock, when that is a different tool from "open"), "compare" (several listings in full), "ask" (sends a message to the shopper), "cart" (changes the cart), "cart_view" (shows the cart), "buy" (checks out), "refuse" (ends without a purchase).
"buy_args": the fixed arguments that make the "buy" tool check out (for a tool with a mode argument, that mode), else {}.
"refuse_args": the fixed arguments that make the "refuse" tool end without a purchase, else {}.
"""


PLAN = """You choose the questions a shopping agent sends to a shopper. Shoppers hold a few firm preferences they did not write down, each about one property of one requested product. A preference comes out only when a message names that property. One message can name several properties of one item: the shopper answers about the named one it holds a preference on, or says it does not mind any of them, so you rank properties for each item and the agent names the best few in one message and the next ones in a later message.

You get the request, the items with what the request already settles for each, and a numbered list of attribute keys seen on opened listings of each item's category (counted over the listings in stock and inside the budget when there are enough of them), with how many of those listings carry the key and how its values split them ("black x3" means three of them say black). Keys whose value is the same on every one of three or more listings are already left out. An unwritten preference may be on a property that no listed key names.

Choose the properties in this order:
0. What the request itself asks you to check with the shopper (listed under ASK FIRST when there is any), each named by the property it is about.
1. Hinted properties. The request's story often points at an unwritten preference without giving its value: a sentence about something the shopper cannot judge or leaves to you, someone whose taste decides, the product's looks or the setting it belongs in, the person meant to use or wear it, or what is inside it. For each such sentence, name the one property it points at and the item it is about, in the order the story gives them.
Name a hinted property by its plain name: a hint about who will wear it points at gender, a hint about its looks at colour or finish (whichever the request leaves open).
2. Then the listed keys, most likely first: keys carried by most of the item's listings whose values split them. Unwritten preferences sit on every kind of property (what kind or type of product it is, how it is built, what it connects to or is for, who it is for, a size or measure, what it is made of, its looks), so let how many listings carry a key and how its values split them decide the order, not the key's kind; avoid keys seen on only one or two listings.
Skip a property the request already settles for that item, a second name for a property already chosen, a key whose values each list several purchase variants at once (a size, a colour and a count joined in one value), identifiers, warranty and other logistics. A key whose value is the same on every listing shown cannot be an unwritten preference: skip it. When one key names the product's own kind (a "<kind> type" or "<kind> style" key next to a bare "type"), ask that key. Rank every item on its own: its hinted properties first, then its listed keys.
"label" is the property's plain name: for a listed key, all of the key's words in plain spelling ("lid colour" for lidColour, "rim finish" for rim_finish; never shortened to "colour" or "finish"); only the property's name, no example values, no brackets, no second property.
Choose up to COUNT for each item, best first. Return ONE JSON object and nothing else: {"ask": [{"item": item number, "label": "...", "hint": "at most 8 of the story's words it comes from, or an empty string"}]}"""


HEAR = """A shopping agent sent a shopper some messages; one message can ask about several properties of one item. Read each answer and return ONE JSON object and nothing else:
{"needs": [{"item": item number, "need": "property: constraint", "points": true when the value is to be read from another listing, else false}], "drop": [{"item": item number, "property": "a requirement the shopper now gives up"}], "free": [{"item": item number, "property": "..."}], "kinds": [one word per message, in order: "preference" when its answer states a firm preference, "free" when it says the shopper does not mind, "unclear" when it did not take the message in (asks for it again, more simply or in other words, or says the shopper did not follow it), "other" otherwise], "budget": new total limit as a number or null}
- "needs": every firm preference an answer states, with the exact value or limit and its unit ("at least 30 cm", "oval", "not wicker"). Name the need by the property the answer itself names, which may differ from the one asked about; several answers about one property make one need. Copy the value word for word as the answer says it, keeping its nouns and units ("at least 3 shelves", not "at least 3").
- An answer can give the value by pointing at another listing (a value like, unlike or near the one on some shop's product, that product picked out by a few of its properties): that is ONE need on the property the answer names, with "points": true. Its value is the answer's own words copied exactly, from the comparison words to the end of that description, including what it says about a listing that page may name; never shorten or reword it. The properties describing the other product are not needs of this item.
- "free": for an answer that says the shopper does not mind (any will do, no view, the agent may choose), every property its message named, each as its own entry; an answer that states a preference frees nothing.
- Folksy or roundabout wording that names a value still states that value."""


NOTE = """While a shopping agent works, the store showed this note (it may come from the shopper or the store). Return ONE JSON object and nothing else:
{"kind": "change" when the note changes the task (budget, requirements, a price, availability, a refusal of the order), "error" when it only reports that a call was malformed or not accepted, "other" for anything else, "budget": the new total limit as a number in the request's unit, or null, "add": [{"item": item number, "need": "property: constraint"}], "drop": [{"item": item number, "need": "the requirement no longer wanted"}], "gone": [listing ids said to be unavailable], "sold_out": true when it says a chosen listing is gone, sold out or no longer available (even without naming it), "price_moved": true when it says a price changed, "summary": "one short sentence"}
A replaced requirement is a drop of the old one plus an add of the new one. Report every change in the note."""


CHECK = """You are the last check before a shopping agent checks out. Decide from the records shown whether this cart is exactly right.
Reading rules: a requirement is met when the listing's attribute for that property states the value (any key spelling, one element of a list is enough, case, singular/plural and plain synonyms count, a value containing the required value's words counts, such as "deep teal" for teal, units converted, limits include the edge, a range must lie wholly inside); an attribute whose key names the same property as the requirement decides, and if it says something else the listing fails; a key for a different property, even a related one (a handle colour is not the body's colour), decides nothing; when the listing has no attribute for that property, the wanted value stated under another attribute or in the title or description counts (a number of another kind does not: a weight is not a capacity; nor does a bare figure whose text does not say what it measures); a requirement the category label itself states is met by that category; the category label must match the requested item as a whole; a rule of being unlike another listing's value counts as met unless the records show that same value. Each requested item must appear as a separate cart line at the requested quantity, nothing else may be in the cart, nothing shown out of stock, and the total of price times quantity over all lines must be within the budget. The newest record of a listing wins.
When the pick rule is cheapest, most or least, also look at the other listings shown: if one of them clearly fits every requirement and beats the cart's choice on that rule, the cart is wrong.
Return ONE JSON object and nothing else: {"ok": true or false, "problems": ["short, concrete problem", ...], "better": [listing ids that should replace a cart line, best first]}. Use ok=false only for a problem the records clearly show or a requirement no record addresses."""


READING = """the listing's attribute for a property decides when its key names that same property (any key spelling, one element of a list is enough, synonyms and unit conversions count, a value containing the required value's words counts, such as "deep teal" for teal, limits include the edge); a key for a different property, even a related one (a handle colour is not the body's colour), decides nothing; when no attribute names that property, the wanted value stated under another attribute or in the title or description counts (a bare figure that does not say what it measures counts for nothing); a requirement the category label itself states is met by that category; the category label must match the item; a rule of being unlike another listing's value counts as met unless the records show that same value; out-of-stock listings and listings that break the budget do not fit."""

AUDIT = """A shopping agent wants to end without buying because it thinks nothing fits. Check the listing records shown against the requirements, with these reading rules: """ + READING + """
Records marked ROW were seen only as a search result row: their page (the attributes) has not been read, so they can be neither passed nor failed on a property the row does not show.
Return ONE JSON object and nothing else: {"fits": {"item number": [listing ids that appear to meet every requirement, cheapest first, at most 4]}, "open": [ids of ROW listings of a requested kind, inside the budget, whose page could show that they fit, most promising first, at most 6], "reason": "one sentence"}"""

UNMIX = """A shopper's request was read into firm requirements on a product. Some requirements take their value from another listing the request points at, and the words that pick out that other listing (its kind, its shop, the properties named after it with "with ...") are sometimes read by mistake as requirements on the product being bought. Using the request's wording, say which of the OTHER REQUIREMENTS below only describe that other listing. Return ONE JSON object and nothing else: {"describe_other": [those requirements copied exactly; empty when every one is a real requirement on the product]}"""

TRAIL = """A shopping agent needs requirement values that live on other listings. Some requirements take their value from another listing instead of stating it, and a listing's page can send the reader on to a further listing for that property. The order counts only when the agent has opened in full every listing along each such chain. Follow each requirement's chain from the requirement's own description, link by link, through the pages' own words, using the records below (OPENED: the agent has opened that page; ROW: seen only as a search result row, its page not read).
Return ONE JSON object and nothing else: {"chains": [{"need": the requirement copied exactly, "item": the number of the requested item whose chosen listing the requirement points at, or null, "links": [{"id": the listing id when a record below is that link, else "", "about": how the requirement or the previous page describes that listing, "search": for a link no record below is: the shop, brand, model or part name the requirement or a page gives for it (copied exactly), then a few words of its kind; "" when the texts give no such name}], "done": true when the last link is an OPENED page that states the property's value itself, else false, "value": the property's value as stated on that last page, or ""}]}
A requirement that takes its value from the listing chosen for another item of this request (that item's own pick, not a listing to find) gets "item" and no links.
List each chain's links in order and stop at the first link whose page is not OPENED or not found, since its words are still unknown."""

LINK = """A shopping agent needs the value of one property (PROPERTY below) that a requirement takes from another listing. The text below is either the requirement itself or a listing's page, and it either gives that value or sends the reader on to a further listing for it. Return ONE JSON object and nothing else:
{"is_it": for a page, true when the page is the listing described under LOOKING FOR (this item's own attributes, title or remarks carry each of those details), false when it shows another value for one of them or says it is not that listing, null when it cannot tell (null for the requirement itself),
"value": the value the text gives for PROPERTY, copied as written (from the listing's attributes, title or remarks), or "" when the text sends the reader on for PROPERTY or does not give it,
"next": the listing the text sends the reader to for PROPERTY, as {"shop": its shop or brand name copied exactly as the text writes it, "kind": the kind of product in the text's words, "details": [each value the text gives to pick that listing out, one entry each, with its property name when the text names one]}, or null when it sends the reader nowhere,
"pairs": [[a value of PROPERTY, the value the text pairs it with or says it goes with], ...] for every pairing rule the text gives for PROPERTY, else [],
"item": the number of the requested item whose own chosen listing the requirement points at, when it points at the listing picked for another item of this request rather than at a listing to find; else null}
A page that says PROPERTY is to be taken from another listing (rather than from this item) sends the reader on even when it lists a value of its own for PROPERTY: give "next" and an empty "value". A remark that other versions exist elsewhere sends the reader nowhere and describes those other listings, not this one; a remark about a different property says nothing about PROPERTY."""

CODES = """Some listings below name what they work with by model or part codes. For each listing record, say what its own texts state. Return ONE JSON object and nothing else: {"codes": [{"id": the listing id, "own": the model or part code that names this listing itself, or "", "fits": [every model or part code it says it works with or fits, one entry each, copied exactly; a range of codes as one entry, either as written with x for each open digit or as FIRST..LAST], "except": [codes it says it does not fit], "rest_at": the part code of another listing said to hold the rest of its list, or "", "notes": [{"part": another listing's part code, "fits": [codes the text says that part also fits]}]}]}
Copy every code exactly as written. Give an entry for each listing whose texts name any code; leave out the others."""

GATE = """A shopping agent is about to STAGE. The shopper's request asks for the steps below along the way. Decide from the numbered RECORD of what the agent has done whether each step due before this STAGE is done for the cart as it is now. A step counts as done only when a RECORD line shows it being done and, for a step about the cart or the order, that line comes after the cart last changed and its reply grants it; questions on a product's properties do not do a step.
When a step is still missing, give the one store call that does it next: a tool from TOOLS with its arguments. A message to the shopper goes through the messaging tool: one short sentence or question about that step alone, in plain words, naming no product properties or values the step does not need. Give no call when every due step is done or when the messages or steps left cannot do the missing one.
Return ONE JSON object and nothing else: {"steps": [{"step": the step, "done": true or false, "lines": [the RECORD line numbers that show it done]}], "call": {"tool": "...", "args": {...}} or null, "hold": what a reply asks to change before going on, or "", "needs": [{"item": item number, "need": "property: constraint"} for each firm requirement such a reply states]}"""

SWAP = """A listing in a shopping agent's cart has just sold out. Find replacements for that cart line among the other listing records shown, with these reading rules: """ + READING + """
Return ONE JSON object and nothing else: {"fits": [{"id": listing id, "price": its price as a number as the record shows it} for every listing that appears to meet every requirement of the sold-out line's item, at most 6], "reason": "one sentence"}"""

ROLES = """A shopping agent works a store through the tools below. Name the exact tool for each job, or null when no tool does it: "search" (finds listings), "open" (shows one listing's attributes), "offer" (shows one listing's current price and stock, when that is a different tool from "open"), "compare" (shows several listings in full), "ask" (sends a message to the shopper), "cart" (changes the cart), "cart_view" (shows the cart), "buy" (checks out), "refuse" (ends without a purchase). When one tool both checks out and ends without a purchase, also give the fixed arguments that choose each. Return ONE JSON object and nothing else: {"roles": {job: tool name or null}, "buy_args": {...} or {}, "refuse_args": {...} or {}}"""


LISTING = """A shopping agent wants the store to list every listing of one category (and, when a tool allows it, only those priced at most a given amount). From the tools below, name the call that does it, by the exact names the schemas use. Return ONE JSON object and nothing else: {"tool": the tool's name, or "" when no tool lists the listings of a category, "category_arg": the parameter that takes the category label, "price_arg": the parameter that takes the highest price, or "" when there is none, "fixed": {any other parameters and the values that make the tool list by category instead of matching words; {} when none}}"""


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


_FIELD = r"[a-z][a-z0-9_]{0,39}"


class _ReadDict(dict):
    """A store reply whose text-written listings were read into records; `raw` keeps the reply as the store wrote it."""
    raw: Any = None


class _ReadList(list):
    """The same for a reply that is a list."""
    raw: Any = None


def _field_value(text: str) -> Any:
    """A listing field's value in its plain type: two or more name=value parts joined by ';' as a dict, whole and
    decimal numbers as numbers (not one written with a leading zero, which reads like a code), the literal spellings of
    true, false and none as those values; anything else stays text."""
    pairs = _name_pairs(text)
    if pairs is not None:
        return pairs
    if re.fullmatch(r"-?(?:0|[1-9]\d{0,14})", text):
        return int(text)
    if re.fullmatch(r"-?(?:0|[1-9]\d{0,14})\.\d{1,6}", text):
        return float(text)
    if re.fullmatch(r"True|False|None", text):
        return {"True": True, "False": False, "None": None}[text]
    if re.fullmatch(r"true|false|null", text):
        return json.loads(text)
    return text


def _name_pairs(text: str) -> dict[str, str] | None:
    """'a=1; b=2' as {'a': '1', 'b': '2'}: two pairs at least, the first part a pair, no name holding a colon (prose such
    as 'Specs: size=XL; ...' is no attribute block); a part without '=' belongs to the value before it ('features=wifi;
    gps' keeps 'gps' with the features), and a name met again adds to its own value. None for any other text."""
    parts = [p.strip() for p in text.split(";")]
    head = parts[0].partition("=")
    if len(parts) < 2 or not head[1] or not head[0].strip() or len(head[0]) > 60 or re.search(r"[:<>{}\"]", head[0]):
        return None
    out: dict[str, str] = {}
    last = None
    for part in parts:
        name, mark, value = part.partition("=")
        name = name.strip()
        if mark and name and len(name) <= 60 and not re.search(r"[:<>{}\"]", name):
            if name in out:
                out[name] += "; " + value.strip()
            else:
                out[name] = value.strip()
            last = name
        elif last is not None and part:
            out[last] += "; " + part
    return out if len(out) >= 2 else None


def _listing_text(text: Any) -> dict[str, Any] | None:
    """A listing the store wrote as one text of 'key: value' fields (race 15: 'ref_ref: p-.. | title: .. | price: 230 |
    on_hand: True | facts: pack type=single; volume=490 ml; organic=no | item_blurb: ..'), read into a record of five
    fields or more; None for any other text. A field starts at the text's start or right after a separator followed by a
    lower-case key, a colon and a blank. The separator is the one that ends the first field: a '|' (also '|' and a line
    break) or a line break alone; a bar inside a value (a title's own bars), the other separator, and a key seen before
    stay in the value."""
    if not isinstance(text, str) or not 16 <= len(text) <= 60000:
        return None
    body = text.strip()
    first = re.match("(" + _FIELD + r"):(?:\s|$)", body)
    if not first:
        return None
    # the separator is matched first and the blanks around it are cut off afterwards (strip): a pattern opening with
    # blanks would retry every place of a long blank run
    marks = list(re.finditer(r"(\|[ \t]*\n?|\n)[ \t]*(" + _FIELD + r"):(?=\s|$)", body))
    if not marks:
        return None
    bar = marks[0].group(1).startswith("|")
    cuts = [m for m in marks if m.group(1).startswith("|") == bar]
    if len(cuts) < 4:
        return None
    fields: dict[str, str] = {}
    key, start = first.group(1), first.end()
    for mark in cuts:
        name = mark.group(2)
        if name == key or name in fields:
            continue
        fields[key] = body[start:mark.start()].strip()
        key, start = name, mark.end()
    fields[key] = body[start:].strip()
    if len(fields) < 5:
        return None
    return {k: _field_value(v) for k, v in fields.items()}


def _read_listings(value: Any, depth: int = 0) -> Any:
    """A store reply with its text-written listings read into records (_listing_text): such a text becomes its record, a
    list of them a list of records, and a reply of one or two fields (a page, or a reply with its notes beside it) has
    such a text field read; the text fields of a larger reply (a record of its own) stay as they are. The reply itself
    comes back when nothing in it was such a text."""
    if depth > 6:
        return value
    if isinstance(value, str):
        got = _listing_text(value)
        return value if got is None else got
    if isinstance(value, list):
        out = [_read_listings(v, depth + 1) for v in value]
        return out if any(a is not b for a, b in zip(out, value)) else value
    if isinstance(value, dict):
        if len(value) == 1:
            got = _listing_text(next(iter(value.values())))
            if got is not None:
                return got
        changed = False
        read: dict[str, Any] = {}
        for k, v in value.items():
            if isinstance(v, str):
                got = _listing_text(v) if len(value) <= 2 else None
                new = got if got is not None else v
            else:
                new = _read_listings(v, depth + 1) if isinstance(v, (list, dict)) else v
            changed = changed or new is not v
            read[k] = new
        return read if changed else value
    return value


def _with_raw(read: Any, raw: Any) -> Any:
    """The read reply, carrying the store's own text along (`raw`) for what the walker is shown."""
    box: Any = _ReadDict(read) if isinstance(read, dict) else _ReadList(read) if isinstance(read, list) else None
    if box is None:
        return read
    box.raw = raw
    return box


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


def _names(prop: frozenset[str], key: frozenset[str]) -> bool:
    if not prop or not key:
        return False
    return prop == key or prop <= key or (len(key) >= 2 and key <= prop)


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


def _num(value: float) -> str:
    """A price or limit as plain text: whole numbers without a decimal point, others to two places."""
    value = float(value)
    return str(int(value)) if value.is_integer() else f"{value:.2f}".rstrip("0").rstrip(".")


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
    if not isinstance(value, str) or not 3 <= len(value.strip()) <= 64 or " " in value.strip():
        return False
    return named or bool(re.search(r"\d", value))


def _learn_id(rows: list[dict[str, Any]]) -> str | None:
    """The key that identifies rows: string values with a digit and no spaces, unique per row, longest on average."""
    best = None
    for key in rows[0]:
        values = [row.get(key) for row in rows]
        if not all(_id_like(v, "id" in str(key).lower()) for v in values):
            continue
        values = [v.strip() for v in values]
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



def _numbers(text: str) -> list[float]:
    return [float(x) for x in re.findall(r"\d+(?:\.\d+)?", text or "")]


def _tokens(text: str) -> frozenset[str]:
    """The words of any text (JSON included): runs of letters or digits, lower case, a plural s dropped."""
    return frozenset(w[:-1] if len(w) > 3 and w.endswith("s") else w for w in re.findall(r"[a-z0-9]+", (text or "").lower()))


def _detail_score(detail: str, numbers: list[float], words: frozenset[str]) -> float:
    """How well one detail a link gives (a value, often with its property's name) matches a listing's text: every number
    in it must be among the text's numbers (else 0); then 1 for the numbers and the share of its other words found."""
    nums = _numbers(detail)
    if any(all(abs(n - m) > 1e-6 * max(1.0, abs(n)) for m in numbers) for n in nums):
        return 0.0
    mine = {w for w in _tokens(re.sub(r"\d+(?:\.\d+)?", " ", detail)) if len(w) >= 2}
    share = len(mine & words) / len(mine) if mine else 0.0
    return (1.0 if nums else 0.0) + share


def _code_tokens(value: Any) -> list[str]:
    """The code-shaped tokens of a text: runs of letters and digits joined by hyphens that carry both a letter and a
    digit, such as model and part numbers (a slash or a comma separates codes)."""
    found = re.findall(r"(?<![0-9A-Za-z])[0-9A-Za-z]+(?:-[0-9A-Za-z]+)*", str(value or ""))
    return [t.upper() for t in found if re.search(r"[A-Za-z]", t) and re.search(r"\d", t) and len(t) >= 3]


def _code_parts(entry: Any) -> list[str]:
    """The comma-, semicolon- or slash-separated parts of a listed entry that carry a code (a FIRST..LAST span stays
    one part)."""
    return [part.strip() for part in re.split(r"[,;/]", str(entry or "")) if _code_tokens(part)]


def _code_span(part: str) -> tuple[str, int, int, int] | None:
    """A listed range read by shape (prefix, low, high, digits): one code whose trailing x stand for any digits, or two
    codes with one prefix and numeric tails of one length written FIRST..LAST."""
    tokens = _code_tokens(part)
    if len(tokens) == 1:
        tail = re.fullmatch(r"(.*?)(\d*)(X+)", tokens[0])
        if tail:
            digits, open_ = tail.group(2), len(tail.group(3))
            return tail.group(1), int((digits or "0") + "0" * open_), int((digits or "0") + "9" * open_), len(digits) + open_
    ends = re.fullmatch(r"\s*([0-9A-Za-z-]+)\s*\.\.\s*([0-9A-Za-z-]+)\s*", part)
    if ends and len(tokens) == 2:
        a, b = (re.fullmatch(r"(.*?)(\d+)", t) for t in tokens)
        if a and b and a.group(1) == b.group(1) and len(a.group(2)) == len(b.group(2)) and int(a.group(2)) < int(b.group(2)):
            return a.group(1), int(a.group(2)), int(b.group(2)), len(a.group(2))
    return None


def _code_match(code: str, entry: Any) -> bool:
    """Whether a model code falls under a listed entry: one of its codes, or inside one of its ranges."""
    code = code.upper()
    found = re.fullmatch(r"(.*?)(\d+)", code)
    for part in _code_parts(entry):
        if code in _code_tokens(part):
            return True
        span = _code_span(part)
        if span and found and found.group(1) == span[0] and len(found.group(2)) == span[3] and span[1] <= int(found.group(2)) <= span[2]:
            return True
    return False


def _code_clean(entry: Any) -> bool:
    """Whether every part of a listed entry reads as plain codes or one range: no digits outside its codes, and each code
    with a single run of digits (else a reading of it may be partial)."""
    parts = _code_parts(entry)
    if not parts:
        return False
    rest = str(entry or "").upper()
    for token in _code_tokens(entry):
        rest = rest.replace(token, " ")
    if re.search(r"\d", rest):
        return False
    for part in parts:
        tokens = _code_tokens(part)
        if any(len(re.findall(r"\d+", t)) != 1 for t in tokens) or (len(tokens) > 1 and _code_span(part) is None):
            return False
    return True

class Item:
    def __init__(self, name: str, shelf: str, qty: int, musts: list[str]):
        self.name = name
        self.shelf = shelf
        self.qty = qty
        self.musts = musts
        self.told: list[str] = []
        self.dropped: list[str] = []
        self.free: list[str] = []


class Raikou:
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
        self.step = 0
        self.finished = bool(self.view.get("finished"))
        self.broke = False
        self.no_credit = False
        self.refused_by: set[str] = set()
        self.stop_room = 0.0        # extra spend a step the request asks for at the order or the ending may still use
        self.msg_error = False
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
        self.cart_lines: dict[str, int] = {}
        self.checked: set[str] = set()
        self.audited = False
        self.offer_seen: dict[str, int] = {}
        self.gone: dict[str, int] = {}
        self.unavailable: set[str] = set()
        self.sold_lines: set[str] = set()
        self.note_sold = False
        self.held = 0
        self.short_held = 0
        self.early_holds = 0
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
        # upgrades: the clock, what the store's own replies teach, the code's own chases and searches
        self.t0 = _now()
        self.cart_mode: dict[str, str] = {}
        self.word_match: str | None = None
        self.search_totals: dict[tuple[str, str, frozenset[str]], int] = {}
        self.ref_steps = 0
        self.chased: set[str] = set()
        self.ref_items: dict[str, int] = {}
        self.ref_paths: dict[str, list[str]] = {}
        self.link_searched: dict[str, Any] = {}
        self.link_reads = 0
        self.first_links: dict[str, dict[str, Any]] = {}
        self.link_refused: dict[tuple[str, tuple[str, ...]], set[str]] = {}
        self.chase_wait: dict[str, tuple[str, int]] = {}
        self.rechased: dict[str, int] = {}
        self.row_fields: set[str] = set()
        self.codes: dict[str, dict[str, Any]] = {}
        self.compat: dict[tuple[int, int], dict[str, set[str]]] = {}
        self.paired = 0
        self.paired_mark: tuple[int, ...] = ()
        self.pair_holds = 0
        self.pair_cheap_told = False
        self.pair_decline_told = False
        self.rest_searched: set[str] = set()
        self.swept = False
        self.shelved: set[str] = set()
        self.shelf_full: dict[str, float] = {}
        self.cart_truth_at = -1
        self.count_checked = ""
        self.search_template: tuple[str, dict[str, Any]] | None = None
        self.cart_list_key: dict[str, str] = {}
        self.rejected: set[str] = set()
        self.loop_secs: list[float] = []
        self.budget_moved = False
        self.budget_listed: set[str] = set()
        self.listing: tuple[str, str, str | None, dict[str, Any]] | None | bool = None

    # ---------- plumbing ----------
    def left(self) -> int:
        return self.max_steps - self.step

    def elapsed(self) -> float:
        return _now() - self.t0

    def turn_secs(self) -> float:
        """How long an agent turn usually takes here (the middle of the last few), so none starts that would end past FINISH_AT."""
        recent = sorted(self.loop_secs)
        return recent[len(recent) // 2] if recent else 0.0

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

    def think(self, messages: list[dict[str, Any]], models: list[str], max_tokens: int, tools: list[dict[str, Any]] | None = None) -> dict[str, Any] | None:
        for model in models:
            # past LATE_AT a call has the late client's 25 s at most, so the last one (started by 264 s) ends by 289 s;
            # above SPEND_STOP no call goes out at all (the run's key pays for every task)
            if self.calls >= ALL_CALLS or self.elapsed() > FINISH_AT + 12 or self.cost >= SPEND_STOP + self.stop_room or self.no_credit:
                return None
            self.calls += 1
            tries = 0
            while True:
                began = _now()
                message, status = self.call_model(model, messages, max_tokens, tools)
                if tools:
                    self.loop_secs = (self.loop_secs + [_now() - began])[-5:]
                if message is not None:
                    self.refused_by.clear()
                    return message
                if status == 403:
                    # refused outright: when two different models are refused in a row the run's key is spent
                    # (its limit is reached), so no further call can succeed and the code finishes the task alone
                    self.refused_by.add(model)
                    if len(self.refused_by) >= 2:
                        self.no_credit = True
                        return None
                # no credit, rate limits, timeouts and server errors pass: wait and send the same call again
                passing = status is None or status in (402, 408, 425, 429) or (isinstance(status, int) and status >= 500)
                # a timeout already took the client's 50 s: it is sent once more at most
                if not passing or tries >= (1 if status in (None, 408) else len(WAITS)) or self.waited + WAITS[tries] > WAIT_TOTAL or self.elapsed() + WAITS[tries] > FINISH_AT:
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
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        client = _llm_late if self.elapsed() > LATE_AT else _llm
        result = client.post_verbose("/inference/chat/completions", body)
        data = result.data
        if isinstance(data, dict) and data.get("choices"):
            usage = data.get("usage") or {}
            if isinstance(usage.get("cost"), (int, float)):
                self.cost += float(usage["cost"])
            message = data["choices"][0].get("message") or {}
            if message.get("content") or message.get("tool_calls"):
                return message, 200
            self.trace.append({"kind": "llm_fail", "model": model, "status": 0})
            return None, 0
        error = result.error or {}
        status = error.get("status")
        if isinstance(data, dict) and isinstance(data.get("error"), dict):
            # an error carried inside a 200 reply, with the provider's own code
            status = _count(data["error"].get("code"))
            error = {"body": data["error"].get("message")}
        self.trace.append({"kind": "llm_fail", "model": model, "status": status, "body": str(error.get("body") or "")[:200]})
        return None, status

    def ask_json(self, prompt: str, models: list[str], max_tokens: int = 1800) -> dict[str, Any] | None:
        for model in models:
            message = self.think([{"role": "user", "content": prompt}], [model], max_tokens)
            got = _json_obj((message or {}).get("content"))
            if got is not None:
                return got
            if self.broke or self.no_credit or self.calls >= ALL_CALLS:
                return None
        return None

    def act(self, name: str, args: dict[str, Any]) -> dict[str, Any] | None:
        payload = dict(self.binding)
        payload["step"] = self.step + 1
        payload["action"] = {"name": name, "args": args}
        refused = 0
        for _ in range(4):
            result = _env.post_verbose("/environment/call", payload)
            if isinstance(result.data, dict):
                data = result.data
                if name != self.roles.get("ask"):
                    # listings the store wrote as 'key: value' text (race 15) are read into records here, once, so every
                    # reader of the reply sees them; what the walker is shown stays the store's own text (show)
                    seen = data.get("observation")
                    read = _read_listings(seen)
                    if read is not seen:
                        data = {**data, "observation": _with_raw(read, seen)}
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
                # a step conflict, read by its shape: a body naming one step number other than the one sent names the step
                # the platform expects; one naming none refused a step already used, or an episode that is over. A try far
                # ahead tells which: a live episode answers with the step it expects, an ended one refuses again
                named = {int(n) for n in re.findall(r"(?<![\w.])\d{1,7}(?![\w.])", body)} - {payload["step"]}
                if len(named) == 1:
                    refused = 0
                    self.step = named.pop() - 1
                    payload["step"] = self.step + 1
                    continue
                if not named:
                    refused += 1
                    if refused >= 2:
                        break
                    payload["step"] = self.step + 1000
                    continue
                return None
            refused = 0
            if status in (400, 401, 402, 403, 404, 422):
                if status == 402:
                    self.broke = True
                return None
        if refused >= 2:
            # refused twice with no step named, the second time far ahead: the episode is over (finished, sealed or
            # quarantined)
            self.finished = True
        return None

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
        self.guess_roles()
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
            # the messages the step at the order takes stay back; the ending's step keeps one only when the questions still
            # get one message per item and one more: an ending without a purchase is rarely right, an unasked property often
            # is the shopper's unwritten preference
            at_order = max([p["messages"] for p in self.process if p["when"] == "order" and p["messages"]] or [0])
            at_end = max([p["messages"] for p in self.process if p["when"] == "end" and p["messages"]] or [0])
            room = self.msg_limit - max(at_order, at_end) >= max(1, len(_items(brief.get("items")))) + 1
            self.reserve = min(max(at_order, at_end) if room else at_order, self.msg_limit)

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
            when = {"choose": "before choosing", "order": "at the order (Raikou does it when you check out)", "end": "at an ending without a purchase (Raikou does it then)"}[entry["when"]]
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
        text = f"[Raikou] step {self.step} of {self.max_steps} used, {left} left."
        if left <= self.tail() + 1:
            text += " FINAL STEPS: check out now when the cart holds a line for each item; end without a purchase only when your search showed that an item has no fitting listing (running out of steps does not show that)."
        elif left <= 6 or (left <= len(self.items) + 5 and self.buys == 0):
            text += " Few steps left: set the cart to the chosen lines and check out now."
        elif self.buys == 0 and left <= 12:
            text += " Steps are running low: settle on your picks, fill the cart and check out within the next few steps."
        elif self.elapsed() > WRAP_AT:
            text += " TIME IS NEARLY UP: set the cart to the chosen lines and check out now; an episode that runs out of time earns nothing."
        elif self.buys == 0 and not self.cart_touched and self.step >= self.max_steps // 2:
            text += " Half the steps are used and the cart is empty: cart each item's fitting listing now (the store may still sell one out)."
        live = {need for _, need in self.needs_now()}
        lookups = [need for _, need in self.refs if need in live and need not in self.ref_values and need not in self.ref_items]
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
        """A listing's id, read with the id key learned from result rows; None for anything else."""
        if not isinstance(node, dict) or len(node) < 3:
            return None
        if self.id_key and _id_like(node.get(self.id_key), True):
            return node[self.id_key].strip()
        nested = any(isinstance(v, dict) for v in node.values())
        if not (in_list or depth == 0 or nested):
            return None
        found = _learn_id([node])
        if found and ("id" in str(found).lower() or node[found].strip() in self.rows):
            return node[found].strip()
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
            if isinstance(node, dict):
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
                    self.rows[ident].update(node)
                    if in_list:
                        self.row_fields.update(k for k, v in node.items() if isinstance(v, (str, int, float)) and not isinstance(v, bool))
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

    def item_number(self, label: str) -> int | None:
        """The requested item a category label belongs to (its shelf or name, or a long word they share), else None."""
        words = {w for w in _stem(label) if len(w) >= 4}
        for number, item in enumerate(self.items, 1):
            if _words(label) in (_words(item.shelf or ""), _words(item.name)):
                return number
        shared = [len(words & {w for w in _stem(item.shelf or "") | _stem(item.name) if len(w) >= 4}) for item in self.items]
        best = max(shared, default=0)
        return shared.index(best) + 1 if best else None

    def key_table(self) -> list[tuple[str, str, int, list[str], str]]:
        """Per category of a requested item: (category, key name, listings carrying it, values most common first, how the
        values split those listings). A hidden preference sits on the listing to be bought, so values are counted over
        the opened listings that are in stock and inside the budget once three of a category are; a key that shows one
        value on every opened listing of the category (three or more carry it) cannot tell listings apart and is left
        out, as are long free texts, lists of purchase variants and a longer key that repeats a shorter one's values."""
        cat_key = self.category_key()
        groups: dict[str, list[str]] = {}
        for ident in self.seen_order:
            if ident in self.full:
                label = str(self.full[ident].get(cat_key) or (self.rows.get(ident) or {}).get(cat_key) or "listings") if cat_key else "listings"
                groups.setdefault(label, []).append(ident)
        mine = {label: ids for label, ids in groups.items() if self.item_number(label) is not None}
        if mine:
            groups = mine
        table: list[tuple[str, str, int, list[str], str]] = []
        for label, ids in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            number = self.item_number(label) or 1
            qty = self.items[number - 1].qty if self.items else 1
            fit = [i for i in ids if self.buyable(i) and (self.budget is None or self.price_of(i) is None or (self.price_of(i) or 0.0) * qty <= self.budget)]
            pool = fit if len(fit) >= 3 else ids
            per: dict[str, dict[str, str]] = {}
            seen_all: dict[str, set[str]] = {}
            for ident in ids:
                for key, value in (_fact_block(self.full[ident]) or {}).items():
                    name = _words(key)
                    text = str(value).strip().lower()
                    if not name or not text:
                        continue
                    seen_all.setdefault(name, set()).add(text)
                    if ident in pool and ident not in per.setdefault(name, {}):
                        per[name][ident] = text
            rows: list[tuple[str, dict[str, str]]] = []
            for name, values in sorted(per.items(), key=lambda kv: (-len(kv[1]), -len(set(kv[1].values())))):
                texts = list(values.values())
                mean = sum(len(t) for t in texts) / len(texts)
                spread = len(set(texts))
                if mean > 40 or 2 * sum(1 for t in texts if " / " in t) >= len(texts):
                    continue
                # one value on every opened listing of the category (not only the buyable ones) tells nothing apart
                if (len(texts) >= 3 and len(seen_all.get(name, ())) == 1) or (len(texts) >= 4 and spread == len(texts) and mean > 12):
                    continue
                if any(name.startswith(other + " ") and values == seen for other, seen in rows):
                    continue
                rows.append((name, values))
            for name, values in rows[:14]:
                tally: dict[str, int] = {}
                for text in values.values():
                    tally[text[:40]] = tally.get(text[:40], 0) + 1
                ranked = sorted(tally, key=lambda t: -tally[t])
                shown = " | ".join(f"{t} x{tally[t]}" for t in ranked[:5]) + (f" | +{len(ranked) - 5} more values" if len(ranked) > 5 else "")
                table.append((label, name, len(values), ranked, f"on {len(values)} of {len(pool)} listings: {shown}"))
        return table

    def table_picks(self, table: list[tuple[str, str, int, list[str], str]], limit: int) -> list[tuple[int, str]]:
        """Questions when the planner's answer names none it can use: per item, the most carried keys whose values differ
        between listings and that the request does not settle, the items taking turns."""
        per: dict[int, list[str]] = {}
        for label, name, _carried, values, _shown in table:
            number = self.item_number(label) or 1
            settled = {_words(m.split(":", 1)[0]) for m in self.items[number - 1].musts}
            if len(set(values)) < 2 or name in settled or name in per.get(number, []):
                continue
            per.setdefault(number, []).append(name)
        picks: list[tuple[int, str]] = []
        while any(per.values()) and len(picks) < limit:
            for number in sorted(per):
                if per[number] and len(picks) < limit:
                    picks.append((number, per[number].pop(0)))
        return picks

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
        if shape["rows"] not in observation:
            # a store note wrapped the real reply under another key
            inner = [v for v in observation.values() if isinstance(v, dict) and shape["rows"] in v]
            if len(inner) != 1:
                return None
            observation = inner[0]
        rows, count, value = observation.get(shape["rows"]), observation.get(shape["count"]), observation.get(shape["next"])
        if not isinstance(rows, list) or not rows or isinstance(count, bool) or not isinstance(count, int):
            return None
        if value is not None and (isinstance(value, bool) or not isinstance(value, (str, int))):
            return None
        return count, len(rows), shape["next_arg"], value

    def price_sorted(self, observation: Any) -> bool:
        prices = [self.price_of(i) for i, _ in self.records_in(observation)]
        return len(prices) >= 3 and None not in prices and all(a <= b for a, b in zip(prices, prices[1:]))

    def more_pages(self, name: str, args: dict[str, Any], observation: Any) -> str:
        """A search that matched more listings than it showed, under a cheapest, most or least pick: Raikou reads a short
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
        pages = -(-(count - shown) // shown)
        if pages > 2 or self.pages_read + pages > AUTO_PAGES or self.left() - self.tail() - pages < 12:
            return (f"\n[Raikou] This search matched {count} listings and showed {shown}, in no price order. The pick counts every fitting listing: "
                    f"search with narrower words (the requirement's value) or page on with {param}={_dump(value)}.")
        texts = []
        for _ in range(pages):
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
        return ("\n[Raikou read the rest of this search itself, one step a page]\n" + "\n".join(texts)) if texts else ""

    def cheapest_here(self, observations: list[Any]) -> str:
        """Under a cheapest pick: the buyable rows of a search result (and the pages Raikou read on) in price order, since
        stores list them in no such order."""
        if self.brief.get("pick") != "cheapest" or not self.price_key:
            return ""
        found = {i for seen in observations for i, _ in self.records_in(seen)}
        rows = [(self.price_of(i), i) for i in found if self.buyable(i) and self.price_of(i) is not None]
        if len(rows) < 2:
            return ""
        return "\n[Raikou] buyable rows here, cheapest first: " + ", ".join(f"{i} ({p:g})" for p, i in sorted(rows)[:6])

    def wider_hold(self) -> str:
        """Once, before a cheapest checkout with many steps left: a store's keyword search reads titles, so fitting
        listings whose titles use other words are missed by searches for a requirement's value."""
        if self.wider_held or self.brief.get("pick") != "cheapest" or self.left() < 14 + 2 * len(self.items) or self.elapsed() > WRAP_AT:
            return ""
        self.wider_held = True
        # when the store lists a category by its label, the code lists each cart line's category before the last check
        # (shelf_rows), which finds those listings without the agent's extra searches
        cat_key = self.category_key()
        labels = [str((self.rows.get(i) or {}).get(cat_key) or "").strip() for i in self.cart_ids()] if cat_key else []
        if labels and all(label and self.shelf_args(label, None) for label in labels):
            return ""
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
            if self.left() <= keep or self.finished or self.facts_from.get(tool) is False:
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
        self.shelf_rows()
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

    def listing_tool(self) -> tuple[str, str, str | None, dict[str, Any]] | None:
        """The store call that lists one category's listings (under a price ceiling when it takes one): named once per
        episode by the light model from the tools' schemas and checked against them. (tool, category parameter, price
        parameter or None, fixed arguments), or None when no tool lists a category."""
        if self.listing is not None:
            return self.listing or None
        self.listing = False
        skip = {self.roles.get(r) for r in ("cart", "cart_view", "buy", "refuse", "ask", "open", "offer", "compare")} - {None}
        tools = [t for t in self.tools if t["function"]["name"] not in skip]
        if not tools or self.cost > SPEND_HARD:
            return None
        lines = [f"- {t['function']['name']}: {t['function'].get('description', '')} | parameters: {_dump(t['function'].get('parameters') or {})}" for t in tools]
        got = self.ask_json(LISTING + "\n\nTOOLS:\n" + "\n".join(lines), [LIGHT, HEAVY], 600) or {}
        name = got.get("tool")
        schema = self.schemas.get(name) if isinstance(name, str) else None
        props = (schema or {}).get("properties") or {}
        cat, price = got.get("category_arg"), got.get("price_arg") or None
        cat_spec = props.get(cat) if isinstance(cat, str) else None
        if not schema or name in skip or not isinstance(cat_spec, dict) or "string" not in _kinds(cat_spec):
            return None
        price_spec = props.get(price) if isinstance(price, str) else None
        if price is not None and (not isinstance(price_spec, dict) or not set(_kinds(price_spec)) & {"number", "integer"}):
            price = None
        fixed = {k: v for k, v in (got.get("fixed") or {}).items() if k in props and k not in (cat, price)} if isinstance(got.get("fixed"), dict) else {}
        faults: list[str] = []
        shaped = _shape(fixed, {**schema, "required": []}, "args", faults)
        fixed = shaped if not faults and isinstance(shaped, dict) else {}
        self.listing = (name, cat, price, fixed)
        self.trace.append({"kind": "listing_tool", "tool": name, "category": cat, "price": price, "fixed": fixed})
        return self.listing

    def shelf_args(self, label: str, cap: float | None) -> tuple[str, dict[str, Any]] | None:
        """Arguments that list one category (under the cap when the listing call takes a price ceiling)."""
        found = self.listing_tool()
        if not found:
            return None
        name, cat, price, fixed = found
        schema = self.schemas[name]
        props = schema.get("properties") or {}
        value: Any = label
        options = [o for o in (props[cat].get("enum") or []) if isinstance(o, str)]
        if options:
            value = next((o for o in options if o.strip().lower() == label.strip().lower()), None)
            if value is None:
                return None
        args: dict[str, Any] = {**fixed, cat: value}
        if price and cap is not None:
            whole = "integer" in _kinds(props[price]) and "number" not in _kinds(props[price])
            args[price] = (int(cap) if float(cap).is_integer() else int(cap) + 1) if whole else cap
        if any(key not in args for key in schema.get("required") or []):
            return None
        faults: list[str] = []
        args = _shape(args, schema, "args", faults)
        return None if faults else (name, args)

    def read_listing(self, name: str, args: dict[str, Any], pages: int, below: float | None = None) -> bool | None:
        """Read a listing call and up to `pages` pages of it (results the same call already returned are not sent again).
        Rows sorted by price stop the paging once a page reaches past `below`: later pages hold only dearer rows.
        Returns whether its last page was reached, or None when the episode ended or a call failed."""
        for _ in range(pages):
            mark = name + _dump(args)
            cached = self.read_cache.get(mark)
            if cached and cached[0] == self.version:
                return False
            reply = self.act(name, args)
            if reply is None or self.finished:
                return None
            seen = reply.get("observation")
            self.observe(name, seen)
            self.read_cache[mark] = (self.version, self.step, self.show(seen))
            plan = self.paging(name, seen)
            if not plan or plan[3] is None:
                return bool(plan)
            if self.left() - self.tail() < 12:
                return False
            if self.price_sorted(seen):
                top = max(self.price_of(i) or 0.0 for i, _ in self.records_in(seen))
                if below is None or top > below:
                    return False
            faults: list[str] = []
            args = _shape({**args, plan[2]: plan[3]}, self.schemas.get(name), "args", faults)
            if faults:
                return False
        return False

    def shelf_rows(self) -> None:
        """Under a cheapest pick: each cart line's category is listed by the store itself (under the line's price when it
        filters by price, else a few pages), since keyword searches miss fitting listings whose titles use other words.
        The rows join the ones the cheaper-row reads and the last check weigh. Once per category and price."""
        cat_key = self.category_key()
        if not cat_key or self.elapsed() > LATE_AT - 30 or self.cost > SPEND_HARD:
            return
        for line in self.cart_ids():
            price = self.price_of(line)
            label = str((self.rows.get(line) or {}).get(cat_key) or "").strip()
            if price is None or not label or self.left() - self.tail() < 10 or self.finished:
                continue
            key = f"{label.lower()}<{price:g}"
            # a category already listed in full up to a higher price holds every row this listing would show
            if key in self.shelved or self.shelf_full.get(label.lower(), -1.0) >= price:
                continue
            self.shelved.add(key)
            call = self.shelf_args(label, price)
            if not call:
                # no category listing: a keyword search for the category's name, shaped like the agent's own searches
                words = self.search_args(label)
                if words is None:
                    return
                call = (self.roles["search"], words)
            name, args = call
            capped = bool(self.listing and self.listing[0] == name and self.listing[2] and self.listing[2] in args)
            whole = self.read_listing(name, args, 2 if capped else 3, price)
            if whole is None:
                return
            if whole:
                self.shelf_full[label.lower()] = max(self.shelf_full.get(label.lower(), -1.0), price if capped else float("inf"))
            self.trace.append({"kind": "shelf", "label": label, "below": price, "capped": capped})

    def units_for(self, ident: str) -> int:
        """Units a cart line must hold: the quantity of the item its category names (the only item when there is one), else
        what the line holds (a line holding another count is fixed by the count check, not by a cheaper listing)."""
        cat_key = self.category_key()
        label = str((self.rows.get(ident) or {}).get(cat_key) or "") if cat_key else ""
        number = 1 if len(self.items) == 1 else self.item_number(label)
        return self.items[number - 1].qty if number else max(1, self.cart_lines.get(ident, 1))

    def over_budget(self, ids: list[str]) -> bool:
        """Whether these lines, each at its item's quantity, break the budget in force (each line on its own under a
        budget for each item)."""
        if self.budget is None:
            return False
        costs = [(self.price_of(i) or 0.0) * self.units_for(i) for i in ids]
        if self.brief.get("budget_scope") == "each":
            return any(c > self.budget + 1e-9 for c in costs)
        return sum(costs) > self.budget + 1e-9

    def budget_rows(self) -> str:
        """After the shopper lowered the budget: a cart line the new budget leaves no room for (the dearest two first) gets
        its category listed under the room left for it, by the store's category listing when it has one, else by a keyword
        search for the category's name shaped like the agent's own searches; after a checkout try, an item with no line
        gets its category listed under what the lines leave. The cheapest buyable rows under the amount are named and the
        first two read, so the cart can be fixed before anyone ends the episode without a purchase (race 11: a third of
        the episodes cut the budget at the first checkout)."""
        self.budget_moved = False
        cat_key = self.category_key()
        if (self.budget is None or not self.items or not cat_key or self.finished or self.left() - self.tail() < 8
                or self.elapsed() > LATE_AT - 30 or self.cost > SPEND_HARD):
            return ""

        def label_of(ident: str) -> str:
            return str((self.rows.get(ident) or {}).get(cat_key) or "").strip()

        each = self.brief.get("budget_scope") == "each"
        budget = float(self.budget)
        cart = set(self.cart_ids())
        lines = [i for i in self.cart_lines if i in cart and not self.sold_out(i)]
        cost = {i: (self.price_of(i) or 0.0) * self.units_for(i) for i in lines}
        total = sum(cost.values())
        numbers = {i: (1 if len(self.items) == 1 else self.item_number(label_of(i))) for i in lines}
        # (category, most a unit may cost, item number or None, the line it would replace or "")
        wants: list[tuple[str, float, int | None, str]] = []
        for ident in sorted(lines, key=lambda i: -cost[i]):
            room = budget if each else budget - (total - cost[ident])
            if room > 0 and cost[ident] > room + 1e-9:
                wants.append((label_of(ident), room / self.units_for(ident), numbers[ident], ident))
        if not wants and lines and not each and total > budget + 1e-9:
            # no single swap can fit: the dearest line's category under an even share of the budget
            dearest = max(lines, key=lambda i: cost[i])
            wants.append((label_of(dearest), budget / len(lines) / self.units_for(dearest), numbers[dearest], dearest))
        if self.buys and len(lines) < len(self.items) and None not in numbers.values():
            room = budget if each else budget - total
            for number, item in enumerate(self.items, 1):
                if number in numbers.values() or room <= 0:
                    continue
                seen = [label_of(i) for i in self.full if label_of(i) and (len(self.items) == 1 or self.item_number(label_of(i)) == number)]
                if seen:
                    wants.append((max(set(seen), key=seen.count), room / max(1, item.qty), number, ""))
        texts: list[str] = []
        for label, cap, number, line in wants[:2]:
            mark = f"{label.lower()}<{cap:g}"
            if not label or mark in self.budget_listed or self.left() - self.tail() < 8:
                continue
            self.budget_listed.add(mark)
            looked = self.shelf_full.get(label.lower(), -1.0) >= cap
            if not looked:
                call = self.shelf_args(label, cap)
                if call:
                    whole = self.read_listing(call[0], call[1], 2, cap)
                    looked = whole is not None
                    capped = bool(self.listing and self.listing[0] == call[0] and self.listing[2] and self.listing[2] in call[1])
                    if whole:
                        self.shelf_full[label.lower()] = max(self.shelf_full.get(label.lower(), -1.0), cap if capped else float("inf"))
                else:
                    args = self.search_args(label)
                    if args is not None:
                        looked = self.read_listing(self.roles["search"], args, 2, cap) is not None
            if self.finished:
                return ""
            pool = [i for i in self.rows if label_of(i).lower() == label.lower() and i not in lines and self.buyable(i)
                    and self.price_of(i) is not None and (self.price_of(i) or 0.0) <= cap + 1e-9]
            pool.sort(key=lambda i: self.price_of(i) or 0.0)
            head = (f"item {number} ({self.items[number - 1].name})" if number else label) + (f" in place of {line}" if line else ", no cart line yet")
            if not pool:
                texts.append(f"{head}: no {label} listing at or under {_num(cap)} each has been seen" if looked
                             else f"{head}: Raikou could not list {label} under {_num(cap)}; search for one at or under that price before ending without a purchase")
                continue
            read = self.fill_facts(pool[:2], self.tail() + 6)
            texts.append(f"{head}, at most {_num(cap)} each: " + ", ".join(f"{i} ({_num(self.price_of(i) or 0.0)})" for i in pool[:6])
                         + ("\n" + "\n".join(read) if read else ""))
        if not texts:
            return ""
        self.trace.append({"kind": "budget_rows", "budget": self.budget, "lines": len(texts)})
        return (f"[Raikou listed what the new budget of {_num(self.budget)} leaves room for (its own steps)]\n" + "\n".join(texts)
                + "\nCheck these against every requirement and put a fitting one in the cart in place of the line named (or for the"
                " item with no line); end without a purchase only if none fits.")

    def observe(self, name: str, observation: Any) -> str:
        """Absorb a result of Raikou's own call and read any note in it; returns what the note changed."""
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
        if self.asked or not self.roles.get("ask") or self.msgs_free() <= 0 or self.elapsed() > WRAP_AT:
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
        for index, (label, name, _carried, _values, shown) in enumerate(table, 1):
            if label != last:
                number = self.item_number(label)
                listing.append(f"category {label!r}" + (f" (item {number})" if number else "") + ":")
                last = label
            listing.append(f"  {index}. {name} ({shown})")
        # each message names up to PACK properties of one item: an item's ranked list must fill the messages it may get
        per_item = max(PACK, min(12, PACK * count, 36 // max(1, len(self.items))))
        prompt = PLAN.replace("COUNT", str(per_item)) + "\n\nREQUEST:\n" + self.request + "\n\nITEMS:\n" + "\n".join(settled)
        first = [str(x) for x in _items(self.brief.get("ask_first")) if str(x).strip()]
        if first:
            prompt += "\n\nASK FIRST (the request asks to be asked about these):\n" + "\n".join(first)
        prompt += "\n\nATTRIBUTE KEYS SEEN:\n" + ("\n".join(listing) or "(no full listing records yet)")
        plan = self.ask_json(prompt, [HEAVY, LIGHT], 2400) or {}
        queues: dict[int, list[str]] = {n: [] for n in range(1, len(self.items) + 1)}
        named: list[int] = []
        hinted: set[int] = set()
        for pick in _items(plan.get("ask")):
            if not isinstance(pick, dict):
                continue
            number = min(max(1, _count(pick.get("item") or 1) or 1), len(self.items))
            # one property per label: a label naming two is cut to the first (the message lists the labels)
            label = re.split(r"\s+(?:and|or)\s+|[/,;(]", str(pick.get("label") or ""))[0].strip().rstrip("?.")
            if not label or len(label.split()) > 5 or any(_words(label) == _words(other) for other in queues[number]):
                continue
            queues[number].append(label)
            if number not in named:
                named.append(number)
            if str(pick.get("hint") or "").strip():
                hinted.add(number)
        if table:
            # the listings' most carried differing keys fill each item's list up to what its messages can name
            added = []
            seen_values: dict[tuple[int, str], set[str]] = {}
            for category, key, _carried, ranked, _shown in table:
                seen_values.setdefault((self.item_number(category) or 1, _words(key)), set()).update(ranked)
            for number, label in self.table_picks(table, per_item * len(self.items)):
                if (len(queues[number]) < per_item and self.plain_label(label)
                        and not any(self.same_question(label, other, seen_values.get((number, _words(label))), seen_values.get((number, _words(other))))
                                    for other in queues[number])):
                    queues[number].append(label)
                    added.append((number, label))
            if added:
                self.trace.append({"kind": "plan_fill", "ask": added})
        order = named + [n for n in queues if n not in named]
        dependents = {d for d, a in self.pair_links()} - {a for d, a in self.pair_links()}
        if dependents and len(dependents) < len(self.items):
            # an item that must work with another item's pick is chosen mostly by that pick (its codes leave few
            # partners), so an answer about it changes the decision less: the anchor items go first, an item the
            # request's story points at keeps its place
            order = [n for n in order if n not in dependents or n in hinted] + [n for n in order if n in dependents and n not in hinted]
        self.trace.append({"kind": "plan", "ask": _clip(plan.get("ask"), 120), "order": order})
        sent: list[tuple[int, str, str]] = []
        told: list[str] = []
        found: set[int] = set()
        spent: dict[int, int] = {}
        puzzled: dict[int, int] = {}
        # how likely each queued property of an item still is the one its unwritten preference sits on: the planner's
        # rank at first, then lowered each time a message named it and drew nothing, by how surely its place in that
        # message was weighed (race 15: the 1st place always, 2nd to 4th about nine times in ten, 5th and 6th under half,
        # later ones hardly ever); the item's next message names the likeliest ones first
        odds = {n: {label: self.rank_odds(i) for i, label in enumerate(queue)} for n, queue in queues.items()}
        values = {n: self.label_values(n, queue, table) for n, queue in queues.items()}
        number = None
        rounds = 0
        asked_sets: set[tuple[int, tuple[str, ...]]] = set()
        retried: dict[int, int] = {}
        heard_props: dict[int, set[frozenset[str]]] = {}
        force: int | None = None
        self.msg_error = False
        while rounds < count and self.msgs_free() > 0 and not self.finished and self.elapsed() < LATE_AT - 20:
            if len(sent) >= 3 and force is None and not any(sum(odds[n].get(label, 0.0) for label in queues[n]) >= 0.05
                                                            for n in queues if queues[n] and not puzzled.get(n)):
                # past three messages only while some item that has not puzzled the shopper may still hold a preference
                # among its queued properties (each message costs a step and a reading)
                break
            rounds += 1
            number = force if force is not None and queues.get(force) else self.next_item(order, queues, found, spent, puzzled)
            force = None
            if number is None:
                break
            queue = queues[number]
            labels = self.fit_labels(number, queue[:PACK])
            # the item's likeliest properties were all in a message already (one that did not get through, or a list
            # too short to reorder): the same list starting further down, else the item is done
            shift = 1
            while (number, tuple(labels)) in asked_sets and shift < len(queue):
                labels = self.fit_labels(number, (queue[shift:] + queue[:shift])[:PACK])
                shift += 1
            if (number, tuple(labels)) in asked_sets:
                # no message went out: the round is not used up
                queues[number] = []
                rounds -= 1
                continue
            asked_sets.add((number, tuple(labels)))
            batch = self.send([(number, ", ".join(labels), self.question(number, labels))])
            if not batch:
                break
            sent += batch
            if self.msg_error:
                # the message tool refused the message: no answer to read and no point in sending more
                break
            before = [len(item.told) for item in self.items]
            heard = self.hear(batch, (LIGHT, HEAVY))
            # the message was about one item: what its answer states belongs to that item
            for entry in _items(heard.get("needs")) + _items(heard.get("free")):
                if isinstance(entry, dict):
                    entry["item"] = number
            needs = [str(e.get("need")).split(":", 1) for e in _items(heard.get("needs")) if isinstance(e, dict) and str(e.get("need") or "").strip()]
            # a preference on a property this interview already heard for the item only repeats it (race 15: asking a
            # found item again mostly drew the same one): it settles nothing new and reads like a "does not mind"
            known = heard_props.setdefault(number, set())
            stated = [s for s in needs if _stem(s[0]) not in known]
            repeated = bool(needs) and not stated
            known.update(_stem(s[0]) for s in needs if _stem(s[0]))
            kinds = [str(k).strip().lower() for k in _items(heard.get("kinds"))]
            kind = kinds[0] if kinds else ""
            missed = not needs and (kind.startswith("unclear") or 1 in {_count(x) for x in _items(heard.get("unclear"))})
            freed = not stated and (repeated or kind.startswith("free") or any(isinstance(e, dict) for e in _items(heard.get("free"))))
            read = isinstance(heard, dict) and any(key in heard for key in ("needs", "free", "kinds", "unclear"))
            if len(labels) > 1:
                # a "does not mind" to a list does not show which of its properties the shopper weighed
                heard.pop("free", None)
            self.take_heard(heard, told)
            gained = {n for n, item in enumerate(self.items, 1) if len(item.told) > before[n - 1]}
            found |= gained
            if not needs and not read and retried.get(number, 0) < 1:
                # the reading failed outright (no usable reply from either model): the same message goes once more, since
                # its answer may have held a preference
                retried[number] = 1
                asked_sets.discard((number, tuple(labels)))
                force = number
                self.trace.append({"kind": "heard", "item": number, "labels": labels, "reading": "failed"})
                continue
            spent[number] = spent.get(number, 0) + 1
            chances = odds[number]
            if stated:
                # one answer gives one preference: the shopper went down the list to the property it holds one on, so the
                # properties named before that one were weighed and passed over, and the ones after it were not reached
                pairs = [(s[0].strip(), s[1].strip() if len(s) > 1 else "") for s in stated]
                places = [labels.index(best) for best in (self.answered_label(prop, labels, value, values[number]) for prop, value in pairs) if best]
                reached = max(places) if places else 0
                for place in places:
                    chances[labels[place]] = 0.0
                for place, label in enumerate(labels[:reached]):
                    if place not in places:
                        chances[label] = chances.get(label, 0.0) * (1.0 - self.heard_at(place))
                if not places:
                    # a preference on a property the message did not name: when it is one still queued it is answered;
                    # the message's first property was weighed either way
                    rest = [label for label in queue if label not in labels]
                    for prop, value in pairs:
                        other = self.answered_label(prop, rest, value, values[number])
                        if other:
                            chances[other] = 0.0
                    chances[labels[0]] = 0.0
                found.add(number)
            elif missed:
                # the answer did not take the message in: nothing was weighed, and the item waits behind the items that
                # have not puzzled the shopper (race 15: the same properties again drew a puzzled reply 100 times in 103,
                # another item's about three in four)
                puzzled[number] = puzzled.get(number, 0) + 1
            if not missed:
                # an item that answers again (a puzzled one included) takes its turns like any other
                puzzled.pop(number, None)
            if freed and not missed:
                # the shopper does not mind: whatever it weighed holds no preference
                for place, label in enumerate(labels):
                    chances[label] = chances.get(label, 0.0) * (1.0 - self.heard_at(place))
            # anything else (a store or shopper note in place of an answer, a preference the reading could not write
            # down, a second failed reading) weighed nothing either: the other items take their turn, and the item's next
            # message names the same properties in another order
            queues[number] = sorted((label for label in queue if chances.get(label, 0.0) > 0.0), key=lambda label: -chances.get(label, 0.0))
            self.trace.append({"kind": "heard", "item": number, "labels": labels, "gained": sorted(gained), "unclear": missed,
                               "stated": [s[0] for s in stated], "next": queues[number][:4]})
        if told:
            self.version += 1
        if not sent:
            return ""
        lines = [f"Q: {q}\nA: {a}" for _, q, a in sent]
        summary = "Raikou's interviewer asked the shopper (each message used one step):\n" + "\n".join(lines)
        summary += "\nNew firm requirements from these answers: " + ("; ".join(told) if told else "none")
        summary += ". Apply them to every pick (search again with the new values where needed)."
        pointed = [need for _, need in self.refs if any(need in t for t in told)]
        if pointed:
            summary += (" These take their value from another listing: " + " | ".join(pointed) + ". Find that listing, open its page in full,"
                        " and open every listing its page refers to; the last page gives the value, and the order counts only after all of them are opened.")
        if self.queued:
            summary += "\n[notes seen meanwhile] " + " | ".join(self.queued)
            self.queued = []
        return summary

    def next_item(self, order: list[int], queues: dict[int, list[str]], found: set[int], spent: dict[int, int], puzzled: dict[int, int]) -> int | None:
        """The item the next message is about, among those with properties left: items that have not puzzled the
        shopper first (race 15: a puzzled reply came back on the same item almost every time, while another item's
        message drew a clear one about a quarter of the time), then items whose preference has not come out (one answer
        gives one preference, and asking that item again mostly drew the same one), then the item asked least, then the
        planner's order."""
        live = [n for n in order if queues.get(n)]
        if not live:
            return None
        return min(live, key=lambda n: (puzzled.get(n, 0) > 0, n in found, spent.get(n, 0), order.index(n)))

    def rank_odds(self, rank: int) -> float:
        """How often the planner's property at this rank (0 = first) held the item's unwritten preference (race 15)."""
        return RANK_ODDS[rank] if rank < len(RANK_ODDS) else RANK_ODDS[-1] * 0.85

    def heard_at(self, place: int) -> float:
        """How surely the shopper weighs the property at this place (0 = first) of a list message (race 15)."""
        return HEARD[place] if place < len(HEARD) else HEARD[-1] / 2

    def fit_labels(self, number: int, labels: list[str]) -> list[str]:
        """A message tool with a length limit: name fewer properties rather than cut the question short."""
        cap = self.message_cap()
        labels = list(labels)
        while cap and len(labels) > 1 and len(self.question(number, labels)) > cap:
            labels = labels[:-1]
        return labels

    def label_values(self, number: int, labels: list[str], table: list[tuple[str, str, int, list[str], str]]) -> dict[str, set[str]]:
        """The words of the values the item's opened listings show under each of these properties' keys, so a
        preference the answer names in other words ("memory" for a "RAM" question) still finds its property by value."""
        found: dict[str, set[str]] = {}
        for label in labels:
            want = _stem(label)
            words: set[str] = set()
            for category, key, _carried, ranked, _shown in table:
                if (self.item_number(category) or 1) == number and _names(want, _stem(key)):
                    for value in ranked:
                        words |= set(re.findall(r"[a-z]+|\d+(?:\.\d+)?", value.lower()))
            found[label] = words
        return found

    def answered_label(self, prop: str, asked: list[str], value: str = "", seen: dict[str, set[str]] | None = None) -> str | None:
        """The asked property an answer's preference is about: most shared words, a word counting as shared when both
        spellings start with the same four letters ("colour" and "color", "sleeves" and "sleeve"); ties go to the
        closer spelling. When no asked property shares a word with it, the one whose values on opened listings share
        most words with the stated value (two at least, numbers counting first); None when neither finds one."""
        want = [w for w in re.findall(r"[a-z0-9]+", prop.lower())]
        best, score = None, (0, 0.0)
        for label in asked:
            have = re.findall(r"[a-z0-9]+", label.lower())
            shared = sum(1 for w in want if any(w == h or (len(w) >= 4 and len(h) >= 4 and w[:4] == h[:4]) for h in have))
            if not shared:
                continue
            close = difflib.SequenceMatcher(None, " ".join(want), " ".join(have)).ratio()
            if (shared, close) > score:
                best, score = label, (shared, close)
        if best or not value or not seen:
            return best
        said = {w for w in re.findall(r"[a-z]+|\d+(?:\.\d+)?", value.lower()) if len(w) >= 2 or w.isdigit()}
        top, mark = None, (0, 0)
        for label in asked:
            common = said & (seen.get(label) or set())
            numbers = sum(1 for w in common if w[0].isdigit())
            if len(common) >= 2 and (numbers, len(common)) > mark:
                top, mark = label, (numbers, len(common))
        return top

    def message_cap(self) -> int | None:
        """The message tool's length limit on its text, when its schema gives one."""
        schema = self.schemas.get(self.roles.get("ask") or "") or {}
        props = schema.get("properties") or {}
        texts = [k for k, v in props.items() if isinstance(v, dict) and "string" in _kinds(v) and "enum" not in v]
        target = next((k for k in schema.get("required") or [] if k in texts), None) or (texts[0] if texts else None)
        cap = (props.get(target) or {}).get("maxLength") if target else None
        return cap if isinstance(cap, int) and not isinstance(cap, bool) and cap > 0 else None

    def plain_label(self, label: str) -> bool:
        """A key worth naming to a shopper: no word of one or two letters at either end (store prefixes and suffixes
        such as a two-letter code), no digits."""
        words = label.split()
        return bool(words) and not re.search(r"\d", label) and len(words[0]) > 2 and len(words[-1]) > 2

    def same_question(self, one: str, other: str, one_values: set[str] | None = None, other_values: set[str] | None = None) -> bool:
        """Two property labels that ask the same thing: the same words (singular or plural alike), or one's words all
        inside the other's when the listings show both keys sharing values ("type" and "pack type" are different
        properties: race 15 hid a preference on a pack type that a bare "type" kept out of every message, while a
        "colour family" that repeats the values of "colour" is the same question). With no values seen for one of them
        both stay: a near repeat costs a place in a list, leaving a property out can lose the preference."""
        a, b = _stem(one), _stem(other)
        if not a or not b:
            return False
        if a == b:
            return True
        if not (a <= b or b <= a):
            return False
        return bool(one_values) and bool(other_values) and bool(one_values & other_values)

    def question(self, number: int, labels: list[str]) -> str:
        item = self.items[number - 1]
        names = [label.strip() for label in labels if label.strip()]
        listed = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " or " + names[-1]
        shelf = (item.shelf or "").strip()
        about = item.name + (f" ({shelf})" if shelf and _words(shelf) != _words(item.name) and len(self.items) > 1 else "")
        return f"What {listed} would you like for the {about}?"

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
            words = self.spoken(answer)
            sent.append((number, text, words))
            self.done_log.append(f"step {self.step}: message to the shopper {text!r} -> reply {words[:300]!r} (cart then: {self.cart_mark()})")
            self.done_steps.append(self.step)
            self.trace.append({"kind": "ask", "q": text, "a": words[:300]})
            if isinstance(answer, dict) and answer.get("error"):
                self.msg_error = True
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
            if self.budget is None or heard["budget"] < self.budget:
                self.budget_moved = True
            self.budget = heard["budget"]

    def hear(self, sent: list[tuple[int, str, str]], models: tuple[str, ...] = (HEAVY, LIGHT)) -> dict[str, Any]:
        prompt = HEAR + "\n\nITEMS:\n" + "\n".join(f"item {n}: {it.name}" for n, it in enumerate(self.items, 1))
        prompt += "\n\nQUESTIONS AND ANSWERS:\n" + "\n".join(f"{i}. (about item {n}) Q: {q}\nA: {a}" for i, (n, q, a) in enumerate(sent, 1))
        return self.ask_json(prompt, list(models), 1500) or {}

    def gate(self, stage: str) -> tuple[bool, str]:
        """Right before a checkout or an ending: the steps the request asks for at that point. Its model calls may go a
        little past the spend stop (a skipped step fails the task however good the cart is)."""
        self.stop_room = GATE_ROOM
        try:
            return self.gate_steps(stage)
        finally:
            self.stop_room = 0.0

    def gate_steps(self, stage: str) -> tuple[bool, str]:
        """Right before a checkout ("order") or an ending without a purchase ("end"): the steps the request asks for at
        that point, done by the model's choice of store calls (a message to the shopper, a page, any store tool) until
        it finds none missing for the cart as it is now. Returns whether to go on, and what to tell the agent."""
        steps = [p for p in self.process if p["when"] == stage]
        if not steps or self.finished or not self.gate_due(stage) or self.elapsed() > FINISH_AT - 10:
            return True, ""
        ending = {self.roles.get("buy"), self.roles.get("refuse")} - {None}
        tools = [f"- {t['function']['name']}: {t['function'].get('description', '')} | parameters: {_dump(t['function'].get('parameters') or {})}"
                 for t in self.tools if t["function"]["name"] not in ending]
        said: list[str] = []
        nudged = False
        for _ in range(GATE_ROUNDS):
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
                return False, ("Before going on, Raikou did the steps the request asks for:\n" + "\n".join(said) + "\n" if said else "") + (
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
                self.done_log.append(f"(Raikou: no line made since the cart last changed at step {self.cart_at} shows " + "; ".join(str(e.get("step") or "") for e in unproven)[:200]
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
        return True, ("Before going on, Raikou did the steps the request asks for:\n" + "\n".join(said)) if said else ""

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
        return " ".join(found)[:800] if found else _dump(observation)[:400]

    # ---------- notes inside tool results ----------
    def loose_lines(self, observation: Any) -> list[str]:
        lines: list[str] = []
        if isinstance(observation, dict) and not self.ident(observation):
            for key, value in observation.items():
                if isinstance(value, str) and len(value.split()) >= 5:
                    lines.append(value)
                elif isinstance(value, list) and value and all(isinstance(v, str) for v in value):
                    lines.extend(v for v in value if len(v.split()) >= 5)
                elif isinstance(value, dict) and not self.ident(value, False, 1):
                    for inner in value.values():
                        if isinstance(inner, str) and len(inner.split()) >= 6:
                            lines.append(inner)
        return [line for line in lines if line not in self.heard]

    def take_note(self, lines: list[str]) -> str:
        self.heard.update(lines)
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
            if self.budget is None or budget < self.budget:
                # a lower limit can leave a cart line out of budget: Raikou lists what still fits
                self.budget_moved = True
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
        if got.get("price_moved") is True:
            changed.append("a price changed")
        summary = str(got.get("summary") or "").strip()
        if changed or summary:
            self.version += 1
            self.recent = []
            self.notes.append(summary or "; ".join(changed))
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
        return "lines by Raikou's record of the cart calls (listing: units): " + _dump(self.cart_lines)

    def cart_ids(self) -> list[str]:
        if self.cart_truth_at >= self.cart_at:
            # the store listed every line after the last change: read the lines, not ids named anywhere in its words
            return list(self.cart_lines)
        if self.cart_view is not None and self.cart_seen_at >= self.cart_at:
            text = _dump(self.cart_view)
            return [i for i in self.rows if i in text]
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

    def late_models(self) -> list[str]:
        """Models for a late check: the heavy one first while this task's spend is low, the light one first after
        SPEND_LIGHT (one run key pays for every task)."""
        return [HEAVY, LIGHT] if self.cost < SPEND_LIGHT else [LIGHT, HEAVY]

    def check_cart(self, ids: list[str] | None = None, limit: float = SPEND_SOFT) -> str:
        lead = ids is not None
        ids = ids if lead else self.cart_ids()
        key = self.check_key(ids)
        if key in self.checked or self.cost > limit or not ids or self.elapsed() > FINISH_AT - 12:
            return ""
        self.checked.add(key)
        prompt = CHECK + "\n\nREQUEST:\n" + self.request + "\n\nREQUIREMENTS IN FORCE:\n" + self.ledger()
        if lead:
            quantity = "; ".join(f"item {n}: quantity {it.qty}" for n, it in enumerate(self.items, 1))
            prompt += "\n\nCART (planned, judge these listings as if they were carted at the requested quantities: " + quantity + "):\n" + ", ".join(ids)
        else:
            prompt += "\n\nCART:\n" + self.cart_text()
        prompt += "\n\nRECORDS OF THE CART'S LISTINGS:\n" + "\n".join(f"{i}: " + ("SOLD OUT (the store says it can no longer be bought). " if self.sold_out(i) else "") + self.record_of(i) for i in ids)
        pairs = self.pair_notes(ids)
        if pairs:
            prompt += "\n\nCOMPATIBILITY BY THE CODES (Raikou's reading of every listing's codes, lists that go on at another part and notes on other listings):\n" + pairs
        if self.brief.get("pick") in ("cheapest", "most", "least"):
            prompt += "\n\nOTHER LISTINGS SEEN (newest first):\n" + self.others(ids, 15)
        got = self.ask_json(prompt, self.late_models(), 1500)
        self.trace.append({"kind": "check", "cart": ids, "got": got})
        self.verdicts[key] = bool(got) and got.get("ok") is not False
        if not got or got.get("ok") is not False:
            return ""
        self.leads = [i for i in self.leads if i not in ids]
        problems = [str(p) for p in _items(got.get("problems"))][:5]
        better = [str(b) for b in _items(got.get("better")) if str(b) in self.rows and not self.sold_out(str(b))][:4]
        if lead and len(ids) == 1 and not better:
            # a review pick judged alone and turned down (not only outranked) is not bought by the code's own fills later
            self.rejected.add(ids[0])
        text = "A careful review says this cart is not right yet: " + "; ".join(problems or ["see the requirements"])
        if better:
            text += ". Listings that look better: " + ", ".join(better)
        return text + ". Fix the cart (or show the review is wrong by re-reading the records), then check out."

    def trail(self) -> tuple[str, bool]:
        """Before a checkout: each requirement whose value is on another listing has every page of its chain opened.
        Raikou opens the links it can name itself; a link nobody has found yet holds the checkout back (twice at most).
        Returns what it saw and whether to hold the checkout."""
        live = {need for _, need in self.needs_now()}
        pending = [need for _, need in self.refs if need in live and need not in self.ref_values and need not in self.ref_items]
        # a chain's links are pages: the page view (the offer view when it is apart from the attributes view)
        tool = self.roles.get("offer") or self.roles.get("open")
        target = self.single_target(tool)
        seen: list[str] = []
        lost: list[str] = []
        while pending and self.trails < 4 and self.cost < SPEND_HARD and self.elapsed() < LATE_AT:
            self.trails += 1
            got = self.trace_chains(pending)
            reach, missing = self.read_chains(got, pending)
            lost = [f"{need} -> find {about or 'the listing it points at'}" for need, about, _ in missing]
            pending = [p for p in pending if p not in self.ref_values and p not in self.ref_items]
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
            text += "[Raikou opened the pages a requirement points at]\n" + "\n".join(seen) + "\n"
        if self.ref_values:
            text += "[values read from the pages requirements point at] " + "; ".join(f"{k} -> {v}" for k, v in self.ref_values.items()) + "\n"
        hold = bool(lost) and self.trail_blocks < 2 and self.left() > 5
        if hold:
            self.trail_blocks += 1
            text += ("The order counts only after every page a requirement points at has been opened in full (a search row is not enough), following each page that refers to a further listing. Still to find and open: "
                     + " | ".join(lost[:3]) + ". Search for it by its shop, brand or words, open it, apply the value its page gives, then check out.")
        return text, hold

    def trace_chains(self, pending: list[str]) -> dict[str, Any]:
        """One tracing call over the listings seen: each pointer requirement's chain of pages, link by link."""
        records = []
        for ident in reversed(self.seen_order):
            # an opened page is the page view's record (with the attributes when they were read too)
            record = (self.full.get(ident) or self.rows.get(ident)) if ident in self.opened else None
            if record:
                records.append("OPENED " + _dump(_clip(record, 240))[:1000])
            else:
                records.append("ROW " + _dump(_clip(self.rows.get(ident), 80))[:240])
            if len(records) >= 40:
                break
        prompt = TRAIL + "\n\nITEMS OF THIS REQUEST:\n" + "\n".join(f"item {n}: {it.name}" for n, it in enumerate(self.items, 1))
        prompt += "\n\nREQUIREMENTS WITH A VALUE ON ANOTHER LISTING:\n" + "\n".join(pending)
        prompt += "\n\nRECORDS (newest first):\n" + "\n".join(records)
        got = self.ask_json(prompt, [LIGHT, HEAVY], 1800) or {}
        self.trace.append({"kind": "trail", "pending": pending, "got": _clip(got, 150)})
        return got

    def read_chains(self, got: dict[str, Any], pending: list[str]) -> tuple[list[str], list[tuple[str, str, str]]]:
        """Finished chains give their value to the ledger. Returns the rows of unfinished chains to open, and the links no
        record is yet as (requirement, how the chain describes that listing, words to search for it)."""
        reach: list[str] = []
        missing: list[tuple[str, str, str]] = []
        for chain in _items(got.get("chains")):
            if not isinstance(chain, dict):
                continue
            need = next((p for p in pending if p == str(chain.get("need") or "").strip()), None) or (pending[0] if len(pending) == 1 else None)
            if not need:
                continue
            links = [l for l in _items(chain.get("links")) if isinstance(l, dict)]
            ids = [str(l.get("id") or "").strip() for l in links]
            other = _count(chain.get("item"))
            own = next((n for n, p in self.refs if p == need), None)
            named = any(str(l.get("search") or "").strip() for l in links)
            if other and 1 <= other <= len(self.items) and other != own and not named and not any(i in self.rows for i in ids):
                # the value comes from the listing chosen for another requested item: that pick settles it, no search can
                self.ref_items[need] = other
                continue
            if chain.get("done") is True and links and all(i in self.opened for i in ids) and str(chain.get("value") or "").strip():
                self.ref_values[need] = str(chain["value"]).strip()
                self.version += 1
                continue
            closed = [i for i in ids if i in self.rows and i not in self.opened]
            if closed:
                reach += [i for i in closed if i not in reach]
            else:
                link = next((l for l, i in zip(links, ids) if i not in self.rows), None)
                about = str((link or {}).get("about") or "")
                missing.append((need, about, str((link or {}).get("search") or "").strip()))
        return reach, missing

    # ---------- the code's own searches ----------
    def search_args(self, words: str) -> dict[str, Any] | None:
        """Arguments for a keyword search with the store's search tool, shaped like the agent's own keyword searches: the
        argument that carried its words takes these words, and the switches it set keep their values. Before the agent has
        searched, only a tool with a single free-text argument (and the first value of each required switch) is used."""
        tool = self.roles.get("search")
        schema = self.schemas.get(tool or "") or {}
        props = schema.get("properties") or {}
        if not tool or not words.strip():
            return None
        if self.search_template:
            key, fixed = self.search_template
            args: dict[str, Any] = {**fixed, key: words.strip()}
        else:
            free = [k for k, v in props.items() if isinstance(v, dict) and "string" in _kinds(v) and "enum" not in v]
            needed = [k for k in free if k in (schema.get("required") or [])]
            key = needed[0] if len(needed) == 1 else (free[0] if len(free) == 1 else None)
            if key is None:
                return None
            args = {key: words.strip()}
            for name in schema.get("required") or []:
                options = (props.get(name) or {}).get("enum") if isinstance(props.get(name), dict) else None
                if name not in args and isinstance(options, list) and options:
                    args[name] = options[0]
        if any(name not in args for name in schema.get("required") or []):
            return None
        faults: list[str] = []
        args = _shape(args, schema, "args", faults)
        return None if faults else args

    def learn_search(self, name: str, args: dict[str, Any]) -> None:
        """The agent's first search with exactly one free-text value shows how a keyword search is sent: which argument takes
        the words, and the switch values (enum arguments) it set. Filters and page values are not kept."""
        if self.search_template or name != self.roles.get("search"):
            return
        props = (self.schemas.get(name) or {}).get("properties") or {}
        shape = self.shapes.get(name) or {}
        texts = [k for k, v in args.items() if isinstance(v, str) and v.strip() and isinstance(props.get(k), dict)
                 and "enum" not in props[k] and k != shape.get("next_arg")]
        if len(texts) != 1:
            return
        # a call shaped like the store's category listing (its category parameter, or its fixed switch values) is no
        # keyword search
        listing = self.listing_tool()
        if listing and listing[0] == name and (texts[0] == listing[1] or (listing[3] and all(args.get(k) == v for k, v in listing[3].items()))):
            return
        if not (listing and listing[0] == name and listing[3]):
            value = str(args[texts[0]]).strip().lower()
            if any(isinstance(v, str) and v.strip().lower() == value for row in self.rows.values() for v in row.values()):
                return
        fixed = {k: v for k, v in args.items() if isinstance(props.get(k), dict) and isinstance(props[k].get("enum"), list) and v in props[k]["enum"]}
        self.search_template = (texts[0], fixed)

    def code_search(self, words: str) -> str | None:
        """Run a keyword search the code chose; the result is absorbed like the agent's own and cached for it."""
        tool = self.roles.get("search")
        args = self.search_args(words)
        if not tool or args is None or self.left() < 6 or self.finished:
            return None
        mark = tool + _dump(args)
        cached = self.read_cache.get(mark)
        if cached and cached[0] == self.version:
            return cached[2]
        reply = self.act(tool, args)
        if reply is None:
            return None
        seen = reply.get("observation")
        self.observe(tool, seen)
        taught = self.learn_words(tool, args, seen)
        if taught:
            self.queued.append(taught)
        shown = self.show(seen)
        self.read_cache[mark] = (self.version, self.step, shown)
        return shown

    def grounded(self, words: str, need: str, before: int | None = None) -> bool:
        """Search words a tracer gave for a link: the first one must be a name as the texts write it, which is a word with
        a digit, or a capitalised word the requirement or a record seen writes inside a sentence (not only at a sentence's
        start, where any word is capitalised), and not a word most rows seen before the chase carry."""
        first = next(iter(re.findall(r"[A-Za-z0-9][A-Za-z0-9'&.-]*", words or "")), "")
        first = re.sub(r"'s$", "", first)
        if len(first) < 2 or not re.search(r"[A-Z0-9]", first):
            return False
        # where the name is written: the requirement and every record seen (pages the chase itself opened included)
        texts = [_dump(r) for r in list(self.rows.values())[-120:]]
        exact = re.compile(r"(?<![A-Za-z0-9])" + re.escape(first) + r"(?![A-Za-z0-9])")
        if not re.search(r"\d", first):
            inside = re.compile(r"[^\s.!?:;\"\[{(]\s*" + re.escape(first) + r"(?![A-Za-z0-9])|[A-Za-z0-9,]\s+" + re.escape(first) + r"(?![A-Za-z0-9])")
            if not any(inside.search(t) for t in [need] + texts):
                return False
        elif not any(exact.search(t) for t in [need] + texts):
            return False
        # how common it is: over the rows seen before the chase (its own searches for that name would inflate it)
        earlier = [_dump(r) for r in (list(self.rows.values())[:before] if before is not None else list(self.rows.values()))[-80:]]
        share = sum(1 for t in earlier if exact.search(t)) / max(1, len(earlier))
        return share <= 0.5 or len(earlier) < 4

    def short_query(self, text: str) -> str:
        """Words to search for a listing a chain describes: its shop or brand name alone when the store matches any word,
        else that name with a couple of words of the kind."""
        words = [w for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9'&.-]*", text) if len(w) >= 2]
        return " ".join(words[:1] if self.word_match == "any" else words[:3])

    # ---------- items that must work with another item's pick ----------
    def pair_links(self) -> list[tuple[int, int]]:
        """(dependent item, the item whose pick it must work with) for every requirement that ties one requested item to
        the listing chosen for another: the request reading's links, and pointer requirements on another item's pick."""
        out: list[tuple[int, int]] = []
        count = len(self.items)
        for entry in _items(self.brief.get("links")):
            if isinstance(entry, dict):
                a, b = _count(entry.get("item")), _count(entry.get("on"))
                if a and b and a != b and 1 <= a <= count and 1 <= b <= count and (a, b) not in out:
                    out.append((a, b))
        for need, other in self.ref_items.items():
            own = next((n for n, p in self.refs if p == need), None)
            if own and other != own and (own, other) not in out:
                out.append((own, other))
        return out

    def item_labels(self) -> dict[int, str]:
        """Each requested item's category label among the rows seen: its shelf or name exactly, else the label sharing
        the most long words with it (one label an item, so two categories that share a word are not merged)."""
        cat_key = self.category_key()
        if not cat_key:
            return {}
        counts: dict[str, int] = {}
        for ident in self.seen_order:
            label = str((self.rows.get(ident) or {}).get(cat_key) or "").strip()
            if label:
                counts[label] = counts.get(label, 0) + 1
        out: dict[int, str] = {}
        for number, item in enumerate(self.items, 1):
            names = (_words(item.shelf or ""), _words(item.name))
            long = {w for w in _stem(item.shelf or "") | _stem(item.name) if len(w) >= 4}
            best = None
            for label, seen in counts.items():
                exact = _words(label) in names
                shared = len({w for w in _stem(label) if len(w) >= 4} & long)
                if exact or shared:
                    score = (exact, shared, seen)
                    if best is None or score > best[0]:
                        best = (score, label)
            if best:
                out[number] = best[1]
        return out

    def item_rows(self) -> dict[int, list[str]]:
        cat_key = self.category_key()
        labels = {label.lower(): n for n, label in self.item_labels().items()}
        out: dict[int, list[str]] = {}
        for ident in self.seen_order:
            label = str((self.rows.get(ident) or {}).get(cat_key) or "").strip().lower() if cat_key else ""
            if label in labels:
                out.setdefault(labels[label], []).append(ident)
        return out

    def affordable(self, ident: str, number: int) -> bool:
        price = self.price_of(ident)
        qty = self.items[number - 1].qty if 1 <= number <= len(self.items) else 1
        return self.budget is None or price is None or price * qty <= float(self.budget) + 1e-9

    def pair_due(self) -> bool:
        links = self.pair_links()
        # one light reading and a few store calls: it may run until the wrap-up, since the cart depends on it
        if (not links or self.paired >= 2 or self.finished or self.left() < 10 or self.elapsed() > WRAP_AT
                or self.cost > SPEND_SOFT):
            return False
        groups = self.item_rows()
        involved = sorted({n for pair in links for n in pair})
        if any(len(groups.get(n, [])) < 2 for n in involved):
            return False
        mark = tuple(len(groups.get(n, [])) for n in involved)
        return not self.paired_mark or sum(mark) >= sum(self.paired_mark) + 4

    def pair_up(self, models: list[str] | None = None) -> str:
        """Read the codes of the linked items' rows (light readings), follow 'rest of the list' parts, and set out
        which listing works with which, and the cheapest sets."""
        self.paired += 1
        links = self.pair_links()
        involved = sorted({n for pair in links for n in pair})
        groups = self.item_rows()
        self.paired_mark = tuple(len(groups.get(n, [])) for n in involved)
        share = max(6, PAIR_ROWS // max(1, len(involved)))
        picks: list[tuple[int, str]] = []
        for number in involved:
            pool = [i for i in groups.get(number, []) if self.buyable(i) and self.affordable(i, number)]
            pool.sort(key=lambda i: self.price_of(i) if self.price_of(i) is not None else float("inf"))
            picks += [(number, i) for i in pool[:share] if i not in self.codes]
        if picks:
            self.read_codes(picks, models)
        self.follow_rest(models)
        text = self.compat_text()
        self.trace.append({"kind": "pair", "links": links, "read": len(picks), "codes": len([c for c in self.codes.values() if c.get("own") or c.get("fits")])})
        return text

    def code_source(self, ident: str) -> str:
        record = {k: v for k, v in (self.rows.get(ident) or {}).items() if k in self.row_fields and k != self.price_key}
        return _dump(_clip(record, 700))[:1600]

    def code_cut(self, ident: str) -> bool:
        record = {k: v for k, v in (self.rows.get(ident) or {}).items() if k in self.row_fields and k != self.price_key}
        return any(isinstance(v, str) and len(v) > 700 for v in record.values()) or len(_dump(_clip(record, 700))) > 1600

    def read_codes(self, picks: list[tuple[int, str]], models: list[str] | None = None) -> None:
        """One light reading of the codes per batch of rows (a long batch could be cut off and lost whole)."""
        late = LATE_AT if models else WRAP_AT
        for start in range(0, len(picks), 12):
            if self.finished or self.cost > SPEND_HARD or self.elapsed() > late or self.calls > ALL_CALLS - 20:
                return
            self.read_code_batch(picks[start:start + 12], models or [LIGHT, HEAVY])

    def read_code_batch(self, picks: list[tuple[int, str]], models: list[str]) -> None:
        lines = [f"{i} | item {n}: {self.items[n - 1].name if 1 <= n <= len(self.items) else ''} | {self.code_source(i)}" for n, i in picks]
        got = self.ask_json(CODES + "\n\nLISTINGS:\n" + "\n".join(lines), models, 2500) or {}
        known = {i for _, i in picks}
        for i in known:
            self.codes.setdefault(i, {"own": "", "fits": [], "except": [], "rest": "", "notes": [], "partial": False})
        for entry in _items(got.get("codes")):
            if not isinstance(entry, dict) or str(entry.get("id") or "").strip() not in known:
                continue
            ident = str(entry["id"]).strip()
            text = self.code_source(ident).upper()

            def grounded(values: Any) -> list[str]:
                return [t for v in _items(values) for t in _code_tokens(v) if t in text]
            def entries(values: Any) -> tuple[list[str], bool]:
                """Listed codes or ranges (one entry a part) whose codes the row's text carries; whether all read cleanly."""
                kept, clean = [], True
                for value in _items(values):
                    if not isinstance(value, (str, int, float)) or isinstance(value, bool):
                        clean = False
                        continue
                    for part in _code_parts(value):
                        if any(t in text for t in _code_tokens(part)):
                            kept.append(part.upper())
                            clean = clean and _code_clean(part)
                        else:
                            clean = False
                return kept, clean
            own = grounded(entry.get("own"))
            rest = grounded(entry.get("rest_at"))
            fits, clean_fits = entries(entry.get("fits"))
            excl, clean_excl = entries(entry.get("except"))
            notes = []
            for note in _items(entry.get("notes")):
                if isinstance(note, dict):
                    part = grounded(note.get("part"))
                    more, _ok = entries(note.get("fits"))
                    if part and more:
                        notes.append((part[0], more))
            self.codes[ident] = {"own": own[0] if own else "", "fits": fits, "except": excl, "rest": rest[0] if rest else "", "notes": notes,
                                 "partial": not (clean_fits and clean_excl) or self.code_cut(ident)}

    def code_owner(self, code: str) -> str | None:
        return next((i for i, c in self.codes.items() if c.get("own") and c["own"] == code.upper()), None)

    def follow_rest(self, models: list[str] | None = None) -> None:
        """A list that goes on at another part: that part's listing is read too (a seen row whose texts carry the part
        code, else a search for the code, a few at most)."""
        for _ in range(2):
            missing = sorted({c["rest"] for c in self.codes.values() if c.get("rest") and not self.code_owner(c["rest"])})
            if not missing or self.finished or self.left() < 12:
                return
            picks: list[tuple[int, str]] = []
            for part in missing:
                holders = [i for i in self.seen_order if i not in self.codes and part in _code_tokens(self.code_source(i))]
                if not holders and part not in self.rest_searched and len(self.rest_searched) < PAIR_SEARCHES and self.left() >= 12:
                    self.rest_searched.add(part)
                    before = self.step
                    self.code_search(part)
                    self.ref_steps += self.step - before
                    holders = [i for i in self.seen_order if i not in self.codes and part in _code_tokens(self.code_source(i))]
                picks += [(0, i) for i in holders[:2] if (0, i) not in picks]
            if not picks:
                return
            self.read_codes(picks, models)

    def code_fits(self, ident: str, depth: int = 0) -> tuple[list[str], list[str], bool]:
        """What a listing works with: its own list, notes other listings give about its part, and the list of the part
        its own list goes on at (followed a few deep); its exceptions; and whether every part it points to was read."""
        entry = self.codes.get(ident) or {}
        fits = list(entry.get("fits") or [])
        excl = list(entry.get("except") or [])
        own = entry.get("own") or ""
        if own:
            for other in self.codes.values():
                for part, more in other.get("notes") or []:
                    if part == own:
                        fits += more
        complete = not entry.get("partial")
        rest = entry.get("rest") or ""
        if rest:
            holder = self.code_owner(rest) if depth < 4 else None
            if holder and holder != ident:
                more, out, whole = self.code_fits(holder, depth + 1)
                fits += more
                excl += out
                complete = complete and whole
            else:
                complete = False
        return fits, excl, complete

    def works(self, dep: str, anchor: str) -> bool | None:
        """Whether listing dep works with the anchor listing by the codes read: True, False, or None when unknown."""
        code = (self.codes.get(anchor) or {}).get("own") or ""
        if not code or dep not in self.codes:
            return None
        fits, excl, complete = self.code_fits(dep)
        if not fits and not (self.codes[dep].get("rest")):
            return None
        if any(_code_match(code, e) for e in excl):
            return False if complete or not any(_code_match(code, e) for e in fits) else None
        if any(_code_match(code, e) for e in fits):
            return True
        return False if complete else None

    def pair_sets(self, limit: int = 6) -> list[tuple[float, list[str]]]:
        """The cheapest sets by the codes: for each anchor listing, the cheapest listing of each dependent item that works
        with it (inside the budget), cheapest total first."""
        links = self.pair_links()
        groups = self.item_rows()
        anchors = sorted({a for _, a in links})
        out: list[tuple[float, list[str]]] = []
        if len(anchors) != 1 or anchors[0] in {d for d, _ in links}:
            # several anchors, or a chain of links: no set covers every linked item this way, so none is claimed
            return out
        # the requested items outside the links still take their cheapest buyable row out of a total budget
        others = 0.0
        for number in range(1, len(self.items) + 1):
            if number in anchors or number in {d for d, _ in links}:
                continue
            prices = [self.price_of(i) for i in groups.get(number, []) if self.buyable(i) and self.price_of(i) is not None]
            others += min(prices) * self.items[number - 1].qty if prices else 0.0
        for anchor_item in anchors:
            deps = [d for d, a in links if a == anchor_item]
            pool = [i for i in groups.get(anchor_item, []) if self.buyable(i) and self.affordable(i, anchor_item) and (self.codes.get(i) or {}).get("own")]
            for anchor in pool:
                chosen = [anchor]
                total = (self.price_of(anchor) or 0.0) * self.items[anchor_item - 1].qty
                for d in deps:
                    fits = [i for i in groups.get(d, []) if self.buyable(i) and self.affordable(i, d) and self.works(i, anchor)]
                    if not fits:
                        chosen = []
                        break
                    best = min(fits, key=lambda i: self.price_of(i) if self.price_of(i) is not None else float("inf"))
                    chosen.append(best)
                    total += (self.price_of(best) or 0.0) * self.items[d - 1].qty
                if not chosen:
                    continue
                if self.budget is not None and self.brief.get("budget_scope") != "each" and total + others > float(self.budget) + 1e-9:
                    continue
                out.append((total, chosen))
        out.sort(key=lambda t: t[0])
        return out[:limit]

    def compat_text(self) -> str:
        links = self.pair_links()
        groups = self.item_rows()
        lines = ["[Raikou matched the model and part codes of the listings seen, its own reading: each listing's own list, ranges"
                 " and their exceptions, the list of the part a list goes on at, and notes other listings give about a part]"]
        coded = 0
        for anchor_item in sorted({a for _, a in links}):
            pool = [i for i in groups.get(anchor_item, []) if self.buyable(i) and self.affordable(i, anchor_item) and (self.codes.get(i) or {}).get("own")]
            if pool:
                lines.append(f"item {anchor_item} ({self.items[anchor_item - 1].name}) codes: "
                             + ", ".join(f"{i} {self.codes[i]['own']} ({_num(self.price_of(i) or 0.0)})" for i in pool[:12]))
        for dep_item, anchor_item in links:
            anchors = [i for i in groups.get(anchor_item, []) if (self.codes.get(i) or {}).get("own")]
            if not anchors:
                continue
            for ident in [i for i in groups.get(dep_item, []) if i in self.codes and self.buyable(i) and self.affordable(i, dep_item)][:14]:
                fits, _excl, _whole = self.code_fits(ident)
                if not fits and not self.codes[ident].get("rest"):
                    # no code of its own read: nothing to say about it
                    continue
                coded += 1
                yes = [a for a in anchors if self.works(ident, a)]
                unknown = [a for a in anchors if self.works(ident, a) is None]
                text = f"item {dep_item}: {ident} ({_num(self.price_of(ident) or 0.0)}) works with: " + (", ".join(yes) if yes else "none of the item " + str(anchor_item) + " listings seen")
                if unknown and self.codes[ident].get("rest") and not self.code_owner(self.codes[ident]["rest"]):
                    text += f" (its list goes on at part {self.codes[ident]['rest']}, not found)"
                elif unknown:
                    text += " (its list could not be read in full)"
                lines.append(text)
        if not coded:
            return ""
        anchors = {a for _, a in links}
        if len(anchors) == 1 and not anchors & {d for d, _ in links}:
            sets = self.pair_sets()
            if sets:
                lines.append("Sets of the linked items that work together, cheapest total first (requested quantities; any other item"
                             " still needs its own line): " + "; ".join(" + ".join(ids) + f" = {_num(total)}" for total, ids in sets))
            else:
                lines.append("No set of the listings seen works together inside the budget yet: list each dependent item's category further.")
        lines.append("A set counts only when each line also meets its own item's other requirements. Under a cheapest pick the lowest"
                     " total of a set that meets everything wins; the cheapest single listing may have no partner.")
        return "\n".join(lines)

    def pair_complete(self) -> None:
        """A forced finish whose cart has the anchor item's line but no line for an item tied to it: the cheapest listing
        that works with that line by the codes goes in (inside the budget); a cart without that line fails anyway."""
        links = self.pair_links()
        if not links or not self.codes or self.left() < 3 or self.finished:
            return
        groups = self.item_rows()
        lines = {i: q for i, q in self.cart_lines.items() if not self.sold_out(i)}
        changed = False
        for dep_item, anchor_item in links:
            if any(i in groups.get(dep_item, []) for i in lines):
                continue
            anchor = next((i for i in lines if i in groups.get(anchor_item, [])), None)
            if anchor is None:
                continue
            qty = self.items[dep_item - 1].qty
            spent = sum((self.price_of(i) or 0.0) * q for i, q in lines.items())
            room = None if self.budget is None else (float(self.budget) if self.brief.get("budget_scope") == "each" else float(self.budget) - spent)
            partners = [i for i in groups.get(dep_item, []) if self.buyable(i) and self.works(i, anchor) and self.price_of(i) is not None
                        and (room is None or (self.price_of(i) or 0.0) * qty <= room + 1e-9)]
            if partners:
                lines[min(partners, key=lambda i: self.price_of(i) or 0.0)] = qty
                changed = True
        if changed:
            self.trace.append({"kind": "pair_complete", "lines": dict(lines)})
            self.put_lines(lines)

    def pair_notes(self, ids: list[str]) -> str:
        groups = self.item_rows()
        out = []
        for dep_item, anchor_item in self.pair_links():
            for dep in [i for i in ids if i in groups.get(dep_item, [])]:
                for anchor in [i for i in ids if i in groups.get(anchor_item, [])]:
                    verdict = self.works(dep, anchor)
                    if verdict is not None:
                        out.append(f"{dep} " + ("works with" if verdict else "does not work with") + f" {anchor} ({(self.codes.get(anchor) or {}).get('own')})")
        return "\n".join(out)

    def pair_clash(self) -> str:
        """Before a checkout: a cart line that, by the codes read, does not work with the line of the item it depends on;
        or, under a cheapest pick, a cheaper set that works together. The codes are read here first when no reading
        ran yet."""
        if self.pair_holds >= 2 or not self.pair_links():
            return ""
        if not self.paired and self.left() >= 8 and self.elapsed() < LATE_AT - 20 and self.cost < SPEND_HARD:
            self.pair_up([LIGHT])
        if not self.codes:
            return ""
        groups = self.item_rows()
        cart = self.cart_ids()
        for dep_item, anchor_item in self.pair_links():
            deps = [i for i in cart if i in groups.get(dep_item, [])]
            anchors = [i for i in cart if i in groups.get(anchor_item, [])]
            if len(deps) != 1 or len(anchors) != 1 or self.works(deps[0], anchors[0]) is not False:
                continue
            self.pair_holds += 1
            others = [i for i in groups.get(dep_item, []) if self.buyable(i) and self.works(i, anchors[0])]
            sets = self.pair_sets(4)
            text = (f"{deps[0]} does not work with {anchors[0]} ({self.codes[anchors[0]]['own']}) by the codes on the listings (its own list, the list of"
                    f" the part it goes on at, and notes on other listings)")
            if others:
                text += "; item " + str(dep_item) + " listings that do: " + ", ".join(f"{i} ({_num(self.price_of(i) or 0.0)})" for i in others[:5])
            if sets:
                text += ". Sets that work together, cheapest first: " + "; ".join(" + ".join(ids) + f" = {_num(total)}" for total, ids in sets)
            self.trace.append({"kind": "pair_clash", "dep": deps[0], "anchor": anchors[0]})
            return text + ". Fix the cart, then check out."
        if self.brief.get("pick") == "cheapest" and not self.pair_cheap_told:
            involved = sorted({n for pair in self.pair_links() for n in pair})
            lines = {n: [i for i in cart if i in groups.get(n, [])] for n in involved}
            if all(len(v) == 1 for v in lines.values()):
                total = sum((self.price_of(v[0]) or 0.0) * self.items[n - 1].qty for n, v in lines.items())
                cheaper = [(t, ids) for t, ids in self.pair_sets(3) if t < total - 1e-9 and set(ids) != {v[0] for v in lines.values()}]
                if cheaper:
                    self.pair_cheap_told = True
                    self.pair_holds += 1
                    self.trace.append({"kind": "pair_cheaper", "cart": total, "best": cheaper[0][0]})
                    return ("By the codes on the listings a cheaper set works together: " + "; ".join(" + ".join(ids) + f" = {_num(t)}" for t, ids in cheaper)
                            + f" (the cart's set comes to {_num(total)}). Check that set's lines against their other requirements on their pages;"
                            " take the cheapest set that meets them all, or check out as it is when none does.")
        return ""

    def shop_seen(self, shop: str) -> int:
        """How many of a shop's rows have been seen."""
        return len(self.name_rows(shop))

    def chase_due(self) -> bool:
        live = {need for _, need in self.needs_now()}
        # a chain that stopped at a listing no row matched is taken up again once more rows of that shop are seen (twice)
        for need, (shop, seen) in list(self.chase_wait.items()):
            if self.shop_seen(shop) > seen and need not in self.ref_values:
                self.chase_wait.pop(need)
                self.chased.discard(need)
        # the chase searches like the agent does, so it waits for the agent's first keyword search when the tool needs one
        return (any(need in live and need not in self.ref_values and need not in self.chased for _, need in self.refs)
                and self.left() >= 14 and self.elapsed() < CHASE_START and self.cost < SPEND_SOFT and self.ref_steps < REF_STEPS
                and not self.finished and self.search_args("probe") is not None)

    def chase_refs(self) -> str:
        """Follow each pointer requirement page by page, code-led: a light reading of the requirement names the first
        listing (its shop, kind and the details that pick it out); the code finds that shop's rows, opens the ones whose
        texts best carry those details and keeps the page that shows them (a decoy of the same shop shows other values);
        a light reading of that page gives the property's value, a pairing rule for it, or the next listing; and so on.
        Bounded: REF_HOPS pages a requirement, REF_STEPS store calls, CHASE_SECS seconds. A requirement still open after
        this is left to the check before checkout."""
        live = {need for _, need in self.needs_now()}
        pending = [need for _, need in self.refs if need in live and need not in self.ref_values and need not in self.chased]
        self.chased.update(pending)
        began = self.elapsed()
        texts: list[str] = []
        for need in pending:
            if self.chase_spent(began):
                break
            texts += self.follow(need, began)
        if self.ref_values:
            texts.append("[values read from the pages requirements point at] " + "; ".join(f"{k} -> {v}" for k, v in self.ref_values.items()))
        self.trace.append({"kind": "chase", "steps": self.ref_steps, "values": dict(self.ref_values)})
        return "\n".join(texts)

    def chase_spent(self, began: float) -> bool:
        return (self.left() < 12 or self.finished or self.elapsed() > CHASE_STOP or self.elapsed() - began > CHASE_SECS
                or self.cost > SPEND_SOFT or self.ref_steps >= REF_STEPS or self.link_reads >= REF_READS or self.calls > ALL_CALLS - 30)

    def follow(self, need: str, began: float) -> list[str]:
        """One pointer requirement's chain, link by link, up to its value (or the caps)."""
        prop = need.split(":", 1)[0].strip() or need
        got = self.first_links.get(need)
        if got is None:
            got = self.read_link(prop, need, False)
            if got:
                self.first_links[need] = got
        own = next((n for n, p in self.refs if p == need), None)
        other = _count(got.get("item"))
        target = self.link_target(got.get("next"), need)
        if other and 1 <= other <= len(self.items) and other != own and target is None:
            # the value comes from the listing chosen for another requested item: that pick settles it, no search can
            self.ref_items[need] = other
            return []
        raw = got.get("next") if isinstance(got.get("next"), dict) else {}
        named = str(raw.get("shop") or "").strip() if isinstance(raw.get("shop"), str) else ""
        if target is None and 2 <= len(named) <= 60 and self.rechased.get(need, 0) < 2:
            # a name the texts do not show as a name yet: taken up again once rows carrying it are seen
            self.rechased[need] = self.rechased.get(need, 0) + 1
            self.chase_wait[need] = (re.sub(r"'s$", "", named), self.shop_seen(re.sub(r"'s$", "", named)))
        texts: list[str] = []
        pairs: dict[str, str] = {}
        path: list[str] = []
        for _hop in range(REF_HOPS):
            if target is None or self.chase_spent(began):
                break
            # the best candidate is read; one the reader says is not the listing described is set aside and the
            # search goes on (a few times), which may open further rows
            mark = (target["shop"].lower(), tuple(d.lower() for d in target["details"]))
            refused = self.link_refused.setdefault(mark, set())
            ident, got, source = None, {}, ""
            unread = False
            for _try in range(3):
                found, notes = self.find_link(target, path + sorted(refused))
                texts += notes
                if not found or (_try and self.chase_spent(began)):
                    break
                candidate = found[0]
                self.page_view(candidate)
                if candidate not in self.opened:
                    # its page could not be opened (steps): a search row is not the page
                    unread = True
                    break
                source = self.page_text(candidate)
                got = self.read_link(prop, source, True, target)
                if not got:
                    # no reading came back (both models failed): no verdict on this page, the chain waits
                    unread = True
                    break
                whole = self.link_score(candidate, target)[1] == len(target["details"])
                if got.get("is_it") is True or (got.get("is_it") is None and whole):
                    ident = candidate
                    break
                refused.add(candidate)
            if ident is None and unread:
                texts.append(f"{need}: the page of a {target['shop']} listing could not be read")
                break
            if ident is None:
                texts.append(f"{need}: no listing of {target['shop']} matched {target['kind']} ({', '.join(target['details']) or 'no details'})")
                if self.rechased.get(need, 0) < 2:
                    self.rechased[need] = self.rechased.get(need, 0) + 1
                    self.chase_wait[need] = (target["shop"], self.shop_seen(target["shop"]))
                break
            path.append(ident)
            for pair in _items(got.get("pairs")):
                if isinstance(pair, list) and len(pair) == 2 and all(isinstance(x, str) and x.strip() for x in pair):
                    pairs.setdefault(pair[0].strip().lower(), pair[1].strip())
            value = str(got.get("value") or "").strip()
            target = self.link_target(got.get("next"), source)
            own_names = {str((self.rows.get(ident) or {}).get(k) or "").strip().lower() for k in self.row_fields}
            if target is not None and target["shop"].lower() in own_names and not target["details"]:
                # a page naming its own shop with nothing to pick another listing out sends the reader nowhere new
                target = None
            if target is None and value and not self.on_page(value, source):
                # a value the page does not carry is a misreading: the chain stays open for the check before checkout
                texts.append(f"{need}: the reading of {ident} gave {value!r}, which its page does not show")
                break
            if target is None and value:
                paired = pairs.get(value.lower())
                self.ref_values[need] = value + (f" (the store's rule pairs {value} with {paired})" if paired else "")
                self.ref_paths[need] = list(path)
                self.version += 1
                texts.append(f"{need} -> {self.ref_values[need]} (pages read: {' -> '.join(path)})")
                break
        self.trace.append({"kind": "follow", "need": need[:80], "path": path, "value": self.ref_values.get(need, "")})
        return texts

    def read_link(self, prop: str, text: str, page: bool, looking: dict[str, Any] | None = None) -> dict[str, Any]:
        prompt = LINK + "\n\nPROPERTY: " + prop
        if not page:
            prompt += "\n\nITEMS OF THIS REQUEST:\n" + "\n".join(f"item {n}: {it.name}" for n, it in enumerate(self.items, 1))
        if looking:
            prompt += f"\n\nLOOKING FOR: the {looking['kind'] or 'listing'} of {looking['shop']}" + (" with " + "; ".join(looking["details"]) if looking["details"] else "")
        prompt += ("\n\nTHE PAGE:\n" if page else "\n\nTHE REQUIREMENT:\n") + text
        self.link_reads += 1
        got = self.ask_json(prompt, [LIGHT, HEAVY], 700) or {}
        self.trace.append({"kind": "link", "page": page, "got": _clip(got, 120)})
        return got

    def link_target(self, raw: Any, source: str) -> dict[str, Any] | None:
        """A link the reader named: a shop or brand name the source text itself writes (a name it did not write is
        dropped), the kind of product and the details that pick the listing out."""
        if not isinstance(raw, dict):
            return None
        shop = re.sub(r"'s$", "", str(raw.get("shop") or "").strip())
        if not 2 <= len(shop) <= 60 or not re.search(r"(?<![A-Za-z0-9])" + re.escape(shop) + r"(?![A-Za-z0-9])", source, re.I):
            return None
        known = any(str((row or {}).get(k) or "").strip().lower() == shop.lower() for row in self.rows.values() for k in self.row_fields)
        inside = re.compile(r"[^\s.!?;\"\[{(]\s*" + re.escape(shop) + r"(?![A-Za-z0-9])|[A-Za-z0-9,:]\s+" + re.escape(shop) + r"(?![A-Za-z0-9])")
        if not known and not (re.search(r"[A-Z0-9]", shop[:1]) and inside.search(source)):
            # a name is written as one: a shop or brand some row carries, or a capitalised word inside a sentence
            return None
        details = [str(d).strip() for d in _items(raw.get("details")) if isinstance(d, (str, int, float)) and not isinstance(d, bool) and str(d).strip()]
        return {"shop": shop, "kind": str(raw.get("kind") or "").strip()[:80], "details": details[:8]}

    def page_text(self, ident: str) -> str:
        """A listing's page as the link reader sees it: the row and the page record together, long texts cut."""
        record = {**(self.rows.get(ident) or {}), **(self.full.get(ident) or {})}
        return _dump(_clip(record, 500))[:3500]

    def name_rows(self, name: str) -> list[str]:
        """Seen rows that carry this shop or brand name as one of their row values (any case); else rows whose row texts
        write it as a word."""
        low = name.strip().lower()
        exact = [i for i in self.seen_order if any(str((self.rows.get(i) or {}).get(k) or "").strip().lower() == low for k in self.row_fields)]
        if exact:
            return exact
        word = re.compile(r"(?<![A-Za-z0-9])" + re.escape(name.strip()) + r"(?![A-Za-z0-9])", re.I)
        return [i for i in self.seen_order if word.search(_dump({k: v for k, v in (self.rows.get(i) or {}).items() if k in self.row_fields}))]

    def link_score(self, ident: str, target: dict[str, Any]) -> tuple[float, int]:
        """How well a seen listing matches a link: the share of its kind words, plus each detail read over the listing's
        attributes (full weight) or its search-row texts (less: a description can mention values the listing lacks);
        a page's free remarks can name other versions, so they do not count. Also how many details it clearly carries
        (every number in the detail, or most of its words)."""
        record = {k: v for k, v in {**(self.rows.get(ident) or {}), **(self.full.get(ident) or {})}.items() if k not in (self.id_key, self.price_key)}
        row = {k: v for k, v in record.items() if k in self.row_fields}
        facts = _fact_block(self.full.get(ident)) if ident in self.full else None
        row_text = _dump(row)
        fact_text = _dump(facts) if facts else ""
        # the page's own sentences: page-only texts that name the shop being looked for (a sentence about another
        # listing names that listing's shop, or none, as in a note on versions found elsewhere)
        shop = re.compile(r"(?<![A-Za-z0-9])" + re.escape(target["shop"]) + r"(?![A-Za-z0-9])", re.I)
        own = " ".join(part for k, v in record.items() if k not in self.row_fields and isinstance(v, str)
                       for part in re.split(r"(?<=[.!?])\s+", v) if shop.search(part))
        fact_text += " " + own
        kind = {w for w in _tokens(target["kind"]) if len(w) >= 3}
        score = 0.5 * len(kind & _tokens(row_text + " " + fact_text)) / len(kind) if kind else 0.0
        hits = 0
        row_nums, row_words = _numbers(row_text), _tokens(row_text)
        fact_nums, fact_words = _numbers(fact_text), _tokens(fact_text)
        for detail in target["details"]:
            on_facts = _detail_score(detail, fact_nums, fact_words) if fact_text.strip() else 0.0
            on_row = _detail_score(detail, row_nums, row_words)
            need = 1.0 if _numbers(detail) else 0.6
            hits += on_facts >= need or on_row >= need
            score += max(on_facts, 0.6 * on_row)
        return score, hits

    def facts_view(self) -> str | None:
        return self.facts_tool() or self.roles.get("open") or self.roles.get("offer")

    def open_page(self, ident: str) -> bool:
        """Open a candidate's page (the view a chain is read from; its own sentences often restate its details)."""
        return self.open_view(ident, self.roles.get("offer") or self.roles.get("open"))

    def open_facts(self, ident: str) -> bool:
        """Open a candidate's attributes when they come from a view apart from the page."""
        tool = self.facts_view()
        if not tool or tool == (self.roles.get("offer") or self.roles.get("open")) or ident in self.full:
            return False
        return self.open_view(ident, tool)

    def page_view(self, ident: str) -> None:
        """The page (with its remarks) of a candidate not opened yet (its attributes may have come from a separate view,
        or a search row): a chain is read from the pages."""
        tool = self.roles.get("offer") or self.roles.get("open")
        if tool and ident not in self.opened and self.ref_steps < REF_STEPS + 2 and self.left() >= 10:
            self.open_view(ident, tool)

    def on_page(self, value: str, source: str) -> bool:
        """Whether a value read from a page is carried by it: its numbers and most of its words are in the page's text."""
        have_nums, have = _numbers(source), _tokens(source)
        if any(all(abs(n - m) > 1e-6 * max(1.0, abs(n)) for m in have_nums) for n in _numbers(value)):
            return False
        words = {w for w in _tokens(re.sub(r"\d+(?:\.\d+)?", " ", value)) if len(w) >= 2}
        return not words or len(words & have) * 2 >= len(words)

    def open_view(self, ident: str, tool: str | None) -> bool:
        target = self.single_target(tool)
        if not tool or not target:
            return False
        faults: list[str] = []
        args = _shape({target: ident}, self.schemas.get(tool), "args", faults)
        if faults:
            return False
        reply = self.act(tool, args)
        self.ref_steps += 1
        if reply is None or self.finished:
            return False
        self.observe(tool, reply.get("observation"))
        self.note_view(tool, reply.get("observation"))
        return True

    def find_link(self, target: dict[str, Any], path: list[str]) -> tuple[list[str], list[str]]:
        """The listings looked at (attributes or page read) that may be the one a link names, best first. The shop's rows (searched for once per shop
        name) are ranked by how well their texts carry the link's kind and details and the best few are opened; the
        search stops at a page whose attributes clearly carry every detail. When no row of the first result page shows a
        sign of the details and the search has a next page, that page is read before rows without a sign are opened.
        A page already on this chain is not taken again."""
        notes: list[str] = []
        shop = target["shop"]
        key = shop.lower()
        if key not in self.link_searched:
            self.link_searched[key] = None
            # the shop's name alone: a store matching any word would flood the result with the kind's words, one matching
            # every word gains nothing from them
            query = shop
            if self.link_search(query, key) is not None:
                notes.append(f"search {query!r}: {len(self.name_rows(shop))} rows of {shop} seen")
        opened = 0
        wanted = len(target["details"])
        whole = None
        for turn in range(2):
            more = self.link_searched.get(key)
            ranked = sorted((i for i in self.name_rows(shop) if i not in path), key=lambda i: (-self.link_score(i, target)[0], self.seen_order.index(i)))
            for ident in ranked:
                if not self.looked(ident):
                    if wanted and self.link_score(ident, target)[1] == 0 and more:
                        continue
                    if opened >= REF_OPENS + 2 * turn or self.ref_steps >= REF_STEPS or self.left() < 12 or self.finished:
                        continue
                    if not self.open_page(ident):
                        break
                    opened += 1
                if self.looked(ident) and self.link_score(ident, target)[1] == wanted and (wanted or self.link_score(ident, target)[0] > 0):
                    whole = ident
                    break
            if whole or self.ref_steps >= REF_STEPS or self.left() < 12 or self.finished:
                break
            if more:
                # the shop has more rows than the first page showed: its next page, once
                self.link_searched[key] = None
                if self.link_page(more) is None:
                    break
                notes.append(f"next page of the search for {shop}")
            elif all(self.looked(i) for i in ranked):
                break
        if not whole and wanted:
            # pages that did not show every detail: their attributes, when a separate view holds them (two at most)
            maybe = sorted((i for i in self.name_rows(shop) if self.looked(i) and i not in path and i not in self.full),
                           key=lambda i: (-self.link_score(i, target)[0], self.seen_order.index(i)))
            for ident in maybe[:2]:
                if self.ref_steps >= REF_STEPS or self.left() < 12 or self.finished or not self.open_facts(ident):
                    break
                if self.link_score(ident, target)[1] == wanted:
                    whole = ident
                    break
        found = [i for i in self.name_rows(shop) if self.looked(i) and i not in path]
        found.sort(key=lambda i: (i != whole, -self.link_score(i, target)[0], self.seen_order.index(i)))
        if wanted:
            found = [i for i in found if i == whole or self.link_score(i, target)[0] > 0.5]
        return found, notes

    def looked(self, ident: str) -> bool:
        return ident in self.full or ident in self.opened

    def link_search(self, query: str, key: str) -> Any:
        """A keyword search the chase chose; remembers how to ask for its next page under the shop's key."""
        tool = self.roles.get("search")
        args = self.search_args(query)
        if not tool or args is None or self.left() < 12 or self.finished:
            return None
        reply = self.act(tool, args)
        self.ref_steps += 1
        if reply is None:
            return None
        seen = reply.get("observation")
        self.observe(tool, seen)
        taught = self.learn_words(tool, args, seen)
        if taught:
            self.queued.append(taught)
        plan = self.paging(tool, seen)
        if plan and plan[3] is not None and plan[0] > plan[1]:
            self.link_searched[key] = (tool, {**args, plan[2]: plan[3]})
        return seen

    def link_page(self, call: tuple[str, dict[str, Any]]) -> Any:
        tool, args = call
        faults: list[str] = []
        args = _shape(args, self.schemas.get(tool), "args", faults)
        if faults:
            return None
        reply = self.act(tool, args)
        self.ref_steps += 1
        if reply is None:
            return None
        seen = reply.get("observation")
        self.observe(tool, seen)
        return seen

    def learn_words(self, name: str, args: dict[str, Any], observation: Any) -> str:
        """Whether the store's keyword search matches any or every word of a query: from the match counts of first-page
        searches whose words nest, or from one search whose rows mostly show a single query word and include categories
        that share none. Returns a note when it is first learned that the store matches any word."""
        if self.word_match is not None or not isinstance(observation, dict):
            return ""
        props = (self.schemas.get(name) or {}).get("properties") or {}
        free = [k for k, v in props.items() if isinstance(v, dict) and "string" in _kinds(v) and "enum" not in v]
        texts = {k: v for k, v in args.items() if k in free and isinstance(v, str) and v.strip()}
        if len(texts) != 1:
            return ""
        key, phrase = next(iter(texts.items()))
        words = frozenset(w for w in re.findall(r"[a-z0-9]+", phrase.lower()) if len(w) >= 3)
        if not words:
            return ""
        shape = self.read_shape(name, observation) or {}
        page_arg = shape.get("next_arg")
        given = args.get(page_arg) if page_arg else None
        if given not in (None, "", 0, 1, "0", "1"):
            return ""
        count = observation.get(shape["count"]) if shape.get("count") else None
        rest = _dump({k: v for k, v in sorted(args.items()) if k not in (key, page_arg)})
        if isinstance(count, int) and not isinstance(count, bool):
            for (tool, other_rest, other), total in self.search_totals.items():
                if tool != name or other_rest != rest or other == words:
                    continue
                if (other < words and total < count) or (words < other and count < total):
                    self.word_match = "any"
                elif (other < words and total > count) or (words < other and count > total):
                    self.word_match = "all"
            self.search_totals[(name, rest, words)] = count
        rows = self.records_in(observation)
        cat_key = self.category_key()
        if self.word_match is None and len(words) >= 2 and len(rows) >= 4 and cat_key:
            def hits(text: Any) -> int:
                have = set(re.findall(r"[a-z0-9]+", str(text).lower()))
                return sum(1 for w in words if w in have or any(h.startswith(w[:max(3, len(w) - 1)]) for h in have))

            sparse = sum(1 for _, r in rows if hits(" ".join(v for v in r.values() if isinstance(v, str))) <= max(1, len(words) - 2))
            stray = sum(1 for _, r in rows if isinstance(r.get(cat_key), str) and hits(r[cat_key]) == 0 and hits(" ".join(v for v in r.values() if isinstance(v, str))) >= 1)
            if sparse >= max(3, (len(rows) + 1) // 2) and stray >= 2:
                self.word_match = "any"
        if self.word_match == "any":
            return ("This store's keyword search matches ANY of the query's words (more words give more rows, not fewer, from other kinds too): "
                    "search one distinctive wanted value, shop or brand at a time, list the item's own category with the category filter, and page on.")
        return ""

    def sweep(self) -> str:
        """Before an ending without a purchase: search each item by a wanted value (with the item's name unless the store
        matches any word), then open the best new rows of that item's kind; SWEEP_STEPS store calls for the searches."""
        if self.swept or self.left() < 10 or self.elapsed() > LATE_AT - 20 or self.cost > SPEND_SOFT:
            return ""
        self.swept = True
        cat_key = self.category_key()
        texts: list[str] = []
        fresh: list[str] = []
        for number, item in enumerate(self.items, 1):
            if len(texts) >= SWEEP_STEPS or self.left() < 9:
                break
            values = [m.split(":", 1)[1].strip() for m in item.musts + item.told if ":" in m and m not in item.dropped]
            values = [v for v in values if v and len(v.split()) <= 3 and not re.search(r"\d", v)]
            words = values[0] if values and self.word_match == "any" else " ".join(values[:1] + [item.name])
            shown = self.code_search(words) if words.strip() else None
            if shown is None:
                continue
            texts.append(f"search {words!r}: " + shown[:1200])
            kind = {w for w in _stem(item.shelf or "") | _stem(item.name) if len(w) >= 4}
            for ident in reversed(self.seen_order):
                row = self.rows.get(ident) or {}
                if ident in self.full or ident in self.opened or ident in fresh or not self.buyable(ident):
                    continue
                if cat_key and kind and not (kind & _stem(str(row.get(cat_key) or ""))):
                    continue
                price = self.price_of(ident)
                if self.budget is not None and price is not None and price * item.qty > self.budget:
                    continue
                fresh.append(ident)
                if len([i for i in fresh]) >= 2 * number:
                    break
        if fresh and self.left() > 8:
            before = set(self.full)
            self.open_rows(fresh[:4])
            opened = [i for i in self.full if i not in before]
            if opened:
                texts.append("opened: " + "\n".join(f"{i}: {self.record_of(i)[:900]}" for i in opened))
        self.trace.append({"kind": "sweep", "searches": len(texts), "fresh": fresh[:4]})
        return "\n".join(texts)

    # ---------- what the cart tool's replies teach ----------
    def cart_rows_in(self, name: str, observation: Any) -> dict[str, int] | None:
        """The whole cart as a cart tool's (or the cart view's) reply lists it, {listing id: units}, read by shape: the one
        list whose rows each name a listing seen, and the one whole-number key those rows share besides the price. A store
        note may wrap the real reply under another key, so the reply and each object one level down are read alike. The key
        that held the lines is kept per tool, so an empty list under it later reads as an empty cart. None when the reply
        lists no lines."""
        if not isinstance(observation, dict):
            return None
        inner = [v for v in observation.values() if isinstance(v, dict)]
        texts = all(isinstance(v, (dict, str)) or (isinstance(v, list) and all(isinstance(x, str) for x in v)) for v in observation.values())
        if len(inner) != 1 or not texts:
            # not a note wrap (one object beside note texts): only the reply's own lists are read
            inner = []
        known = self.cart_list_key.get(name)
        if known is not None:
            if isinstance(observation.get(known), list):
                return self.lines_in(observation[known]) if observation[known] else {}
            held = [v[known] for v in inner if isinstance(v.get(known), list)]
            if len(held) == 1:
                return self.lines_in(held[0]) if held[0] else {}
            read = [x for x in (self.lines_in(h) for h in held) if x]
            return read[0] if len(read) == 1 else None
        # the reply's own lists first; the objects one level down only when none of those is the cart
        for layers in ([observation], inner):
            found = [(key, lines) for layer in layers for key, value in layer.items() for lines in [self.lines_in(value)] if lines is not None]
            if len(found) == 1:
                self.cart_list_key[name] = found[0][0]
                return found[0][1]
            if found:
                return None
        return None

    def lines_in(self, value: Any) -> dict[str, int] | None:
        """Cart lines in a list: every row names a listing seen and the rows share one whole-number key from 1 to 999 that
        is not the price (by the price key or by value; the reply's rows were already merged into the listings' records, so
        their keys prove nothing). None when the list is not that."""
        if not isinstance(value, list) or not value or not all(isinstance(row, dict) for row in value):
            return None
        idents = [next((v.strip() for v in row.values() if isinstance(v, str) and v.strip() in self.rows), None) for row in value]
        if None in idents:
            return None
        keys = None
        for ident, row in zip(idents, value):
            price = self.price_of(str(ident))
            here = {k for k, v in row.items() if isinstance(v, int) and not isinstance(v, bool) and 0 < v < 1000
                    and k != self.price_key and (price is None or v != price)}
            keys = here if keys is None else keys & here
        if not keys or len(keys) != 1:
            return None
        unit = next(iter(keys))
        return {str(ident): row[unit] for ident, row in zip(idents, value)}

    def learn_cart_mode(self, name: str, args: dict[str, Any], before: dict[str, int], after: dict[str, int]) -> str:
        """What one add did, read from the cart the reply lists: one unit a call whatever number was passed, the number set as
        the line's units, or the number added to it. Returns a note for the agent when the tool's way is first learned."""
        spec = (self.schemas.get(name) or {}).get("properties") or {}
        if any(isinstance((spec.get(k) or {}).get("enum"), list) and len(spec[k]["enum"]) >= 2 and v in spec[k]["enum"][1:] for k, v in args.items()):
            return ""
        ident = next((v for v in args.values() if isinstance(v, str) and v in self.rows), None)
        if not ident:
            return ""
        number = next((_count(v) for v in args.values() if not isinstance(v, (str, bool, list, dict)) and _count(v) is not None), None)
        had, now = before.get(ident, 0), after.get(ident, 0)
        # a way is learned only when one reading alone explains the change (a count of 1 cannot tell adding from one unit)
        fits = []
        if now == had + 1 and number is not None and number >= 2:
            fits.append("one")
        if number and now == number and had not in (0, number):
            fits.append("set")
        if number and number >= 2 and had and now == had + number:
            fits.append("add")
        mode = fits[0] if len(fits) == 1 else None
        if not mode or self.cart_mode.get(name) == mode:
            return ""
        self.cart_mode[name] = mode
        if mode == "one":
            return f"This cart tool adds one unit per call, whatever number you pass (the cart now holds {now} of {ident}): call it once more for each further unit, and take a line out with its removal action."
        if mode == "set":
            return "This cart tool sets the line to the number of units you pass (it does not add to it)."
        return "This cart tool adds the number of units you pass to what the line already holds."

    def offer_tool(self) -> str | None:
        """Tool for re-reading a listing's live offer: the offer view when the store has one, else the single-listing view."""
        return self.roles.get("offer") or self.roles.get("open")

    def reread_offers(self) -> str:
        """Re-read each cart line with the store's offer view just before checkout: stock and price can change unannounced."""
        tool = self.offer_tool()
        target = self.single_target(tool)
        if not tool or not target:
            return ""
        since = max(self.step - 2, self.refused_at)
        stale = [i for i in self.cart_ids() if self.offer_seen.get(i, -99) < since and i not in self.gone]
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
        text = f"[Raikou] Sold out now: {', '.join(lost)} (the store shows it cannot be bought). It will not come back, so do not open it again."
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
        if len(ids) == 1:
            self.unavailable.add(ids[0])
            return ""
        return self.reread_offers() if ids else ""

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
        got = self.ask_json(prompt, self.late_models(), 1200)
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
        if self.audited or self.cost > SPEND_SOFT or self.elapsed() > LATE_AT:
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

    def audit_call(self, models: list[str] | None = None) -> dict[str, Any] | None:
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
        got = self.ask_json(prompt, models or self.late_models(), 1500)
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
        one = self.cart_mode.get(name) == "one"
        for ident, want in lines.items():
            have = self.cart_lines.get(ident, 0) if ident in self.cart_ids() else 0
            if have == want:
                continue
            if one:
                if have > want:
                    return None
                # the number is no quantity here: left out, or 1 where the tool requires it
                single = {count: 1} if count and count in (schema.get("required") or []) else {}
                calls.extend({target: ident, **({mode: props[mode]["enum"][0]} if mode else {}), **single} for _ in range(want - have))
                continue
            if have and not self.cart_sets(name):
                # a tool seen adding the number passed takes the difference; otherwise the change cannot be said
                if self.cart_mode.get(name) != "add" or not count or want < have:
                    return None
                calls.append({target: ident, **({mode: props[mode]["enum"][0]} if mode else {}), count: want - have})
                continue
            calls.append({target: ident, **({mode: props[mode]["enum"][0]} if mode else {}), **({count: want} if count else {})})
        return calls

    def complete_cart(self) -> str:
        """The agent ends without buying although the independent check confirmed a reviewed listing: Raikou carts the reviewed
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
        if not self.put_lines(lines, drop):
            return ""
        done = [f"{i} x{n}" for i, n in lines.items()]
        faults: list[str] = []
        args = _shape(dict(self.brief.get("buy_args") or {}), self.schemas.get(buy), "args", faults)
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
        return ("Not run (Raikou acted instead): the independent check confirmed " + self.confirmed + ", so Raikou set the cart (" + "; ".join(done)
                + ") and checked out. Store reply: " + " | then: ".join(replies))

    # ---------- tool use ----------
    def guess_roles(self) -> None:
        """After a reading of the request that named no checkout tool (its model calls failed or answered off-form): one
        short call that only maps the tools to their jobs, so the episode can still be worked and finished."""
        if self.roles.get("buy") or self.no_credit:
            return
        lines = [f"- {t['function']['name']}: {t['function'].get('description', '')} | parameters: {_dump(t['function'].get('parameters') or {})}" for t in self.tools]
        got = self.ask_json(ROLES + "\n\nTOOLS:\n" + "\n".join(lines), [LIGHT, HEAVY], 800) or {}
        roles = got.get("roles") if isinstance(got.get("roles"), dict) else {}
        for job, tool in roles.items():
            if isinstance(job, str) and isinstance(tool, str) and tool in self.schemas and not self.roles.get(job):
                self.roles[job] = tool
        for key in ("buy_args", "refuse_args"):
            if not self.brief.get(key) and isinstance(got.get(key), dict):
                self.brief[key] = got[key]
        self.trace.append({"kind": "roles_retry", "roles": dict(self.roles)})

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
        elif left <= len(self.items) + 5 and self.buys == 0:
            keep = set(ends) | {self.roles.get("cart"), self.roles.get("cart_view"), self.roles.get("offer")}
        elif self.elapsed() > WRAP_AT + 20:
            keep = set(ends) | {self.roles.get("cart"), self.roles.get("cart_view")}
        tools = [t for t in self.tools if t["function"]["name"] != ask]
        if keep is not None and ends:
            narrowed = [t for t in tools if t["function"]["name"] in keep]
            tools = narrowed or tools
        return tools

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
            named = any(ident in text for ident in self.rows) or self.brief.get("quiet_decline") is not False
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
        """A cart tool whose count may be exactly 0 and that has no mode for taking lines off sets a line's quantity: 0 removes it.
        What the store's own cart replies showed the tool doing comes first."""
        if self.cart_mode.get(name):
            return self.cart_mode[name] == "set"
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
        if self.cart_mode.get(name) == "one":
            # the number passed is not a quantity: each call moves one unit
            loose_count = None
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
        text = _dump(_clip(getattr(observation, "raw", observation), SHOW_CHARS))
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
                out.append((cid, "Shown in the side-by-side result above: Raikou opened these listings together in one step."))
                continue
            fn = call.get("function") or {}
            name = fn.get("name")
            raw = fn.get("arguments")
            if self.finished:
                out.append((cid, "Not run: the episode is over."))
                continue
            if name == self.roles.get("ask"):
                out.append((cid, "Not run (no step used): Raikou's interviewer handles every message to the shopper."))
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
                trimmed = f"\n[Raikou] Sent the first {room} listings only: this store shows at most {room} side by side. Not shown yet: {', '.join(map(str, args[many][room:]))}."
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
                    trimmed += f"\n[Raikou] Left out {', '.join(str(i) for i in args[many] if i not in keep)}: shown in full already, nothing changed since."
                    args = {**args, many: keep}
                    mark = name + _dump(args)
            if self.is_buy(name, args) and self.left() >= 2:
                short = self.short_cart() or self.count_hold()
                if short:
                    out.append((cid, short + "\nCart now: " + self.cart_text()))
                    continue
            if ending and self.left() > 2:
                if self.is_buy(name, args):
                    wider = self.wider_hold()
                    if wider:
                        out.append((cid, wider + "\nCart now: " + self.cart_text()))
                        continue
                    reread = self.reread_offers()
                    if self.cart_touched and self.cart_seen_at < self.cart_at and self.roles.get("cart_view") and self.left() > 3:
                        faults = []
                        view_args = _shape({}, self.schemas.get(self.roles["cart_view"]), "args", faults)
                        view = self.act(self.roles["cart_view"], view_args) if not faults else None
                        if view is not None:
                            self.cart_view = view.get("observation")
                            self.cart_seen_at = self.step
                            self.read_cart_reply(self.roles["cart_view"], self.cart_view)
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
                    before = self.step
                    clash = self.pair_clash()
                    if clash:
                        out.append((cid, ("Not run yet (no step used). " if self.step == before else "Not run yet. ") + clash + "\nCart now: " + self.cart_text()))
                        continue
                    chain, hold = self.trail()
                    if hold:
                        out.append((cid, "Not run yet (no step used). " + chain + "\nCart now: " + self.cart_text()))
                        continue
                    # the check judges attributes: read the cart lines' own (and, for a cheapest pick, a few cheaper rows')
                    read = self.fill_facts(self.cart_ids(), self.tail() + 3) + self.cheaper_rows()
                    verdict = self.check_cart(limit=SPEND_FINAL)
                    if verdict:
                        seen = ("Raikou read these listings' attributes for the check (one step each):\n" + "\n".join(read) + "\n") if read else ""
                        out.append((cid, "Not run yet. " + chain + seen + verdict + "\nCart now: " + self.cart_text()))
                        continue
                    if reread and self.cost > SPEND_SOFT:
                        out.append((cid, "Not run yet (no step used). Raikou re-read the cart's listings just now:\n" + reread + "\nCheck out again when every line still shows stock and a price inside the budget; otherwise replace that line first."))
                        continue
                else:
                    if self.budget_moved and self.left() > 8:
                        found = self.budget_rows()
                        if found:
                            out.append((cid, "Not run yet. " + found))
                            continue
                    if self.pair_links() and not self.pair_decline_told and self.left() > 6:
                        sets = self.pair_sets(4)
                        if sets:
                            self.pair_decline_told = True
                            out.append((cid, "Not run yet (no step used). By the codes on the listings these sets of the linked items work together"
                                        " and may fit the budget: " + "; ".join(" + ".join(ids) + f" = {_num(total)}" for total, ids in sets)
                                        + ". Check each line's other requirements on its page; buy the best set that meets them, with a line for every"
                                        " other requested item too, and end without a purchase only if none does."))
                            continue
                    if self.left() > 5:
                        swept = self.sweep()
                        if swept:
                            out.append((cid, "Not run yet. Before ending without a purchase, Raikou searched each item by a wanted value and opened new candidates (its own steps):\n"
                                        + swept + "\nCheck them against every requirement: buy what fits, and end without a purchase only if none does."))
                            continue
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
                # a cart the budget in force no longer allows (the shopper lowered it) is not offered back
                if len(lines) >= len(self.items) and len(missing) + 1 <= self.left() and not self.over_budget(lines):
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
                args = self.quiet(name, args)
            if ending and self.left() >= 2:
                go, said = self.gate("order" if self.is_buy(name, args) else "end")
                if not go:
                    out.append((cid, "Not run yet. " + said + "\nCart now: " + self.cart_text()))
                    continue
                if said:
                    trimmed = "\n[Raikou] " + said + trimmed
            reply = self.act(name, args)
            if reply is None:
                out.append((cid, "The call failed (network or server). You may try again."))
                continue
            observation = reply.get("observation")
            if many and isinstance(args.get(many), list) and len(args[many]) > 1:
                shown = _dump(observation)
                if not any(str(i) in shown for i in args[many]):
                    self.compare_bad = min(self.compare_bad or 999, len(args[many]))
                    trimmed += f"\n[Raikou] The side-by-side view showed nothing for {len(args[many])} listings: send at most {self.compare_room()} per call."
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
            paged = self.more_pages(name, args, observation) if name not in plain and not ending else ""
            if name == self.roles.get("search") and not ending:
                self.learn_search(name, args)
                taught = self.learn_words(name, args, observation)
                if taught:
                    trimmed += "\n[Raikou] " + taught
            if self.single_target(name) and name not in (self.roles.get("cart"), self.roles.get("buy"), self.roles.get("refuse"), self.roles.get("ask")):
                listing = self.ident(observation) if isinstance(observation, dict) else None
                if listing:
                    self.facts_from[name] = self.facts_from.get(name, False) or bool(_fact_block(observation))
                    if name == self.roles.get("open") and not self.facts_from[name]:
                        other = next((t for t in self.schemas if t != name and self.single_target(t) and self.facts_from.get(t) is not False
                                      and t not in (self.roles.get("cart"), self.roles.get("buy"), self.roles.get("refuse"), self.roles.get("ask"), self.roles.get("search"))), None)
                        if other:
                            self.roles["open"] = other
                    if not self.facts_from[name] and name != self.roles.get("open") and not self.roles.get("offer"):
                        self.roles["offer"] = name
                    if name == self.offer_tool():
                        self.offer_seen[listing] = self.step
            if name == self.roles.get("cart"):
                before = {i: self.cart_lines.get(i, 1) for i in self.cart_ids()}
                self.note_cart(name, args)
                truth = self.read_cart_reply(name, observation)
                if truth is not None:
                    learned = self.learn_cart_mode(name, args, before, truth)
                    if learned:
                        trimmed += "\n[Raikou] " + learned
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
                self.read_cart_reply(name, observation)
            if self.is_buy(name, args):
                self.buys += 1
                if not reply.get("finished"):
                    self.refused_at = self.step
            text = self.show(observation) + paged + trimmed
            if name not in plain and not ending:
                text += self.cheapest_here([observation] + self.paged_obs)
            base = text
            proof = self.evidence(observation)
            if proof:
                text += "\n" + proof
            if name == self.roles.get("cart") and any(_count(v) == 0 for v in args.values() if not isinstance(v, str)):
                if any(isinstance(v, str) and v in self.rows and _in_rows(observation, v) for v in args.values()):
                    text += "\n[Raikou] A count of 0 left that line in the cart: this tool adds the count to the line, so a negative count takes units off."
            self.note_sold = False
            lines = self.loose_lines(observation)
            if lines:
                meaning = self.take_note(lines)
                if meaning:
                    text += "\n[note read] " + meaning + ". Apply this change to your picks before checking out."
            if self.is_buy(name, args) and not reply.get("finished") and (self.note_sold or any(line in self.spoken(observation) for line in self.sold_lines)):
                seen = self.find_sold()
                if seen:
                    text += "\n[Raikou re-read the cart's lines after the refusal]\n" + seen
            swap = self.swap_hint()
            if swap:
                text += "\n" + swap
            if self.queued:
                text += "\n[notes seen on Raikou's own calls] " + " | ".join(self.queued)
                self.queued = []
            if self.budget_moved and not reply.get("finished") and not self.finished:
                found = self.budget_rows()
                if found:
                    text += "\n" + found
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
        fresh = self.cart_view is not None and self.cart_seen_at >= self.cart_at
        holding = bool(self.cart_ids()) if fresh else bool(self.cart_lines)
        if buy and not holding and self.fill_empty_cart():
            fresh = self.cart_view is not None and self.cart_seen_at >= self.cart_at
            holding = bool(self.cart_ids()) if fresh else bool(self.cart_lines)
        if buy and holding:
            self.pair_complete()
            self.top_up()
        if buy and (holding or (self.cart_touched and not fresh and self.left() >= 2)):
            faults: list[str] = []
            args = _shape(dict(self.brief.get("buy_args") or {}), self.schemas.get(buy), "args", faults)
            if not faults:
                # a reply that adds a requirement does not stop a forced finish: no steps are left to act on it
                if self.left() >= 2:
                    self.gate("order")
                if self.finished or self.left() <= 0:
                    return ""
                reply = self.act(buy, args)
                self.trace.append({"kind": "forced_buy", "reply": _clip(reply, 200)})
                if self.finished or self.left() <= 0:
                    return ""
                if reply is not None and again:
                    self.buys += 1
                    self.refused_at = self.step
                    self.observe(buy, reply.get("observation"))
                    return "the checkout was refused: " + self.show(reply.get("observation"))
        if refuse:
            faults = []
            args = _shape(self.quiet(refuse, dict(self.brief.get("refuse_args") or {})), self.schemas.get(refuse), "args", faults)
            if not faults:
                if self.left() >= 2:
                    self.gate("end")
                if self.finished or self.left() <= 0:
                    return ""
                reply = self.act(refuse, args)
                self.trace.append({"kind": "forced_refuse", "reply": _clip(reply, 200)})
                if reply is not None and not self.finished:
                    self.observe(refuse, reply.get("observation"))
                    return "ending without a purchase was refused: " + self.show(reply.get("observation"))
        return ""

    def short_cart(self) -> str:
        """A checkout with fewer cart lines than requested items orders without some item, and a partial order fails the
        request. Held twice at most for an empty cart and once for a part-filled one, so a wrong cart record cannot stall
        the episode. Late, an empty cart is filled from the review's fitting listings instead (no model turn)."""
        if not self.items or not self.roles.get("cart"):
            return ""
        lines = [i for i in self.cart_ids() if not self.sold_out(i) and i not in self.dropped]
        if len(lines) >= len(self.items) or self.short_held >= (1 if lines else 2):
            return ""
        late = self.elapsed() > WRAP_AT or self.left() <= self.tail() + 2
        if late and not lines:
            self.short_held = 2
            if self.fill_empty_cart():
                return ""
        self.short_held += 1
        self.trace.append({"kind": "short_cart", "lines": lines, "items": len(self.items)})
        wanted = "; ".join(f"item {n}: {it.name} x{it.qty}" for n, it in enumerate(self.items, 1))
        head = ("Not run (no step used): the cart is empty, so the order would buy nothing." if not lines else
                f"Not run (no step used): the cart holds {len(lines)} line(s) but the request has {len(self.items)} items ({wanted}).")
        return (head + " Put a fitting listing for each requested item in the cart at its quantity, then check out."
                " If no listing meets some item's requirements, end without a purchase instead."
                + (" If one cart line already covers that item, check out again." if lines else ""))

    def fill_empty_cart(self) -> bool:
        """A forced finish with nothing in the cart: an ending without a purchase is rarely right, so when the review of the
        records names a fitting listing for every item, those lines go in the cart (steps allowing) for the finish to check
        out. Returns whether the cart was filled."""
        cart = self.roles.get("cart")
        fresh = self.cart_view is not None and self.cart_seen_at >= self.cart_at
        if not cart or self.left() < 2 or self.finished or (self.cart_touched and not fresh and self.cart_truth_at < self.cart_at):
            return False
        # stored review picks that a later check turned down left the lead list (check_cart prunes it)
        named = {k: [i for i in ids if i in self.leads] for k, ids in (self.lead_items or {}).items()}
        named = {k: ids for k, ids in named.items() if ids}
        # a model review late in the episode must still leave time for the cart calls and the checkout: past LATE_AT only
        # the light model reads the records (an ending without a purchase is wrong far more often than right)
        if not named and self.cost < SPEND_FINAL and self.elapsed() < FINISH_AT + 6:
            named = self.audit_fits(self.audit_call(None if self.elapsed() < LATE_AT else [LIGHT]))
        options: list[tuple[int, list[str]]] = []
        for number, ids in named.items():
            index = _count(number)
            if not index or not 1 <= index <= len(self.items):
                continue
            if self.brief.get("pick") == "cheapest":
                # the review names fits well but orders them by price unreliably
                ids = sorted(ids, key=lambda i: self.price_of(i) if self.price_of(i) is not None else float("inf"))
            options.append((index, [i for i in ids if not self.sold_out(i) and i not in self.rejected]))
        if len(options) < len(self.items):
            return False
        if self.pair_links():
            # items that must work with another item's pick: the cheapest set by the codes whose every line the review
            # names as fitting goes first
            groups = self.item_rows()
            fits = dict(options)
            for _total, ids in self.pair_sets(6):
                by_item = {n: i for i in ids for n, g in groups.items() if i in g}
                if by_item and all(i in fits.get(n, []) for n, i in by_item.items()):
                    options = [(n, ([by_item[n]] + [i for i in got if i != by_item[n]]) if n in by_item else got) for n, got in options]
                    break
        plan: dict[str, int] = {}
        for index, ids in options:
            # an item whose pick sells out on the add moves on to its next fit
            for pick in ids:
                if pick in plan:
                    continue
                self.note_sold = False
                if not self.put_lines({**plan, pick: self.items[index - 1].qty}):
                    return False
                if self.note_sold and pick not in self.unavailable:
                    # a sell-out note right after adding this one line names it, even without its id
                    self.unavailable.add(pick)
                if not self.sold_out(pick):
                    plan[pick] = self.items[index - 1].qty
                    break
                self.put_lines({}, [pick])
            else:
                return False
        self.trace.append({"kind": "filled", "plan": plan})
        return len(plan) == len(self.items)

    def put_lines(self, lines: dict[str, int], drop: list[str] | None = None) -> bool:
        """Make the cart hold these lines (and none of `drop`) with the code's own cart calls. The cart replies are read
        back, so a tool found to move one unit a call gets a second pass that tops the short lines up."""
        cart = self.roles.get("cart")
        if not cart:
            return False
        for round_ in range(2):
            calls = self.cart_calls(lines, drop or [])
            if calls is None or len(calls) + 1 > self.left():
                return round_ > 0
            for args in calls:
                faults: list[str] = []
                args = _shape(args, self.schemas.get(cart), "args", faults)
                if faults:
                    return False
                before = {i: self.cart_lines.get(i, 1) for i in self.cart_ids()}
                reply = self.act(cart, args)
                if reply is None or self.finished:
                    return False
                self.note_cart(cart, args)
                seen = reply.get("observation")
                self.observe(cart, seen)
                truth = self.read_cart_reply(cart, seen)
                if truth is not None:
                    self.learn_cart_mode(cart, args, before, truth)
            drop = []
            if self.cart_truth_at < self.cart_at or all(self.cart_lines.get(i, 0) == n for i, n in lines.items()):
                break
        return True

    def read_cart_reply(self, name: str, observation: Any) -> dict[str, int] | None:
        """A cart reply or cart view that lists every line replaces Raikou's own record of the cart (the store's word wins)."""
        truth = self.cart_rows_in(name, observation)
        if truth is not None:
            self.cart_lines = dict(truth)
            self.cart_view, self.cart_seen_at = observation, self.step
            self.cart_truth_at = self.step
        return truth

    def short_lines(self) -> dict[str, tuple[int, int, int]]:
        """Cart lines the store listed back after the last cart change that hold another number of units than their item
        asks for: {listing: (item number, units held, units asked)}, counting only items with exactly one line. Empty when
        the cart was not listed back or a line belongs to no requested item (the model check judges those)."""
        if self.cart_truth_at < self.cart_at or not self.items:
            return {}
        cat_key = self.category_key()
        lines = {i: n for i, n in self.cart_lines.items() if not self.sold_out(i)}
        by_item: dict[int, list[str]] = {}
        for ident in lines:
            label = str((self.rows.get(ident) or {}).get(cat_key) or "") if cat_key else ""
            number = 1 if len(self.items) == 1 else self.item_number(label)
            if number is None:
                return {}
            by_item.setdefault(number, []).append(ident)
        return {ids[0]: (n, lines[ids[0]], self.items[n - 1].qty) for n, ids in sorted(by_item.items())
                if len(ids) == 1 and lines[ids[0]] != self.items[n - 1].qty}

    def count_hold(self) -> str:
        """A code-only check before a checkout, once per cart: each item's single line must hold the item's quantity (a cart
        tool that moves one unit a call often leaves it short)."""
        mark = self.cart_mark()
        if self.count_checked == mark:
            return ""
        self.count_checked = mark
        off = self.short_lines()
        if not off:
            return ""
        wrong = [f"{i} holds {have} unit(s) but item {n} ({self.items[n - 1].name}) asks for {want}" for i, (n, have, want) in off.items()]
        self.trace.append({"kind": "count_hold", "wrong": wrong})
        one = self.cart_mode.get(self.roles.get("cart") or "") == "one"
        return ("Not run (no step used): the store's cart shows " + "; ".join(wrong) + ". Set each line to its item's quantity"
                + (" (this cart tool adds one unit per call: call it once for each missing unit; take a line out to start it again)" if one else "")
                + ", then check out. If the quantity in the request means something else (a pack counted as one), check out again.")

    def top_up(self) -> None:
        """Before a forced checkout (no model turn left): lines the store shows short of their item's quantity are set to it
        with the code's own cart calls, steps allowing; the other lines stay as they are."""
        off = self.short_lines()
        if off and self.left() > 2:
            fixed = {**self.cart_lines, **{i: want for i, (_, _, want) in off.items()}}
            self.trace.append({"kind": "top_up", "lines": fixed, "ok": self.put_lines(fixed)})

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
        self.read_request()
        self.trace.append({"kind": "brief", "brief": self.brief})
        opening = self.view.get("opening")
        if opening is not None:
            read = _read_listings(opening)
            opening = _with_raw(read, opening) if read is not opening else opening
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
            if turned and not self.finished and not self.broke and not self.no_credit and self.left() > 1 and self.elapsed() < FINISH_AT + 15:
                found = self.budget_rows() if self.budget_moved else ""
                self.chat.append({"role": "user", "content": "[Raikou] Raikou tried to end the episode and " + turned + ("\n" + found if found else "")
                                  + "\nApply what the store says, then finish.\n" + self.status()})
                self.work(self.loop_calls + 10, min(SPEND_STOP, max(self.cost, SPEND_HARD) + 0.006))
            if not self.finished:
                self.finish()
        self.trace.append({"kind": "end", "finished": self.finished, "steps": self.step, "cost": round(self.cost, 5), "calls": self.calls})
        return self.trace

    def work(self, loop_cap: int, spend_cap: float) -> None:
        idle = 0
        while (not self.finished and not self.broke and self.left() > 0 and self.loop_calls < loop_cap and self.cost < spend_cap
               and self.elapsed() + self.turn_secs() < FINISH_AT):
            if self.chase_due():
                found = self.chase_refs()
                if found:
                    self.chat.append({"role": "user", "content": "[Raikou looked up the listings some requirements point at, its own steps]\n" + found + "\n" + self.status()})
                continue
            if self.pair_due():
                found = self.pair_up()
                if found:
                    self.chat.append({"role": "user", "content": found + "\n" + self.status()})
                continue
            if self.ready_to_ask():
                summary = self.interview()
                if summary:
                    self.chat.append({"role": "user", "content": summary + "\n" + self.status()})
                continue
            self.trim()
            self.loop_calls += 1
            message = self.think(self.chat, [LIGHT, SPARE], 3000, self.offered())
            if message is None:
                self.misses += 1
                if self.no_credit or self.calls >= ALL_CALLS or self.misses > 2 or self.cost >= SPEND_STOP:
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
            if not self.asked and self.roles.get("ask") and self.msgs_free() > 0 and any(c["function"]["name"] in committing for c in calls) and self.left() > 8 and self.elapsed() < WRAP_AT:
                self.chat.append({"role": "assistant", "content": content or None, "tool_calls": calls})
                # the questions are chosen from the attributes of opened listings: with too few opened they would be blind
                if len(self.full) < len(self.items) + 1 and self.early_holds < 2 and self.elapsed() < WRAP_AT - 40:
                    self.early_holds += 1
                    self.trace.append({"kind": "early_hold", "opened": len(self.full)})
                    for call in calls:
                        self.chat.append({"role": "tool", "tool_call_id": call["id"], "content": "Not run (no step used): too few listings are open in full to judge a cart or to choose the shopper's questions."})
                    self.chat.append({"role": "user", "content": "Search for each requested item and open several of its listings in full first: their attributes show which properties to ask the shopper about and which listing fits.\n" + self.status()})
                    continue
                for call in calls:
                    self.chat.append({"role": "tool", "tool_call_id": call["id"], "content": "Not run (no step used): the shopper is asked first."})
                summary = self.interview()
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
        episode = Raikou(problem_data)
    except Exception as exc:
        return [{"kind": "crash", "error": f"{type(exc).__name__}: {exc}"[:400]}]
    try:
        return episode.run()
    except Exception as exc:
        episode.trace.append({"kind": "crash", "error": f"{type(exc).__name__}: {exc}"[:400]})
        try:
            if not episode.finished:
                episode.finish()
        except Exception as again:
            episode.trace.append({"kind": "crash", "error": f"{type(again).__name__}: {again}"[:400]})
        return episode.trace


__all__ = ["agent_main"]
