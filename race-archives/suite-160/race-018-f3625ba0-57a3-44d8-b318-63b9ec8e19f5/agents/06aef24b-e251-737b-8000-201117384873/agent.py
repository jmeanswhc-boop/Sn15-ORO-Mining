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
ALL_CALLS = 60
SPEND_SOFT = 0.036
SPEND_HARD = 0.044
SPEND_STOP = 0.050
BIG_SCALE = 1.25
WAITS = (4, 10, 20, 35)
WAIT_TOTAL = 45
EPISODE_SECONDS = 300.0
TIME_MARGIN = 25.0
CALL_FLOOR = 8.0
INTERVIEW_BY = 150.0
QUESTION_BY = 195.0
EXTRAS_BY = 185.0
WIND_DOWN = 205.0
HOLDS_BY = 235.0
SHOW_CHARS = 450
ROW_CHARS = 200
OBS_LIMIT = 15000
HISTORY_LIMIT = 150000
GATE_ROUNDS = 3
AUTO_PAGES = 8
CHEAPER_OPENS = 6
LIST_SIZE = 1
PLAN_PICKS = 5

_llm = ProxyClient(timeout=60, max_retries=1, retry_delay=1)
_env = ProxyClient(timeout=95, max_retries=3, retry_delay=1)


GUIDE = """You are Orochi, a careful buyer working for one shopper in an online store. You can act only by calling the store's tools. The episode is graded 1 or 0: it earns 1 only when the finish is exactly right. A purchase must contain every requested item at the requested quantity, each meeting every requirement in force when you check out, inside the budget; or, when the store truly has nothing that meets everything, the episode must end with the no-purchase tool in the way the request asks.

HOW THE STORE WORKS
- Every tool call is one step and steps are limited; thinking costs nothing. Tool names, argument names and result shapes are particular to this store: read each tool's description and schema and use the exact argument names and allowed values.
- Search tools can be narrow: a category filter, a keyword mode that may only read the first words of the query, a page size of a few rows, a token or offset for the next page. Compare the reported total with the rows you have seen and page on (with the exact token or offset the store gives) when the answer depends on seeing all of them. Keyword queries work best short and value-first (the wanted value plus the product noun). A filtered search that returns no rows usually means the filter value or the filter parameter is wrong: retry with the label exactly as result rows show it, or with another filter parameter.
- Every word of a keyword query must match the listing's title, attribute values or shop name (descriptions do not count), so a product noun that a fitting title lacks hides the listing. When the search tool takes a category argument, name the category there (the label the rows carry) and put only the value words in the words argument ("stripe", not "stripe jumpsuit"); without a category argument, use the value and the plainest one-word name of the product.
- Search rows are summaries. Stores split a listing's record differently: one view can show its attributes (properties), another only its offer and description, and a side-by-side view may carry either. The ledger names the tool that shows attributes once Orochi knows it. Judge every requirement on a property from those attributes, so read them for each listing you may buy; one call that shows several listings saves steps when it shows what you need. Never open a listing again when nothing has changed since you read it.
- Questions to the shopper are handled for you by Orochi's interviewer, so you never message the shopper yourself. Its answers arrive in the conversation and are added to the ledger.
- The request can also ask for steps along the way: an approval, a message, something to check or read first. The ledger lists them. Orochi carries out the ones due just before the order or the ending itself when you call the checkout or the ending tool, so never wait for those and never end without a purchase because of them; do any others with the store's tools (product questions go through Orochi's interviewer).

WHAT COUNTS AS A FIT
- The kind of product is decided by the listing's category label: only the label that matches the requested item as a whole counts, never a neighbouring one, and a listing filed under that label is that kind of product even when its title reads like an accessory or a bundle sold for it.
- A requirement on a property is met when the listing's attribute for that property states the value. Key spelling does not matter (spaces, underscores, camel case, prefixes), and one element of a comma or slash list is enough. Values match by their words: case, singular or plural and plain synonyms are fine, and a value that contains the required value's words counts ("deep teal" meets teal, "extra roomy fit" meets roomy). A value with different words does not count (an abbreviation or code is not the word it stands for, and "teal green" is not "sea green"), and a value required as a word is never met by a figure or a time that seems to imply it; under a most or least pick that matters most, since a loose reading changes which listing wins. Units are converted before comparing numbers, limits include the edge value, and a range must lie wholly inside a limit.
- An attribute whose key names the requirement's property (under that name or a plain synonym) decides: if it says something else, the listing fails whatever other keys or the title say. A key for a different property, even a related one (a handle colour is not the body's colour), decides nothing about the requirement. Orochi adds [evidence] lines after opened listings with the keys whose names match a requirement.
- A requirement the ledger marks as coming from the shopper's answer is met only by an attribute whose key names that property; the title, the description and other attributes never count for it (race 14: a drink mix whose description promised 500 ml a stick failed a volume need for the 15 agents that trusted it).
- When the listing has no attribute for that property, the wanted value stated under another attribute (for example a list of styles or features that includes it) or in the title or description counts; prefer a listing whose attribute states it when one fits at the same price. A number of another kind does not count (a weight is not a capacity). A bare figure counts when it stands under the attribute that names the property (read it in the request's unit when the attribute gives none); a figure elsewhere whose text does not say what it measures does not.
- A requirement's value is literal: a value like "nil", "none" or "not available" asks for a listing whose attribute states exactly that word, not for a listing without the attribute. A closed list ("from A and B only") is met by either value and nothing else; an exclusion ("not round") is met by any other stated value.
- When the category label itself states a requirement's value (a season or audience in the label), that requirement is met by the category.
- When the ledger marks a requirement as one no listing records, it decides nothing: judge by the other requirements.
- Two keys that are one name written differently (spaces, underscores or case, such as "lid colour" and "lid_colour") can disagree on a listing: prefer a listing whose keys agree; only when no candidate's keys agree is the one that states the wanted value enough. A key with more or other words ("handle material" beside "material") is another key, and the key that names the property exactly decides.
- A requirement can point at another listing instead of giving a value ("the same X as the listing sold by <shop> with <property>", "within N of the X on the page of ..."). Orochi follows such chains itself (it searches for that listing, opens its page in full, follows the pages it refers to for the property, and writes the value into the ledger), so do not search for those pages yourself: work on the requested items meanwhile. Only if the ledger still shows no value for it when you are otherwise ready to buy, find that listing (search its category, brand, shop or words), open its page in full with the single-item or side-by-side tool (a search row is not enough), and when that page refers you to a further listing for the property, find and open that one the same way; the last page gives the value. The order counts only when every page of that chain was opened. When the wanted value is anything but that other listing's value, the rule excludes only listings sharing that exact value, and never end without a purchase because of it alone.
- When a listing you are about to buy sends you to another page for more about it, look at that page before you buy.
- When one requested item must fit another (an accessory for a device), choose the two together from the listings' own words (a list of what it fits, a model or series code that both pages carry). The codes decide it, not the brand or model names in titles: a device's own text gives its model number (and other names it is sold as), and an accessory fits that device when its fits list names that number or one of those names, or names a family that covers it (a family written with x for any digit covers every number of that form) and does not except it. When no accessory that meets its own requirements fits the device you picked, switch to another device that meets its requirements and that such an accessory fits; never end without a purchase while that switch is untried. Only the items the request ties to another item must fit it; any other item only has to meet its own requirements.
- A requirement with a condition on another requested item (it applies only if, unless or except when something holds for the listing bought for that other item) applies only when that listing meets the condition: choose that item first, read the condition on its record, then apply the requirement or drop it.
- Respect exclusions and closed lists. Wishes ("ideally", "would be nice") are tie-breakers only.
- Never buy an item that the store shows as sold out; the total (price times quantity, summed over every line) may not exceed the budget; equal is fine.

CHOOSING
- When the request asks for the cheapest (lowest total, smallest price), the order must hold the lowest-priced fitting listing among every listing that has appeared in your results, rows included: a cheapest order is held against each cheaper fitting listing you have seen, opened or not, and never against ones you have not seen. So read candidates in ascending price (Orochi does this itself for a single item after the interview) and take the first that fully fits; before checking out, open any cheaper row of that kind that could fit and rule it out; once the pick is settled, run no new broad search of that kind. When a search takes a maximum price, use it: list the category below the price of your current best pick, so only cheaper candidates are left to check.
- When requested items must fit each other (a model, series or part code that both pages carry) and the pick is the cheapest, compare whole sets: a set built around another model or series can cost less in total, and the order must be the cheapest complete set that fits.
- When the request asks for the most or least of some measure, compare every fitting listing on that measure as its attribute states it, one figure in one unit (converted when units differ), and take the extreme; a listing that does not state the measure cannot win, and a value made of several figures (dimensions such as "500 x 350 mm") is not that measure.
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

"items": a list with one entry per distinct product the shopper wants bought, each {"name": the kind of product in the request's words, "shelf": the store's category label for it when a tool lists labels (copy it exactly), else "", "quantity": units of that one listing (integer, 1 when not stated; a pair or a set that is one listing counts as 1), "musts": a list of firm requirements on that product, each "property: constraint" with the exact value, limit, exclusion or closed list. Name the property with the request's own word for it ("meant for a lakeside occasion" gives "occasion: lakeside", "a 12-inch blade" gives "blade length: 12 inch"); do not swap it for a neighbouring property. When the request says how the listing itself writes the value, often quoting it, the must takes that written form exactly, even when it looks misspelled, and adds "(spelled so on the listing)". A requirement that holds only under a condition on another requested item (a strap that must be leather unless the watch bought is a diving model; a lid that has to be glass if the kettle chosen turns out to hold over two litres) is ONE must that keeps its condition in plain words after the value ("strap material: leather, unless the watch bought is a diving model"); never drop the condition and never state the value alone. A must whose value is read from another listing is ONE must holding the whole description of that listing ("rim colour: unlike the rim colour of the kettle Ovarno sells with a glass lid"): words that describe that other listing (its shop, its kind, "with <property> <value> and <property> <value>" after it) stay inside that one must and are never musts of this product}. Never put price, budget, stock or quantity into musts. The story around the request (the person meant to use it, where it will stand, what went wrong before, the occasion behind it) is context, not a must, unless it states a property's value or limit outright; a sentence saying only that some property matters or counts ("how it does up matters", "where it sits on me counts", "the band matters") names no value and is never a must (the interviewer asks about it); and never turn words that are part of what the product is called (a season, an audience or a use named with the product) into musts: those belong to "name". A product the request names in one phrase joined by "and" or "&" (one "I want ..." with one quantity, often the name of one store department) is ONE item; make separate items only when the request gives each product its own quantity or its own requirements.
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
"ask_first": a list of things the request explicitly says to check with the shopper before choosing (empty when none). A property whose value the shopper says they will give when asked ("she'd tell me which cut she fancies once I got to that bit", "ask me which range of cuts to look through") goes here as that property and never into musts: it is not a requirement until the shopper answers.
"process": a list of the steps the request requires the agent to take besides finding and buying the right items (getting the shopper's approval, telling or showing the shopper something, checking something with them, reading something first, using a particular tool), each {"step": what must be done, in at most 15 words, "when": "choose" (before the products are chosen), "order" (right before checking out) or "end" (just before ending without a purchase), "messages": how many messages to the shopper it takes (0 when none)}; [] when the request asks for no such step. Requirements on the items, the budget, stock, the limits on messages and steps, and how to end when nothing fits are not steps.
"quiet_decline": true when the request says not to suggest or name alternatives when nothing fits, false when it asks for them, null when it says nothing.
"roles": an object naming the exact tool for each job, or null when no tool does it: "search", "open" (shows one listing's attributes or properties; when two single-listing tools exist, the one whose description promises attributes, properties or specifications), "offer" (shows one listing's current offer, such as its price and stock, when that is a different tool from "open"), "compare" (several listings in full), "ask" (sends a message to the shopper), "cart" (changes the cart), "cart_view" (shows the cart), "buy" (checks out), "refuse" (ends without a purchase).
"search_words_arg": the exact parameter of the "search" tool that takes the free search words, or null.
"buy_args": the fixed arguments that make the "buy" tool check out (for a tool with a mode argument, that mode), else {}.
"refuse_args": the fixed arguments that make the "refuse" tool end without a purchase, else {}.
"""


PLAN = """You rank the properties a shopping agent will ask a shopper about. Shoppers hold a few firm preferences they did not write down, each about one property of one requested product. Each message asks about ONE property of one item, and only a few messages are allowed: the shopper states the preference when the property asked is one of them, and otherwise answers that they do not mind or that they do not follow, so the property chosen first for each item decides what comes out.

You get the request, the items with what the request already settles for each, and a numbered list of attribute keys seen on opened listings of each item's category, with how many of those listings carry the key and example values. An unwritten preference may be on a property that no listed key names.

Rank the properties of each item in this order:
0. What the request itself asks you to check with the shopper (listed under ASK FIRST when there is any), each named by the property it is about.
1. Hinted properties. The request's story often points at an unwritten preference without giving its value: a sentence about something the shopper cannot judge or leaves to you, someone whose taste or build decides, how the product will look to others, the setting it belongs in, the person meant to use or wear it, what is inside it, how it is powered or connected, or how it will be read or heard. Read each such sentence as pointing at one property of one item: a remark about how a garment will look to other people points at its pattern or colour; about which parts the inside of a machine takes, at the type of those parts; about telling machines apart by the numbers in their specs, at the chip or model line; about a person's build or the cut that suits them, at the size class or the fit; about plugging straight into the wall, at the plug type; about reading a screen from a distance, at its screen size; about who will wear it, at gender; about reading the label, at the material; about what is inside a tin or bottle, at its contents. Name the property the item's listings would call it.
2. Then the listed keys, ranked: favour the properties a buyer of that kind of product chooses between first (often its type or kind, its main measure, its material or its colour, whichever that kind of product is bought by) and keys carried by most of the item's listings; avoid keys seen on only one or two of them.
Skip a property the request already settles for that item, a second name for a property already chosen, a key that lists purchase variants (a mix of sizes, colours and pack counts), identifiers, warranty and other logistics. With several items, rank each item's properties separately, hinted ones first.
"label" is the property's plain name: for a listed key, all of the key's words in plain spelling ("lid colour" for lidColour, "rim finish" for rim_finish; never shortened to "colour" or "finish"); only the property's name, no example values, no brackets, no second property.
Give up to COUNT per item, best first. Return ONE JSON object and nothing else: {"ask": [{"item": item number, "label": "...", "hint": "at most 8 of the story's words it comes from, or an empty string"}]}"""


HEAR = """A shopping agent asked a shopper some questions. Read each answer and return ONE JSON object and nothing else:
{"needs": [{"item": item number, "need": "property: constraint", "points": true when the value is to be read from another listing, else false}], "drop": [{"item": item number, "property": "a requirement the shopper now gives up"}], "free": [{"item": item number, "property": "..."}], "crowded": [numbers of the questions whose answer asks for one property or one question at a time, or says the message asked about too many things; an answer that only says it did not follow, did not hear, or wants plainer or shorter words is NOT crowded], "budget": new total limit as a number or null}
- "needs": every firm preference an answer states, with the exact value or limit and its unit ("at least 30 cm", "oval", "not wicker"). Name the need by the property the answer itself names, which may differ from the one asked about; several answers about one property make one need. Copy the value word for word as the answer says it, keeping its nouns and units ("at least 3 shelves", not "at least 3"). When the answer names the item the preference is about, the need belongs to that item even when the question was about another.
- An answer can give the value by pointing at another listing (a value like, unlike or near the one on some shop's product, that product picked out by a few of its properties): that is ONE need on the property the answer names, with "points": true. Its value is the answer's own words copied exactly, from the comparison words to the end of that description, including what it says about a listing that page may name; never shorten or reword it. The copy starts at the comparison itself, never at the words that join the property to it. The properties describing the other product are not needs of this item.
- "free": the property of a question about ONE property that the shopper answered with no preference; a question that lists several properties and gets a no-preference or a puzzled answer gives no "free" entry (the shopper answered only about one of them, unnamed).
- Folksy or roundabout wording that names a value still states that value, and the need's value is the property's value alone, stripped of the idiom around it ("nowt but oak will do for the material" states material: oak; "anything other than oak gets a no from me" states material: oak; "I'm holding out for oak" states material: oak); a puzzled or dismissive answer states nothing."""


NOTE = """While a shopping agent works, the store showed this note (it may come from the shopper or the store). Return ONE JSON object and nothing else:
{"kind": "change" when the note changes the task (budget, requirements, a price, availability, a refusal of the order), "error" when it only reports that a call was malformed or not accepted, "other" for anything else, "budget": the new total limit as a number in the request's unit, or null, "add": [{"item": item number, "need": "property: constraint"}], "drop": [{"item": item number, "need": "the requirement no longer wanted"}], "gone": [listing ids said to be unavailable], "sold_out": true when it says a chosen listing is gone, sold out or no longer available (even without naming it), "price_moved": true when it says a price changed, "summary": "one short sentence"}
A replaced requirement is a drop of the old one plus an add of the new one. Report every change in the note."""


CHECK = """You are the last check before a shopping agent checks out. Decide from the records shown whether this cart is exactly right.
Reading rules: a requirement marked as coming from the shopper's answer is met only when an attribute whose key names that property states the value (the title, the description and other attributes do not count for it); a requirement is met when the listing's attribute for that property states the value (any key spelling, one element of a list is enough, case, singular/plural and plain synonyms count, a value containing the required value's words counts, such as "deep teal" for teal, whatever the added words, even when they name a variant or a smaller form of it, units converted, limits include the edge, a range must lie wholly inside); a value required as a word is met only by an attribute stating that word or a plain synonym of it, never by a figure or a time that seems to imply it; an attribute whose key names the same property as the requirement decides, and if it says something else the listing fails, but two keys that are one name written differently (spaces, underscores or case) may disagree on a listing, and then the one that states the wanted value is enough; a key for a different property, even a related one (a handle colour is not the body's colour), decides nothing; when the listing has no attribute for that property, the wanted value stated under another attribute or in the title or description counts (a value like "nil" or "none" is literal and the attribute must state it; a number of another kind does not count: a weight is not a capacity; a bare figure counts under the attribute that names the property, in the request's unit when the attribute gives none, and nowhere else); a requirement the category label itself states is met by that category; a requirement marked as one no listing records decides nothing; the category label must match the requested item as a whole, and a listing filed under that label counts as that kind of product even when its title reads like an accessory or a bundle sold for it; a rule of being unlike another listing's value counts as met unless the records show that same value. Each requested item must appear as a separate cart line at the requested quantity, nothing else may be in the cart, nothing shown out of stock, and the total of price times quantity over all lines must be within the budget. The newest record of a listing wins.
When the pick rule is cheapest, most or least, also look at the other listings shown: if one of them clearly fits every requirement and beats the cart's choice on that rule, the cart is wrong. When the items must fit each other, a complete set built around another model or series that fits and costs less in total also makes a cheapest cart wrong.
Return ONE JSON object and nothing else: {"ok": true or false, "problems": ["short, concrete problem", ...], "better": [listing ids that should replace a cart line, best first]}. Use ok=false only for a problem the records clearly show or a requirement no record addresses."""


READING = """a requirement marked as coming from the shopper's answer is met only when an attribute whose key names that property states the value (the title, the description and other attributes do not count for it); the listing's attribute for a property decides when its key names that same property (any key spelling, one element of a list is enough, synonyms and unit conversions count, a value containing the required value's words counts, such as "deep teal" for teal, whatever the added words, even when they name a variant or a smaller form of it, limits include the edge); a value required as a word is met only by an attribute stating that word or a plain synonym of it, never by a figure or a time that seems to imply it; a key for a different property, even a related one (a handle colour is not the body's colour), decides nothing; when no attribute names that property, the wanted value stated under another attribute or in the title or description counts; a value like "nil" or "none" is literal and must be stated by the attribute; a bare figure counts under the attribute that names the property (in the request's unit when the attribute gives none) and nowhere else; a requirement the category label itself states is met by that category; a requirement marked as one no listing records decides nothing; the category label must match the item, and a listing filed under it counts as that kind of product even when its title reads like an accessory or a bundle sold for it; a rule of being unlike another listing's value counts as met unless the records show that same value; out-of-stock listings and listings that break the budget do not fit."""

AUDIT = """A shopping agent wants to end without buying because it thinks nothing fits. Check the listing records shown against the requirements, with these reading rules: """ + READING + """
Two keys that are one name written differently (spaces, underscores or case) may disagree on a listing: then the one that states the wanted value is enough.
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

FIT = """A shopping agent needs one listing for the requested item below. Using only the listing records shown (their attributes were read), name every listing that meets every requirement of that item; a requirement with a condition on another item (only if, unless or except when something holds for it) applies only when the listing chosen for that other item, shown at the end, meets its condition, is void when that listing does not, and applies when no listing is shown for that item; with these reading rules: """ + READING.replace(
    "out-of-stock listings and listings that break the budget do not fit.",
    "out-of-stock listings do not fit; a price cap the request sets for this item alone is one of its requirements, but a budget for the whole order is left out here (it is checked later, for the whole order).") + """
A requirement's value is literal: "not available", "none" or "no" asks for a listing whose attribute states exactly that, not for any value. An item named with alternatives ("A or B") is met by a listing of either kind.
Return ONE JSON object and nothing else: {"fits": [ids of the listings that meet every requirement, best first], "reason": "one short sentence"}"""

SHAPE = """A store tool returned the result below. Name its parts by the exact keys the result and the tool's parameters use. Return ONE JSON object and nothing else: {"rows": the key of the list of listings, "count": the key holding how many listings matched in all (not how many are shown), or null, "next": the key holding the value that asks for the following page, or null, "next_arg": the tool parameter that takes that value, or null, "price": the key, inside a row, of the listing's price, or null, "stock": the key, inside a row, that says whether the listing can be bought now, or null}"""


START = """A shopper's requirement takes its value from another listing that it describes. Return ONE JSON object and nothing else: {"property": the requirement's own property, "relation": how the wanted value relates to that listing's value ("same", "different", "at least", "at most", "within N of", "goes with", or a few words), "seller": the exact name of the shop that sells that listing (shops have made-up names; the name can come after the kind, as in "the kettle at Ovarno" or "Ovarno's kettle", or right before it, as in "the Ovarno kettle page" or "the Ovarno kettles listing"), "kind": its kind of product, "features": [the values the description gives for that listing's properties (after words such as "has", "with" or "that comes with"), each value alone without its property name ("silver" for "color silver", "64 gb" for "storage 64 gb"), up to 4; never empty when the description names any]}"""

JUDGE = """A shopping agent follows a chain of listing pages to read one property's value: PROPERTY. It opened the page below as the KIND from SELLER with FEATURES. Return ONE JSON object and nothing else:
{"shop": the page's shop as the page names it,
"contradicts": words from the page's attributes or from the store's short notes about the listing (fields such as fine print, notes or remarks) that give a different value for one of the named values or say it is not one of them ("this is not the silver one", "a 0.8 kg version exists elsewhere" when 0.8 kg is a named value), quoted briefly; "" when none do (the title and the long sales description never count: they are the seller's copied text),
"confirmed": true when the page's attributes or the store's short notes about it positively state the named values (a value may be stated in other units or as a product, "16 x 500 gb" for 8000 gb, or as a bare figure under the property's own key), false when they do not,
"next": when the page sends the reader to another listing for PROPERTY itself: {"seller": the exact name of the shop that sells it (shops have made-up names, often standing right before the kind, as in "the Ovarno kettle"), "kind": its kind of product, "features": [the values the page gives for that listing's properties, each value alone without its property name ("silver" for "color silver"), up to 4]}; null when it does not (a pointer about any other property does not count),
"value": the page's own value for PROPERTY under the attribute whose key names PROPERTY itself (a key with fewer or more words is another property: "memory" is not "memory type", "size" is not "size class"), or the page's own words about PROPERTY when no attribute names it, else "",
"rule": a pairing rule for PROPERTY that the page states, that is a list saying which value of PROPERTY goes with which other value of it, copied as written; else ""}"""

SETTLE = """A shopper's requirement takes its value from another listing. That listing's own value has been read. Return ONE JSON object and nothing else: {"constraint": what the requirement now asks of the product being bought, as "property: constraint" in plain words, built from THAT LISTING'S OWN VALUE given below and nothing else: the value itself, word for word, when the requirement wants the same value; "not <value>" when it wants a different one; "at least <value>" or "at most <value>" when it sets a limit from it; "between <value minus N> and <value plus N>" (with units) when it allows a margin of N around it; when it wants something that goes with that listing, the value the stated pairing rule matches with it (pairings work both ways), or the value itself when no rule is stated}"""


REFS = """A shopper's request is below. Some of its requirements may take a product property's value from another listing's page instead of stating it. List every such requirement. Return ONE JSON object and nothing else: {"refs": [{"item": the item number from ITEMS, "need": "property: the request's own words for the requirement, from the comparison word to the end of its description of that other listing"}]}; {"refs": []} when there is none."""


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


def _shop(name: str) -> str:
    return re.sub(r"(?i)['’]s\s+\S.*$", "", name.strip()).strip()


def _alike(a: str, b: str) -> bool:
    return len(a) >= 4 and len(b) >= 4 and a[0] == b[0] and abs(len(a) - len(b)) <= 2 and difflib.SequenceMatcher(None, a, b).ratio() >= 0.75


def _stem(text: str) -> frozenset[str]:
    return frozenset(w[:-1] if len(w) > 3 and w.endswith("s") else w for w in _words(text).split())


_LABEL_STOP = frozenset({"and", "or", "the", "for", "with", "of", "a", "an", "in", "on", "to"})


def _label_words(text: str) -> frozenset[str]:
    return frozenset((w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w)
                     for w in re.findall(r"[a-z0-9]+", (text or "").lower()) if w not in _LABEL_STOP)


def _names(prop: frozenset[str], key: frozenset[str]) -> bool:
    if not prop or not key:
        return False
    return prop == key or prop <= key or (len(key) >= 2 and key <= prop)


_POINTER = re.compile(r"(?i)\bpage (?:of|for)\b|\w's listing\b|\bthe listing for\b")
_DETAIL_KEY = re.compile(r"(?i)\b(?:ids?|sku|skus|asin|upc|ean|barcodes?|models?|codes?|options?|variants?|urls?|links?|images?|photos?|videos?|warrant\w*)\b")
def _key_name(name: str) -> str:
    words = name.strip().split()
    while len(words) > 1 and len(words[-1]) <= 3 and not re.search(r"[aeiouy0-9]", words[-1]):
        words.pop()
    if not words or any(len(w) >= 4 and not re.search(r"[aeiouy0-9]", w) for w in words):
        return ""
    return " ".join(words)


def _dedupe_labels(labels: list[str]) -> list[str]:
    out: list[str] = []
    for label in labels:
        if label and not any(_words(label) == _words(other) or _same_prop(label, other) for other in out):
            out.append(label)
    return out


_CONDITION = re.compile(r"(?i)\b(?:only if|unless|except (?:when|if)|should\b.{0,80}\bturns? out|if (?:the|it|its)\b)")
_MIXED_KEY = re.compile(r"(?i)^(?!.*\b(?:types?|styles?|kinds?|shapes?)\b).*\b(?:features?|functions?|highlights?|specs?|specifications?|descriptions?|includ\w*|accessor\w*|extras?|misc\w*)\b")
_CHOSEN = re.compile(r"(?i)\b(?:chos(?:en|e)|choos\w*|pick\w*|select\w*|ends? up|ending up|i (?:get|buy|order)|in (?:this|the same) order|(?:being |you )?(?:bought|buying|buy|purchased?)|you get)\b")


_SERIES = r"[A-Z]{1,4}\d{1,3}"
_SERIES_LIST = re.compile(r"(?i)\b(?:fits|takes|accepts|compatible|works with)\b[^.;:]{0,25}?\bseries\b\s*:?\s*(" + _SERIES + r"(?:(?:\s*(?:,|and|or|/|&)\s*)+" + _SERIES + r")*)")
_SERIES_CODE = re.compile(r"\[\s*([A-Z]{1,4}\d{1,3})-\d{2,8}")
_SEE_PART = re.compile(r"(?i)\b(?:see|more (?:on|in)|rest (?:on|in))\s+part\s+([A-Z]{1,4}\d{1,3}-\d{2,8})")
_CODE_BODY = r"(?:(?i:[a-z]{2,5}-?\d{2,6})|[A-Z]{2,5} \d{2,6})(?![\w-])"
_MODEL_CODE = re.compile(r"(?i:\bmodel(?:\s+(?:no|ref|number)\.?)?\s*:?\s*)(" + _CODE_BODY + r")")
_ALSO_SERIES = re.compile(r"(?i:\bmodel\s+)(" + _CODE_BODY + r")(?i:\s+also\s+(?:takes|fits|accepts)\s+(?:part\s+)?series\s*:?\s*)((?i:" + _SERIES + r"(?:(?:\s*(?:,|and|or|/|&)\s*)+" + _SERIES + r")*))")

_ALSO_SOLD = re.compile(r"(?i)\b(?:also|or) (?:sold|known|listed) as\s+([^.;]{2,40})")
_WORKS_WITH = re.compile(r"(?i)\b(?:works with|goes with|also fits|compatible with|fits\s*:)\s*:?\s*([^.;]{2,160})")
_MORE_PART = re.compile(r"(?i)\b(?:(?:more|details|rest) (?:on|in|about|for) part\s+([A-Z]{1,4}-?\d{3,8})|see part\s+([A-Z]{1,4}\d{3,8})\s+for the rest)")
_EXCEPT = re.compile(r"(?i)\b(?:except(?:ing)?|excluding|but not|other than|apart from)\s+([^.;]{2,80})")
_FAMILY = re.compile(r"(?i)\b([A-Z]{2,5})[- ]?(\d{1,4}x{1,3})((?:\s*/\s*\d{1,4}x{1,3})*)\b")


def _families(text: str) -> set[str]:
    out: set[str] = set()
    for match in _FAMILY.finditer(text):
        prefix = re.escape(match.group(1).lower())
        for tail in [match.group(2)] + re.findall(r"\d{1,4}x{1,3}", match.group(3), re.I):
            out.add(prefix + re.sub(r"(?i)x", r"\\d", tail.lower()))
    return out


def _more_code(text: str) -> str:
    match = _MORE_PART.search(text)
    return (match.group(1) or match.group(2) or "").upper() if match else ""

def _singular(word: str) -> str:
    low = word.lower()
    if len(low) <= 3 or not low.endswith("s") or low.endswith("ss"):
        return word
    if low.endswith("ves"):
        return word[:-3] + "f"
    if low.endswith("ies"):
        return word[:-3] + "y"
    if re.search(r"(?:s|x|z|ch|sh)es$", low):
        return word[:-2]
    return word[:-1]


_LIMIT = re.compile(r"(?i)\b(at least|no less than|minimum of|at most|no more than|up to|maximum of|under)\s+(\d[\d.,]*)\s*([a-z\"]{0,6})")


def _same_unit(a: str, b: str) -> bool:
    a, b = a.strip().lower().rstrip("."), b.strip().lower().rstrip(".")
    return a == b or a + "s" == b or b + "s" == a or a + "es" == b or b + "es" == a


def _within(text: str, limit: Any) -> bool:
    unit = limit.group(3) or ""
    try:
        bound = float(limit.group(2).replace(",", ""))
    except ValueError:
        return False
    low = bool(re.match(r"(?i)at least|no less|minimum", limit.group(1)))
    for figure, mark in re.findall(r"(\d[\d.,]*)\s*([a-z\"]{0,6})(?![a-z])", text.lower()):
        if not _same_unit(mark, unit):
            continue
        try:
            value = float(figure.replace(",", ""))
        except ValueError:
            continue
        if (value >= bound) if low else (value <= bound):
            return True
    return False


def _core(name: str) -> frozenset[str]:
    return frozenset(w for w in _stem(name) if w not in ("of", "the", "a", "an", "for", "and", "type"))


def _prop_key(label: str) -> frozenset[str]:
    return frozenset(w for w in _core(label) if len(w) >= 3 and w != "filter")


def _same_prop(a: str, b: str) -> bool:
    x, y = _prop_key(a), _prop_key(b)
    return bool(x) and bool(y) and (x == y or x <= y or y <= x)


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
    if not isinstance(value, str) or not 3 <= len(value.strip()) <= 64 or " " in value.strip():
        return False
    return named or bool(re.search(r"\d", value))


def _learn_id(rows: list[dict[str, Any]]) -> str | None:
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
    if depth > 5:
        return False
    if isinstance(node, list):
        return any(isinstance(v, dict) and ident in [str(x).strip() for x in v.values() if isinstance(x, str)] for v in node) or any(_in_rows(v, ident, depth + 1) for v in node)
    if isinstance(node, dict):
        return any(_in_rows(v, ident, depth + 1) for v in node.values())
    return False


_FLAT_KEY = r"[A-Za-z_][A-Za-z0-9_ ]{0,30}"
_FLAT_SPLIT = re.compile(r"\s+\|\s+(?=" + _FLAT_KEY + r":(?:\s|$))")
_FLAT_HEAD = re.compile(r"^\s*(" + _FLAT_KEY + r"):(?:\s|$)")
_PAIR_SPLIT = re.compile(r";\s*(?=[^=;]{1,40}=)")


def _flat_value(text: str) -> Any:
    value = text.strip()
    if value in ("None", "null", ""):
        return None
    if value in ("True", "true"):
        return True
    if value in ("False", "false"):
        return False
    if re.fullmatch(r"-?\d+", value):
        return int(value)
    if re.fullmatch(r"-?\d+\.\d+", value):
        return float(value)
    pairs = _PAIR_SPLIT.split(value)
    if len(pairs) >= 2 and all("=" in p for p in pairs):
        out: dict[str, Any] = {}
        for pair in pairs:
            key, _, val = pair.partition("=")
            if key.strip():
                out[key.strip()] = val.strip()
        return out
    return value


def _unflatten(node: Any, depth: int = 0) -> Any:
    if depth > 8:
        return node
    if isinstance(node, dict):
        return {k: _unflatten(v, depth + 1) for k, v in node.items()}
    if isinstance(node, list):
        return [_unflatten(v, depth + 1) for v in node]
    if isinstance(node, str) and len(node) < 400000 and node.strip()[:1] in ("{", "[") and node.strip()[-1:] in ("}", "]"):
        try:
            parsed = json.loads(node)
        except ValueError:
            parsed = None
        if isinstance(parsed, (dict, list)):
            return _unflatten(parsed, depth + 1)
    lined = False
    if isinstance(node, str) and "\n" in node and node.count(" | ") < 3:
        lines = [l for l in node.strip().splitlines() if l.strip()]
        if len(lines) >= 3 and all(_FLAT_HEAD.match(l) for l in lines):
            node = " | ".join(l.strip() for l in lines)
            lined = True
    if isinstance(node, str) and (node.count(" | ") >= 3 or lined) and _FLAT_HEAD.match(node):
        parts = _FLAT_SPLIT.split(node.strip())
        record: dict[str, Any] = {}
        for part in parts:
            head = _FLAT_HEAD.match(part)
            if not head:
                return node
            key = head.group(1).strip()
            record[key] = _flat_value(part[head.end():])
        if len(record) >= 3:
            return record
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


class Item:
    def __init__(self, name: str, shelf: str, qty: int, musts: list[str]):
        self.name = name
        self.shelf = shelf
        self.qty = qty
        self.musts = musts
        self.told: list[str] = []
        self.dropped: list[str] = []
        self.free: list[str] = []


class Orochi:
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
        self.waited = 0
        self.misses = 0
        self.cost = 0.0
        self.calls = 0
        self.call_secs: dict[str, list[float]] = {}
        self.chosen: dict[int, str] = {}
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
        self.cart_empty = False
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
        self.id_args: set[tuple[str, str]] = set()
        self.cart_said = ""
        self.refused_at = -1
        self.leads: list[str] = []
        self.refuse_blocks = 0
        self.pushed = False
        self.terms_eased = False
        self.eased_block = False
        self.price_moved = False
        self.fixed_codes: dict[int, set[str]] = {}
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
        self.bargain_done = False
        self.best_plan: dict[int, str] = {}
        self.bargained: set[int] = set()
        self.shelf_counts: dict[str, int] = {}
        self.set_soft = False
        self.ref_constraint: dict[str, str] = {}
        self.paged_obs: list[Any] = []
        self.chased: set[str] = set()
        self.last_bought = False
        self.search_template: tuple[str, dict[str, Any]] | None = None
        self.template_extra = False
        self.search_ok: list[tuple[frozenset[str], dict[str, Any]]] = []
        self.search_bad: set[tuple[frozenset[str], str]] = set()
        self.searched = False
        self.found_by: dict[str, dict[str, Any]] = {}
        self.refilled = False
        self.t0 = time.time()
        self.audit_calls = 0
        self.chase_from = 0
        self.chase_rows: dict[str, list[str]] = {}
        self.rescued = False
        self.set_tries = 0
        self.order_rules: dict[str, int] = {}
        self.set_pages: dict[str, tuple[str, str]] = {}
        self.set_queries: set[str] = set()
        self.set_fit: tuple[str, str, set[str]] | None = None
        self.set_ext: dict[str, set[str]] = {}
        self.compat_holds = 0
        self.line_holds = 0
        self.fetched: set[str] = set()
        self.fit_cache: dict[tuple[int, tuple[str, ...], str], list[str]] = {}
        self.soft, self.hard, self.stop = SPEND_SOFT, SPEND_HARD, SPEND_STOP
        self.row_fields: set[str] = set()
        self.last_search_text = ""
        self.queries: list[str] = []

    def left(self) -> int:
        return self.max_steps - self.step

    def msgs_free(self) -> int:
        return self.msg_limit - self.msg_used - max(0, self.reserve - self.gate_msgs)

    def cart_mark(self) -> str:
        return _dump(sorted(self.cart_lines.items()))

    def page_tools(self) -> set[str]:
        page = self.roles.get("offer") or self.roles.get("open")
        return {t for t in (page, self.roles.get("compare")) if t}

    def facts_tool(self) -> str | None:
        known = next((t for t, ok in self.facts_from.items() if ok and self.single_target(t)), None)
        if known:
            return known
        tool = self.roles.get("open")
        return tool if self.facts_from.get(tool or "") is not False and self.single_target(tool) else None

    def compare_facts(self) -> bool | None:
        return self.facts_from.get(self.roles.get("compare") or "")

    def compare_bare(self) -> bool:
        seen = self.compare_facts()
        return seen is False or (seen is None and (self.facts_from.get(self.facts_tool() or "") is True or bool(self.roles.get("offer"))))

    def gate_due(self, stage: str) -> bool:
        return any(p["when"] == stage for p in self.process) and (stage, self.cart_mark() + f"#{self.version}") not in self.gate_ok

    def tail(self) -> int:
        talk = any(p["when"] == "order" and p["messages"] > 0 for p in self.process)
        return 2 if talk and self.msg_used < self.msg_limit and self.gate_due("order") else 1

    def elapsed(self) -> float:
        return time.time() - self.t0

    def room(self) -> float:
        return EPISODE_SECONDS - TIME_MARGIN - self.elapsed()

    def think(self, messages: list[dict[str, Any]], models: list[str], max_tokens: int, tools: list[dict[str, Any]] | None = None) -> dict[str, Any] | None:
        for model in models:
            if self.calls >= ALL_CALLS or self.cost >= self.stop:
                return None
            if self.room() < CALL_FLOOR:
                self.trace.append({"kind": "late_call_skipped", "at": round(self.elapsed(), 1)})
                return None
            if model != LIGHT and (self.cost >= self.hard or self.cost + self.guess(messages) > self.stop or self.room() < 45):
                continue
            if self.room() < self.call_estimate(model):
                if model != LIGHT:
                    continue
                self.trace.append({"kind": "late_call_skipped", "at": round(self.elapsed(), 1), "estimate": round(self.call_estimate(model), 1)})
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
                if self.room() - WAITS[tries] * 1.2 < CALL_FLOOR + 5:
                    break
                pause = WAITS[tries] * (0.8 + 0.4 * random.random())
                self.waited += WAITS[tries]
                tries += 1
                time.sleep(pause)
            if status == 402:
                self.no_credit = True
                return None
        return None

    def call_estimate(self, model: str) -> float:
        seen = self.call_secs.get(model) or []
        base = max(seen[-4:]) * 1.1 if seen else (10.0 if model == LIGHT else 25.0)
        return max(CALL_FLOOR, base)

    @staticmethod
    def guess(messages: list[dict[str, Any]]) -> float:
        return sum(len(_dump(m.get("content"))) for m in messages) / 3.3 * 2e-6 + 0.02

    def call_model(self, model: str, messages: list[dict[str, Any]], max_tokens: int, tools: list[dict[str, Any]] | None) -> tuple[dict[str, Any] | None, Any]:
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
        _llm.timeout = max(CALL_FLOOR, min(60.0, self.room() - 2.0))
        began = time.time()
        result = _llm.post_verbose("/inference/chat/completions", body)
        self.call_secs.setdefault(model, []).append(time.time() - began)
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
            if self.broke or self.no_credit or self.calls >= ALL_CALLS or self.cost >= self.stop:
                return None
        return None

    def act(self, name: str, args: dict[str, Any]) -> dict[str, Any] | None:
        payload = dict(self.binding)
        payload["step"] = self.step + 1
        payload["action"] = {"name": name, "args": args}
        for _ in range(4):
            _env.timeout = max(10.0, min(95.0, self.room() + TIME_MARGIN - 5.0))
            result = _env.post_verbose("/environment/call", payload)
            if isinstance(result.data, dict):
                data = result.data
                if "observation" in data:
                    data["observation"] = _unflatten(data["observation"])
                self.step = data["step"] if isinstance(data.get("step"), int) else self.step + 1
                if data.get("finished"):
                    self.finished = True
                seen = data.get("observation")
                if name in self.page_tools() and not (isinstance(seen, dict) and seen.get("error")):
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

    def read_request(self) -> None:
        lines = []
        for tool in self.tools:
            fn = tool["function"]
            lines.append(f"- {fn['name']}: {fn.get('description', '')} | parameters: {_dump(fn.get('parameters') or {})}")
        prompt = BRIEF + "\nTOOLS:\n" + "\n".join(lines) + "\n\nREQUEST:\n" + self.request
        if self.view.get("opening") is not None:
            prompt += "\n\nOPENING NOTE FROM THE STORE:\n" + _dump(_clip(self.view.get("opening"), 1500))
        brief = self.ask_json(prompt, [LIGHT], 6000) or {}
        if not self.no_credit and (not _items(brief.get("items")) or not isinstance(brief.get("roles"), dict) or not brief["roles"].get("buy")):
            brief = self.ask_json(prompt, [HEAVY, LIGHT], 5000) or brief
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
            fixed = self.ask_json(ask, [LIGHT, HEAVY], 1500) or {}
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
                key = _words(text.split(":")[0])
                text = next((m for item in self.items for m in item.musts if _words(m.split(":")[0]) == key), text)
            number = next((n for n, item in enumerate(self.items, 1) if text in item.musts), 1)
            if text and (number, text) not in self.refs:
                self.refs.append((number, text))
        self.find_refs()
        self.dedupe_pointers()
        self.fixed_code_musts()
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
            talk = [p["messages"] for p in self.process if p["when"] in ("order", "end") and p["messages"]]
            self.reserve = min(max(talk), self.msg_limit) if talk else 0

    def fixed_code_musts(self) -> None:
        for number, item in enumerate(self.items, 1):
            for need in item.musts:
                if need in self.ref_values or _POINTER.search(need) or need in item.dropped:
                    continue
                code = re.search(r"\b([A-Z]{2,5}[- ]?\d{3,6})\b", need)
                if not code or not re.search(r"(?i)\b(?:fit\w*|compatib\w*|works? with|goes with)", need):
                    continue
                label = code.group(1)
                self.fixed_codes.setdefault(number, set()).add(re.sub(r"[^a-z0-9]", "", label.lower()))
                self.ref_values[need] = f"a fixed code, {label}: the listing's own words must say it goes with {label} (or name it among the models it also fits)"
                self.chased.add(need)
                self.trace.append({"kind": "fixed_code", "need": need, "code": label, "item": number, "from": "must"})

    def dedupe_pointers(self) -> None:
        def words(text: str) -> set[str]:
            return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) >= 3}

        for number, item in enumerate(self.items, 1):
            pointing = [m for m in item.musts if ":" in m and _POINTER.search(m)]
            drop: set[str] = set()
            for a in pointing:
                for b in pointing:
                    if a == b or a in drop or b in drop:
                        continue
                    prop_a, value_a = [x.strip() for x in a.split(":", 1)]
                    prop_b, value_b = [x.strip() for x in b.split(":", 1)]
                    wa, wb = words(value_a), words(value_b)
                    if not wa or not wb or len(wa & wb) < 0.6 * min(len(wa), len(wb)):
                        continue
                    if words(prop_a) and words(prop_a) <= words(value_b) and not (words(prop_b) and words(prop_b) <= words(value_a)):
                        drop.add(a)
            if drop:
                self.trace.append({"kind": "pointer_dedupe", "item": number, "dropped": sorted(drop)})
                item.musts = [m for m in item.musts if m not in drop]
                self.refs = [(n, m) for n, m in self.refs if not (n == number and m in drop)]

    def unmix(self) -> None:
        for number, item in enumerate(self.items, 1):
            pointers = [need for n, need in self.refs if n == number and need in item.musts]
            rest = [m for m in item.musts if m not in pointers]
            if not pointers or not rest:
                continue
            prompt = (UNMIX + "\n\nREQUEST:\n" + self.request + f"\n\nPRODUCT: {item.name}\nREQUIREMENTS POINTING AT ANOTHER LISTING:\n"
                      + "\n".join(pointers) + "\nOTHER REQUIREMENTS:\n" + "\n".join(rest))
            got = self.ask_json(prompt, [HEAVY, LIGHT], 1500) or {}
            moved = [m for m in rest if m in [str(x).strip() for x in _items(got.get("describe_other"))]]
            if not moved:
                continue
            item.musts = [m for m in item.musts if m not in moved]
            joined = pointers[0] + " (that listing: " + "; ".join(moved) + ")"
            item.musts = [joined if m == pointers[0] else m for m in item.musts]
            self.refs = [(n, joined if (n, need) == (number, pointers[0]) else need) for n, need in self.refs]
            self.trace.append({"kind": "unmix", "item": number, "moved": moved})

    def unrecorded(self) -> set[str]:
        if len(self.full) < 8:
            return set()
        key_words = {w for record in self.full.values() for k in (_fact_block(record) or {}) for w in _stem(k)}
        texts = [_dump(record).lower() for record in self.rows.values()]
        asked = [q.lower() for q in self.queries]
        pointers = {need for _, need in self.refs}
        out: set[str] = set()
        for item in self.items:
            for need in item.musts:
                if need in item.dropped or need in pointers or ":" not in need:
                    continue
                prop, value = need.split(":", 1)
                words_of_prop = {w for w in _stem(prop) if len(w) >= 3}
                if not words_of_prop or any(w in key_words or any(_alike(w, k) for k in key_words) for w in words_of_prop):
                    continue
                words = [w for w in re.findall(r"[a-z0-9]+", value.lower()) if len(w) >= 3 and w not in ("the", "and", "for", "with", "any", "all", "not")]

                def holds(text: str) -> bool:
                    return all(re.search(r"\b" + re.escape(w) + r"\b", text) for w in words)

                if not words or not any(holds(q) for q in asked) or any(holds(t) for t in texts):
                    continue
                out.add(need)
        return out

    def need_text(self, need: str) -> str:
        if need in self.ref_constraint:
            return (f"{self.ref_constraint[need]} (settled from a chain of listing pages; the requirement's own wording named another listing, "
                    "and that listing's shop, kind and properties are not requirements on this item)")
        return need + (f" (the page it points at gives: {self.ref_values[need]})" if need in self.ref_values else "")

    def ledger(self) -> str:
        parts = []
        many = len(self.items) > 1
        loose = self.unrecorded()
        for number, item in enumerate(self.items, 1):
            needs = [self.need_text(m)
                     + (" [from the shopper's answer: met only by an attribute naming this property]" if m in item.told and m not in self.ref_constraint else "")
                     + (" (no listing read so far records this property or value, so it decides nothing)" if m in loose else "")
                     for m in item.musts + item.told if m not in item.dropped]
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
            when = {"choose": "before choosing", "order": "at the order (Orochi does it when you check out)", "end": "at an ending without a purchase (Orochi does it then)"}[entry["when"]]
            parts.append(f"step the request asks for, {when}: {entry['step']}")
        if self.set_fit:
            anchor, kind, values = self.set_fit
            if kind == "series":
                parts.append(f"fit rule from the listings' codes: {anchor} takes part series {', '.join(sorted(values))}; a listing with another series code"
                             " does not fit it, a listing with no series code is not restricted by codes")
            else:
                parts.append(f"fit rule from the listings' codes: {anchor} is sold as {', '.join(sorted(values))}; a listing fits it when its own text (or the"
                             " page its text names for more) says it works with one of those")
            if self.set_soft:
                parts[-1] += (" (the request itself does not tie these items together: a listing outside these codes is acceptable when no listing"
                              " within them meets that item's requirements)")
            more = [f"{i}'s page for more (part {self.set_pages[i][0]}, listing {self.set_pages[i][1]}) says it works with {', '.join(sorted(n))}"
                    for i, n in self.set_ext.items() if n and i in self.set_pages]
            if more:
                parts.append("pages for more opened by code: " + "; ".join(more[:4]))
        facts = self.facts_tool()
        if facts and (self.facts_from.get(facts) is True or self.roles.get("offer")):
            bare = sorted(t for t in self.page_tools() | {self.roles.get("offer") or ""} if t and self.facts_from.get(t) is False)
            pages = sorted(t for t in self.page_tools() if t != facts)
            parts.append(f"listing attributes come from {facts}" + (f"; {', '.join(bare)} show none" if bare else "")
                         + (f"; a listing's page (what a chain of pages or a page check means) is shown by {' or '.join(pages)}" if pages else ""))
        return "\n".join("- " + p for p in parts)

    def status(self) -> str:
        left = self.left()
        text = f"[Orochi] step {self.step} of {self.max_steps} used, {left} left."
        if left <= self.tail() + 1:
            text += " FINAL STEPS: check out now when the cart holds a line for each item; end without a purchase only when your search showed that an item has no fitting listing (running out of steps does not show that)."
        elif left <= 6 or (left <= len(self.items) + 5 and self.buys == 0):
            text += " Few steps left: set the cart to the chosen lines and check out now."
        elif self.buys == 0 and left <= 12:
            text += " Steps are running low: settle on your picks, fill the cart and check out within the next few steps."
        elif self.buys == 0 and self.cost > self.hard:
            text += " Only a few turns are left: set the cart to your picks and check out now."
        elif self.buys == 0 and not self.cart_touched and self.step >= self.max_steps // 2:
            text += " Half the steps are used and the cart is empty: cart each item's fitting listing now (the store may still sell one out)."
        if self.elapsed() >= WIND_DOWN and left > self.tail() + 1:
            text += (" TIME IS NEARLY UP (the episode has a hard time limit and an unfinished episode earns nothing): set the cart to your"
                     " chosen lines and check out within the next two turns.")
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

    def ident(self, node: Any, in_list: bool = False, depth: int = 0) -> str | None:
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
                    old = self.rows[ident]
                    if any(old.get(k) is True and v is False for k, v in node.items()):
                        self.unavailable.add(ident)
                    elif any(old.get(k) is False and v is True for k, v in node.items()):
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
            for option in schema.get("enum") or []:
                if isinstance(option, str):
                    found.add(option.strip().lower())
            for value in schema.values():
                if isinstance(value, dict):
                    walk(value, depth + 1)

        for schema in self.schemas.values():
            walk(schema, 0)
        return found

    def known_labels(self) -> set[str]:
        found = {lab for lab in self.labels()}
        cat_key = self.category_key()
        if cat_key:
            found |= {str(row.get(cat_key)).strip() for row in self.rows.values() if isinstance(row.get(cat_key), str) and row.get(cat_key).strip()}
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
                name = _key_name(_words(key))
                if not name or name in done:
                    continue
                done.add(name)
                slot = group["keys"].setdefault(name, {"n": 0, "values": [], "per": {}})
                slot["n"] += 1
                text = str(value).strip().lower()
                slot["per"][ident] = text
                if text not in slot["values"]:
                    slot["values"].append(text)
        wanted = [(_label_words(item.shelf or ""), _label_words(item.name), {w for w in _label_words(item.shelf or "") | _label_words(item.name) if len(w) >= 4}) for item in self.items]
        matched = {label for label in groups if any(_label_words(label) in (shelf, name) or {w for w in _label_words(label) if len(w) >= 4} & long for shelf, name, long in wanted)}
        if matched:
            groups = {label: group for label, group in groups.items() if label in matched}
        table = []
        for label, group in sorted(groups.items(), key=lambda kv: -kv[1]["n"]):
            keys = group["keys"]
            heads: dict[str, list[str]] = {}
            for name in keys:
                parts = name.split()
                if len(parts) >= 3 and parts[-2] not in ("of", "per", "for", "with", "to", "in", "on", "and", "or", "by"):
                    heads.setdefault(" ".join(parts[:-1]), []).append(name)
            for head, names in heads.items():
                if head in keys:
                    slot = keys.setdefault(head, {"n": 0, "values": [], "per": {}})
                    for name in names:
                        extra = keys.pop(name)
                        for ident, text in extra["per"].items():
                            if ident not in slot["per"]:
                                slot["per"][ident] = text
                                if text not in slot["values"]:
                                    slot["values"].append(text)
                    slot["n"] = len(slot["per"])
            kept: list[tuple[str, dict[str, Any]]] = []
            for name, slot in sorted(group["keys"].items(), key=lambda kv: -kv[1]["n"]):
                if slot["n"] >= 8 and len(set(slot["per"].values())) == 1:
                    continue
                values = list(slot["per"].values())
                if not values:
                    continue
                mean = sum(len(v) for v in values) / len(values)
                if mean > 40 or sum(1 for v in values if " / " in v) * 2 >= len(values):
                    continue
                if len(values) >= 4 and len(set(values)) == len(values) and mean > 12:
                    continue
                core = _core(name)
                twin = any((name.startswith(other + " ") and slot["per"] == spec["per"]) or (core and core == _core(other)) for other, spec in kept)
                if twin:
                    continue
                kept.append((name, slot))
            for name, slot in kept[:14]:
                table.append((label, name, slot["n"], [v[:40] for v in slot["values"][:4]]))
        return table

    def item_groups(self) -> list[list[str]]:
        cat_key = self.category_key()
        groups: dict[str, list[str]] = {}
        for ident in self.seen_order:
            label = str(self.rows[ident].get(cat_key) or "") if cat_key else ""
            groups.setdefault(label, []).append(ident)
        wanted = [(_label_words(item.shelf or ""), _label_words(item.name), {w for w in _label_words(item.shelf or "") | _label_words(item.name) if len(w) >= 4}) for item in self.items]
        matched = [ids for label, ids in groups.items() if any(_label_words(label) in (shelf, name) or {w for w in _label_words(label) if len(w) >= 4} & long for shelf, name, long in wanted)]
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
        value = (self.rows.get(ident) or {}).get(self.price_key) if self.price_key else None
        if isinstance(value, dict):
            value = next((v for v in value.values() if isinstance(v, (int, float)) and not isinstance(v, bool)), None)
        if isinstance(value, str) and re.fullmatch(r"\s*\d+(\.\d+)?\s*", value):
            value = float(value)
        return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None

    def buyable(self, ident: str) -> bool:
        if self.sold_out(ident):
            return False
        value = (self.rows.get(ident) or {}).get(self.stock_key) if self.stock_key else None
        return not (value is False or (isinstance(value, (int, float)) and not isinstance(value, bool) and value <= 0))

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
        return bool(keys) and version == self.version and keys <= have and ident not in self.gone

    def read_shape(self, name: str, observation: Any) -> dict[str, Any] | None:
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
        got = self.ask_json(prompt, [LIGHT, HEAVY], 2500) or {}
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

    def more_pages(self, name: str, args: dict[str, Any], observation: Any, max_pages: int = 2) -> str:
        self.paged_obs = []
        pick = self.brief.get("pick")
        plan = self.paging(name, observation) if pick in ("cheapest", "most", "least") else None
        if not plan:
            return ""
        count, shown, param, value = plan
        props = (self.schemas.get(name) or {}).get("properties") or {}
        texts = {k for k, v in props.items() if isinstance(v, dict) and "string" in _kinds(v) and "enum" not in v}
        labels = [v for k, v in args.items() if k in texts and isinstance(v, str) and _words(v) in {_words(lab) for lab in self.known_labels()}]
        if labels and isinstance(count, int) and not any(isinstance(v, str) and v.strip() and v not in labels for k, v in args.items() if k in texts):
            self.shelf_counts[_words(labels[0])] = count
        given = args.get(param)
        later = given not in (None, 0, 1) if isinstance(given, int) and not isinstance(given, bool) else bool(str(given or "").strip())
        if value is None or later or count <= shown:
            return ""
        if pick == "cheapest" and self.price_sorted(observation):
            return ""
        pages = -(-(count - shown) // shown)
        if pages > max_pages or self.pages_read + pages > AUTO_PAGES + max(0, max_pages - 2) or self.left() - self.tail() - pages < 12:
            return (f"\n[Orochi] This search matched {count} listings and showed {shown}, in no price order. The pick counts every fitting listing: "
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
            shown_text = self.show(seen, ROW_CHARS)
            self.read_cache[name + _dump(call)] = (self.version, self.step, shown_text)
            texts.append(f"{param}={_dump(value)}: {shown_text}")
            after = self.paging(name, seen)
            value = after[3] if after else None
            if value is None:
                break
        self.trace.append({"kind": "pages", "tool": name, "read": len(texts)})
        return ("\n[Orochi read the rest of this search itself, one step a page]\n" + "\n".join(texts)) if texts else ""

    def cheapest_here(self, observations: list[Any]) -> str:
        if self.brief.get("pick") != "cheapest" or not self.price_key:
            return ""
        found = {i for seen in observations for i, _ in self.records_in(seen)}
        rows = [(self.price_of(i), i) for i in found if self.buyable(i) and self.price_of(i) is not None]
        if len(rows) < 2:
            return ""
        return "\n[Orochi] buyable rows here, cheapest first: " + ", ".join(f"{i} ({p:g})" for p, i in sorted(rows)[:6])

    def wider_hold(self) -> str:
        if self.wider_held or self.bargain_done or self.brief.get("pick") != "cheapest" or self.left() < 14 + 2 * len(self.items):
            return ""
        self.wider_held = True
        lines = [f"{i} at {self.price_of(i):g}" for i in self.cart_ids() if self.price_of(i) is not None]
        return (f"Not run yet (no step used). {self.left()} steps are left and the pick is the cheapest. Searches match a listing's title,"
                " so a requirement's value (often only in the attributes) finds few of the fitting listings, and rows come in no price order."
                " Before checking out, run a few more searches for each item: each required value on its own with the product's plain name"
                " (a title often names only one of them), and other names for the product (its plain one-word name,"
                " synonyms, words from fitting titles you have seen); look only at buyable rows priced below the cart's line"
                + (f" ({'; '.join(lines)})" if lines else "") + "; read the attributes of the cheapest such rows, swap in any that fits, then check out.")

    def fill_facts(self, ids: list[str], keep: int) -> list[str]:
        tool = self.facts_tool()
        target = self.single_target(tool)
        if not tool or not target:
            return []
        texts = []
        for ident in [i for i in ids if i in self.rows and i not in self.full and i not in self.fetched and not self.sold_out(i)]:
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
                self.fetched.add(ident)
            texts.append(f"{ident}: " + self.show(seen)[:1500])
        if texts:
            self.trace.append({"kind": "fill_facts", "n": len(texts)})
        return texts

    def cheaper_rows(self) -> list[str]:
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
            picks += [i for i in pool[:4] if i not in picks]
        return self.fill_facts(picks[:CHEAPER_OPENS], self.tail() + 4)

    def observe(self, name: str, observation: Any) -> str:
        self.absorb(observation)
        lines = self.loose_lines(observation)
        if not lines:
            return ""
        meaning = self.take_note(lines)
        if meaning:
            self.queued.append(f"(seen on {name}) {meaning}")
        return meaning

    def widen(self) -> None:
        tool = self.roles.get("compare")
        schema = self.schemas.get(tool or "") or {}
        props = schema.get("properties") or {}
        target = next((k for k, v in props.items() if isinstance(v, dict) and "array" in _kinds(v)), None)
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
        tool = self.facts_tool()
        target = self.single_target(tool)
        if not tool or not target:
            return
        lanes: list[list[str]] = []
        for ids in self.item_groups():
            opened = sum(1 for i in ids if i in self.full)
            pool = [i for i in ids if i not in self.full and self.buyable(i)]
            if self.brief.get("pick") == "cheapest":
                pool.sort(key=lambda i: self.price_of(i) if self.price_of(i) is not None else float("inf"))
            lanes.append(pool[: max(0, min(2, 8 - opened))])
        queue = [lane[k] for k in range(2) for lane in lanes if k < len(lane)]
        self.fill_facts(queue[:5], 22)

    def ready_to_ask(self) -> bool:
        if self.asked or not self.roles.get("ask") or self.msgs_free() <= 0:
            return False
        if self.left() <= 8 or self.elapsed() > INTERVIEW_BY:
            return False
        opened = len(self.full)
        if opened >= 3 * len(self.items) + 1:
            return True
        return self.step >= max(8, self.max_steps // 4) or self.elapsed() > INTERVIEW_BY - 40

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
        prompt = PLAN.replace("COUNT", str(max(PLAN_PICKS, LIST_SIZE))) + "\n\nREQUEST:\n" + self.request + "\n\nITEMS:\n" + "\n".join(settled)
        first = [str(x) for x in _items(self.brief.get("ask_first")) if str(x).strip()]
        if first:
            prompt += "\n\nASK FIRST (the request asks to be asked about these):\n" + "\n".join(first)
        prompt += "\n\nATTRIBUTE KEYS SEEN:\n" + ("\n".join(listing) or "(no full listing records yet)")
        plan = self.ask_json(prompt, [HEAVY, LIGHT], 3500) or {}
        lanes, hinted, order = self.guarded_lanes(plan, table)
        parts = self.dependent_items()
        tier = {n: (1 if n in parts and not hinted[n] else 0) for n in lanes}
        self.trace.append({"kind": "plan", "ask": _clip(plan.get("ask"), 120), "lanes": {n: v[:6] for n, v in lanes.items()}, "parts": sorted(parts)})
        told: list[str] = []
        before = [len(item.told) for item in self.items]
        sent: list[tuple[int, str, str]] = []
        asked: dict[int, set[str]] = {n: set() for n in lanes}
        found: dict[int, int] = {n: 0 for n in lanes}
        spent: dict[int, int] = {n: 0 for n in lanes}
        shown: dict[int, list[list[str]]] = {n: [] for n in lanes}
        outcome: dict[int, list[str]] = {n: [] for n in lanes}
        follow: list[tuple[int, list[str]]] = []
        while self.msgs_free() > 0 and not self.finished and self.left() > 6 and self.elapsed() < QUESTION_BY:
            if follow:
                number, labels = follow.pop(0)
            else:
                number, labels = self.next_item(lanes, asked, found, spent, order, sent, tier, shown, outcome)
                if not labels:
                    break
            labels = _dedupe_labels(labels)
            for label in labels:
                asked[number].add(_words(label))
            shown[number].append(labels)
            batch = self.send([(number, labels[0], self.list_question(number, labels))])
            if not batch:
                break
            sent += batch
            spent[number] += 1
            had = [len(item.told) for item in self.items]
            heard = self.hear(batch, (LIGHT, HEAVY) if self.plain_answer(batch[0][2]) else (HEAVY, LIGHT))
            self.take_heard(heard, told, [a for _, _, a in batch])
            for n, item in enumerate(self.items, 1):
                if len(item.told) > had[n - 1]:
                    found[n] += 1
            outcome[number].append("need" if any(len(item.told) > had[n] for n, item in enumerate(self.items)) else "none")
            crowded = 1 in [_count(x) for x in _items(heard.get("crowded"))] and len(labels) > 1
            if crowded and len(self.items[number - 1].told) == had[number - 1]:
                if len(labels) > 1:
                    shown[number].pop()
                    outcome[number].pop()
                    for label in labels[1:]:
                        asked[number].discard(_words(label))
                follow.insert(0, (number, [labels[0]]))
        if not sent:
            return ""
        if told:
            self.version += 1
        found = self.value_searches(before)
        bargain = ""
        try:
            bargain = "\n".join(t for t in (self.bargain_pass(n) for n in range(1, len(self.items) + 1)) if t)
        except Exception as exc:
            self.trace.append({"kind": "set_error", "error": f"bargain {type(exc).__name__}: {exc}"[:200]})
        lines = [f"Q: {q}\nA: {a}" for _, q, a in sent]
        summary = "Orochi's interviewer asked the shopper (each question used one step):\n" + "\n".join(lines)
        summary += "\nNew firm requirements from these answers: " + ("; ".join(told) if told else "none")
        summary += ". Apply them to every pick (search again with the new values where needed)."
        if found:
            summary += "\n" + found
        if bargain:
            summary += "\n" + bargain
        pointed = [need for _, need in self.refs if any(need in t for t in told)]
        if pointed:
            summary += (" These take their value from another listing: " + " | ".join(pointed) + ". Orochi follows their chains of pages next (below);"
                        " when a chain is not settled there, find that listing, open its page in full, and open every listing its page refers to;"
                        " the last page gives the value, and the order counts only after all of them are opened.")
        if self.queued:
            summary += "\n[notes seen meanwhile] " + " | ".join(self.queued)
            self.queued = []
        return summary

    def covered(self, number: int, label: str, sent: list[tuple[int, str, str]]) -> bool:
        core = {w for w in _core(label) if len(w) >= 3}
        return bool(core) and any(n == number and core <= _stem(answer) and not _POINTER.search(answer)
                                  and not re.search(r"(?i)\bpage\b|\blisting\b|\bsold by\b", answer) for n, _, answer in sent)

    @staticmethod
    def plain_answer(text: str) -> bool:
        return len(text.split()) <= 30 and not _POINTER.search(text) and not re.search(r"(?i)\bpage\b|\blisting\b|\bsold by\b|\bshop\b|\bseller\b", text)

    def label_owner(self, label: str) -> int | None:
        words = set(_label_words(label))
        scores = []
        for number, item in enumerate(self.items, 1):
            stems = set(_label_words(item.shelf or "") | _label_words(item.name))
            exact = 10 * (bool(words) and frozenset(words) in (_label_words(item.shelf or ""), _label_words(item.name)))
            scores.append((exact + (len(words & stems) / len(words | stems) if words | stems else 0.0), number))
        top = max(scores)
        return top[1] if top[0] > 0 and [s for s, _ in scores].count(top[0]) == 1 else None

    def lane_keys(self, number: int, table: list[tuple[str, str, int, list[str]]]) -> list[tuple[str, float]]:
        item = self.items[number - 1]
        settled = [_stem(m.split(":", 1)[0]) for m in item.musts if m not in item.dropped]
        rows = [(name, carried) for label, name, carried, _ in table if self.label_owner(label) == number]
        if len(self.items) == 1 and not rows:
            rows = [(name, carried) for _, name, carried, _ in table]
        cat_key = self.category_key()
        read = sum(1 for record in self.full.values() if cat_key and self.label_owner(str(record.get(cat_key) or "")) == number)
        read = max(read, max([c for _, c in rows] + [1]))
        kept: list[tuple[str, float]] = []
        for name, carried in sorted(rows, key=lambda r: -r[1]):
            key = _stem(name)
            if any(prop and key and (prop == key or prop <= key or (len(key) >= 2 and key <= prop)) for prop in settled):
                continue
            if _DETAIL_KEY.search(name):
                continue
            kept.append((name, carried / read))
        return [k for k in kept if not _MIXED_KEY.search(k[0])] + [k for k in kept if _MIXED_KEY.search(k[0])]

    def guarded_lanes(self, plan: dict[str, Any], table: list[tuple[str, str, int, list[str]]]) -> tuple[dict[int, list[str]], dict[int, set[str]], list[int]]:
        lanes: dict[int, list[str]] = {n: [] for n in range(1, len(self.items) + 1)}
        hinted: dict[int, set[str]] = {n: set() for n in lanes}
        order: list[int] = []
        for pick in _items(plan.get("ask")):
            if not isinstance(pick, dict):
                continue
            number = min(max(1, _count(pick.get("item") or 1) or 1), len(self.items))
            label = re.split(r"\s+(?:and|or)\s+|[/,;(]", str(pick.get("label") or ""))[0].strip().rstrip("?.")
            label = _key_name(label.lower()) if label else ""
            if not label or len(label.split()) > 5:
                continue
            if not any(_same_prop(label, other) for other in lanes[number]):
                lanes[number].append(label)
            if str(pick.get("hint") or "").strip():
                hinted[number].add(_words(label))
            if number not in order:
                order.append(number)
        own = {n: {_words(k) for k, _ in self.lane_keys(n, table)} for n in lanes}
        for number in lanes:
            others = set().union(*[own[n] for n in lanes if n != number]) if len(lanes) > 1 else set()
            if own[number]:
                lanes[number] = [label for label in lanes[number] if _words(label) in own[number] or _words(label) not in others]
        for number in lanes:
            keys = self.lane_keys(number, table)
            share = {name: s for name, s in keys}

            def share_of(label: str) -> float:
                if _words(label) in share:
                    return share[_words(label)]
                near = [s for name, s in share.items() if _stem(name) & _stem(label)]
                return max(near) if near else 0.0

            lane = list(lanes[number])
            for key, strength in [k for k in keys if not _MIXED_KEY.search(k[0])][:3]:
                if any(_same_prop(key, other) for other in lane[:3]):
                    continue
                weak = next((i for i, other in enumerate(lane[:3]) if _words(other) not in hinted[number] and share_of(other) + 0.34 <= strength), None)
                if weak is None and len(lane) >= 3:
                    continue
                lane = [other for other in lane if not _same_prop(key, other)]
                lane.insert(weak if weak is not None else len(lane), key)
            for key, _ in keys:
                if not any(_same_prop(key, other) for other in lane):
                    lane.append(key)
            lanes[number] = lane
        return lanes, hinted, order

    def request_ties(self) -> bool:
        if self.fixed_codes:
            return True
        if any(self.tie_of(n, m) for n, item in enumerate(self.items, 1) for m in item.musts + item.told):
            return True
        text = " ".join(m for item in self.items for m in item.musts + item.told) + " " + self.request
        return bool(re.search(r"(?i)\b(?:works? with|goes? with|go with|fits?|made to go with|compatible with|pairs? with|paired with|match(?:es|ing)?)"
                              r"\s+(?:the|whichever|whatever|my|an?|that|those|any|it|them|each)\b", text))

    def dependent_items(self) -> set[int]:
        out = {n for n, item in enumerate(self.items, 1) if any(self.tie_of(n, m) for m in item.musts + item.told)}
        tied = bool(out) or self.request_ties()
        try:
            groups = self.item_rows()
            anchor = (self.anchor_item(groups) or self.model_anchor(groups)) if tied else None
        except Exception as exc:
            self.trace.append({"kind": "set_error", "error": f"{type(exc).__name__}: {exc}"[:200]})
            anchor = None
        if anchor:
            out |= {n for n in range(1, len(self.items) + 1) if n != anchor}
        return out

    def next_item(self, lanes: dict[int, list[str]], asked: dict[int, set[str]], found: dict[int, int], spent: dict[int, int],
                  order: list[int], sent: list[tuple[int, str, str]], tier: dict[int, int], shown: dict[int, list[list[str]]],
                  outcome: dict[int, list[str]]) -> tuple[int, list[str]]:
        first = {n: i for i, n in enumerate(order)}
        for number in sorted(lanes, key=lambda n: (found[n] > 0, spent[n], tier.get(n, 0), first.get(n, len(first) + n))):
            labels = self.next_list(number, lanes[number], asked, sent, shown, outcome)
            if labels:
                return number, labels
        return 0, []

    @staticmethod
    def plain_value(need: str, figures: bool = False) -> str:
        value = need.split(":", 1)[1].strip() if ":" in need else ""
        value = re.sub(r"(?i)^(?:only|exactly|just|an?|the)\s+", "", value).strip(" .\"'")
        if not value or len(value.split()) > (6 if figures else 4) or (not figures and re.search(r"\d", value)):
            return ""
        if re.search(r"(?i)\b(?:not|no|never|without|except|other|anything|unlike|least|most|under|over|less|more|between|within|either|neither|same|like)\b", value):
            return ""
        return value

    def value_searches(self, before: list[int]) -> str:
        if not self.roles.get("search") or not self.searched or self.left() < 10 + 2 * len(self.items):
            return ""
        pointers = {need for _, need in self.refs}
        queries: list[str] = []
        shelves: dict[str, str | None] = {}
        for number, item in enumerate(self.items, 1):
            head = re.split(r",|\band\b|\bwith\b|\bfor\b|\(", item.name)[0].split()
            noun = head[-1] if head else ""
            for need in item.told[before[number - 1]:]:
                value = "" if need in pointers else self.plain_value(need)
                if value:
                    query = value if (self.shelf_filter(item.shelf) or noun.lower() in value.lower()) else f"{value} {noun}".strip()
                    if query not in queries:
                        queries.append(query)
                        shelves[query] = item.shelf or None
                    break
        texts = []
        for index, query in enumerate(queries[:2]):
            if self.left() < (10 if index == 0 else 18) + 2 * len(self.items) or self.finished:
                break
            self.last_search_text = ""
            ids = self.search_rows(query, shelves.get(query), pages=4 if self.brief.get("pick") in ("cheapest", "most", "least") else 0)
            texts.append(f"{query!r}: " + (self.last_search_text[:6000] if ids else "no rows"))
        if not texts:
            return ""
        return "[Orochi searched the values just required, one step each]\n" + "\n".join(texts)

    def shelf_filter(self, shelf: str | None) -> bool:
        if not shelf:
            return False
        narrow = self.search_args("x", shelf) or {}
        plain = self.search_args("x", None) or {}
        return len(narrow) > len(plain)

    def shelf_total(self, shelf: str) -> int:
        return int(self.shelf_counts.get(_words(shelf), 0) or 0) if shelf else 0

    def bargain_pass(self, number: int = 1) -> str:
        count = len(self.items)
        if (self.brief.get("pick") != "cheapest" or count > 2 or not self.roles.get("search") or self.finished
                or number in self.bargained or not 1 <= number <= count):
            return ""
        if count == 2 and self.request_ties():
            return ""
        if self.left() < 12 or self.cost > self.soft or not self.facts_tool() or self.elapsed() > EXTRAS_BY:
            return ""
        item = self.items[number - 1]
        pointers = {need for _, need in self.refs}
        live = [m for m in item.musts + item.told if m not in item.dropped]
        if any(m in pointers and m not in self.ref_values and m not in self.chased for m in live):
            return ""
        self.bargained.add(number)
        self.bargain_done = len(self.bargained) >= count
        values: list[str] = []
        for need in live:
            if need in pointers:
                continue
            value = self.plain_value(need, figures=True)
            if not value:
                limit = _LIMIT.search(need)
                if limit and not re.match(r"(?i)at most|no more than|up to|maximum of|under", limit.group(1)):
                    value = f"{limit.group(2)} {limit.group(3) or ''}".strip()
            if value and value not in values:
                values.append(value)
        singulars: list[str] = []
        plurals: list[str] = []
        for phrase in re.split(r"\s*(?:,|\band\b|\bor\b|/)\s*", item.name):
            phrase = phrase.strip(" .")
            if phrase:
                singular = " ".join(_singular(w) for w in phrase.split())
                if singular.lower() not in [p.lower() for p in singulars]:
                    singulars.append(singular)
                if phrase.lower() != singular.lower() and phrase.lower() not in [p.lower() for p in plurals]:
                    plurals.append(phrase)
        phrases = singulars + [p for p in plurals if p.lower() not in [s.lower() for s in singulars]]
        queries = values[:2] + [p for p in phrases if p.lower() not in [v.lower() for v in values]][: max(0, 2 - len(values[:2]))]
        wanted = {w for need in live if need not in pointers for w in re.findall(r"[a-z0-9]{3,}", self.plain_value(need, figures=True).lower())
                  if w not in ("the", "and", "for", "with")}
        shelf = self.shelf_label(number) or item.shelf or ""
        narrow = self.shelf_filter(shelf)
        noun = item.name.split()[-1] if item.name.split() else ""
        done = {q.lower() for q in self.queries}
        room = self.left() - self.tail() - 8
        total = self.shelf_total(shelf)
        pages_needed = -(-total // 12) if total else 0
        if narrow and count == 1 and 0 < pages_needed <= 18 and pages_needed <= room // 2:
            self.browse_rows(shelf, pages_needed)
        else:
            for query in queries:
                if self.left() < 12 or self.finished:
                    break
                if query in values and not (narrow or noun.lower() in query.lower()):
                    query = f"{query} {noun}".strip()
                if query.lower() in done:
                    continue
                done.add(query.lower())
                self.search_rows(query, shelf or None, pages=4 if count == 1 else 2)
            if not queries and narrow and self.left() >= 14 and not self.finished:
                self.browse_rows(shelf, 3 if count == 1 else 2)

        def price(ident: str) -> float:
            p = self.price_of(ident)
            return p if p is not None else float("inf")

        budget = float(self.budget) if isinstance(self.budget, (int, float)) else None
        pool = [i for i in self.item_rows().get(number, []) if self.buyable(i) and (budget is None or price(i) * max(1, item.qty) <= budget)]

        def names_value(ident: str) -> bool:
            text = self.listing_text(ident).lower()
            return bool(wanted) and any(re.search(r"\b" + re.escape(w) + r"\b", text) for w in wanted)

        pool = sorted(dict.fromkeys(pool), key=lambda i: (not names_value(i), price(i)))
        if not pool:
            return ""
        cap = max(0, min(8 if count == 1 else 4, (self.left() - self.tail() - 6) // count))
        opened = 0
        read_all: list[str] = []
        rejected: list[str] = []
        best = ""
        index = 0
        while index < len(pool) and not best and not self.finished:
            batch: list[str] = []
            while index < len(pool) and len(batch) < 3:
                ident = pool[index]
                if ident in self.full or ident in self.fetched:
                    batch.append(ident)
                    index += 1
                elif opened < cap:
                    batch.append(ident)
                    opened += 1
                    index += 1
                else:
                    break
            if not batch:
                break
            self.fill_facts([i for i in batch if i not in self.full and i not in self.fetched], self.tail() + 4)
            read = [i for i in batch if i in self.full or i in self.fetched]
            if not read:
                break
            read_all += read
            fits = sorted(self.fit_ids(number, read), key=price)
            rejected += [i for i in read if i not in fits]
            if fits:
                best = fits[0]
        if best and self.price_cap_arg() and shelf and narrow:
            swept, seen_read, seen_rejected = self.cap_sweep(number, best)
            read_all += [i for i in seen_read if i not in read_all]
            rejected += [i for i in seen_rejected if i not in rejected]
            best = swept or best
        unread = [i for i in pool if i not in self.full and i not in self.fetched and (not best or price(i) < price(best))][:6]
        if best:
            self.best_plan[number] = best
        self.trace.append({"kind": "bargain", "item": number, "pool": len(pool), "shelf_total": total, "read": read_all, "best": best, "unread_cheaper": unread})
        if not read_all:
            return ""
        lines = [f"{i} ({price(i):g}) does not meet every requirement" for i in rejected]
        text = (f"[Orochi read the rows of item {number} ({item.name}) in ascending price order (one step each): "
                + ("; ".join(lines) if lines else "nothing cheaper was read") + ". ")
        if best:
            text += (f"{best} ({price(best):g}) meets every requirement and is the cheapest fitting listing seen: cart it and check out. Run no new broad"
                     " search of this kind now: the order is held against every cheaper fitting listing that appears in any result you see, so a cheaper row"
                     " left unread can cost the episode.")
        else:
            text += f"No fit yet among the {len(read_all)} cheapest rows read" + (f"; the next unread rows by price: {', '.join(unread)}" if unread else "") + "."
        return text + "]"

    def page_all(self, args: dict[str, Any] | None, max_pages: int) -> tuple[list[str], bool]:
        name = self.roles.get("search")
        if not name or not args or self.finished:
            return [], False
        ids: list[str] = []
        call = dict(args)
        for page in range(max(1, max_pages)):
            if self.left() <= self.tail() + 6 or self.finished:
                break
            mark = name + _dump(call)
            reply = self.act(name, call)
            if reply is None or self.finished:
                break
            seen = reply.get("observation")
            self.note_search(call, seen)
            self.observe(name, seen)
            self.learn_fields(seen)
            self.read_cache[mark] = (self.version, self.step, self.show(seen, ROW_CHARS))
            got = [i for i, _ in self.records_in(seen)]
            if page == 0 and not got and self.search_sig(call) in self.search_bad:
                return [], True
            ids += got
            plan = self.paging(name, seen)
            if not got or not plan or plan[3] is None:
                break
            call = {**call, plan[2]: plan[3]}
        for ident in ids:
            self.found_by.setdefault(ident, dict(args))
        return ids, False

    def cap_sweep(self, number: int, best: str, rounds: int = 3) -> tuple[str, list[str], list[str]]:
        item = self.items[number - 1]
        shelf = self.shelf_label(number) or item.shelf or ""
        read: list[str] = []
        failed: list[str] = []
        prices = [p for p in (self.price_of(i) for i in self.seen_order) if p is not None]
        step = 1.0 if prices and all(float(p).is_integer() for p in prices) else 0.01
        values: list[str] = []
        for need in [m for m in item.musts + item.told if m not in item.dropped]:
            value = self.plain_value(self.ref_constraint.get(need, need), figures=True)
            if value and value.lower() not in [v.lower() for v in values]:
                values.append(value)
        head = re.split(r",|\band\b|\bwith\b|\bfor\b|\(", item.name)[0].strip() or item.name
        for _ in range(rounds):
            top = self.price_of(best)
            if top is None or self.finished or self.left() < self.tail() + 12 or self.elapsed() > EXTRAS_BY:
                break
            cap = top - step
            room = max(1, min(5, (self.left() - self.tail() - 14) // 3))
            ids: list[str] = []
            tries = ([f"{values[0]} {values[1]}"] if len(values) >= 2 else []) + values[:2]
            for value in tries:
                ids, _ = self.page_all(self.search_args(value, shelf, cap), room)
                if ids:
                    break
            if not ids:
                ids, refused = self.page_all(self.browse_args(shelf, None, cap), room)
                if refused:
                    ids, _ = self.page_all(self.browse_args(shelf, None, cap), room)
            if not ids:
                ids, _ = self.page_all(self.search_args(" ".join(_singular(w) for w in head.split()), shelf, cap), room)
            pool = [i for i in dict.fromkeys(ids) if i != best and self.buyable(i) and (self.price_of(i) if self.price_of(i) is not None else float("inf")) < top
                    and i in self.item_rows().get(number, [])]
            pool.sort(key=lambda i: self.price_of(i) or 0.0)
            found = ""
            index = 0
            while index < len(pool) and not found and self.left() > self.tail() + 8 and not self.finished:
                batch = pool[index:index + 3]
                index += 3
                self.fill_facts([i for i in batch if i not in self.full and i not in self.fetched], self.tail() + 7)
                done = [i for i in batch if i in self.full or i in self.fetched]
                if not done:
                    break
                read += done
                fits = sorted(self.fit_ids(number, done), key=lambda i: self.price_of(i) or 0.0)
                failed += [i for i in done if i not in fits]
                if fits:
                    found = fits[0]
            self.trace.append({"kind": "cap_sweep", "item": number, "under": top, "rows": len(ids), "pool": len(pool), "found": found})
            if not found:
                break
            best = found
        return best, read, failed

    def learn_fields(self, observation: Any) -> None:
        for _, record in self.records_in(observation):
            self.row_fields.update(k for k, v in record.items() if not isinstance(v, (dict, list)))

    def page_notes(self, ident: str) -> str:
        record = self.rows.get(ident) or {}
        if self.row_fields:
            texts = [v.strip() for k, v in record.items() if isinstance(v, str) and k not in self.row_fields and k != self.id_key]
        else:
            texts = [v.strip() for k, v in record.items() if isinstance(v, str) and re.search(r"(?i)note|remark|fine|print|memo|comment", str(k))]
        return " ".join(t for t in texts if len(t) >= 12)

    def note_field(self, ident: str) -> bool:
        record = self.rows.get(ident) or {}
        return bool(self.row_fields) and any(k not in self.row_fields and k != self.id_key and (v is None or isinstance(v, str)) for k, v in record.items())

    def question(self, number: int, label: str, again: bool = False) -> str:
        name = self.items[number - 1].name
        return f"Which {label} do you want for the {name}?" if again else f"What {label} would you like for the {name}?"

    def list_question(self, number: int, labels: list[str]) -> str:
        if len(labels) == 1:
            return self.question(number, labels[0])
        name = self.items[number - 1].name
        return f"For the {name}, what {', '.join(labels[:-1])} or {labels[-1]} would you like?"

    def next_list(self, number: int, lane: list[str], asked: dict[int, set[str]], sent: list[tuple[int, str, str]],
                  shown: dict[int, list[list[str]]], outcome: dict[int, list[str]]) -> list[str]:
        fresh = []
        for label in lane:
            if _words(label) in asked[number] or any(_same_prop(label, other) for other in asked[number]):
                continue
            if self.covered(number, label, sent):
                self.trace.append({"kind": "ask_skipped", "item": number, "label": label})
                asked[number].add(_words(label))
                continue
            fresh.append(label)
        last = shown[number][-1] if shown[number] else []
        kind = outcome[number][-1] if outcome[number] else ""
        dead = {_words(lst[0]) for lst, got in zip(shown[number], outcome[number]) if len(lst) == 1 and got == "none"}
        fresh = [label for label in fresh if _words(label) not in dead]
        if kind == "none" and len(last) == 1:
            kind = ""
        if kind == "need":
            hit = next((i for i, label in enumerate(last) if _words(label) in asked[number] and any(_same_prop(label, t.split(":")[0]) for t in self.items[number - 1].told)), -1)
            tail = [label for label in last[hit + 1:] if not any(_same_prop(label, t.split(":")[0]) for t in self.items[number - 1].told)] if hit >= 0 else []
            picks = tail + [label for label in fresh if label not in tail]
        elif kind == "none" and last:
            nones = 0
            for got in reversed(outcome[number]):
                if got != "none":
                    break
                nones += 1
            if nones >= 3:
                picks = fresh
            elif nones == 2 and len(shown[number]) >= 2:
                first_list, second_list = shown[number][-2], last
                added = [label for label in second_list if label not in first_list]
                interior = [label for label in first_list[1:-1] if label not in added]
                picks = [label for label in added + interior + fresh if not self.covered(number, label, sent) and _words(label) not in dead]
            else:
                reversed_last = [label for label in reversed(last) if not self.covered(number, label, sent) and _words(label) not in dead]
                picks = reversed_last[:LIST_SIZE - 2] + [label for label in fresh if label not in reversed_last][:2] + reversed_last[LIST_SIZE - 2:]
        else:
            picks = fresh
        out: list[str] = []
        for label in picks:
            if not any(_same_prop(label, other) for other in out):
                out.append(label)
            if len(out) >= LIST_SIZE:
                break
        return out

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
                break
        return sent

    def answer_item(self, need: str, answers: list[str]) -> int | None:
        prop = [w for w in re.findall(r"[a-z0-9]+", need.split(":", 1)[0].lower()) if len(w) >= 2]
        if not prop or len(self.items) < 2:
            return None
        known = set()
        for item in self.items:
            known |= _stem(item.name) | _stem(item.shelf or "")
        for text in answers:
            low = text.lower()
            for match in re.finditer(r"\bof (?:the|your|my|these|those)\s+", low):
                before = re.findall(r"[a-z0-9]+", low[max(0, match.start() - 60):match.start()])
                if not before or not all(w in before[-len(prop) - 2:] for w in prop):
                    continue
                words: list[str] = []
                for word in re.findall(r"[a-z0-9&]+", low[match.end():match.end() + 80])[:6]:
                    if word in ("&", "and") or _stem(word) & known:
                        words.append(word)
                    else:
                        break
                owner = self.label_owner(" ".join(words)) if words else None
                if owner:
                    return owner
        return None

    def take_heard(self, heard: dict[str, Any], told: list[str], answers: list[str] | None = None) -> None:
        many = len(self.items) > 1
        for entry in _items(heard.get("needs")):
            if not isinstance(entry, dict) or not str(entry.get("need") or "").strip():
                continue
            number = min(max(1, _count(entry.get("item")) or 1), len(self.items))
            need = str(entry["need"]).strip()
            named = self.answer_item(need, answers or [])
            if named and named != number:
                self.trace.append({"kind": "need_moved", "need": need, "from": number, "to": named})
                number = named
            if need not in self.items[number - 1].told:
                self.items[number - 1].told.append(need)
                told.append(f"item {number}: {need}" if many else need)
            if (entry.get("points") is True or _POINTER.search(need)) and (number, need) not in self.refs:
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
        return self.ask_json(prompt, list(models), 3000) or {}

    def gate(self, stage: str) -> tuple[bool, str]:
        steps = [p for p in self.process if p["when"] == stage]
        if not steps or self.finished or not self.gate_due(stage) or time.time() - self.t0 > 255:
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
            got = self.ask_json(prompt, [HEAVY, LIGHT] if nudged else [LIGHT, HEAVY], 3500)
            if got is None:
                break
            told: list[str] = []
            hold = ""
            if said:
                self.take_heard({"needs": got.get("needs")}, told)
                hold = str(got.get("hold") or "").strip()
            call = got.get("call") if isinstance(got.get("call"), dict) else None
            self.trace.append({"kind": "gate", "stage": stage, "call": _clip(call, 200), "steps": _clip(got.get("steps"), 120), "hold": hold[:200], "told": told})
            if told or hold:
                self.version += 1
                return False, ("Before going on, Orochi did the steps the request asks for:\n" + "\n".join(said) + "\n" if said else "") + (
                    "The shopper's reply changes the task: " + "; ".join(told + ([hold] if hold else [])) + ". Fix the cart for it, then check out again.")
            name = str((call or {}).get("tool") or "")
            shown = [e for e in _items(got.get("steps")) if isinstance(e, dict)]
            talk = stage == "order" and any(p["messages"] > 0 for p in steps)
            lines = range(1, len(self.done_log) + 1)
            proof = {i for i in lines if not talk or (i <= len(self.done_steps) and self.done_steps[i - 1] > self.cart_at)}
            unproven = [e for e in shown if e.get("done") is not True or not any((_count(n) or 0) in proof for n in _items(e.get("lines")))]
            if not call and unproven and not said and not nudged and self.roles.get("ask") and self.msg_used < self.msg_limit and self.left() >= 2:
                nudged = True
                self.trace.append({"kind": "gate_unproven", "steps": _clip(unproven, 120)})
                self.done_log.append(f"(Orochi: no line made since the cart last changed at step {self.cart_at} shows " + "; ".join(str(e.get("step") or "") for e in unproven)[:200]
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
        return True, ("Before going on, Orochi did the steps the request asks for:\n" + "\n".join(said)) if said else ""

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

    @staticmethod
    def record_like(line: str) -> bool:
        return line.count(" | ") >= 3 or len(re.findall(r"(?:^|[|;,]\s*|\s)[A-Za-z_][A-Za-z0-9_]{0,30}:\s", line)) >= 4

    def loose_lines(self, observation: Any) -> list[str]:
        lines = [line for line in self.loose_lines_all(observation) if not self.record_like(line)]
        return lines

    def loose_lines_all(self, observation: Any) -> list[str]:
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
        got = self.ask_json(prompt, [LIGHT, SPARE], 2500) or {}
        gone = [str(g) for g in _items(got.get("gone")) if isinstance(g, (str, int))]
        self.unavailable.update(g for g in gone if g in self.rows)
        self.note_sold = bool(gone) or got.get("sold_out") is True
        if self.note_sold and len(lines) == 1:
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
            number = self.items.index(item) + 1
            if _POINTER.search(need) and (number, need) not in self.refs:
                self.refs.append((number, need))
            added.add(need)
            changed.append(f"added {need}")
        if gone:
            changed.append("unavailable: " + ", ".join(gone))
        if got.get("price_moved") is True:
            changed.append("a price changed")
            self.price_moved = True
        if drops:
            self.terms_eased = True
        summary = str(got.get("summary") or "").strip()
        if changed:
            self.rearm()
        if changed or summary:
            self.version += 1
            self.recent = []
            self.notes.append(summary or "; ".join(changed))
        return (summary + (" (" + "; ".join(changed) + ")" if changed else "")).strip()

    def rearm(self) -> None:
        if self.audited or self.pushed or self.refuse_blocks or self.late_held:
            self.trace.append({"kind": "rearm"})
        self.audited = False
        self.pushed = False
        self.refuse_blocks = 0
        self.late_held = False

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
            parts += [f"NOT MET: {m}" for m in self.figure_for_word(ident)]
            if parts:
                lines.append(f"{ident}: " + "; ".join(parts))
        if not lines:
            return ""
        return "[evidence: keys whose names contain a requirement's property; one naming a different property (another part or aspect) decides nothing, and a key with another name may still hold the same property] " + " | ".join(lines)

    def figure_for_word(self, ident: str) -> list[str]:
        record = self.full.get(ident) or {}
        block = _fact_block(record) or {}
        cat_key = self.category_key()
        owner = self.label_owner(str((self.rows.get(ident) or record).get(cat_key) or "")) if cat_key else None
        out: list[str] = []
        for number, need in self.needs_now():
            if owner is not None and number != owner:
                continue
            value = self.plain_value(need)
            if not value or not block:
                continue
            prop = need.split(":", 1)[0].strip()
            hits = [(k, str(v)) for k, v in block.items() if _names(_stem(prop), _stem(k))]
            if not hits:
                continue
            alternatives = [{w for w in _stem(alt) if len(w) >= 3} for alt in re.split(r"\s*(?:,|/|\bor\b)\s*", value.lower())]
            alternatives = [a for a in alternatives if a]
            if not alternatives:
                continue
            stated = any(any(a <= _stem(v) for a in alternatives) for _, v in hits)
            if not stated and all(re.search(r"\d", v) for _, v in hits):
                out.append(f"{prop} -> {hits[0][0]}={hits[0][1][:40]} is a figure, not the word '{value}'")
        return out

    def cart_text(self) -> str:
        if self.cart_view is not None and self.cart_seen_at >= self.cart_at:
            return _dump(_clip(self.cart_view, 300))[:3000]
        return "lines by Orochi's record of the cart calls (listing: units): " + _dump(self.cart_lines)

    def cart_ids(self) -> list[str]:
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
            order.sort(key=lambda i: (i not in self.full, self.price_of(i) if self.price_of(i) is not None else float("inf")))
        for ident in order:
            if ident in skip or self.sold_out(ident):
                continue
            record = self.full.get(ident) or self.rows.get(ident)
            lines.append(_dump(_clip(record, 120))[:700])
            if len(lines) >= limit:
                break
        return "\n".join(lines)

    def check_key(self, ids: list[str]) -> str:
        return "|".join(sorted(ids)) + f"#{self.version}"

    def check_cart(self, ids: list[str] | None = None, limit: float | None = None) -> str:
        limit = self.soft if limit is None else limit
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
        word = [f"{i}: {m}" for i in ids for m in self.figure_for_word(i)]
        if word:
            prompt += "\n\nWORD VALUES GIVEN ONLY AS A FIGURE (each such requirement is not met by that listing):\n" + "\n".join(word)
        if self.brief.get("pick") in ("cheapest", "most", "least"):
            prompt += "\n\nOTHER LISTINGS SEEN (newest first):\n" + self.others(ids, 10)
        got = self.ask_json(prompt, [HEAVY, LIGHT], 3500)
        self.trace.append({"kind": "check", "cart": ids, "got": got})
        self.verdicts[key] = bool(got) and got.get("ok") is not False
        if not got or got.get("ok") is not False:
            return ""
        self.leads = [i for i in self.leads if i not in ids]
        problems = [str(p) for p in _items(got.get("problems"))][:5]
        better = [str(b) for b in _items(got.get("better")) if str(b) in self.rows and not self.sold_out(str(b))][:4]
        text = "A careful review says this cart is not right yet: " + "; ".join(problems or ["see the requirements"])
        if better:
            text += ". Listings that look better: " + ", ".join(better)
        return text + ". Fix the cart (or show the review is wrong by re-reading the records), then check out."

    def trail(self) -> tuple[str, bool]:
        live = {need for _, need in self.needs_now()}
        pending = [need for _, need in self.refs if need in live and need not in self.ref_values]
        tool = self.roles.get("offer") or self.roles.get("open")
        target = self.single_target(tool)
        seen: list[str] = []
        lost: list[str] = []
        while pending and self.trails < 4 and self.cost < self.hard:
            self.trails += 1
            records = []
            for ident in reversed(self.seen_order):
                record = (self.full.get(ident) or self.rows.get(ident)) if ident in self.opened else None
                if record:
                    records.append("OPENED " + _dump(_clip(record, 300))[:1400])
                else:
                    records.append("ROW " + _dump(_clip(self.rows.get(ident), 80))[:300])
                if len(records) >= 60:
                    break
            prompt = TRAIL + "\n\nREQUIREMENTS WITH A VALUE ON ANOTHER LISTING:\n" + "\n".join(pending)
            prompt += "\n\nRECORDS (newest first):\n" + "\n".join(records)
            got = self.ask_json(prompt, [LIGHT, HEAVY], 3500) or {}
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
            text += "[Orochi opened the pages a requirement points at]\n" + "\n".join(seen) + "\n"
        if self.ref_values:
            text += "[values read from the pages requirements point at] " + "; ".join(f"{k} -> {v}" for k, v in self.ref_values.items()) + "\n"
        hold = bool(lost) and self.trail_blocks < 2 and self.left() > 5
        if hold:
            self.trail_blocks += 1
            text += ("The order counts only after every page a requirement points at has been opened in full (a search row is not enough), following each page that refers to a further listing. Still to find and open: "
                     + " | ".join(lost[:3]) + ". Search for it by its shop, brand or words, open it, apply the value its page gives, then check out.")
        return text, hold

    def offer_tool(self) -> str | None:
        return self.roles.get("offer") or self.roles.get("open")

    def reread_offers(self) -> str:
        tool = self.offer_tool()
        target = self.single_target(tool)
        if not tool or not target:
            return ""
        since = max(self.step - 2, self.refused_at)
        stale = [i for i in self.cart_ids() if (self.price_moved or self.offer_seen.get(i, -99) < since) and i not in self.gone]
        if not stale or self.left() < len(stale) + 3:
            return ""
        self.price_moved = False
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
            after = self.rows.get(ident) or {}
            moved = [k for k, v in after.items() if k in before and before[k] != v and not isinstance(v, (dict, list))]
            if moved:
                lines.append(f"{ident} changed ({', '.join(moved[:4])}): {_dump(_clip(observation, 300))[:900]}")
        self.trace.append({"kind": "reread", "ids": stale, "changed": len(lines)})
        if lines:
            self.version += 1
        return "\n".join(lines)

    def sold_out(self, ident: str) -> bool:
        return ident in self.unavailable or ident in self.gone

    def swap_hint(self) -> str:
        lost = [i for i in self.cart_ids() if i not in self.gone and self.sold_out(i)]
        if not lost:
            return ""
        for ident in lost:
            self.gone[ident] = self.step
        self.leads = [i for i in self.leads if i not in lost]
        self.version += 1
        picks = self.replacements(lost) or self.refill(lost)
        swept = ""
        if self.brief.get("pick") == "cheapest" and picks and self.price_cap_arg() and self.elapsed() < HOLDS_BY:
            groups = self.item_rows()
            number = next((n for n, ids in groups.items() if lost[0] in ids), 0)
            if number:
                ran = sum(1 for t in self.trace if t.get("kind") == "cap_sweep")
                best, _, _ = self.cap_sweep(number, picks[0], rounds=2)
                if best and best != picks[0]:
                    picks = [best] + [i for i in picks if i != best]
                swept = best if sum(1 for t in self.trace if t.get("kind") == "cap_sweep") > ran else ""
        elif self.brief.get("pick") == "cheapest" and self.cheaper_unread(lost, picks):
            picks = self.replacements(lost) or picks
        text = f"[Orochi] Sold out now: {', '.join(lost)} (the store shows it cannot be bought). It will not come back, so do not open it again."
        if swept and picks:
            return text + (f" Orochi listed that item's rows priced under {picks[0]}'s and read the cheaper ones: {picks[0]} is the cheapest fitting listing left."
                           f" In this turn take the sold-out line out of the cart and add {picks[0]} at the requested quantity, then check out.")
        if picks and self.left() <= 6:
            return text + f" Few steps are left: in this turn add {picks[0]} at the requested quantity and remove the sold-out line (two calls), then check out. Other listings already read that appear to fit: {', '.join(picks[1:]) or 'none'}."
        if picks and self.brief.get("pick") == "cheapest":
            return text + f" Listings already read that appear to fit that item instead, cheapest first: {', '.join(picks)}. The pick rule is the cheapest, so unless you have already seen the whole fitting set, run one search under {picks[0]}'s price (the search's maximum price, if it has one) before you swap the sold-out line for the cheapest fitting listing."
        if picks:
            return text + f" Listings already read that appear to fit that item instead: {', '.join(picks)}. In this turn take the sold-out line out of the cart and add {picks[0]} at the requested quantity, then check out."
        return text + " Remove that line from the cart and add the best other listing that fits; search for one if none you have read fits."

    def find_sold(self) -> str:
        ids = [i for i in self.cart_ids() if i not in self.gone]
        if any(self.sold_out(i) for i in ids):
            return ""
        if len(ids) == 1:
            self.unavailable.add(ids[0])
            return ""
        return self.reread_offers() if ids else ""

    def cheaper_unread(self, lost: list[str], picks: list[str]) -> bool:
        cat_key = self.category_key()
        if not cat_key or self.cost > self.soft or self.left() < 10 + 2 * len(self.items):
            return False
        kinds = {str((self.rows.get(i) or {}).get(cat_key) or "").strip().lower() for i in lost} - {""}
        best = self.price_of(picks[0]) if picks else None
        bound = best if best is not None else float("inf")
        pool = [i for i in self.seen_order if i not in lost and i not in self.full and self.buyable(i)
                and str((self.rows.get(i) or {}).get(cat_key) or "").strip().lower() in kinds
                and (self.price_of(i) if self.price_of(i) is not None else float("inf")) < bound]
        pool.sort(key=lambda i: self.price_of(i) or 0.0)
        read = self.fill_facts(pool[:3], self.tail() + 6)
        if read:
            self.trace.append({"kind": "cheaper_unread", "read": len(read), "bound": bound})
        return bool(read)

    def replacements(self, lost: list[str]) -> list[str]:
        if self.cost > self.soft:
            return []
        records = []
        for ident in reversed(self.seen_order):
            if ident in lost or ident in self.gone or self.sold_out(ident):
                continue
            records.append(_dump(_clip(self.full.get(ident) or self.rows.get(ident), 200))[:1200])
            if len(records) >= 20:
                break
        if not records:
            return []
        prompt = SWAP + "\n\nREQUEST:\n" + self.request + "\n\nREQUIREMENTS IN FORCE:\n" + self.ledger()
        prompt += "\n\nSOLD-OUT CART LINE:\n" + "\n".join(f"{i}: {self.record_of(i)}" for i in lost) + "\n\nCART NOW:\n" + self.cart_text()
        prompt += "\n\nOTHER LISTING RECORDS SEEN (newest first):\n" + "\n".join(records)
        got = self.ask_json(prompt, [LIGHT], 3000)
        if not _items((got or {}).get("fits")) and self.cost < self.hard:
            got = self.ask_json(prompt, [HEAVY], 3000) or got
        fits: list[tuple[float, str]] = []
        for entry in _items((got or {}).get("fits")):
            ident = str(entry.get("id") if isinstance(entry, dict) else entry)
            price = entry.get("price") if isinstance(entry, dict) else None
            if ident in self.rows and ident not in lost and ident not in self.gone and ident not in self.cart_ids() and ident not in [i for _, i in fits]:
                fits.append((float(price) if isinstance(price, (int, float)) and not isinstance(price, bool) else float("inf"), ident))
        ranked = [i for _, i in sorted(fits, key=lambda f: f[0])] if self.brief.get("pick") == "cheapest" else [i for _, i in fits]
        self.trace.append({"kind": "swap", "lost": lost, "fits": fits})
        return ranked[:4]

    def audit(self) -> str:
        if self.audited or self.cost > self.hard - 0.008 or self.audit_calls >= 3 or self.room() < 30:
            return ""
        self.audited = True
        got = self.audit_call()
        named = self.audit_fits(got)
        if not named:
            rows = [str(i) for i in _items((got or {}).get("open")) if str(i) in self.rows and str(i) not in self.opened and not self.sold_out(str(i))]
            if rows and self.cost <= self.hard - 0.008 and self.open_rows(rows):
                got = self.audit_call()
                named = self.audit_fits(got)
        if not named:
            return ""
        self.lead_items = named
        self.leads = [i for ids in named.values() for i in ids]
        return "A careful review thinks these listings may fit: " + "; ".join(f"item {k}: {', '.join(v)}" for k, v in named.items()) + (f" ({got.get('reason')})" if got.get("reason") else "") + ". Open them and buy what truly fits; end without a purchase only if each one fails."

    def audit_call(self) -> dict[str, Any] | None:
        live = [i for i in reversed(self.seen_order) if not self.sold_out(i)]
        records = ["OPENED " + _dump(_clip(self.full[i], 160))[:800] for i in live if i in self.full][:12]
        records += ["ROW " + _dump(_clip(self.rows.get(i), 100))[:350] for i in live if i not in self.full][: max(6, 20 - len(records))]
        if not records:
            return None
        prompt = AUDIT + "\n\nREQUEST:\n" + self.request + "\n\nREQUIREMENTS IN FORCE:\n" + self.ledger()
        prompt += "\n\nLISTING RECORDS SEEN (newest first):\n" + "\n".join(records)
        self.audit_calls += 1
        keep = (self.hard, self.stop)
        self.hard, self.stop = self.hard + 0.012, self.stop + 0.012
        try:
            got = self.ask_json(prompt, [HEAVY, LIGHT], 4000)
        finally:
            self.hard, self.stop = keep
        self.trace.append({"kind": "audit", "got": got})
        return got

    def audit_fits(self, got: dict[str, Any] | None) -> dict[str, list[str]]:
        fits = (got or {}).get("fits")
        if not isinstance(fits, dict):
            return {}
        named = {str(k): [str(i) for i in _items(v) if str(i) in self.rows and not self.sold_out(str(i))][:4] for k, v in fits.items()}
        return {k: v for k, v in named.items() if v}

    def rescue(self) -> str:
        if (self.rescued or self.cost > self.hard - 0.008 or self.audit_calls >= 4 or not self.roles.get("cart")
                or not self.roles.get("buy") or self.left() < 2 or self.elapsed() > 220 or self.room() < 30):
            return ""
        self.rescued = True
        named = self.audit_fits(self.audit_call())
        for number, ident in self.best_plan.items():
            if ident and not self.sold_out(ident):
                named = {**(named or {})}
                key = next((k for k in named if _count(k) == number), str(number))
                named[key] = [ident] + [i for i in named.get(key, []) if i != ident]
        self.trace.append({"kind": "rescue", "named": named})
        if not named:
            return ""
        self.lead_items = named
        self.leads = [i for ids in named.values() for i in ids]
        self.confirmed = self.leads[0]
        self.completed = False
        return self.complete_cart()

    def open_rows(self, rows: list[str]) -> bool:
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
            calls.append({target: ident, **({mode: props[mode]["enum"][0]} if mode else {}), **({count: want} if count else {})})
        return calls

    def complete_cart(self) -> str:
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
        self.fill_facts(list(lines), len(calls) + 3)
        self.check_cart(list(lines), self.hard)
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
        args = _shape(dict(self.brief.get("buy_args") or {}), self.schemas.get(buy), "args", faults)
        if faults:
            return ""
        replies = []
        for attempt in range(2):
            if attempt:
                self.reread_offers()
                self.check_cart(list(lines), self.hard)
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
        return ("Not run (Orochi acted instead): the independent check confirmed " + self.confirmed + ", so Orochi set the cart (" + "; ".join(done)
                + ") and checked out. Store reply: " + " | then: ".join(replies))

    def find_refs(self) -> None:
        if not _POINTER.search(self.request) or self.no_credit:
            return
        prompt = (REFS + "\n\nITEMS:\n" + "\n".join(f"item {n}: {it.name}" for n, it in enumerate(self.items, 1))
                  + "\n\nREQUEST:\n" + self.request)
        got = self.ask_json(prompt, [LIGHT, HEAVY], 2500) or {}
        for entry in _items(got.get("refs")):
            if not isinstance(entry, dict) or not str(entry.get("need") or "").strip():
                continue
            need = str(entry["need"]).strip()
            number = min(max(1, _count(entry.get("item")) or 1), len(self.items))
            item = self.items[number - 1]
            prop = _words(need.split(":")[0])
            if ":" not in need or len(need.split(":")[0].split()) > 4:
                words = set(re.findall(r"[a-z]{4,}", need.lower()))
                if any(len(words & set(re.findall(r"[a-z]{4,}", m.lower()))) >= 0.5 * max(1, min(len(words), len(set(re.findall(r"[a-z]{4,}", m.lower())))))
                       for m in item.musts if _POINTER.search(m) or (number, m) in self.refs):
                    continue
            known = [m for m in item.musts if _words(m.split(":")[0]) == prop]
            if any(_POINTER.search(m) for m in known) or any(_words(r.split(":")[0]) == prop for n, r in self.refs if n == number):
                continue
            item.musts = [m for m in item.musts if m not in known] + [need]
            self.refs.append((number, need))
            self.trace.append({"kind": "ref_added", "item": number, "need": need})

    def learn_search(self, args: dict[str, Any], observation: Any) -> None:
        ids = [i for i, _ in self.records_in(observation)]
        if not ids:
            return
        self.searched = True
        for ident in ids:
            self.found_by.setdefault(ident, dict(args))
        schema = self.schemas.get(self.roles.get("search") or "") or {}
        props = schema.get("properties") or {}
        named = self.brief.get("search_words_arg")
        if isinstance(named, str) and named in props:
            texts = [named] if isinstance(args.get(named), str) and args[named].strip() else []
        else:
            texts = [k for k, v in args.items() if isinstance(v, str) and v.strip() and isinstance(props.get(k), dict) and "enum" not in props[k]]
        if texts:
            key = max(texts, key=lambda k: len(args[k].split()))
            keep = {k: v for k, v in args.items() if k == key or k in (schema.get("required") or [])}
            extra = any(k != key and k not in (schema.get("required") or []) and v not in (None, "", []) for k, v in args.items())
            if not self.search_template or not extra or self.template_extra:
                self.search_template = (key, keep)
                self.template_extra = extra

    def search_args(self, query: str, shelf: str | None = None, cap: float | None = None) -> dict[str, Any] | None:
        name = self.roles.get("search")
        schema = self.schemas.get(name or "") or {}
        props = schema.get("properties") or {}
        texts = [k for k, v in props.items() if isinstance(v, dict) and "string" in _kinds(v) and "enum" not in v]
        if self.search_template and self.search_template[0] in props:
            key, base = self.search_template
            args = dict(base)
            paging = (self.shapes.get(name or "") or {}).get("next_arg")
            if paging and paging != key:
                args.pop(paging, None)
        else:
            named = self.brief.get("search_words_arg")
            key = named if isinstance(named, str) and named in texts else (
                next((k for k in schema.get("required") or [] if k in texts), None) or (texts[0] if len(texts) == 1 else None))
            if not key:
                return None
            args = {}
            for other in schema.get("required") or []:
                spec = props.get(other) or {}
                if other != key and isinstance(spec.get("enum"), list) and spec["enum"]:
                    args[other] = spec["enum"][0]
        args[key] = query
        if shelf:
            label = next((lab for lab in self.known_labels() if _words(lab) == _words(shelf)), None)
            filt = next((k for k, v in props.items() if k != key and isinstance(v, dict) and "string" in _kinds(v) and "enum" not in v
                         and re.search(r"(?i)section|categor|shelf|department|group|label", k)), None)
            if label and filt:
                args[filt] = label
        cap_arg = self.price_cap_arg()
        if cap is not None and cap_arg:
            args[cap_arg] = int(cap) if "integer" in _kinds(props.get(cap_arg) or {}) else round(float(cap), 2)
        faults: list[str] = []
        args = _shape(args, schema, "args", faults)
        return None if faults else self.fit_search_mode(args)

    def mode_args(self) -> list[str]:
        props = (self.schemas.get(self.roles.get("search") or "") or {}).get("properties") or {}
        return [k for k, v in props.items() if isinstance(v, dict) and isinstance(v.get("enum"), list) and len(v["enum"]) > 1]

    def search_sig(self, args: dict[str, Any]) -> tuple[frozenset[str], str]:
        modes = self.mode_args()
        present = frozenset(k for k, v in args.items() if k not in modes and v not in (None, "", []))
        return present, _dump({k: args.get(k) for k in modes})

    def note_search(self, args: dict[str, Any], observation: Any) -> None:
        if not self.mode_args() or not isinstance(args, dict):
            return
        sig = self.search_sig(args)
        failed = isinstance(observation, dict) and 0 < len(observation) <= 2 and not any(isinstance(v, (list, dict)) for v in observation.values()) \
            and any(isinstance(v, str) for v in observation.values()) and not self.records_in(observation)
        if failed:
            self.search_bad.add(sig)
        elif isinstance(observation, (dict, list)):
            modes = self.mode_args()
            entry = (sig[0], {k: args[k] for k in modes if k in args})
            if entry not in self.search_ok:
                self.search_ok.append(entry)

    def fit_search_mode(self, args: dict[str, Any]) -> dict[str, Any]:
        modes = self.mode_args()
        if not modes or self.search_sig(args) not in self.search_bad:
            return args
        present = self.search_sig(args)[0]
        for ok_present, ok_modes in reversed(self.search_ok):
            if ok_present <= present:
                fixed = {**args, **ok_modes}
                if self.search_sig(fixed) not in self.search_bad:
                    self.trace.append({"kind": "mode_fixed", "from": _dump({k: args.get(k) for k in modes}), "to": _dump(ok_modes)})
                    return fixed
        props = (self.schemas.get(self.roles.get("search") or "") or {}).get("properties") or {}
        for key in modes:
            for value in props[key]["enum"]:
                fixed = {**args, key: value}
                if self.search_sig(fixed) not in self.search_bad:
                    self.trace.append({"kind": "mode_fixed", "from": _dump({k: args.get(k) for k in modes}), "to": _dump({key: value})})
                    return fixed
        return args

    def search_rows(self, query: str, shelf: str | None = None, pages: int = 0, cap: float | None = None) -> list[str]:
        name = self.roles.get("search")
        args = self.search_args(query, shelf, cap)
        if not name or not args or self.finished:
            return []
        reply = self.act(name, args)
        if reply is None or self.finished:
            return []
        seen = reply.get("observation")
        self.note_search(args, seen)
        if self.set_tries and not self.price_key:
            self.read_shape(name, seen)
        self.observe(name, seen)
        self.learn_fields(seen)
        self.queries.append(query)
        self.last_search_text = self.show(seen, ROW_CHARS)
        if pages > 0:
            more = self.more_pages(name, args, seen, max_pages=pages)
            if more:
                self.last_search_text += more
        self.read_cache[name + _dump(args)] = (self.version, self.step, self.last_search_text)
        ids = [i for i, _ in self.records_in(seen)] + [i for obs in self.paged_obs for i, _ in self.records_in(obs)] if pages > 0 else [i for i, _ in self.records_in(seen)]
        for ident in ids:
            self.found_by.setdefault(ident, dict(args))
        self.trace.append({"kind": "code_search", "query": query, "rows": len(ids)})
        return ids

    def key_conflict(self, ident: str, needs: list[str]) -> bool:
        facts = _fact_block(self.full.get(ident) or self.rows.get(ident) or {}) or {}
        props = [_words(need.split(":", 1)[0]) for need in needs if ":" in need]
        seen: dict[str, set[str]] = {}
        for key, value in facts.items():
            norm = _words(str(key))
            if norm in props and isinstance(value, (str, int, float)):
                seen.setdefault(norm, set()).add(str(value).strip().lower())
        return any(len(values) > 1 for values in seen.values())

    def shelf_label(self, number: int) -> str:
        cat_key = self.category_key()
        if cat_key:
            counts: dict[str, int] = {}
            for ident in self.item_rows().get(number, []):
                label = str((self.rows.get(ident) or {}).get(cat_key) or "").strip()
                if label:
                    counts[label] = counts.get(label, 0) + 1
            if counts:
                return max(counts, key=counts.get)
        return self.items[number - 1].shelf or ""

    def browse_args(self, label: str, offset: Any = None, cap: float | None = None) -> dict[str, Any] | None:
        name = self.roles.get("search")
        schema = self.schemas.get(name or "") or {}
        props = schema.get("properties") or {}
        words_key = self.search_template[0] if self.search_template else self.brief.get("search_words_arg")
        cat_arg = next((k for k, v in props.items() if k != words_key and isinstance(v, dict) and "string" in _kinds(v) and "enum" not in v
                        and re.search(r"(?i)section|categor|shelf|department|group|label", k)), None)
        if not name or not cat_arg or not label:
            return None
        args: dict[str, Any] = {cat_arg: label}
        for key, spec in props.items():
            options = spec.get("enum") if isinstance(spec, dict) else None
            if isinstance(options, list) and options and key != cat_arg:
                pick = next((o for o in options if isinstance(o, str) and re.search(r"(?i)browse|field|filter|criteria|categor|list", o)), None)
                if pick:
                    args[key] = pick
                elif key in (schema.get("required") or []):
                    args[key] = options[0]
        if offset is not None:
            shape = self.shapes.get(name) or {}
            if shape.get("next_arg"):
                args[shape["next_arg"]] = offset
        cap_arg = self.price_cap_arg()
        if cap is not None and cap_arg:
            spec = props.get(cap_arg) or {}
            args[cap_arg] = int(cap) if "integer" in _kinds(spec) else round(float(cap), 2)
        present = self.search_sig(args)[0]
        known = [modes for ok, modes in self.search_ok if ok == present or (cat_arg in ok and ok <= present | {cap_arg} and words_key not in ok)]
        if known and self.search_sig({**args, **known[-1]}) not in self.search_bad:
            args.update(known[-1])
        faults: list[str] = []
        args = _shape(args, schema, "args", faults)
        return None if faults else self.fit_search_mode(args)

    def price_cap_arg(self) -> str | None:
        name = self.roles.get("search")
        props = (self.schemas.get(name or "") or {}).get("properties") or {}
        paging = (self.shapes.get(name or "") or {}).get("next_arg")
        for key, spec in props.items():
            if key == paging or not isinstance(spec, dict) or not ({"integer", "number"} & set(_kinds(spec))):
                continue
            words = f"{key} {spec.get('description') or ''}"
            if re.search(r"(?i)budget|price|cost|ceiling|spend", words) and not re.search(r"(?i)\bmin|lowest|least|page|offset|count|quantity", key):
                return key
        return None

    def browse_rows(self, label: str, pages: int = 2, cap: float | None = None) -> list[str]:
        name = self.roles.get("search")
        ids: list[str] = []
        offset: Any = None
        for _ in range(max(1, pages)):
            args = self.browse_args(label, offset, cap)
            if not name or not args or self.finished or self.left() <= self.tail() + 3:
                break
            args = self.fit_search_mode(args)
            mark = name + _dump(args)
            if mark in self.read_cache:
                break
            reply = self.act(name, args)
            if reply is None or self.finished:
                break
            seen = reply.get("observation")
            self.note_search(args, seen)
            self.observe(name, seen)
            self.learn_fields(seen)
            self.read_cache[mark] = (self.version, self.step, self.show(seen, ROW_CHARS))
            got = [i for i, _ in self.records_in(seen)]
            ids += got
            plan = self.paging(name, seen)
            if plan and isinstance(plan[0], int):
                self.shelf_counts[_words(label)] = plan[0]
            if not got or not plan or plan[3] is None:
                break
            offset = plan[3]
        self.trace.append({"kind": "browse", "label": label, "rows": len(ids)})
        return ids

    def search_words(self, args: dict[str, Any] | None) -> str:
        if not args:
            return ""
        texts = [v for v in args.values() if isinstance(v, str) and v.strip()]
        return max(texts, key=lambda v: len(v.split())) if texts else ""

    def refill(self, lost: list[str]) -> list[str]:
        if self.refilled or self.left() < 12 or self.cost > self.soft or not self.roles.get("search"):
            return []
        self.refilled = True
        cat_key = self.category_key()
        kinds = {str((self.rows.get(i) or {}).get(cat_key) or "").strip().lower() for i in lost} - {""} if cat_key else set()
        queries: list[str] = []
        for ident in lost:
            words = self.search_words(self.found_by.get(ident))
            if words and words not in queries:
                queries.append(words)
        for item in self.items:
            label = _stem(item.shelf or item.name)
            if not kinds or any(label & _stem(k) for k in kinds):
                if item.name not in queries:
                    queries.append(item.name)
        fresh: list[str] = []
        for query in queries[:2]:
            if self.left() < 10:
                break
            for ident in self.search_rows(query):
                row = self.rows.get(ident) or {}
                if ident in lost or ident in fresh or ident in self.full or not self.buyable(ident):
                    continue
                if kinds and str(row.get(cat_key) or "").strip().lower() not in kinds:
                    continue
                fresh.append(ident)
        if not fresh:
            self.trace.append({"kind": "refill", "lost": lost, "fresh": 0})
            return []
        if self.brief.get("pick") == "cheapest":
            fresh.sort(key=lambda i: self.price_of(i) if self.price_of(i) is not None else float("inf"))
        self.fill_facts(fresh[:3], self.tail() + 6)
        self.trace.append({"kind": "refill", "lost": lost, "fresh": len(fresh)})
        return self.replacements(lost)

    def chase_due(self) -> list[str]:
        live = {need for _, need in self.needs_now()}
        return [need for _, need in self.refs if need in live and need not in self.ref_values and need not in self.chased]

    def chase_room(self) -> bool:
        return (not self.finished and self.cost < self.soft + 0.006 and self.calls < ALL_CALLS - 10 and self.elapsed() < EXTRAS_BY
                and self.left() >= 2 + 2 * len(self.items) + 8 and self.step - self.chase_from < max(8, self.max_steps // 4 + 1))

    def chase_all(self) -> str:
        texts = []
        self.chase_from = self.step
        for need in self.chase_due()[:2]:
            self.chased.add(need)
            text = self.chase(need)
            if text:
                texts.append(text)
        return "\n".join(texts)

    def settled_judge(self, prop: str, spec: dict[str, Any], ident: str) -> dict[str, Any]:
        verdict = self.judge_page(prop, spec, ident)
        pointer = verdict.get("next") if isinstance(verdict.get("next"), dict) else {}
        if pointer and not str(pointer.get("seller") or "").strip() and (pointer.get("kind") or pointer.get("features")) and self.cost < self.hard:
            second = self.judge_page(prop, spec, ident, (HEAVY, LIGHT))
            again = second.get("next") if isinstance(second.get("next"), dict) else {}
            if str(again.get("seller") or "").strip():
                return second
        if verdict.get("match") is False and self.cost < self.hard:
            second = self.judge_page(prop, spec, ident, (HEAVY, LIGHT))
            if second.get("match") is not False:
                return second
        return verdict

    def judge_page(self, prop: str, spec: dict[str, Any], ident: str, models: tuple[str, ...] = (LIGHT, HEAVY)) -> dict[str, Any]:
        prompt = (JUDGE.replace("PROPERTY", prop or "the property").replace("KIND", spec.get("kind") or "listing")
                  .replace("SELLER", spec.get("seller") or "?").replace("FEATURES", ", ".join(spec.get("features") or []) or "no named values")
                  + "\n\nPAGE:\n" + _dump(_clip(self.rows.get(ident) or {}, 900))[:3500])
        got = self.ask_json(prompt, list(models), 2500) or {}
        row = self.rows.get(ident) or {}
        seller = spec.get("seller") or ""
        shop_ok = self.sold_by(row, seller) or self.sold_by({"shop": str(got.get("shop") or "")}, seller)
        got["match"] = bool(shop_ok) and not str(got.get("contradicts") or "").strip()
        return got

    @staticmethod
    def sold_by(row: dict[str, Any], seller: str) -> bool:
        want = re.sub(r"['’]s$", "", seller.strip()).strip().lower()
        return bool(want) and any(isinstance(v, str) and v.strip().lower() == want for v in row.values())

    def link_score(self, row: dict[str, Any], kind: str, features: list[str]) -> int:
        text = _dump(row).lower()
        score = 0
        cat_key = self.category_key()
        label = str(row.get(cat_key) or "") if cat_key else ""
        if kind and label and {w for w in _stem(kind) if len(w) >= 4} & {w for w in _stem(label) if len(w) >= 4}:
            score += 2
        for feature in features:
            words = [w for w in re.findall(r"[a-z0-9.]+", feature.lower()) if len(w) >= 2]
            if words and all(w in text for w in words):
                score += 1
        return score

    def locate(self, seller: str, kind: str, features: list[str], tried: set[str]) -> list[str]:
        seller = seller.strip()
        if not seller:
            return []

        def ranked(pool: list[str]) -> list[str]:
            scored = [(self.link_score(self.rows.get(i) or {}, kind, features), i) for i in pool
                      if i not in tried and i in self.rows and self.sold_by(self.rows[i], seller)]
            scored.sort(key=lambda x: -x[0])
            return [i for _, i in scored][:4]

        known = ranked(list(dict.fromkeys(reversed(self.seen_order))))
        if len(known) == 1 and self.link_score(self.rows[known[0]], kind, features) >= 2 + len(features):
            return known
        words = " ".join(f.strip() for f in features if f.strip())
        first: list[str] = []
        shelf = next((lab for lab in self.labels() if kind and {w for w in _stem(kind) if len(w) >= 4} and {w for w in _stem(kind) if len(w) >= 4} <= _stem(lab)), None)
        singles: list[str] = []
        for feature in [f.strip() for f in features if f.strip()][:2]:
            lead = re.match(r"(\d[\d.,]*)\s+[a-z]", feature, re.I)
            query = f"{seller} {lead.group(1) if lead else feature}"
            if query not in singles and query != f"{seller} {words}".strip():
                singles.append(query)
        for number, query in enumerate(q for q in dict.fromkeys([f"{seller} {words}".strip()] + singles + [f"{seller} {kind}".strip(), seller]) if q):
            if query in self.chase_rows:
                rows = self.chase_rows[query]
            elif not self.chase_room():
                break
            else:
                rows = self.search_rows(query, shelf)
                self.chase_rows[query] = rows
            if number == 0:
                first = [i for i in rows if i not in tried]
            hits = ranked(rows)
            if hits:
                return hits
        if known or any(self.sold_by(self.rows.get(i) or {}, seller) for i in self.rows):
            return known
        if 0 < len(first) <= 3:
            return first[:1]
        if kind and words and self.chase_room():
            rows = [i for i in self.search_rows(f"{kind} {words}") if i not in tried]
            if 0 < len(rows) <= 3:
                return rows[:1]
        return []

    def open_page(self, tool: str, target: str, ident: str) -> bool:
        faults: list[str] = []
        args = _shape({target: ident}, self.schemas.get(tool), "args", faults)
        if faults or self.finished:
            return False
        reply = self.act(tool, args)
        if reply is None or self.finished:
            return False
        seen = reply.get("observation")
        self.observe(tool, seen)
        self.note_view(tool, seen)
        self.read_cache[tool + _dump(args)] = (self.version, self.step, self.show(seen))
        if tool == self.offer_tool():
            self.offer_seen[ident] = self.step
        if isinstance(seen, dict) and self.ident(seen):
            self.facts_from[tool] = self.facts_from.get(tool, False) or bool(_fact_block(seen))
        return True

    def read_facts(self, ident: str) -> bool:
        tool = self.facts_tool()
        target = self.single_target(tool)
        if not tool or not target or ident in self.full:
            return False
        return bool(self.fill_facts([ident], 2 + 2 * len(self.items) + 4))

    def chase(self, need: str) -> str:
        tool = self.roles.get("offer") or self.roles.get("open")
        target = self.single_target(tool)
        if not tool or not target or not self.roles.get("search") or not self.chase_room():
            return ""
        start = self.ask_json(START + "\n\nREQUIREMENT: " + need, [LIGHT, HEAVY], 2000) or {}
        prop = str(start.get("property") or need.split(":", 1)[0]).strip()

        def spec_of(raw: Any) -> dict[str, Any]:
            raw = raw if isinstance(raw, dict) else {}
            return {"seller": _shop(str(raw.get("seller") or "")), "kind": str(raw.get("kind") or "").strip(),
                    "features": [str(f).strip() for f in _items(raw.get("features")) if str(f).strip()][:4]}

        spec = spec_of(start)
        if not spec["seller"]:
            named = re.search(r"(?i)\bthe ([A-Z][a-z]{3,})\s+[a-z][a-z &,'-]{2,60}?\s+(?:page|listing)\b", need)
            if named and named.group(1).lower() not in {w for item in self.items for w in _stem(item.name)}:
                spec["seller"] = named.group(1)
        fixed = re.search(r"\b([A-Z]{2,5}[- ]?\d{3,6})\b", need)
        if not spec["seller"] and fixed and not self.order_item(need, spec):
            code = fixed.group(1)
            number = next((n for n, m in self.refs if m == need), 0)
            if number:
                self.fixed_codes.setdefault(number, set()).add(re.sub(r"[^a-z0-9]", "", code.lower()))
            self.ref_values[need] = f"a fixed code, {code}: the listing's own words must say it goes with {code} (or name it among the models it also fits)"
            self.version += 1
            self.trace.append({"kind": "fixed_code", "need": need, "code": code, "item": number})
            return (f"[Orochi] The requirement '{need}' names a device by its code, {code}, and no page to follow: choose that item among the listings whose own "
                    f"words say they go with {code} (a 'goes with' or 'also fits' list naming it, or the page their text names for more). Orochi's set builder applies it.")
        tied = self.order_item(need, spec)
        if tied:
            self.order_rules[need] = tied
            self.ref_values[need] = (f"no page to follow: it ties this item to item {tied} ({self.items[tied - 1].name}) of this order, "
                                     "and the listings' own codes decide which pairs fit")
            self.version += 1
            self.trace.append({"kind": "order_rule", "need": need, "item": tied})
            return (f"[Orochi] The requirement '{need}' names no shop and no page: it ties this item to item {tied} "
                    f"({self.items[tied - 1].name}) of this order. The listings' own codes decide which pairs fit (a part's series code "
                    "among the series the other listing takes, or a model name the other listing carries).")
        if not spec["features"] and re.search(r"(?i)\b(?:has|have|with|comes with)\b", need) and self.cost < self.soft:
            again = spec_of(self.ask_json(START + "\n\nREQUIREMENT: " + need, [HEAVY, LIGHT], 2000))
            if again["features"] and again["seller"]:
                spec = again
        chain: list[str] = []
        tried: set[str] = set()
        steps: list[str] = []
        value, rule = "", ""
        alike = 0
        for _ in range(10):
            if not spec["seller"] or not self.chase_room():
                break
            found = self.locate(spec["seller"], spec["kind"], spec["features"], tried)
            if not found:
                steps.append(f"found no {spec['kind'] or 'listing'} sold by {spec['seller']}")
                break
            ident, others = found[0], len(found) > 1
            tried.add(ident)
            if not self.open_page(tool, target, ident):
                break
            verdict = self.settled_judge(prop, spec, ident)
            nxt = spec_of(verdict.get("next"))
            if not nxt["seller"] and self.note_field(ident) and not self.page_notes(ident):
                alts = [i for i in found[1:6] if i not in tried] or self.locate(spec["seller"], spec["kind"], spec["features"], tried)[:4]
                for alt in alts:
                    if not self.chase_room():
                        break
                    tried.add(alt)
                    if not self.open_page(tool, target, alt):
                        break
                    if not self.page_notes(alt):
                        continue
                    second = self.settled_judge(prop, spec, alt)
                    if second.get("match") is not False:
                        steps.append(f"{ident} has no store notes; took {alt}, whose page has them")
                        ident, verdict = alt, second
                        nxt = spec_of(verdict.get("next"))
                        break
            if (verdict.get("match") is not False and (not nxt["seller"] or (others and verdict.get("confirmed") is False))
                    and ident not in self.full and self.facts_tool() and self.read_facts(ident)):
                steps.append(f"read the attributes of {ident}")
                verdict = self.settled_judge(prop, spec, ident)
                nxt = spec_of(verdict.get("next"))
            value = str(verdict.get("value") or "").strip()
            noteless = self.note_field(ident) and not self.page_notes(ident)
            silent = not nxt["seller"] and ((not value and verdict.get("confirmed") is False) or (noteless and verdict.get("confirmed") is not True))
            if (not silent and verdict.get("match") is not False and not nxt["seller"] and spec["features"]
                    and verdict.get("confirmed") is not True):
                alts = [i for i in found[1:4] if i not in tried] or self.locate(spec["seller"], spec["kind"], spec["features"], tried)[:3]
                for alt in alts:
                    if not self.chase_room():
                        break
                    tried.add(alt)
                    if not self.open_page(tool, target, alt):
                        break
                    second = self.settled_judge(prop, spec, alt)
                    if second.get("match") is not False and second.get("confirmed") is True:
                        steps.append(f"{ident} does not state {', '.join(spec['features'])}; took {alt}, whose page does")
                        ident, verdict = alt, second
                        nxt = spec_of(verdict.get("next"))
                        value = str(verdict.get("value") or "").strip()
                        if not value and not nxt["seller"] and ident not in self.full and self.facts_tool() and self.read_facts(ident):
                            verdict = self.settled_judge(prop, spec, ident)
                            nxt = spec_of(verdict.get("next"))
                            value = str(verdict.get("value") or "").strip()
                        break
            if verdict.get("match") is False or silent:
                steps.append(f"{ident} is not the link" + (" (a look-alike)" if verdict.get("match") is False else " (no value)"))
                value = ""
                alike += 1
                crowd = any(re.search(r"(?i)exists elsewhere|is not the|not the \S+ (?:one|version)|another listing", self.page_notes(i) or "") for i in tried)
                if alike >= (6 if crowd else 3):
                    break
                continue
            alike = 0
            chain.append(ident)
            steps.append(f"opened {ident} ({spec['kind'] or 'listing'} sold by {spec['seller']})")
            rule = str(verdict.get("rule") or "").strip() or rule
            if nxt["seller"]:
                spec = nxt
                value = ""
                continue
            break
        if value and chain:
            prompt = (SETTLE + "\n\nREQUIREMENT: " + need + f"\nTHAT LISTING'S OWN VALUE FOR {prop or 'the property'}: {value}"
                      + (f"\nPAIRING RULE STATED ON THE CHAIN'S PAGES: {rule}" if rule else ""))
            settled = self.ask_json(prompt, [HEAVY, LIGHT] if self.cost < self.hard else [LIGHT], 2000) or {}
            constraint = str(settled.get("constraint") or "").strip()
            relation = str(start.get("relation") or "same").strip().lower()
            marks = [w for w in re.findall(r"[a-z0-9.]+", value.lower()) if len(w) >= 3 and w not in ("the", "and", "with")]
            if relation in ("same", "") and not rule and marks and not any(w in constraint.lower() for w in marks):
                self.trace.append({"kind": "settle_drift", "need": need, "value": value, "constraint": constraint})
                constraint = f"{prop}: {value}"
            self.ref_values[need] = value + (f"; pairing rule on the page: {rule}" if rule else "") + (f"; so the product needs {constraint}" if constraint else "")
            if constraint:
                self.ref_constraint[need] = constraint
            self.version += 1
            self.trace.append({"kind": "chase", "need": need, "chain": chain, "value": value, "rule": rule, "constraint": constraint, "steps": steps})
            return (f"[Orochi followed the chain of pages for the requirement '{need}': " + "; ".join(steps) + f". The last page gives: {value}"
                    + (f" (pairing rule: {rule})" if rule else "") + (f". So the product needs {constraint}" if constraint else "")
                    + ". Every page of the chain is opened now. Choose that item by this value.]")
        self.trace.append({"kind": "chase_lost", "need": need, "chain": chain, "steps": steps, "start": _clip(start, 100)})
        if steps:
            return (f"[Orochi tried to follow the chain of pages for the requirement '{need}': " + "; ".join(steps)
                    + ". The value is not known yet: find the listing the last page points at by its shop's name and the values it names, and open its page.]")
        return ""

    def listing_text(self, ident: str) -> str:
        record = self.rows.get(ident) or {}
        return " ".join(str(v) for v in record.values() if isinstance(v, str))

    def accepted_series(self, ident: str) -> tuple[set[str], str, str]:
        text = self.listing_text(ident)
        found: set[str] = set()
        for match in _SERIES_LIST.finditer(_ALSO_SERIES.sub(" ", text)):
            found |= {s.upper() for s in re.findall(_SERIES, match.group(1), re.I)}
        see = _SEE_PART.search(text)
        model = _MODEL_CODE.search(text)
        return found, (see.group(1).upper() if see else ""), (model.group(1).upper() if model else "")

    def part_series(self, ident: str) -> str:
        match = _SERIES_CODE.search(self.listing_text(ident))
        return match.group(1).upper() if match else ""

    def extension_series(self, anchor: str, code: str, model: str) -> set[str]:
        if anchor in self.set_ext:
            return self.set_ext[anchor]
        pattern = re.compile(r"\[\s*" + re.escape(code) + r"\b", re.I)

        def holder() -> str | None:
            return next((i for i in self.seen_order if i != anchor and pattern.search(self.listing_text(i))), None)

        ident = holder()
        if not ident and self.left() > 2 * len(self.items) + 6:
            self.search_rows(code)
            ident = holder()
        added: set[str] = set()
        if ident:
            tool = self.offer_tool()
            target = self.single_target(tool)
            if tool and target and ident not in self.opened and self.left() > 2 * len(self.items) + 5:
                self.open_page(tool, target, ident)
            for match in _ALSO_SERIES.finditer(self.listing_text(ident)):
                if not model or match.group(1).upper() == model:
                    added |= {s.upper() for s in re.findall(_SERIES, match.group(2), re.I)}
        self.set_ext[anchor] = added
        self.trace.append({"kind": "set_extension", "anchor": anchor, "code": code, "page": ident, "added": sorted(added)})
        return added

    def item_rows(self) -> dict[int, list[str]]:
        cat_key = self.category_key()
        out: dict[int, list[str]] = {n: [] for n in range(1, len(self.items) + 1)}
        names = [(set(_label_words(item.shelf or "") | _label_words(item.name)), _label_words(item.shelf or ""), _label_words(item.name)) for item in self.items]

        def best(label: str) -> int | None:
            words = set(_label_words(label))
            scores = [(10 * (bool(words) and frozenset(words) in (shelf, name)) + (len(words & stems) / len(words | stems) if words | stems else 0.0), number)
                      for number, (stems, shelf, name) in enumerate(names, 1)]
            top = max(scores)
            return top[1] if top[0] > 0 and [s for s, _ in scores].count(top[0]) == 1 else None

        if cat_key:
            labels: dict[str, list[str]] = {}
            for ident in self.seen_order:
                labels.setdefault(str((self.rows.get(ident) or {}).get(cat_key) or ""), []).append(ident)
            mapped: dict[int, list[tuple[str, list[str]]]] = {}
            for label, ids in labels.items():
                number = best(label)
                if number:
                    mapped.setdefault(number, []).append((label, ids))
            for number, entries in mapped.items():
                _, shelf, name = names[number - 1]
                own = shelf or name
                exact = {label for label, _ in entries if frozenset(_label_words(label)) in (shelf, name)}
                for label, ids in entries:
                    if exact and label not in exact and not own <= _label_words(label):
                        continue
                    out[number].extend(ids)
        else:
            for ident in self.seen_order:
                number = best(self.listing_text(ident)[:200])
                if number:
                    out[number].append(ident)
        return out

    def anchor_item(self, groups: dict[int, list[str]]) -> int | None:
        best = None
        for number, ids in groups.items():
            listed = sum(1 for i in ids if self.accepted_series(i)[0])
            if listed and listed * 2 >= len(ids) and (best is None or listed > best[0]):
                best = (listed, number)
        if not best or not any(self.part_series(i) for n, ids in groups.items() if n != best[1] for i in ids):
            return None
        return best[1]

    @staticmethod
    def name_set(text: str) -> set[str]:
        return {n for n in (re.sub(r"[^a-z0-9]", "", p.lower()) for p in re.split(r",|;|/|\band\b|\bor\b", text)) if len(n) >= 3}

    def model_names(self, ident: str) -> set[str]:
        text = self.listing_text(ident)
        names: set[str] = set()
        model = _MODEL_CODE.search(text)
        if model:
            names.add(re.sub(r"[^a-z0-9]", "", model.group(1).lower()))
        sold = _ALSO_SOLD.search(text)
        if sold:
            names |= self.name_set(sold.group(1))
        return names

    def works_spec(self, ident: str, deep: bool = False) -> tuple[set[str], set[str], set[str]]:
        text = self.listing_text(ident)
        names: set[str] = set()
        families: set[str] = set()
        excluded: set[str] = set()
        for match in _WORKS_WITH.finditer(text):
            clause = match.group(1)
            exc = _EXCEPT.search(clause)
            if exc:
                excluded |= self.name_set(exc.group(1))
                clause = clause[: exc.start()]
            families |= _families(clause)
            names |= self.name_set(clause)
        code = _more_code(text)
        if code and deep:
            self.more_names(ident, code)
            page = (self.set_pages.get(ident) or ("", ""))[1]
            if page and page != ident:
                more_names, more_families, more_excluded = self.works_spec(page)
                names, families, excluded = names | more_names, families | more_families, excluded | more_excluded
        return names, families, excluded

    def works_names(self, ident: str, deep: bool = False) -> set[str]:
        names, families, _ = self.works_spec(ident, deep)
        return names | families

    def works_with(self, ident: str, device: set[str], deep: bool = False) -> bool:
        names, families, excluded = self.works_spec(ident, deep)
        for name in device:
            if name in excluded:
                continue
            if name in names or any(re.fullmatch(family, name) for family in families):
                return True
        return False

    def more_names(self, ident: str, code: str) -> set[str]:
        if ident in self.set_ext:
            return self.set_ext[ident]
        self.set_ext[ident] = set()
        pattern = re.compile(r"\[\s*" + re.escape(code) + r"\s*\]", re.I)

        def holder() -> str | None:
            return next((i for i in self.seen_order if i != ident and pattern.search(self.listing_text(i))), None)

        page = holder()
        if not page and self.left() > 2 * len(self.items) + 6:
            self.search_rows(code)
            page = holder()
        names: set[str] = set()
        if page:
            tool = self.offer_tool()
            target = self.single_target(tool)
            if tool and target and page not in self.opened and self.left() > 2 * len(self.items) + 5:
                self.open_page(tool, target, page)
            names = self.works_names(page)
            self.set_pages[ident] = (code, page)
        self.set_ext[ident] = names
        self.trace.append({"kind": "set_more", "listing": ident, "code": code, "page": page, "names": sorted(names)})
        return names

    def model_anchor(self, groups: dict[int, list[str]]) -> int | None:
        best = None
        for number, ids in groups.items():
            named = sum(1 for i in ids if self.model_names(i) and not self.works_names(i))
            if named and named * 2 >= len(ids) and (best is None or named > best[0]):
                best = (named, number)
        if not best or not any(self.works_names(i) for n, ids in groups.items() if n != best[1] for i in ids):
            return None
        return best[1]

    def tie_of(self, number: int, need: str) -> int:
        if need in self.order_rules:
            return self.order_rules[need]
        if (not re.search(r"(?i)\b(?:works? with|fits?|compatible|goes with|matches)\b", need)
                or re.search(r"(?i)\b(?:sold by|page|listing|shop|store|seller)\b", need)
                or not _CHOSEN.search(need)):
            return 0
        named = _stem(need.split(":", 1)[-1])
        for other, item in enumerate(self.items, 1):
            if other != number and {w for w in _stem(item.name) if len(w) >= 3} & named:
                return other
        return 0

    def order_item(self, need: str, spec: dict[str, Any]) -> int:
        if spec.get("seller") or spec.get("features") or re.search(r"(?i)\b(?:sold by|page|listing|shop|store|seller)\b", need):
            return 0
        if not _CHOSEN.search(need):
            return 0
        own = next((n for n, m in self.refs if m == need), 0)
        named = _stem(str(spec.get("kind") or "") + " " + need.split(":", 1)[-1])
        for number, item in enumerate(self.items, 1):
            if number != own and {w for w in _stem(item.name) if len(w) >= 3} & named:
                return number
        return 0

    def item_needs(self, number: int) -> list[str]:
        item = self.items[number - 1]
        loose = self.unrecorded()
        return [self.need_text(m)
                + (" [from the shopper's answer: met only by an attribute naming this property]" if m in item.told and m not in self.ref_constraint else "")
                + (" (no listing records this, so it decides nothing)" if m in loose else "")
                for m in item.musts + item.told if m not in item.dropped and not self.tie_of(number, m)]

    def fit_ids(self, number: int, ids: list[str]) -> list[str]:
        if not ids:
            return []
        item = self.items[number - 1]
        chosen = [(n, i) for n, i in sorted(self.chosen.items()) if n != number and i] if any(_CONDITION.search(n) for n in self.item_needs(number)) else []
        key = (number, tuple(ids), "; ".join(self.item_needs(number)), tuple(chosen))
        if key in self.fit_cache:
            return [i for i in self.fit_cache[key] if not self.sold_out(i)]
        records = [f"{i}: " + ("SOLD OUT. " if self.sold_out(i) else "") + _dump(_clip(self.full.get(i) or self.rows.get(i), 200))[:1400] for i in ids]
        prompt = (FIT + "\n\nREQUEST:\n" + self.request + f"\n\nITEM: {item.name}" + (f" (store label: {item.shelf})" if item.shelf else "")
                  + "\nREQUIREMENTS OF THIS ITEM: " + ("; ".join(self.item_needs(number)) or "none beyond the kind of product")
                  + "\n\nLISTING RECORDS:\n" + "\n".join(records))
        if chosen:
            prompt += ("\n\nLISTINGS CHOSEN FOR THE OTHER ITEMS (a requirement's condition on another item is judged on these):\n"
                       + "\n".join(f"item {n} ({self.items[n - 1].name}) {i}: " + _dump(_clip(self.full.get(i) or self.rows.get(i), 200))[:1400] for n, i in chosen))
        got = self.ask_json(prompt, [LIGHT, HEAVY], 3000) or {}
        fits = [str(i) for i in _items(got.get("fits")) if str(i) in ids]
        if got:
            self.fit_cache[key] = fits
        self.trace.append({"kind": "set_fit", "item": number, "read": ids, "fits": fits, "reason": str(got.get("reason") or "")[:160]})
        return fits

    def item_fits(self, number: int, anchor_id: str | None, accepted: set[str], names: set[str] | None = None,
                  takes: set[str] | None = None) -> tuple[list[str], str]:
        item = self.items[number - 1]
        pointers = {need for _, need in self.refs}
        waiting = [m for m in item.musts + item.told if m not in item.dropped and m in pointers and m not in self.ref_values
                   and not self.tie_of(number, m) and m not in self.chased]
        if waiting:
            return [], f"item {number} ({item.name}) waits for the listing its requirement points at: {waiting[0][:120]}"
        names = names or self.fixed_codes.get(number) or None
        cheapest = self.brief.get("pick") == "cheapest"
        each = self.brief.get("budget_scope") == "each" and isinstance(self.budget, (int, float))
        reserve = 2 * len(self.items) + 8
        noun = " ".join(item.name.split()[:3])
        queries = [f"{noun} {s}" for s in sorted(accepted)][:3] if anchor_id and accepted else []
        if anchor_id and names and not accepted:
            model = _MODEL_CODE.search(self.listing_text(anchor_id))
            if model:
                queries += [model.group(1), f"{noun} {model.group(1)}"]
            sold = _ALSO_SOLD.search(self.listing_text(anchor_id))
            if sold:
                queries += [alias.strip() for alias in re.split(r",|\band\b", sold.group(1)) if alias.strip()][:2]
        live = [need for need in item.musts + item.told if need not in item.dropped and not self.tie_of(number, need)]
        values = 0
        for need in live:
            value = self.plain_value(need, figures=True)
            if not value:
                limit = _LIMIT.search(need)
                value = f"{limit.group(2)} {limit.group(3) or ''}".strip() if limit else ""
            if value and values < 2:
                values += 1
                queries.append(value if (self.shelf_filter(item.shelf) or noun.lower() in value.lower()) else f"{value} {noun}")

        wanted = {w for need in live for w in re.findall(r"[a-z0-9]{2,}", self.plain_value(need, figures=True).lower()) if w not in ("an", "of", "or", "in", "on", "at", "to", "the", "and", "for", "with")}
        limits = [m for m in (_LIMIT.search(need) for need in live) if m]

        if takes:
            queries = [s for s in sorted(takes)][:2] + queries

        def pool() -> list[str]:
            ids = [i for i in self.item_rows().get(number, []) if self.buyable(i)]
            if takes:
                ids = [i for i in ids if self.accepted_series(i)[0] & takes]
            if anchor_id and accepted:
                ids = [i for i in ids if not self.part_series(i) or self.part_series(i) in accepted]
            if names:
                ids = [i for i in ids if self.works_with(i, names) or _more_code(self.listing_text(i))]
            if each:
                ids = [i for i in ids if (self.price_of(i) if self.price_of(i) is not None else 0.0) <= float(self.budget)]
            rank = {i: n for n, i in enumerate(ids)}

            def key(i: str) -> tuple[int, float]:
                text = self.listing_text(i).lower()
                named = sum(1 for w in wanted if re.search(r"\b" + re.escape(w) + r"\b", text)) if wanted else 0
                named += sum(1 for m in limits if _within(text, m))
                price = self.price_of(i) if self.price_of(i) is not None else float("inf")
                return (-named, price if cheapest else rank[i])

            return sorted(ids, key=key)

        if len(pool()) < 4:
            queries.append(noun)
        for query in dict.fromkeys(queries):
            if self.left() <= reserve + 3 or self.finished:
                break
            if query.lower() not in self.set_queries:
                self.set_queries.add(query.lower())
                self.search_rows(query, item.shelf or None)
        ids = pool()
        if names and len(ids) < 3 and self.left() > reserve + 6:
            label = self.shelf_label(number)
            if label and label.lower() not in self.set_queries:
                self.set_queries.add(label.lower())
                self.browse_rows(label, 2)
                ids = pool()
        if not ids:
            kind = [i for i in self.item_rows().get(number, []) if self.buyable(i)]
            if not kind:
                return [], f"item {number} ({item.name}): no listing of that kind identified yet"
            rule = (f"is in the part series {', '.join(sorted(accepted))} that {anchor_id} takes" if anchor_id and accepted
                    else f"works with {anchor_id or ', '.join(sorted(names or []))}" if names else "is within the budget")
            return [], f"item {number} ({item.name}): none of the {len(kind)} listings of that kind found {rule}"
        unread = [i for i in ids if i not in self.full and i not in self.fetched][: max(0, min(4, (self.left() - reserve) // 2))]
        if unread:
            self.fill_facts(unread, reserve)
        read = [i for i in ids if i in self.full or i in self.fetched][:8]
        fits = self.fit_ids(number, read)
        more = [i for i in ids if i not in self.full and i not in self.fetched][: max(0, min(4, (self.left() - reserve) // 2))]
        if not fits and more and self.left() - reserve >= 10 and time.time() - self.t0 < 200:
            self.fill_facts(more, reserve)
            read = [i for i in ids if (i in self.full or i in self.fetched) and i not in read][:8]
            fits = self.fit_ids(number, read)
        if isinstance(self.budget, (int, float)) and not each and fits:
            kept = [i for i in fits if self.price_of(i) is None or self.price_of(i) * max(1, item.qty) <= float(self.budget)]
            fits = kept or fits
        if cheapest:
            fits.sort(key=lambda i: self.price_of(i) if self.price_of(i) is not None else float("inf"))
        clean = [i for i in fits if not self.key_conflict(i, live)]
        fits = clean or fits
        if names and fits:
            fits = [i for i in fits[:3] if self.works_with(i, names, deep=True)]
        why = "" if fits else (f"item {number} ({item.name}): none of the {len(read)} listings read meets its requirements"
                               + (f" among the part series {', '.join(sorted(accepted))} that {anchor_id} takes" if anchor_id and accepted else "")
                               + (f" among the listings that work with {anchor_id or ', '.join(sorted(names or []))}" if names else ""))
        return fits, why

    def plan_set(self) -> tuple[dict[int, str], list[str]]:
        if self.finished or time.time() - self.t0 > 215:
            return {}, []
        groups = self.item_rows()
        tied = self.request_ties()
        explicit: set[int] = set()
        for n in range(1, len(self.items) + 1):
            for m in self.items[n - 1].musts + self.items[n - 1].told:
                other = self.tie_of(n, m) if m not in self.items[n - 1].dropped else 0
                if other:
                    explicit |= {n, other}
        anchor = self.anchor_item(groups)
        scheme = "series" if anchor else ""
        if anchor is None:
            anchor = self.model_anchor(groups)
            scheme = "models" if anchor else ""
        numbers = list(range(1, len(self.items) + 1))
        if anchor is None:
            plan: dict[int, str] = {}
            notes: list[str] = []
            self.chosen = {}
            for number in numbers:
                ties = [self.tie_of(number, m) for m in self.items[number - 1].musts + self.items[number - 1].told if self.tie_of(number, m)]
                if ties:
                    notes.append(f"item {number} ({self.items[number - 1].name}): its tie to item {ties[0]} is decided by codes on the listings, and none were found")
                    continue
                fits, why = self.item_fits(number, None, set())
                if fits:
                    plan[number] = fits[0]
                    self.chosen[number] = fits[0]
                elif why:
                    notes.append(why)
            return plan, notes
        heads, why = self.item_fits(anchor, None, set())
        if scheme == "series" and tied and len(self.items) >= 2:
            wanted = self.series_wanted(anchor, numbers, explicit)
            if wanted:
                takes_wanted = [h for h in heads if self.accepted_series(h)[0] & wanted]
                if not takes_wanted and self.left() > 2 * len(self.items) + 14:
                    more, _ = self.item_fits(anchor, None, set(), takes=wanted)
                    takes_wanted = [h for h in more if h not in heads]
                    heads = takes_wanted + heads
                    self.trace.append({"kind": "series_anchor", "wanted": sorted(wanted), "found": takes_wanted[:4]})
                heads = sorted(heads, key=lambda h: (not (self.accepted_series(h)[0] & wanted), heads.index(h)))
        if not heads:
            return {}, [why] if why else []
        if scheme == "series" and self.brief.get("pick") != "cheapest" and len(heads) > 2 and not (tied and len(self.items) >= 2):
            seen = {n: [i for i in groups.get(n, []) if self.buyable(i)] for n in numbers if n != anchor}

            def cover(head: str) -> int:
                takes = self.accepted_series(head)[0]
                return sum(1 for ids in seen.values() if any(not self.part_series(i) or self.part_series(i) in takes for i in ids))

            heads = sorted(heads, key=lambda h: -cover(h))
        best: tuple[dict[int, str], list[str]] = ({}, [])
        for head in heads[: (3 if tied else 2)]:
            if self.finished or time.time() - self.t0 > 215:
                break
            names: set[str] = set()
            if scheme == "series":
                accepted, see, model = self.accepted_series(head)
                if see:
                    accepted = accepted | self.extension_series(head, see, model)
            else:
                accepted, names = set(), self.model_names(head)
            plan = {anchor: head}
            self.chosen = {anchor: head}
            notes = []
            for number in numbers:
                if number == anchor:
                    continue
                free = bool(explicit) and number not in explicit
                fits, why = self.item_fits(number, None, set()) if free else self.item_fits(number, head, accepted, names)
                if fits:
                    plan[number] = fits[0]
                    self.chosen[number] = fits[0]
                elif why:
                    notes.append(why)
            if len(plan) > len(best[0]):
                best = (plan, notes)
                self.set_fit = (head, scheme, set(accepted) if scheme == "series" else set(names))
                self.set_soft = not tied
            if len(plan) == len(self.items):
                break
        if best[0] and len(best[0]) < len(self.items) and not tied and not self.finished:
            plan, notes = dict(best[0]), list(best[1])
            for number in numbers:
                if number in plan or not any(f"item {number} " in n and "meets its requirements" in n for n in notes):
                    continue
                fits, why = self.item_fits(number, None, set())
                if fits:
                    plan[number] = fits[0]
                    notes.append(f"item {number} ({self.items[number - 1].name}): no compatible listing read meets its requirements, and the request does not tie it to the other items, so any fit serves")
            best = (plan, notes)
        return best

    def series_wanted(self, anchor: int, numbers: list[int], explicit: set[int]) -> set[str]:
        sets: list[set[str]] = []
        for number in numbers:
            if number == anchor or (explicit and number not in explicit) or self.left() <= 2 * len(self.items) + 14:
                continue
            fits, _ = self.item_fits(number, None, set())
            series = {self.part_series(i) for i in fits if self.part_series(i)}
            if not series:
                return set()
            sets.append(series)
        if not sets:
            return set()
        common = set.intersection(*sets)
        wanted = common or min(sets, key=len)
        self.trace.append({"kind": "series_wanted", "sets": [sorted(x) for x in sets], "wanted": sorted(wanted)})
        return wanted

    def build_set(self, why: str) -> str:
        try:
            return self.build_set_once(why)
        except Exception as exc:
            self.trace.append({"kind": "set_error", "error": f"{type(exc).__name__}: {exc}"[:300]})
            self.set_tries = 2
            return ""

    def build_set_once(self, why: str) -> str:
        if (len(self.items) < 2 or self.set_tries >= 2 or self.finished or not self.roles.get("search") or not self.roles.get("cart")
                or not self.roles.get("buy") or self.left() < 3 * len(self.items) + 4 or self.elapsed() > EXTRAS_BY or self.cost > self.hard):
            return ""
        if len(self.items) == 2 and self.max_steps < 50:
            groups = self.item_rows()
            if self.anchor_item(groups) is None and self.model_anchor(groups) is None:
                return ""
        self.set_tries += 1
        said: list[str] = []
        for attempt in range(2):
            plan, notes = self.plan_set()
            if plan and len(plan) >= len(self.best_plan):
                self.best_plan = dict(plan)
            lines = "; ".join(f"item {n} ({self.items[n - 1].name}): {i}" + (f" at {self.price_of(i):g}" if self.price_of(i) is not None else "")
                              for n, i in sorted(plan.items()))
            self.trace.append({"kind": "set_plan", "why": why, "attempt": attempt, "plan": plan, "notes": notes[:6]})
            if len(plan) < len(self.items):
                if plan or notes:
                    said.append("[Orochi searched each item by code] Fits so far: " + (lines or "none") + ". " + " ".join(notes[:4])
                                + " Search those items further (their requirements and the fit rule in the ledger), cart one fitting listing per item, then check out.")
                break
            ids = [plan[n] for n in sorted(plan)]
            problems = self.check_cart(ids, self.stop - 0.004)
            if problems:
                said.append("[Orochi planned a set by code: " + lines + "] " + problems)
                break
            if not self.verdicts.get(self.check_key(ids)) or self.finished:
                said.append("[Orochi planned a set by code] " + lines + ". Cart each at its item's quantity and check out.")
                break
            self.lead_items = {str(n): [i] for n, i in plan.items()}
            self.leads = list(ids)
            self.confirmed = ids[0]
            self.completed = False
            done = self.complete_cart()
            if not done:
                said.append("[Orochi planned a set by code and the cart check passed it] " + lines + ". Cart each at its item's quantity and check out.")
                break
            said.append("[Orochi planned the whole set by code (" + lines + ") and the cart check passed it] " + done)
            lost = [i for i in ids if self.sold_out(i)]
            if (self.finished or not lost or attempt or self.left() < 3 * len(self.items) + 2 or time.time() - self.t0 > 200
                    or self.cost > self.hard):
                break
            said.append(f"[Orochi] {', '.join(lost)} sold out at the checkout, so the set is planned again without it.")
        return "\n".join(said)

    def line_problems(self) -> str:
        cat_key = self.category_key()
        ids = [i for i in self.cart_ids() if self.cart_lines.get(i, 1) > 0]
        if not ids or not cat_key:
            return ""
        owner = {i: self.label_owner(str((self.rows.get(i) or {}).get(cat_key) or "")) for i in ids}
        known = {self.label_owner(str(r.get(cat_key) or "")) for r in self.rows.values()} - {None}
        faults: list[str] = []
        wrong: dict[str, int] = {}
        for n, item in enumerate(self.items, 1):
            if n not in known:
                continue
            lines = [i for i in ids if owner[i] == n]
            if not lines:
                faults.append(f"item {n} ({item.name}) has no line in the cart")
            elif len(lines) > 1:
                faults.append(f"item {n} ({item.name}) has {len(lines)} lines ({', '.join(lines)}); keep one")
            elif self.cart_lines.get(lines[0], 1) != item.qty:
                faults.append(f"{lines[0]} is held at {self.cart_lines.get(lines[0], 1)} unit(s); item {n} ({item.name}) wants {item.qty}")
                wrong[lines[0]] = item.qty
        if len(known) == len(self.items):
            faults += [f"{i} is not a kind the shopper asked for: take it out" for i in ids if owner[i] is None]
        faults += [f"{i} is sold out" for i in ids if self.sold_out(i)]
        prices = {i: self.price_of(i) for i in ids}
        if isinstance(self.budget, (int, float)) and all(p is not None for p in prices.values()):
            if self.brief.get("budget_scope") == "each":
                faults += [f"{i} costs {prices[i]:g}, over the {self.budget:g} limit for each item" for i in ids if self.budget < prices[i] <= 10 * self.budget]
            else:
                total = sum(prices[i] * self.cart_lines.get(i, 1) for i in ids)
                if self.budget + 1e-9 < total <= 10 * self.budget:
                    faults.append(f"the total is {total:g}, over the budget of {self.budget:g}")
        if not faults:
            return ""
        if wrong and len(wrong) == len(faults):
            done = self.fix_lines(wrong)
            if done:
                self.trace.append({"kind": "line_fix", "calls": done})
                return "Orochi set the cart's quantities itself: " + "; ".join(done) + ". If the store refused any of these, fix those lines; then check out."
        self.trace.append({"kind": "line_hold", "faults": faults[:6]})
        return "The cart does not match the request yet: " + "; ".join(faults[:6]) + ". Fix those lines (the cart tool's own way to add or take off units), then check out."

    def fix_lines(self, wrong: dict[str, int]) -> list[str]:
        name = self.roles.get("cart")
        if not name:
            return []
        lines = {i: wrong.get(i, self.cart_lines.get(i, 1)) for i in self.cart_ids() if self.cart_lines.get(i, 1) > 0}
        calls = self.cart_calls(lines, [])
        if not calls or len(calls) + 3 > self.left():
            return []
        done = []
        for args in calls:
            faults: list[str] = []
            args = _shape(args, self.schemas.get(name), "args", faults)
            if faults:
                break
            reply = self.act(name, args)
            if reply is None or self.finished:
                break
            self.note_cart(name, args)
            self.observe(name, reply.get("observation"))
            done.append(_dump(args) + " -> " + self.show(reply.get("observation"))[:200])
        return done

    def compat_problems(self) -> str:
        try:
            return self.compat_problems_once()
        except Exception as exc:
            self.trace.append({"kind": "set_error", "error": f"{type(exc).__name__}: {exc}"[:300]})
            return ""

    def compat_problems_once(self) -> str:
        cart = self.cart_ids()
        if len(cart) < 2:
            return ""
        models = [i for i in cart if self.model_names(i) and not self.works_names(i)]
        wanting = [i for i in cart if i not in models and (self.works_names(i) or _more_code(self.listing_text(i)))]
        if len(models) == 1 and wanting and not self.accepted_series(models[0])[0]:
            names = self.model_names(models[0])
            bad = [i for i in wanting if not self.works_with(i, names, deep=True)]
            if not bad:
                return ""
            self.set_fit = (models[0], "models", names)
            self.trace.append({"kind": "compat_hold", "anchor": models[0], "names": sorted(names), "bad": bad})
            return (f"[Orochi] By the listings' own words, {models[0]} is sold as {', '.join(sorted(names))}, and " + ", ".join(bad)
                    + " names none of those (with any page its text names for more), so it does not fit. Replace it with a listing that works with"
                    " that model and meets its item's requirements, then check out.")
        anchors = [(i, self.accepted_series(i)) for i in cart]
        anchors = [(i, info) for i, info in anchors if info[0]]
        if len(anchors) != 1:
            return ""
        anchor, (accepted, see, model) = anchors[0]
        accepted = set(accepted) | (self.extension_series(anchor, see, model) if see else set())
        bad = [i for i in cart if i != anchor and self.part_series(i) and self.part_series(i) not in accepted]
        if not bad:
            return ""
        self.set_fit = (anchor, "series", accepted)
        self.trace.append({"kind": "compat_hold", "anchor": anchor, "accepted": sorted(accepted), "bad": bad})
        return (f"[Orochi] By the listings' own codes, {anchor} takes part series {', '.join(sorted(accepted))}"
                + (f" (with what its page for the rest, part {see}, adds)" if see else "") + "; "
                + ", ".join(f"{i} carries series {self.part_series(i)}" for i in bad)
                + ", so it does not fit. Replace each such line with a listing of an accepted series (or one with no series code) that meets its item's requirements, then check out.")

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
        if name not in {self.roles.get(r) for r in ("cart", "open", "offer", "compare")} - {None} or not self.id_key:
            return args, ""
        props = (self.schemas.get(name) or {}).get("properties") or {}
        free = [k for k, v in props.items() if isinstance(v, dict) and ("string" in _kinds(v) or "array" in _kinds(v)) and "enum" not in v]
        if len(free) != 1:
            return args, ""
        known = name == self.roles.get("cart") or (name, free[0]) in self.id_args

        def fix(value: Any) -> tuple[Any, str]:
            if isinstance(value, str) and value.strip() in self.rows:
                return value.strip(), ""
            code = value.strip().strip("[]() ") if isinstance(value, str) else ""
            if code and re.fullmatch(r"[A-Za-z]{1,6}[- ]?\d{2,8}(?:-\d{1,8})?", code):
                mark = re.compile(r"[\[(]\s*" + re.escape(code) + r"\s*[\])]", re.I)
                holders = [i for i in self.rows if mark.search(self.listing_text(i))]
                if len(holders) == 1:
                    self.trace.append({"kind": "code_id", "code": code, "id": holders[0]})
                    return holders[0], ""
                if len(holders) > 1:
                    return value, f"{code} is a part code printed in several listings' titles ({', '.join(holders[:6])}); pass one listing id (their {self.id_key} field)"
            if not known or not isinstance(value, str) or _id_like(value, True) or len(value.strip()) < 8:
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
            self.cart_empty = not self.cart_lines
            return
        self.cart_empty = False
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

    def show(self, observation: Any, limit: int = SHOW_CHARS) -> str:
        text = _dump(_clip(observation, limit))
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
        spec = ((self.schemas.get(self.roles.get("compare") or "") or {}).get("properties") or {}).get(self.compare_target() or "") or {}
        if isinstance(spec.get("maxItems"), int):
            return spec["maxItems"]
        if self.compare_bad:
            return max(2, min(self.compare_bad - 1, max(self.compare_ok, 2)))
        return max(4, self.compare_ok)

    def join_opens(self, calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        opener, joiner, target = self.roles.get("open"), self.roles.get("compare"), self.single_target(self.roles.get("open"))
        many = self.compare_target()
        if not opener or not joiner or not target or not many or opener == joiner:
            return calls
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
                out.append((cid, "Shown in the side-by-side result above: Orochi opened these listings together in one step."))
                continue
            fn = call.get("function") or {}
            name = fn.get("name")
            raw = fn.get("arguments")
            if self.finished:
                out.append((cid, "Not run: the episode is over."))
                continue
            if name == self.roles.get("ask"):
                out.append((cid, "Not run (no step used): Orochi's interviewer handles every message to the shopper."))
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
                trimmed = f"\n[Orochi] Sent the first {room} listings only: this store shows at most {room} side by side. Not shown yet: {', '.join(map(str, args[many][room:]))}."
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
                    trimmed += f"\n[Orochi] Left out {', '.join(str(i) for i in args[many] if i not in keep)}: shown in full already, nothing changed since."
                    args = {**args, many: keep}
                    mark = name + _dump(args)
            if ending and self.left() > 2:
                if self.is_buy(name, args):
                    late = self.elapsed() > HOLDS_BY
                    if not late and not self.bargain_done and self.brief.get("pick") == "cheapest" and len(self.items) <= 2:
                        try:
                            bargain = "\n".join(t for t in (self.bargain_pass(n) for n in range(1, len(self.items) + 1)) if t)
                        except Exception as exc:
                            bargain = ""
                            self.trace.append({"kind": "set_error", "error": f"bargain {type(exc).__name__}: {exc}"[:200]})
                        if bargain:
                            out.append((cid, "Not run yet. " + bargain + "\nCart now: " + self.cart_text()))
                            continue
                    wider = self.wider_hold() if not late else ""
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
                            change = self.observe(self.roles["cart_view"], self.cart_view)
                            if change:
                                self.queued = []
                                out.append((cid, "Not run yet (no step used). Looking at the cart showed a change: " + change + "\nCart now: " + self.cart_text() + "\nApply the change, fix the cart if needed, then check out."))
                                continue
                    swap = self.swap_hint()
                    held = [i for i in self.cart_ids() if i in self.gone] if self.held < 2 else []
                    if swap or held:
                        self.held += 0 if swap else 1
                        out.append((cid, "Not run yet (no step used). " + (swap or f"{', '.join(held)} in the cart is sold out: replace that line first.") + "\nCart now: " + self.cart_text()))
                        continue
                    if not late and self.chase_due():
                        chased = self.chase_all()
                        if chased:
                            out.append((cid, "Not run yet (no step used). " + chased + "\nCheck the cart against it, then check out.\nCart now: " + self.cart_text()))
                            continue
                    chain, hold = self.trail() if not late else ("", False)
                    if hold:
                        out.append((cid, "Not run yet (no step used). " + chain + "\nCart now: " + self.cart_text()))
                        continue
                    compat = self.compat_problems() if not late and self.compat_holds < 2 and len(self.items) >= 2 else ""
                    if compat:
                        self.compat_holds += 1
                        out.append((cid, "Not run yet (no step used). " + compat + "\nCart now: " + self.cart_text()))
                        continue
                    lines_fault = ""
                    if not late and self.line_holds < 2 and self.left() > self.tail() + 2:
                        try:
                            lines_fault = self.line_problems()
                        except Exception as exc:
                            self.trace.append({"kind": "set_error", "error": f"{type(exc).__name__}: {exc}"[:200]})
                    if lines_fault:
                        self.line_holds += 1
                        head = "Not run yet. " if lines_fault.startswith("Orochi set") else "Not run yet (no step used). "
                        out.append((cid, head + lines_fault + "\nCart now: " + self.cart_text()))
                        continue
                    read = (self.fill_facts(self.cart_ids(), self.tail() + 3) + self.cheaper_rows()) if not late else []
                    verdict = self.check_cart(None, self.stop - 0.004) if not late else ""
                    if verdict:
                        seen = ("Orochi read these listings' attributes for the check (one step each):\n" + "\n".join(read) + "\n") if read else ""
                        out.append((cid, "Not run yet. " + chain + seen + verdict + "\nCart now: " + self.cart_text()))
                        continue
                    if reread and self.cost > self.soft:
                        out.append((cid, "Not run yet (no step used). Orochi re-read the cart's listings just now:\n" + reread + "\nCheck out again when every line still shows stock and a price inside the budget; otherwise replace that line first."))
                        continue
                else:
                    if self.left() > 5 and self.chase_due():
                        chased = self.chase_all()
                        if chased:
                            out.append((cid, "Not run yet (no step used). " + chased + "\nSearch for a listing that meets it before ending without a purchase."))
                            continue
                    if self.left() > 5:
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
                            self.check_cart([first], self.hard)
                            if self.verdicts.get(self.check_key([first])) and first in self.leads:
                                self.confirmed = first
                                self.trace.append({"kind": "refuse_blocked", "confirmed": first})
                                out.append((cid, f"Not run yet (no step used). An independent check of {first}'s records says it meets every requirement. Add it to the cart at the requested quantity and check out; end without a purchase only if its record clearly breaks a requirement."))
                                continue
            if self.is_refuse(name, args) and self.terms_eased and not self.eased_block and self.left() > 3:
                self.eased_block = True
                self.trace.append({"kind": "eased_decline_held"})
                done = self.rescue() if self.left() <= max(6, self.max_steps // 3) else ""
                if done:
                    out.append((cid, done + (" [episode finished]" if self.finished else "")))
                    continue
                out.append((cid, "Not run yet (no step used). The shopper has dropped a requirement, so the reason to end without a purchase is gone: "
                            "search again for each requested item without the dropped requirement (the ledger shows what is still in force), "
                            "open the listings that may fit, cart the best one at the requested quantity and check out. "
                            "End without a purchase only if, even without it, no listing meets the remaining requirements."))
                continue
            if self.is_refuse(name, args) and self.left() <= 2 and self.roles.get("buy") and not self.last_bought:
                ids = self.cart_ids()
                plain = "|".join(sorted(ids))
                passed = any(v for k, v in self.verdicts.items() if k.split("#")[0] == plain)
                total = sum((self.price_of(i) or 0.0) * (self.cart_lines.get(i, 1) or 1) for i in ids)
                within = not isinstance(self.budget, (int, float)) or total <= float(self.budget)
                if ids and passed and within and not any(self.sold_out(i) for i in ids) and len(ids) >= len(self.items):
                    faults: list[str] = []
                    buy_args = _shape(dict(self.brief.get("buy_args") or {}), self.schemas.get(self.roles["buy"]), "args", faults)
                    if not faults:
                        self.last_bought = True
                        self.trace.append({"kind": "last_step_buy", "cart": ids})
                        name, args = self.roles["buy"], buy_args
                        ending = True
            if self.is_refuse(name, args) and self.left() <= 5 and not self.late_held:
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
            if self.is_refuse(name, args) and len(self.items) >= 2 and self.set_tries < 2:
                built = self.build_set("before ending without a purchase")
                if built:
                    out.append((cid, ("" if self.finished else "Not run yet (no step used). ") + built + (" [episode finished]" if self.finished else "")))
                    continue
            if self.is_refuse(name, args) and 2 <= self.left() <= max(5, self.max_steps // 3):
                done = self.rescue()
                if done:
                    out.append((cid, done + (" [episode finished]" if self.finished else "")))
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
                    trimmed = "\n[Orochi] " + said + trimmed
            if name == self.roles.get("search"):
                fitted = self.fit_search_mode(args)
                if fitted != args:
                    trimmed += f"\n[Orochi] Sent with {_dump({k: fitted.get(k) for k in self.mode_args()})}: the store turned back this search's mode with these filters earlier."
                    args = fitted
            reply = self.act(name, args)
            if reply is None:
                out.append((cid, "The call failed (network or server). You may try again."))
                continue
            observation = reply.get("observation")
            if name == self.roles.get("search"):
                self.note_search(args, observation)
            if many and isinstance(args.get(many), list) and len(args[many]) > 1:
                shown = _dump(observation)
                if not any(str(i) in shown for i in args[many]):
                    self.compare_bad = min(self.compare_bad or 999, len(args[many]))
                    trimmed += f"\n[Orochi] The side-by-side view showed nothing for {len(args[many])} listings: send at most {self.compare_room()} per call."
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
            if name == self.roles.get("search"):
                self.learn_search(args, observation)
                self.learn_fields(observation)
                self.queries.append(self.search_words(args))
            plain = {tool for job, tool in self.roles.items() if job != "search"}
            paged = self.more_pages(name, args, observation) if name not in plain and not ending else ""
            if self.single_target(name) and name not in (self.roles.get("cart"), self.roles.get("buy"), self.roles.get("refuse"), self.roles.get("ask")):
                listing = self.ident(observation) if isinstance(observation, dict) else None
                if listing:
                    self.facts_from[name] = self.facts_from.get(name, False) or bool(_fact_block(observation))
                    if name == self.roles.get("open"):
                        self.fetched.add(listing)
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
                self.note_cart(name, args)
                self.cart_said = _dump(observation)
                self.recent = []
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
            text = self.show(observation, ROW_CHARS if name == self.roles.get("search") else SHOW_CHARS) + paged + trimmed
            if name not in plain and not ending:
                text += self.cheapest_here([observation] + self.paged_obs)
            base = text
            proof = self.evidence(observation)
            if proof:
                text += "\n" + proof
            if name == self.roles.get("cart") and any(_count(v) == 0 for v in args.values() if not isinstance(v, str)):
                if any(isinstance(v, str) and v in self.rows and _in_rows(observation, v) for v in args.values()):
                    text += "\n[Orochi] A count of 0 left that line in the cart: this tool adds the count to the line, so a negative count takes units off."
            self.note_sold = False
            lines = self.loose_lines(observation)
            if lines:
                meaning = self.take_note(lines)
                if meaning:
                    text += "\n[note read] " + meaning + ". Apply this change to your picks before checking out."
            if self.is_buy(name, args) and not reply.get("finished") and (self.note_sold or any(line in self.spoken(observation) for line in self.sold_lines)):
                seen = self.find_sold()
                if seen:
                    text += "\n[Orochi re-read the cart's lines after the refusal]\n" + seen
            swap = self.swap_hint()
            if swap:
                text += "\n" + swap
            if self.queued:
                text += "\n[notes seen on Orochi's own calls] " + " | ".join(self.queued)
                self.queued = []
            if name in readers:
                self.read_cache[mark] = (self.version, self.step, base)
            if reply.get("finished"):
                text += " [episode finished]"
            out.append((cid, text))
        return out

    def finish(self, again: bool = False) -> str:
        if self.finished or self.broke or self.left() <= 0:
            return ""
        buy, refuse = self.roles.get("buy"), self.roles.get("refuse")
        fresh = self.cart_view is not None and self.cart_seen_at >= self.cart_at
        holding = bool(self.cart_ids()) if fresh else bool(self.cart_lines)
        plan = {n: i for n, i in self.best_plan.items() if not self.sold_out(i)}
        if buy and not holding and self.roles.get("cart") and self.left() >= 3 and self.buys == 0 and (
                (self.leads and len(self.items) == 1) or len(plan) == len(self.items)):
            if len(plan) == len(self.items):
                lines_now = {i: self.items[n - 1].qty for n, i in plan.items()}
                pick = next(iter(lines_now))
            else:
                pick = self.confirmed if self.confirmed in self.leads else self.leads[0]
                lines_now = {pick: self.items[0].qty}
            calls = self.cart_calls(lines_now, []) if not any(self.sold_out(i) for i in lines_now) else None
            if calls and len(calls) + 1 <= self.left():
                carted = True
                for args in calls:
                    faults = []
                    shaped = _shape(args, self.schemas.get(self.roles["cart"]), "args", faults)
                    if faults or self.act(self.roles["cart"], shaped) is None or self.finished:
                        carted = False
                        break
                if carted:
                    holding = True
                    self.trace.append({"kind": "lead_buy", "pick": pick})
        if buy and (holding or (self.cart_touched and not fresh and not self.cart_empty and self.left() >= 2)):
            faults: list[str] = []
            args = _shape(dict(self.brief.get("buy_args") or {}), self.schemas.get(buy), "args", faults)
            if not faults:
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
        if len(self.items) >= 3 or self.max_steps > 60:
            self.soft, self.hard, self.stop = (round(c * BIG_SCALE, 4) for c in (SPEND_SOFT, SPEND_HARD, SPEND_STOP))
        self.trace.append({"kind": "brief", "brief": self.brief, "caps": [self.soft, self.hard, self.stop]})
        opening = _unflatten(self.view.get("opening")) if self.view.get("opening") is not None else None
        if opening is not None:
            self.view["opening"] = opening
            self.absorb(opening)
        intro = "SHOPPER'S REQUEST:\n" + self.request
        if opening is not None:
            intro += "\n\nWHAT THE STORE SHOWS FIRST:\n" + self.show(opening)
        intro += "\n\nStart by searching for each item in its category and opening several listings in full, so the interviewer can see which properties these listings carry.\n\n" + self.status()
        self.chat = [{"role": "system", "content": GUIDE}, {"role": "user", "content": intro}]
        self.work(LOOP_CALLS, self.stop)
        if not self.finished:
            turned = self.finish(again=True)
            if turned and not self.finished and not self.broke and not self.no_credit and self.left() > 1:
                self.chat.append({"role": "user", "content": "[Orochi] Orochi tried to end the episode and " + turned + "\nApply what the store says, then finish.\n" + self.status()})
                self.work(self.loop_calls + 10, self.stop)
            if not self.finished:
                self.finish()
        self.trace.append({"kind": "end", "finished": self.finished, "steps": self.step, "cost": round(self.cost, 5), "calls": self.calls})
        return self.trace

    def work(self, loop_cap: int, spend_cap: float) -> None:
        idle = 0
        while (not self.finished and not self.broke and self.left() > 0 and self.loop_calls < loop_cap and self.cost < spend_cap
               and time.time() - self.t0 < 240):
            if self.ready_to_ask():
                summary = self.interview()
                chased = self.chase_all() if self.chase_due() else ""
                built = self.build_set("after the interview") if len(self.items) >= 2 and not self.finished else ""
                text = "\n".join(t for t in (summary, chased, built) if t)
                if text:
                    self.chat.append({"role": "user", "content": text + "\n" + self.status()})
                continue
            if self.chase_due() and self.searched and (self.asked or not self.roles.get("ask") or self.msgs_free() <= 0
                                                       or any(need in self.items[n - 1].musts for n, need in self.refs if need in self.chase_due())):
                chased = self.chase_all()
                if chased:
                    self.chat.append({"role": "user", "content": chased + "\n" + self.status()})
                    continue
            if (len(self.items) >= 2 and not self.set_tries and self.searched and not self.chase_due()
                    and (self.asked or not self.roles.get("ask") or self.msgs_free() <= 0)):
                built = self.build_set("with no interview to wait for")
                if built:
                    self.chat.append({"role": "user", "content": built + "\n" + self.status()})
                    continue
            if not self.rescued and self.buys == 0 and 2 <= self.left() <= 4 and self.searched and len(self.cart_ids()) < len(self.items):
                done = self.rescue()
                if done:
                    self.chat.append({"role": "user", "content": "[Orochi] Few steps were left and the cart lacked an item. " + done + "\n" + self.status()})
                    continue
            self.trim()
            self.loop_calls += 1
            message = self.think(self.chat, [LIGHT, SPARE], 4000, self.offered())
            if message is None:
                self.misses += 1
                if self.no_credit or self.calls >= ALL_CALLS or self.misses > 2:
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
                summary = self.interview()
                chased = self.chase_all() if self.chase_due() else ""
                text = "\n".join(t for t in (summary or "The interviewer had nothing to ask.", chased) if t)
                self.chat.append({"role": "user", "content": text + "\n" + self.status()})
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
        episode = Orochi(problem_data)
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
