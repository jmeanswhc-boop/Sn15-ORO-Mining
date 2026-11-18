"""Kyo: buys for one shopper inside an ORO store world.

A light model walks the store with the world's own tools. Code keeps the
ledger (items, needs, budget, listings seen), plans the questions to the
shopper, and stops wasted steps. A heavy model reads the request once and
checks the cart before an order and the evidence before a decline.
"""
from __future__ import annotations

import json
import random
import re
import time
from typing import Any

from src.agent.proxy_client import ProxyClient

LIGHT = "anthropic/claude-haiku-5.5"
HEAVY = "anthropic/claude-sonnet-5.5"
SPARE = "openai/gpt-6-luna"

LOOP_CALLS = 42             # model turns of the tool loop at least; a longer step cap allows more (Kyo.loop_cap)
ALL_CALLS = 60              # model calls at least; a longer step cap allows more (Kyo.call_cap)
SPEND_SOFT = 0.046
SPEND_HARD = 0.064
SPEND_STOP = 0.070          # no model call goes out above this spend: the code finishes alone (one run key pays for every task)
SPEND_LIGHT = 0.032         # above this spend the late checks (cart check, decline review, swaps) try the light model first
# each task has 300 s in all on the network: Kyo keeps its own clock from the start of agent_main
DEADLINE = 250.0            # seconds Kyo plans to use, leaving room for one slow call and the finish
CLOSING = 70.0              # with less time than this left, the tool loop stops and the episode heads for its finish
LAST_CALL = 28.0            # with less time than this left, no model call starts: the code finishes alone
QUESTION_TIME = 20.0        # the interviewer asks only while more time than CLOSING plus the slowest model call so far (this at least) is left
SINGLE_READS = 6            # listings of a single item's own category read in full before its first question
LISTS = True                # a question may name several properties of one item (a crowded answer turns it off)
LIST_NAMES = 8              # properties one question names at most
BUNDLE_NAMES = 8            # the same for an item of a request that lists several items
# waits before resending a failed model call; an episode has 300 s in all, so at most 45 s go to waiting
WAITS = (4, 10, 20, 35)
WAIT_TOTAL = 45
SHOW_CHARS = 450
OBS_LIMIT = 15000
HISTORY_LIMIT = 150000
GATE_ROUNDS = 3             # store calls the process gate may make before one checkout or ending
AUTO_PAGES = 6              # further search pages Kyo reads itself for a cheapest, most or least pick
CHEAPER_OPENS = 3           # cheaper rows of a cart line's kind whose attributes Kyo reads before a cheapest checkout
VALUE_OPENS = 3             # unread rows of a search for a newly heard value that Kyo opens itself
DECLINE_OPENS = 4           # unread rows from searches for a requirement's value that Kyo opens before an ending without a purchase
EXCLUDE_CALLS = 6           # most model calls of the exclusion check per episode (one per "not <value>" need; the heavy model only after a failed reply)

# one attempt per call (think() resends): a stuck call costs 50 s, not minutes, inside the 300 s
_llm = ProxyClient(timeout=50, max_retries=1, retry_delay=1)
_env = ProxyClient(timeout=50, max_retries=3, retry_delay=1)


GUIDE = """You are Kyo, a careful buyer working for one shopper in an online store. You can act only by calling the store's tools. The episode is graded 1 or 0: it earns 1 only when the finish is exactly right. A purchase must contain every requested item at the requested quantity, each meeting every requirement in force when you check out, inside the budget; or, when the store truly has nothing that meets everything, the episode must end with the no-purchase tool in the way the request asks.

HOW THE STORE WORKS
- Every tool call is one step and steps are limited; thinking costs nothing. Tool names, argument names and result shapes are particular to this store: read each tool's description and schema and use the exact argument names and allowed values.
- Search tools can be narrow: a category filter, a keyword mode that may only read the first words of the query, a page size of a few rows, a token or offset for the next page. Compare the reported total with the rows you have seen and page on (with the exact token or offset the store gives) when the answer depends on seeing all of them. Keyword queries work best short and value-first (the wanted value plus the product noun). A keyword search that returns no rows usually means one of its words is not on the listings: drop words (keep the product noun) or browse the item's category instead; no rows never shows that the store lacks the item. When a keyword search matches every word and returns too many rows to see, put all the item's required values in one query to narrow it. A filtered search that returns no rows usually means the filter value or the filter parameter is wrong: retry with the label exactly as result rows show it, or with another filter parameter.
- A value that appears only among a listing's attributes, not in titles, is a poor search word: then search the product noun plus words from the titles of fitting listings you have already seen.
- Some details (what other products an item suits, codes it carries) may appear only in a listing's description, which a keyword search may not read: browse the item's category, sliced by brand or a price ceiling when the store allows, and read the rows' descriptions.
- Search rows are summaries. Stores split a listing's record differently: one view can show its attributes (properties), another only its offer and description, and a side-by-side view may carry either. The ledger names the tool that shows attributes once Kyo knows it. Judge every requirement on a property from those attributes, so read them for each listing you may buy; one call that shows several listings saves steps when it shows what you need. Never open a listing again when nothing has changed since you read it.
- Questions to the shopper are handled for you by Kyo's interviewer, so you never message the shopper yourself. Its answers arrive in the conversation and are added to the ledger.
- The request can also ask for steps along the way: an approval, a message, something to check or read first. The ledger lists them. Kyo carries out the ones due just before the order or the ending itself when you call the checkout or the ending tool, so never wait for those and never end without a purchase because of them; do any others with the store's tools (product questions go through Kyo's interviewer).

WHAT COUNTS AS A FIT
- The kind of product is decided by the listing's category label: only the label that matches the requested item as a whole counts, never a neighbouring one.
- A requirement on a property is met when the listing's attribute for that property states the value. Key spelling does not matter (spaces, underscores, camel case, prefixes), and one element of a comma or slash list is enough. Values match by their words: case, singular or plural and spelling variants of the same word are fine, and a value that contains all of the required value's words counts (a value "<A> <B>" meets a requirement "<B>"). A value in other words also counts when it plainly means the same thing to a shopper (a plain synonym or an everyday rewording of the same value). A value that names a different thing does not: a near kind, a neighbouring grade or size, or a value that only shares one word with the wanted one. Units are converted before comparing numbers, limits include the edge value, and a range must lie wholly inside a limit.
- An attribute whose key names the requirement's property (under that name or a plain synonym) decides: if it says something else, the listing fails whatever other keys or the title say. A key for a different property, even a related one (a key about one part of the product says nothing about another part), decides nothing about the requirement. Kyo adds [evidence] lines after opened listings with the keys whose names match a requirement, and marks each requirement met, fails or unproven where its own reading settles it.
- When the listing has no attribute for that property, the wanted value may show in its title, in its description or under another of its attributes (for example a list attribute that includes it), though never only among its purchase choices (a list of variants on offer). The exception: when three or more listings of its category carry a key for that property while this one carries none, only its other attributes can show the value, never its title or description (what a listing fits is read by the fitting rule below); when they lack it, the listing does not meet the requirement: choose a listing whose own attributes state it. A number of another kind does not count (a figure that measures one quantity is not a figure of another quantity), nor a bare figure whose text does not say what it measures. Between listings that both meet a requirement and tie under the pick rule, prefer the one whose value is exactly the wanted value over one that adds words to it.
- When the category label itself states a requirement's value, that requirement is met by the category.
- A requirement can point at another listing instead of giving a value (the same value as, a value unlike, or a value near the one on another listing that the request describes). Resolve it first: find that listing (search its category, brand, shop or words), open its page in full with the single-item or side-by-side tool (a search row is not enough), and when that page refers you to a further listing for the property, find and open that one the same way; the last page gives the value. Open every page of that chain before checking out. When the wanted value is anything but that other listing's value, the rule excludes only listings sharing that exact value, and never end without a purchase because of it alone. When the request ties the value to a rule the store states, the value that rule gives decides: apply the rule exactly as the page states it.
- When a listing you are about to buy sends you to another page for more about it, look at that page before you buy.
- When one requested item must fit another (an accessory for a device), choose the two together from the listings' own words (a list of what it fits, a model or series code that both pages carry). When no accessory that meets its own requirements fits the device you picked, switch to another device that meets its requirements and that such an accessory fits; never end without a purchase while that switch is untried. A sentence on one listing that names another part or model code describes that other part, not this listing: a listing fits only the models its own list of what it fits names.
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

"items": a list with one entry per distinct product the shopper wants bought, each {"name": the kind of product in the request's words, "noun": the shortest noun the request itself uses for that product, in the singular (one or two words), "shelf": the store's category label for it when a tool lists labels (copy it exactly), else "", "quantity": units of that one listing (integer, 1 when not stated; a pair or a set that is one listing counts as 1), "musts": a list of firm requirements on that product, each "property: constraint" with the exact value, limit, exclusion or closed list. Name the property with the request's own word for it (a request that gives "<value>" as the "<property>" gives "<property>: <value>"); do not swap it for a neighbouring property. A must whose value is read from another listing is ONE must holding the whole description of that listing, with the relation in the request's own words ("<property>: <relation> the <property> of the <kind> <shop> sells with <feature>", where <relation> is the request's own wording for how the two values relate, copied exactly): words that describe that other listing (its shop, its kind, "with <property> <value> and <property> <value>" after it) stay inside that one must and are never musts of this product}. Never put price, budget, stock or quantity into musts. The story around the request (the person meant to use it, where it will stand, what went wrong before, the occasion behind it) is context, not a must, unless it states a property's value or limit outright, or the shopper follows it with a promise to check that point themselves ("I'll check it myself", "I'll double-check that"): such a promise marks a firm requirement, so record what it is about as a must (an occasion, a use, a type, a style: "<property>: <value>"). Never turn words that are part of what the product is called into musts: those belong to "name"; keep the request's own full wording for the kind in "name" (a department name such as words joined by "&" stays whole). Only when the request names a kind and then states a type, style or variety of it on its own ("a <kind>, and the type of <kind> is <value>", "a <kind>, <value> style") is the name the kind and the type or style a must named with the kind's own word ("<kind> type: <value>", "<kind> style: <value>"), never a bare "type" or "style". A product the request names in one phrase joined by "and" (one "I want ..." with one quantity, often the name of one store department) is ONE item; make separate items only when the request gives each product its own quantity or its own requirements.
"budget": the most the shopper will pay in total as a number in the request's own unit, or null.
"budget_unit": that unit in the request's own words, or null.
"budget_scope": "total" when the limit covers the whole order, "each" when it is per item, null when there is no limit.
"in_stock_only": true unless the request clearly accepts items that are not available.
"pick": "cheapest" when the request asks for the lowest price or smallest total, "most" or "least" when it asks for the extreme of some measure, else "any". Remarks that money is tight are not a pick rule.
"pick_measure": what is maximised or minimised for "most"/"least", else "".
"messages": how many messages the shopper allows the agent to send (integer), or null when not stated.
"one_per_message": true when the request limits each message to a single question, else false.
"steps": how many steps, moves or tool calls the request allows (integer), or null.
"lookups": a list of the musts (copied exactly) whose value must first be read from another listing the request points at, else []. A requirement that one requested item must suit or fit another item of this same order is not a lookup: keep it as a must of that item in the request's own words ("<property>: <relation> the <other item> in this order") and leave it out of lookups.
"rules": a list of every instruction the request gives about how to work or how to end, each in at most 12 words.
"wishes": a list of things the shopper would merely like.
"ask_first": a list of things the request explicitly says to check with the shopper before choosing (empty when none).
"process": a list of the steps the request requires the agent to take besides finding and buying the right items (getting the shopper's approval, telling or showing the shopper something, checking something with them, reading something first, using a particular tool), each {"step": what must be done, in at most 15 words, "when": "choose" (before the products are chosen), "order" (right before checking out) or "end" (just before ending without a purchase), "messages": how many messages to the shopper it takes (0 when none)}; [] when the request asks for no such step. Requirements on the items, the budget, stock, the limits on messages and steps, and how to end when nothing fits are not steps.
"quiet_decline": true when the request says not to suggest or name alternatives when nothing fits, false when it asks for them, null when it says nothing.
"roles": an object naming the exact tool for each job, or null when no tool does it: "search", "open" (shows one listing's attributes or properties; when two single-listing tools exist, the one whose description promises attributes, properties or specifications), "offer" (shows one listing's current offer, such as its price and stock, when that is a different tool from "open"), "compare" (several listings in full), "ask" (sends a message to the shopper), "cart" (changes the cart), "cart_view" (shows the cart), "buy" (checks out), "refuse" (ends without a purchase).
"buy_args": the fixed arguments that make the "buy" tool check out (for a tool with a mode argument, that mode), else {}.
"refuse_args": the fixed arguments that make the "refuse" tool end without a purchase, else {}.
"""


PLAN = """You choose the questions a shopping agent sends to a shopper. Shoppers hold a few firm preferences they did not write down, each about one property of one requested product. A question reveals a preference only when it names that exact property, and each answer reveals at most one preference, the first one the question names, so the most likely property goes first.

You get the request, the items with what the request already settles for each, and a numbered list of attribute keys seen on opened listings of each item's category, with how many of those listings carry the key and example values. An unwritten preference may be on a property that no listed key names.

Choose the properties in this order:
0. What the request itself asks you to check with the shopper (listed under ASK FIRST when there is any), each named by the property it is about.
1. Hinted properties: when a sentence of the request points at one of the listed keys without giving its value, that key, for the item the sentence is about.
2. Then the listed keys in the order given (most carried first); avoid keys seen on only one or two of the item's listings.
Choose a property the request already settles for that item after all the others (never leave it out). Skip a second name for a property already chosen, a key whose values list several purchase options at once, a key whose value differs on every listing (an identifier or code), and keys about selling terms rather than the product itself. With several items, take turns between the items.
"label" is the property's plain name: for a listed key, all of the key's words in plain spelling ("<word one> <word two>" for wordOneWordTwo or word_one_word_two; never shortened to one of its words); only the property's name, no example values, no brackets, no second property.
Choose COUNT, best first (fewer only when the lists hold fewer usable keys). Return ONE JSON object and nothing else: {"ask": [{"item": item number, "label": "...", "hint": "at most 8 of the story's words it comes from, or an empty string"}]}"""


HEAR = """A shopping agent asked a shopper some questions. Read each answer and return ONE JSON object and nothing else:
{"needs": [{"item": item number, "need": "property: constraint", "points": true when the value is to be read from another listing, else false}], "drop": [{"item": item number, "property": "a requirement the shopper now gives up"}], "free": [{"item": item number, "property": "..."}], "crowded": [numbers of the questions whose answer complains that the message asked about too many things or wants a single question or fewer questions per message; not answers that only say they did not follow it, an answer that only asks to have the question repeated or reworded (even more simply or more briefly), or one that only says the shopper does not mind], "lost": [numbers of the questions whose answer only says the shopper did not follow it or asks to have it again or in plainer words, without stating, refusing or waving off any preference], "budget": new total limit as a number or null}
- "needs": every firm preference an answer states, with the exact value or limit and its unit ("at least <number> <unit>", "<value>", "not <value>"). Name the need by the property the answer itself names, which may differ from the one asked about; several answers about one property make one need. Copy the value word for word as the answer says it, keeping its nouns and units ("at least <number> <noun>", not "at least <number>").
- An answer can give the value by pointing at another listing (a value like, unlike or near the one on some shop's product, that product picked out by a few of its properties): that is ONE need on the property the answer names, with "points": true. Its value is the answer's own words copied exactly, from the comparison words to the end of that description, including what it says about a listing that page may name; never shorten or reword it. The properties describing the other product are not needs of this item.
- "free": each property of a question that the shopper said they do not mind about.
- An answer that turns away every value except one asks for that one value: its need is "<property>: <that value>". Only an answer that turns away a value itself gives "not <value>".
- A remark about the shopper's own day, plans or whereabouts states nothing about the product: it adds no need.
- Folksy or roundabout wording that names a value still states that value."""


EXCLUDE = """A shopper answered a shopping agent's question, and a reader wrote one requirement from the answer as "<property>: not <value>", a value the shopper turns away. Such an answer means one of two things: the shopper turns away that value itself, and other values stay open; or the shopper turns away every value except that one, which is then the only value the shopper asks for. Decide from the answer's own words which of the two this answer means. Return ONE JSON object and nothing else: {"asks_for": true when the answer turns away every value except that one, false when it turns away that value itself}"""


NOTE = """While a shopping agent works, the store showed this note (it may come from the shopper or the store). Return ONE JSON object and nothing else:
{"kind": "change" when the note changes the task (budget, requirements, a price, availability, a refusal of the order), "error" when it only reports that a call was malformed or not accepted, "other" for anything else, "budget": the new total limit as a number in the request's unit, or null, "add": [{"item": item number, "need": "property: constraint"}], "drop": [{"item": item number, "need": "the requirement no longer wanted"}], "gone": [listing ids said to be unavailable], "sold_out": true when it says a chosen listing is gone, sold out or no longer available (even without naming it), "price_moved": true when it says a price changed, "summary": "one short sentence"}
A replaced requirement is a drop of the old one plus an add of the new one. Report every change in the note."""


CHECK = """You are the last check before a shopping agent checks out. Decide from the records shown whether this cart is exactly right.
Reading rules: a requirement is met when the listing's attribute for that property states the value (any key spelling, one element of a list is enough, case, singular/plural and spelling variants of the same word count, a value containing all of the required value's words counts, a plain synonym or everyday rewording that means the same thing counts, a value naming a different thing (a near kind, a neighbouring grade or size) does not, units converted, limits include the edge, a range must lie wholly inside); a requirement the request ties to a rule the store states is met by the value that rule gives; an attribute whose key names the same property as the requirement decides, and if it says something else the listing fails; a key for a different property, even a related one (a key about one part of the product says nothing about another part), decides nothing; when the listing has no attribute for that property, the wanted value may show under another of its attributes, in its title or in its description (never only among its purchase choices; a number of another kind or a bare figure that does not say what it measures does not count), except that when three or more listings of its category carry a key for that property and this one carries none, only its other attributes can show it, never its title or description (what the listing fits is read by the pair rule below), and when they lack it the listing does not meet that requirement (its [evidence] line marks it unproven); a requirement the category label itself states is met by that category; the category label must match the requested item as a whole; a rule of being unlike another listing's value counts as met unless the records show that same value. Each requested item must appear as a separate cart line at the requested quantity, nothing else may be in the cart, nothing shown out of stock, and the total of price times quantity over all lines must be within the budget. The newest record of a listing wins.
Category: when the CATEGORIES section shows a store category whose name carries the item's own words while a cart line's category does not, that line comes from a neighbouring category and is wrong. Pairs: when the request or a requirement says one item must work with, fit or suit another requested item (or a model the shopper names), the two listings' own words must connect them (a code or name that both records carry, or a statement on one record that covers the other); a pair whose records show no such link is wrong. A sentence on one record that names another part or model code describes that other part, never this listing: a listing fits only what its own list of what it fits names.
When the pick rule is cheapest, most or least, also look at the other listings shown: if one of them clearly fits every requirement and beats the cart's choice on that rule, the cart is wrong.
Return ONE JSON object and nothing else: {"ok": true or false, "problems": ["short, concrete problem", ...], "better": [listing ids that should replace a cart line, best first]}. Use ok=false only for a problem the records clearly show or a requirement a cart line's own record does not show by these rules. A line whose record is shown as missing is a gap in Kyo's notes: judge it only on what is shown, never fail it for the missing record."""


READING = """the listing's attribute for a property decides when its key names that same property (any key spelling, one element of a list is enough, case, plural, spelling variants and unit conversions count, a value containing all of the required value's words counts, a plain synonym or everyday rewording that means the same thing counts, a value naming a different thing (a near kind, a neighbouring grade or size) does not, limits include the edge); a requirement the request ties to a rule the store states is met by the value that rule gives; a key for a different property, even a related one (a key about one part of the product says nothing about another part), decides nothing; when no attribute names that property, the wanted value may show under another of its attributes, in its title or in its description (never only among purchase choices; a bare figure that does not say what it measures counts for nothing), except that when three or more listings of its category carry a key for that property and this one carries none, only its other attributes can show it, never its title or description (except for what the listing fits), and when they lack it the listing does not meet that requirement; of two listings that both meet a requirement, the one whose value is exactly the wanted value ranks ahead of the one that adds words to it (an order, never a reason to drop one); a sentence on one record naming another part or model code describes that other part, and a listing fits only what its own list of what it fits names; a requirement the category label itself states is met by that category; the category label must match the item; a rule of being unlike another listing's value counts as met unless the records show that same value; out-of-stock listings and listings that break the budget do not fit."""

AUDIT = """A shopping agent wants to end without buying because it thinks nothing fits. Check the listing records shown against the requirements, with these reading rules: """ + READING + """
Records marked ROW were seen only as a search result row: their page (the attributes) has not been read, so they can be neither passed nor failed on a property the row does not show.
Return ONE JSON object and nothing else: {"fits": {"item number": [listing ids that appear to meet every requirement, cheapest first, at most 4]}, "open": [ids of ROW listings of a requested kind, inside the budget, whose page could show that they fit, most promising first, at most 6], "reason": "one sentence"}"""

UNMIX = """A shopper's request was read into firm requirements on a product. Some requirements take their value from another listing the request points at, and the words that pick out that other listing (its kind, its shop, the properties named after it with "with ...") are sometimes read by mistake as requirements on the product being bought. Using the request's wording, say which of the OTHER REQUIREMENTS below only describe that other listing. Return ONE JSON object and nothing else: {"describe_other": [those requirements copied exactly; empty when every one is a real requirement on the product]}"""

TRAIL = """A shopping agent is about to check out. Some requirements take their value from another listing instead of stating it, and a listing's page can send the reader on to a further listing for that property. The order counts only when the agent has opened in full every listing along each such chain. Follow each requirement's chain from the requirement's own description, link by link, through the pages' own words, using the records below (OPENED: the agent has opened that page; ROW: seen only as a search result row, its page not read).
Return ONE JSON object and nothing else: {"chains": [{"need": the requirement copied exactly, "links": [{"id": the listing id when a record below is that link, else "", "about": how the requirement or the previous page describes that listing}], "done": true when the last link is an OPENED page that states the property's value itself, else false, "value": the property's own value as stated on that last page, or "", "rule_value": when that page states a rule that gives a value for this requirement, the value the rule gives, else ""}]}
List each chain's links in order and stop at the first link whose page is not OPENED or not found, since its words are still unknown."""

GATE = """A shopping agent is about to STAGE. The shopper's request asks for the steps below along the way. Decide from the numbered RECORD of what the agent has done whether each step due before this STAGE is done for the cart as it is now. A step counts as done only when a RECORD line shows it being done and, for a step about the cart or the order, that line comes after the cart last changed and its reply grants it; questions on a product's properties do not do a step.
When a step is still missing, give the one store call that does it next: a tool from TOOLS with its arguments. A message to the shopper goes through the messaging tool: one short sentence or question about that step alone, in plain words, naming no product properties or values the step does not need. Give no call when every due step is done or when the messages or steps left cannot do the missing one.
Return ONE JSON object and nothing else: {"steps": [{"step": the step, "done": true or false, "lines": [the RECORD line numbers that show it done]}], "call": {"tool": "...", "args": {...}} or null, "hold": what a reply asks to change before going on, or "", "needs": [{"item": item number, "need": "property: constraint"} for each firm requirement such a reply states]}"""

SWAP = """A listing in a shopping agent's cart has just sold out. Find replacements for that cart line among the other listing records shown, with these reading rules: """ + READING + """
Return ONE JSON object and nothing else: {"fits": [{"id": listing id, "price": its price as a number as the record shows it} for every listing that appears to meet every requirement of the sold-out line's item, at most 6], "reason": "one sentence"}"""

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


# a search argument that caps the price, one that names a category, and one that pages (matched by name, any store)
_PHRASING = {"include", "including", "contain", "containing", "with", "has", "have", "say", "says", "said", "having", "be", "is", "are", "must",
             "should", "only", "exactly", "just", "a", "an", "the", "of", "that", "which", "it", "listed", "labeled", "labelled", "marked"}
_CAP_WORDS = {"cap", "ceiling", "max", "maximum", "under", "upto"}
_CAP_PRICE = {"price", "spend", "budget", "cost"}
_CAP_NOT = {"row", "rows", "result", "results", "page", "pages", "count", "size", "num", "number", "item", "items", "step", "steps", "len", "length", "token", "char", "chars", "k"}
_SHELF_ARG = re.compile(r"section|department|categor|label|shelf|aisle|group|kind", re.I)
_PAGE_ARG = re.compile(r"page|offset|start|token|cursor|from|next|begin|after|skip", re.I)

# a store may write each record as one line of text instead of an object: "name: value | name: value | ..."
_NOTE_ARG = re.compile(r"note|reason|justif|message|comment|explan|summary|remark|why|text", re.I)
_ID_ARG = re.compile(r"(?:^|[_\W]|[a-z])(?:id|ids|sku|code|ref)$|listing|product|item|quantity|count|price|line", re.I)
_DEAD_KEY = re.compile(r"key limit|limit exceeded|exceeded (?:its|the|your) (?:key )?limit|out of credit|insufficient credit|quota", re.I)
_TEXT_KEY = re.compile(r"\s*([a-z_][A-Za-z0-9_]{0,39}):(?:\s|$)(.*)\Z", re.S)   # a field name starts in lower case
_PAIR_CUT = re.compile(r";\s*(?=[^;=|]{1,48}=)")


def _scalar(text: str) -> Any:
    """A field of a text record: the words true, false and none as values, a plain whole or decimal number as a
    number, anything else as the text itself."""
    word = text.strip()
    low = word.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null", ""):
        return None
    if re.fullmatch(r"-?\d{1,15}", word):
        return int(word)
    if re.fullmatch(r"-?\d{1,12}\.\d{1,6}", word):
        return float(word)
    return word


def _pairs(text: str) -> dict[str, Any] | None:
    """A field written as "name=value; name=value" read into a dict (an empty value stays an empty string); None
    for any other text."""
    if "=" not in text:
        return None
    out: dict[str, Any] = {}
    for part in _PAIR_CUT.split(text.strip()):
        if not part.strip():
            continue
        name, sep, value = part.partition("=")
        name = name.strip()
        if not sep or not name or len(name) > 48 or not re.match(r"[A-Za-z_]", name) or re.search(r"[.!?]\s", name):
            return None
        out[name] = value.strip()
    return out if out and (len(out) >= 2 or ";" not in text) else None


def _record_text(text: str) -> dict[str, Any] | None:
    """A record written as text: one line of "name: value" parts joined by " | ", or one "name: value" per line (a
    part without a name of its own goes on the field before it, as a title holding the separator does), read into a
    dict, a "name=value; ..." field into a nested dict. None for any other text: it needs three named fields and must
    start with one."""
    if not isinstance(text, str):
        return None
    if text.count(" | ") >= 2:
        cut = " | "
    elif text.count("\n") >= 2 and _TEXT_KEY.match(text.split("\n", 1)[0]):
        cut = "\n"
    else:
        return None
    out: dict[str, Any] = {}
    last = None
    for part in text.split(cut):
        if cut == "\n" and not part.strip():
            continue
        found = _TEXT_KEY.match(part)
        if found and found.group(1) not in out:
            last = found.group(1)
            out[last] = found.group(2)
        elif cut == "\n" and last is not None:
            # a line without a name of its own in a one-field-per-line record is free text, kept apart
            out["text"] = (str(out.get("text") or "") + " " + part.strip()).strip()
        elif last is not None:
            out[last] += cut + part
        else:
            return None
    if len(out) < 3:
        return None
    for key, value in list(out.items()):
        pairs = _pairs(value)
        out[key] = pairs if pairs is not None else _scalar(value)
    return out


def _safe(default: Any):
    """A helper Kyo added for new worlds never stops an episode: on an unexpected error it notes it in the trace and
    returns `default` (a fresh copy for a list, dict or set)."""
    def wrap(fn):
        def run(self, *args, **kwargs):
            try:
                return fn(self, *args, **kwargs)
            except Exception as exc:
                try:
                    self.trace.append({"kind": "helper_error", "helper": fn.__name__, "error": f"{type(exc).__name__}: {exc}"[:200]})
                except Exception:
                    pass
                return type(default)() if isinstance(default, (list, dict, set)) else default
        run.__name__ = fn.__name__
        run.__doc__ = fn.__doc__
        return run
    return wrap


def _text_records(node: Any, depth: int = 0) -> Any:
    """A tool result with every text record inside it (see _record_text) and every JSON object or list written as a
    string turned into data; the rest unchanged. The fields of a record that already is an object (an id and three or
    more plain values, no list) stay as they are: a description is not a record."""
    if depth > 6:
        return node
    if isinstance(node, str):
        text = node.strip()
        if len(text) > 1 and text[0] in "{[" and text[-1] in "}]":
            try:
                inner = json.loads(text)
            except ValueError:
                inner = None
            if isinstance(inner, (dict, list)):
                return _text_records(inner, depth + 1)
        return _record_text(node) or node
    if isinstance(node, list):
        return [_text_records(v, depth + 1) for v in node]
    if isinstance(node, dict):
        plain = sum(1 for v in node.values() if isinstance(v, (str, int, float)) and not isinstance(v, bool))
        if plain >= 3 and not any(isinstance(v, list) for v in node.values()) and _learn_id([node]):
            return node
        return {k: _text_records(v, depth + 1) for k, v in node.items()}
    return node


class Item:
    def __init__(self, name: str, shelf: str, qty: int, musts: list[str]):
        self.name = name
        self.shelf = shelf
        self.qty = qty
        self.musts = musts
        self.told: list[str] = []
        self.dropped: list[str] = []
        self.free: list[str] = []


class Kyo:
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
        self.t0 = time.monotonic()
        self.loop_cap = LOOP_CALLS
        self.call_cap = ALL_CALLS
        self.join_failed = False
        self.out_of_time = False
        self.cart_shows_lines = False
        self.count_key: str | None = None
        self.msg_refused = False
        self.topup: set[str] = set()
        self.ref_rules: dict[str, str] = {}
        self.asked_labels: list[tuple[int, str]] = []
        self.key_distinct: dict[tuple[str, str], int] = {}
        self.key_owner: dict[str, set[int]] = {}
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
        self.qa: list[tuple[str, str]] = []
        self.rows: dict[str, dict[str, Any]] = {}
        self.full: dict[str, dict[str, Any]] = {}
        self.raw_seen: list[str] = []
        self.page_texts: list[str] = []
        self.browse_forms: dict[str, tuple[str, dict[str, Any], str]] = {}
        self.capped_done: set[str] = set()
        self.seen_order: list[str] = []
        self.heard: set[str] = set()
        self.slowest = QUESTION_TIME
        self.ask_tries = 0
        self.ask_missed_at = -1
        self.rounds = 0
        self.dry_at = -1
        self.dry_own: list[int] = []
        self.last_finish = ""
        self.finish_refused = 0
        self.wanted: list[str] = []
        self.shrugs: dict[int, int] = {}  # answers in a row per item that did not mind (no news, not lost)  # listings the agent tried to open when too few steps were left
        self.stack_ok = True
        self.lost_switch = False
        self.listing = False
        self.questions = 0
        self.reasked = False
        self.notes: list[str] = []
        self.version = 0
        self.cart_touched = False
        self.cart_at = -1
        self.cart_seen_at = -1
        self.cart_view: Any = None
        self.cart_lines: dict[str, int] = {}
        self.checked: set[str] = set()
        self.need_held: set[str] = set()
        self.stem_memo: dict[str, frozenset[str]] = {}
        self.kept: dict[str, str] = {}
        self.kept_held: set[tuple[str, str]] = set()
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
        self.built: list[str] = []
        self.code_steps = 0
        self.end_steps = 0
        self.word_form: tuple[dict[str, Any], str] | None = None
        self.word_seen: list[tuple[frozenset[str], str]] = []
        self.word_misses = 0
        self.valued: set[tuple[int, str]] = set()
        self.decline_marks: set[int] = set()
        self.shelf_asked: set[str] = set()
        self.dropped: set[str] = set()
        self.late_held = False
        self.proof_holds: dict[str, int] = {}
        self.cart_holds: dict[str, int] = {}
        self.check_rejected: set[str] = set()
        self.kind_held: set[str] = set()
        self.check_better: list[str] = []
        self.stock_views: set[str] = set()
        self.flagless_views: set[str] = set()
        self.word_calls: list[tuple[str, str, str, tuple[str, ...]]] = []
        self.deaf_noted: set[tuple[str, str]] = set()
        self.declines_refused = 0
        self.rereviewed: set[str] = set()
        self.cart_noted: set[str] = set()
        self.fit_memo: tuple[int, str, dict[str, list[int]]] = (-1, "", {})
        self.review_short: set[str] = set()
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
        self.exclude_calls = 0
        self.exclude_memo: dict[str, bool | None] = {}

    # ---------- plumbing ----------
    def left(self) -> int:
        return self.max_steps - self.step

    def time_left(self) -> float:
        """Seconds of Kyo's own budget still left (the network stops a task at 300 s, finished or not)."""
        return DEADLINE - (time.monotonic() - self.t0)

    def checker(self) -> list[str]:
        """Models for a late check (cart check, decline review, sold-out swap, requirement unmixing): the heavy model
        first, the light one first once this task's spend passes SPEND_LIGHT (one run key pays for every task)."""
        return [HEAVY, LIGHT] if self.cost < SPEND_LIGHT else [LIGHT, HEAVY]

    def set_caps(self) -> None:
        """Model turns and calls grow with the task's step cap (long orders have more steps); the clock and the spend
        stop still bound them."""
        self.loop_cap = max(LOOP_CALLS, min(110, self.max_steps))
        self.call_cap = max(ALL_CALLS, self.loop_cap + 25)

    def rearm(self) -> None:
        """A change (new budget, requirement added or dropped, a refused finish) re-opens the one-shot checks: the
        decline review, the search-more push, the late hold and the cheapest widening run again on the new task."""
        self.audited = False
        self.pushed = False
        self.refuse_blocks = 0
        self.late_held = False
        self.wider_held = False

    def msgs_free(self) -> int:
        """Messages the interviewer may send: the ones a step the request asks for right before the order takes stay
        back until the process gate has seen that step done."""
        held = 0 if any(stage == "order" for stage, _ in self.gate_ok) else max(0, self.reserve - self.gate_msgs)
        return self.msg_limit - self.msg_used - held

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
        """Steps kept for the finish: the checkout, a second checkout after a refusal and one read between them, and
        one more while a step due at the order needs a message."""
        talk = any(p["when"] == "order" and p["messages"] > 0 for p in self.process)
        return 4 if talk and self.msg_used < self.msg_limit and self.gate_due("order") else 3

    def think(self, messages: list[dict[str, Any]], models: list[str], max_tokens: int, tools: list[dict[str, Any]] | None = None) -> dict[str, Any] | None:
        for model in models:
            if self.calls >= self.call_cap or self.cost >= SPEND_STOP or self.no_credit:
                return None
            if self.time_left() < LAST_CALL:
                self.out_of_time = True
                return None
            self.calls += 1
            tries = 0
            while True:
                message, status = self.call_model(model, messages, max_tokens, tools)
                if message is not None:
                    return message
                # no credit, rate limits, timeouts and server errors pass: wait and send the same call again
                passing = status is None or status in (402, 408, 425, 429) or (isinstance(status, int) and status >= 500)
                # a timeout already took the client's 50 s: it is sent once more at most
                if not passing or tries >= (1 if status in (None, 408) else len(WAITS)) or self.waited + WAITS[tries] > WAIT_TOTAL:
                    break
                if self.time_left() - WAITS[tries] < LAST_CALL:
                    self.out_of_time = True
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
        began = time.monotonic()
        result = _llm.post_verbose("/inference/chat/completions", body)
        spent = time.monotonic() - began
        data = result.data
        if isinstance(data, dict) and data.get("choices"):
            usage = data.get("usage") or {}
            if isinstance(usage.get("cost"), (int, float)):
                self.cost += float(usage["cost"])
            message = data["choices"][0].get("message") or {}
            if message.get("content") or message.get("tool_calls"):
                self.slowest = max(self.slowest, spent)
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
        if status == 403 and _DEAD_KEY.search(str(error.get("body") or "")):
            # the run's key reached its cap: no model call can run again, so the code finishes alone (and never messages
            # the shopper, whose reply would need the same key)
            self.no_credit = True
        return None, status

    def ask_json(self, prompt: str, models: list[str], max_tokens: int = 1800) -> dict[str, Any] | None:
        for model in models:
            message = self.think([{"role": "user", "content": prompt}], [model], max_tokens)
            got = _json_obj((message or {}).get("content"))
            if got is not None:
                return got
            if self.broke or self.no_credit or self.calls >= self.call_cap or self.out_of_time or self.cost >= SPEND_STOP:
                return None
        return None

    def act(self, name: str, args: dict[str, Any]) -> dict[str, Any] | None:
        payload = dict(self.binding)
        payload["step"] = self.step + 1
        payload["action"] = {"name": name, "args": args}
        for _ in range(4):
            result = _env.post_verbose("/environment/call", payload)
            if isinstance(result.data, dict):
                data = result.data
                if "observation" in data:
                    # records written as lines of text are read into objects, so the ledger sees every listing
                    raw = data.get("observation")
                    try:
                        data["observation"] = _text_records(raw)
                    except Exception:
                        data["observation"] = raw
                    if raw is not data["observation"] and _dump(raw) != _dump(data["observation"]):
                        self.raw_seen.append(_dump(raw)[:40000])
                        self.raw_seen = self.raw_seen[-60:]
                    if name in self.page_tools() and raw:
                        self.page_texts = (self.page_texts + [_dump(raw)[:2500]])[-6:]
                    self.note_call(name, args, data["observation"])
                    self.note_stock_view(name, data["observation"])
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
        # room for the light model's thinking: a reply cut at its limit would hand many briefs to the heavy one
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
            # price, budget, stock and quantity have their own fields: as a must they would be judged against attributes
            musts = [m for m in musts if not (set(_words(m.split(":", 1)[0]).split()) & {"stock", "price", "budget", "quantity", "availability", "available", "cost"})]
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
            # the messages a step right before the order takes; an ending without a purchase keeps none back
            talk = [p["messages"] for p in self.process if p["when"] == "order" and p["messages"]]
            self.reserve = min(max(talk), self.msg_limit) if talk else 0

    def unmix(self) -> None:
        """Musts that only describe the listing a lookup points at (read as musts of the product) move into that lookup."""
        for number, item in enumerate(self.items, 1):
            pointers = [need for n, need in self.refs if n == number and need in item.musts]
            rest = [m for m in item.musts if m not in pointers]
            if not pointers or not rest:
                continue
            prompt = (UNMIX + "\n\nREQUEST:\n" + self.request + f"\n\nPRODUCT: {item.name}\nREQUIREMENTS POINTING AT ANOTHER LISTING:\n"
                      + "\n".join(pointers) + "\nOTHER REQUIREMENTS:\n" + "\n".join(rest))
            got = self.ask_json(prompt, self.checker(), 500) or {}
            moved = [m for m in rest if m in [str(x).strip() for x in _items(got.get("describe_other"))]]
            if not moved:
                continue
            item.musts = [m for m in item.musts if m not in moved]
            joined = pointers[0] + " (that listing: " + "; ".join(moved) + ")"
            item.musts = [joined if m == pointers[0] else m for m in item.musts]
            self.refs = [(n, joined if (n, need) == (number, pointers[0]) else need) for n, need in self.refs]
            self.trace.append({"kind": "unmix", "item": number, "moved": moved})

    @staticmethod
    def free_now(item: Item) -> list[str]:
        """The properties the shopper does not mind, less any a requirement in force names: a non-answer to a list a
        stated property led never waives that property."""
        held = [_stem(m.split(":")[0]) for m in item.musts + item.told if m not in item.dropped]
        out = []
        for name in item.free:
            mine = _stem(name)
            if mine and not any(k and (k <= mine or mine <= k) for k in held) and name not in out:
                out.append(name)
        return out

    def ledger(self) -> str:
        parts = []
        many = len(self.items) > 1
        for number, item in enumerate(self.items, 1):
            needs = [m + (f" (value on the page it points at: {self.ref_values[m]}" + (f"; a rule stated on that page gives: {self.ref_rules[m]}" if m in self.ref_rules else "") + ")" if m in self.ref_values else "") for m in item.musts + item.told if m not in item.dropped]
            head = f"item {number}: {item.name}" if many else f"item: {item.name}"
            if item.shelf:
                head += f" (store label: {item.shelf})"
            head += f", quantity {item.qty}"
            text = head + "; requirements: " + ("; ".join(needs) if needs else "none beyond the kind of product")
            free = self.free_now(item)
            if free:
                text += "; shopper does not mind: " + ", ".join(free)
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
        elif left <= 6 or (left <= len(self.items) + 5 and self.buys == 0):
            text += " Few steps left: set the cart to the chosen lines and check out now."
        elif self.buys == 0 and left <= 12:
            text += " Steps are running low: settle on your picks, fill the cart and check out within the next few steps."
        elif self.buys == 0 and not self.cart_touched and self.step >= self.max_steps // 2:
            text += " Half the steps are used and the cart is empty: cart each item's fitting listing now (the store may still sell one out)."
        secs = self.time_left()
        if secs < CLOSING + 50:
            text += f" Time is short: about {max(0, int(secs))} seconds are left for this task. Cart the fitting listings you have and check out now."
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
        """A listing's id, read with the id key learned from result rows; None for anything else."""
        if not isinstance(node, dict) or len(node) < 3:
            return None
        if self.id_key and _id_like(node.get(self.id_key), True):
            return node[self.id_key].strip()
        nested = any(isinstance(v, dict) for v in node.values())
        if not (in_list or depth == 0 or nested):
            return None
        found = _learn_id([node])
        if found and ("id" in str(found).lower() or node[found].strip() in self.rows or (nested and _fact_block(node))):
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

    def label_fit(self, label: str, item: Item) -> int:
        """How well a store category label fits a requested item: 3 the label is the item's shelf or name, 2 every
        word of the label is among the item's words (a short label such as an abbreviation), 1 they share a word of
        four letters or more, 0 not at all."""
        mine = _stem(item.shelf or "") | _stem(item.name)
        theirs = _stem(label)
        if not theirs:
            return 0
        if _words(label) in (_words(item.shelf or ""), _words(item.name)):
            return 3
        if theirs <= mine:
            return 2
        if {w for w in theirs if len(w) >= 4} & {w for w in mine if len(w) >= 4}:
            return 1
        return 0

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
        # only categories that belong to a requested item: another product's listings (one a requirement points at,
        # read first) would give the item questions about that product's properties
        owners = self.category_owners(list(groups), cat_key)
        self.key_owner = owners or {}
        if owners is not None:
            groups = {label: group for label, group in groups.items() if label in owners}
        table = []
        for label, group in sorted(groups.items(), key=lambda kv: -kv[1]["n"]):
            group["keys"] = self.merge_keys(group["keys"])
            kept: list[tuple[str, dict[str, Any]]] = []
            for name, slot in sorted(group["keys"].items(), key=lambda kv: -kv[1]["n"]):
                values = list(slot["per"].values())
                if not values:
                    continue
                mean = sum(len(v) for v in values) / len(values)
                if mean > 40 or sum(1 for v in values if " / " in v or not v) * 3 >= len(values):
                    continue
                if set(name.split()) & {"option", "options", "variant", "variants", "sku", "model", "id"}:
                    # purchase variants or identifiers: never a preference a shopper holds
                    continue
                if len(values) >= 4 and len(set(values)) == len(values) and mean > 12 and not self.shared_word(values):
                    # a long value different on every listing is an identifier, unless one word recurs in half the
                    # values: a property's values share words ("intel core i5-6500", "intel n100")
                    continue
                twin = any(name.startswith(other + " ") and slot["per"] == spec["per"] for other, spec in kept)
                if twin:
                    continue
                kept.append((name, slot))
            for name, slot in kept[:14]:
                table.append((label, name, slot["n"], [v[:40] for v in slot["values"][:4]]))
                self.key_distinct[(label, name)] = len(set(slot["per"].values()))
        return table

    @staticmethod
    def shared_word(values: list[str]) -> bool:
        """One word of three letters or more is in at least half of the values (each value counted once)."""
        count: dict[str, int] = {}
        for value in values:
            for word in set(re.findall(r"[a-z]{3,}", value)):
                count[word] = count.get(word, 0) + 1
        return any(n * 2 >= len(values) for n in count.values())

    def kin_fit(self, label: str, item: Item) -> int:
        """label_fit, else 1 when the label and the item share a word of four letters or more once a word ending in "ve"
        (a plural in "-ves" less its "s") is also read with "f" for that ending."""
        fit = self.label_fit(label, item)
        if fit:
            return fit

        def forms(text: str) -> set[str]:
            words = {w for w in _stem(text) if len(w) >= 4}
            return words | {w[:-2] + "f" for w in words if w.endswith("ve")}

        return 1 if forms(label) & (forms(item.shelf or "") | forms(item.name)) else 0

    def category_owners(self, labels: list[str], cat_key: str | None) -> dict[str, set[int]] | None:
        """The requested items (numbers) each category label belongs to: the items it fits best (kin_fit); a label
        missing from the result belongs to no item. None while no category label seen in the store (on rows, or among
        the tools' allowed values) fits any requested item, or when the store shows no category key: the labels cannot
        be matched to the request, so all stay."""
        if not cat_key:
            return None
        seen = set(labels) | set(self.seen_groups(cat_key)) | self.labels()
        if not any(self.kin_fit(label, item) for label in seen for item in self.items):
            return None
        owners: dict[str, set[int]] = {}
        for label in labels:
            scores = [self.kin_fit(label, item) for item in self.items]
            top = max(scores, default=0)
            if top > 0:
                owners[label] = {n for n, s in enumerate(scores, 1) if s == top}
        return owners

    def owns(self, number: int, label: str) -> bool:
        """Whether a category label belongs to requested item `number` (as the last key_table sorted it; a label it did
        not sort goes to the items it fits best)."""
        got = getattr(self, "key_owner", {}).get(label)
        if got is not None:
            return number in got
        scores = [self.kin_fit(label, item) for item in self.items]
        return 0 < scores[number - 1] == max(scores)

    def seen_groups(self, cat_key: str) -> dict[str, list[str]]:
        """Ids of the rows seen so far, by category label."""
        groups: dict[str, list[str]] = {}
        for ident in self.seen_order:
            label = str((self.rows.get(ident) or {}).get(cat_key) or "").strip()
            if label:
                groups.setdefault(label, []).append(ident)
        return groups

    def seek_own_rows(self) -> None:
        """Before the first question, when no row seen so far is of a category that fits a requested item (the listings
        read are other products', such as one a requirement points at): one word search for the first item's product
        noun, inside the code step budget and only while the reads after it still fit, so widen() reads rows of its
        own category and the other products' keys leave the table."""
        cat_key = self.category_key()
        if self.asked or not self.items or not cat_key or self.left() <= 26:
            return
        if any(self.kin_fit(label, item) for label in self.seen_groups(cat_key) for item in self.items):
            return
        noun = self.search_noun(self.items[0])
        if noun:
            ids = self.code_search(noun)
            self.trace.append({"kind": "own_rows", "words": noun, "rows": len(ids)})

    def merge_keys(self, keys: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        """Stores spell one property several ways across listings (a shorter or longer key, a run-together word).
        Two keys are one property when one key's words are all in the other's and their values agree on a listing
        both carry or share a value, or when they share a word and agree on every one of at least three listings
        both carry. The merged key counts every listing carrying either and is named by its longest spelling (the
        one a question can name without also naming a neighbour)."""
        def flat(text: str) -> str:
            return re.sub(r"[^a-z0-9]+", "", text)

        names = sorted(keys, key=lambda n: (-len(n.split()), -keys[n]["n"], n))
        merged: dict[str, dict[str, Any]] = {}
        for name in names:
            slot = keys[name]
            mine = _stem(name)
            target = None
            for big, spec in merged.items():
                theirs = _stem(big)
                both = [i for i in slot["per"] if i in spec["per"]]
                agree = sum(1 for i in both if flat(slot["per"][i]) == flat(spec["per"][i]))
                share = bool({flat(v) for v in slot["values"]} & {flat(v) for v in spec["values"]})
                if (mine <= theirs or theirs <= mine) and (agree or share) and agree == len(both):
                    target = big
                    break
                if mine & theirs and len(both) >= 3 and agree == len(both):
                    target = big
                    break
            if target is None:
                merged[name] = {"n": slot["n"], "values": list(slot["values"]), "per": dict(slot["per"])}
                continue
            spec = merged[target]
            for ident, text in slot["per"].items():
                spec["per"].setdefault(ident, text)
            for value in slot["values"]:
                if value not in spec["values"]:
                    spec["values"].append(value)
            spec["n"] = len(spec["per"])
        return merged

    def item_groups(self) -> list[list[str]]:
        """Listing ids per category, keeping the categories that match a requested item (largest first)."""
        cat_key = self.category_key()
        groups: dict[str, list[str]] = {}
        for ident in self.seen_order:
            label = str(self.rows[ident].get(cat_key) or "") if cat_key else ""
            groups.setdefault(label, []).append(ident)
        matched = [ids for label, ids in groups.items() if any(self.label_fit(label, item) for item in self.items)]
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

    @_safe("")
    def deaf_words(self, name: str, args: Any, observation: Any) -> str:
        """A search that showed exactly the rows of an earlier call to the same tool whose other arguments (page
        included) were the same and only the free words differed: in that mode the store ignores the words (noted once
        per argument shape, no step)."""
        if not isinstance(args, dict):
            return ""
        props = (self.schemas.get(name) or {}).get("properties") or {}
        texts = sorted(k for k, v in args.items() if isinstance(v, str) and re.search(r"[^\W\d_]", v) and isinstance(props.get(k), dict)
                       and "enum" not in props[k] and not _SHELF_ARG.search(k))
        ids = tuple(i for i, _ in self.records_in(observation))
        if not texts or len(ids) < 3:
            return ""
        rest = _dump({k: v for k, v in args.items() if k not in texts})
        words = _dump({k: args[k] for k in texts})
        hit = any(n == name and r == rest and w != words and prev == ids for n, r, w, prev in self.word_calls)
        self.word_calls = (self.word_calls + [(name, rest, words, ids)])[-30:]
        if not hit or (name, rest) in self.deaf_noted:
            return ""
        self.deaf_noted.add((name, rest))
        self.trace.append({"kind": "deaf_words", "tool": name, "args": rest[:200]})
        return ("\n[Kyo] This search showed exactly the rows an earlier call showed with other words and the same other arguments:"
                " in this mode the store ignores the words. Page on with its page argument, or use the store's word search to reach other listings.")

    def more_pages(self, name: str, args: dict[str, Any], observation: Any) -> str:
        """A search that matched more listings than it showed, under a cheapest, most or least pick: Kyo reads a short
        rest itself (a step a page, no model turn); a long rest is named so the agent can narrow the search."""
        self.paged_obs = []
        fresh = self.word_form is None
        self.note_words(name, args, observation)
        text = self.rest_pages(name, args, observation)
        if fresh and self.word_form is not None:
            # the first ranked word search seen: requirements heard before it get their value search now
            text += self.value_search()
        return text

    def rest_pages(self, name: str, args: dict[str, Any], observation: Any) -> str:
        """The pages of a search after its first that Kyo reads itself (a step a page), or a note that names them."""
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
        if self.id_ordered(name, args, observation, param):
            # a shelf listed in id order: the next pages bring more of the same order, so the words narrow it instead
            self.trace.append({"kind": "pages_skipped", "tool": name})
            return (f"\n[Kyo] This list matched {count} listings and showed {shown}, in id order (not price order). The pick counts every fitting listing: "
                    f"search with narrower words (the requirement's value plus the product noun) or page on with {param}={_dump(value)}.")
        pages = -(-(count - shown) // shown)
        if pages > 2 or self.pages_read + pages > AUTO_PAGES or self.left() - self.tail() - pages < 12:
            return (f"\n[Kyo] This search matched {count} listings and showed {shown}, in no price order. The pick counts every fitting listing: "
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
            self.note_words(name, call, seen)
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

    def code_room(self, steps: int, ending: bool = False) -> int:
        """How many of `steps` store calls Kyo may start on its own (searches, opens): the finish steps, two for the agent
        and, except for the opens before an ending without a purchase, the steps kept for those opens stay back; and all
        such calls stay within a quarter of the episode's steps, the ending's opens kept inside that from the start."""
        if self.finished or self.time_left() <= CLOSING + (0 if ending else 20):
            return 0
        keep = self.tail() + 2 + (0 if ending else DECLINE_OPENS)
        share = self.max_steps // 4 - self.code_steps - (self.end_steps if ending else DECLINE_OPENS)
        return max(0, min(steps, self.left() - keep, share))

    def singular(self, word: str) -> str:
        """A plural word in the singular (a store's search may not match a plural): of the forms its ending allows, the
        one the listings seen use most in their own words; without such a use, the word less its last letter."""
        low = word.lower()
        if len(low) <= 3 or not low.endswith("s") or low.endswith("ss"):
            return word
        forms = [low[:-1], low[:-2], low[:-3] + "y", low[:-3] + "f"]
        used: dict[str, int] = {}
        for row in self.rows.values():
            for value in row.values():
                if isinstance(value, str):
                    for token in re.findall(r"[^\W_]+", value.lower()):
                        if token in forms:
                            used[token] = used.get(token, 0) + 1
        return max(forms, key=lambda f: used.get(f, 0)) if used else low[:-1]

    def label_parts(self, label: str) -> list[str]:
        """The parts of a name or category label that joins several (split as the planner's labels are, and at "&"),
        each with its last word in the singular."""
        out = []
        for part in re.split(r"\s+(?:and|or)\s+|[/,;(&]", label):
            words = part.split()
            if words and re.search(r"[^\W\d_]", part):
                out.append(" ".join(words[:-1] + [self.singular(words[-1])]))
        return out

    def search_noun(self, item: Item) -> str:
        """The product noun for a search: the first part of the item's name as the request words it (its shelf label when
        the name gives none), at most its last two words."""
        for text in (item.name, item.shelf):
            part = re.split(r"\s+(?:and|or)\s+|[/,;(&]", text)[0].split()
            if part and re.search(r"[^\W\d_]", " ".join(part)):
                return " ".join(part[-2:])
        return ""

    def value_terms(self, need: str) -> str:
        """Search words for a requirement's value: a figure with the word written after it in the requirement's own text
        (the edge of a limit too), else the first value it accepts; "" for an exclusion (written "not <value>" by the
        answer reader), an empty value or a text too long to be one."""
        text = (need.split(":", 1)[1] if ":" in need else need).strip().strip(".")
        # a property's value is a few words: a long text describes something else (another listing, a story)
        if not text or text.split()[0].lower() == "not" or len(text.split()) > 6:
            return ""
        figures = re.findall(r"(?<![\w.])\d+(?:\.\d+)?(?:\s*-?\s*[^\W\d_]+)?", text)
        if figures:
            return figures[-1].strip()
        first = re.split(r"\s+(?:and|or)\s+|[/,;(]", text)[0].strip()
        return " ".join(first.split()[:3])

    def word_set(self, text: str) -> frozenset[str]:
        return _stem(re.sub(r"[\W_]+", " ", str(text)))

    def shows_value(self, ident: str, words: str) -> bool:
        """Whether a listing's record (its row, or its page once read) carries every word of a value."""
        want = self.word_set(words)
        return bool(want) and want <= self.word_set(_dump(self.full.get(ident) or self.rows.get(ident) or {}))

    def own_label(self, item: Item) -> str:
        """The category label among the rows seen that fits the item best (more rows break a tie); "" when none fits."""
        cat_key = self.category_key()
        if not cat_key:
            return ""
        counts: dict[str, int] = {}
        for ident in self.seen_order:
            label = str(self.rows[ident].get(cat_key) or "").strip()
            if label:
                counts[label] = counts.get(label, 0) + 1
        fit, _, label = max(((self.label_fit(l, item), n, l) for l, n in counts.items()), default=(0, 0, ""))
        return label if fit else ""

    def of_label(self, ident: str, label: str) -> bool:
        cat_key = self.category_key()
        return bool(label and cat_key) and str((self.rows.get(ident) or {}).get(cat_key) or "").strip() == label

    def row_price(self, ident: str) -> float | None:
        """The price under the named price key; before one is named, a row's only number."""
        price = self.price_of(ident)
        if price is not None or self.price_key:
            return price
        numbers = [v for v in (self.rows.get(ident) or {}).values() if isinstance(v, (int, float)) and not isinstance(v, bool)]
        return float(numbers[0]) if len(numbers) == 1 else None

    def in_budget(self, ident: str, item: Item) -> bool:
        """The row's price for the item's count stays inside the budget (a budget no row seen fits is read as one in
        another unit and decides nothing)."""
        price = self.row_price(ident)
        if price is None or isinstance(self.budget, bool) or not isinstance(self.budget, (int, float)):
            return True
        if price * (1 if self.brief.get("budget_scope") == "each" else item.qty) <= self.budget:
            return True
        return not any((self.row_price(i) if self.row_price(i) is not None else float("inf")) <= self.budget for i in self.seen_order)

    def unread_row(self, ident: str) -> bool:
        """A row whose page was never read and that can be bought (before a stock key is named, a row's only yes/no
        field must not say no)."""
        row = self.rows.get(ident)
        if row is None or ident in self.opened or ident in self.full or not self.buyable(ident):
            return False
        flags = [v for v in row.values() if isinstance(v, bool)]
        return bool(self.stock_key) or len(flags) != 1 or flags[0]

    def note_words(self, name: str, args: dict[str, Any], observation: Any) -> None:
        """A search with words: keep its rows with its words (first seen first); and from one whose rows come in no id
        order (a ranked search, not a shelf list), learn which arguments set that mode and which takes the words (the
        one text argument, or the one that is no category label), for Kyo's own searches."""
        if name != self.roles.get("search") or not isinstance(args, dict):
            return
        props = (self.schemas.get(name) or {}).get("properties") or {}
        texts = [k for k, v in args.items() if isinstance(v, str) and re.search(r"[^\W\d_]", v) and isinstance(props.get(k), dict) and "enum" not in props[k]]
        if not texts:
            return
        ids = [i for i, _ in self.records_in(observation)]
        words = frozenset().union(*[self.word_set(args[k]) for k in texts])
        self.word_seen += [(words, i) for i in ids]
        if self.word_form is not None or len(ids) < 3 or ids == sorted(ids):
            return
        cat_key = self.category_key()
        labels = {_words(str(r.get(cat_key) or "")) for r in self.rows.values()} if cat_key else set()
        labels |= {_words(item.shelf) for item in self.items}
        free = [k for k in texts if _words(args[k]) not in labels]
        key = texts[0] if len(texts) == 1 else (free[0] if len(free) == 1 else None)
        if key:
            self.word_form = ({k: v for k, v in args.items() if isinstance(props.get(k), dict) and "enum" in props[k]}, key)

    def code_search(self, words: str, ending: bool = False) -> list[str]:
        """One word search Kyo runs itself (one step), shaped like the agent's own word searches; the ids of the rows
        it showed, in the store's order."""
        name = self.roles.get("search")
        if not name or not self.word_form or self.word_misses >= 2 or not words.strip() or not self.code_room(1, ending):
            return []
        fixed, key = self.word_form
        faults: list[str] = []
        args = _shape({**fixed, key: words.strip()}, self.schemas.get(name), "args", faults)
        mark = name + _dump(args)
        cached = self.read_cache.get(mark)
        if faults or (cached and cached[0] == self.version):
            return []
        before = self.step
        reply = self.act(name, args)
        if ending:
            self.end_steps += max(1, self.step - before)
        else:
            self.code_steps += max(1, self.step - before)
        if reply is None or self.finished:
            return []
        seen = reply.get("observation")
        self.observe(name, seen)
        self.note_words(name, args, seen)
        self.read_cache[mark] = (self.version, self.step, self.show(seen))
        ids = [i for i, _ in self.records_in(seen)]
        # a search that shows no rows twice means the words do not reach this store's search: no more of them
        self.word_misses = 0 if ids else self.word_misses + 1
        self.trace.append({"kind": "code_search", "words": words, "rows": len(ids), "ending": ending})
        return ids

    def code_open(self, ids: list[str], ending: bool = False) -> list[str]:
        """Read listings' attributes on Kyo's own initiative (one step each, within the step budget); what each showed,
        with its [evidence] line."""
        tool = self.facts_tool()
        target = self.single_target(tool)
        texts: list[str] = []
        for ident in ids:
            if not tool or not target or self.finished or self.facts_from.get(tool) is False or not self.code_room(1, ending):
                break
            faults: list[str] = []
            args = _shape({target: ident}, self.schemas.get(tool), "args", faults)
            if faults:
                break
            before = self.step
            reply = self.act(tool, args)
            if ending:
                self.end_steps += max(1, self.step - before)
            else:
                self.code_steps += max(1, self.step - before)
            if reply is None:
                break
            seen = reply.get("observation")
            self.observe(tool, seen)
            self.note_view(tool, seen)
            if isinstance(seen, dict) and self.ident(seen):
                self.facts_from[tool] = self.facts_from.get(tool, False) or bool(_fact_block(seen))
            proof = self.evidence(seen)
            texts.append(f"{ident}: " + self.show(seen)[:1500] + ("\n" + proof if proof else ""))
        if texts:
            self.trace.append({"kind": "code_open", "n": len(texts), "ending": ending})
        return texts

    def carried(self, item: Item, need: str) -> bool:
        """Whether a requirement names a property the listings carry: some fully read listing of the item's own
        category has an attribute whose key names it. A wish no listing has an attribute for is no search word."""
        prop = _stem(need.split(":", 1)[0]) if ":" in need else frozenset()
        if not prop:
            return False
        label = self.own_label(item)
        for ident, record in self.full.items():
            if label and not self.of_label(ident, label):
                continue
            if any(_names(prop, _stem(key)) for key in (_fact_block(record) or {})):
                return True
        return False

    def value_queries(self, item: Item, value: str, pointed: set[str]) -> list[str]:
        """The words of a value search: the heard value, the request's own values for the item (not its limits: a search
        that must match every word then narrows to rows carrying them all), then the product noun; and the value with
        the noun alone, for when that finds no row stating the value."""
        noun = self.search_noun(item)
        if not value or not noun:
            return []
        stated: list[str] = []
        for must in item.musts:
            word = self.value_terms(must) if must not in pointed and must not in item.dropped and self.carried(item, must) else ""
            if word and not re.search(r"\d", word) and word not in stated and not self.word_set(word) <= self.word_set(f"{value} {noun}"):
                stated.append(word)
        return list(dict.fromkeys([" ".join([value] + stated + [noun]), f"{value} {noun}"]))

    def value_search(self) -> str:
        """After the interviewer's answers (and, for answers heard before any ranked word search of the agent's, at the
        first such search): for each new requirement with a value or a limit, one word search (see value_queries; the
        short one too when page 1 shows no row stating the value), and Kyo opens the unread rows of the item's category
        they showed (in stock, inside the budget, rows that state the value first). Returns a short note of what was
        opened."""
        if self.finished or not self.word_form:
            # Kyo's searches copy a ranked word search of the agent's: until one comes, requirements wait (none marked done)
            return ""
        notes: list[str] = []
        pointed = {need for _, need in self.refs}
        start = self.step
        for number, item in enumerate(self.items, 1):
            for need in item.told:
                if (number, need) in self.valued or need in item.dropped:
                    continue
                self.valued.add((number, need))
                value = self.value_terms(need) if need not in pointed and self.carried(item, need) else ""
                queries = self.value_queries(item, value, pointed)
                if not queries or self.code_room(2) < 2:
                    continue
                query = queries[0]
                ids = self.code_search(query)
                if len(queries) > 1 and not any(self.shows_value(i, value) for i in ids) and self.code_room(2) >= 2:
                    # no row on page 1 states the value (or none came back): the value and the noun alone
                    query = queries[1]
                    ids += self.code_search(query)
                label = self.own_label(item)
                rows = [i for i in dict.fromkeys(ids) if self.unread_row(i) and self.of_label(i, label) and self.in_budget(i, item)]
                cheap = self.brief.get("pick") == "cheapest"
                rows.sort(key=lambda i: (not self.shows_value(i, value), (self.row_price(i) if self.row_price(i) is not None else float("inf")) if cheap else 0))
                read = self.code_open(rows[: self.code_room(VALUE_OPENS)])
                if read:
                    tag = f"item {number} " if len(self.items) > 1 else ""
                    notes.append(f"{tag}'{query}' -> opened:\n" + "\n".join(read))
        if self.step > start:
            self.trace.append({"kind": "value_search", "steps": self.step - start})
        if not notes:
            return ""
        return ("\n[Kyo searched the newly heard values itself and opened unread rows of the item's category (one step each); judge them like any listing]\n"
                + "\n".join(notes))

    def decline_rows(self) -> str:
        """Before an ending without a purchase (once per change of the requirements): Kyo opens the unread rows of each
        item's category that word searches holding a requirement's value returned (in stock, inside the budget, first
        seen first, whatever their titles say, a second row of a title already picked after the new titles), and the
        chain check runs once; returns a note for the agent, or ""."""
        if self.version in self.decline_marks or self.time_left() <= CLOSING + 15:
            return ""
        self.decline_marks.add(self.version)
        picks: list[str] = []
        pointed = {need for _, need in self.refs}
        for number, item in enumerate(self.items, 1):
            label = self.own_label(item)
            values: set[str] = set()
            for n, need in self.needs_now():
                # a requirement whose value is on another listing: its words describe that other listing
                if n == number and need not in pointed:
                    values |= {w for w in self.word_set(need.split(":", 1)[-1]) if re.search(r"[^\W\d_]", w)}
            values -= self.word_set(item.name) | self.word_set(item.shelf) | self.word_set(label)
            if not label or not values:
                continue
            for words, ident in self.word_seen:
                if words & values and ident not in picks and self.unread_row(ident) and self.of_label(ident, label) and self.in_budget(ident, item):
                    picks.append(ident)
        picks = self.new_titles_first(picks)
        # listings the agent itself tried to open when too few steps were left come first (still unread, in budget)
        wanted = [i for i in self.wanted if self.unread_row(i) and any(self.in_budget(i, it) for it in self.items)]
        picks = wanted + [i for i in picks if i not in wanted]
        start = self.step
        read = self.code_open(picks[: self.code_room(DECLINE_OPENS, True)], True)
        before = self.step
        chain, hold = self.trail(True)
        self.end_steps += self.step - before
        if self.step == before and not hold:
            # nothing new on the chains: values read earlier are in the ledger already
            chain = ""
        self.trace.append({"kind": "decline_rows", "picks": picks[:8], "read": len(read), "chain": bool(chain), "steps": self.step - start})
        if not read and not chain:
            return ""
        text = ""
        if read:
            text += ("Before ending without a purchase, Kyo opened unread rows that searches for a requirement's value returned (one step each):\n"
                     + "\n".join(read) + "\n")
        return text + chain + ("Check these against every requirement in force: when one meets them all, add it to the cart at the requested quantity"
                               " and check out; end without a purchase only when each clearly fails, then send the ending call again.")

    def new_titles_first(self, ids: list[str]) -> list[str]:
        """The rows in their order, except that a row titled like an earlier row of the list goes after the rows with
        new titles. Rows sharing a title are often variants whose facts differ, so none is assumed wrong: the first row
        of each title keeps its place and the later ones still follow, but a few opens reach more different products
        first. Listings already read do not move any row."""
        key = self.name_key(self.category_key())
        if not key or len(ids) < 2:
            return ids

        def title(ident: str) -> frozenset[str]:
            text = (self.full.get(ident) or self.rows.get(ident) or {}).get(key)
            return self.word_set(text) if isinstance(text, str) else frozenset()

        seen: set[frozenset[str]] = set()
        fresh: list[str] = []
        again: list[str] = []
        for ident in ids:
            words = title(ident)
            (again if words and words in seen else fresh).append(ident)
            if words:
                seen.add(words)
        return fresh + again

    def cap_param(self, name: str) -> str | None:
        """The argument of a search tool that caps the price (a number argument named like a ceiling), or None."""
        props = (self.schemas.get(name) or {}).get("properties") or {}
        for key, spec in props.items():
            kinds = _kinds(spec) if isinstance(spec, dict) else []
            parts = {w.lower() for w in re.split(r"[_\W]+|(?<=[a-z])(?=[A-Z])", key) if w}
            capped = bool(parts & _CAP_WORDS) or ("limit" in parts and bool(parts & _CAP_PRICE))
            if capped and not parts & _CAP_NOT and ("number" in kinds or "integer" in kinds):
                return key
        return None

    @_safe(None)
    def note_call(self, name: str, args: Any, observation: Any) -> None:
        """A store call that named a category and showed rows, on a tool that also takes a price cap: remembered per
        category label its rows carry, as the shape of a capped browse Kyo can run itself later."""
        if not isinstance(args, dict) or not self.cap_param(name):
            return
        shelf = next((k for k, v in args.items() if isinstance(v, str) and v.strip() and _SHELF_ARG.search(k)), None)
        if not shelf:
            return
        self.absorb(observation)
        cat_key = self.category_key()
        if not cat_key:
            return
        for ident, record in self.records_in(observation):
            label = str(record.get(cat_key) or (self.rows.get(ident) or {}).get(cat_key) or "").strip().lower()
            if label:
                self.browse_forms[label] = (name, dict(args), shelf)

    @_safe([])
    def capped_browse(self, line: str) -> list[str]:
        """Under a cheapest pick: the agent's own browse of a cart line's category run again by Kyo (one step) with the
        store's price cap just below that line's price and from the first page; the ids of the rows it showed."""
        cat_key = self.category_key()
        label = str((self.rows.get(line) or {}).get(cat_key) or "").strip().lower() if cat_key else ""
        form = self.browse_forms.get(label)
        price = self.price_of(line)
        if not form or price is None or price <= 0 or label in self.capped_done or not self.code_room(1):
            return []
        name, used, shelf = form
        cap = self.cap_param(name)
        if not cap:
            return []
        self.capped_done.add(label)
        spec = ((self.schemas.get(name) or {}).get("properties") or {}).get(cap) or {}
        below = (int(price) - 1) if "integer" in _kinds(spec) or float(price).is_integer() else round(price * 0.999, 2)
        if below <= 0:
            return []
        new = {k: v for k, v in used.items() if k == shelf or not _PAGE_ARG.search(k)}
        new[cap] = below
        faults: list[str] = []
        args = _shape(new, self.schemas.get(name), "args", faults)
        if faults:
            return []
        before = self.step
        reply = self.act(name, args)
        self.code_steps += max(1, self.step - before)
        if reply is None or self.finished:
            return []
        seen = reply.get("observation")
        self.observe(name, seen)
        ids = [i for i, _ in self.records_in(seen)]
        self.trace.append({"kind": "capped_browse", "label": label, "cap": below, "rows": len(ids)})
        return ids

    def shelf_search(self) -> list[str]:
        """Under a cheapest pick, before the checkout: one word search per part of each cart line's category label (a
        search for the product's own name finds the listings whose titles carry it, which a shelf list in id order or a
        search for a value can miss); the ids of the rows they showed."""
        cat_key = self.category_key()
        found: list[str] = []
        start = self.step
        for line in self.cart_ids():
            found += [i for i in self.capped_browse(line) if i not in found]
        for line in self.cart_ids():
            label = str((self.rows.get(line) or {}).get(cat_key) or "").strip() if cat_key else ""
            for part in self.label_parts(label):
                if part.lower() in self.shelf_asked or not self.code_room(1):
                    continue
                self.shelf_asked.add(part.lower())
                found += [i for i in self.code_search(part) if i not in found]
        if self.step > start:
            self.trace.append({"kind": "shelf_search", "steps": self.step - start})
        return found

    def id_ordered(self, name: str, args: dict[str, Any], observation: Any, param: str) -> bool:
        """A list of a shelf in plain id order: its rows come in ascending id order (and not in price order), no words were
        searched (each text argument is a mode or the rows' own category label) and no price ceiling applies (no number
        argument but the page one that every row's price stays under)."""
        ids = [i for i, _ in self.records_in(observation)]
        if len(ids) < 3 or self.price_sorted(observation):
            return False
        numeric = all(i.isdigit() for i in ids)
        if ids != sorted(ids, key=lambda i: (len(i), i) if numeric else i):
            return False
        props = (self.schemas.get(name) or {}).get("properties") or {}
        cat_key = self.category_key()
        labels = {_words(str((self.rows.get(i) or {}).get(cat_key) or "")) for i in ids} if cat_key else set()
        prices = [p for p in (self.price_of(i) for i in ids) if p is not None]
        for key, value in args.items():
            if key == param:
                continue
            spec = props.get(key) if isinstance(props.get(key), dict) else {}
            if isinstance(value, str) and value.strip() and "enum" not in spec and _words(value) not in labels:
                return False
            if isinstance(value, (int, float)) and not isinstance(value, bool) and (not prices or max(prices) <= value):
                return False
        return True

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
        searched = self.shelf_search()
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
            # the rows Kyo's own shelf search showed come first (cheapest first), up to CHEAPER_OPENS of them
            pool.sort(key=lambda i: (i not in searched, self.price_of(i) or 0.0))
            picks += [i for i in pool[: CHEAPER_OPENS if searched else 2] if i not in picks]
        if not searched:
            return self.fill_facts(picks[:CHEAPER_OPENS], self.tail() + 4)
        # rows a shelf search of Kyo's own brought in: their opens count in its step budget
        before = self.step
        read = self.fill_facts(picks[: self.code_room(CHEAPER_OPENS)], self.tail() + 4)
        self.code_steps += self.step - before
        if self.step > before:
            self.trace.append({"kind": "shelf_opens", "steps": self.step - before})
        return read

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
        cap = self.compare_room()
        if cap < 2:
            self.widen_singly()
            return
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
            if not any(str(i) in _dump(seen) for i in picks):
                self.compare_bad = min(self.compare_bad or 999, len(picks))
            else:
                self.compare_ok = max(self.compare_ok, len(picks))
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
                        if any(str(i) in _dump(reply.get("observation")) for i in picks[:2]):
                            self.compare_ok = max(self.compare_ok, 2)

    def widen_singly(self, keep: int = 22, rows: list[str] | None = None) -> None:
        """No side-by-side view that shows attributes: read a few more listings' attributes singly. Given `rows` (rows
        of an item whose own category the last round had not read), those are read instead, `keep` steps kept back,
        as Kyo's own calls inside the step budget."""
        tool = self.facts_tool()
        target = self.single_target(tool)
        if not tool or not target:
            return
        if rows is not None:
            before = self.step
            read = self.fill_facts(rows[: self.code_room(len(rows))], keep)
            self.code_steps += self.step - before
            self.trace.append({"kind": "dry_reads", "rows": len(rows), "read": len(read)})
            return
        # turns between the item groups, up to two each; with a cheapest pick the cheapest buyable rows first
        lanes: list[list[str]] = []
        for ids in self.item_groups():
            opened = sum(1 for i in ids if i in self.full)
            pool = [i for i in ids if i not in self.full and self.buyable(i)]
            if self.brief.get("pick") == "cheapest":
                pool.sort(key=lambda i: self.price_of(i) if self.price_of(i) is not None else float("inf"))
            # a single item gets more of its listings read: key counts from three or four cards are mostly ties, and
            # before its first question SINGLE_READS cards of its own category rank the carried keys as well as more do
            full = SINGLE_READS if len(self.items) == 1 and not self.asked else 8
            lanes.append(pool[: max(0, min(4 if len(self.items) == 1 else 2, full - opened))])
        queue = [lane[k] for k in range(4) for lane in lanes if k < len(lane)]
        # while steps are plentiful: 22 stay for the rest of the episode
        self.fill_facts(queue[: min(8, max(4, 2 * len(lanes)))], 22)

    def own_rows(self) -> list[list[str]]:
        """Per requested item, the rows seen whose category label fits it, first seen first (none while no seen label
        fits it); with no category field every row counts for every item."""
        cat_key = self.category_key()
        fits: dict[str, list[bool]] = {}
        out: list[list[str]] = [[] for _ in self.items]
        for ident in self.seen_order:
            label = str((self.rows.get(ident) or {}).get(cat_key) or "") if cat_key else ""
            if label not in fits:
                fits[label] = [not cat_key or self.label_fit(label, item) > 0 for item in self.items]
            for number, fit in enumerate(fits[label]):
                if fit:
                    out[number].append(ident)
        return out

    def own_reads(self) -> list[int]:
        """Per requested item, how many listings of its own category were read in full."""
        return [sum(1 for i in ids if i in self.full) for ids in self.own_rows()]

    def read_since_dry(self, need: int) -> bool:
        """Since the round that ran dry, `need` more listings of some item's own category were read in full (of any
        category while no category seen fits a requested item)."""
        rows = self.own_rows()
        if not any(rows):
            return len(self.full) >= self.dry_at + need
        now = [sum(1 for i in ids if i in self.full) for ids in rows]
        then = self.dry_own + [0] * (len(now) - len(self.dry_own))
        return any(n >= t + need for n, t in zip(now, then))

    def starved(self) -> bool:
        """The last round ran dry while some item had fewer than two listings of its own category read: its keys could
        not be ranked (a key counts once two listings carry it), so what it asked came from the request alone."""
        return self.dry_at >= 0 and any(n < 2 for n in self.dry_own)

    def starved_rows(self) -> list[str]:
        """Rows Kyo can read now for a starved item: unread, buyable rows of its own category (of the fitting labels,
        the best fitting one with the most rows seen; cheapest first under a cheapest pick), up to four for a single
        item and two per item in a bundle, while the step budget allows."""
        tool = self.facts_tool()
        if not self.starved() or not tool or not self.single_target(tool) or self.facts_from.get(tool) is False:
            return []
        if self.left() <= self.tail() + 8 or not self.code_room(1):
            return []
        cat_key = self.category_key()
        lanes: list[list[str]] = []
        for number, ids in enumerate(self.own_rows()):
            if number < len(self.dry_own) and self.dry_own[number] >= 2:
                continue
            groups: dict[str, list[str]] = {}
            for ident in ids:
                groups.setdefault(str((self.rows.get(ident) or {}).get(cat_key) or "") if cat_key else "", []).append(ident)
            if not groups:
                continue
            label = max(groups, key=lambda g: (self.label_fit(g, self.items[number]) if cat_key else 0, len(groups[g])))
            pool = [i for i in groups[label] if i not in self.full and i not in self.opened and self.buyable(i)]
            if self.brief.get("pick") == "cheapest":
                pool.sort(key=lambda i: self.price_of(i) if self.price_of(i) is not None else float("inf"))
            lanes.append(pool)
        width = 4 if len(self.items) == 1 else 2
        return [lane[k] for k in range(width) for lane in lanes if k < len(lane)]

    def row_keys(self, number: int) -> list[str]:
        """For an item with fewer than two listings of its category read: the keys its category's search rows carry and
        the rows of other categories never do (the category's own properties, when rows show some), most carried first."""
        cat_key = self.category_key()
        mine = self.own_rows()[number - 1] if cat_key and 0 < number <= len(self.items) else []
        if not mine or len(mine) == len(self.seen_order):
            return []
        own = set(mine)
        other = {key for i in self.seen_order if i not in own for key in (self.rows.get(i) or {})}
        count: dict[str, int] = {}
        for ident in mine:
            for key, value in (self.rows.get(ident) or {}).items():
                if key in other or key in (cat_key, self.price_key, self.stock_key) or isinstance(value, bool):
                    continue
                if (isinstance(value, str) and 0 < len(value.strip()) <= 40) or isinstance(value, (int, float)):
                    name = _words(key)
                    if name:
                        count[name] = count.get(name, 0) + 1
        return [name for name, n in sorted(count.items(), key=lambda kv: -kv[1]) if n >= 2][:LIST_NAMES]

    @_safe(False)
    def blind(self) -> bool:
        """Listing pages were opened, yet none could be read into attributes (a record format Kyo's reader does not
        know): the code's key counts and evidence are empty, so the interview works from the pages' own text."""
        return len(self.opened) >= 2 and not self.full

    def ready_to_ask(self) -> bool:
        if not self.can_ask() or self.left() <= 8:
            return False
        opened = len(self.full)
        if not self.asked and self.dry_at < 0 and self.blind() and len(self.opened) >= 2 * len(self.items) + 1:
            return True
        if self.asked or self.dry_at >= 0:
            # a later round: the last one ran out of properties to ask, and more listings of an item's own category were
            # read since, or an item that had fewer than two of them read then shows rows of it that Kyo can read itself
            return self.read_since_dry(2) or bool(self.starved_rows())
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
        """Ask the shopper one question at a time and read each answer before choosing the next. One plan ranks each
        item's properties; the questions take turns between the items, an item whose answer revealed a preference waits
        until the others had their turn, and an answer that wants a single question gets that property asked alone once
        more. Every message goes to a question (the request's own steps keep theirs). An attempt that could send nothing
        is tried again later, and a round left with messages but no property to ask resumes after more reads."""
        count = self.msgs_free()
        if count <= 0 or not self.ask_time() or not self.roles.get("ask") or self.no_credit:
            return self.ask_missed()
        if self.message_args("?") is None:
            # the message tool takes no question text in this store: nothing can be asked
            self.asked = True
            return ""
        self.seek_own_rows()
        rows = self.starved_rows()
        if rows:
            # the last round ran dry before an item's own category was read: its rows now, fewer steps kept back
            self.widen_singly(self.tail() + 8, rows)
        else:
            self.widen()
        if self.asked and self.dry_at >= 0 and not self.read_since_dry(1) and not self.starved():
            # no listing of an item's own category read since the last round: the plan would name the same properties again
            return self.ask_missed()
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
        # more picks than messages: each item keeps a queue of its own, so a later turn has a next property to ask
        want = count + len(self.items) + 2
        if self.list_mode():
            # a question names up to a list of properties of one item: the plan ranks a full list for every item
            want = max(want, (LIST_NAMES if len(self.items) == 1 else BUNDLE_NAMES) * len(self.items))
        prompt = PLAN.replace("COUNT", str(want)) + "\n\nREQUEST:\n" + self.request + "\n\nITEMS:\n" + "\n".join(settled)
        first = [str(x) for x in _items(self.brief.get("ask_first")) if str(x).strip()]
        if first:
            prompt += "\n\nASK FIRST (the request asks to be asked about these):\n" + "\n".join(first)
        prompt += "\n\nATTRIBUTE KEYS SEEN:\n" + ("\n".join(listing) or "(no full listing records yet)")
        if not listing and self.blind() and self.page_texts:
            prompt += ("\n\nLISTING PAGES AS THE STORE SHOWED THEM (Kyo could not read them into attribute keys; choose the properties"
                       " these pages state for each item's kind, as plain property names):\n" + "\n".join(t[:1200] for t in self.page_texts[-3:]))
        bare = [n for n, done in enumerate(self.own_reads(), 1) if done < 2] if self.starved() else []
        if self.asked and self.asked_labels:
            prompt += "\n\nASKED ALREADY (choose other properties):\n" + "\n".join(f"item {n}: {label}" for n, label in self.asked_labels)
        if bare:
            prompt += ("\n\nFEWER THAN TWO LISTINGS OF ITS OWN CATEGORY READ (choose the properties such listings most often state): "
                       + ", ".join(f"item {n}" for n in bare))
        plan = self.ask_json(prompt, [HEAVY, LIGHT], 1800) or {}
        queues: dict[int, list[str]] = {}
        order: list[int] = []
        for pick in _items(plan.get("ask")):
            if not isinstance(pick, dict):
                continue
            number = min(max(1, _count(pick.get("item") or 1) or 1), len(self.items))
            # one property per question: a label naming two is cut to the first
            label = re.split(r"\s+(?:and|or)\s+|[/,;(]", str(pick.get("label") or ""))[0].strip().rstrip("?.")
            if not label or len(label.split()) > 5:
                continue
            queue = queues.setdefault(number, [])
            if all(_words(label) != _words(other) for other in queue):
                queue.append(label)
            if number not in order:
                order.append(number)
        self.listing = self.list_mode()
        self.top_up(queues, order, table)
        for number in bare:
            # fewer than two listings of the item's category read: the keys its search rows alone carry
            queue = queues.setdefault(number, [])
            queue += [k for k in self.row_keys(number) if all(_words(k) != _words(other) for other in queue)]
            if queue and number not in order:
                order.append(number)
        if self.listing:
            for number, queue in queues.items():
                # a property the request seems to settle is asked last, never dropped
                stated = [_stem(m.split(":")[0]) for m in self.items[number - 1].musts]
                late = [label for label in queue if any(k and (k <= _stem(label) or _stem(label) <= k) for k in stated)]
                queue[:] = [label for label in queue if label not in late] + late
        self.trace.append({"kind": "plan", "ask": _clip(plan.get("ask"), 120), "queues": _clip(queues, 60), "lists": self.listing})
        sent: list[tuple[int, str, str]] = []
        told: list[str] = []
        found: set[int] = {n for n, item in enumerate(self.items, 1) if item.told}
        last_item: int | None = None
        dry = False
        # a question names its item, so a property two items share is still asked about (no exclusion)
        shared: list[frozenset[str]] = []
        held: dict[int, list[str]] = {}
        seen = len(self.full)
        again: tuple[int, list[str], str] | None = None
        while self.msgs_free() > 0 and not self.finished and self.ask_time():
            if self.listing and len(self.full) != seen:
                # listings read since the last question: their keys join the queues
                seen = len(self.full)
                table = self.key_table()
                self.top_up(queues, order, table)
            if again:
                number, names, text = again
                again = None
            else:
                pick = self.next_pick(order, queues, found, last_item)
                if pick is None and self.listing:
                    # no fresh property left: an item's untested ones are asked again
                    spare = next((n for n in order if held.get(n)), 0)
                    pick = (spare, held[spare].pop(0)) if spare else None
                if pick is None:
                    dry = True
                    break
                number, label = pick
                names = [label]
                # a list only when more properties than messages are left, so every message keeps one to ask
                spare = 1 + sum(len(q) for q in queues.values()) + sum(len(h) for h in held.values())
                if self.listing and spare > self.msgs_free():
                    width = LIST_NAMES if len(self.items) == 1 else BUNDLE_NAMES
                    names = self.ask_list([label] + queues.get(number, []), held.get(number, []), width, self.skip_keys(number), shared) or [label]
                    if len(names) > 1 and self.stated(number, names[0]):
                        # a property the request states never leads a list (an answer most often speaks to the first
                        # one named it knows, and a shrug there settles nothing new): such properties are still asked, last
                        late = [n for n in names if self.stated(number, n)]
                        if len(late) < len(names):
                            names = [n for n in names if n not in late] + late
                text = self.question(number, names[0], tuple(names[1:]))
            batch = self.send([(number, names[0], text)])
            if not batch:
                break
            self.asked = True
            self.questions += 1
            last_item = number
            if self.msg_refused:
                # the store refused the message (its real limit is lower than the one read): no more questions
                self.asked_labels.append((number, names[0]))
                self.msg_used -= 1
                self.msg_limit = self.msg_used
                break
            sent += batch
            # one answer: the light model reads it (the heavy one when the light one fails); an answer that wants a
            # single question to a one-property question is the shopper's way of not minding it: move on
            heard = self.hear(batch, (LIGHT, HEAVY))
            news = [str(e.get("need") or "") for e in _items(heard.get("needs")) if isinstance(e, dict) and str(e.get("need") or "").strip()]
            crowded = not news and 1 in [_count(x) for x in _items(heard.get("crowded"))]
            if (not crowded and not news and len(names) > 1 and len(self.items) == 1 and not self.lost_switch
                    and 1 in [_count(x) for x in _items(heard.get("lost"))]):
                # a single item's list drew "say again": with one item there is no other item the answer can be about,
                # so the list itself lost the shopper; one property per question from here (in-episode test)
                self.lost_switch = True
                crowded = True
                self.trace.append({"kind": "lost_switch", "names": names})
            if crowded and len(names) > 1:
                # the shopper wants fewer at once: this list's head alone next, then one property per question
                self.stack_ok = self.listing = False
                self.trace.append({"kind": "stack_off", "names": names})
                held[number] = names[1:] + [x for x in held.get(number, []) if x not in names]
                for other, rest in held.items():
                    queue = queues.setdefault(other, [])
                    queue[:0] = [x for x in rest if x not in queue]
                held = {}
                again = (number, names[:1], self.question(number, names[0]))
                self.take_heard({"budget": heard.get("budget")}, told)
                continue
            if crowded and LISTS and not self.reasked:
                # one property drew a crowded answer (an item noun joining two, a property two items share): it is
                # asked once more with the request's own noun and the property's last word
                self.reasked = True
                self.asked_labels.append((number, names[0]))
                word = names[0].split()[-1]
                again = (number, [word], f"What {word} would you like for the {self.noun(number)}?")
                self.trace.append({"kind": "reask", "q": again[2]})
                self.take_heard({"budget": heard.get("budget")}, told)
                continue
            # the code moves on: the properties up to the one revealed are settled, later ones are asked again
            free_named = [str(e.get("property") or "") for e in _items(heard.get("free")) if isinstance(e, dict)]
            confused = not news and not free_named and 1 in [_count(x) for x in _items(heard.get("lost"))]
            cut = self.retire(names, news, confused)
            self.asked_labels.extend((number, name) for name in names[:cut])
            queues[number] = [x for x in queues.get(number, []) if x not in names]
            rest = names[cut:]
            if news:
                self.shrugs[number] = 0
            elif not confused:
                self.shrugs[number] = self.shrugs.get(number, 0) + 1
            if not news and not confused and self.shrugs.get(number, 0) >= 2 and len(rest) > 2:
                # a second answer in a row that did not mind: both earlier lists met such a property before any need
                # from either end, so a need left among these sits between them: the untested ones are asked from the
                # middle of their first order (the ends follow; no property is dropped)
                ahead = list(reversed(rest))
                half = (len(ahead) + 1) // 2
                rest = ahead[half:] + ahead[:half]
            elif not news:
                # the answer spoke to one property it picked: the untested ones lead the next list, last named first
                rest = list(reversed(rest))
            held[number] = rest + [x for x in held.get(number, []) if x not in names]
            if len(names) > 1:
                self.trace.append({"kind": "ask_list", "item": number, "names": names, "settled": cut})
                # "does not mind" only for the properties this answer settled, never for untested ones
                free = [_stem(str(e.get("property") or "")) for e in _items(heard.get("free")) if isinstance(e, dict)]
                heard = {**heard, "free": [{"item": number, "property": name} for name in names[:cut]
                                           if any(f and (f <= _stem(name) or _stem(name) <= f) for f in free)]}
            before = len(told)
            self.take_heard(heard, told)
            if len(told) > before:
                found.add(number)
        # messages left and no property to ask: one later round, once more listings are read or before the first cart write
        self.dry_at = len(self.full) if dry and self.msgs_free() > 0 else -1
        # listings of each item's own category read by then: the next round waits for more of these, not any read
        self.dry_own = self.own_reads()
        if not sent:
            return self.ask_missed()
        self.rounds += 1
        self.trace.append({"kind": "ask_round", "round": self.rounds, "sent": len(sent), "dry": dry, "left": self.msgs_free()})
        if told:
            self.version += 1
        lines = [f"Q: {q}\nA: {a}" for _, q, a in sent]
        summary = "Kyo's interviewer asked the shopper (each question used one step):\n" + "\n".join(lines)
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

    def stated(self, number: int, label: str) -> bool:
        """The request's own requirements for the item name this property (the plan's "asked last" rule)."""
        mine = _stem(label)
        stated = [_stem(m.split(":")[0]) for m in self.items[number - 1].musts]
        return bool(mine) and any(k and (k <= mine or mine <= k) for k in stated)

    def next_pick(self, order: list[int], queues: dict[int, list[str]], found: set[int], last_item: int | None) -> tuple[int, str] | None:
        """The next question: the item after the one asked last that still has a property queued, items whose answers
        revealed nothing yet first. A property an answer has already settled for that item is not asked again."""
        for number, queue in queues.items():
            if number <= len(self.items):
                item = self.items[number - 1]
                told = [_stem(t.split(":")[0]) for t in item.told]
                stated = [_stem(t.split(":")[0]) for t in item.musts]
                # a property an answer settled is not asked again under any spelling that holds its words; a property
                # the request states is skipped among the code's own top-up keys (the plan already saw the request)
                queue[:] = [label for label in queue
                            if not any(k and k <= _stem(label) for k in told)
                            and not (not self.listing and label in self.topup and any(k and (k <= _stem(label) or _stem(label) <= k) for k in stated))]
                # a narrower spelling of a property already asked about (all its words and more) is not asked again
                asked = [_stem(label) for n, label in self.asked_labels if n == number]
                queue[:] = [label for label in queue if not any(a and a <= _stem(label) for a in asked)]
        live = [n for n in order if queues.get(n)]
        if not live:
            return None
        pool = [n for n in live if n not in found] or live
        start = (order.index(last_item) + 1) if last_item in order else 0
        for step in range(len(order)):
            number = order[(start + step) % len(order)]
            if number in pool:
                return number, queues[number].pop(0)
        return None

    def item_keys(self, number: int, table: list[tuple[str, str, int, list[str]]], limit: int = 4) -> list[str]:
        """The most carried keys of the category that matches an item (its label or the words of its name)."""
        item = self.items[number - 1]
        settled = " ".join(item.musts).lower()
        # only a category this item owns, the one that fits it best: in a bundle another item's shelf can share a
        # word with it
        fits = {label: self.kin_fit(label, item) for label, _, _, _ in table if self.owns(number, label)}
        best = max(fits.values(), default=0)
        shared: list[str] = []
        single: list[str] = []
        for label, name, carried, values in table:
            near = best > 0 and fits.get(label) == best
            if near and carried >= 2 and (self.listing or name not in settled) and name not in shared + single:
                # a value two listings share marks a property; an identifier or a list of purchase options differs on
                # every listing, so keys with fewer distinct values than listings come first
                (shared if self.key_distinct.get((label, name), len(values)) < carried else single).append(name)
        return (shared + single)[:limit]

    def list_mode(self) -> bool:
        """Several properties per question: LISTS is set, no answer has asked for fewer at once, the shopper allows
        three messages or more, and the request does not limit a message to a single question."""
        return LISTS and self.stack_ok and self.msg_limit >= 3 and self.brief.get("one_per_message") is not True

    def top_up(self, queues: dict[int, list[str]], order: list[int], table: list[tuple[str, str, int, list[str]]]) -> None:
        """Every item's queue ends with its category's most carried keys the plan did not name, so no message is left
        unused for want of a property (an item the plan left out gets these alone)."""
        for number in range(1, len(self.items) + 1):
            queue = queues.setdefault(number, [])
            for name in self.item_keys(number, table, 14 if self.list_mode() else 8):
                if all(_words(name) != _words(other) for other in queue):
                    queue.append(name)
                    self.topup.add(name)
            if queue and number not in order:
                order.append(number)

    def skip_keys(self, number: int) -> list[frozenset[str]]:
        """Properties never named again for an item: the ones an answer revealed and the ones already settled."""
        item = self.items[number - 1]
        keys = [_stem(t.split(":")[0]) for t in item.told] + [_stem(label) for n, label in self.asked_labels if n == number]
        return [k for k in keys if k]

    def shared_keys(self, table: list[tuple[str, str, int, list[str]]], queues: dict[int, list[str]]) -> list[frozenset[str]]:
        """Properties two requested items both carry: a list naming one of them leaves the shopper unsure which item
        it is about."""
        if len(self.items) < 2:
            return []
        count: dict[str, int] = {}
        for number in range(1, len(self.items) + 1):
            for name in {_words(k) for k in self.item_keys(number, table, 14) + queues.get(number, [])}:
                count[name] = count.get(name, 0) + 1
        return [_stem(name) for name, n in count.items() if n >= 2 and _stem(name)]

    @staticmethod
    def one_label(text: str) -> str:
        """A planner's label cut to its first property; '' when that is longer than five words."""
        label = re.split(r"\s+(?:and|or)\s+|[/,;(]", str(text or ""))[0].strip().rstrip("?.")
        return label if len(label.split()) <= 5 else ""

    @staticmethod
    def ask_list(fresh: list[str], held: list[str], width: int, skip: list[frozenset[str]], shared: list[frozenset[str]]) -> list[str]:
        """The properties one question names, in order: the first fresh property that may be named heads it, then the
        ones earlier lists left untested, then the next fresh ones. Each is cut to one property, five words at most;
        never one in `skip`, never a second spelling of one already named (all the words of either inside the other),
        and after the head never one in `shared`. A narrower spelling (all the words of a property already named or
        settled, and more) is a second spelling; a wider property is not (\"colour\" after \"leg colour\" is asked)."""
        names: list[str] = []

        def fits(raw: str) -> str:
            label = Kyo.one_label(raw)
            mine = _stem(label)
            if not mine or any(other <= mine for other in skip + [_stem(n) for n in names]):
                return ""
            if names and any(mine <= other or other <= mine for other in shared):
                return ""
            return label

        head = next((label for label in map(fits, fresh) if label), "")
        if head:
            names.append(head)
        for raw in list(held) + list(fresh):
            if len(names) >= width:
                break
            label = fits(raw)
            if label:
                names.append(label)
        return names

    @staticmethod
    def retire(names: list[str], needs: list[str], confused: bool = False) -> int:
        """How many of a question's properties, in order, its answer settles: up to the one it revealed; the first
        only when it revealed another property; every one when the shopper did not follow (no named property is one the
        shopper holds a view on); the first when the shopper did not mind (the answer spoke to one property, most often
        the first; the others stay untested)."""
        if not needs:
            return len(names) if confused else min(1, len(names))
        spots = []
        for index, name in enumerate(names, 1):
            mine = _stem(name)
            for need in needs:
                key = _stem(need.split(":")[0])
                if key and mine and (key <= mine or mine <= key):
                    spots.append(index)
        return max(spots) if spots else 1

    def noun(self, number: int) -> str:
        """The request's own short noun for an item: the brief's, else the item's name cut before a joining word."""
        entries = [e for e in _items(self.brief.get("items")) if isinstance(e, dict)]
        given = str(entries[number - 1].get("noun") or "").strip() if number <= len(entries) else ""
        name = self.items[number - 1].name
        if given and len(given.split()) <= 3:
            return given
        return re.split(r"\s+(?:and|or)\s+|[&/,;(]", name)[0].strip() or name

    def ask_time(self) -> bool:
        """Time for one more question: its answer is read by a model, so the slowest model call so far must fit."""
        return self.time_left() > CLOSING + self.slowest

    def can_ask(self) -> bool:
        """Questions may start: a message tool, a free message and time for an answer; an attempt that sent nothing is
        tried again only at a later step, twice at most; after a round, another one only when the last ran dry
        (properties, not messages, ran out) and no cart line is written yet."""
        if not self.roles.get("ask") or self.msgs_free() <= 0 or not self.ask_time() or self.no_credit:
            return False
        if self.ask_tries >= 2 or self.ask_missed_at == self.step:
            return False
        return not self.asked or (self.dry_at >= 0 and self.rounds < 2 and not self.cart_touched)

    def ask_missed(self) -> str:
        self.ask_tries += 1
        self.ask_missed_at = self.step
        self.trace.append({"kind": "ask_missed", "step": self.step, "tries": self.ask_tries})
        return ""

    def finish_mark(self, calls: list[dict[str, Any]]) -> str:
        """A lone checkout or ending call with the cart and the requirements as they are now; '' for anything else."""
        if len(calls) != 1:
            return ""
        fn = calls[0].get("function") or {}
        name = str(fn.get("name") or "")
        raw = fn.get("arguments")
        args = raw if isinstance(raw, dict) else (_json_obj(raw) if isinstance(raw, str) and raw.strip() else {})
        args = args if isinstance(args, dict) else {}
        if not (self.is_buy(name, args) or self.is_refuse(name, args)):
            return ""
        return name + _dump(args) + "|" + self.cart_mark() + f"#{self.version}"

    def count_finish(self, mark: str, replies: list[tuple[str, str]]) -> None:
        """Finishing calls the store turned down in a row, each the same call on the same cart and requirements (a held
        or failed call was not sent and counts for nothing)."""
        if not mark or self.finished or not replies or replies[-1][1].startswith(("Not run", "The call failed")):
            return
        self.finish_refused = self.finish_refused + 1 if mark == self.last_finish else 1
        self.last_finish = mark

    def question(self, number: int, label: str, more: tuple[str, ...] = ()) -> str:
        names = [label, *more]
        said = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " or " + names[-1]
        name = self.items[number - 1].name
        if LISTS and re.search(r"\s(?:and|or)\s|[&/]", name):
            # a name joining two nouns reads as two products: the request's own short noun instead
            name = self.noun(number)
        name = re.sub(r"^(?:a|an|the|some)\s+", "", name.strip(), flags=re.I) or name
        return f"What {said} would you like for the {name}?"

    def send(self, questions: list[tuple[int, str, str]], floor: int = 6) -> list[tuple[int, str, str]]:
        tool = self.roles["ask"]
        sent: list[tuple[int, str, str]] = []
        for number, _, text in questions:
            if self.finished or self.left() <= floor or self.msg_used >= self.msg_limit or self.no_credit:
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
            self.msg_refused = isinstance(answer, dict) and bool(answer.get("error"))
            if self.msg_refused:
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
        heard = self.ask_json(prompt, list(models), 1500) or {}
        self.exclusion_net(heard, sent)
        return heard

    def exclusion_net(self, heard: dict[str, Any], sent: list[tuple[int, str, str]]) -> None:
        """The answer reader writes a value the shopper turns away as "<property>: not <value>". An answer that turns away
        every value except one asks for that one value, and the reader still writes a few of those as exclusions. So
        each such need (not one that points at another listing) gets one yes/no check on the answer's own words; when
        the answer turns away every other value, the need becomes "<property>: <value>". A failed, skipped or capped
        check keeps the reader's need."""
        for entry in _items(heard.get("needs")):
            if not isinstance(entry, dict) or entry.get("points") is True:
                continue
            need = str(entry.get("need") or "").strip()
            key, _, said = need.partition(":")
            words = said.split()
            # only the reader's own "not <value>" wording, with a value
            if not key.strip() or len(words) < 2 or words[0].lower() != "not":
                continue
            value = said.strip()[len(words[0]):].strip()
            number = _count(entry.get("item")) or 1
            pairs = [(q, a) for n, q, a in sent if n == number] or [(q, a) for _, q, a in sent]
            if not pairs:
                continue
            prompt = self.exclusion_prompt(key.strip(), value, pairs)
            if prompt not in self.exclude_memo:
                self.exclude_memo[prompt] = self.exclusion_ask(prompt)
            asks_for = self.exclude_memo[prompt]
            self.trace.append({"kind": "exclusion", "need": need[:160], "asks_for": asks_for, "calls": self.exclude_calls})
            if asks_for is True:
                entry["need"] = f"{key.strip()}: {value}"

    def exclusion_prompt(self, key: str, value: str, pairs: list[tuple[str, str]]) -> str:
        return (EXCLUDE + f"\n\nPROPERTY: {key}\nVALUE: {value}\n\n"
                + "\n".join(f"QUESTION: {q[:300]}\nANSWER: {a[:800]}" for q, a in pairs))

    def exclusion_ask(self, prompt: str) -> bool | None:
        """One light yes/no call; the heavy model only when the light reply gives no true or false. None when no call
        may start (spend stop, last-call time, the call cap, the check's own cap) or no reply gives one."""
        for model in (LIGHT, HEAVY):
            if (self.exclude_calls >= EXCLUDE_CALLS or self.calls >= self.call_cap or self.cost >= SPEND_STOP
                    or self.time_left() < LAST_CALL or self.no_credit or self.broke or self.out_of_time):
                return None
            before = self.calls
            got = self.ask_json(prompt, [model], 300)
            self.exclude_calls += self.calls - before
            if isinstance((got or {}).get("asks_for"), bool):
                return got["asks_for"]
        return None

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
        return True, ("Before going on, Kyo did the steps the request asks for:\n" + "\n".join(said)) if said else ""

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
            self.rearm()
        kept = self.price_keep() if changed == ["a price changed"] and got.get("sold_out") is not True else []
        if kept:
            self.trace.append({"kind": "price_kept", "lines": kept})
            changed.append(f"only a price moved: {', '.join(kept)} still meet every requirement the records show and fit the budget at the prices seen, so keep them and check out the same cart again")
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
        return self.proof(ids)

    def proof(self, ids: list[str]) -> str:
        """The [evidence] line for opened listings: per requirement the keys naming its property, with the code's mark."""
        needs = self.needs_now()
        if not ids or not needs:
            return ""
        many = len(self.items) > 1
        lines = []
        for ident in ids[:12]:
            block = _fact_block(self.full.get(ident) or {}) or {}
            marks = {need: (state, key) for need, state, key in self.item_marks(ident)}
            parts = []
            for number, need in needs[:8]:
                prop = need.split(":", 1)[0].strip()
                hits = [f"{k}={str(v)[:50]}" for k, v in block.items() if _names(_stem(prop), _stem(k))]
                tag = f"(item {number}) " if many else ""
                state, key = marks.get(need, ("", ""))
                if state in ("met", "fails"):
                    parts.append(f"{tag}{prop} -> " + (", ".join(hits[:3]) or (f"{key}={str(block.get(key))[:50]}" if key else "the category label")) + f" ({state})")
                elif state == "unproven":
                    parts.append(f"{tag}{prop} -> no such key, and no other attribute holds the value (unproven; a title alone does not count here)")
                elif state == "choice":
                    parts.append(f"{tag}{prop} -> only among the choices {key}={str(block.get(key))[:50]} (not a stated value)")
                elif hits:
                    parts.append(f"{tag}{prop} -> " + ", ".join(hits[:3]))
            if parts:
                lines.append(f"{ident}: " + "; ".join(parts))
        if not lines:
            return ""
        return ("[evidence: keys whose names contain a requirement's property; one naming a different property (another part or aspect) decides nothing, and a key with another name may still hold the same property."
                " Marks: met = such a key states the value; fails = it states another value that listings of this kind use (still a fit when that value plainly means the same thing); unproven = no such key while three or more opened listings of its kind carry one, and no other attribute holds the value (for an amount: no figure in its unit; a title alone does not count then), so the listing does not meet that requirement] " + " | ".join(lines))

    # ---------- checks ----------
    def cart_text(self) -> str:
        if self.cart_view is not None and self.cart_seen_at >= self.cart_at:
            return _dump(_clip(self.cart_view, 300))[:3000]
        # a cart view older than the last cart change would show lines that are gone
        return "lines by Kyo's record of the cart calls (listing: units): " + _dump(self.cart_lines)

    def cart_ids(self) -> list[str]:
        if self.cart_view is not None and self.cart_seen_at >= self.cart_at:
            text = _dump(self.cart_view)
            return [i for i in self.rows if i in text]
        return list(self.cart_lines)

    def record_of(self, ident: str) -> str:
        record = self.full.get(ident) or self.rows.get(ident)
        if record:
            return _dump(_clip(record, 400))[:3000]
        # a record Kyo could not read into fields: the store's own text around the id, newest first
        for text in reversed(self.raw_seen):
            at = text.find(ident)
            if at >= 0:
                return "(as the store wrote it) " + text[max(0, at - 40): at + 2400]
        return "(no record seen: Kyo has no copy of this listing's record, which says nothing against it)"

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

    def stem_of(self, key: str) -> frozenset[str]:
        if key not in self.stem_memo:
            self.stem_memo[key] = _stem(key)
        return self.stem_memo[key]

    def need_terms(self, text: Any) -> list[str]:
        """Words and figures of a text in lower case (a figure stuck to a word is split from it, a plural s dropped)."""
        return [w[:-1] if len(w) > 3 and w.endswith("s") else w for w in re.findall(r"[a-z]+|\d+(?:\.\d+)?", str(text or "").lower())]

    def holds(self, text: Any, want: list[str]) -> bool:
        """The text carries every wanted word, or the wanted words written together as one to three of its words."""
        got = self.need_terms(text)
        if not want or not got:
            return False
        if set(want) <= set(got):
            return True
        joined = "".join(want)
        return any("".join(got[i:i + n]) == joined for n in (1, 2, 3) for i in range(len(got) - n + 1))

    def want_terms(self, need: str) -> tuple[list[str], bool]:
        """The words a listing must carry for a requirement's value, and whether that value holds a figure: then only its
        figures and the word after each (its unit, as the requirement writes it), since comparing amounts is the model's."""
        words = self.need_terms(need.split(":", 1)[1] if ":" in need else "")
        # words that only phrase the requirement ("include low fat", "must be red") are not part of the value
        words = [w for w in words if w not in _PHRASING] or words
        spots = [i for i, w in enumerate(words) if w[0].isdigit()]
        if not spots:
            return words, False
        return [words[i] for i in spots] + [words[i + 1] for i in spots if i + 1 < len(words) and not words[i + 1][0].isdigit()], True

    def sized(self, text: Any, units: list[str]) -> bool:
        """The text states some figure followed by a unit word of the requirement (the same word, or one that starts the other):
        presence only, since whether that amount meets the requirement is the model's call."""
        got = self.need_terms(text)
        return any(a[0].isdigit() and any(b == u or min(len(b), len(u)) >= 3 and (b.startswith(u) or u.startswith(b)) for u in units)
                   for a, b in zip(got, got[1:]))

    def name_key(self, cat_key: str | None) -> str | None:
        """The record key holding a listing's title: on most opened listings a text of two words or more, not the same on
        most listings, the shortest such text on average (a description is longer, a shop's name is shared)."""
        records = list(self.full.values())
        best: tuple[float, str] | None = None
        for key in sorted({k for r in records for k, v in r.items() if isinstance(v, str) and k != cat_key}):
            texts = [r[key] for r in records if isinstance(r.get(key), str) and len(r[key].split()) >= 2]
            if not texts or len(texts) < 0.8 * len(records) or len(set(texts)) * 2 <= len(texts):
                continue
            mean = sum(len(t.split()) for t in texts) / len(texts)
            if best is None or mean < best[0]:
                best = (mean, key)
        return best[1] if best else None

    def need_mark(self, record: dict[str, Any], need: str, label: str, peers: list[dict[str, Any]], title: str | None) -> tuple[str, str]:
        """How one opened listing stands on one requirement, from its own record against the attributes of the other
        opened listings of its category (`peers`): ("met", key) a key naming the property, or the category label, states
        the value (with a figure: every word of it); ("fails", key) such a key states another value while a listing in that category states the wanted one;
        ("unproven", "") no key names the property though three or more of those listings carry one, and neither an
        attribute nor the title holds the wanted words, at most three (a description never counts then; with a figure in the
        value only presence is read: any figure in the requirement's unit word keeps the mark open); ("choice", key) the
        wanted words appear only inside a list of variants on offer; ("open", key) anything else, every comparison of
        figures among it, is the model's to weigh."""
        block = _fact_block(record) or {}
        prop = self.stem_of(need.split(":", 1)[0])
        want, figure = self.want_terms(need)
        if not block or not prop or not want:
            return "open", ""
        if label and not figure and self.holds(label, want):
            return "met", ""
        named = [k for k in block if _names(prop, self.stem_of(k))]
        exact = [k for k in named if self.stem_of(k) == prop]
        if exact and not figure:
            # a key spelled as the requirement's own property (spacing and case aside) decides over keys that add words
            # (measured on the practice grader: 14 of 17 conflicting purchases, against 12 for "any key")
            hit = next((k for k in exact if self.holds(block[k], want)), None)
            if hit:
                return "met", hit
            if all(not str(block[k]).strip() for k in exact):
                named = [k for k in named if k not in exact]
            else:
                used = any(self.holds(v, want) for p in peers for k, v in p.items() if self.stem_of(k) == prop)
                return ("fails" if used else "open"), exact[0]
        if named:
            # a figure in the value: met only when the key holds every word the requirement writes; else the model decides
            whole = self.need_terms(need.split(":", 1)[1]) if figure else want
            hit = next((k for k in named if self.holds(block[k], whole)), None)
            if hit and not figure and any(not self.holds(block[k], whole) and str(block[k]).strip() for k in named):
                # keys naming the property disagree (one holds the value, another states something else): the model
                # weighs it; never a clean "met" for the code's own carts, never a "fails"
                return "open", hit
            if figure or hit:
                return ("met" if hit else "open"), hit or named[0]
            used = any(self.holds(v, want) for p in peers for k, v in p.items() if _names(prop, self.stem_of(k)))
            return ("fails" if used else "open"), named[0]
        inside = [k for k, v in block.items() if self.holds(v, want)]
        if inside:
            return ("choice" if all(" / " in str(block[k]) for k in inside) else "open"), inside[0]
        carried = sum(1 for p in peers if any(_names(prop, self.stem_of(k)) for k in p))
        shown = record.get(title) if title else " ".join(v for v in record.values() if isinstance(v, str) and len(v.split()) <= 60)
        if self.holds(shown, want) and not (carried >= 3 and not figure and len(want) <= 3):
            # the title counts only while the listings of its kind carry no key for the property
            return "open", title or ""
        units = [w for w in want if not w[0].isdigit()]
        if figure and (not units or any(self.sized(v, units) for v in [*block.values(), shown])):
            return "open", ""
        carried = sum(1 for p in peers if any(_names(prop, self.stem_of(k)) for k in p))
        return ("unproven" if carried >= 3 and len(want) <= 3 else "open"), ""

    def line_item(self, ident: str, cat_key: str | None) -> int | None:
        """The requested item a listing belongs to: the one item, else the only item its category label fits best."""
        if len(self.items) == 1:
            return 1
        label = str((self.rows.get(ident) or {}).get(cat_key) or "") if cat_key else ""
        fits = [self.label_fit(label, item) for item in self.items] if label else []
        best = max(fits, default=0)
        return fits.index(best) + 1 if best and fits.count(best) == 1 else None

    def item_marks(self, ident: str, number: int | None = None) -> list[tuple[str, str, str]]:
        """(requirement, outcome, key) for each live requirement of the item that an opened listing belongs to, which the code
        can read: not one whose value lives on another listing, nor one naming another requested item (a pair)."""
        record = self.full.get(ident)
        cat_key = self.category_key()
        number = number or self.line_item(ident, cat_key)
        label = str((record or {}).get(cat_key) or "") if cat_key else ""
        if not record or not number or (not label and len(self.items) > 1):
            return []
        peers = [_fact_block(r) or {} for i, r in self.full.items() if i != ident and (not cat_key or str(r.get(cat_key) or "") == label)]
        title = self.name_key(cat_key)
        pointed = {need for _, need in self.refs} | set(self.ref_values)
        others = [set(self.need_terms(it.name)) for n, it in enumerate(self.items, 1) if n != number]
        out = []
        for n, need in self.needs_now():
            if n != number or need in pointed or any(o and o <= set(self.need_terms(need)) for o in others):
                continue
            out.append((need, *self.need_mark(record, need, label, peers, title)))
        return out

    def against(self, ident: str, number: int | None = None) -> list[str]:
        """The live requirements an opened listing fails, or leaves unproven, by its own record."""
        return [need for need, state, _ in self.item_marks(ident, number) if state in ("fails", "unproven")]

    def met_all(self, number: int | None, label: str, skip: list[str]) -> list[str]:
        """Opened, buyable listings of one category whose every live requirement of the item the code finds met, cheapest
        first, with their prices."""
        cat_key = self.category_key()
        qty = self.items[number - 1].qty if number and number <= len(self.items) else 1
        alone = len(self.items) == 1 or self.brief.get("budget_scope") == "each"
        found = []
        for ident, record in self.full.items():
            price = self.price_of(ident)
            if ident in skip or not self.buyable(ident) or (cat_key and str(record.get(cat_key) or "") != label):
                continue
            if alone and isinstance(self.budget, (int, float)) and price is not None and price * qty > self.budget:
                continue
            marks = self.item_marks(ident, number)
            if marks and all(state == "met" for _, state, _ in marks):
                found.append((price if price is not None else float("inf"), ident))
        return [i if p == float("inf") else f"{i} ({p:g})" for p, i in sorted(found)[:3]]

    @_safe("")
    def kind_hold(self) -> str:
        """Code only, once per cart (no step): a cart line from a store category that covers fewer of its item's own words
        than another category the store showed, whose name lies wholly inside the item's words and which has a buyable
        listing: the line's category is a neighbouring one ("Jeans" for "pants or jeans" beside "Pants & Jeans")."""
        cat_key = self.category_key()
        mark = self.cart_mark() + f"#{self.version}"
        if not cat_key or mark in self.kind_held or self.left() < self.tail() + 3:
            return ""
        labels: dict[str, list[str]] = {}
        for ident, row in self.rows.items():
            label = str(row.get(cat_key) or "").strip()
            if label:
                labels.setdefault(label, []).append(ident)
        joins = {"or", "and", "the", "a", "an", "for", "of", "some", "&", "/", "+", "-"}
        parts = []
        for ident in self.cart_ids():
            number = self.item_of(ident, cat_key)
            if not number or number > len(self.items):
                continue
            item = self.items[number - 1]
            mine = set(_stem(item.name) | _stem(item.shelf or "")) - joins
            here = str((self.rows.get(ident) or {}).get(cat_key) or "")
            cover = lambda label: len((set(_stem(label)) - joins) & mine)
            whole = [label for label in labels if (set(_stem(label)) - joins) and (set(_stem(label)) - joins) <= mine
                     and cover(label) > cover(here) and any(self.buyable(i) for i in labels[label])]
            if whole:
                best = max(whole, key=cover)
                parts.append(f"{ident} is in the store category {here!r}, which covers only part of item {number} ({item.name}), while the category {best!r} matches it as a whole")
        if not parts:
            return ""
        self.kind_held.add(mark)
        self.trace.append({"kind": "kind_hold", "lines": parts})
        return ("Kyo's reading of the store's categories: " + " | ".join(parts) + ". A neighbouring category is the wrong kind of product:"
                " choose that item from the category that matches it as a whole, then check out.")

    def need_hold(self) -> str:
        """Code only, before a checkout (no step, no model call, regardless of the spend limits of the model checks): a cart
        line that fails, or leaves unproven, a live requirement holds the checkout once per cart and version, naming the
        cheapest opened listings of that line's category whose requirements the code finds met."""
        ids = self.cart_ids()
        mark = self.check_key(ids)
        if mark in self.need_held or self.cost >= SPEND_STOP or self.time_left() <= LAST_CALL or self.left() < self.tail() + 2:
            return ""
        cat_key = self.category_key()
        parts = []
        for ident in ids:
            number = self.line_item(ident, cat_key)
            bad = [(need, state, key) for need, state, key in self.item_marks(ident, number) if state in ("fails", "unproven")] if not self.sold_out(ident) else []
            if not bad:
                continue
            block = _fact_block(self.full.get(ident) or {}) or {}
            better = self.met_all(number, str((self.rows.get(ident) or {}).get(cat_key) or "") if cat_key else "", ids)
            why = [f"its {key} says {str(block.get(key))[:60]!r}, not {need} (a fault unless that value plainly means the same thing)" if state == "fails" else
                   f"its record never shows {need} (no attribute for it while other opened listings in its category carry one, and neither another attribute nor the title holds that value or, for an amount, any figure in its unit)" for need, state, key in bad]
            parts.append(f"{ident}: " + "; ".join(why) + (". Opened listings in its category whose attributes show every requirement, cheapest first: " + ", ".join(better) if better
                         else ". No opened listing in its category shows every requirement yet: search with the wanted value and the product noun, and open listings whose attributes state it"))
        if not parts:
            return ""
        self.need_held.add(mark)
        self.trace.append({"kind": "need_hold", "cart": ids, "lines": parts})
        return ("Kyo's reading of the records says the cart does not show every requirement: " + " | ".join(parts)
                + ". Swap such a line for a listing whose own attributes show every requirement, then check out; when no such listing is found, check out this cart (a doubtful line is better than no purchase).")

    def needs_sig(self) -> str:
        """The requirements and the budget in force as one text: a change to either starts a new needs version."""
        return "|".join(f"{n}:{need}" for n, need in self.needs_now()) + f"#{self.budget}"

    def price_keep(self) -> list[str]:
        """After a note that only moved a price: the cart's lines, when none is sold out, none fails or leaves unproven a
        requirement, and at the newest prices seen the cart is inside the budget; those lines are kept for this needs version."""
        ids = self.cart_ids()
        if not ids or any(self.sold_out(i) or self.against(i) for i in ids):
            return []
        costs = [(self.price_of(i) or 0.0) * self.cart_lines.get(i, 1) for i in ids]
        if isinstance(self.budget, (int, float)) and (max(costs) if self.brief.get("budget_scope") == "each" else sum(costs)) > self.budget:
            return []
        sig = self.needs_sig()
        for ident in ids:
            self.kept[ident] = sig
        return ids

    def keep_line(self, name: str, args: dict[str, Any]) -> str:
        """A cart call that would remove a line kept after a price-only note is held back, once per line and needs
        version (no step used): the same cart sent again is what such a note asks for."""
        sig = self.needs_sig()
        kept = [i for i, s in self.kept.items() if s == sig and (i, sig) not in self.kept_held and i in self.cart_ids()]
        if name != self.roles.get("cart") or not kept:
            return ""
        saved = (dict(self.cart_lines), self.cart_touched, self.cart_at)
        self.note_cart(name, args)
        after = dict(self.cart_lines)
        self.cart_lines, self.cart_touched, self.cart_at = saved
        lost = [i for i in kept if after.get(i, 0) <= 0 and (saved[0].get(i, 0) > 0 or i in _dump(args))]
        if not lost:
            return ""
        self.kept_held.update((i, sig) for i in lost)
        self.trace.append({"kind": "kept_line", "lines": lost})
        return (f"Not run (no step used): only a price moved, and {', '.join(lost)} still meet every requirement the records show and fit the budget at the prices seen,"
                " so keep them and check out the same cart again (a checkout sent again after such a note is usually accepted). Cart now: " + self.cart_text())

    def keep_hold(self) -> str:
        """An ending without a purchase when the cart holds a line for each item, all kept after a price-only note: held
        back once per needs version (no step used)."""
        sig = self.needs_sig()
        ids = self.cart_ids()
        if not ids or len(ids) < len(self.items) or any(self.kept.get(i) != sig for i in ids) or ("", sig) in self.kept_held:
            return ""
        self.kept_held.add(("", sig))
        self.trace.append({"kind": "kept_cart", "lines": ids})
        return ("Only a price changed after the cart was chosen: its lines (" + ", ".join(ids) + ") still meet every requirement the records show and fit the budget at the prices seen."
                " Check out the same cart again; end without a purchase only if a line now breaks the budget or a requirement.")

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
        proof = self.proof([i for i in ids if i in self.full])
        if proof:
            prompt += "\n" + proof
        depts = self.dept_lines(ids)
        if depts:
            prompt += "\n\nCATEGORIES (each cart line's store category; per item, the store categories seen whose names share the item's words):\n" + depts
        if self.brief.get("pick") in ("cheapest", "most", "least"):
            prompt += "\n\nOTHER LISTINGS SEEN (newest first):\n" + self.others(ids, 15)
        got = self.ask_json(prompt, self.checker(), 1500)
        self.trace.append({"kind": "check", "cart": ids, "got": got})
        self.verdicts[key] = bool(got) and got.get("ok") is not False
        if not got or got.get("ok") is not False:
            return ""
        self.leads = [i for i in self.leads if i not in ids]
        problems = [str(p) for p in _items(got.get("problems"))][:5]
        better = [str(b) for b in _items(got.get("better")) if str(b) in self.rows and not self.sold_out(str(b)) and not self.against(str(b))][:4]
        # the check's own picks lead the next fill of the cart (before the code's ranking)
        self.check_better = better + [b for b in self.check_better if b not in better]
        text = "A careful review says this cart is not right yet: " + "; ".join(problems or ["see the requirements"])
        if better:
            text += ". Listings that look better: " + ", ".join(better)
        return text + ". Fix the cart (or show the review is wrong by re-reading the records), then check out."

    def dept_lines(self, ids: list[str]) -> str:
        """Each cart line's store category, and per item the categories seen whose names share the item's words: a
        neighbouring category (a near kind of product on another shelf) shows up next to the right one."""
        key = self.category_key()
        if not key:
            return ""
        labels: list[str] = []
        for ident in self.seen_order:
            label = str((self.rows.get(ident) or {}).get(key) or "").strip()
            if label and label not in labels:
                labels.append(label)
        out = []
        for ident in ids:
            label = str((self.rows.get(ident) or {}).get(key) or "").strip()
            if label:
                out.append(f"{ident}: category {label!r}")
        for number, item in enumerate(self.items, 1):
            near = sorted((label for label in labels if self.label_fit(label, item)), key=lambda label: -self.label_fit(label, item))
            if near:
                out.append(f"item {number} ({item.name}): categories seen whose names share its words: " + ", ".join(repr(l) for l in near[:4]))
        return "\n".join(out)

    def trail(self, force: bool = False) -> tuple[str, bool]:
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
        # before an ending without a purchase it runs whatever the spend and the time, short of the last call's limits
        while pending and self.trails < 6 and ((self.cost < SPEND_STOP and self.time_left() > LAST_CALL) if force else (self.cost < SPEND_HARD and self.time_left() > CLOSING)):
            self.trails += 1
            records = []
            newest = list(reversed(self.seen_order))
            # every opened page stays in view (a chain's first page is often read early in a long episode), then the
            # newest search rows up to 60 records in all, all in their newest-first order
            keep = set([i for i in newest if i in self.opened and (self.full.get(i) or self.rows.get(i))][:45])
            for ident in newest:
                if len(keep) >= 60:
                    break
                keep.add(ident)
            for ident in newest:
                if ident not in keep:
                    continue
                # an opened page is the page view's record (with the attributes when they were read too)
                record = (self.full.get(ident) or self.rows.get(ident)) if ident in self.opened else None
                if record:
                    records.append("OPENED " + _dump(_clip(record, 300))[:1400])
                else:
                    records.append("ROW " + _dump(_clip(self.rows.get(ident), 80))[:300])
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
                    if str(chain.get("rule_value") or "").strip():
                        self.ref_rules[need] = str(chain["rule_value"]).strip()
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
            text += "[Kyo opened the pages a requirement points at]\n" + "\n".join(seen) + "\n"
        if self.ref_values:
            text += "[values read from the pages requirements point at] " + "; ".join(f"{k} -> {v}" for k, v in self.ref_values.items()) + "\n"
        hold = bool(lost) and self.trail_blocks < 3 and self.left() > 5 and self.time_left() > CLOSING
        if hold:
            self.trail_blocks += 1
            text += ("The order counts only after every page a requirement points at has been opened in full (a search row is not enough), following each page that refers to a further listing. Still to find and open: "
                     + " | ".join(lost[:3]) + ". Search for it by its shop, brand or words, open it, apply the value its page gives, then check out.")
        return text, hold

    def offer_tool(self) -> str | None:
        """Tool for re-reading a listing's live offer: the offer view when the store has one; else the single-listing
        view when it shows stock; else a single-listing tool seen showing the stock flag (learned in the episode); else
        the single-listing view."""
        if self.roles.get("offer"):
            return self.roles["offer"]
        opener = self.roles.get("open")
        if opener and (opener in self.stock_views or opener not in self.flagless_views):
            # the single-listing view shows stock, or nothing shows yet that it does not
            return opener
        busy = {self.roles.get(j) for j in ("cart", "buy", "refuse", "ask", "search", "compare", "cart_view")}
        learned = next((t for t in self.stock_views if t not in busy and self.single_target(t)), None)
        if learned:
            return learned
        # none seen yet: a single-listing tool with no other job whose own description speaks of stock or the offer
        for spec in self.tools:
            fn = spec["function"]; name = fn["name"]
            if name not in busy and name != opener and self.single_target(name) and re.search(r"stock|availab|offer|in.?store|on.?hand", str(fn.get("description") or ""), re.I):
                return name
        return opener

    @_safe(None)
    def note_stock_view(self, name: str, observation: Any) -> None:
        """A single-listing tool whose record carries the store's stock flag (the stock key of the result rows, or a
        true/false field those rows also carry) can show a silent stock change: remembered for the re-read."""
        if name in self.stock_views or not isinstance(observation, dict) or not self.single_target(name):
            return
        record = observation if any(not isinstance(v, (dict, list)) for v in observation.values()) else next(
            (v for v in observation.values() if isinstance(v, dict)), {})
        flags = {k for k, v in record.items() if isinstance(v, bool)}
        if not flags:
            if record:
                self.flagless_views.add(name)
            return
        if (self.stock_key and self.stock_key in flags) or any(k in flags for row in list(self.rows.values())[:40] for k, v in row.items() if isinstance(v, bool)):
            self.stock_views.add(name)

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
        cat_key = self.category_key()
        for ident in lost:
            # the cheaper sweep of that category runs again at the next checkout, below the replacement's price
            self.capped_done.discard(str((self.rows.get(ident) or {}).get(cat_key) or "").strip().lower() if cat_key else "")
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
        got = self.ask_json(prompt, self.checker(), 1200)
        fits: list[tuple[float, str]] = []
        for entry in _items((got or {}).get("fits")):
            ident = str(entry.get("id") if isinstance(entry, dict) else entry)
            price = entry.get("price") if isinstance(entry, dict) else None
            if self.price_of(ident) is not None:
                # the store's own price as Kyo's ledger has it, not the model's reading of it
                price = self.price_of(ident)
            if ident in self.rows and ident not in lost and ident not in self.gone and ident not in self.cart_ids() and ident not in [i for _, i in fits] and not self.against(ident):
                fits.append((float(price) if isinstance(price, (int, float)) and not isinstance(price, bool) else float("inf"), ident))
        # sorted here: the model names fits well but orders them by price unreliably
        ranked = [i for _, i in sorted(fits, key=lambda f: f[0])] if self.brief.get("pick") == "cheapest" else [i for _, i in fits]
        self.trace.append({"kind": "swap", "lost": lost, "fits": fits})
        return ranked[:4]

    def audit(self) -> str:
        kept = self.keep_hold()
        if kept:
            return kept
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
        proof = self.proof([i for i in reversed(self.seen_order) if i in self.full and not self.sold_out(i)][:12])
        if proof:
            prompt += "\n" + proof
        got = self.ask_json(prompt, self.checker(), 1500)
        self.trace.append({"kind": "audit", "got": got})
        return got

    def audit_fits(self, got: dict[str, Any] | None) -> dict[str, list[str]]:
        fits = (got or {}).get("fits")
        if not isinstance(fits, dict):
            return {}
        named = {str(k): [str(i) for i in _items(v) if str(i) in self.rows and not self.sold_out(str(i)) and not self.against(str(i), _count(k))][:4] for k, v in fits.items()}
        return {k: v for k, v in named.items() if v}

    def open_rows(self, rows: list[str]) -> bool:
        """Open listings seen only as search rows: side by side in one step when the store can, else the first two one by one."""
        many, joiner = self.compare_target(), self.roles.get("compare")
        tool = self.facts_tool()
        target = self.single_target(tool)
        if many and joiner and not self.compare_bare():
            plan = [(joiner, {many: rows[: self.compare_room()]})]
        elif tool and target:
            # more rows when steps and time are plentiful: a decline is wrong whenever a fitting row was left unread
            room = 4 if self.left() >= 15 and self.time_left() > CLOSING + 30 else 2
            plan = [(tool, {target: i}) for i in rows[:room]]
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
            if name == joiner and isinstance(args.get(many), list) and len(args[many]) > 1:
                if any(str(i) in _dump(reply.get("observation")) for i in args[many]):
                    self.compare_ok = max(self.compare_ok, len(args[many]))
                else:
                    self.compare_bad = min(self.compare_bad or 999, len(args[many]))
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
            calls.append({target: ident, **({mode: props[mode]["enum"][0]} if mode else {}), **({count: want} if count else {})})
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
            pick = next((i for i in ids if not self.sold_out(i) and not self.against(i, index)), None)
            if pick and index and 1 <= index <= len(self.items) and kind(pick) not in {kind(p) for p in plan}:
                plan[pick] = self.items[index - 1].qty
        if self.confirmed not in plan:
            return ""
        now = self.cart_ids()
        keep = {i: self.cart_lines.get(i, 1) for i in now if i not in plan and not self.sold_out(i) and kind(i) not in {kind(p) for p in plan} and not self.against(i)}
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
            self.book_cart(cart, args, reply.get("observation"))
            self.observe(cart, reply.get("observation"))
            done.append(_dump(args))
        faults = []
        args = _shape(self.order_args(buy), self.schemas.get(buy), "args", faults)
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
    def order_args(self, buy: str) -> dict[str, Any]:
        """The checkout's arguments for the code's own checkout: the brief's fixed ones, plus a short neutral note in each
        required free-text argument (a note, reason, justification, comment or message; never an id) the brief left out,
        so a missing explanation field does not turn the code's checkout into an ending without a purchase (the ending's
        own call fills its text arguments the same way)."""
        fixed = self.brief.get("buy_args")
        out = dict(fixed) if isinstance(fixed, dict) else {}
        schema = self.schemas.get(buy) or {}
        props = schema.get("properties") or {}
        for key in schema.get("required") or []:
            spec = props.get(key)
            if key in out or not isinstance(spec, dict) or "enum" in spec or "string" not in _kinds(spec):
                continue
            if _NOTE_ARG.search(key) and not _ID_ARG.search(key):
                out[key] = "Placing the order for the listings in the cart."[: spec.get("maxLength", 200)]
        return out

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
                # a removal without a count takes off one unit (the store's default), not the whole line
                have = have - (count if count else 1)
            elif sets:
                have = count if count is not None else 1
            else:
                have = have + (count if count is not None else 1)
            if have > 0:
                self.cart_lines[ident] = have
            else:
                self.cart_lines.pop(ident, None)

    def sync_cart(self, observation: Any, name: str | None = None, args: dict[str, Any] | None = None) -> str:
        """Read the cart from a cart tool's own reply when it lists the cart's lines (listing records, each with one
        small count): "synced" when that list was read and is now Kyo's record, "unreadable" when such a list is there
        but its counts cannot be told apart from other small numbers, "none" when the reply carries no such list."""
        lists: list[list[Any]] = []

        def walk(node: Any, depth: int) -> None:
            if depth > 3:
                return
            if isinstance(node, list):
                if all(isinstance(r, dict) for r in node):
                    lists.append(node)
                return
            if isinstance(node, dict):
                for value in node.values():
                    walk(value, depth + 1)

        if not isinstance(observation, dict) or self.ident(observation):
            return "none"
        walk(observation, 0)
        touched = [v for value in (args or {}).values() for v in (value if isinstance(value, list) else [value]) if isinstance(v, str) and v in self.rows]
        with_ids = [rows for rows in lists if rows and any(self.ident(r, True) for r in rows)]
        # the list that carries the listing just changed, else any list of listing records; an empty list is the
        # emptied cart only when it is the reply's one list
        rows = next((r for r in with_ids if any(self.ident(x, True) in touched for x in r)), with_ids[0] if with_ids else None)
        if rows is None:
            if len(lists) == 1 and not lists[0]:
                rows = lists[0]
            else:
                return "none"
        idents = [self.ident(r, True) for r in rows]
        if not all(idents):
            return "unreadable"
        key = self.count_key
        if not key and rows:
            keys = [k for k in rows[0] if k != self.price_key
                    and all(isinstance(r.get(k), int) and not isinstance(r.get(k), bool) and 0 <= r[k] <= 99 for r in rows)]
            if len(keys) > 1 and name and args is not None:
                # several small numbers on each line: the one that matches Kyo's own arithmetic for the changed line
                saved = dict(self.cart_lines)
                self.note_cart(name, args)
                expected = dict(self.cart_lines)
                self.cart_lines = saved
                keys = [k for k in keys if all(r.get(k) == expected.get(i, 0) for r, i in zip(rows, idents) if i in touched)]
            if len(keys) != 1:
                return "unreadable"
            key = keys[0]
            self.count_key = key
        out: dict[str, int] = {}
        for row, ident in zip(rows, idents):
            count = row.get(key) if key else None
            if not isinstance(count, int) or isinstance(count, bool):
                return "unreadable"
            if count > 0:
                out[ident] = count
        self.cart_lines = out
        self.cart_view = observation
        self.cart_seen_at = self.step
        self.cart_shows_lines = True
        return "synced"

    def book_cart(self, name: str, args: dict[str, Any], observation: Any) -> None:
        """Record a cart call: from the reply's own line list when it can be read; a store that lists the cart after
        every change it made changed nothing when a reply has no list of listings at all (a refusal); otherwise
        (no list, or one that cannot be read) Kyo's own arithmetic."""
        self.cart_touched = True
        self.cart_at = self.step
        state = self.sync_cart(observation, name, args)
        if state == "synced":
            return
        if state == "none" and self.cart_shows_lines:
            self.trace.append({"kind": "cart_unchanged", "args": _clip(args, 60)})
            return
        self.note_cart(name, args)

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
        """How many listings one side-by-side call may carry: once a call has failed, what the store has shown to work
        (a stated limit can be wrong), else the schema's limit, else what has worked."""
        spec = ((self.schemas.get(self.roles.get("compare") or "") or {}).get("properties") or {}).get(self.compare_target() or "") or {}
        cap = spec["maxItems"] if isinstance(spec.get("maxItems"), int) else None
        if self.compare_bad:
            # step down from the failed size (halving at most), never below what has worked or below two
            room = max(2, self.compare_ok, min(self.compare_bad - 1, max(2, self.compare_bad // 2)))
            return min(room, cap) if cap else room
        if cap:
            return cap
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
        self.join_failed = False
        for index, call in enumerate(calls):
            allowed = {t["function"]["name"] for t in self.offered()}
            cid = call["id"]
            if call.get("joined"):
                if self.join_failed:
                    out.append((cid, "Not shown: the side-by-side call above showed nothing for these listings (this store shows fewer at once). Open this one on its own."))
                else:
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
                if name in {self.roles.get("open"), self.roles.get("offer")} - {None}:
                    # the agent wanted this listing read: the opens before an ending without a purchase start with it
                    text = raw if isinstance(raw, str) else json.dumps(raw)
                    self.wanted += [i for i in self.rows if i in text and i not in self.wanted]
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
            kept = self.keep_line(name, args)
            if kept:
                out.append((cid, kept))
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
            if self.is_refuse(name, args) and self.roles.get("refuse"):
                opened = self.decline_rows()
                if opened:
                    out.append((cid, "Not run yet (the ending used no step). " + opened + "\nCart now: " + self.cart_text()))
                    continue
            if ending and self.left() > 2 and self.time_left() > CLOSING - 25:
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
                            change = self.observe(self.roles["cart_view"], self.cart_view)
                            if change:
                                self.queued = []
                                out.append((cid, "Not run yet (no step used). Looking at the cart showed a change: " + change + "\nCart now: " + self.cart_text() + "\nApply the change, fix the cart if needed, then check out."))
                                continue
                    mended = self.cart_note()
                    if mended:
                        out.append((cid, mended + "\nCart now: " + self.cart_text()))
                        continue
                    swap = self.swap_hint()
                    # the checkout is held back at most twice for a sold-out line, so a wrong cart record cannot stall the episode
                    held = [i for i in self.cart_ids() if i in self.gone] if self.held < 2 else []
                    if swap or held:
                        self.held += 0 if swap else 1
                        out.append((cid, "Not run yet (no step used). " + (swap or f"{', '.join(held)} in the cart is sold out: replace that line first.") + "\nCart now: " + self.cart_text()))
                        continue
                    chain, hold = self.trail()
                    if hold:
                        out.append((cid, "Not run yet (no step used). " + chain + "\nCart now: " + self.cart_text()))
                        continue
                    # the check judges attributes: read the cart lines' own (and, for a cheapest pick, a few cheaper rows')
                    read = self.fill_facts(self.cart_ids(), self.tail() + 3) + self.cheaper_rows()
                    lacking = self.kind_hold() or self.need_hold()
                    if lacking:
                        seen = ("Kyo read these listings' attributes for the check (one step each):\n" + "\n".join(read) + "\n") if read else ""
                        out.append((cid, "Not run yet. " + chain + seen + lacking + "\nCart now: " + self.cart_text()))
                        continue
                    verdict = self.check_cart()
                    if verdict:
                        seen = ("Kyo read these listings' attributes for the check (one step each):\n" + "\n".join(read) + "\n") if read else ""
                        out.append((cid, "Not run yet. " + chain + seen + verdict + "\nCart now: " + self.cart_text()))
                        continue
                    if reread and self.cost > SPEND_SOFT:
                        out.append((cid, "Not run yet (no step used). Kyo re-read the cart's listings just now:\n" + reread + "\nCheck out again when every line still shows stock and a price inside the budget; otherwise replace that line first."))
                        continue
                else:
                    # steps decide whether an ending is held (the clock stops only model calls): a hold the clock or the spend
                    # cuts short is settled by Kyo's own finish, which keeps the agent's ending unless a checked cart is held
                    if self.left() > 5:
                        unreviewed = not self.audited
                        review = self.audit()
                        if unreviewed and self.audited and (not review or (review.startswith("A careful review") and not self.leads_whole())):
                            # the decline review of this ending found no lead for some requested item: no further code hold
                            self.review_short.add(self.needs_mark())
                        if not review and self.declines_refused and not self.audited:
                            # the store turned an ending down: the review runs again whatever the soft spend limit
                            review = self.review_again()
                        if review and (self.leads_whole() or not review.startswith("A careful review")):
                            out.append((cid, "Not run yet (no step used). " + review))
                            continue
                        if self.leads and self.leads_whole() and self.left() > 4 and self.refuse_blocks == 0:
                            self.refuse_blocks = 1
                            self.trace.append({"kind": "refuse_blocked", "leads": list(self.leads)})
                            out.append((cid, "Not run yet (no step used). The review found listings that appear to fit: " + ", ".join(self.leads) + ". Open them if needed, put the best fitting one in the cart (respecting the pick rule) and check out. End without a purchase only if each of them clearly fails a requirement; then say which one in your text and send the ending call again."))
                            continue
                        if self.leads and self.leads_whole() and self.left() > 4 and self.refuse_blocks == 1:
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
            if self.is_refuse(name, args):
                held = self.decline_hold()
                if held:
                    out.append((cid, held))
                    continue
            if self.is_refuse(name, args):
                done = self.complete_cart()
                if done:
                    out.append((cid, done + (" [episode finished]" if self.finished else "")))
                    continue
                args = self.quiet(name, args)
            if ending and self.left() >= 2 and self.time_left() > LAST_CALL + 12:
                go, said = self.gate("order" if self.is_buy(name, args) else "end")
                if not go:
                    out.append((cid, "Not run yet. " + said + "\nCart now: " + self.cart_text()))
                    continue
                if said:
                    trimmed = "\n[Kyo] " + said + trimmed
            reply = self.act(name, args)
            if reply is None:
                out.append((cid, "The call failed (network or server). You may try again."))
                continue
            observation = reply.get("observation")
            if many and isinstance(args.get(many), list) and len(args[many]) > 1:
                shown = _dump(observation)
                if not any(str(i) in shown for i in args[many]):
                    # a size limit only when every id is a listing the store has shown (an unknown id fails a call too)
                    if all(str(i) in self.rows for i in args[many]):
                        self.compare_bad = min(self.compare_bad or 999, len(args[many]))
                    self.join_failed = True
                    trimmed += f"\n[Kyo] The side-by-side view showed nothing for {len(args[many])} listings: send at most {self.compare_room()} per call."
                else:
                    self.compare_ok = max(self.compare_ok, len(args[many]))
                    self.join_failed = False
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
            if name not in plain and not ending:
                paged += self.deaf_words(name, args, observation)
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
                self.book_cart(name, args, observation)
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
                    self.rearm()
            elif self.is_refuse(name, args) and not reply.get("finished"):
                self.rearm()
                self.note_refusal()
            text = self.show(observation) + paged + trimmed
            if name not in plain and not ending:
                text += self.cheapest_here([observation] + self.paged_obs)
            base = text
            proof = self.evidence(observation)
            if proof:
                text += "\n" + proof
            if name == self.roles.get("cart") and any(_count(v) == 0 for v in args.values() if not isinstance(v, str)):
                if any(isinstance(v, str) and v in self.rows and _in_rows(observation, v) for v in args.values()):
                    text += "\n[Kyo] A count of 0 left that line in the cart: this tool adds the count to the line, so a negative count takes units off."
            self.note_sold = False
            lines = self.loose_lines(observation)
            if lines:
                meaning = self.take_note(lines)
                if meaning:
                    text += "\n[note read] " + meaning + ". Apply this change to your picks before checking out."
            if self.is_buy(name, args) and not reply.get("finished") and (self.note_sold or any(line in self.spoken(observation) for line in self.sold_lines)):
                seen = self.find_sold()
                if seen:
                    text += "\n[Kyo re-read the cart's lines after the refusal]\n" + seen
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

    # ---------- the ending: what shows that nothing fits, and Kyo's own cart ----------
    def needs_mark(self) -> str:
        """The requirements in force and the budget: a change of either starts the ending's holds afresh."""
        return _dump([self.needs_now(), self.budget])

    def bits(self, text: Any) -> frozenset[str]:
        """A text's words and figures in lower case, a plural s dropped."""
        return frozenset(w[:-1] if len(w) > 3 and w.endswith("s") else w for w in re.findall(r"[a-z0-9]+", str(text or "").lower()))

    def line_price(self, ident: str) -> float | None:
        """A listing's price under the price key the model named; before one is named, a row's only number."""
        price = self.price_of(ident)
        if price is not None or self.price_key:
            return price
        numbers = [v for v in (self.rows.get(ident) or {}).values() if isinstance(v, (int, float)) and not isinstance(v, bool)]
        return float(numbers[0]) if len(numbers) == 1 else None

    def label_fits(self, cat_key: str | None) -> dict[str, list[int]]:
        """Each category label seen, with how well it fits each requested item (kept until more listings are seen)."""
        if self.fit_memo[0] != len(self.rows) or self.fit_memo[1] != (cat_key or ""):
            labels = {str(r.get(cat_key) or "") for r in self.rows.values()} if cat_key else set()
            self.fit_memo = (len(self.rows), cat_key or "", {l: [self.label_fit(l, item) for item in self.items] for l in labels if l})
        return self.fit_memo[2]

    def item_of(self, ident: str, cat_key: str | None) -> int | None:
        """The requested item a listing serves: the one item its category label fits best, when no category seen fits that
        item better (a neighbouring category serves none); with one item and no category seen that fits it, any listing."""
        table = self.label_fits(cat_key)
        label = str((self.rows.get(ident) or {}).get(cat_key) or "") if cat_key else ""
        fits = table.get(label) or [0] * len(self.items)
        best = max(fits, default=0)
        tops = [max((v[n] for v in table.values()), default=0) for n in range(len(self.items))]
        if len(self.items) == 1:
            return 1 if not tops[0] or best == tops[0] else None
        if not best or fits.count(best) != 1:
            return None
        number = fits.index(best)
        return number + 1 if best == tops[number] else None

    def marks_of(self, ident: str, number: int, cat_key: str | None) -> dict[str, str]:
        """How an opened listing stands on each live requirement of item `number`, by its own record: "met" when a key
        naming the property (or the category label) carries every word of the value; "fails" when such a key carries
        other words while another opened listing of its category carries the wanted ones under such a key; "open" for
        the rest (figures, values read from another listing, anything the model must weigh)."""
        record = self.full.get(ident) or {}
        block = _fact_block(record) or {}
        label = str(record.get(cat_key) or "") if cat_key else ""
        peers = [_fact_block(r) or {} for i, r in self.full.items() if i != ident and (not cat_key or str(r.get(cat_key) or "") == label)]
        pointed = {need for _, need in self.refs}
        out: dict[str, str] = {}
        for n, need in self.needs_now():
            if n != number or need in pointed or ":" not in need:
                continue
            prop, value = need.split(":", 1)
            want = self.bits(value)
            if not want or any(w[0].isdigit() for w in want):
                out[need] = "open"
                continue
            stem = _stem(prop)
            keys = [k for k in block if _names(stem, _stem(k))]
            if (label and want <= self.bits(label)) or any(want <= self.bits(block[k]) for k in keys):
                out[need] = "met"
            elif keys and any(want <= self.bits(v) for p in peers for k, v in p.items() if _names(stem, _stem(k))):
                out[need] = "fails"
            else:
                out[need] = "open"
        return out

    def all_met(self, ident: str, number: int, cat_key: str | None) -> bool:
        return ident in self.full and all(v == "met" for v in self.marks_of(ident, number, cat_key).values())

    def affordable(self, price: float | None, qty: int, spent: float) -> bool:
        """A line at this price keeps the cart inside the budget (before the store has turned an ending down, a budget no
        row seen fits is read as one in another unit and decides nothing; after that it is read as stated)."""
        budget = self.budget
        if price is None or isinstance(budget, bool) or not isinstance(budget, (int, float)):
            return True
        if not self.declines_refused and not any((self.line_price(i) if self.line_price(i) is not None else float("inf")) <= budget for i in self.seen_order):
            return True
        if self.brief.get("budget_scope") == "each":
            return price <= budget
        return spent + price * qty <= budget

    def counts_units(self) -> bool:
        """The cart tool takes a count (on its own or inside each line of a whole-cart list)."""
        props = (self.schemas.get(self.roles.get("cart") or "") or {}).get("properties") or {}
        for spec in props.values():
            if not isinstance(spec, dict):
                continue
            if "integer" in _kinds(spec):
                return True
            inner = spec.get("items") if "array" in _kinds(spec) else None
            if isinstance(inner, dict) and any(isinstance(s, dict) and "integer" in _kinds(s) for s in (inner.get("properties") or {}).values()):
                return True
        return False

    def candidates(self, number: int, cat_key: str | None, taken: set[str], strict: bool, partial: bool = False) -> list[str]:
        """Listings that could fill item `number`'s cart line, best first: the decline review's leads, opened listings whose
        every requirement the code finds met, the agent's own earlier cart lines; after the store turned an ending down,
        also other opened listings, then rows never opened (newest first in each group; cheapest first under a cheapest
        pick). Never one the code finds failing a requirement, one sold out or one already taken; one leaving a requirement
        unproven only after the others; with `strict`, the review's leads only. With `partial` (the cart holds the other
        items' lines) an opened listing of the item's own kind may fill the line too: a set missing one item is never right."""
        leads = list(self.lead_items.get(str(number), []))
        if len(self.items) == 1:
            leads += [i for i in self.leads if i not in leads]
        pool = list(dict.fromkeys(leads if strict else leads + list(reversed(self.seen_order))))
        cheap = self.brief.get("pick") == "cheapest"
        ranked = []
        for order, ident in enumerate(pool):
            if ident in taken or ident in self.check_rejected or ident not in self.rows or not self.buyable(ident):
                continue
            mine = self.item_of(ident, cat_key)
            if mine != number and not (ident in leads and mine is None):
                continue
            marks = self.marks_of(ident, number, cat_key) if ident in self.full else {}
            states = [state for _, state, _ in self.item_marks(ident, number)] if ident in self.full else []
            if "fails" in marks.values() or "fails" in states:
                continue
            # an item without any requirement in force: no listing of its kind can fail one
            needless = not any(n == number for n, _ in self.needs_now())
            tier = (0 if ident in leads else 1 if (states and all(s == "met" for s in states)) or (needless and ident in self.full) else
                    2 if ident in self.built and ident not in self.dropped else 3 if ident in self.full else 4)
            if "unproven" in states:
                tier = max(tier, 3)
            if ident in self.check_better and ident not in self.check_rejected:
                # a listing the cart check named as the better fit comes first
                tier = -1
            if tier > 2 and not self.declines_refused and not needless and not (partial and tier == 3):
                # before the store turns an ending down only a qualifying listing fills a line
                continue
            price = self.line_price(ident)
            ranked.append((tier, (price if price is not None else float("inf")) if cheap else 0.0, order, ident))
        return [ident for *_, ident in sorted(ranked)]

    def cart_faults(self, fill: bool = False) -> tuple[list[str], dict[str, int], list[str]]:
        """Kyo's own cart audit (code only: no step, no model call): one line per requested item at its quantity, and
        nothing else. Returns the problems, the lines the cart should hold (listing: units) and the lines to take off.
        With `fill`, an item without a line gets its best qualifying candidate inside the budget; an empty cart takes only
        the decline review's leads (any candidate once the store has turned an ending without a purchase down). It says
        nothing when Kyo's record of the cart lacks a line a fresh cart view shows, when a line cannot be told to serve one
        item, or when two lines seem to serve one item while the cart holds no more lines than items (the lines may serve
        the items in a way the code cannot read)."""
        if not self.roles.get("cart"):
            return [], {}, []
        cat_key = self.category_key()
        now = [i for i in self.cart_ids() if i in self.rows]
        if any(i not in self.cart_lines for i in now):
            return [], {}, []
        per: dict[int, list[str]] = {}
        for ident in now:
            number = self.item_of(ident, cat_key)
            if number is None:
                return [], {}, []
            per.setdefault(number, []).append(ident)
        live = [i for i in now if not self.sold_out(i)]
        if any(len([i for i in ids if i in live]) > 1 for ids in per.values()) and len(now) <= len(self.items):
            return [], {}, []
        counted = self.counts_units()
        problems: list[str] = []
        lines: dict[str, int] = {}
        drop: list[str] = []
        for number, item in enumerate(self.items, 1):
            mine = per.get(number, [])
            gone = [i for i in mine if self.sold_out(i)]
            for ident in gone:
                drop.append(ident)
                problems.append(f"{ident} is sold out")
            have = [i for i in mine if i not in gone]
            if len(have) > 1:
                have.sort(key=lambda i: ("fails" in self.marks_of(i, number, cat_key).values(), not self.all_met(i, number, cat_key)))
                drop += have[1:]
                problems.append(f"item {number} ({item.name}) has {len(have)} lines ({', '.join(have)}); one is right")
                have = have[:1]
            if have:
                lines[have[0]] = item.qty
                if counted and self.cart_lines.get(have[0]) != item.qty:
                    problems.append(f"{have[0]} holds {self.cart_lines.get(have[0])} units where item {number} ({item.name}) wants {item.qty}")
            else:
                problems.append(f"item {number} ({item.name}) has no line")
        if fill:
            strict = not now and not self.declines_refused
            taken = set(lines) | set(drop)
            for number, item in enumerate(self.items, 1):
                if any(self.item_of(i, cat_key) == number for i in lines):
                    continue
                spent = sum((self.line_price(i) or 0.0) * n for i, n in lines.items())
                for ident in self.candidates(number, cat_key, taken, strict, partial=bool(now)):
                    if self.affordable(self.line_price(ident), item.qty, spent):
                        lines[ident] = item.qty
                        taken.add(ident)
                        break
        prices = [self.line_price(i) for i in lines]
        if lines and None not in prices and not self.affordable(0.0, 1, sum(p * n for p, n in zip(prices, lines.values()))):
            problems.append(f"these lines come to {sum(p * n for p, n in zip(prices, lines.values())):g}, over the budget of {self.budget:g}")
        return problems, lines, drop

    def send_cart(self, calls: list[dict[str, Any]]) -> list[str]:
        """Kyo's own cart calls, one step each, keeping one step for the checkout; what was sent."""
        cart = self.roles.get("cart") or ""
        done: list[str] = []
        for args in calls:
            if self.finished or self.left() <= 1:
                break
            faults: list[str] = []
            args = _shape(args, self.schemas.get(cart), "args", faults)
            if faults:
                break
            reply = self.act(cart, args)
            if reply is None or self.finished:
                break
            self.book_cart(cart, args, reply.get("observation"))
            self.observe(cart, reply.get("observation"))
            done.append(_dump(args))
        return done

    def set_lines(self, lines: dict[str, int], drop: list[str]) -> list[str] | None:
        """Make the cart hold `lines` and none of `drop` with the cart tool, within the steps left (one kept for the
        checkout). A line of an adding cart that needs another count is taken off first and added again. The calls sent,
        or None when the cart tool cannot say it in the steps left or a call did not go through."""
        now = self.cart_ids()
        calls = self.cart_calls(lines, drop)
        redo: list[str] = []
        later = 0
        if calls is None:
            redo = [i for i, n in lines.items() if i in now and self.cart_lines.get(i, 0) != n]
            calls = self.cart_calls({}, list(dict.fromkeys(drop + redo)))
            later = len(redo) + len([i for i in lines if i not in now])
            if calls is None or not redo:
                return None
        if len(calls) + later + 1 > self.left():
            return None
        done = self.send_cart(calls)
        if len(done) < len(calls):
            return None
        if redo:
            more = self.cart_calls(lines, [])
            if more is None:
                return None
            sent = self.send_cart(more)
            if len(sent) < len(more):
                return None
            done += sent
        return done

    def mend_cart(self) -> str:
        """Kyo's own cart audit before its own finish (code only but for the cart check; store calls only within the
        steps left, one kept for the checkout): one line per requested item at its quantity and nothing else. Returns ""
        to go on with the checkout of what the cart holds, or why the finish ends without a purchase instead:
        - the agent's last cart, checkout or ending call was an ending without a purchase, and the cart (as mended) has not
          passed the cart check (one runs now while a model call can);
        - the cart lacks a line for a requested item and no qualifying listing fills it (a partial order is never right).
        A fill (an empty cart: the decline review's leads only) is carted once the check passes it, or on the code's own
        reading when no model call can run any more; after the store turned an ending down, from each item's best
        candidate, unless the check turns that cart down (and what is held is checked out, as before). Store calls
        need the steps (set_lines keeps one for the checkout); the decision itself needs none."""
        if self.finished or not self.roles.get("buy") or not self.roles.get("cart"):
            return ""
        refused = self.declines_refused > 0
        if self.no_credit and not refused:
            dead = self.dead_cart()
            if dead is not None:
                return dead
        # the agent's own ending without a purchase is kept only when the cart is not a whole, unbroken set
        wished = not refused and self.wished() and self.whole_cart() is None
        problems, lines, drop = self.cart_faults(True)
        now = [i for i in self.cart_ids() if i in self.rows and not self.sold_out(i)]
        if not problems and not lines:
            # the audit cannot read this cart line by line: the plain finish, unless the agent asked to end without a purchase
            if wished and now and self.passes(now) is not True:
                return "the agent's last call ended without a purchase and its cart has not passed the cart check"
            return ""
        if len(lines) < len(self.items):
            # (after a refused ending the store's next answer decides: what is held is checked out, as before)
            self.trace.append({"kind": "cart_audit", "problems": problems[:6], "lines": lines, "short": True})
            return "the cart lacks a line for a requested item and no listing read qualifies for it" if now and not refused else ""
        added = [i for i in lines if i not in now]
        change = bool(drop) or bool(added) or any(self.cart_lines.get(i) != n for i, n in lines.items())
        if refused:
            if (added and not now and self.ending_open(lines)) or (added and self.passes(list(lines)) is False):
                # after a refused ending an empty cart is filled only when the code can rule out a request no set meets
                self.trace.append({"kind": "cart_audit", "problems": problems[:6], "lines": lines, "passed": False, "open": True})
                return ""
        elif added or wished:
            passed = self.passes(list(lines))
            tries = 0
            while passed is False and added and not wished and tries < 2:
                # the check turned the fill down: the next candidate for that line (a rejected one is never offered again)
                self.check_rejected.update(added)
                tries += 1
                problems, lines, drop = self.cart_faults(True)
                if len(lines) < len(self.items):
                    break
                added = [i for i in lines if i not in now]
                change = bool(drop) or bool(added) or any(self.cart_lines.get(i) != n for i, n in lines.items())
                passed = self.passes(list(lines)) if added else True
            if len(lines) < len(self.items) or passed is False or (passed is None and wished):
                self.trace.append({"kind": "cart_audit", "problems": problems[:6], "lines": lines, "passed": passed, "wished": wished, "tries": tries})
                if wished:
                    return "the agent's last call ended without a purchase and the cart check has not passed the cart"
                if now or any(self.sold_out(i) for i in self.cart_ids()):
                    # never check out a line the store says is gone: an order holding it is wrong whatever else it holds
                    return "the cart check turned down the listing that would complete the cart"
                return ""
        if not change:
            return ""
        done = self.set_lines(lines, drop)
        self.trace.append({"kind": "cart_audit", "problems": problems[:6], "lines": lines, "drop": drop, "sent": None if done is None else len(done)})
        if done is None and now and (added or wished):
            return "the cart could not be completed in the steps left"
        return ""

    def cart_note(self) -> str:
        """Kyo's own cart audit at a checkout (code only, for the checkout branch; no spend or time limit). The first time a
        cart shows a problem the checkout is held with a note (no step); when the agent checks out that same cart again,
        Kyo sets the lines itself (one step a cart call, within the steps left) and holds once more so the mended cart is
        checked out. "" when the cart is right or nothing can be done."""
        if not self.cart_ids():
            return ""
        problems, lines, drop = self.cart_faults(True)
        if not problems:
            return ""
        mark = self.cart_mark() + "|" + self.needs_mark()
        whole = len(lines) >= len(self.items)
        plan = ", ".join(f"{i} x {n}" for i, n in lines.items()) if whole else ""
        if mark not in self.cart_noted:
            self.cart_noted.add(mark)
            self.trace.append({"kind": "cart_note", "problems": problems[:6], "lines": lines, "drop": drop})
            return ("Not run yet (no step used). Kyo's own cart audit: " + "; ".join(problems[:5]) + "."
                    + (f" A cart that puts it right: {plan}" + (f", without {', '.join(drop)}" if drop else "") + "." if plan else "")
                    + ("" if whole else " No listing read so far is known to meet every requirement of the item without a line: find one (search with its"
                       " wanted values and open the rows), or end without a purchase if the store has none, since an order missing an item earns nothing.")
                    + " Make the cart one line per requested item at its quantity and nothing else, then check out.")
        if mark + "|set" in self.cart_noted:
            return ""
        self.cart_noted.add(mark + "|set")
        done = self.set_lines(lines, drop)
        self.trace.append({"kind": "cart_audit", "problems": problems[:6], "lines": lines, "drop": drop, "sent": None if done is None else len(done), "at": "checkout"})
        if not done:
            return ""
        return ("Not run yet. Kyo's own cart audit: " + "; ".join(problems[:4]) + ". Kyo set the cart itself (" + "; ".join(done)
                + "). Check out now unless a line breaks a requirement.")

    def code_order(self, lines: dict[str, int], drop: list[str], why: str) -> str:
        """Kyo sets the cart to `lines` and checks out itself (code only but for the cart check); what happened, or ""
        when it could not start (never for lines that break the budget as stated or that the cart check turns down)."""
        buy = self.roles.get("buy")
        if not buy or not lines or self.finished or self.breaks_budget(lines) or self.passes(list(lines)) is False:
            return ""
        faults: list[str] = []
        args = _shape(self.order_args(buy), self.schemas.get(buy), "args", faults)
        if faults:
            return ""
        change = bool(drop) or any(i not in self.cart_ids() or self.cart_lines.get(i) != n for i, n in lines.items())
        before = self.cart_mark()
        done = self.set_lines(lines, drop) if change else []
        if done is None:
            # nothing sent (too few steps, or the cart tool cannot say it): the agent's own call goes on as it was
            return ("Not run yet. Kyo tried to set the cart to " + ", ".join(f"{i} x {n}" for i, n in lines.items())
                    + " and could not finish that. Cart now: " + self.cart_text() + "\nSet the cart to one fitting line per item, then check out.") if self.cart_mark() != before else ""
        if self.finished or self.left() < 1:
            return ""
        said = ""
        if self.left() >= 2 and self.time_left() > LAST_CALL + 12:
            go, said = self.gate("order")
            if not go:
                return "Not run yet. " + said + "\nCart now: " + self.cart_text()
        reply = self.act(buy, args)
        self.trace.append({"kind": "code_order", "lines": lines, "drop": drop, "reply": _clip(reply, 200)})
        if reply is None:
            return ""
        self.buys += 1
        if not reply.get("finished"):
            self.refused_at = self.step
            self.rearm()
        self.observe(buy, reply.get("observation"))
        return ("Not run (Kyo acted instead): " + why + " Kyo set the cart (" + ("; ".join(done) or "as it was") + ") and checked out. "
                + (said + " " if said else "") + "Store reply: " + self.show(reply.get("observation")) + (" [episode finished]" if self.finished else ""))

    def note_refusal(self) -> None:
        """Called on the store's reply to an ending without a purchase that reached it and left the episode open: the store
        turned that ending down."""
        self.declines_refused += 1
        self.trace.append({"kind": "decline_refused", "step": self.step})

    def breaks_budget(self, lines: dict[str, int]) -> bool:
        """These lines break the budget as stated (every price known; a total, or each line under a per-item budget)."""
        budget = self.budget
        prices = [self.line_price(i) for i in lines]
        if not lines or None in prices or isinstance(budget, bool) or not isinstance(budget, (int, float)) or budget <= 0:
            return False
        if self.brief.get("budget_scope") == "each":
            return any(p > budget for p in prices)
        return sum(p * n for p, n in zip(prices, lines.values())) > budget

    def ending_open(self, lines: dict[str, int]) -> str:
        """After a refused ending, what the code cannot rule out and so leaves a re-decline to the model: two requirements
        of one item on one property that no opened listing meets together (figures and ranges the code does not weigh),
        or a planned cart that breaks the budget as stated. "" when neither."""
        cat_key = self.category_key()
        pointed = {need for _, need in self.refs}
        for number in range(1, len(self.items) + 1):
            needs = [need for n, need in self.needs_now() if n == number and need not in pointed and ":" in need]
            for at, first in enumerate(needs):
                for second in needs[at + 1:]:
                    if not _names(_stem(first.split(":", 1)[0]), _stem(second.split(":", 1)[0])):
                        continue
                    one, two = self.bits(first.split(":", 1)[1]), self.bits(second.split(":", 1)[1])
                    if not one or not two or one <= two or two <= one:
                        continue
                    if not any(self.item_of(i, cat_key) == number and all(self.marks_of(i, number, cat_key).get(x) == "met" for x in (first, second))
                               for i in self.full):
                        return f"item {number} asks {first} and {second}, which may not both hold"
        if self.breaks_budget(lines):
            return "the best cart seen breaks the budget"
        return ""

    def review_again(self) -> str:
        """After the store turned an ending without a purchase down (once per needs version): the decline review runs again
        on the listings seen, under the requirements now in force, whatever the soft spend limit and the step count (the
        model call itself still stops at the last-call time and the spend stop). When no model call can run, the code's
        reading names the opened listings whose every requirement it finds met."""
        mark = self.needs_mark()
        if mark in self.rereviewed:
            return ""
        self.rereviewed.add(mark)
        self.audited = True
        got = self.audit_call() if self.time_left() > LAST_CALL and self.cost < SPEND_STOP else None
        named = self.audit_fits(got)
        if not named and got is not None:
            rows = [str(i) for i in _items(got.get("open")) if str(i) in self.rows and str(i) not in self.opened and not self.sold_out(str(i))]
            if rows and self.open_rows(rows):
                got = self.audit_call()
                named = self.audit_fits(got)
        if not named and got is None:
            cat_key = self.category_key()
            for number in range(1, len(self.items) + 1):
                met = [i for i in reversed(self.seen_order) if self.buyable(i) and self.item_of(i, cat_key) == number and self.all_met(i, number, cat_key)]
                if met:
                    named[str(number)] = met[:4]
        self.trace.append({"kind": "review_again", "named": named, "model": got is not None})
        if not named:
            return ""
        self.lead_items = named
        self.leads = [i for ids in named.values() for i in ids]
        return ("The store turned ending without a purchase down, and a careful review of the listings seen thinks these may fit: "
                + "; ".join(f"item {k}: {', '.join(v)}" for k, v in named.items())
                + ". Open them if needed, put the best fitting one for each item in the cart at the requested quantity and check out.")

    def searches_seen(self) -> list[tuple[dict[str, Any], list[str]]]:
        """Each search call of the episode (Kyo's record of what it read): its arguments and the listings it showed."""
        name = self.roles.get("search")
        found: list[tuple[dict[str, Any], list[str]]] = []
        for mark, entry in self.read_cache.items():
            if not name or not mark.startswith(name + "{"):
                continue
            args = _json_obj(mark[len(name):])
            if args is not None:
                quoted = re.findall(r'"([^"\\]{1,64})"', entry[2])
                found.append((args, [i for i in dict.fromkeys(quoted) if i in self.rows]))
        return found

    def ascending(self, ids: list[str]) -> bool:
        """A result whose rows come cheapest first (three priced rows at least)."""
        prices = [self.line_price(i) for i in ids]
        return len(prices) >= 3 and None not in prices and all(a <= b for a, b in zip(prices, prices[1:]))

    def plain_args(self, args: dict[str, Any]) -> tuple[list[str], list[str]]:
        """A search's text arguments split into category labels or fixed choices, and free words."""
        cat_key = self.category_key()
        known = {_words(str(r.get(cat_key) or "")) for r in self.rows.values()} if cat_key else set()
        known |= {_words(x) for x in self.labels()}
        known.discard("")
        texts = [v for v in args.values() if isinstance(v, str) and v.strip()]
        return [v for v in texts if _words(v) in known], [v for v in texts if _words(v) not in known]

    def decline_ground(self, only_contradiction: bool = False) -> str:
        """Code only: what the episode saw that shows no order can be right, or "": a contradiction (two values one item's
        requirements give one property, each stated on that item's opened listings, never both on one); a conflict (a
        search for two of an item's values showed listings and none stating both, and the item's opened listings state
        each value, never both on one); a budget that even the cheapest buyable listing of each item seen breaks, each item's category browsed in a list
        the store gives cheapest first; a shelf
        whose browse, like every other read, showed listings of the item's kind and none in stock. A value read from
        another listing and a requirement no opened listing states are never a proof (they are pages still to open)."""
        cat_key = self.category_key()
        pointed = {need for _, need in self.refs}
        searches = self.searches_seen()
        for number in range(1, len(self.items) + 1):
            needs = [need for n, need in self.needs_now() if n == number and need not in pointed and ":" in need]
            opened = [i for i in self.full if self.item_of(i, cat_key) == number]
            marks = {i: self.marks_of(i, number, cat_key) for i in opened}
            for at, first in enumerate(needs):
                for second in needs[at + 1:]:
                    one, two = self.bits(first.split(":", 1)[1]), self.bits(second.split(":", 1)[1])
                    if not one or not two or one <= two or two <= one:
                        continue
                    says_one = {i for i in opened if marks[i].get(first) == "met"}
                    says_two = {i for i in opened if marks[i].get(second) == "met"}
                    if not says_one or not says_two or says_one & says_two:
                        continue
                    if _names(_stem(first.split(":", 1)[0]), _stem(second.split(":", 1)[0])):
                        return f"contradiction: item {number} asks {first} and {second}; opened listings state each, never both"
                    if only_contradiction:
                        continue
                    shown = [i for args, ids in searches if one | two <= self.bits(" ".join(self.plain_args(args)[1])) for i in ids]
                    if shown and not any(one | two <= self.bits(_dump(self.full.get(i) or self.rows.get(i))) for i in shown):
                        return f"conflict: item {number}: a search for both {first} and {second} showed no listing with both, and its opened listings state one or the other, never both"
        if only_contradiction:
            return ""
        budget = self.budget
        if isinstance(budget, (int, float)) and not isinstance(budget, bool) and budget > 0:
            total = 0.0
            for number, item in enumerate(self.items, 1):
                # the cheapest of the kind is known only from a browse of its category that the store lists cheapest first
                cheapest_first = any(not self.plain_args(args)[1] and any(self.label_fit(v, item) >= 2 for v in self.plain_args(args)[0])
                                     and self.ascending([i for i in ids if self.item_of(i, cat_key) == number]) for args, ids in searches)
                prices = [p for p in (self.line_price(i) for i in self.rows if self.item_of(i, cat_key) == number and self.buyable(i)) if p is not None]
                if not prices or not cheapest_first:
                    total = -1.0
                    break
                total = max(total, min(prices)) if self.brief.get("budget_scope") == "each" else total + min(prices) * item.qty
            # the browse listed at least three priced rows of each kind cheapest first, so the budget is read as stated
            if total > budget:
                return f"budget: the cheapest buyable listings seen come to {total:g}, over the budget of {budget:g}"
        for number, item in enumerate(self.items, 1):
            kind = [i for i in self.rows if self.item_of(i, cat_key) == number]
            browsed = [ids for args, ids in searches if not self.plain_args(args)[1]
                       and any(self.label_fit(v, item) >= 2 for v in self.plain_args(args)[0]) and any(i in kind for i in ids)]
            if browsed and len(kind) >= 2 and self.stock_key and not any(self.buyable(i) for i in kind):
                return f"stock: item {number} ({item.name}): its category shows listings of the kind and none in stock"
        return ""

    def ending_gaps(self) -> list[str]:
        """What is still unread before an ending without a purchase can be shown right: requirements whose value lives on a
        listing not opened yet, requirements no opened listing of the item has an attribute for, and items whose category
        no browse has shown."""
        cat_key = self.category_key()
        live = {need for _, need in self.needs_now()}
        gaps = [f"the pages {need!r} points at" for _, need in self.refs if need in live and need not in self.ref_values][:2]
        pointed = {need for _, need in self.refs}
        searches = self.searches_seen()
        for number, item in enumerate(self.items, 1):
            tag = f"item {number} ({item.name})" if len(self.items) > 1 else item.name
            opened = [_fact_block(self.full[i]) or {} for i in self.full if self.item_of(i, cat_key) == number]
            for n, need in self.needs_now():
                prop = need.split(":", 1)[0].strip()
                if n == number and need not in pointed and ":" in need and not any(_names(_stem(prop), _stem(k)) for block in opened for k in block):
                    gaps.append(f"{tag}: no opened listing has an attribute for {prop} yet")
            if not any(not self.plain_args(args)[1] and any(self.label_fit(v, item) >= 2 for v in self.plain_args(args)[0]) for args, _ in searches):
                gaps.append(f"{tag}: its category has not been browsed")
        return gaps[:5]

    @_safe(None)
    def whole_cart(self) -> tuple[dict[str, int], list[str]] | None:
        """The cart as Kyo's own audit reads it, when it holds one line per requested item at its quantity, inside the
        budget, none sold out and none whose record states another value for a requirement in force or leaves one unproven: (lines, lines to
        drop). None otherwise, or when the audit cannot read the cart line by line. Such a cart is checked out rather than
        thrown away: an ending without a purchase is right only when the store has no fitting set at all."""
        if not self.roles.get("cart") or not self.roles.get("buy") or not self.items:
            return None
        problems, lines, drop = self.cart_faults(False)
        if not lines or len(lines) < len(self.items) or any("over the budget" in p or "has no line" in p for p in problems):
            return None
        cat_key = self.category_key()
        for ident in lines:
            if self.sold_out(ident) or ident not in self.rows:
                return None
            number = self.item_of(ident, cat_key)
            if number is None or any(state in ("fails", "unproven") for _, state, _ in self.item_marks(ident, number)):
                return None
        if self.breaks_budget(lines):
            return None
        return lines, drop

    @_safe(None)
    def dead_cart(self) -> str | None:
        """No model call can run any more (the run's key is spent): the code's own reading finishes. A whole cart (no line
        failing or unproven, inside the budget) is checked out; else the code-met set (met_cart) is carted and checked
        out. Returns "" to check out, None to leave the finish to the usual audit (which then ends without a purchase)."""
        if self.whole_cart() is not None:
            self.trace.append({"kind": "dead_cart", "whole": True})
            return ""
        lines = self.met_cart()
        if not lines:
            return None
        drop = [i for i in self.cart_ids() if i not in lines]
        done = self.set_lines(lines, drop)
        self.trace.append({"kind": "dead_cart", "lines": lines, "drop": drop, "sent": None if done is None else len(done)})
        return "" if done is not None else None

    def met_cart(self) -> dict[str, int] | None:
        """Code only: for every requested item an opened, buyable listing of its own kind whose every live requirement
        the code reads as met by the listing's own attributes (the cheapest such under a cheapest pick), the set inside
        the budget: {listing: units}. None when some item has no such listing or a requirement the code cannot read (a
        value on another listing, a pair, an amount or an exclusion), or the pick is the most or least of a measure."""
        if self.brief.get("pick") in ("most", "least") or not self.items or not self.roles.get("buy") or not self.roles.get("cart"):
            return None
        cat_key = self.category_key()
        pointed = {need for _, need in self.refs} | set(self.ref_values)
        lines: dict[str, int] = {}
        for number, item in enumerate(self.items, 1):
            live = [need for n, need in self.needs_now() if n == number]
            if any(need in pointed for need in live):
                return None
            found = []
            for ident in self.full:
                if ident in lines or not self.buyable(ident) or self.sold_out(ident) or self.item_of(ident, cat_key) != number:
                    continue
                marks = self.item_marks(ident, number)
                if len(marks) == len(live) and all(state == "met" for _, state, _ in marks):
                    price = self.line_price(ident)
                    found.append((price if price is not None else float("inf"), ident))
            if not found:
                return None
            lines[min(found)[1]] = item.qty
        return None if self.breaks_budget(lines) else lines

    def decline_hold(self) -> str:
        """Before an ending without a purchase goes out (code; a hold uses no step). An ending with a proof from what the
        episode saw (decline_ground) goes through. Without one it is held once at most per needs version: the search-more push when
        no lead is known and more than a third of the steps are left; else only when no other hold already settled this
        version's ending (Kyo's decline-rows note, the lead holds, a review that found nothing for some item) and the
        spend is under the soft limit (the agent could hardly act on it); never with too few steps left to cart and check
        out. A hold the spend or the clock cuts short is settled by finish(), which keeps the agent's ending. After the store has turned such an ending down, only a contradiction lets it through at
        once: the review runs again, one hold names the best cart, and then Kyo checks out that cart itself when the cart
        check does not turn it down, unless the code cannot rule out requirements that may not all hold or a cart over the
        budget (ending_open), when the agent's next ending goes out."""
        mark = self.needs_mark()
        holds = self.proof_holds.get(mark, 0) + self.refuse_blocks
        refused = self.declines_refused > 0
        ground = self.decline_ground(refused)
        room = self.left() >= self.tail() + len(self.items)
        quiet = ("lead holds" if self.refuse_blocks else "rows note" if self.rows_held()
                 else "review found nothing for an item" if mark in self.review_short else "")
        self.trace.append({"kind": "decline_proof", "ground": ground[:160], "holds": holds, "refused": refused, "room": room, "quiet": quiet, "cost": round(self.cost, 4)})
        if ground:
            return ""
        whole_now = None if refused else self.whole_cart()
        if whole_now:
            lines, drop = whole_now
            plan = ", ".join(f"{i} x {n}" for i, n in lines.items())
            if self.cart_holds.get(mark, 0) < 1 and room:
                self.cart_holds[mark] = 1
                self.trace.append({"kind": "whole_cart_hold", "lines": lines})
                return ("Not run yet (no step used). The cart already holds a line for every requested item (" + plan + "), inside the budget, and no record"
                        " states a different value for a requirement. Ending without a purchase earns nothing when a fitting set exists, and a value in other words"
                        " that means the same thing is a fit. Check this cart out (swap a line first only for a listing you have read that fits better)."
                        " End without a purchase only if a line's record clearly breaks a requirement and no listing of its kind meets it.")
            done = self.code_order(lines, drop, "the cart holds a line for every requested item, inside the budget, and no record breaks a requirement, so Kyo checked it out instead of ending without a purchase.")
            self.trace.append({"kind": "whole_cart_order", "lines": lines, "done": bool(done)})
            if done:
                return done
        met_now = None if refused or whole_now else self.met_cart()
        if met_now:
            plan = ", ".join(f"{i} x {n}" for i, n in met_now.items())
            if self.cart_holds.get(mark + "#met", 0) < 1 and room:
                self.cart_holds[mark + "#met"] = 1
                self.trace.append({"kind": "met_cart_hold", "lines": met_now})
                return ("Not run yet (no step used). Listings you have opened state every requirement in force for each requested item, inside the budget: "
                        + plan + ". Put them in the cart at the requested quantities and check out (a better fit you have read may replace one)."
                        " End without a purchase only if one of them clearly breaks a requirement; then say which one and send the ending call again.")
            drop = [i for i in self.cart_ids() if i not in met_now]
            done = self.code_order(met_now, drop, "listings already opened state every requirement for each requested item inside the budget, so Kyo carted them and checked out instead of ending without a purchase.")
            self.trace.append({"kind": "met_cart_order", "lines": met_now, "done": bool(done)})
            if done:
                return done
        if refused:
            self.review_again()
            _, lines, drop = self.cart_faults(True)
            whole = len(lines) >= len(self.items)
            unsure = self.ending_open(lines)
            if unsure:
                self.trace.append({"kind": "decline_open", "why": unsure[:160]})
            if whole and not unsure and (self.proof_holds.get(mark, 0) >= 1 or not room or self.time_left() <= CLOSING):
                done = self.code_order(lines, drop, "the store has turned ending without a purchase down, and only requirements that contradict each other would show that nothing fits.")
                if done:
                    return done
            if self.proof_holds.get(mark, 0) < 1 and room:
                self.proof_holds[mark] = self.proof_holds.get(mark, 0) + 1
                plan = ", ".join(f"{i} x {n}" for i, n in lines.items())
                return ("Not run yet (no step used). The store turned ending without a purchase down, which usually means a fitting set exists."
                        + (f" The best cart Kyo can see: {plan}." if whole and not unsure else "")
                        + " Put the best listing for each item in the cart at the requested quantity and check out; end without a purchase only if two requirements"
                        " contradict each other or no fitting set stays inside the budget.")
            return ""
        # the search-more push (an early ending, no lead, plenty of steps) stays; otherwise the hold needs nothing settled and spend left
        push_due = not self.leads and not self.pushed and self.left() > self.max_steps // 3
        if self.proof_holds.get(mark, 0) >= 1 or not room or (not push_due and (quiet or self.cost >= SPEND_SOFT)):
            return ""
        self.proof_holds[mark] = 1
        gaps = self.ending_gaps()
        still = (" Still unread: " + "; ".join(gaps) + ".") if gaps else ""
        push = ""
        if push_due:
            self.pushed = True
            push = (f" {self.left()} steps are still left: browse the category of each requested item and page through it with the token or offset the store returns,"
                    " try single-word value queries, and open the listings that might fit.")
        return ("Not run yet (no step used). Nothing read so far shows that the store has no fitting listing, and ending without a purchase earns nothing when it has one."
                " What would show it: the item's category, browsed (a word search that finds nothing proves nothing), with no listing of the kind in stock;"
                " a search for two of an item's values together finding no listing with both while each opened listing states one of them;"
                " the cheapest listings breaking the budget; or two requirements that contradict each other." + still + push
                + " When each requested item has a listing you have read that fails none of its requirements, check out those (the best for each item, at the requested quantity, inside the budget)."
                " End without a purchase only if, for some requested item, every listing you have read clearly fails one of its requirements; then send the ending call again.")

    def wished(self) -> bool:
        """The agent's last call to the cart, checkout or ending tools was an ending without a purchase (Kyo's holds may have
        kept it from the store): its own word on how the episode ends."""
        tools = {self.roles.get(job) for job in ("cart", "buy", "refuse")} - {None}
        for entry in reversed(self.trace):
            if entry.get("kind") != "turn":
                continue
            names = [str(c.get("name")) for c in entry.get("calls") or [] if isinstance(c, dict) and c.get("name") in tools]
            if names:
                return names[-1] == self.roles.get("refuse")
        return False

    def leads_whole(self) -> bool:
        """The decline review named a lead for every requested item, so a full set may exist."""
        if len(self.items) <= 1:
            return bool(self.leads)
        return all(self.lead_items.get(str(n)) for n in range(1, len(self.items) + 1))

    def rows_held(self) -> bool:
        """Kyo's opening of unread rows before an ending (decline_rows) already held an ending of this version."""
        if self.version not in self.decline_marks:
            return False
        last = next((t for t in reversed(self.trace) if t.get("kind") == "decline_rows"), {})
        return bool(last.get("read") or last.get("chain"))

    def passes(self, ids: list[str]) -> bool | None:
        """The cart check on these lines at their requested quantities: the result it gave them under the requirements
        in force, else a check now while a model call can still run (up to the spend stop); None when none can run."""
        key = self.check_key(ids)
        if ids and key not in self.verdicts and self.time_left() > LAST_CALL and self.cost < SPEND_STOP:
            self.check_cart(ids, SPEND_STOP)
        return bool(self.verdicts[key]) if key in self.verdicts else None

    def finish(self, again: bool = False) -> str:
        """End the episode: check out a held cart, else end without a purchase. Kyo's own cart audit runs first (one line
        per requested item at its quantity, nothing else; a missing line only from a qualifying listing, an empty cart from
        the decline review's leads, each once the cart check passes it while a model call can run). The checkout is
        skipped for an ending without a purchase when the cart still lacks a requested item, or when the agent's last word
        was such an ending and the cart has not passed the check. An ending the store turns down is a change: the review
        runs again and the audit carts what it finds for one more checkout. With `again`, a refused checkout is returned
        (as the store's reply) for the agent to deal with instead of being followed by an ending without a purchase."""
        if self.finished or self.broke or self.left() <= 0:
            return ""
        buy, refuse = self.roles.get("buy"), self.roles.get("refuse")
        turned = ""
        for attempt in range(2):
            stay = self.mend_cart() if buy else ""
            if self.finished or self.left() <= 0:
                return ""
            probe: list[str] = []
            if stay and refuse:
                _shape(self.quiet(refuse, dict(self.brief.get("refuse_args") or {})), self.schemas.get(refuse), "args", probe)
            # the checkout is skipped only for an ending call that can go out
            stay = stay if refuse and not probe else ""
            if stay:
                self.trace.append({"kind": "finish_out", "why": stay[:160]})
            fresh = self.cart_view is not None and self.cart_seen_at >= self.cart_at
            holding = bool(self.cart_ids()) if fresh else bool(self.cart_lines)
            gone = [i for i in self.cart_ids() if self.sold_out(i)]
            if buy and not stay and gone and refuse:
                # a cart holding a line the store says is gone is never checked out (always a wrong order)
                stay = "a cart line is sold out and no replacement was set"
                self.trace.append({"kind": "finish_out", "why": stay, "gone": gone})
            if buy and not stay and (holding or (self.cart_touched and not fresh and self.left() >= 2)):
                faults: list[str] = []
                args = _shape(self.order_args(buy), self.schemas.get(buy), "args", faults)
                if not faults:
                    # a reply that adds a requirement does not stop a forced finish: no steps are left to act on it
                    if self.left() >= 2 and self.time_left() > LAST_CALL + 12:
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
            if not refuse:
                return turned
            faults = []
            args = _shape(self.quiet(refuse, dict(self.brief.get("refuse_args") or {})), self.schemas.get(refuse), "args", faults)
            if faults:
                return turned
            if self.left() >= 2 and self.time_left() > LAST_CALL + 12:
                self.gate("end")
            if self.finished or self.left() <= 0:
                return ""
            reply = self.act(refuse, args)
            self.trace.append({"kind": "forced_refuse", "reply": _clip(reply, 200)})
            if reply is None or self.finished:
                return turned
            self.observe(refuse, reply.get("observation"))
            self.note_refusal()
            turned = "ending without a purchase was refused: " + self.show(reply.get("observation"))
            # a refused ending is a change: the review runs again, and the next round's audit carts what it finds
            # (not when the code cannot rule out that the requirements or the budget leave no fitting set)
            if attempt or self.left() < 2 or not buy:
                return turned
            self.review_again()
            lines = self.cart_faults(True)[1]
            if len(lines) < len(self.items) or self.ending_open(lines):
                return turned
        return turned

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
            self.absorb(opening)
        intro = "SHOPPER'S REQUEST:\n" + self.request
        if opening is not None:
            intro += "\n\nWHAT THE STORE SHOWS FIRST:\n" + self.show(opening)
        intro += "\n\nStart by searching for each item in its category and opening several listings in full, so the interviewer can see which properties these listings carry.\n\n" + self.status()
        self.chat = [{"role": "system", "content": GUIDE}, {"role": "user", "content": intro}]
        self.set_caps()
        self.work(self.loop_cap, SPEND_HARD)
        if not self.finished:
            turned = self.finish(again=True)
            # the store turned the ending down (a change from the shopper, a sold-out line): a short second round deals with it
            if turned and not self.finished and not self.broke and not self.no_credit and self.left() > 1 and self.time_left() > CLOSING:
                after = ("\nA store that turns down an ending without a purchase almost always has a fitting set: put the best listing for each item in the cart at the requested quantity and check out."
                         if turned.startswith("ending without") else "")
                self.chat.append({"role": "user", "content": "[Kyo] Kyo tried to end the episode and " + turned + "\nApply what the store says, then finish." + after + "\n" + self.status()})
                self.work(self.loop_calls + 10, SPEND_HARD + 0.01)
            if not self.finished:
                self.finish()
        self.trace.append({"kind": "end", "finished": self.finished, "steps": self.step, "cost": round(self.cost, 5), "calls": self.calls})
        return self.trace

    def work(self, loop_cap: int, spend_cap: float) -> None:
        idle = 0
        while (not self.finished and not self.broke and self.left() > 0 and self.loop_calls < loop_cap and self.cost < spend_cap
               and self.time_left() > CLOSING):
            if self.ready_to_ask():
                summary = self.interview()
                summary += self.value_search()
                if summary:
                    self.chat.append({"role": "user", "content": summary + "\n" + self.status()})
                continue
            self.trim()
            self.loop_calls += 1
            message = self.think(self.chat, [LIGHT, SPARE], 3000, self.offered())
            if message is None:
                self.misses += 1
                if self.no_credit or self.calls >= self.call_cap or self.misses > 2 or self.out_of_time or self.cost >= SPEND_STOP:
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
            if self.can_ask() and any(c["function"]["name"] in committing for c in calls) and self.left() > 8:
                self.chat.append({"role": "assistant", "content": content or None, "tool_calls": calls})
                for call in calls:
                    self.chat.append({"role": "tool", "tool_call_id": call["id"], "content": "Not run (no step used): the shopper is asked first."})
                summary = self.interview()
                summary += self.value_search()
                self.chat.append({"role": "user", "content": (summary or "The interviewer had nothing to ask.") + "\n" + self.status()})
                continue
            mark = self.finish_mark(calls)
            if mark and mark == self.last_finish and self.finish_refused >= 2:
                # the store turned this same finish down twice, and nothing changed since: Kyo's own finish takes over
                self.trace.append({"kind": "finish_loop", "calls": [c["function"] for c in calls], "refused": self.finish_refused})
                break
            self.chat.append({"role": "assistant", "content": content or None, "tool_calls": calls})
            self.trace.append({"kind": "turn", "text": content[:300], "calls": [c["function"] for c in calls]})
            replies = self.run_calls(calls)
            self.count_finish(mark, replies)
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
            if not episode.finished:
                episode.finish()
        except Exception as again:
            episode.trace.append({"kind": "crash", "error": f"{type(again).__name__}: {again}"[:400]})
        return episode.trace


__all__ = ["agent_main"]
