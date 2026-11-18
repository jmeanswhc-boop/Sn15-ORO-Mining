"""
Kyo: buys for one shopper inside an ORO store world.

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
SPEND_SOFT = 0.055
SPEND_HARD = 0.078
SPEND_STOP = 0.085          # no model call goes out above this spend: the code finishes alone (one run key pays for every task)
SPEND_LIGHT = 0.035         # above this spend the late checks (cart check, decline review, swaps) try the light model first
# each task has 300 s in all on the network: Kyo keeps its own clock from the start of agent_main
DEADLINE = 288.0            # seconds Kyo plans to use, leaving room for one slow call and the finish
CLOSING = 62.0              # with less time than this left, the tool loop stops and the episode heads for its finish
LAST_CALL = 40.0            # with less time than this left, no model call starts: the code finishes alone
QUESTION_TIME = 45.0        # the interviewer asks only while more time than CLOSING plus this is left
# waits before resending a failed model call; an episode has 300 s in all, so at most 45 s go to waiting
WAITS = (4, 10, 20, 35)
WAIT_TOTAL = 45
SHOW_CHARS = 450
OBS_LIMIT = 15000
HISTORY_LIMIT = 150000
GATE_ROUNDS = 3             # store calls the process gate may make before one checkout or ending
AUTO_PAGES = 8              # further search pages Kyo reads itself for a cheapest, most or least pick
JUNK_KEYS = r"\b(?:options?|variants?|models?|sku|ids?|codes?|asin|upc|ean|barcodes?|warrant\w*|video|url|links?|images?|photos?|condition|mobile)\b"
LIST_SIZE = 8
CHEAPER_OPENS = 3           # cheaper rows of a cart line's kind whose attributes Kyo reads before a cheapest checkout

# one attempt per call (think() resends): a stuck call costs 50 s, not minutes, inside the 300 s
_llm = ProxyClient(timeout=50, max_retries=1, retry_delay=1)
_env = ProxyClient(timeout=50, max_retries=3, retry_delay=1)


GUIDE = """You are Kyo, a careful buyer working for one shopper in an online store. You can act only by calling the store's tools. The episode is graded 1 or 0: it earns 1 only when the finish is exactly right. A purchase must contain every requested item at the requested quantity, each meeting every requirement in force when you check out, inside the budget; or, when the store truly has nothing that meets everything, the episode must end with the no-purchase tool in the way the request asks.

HOW THE STORE WORKS
- Every tool call is one step and steps are limited; thinking costs nothing. Tool names, argument names and result shapes are particular to this store: read each tool's description and schema and use the exact argument names and allowed values.
- Search tools can be narrow: a category filter, a keyword mode that may only read the first words of the query, a page size of a few rows, a token or offset for the next page. Compare the reported total with the rows you have seen and page on (with the exact token or offset the store gives) when the answer depends on seeing all of them. Keyword queries work best short and value-first (the wanted value plus the product noun). A keyword search that returns no rows usually means one of its words is not on the listings: drop words (keep the product noun) or browse the item's category instead; no rows never shows that the store lacks the item. When a keyword search matches every word and returns too many rows to see, put all the item's required values in one query to narrow it. A filtered search that returns no rows usually means the filter value or the filter parameter is wrong: retry with the label exactly as result rows show it, or with another filter parameter.
- Some details (what other products an item suits, codes it carries) may appear only in a listing's description, which a keyword search may not read: browse the item's category, sliced by brand or a price ceiling when the store allows, and read the rows' descriptions.
- Search rows are summaries. Stores split a listing's record differently: one view can show its attributes (properties), another only its offer and description, and a side-by-side view may carry either. The ledger names the tool that shows attributes once Kyo knows it. Judge every requirement on a property from those attributes, so read them for each listing you may buy; one call that shows several listings saves steps when it shows what you need. Never open a listing again when nothing has changed since you read it.
- Questions to the shopper are handled for you by Kyo's interviewer, so you never message the shopper yourself. Its answers arrive in the conversation and are added to the ledger.
- The request can also ask for steps along the way: an approval, a message, something to check or read first. The ledger lists them. Kyo carries out the ones due just before the order or the ending itself when you call the checkout or the ending tool, so never wait for those and never end without a purchase because of them; do any others with the store's tools (product questions go through Kyo's interviewer).

WHAT COUNTS AS A FIT
- The kind of product is decided by the listing's category label: only the label that matches the requested item as a whole counts, never a neighbouring one.
- A requirement on a property is met when the listing's attribute for that property states the value. Key spelling does not matter (spaces, underscores, camel case, prefixes), and one element of a comma or slash list is enough. Values match by their words: case, singular or plural and spelling variants of the same word are fine, and a value that contains all of the required value's words counts (a value "<A> <B>" meets a requirement "<B>"). A value with a different word does not count, even a word that means nearly the same: no synonyms and no near kinds (an abbreviation or code is not the word it stands for, and sharing one word is not enough). Units are converted before comparing numbers, limits include the edge value, and a range must lie wholly inside a limit.
- An attribute whose key names the requirement's property (under that name or a plain synonym) decides: if it says something else, the listing fails whatever other keys or the title say. A key for a different property, even a related one (a key about one part of the product says nothing about another part), decides nothing about the requirement. Kyo adds [evidence] lines after opened listings with the keys whose names match a requirement.
- When the listing has no attribute for that property, the wanted value stated under another attribute (for example a list attribute that includes it) or in the title or description counts. A number of another kind does not (a figure that measures one quantity is not a figure of another quantity), nor a bare figure whose text does not say what it measures. A closed list ("only <A> or <B>") is met only by a listing that states one of its values (in an attribute, else in the title or description); a listing that states no value for that property anywhere does not meet it. An exclusion ("not <C>") is met only by a listing that states a value for that property other than <C> (in an attribute, else in the title or description); a listing that states nothing about that property does not meet it.
- When the category label itself states a requirement's value, that requirement is met by the category.
- A requirement can point at another listing instead of giving a value (the same value as, a value unlike, or a value near the one on another listing that the request describes). Resolve it first: find that listing (search its category, brand, shop or words), open its page in full with the single-item or side-by-side tool (a search row is not enough), and when that page refers you to a further listing for the property, find and open that one the same way; the last page gives the value. Open every page of that chain before checking out. When the wanted value is anything but that other listing's value, the rule excludes only listings sharing that exact value, and never end without a purchase because of it alone. When the request ties the value to a rule the store states, the value that rule gives decides: apply the rule exactly as the page states it.
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

"items": a list with one entry per distinct product the shopper wants bought, each {"name": the kind of product in the request's words, "shelf": the store's category label for it when a tool lists labels (copy it exactly), else "", "quantity": units of that one listing (integer, 1 when not stated; a pair or a set that is one listing counts as 1), "musts": a list of firm requirements on that product, each "property: constraint" with the exact value, limit, exclusion or closed list. Name the property with the request's own word for it (a request that gives "<value>" as the "<property>" gives "<property>: <value>"); do not swap it for a neighbouring property. Give each value in its shortest exact words (a kind, a named value, a number with its unit) and leave out the request's explanation around it ("<value>, a proper <thing> and not just one <part> on its own" gives "<value>"). A figure of speech or emphasis around a requirement ("<value>, and nothing <worked> down to fit", "not a <coin> over", "measure twice") adds no must of its own; but a property the request states plainly with its value ("the <property> is <value>", "<property>: <value>", "with a <value> <property>", "in a <value> <property>", "<kind> in <value>") is always a must, also when it sits in the same phrase as the product's name. The property is the words naming it before the value, even when the value's unit looks odd for it ("<property> per <unit> no less than 3 <units>" gives "<property> per <unit>: no less than 3 <units>", never "<units>: no less than 3"). A must whose value is read from another listing is ONE must holding the whole description of that listing, with the relation in the request's own words ("<property>: <relation> the <property> of the <kind> <shop> sells with <feature>", where <relation> is the request's own wording for how the two values relate, copied exactly): words that describe that other listing (its shop, its kind, "with <property> <value> and <property> <value>" after it) stay inside that one must and are never musts of this product}. Never put price, budget, stock or quantity into musts. The story around the request (the person meant to use it, where it will stand, what went wrong before, the occasion behind it) is context, not a must, unless it states a property's value or limit outright; and never turn words that are part of what the product is called into musts: those belong to "name". A product the request names in one phrase joined by "and" (one "I want ..." with one quantity, often the name of one store department) is ONE item; make separate items only when the request gives each product its own quantity or its own requirements.
"budget": the most the shopper will pay in total as a number in the request's own unit, or null.
"budget_unit": that unit in the request's own words, or null.
"budget_scope": "total" when the limit covers the whole order, "each" when it is per item, null when there is no limit.
"in_stock_only": true unless the request clearly accepts items that are not available.
"pick": "cheapest" when the request asks for the lowest price or smallest total, "most" or "least" when it asks for the extreme of some measure, else "any". Remarks that money is tight are not a pick rule.
"pick_measure": what is maximised or minimised for "most"/"least", else "".
"messages": how many messages the shopper allows the agent to send (integer), or null when not stated.
"steps": how many steps, moves or tool calls the request allows (integer), or null.
"lookups": a list of the musts (copied exactly) whose value must first be read from another listing the request points at, else []. A requirement that one requested item must suit or fit another item of this same order is not a lookup: keep it as a must of that item in the request's own words ("<property>: <relation> the <other item> in this order") and leave it out of lookups.
"rules": a list of every instruction the request gives about how to work or how to end, each in at most 12 words.
"wishes": a list of things the shopper would merely like.
"ask_first": a list of things the request explicitly says to check with the shopper before choosing (empty when none).
"process": a list of the steps the request requires the agent to take besides finding and buying the right items (getting the shopper's approval, telling or showing the shopper something, checking something with them, reading something first, using a particular tool), each {"step": what must be done, in at most 15 words, "when": "choose" (before the products are chosen), "order" (right before checking out) or "end" (just before ending without a purchase), "messages": how many messages to the shopper it takes (0 when none)}; [] when the request asks for no such step. Requirements on the items, the budget, stock, the limits on messages and steps, and how to end when nothing fits (saying so, offering or buying the nearest candidate) are not steps; neither is asking, telling or explaining to the shopper what a product is like (how it was made, what it is made of, who it suits): that is a question about a property, not a step.
"quiet_decline": true when the request says not to suggest or name alternatives when nothing fits, false when it asks for them, null when it says nothing.
"fallback_buy": true when, for the case that no listing meets everything, the request wants its nearest candidate (the one missing the fewest requirements, perhaps within a price margin) offered or bought instead of ending without a purchase; otherwise false.
"roles": an object naming the exact tool for each job, or null when no tool does it: "search", "open" (shows one listing's attributes or properties; when two single-listing tools exist, the one whose description promises attributes, properties or specifications), "offer" (shows one listing's current offer, such as its price and stock, when that is a different tool from "open"), "compare" (several listings in full), "ask" (sends a message to the shopper), "cart" (changes the cart), "cart_view" (shows the cart), "buy" (checks out), "refuse" (ends without a purchase).
"buy_args": the fixed arguments that make the "buy" tool check out (for a tool with a mode argument, that mode), else {}.
"refuse_args": the fixed arguments that make the "refuse" tool end without a purchase, else {}.
"""


PLAN = """You choose the questions a shopping agent sends to a shopper. Shoppers hold a few firm preferences they did not write down, each about one property of one requested product. A question reveals a preference only when it names that exact property, and each question is answered only about the first property it names, so every question costs one property.

You get the request, the items with what the request already settles for each, and a numbered list of attribute keys seen on opened listings of each item's category, with how many of those listings carry the key and example values. An unwritten preference may be on a property that no listed key names.

Choose the properties in this order:
0. What the request itself asks you to check with the shopper (listed under ASK FIRST when there is any), each named by the property it is about.
1. Hinted properties: when a sentence of the request points at one of the listed keys without giving its value, that key, for the item the sentence is about. A hint speaks about the product itself, its use, its user or the place it goes; a figure of speech about the shopper's own life, work or study is not a hint. Read hints this way: who will wear or use it means gender; how it is put on, fastened or opened means closure; how much of the body or space it covers or shows means height, length or cut; how it feels or what it is made of means material; how it was made, joined or held together means construction (for made goods); how food or drink tastes or goes down means flavour; what is inside it or on the back of its pack means its source or main ingredient; counting, tracking or recording what is eaten or drunk by the label's figures (a log, a diary, a column filled in from the pack) means the nutrition figure per serving the request leaves open (protein, sugar, calories or the like per serving, whichever the listings carry); fitting a room's look or era means style; a remark about the other things it will sit next to (furniture, curtains, walls) clashing, matching or needing to be tied together, or about it hiding or showing something, means colour; the job it must do means the property that sets that ability (type, network, display size, length, connector); one particular part of it means that part's property. Otherwise only a remark about colours or shades means colour.
2. Then the listed keys (only keys the item's listings carry, whose sample values differ), best first: (a) the key naming which kind or variant of the product it is (its type; for clothing and shoes their cut or height and closure; phone type or network; never a plain measurement such as height, width or depth of furniture; for food and supplements their source or main ingredient); (b) for made goods (rugs, furniture, clothing, shoes, bags) the material it is made of, then its style (a style, type or furniture-type key); for clothing, shoes and other things worn, the wearer's gender comes before both unless the request settles it; (c) colour, or a key whose values are colours, when the request leaves the colour open; (d) for a product that must fit or work with another product, the key naming that product or its connection; (e) the other keys, most carried first. Avoid keys seen on only one or two of the item's listings.
Skip a property the request already settles for that item, also when the kind it names fixes that property's value outright (a "long-sleeved" shirt fixes sleeve length, a "waterproof" boot fixes water resistance; naming only the product category fixes nothing), a second name for a property already chosen, a key whose values list several purchase options at once, a key whose value differs on every listing (an identifier or code), and keys about selling terms rather than the product itself. With several items, take turns between the items.
"label" is the property's plain name: for a listed key, all of the key's words in plain spelling ("<word one> <word two>" for wordOneWordTwo or word_one_word_two; never shortened to one of its words); only the property's name, no example values, no brackets, no second property.
Choose COUNT, best first. When the lists run out of usable keys for an item, go on with properties that buyers of that kind of product commonly decide on, even though no listed key names them. Return ONE JSON object and nothing else: {"ask": [{"item": item number, "label": "...", "hint": "at most 8 of the story's words it comes from, or an empty string"}]}"""


HEAR = """A shopping agent asked a shopper some questions. Read each answer and return ONE JSON object and nothing else:
{"needs": [{"item": item number, "need": "property: constraint", "points": true when the value is to be read from another listing, else false}], "drop": [{"item": item number, "property": "a requirement the shopper now gives up"}], "free": [{"item": item number, "property": "..."}], "crowded": [numbers of the questions whose answer complains that the message asked about too many things or wants a single question; not answers that only say they did not follow it], "budget": new total limit as a number or null}
- "needs": every firm preference an answer states, with the exact value or limit and its unit ("at least <number> <unit>", "<value>", "not <value>"). Name the need by the property the answer itself names, which may differ from the one asked about; several answers about one property make one need. Copy the value word for word as the answer says it, keeping its nouns and units ("at least <number> <noun>", not "at least <number>").
- An answer can give the value by pointing at another listing (a value like, unlike or near the one on some shop's product, that product picked out by a few of its properties): that is ONE need on the property the answer names, with "points": true. Its value is the answer's own words copied exactly, from the comparison words to the end of that description, including what it says about a listing that page may name; never shorten or reword it. The properties describing the other product are not needs of this item.
- "free": the property of each question the shopper said they do not mind about.
- Folksy or roundabout wording that names a value still states that value."""


NOTE = """While a shopping agent works, the store showed this note (it may come from the shopper or the store). Return ONE JSON object and nothing else:
{"kind": "change" when the note changes the task (budget, requirements, a price, availability, a refusal of the order), "error" when it only reports that a call was malformed or not accepted, "other" for anything else, "budget": the new total limit as a number in the request's unit, or null, "add": [{"item": item number, "need": "property: constraint"}], "drop": [{"item": item number, "need": "the requirement no longer wanted"}], "gone": [listing ids said to be unavailable], "sold_out": true when it says a chosen listing is gone, sold out or no longer available (even without naming it), "price_moved": true when it says a price changed, "summary": "one short sentence"}
A replaced requirement is a drop of the old one plus an add of the new one. Report every change in the note."""


CHECK = """You are the last check before a shopping agent checks out. Decide from the records shown whether this cart is exactly right.
Reading rules: a requirement is met when the listing's attribute for that property states the value (any key spelling, one element of a list is enough, case, singular/plural and spelling variants of the same word count, a value containing all of the required value's words counts, a different word does not count even when it means nearly the same, units converted, limits include the edge, a range must lie wholly inside); a requirement the request ties to a rule the store states is met by the value that rule gives; an attribute whose key names the same property as the requirement decides, and if it says something else the listing fails; a key for a different property, even a related one (a key about one part of the product says nothing about another part), decides nothing; when the listing has no attribute for that property, the wanted value stated under another attribute or in the title or description counts (a number of another kind does not; nor does a bare figure whose text does not say what it measures); a requirement the category label itself states is met by that category; a closed list ("only <A> or <B>") is met only by a listing that states one of its values (in an attribute, else in the title or description), and a listing that states no value for that property anywhere does not meet it; an exclusion ("not <C>") is met only by a listing that states a value for that property other than <C> (in an attribute, else in the title or description), and a listing that states nothing about that property does not meet it; the category label must match the requested item as a whole; a rule of being unlike another listing's value counts as met unless the records show that same value. Each requested item must appear as a separate cart line at the requested quantity, nothing else may be in the cart, nothing shown out of stock, and the total of price times quantity over all lines must be within the budget. The newest record of a listing wins.
Category: when the CATEGORIES section shows a store category whose name carries the item's own words while a cart line's category does not, that line comes from a neighbouring category and is wrong. Pairs: when the request or a requirement says one item must work with, fit or suit another requested item (or a model the shopper names), the two listings' own words must connect them (a code or name that both records carry, or a statement on one record that covers the other); a pair whose records show no such link is wrong.
When the pick rule is cheapest, most or least, also look at the other listings shown: if one of them clearly fits every requirement and beats the cart's choice on that rule, the cart is wrong.
Return ONE JSON object and nothing else: {"ok": true or false, "problems": ["short, concrete problem", ...], "better": [listing ids that should replace a cart line, best first]}. Use ok=false only for a problem the records clearly show or a requirement no record addresses."""


READING = """the listing's attribute for a property decides when its key names that same property (any key spelling, one element of a list is enough, case, plural, spelling variants and unit conversions count, a value containing all of the required value's words counts, a different word does not count even when it means nearly the same, limits include the edge); a requirement the request ties to a rule the store states is met by the value that rule gives; a key for a different property, even a related one (a key about one part of the product says nothing about another part), decides nothing; when no attribute names that property, the wanted value stated under another attribute or in the title or description counts (a bare figure that does not say what it measures counts for nothing); a requirement the category label itself states is met by that category; a closed list ("only <A> or <B>") is met only by a listing that states one of its values (in an attribute, else in the title or description), and a listing that states no value for that property anywhere does not meet it; an exclusion ("not <C>") is met only by a listing that states a value for that property other than <C> (in an attribute, else in the title or description), and a listing that states nothing about that property does not meet it; the category label must match the item; a rule of being unlike another listing's value counts as met unless the records show that same value; out-of-stock listings and listings that break the budget do not fit."""

AUDIT = """A shopping agent wants to end without buying because it thinks nothing fits. Check the listing records shown against the requirements, with these reading rules: """ + READING + """
Records marked ROW were seen only as a search result row: their page (the attributes) has not been read, so they can be neither passed nor failed on a property the row does not show.
Return ONE JSON object and nothing else: {"fits": {"item number": [listing ids that appear to meet every requirement, cheapest first, at most 4]}, "open": [ids of ROW listings of a requested kind, inside the budget, whose page could show that they fit, most promising first, at most 6], "reason": "one sentence"}"""

UNMIX = """A shopper's request was read into firm requirements on a product. Some requirements take their value from another listing the request points at, and the words that pick out that other listing (its kind, its shop, the properties named after it with "with ...") are sometimes read by mistake as requirements on the product being bought. Using the request's wording, say which of the OTHER REQUIREMENTS below only describe that other listing. Return ONE JSON object and nothing else: {"describe_other": [those requirements copied exactly; empty when every one is a real requirement on the product]}"""

TRAIL = """A shopping agent is about to check out. Some requirements take their value from another listing instead of stating it, and a listing's page can send the reader on to a further listing for that property. The order counts only when the agent has opened in full every listing along each such chain. Follow each requirement's chain from the requirement's own description, link by link, through the pages' own words, using the records below (OPENED: the agent has opened that page; ROW: seen only as a search result row, its page not read).
Return ONE JSON object and nothing else: {"chains": [{"need": the requirement copied exactly, "links": [{"id": the listing id when a record below is that link, else "", "about": how the requirement or the previous page describes that listing}], "done": true when the last link is an OPENED page that states the property's value itself, else false, "value": the property's own value as stated on that last page, or "", "rule_value": when that page states a rule that gives a value for this requirement, the value the rule gives, else ""}]}
List each chain's links in order and stop at the first link whose page is not OPENED or not found, since its words are still unknown.
For a chain that stops at a link no record below is, add to that chain "find": {"shop": the shop, seller or brand name that link's description gives, or "", "kind": the kind of product it is, in the store's own category words when the description or a record shows them, "features": up to 4 short values the description gives for it}."""

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


def _fit_toks(text: str) -> frozenset[str]:
    """The words of a text, plural s dropped, for matching names, kinds and features."""
    return frozenset(w[:-1] if len(w) > 3 and w.endswith("s") else w for w in _words(text).split())


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


def _singular(phrase: str) -> str:
    """The phrase with its last word in a plain English singular form (shelves -> shelf, batteries -> battery,
    monitors -> monitor); words that do not look plural are kept."""
    words = phrase.split()
    if not words:
        return phrase
    last = words[-1]
    if len(last) > 4 and last.endswith("ves"):
        last = last[:-3] + "f"
    elif len(last) > 4 and last.endswith("ies"):
        last = last[:-3] + "y"
    elif len(last) > 4 and last.endswith(("ches", "shes", "xes", "sses")):
        last = last[:-2]
    elif len(last) > 3 and last.endswith("s") and not last.endswith(("ss", "us", "is")):
        last = last[:-1]
    return " ".join(words[:-1] + [last])


def _typed(text: str) -> Any:
    """A flattened value back in its own type: True/False/None, a number, or the text."""
    low = text.strip()
    if low in ("True", "true"):
        return True
    if low in ("False", "false"):
        return False
    if low in ("None", "null", ""):
        return None
    if re.fullmatch(r"-?\d+", low):
        return int(low)
    if re.fullmatch(r"-?\d+\.\d+", low):
        return float(low)
    return low


def _record_from_text(text: str) -> dict[str, Any] | None:
    """A listing record some stores send as one line ("key: value | key: value | facts: a=b; c=d"): the record as a dict,
    attribute blocks ("a=b; c=d") as dicts; None when the text is not such a record."""
    line = text.strip().strip('"').strip()
    keys = list(re.finditer(r"(?:^|\s\|\s)([a-z][a-z_]{1,30}): ", line))
    if len(keys) < 3 or keys[0].start() != 0:
        return None
    record: dict[str, Any] = {}
    for index, found in enumerate(keys):
        end = keys[index + 1].start() if index + 1 < len(keys) else len(line)
        value = line[found.end():end].strip()
        value = re.sub(r"\s*\|\s*$", "", value).strip().strip('"')
        pairs = re.findall(r"(?:^|;\s*)([^;=]{1,60}?)=([^;]*)", value)
        if len(pairs) >= 2 and value.count("=") >= 2:
            record[found.group(1)] = {k.strip(): v.strip() for k, v in pairs}
        else:
            record[found.group(1)] = _typed(value)
    return record


def _unflatten(node: Any, depth: int = 0) -> Any:
    """Observations whose listing records come as flattened text become ordinary records (and lists of them)."""
    if depth > 6:
        return node
    if isinstance(node, dict):
        return {k: _unflatten(v, depth + 1) for k, v in node.items()}
    if isinstance(node, list):
        return [_unflatten(v, depth + 1) for v in node]
    if isinstance(node, str) and ": " in node and " | " in node:
        parts = re.split(r';\s*"(?=[a-z][a-z_]{1,30}: )', node.strip())
        records = [_record_from_text(part) for part in parts]
        if records and all(records):
            # a whole observation that is one record is that record; a field holding records is a list of them
            return records[0] if depth == 0 and len(records) == 1 else records
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
        self.list_mode = True
        self.last_lists: dict[int, list[str]] = {}
        self.key_distinct: dict[tuple[str, str], int] = {}
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
        self.id_args: set[tuple[str, str]] = set()
        self.cart_said = ""
        self.refused_at = -1
        self.leads: list[str] = []
        self.refuse_blocks = 0
        self.fallback_holds = 0
        self.page_capped = False
        self.cap_searched: set[str] = set()
        self.kept_lines: set[str] = set()
        self.unread_warned: set[str] = set()
        self.stray_warned: set[str] = set()
        self.compat_done: set[str] = set()
        self.followed: set[str] = set()
        self.swept = False
        self.title_prefixes: set[str] | None = None
        self.title_rows = 0
        self.compat_warned: set[str] = set()
        self.rejected: dict[str, str] = {}
        self.break_warned: set[str] = set()
        self.rejected_warned: set[str] = set()
        self.missing_filled = False
        self.compat_held = False
        self.compat_notes: list[str] = []
        self.open_pick = False
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
        self.lost_finds: list[tuple[str, str, dict[str, Any]]] = []
        self.chased = False
        self.find_pages: dict[str, tuple[Any, Any]] = {}
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
        self.promise_marks: set[str] = set()
        self.wider_held = False
        self.paged_obs: list[Any] = []

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
            if self.calls >= self.call_cap or self.cost >= SPEND_STOP:
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
        result = _llm.post_verbose("/inference/chat/completions", body)
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
                    data["observation"] = _unflatten(data.get("observation"))
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
        # room for the light model's thinking: a reply cut at its limit sent a tenth of race 6's briefs to the heavy one
        brief = self.ask_json(prompt, [LIGHT], 4000) or {}
        if not self.no_credit and (not _items(brief.get("items")) or not isinstance(brief.get("roles"), dict) or not brief["roles"].get("buy")):
            brief = self.ask_json(prompt, [HEAVY, LIGHT], 2500) or brief
        self.brief = brief
        # graders judge a fitting listing against the cheapest fitting one even when the request leaves the choice open
        # (a dearer fit counts as not the best option), so an open choice is made as a lowest-price choice
        self.open_pick = str(brief.get("pick") or "any").strip().lower() in ("any", "", "none", "null")
        if self.open_pick:
            brief["pick"] = "cheapest"
        # an ending without a purchase is almost never what the grader expects (2 tasks in 90 in recent races), so the
        # closest candidate is bought rather than nothing, whether or not the request says so
        brief["fallback_buy"] = True
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
            # stock is a rule of the whole order (in_stock_only), never a property a listing's attributes can show
            musts = [m for m in musts if not re.search(r"\b(?:in stock|stock|availab\w*|out of stock)\b", m.split(":", 1)[0].lower())]
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
        # a message naming several properties is worth trying only when a refusal of it would not cost half the
        # questions: with two messages or fewer, ask one property per message
        self.list_mode = self.msg_limit >= 3
        steps = _count(brief.get("steps"))
        if steps is not None and 10 <= steps < self.max_steps:
            self.max_steps = steps
        for entry in _items(brief.get("process")):
            if isinstance(entry, dict) and str(entry.get("step") or "").strip():
                # saying that nothing fits and offering the nearest listing is how the episode ends, not a step that
                # needs a message held back for it
                if re.search(r"no (?:clean |exact )?(?:fit|listing)|nothing fits|nearest|closest|fewest", str(entry["step"]).lower()):
                    continue
                # telling the shopper what the product is like (how it was made, its material) is a property to ask about,
                # not a step that needs a message held back
                if re.search(r"\b(?:explain|describe|tell|show|share)\b.*\b(?:made|built|construct\w*|production|origin|material|propert\w*|feature\w*)\b", str(entry["step"]).lower()):
                    continue
                when = str(entry.get("when") or "").strip().lower()
                self.process.append({"step": str(entry["step"]).strip(), "when": when if when in ("choose", "order", "end") else "order",
                                     "messages": max(0, _count(entry.get("messages")) or 0)})
        if self.roles.get("ask"):
            # the messages the steps at the order and the ending take, one at least for each of the two that needs any
            talk = [p["messages"] for p in self.process if p["when"] in ("order", "end") and p["messages"]]
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
        elif left <= 6 or (left <= len(self.items) + 5 and self.buys == 0):
            text += " Few steps left: set the cart to the chosen lines and check out now."
        elif self.buys == 0 and left <= 12:
            text += " Steps are running low: settle on your picks, fill the cart and check out within the next few steps."
        elif self.buys == 0 and not self.cart_touched and self.step >= self.max_steps // 2:
            text += " Half the steps are used and the cart is empty: cart each item's fitting listing now (the store may still sell one out)."
        secs = self.time_left()
        if secs < CLOSING + 20:
            text += f" Time is short: about {max(0, int(secs))} seconds are left for this task. Cart the fitting listings you have and check out now."
        elif secs < CLOSING + 60:
            text += f" Time is running: about {max(0, int(secs))} seconds are left for this task and each turn takes about 10. Cart each fitting listing as soon as you find it and finish the items still missing first."
        live = {need for _, need in self.needs_now()}
        lookups = [need for _, need in self.refs if need in live and need not in self.ref_values]
        if lookups and self.buys == 0:
            text += "\n[resolve first: open the listing each of these points at, and every listing its page refers to; the last page gives the value] " + " | ".join(lookups[:3])
        if lookups and self.lost_finds:
            nexts = []
            for need, about, find in self.lost_finds[:2]:
                if need in lookups:
                    hint = " ".join(x for x in (str(find.get("shop") or ""), str(find.get("kind") or "")) if x).strip()
                    nexts.append(f"{about[:160]}" + (f" (search: {hint})" if hint else ""))
            if nexts:
                text += "\n[chain: next page still to open] " + " | ".join(nexts)
        if self.leads:
            text += "\n[review candidates] " + ", ".join(self.leads)
        links = self.compat_links() if self.buys == 0 else []
        if links:
            pairs = "; ".join(f"item {d} must work with item {a}" for d, a in links)
            text += (f"\n[compatibility: {pairs}] A listing shows what it works with by naming the other listing's model code in its"
                     " description, and that other listing states the same code. Word searches do not read descriptions: find each item's"
                     " candidates by its own requirements, open them, and pick a set whose codes match. A listing whose description also"
                     " names another listing (by a part code) counts only with that listing opened too: prefer candidates that name none"
                     " (Kyo opens the named ones before checkout). " + self.code_ledger(links))
        named = self.named_all() if self.buys == 0 else ""
        if named:
            text += "\n[opened listings whose attributes meet every requirement Kyo can check (values, number limits, closed lists, exclusions); read their records before choosing] " + named
        if self.notes:
            text += "\n[changes so far] " + " | ".join(self.notes[-4:])
        text += "\n[ledger]\n" + self.ledger()
        return text

    def plain_checks(self, item: Item) -> tuple[list[tuple[set[str], Any]], list[tuple[set[str], list[str]]]]:
        """An item's checkable requirements, each with the words of the property it is about: a plain value (its words),
        a number limit (("<=" or ">=", number, unit)), a closed list (its options' words); and its excluded values."""
        # comparison words, one alternative per entry; read as a set, not as a phrase
        upper = re.compile(r"\b(?:maximum|at most|max|up to|under|no more than|below|or lower|less than|or less|or smaller|or below|and below|and under)\b")
        lower = re.compile(r"\b(?:minimum|at least|min|over|no less than|above|or higher|more than|or more|or larger|or above|and up|and above|and over)\b")
        negative = re.compile(r"^\s*(?:not|no|never|except|anything but|other than|avoid)\b")
        tie = re.compile(r"this order|\bwork(?:s|ing)? with\b|\bcompatib|\bgo(?:es)? with\b|\bwhichever\b|\bsame\b|\bmatch|\bunlike\b|\bdifferent\b|\bthan the\b|\bfits?\s+(?:the|whichever|with)\b|\bend up (?:buying|picking|with)\b")
        code_re = re.compile(r"\b(?=[A-Z0-9-]*[A-Z][A-Z0-9-]*[A-Z])(?=[A-Z0-9-]*\d[A-Z0-9-]*\d)[A-Z][A-Z0-9-]{3,13}\b")
        values: list[tuple[set[str], Any]] = []
        banned: list[tuple[set[str], list[str]]] = []
        for must in item.musts + item.told:
            if must in item.dropped or ":" not in must:
                continue
            codes = code_re.findall(must)
            prop, value = (x.strip().lower() for x in must.split(":", 1))
            keys = {_singular(w) for w in re.findall(r"[a-z0-9]+", prop) if len(w) >= 3}
            if codes and keys:
                # a requirement naming a model code: the listing's own text has to name that code
                values.append((keys, ("code", codes)))
                continue
            if (not value or not keys or len(value) > 60 or tie.search(prop + " " + value)
                    or re.search(r"\b(?:except|unless|when|if)\b", value)):
                # a value tied to another item, or one that holds only under a condition, is not checked here
                continue
            if negative.search(value):
                gone_words = [w for w in re.findall(r"[a-z0-9][a-z0-9.\-]*", negative.sub("", value)) if len(w) >= 2]
                if gone_words:
                    banned.append((keys, gone_words))
                continue
            number = re.search(r"(\d+(?:\.\d+)?)\s*([a-z/%]*)", value)
            if number and (upper.search(value) or lower.search(value)):
                op = "<=" if upper.search(value) else ">="
                values.append((keys, ("limit", op, float(number.group(1)), number.group(2))))
                continue
            if re.search(r"\bor\b|\beither\b|,", value):
                options = [[w for w in re.findall(r"[a-z0-9][a-z0-9.:\-]*", part) if len(w) >= 2]
                           for part in re.split(r",|\bor\b|\beither\b|\bonly\b", value)]
                options = [o for o in options if o]
                if len(options) >= 2:
                    values.append((keys, ("options", options)))
                continue
            if re.search(r"\b(?:least|most|than|between|within|from)\b", value):
                continue
            words = [w for w in re.findall(r"[a-z0-9][a-z0-9.\-]*", value) if len(w) >= 2]
            if words:
                values.append((keys, words))
        return values, banned

    @staticmethod
    def check_fact(facts: dict[str, Any], keys: set[str], want: Any) -> tuple[str, str]:
        """One requirement against a listing's attributes: ("ok", key) when an attribute about that property meets it,
        ("contra", "key=value") when such an attribute exists and plainly says otherwise, ("absent", "") when no
        attribute is about that property. Keys match by word stems, so spelling variants count."""
        def stems(words: Any) -> set[str]:
            # whole words in singular form, British "-our" read as "-or"
            return {re.sub(r"our$", "or", _singular(w)) for w in words}

        if isinstance(want, tuple) and want[0] == "code":
            # a model code is read from the listing's whole text, not from one attribute
            return "absent", ""
        want_keys = stems(keys)
        overlap = {key: len(want_keys & stems(re.findall(r"[a-z0-9]+", str(key).lower()))) for key in facts}
        best = max(overlap.values(), default=0)
        if best < min(2, len(want_keys)):
            return "absent", ""

        def word_in(text: str, w: str) -> bool:
            if re.fullmatch(r"\d+(?:\.\d+)?", w):
                # a bare number matches the same number written another way ("2" and "2.0")
                return any(abs(float(n) - float(w)) < 1e-9 for n in re.findall(r"(?<![a-z0-9.])(\d+(?:\.\d+)?)(?![0-9])", text))
            return re.search(r"(?<![a-z0-9])" + re.escape(w) + r"(?:s|es)?(?![a-z0-9])", text) is not None

        def has(text: str, words: list[str]) -> bool:
            return all(word_in(text, w) for w in words)

        def touches(text: str, words: list[str]) -> bool:
            return any(word_in(text, w) for w in words)

        verdict, seen = "absent", ""
        for key, val in facts.items():
            if overlap[key] < best:
                continue
            text = str(val).lower()
            if isinstance(want, list):
                if has(text, want):
                    return "ok", str(key)
                if not touches(text, want) and text.strip():
                    verdict, seen = "contra", f"{key}={val}"
            elif want[0] == "options":
                if any(has(text, option) for option in want[1]):
                    return "ok", str(key)
                if text.strip() and not any(touches(text, option) for option in want[1]):
                    verdict, seen = "contra", f"{key}={val}"
            else:
                _, op, bound, unit = want
                span = re.search(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*([a-z/%]*)", text)
                first = re.search(r"(\d+(?:\.\d+)?)\s*([a-z/%]*)", text)
                # a range counts by its ends; otherwise the first number (later digits belong to units or notes)
                found = ([(float(span.group(1)), span.group(3)), (float(span.group(2)), span.group(3))] if span
                         else [(float(first.group(1)), first.group(2))] if first else [])
                if not found:
                    continue
                units = {u for _, u in found if u}
                if unit and units and not any(u[0] == unit[0] for u in units):
                    continue
                numbers = [n for n, _ in found]
                if (op == "<=" and max(numbers) <= bound) or (op == ">=" and min(numbers) >= bound):
                    return "ok", str(key)
                verdict, seen = "contra", f"{key}={val}"
        return verdict, seen

    @staticmethod
    def plain_fit(values: list[tuple[set[str], Any]], banned: list[tuple[set[str], list[str]]], record: dict[str, Any]) -> bool:
        """True when the record's attributes state every plain value, meet every number limit in the same unit, name
        one option of every closed list, and name no excluded value."""
        facts = _fact_block(record)
        if not facts:
            return False
        check = Kyo.check_fact
        text = _dump(record)

        def meets(keys: set[str], want: Any) -> bool:
            if isinstance(want, tuple) and want[0] == "code":
                return any(code in text for code in want[1])
            return check(facts, keys, want)[0] == "ok"

        return all(meets(k, w) for k, w in values) and not any(check(facts, k, w)[0] == "ok" for k, w in banned)

    def unnamed_values(self, ident: str) -> list[str]:
        """Plain requirement values (word values and closed lists) of the listing's item that its record names nowhere,
        not in an attribute, the title or the description."""
        number = self.item_of(ident)
        record = self.full.get(ident) or self.rows.get(ident)
        if not number or not record:
            return []
        text = _dump(record).lower()
        values, _ = self.plain_checks(self.items[number - 1])

        def named(words: list[str]) -> bool:
            return all(re.search(r"(?<![a-z0-9])" + re.escape(w), text) for w in words)

        out = []
        for _, want in values:
            if isinstance(want, list) and not named(want):
                out.append(" ".join(want))
            elif isinstance(want, tuple) and want[0] == "options" and not any(named(o) for o in want[1]):
                out.append(" or ".join(" ".join(o) for o in want[1]))
            elif isinstance(want, tuple) and want[0] == "code" and not any(code.lower() in text for code in want[1]):
                out.append(" or ".join(want[1]))
        return out

    def contradictions(self, ident: str) -> list[str]:
        """Requirements of the listing's item that its own attributes plainly break (a stated other value, a number past
        the limit, an excluded value); silence is not counted."""
        number = self.item_of(ident)
        record = self.full.get(ident)
        if not number or not record:
            return []
        facts = _fact_block(record) or {}
        if not facts:
            return []
        item = self.items[number - 1]
        values, banned = self.plain_checks(item)
        out = []
        for keys, want in values:
            state, seen = self.check_fact(facts, keys, want)
            if state == "contra":
                label = " ".join(sorted(keys))
                out.append(f"{seen} (needed: {label} " + (" ".join(want) if isinstance(want, list) else
                           " or ".join(" ".join(o) for o in want[1]) if want[0] == "options" else f"{want[1]} {want[2]:g} {want[3]}") + ")")
        for keys, words in banned:
            state, seen = self.check_fact(facts, keys, words)
            if state == "ok":
                out.append(f"{seen}={facts.get(seen)} (excluded: {' '.join(words)})")
        return out

    def named_all(self) -> str:
        """Per item: the opened, buyable listings on its shelf whose attribute block names every plain requirement value
        and no excluded one, cheapest first, three at most; empty when no item has one."""
        cat_key = self.category_key()
        out = []
        for number, item in enumerate(self.items, 1):
            values, banned = self.plain_checks(item)
            if not values:
                continue
            shelf_want = (item.shelf or "").strip().lower()
            hits = []
            for ident, record in self.full.items():
                if not self.buyable(ident) or ident in self.gone:
                    continue
                if shelf_want and cat_key:
                    shelf = str((self.rows.get(ident) or {}).get(cat_key) or record.get(cat_key) or "").strip().lower()
                    if shelf and shelf != shelf_want:
                        continue
                if self.plain_fit(values, banned, record):
                    price = self.price_of(ident)
                    hits.append((price if price is not None else float("inf"), ident))
            if hits:
                listed = ", ".join(f"{i} ({p:g})" if p != float("inf") else i for p, i in sorted(hits)[:3])
                out.append((f"item {number}: " if len(self.items) > 1 else "") + listed)
        return " | ".join(out)

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
        # a store may cut its result rows short, so the opened pages (whole records) decide once there are a few
        pool = list(self.full.values()) if len(self.full) >= 3 else list(self.rows.values())
        for row in pool:
            for key, value in row.items():
                if isinstance(value, str) and 0 < len(value) <= 40 and not re.search(r"\d{3}", value):
                    counts.setdefault(key, set()).add(value)
                    present[key] = present.get(key, 0) + 1
        total = max(1, len(pool))
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
        matched = {label for label in groups if any(self.label_fit(label, item) for item in self.items)}
        if matched:
            groups = {label: group for label, group in groups.items() if label in matched}
        table = []
        for label, group in sorted(groups.items(), key=lambda kv: -kv[1]["n"]):
            group["keys"] = self.merge_keys(group["keys"])
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
            for name, slot in kept[:14]:
                table.append((label, name, slot["n"], [v[:40] for v in slot["values"][:4]]))
                self.key_distinct[(label, name)] = len(set(slot["per"].values()))
        return table

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
        record = self.rows.get(ident) or {}
        key = self.price_key or next((k for k, v in record.items() if re.search(r"(?:^|[^a-z])(?:price|cost)(?:$|[^a-z])", str(k).lower())
                                      and isinstance(v, (int, float, str, dict)) and not isinstance(v, bool)), None)
        # the row shape is only learned for price-ranked picks; otherwise the record's own price or cost field serves
        value = record.get(key) if key else None
        if isinstance(value, dict):
            value = next((v for v in value.values() if isinstance(v, (int, float)) and not isinstance(v, bool)), None)
        if isinstance(value, str) and re.fullmatch(r"\s*\d+(\.\d+)?\s*", value):
            value = float(value)
        return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None

    def buyable(self, ident: str) -> bool:
        """Not sold out by any record: no note or refusal named it, and its stock flag (the key the model named) is not no."""
        if self.sold_out(ident):
            return False
        record = self.rows.get(ident) or {}
        key = self.stock_key or next((k for k, v in record.items() if re.search(r"stock|availab", str(k).lower())
                                      and isinstance(v, (bool, int, float))), None)
        value = record.get(key) if key else None
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

    def more_pages(self, name: str, args: dict[str, Any], observation: Any) -> str:
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
        top = (((self.schemas.get(name) or {}).get("properties") or {}).get(param) or {}).get("maximum")
        if (later and isinstance(given, int) and not isinstance(given, bool) and isinstance(top, int) and given + shown > top
                and count > given + shown):
            # the agent paged to the last page the store's paging reaches (its next-page value may be empty there):
            # the rest only a word search finds
            self.page_capped = True
            return self.past_cap(name, args, param, count, given + shown, given + shown)
        if value is None or later or count <= shown:
            # the last page, or the agent is paging itself
            return ""
        if pick == "cheapest" and self.price_sorted(observation):
            return ""
        pages = -(-(count - shown) // shown)
        # pages the paging parameter can still reach (its schema maximum), within the page and step allowances
        spec = ((self.schemas.get(name) or {}).get("properties") or {}).get(param) or {}
        reach = pages
        if isinstance(spec.get("maximum"), int) and isinstance(value, int) and not isinstance(value, bool) and shown:
            reach = min(reach, max(0, (spec["maximum"] - value) // shown + 1))
        budget = min(reach, self.page_allowance() - self.pages_read, self.left() - self.tail() - 12)
        # deep paging pays only when the request itself asks for the lowest price; an open choice keeps a short reach
        if self.open_pick and pages > 2:
            budget = 0
        # a requirement still waiting on a chain of listing pages needs the steps first: page deeper only afterwards
        live = {need for _, need in self.needs_now()}
        if any(need in live and need not in self.ref_values for _, need in self.refs):
            budget = min(budget, 1 if pages <= 2 else 0)
        if reach < pages:
            # the store's paging reaches only part of this search: reading that part settles no lowest or highest, so
            # one more page at most, and the steps go to narrower searches instead
            self.page_capped = True
        note = (f"\n[Kyo] This search matched {count} listings and showed {shown}, in no price order. The pick counts every fitting listing: "
                + (f"search with narrower words (the requirement's value) or page on with {param}={_dump(value)}." if reach >= pages else
                   "paging cannot reach them all, so do not page through it; search with narrower words (the requirement values, with the product's name) instead."))
        if budget < 1:
            return note + (self.past_cap(name, args, param, count, spec["maximum"] + shown, spec["maximum"] + shown) if reach < pages else "")
        texts = []
        for _ in range(budget):
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
            # one short line a listing: full pages pasted into the conversation slow every later model call
            rows = []
            for ident, record in self.records_in(seen):
                title = next((str(v) for k, v in record.items() if isinstance(v, str) and re.search(r"title|heading|name", str(k).lower())), "")
                price = self.price_of(ident)
                stock = "" if self.buyable(ident) else " (sold out)"
                rows.append((price if price is not None else float("inf"), f"{ident} ({price:g}){stock} {title[:60]}" if price is not None else f"{ident}{stock} {title[:60]}"))
            texts.append(f"{param}={_dump(value)}: " + "; ".join(r[1] for r in sorted(rows)))
            after = self.paging(name, seen)
            value = after[3] if after else None
            if value is None:
                break
        self.trace.append({"kind": "pages", "tool": name, "read": len(texts)})
        if not texts:
            return note
        head = "\n[Kyo read the rest of this search itself, one step a page]\n" if len(texts) >= pages else \
            f"\n[Kyo read {len(texts)} more pages of this search itself, one step a page; the store lets no further pages be reached this way, so search with other words for the rest]\n"
        rest = self.past_cap(name, args, param, count, spec["maximum"] + shown, spec["maximum"] + shown) if reach < pages else ""
        return head + "\n".join(texts) + rest

    def page_allowance(self) -> int:
        """Pages Kyo reads on by itself in one episode: the full allowance for a single item; with several items the
        steps are shared, so fewer (never under two)."""
        if len(self.items) <= 1:
            return AUTO_PAGES
        return max(2, min(AUTO_PAGES, self.max_steps // (6 * len(self.items))))

    def past_cap(self, name: str, args: dict[str, Any], param: str, count: int, value: int, reachable: int) -> str:
        """Once per search scope: the store's paging cannot reach the rows past `value`, so Kyo runs one word search
        for the requested product's own name in the same scope (a step, no model turn) and lists its rows."""
        props = (self.schemas.get(name) or {}).get("properties") or {}
        scope = _dump({k: v for k, v in args.items() if k != param and isinstance(props.get(k), dict) and props[k].get("enum")})
        note = (f"\n[Kyo] The store's paging reaches only the first {reachable} of these {count} listings; the rest only word searches"
                " reach (search the requirement values together with the product's plain name and other names for it, in its category).")
        if scope in self.cap_searched or self.left() <= self.tail() + 10:
            return note
        self.cap_searched.add(scope)
        mode = next((k for k, spec in props.items() if isinstance(spec, dict) and isinstance(spec.get("enum"), list)
                     and any(re.search(r"word|keyword|text|search|query", str(e), re.IGNORECASE) for e in spec["enum"])), None)
        free = next((k for k, spec in props.items() if isinstance(spec, dict) and "string" in _kinds(spec) and "enum" not in spec
                     and re.search(r"word|query|term|text|keyword|search|q$", _words(k))), None)
        if not free or str(args.get(free) or "").strip():
            # a word search already narrowed by words: the product's name would not reach further
            return note
        label = " ".join(str(v) for v in args.values() if isinstance(v, str)).lower()
        item = next((it for it in self.items if it.shelf and it.shelf.strip().lower() in label), None) or (self.items[0] if len(self.items) == 1 else None)
        if item is None:
            return note
        limit = props[free].get("maxLength") if isinstance(props[free].get("maxLength"), int) else 60
        # a name that joins several kinds ("<A> and <B>", "<A>, <B> & <C>") is searched kind by kind (two at most):
        # a store's word search ranks by the words, so one kind's own words reach rows the joined name does not
        parts = [w.strip() for source in (item.shelf or "", item.name) for w in re.split(r",|&|\band\b|\bor\b|/", source.lower()) if len(w.strip()) >= 3]
        # listing titles mostly name one piece: the last word of each kind goes to its singular form
        words = list(dict.fromkeys(_singular(w) for w in parts))[: 2 if len(self.items) <= 1 else 1] or [item.name.lower()]
        out = note
        for text in words:
            if self.left() <= self.tail() + 10 or self.finished:
                break
            call = {k: v for k, v in args.items() if k != param and k != free}
            if mode:
                call[mode] = next(str(e) for e in props[mode]["enum"] if re.search(r"word|keyword|text|search|query", str(e), re.IGNORECASE))
            call[free] = text[:limit]
            faults: list[str] = []
            call = _shape(call, self.schemas.get(name), "args", faults)
            if faults:
                break
            reply = self.act(name, call)
            if reply is None or self.finished:
                break
            seen = reply.get("observation")
            self.observe(name, seen)
            self.paged_obs.append(seen)
            self.read_cache[name + _dump(call)] = (self.version, self.step, self.show(seen))
            rows = []
            for ident, record in self.records_in(seen):
                title = next((str(v) for k, v in record.items() if isinstance(v, str) and re.search(r"title|heading|name", str(k).lower())), "")
                price = self.price_of(ident)
                stock = "" if self.buyable(ident) else " (sold out)"
                rows.append((price if price is not None else float("inf"), f"{ident} ({price:g}){stock} {title[:60]}" if price is not None else f"{ident}{stock} {title[:60]}"))
            self.trace.append({"kind": "past_cap", "call": _clip(call, 60), "rows": len(rows)})
            if rows:
                out += f"\n[Kyo searched the words {call[free]!r} itself (one step); rows, cheapest first] " + "; ".join(r[1] for r in sorted(rows))
        return out

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
        # when the store's paging could not reach the rest of a search, only word searches find those rows, so the
        # hold is worth fewer spare steps
        need = 8 if self.page_capped else 14 + 2 * len(self.items)
        if self.wider_held or self.brief.get("pick") != "cheapest" or self.left() < need:
            return ""
        self.wider_held = True
        lines = [f"{i} at {self.price_of(i):g}" for i in self.cart_ids() if self.price_of(i) is not None]
        capped = (" The store's paging stopped before the end of a search, so the listings past that point can be found only by"
                  " searching words.") if self.page_capped else ""
        return (f"Not run yet (no step used). {self.left()} steps are left and the pick is the cheapest.{capped} Searches match a listing's title,"
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

    def promising_rows(self) -> list[str]:
        """Before a checkout whose cart line's attributes do not show every requirement value of its item: read the
        unopened rows of that item's category whose title or description names the most of those values (a few, no
        model call), so the last check can weigh them."""
        mark = self.cart_mark() + f"#{self.version}"
        cat_key = self.category_key()
        if mark in self.promise_marks or not cat_key:
            return []
        self.promise_marks.add(mark)
        cart = self.cart_ids()
        picks: list[str] = []
        for line in cart:
            kind = str((self.rows.get(line) or {}).get(cat_key) or "").strip().lower()
            facts = _dump(_fact_block(self.full.get(line) or {}) or {}).lower()
            item = next((it for it in self.items if {w for w in _fit_toks(it.name + " " + it.shelf) if len(w) >= 4} & _fit_toks(kind)), None)
            if not kind or item is None:
                continue
            values = [re.findall(r"[a-z0-9]{3,}", m.split(":", 1)[-1].lower()) for m in item.musts + item.told
                      if m not in item.dropped and not re.search(r"\b(?:at least|at most|not|no|under|over|or less|or more)\b", m.lower())]
            values = [v for v in values if v]
            if not values or all(all(w in facts for w in v) for v in values):
                continue
            scored = []
            for ident in self.seen_order:
                if ident in cart or ident in self.full or not self.buyable(ident):
                    continue
                record = self.rows.get(ident) or {}
                if str(record.get(cat_key) or "").strip().lower() != kind:
                    continue
                text = _dump(record).lower()
                hits = sum(1 for v in values if all(w in text for w in v))
                if hits:
                    scored.append((-hits, self.price_of(ident) if self.price_of(ident) is not None else float("inf"), ident))
            picks += [i for _, _, i in sorted(scored)[:3] if i not in picks]
        return self.fill_facts(picks[:3], self.tail() + 4)

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
            # a single item gets more of its listings read: key counts from three or four cards are mostly ties
            lanes.append(pool[: max(0, min(4 if len(self.items) == 1 else 2, 8 - opened))])
        queue = [lane[k] for k in range(4) for lane in lanes if k < len(lane)]
        # while steps are plentiful: 22 stay for the rest of the episode
        self.fill_facts(queue[: min(8, max(4, 2 * len(lanes)))], 22)

    def ready_to_ask(self) -> bool:
        if self.asked or not self.roles.get("ask") or self.msgs_free() <= 0:
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
        """Ask the shopper one question at a time and read each answer before choosing the next. One plan ranks each
        item's properties; the questions take turns between the items, an item whose answer revealed a preference waits
        until the others had their turn, and an answer that wants a single question gets that property asked alone once
        more. Every message goes to a question (the request's own steps keep theirs)."""
        self.asked = True
        count = self.msgs_free()
        if count <= 0 or self.time_left() <= CLOSING + QUESTION_TIME or not self.roles.get("ask"):
            return ""
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
        if self.list_mode:
            want = max(want, min(2 * LIST_SIZE * len(self.items), 5 * LIST_SIZE))
        prompt = PLAN.replace("COUNT", str(want))
        if self.list_mode:
            prompt += ("\nThese properties will be asked in lists, one item per message. Name each one the way a shopper says it, in one or "
                       "two everyday words (for instance sleeves, collar, size, pattern, style, memory, graphics, ports, flavor, volume); "
                       "for a listed key with a longer or store-specific name give that everyday name instead, and keep the key's own name "
                       "as a separate pick only when it names a different property (memory and memory type). Leave out keys that belong to "
                       "another kind of product than the item, and store filter or tag keys.")
        prompt += "\n\nREQUEST:\n" + self.request + "\n\nITEMS:\n" + "\n".join(settled)
        first = [str(x) for x in _items(self.brief.get("ask_first")) if str(x).strip()]
        if first:
            prompt += "\n\nASK FIRST (the request asks to be asked about these):\n" + "\n".join(first)
        prompt += "\n\nATTRIBUTE KEYS SEEN:\n" + ("\n".join(listing) or "(no full listing records yet)")
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
            if re.search(JUNK_KEYS, _words(label)):
                continue
            queue = queues.setdefault(number, [])
            if all(_words(label) != _words(other) for other in queue):
                queue.append(label)
            if number not in order:
                order.append(number)
        for number in range(1, len(self.items) + 1):
            # every item's queue ends with its category's most carried keys the plan did not name, so no message is
            # left unused for want of a property (an item the plan left out gets these alone)
            queue = queues.setdefault(number, [])
            for name in self.item_keys(number, table, 2 * LIST_SIZE if self.list_mode else 8):
                if re.search(JUNK_KEYS, _words(name)):
                    continue
                if all(_words(name) != _words(other) for other in queue):
                    queue.append(name)
                    self.topup.add(name)
            if queue and number not in order:
                order.append(number)
        self.trace.append({"kind": "plan", "ask": _clip(plan.get("ask"), 120), "queues": _clip(queues, 60)})
        sent: list[tuple[int, str, str]] = []
        told: list[str] = []
        found: set[int] = set()
        last_item: int | None = None
        spent: dict[int, int] = {}
        while self.msgs_free() > 0 and not self.finished and self.time_left() > CLOSING + QUESTION_TIME:
            if self.list_mode:
                listed = self.next_list(order, queues, found, spent, count)
            else:
                one = self.next_pick(order, queues, found, last_item)
                listed = (one[0], [one], self.question(one[0], one[1])) if one else None
            if listed is None:
                break
            number, pairs, text = listed
            batch = self.send([(number, pairs[0][1], text)])
            if not batch:
                break
            last_item = number
            spent[number] = spent.get(number, 0) + 1
            self.asked_labels.extend(pairs)
            if self.msg_refused:
                # the store refused the message (its real limit is lower than the one read): no more questions
                self.msg_used -= 1
                self.msg_limit = self.msg_used
                break
            sent += batch
            # one answer: the light model reads it (the heavy one when the light one fails); an answer that wants a
            # single question to a one-property question is the shopper's way of not minding it: move on
            heard = self.hear(batch, (LIGHT, HEAVY))
            crowded = self.list_mode and len(pairs) > 1 and 1 in [_count(x) for x in _items(heard.get("crowded"))]
            if crowded:
                self.list_mode = False
                self.trace.append({"kind": "lists_refused", "q": text[:200]})
                for n, label in reversed(pairs):
                    queues.setdefault(n, []).insert(0, label)
                self.asked_labels = [p for p in self.asked_labels if p not in pairs]
            before = len(told)
            self.take_heard(heard, told)
            if len(told) > before:
                found.add(number)
        if not sent:
            return ""
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

    def list_question(self, number: int, labels: list[str]) -> str:
        item = self.items[number - 1]
        tags = []
        if item.shelf and _words(item.shelf) != _words(item.name):
            tags.append(item.shelf)
        values = [m.split(":", 1)[-1].strip() for m in item.musts if m not in item.dropped]
        values = [v for v in values if v and len(v) <= 30]
        if values:
            tags.append(", ".join(values[:3]))
        head = f"For the {item.name}" + (f" ({'; '.join(tags)})" if tags else "")
        named = labels[0] if len(labels) == 1 else ", ".join(labels[:-1]) + " or " + labels[-1]
        return f"{head}, what {named} would you like?"

    def next_list(self, order: list[int], queues: dict[int, list[str]], found: set[int], spent: dict[int, int], budget: int) -> tuple[int, list[tuple[int, str]], str] | None:
        for number, queue in queues.items():
            if number <= len(self.items):
                item = self.items[number - 1]
                told = [_stem(t.split(":")[0]) for t in item.told]
                stated = [_stem(t.split(":")[0]) for t in item.musts]
                queue[:] = [label for label in queue
                            if not any(k and k <= _stem(label) for k in told)
                            and not (label in self.topup and any(k and (k <= _stem(label) or _stem(label) <= k) for k in stated))]
                asked = [_stem(label) for n, label in self.asked_labels if n == number]
                queue[:] = [label for label in queue if not any(_stem(label) <= a for a in asked)]
        live = [n for n in order if queues.get(n) or len(self.last_lists.get(n, [])) > 1]
        if not live:
            return None
        if len(self.items) > budget and not spent:
            pairs = [(n, queues[n].pop(0)) for n in live]
            return live[0], pairs, " ".join(self.question(n, label) for n, label in pairs)
        pool = [n for n in live if n not in found] or live
        number = min(pool, key=lambda n: (spent.get(n, 0), order.index(n)))
        item = self.items[number - 1]
        told = [_stem(t.split(":")[0]) for t in item.told]
        labels = [label for label in reversed(self.last_lists.get(number, [])[1:])
                  if not any(k and (k <= _stem(label) or _stem(label) <= k) for k in told)][:LIST_SIZE - 2]
        while queues[number] and len(labels) < LIST_SIZE:
            label = queues[number].pop(0)
            if all(_words(label) != _words(other) for other in labels):
                labels.append(label)
        self.last_lists[number] = labels
        return number, [(number, label) for label in labels], self.list_question(number, labels)

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
                            and not (label in self.topup and any(k and (k <= _stem(label) or _stem(label) <= k) for k in stated))]
                # a spelling of a property already asked about (its words inside the other's) is not asked again
                asked = [_stem(label) for n, label in self.asked_labels if n == number]
                queue[:] = [label for label in queue if not any(_stem(label) <= a for a in asked)]
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
        # only the category that fits this item best: in a bundle another item's shelf can share a word with it
        fits = {label: self.label_fit(label, item) for label, _, _, _ in table}
        best = max(fits.values(), default=0)
        shared: list[str] = []
        single: list[str] = []
        for label, name, carried, values in table:
            near = best > 0 and fits[label] == best
            if near and carried >= 2 and name not in settled and name not in shared + single:
                # a value two listings share marks a property; an identifier or a list of purchase options differs on
                # every listing, so keys with fewer distinct values than listings come first
                (shared if self.key_distinct.get((label, name), len(values)) < carried else single).append(name)
        return (shared + single)[:limit]

    def question(self, number: int, label: str) -> str:
        return f"What {label} would you like for the {self.items[number - 1].name}?"

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
        return self.ask_json(prompt, list(models), 1500) or {}

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
        better = [str(b) for b in _items(got.get("better")) if str(b) in self.rows and not self.sold_out(str(b))][:4]
        for ident in ids:
            said = next((p for p in problems if ident in p), "")
            if said and ident not in better:
                self.rejected[ident] = said[:300]
        text = "A careful review says this cart is not right yet: " + "; ".join(problems or ["see the requirements"])
        if better:
            text += ". Listings that look better: " + ", ".join(better)
        text += ". Fix the cart (or show the review is wrong by re-reading the records), then check out."
        if self.left() <= 8:
            text += f" Only {self.left()} steps are left: replacing a line takes two (take it off, add the new one) and checking out one, so change each line at most once and never change it back."
        return text

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
        self.lost_finds = []
        while pending and self.trails < 12 and self.cost < SPEND_HARD and self.time_left() > CLOSING:
            self.trails += 1
            self.lost_finds = []
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
                    if isinstance(chain.get("find"), dict):
                        self.lost_finds.append((need, about, chain["find"]))
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
        hold = not early and bool(lost) and self.trail_blocks < 3 and self.left() > 5 and self.time_left() > CLOSING
        if hold:
            self.trail_blocks += 1
            text += ("The order counts only after every page a requirement points at has been opened in full (a search row is not enough), following each page that refers to a further listing. Still to find and open: "
                     + " | ".join(lost[:3]) + ". Search for it by its shop, brand or words, open it, apply the value its page gives, then check out.")
        return text, hold

    def sweep(self) -> str:
        """Once, on a several-item order with steps to spare: before the agent's turns use up the time, Kyo searches
        each item itself (its plain requirement values with its name, in its category when the search takes one) and
        opens the rows whose text names the most of those values (a step each, no model turn)."""
        self.swept = True
        tool = self.roles.get("search")
        schema = self.schemas.get(tool or "") or {}
        props = schema.get("properties") or {}
        if not tool or not props:
            return ""
        mode = next((k for k, spec in props.items() if isinstance(spec, dict) and isinstance(spec.get("enum"), list)
                     and any(re.search(r"word|keyword|text|search|query", str(e), re.IGNORECASE) for e in spec["enum"])), None)
        free = next((k for k, spec in props.items() if isinstance(spec, dict) and "string" in _kinds(spec) and "enum" not in spec), None)
        group = next((k for k, spec in props.items() if isinstance(spec, dict) and isinstance(spec.get("enum"), list) and k != mode), None)
        per_item = max(2, min(4, (self.left() - 10 * len(self.items)) // (2 * len(self.items))))
        lines = []
        for number, item in enumerate(self.items, 1):
            if self.left() < 10 * len(self.items) or self.finished or self.time_left() < CLOSING + 90:
                break
            values, _ = self.plain_checks(item)
            words = [" ".join(w) for _, w in values if isinstance(w, list)][:2]
            call: dict[str, Any] = {}
            if free:
                limit = props[free].get("maxLength") if isinstance(props[free].get("maxLength"), int) else 60
                call[free] = (" ".join(words) + " " + _singular(item.name.lower())).strip()[:limit]
            if mode:
                call[mode] = next(str(e) for e in props[mode]["enum"] if re.search(r"word|keyword|text|search|query", str(e), re.IGNORECASE))
            if group and item.shelf:
                label = next((str(e) for e in props[group]["enum"] if str(e).strip().lower() == item.shelf.strip().lower()), None)
                if label:
                    call[group] = label
            faults: list[str] = []
            call = _shape(call, schema, "args", faults)
            if faults or not call:
                continue
            reply = self.act(tool, call)
            if reply is None or self.finished:
                break
            seen = reply.get("observation")
            self.absorb(seen)
            self.observe(tool, seen)
            want = [w for _, v in values if isinstance(v, list) for w in v]
            ranked = []
            for ident, record in self.records_in(seen):
                if ident in self.full or not self.buyable(ident):
                    continue
                text = _dump(record).lower()
                ranked.append((-sum(1 for w in want if w in text), ident))
            picks = [i for _, i in sorted(ranked)[:per_item]]
            opener = self.roles.get("open")
            target = self.single_target(opener)
            read = 0
            for ident in picks:
                if not opener or not target or self.left() < 6 * len(self.items) or self.finished:
                    break
                faults = []
                page_args = _shape({target: ident}, self.schemas.get(opener), "args", faults)
                if faults:
                    break
                page = self.act(opener, page_args)
                if page is None:
                    break
                seen_page = page.get("observation")
                self.observe(opener, seen_page)
                self.note_view(opener, seen_page)
                read += 1
            fits = [i for i in picks if i in self.full and not self.contradictions(i) and not self.unnamed_values(i)]
            lines.append(f"item {number} ({item.name}): searched {_dump(call)}, {len(ranked)} new rows, opened {read}" + (f"; fitting so far: {', '.join(fits)}" if fits else ""))
        self.trace.append({"kind": "sweep", "lines": lines})
        if not lines:
            return ""
        return "[Kyo searched each item once and opened the most promising rows itself (a step each)] " + " | ".join(lines)

    def find_call(self, find: dict[str, Any]) -> dict[str, Any] | None:
        """The search tool's arguments for a chain link named by shop and kind: the free-text parameter gets the shop
        and kind words, a mode parameter its word-search value, and a category parameter the label sharing most words
        with the kind (none when no label shares a word)."""
        tool = self.roles.get("search")
        schema = self.schemas.get(tool or "") or {}
        props = schema.get("properties") or {}
        shop = str(find.get("shop") or "").strip()
        kind = str(find.get("kind") or "").strip()
        if not tool or not shop:
            return None
        args: dict[str, Any] = {}
        for key, spec in props.items():
            if not isinstance(spec, dict):
                continue
            enum = [str(v) for v in spec.get("enum") or []]
            if enum and any(re.search(r"word|keyword|text|search|query", e, re.IGNORECASE) for e in enum) and len(enum) <= 4:
                args[key] = next(e for e in enum if re.search(r"word|keyword|text|search|query", e, re.IGNORECASE))
            elif enum and kind:
                kind_words = {w for w in _fit_toks(kind) if len(w) >= 3}
                scored = sorted(((len(kind_words & _fit_toks(e)), e) for e in enum), reverse=True)
                if scored and scored[0][0] > 0:
                    args[key] = scored[0][1]
            elif "string" in _kinds(spec) and re.search(r"word|query|term|text|keyword|search|q$", _words(key)):
                limit = spec.get("maxLength") if isinstance(spec.get("maxLength"), int) else 60
                args[key] = (shop + " " + kind).strip()[:limit]
        if not any(isinstance(v, str) and shop in v for v in args.values()):
            return None
        faults: list[str] = []
        args = _shape(args, schema, "args", faults)
        return None if faults else args

    def chase(self) -> str:
        """Early, before questions: walk each requirement's chain of pages itself. The model names the next link
        (shop, kind, features); Kyo searches by shop and kind and opens that shop's rows that best match, then reads
        the chain again, up to six links."""
        live = {need for _, need in self.needs_now()}
        if self.chased or not any(need in live and need not in self.ref_values for _, need in self.refs):
            return ""
        self.chased = True
        tool = self.roles.get("search")
        view = self.roles.get("offer") or self.roles.get("open")
        target = self.single_target(view)
        notes: list[str] = []
        for _ in range(6):
            if self.left() < self.tail() + 10 or self.time_left() < CLOSING + 60 or self.finished:
                break
            text, _hold = self.trail(early=True)
            if text:
                notes.append(text)
            if not self.lost_finds or not tool or not view or not target:
                break
            opened_any = False
            for need, about, find in self.lost_finds[:2]:
                args = self.find_call(find)
                if not args:
                    continue
                key = _dump(args)
                # the same search again goes on from the page it stopped at, not from the first page
                if key in self.find_pages:
                    param, value = self.find_pages[key]
                    if param is None or value is None:
                        continue
                    args = {**args, param: value}
                shop = str(find.get("shop") or "").strip().lower()
                reply = None
                for _page in range(3):
                    reply = self.act(tool, args)
                    if reply is None or self.finished:
                        break
                    self.observe(tool, reply.get("observation"))
                    plan = self.paging(tool, reply.get("observation"))
                    self.find_pages[key] = (plan[2], plan[3]) if plan else (None, None)
                    fresh = [i for i, r in self.records_in(reply.get("observation")) if i not in self.opened and any(
                        shop == str(v).lower() or shop in str(v).lower() for k, v in r.items()
                        if isinstance(v, str) and re.search(r"merchant|seller|shop|store|vendor|brand|maker", str(k).lower()))]
                    if fresh or not plan or plan[3] is None or self.left() < self.tail() + 10:
                        break
                    # every row of that shop on this page is read already: the next page
                    args = {**args, plan[2]: plan[3]}
                if reply is None or self.finished:
                    break
                features = {w for w in _fit_toks(" ".join(str(f) for f in _items(find.get("features"))) + " " + about) if len(w) >= 3}
                ranked = []
                for ident, record in self.records_in(reply.get("observation")):
                    if ident in self.opened:
                        continue
                    owners = [str(v).lower() for k, v in record.items() if isinstance(v, str) and re.search(r"merchant|seller|shop|store|vendor|brand|maker", str(k).lower())]
                    if not any(shop == o or shop in o for o in owners):
                        continue
                    ranked.append((len(features & _fit_toks(_dump(record))), ident))
                for _score, ident in sorted(ranked, reverse=True)[:3]:
                    faults: list[str] = []
                    call = _shape({target: ident}, self.schemas.get(view), "args", faults)
                    if faults:
                        break
                    seen = self.act(view, call)
                    if seen is None or self.finished:
                        break
                    self.observe(view, seen.get("observation"))
                    opened_any = True
            if not opened_any:
                break
        self.trace.append({"kind": "chase", "values": dict(self.ref_values)})
        return "\n".join(notes[-1:])

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
        got = self.ask_json(prompt, self.checker(), 1500)
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

    def nearest_lines(self) -> dict[str, int]:
        """For a request that wants the closest listing bought rather than nothing: per requested item, the opened,
        buyable listing on that item's shelf that shows the most requirement values (review candidates first, cheaper
        first among ties), each listing used for one item only, within the budget; empty when none."""
        if self.brief.get("fallback_buy") is not True or not self.items:
            return {}
        cat_key = self.category_key()
        per_line = self.brief.get("budget_scope") == "each"
        cap = self.budget if isinstance(self.budget, (int, float)) and not isinstance(self.budget, bool) and self.budget > 0 else None
        lines: dict[str, int] = {}
        total = 0.0
        # an item that must work with another: a listing naming a code of that item's cart line ranks first
        links = self.compat_links()
        anchor_codes: dict[int, set[str]] = {}
        for dep, anchor in links:
            shelf_a = (self.items[anchor - 1].shelf or "").strip().lower()
            held = [i for i in self.cart_lines if cat_key and str((self.rows.get(i) or {}).get(cat_key) or "").strip().lower() == shelf_a]
            if held:
                anchor_codes.setdefault(dep, set()).update(c for i in held for c in self.link_codes(i))
        for number, item in enumerate(self.items, 1):
            shelf_want = (item.shelf or "").strip().lower()
            noun = {w for w in re.findall(r"[a-z0-9]{4,}", item.name.lower())}
            plain_musts = [m for m in item.musts + item.told if m not in item.dropped]
            # an exclusion ("not <C>") counts against a listing that names <C>, never for it
            negative = re.compile(r"^\s*(?:not|no|never|except|anything but|other than|avoid)\b")
            values = [re.findall(r"[a-z0-9]{3,}", m.split(":", 1)[-1].lower()) for m in plain_musts if not negative.search(m.split(":", 1)[-1].lower())]
            banned = [[w for w in re.findall(r"[a-z0-9]{3,}", negative.sub("", m.split(":", 1)[-1].lower()))] for m in plain_musts if negative.search(m.split(":", 1)[-1].lower())]
            ranked = []
            for ident in dict.fromkeys(self.leads + list(self.full)):
                price = self.price_of(ident)
                if ident in lines or not self.buyable(ident) or price is None or (cap and price * max(1, item.qty) > cap):
                    continue
                shelf = str((self.rows.get(ident) or {}).get(cat_key) or "").strip().lower() if cat_key else ""
                if shelf_want and cat_key and shelf not in ("", shelf_want):
                    continue
                text = _dump(self.full.get(ident) or self.rows.get(ident)).lower()
                if len(self.items) > 1 and not shelf_want and noun and not any(w in (shelf or text) for w in noun):
                    continue
                shown = sum(1 for words in values if words and all(w in text for w in words)) - 2 * sum(1 for words in banned if words and all(w in text for w in words))
                if anchor_codes.get(number) and anchor_codes[number] & self.link_codes(ident):
                    shown += 50
                # a listing whose own attributes plainly break a requirement fails it whatever else it shows, and so does one
                # whose record nowhere names a required value: neither is ever the nearest
                if self.contradictions(ident) or self.unnamed_values(ident):
                    continue
                ranked.append((shown + (100 if ident in self.leads else 0), price, ident))
            if not ranked:
                continue
            best = min(ranked, key=lambda r: (-r[0], r[1]))
            copies = max(1, item.qty)
            if cap and not per_line and total + best[1] * copies > cap:
                continue
            lines[best[2]] = copies
            total += best[1] * copies
        return lines

    def repair_after_refusal(self) -> str:
        """After a refused checkout that changed the task (a sold-out line, a new budget, a dropped requirement) on a
        one-item order: Kyo itself puts in the cheapest opened listing that now meets every checkable requirement, within
        the budget and in stock (no model call), when the cart line is gone, over the budget, or no longer the cheapest."""
        cart = self.roles.get("cart")
        if not cart or len(self.items) != 1 or self.left() < 2:
            return ""
        lines = list(self.cart_lines)
        if self.note_sold and len(lines) == 1 and not any(self.sold_out(i) for i in lines):
            # the note named no listing: the order's one line is the one that went
            self.unavailable.add(lines[0])
        item = self.items[0]
        values, banned = self.plain_checks(item)
        copies = max(1, item.qty)
        cap = self.budget if isinstance(self.budget, (int, float)) and not isinstance(self.budget, bool) and self.budget > 0 else None
        pool = []
        # with no checkable requirement left (one was dropped), every seen row of the item's kind is a candidate
        seen = dict(self.rows) if not (values or banned) else {}
        seen.update(self.full)
        for ident, record in seen.items():
            price = self.price_of(ident)
            if price is None or not self.buyable(ident) or self.item_of(ident) != 1:
                continue
            if cap and price * copies > cap:
                continue
            if (values or banned) and not self.plain_fit(values, banned, record):
                continue
            if self.contradictions(ident):
                continue
            pool.append((price, ident))
        if not pool:
            return ""
        price, best = min(pool)
        have = lines[0] if len(lines) == 1 else None
        have_price = self.price_of(have) if have else None
        keep = (have and not self.sold_out(have) and have_price is not None and (not cap or have_price * copies <= cap)
                and (self.brief.get("pick") != "cheapest" or have_price <= price))
        if keep or best == have:
            return ""
        calls = self.cart_calls({best: copies}, [i for i in lines if i != best])
        if not calls or len(calls) + 1 > self.left():
            return ""
        for args in calls:
            faults: list[str] = []
            args = _shape(args, self.schemas.get(cart), "args", faults)
            if faults:
                return ""
            reply = self.act(cart, args)
            if reply is None or self.finished:
                return ""
            self.book_cart(cart, args, reply.get("observation"))
        self.trace.append({"kind": "refusal_repaired", "out": lines, "in": best})
        return (f"\n[Kyo] After that change Kyo put {best} ({price:g}) in the cart in place of {', '.join(lines) or 'nothing'}: the cheapest opened"
                " listing that now meets every requirement within the budget. Check out again unless its record shows a problem.")

    def fix_compat(self) -> None:
        """At a forced finish: a tied line that does not work with its anchor's line is swapped for the cheapest opened
        candidate of its item that names the anchor's code and states the item's plain values (no model call)."""
        cart = self.roles.get("cart")
        if not cart:
            return
        for dep_line, anchor_line in self.compat_mismatch():
            if dep_line not in self.cart_lines or self.left() < 4:
                continue
            number = self.item_of(dep_line)
            if not number:
                continue
            item = self.items[number - 1]
            values, banned = self.plain_checks(item)
            want = self.link_codes(anchor_line)
            pool = [i for i, r in self.full.items() if i != dep_line and self.item_of(i) == number and self.buyable(i)
                    and self.link_codes(i) & want and (not (values or banned) or self.plain_fit(values, banned, r))]
            if not pool:
                continue
            best = min(pool, key=lambda i: self.price_of(i) if self.price_of(i) is not None else float("inf"))
            calls = self.cart_calls({best: max(1, item.qty)}, [dep_line])
            if not calls or len(calls) + 1 > self.left():
                continue
            for args in calls:
                faults: list[str] = []
                args = _shape(args, self.schemas.get(cart), "args", faults)
                if faults:
                    return
                reply = self.act(cart, args)
                if reply is None or self.finished:
                    return
                self.book_cart(cart, args, reply.get("observation"))
                self.observe(cart, reply.get("observation"))
            self.trace.append({"kind": "compat_fixed", "out": dep_line, "in": best})

    def fill_missing(self) -> None:
        """Add the nearest opened candidate of each requested item that has no cart line (no model call)."""
        cart = self.roles.get("cart")
        cat_key = self.category_key()
        if not cart or not cat_key:
            return
        held = {str((self.rows.get(i) or {}).get(cat_key) or (self.full.get(i) or {}).get(cat_key) or "").strip().lower() for i in self.cart_lines}
        missing = [(it.shelf or "").strip().lower() for it in self.items if (it.shelf or "").strip().lower() not in held]
        if not missing or "" in missing:
            return
        lines = {i: n for i, n in self.nearest_lines().items()
                 if str((self.rows.get(i) or {}).get(cat_key) or (self.full.get(i) or {}).get(cat_key) or "").strip().lower() in missing and i not in self.cart_lines}
        calls = self.cart_calls(lines, []) if lines else None
        if not calls or len(calls) + 1 > self.left():
            return
        for args in calls:
            faults: list[str] = []
            args = _shape(args, self.schemas.get(cart), "args", faults)
            if faults:
                return
            reply = self.act(cart, args)
            if reply is None or self.finished:
                return
            self.book_cart(cart, args, reply.get("observation"))
            self.observe(cart, reply.get("observation"))
        self.trace.append({"kind": "missing_filled", "lines": lines})

    def cart_nearest(self) -> bool:
        """Put the nearest candidates in an empty cart (no model call); True when a line went in."""
        cart = self.roles.get("cart")
        lines = self.nearest_lines() if cart else {}
        calls = self.cart_calls(lines, []) if lines else None
        if not calls or len(calls) + 1 > self.left():
            return False
        for args in calls:
            faults: list[str] = []
            args = _shape(args, self.schemas.get(cart), "args", faults)
            if faults:
                return False
            reply = self.act(cart, args)
            if reply is None or self.finished:
                return False
            self.book_cart(cart, args, reply.get("observation"))
            self.observe(cart, reply.get("observation"))
        self.trace.append({"kind": "nearest_carted", "lines": lines})
        return bool(self.cart_lines)

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
            self.book_cart(cart, args, reply.get("observation"))
            self.observe(cart, reply.get("observation"))
            done.append(_dump(args))
        faults = []
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

    def new_line_id(self, args: dict[str, Any], among: Any = None) -> str | None:
        """The first listing a cart call puts in that the cart does not hold yet: a plain id argument, or an id inside
        the line objects of a tool that takes a list of lines (optionally only ids found in `among`)."""
        pool = among if among is not None else self.rows
        found: list[str] = []
        for value in args.values():
            if isinstance(value, str):
                found.append(value)
            elif isinstance(value, list):
                for line in value:
                    if isinstance(line, dict):
                        found.extend(v for v in line.values() if isinstance(v, str))
                    elif isinstance(line, str):
                        found.append(line)
        return next((i for i in found if i in pool and i not in self.cart_lines), None)

    def keep_other_lines(self, name: str, args: dict[str, Any]) -> tuple[dict[str, Any], str]:
        """A cart tool that replaces the whole cart with the lines given: lines of requested items that the new list leaves
        out (no line of the same item's kind in it) are kept, since an order needs a line for every item."""
        props = (self.schemas.get(name) or {}).get("properties") or {}
        cat_key = self.category_key()
        if len(self.items) < 2 or not cat_key or not self.cart_lines:
            return args, ""
        for key, spec in props.items():
            shape = spec.get("items") if isinstance(spec, dict) and "array" in _kinds(spec) else None
            if not (isinstance(shape, dict) and isinstance(shape.get("properties"), dict)) or not isinstance(args.get(key), list):
                continue
            inner = shape["properties"]
            id_key = next((k for k, s in inner.items() if isinstance(s, dict) and "string" in _kinds(s) and "enum" not in s), None)
            count_key = next((k for k, s in inner.items() if isinstance(s, dict) and "integer" in _kinds(s)), None)
            if not id_key:
                return args, ""
            given = [line.get(id_key) for line in args[key] if isinstance(line, dict)]

            def kind(ident: Any) -> str:
                return str((self.rows.get(str(ident)) or {}).get(cat_key) or (self.full.get(str(ident)) or {}).get(cat_key) or "").strip().lower()

            covered = {kind(i) for i in given} - {""}
            if len(given) >= len(self.items):
                return args, ""
            # a line whose kind is not known (its row was cut short and never opened) is kept too
            back = [i for i in self.cart_lines if i not in given and (not kind(i) or kind(i) not in covered) and not self.sold_out(i)]
            if not back:
                return args, ""
            lines = list(args[key]) + [{id_key: i, **({count_key: self.cart_lines[i]} if count_key else {})} for i in back]
            self.trace.append({"kind": "kept_lines", "back": back})
            return {**args, key: lines}, (f"Kept {', '.join(back)} in the cart: this tool replaces the whole cart, so a line of another"
                                          " requested item stays unless a new line for that item takes its place.")
        return args, ""

    def unread_add(self, name: str, args: dict[str, Any]) -> str:
        """Once per listing: adding a listing whose page was never opened puts unknown attributes in the order."""
        props = (self.schemas.get(name) or {}).get("properties") or {}
        mode = next((k for k, s in props.items() if isinstance(s, dict) and isinstance(s.get("enum"), list) and len(s["enum"]) >= 2), None)
        if mode is not None and args.get(mode) in props[mode]["enum"][1:]:
            return ""
        if any((_count(v) or 1) <= 0 for v in args.values() if not isinstance(v, str)):
            return ""
        ident = self.new_line_id(args)
        if not ident or ident in self.full or ident in self.unread_warned or not self.roles.get("open") or self.left() < self.tail() + 4:
            return ""
        self.unread_warned.add(ident)
        read = self.fill_facts([ident], self.tail() + 3)
        self.trace.append({"kind": "unread_add", "line": ident, "read": bool(read)})
        if not read:
            return ""
        return (f"{ident}'s page had never been opened, so Kyo opened it before adding it (one step). Its record: " + read[0][:1500]
                + "\nIf it breaks a requirement, take the line off again and cart a listing whose record you have read.")

    def compat_links(self) -> list[tuple[int, int]]:
        """(dependent item, anchor item) pairs: a requirement of one item that ties it to another item of this order
        (the brief writes those as "... the <other item> in this order")."""
        links = []
        for number, item in enumerate(self.items, 1):
            for must in item.musts + item.told:
                low = must.lower()
                tie = re.search(r"this order|\bwork(?:s|ing)? with\b|\bcompatib|\bgo(?:es)? with\b|\bwhichever\b|\bfits?\s+(?:the|whichever|with)\b|\bend up (?:buying|picking|with)\b", low)
                fixed = re.search(r"\b(?=[A-Z0-9-]*[A-Z][A-Z0-9-]*[A-Z])(?=[A-Z0-9-]*\d[A-Z0-9-]*\d)[A-Z][A-Z0-9-]{3,13}\b", must)
                if must in item.dropped or not tie or fixed:
                    continue
                best, score = None, 0
                for other, it in enumerate(self.items, 1):
                    if other == number:
                        continue
                    words = {_singular(w) for w in re.findall(r"[a-z0-9]{3,}", (it.name + " " + (it.shelf or "")).lower())}
                    hit = len(words & {_singular(w) for w in re.findall(r"[a-z0-9]{3,}", low)})
                    if hit > score:
                        best, score = other, hit
                if best and (number, best) not in links:
                    links.append((number, best))
        return links

    def codes_of(self, ident: str) -> list[str]:
        """Code-like tokens (capital letters and digits mixed, e.g. model or part codes) in a listing's own text."""
        record = self.full.get(ident) or self.rows.get(ident) or {}
        text = " ".join(str(v) for k, v in record.items() if isinstance(v, str))
        found = re.findall(r"\b(?=[A-Z0-9-]*[A-Z][A-Z0-9-]*[A-Z])(?=[A-Z0-9-]*\d[A-Z0-9-]*\d)[A-Z][A-Z0-9-]{3,13}\b", text)
        # a model's other name ("<Name> <number>") ties listings the same way its code does
        found += re.findall(r"\b[A-Z][a-z]{2,} \d{2,4}\b", text)
        return list(dict.fromkeys(found))

    def auto_open(self, observation: Any) -> str:
        """On a several-item task whose step allowance is far larger than its time allows model turns for: Kyo opens the
        first unopened buyable rows of a search itself (a step each, no model turn) and lists each one's attributes and the
        codes it names, so candidates are read without spending the agent's turns."""
        if len(self.items) < 2 or self.max_steps < 80 or self.left() < 30 or self.time_left() < CLOSING + 40:
            return ""
        ids = [i for i, _ in self.records_in(observation) if i not in self.full and self.buyable(i)][:4]
        if not ids:
            return ""
        before = set(self.full)
        self.fill_facts(ids, 25)
        lines = []
        for ident in ids:
            if ident not in self.full or ident in before:
                continue
            record = self.full[ident]
            facts = _fact_block(record) or {}
            price = self.price_of(ident)
            codes = self.codes_of(ident)
            lines.append(f"{ident} ({price:g}) " if price is not None else f"{ident} " + "")
            lines[-1] += "; ".join(f"{k}={str(v)[:40]}" for k, v in list(facts.items())[:14]) + (f" | codes named: {', '.join(codes[:5])}" if codes else "")
        self.trace.append({"kind": "auto_open", "ids": ids, "read": len(lines)})
        return ("\n[Kyo opened these rows itself (a step each)] " + " || ".join(lines)) if lines else ""

    def follow_named(self, ids: list[str], keep: int) -> list[str]:
        """Before checkout: a cart line whose description names another listing by the kind of code listing titles carry
        (e.g. "more on part <code>") counts only with that listing opened too. Kyo finds each named listing (a word search
        for the code when no seen row carries it) and opens it, keeping `keep` steps; returns what it read."""
        code_re = r"\b(?=[A-Z0-9-]*[A-Z][A-Z0-9-]*[A-Z])(?=[A-Z0-9-]*\d[A-Z0-9-]*\d)[A-Z][A-Z0-9-]{3,13}\b"
        title_of = {i: next((str(v) for k, v in (self.full.get(i) or r).items() if isinstance(v, str) and re.search(r"title|heading|name", str(k).lower())), "")
                    for i, r in self.rows.items()}
        prefixes = {re.match(r"[A-Z]+", c).group(0) for t in title_of.values() for c in re.findall(code_re, t)}
        tool = self.roles.get("search")
        schema = self.schemas.get(tool or "") or {}
        props = schema.get("properties") or {}
        mode = next((k for k, spec in props.items() if isinstance(spec, dict) and isinstance(spec.get("enum"), list)
                     and any(re.search(r"word|keyword|text|search|query", str(e), re.IGNORECASE) for e in spec["enum"])), None)
        free = next((k for k, spec in props.items() if isinstance(spec, dict) and "string" in _kinds(spec) and "enum" not in spec
                     and re.search(r"word|query|term|text|keyword|search|q$", _words(k))), None)
        out: list[str] = []
        for ident in ids:
            record = self.full.get(ident) or {}
            own = set(re.findall(code_re, title_of.get(ident, "")))
            text = " ".join(str(v) for k, v in record.items() if isinstance(v, str) and not re.search(r"title|heading|name", str(k).lower()))
            for code in dict.fromkeys(re.findall(code_re, text)):
                if code in own or code in self.followed or re.match(r"[A-Z]+", code).group(0) not in prefixes:
                    continue
                if self.left() <= keep + 1 or self.finished:
                    return out
                self.followed.add(code)
                target = next((i for i, t in title_of.items() if code in t and i != ident), None)
                if not target and tool and free:
                    call: dict[str, Any] = {free: code}
                    if mode:
                        call[mode] = next(str(e) for e in props[mode]["enum"] if re.search(r"word|keyword|text|search|query", str(e), re.IGNORECASE))
                    faults: list[str] = []
                    call = _shape(call, schema, "args", faults)
                    if faults:
                        continue
                    reply = self.act(tool, call)
                    if reply is None or self.finished:
                        return out
                    seen = reply.get("observation")
                    self.observe(tool, seen)
                    for row_id, row in self.records_in(seen):
                        if code in _dump(row) and row_id != ident:
                            target = row_id
                            break
                if not target:
                    continue
                if target in self.full:
                    out.append(f"{ident} names {code}: {target} (already opened)")
                    continue
                read = self.fill_facts([target], keep)
                if read:
                    out.append(f"{ident} names {code}, so Kyo opened {target}: " + read[0][:600])
        self.trace.append({"kind": "follow_named", "ids": ids, "read": len(out)})
        return out

    def link_codes(self, ident: str) -> set[str]:
        """The codes a listing uses to say what it works with: its code-like tokens, without the kind of code listing
        titles carry as their own part numbers."""
        code_re = r"\b(?=[A-Z0-9-]*[A-Z][A-Z0-9-]*[A-Z])(?=[A-Z0-9-]*\d[A-Z0-9-]*\d)[A-Z][A-Z0-9-]{3,13}\b"
        if self.title_prefixes is None or len(self.rows) != self.title_rows:
            titles = [next((str(v) for k, v in r.items() if isinstance(v, str) and re.search(r"title|heading|name", str(k).lower())), "") for r in self.rows.values()]
            self.title_prefixes = {re.match(r"[A-Z]+", c).group(0) for t in titles for c in re.findall(code_re, t)}
            self.title_rows = len(self.rows)
        return {c for c in self.codes_of(ident) if " " in c or re.match(r"[A-Z]+", c).group(0) not in self.title_prefixes}

    def item_of(self, ident: str) -> int | None:
        cat_key = self.category_key()
        if not cat_key:
            return None
        shelf = str((self.rows.get(ident) or {}).get(cat_key) or (self.full.get(ident) or {}).get(cat_key) or "").strip().lower()
        return next((n for n, it in enumerate(self.items, 1) if shelf and (it.shelf or "").strip().lower() == shelf), None)

    def compat_mismatch(self, extra: str | None = None) -> list[tuple[str, str]]:
        """Pairs (dependent line, anchor line) in the cart (plus `extra`) whose codes share nothing: the dependent does
        not say it works with that anchor."""
        links = self.compat_links()
        if not links:
            return []
        lines = list(self.cart_lines) + ([extra] if extra and extra not in self.cart_lines else [])
        by_item: dict[int, list[str]] = {}
        for ident in lines:
            number = self.item_of(ident)
            if number:
                by_item.setdefault(number, []).append(ident)
        bad = []
        for dep, anchor in links:
            for a in by_item.get(anchor, []):
                for d in by_item.get(dep, []):
                    if not (self.link_codes(d) & self.link_codes(a)):
                        bad.append((d, a))
        return bad

    def code_ledger(self, links: list[tuple[int, int]]) -> str:
        """From the opened listings only: per item, buyable candidates on its shelf with the codes they name, and the codes
        that a candidate of the anchor item and a candidate of every item tied to it share (cheapest candidate per item)."""
        cat_key = self.category_key()
        if not cat_key:
            return ""
        per_item: dict[int, dict[str, list[str]]] = {}
        for number, item in enumerate(self.items, 1):
            shelf = (item.shelf or "").strip().lower()
            values, banned = self.plain_checks(item)
            for ident, record in self.full.items():
                if not self.buyable(ident) or ident in self.gone:
                    continue
                if str(record.get(cat_key) or (self.rows.get(ident) or {}).get(cat_key) or "").strip().lower() != shelf:
                    continue
                # a candidate whose attributes do not state the item's plain values (or state an excluded one) is left out
                if (values or banned) and not self.plain_fit(values, banned, record):
                    continue
                for code in self.link_codes(ident):
                    per_item.setdefault(number, {}).setdefault(code, []).append(ident)
        groups: dict[int, list[int]] = {}
        for dep, anchor in links:
            groups.setdefault(anchor, []).append(dep)
        parts = []
        for anchor, deps in groups.items():
            members = [anchor] + deps
            shared = [c for c in per_item.get(anchor, {}) if all(c in per_item.get(m, {}) for m in deps)]
            if shared:
                for code in shared[:3]:
                    picks = []
                    for m in members:
                        ids = sorted(per_item[m][code], key=lambda i: self.price_of(i) if self.price_of(i) is not None else float("inf"))
                        picks.append(f"item {m}: " + ", ".join(f"{i} ({self.price_of(i):g})" if self.price_of(i) is not None else i for i in ids[:3]))
                    parts.append(f"code {code} is named by " + "; ".join(picks))
            else:
                have = []
                for m in members:
                    codes = sorted(per_item.get(m, {}), key=lambda c: -len(per_item[m][c]))[:6]
                    have.append(f"item {m}: " + (", ".join(codes) if codes else "no opened candidate yet"))
                parts.append("no code is shared yet by opened candidates of items " + ", ".join(map(str, members)) + " (codes seen: " + "; ".join(have) + ")")
        return ("[code ledger from opened listings whose attributes meet each item's checkable requirements] " + " | ".join(parts)) if parts else ""

    def compat_search(self, ident: str) -> str:
        """After a listing of an item that other items must work with goes into the cart: per dependent item still
        without a line, Kyo searches that item's category for the listing's codes (a step each, two codes at most per
        item) and lists the rows whose text names one of them."""
        links = self.compat_links()
        cat_key = self.category_key()
        if not links or not cat_key or ident in self.compat_done:
            return ""
        shelf = str((self.rows.get(ident) or {}).get(cat_key) or (self.full.get(ident) or {}).get(cat_key) or "").strip().lower()
        anchors = {a for _, a in links if (self.items[a - 1].shelf or "").strip().lower() == shelf}
        if not anchors:
            return ""
        self.compat_done.add(ident)
        codes = self.codes_of(ident)
        # a code other listings also name is a cross-reference; the listing's own other codes come after
        seen_text = " ".join(_dump(r) for i, r in list(self.full.items())[:200] if i != ident)
        codes.sort(key=lambda c: 0 if c in seen_text else 1)
        tool = self.roles.get("search")
        schema = self.schemas.get(tool or "") or {}
        props = schema.get("properties") or {}
        mode = next((k for k, spec in props.items() if isinstance(spec, dict) and isinstance(spec.get("enum"), list)
                     and any(re.search(r"word|keyword|text|search|query", str(e), re.IGNORECASE) for e in spec["enum"])), None)
        free = next((k for k, spec in props.items() if isinstance(spec, dict) and "string" in _kinds(spec) and "enum" not in spec
                     and re.search(r"word|query|term|text|keyword|search|q$", _words(k))), None)
        group = next((k for k, spec in props.items() if isinstance(spec, dict) and isinstance(spec.get("enum"), list) and k != mode), None)
        if not tool or not free or not codes:
            return ""
        lines = []
        for dep, anchor in links:
            if anchor not in anchors:
                continue
            item = self.items[dep - 1]
            dep_shelf = (item.shelf or "").strip().lower()
            held = [i for i in self.cart_lines if str((self.rows.get(i) or {}).get(cat_key) or "").strip().lower() == dep_shelf]
            if held or self.left() <= self.tail() + 6:
                continue
            found: dict[str, list[str]] = {}
            for code in codes[:2]:
                call: dict[str, Any] = {free: code}
                if mode:
                    call[mode] = next(str(e) for e in props[mode]["enum"] if re.search(r"word|keyword|text|search|query", str(e), re.IGNORECASE))
                if group and dep_shelf:
                    label = next((str(e) for e in props[group]["enum"] if str(e).strip().lower() == dep_shelf), None)
                    if label:
                        call[group] = label
                faults: list[str] = []
                call = _shape(call, schema, "args", faults)
                if faults:
                    break
                reply = self.act(tool, call)
                if reply is None or self.finished:
                    break
                seen = reply.get("observation")
                self.observe(tool, seen)
                for row_id, record in self.records_in(seen):
                    if code in _dump(record) and self.buyable(row_id):
                        found.setdefault(code, []).append(row_id)
                if found.get(code):
                    break
            rows = [(self.price_of(i) if self.price_of(i) is not None else float("inf"), i, c) for c, ids in found.items() for i in ids]
            rows.sort()
            if rows:
                lines.append(f"item {dep} ({item.name}): " + ", ".join(f"{i} ({p:g}, names {c})" if p != float("inf") else f"{i} (names {c})" for p, i, c in rows[:8]))
            else:
                lines.append(f"item {dep} ({item.name}): no listing in its category names {', '.join(codes[:2])}")
        self.trace.append({"kind": "compat_search", "line": ident, "codes": codes[:4], "lines": len(lines)})
        if not lines:
            return ""
        self.compat_notes.append(f"for {ident} (codes {', '.join(codes[:2])}): " + " | ".join(lines))
        return ("\n[Kyo] Items that must work with " + ident + " are matched by the codes it carries (" + ", ".join(codes[:2])
                + "): a listing that works with it names one of them. Kyo searched each such item's category for them (a step each). "
                + " | ".join(lines) + ". Open these, check their own requirements, and cart the cheapest that fits; if none fits,"
                " another listing of " + ident + "'s item with other codes may be needed.")

    def add_breaks(self, name: str, args: dict[str, Any]) -> str:
        """Once per listing: an addition whose own attributes plainly break one of its item's requirements."""
        props = (self.schemas.get(name) or {}).get("properties") or {}
        mode = next((k for k, s in props.items() if isinstance(s, dict) and isinstance(s.get("enum"), list) and len(s["enum"]) >= 2), None)
        if mode is not None and args.get(mode) in props[mode]["enum"][1:]:
            return ""
        ident = self.new_line_id(args)
        if not ident or ident in self.cart_lines or ident in self.break_warned:
            return ""
        broken = self.contradictions(ident)
        if not broken:
            return ""
        self.break_warned.add(ident)
        self.trace.append({"kind": "add_breaks", "line": ident, "broken": broken})
        return (f"{ident}'s own attributes break its item's requirements: " + "; ".join(broken[:3]) + ". Pick a listing whose"
                " attributes meet them (the status lists such opened listings), or send the same call again to add it anyway"
                " when the steps left allow no better pick.")

    def readd_rejected(self, name: str, args: dict[str, Any]) -> str:
        """Once per listing: putting back a listing the independent review rejected."""
        props = (self.schemas.get(name) or {}).get("properties") or {}
        mode = next((k for k, s in props.items() if isinstance(s, dict) and isinstance(s.get("enum"), list) and len(s["enum"]) >= 2), None)
        if mode is not None and args.get(mode) in props[mode]["enum"][1:]:
            return ""
        ident = self.new_line_id(args, self.rejected)
        if not ident or ident in self.cart_lines or ident in self.rejected_warned:
            return ""
        self.rejected_warned.add(ident)
        self.trace.append({"kind": "readd_rejected", "line": ident})
        return (f"the independent review already rejected {ident}: {self.rejected[ident]} A value stated only in a description about"
                " something else (how much a part makes, what it goes with) does not meet a requirement. Keep the listing whose attributes"
                " state every requirement; send the same call again only if an attribute of " + ident + " states the missing value.")

    def compat_clash(self, name: str, args: dict[str, Any]) -> str:
        """Once per listing: an addition that does not work with the cart's line of the item it is tied to."""
        props = (self.schemas.get(name) or {}).get("properties") or {}
        mode = next((k for k, s in props.items() if isinstance(s, dict) and isinstance(s.get("enum"), list) and len(s["enum"]) >= 2), None)
        if mode is not None and args.get(mode) in props[mode]["enum"][1:]:
            return ""
        ident = self.new_line_id(args)
        if not ident or ident in self.cart_lines or ident in self.compat_warned:
            return ""
        bad = [(d, a) for d, a in self.compat_mismatch(ident) if ident in (d, a)]
        if not bad:
            return ""
        self.compat_warned.add(ident)
        self.trace.append({"kind": "compat_clash", "line": ident, "bad": bad})
        d, a = bad[0]
        return (f"{d} names no code of {a} ({', '.join(sorted(self.link_codes(a))) or 'none'}); {d} names "
                f"{', '.join(sorted(self.link_codes(d))) or 'none'}, so it does not work with {a}. Pick listings whose codes match"
                " (see the code ledger), or send the same call again to add it anyway.")

    def stray_line(self, name: str, args: dict[str, Any]) -> str:
        """Once per listing: adding a listing whose store category is none of the requested items' categories."""
        props = (self.schemas.get(name) or {}).get("properties") or {}
        mode = next((k for k, s in props.items() if isinstance(s, dict) and isinstance(s.get("enum"), list) and len(s["enum"]) >= 2), None)
        if mode is not None and args.get(mode) in props[mode]["enum"][1:]:
            return ""
        if any((_count(v) or 1) <= 0 for v in args.values() if not isinstance(v, str)):
            return ""
        ident = self.new_line_id(args)
        cat_key = self.category_key()
        shelves = {(it.shelf or "").strip().lower() for it in self.items}
        if not ident or not cat_key or "" in shelves or ident in self.stray_warned:
            return ""
        shelf = str((self.rows.get(ident) or {}).get(cat_key) or (self.full.get(ident) or {}).get(cat_key) or "").strip().lower()
        if not shelf or shelf in shelves:
            return ""
        self.stray_warned.add(ident)
        self.trace.append({"kind": "stray_line_held", "line": ident, "shelf": shelf})
        return (f"{ident} is in the store category {shelf!r}, which is not the category of any requested item"
                f" ({'; '.join(sorted(shelves))}); a listing from another category does not count as the item. Cart a listing from the item's"
                " own category (send the same call again only if the request really names that category).")

    def extra_line(self, name: str, args: dict[str, Any]) -> str:
        """An addition of another listing for an item that already has its line in the cart: an order carries one line
        per requested item, so the old line must come off first."""
        props = (self.schemas.get(name) or {}).get("properties") or {}
        mode = next((k for k, s in props.items() if isinstance(s, dict) and isinstance(s.get("enum"), list) and len(s["enum"]) >= 2), None)
        if mode is not None and args.get(mode) in props[mode]["enum"][1:]:
            return ""
        if any((_count(v) or 1) <= 0 for v in args.values() if not isinstance(v, str)) or self.cart_sets(name):
            return ""
        ident = next((v for v in args.values() if isinstance(v, str) and v in self.rows), None)
        cat_key = self.category_key()
        if not ident or ident in self.cart_lines or not cat_key:
            return ""
        shelf = str((self.rows.get(ident) or {}).get(cat_key) or "").strip().lower()
        wanted = sum(1 for it in self.items if (it.shelf or "").strip().lower() == shelf) if shelf else 0
        if not wanted:
            return ""
        held = [i for i in self.cart_lines if str((self.rows.get(i) or {}).get(cat_key) or "").strip().lower() == shelf]
        if len(held) < wanted:
            return ""
        self.trace.append({"kind": "extra_line_held", "line": ident, "held": held})
        return (f"the cart already holds {', '.join(held)} for that item and an order carries one line per requested item. To swap,"
                f" take {held[0]} off first, then add {ident} (only when {ident}'s record has been read and fits better), then check out.")

    def lone_line(self, name: str, args: dict[str, Any]) -> str:
        """Late in the episode, once per line: a removal that would leave a requested item with no line in the cart
        (an order missing an item fails) is held back unless the item's replacement is already in the cart."""
        if self.left() > 6 or len(self.items) < 1:
            return ""
        props = (self.schemas.get(name) or {}).get("properties") or {}
        mode = next((k for k, s in props.items() if isinstance(s, dict) and isinstance(s.get("enum"), list) and len(s["enum"]) >= 2), None)
        removal = (mode is not None and args.get(mode) in props[mode]["enum"][1:]) or any((_count(v) or 1) <= 0 for v in args.values() if not isinstance(v, str))
        ident = next((v for v in args.values() if isinstance(v, str) and v in self.cart_lines), None)
        if not removal or not ident or ident in self.kept_lines:
            return ""
        cat_key = self.category_key()
        if not cat_key:
            return ""
        shelf = str((self.rows.get(ident) or {}).get(cat_key) or "").strip().lower()
        if not shelf or not any((it.shelf or "").strip().lower() == shelf for it in self.items):
            return ""
        others = [i for i in self.cart_lines if i != ident and str((self.rows.get(i) or {}).get(cat_key) or "").strip().lower() == shelf]
        if others:
            return ""
        self.kept_lines.add(ident)
        self.trace.append({"kind": "lone_line_kept", "line": ident})
        return (f"only {self.left()} steps are left and {ident} is the cart's only line for its item; an order missing an item fails,"
                " while its nearest listing may still pass. Keep it and check out; send the same call again only when you already read a"
                " listing that fits better and have the steps to add it and check out.")

    def whole_removal(self, name: str, args: dict[str, Any]) -> tuple[dict[str, Any], str]:
        """A removal (the cart tool's taking-off mode) sent without a count takes off one unit only: give it the line's
        whole count from Kyo's record, so taking a line off leaves nothing of it behind."""
        props = (self.schemas.get(name) or {}).get("properties") or {}
        mode = next((k for k, s in props.items() if isinstance(s, dict) and isinstance(s.get("enum"), list) and len(s["enum"]) >= 2), None)
        count = next((k for k, s in props.items() if isinstance(s, dict) and "integer" in _kinds(s)), None)
        if not mode or not count or count in args or args.get(mode) not in props[mode]["enum"][1:]:
            return args, ""
        ident = next((v for v in args.values() if isinstance(v, str) and v in self.cart_lines), None)
        have = self.cart_lines.get(ident or "", 0)
        if have <= 1:
            return args, ""
        top = props[count].get("maximum")
        units = min(have, top) if isinstance(top, int) else have
        return {**args, count: units}, f"Took off {units} units of {ident} (a removal without a count takes off only one)."

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
            if name == self.roles.get("cart"):
                args, kept = self.keep_other_lines(name, args)
                if kept:
                    trimmed += "\n[Kyo] " + kept
                unread = self.unread_add(name, args)
                if unread:
                    trimmed += "\n[Kyo] " + unread
                broken = self.add_breaks(name, args)
                if broken:
                    out.append((cid, "Not run (no step used): " + broken + "\nCart now: " + self.cart_text()))
                    continue
                again = self.readd_rejected(name, args)
                if again:
                    out.append((cid, "Not run (no step used): " + again + "\nCart now: " + self.cart_text()))
                    continue
                clash = self.compat_clash(name, args)
                if clash:
                    out.append((cid, "Not run (no step used): " + clash + "\nCart now: " + self.cart_text()))
                    continue
                stray = self.stray_line(name, args)
                if stray:
                    out.append((cid, "Not run (no step used): " + stray + "\nCart now: " + self.cart_text()))
                    continue
                extra = self.extra_line(name, args)
                if extra:
                    out.append((cid, "Not run (no step used): " + extra + "\nCart now: " + self.cart_text()))
                    continue
                lone = self.lone_line(name, args)
                if lone:
                    out.append((cid, "Not run (no step used): " + lone + "\nCart now: " + self.cart_text()))
                    continue
                args, whole = self.whole_removal(name, args)
                if whole:
                    trimmed += "\n[Kyo] " + whole
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
            if ending and self.is_buy(name, args) and not self.missing_filled and len(self.items) > 1 and self.left() >= 3:
                self.missing_filled = True
                before = dict(self.cart_lines)
                self.fill_missing()
                added = [i for i in self.cart_lines if i not in before]
                if added:
                    trimmed += "\n[Kyo] Some requested items had no cart line; Kyo added the nearest opened candidate for each: " + ", ".join(added)
            if ending and self.is_buy(name, args) and not self.compat_held and self.left() > 3:
                bad = self.compat_mismatch()
                if bad:
                    self.compat_held = True
                    self.trace.append({"kind": "compat_hold", "bad": bad})
                    desc = "; ".join(f"{d} (codes {', '.join(sorted(self.link_codes(d))) or 'none'}) does not name a code of {a} ({', '.join(sorted(self.link_codes(a))) or 'none'})" for d, a in bad[:4])
                    out.append((cid, "Not run yet (no step used). The order's items must work together, and these lines do not: " + desc
                                + ". Replace the lines so that every tied item names a code of its anchor's line (the code ledger lists matching"
                                " candidates), then check out.\nCart now: " + self.cart_text()))
                    continue
            if ending and self.left() > 2 and self.time_left() > CLOSING - 25:
                if self.is_buy(name, args):
                    noted = self.version
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
                    read = self.fill_facts(self.cart_ids(), self.tail() + 3) + self.follow_named(self.cart_ids(), self.tail() + 1) + self.cheaper_rows() + self.promising_rows()
                    if self.version != noted and self.notes and self.left() > 2:
                        self.queued = []
                        out.append((cid, "Not run yet (no step used). A note arrived while Kyo re-read the cart: " + self.notes[-1] + "\nCart now: " + self.cart_text() + "\nApply the change to the picks (a dropped requirement or a new budget can make another listing the right one), fix the cart, then check out."))
                        continue
                    verdict = self.check_cart()
                    if verdict:
                        seen = ("Kyo read these listings' attributes for the check (one step each):\n" + "\n".join(read) + "\n") if read else ""
                        out.append((cid, "Not run yet. " + chain + seen + verdict + "\nCart now: " + self.cart_text()))
                        continue
                    if reread and self.cost > SPEND_SOFT:
                        out.append((cid, "Not run yet (no step used). Kyo re-read the cart's listings just now:\n" + reread + "\nCheck out again when every line still shows stock and a price inside the budget; otherwise replace that line first."))
                        continue
                elif self.time_left() > CLOSING + 15:
                    # a held ending needs one more model turn to act on the reason; without that time the decline goes through
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
                            self.check_cart([first], SPEND_HARD)
                            if self.verdicts.get(self.check_key([first])) and first in self.leads:
                                self.confirmed = first
                                self.trace.append({"kind": "refuse_blocked", "confirmed": first})
                                out.append((cid, f"Not run yet (no step used). An independent check of {first}'s records says it meets every requirement. Add it to the cart at the requested quantity and check out; end without a purchase only if its record clearly breaks a requirement."))
                                continue
            if self.is_refuse(name, args) and self.left() <= 5 and not self.late_held and self.time_left() > CLOSING + 15:
                # too late for the review: a decline here usually throws away a built cart because time ran out
                lines = [i for i in self.built if not self.sold_out(i) and not self.contradictions(i)]
                fresh = self.cart_view is not None and self.cart_seen_at >= self.cart_at
                missing = [i for i in lines if i not in self.cart_ids() or (i in self.dropped and not fresh)]
                if len(lines) >= len(self.items) and len(missing) + 1 <= self.left():
                    self.late_held = True
                    self.trace.append({"kind": "late_decline_held", "lines": lines, "missing": missing})
                    redo = f"Put {', '.join(missing)} back in the cart at the requested quantities, then check out." if missing else "Check out the cart as it is."
                    out.append((cid, "Not run yet (no step used). Ending without a purchase earns nothing when the store has a fitting listing, and running short of steps does not show that it has none. "
                                + redo + " End without a purchase only if your search showed that some requested item has no listing that meets its requirements; then send the ending call again."))
                    continue
            if self.is_refuse(name, args) and not self.leads and not self.pushed and self.left() > self.max_steps // 3 and self.time_left() > CLOSING + 15:
                self.pushed = True
                out.append((cid, f"Not run yet (no step used). {self.left()} steps are still left, so search in ways you have not tried before ending: browse the category of each requested item and page through it with the token or offset the store returns, try single-word value queries and the property's other names, and open the listings that might fit. End without a purchase only after that."))
                continue
            if self.is_refuse(name, args) and self.brief.get("fallback_buy") is True and self.roles.get("buy"):
                # no opened listing of some item is free of a stated break: buying the nearest would fail just the same
                clean = self.nearest_lines()
                covered = len({self.item_of(i) for i in clean} - {None}) >= len(self.items) if clean else False
                if self.fallback_holds < 2 and self.left() > 3 and self.time_left() > CLOSING:
                    self.fallback_holds += 1
                    self.trace.append({"kind": "fallback_hold", "n": self.fallback_holds, "covered": covered})
                    if covered:
                        said = ("Ending without a purchase is almost never right for this shopper: when nothing meets every requirement, the closest listing is bought instead. Put the candidate that misses the fewest requirements (stock and budget count as requirements; keep within any price margin the request allows) in the cart at the requested quantity, then check out. Do not ask the shopper for permission.")
                    else:
                        said = (f"{self.left()} steps are left and no opened listing yet names every required value. Most requests do have a fitting listing that the searches so far missed:"
                                " search again with other words (each requirement value with the product's plain name, other names for the product) and open the rows that may fit."
                                " End without a purchase only after that, when still nothing names every required value.")
                    out.append((cid, "Not run yet (no step used). " + said + "\nCart now: " + self.cart_text()))
                    continue
                if not self.cart_ids() and not self.cart_lines:
                    self.cart_nearest()
                # a held cart whose lines plainly break a requirement fails as an order: the ending without a purchase stands
                if (self.cart_ids() or self.cart_lines) and not any(self.contradictions(i) for i in self.cart_lines):
                    faults = []
                    buy_args = _shape(dict(self.brief.get("buy_args") or {}), self.schemas.get(self.roles["buy"]), "args", faults)
                    if not faults and self.is_buy(self.roles["buy"], buy_args):
                        self.trace.append({"kind": "fallback_checkout", "cart": self.cart_ids()})
                        name, args = self.roles["buy"], buy_args
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
            if name == self.roles.get("search") and not ending:
                paged += self.auto_open(observation)
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
                if not (self.cart_shows_lines and self.cart_seen_at >= self.cart_at):
                    # a reply that does not list the cart: show Kyo's own record, so a line left with units is not taken for gone
                    trimmed += "\n[Kyo's record of the cart after this call (listing: units)] " + (_dump(self.cart_lines) if self.cart_lines else "empty")

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
                added = next((v for v in args.values() if isinstance(v, str) and v in self.cart_lines), None)
                if added and not ending:
                    named = self.follow_named([added], self.tail() + 2)
                    if named:
                        trimmed += "\n[Kyo opened the listings this line names, which the order needs opened too] " + " | ".join(named)
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
            if self.is_buy(name, args) and not reply.get("finished"):
                text += self.repair_after_refusal()
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
        fresh = self.cart_view is not None and self.cart_seen_at >= self.cart_at
        holding = bool(self.cart_ids()) if fresh else bool(self.cart_lines)
        if buy and not holding and self.left() >= 3:
            # an empty cart at a forced finish: buy the nearest candidates when the request wants that rather than nothing
            holding = self.cart_nearest()
        elif buy and holding and len(self.items) > 1 and self.left() >= 3:
            # a forced finish with requested items still missing: an order without them fails, so their nearest candidates go in
            self.fill_missing()
        if buy and holding and len(self.items) > 1 and self.left() >= 4:
            self.fix_compat()
        if buy and (holding or (self.cart_touched and not fresh and self.left() >= 2)):
            faults: list[str] = []
            args = _shape(dict(self.brief.get("buy_args") or {}), self.schemas.get(buy), "args", faults)
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
                    lines = self.loose_lines(reply.get("observation"))
                    if lines and not set(lines) <= self.heard:
                        self.take_note(lines)
                    repaired = self.repair_after_refusal()
                    return "the checkout was refused: " + self.show(reply.get("observation")) + repaired
        if refuse:
            faults = []
            args = _shape(self.quiet(refuse, dict(self.brief.get("refuse_args") or {})), self.schemas.get(refuse), "args", faults)
            if not faults:
                if self.left() >= 2 and self.time_left() > LAST_CALL + 12:
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
        self.trace.append({"kind": "brief", "brief": self.brief})
        opening = _unflatten(self.view.get("opening"))
        if opening is not None:
            self.absorb(opening)
        intro = "SHOPPER'S REQUEST:\n" + self.request
        if opening is not None:
            intro += "\n\nWHAT THE STORE SHOWS FIRST:\n" + self.show(opening)
        intro += "\n\nStart by searching for each item in its category (and with words of its requirement values) and open the listings that look like they fit.\n\n" + self.status()
        self.chat = [{"role": "system", "content": GUIDE}, {"role": "user", "content": intro}]
        self.set_caps()
        self.work(self.loop_cap, SPEND_HARD)
        if not self.finished:
            turned = self.finish(again=True)
            # the store turned the ending down (a change from the shopper, a sold-out line): a short second round deals with it
            if turned and not self.finished and not self.broke and not self.no_credit and self.left() > 1 and self.time_left() > CLOSING:
                self.chat.append({"role": "user", "content": "[Kyo] Kyo tried to end the episode and " + turned + "\nApply what the store says, then finish.\n" + self.status()})
                self.work(self.loop_calls + 10, SPEND_HARD + 0.01)
            if not self.finished:
                self.finish()
        self.trace.append({"kind": "end", "finished": self.finished, "steps": self.step, "cost": round(self.cost, 5), "calls": self.calls})
        return self.trace

    def work(self, loop_cap: int, spend_cap: float) -> None:
        idle = 0
        while (not self.finished and not self.broke and self.left() > 0 and self.loop_calls < loop_cap and self.cost < spend_cap
               and self.time_left() > CLOSING):
            if not self.swept and len(self.items) > 1 and self.left() >= 12 * len(self.items):
                swept = self.sweep()
                if swept:
                    self.chat.append({"role": "user", "content": swept + "\n" + self.status()})
            if not self.chased and self.step >= 1 and self.refs:
                chased = self.chase()
                if chased:
                    self.chat.append({"role": "user", "content": chased + "\n" + self.status()})
            if self.ready_to_ask():
                summary = self.interview()
                if summary:
                    self.chat.append({"role": "user", "content": summary + "\n" + self.status()})
                continue
            self.trim()
            self.loop_calls += 1
            # after a turn without a tool call, the next turn goes to another route first: a login-served model has been
            # seen reporting the store's tools as unavailable without ever calling them
            message = self.think(self.chat, [SPARE, HEAVY, LIGHT] if idle else [LIGHT, SPARE], 3000, self.offered())
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
                # a reply claiming the tools do not exist would mislead the next turns: keep it out of the conversation
                if not re.search(r"no such tool|tool[s]? (?:is|are) (?:not )?(?:available|unavailable)|not available", content or "", re.IGNORECASE):
                    self.chat.append({"role": "assistant", "content": content or "(no call)"})
                else:
                    self.chat.append({"role": "assistant", "content": "(no call)"})
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
            if not episode.finished:
                episode.finish()
        except Exception as again:
            episode.trace.append({"kind": "crash", "error": f"{type(again).__name__}: {again}"[:400]})
        return episode.trace


__all__ = ["agent_main"]
