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

LOOP_CALLS = 42
ALL_CALLS = 60
# dollars per episode. One key with a fixed cap pays for every episode of a run, and once it is spent every unfinished
# episode scores 0, so the mean has to stay far under that cap divided by the episodes (race worlds have long tasks).
# Above SOFT the reviews read with the light model first; at HARD the tool loop ends and Kyo finishes in code; at CEIL
# no model call goes out at all.
# race 15: 0.040 a race episode on average at the old 0.040 cap (long tasks ended in code, 41 steps against Halo's 47
# at 0.047); a little more room keeps the run near 4.0 of the key's 5
SPEND_SOFT = 0.028
SPEND_HARD = 0.046
SPEND_CEIL = 0.052
# seconds: the network ends an episode at 300 s whatever its state, so Kyo keeps its own clock: the tool loop ends
# at LOOP_SECONDS (time left to finish), and no model call starts after CALL_SECONDS
LOOP_SECONDS = 225.0
CALL_SECONDS = 262.0
# waits before resending a failed model call; an episode has 300 s in all, so at most 45 s go to waiting
WAITS = (4, 10, 20, 35)
WAIT_TOTAL = 45
SHOW_CHARS = 450
ROW_CHARS = 240             # strings inside the rows of a result list (search results): long descriptions are cut harder
OBS_LIMIT = 15000
HISTORY_LIMIT = 150000
GATE_ROUNDS = 3             # store calls the process gate may make before one checkout or ending
AUTO_PAGES = 6              # further search pages Kyo reads itself for a cheapest, most or least pick
CHEAPER_OPENS = 3           # cheaper rows of a cart line's kind whose attributes Kyo reads before a cheapest checkout
# properties one message to the shopper names, best first. The shopper answers about the first property of a message
# that it keeps a view on (a firm preference, or one it does not mind); properties it gives no thought at all are passed
# over, and a message naming none of its properties gets a puzzled reply. So a list reaches further than one name: each
# message names up to eight, a puzzled reply settles the whole list, an answer that it does not mind settles only the
# first name (the property it answered about may be any of them) and the later ones lead that item's next list, deepest
# first, once; an answer stating a preference settles the names up to the one it answered and the rest come next.
ASK_KEYS = 8

# one attempt per call (think() resends): a stuck call costs 50 s, not minutes, inside the 300 s
_llm = ProxyClient(timeout=50, max_retries=1, retry_delay=1)
_env = ProxyClient(timeout=95, max_retries=3, retry_delay=1)


GUIDE = """You are Kyo, a careful buyer working for one shopper in an online store. You can act only by calling the store's tools. The episode is graded 1 or 0: it earns 1 only when the finish is exactly right. A purchase must contain every requested item at the requested quantity, each meeting every requirement in force when you check out, inside the budget; or, when the store truly has nothing that meets everything, the episode must end with the no-purchase tool in the way the request asks.

HOW THE STORE WORKS
- Every tool call is one step and steps are limited; thinking costs nothing. Tool names, argument names and result shapes are particular to this store: read each tool's description and schema and use the exact argument names and allowed values.
- Search tools can be narrow: a category filter, a keyword mode that may only read the first words of the query, a page size of a few rows, a token or offset for the next page. Compare the reported total with the rows you have seen and page on (with the exact token or offset the store gives) when the answer depends on seeing all of them. Keyword queries work best short and value-first (the wanted value plus the product noun). A filtered search that returns no rows usually means the filter value or the filter parameter is wrong: retry with the label exactly as result rows show it, or with another filter parameter.
- Search rows are summaries. Stores split a listing's record differently: one view can show its attributes (properties), another only its offer and description, and a side-by-side view may carry either. The ledger names the tool that shows attributes once Kyo knows it. Judge every requirement on a property from those attributes, so read them for each listing you may buy; one call that shows several listings saves steps when it shows what you need. Never open a listing again when nothing has changed since you read it.
- Questions to the shopper are handled for you by Kyo's interviewer, so you never message the shopper yourself. Its answers arrive in the conversation and are added to the ledger.
- The request can also ask for steps along the way: an approval, a message, something to check or read first. The ledger lists them. Kyo carries out the ones due just before the order or the ending itself when you call the checkout or the ending tool, so never wait for those and never end without a purchase because of them; do any others with the store's tools (product questions go through Kyo's interviewer).

WHAT COUNTS AS A FIT
- The kind of product is decided by the listing's category label: only the label that matches the requested item as a whole counts, never a neighbouring one.
- A requirement on a property is met when the listing's attribute for that property states the value. Key spelling does not matter (spaces, underscores, camel case, prefixes), and one element of a list is enough, whether its options are joined by commas, slashes, "and", "or" or "&" ("indoor and outdoor" meets outdoor). A list of choices the buyer picks from (three or more different colours, flavours or versions on offer) does not say which one this listing is: by itself it neither meets nor breaks the requirement. Values match by their words: case, singular or plural and plain synonyms are fine, and a value that contains the required value's words counts ("deep teal" meets teal, "extra roomy fit" meets roomy). A value with different words does not count (an abbreviation or code is not the word it stands for, and "teal green" is not "sea green"). Units are converted before comparing numbers, limits include the edge value, and a range must lie wholly inside a limit.
- An attribute whose key names the requirement's property (under that name or a plain synonym) decides: if it says something else, the listing fails whatever other keys or the title say. A key for a different property, even a related one (a handle colour is not the body's colour), decides nothing about the requirement. Kyo adds [evidence] lines after opened listings with the keys whose names match a requirement.
- When the listing has no attribute for that property, the wanted value stated under another attribute (for example a list of styles or features that includes it) or in the title or description counts, unless other listings of its kind carry an attribute for that property: the store then reads the requirement from that attribute, so prefer a listing that states it there. A number of another kind does not count (a weight is not a capacity), nor a bare figure whose text does not say what it measures.
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
- End without a purchase only after a real search finds no listing that meets everything. Running short of steps proves nothing of the kind: then check out the best cart you have. Follow the request's wording for that ending: when it asks to be pointed to the nearest option, name the closest listing by its id with its price and what it misses; otherwise keep any remark short and name no listing.
- Every reply must call at least one tool. Keep your text short: what you know, then the next move."""


BRIEF = """You prepare a shopping task for an agent. Read the shopper's request and the store's tools and return ONE JSON object (compact, no indentation, nothing else) with these keys:

"items": a list with one entry per distinct product the shopper wants bought, each {"name": the kind of product in the request's words, "shelf": the store's category label for it when a tool lists labels (copy it exactly), else "", "quantity": units of that one listing (integer, 1 when not stated; a pair or a set that is one listing counts as 1), "musts": a list of firm requirements on that product, each "property: constraint" with the exact value, limit, exclusion or closed list. Name the property with the request's own word for it ("meant for a lakeside occasion" gives "occasion: lakeside", "a 12-inch blade" gives "blade length: 12 inch"); do not swap it for a neighbouring property. A must whose value is read from another listing is ONE must holding the whole description of that listing ("rim colour: unlike the rim colour of the kettle Ovarno sells with a glass lid"): words that describe that other listing (its shop, its kind, "with <property> <value> and <property> <value>" after it) stay inside that one must and are never musts of this product}. Never put price, budget, stock or quantity into musts. The story around the request (the person meant to use it, where it will stand, what went wrong before, the occasion behind it) is context, not a must, unless it states a property's value or limit outright; and never turn words that are part of what the product is called (a season, an audience or a use named with the product) into musts: those belong to "name"; but a word of the name that gives the value of one of the product's own properties (how it is made, styled or coloured: "a wicker basket" gives "material: wicker") stays in the name and is also a must on that property. A product the request names in one phrase joined by "and" (one "I want ..." with one quantity, often the name of one store department) is ONE item; make separate items only when the request gives each product its own quantity or its own requirements.
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
"together": the requested items the request wants to work with one another, as pairs of item numbers ([[2, 1]] when item 2 is an accessory for item 1; pieces that must suit one model or set pair up too), else [].
"browse": {"tool": the exact tool that lists the listings of one store category given the category's label, "label": that tool's parameter taking the label, "cap": its parameter that sets the highest price a listed row may have, or null when it has none, "with": the other arguments (with their values) that make the tool list a category and apply that cap, or {}}, or null when no tool lists a category.
"buy_args": the fixed arguments that make the "buy" tool check out (for a tool with a mode argument, that mode), else {}.
"refuse_args": the fixed arguments that make the "refuse" tool end without a purchase, else {}.
"""


PLAN = """You help a shopping agent choose what to ask a shopper. The shopper holds back a firm preference on one property of a requested product (now and then two), and states it only when a question names that property by its own name. The agent can send only a few messages; each names a short list of one product's properties, best first, and properties far down a list are answered less surely, so the order decides whether the preference is found.
For each item give two lists:
- "pointed": the properties a sentence of the REQUEST points at without giving their value: what it asks the agent to check with the shopper (listed under ASK FIRST), a choice it says someone else's taste or needs decide, something the shopper says they cannot judge or leave open, how the product should look or what it should match, what it must hold, fit or work with. Name each by one of the item's listed keys when one names it (several keys may be pointed at); only when no listed key names it, use a plain name of at most four words, and at most one such name per item. Give none when no sentence points at a property.
- "likely": the item's listed keys a person buying this kind of product most often has a firm view on, judged from what the product is for: keys whose values differ from listing to listing and that a buyer chooses by, ahead of keys that merely record a listing's particulars. When an item has fewer than four listed keys, give instead the properties a buyer of that kind of product most often chooses between, each by the plain name a store page would use for it.
Never give a property whose value the request already gives for that item, nor price, stock, delivery, seller, brand, an id, a model code, or a key that only lists a listing's variants or options on offer. Use the key's words in plain spelling ("lid colour" for lidColour) and give the whole key, never a shorter word inside it ("strap length", not "length"): two keys that share a word are two properties. One property per entry, no values.
Return ONE JSON object and nothing else: {"ask": [{"item": item number, "pointed": [{"label": "...", "first": true when it comes from ASK FIRST, "hint": "the request's own words that point at it, at most 10"}], "likely": ["listed key", ...]}]} with at most COUNT entries in each list."""


HEAR = """A shopping agent asked a shopper some questions. Read each answer and return ONE JSON object and nothing else:
{"needs": [{"item": item number, "need": "property: constraint", "points": true when the value is to be read from another listing, else false}], "drop": [{"item": item number, "property": "a requirement the shopper now gives up"}], "free": [{"item": item number, "property": "..."}], "lost": [numbers of the answers that say the shopper did not follow the question or ask for it again, simpler or in other words], "single": [numbers of the answers that complain the message asked about too many things at once or want one question at a time; not answers that only say they did not follow it or want it simpler or in plainer words], "budget": new total limit as a number or null}
- "needs": every firm preference an answer states, with the exact value or limit and its unit ("at least 30 cm", "oval", "not wicker"). Name the need by the property the answer itself names, which may differ from the one asked about; several answers about one property make one need. Copy the value word for word as the answer says it, keeping its nouns and units ("at least 3 shelves", not "at least 3").
- An answer can give the value by pointing at another listing (a value like, unlike or near the one on some shop's product, that product picked out by a few of its properties): that is ONE need on the property the answer names, with "points": true. Its value is the answer's own words copied exactly, from the comparison words to the end of that description, including what it says about a listing that page may name; never shorten or reword it. The properties describing the other product are not needs of this item.
- "free": each property a question named that the shopper said they do not mind about; an answer that waves off a whole question naming several properties frees every one of them, while an answer that states a need about one of them frees none of the others. An answer listed under "lost" states nothing: no needs and nothing free from it.
- Folksy or roundabout wording that names a value still states that value. Wording that refuses to give a value up (it is not up for trade or for bargaining, will not budge or change) states that value as wanted: write "not X" only when the answer rejects X itself. An answer that names one of the item's properties and ties it to a word or a figure states that word or figure as its value, however odd the word looks for that property or the phrase around it.
- An answer that hands the choice back or says the shopper holds no view on the property states no need, however figurative its wording: list that property under "free". Never read a value out of an image used to say the shopper does not care (such as plain, none or any)."""


NOTE = """While a shopping agent works, the store showed this note (it may come from the shopper or the store). Return ONE JSON object and nothing else:
{"kind": "change" when the note changes the task (budget, requirements, a price, availability, a refusal of the order), "error" when it only reports that a call was malformed or not accepted, "other" for anything else, "budget": the new total limit as a number in the request's unit, or null, "add": [{"item": item number, "need": "property: constraint"}], "drop": [{"item": item number, "need": "the requirement no longer wanted"}], "gone": [listing ids said to be unavailable], "sold_out": true when it says, or asks whether, a chosen listing is gone, taken, sold out, no longer available or could not be added (even without naming it), "price_moved": true when it says a price changed, "summary": "one short sentence"}
A replaced requirement is a drop of the old one plus an add of the new one. Report every change in the note."""


CHECK = """You are the last check before a shopping agent checks out. Decide from the records shown whether this cart is exactly right.
Reading rules: a requirement is met when the listing's attribute for that property states the value (any key spelling, one element of a list is enough, also when its options are joined by "and", "or" or "&", except a list of three or more choices on offer, which does not say which one this listing is; case, singular/plural and plain synonyms count, a value containing the required value's words counts, such as "deep teal" for teal, units converted, limits include the edge, a range must lie wholly inside); an attribute whose key names the same property as the requirement decides, and if it says something else the listing fails; a key for a different property, even a related one (a handle colour is not the body's colour), decides nothing; when the listing has no attribute for that property, the wanted value stated under another attribute or in the title or description counts (a number of another kind does not: a weight is not a capacity; nor does a bare figure whose text does not say what it measures), except where the evidence marks it NOT IN ITS ATTRIBUTES (other listings of its kind carry that attribute, so the store reads the requirement there and this listing does not meet it); a requirement the category label itself states is met by that category; the category label must match the requested item as a whole; a rule of being unlike another listing's value counts as met unless the records show that same value. Each requested item must appear as a separate cart line at the requested quantity, nothing else may be in the cart, nothing shown out of stock, and the total of price times quantity over all lines must be within the budget. The newest record of a listing wins.
When the pick rule is cheapest, most or least, also look at the other listings shown: if one of them clearly fits every requirement and beats the cart's choice on that rule, the cart is wrong.
Return ONE JSON object and nothing else: {"ok": true or false, "problems": ["short, concrete problem", ...], "better": [listing ids that should replace a cart line, best first]}. Use ok=false only for a problem the records clearly show or a requirement no record addresses."""


READING = """the listing's attribute for a property decides when its key names that same property (any key spelling, one element of a list is enough, also when its options are joined by "and", "or" or "&", except a list of three or more choices on offer, which does not say which one this listing is; synonyms and unit conversions count, a value containing the required value's words counts, such as "deep teal" for teal, limits include the edge); a key for a different property, even a related one (a handle colour is not the body's colour), decides nothing; when no attribute names that property, the wanted value stated under another attribute or in the title or description counts (a bare figure that does not say what it measures counts for nothing), unless other listings of its kind carry an attribute for that property (marked NOT IN ITS ATTRIBUTES): then it does not; a requirement the category label itself states is met by that category; the category label must match the item; a rule of being unlike another listing's value counts as met unless the records show that same value; out-of-stock listings and listings that break the budget do not fit."""

AUDIT = """A shopping agent wants to end without buying because it thinks nothing fits. Check the listing records shown against the requirements, with these reading rules: """ + READING + """
Records marked ROW were seen only as a search result row: their page (the attributes) has not been read, so they can be neither passed nor failed on a property the row does not show.
Return ONE JSON object and nothing else: {"fits": {"item number": [listing ids that appear to meet every requirement, cheapest first, at most 4]}, "open": [ids of ROW listings of a requested kind, inside the budget, whose page could show that they fit, most promising first, at most 6], "reason": "one sentence"}"""

UNMIX = """A shopper's request was read into firm requirements on a product. Some requirements take their value from another listing the request points at, and the words that pick out that other listing (its kind, its shop, the properties named after it with "with ...") are sometimes read by mistake as requirements on the product being bought. Using the request's wording, say which of the OTHER REQUIREMENTS below only describe that other listing. Return ONE JSON object and nothing else: {"describe_other": [those requirements copied exactly; empty when every one is a real requirement on the product]}"""

TRAIL = """A shopping agent is about to check out. Some requirements take their value from another listing instead of stating it, and a listing's page can send the reader on to a further listing for that property. The order counts only when the agent has opened in full every listing along each such chain. Follow each requirement's chain from the requirement's own description, link by link, through the pages' own words, using the records below (OPENED: the agent has opened that page; ROW: seen only as a search result row, its page not read).
Return ONE JSON object and nothing else: {"chains": [{"need": the requirement copied exactly, "links": [{"id": the listing id when a record below is that link, else "", "about": how the requirement or the previous page describes that listing, "shop": the shop or brand that sells that listing as those words give it, or "", "words": its kind and the properties those words give it}], "done": true when the last link is an OPENED page that states the property's value itself, else false, "value": the property's value as stated on that last page, or ""}]}
List each chain's links in order and stop at the first link whose page is not OPENED or not found, since its words are still unknown."""

GATE = """A shopping agent is about to STAGE. The shopper's request asks for the steps below along the way. Decide from the numbered RECORD of what the agent has done whether each step due before this STAGE is done for the cart as it is now. A step counts as done only when a RECORD line shows it being done and, for a step about the cart or the order, that line comes after the cart last changed and its reply grants it; questions on a product's properties do not do a step.
When a step is still missing, give the one store call that does it next: a tool from TOOLS with its arguments. A message to the shopper goes through the messaging tool: one short sentence or question about that step alone, in plain words, naming no product properties or values the step does not need. Give no call when every due step is done or when the messages or steps left cannot do the missing one.
Return ONE JSON object and nothing else: {"steps": [{"step": the step, "done": true or false, "lines": [the RECORD line numbers that show it done]}], "call": {"tool": "...", "args": {...}} or null, "hold": what a reply asks to change before going on, or "", "needs": [{"item": item number, "need": "property: constraint"} for each firm requirement such a reply states]}"""

SWAP = """A listing in a shopping agent's cart has just sold out. Find replacements for that cart line among the other listing records shown, with these reading rules: """ + READING + """
Return ONE JSON object and nothing else: {"fits": [{"id": listing id, "price": its price as a number as the record shows it} for every listing that appears to meet every requirement of the sold-out line's item, at most 6], "reason": "one sentence"}"""

ALIGN = """A shopping agent read a shopper's request into the requirements below. For each, read the request again and say which of the item's listed keys the requirement constrains: the property the request's own words give that value or figure to, which may differ from the word the requirement was filed under (a unit, or a counted thing named next to a figure, is not the property). Give null when none of the listed keys is that property.
Return ONE JSON object and nothing else: {"map": [{"must": the requirement exactly as given, "key": the listed key exactly as given, or null}]}"""

COMPAT = """Some requested items must work together: the request wants each PAIR below to fit. For each item, the CANDIDATES are listings that meet its own requirements, with the words their pages give. Find combinations of one candidate per item in which every pair is backed by the listings' own words: one names a model, series or part that the other carries, read as written (a range, or a code with letters standing for some of its digits, covers the codes it spans; an exception named with it leaves that code out). A candidate whose words name nothing for the pair does not prove the pair.
Return ONE JSON object and nothing else: {"sets": [{"lines": {"item number": "listing id"}, "why": "one short sentence"}] (at most 4, best backed first), "need": "for an item where no candidate works, the model, series or part its listing must name, else empty"}"""

SHELVES = """A shopping agent is searching a store for the requested items below. The store's results carry the category labels listed under LABELS, each with a few listing titles. For each requested item, give the one label whose listings are that kind of product, copied exactly, or null when none of them is. Two items may share a label.
Return ONE JSON object and nothing else: {"items": [{"item": item number, "label": "the label" or null}]}"""

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


def _clip_view(value: Any, depth: int = 0, row: bool = False) -> Any:
    """What the tool loop reads of a result: strings inside the rows of a result list (three or more records, such as a
    search page) are cut at ROW_CHARS, other strings at SHOW_CHARS. Every step re-reads the whole conversation, so long
    descriptions of rows nobody opened cost on every later turn."""
    if depth > 7:
        return "..."
    if isinstance(value, str):
        limit = ROW_CHARS if row else SHOW_CHARS
        return value if len(value) <= limit else value[:limit] + "..."
    if isinstance(value, dict):
        return {k: _clip_view(v, depth + 1, row) for k, v in value.items()}
    if isinstance(value, list):
        rows = len(value) >= 3 and all(isinstance(v, dict) for v in value)
        return [_clip_view(v, depth + 1, row or rows) for v in value]
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
    """Plain lower-case words: camel case, underscores, dashes and punctuation split; a decimal figure stays whole ("0.73")."""
    key = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(key))
    key = re.sub(r"(?<=\d)\.(?=\d)", "\u00b7", key)
    return re.sub(r"[^\w\u00b7]+|_", " ", key).strip().lower().replace("\u00b7", ".")


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


def _fact_block(node: Any, least: int = 2) -> dict[str, Any] | None:
    """The listing's attribute block: its largest nested dict of plain values (at least `least` of them)."""
    if not isinstance(node, dict):
        return None
    best = None
    for value in node.values():
        if isinstance(value, dict) and len(value) >= least:
            scalars = {k: v for k, v in value.items() if isinstance(v, (str, int, float)) and not isinstance(v, bool)}
            if len(scalars) >= least and (best is None or len(scalars) > len(best)):
                best = scalars
    return best


# ---------- records a store sends as one line of text ----------
# Some stores send each listing as a single string, "key: value | key: value | ...", with the attribute block inside one
# field as "name=value; name=value". Read that way, a result has the same records a store with structured rows sends,
# so the ledger, the key table, the fit checks and the tool loop's view see every field instead of a string cut short.
_TEXT_FIELD = re.compile(r"([a-z][a-z0-9_]{0,40}(?: [a-z0-9_]{1,20}){0,3}):(?: (.*)|)$", re.S)
_TEXT_PAIR = re.compile(r"([A-Za-z][\w .&/\-]{0,48}?)=(.*)$", re.S)


def _text_value(text: str) -> Any:
    word = text.strip()
    if word in ("True", "False"):
        return word == "True"
    if word in ("None", "null"):
        return None
    if re.fullmatch(r"-?\d{1,15}", word) and not re.fullmatch(r"0\d+", word):
        return int(word)
    if re.fullmatch(r"-?\d{1,12}\.\d{1,6}", word):
        return float(word)
    return word


def _text_pairs(text: str) -> dict[str, Any] | None:
    """'a=1; b=two words' as {'a': '1', 'b': 'two words'}: at least two pairs, the text opening with one, and most of
    its parts pairs (a part without '=' continues the value before it)."""
    parts = re.split(r";\s+", text.strip())
    out: dict[str, str] = {}
    last = None
    loose = 0
    for part in parts:
        found = _TEXT_PAIR.match(part.strip())
        if found and found.group(1).strip() not in out:
            last = found.group(1).strip()
            out[last] = found.group(2).strip()
        elif last is None:
            return None
        else:
            out[last] += "; " + part
            loose += 1
    if len(out) == 1 and not loose and re.fullmatch(r"[a-z][a-z _]{1,48}", next(iter(out))):
        return dict(out)                     # a block of one attribute ("compatible brand=hp")
    if len(out) < 2 or loose * 2 > len(out):
        return None
    return dict(out)


_TEXT_LINE = re.compile(r"([A-Za-z][A-Za-z0-9_]{0,40}(?: [A-Za-z0-9_]{1,20}){0,3}):(?: (.*)|)$", re.S)


def _text_record(text: Any) -> dict[str, Any] | None:
    """One listing sent as text, as a dict; None for any other string (a reply, a note, a sentence with a colon). Fields
    are split by ' | ' (a piece that does not open with a lower-case key continues the field before it, as a title
    holding ' | ' does) or, in a text of three or more "Key: value" lines, by line."""
    if not isinstance(text, str) or len(text) > 60000:
        return None
    if text.count(" | ") >= 2:
        pieces, pattern, joint = text.split(" | "), _TEXT_FIELD, " | "
    elif text.count("\n") >= 2 and sum(1 for line in text.split("\n") if _TEXT_LINE.match(line.strip())) >= 3:
        pieces, pattern, joint = [line.strip() for line in text.split("\n") if line.strip()], _TEXT_LINE, "\n"
    else:
        return None
    fields: dict[str, str] = {}
    last = None
    for piece in pieces:
        found = pattern.match(piece)
        if found and found.group(1) not in fields:
            last = found.group(1)
            fields[last] = found.group(2) or ""
        elif last is None:
            return None
        else:
            fields[last] += joint + piece
    if len(fields) < 3:
        return None
    record: dict[str, Any] = {}
    for key, value in fields.items():
        block = _text_pairs(value) if "=" in value else None
        record[key] = block if block else _text_value(value)
    if not any(_id_like(v) for v in record.values() if isinstance(v, str)):
        return None
    return record


def _from_text(node: Any, depth: int = 0) -> Any:
    """The observation with every listing sent as text read into a dict; everything else unchanged."""
    if depth > 7:
        return node
    if isinstance(node, str):
        record = _text_record(node)
        return node if record is None else record
    if isinstance(node, list):
        return [_from_text(v, depth + 1) for v in node]
    if isinstance(node, dict):
        if len(node) >= 3 and any(_id_like(v) for v in node.values() if isinstance(v, str)):
            # already a structured record (or a result with a cursor): its own texts stay as they are, rows inside it are read
            return {k: v if isinstance(v, str) else _from_text(v, depth + 1) for k, v in node.items()}
        return {k: _from_text(v, depth + 1) for k, v in node.items()}
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


# ---------- fit judge: where a requirement's evidence sits on a listing ----------
# filler words that carry no value in a requirement or an attribute value
_FILLER = frozenset("a an the of in on for with to be is are it its as at by from that this".split())
# a requirement that compares, excludes or points elsewhere is left to the model's reading
_READ_ME = re.compile(r"\b(not|no|none|never|without|except|unlike|other than|anything but|rather than|instead of|at least|at most|up to|under|over|more than|less than|between|same as|similar|compatible|page|listing|sold by|or less|or more|minimum|maximum)\b|[<>≤≥]")
_JOINERS = re.compile(r"\s*(?:,|/|;|\||&|\band\b|\bor\b|\+)\s*")


def _tokens(text: str) -> frozenset[str]:
    return frozenset(w for w in _stem(text) if w not in _FILLER)


def _one_off(a: str, b: str) -> bool:
    """Two long words one letter apart (an added, dropped or changed letter): spellings of one word."""
    if a == b:
        return True
    if min(len(a), len(b)) < 5 or abs(len(a) - len(b)) > 1:
        return False
    if len(a) > len(b):
        a, b = b, a
    i = 0
    while i < len(a) and a[i] == b[i]:
        i += 1
    # an added or dropped letter from five letters on, a changed letter only in words of seven or more
    return a[i:] == b[i + 1:] or (len(a) == len(b) >= 7 and a[i + 1:] == b[i + 1:])


def _near_word(a: str, b: str) -> bool:
    """One word in two inflections or spellings: equal, one letter apart, or one a stem of the other by at most two letters."""
    if a == b or _one_off(a, b):
        return True
    short, long = (a, b) if len(a) <= len(b) else (b, a)
    return len(short) >= 3 and not short.isdigit() and long.startswith(short) and len(long) - len(short) <= 2


def _voiced(need: str, answer: str) -> bool:
    """The answer says the need's value itself: every word of the value, apart from the property's own words and
    comparison words, shows up in the answer (inflections and one-letter spellings allowed, figures exactly)."""
    prop, _, value = need.partition(":")
    if not value.strip():
        value, prop = prop, ""
    def bare(text: str) -> set[str]:
        return {w[:-1] if len(w) > 3 and w.endswith("s") else w for w in re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text.lower())} - _FILLER
    words = bare(_READ_ME.sub(" ", value.lower())) - bare(prop)
    if not words:
        return True
    heard = bare(answer)
    return all(any(_near_word(w, h) for h in heard) for w in words)


def _spelled_alike(want: str, value: Any) -> bool:
    """A listing value that may be another spelling of the wanted word ("flat-woven" for "flatweave"): with spaces and
    hyphens taken out, one starts with the other or the two share their first five letters or more."""
    a = re.sub(r"[^a-z0-9]", "", str(want).lower())
    for option in re.split(r"\s*(?:,|/|;|\|)\s*", str(value).lower()):
        b = re.sub(r"[^a-z0-9]", "", option)
        if len(a) >= 5 and len(b) >= 5 and a != b and (a.startswith(b) or b.startswith(a) or a[:5] == b[:5]):
            return True
    return False


def _under(price: float) -> float:
    """The highest price below `price` worth listing: one unit less for whole prices, a cent less otherwise."""
    return price - 1 if float(price).is_integer() else round(price - 0.01, 2)


def _key_names(prop: str, key: str) -> bool:
    """The key names the requirement's property: same words, a subset of them, or the same words spelled apart by one letter."""
    p, k = _stem(prop), _stem(key)
    if _names(p, k):
        return True
    near = {w for w in p if any(_one_off(w, x) for x in k)}
    return bool(p) and len(near) == len(p)


# store codes: two to five capitals and three to six figures ("AB1234"), as a listing's own part number or a model it pairs with
_CODE = re.compile(r"\b[A-Z]{2,5}-?\d{3,6}\b")
_SENTENCE = re.compile(r"(?<=[.;!?])\s+")


def _family(code: str) -> str:
    return re.match(r"[A-Z]+", code).group()


# inside a sentence that is mostly codes, a store also writes a model as a name with a number ("Name 123") and a run of
# models as a code with trailing placeholders ("AB12xx"); codes after a negation ("except AB120") are left out of it
_CODE_NAME = re.compile(r"\b[A-Z][a-z]{2,11} \d{3,5}\b")
_CODE_RUN = re.compile(r"\b[A-Z]{2,5}-?\d{1,5}[xX]{1,4}\b")
_CODE_NOT = re.compile(r"\b(?:except|excluding|but not|not|other than|apart from)\b", re.I)


def _code_sentences(text: Any) -> list[tuple[list[str], list[str]]]:
    """Each sentence of a text that is mostly codes (at most four plain words before any negation), as (the codes it
    names, the codes it names after a negation); running prose that happens to name a model code is left out. A
    sentence naming models only as names with a number counts in a text that also has a sentence of plain codes, or
    when it lists two or more of them after a colon ("Fits: Name 123, Name 456."): a short title such as a brand and a
    model number is no code sentence."""
    found = []
    for sentence in _SENTENCE.split(str(text or "")):
        marks = [(m.start(), m.group()) for pattern in (_CODE, _CODE_NAME, _CODE_RUN) for m in pattern.finditer(sentence)]
        if not marks:
            continue
        cut = _CODE_NOT.search(sentence)
        head = sentence[:cut.start()] if cut else sentence
        tail = sentence[cut.end():] if cut else ""
        for pattern in (_CODE_RUN, _CODE, _CODE_NAME):
            head, tail = pattern.sub(" ", head), pattern.sub(" ", tail)
        plain = len(re.findall(r"[A-Za-z]{2,}", head))
        plain += len([w for w in re.findall(r"[A-Za-z]{2,}", tail) if w.lower() not in ("and", "or", "nor")])
        coded = any(_CODE.fullmatch(c) or _CODE_RUN.fullmatch(c) for _, c in marks)
        if plain > (4 if coded else 3):
            continue
        marks = sorted(set(marks))
        named = [c for at, c in marks if not cut or at < cut.start()]
        barred = [c for at, c in marks if cut and at >= cut.start()]
        found.append((named, barred, coded, ":" in sentence and len(marks) >= 2))
    anchored = any(coded for _, _, coded, _ in found)
    return [(named, barred) for named, barred, coded, listed in found if coded or anchored or listed]


def _code_lines(text: Any) -> list[list[str]]:
    """The codes each code sentence of a text names (see _code_sentences), the ones after a negation left out."""
    return [named for named, _ in _code_sentences(text) if named]


def _code_hit(codes: set[str], wanted: set[str]) -> bool:
    """One of `codes` is among `wanted`, directly or inside a run of models ("AB12xx" holds AB1240)."""
    if codes & wanted:
        return True
    runs = [w.replace("-", "") for w in wanted if _CODE_RUN.fullmatch(w)]
    for run in runs:
        head = run.rstrip("xX")
        for code in codes:
            flat = code.replace("-", "")
            if len(flat) == len(run) and flat.startswith(head) and flat[len(head):].isdigit():
                return True
    return False


def _runs_meet(a: set[str], b: set[str]) -> bool:
    """Two pages' runs of models share a model: the same run, or one run inside the other ("AB1xxx" holds "AB12xx")."""
    for x in (r.replace("-", "") for r in a if _CODE_RUN.fullmatch(r)):
        for y in (r.replace("-", "") for r in b if _CODE_RUN.fullmatch(r)):
            hx, hy = x.rstrip("xX"), y.rstrip("xX")
            if len(x) == len(y) and (hx.startswith(hy) or hy.startswith(hx)):
                return True
    return False


def _same_prop(a: str, b: str) -> bool:
    """Two labels for one property: the same words, allowing plural endings and one-letter spellings. A label whose words
    hold another's and more ("strap length" beside "length") is another property."""
    p, k = _stem(a), _stem(b)
    return bool(p) and len(p) == len(k) and all(any(_one_off(w, x) for x in k) for w in p)


def _spoken(name: str) -> str:
    """A key as a person says it: leading one- or two-letter code fragments of a store key are dropped ("m top neckline")."""
    words = _words(name).split()
    while len(words) > 1 and len(words[0]) <= 2 and not words[0].isdigit():
        words = words[1:]
    return " ".join(words)


class FitJudge:
    """Reads, without a model, where each requirement is decided on a listing record.

    Verdicts: "meets" (a key naming the property holds the value, whole or as one option of a list),
    "differs" (a key naming the property holds other words), "elsewhere" (no key names the property,
    the value's words stand in another value or in the listing's text), "silent" (nothing says it),
    "read" (a number, a limit, an exclusion or a pointer: the model reads it)."""

    @staticmethod
    def split(need: str) -> tuple[str, str]:
        prop, _, value = str(need).partition(":")
        return (prop.strip(), value.strip()) if value.strip() else ("", prop.strip())

    @staticmethod
    def options(value: Any) -> list[frozenset[str]]:
        parts = [p for p in _JOINERS.split(str(value).lower()) if p.strip()]
        whole = _tokens(str(value))
        found = [_tokens(p) for p in parts]
        return [whole] + [t for t in found if t and t != whole]

    @staticmethod
    def choices(want: str) -> list[frozenset[str]]:
        """The values a requirement accepts: "A or B", "A, B or C" and "A/B" name alternatives, any one of which will do;
        a value without such a list is one choice (words joined by "and" stay together: all of them are wanted)."""
        low = re.sub(r"(?<=\d)[,/](?=\d)", "\u2060", str(want).lower())
        whole = _tokens(want)
        if not re.search(r"\bor\b|,|/|\|", low):
            return [whole]
        found = [_tokens(p.replace("\u2060", "/")) for p in re.split(r"\s*(?:,|/|\||\bor\b)\s*", low) if p.strip()]
        return [t for t in found if t] or [whole]

    @classmethod
    def verdict(cls, need: str, block: dict[str, Any], text: str, decide: str | None = None) -> tuple[str, str]:
        prop, want = cls.split(need)
        low = want.lower()
        if not want or _READ_ME.search(low):
            return "read", ""
        if not _tokens(want):
            return "read", ""
        allowed = cls.choices(want)
        # a figure is a number to read; a code with letters before its digits ("i5", "ddr4") is a word
        numeric = bool(re.search(r"(?<![a-z])\d", low))

        def held(value: Any) -> bool:
            return any(choice <= option for option in cls.options(value) for choice in allowed)

        def offered(value: Any) -> bool:
            """Three or more different alternatives listed, the wanted one only among them: a choice on offer."""
            whole = _tokens(str(value))
            parts = [t for t in cls.options(value)[1:] if t]
            return len(parts) >= 3 and not any(choice <= whole and len(choice) * 2 > len(whole) for choice in allowed)

        named = [(k, v) for k, v in block.items() if prop and _key_names(prop, k)]
        if decide and len(named) > 1:
            # several keys name the property: the one the item's listings show the wanted value under decides
            named = [(k, v) for k, v in named if _words(k) == decide] or named
        for key, value in named:
            if held(value):
                return ("read" if offered(value) else "meets"), f"{key}={str(value)[:60]}"
        if named:
            key, value = named[0]
            bare = re.sub(r"[^a-z0-9]", "", low)
            if numeric or any(_spelled_alike(want, v) or (bare and bare in re.sub(r"[^a-z0-9]", "", str(v).lower())) for _, v in named):
                return "read", f"{key}={str(value)[:60]}"
            return "differs", f"{key}={str(value)[:60]}"
        for key, value in block.items():
            if held(value) and not offered(value):
                return "elsewhere", f"{key}={str(value)[:60]}"
        words = _tokens(text)
        if any(choice <= words for choice in allowed):
            return "elsewhere", "title/description"
        return ("read" if numeric else "silent"), ""

    @classmethod
    def sheet(cls, needs: list[str], record: dict[str, Any], deciders: dict[str, str] | None = None, least: int = 2) -> list[tuple[str, str, str]]:
        block = _fact_block(record, least) or {}
        text = " ".join(str(v) for v in record.values() if isinstance(v, str))[:4000]
        return [(need, *cls.verdict(need, block, text, (deciders or {}).get(need))) for need in needs]

    @staticmethod
    def clear(sheet: list[tuple[str, str, str]]) -> bool:
        """Every requirement met by a key naming it or stated elsewhere on the listing, none contradicted."""
        return bool(sheet) and all(v in ("meets", "elsewhere") for _, v, _ in sheet)

    @staticmethod
    def broken(sheet: list[tuple[str, str, str]]) -> list[str]:
        return [need for need, v, _ in sheet if v == "differs"]

    @staticmethod
    def render(ident: str, sheet: list[tuple[str, str, str]]) -> str:
        marks = {"meets": "MEETS", "differs": "DIFFERS", "elsewhere": "STATED ELSEWHERE", "silent": "NOT STATED", "read": "TO READ",
                 "keyless": "NOT IN ITS ATTRIBUTES"}
        parts = [f"{need} -> {marks[v]}" + (f" ({where})" if where else "") for need, v, where in sheet]
        return f"{ident}: " + "; ".join(parts)


# ---------- key planner: which listing keys are worth a question ----------
class KeyPlanner:
    """Orders the attribute keys seen on opened listings for the interview, by rules that need no list of key names.

    A held-back preference has to tell listings of its kind apart, so it sits on a key that most of them carry and whose
    values differ between them. Keys are ordered by: values that differ (a key showing one value on every listing that
    carries it goes last), then the number of opened listings carrying it, then the number of distinct values. Words or
    figures make no difference: a capacity or a size is held back as often as a colour or a material.
    Keys the request already settles, or that answers already decided, are left out."""

    @classmethod
    def rank(cls, table: list[tuple[str, str, int, list[str], int]], decided: Any) -> list[tuple[str, str, int, list[str], int]]:
        """`decided` is a list of settled requirements, or a function giving that list for a category label (the
        requirements of the item the category belongs to: a colour settled for one item leaves another item's colour open)."""
        out = []
        for label in dict.fromkeys(row[0] for row in table):
            settled = decided(label) if callable(decided) else decided
            done = [_stem(d.split(":", 1)[0]) for d in settled if d.strip()]
            keys = [row for row in table if row[0] == label and not any(_stem(row[1]) <= d for d in done if _stem(row[1]))]
            # a key showing one value on every listing carrying it goes last only when five or more listings carry it:
            # on a few pages the same value is often chance, or the very value the shopper holds back
            keys.sort(key=lambda row: (row[4] >= 2 or row[2] < 5, row[2], row[4]), reverse=True)
            out.extend(keys)
        return out


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
        self.size_caps()
        self.started = time.monotonic()
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
        self._cat_memo: tuple[tuple[int, int], str | None] | None = None
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
        self.pushed = False
        self.built: list[str] = []
        self.dropped: set[str] = set()
        self.late_held = False
        self.fit_guarded = False
        self.cheaper_said: set[str] = set()
        self.browse_asked: set[int] = set()
        self.lead_items: dict[str, list[str]] = {}
        self.verdicts: dict[str, bool] = {}
        self.heavy_checked = False
        self.rescued = False
        self.budget_held: set[str] = set()
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
        self.empty_holds = 0
        self.browse: dict[str, Any] = {}
        self.cap_bad = False
        self.scanned: set[tuple[str, float]] = set()
        self.turns = 0
        self.shelf_reads = 0
        self.turned_version = -1
        self.aligned = False
        self.link_searched: set[str] = set()
        self.code_memo: dict[str, tuple[int, tuple[set[str], set[str], set[str]]]] = {}
        self.code_bars: dict[str, set[str]] = {}
        self.say_shelf: set[int] = set()
        self.end_rescued = False
        self.family_memo: tuple[int, set[str]] = (-1, set())
        self.parts_looked: set[str] = set()
        self.last_seen: dict[str, int] = {}
        self.cut_short = ""
        self.fit_pairs: list[tuple[int, int]] = []
        self.fit_marks: set[str] = set()
        self.browsed: set[str] = set()
        self.search_seen: dict[str, Any] = {}
        self.twinned: set[str] = set()
        self.browse_empty = 0
        self.checkout_scanned: set[int] = set()
        self.decider_cache: dict[tuple[int, str, int], str | None] = {}
        self.shape_tries: dict[str, int] = {}
        self.trail_memo: tuple[Any, dict[str, Any]] | None = None
        self.turned_step = -99
        self.shelf_seen: set[str] = set()
        self.paged_obs: list[Any] = []

    # ---------- plumbing ----------
    def size_caps(self) -> None:
        """Turn, call and dollar caps grow with the steps the task allows: a long many-item task needs more turns and
        model calls than a short one, while a short task keeps the tight caps that kept the race key from running dry."""
        extra = min(65, max(0, self.max_steps - 45))
        self.loop_cap = LOOP_CALLS + extra // 2
        self.call_cap = ALL_CALLS + (extra * 3) // 5
        self.soft = SPEND_SOFT + extra * 0.0001
        self.hard = SPEND_HARD + extra * 0.00015
        self.ceil = SPEND_CEIL + extra * 0.00015

    def left(self) -> int:
        return self.max_steps - self.step

    def elapsed(self) -> float:
        return time.monotonic() - self.started

    def spent_out(self) -> bool:
        """No model call may go out: the episode's dollars or seconds are used up."""
        return self.cost >= self.ceil or self.elapsed() >= CALL_SECONDS

    def judges(self) -> list[str]:
        """Models for a review or a reading that weighs evidence: the heavy model while the episode is cheap so far."""
        return [HEAVY, LIGHT] if self.cost < self.soft else [LIGHT, HEAVY]

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

    def think(self, messages: list[dict[str, Any]], models: list[str], max_tokens: int, tools: list[dict[str, Any]] | None = None,
              effort: str | None = None) -> dict[str, Any] | None:
        for model in models:
            if self.calls >= self.call_cap or self.spent_out():
                return None
            if self.cost + self.estimate(model, messages, max_tokens) > self.ceil:
                # this call could carry the episode past its cap: the next (cheaper) model, if any
                self.trace.append({"kind": "model_skipped", "model": model, "cost": round(self.cost, 4)})
                continue
            self.calls += 1
            tries = 0
            while True:
                message, status = self.call_model(model, messages, max_tokens, tools, effort)
                if message is not None:
                    return message
                # no credit, rate limits, timeouts and server errors pass: wait and send the same call again
                passing = status is None or status in (402, 408, 425, 429) or (isinstance(status, int) and status >= 500)
                # a timeout already took the client's 50 s: it is sent once more at most
                if (not passing or tries >= (1 if status in (None, 408) else len(WAITS)) or self.waited + WAITS[tries] > WAIT_TOTAL
                        or self.elapsed() + WAITS[tries] > CALL_SECONDS):
                    break
                pause = WAITS[tries] * (0.8 + 0.4 * random.random())
                self.waited += WAITS[tries]
                tries += 1
                time.sleep(pause)
            if status == 402:
                self.no_credit = True
                return None
        return None

    @staticmethod
    def estimate(model: str, messages: list[dict[str, Any]], max_tokens: int) -> float:
        """A pessimistic price of one call: the prompt unread from cache at about 4 characters a token, and the output
        (thinking included) at about half its token limit. Only the heavy model's calls can move an episode's spend
        much; the light ones are priced too, at their own rates."""
        chars = sum(len(m["content"]) if isinstance(m.get("content"), str) else len(_dump(m.get("content"))) for m in messages)
        dollars_in, dollars_out = (2.0, 10.0) if model == HEAVY else (0.1, 0.5)
        return (chars / 4 * dollars_in + max_tokens / 2 * dollars_out) / 1e6

    def call_model(self, model: str, messages: list[dict[str, Any]], max_tokens: int, tools: list[dict[str, Any]] | None,
                   effort: str | None = None) -> tuple[dict[str, Any] | None, Any]:
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
        if effort:
            # both models think before they answer, and the thinking is billed and counts against max_tokens:
            # a call that only lists candidates asks for less of it, so the answer fits
            body["reasoning"] = {"effort": effort}
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

    def ask_json(self, prompt: str, models: list[str], max_tokens: int = 1800, effort: str | None = None) -> dict[str, Any] | None:
        for model in models:
            message = self.think([{"role": "user", "content": prompt}], [model], max_tokens, None, effort)
            got = _json_obj((message or {}).get("content"))
            if got is not None:
                return got
            if self.broke or self.no_credit or self.calls >= self.call_cap or self.spent_out():
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
                    data["observation"] = _from_text(data["observation"])
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
        roles = brief.get("roles") if isinstance(brief.get("roles"), dict) else {}
        self.roles = {k: v for k, v in roles.items() if isinstance(v, str) and v in self.schemas}
        for key in ("buy_args", "refuse_args"):
            if not isinstance(brief.get(key), dict):
                brief[key] = {}
        browse = brief.get("browse") if isinstance(brief.get("browse"), dict) else {}
        tool = browse.get("tool") if isinstance(browse.get("tool"), str) else ""
        if tool in {self.roles.get(r) for r in ("buy", "refuse", "cart", "cart_view", "ask")}:
            tool = ""
        props = ((self.schemas.get(tool) or {}).get("properties") or {}) if tool else {}
        if not isinstance(browse.get("label"), str) or not (browse.get("cap") is None or isinstance(browse.get("cap"), str)):
            props = {}
        if props and browse.get("label") in props and "string" in _kinds(props[browse["label"]]):
            cap = browse.get("cap") if browse.get("cap") in props and {"integer", "number"} & set(_kinds(props[browse["cap"]])) else None
            extra = browse.get("with") if isinstance(browse.get("with"), dict) else {}
            self.browse = {"tool": tool, "label": browse["label"], "cap": cap, "with": {k: v for k, v in extra.items() if k in props and k not in (browse["label"], cap)}}
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
        for pair in _items(brief.get("together")):
            a, b = (_count(pair[0]), _count(pair[1])) if isinstance(pair, (list, tuple)) and len(pair) == 2 else (None, None)
            if a and b and a != b and 1 <= a <= len(self.items) and 1 <= b <= len(self.items) and (a, b) not in self.fit_pairs and (b, a) not in self.fit_pairs:
                self.fit_pairs.append((a, b))
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
            self.size_caps()
        for entry in _items(brief.get("process")):
            if isinstance(entry, dict) and str(entry.get("step") or "").strip():
                when = str(entry.get("when") or "").strip().lower()
                self.process.append({"step": str(entry["step"]).strip(), "when": when if when in ("choose", "order", "end") else "order",
                                     "messages": max(0, _count(entry.get("messages")) or 0)})
        if self.roles.get("ask"):
            # the messages a step due at the checkout takes are kept back; a step due at an ending without a purchase gets
            # a message only if one is left (online, those endings almost never pass, a question often does)
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
            got = self.ask_json(prompt, self.judges(), 500) or {}
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
        elif left <= 6 or (left <= len(self.items) + 5 and self.buys == 0):
            text += " Few steps left: set the cart to the chosen lines and check out now."
        elif self.buys == 0 and not self.turns and left <= 10 + 2 * len(self.items):
            text += (" Steps are running low: settle on your picks, fill the cart and check out within the next few steps."
                     " The store often answers the first checkout (or ending) with a change from the shopper, and the steps left are needed to act on it.")
        elif self.buys == 0 and not self.cart_touched and self.step >= self.max_steps // 2:
            text += " Half the steps are used and the cart is empty: cart each item's fitting listing now (the store may still sell one out)."
        live = {need for _, need in self.needs_now()}
        lookups = [need for _, need in self.refs if need in live and need not in self.ref_values]
        if lookups and self.buys == 0:
            text += "\n[resolve first: open the listing each of these points at, and every listing its page refers to; the last page gives the value] " + " | ".join(lookups[:3])
        if self.leads:
            text += "\n[review candidates] " + ", ".join(self.leads)
        if self.notes:
            text += "\n[changes so far] " + " | ".join(self.notes[-4:])
        text += self.item_board()
        text += self.code_board()
        text += "\n[ledger]\n" + self.ledger()
        return text

    def code_board(self) -> str:
        """For items the request pairs, once the cart holds one of a pair: the codes that line carries and the rows seen of
        its partner whose own text names one of them, so the loop reads codes in result rows instead of searching them."""
        if not self.fit_pairs or not self.cart_lines:
            return ""
        held = self.lines_by_item()
        lines: list[str] = []
        for a, b in self.fit_pairs:
            for mine, other in ((a, b), (b, a)):
                partner = next(iter(held.get(other) or []), None)
                if not partner or any(self.backs(partner, i) for i in held.get(mine) or []):
                    continue
                # the codes the partner's page names in its code sentences (a model's number, the models a part fits),
                # not its own part number; only in a store whose pages pair by plain codes or runs of them
                named = sorted(self.codes_of(partner)[1])
                if not any(_CODE.fullmatch(c) or _CODE_RUN.fullmatch(c) for c in named):
                    continue
                bars = sorted(self.code_bars.get(partner) or [])
                rows = [i for i in reversed(self.seen_order) if self.item_of(i) == mine and self.buyable(i) and self.backs(partner, i)][:5]
                said = (f"item {mine} ({self.items[mine - 1].name}) must carry or name a code that {partner}'s page names ({', '.join(named[:4])}"
                        + (f"; not {', '.join(bars[:3])}" if bars else "") + ") in its own text: ")
                lines.append(said + ("rows seen that do: " + ", ".join(rows) if rows else
                                     "no row seen does yet; a store's search rarely finds codes, so search the item by its kind and brand words and read the codes in the rows"))
        return ("\n[fits by code] " + " | ".join(lines[:3])) if lines else ""

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
                    self.last_seen[ident] = self.step
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
        mark = (len(self.rows), len(self.full))
        if self._cat_memo and self._cat_memo[0] == mark:
            return self._cat_memo[1]
        key = self.category_key_now()
        self._cat_memo = (mark, key)
        return key

    def category_key_now(self) -> str | None:
        counts: dict[str, set[str]] = {}
        present: dict[str, int] = {}
        for row in self.rows.values():
            for key, value in row.items():
                # the listing's own id is never its category, however few rows there are
                if key == self.id_key:
                    continue
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

    def key_table(self) -> list[tuple[str, str, int, list[str], int]]:
        """(category, key name, listings carrying it, sample values, distinct values), most carried first per category."""
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
        for label, group in sorted(groups.items(), key=lambda kv: -kv[1]["n"]):
            kept: list[tuple[str, dict[str, Any]]] = []
            for name, slot in sorted(group["keys"].items(), key=lambda kv: -kv[1]["n"]):
                values = list(slot["per"].values())
                if not values:
                    continue
                mean = sum(len(v) for v in values) / len(values)
                if mean > 40 or sum(1 for v in values if " / " in v) * 2 >= len(values):
                    continue
                # a key whose every value is a different longer text is a listing's own particulars (a name, a blurb),
                # unless its values are mostly figures: a measure such as a display size differs from listing to listing
                figures = sum(1 for v in values if re.search(r"\d", v)) * 2 >= len(values) and mean <= 30
                if len(values) >= 4 and len(set(values)) == len(values) and mean > 12 and not figures:
                    continue
                twin = any(name.startswith(other + " ") and slot["per"] == spec["per"] for other, spec in kept)
                if twin:
                    continue
                kept.append((name, slot))
            # stores of many-featured kinds carry fifty keys or more: the planner sees the 26 most carried
            for name, slot in kept[:26]:
                table.append((label, name, slot["n"], [v[:40] for v in slot["values"][:4]], len(slot["values"])))
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
        if name in self.shapes and (self.shapes[name] is not None or self.shape_tries.get(name, 0) >= 2):
            return self.shapes[name]
        if not isinstance(observation, dict):
            return None
        lists = [k for k, v in observation.items() if isinstance(v, list) and len(v) >= 2 and all(isinstance(r, dict) for r in v)]
        if not lists:
            return None
        self.shapes[name] = None
        self.shape_tries[name] = self.shape_tries.get(name, 0) + 1
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
        if value is None and isinstance(given, int) and not isinstance(given, bool) and given >= shown and given + shown < count and not self.cut_short:
            # the store stops listing before the count it reports: the rest is only reached by other searches
            self.cut_short = f"{given + shown} of {count}"
            return (f"\n[Kyo] The store stopped this listing at row {given + shown} of the {count} it reports: the other {count - given - shown}"
                    " are reached only by other searches (the product's plain name, its synonyms, words from fitting titles), and the pick counts them too.")
        later = given not in (None, 0, 1) if isinstance(given, int) and not isinstance(given, bool) else bool(str(given or "").strip())
        if value is None or later or count <= shown:
            # the last page, or the agent is paging itself
            return ""
        if pick == "cheapest" and self.price_sorted(observation):
            return ""
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

    def over_budget(self) -> str:
        """Once per cart state: the cart's known prices break the budget now in force (a note can cut it mid-task)."""
        if not isinstance(self.budget, (int, float)) or isinstance(self.budget, bool):
            return ""
        ids = self.cart_ids()
        prices = {i: self.price_of(i) for i in ids}
        if not ids or None in prices.values():
            return ""
        mark = self.cart_mark() + f"#{self.version}#{self.budget}"
        if mark in self.budget_held:
            return ""
        if self.brief.get("budget_scope") == "each":
            # a price ten times the budget or more means the two are in different units: nothing is said then
            over = [f"{i} at {p:g}" for i, p in prices.items() if self.budget < p <= 10 * self.budget]
            text = f"{', '.join(over)} costs more than the budget of {self.budget:g} per item" if over else ""
        else:
            total = sum(p * self.cart_lines.get(i, 1) for i, p in prices.items())
            text = f"the cart totals {total:g}, more than the budget of {self.budget:g} now in force" if self.budget + 1e-9 < total <= 10 * self.budget else ""
        if not text:
            return ""
        self.budget_held.add(mark)
        return (text[0].upper() + text[1:] + ". Replace a line with a cheaper listing that fits, or, when no fitting set is inside "
                "the budget, end without a purchase as the request says.")

    def wider_hold(self) -> str:
        """Once, before a cheapest checkout with many steps left: a store's keyword search reads titles, so fitting
        listings whose titles use other words are missed by searches for a requirement's value."""
        if self.wider_held or self.brief.get("pick") != "cheapest" or self.left() < (8 if self.cut_short else 14 + 2 * len(self.items)):
            return ""
        self.wider_held = True
        lines = [f"{i} at {self.price_of(i):g}" for i in self.cart_ids() if self.price_of(i) is not None]
        return (f"Not run yet (no step used). {self.left()} steps are left and the pick is the cheapest. Searches match a listing's title,"
                " so a requirement's value (often only in the attributes) finds few of the fitting listings, and rows come in no price order."
                + (f" This store's category listing stopped at row {self.cut_short} it reports, so its other rows are unseen." if self.cut_short else "")
                + " Before checking out, run a few more searches for each item under other names for the product (its plain one-word name,"
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
            number = self.item_of(line) or 0
            pool.sort(key=lambda i: (-self.visible_fit(i, number), self.price_of(i) or 0.0))
            picks += [i for i in pool[:2] if i not in picks]
        return self.fill_facts(picks[:CHEAPER_OPENS], self.tail() + 4)

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
            reads = 3 if self.max_steps >= 50 else 2
            lanes.append(pool[: max(0, min(reads, 8 - opened))])
        queue = [lane[k] for k in range(3) for lane in lanes if k < len(lane)]
        # while steps are plentiful: 22 stay for the rest of the episode
        self.fill_facts(queue[:6], 22)

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
        """Each message names a short list of one requested item's properties, best first, chosen after the previous
        answer was read. Every requested item has its own queue: what the request asks to be consulted on and what its
        story points at (the planner's reading), then the item's attribute keys in KeyPlanner order. Items take turns, and
        an item whose answer stated a preference gives its turn to the items still without one: most items hold back a
        single preference. An answer that did not follow the question states nothing: its properties go back to the head
        of that item's queue, and the item's next list is half as long."""
        self.asked = True
        if self.msgs_free() <= 0:
            return ""
        self.map_shelves()
        self.browse_kinds()
        self.cover_kinds()
        self.widen()
        self.align_musts()
        queues = self.ask_queues()
        self.trace.append({"kind": "ask_queues", "queues": {n: q[:12] for n, q in queues.items()}})
        sent: list[tuple[int, str, str]] = []
        told: list[str] = []
        found: dict[int, int] = {}
        room = {n: ASK_KEYS for n in queues}
        again: set[tuple[int, str]] = set()
        misses: dict[int, int] = {}
        last_list: dict[int, list[str]] = {}
        last = 0
        while self.msgs_free() > 0 and not self.finished and self.left() > 6 and self.elapsed() < LOOP_SECONDS - 20:
            pick = self.next_ask(queues, found, last, room)
            if pick is None:
                break
            number, labels = pick
            if labels == last_list.get(number):
                # never the same message twice: its first name was surely heard
                labels = labels[1:]
                if not labels:
                    continue
            last_list[number] = list(labels)
            batch = self.send([(number, ", ".join(labels), self.question(number, labels))])
            if not batch:
                break
            sent += batch
            last = number
            before = len(told)
            # the light model reads each answer; a need whose value the answer never voices goes to the heavy one
            readers = [LIGHT, HEAVY]
            heard = self.hear(batch, tuple(readers))
            if len(labels) > 1:
                # waving off a list says nothing sure about its later properties: only the first counts as free
                heard["free"] = [e for e in _items(heard.get("free")) if isinstance(e, dict)
                                 and not any(_same_prop(str(e.get("property") or ""), label) for label in labels[1:])]
            unvoiced = self.take_heard(heard, told, " ".join(a for _, _, a in batch))
            if unvoiced:
                self.take_heard(self.second_reading(batch, unvoiced, readers), told)
            if _items(heard.get("single")) and len(labels) > 1 and len(told) == before:
                # the shopper wants one thing at a time: lists end here and this list's properties are asked singly
                room = {n: 1 for n in room}
                queues[number] = labels + queues[number]
                self.trace.append({"kind": "ask_single", "item": number})
                continue
            if len(told) > before:
                found[number] = found.get(number, 0) + 1
                at = self.answered_at(labels, told[before:])
                if at >= 0:
                    queues[number] = labels[at + 1:] + queues[number]
                else:
                    # the answer named a property under a word of its own: the names after the first go again, once
                    rest = [label for label in labels[1:] if (number, label) not in again]
                    again.update((number, label) for label in rest)
                    queues[number] = rest + queues[number]
                self.trace.append({"kind": "ask_hit", "item": number, "at": at})
            elif _items(heard.get("lost")):
                # a puzzled reply: the message named none of the properties the shopper keeps a view on, as the shopper
                # heard the item; when the store's category names the item otherwise, that item is asked in the
                # category's words from now on, its puzzled names behind the new ones
                item = self.items[number - 1]
                if number not in self.say_shelf and item.shelf and _stem(item.shelf) != _stem(item.name):
                    self.say_shelf.add(number)
                    queues[number] = queues[number] + [label for label in labels if (number, label) not in again]
                    again.update((number, label) for label in labels)
                self.trace.append({"kind": "ask_none", "item": number, "names": len(labels), "reword": number in self.say_shelf})
            elif len(labels) > 1:
                # waved off: the property answered about may be any of them; the later ones go again once, the deepest
                # first, ahead of the queue after one such answer and behind the new names after the next
                deep = [label for label in labels[:0:-1] if (number, label) not in again]
                again.update((number, label) for label in deep)
                misses[number] = misses.get(number, 0) + 1
                queues[number] = deep + queues[number] if misses[number] % 2 else queues[number] + deep
                self.trace.append({"kind": "ask_again", "item": number, "deep": deep[:4]})
        if not sent:
            return ""
        if told:
            self.version += 1
        lines = [f"Q: {q}\nA: {a}" for _, q, a in sent]
        summary = "Kyo's interviewer asked the shopper (each question used one step):\n" + "\n".join(lines)
        summary += "\nNew firm requirements from these answers: " + ("; ".join(told) if told else "none")
        summary += ". Apply them to every pick (search again with the new values where needed)."
        pointed = [need for _, need in self.refs if any(need in t for t in told)]
        if pointed and self.left() > 14:
            walked, _ = self.trail(may_hold=False)
            if walked:
                summary += "\n" + walked
        if pointed:
            summary += (" These take their value from another listing: " + " | ".join(pointed) + ". Find that listing, open its page in full,"
                        " and open every listing its page refers to; the last page gives the value, and the order counts only after all of them are opened.")
        if self.queued:
            summary += "\n[notes seen meanwhile] " + " | ".join(self.queued)
            self.queued = []
        return summary

    def cover_kinds(self) -> None:
        """Before the interview: an item that no opened listing shows the kind of gets one search by its name (in its store
        category when the store's search takes one) and two of its rows read, so the planner sees that item's keys too;
        only while twenty steps or more are left."""
        for number, item in enumerate(self.items, 1):
            if self.left() < 20 or self.finished:
                return
            if any(self.item_of(i) == number for i in self.full):
                continue
            rows = [i for i in self.seen_order if self.item_of(i) == number]
            if not rows:
                rows = [i for i in self.code_search(item.name, item.shelf or None, plain=not item.shelf) if self.item_of(i) == number]
            pool = [i for i in rows if i not in self.full and self.buyable(i)][:2]
            self.trace.append({"kind": "cover_kind", "item": number, "rows": len(rows), "read": pool})
            if pool:
                self.fill_facts(pool, 18)

    def ask_queues(self) -> dict[int, list[str]]:
        """Per requested item, the properties to ask about, best first."""
        settled: list[str] = []
        decided: list[str] = []
        for number, item in enumerate(self.items, 1):
            settled.append(f"item {number}: {item.name} - settled by the request: " + ("; ".join(item.musts) or "nothing"))
            decided += item.musts + item.told + item.free

        def decided_for(label: str) -> list[str]:
            number = self.item_for_label(label)
            if number is None and len(self.items) == 1:
                number = 1
            if number is None:
                return decided
            item = self.items[number - 1]
            return item.musts + item.told + item.free

        table = KeyPlanner.rank(self.key_table(), decided_for)
        queues: dict[int, list[str]] = {n: [] for n in range(1, len(self.items) + 1)}

        def add(number: int, label: str) -> None:
            label = _spoken(label)
            if not label or len(label.split()) > 5:
                return
            if any(_same_prop(label, other) for other in queues[number]):
                return
            queues[number].append(label)

        for number, label in self.plan_hints(settled, table):
            add(number, label)
        for row in table:
            number = self.item_for_label(row[0])
            if number is None and len(self.items) == 1:
                number = 1
            if number:
                add(number, row[1])
        return queues

    def plan_hints(self, settled: list[str], table: list[tuple[str, str, int, list[str], int]]) -> list[tuple[int, str]]:
        """The careful model ranks, per item, the properties most worth a question: what the request singles out first,
        then the item's listed keys a buyer of that kind of product most often holds a view on. A pick must name a key
        the item's listings carry, unless the request's own words point at it."""
        listing = []
        last = None
        for label, name, carried, values, _ in table:
            if label != last:
                owner = self.item_for_label(label) or (1 if len(self.items) == 1 else None)
                listing.append(f"category {label!r}" + (f" (item {owner}):" if owner else ":"))
                last = label
            listing.append(f"  - {name} (on {carried}): " + " | ".join(values))
        per_item = -(-max(1, self.msgs_free()) // max(1, len(self.items)))
        count = min(12, max(6, per_item * 4))
        prompt = PLAN.replace("COUNT", str(count)) + "\n\nREQUEST:\n" + self.request + "\n\nITEMS:\n" + "\n".join(settled)
        first = [str(x) for x in _items(self.brief.get("ask_first")) if str(x).strip()]
        if first:
            prompt += "\n\nASK FIRST (the request asks to be consulted on these):\n" + "\n".join(first)
        prompt += "\n\nATTRIBUTE KEYS SEEN:\n" + ("\n".join(listing) or "(no full listing records yet)")
        plan = self.ask_json(prompt, self.judges(), 2500) or {}
        said = _tokens(self.request)
        keyed: dict[int, list[str]] = {}
        for label, name, *_ in table:
            number = self.item_for_label(label)
            if number is None and len(self.items) == 1:
                number = 1
            if number:
                keyed.setdefault(number, []).append(name)
        picks: list[tuple[int, str]] = []

        def clean(text: Any) -> str:
            # one property per question: a label naming two is cut to the first
            return re.split(r"\s+(?:and|or)\s+|[/,;(]", str(text or ""))[0].strip().rstrip("?.")

        for entry in _items(plan.get("ask")):
            if not isinstance(entry, dict):
                continue
            number = min(max(1, _count(entry.get("item") or 1) or 1), len(self.items))
            listed_keys = keyed.get(number, [])
            unlisted = 0
            # the story's own pointers first (a key they name, or one plain name the request's words point at), then likely keys
            for pick in _items(entry.get("pointed")):
                if not isinstance(pick, dict):
                    continue
                label = clean(pick.get("label"))
                quoted = _tokens(str(pick.get("hint") or ""))
                pointed = pick.get("first") is True or (bool(quoted) and len(quoted & said) * 2 >= len(quoted))
                listed = any(_key_names(label, key) or _key_names(key, label) for key in listed_keys)
                if not label or not (listed or (pointed and unlisted == 0)):
                    continue
                unlisted += 0 if listed else 1
                if sum(1 for n, _ in picks if n == number) < count:
                    picks.append((number, label))
            # an item few opened listings describe (a store whose pages give little, or none read yet) takes the
            # planner's own names for what buyers of that kind of product choose by
            thin = len(set(listed_keys)) < 4
            for key in _items(entry.get("likely")):
                label = clean(key.get("label") if isinstance(key, dict) else key)
                if label and (thin or any(_key_names(label, k) or _key_names(k, label) for k in listed_keys)) and (number, label) not in picks \
                        and len(label.split()) <= 4 and sum(1 for n, _ in picks if n == number) < count + 1:
                    picks.append((number, label))
        self.trace.append({"kind": "plan", "ask": _clip(plan.get("ask"), 160), "kept": picks})
        return picks

    def next_ask(self, queues: dict[int, list[str]], found: dict[int, int], last: int,
                 room: dict[int, int] | None = None) -> tuple[int, list[str]] | None:
        """The next (item, properties): items in turn after the last one asked, those without a stated preference first;
        properties already settled, stated or waved off for that item are skipped (a key more specific than a settled
        property, such as "strap length" beside a settled "length", stays). The item's room says how many go in one list."""
        numbers = sorted(queues)
        for number in numbers:
            item = self.items[number - 1]
            done = [_stem(m.split(":", 1)[0]) for m in item.musts + item.told + item.free if m.strip()]
            settled = [m for m in item.musts + item.told if ":" in m and m not in item.dropped]
            queues[number] = [label for label in queues[number]
                              if not any(_stem(label) <= d or _same_prop(label, " ".join(sorted(d))) for d in done if d)
                              and not any(self.holds_value(number, label, m) or self.same_values(number, label, m) for m in settled)]
        live = [n for n in numbers if queues[n]]
        if not live:
            return None
        waiting = [n for n in live if not found.get(n)] or live
        start = numbers.index(last) + 1 if last in numbers else 0
        for offset in range(len(numbers)):
            number = numbers[(start + offset) % len(numbers)]
            if number in waiting:
                take = max(1, (room or {}).get(number, 1))
                labels: list[str] = []
                while queues[number] and len(labels) < take:
                    label = queues[number].pop(0)
                    if not any(_same_prop(label, other) for other in labels):
                        labels.append(label)
                return number, labels
        return None

    def answered_at(self, labels: list[str], new: list[str]) -> int:
        """Position in a list of the earliest property a new requirement names (by its words, colour and color alike);
        -1 when none does, as when the answer names a property the list did not."""
        hits = []
        for need in new:
            prop = re.sub(r"^item \d+:\s*", "", need).split(":", 1)[0]
            words = _stem(prop)
            for at, label in enumerate(labels):
                mine = _stem(label)
                # words that open alike count as one ("pack type" for "packaging type")
                near = words and mine and all(any(w[:4] == x[:4] for x in words) for w in mine if len(w) >= 4) \
                    and any(len(w) >= 4 for w in mine) and any(w[:4] == x[:4] for w in mine for x in words if len(w) >= 4)
                if _same_prop(label, prop) or (words and mine and (mine <= words or words <= mine)) or near:
                    hits.append(at)
                    break
        return min(hits) if hits else -1

    def holds_value(self, number: int, label: str, need: str) -> bool:
        """The key behind a label holds a settled requirement's value on an opened listing of the item, and the label names
        that requirement's property: it is the same property under a longer key ("jackets closure type" for "closure:
        zipper"), not a different one sharing a word ("memory type" beside "memory: 16 gb")."""
        prop, want = FitJudge.split(need)
        if not prop or not _names(_stem(prop), _stem(label)):
            return False
        allowed = FitJudge.choices(want)
        for ident, record in self.full.items():
            if self.item_of(ident) != number:
                continue
            for key, value in (_fact_block(record) or {}).items():
                if _spoken(key) == label and any(c and c <= o for o in FitJudge.options(value) for c in allowed):
                    return True
        return False

    def same_values(self, number: int, label: str, need: str) -> bool:
        """The key behind a longer label and the key named by a settled requirement's property show the same value on one
        opened listing ("clothing material" and "material" both "cotton"): one property under two keys."""
        prop = FitJudge.split(need)[0]
        if not prop or not _names(_stem(prop), _stem(label)) or _same_prop(prop, label):
            return False
        for ident, record in self.full.items():
            if self.item_of(ident) != number:
                continue
            block = _fact_block(record, 1) or {}
            mine = [str(v).strip().lower() for k, v in block.items() if _spoken(k) == label]
            theirs = [str(v).strip().lower() for k, v in block.items() if _same_prop(_spoken(k), prop)]
            if any(a and a == b for a in mine for b in theirs):
                return True
        return False

    def question(self, number: int, labels: list[str] | str) -> str:
        # properties only, no example values: values make the shopper ask for one thing at a time
        names = [labels] if isinstance(labels, str) else list(labels)
        item = self.items[number - 1].name
        if number in self.say_shelf and self.items[number - 1].shelf:
            item = self.items[number - 1].shelf.strip().lower()
        parts = [p.strip() for p in re.split(r"\s*(?:,|\band\b)\s*", item) if p.strip()]
        if len(parts) >= 2 and all(re.search(r"[a-z]{3}(?<![su])s$", p, re.I) for p in parts):
            # an item naming several products ("door handles and locks") is one choice to the shopper: "or" says so
            item = re.sub(r"\band\b", "or", item)
        if len(names) == 1:
            return f"What {names[0]} would you like for the {item}?"
        return f"What {', '.join(names[:-1])} or {names[-1]} would you like for the {item}?"

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
            notes: list[str] = []
            if isinstance(answer, dict):
                # a reply can come with notes beside it (a list of lines next to the one answer): the answer is read as
                # the shopper's reply, the notes as changes to the task, the way a note on any other result is
                texts = [v for v in answer.values() if isinstance(v, str) and v.strip()]
                lines = [x for v in answer.values() if isinstance(v, list) for x in v if isinstance(x, str) and len(x.split()) >= 4]
                if len(texts) == 1 and lines:
                    words = texts[0]
                    notes = [x for x in lines if x not in self.heard]
            sent.append((number, text, words))
            if notes:
                meaning = self.take_note(notes)
                if meaning:
                    self.queued.append(f"(seen with a reply) {meaning}")
            self.done_log.append(f"step {self.step}: message to the shopper {text!r} -> reply {words[:300]!r} (cart then: {self.cart_mark()})")
            self.done_steps.append(self.step)
            self.trace.append({"kind": "ask", "q": text, "a": words[:300]})
            if isinstance(answer, dict) and answer.get("error"):
                break
        return sent

    def take_heard(self, heard: dict[str, Any], told: list[str], said: str = "") -> list[dict[str, Any]]:
        """Add what the answers stated to the ledger; `told` collects the new requirements for the summary. With the
        answers' text in `said`, a need whose value the answers never voice is held back and returned instead: the
        interviewer has it read a second time before it may count."""
        many = len(self.items) > 1
        unvoiced: list[dict[str, Any]] = []
        for entry in _items(heard.get("needs")):
            if not isinstance(entry, dict) or not str(entry.get("need") or "").strip():
                continue
            number = min(max(1, _count(entry.get("item")) or 1), len(self.items))
            need = str(entry["need"]).strip()
            if said and entry.get("points") is not True and not _voiced(need, said):
                unvoiced.append(entry)
                continue
            if need not in self.items[number - 1].told and need not in self.items[number - 1].musts:
                self.items[number - 1].told.append(need)
                told.append(f"item {number}: {need}" if many else need)
            points = entry.get("points") is True or self.names_shop(need)
            if points and (number, need) not in self.refs:
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
                item = self.items[number - 1]
                prop = str(entry["property"])
                # a "don't mind" about a property the request or an answer settles does not unsettle it
                if any(_key_names(m.split(":")[0], prop) or _key_names(prop, m.split(":")[0]) for m in item.musts + item.told if ":" in m and m not in item.dropped):
                    continue
                item.free.append(prop)
        if isinstance(heard.get("budget"), (int, float)) and not isinstance(heard.get("budget"), bool) and heard["budget"] > 0:
            self.budget = heard["budget"]
        return unvoiced

    def names_shop(self, need: str) -> bool:
        """A need whose value is a whole description (eight words or more) naming, as a proper name, a shop or brand the
        store's rows carry: its value is read from that listing, whatever the reader said."""
        value = need.partition(":")[2]
        if len(value.split()) < 8:
            return False
        names = set(re.findall(r"\b[A-Z][a-z]{3,}\b", value))
        return bool(names) and any(isinstance(v, str) and v.strip() in names for row in self.rows.values() for v in row.values())

    def second_reading(self, batch: list[tuple[int, str, str]], unvoiced: list[dict[str, Any]], first: list[str]) -> dict[str, Any]:
        """A need whose value the answer never says aloud counts only when a second reader, the other model, finds the
        same property with the same value words; otherwise the answer is taken as stating nothing about it."""
        other = [m for m in (HEAVY, LIGHT) if m != (first[0] if first else LIGHT)][:1]
        again = self.hear(batch, tuple(other)) if other and not self.spent_out() else {}
        seen = [str(e.get("need") or "") for e in _items(again.get("needs")) if isinstance(e, dict)]
        kept = []
        for entry in unvoiced:
            prop, _, value = str(entry["need"]).partition(":")
            same = any(_tokens(prop) == _tokens(n.partition(":")[0]) and _tokens(value) == _tokens(n.partition(":")[2]) for n in seen)
            self.trace.append({"kind": "unvoiced_need", "need": entry["need"], "kept": same})
            if same:
                kept.append(entry)
        return {"needs": kept}

    def hear(self, sent: list[tuple[int, str, str]], models: tuple[str, ...] | None = None) -> dict[str, Any]:
        prompt = HEAR + "\n\nITEMS:\n" + "\n".join(f"item {n}: {it.name}" for n, it in enumerate(self.items, 1))
        prompt += "\n\nQUESTIONS AND ANSWERS:\n" + "\n".join(f"{i}. (about item {n}) Q: {q}\nA: {a}" for i, (n, q, a) in enumerate(sent, 1))
        return self.ask_json(prompt, list(models) if models else self.judges(), 1500) or {}

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
        # a note about a moved price is not a sell-out, even when it came as a refused order: the line stays and its
        # offer is re-read before the next checkout
        self.note_sold = bool(gone) or (got.get("sold_out") is True and got.get("price_moved") is not True)
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
        return (summary + (" (" + "; ".join(changed) + ")" if changed else "")).strip()

    def needs_now(self) -> list[tuple[int, str]]:
        out = []
        for number, item in enumerate(self.items, 1):
            for need in item.musts + item.told:
                if need not in item.dropped:
                    out.append((number, need))
        return out

    def evidence(self, observation: Any) -> str:
        """Verdicts of the fit judge for the listings this result shows in full, against their item's requirements."""
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
        lines = [self.judged_line(ident) for ident in ids[:12]]
        lines = [line for line in lines if line]
        if not lines:
            return ""
        return ("[fit judge: how each requirement reads on these records. MEETS: a key naming the property holds the value (whole or as one option of a list); "
                "STATED ELSEWHERE: no key names it, another value or the text states it; DIFFERS: a key naming it holds something else; "
                "NOT STATED: nothing says it; TO READ: a number, limit, exclusion or pointer you must read yourself] " + " | ".join(lines))

    def item_of(self, ident: str) -> int | None:
        """The requested item whose kind this listing's category label names, by the item's store label or its name."""
        cat_key = self.category_key()
        return self.item_for_label(str((self.rows.get(ident) or {}).get(cat_key) or "") if cat_key else "")

    def item_for_label(self, label: str) -> int | None:
        """The requested item a category label names: the item's store label, else the one item sharing its long words."""
        if not label:
            return 1 if len(self.items) == 1 else None
        words = _words(label)
        for number, item in enumerate(self.items, 1):
            if item.shelf and _words(item.shelf) == words:
                return number
        long = {w for w in _stem(label) if len(w) >= 4}
        hits = [n for n, item in enumerate(self.items, 1) if long & {w for w in _stem(item.shelf or "") | _stem(item.name) if len(w) >= 4}]
        return hits[0] if len(hits) == 1 else None

    def sheet_for(self, ident: str, number: int) -> list[tuple[str, str, str]]:
        record = self.full.get(ident) or self.rows.get(ident) or {}
        needs = [need for n, need in self.needs_now() if n == number]
        opened = ident in self.full
        sheet = FitJudge.sheet(needs, record, {need: key for need in needs if (key := self.deciding_key(number, need))}, 1 if opened else 2)
        cat_key = self.category_key()
        label = _tokens(str(record.get(cat_key) or "")) if cat_key else frozenset()
        out = []
        for need, verdict, where in sheet:
            if verdict in ("differs", "silent", "elsewhere") and label and not _READ_ME.search(FitJudge.split(need)[1].lower()) \
                    and any(choice and choice <= label for choice in FitJudge.choices(FitJudge.split(need)[1])):
                # the listing's category label states the value: the category meets it, whatever a key of its own says
                out.append((need, "meets", f"category {record.get(cat_key)}"))
                continue
            # the store records the property as an attribute of this kind: a listing without it does not show the value
            key = self.keyed_kind(number, FitJudge.split(need)[0]) if opened and verdict in ("elsewhere", "silent") else None
            out.append((need, "keyless", f"no {key} attribute, which other listings of its kind carry") if key else (need, verdict, where))
        return out

    def keyed_kind(self, number: int, prop: str) -> str | None:
        """A key naming the property that two or more opened listings of the item's kind carry, else None."""
        if not prop:
            return None
        mark = ("keyed", number, prop, len(self.full))
        if mark in self.decider_cache:
            return self.decider_cache[mark]
        counts: dict[str, int] = {}
        for ident, record in self.full.items():
            if self.item_of(ident) != number:
                continue
            for key in {_words(k) for k in (_fact_block(record, 1) or {}) if _key_names(prop, k)}:
                counts[key] = counts.get(key, 0) + 1
        best = max(counts, key=lambda k: counts[k]) if counts else None
        self.decider_cache[mark] = best if best and counts[best] >= 2 else None
        return self.decider_cache[mark]

    def deciding_key(self, number: int, need: str) -> str | None:
        """When the item's listings carry several keys naming a requirement's property: the keys that hold the wanted value
        on some opened listing of the item are the ones whose values are of that kind. The plain key decides when it is one
        of them, else the one holding it most often; None when one key or none holds it (every naming key then counts)."""
        prop, want = FitJudge.split(need)
        if not prop or _READ_ME.search(want.lower()):
            return None
        mark = (number, need, len(self.full))
        if mark in self.decider_cache:
            return self.decider_cache[mark]
        allowed = FitJudge.choices(want)
        hits: dict[str, int] = {}
        for ident in self.seen_order:
            if ident not in self.full or self.item_of(ident) != number:
                continue
            for key, value in (_fact_block(self.full[ident]) or {}).items():
                if _key_names(prop, key):
                    name = _words(key)
                    hits[name] = hits.get(name, 0) + (1 if any(c <= o for o in FitJudge.options(value) for c in allowed) else 0)
        holding = [k for k, n in hits.items() if n > 0]
        pick = None
        if len(hits) >= 2 and holding:
            pick = _words(prop) if _words(prop) in holding else max(holding, key=lambda k: hits[k])
        self.decider_cache[mark] = pick
        return pick

    def judged_line(self, ident: str) -> str:
        number = self.item_of(ident)
        if number is None or not [1 for n, _ in self.needs_now() if n == number]:
            return ""
        tag = f"(item {number}) " if len(self.items) > 1 else ""
        return tag + FitJudge.render(ident, self.sheet_for(ident, number))

    def judged_block(self, ids: list[str]) -> str:
        lines = [self.judged_line(i) for i in ids]
        lines = [line for line in lines if line]
        return ("\n\nCOMPUTED EVIDENCE (code located each requirement on each record; it does not judge synonyms, units or numbers, so read TO READ and NOT STATED yourself "
                "and overrule a line only when the record clearly says otherwise):\n" + "\n".join(lines)) if lines else ""

    def clear_fits(self, number: int) -> list[str]:
        """Opened, buyable, affordable listings of the item's kind on which every requirement in force is met or stated."""
        item = self.items[number - 1]
        out = []
        for ident in reversed(self.seen_order):
            if ident not in self.full or self.item_of(ident) != number or not self.buyable(ident):
                continue
            price = self.price_of(ident)
            if isinstance(self.budget, (int, float)) and price is not None and price * item.qty > self.budget:
                continue
            if FitJudge.clear(self.sheet_for(ident, number)):
                out.append(ident)
        return out

    def plausible_fits(self, number: int) -> list[str]:
        """Opened, buyable, affordable listings of the item's kind on which the fit judge finds no requirement contradicted
        and at least as many met as unstated; most requirements met first (for a cheapest pick, the cheaper among equals)."""
        item = self.items[number - 1]
        ranked = []
        for order, ident in enumerate(reversed(self.seen_order)):
            if ident not in self.full or self.item_of(ident) != number or not self.buyable(ident):
                continue
            price = self.price_of(ident)
            if isinstance(self.budget, (int, float)) and price is not None and price * item.qty > self.budget:
                continue
            sheet = self.sheet_for(ident, number)
            if FitJudge.broken(sheet):
                continue
            met = sum(1 for _, v, _ in sheet if v in ("meets", "elsewhere"))
            silent = sum(1 for _, v, _ in sheet if v in ("silent", "keyless"))
            if silent > met:
                continue
            cheap = price if self.brief.get("pick") == "cheapest" and price is not None else 0.0
            ranked.append((-met, silent, cheap, order, ident))
        return [row[-1] for row in sorted(ranked)]

    # ---------- item tracker ----------
    def lines_by_item(self) -> dict[int, list[str]]:
        """Cart lines that are still buyable, grouped by the requested item their category names."""
        out: dict[int, list[str]] = {}
        for ident in self.cart_ids():
            number = self.item_of(ident)
            if number is not None and not self.sold_out(ident):
                out.setdefault(number, []).append(ident)
        return out

    def kind_seen(self, number: int) -> int:
        return sum(1 for ident in self.seen_order if self.item_of(ident) == number)

    def item_board(self) -> str:
        """One line per requested item: its cart line, the listings the fit judge clears for it, how many of its kind were seen."""
        if len(self.items) < 2:
            return ""
        held = self.lines_by_item()
        parts = []
        for number, item in enumerate(self.items, 1):
            fits = [i for i in self.clear_fits(number) if i not in held.get(number, [])][:3]
            state = ("in cart: " + ", ".join(held[number])) if number in held else "no cart line"
            parts.append(f"{number} {item.name}: {state}; cleared by the fit judge: {', '.join(fits) or 'none'}; seen of its kind: {self.kind_seen(number)}")
        return "\n[items] " + " | ".join(parts)

    def apply_cart(self, calls: list[dict[str, Any]]) -> list[str] | None:
        """Run cart calls Kyo planned itself; None when one cannot be shaped or the store does not answer."""
        cart = self.roles.get("cart")
        done = []
        for args in calls:
            faults: list[str] = []
            args = _shape(args, self.schemas.get(cart), "args", faults)
            if faults or not cart:
                return None
            reply = self.act(cart, args)
            if reply is None or self.finished:
                return None
            self.note_cart(cart, args)
            self.observe(cart, reply.get("observation"))
            done.append(_dump(args))
        return done

    def cheaper_swap(self, swaps: list[tuple[str, str]]) -> str:
        """Make the cheaper swap in code when the cart check passes the cart it leads to; empty when it cannot."""
        plan = {i: self.cart_lines.get(i, 1) for i in self.cart_ids()}
        drop = []
        for old, new in swaps:
            number = self.item_of(old) or 1
            plan[new] = plan.pop(old, self.items[number - 1].qty)
            drop.append(old)
        calls = self.cart_calls(plan, drop)
        if not calls or len(calls) + 2 > self.left():
            return ""
        self.fill_facts(list(plan), len(calls) + 3)
        self.check_cart(list(plan), self.hard)
        if not self.verdicts.get(self.check_key(list(plan))):
            return ""
        done = self.apply_cart(calls)
        return ("; ".join(done)) if done else ""

    def cheaper_cleared(self) -> list[tuple[str, str]]:
        """For a cheapest pick: cart lines that an opened, buyable, affordable listing of the same item undercuts while a key
        naming each requirement shows its value (every requirement MEETS; nothing left to wording elsewhere)."""
        if self.brief.get("pick") != "cheapest":
            return []
        cart = self.cart_ids()
        swaps = []
        for line in cart:
            number, price = self.item_of(line), self.price_of(line)
            if number is None or price is None:
                continue
            item = self.items[number - 1]
            pool = []
            for ident in self.seen_order:
                if ident in cart or ident not in self.full or self.item_of(ident) != number or not self.buyable(ident):
                    continue
                cost = self.price_of(ident)
                if cost is None or cost >= price or (isinstance(self.budget, (int, float)) and cost * item.qty > self.budget):
                    continue
                sheet = self.sheet_for(ident, number)
                if sheet and all(v == "meets" for _, v, _ in sheet):
                    pool.append((cost, ident))
            if pool:
                swaps.append((line, min(pool)[1]))
        return swaps

    def fill_missing(self) -> None:
        """The tool loop ended (dollars, seconds or turns) before the cart held every requested item: Kyo sets the cart to
        the rescue plan (cart lines kept, missing items filled with the listing the fit judge clears or finds most
        plausible) without a model call, so the finish orders instead of ending without a purchase."""
        held = self.lines_by_item()
        if not self.roles.get("cart") or self.left() < 3 or all(held.get(n) for n in range(1, len(self.items) + 1)):
            return
        plan = self.rescue_plan()
        if not plan:
            return
        calls = self.cart_calls(plan, [i for i in self.cart_ids() if i not in plan])
        if not calls or len(calls) + 1 >= self.left():
            return
        done = self.apply_cart(calls)
        self.trace.append({"kind": "fill_missing", "plan": plan, "done": bool(done)})

    def decline_guard(self) -> str:
        """Before an ending without a purchase: hold it once when code reading finds a fitting listing for every item,
        or ask once for a browse of an item's own category when nothing of that kind fits yet."""
        self.map_shelves()
        self.align_musts()
        held = self.lines_by_item()
        plan: list[tuple[int, str, bool]] = []
        empty: list[int] = []
        for number in range(1, len(self.items) + 1):
            if held.get(number):
                plan.append((number, held[number][0], True))
                continue
            fits = self.clear_fits(number)
            if fits:
                plan.append((number, fits[0], False))
            else:
                empty.append(number)
        to_add = [p for p in plan if not p[2]]
        if not empty and not self.fit_guarded and self.left() >= len(to_add) + 2:
            self.fit_guarded = True
            self.trace.append({"kind": "decline_guard_fit", "plan": plan})
            many = len(self.items) > 1
            said = "; ".join((f"item {n}: " if many else "") + f"{i}" + (" (already in the cart)" if inside else "") for n, i, inside in plan)
            return ("Reading the records shows a listing that meets every requirement in force for "
                    + ("each requested item" if many else "the request") + f": {said}. "
                    + ("Put " + ", ".join(i for _, i, inside in plan if not inside) + " in the cart at the requested quantities, then check out. " if to_add else "Check out the cart. ")
                    + "End without a purchase only if one of these clearly breaks a requirement: name the requirement and the record line in your text, then send the ending call again.")
        open_kinds = [n for n in empty if n not in self.browse_asked]
        if open_kinds and self.left() > 6:
            number = open_kinds[0]
            self.browse_asked.add(number)
            item = self.items[number - 1]
            label = item.shelf or item.name
            self.trace.append({"kind": "decline_guard_browse", "item": number})
            return (f"Nothing seen so far clears every requirement of {item.name}. Before ending, browse its own store category ({label!r}) as the store spells it, "
                    "with the category filter when the search tool has one, else the label itself as the query, and page through the results with the store's next-page value; "
                    "open the listings whose rows could fit. End without a purchase only after that.")
        return ""

    # ---------- checks ----------
    def cart_text(self) -> str:
        if self.cart_view is not None and self.cart_seen_at >= self.cart_at:
            return _dump(_clip(self.cart_view, 300))[:3000]
        # a cart view older than the last cart change would show lines that are gone
        return "lines by Kyo's record of the cart calls (listing: units): " + _dump(self.cart_lines)

    def empty_checkout(self, name: str, args: dict[str, Any]) -> str:
        """A checkout of an empty cart is refused by the store and costs a step: hold it and say which listings fit.
        A buy call that names a listing itself (a store that sells without a cart) is left alone."""
        if not self.is_buy(name, args) or self.cart_ids() or not self.roles.get("cart"):
            return ""
        if any(isinstance(v, str) and v in self.rows for v in args.values()):
            return ""
        self.empty_holds += 1
        if self.empty_holds > 3:
            return ""
        picks = []
        for number in range(1, len(self.items) + 1):
            found = self.clear_fits(number) or self.plausible_fits(number)
            if found:
                picks.append((f"item {number} ({self.items[number - 1].name}): " if len(self.items) > 1 else "") + found[0])
        self.trace.append({"kind": "empty_checkout_held", "picks": picks})
        return ("Not run (no step used): the cart is empty, and the store refuses a checkout of an empty cart (the step is lost). "
                + ("Listings that appear to fit: " + "; ".join(picks) + ". " if picks else "")
                + "Put the listing that fits each requested item in the cart at the requested quantity, then check out; "
                "end without a purchase only if no listing fits.")

    def code_search(self, text: str, label: str | None = None, plain: bool = False) -> list[str]:
        """One keyword search Kyo makes itself, with the search tool's text parameter: its required string parameter, else
        the one the agent's own last search filled with the most words, else its only one. With `label`, the store's
        category parameter (when the search tool takes one) is set to it. Returns the ids of the rows shown."""
        tool = self.roles.get("search")
        schema = self.schemas.get(tool or "") or {}
        props = schema.get("properties") or {}
        shelf = (self.browse or {}).get("label") if (self.browse or {}).get("tool") == tool else None
        # the store's category parameter never takes the words
        texts = [k for k, v in props.items() if isinstance(v, dict) and "string" in _kinds(v) and "enum" not in v and k != shelf]
        typed = sorted((k for k in texts if isinstance((self.search_seen or {}).get(k), str)), key=lambda k: -len(str(self.search_seen[k]).split()))
        param = (next((k for k in texts if k in (schema.get("required") or [])), None) or (typed[0] if typed else None)
                 or (texts[0] if len(texts) == 1 else None))
        if not tool or not param or self.finished or self.left() <= self.tail() + 2:
            return []
        faults: list[str] = []
        # the other arguments of the agent's own last search of that tool (a mode, a section) go along; paging starts over
        base = {k: v for k, v in (self.search_seen or {}).items() if k != param and (isinstance(v, bool) or not isinstance(v, (int, float)))}
        if label and shelf and shelf in props:
            base[shelf] = label
        elif plain and shelf and shelf not in (schema.get("required") or []):
            # a look-up for a listing of unknown category: the last search's category would hide it
            base.pop(shelf, None)
        args = _shape({**base, param: text}, schema, "args", faults)
        if faults:
            return []
        reply = self.act(tool, args)
        if reply is None:
            return []
        seen = reply.get("observation")
        self.observe(tool, seen)
        rows = [i for i, _ in self.records_in(seen)]
        self.trace.append({"kind": "code_search", "text": text[:80], "rows": len(rows)})
        return rows

    def link_rows(self, shop: str, words: str, told: str = "") -> list[str]:
        """Unopened rows that may be a chain's next listing, named by the shop that sells it: rows whose fields hold the
        shop's name exactly (seen already, else shown by a search for the shop with the link's words, then the bare
        name, in every category), most words of the link's description (and of the requirement's own text) first."""
        name = _words(shop)
        if not name:
            return []

        def carries(ident: str) -> bool:
            return any(isinstance(v, str) and _words(v) == name for v in (self.rows.get(ident) or {}).values())

        def pool() -> list[str]:
            return [i for i in self.seen_order if i not in self.opened and carries(i)]

        if not pool() and name not in self.link_searched:
            self.link_searched.add(name)
            # the link's words only when they are words of the requirement or of a page read (not the reader's prose)
            said = _tokens(told) | {w for r in self.full.values() for w in _tokens(_dump(r)[:600])} if told or self.full else set()
            kind = " ".join(w for w in str(words).split()[:6] if _tokens(w) and _tokens(w) <= said)[:60]
            for text in ([f"{shop} {kind}".strip()] if kind else []) + [shop]:
                self.code_search(text, plain=True)
                if pool():
                    break
        want = _tokens(words) | _tokens(told)
        return sorted(pool(), key=lambda i: -len(want & _tokens(" ".join(str(v) for v in (self.rows.get(i) or {}).values() if isinstance(v, str)))))

    def texts_of(self, record: dict[str, Any]) -> list[str]:
        return [v for v in record.values() if isinstance(v, str) and v.strip()]

    def part_families(self) -> set[str]:
        """Letter prefixes of codes that two or more listings carry as their own, in running text such as a title: the
        store's part numbers. A family seen only in code sentences ("Fits AB120.") is a model that parts pair with."""
        if self.family_memo[0] == len(self.rows):
            return self.family_memo[1]
        seen: dict[str, set[str]] = {}
        for ident, record in self.rows.items():
            for text in self.texts_of(record):
                lined = {c for named, barred in _code_sentences(text) for c in named + barred}
                for code in set(_CODE.findall(text)) - lined:
                    seen.setdefault(_family(code), set()).add(ident)
        families = {f for f, ids in seen.items() if len(ids) >= 2}
        self.family_memo = (len(self.rows), families)
        return families

    def codes_of(self, ident: str) -> tuple[set[str], set[str], set[str]]:
        """(own codes, codes it pairs by, parts it sends the reader to). Own codes stand in running text (its title); the
        codes it pairs by are those of its code sentences, apart from a sentence about another part number, which says
        what THAT part fits ("Part XY123 also fits: AB120") or, naming no model, sends the reader to that part."""
        record = self.full.get(ident) or self.rows.get(ident) or {}
        size = len(_dump(record))
        memo = self.code_memo.get(ident)
        families = self.part_families()
        if memo and memo[0] == size + len(families):
            return memo[1]
        own: set[str] = set()
        lines: list[list[str]] = []
        barred: set[str] = set()
        for text in self.texts_of(record):
            found = _code_sentences(text)
            own |= set(_CODE.findall(text)) - {c for named, bars in found for c in named + bars}
            # "(part AB1234)" or "(AB1234)": the listing names its own number, even in a title short enough to read as a code line
            own |= set(re.findall(r"\(\s*(?:[A-Za-z]+\s+)?([A-Z]{2,5}-?\d{3,6})\s*\)", text))
            lines += [named for named, _ in found if named]
            barred |= {c for _, bars in found for c in bars}
        self.code_bars[ident] = barred
        pairs: set[str] = set()
        refs: set[str] = set()
        for codes in lines:
            # only plain codes can send the reader to a part; a name with a number or a run of models is a model
            other = [c for c in codes if _CODE.fullmatch(c) and _family(c) in families and c not in own]
            if other:
                if all(_CODE.fullmatch(c) and _family(c) in families for c in codes):
                    refs.update(other)
                continue
            pairs.update(c for c in codes if c not in own)
        out = (own, pairs, refs)
        self.code_memo[ident] = (size + len(families), out)
        return out

    def backs(self, a: str, b: str) -> bool:
        """Two listings' own pages pair them: one names a code the other carries (a run of models holding it counts), or
        both name the same model; never when one page leaves out a code the other carries ("except AB120")."""
        own_a, pairs_a, _ = self.codes_of(a)
        own_b, pairs_b, _ = self.codes_of(b)
        carry_a = {c for c in own_a | pairs_a if not _CODE_RUN.fullmatch(c)}
        carry_b = {c for c in own_b | pairs_b if not _CODE_RUN.fullmatch(c)}
        if carry_b & self.code_bars.get(a, set()) or carry_a & self.code_bars.get(b, set()):
            return False
        # two parts naming the same run of models (each "fits every AB12xx model") work with the same model
        return bool(carry_a & carry_b & (pairs_a | pairs_b) or _code_hit(carry_b, pairs_a) or _code_hit(carry_a, pairs_b)
                    or _runs_meet(pairs_a, pairs_b))

    def coded(self, candidates: dict[int, list[str]]) -> bool:
        """The store pairs these items by codes: every paired item has a candidate whose page names codes it pairs by."""
        # only plain codes and runs of them make a store one that pairs by codes; names with a number alone leave it to the model
        return bool(candidates) and all(any(any(_CODE.fullmatch(c) or _CODE_RUN.fullmatch(c) for c in self.codes_of(i)[1]) for i in ids)
                                        for ids in candidates.values())

    def code_sets(self, candidates: dict[int, list[str]]) -> list[dict[int, str]]:
        """Every combination of one candidate per paired item in which each requested pair is backed by codes (at most 60)."""
        numbers = sorted(candidates)
        out: list[dict[int, str]] = []

        def grow(at: int, chosen: dict[int, str]) -> None:
            if len(out) >= 60:
                return
            if at == len(numbers):
                out.append(dict(chosen))
                return
            number = numbers[at]
            for ident in candidates[number]:
                chosen[number] = ident
                if all(self.backs(chosen[a], chosen[b]) for a, b in self.fit_pairs if a in chosen and b in chosen):
                    grow(at + 1, chosen)
                del chosen[number]

        grow(0, {})
        return out

    def pairing_rows(self, keep: dict[int, str]) -> list[str]:
        """Unopened rows of a paired item whose codes pair with the line kept for its partner, newest first (three at most)."""
        out = []
        for a, b in self.fit_pairs:
            for mine, other in ((a, b), (b, a)):
                partner = keep.get(other)
                if not partner:
                    continue
                for ident in reversed(self.seen_order):
                    if ident not in self.full and ident not in out and self.buyable(ident) and self.item_of(ident) == mine \
                            and self.backs(ident, partner):
                        out.append(ident)
        return out[:3]

    def part_trail(self) -> None:
        """Before a checkout: a cart line whose page sends the reader to another part by its number ("more on part
        XY123") has that number searched in the line's own category, as shoppers who pass do; the order counts only
        after it. Four look-ups at most, never into the last steps."""
        cat_key = self.category_key()
        for ident in self.cart_ids():
            for code in sorted(self.codes_of(ident)[2]):
                if code in self.parts_looked or len(self.parts_looked) >= 4 or self.finished or self.left() < self.tail() + 4 + len(self.cart_ids()):
                    continue
                self.parts_looked.add(code)
                label = str((self.rows.get(ident) or {}).get(cat_key) or "") if cat_key else ""
                rows = self.code_search(code, label or None)
                self.trace.append({"kind": "part_trail", "line": ident, "code": code, "rows": len(rows)})

    def fit_candidates(self, keep: dict[int, str] | None = None) -> dict[int, list[str]]:
        """Per item in a fit pair: its kept line first, then opened listings the fit judge clears, then plausible ones (six at most)."""
        out: dict[int, list[str]] = {}
        for number in sorted({n for pair in self.fit_pairs for n in pair}):
            pool = ([keep[number]] if keep and keep.get(number) else []) + self.cleared(number) + self.plausible_fits(number)
            out[number] = [i for i in dict.fromkeys(pool) if i in self.full and self.buyable(i)][:6]
        return out

    def fit_sets(self, candidates: dict[int, list[str]]) -> tuple[list[dict[int, str]], str]:
        """Combinations of one candidate per paired item whose own words back every pair, read by the light model."""
        if not self.fit_pairs or any(not ids for ids in candidates.values()):
            return [], ""
        if self.coded(candidates):
            # the pages pair by codes: code decides, the model's word is not needed (and may pair unbacked lines)
            sets = self.code_sets(candidates)
            need = ""
            if not sets:
                wanted = sorted({c for ids in candidates.values() for c in self.codes_of(ids[0])[1]})[:8]
                need = "a listing whose page names one of the codes the other lines pair by (" + ", ".join(wanted) + ")"
            self.trace.append({"kind": "fit_sets_codes", "sets": sets[:4], "count": len(sets)})
            return sets, need
        if self.spent_out():
            return [], ""
        blocks = []
        for number, ids in candidates.items():
            blocks.append(f"item {number} ({self.items[number - 1].name}):")
            for ident in ids:
                record = self.full.get(ident) or self.rows.get(ident) or {}
                blocks.append(f"  - {ident}: " + _dump(_clip(record, 160))[:700])
        prompt = (COMPAT + "\n\nPAIRS:\n" + "\n".join(f"- item {a} with item {b}" for a, b in self.fit_pairs)
                  + "\n\nCANDIDATES:\n" + "\n".join(blocks))
        got = self.ask_json(prompt, [LIGHT, HEAVY], 1500) or {}
        sets = []
        for entry in _items(got.get("sets")):
            lines = entry.get("lines") if isinstance(entry, dict) and isinstance(entry.get("lines"), dict) else {}
            picked = {int(m.group()): str(v).strip() for k, v in lines.items() for m in [re.search(r"\d+", str(k))] if m}
            if all(picked.get(n) in candidates[n] for n in candidates):
                sets.append({n: picked[n] for n in candidates})
        self.trace.append({"kind": "fit_sets", "sets": sets[:4], "need": _clip(got.get("need"), 80)})
        return sets, str(got.get("need") or "").strip()

    def best_set(self, sets: list[dict[int, str]], keep: dict[int, str]) -> dict[int, str] | None:
        """The cheapest backed set for a cheapest pick, else the one keeping most of the current lines (then the first)."""
        if not sets:
            return None
        def total(chosen: dict[int, str]) -> float:
            return sum((self.price_of(i) or 0.0) * self.items[n - 1].qty for n, i in chosen.items())
        if self.brief.get("pick") == "cheapest":
            return min(sets, key=total)
        return max(sets, key=lambda chosen: sum(1 for n, i in chosen.items() if keep.get(n) == i))

    def fit_plan(self, plan: dict[str, int] | None) -> dict[str, int] | None:
        """A per-item plan made to work together: when the request pairs items, the paired lines are replaced by the best
        backed set (the plan's own lines kept where they belong to one); unchanged when nothing backs a set."""
        if not plan or not self.fit_pairs:
            return plan
        keep = {self.item_of(i): i for i in plan if self.item_of(i)}
        sets, _ = self.fit_sets(self.fit_candidates(keep))
        chosen = self.best_set(sets, keep)
        if not chosen:
            return plan
        out = {i: q for i, q in plan.items() if self.item_of(i) not in chosen}
        for number, ident in chosen.items():
            out[ident] = self.items[number - 1].qty
        return out

    def fit_hold(self) -> str:
        """Before a checkout when the request pairs items: the cart's paired lines must be backed by their own words as
        working together. Otherwise Kyo sets the cart to the best backed set it can find among opened listings, or holds
        the checkout once, naming what the missing piece must carry."""
        if not self.fit_pairs or self.left() < 4:
            return ""
        mark = self.cart_mark() + f"#{self.version}"
        if mark in self.fit_marks:
            return ""
        self.fit_marks.add(mark)
        held = self.lines_by_item()
        keep = {n: held[n][0] for n in held if held.get(n)}
        if any(n not in keep for pair in self.fit_pairs for n in pair):
            return ""
        candidates = self.fit_candidates(keep)
        sets, need = self.fit_sets(candidates)
        if any(all(keep.get(n) == i for n, i in chosen.items()) for chosen in sets):
            return ""
        chosen = self.best_set(sets, keep)
        if not chosen and self.coded(candidates) and self.left() >= 8:
            # rows already shown whose codes pair with a kept line: read them before holding, then try the sets again
            rows = self.pairing_rows(keep)
            if rows and self.open_rows(rows):
                sets, need = self.fit_sets(self.fit_candidates(keep))
                if any(all(keep.get(n) == i for n, i in chosen.items()) for chosen in sets):
                    return ""
                chosen = self.best_set(sets, keep)
        if not chosen:
            return ("Not run yet (no step used). The items the request wants to work together are not backed by their own pages:"
                    " no opened listing of one names a model, series or part the other carries" + (f" ({need})" if need else "") + "."
                    " Search for the missing piece by that code or by the other item's model (when a search for a code finds nothing, search the"
                    " piece by its kind and brand words and read the codes in the rows' texts), open the rows that could fit, then check out.")
        plan = {i: self.cart_lines.get(i, 1) for i in self.cart_ids() if not self.sold_out(i)}
        for number, ident in chosen.items():
            plan.pop(keep.get(number, ""), None)
            plan[ident] = self.items[number - 1].qty
        drop = [i for i in self.cart_ids() if i not in plan]
        calls = self.cart_calls(plan, drop)
        named = "; ".join(f"item {n}: {i}" for n, i in chosen.items())
        if calls and len(calls) + 2 <= self.left():
            done = self.apply_cart(calls)
            if done:
                self.trace.append({"kind": "fit_swap", "set": chosen})
                return (f"Not run yet. The cart's lines are not backed as working together, while these are by their own pages ({named}),"
                        " so Kyo set the cart to them. Check out now, unless one of them clearly breaks a requirement.")
        return f"Not run yet (no step used). Put this set in the cart instead, as its pages back each pair: {named}. Then check out."

    def keys_of(self) -> dict[int, list[str]]:
        """The attribute keys carried by each requested item's opened listings (by the item its category belongs to)."""
        out: dict[int, list[str]] = {}
        for label, name, *_ in self.key_table():
            number = self.item_for_label(label) or (1 if len(self.items) == 1 else None)
            if number:
                out.setdefault(number, []).append(name)
        return out

    def align_musts(self) -> None:
        """Requirements that may have been filed under a neighbouring word (a unit taken for the property) are checked once
        against the request and the item's keys (three or more read): those whose property names no key, and every one
        holding a figure. The careful model names the key each one constrains; one it moves is filed under that key."""
        if self.aligned or self.spent_out():
            return
        keys = self.keys_of()
        pointed = {need for _, need in self.refs}
        odd = []
        for number, item in enumerate(self.items, 1):
            listed = keys.get(number) or []
            if len(listed) < 3:
                continue
            for must in item.musts:
                prop = must.split(":", 1)[0].strip()
                if ":" not in must or must in pointed or not prop or must in item.dropped:
                    continue
                figure = bool(re.search(r"\d", must.split(":", 1)[1]))
                if figure or not any(_key_names(prop, k) or _key_names(k, prop) for k in listed):
                    odd.append((number, must))
        if not odd:
            return
        self.aligned = True
        prompt = (ALIGN + "\n\nREQUEST:\n" + self.request + "\n\nREQUIREMENTS:\n"
                  + "\n".join(f"- item {n} ({self.items[n - 1].name}): {m} | listed keys: " + ", ".join(keys.get(n, [])[:25]) for n, m in odd))
        got = self.ask_json(prompt, self.judges(), 1200) or {}
        for entry in _items(got.get("map")):
            if not isinstance(entry, dict) or not isinstance(entry.get("key"), str):
                continue
            hit = next(((n, m) for n, m in odd if m == str(entry.get("must") or "").strip()), None)
            if not hit or entry["key"] not in (keys.get(hit[0]) or []):
                continue
            number, must = hit
            item = self.items[number - 1]
            if _stem(entry["key"]) == _stem(must.split(":", 1)[0]):
                continue
            renamed = _spoken(entry["key"]) + ":" + must.split(":", 1)[1]
            if renamed not in item.musts and must in item.musts:
                item.musts[item.musts.index(must)] = renamed
                self.version += 1
                self.trace.append({"kind": "aligned", "item": number, "from": must, "to": renamed})

    def map_shelves(self) -> None:
        """Requested items that no category label seen in results names (by the item's shelf or shared words) get one from
        the light model, which reads the labels with a few titles each; only a label seen in results is taken, and the
        store's spelling becomes the item's shelf. Runs again (twice at most) when new labels have shown up."""
        cat = self.category_key()
        if not cat or self.shelf_reads >= 2 or self.spent_out():
            return
        seen: dict[str, list[str]] = {}
        for ident in self.seen_order:
            row = self.rows.get(ident) or {}
            label = row.get(cat)
            if isinstance(label, str) and label.strip():
                title = next((v for k, v in row.items() if k != cat and isinstance(v, str) and len(v.split()) >= 3), "")
                seen.setdefault(label.strip(), []).append(title[:70])
        missing = [n for n in range(1, len(self.items) + 1) if not any(self.item_for_label(label) == n for label in seen)]
        if not missing or not seen or set(seen) <= self.shelf_seen:
            return
        self.shelf_reads += 1
        self.shelf_seen = set(seen)
        prompt = (SHELVES + "\n\nITEMS:\n" + "\n".join(f"item {n}: {item.name}" for n, item in enumerate(self.items, 1))
                  + "\n\nLABELS:\n" + "\n".join(f"- {label}: " + " | ".join(list(dict.fromkeys(t for t in titles if t))[:3]) for label, titles in
                                                 sorted(seen.items(), key=lambda kv: -len(kv[1]))[:20]))
        got = self.ask_json(prompt, [LIGHT, HEAVY], 800) or {}
        for entry in _items(got.get("items")):
            if not isinstance(entry, dict):
                continue
            number = _count(entry.get("item"))
            label = entry.get("label")
            owner = self.item_for_label(label.strip()) if isinstance(label, str) else None
            if number in missing and isinstance(label, str) and label.strip() in seen and owner in (None, number):
                self.items[number - 1].shelf = label.strip()
                self.trace.append({"kind": "shelf", "item": number, "label": label.strip()})

    def browse_kinds(self) -> None:
        """Before the interview, while steps are plentiful: each item's own store category, by a label seen in results,
        is listed once with the browse tool, so the listings read for the key counts are not all from one keyword search."""
        spec = self.browse
        cat = self.category_key()
        if not spec or not cat or self.browse_empty >= 2 or self.max_steps < 50:
            return
        labels = {str((self.rows.get(i) or {}).get(cat) or "").strip() for i in self.seen_order} - {""}
        for number in range(1, len(self.items) + 1):
            label = self.kind_label(number)
            if not label or label not in labels or label.lower() in self.browsed or self.left() <= 24 or self.finished:
                continue
            self.browsed.add(label.lower())
            faults: list[str] = []
            args = _shape({**(spec.get("with") or {}), spec["label"]: label}, self.schemas.get(spec["tool"]), "args", faults)
            if faults:
                continue
            reply = self.act(spec["tool"], args)
            if reply is None:
                break
            seen = reply.get("observation")
            self.observe(spec["tool"], seen)
            if not self.records_in(seen) and not self.records_in(next((x for x in (seen or {}).values() if isinstance(x, dict)), {}) if isinstance(seen, dict) else None):
                self.browse_empty += 1
            self.trace.append({"kind": "browse_kind", "item": number, "label": label})

    def kind_label(self, number: int) -> str:
        """The store's own label for an item's category: read off a held line or a fitting listing, else the brief's shelf."""
        cat = self.category_key()
        lost = [i for i in self.gone if self.item_of(i) == number]
        opened = [i for i in reversed(self.seen_order) if i in self.full and self.item_of(i) == number]
        for ident in (self.lines_by_item().get(number) or []) + lost + self.clear_fits(number) + self.plausible_fits(number) + opened:
            value = (self.rows.get(ident) or {}).get(cat) if cat else None
            if isinstance(value, str) and value.strip():
                return value.strip()
        return self.items[number - 1].shelf

    def visible_fit(self, ident: str, number: int) -> int:
        """How many of the item's requirement values a row's own text already shows (title, description and the like)."""
        words = _tokens(" ".join(str(v) for v in (self.rows.get(ident) or {}).values() if isinstance(v, str)))
        return sum(1 for n, need in self.needs_now() if n == number and (value := _tokens(FitJudge.split(need)[1])) and value <= words)

    def capped_scan(self, number: int, cap: float, opens: int = 2) -> list[str]:
        """List the item's own store category with the browse tool's price cap set to `cap` (once per category and cap),
        so cheaper listings that no keyword search showed come into view; then read the attributes of the unread rows of
        the item's kind at or below the cap that look most like fits, the cheaper first. Rows priced above the cap show
        the parameter caps nothing; it is not used again. Returns the ids of rows not seen before."""
        spec = self.browse
        room = self.left() - self.tail() - 2
        if self.finished or room < 1 or cap is None or cap < 1:
            return []
        fresh: list[str] = []
        label = self.kind_label(number)
        whole = "integer" in _kinds(((self.schemas.get(spec.get("tool") or "") or {}).get("properties") or {}).get(spec.get("cap") or "") or {})
        cap = int(cap) if whole else cap
        key = (label.lower(), cap)
        if spec.get("cap") and not self.cap_bad and label and key not in self.scanned:
            self.scanned.add(key)
            faults: list[str] = []
            args = _shape({**(spec.get("with") or {}), spec["label"]: label, spec["cap"]: cap}, self.schemas.get(spec["tool"]), "args", faults)
            reply = self.act(spec["tool"], args) if not faults else None
            if reply is not None:
                before = set(self.rows)
                seen = reply.get("observation")
                self.observe(spec["tool"], seen)
                rows = [i for i, _ in self.records_in(seen)]
                if any((self.price_of(i) or 0) > cap for i in rows):
                    self.cap_bad = True
                self.browse_empty += 0 if rows else 1
                if self.browse_empty >= 2:
                    # two empty listings: the brief's label or cap parameter is not what the store takes
                    self.cap_bad = True
                fresh = [i for i in rows if i not in before]
                room -= 1
        opens = max(0, min(opens, room))
        pool = [i for i in self.seen_order if i not in self.full and self.item_of(i) == number and self.buyable(i)
                and self.price_of(i) is not None and self.price_of(i) <= cap]
        pool.sort(key=lambda i: (-self.visible_fit(i, number), self.price_of(i)))
        read = self.fill_facts(pool[:opens], self.tail() + 2) if opens else []
        self.trace.append({"kind": "capped_scan", "item": number, "label": label, "cap": cap, "new": len(fresh), "read": len(read)})
        return fresh

    def cleared(self, number: int) -> list[str]:
        """Listings that meet every requirement in force for the item; once none is left in force (all dropped), every
        buyable, affordable listing of its kind."""
        if any(n == number for n, _ in self.needs_now()):
            return self.clear_fits(number)
        qty = self.items[number - 1].qty
        return [i for i in reversed(self.seen_order) if self.item_of(i) == number and self.buyable(i) and self.affordable(i, qty)]

    def affordable(self, ident: str, qty: int) -> bool:
        price = self.price_of(ident)
        return not isinstance(self.budget, (int, float)) or price is None or price * qty <= self.budget + 1e-9

    def change_plan(self) -> dict[str, int] | None:
        """One listing per requested item for the task as it now stands: a held line that is still buyable, affordable and
        not contradicted stays, unless the pick is the cheapest and a cleared listing costs less; otherwise the cheapest
        cleared listing (cheapest pick) or the first cleared one. None when some item has no cleared listing or no such
        held line, or when the lines together break the budget."""
        self.map_shelves()
        held = self.lines_by_item()
        cheapest = self.brief.get("pick") == "cheapest"
        plan: dict[str, int] = {}
        for number, item in enumerate(self.items, 1):
            clear = [i for i in self.cleared(number) if i not in plan]
            lost = [i for i in self.gone if self.item_of(i) == number]
            if cheapest:
                clear.sort(key=lambda i: (self.price_of(i) if self.price_of(i) is not None else float("inf"), -self.likeness(lost[-1], i) if lost else 0.0))
            elif lost:
                # a stand-in for a sold-out pick: the cleared listing most like it (it shares what made the pick right)
                clear.sort(key=lambda i: -self.likeness(lost[-1], i))
            keep = [i for i in held.get(number) or [] if not FitJudge.broken(self.sheet_for(i, number)) and self.affordable(i, item.qty)]
            cleared_keep = [i for i in keep if i in clear]
            pick = cleared_keep[0] if cleared_keep else (clear[0] if clear else (keep[0] if keep else None))
            if cheapest and clear and pick in keep and (self.price_of(clear[0]) or 0) < (self.price_of(pick) or 0):
                pick = clear[0]
            if not pick or pick in plan:
                return None
            plan[pick] = item.qty
        plan = self.fit_plan(plan) or plan
        prices = [self.price_of(i) for i in plan]
        if isinstance(self.budget, (int, float)) and self.brief.get("budget_scope") != "each":
            total = sum(p * n for p, n in zip(prices, plan.values())) if None not in prices else None
            if total is not None and self.budget + 1e-9 < total <= 10 * self.budget:
                return None
        return plan

    def likeness(self, a: str, b: str) -> float:
        """How much listing b looks like listing a: attribute values they share, then the words their titles share."""
        ra, rb = self.full.get(a) or self.rows.get(a) or {}, self.full.get(b) or self.rows.get(b) or {}
        fa, fb = _fact_block(ra) or {}, _fact_block(rb) or {}
        shared = sum(1 for k, v in fa.items() if any(_words(k) == _words(x) and _tokens(str(v)) == _tokens(str(w)) for x, w in fb.items()))

        def title(record: dict[str, Any]) -> frozenset[str]:
            texts = [v for v in record.values() if isinstance(v, str) and 3 <= len(v.split()) <= 40]
            return _tokens(max(texts, key=lambda t: len(t.split()))) if texts else frozenset()

        ta, tb = title(ra), title(rb)
        return shared + 2.0 * len(ta & tb) / max(1, len(ta | tb))

    def twin_rows(self, lost: str) -> None:
        """Before choosing a stand-in for a sold-out pick: one search for the lost listing's own title words brings the same
        product from other sellers into view, and the two rows that look most like it are read."""
        if lost in self.twinned or self.left() <= self.tail() + 5:
            return
        self.twinned.add(lost)
        record = self.full.get(lost) or self.rows.get(lost) or {}
        texts = [v for v in record.values() if isinstance(v, str) and 3 <= len(v.split()) <= 40]
        if not texts:
            return
        words = [w for w in re.findall(r"[A-Za-z][A-Za-z0-9'-]+", max(texts, key=lambda t: len(t.split()))) if len(w) > 2][:6]
        if len(words) < 2:
            return
        self.code_search(" ".join(words))
        number = self.item_of(lost)
        pool = [i for i in self.seen_order if i != lost and i not in self.full and self.item_of(i) == number and self.buyable(i)]
        pool.sort(key=lambda i: -self.likeness(lost, i))
        self.fill_facts(pool[:2], self.tail() + 3)

    def after_turn(self, stage: str, meaning: str, in_code: bool | None = None) -> str:
        """The store turned an ending down with a change from the shopper (a new budget, a requirement dropped or added).
        The holds and reviews that fire once are armed again; for a cheapest pick the item's category is listed again under
        a price cap; when every item then has a listing that meets each requirement now in force, Kyo sets the cart to
        them in code. Returns what was done, for the agent's next turn."""
        self.turns += 1
        self.turned_version = self.version
        self.turned_step = self.step
        if self.turns > 3 or self.finished:
            return ""
        self.wider_held = self.fit_guarded = self.rescued = self.pushed = self.late_held = False
        self.leads = []
        self.refuse_blocks = 0
        self.cheaper_marks.clear()
        self.cheaper_said.clear()
        self.browse_asked.clear()
        if self.note_sold:
            self.find_sold()
        for ident in [i for i in self.cart_ids() if self.sold_out(i)]:
            self.gone.setdefault(ident, self.step)
            self.twin_rows(ident)
        if self.brief.get("pick") == "cheapest":
            for number, item in enumerate(self.items, 1):
                best = [self.price_of(i) for i in self.cleared(number) + (self.lines_by_item().get(number) or []) if self.price_of(i) is not None]
                cap = _under(min(best)) if best else (self.budget / max(1, item.qty) if isinstance(self.budget, (int, float)) else None)
                if cap is not None and isinstance(self.budget, (int, float)):
                    cap = min(cap, self.budget / max(1, item.qty))
                if cap is not None:
                    self.capped_scan(number, cap)
                    # this re-scan is the wider look: the checkout that follows is not held for another one
                    self.wider_held = True
        plan = self.change_plan()
        self.trace.append({"kind": "after_turn", "stage": stage, "plan": plan})
        head = f"[Kyo] The store turned the {'checkout' if stage == 'order' else 'ending without a purchase'} down with a change ({meaning or 'see the note'})."
        if not plan:
            return (head + " No listing read so far meets every requirement now in force for each item: search again with the changed"
                    " requirements and budget (browse each item's category, with the price cap when the store has one), open the"
                    " listings that could fit, then check out; end without a purchase only when none fits.")
        names = "; ".join(f"{i} ({self.price_of(i):g})" if self.price_of(i) is not None else i for i in plan)
        drop = [i for i in self.cart_ids() if i not in plan]
        same = not drop and set(plan) == set(self.cart_ids()) and all(self.cart_lines.get(i, q) == q for i, q in plan.items())
        calls = [] if same else self.cart_calls(plan, drop)
        if in_code is None:
            # the agent itself acts on the change while it can (its own reading wins); code steps in when it cannot
            in_code = self.spent_out() or self.cost >= self.hard or self.loop_calls >= self.loop_cap or self.left() <= self.tail() + 2
        if not in_code and calls:
            return head + f" With the change, these appear to fit: {names}. Put them in the cart at the requested quantities (take sold-out lines out) and check out, unless one of them clearly breaks a requirement now in force."
        if calls is None or len(calls) + 2 > self.left():
            return head + f" Listings that meet every requirement now in force: {names}. Put them in the cart at the requested quantities and check out."
        done = self.apply_cart(calls) if calls else []
        if calls and not done:
            return head + f" Listings that meet every requirement now in force: {names}. Put them in the cart at the requested quantities and check out."
        verified = all(i in self.cleared(self.item_of(i) or 0) for i in plan)
        return (head + f" With the change, {names} " + ("meet" + ("s" if len(plan) == 1 else "") + " every requirement now in force" if verified else ("appear" + ("s" if len(plan) == 1 else "") + " to fit, with nothing in force contradicted"))
                + (", so Kyo set the cart to " + ("them" if len(plan) > 1 else "it") if calls else ", and the cart already holds " + ("them" if len(plan) > 1 else "it"))
                + ". Check out now; end without a purchase only if one of them clearly breaks a requirement now in force (name it).")

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
        if self.brief.get("pick") in ("cheapest", "most", "least"):
            prompt += "\n\nOTHER LISTINGS SEEN (newest first):\n" + self.others(ids, 15)
        prompt += self.judged_block(ids)
        # the heavy model while the episode is cheap, and once more for the first look at the real cart before a
        # checkout while a margin under the loop's cap is left; the light model reads the rest
        heavy = self.cost < self.soft or (not lead and not self.heavy_checked and self.cost < self.hard - 0.01)
        got = self.ask_json(prompt, [HEAVY, LIGHT] if heavy else [LIGHT, HEAVY], 1500)
        if heavy and not lead:
            self.heavy_checked = True
        self.trace.append({"kind": "check", "cart": ids, "got": got, "heavy": heavy})
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

    def trail(self, may_hold: bool = True) -> tuple[str, bool]:
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
        rounds = 0
        while pending and self.trails < 6 and self.cost < self.hard:
            self.trails += 1
            rounds += 1
            records = []
            for ident in reversed(self.seen_order):
                # an opened page is the page view's record (with the attributes when they were read too)
                record = (self.full.get(ident) or self.rows.get(ident)) if ident in self.opened else None
                if record:
                    records.append("OPENED " + _dump(_clip(record, 300))[:1400])
                else:
                    records.append("ROW " + _dump(_clip(self.rows.get(ident), 160))[:520])
                if len(records) >= 60:
                    break
            prompt = TRAIL + "\n\nREQUIREMENTS WITH A VALUE ON ANOTHER LISTING:\n" + "\n".join(pending)
            prompt += "\n\nRECORDS (newest first):\n" + "\n".join(records)
            state = (tuple(pending), len(self.opened), len(self.rows), self.version)
            got = self.trail_memo[1] if self.trail_memo and self.trail_memo[0] == state else (self.ask_json(prompt, [LIGHT, HEAVY], 1800) or {})
            self.trail_memo = (state, got)
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
                    link = next((l for l, i in zip(links, ids) if i not in self.rows), None) or {}
                    first = not any(i in self.rows for i in ids)
                    words = str(link.get("words") or link.get("about") or "")
                    # the chain's first link is described by the requirement itself: its words rank the shop's rows
                    found = self.link_rows(str(link.get("shop") or ""), words, need.partition(":")[2] if first else "")
                    if found:
                        reach += [i for i in found[:2] if i not in reach]
                    else:
                        about = str(link.get("about") or "") or "the listing it points at"
                        lost.append(f"{need} -> find {about}")
            answered = {str(c.get("need") or "").strip() for c in _items(got.get("chains")) if isinstance(c, dict)}
            if len(pending) > 1 or not _items(got.get("chains")):
                lost += [f"{p} -> find the listing it points at" for p in pending if p not in answered and p not in self.ref_values
                         and not any(p in entry for entry in lost)]
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
        # whatever stopped the walk (its rounds, the spend guard, the steps), a chain not read to its end is still to find
        if rounds or self.cost < self.soft:
            lost += [f"{p} -> find the listing it points at" for p in pending if p not in self.ref_values and not any(p in e for e in lost)]
        text = ""
        if seen:
            text += "[Kyo opened the pages a requirement points at]\n" + "\n".join(seen) + "\n"
        if self.ref_values:
            text += "[values read from the pages requirements point at] " + "; ".join(f"{k} -> {v}" for k, v in self.ref_values.items()) + "\n"
        hold = may_hold and bool(lost) and self.trail_blocks < 2 and self.left() > 5
        if hold:
            self.trail_blocks += 1
            text += ("The order counts only after every page a requirement points at has been opened in full (a search row is not enough), following each page that refers to a further listing. Still to find and open: "
                     + " | ".join(lost[:3]) + ". Search for it by its shop, brand or words, open it, apply the value its page gives, then check out.")
        return text, hold

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
        # a sold-out cart line without a replacement ends the episode in a refusal, so this runs up to the hard cap
        if self.cost > self.hard:
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
        prompt += self.judged_block([i for i in reversed(self.seen_order) if i in self.full and i not in lost and not self.sold_out(i)][:25])
        # listing the rows that still fit is reading, not weighing: the light model does it with little thinking
        # (enough room left for the answer), the heavy one when its reply does not parse
        got = self.ask_json(prompt, [LIGHT, HEAVY], 2000, "low")
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
        # the review is what stands between a wrong ending without a purchase and an order: up to the hard cap
        if self.audited or self.cost > self.hard:
            return ""
        self.audited = True
        got = self.audit_call()
        named = self.audit_fits(got)
        if not named:
            # rows whose page was never read can still fit: open the promising ones (one step) and look again
            rows = [str(i) for i in _items((got or {}).get("open")) if str(i) in self.rows and str(i) not in self.opened and not self.sold_out(str(i))]
            if rows and self.open_rows(rows, 3):
                got = self.audit_call()
                named = self.audit_fits(got)
        if not named:
            return ""
        self.lead_items = named
        self.leads = [i for ids in named.values() for i in ids]
        return "A careful review thinks these listings may fit: " + "; ".join(f"item {k}: {', '.join(v)}" for k, v in named.items()) + (f" ({got.get('reason')})" if got.get("reason") else "") + ". Open them and buy what truly fits; end without a purchase only if each one fails."

    def audit_call(self) -> dict[str, Any] | None:
        records = []
        # newest sighting first: a row a late search showed again counts as recent, however early it was first seen
        order = sorted(self.seen_order, key=lambda i: -self.last_seen.get(i, -1))
        for ident in order:
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
        prompt += self.judged_block([i for i in order if i in self.full and not self.sold_out(i)][:30])
        got = self.ask_json(prompt, self.judges(), 1500)
        self.trace.append({"kind": "audit", "got": got})
        return got

    def audit_fits(self, got: dict[str, Any] | None) -> dict[str, list[str]]:
        fits = (got or {}).get("fits")
        if not isinstance(fits, dict):
            return {}
        named = {str(k): [str(i) for i in _items(v) if str(i) in self.rows and not self.sold_out(str(i))][:4] for k, v in fits.items()}
        return {k: v for k, v in named.items() if v}

    def open_rows(self, rows: list[str], spare: int = 5) -> bool:
        """Open listings seen only as search rows: side by side in one step when the store can, else the first two one by
        one (only the first when few steps are left); `spare` steps stay free after the opens."""
        many, joiner = self.compare_target(), self.roles.get("compare")
        tool = self.facts_tool()
        target = self.single_target(tool)
        if many and joiner and not self.compare_bare():
            plan = [(joiner, {many: rows[: self.compare_room()]})]
        elif tool and target:
            plan = [(tool, {target: i}) for i in rows[:2 if self.left() >= spare + 2 else 1]]
        else:
            return False
        if self.left() < len(plan) + spare:
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
        self.check_cart(list(lines), self.hard)
        if not self.verdicts.get(self.check_key(list(lines))):
            return ""
        done, replies = self.order_in_code(lines, drop, calls, True)
        self.trace.append({"kind": "completed", "lines": lines, "drop": drop, "replies": len(replies)})
        if not replies:
            return ""
        return ("Not run (Kyo acted instead): the independent check confirmed " + self.confirmed + ", so Kyo set the cart (" + "; ".join(done)
                + ") and checked out. Store reply: " + " | then: ".join(replies))

    def order_in_code(self, lines: dict[str, int], drop: list[str], calls: list[dict[str, Any]], checked: bool) -> tuple[list[str], list[str]]:
        """Set the cart with Kyo's own calls and check out (the request's steps at the order first). A refused checkout (a
        price, budget or stock change) gets one retry after re-reading the lines, and, for a cart the checks passed, a
        fresh check. Returns the cart calls made and the store's replies to the checkouts."""
        cart, buy = self.roles.get("cart"), self.roles.get("buy")
        done: list[str] = []
        for args in calls:
            faults: list[str] = []
            args = _shape(args, self.schemas.get(cart), "args", faults)
            if faults:
                return done, []
            reply = self.act(cart, args)
            if reply is None or self.finished:
                return done, []
            self.note_cart(cart, args)
            self.observe(cart, reply.get("observation"))
            done.append(_dump(args))
        faults = []
        args = _shape(dict(self.brief.get("buy_args") or {}), self.schemas.get(buy), "args", faults)
        if faults:
            return done, []
        replies: list[str] = []
        for attempt in range(2):
            if attempt:
                self.reread_offers()
                if checked:
                    self.check_cart(list(lines), self.hard)
                    if not self.verdicts.get(self.check_key(list(lines))):
                        break
                if self.left() < 1 or self.finished or any(self.sold_out(i) for i in lines):
                    break
            go, said = self.gate("order")
            if said:
                replies.append(said)
            if not go:
                break
            reply = self.act(buy, args)
            if reply is None:
                break
            self.buys += 1
            if not reply.get("finished"):
                self.refused_at = self.step
            self.observe(buy, reply.get("observation"))
            replies.append(self.show(reply.get("observation")))
            if self.finished or reply.get("finished"):
                break
        return done, replies

    def rescue_plan(self) -> dict[str, int] | None:
        """Lines for an order in place of an ending without a purchase: each item's buyable cart line, else the listing
        the fit judge clears, else the most plausible one (opened, buyable, affordable, nothing contradicted). None when
        some item has none of these or the lines together break the budget."""
        self.map_shelves()
        held = self.lines_by_item()
        plan: dict[str, int] = {}
        # what the careful review named as fitting comes before the code judge's picks: it may have read a value the judge
        # could not ("full sleeve" for long sleeve)
        reviewed: dict[int, list[str]] = {}
        for key, ids in (self.lead_items or {}).items():
            found = re.search(r"\d+", str(key))
            index = int(found.group()) if found else (1 if len(self.items) == 1 else 0)
            if 1 <= index <= len(self.items):
                reviewed[index] = [i for i in ids if self.buyable(i) and not self.sold_out(i)]
        for number, item in enumerate(self.items, 1):
            pick = next(iter(held.get(number) or []), None) or next(iter(reviewed.get(number, []) + self.clear_fits(number) + self.plausible_fits(number)), None)
            if not pick or pick in plan:
                return None
            plan[pick] = item.qty
        plan = self.fit_plan(plan) or plan
        prices = [self.price_of(i) for i in plan]
        if isinstance(self.budget, (int, float)) and self.brief.get("budget_scope") != "each":
            total = sum(p * n for p, n in zip(prices, plan.values())) if None not in prices else None
            if total is None or self.budget + 1e-9 < total <= 10 * self.budget:
                return None
        return plan

    def rescue_order(self) -> str:
        """Before an ending without a purchase goes out: race worlds almost never want one (online, orders pass several times
        as often as endings without a purchase), so when every item has a plausible listing Kyo orders it, once. A cart
        check reads the planned lines first while the dollars allow; when it finds a clear problem, the ending goes on."""
        if self.rescued or not self.roles.get("cart") or not self.roles.get("buy"):
            return ""
        self.rescued = True
        plan = self.rescue_plan()
        if not plan:
            self.trace.append({"kind": "rescue_none"})
            return ""
        drop = [i for i in self.cart_ids() if i not in plan]
        calls = self.cart_calls(plan, drop)
        if calls is None or len(calls) + 2 > self.left():
            return ""
        checked = not self.spent_out()
        if checked:
            self.fill_facts(list(plan), len(calls) + 3)
            self.check_cart(list(plan), self.ceil)
            if self.verdicts.get(self.check_key(list(plan))) is False:
                self.trace.append({"kind": "rescue_refused", "plan": plan})
                return ""
        done, replies = self.order_in_code(plan, drop, calls, checked)
        self.trace.append({"kind": "rescue", "plan": plan, "checked": checked, "replies": len(replies)})
        if not replies:
            return ""
        return ("Not run (Kyo acted instead): a listing that appears to fit was found for every item, so Kyo set the cart ("
                + ("; ".join(done) or "as it was") + ") and checked out. Store reply: " + " | then: ".join(replies))

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
        elif left <= len(self.items) + 5 and self.buys == 0 and not (self.turns and self.step - self.turned_step <= 4):
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
            if self.brief.get("quiet_decline") is False:
                # the request wants to hear of the nearest option: the ending names it (the model's own words kept)
                if not any(ident in text for ident in self.rows):
                    near = self.closest()
                    text = (text.strip() + " " if text.strip() else "No listing meets every requirement. ") + near
                out[key] = text.strip()[: spec.get("maxLength", 600)]
                continue
            named = any(ident in text for ident in self.rows) or self.brief.get("quiet_decline") is not False
            if named or not text:
                out[key] = "Sorry, nothing in the store meets all of your requirements, so I am not buying anything."[: spec.get("maxLength", 200)]
        if self.brief.get("quiet_decline") is False:
            # an optional remark left out by the agent is written all the same: it carries the nearest option
            for key, spec in props.items():
                if key not in out and isinstance(spec, dict) and "string" in _kinds(spec) and "enum" not in spec and self.closest():
                    out[key] = ("No listing meets every requirement. " + self.closest())[: spec.get("maxLength", 600)]
                    break
        return out

    def closest(self) -> str:
        """Per requested item, the opened buyable listing nearest to a fit: nothing contradicted first, then inside the
        budget, then the fewest requirements unmet, then the cheaper; named with its price and what it misses."""
        parts = []
        for number, item in enumerate(self.items, 1):
            best = None
            for ident in self.seen_order:
                if ident not in self.full or self.item_of(ident) != number or not self.buyable(ident) or self.price_of(ident) is None:
                    continue
                sheet = self.sheet_for(ident, number)
                miss = [need for need, verdict, _ in sheet if verdict in ("differs", "silent", "keyless")]
                over = not self.affordable(ident, item.qty)
                rank = (len(FitJudge.broken(sheet)), over, len(miss), self.price_of(ident))
                if best is None or rank < best[0]:
                    best = (rank, ident, miss, over)
            if best:
                _, ident, miss, over = best
                short = ", ".join(miss[:2]) + ("; " if miss and over else "") + ("it is over the budget" if over else "")
                parts.append(f"{ident} at {self.price_of(ident):g}" + (f" ({short})" if short else ""))
        return ("Closest listing" + ("s" if len(parts) > 1 else "") + ": " + "; ".join(parts) + ".") if parts else ""

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
            if isinstance(value, str) and _id_like(value, True):
                # an id typed with a letter added, dropped or changed: the one known id it misspells, else as given
                near = [i for i in self.rows if _one_off(value.strip(), i)]
                return (near[0] if len(near) == 1 else value), ""
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
        text = _dump(_clip_view(observation))
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
            empty = self.empty_checkout(name, args)
            if empty:
                out.append((cid, empty))
                continue
            if ending and self.left() > 2:
                if self.is_buy(name, args):
                    over = self.over_budget()
                    if over:
                        out.append((cid, "Not run yet (no step used). " + over + "\nCart now: " + self.cart_text()))
                        continue
                    wider = self.wider_hold()
                    if wider:
                        out.append((cid, wider + "\nCart now: " + self.cart_text()))
                        continue
                    self.part_trail()
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
                    fitted = self.fit_hold()
                    if fitted:
                        out.append((cid, fitted + "\nCart now: " + self.cart_text()))
                        continue
                    # the check judges attributes: read the cart lines' own (and, for a cheapest pick, a few cheaper rows')
                    if self.brief.get("pick") == "cheapest" and self.left() > self.tail() + 8:
                        for line in self.cart_ids():
                            number, price = self.item_of(line), self.price_of(line)
                            # one such look per item and episode: on short tasks every step counts
                            if number and price is not None and number not in self.checkout_scanned:
                                self.checkout_scanned.add(number)
                                self.capped_scan(number, _under(price), 0)
                    read = self.fill_facts(self.cart_ids(), self.tail() + 3) + self.cheaper_rows()
                    mark = self.cart_mark() + f"#{self.version}"
                    swaps = self.cheaper_cleared() if mark not in self.cheaper_said and self.left() >= 4 else []
                    if swaps:
                        self.cheaper_said.add(mark)
                        said = "; ".join(f"{new} ({self.price_of(new)}) instead of {old} ({self.price_of(old)})" for old, new in swaps)
                        made = self.cheaper_swap(swaps)
                        self.trace.append({"kind": "cheaper_swap", "swaps": swaps, "made": bool(made)})
                        if made:
                            out.append((cid, "Not run yet. The pick rule is the cheapest, and a listing you opened costs less while a key naming each requirement shows its value: "
                                        + said + ". The cart check passed the cheaper cart, so Kyo swapped the lines itself (" + made + "). Check out now.\nCart now: " + self.cart_text()))
                            continue
                        out.append((cid, "Not run yet (no step used). The pick rule is the cheapest, and a listing you opened costs less while a key naming each requirement shows its value: "
                                    + said + ". Swap those lines (take the dearer one out, add the cheaper one at the requested quantity), then check out. "
                                    "Keep the cart only if the cheaper record clearly breaks a requirement; then say which and check out again.\nCart now: " + self.cart_text()))
                        continue
                    # the last look at the cart matters most in the long, many-item episodes that cost the most: up to the hard cap
                    verdict = self.check_cart(None, self.hard)
                    if verdict:
                        seen = ("Kyo read these listings' attributes for the check (one step each):\n" + "\n".join(read) + "\n") if read else ""
                        out.append((cid, "Not run yet. " + chain + seen + verdict + "\nCart now: " + self.cart_text()))
                        continue
                    if reread and self.cost > self.soft:
                        out.append((cid, "Not run yet (no step used). Kyo re-read the cart's listings just now:\n" + reread + "\nCheck out again when every line still shows stock and a price inside the budget; otherwise replace that line first."))
                        continue
                else:
                    chain, hold = self.trail() if self.left() > 6 else ("", False)
                    if hold or (chain and "[Kyo opened" in chain):
                        out.append((cid, "Not run yet (no step used). " + chain + "\nCart now: " + self.cart_text()))
                        continue
                    guard = self.decline_guard()
                    if guard:
                        out.append((cid, "Not run yet (no step used). " + guard + "\nCart now: " + self.cart_text()))
                        continue
                    if self.left() > 3:
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
                # a built line that sold out is not a reason to end: one replacement for it, while the steps still allow it
                lost = [i for i in self.built if self.sold_out(i)]
                if lost and len(lines) + len(lost) >= len(self.items) and len(missing) + len(lost) + 1 <= self.left():
                    picks = self.replacements(lost)
                    if picks:
                        self.late_held = True
                        self.trace.append({"kind": "late_decline_swap", "lost": lost, "picks": picks, "missing": missing})
                        back = f" Put {', '.join(missing)} back at the requested quantities too." if missing else ""
                        out.append((cid, f"Not run yet (no step used). {', '.join(lost)} sold out, and listings already read appear to fit in its place: {', '.join(picks)}. "
                                    f"Add {picks[0]} at the requested quantity (one per sold-out line, of that line's kind).{back} Then check out. "
                                    "End without a purchase only if each of them clearly breaks a requirement; then send the ending call again."))
                        continue
            if self.is_refuse(name, args) and not self.leads and not self.pushed and self.left() > self.max_steps // 3:
                self.pushed = True
                out.append((cid, f"Not run yet (no step used). {self.left()} steps are still left, so search in ways you have not tried before ending: browse the category of each requested item and page through it with the token or offset the store returns, try single-word value queries and the property's other names, and open the listings that might fit. End without a purchase only after that."))
                continue
            if self.is_refuse(name, args):
                done = self.complete_cart() or self.rescue_order()
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
                    trimmed = "\n[Kyo] " + said + trimmed
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
            if name == self.roles.get("search") and self.records_in(observation):
                self.search_seen = dict(args)
            if (not self.price_key or not self.stock_key) and name in readers and isinstance(observation, dict):
                # the keys of a row's price and stock flag: needed by every budget and stock check, whatever the pick
                body = observation
                if not any(isinstance(x, list) and len(x) >= 2 and all(isinstance(r, dict) for r in x) for x in body.values()):
                    body = next((x for x in body.values() if isinstance(x, dict)), body)
                self.read_shape(name, body)
            if self.browse and name == self.browse.get("tool") and isinstance(args.get(self.browse.get("label")), str):
                self.browsed.add(args[self.browse["label"]].strip().lower())
            if name == self.roles.get("compare"):
                rows = self.records_in(observation)
                if rows:
                    self.facts_from[name] = self.facts_from.get(name, False) or any(_fact_block(r) for _, r in rows)
            if name in readers - {self.roles.get("search")}:
                self.note_view(name, observation)
            # every job but searching: only a search result is paged
            plain = {tool for job, tool in self.roles.items() if job != "search"}
            paged = self.more_pages(name, args, observation) if name not in plain and not ending else ""
            if self.single_target(name) and name not in (self.roles.get("cart"), self.roles.get("buy"), self.roles.get("refuse"), self.roles.get("ask")):
                listing = self.ident(observation) if isinstance(observation, dict) else None
                if listing:
                    self.facts_from[name] = self.facts_from.get(name, False) or bool(_fact_block(observation))
                    # a page whose attribute block holds a single value is still read in full: opening it again shows nothing new
                    if name == self.roles.get("open") and self.facts_from[name] and listing not in self.full and _fact_block(observation, 1):
                        self.full[listing] = dict(self.rows.get(listing) or {})
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
                text += self.cheapest_here([observation] + self.paged_obs)
            base = text
            proof = self.evidence(observation)
            if proof:
                text += "\n" + proof
            if name == self.roles.get("cart") and any(_count(v) == 0 for v in args.values() if not isinstance(v, str)):
                if any(isinstance(v, str) and v in self.rows and _in_rows(observation, v) for v in args.values()):
                    text += "\n[Kyo] A count of 0 left that line in the cart: this tool adds the count to the line, so a negative count takes units off."
            self.note_sold = False
            turned_on = ""
            lines = self.loose_lines(observation)
            if lines:
                meaning = self.take_note(lines)
                if meaning:
                    text += "\n[note read] " + meaning + ". Apply this change to your picks before checking out."
                    if ending and not reply.get("finished") and not self.finished:
                        turned_on = meaning
            if self.is_buy(name, args) and not reply.get("finished") and (self.note_sold or any(line in self.spoken(observation) for line in self.sold_lines)):
                seen = self.find_sold()
                if seen:
                    text += "\n[Kyo re-read the cart's lines after the refusal]\n" + seen
            if turned_on and not self.finished:
                again = self.after_turn("order" if self.is_buy(name, args) else "end", turned_on)
                if again:
                    text += "\n" + again
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
        dead = [i for i in self.cart_ids() if self.sold_out(i)]
        if holding and dead and len(dead) < len(self.cart_ids()) and self.left() >= len(dead) + 2:
            # lines that sold out come off before a forced checkout: the order holds only what can still be bought
            keep = {i: self.cart_lines.get(i, 1) for i in self.cart_ids() if i not in dead}
            calls = self.cart_calls(keep, dead)
            if calls and len(calls) + 1 < self.left():
                self.apply_cart(calls)
        if holding and all(self.sold_out(i) for i in self.cart_ids()):
            # every line is gone: a stand-in per item when one appears to fit, else no order of dead lines
            plan = self.change_plan() if self.left() >= 3 else None
            calls = self.cart_calls(plan, [i for i in self.cart_ids() if i not in plan]) if plan else None
            done = self.apply_cart(calls) if calls and len(calls) + 1 < self.left() else None
            holding = bool(done) and any(not self.sold_out(i) for i in self.cart_ids())
        tried = False
        if buy and not holding and refuse and not self.rescued and not self.end_rescued and self.left() >= 3:
            # a forced end with an empty cart: an order of the listings that appear to fit beats an ending without one
            # (race 15: 1 of Vela's 44 endings without a purchase was right); once an episode
            self.end_rescued = True
            said = self.rescue_order()
            if self.finished or self.left() <= 0:
                return ""
            if said:
                # that order went out and the store turned it down: no second checkout of the same cart here
                if again:
                    return said
                tried = True
            fresh = self.cart_view is not None and self.cart_seen_at >= self.cart_at
            holding = bool(self.cart_ids()) if fresh else bool(self.cart_lines)
        if buy and holding and not tried:
            if self.left() >= 5 and not self.spent_out() and self.elapsed() < LOOP_SECONDS + 15:
                self.trail()
            if self.fit_pairs and self.left() >= 4 and self.elapsed() < CALL_SECONDS:
                self.fit_hold()
            self.part_trail()
            if self.finished or self.left() <= 0:
                return ""
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
        opening = _from_text(self.view.get("opening"))
        if opening is not None:
            self.absorb(opening)
        intro = "SHOPPER'S REQUEST:\n" + self.request
        if opening is not None:
            intro += "\n\nWHAT THE STORE SHOWS FIRST:\n" + self.show(opening)
        intro += "\n\nStart by searching for each item in its category and opening several listings in full, so the interviewer can see which properties these listings carry.\n\n" + self.status()
        self.chat = [{"role": "system", "content": GUIDE}, {"role": "user", "content": intro}]
        self.work(self.loop_cap, self.hard)
        if not self.finished:
            self.fill_missing()
            turned = self.finish(again=True)
            # the store turned the ending down (a change from the shopper, a sold-out line): a short second round deals with it
            if turned and not self.finished and not self.broke and not self.no_credit and self.left() > 1:
                if self.note_sold:
                    self.find_sold()
                planned = self.after_turn("order" if turned.startswith("the checkout") else "end", self.notes[-1] if self.notes else "")
                self.chat.append({"role": "user", "content": "[Kyo] Kyo tried to end the episode and " + turned + "\n" + (planned + "\n" if planned else "") + "Apply what the store says, then finish.\n" + self.status()})
                self.work(self.loop_calls + 10, self.ceil)
            if not self.finished:
                self.finish()
        self.trace.append({"kind": "end", "finished": self.finished, "steps": self.step, "cost": round(self.cost, 5), "calls": self.calls,
                           "seconds": round(self.elapsed(), 1)})
        return self.trace

    def work(self, loop_cap: int, spend_cap: float) -> None:
        idle = 0
        while (not self.finished and not self.broke and self.left() > 0 and self.loop_calls < loop_cap and self.cost < spend_cap
               and self.elapsed() < LOOP_SECONDS):
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
                if self.no_credit or self.calls >= self.call_cap or self.misses > 2 or self.spent_out():
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
