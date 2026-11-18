"""Kyo: buys for one shopper inside an ORO store world.

A light model walks the store with the world's own tools. Code keeps the
ledger (items, needs, budget, listings seen), plans the questions to the
shopper, and stops wasted steps. A heavy model reads the request once and
checks the cart before an order and the evidence before a decline.

This build also: learns the message allowance (default and refusals), spreads the
questions over the requested items and moves on from a line once it revealed a need,
reads compatibility and fitment lists from the listings' own words, reviews the
evidence again after any change before ending without a purchase, checks out the
best confirmed cart when steps or time run short, and learns the side-by-side
view's real capacity from its failures.
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

LOOP_CALLS = 42
ALL_CALLS = 60
SPEND_SOFT = 0.055
SPEND_HARD = 0.078
# waits before resending a failed model call; an episode has 300 s in all, so at most 45 s go to waiting
WAITS = (4, 10, 20, 35)
WAIT_TOTAL = 45
SHOW_CHARS = 450
OBS_LIMIT = 15000
HISTORY_LIMIT = 150000
GATE_ROUNDS = 3             # store calls the process gate may make before one checkout or ending
AUTO_PAGES = 8              # further search pages Kyo reads itself for a cheapest, most or least pick
CHEAPER_OPENS = 3           # cheaper rows of a cart line's kind whose attributes Kyo reads before a cheapest checkout
DEFAULT_MESSAGES = 5        # messages assumed when neither the request nor the store states the allowance (a refusal corrects it)
CLOCK_SOFT = 215            # seconds into the 300 s episode after which the loop stops and the best cart is checked out
CLOCK_HARD = 262            # no new model call starts after this many seconds

# one attempt per call (think() resends): a stuck call costs 40 s, not minutes, inside the 300 s
_llm = ProxyClient(timeout=40, max_retries=1, retry_delay=1)
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
- A requirement on a property is met when the listing's attribute for that property states the value. Key spelling does not matter (spaces, underscores, camel case, prefixes), and one element of a comma or slash list is enough. Values match by their words: case, singular or plural and plain synonyms are fine, and a value that contains the required value's words counts ("deep teal" meets teal, "extra roomy fit" meets roomy). The initials of a required multi-word value count too (a value "UHD" meets "ultra high definition"). A value with different words does not count (a model code or part number is not a word it stands for, and "teal green" is not "sea green"). Units are converted before comparing numbers, limits include the edge value, and a range must lie wholly inside a limit.
- An attribute whose key names the requirement's property (under that name or a plain synonym) decides: if it says something else, the listing fails whatever other keys or the title say. When several keys name that same property (spellings such as "pack type" and "pack_type"), the listing meets the requirement when any one of them states the value. A key for a different property, even a related one (a handle colour is not the body's colour), decides nothing about the requirement. Kyo adds [evidence] lines after opened listings with the keys whose names match a requirement.
- When the listing has no attribute for that property, the wanted value stated under another attribute (for example a list of styles or features that includes it) or in the title or description counts. A number of another kind does not (a weight is not a capacity), nor a bare figure whose text does not say what it measures.
- When the category label itself states a requirement's value (a season or audience in the label), that requirement is met by the category.
- A requirement can point at another listing instead of giving a value ("the same X as the listing sold by <shop> with <property>", "within N of the X on the page of ..."). Resolve it first: find that listing (search its category, brand, shop or words), open its page in full with the single-item or side-by-side tool (a search row is not enough), and when that page refers you to a further listing for the property, find and open that one the same way; the last page gives the value. The order counts only when every page of that chain was opened. A link is the listing that matches every detail the words give (its shop, its kind and each property they name), not merely the first listing of that shop. When the wanted value is anything but that other listing's value, the rule excludes only listings sharing that exact value, and never end without a purchase because of it alone. A page's note may also state a pairing rule ("X goes together with Y"): use the partner only when the requirement asks for a value that goes with, pairs with or works alongside the other listing's value; when it asks for that same value ("nothing other than", "just what is given", "matching"), take the page's own value and ignore the pairing rule.
- When a listing you are about to buy sends you to another page for more about it, look at that page before you buy.
- When one requested item must fit another (an accessory for a device), choose the two together from the listings' own words (a list of what it fits, a model or series code that both pages carry). When no accessory that meets its own requirements fits the device you picked, switch to another device that meets its requirements and that such an accessory fits; never end without a purchase while that switch is untried.
- Model numbers and fit lists are usually not searchable: find accessories and parts by their product words, then read their pages. A name the [fit] lines show for the item others must fit can still be tried as a search word to narrow a long list, written exactly as shown (spaces and hyphens kept, never squeezed together). Kyo adds [fit] lines after opened listings with what each page's own fit list says: a family such as "every ABC-47xx model" covers each code of that family except the ones listed after "except", and a sentence about another part ("Part X also fits ...") says nothing about this listing. A part must fit the model the request gives for the shopper's own device or vehicle.
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
- After a change, look again at the listings you have already read before searching further: one that failed only on a requirement now dropped, or only on the old budget, may fit now.
- Re-read a chosen listing's record shortly before checking out, because price and stock can change silently.

FINISHING
- Put exactly the chosen lines in the cart (remove anything else), look at the cart once, then check out with a call on its own.
- End without a purchase only after a real search finds no listing that meets everything. Running short of steps proves nothing of the kind: then check out the best cart you have. Follow the request's wording for that ending, and when a remark is needed, keep it short and name no listing.
- Every reply must call at least one tool. Keep your text short: what you know, then the next move."""


BRIEF = """You prepare a shopping task for an agent. Read the shopper's request and the store's tools and return ONE JSON object (compact, no indentation, nothing else) with these keys:

"items": a list with one entry per distinct product the shopper wants bought, each {"name": the kind of product in the request's words, "shelf": the store's category label for it when a tool lists labels (copy it exactly), else "", "quantity": units of that one listing (integer, 1 when not stated; a pair or a set that is one listing counts as 1), "musts": a list of firm requirements on that product, each "property: constraint" with the exact value, limit, exclusion or closed list. Name the property with the request's own word for it ("meant for a lakeside occasion" gives "occasion: lakeside", "a 12-inch blade" gives "blade length: 12 inch"); do not swap it for a neighbouring property. A must whose value is read from another listing is ONE must holding the whole description of that listing ("rim colour: unlike the rim colour of the kettle Ovarno sells with a glass lid"): words that describe that other listing (its shop, its kind, "with <property> <value> and <property> <value>" after it) stay inside that one must and are never musts of this product, and the request's own words for how the value relates to that listing (compatible with, very same as, unlike, within N either way of, no higher than, at least) are kept as they are, never reworded}. Never put price, budget, stock or quantity into musts. The story around the request (the person meant to use it, where it will stand, what went wrong before, the occasion behind it) is context, not a must, unless it states a property's value or limit outright; and never turn words that are part of what the product is called (a season, an audience or a use named with the product) into musts: those belong to "name". A product the request names in one phrase joined by "and" (one "I want ..." with one quantity, often the name of one store department) is ONE item; make separate items only when the request gives each product its own quantity or its own requirements. Each item also carries "fits": the number (counting from 1 in this list) of another requested item that this product must work with, fit or be a compatible part for (an accessory for a requested device), else null. Set it only when the request's own words say the product must fit, suit, match or work with that other requested item; products that are merely bought together, or belong in the same room or bed, are not tied to each other.
"own_model": the model name or code of a device or vehicle the shopper already owns and that the requested products must fit (it is not itself being bought), copied exactly, else "".
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


PLAN = """You choose the questions a shopping agent sends to a shopper. Shoppers hold a few firm preferences they did not write down, each about one property of one requested product. A question reveals a preference only when it names that exact property, and each question is answered only about the first property it names, so every question costs one property.

You get the request, the items with what the request already settles for each, and a numbered list of attribute keys seen on opened listings of each item's category, with how many of those listings carry the key and example values. An unwritten preference may be on a property that no listed key names.

Choose the properties in this order:
0. What the request itself asks you to check with the shopper (listed under ASK FIRST when there is any), each named by the property it is about.
1. Hinted properties. The request's story often points at an unwritten preference without giving its value: a sentence about something the shopper cannot judge or leaves to you, someone whose taste decides, the product's looks or the setting it belongs in, the person meant to use or wear it, or what is inside it. For each such sentence, name the one property it points at and the item it is about, in the order the story gives them.
Name a hinted property by its plain name: a hint about who will wear it points at gender (only for a product a person wears), a hint about its looks at colour or finish (whichever the request leaves open).
2. Then the listed keys, ranked: favour the properties a buyer of that kind of product chooses between first (often its size, capacity or dimensions, its type, its material or its colour, whichever that kind of product is bought by) and keys carried by most of the item's listings; avoid keys seen on only one or two of them.
Skip a property the request already settles for that item, a second name for a property already chosen, a key that lists purchase variants (a mix of sizes, colours and pack counts), identifiers, warranty and other logistics. With several items, put every item's hinted properties first, then take turns between the items, so that every item gets one property before any item gets a second; give each item at least two picks in all when COUNT allows.
"label" is the property's plain name: for a listed key, all of the key's words in plain spelling ("lid colour" for lidColour, "rim finish" for rim_finish; never shortened to "colour" or "finish"); only the property's name, no example values, no brackets, no second property.
Choose up to COUNT, best first. Return ONE JSON object and nothing else: {"ask": [{"item": item number, "label": "...", "hint": "at most 8 of the story's words it comes from, or an empty string"}]}"""


HEAR = """A shopping agent asked a shopper some questions. Read each answer and return ONE JSON object and nothing else:
{"needs": [{"item": item number, "need": "property: constraint", "points": true when the value is to be read from another listing, else false}], "drop": [{"item": item number, "property": "a requirement the shopper now gives up"}], "free": [{"item": item number, "property": "..."}], "crowded": [numbers of the questions whose answer complains that the message asked about too many things or wants a single question; not answers that only say they did not follow it], "budget": new total limit as a number in the unit the answer itself uses (its own figure, never converted), or null}
- "needs": every firm preference an answer states, with the exact value or limit and its unit ("at least 30 cm", "oval", "not wicker"). Name the need by the property the answer itself names, which may differ from the one asked about; several answers about one property make one need. Copy the value word for word as the answer says it, keeping its nouns and units ("at least 3 shelves", not "at least 3").
- An answer can give the value by pointing at another listing (a value like, unlike or near the one on some shop's product, that product picked out by a few of its properties): that is ONE need on the property the answer names, with "points": true. Its value is the answer's own words copied exactly, from the comparison words to the end of that description, including what it says about a listing that page may name; never shorten or reword it. The properties describing the other product are not needs of this item.
- "free": the property of each question the shopper said they do not mind about.
- Folksy or roundabout wording that names a value still states that value."""


NOTE = """While a shopping agent works, the store showed this note (it may come from the shopper or the store). Return ONE JSON object and nothing else:
{"kind": "change" when the note changes the task (budget, requirements, a price, availability, a refusal of the order), "error" when it only reports that a call was malformed or not accepted, "other" for anything else, "budget": the new total limit as a number in the request's unit, or null, "add": [{"item": item number, "need": "property: constraint"}], "drop": [{"item": item number, "need": "the requirement no longer wanted"}], "gone": [listing ids said to be unavailable], "sold_out": true when it says a chosen listing is gone, sold out or no longer available (even without naming it), "price_moved": true when it says a price changed, "summary": "one short sentence"}
A replaced requirement is a drop of the old one plus an add of the new one. Report every change in the note."""


CHECK = """You are the last check before a shopping agent checks out. Decide from the records shown whether this cart is exactly right.
Reading rules: a requirement is met when the listing's attribute for that property states the value (any key spelling, one element of a list is enough, case, singular/plural and plain synonyms count, a value containing the required value's words counts, such as "deep teal" for teal, so do the initials of a required multi-word value ("UHD" for "ultra high definition"), units converted, limits include the edge, a range must lie wholly inside); an attribute whose key names the same property as the requirement decides, and if it says something else the listing fails, but when several keys name that property (spellings such as "pack type" and "pack_type") any one of them stating the value is enough; an item that must work with or fit another requested item, or the shopper's own model, fits only when its own fit list names that item's model code or other name (a family such as "every ABC-47xx model" covers its codes except those listed after "except"; a sentence about another part says nothing), and the FIT EVIDENCE lines, read by code from the listings' own words, show this; a key for a different property, even a related one (a handle colour is not the body's colour), decides nothing; when the listing has no attribute for that property, the wanted value stated under another attribute or in the title or description counts (a number of another kind does not: a weight is not a capacity; nor does a bare figure whose text does not say what it measures); a requirement the category label itself states is met by that category; the category label must match the requested item as a whole; a rule of being unlike another listing's value counts as met unless the records show that same value. Each requested item must appear as a separate cart line at the requested quantity, nothing else may be in the cart, nothing shown out of stock, and the total of price times quantity over all lines must be within the budget. The newest record of a listing wins.
When the pick rule is cheapest, most or least, also look at the other listings shown: if one of them clearly fits every requirement and beats the cart's choice on that rule, the cart is wrong.
Return ONE JSON object and nothing else: {"ok": true or false, "problems": ["short, concrete problem", ...], "better": [listing ids that should replace a cart line, best first]}. Use ok=false only for a problem the records clearly show or a requirement no record addresses."""


READING = """the listing's attribute for a property decides when its key names that same property (any key spelling, one element of a list is enough, synonyms and unit conversions count, a value containing the required value's words counts, such as "deep teal" for teal, and so do the initials of a required multi-word value ("UHD" for "ultra high definition"), limits include the edge; when several keys name the same property, any one of them stating the value is enough); a key for a different property, even a related one (a handle colour is not the body's colour), decides nothing; when no attribute names that property, the wanted value stated under another attribute or in the title or description counts (a bare figure that does not say what it measures counts for nothing); a requirement the category label itself states is met by that category; the category label must match the item; a rule of being unlike another listing's value counts as met unless the records show that same value; an item that must work with or fit another requested item, or the shopper's own model, fits only when its own fit list names that model (FIT EVIDENCE lines, when shown, are read by code from the listings' own words); out-of-stock listings and listings that break the budget do not fit."""

AUDIT = """A shopping agent wants to end without buying because it thinks nothing fits. Check the listing records shown against the requirements, with these reading rules: """ + READING + """
Records marked ROW were seen only as a search result row: their page (the attributes) has not been read, so they can be neither passed nor failed on a property the row does not show.
Return ONE JSON object and nothing else: {"fits": {"item number": [listing ids that appear to meet every requirement, cheapest first, at most 4]}, "open": [ids of ROW listings of a requested kind, inside the budget, whose page could show that they fit, most promising first, at most 6], "reason": "one sentence"}"""

UNMIX = """A shopper's request was read into firm requirements on a product. Some requirements take their value from another listing the request points at, and the words that pick out that other listing (its kind, its shop, the properties named after it with "with ...") are sometimes read by mistake as requirements on the product being bought. Using the request's wording, say which of the OTHER REQUIREMENTS below only describe that other listing. Return ONE JSON object and nothing else: {"describe_other": [those requirements copied exactly; empty when every one is a real requirement on the product]}"""

TRAIL = """A shopping agent is about to check out. Some requirements take their value from another listing instead of stating it, and a listing's page can send the reader on to a further listing for that property. The order counts only when the agent has opened in full every listing along each such chain. Follow each requirement's chain from the requirement's own description, link by link, through the pages' own words, using the records below (OPENED: the agent has opened that page; ROW: seen only as a search result row, its page not read).
Return ONE JSON object and nothing else: {"chains": [{"need": the requirement copied exactly, "links": [{"id": the listing id when a record below is that link, else "", "about": how the requirement or the previous page describes that listing}], "done": true when the last link is an OPENED page that states the property's value itself, else false, "value": the property's value as stated on that last page, or ""}]}
List each chain's links in order and stop at the first link whose page is not OPENED or not found, since its words are still unknown. A record is a link only when it matches every detail the words give for that listing (its shop, its kind and each property they name): an OPENED record whose page contradicts one of those details is not the link, so look for another record that fits them all; a ROW whose few visible words neither confirm nor contradict the details may be named (its page will be opened and checked); when no record fits, give "id": "" and the full description in "about"."""

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
    # words only: "Hubcaps, Centercaps & Trim Rings" must give hubcap, not "hubcaps,"
    return frozenset(w[:-1] if len(w) > 3 and w.endswith("s") else w for w in re.findall(r"[a-z0-9]+", _words(text)))


def _names(prop: frozenset[str], key: frozenset[str]) -> bool:
    if not prop or not key:
        return False
    return prop == key or prop <= key or (len(key) >= 2 and key <= prop)


_ONE_AT_A_TIME = re.compile(r"\bone (?:question|thing|item|at a time)\b|\bsingle question\b|\bone by one\b|\bhold the rest\b|"
                            r"\btoo (?:much|many questions) in one\b|\bnet of questions\b|\btoo many questions\b|\bone per message\b|"
                            r"\bseparate (?:them|questions)\b|\bpick one question\b", re.I)
_SPELLING = {"colour": "color", "colours": "color", "colors": "color", "grey": "gray"}
_LOOKS = {"color", "finish"}
_BROAD = {"type", "size", "style", "number", "option", "feature", "kind", "level", "count", "range", "rating", "class",
          "name", "general", "item", "product", "detail", "info", "spec"}


_HEDGED = re.compile(r"\b(?:not|no|other than|unlike|like|similar|at least|at most|under|over|less|more|between|than|up to|either|or)\b", re.I)
_PLAIN_STOP = {"the", "and", "for", "with", "any", "only", "value", "shown", "page", "listing", "one", "type", "kind"}
_VARIANTS = {"aluminium": "aluminum", "grey": "gray", "colour": "color", "colours": "color"}


def _plain_words(text: Any) -> set[str]:
    out = set()
    for w in re.findall(r"[a-z0-9]+", str(text).lower()):
        w = _VARIANTS.get(w, w)
        out.add(w[:-1] if len(w) > 3 and w.endswith("s") else w)
    return out


def _key_words(text: str) -> set[str]:
    """A key's words, singular and in one spelling, for comparing property names."""
    out = set()
    for w in _words(text).split():
        w = _SPELLING.get(w, w)
        out.add(w[:-1] if len(w) > 3 and w.endswith("s") else w)
    return out


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


# how listings name themselves and what they fit, in plain store wording (read by code so the check does not have to)
_MODEL_NO = re.compile(r"\bmodel\s*(?:no\.?|number|num\.?|#)\s*[:#]?\s*([A-Za-z]{2,6}[- ]?\d{2,6}[A-Za-z]?)\b", re.I)
_SOLD_AS = re.compile(r"\b[Aa]lso sold as\s+([A-Z][A-Za-z]+[- ]?\d{2,6})\b")
_SERIES = re.compile(r"\bpart series\s*:?\s*([A-Z]{1,4}\d{0,3})\b", re.I)
_SERIES_TAG = re.compile(r"\[([A-Z]{1,4}\d{0,3})-\d{3,}\]")
_FIT_LIST = re.compile(r"\b(?:works with|fits|compatible with|for use with)\b\s*:?\s*([^.;\n|]{2,200})", re.I)
_FAMILY = re.compile(r"\b(?:[Ee]very|[Aa]ll|[Aa]ny)\s+([A-Za-z]{2,6})[- ]?(\d{1,4})([xX]{1,3})\b")
_EXCEPT = re.compile(r"\bexcept\b\s*:?\s*([^.;\n|]+)", re.I)
_OTHER_PART = re.compile(r"\bpart\s+[A-Za-z0-9-]+\s+also fits\b[^.\n|]*", re.I)
_CODE = re.compile(r"\b([A-Z]{2,6}[- ]?\d{2,6}[A-Z]?|[A-Z][a-z]{2,} \d{2,6})\b")


def _code_key(text: Any) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(text).upper())


def _written(text: Any) -> str:
    """A code or name as the listing's page writes it (spaces and hyphens kept): the form a search needs."""
    return re.sub(r"\s+", " ", str(text)).strip()


def _fit_read(text: str) -> tuple[dict[str, str], list[tuple[str, int, dict[str, str], str]]]:
    """What the fit lists in a listing's text name: codes (key -> as written) and families ("every ABC-47xx model,
    except ..."): (prefix key, width, exceptions key -> as written, the family as written). A sentence about another
    part is left out."""
    text = _OTHER_PART.sub(" ", text)
    exact: dict[str, str] = {}
    families: list[tuple[str, int, dict[str, str], str]] = []
    for segment in _FIT_LIST.findall(text):
        family = _FAMILY.search(segment)
        if family:
            rest = _EXCEPT.search(segment)
            skip: dict[str, str] = {}
            for code in (_CODE.findall(rest.group(1)) if rest else []):
                skip.setdefault(_code_key(code), _written(code))
            families.append((_code_key(family.group(1) + family.group(2)), len(family.group(3)), skip, _written(family.group(0))))
            continue
        for code in _CODE.findall(segment):
            exact.setdefault(_code_key(code), _written(code))
    for code in _SERIES_TAG.findall(text):
        exact.setdefault(_code_key(code), _written(code))
    return exact, families


def _covers(code: str, exact: set[str], families: list[tuple[str, int, set[str]]]) -> bool:
    """A fit list names the code exactly, or a family ("every ABC-47xx model") holds it and does not except it."""
    if code in exact:
        return True
    return any(code.startswith(prefix) and len(code) == len(prefix) + width and code not in skip for prefix, width, skip in families)


# ---- reference-chain component (ported from the H05 line; switches on only for requirements that point at another listing)
REF_HOPS = 6

REF_STEPS = 56

REF_CANDIDATES = 4

REF_ACCEPT = 4      # passing pages collected at one hop before the best match is taken (when none of them is certain)
REF_PAGES = 6

REF_CALLS = 24

POINTER = """A shopper's requirement takes its value from the page of another listing. Read the requirement and describe that listing. Return ONE JSON object and nothing else:
{"property": the property whose value is to be read from that page (the requirement's own property), "kind": the kind of product that listing is, as the requirement writes it, or "", "seller": the shop that sells it, or "", "attrs": [{"property": "...", "value": "..."} for each property the requirement names to pick that listing out], "says": the requirement's own words that describe that listing, copied exactly}
Words that only tie the requirement to the page (a comparison word, a closing remark about where the page leads) are not part of the description."""

PAGE_NOTE = """A shopper's requirement takes its value for PROPERTY from a listing's page, and a page can send the reader on to a further listing for it. Read the page record below (its text, its notes and its attributes) and return ONE JSON object and nothing else:
{"value": the value of PROPERTY that this page itself states, from its attributes or its text, or "" when it states none, "next": null, or {"kind": the kind of product of the listing the page sends the reader to for PROPERTY, "seller": the shop that sells it, "attrs": [{"property": "...", "value": "..."} for each property the page names to pick that listing out], "says": the page's own words that describe that listing, copied exactly} when the page sends the reader on for PROPERTY, "is_it": true when the page is the listing the DESCRIPTION names (its shop, its kind and every described property fit its text and attributes), false when its text or attributes say it is another listing or another version of it, null when nothing tells}
A page that sends the reader on for PROPERTY states no value of its own for it. A sentence about another property says nothing about PROPERTY, and a bare figure whose text does not say what it measures is no value."""

NOTE_ONLY = """A shop wrote the NOTE below on a listing's page. A shopper's requirement needs the value of PROPERTY, and a page can send the reader on to a further listing for it. Read the NOTE alone (the page's attributes are not shown) and return ONE JSON object and nothing else:
{"sends": null, or {"kind": the kind of product of the listing the note sends the reader to for PROPERTY, "seller": the shop that sells it, "attrs": [{"property": "...", "value": "..."} for each property the note names to pick that listing out: the property as the note names it, and its value alone], "says": the note's own words that describe that listing, copied exactly} when the note sends the reader to another listing for PROPERTY, "value": the value of PROPERTY that the note states for the listing it is written on, or "" when it states none, "is_it": true when the note's own description of the listing it is written on fits the DESCRIPTION (its shop, its kind and each described property), false when that description gives another value for a described property, null when the note does not say}
A note that sends the reader on for PROPERTY states no value of its own for it. What a note says about the listing it sends the reader to tells nothing about the listing it is written on. A sentence about another property says nothing about PROPERTY."""

RESOLVE = """A shopper asks for a value that depends on another listing's page. Take the relation from the shopper's own words: the REQUEST when it words this requirement (the agent's note of it, the REQUIREMENT, may have reworded it: the REQUEST wins), else the REQUIREMENT itself (it may be the shopper's answer to a question). Say what it means for the product being bought. Return ONE JSON object and nothing else:
{"relation": "same" when the request asks for the same value as the page's (the very same, just the same, matching, identical), "partner" when it asks for a value that goes with, works alongside, is compatible with or pairs with the page's value, "different" when it asks for anything but the page's value, "limit" when it asks for a figure compared with the page's (at least, "at X or above", no lower than, at most, "at X or below", no higher than, within N either way of), "other" when none of these fits, "value": for same the page's value, for partner the value the page's note pairs with the page's value (copied from the note, in the note's own words; "" when the note names no partner for it), for different the page's value, for limit the limit as a plain comparison in the page's unit ("at most 7.4 inch", "at least 8", "between 10.88 and 10.92 inches"), else ""}
A pairing rule is stated in the note ("acrylic goes with pvc, copper with plastic" pairs copper with plastic); it may be read from either side of "goes with". Use only the note given."""

CHOOSE = """A shopper's requirement describes a listing of a given shop by some of its properties. Below are that shop's listings of the kind, with their attributes. Pick the listings that fit the description, best first; convert units where the description and the attributes use different ones (a length in metres fits the same length in feet). A listing that holds another value for a described property does not fit. Return ONE JSON object and nothing else: {"ids": [up to 4 listing ids, best first; empty when none fits]}"""

def _singular(text: str) -> set[str]:
    """Words in their singular form: mattresses -> mattress, boxes -> box, batteries -> battery, pads -> pad, glass stays glass."""
    out = set()
    for w in re.findall(r"[a-z0-9]+", _words(text)):
        if len(w) > 4 and w.endswith("ies"):
            w = w[:-3] + "y"
        elif len(w) > 4 and w.endswith("es") and w[:-2].endswith(("ss", "x", "ch", "sh", "z")):
            w = w[:-2]
        elif len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
            w = w[:-1]
        out.add(w)
    return out

_CMP_LOW = re.compile(r"\b(?:at least|no less than|not less than|minimum(?: of)?|min\.?|or more|or above|or higher|more than|over|above|greater than|larger than|bigger than)\b", re.I)

_CMP_HIGH = re.compile(r"\b(?:at most|no more than|not more than|maximum(?: of)?|max\.?|or less|or below|or lower|less than|under|below|up to|smaller than|within)\b", re.I)

_NEGATED = re.compile(r"^\s*(?:not|no|other than|anything but|anything other than|except|without|never)\b\s*", re.I)

_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")

_RANGE = re.compile(r"^(?:between\s+|from\s+)?(-?\d+(?:\.\d+)?)\s*([a-z%\"']{0,12})\s*(?:-|\u2013|\u2014|to|and)\s*(-?\d+(?:\.\d+)?)\s*([a-z%\"' ]{0,14})$", re.I)

_STOCKISH = {"stock", "availability", "available", "orderable", "price", "budget", "quantity"}

_REACH = re.compile(r"\b(?:expandable|expandible|extendable|upgradeable|upgradable|supports?|supporting|up\s*to|upto|max(?:imum)?(?:\s+of)?)\b[^,;|/()]*", re.I)

def _covered(want: set[str], have: set[str]) -> bool:
    """Every wanted word is among the listing's, or a word of the same stem with another ending ("stripe" for "striped")."""
    return all(w in have or any(len(w) >= 4 and len(h) >= 4 and abs(len(w) - len(h)) <= 3 and (w.startswith(h) or h.startswith(w)) for h in have) for w in want)

def _cmp_figure(value: str, m: "re.Match[str]") -> bool:
    """A comparison word counts only beside the figure it compares: "under 30" or "at least about 30" (the figure follows), "30 gb or more"
    (it comes before). "9 size showing under size" holds no comparison."""
    if m.group(0).lower().startswith("or "):
        return bool(re.search(r"\d[\w\"'\s]{0,16}$", value[: m.start()]))
    return bool(re.match(r"\s*(?:[a-z$\u20ac\u00a3\"']+\s+){0,2}-?\d", value[m.end():], re.I))

def _comparison(value: str) -> tuple[str | None, int]:
    """The comparison a value holds, read from the earliest and longest comparison phrase ("no more than" is one phrase, and the
    "more than" inside it is not another): ("low", where the phrase starts), ("high", where), or (None, -1) when it holds none or
    holds a low and a high one apart from each other."""
    found = [(m.start(), m.start() - m.end(), kind, m.end()) for kind, rx in (("low", _CMP_LOW), ("high", _CMP_HIGH)) for m in rx.finditer(value) if _cmp_figure(value, m)]
    if not found:
        return None, -1
    found.sort()
    first = found[0]
    apart = [f for f in found[1:] if f[0] >= first[3] and f[2] != first[2]]
    return (None, -1) if apart else (first[2], first[0])

def _same_unit(a: str, b: str) -> bool:
    """Two unit words that name one unit: the same word, one the start of the other ("in", "inch"), or one an abbreviation of the
    other ("ft", "feet"; "pcs", "piece"). A single letter never matches another word ("m" is not "mm")."""
    if a == b:
        return True
    short, long = (a, b) if len(a) <= len(b) else (b, a)
    short = short[:-1] if short.endswith("s") and len(short) > 2 else short
    if len(short) < 2 or len(long) - len(short) > 6:
        return False
    if long.startswith(short):
        return True
    if short[0] != long[0]:
        return False
    at = 0
    for ch in long:
        if at < len(short) and ch == short[at]:
            at += 1
    return at == len(short)

def _figures(text: str) -> tuple[list[str], set[str]]:
    """The figures in a text and its other words (a unit among them), minus the filler words."""
    return _NUMBER.findall(text), {w for w in _plain_words(_NUMBER.sub(" ", text)) if w not in _PLAIN_STOP}

def _units_agree(a: set[str], b: set[str]) -> bool | None:
    """Whether two sets of unit words name a common unit; None when either has none."""
    if not a or not b:
        return None
    return any(_same_unit(x, y) for x in a for y in b)

_NEGATING = re.compile(r"\b(?:not|unlike|isn't|aren't|other than|instead of|rather than|doesn't|does not|exists? elsewhere|elsewhere as)\b", re.I)

_POINTS_AT = re.compile(r"\b(?:listing|page|sold by|sells|seller|shop)\b", re.I)

def _budget_figure(value: float, text: str, unit: str = "") -> float:
    """A budget a model read off a shopper's line, put back in the line's own figures: a reading that is one of the line's figures
    in the request's unit divided or multiplied by a hundred (a figure in cents read as dollars) becomes that figure. Only figures
    written with the request's unit word count, so a figure in another unit, converted by the model, is left as read."""
    word = (re.findall(r"[a-z]+", (unit or "").lower()) or ["cent"])[0].rstrip("s")
    figures = []
    for raw in re.findall(r"(\d[\d,]*(?:\.\d+)?)\s*" + re.escape(word), (text or "").lower()):
        try:
            figures.append(float(raw.replace(",", "")))
        except ValueError:
            continue
    if not figures or any(abs(value - f) < 0.005 for f in figures):
        return value
    for f in figures:
        if abs(value * 100 - f) < 0.5 or abs(value / 100 - f) < 0.005:
            return f
    return value


def _no_commas(text: str) -> str:
    """A figure written with thousands separators ("1,250") as the plain figure ("1250")."""
    return re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", text)


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
        self.queries: list[str] = []            # every search the store was sent (all its text arguments)
        self.row_item: dict[str, int] = {}      # listing -> the item whose search first showed it
        self.search_item: int | None = None
        self.vote_stamp: tuple[int, int] = (-1, -1)
        self.label_votes: dict[str, int] = {}
        self.cuts = 0                            # replies that were all thinking, cut at the token limit
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
        self.lead_items: dict[str, list[str]] = {}
        self.verdicts: dict[str, bool] = {}
        self.confirmed: str | None = None
        self.completed = False
        self.id_key: str | None = None
        self.facts_from: dict[str, bool] = {}
        self.facts_miss: dict[str, int] = {}
        self.queued: list[str] = []
        self.buys = 0
        self.refusals = 0
        self.recent: list[str] = []
        self.chat: list[dict[str, Any]] = []
        self.opened: set[str] = set()
        self.refs: list[tuple[int, str]] = []
        self.ref_values: dict[str, str] = {}
        self.ref_final: dict[str, tuple[str, str]] = {}      # a requirement that points at a page -> (relation, the value it means) once the last page is read
        self.ref_tried: set[str] = set()
        self.ref_internal: set[str] = set()      # pointers at a product this same order buys (the walker leaves them to the fit between items)
        self.search_keys: set[str] = set()      # the keys a search row carries (what a page view adds is the shop's note)
        self.walked: set[str] = set()           # listings the chain walker read
        self.walk_steps = 0
        self.walk_calls = 0
        self.thrift = False                           # the walker's budget is small: fewer reads a hop
        self.heard_text = ""                           # the shopper's answers the last hear() read (budget figures are checked against them)
        self.fills = 0                                 # times the cart was set right by code (missing, doubled and extra lines)
        self.insteads = 0                              # times an order of three or more items was bought where the agent meant to decline
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
        # elapsed-time guard for the 300 s episode
        self.t0 = time.monotonic()
        # step of the last review before an ending, and of the last change to the task (a refusal note, a drop, a new budget)
        self.audit_at = -1
        self.change_at = -1
        self.msg_known = False
        # compatibility: item number -> the requested item it must fit; the shopper's own model the parts must fit
        self.fit_of: dict[int, int] = {}
        self.own_model = ""
        self.finish_tries = 0
        self.rescued = False
        self.code_rescued = False

    # ---------- plumbing ----------
    def elapsed(self) -> float:
        return time.monotonic() - self.t0
    def left(self) -> int:
        return self.max_steps - self.step

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
        """Steps kept for the finish: the checkout, and one more while a step due at the order needs a message. Before
        the first finish, with a cart held, two more stay free: the first checkout is often refused with a change."""
        talk = any(p["when"] == "order" and p["messages"] > 0 for p in self.process)
        base = 2 if talk and self.msg_used < self.msg_limit and self.gate_due("order") else 1
        if self.finish_tries == 0 and self.cart_lines and self.max_steps >= 24:
            base += 2
        return base

    def think(self, messages: list[dict[str, Any]], models: list[str], max_tokens: int, tools: list[dict[str, Any]] | None = None,
              effort: str | None = None, plain: bool = False) -> dict[str, Any] | None:
        for model in models:
            if self.calls >= ALL_CALLS or self.elapsed() > CLOCK_HARD:
                return None
            self.calls += 1
            tries = 0
            # after two replies that were all thinking, the light model works without thinking for the rest of the
            # episode: each such reply costs about 15 s of the 300 s clock and returns nothing
            # (a plain call, one that only reads facts off a text, never thinks: it is quicker and costs a tenth)
            quiet = (self.cuts >= 2 or plain) and model == LIGHT
            room = max_tokens
            # a reasoning effort is asked of the heavy model only (the light one does not keep to it)
            asked = effort if model == HEAVY else None
            while True:
                message, status = self.call_model(model, messages, room, tools, quiet, asked)
                if message is not None:
                    return message
                if status == 400 and asked:
                    # the model's route does not take that setting: the same call without it
                    asked = None
                    continue
                if status == "length":
                    self.cuts += 1
                    if self.elapsed() < CLOCK_HARD - 15:
                        if model == LIGHT and not quiet:
                            # the reply was all thinking, cut at its limit: the same call once more with thinking off
                            # (a thinking budget is not honoured, and more room only buys more thinking)
                            quiet = True
                            continue
                        if model != LIGHT and room == max_tokens:
                            # a model whose thinking cannot be turned off: the same call once more with twice the room
                            room = min(8000, max(2 * room, 2400))
                            continue
                # no credit, rate limits and server errors pass: wait and send the same call again
                passing = status is None or status in (402, 408, 425, 429) or (isinstance(status, int) and status >= 500)
                # a timeout already took the client's 40 s: the next model gets the call instead of the same one again
                if not passing or status in (None, 408) or tries >= len(WAITS) or self.waited + WAITS[tries] > WAIT_TOTAL or self.elapsed() > CLOCK_SOFT:
                    break
                pause = WAITS[tries] * (0.8 + 0.4 * random.random())
                self.waited += WAITS[tries]
                tries += 1
                time.sleep(pause)
            if status == 402:
                self.no_credit = True
                return None
        return None

    def call_model(self, model: str, messages: list[dict[str, Any]], max_tokens: int, tools: list[dict[str, Any]] | None,
                   quiet: bool = False, effort: str | None = None) -> tuple[dict[str, Any] | None, Any]:
        """One inference call: the reply message, or None and the failure's status (0 for a reply without content).
        With quiet, an Anthropic model answers without thinking first; with effort, it thinks at that level."""
        sent = messages
        if model.startswith("anthropic/") and len(messages) > 1:
            sent = [dict(m) for m in messages]
            marks = [0] + ([len(sent) - 1] if len(sent) > 1 else [])
            for index in marks:
                content = sent[index].get("content")
                if isinstance(content, str) and content:
                    sent[index]["content"] = [{"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}]
        body: dict[str, Any] = {"model": model, "messages": sent, "max_tokens": max_tokens, "temperature": 0}
        if quiet and model.startswith("anthropic/"):
            body["reasoning"] = {"enabled": False}
        elif effort:
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
            # nothing but thinking, cut at the token limit, reads "length"; any other empty reply 0
            cut = data["choices"][0].get("finish_reason") == "length"
            self.trace.append({"kind": "llm_fail", "model": model, "status": "length" if cut else 0})
            return None, ("length" if cut else 0)
        error = result.error or {}
        status = error.get("status")
        if isinstance(data, dict) and isinstance(data.get("error"), dict):
            # an error carried inside a 200 reply, with the provider's own code
            status = _count(data["error"].get("code"))
            error = {"body": data["error"].get("message")}
        self.trace.append({"kind": "llm_fail", "model": model, "status": status, "body": str(error.get("body") or "")[:200]})
        return None, status

    def ask_json(self, prompt: str, models: list[str], max_tokens: int = 1800, effort: str | None = None, plain: bool = False) -> dict[str, Any] | None:
        for model in models:
            message = self.think([{"role": "user", "content": prompt}], [model], max_tokens, effort=effort, plain=plain)
            got = _json_obj((message or {}).get("content"))
            if got is not None:
                return got
            if self.broke or self.no_credit or self.calls >= ALL_CALLS:
                return None
        return None

    def act(self, name: str, args: dict[str, Any]) -> dict[str, Any] | None:
        self.search_item = self.item_for_query(args) if name == self.roles.get("search") and isinstance(args, dict) else None
        if name == self.roles.get("search") and isinstance(args, dict):
            self.queries.append(" ".join(str(v) for v in args.values() if isinstance(v, str)))
        payload = dict(self.binding)
        payload["step"] = self.step + 1
        payload["action"] = {"name": name, "args": args}
        for _ in range(4):
            result = _env.post_verbose("/environment/call", payload)
            if isinstance(result.data, dict):
                data = result.data
                self.step = data["step"] if isinstance(data.get("step"), int) else self.step + 1
                if data.get("finished"):
                    self.finished = True
                seen = data.get("observation")
                if name in self.page_tools() and not (isinstance(seen, dict) and seen.get("error")):
                    # listings whose full page the agent has opened (a search row is not that); a reply that does not
                    # show the listing (an error worded in a key of its own) opened nothing
                    shown = _dump(seen)
                    for value in args.values():
                        self.opened.update(v.strip() for v in (value if isinstance(value, list) else [value]) if isinstance(v, str) and v.strip() and v.strip() in shown)
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
        buy, refuse = self.roles.get("buy"), self.roles.get("refuse")
        if buy and buy == refuse and (not brief["buy_args"] or not brief["refuse_args"] or brief["buy_args"] == brief["refuse_args"]):
            ask = (f"The store tool {buy} both checks out and ends without a purchase, depending on its arguments. Its schema: "
                   + _dump(self.schemas.get(buy)) + "\nReturn ONE JSON object and nothing else: {\"buy_args\": the fixed arguments that check out, \"refuse_args\": the fixed arguments that end without a purchase} (only the arguments that choose the mode).")
            fixed = self.ask_json(ask, [LIGHT, HEAVY], 600) or {}
            if isinstance(fixed.get("buy_args"), dict) and isinstance(fixed.get("refuse_args"), dict) and fixed["buy_args"] != fixed["refuse_args"]:
                brief["buy_args"], brief["refuse_args"] = fixed["buy_args"], fixed["refuse_args"]
        fits_raw: list[int | None] = []
        for entry in _items(brief.get("items")):
            if not isinstance(entry, dict):
                continue
            quantity = _count(entry.get("quantity"))
            count = quantity if quantity and quantity > 0 else 1
            musts = [str(m) for m in _items(entry.get("musts")) if m]
            self.items.append(Item(str(entry.get("name") or "the product"), str(entry.get("shelf") or ""), count, musts))
            fits_raw.append(_count(entry.get("fits")))
        if not self.items:
            self.items = [Item("the requested product", "", 1, [])]
        # the request must use a word for fitting for one item to be tied to another
        tied = re.search(r"\b(?:fit|fits|fitting|compatible|compatibility|work(?:s|ing)? with|go(?:es)? with|match(?:es)?|suit(?:s|able)?|made for)\b", self.request, re.I)
        for number, other in enumerate(fits_raw, 1):
            if other and 1 <= other <= len(self.items) and other != number and tied:
                self.fit_of[number] = other
        own = brief.get("own_model")
        if isinstance(own, str) and 2 <= len(own.strip()) <= 60:
            self.own_model = own.strip()
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
            self.msg_known = True
        elif self.roles.get("ask"):
            # no allowance stated: the usual one; a refused message lowers it to what was sent
            self.msg_limit = DEFAULT_MESSAGES
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
            needs = [m + ((f" (the page it points at gives: {self.ref_values[m]}" + (f"; so this requirement means: {self.ref_final[m][1]}" if m in self.ref_final else "") + ")") if m in self.ref_values else "")
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
            when = {"choose": "before choosing", "order": "at the order (Kyo does it when you check out)", "end": "at an ending without a purchase (Kyo does it then)"}[entry["when"]]
            parts.append(f"step the request asks for, {when}: {entry['step']}")
        facts = self.facts_tool()
        if facts and (self.facts_from.get(facts) is True or self.roles.get("offer")):
            bare = sorted(t for t in self.page_tools() | {self.roles.get("offer") or ""} if t and self.facts_from.get(t) is False)
            pages = sorted(t for t in self.page_tools() if t != facts)
            parts.append(f"listing attributes come from {facts}" + (f"; {', '.join(bare)} show none" if bare else "")
                         + (f"; a listing's page (what a chain of pages or a page check means) is shown by {' or '.join(pages)}" if pages else ""))
        return "\n".join("- " + p for p in parts)

    def name_searched(self, name: str) -> bool:
        """Some search used every word of this name as the page writes it (a squeezed code is a word nothing carries)."""
        parts = [w for w in re.split(r"[\s\-]+", name.lower()) if w]
        for query in self.queries:
            words = set(re.findall(r"[a-z0-9]+", query.lower()))
            if parts and all(w in words for w in parts):
                return True
        return False

    def name_hint(self) -> str:
        """A part that must fit another requested item is found by the names that item's listing is sold under (a search
        finds the words a listing carries, and a fit list names the item by its model number or another name). Until the
        leading candidates' names have each been tried as written, with the part's own words, say so."""
        if not self.fit_of or self.buys:
            return ""
        cat_key = self.category_key()
        cart = set(self.cart_ids())
        todo: list[str] = []
        for main in sorted(set(self.fit_of.values())):
            item = self.items[main - 1]
            needs = [n for n in item.musts + item.told if n not in item.dropped]
            ranked = []
            for ident in self.full:
                if self.item_of(ident, cat_key) != main or not self.buyable(ident):
                    continue
                if "bad" in [self.need_state(ident, n) for n in needs]:
                    continue
                price = self.price_of(ident)
                ranked.append((ident not in cart, price if price is not None else float("inf"), ident))
            for _, _, ident in sorted(ranked)[:2]:
                for name in self.names_of(ident):
                    if name not in todo and not self.name_searched(name):
                        todo.append(name)
        if not todo:
            return ""
        mains = sorted(set(self.fit_of.values()))
        parts = sorted(self.fit_of)
        words = " and ".join(f"item {n} ({self.items[n - 1].name})" for n in parts)
        tries = "; ".join(f'"{self.items[n - 1].name} {name}"' for name in todo[:2] for n in parts[:2])
        return (f"\n[names to search] {words} must fit item {mains[0]}: a fit list names the item it fits by the names its listing is sold under, "
                f"so look for the part with each of those names written exactly as shown (spaces and hyphens kept), one name per search. Try: {tries}")

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
        live = {need for _, need in self.needs_now()}
        lookups = [need for _, need in self.refs if need in live and need not in self.ref_values]
        if lookups and self.buys == 0:
            text += "\n[resolve first: open the listing each of these points at, and every listing its page refers to; the last page gives the value] " + " | ".join(lookups[:3])
        text += self.name_hint()
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
                        if self.search_item:
                            self.row_item[ident] = self.search_item
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
        matched = {label for label in groups if self.label_item(label) is not None}
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
                # all-different longer values are model names or codes only when most carry digits; plain phrases
                # ("wall mount", "over the door") are a property's values
                if (len(values) >= 4 and len(set(values)) == len(values) and mean > 12
                        and sum(1 for v in values if re.search(r"\d", v)) * 2 > len(values)):
                    continue
                twin = any(name.startswith(other + " ") and slot["per"] == spec["per"] for other, spec in kept)
                if twin:
                    continue
                kept.append((name, slot))
            for name, slot in kept[:14]:
                table.append((label, name, slot["n"], [v[:40] for v in slot["values"][:4]]))
        return table

    def item_groups(self) -> list[list[str]]:
        """Listing ids per category, keeping the categories that match a requested item (largest first)."""
        cat_key = self.category_key()
        groups: dict[str, list[str]] = {}
        for ident in self.seen_order:
            label = str(self.rows[ident].get(cat_key) or "") if cat_key else ""
            groups.setdefault(label, []).append(ident)
        matched = [ids for label, ids in groups.items() if label and self.label_item(label) is not None]
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
        self.complete_shape(shape, observation, props, set((self.schemas.get(name) or {}).get("required") or []))
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
        if value is None or later or count <= shown:
            # the last page, or the agent is paging itself
            return ""
        if pick == "cheapest" and self.price_sorted(observation):
            return ""
        pages = -(-(count - shown) // shown)
        if pages > (3 if self.left() > 40 else 2) or self.pages_read + pages > AUTO_PAGES or self.left() - self.tail() - pages < 12:
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
                self.facts_seen(tool, seen)
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
        if self.asked or not self.roles.get("ask") or self.msgs_free() <= 0:
            return False
        if self.left() <= 8:
            return False
        opened = len(self.full)
        if opened >= min(3 * len(self.items) + 1, 10):
            return True
        return self.step >= max(8, min(self.max_steps // 4, 18))

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
        # spare picks beyond the messages: the code spreads the questions over the items and moves on from an item
        # once it revealed a need
        prompt = PLAN.replace("COUNT", str(count + 2)) + "\n\nREQUEST:\n" + self.request + "\n\nITEMS:\n" + "\n".join(settled)
        first = [str(x) for x in _items(self.brief.get("ask_first")) if str(x).strip()]
        if first:
            prompt += "\n\nASK FIRST (the request asks to be asked about these):\n" + "\n".join(first)
        prompt += "\n\nATTRIBUTE KEYS SEEN:\n" + ("\n".join(listing) or "(no full listing records yet)")
        plan = self.ask_json(prompt, [HEAVY, LIGHT], 1800) or {}
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
        self.trace.append({"kind": "plan", "ask": _clip(plan.get("ask"), 120)})
        sent: list[tuple[int, str, str]] = []
        told: list[str] = []
        order = list(dict.fromkeys(number for number, _ in ordered))
        queues = {number: [label for n, label in ordered if n == number] for number in order}
        # first round, all messages but one: one property per item in turn (the planner's order inside an item)
        first = count - 1 if count >= 2 else count
        head: list[tuple[int, str]] = []
        depth = 0
        while len(head) < first and any(depth < len(q) for q in queues.values()):
            for number in order:
                if depth < len(queues[number]) and len(head) < first:
                    head.append((number, queues[number][depth]))
            depth += 1
        batch = self.send([(number, label, self.question(number, label)) for number, label in head])
        if not batch:
            return ""
        sent += batch
        asked = [(number, label) for (number, label), _ in zip(head, batch)]
        heard = self.hear(batch)
        self.take_heard(heard, told)
        revealed = self.revealed_items(heard)
        # an answer saying a question asked two things: that property once more with its look-alikes set aside, before
        # any new property (the clash involves a property the shopper holds more often than a fresh guess does)
        again: list[tuple[int, str, str]] = []
        retried: set[tuple[int, str]] = set()

        def queue_again(pairs: list[tuple[int, str]], answers: list[tuple[int, str, str]], said: dict[str, Any]) -> None:
            flagged = {_count(x) for x in _items(said.get("crowded"))}
            for index, ((number, label), (_, text, answer)) in enumerate(zip(pairs, answers), 1):
                if (index in flagged or _ONE_AT_A_TIME.search(str(answer))) and (number, label) not in retried and number not in revealed:
                    retried.add((number, label))
                    redo = self.reask(number, label, text)
                    if not redo or (redo[0] != label and ((number, redo[0]) in retried or (number, redo[0]) in asked)):
                        continue
                    if all((number, redo[1]) != (n, q) for n, _, q in again):
                        again.append((number, redo[0], redo[1]))

        queue_again(asked, batch, heard)
        # the rest one at a time: an item that has revealed nothing yet first, then the item asked least, then the plan's order
        # (a reply that did not follow says nothing about the property: never re-asked)
        while self.msgs_free() > 0 and not self.finished:
            again = [entry for entry in again if entry[0] not in revealed]
            if again:
                number, label, text = again.pop(0)
                retried.add((number, label))
                self.trace.append({"kind": "reask", "item": number, "q": text})
            else:
                rest = [pick for pick in ordered if pick not in asked]
                if not rest:
                    break
                times = {number: sum(1 for n, _ in asked if n == number) for number in order}
                rest.sort(key=lambda pick: (pick[0] in revealed, times.get(pick[0], 0), ordered.index(pick)))
                number, label = rest[0]
                text = self.question(number, label)
            batch = self.send([(number, label, text)])
            if not batch:
                break
            sent += batch
            asked.append((number, label))
            # one answer: the light model reads it
            heard = self.hear(batch, (LIGHT, HEAVY))
            self.take_heard(heard, told)
            revealed |= self.revealed_items(heard)
            queue_again([(number, label)], batch, heard)
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

    def question(self, number: int, label: str, samples: list[str] | None = None, sure: bool = True) -> str:
        name = self.items[number - 1].name
        near = self.look_alikes(number, label, sure)
        if near:
            # a shopper reads a question that could mean two of their properties as asking two things at once;
            # naming the look-alike as set aside leaves one property
            lead = f"Apart from the {near[0]}" + (f" (and the {near[1]})" if len(near) > 1 else "")
            return f"{lead}, which {label} would you like for the {name}?"
        text = f"What {label} would you like for the {name}"
        return text + (f", for example {samples[0]} or {samples[1]}?" if samples else "?")

    def look_alikes(self, number: int, label: str, sure: bool = True) -> list[str]:
        """Up to two other keys on this item's opened listings that a shopper could take for the asked property: keys
        sharing a word with it whose values differ from its own (pack type: multipack, single / packaging type: bag,
        sachet), and for a colour question the finish. Keys with the same values are the same property under another
        name (connectivity / connection type), and narrower or wider forms of it (color family for color) are the
        property itself: naming those as set aside would hide it. With sure=False (the shopper already said the
        question asked two things) a shared word is enough."""
        mine = _key_words(label)
        roots = {w[:4] for w in mine if len(w) >= 4 and w not in _BROAD}
        if not mine:
            return []
        counts: dict[str, int] = {}
        values: dict[str, set[str]] = {}
        for ident, record in self.full.items():
            if self.item_of(ident) != number:
                continue
            for key, value in (_fact_block(record) or {}).items():
                name = _words(key)
                if name and len(name) <= 30:
                    counts[name] = counts.get(name, 0) + 1
                    values.setdefault(name, set()).update(re.findall(r"[a-z]{3,}", str(value).lower()))
        own = set().union(*[v for name, v in values.items() if _key_words(name) == mine]) if values else set()
        near: list[str] = []
        looks: list[str] = []
        for name, _ in sorted(counts.items(), key=lambda kv: -kv[1]):
            theirs = _key_words(name)
            if not theirs or theirs <= mine or mine <= theirs:
                continue
            if any(name == other or _key_words(other) == theirs for other in near + looks):
                continue
            if roots & {w[:4] for w in theirs if len(w) >= 4 and w not in _BROAD}:
                other = values.get(name) or set()
                shared = len(own & other) / max(1, min(len(own), len(other)))
                if not sure or (own and other and shared < 0.5):
                    near.append(name)
            elif "color" in mine and "finish" in theirs and "color" not in theirs:
                looks.append(name)
        return (near + looks)[:2]

    def reask(self, number: int, label: str, sent: str) -> tuple[str, str] | None:
        """After an answer saying a question asked two things: the same property again with its look-alikes set
        aside, or None when there is none to name. A finish question that collided is asked as colour without the
        finish, the looks property shoppers hold far more often."""
        mine = _key_words(label)
        if "finish" in mine and "color" not in mine:
            label = "color"
        near = self.look_alikes(number, label, sure=False)
        if "color" in _key_words(label) and not any("finish" in _key_words(n) for n in near) and "finish" in mine:
            near = ([n for n in near] + ["finish"])[:2]
        if not near:
            return None
        lead = f"Apart from the {near[0]}" + (f" (and the {near[1]})" if len(near) > 1 else "")
        text = f"{lead}, which {label} would you like for the {self.items[number - 1].name}?"
        return (label, text) if text != sent else None

    def revealed_items(self, heard: dict[str, Any]) -> set[int]:
        """Item numbers for which the answers stated a firm need."""
        out: set[int] = set()
        for entry in _items(heard.get("needs")):
            if isinstance(entry, dict) and str(entry.get("need") or "").strip():
                out.add(min(max(1, _count(entry.get("item")) or 1), len(self.items)))
        return out

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
            answer = reply.get("observation")
            if isinstance(answer, dict) and answer.get("error") and not any(isinstance(v, str) and len(v.split()) >= 4 for k, v in answer.items() if k != "error"):
                # the store refused the message (the allowance is used up): the allowance is what was sent
                self.msg_limit = self.msg_used
                self.msg_known = True
                self.trace.append({"kind": "message_refused", "error": _clip(answer, 120), "limit": self.msg_limit})
                break
            self.msg_used += 1
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
            self.budget = _budget_figure(heard["budget"], self.heard_text, str(self.brief.get("budget_unit") or ""))

    def hear(self, sent: list[tuple[int, str, str]], models: tuple[str, ...] = (HEAVY, LIGHT)) -> dict[str, Any]:
        prompt = HEAR + "\n\nITEMS:\n" + "\n".join(f"item {n}: {it.name}" for n, it in enumerate(self.items, 1))
        prompt += "\n\nQUESTIONS AND ANSWERS:\n" + "\n".join(f"{i}. (about item {n}) Q: {q}\nA: {a}" for i, (n, q, a) in enumerate(sent, 1))
        self.heard_text = " ".join(a for _, _, a in sent)
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
        if isinstance(budget, (int, float)) and not isinstance(budget, bool) and budget > 0:
            budget = _budget_figure(budget, " ".join(lines), str(self.brief.get("budget_unit") or ""))
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
        if changed:
            # what was reviewed before this no longer settles whether anything fits
            self.change_at = self.step
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

    # ---------- compatibility read from the listings' own words ----------
    def listing_text(self, ident: str) -> str:
        parts: list[str] = []

        def walk(node: Any, depth: int) -> None:
            if depth > 4:
                return
            if isinstance(node, str):
                parts.append(node)
            elif isinstance(node, dict):
                for value in node.values():
                    walk(value, depth + 1)
            elif isinstance(node, list):
                for value in node:
                    walk(value, depth + 1)

        walk(self.rows.get(ident) or {}, 0)
        return " | ".join(parts)

    def identity(self, ident: str) -> set[str]:
        """The codes a listing is known by: its model number, the other names it is sold as, its part series."""
        text = self.listing_text(ident)
        found = {_code_key(c) for c in _MODEL_NO.findall(text)} | {_code_key(c) for c in _SOLD_AS.findall(text)} | {_code_key(c) for c in _SERIES.findall(text)}
        return {c for c in found if len(c) >= 2}

    def fit_targets(self, ident: str) -> tuple[set[str], list[tuple[str, int, set[str]]]]:
        """What a listing's own words say it fits: exact codes and names, and families ("every ABC-47xx model, except ...").
        A sentence about another part ("Part X also fits ...") is left out."""
        exact, families = _fit_read(self.listing_text(ident))
        return set(exact), [(prefix, width, set(skip)) for prefix, width, skip, _ in families]

    def fit_shown(self, ident: str) -> list[str]:
        """What a listing's fit list names, worded as its page writes it: codes and names first, then families with
        their exceptions. Models read this and search with it, so a name keeps its spaces and hyphens."""
        exact, families = _fit_read(self.listing_text(ident))
        return list(exact.values()) + [word + (f" except {', '.join(skip.values())}" if skip else "") for _, _, skip, word in families]

    def names_of(self, ident: str) -> list[str]:
        """The names a listing is known by (model number, other names it is sold as, part series), as its page writes
        them: the form a search for a listing that fits it needs (a search finds the words, not a squeezed code)."""
        text = self.listing_text(ident)
        found: dict[str, str] = {}
        for pattern in (_MODEL_NO, _SOLD_AS, _SERIES):
            for code in pattern.findall(text):
                if len(_code_key(code)) >= 2:
                    found.setdefault(_code_key(code), _written(code))
        return list(found.values())

    def item_of(self, ident: str, cat_key: str | None = None) -> int | None:
        """The requested item whose category a listing is in, by the category label's words."""
        cat_key = cat_key or self.category_key()
        label = str((self.rows.get(ident) or {}).get(cat_key) or "") if cat_key else ""
        return self.label_item(label) if label else None

    def item_by_words(self, label: str) -> int | None:
        """The requested item a category label names: the same words, else the item sharing most words with the label (singular
        forms: "Mattresses" is the mattress), its own main noun (the last word of its name) in the label first ("Mattress Pads, ..."
        is the mattress pad, not the mattress), then the larger share of the item's words, then the request's order."""
        whole = _words(label)
        for number, item in enumerate(self.items, 1):
            if whole in (_words(item.shelf or ""), _words(item.name)):
                return number
        glue = {"and", "the", "for", "with", "all", "any", "one", "two", "its", "own"}
        mine = {w for w in _singular(label) if len(w) >= 3 and w not in glue}
        best: tuple[tuple[int, bool, float, int], int] | None = None
        for number, item in enumerate(self.items, 1):
            theirs = {w for w in _singular(item.shelf or "") | _singular(item.name) if len(w) >= 3 and w not in glue}
            common = mine & theirs
            if not common:
                continue
            head = sorted(_singular(item.name.split()[-1]) if item.name.split() else set())
            # a match through the item's own main noun outranks one through a describing word ("hard" in "Hard Drives")
            rank = (bool(head) and head[0] in mine, len(common), len(common) / len(theirs), -number)
            if best is None or rank > best[0]:
                best = (rank, number)
        return best[1] if best else None

    def label_item(self, label: str) -> int | None:
        """The item a category label belongs to: by shared words, else (for an item no seen category shares a word
        with, like a clutch master in "Transmission Assembly, Parts & Mounts") the item whose own searches brought up
        at least two of that category's listings."""
        found = self.item_by_words(label)
        if found is not None or not self.row_item:
            return found
        cat_key = self.category_key()
        if not cat_key:
            return None
        stamp = (len(self.rows), len(self.row_item))
        if self.vote_stamp != stamp:
            self.vote_stamp = stamp
            labels = {str(r.get(cat_key) or "") for r in self.rows.values()} - {""}
            covered = {self.item_by_words(other) for other in labels} - {None}
            votes: dict[str, dict[int, int]] = {}
            for ident, number in self.row_item.items():
                if number in covered:
                    continue
                other = str((self.rows.get(ident) or {}).get(cat_key) or "")
                if other and self.item_by_words(other) is None:
                    tally = votes.setdefault(other, {})
                    tally[number] = tally.get(number, 0) + 1
            self.label_votes = {other: max(tally, key=tally.get) for other, tally in votes.items() if max(tally.values()) >= 2}
        return self.label_votes.get(label)

    def item_for_query(self, args: dict[str, Any]) -> int | None:
        """The one requested item a search's words name (None when they name none or two equally)."""
        words = {w for w in _stem(" ".join(str(v) for v in args.values() if isinstance(v, str))) if len(w) >= 4}
        scores = [len(words & {w for w in _stem(item.name) | _stem(item.shelf or "") if len(w) >= 4}) for item in self.items]
        best = max(scores, default=0)
        return scores.index(best) + 1 if best > 0 and scores.count(best) == 1 else None

    def view_order(self, limit: int = 30, skip: Any = (), cheap_first: bool = False) -> list[str]:
        """Listings to show a reviewer, newest first. With several items, each item keeps its own share (pages read
        first, then for a cheapest pick its cheapest rows) so a long search for one item cannot push the others out
        of view; one item keeps the plain newest-first order."""
        order = [i for i in reversed(self.seen_order) if i not in skip and not self.sold_out(i)]
        price = lambda i: self.price_of(i) if self.price_of(i) is not None else float("inf")  # noqa: E731
        if cheap_first:
            order.sort(key=lambda i: (i not in self.full, price(i)))
        if len(self.items) <= 1:
            return order[:limit]
        cheapest = self.brief.get("pick") == "cheapest"
        share = max(4, limit // len(self.items))
        groups: dict[Any, list[str]] = {}
        for ident in order:
            groups.setdefault(self.item_of(ident), []).append(ident)
        picked: list[str] = []
        for number in range(1, len(self.items) + 1):
            ids = groups.get(number, [])
            ids = sorted(ids, key=lambda i: (i not in self.full, price(i) if cheapest else 0))
            picked += ids[:share]
        chosen = set(picked)
        for ident in order:
            if len(chosen) >= max(limit, len(picked)):
                break
            chosen.add(ident)
        return [i for i in order if i in chosen]

    def fit_lines(self, ids: list[str], limit: int = 8) -> list[str]:
        """FIT EVIDENCE: for an item that must fit another requested item or the shopper's own model, what each listing's
        own fit list says; for the item others must fit, the codes each listing is known by."""
        if not self.fit_of and not self.own_model:
            return []
        own = {_code_key(c) for c in _CODE.findall(self.own_model)} or ({_code_key(self.own_model)} if self.own_model else set())
        mains = set(self.fit_of.values())
        cat_key = self.category_key()
        lines: list[str] = []
        known_by: dict[int, list[tuple[str, set[str]]]] = {}
        cart = set(self.cart_ids())
        for ident in ids:
            if len(lines) >= limit:
                break
            number = self.item_of(ident, cat_key)
            if number is None:
                continue
            if number in mains and number not in self.fit_of:
                known = self.names_of(ident)
                if known:
                    lines.append(f"{ident} (item {number}) is known as: {', '.join(known)}")
                continue
            if not self.own_model and number not in self.fit_of:
                continue
            exact, families = self.fit_targets(ident)
            listed = self.fit_shown(ident)
            if not listed:
                lines.append(f"{ident} (item {number}): no fit list " + ("on its page" if ident in self.opened else "in what was shown (its page may have one)"))
                continue
            parts = []
            if own:
                ok = any(_covers(c, exact, families) for c in own)
                parts.append(("FITS " if ok else "does NOT fit ") + self.own_model)
            if number in self.fit_of:
                main = self.fit_of[number]
                if main not in known_by:
                    known_by[main] = [(m, codes) for m in self.seen_order if self.item_of(m, cat_key) == main for codes in [self.identity(m)] if codes]
                hits = [m for m, codes in known_by[main] if m != ident and any(_covers(c, exact, families) for c in codes)]
                # the cart's own line first, then listings whose page was opened
                hits.sort(key=lambda m: (m not in cart, m not in self.opened))
                more = f" and {len(hits) - 6} more" if len(hits) > 6 else ""
                parts.append(f"fits item {main} listings: {', '.join(hits[:6])}{more}" if hits else f"names none of the item {main} listings read so far")
            lines.append(f"{ident} (item {number}) fit list: {'; '.join(listed)[:160]} -> " + "; ".join(parts))
        return lines

    def fit_evidence(self, observation: Any) -> str:
        lines = self.fit_lines([i for i, _ in self.records_in(observation)], 7)
        return ("[fit: read from each listing's own words, names written as the listing writes them; a family 'every ABC-47xx' holds its codes except the ones after 'except'] " + " | ".join(lines)) if lines else ""

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
        # listings whose attributes were read first, each group cheapest first, for a cheapest pick
        order = self.view_order(limit, skip, cheap_first=self.brief.get("pick") == "cheapest")
        for ident in order:
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
        fits = self.fit_lines(ids, 12)
        if fits:
            prompt += "\n\nFIT EVIDENCE (read by code from the listings' own words; an item that must fit another fits only when its fit list names that one's codes):\n" + "\n".join(fits)
        if self.brief.get("pick") in ("cheapest", "most", "least"):
            prompt += "\n\nOTHER LISTINGS SEEN (newest first):\n" + self.others(ids, 15)
        got = self.ask_json(prompt, [HEAVY, LIGHT], 1500)
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

    # ---------- reference chains (RC / RQ) ----------
    def safe(self, work: Any, default: Any, *args: Any) -> Any:
        """Run one of the newer helpers; a failure in it is traced and the default stands in, so the episode goes on."""
        try:
            return work(*args)
        except Exception as exc:  # noqa: BLE001
            self.trace.append({"kind": "helper_error", "where": getattr(work, "__name__", "?"), "error": f"{type(exc).__name__}: {exc}"[:200]})
            return default

    def facts_seen(self, tool: str, observation: Any) -> None:
        """Record whether a single-listing view shows attributes: True once it showed a table; False only after two listings
        without one (a listing's table can hold a single entry, which is no table by itself)."""
        if _fact_block(observation):
            self.facts_from[tool] = True
            return
        if self.facts_from.get(tool) is True:
            return
        self.facts_miss[tool] = self.facts_miss.get(tool, 0) + 1
        if self.facts_miss[tool] >= 2:
            self.facts_from[tool] = False

    def complete_shape(self, shape: dict[str, Any], observation: dict[str, Any], props: dict[str, Any], required: set[str]) -> None:
        """What the light model left unnamed in a search result's shape is read off the result by code: the match count is the integer key
        (a count, a total, a tally...) that is no smaller than the rows shown, the next-page value the other key holding a number, a
        string or nothing beside it, and the parameter that takes it the tool's optional argument named like a position (after, page,
        offset, cursor, skip, token, row, start, from). Without these a store with an unusual word for its count could not be paged."""
        rows = observation.get(shape["rows"]) if shape.get("rows") else None
        if not isinstance(rows, list):
            return
        others = [k for k in observation if k != shape["rows"] and not isinstance(observation[k], (list, dict))]
        if not shape.get("count"):
            ints = [k for k in others if isinstance(observation[k], int) and not isinstance(observation[k], bool) and observation[k] >= len(rows)]
            pick = [k for k in ints if re.search(r"total|count|found|tally|match|hit|number", k, re.I)] or (ints if len(ints) == 1 else [])
            if pick:
                shape["count"] = pick[0]
        if not shape.get("next"):
            rest = [k for k in others if k != shape.get("count") and (observation[k] is None or (isinstance(observation[k], (int, str)) and not isinstance(observation[k], bool)))]
            pick = [k for k in rest if re.search(r"next|more|follow|cursor|token|after|page|continue", k, re.I)] or (rest if len(rest) == 1 else [])
            if pick:
                shape["next"] = pick[0]
        if shape.get("next") and not shape.get("next_arg"):
            opts = [k for k in props if k not in required and isinstance(props[k], dict) and ("integer" in _kinds(props[k]) or "string" in _kinds(props[k]))]
            pick = [k for k in opts if re.search(r"after|page|offset|cursor|skip|token|row|start|from|follow|next|more|continue", k, re.I)]
            if pick:
                shape["next_arg"] = pick[0]
        self.trace.append({"kind": "shape_completed", "count": shape.get("count"), "next": shape.get("next"), "next_arg": shape.get("next_arg")})

    def resolve_need(self, need: str, value: str, last: str) -> None:
        """What a requirement that points at another listing means once the last page is read. The shopper's own words give the
        relation (the agent's note of them may have reworded it: "compatible with" came out as "same as"), the last page's note the
        store's pairing rule. Kept only for a partner or a limit: code reads "the same" and "other than" off the requirement itself."""
        if need in self.ref_final or not value or not self.walk_ok():
            return
        prop = need.partition(":")[0].strip()
        note = self.note_text(last).strip()[:600] if last else ""
        prompt = (RESOLVE + f"\n\nREQUEST:\n{self.request[:1500]}\n\nREQUIREMENT (the agent's note of it): {need[:500]}\nPROPERTY: {prop}"
                  + f"\nVALUE ON THE LAST PAGE: {value[:200]}\nNOTE ON THE LAST PAGE: {note}")
        got = self.ask_walk(prompt, 500)
        relation = str(got.get("relation") or "").strip().lower()
        final = str(got.get("value") or "").strip()
        self.trace.append({"kind": "resolve", "need": need[:100], "page": value[:60], "relation": relation, "value": final[:80]})
        if relation == "partner" and not final and not re.search(r"\b(?:goes?|pairs?|paired|matches?|partner)\b", note, re.I):
            relation, final = "same", value.strip()          # no pairing rule on the page: the page's own value is what the words can mean
        if relation in ("partner", "limit", "same") and final and len(final) <= 120 and (relation != "same" or final == value.strip()):
            self.ref_final[need] = (relation, final)
            self.version += 1

    def follow_refs(self) -> str:
        """For each requirement whose value is on another listing: walk its chain of pages in code. The requirement names a
        shop and a kind of product (and some properties) for the first page; each page either states the value or sends the
        reader on to another listing. Every page along the way is opened with the page view, as the order requires. Returns what
        was found for the agent, or "" when there is nothing to follow."""
        live = {need for _, need in self.needs_now()}
        todo = [need for _, need in self.refs if need in live and need not in self.ref_values and need not in self.ref_tried]
        if not todo or not self.roles.get("search") or self.left() < 16 or self.finished:
            return ""
        # at most REF_STEPS store calls, and no more than half of the steps that remain after the steps kept for the finish
        # the share of the steps left and the reserve slide with the limit: 65% after a reserve of 10 at a limit of 50 or less, 50% after 14 at 100 or more
        slide = min(1.0, max(0.0, (self.max_steps - 50) / 50))
        budget = [min(REF_STEPS, max(8, int((self.left() - self.tail() - (10 + 2 * slide)) * (0.65 - 0.10 * slide))))]
        self.thrift = budget[0] < 28      # (an episode with the full limit of 100 that starts walking after the first questions, at step ~18, still has 33)
        before = self.step
        lines: list[str] = []
        for need in todo[:3]:
            if budget[0] <= 4 or self.cost > SPEND_HARD or self.elapsed() > CLOCK_SOFT - 60 or self.finished:
                break
            self.ref_tried.add(need)
            chain, value = self.walk_pointer(need, budget)
            self.trace.append({"kind": "follow", "need": need[:160], "chain": chain, "value": value})
            if value:
                self.ref_values[need] = value
                self.version += 1
                self.resolve_need(need, value, chain[-1] if chain else "")
            if chain:
                # the last page's note goes with the value: it may state a pairing rule the requirement asks for ("goes together with")
                last = self.note_text(chain[-1]).strip()[:320]
                lines.append(f"{need[:200]} -> opened {', '.join(chain)}; " + (f"the last page gives: {value}" if value else "no page along it stated the value yet")
                             + (f" (its note: {last})" if value and last else "")
                             + (f"; so the requirement means: {need.partition(':')[0].strip()}: {self.ref_final[need][1]}" if need in self.ref_final else ""))
        self.walk_steps += self.step - before
        return ("[Kyo followed the pages a requirement points at, opening each in full] " + " | ".join(lines)) if lines else ""

    def walk_ok(self) -> bool:
        """Whether the chain walker may still make a model call (its own count, the shared count with room kept for the rest, spend, clock)."""
        return (self.walk_calls < REF_CALLS and self.calls + 14 <= ALL_CALLS and self.cost < SPEND_HARD
                and self.elapsed() < CLOCK_SOFT - (100 if len(self.items) >= 3 else 30) and not self.finished)

    def ask_walk(self, prompt: str, room: int) -> dict[str, Any]:
        if not self.walk_ok():
            return {}
        self.walk_calls += 1
        got = self.ask_json(prompt, [LIGHT, HEAVY], room, plain=True)
        return got if isinstance(got, dict) else {}

    def walk_pointer(self, need: str, budget: list[int]) -> tuple[list[str], str]:
        got = self.ask_walk(POINTER + "\n\nREQUIREMENT:\n" + need, 700)
        if not (str(got.get("seller") or "").strip() or str(got.get("kind") or "").strip()):
            return [], ""
        prop = str(got.get("property") or "").strip() or need.partition(":")[0].strip()
        kind_words = _plain_words(got.get("kind") or "")
        if not str(got.get("seller") or "").strip() and kind_words and any(kind_words <= (_plain_words(it.name) | _plain_words(it.shelf)) for it in self.items):
            self.ref_internal.add(need)
            self.trace.append({"kind": "pointer_internal", "need": need[:160]})
            return [], ""
        hop: dict[str, Any] = got
        chain: list[str] = []
        for _ in range(REF_HOPS):
            if self.elapsed() > CLOCK_SOFT - 30:
                break
            ident, read = self.find_pointed(hop, prop, budget, frozenset(chain))
            if not ident:
                break
            chain.append(ident)
            nxt = read.get("next")
            if isinstance(nxt, dict) and (str(nxt.get("seller") or "").strip() or str(nxt.get("kind") or "").strip()) and ident not in chain[:-1]:
                hop = nxt
                continue
            return chain, str(read.get("value") or "").strip()
        return chain, ""

    def seller_of(self, ident: str) -> set[str]:
        row = self.rows.get(ident) or {}
        words: set[str] = set()
        for key, value in row.items():
            if isinstance(value, str) and "seller" in key.lower() and len(value.split()) <= 4:
                words |= _plain_words(value)
        return words

    def page_is_the_shops(self, seen: Any, want: set[str]) -> bool:
        """A result page made (at least 80%) of the rows of the shop a pointer names: more of them may follow on the next page."""
        rows = list(self.records_in(seen))
        return bool(rows) and sum(1 for i, _ in rows if not want or want <= self.seller_of(i)) >= 0.8 * len(rows)

    def of_kind(self, ident: str, kind: str) -> bool:
        """The listing's category label names the kind of product a pointer describes."""
        cat_key = self.category_key()
        return bool(kind and cat_key and _stem(kind) & _stem(str((self.rows.get(ident) or {}).get(cat_key) or "")))

    def pointer_rows(self, hop: dict[str, Any], budget: list[int]) -> list[str]:
        """Listings of the shop a pointer names, found by searching under the words the pointer gives: the shop, the kind and the
        described values first (the narrowest), then shop plus kind, then shop alone. A store's search wants every word
        of a query, so the narrow one often finds none and costs a step. Rows of another shop or kind are left out."""
        search = self.roles.get("search")
        kind, seller = str(hop.get("kind") or "").strip(), re.sub(r"[\u2019']s\b", "", str(hop.get("seller") or "")).strip()
        values = " ".join(a["value"] for a in self.attr_list(hop))
        queries: list[str] = []
        for text in (f"{seller} {kind} {values}", f"{seller} {kind}", seller):
            text = re.sub(r"\s+", " ", text).strip()
            if text and text not in queries:
                queries.append(text)
        want = _plain_words(seller)
        found: list[str] = []
        for number, query in enumerate(queries[: 2 if self.thrift else 3]):
            args = self.search_args(query)
            if not search or not args or budget[0] <= 0 or self.finished or self.left() < 8:
                break
            reply = self.act(search, args)
            budget[0] -= 1
            if reply is None:
                break
            seen = reply.get("observation")
            self.observe(search, seen)
            added = 0
            for ident, row in self.records_in(seen):
                self.search_keys |= set(row)
                if ident not in found and (not want or want <= self.seller_of(ident)):
                    found.append(ident)
                    added += 1
            # the search matched more rows than it showed: further pages while they bring more of the shop's rows (a shop sells
            # a dozen or two listings of a kind, and the one a pointer means can sit on the second page)
            later = 0
            plan = self.paging(search, seen)
            full_page = self.page_is_the_shops(seen, want)
            while plan and plan[3] is not None and (full_page or (len(found) < 2 and later < 1)) and later < (1 if self.thrift else 2) and budget[0] > 0 and self.left() >= 12:
                more = _shape({**args, plan[2]: plan[3]}, self.schemas.get(search), "args", [])
                reply = self.act(search, more)
                budget[0] -= 1
                later += 1
                if reply is None:
                    break
                seen = reply.get("observation")
                self.observe(search, seen)
                added = 0
                for ident, row in self.records_in(seen):
                    self.search_keys |= set(row)
                    if ident not in found and (not want or want <= self.seller_of(ident)):
                        found.append(ident)
                        added += 1
                plan = self.paging(search, seen)
                full_page = self.page_is_the_shops(seen, want)
            # at least two of the shop's rows seen, one of the pointer's kind (or a second query done): enough candidates
            if len(found) >= 2 and (number >= 1 or any(self.of_kind(i, kind) for i in found)):
                break
        if not found and want:
            # rows that carry no seller key: the shop's name in the listing's own words
            found = [i for i in self.seen_order if i in self.rows and want <= _plain_words(self.listing_text(i)) and self.buyable(i)][:8]
        if kind and found:
            cat_key = self.category_key()
            same = [i for i in found if cat_key and _stem(kind) & _stem(str((self.rows.get(i) or {}).get(cat_key) or ""))]
            found = same or found
        self.walked.update(found)
        return found

    def search_args(self, text: str) -> dict[str, Any] | None:
        tool = self.roles.get("search")
        schema = self.schemas.get(tool or "") or {}
        props = schema.get("properties") or {}
        texts = [k for k, v in props.items() if isinstance(v, dict) and "string" in _kinds(v) and "enum" not in v]
        target = next((k for k in schema.get("required") or [] if k in texts), None) or (texts[0] if len(texts) == 1 else None)
        if not tool or not target:
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

    def clean_text(self, ident: str) -> str:
        """A listing's own words without the sentences that say what it is not or that send the reader to another listing."""
        parts = re.split(r"(?<=[.!?;])\s+|\s\|\s|\n", self.listing_text(ident))
        return " ".join(part for part in parts if part and not _NEGATING.search(part))

    def squeezed_text(self, ident: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", self.clean_text(ident).lower())

    def attr_list(self, hop: dict[str, Any]) -> list[dict[str, str]]:
        """The properties a pointer names for the listing as {"property", "value"} pairs, whatever shape the reader gave them in."""
        raw = hop.get("attrs")
        if isinstance(raw, dict):
            raw = [{"property": k, "value": v} for k, v in raw.items()]
        out: list[dict[str, str]] = []
        for entry in raw if isinstance(raw, list) else []:
            if isinstance(entry, dict):
                prop = str(entry.get("property") or entry.get("name") or entry.get("key") or "").strip()
                value = str(entry.get("value") if entry.get("value") is not None else "").strip()
            elif isinstance(entry, str) and ":" in entry:
                prop, _, value = entry.partition(":")
                prop, value = prop.strip(), value.strip()
            else:
                continue
            if prop and value:
                out.append({"property": prop, "value": value})
        return out

    def says_parts(self, text: str) -> tuple[set[str], set[str]]:
        """What a pointer's own words commit to: its figures with their units, squeezed ("16 gb" -> "16gb"), and its content words."""
        text = re.split(r";|\s[\u2014\u2013-]\s", text)[0]
        cut = _NEGATING.search(text)
        text = text[:cut.start()] if cut else text
        figures = {m.group(1) + (m.group(2) or "").lower() for m in re.finditer(r"(\d+(?:\.\d+)?)\s*([A-Za-z]{1,12})?", text) if m.group(2) or len(m.group(1)) >= 2}
        words = {w for w in _plain_words(_NUMBER.sub(" ", text)) if len(w) >= 3 and w not in _PLAIN_STOP}
        return figures, words

    def says_score(self, ident: str, parts: tuple[set[str], set[str]]) -> int:
        """How well a page's own words and attributes match a pointer's words: a figure counts three, a content word one."""
        figures, words = parts
        squeezed = self.squeezed_text(ident)
        have = _plain_words(self.clean_text(ident))
        return 3 * sum(1 for f in figures if f in squeezed) + sum(1 for w in words if w in have)

    def attr_fit(self, ident: str, attrs: list[Any]) -> tuple[int, int]:
        """(described properties the listing's own attributes state, ones they contradict): a property whose key holds
        another word value is a miss; numbers count only when the units agree, since no table converts them here."""
        block = _fact_block(self.full.get(ident) or {}) or {}
        text = _plain_words(self.clean_text(ident))
        squeezed_page = self.squeezed_text(ident)
        hit = miss = 0
        for attr in attrs:
            if not isinstance(attr, dict) or not str(attr.get("property") or "").strip() or not str(attr.get("value") or "").strip():
                continue
            prop, value = str(attr["property"]), str(attr["value"])
            if value.strip().lower() in ("yes", "true", "y") and not any(_names(_stem(prop), _stem(k)) for k in block):
                # a note that lists values only ("dimmable, usb") read as properties that hold a yes: the property's words are the value
                prop, value = "", prop
            want = {w for w in _plain_words(value) if w not in _PLAIN_STOP}
            if not want:
                continue
            keys = [k for k in block if _names(_stem(prop), _stem(k))]
            have = set().union(*[_plain_words(block[k]) for k in keys]) if keys else set()
            # a note writes some values with their property's name ("rubber accessories material"): the name's words are no part of the value
            named = _plain_words(prop).union(*[_plain_words(k) for k in keys])
            core = want - named
            if keys and (want <= have or (core and core <= have)):
                hit += 1
            elif keys:
                numbers = re.findall(r"\d+(?:\.\d+)?", _no_commas(value))
                if numbers and any(re.findall(r"\d+(?:\.\d+)?", _no_commas(str(block[k]))) for k in keys):
                    # a figure: the same number states it, another number with the same unit contradicts it, other units say nothing
                    held = [(re.findall(r"\d+(?:\.\d+)?", _no_commas(str(block[k]))), _plain_words(re.sub(r"[\d.]+", " ", str(block[k])))) for k in keys]
                    unit = _plain_words(re.sub(r"[\d.]+", " ", value)) - _PLAIN_STOP
                    if any(set(nums) & set(numbers) for nums, _ in held):
                        hit += 1
                    elif any(_units_agree(unit, words - _PLAIN_STOP) is not False for _, words in held):
                        miss += 1
                elif any((core or want) <= _plain_words(block[k]) for k in block if k not in keys and (_stem(k) & _stem(prop)) - _BROAD):
                    # a note names a value without its property, and the reader files it under a near one ("platform heel" under the
                    # heel height): the page states it under a key of kindred name
                    hit += 1
                else:
                    miss += 1
            elif (want <= text or (len(re.sub(r"[^a-z0-9]+", "", value.lower())) >= 3 and re.sub(r"[^a-z0-9]+", "", value.lower()) in squeezed_page)
                  or any(want <= _plain_words(v) for v in block.values() if not isinstance(v, (dict, list)))):
                # no key names the property (a note lists values without their properties: "led, colour-changing, solar"): the
                # value stated under any key, or in the page's words, counts
                hit += 1
        return hit, miss

    def find_pointed(self, hop: dict[str, Any], prop: str, budget: list[int], visited: frozenset[str] = frozenset()) -> tuple[str | None, dict[str, Any]]:
        """The page a pointer describes among the listings of its shop. Attributes are read, closest candidates first, until one
        listing states every described property; only listings that do (or that the light model picks, converting units) are
        opened, and a page counts only when it carries the shop's note and the note states the value or sends the reader on.
        A look-alike of the same shop states the same properties and its own value: the note is what tells the page."""
        # a page earlier in this chain is not the next link (a page that names the shop in its own note would be found again)
        rows = [i for i in self.pointer_rows(hop, budget) if i not in visited]
        if not rows:
            return None, {}
        valid = self.attr_list(hop)
        squeeze = lambda t: re.sub(r"[^a-z0-9]+", "", str(t).lower())  # noqa: E731
        wanted = set().union(*[_plain_words(a.get("value") or "") for a in valid]) if valid else set()

        def closeness(i: str) -> int:
            text = squeeze(self.listing_text(i))
            return -(2 * sum(1 for a in valid if squeeze(a["value"]) and squeeze(a["value"]) in text) + len(wanted & _plain_words(self.listing_text(i))))

        order = sorted(rows, key=lambda i: (closeness(i), rows.index(i)))
        if self.facts_tool() and (len(rows) > 1 or valid):
            reads = 0
            while budget[0] > 0 and reads < (4 if self.thrift else 3 * REF_CANDIDATES):
                # one listing stating every described property is no proof: look-alikes of one shop state the same ones,
                # so at least REF_CANDIDATES listings are read before the first match is trusted
                fitting = [i for i in rows if i in self.full and self.attr_fit(i, valid) == (len(valid), 0)]
                if valid and reads >= (2 if self.thrift else REF_CANDIDATES) and fitting and (len(fitting) == 1 or reads >= (2 if self.thrift else REF_PAGES)):
                    break
                ranked = [i for i in rows if i not in self.full][:1]
                todo = ranked + [i for i in order if i not in self.full and i not in ranked][:1]
                if not todo or not self.fill_facts(todo, 8):
                    break
                budget[0] -= len(todo)
                reads += len(todo)
        fits = {i: self.attr_fit(i, valid) for i in rows}
        live = [i for i in order if fits[i][1] == 0]
        full = [i for i in live if valid and fits[i] == (len(valid), 0)]
        self.trace.append({"kind": "pointed", "rows": rows[:14], "attrs": _clip(valid, 60), "full": full[:6], "contradicted": [i for i in rows if fits[i][1] > 0][:6]})
        tried: list[str] = []

        def attempt(ident: str, by_words: bool = False, picked: bool = False) -> dict[str, Any] | None:
            tried.append(ident)
            read = self.read_page(ident, prop, budget, hop, bare_ok=bool(valid) and len(full) == 1 and ident in full)
            value = str(read.get("value") or "").strip()
            nxt = read.get("next")
            sends = isinstance(nxt, dict) and bool(str(nxt.get("seller") or "").strip() or str(nxt.get("kind") or "").strip())
            if read.get("note") is not True or not (value or sends):
                return None
            # confirmed by the page's own words and attributes (the note is among them now) or by the light model's reading of the
            # page against the description; a look-alike's note says its own version ("a 1 piece pack" for "1 set")
            hits, misses = self.attr_fit(ident, valid)
            # the reader's "it is another listing" vetoes a page only where the code cannot tell the page itself (a note that sends the
            # reader on says "not this item" and is taken for that): the one listing of the shop that states every described property,
            # the shop's only listing of the kind, and the light model's first pick among the shop's listings (it read their attributes
            # side by side, converting units) are the page unless the code finds a described property contradicted
            alone = (bool(valid) and hits == len(valid) and not misses and len(full) == 1) or (by_words and len(pool) == 1) or picked
            if misses or (read.get("is_it") is False and not alone) or not (read.get("is_it") is True or (valid and hits == len(valid)) or by_words or picked):
                return None
            return read

        def usable() -> bool:
            return budget[0] > 0 and not self.finished and self.left() >= 8 and self.elapsed() < CLOCK_SOFT - 30 and self.walk_ok()

        says = self.says_parts(str(hop.get("says") or ""))
        has_says = bool(says[0] or says[1])
        score = (lambda i: self.says_score(i, says)) if has_says else (lambda i: 0)
        by_words = False
        if valid:
            pool = list(full)
        else:
            # nothing to check the attributes against: the pointer's own words must pick one page out clearly
            pool = sorted(live, key=lambda i: (-score(i), live.index(i)))
            if len(pool) == 1:
                by_words = True        # the shop has one listing of that kind: no other page it could be
            elif len(pool) > 1:
                top = score(pool[0])
                tied = [i for i in pool if score(i) == top]
                if not has_says or top < 2 or len(tied) > 2:
                    return None, {}
                pool, by_words = tied, True
        accepted: list[tuple[str, dict[str, Any]]] = []
        pair = by_words and len(pool) == 2          # two pages the words cannot tell apart: both are read and must agree
        for ident in pool[:REF_PAGES]:
            if not usable():
                break
            got = attempt(ident, by_words)
            if not got:
                continue
            accepted.append((ident, got))
            # done when the reader says it is the described listing, when every figure of the pointer's words is on the page, when
            # there are no words to compare by, or when a second page has passed
            sure = got.get("is_it") is True or (bool(says[0]) and all(f in self.squeezed_text(ident) for f in says[0]))
            if not has_says or (not pair and sure) or len(accepted) >= (2 if (len(pool) <= 2 or self.thrift) else REF_ACCEPT):
                break
        if not accepted and valid and len(tried) < REF_PAGES and usable():
            # no listing stating every described property was the page: the light model picks among the rest
            rest = [i for i in live if i not in tried][:10]
            # the chooser tells listings apart by their attributes: those not read yet are read first (a few, within the budget)
            unread = [i for i in rest if i not in self.full][:6]
            while unread and budget[0] > 6 and self.facts_tool() and self.walk_ok():
                part, unread = unread[:2], unread[2:]
                if not self.fill_facts(part, 8):
                    break
                budget[0] -= len(part)
            for rank, ident in enumerate(self.choose_described(hop, valid, rest)[: REF_PAGES - len(tried)]):
                if not usable():
                    break
                got = attempt(ident, picked=rank == 0)
                if got:
                    accepted.append((ident, got))
                    break
        if len(accepted) <= 1:
            return accepted[0] if accepted else (None, {})
        # several pages passed: the pointer's figures and words that tell them apart decide (a word every one of them has, such as the
        # pointer's closing remark about taking the value from that listing, tells nothing)
        figs, words = says
        cover = {a[0]: ({f for f in figs if f in self.squeezed_text(a[0])}, {w for w in words if w in _plain_words(self.clean_text(a[0]))}) for a in accepted}
        common_f = set.intersection(*[c[0] for c in cover.values()])
        common_w = set.intersection(*[c[1] for c in cover.values()])
        rel = lambda i: 3 * len(cover[i][0] - common_f) + len(cover[i][1] - common_w)  # noqa: E731
        ranked = sorted(accepted, key=lambda a: -rel(a[0]))
        first, second = ranked[0], ranked[1]
        if rel(first[0]) != rel(second[0]):
            return first
        same = str(first[1].get("value") or "") == str(second[1].get("value") or "") and bool(first[1].get("next")) == bool(second[1].get("next"))
        return first if same else (None, {})

    def choose_described(self, hop: dict[str, Any], attrs: list[dict[str, Any]], cands: list[str]) -> list[str]:
        """The listings, among a shop's candidates, that the described properties fit, best first (read by the light model off
        the candidates' own attributes, so a length in metres meets the same length in feet)."""
        # only listings whose attributes have been read can be told apart by them
        cands = [i for i in cands if i in self.full]
        if not cands or self.cost > SPEND_HARD:
            return []
        lines = []
        for ident in cands:
            block = _fact_block(self.full.get(ident) or {}) or {}
            title = str((self.rows.get(ident) or {}).get("title") or "")[:80]
            shown = "; ".join(f"{k}: {v}" for k, v in list(block.items())[:24])[:420]
            lines.append(f"{ident} | {title} | {shown}")
        wanted = "; ".join(f"{a.get('property')}: {a.get('value')}" for a in attrs) or "(none given)"
        prompt = (CHOOSE + f"\n\nSHOP: {hop.get('seller') or ''}\nKIND: {hop.get('kind') or ''}\nDESCRIPTION: {wanted}\n\nLISTINGS:\n" + "\n".join(lines))
        got = self.ask_walk(prompt, 600)
        out = [str(i) for i in _items(got.get("ids")) if str(i) in cands]
        return list(dict.fromkeys(out))

    def note_text(self, ident: str) -> str:
        """The words a page view adds to the listing's search row (the keys no search row carries, apart from the attribute
        tables): the shop's note on the page. Empty when the page has none."""
        record = self.rows.get(ident) or {}
        if not self.search_keys:
            return ""
        return " ".join(str(v).strip() for k, v in record.items() if k not in self.search_keys and k != self.id_key and isinstance(v, str) and v.strip())

    def real_redirect(self, onward: Any, prop: str) -> Any:
        """A redirect the note reader returned, dropped when its quoted words are only a "... exists elsewhere" remark that does not name the property."""
        if isinstance(onward, dict):
            says = str(onward.get("says") or "")
            if re.search(r"\belsewhere\b", says, re.I) and not (_plain_words(prop) & _plain_words(says)):
                self.trace.append({"kind": "decoy_redirect_dropped", "says": says[:120]})
                return None
        return onward

    def fix_next_shop(self, onward: Any, ident: str) -> Any:
        """The shop of the listing a note sends on to, when the reader named the page's own shop but its own quoted words name another."""
        if not isinstance(onward, dict):
            return onward
        own = self.seller_of(ident)
        named = _plain_words(str(onward.get("seller") or "").replace("'s", ""))
        if not own or (named and not named <= own):
            return onward                                   # the reader named another shop than the page's own: taken as it is
        skip = _plain_words(str(onward.get("kind") or ""))   # words of the kind and of the properties are no shop
        for a in self.attr_list(onward):
            skip |= _plain_words(str(a.get("property")) + " " + str(a.get("value")))
        known = set().union(*[self.seller_of(i) for i in self.rows]) if self.rows else set()
        words = [w for w in re.findall(r"[A-Z][A-Za-z]{2,}", str(onward.get("says") or "").replace("'s", " "))
                 if w.lower() not in ("the", "this", "that", "these", "those") and not _plain_words(w) <= own and not _plain_words(w) <= skip]
        seen = [w for w in words if _plain_words(w) <= known]
        pick = seen[0] if seen else (words[0] if len({w.lower() for w in words}) == 1 else "")
        if pick:
            self.trace.append({"kind": "next_shop_fixed", "was": str(onward.get("seller") or ""), "now": pick})
            return {**onward, "seller": pick}
        return onward

    def read_page(self, ident: str, prop: str, budget: list[int], hop: dict[str, Any] | None = None, bare_ok: bool = False) -> dict[str, Any]:
        """Open a listing's page in full (the page view, then its attributes if no view gave them) and read it for the property.
        Without the shop's note on the page nothing is asked of the model: such a page is no link of a chain, unless it is the
        last one (bare_ok: the code found it to be the one listing that states every described property) and its attributes state the value."""
        tool = self.roles.get("offer") or self.roles.get("open")
        target = self.single_target(tool)
        self.walked.add(ident)
        if tool and target and ident not in self.opened and budget[0] > 0:
            faults: list[str] = []
            args = _shape({target: ident}, self.schemas.get(tool), "args", faults)
            if not faults:
                reply = self.act(tool, args)
                budget[0] -= 1
                if reply is not None:
                    seen = reply.get("observation")
                    self.observe(tool, seen)
                    self.note_view(tool, seen)
                    if tool == self.offer_tool():
                        self.offer_seen[ident] = self.step
        bare = not self.note_text(ident)
        if bare and not (bare_ok and ident in self.opened):
            return {"note": False}
        if ident not in self.full and self.facts_tool() and budget[0] > 0:
            budget[0] -= len(self.fill_facts([ident], 6))
        hop = hop or {}
        described = "; ".join(f"{a['property']}: {a['value']}" for a in self.attr_list(hop)) or "none given"
        says = str(hop.get("says") or "").strip()
        wanted = (f"\n\nDESCRIPTION OF THE WANTED LISTING: shop {hop.get('seller') or '?'}; kind {hop.get('kind') or '?'}; properties {described}"
                  + (f"; in the words of the page that points at it: {says[:300]}" if says else ""))
        # the shop's note is read alone first: a page whose attributes state a value of their own still sends the reader on when
        # its note says so, and a reader who sees the whole record takes "not this item" for a sign that the page is another listing
        sent = {} if bare else self.ask_walk(NOTE_ONLY.replace("PROPERTY", f"{prop!r}") + wanted + f"\n\nPROPERTY: {prop}\nNOTE: {self.note_text(ident)[:900]}", 700)
        onward = self.fix_next_shop(self.real_redirect(sent.get("sends"), prop), ident)
        if isinstance(onward, dict) and (str(onward.get("seller") or "").strip() or str(onward.get("kind") or "").strip()):
            same = sent.get("is_it")
            return {"note": True, "value": "", "next": onward, "is_it": same if isinstance(same, bool) else None}
        if not sent.get("sends") and str(sent.get("value") or "").strip():
            same = sent.get("is_it")                        # the note sends nowhere and states the value: the last page of the chain
            return {"note": True, "value": str(sent["value"]).strip(), "next": None, "is_it": same if isinstance(same, bool) else None}
        prompt = (PAGE_NOTE.replace("PROPERTY", f"{prop!r}") + wanted
                  + f"\n\nPROPERTY: {prop}\nPAGE ID: {ident}\nPAGE RECORD:\n" + self.page_record(ident))
        got = self.ask_walk(prompt, 800)
        got["note"] = True
        got["next"] = self.fix_next_shop(got.get("next"), ident)
        if not str(got.get("value") or "").strip() and not got.get("next") and str(sent.get("value") or "").strip():
            got["value"] = str(sent["value"]).strip()
        return got

    def page_record(self, ident: str) -> str:
        """A listing's record for the page reader: the page's own words first (a seller's note sits among them), then the
        attributes, each value kept short, so a long description or a long attribute table cannot push the note out of view."""
        record = self.full.get(ident) or self.rows.get(ident) or {}
        words = {k: v for k, v in record.items() if not isinstance(v, (dict, list))}
        facts = {k: v for k, v in record.items() if isinstance(v, (dict, list))}
        return (_dump(_clip(words, 700)) + "\n" + _dump(_clip(facts, 120)))[:6000] if record else "(no record seen)"

    def property_keys(self, prop: str, block: dict[str, Any]) -> tuple[list[str], bool]:
        """The listing's attribute keys that name a requirement's property, and whether any names it exactly (the same words).
        A property of only broad words (type, size, style) is named by an exact key alone."""
        mine = _key_words(prop)
        if not mine:
            return [], False
        exact = [k for k in block if _key_words(k) == mine]
        broad_only = mine <= _BROAD
        loose = [] if broad_only else [k for k in block if k not in exact and (mine <= _key_words(k) or (len(_key_words(k)) >= 1 and _key_words(k) <= mine and not _key_words(k) <= _BROAD))]
        return exact + loose, bool(exact)

    def judge(self, ident: str, need: str) -> tuple[str, str]:
        """One requirement against one listing's own attributes, read by code: ("ok", ""), ("bad", why) or ("unknown", why).
        "bad" is said only when a key that names the property exactly holds another value, or a figure in the same unit that
        is not the wanted one or lies outside a limit: no table converts units here, and a different unit says nothing."""
        prop, _, value = need.partition(":")
        prop, value = prop.strip(), value.strip()
        if not prop or not value:
            return "unknown", "no value to check"
        if need not in [other for _, other in self.refs] and (_POINTS_AT.search(value) or len(value.split()) > 12):
            return "unknown", "the value points at another listing"
        if any(need == other for _, other in self.refs):
            known = self.ref_values.get(need)
            if not known:
                return "unknown", "its value is on another listing, not known yet"
            if need in self.ref_final:
                value = self.ref_final[need][1]
            elif re.search(r"\b(?:other than|different|unlike|not the same|differs?)\b", value, re.I):
                value = "not " + known
            elif re.search(r"\b(?:very same|same as|same|identical|equal)\b", value, re.I):
                value = known
            else:
                return "unknown", "a relation to another listing's value that code does not judge"
        block = _fact_block(self.full.get(ident) or {}) or {}
        if not block:
            return "unknown", "attributes not read"
        keys, exact = self.property_keys(prop, block)
        if not keys:
            # no attribute names the property: the wanted value stated in the listing's own title or description counts (words only: a
            # figure, a limit or an exclusion is not read off a title)
            words = {w for w in _plain_words(value) if (len(w) >= 2 or w.isdigit()) and w not in _PLAIN_STOP}
            # (a sentence that says what the listing is NOT, such as a look-alike's note, states nothing)
            top = _plain_words(" ".join(part for v in (self.rows.get(ident) or {}).values() if isinstance(v, str)
                                        for part in re.split(r"(?<=[.!?;])\s+|\s\|\s|\n", v) if part and not _NEGATING.search(part)))
            if words and not _NUMBER.search(value) and not _comparison(value)[0] and not _NEGATED.match(value) and not _HEDGED.search(value) and _covered(words, top):
                return "ok", ""
            return "unknown", f"no attribute names {prop}"
        shown = [str(block[k]) for k in keys]
        have = set().union(*[_plain_words(s) for s in shown])
        content = lambda text: {w for w in _plain_words(text) if (len(w) >= 2 or w.isdigit()) and w not in _PLAIN_STOP}  # noqa: E731
        whole = content(value)
        if whole and _covered(whole, have):
            return "ok", ""
        # the same word spelled with a hyphen or a space ("hi-ace", "hiace")
        squeezed = re.sub(r"[^a-z0-9]+", "", value.lower())
        if len(squeezed) >= 4 and any(squeezed in re.sub(r"[^a-z0-9]+", "", s.lower()) for s in shown):
            return "ok", ""
        negated = _NEGATED.match(value)
        direction, starts = _comparison(value)
        figures = _NUMBER.findall(value)
        spans = _RANGE.match(value.strip())
        both = bool(_CMP_LOW.search(value)) and bool(_CMP_HIGH.search(value))
        # figures are judged on the key that names the property exactly when there is one (a neighbouring key such as a seat height
        # says nothing about a height)
        measured = [str(block[k]) for k in keys if _key_words(k) == _key_words(prop)]
        neighbours = [str(block[k]) for k in keys if _key_words(k) != _key_words(prop)]
        if spans:
            # a plain range ("20-30 cm", "from 20 to 30 cm", "between 20 and 30 cm"): the listing's figure, in the same unit, lies inside it or not
            lo, hi = sorted((float(spans.group(1)), float(spans.group(3))))
            return self.judge_span(measured, neighbours, _figures(spans.group(2) + " " + spans.group(4))[1], lo, hi, f"the range {value!r}", keys[0])
        if both and direction is None and len(figures) == 2:
            # a two-sided limit ("at least 20 and at most 30 cm")
            lo, hi = sorted((float(figures[0]), float(figures[1])))
            return self.judge_span(measured, neighbours, _figures(_CMP_HIGH.sub(" ", _CMP_LOW.sub(" ", value)))[1], lo, hi, f"the limits {value!r}", keys[0])
        if (both and direction is None) or (negated and figures and direction is None):
            return "unknown", "a phrasing of limits that code does not read"
        if direction and figures:
            limit = float(figures[0])
            unit = _figures(_CMP_HIGH.sub(" ", _CMP_LOW.sub(" ", value)))[1]
            # "not under 30" negates the comparison it holds ("no more than 5" is a comparison of its own, the negation inside the phrase)
            flip = bool(negated) and starts >= negated.end()
            wants_low = (direction == "low") != flip
            return self.judge_span(measured, neighbours, unit, limit if wants_low else None, None if wants_low else limit, f"the limit {value!r}", keys[0])
        if negated:
            want = content(value[negated.end():])
            exact_have = set().union(*[_plain_words(str(block[k])) for k in keys if _key_words(k) == _key_words(prop)]) if exact else set()
            if want and exact_have and _covered(want, exact_have):
                return "bad", f"{keys[0]} is {shown[0]!r}, which the requirement excludes"
            return ("ok", "") if exact else ("unknown", "")
        options = [o.strip() for o in re.split(r"\s*(?:;|\bor\b)\s*", value) if o.strip()] or [value]
        # a bracketed word is another name for the value ("hook and loop (velcro)"): either name counts (not a bracketed figure)
        for option in list(options):
            inner = re.fullmatch(r"(.+?)\s*\(([^()\d]+)\)", option)
            if inner:
                options.extend(o for o in (inner.group(1).strip(), inner.group(2).strip()) if o and o not in options)
        against: list[bool | None] = []
        for option in options:
            want = content(option)
            if not want:
                against.append(None)
                continue
            words = [w for w in re.findall(r"[a-z0-9]+", option.lower()) if w not in _PLAIN_STOP]
            initials = "".join(w[0] for w in words) if len(words) >= 2 else ""
            if _covered(want, have) or (initials and initials in have):
                return "ok", ""
            wanted, unit = _figures(option)
            if not wanted:
                against.append(False)
                continue
            verdict: bool | None = None
            for s in shown:
                seen, theirs = _figures(s)
                agree = _units_agree(unit, theirs)
                if not seen or agree is False:
                    continue
                if set(wanted) <= set(seen):
                    return "ok", ""
                if agree is True and verdict is None:
                    verdict = False
            against.append(verdict)
        if not exact:
            return "unknown", f"{keys[0]} may name another property"
        if against and all(v is False for v in against):
            # the wanted words stated anywhere on the listing (its text, or another attribute such as "Internal, Swappable: Yes" under
            # a key of another name) leave the verdict open
            # (words only: a figure is found in some other attribute by chance, and figures are judged by their units above)
            anywhere = _plain_words(self.listing_text(ident) + ("" if re.search(r"\d", options[0]) else " " + _dump(self.full.get(ident) or {})))
            if all({w for w in _plain_words(o) if w not in _PLAIN_STOP} <= anywhere for o in options[:1]):
                return "unknown", "the value appears in the listing's text"
            return "bad", f"{keys[0]} is {shown[0]!r}, not {value!r}"
        return "unknown", "not decided"

    def judge_span(self, shown: list[str], loose: list[str], unit: set[str], low: float | None, high: float | None, why: str, key: str) -> tuple[str, str]:
        """A listing's figure against a lower and/or an upper bound, in the same unit only. `shown` are the values of the keys that name
        the property exactly, `loose` those of neighbouring keys (a seat height beside a height, a height in millimetres): the exact keys
        decide when they can, a neighbour that says otherwise or cannot be compared keeps the verdict open, and "bad" needs every key
        that can speak to be outside the bounds."""
        def verdicts(values: list[str]) -> list[bool | None]:
            out: list[bool | None] = []
            for text in values:
                # a figure after a word of reach ("expandable ... up to 1 tb", "supports 256 gb") says what the product can take, not what it is
                seen, theirs = _figures(_REACH.sub(" ", text))
                if len(seen) != 1 or _units_agree(unit, theirs) is False:
                    out.append(None)
                    continue
                figure = float(seen[0])
                out.append((low is None or figure >= low) and (high is None or figure <= high))
            return out

        own, other = verdicts(shown), verdicts(loose)
        if any(v is True for v in own) or (not own and any(v is True for v in other)):
            return "ok", ""
        if own and all(v is False for v in own) and all(v is False for v in other):
            return "bad", f"{key} is {shown[0]!r}, outside {why}"
        return "unknown", "figures in other units or ranges, or keys that disagree"

    def item_needs(self, number: int) -> list[str]:
        """The item's requirements that code may judge: none the shopper dropped (the note's wording of a drop may differ from
        the requirement's, so a dropped text whose words all lie in the requirement counts) and none a store or shopper note
        touched (a change in force that code does not read)."""
        item = self.items[number - 1]
        dropped = [_plain_words(d) for d in item.dropped if d]
        noted = [_plain_words(n) for n in list(self.notes) + sorted(self.heard)]
        out = []
        for need in item.musts + item.told:
            words = _plain_words(need)
            prop = _plain_words(need.partition(":")[0])
            if need in item.dropped or any(d and d <= words for d in dropped) or re.search(r"\bfit", need.partition(":")[0], re.I):
                continue
            if prop and (prop <= _STOCKISH or any(prop <= n for n in noted)):
                continue
            out.append(need)
        return out

    def live_needs(self, number: int) -> list[str]:
        """Every requirement of the item that is still in force: only what the shopper dropped is left out (a requirement a note
        touched, or one about fitting, stays and is then judged unknown, so it keeps a listing from being called verified)."""
        item = self.items[number - 1]
        dropped = [_plain_words(d) for d in item.dropped if d]
        return [n for n in item.musts + item.told if n not in item.dropped and not any(d and d <= _plain_words(n) for d in dropped)
                and not (_plain_words(n.partition(":")[0]) and _plain_words(n.partition(":")[0]) <= _STOCKISH)]

    def verified(self, number: int, skip: str = "") -> list[str]:
        """Listings of an item whose own attributes meet every requirement it has in force (none unknown), that cover the shopper's
        own model when one is named, buyable, cheapest first."""
        needs = self.live_needs(number)
        if not needs:
            return []
        own = {_code_key(c) for c in _CODE.findall(self.own_model)} or ({_code_key(self.own_model)} if self.own_model else set())
        noted = [_plain_words(n) for n in list(self.notes) + sorted(self.heard)]
        out = []
        for ident in self.full:
            if ident == skip or self.item_of(ident) != number or not self.buyable(ident) or ident in self.gone:
                continue
            if own:
                exact, families = self.fit_targets(ident)
                if not any(_covers(c, exact, families) for c in own):
                    continue
            # a requirement a store or shopper note touched is in force as changed: code cannot say it is met
            if any(_plain_words(n.partition(":")[0]) and _plain_words(n.partition(":")[0]) <= w for n in needs for w in noted):
                continue
            if all(self.judge(ident, n)[0] == "ok" for n in needs):
                out.append(ident)
        return sorted(out, key=lambda i: (self.price_of(i) if self.price_of(i) is not None else float("inf"), i))

    def clean_pick(self, number: int, fitted: bool = True) -> str | None:
        """The listing to cart for an item: a verified one, else the listing whose page was read that code finds no conflict in
        (same test as code_basket, fewest unknowns first, then the price)."""
        options = self.verified(number)
        if options:
            return options[0]
        item = self.items[number - 1]
        own = {_code_key(c) for c in _CODE.findall(self.own_model)} or ({_code_key(self.own_model)} if self.own_model else set())
        needs = [n for n in item.musts + item.told if n not in item.dropped and not re.search(r"\bfit", n.split(":")[0], re.I)]
        best: tuple[int, float, str] | None = None
        for ident in self.full:
            price = self.price_of(ident)
            if price is None or ident in self.gone or not self.buyable(ident) or self.item_of(ident) != number:
                continue
            if own and fitted:
                exact, families = self.fit_targets(ident)
                if not any(_covers(c, exact, families) for c in own):
                    continue
            states = [self.need_state(ident, n) for n in needs]
            if "bad" in states:
                continue
            rank = (states.count("unknown"), price, ident)
            if best is None or rank < best:
                best = rank
        return best[2] if best else None

    def best_listing(self, number: int, reads: int = 3) -> str | None:
        """The listing for an item the cart has no line for. Read listings first; when none was read, the attributes of the
        cheapest rows of the item (those sharing most words with its name first) are read, a few steps at most."""
        pick = self.clean_pick(number) or self.clean_pick(number, False)
        if pick:
            return pick
        item = self.items[number - 1]
        name = _stem(item.name) | _stem(item.shelf or "")
        rows = [i for i in self.seen_order if self.item_of(i) == number and i not in self.full and i not in self.gone and self.buyable(i) and self.price_of(i) is not None]
        rows.sort(key=lambda i: (-len(name & _stem(_dump(self.rows.get(i) or {}))), self.price_of(i), i))
        if rows and self.left() > self.tail() + 6 and self.elapsed() < CLOCK_HARD - 30:
            self.fill_facts(rows[:reads], self.tail() + 4)
            pick = self.clean_pick(number) or self.clean_pick(number, False)
            if pick:
                return pick
        return rows[0] if rows else None

    def basket_plan(self, whole: bool = False) -> tuple[dict[str, int], list[str], dict[str, int]] | None:
        """The cart as it should be: (lines that stay with their counts, lines to take out, lines to put in). A requested item with
        no line gets its best listing; of several lines for one item (its category label shared by no other item) the one code
        finds fewest faults in stays; a line whose category label no requested item owns is taken out. `whole` plans a basket
        from nothing (all items): None when some item has no listing to put in."""
        cat_key = self.category_key()
        if not cat_key or not self.items:
            return None
        ids = self.cart_ids() if not whole else []
        if not ids and not whole:
            return None
        count = len(self.items)
        # items with the same main noun (two kinds of pumps) cannot be told apart by a category label
        shared = set()

        def head(n: int) -> str:
            words = self.items[n - 1].name.split() or (self.items[n - 1].shelf or "").split()
            return sorted(_singular(words[-1]))[0] if words else ""

        for a in range(1, count + 1):
            for b in range(a + 1, count + 1):
                if head(a) and head(a) == head(b):
                    shared.update((a, b))
        owner: dict[str, int | None] = {}
        for ident in ids:
            label = str((self.rows.get(ident) or {}).get(cat_key) or "").strip()
            owner[ident] = self.item_of(ident, cat_key) if label else 0
        per_item: dict[int, list[str]] = {}
        for ident in ids:
            if owner[ident]:
                per_item.setdefault(owner[ident], []).append(ident)
        if ids and not per_item:
            # no line is owned by any item: the category labels cannot be matched to the items, so nothing is judged by them
            return None
        missing = [n for n in range(1, count + 1) if n not in per_item]
        if shared and missing:
            # items that share their words cannot be told apart by the category label
            missing = [n for n in missing if n not in shared]
        drop: list[str] = []
        stay = {i: self.cart_lines.get(i, 1) for i in ids}
        # lines no requested item owns (the label is known): an item's reference page, a part of a neighbouring bundle; a line that
        # shares words with a missing item may be that item's own line under a label nothing matched: it stays and the item is not filled
        for ident in ids:
            if owner[ident] is not None:
                continue
            seen = {w for w in _stem(_dump(self.rows.get(ident) or {})) if len(w) >= 4}
            claims = [n for n in missing if {w for w in _stem(self.items[n - 1].name) | _stem(self.items[n - 1].shelf or "") if len(w) >= 4} & seen]
            if claims:
                missing = [n for n in missing if n != claims[0]]
            else:
                drop.append(ident)
        # several lines for one item
        for number, lines in per_item.items():
            if len(lines) < 2 or number in shared:
                continue

            def faults(ident: str, number: int = number) -> tuple[int, int, float]:
                states = [self.judge(ident, n)[0] for n in self.item_needs(number)]
                price = self.price_of(ident)
                return (states.count("bad"), states.count("unknown"), price if price is not None else float("inf"))

            best = min(lines, key=faults)
            drop += [i for i in lines if i != best]
        if drop and not whole:
            # a line is taken out only when the cart holds more lines than the request has items and every item still has
            # a line afterwards: a cart with one line per item and a label code cannot place is the code's doubt, not an extra line
            kept_items = {owner[i] for i in ids if i not in drop and owner[i]}
            if len(ids) <= count or any(n not in kept_items for n in per_item):
                drop = []
        for ident in drop:
            stay.pop(ident, None)
        add: dict[str, int] = {}
        for number in (range(1, count + 1) if whole else missing):
            if number in shared and not whole:
                continue
            pick = self.best_listing(number)
            if not pick:
                if whole:
                    return None
                continue
            add[pick] = self.items[number - 1].qty
        return stay, drop, add

    def over_budget(self, lines: dict[str, int]) -> bool:
        total = 0.0
        for ident, count in lines.items():
            price = self.price_of(ident)
            if price is None:
                return False
            total += price * count
        return self.floor_over(total)

    def floor_over(self, total: float) -> bool:
        budget = self.budget
        return bool(total and isinstance(budget, (int, float)) and not isinstance(budget, bool) and budget > 0 and self.brief.get("budget_scope") != "each" and total > budget)

    def put_basket(self, stay: dict[str, int], drop: list[str], add: dict[str, int]) -> bool:
        """Make the cart hold `stay` and `add` and none of `drop`. True when every cart call went through."""
        cart = self.roles.get("cart")
        if not cart or self.finished:
            return False
        calls = self.cart_calls({**stay, **add}, drop)
        if calls is None or len(calls) + 2 > self.left():
            return False
        for args in calls:
            faults: list[str] = []
            args = _shape(args, self.schemas.get(cart), "args", faults)
            if faults:
                return False
            reply = self.act(cart, args)
            if reply is None or self.finished:
                return False
            self.note_cart(cart, args)
            self.observe(cart, reply.get("observation"))
        self.version += 1
        return True

    def fill_basket(self) -> str:
        """Before a checkout: a requested item with no line gets one, doubled lines of one item become one, a line no requested
        item owns is taken out. No model call. Returns what was done ("" when the cart is as it should be)."""
        if self.fills >= 2 or self.finished or not self.roles.get("cart"):
            return ""
        plan = self.basket_plan()
        if not plan:
            return ""
        stay, drop, add = plan
        if not drop and not add:
            return ""
        self.fills += 1
        done = self.put_basket(stay, drop, add)
        self.trace.append({"kind": "fill_basket", "stay": list(stay), "drop": drop, "add": add, "done": done})
        if not done:
            return ""
        return ("Kyo set the cart right by the request's items: " + ("took out " + ", ".join(drop) + "; " if drop else "")
                + ("put in " + ", ".join(f"{i} (item {self.item_of(i)})" for i in add) + "; " if add else "")
                + "each requested item now has one line.")

    def buy_instead(self) -> str:
        """The agent means to end an order of three or more items without a purchase. Such a decline was never right in the last
        race's 47 multi-item tasks (and orders of two items or fewer keep their review): a basket is built from the listings seen,
        one per item, and bought, unless an item has no listing at all or even the cheapest basket is over the budget."""
        cart, buy = self.roles.get("cart"), self.roles.get("buy")
        if len(self.items) < 3 or self.insteads >= 1 or not cart or not buy or self.finished or self.left() < len(self.items) + 4:
            return ""
        self.insteads += 1
        # the cheapest basket the rows seen make up, before any page is read: over the budget, a decline can be right
        floor = 0.0
        for number, item in enumerate(self.items, 1):
            prices = [self.price_of(i) for i in self.seen_order if self.item_of(i) == number and i not in self.gone and self.buyable(i) and self.price_of(i) is not None]
            if not prices:
                self.trace.append({"kind": "buy_instead", "no_listing_for": number})
                return ""
            floor += min(prices) * item.qty
        if self.floor_over(floor):
            self.trace.append({"kind": "buy_instead", "floor": floor, "over_budget": True})
            return ""
        plan = self.basket_plan() if self.cart_ids() else self.basket_plan(whole=True)
        if not plan:
            self.trace.append({"kind": "buy_instead", "basket": None})
            return ""
        stay, drop, add = plan
        lines = {**stay, **add}
        covered = {self.item_of(i) for i in lines}
        if any(n not in covered for n in range(1, len(self.items) + 1)):
            self.trace.append({"kind": "buy_instead", "basket": lines, "uncovered": True})
            return ""
        if self.over_budget(lines):
            self.trace.append({"kind": "buy_instead", "basket": lines, "over_budget": True})
            return ""
        if not self.put_basket(stay, drop, add):
            self.trace.append({"kind": "buy_instead", "basket": lines, "done": False})
            return ""
        if self.left() >= 2:
            self.gate("order")
        if self.finished or self.left() <= 0:
            return "bought"
        faults: list[str] = []
        args = _shape(dict(self.brief.get("buy_args") or {}), self.schemas.get(buy), "args", faults)
        if faults:
            return ""
        reply = self.act(buy, args)
        self.trace.append({"kind": "buy_instead", "basket": lines, "reply": _clip(reply, 200)})
        if reply is None:
            return ""
        self.buys += 1
        self.finish_tries += 1
        if not reply.get("finished"):
            self.refused_at = self.step
        self.observe(buy, reply.get("observation"))
        return "Kyo built a basket from the listings seen, one for each of the " + str(len(self.items)) + " requested items, and checked out: " + self.show(reply.get("observation"))

    def trail(self) -> tuple[str, bool]:
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
        while pending and self.trails < 4 and self.cost < SPEND_HARD:
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
            # a plain call: the light model's thinking filled the 1800 tokens and cut the reply in three of three checks of a live-shaped chain
            got = self.ask_json(prompt, [LIGHT, HEAVY], 1800, plain=True) or {}
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
                    self.resolve_need(need, self.ref_values[need], ids[-1])
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
        hold = bool(lost) and self.trail_blocks < 2 and self.left() > 5
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
        got = self.ask_json(prompt, [HEAVY, LIGHT], 2400)
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

    def audit(self, limit: float = SPEND_SOFT) -> str:
        if self.audited or self.cost > limit:
            return ""
        self.audited = True
        self.audit_at = self.step
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
        view = self.view_order(30)
        for ident in view:
            record = self.full.get(ident) if ident in self.full else None
            records.append(("OPENED " + _dump(_clip(record, 200))[:1200]) if record else ("ROW " + _dump(_clip(self.rows.get(ident), 120))[:500]))
        if not records:
            return None
        prompt = AUDIT + "\n\nREQUEST:\n" + self.request + "\n\nREQUIREMENTS IN FORCE:\n" + self.ledger()
        prompt += "\n\nLISTING RECORDS SEEN (newest first):\n" + "\n".join(records)
        fits = self.fit_lines(view, 20)
        if fits:
            prompt += "\n\nFIT EVIDENCE (read by code from the listings' own words):\n" + "\n".join(fits)
        got = self.ask_json(prompt, [HEAVY, LIGHT], 3000, effort="minimal")
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
        # which of a mode argument's values adds and which takes out: by the words of the values, else by their order (add first)
        options = [str(v) for v in (props[mode]["enum"] if mode else [])]
        put_in = next((v for v in options if re.search(r"add|put|insert|place|set", v, re.I)), options[0] if options else "")
        take_out = next((v for v in options if re.search(r"remov|withdraw|delet|drop|take|clear|minus|subtract", v, re.I)), options[1] if len(options) > 1 else "")
        calls: list[dict[str, Any]] = []
        for ident in drop:
            have = self.cart_lines.get(ident, 1)
            if mode:
                calls.append({target: ident, mode: take_out, **({count: have} if count else {})})
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
            calls.append({target: ident, **({mode: put_in} if mode else {}), **({count: want} if count else {})})
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
            live = [i for i in ids if not self.sold_out(i)]
            # the confirmed lead for its own item, else a lead whose attributes were read, else the first
            pick = self.confirmed if self.confirmed in live else (next((i for i in live if i in self.full), None) or (live[0] if live else None))
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
            self.finish_tries += 1
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

    def confirm_and_complete(self) -> str:
        """A reviewed lead is carted and bought by code (complete_cart checks the whole planned cart first)."""
        # a lead whose attributes were read first; a row-only lead gets its attributes read by complete_cart before the check
        live = [i for i in self.leads if not self.sold_out(i)]
        lead = next((i for i in live if i in self.full), None) or (live[0] if live else None)
        if not lead:
            return ""
        self.confirmed = lead
        self.trace.append({"kind": "decline_turned", "lead": lead, "step": self.step})
        return self.complete_cart()

    def need_state(self, ident: str, need: str) -> str:
        """What a listing's own record says about one requirement, read by code: "ok" (its value is there), "bad" (a key
        for the property holds something else) or "unknown" (no key, or a requirement code cannot judge)."""
        prop, _, value = need.partition(":")
        value = value.strip()
        if not value or _HEDGED.search(value) or any(need == other for _, other in self.refs):
            return "unknown"
        want = {w for w in _plain_words(value) if len(w) >= 2 and w not in _PLAIN_STOP}
        if not want:
            return "unknown"
        # the initials of a multi-word value ("sff" for "small form factor") state it as well
        ordered = [w for w in re.findall(r"[a-z0-9]+", value.lower()) if w not in _PLAIN_STOP]
        initials = "".join(w[0] for w in ordered) if len(ordered) >= 2 else ""
        block = _fact_block(self.full.get(ident) or {}) or {}
        keys = [k for k in block if _names(_stem(prop), _stem(k))]
        have = set().union(*[_plain_words(block[k]) for k in keys]) if keys else set()
        if keys and (want <= have or (initials and initials in have)):
            return "ok"
        if want <= _plain_words(self.listing_text(ident)):
            return "ok"
        return "bad" if keys else "unknown"

    def code_basket(self) -> dict[str, int] | None:
        """No time is left to ask a model and the cart is empty: for every item, the cheapest listing whose page was read
        that code finds no conflict in (orderable, each requirement's value where a key names the property, the fit list
        against the shopper's own model), fewest unknowns first; None when some item has none or the total breaks the budget."""
        own = {_code_key(c) for c in _CODE.findall(self.own_model)} or ({_code_key(self.own_model)} if self.own_model else set())
        lines: dict[str, int] = {}
        total = 0.0
        for number, item in enumerate(self.items, 1):
            needs = [n for n in item.musts + item.told if n not in item.dropped and not re.search(r"\bfit", n.split(":")[0], re.I)]
            best: tuple[int, float, str] | None = None
            for ident in self.full:
                price = self.price_of(ident)
                if price is None or ident in self.gone or not self.buyable(ident) or self.item_of(ident) != number:
                    continue
                if own:
                    exact, families = self.fit_targets(ident)
                    if not any(_covers(c, exact, families) for c in own):
                        continue
                states = [self.need_state(ident, n) for n in needs]
                if "bad" in states:
                    continue
                rank = (states.count("unknown"), price, ident)
                if best is None or rank < best:
                    best = rank
            if best is None:
                return None
            lines[best[2]] = item.qty
            total += best[1] * item.qty
        budget = self.budget
        if isinstance(budget, (int, float)) and not isinstance(budget, bool) and budget > 0 and total > budget:
            return None
        return lines

    def code_rescue(self) -> bool:
        """Cart the code basket and check out (two or three store calls, no model call). True when an order went in."""
        cart, buy = self.roles.get("cart"), self.roles.get("buy")
        if self.code_rescued or not cart or not buy or self.finished:
            return False
        self.code_rescued = True
        lines = self.code_basket()
        if not lines:
            self.trace.append({"kind": "code_rescue", "lines": None})
            return False
        now = self.cart_ids()
        calls = self.cart_calls(lines, [i for i in now if i not in lines])
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
            self.note_cart(cart, args)
            self.observe(cart, reply.get("observation"))
        faults = []
        args = _shape(dict(self.brief.get("buy_args") or {}), self.schemas.get(buy), "args", faults)
        if faults:
            return False
        reply = self.act(buy, args)
        self.trace.append({"kind": "code_rescue", "lines": lines, "reply": _clip(reply, 200)})
        if reply is not None:
            self.buys += 1
            self.finish_tries += 1
            self.observe(buy, reply.get("observation"))
        return bool(self.finished)

    def decline_hold(self) -> str:
        """Before ending without a purchase: review the evidence, again after any change since the last review. With few
        steps left, or once a finish has been refused, a reviewed lead is carted and bought at once when the check passes."""
        if self.audited and self.change_at > self.audit_at:
            # the task changed since the review (a refusal note, a drop, a new budget): what failed then may fit now
            self.audited = False
            self.leads, self.lead_items = [], {}
            self.refuse_blocks = 0
            self.confirmed = None
            self.completed = False
            self.pushed = False
        if self.left() <= 3:
            return ""
        urgent = self.left() <= 8 or self.finish_tries > 0 or self.pointed()
        review = self.audit(SPEND_HARD if urgent else SPEND_SOFT)
        if review:
            if urgent:
                done = self.confirm_and_complete()
                if done:
                    return done
            return "Not run yet (no step used). " + review
        if self.leads and self.left() > 4 and self.refuse_blocks == 0:
            self.refuse_blocks = 1
            self.trace.append({"kind": "refuse_blocked", "leads": list(self.leads)})
            return ("Not run yet (no step used). The review found listings that appear to fit: " + ", ".join(self.leads) + ". Open them if needed, put the best fitting one in the cart (respecting the pick rule) and check out. End without a purchase only if each of them clearly fails a requirement; then say which one in your text and send the ending call again.")
        if self.leads and self.refuse_blocks == 1:
            self.refuse_blocks = 2
            done = self.confirm_and_complete()
            if done:
                return done
        return ""

    def pointed(self) -> bool:
        """A requirement still in force takes its value from another listing (the request's or a shopper's answer's pointer)."""
        live = {need for _, need in self.needs_now()}
        return any(need in live and need not in self.ref_internal for _, need in self.refs)

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
        declared = spec["maxItems"] if isinstance(spec.get("maxItems"), int) and spec["maxItems"] > 0 else None
        if self.compare_bad:
            # what the store has refused beats what its schema declares
            learned = max(1, min(self.compare_bad - 1, max(self.compare_ok, 2)))
            return min(declared, learned) if declared else learned
        if declared:
            return declared
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
            if ending and self.left() > 2:
                if self.is_buy(name, args):
                    gap = self.safe(self.fill_basket, "")
                    if gap:
                        out.append((cid, "Not run yet. " + gap + "\nCart now: " + self.cart_text()))
                        continue
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
                    read = self.fill_facts(self.cart_ids(), self.tail() + 3) + self.cheaper_rows()
                    verdict = self.check_cart()
                    if verdict:
                        seen = ("Kyo read these listings' attributes for the check (one step each):\n" + "\n".join(read) + "\n") if read else ""
                        out.append((cid, "Not run yet. " + chain + seen + verdict + "\nCart now: " + self.cart_text()))
                        continue
                    if reread and self.cost > SPEND_SOFT:
                        out.append((cid, "Not run yet (no step used). Kyo re-read the cart's listings just now:\n" + reread + "\nCheck out again when every line still shows stock and a price inside the budget; otherwise replace that line first."))
                        continue
                else:
                    bought = self.safe(self.buy_instead, "")
                    if bought:
                        out.append((cid, bought + (" [episode finished]" if self.finished else "")))
                        continue
                    held_text = self.decline_hold()
                    if held_text:
                        out.append((cid, held_text + (" [episode finished]" if self.finished else "")))
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
                    self.facts_seen(name, observation)
                    if name == self.roles.get("open") and self.facts_from.get(name) is False:
                        other = next((t for t in self.schemas if t != name and self.single_target(t) and self.facts_from.get(t) is not False
                                      and t not in (self.roles.get("cart"), self.roles.get("buy"), self.roles.get("refuse"), self.roles.get("ask"), self.roles.get("search"))), None)
                        if other:
                            self.roles["open"] = other
                    if self.facts_from.get(name) is False and name != self.roles.get("open") and not self.roles.get("offer"):
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
            if ending:
                self.finish_tries += 1
                if not reply.get("finished"):
                    self.refused_at = self.step
            if self.is_buy(name, args):
                self.buys += 1
            text = self.show(observation) + paged + trimmed
            if name not in plain and not ending:
                text += self.cheapest_here([observation] + self.paged_obs)
            base = text
            proof = self.evidence(observation)
            if proof:
                text += "\n" + proof
            if name in readers:
                fit = self.fit_evidence(observation)
                if fit:
                    text += "\n" + fit
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

    def finish(self, again: bool = False) -> str:
        """End the episode: check out a held cart, else end without a purchase. With `again`, a refused checkout is
        returned (as the store's reply) for the agent to deal with instead of being followed by an ending without a purchase."""
        if self.finished or self.broke or self.left() <= 0:
            return ""
        buy, refuse = self.roles.get("buy"), self.roles.get("refuse")
        fresh = self.cart_view is not None and self.cart_seen_at >= self.cart_at
        holding = bool(self.cart_ids()) if fresh else bool(self.cart_lines)
        if buy and (holding or (self.cart_touched and not fresh and self.left() >= 2)):
            faults: list[str] = []
            args = _shape(dict(self.brief.get("buy_args") or {}), self.schemas.get(buy), "args", faults)
            if not faults:
                self.safe(self.fill_basket, "")
                # a reply that adds a requirement does not stop a forced finish: no steps are left to act on it
                if self.left() >= 2:
                    self.gate("order")
                if self.finished or self.left() <= 0:
                    return ""
                reply = self.act(buy, args)
                self.trace.append({"kind": "forced_buy", "reply": _clip(reply, 200)})
                if reply is not None:
                    self.finish_tries += 1
                if self.finished or self.left() <= 0:
                    return ""
                if reply is not None and again:
                    self.buys += 1
                    self.refused_at = self.step
                    self.observe(buy, reply.get("observation"))
                    return "the checkout was refused: " + self.show(reply.get("observation"))
        elif buy and not self.rescued and self.left() >= 3 and self.elapsed() < CLOCK_HARD - 20 and not self.no_credit:
            # nothing held: a reviewed basket the check passes is bought rather than ending without a purchase
            self.rescued = True
            if self.change_at > self.audit_at:
                self.audited = False
                self.completed = False
                self.leads, self.lead_items = [], {}
            if not self.leads:
                self.audit(SPEND_HARD)
            done = self.confirm_and_complete() if self.leads else ""
            if self.finished or self.left() <= 0:
                return ""
            if done and again:
                return "the checkout was refused: " + done
        if buy and not self.finished and not self.cart_ids() and self.left() >= 3:
            # an empty cart at the end: ending without a purchase is right about once in ninety races' tasks, so a
            # basket code finds no fault in is bought instead (it costs no model call, so the clock cannot stop it)
            if len(self.items) >= 3:
                self.safe(self.buy_instead, "")
                if self.finished or self.left() <= 0:
                    return ""
            if self.code_rescue() or self.finished or self.left() <= 0:
                return ""
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
                if reply is not None:
                    self.finish_tries += 1
                if reply is not None and not self.finished:
                    self.refused_at = self.step
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
        opening = self.view.get("opening")
        if opening is not None:
            self.absorb(opening)
        followed = self.safe(self.follow_refs, "")
        intro = "SHOPPER'S REQUEST:\n" + self.request
        if opening is not None:
            intro += "\n\nWHAT THE STORE SHOWS FIRST:\n" + self.show(opening)
        if followed:
            intro += "\n\n" + followed
        intro += "\n\nStart by searching for each item in its category and opening several listings in full, so the interviewer can see which properties these listings carry.\n\n" + self.status()
        self.chat = [{"role": "system", "content": GUIDE}, {"role": "user", "content": intro}]
        self.work(LOOP_CALLS, SPEND_HARD)
        if not self.finished:
            turned = self.finish(again=True)
            # the store turned the ending down (a change from the shopper, a sold-out line): a short second round deals with it
            if turned and not self.finished and not self.broke and not self.no_credit and self.left() > 1 and self.elapsed() < CLOCK_SOFT + 20:
                self.chat.append({"role": "user", "content": "[Kyo] Kyo tried to end the episode and " + turned + "\nApply what the store says, then finish.\n" + self.status()})
                self.work(self.loop_calls + 10, SPEND_HARD + 0.01, CLOCK_SOFT + 25)
            if not self.finished:
                self.finish()
        self.trace.append({"kind": "end", "finished": self.finished, "steps": self.step, "cost": round(self.cost, 5), "calls": self.calls})
        return self.trace

    def work(self, loop_cap: int, spend_cap: float, clock: float = CLOCK_SOFT) -> None:
        idle = 0
        while (not self.finished and not self.broke and self.left() > 0 and self.loop_calls < loop_cap and self.cost < spend_cap
               and self.elapsed() < clock):
            if self.ready_to_ask():
                summary = self.interview()
                if summary:
                    followed = self.safe(self.follow_refs, "")
                    self.chat.append({"role": "user", "content": summary + ("\n" + followed if followed else "") + "\n" + self.status()})
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
                followed = self.safe(self.follow_refs, "")
                self.chat.append({"role": "user", "content": (summary or "The interviewer had nothing to ask.") + ("\n" + followed if followed else "") + "\n" + self.status()})
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
