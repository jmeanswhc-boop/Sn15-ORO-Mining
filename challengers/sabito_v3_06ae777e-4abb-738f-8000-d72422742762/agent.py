"""Sabito v8.16n = v8.11b (1f65846d) + v8_12b (make_sabito_v8_16n.py).
Sabito v8.11b = v8.11 (92e99d09) + a semantic budget pick (make_sabito_v8_11b.py).
Sabito v8.11 = v8.10 (b108c48a) + the 3 accepted Codex fixes (make_sabito_v8_11.py).
Sabito v8.10 = v8.9 (07b3b770) + v8_10x+v8_10y (make_sabito_v8_10.py).
Sabito v8.9 = v8.8 (5d7aed4b) - R4: v8.0 logic + R1 + b + c1 + d1 (make_sabito_v8_9.py).
Sabito v8.8 = v8.3 (1b75bd5d) - R2 wall clock - d2 retry + R4 only after a fuller page (make_sabito_v8_8.py).
Sabito v8.3 = v8.2 (700d9e9c) - c2 cheaper-read skip, d2 retry only before CLOCK_STOP (make_sabito_v8_3.py).
Sabito v8.2 = v8.1 (8feb1954) + round-2 branches b+c+d (make_sabito_v8_2.py).
Sabito v8.1 = v8.0 (567c346f) + side-by-side limit learning, a wall clock, note-only replies (make_sabito_v8_1.py).
Sabito v8.0 - a shopping agent for the ORO benchmark (Bittensor SN15), built by make_sabito_v8_0.py.
Base: KYO v6 (public agent version 06ae3865), the race-10 winner. The logic is the base's; one constant
is spelled with a digit separator and keeps its value. Every docstring is our own and the base's comments are gone.
Entry point: agent_main(problem_data). A small model picks the store calls turn by turn. Program code tracks requested
items, wanted properties, seen listings and spend, and chooses what to ask the shopper; a stronger model reviews the
basket before checkout and reviews any planned refusal."""
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
SPEND_MORE = 0.02
SPARE_TURNS = 8
WAITS = (4, 10, 20, 35)
WAIT_TOTAL = 45
SHOW_CHARS = 450
OBS_LIMIT = 15000
HISTORY_LIMIT = 150_000
GATE_ROUNDS = 3
AUTO_PAGES = 6
CHEAPER_OPENS = 3
SEARCH_DRY = 3

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
- A requirement on a property is met when the listing's attribute for that property states the value. Key spelling does not matter (spaces, underscores, camel case, prefixes), and one element of a comma or slash list is enough. Values match by their words: case, singular or plural and plain synonyms are fine, and a value that contains the required value's words counts ("deep teal" meets teal, "extra roomy fit" meets roomy). A value with different words does not count (an abbreviation or code is not the word it stands for, and "teal green" is not "sea green"). Units are converted before comparing numbers, limits include the edge value, and a range must lie wholly inside a limit.
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


PLAN = """You choose the questions a shopping agent sends to a shopper. Shoppers hold a few firm preferences they did not write down, each about one property of one requested product. A question reveals a preference only when it names that exact property, and each question is answered only about the first property it names, so every question costs one property.

You get the request, the items with what the request already settles for each, and a numbered list of attribute keys seen on opened listings of each item's category, with how many of those listings carry the key and example values. An unwritten preference may be on a property that no listed key names.

Choose the properties in this order:
0. What the request itself asks you to check with the shopper (listed under ASK FIRST when there is any), each named by the property it is about.
1. Hinted properties. The request's story often points at an unwritten preference without giving its value: a sentence about something the shopper cannot judge or leaves to you, someone whose taste decides, the product's looks or the setting it belongs in, the person meant to use or wear it, or what is inside it. For each such sentence, name the one property it points at and the item it is about, in the order the story gives them.
Name a hinted property by its plain name: a hint about who will wear it points at gender, a hint about its looks at colour or finish (whichever the request leaves open).
2. Then the listed keys, ranked: favour the properties a buyer of that kind of product chooses between first (often its size, capacity or dimensions, its type, its material or its colour, whichever that kind of product is bought by) and keys carried by most of the item's listings; avoid keys seen on only one or two of them.
Skip a property the request already settles for that item, a second name for a property already chosen, a key that lists purchase variants (a mix of sizes, colours and pack counts), identifiers, warranty and other logistics. With several items, put every item's hinted properties first, then take turns between the items.
"label" is the property's plain name: for a listed key, all of the key's words in plain spelling ("lid colour" for lidColour, "rim finish" for rim_finish; never shortened to "colour" or "finish"); only the property's name, no example values, no brackets, no second property.
Choose up to COUNT, best first. Return ONE JSON object and nothing else: {"ask": [{"item": item number, "label": "...", "hint": "at most 8 of the story's words it comes from, or an empty string"}]}"""


HEAR = """A shopping agent asked a shopper some questions. Read each answer and return ONE JSON object and nothing else:
{"needs": [{"item": item number, "need": "property: constraint", "points": true when the value is to be read from another listing, else false}], "drop": [{"item": item number, "property": "a requirement the shopper now gives up"}], "free": [{"item": item number, "property": "..."}], "crowded": [numbers of the questions whose answer complains that the message asked about too many things or wants a single question; not answers that only say they did not follow it], "budget": new total limit as a number or null}
- "needs": every firm preference an answer states, with the exact value or limit and its unit ("at least 30 cm", "oval", "not wicker"). Name the need by the property the answer itself names, which may differ from the one asked about; several answers about one property make one need. Copy the value word for word as the answer says it, keeping its nouns and units ("at least 3 shelves", not "at least 3").
- An answer can give the value by pointing at another listing (a value like, unlike or near the one on some shop's product, that product picked out by a few of its properties): that is ONE need on the property the answer names, with "points": true. Its value is the answer's own words copied exactly, from the comparison words to the end of that description, including what it says about a listing that page may name; never shorten or reword it. The properties describing the other product are not needs of this item.
- "free": the property of each question the shopper said they do not mind about.
- Folksy or roundabout wording that names a value still states that value."""

BUDGET_PICK = """A shopper answered some questions about an order. Find the number in the answers that the shopper gives as the new total budget: the most they will spend on the whole order from now on. The price of one listing, a bundle or a single item is not the budget, and neither is an old budget that the shopper replaces. UNIT names the unit of the order's budget, for reference. Copy the budget number exactly as the answers write it, whether or not a unit is written next to it, with no conversion to another unit. If the answers give no new total budget, use null.
Return ONE JSON object and nothing else: {"budget": the number or null}"""


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

TRAIL = """A shopping agent is about to check out. Some requirements take their value from another listing instead of stating it, and a listing's page can send the reader on to a further listing for that property. The order counts only when the agent has opened in full every listing along each such chain. Follow each requirement's chain from the requirement's own description, link by link, through the pages' own words, using the records below (OPENED: the agent has opened that page; ROW: seen only as a search result row, its page not read).
Return ONE JSON object and nothing else: {"chains": [{"need": the requirement copied exactly, "links": [{"id": the listing id when a record below is that link, else "", "about": how the requirement or the previous page describes that listing}], "done": true when the last link is an OPENED page that states the property's value itself, else false, "value": the property's value as stated on that last page, or ""}]}
List each chain's links in order and stop at the first link whose page is not OPENED or not found, since its words are still unknown."""

GATE = """A shopping agent is about to STAGE. The shopper's request asks for the steps below along the way. Decide from the numbered RECORD of what the agent has done whether each step due before this STAGE is done for the cart as it is now. A step counts as done only when a RECORD line shows it being done and, for a step about the cart or the order, that line comes after the cart last changed and its reply grants it; questions on a product's properties do not do a step.
When a step is still missing, give the one store call that does it next: a tool from TOOLS with its arguments. A message to the shopper goes through the messaging tool: one short sentence or question about that step alone, in plain words, naming no product properties or values the step does not need. Give no call when every due step is done or when the messages or steps left cannot do the missing one.
Return ONE JSON object and nothing else: {"steps": [{"step": the step, "done": true or false, "lines": [the RECORD line numbers that show it done]}], "call": {"tool": "...", "args": {...}} or null, "hold": what a reply asks to change before going on, or "", "needs": [{"item": item number, "need": "property: constraint"} for each firm requirement such a reply states]}"""

SWAP = """A listing in a shopping agent's cart has just sold out. Find replacements for that cart line among the other listing records shown, with these reading rules: """ + READING + """
Return ONE JSON object and nothing else: {"fits": [{"id": listing id, "price": its price as a number as the record shows it} for every listing that appears to meet every requirement of the sold-out line's item, at most 6], "reason": "one sentence"}"""

SHAPE = """A store tool returned the result below. Name its parts by the exact keys the result and the tool's parameters use. Return ONE JSON object and nothing else: {"rows": the key of the list of listings, "count": the key holding how many listings matched in all (not how many are shown), or null, "next": the key holding the value that asks for the following page, or null, "next_arg": the tool parameter that takes that value, or null, "price": the key, inside a row, of the listing's price, or null, "stock": the key, inside a row, that says whether the listing can be bought now, or null}"""


def _dump(value: Any) -> str:
    """Serialise any value to compact JSON text for prompts, traces and comparison keys.
    Separators carry no spaces, and non-ASCII characters are written as they are.
    When encoding fails with a TypeError or ValueError it falls back to str(value).
    Returns a string and does not raise for the values the agent handles.
    Called all over the class: prompt building, showing tool results, the cart fingerprint,
    the arguments of model tool calls and the size count used when trimming the chat."""
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        return str(value)


def _clip(value: Any, limit: int, depth: int = 0) -> Any:
    """Return a copy of a nested value in which every string is at most `limit` characters long.
    A string over the limit keeps its first `limit` characters and gets three dots appended.
    Dicts and lists are rebuilt recursively; numbers, booleans and None come back unchanged.
    Past seven levels of nesting the walk stops and that whole subtree becomes three dots.
    Used before results go into prompts: the opening note is clipped at 1500 characters in
    read_request, and shown tool results use SHOW_CHARS per string."""
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
    """Pull one JSON object out of a model reply's text.
    Non-string input gives None. The text is trimmed, and one Markdown code fence wrapping the
    whole text (with or without a language tag) is removed before parsing.
    First the full text is parsed; when that fails, the span from the first opening brace to
    the last closing brace is tried instead.
    Only a dict counts: a JSON list, a bare number or text that will not parse returns None.
    Used by ask_json for every JSON prompt, and by the main loop, run_calls and join_opens to
    decode tool-call arguments that arrive as strings."""
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
    """Normalise an attribute key or a label into lower-case words joined by single spaces.
    camelCase is split where a lower-case letter or digit meets a capital; runs of whitespace,
    underscores, dots and hyphens then count as one separator.
    Non-string input goes through str() first, and the result is trimmed.
    This lets seatHeight, seat_height and seat-height compare equal. Used to match lookup musts
    in read_request, to group keys in key_table and item_groups, and in interview and note code."""
    key = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(key))
    return re.sub(r"[\s_.\-]+", " ", key).strip().lower()


def _stem(text: str) -> frozenset[str]:
    """Turn a text into a frozen set of its normalised words (through _words) with a crude plural trim.
    A word longer than three letters that ends in s loses that last s; shorter words stay as is.
    Word order and repeats are lost, so the result works as a bag-of-words key.
    evidence compares a requirement's property with attribute keys this way, and key_table and
    item_groups use it to match category labels against the requested items."""
    return frozenset(w[:-1] if len(w) > 3 and w.endswith("s") else w for w in _words(text).split())


def _names(prop: frozenset[str], key: frozenset[str]) -> bool:
    """Decide whether an attribute key refers to the property a requirement is about.
    Both arguments are word sets made by _stem: the property first, then the key.
    True when the sets are equal, when every property word is in the key, or when the key has
    two or more words and all of them are in the property.
    A one-word key that is only part of a longer property does not match, so a generic key
    cannot claim a more specific property. An empty set on either side gives False.
    Only caller: evidence, which lists the matching keys after listings are opened."""
    if not prop or not key:
        return False
    return prop == key or prop <= key or (len(key) >= 2 and key <= prop)


def _kinds(schema: dict[str, Any]) -> list[str]:
    """Read the type field of one JSON-schema node as a list of type names.
    A single string type becomes a one-element list; a list of types keeps only its string
    entries; a missing or empty type gives an empty list.
    Expects a dict; callers pass parameter schemas taken from the store's tool definitions.
    Used by _shape and by the methods that look for free-text or id arguments on store tools
    (single_target, resolve_ids, compare_target, the cart and message argument builders)."""
    kind = schema.get("type")
    if isinstance(kind, str):
        return [kind]
    return [k for k in kind or [] if isinstance(k, str)]


def _fits_kind(value: Any, kind: str) -> bool:
    """Check a value against one JSON-schema type name and return a bool.
    integer and number both reject booleans (Python counts bool as int); number takes int or float.
    string, boolean, array, object and null map to str, bool, list, dict and None.
    Any other type name, including one the code does not know, is accepted.
    Helper of _shape only, which calls it before and after trying to coerce a value."""
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
    """Coerce a tool-call argument toward its JSON schema and collect the faults it cannot fix.
    Inputs: the value, its schema (a non-dict one leaves it alone), a path label for messages
    such as args.ids[2], and a list that receives fault strings. Returns the repaired value.
    Type repair, first fitting type wins: whole float to int, numeric text to a number, number to
    text, true/false text to bool, a lone non-null value wrapped for an array; still no fit gives
    one fault and the value back unchecked. Enums take a case-insensitive spelling match, else a
    fault lists the allowed values; text over maxLength is cut silently. Faults also cover length,
    range and item-count limits, missing required keys and unknown keys under additionalProperties
    false; it recurses into array items and known keys. run_calls bounces any faulted call unsent."""
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
    """Normalise a brief field that ought to be a list.
    A list comes back unchanged; a non-empty string or dict becomes a one-element list.
    Anything else (None, an empty string or dict, numbers, booleans) gives an empty list.
    Lets the code loop over model JSON fields such as items, musts, process, wishes or rules
    even when the model wrote a single value where a list was asked for."""
    if isinstance(value, list):
        return value
    if isinstance(value, (str, dict)) and value:
        return [value]
    return []


def _count(value: Any) -> int | None:
    """Read a whole number from a loosely typed value, or None when it is not one.
    Integers pass through; floats with no fractional part become int; a string of digits with
    no sign (an optional .0 tail and outer spaces allowed) is parsed.
    Booleans, fractional numbers, signed strings, words and other types give None.
    Used for item quantities, message and step limits, counts in cart tool arguments, the
    store's step limit and the error code inside a provider reply.
    Callers read None as not stated and apply their own defaults."""
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
    """Test whether a value could be a listing identifier and return a bool.
    It must be a string whose trimmed form is 3 to 64 characters long and holds no space.
    When `named` is false it must also contain at least one digit; true skips that test,
    for keys the caller already trusts (a key with id in its name, or the learned id key).
    Used by _learn_id to vet candidate keys, by ident and learn_rows to accept values under
    the learned key, and by resolve_ids so that real ids are left untouched."""
    if not isinstance(value, str) or not 3 <= len(value.strip()) <= 64 or " " in value.strip():
        return False
    return named or bool(re.search(r"\d", value))


def _learn_id(rows: list[dict[str, Any]]) -> str | None:
    """Guess which key of a set of result rows holds the listing identifier.
    Candidate keys come from the first row; a key qualifies only when every row's value passes
    _id_like (a key named with id skips the digit test) and, with two or more rows, the trimmed
    values are all different.
    Among qualifying keys a name containing id ranks first, then the longest average value.
    Returns that key, or None when no key qualifies. The caller must pass a non-empty list.
    Called by learn_rows on whole row lists (the winner becomes the agent's id_key) and by ident
    on a single record when the learned key does not fit it."""
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
    """Search a tool result for a list entry that still carries a given id.
    Walks dicts and lists down to depth 5; a hit is a dict inside a list whose string values,
    once trimmed, include the id exactly.
    Scalars and anything deeper than five levels give False.
    Used by run_calls after a cart change: a line given a count of zero or less counts as
    removed only when the cart reply no longer lists it, and a zero count that left the line
    in place adds a hint that this cart tool adds counts rather than setting them."""
    if depth > 5:
        return False
    if isinstance(node, list):
        return any(isinstance(v, dict) and ident in [str(x).strip() for x in v.values() if isinstance(x, str)] for v in node) or any(_in_rows(v, ident, depth + 1) for v in node)
    if isinstance(node, dict):
        return any(_in_rows(v, ident, depth + 1) for v in node.values())
    return False


def _fact_block(node: Any) -> dict[str, Any] | None:
    """Find the attribute block of a listing record and return it as a flat dict.
    Looks only at the record's direct values and considers nested dicts with two or more keys.
    From each it keeps the scalar entries (strings and numbers, booleans excluded) and returns
    the candidate with the most scalars, as long as it holds at least two.
    Returns None for non-dict input or when no nested dict qualifies.
    Tells a full listing page from a search row: absorb copies a record into self.full when it
    has one, and run_calls, fill_facts and widen set facts_from from it. evidence and key_table
    read the attribute keys through it."""
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
    """Plain record for one product the shopper wants, created by read_request from the brief.
    Holds the product name, its category label in the store (may be empty), the quantity and
    the firm requirements the request states (musts).
    Also collects requirements learned later from the shopper (told), requirements the shopper
    or a store note gave up (dropped), and properties the shopper does not care about (free).
    The live requirements are musts plus told minus dropped, as needs_now and ledger use them."""
    def __init__(self, name: str, shelf: str, qty: int, musts: list[str]):
        """Store the name, category label, quantity and initial musts exactly as passed.
        Starts told, dropped and free as empty lists; interviews and store notes fill them later.
        Nothing is validated here: read_request has already defaulted the quantity to 1 and the
        name to a generic product phrase when the brief left them out.
        Returns nothing."""
        self.name = name
        self.shelf = shelf
        self.qty = qty
        self.musts = musts
        self.told: list[str] = []
        self.dropped: list[str] = []
        self.free: list[str] = []


class Kyo:
    """Drives one shopping episode; agent_main builds it from the problem data and calls run, whose
    result is the event trace. Holds the store binding, the request text, tools and their schemas.
    Counters: max_steps and step, model calls (calls, loop_calls), spend (cost), retry wait
    (waited); broke and no_credit halt the work after payment errors.
    Task state from the brief: items, budget, roles (the tool for each job), message limit and
    use, process steps with their message reserve, and lookup musts (refs, ref_values).
    Listing memory: rows (newest record per id), full (records with attributes), opened pages,
    gone and unavailable ids, learned id/price/stock keys. Cart state: cart_lines, cart_view,
    built, verdicts and gate_ok marks. chat is the light model's history; trace logs events."""
    def __init__(self, problem: dict[str, Any]):
        """Read the episode from problem['environment']: the binding sent with every store call and
        the policy view (request text, tool list, step limit, finished flag, opening note).
        Keeps only tools that carry a function name, and stores each one's parameter schema, using
        an empty object schema when the parameters are missing or not a dict.
        max_steps comes from the view (a fractional value is truncated); a missing or non-positive
        value falls back to 30. A view already marked finished makes run return at once.
        All other state starts empty or zero: cost and call counters, message counts, step markers
        (-1 for never), listing stores, notes, the requirements version and the learned keys.
        Raises KeyError when environment, binding or policy_view is missing (agent_main catches it)."""
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
        self.end_mark = ""
        self.end_reply = False
        self.replan = ""
        self.replanned = False
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
        self.dry_searches = 0
        self.wider_held = False
        self.paged_obs: list[Any] = []

    def left(self) -> int:
        """Return how many store steps remain: max_steps minus the steps used so far.
        step follows the store's own counter, kept in sync by act.
        max_steps may have been lowered by read_request when the request sets a smaller limit.
        Read by the main loop, the status hints, the gates and the self-run helpers to decide
        whether one more store call still fits."""
        return self.max_steps - self.step

    def msgs_free(self) -> int:
        """Count the shopper messages the interviewer may still spend.
        Equals the message limit minus messages used, minus the part of the reserve the process
        gate has not yet used (reserve minus gate_msgs, never below zero).
        The reserve is set in read_request for steps due at checkout or at a no-purchase ending.
        The result can go negative; callers read zero or less as no room.
        Used by ready_to_ask, interview and work, which holds back a commit call to ask first."""
        return self.msg_limit - self.msg_used - max(0, self.reserve - self.gate_msgs)

    def cart_mark(self) -> str:
        """Build a stable text fingerprint of the cart as the agent tracks it in cart_lines.
        Sorts the (listing id, quantity) pairs and dumps them as compact JSON.
        Equal carts give equal marks whatever order the lines were added in.
        gate_due and gate join it with the requirements version to remember which cart a process
        step was cleared for; cheaper_rows uses that pair to run once per cart; send writes it into
        the done log so the gate can see which cart a shopper message was about."""
        return _dump(sorted(self.cart_lines.items()))

    def page_tools(self) -> set[str]:
        """Return the set of tools whose result counts as opening a listing's own page.
        The single-listing pick is the offer view when the brief named one, else the open view;
        the side-by-side compare tool joins it when the store has one. Missing roles are skipped.
        act records the ids passed to such a tool as opened (unless the result is an error),
        which the lookup-chain check depends on.
        ledger uses it to tell the model which tool shows a listing's page."""
        page = self.roles.get("offer") or self.roles.get("open")
        return {t for t in (page, self.roles.get("compare")) if t}

    def facts_tool(self) -> str | None:
        """Pick the single-listing tool to use for reading a listing's attributes.
        First choice: a tool already seen returning an attribute block (facts_from true) whose
        schema has one plain text target argument (single_target finds one).
        Otherwise the brief's open role, unless that tool was seen returning no attributes or its
        schema has no such single target.
        Returns the tool name, or None when neither applies.
        Used by fill_facts, open_rows, widen_singly, ledger and compare_bare."""
        known = next((t for t, ok in self.facts_from.items() if ok and self.single_target(t)), None)
        if known:
            return known
        tool = self.roles.get("open")
        return tool if self.facts_from.get(tool or "") is not False and self.single_target(tool) else None

    def compare_facts(self) -> bool | None:
        """Report what is known about the side-by-side tool showing attributes.
        Returns the facts_from entry for the compare role: True or False once its results were
        seen, None when it has not been used yet.
        With no compare role the lookup uses an empty name and so also gives None.
        Only caller: compare_bare."""
        return self.facts_from.get(self.roles.get("compare") or "")

    def compare_bare(self) -> bool:
        """Decide whether the compare tool should be avoided as a way to read attributes.
        True when it has been seen returning none. Before it is tried, also True when another
        route exists: the attribute tool is known to return attributes, or the store has a
        separate offer view.
        False when it showed attributes, or when it is untried and no other route is known.
        Steers join_opens, open_rows and widen toward single-listing opens instead."""
        seen = self.compare_facts()
        return seen is False or (seen is None and (self.facts_from.get(self.facts_tool() or "") is True or bool(self.roles.get("offer"))))

    def gate_due(self, stage: str) -> bool:
        """Tell whether the request's process steps for one stage still need checking.
        stage is choose, order or end. True when some process step has that stage and the pair
        (stage, cart mark plus requirements version) is not yet in gate_ok.
        gate adds that pair once its check finds nothing missing, so a different cart or a change
        to the requirements opens the gate again.
        Used by gate before doing any work and by tail to keep a step for a gate message."""
        return any(p["when"] == stage for p in self.process) and (stage, self.cart_mark() + f"#{self.version}") not in self.gate_ok

    def tail(self) -> int:
        """Return how many steps to hold back for finishing; normally 1, for the checkout itself.
        Returns 2 when an order-stage process step needs messages, the message limit is not used
        up and the order gate is still due, leaving room for that message before checking out.
        Used by status (the final-steps warning), offered (which tools are shown), run_calls and
        the self-run page readers, so they do not eat the steps the finish needs."""
        talk = any(p["when"] == "order" and p["messages"] > 0 for p in self.process)
        return 2 if talk and self.msg_used < self.msg_limit and self.gate_due("order") else 1

    def think(self, messages: list[dict[str, Any]], models: list[str], max_tokens: int, tools: list[dict[str, Any]] | None = None) -> dict[str, Any] | None:
        """Get one chat reply from the first model in `models` that answers, or None.
        Inputs: chat messages, model names in fallback order, a token cap and optional tool specs.
        Each model tried adds one to calls; once ALL_CALLS (60) is reached it returns None.
        Retryable failures (no status, 402, 408, 425, 429, 5xx) are resent after jittered waits from
        WAITS (4, 10, 20, 35 s, scaled by 0.8 to 1.2); a timeout (no status or 408) only once more.
        Resending stops when the planned waits would pass WAIT_TOTAL (45 s); other errors, an empty
        reply (status 0) included, give up on that model at once and move to the next.
        A 402 still failing after its resends sets no_credit and returns None with no further model.
        Called by ask_json and by the main loop in work."""
        for model in models:
            if self.calls >= ALL_CALLS:
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
                pause = WAITS[tries] * (0.8 + 0.4 * random.random())
                self.waited += WAITS[tries]
                tries += 1
                time.sleep(pause)
            if status == 402:
                self.no_credit = True
                return None
        return None

    def call_model(self, model: str, messages: list[dict[str, Any]], max_tokens: int, tools: list[dict[str, Any]] | None) -> tuple[dict[str, Any] | None, Any]:
        """Send one chat-completions request through the proxy; returns (message or None, status).
        For anthropic models with more than one message, works on a copy and marks the first and
        the last message as ephemeral cache points (string content only).
        Temperature is 0; tools, when given, are attached with automatic tool choice.
        A reply with choices adds usage.cost to self.cost; a message with content or tool calls
        returns status 200, an empty one logs llm_fail and returns status 0.
        On failure the status comes from the client error, or from an error object inside an
        otherwise normal body (its code); the failure and 200 chars of its body go to the trace.
        One attempt only (the client allows a single try); think does the resending."""
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
            status = _count(data["error"].get("code"))
            error = {"body": data["error"].get("message")}
        self.trace.append({"kind": "llm_fail", "model": model, "status": status, "body": str(error.get("body") or "")[:200]})
        return None, status

    def ask_json(self, prompt: str, models: list[str], max_tokens: int = 1800) -> dict[str, Any] | None:
        """Ask a one-shot JSON question and return the parsed object, or None.
        Sends the prompt as a lone user message to each model in turn (one think call per model)
        and decodes the reply with _json_obj.
        The first reply that parses to a dict wins; an unparsable reply falls through to the next
        model in the list.
        Stops early with None after a payment failure (broke or no_credit) or at ALL_CALLS.
        Token cap defaults to 1800. Used for the brief, unmix, interviews, notes, cart checks,
        audits, gates, swap searches, chain tracing and reading result shapes."""
        for model in models:
            message = self.think([{"role": "user", "content": prompt}], [model], max_tokens)
            got = _json_obj((message or {}).get("content"))
            if got is not None:
                return got
            if self.broke or self.no_credit or self.calls >= ALL_CALLS:
                return None
        return None

    def act(self, name: str, args: dict[str, Any]) -> dict[str, Any] | None:
        """Perform one store tool call through the environment proxy and return its reply dict.
        The payload is the binding plus the next step number and the action (name and args).
        On success step follows the store's reported step (or adds one), finished is set when the
        store says so, and ids sent to a page tool are marked opened unless the result is an error.
        Up to 4 attempts. A 409 that mentions finishing or quarantine marks the episode finished;
        a 409 naming the expected step resyncs the counter and retries; a reuse complaint skips
        one step number and retries; any other 409 gives up.
        Statuses 400 to 404 and 422 return None at once (402 also sets broke); other failures such
        as timeouts or 5xx are tried again. Every failure is logged as env_fail in the trace."""
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
        """Turn the request into the brief once, at the start of run, and set up the task state.
        The prompt holds the brief instructions, each tool's name, description and parameters, the
        request and the clipped opening note. The light model answers first (4000 tokens); heavy, then
        light, is asked again when no items, roles or buy tool came back. roles keep real tools only.
        A tool that both buys and declines without distinct fixed arguments gets a small extra prompt.
        Builds one Item per entry (quantity defaults to 1), or a generic Item when none; lookup musts
        are tied to an item's must (loosely, by property name) in refs, then unmix runs. Sets budget
        (positive number or numeric text), msg_limit (stated, else 3 with an ask tool), max_steps
        (a stated 10 or more below the store's) and process steps; reserve = largest order/end need."""
        lines = []
        for tool in self.tools:
            fn = tool["function"]
            lines.append(f"- {fn['name']}: {fn.get('description', '')} | parameters: {_dump(fn.get('parameters') or {})}")
        prompt = BRIEF + "\nTOOLS:\n" + "\n".join(lines) + "\n\nREQUEST:\n" + self.request
        if self.view.get("opening") is not None:
            prompt += "\n\nOPENING NOTE FROM THE STORE:\n" + _dump(_clip(self.view.get("opening"), 1500))
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
            talk = [p["messages"] for p in self.process if p["when"] in ("order", "end") and p["messages"]]
            self.reserve = min(max(talk), self.msg_limit) if talk else 0

    def unmix(self) -> None:
        """Repair musts that were misread from a lookup's description of another listing.
        For each item with both lookup musts (refs) and other musts, the heavy model (light as
        fallback) is asked which of the other musts only describe the pointed-at listing.
        Those musts leave the item and are folded into its first lookup must, listed in
        parentheses after it; refs is rewritten to the new text so the two stay in step.
        Items without a lookup or without other musts are skipped; only exact copies of the
        model's list are moved. Each change is logged as an unmix trace event.
        Called once, from read_request."""
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
        """Render the task state as a bullet list for the light model (via status) and heavy prompts
        (cart check, audit, gate, swap search, note reading). Read-only; each line starts with a dash.
        One line per item: name, store label, quantity and live requirements (musts plus told
        minus dropped), each lookup must followed by the value its chain resolved to, and the
        properties the shopper does not mind.
        Then the budget with unit and scope, the pick rule, an in-stock line unless the brief allows
        unavailable items, wishes, the request's own rules, the quiet-decline order and each
        process step with the moment it is due. Last, which tool gives attributes (which views show
        none, which shows a page) once that is known or the store has a separate offer view."""
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
        """Build the text appended after each tool result and interview: steps used, total and left.
        Warnings by steps left: within tail plus one, a final-steps notice; at 6 or fewer (or few
        for the item count with no checkout yet) a check-out-now notice; at 12 or fewer with no
        checkout a settle-soon notice; half the steps gone with an untouched cart, a cart-it notice.
        Before any checkout it names up to three unresolved lookup musts that are still in force.
        Then review candidates (leads), the last four change notes and the full ledger.
        Read-only. Called by run for the opening message and by work after interviews and replies."""
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
        if self.leads:
            text += "\n[review candidates] " + ", ".join(self.leads)
        if self.notes:
            text += "\n[changes so far] " + " | ".join(self.notes[-4:])
        text += "\n[ledger]\n" + self.ledger()
        return text

    def ident(self, node: Any, in_list: bool = False, depth: int = 0) -> str | None:
        """Return a listing's id from a record dict, or None when it does not look like a listing.
        Records with fewer than three keys are ignored. With a learned id key, a value under that
        key that passes _id_like is returned trimmed.
        Otherwise it guesses only for a record inside a row list, at the top level, or holding
        nested dicts: _learn_id on that one record must find a key whose name holds id, or whose
        value is an id already in rows.
        Used by absorb, evidence, records_in, run_calls, fill_facts and loose_lines."""
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
        """Decide whether a list from a tool result is a list of listing rows, learning the id key.
        Needs dicts with three or more keys making up at least half the list; otherwise False.
        If the current id_key gives a valid id on every such row, True with no change.
        Else _learn_id runs on the rows and, when it finds a key, that key replaces self.id_key.
        Returns False when no key qualifies.
        Called by absorb while walking each result; its answer tells ident the entries are rows."""
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
        """Folds one tool result into the agent's memory of listings; returns None.
        Input is any parsed observation (dict, list or scalar); the inner walk does the traversal.
        A dict that ident() accepts as a listing is merged into self.rows under its id; an id met
        for the first time is also appended to self.seen_order.
        When the record carries a nested attribute block (_fact_block), a copy of the merged row goes
        to self.full; a later partial record for an id already in self.full updates that copy.
        Stock tracking: a key stored as True that now arrives as False adds the id to
        self.unavailable, and the opposite flip removes it again.
        Callers: observe(), run_calls() after each store call, gate() after its own calls, run()."""
        def walk(node: Any, depth: int, in_list: bool = False) -> None:
            """Recursive worker of absorb() that visits one node of the result tree.
            depth counts levels from the root; a node deeper than 6 is ignored.
            in_list says the dict sits in a list that learn_rows() judged to be result rows, which lets
            ident() accept ids it would otherwise reject.
            A recognised listing is merged and not descended into; other dicts recurse into their values.
            A list calls learn_rows() once (this may set self.id_key) and passes the verdict to each item."""
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
        """Gathers every string enum option declared anywhere in the tool parameter schemas.
        Walks each schema in self.schemas through the nested walk helper, at most 6 levels down.
        Options are trimmed and lowercased; enum entries that are not strings are skipped.
        Returns a set of strings, empty when no tool declares an enum.
        Only category_key() uses it, to favour a row key whose values the store lists as options.
        Read-only: no attribute is changed."""
        found: set[str] = set()

        def walk(schema: Any, depth: int) -> None:
            """Recursive visitor over one JSON-schema dict for labels().
            Returns at once past depth 6 or when the node is not a dict.
            Adds each string in the node's enum list to the enclosing found set, trimmed and lowercased.
            Recurses only into dict values, so schemas nested inside lists (anyOf and the like) are missed."""
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
        """Guesses which key of the listing rows holds the category label.
        Only short string values count: 1 to 40 chars with no run of three digits, over all self.rows.
        A key qualifies when at least 80 percent of rows carry it and it has at most 40 distinct values.
        Ranking among qualifiers: largest share of values found in labels(), then fewer distinct
        values, then present on more rows, then longer average value length.
        Returns the winning key, or None when no key qualifies (for example before any rows are known).
        Recomputed on each call; used by key_table, item_groups, cheaper_rows and complete_cart."""
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
        """Builds the attribute overview the question planner sees in interview().
        Each entry is (category, key name, count of listings that have it, up to 4 sample values).
        Only listings with a full record in self.full count, taken in first-seen order and grouped by
        their category_key() value (one shared bucket when no category key is known).
        Keys come from each record's nested attribute block, normalised by _words, counted once each.
        If some groups match a requested item's shelf or name (same words or a shared stem of 4+
        letters) the rest are dropped. Keys with long values, mostly slash-joined values, all-distinct
        long text, or a name extending a kept key with the same per-listing values are skipped.
        Groups go largest first, keys by count, 14 keys per group at most, samples cut to 40 chars."""
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
                if len(values) >= 4 and len(set(values)) == len(values) and mean > 12:
                    continue
                twin = any(name.startswith(other + " ") and slot["per"] == spec["per"] for other, spec in kept)
                if twin:
                    continue
                kept.append((name, slot))
            for name, slot in kept[:14]:
                table.append((label, name, slot["n"], [v[:40] for v in slot["values"][:4]]))
        return table

    def item_groups(self) -> list[list[str]]:
        """Splits every listing in self.seen_order into id lists by its category value.
        Without a category key all listings fall into a single group.
        Keeps the groups whose label matches a requested item (same normalised words as its shelf or
        name, or a shared stem of four or more letters); when none match, every group stays.
        Returns the groups largest first, cut to the number of requested items (at least one).
        Read-only; widen() and widen_singly() use it to spread extra reads across the items.
        It groups the merged rows, so listings never opened in full are included too."""
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
        """Tells whether a tool is called with exactly one free-text argument, such as a listing id.
        Free text means a string-typed property that has no enum.
        The chosen property is the first required one of that kind, else the only such property.
        Returns None for a missing or unknown tool, when no property fits, or when any other
        parameter is required; otherwise returns the property name.
        Used by facts_tool, fill_facts, widen_singly, open_rows and run_calls to build one-id calls."""
        schema = self.schemas.get(tool or "") or {}
        props = schema.get("properties") or {}
        texts = [k for k, v in props.items() if isinstance(v, dict) and "string" in _kinds(v) and "enum" not in v]
        target = next((k for k in schema.get("required") or [] if k in texts), None) or (texts[0] if len(texts) == 1 else None)
        if not tool or not target or any(k != target for k in schema.get("required") or []):
            return None
        return target

    def price_of(self, ident: str) -> float | None:
        """Reads a listing's current price from its merged record in self.rows via self.price_key.
        That key is learned by read_shape() from a search result, so it is None until then.
        A dict-valued price yields its first numeric value; a numeric string is converted to float.
        Booleans never count as prices.
        Returns a float, or None when the key, the record or a usable number is missing.
        Used by price_sorted, cheapest_here, wider_hold, cheaper_rows and widen_singly."""
        value = (self.rows.get(ident) or {}).get(self.price_key) if self.price_key else None
        if isinstance(value, dict):
            value = next((v for v in value.values() if isinstance(v, (int, float)) and not isinstance(v, bool)), None)
        if isinstance(value, str) and re.fullmatch(r"\s*\d+(\.\d+)?\s*", value):
            value = float(value)
        return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None

    def buyable(self, ident: str) -> bool:
        """True when nothing known marks the listing as unavailable.
        First asks sold_out(), which covers ids in self.unavailable or self.gone.
        Then reads self.stock_key in the merged row: False, or a number at or below zero, means no.
        A missing stock key, value or record counts as buyable.
        Filters candidate rows in cheapest_here, cheaper_rows and widen_singly.
        Price and budget are not looked at here; other checks judge those."""
        if self.sold_out(ident):
            return False
        value = (self.rows.get(ident) or {}).get(self.stock_key) if self.stock_key else None
        return not (value is False or (isinstance(value, (int, float)) and not isinstance(value, bool) and value <= 0))

    def records_in(self, observation: Any) -> list[tuple[str, dict[str, Any]]]:
        """Extracts (id, record) pairs for the listings a result contains.
        A dict that ident() accepts as a listing on its own returns just itself.
        Otherwise scans lists held directly by the dict, and lists one level inside a nested dict,
        keeping dict rows that ident() names with the in-list relaxation.
        A top-level list is scanned the same way; any other type yields an empty list.
        Read-only; used by note_view, price_sorted, cheapest_here, widen and run_calls."""
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
        """Remembers which fields a viewing tool returns, so a repeat view can be skipped later.
        For each listing record in the result, its keys are added to self.tool_keys[name].
        self.view_keys[id] keeps (version, keys shown so far): keys pile up while self.version stays
        the same and start over once the version has moved on.
        Called after open, offer and compare calls (run_calls, fill_facts, widen), never for searches.
        Returns None; shown_already() reads what it stores."""
        for ident, record in self.records_in(observation):
            keys = set(record)
            self.tool_keys.setdefault(name, set()).update(keys)
            version, have = self.view_keys.get(ident, (-1, set()))
            self.view_keys[ident] = (self.version, (have if version == self.version else set()) | keys)

    def shown_already(self, name: str, ident: str) -> bool:
        """True when this listing has shown every field the tool is known to return, at the current
        version of the requirements.
        Needs a non-empty self.tool_keys entry for the tool, a matching version in self.view_keys,
        and the tool's keys being a subset of the keys shown.
        An id in self.gone is never treated as shown, so a sold-out line can be looked at again.
        run_calls() uses it to answer a repeated view without spending a step."""
        version, have = self.view_keys.get(ident, (-1, set()))
        keys = self.tool_keys.get(name)
        return bool(keys) and version == self.version and keys <= have and ident not in self.gone

    def read_shape(self, name: str, observation: Any) -> dict[str, Any] | None:
        """Learns, once per tool, how a result holding a list of listing rows is laid out.
        Needs a dict result with a key whose value is a list of two or more dicts; else returns None.
        Caches in self.shapes, set to None before asking so a failed attempt is never retried.
        Sends the SHAPE prompt with the tool's parameter schema and a clipped sample (three rows per
        list, under 4000 chars) to the light model, falling back to the heavy one; 1500 tokens.
        Each named key must exist: rows among the list keys, count and next in the result, next_arg
        among the parameters, price and stock in the first row; a non-integer count is dropped.
        Fills self.price_key and self.stock_key when still unset and adds a trace entry.
        Returns the shape dict or None; reached through paging() from more_pages()."""
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
            """Validator for one field of the model's shape answer.
            Returns the key unchanged when it is a string present in the given dict.
            Returns None for null, a non-string, or a key the model made up.
            Applied against the result, the parameter list or the first row, depending on the field."""
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
        """Reports the paging state of a search result using the shape learned for the tool.
        Returns (total matches, rows on this page, page parameter, next value); the next value is None
        on the final page.
        Returns None when the shape lacks count, next or next_arg, the result is not a dict, the row
        list is empty, the count is not an integer, or the next value is neither str nor int.
        The first sight of a tool can cost a model call inside read_shape().
        Called by more_pages() for the first page and again after every page it fetches."""
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
        """True when the rows of a result already run in ascending price order.
        Needs at least three rows, each with a price that price_of() can read.
        Returns False otherwise, including before any price key is learned.
        more_pages() uses it to skip fetching more pages for a cheapest pick.
        One row without a readable price is enough to make it return False."""
        prices = [self.price_of(i) for i, _ in self.records_in(observation)]
        return len(prices) >= 3 and None not in prices and all(a <= b for a, b in zip(prices, prices[1:]))

    def more_pages(self, name: str, args: dict[str, Any], observation: Any) -> str:
        """After a search, may read the remaining result pages itself, with no model turn in between.
        Only for a brief pick of cheapest, most or least, and only when paging() describes the result.
        Clears self.paged_obs first. Does nothing on the final page, when the agent already asked for
        a later page, when every match is shown, or (cheapest) when rows are already price-sorted.
        If the pages needed exceed 2, would lift self.pages_read past AUTO_PAGES (6), or would leave
        under 12 steps beyond tail(), it returns a hint to search with tighter terms or page manually.
        Else per page: call with the page parameter, observe(), store in self.paged_obs and
        self.read_cache, follow the next value; stops on faults, a failed call, finish or last page.
        Counts self.pages_read, adds a trace entry, returns the page texts or an empty string."""
        self.paged_obs = []
        pick = self.brief.get("pick")
        plan = self.paging(name, observation) if pick in ("cheapest", "most", "least") else None
        if not plan:
            return ""
        count, shown, param, value = plan
        given = args.get(param)
        later = given not in (None, 0, 1) if isinstance(given, int) and not isinstance(given, bool) else bool(str(given or "").strip())
        if value is None or later or count <= shown:
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
        """For a cheapest pick, adds a price-ordered list of buyable rows to a search reply.
        Input is the search result plus any pages more_pages() fetched (self.paged_obs).
        Gathers the distinct listing ids, keeps buyable ones with a known price, sorts them upward
        and shows at most six as id with price.
        Returns an empty string when the pick is not cheapest, no price key is known, or fewer than
        two rows qualify. Called by run_calls() after search calls that are not ending calls.
        Read-only; the ids come from records_in() applied to each observation."""
        if self.brief.get("pick") != "cheapest" or not self.price_key:
            return ""
        found = {i for seen in observations for i, _ in self.records_in(seen)}
        rows = [(self.price_of(i), i) for i in found if self.buyable(i) and self.price_of(i) is not None]
        if len(rows) < 2:
            return ""
        return "\n[Kyo] buyable rows here, cheapest first: " + ", ".join(f"{i} ({p:g})" for p, i in sorted(rows)[:6])

    def wider_hold(self) -> str:
        """One-time hold on a checkout for a cheapest pick, sent back as the reply instead of buying.
        Fires only when unused so far, the pick is cheapest, and at least 14 plus 2 per item steps
        remain; it then sets self.wider_held so it never fires twice.
        The text says no step was spent, that title-based search misses many fitting listings, and
        asks for more searches under other product names before checking out.
        The cart lines with their prices are quoted as the bar a cheaper fit must beat.
        Returns an empty string when it does not fire; called by run_calls() on the buy call."""
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
        """Opens the attribute view for listings seen only as rows, so the checks can judge attributes.
        Uses facts_tool() and single_target(); returns an empty list when either is missing.
        Skips ids not in self.rows, already in self.full, or sold out, and keeps the given order.
        Stops once steps left fall to keep or below, the episode is finished, the tool proved to show
        no attributes (facts_from False), the args fail validation, or a call fails.
        Each reply goes through observe() and note_view(); a single-listing reply updates
        self.facts_from for the tool.
        Returns lines of id plus shown text (1500 chars each) and logs a trace entry if any.
        Callers: run_calls before checkout, cheaper_rows, widen_singly and complete_cart."""
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

    def search_note(self, name: str, observation: Any) -> str:
        """Count searches in a row whose reply held only empty row lists; from SEARCH_DRY on,
        return a note asking for broader words. A note only: no call is blocked or held.
        A search is a call of the brief's search tool, or, when the brief named none, of a
        tool outside the other roles. A reply with listings sets the count back to zero; a
        reply with no row list at all (an error, a short notice) leaves the count as it is.
        run_calls adds the note to the search's own result; else returns an empty string."""
        search = self.roles.get("search")
        others = {tool for job, tool in self.roles.items() if job != "search"}
        if (search and name != search) or (not search and name in others):
            return ""
        if self.records_in(observation):
            self.dry_searches = 0
            return ""
        node = observation if isinstance(observation, dict) else {"rows": observation}
        lists = [v for v in node.values() if isinstance(v, list)]
        lists += [w for v in node.values() if isinstance(v, dict) for w in v.values() if isinstance(w, list)]
        rows = [v for v in lists if all(isinstance(r, dict) for r in v)]
        if not rows or any(rows):
            return ""
        self.dry_searches += 1
        if self.dry_searches < SEARCH_DRY:
            return ""
        self.trace.append({"kind": "search_note", "dry": self.dry_searches})
        return (f"\n[Kyo] {self.dry_searches} searches in a row found no listings. Narrow words (model codes, part numbers,"
                " long phrases, several filters at once) seldom match: search with broader words, such as the product's"
                " kind or its store category in one or two plain words, then open the closest rows to check the exact"
                " requirement.")

    def cheaper_rows(self) -> list[str]:
        """For a cheapest pick, right before the cart check, reads a few cheaper rivals of the cart lines.
        Runs once per cart state and version (the mark is kept in self.cheaper_marks) and needs a
        category key.
        For each cart line with a price and a category value it gathers seen listings of the same
        category that are not in the cart, not fully read, buyable and strictly cheaper.
        The two cheapest per line are taken; the total is capped at CHEAPER_OPENS (3).
        They go to fill_facts() keeping tail() plus 4 steps; returns its text lines or an empty list."""
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
        """Handles the result of a store call the agent made on its own rather than at the model's ask.
        Absorbs the result into the listing memory, then looks for free-text lines via loose_lines().
        New lines go to take_note() (a light model call) which may change requirements or budget.
        A non-empty meaning is pushed onto self.queued, tagged with the tool name, for the next reply.
        Returns that meaning, or an empty string when there was no new note.
        Used by more_pages, fill_facts, widen, trail, reread_offers, complete_cart and finish.
        Results of calls the model asked for are handled inline in run_calls() instead."""
        self.absorb(observation)
        lines = self.loose_lines(observation)
        if not lines:
            return ""
        meaning = self.take_note(lines)
        if meaning:
            self.queued.append(f"(seen on {name}) {meaning}")
        return meaning

    def widen(self) -> None:
        """First step of interview(): reads several listings in one compare call so key counts are fair.
        Needs a compare tool whose only required parameter is an array; otherwise, or when
        compare_bare() says that view lacks attributes, it hands over to widen_singly().
        Does nothing with 12 or fewer steps left.
        Batch size is the array's maxItems (else 4), shared evenly over item_groups(); only ids not
        in self.full are sent, and at least two are needed.
        After the call: observe, note_view, update self.facts_from, trace; when the view proved to
        carry no attributes it falls back to widen_singly().
        If no new full record appeared from 3+ ids, it retries once with the first two."""
        tool = self.roles.get("compare")
        schema = self.schemas.get(tool or "") or {}
        props = schema.get("properties") or {}
        target = next((k for k, v in props.items() if isinstance(v, dict) and "array" in _kinds(v)), None)
        if not tool or not target or any(k != target for k in schema.get("required") or []) or self.compare_bare():
            self.widen_singly()
            return
        if self.left() <= 12:
            return
        cap = self.compare_room()
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
            self.learn_compare(args.get(target), seen)
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
                        self.learn_compare(args.get(target), reply.get("observation"))
                        self.observe(tool, reply.get("observation"))

    def widen_singly(self) -> None:
        """Fallback of widen() when no usable side-by-side view exists: reads listings one at a time.
        Uses facts_tool() and single_target(); returns quietly if either is missing.
        From each item group it takes up to two buyable unread listings, fewer once 7 or more of the
        group are read (8 at most per group); for a cheapest pick the cheapest go first.
        The groups are interleaved and at most five ids go to fill_facts() with 22 steps kept back.
        Returns None; only widen() calls it.
        Unlike widen(), it leaves out listings that are sold out or flagged as out of stock."""
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
        """Decides whether work() should run the interview before the next model turn.
        False when already asked, no ask tool exists, msgs_free() is zero, or 8 or fewer steps remain.
        True once self.full holds at least three listings per item plus one.
        Otherwise True from step max(8, a quarter of max_steps) on, so a slow start still gets asked.
        Pure check run at the top of every work() iteration.
        It changes no state; interview() itself sets self.asked."""
        if self.asked or not self.roles.get("ask") or self.msgs_free() <= 0:
            return False
        if self.left() <= 8:
            return False
        opened = len(self.full)
        if opened >= 3 * len(self.items) + 1:
            return True
        return self.step >= max(8, self.max_steps // 4)

    def message_args(self, text: str) -> dict[str, Any] | None:
        """Builds the arguments of the shopper messaging tool for one piece of text.
        The text goes to the first required free-text string property, else the first such property.
        Each other required parameter takes the first value of its enum; one with no enum aborts.
        The dict is then checked and coerced by _shape(); any fault returns None.
        Returns the argument dict or None; send() calls it for every question.
        A tool with several free-text fields gets the text in only one of them."""
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
        """Runs the scripted round of questions to the shopper; returns a chat summary or an empty string.
        Sets self.asked first and stops when msgs_free() is zero; then calls widen() and key_table().
        The PLAN prompt (heavy model first) gets the request, each item's settled requirements, any
        ask-first topics from the brief and the key table, and asks for one pick beyond the budget.
        Picks are cleaned: item clamped, label cut at and/or or punctuation, at most 5 words, unique.
        All but one message go out as a batch, are read by hear() and stored via take_heard().
        The final message is the next pick, asked plainly; none is sent when no pick is left.
        A crowded flag from hear() brings no re-ask: each question names one property, and
        re-asks of those got 0 firm answers in 33 (races 9-10); a new property sometimes does.
        New needs bump self.version; the summary lists Q and A pairs, needs, refs and queued notes.
        Called from work() when ready_to_ask() holds or the model tries to commit before asking."""
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
        prompt = PLAN.replace("COUNT", str(count + 1)) + "\n\nREQUEST:\n" + self.request + "\n\nITEMS:\n" + "\n".join(settled)
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
            label = re.split(r"\s+(?:and|or)\s+|[/,;(]", str(pick.get("label") or ""))[0].strip().rstrip("?.")
            if not label or len(label.split()) > 5:
                continue
            if all(_words(label) != _words(other) for n, other in ordered if n == number):
                ordered.append((number, label))
        self.trace.append({"kind": "plan", "ask": _clip(plan.get("ask"), 120)})
        sent: list[tuple[int, str, str]] = []
        told: list[str] = []
        head = ordered[: count - 1] if count >= 2 else ordered[:count]
        batch = self.send([(number, label, self.question(number, label)) for number, label in head])
        if not batch:
            return ""
        sent += batch
        heard = self.hear(batch)
        self.take_heard(heard, told)
        if count >= 2 and self.msgs_free() > 0 and len(ordered) > len(head):
            number, label = ordered[len(head)]
            batch = self.send([(number, label, self.question(number, label))])
            if batch:
                sent += batch
                self.take_heard(self.hear(batch, (LIGHT, HEAVY)), told)
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

    def question(self, number: int, label: str, samples: list[str] | None = None) -> str:
        """Formats one question to the shopper about a property of item number (1-based).
        Takes the item's name from self.items and puts the label in front of it.
        With samples, two example values are appended so the meant property is unambiguous.
        Returns the text; interview() also rebuilds it to match a crowded answer to its label.
        When samples are given at least two are expected; callers only pass lists of two."""
        text = f"What {label} would you like for the {self.items[number - 1].name}"
        return text + (f", for example {samples[0]} or {samples[1]}?" if samples else "?")

    def samples(self, number: int, label: str) -> list[str]:
        """Finds two example values for a property, for re-asking a question more clearly.
        Looks for the key_table() entry whose key name equals the label's normalised words.
        Drops empty values, values over 30 chars, and values already inside the item's settled
        requirements text.
        Returns up to two strings, or an empty list when no key matches; rebuilds the table each call.
        Matching is exact on the normalised name, so a reworded label finds no samples."""
        want = _words(label)
        settled = " ".join(self.items[number - 1].musts).lower()
        for _, name, _, values in self.key_table():
            if name == want:
                return [v for v in values if v and v not in settled and len(v) <= 30][:2]
        return []

    def send(self, questions: list[tuple[int, str, str]], floor: int = 6) -> list[tuple[int, str, str]]:
        """Delivers (item number, label, text) questions through the ask tool, one step each.
        Stops early when the episode is finished, steps left are at or below floor, self.msg_limit is
        used up, message_args() fails, or the call fails.
        Each delivered message adds one to self.msg_used; the answer text comes from spoken().
        The exchange is logged in self.done_log and self.done_steps (proof for gate()) and the trace.
        A reply that is a dict with an error key ends the loop after being kept.
        Returns the (number, text, answer) triples actually sent; used by interview() and gate()."""
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

    def take_heard(self, heard: dict[str, Any], told: list[str]) -> None:
        """Applies a parsed answer verdict from hear() or the gate model to the per-item requirements.
        needs: a new need joins the item's told list and the told output (with the item number when
        there are several items); points true also records the pair in self.refs.
        drop: matches property names (text before a colon) against the item's musts and told
        entries; hits go to item.dropped and a dropped line is added to told.
        free: properties the shopper is indifferent to are appended to item.free.
        budget: a positive number replaces self.budget.
        Item numbers are clamped to the valid range and malformed entries are skipped; returns None."""
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
        """Asks a model to turn the shopper's answers into structured needs.
        Builds the HEAR prompt from the numbered item names and the numbered question and answer pairs.
        Models are tried in the given order (heavy, then light by default) with a 1500 token cap.
        Returns the parsed dict, or an empty dict when no model gave valid JSON.
        interview() calls it and passes the result to take_heard().
        For the single follow-up answer the interview passes the light model first.
        A heard budget passes said_budget() first, so it stays in the request's unit (the reader
        is shown no unit); a changed value is logged as a budget_unit trace event."""
        prompt = HEAR + "\n\nITEMS:\n" + "\n".join(f"item {n}: {it.name}" for n, it in enumerate(self.items, 1))
        prompt += "\n\nQUESTIONS AND ANSWERS:\n" + "\n".join(f"{i}. (about item {n}) Q: {q}\nA: {a}" for i, (n, q, a) in enumerate(sent, 1))
        got = self.ask_json(prompt, list(models), 1500) or {}
        kept = self.said_budget(got.get("budget"), [a for _, _, a in sent])
        if kept != got.get("budget"):
            self.trace.append({"kind": "budget_unit", "read": got.get("budget"), "kept": kept})
            got["budget"] = kept
        return got

    def said_budget(self, value: Any, answers: list[str]) -> Any:
        """Keep a budget heard in the shopper's answers in the unit of the request's budget.
        The HEAR reader is shown no unit and may turn "769 cents" into 7.69. A unit mark is
        a sign or a word of the brief's budget_unit (a plural s ignored) written right next to
        the request's own budget figure (the figure equal to the brief's budget); with no such
        word there, the first word of budget_unit (so "usd" of "cents in USD" marks nothing).
        Figures alone cannot tell a new total from an item price written next to it, so no
        figure is chosen by its size. value is returned unchanged unless the answers hold a
        marked figure, value is written nowhere in an answer that holds a marked figure (a
        number the reader copied there is no unit slip), and a marked figure is value times a
        power of ten (10**-3 to 10**3, up or down). Only then pick_budget() asks the light model
        which written figure is the new total budget. Used by hear() before take_heard()."""
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
            return value
        read = self.brief.get("budget")
        if isinstance(read, str) and re.fullmatch(r"\s*\d+(\.\d+)?\s*", read):
            read = float(read)
        unit_words = re.findall(r"[a-z]{3,}", str(self.brief.get("budget_unit") or "").lower())
        words = {w.rstrip("s") for w in unit_words}
        plain = ".,;:!?'\"()[]{}<>-_/\\|&*#%+=~^`"
        signs: set[str] = set()
        names: set[str] = set()
        marked: list[float] = []
        seen: list[float] = []
        beside: list[float] = []
        for answer, text in [(False, self.request)] + [(True, str(a)) for a in answers]:
            if answer and not names:
                names = {w.rstrip("s") for w in unit_words[:1]}
            here: list[float] = []
            flagged = False
            for hit in re.finditer(r"\d[\d,]*(?:\.\d+)?", text):
                figure = float(hit.group().replace(",", ""))
                left, right = text[: hit.start()].rstrip(" "), text[hit.end():].lstrip(" ")
                near = {c for c in (left[-1:], right[:1]) if c and not c.isalnum() and not c.isspace() and c not in plain}
                said = {w.lower().rstrip("s") for w in re.findall(r"[A-Za-z]+$", left)[-1:] + re.findall(r"^[A-Za-z]+", right)[:1]}
                if not answer:
                    if isinstance(read, (int, float)) and not isinstance(read, bool) and abs(figure - read) < 1e-9:
                        signs |= near
                        names |= said & words
                    continue
                here.append(figure)
                if near & signs or said & names:
                    marked.append(figure)
                    flagged = True
            seen += here
            if flagged:
                beside += here
        if not marked or any(abs(figure - value) <= 1e-6 * max(1.0, figure) for figure in beside):
            return value
        if not any(abs(f - value * 10 ** power) <= 1e-6 * max(1.0, f) for f in marked for power in (-3, -2, -1, 1, 2, 3)):
            return value
        unit = str(self.brief.get("budget_unit") or "").strip()
        if signs:
            unit = (unit + " " if unit else "") + "(written " + " or ".join(sorted(signs)) + ")"
        return self.pick_budget(value, [str(a) for a in answers], marked, seen, unit)

    def pick_budget(self, value: Any, answers: list[str], marked: list[float], seen: list[float], unit: str) -> Any:
        """Ask the light model (heavy next, 300 tokens) which number in the shopper's answers is
        the new total budget, and return that figure as written when it has the request's unit.
        The model is asked for the figure as written, unit or not; its reply counts only when it
        is positive, equals one of marked (the answers' figures written with the request's unit;
        float formatting aside) and every figure of that number in seen (all the answers'
        figures) is marked: a bare or other-unit twin ("5 USD" beside "5 cents") makes it
        ambiguous. No converted, rounded, unmarked or outside figure is taken. Otherwise value
        is kept: also with no usable reply, and with no call when cost is already above
        SPEND_SOFT (as other optional model seats).
        Each decision is a budget_pick trace event. Called only by said_budget()."""
        said = None
        asked = self.cost <= SPEND_SOFT
        if asked:
            prompt = BUDGET_PICK + "\n\nUNIT: " + unit + "\n\nANSWERS:\n" + "\n".join(f"{i}. {a}" for i, a in enumerate(answers, 1))
            said = (self.ask_json(prompt, [LIGHT, HEAVY], 300) or {}).get("budget")
        number = None
        if isinstance(said, (int, float)) and not isinstance(said, bool):
            number = float(said)
        elif isinstance(said, str) and re.fullmatch(r"\s*\d[\d,]*(?:\.\d+)?\s*", said):
            number = float(said.replace(",", ""))
        hits = [f for f in marked if number is not None and number > 0 and abs(f - number) <= 1e-6 * max(1.0, f)]
        same = [f for f in seen if number is not None and abs(f - number) <= 1e-6 * max(1.0, f)]
        kept = value
        if hits and len(hits) == len(same):
            kept = int(hits[0]) if hits[0] == int(hits[0]) else hits[0]
        self.trace.append({"kind": "budget_pick", "asked": asked, "read": value, "said": _clip(said, 40), "kept": kept})
        return kept

    def gate(self, stage: str) -> tuple[bool, str]:
        """Guard before a checkout (stage order) or an ending with no purchase (stage end).
        Returns (True, empty) at once when no process step is due, the episode is over, or this cart
        and version already passed (gate_due); otherwise up to GATE_ROUNDS (3) model rounds.
        Each round shows the request, due steps, ledger, cart, numbered done_log, recent notes, budgets
        left and the non-ending tools; light model first, heavy after a nudge.
        Once the gate has acted, a reply stating needs or a hold bumps self.version: (False, text).
        A done step must cite done_log lines (at order, message steps only after the cart last
        changed); unproven steps with no call get one nudge line; no call at all marks gate_ok.
        Messages use send() and count in self.gate_msgs; other tools run via act(), absorbed, logged."""
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
            got = self.ask_json(prompt, [HEAVY, LIGHT] if nudged else [LIGHT, HEAVY], 2000)
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
                return False, ("Before going on, Kyo did the steps the request asks for:\n" + "\n".join(said) + "\n" if said else "") + (
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
        """Pulls the readable reply text out of a message-tool result.
        A plain string is returned as it is.
        Otherwise collects strings of two or more words found within 4 levels, joined by spaces and
        cut to 800 chars; with none, the dumped result cut to 400 chars is returned.
        Used by send() for answers and by run_calls() to spot sold-out lines after a refused checkout.
        Strings are kept in traversal order, so several reply fields read as one paragraph."""
        if isinstance(observation, str):
            return observation
        found: list[str] = []

        def walk(node: Any, depth: int) -> None:
            """Recursive collector for spoken().
            Appends each string of at least two words to the enclosing found list.
            Descends into dict values and list items; anything deeper than 4 levels is ignored.
            Single-word strings, numbers and booleans are left out.
            Returns None; its only effect is on the found list."""
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

    def loose_lines(self, observation: Any) -> list[str]:
        """Pull prose out of a store result that could be a notice from the shopper or the shop.
        Input: one observation. Only a dict that is not itself a listing record gets scanned.
        It keeps top-level strings of five words or more, such strings inside lists made only
        of strings, and strings of six words or more one level down in dicts that are not listings.
        Lines already in self.heard are filtered out, so each note is handled a single time.
        Returns a list of strings; empty for lists, plain text, listing records or prose-free results.
        Called by observe for the agent's own store calls and by run_calls after each driver call."""
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
        """Interpret fresh note lines with the light model (spare model next, 1200 tokens), showing it
        the ledger of items and requirements; every line is added to self.heard before the call.
        Ids the reply lists as gone, when they are known rows, go into self.unavailable; note_sold
        turns true on any such id or a sold_out flag, and a one-line sell-out note joins sold_lines.
        If the reply kind is neither missing nor a change, it stops there and returns an empty string.
        A change may set a new positive non-bool budget, drop a requirement (exact text, else the first
        with the same property word before the colon, else the text as given) or add one to told;
        the item number is clamped into 1..len(items). Any change or summary bumps version, empties
        recent and appends to notes. Returns the summary followed by the changes in parentheses.
        A relaxation (a dropped need, or a budget raised above a known one) read on the store's
        reply to an ending attempt (the actor's decline just sent, a refused checkout at this
        step, or the forced finish's decline) is kept in replan for one hold of the next decline."""
        self.heard.update(lines)
        prompt = NOTE + "\n\nITEMS AND REQUIREMENTS NOW:\n" + self.ledger() + "\n\nTHE NOTE:\n" + "\n".join(lines)
        got = self.ask_json(prompt, [LIGHT, SPARE], 1200) or {}
        gone = [str(g) for g in _items(got.get("gone")) if isinstance(g, (str, int))]
        self.unavailable.update(g for g in gone if g in self.rows)
        self.note_sold = bool(gone) or got.get("sold_out") is True
        if self.note_sold and len(lines) == 1:
            self.sold_lines.update(lines)
        if got.get("kind") not in (None, "change"):
            return ""
        changed = []
        before = self.budget
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
        meaning = (summary + (" (" + "; ".join(changed) + ")" if changed else "")).strip()
        wider = bool(drops) or (isinstance(before, (int, float)) and not isinstance(before, bool) and before > 0 and self.budget > before)
        answered = self.end_reply or bool(self.end_mark and self.recent[-1:] == [self.end_mark]) or (self.refused_at >= 0 and self.step == self.refused_at)
        if wider and answered and not self.replanned:
            self.replan = meaning
            self.trace.append({"kind": "replan_armed", "step": self.step, "note": meaning[:200]})
        if changed or summary:
            self.version += 1
            self.recent = []
            self.notes.append(summary or "; ".join(changed))
        return meaning

    def needs_now(self) -> list[tuple[int, str]]:
        """List the requirements still in force, each paired with its item number counted from one.
        For every item it takes the original musts, then the told additions from the shopper,
        and leaves out any text that is on that item's dropped list.
        Returns a list of (number, requirement) tuples; empty when no item has a live requirement.
        Read-only; used by status, evidence and trail to know what still has to be satisfied."""
        out = []
        for number, item in enumerate(self.items, 1):
            for need in item.musts + item.told:
                if need not in item.dropped:
                    out.append((number, need))
        return out

    def evidence(self, observation: Any) -> str:
        """Build a hint that maps live requirements onto attribute keys of the listings in a result.
        It walks the observation (depth limit 6) and collects ids of listing records that carry a
        nested attribute block, without descending into those records.
        For up to 12 such ids and up to 8 requirements it compares the property part (before the
        first colon) with each stored attribute key by stemmed word sets, keeping up to 3 key=value
        pairs per requirement with values cut to 50 characters; an item tag is added when the
        request has several items. Returns an empty string when no id, no requirement or no match.
        Called by run_calls after each executed call; the text is appended to that call's result."""
        ids: list[str] = []

        def walk(node: Any, depth: int) -> None:
            """Recursive helper of evidence: visits dicts and lists of the observation down to depth 6.
            A dict that is a listing record with an attribute block, met for the first time, adds its
            id to the outer ids list and is not explored further; other dicts and lists are descended.
            Returns nothing; its only effect is appending to the enclosing ids list."""
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

    def cart_text(self) -> str:
        """Describe the cart for prompts and for the driver model.
        When a cart view was fetched at or after the newest cart change (cart_seen_at >= cart_at),
        that view is dumped with strings clipped to 300 characters and the whole cut to 3000.
        Otherwise the view could be stale, so it falls back to the agent's own cart_lines record
        (listing id to units) as built from the cart calls it has made.
        Returns a string; used by check_cart, replacements, gate and many run_calls messages."""
        if self.cart_view is not None and self.cart_seen_at >= self.cart_at:
            return _dump(_clip(self.cart_view, 300))[:3000]
        return "lines by Kyo's record of the cart calls (listing: units): " + _dump(self.cart_lines)

    def cart_ids(self) -> list[str]:
        """Return the listing ids believed to be in the cart right now.
        With a cart view at least as new as the last cart change, it returns every known row id
        whose text appears anywhere in the dumped view, in the order the rows were first stored.
        Otherwise it returns the keys of cart_lines, the agent's own bookkeeping of cart calls.
        Read-only. Edge case: an id that is a substring of other text in the view still counts."""
        if self.cart_view is not None and self.cart_seen_at >= self.cart_at:
            text = _dump(self.cart_view)
            return [i for i in self.rows if i in text]
        return list(self.cart_lines)

    def record_of(self, ident: str) -> str:
        """Give the best stored record of one listing as text for a model prompt.
        Prefers the full record (one with attributes) over the search row, clips each string to
        400 characters and the dump to 3000 characters.
        Returns a fixed placeholder saying no record was seen when the id is unknown.
        Used by check_cart, replacements and the not-run replies in run_calls."""
        record = self.full.get(ident) or self.rows.get(ident)
        return _dump(_clip(record, 400))[:3000] if record else "(no record seen)"

    def others(self, skip: list[str], limit: int = 30) -> str:
        """List other listings seen so far, newest first, as context for the last cart check.
        Inputs: ids to skip (the cart being judged) and a limit, 30 by default.
        For a cheapest pick the order is re-sorted: listings with read attributes first, then each
        group by known price ascending, unknown prices last.
        Skipped ids and sold-out listings are left out; each record is clipped (160 per string,
        900 per line) and the walk stops at the limit. Returns the lines joined by newlines.
        Only check_cart calls it, with a limit of 15, for cheapest, most or least picks."""
        lines = []
        order = list(reversed(self.seen_order))
        if self.brief.get("pick") == "cheapest":
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
        """Make the memo key for one cart check: the sorted ids joined by a bar plus the version.
        Because version rises whenever requirements, prices or stock records change, a cart that
        was judged before a change is judged again afterwards.
        Returns the key string; used with self.checked and self.verdicts."""
        return "|".join(sorted(ids)) + f"#{self.version}"

    def check_cart(self, ids: list[str] | None = None, limit: float = SPEND_SOFT) -> str:
        """Ask the heavy model (light model next, 1500 tokens) whether a cart meets every requirement.
        With ids given it judges a planned cart at the requested quantities; without, the live cart.
        It returns early with an empty string when the key was checked already, cost is above the
        limit (SPEND_SOFT by default; callers that act on the verdict pass SPEND_HARD) or no ids.
        The prompt holds the request, ledger, cart and each record with a sold-out mark, plus up to
        15 other listings if the brief wants an extreme (cheapest, most, least). The key joins
        checked, the verdict goes to verdicts (no reply counts as not passed), the reply to trace.
        On a rejection the ids leave leads; returns up to 5 problems and up to 4 better known,
        buyable listings as an instruction to fix the cart. A pass or no reply returns empty text."""
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

    def trail(self) -> tuple[str, bool]:
        """Before a checkout, resolve requirements whose value lives on a different listing's page.
        Pending ones are live refs without a value; up to 4 rounds run while cost < SPEND_HARD.
        Each round shows up to 60 newest records (opened pages fuller than rows) to the light model
        (heavy next, 1800 tokens), which returns chains of links per requirement. A chain marked done
        whose links were all opened and that gives a value fills ref_values and bumps version.
        Unopened known links are opened with the offer or single view (at most 3 per round, 6 in all)
        when steps allow; links to unknown listings become lost entries.
        Returns (text, hold): the pages opened and values found, plus, when something is lost, under
        2 earlier holds and over 5 steps left, a request to find those pages (trail_blocks rises)."""
        live = {need for _, need in self.needs_now()}
        pending = [need for _, need in self.refs if need in live and need not in self.ref_values]
        tool = self.roles.get("offer") or self.roles.get("open")
        target = self.single_target(tool)
        seen: list[str] = []
        lost: list[str] = []
        while pending and self.trails < 4 and self.cost < SPEND_HARD:
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
        """Name the tool used to look at a listing's current price and stock.
        Returns the offer role when the store has a separate offer view, else the single-listing
        open role, else None when neither role was found.
        Used by reread_offers and by run_calls to stamp offer_seen for a listing."""
        return self.roles.get("offer") or self.roles.get("open")

    def reread_offers(self) -> str:
        """Fetch the latest offer of each cart line again right before a checkout.
        A line counts as stale when its offer was last seen before max(step - 2, refused_at) and it
        is not already marked gone. Nothing runs without a tool, a single target argument, stale
        lines, or enough steps (stale count plus 3).
        Each stale line costs one store step; offer_seen is stamped and the result goes through
        observe. Scalar fields of the stored row that differ from before are reported per line.
        Logs a reread trace entry and bumps version when any line changed. Returns the change lines
        joined by newlines, or an empty string. Callers: run_calls, complete_cart and find_sold."""
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
            after = self.rows.get(ident) or {}
            moved = [k for k, v in after.items() if k in before and before[k] != v and not isinstance(v, (dict, list))]
            if moved:
                lines.append(f"{ident} changed ({', '.join(moved[:4])}): {_dump(_clip(observation, 300))[:900]}")
        self.trace.append({"kind": "reread", "ids": stale, "changed": len(lines)})
        if lines:
            self.version += 1
        return "\n".join(lines)

    def sold_out(self, ident: str) -> bool:
        """Tell whether a listing must be treated as impossible to buy.
        True when its id is in self.unavailable (a store note, a refusal, or a stock flag of its
        record flipping from true to false) or in self.gone (already handled as sold out).
        Pure lookup with no side effects; used across checks, swaps, audits and cart planning."""
        return ident in self.unavailable or ident in self.gone

    def swap_hint(self) -> str:
        """React to cart lines that have just become unavailable and are not yet marked gone.
        Each such line is stamped in self.gone with the current step, removed from leads, and
        version rises, so the hint fires once per line. replacements then suggests substitutes.
        The text tells the driver the line cannot be bought and must not be reopened, then adds:
        with 6 or fewer steps left, a two-call swap to the first pick; for a cheapest pick, the
        picks by price and one search under the first pick's price before swapping; otherwise a
        direct swap to the first pick; with no pick, a plain request to find another fitting one.
        Returns an empty string when nothing new sold out. Called by run_calls at checkout and after
        every executed call."""
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
        """Work out which cart line a refused checkout meant when the refusal names no listing.
        Considers the cart ids not yet marked gone. If one of them is already known to be sold
        out there is nothing to find, so it returns an empty string.
        A lone line is marked unavailable directly (empty return); with several lines it calls
        reread_offers and hands back its change text; an empty cart returns an empty string.
        Called by run_calls after a refused checkout that carried a sell-out note or wording."""
        ids = [i for i in self.cart_ids() if i not in self.gone]
        if any(self.sold_out(i) for i in ids):
            return ""
        if len(ids) == 1:
            self.unavailable.add(ids[0])
            return ""
        return self.reread_offers() if ids else ""

    def replacements(self, lost: list[str]) -> list[str]:
        """Ask the heavy model (light next, 1200 tokens) for substitutes of sold-out cart lines.
        Input: the ids that sold out. Returns nothing when cost is already above SPEND_SOFT.
        Up to 25 newest other records (not lost, gone or unavailable; 200 per string, 1200 each)
        go in the prompt along with the request, ledger, the lost records and the cart text.
        A fit is kept only if it is a known row, not lost, not gone, not in the cart and not repeated;
        its price is used when numeric, else treated as infinite.
        For a cheapest pick the fits are sorted by that price here, otherwise the model's order stays.
        Logs a swap trace entry and returns at most 4 ids. Only swap_hint calls it."""
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
        ranked = [i for _, i in sorted(fits, key=lambda f: f[0])] if self.brief.get("pick") == "cheapest" else [i for _, i in fits]
        self.trace.append({"kind": "swap", "lost": lost, "fits": fits})
        return ranked[:4]

    def audit(self) -> str:
        """Second opinion when the driver wants to end the episode without buying anything.
        Runs once per episode (audited flag) and only while cost is at or under SPEND_SOFT.
        It calls audit_call and audit_fits; if nothing fits, ids the reply asked to open (known rows,
        not opened, buyable) are opened through open_rows and, when that adds full records, the
        audit runs again.
        When listings are named, lead_items maps item keys to them and leads becomes the flat list.
        Returns a hold message naming the per-item fits and the model's reason, or an empty string.
        Called by run_calls for a decline call while more than 5 steps remain."""
        if self.audited or self.cost > SPEND_SOFT:
            return ""
        self.audited = True
        got = self.audit_call()
        named = self.audit_fits(got)
        if not named:
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
        """Send the audit prompt once: request, ledger and up to 30 newest listing records.
        Sold-out listings are skipped; opened records are clipped 200 per string and 1200 in all,
        search rows 120 per string and 500 in all, each tagged so the model knows which is which.
        Uses the heavy model, then the light one, with 1500 tokens, and logs an audit trace entry.
        Returns the parsed reply dict, or None when no records exist or no model answered."""
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
        """Turn an audit reply into item key to listing ids, keeping only usable listings.
        Reads the fits field, which must be a dict; anything else yields an empty dict.
        Each value is normalised to a list, ids must be known rows and not sold out, at most 4 per
        item are kept, and items left with no ids are removed.
        Pure function of the reply and stored rows; used twice by audit."""
        fits = (got or {}).get("fits")
        if not isinstance(fits, dict):
            return {}
        named = {str(k): [str(i) for i in _items(v) if str(i) in self.rows and not self.sold_out(str(i))][:4] for k, v in fits.items()}
        return {k: v for k, v in named.items() if v}

    def open_rows(self, rows: list[str]) -> bool:
        """Read the attributes of listings known only from search rows, for a repeated audit.
        When the store has a side-by-side tool with a lone array argument and compare_bare is false,
        one call carries up to compare_room ids; else the attributes tool opens the first two singly.
        Gives up (False) when neither path exists, fewer than plan length plus 5 steps remain, or
        argument shaping fails. Each call goes through act and observe; a missing reply or the end
        of the episode stops the plan. Logs an audit_open trace entry.
        Returns True only when the number of full records grew. Only audit calls it."""
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
            if name == joiner:
                self.learn_compare(args.get(many), reply.get("observation"))
            self.observe(name, reply.get("observation"))
        self.trace.append({"kind": "audit_open", "rows": rows, "opened": len(self.full) - before})
        return len(self.full) > before

    def cart_calls(self, lines: dict[str, int], drop: list[str]) -> list[dict[str, Any]] | None:
        """Plan cart tool arguments that leave the cart holding lines (id to units) and none of drop.
        If the schema has an array of objects, one call lists every line using the first plain
        string field as id and the first integer field as count (None when no id field exists).
        Otherwise each line gets its own call: the id argument is one seen carrying ids, a required
        string, or the only string; the count is the first integer; the mode is an enum of 2+ values,
        first value taken as add and second as remove. Drops use the remove mode, a zero count on a
        setting cart, or a negative count the minimum allows; unchanged lines are skipped.
        Returns the list of argument dicts, or None when no id argument exists, a drop cannot be
        expressed, or an additive cart would need a quantity change. Used by complete_cart only."""
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
        """Buy on the driver's behalf when it tries to decline although a reviewed listing passed a check.
        Needs confirmed set and buyable, cart and buy roles, and runs once (completed flag).
        Plan: per lead item the first buyable pick at that item's quantity, one per category value;
        cart lines of other categories stay, the rest are dropped, and all items must be covered.
        It reads missing attributes, re-checks the planned cart up to SPEND_HARD and gives up unless
        that passes and the cart calls plus one fit in the steps left. Cart calls are then sent and
        recorded, followed by checkout through the order gate; a refusal gets one more try after
        reread_offers and a fresh passing check. buys and refused_at are updated per attempt.
        Returns a summary of the cart calls and store replies, or an empty string when it did not act."""
        cart, buy = self.roles.get("cart"), self.roles.get("buy")
        if self.completed or not self.confirmed or self.sold_out(self.confirmed) or not cart or not buy:
            return ""
        self.completed = True
        cat_key = self.category_key()

        def kind(ident: str) -> str:
            """Map a listing id to the category value used to keep one pick per kind of product.
            Reads the row's value under the learned category key, stripped and lowercased; a missing
            value gives an empty string.
            When the store has no category key the id itself is returned, so every listing is its own kind."""
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

    def is_buy(self, name: str, args: dict[str, Any]) -> bool:
        """Decide whether a tool call is the checkout that ends the episode with an order.
        The tool name must equal the buy role, and every fixed argument the brief recorded in
        buy_args (when it is a dict) must appear in args with the same value.
        Returns a bool; extra arguments in the call are ignored.
        Used throughout run_calls to route the checkout path and count buys."""
        if name != self.roles.get("buy"):
            return False
        fixed = self.brief.get("buy_args") if isinstance(self.brief.get("buy_args"), dict) else {}
        return all(args.get(k) == v for k, v in fixed.items())

    def is_refuse(self, name: str, args: dict[str, Any]) -> bool:
        """Decide whether a tool call ends the episode without a purchase.
        The tool name must equal the refuse role, and every fixed argument the brief stored in
        refuse_args (when it is a dict) must appear in args with the same value.
        Returns a bool; extra arguments are ignored.
        Used by run_calls to route declines through audit, holds and complete_cart."""
        if name != self.roles.get("refuse"):
            return False
        fixed = self.brief.get("refuse_args") if isinstance(self.brief.get("refuse_args"), dict) else {}
        return all(args.get(k) == v for k, v in fixed.items())

    def offered(self) -> list[dict[str, Any]]:
        """Choose which tool schemas the driver model may see on this turn.
        The message tool is always hidden because the interviewer alone writes to the shopper.
        With steps left at or under tail only the buy and refuse tools stay; within tail plus 3 the
        cart and cart view are added; within item count plus 5 and before any buy, the offer view too.
        Narrowing happens only when an ending tool exists, and an empty result falls back to all.
        Returns a list of tool dicts. Called by work for each model turn and by run_calls per call."""
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
        """Rewrite the free-text arguments of a decline call before it is sent.
        Looks at every string argument without an enum; optional ones that are absent are skipped.
        The text is replaced by a fixed polite sentence saying nothing fits (cut to the schema's
        maxLength, default 200) when it is empty, names a known listing id, or the brief's
        quiet_decline flag is anything but False.
        Returns a new args dict; the input is not modified. Used by run_calls and finish."""
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
        """Turn a listing title typed in place of an id into the id, before a step is spent.
        Applies to the cart, open, offer and compare tools once the id key is known, and only when
        the tool has exactly one free string or array argument that is known to carry ids.
        Each value (or each list element) goes through fix; the first problem found is returned.
        Returns (args, problem): corrected args with an empty problem, or the original args with a
        message the driver sees as a not-run result. Called by run_calls for every call."""
        if name not in {self.roles.get(r) for r in ("cart", "open", "offer", "compare")} - {None} or not self.id_key:
            return args, ""
        props = (self.schemas.get(name) or {}).get("properties") or {}
        free = [k for k, v in props.items() if isinstance(v, dict) and ("string" in _kinds(v) or "array" in _kinds(v)) and "enum" not in v]
        if len(free) != 1 or (name != self.roles.get("cart") and (name, free[0]) not in self.id_args):
            return args, ""

        def fix(value: Any) -> tuple[Any, str]:
            """Resolve one argument value against the stored rows.
            An exact known id is returned stripped; non-strings, id-like strings and texts shorter than
            8 characters pass through untouched.
            Otherwise a case-insensitive prefix match runs on each row's non-id string fields: one hit
            returns its id, several hits return an error listing up to 6, none returns an error that
            points at the id field. Returns (value, problem text)."""
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
        """Tell whether a cart tool sets a line's quantity rather than adding to it.
        False when any argument is an enum of two or more values (a mode switch); otherwise True
        only when some integer argument has a minimum of exactly 0, so 0 means remove the line.
        Returns a bool. Used by cart_calls and note_cart to model what a count does."""
        props = self.schemas.get(name, {}).get("properties") or {}
        if any(isinstance(s, dict) and isinstance(s.get("enum"), list) and len(s["enum"]) >= 2 for s in props.values()):
            return False
        return any(isinstance(s, dict) and "integer" in _kinds(s) and s.get("minimum") == 0 for s in props.values())

    def note_cart(self, name: str, args: dict[str, Any]) -> None:
        """Update the agent's own record of the cart (cart_lines) after a cart call was sent.
        Sets cart_touched and stamps cart_at with the current step, making older cart views stale.
        An enum value past the first option marks the call as a removal. Lines come from string ids,
        string lists, or object lists (first known id, first numeric count, default 1).
        Object lines, or an empty list of objects, replace the whole record with the positive counts.
        Otherwise each line is reduced (removal), set (setting cart) or increased (additive cart),
        and a line at zero or below is deleted. Called by run_calls and complete_cart."""
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
        """Render a store observation as text for the chat with the driver model.
        Strings inside it are clipped to SHOW_CHARS (450) and the dump is cut at OBS_LIMIT (15000)
        characters with a short cut marker at the end.
        Returns the string; used for tool results, gate logs, trail pages and forced endings."""
        text = _dump(_clip(observation, SHOW_CHARS))
        if len(text) > OBS_LIMIT:
            text = text[:OBS_LIMIT] + "...(cut)"
        return text

    def compare_target(self) -> str | None:
        """Find the array argument of the side-by-side compare tool.
        Returns the first argument whose type includes array, but only when no other argument is
        required; otherwise, or when the store has no compare tool, it returns None.
        Used by open_rows, compare_room, join_opens and run_calls."""
        schema = self.schemas.get(self.roles.get("compare") or "") or {}
        props = schema.get("properties") or {}
        target = next((k for k, v in props.items() if isinstance(v, dict) and "array" in _kinds(v)), None)
        if not target or any(k != target for k in schema.get("required") or []):
            return None
        return target

    def learn_compare(self, ids: Any, seen: Any) -> None:
        """Record what one side-by-side call of several listings showed, for compare_room.
        A reply that names none of the sent ids marks that size as too big (compare_bad keeps the
        smallest such size); a reply that names one of them marks the size as working (compare_ok).
        Used by the code's own side-by-side reads; the actor's calls are recorded in run_calls."""
        if not isinstance(ids, list) or len(ids) < 2:
            return
        shown = _dump(seen)
        if any(str(i) in shown for i in ids):
            self.compare_ok = max(self.compare_ok, len(ids))
        else:
            self.compare_bad = min(self.compare_bad or 999, len(ids))

    def compare_room(self) -> int:
        """Decide how many listings to send in one side-by-side call.
        An integer maxItems on the compare argument wins. Once a size has failed (compare_bad),
        it takes the smaller of that size minus one and max(compare_ok, 2), but never under 2.
        With no failure seen it allows max(4, compare_ok).
        Returns an int; used by open_rows, join_opens and the trimming in run_calls."""
        spec = ((self.schemas.get(self.roles.get("compare") or "") or {}).get("properties") or {}).get(self.compare_target() or "") or {}
        if isinstance(spec.get("maxItems"), int) and not (self.compare_bad and self.compare_bad <= spec["maxItems"]):
            return spec["maxItems"]
        if self.compare_bad:
            return max(2, min(self.compare_bad - 1, max(self.compare_ok, 2)))
        return max(4, self.compare_ok)

    def join_opens(self, calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Merge several single-listing opens from one turn into one side-by-side call to save steps.
        Needs distinct open and compare roles, a single target for open, an array target for
        compare, and compare_bare false (the view may show attributes); else no change is made.
        Opens whose only argument is a known, not-gone id are picked, up to compare_room of them.
        With at least 2, the first becomes the compare call and the others become placeholders
        flagged as joined, which run_calls answers without a store call. Logs a joined trace entry.
        Returns the new call list in the original order."""
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
        """Execute one driver turn of tool calls and return (call id, result text) pairs in order.
        After join_opens, a call is answered without a store step when: the episode ended, it is the
        message tool, unknown, outside offered(), badly formed, a title not an id, a sold listing,
        a removal of an absent line, an ending mixed with other calls, a non-ending in the last tail
        steps, a recent repeat, or a read whose cached or fully shown result is still current.
        A checkout with over 2 steps left may be held by wider_hold, a cart view change, swap_hint,
        trail, check_cart or a re-read; a decline by audit and leads, a late hold, or one push.
        A decline after take_note armed replan (a relaxing notice answered an ending attempt) is held
        once to re-plan when more than items + tail steps are left; a decline that goes on to the
        gate keeps its mark in end_mark, so take_note can tell that the next reply answers it.
        Declines then try complete_cart or get quiet text; endings pass the gate. After act it
        learns compare sizes, id args, rows, views and cart state, and appends pages, evidence, notes."""
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
                    held = [i for i in self.cart_ids() if i in self.gone] if self.held < 2 else []
                    if swap or held:
                        self.held += 0 if swap else 1
                        out.append((cid, "Not run yet (no step used). " + (swap or f"{', '.join(held)} in the cart is sold out: replace that line first.") + "\nCart now: " + self.cart_text()))
                        continue
                    chain, hold = self.trail()
                    if hold:
                        out.append((cid, "Not run yet (no step used). " + chain + "\nCart now: " + self.cart_text()))
                        continue
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
            if self.is_refuse(name, args) and self.replan and not self.replanned and self.left() > len(self.items) + self.tail():
                self.replanned = True
                self.trace.append({"kind": "replan_held", "note": self.replan[:200], "left": self.left()})
                out.append((cid, "Not run yet (no step used). The store's answer to an earlier attempt to finish changed the task: " + self.replan
                            + ". The reason to end without a purchase was judged under the old terms, and a listing ruled out only by a dropped requirement or the old budget may fit now. "
                            "Plan again under the terms now in force before ending: re-check the listings already opened or seen against them, search again where the change widens what fits if steps allow, "
                            "then put the best fitting listing in the cart at the requested quantity (respecting the pick rule) and check out. "
                            "End without a purchase only if no listing meets the requirements now in force; then send the ending call again.\nCart now: " + self.cart_text()))
                continue
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
                self.end_mark = mark
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
            plain = {tool for job, tool in self.roles.items() if job != "search"}
            paged = self.more_pages(name, args, observation) if name not in plain and not ending else ""
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
            text = self.show(observation) + paged + trimmed
            if name not in plain and not ending:
                text += self.cheapest_here([observation] + self.paged_obs)
                text += self.search_note(name, observation)
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

    def finish(self, again: bool = False) -> str:
        """Force an end once the driving loop has stopped without one.
        Does nothing when finished, out of credit (broke) or out of steps. The checkout, with
        the brief's buy arguments and after the order gate when 2 or more steps are left, goes
        out when the cart holds lines (by a fresh cart view, else by Kyo's cart-call record),
        or when it was touched with no fresh view and 2 or more steps remain. Whether such a
        cart is empty is left to the store: lines the gate adds are not in Kyo's record.
        With again set, or with no decline role, a refused checkout (an empty-cart notice
        too) is observed, counted and returned as text, for run's extra round of work.
        Otherwise it falls through to a quieted decline after the end gate.
        Returns refusal text for the caller or an empty string; trace gets forced_buy or
        forced_refuse. The forced decline's reply is read with end_reply set (see take_note).
        Called by run (with again, then plain, then at most twice more: y2) and by agent_main
        after a crash."""
        if self.finished or self.broke or self.left() <= 0:
            return ""
        buy, refuse = self.roles.get("buy"), self.roles.get("refuse")
        fresh = self.cart_view is not None and self.cart_seen_at >= self.cart_at
        holding = bool(self.cart_ids()) if fresh else bool(self.cart_lines)
        if buy and (holding or (self.cart_touched and not fresh and self.left() >= 2)):
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
                if reply is not None and (again or not refuse):
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
                    self.end_reply = True
                    self.observe(refuse, reply.get("observation"))
                    self.end_reply = False
                    return "ending without a purchase was refused: " + self.show(reply.get("observation"))
        return ""

    def trim(self) -> None:
        """Keep the chat history under HISTORY_LIMIT (150 thousand characters of message content).
        When the total is over that limit, the oldest half of the tool messages (at least one) that
        are longer than 600 characters are cut to their first 400 characters.
        Each shortened message ends with a list of up to 20 known listing ids quoted in it, so
        earlier listings stay findable. Edits self.chat in place; returns None.
        Called by work before every driver model call."""
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
        """Play one whole episode and return the trace list.
        Returns at once if the store says the episode is already over. Otherwise read_request builds
        the brief and roles, the opening view is absorbed, and the chat starts with the system guide
        plus the request, the opening and the status line.
        work runs with LOOP_CALLS (42) and SPEND_HARD. If no end came, finish(again=True) tries one;
        when the store refuses it and steps and credit remain, the refusal is shown and work gets
        10 more turns with a tiny extra spend allowance, then a plain finish runs if still needed.
        y2: while that finish's ending is refused (its reply already read) and steps remain,
        finish runs again, at most twice more (ending_again), so a refusal never leaves the
        episode without an ending and a store that keeps refusing cannot loop.
        An end entry with steps, rounded cost and model call count closes the trace."""
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
        self.work(LOOP_CALLS, SPEND_HARD)
        if not self.finished:
            turned = self.finish(again=True)
            if turned and not self.finished and not self.broke and not self.no_credit and self.left() > 1:
                self.chat.append({"role": "user", "content": "[Kyo] Kyo tried to end the episode and " + turned + "\nApply what the store says, then finish.\n" + self.status()})
                self.work(self.loop_calls + 10, SPEND_HARD + 0.01)
            if not self.finished:
                turned = self.finish()
                for _ in range(2):
                    if self.finished or self.broke or not turned or self.left() <= 0:
                        break
                    self.trace.append({"kind": "ending_again", "step": self.step, "refused": turned[:200]})
                    turned = self.finish()
        self.trace.append({"kind": "end", "finished": self.finished, "steps": self.step, "cost": round(self.cost, 5), "calls": self.calls})
        return self.trace

    def short_items(self) -> int:
        """Count requested items that surely have no cart line: the item count less the lines in Kyo's
        record of the cart calls (cart_lines) that the store has not marked sold out, never below 0.
        Any doubt gives 0, so the loop is not extended: a store cart view fresher than the last cart
        change (Kyo cannot count lines a view shows without ids, so the store's view is never
        overruled), a gate round that chose the cart tool (the gate's cart calls are not recorded),
        or a touched cart whose record is empty (lines may have reached the store unrecorded). One
        line serves one item, so fewer lines than items proves an item is uncovered; no label is
        matched, so a recorded line per item never counts as short. Read-only; used by spare_room."""
        cart = self.roles.get("cart")
        if any(x.get("kind") == "gate" and isinstance(x.get("call"), dict) and x["call"].get("tool") == cart for x in self.trace):
            return 0
        if self.cart_view is not None and self.cart_seen_at >= self.cart_at:
            return 0
        if self.cart_touched and not self.cart_lines:
            return 0
        lines = [ident for ident in self.cart_lines if not self.sold_out(ident)]
        return max(0, len(self.items) - len(lines))

    def spare_room(self, spend_cap: float, spare: int, last_cost: float, last_calls: int) -> bool:
        """Tell work whether its loop may go on although cost has reached spend_cap.
        last_cost and last_calls are what the previous loop round of this work call spent.
        True only while fewer than SPARE_TURNS rounds ran past the cap in this work call, cost plus
        last_cost stays under SPEND_HARD plus SPEND_MORE, calls plus last_calls (at least 1) leave
        2 x GATE_ROUNDS model calls free for finish's gates, the interview cannot start (it has run, or
        no message is free: its plan and hear calls are dear), some requested item has no live cart
        line (short_items) and more steps are left than those items plus tail(). These are checks
        before a round, so one round can still pass them by its own cost. A cart that covers every
        item ends the loop at the cap as before."""
        if spare >= SPARE_TURNS or self.cost < spend_cap or self.cost + max(0.0, last_cost) >= SPEND_HARD + SPEND_MORE:
            return False
        if self.calls + max(1, last_calls) + 2 * GATE_ROUNDS >= ALL_CALLS:
            return False
        if not self.asked and self.roles.get("ask") and self.msgs_free() > 0:
            return False
        short = self.short_items()
        return short > 0 and self.left() > short + self.tail()

    def work(self, loop_cap: int, spend_cap: float) -> None:
        """Main loop: let the driver model pick store calls until the episode or a budget runs out.
        Stops when finished, broke, out of steps, at loop_cap turns or at spend_cap cost; past the
        cap it goes on while spare_room allows (an item has no cart line, SPEND_MORE at most, judged
        before each round with the previous round's cost and model calls).
        When the interviewer is ready, interview runs first and its summary is appended.
        Each turn trims history, counts loop_calls and asks the light model (spare next, 3000 tokens)
        with offered tools; a failed reply adds to misses and ends the loop past 2, at ALL_CALLS (60)
        or without credit. Calls get stable ids and re-encoded JSON arguments.
        A turn with no calls gets a nudge; three in a row end the loop. A first committing call
        (cart, buy, refuse) with free messages and over 8 steps left is answered as not run and the
        shopper is interviewed first. Otherwise run_calls results go back as tool messages."""
        idle = spare = 0
        start_cost, start_calls = self.cost, self.calls
        while not self.finished and not self.broke and self.left() > 0 and self.loop_calls < loop_cap and (
                self.cost < spend_cap or self.spare_room(spend_cap, spare, self.cost - start_cost, self.calls - start_calls)):
            start_cost, start_calls = self.cost, self.calls
            if self.cost >= spend_cap:
                if not spare:
                    self.trace.append({"kind": "spend_more", "cost": round(self.cost, 5), "left": self.left(), "short": self.short_items()})
                spare += 1
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
    """Entry point the benchmark harness calls with the problem data of one episode.
    Builds the agent from the data; an exception there returns a trace holding only a crash
    entry (type and message, cut to 400 characters).
    Then it runs the episode and returns its trace. An exception during the run is logged as a
    crash, and a forced finish is attempted when the episode is still open; a failure of that
    attempt is logged as a second crash. The trace is returned in every case."""
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
