"""Tamarind: a shopping agent that works one shopper's request through a store's own tools.

Structure. A brief model turns the request and the tool list into items, requirements, a budget, limits and tool
roles. A light model then drives the store turn by turn, while the code keeps the bookkeeping: every listing record
seen (text-written records are read into maps first), the cart lines, the pages opened, the notes the store or the
shopper adds, and the steps, seconds and spend still available.

Questions. One interviewer plans the shopper questions from the attribute keys of opened listings and sends them one
property at a time, rotating across items, inside a fixed time window; each answer is read into requirements.

Checks. Before a checkout the code re-reads the cart's offers, opens pages never viewed, follows chains of pages that
a requirement points at, and asks a stronger model to review the cart. Before an ending without a purchase it follows
open chains and has the records reviewed for a fit that was missed (a review that comes back empty is asked once
more at low reasoning effort). Calls the code builds itself fill required
parameters only where a value is safe, never guess a checkout, ending or cart choice, and put a listing id or a
question into the parameter named for it.
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
SPEND_STOP = 0.085
SPEND_LIGHT = 0.035
DEADLINE = 250.0
CLOSING = 70.0
LAST_CALL = 28.0
QUESTION_TIME = 30.0        # an interview starts only while more time than CLOSING plus this is left
WAITS = (4, 10, 20, 35)
WAIT_TOTAL = 45
SHOW_CHARS = 450
OBS_LIMIT = 15000
HISTORY_LIMIT = 150000
GATE_ROUNDS = 3
AUTO_PAGES = 6
CHEAPER_OPENS = 3

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
- When the listing has no attribute for that property, the wanted value stated under another attribute (for example a list attribute that includes it) or in the title or description counts. A number of another kind does not (a figure that measures one quantity is not a figure of another quantity), nor a bare figure whose text does not say what it measures.
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

"items": a list with one entry per distinct product the shopper wants bought, each {"name": the kind of product in the request's words, "shelf": the store's category label for it when a tool lists labels (copy it exactly), else "", "quantity": units of that one listing (integer, 1 when not stated; a pair or a set that is one listing counts as 1), "musts": a list of firm requirements on that product, each "property: constraint" with the exact value, limit, exclusion or closed list. Name the property with the request's own word for it (a request that gives "<value>" as the "<property>" gives "<property>: <value>"); do not swap it for a neighbouring property. A must whose value is read from another listing is ONE must holding the whole description of that listing, with the relation in the request's own words ("<property>: <relation> the <property> of the <kind> <shop> sells with <feature>", where <relation> is the request's own wording for how the two values relate, copied exactly): words that describe that other listing (its shop, its kind, "with <property> <value> and <property> <value>" after it) stay inside that one must and are never musts of this product}. Never put price, budget, stock or quantity into musts. The story around the request (the person meant to use it, where it will stand, what went wrong before, the occasion behind it) is context, not a must, unless it states a property's value or limit outright; and never turn words that are part of what the product is called into musts: those belong to "name". A product the request names in one phrase joined by "and" (one "I want ..." with one quantity, often the name of one store department) is ONE item; make separate items only when the request gives each product its own quantity or its own requirements.
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
1. Hinted properties: when a sentence of the request points at one of the listed keys without giving its value, that key, for the item the sentence is about.
2. Then the listed keys in the order given (most carried first); avoid keys seen on only one or two of the item's listings.
Skip a property the request already settles for that item, a second name for a property already chosen, a key whose values list several purchase options at once, a key whose value differs on every listing (an identifier or code), and keys about selling terms rather than the product itself. With several items, take turns between the items.
"label" is the property's plain name: for a listed key, all of the key's words in plain spelling ("<word one> <word two>" for wordOneWordTwo or word_one_word_two; never shortened to one of its words); only the property's name, no example values, no brackets, no second property.
Choose COUNT, best first (fewer only when the lists hold fewer usable keys). Return ONE JSON object and nothing else: {"ask": [{"item": item number, "label": "...", "hint": "at most 8 of the story's words it comes from, or an empty string"}]}"""


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
Reading rules: a requirement is met when the listing's attribute for that property states the value (any key spelling, one element of a list is enough, case, singular/plural and spelling variants of the same word count, a value containing all of the required value's words counts, a different word does not count even when it means nearly the same, units converted, limits include the edge, a range must lie wholly inside); a requirement the request ties to a rule the store states is met by the value that rule gives; an attribute whose key names the same property as the requirement decides, and if it says something else the listing fails; a key for a different property, even a related one (a key about one part of the product says nothing about another part), decides nothing; when the listing has no attribute for that property, the wanted value stated under another attribute or in the title or description counts (a number of another kind does not; nor does a bare figure whose text does not say what it measures); a requirement the category label itself states is met by that category; the category label must match the requested item as a whole; a rule of being unlike another listing's value counts as met unless the records show that same value. Each requested item must appear as a separate cart line at the requested quantity, nothing else may be in the cart, nothing shown out of stock, and the total of price times quantity over all lines must be within the budget. The newest record of a listing wins.
Category: when the CATEGORIES section shows a store category whose name carries the item's own words while a cart line's category does not, that line comes from a neighbouring category and is wrong. Pairs: when the request or a requirement says one item must work with, fit or suit another requested item (or a model the shopper names), the two listings' own words must connect them (a code or name that both records carry, or a statement on one record that covers the other); a pair whose records show no such link is wrong.
When the pick rule is cheapest, most or least, also look at the other listings shown: if one of them clearly fits every requirement and beats the cart's choice on that rule, the cart is wrong.
Return ONE JSON object and nothing else: {"ok": true or false, "problems": ["short, concrete problem", ...], "better": [listing ids that should replace a cart line, best first]}. Use ok=false only for a problem the records clearly show or a requirement no record addresses."""


READING = """the listing's attribute for a property decides when its key names that same property (any key spelling, one element of a list is enough, case, plural, spelling variants and unit conversions count, a value containing all of the required value's words counts, a different word does not count even when it means nearly the same, limits include the edge); a requirement the request ties to a rule the store states is met by the value that rule gives; a key for a different property, even a related one (a key about one part of the product says nothing about another part), decides nothing; when no attribute names that property, the wanted value stated under another attribute or in the title or description counts (a bare figure that does not say what it measures counts for nothing); a requirement the category label itself states is met by that category; the category label must match the item; a rule of being unlike another listing's value counts as met unless the records show that same value; out-of-stock listings and listings that break the budget do not fit."""

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
    """Serialize a value as compact JSON text, leaving non-ASCII characters unescaped. If the value cannot be encoded as
    JSON, its plain string conversion is returned instead."""
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        return str(value)


def _clip(value: Any, limit: int, depth: int = 0) -> Any:
    """Return a copy of a nested value in which every string longer than limit characters is cut to that length with an
    ellipsis appended. Dicts and lists are walked recursively, anything nested more than seven levels down is
    replaced by an ellipsis string, and other scalars pass through unchanged."""
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
    """Try to pull one JSON object out of a text reply. A code fence around the whole text is removed and the text is
    parsed; if that fails, the span from the first opening curly bracket to the last closing one is parsed instead.
    Returns the dict, or None for non-text input, unparsable text or JSON that is not an object."""
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
    """Normalize a key or label into lowercase words separated by single spaces. Camel-case and snake-case boundaries
    become spaces, as do dots, hyphens and runs of whitespace."""
    key = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(key))
    return re.sub(r"[\s_.\-]+", " ", key).strip().lower()


def _stem(text: str) -> frozenset[str]:
    """Turn text into a frozen set of its normalized words, dropping a trailing s from any word longer than three
    letters so that simple plurals match their singular."""
    return frozenset(w[:-1] if len(w) > 3 and w.endswith("s") else w for w in _words(text).split())


def _names(prop: frozenset[str], key: frozenset[str]) -> bool:
    """Test whether an attribute key's word set matches a requirement property's word set: true when the sets are equal,
    when the key holds every property word, or when a key of at least two words lies wholly inside the property.
    Empty sets never match."""
    if not prop or not key:
        return False
    return prop == key or prop <= key or (len(key) >= 2 and key <= prop)


def _kinds(schema: dict[str, Any]) -> list[str]:
    """Return the type names a JSON schema declares as a list, accepting either one type string or a list of them; a
    schema without a type yields an empty list."""
    kind = schema.get("type")
    if isinstance(kind, str):
        return [kind]
    return [k for k in kind or [] if isinstance(k, str)]


def _fits_kind(value: Any, kind: str) -> bool:
    """Check whether a Python value matches one JSON schema type name, treating booleans as neither integers nor
    numbers. Type names it does not recognize are accepted."""
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
    """Make a value conform to a JSON schema as far as safe conversions allow, appending every remaining problem to
    faults, prefixed with the where path. It turns whole floats and numeric strings into numbers, numbers into
    strings, true or false text into booleans and a lone value into a one-item list, matches enum text ignoring case,
    truncates over-long strings and recurses into items and properties. Bounds, short strings, item counts, unknown
    keys and missing required keys are only reported."""
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
    """Normalize a field that should be a list: a list is returned unchanged, a non-empty string or dict is wrapped in a
    one-element list, and anything else gives an empty list."""
    if isinstance(value, list):
        return value
    if isinstance(value, (str, dict)) and value:
        return [value]
    return []


def _count(value: Any) -> int | None:
    """Read a whole number from a value: plain integers are returned as they are, integral floats and strings of digits
    (optionally followed by a point and zeros) are converted, and booleans or anything else give None."""
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
    """Judge whether a value could be a listing identifier. A string qualifies when it has 3 to 64 characters and no
    spaces once trimmed and holds a digit, unless named says its key marks it as an id; when named, a whole number
    from 100 upward qualifies too."""
    if named and isinstance(value, int) and not isinstance(value, bool):
        return 100 <= value < 10 ** 18
    if not isinstance(value, str) or not 3 <= len(value.strip()) <= 64 or " " in value.strip():
        return False
    return named or bool(re.search(r"\d", value))


def _id_number_key(props: dict[str, Any]) -> str | None:
    """Return the name of the one whole-number parameter whose words include id, or None when a tool has none or
    several; it lets a tool that takes numeric listing ids be driven by code."""
    keys = [k for k, s in props.items() if isinstance(s, dict) and "integer" in _kinds(s) and "id" in _words(k).split()]
    return keys[0] if len(keys) == 1 else None


LISTING_WORDS = frozenset({"sku", "listing", "listings", "product", "products", "item", "items", "article", "articles", "ref"})
TEXT_WORDS = frozenset({"text", "message", "msg", "body", "content", "question", "query", "prompt"})
NOT_TEXT = frozenset({"type", "kind", "format", "mode", "language", "lang", "code", "id", "ids"})


def _id_rank(key: str) -> int:
    """How surely a parameter name names a listing: 0 a bare id or a listing's id, 1 a listing word, 2 another id, 3 none."""
    words = set(_words(key).split())
    ided = bool(words & {"id", "ids"})
    listed = bool(words & LISTING_WORDS)
    if ided and (listed or len(words) == 1):
        return 0
    return 1 if listed else 2 if ided else 3


def _text_rank(key: str) -> int:
    """0 when a parameter name names a message's text (not its type, format, language or an id), else 1."""
    words = set(_words(key).split())
    return 0 if words & TEXT_WORDS and not words & NOT_TEXT else 1


def _best(keys: list[str], rank: Any) -> str | None:
    """The key of lowest rank, the earliest on ties; None when there is none."""
    return min(keys, key=lambda k: (rank(k), keys.index(k))) if keys else None


def _learn_id(rows: list[dict[str, Any]]) -> str | None:
    """Pick which key of a list of row dicts serves as the listing identifier, or None. Candidates are the first row's
    keys. Each row's value must look like an id: text needs no digit when the key name contains id, and whole numbers
    count only when id is a whole word of the key name (item_id, itemId, ID). With several rows, the values in text
    form must all differ. Keys with text values come first, then keys named with id, then the greater mean length."""
    best = None
    for key in rows[0]:
        values = [row.get(key) for row in rows]
        named, whole = "id" in str(key).lower(), "id" in _words(str(key)).split()
        if not all(_id_like(v, whole if isinstance(v, int) and not isinstance(v, bool) else named) for v in values):
            continue
        text = all(isinstance(v, str) for v in values)
        values = [str(v).strip() for v in values]
        if len(rows) > 1 and len(set(values)) < len(values):
            continue
        score = (text, "id" in str(key).lower(), sum(len(v) for v in values) / len(values))
        if best is None or score > best[0]:
            best = (score, key)
    return best[1] if best else None


def _in_rows(node: Any, ident: str, depth: int = 0) -> bool:
    """Search a nested result, at most five levels down, for a list containing a dict whose text or whole-number values
    include the given id in text form. run_calls uses it to judge whether a line given a count of zero or less really
    left a cart reply."""
    if depth > 5:
        return False
    if isinstance(node, list):
        return any(isinstance(v, dict) and ident in [str(x).strip() for x in v.values() if isinstance(x, (str, int)) and not isinstance(x, bool)] for v in node) or any(_in_rows(v, ident, depth + 1) for v in node)
    if isinstance(node, dict):
        return any(_in_rows(v, ident, depth + 1) for v in node.values())
    return False


def _fact_block(node: Any) -> dict[str, Any] | None:
    """Find the attribute block of a listing record: among the record's nested dict values, the one with the most string
    or numeric entries (two or more), returned with only those scalar entries. None when the node is not a dict or
    holds no such nested dict."""
    if not isinstance(node, dict):
        return None
    best = None
    for value in node.values():
        if isinstance(value, dict) and len(value) >= 2:
            scalars = {k: v for k, v in value.items() if isinstance(v, (str, int, float)) and not isinstance(v, bool)}
            if len(scalars) >= 2 and (best is None or len(scalars) > len(best)):
                best = scalars
    return best


LINE_FIELD = re.compile(r"(?P<key>[A-Za-z_][\w .()/+-]{0,47}?):(?: (?P<value>.*))?", re.S)
PAIR_NAME = re.compile(r"[A-Za-z0-9][\w .()/+-]{0,48}")


def _plain_text(text: str) -> Any:
    """Turn a field value read out of a text record into the type a JSON store would send: None for None or null, a
    boolean for True or False, an int or a float for a plain number without leading zeros, otherwise the stripped
    text."""
    value = text.strip()
    if value in ("True", "true") or value in ("False", "false"):
        return value[0] in "Tt"
    if value in ("null", "None"):
        return None
    number = re.fullmatch(r"-?(0|[1-9][0-9]*)(\.[0-9]+)?", value)
    if number and len(value) <= 18:
        return float(value) if number.group(2) else int(value)
    return value


def _pair_map(text: str, least: int = 2) -> dict[str, str] | None:
    """Read text made of name=value parts joined by semicolons into a dict. A part without an equals sign is glued back
    onto the value before it; the text must open with a named part, names hold at most six words, and fewer than
    `least` named parts give None."""
    out: dict[str, str] = {}
    last = None
    for part in text.split(";"):
        if not part.strip():
            continue
        name, eq, value = part.partition("=")
        if eq and PAIR_NAME.fullmatch(name.strip()) and len(name.split()) <= 6:
            last = name.strip()
            out[last] = value.strip()
        elif last is None:
            return None
        else:
            out[last] = (out[last] + ";" + part).strip()
    return out if len(out) >= least else None


def _line_map(text: str) -> dict[str, Any] | None:
    """Read one record written as key: value pieces joined by spaced bars into a dict, or return None. A piece without
    its own key goes back onto the previous value, at least three keyed pieces and one id-like value are needed, a
    value made of name=value parts becomes a nested dict, and other values go through _plain_text."""
    if " | " not in text:
        return None
    fields: list[list[str]] = []
    for piece in text.split(" | "):
        hit = LINE_FIELD.fullmatch(piece.strip())
        if hit:
            fields.append([hit.group("key").strip(), hit.group("value") or ""])
        elif fields:
            fields[-1][1] += " | " + piece
        else:
            return None
    if len(fields) < 4 or not any(_id_like(value.strip()) for _, value in fields):
        return None
    record: dict[str, Any] = {}
    for key, value in fields:
        pairs = _pair_map(value) if "=" in value else None
        record[key] = pairs if pairs else _plain_text(value)
    return record


def _text_map(text: str, depth: int) -> Any:
    """Decide what a string that stands for a whole record holds: a bar-joined line record, JSON text of a non-empty
    dict or list (decoded and normalised again), or three or more name=value parts with an id-like value. Returns the
    structure, or None when the string is ordinary text."""
    found = _line_map(text)
    if found is not None:
        return found
    stripped = text.strip()
    if len(stripped) >= 2 and stripped[0] in "{[":
        try:
            decoded = json.loads(stripped)
        except ValueError:
            decoded = None
        if isinstance(decoded, (dict, list)) and decoded:
            return _unline(decoded, depth + 1)
    pairs = _pair_map(stripped, 4) if "=" in stripped and " | " not in stripped else None
    if pairs and any(_id_like(v) for v in pairs.values()):
        return {k: _plain_text(v) for k, v in pairs.items()}
    return None


def _unline(node: Any, depth: int = 0, whole: bool = True) -> Any:
    """Normalise a store result so listing records written as text become dicts. Only strings standing for a whole
    record are read: the result itself, list members, and values of dicts with at most two keys; a single-key wrapper
    around a line record is replaced by that record. Anything else is returned unchanged, and recursion stops five
    levels down."""
    if depth > 5:
        return node
    if isinstance(node, str):
        found = _text_map(node, depth) if whole else None
        return node if found is None else found
    if isinstance(node, list):
        return [_unline(v, depth + 1) for v in node]
    if isinstance(node, dict):
        small = len(node) <= 2
        out = {k: _unline(v, depth + 1, small) for k, v in node.items()}
        if depth == 0 and len(node) == 1:
            (key, value), = node.items()
            if isinstance(value, str) and isinstance(out[key], dict) and _line_map(value) is not None:
                return out[key]
        return out
    return node


def _read_reply(data: dict[str, Any]) -> dict[str, Any]:
    """Pass the result inside an environment reply through _unline, in place, and hand the reply back; a reply without a
    result is returned untouched."""
    if "observation" in data:
        data["observation"] = _unline(data["observation"])
    return data


WHOLE_VIEW = ("everything", "complete", "full", "all", "detailed", "details", "detail", "standard", "default")


class Item:
    """One requested product: its name, the store category label it belongs under, the wanted quantity and the firm
    requirements from the request, along with requirements learned later from answers or notes, requirements no
    longer wanted and properties the shopper is indifferent to."""
    def __init__(self, name: str, shelf: str, qty: int, musts: list[str]):
        """Store the product name, category label, quantity and initial requirement list, and start empty lists for
        requirements added later, requirements dropped and properties the shopper does not care about."""
        self.name = name
        self.shelf = shelf
        self.qty = qty
        self.musts = musts
        self.told: list[str] = []
        self.dropped: list[str] = []
        self.free: list[str] = []


class Kyo:
    """Shopping agent for a single episode. It turns the request into items and constraints, lets a light model call the
    store's tools while code keeps track of listings, cart, notes and budget, interviews the shopper with planned
    questions, and adds checks before a checkout or an ending without a purchase."""
    def __init__(self, problem: dict[str, Any]):
        """Set up the episode from the problem payload: copy the environment binding, take the request text and the
        well-formed tool definitions with their parameter schemas, read the step limit (30 when missing or not
        positive), start the clock and initialize the bookkeeping for listings, cart, messages, checks and spending."""
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
        self.given: dict[tuple[str, str], Any] = {}
        self.decline_chain_runs = 0

    def left(self) -> int:
        """Return how many store steps remain: the step limit minus the number of steps the store has counted so far."""
        return self.max_steps - self.step

    def time_left(self) -> float:
        """Return the seconds remaining in the agent's self-imposed time allowance, measured from construction against
        the DEADLINE constant; the value turns negative once that allowance is spent."""
        return DEADLINE - (time.monotonic() - self.t0)

    def checker(self) -> list[str]:
        """Return the model order used by review prompts (cart reviews, the no-purchase review, sold-out replacements
        and splitting lookup descriptions): the stronger model first while the spend is under SPEND_LIGHT, the
        cheaper model first beyond it."""
        return [HEAVY, LIGHT] if self.cost < SPEND_LIGHT else [LIGHT, HEAVY]

    def set_caps(self) -> None:
        """Scale the limits on tool-loop turns and total model calls with the step limit: the turn limit is the step
        limit clamped between LOOP_CALLS and 110, and the call limit is the larger of ALL_CALLS and the turn limit
        plus 25."""
        self.loop_cap = max(LOOP_CALLS, min(110, self.max_steps))
        self.call_cap = max(ALL_CALLS, self.loop_cap + 25)

    def rearm(self) -> None:
        """Reset the once-only guards, namely the no-purchase review flag, the keep-searching prompt flag, the refusal
        block counter, the late ending hold and the wider cheapest search hold, so they can fire again after the task
        changes."""
        self.audited = False
        self.pushed = False
        self.refuse_blocks = 0
        self.late_held = False
        self.wider_held = False

    def msgs_free(self) -> int:
        """Count the shopper messages still available to the interview: the message limit minus messages sent, minus
        whatever part of the reserve for checkout or ending steps the gate has yet to use."""
        return self.msg_limit - self.msg_used - max(0, self.reserve - self.gate_msgs)

    def cart_mark(self) -> str:
        """Return a stable text fingerprint of the cart as Kyo records it, the sorted listing and quantity pairs as
        compact JSON, used to tell whether the cart has changed."""
        return _dump(sorted(self.cart_lines.items()))

    def page_tools(self) -> set[str]:
        """Collect the tools whose successful calls mark listings as opened: the offer tool, or the open tool when no
        offer tool is assigned, plus the compare tool when there is one."""
        page = self.roles.get("offer") or self.roles.get("open")
        return {t for t in (page, self.roles.get("compare")) if t}

    def facts_tool(self) -> str | None:
        """Choose which tool that takes one listing should be used to read attributes, or None. A tool already seen
        returning an attribute block and taking one listing argument is preferred; otherwise the open role is used
        unless it has been seen without attributes or lacks a single listing argument."""
        known = next((t for t, ok in self.facts_from.items() if ok and self.single_target(t)), None)
        if known:
            return known
        tool = self.roles.get("open")
        return tool if self.facts_from.get(tool or "") is not False and self.single_target(tool) else None

    def compare_facts(self) -> bool | None:
        """Report what is known about the compare tool's output: True after it returned attribute blocks, False after it
        returned listing records without any, and None while untested or when no compare tool exists."""
        return self.facts_from.get(self.roles.get("compare") or "")

    def compare_bare(self) -> bool:
        """True when the compare tool should not be relied on for attributes: it has already returned records without
        them, or it is untested while the attribute tool has proven itself or a distinct offer tool exists."""
        seen = self.compare_facts()
        return seen is False or (seen is None and (self.facts_from.get(self.facts_tool() or "") is True or bool(self.roles.get("offer"))))

    def gate_due(self, stage: str) -> bool:
        """True when steps are listed for this stage and no clearance is on file for the present cart fingerprint and
        task version."""
        return any(p["when"] == stage for p in self.process) and (stage, self.cart_mark() + f"#{self.version}") not in self.gate_ok

    def tail(self) -> int:
        """Return 2 when an order-time step of the request involves a shopper message, that stage is still pending and
        messages remain, otherwise 1; this many final steps stay protected for finishing."""
        talk = any(p["when"] == "order" and p["messages"] > 0 for p in self.process)
        return 2 if talk and self.msg_used < self.msg_limit and self.gate_due("order") else 1

    def think(self, messages: list[dict[str, Any]], models: list[str], max_tokens: int, tools: list[dict[str, Any]] | None = None) -> dict[str, Any] | None:
        """Send a chat request to each model in turn and return the first reply message obtained. Each model costs one
        counted call; missing replies and statuses 402, 408, 425, 429 or 500 and above are resent after jittered
        waits (missing replies and 408 only once), bounded by the WAITS schedule, the total wait allowance and the
        clock. Returns None when the call cap, spend stop or time limit is reached, when credit runs out, or when
        every model fails."""
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
                passing = status is None or status in (402, 408, 425, 429) or (isinstance(status, int) and status >= 500)
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

    def call_model(self, model: str, messages: list[dict[str, Any]], max_tokens: int, tools: list[dict[str, Any]] | None, effort: str | None = None) -> tuple[dict[str, Any] | None, Any]:
        """Make one chat completion request through the inference proxy at temperature 0. Tools are passed with
        automatic choice when given; a reasoning effort, when given, goes in the body's reasoning field. For model
        ids starting with anthropic/, the first and last messages are marked for prompt caching. Any reported cost is
        added to the running total. On success it returns the reply message with 200. Otherwise it returns None with
        the error status: 0 when the reply carried neither text nor tool calls, or the provider's code when the
        response embeds an error object."""
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
        if effort:
            body["reasoning"] = {"effort": effort}
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
        """Send a single-message prompt to each model in turn and return the first reply that parses as a JSON object.
        Returns None when no model yields one, stopping early once the store balance or model credit is gone or the
        call cap, time limit or spend stop has been reached."""
        for model in models:
            message = self.think([{"role": "user", "content": prompt}], [model], max_tokens)
            got = _json_obj((message or {}).get("content"))
            if got is not None:
                return got
            if self.broke or self.no_credit or self.calls >= self.call_cap or self.out_of_time or self.cost >= SPEND_STOP:
                return None
        return None

    def act(self, name: str, args: dict[str, Any]) -> dict[str, Any] | None:
        """Perform one store action through the environment endpoint with the next step number, returning the reply data
        or None. The observation is normalised by _unline first; the step count and finished flag are updated and,
        for page-showing tools without an error, the text or whole-number arguments are recorded as opened listings.
        Up to four attempts: step conflicts resync or advance the step, an episode-over conflict marks it finished,
        client errors stop at once, and 402 flags an empty balance."""
        payload = dict(self.binding)
        payload["step"] = self.step + 1
        payload["action"] = {"name": name, "args": args}
        for _ in range(4):
            result = _env.post_verbose("/environment/call", payload)
            if isinstance(result.data, dict):
                data = _read_reply(result.data)
                self.step = data["step"] if isinstance(data.get("step"), int) else self.step + 1
                if data.get("finished"):
                    self.finished = True
                seen = data.get("observation")
                if name in self.page_tools() and not (isinstance(seen, dict) and seen.get("error")):
                    for value in args.values():
                        self.opened.update(str(v).strip() for v in (value if isinstance(value, list) else [value]) if isinstance(v, (str, int)) and not isinstance(v, bool))
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
        """Have a model read the request and tool list into a brief, retrying with the stronger model when the first
        reading lacks items, roles or a checkout tool. From the brief it sets tool roles (keeping the messaging role
        off closing tools), fixed checkout and no-purchase arguments (asked separately when one tool does both),
        items with quantities and requirements, lookup requirements (cleaned by unmix), budget, message and step
        limits, the request's required steps and the messages reserved for them."""
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
        self.keep_closing_roles()
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

    def keep_closing_roles(self) -> None:
        """Make sure the messaging role is not held by the checkout, the ending or the cart tool. When it is, the role
        moves to the single unassigned tool whose parameters are all free text, or is removed when there is not
        exactly one such tool, and the change is logged."""
        ask = self.roles.get("ask")
        closing = {self.roles.get(job) for job in ("buy", "refuse", "cart")} - {None}
        if not ask or ask not in closing:
            return
        taken = set(self.roles.values())
        spare = []
        for name in sorted(self.schemas):
            props = (self.schemas.get(name) or {}).get("properties") or {}
            if name in taken or not props:
                continue
            if all(isinstance(spec, dict) and "string" in _kinds(spec) and "enum" not in spec for spec in props.values()):
                spare.append(name)
        if len(spare) == 1:
            self.roles["ask"] = spare[0]
        else:
            self.roles.pop("ask", None)
        self.trace.append({"kind": "ask_role", "was": ask, "now": self.roles.get("ask")})

    def unmix(self) -> None:
        """For each item holding both lookup requirements and ordinary ones, ask a review model which ordinary
        requirements are really just words identifying the referenced listing rather than demands on the product.
        Those are removed from the item and appended in parentheses to its first lookup requirement, and the lookup
        list is updated to the joined text."""
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
        """Render the current task as dash-prefixed lines: each item with its category label, quantity, live
        requirements (with any value or rule read from a referenced page) and indifferent properties, then budget and
        scope, pick rule, stock rule, wishes, request rules, the quiet-ending wish, required steps by stage and, once
        known, which tools show listing attributes and pages."""
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
        """Build the status block appended to messages for the agent: steps used and left, with sharper warnings as
        steps or time run short, unresolved lookup requirements before any checkout attempt, candidate listings from
        a review, the latest store notes and the full ledger."""
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

    def ident(self, node: Any, in_list: bool = False, depth: int = 0) -> str | None:
        """Return the listing id a dict carries, in text form, or None. Dicts with fewer than three entries never
        qualify; the learned id key is used when it holds an id-like value, and otherwise, for top-level, list-member
        or nesting dicts only, an id guessed from that single record is accepted if its key mentions id or its value
        is already a known listing."""
        if not isinstance(node, dict) or len(node) < 3:
            return None
        if self.id_key and _id_like(node.get(self.id_key), True):
            return str(node[self.id_key]).strip()
        nested = any(isinstance(v, dict) for v in node.values())
        if not (in_list or depth == 0 or nested):
            return None
        found = _learn_id([node])
        if found and ("id" in str(found).lower() or str(node[found]).strip() in self.rows):
            return str(node[found]).strip()
        return None

    def id_of(self, value: Any) -> str | None:
        """Map a call argument, text or whole number, to the known listing it names (the id in text form), or return
        None when it names none or is a boolean or another type."""
        if isinstance(value, bool) or not isinstance(value, (str, int)):
            return None
        key = str(value).strip()
        return key if key in self.rows else None

    def learn_rows(self, node: list[Any]) -> bool:
        """Decide whether a list holds identifiable listing rows, which requires at least half its entries to be dicts
        with three or more keys. Returns True when the known id key works for every such row or a new id key can be
        learned from them, which then replaces the old one, and False otherwise."""
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
        """Fold every listing record in a tool result into the stored rows. The result is walked up to six levels; each
        dict carrying a listing id gets its first-seen position noted and its fields merged into the stored row, a
        boolean field that was true and is now false marks the listing unavailable (the reverse change clears that
        mark), and a dict holding an attribute block refreshes the listing's full record, which later partial records
        keep updated."""
        def walk(node: Any, depth: int, in_list: bool = False) -> None:
            """Recursive helper over one node: an identified dict is stored without descending into it, other dicts have
            their values visited, and lists are first checked as row lists so their members can be identified; nodes
            deeper than six levels are ignored."""
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
        """Gather every string enum option, lowercased and trimmed, from all tool parameter schemas; category_key uses
        this set to recognize category labels."""
        found: set[str] = set()

        def walk(schema: Any, depth: int) -> None:
            """Recursive helper that adds a schema's string enum options to the collected set and descends into its
            nested dict values, up to six levels deep."""
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
        """Guess which row key holds the store category, or None. Only keys with short text values (at most 40
        characters and no run of three digits) that appear on at least 80 percent of stored rows and take at most 40
        distinct values are considered, preferring values found among the tools' enum options, then fewer distinct
        values, wider presence and longer values."""
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
        """Measure the match between a category label and an item as a number from 0 to 3: 3 when the normalized label
        equals the item's shelf label or name, 2 when every stemmed label word appears among the item's shelf and
        name words, 1 when some stemmed word of four or more letters is common to both, otherwise 0, as for an empty
        label."""
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
        """Summarize attribute keys of listings with stored attribute blocks for the question planner, as tuples of
        category, key, number of listings that have it and up to four sample values. Keys are grouped by category
        (only item-matching categories when any match), merged across spellings, and cleared of long,
        slash-separated, every-listing-different or duplicate keys, keeping up to 14 per category by frequency; each
        kept key's distinct value count is saved as well."""
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
        """Combine attribute keys that look like spellings of one property, visiting names from most words to fewest so
        the wordiest spelling survives. A key folds into a kept one when one word set includes the other and their
        letters-and-digits values never clash on common listings while coinciding once or overlapping somewhere, or
        when the names share a word and the values coincide on all of three or more common listings. Folded keys pool
        their per-listing values and recount listings."""
        def flat(text: str) -> str:
            """Remove every character other than lowercase letters and digits, so values that differ only in spacing or
            punctuation compare equal."""
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
        """Group seen listing ids by their category value and return the largest groups, as many as there are items and
        at least one. Categories whose label fits some requested item are preferred; if none fit, all groups compete."""
        cat_key = self.category_key()
        groups: dict[str, list[str]] = {}
        for ident in self.seen_order:
            label = str(self.rows[ident].get(cat_key) or "") if cat_key else ""
            groups.setdefault(label, []).append(ident)
        matched = [ids for label, ids in groups.items() if any(self.label_fit(label, item) for item in self.items)]
        pool = matched or list(groups.values())
        return sorted(pool, key=lambda ids: -len(ids))[: max(1, len(self.items))]

    def single_target(self, tool: str | None) -> str | None:
        """Return the argument through which a tool takes one listing, or None. Among the required plain string
        parameters it takes, in order: one through which the agent already passed known listing ids; else the
        best-named one (a bare id or a listing's id, then a listing word such as sku, listing, product, item, article
        or ref, then any other id), the earlier on ties. With none required, it takes the only plain string one; for
        a tool with no plain string parameter, its single id-named whole-number parameter. Every other required
        parameter must be one can_fill accepts."""
        schema = self.schemas.get(tool or "") or {}
        props = schema.get("properties") or {}
        texts = [k for k, v in props.items() if isinstance(v, dict) and "string" in _kinds(v) and "enum" not in v]
        required = [k for k in schema.get("required") or [] if k in texts]
        target = (next((k for k in required if (tool, k) in self.id_args), None) or _best(required, _id_rank)
                  or (texts[0] if len(texts) == 1 else None))
        if not target and not texts:
            target = _id_number_key(props)
        if not tool or not target or any(k != target and not self.can_fill(tool, k) for k in schema.get("required") or []):
            return None
        return target

    def safe_value(self, name: str | None, key: str, spec: dict[str, Any]) -> Any:
        """Choose a value for a required parameter of a call Kyo builds itself, or None. A schema default wins; on the
        checkout, ending and cart tools only free text gets a short neutral line; elsewhere a choice list takes its
        fullest view option or else the first, a switch is set by the words of its name (preview or dry run off,
        include or show on), page numbers, offsets and page sizes get first-page values within the bounds, money
        amounts get nothing, and free text reuses what the loop model last passed, or a subject line on the message
        tool."""
        kinds = _kinds(spec)
        if name and name == self.roles.get("cart"):
            return None
        if name and name in (self.roles.get("buy"), self.roles.get("refuse")):
            # the action itself (buy or not) is never guessed, not even from a schema default: only a free-text note
            if "string" not in kinds or spec.get("enum"):
                return None
            return "Checkout requested for the cart shown." if name == self.roles.get("buy") else "Ending without an order."
        if spec.get("default") is not None:
            return spec["default"]
        words = set(_words(key).split())
        choices = [o for o in spec.get("enum") or [] if isinstance(o, (str, int, float)) and not isinstance(o, bool)]
        if choices:
            fuller = [o for o in choices if isinstance(o, str) and any(w in WHOLE_VIEW for w in _words(o).split())]
            return (fuller or choices)[0]
        if "boolean" in kinds:
            if words & {"simulate", "rehearse", "dry", "preview", "simulation"}:
                return False
            if words & {"expand", "show", "include", "includes", "with", "full", "all", "detail", "details"}:
                return True
            return None
        if "integer" in kinds or "number" in kinds:
            low, high = spec.get("minimum"), spec.get("maximum")
            bounded = [b for b in (low, high) if isinstance(b, (int, float)) and not isinstance(b, bool)]
            if words & {"cost", "price"}:
                return None
            if words & {"rows", "results", "per", "limit", "size"}:
                size = min(20, int(high)) if high in bounded else 20
                return max(size, int(low)) if low in bounded else size
            if "page" in words:
                return int(low) if low in bounded else 1
            if words & {"skip", "start", "offset"}:
                return int(low) if low in bounded else 0
            return None
        if "string" in kinds:
            return "Question" if name == self.roles.get("ask") else self.given.get((name or "", key))
        return None

    def can_fill(self, name: str | None, key: str) -> bool:
        """Say whether safe_value can supply the named required parameter of a tool."""
        spec = ((self.schemas.get(name or "") or {}).get("properties") or {}).get(key)
        return isinstance(spec, dict) and self.safe_value(name, key, spec) is not None

    def fill_required(self, name: str | None, args: dict[str, Any]) -> dict[str, Any]:
        """Return a copy of a call's arguments with a safe value added for each required parameter the call lacks;
        parameters with no safe value stay missing, so the schema check still stops the call."""
        schema = self.schemas.get(name or "") or {}
        props = schema.get("properties") or {}
        out = dict(args)
        for key in schema.get("required") or []:
            spec = props.get(key)
            if key in out or not isinstance(spec, dict):
                continue
            value = self.safe_value(name, key, spec)
            if value is not None:
                out[key] = value
        return out

    def price_of(self, ident: str) -> float | None:
        """Read a listing's price as a float from the stored row field named by the learned price key, taking the first
        number inside a nested price object and parsing numeric strings; None when no price key is known or no number
        is found."""
        value = (self.rows.get(ident) or {}).get(self.price_key) if self.price_key else None
        if isinstance(value, dict):
            value = next((v for v in value.values() if isinstance(v, (int, float)) and not isinstance(v, bool)), None)
        if isinstance(value, str) and re.fullmatch(r"\s*\d+(\.\d+)?\s*", value):
            value = float(value)
        return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None

    def buyable(self, ident: str) -> bool:
        """True unless the listing is marked sold out or its stored stock field, under the learned stock key, is false
        or a number of zero or less."""
        if self.sold_out(ident):
            return False
        value = (self.rows.get(ident) or {}).get(self.stock_key) if self.stock_key else None
        return not (value is False or (isinstance(value, (int, float)) and not isinstance(value, bool) and value <= 0))

    def records_in(self, observation: Any) -> list[tuple[str, dict[str, Any]]]:
        """List the (id, record) pairs a tool result holds. A result that is itself a listing yields only that record;
        otherwise identified dicts are collected from lists directly under the result and from lists one dict level
        further down, or from the result itself when it is a list."""
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
        """Record, per viewing tool, every field name its results have contained, and per listing, the field names shown
        during the current task version, starting that set afresh when the version has moved on. The shown_already
        check relies on this to skip views that would add nothing."""
        for ident, record in self.records_in(observation):
            keys = set(record)
            self.tool_keys.setdefault(name, set()).update(keys)
            version, have = self.view_keys.get(ident, (-1, set()))
            self.view_keys[ident] = (self.version, (have if version == self.version else set()) | keys)

    def shown_already(self, name: str, ident: str) -> bool:
        """True when every field this tool has ever returned was already shown for the listing during the current task
        version, so another call would reveal nothing new; listings recorded as sold-out cart lines never count as
        fully shown."""
        version, have = self.view_keys.get(ident, (-1, set()))
        keys = self.tool_keys.get(name)
        return bool(keys) and version == self.version and keys <= have and ident not in self.gone

    def read_shape(self, name: str, observation: Any) -> dict[str, Any] | None:
        """Learn the layout of a tool's list results once per tool: where the listings sit, where the overall hit count
        and the following-page token are kept, which parameter takes that token, and which listing fields carry price
        and availability. A model names these from a clipped sample; any name absent from the result, schema or first
        row is discarded, and a failed reading is cached as None. The first price and stock keys found serve all
        listings."""
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
            """Accept a model-named key only when it is a string actually present in the given mapping; otherwise return
            None."""
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
        """Using the tool's learned result layout, return a tuple of the reported match total, the number of rows in
        this result, the parameter used for paging and the token for the following page, which is None when there is
        none. Returns None when the layout lacks a count, next-page key or paging parameter, or when the found values
        have unexpected types."""
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
        """True when the result holds three or more listings, every one with a known price, arranged in non-decreasing
        price order."""
        prices = [self.price_of(i) for i, _ in self.records_in(observation)]
        return len(prices) >= 3 and None not in prices and all(a <= b for a, b in zip(prices, prices[1:]))

    def more_pages(self, name: str, args: dict[str, Any], observation: Any) -> str:
        """Follow up a search result whose reported total exceeds the rows it returned, but only when the pick rule is
        cheapest, most or least. If at most two pages are missing and the auto-page allowance and step reserve
        permit, the code requests them directly and returns their absorbed, cached contents; otherwise it returns a
        hint to tighten the query or continue paging. Final pages, pages the agent already asked for, and cheapest
        results already sorted by price are left alone."""
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
        """When the pick rule is cheapest, return a line naming at most six buyable listings from the given results with
        their prices, in rising price order. Empty when the pick differs, no price key is known, or fewer than two
        priced buyable listings are present."""
        if self.brief.get("pick") != "cheapest" or not self.price_key:
            return ""
        found = {i for seen in observations for i, _ in self.records_in(seen)}
        rows = [(self.price_of(i), i) for i in found if self.buyable(i) and self.price_of(i) is not None]
        if len(rows) < 2:
            return ""
        return "\n[Kyo] buyable rows here, cheapest first: " + ", ".join(f"{i} ({p:g})" for p, i in sorted(rows)[:6])

    def wider_hold(self) -> str:
        """Once until the task changes, when the pick is cheapest and at least 14 plus two per item steps remain, hold
        the checkout with a message urging extra searches under other product names for cheaper fitting listings,
        quoting the cart lines' prices. Returns that message, or an empty string when the hold does not apply."""
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
        """Open, one call each with the attribute tool and its other required parameters filled, the given listings that
        are known as rows but lack a full record and are not sold out, stopping once no more than keep steps remain
        or the tool proves not to show attributes. Returns a short text of each result."""
        tool = self.facts_tool()
        target = self.single_target(tool)
        if not tool or not target:
            return []
        texts = []
        for ident in [i for i in ids if i in self.rows and i not in self.full and not self.sold_out(i)]:
            if self.left() <= keep or self.finished or self.facts_from.get(tool) is False:
                break
            faults: list[str] = []
            args = _shape(self.fill_required(tool, {target: ident}), self.schemas.get(tool), "args", faults)
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
        """With the cheapest pick rule in force, and once per cart state, find buyable listings that share a cart line's
        category, cost less than that line and have no full record yet; the two cheapest per line are gathered and up
        to CHEAPER_OPENS of them opened, so the cart review can compare them with the chosen lines. Returns the texts
        read."""
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
        """Handle the result of a store call issued by the code rather than the agent: store its listing records,
        interpret any new free-text lines as a note, and queue that interpretation for the agent. Returns the note's
        summary or an empty string."""
        self.absorb(observation)
        lines = self.loose_lines(observation)
        if not lines:
            return ""
        meaning = self.take_note(lines)
        if meaning:
            self.queued.append(f"(seen on {name}) {meaning}")
        return meaning

    def widen(self) -> None:
        """Read attributes of more listings before the questions are planned, so key frequencies rest on a broader
        sample. One side-by-side call covers unopened listings shared across the item groups, with its other required
        parameters filled; it falls back to single opens when that tool is missing, unsuitable or shows no
        attributes, and retries with two listings when a larger call added nothing."""
        tool = self.roles.get("compare")
        schema = self.schemas.get(tool or "") or {}
        props = schema.get("properties") or {}
        target = next((k for k, v in props.items() if isinstance(v, dict) and "array" in _kinds(v)), None)
        if not tool or not target or any(k != target and not self.can_fill(tool, k) for k in schema.get("required") or []) or self.compare_bare():
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
        args = _shape(self.fill_required(tool, {target: picks}), schema, "args", faults)
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
                self.widen_singly()
                return
            if len(self.full) == before and len(picks) > 2 and self.left() > 12:
                faults = []
                args = _shape(self.fill_required(tool, {target: picks[:2]}), schema, "args", faults)
                if not faults:
                    reply = self.act(tool, args)
                    if reply is not None:
                        self.observe(tool, reply.get("observation"))
                        if any(str(i) in _dump(reply.get("observation")) for i in picks[:2]):
                            self.compare_ok = max(self.compare_ok, 2)

    def widen_singly(self) -> None:
        """Open several unread, buyable listings individually with the attribute tool, taking listings from the item
        groups in turn, with a cap of four per group for a lone item and two otherwise and no group pushed past eight
        opened, lowest priced first when the pick rule is cheapest, at most eight in all, while 22 steps stay in
        hand."""
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
            lanes.append(pool[: max(0, min(4 if len(self.items) == 1 else 2, 8 - opened))])
        queue = [lane[k] for k in range(4) for lane in lanes if k < len(lane)]
        self.fill_facts(queue[: min(8, max(4, 2 * len(lanes)))], 22)

    def ready_to_ask(self) -> bool:
        """Decide whether to start the interview now. It requires a messaging tool, free messages, no earlier interview
        and more than eight steps left, and then waits until enough listings have full records (three per item plus
        one) or a quarter of the step limit, and at least eight steps, has been used."""
        if self.asked or not self.roles.get("ask") or self.msgs_free() <= 0:
            return False
        if self.left() <= 8:
            return False
        opened = len(self.full)
        if opened >= 3 * len(self.items) + 1:
            return True
        return self.step >= max(8, self.max_steps // 4)

    def message_args(self, text: str) -> dict[str, Any] | None:
        """Build the arguments for sending text with the messaging tool. The text goes in the required plain string
        parameter named for a message's text (text, message, msg, body, content, question, query or prompt, but not a
        type, format, language, code or id), else the first required one, else the first plain string one. Other
        required choice parameters take their first option and any other required parameter comes from fill_required.
        Returns None when a required parameter stays empty or the result breaks the schema."""
        name = self.roles.get("ask")
        schema = self.schemas.get(name or "") or {}
        props = schema.get("properties") or {}
        texts = [k for k, v in props.items() if isinstance(v, dict) and "string" in _kinds(v) and "enum" not in v]
        target = _best([k for k in schema.get("required") or [] if k in texts], _text_rank) or (texts[0] if texts else None)
        if not target:
            return None
        args: dict[str, Any] = {target: text}
        for key in schema.get("required") or []:
            if key not in args:
                spec = props.get(key) or {}
                if isinstance(spec.get("enum"), list) and spec["enum"]:
                    args[key] = spec["enum"][0]
                else:
                    args = self.fill_required(name, args)
                    if key not in args:
                        return None
        faults: list[str] = []
        args = _shape(args, schema, "args", faults)
        return None if faults else args

    def interview(self) -> str:
        """Run the shopper interview once. It starts only while more than CLOSING plus QUESTION_TIME seconds remain;
        after widening the attribute sample a planning model ranks properties per item from the key table, the
        request and any ask-first points, and frequent category keys are appended to each queue. Questions go out
        singly, rotating across items; the first one needs more than CLOSING plus half of QUESTION_TIME seconds left,
        later ones the full margin, and each answer is parsed into requirement changes. Returns a summary, or an
        empty string when nothing was asked."""
        self.asked = True
        count = self.msgs_free()
        if count <= 0 or self.time_left() <= CLOSING + QUESTION_TIME or not self.roles.get("ask"):
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
        want = count + len(self.items) + 2
        prompt = PLAN.replace("COUNT", str(want)) + "\n\nREQUEST:\n" + self.request + "\n\nITEMS:\n" + "\n".join(settled)
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
            label = re.split(r"\s+(?:and|or)\s+|[/,;(]", str(pick.get("label") or ""))[0].strip().rstrip("?.")
            if not label or len(label.split()) > 5:
                continue
            queue = queues.setdefault(number, [])
            if all(_words(label) != _words(other) for other in queue):
                queue.append(label)
            if number not in order:
                order.append(number)
        for number in range(1, len(self.items) + 1):
            queue = queues.setdefault(number, [])
            for name in self.item_keys(number, table, 8):
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
        while self.msgs_free() > 0 and not self.finished and self.time_left() > CLOSING + (QUESTION_TIME if sent else QUESTION_TIME / 2):
            pick = self.next_pick(order, queues, found, last_item)
            if pick is None:
                break
            number, label = pick
            text = self.question(number, label)
            batch = self.send([(number, label, text)])
            if not batch:
                break
            last_item = number
            self.asked_labels.append((number, label))
            if self.msg_refused:
                self.msg_used -= 1
                self.msg_limit = self.msg_used
                break
            sent += batch
            heard = self.hear(batch, (LIGHT, HEAVY))
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

    def next_pick(self, order: list[int], queues: dict[int, list[str]], found: set[int], last_item: int | None) -> tuple[int, str] | None:
        """Choose the next (item number, property) to ask about, or None. Each queue first loses labels holding the
        words of a property an answer already fixed for the item, top-up labels overlapping a property the request
        names, and labels whose words sit inside one asked earlier for that item. Rotation then resumes after the
        last asked item and takes the first with a queued label, favouring items whose answers yielded nothing new."""
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
        """Return up to limit key names from the table categories sharing the item's top label_fit rating (none when no
        category fits), skipping keys seen on fewer than two listings or whose name already appears in the item's
        requirement text. Keys whose values repeat on some listings are listed ahead of keys with a different value
        on every listing."""
        item = self.items[number - 1]
        settled = " ".join(item.musts).lower()
        fits = {label: self.label_fit(label, item) for label, _, _, _ in table}
        best = max(fits.values(), default=0)
        shared: list[str] = []
        single: list[str] = []
        for label, name, carried, values in table:
            near = best > 0 and fits[label] == best
            if near and carried >= 2 and name not in settled and name not in shared + single:
                (shared if self.key_distinct.get((label, name), len(values)) < carried else single).append(name)
        return (shared + single)[:limit]

    def question(self, number: int, label: str) -> str:
        """Phrase a question about a single property, asking the shopper which value they want for the numbered item."""
        return f"What {label} would you like for the {self.items[number - 1].name}?"

    def send(self, questions: list[tuple[int, str, str]], floor: int = 6) -> list[tuple[int, str, str]]:
        """Send each question text through the messaging tool while more than floor steps and some messages remain,
        returning (item, question, answer text) triples. Every send is counted and logged with its step and the cart
        state for the later review of required steps; an error in the reply flags the message as refused and stops
        sending."""
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
        """Apply a parsed reading of answers to the items: each requirement an answer gives joins the item's learned
        list (and the lookup list when they point at another listing), dropped properties retire matching live
        requirements, indifferent properties are noted, and a positive budget replaces the old one. Short
        descriptions of new and dropped requirements are appended to told."""
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
        """Ask a model to read the shopper's answers to the sent questions and return its JSON reading of the
        requirements given, dropped or indifferent properties and any new budget, or an empty dict when no reading is
        obtained."""
        prompt = HEAR + "\n\nITEMS:\n" + "\n".join(f"item {n}: {it.name}" for n, it in enumerate(self.items, 1))
        prompt += "\n\nQUESTIONS AND ANSWERS:\n" + "\n".join(f"{i}. (about item {n}) Q: {q}\nA: {a}" for i, (n, q, a) in enumerate(sent, 1))
        return self.ask_json(prompt, list(models), 1500) or {}

    def gate(self, stage: str) -> tuple[bool, str]:
        """Carry out the steps the request demands at a stage, 'order' before a checkout or 'end' before an ending
        without a purchase. For up to GATE_ROUNDS rounds a model reviews the numbered action record and names one
        store call doing a missing step, which Kyo runs, or none, which clears the stage for this cart; a done claim
        that no record line supports earns one nudge. A reply to those calls that adds requirements or asks for
        changes halts the finish. Returns whether to proceed and a report."""
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
        """Extract the readable reply text from a tool result: a string result as is, otherwise every multi-word string
        found within four levels joined by spaces (up to 800 characters), or the compact JSON of the result (up to
        400) when there is none."""
        if isinstance(observation, str):
            return observation
        found: list[str] = []

        def walk(node: Any, depth: int) -> None:
            """Recursive helper that collects strings of two or more words from nested dicts and lists, stopping past
            depth four."""
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
        """Collect free-text lines from a non-listing result that may be store or shopper notes: top-level strings and
        string-list entries of five or more words, and strings of six or more words inside nested non-listing dicts,
        leaving out lines already interpreted."""
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
        """Have a model interpret unseen note lines against the ledger. Listings it calls gone become unavailable and a
        sell-out is flagged, a one-line sell-out note being kept so repeats are recognized. Unless the note is
        classed as something other than a change, a new budget and added or dropped requirements are applied, and any
        change or summary bumps the version, clears recent calls, logs the note and resets once-only checks. Returns
        a summary of what changed."""
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
        """List every requirement currently in force as (item number, requirement text) pairs, covering the request's
        own and the learned ones minus those dropped."""
        out = []
        for number, item in enumerate(self.items, 1):
            for need in item.musts + item.told:
                if need not in item.dropped:
                    out.append((number, need))
        return out

    def evidence(self, observation: Any) -> str:
        """For listings in a result that carry attribute blocks, build a hint line pairing live requirements' properties
        with stored attribute keys whose names match them, covering up to 12 listings, 8 requirements and 3 keys per
        requirement. Returns an empty string when nothing matches."""
        ids: list[str] = []

        def walk(node: Any, depth: int) -> None:
            """Recursive helper that records the id of each listing dict carrying an attribute block, without looking
            inside it, and searches other dicts and lists down to depth six."""
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
        """Describe the cart for prompts and messages: the most recent cart view, clipped, if no cart call has come
        after it, otherwise the listing to quantity map from Kyo's own record of cart calls."""
        if self.cart_view is not None and self.cart_seen_at >= self.cart_at:
            return _dump(_clip(self.cart_view, 300))[:3000]
        return "lines by Kyo's record of the cart calls (listing: units): " + _dump(self.cart_lines)

    def cart_ids(self) -> list[str]:
        """Return the listing ids in the cart: known listings whose id appears in a current cart view (an all-digit id
        only as a whole number), or else the listings in Kyo's own line record."""
        if self.cart_view is not None and self.cart_seen_at >= self.cart_at:
            text = _dump(self.cart_view)
            return [i for i in self.rows if (re.search(r"(?<![\w.])" + re.escape(i) + r"(?![\w.])", text) if i.isdigit() else i in text)]
        return list(self.cart_lines)

    def record_of(self, ident: str) -> str:
        """Return the best stored record of a listing, the full one if present and otherwise the row, as clipped JSON of
        at most 3000 characters, or a placeholder when nothing is stored."""
        record = self.full.get(ident) or self.rows.get(ident)
        return _dump(_clip(record, 400))[:3000] if record else "(no record seen)"

    def others(self, skip: list[str], limit: int = 30) -> str:
        """List clipped records of other seen listings for a review prompt, newest first, skipping the given ids and
        sold-out listings, up to limit lines. When the pick rule is cheapest, listings with full records are placed
        ahead of bare rows, both sorted by ascending price."""
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
        """Return a cache key for a cart review made of the sorted listing ids and the current task version."""
        return "|".join(sorted(ids)) + f"#{self.version}"

    def check_cart(self, ids: list[str] | None = None, limit: float = SPEND_SOFT) -> str:
        """Ask a review model whether the current cart, or a planned list of ids at the requested quantities, is exactly
        right, supplying the request, ledger, records, category context and, under a pick rule, other listings. Each
        id set is reviewed once per task version and only within the spend limit; the verdict is stored, and a
        failing review removes those ids from the candidates and returns the problems and better listings as a
        message."""
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
        text = "A careful review says this cart is not right yet: " + "; ".join(problems or ["see the requirements"])
        if better:
            text += ". Listings that look better: " + ", ".join(better)
        return text + ". Fix the cart (or show the review is wrong by re-reading the records), then check out."

    def dept_lines(self, ids: list[str]) -> str:
        """Write the category lines for the cart review: one per given listing naming its category, and one per item
        listing up to four observed category labels related to it, closest first, so a near-miss category is easy to
        spot. Empty when no category key is known."""
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

    def trail(self) -> tuple[str, bool]:
        """Ahead of checking out, resolve requirements defined by reference to another listing. A model follows each
        chain of linked pages through the stored records; a chain ending on an opened page stores its value and any
        rule-derived value, while links seen only as rows are opened with the page tool, a few per round, within
        limits on rounds, spend and time. Returns the pages opened and values found, and signals a hold on the
        checkout, at most three times, while some link is still unfound."""
        live = {need for _, need in self.needs_now()}
        pending = [need for _, need in self.refs if need in live and need not in self.ref_values]
        tool = self.roles.get("offer") or self.roles.get("open")
        target = self.single_target(tool)
        seen: list[str] = []
        lost: list[str] = []
        while pending and self.trails < 6 and self.cost < SPEND_HARD and self.time_left() > CLOSING:
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
                args = _shape(self.fill_required(tool, {target: ident}), self.schemas.get(tool), "args", faults)
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
        """Pick the tool for refreshing a listing's current price and stock, preferring the offer role and otherwise
        using the open role."""
        return self.roles.get("offer") or self.roles.get("open")

    def reread_offers(self) -> str:
        """Refresh cart listings whose offer was last read more than two steps ago or before the latest refused
        checkout, one offer-view call each, when enough steps remain; when too few remain for all of them, only the
        lines whose page was never opened are opened through view_unopened. Returns a line for each listing whose
        stored scalar fields changed and bumps the task version if any did."""
        tool = self.offer_tool()
        target = self.single_target(tool)
        if not tool or not target:
            return ""
        since = max(self.step - 2, self.refused_at)
        stale = [i for i in self.cart_ids() if self.offer_seen.get(i, -99) < since and i not in self.gone]
        if not stale or self.left() < len(stale) + 3:
            self.view_unopened(self.tail())
            return ""
        lines = []
        for ident in stale:
            faults: list[str] = []
            args = _shape(self.fill_required(tool, {target: ident}), self.schemas.get(tool), "args", faults)
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

    def view_unopened(self, keep: int) -> list[str]:
        """Open, with the page tool and one step each, every cart line whose page was never opened, as long as more than
        `keep` steps remain; each read is absorbed and its offer step noted. Returns the ids opened."""
        tool = self.offer_tool()
        target = self.single_target(tool)
        if not tool or not target or tool not in self.page_tools():
            return []
        done: list[str] = []
        for ident in self.cart_ids():
            if ident in self.opened or ident in self.gone:
                continue
            if self.finished or self.left() <= keep:
                break
            faults: list[str] = []
            args = _shape(self.fill_required(tool, {target: ident}), self.schemas.get(tool), "args", faults)
            if faults:
                break
            reply = self.act(tool, args)
            if reply is None:
                break
            self.offer_seen[ident] = self.step
            self.observe(tool, reply.get("observation"))
            done.append(ident)
        if done:
            self.trace.append({"kind": "page_check", "lines": done})
        return done

    def sold_out(self, ident: str) -> bool:
        """True when the listing is recorded as unavailable or as a cart line already handled as sold out."""
        return ident in self.unavailable or ident in self.gone

    def swap_hint(self) -> str:
        """When cart lines have newly become sold out, mark them as gone, drop them from the candidates, bump the
        version and ask for replacements among stored listings. Returns guidance naming those replacements, with
        variants for few steps left or a cheapest pick, or general advice when none fit; empty when nothing new sold
        out."""
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
        """Work out which cart line a refused checkout's unnamed sell-out refers to. If only one line not yet marked
        gone remains, it is flagged unavailable; with more lines, their offers are refreshed through reread_offers
        and that report is returned. Nothing happens when a cart line is already known to be sold out."""
        ids = [i for i in self.cart_ids() if i not in self.gone]
        if any(self.sold_out(i) for i in ids):
            return ""
        if len(ids) == 1:
            self.unavailable.add(ids[0])
            return ""
        return self.reread_offers() if ids else ""

    def replacements(self, lost: list[str]) -> list[str]:
        """Ask a review model which stored listings could replace sold-out cart lines, given the requirements, the lost
        lines, the cart and up to 25 other records. Keeps known ids that are not lost, gone or in the cart, sorts
        them by the reported price when the pick rule is cheapest, and returns at most four; returns nothing above
        the soft spend limit."""
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
        ranked = [i for _, i in sorted(fits, key=lambda f: f[0])] if self.brief.get("pick") == "cheapest" else [i for _, i in fits]
        self.trace.append({"kind": "swap", "lost": lost, "fits": fits})
        return ranked[:4]

    def chains_before_decline(self) -> str:
        """Before an ending without a purchase, follow the chains of requirements whose value still has to be read from
        another listing, at most twice in an episode and only with steps and spend to spare. Returns text that holds
        the ending when the chain step opened a page, found a value or still has links to find, else an empty string."""
        live = {need for _, need in self.needs_now()}
        pending = [need for _, need in self.refs if need in live and need not in self.ref_values]
        if not pending or self.decline_chain_runs >= 2 or self.cost >= SPEND_HARD - 0.01 or self.left() <= self.tail() + 2:
            return ""
        self.decline_chain_runs += 1
        before = (len(self.ref_values), len(self.opened))
        text, hold = self.trail()
        moved = (len(self.ref_values), len(self.opened)) != before
        self.trace.append({"kind": "chain_hold", "pending": pending, "moved": moved, "hold": hold})
        if not moved and not hold:
            return ""
        return (text.rstrip("\n") + "\n" if text else "") + ("The ending is on hold: a requirement still waits for a value"
                " from another listing's page. Use the value Kyo read above, or open the pages it names, then choose again;"
                " decline only when no fitting listing exists once that value is known.")

    def audit(self) -> str:
        """Review, once until the task changes and only below the soft spend limit, whether ending without a purchase is
        justified. If the review model finds no fitting listing it may nominate bare rows worth reading; those are
        opened and, if that yields full records, the review runs a second time. Any fits become the candidate lists,
        and a message urging the agent to try them before ending is returned."""
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

    def review_pool(self, limit: int = 30) -> list[str]:
        """Choose up to `limit` live listing ids for the no-purchase review, newest first. With no more than that many
        seen, all of them; otherwise current and fullest-cart lines and review leads first, then opened listings
        taken in turns across their store categories, then the newest search rows until the limit is reached."""
        live = [i for i in reversed(self.seen_order) if not self.sold_out(i)]
        if len(live) <= limit:
            return live
        chosen: list[str] = []
        for ident in self.cart_ids() + self.built + self.leads:
            if ident in live and ident not in chosen and len(chosen) < limit:
                chosen.append(ident)
        cat_key = self.category_key()
        lanes: dict[str, list[str]] = {}
        for ident in live:
            if ident in self.full and ident not in chosen:
                label = str((self.rows.get(ident) or {}).get(cat_key) or "") if cat_key else ""
                lanes.setdefault(label, []).append(ident)
        depth = 0
        while len(chosen) < limit and any(depth < len(lane) for lane in lanes.values()):
            for lane in lanes.values():
                if depth < len(lane) and len(chosen) < limit:
                    chosen.append(lane[depth])
            depth += 1
        for ident in live:
            if len(chosen) >= limit:
                break
            if ident not in chosen:
                chosen.append(ident)
        keep = set(chosen)
        return [i for i in live if i in keep]

    def audit_call(self) -> dict[str, Any] | None:
        """Build the no-purchase review prompt from the listings review_pool chooses, each marked as a full record or a
        bare row, and return the review model's parsed reply. When both checker models leave no usable reply (they
        can spend the whole budget on reasoning), the light model is asked once more with a larger budget and low
        reasoning effort. This needs the call cap, the spend stop and the clock to allow it. Returns None when there
        are no records or still no usable reply."""
        records = []
        for ident in self.review_pool(30):
            record = self.full.get(ident) if ident in self.full else None
            records.append(("OPENED " + _dump(_clip(record, 200))[:1200]) if record else ("ROW " + _dump(_clip(self.rows.get(ident), 120))[:500]))
            if len(records) >= 30:
                break
        if not records:
            return None
        prompt = AUDIT + "\n\nREQUEST:\n" + self.request + "\n\nREQUIREMENTS IN FORCE:\n" + self.ledger()
        prompt += "\n\nLISTING RECORDS SEEN (newest first):\n" + "\n".join(records)
        got = self.ask_json(prompt, self.checker(), 1500)
        if (got is None and not self.broke and not self.no_credit and not self.out_of_time and self.calls < self.call_cap
                and self.cost < SPEND_STOP and self.time_left() > CLOSING + 30):
            # the review models can spend the whole budget on reasoning and send no content: ask once more, briefly
            self.calls += 1
            message, _ = self.call_model(LIGHT, [{"role": "user", "content": prompt}], 4000, None, "low")
            got = _json_obj((message or {}).get("content"))
            self.trace.append({"kind": "audit_retry", "answered": got is not None})
        self.trace.append({"kind": "audit", "got": got})
        return got

    def audit_fits(self, got: dict[str, Any] | None) -> dict[str, list[str]]:
        """Extract from a review reply, per item number, up to four named listings that are known and not sold out,
        dropping item entries left with none."""
        fits = (got or {}).get("fits")
        if not isinstance(fits, dict):
            return {}
        named = {str(k): [str(i) for i in _items(v) if str(i) in self.rows and not self.sold_out(str(i))][:4] for k, v in fits.items()}
        return {k: v for k, v in named.items() if v}

    def open_rows(self, rows: list[str]) -> bool:
        """Fetch full pages for listings so far seen only in search results: one side-by-side call when that tool is not
        ruled out for attributes, otherwise single opens of the leading two, or four when plenty of steps and seconds
        remain, with required parameters filled. It needs five spare steps beyond those calls and returns True only
        when new full records were gained."""
        many, joiner = self.compare_target(), self.roles.get("compare")
        tool = self.facts_tool()
        target = self.single_target(tool)
        if many and joiner and not self.compare_bare():
            plan = [(joiner, {many: rows[: self.compare_room()]})]
        elif tool and target:
            room = 4 if self.left() >= 15 and self.time_left() > CLOSING + 30 else 2
            plan = [(tool, {target: i}) for i in rows[:room]]
        else:
            return False
        if self.left() < len(plan) + 5:
            return False
        before = len(self.full)
        for name, args in plan:
            faults: list[str] = []
            args = _shape(self.fill_required(name, args), self.schemas.get(name), "args", faults)
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
        """Plan cart tool argument sets that leave exactly the wanted lines and remove the given ones, from the tool's
        schema. A tool taking an array of line objects gets one call listing every wanted line; otherwise each
        removal uses the second mode option, a zero count on a set-style tool or a negative count when the minimum
        allows, and each changed line uses the first mode option with its count. The id parameter may be an id-named
        whole number, never the count. Returns None when no safe way exists."""
        name = self.roles.get("cart") or ""
        schema = self.schemas.get(name) or {}
        props = schema.get("properties") or {}
        for key, spec in props.items():
            shape = spec.get("items") if isinstance(spec, dict) and "array" in _kinds(spec) else None
            if isinstance(shape, dict) and isinstance(shape.get("properties"), dict):
                inner = shape["properties"]
                id_key = next((k for k, s in inner.items() if isinstance(s, dict) and "string" in _kinds(s) and "enum" not in s), None) or _id_number_key(inner)
                count_key = next((k for k, s in inner.items() if isinstance(s, dict) and "integer" in _kinds(s) and k != id_key), None)
                if not id_key:
                    return None
                return [{key: [{id_key: i, **({count_key: n} if count_key else {})} for i, n in lines.items()]}]
        texts = [k for k, s in props.items() if isinstance(s, dict) and "string" in _kinds(s) and "enum" not in s]
        target = (next((k for t, k in sorted(self.id_args) if t == name and k in texts), None)
                  or next((k for k in schema.get("required") or [] if k in texts), None) or (texts[0] if len(texts) == 1 else None))
        if not target and not texts:
            target = _id_number_key(props)
        count = next((k for k, s in props.items() if isinstance(s, dict) and "integer" in _kinds(s) and k != target), None)
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
        """When the agent tries to end without a purchase after a review confirmed a candidate, build and buy the cart
        in code, at most once. One candidate per item, each from a different category, joins current lines covering
        the other items; after reading attributes and passing a new cart review, Kyo issues the cart calls, runs the
        order-time steps and checks out, retrying once with refreshed offers and another review if the store refuses.
        Returns a report, or an empty string when it stopped short."""
        cart, buy = self.roles.get("cart"), self.roles.get("buy")
        if self.completed or not self.confirmed or self.sold_out(self.confirmed) or not cart or not buy:
            return ""
        self.completed = True
        cat_key = self.category_key()

        def kind(ident: str) -> str:
            """Return a listing's lowercased category value, or the id itself when no category key is known, used to
            keep at most one planned line per category."""
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
            self.book_cart(cart, args, reply.get("observation"))
            self.observe(cart, reply.get("observation"))
            done.append(_dump(args))
        faults = []
        args = _shape(self.fill_required(buy, dict(self.brief.get("buy_args") or {})), self.schemas.get(buy), "args", faults)
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
        """True when the call targets the checkout tool and carries every fixed checkout argument from the brief with
        its expected value."""
        if name != self.roles.get("buy"):
            return False
        fixed = self.brief.get("buy_args") if isinstance(self.brief.get("buy_args"), dict) else {}
        return all(args.get(k) == v for k, v in fixed.items())

    def is_refuse(self, name: str, args: dict[str, Any]) -> bool:
        """True when the call targets the no-purchase tool and carries every fixed argument the brief gives for ending
        without a purchase."""
        if name != self.roles.get("refuse"):
            return False
        fixed = self.brief.get("refuse_args") if isinstance(self.brief.get("refuse_args"), dict) else {}
        return all(args.get(k) == v for k, v in fixed.items())

    def offered(self) -> list[dict[str, Any]]:
        """Return the tool definitions the agent may call this turn, never including the messaging tool. As steps run
        out the list narrows to the finishing tools at the very end, finishing plus cart tools a little earlier, and
        those plus the offer tool when few steps remain before any checkout attempt, falling back to the full list if
        narrowing leaves nothing."""
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
        """Fill or overwrite the free-text arguments of an ending without a purchase with a short fixed apology. It
        applies to plain string parameters that are present or required, whenever the text is empty, mentions a known
        listing id, or the request did not explicitly invite alternatives."""
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
        """Swap listing titles for ids in the single free argument of a cart, open, offer or compare tool, where that
        argument belongs to the cart tool or has been seen carrying ids. Text that begins a field of exactly one
        stored row is replaced by that row's id, while text matching several rows or none yields a problem message so
        the caller can reject the call. Returns the arguments and the problem text, empty when all is well."""
        if name not in {self.roles.get(r) for r in ("cart", "open", "offer", "compare")} - {None} or not self.id_key:
            return args, ""
        props = (self.schemas.get(name) or {}).get("properties") or {}
        free = [k for k, v in props.items() if isinstance(v, dict) and ("string" in _kinds(v) or "array" in _kinds(v)) and "enum" not in v]
        if len(free) != 1 or (name != self.roles.get("cart") and (name, free[0]) not in self.id_args):
            return args, ""

        def fix(value: Any) -> tuple[Any, str]:
            """Map one argument value to a listing id: known ids pass through trimmed, non-text, id-shaped or short
            values stay as they are, and longer free text is matched against the start of stored row fields, giving
            the id on a unique match or a problem message otherwise."""
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
        """Guess whether a cart tool sets quantities outright rather than adding to them: it has no multi-option mode
        parameter and some integer parameter with a declared minimum of zero."""
        props = self.schemas.get(name, {}).get("properties") or {}
        if any(isinstance(s, dict) and isinstance(s.get("enum"), list) and len(s["enum"]) >= 2 for s in props.values()):
            return False
        return any(isinstance(s, dict) and "integer" in _kinds(s) and s.get("minimum") == 0 for s in props.values())

    def note_cart(self, name: str, args: dict[str, Any]) -> None:
        """Update Kyo's own cart record from a cart call's arguments and stamp the change step. Listings are recognised
        by id_of, so numeric ids are never read as counts. Lists of line objects, or an empty list meant for them,
        replace the whole record; otherwise each named listing is reduced under a removal mode (one unit without a
        count), set to the count on a set-style tool, or raised by the count, one by default, and lines reaching zero
        are deleted."""
        self.cart_touched = True
        self.cart_at = self.step
        spec = self.schemas.get(name, {}).get("properties") or {}
        removal = False
        for key, value in args.items():
            options = (spec.get(key) or {}).get("enum")
            if isinstance(options, list) and len(options) >= 2 and value in options[1:]:
                removal = True
        loose_count = next((_count(v) for v in args.values() if _count(v) is not None and not isinstance(v, str) and not self.id_of(v)), None)
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
                        ident = next((self.id_of(v) for v in line.values() if self.id_of(v)), None)
                        count = next((_count(v) for v in line.values() if _count(v) is not None and not isinstance(v, str) and not self.id_of(v)), 1)
                        if ident:
                            lines.append((ident, count))
                    elif self.id_of(line):
                        lines.append((self.id_of(line), loose_count))
            elif self.id_of(value):
                lines.append((self.id_of(value), loose_count))
        if whole:
            self.cart_lines = {i: c for i, c in lines if c and c > 0}
            return
        sets = self.cart_sets(name)
        for ident, count in lines:
            have = self.cart_lines.get(ident, 0)
            if removal:
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
        """Try to take the cart contents straight from a cart tool reply. It searches up to three levels for a list of
        identifiable listing dicts, preferring one holding a listing the call touched (a lone empty list means an
        empty cart), and finds the per-line quantity key among small integers, settling ties by simulating the
        expected counts, then remembers it. Returns 'synced' after replacing Kyo's line record and cart view,
        'unreadable' when lines or counts cannot be resolved, and 'none' when the reply has nothing of the kind."""
        lists: list[list[Any]] = []

        def walk(node: Any, depth: int) -> None:
            """Recursive helper that gathers each list consisting only of dicts, an empty list included, from nested
            dict values, without descending into lists and stopping past depth three."""
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
        touched = [self.id_of(v) for value in (args or {}).values() for v in (value if isinstance(value, list) else [value]) if self.id_of(v)]
        with_ids = [rows for rows in lists if rows and any(self.ident(r, True) for r in rows)]
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
        """Account for a cart call: stamp the cart change step, then adopt the reply's lines whenever sync_cart can read
        them. If no line list comes back although earlier replies from this store did list the lines, the call is
        assumed to have left the cart as it was; otherwise the change is worked out from the call's arguments."""
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
        """Render a tool result as compact JSON for the agent, cutting each string to SHOW_CHARS characters and the
        whole text to OBS_LIMIT with a cut marker."""
        text = _dump(_clip(observation, SHOW_CHARS))
        if len(text) > OBS_LIMIT:
            text = text[:OBS_LIMIT] + "...(cut)"
        return text

    def compare_target(self) -> str | None:
        """Return the array parameter through which the side-by-side tool takes several listings, or None when there is
        no such tool or array parameter, or when another required parameter is one can_fill rejects."""
        side = self.roles.get("compare")
        schema = self.schemas.get(side or "") or {}
        props = schema.get("properties") or {}
        target = next((k for k, v in props.items() if isinstance(v, dict) and "array" in _kinds(v)), None)
        if not target or any(k != target and not self.can_fill(side, k) for k in schema.get("required") or []):
            return None
        return target

    def compare_room(self) -> int:
        """Return the batch size for the next compare call. After a recorded failure it takes the smaller of the failed
        size minus one and half the failed size, lifted to at least two and to the biggest size that has succeeded,
        then capped by the schema's maxItems; with no failure it uses maxItems when given, else the larger of four
        and the biggest successful size."""
        spec = ((self.schemas.get(self.roles.get("compare") or "") or {}).get("properties") or {}).get(self.compare_target() or "") or {}
        cap = spec["maxItems"] if isinstance(spec.get("maxItems"), int) else None
        if self.compare_bad:
            room = max(2, self.compare_ok, min(self.compare_bad - 1, max(2, self.compare_bad // 2)))
            return min(room, cap) if cap else room
        if cap:
            return cap
        return max(4, self.compare_ok)

    def join_opens(self, calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Fold the agent's separate open calls for known, not-gone listings in one turn into a single compare call, up
        to the compare size limit, when distinct open and compare tools exist and the compare tool is not ruled out
        for attributes. The first folded slot carries the compare call and the rest are marked as joined; with fewer
        than two candidates the calls stay unchanged."""
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
        """Execute one agent turn's tool calls and return (call id, reply text) pairs. After merging opens, calls that
        message the shopper, are unknown, malformed, repeated, answerable from cache, not alone when finishing, or
        beyond the remaining steps are answered without spending a step. Finishing calls first pass Kyo's checks
        (wider search, offer refresh, sold-out swaps, page chains, cart review; for an ending without a purchase the
        open chains and the review). Results are absorbed, string arguments remembered, cart and tool-role knowledge
        updated, extra pages fetched, and replies annotated."""
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
            if ending and self.left() > 2 and self.time_left() > CLOSING - 25:
                if self.is_buy(name, args):
                    wider = self.wider_hold()
                    if wider:
                        out.append((cid, wider + "\nCart now: " + self.cart_text()))
                        continue
                    reread = self.reread_offers()
                    if self.cart_touched and self.cart_seen_at < self.cart_at and self.roles.get("cart_view") and self.left() > 3:
                        faults = []
                        view_args = _shape(self.fill_required(self.roles["cart_view"], {}), self.schemas.get(self.roles["cart_view"]), "args", faults)
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
                elif self.time_left() > CLOSING + 15:
                    if self.left() > 5:
                        chain = self.chains_before_decline()
                        if chain:
                            out.append((cid, "Not run yet. " + chain))
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
            if self.is_refuse(name, args) and self.left() <= 5 and not self.late_held and self.time_left() > CLOSING + 15:
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
            if self.is_refuse(name, args) and not self.leads and not self.pushed and self.left() > self.max_steps // 3 and self.time_left() > CLOSING + 15:
                self.pushed = True
                out.append((cid, f"Not run yet (no step used). {self.left()} steps are still left, so search in ways you have not tried before ending: browse the category of each requested item and page through it with the token or offset the store returns, try single-word value queries and the property's other names, and open the listings that might fit. End without a purchase only after that."))
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
                    if all(str(i) in self.rows for i in args[many]):
                        self.compare_bad = min(self.compare_bad or 999, len(args[many]))
                    self.join_failed = True
                    trimmed += f"\n[Kyo] The side-by-side view showed nothing for {len(args[many])} listings: send at most {self.compare_room()} per call."
                else:
                    self.compare_ok = max(self.compare_ok, len(args[many]))
                    self.join_failed = False
            self.recent = (self.recent + [mark])[-3:]
            self.given.update(((name, k), v) for k, v in args.items() if isinstance(v, str) and v.strip() and v.strip() not in self.rows)
            self.id_args.update((name, k) for k, v in args.items() if (v if isinstance(v, list) else [v]) and all(self.id_of(x) for x in (v if isinstance(v, list) else [v])))
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
                self.book_cart(name, args, observation)
                self.cart_said = _dump(observation)
                self.recent = []
                counts = [_count(v) for v in args.values() if not isinstance(v, str) and _count(v) is not None and not self.id_of(v)]
                for value in args.values():
                    named = self.id_of(value)
                    if counts and named:
                        if counts[0] <= 0 and not _in_rows(observation, named):
                            self.dropped.add(named)
                        elif counts[0] > 0:
                            self.dropped.discard(named)
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

    def finish(self, again: bool = False) -> str:
        """Close the episode from code. If the cart seems to hold lines, it first opens any cart page never viewed, runs
        the order-time steps when at least two steps and enough seconds remain, and checks out with required
        parameters filled; with again set, a refused checkout is reported back instead of falling through. Otherwise,
        or after such a refusal without again, it runs the ending-time steps and ends without a purchase, its free
        text passed through quiet. Returns a description of a refusal, or an empty string."""
        if self.finished or self.broke or self.left() <= 0:
            return ""
        buy, refuse = self.roles.get("buy"), self.roles.get("refuse")
        fresh = self.cart_view is not None and self.cart_seen_at >= self.cart_at
        holding = bool(self.cart_ids()) if fresh else bool(self.cart_lines)
        if buy and (holding or (self.cart_touched and not fresh and self.left() >= 2)):
            faults: list[str] = []
            args = _shape(self.fill_required(buy, dict(self.brief.get("buy_args") or {})), self.schemas.get(buy), "args", faults)
            if not faults:
                self.view_unopened(self.tail())
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
        if refuse:
            faults = []
            args = _shape(self.fill_required(refuse, self.quiet(refuse, dict(self.brief.get("refuse_args") or {}))), self.schemas.get(refuse), "args", faults)
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
        """Keep the conversation under HISTORY_LIMIT characters by shortening the older half of tool replies longer than
        600 characters to their first 400, followed by a list of up to 20 known listing ids they mentioned."""
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
        """Drive the whole episode: read the request, seed the conversation with the guide, the request, any opening
        store view (normalised by _unline) and the status, set the call limits and run the tool loop. If that leaves
        the episode open, Kyo forces a finish; when the store rejects it and steps, time and funds remain, the loop
        gets a few more turns before a final forced finish. Returns the event log."""
        if self.finished:
            return self.trace
        self.read_request()
        self.trace.append({"kind": "brief", "brief": self.brief})
        opening = self.view.get("opening")
        if opening is not None:
            opening = _unline(opening)
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
            if turned and not self.finished and not self.broke and not self.no_credit and self.left() > 1 and self.time_left() > CLOSING:
                self.chat.append({"role": "user", "content": "[Kyo] Kyo tried to end the episode and " + turned + "\nApply what the store says, then finish.\n" + self.status()})
                self.work(self.loop_calls + 10, SPEND_HARD + 0.01)
            if not self.finished:
                self.finish()
        self.trace.append({"kind": "end", "finished": self.finished, "steps": self.step, "cost": round(self.cost, 5), "calls": self.calls})
        return self.trace

    def work(self, loop_cap: int, spend_cap: float) -> None:
        """Run the agent's tool loop until the episode finishes or a turn, spend, step or time limit is hit. A round
        runs the interview when due; otherwise the light model, falling back to the spare one, gets the trimmed
        history and the offered tools, and its normalized calls go through run_calls with the status block after the
        last reply. Turns without calls get a nudge (three in a row end the loop), repeated model failures end it,
        and a cart or finishing call before the interview triggers the interview instead."""
        idle = 0
        while (not self.finished and not self.broke and self.left() > 0 and self.loop_calls < loop_cap and self.cost < spend_cap
               and self.time_left() > CLOSING):
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
    """Entry point: create the agent for the given problem, run it and return its event log. A failure during
    construction yields a one-entry crash log; a failure while running is logged and followed by an attempt to finish
    the episode, whose own error, if any, is logged too."""
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
