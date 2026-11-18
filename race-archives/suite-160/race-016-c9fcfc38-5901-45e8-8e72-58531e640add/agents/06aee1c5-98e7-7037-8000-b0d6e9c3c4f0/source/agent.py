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
SPEND_SOFT = 0.040
SPEND_HARD = 0.052
SPEND_STOP = 0.055          # no model call goes out above this spend: the code finishes alone (one run key pays for every task)
SPEND_LIGHT = 0.028         # above this spend the late checks (cart check, decline review, swaps) try the light model first
# each task has 300 s in all on the network: Kyo keeps its own clock from the start of agent_main
DEADLINE = 250.0            # seconds Kyo plans to use, leaving room for one slow call and the finish
CLOSING = 70.0              # with less time than this left, the tool loop stops and the episode heads for its finish
LAST_CALL = 28.0            # with less time than this left, no model call starts: the code finishes alone
QUESTION_GAP = 30.0         # once the interview has started, a further question needs this much time before the closing (room for a slow reply)
QUESTION_TIME = 45.0        # the interviewer asks only while more time than CLOSING plus this is left
# waits before resending a failed model call; an episode has 300 s in all, so at most 45 s go to waiting
WAITS = (4, 10, 20, 35)
WAIT_TOTAL = 45
SHOW_CHARS = 450
OBS_LIMIT = 15000
HISTORY_LIMIT = 150000
GATE_ROUNDS = 3             # store calls the process gate may make before one checkout or ending
AUTO_PAGES = 6              # further search pages Kyo reads itself for a cheapest, most or least pick
# Revisions that wait for a paired test: each is off until it is proven no worse than the file without it.
SWITCH = {"LINK_FIRST": True, "UNIT_JOIN": False, "DOUBT_HOLD": False, "FIT_LINK": False, "OPEN_SHOWN": False,
          # the question plan, each part on its own
          "CAND_LIST": False, "MSG_SHARE": False, "LEAD_FORM": False, "LEAD_ROTATE": False, "REPLY_ADAPT": False,
          "ASK_TIMING": False, "CAT_FROM_LABELS": False,
          # cheaper fits under a cheapest pick
          "CHEAP_SWEEP": False}
ASK_EXTRAS = 0              # extra properties named after the lead in one question: 0 asks one property per message (many
                            # shoppers refuse lists and the message is spent); 2 for a world known to take lists
CHEAPER_OPENS = 3           # cheaper rows of a cart line's kind whose attributes Kyo reads before a cheapest checkout

# one attempt per call (think() resends): a stuck call costs 50 s, not minutes, inside the 300 s
_llm = ProxyClient(timeout=50, max_retries=1, retry_delay=1)
_env = ProxyClient(timeout=50, max_retries=3, retry_delay=1)
_env_late = ProxyClient(timeout=12, max_retries=1, retry_delay=1)   # near the deadline a hanging store call cannot carry the task past it


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


# ---------- universal reader (M1): records the store sends as text become records ----------
# A store may send a listing as an object, as JSON inside a string, or as one line of "label: value" parts joined by a
# separator. The reader finds the separator from the text itself (the one that splits it into the most labelled parts),
# never from a list of store words, and leaves sentences (notes, answers, refusals) untouched.
_SEPARATORS = (" | ", " || ", "|", " ¦ ", " · ", " ; ", "; ", "\n", "\t")
_LABELLED = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_ ./-]{0,40}?)\s*:\s?(.*)$", re.S)
_PAIRED = re.compile(r"^\s*([^=:;,]{1,40}?)\s*[=]\s*(.*)$", re.S)
_EQUALS = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_ ./-]{0,40}?)\s*=\s?(.*)$", re.S)


def _ident_text(value: Any) -> bool:
    """A raw value that names a listing: one token of 3 to 64 characters holding a letter and a digit, or six digits or
    more. Page numbers, counts and short prices are not identifiers."""
    if not isinstance(value, str):
        return False
    token = value.strip()
    if not 3 <= len(token) <= 64 or " " in token:
        return False
    return bool(re.search(r"\d", token) and re.search(r"[A-Za-z]", token)) or bool(re.fullmatch(r"\d{6,}", token))


def _scalar(text: str, keep_text: bool = False) -> Any:
    """A record value as the store meant it: whole and decimal numbers become numbers (a value starting with 0, and the
    record's first field, which names the listing, stay text so codes and ids keep their form), none/null become None,
    true/false booleans; everything else stays text."""
    low = text.strip().lower()
    if low in ("none", "null", ""):
        return None
    if low in ("true", "false"):
        return low == "true"
    if keep_text:
        return text.strip()
    if re.fullmatch(r"-?[1-9]\d{0,4}|0", text.strip()):
        return int(text.strip())            # six digits and more stay text: they may be a code (prices still read as numbers)
    if re.fullmatch(r"-?(?:[1-9]\d{0,12}|0)\.\d+", text.strip()):
        return float(text.strip())
    return text.strip()


def _pairs_table(text: str) -> dict[str, Any] | None:
    """A value made of several name=value pairs (a property table written as text) as a table; None otherwise."""
    if "=" not in text:
        return None
    for sep in (";", ","):
        parts = [p for p in text.split(sep) if p.strip()]
        if len(parts) < 2:
            continue
        hits = [_PAIRED.match(p) for p in parts]
        good = [(h.group(1).strip(), h.group(2).strip()) for h in hits if h and h.group(1).strip()]
        # commas also separate words of a title ("Buy2 Take1=3PCS, gift"): there, every part must be a pair (or three pairs and
        # one continuation)
        enough = ((len(good) >= 2 and len(good) >= len(parts) - 1) if sep == ";"
                  else ((len(good) >= 2 and len(good) == len(parts)) or (len(good) >= 3 and len(good) >= len(parts) - 1))
                  and not any(re.search(r"\d", k) for k, _ in good))
        if enough and hits[0] and hits[0].group(1).strip():   # a value that starts with free text is not a table
            table: dict[str, Any] = {}
            last = None
            for part, hit in zip(parts, hits):
                if hit and hit.group(1).strip():
                    last = hit.group(1).strip()
                    table[last] = hit.group(2).strip()
                elif last is not None:
                    table[last] += sep + part
            return table
    return None


def _best_separator(text: str, labelled: Any = None) -> str | None:
    """The separator that splits a text into the most labelled parts, when the first part is labelled, three or more
    are, and labelled parts are not lost among free text; None when the text is not written as labelled parts."""
    labelled = labelled or _LABELLED
    best: tuple[tuple[bool, int], str] | None = None
    for sep in _SEPARATORS:
        if sep not in text:
            continue
        parts = [p for p in text.split(sep) if p.strip()]
        keyed = sum(1 for p in parts if labelled.match(p))
        head = labelled.match(parts[0]) if parts else None
        if head and keyed >= 3 and keyed >= 0.3 * len(parts):
            # a separator under which the first field holds an identifier wins over one that only yields more labels
            # (a listing's description may itself hold many "label: value" lines)
            value = head.group(2).strip()
            score = (_ident_text(value), keyed)
            if best is None or score > best[0]:
                best = (score, sep)
    return best[1] if best else None


def _has_ident(rec: dict[str, Any]) -> bool:
    """A record (raw values, before numbers are read) names a listing: one of its values is an identifier. A notice
    written as labelled lines ("Budget: lowered to 40") names none, so it stays a sentence for the note reader."""
    return any(_ident_text(v) for v in rec.values())


def _text_record(text: Any) -> dict[str, Any] | None:
    """One listing or page written as labelled parts as a record; None for anything else (sentences, notices, answers:
    a record must carry an identifier). Labels end with a colon, or, when no part is written that way, with an equals
    sign (a label, an equals sign, then the value). A part without a label of its own continues the value before it."""
    if not isinstance(text, str) or len(text) < 12:
        return None
    labelled = _LABELLED
    sep = _best_separator(text)
    if sep is None:
        sep = _best_separator(text, _EQUALS) if "=" in text and ":" not in text else None
        labelled = _EQUALS
    if sep is None:
        return None
    rec: dict[str, Any] = {}
    last = None
    shouting = None                 # whether the store writes its labels in capitals (from the first label)
    for part in text.split(sep):
        hit = labelled.match(part)
        label = hit.group(1).strip() if hit else ""
        if label and shouting is None:
            shouting = label.isupper()
        # a phrase in capitals with a colon inside a value (a title's "FREE: TWO SPEAKERS") is not a field of a store that labels in lower case
        value = hit.group(2).strip() if hit else ""
        shouted = label.isupper() and len(label) > 2 and value.isupper() and len(value.split()) >= 2
        if label and label not in rec and not (last is not None and shouted and not shouting):
            last = label
            rec[last] = hit.group(2)
        elif last is not None:
            rec[last] += sep + part
        elif part.strip():
            return None
    if len(rec) < 3 or not _has_ident(rec):
        return None
    first = next(iter(rec))
    for key, value in list(rec.items()):
        table = _pairs_table(value) if labelled is _LABELLED else None
        rec[key] = table if table is not None else _scalar(value, key == first)
    return rec


def _table_records(text: str) -> list[Any] | None:
    """A text table (a header line of labels, then one line per listing, cells split by the same mark; rule lines such
    as |---| are skipped) as one record per row; None when the text is not such a table or no column holds identifiers."""
    lines = [l for l in text.split("\n") if l.strip() and not re.fullmatch(r"[\s|:+=-]+", l)]
    if len(lines) < 2:
        return None
    for sep in ("|", "\t", ";"):
        rows = [[c.strip() for c in l.strip().strip("|").split(sep)] for l in lines if sep in l]
        if len(rows) != len(lines) or len(rows[0]) < 3 or any(len(r) != len(rows[0]) for r in rows):
            continue
        head = rows[0]
        if not all(c and len(c) <= 40 and not _ident_text(c) for c in head) or len(set(head)) < len(head):
            continue
        body = rows[1:]
        if not any(sum(1 for r in body if _ident_text(r[i])) * 2 >= len(body) for i in range(len(head))):
            continue
        return [{h: _scalar(v, i == 0) for i, (h, v) in enumerate(zip(head, r))} for r in body]
    return None


def _text_records(text: Any) -> Any:
    """A text holding one record, or one record per line (a page listing several listings), as a record or a list of
    records; None when it holds none. A label met twice on one line is the start of a next record."""
    if not isinstance(text, str):
        return None
    if "\n" in text:
        table = _table_records(text)
        if table is not None:
            return table
    lines = [l for l in text.split("\n") if l.strip()]
    if len(lines) >= 2:
        recs = [_text_record(l) for l in lines]
        found = [r for r in recs if r is not None]
        shaped = [l for l in lines if _best_separator(l)]
        # two or more lines are records of one shape (same first label); a header or footer line stays as text
        if len(found) >= 2 and len({next(iter(r)) for r in found}) == 1 and len(found) >= len(shaped) - 1:
            return [r if r is not None else l for r, l in zip(recs, lines)]
    labelled = _LABELLED
    sep = _best_separator(text)
    if sep is None and "=" in text and ":" not in text:
        labelled = _EQUALS
        sep = _best_separator(text, _EQUALS)
    if sep and sep != "\n":
        labels = [m.group(1).strip() for m in (labelled.match(p) for p in text.split(sep)) if m]
        if labels and labels.count(labels[0]) >= 2:
            chunks: list[str] = []
            for part in text.split(sep):
                hit = labelled.match(part)
                if hit and hit.group(1).strip() == labels[0] or not chunks:
                    chunks.append(part)
                else:
                    chunks[-1] += sep + part
            recs = [_text_record(c) for c in chunks]
            if len(recs) >= 2 and all(r is not None for r in recs):
                return recs
    return _text_record(text)


def _json_text(text: str) -> Any:
    """JSON a store wrapped in a string, parsed; None when the string is not JSON."""
    body = text.strip()
    if len(body) < 2 or body[0] not in "[{" or body[-1] not in "]}":
        return None
    try:
        return json.loads(body)
    except ValueError:
        return None


def _is_record(node: Any) -> bool:
    """A dict that already is a record: three or more plain values. Its text fields (descriptions) are never split. One
    exception: a result wrapper (counts, page numbers) holding its listings as one text (a table, lines of records) and
    no identifier among its other values is not a record, so that text is read."""
    if not isinstance(node, dict) or sum(1 for v in node.values() if v is None or isinstance(v, (str, int, float, bool))) < 3:
        return False
    holders = [k for k, v in node.items() if isinstance(v, str) and ("\n" in v or " | " in v) and _text_records(v) is not None]
    named = lambda k, v: _ident_text(v) or (isinstance(v, str) and _id_like(v, "id" in str(k).lower())) or (
        isinstance(v, int) and not isinstance(v, bool) and "id" in str(k).lower())
    if holders and not any(named(k, v) for k, v in node.items() if k not in holders):
        return False
    return True


def _records(node: Any, depth: int = 0, inside: bool = False) -> Any:
    """A store reply with every record written as text turned into a record (M1). Only the reply itself, list entries
    and the values of wrapper objects are read; everything inside a record that already is an object (its text fields,
    nested parts and lists) stays as it is. A reply that only wraps one record (a page under one key) becomes that
    record; replies already made of records, and sentences, come back unchanged."""
    if depth > 7:
        return node
    if isinstance(node, str):
        if inside:
            return node
        inner = _json_text(node)
        if inner is not None and depth < 6:
            return _records(inner, depth + 1)
        rec = _text_records(node)
        return rec if rec is not None else node
    if isinstance(node, list):
        return [_records(v, depth + 1, inside) for v in node]
    if isinstance(node, dict):
        record = inside or _is_record(node)
        out = {k: _records(v, depth + 1, record) for k, v in node.items()}
        if depth == 0 and len(out) == 1:
            only = next(iter(out.values()))
            if isinstance(only, dict) and isinstance(next(iter(node.values())), str):
                return only
        return out
    return node


def _unread(node: Any, depth: int = 0, inside: bool = False) -> int:
    """The reader's self-check: strings outside records that still look like a listing written as labelled parts (the
    same separators the reader uses, three or more labels, and an identifier among the values) but were not read."""
    if depth > 7:
        return 0
    if isinstance(node, str):
        if inside or len(node) < 12:
            return 0
        labelled = _LABELLED
        sep = _best_separator(node)
        if not sep and "=" in node and ":" not in node:
            labelled = _EQUALS
            sep = _best_separator(node, _EQUALS)
        if not sep:
            return 0
        values = [m.group(2).strip() for m in (labelled.match(p) for p in node.split(sep)) if m]
        return 1 if any(_ident_text(v) for v in values) else 0
    if isinstance(node, list):
        return sum(_unread(v, depth + 1, inside) for v in node)
    if isinstance(node, dict):
        record = inside or _is_record(node)
        return sum(_unread(v, depth + 1, record) for v in node.values())
    return 0


def _labelled_strings(node: Any) -> int:
    """Strings outside records that read as listings written as labelled parts (the reader's self-check count)."""
    return _unread(node)


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
        # M1 reader self-check, M4 search budget, M5 question form, M8 refusal rounds, M2 world features (telemetry)
        self.read_stats = {"replies": 0, "text_records": 0, "left_as_text": 0}
        self.raw_last = False
        self.text_last = False
        self.searches_since_cart = 0
        self.search_holds = 0
        self.search_hold_version = -1
        self.single_only = False
        self.refusal_rounds = 0
        self.world: dict[str, Any] = {}
        self.search_keys: dict[str, tuple[int, int]] = {}
        self.zero_hits = 0
        self.budget_held = -1
        self.answer_marks: dict[int, list[str]] = {}
        self.lines_held = -1
        self.same_run = 0
        self.ask_tries = 0
        self.ask_opened = 0
        self.last_search_words = ""
        self.last_search: tuple[str, dict[str, Any]] | None = None
        self.last_rows = False
        self.free_pages: dict[str, int] = {}
        self.last_count = 0
        self.commit_broken = False
        self.topup_item: dict[int, set[str]] = {}
        self.spend_soft, self.spend_hard, self.spend_stop, self.spend_light = SPEND_SOFT, SPEND_HARD, SPEND_STOP, SPEND_LIGHT
        self.situation = "unread"
        self.item_searches: dict[int, int] = {}
        self.doubt_held = -1
        self.swept: set[tuple[str, int]] = set()
        self.fit_flagged: set[str] = set()
        self.left_cart: set[tuple[str, int]] = set()
        self.rejected_ids: set[str] = set()
        self.guard_refused: set[tuple[str, str]] = set()
        self.list_extras = 2                  # LEAD_FORM: properties after the lead; REPLY_ADAPT lowers it after an unclear reply
        self.listed: dict[tuple[int, str], int] = {}
        self.interview_msgs = 0
        self.ask_wait: list[int] = []

        self.fit_held = -1
        self.shown_opened = -1

    # ---------- plumbing ----------
    def left(self) -> int:
        return self.max_steps - self.step

    def time_left(self) -> float:
        """Seconds of Kyo's own budget still left (the network stops a task at 300 s, finished or not)."""
        return DEADLINE - (time.monotonic() - self.t0)

    def checker(self) -> list[str]:
        """Models for a late check (cart check, decline review, sold-out swap, requirement unmixing): the heavy model
        first, the light one first once this task's spend passes SPEND_LIGHT (one run key pays for every task)."""
        return [HEAVY, LIGHT] if self.cost < self.spend_light else [LIGHT, HEAVY]

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
        self.searches_since_cart = 0
        self.same_run = 0
        self.free_pages = {}

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
            if self.calls >= self.call_cap or self.cost >= self.spend_stop:
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
            if self.broke or self.no_credit or self.calls >= self.call_cap or self.out_of_time or self.cost >= self.spend_stop:
                return None
        return None

    def act(self, name: str, args: dict[str, Any]) -> dict[str, Any] | None:
        payload = dict(self.binding)
        payload["step"] = self.step + 1
        payload["action"] = {"name": name, "args": args}
        for _ in range(4):
            readonly = name in {self.roles.get(r) for r in ("search", "open", "offer", "compare", "cart_view")} - {None} or bool(self.shapes.get(name))
            client = _env_late if self.time_left() < CLOSING and readonly else _env
            result = client.post_verbose("/environment/call", payload)
            if isinstance(result.data, dict):
                data = result.data
                if "observation" in data:
                    try:
                        data["observation"] = self.read_reply(name, data["observation"])
                    except Exception as exc:          # the reader never stops a store call: the reply goes on as sent, whole
                        self.raw_last = True
                        self.text_last = False
                        self.trace.append({"kind": "reader_error", "error": f"{type(exc).__name__}: {exc}"[:200]})
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

    def read_reply(self, name: str, observation: Any) -> Any:
        """M1: every store reply passes the universal reader before anything else sees it, except the replies of the
        messaging and finishing tools (answers, refusals and notices stay as written). The self-check counts the records
        found in text and any listing-like text the reader could not read: that reply is shown to the walker unclipped
        (raw_last), and when unread text outweighs read records the table counts as blind, so the empty-table guards step
        aside (an empty table means 'not read', not 'none exists')."""
        if name in (self.roles.get("ask"), self.roles.get("buy"), self.roles.get("refuse")):
            self.raw_last = False
            self.text_last = False
            return observation
        before = _unread(observation)
        read = _records(observation)
        after = _unread(read)
        self.read_stats["replies"] += 1
        self.read_stats["text_records"] += max(0, before - after)
        self.read_stats["left_as_text"] += after
        self.raw_last = after > 0
        self.text_last = before > after
        if before and "format" not in self.world:
            self.world["format"] = "text records" if after < before else "unread text"
        elif not before and "format" not in self.world and isinstance(read, (dict, list)) and self.records_in(read):
            self.world["format"] = "objects"
        return read

    def table_blind(self) -> bool:
        """True when the code's listing table cannot be trusted to be complete: more listing-like text stayed unread than
        was read, or listings were shown but none reached the table."""
        stats = self.read_stats
        return stats["left_as_text"] > max(2, stats["text_records"]) or (stats["replies"] >= 3 and not self.rows and stats["text_records"] == 0 and self.step >= 6)

    # ---------- search budget (M4) ----------
    def is_search(self, name: str) -> bool:
        """A store call that lists many listings (a search or a category browse): the brief's search tool, or any tool
        whose replies came back as lists of listings; single-listing views and the cart are not searches."""
        if name in (self.roles.get("cart"), self.roles.get("cart_view"), self.roles.get("buy"), self.roles.get("refuse"), self.roles.get("ask")):
            return False
        if name == self.roles.get("search"):
            return True
        return bool(self.shapes.get(name)) and name not in self.page_tools()

    def search_cap(self) -> int:
        """Searches allowed between cart changes: about three per requested item and a few spare, never more than a
        third of the task's steps (one item's unprovable value must not use up the steps of the whole basket)."""
        return max(6, min(3 * len(self.items) + 4, max(6, self.max_steps // 3)))

    def search_hold(self, name: str, args: dict[str, Any] | None = None) -> str:
        """Hold one search (no step used) when the searching since the last cart change has reached the cap while
        listings were seen: the walker is told to cart the closest fit for each item. At most twice per task version;
        after a hold four more searches pass before the next one, so a real gap can still be searched."""
        if not self.is_search(name) or self.searches_since_cart < self.search_cap() or not self.rows:
            return ""
        if args is not None and self.is_paging(name, args):
            return ""                   # paging on through the same search is never held (four pages at most)
        if self.search_hold_version != self.version:
            self.search_hold_version = self.version
            self.search_holds = 0
        if self.search_holds >= 2 or self.left() <= self.tail() + 2:
            return ""
        self.search_holds += 1
        self.searches_since_cart = self.search_cap() - 4
        self.world["search_holds"] = self.world.get("search_holds", 0) + 1
        self.trace.append({"kind": "search_hold", "step": self.step, "rows": len(self.rows)})
        many = len(self.items) > 1
        return (f"Not run (no step used): {self.search_cap()} searches since the cart last changed, and {len(self.rows)} listings are already seen. "
                + ("Put the best-fitting listing you have seen for EACH requested item in the cart now" if many else "Put the best-fitting listing you have seen in the cart now")
                + " (a requirement no listing shows is no reason to buy nothing: take the closest fit and the lower price), then check out. "
                "Search again only for an item that has no candidate at all."
                + (" Closest fits by their own fields: " + self.picks_text(self.code_picks()) + "." if self.code_picks() else ""))

    # ---------- refused finish (M8) ----------
    def refusal_brief(self) -> str:
        """What code can establish after a refused finish, before the walker acts: the cart's lines re-read once (stock
        and price), the lines now gone, the budget as it now stands against the cart's total, and the code's closest
        replacement for a gone line among listings already seen."""
        notes: list[str] = []
        changed = self.reread_offers() if self.left() > len(self.cart_ids()) + 3 else ""
        if changed:
            notes.append("Re-read just now: " + changed[:1200])
        gone = [i for i in self.cart_ids() if self.sold_out(i)]
        if gone:
            notes.append("Gone from the cart's lines: " + ", ".join(gone) + ".")
            picks = self.code_picks()
            swaps = []
            for number, (ident, proven, _unknown) in picks.items():
                if ident not in gone and ident not in self.cart_ids():
                    swaps.append(f"item {number}: {ident}" + (f" ({self.price_of(ident):g})" if self.price_of(ident) is not None else ""))
            if swaps:
                notes.append("Closest seen replacements by their own fields: " + "; ".join(swaps) + ".")
        if isinstance(self.budget, (int, float)) and self.budget > 0:
            prices = [(self.price_of(i), self.cart_lines.get(i, 1)) for i in self.cart_ids()]
            if prices and all(p is not None for p, _ in prices):
                total = sum(p * n for p, n in prices)
                notes.append(f"Budget now {self.budget:g} ({self.budget * self.price_scale():g} in the store's prices); the cart totals {total:g}.")
        self.world.setdefault("refusals", []).append({"gone": len(gone), "reread": bool(changed)})
        return "[Kyo checked] " + " ".join(notes) if notes else ""

    # ---------- question allocation over a basket and answer classes (M5) ----------
    def item_spread(self, number: int, table: list[tuple[str, str, int, list[str]]]) -> float:
        """How much an item's seen listings still differ on the properties a question could ask: the sum, over the
        item's best-fitting category keys that the request does not settle, of the share of distinct values. An item
        whose listings all agree has little a hidden preference could still decide."""
        item = self.items[number - 1]
        fits = {label: self.label_fit(label, item) for label, _, _, _ in table}
        best = max(fits.values(), default=0)
        settled = " ".join(item.musts + item.told).lower()
        score = 0.0
        for label, name, carried, _values in table:
            if best <= 0 or fits[label] != best or carried < 2 or name in settled:
                continue
            distinct = self.key_distinct.get((label, name), 1)
            score += min(1.0, (distinct - 1) / max(1, carried - 1))
        return score

    def key_spread(self, number: int, table: list[tuple[str, str, int, list[str]]]) -> dict[str, float]:
        """Per property key of the item's best-fitting category: the share of distinct values among the listings that carry
        it, weighted by how many carry it. A preference the shopper holds back decides between listings only where they
        differ, so a key on which every listing agrees is a weak question."""
        item = self.items[number - 1]
        fits = {label: self.label_fit(label, item) for label, _, _, _ in table}
        best = max(fits.values(), default=0)
        out: dict[str, float] = {}
        for label, name, carried, _values in table:
            if best > 0 and fits[label] != best:
                continue
            distinct = self.key_distinct.get((label, name), 1)
            share = (distinct - 1) / max(1, carried - 1)
            out[_words(name)] = max(out.get(_words(name), 0.0), min(1.0, share) * min(1.0, carried / 4))
        return out

    def rank_queue(self, number: int, queue: list[str], table: list[tuple[str, str, int, list[str]]]) -> list[str]:
        """The plan's own picks keep their order and come first; the code's top-up keys after them go by spread, most
        differing first, so the properties named after the lead in a question are the likelier deciders."""
        spread = self.key_spread(number, table)
        mine = self.topup_item.get(number, set())
        planned = [x for x in queue if x not in mine]
        topped = [x for x in queue if x in mine]
        if SWITCH["CAND_LIST"]:
            # CAND_LIST: the hidden property is mostly among the category's most carried keys: carried count first,
            # the spread of values second
            carried = self.key_carried(number, table)
            topped.sort(key=lambda x: (-carried.get(_words(x), 0), -spread.get(_words(x), 0.0)))
        else:
            topped.sort(key=lambda x: -spread.get(_words(x), 0.0))
        return planned + topped

    def key_carried(self, number: int, table: list[tuple[str, str, int, list[str]]]) -> dict[str, int]:
        """How many listings of the item's best-fitting category carry each key."""
        item = self.items[number - 1]
        fits = {label: self.label_fit(label, item) for label, _, _, _ in table}
        best = max(fits.values(), default=0)
        out: dict[str, int] = {}
        for label, name, carried, _ in table:
            if best > 0 and fits[label] == best:
                out[_words(name)] = max(out.get(_words(name), 0), carried)
        return out

    def basket_order(self, order: list[int], table: list[tuple[str, str, int, list[str]]], count: int) -> list[int]:
        """With more items than messages, the items whose listings differ most are asked first (a basket of four or five
        items left some never asked); otherwise the plan's own order stands."""
        if SWITCH["MSG_SHARE"] and len(order) > 1:
            # MSG_SHARE: the products the request says least about are asked about first (their hidden needs are likelier)
            spread = {n: self.item_spread(n, table) for n in order}
            said = {n: len([m for m in self.items[n - 1].musts if m not in self.items[n - 1].dropped]) for n in order}
            ranked = sorted(order, key=lambda n: (said[n], -spread[n], order.index(n)))
            self.trace.append({"kind": "basket_order", "said": {str(k): v for k, v in said.items()}, "order": ranked})
            return ranked
        if len(self.items) <= count or len(order) <= 1:
            return order
        spread = {n: self.item_spread(n, table) for n in order}
        ranked = sorted(order, key=lambda n: (-spread[n], order.index(n)))
        self.trace.append({"kind": "basket_order", "spread": {str(k): round(v, 2) for k, v in spread.items()}, "order": ranked})
        return ranked

    def answer_class(self, heard: dict[str, Any], number: int) -> str:
        """What one answer gave for its item: 'heard' (a firm preference), 'free' (the shopper does not mind) or 'none'
        (a reply that names no property or value). Counted per item for the task's world record."""
        if any(isinstance(e, dict) and (_count(e.get("item")) or 1) == number and str(e.get("need") or "").strip() for e in _items(heard.get("needs"))):
            mark = "heard"
        elif any(isinstance(e, dict) and (_count(e.get("item")) or 1) == number for e in _items(heard.get("free"))):
            mark = "free"
        else:
            mark = "none"
        self.answer_marks.setdefault(number, []).append(mark)
        self.world.setdefault("answers", {}).setdefault(mark, 0)
        self.world["answers"][mark] += 1
        return mark

    # ---------- search screen (M10) and the store's search word rule (M2) ----------
    def is_paging(self, name: str, args: dict[str, Any]) -> bool:
        """The call asks for another page of the search just made: same tool, same words, exactly one numeric argument
        changed (or added), the page before showed listings, and fewer than four such pages of this search so far."""
        if not self.last_search or self.last_search[0] != name or not self.last_rows:
            return False
        before = self.last_search[1]
        if self.words_key(name, args) != self.words_key(name, before):
            return False
        changed = [k for k in set(args) | set(before) if args.get(k) != before.get(k)]
        if len(changed) != 1:
            return False
        key = changed[0]
        number = lambda v: int(v) if (isinstance(v, int) and not isinstance(v, bool)) or (isinstance(v, str) and re.fullmatch(r"\s*\d+\s*", v)) else None
        new, old = number(args.get(key)), number(before.get(key))
        if new is None or (before.get(key) is not None and old is None):
            return False
        shape = self.shapes.get(name) or {}
        if shape.get("next_arg"):
            if key != shape["next_arg"]:
                return False                # the store's own page argument is known: only it pages
        elif not ((old is None and new in (2, self.last_count)) or (old is not None and new - old in (1, self.last_count))):
            return False                    # without it, only the next page number or an offset by the rows shown pages
        return self.free_pages.get(self.words_key(name, args), 0) < 4

    def words_key(self, name: str, args: dict[str, Any]) -> str:
        """A search's words alone (no page or sort arguments): the next page of the same search has the same words key."""
        words: list[str] = []
        for value in args.values():
            if isinstance(value, str) and not re.fullmatch(r"\s*\d+\s*", value):
                words.extend(_stem(value))
        return name + "|" + " ".join(sorted(set(words)))

    def search_key(self, name: str, args: dict[str, Any]) -> str:
        """A search's words regardless of order, case and plural, with its other arguments (page, sort): two searches
        with the same key show the same rows."""
        words: list[str] = []
        rest = {}
        for key, value in sorted(args.items()):
            if isinstance(value, str) and len(value.split()) >= 1 and not re.fullmatch(r"\s*\d+\s*", value):
                words.extend(sorted(_stem(value)))
            else:
                rest[key] = value
        return name + "|" + " ".join(sorted(set(words))) + "|" + _dump(rest)

    def zero_rows(self, observation: Any) -> bool:
        """A search reply that holds a list and no listing in it (nothing matched)."""
        if not isinstance(observation, dict) or self.records_in(observation):
            return False
        lists = [v for v in observation.values() if isinstance(v, list)]
        return bool(lists) and all(not v for v in lists)

    def search_words(self, args: dict[str, Any]) -> int:
        return max((len(v.split()) for v in args.values() if isinstance(v, str)), default=0)

    # ---------- budget fit in store prices (M7, unit learned from the store's prices) ----------
    def price_scale(self) -> float:
        """How many store price units make one unit of the request's budget: 100 when, read as written, the budget
        affords almost none of this store's listings while a hundred times it affords most (a budget in dollars, prices
        in cents), else 1. Learned from the prices seen, never from a currency word."""
        if not isinstance(self.budget, (int, float)) or self.budget <= 0:
            return 1.0
        cat = self.category_key()
        mine = [i for i in self.rows if cat and any(self.label_fit(str((self.rows.get(i) or {}).get(cat) or ""), item) >= 2 for item in self.items)]
        prices = [p for p in (self.price_of(i) for i in (mine if len(mine) >= 4 else self.rows)) if p is not None and p > 0]
        if len(prices) < 4:
            return 1.0
        as_written = sum(1 for p in prices if p <= self.budget) / len(prices)
        reachable = sum(1 for p in prices if p <= self.budget * 100)
        middle = sorted(prices)[len(prices) // 2]
        return 100.0 if as_written < 0.05 and reachable >= max(2, len(prices) // 10) and middle > self.budget * 20 else 1.0

    def budget_hold(self) -> str:
        """Once per task version, before a checkout: the cart's own total (store prices times quantities) is over the
        budget in store units under the learned scale. Only when every line's price is known."""
        if self.budget_held == self.version or not isinstance(self.budget, (int, float)) or self.budget <= 0:
            return ""
        lines = {i: self.cart_lines.get(i, 1) for i in self.cart_ids()}
        prices = {i: self.price_of(i) for i in lines}
        if not lines or any(p is None for p in prices.values()):
            return ""
        limit = self.budget * self.price_scale()
        each = self.brief.get("budget_scope") == "each"
        over = [i for i in lines if prices[i] * (1 if each else lines[i]) > limit * 1.0001] if each else []
        total = sum(prices[i] * lines[i] for i in lines)
        if not over and (each or total <= limit * 1.0001):
            return ""
        self.budget_held = self.version
        self.world["budget_holds"] = self.world.get("budget_holds", 0) + 1
        self.trace.append({"kind": "budget_hold", "total": total, "limit": limit})
        if each:
            return (f"Not run yet (no step used). The budget is {self.budget:g} for each item ({limit:g} in the store's prices), and "
                    + ", ".join(f"{i} costs {prices[i]:g}" for i in over) + ": replace those lines with cheaper fitting listings, then check out.")
        return (f"Not run yet (no step used). The cart's total is {total:g} in the store's prices ("
                + " + ".join(f"{i} {prices[i]:g} x {lines[i]}" for i in lines)
                + f"), over the budget of {self.budget:g} ({limit:g} in the store's prices). Swap the dearest line for a cheaper fitting "
                "listing (or fewer units where the request leaves the count open), then check out.")

    # ---------- links between requested items (LINK_FIRST, DOUBT_HOLD, FIT_LINK) ----------
    def fit_links(self) -> list[tuple[int, int, str]]:
        """(dependent item, anchor item, requirement) for each requirement of one requested item that names another
        requested item of this order (the request reading keeps "fits / works with / matches the <other item>" as a
        requirement of the dependent item, in the request's words). Found by the anchor's name words, never by a word list."""
        out = []
        for a, item in enumerate(self.items, 1):
            for need in item.musts + item.told:
                if need in item.dropped:
                    continue
                tokens = _words(need.partition(":")[2]).replace(",", " ").split()
                # a fit rule ends with the other item's name and "in this order" (the request reading's own form); a rule
                # that only mentions the other item inside a condition is not a fit link
                if tokens[-3:] != ["in", "this", "order"]:
                    continue
                head = tokens[:-3]
                for b, other in enumerate(self.items, 1):
                    name = {w for w in _stem(other.name) if len(w) >= 3}
                    tail = _stem(" ".join(head[-(len(name) + 1):])) if name else frozenset()
                    if b != a and name and name <= tail:
                        out.append((a, b, need))
        return out

    def link_line(self) -> str:
        """LINK_FIRST: the order of work for a basket whose items depend on each other, and the chains still open."""
        links = self.fit_links()
        if not links:
            return ""
        lines = self.line_map()
        parts = []
        for a, b, need in links:
            if lines.get(a) and lines.get(b):
                continue
            parts.append(f"settle item {b} ({self.items[b - 1].name}) first, then choose item {a} ({self.items[a - 1].name}) against the listing "
                         f"chosen for item {b}, by both listings' own words (a model or series both name, a list of what it fits): {need}")
        return ("\n[order of work] " + " | ".join(parts[:2])) if parts else ""

    @staticmethod
    def _codes(record: dict[str, Any], skip: set[str]) -> set[str]:
        """The model, series and part codes a listing's own fields carry (tokens with letters and digits), its own id left out."""
        text = " ".join(str(v) for v in record.values() if isinstance(v, (str, int, float)) and not isinstance(v, bool))
        for v in record.values():
            if isinstance(v, dict):
                text += " " + " ".join(f"{k} {x}" for k, x in v.items())
        tokens = set(re.findall(r"[A-Za-z0-9][A-Za-z0-9-]{2,30}", text))
        return {t.lower() for t in tokens if re.search(r"\d", t) and re.search(r"[A-Za-z]", t) and t not in skip}

    def fit_hold(self) -> str:
        """FIT_LINK: once per task version, before a checkout: for each pair of cart lines where one requested item must fit
        another, the two listings' own fields share no model, series or part code. A fit shows in the listings' own words."""
        if not SWITCH["FIT_LINK"] or self.fit_held == self.version:
            return ""
        lines = self.line_map()
        notes = []
        for a, b, need in self.fit_links():
            la, lb = lines.get(a, []), lines.get(b, [])
            if len(la) != 1 or len(lb) != 1 or la[0] == lb[0]:
                continue
            ra, rb = self.full.get(la[0]) or self.rows.get(la[0]) or {}, self.full.get(lb[0]) or self.rows.get(lb[0]) or {}
            skip = {la[0], lb[0]}
            if not (self._codes(ra, skip) & self._codes(rb, skip)):
                self.fit_flagged.add(la[0])
                notes.append(f"item {a}'s line {la[0]} and item {b}'s line {lb[0]} share no model, series or part code ({need})")
        if not notes:
            return ""
        self.fit_held = self.version
        self.world["fit_holds"] = self.world.get("fit_holds", 0) + 1
        self.trace.append({"kind": "fit_hold", "notes": notes})
        return ("Not run yet (no step used). " + "; ".join(notes) + ". Check on both pages that the one fits the other; if it does not, "
                "switch to a listing that names the other, or switch the other item to one the first listing names. Then check out.")

    def doubt_hold(self) -> str:
        """DOUBT_HOLD: once per task version, an ending without a purchase is held while a requirement's value still sits
        on another listing that nobody has opened, or while an item must fit another and that other item has listings
        not tried yet."""
        if not SWITCH["DOUBT_HOLD"] or self.doubt_held == self.version or self.left() <= 5:
            return ""
        live = {need for _, need in self.needs_now()}
        pending = [need for _, need in self.refs if need in live and need not in self.ref_values]
        notes = []
        if pending:
            notes.append("a requirement takes its value from another listing that is not resolved yet (" + pending[0]
                         + "): find that listing, open it and every page it sends you to, and read the value there")
        cat = self.category_key()
        for a, b, need in self.fit_links():
            anchors = [i for i in self.rows if cat and self.label_fit(str((self.rows.get(i) or {}).get(cat) or ""), self.items[b - 1]) >= 2]
            if len(anchors) >= 2:
                notes.append(f"item {a} must fit item {b} ({need}): try another listing of item {b} that item {a}'s candidates name before deciding nothing fits")
                break
        if not notes:
            return ""
        self.doubt_held = self.version
        self.world["doubt_holds"] = self.world.get("doubt_holds", 0) + 1
        self.trace.append({"kind": "doubt_hold", "notes": notes})
        return "Not run yet (no step used). Before ending without a purchase: " + "; ".join(notes) + "."

    def open_shown(self) -> list[str]:
        """OPEN_SHOWN: once per task version, before a checkout: for a cart line whose own fields leave requirements
        unshown, the search rows of the same item that already show more of them are opened (two at most, one step
        each), so the last check judges them too. Returns what was read."""
        if not SWITCH["OPEN_SHOWN"] or self.shown_opened == self.version or self.left() < self.tail() + 4:
            return []
        self.shown_opened = self.version
        tool = self.facts_tool() or self.roles.get("open")
        target = self.single_target(tool)
        cat = self.category_key()
        if not tool or not target or not cat:
            return []
        lines = self.line_map()
        wanted: list[str] = []
        for number, item in enumerate(self.items, 1):
            needs = [m for m in item.musts + item.told if m not in item.dropped]
            for line in lines.get(number, [])[:1]:
                have = [self.need_status(n, line) for n in needs].count("proven")
                if have == len(needs):
                    continue
                for ident in self.rows:
                    if ident in self.opened or ident in self.full or ident in self.cart_ids() or not self.buyable(ident):
                        continue
                    if self.label_fit(str((self.rows.get(ident) or {}).get(cat) or ""), item) < 2:
                        continue
                    marks = [self.need_status(n, ident) for n in needs]
                    if "against" not in marks and marks.count("proven") > have and ident not in wanted:
                        wanted.append(ident)
        read = []
        for ident in wanted[:2]:
            faults: list[str] = []
            args = _shape({target: ident}, self.schemas.get(tool), "args", faults)
            if faults or self.left() < self.tail() + 3:
                break
            reply = self.act(tool, args)
            if reply is None or self.finished:
                break
            self.observe(tool, reply.get("observation"))
            read.append(f"{ident}: {self.record_of(ident)[:600]}")
        if read:
            self.world["shown_opened"] = self.world.get("shown_opened", 0) + len(read)
            self.trace.append({"kind": "open_shown", "ids": wanted[:2]})
        return read

    # ---------- basket coverage (M3/M4) ----------
    def search_item(self, args: dict[str, Any]) -> int | None:
        """The requested item a search is for: the one whose name, shelf and requirements share the most words with the
        search's words (none on a tie or no shared word)."""
        words: frozenset[str] = frozenset()
        for value in args.values():
            if isinstance(value, str):
                words |= _stem(value)
        if not words or len(self.items) < 2:
            return 1 if words and self.items else None
        scores = []
        for number, item in enumerate(self.items, 1):
            mine = _stem(item.name) | _stem(item.shelf or "") | frozenset(w for m in item.musts + item.told for w in _stem(m.partition(":")[2]))
            scores.append((len(words & mine), number))
        scores.sort(reverse=True)
        if scores[0][0] == 0 or (len(scores) > 1 and scores[0][0] == scores[1][0]):
            return None
        return scores[0][1]

    def every_item_seen(self) -> bool:
        """ASK_TIMING: every requested item has two or more listings of its category seen and one of them read in full,
        so each product's candidate properties exist."""
        cat = self.category_key()
        if not cat or not self.rows:
            return False
        for item in self.items:
            mine = [i for i in self.rows if self.label_fit(str((self.rows.get(i) or {}).get(cat) or ""), item) >= 2]
            if len(mine) < 2 or not any(i in self.full for i in mine):
                return False
        return True

    def coverage(self) -> list[tuple[int, int, int, bool]]:
        """Per requested item: (number, listings of its category seen, searches spent on it, has a cart line)."""
        cat = self.category_key()
        lines = self.line_map()
        out = []
        for number, item in enumerate(self.items, 1):
            seen = sum(1 for i in self.rows if cat and self.label_fit(str((self.rows.get(i) or {}).get(cat) or ""), item) >= 2) if cat else 0
            out.append((number, seen, self.item_searches.get(number, 0), bool(lines.get(number))))
        return out

    def coverage_line(self) -> str:
        """For a basket: which items have candidates, searches and a cart line, so the walker searches for the item that
        lacks listings instead of searching one item again and again."""
        if len(self.items) < 2 or not self.rows or not self.category_key():
            return ""
        parts = []
        for number, seen, searched, carted in self.coverage():
            parts.append(f"item {number}: {seen} listings seen, {searched} searches" + (", in the cart" if carted else ""))
        thin = [n for n, seen, _, carted in self.coverage() if not carted and seen == 0]
        text = "\n[basket] " + "; ".join(parts)
        if thin:
            text += ". Search next for: " + ", ".join(f"item {n} ({self.items[n - 1].name})" for n in thin)
        return text

    # ---------- tool roles checked against the tools themselves (M2) ----------
    @staticmethod
    def _types(spec: Any) -> set[str]:
        """The JSON types a parameter takes, including those listed under anyOf / oneOf."""
        if not isinstance(spec, dict):
            return set()
        found = set(_kinds(spec))
        for alt in (spec.get("anyOf") or []) + (spec.get("oneOf") or []):
            if isinstance(alt, dict):
                found |= set(_kinds(alt))
        return found

    def role_clash(self) -> str:
        """Why the tool roles read from the request cannot all be right, or "": a store may give a tool a name that
        suggests another job than the one it does, so the roles are checked against the tools' own parameters. One tool
        cannot both message the shopper and search, change the cart or finish; a search needs some text to search with;
        the search is not the cart."""
        roles = self.roles
        ask = roles.get("ask")
        if ask and ask in {roles.get(j) for j in ("search", "cart", "buy", "refuse", "cart_view")} - {None}:
            return f"{ask} is read as both the messaging tool and another job"
        search = roles.get("search")
        if search and not any("string" in self._types(v) for v in ((self.schemas.get(search) or {}).get("properties") or {}).values()):
            return f"{search} is read as the search but takes no text"
        if search and search == roles.get("cart"):
            return f"{search} is read as both the search and the cart"
        return ""

    def check_roles(self) -> None:
        """M2: when the roles clash, the stronger model reads the tools once more, told to judge each tool by its description
        and parameters only. Its roles are taken only when they name every job the first reading named, name a checkout,
        and do not clash; the checkout and ending modes are then read again for a tool that does both."""
        why = self.role_clash()
        if not why or self.no_credit:
            return
        lines = [f"- {t['function']['name']}: {t['function'].get('description', '')} | parameters: {_dump(t['function'].get('parameters') or {})}" for t in self.tools]
        prompt = (BRIEF + "\nJudge each tool only by its description and parameters: a tool's name may suggest another job than the one it does. "
                  f"A first reading was inconsistent: {why}.\nTOOLS:\n" + "\n".join(lines) + "\n\nREQUEST:\n" + self.request)
        got = self.ask_json(prompt, [HEAVY, LIGHT], 2500) or {}
        roles = got.get("roles") if isinstance(got.get("roles"), dict) else {}
        fresh = {k: v for k, v in roles.items() if isinstance(v, str) and v in self.schemas}
        before = dict(self.roles)
        self.roles = fresh
        shared = {j for j, v in before.items() if j != "ask" and v == before.get("ask")}
        if not fresh.get("buy") or not set(before) - shared <= set(fresh) or self.role_clash():
            self.roles = before
            self.world["role_check"] = "kept first reading: " + why
            return
        for key in ("buy_args", "refuse_args"):
            self.brief[key] = got[key] if isinstance(got.get(key), dict) else ({} if fresh.get(key[:-5]) != before.get(key[:-5]) else self.brief.get(key) or {})
        self.fix_finish_modes()
        self.world["role_check"] = "re-read: " + why
        self.trace.append({"kind": "role_check", "why": why, "roles": self.roles})

    def fix_finish_modes(self) -> None:
        """For one tool that both checks out and ends without a purchase, the arguments that choose each mode (as the
        request reading does)."""
        buy, refuse = self.roles.get("buy"), self.roles.get("refuse")
        brief = self.brief
        if buy and buy == refuse and (not brief.get("buy_args") or not brief.get("refuse_args") or brief["buy_args"] == brief["refuse_args"]):
            ask = (f"The store tool {buy} both checks out and ends without a purchase, depending on its arguments. Its schema: "
                   + _dump(self.schemas.get(buy)) + "\nReturn ONE JSON object and nothing else: {\"buy_args\": the fixed arguments that check out, \"refuse_args\": the fixed arguments that end without a purchase} (only the arguments that choose the mode).")
            fixed = self.ask_json(ask, [LIGHT, HEAVY], 600) or {}
            if isinstance(fixed.get("buy_args"), dict) and isinstance(fixed.get("refuse_args"), dict) and fixed["buy_args"] != fixed["refuse_args"]:
                brief["buy_args"], brief["refuse_args"] = fixed["buy_args"], fixed["refuse_args"]

    # ---------- situation and money per task type (M3 + M9) ----------
    def judge_situation(self) -> None:
        """After the request is read: the task's situation sets its money. One run key pays for every task and each task
        runs in a process of its own, so no task can see what the others spent; the run stays under its key only when
        every task's ceiling does. Money goes where passes are likelier: single-product tasks get more room, baskets of
        four or more items, which rarely pass, commit early on a small budget; the ceilings are set so that a full run of
        mixed tasks stays under one key."""
        count = len(self.items)
        if count <= 1:
            self.situation = "single"
            self.spend_soft, self.spend_hard, self.spend_stop, self.spend_light = 0.046, 0.060, 0.064, 0.032
        elif count <= 3:
            self.situation = "basket"
            self.spend_soft, self.spend_hard, self.spend_stop, self.spend_light = SPEND_SOFT, SPEND_HARD, SPEND_STOP, SPEND_LIGHT
        else:
            self.situation = "long basket"
            self.spend_soft, self.spend_hard, self.spend_stop, self.spend_light = 0.026, 0.035, 0.038, 0.018
        if self.process:
            self.situation += " + steps"
        if isinstance(self.budget, (int, float)) and self.budget > 0:
            self.situation += " + budget"
        self.world["situation"] = self.situation
        self.world["limits"] = {"steps": self.max_steps, "messages": self.msg_limit, "items": count}

    # ---------- basket lines against the request (M7) ----------
    def line_map(self) -> dict[int, list[str]]:
        """Cart lines per requested item, matched by the store's category label (the label is the item's shelf or name,
        or all its words are the item's); a line no item claims belongs to none."""
        cat = self.category_key()
        out: dict[int, list[str]] = {}
        if not cat:
            return out
        for ident in self.cart_ids():
            label = str((self.rows.get(ident) or {}).get(cat) or "")
            fits = [(self.label_fit(label, item), n) for n, item in enumerate(self.items, 1)]
            best = max(fits, default=(0, 0))
            if best[0] >= 2:
                for f, n in fits:
                    if f == best[0]:
                        out.setdefault(n, []).append(ident)
        return out

    def shared_line(self, ident: str, lines: dict[int, list[str]]) -> bool:
        return sum(1 for ids in lines.values() if ident in ids) > 1

    def lines_hold(self) -> str:
        """Once per task version, before a checkout: an item whose one matching cart line holds another count than the
        request's quantity, or an item of a basket with no line while listings of its category were seen (a basket with
        a wrong count or a missing line fails whole)."""
        if self.lines_held == self.version or not self.cart_ids():
            return ""
        cat = self.category_key()
        if not cat:
            return ""
        lines = self.line_map()
        notes = []
        claimed = [n for n in lines if lines[n]]
        used = {i for n in claimed for i in lines[n]}
        if len(self.items) > 1 and len(claimed) > len(used):
            notes.append(f"items {', '.join(str(n) for n in claimed)} share {len(used)} cart line(s): each requested item needs its own line")
        for number, item in enumerate(self.items, 1):
            mine = lines.get(number, [])
            if len(mine) == 1 and not self.shared_line(mine[0], lines):
                have = self.cart_lines.get(mine[0], 1)
                if have != item.qty:
                    notes.append(f"item {number} ({item.name}) is wanted {item.qty} times but its line {mine[0]} holds {have}")
            elif not mine and len(self.items) > 1:
                known = any(self.label_fit(str((self.rows.get(i) or {}).get(cat) or ""), item) >= 2 for i in self.rows)
                if known:
                    notes.append(f"item {number} ({item.name}) has no line in the cart")
        if not notes:
            return ""
        self.lines_held = self.version
        self.world["line_holds"] = self.world.get("line_holds", 0) + 1
        self.trace.append({"kind": "lines_hold", "notes": notes})
        return "Not run yet (no step used). " + "; ".join(notes) + ". Set the cart so every requested item has its line at the requested quantity, then check out."

    # ---------- own-field evidence and code picks (M4/M7) ----------
    def fields_of(self, ident: str) -> dict[str, Any]:
        """A listing's own fields, flat: its plain values and the entries of its property tables (the fullest record seen)."""
        record = self.full.get(ident) or self.rows.get(ident) or {}
        flat: dict[str, Any] = {}
        for key, value in record.items():
            if isinstance(value, dict):
                for inner, item in value.items():
                    if isinstance(item, (str, int, float)) and not isinstance(item, bool):
                        flat.setdefault(str(inner), item)
            elif isinstance(value, (str, int, float)) and not isinstance(value, bool):
                flat.setdefault(str(key), value)
        return flat

    def need_status(self, need: str, ident: str) -> str:
        """One requirement on one listing, from that listing's own fields only: 'proven' when a field naming the
        property holds every word and number of the wanted value, 'against' when that field is a short value sharing
        none of them, else 'unknown' (limits, ranges and wordings the code cannot settle are left to the model)."""
        prop, _, want = need.partition(":")
        want = want.strip().lower()
        if SWITCH["UNIT_JOIN"]:
            want = self.joined(want)
        words = [w for w in re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", want) if len(w) >= 3 or w[0].isdigit()]
        if not want or not words:
            return "unknown"
        named = _stem(prop)
        hits = [str(v).lower() for k, v in self.fields_of(ident).items() if _names(named, _stem(k)) and str(v).strip()]
        if SWITCH["UNIT_JOIN"]:
            hits = [self.joined(v) for v in hits]
        if not hits:
            return "unknown"
        numbers = [w for w in words if w[0].isdigit()]
        for value in hits:
            have = set(re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", value))
            have_numbers = {w for w in have if w[0].isdigit()}
            if numbers and have_numbers and not set(numbers) <= have_numbers:
                return "unknown"            # a different figure may still sit inside a stated limit
            if all(w in have or any((h.startswith(w) or w.startswith(h)) and min(len(h), len(w)) >= 4 and abs(len(h) - len(w)) <= 2
                                    for h in have) for w in words):
                return "proven"
        short = [v for v in hits if len(v.split()) <= 5]
        first = want.split()[0]
        # a short field sharing none of the wanted words is against the need; a value led by a short word (a negation or
        # a qualifier the code cannot read) stays unknown
        if short and not (len(want.split()) >= 2 and len(first) <= 3) and not any(w in v for v in short for w in words):
            return "against"
        return "unknown"

    @staticmethod
    def joined(text: str) -> str:
        """UNIT_JOIN: a figure and the unit written after it as one token ("16 gb" and "16gb" alike, "15.6 inch" and
        "15.6inch"), and the inch mark as part of its figure; generic, no unit list."""
        text = re.sub(r"(\d)\s*[\"″”]", r"\1in", text)
        return re.sub(r"(\d)\s+([a-z]{1,8})\b", r"\1\2", text)

    def code_picks(self) -> dict[int, tuple[str, int, int]]:
        """For each requested item, the listing its own fields fit best: candidates are the review's leads for that item,
        else the seen listings of the category that matches the item best; a listing that is sold out or has a field
        against a live requirement is out; then most requirements proven, then the lower price. Returns
        {item number: (listing id, proven, unknown)} for the items that have a candidate."""
        cat = self.category_key()
        picks: dict[int, tuple[str, int, int]] = {}
        for number, item in enumerate(self.items, 1):
            needs = [m for m in item.musts + item.told if m not in item.dropped]
            pool = [i for i in self.lead_items.get(str(number), []) if i in self.rows]
            if not pool and cat:
                fits = {i: self.label_fit(str((self.rows.get(i) or {}).get(cat) or ""), item) for i in self.rows}
                best = max(fits.values(), default=0)
                pool = [i for i, f in fits.items() if best > 0 and f == best]
            ranked = []
            for ident in pool:
                if not self.buyable(ident) or ident in self.unavailable:
                    continue
                marks = [self.need_status(n, ident) for n in needs]
                if "against" in marks:
                    continue
                price = self.price_of(ident)
                ranked.append(((-marks.count("proven"), price if price is not None else float("inf")), ident, marks.count("proven"), marks.count("unknown")))
            if ranked:
                ranked.sort(key=lambda r: r[0])
                picks[number] = (ranked[0][1], ranked[0][2], ranked[0][3])
        return picks

    def picks_text(self, picks: dict[int, tuple[str, int, int]]) -> str:
        parts = []
        for number, (ident, proven, unknown) in sorted(picks.items()):
            price = self.price_of(ident)
            parts.append(f"item {number}: {ident}" + (f" (price {price:g})" if price is not None else "") + f", {proven} requirement(s) shown by its own fields, {unknown} not shown")
        return "; ".join(parts)

    def commit_picks(self) -> bool:
        """M7 commit-best: the episode is ending without the walker having finished and the cart is empty. When the
        code's picks cover every item with at least one requirement shown each (or the item has none) and no field
        against one, Kyo carts them at the requested quantities instead of ending without a purchase (an ending without a
        purchase earns nothing when a fitting listing exists; a committed basket may)."""
        if self.completed or self.table_blind() or not self.roles.get("cart") or not self.rows or self.brief.get("pick") in ("most", "least"):
            return False
        picks = self.code_picks()
        if len(picks) < len(self.items) or len({p[0] for p in picks.values()}) < len(picks):
            return False
        for number, (ident, proven, unknown) in picks.items():
            item = self.items[number - 1]
            if not proven and [m for m in item.musts + item.told if m not in item.dropped]:
                return False
        lines = {ident: self.items[number - 1].qty for number, (ident, _, _) in picks.items()}
        if isinstance(self.budget, (int, float)) and self.budget > 0:
            prices = [self.price_of(i) for i in lines]
            limit = self.budget * self.price_scale() * 1.0001
            if any(p is None for p in prices):
                return False
            if self.brief.get("budget_scope") == "each":
                if any(p > limit for p in prices):
                    return False
            elif sum(p * n for p, n in zip(prices, lines.values())) > limit:
                return False
        calls = self.cart_calls(lines, [])
        if not calls or len(calls) + 1 > self.left():
            return False
        shaped = []
        for args in calls:
            faults: list[str] = []
            shaped.append(_shape(args, self.schemas.get(self.roles["cart"]), "args", faults))
            if faults:
                return False
        calls = shaped
        self.completed = True
        self.trace.append({"kind": "commit_picks", "lines": lines})
        self.world["commit_picks"] = True
        cart = self.roles["cart"]
        for args in calls:
            reply = self.act(cart, args)
            if reply is None or self.finished:
                self.commit_broken = True       # a half-set cart is never checked out
                return False
            self.book_cart(cart, args, reply.get("observation"))
            self.observe(cart, reply.get("observation"))
        if not set(lines) <= set(self.cart_ids()):
            self.commit_broken = True           # the store's cart does not hold every planned line: no checkout
            self.trace.append({"kind": "commit_unverified", "planned": lines, "cart": self.cart_ids()})
            return False
        return True

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
        # room for the light model's thinking: a reply cut at its limit would send the brief to the heavy model
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
        elif self.buys == 0 and len(self.items) > 1 and self.cart_touched and len(self.cart_ids()) < len(self.items) and self.step >= self.max_steps // 2:
            # M3 basket: a basket is bought whole or not at all; a missing line is filled with the closest fit
            text += f" The cart holds {len(self.cart_ids())} lines for {len(self.items)} requested items: give every item its line (the closest fit when none is perfect), then check out."
        secs = self.time_left()
        if secs < CLOSING + 50:
            text += f" Time is short: about {max(0, int(secs))} seconds are left for this task. Cart the fitting listings you have and check out now."
        live = {need for _, need in self.needs_now()}
        lookups = [need for _, need in self.refs if need in live and need not in self.ref_values]
        if lookups and self.buys == 0:
            text += "\n[resolve first: open the listing each of these points at, and every listing its page refers to; the last page gives the value] " + " | ".join(lookups[:3])
        if self.leads:
            text += "\n[review candidates] " + ", ".join(self.leads)
        text += self.coverage_line()
        if SWITCH["LINK_FIRST"]:
            text += self.link_line()
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
        if SWITCH["CAT_FROM_LABELS"]:
            found = self.category_by_labels()
            if found:
                return found
        return self.category_by_count()

    def category_by_labels(self) -> str | None:
        """CAT_FROM_LABELS: the field whose values are the store's category labels, read as the field whose values fit the
        requested items' shelf or name words (the count rule's candidate filter applies: on most rows, a few dozen values
        at most). On a tie, or when no field's values fit any item, the count rule decides."""
        counts: dict[str, set[str]] = {}
        present: dict[str, int] = {}
        for row in self.rows.values():
            for key, value in row.items():
                if isinstance(value, str) and 0 < len(value) <= 60:
                    counts.setdefault(key, set()).add(value)
                    present[key] = present.get(key, 0) + 1
        total = max(1, len(self.rows))
        scored = []
        for key, values in counts.items():
            if present[key] < 0.8 * total or len(values) > 40:
                continue
            hits = sum(1 for v in values if any(self.label_fit(v, item) >= 2 for item in self.items))
            if hits:
                scored.append(((hits / len(values), hits), key))
        if not scored:
            return None
        scored.sort(reverse=True)
        top = [k for sc, k in scored if sc == scored[0][0]]
        if len(top) > 1:
            fallback = self.category_by_count()
            return fallback if fallback in top else top[0]
        return top[0]

    def category_by_count(self) -> str | None:
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

    def wider_hold(self) -> str:
        """Once, before a cheapest checkout with many steps left: a store's keyword search reads titles, so fitting
        listings whose titles use other words are missed by searches for a requirement's value."""
        if self.wider_held or self.refusal_rounds > 0 or self.brief.get("pick") != "cheapest" or self.left() < 14 + 2 * len(self.items):
            return ""
        self.wider_held = True
        lines = [f"{i} at {self.price_of(i):g}" for i in self.cart_ids() if self.price_of(i) is not None]
        tries = []
        if SWITCH["CHEAP_SWEEP"]:
            # one value-plus-product search per stated value, from the request's own words (no word list)
            for item in self.items:
                noun = (item.name.split() or [""])[-1]
                for need in [m for m in item.musts + item.told if m not in item.dropped][:3]:
                    value = need.partition(":")[2].strip()
                    if value and len(value.split()) <= 3 and noun:
                        tries.append(f"{value} {noun}")
            if tries:
                self.world["sweep_searches"] = len(tries)
        return (f"Not run yet (no step used). {self.left()} steps are left and the pick is the cheapest. Searches match a listing's title,"
                " so a requirement's value (often only in the attributes) finds few of the fitting listings, and rows come in no price order."
                " Before checking out, run a few more searches for each item under other names for the product (its plain one-word name,"
                " synonyms, words from fitting titles you have seen) and look only at buyable rows priced below the cart's line"
                + (f" ({'; '.join(lines)})" if lines else "") + "; read the attributes of the cheapest such rows, swap in any that fits, then check out."
                + (" Searches to try, one each: " + "; ".join(f'"{t}"' for t in tries[:4]) + "." if tries else ""))

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
        pools: list[list[str]] = []
        for line in cart:
            price = self.price_of(line)
            kind = str((self.rows.get(line) or {}).get(cat_key) or "").strip().lower()
            if price is None or not kind:
                continue
            pool = [i for i in self.seen_order if i not in cart and i not in self.full and self.buyable(i)
                    and str((self.rows.get(i) or {}).get(cat_key) or "").strip().lower() == kind
                    and (self.price_of(i) if self.price_of(i) is not None else float("inf")) < price]
            pool.sort(key=lambda i: self.price_of(i) or 0.0)
            if SWITCH["CHEAP_SWEEP"]:
                pools.append([i for i in pool[:2] if i not in picks])
                continue
            picks += [i for i in pool[:2] if i not in picks]
        if SWITCH["CHEAP_SWEEP"]:
            # CHEAP_SWEEP: every line's cheaper rows in turn (line 1's cheapest, line 2's, ...), two per line; only the step
            # guard limits the total, so later lines are not starved
            for rank in range(2):
                for pool in pools:
                    if rank < len(pool) and pool[rank] not in picks:
                        picks.append(pool[rank])
            read = self.fill_facts(picks, self.tail() + 4)
            if read:
                self.world["sweep_reads"] = self.world.get("sweep_reads", 0) + len(read)
            return read
        return self.fill_facts(picks[:CHEAPER_OPENS], self.tail() + 4)

    # ---------- cheaper fits under a cheapest pick (CHEAP_SWEEP) ----------
    def sweep_key(self) -> str | None:
        """The field the sweep reads a listing's kind from: the one whose values are the store's category labels for the
        requested items, else the general category key."""
        return self.category_by_labels() or self.category_key()

    def line_item(self, line: str) -> int | None:
        """The requested item a cart line belongs to: the one item its category label fits best."""
        key = self.sweep_key()
        label = str((self.rows.get(line) or {}).get(key) or "") if key else ""
        fits = [(self.label_fit(label, item), n) for n, item in enumerate(self.items, 1)]
        best = max(fits, default=(0, 0))
        return best[1] if best[0] >= 2 and sum(1 for f, _ in fits if f == best[0]) == 1 else None

    def kind_of(self, ident: str) -> str:
        key = self.sweep_key()
        return str((self.rows.get(ident) or {}).get(key) or "").strip().lower() if key else ""

    def name_words_kept(self, old: str, new: str, number: int) -> bool:
        """The request's words for the item that the old line's record carries are on the new one too (a cheaper listing
        of the same category may be for another model)."""
        words = {w for w in _stem(self.items[number - 1].name) if len(w) >= 3}
        text = lambda i: _stem(" ".join(str(v) for v in (self.full.get(i) or self.rows.get(i) or {}).values() if isinstance(v, (str, int, float))))
        need = words & text(old)
        return bool(need) and need <= text(new)

    def sweep_line(self, line: str) -> str:
        """CHEAP_SWEEP: after a line is carted, its cheaper seen rows of the same kind are read in rising price, three at
        most; the first whose own fields prove every live requirement of the item, fail none and keep the item's name
        words replaces the line. Items tied to another by a fit rule, listings that left the cart in this task version and
        listings of a cart the last check rejected are never swept in. Returns what happened, for the walker."""
        key = self.sweep_key()
        kind_of = lambda i: str((self.rows.get(i) or {}).get(key) or "").strip().lower() if key else ""
        number = self.line_item(line)
        price = self.price_of(line)
        kind = kind_of(line)
        if number is None or price is None or not kind or (line, self.version) in self.swept:
            return ""
        self.swept.add((line, self.version))
        if any(number in (a, b) for a, b, _ in self.fit_links()):
            return ""
        item = self.items[number - 1]
        needs = [m for m in item.musts + item.told if m not in item.dropped]
        if not needs:
            return ""
        cart = self.cart_ids()
        lines_now = self.line_map()
        uncarted = sum(1 for n in range(1, len(self.items) + 1) if not lines_now.get(n))
        reserve = self.tail() + 6 + 4 * uncarted
        if not self.cart_calls({**{i: self.cart_lines.get(i, 1) for i in cart if i != line}}, [line]):
            return ""                      # the cart tool cannot take this line off: nothing to sweep
        pool = [i for i in self.seen_order if i not in cart and self.buyable(i) and kind_of(i) == kind
                and (i, self.version) not in self.left_cart and i not in self.rejected_ids
                and (self.price_of(i) if self.price_of(i) is not None else float("inf")) < price
                and "against" not in [self.need_status(n, i) for n in needs]]
        pool.sort(key=lambda i: self.price_of(i) or 0.0)
        tried = 0
        for ident in pool:
            if tried >= 3 or self.left() <= reserve:
                break
            tried += 1
            if ident not in self.full:
                self.fill_facts([ident], reserve)
            marks = [self.need_status(n, ident) for n in needs]
            if "against" in marks or any(x != "proven" for x in marks) or not self.name_words_kept(line, ident, number):
                continue
            new_price = self.price_of(ident)
            lines = {i: self.cart_lines.get(i, 1) for i in cart if i != line}
            lines[ident] = self.cart_lines.get(line, item.qty)
            calls = self.cart_calls(lines, [line])
            if not calls or len(calls) + reserve > self.left():
                break
            calls = calls[::-1] if len(calls) > 1 else calls      # the new line goes in before the old one comes off
            cartool = self.roles.get("cart") or ""
            for args in calls:
                faults: list[str] = []
                args = _shape(args, self.schemas.get(cartool), "args", faults)
                if faults:
                    return ""
                reply = self.act(cartool, args)
                if reply is None or self.finished:
                    return ""
                self.book_cart(cartool, args, reply.get("observation"))
                self.observe(cartool, reply.get("observation"))
            if ident not in self.cart_ids():
                self.trace.append({"kind": "sweep_unverified", "from": line, "to": ident})
                return f"[Kyo] Kyo tried to swap {line} for the cheaper {ident}, but the cart does not show {ident}: check the cart."
            self.world["sweep_swaps"] = self.world.get("sweep_swaps", 0) + 1
            self.trace.append({"kind": "sweep_swap", "from": line, "to": ident, "prices": [price, new_price]})
            return (f"[Kyo] Under the cheapest rule Kyo read {line}'s cheaper rows of the same kind and swapped in {ident} "
                    f"({new_price if new_price is not None else '?'} instead of {price:g}): its own fields show every requirement of item {number}.")
        if tried:
            self.trace.append({"kind": "sweep_read", "line": line, "tried": tried})
        return ""

    def fit_repair(self, a: str, b: str) -> list[str]:
        """The fit links a swap from line a to listing b repairs: a was flagged by the fit check, or b shares a model,
        series or part code with the other item's cart line while a shares none (a dearer listing that fits is right)."""
        if a in self.fit_flagged:
            return ["fit check flagged " + a]
        number = self.line_item(a)
        lines = self.line_map()
        out = []
        for dep, anchor, need in self.fit_links():
            other = anchor if dep == number else dep if anchor == number else None
            if other is None or len(lines.get(other, [])) != 1:
                continue
            lb = lines[other][0]
            rb = self.full.get(lb) or self.rows.get(lb) or {}
            codes = self._codes(rb, {lb, a, b})
            if codes & self._codes(self.full.get(b) or self.rows.get(b) or {}, {lb, a, b}) and not codes & self._codes(self.full.get(a) or self.rows.get(a) or {}, {lb, a, b}):
                out.append(need)
        return out

    def predict_cart(self, name: str, args: dict[str, Any]) -> dict[str, int]:
        """The cart a cart call would leave (a whole-cart list sets it; single lines add, a zero or removal mode takes off)."""
        spec = self.schemas.get(name, {}).get("properties") or {}
        removal = any(isinstance((spec.get(k) or {}).get("enum"), list) and len(spec[k]["enum"]) >= 2 and v in spec[k]["enum"][1:] for k, v in args.items())
        loose = next((_count(v) for v in args.values() if _count(v) is not None and not isinstance(v, str)), None)
        new = dict(self.cart_lines)
        whole = False
        lines: list[tuple[str, int | None]] = []
        for value in args.values():
            if isinstance(value, list):
                for line in value:
                    if isinstance(line, dict):
                        whole = True
                        ident = next((v for v in line.values() if isinstance(v, str) and v in self.rows), None)
                        if ident:
                            lines.append((ident, next((_count(v) for v in line.values() if _count(v) is not None and not isinstance(v, str)), 1)))
                    elif isinstance(line, str) and line in self.rows:
                        lines.append((line, loose))
            elif isinstance(value, str) and value in self.rows:
                lines.append((value, loose))
        if whole:
            return {i: c for i, c in lines if c and c > 0}
        for ident, count in lines:
            if removal or (count is not None and count <= 0):
                new.pop(ident, None)
            else:
                new[ident] = count or 1
        return new

    def price_guard(self, name: str, args: dict[str, Any]) -> str:
        """CHEAP_SWEEP: under a cheapest pick, a cart call that replaces a line with a dearer listing of the same kind is
        held once per pair, unless the dearer one fixes a requirement the cheaper one fails."""
        if not SWITCH["CHEAP_SWEEP"] or self.brief.get("pick") != "cheapest" or name != self.roles.get("cart"):
            return ""
        if self.verdicts.get(self.check_key(self.cart_ids())) is False:
            return ""                       # the last check rejected this cart: changing it is right
        now = set(self.cart_ids())
        new = self.predict_cart(name, args)
        added = [i for i in new if i not in now]
        removed = [i for i in now if i not in new]
        for b in added:
            for a in removed:
                if self.kind_of(a) != self.kind_of(b) or (a, b) in self.guard_refused or not self.buyable(a):
                    continue
                if self.line_item(a) is None or self.line_item(a) != self.line_item(b):
                    continue                # only one-for-one swaps within one requested item
                pa, pb = self.price_of(a), self.price_of(b)
                if pa is None or pb is None or pb <= pa:
                    continue
                number = self.line_item(a)
                needs = [m for m in self.items[number - 1].musts + self.items[number - 1].told if m not in self.items[number - 1].dropped] if number else []
                fixes = [n for n in needs if self.need_status(n, a) == "against" and self.need_status(n, b) != "against"]
                fixes += self.fit_repair(a, b)
                if fixes:
                    self.trace.append({"kind": "guard_pass", "from": a, "to": b, "fixes": fixes})
                    continue
                self.guard_refused.add((a, b))
                self.world["guard_refusals"] = self.world.get("guard_refusals", 0) + 1
                self.trace.append({"kind": "guard_refused", "from": a, "to": b, "prices": [pa, pb]})
                return (f"Not run (no step used): the pick is the cheapest, and {b} ({pb:g}) would replace {a} ({pa:g}), which costs less "
                        "and shows no failed requirement. Keep the cheaper line unless its own page shows it fails a requirement; "
                        "then send this call again.")
        return ""

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
        if self.ask_tries and len(self.full) <= self.ask_opened:
            return False            # after an empty try, ask again only once more listings are open
        if self.left() <= 8:
            return False
        if self.ask_wait:
            cat = self.category_key()
            ready = cat and all(any(i in self.full and self.label_fit(str((self.rows.get(i) or {}).get(cat) or ""), self.items[n - 1]) >= 2
                                    for i in self.rows) for n in self.ask_wait)
            if not ready:
                return False        # MSG_SHARE: the second round waits for the waiting products' listings
            self.ask_wait = []
            return True
        opened = len(self.full)
        if SWITCH["ASK_TIMING"] and self.every_item_seen():
            return True
        if opened >= 3 * len(self.items) + 1:
            return True
        # a slow task asks once some listings are open, before the question window closes
        if opened >= len(self.items) and DEADLINE - self.time_left() >= 70:
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
        # the interview counts as done only once a message went out, or when it can never run (no messages, no
        # messaging tool, no time left); a plan that found nothing to ask is tried once more later
        self.ask_tries += 1
        self.ask_opened = len(self.full)
        count = self.msgs_free()
        self.interview_msgs = count
        if count <= 0 or self.time_left() <= CLOSING + QUESTION_TIME or not self.roles.get("ask"):
            self.asked = True
            return ""
        if self.ask_tries >= 2:
            self.asked = True
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
        # more picks than messages: each item keeps a queue of its own, so a later turn has a next property to ask
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
            # one property per question: a label naming two is cut to the first
            label = re.split(r"\s+(?:and|or)\s+|[/,;(]", str(pick.get("label") or ""))[0].strip().rstrip("?.")
            if not label or len(label.split()) > 5:
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
            for name in self.item_keys(number, table, 8):
                if all(_words(name) != _words(other) for other in queue):
                    queue.append(name)
                    self.topup.add(name)
                    self.topup_item.setdefault(number, set()).add(name)
            if queue and number not in order:
                order.append(number)
        for number in list(queues):
            if 1 <= number <= len(self.items):
                queues[number] = self.rank_queue(number, queues[number], table)
        order = self.basket_order(order, table, count)
        self.trace.append({"kind": "plan", "ask": _clip(plan.get("ask"), 120), "queues": _clip(queues, 60)})
        sent: list[tuple[int, str, str]] = []
        told: list[str] = []
        found: set[int] = set()
        last_item: int | None = None
        while self.msgs_free() > 0 and not self.finished and self.time_left() > CLOSING + QUESTION_GAP:
            pick = self.next_pick(order, queues, found, last_item)
            waiting = [n for n in range(1, len(self.items) + 1) if not queues.get(n) and not any(k == n for k, _ in self.asked_labels)]
            if pick is None:
                if SWITCH["MSG_SHARE"] and waiting and self.ask_tries < 2:
                    self.ask_wait = waiting     # nothing left to ask now: the rest of the messages wait for those products
                break
            if SWITCH["MSG_SHARE"] and self.ask_tries < 2:
                # MSG_SHARE: products with nothing to ask yet (no listing of theirs read) keep their share of the messages
                # (first round only; the second round asks whoever has a property left)
                share = -(-max(1, self.interview_msgs) // len(self.items))
                if waiting and sum(1 for k, _ in self.asked_labels if k == pick[0]) >= share:
                    queues.setdefault(pick[0], []).insert(0, pick[1])
                    self.ask_wait = waiting
                    break
            number, label = pick
            room = self.list_extras if SWITCH["LEAD_FORM"] else ASK_EXTRAS
            rest = [x for x in queues.get(number, []) if _words(x) != _words(label)]
            if SWITCH["LEAD_ROTATE"]:
                # LEAD_ROTATE: a property listed twice after the lead without a reveal is left out of the tail
                rest = [x for x in rest if self.listed.get((number, _words(x)), 0) < 2]
            extra = [] if self.single_only or room <= 0 else rest[:room]
            for x in extra:
                self.listed[(number, _words(x))] = self.listed.get((number, _words(x)), 0) + 1
            text = self.question(number, label, extra)
            batch = self.send([(number, label, text)])
            if not batch:
                break
            last_item = number
            self.asked_labels.append((number, label))
            if self.msg_refused:
                # the store refused the message (its real limit is lower than the one read): no more questions
                self.msg_used -= 1
                self.msg_limit = self.msg_used
                break
            sent += batch
            # one answer: the light model reads it (the heavy one when the light one fails); an answer that wants a
            # single question to a one-property question is the shopper's way of not minding it: move on
            heard = self.hear(batch, (LIGHT, HEAVY))
            if SWITCH["REPLY_ADAPT"] and _items(heard.get("unclear")):
                # REPLY_ADAPT: the shopper did not follow: the next question names fewer properties
                self.list_extras = max(0, self.list_extras - 1)
                self.trace.append({"kind": "unclear", "q": text})
            if extra and _items(heard.get("crowded")):
                # the shopper wants one thing per message: single questions from now on, this property asked again alone
                self.single_only = True
                queues.setdefault(number, []).insert(0, label)
                self.asked_labels = [a for a in self.asked_labels if a != (number, label)]
                self.trace.append({"kind": "crowded", "q": text})
            before = len(told)
            self.take_heard(heard, told)
            if len(told) > before:
                found.add(number)
            if self.answer_class(heard, number) == "none" and extra:
                # a reply that names nothing: the properties listed after the first are asked again later, each named first
                # (a held property is revealed far more often when it is named first than when it comes later)
                self.trace.append({"kind": "none_answer", "item": number, "lead": label})
        if not sent:
            self.world["empty_interviews"] = self.world.get("empty_interviews", 0) + 1
            return ""
        self.asked = True
        if SWITCH["MSG_SHARE"] and self.ask_wait and self.msgs_free() > 0 and self.ask_tries < 2:
            self.asked = False                  # a second round once the waiting products have listings read
            self.trace.append({"kind": "ask_wait", "items": list(self.ask_wait)})
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
        if SWITCH["MSG_SHARE"] and len(live) > 1:
            # MSG_SHARE: no product takes a second message while another product with a property to ask has had fewer
            asked = {n: sum(1 for k, _ in self.asked_labels if k == n) for n in live}
            low = min(asked.values())
            pool = [n for n in pool if asked[n] == low] or [n for n in live if asked[n] == low]
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

    def store_name(self, number: int) -> str:
        """The requested item named in the store's own category words when a seen category label fits it, else in the
        request's words."""
        item = self.items[number - 1]
        cat = self.category_key()
        labels = {str((r or {}).get(cat) or "").strip() for r in self.rows.values()} if cat else set()
        mine = _stem(item.shelf or "") | _stem(item.name)
        ranked = []
        for label in labels:
            fit = self.label_fit(label, item)
            if not label or fit < 2:
                continue
            words = _stem(label)
            other_fit = max((self.label_fit(label, o) for n, o in enumerate(self.items, 1) if n != number), default=0)
            other_words = max((len(words & (_stem(o.shelf or "") | _stem(o.name))) for n, o in enumerate(self.items, 1) if n != number), default=0)
            if fit > other_fit or (fit == other_fit and len(words & mine) > other_words):
                ranked.append((fit, len(words & mine), -len(words), label))
        # a label that fits another requested item as well (a phone's label for a phone case) never names this one
        return sorted(ranked, reverse=True)[0][3].lower() if ranked else item.name

    def lead_question(self, number: int, label: str, extra: list[str] | None) -> str:
        """LEAD_FORM: one product per message, named in the store's category words, the likeliest property first."""
        names = [label] + [x for x in (extra or []) if x]
        said = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " or " + names[-1]
        return f"For the {self.store_name(number)}, what {said} would you like?"

    def question(self, number: int, label: str, extra: list[str] | None = None) -> str:
        """One question about one item: the chosen property first (a shopper who answers only the first property it
        knows answers that one), then up to two more of the item's queued properties when the world takes lists."""
        if SWITCH["LEAD_FORM"]:
            return self.lead_question(number, label, extra)
        names = [label] + [x for x in (extra or []) if x]
        said = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " or " + names[-1]
        return f"What {said} would you like for the {self.items[number - 1].name}?"

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
        if SWITCH["REPLY_ADAPT"]:
            prompt += ('\n\nAlso return "unclear": [numbers of the questions whose answer says the shopper did not follow the question or '
                       'wants it put more simply]. An answer that only says the shopper does not mind is not unclear.')
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
                elif isinstance(value, list) and any(isinstance(v, str) for v in value):
                    # a note slipped in among the listings of a result (a sentence, not a listing the reader left as text)
                    lines.extend(v for v in value if isinstance(v, str) and len(v.split()) >= 5 and not _unread(v))
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

    def check_cart(self, ids: list[str] | None = None, limit: float | None = None) -> str:
        limit = self.spend_soft if limit is None else limit
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
        if not self.verdicts[key]:
            self.rejected_ids.update(ids)
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
        while pending and self.trails < 6 and self.cost < self.spend_hard and self.time_left() > CLOSING:
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
        if self.cost > self.spend_soft:
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
        if self.audited or self.cost > self.spend_soft:
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
        self.check_cart(list(lines), self.spend_hard)
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
                self.check_cart(list(lines), self.spend_hard)
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
        if not self.text_last:
            # a cart the store sent as text is read when it is there, but an absent line list is no proof of "unchanged"
            self.cart_shows_lines = True
        return "synced"

    def book_cart(self, name: str, args: dict[str, Any], observation: Any) -> None:
        """Record a cart call: from the reply's own line list when it can be read; a store that lists the cart after
        every change it made changed nothing when a reply has no list of listings at all (a refusal); otherwise
        (no list, or one that cannot be read) Kyo's own arithmetic."""
        self.cart_touched = True
        self.cart_at = self.step
        self.searches_since_cart = 0
        self.free_pages = {}
        held_before = set(self.cart_ids())
        state = self.sync_cart(observation, name, args)
        self._note_left(held_before)
        if state == "synced":
            return
        if state == "none" and self.cart_shows_lines:
            self.trace.append({"kind": "cart_unchanged", "args": _clip(args, 60)})
            return
        self.note_cart(name, args)
        self._note_left(held_before)

    def _note_left(self, held_before: set[str]) -> None:
        for ident in held_before - set(self.cart_ids()):
            self.left_cart.add((ident, self.version))

    def show(self, observation: Any) -> str:
        # a reply holding labelled text the reader could not turn into records is shown whole (M1 fallback)
        text = _dump(_clip(observation, SHOW_CHARS if not self.raw_last else 4000))
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
            if not ending and self.is_search(name):
                key = self.search_key(name, args)
                seen = self.search_keys.get(key)
                if seen and seen[0] == self.version and mark not in self.recent:
                    self.world["same_searches"] = self.world.get("same_searches", 0) + 1
                    self.same_run += 1
                    out.append((cid, f"Not run (no step used): a search with these same words (in another order or form) ran at step {seen[1]} and "
                                "nothing has changed since; its rows are above. Use other words (the product's plain name, one value), or browse the item's category."))
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
                    over = self.lines_hold() or self.budget_hold() or self.fit_hold()
                    if over:
                        out.append((cid, over + "\nCart now: " + self.cart_text()))
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
                    read = self.fill_facts(self.cart_ids(), self.tail() + 3) + self.cheaper_rows() + self.open_shown()
                    verdict = self.check_cart()
                    if verdict:
                        seen = ("Kyo read these listings' attributes for the check (one step each):\n" + "\n".join(read) + "\n") if read else ""
                        out.append((cid, "Not run yet. " + chain + seen + verdict + "\nCart now: " + self.cart_text()))
                        continue
                    if reread and self.cost > self.spend_soft:
                        out.append((cid, "Not run yet (no step used). Kyo re-read the cart's listings just now:\n" + reread + "\nCheck out again when every line still shows stock and a price inside the budget; otherwise replace that line first."))
                        continue
                elif self.time_left() > CLOSING + 15:
                    # a held ending needs one more model turn to act on the reason; without that time the decline goes through
                    if self.left() > 5:
                        doubt = self.doubt_hold()
                        if doubt:
                            out.append((cid, doubt))
                            continue
                        review = self.audit()
                        if review:
                            out.append((cid, "Not run yet (no step used). " + review))
                            continue
                        if not self.leads and not self.table_blind():
                            picks = self.code_picks()
                            if len(picks) == len(self.items) and all(p[1] > 0 for p in picks.values()):
                                self.leads = [p[0] for p in picks.values()]
                                self.lead_items = {str(n): [p[0]] for n, p in picks.items()}
                                self.trace.append({"kind": "code_leads", "leads": list(self.leads)})
                        if self.leads and self.left() > 4 and self.refuse_blocks == 0:
                            self.refuse_blocks = 1
                            self.trace.append({"kind": "refuse_blocked", "leads": list(self.leads)})
                            out.append((cid, "Not run yet (no step used). The review found listings that appear to fit: " + ", ".join(self.leads) + ". Open them if needed, put the best fitting one in the cart (respecting the pick rule) and check out. End without a purchase only if each of them clearly fails a requirement; then say which one in your text and send the ending call again."))
                            continue
                        if self.leads and self.left() > 4 and self.refuse_blocks == 1:
                            self.refuse_blocks = 2
                            first = self.leads[0]
                            self.check_cart([first], self.spend_hard)
                            if self.verdicts.get(self.check_key([first])) and first in self.leads:
                                self.confirmed = first
                                self.trace.append({"kind": "refuse_blocked", "confirmed": first})
                                out.append((cid, f"Not run yet (no step used). An independent check of {first}'s records says it meets every requirement. Add it to the cart at the requested quantity and check out; end without a purchase only if its record clearly breaks a requirement."))
                                continue
            if self.is_refuse(name, args) and self.left() <= 5 and not self.late_held and self.time_left() > CLOSING + 15:
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
            if (self.is_refuse(name, args) and not self.leads and not self.pushed and self.left() > self.max_steps // 3 and self.time_left() > CLOSING + 15
                    and not self.table_blind() and self.searches_since_cart < self.search_cap()):
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
            held = self.search_hold(name, args) or self.price_guard(name, args)
            if held:
                out.append((cid, held))
                continue
            self.same_run = 0
            paging = self.is_search(name) and self.is_paging(name, args)
            if self.is_search(name):
                key = self.words_key(name, args)
                if paging:
                    # the next page of the same search is not a new search: a store with few rows a page is read deep
                    self.free_pages[key] = self.free_pages.get(key, 0) + 1
                else:
                    self.searches_since_cart += 1
                self.last_search_words = key
                self.last_search = (name, dict(args))
                self.last_rows = False          # set again once this search's reply shows listings
                target_item = self.search_item(args)
                if target_item:
                    self.item_searches[target_item] = self.item_searches.get(target_item, 0) + 1
            reply = self.act(name, args)
            raw_now = self.raw_last
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
            if self.is_search(name):
                self.last_count = len(self.records_in(observation))
                self.last_rows = self.last_count > 0
                if not self.price_key and name not in self.shapes and (self.budget or len(self.items) > 1):
                    # prices are read for the budget and line checks too, not only for a cheapest pick
                    self.read_shape(name, observation)
                self.search_keys[self.search_key(name, args)] = (self.version, self.step)
                if self.zero_rows(observation) and self.search_words(args) >= 3:
                    self.zero_hits += 1
                    self.world["zero_hit_long"] = self.zero_hits
                    trimmed += ("\n[Kyo] No listing matched all of these words. This store's search seems to need every word in a listing: "
                                "search with one or two words (the product's plain name, or a single value) and judge the values on the listings' own fields.")
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
                before_lines = set(self.cart_ids())
                self.book_cart(name, args, observation)
                if SWITCH["CHEAP_SWEEP"] and self.brief.get("pick") == "cheapest":
                    for line in [i for i in self.cart_ids() if i not in before_lines]:
                        swept = self.sweep_line(line)
                        if swept:
                            trimmed += "\n" + swept
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
            self.raw_last = raw_now
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
        """End the episode: check out a held cart, else end without a purchase. With `again`, a refused checkout is
        returned (as the store's reply) for the agent to deal with instead of being followed by an ending without a purchase."""
        if self.finished or self.broke or self.left() <= 0:
            return ""
        buy, refuse = self.roles.get("buy"), self.roles.get("refuse")
        fresh = self.cart_view is not None and self.cart_seen_at >= self.cart_at
        holding = bool(self.cart_ids()) if fresh else bool(self.cart_lines)
        if buy and not holding and not self.cart_touched and self.left() >= len(self.items) + 1 and self.commit_picks():
            fresh = self.cart_view is not None and self.cart_seen_at >= self.cart_at
            holding = bool(self.cart_ids()) if fresh else bool(self.cart_lines)
        if buy and not self.commit_broken and (holding or (self.cart_touched and not fresh and self.left() >= 2)):
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
                    return "the checkout was refused: " + self.show(reply.get("observation"))
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
        self.check_roles()
        self.judge_situation()
        self.trace.append({"kind": "brief", "brief": self.brief})
        opening = self.view.get("opening")
        if opening is not None:
            try:
                opening = self.read_reply("opening", opening)
            except Exception as exc:
                self.trace.append({"kind": "reader_error", "error": f"{type(exc).__name__}: {exc}"[:200]})
            self.absorb(opening)
        intro = "SHOPPER'S REQUEST:\n" + self.request
        if opening is not None:
            intro += "\n\nWHAT THE STORE SHOWS FIRST:\n" + self.show(opening)
        intro += "\n\nStart by searching for each item in its category and opening several listings in full, so the interviewer can see which properties these listings carry.\n\n" + self.status()
        self.chat = [{"role": "system", "content": GUIDE}, {"role": "user", "content": intro}]
        self.set_caps()
        self.work(self.loop_cap, self.spend_hard)
        # M8: the store may turn the ending down (a new budget, a sold-out line, a dropped or new requirement): up to three
        # short rounds apply the change and finish again, each while steps, time and money last
        while not self.finished and self.refusal_rounds < 3:
            turned = self.finish(again=True)
            if not (turned and not self.finished and not self.broke and not self.no_credit and self.left() > 1 and self.time_left() > CLOSING
                    and self.cost < self.spend_stop - 0.002):
                break
            self.refusal_rounds += 1
            self.rearm()
            self.wider_held = True          # a round applies the change; it does not widen the search again
            before = (self.cart_mark(), self.version, self.step)
            self.chat.append({"role": "user", "content": "[Kyo] Kyo tried to end the episode and " + turned + "\n" + self.refusal_brief()
                              + "\nApply what the store says: replace a line that is gone, keep the order inside a new budget, drop what the shopper gave up, "
                              "then finish. Keep every other cart line as it is.\n" + self.status()})
            self.work(self.loop_calls + 10, self.spend_stop)
            if (self.cart_mark(), self.version, self.step) == before:
                break                       # the round changed nothing: checking out the same refused cart again is no use
        if not self.finished:
            self.finish()
        self.world.update({"reader": dict(self.read_stats), "single_only": self.single_only, "refusal_rounds": self.refusal_rounds})
        self.trace.append({"kind": "world", "world": self.world})
        self.trace.append({"kind": "end", "finished": self.finished, "steps": self.step, "cost": round(self.cost, 5), "calls": self.calls})
        return self.trace

    def work(self, loop_cap: int, spend_cap: float) -> None:
        idle = 0
        while (not self.finished and not self.broke and self.left() > 0 and self.loop_calls < loop_cap and self.cost < spend_cap
               and self.time_left() > CLOSING and self.same_run < 6):
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
                if self.no_credit or self.calls >= self.call_cap or self.misses > 2 or self.out_of_time or self.cost >= self.spend_stop:
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
            if (not self.asked and self.roles.get("ask") and self.msgs_free() > 0 and any(c["function"]["name"] in committing for c in calls) and self.left() > 8
                    and not (self.ask_tries and len(self.full) <= self.ask_opened)):
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
