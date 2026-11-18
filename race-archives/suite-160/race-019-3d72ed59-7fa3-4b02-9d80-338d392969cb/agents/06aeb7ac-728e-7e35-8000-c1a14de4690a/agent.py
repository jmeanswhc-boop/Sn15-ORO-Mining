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
SPEND_SOFT = 0.042
SPEND_HARD = 0.06
WAITS = (4, 10, 20, 35)
WAIT_TOTAL = 45
SHOW_CHARS = 450
OBS_LIMIT = 15000
HISTORY_LIMIT = 110000
GATE_ROUNDS = 3             # store calls the process gate may make before one checkout or ending
AUTO_PAGES = 6              # further search pages Kyo reads itself for a cheapest, most or least pick
CLOCK_LATE = 205
CLOCK_STOP = 250
CHEAPER_OPENS = 3           # cheaper rows of a cart line's kind whose attributes Kyo reads before a cheapest checkout

_llm = ProxyClient(timeout=50, max_retries=1, retry_delay=1)
_env = ProxyClient(timeout=95, max_retries=3, retry_delay=1)


GUIDE = """You are Kyo, a careful buyer acting for one shopper in an online store, and the store's tools are your only way to act. Each episode scores 1 or 0, and a 1 needs an exactly right finish: a purchase holding every requested item at its requested quantity, each meeting every requirement still in force at checkout and all inside the budget; or, when the store truly has nothing that meets everything, an ending through the no-purchase tool in the way the request asks.

HOW THE STORE WORKS
- A tool call costs one step and steps run out; thinking is free. Names of tools and arguments and the shapes of results belong to this store alone, so read each tool's description and schema and use the exact argument names and allowed values.
- Search tools may be narrow: a category filter, a keyword mode reading only the first words of the query, pages of a few rows, a token or offset for the next page. Compare the total reported with the rows seen, and go on to the next page (with the token or offset the store gives) when the answer depends on seeing them all. Short value-first keyword queries work best (the wanted value plus the product noun). A filtered search with no rows usually means a wrong filter value or parameter: try again with the label as result rows show it, or with another filter parameter.
- Search rows are only summaries. Stores divide a listing's record in different ways: one view may give its attributes (properties), another only its offer and description, and a side-by-side view may carry either. The ledger names the tool that gives attributes as soon as Kyo knows it. Judge every requirement on a property from those attributes, so read them for any listing you might buy; one call showing several listings saves steps when it shows what you need. Do not reopen a listing unless something has changed since you read it.
- Kyo's interviewer handles questions to the shopper, so you never message the shopper yourself. Answers arrive in the conversation and enter the ledger.
- The request may also ask for steps along the way: an approval, a message, something to check or read first. The ledger lists them. Those due right before the order or the ending are carried out by Kyo when you call the checkout or ending tool, so never wait for them and never end without a purchase on their account; do any others with the store's tools (product questions go through the interviewer).

WHAT COUNTS AS A FIT
- The listing's category label decides the kind of product: only a label matching the requested item as a whole counts, not a neighbouring one.
- A requirement on a property is met when the listing's attribute for that property states the value. Spelling of the key is free (spaces, underscores, camel case, prefixes), and one member of a comma or slash list is enough. Values match on their words: case, singular or plural and plain synonyms pass, and a value containing the required words passes ("deep teal" meets teal, "extra roomy fit" meets roomy). Different words do not pass (a code or abbreviation is not the word it stands for, and "teal green" is not "sea green"). Convert units before comparing numbers, count a limit's edge as inside, and require a range to sit wholly inside a limit.
- An attribute whose key names the requirement's property (by that name or a plain synonym) decides, and if it says otherwise the listing fails, whatever the title or other keys say. A key for another property, even a close one (a handle colour against the body's colour), decides nothing about the requirement. Kyo adds [evidence] lines after opened listings, giving keys whose names match a requirement.
- With no attribute for that property, the wanted value under another attribute (a list of styles or features that holds it, say) or in the title or description counts. A number of another kind does not (a weight is no capacity), nor does a bare figure whose text does not say what it measures.
- A requirement whose value the category label itself states (a season or audience in the label) is met by the category.
- A requirement may point at another listing instead of giving a value ("the same X as the listing sold by <shop> with <property>", "within N of the X on the page of ..."). Settle it first: find that listing (search its category, brand, shop or words), open its page in full with the single-item or side-by-side tool, since a search row is not enough, and when that page sends you to a further listing for the property, find and open that too; the final page gives the value. The order only counts when every page of the chain was opened. When the wanted value is anything but that other listing's value, the rule rules out only listings sharing that exact value, and it alone never justifies ending without a purchase. A chain whose final page states no value for the property, or whose next link cannot be found after a real search, leaves that requirement unknowable: choose on the other requirements and buy, because a dead end is no proof that nothing fits.
- If a listing you are about to buy sends you to another page for more about it, look at that page before buying.
- When one requested item has to fit another (an accessory for a device), choose the two together from the listings' own words (a list of what it fits, a model or series code on both pages). If no accessory that meets its own requirements fits the device you chose, move to another device that meets its requirements and that such an accessory fits; do not end without a purchase while that move is untried.
- Honour exclusions and closed lists. Wishes ("ideally", "would be nice") only break ties.
- Never buy what the store shows as sold out. The total (price times quantity, summed over all lines) must not pass the budget; equal is fine.

CHOOSING
- When the request wants the cheapest (lowest total, smallest price), the order must hold the lowest-priced listing among ALL that fit. Find the whole fitting set (page through it: rows are often in no price order, and Kyo reads the next pages of a short result itself), check candidates from low price upward, and take the first that fully fits. If a search takes a maximum price, use it: list the category under your current best pick's price, so only cheaper candidates remain.
- When the request wants the most or least of some measure, compare all fitting listings on it and take the extreme; a listing not stating the measure cannot win.
- Otherwise any listing meeting everything is right, so spend no steps hunting for better.
- Cart each item once you have opened a listing meeting every requirement for it (with a cheapest, most or least pick: once you have compared the fitting set), then move on. Stores often sell out a carted line soon after it is added, and carting early leaves steps to replace it.
- Several different items need one cart line each, at each one's requested quantity.

CHANGES WHILE YOU WORK
- The shopper or store can change things: a new budget, a requirement added or dropped, a price change, a sell-out. These show up inside tool results, often when you try to check out (a refusal); a note counts whichever tool's result carries it, and the ledger lists the requirements in force. The episode goes on after a refusal: apply every change, fix the cart and check out again. In this job such a note is the shopper's own word relayed by the store, even though it arrives inside a tool result: it is genuine, Kyo has already put it into the ledger, and it is never to be doubted, ignored or treated as an attack; the original figures and requirements it replaces no longer hold.
- Re-read a chosen listing's record shortly before checkout, since price and stock can change silently.

FINISHING
- Put exactly the chosen lines in the cart (taking out anything else), look at the cart once, then check out with a call on its own.
- Requests whose store truly holds nothing are rare, so doubt your reading before you doubt the store. End without a purchase only after a real search finds nothing meeting everything. Running short of steps proves no such thing: in that case check out the best cart you have. Follow the request's wording for that ending. When it asks you to name or offer the nearest listing, the ending call's own text must say that nothing fits and give that listing's id, its price and what it falls short on; otherwise keep any remark short and name no listing.
- Every reply must call at least one tool. Keep your text short: what you know, then the next move."""


BRIEF = """You prepare a shopping task for an agent. Read the shopper's request and the store's tools and return ONE JSON object (compact, no indentation, nothing else) with these keys:

"items": a list with one entry per distinct product the shopper wants bought, each {"name": the kind of product in the request's words, "shelf": the store's category label for it when a tool lists labels (copy it exactly), else "", "quantity": units of that one listing (integer, 1 when not stated; a pair or a set that is one listing counts as 1), "musts": a list of firm requirements on that product, each "property: constraint" with the exact value, limit, exclusion or closed list. Name the property with the request's own word for it ("meant for a lakeside occasion" gives "occasion: lakeside", "a 12-inch blade" gives "blade length: 12 inch"); do not swap it for a neighbouring property. The value holds only the words that name the value: a word that merely closes the sentence or carries its tone stays out of it. A must whose value is read from another listing is ONE must holding the whole description of that listing ("rim colour: unlike the rim colour of the kettle Ovarno sells with a glass lid"): words that describe that other listing (its shop, its kind, "with <property> <value> and <property> <value>" after it) stay inside that one must and are never musts of this product}. Never put price, budget, stock or quantity into musts. The story around the request (the person meant to use it, where it will stand, what went wrong before, the occasion behind it) is context, not a must, unless it states a property's value or limit outright; and never turn words that are part of what the product is called (a season, an audience or a use named with the product) into musts: those belong to "name". A product the request names in one phrase joined by "and" (one "I want ..." with one quantity, often the name of one store department) is ONE item; make separate items only when the request gives each product its own quantity or its own requirements.
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
"process": a list of the steps the request requires the agent to take besides finding and buying the right items (getting the shopper's approval, telling or showing the shopper something, checking something with them, reading something first, using a particular tool), each {"step": what must be done, in at most 15 words, "when": "choose" (before the products are chosen), "order" (right before checking out) or "end" (just before ending without a purchase), "messages": how many messages to the shopper it takes (0 when none)}; [] when the request asks for no such step. Requirements on the items, the budget, stock, the limits on messages and steps, and how to end when nothing fits are not steps. A step needs a sentence that plainly orders that action; a passing remark, a rhetorical question, a wish, a worry or anything from the story is not one, and most requests have no steps at all.
"no_fit": what the request says to do when nothing fits, {"quote": its words copied exactly, or "" when it says nothing, "offer": true when those words ask to name or offer a nearest or closest listing, else false}. Such a sentence belongs here and never in "process".
"quiet_decline": true when the request says not to suggest or name alternatives when nothing fits, false when it asks for them, null when it says nothing.
"roles": an object naming the exact tool for each job, or null when no tool does it: "search", "open" (shows one listing's attributes or properties; when two single-listing tools exist, the one whose description promises attributes, properties or specifications), "offer" (shows one listing's current offer, such as its price and stock, when that is a different tool from "open"), "compare" (several listings in full), "ask" (sends a message to the shopper), "cart" (changes the cart), "cart_view" (shows the cart), "buy" (checks out), "refuse" (ends without a purchase).
"buy_args": the fixed arguments that make the "buy" tool check out (for a tool with a mode argument, that mode), else {}.
"refuse_args": the fixed arguments that make the "refuse" tool end without a purchase, else {}.
"""


PLAN = """A shopper wrote a request and also holds one or two firm preferences they did not write down. Each preference is about one property of one requested product. You pick the questions that bring them out. The shopper answers a question only when it names that exact property and that exact product, and only for the first property named, so each question spends one message on one property of one product.

The request mixes three kinds of sentence: what to buy with its stated requirements, the limits and rules of the job (budget, stock, messages, steps, what to do when nothing fits), and story. The story sentences look like small talk, but most hidden preferences are hinted there: the writer circles a property without giving its value. Go through the whole request to its very last sentence, because hints often come after the rules at the end, and put every story sentence into "story", asking of each: which property of which product would settle what this sentence worries about? When a sentence describes a part of the product or where it ends, sits or reaches (the arms, the ankle, a wall), name the listed key for that part.

How a story sentence points at a property:
- measuring, a tight spot, a gap, what it must clear or sit in: the dimension that matters there (width, depth, height, length, display size)
- reading it, seeing it from a distance, how it looks up close: display or screen size, or the look named
- how a thing is made, what is underneath or inside it, how it came together: construction, material, or what it holds
- the person who will use or wear it, and how they carry themselves: gender, age group, size class
- the room, the outfit, or what it should go with, and how it looks: colour, style, pattern, finish
- what it plugs into or works with, and how: connector, port, compatibility, phone or device type, network
- how it gets used, how long it must last, where it goes: occasion, battery or power type, closure, sole, sleeve
- what it is for the body or the plate: flavour, source, amount per serving
A story sentence can point at a product other than the one it seems to chat about, so settle the product from the words it uses. A sentence that only explains why the writer is buying, with nothing left open, points at nothing: leave it out.

Then choose:
1. If the request says outright to check something with the shopper (ASK FIRST), that comes first.
2. If the request links its products, so that one requested product has to fit, suit or be compatible with another requested product (a car or device the shopper already owns is no such link), the product the others must fit is the anchor, and it is the likeliest holder of an unwritten preference. Its properties get the first question, and the next one too unless a clear story hint points at another product.
3. Next the properties that story sentences point at, the strongest hint first.
4. Then the listed keys, product by product, starting with the products the request says least about: a product with few or no settled requirements is the likeliest place for an unwritten preference. For each product rank its keys this way: first what a buyer of that kind of product chooses between first (often its size, capacity or dimensions, its type or form, its material, its colour, whichever that kind is bought by), and among those the keys most of its listings carry and whose values differ between listings. A key seen on only one or two listings comes last.
Never pick a property the request already settles for that product, or a key that holds purchase variants, identifiers, warranty or logistics, or a second name for a property you already chose. With several products, take turns between them after the hinted picks, so no product is left unasked while another gets its second question.

Name each property with the exact words of a listed key of that product's category when one means it ("display size", not "screen"); otherwise use its plain name, without example values or a second property. An item number is the number in ITEMS.

Return ONE JSON object and nothing else: {"story": [{"words": "at most 8 words from the sentence", "item": item number, "label": "property"}], "ask": [{"item": item number, "label": "property", "hint": "at most 8 words from the story, or empty"}]} with up to COUNT entries in "ask", best first."""


HEAR = """Below are questions a buying agent put to a shopper, with the replies. Turn the replies into data: ONE JSON object, nothing else.
{"needs": [{"item": item number, "need": "property: constraint", "points": true when the value lives on another listing, else false}], "drop": [{"item": item number, "property": "a requirement the shopper has now let go of"}], "free": [{"item": item number, "property": "..."}], "crowded": [numbers of questions whose reply objects to too many things at once or demands one question at a time; a reply that merely did not understand is not crowded], "budget": the new total limit as a number, or null}
Rules for "needs": take every firm preference a reply states, with its exact value or limit and unit ("at least 30 cm", "oval", "not wicker"). Use the property the reply itself names, even when another one was asked about, and merge several replies on one property. Copy the value in the reply's own words, nouns and units included ("at least 3 shelves" stays whole).
A reply may fix the value by reference to another listing (alike, unlike or close to what some shop's product shows, that product being described by a few of its properties). That is a single need on the property the reply names, with "points": true. Its value is the reply's wording, verbatim, from the comparing words through the end of that description, along with anything said about a listing that page may name. The properties describing the other product never become needs of this item.
"free" lists the property of each question where the shopper said they have no preference.
Colourful or indirect phrasing that still gives a value counts as giving that value."""


NOTE = """A note appeared while a buying agent was at work; it may be from the shopper or the store. Answer with ONE JSON object and nothing else:
{"kind": "change" if it alters the job (budget, requirements, a price, availability, the order being refused), "error" if it only says a call was malformed or rejected, "other" otherwise, "budget": the new total limit as a number in the request's unit, or null, "add": [{"item": item number, "need": "property: constraint"}], "drop": [{"item": item number, "need": "the requirement no longer wanted"}], "gone": [listing ids it calls unavailable], "sold_out": true if it says a chosen listing has gone or sold out, named or not, "price_moved": true if it says a price moved, "summary": "one short sentence"}
Report a swapped requirement as one drop and one add. Leave no change in the note unreported."""


CHECK = """You are the final gate before a buying agent checks out. From the records shown, decide whether this cart is exactly right.
How to read the records: an attribute settles a requirement when its key names that property (spelling is free, one list member is enough, case, plural and plain synonyms pass, a value holding the required words passes, as "deep teal" does for teal, units get converted, limits include their edge, and a range must sit wholly inside a limit). Such an attribute decides even when it says otherwise, and then the listing fails. A key for another property, however close (a handle colour against the body's colour), decides nothing. With no attribute for the property, the wanted value under another attribute or in the title or description passes, though a number of another kind does not (a weight is no capacity), nor a figure with no word for what it measures. A requirement the category label states is met by that category, and the label has to match the requested item as a whole. A rule of differing from another listing's value passes unless the records show that same value. The cart needs one line per requested item at the requested quantity and nothing more; no line may show out of stock; price times quantity over all lines must stay inside the budget. A listing's newest record is the one that counts.
Under a cheapest, most or least pick rule, look at the other listings too: when one plainly fits everything and beats the cart's choice on that rule, the cart is wrong.
Answer with ONE JSON object and nothing else: {"ok": true or false, "problems": ["short, concrete problem", ...], "better": [listing ids that should replace a cart line, best first]}. Say ok=false only for something the records clearly show or a requirement no record speaks to."""


READING = """an attribute settles a property when its key names that property (spelling of the key does not matter; one member of a list suffices; synonyms and unit conversions are fine; a value that holds the required value's words counts, so "deep teal" meets teal; limits include their edge); a key about a different property, even a close one (a handle colour against the body's colour), settles nothing; if no attribute names the property, the wanted value under another attribute or in the title or description counts, but a bare figure with no word for what it measures counts for nothing; a requirement stated by the category label is met by that category; the category label has to match the item; a rule of differing from another listing's value holds unless the records show that very value; listings out of stock or over budget do not fit."""

AUDIT = """A buying agent is about to give up without a purchase, believing nothing fits. Test the listing records below against the requirements, reading them this way: """ + READING + """
A record marked ROW came only from a search result, its page unread, so it can neither pass nor fail on a property the row leaves out.
Answer with ONE JSON object and nothing else: {"fits": {"item number": [ids that seem to meet every requirement, cheapest first, at most 4]}, "open": [ids of ROW listings of a requested kind and inside the budget whose page might show a fit, best first, at most 6], "reason": "one sentence"}"""

UNMIX = """A request was read into firm requirements on a product. Some of those take their value from another listing the request points at, and words that only identify that other listing (its kind, its shop, the properties listed after "with ...") sometimes get read by mistake as requirements of the product being bought. Going by the request's wording, name the requirements below that merely describe the other listing. Answer with ONE JSON object and nothing else: {"describe_other": [those requirements copied exactly; empty if all are real requirements of the product]}"""

TRAIL = """A buying agent is about to check out. Some requirements take their value from another listing, and that listing's page may send the reader on to a further listing for the property. The order only counts once the agent has opened in full each listing on such a chain. Trace every chain from the requirement's own description, one link at a time, through the words on the pages, using the records below (OPENED: page read by the agent; ROW: search row only, page unread).
Answer with ONE JSON object and nothing else: {"chains": [{"need": the requirement copied exactly, "links": [{"id": the listing id when a record below is that link, else "", "about": how the requirement or the page before describes it, "shop": the shop or seller named for it, or "", "kind": the category words of that listing, or "", "search": two to four words likely to stand in that listing's title, without the shop's name}], "done": true when the final link is an OPENED page stating the property's value itself, else false, "value": that value as the final page states it, or "", "quote": the final page's own words stating that value, copied exactly, or ""}]}
Give the links in order and stop at the first link that is unopened or not found, since what it says is still unknown."""

GATE = """A buying agent is about to STAGE. The request asks for the steps listed below along the way. Using the numbered RECORD of what the agent has done, decide for each step due before this STAGE whether it has been done for the cart as it stands. A step is done only when a RECORD line shows it, and for a step about the cart or the order, only when that line comes after the last cart change and the reply grants it. Questions on product properties do not do a step.
If a step is missing, give the single store call that does it next, using a tool from TOOLS with its arguments. A message to the shopper goes through the messaging tool as one short plain sentence or question about that step only, without product properties or values the step does not need. Give no call when nothing due is missing, or when the messages or steps left cannot do what is missing.
Answer with ONE JSON object and nothing else: {"steps": [{"step": the step, "done": true or false, "lines": [the RECORD line numbers that show it done]}], "call": {"tool": "...", "args": {...}} or null, "hold": what a reply asks to change before going on, or "", "needs": [{"item": item number, "need": "property: constraint"} for each firm requirement such a reply states]}"""

NEAREST = """A buying agent found no way to meet everything and is ending without a purchase, and the request has a rule for this case. From the records below, work out what that rule wants named, one listing for each requested item:
- For each item, score every candidate of that item's kind on that item's own requirements: one point for each requirement its record breaks or does not state, one point if it is unavailable. A range or list that takes in the wanted value breaks nothing. Take the lowest score; among equal scores take the lowest price, unless the rule says otherwise.
- Prices and the budget are in one unit, the figures as shown (a price of 348 against a budget of 1263 is inside it). Going over the budget is one point for the whole order, not for any listing, so never take a worse-fitting listing just to stay inside the budget.
Name every item this way, including items that fit fully. Answer with ONE JSON object and nothing else: {"why": "one short sentence saying what could not be met", "offers": [{"item": item number, "id": listing id, "price": its price as the record shows it, "breaks": ["each requirement its record plainly contradicts, unavailability included"], "unstated": ["each requirement its record does not address"]}]}"""

SWAP = """A listing in a buying agent's cart has just gone out of stock. Among the other records shown, find what could replace that line. The same product under another listing (the same or nearly the same title) comes first, and a listing that states a requirement's value comes before one whose record is silent on it. Read the records this way: """ + READING + """
Answer with ONE JSON object and nothing else: {"fits": [{"id": listing id, "price": its price as a number, as shown} for each listing that seems to meet every requirement of the gone line's item, at most 6], "reason": "one sentence"}"""

SHAPE = """Below is what a store tool returned. Identify its parts using the exact keys found in the result and the tool's parameters. Answer with ONE JSON object and nothing else: {"rows": the key of the listings list, "count": the key for how many listings matched overall (not how many are shown), or null, "next": the key whose value asks for the next page, or null, "next_arg": the tool parameter taking that value, or null, "price": the key, within a row, of the listing's price, or null, "stock": the key, within a row, saying whether the listing can be bought now, or null}"""


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


FIT_CODE = r"[A-Z]{2,6}[- ]?\d{1,5}(?:[xX]{1,3})?"


def fit_key(text: Any) -> str:
    return re.sub(r"[\s\-]+", "", str(text)).upper()


def fit_family(key: str) -> str:
    hit = re.match(r"[A-Z]+", key)
    return hit.group(0) if hit else ""


def read_fit_text(text: Any) -> dict[str, Any]:
    body = str(text or "")
    marks = {"own": set(), "takes": set(), "wild": [], "skip": set(), "tag": "", "also": [], "more": ""}
    tag = re.search(r"[\[(]\s*(?:[Pp]art\s+)?([A-Z]{2,6}\d{2,6})\s*[\])]", body)
    if tag:
        marks["tag"] = tag.group(1)
    for hit in re.finditer(r"(?i:\bpart\s+)(" + FIT_CODE + r")(?i:\s+also\s+(?:fits|works with|suits)\s*:?\s*)([^.;\n]+)", body):
        marks["also"].append((fit_key(hit.group(1)), [fit_key(c) for c in re.findall(r"\b" + FIT_CODE + r"\b", hit.group(2))]))
    body = re.sub(r"(?i:\bpart\s+)" + FIT_CODE + r"(?i:\s+also\s+(?:fits|works with|suits))[^.;\n]*", " ", body)
    more = re.search(r"(?i:\bsee\s+part\s+)(" + FIT_CODE + r")", body)
    if more:
        marks["more"] = fit_key(more.group(1))
    for hit in re.finditer(r"(?:^|[.;\n]\s*)\s*(?i:model\s*(?:no\.?|number|#)?\s*[:#]?\s*)(" + FIT_CODE + r")\b(?!\s+also)", body):
        marks["own"].add(fit_key(hit.group(1)))
    for hit in re.finditer(r"(?<![Nn]ot )(?<![Nn]ever )\b(?i:compatible with|works? with|fits?|suits?)\b\s*:?\s*([^.;\n]*)", body):
        clause = hit.group(1)
        if re.match(r"(?i)\s*part\s+series", clause):
            continue
        head, _, tail = clause.partition(" except ") if " except " in clause else (clause, "", "")
        for code in re.findall(r"\b" + FIT_CODE + r"\b", head):
            key = fit_key(code)
            if key[-1] == "X" and re.search(r"\dX+$", key):
                marks["wild"].append(re.sub(r"X+$", "", key))
            else:
                marks["takes"].add(key)
        marks["skip"].update(fit_key(c) for c in re.findall(r"\b" + FIT_CODE + r"\b", tail))
    return marks


def fit_state(marks: dict[str, Any], codes: set[str], notes: dict[str, set[str]]) -> bool | None:
    families = {fit_family(c) for c in codes}
    pool = {c for c in marks["takes"] | set(notes.get(marks["tag"], ())) if fit_family(c) in families}
    wild = [w for w in marks["wild"] if fit_family(w) in families]
    if any(c in marks["skip"] for c in codes):
        return False
    if any(c in pool or any(c.startswith(w) for w in wild) for c in codes):
        return True
    if pool or wild:
        return None if marks["more"] and marks["more"] not in notes else False
    return None


LINKS = """A shopping request can tie its products together or add rules beyond each product's own requirements. Read the request against its numbered ITEMS and report only what a sentence plainly says. Hidden wishes and story do not count.

Return ONE JSON object and nothing else:
{"fits": [{"item": number of the product that must work with something, "to_item": number of the requested product it must work with whichever one is chosen, or null, "code": the model code the shopper gives for a device or vehicle they own, copied exactly, or "", "quote": the words of that sentence}],
"rules": [{"then_item": number, "then_key": property, "then_values": [allowed values], "if_item": number, "if_key": property, "if_values": [values], "unless": true when the requirement stands except when the condition holds, false when it stands only when the condition holds, "quote": the words of that sentence}],
"caps": [{"item": number, "max": the highest price for one unit of that product, as a number, "quote": the words of that sentence}],
"total_pick": {"yes": true when the shopper wants the whole order at its lowest total, else false, "quote": the words, or ""}}
Use empty lists when a kind is absent. Copy each quote word for word from the request."""


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
        self.links: dict[str, list[dict[str, Any]]] = {"fits": [], "rules": [], "caps": []}
        self.fit_cache: dict[tuple[str, bool], dict[str, Any]] = {}
        self.link_holds = 0
        self.combo_tries = 0
        self.hunts = 0
        self.chain_early = False
        self.chain_on_decline = False
        self.scanned: set[str] = set()
        self.forced_at: set[int] = set()
        self.early_done = False
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
        self.row_reads = 0
        self.began = time.monotonic()
        self.near_got: dict[str, Any] | None = None
        self.near_version = -1
        self.near_tried = False
        self.swept: set[str] = set()
        self.gap_blocks = 0
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

    def think(self, messages: list[dict[str, Any]], models: list[str], max_tokens: int, tools: list[dict[str, Any]] | None = None) -> dict[str, Any] | None:
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
        self.check_quantities()
        self.read_links()
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
            talk = [p["messages"] for p in self.process if p["when"] == "order" and p["messages"]]
            self.reserve = min(max(talk), self.msg_limit) if talk else 0

    def unmix(self) -> None:
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
        parts += self.links_text()
        if brief.get("quiet_decline") is True:
            parts.append("if nothing fits: end without naming or suggesting any listing")
        rule = self.no_fit_rule()
        if rule:
            parts.append("if nothing fits, the request says: " + rule)
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
        if self.elapsed() > CLOCK_LATE:
            text += " TIME IS NEARLY UP: set the cart to your best lines now and check out on the very next call."
        elif left <= self.tail() + 1:
            text += " FINAL STEPS: check out now when the cart holds a line for each item; end without a purchase only when your search showed that an item has no fitting listing (running out of steps does not show that)."
        elif left <= 6 or (left <= len(self.items) + 5 and self.buys == 0):
            text += " Few steps left: set the cart to the chosen lines and check out now."
        elif self.buys == 0 and left <= max(12, self.max_steps // 4 + 2):
            text += " Steps are running low: settle on your picks, fill the cart and check out within the next few steps."
        elif self.buys == 0 and not self.cart_touched and self.step >= self.max_steps // 2:
            text += " Half the steps are used and the cart is empty: cart each item's fitting listing now (the store may still sell one out)."
        live = {need for _, need in self.needs_now()}
        lookups = [need for _, need in self.refs if need in live and need not in self.ref_values]
        if lookups and self.buys == 0:
            text += "\n[resolve first: open the listing each of these points at, and every listing its page refers to; the last page gives the value] " + " | ".join(lookups[:3])
        if self.leads:
            text += "\n[review candidates] " + ", ".join(self.leads)
        unread = self.rows_stating_needs() if self.buys == 0 and left > 8 else []
        if unread:
            text += "\n[unread rows whose own text states a required value: open these first] " + ", ".join(unread)
        board = self.fit_board() if self.buys == 0 and left > 4 else ""
        if board:
            text += "\n[fit board] " + board
        if self.notes:
            text += "\n[shopper's updates in force] " + " | ".join(self.notes[-4:])
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
        table: list[tuple[str, str, int, list[str]]] = []
        for label, keys in self.menu_groups():
            for name, per in self.menu_keys(keys)[:14]:
                seen: list[str] = []
                for value in per.values():
                    if value not in seen:
                        seen.append(value)
                table.append((label, name, len(per), [v[:40] for v in seen[:4]]))
        return table

    def menu_groups(self) -> list[tuple[str, dict[str, dict[str, str]]]]:
        cat_key = self.category_key()
        groups: dict[str, dict[str, dict[str, str]]] = {}
        sizes: dict[str, int] = {}
        for ident in self.seen_order:
            record = self.full.get(ident)
            if not record:
                continue
            label = str(record.get(cat_key) or "listings") if cat_key else "listings"
            sizes[label] = sizes.get(label, 0) + 1
            keys = groups.setdefault(label, {})
            for key, value in (_fact_block(record) or {}).items():
                name = _words(key)
                if name:
                    keys.setdefault(name, {}).setdefault(ident, str(value).strip().lower())
        wanted = [(_words(item.shelf or ""), _words(item.name), {w for w in _stem(item.shelf or "") | _stem(item.name) if len(w) >= 4}) for item in self.items]
        matched = {label for label in groups if any(_words(label) in (shelf, name) or {w for w in _stem(label) if len(w) >= 4} & long for shelf, name, long in wanted)}
        chosen = [label for label in groups if label in matched] or list(groups)
        return [(label, groups[label]) for label in sorted(chosen, key=lambda name: -sizes[name])]

    def menu_keys(self, keys: dict[str, dict[str, str]]) -> list[tuple[str, dict[str, str]]]:
        usable: list[tuple[str, dict[str, str]]] = []
        for name, per in keys.items():
            values = list(per.values())
            mean = sum(len(v) for v in values) / len(values)
            listy = sum(1 for v in values if " / " in v) * 2 >= len(values)
            unique_long = len(values) >= 4 and len(set(values)) == len(values) and mean > 12
            if mean <= 40 and not listy and not unique_long:
                usable.append((name, per))
        usable.sort(key=lambda pair: (-len(pair[1]), len(set(pair[1].values())) < 2))
        kept: list[tuple[str, dict[str, str]]] = []
        for name, per in usable:
            if not any(name.startswith(other + " ") and per == seen for other, seen in kept):
                kept.append((name, per))
        return kept

    def item_groups(self) -> list[list[str]]:
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
        if self.brief.get("pick") != "cheapest" or not self.price_key:
            return ""
        found = {i for seen in observations for i, _ in self.records_in(seen)}
        rows = [(self.price_of(i), i) for i in found if self.buyable(i) and self.price_of(i) is not None]
        if len(rows) < 2:
            return ""
        return "\n[Kyo] buyable rows here, cheapest first: " + ", ".join(f"{i} ({p:g})" for p, i in sorted(rows)[:6])

    def wider_hold(self) -> str:
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

    def cap_search(self) -> tuple[str, str, str] | None:
        for name, schema in self.schemas.items():
            props = schema.get("properties") or {}
            cap = next((k for k, v in props.items() if isinstance(v, dict) and {"number", "integer"} & set(_kinds(v)) and re.search(r"cap|max|price|budget|limit", k, re.I)), None)
            shelf = next((k for k, v in props.items() if isinstance(v, dict) and "string" in _kinds(v) and "enum" not in v and re.search(r"section|categ|shelf|group|depart|aisle", k, re.I)), None)
            if cap and shelf:
                return name, shelf, cap
        return None

    def cheapest_sweep(self) -> None:
        if self.brief.get("pick") != "cheapest" or self.left() <= self.tail() + 3 or self.elapsed() > CLOCK_LATE:
            return
        mark = self.cart_mark() + f"#{self.version}"
        if mark in self.swept:
            return
        self.swept.add(mark)
        cat_key = self.category_key()
        capped = self.cap_search()
        tool = self.roles.get("search")
        props = (self.schemas.get(tool or "") or {}).get("properties") or {}
        word = next((k for k, v in props.items() if isinstance(v, dict) and "string" in _kinds(v) and "enum" not in v and re.search(r"term|phrase|query|text|find|q$", k, re.I)), None)
        for line in self.cart_ids():
            price = self.price_of(line)
            label = str((self.rows.get(line) or {}).get(cat_key) or "").strip() if cat_key else ""
            if price is None or self.left() <= self.tail() + 2 or (capped and label):
                continue
            number = next((n for n, it in enumerate(self.items, 1) if {w for w in _stem(it.name) if len(w) >= 4} & {w for w in _stem(label) if len(w) >= 4}), 0)
            item = self.items[number - 1] if number else None
            if capped and label:
                name, shelf, cap = capped
                args: dict[str, Any] = {shelf: label, cap: max(0, int(price) - 1)}
            elif tool and word and item:
                values = [re.sub(r"\b(at most|at least|no more than|up to|exactly|only|about)\b", " ", m.split(":", 1)[-1], flags=re.I).strip() for m in item.musts[:2] + item.told[:1] if not re.search(r"\b(not|unlike|same as|page|listing)\b", m, re.I)]
                room = props[word].get("maxLength") if isinstance(props[word].get("maxLength"), int) else 60
                args = {word: (" ".join(v for v in values if v) + " " + (item.name.split() or [""])[-1]).strip()[:room].strip()}
                name = tool
            else:
                continue
            faults: list[str] = []
            args = _shape(args, self.schemas.get(name), "args", faults)
            if faults:
                continue
            reply = self.act(name, args)
            if reply is None or self.finished:
                return
            self.observe(name, reply.get("observation"))
            self.trace.append({"kind": "sweep", "args": _clip(args, 80)})

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
            picks += [i for i in pool[:2] if i not in picks]
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

    def reading_lanes(self) -> list[list[str]]:
        wanted = self.wanted_words()
        cheapest = self.brief.get("pick") == "cheapest"
        lanes: list[list[str]] = []
        for ids in self.item_groups():
            room = max(0, 8 - sum(1 for i in ids if i in self.full))
            pool = [i for i in ids if i not in self.full and self.buyable(i)]
            def rank(ident: str) -> tuple[int, float]:
                price = self.price_of(ident)
                return (-self.value_hits(ident, wanted), price if cheapest and price is not None else 0.0)
            lanes.append(sorted(pool, key=rank)[:room])
        return lanes

    def side_view(self) -> tuple[str, str, int] | None:
        tool = self.roles.get("compare")
        schema = self.schemas.get(tool or "") or {}
        props = schema.get("properties") or {}
        target = next((k for k, v in props.items() if isinstance(v, dict) and "array" in _kinds(v)), None)
        if not tool or not target or any(k != target for k in schema.get("required") or []) or self.compare_bare():
            return None
        limit = props[target].get("maxItems")
        return tool, target, limit if isinstance(limit, int) else 4

    def widen(self) -> None:
        lanes = self.reading_lanes()
        view = self.side_view()
        if view and self.left() > 12:
            tool, target, limit = view
            share = max(1, limit // max(1, len(lanes)))
            picks = [i for lane in lanes for i in lane[:share]][:limit]
            if len(picks) >= 2 and self.read_side_by_side(tool, target, picks):
                return
        if not view or self.facts_from.get(view[0]) is False:
            turns = [lane[k] for k in range(2) for lane in lanes if k < len(lane)]
            self.fill_facts(turns[:5], 22)

    def read_side_by_side(self, tool: str, target: str, picks: list[str]) -> bool:
        faults: list[str] = []
        args = _shape({target: picks}, self.schemas.get(tool), "args", faults)
        if faults:
            return False
        before = len(self.full)
        reply = self.act(tool, args)
        if reply is None:
            return False
        seen = reply.get("observation")
        self.observe(tool, seen)
        self.note_view(tool, seen)
        rows = self.records_in(seen)
        if rows:
            self.facts_from[tool] = self.facts_from.get(tool, False) or any(_fact_block(r) for _, r in rows)
        self.trace.append({"kind": "widen", "ids": picks, "opened": len(self.full) - before})
        if rows and self.facts_from.get(tool) is False:
            return False
        if len(self.full) == before and len(picks) > 2 and self.left() > 12:
            return self.read_side_by_side(tool, target, picks[:2])
        return True

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
        self.asked = True
        room = self.msgs_free()
        if room <= 0:
            return ""
        self.widen()
        waiting = self.question_queue(room)
        self.trace.append({"kind": "queue", "ask": waiting[:8]})
        sent: list[tuple[int, str, str]] = []
        told: list[str] = []
        fresh_from = {number: len(item.told) for number, item in enumerate(self.items, 1)}
        spent: set[tuple[int, str]] = set()
        settled_items: set[int] = set()
        retried = False
        while waiting and self.msgs_free() > 0:
            number, label = self.next_question(waiting, settled_items)
            spent.add((number, _words(label)))
            batch = self.send([(number, label, self.question(number, label))])
            if not batch:
                break
            sent += batch
            heard = self.hear(batch, (LIGHT, HEAVY))
            before = len(told)
            self.take_heard(heard, told)
            if len(told) > before:
                settled_items.add(number)
                continue
            if not retried and self.msgs_free() > 1 and _items(heard.get("crowded")):
                samples = self.samples(number, label)
                if len(samples) >= 2:
                    retried = True
                    again = self.send([(number, label, self.question(number, label, samples))])
                    if not again:
                        break
                    sent += again
                    before = len(told)
                    self.take_heard(self.hear(again, (LIGHT, HEAVY)), told)
                    if len(told) > before:
                        settled_items.add(number)
        summary = self.interview_summary(sent, told)
        found = self.chase_reveals(fresh_from)
        return summary + ("\n" + found if found else "")

    def chase_reveals(self, fresh_from: dict[int, int]) -> str:
        tool = self.roles.get("search")
        schema = self.schemas.get(tool or "") or {}
        props = schema.get("properties") or {}
        target = next((k for k, v in props.items() if isinstance(v, dict) and "string" in _kinds(v) and "enum" not in v and re.search(r"term|phrase|query|text|find|q$", k, re.I)), None)
        if not tool or not target:
            return ""
        room = props[target].get("maxLength") if isinstance(props[target].get("maxLength"), int) else 60
        lines: list[str] = []
        for number, item in enumerate(self.items, 1):
            for need in item.told[fresh_from.get(number, 0):][:2]:
                if self.finished or self.left() <= 14:
                    return "\n".join(lines)
                value = re.sub(r"\b(at most|at least|no more than|up to|exactly|not|only|about)\b", " ", need.split(":", 1)[-1], flags=re.I)
                value = " ".join(value.split())
                noun = item.name.split()[-1] if item.name.split() else ""
                terms = (value + " " + noun).strip()[:room].strip()
                if len(value) < 2 or len(terms) < 3:
                    continue
                known = set(self.seen_order)
                args: dict[str, Any] = {target: terms}
                faults: list[str] = []
                args = _shape(args, schema, "args", faults)
                if faults:
                    continue
                reply = self.act(tool, args)
                if reply is None or self.finished:
                    return "\n".join(lines)
                self.observe(tool, reply.get("observation"))
                fresh = [i for i in self.seen_order if i not in known and self.buyable(i)]
                lead = [i for i in fresh if value.lower() in _dump(self.rows.get(i)).lower()] or fresh
                self.trace.append({"kind": "chase", "terms": terms, "fresh": len(fresh)})
                read = self.fill_facts(lead[:2], 16) if lead else []
                lines.append(f"Kyo searched {terms!r} for item {number}: {len(fresh)} new listings." + ("\n" + "\n".join(read) if read else ""))
        return "\n".join(lines)

    def wanted_words(self) -> list[list[str]]:
        wanted: list[list[str]] = []
        for number, need in self.needs_now():
            value = re.sub(r"\b(at most|at least|no more than|up to|exactly|only|about|in|of|a|an|the)\b", " ", need.split(":", 1)[-1], flags=re.I)
            words = [w for w in re.findall(r"[a-z0-9.]+", value.lower()) if len(w) >= 2]
            if words and not re.search(r"\b(not|unlike|other than|same as|page|listing)\b", need, re.I):
                wanted.append(words)
        return wanted

    def value_hits(self, ident: str, wanted: list[list[str]]) -> int:
        text = _dump(self.rows.get(ident)).lower()
        return sum(1 for words in wanted if all(re.search(r"(?<![a-z0-9.])" + re.escape(w) + r"(?![a-z0-9])", text) for w in words))

    def rows_stating_needs(self, limit: int = 4) -> list[str]:
        wanted = self.wanted_words()
        if not wanted:
            return []
        scored: list[tuple[int, int, str]] = []
        for order, ident in enumerate(self.seen_order):
            if ident in self.full or self.sold_out(ident) or not self.buyable(ident):
                continue
            hits = self.value_hits(ident, wanted)
            if hits:
                scored.append((-hits, order, ident))
        return [ident for _, _, ident in sorted(scored)[:limit]]

    def question_queue(self, room: int) -> list[tuple[int, str]]:
        settled = [f"item {n}: {it.name} - settled by the request: " + ("; ".join(it.musts) or "nothing") for n, it in enumerate(self.items, 1)]
        listing: list[str] = []
        last = None
        table = self.key_table()
        for index, (label, name, carried, values) in enumerate(table, 1):
            if label != last:
                listing.append(f"category {label!r}:")
                last = label
            listing.append(f"  {index}. {name} (on {carried}): " + " | ".join(values))
        prompt = PLAN.replace("COUNT", str(room + 2)) + "\n\nREQUEST:\n" + self.request + "\n\nITEMS:\n" + "\n".join(settled)
        first = [str(x) for x in _items(self.brief.get("ask_first")) if str(x).strip()]
        if first:
            prompt += "\n\nASK FIRST (the request asks to be asked about these):\n" + "\n".join(first)
        prompt += "\n\nATTRIBUTE KEYS SEEN:\n" + ("\n".join(listing) or "(no full listing records yet)")
        plan = self.ask_json(prompt, [HEAVY, LIGHT], 2600) or {}
        self.trace.append({"kind": "plan", "ask": _clip(plan.get("ask"), 120), "story": _clip(plan.get("story"), 120)})
        queue: list[tuple[int, str]] = []
        for pick in _items(plan.get("ask")):
            if isinstance(pick, dict):
                self.queue_add(queue, _count(pick.get("item") or 1) or 1, str(pick.get("label") or ""))
        return self.queue_fill(queue, table, room + 2)

    def queue_add(self, queue: list[tuple[int, str]], number: int, label: str) -> None:
        number = min(max(1, number), len(self.items))
        label = re.split(r"\s+(?:and|or)\s+|[/,;(]", label)[0].strip().rstrip("?.")
        if not label or len(label.split()) > 5:
            return
        name = _words(label)
        settled = {_words(m.split(":")[0]) for m in self.items[number - 1].musts}
        if name in settled or (number, name) in {(n, _words(old)) for n, old in queue}:
            return
        queue.append((number, label))

    def queue_fill(self, queue: list[tuple[int, str]], table: list[tuple[str, str, int, list[str]]], want: int) -> list[tuple[int, str]]:
        if len(queue) >= want:
            return queue
        for number, item in enumerate(self.items, 1):
            words = {w for w in _stem(item.name) | _stem(item.shelf or "") if len(w) >= 4}
            rows = [row for row in table if {w for w in _stem(row[0]) if len(w) >= 4} & words and len(row[3]) >= 2]
            for row in rows[:3]:
                if len(queue) >= want:
                    return queue
                if not any(n == number for n, _ in queue) or len(self.items) == 1:
                    self.queue_add(queue, number, row[1])
        return queue

    def next_question(self, waiting: list[tuple[int, str]], settled_items: set[int]) -> tuple[int, str]:
        spot = next((i for i, (number, _) in enumerate(waiting) if number not in settled_items), 0)
        return waiting.pop(spot)

    def interview_summary(self, sent: list[tuple[int, str, str]], told: list[str]) -> str:
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

    def question(self, number: int, label: str, samples: list[str] | None = None) -> str:
        text = f"What {label} would you like for the {self.items[number - 1].name}"
        return text + (f", for example {samples[0]} or {samples[1]}?" if samples else "?")

    def samples(self, number: int, label: str) -> list[str]:
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
            words = self.spoken(answer)
            sent.append((number, text, words))
            self.done_log.append(f"step {self.step}: message to the shopper {text!r} -> reply {words[:300]!r} (cart then: {self.cart_mark()})")
            self.done_steps.append(self.step)
            self.trace.append({"kind": "ask", "q": text, "a": words[:300]})
            if isinstance(answer, dict) and answer.get("error"):
                break
        return sent

    def take_heard(self, heard: dict[str, Any], told: list[str]) -> None:
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

    def cart_text(self) -> str:
        if self.cart_view is not None and self.cart_seen_at >= self.cart_at:
            return _dump(_clip(self.cart_view, 300))[:3000]
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
        if key in self.checked or self.cost > limit or not ids or self.elapsed() > CLOCK_LATE:
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

    def cart_gaps(self) -> str:
        lines = {i: q for i, q in self.cart_lines.items() if q}
        if not lines:
            return ""
        problems: list[str] = []
        if len(lines) < len(self.items):
            problems.append(f"the request has {len(self.items)} items and the cart has {len(lines)} lines")
        elif len(lines) > len(self.items):
            problems.append(f"the cart has {len(lines)} lines for {len(self.items)} requested items, so one line is extra")
        for number, item in enumerate(self.items, 1):
            owners = [i for i in lines if self.item_of_row(i) == number]
            if len(owners) == 1 and lines[owners[0]] != item.qty:
                problems.append(f"item {number} wants {item.qty} units and {owners[0]} has {lines[owners[0]]}")
        total = 0.0
        known = True
        for ident, units in lines.items():
            price = self.price_of(ident)
            if price is None:
                known = False
            else:
                total += price * units
            if self.sold_out(ident):
                problems.append(f"{ident} is sold out")
        if known and isinstance(self.budget, (int, float)) and self.brief.get("budget_scope") != "each" and total > self.budget:
            problems.append(f"the lines come to {total:g}, above the budget of {self.budget:g}")
        return "Kyo's count of the cart found: " + "; ".join(problems) + ". Fix the cart, then check out." if problems else ""

    def trail(self) -> tuple[str, bool]:
        live = {need for _, need in self.needs_now()}
        pending = [need for _, need in self.refs if need in live and need not in self.ref_values]
        tool = self.roles.get("offer") or self.roles.get("open")
        target = self.single_target(tool)
        seen: list[str] = []
        lost: list[str] = []
        while pending and self.trails < 8 and self.cost < SPEND_HARD and self.elapsed() < CLOCK_LATE:
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
                if chain.get("done") is True and links and all(i in self.opened for i in ids) and str(chain.get("value") or "").strip() and self.page_says(ids[-1], chain):
                    self.ref_values[need] = str(chain["value"]).strip()
                    self.version += 1
                    continue
                closed = [i for i in ids if i in self.rows and i not in self.opened]
                if closed:
                    reach += [i for i in closed if i not in reach]
                else:
                    gap = next((l for l, i in zip(links, ids) if i not in self.rows), None) or {}
                    about = str(gap.get("about") or "") or "the listing it points at"
                    found = self.hunt_link(gap) if gap else []
                    if found:
                        reach += [i for i in found if i not in reach]
                    else:
                        lost.append(f"{need} -> find {about}")
            pending = [p for p in pending if p not in self.ref_values]
            if not reach or not tool or not target or self.left() < len(reach) + self.deadline_steps() // 2 + 1 or self.finished:
                break
            for ident in reach[: max(0, min(3, 10 - len(seen)))]:
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

    def page_says(self, ident: str, chain: dict[str, Any]) -> bool:
        said = self.value_words(chain.get("quote") or "")
        if not said:
            return False
        have = self.value_words(_dump(self.full.get(ident) or self.rows.get(ident) or {}))
        return len(said - have) <= len(said) // 6

    def rows_of_shop(self, gap: dict[str, Any]) -> list[str]:
        shop = str(gap.get("shop") or "").strip().lower()
        if len(shop) < 3:
            return []
        wanted = self.value_words(str(gap.get("about") or "") + " " + str(gap.get("kind") or ""))
        hits = []
        for ident in dict.fromkeys(self.seen_order):
            record = self.rows.get(ident) or {}
            if ident in self.opened or not any(isinstance(v, str) and v.strip().lower() == shop for v in record.values()):
                continue
            hits.append((-len(wanted & self.value_words(_dump(record))), ident))
        return [i for _, i in sorted(hits)[:2]]

    def hunt_link(self, gap: dict[str, Any]) -> list[str]:
        known = self.rows_of_shop(gap)
        if known or self.hunts >= 6 or self.left() <= self.deadline_steps():
            return known
        tool = self.roles.get("search")
        props = (self.schemas.get(tool or "") or {}).get("properties") or {}
        word = next((k for k, v in props.items() if isinstance(v, dict) and "string" in _kinds(v) and "enum" not in v and re.search(r"term|phrase|query|text|find|q$", k, re.I)), None)
        tries: list[tuple[str, dict[str, Any]]] = []
        if tool and word and str(gap.get("search") or "").strip():
            room = props[word].get("maxLength") if isinstance(props[word].get("maxLength"), int) else 60
            tries.append((tool, {word: str(gap["search"]).strip()[:room].strip()}))
        plan = self.cap_search()
        if plan and str(gap.get("kind") or "").strip():
            tries.append((plan[0], {plan[1]: str(gap["kind"]).strip()}))
        for name, args in tries:
            if self.hunts >= 6 or self.left() <= self.deadline_steps():
                break
            faults: list[str] = []
            args = _shape(args, self.schemas.get(name), "args", faults)
            if faults:
                continue
            self.hunts += 1
            reply = self.act(name, args)
            if reply is None or self.finished:
                break
            self.observe(name, reply.get("observation"))
            known = self.rows_of_shop(gap)
            self.trace.append({"kind": "hunt", "args": _clip(args, 60), "rows": known})
            if known:
                break
        return known

    def early_chain(self) -> str:
        if self.chain_early or not self.refs or self.finished:
            return ""
        self.chain_early = True
        text, _ = self.trail()
        self.trail_blocks = 0
        return text

    def offer_tool(self) -> str | None:
        return self.roles.get("offer") or self.roles.get("open")

    def reread_offers(self) -> str:
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
        return ident in self.unavailable or ident in self.gone

    def swap_hint(self) -> str:
        lost = [i for i in self.cart_ids() if i not in self.gone and self.sold_out(i)]
        if not lost:
            return ""
        for ident in lost:
            self.gone[ident] = self.step
        self.leads = [i for i in self.leads if i not in lost]
        self.version += 1
        twins = self.twin_rows(lost)
        picks = list(dict.fromkeys(twins + self.replacements(lost)))[:4]
        text = f"[Kyo] Sold out now: {', '.join(lost)} (the store shows it cannot be bought). It will not come back, so do not open it again."
        if picks and self.left() <= 6:
            return text + f" Few steps are left: in this turn add {picks[0]} at the requested quantity and remove the sold-out line (two calls), then check out. Other listings already read that appear to fit: {', '.join(picks[1:]) or 'none'}."
        if picks and self.brief.get("pick") == "cheapest":
            return text + f" Listings already read that appear to fit that item instead, cheapest first: {', '.join(picks)}. The pick rule is the cheapest, so unless you have already seen the whole fitting set, run one search under {picks[0]}'s price (the search's maximum price, if it has one) before you swap the sold-out line for the cheapest fitting listing."
        if picks:
            return text + f" Listings already read that appear to fit that item instead: {', '.join(picks)}. In this turn take the sold-out line out of the cart and add {picks[0]} at the requested quantity, then check out."
        return text + " Remove that line from the cart and add the best other listing that fits; search for one if none you have read fits."

    def title_of(self, ident: str) -> str:
        record = self.rows.get(ident) or {}
        named = [str(v) for k, v in record.items() if isinstance(v, str) and re.search(r"title|name|heading|headline", k, re.I)]
        return max(named, key=len) if named else ""

    def twin_rows(self, lost: list[str]) -> list[str]:
        tool = self.roles.get("search")
        props = (self.schemas.get(tool or "") or {}).get("properties") or {}
        word = next((k for k, v in props.items() if isinstance(v, dict) and "string" in _kinds(v) and "enum" not in v and re.search(r"term|phrase|query|text|find|q$", k, re.I)), None)
        found: list[str] = []
        for ident in lost[:2]:
            title = self.title_of(ident)
            words = _stem(title)
            if not words:
                continue
            if tool and word and self.left() > self.tail() + 3:
                room = props[word].get("maxLength") if isinstance(props[word].get("maxLength"), int) else 60
                args = _shape({word: re.sub(r"\[[^\]]*\]", " ", title).strip()[:room].strip()}, self.schemas.get(tool), "args", [])
                reply = self.act(tool, args)
                if reply is None or self.finished:
                    return found
                self.observe(tool, reply.get("observation"))
            for other in self.seen_order:
                if other in lost or other in found or self.sold_out(other) or not self.buyable(other) or other in self.gone:
                    continue
                theirs = _stem(self.title_of(other))
                if theirs and len(words & theirs) * 3 >= len(words | theirs) * 2:
                    found.append(other)
        self.trace.append({"kind": "twins", "lost": lost, "found": found[:4]})
        if found and self.left() > self.tail() + 2:
            self.fill_facts(found[:2], self.tail() + 1)
        return found[:3]

    def find_sold(self) -> str:
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
        if self.audited or self.cost > SPEND_SOFT or self.elapsed() > CLOCK_LATE:
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
            limit = spec.get("maxLength", 600)
            if self.wants_offer():
                text = self.nearest_offer() or text
                out[key] = (text or "No clean fit: nothing in the store meets all of your requirements, so nothing was ordered.")[:limit]
                continue
            named = any(ident in text for ident in self.rows) or self.brief.get("quiet_decline") is not False
            if named or not text:
                out[key] = "Sorry, nothing in the store meets all of your requirements, so I am not buying anything."[: spec.get("maxLength", 200)]
        return out

    def no_fit_rule(self) -> str:
        entry = self.brief.get("no_fit")
        quote = str(entry.get("quote") or "").strip() if isinstance(entry, dict) else ""
        return quote if quote and _words(quote)[:60] in _words(self.request) else ""

    def wants_offer(self) -> bool:
        entry = self.brief.get("no_fit")
        return bool(self.no_fit_rule()) and isinstance(entry, dict) and entry.get("offer") is True and self.brief.get("quiet_decline") is not True

    def widen_for_nearest(self) -> None:
        capped = self.cap_search()
        if not capped or not isinstance(self.budget, (int, float)) or self.left() <= 3:
            return
        name, shelf, cap = capped
        for item in self.items[:2]:
            if self.left() <= 3 or self.finished:
                return
            faults: list[str] = []
            label = item.shelf or item.name
            args = _shape({shelf: label, cap: self.budget}, self.schemas.get(name), "args", faults)
            if faults:
                continue
            reply = self.act(name, args)
            if reply is None or self.finished:
                return
            self.observe(name, reply.get("observation"))

    def nearest_offer(self) -> str:
        got = self.nearest_call()
        picks = [e for e in _items((got or {}).get("offers")) if isinstance(e, dict) and str(e.get("id")) in self.rows]
        if not picks:
            return ""
        parts = []
        for e in picks[:4]:
            misses = [str(x) for x in _items(e.get("breaks")) + _items(e.get("unstated"))]
            parts.append(f"{e['id']} at {e.get('price')}, short on: {'; '.join(misses) or 'nothing'}")
        return "No clean fit: " + str(got.get("why") or "nothing meets every requirement").strip() + " Closest offer: " + " | ".join(parts) + ". Nothing was ordered."

    def nearest_call(self) -> dict[str, Any] | None:
        if self.near_got is not None and self.near_version == self.version:
            return self.near_got
        self.widen_for_nearest()
        records = []
        ranked = sorted(self.seen_order, key=lambda i: self.price_of(i) if self.price_of(i) is not None else float("inf"))
        for ident in ranked:
            if self.sold_out(ident):
                continue
            record = self.full.get(ident)
            if record:
                records.append(_dump(_clip(record, 160))[:900])
            elif len(records) < 12:
                records.append("ROW " + _dump(_clip(self.rows.get(ident), 100))[:400])
            if len(records) >= 30:
                break
        if not records:
            return None
        rule = self.no_fit_rule() or "(the request states none: use the scoring below)"
        prompt = (NEAREST + "\n\nTHE REQUEST'S RULE:\n" + rule + "\n\nITEMS AND REQUIREMENTS IN FORCE:\n" + self.ledger()
                  + "\n\nLISTING RECORDS (cheapest first):\n" + "\n".join(records))
        got = self.ask_json(prompt, [HEAVY, LIGHT], 1100) or {}
        self.trace.append({"kind": "nearest", "got": _clip(got, 160)})
        self.near_got, self.near_version = got, self.version
        return got

    def unbroken_cart(self) -> str:
        plan = self.unbroken_plan()
        return ", ".join(f"{i} x{n}" for i, n in plan.items())

    def unbroken_plan(self) -> dict[str, int]:
        got = self.nearest_call() or {}
        chosen: dict[int, str] = {}
        for entry in _items(got.get("offers")):
            if isinstance(entry, dict) and str(entry.get("id")) in self.rows and not _items(entry.get("breaks")):
                number = _count(entry.get("item"))
                ident = str(entry["id"])
                if number and 1 <= number <= len(self.items) and not self.sold_out(ident) and not self.contradicted(number, ident) and self.backed(number, ident):
                    chosen.setdefault(number, ident)
        for ident in self.cart_ids():
            number = self.item_of_row(ident)
            if number and number not in chosen and not self.sold_out(ident) and not self.contradicted(number, ident):
                chosen[number] = ident
        if len(chosen) < len(self.items):
            return {}
        total = sum((self.price_of(ident) or 0.0) * self.items[number - 1].qty for number, ident in chosen.items())
        if isinstance(self.budget, (int, float)) and self.brief.get("budget_scope") != "each" and total > self.budget:
            return {}
        if self.plan_breaks(chosen):
            return {}
        plan = {ident: self.items[number - 1].qty for number, ident in sorted(chosen.items())}
        self.trace.append({"kind": "unbroken", "lines": plan})
        return plan

    def read_audit_rows(self) -> None:
        room = self.left() - len(self.items) - 1
        if room < 1 or self.elapsed() > CLOCK_LATE:
            return
        got = self.audit_call() or {}
        wanted = [str(i) for i in _items(got.get("open")) if str(i) in self.rows and str(i) not in self.full and not self.sold_out(str(i))]
        ranked = self.rows_stating_needs(3) + wanted
        rows = list(dict.fromkeys(ranked))[:min(room, 4)]
        if rows and self.open_rows(rows):
            self.trace.append({"kind": "audit_rows", "rows": rows})

    def links_text(self) -> list[str]:
        out = []
        for link in self.links["fits"]:
            target = f"model {link['code']}" if link.get("code") else f"whichever item {link.get('to_item')} is chosen"
            out.append(f"item {link['item']} must work with {target}: compare the codes on the listings' own text")
        for rule in self.links["rules"]:
            word = "unless" if rule.get("unless") else "only if"
            out.append(f"rule: item {rule['then_item']}'s {rule['then_key']} must be {' or '.join(map(str, rule['then_values']))}, {word} item {rule['if_item']}'s {rule['if_key']} is {' or '.join(map(str, rule['if_values']))}")
        for cap in self.links["caps"]:
            out.append(f"item {cap['item']}: at most {cap['max']:g} per unit")
        return out

    def check_quantities(self) -> None:
        tokens = [w[:-1] if len(w) > 3 and w.endswith("s") else w for w in re.findall(r"[a-z0-9]+", self.request.lower())]
        for item in self.items:
            name = [w[:-1] if len(w) > 3 and w.endswith("s") else w for w in re.findall(r"[a-z0-9]+", item.name.lower())]
            name = [w for w in name if w not in ("and", "for", "the", "of", "item")][:2]
            found = set()
            for at in range(len(tokens) - len(name) + 1 if name else 0):
                if tokens[at:at + len(name)] != name:
                    continue
                back = at - 1
                while back >= 0 and tokens[back] in ("x", "set", "of", "pair", "unit", "piece", "a"):
                    back -= 1
                if back >= 0:
                    word = tokens[back]
                    if word.isdigit() and 0 < int(word) < 50:
                        found.add(int(word))
            if len(found) == 1 and item.qty not in found:
                self.trace.append({"kind": "quantity_fixed", "item": item.name, "was": item.qty, "now": min(found)})
                item.qty = min(found)

    def read_links(self) -> None:
        if len(self.items) < 2 and not re.search(r"\b" + FIT_CODE + r"\b", self.request):
            return
        names = "\n".join(f"{n}. {it.name}" for n, it in enumerate(self.items, 1))
        prompt = LINKS + "\n\nITEMS:\n" + names + "\n\nREQUEST:\n" + self.request
        got = self.ask_json(prompt, [LIGHT], 900) or {}
        known = {fit_key(c) for c in re.findall(r"\b" + FIT_CODE + r"\b", self.request)}
        seen = {fit_key(e.get("code") or "") for e in _items(got.get("fits")) if isinstance(e, dict)}
        if not got or (known and len(self.items) >= 2 and not seen & known and re.search(r"(?i)\bmodel\b|\bsuit\b|\bfits?\b|compatible", self.request) and not _items(got.get("fits"))):
            got = self.ask_json(prompt, [HEAVY, LIGHT], 900) or got
        self.keep_links(got)
        self.trace.append({"kind": "links", "links": self.links})

    def keep_links(self, got: dict[str, Any]) -> None:
        words = set(re.findall(r"[a-z0-9]+", self.request.lower()))

        def quoted(entry: dict[str, Any]) -> bool:
            said = set(re.findall(r"[a-z0-9]+", str(entry.get("quote") or "").lower()))
            return bool(said) and len(said - words) <= len(said) // 8

        def number(value: Any) -> int:
            found = _count(value)
            return found if found and 1 <= found <= len(self.items) else 0

        for entry in _items(got.get("fits")):
            if not isinstance(entry, dict) or not quoted(entry) or not number(entry.get("item")):
                continue
            code = str(entry.get("code") or "").strip()
            code = code if code and fit_key(code) in {fit_key(c) for c in re.findall(r"\b" + FIT_CODE + r"\b", self.request)} else ""
            other = number(entry.get("to_item"))
            if code or (other and other != number(entry.get("item"))):
                self.links["fits"].append({"item": number(entry.get("item")), "to_item": other if not code else 0, "code": code})
        for entry in _items(got.get("rules")):
            if not isinstance(entry, dict) or not quoted(entry) or not number(entry.get("then_item")) or not number(entry.get("if_item")):
                continue
            then_values, if_values = [str(v) for v in _items(entry.get("then_values"))], [str(v) for v in _items(entry.get("if_values"))]
            if then_values and if_values and entry.get("then_key") and entry.get("if_key"):
                self.links["rules"].append({"then_item": number(entry.get("then_item")), "then_key": str(entry["then_key"]), "then_values": then_values,
                                            "if_item": number(entry.get("if_item")), "if_key": str(entry["if_key"]), "if_values": if_values, "unless": entry.get("unless") is True})
        for entry in _items(got.get("caps")):
            limit = entry.get("max") if isinstance(entry, dict) else None
            if isinstance(entry, dict) and quoted(entry) and number(entry.get("item")) and isinstance(limit, (int, float)) and not isinstance(limit, bool) and limit > 0 and str(int(limit)) in self.request:
                self.links["caps"].append({"item": number(entry.get("item")), "max": float(limit)})
        total = got.get("total_pick")
        if isinstance(total, dict) and total.get("yes") is True and quoted(total) and self.brief.get("pick") not in ("cheapest", "most", "least"):
            self.brief["pick"] = "cheapest"

    def item_of_row(self, ident: str) -> int:
        cat_key = self.category_key()
        label = str((self.rows.get(ident) or {}).get(cat_key) or "").strip() if cat_key else ""
        best, top = 0, 0.0
        for number, item in enumerate(self.items, 1):
            mine = {w for w in _stem(item.shelf or "") | _stem(item.name) if len(w) >= 3}
            if label:
                if _words(label) in (_words(item.shelf or ""), _words(item.name)):
                    return number
                theirs = {w for w in _stem(label) if len(w) >= 3}
                score = len(mine & theirs) / max(1, len(mine | theirs))
            else:
                theirs = {w for w in _stem(self.title_of(ident)) if len(w) >= 3}
                named = {w for w in _stem(item.name) if len(w) >= 3}
                score = len(named & theirs) / max(1, len(named))
                score = score if score >= 0.5 else 0.0
            if score > top:
                best, top = number, score
        return best

    def fit_marks_of(self, ident: str) -> dict[str, Any]:
        key = (ident, ident in self.full)
        if key not in self.fit_cache:
            record = self.full.get(ident) or self.rows.get(ident) or {}
            self.fit_cache[key] = read_fit_text("\n".join(v for v in record.values() if isinstance(v, str)))
        return self.fit_cache[key]

    def fit_notes(self) -> dict[str, set[str]]:
        notes: dict[str, set[str]] = {}
        for ident in self.seen_order:
            for part, codes in self.fit_marks_of(ident)["also"]:
                notes.setdefault(part, set()).update(codes)
        return notes

    def link_codes(self, link: dict[str, Any], chosen: dict[int, str]) -> set[str]:
        if link.get("code"):
            return {fit_key(link["code"])}
        device = chosen.get(link.get("to_item") or 0)
        return set(self.fit_marks_of(device)["own"]) if device else set()

    def fact_value(self, ident: str, key: str) -> str | None:
        block = _fact_block(self.full.get(ident) or {}) or {}
        wanted = _stem(key)
        for name, value in block.items():
            if _words(str(name)) == _words(key):
                return str(value)
        for name, value in block.items():
            have = _stem(re.sub(r"(?<=[a-z])(?=[A-Z])", " ", str(name)))
            if wanted and (wanted <= have or have <= wanted):
                return str(value)
        return None

    @staticmethod
    def value_words(text: Any) -> set[str]:
        return {w[:-1] if len(w) > 3 and w.endswith("s") else w for w in re.findall(r"[a-z0-9]+", str(text).lower())}

    @classmethod
    def value_is(cls, value: str, wanted: list[str]) -> bool:
        have = cls.value_words(value)
        return any(cls.value_words(w) and cls.value_words(w) <= have for w in wanted)

    def holds_value(self, ident: str, need: str) -> bool:
        value = need.partition(":")[2] or need
        if re.search(r"\b(?:not|no|never|except|other than|unlike|without|avoid)\b", value, re.I):
            return True
        record = self.full.get(ident) or self.rows.get(ident) or {}
        have = self.value_words(_dump(record))
        words = {w for w in self.value_words(re.sub(r"\b(?:at least|at most|or more|or less|up to|no more than|exactly|only|about|under|over)\b", " ", value, flags=re.I)) if len(w) >= 3 or w.isdigit()}
        return not words or any(w in have for w in words) if re.search(r"\bor\b|,|/", value) else words <= have

    def contradicted(self, number: int, ident: str) -> bool:
        item = self.items[number - 1]
        for must in item.musts + item.told:
            if must in item.dropped or any(need == must for _, need in self.refs):
                continue
            prop, _, value = must.partition(":")
            if not value.strip() or re.search(r"\d|\b(?:not|no|except|other than|unlike|same|at most|at least|under|over|below|above|less|more|between|within|up to|any)\b", value, re.I):
                continue
            stated = self.fact_value(ident, prop)
            if stated is None or not stated.strip():
                continue
            if not self.value_is(stated, [v for v in re.split(r",|\bor\b|/", value) if v.strip()]):
                return True
        return False

    def meets(self, number: int, ident: str) -> bool:
        return ident in self.full and not self.contradicted(number, ident) and self.backed(number, ident)

    def backed(self, number: int, ident: str) -> bool:
        item = self.items[number - 1]
        for must in item.musts + item.told:
            if must in item.dropped or any(need == must for _, need in self.refs):
                continue
            value = must.partition(":")[2] or must
            if re.search(r"\d", value) and re.search(r"\b(?:at most|at least|under|over|below|above|less|more|between|within|up to|or more|or less)\b", value, re.I):
                continue
            if not self.holds_value(ident, must):
                return False
        return True

    def list_below(self, label: str, cap: float) -> bool:
        plan = self.cap_search()
        if not plan or self.left() <= self.tail() + 2:
            return False
        name, shelf, field = plan
        faults: list[str] = []
        args = _shape({shelf: label, field: max(0, int(cap))}, self.schemas.get(name), "args", faults)
        if faults:
            return False
        reply = self.act(name, args)
        if reply is None or self.finished:
            return False
        observation = reply.get("observation")
        self.observe(name, observation)
        self.more_pages(name, args, observation)
        page = self.paging(name, observation)
        return bool(page and page[0] > 2 * page[1])

    def cheapest_scan(self) -> None:
        if self.brief.get("pick") != "cheapest" or not self.cap_search() or self.elapsed() > CLOCK_LATE:
            return
        mark = self.cart_mark() + f"#{self.version}"
        if mark in self.scanned or len(self.scanned) >= 3:
            return
        self.scanned.add(mark)
        chosen = self.cart_chosen()
        cat_key = self.category_key()
        notes = self.fit_notes()
        spare = self.left() - self.tail() - 2 * len(self.items) - 3
        if spare < 3 or not cat_key:
            return
        share = min(12, max(3, spare // max(1, len(chosen))))
        for number, line in sorted(chosen.items(), key=lambda kv: -((self.price_of(kv[1]) or 0.0) * self.items[kv[0] - 1].qty)):
            price = self.price_of(line)
            label = str((self.rows.get(line) or {}).get(cat_key) or "").strip()
            if price is None or price <= 1 or not label:
                continue
            start = self.step
            ceiling = price - 1
            for _ in range(3):
                known = set(self.seen_order)
                self.list_below(label, ceiling)
                fresh = [self.price_of(i) for i in self.seen_order if i not in known and self.item_of_row(i) == number and self.price_of(i) is not None]
                if not fresh or self.left() <= self.tail() + 2 * len(self.items) + 2:
                    break
                ceiling = min(fresh) - 1
                if ceiling < 1:
                    break
            demands = [link for link in self.links["fits"] if link["item"] == number]
            pool = [i for p, i in self.pool_of(number) if p < price and i not in self.full]
            for ident in pool:
                if self.step - start >= share or self.left() <= self.tail() + 2 * len(self.items) + 1 or self.elapsed() > CLOCK_LATE:
                    break
                if any(self.link_codes(link, chosen) and fit_state(self.fit_marks_of(ident), self.link_codes(link, chosen), notes) is False for link in demands):
                    continue
                self.fill_facts([ident], self.tail() + 2 * len(self.items))
                if self.meets(number, ident):
                    self.trace.append({"kind": "scan_hit", "item": number, "id": ident, "price": self.price_of(ident)})
                    break

    def plan_breaks(self, chosen: dict[int, str]) -> list[str]:
        out = []
        notes = self.fit_notes()
        for link in self.links["fits"]:
            line = chosen.get(link["item"])
            codes = self.link_codes(link, chosen)
            if line and codes and fit_state(self.fit_marks_of(line), codes, notes) is False:
                out.append(f"{line} (item {link['item']}) names no fit with {', '.join(sorted(codes))}")
        for rule in self.links["rules"]:
            when, then = chosen.get(rule["if_item"]), chosen.get(rule["then_item"])
            if not when or not then:
                continue
            seen = self.fact_value(when, rule["if_key"])
            if seen is None:
                continue
            if self.value_is(seen, rule["if_values"]) != bool(rule["unless"]):
                stated = self.fact_value(then, rule["then_key"])
                if stated is not None and not self.value_is(stated, rule["then_values"]):
                    out.append(f"{then} has {rule['then_key']} {stated}, but with {when}'s {rule['if_key']} {seen} the request needs {' or '.join(rule['then_values'])}")
        for cap in self.links["caps"]:
            line = chosen.get(cap["item"])
            price = self.price_of(line) if line else None
            if price is not None and price > cap["max"] and self.category_key():
                out.append(f"{line} costs {price:g}, over the {cap['max']:g} per unit the request allows for item {cap['item']}")
        return out

    def cart_chosen(self) -> dict[int, str]:
        chosen: dict[int, str] = {}
        for ident in self.cart_ids():
            number = self.item_of_row(ident)
            if number:
                chosen.setdefault(number, ident)
        return chosen

    def good_rows(self, number: int, codes: set[str], notes: dict[str, set[str]]) -> list[str]:
        rows = [i for i in self.seen_order if self.item_of_row(i) == number and self.buyable(i) and fit_state(self.fit_marks_of(i), codes, notes) is True]
        return sorted(dict.fromkeys(rows), key=lambda i: self.price_of(i) if self.price_of(i) is not None else float("inf"))

    def link_faults(self) -> list[str]:
        chosen = self.cart_chosen()
        faults = self.plan_breaks(chosen)
        notes = self.fit_notes()
        for link in self.links["fits"]:
            line = chosen.get(link["item"])
            codes = self.link_codes(link, chosen)
            if line and codes and fit_state(self.fit_marks_of(line), codes, notes) is False:
                good = [i for i in self.good_rows(link["item"], codes, notes) if i != line][:4]
                faults.append(f"seen rows of item {link['item']} that do name it: " + ", ".join(good) if good else f"no seen row of item {link['item']} names it, so search that item again or change item {link.get('to_item') or link['item']}")
        return faults

    def pool_of(self, number: int) -> list[tuple[float, str]]:
        cap = min([c["max"] for c in self.links["caps"] if c["item"] == number] or [float("inf")])
        out = []
        for ident in dict.fromkeys(self.seen_order):
            price = self.price_of(ident)
            if price is not None and price <= cap and self.item_of_row(ident) == number and self.buyable(ident) and not self.contradicted(number, ident):
                out.append((price, ident))
        return sorted(out)

    def proven_pool(self, number: int) -> list[tuple[float, str]]:
        line = self.cart_chosen().get(number)
        return [(p, i) for p, i in self.pool_of(number) if i == line or self.meets(number, i)]

    def combo_plan(self) -> dict[int, str]:
        notes = self.fit_notes()
        pools = {n: self.proven_pool(n) for n in range(1, len(self.items) + 1)}
        if not all(pools.values()):
            return {}
        anchors = sorted({link["to_item"] for link in self.links["fits"] if link.get("to_item")})
        main = anchors[0] if anchors else 0
        devices = [i for _, i in pools[main] if self.fit_marks_of(i)["own"]][:14] if main else [""]
        best: dict[int, str] = {}
        best_total = float("inf")
        for device in devices:
            chosen = {main: device} if main else {}
            for number in pools:
                if number in chosen:
                    continue
                demands = [link for link in self.links["fits"] if link["item"] == number]
                for price, ident in pools[number]:
                    trial = {**chosen, number: ident}
                    if all(fit_state(self.fit_marks_of(ident), self.link_codes(link, trial), notes) is not False for link in demands if self.link_codes(link, trial)):
                        chosen[number] = ident
                        break
            if len(chosen) < len(pools) or self.plan_breaks(chosen):
                continue
            total = sum((self.price_of(i) or 0.0) * self.items[n - 1].qty for n, i in chosen.items())
            if total < best_total:
                best, best_total = chosen, total
        return best

    def combo_total(self, chosen: dict[int, str]) -> float:
        return sum((self.price_of(i) or 0.0) * self.items[n - 1].qty for n, i in chosen.items())

    def set_cart(self, lines: dict[str, int]) -> bool:
        cart = self.roles.get("cart")
        now = {i: q for i, q in self.cart_lines.items() if q}
        if now == lines:
            return True
        calls = self.cart_calls(lines, [i for i in now if i not in lines])
        if not cart or not calls or len(calls) + 1 > self.left():
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
        return True

    def combo_hold(self) -> str:
        if self.combo_tries >= 2 or self.brief.get("pick") != "cheapest" or self.left() <= self.tail() + len(self.items) + 2 or self.elapsed() > CLOCK_LATE:
            return ""
        now = self.cart_chosen()
        if len(now) < len(self.items) or any(self.price_of(i) is None for i in now.values()):
            return ""
        plan = self.combo_plan()
        if not plan or self.combo_total(plan) >= self.combo_total(now):
            return ""
        self.combo_tries += 1
        ids = list(plan.values())
        self.check_cart(ids, SPEND_SOFT)
        passed = self.verdicts.get(self.check_key(ids))
        self.trace.append({"kind": "combo", "plan": plan, "total": self.combo_total(plan), "was": self.combo_total(now), "passed": passed})
        if not passed or not self.set_cart({i: self.items[n - 1].qty for n, i in plan.items()}):
            return ""
        return f"Kyo found a choice with a lower total ({self.combo_total(plan):g}, the cart was {self.combo_total(now):g}) that passed the review and set the cart to it. Check out unless a line breaks a requirement."

    def fit_board(self) -> str:
        if not self.links["fits"]:
            return ""
        notes = self.fit_notes()
        out = []
        for link in self.links["fits"]:
            if link.get("code"):
                codes = {fit_key(link["code"])}
                rows = self.good_rows(link["item"], codes, notes)[:4]
                if rows:
                    out.append(f"item {link['item']} rows naming {link['code']}: " + ", ".join(f"{i} ({self.price_of(i):g})" if self.price_of(i) is not None else i for i in rows))
        anchors = sorted({link["to_item"] for link in self.links["fits"] if link.get("to_item")})
        for main in anchors:
            parts = [link["item"] for link in self.links["fits"] if link.get("to_item") == main]
            ranked = []
            for device in dict.fromkeys(i for i in self.seen_order if self.item_of_row(i) == main and self.buyable(i)):
                codes = set(self.fit_marks_of(device)["own"])
                if not codes:
                    continue
                found = {p: self.good_rows(p, codes, notes)[:2] for p in parts}
                ranked.append((-sum(bool(v) for v in found.values()), self.price_of(device) or 0.0, device, codes, found))
            for _, _, device, codes, found in sorted(ranked, key=lambda r: r[:2])[:3]:
                out.append(f"item {main} {device} ({', '.join(sorted(codes))}) fits " + "; ".join(f"item {p}: {', '.join(v) or 'none seen'}" for p, v in found.items()))
        return " | ".join(out)[:700]

    def place_order(self, lines: dict[str, int], why: str) -> str:
        buy = self.roles.get("buy")
        if not buy or not lines or not self.set_cart(lines) or self.finished:
            return ""
        faults: list[str] = []
        args = _shape(dict(self.brief.get("buy_args") or {}), self.schemas.get(buy), "args", faults)
        if faults or self.left() < 1:
            return ""
        if self.brief.get("pick") == "cheapest" and self.left() > self.tail() + 4:
            self.cheapest_scan()
            better = self.combo_plan()
            if better and self.combo_total(better) < self.combo_total(self.cart_chosen()):
                self.set_cart({i: self.items[n - 1].qty for n, i in better.items()})
        go, said = self.gate("order")
        if not go:
            return ""
        reply = self.act(buy, args)
        self.trace.append({"kind": "placed", "lines": lines, "why": why, "reply": _clip(reply, 200)})
        if reply is None:
            return ""
        self.buys += 1
        if not reply.get("finished"):
            self.refused_at = self.step
        self.observe(buy, reply.get("observation"))
        return f"Not run (Kyo acted instead): {why}, so Kyo set the cart to {_dump(lines)} and checked out. " + (said + " " if said else "") + "Store reply: " + self.show(reply.get("observation"))

    def deadline_steps(self) -> int:
        return min(8, max(6, self.max_steps // 6)) + len(self.items) - 1

    def checkout_due(self) -> bool:
        return self.buys == 0 and not self.early_done and (self.left() <= self.deadline_steps() or self.elapsed() > CLOCK_LATE) and len(self.cart_chosen()) >= len(self.items) and not any(self.sold_out(i) for i in self.cart_ids())

    def scan_due(self) -> bool:
        if self.brief.get("pick") != "cheapest" or not self.cap_search() or self.elapsed() > CLOCK_LATE:
            return False
        if self.roles.get("ask") and not self.asked and self.msgs_free() > 0:
            return False
        if len(self.cart_chosen()) < len(self.items) or any(self.sold_out(i) for i in self.cart_ids()):
            return False
        return self.cart_mark() + f"#{self.version}" not in self.scanned and len(self.scanned) < 3

    def deadline_step(self) -> str:
        if self.scan_due():
            self.wider_held = True
            before = self.step
            self.cheapest_scan()
            combo = self.combo_hold()
            if self.finished:
                return ""
            self.trace.append({"kind": "scan_done", "steps": self.step - before, "combo": bool(combo)})
            return combo or f"Kyo listed each item below its cart price and read the cheapest rows ({self.step - before} steps); none states every requirement at a lower price. Unless a line breaks a requirement, check out now."
        if self.checkout_due():
            self.early_done = True
            lines = {i: q for i, q in self.cart_lines.items() if q}
            return self.place_order(lines, "few steps are left and the store can still turn the first checkout down")
        short = (self.left() <= self.deadline_steps() or self.elapsed() > CLOCK_LATE) and self.left() >= 2
        broken = any(self.sold_out(i) for i in self.cart_ids()) or not self.cart_ids()
        if short and broken and self.version not in self.forced_at and len(self.forced_at) < 3:
            self.forced_at.add(self.version)
            plan = self.unbroken_plan()
            return self.place_order(plan, "few steps are left, the cart cannot be bought as it is and these listings contradict no requirement") if plan else ""
        return ""

    def resolve_ids(self, name: str, args: dict[str, Any]) -> tuple[dict[str, Any], str]:
        if name not in {self.roles.get(r) for r in ("cart", "open", "offer", "compare")} - {None} or not self.id_key:
            return args, ""
        props = (self.schemas.get(name) or {}).get("properties") or {}
        free = [k for k, v in props.items() if isinstance(v, dict) and ("string" in _kinds(v) or "array" in _kinds(v)) and "enum" not in v]
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
            if ending and self.is_buy(name, args) and self.roles.get("refuse") and (self.left() <= self.tail() + 1 or self.elapsed() > CLOCK_LATE) and len(self.items) - len(self.cart_chosen()) >= 2:
                self.trace.append({"kind": "partial_cart_ended", "lines": list(self.cart_lines)})
                name = self.roles["refuse"]
                args = self.quiet(name, dict(self.brief.get("refuse_args") or {}))
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
                    gaps = self.cart_gaps() if self.gap_blocks < 2 else ""
                    if gaps:
                        self.gap_blocks += 1
                        self.trace.append({"kind": "cart_gaps", "gaps": gaps})
                        out.append((cid, "Not run yet (no step used). " + gaps + "\nCart now: " + self.cart_text()))
                        continue
                    self.cheapest_sweep()
                    self.cheapest_scan()
                    read = self.fill_facts(self.cart_ids(), self.tail() + 3) + self.cheaper_rows()
                    faults = self.link_faults() if self.link_holds < 3 and self.left() > self.tail() + 1 else []
                    if faults:
                        self.link_holds += 1
                        self.trace.append({"kind": "link_faults", "faults": faults})
                        out.append((cid, "Not run yet (no step used). Kyo's cart rules found: " + "; ".join(faults) + ". Fix those lines, then check out.\nCart now: " + self.cart_text()))
                        continue
                    combo = self.combo_hold()
                    if combo:
                        out.append((cid, "Not run yet (no step used). " + combo + "\nCart now: " + self.cart_text()))
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
            if self.is_refuse(name, args) and not self.late_held:
                lines = [i for i in self.built if not self.sold_out(i)]
                fresh = self.cart_view is not None and self.cart_seen_at >= self.cart_at
                missing = [i for i in lines if i not in self.cart_ids() or (i in self.dropped and not fresh)]
                if len(lines) >= len(self.items) and len(missing) + 1 <= self.left():
                    self.late_held = True
                    self.trace.append({"kind": "late_decline_held", "lines": lines, "missing": missing})
                    redo = f"Put {', '.join(missing)} back in the cart at the requested quantities, then check out." if missing else "Check out the cart as it is."
                    out.append((cid, "Not run yet (no step used). You already built a cart with a line for every item, and ending without a purchase earns nothing when the store has a fitting listing; neither running short of steps nor one doubtful requirement shows that it has none. "
                                + redo + " End without a purchase only if your search showed that some requested item has no listing that meets its requirements; then send the ending call again."))
                    continue
            if self.is_refuse(name, args) and self.left() > 6 and self.row_reads < 2:
                unread = self.rows_stating_needs(3)
                if unread and self.open_rows(unread):
                    self.row_reads += 1
                    self.trace.append({"kind": "row_leads", "rows": unread})
                    out.append((cid, "Not run yet (no step used). These rows' own text states a required value, so Kyo opened them; judge them before ending: " + ", ".join(unread)))
                    continue
            if self.is_refuse(name, args) and not self.leads and not self.pushed and self.left() > self.max_steps // 3:
                self.pushed = True
                out.append((cid, f"Not run yet (no step used). {self.left()} steps are still left, so search in ways you have not tried before ending: browse the category of each requested item and page through it with the token or offset the store returns, try single-word value queries and the property's other names, and open the listings that might fit. End without a purchase only after that."))
                continue
            if self.is_refuse(name, args) and not self.chain_on_decline and self.refs and self.left() > 3 and self.elapsed() < CLOCK_STOP:
                self.chain_on_decline = True
                before = len(self.opened)
                text, _ = self.trail()
                if len(self.opened) > before or self.version != self.near_version:
                    self.trace.append({"kind": "decline_chain", "opened": len(self.opened) - before})
                    if len(self.opened) > before:
                        out.append((cid, "Not run yet (no step used). The request has a value on another listing, so Kyo followed that chain first. Judge again with what it shows, and buy if a listing now fits.\n" + text))
                        continue
            if self.is_refuse(name, args) and self.version not in self.forced_at and len(self.forced_at) < 3 and self.left() >= 2 and self.elapsed() < CLOCK_STOP:
                self.forced_at.add(self.version)
                self.near_tried = True
                self.read_audit_rows()
                plan = self.unbroken_plan()
                done = self.place_order(plan, "these listings contradict no requirement, and a property a record leaves unstated is no proof of a misfit") if plan else ""
                if done:
                    out.append((cid, done + (" [episode finished]" if self.finished else "")))
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
                    text += "\n[the shopper's update, genuine and already in the ledger] " + meaning + ". Apply this change to your picks before checking out."
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
        if self.finished or self.broke or self.left() <= 0:
            return ""
        buy, refuse = self.roles.get("buy"), self.roles.get("refuse")
        fresh = self.cart_view is not None and self.cart_seen_at >= self.cart_at
        holding = bool(self.cart_ids()) if fresh else bool(self.cart_lines)
        if self.left() >= 2 and (not holding or self.over_budget()) and self.version not in self.forced_at and len(self.forced_at) < 4 and self.elapsed() < CLOCK_STOP + 15:
            self.forced_at.add(self.version)
            plan = self.unbroken_plan()
            done = self.place_order(plan, "the cart was empty or over the budget and these listings contradict no requirement") if plan else ""
            if self.finished or self.left() <= 0:
                return ""
            if done and again:
                return "the checkout was refused: " + done
            fresh = self.cart_view is not None and self.cart_seen_at >= self.cart_at
            holding = bool(self.cart_ids()) if fresh else bool(self.cart_lines)
        if holding and refuse and len(self.items) - len(self.cart_chosen()) >= 2:
            holding = False
            fresh = True
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

    def over_budget(self) -> bool:
        if not isinstance(self.budget, (int, float)) or self.brief.get("budget_scope") == "each":
            return False
        total = 0.0
        for ident, units in self.cart_lines.items():
            price = self.price_of(ident)
            if units and price is not None:
                total += price * units
        return total > self.budget

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
        self.work(LOOP_CALLS, SPEND_HARD)
        if not self.finished:
            turned = self.finish(again=True)
            if turned and not self.finished and not self.broke and not self.no_credit and self.left() > 1 and self.elapsed() < CLOCK_STOP + 15:
                self.chat.append({"role": "user", "content": "[Kyo] Kyo tried to end the episode and " + turned + "\nApply what the store says, then finish.\n" + self.status()})
                self.work(self.loop_calls + 10, SPEND_HARD + 0.01)
            if not self.finished:
                self.finish()
        self.trace.append({"kind": "end", "finished": self.finished, "steps": self.step, "cost": round(self.cost, 5), "calls": self.calls})
        return self.trace

    def elapsed(self) -> float:
        return time.monotonic() - self.began

    def work(self, loop_cap: int, spend_cap: float) -> None:
        idle = 0
        while not self.finished and not self.broke and self.left() > 0 and self.loop_calls < loop_cap and self.cost < spend_cap and self.elapsed() < CLOCK_STOP:
            if self.ready_to_ask():
                summary = self.interview()
                chain = self.early_chain()
                if summary or chain:
                    self.chat.append({"role": "user", "content": (summary or "") + ("\n" + chain if chain else "") + "\n" + self.status()})
                continue
            acted = self.deadline_step()
            if acted:
                self.chat.append({"role": "user", "content": "[Kyo] " + acted + "\n" + self.status()})
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
                self.chat.append({"role": "user", "content": "Your last reply made no tool call, so nothing happened. Reply with a tool call now.\n" + self.status()})
                if idle >= 5:
                    break
                continue
            idle = 0
            committing = {self.roles.get("cart"), self.roles.get("buy"), self.roles.get("refuse")} - {None}
            if not self.asked and self.roles.get("ask") and self.msgs_free() > 0 and any(c["function"]["name"] in committing for c in calls) and self.left() > 8:
                self.chat.append({"role": "assistant", "content": content or None, "tool_calls": calls})
                for call in calls:
                    self.chat.append({"role": "tool", "tool_call_id": call["id"], "content": "Not run (no step used): the shopper is asked first."})
                summary = self.interview()
                chain = self.early_chain()
                self.chat.append({"role": "user", "content": (summary or "The interviewer had nothing to ask.") + ("\n" + chain if chain else "") + "\n" + self.status()})
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
