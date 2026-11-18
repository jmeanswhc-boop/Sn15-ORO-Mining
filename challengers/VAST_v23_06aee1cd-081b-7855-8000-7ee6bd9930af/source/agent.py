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
HEAVY = "anthropic/claude-haiku-5.5"          # fixed-form calls (plan, hear, checks): level with the larger model on the benches at a fraction of the spend and latency
HEAVY_EFFORT = "low"      # reasoning effort for the heavy model's fixed-form calls; the provider default (high) costs the most time and tokens
SPARE = "openai/gpt-6-luna"

LOOP_CALLS = 42             # model turns of the tool loop at least; a longer step cap allows more (Kyo.loop_cap)
ALL_CALLS = 60              # model calls at least; a longer step cap allows more (Kyo.call_cap)
SPEND_SOFT = 0.055
SPEND_HARD = 0.078
SPEND_STOP = 0.085          # no model call goes out above this spend: the code finishes alone (one run key pays for every task)
SPEND_LIGHT = 0.035         # above this spend the late checks (cart check, decline review, swaps) try the light model first
# each task has 300 s in all on the network: Kyo keeps its own clock from the start of agent_main
DEADLINE = 250.0            # seconds Kyo plans to use, leaving room for one slow call and the finish
CLOSING = 70.0              # with less time than this left, the tool loop stops and the episode heads for its finish
LAST_CALL = 28.0            # with less time than this left, no model call starts: the code finishes alone
QUESTION_TIME = 12.0        # the interviewer asks while more time than CLOSING plus this is left (race 15: with 45 the slow store left 116/180 episodes unasked)
# waits before resending a failed model call; an episode has 300 s in all, so at most 45 s go to waiting
WAITS = (4, 10, 20, 35)
WAIT_TOTAL = 45
SHOW_CHARS = 450
OBS_LIMIT = 15000
HISTORY_LIMIT = 150000
GATE_ROUNDS = 3             # store calls the process gate may make before one checkout or ending
AUTO_PAGES = 6              # further search pages Kyo reads itself for a cheapest, most or least pick
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
- A wish in the request that names no value a listing could state ("legs cut comfortably", "should not stick out far") is not a requirement: the interviewer asks the shopper about that property, and when the answer gives no value, any listing meeting the stated requirements fits. Never decline or reject a listing over an unstated value you inferred yourself.
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


PLAN = """You choose the questions a shopping agent sends to a shopper. Shoppers hold a few firm preferences they did not write down, each about one property of one requested product. A question reveals a preference only when it names that exact property. A question may list up to seven properties of one product: the shopper answers about the first listed property they hold a preference on, and the chance of an answer falls the further down the list that property sits, so rank the properties by how likely a preference is. A settled requirement whose value is only a wish in words (no figure, no value listings use) is a hint that the shopper holds an exact preference on that property: list that property first.

You get the request, the items with what the request already settles for each, and a numbered list of attribute keys seen on opened listings of each item's category, with how many of those listings carry the key and example values. An unwritten preference may be on a property that no listed key names.

Choose the properties in this order:
0. What the request itself asks you to check with the shopper (listed under ASK FIRST when there is any), each named by the property it is about.
1. Hinted properties: when a sentence of the request points at one of the listed keys without giving its value, that key, for the item the sentence is about.
2. Then the listed keys in the order given (most carried first); avoid keys seen on only one or two of the item's listings.
Skip a property the request already settles for that item, a second name for a property already chosen, a key whose values list several purchase options at once, a key whose value differs on every listing (an identifier or code), and keys about selling terms rather than the product itself. With several items, take turns between the items.
"label" is the property's plain name: for a listed key, all of the key's words in plain spelling ("<word one> <word two>" for wordOneWordTwo or word_one_word_two; never shortened to one of its words); only the property's name, no example values, no brackets, no second property.
Choose COUNT, best first (fewer only when the lists hold fewer usable keys). Also return "slots": the item numbers the MESSAGES messages should go to, in sending order, one number per message: spend the messages where an unwritten preference is most likely and a wrong guess hurts most; an item the request already pins down, or whose listed keys carry nothing a buyer chooses by, may get none, and the item with the most open properties may get two in a row. Return ONE JSON object and nothing else: {"ask": [{"item": item number, "label": "...", "hint": "at most 8 of the story's words it comes from, or an empty string"}], "slots": [item number, ... MESSAGES of them]}"""


HEAR = """A shopping agent asked a shopper some questions. Read each answer and return ONE JSON object and nothing else:
{"needs": [{"item": item number, "need": "property: constraint", "points": true when the value is to be read from another listing, else false}], "drop": [{"item": item number, "property": "a requirement the shopper now gives up"}], "free": [{"item": item number, "property": "..."}], "crowded": [numbers of the questions whose answer complains that the message asked about too many things or wants a single question; not answers that only say they did not follow it], "replies": [{"question": number, "kind": "value|no_preference|confusion|crowded|restated|other", "property": "the property the answer speaks about, in the answer's own words, or null"}], "next": {"form": "list|single", "properties": ["property", ...]}, "budget": new total limit as a number or null}
- "replies": one entry per question, read from the answer alone: value = it states a preference; no_preference = they do not mind; confusion = they did not follow or ask what was meant; crowded = they complain the message asked too many things; restated = they say the request already fixed that property ("I told you at the start ..."); other = anything else.
- "next": your advice for the agent's next message to the same product, read from how this shopper answers: "form" is "list" when they answer a list of properties (a value or a plain no-preference came back), "single" when they want one thing at a time, did not follow a list, or repeat the request; "properties" are the properties worth asking next, likeliest first, none the shopper has settled, waved off or been asked about already (empty when nothing is left).
- An answer that turns down every value except one ("anything other than X gets a no", "nothing but X", "only X will do") states X as a positive need, never an exclusion; only an answer that rules a value out without naming the wanted one is "not <value>".
- A question that listed several properties is answered about the one the answer itself names: that property is the need's property.
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
Every attribute the request itself states about an item (material, size or count words, colour, type, model or edition words, what it is for) is a requirement even when the requirements list does not repeat it: a cart line that satisfies the shopper's answers but not one of the request's own words is not the best option, and a listing shown that meets all of them is better.
When the pick rule is cheapest, most or least, also look at the other listings shown: if one of them clearly fits every requirement and beats the cart's choice on that rule, the cart is wrong.
Return ONE JSON object and nothing else: {"ok": true or false, "problems": ["short, concrete problem", ...], "better": [listing ids that should replace a cart line, best first], "evidence": {"<listing id>": [{"requirement": "<a requirement in force on that line, as listed>", "key": "<the record's attribute key that decides it, or title / description, or null when no record text decides it>", "value": "<the deciding text copied exactly from the record, or null>"}]}}. Give one evidence row for EVERY requirement in force on each cart line (stated by the request and answered by the shopper), copying the key and value exactly from the record shown; a requirement with no deciding record text gets null and makes the cart not right. For an exclusion ("not X", "anything but X") cite the record's own value for that property (the row shows it is not X). A wish in words that names no value a listing could state needs no row. Use ok=false only for a problem the records clearly show or a requirement no record addresses."""


READING = """the listing's attribute for a property decides when its key names that same property (any key spelling, one element of a list is enough, case, plural, spelling variants and unit conversions count, a value containing all of the required value's words counts, a different word does not count even when it means nearly the same, limits include the edge); a requirement the request ties to a rule the store states is met by the value that rule gives; a key for a different property, even a related one (a key about one part of the product says nothing about another part), decides nothing; when no attribute names that property, the wanted value stated under another attribute or in the title or description counts (a bare figure that does not say what it measures counts for nothing); a requirement the category label itself states is met by that category; the category label must match the item; a rule of being unlike another listing's value counts as met unless the records show that same value; out-of-stock listings and listings that break the budget do not fit. A requirement counts as met only by the listing's own attribute for that property stating the value, exactly for a number (30.3 is not 33) or within the stated bound; a value found only in the title or description, or under another attribute, counts only when no listing seen states it under its own attribute. Between a listing that states every requirement in its table and one that leaves a requirement unstated or merely close, the first is the fit and the second is not the best option."""

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



def _vague(value: str) -> bool:
    """A requirement value that is a wish in words (no figure, more than three words): the shopper holds the exact preference, so the
    property is still worth a question."""
    v = str(value).strip()
    return not re.search(r"\d", v) and len(v.split()) > 3


def _roles_by_words(enders: list[str], tools: list[dict[str, Any]], got: dict[str, Any]) -> dict[str, Any]:
    """Last resort without a model: the ending tools told apart by the words of their names and descriptions."""
    text = {t["function"]["name"]: re.sub(r"[^a-z]+", " ", (t["function"]["name"] + " " + str(t["function"].get("description") or "")).lower()) for t in tools}
    buy_words = ("order", "buy", "checkout", "purchase", "settle", "pay", "confirm")
    no_words = ("abstain", "refuse", "declin", "give up", "cancel", "quit", "without")
    def side(name: str) -> str | None:
        words = " " + text.get(name, name.lower()) + " "
        b = any(w in words for w in buy_words); n = any(w in words for w in no_words)
        return "buy" if b and not n else ("refuse" if n and not b else None)
    sides = {n: side(n) for n in enders}
    buys = [n for n, s in sides.items() if s == "buy"]; nos = [n for n, s in sides.items() if s == "refuse"]
    if len(enders) == 2 and len(buys) + len(nos) == 1:              # one side named: the other tool is the other ending
        other = next(n for n in enders if n not in buys + nos)
        (nos if buys else buys).append(other)
    got = dict(got)
    if len(buys) == 1 and not got.get("buy"):
        got["buy"] = buys[0]
    if len(nos) == 1 and not got.get("refuse"):
        got["refuse"] = nos[0]
    return got

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
    """An identifier-looking value: a number, or a short token with a digit; under a key named like an id, any short token."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return True
    if not isinstance(value, str):
        return False
    v = value.strip()
    if not 2 <= len(v) <= 64:
        return False
    if named:
        return True
    return " " not in v and (bool(re.search(r"\d", v)) or bool(re.fullmatch(r"[A-Za-z]{1,4}[-_][A-Za-z0-9]{3,}", v)))


def _learn_id(rows: list[dict[str, Any]]) -> str | None:
    """The key that identifies rows: string values with a digit and no spaces, unique per row, longest on average."""
    best = None
    for key in rows[0]:
        values = [row.get(key) for row in rows]
        if not all(_id_like(v, "id" in str(key).lower()) for v in values):
            continue
        values = [str(v).strip() for v in values]
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


_TEXT_PAIR = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]{0,30}):\s(.*)$", re.S)


def _textval(raw: str) -> Any:
    if raw in ("None", "null", ""):
        return None
    if raw in ("True", "true"):
        return True
    if raw in ("False", "false"):
        return False
    if re.fullmatch(r"-?\d+", raw):
        return int(raw)
    if re.fullmatch(r"-?\d+\.\d+", raw):
        return float(raw)
    return raw


def _textrow(text: Any) -> dict[str, Any] | None:
    """A record the store serves as one text line ("ref_ref: p-1 | title: ... | price: 1099 | facts: colour=red; size=large | ...")
    read back into a dict, a "k=v; k=v" part into a nested dict; None for ordinary text (fewer than three "key: value" parts)."""
    if not isinstance(text, str) or " | " not in text or ": " not in text[:60]:
        return None
    out: dict[str, Any] = {}
    last: str | None = None
    for part in re.split(r"\s\|\s", text):
        m = _TEXT_PAIR.match(part)
        if m:
            key = m.group(1).strip().lower()
            raw = m.group(2).strip()
            pairs = [kv for kv in re.split(r";\s*", raw) if "=" in kv] if "=" in raw else []
            if pairs and (len(pairs) >= 2 or key in ("facts", "attributes", "specs", "details", "properties")) and all(len(kv.split("=", 1)[0]) <= 40 for kv in pairs):
                out[key] = {kv.split("=", 1)[0].strip().lower(): kv.split("=", 1)[1].strip() for kv in pairs}
            else:
                out[key] = _textval(raw)
            last = key
        elif last is not None:
            if isinstance(out[last], dict):
                for kv in re.split(r";\s*", part):
                    if "=" in kv:
                        k2, v2 = kv.split("=", 1)
                        out[last][k2.strip().lower()] = v2.strip()
            else:
                out[last] = f"{out[last]} | {part}" if out[last] is not None else part
    return out if len(out) >= 3 else None


FORMAT_MEMORY: dict[str, Any] = {}   # the record layout a model described during this run (one store serves a whole run; a fresh process knows nothing)

FORMAT_PROMPT = """A store answered a tool call with the text below. It describes one product listing or a list of listings in some text layout. Return ONE JSON object and nothing else:
{"records": [{"id": "the listing's identifier as written", "title": "its title as written", "price": a number or null, "fields": {"every other field name exactly as written": "its value as written"}, "attributes": {"attribute name as written": "its value as written"}}],
 "format": {"record_sep": "the exact text between two listings, or null when the text holds one listing", "field_sep": "the exact text between two fields", "kv_sep": "the exact text between a field's name and its value", "attributes_field": "the field whose value lists the listing's attributes, or null", "attr_sep": "the exact text between two attributes, or null", "attr_kv": "the exact text between an attribute's name and its value, or null"}}
Add "attributes_shape": one of "dict" (name: value inside one field), "pairs" (a list of name/value entries), "kv_string" (name=value items in one text), "prefixed" (attribute names carry a common prefix), "none"; and "rows_shape": one of "text" (each listing one text), "dicts", "arrays_with_header".
Copy every separator exactly, spaces included. Keep field and attribute names exactly as the text spells them. Read every listing the text holds."""


def _kv_lines(s: str) -> dict[str, Any] | None:
    """'key: value' or 'key = value' lines, one field a line (three or more of them)."""
    lines = [ln for ln in s.splitlines() if ln.strip()]
    if len(lines) < 3:
        return None
    out: dict[str, Any] = {}
    for ln in lines:
        m = re.match(r"^\s*[-*\u2022]?\s*([A-Za-z_][A-Za-z0-9_ ]{0,40}?)\s*[:=]\s*(.*)$", ln)
        if not m:
            continue
        key = m.group(1).strip().lower()
        raw = m.group(2).strip()
        pairs = [kv for kv in re.split(r";\s*", raw) if "=" in kv] if "=" in raw else []
        out[key] = {kv.split("=", 1)[0].strip().lower(): kv.split("=", 1)[1].strip() for kv in pairs} if len(pairs) >= 2 else _textval(raw)
    return out if len(out) >= 3 and len(out) >= 0.6 * len(lines) else None


def _decode_with(s: str, spec: dict[str, Any]) -> Any:
    """Read text with the separators a model described: {record_sep, field_sep, kv_sep, attributes_field, attr_sep, attr_kv}."""
    try:
        fs = str(spec.get("field_sep") or "")
        kv = str(spec.get("kv_sep") or "")
        if not fs or not kv or fs == kv:
            return None
        rs = spec.get("record_sep")
        chunks = [c for c in (s.split(str(rs)) if rs else [s]) if c.strip()]
        af = str(spec.get("attributes_field") or "").strip().lower()
        asep = str(spec.get("attr_sep") or "")
        akv = str(spec.get("attr_kv") or "")
        recs = []
        for chunk in chunks:
            out: dict[str, Any] = {}
            for part in chunk.split(fs):
                if kv not in part:
                    continue
                key, raw = part.split(kv, 1)
                key = key.strip().lower()
                raw = raw.strip()
                if not key or len(key) > 40:
                    continue
                if af and key == af and asep and akv:
                    out[key] = {p.split(akv, 1)[0].strip().lower(): p.split(akv, 1)[1].strip() for p in raw.split(asep) if akv in p}
                else:
                    out[key] = _textval(raw)
            if len(out) >= 3:
                recs.append(out)
        if not recs:
            return None
        return recs if (len(recs) > 1 or rs) else recs[0]
    except Exception:
        return None


def _decode_text(text: Any) -> Any:
    """Text-serialised record(s) read back into dict(s): JSON, 'k: v | k: v' lines, one-field-a-line text, then the layout a model
    described earlier in this run (FORMAT_MEMORY); None for ordinary prose such as a shopper's answer."""
    if not isinstance(text, str) or len(text) < 20:
        return None
    s = text.strip()
    spec = FORMAT_MEMORY.get("spec")
    if isinstance(spec, dict):   # the layout the model described for this store comes first
        got = _decode_with(s, spec)
        if got is not None:
            return got
    if s[:1] in "[{":
        try:
            got = json.loads(s)
            if isinstance(got, (dict, list)):
                return got
        except Exception:
            pass
    row = _textrow(s)
    if row is not None:
        return row
    row = _kv_lines(s)
    if row is not None:
        return row
    return None


def _clean_keys(rows: list[Any]) -> bool:
    """Field names a real layout produces: short, without JSON punctuation (a layout that splits a JSON dump yields keys with quotes/braces)."""
    bad = "{}[]" + chr(34)
    keys = [str(k) for r in rows if isinstance(r, dict) for k in r]
    return bool(keys) and all(len(k) <= 40 and not any(ch in k for ch in bad) for k in keys)


RAW_BY_ID: dict[str, str] = {}   # the store's own text of a record the code decoded, by its identifier (for the attribute follow-up)


def _remember_raw(decoded: Any, text: str) -> None:
    rec = decoded if isinstance(decoded, dict) else (decoded[0] if isinstance(decoded, list) and decoded and isinstance(decoded[0], dict) else None)
    if not isinstance(rec, dict):
        return
    for key, value in rec.items():
        if isinstance(value, str) and 2 <= len(value) <= 64 and " " not in value and (re.search(r"\d", value) or "id" in str(key).lower()):
            RAW_BY_ID[value.strip()] = text[:4000]
            if len(RAW_BY_ID) > 400:
                RAW_BY_ID.pop(next(iter(RAW_BY_ID)))
            return


def _split_pairs(text: str) -> dict[str, Any] | None:
    """Name/value items inside one field read by their own repeated separators (used only after the model's attribute follow-up
    failed): a name/value mark repeated two or more times and an item mark that splits the text into those items."""
    if not isinstance(text, str) or len(text) < 7 or len(text) > 3000:
        return None
    for kv in ("=", "~", ": ", "->", "=>", ":"):
        if text.count(kv) < 2:
            continue
        for sep in ("; ", " / ", " | ", ", ", chr(10), ";", "/"):
            items = [x.strip() for x in text.split(sep) if x.strip()]
            if len(items) < 2 or not all(kv in x for x in items):
                continue
            out = {}
            for x in items:
                name, value = x.split(kv, 1)
                name = name.strip().lower()
                if not name or len(name) > 30 or len(name.split()) > 3 or re.search(r"\d{4,}", name):
                    out = {}
                    break
                out[name] = value.strip()
            if len(out) >= 2:
                return out
    return None


def _unread_block(rec: Any) -> bool:
    """True when a record carries a string field that reads like an attribute block (repeated name/value marks) which no decoder has
    turned into a dict yet: the only situation worth a model question about the attribute layout. Search rows without a block are not."""
    if not isinstance(rec, dict) or any(isinstance(v, dict) and len(v) >= 2 for v in rec.values()):
        return False
    return any(isinstance(v, str) and len(v) >= 12 and _split_pairs(v) for v in rec.values())


def _structured_texts(node: Any, depth: int = 0) -> list[str]:
    """Every record-like text inside an observation (a page served as one line, rows served as lines), longest first."""
    out: list[str] = []
    if depth > 3:
        return out
    if isinstance(node, str):
        if _looks_structured(node):
            out.append(node)
    elif isinstance(node, dict):
        for v in node.values():
            out += _structured_texts(v, depth + 1)
    elif isinstance(node, list):
        for v in node[:6]:
            out += _structured_texts(v, depth + 1)
    return sorted(out, key=len, reverse=True)


def _looks_structured(text: str) -> bool:
    """Text that reads like a serialised record rather than prose: several field separators and a digit."""
    if not isinstance(text, str) or len(text) < 120 or text.lstrip()[:1] in "[{":
        return False
    marks = len(re.findall(r"\w\s*[:=]\s*\S", text)) + text.count(" | ") + text.count("\n")
    return marks >= 4 and bool(re.search(r"\d", text))


def _normalize_record(rec: dict[str, Any]) -> dict[str, Any]:
    """One record in the shape the code keeps: an attribute block as a nested dict whether it came as a dict, a list of {name, value}
    pairs, a 'k=v; k=v' string or flat prefixed keys (attr_colour …); rows given as arrays next to a header list become dicts; a price
    written as text ('$1,299.00') becomes a number; a numeric identifier becomes text."""
    out: dict[str, Any] = dict(rec)
    shape = str((FORMAT_MEMORY.get("spec") or {}).get("attributes_shape") or "")
    header = next((v for v in out.values() if isinstance(v, list) and len(v) >= 3 and all(isinstance(x, str) for x in v)), None)
    for key, value in list(out.items()):
        if isinstance(value, list) and value and all(isinstance(x, dict) for x in value) and len(value[0]) <= 4 and shape in ("", "pairs"):
            names = [k for k in value[0] if re.search(r"name|key|attr|prop|label|field|spec", str(k), re.I)]
            vals = [k for k in value[0] if k not in names and re.search(r"value|val|content|text|data", str(k), re.I)]
            if names and vals:
                out[key] = {str(x.get(names[0])).strip().lower(): x.get(vals[0]) for x in value if isinstance(x, dict) and x.get(names[0]) is not None}
        elif isinstance(value, list) and value and header is not None and all(isinstance(x, list) and len(x) == len(header) for x in value):
            out[key] = [dict(zip(header, row)) for row in value]
        elif isinstance(value, str) and _decode_text(value) is not None:
            out[key] = _decode_text(value)   # a record (or list of records) served as text under this key
            _remember_raw(out[key], value)
        elif isinstance(value, str) and value.count("=") >= 2 and len(value) < 3000 and " | " not in value and shape in ("", "kv_string", "dict"):
            pairs = [kv for kv in re.split(r";\s*|,\s*(?=[^=,;]{1,40}=)", value) if "=" in kv]
            if len(pairs) >= 2 and all(0 < len(kv.split("=", 1)[0].strip()) <= 40 for kv in pairs):
                out[key] = {kv.split("=", 1)[0].strip().lower(): kv.split("=", 1)[1].strip() for kv in pairs}
        elif isinstance(value, str) and re.search(r"price|cost|amount|total", str(key), re.I):
            m = re.fullmatch(r"\s*(?:[A-Z]{3}|[$€£¥₩])?\s*(-?\d{1,3}(?:[,\s]\d{3})*(?:\.\d+)?|-?\d+(?:\.\d+)?)\s*(?:[A-Z]{3}|[$€£¥₩])?\s*", value)
            if m:
                number = float(re.sub(r"[,\s]", "", m.group(1)))
                out[key] = int(number) if number.is_integer() else number
        elif isinstance(value, (int, float)) and not isinstance(value, bool) and re.search(r"(^|_)(id|code|ref|sku|number)$|identifier", str(key), re.I):
            out[key] = str(value)
    if (FORMAT_MEMORY.get("attr_tries", 0) >= 1 and not (FORMAT_MEMORY.get("spec") or {}).get("attributes_field")
            and not any(isinstance(v, dict) for v in out.values())):
        # the model's attribute follow-up named no field: the string field whose own repeated separators yield the most name/value items is the block
        found = [(len(pairs), key, pairs) for key, value in out.items() if isinstance(value, str) for pairs in [_split_pairs(value)] if pairs]
        if found:
            found.sort(key=lambda x: -x[0])
            out[found[0][1]] = found[0][2]
    groups: dict[str, dict[str, Any]] = {}
    for key, value in out.items():
        m = re.match(r"^(attr|attribute|spec|specs|prop|property|feature)[_\-:. ](.+)$", str(key), re.I)
        if m and not isinstance(value, (dict, list)):
            groups.setdefault(m.group(1).lower(), {})[m.group(2).strip().lower()] = value
    for prefix, table in groups.items():
        if len(table) >= 2 and shape in ("", "prefixed") and not any(isinstance(v, dict) and len(v) >= 2 for v in out.values()):
            out[prefix + "_table"] = table
    return out


def _decode_deep(node: Any, depth: int = 0) -> Any:
    """Text-serialised records anywhere inside an observation read back into structures (lists of lines, a page as one line)."""
    if depth > 5:
        return node
    if isinstance(node, str):
        got = _decode_text(node)
        return got if got is not None else node
    if isinstance(node, list):
        return [_decode_deep(x, depth + 1) for x in node]
    if isinstance(node, dict):
        return {k: _decode_deep(v, depth + 1) for k, v in node.items()}
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


SHOPPER_MEMORY: dict[str, int] = {}   # what this process's shopper did with earlier questions (one shopper answers a whole run; a fresh process knows nothing)


def shopper_memory_read() -> dict[str, Any]:
    """Counts of list questions this run's shopper answered or turned down so far (and one-property questions answered)."""
    return dict(SHOPPER_MEMORY)


def shopper_memory_note(key: str) -> None:
    """Count one more observation of the shopper's answering (lists_answered, lists_refused, singles_answered); never fails."""
    try:
        SHOPPER_MEMORY[key] = int(SHOPPER_MEMORY.get(key) or 0) + 1
    except Exception:
        pass


class Item:
    def __init__(self, name: str, shelf: str, qty: int, musts: list[str]):
        self.name = name
        self.shelf = shelf
        self.qty = qty
        self.musts = musts
        self.told: list[str] = []
        self.dropped: list[str] = []
        self.free: list[str] = []
        self.restated: list[str] = []   # properties the shopper said the request already fixed


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
        self.raw_texts: list[str] = []
        self.format_tries = 0
        self.learning_format = False
        self.store_secs: list[float] = []
        self.model_secs: list[float] = []
        self.dict_without_facts: dict[str, Any] | None = None
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

    def think(self, messages: list[dict[str, Any]], models: list[str], max_tokens: int, tools: list[dict[str, Any]] | None = None, reasoning: dict[str, Any] | None = None) -> dict[str, Any] | None:
        for model in models:
            if self.calls >= self.call_cap or self.cost >= SPEND_STOP:
                return None
            if self.time_left() < LAST_CALL:
                self.out_of_time = True
                return None
            self.calls += 1
            tries = 0
            while True:
                message, status = self.call_model(model, messages, max_tokens, tools, reasoning)
                if message is None and status == 0 and str((reasoning or {}).get("effort") or "") not in ("low", "none"):
                    # a reply with no content: the thinking used the whole completion; the same call once more with no thinking
                    message, status = self.call_model(model, messages, max_tokens, tools, {"effort": "none" if model.startswith("anthropic/") else "low"})
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

    def call_model(self, model: str, messages: list[dict[str, Any]], max_tokens: int, tools: list[dict[str, Any]] | None, reasoning: dict[str, Any] | None = None) -> tuple[dict[str, Any] | None, Any]:
        """One inference call: the reply message, or None and the failure's status (0 for a reply without content)."""
        sent = messages
        if model.startswith("anthropic/") and len(messages) > 1:
            sent = [dict(m) for m in messages]
            marks = [0] + ([len(sent) - 1] if len(sent) > 1 else [])
            for index in marks:
                content = sent[index].get("content")
                if isinstance(content, str) and content:
                    sent[index]["content"] = [{"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}]
        if reasoning is None and model in (HEAVY, LIGHT) and HEAVY_EFFORT and not tools:   # fixed-form calls only (both are Haiku here); the walker keeps the default
            reasoning = {"effort": HEAVY_EFFORT}
        body: dict[str, Any] = {"model": model, "messages": sent, "max_tokens": max_tokens, "temperature": 0}
        if reasoning:
            body["reasoning"] = reasoning
        if str((reasoning or {}).get("effort") or "") != "none":
            body["max_tokens"] = max(max_tokens, 2500)      # the model thinks by default: a short completion can be used up by the thinking alone
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        try:
            started = time.monotonic()
            result = _llm.post_verbose("/inference/chat/completions", body)
            self.model_secs = (getattr(self, "model_secs", []) + [time.monotonic() - started])[-12:]
        except Exception as exc:                              # a client that raises (network down, dead key) is a failed call, not the episode's end
            self.trace.append({"kind": "llm_fail", "model": model, "status": None, "body": f"{type(exc).__name__}: {exc}"[:200]})
            return None, None
        data = result.data
        if isinstance(data, dict) and data.get("choices"):
            usage = data.get("usage") or {}
            paid = usage.get("cost")
            if not paid and isinstance(usage.get("cost_details"), dict):
                paid = usage["cost_details"].get("upstream_inference_cost")   # a BYOK run key: the proxy reports cost 0 and the real charge here, so the spend caps bind
            if isinstance(paid, (int, float)):
                self.cost += float(paid)
            message = data["choices"][0].get("message") or {}
            if message.get("content") or message.get("tool_calls"):
                return message, 200
            details = usage.get("completion_tokens_details") if isinstance(usage.get("completion_tokens_details"), dict) else {}
            self.trace.append({"kind": "llm_fail", "model": model, "status": 0, "finish": data["choices"][0].get("finish_reason"), "reasoning_tokens": details.get("reasoning_tokens")})
            return None, 0
        error = result.error or {}
        status = error.get("status")
        if isinstance(data, dict) and isinstance(data.get("error"), dict):
            # an error carried inside a 200 reply, with the provider's own code
            status = _count(data["error"].get("code"))
            error = {"body": data["error"].get("message")}
        self.trace.append({"kind": "llm_fail", "model": model, "status": status, "body": str(error.get("body") or "")[:200]})
        return None, status

    def ask_json(self, prompt: str, models: list[str], max_tokens: int = 1800, reasoning: dict[str, Any] | None = None) -> dict[str, Any] | None:
        for model in models:
            message = self.think([{"role": "user", "content": prompt}], [model], max_tokens, reasoning=reasoning)
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
            started = time.monotonic()
            result = _env.post_verbose("/environment/call", payload)
            self.store_secs = (getattr(self, "store_secs", []) + [time.monotonic() - started])[-12:]
            if isinstance(result.data, dict):
                data = result.data
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
        if limit is None:
            found = re.search(r"(\d+)\s+(?:messages?|questions?|replies|times|turns)\b", self.request, re.I)
            limit = int(found.group(1)) if found else None
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

        self.roles_from_shape()   # the finishing roles when the brief named none (a dead key, a network fault): read off the schemas, so the episode still ends with a store call

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

    def roles_from_shape(self) -> None:
        """The finishing roles when the brief named none (or a tool the store does not have): read off the schemas, since a
        world's names are new every race. A tool that takes no free text and no listing id (nothing, or only enums, numbers and
        booleans) ends the episode. One such tool with a two-way enum is both roles, its modes named by a model call over the
        schema; two such tools are told apart the same way; a lone argument-less tool is the order."""
        if self.roles.get("buy") and self.roles.get("refuse"):
            return
        taken = {v for k, v in self.roles.items() if k not in ("buy", "refuse") and v}
        plain = []
        for name, schema in self.schemas.items():
            props = schema.get("properties") or {}
            free = any(isinstance(v, dict) and "string" in _kinds(v) and "enum" not in v for v in props.values())
            if name in taken or free:
                continue
            plain.append(name)
        if not plain:
            return
        self.trace.append({"kind": "roles_from_shape", "candidates": plain})
        if len(plain) == 1:
            name = plain[0]
            self.roles["buy"] = self.roles.get("buy") or name
            self.roles["refuse"] = self.roles.get("refuse") or name
            self.resolve_modes(name)
            return
        listing = "\n".join(f"- {n}: {_dump(self.schemas.get(n))}" for n in plain)
        ask = ("These store tools take no free text and no listing id, so they end the task. Which one checks out (buys the basket) and which ends "
               "without a purchase? Return ONE JSON object and nothing else: {\"buy\": tool name or null, \"refuse\": tool name or null, "
               "\"buy_args\": the fixed arguments that make the buy tool check out, else {}, \"refuse_args\": the fixed arguments that end without a purchase, else {}}\n" + listing)
        got = self.ask_json(ask, [LIGHT, HEAVY], 500) or {}
        if not (got.get("buy") or got.get("refuse")):
            got = _roles_by_words(plain, self.tools, got)       # no model answered: the names and descriptions decide
        for role in ("buy", "refuse"):
            tool = got.get(role)
            if isinstance(tool, str) and tool in plain and not self.roles.get(role):
                self.roles[role] = tool
        for key in ("buy_args", "refuse_args"):
            if isinstance(got.get(key), dict) and not self.brief.get(key):
                self.brief[key] = got[key]
        if self.roles.get("buy") and self.roles.get("buy") == self.roles.get("refuse"):
            self.resolve_modes(self.roles["buy"])

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
        """A listing's id, read with the id key learned from result rows (or named by the structure learner); None for anything else."""
        if not isinstance(node, dict) or len(node) < 3:
            return None
        if not self.id_key and FORMAT_MEMORY.get("id_key") in node and _id_like(node.get(FORMAT_MEMORY["id_key"]), True):
            self.id_key = str(FORMAT_MEMORY["id_key"])
        if self.id_key and _id_like(node.get(self.id_key), True):
            return str(node[self.id_key]).strip()
        nested = any(isinstance(v, dict) for v in node.values())
        if not (in_list or depth == 0 or nested):
            return None
        found = _learn_id([node])
        if found and ("id" in str(found).lower() or str(node[found]).strip() in self.rows):
            return str(node[found]).strip()
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
            if isinstance(node, str):   # a record served as text (race 15: one 'k: v | k: v' line): the model describes the layout first, code reads with it
                if _looks_structured(node) and not FORMAT_MEMORY.get("spec") and FORMAT_MEMORY.get("asked", 0) < 2 and not getattr(self, "learning_format", False):
                    self.learn_layout(node)
                row = _decode_text(node)
                spec_now = FORMAT_MEMORY.get("spec")
                if (isinstance(row, dict) and isinstance(spec_now, dict) and not spec_now.get("attributes_field") and not _fact_block(_normalize_record(row))
                        and FORMAT_MEMORY.get("attr_tries", 0) < 2 and not getattr(self, "learning_format", False)
                        and _unread_block(row)):
                    FORMAT_MEMORY["attr_tries"] = FORMAT_MEMORY.get("attr_tries", 0) + 1   # a page read without an attribute block: ask once which field holds it
                    self.learning_format = True
                    try:
                        self.learn_attributes(node, spec_now)
                    finally:
                        self.learning_format = False
                    row = _decode_text(node)
                if row is not None:
                    walk(row, depth, in_list)
                elif len(node) >= 150 and depth <= 3:
                    self.note_raw(node)
                return
            if isinstance(node, dict):
                node = _normalize_record(node)
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
                    spec_now = FORMAT_MEMORY.get("spec")
                    if (not _fact_block(node) and isinstance(spec_now, dict) and not spec_now.get("attributes_field") and ident in RAW_BY_ID
                            and FORMAT_MEMORY.get("attr_tries", 0) < 2 and not getattr(self, "learning_format", False)
                            and _unread_block(node)):
                        FORMAT_MEMORY["attr_tries"] = FORMAT_MEMORY.get("attr_tries", 0) + 1   # a page read without an attribute block: ask once which field holds it
                        self.learning_format = True
                        try:
                            self.learn_attributes(RAW_BY_ID[ident], spec_now)
                        finally:
                            self.learning_format = False
                        again = _decode_text(RAW_BY_ID[ident])
                        if isinstance(again, dict):
                            node = _normalize_record(again)
                    self.rows[ident].update(node)
                    if _fact_block(node):
                        self.full[ident] = dict(self.rows[ident])
                    elif ident in self.full:
                        self.full[ident].update(node)
                    elif len(node) >= 6 and depth <= 1 and self.id_key and self.id_key in node:
                        self.dict_without_facts = node       # a page-sized listing record (the store's own id key) with no attribute block the code can read
                    return
                for value in node.values():
                    walk(value, depth + 1)
            elif isinstance(node, list):
                node = [(_decode_text(x) if isinstance(x, str) else None) or x for x in node]
                rows = self.learn_rows(node)
                for value in node:
                    walk(value, depth + 1, rows)

        walk(observation, 0)
        self.maybe_learn_format()

    def note_raw(self, text: str) -> None:
        """Long store text no decoder could read: kept for the format learner and as the planner's evidence of last resort."""
        self.raw_texts = (getattr(self, "raw_texts", []) + [text[:4000]])[-4:]

    def raw_hint(self) -> str:
        """What the planner sees when no listing record could be parsed: the store's own text of the last listings."""
        raws = getattr(self, "raw_texts", [])
        if not raws:
            return "(no full listing records yet)"
        self.trace.append({"kind": "raw_hint", "n": len(raws)})
        return ("(no listing record could be parsed from the store's replies; the store's own text of the last listings follows: read the attribute names it states)\n"
                + "\n---\n".join(r[:1500] for r in raws[-2:]))

    def flatten_model_record(self, rec: dict[str, Any]) -> dict[str, Any] | None:
        """A record as the format learner returns it -> the flat dict the record store keeps (id, title, price, other fields, attributes)."""
        ident = str(rec.get("id") or "").strip()
        if not ident:
            return None
        flat: dict[str, Any] = {"id": ident}
        if rec.get("title") is not None:
            flat["title"] = rec.get("title")
        if rec.get("price") is not None:
            flat["price"] = rec.get("price")
        fields = rec.get("fields") if isinstance(rec.get("fields"), dict) else {}
        for key, value in fields.items():
            flat[str(key).strip().lower()] = value
        attrs = rec.get("attributes") if isinstance(rec.get("attributes"), dict) else {}
        if attrs:
            flat["attributes"] = {str(k).strip().lower(): v for k, v in attrs.items()}
        return flat if len(flat) >= 3 else None

    def learn_layout(self, sample: str) -> bool:
        """One light call on the first record-like text of the run: the model returns the layout (separators, attribute and row shapes)
        and the records it read; the layout is kept for the run when it reads the sample back, else the code's decoders take over."""
        FORMAT_MEMORY["asked"] = FORMAT_MEMORY.get("asked", 0) + 1
        self.learning_format = True
        try:
            got = self.ask_json(FORMAT_PROMPT + "\n\nSTORE TEXT:\n" + sample[:3000], [LIGHT, HEAVY], 2000) or {}
            spec = got.get("format") if isinstance(got.get("format"), dict) else None
            if spec:
                for key in ("attributes_shape", "rows_shape"):
                    if got.get(key):
                        spec[key] = str(got.get(key))
                read = _decode_with(sample, spec)
                rows = read if isinstance(read, list) else ([read] if isinstance(read, dict) else [])
                if rows and _clean_keys(rows) and any(_learn_id([r]) or any(_id_like(v, True) and "id" in str(k).lower() for k, v in r.items()) for r in rows if isinstance(r, dict)):
                    FORMAT_MEMORY["spec"] = spec
                    FORMAT_MEMORY["stale"] = False
                    if not spec.get("attributes_field") and str(spec.get("attributes_shape") or "") != "none":
                        self.learn_attributes(sample, spec)   # the attribute block matters most to the planner: one focused call when the layout left it out
                    self.trace.append({"kind": "format_learned", "first": True, "spec": str(spec)[:240], "sample": sample[:100]})
                    return True
            self.trace.append({"kind": "format_fallback", "n": len(_items(got.get("records")))})
            for rec in _items(got.get("records")):
                flat = self.flatten_model_record(rec) if isinstance(rec, dict) else None
                if flat:
                    self.absorb(flat)
            return False
        finally:
            self.learning_format = False

    def learn_attributes(self, sample: str, spec: dict[str, Any]) -> None:
        """A layout whose attribute block the model did not name: one focused call asks which field lists the listing's attributes and
        what separates the items and names from values; kept only when it reads two or more attributes from the sample."""
        prompt = ("A store serves each product listing as text. Its fields are separated by " + repr(spec.get("field_sep")) + " and a field's name and value by " + repr(spec.get("kv_sep")) + ". "
                  "One field holds the listing's ATTRIBUTES (several name/value items such as colour, size, material). Return ONE JSON object and nothing else: "
                  "{\"attributes_field\": \"that field's name exactly as written, or null when no field lists attributes\", \"attr_sep\": \"the exact text between two attribute items\", \"attr_kv\": \"the exact text between an attribute's name and its value\"}\n\nLISTING TEXT:\n" + sample[:2500])
        got = self.ask_json(prompt, [LIGHT, HEAVY], 600) or {}
        trial = dict(spec)
        for key in ("attributes_field", "attr_sep", "attr_kv"):
            if got.get(key) is not None:
                trial[key] = str(got.get(key))
        read = _decode_with(sample, trial)
        rows = read if isinstance(read, list) else ([read] if isinstance(read, dict) else [])
        if any(isinstance(r, dict) and any(isinstance(v, dict) and len(v) >= 2 for v in r.values()) for r in rows):
            spec.update({k: trial[k] for k in ("attributes_field", "attr_sep", "attr_kv") if k in trial})
            self.trace.append({"kind": "attributes_learned", "field": spec.get("attributes_field"), "sep": spec.get("attr_sep"), "kv": spec.get("attr_kv")})
        else:
            self.trace.append({"kind": "attributes_failed", "got": str(got)[:160]})

    def maybe_learn_format(self) -> None:
        """A store whose records no decoder reads: a light model reads one such text into records and describes its separators; the code
        then reads the rest of this run with that description (FORMAT_MEMORY), asking again only when a description fails."""
        raws = [r for r in getattr(self, "raw_texts", []) if _looks_structured(r)]
        json_sample = False
        if not raws and getattr(self, "dict_without_facts", None) and not self.full:
            raws = [_dump(self.dict_without_facts)[:3000]]   # a record the code identified but whose attributes it could not read
            json_sample = True
        if not raws or self.full or getattr(self, "learning_format", False) or getattr(self, "format_tries", 0) >= 2:
            return
        if FORMAT_MEMORY.get("spec") and not FORMAT_MEMORY.get("stale"):
            return                                         # a learned layout exists and _decode_text already tried it on this text
        if FORMAT_MEMORY.get("failed", 0) >= 4:
            return
        self.learning_format = True
        self.format_tries = getattr(self, "format_tries", 0) + 1
        try:
            sample = raws[-1][:3000]
            got = self.ask_json(FORMAT_PROMPT + "\n\nSTORE TEXT:\n" + sample, [LIGHT, HEAVY], 2000) or {}
            spec = got.get("format") if isinstance(got.get("format"), dict) else None
            records = [r for r in _items(got.get("records")) if isinstance(r, dict)]
            read_v = _decode_with(sample, spec) if spec else None
            if spec and not json_sample and read_v is not None and _clean_keys(read_v if isinstance(read_v, list) else [read_v]):
                FORMAT_MEMORY["spec"] = spec
                FORMAT_MEMORY["stale"] = False
                self.trace.append({"kind": "format_learned", "spec": str(spec)[:240]})
                for raw in list(raws):
                    decoded = _decode_text(raw)
                    if decoded is not None:
                        self.absorb(decoded)
            else:
                FORMAT_MEMORY["failed"] = FORMAT_MEMORY.get("failed", 0) + 1
                if FORMAT_MEMORY.get("spec"):
                    FORMAT_MEMORY["stale"] = True
                self.trace.append({"kind": "format_records_only", "n": len(records)})
                for rec in records:
                    flat = self.flatten_model_record(rec)
                    if flat:
                        self.absorb(flat)
        finally:
            self.learning_format = False


    def fill_required(self, tool: str | None, args: dict[str, Any], kind: str) -> dict[str, Any]:
        """Required arguments of a finishing tool the brief did not fix: an enum takes its first option, a free text takes the code's own
        one-line account of the ending, a number 0, a flag true, a list [] (a store that wants a justification gets one)."""
        schema = self.schemas.get(tool or "") or {}
        props = schema.get("properties") or {}
        out = dict(args)
        for key in schema.get("required") or []:
            if key in out and out[key] not in (None, ""):
                continue
            spec = props.get(key) or {}
            kinds = _kinds(spec) if isinstance(spec, dict) else []
            if isinstance(spec, dict) and isinstance(spec.get("enum"), list) and spec["enum"]:
                out[key] = spec["enum"][0]
            elif "string" in kinds or not kinds:
                out[key] = self.finish_text(kind)
            elif "integer" in kinds or "number" in kinds:
                out[key] = 0
            elif "boolean" in kinds:
                out[key] = True
            elif "array" in kinds:
                out[key] = []
            elif "object" in kinds:
                out[key] = {}
        return out

    def finish_text(self, kind: str) -> str:
        """A plain account of the ending for a store that asks for one."""
        if kind == "buy":
            lines = [f"{ident} x{count}" for ident, count in (self.cart_lines or {}).items()] or list(self.cart_ids())
            return ("Checking out " + (", ".join(lines) if lines else "the cart") + ": each line is the listing whose attributes state the request's requirements and the shopper's answers.")[:400]
        return "Ending without a purchase: no listing seen states every requirement of the request, so nothing is bought."

    def learn_reply_key(self, observation: Any, question: str) -> None:
        """A reply whose shape leaves several candidate texts: one light call names the field that holds the shopper's words (kept for the run)."""
        if FORMAT_MEMORY.get("reply_key") or FORMAT_MEMORY.get("reply_key_tries", 0) >= 2 or not isinstance(observation, dict):
            return
        FORMAT_MEMORY["reply_key_tries"] = FORMAT_MEMORY.get("reply_key_tries", 0) + 1
        prompt = ("A shopping agent asked a shopper a question and the store returned the JSON below. Return ONE JSON object and nothing else: "
                  "{\"reply_key\": \"the top-level key whose value is the shopper's own answer\", \"notice_keys\": [\"keys whose values are the store's or shopper's side notices, if any\"]}\n\n"
                  "QUESTION:\n" + question[:300] + "\n\nREPLY JSON:\n" + _dump(_clip(observation, 1200))[:3000])
        got = self.ask_json(prompt, [LIGHT, HEAVY], 600) or {}
        key = str(got.get("reply_key") or "").strip()
        if key and key in observation and isinstance(observation.get(key), str):
            FORMAT_MEMORY["reply_key"] = key
            self.trace.append({"kind": "reply_key_learned", "key": key})

    def learn_cart_structure(self, observation: Any) -> dict[str, Any] | None:
        """A cart reply the code cannot read: one light call names the list of lines and the id and count fields (kept for the run)."""
        if FORMAT_MEMORY.get("cart_tries", 0) >= 2:
            return None
        FORMAT_MEMORY["cart_tries"] = FORMAT_MEMORY.get("cart_tries", 0) + 1
        prompt = ("A store's cart tool returned the JSON below (the cart after a change, or a view of it). Return ONE JSON object and nothing else: "
                  "{\"lines_key\": \"the key whose value lists the cart's lines (dotted path when nested, e.g. basket.items), or null\", "
                  "\"id_field\": \"the field of a line that holds the listing identifier\", \"qty_field\": \"the field of a line that holds the quantity\", "
                  "\"total_key\": \"the key holding the cart's total, or null\"}\n\nCART JSON:\n" + _dump(_clip(observation, 1500))[:3500])
        got = self.ask_json(prompt, [LIGHT, HEAVY], 600) or {}
        spec = {k: (str(got.get(k)).strip() if got.get(k) is not None else None) for k in ("lines_key", "id_field", "qty_field", "total_key")}
        if spec["id_field"] and spec["qty_field"]:
            FORMAT_MEMORY["cart"] = spec
            self.trace.append({"kind": "cart_structure_learned", "spec": str(spec)[:200]})
            return spec
        return None

    def cart_by_spec(self, observation: Any, spec: dict[str, Any]) -> str:
        """Read the cart lines with a learned structure: 'synced' when the lines were read, else 'unreadable'."""
        node: Any = observation
        path = [p for p in str(spec.get("lines_key") or "").split(".") if p]
        for part in path:
            node = node.get(part) if isinstance(node, dict) else None
        if node is None and not path:
            node = next((v for v in (observation.values() if isinstance(observation, dict) else []) if isinstance(v, list)), None)
        if not isinstance(node, list) or not all(isinstance(r, dict) for r in node):
            return "unreadable"
        out: dict[str, int] = {}
        for row in node:
            ident = row.get(spec["id_field"]); count = _count(row.get(spec["qty_field"]))
            if ident is None or count is None:
                return "unreadable"
            if count > 0:
                out[str(ident).strip()] = count
        self.cart_lines = out
        self.cart_view = observation
        self.cart_seen_at = self.step
        self.cart_shows_lines = True
        return "synced"

    def sync_cart(self, observation: Any, name: str | None = None, args: dict[str, Any] | None = None) -> str:
        """The cart from the store's reply in whatever shape it comes: decoded text, the code's own reading, then the structure a
        model described for this run (asked once when the code cannot read a reply that plainly lists lines)."""
        observation = _decode_deep(observation)
        state = self.sync_cart_core(observation, name, args)
        if state == "synced" or not isinstance(observation, dict):
            return state
        spec = FORMAT_MEMORY.get("cart")
        if isinstance(spec, dict) and self.cart_by_spec(observation, spec) == "synced":
            return "synced"
        has_list = any(isinstance(v, list) and v and all(isinstance(x, dict) for x in v) for v in observation.values()) or any(
            isinstance(v, dict) and any(isinstance(x, list) and x and all(isinstance(y, dict) for y in x) for x in v.values()) for v in observation.values())
        if state == "unreadable" or (state == "none" and has_list):
            spec = self.learn_cart_structure(observation)
            if spec and self.cart_by_spec(observation, spec) == "synced":
                return "synced"
        return state

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

    def q_time(self) -> float:
        """Time one more question needs on this store: a store call and a model reading, measured in this episode (QUESTION_TIME at the least)."""
        store = sorted(getattr(self, "store_secs", []) or [1.0])
        model = sorted(getattr(self, "model_secs", []) or [4.0])
        return max(QUESTION_TIME, 2.5 * (store[len(store) // 2] + model[len(model) // 2]))

    def ready_to_ask(self) -> bool:
        if self.asked or not self.roles.get("ask") or self.msgs_free() <= 0:
            return False
        if self.left() <= 8:
            return False
        opened = len(self.full)
        if opened >= 2 * len(self.items) + 1:
            return True
        if DEADLINE - self.time_left() >= 45.0 and (opened or self.step >= 4):
            return True                                    # a slow store: the questions come before the clock runs the interview out
        return self.step >= max(6, self.max_steps // 5)

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
        if count <= 0 or self.time_left() <= CLOSING + self.q_time() or not self.roles.get("ask"):
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
        # more picks than messages: each item keeps a queue of its own, so a later turn has a next property to ask
        want = 7 * count + len(self.items) + 2     # up to seven properties a message
        prompt = PLAN.replace("COUNT", str(want)).replace("MESSAGES", str(count)) + "\n\nREQUEST:\n" + self.request + "\n\nITEMS:\n" + "\n".join(settled)
        first = [str(x) for x in _items(self.brief.get("ask_first")) if str(x).strip()]
        if first:
            prompt += "\n\nASK FIRST (the request asks to be asked about these):\n" + "\n".join(first)
        prompt += "\n\nATTRIBUTE KEYS SEEN:\n" + ("\n".join(listing) or self.raw_hint())
        plan = self.ask_json(prompt, [HEAVY, LIGHT], 1800) or {}
        # the planner's split of the messages between the items (one item number per message); next_pick's turn rule is the fallback
        slots = [n for n in (_count(x) for x in _items(plan.get("slots"))) if n and 1 <= n <= len(self.items)][:count] if len(self.items) > 1 else []
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
        # a question names a property the listings carry: a label matching no listed key (and not asked for by the request) is dropped
        # while listed ones remain, so the shopper hears the store's own vocabulary (race 15: 'dimensions' drew nothing where 'width' did)
        key_names = [_words(name) for _, name, _, _ in table]
        if key_names:
            def listed(label: str) -> bool:
                plain = _words(label)
                if any(plain in str(x).lower() or str(x).lower() in plain for x in first):
                    return True
                stems = {w[:4] for w in plain.split() if len(w) >= 3}
                return any(stems & {w[:4] for w in k.split() if len(w) >= 3} for k in key_names)
            for number, queue in queues.items():
                kept = [lab for lab in queue if listed(lab)]
                if kept:
                    queues[number] = kept + [lab for lab in queue if lab not in kept][: max(0, 2 - len(kept))]
        for number in range(1, len(self.items) + 1):
            # every item's queue ends with its category's most carried keys the plan did not name, so no message is
            # left unused for want of a property (an item the plan left out gets these alone)
            queue = queues.setdefault(number, [])
            for name in self.item_keys(number, table, 21):
                if all(_words(name) != _words(other) for other in queue):
                    queue.append(name)
                    self.topup.add(name)
            if queue and number not in order:
                order.append(number)
        self.trace.append({"kind": "plan", "ask": _clip(plan.get("ask"), 120), "queues": _clip(queues, 60), "slots": slots})
        sent: list[tuple[int, str, str]] = []
        told: list[str] = []
        found: set[int] = set()
        last_item: int | None = None
        one_only = False          # the shopper turned a list down: one property a message from then on
        memory = shopper_memory_read()
        if int(memory.get("lists_refused") or 0) >= 2 and int(memory.get("lists_answered") or 0) == 0:
            # this run's shopper has turned lists down before and never answered one: one property a message from the first question
            one_only = True
            self.trace.append({"kind": "shopper_memory", "form": "singles", "memory": _clip(memory, 80)})
        sent_count = 0
        probe_single = False      # the model's advice: one property next
        probe_next = False        # a list met confusion in a bundle: the next message is a one-property probe
        probing = False           # the message just sent was that probe
        while self.msgs_free() > 0 and not self.finished and self.time_left() > CLOSING + self.q_time():
            pick = None
            if slots:
                # the planner's slot for this message when that item still has properties to ask; otherwise the turn rule decides
                pick = self.next_pick([slots[min(sent_count, len(slots) - 1)]], queues, found, last_item)
            if pick is None:
                pick = self.next_pick(order, queues, found, last_item)
            if pick is None:
                break
            number, label = pick
            labels = [label]
            if not (one_only or probe_single or probe_next):
                # race 14: a message naming six or seven ranked properties revealed a need on a third of the questions, one property a message on a fifth
                queue = queues.get(number) or []
                labels += [queue.pop(0) for _ in range(min(6, len(queue)))]
            text = self.question_list(number, labels)
            batch = self.send([(number, " / ".join(labels), text)])
            probing, probe_next = probe_next, False
            if not batch:
                break
            last_item = number
            self.asked_labels.extend((number, lab) for lab in labels)
            if self.msg_refused:
                # the store refused the message (its real limit is lower than the one read): no more questions
                self.msg_used -= 1
                self.msg_limit = self.msg_used
                break
            sent += batch
            sent_count += 1
            # one answer: the light model reads it (the heavy one when the light one fails); an answer that wants a
            # single question to a one-property question is the shopper's way of not minding it: move on
            heard = self.hear(batch, (LIGHT, HEAVY))
            before = len(told)
            self.take_heard(heard, told)
            if len(told) > before:
                found.add(number)
            # the reply's kind steers the next message (no persona rule is assumed beforehand)
            first = next((r for r in _items(heard.get("replies")) if isinstance(r, dict)), {})
            kind = str(first.get("kind") or "")
            if len(labels) > 1 and (kind in ("value", "no_preference") or len(told) > before):
                shopper_memory_note("lists_answered")
            elif len(labels) > 1 and (kind == "crowded" or _items(heard.get("crowded")) or (kind == "confusion" and len(self.items) == 1)):
                shopper_memory_note("lists_refused")
            elif len(labels) == 1 and kind in ("value", "no_preference"):
                shopper_memory_note("singles_answered")
            item = self.items[number - 1]
            advice = heard.get("next") if isinstance(heard.get("next"), dict) else {}
            probe_single = str(advice.get("form") or "") == "single" and not one_only      # the model's advice for the next message
            wanted = [_words(str(x)) for x in _items(advice.get("properties")) if str(x).strip()]
            if wanted and queues.get(number):
                front = [lab for lab in queues[number] if any(_words(lab) == w or _names(_stem(_words(lab)), _stem(w)) for w in wanted)]
                queues[number] = front + [lab for lab in queues[number] if lab not in front]
            if len(labels) > 1 and (_items(heard.get("crowded")) or kind == "crowded"):
                one_only = True                            # the list was turned down unanswered: its properties come back, one a message
                queues[number] = labels + (queues.get(number) or [])
                self.asked_labels = [(n, lab) for n, lab in self.asked_labels if not (n == number and lab in labels)]
            elif len(labels) > 1 and kind == "restated":
                # the shopper repeated a requirement the request already fixed: that property is settled, the other listed ones are asked again
                named = _words(str(first.get("property") or ""))
                hit = [lab for lab in labels if named and (_words(lab) == named or _names(_stem(_words(lab)), _stem(named)) or _names(_stem(named), _stem(_words(lab))))] or ([labels[0]] if not named else [])
                item.restated.extend(hit)
                rest = [lab for lab in labels if lab not in hit]
                queues[number] = rest + (queues.get(number) or [])
                self.asked_labels = [(n, lab) for n, lab in self.asked_labels if not (n == number and lab in rest)]
                self.trace.append({"kind": "restated", "item": number, "property": named, "settled": hit})
            elif probing:
                probing = False
                if kind in ("value", "no_preference") or len(told) > before or _items(heard.get("free")):
                    one_only = True                        # one property drew an answer where the list drew none
                    self.trace.append({"kind": "probe", "verdict": "singles"}); shopper_memory_note("lists_refused")
                else:
                    self.trace.append({"kind": "probe", "verdict": "lists"}); shopper_memory_note("lists_answered")
            elif len(labels) > 1 and kind in ("confusion", "other") and len(told) == before and len(self.items) > 1:
                probe_next = True                          # a bundle: one cheap probe tells list-refusal from "no listed property is a need"
            elif len(labels) > 1 and kind in ("confusion", "other") and len(told) == before and len(self.items) == 1:
                # one item: a shopper who takes lists answers one with a value or no preference, never confusion; so this one wants one property a message
                one_only = True
                queues[number] = labels + (queues.get(number) or [])
                self.asked_labels = [(n, lab) for n, lab in self.asked_labels if not (n == number and lab in labels)]
                self.trace.append({"kind": "probe", "verdict": "singles"}); shopper_memory_note("lists_refused")
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
        """The next question: the item after the one asked last that still has a property queued, items whose answers
        revealed nothing yet first. A property an answer has already settled for that item is not asked again."""
        for number, queue in queues.items():
            if number <= len(self.items):
                item = self.items[number - 1]
                told = [_stem(t.split(":")[0]) for t in item.told] + [_stem(t) for t in item.restated]
                stated = [_stem(t.split(":")[0]) for t in item.musts if not _vague(t.split(":", 1)[1] if ":" in t else "")]   # a wish in words leaves the property askable
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

    def question_list(self, number: int, labels: list[str]) -> str:
        """One message naming the item and up to seven of its properties, the likeliest first."""
        if len(labels) <= 1:
            return self.question(number, labels[0]) if labels else ""
        return f"For the {self.items[number - 1].name}, what {', '.join(labels[:-1])} or {labels[-1]} would you like?"

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
            if isinstance(answer, dict) and not FORMAT_MEMORY.get("reply_key") and sum(1 for v in answer.values() if isinstance(v, str) and len(v.split()) >= 2) + sum(1 for v in answer.values() if isinstance(v, list) and v and all(isinstance(x, str) for x in v)) >= 2:
                self.learn_reply_key(answer, text)
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
        if stage == "order" and any(line.startswith("Q: ") for line in said) and self.left() >= len(self.cart_ids()) + 2 and self.time_left() > LAST_CALL + 20:
            self.reread_offers()                              # the lines are read again after the shopper's reply (race 14: orders not re-read after the last answer passed 0.17 vs 0.62)
        return True, ("Before going on, Kyo did the steps the request asks for:\n" + "\n".join(said)) if said else ""

    def spoken(self, observation: Any) -> str:
        """The shopper's words in a reply: the field the structure learner named, else the payload's text (strings that stand
        alone, before any list of notice lines), else the whole reply."""
        if isinstance(observation, str):
            return observation
        key = FORMAT_MEMORY.get("reply_key")
        if isinstance(observation, dict) and key and isinstance(observation.get(key), str):
            return observation[key]
        found: list[str] = []
        listed: list[str] = []

        def walk(node: Any, depth: int, in_list: bool) -> None:
            if depth > 4:
                return
            if isinstance(node, str) and len(node.split()) >= 2 and _decode_text(node) is None:
                (listed if in_list else found).append(node)
            elif isinstance(node, dict):
                for value in node.values():
                    walk(value, depth + 1, False)
            elif isinstance(node, list):
                for value in node:
                    walk(value, depth + 1, True)

        walk(observation, 0, False)
        if found:
            return " ".join(found)[:800]
        if listed:
            return " ".join(listed)[:800]
        return _dump(observation)[:400]

    # ---------- notes inside tool results ----------
    def loose_lines(self, observation: Any) -> list[str]:
        """The shopper's or store's loose sentences in a result (a budget change, a dropped requirement, a sell-out), wherever the
        envelope puts them: not a listing record, not the text of one, not a value inside a record."""
        lines: list[str] = []
        def note_like(text: str) -> bool:
            return len(text.split()) >= 5 and _decode_text(text) is None
        if isinstance(observation, str):
            lines = [observation] if note_like(observation) else []
        elif isinstance(observation, dict) and not self.ident(observation):
            for key, value in observation.items():
                if isinstance(value, str) and note_like(value):
                    lines.append(value)
                elif isinstance(value, list) and value and all(isinstance(v, str) for v in value):
                    lines.extend(v for v in value if note_like(v))
                elif isinstance(value, dict) and not self.ident(value, False, 1):
                    for inner in value.values():
                        if isinstance(inner, str) and len(inner.split()) >= 6 and note_like(inner):
                            lines.append(inner)
                        elif isinstance(inner, list) and inner and all(isinstance(v, str) for v in inner):
                            lines.extend(v for v in inner if note_like(v))
        elif isinstance(observation, list) and observation and all(isinstance(v, str) for v in observation):
            lines = [v for v in observation if note_like(v)]
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
        got = self.ask_json(prompt, self.checker(), 2500)
        if isinstance(got, dict):
            # the model's evidence must exist in the records it was shown: a cited key or value absent from the record text, or a null
            # citation, means the requirement is not shown to be met (a judgement without evidence is what lost race 14's revealed-need orders)
            unproven: list[str] = []
            evidence = got.get("evidence") if isinstance(got.get("evidence"), dict) else {}
            for pid in ids:
                shown = self.record_of(pid).lower()
                for row in _items(evidence.get(pid)):
                    if not isinstance(row, dict):
                        continue
                    requirement = str(row.get("requirement") or "")
                    if _vague(requirement.split(":", 1)[1] if ":" in requirement else requirement):
                        continue                                  # a wish in words settles nothing and needs no record text
                    key = str(row.get("key") or "").strip().lower(); value = str(row.get("value") or "").strip().lower()
                    words = re.findall(r"[a-z0-9.]+", value)
                    if re.search(r"\b(?:not|never|except|without)\b|other than|anything but", requirement.lower()) and key not in ("", "null", "none"):
                        if all(w in shown for w in re.findall(r"[a-z0-9]+", key)[:3]):
                            continue                              # an exclusion: the record's own value under that key is the evidence
                    cited = bool(value) and key not in ("", "null", "none") and (value in shown or (words and all(w in shown for w in words[:5])))
                    if not cited:
                        unproven.append(f"{pid}: {str(row.get('requirement') or '?')[:60]}")
            if unproven:
                got["ok"] = False
                got["problems"] = [str(x) for x in _items(got.get("problems"))][:3] + ["no record text decides: " + "; ".join(unproven[:4])]
                self.trace.append({"kind": "unproven", "rows": unproven[:6]})
        self.trace.append({"kind": "check", "cart": ids, "got": _clip(got, 400)})
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
        while pending and self.trails < 6 and self.cost < SPEND_HARD and self.time_left() > CLOSING:
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
        if not named and self.sweep_before_decline():
            # the store was searched once more for each item (its own words and stated values, cheapest unread rows opened): look again
            got = self.audit_call()
            named = self.audit_fits(got)
        if not named:
            return ""
        self.lead_items = named
        self.leads = [i for ids in named.values() for i in ids]
        return "A careful review thinks these listings may fit: " + "; ".join(f"item {k}: {', '.join(v)}" for k, v in named.items()) + (f" ({got.get('reason')})" if got.get("reason") else "") + ". Open them and buy what truly fits; end without a purchase only if each one fails."

    def sweep_before_decline(self) -> bool:
        """The audit found no fitting listing among those seen: one search per requested item with the item's own words and its stated
        values, the cheapest unread rows of the result opened, so a decline never rests on a half-searched store (bqL1: 23 of 24 declines
        fell on tasks that had a fit). Runs once an episode."""
        search = self.roles.get("search")
        schema = self.schemas.get(search or "") or {}
        props = schema.get("properties") or {}
        shape = getattr(self, "shapes", {}).get(search) if isinstance(getattr(self, "shapes", None), dict) else None
        skip = (shape or {}).get("next_arg") if isinstance(shape, dict) else None
        param = next((k for k, p in props.items() if isinstance(p, dict) and p.get("type") == "string" and k != skip and not any(w in k.lower() for w in ("token", "page", "cursor", "offset"))), None)
        if not search or not param or getattr(self, "swept", False) or self.left() < 10 or self.time_left() < CLOSING + 30:
            return False
        self.swept = True
        opened_any = False
        for number, item in enumerate(self.items, 1):
            if self.left() < 8 or self.finished:
                break
            values = [m.split(":", 1)[1].strip() for m in item.musts + item.told if ":" in m and not _vague(m.split(":", 1)[1])]
            phrase = " ".join(item.name.split()[:3] + [v for v in values if len(v.split()) <= 3][:2])[:120]
            faults: list[str] = []
            args = _shape({param: phrase}, schema, "args", faults)
            if faults:
                continue
            reply = self.act(search, args)
            if reply is None or self.finished:
                break
            observation = reply.get("observation")
            self.observe(search, observation)
            rows = [i for i, _ in self.records_in(observation) if i not in self.full and not self.sold_out(i)]
            rows.sort(key=lambda i: (self.price_of(i) is None, self.price_of(i) or 0))
            self.trace.append({"kind": "sweep", "item": number, "args": args, "rows": len(rows)})
            if rows and self.open_rows(rows[:3]):
                opened_any = True
        return opened_any

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
        args = _shape(self.fill_required(buy, dict(self.brief.get("buy_args") or {}), "buy"), self.schemas.get(buy), "args", faults)
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

    def sync_cart_core(self, observation: Any, name: str | None = None, args: dict[str, Any] | None = None) -> str:
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
                    if not getattr(self, "swept", False):
                        # coverage before the first check: one search per item with its plain words and stated values, the cheapest unread rows
                        # opened (race 14: the fitting listing often surfaced only for the plain kind words or a stated value; other picks failed)
                        self.sweep_before_decline()
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
        if buy and (holding or (self.cart_touched and not fresh and self.left() >= 2)):
            faults: list[str] = []
            args = _shape(self.fill_required(buy, dict(self.brief.get("buy_args") or {}), "buy"), self.schemas.get(buy), "args", faults)
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
            args = _shape(self.fill_required(refuse, self.quiet(refuse, dict(self.brief.get("refuse_args") or {})), "refuse"), self.schemas.get(refuse), "args", faults)
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
