"""Wren: an ORO shopping agent that serves one customer per episode (entry point: agent_main).

State lives in small components: Link numbers the store calls and makes the model calls, Toolbox learns what each
shop tool does, Task holds the request read into items, duties and rules, Stock keeps every listing row and page
seen, Basket mirrors the cart, and Track keeps per-run counters and memory. Five parts act on them and are joined in
Shopper: _TaskPart (reading the request, the ledger and the status line), _ListingPart (results, pages and shop
notes), _AskPart (bulk opens and the interview), _GuardPart (checks that hold a checkout or a decline) and _LoopPart
(the model's turns and its tool calls).

Link also keeps the clock: past set moments no interview, question, loop turn or model call starts, and a model call
gets a timeout that fits the seconds left. A question names a whole list of planned properties while the customer
answers lists, and one last question can go out just before checkout on a message the order does not need.
"""
from __future__ import annotations

import json
import random
import re
import time
from typing import Any

from src.agent.proxy_client import ProxyClient

# ---------------------------------------------------------------------------------------------------- models, limits
# models: quick for the shopping loop and most reads, strong for reviews, a second choice when a call fails
QUICK_MODEL, STRONG_MODEL, FALLBACK_MODEL = "anthropic/claude-haiku-5.5", "anthropic/claude-sonnet-5.5", "openai/gpt-6-luna"
# budgets of one run: loop turns, model calls, spend levels, pauses between retries of a model call
TURN_LIMIT, MODEL_CALL_LIMIT = 42, 60
SPEND_CALM, SPEND_CEILING, STRONG_HEADROOM = 0.055, 0.078, 0.02
RETRY_PAUSES, PAUSE_BUDGET = (4, 10, 20, 35), 45
# text sizes in characters: one field, one tool result, the whole conversation
FIELD_CHARS, RESULT_CHARS, CHAT_CHARS = 450, 15000, 150000
# rounds and caps of the episode's own checks
DUTY_ROUNDS, SELF_PAGES, CHEAP_PEEKS, KEY_GUARDS, BULK_PER_ITEM = 3, 6, 3, 3, 6
WHOLE_RESULT_PAGES, WHOLE_RESULT_TOTAL = 6, 18
GUARD_MARGIN = 0.34
# the clock, in seconds since the episode began: the validator's limit, what stays unused, the least room for a model call
EPISODE_LIMIT, CLOSING_RESERVE, CALL_MINIMUM = 300.0, 20.0, 6.0
# the moments after which nothing new of a sort starts
INTERVIEW_LATEST, EXTRA_QUESTION_LATEST, ASKING_LATEST, TURN_LATEST = 165.0, 190.0, 210.0, 255.0
# the moment the status line starts to hurry, and the room a turn needs once TURN_LATEST has gone by with an empty cart
HURRY_FROM, BARE_CART_ROOM = 200.0, 12.0
# questions: steps the extra question needs, a message cap that counts as tight, properties named by one message
EXTRA_QUESTION_STEPS, TIGHT_MESSAGE_CAP, LIST_LENGTH = 8, 2, 8

MODEL_LINE = ProxyClient(timeout=50, max_retries=1, retry_delay=1)
STORE_LINE = ProxyClient(timeout=95, max_retries=3, retry_delay=1)
SHORT_LINES: dict = {}


def line_within(seconds: float) -> ProxyClient:
    """The model client for a call with this much room: the standing one, or one timing out at a multiple of 5 s."""
    if seconds >= 50:
        return MODEL_LINE
    timeout = max(5, int(seconds) // 5 * 5)
    if timeout not in SHORT_LINES:
        SHORT_LINES[timeout] = ProxyClient(timeout=timeout, max_retries=1, retry_delay=1)
    return SHORT_LINES[timeout]


# ---------------------------------------------------------------------------------------------------- prompts
# The model-facing texts. JSON keys and enum values inside them are the contract the code parses: keep them exact.

PLAYBOOK = """You are Wren, a careful buyer acting for one customer inside an online shop. Your only means of acting: the shop's tools.

Scoring: one point or none, and only an exactly right finish earns it. Two finishes are right.
(1) A purchase: every product requested, each at its requested quantity, each satisfying every condition in force at checkout, the total within the spending limit.
(2) If the shop genuinely offers nothing that satisfies everything: the run must close through the no-purchase tool, worded as the request wants.

THE SHOP AND ITS TOOLS
- A tool call costs a step. Steps are limited; thinking is free. Every shop has its own wording for tools, parameters and replies: before calling a tool, study its description and schema, and spell parameter names and permitted settings exactly.
- Search may be limited: a category filter; keywords matched against just the opening words of your query; a handful of rows per page; a token or offset needed for the next page. When an answer hinges on the complete list, weigh the reported total against the rows you hold and keep fetching pages, handing back the token or offset unchanged. Brief keyword queries led by the value do best: the wanted value, then the product's noun. Zero rows from a filter most likely mean its setting is off or it sits on the wrong parameter: copy the label letter for letter from the rows, or switch to another filter parameter.
- Result rows merely summarise. A shop can spread one listing over several views: here its attributes (properties); there only price, stock and blurb; a comparison view may hold one or the other. The ledger names the attribute tool once Wren has worked it out. Conditions on a trait are judged from those attributes, so read them for anything you might buy. One call showing a batch of entries, with the details required, saves steps. Never open a read listing twice unless something changed in between.
- Wren's interviewer handles the customer for you; you never send the customer anything. Their answers come into the conversation, and Wren enters them in the ledger.
- Some requests add duties along the route: getting approval, sending a message, checking or reading something first. The ledger lists them. Those due right before ordering or a no-purchase ending Wren performs itself the moment you call the checkout or ending tool, so never stall over them, and they are never grounds for buying nothing. Every other duty is yours, with the shop's tools; product questions still travel through Wren's interviewer.

DOES A LISTING FIT?
- What sort of thing an entry is comes from its category label: only the label matching the wanted product as a whole counts, never a nearby one.
- A condition about some property is met once the listing's attribute for that property shows the value. Key spelling is irrelevant: spaces, underscores, camel case, prefixes. In a list separated by commas or slashes, one member suffices. Compare values by their words. Case, singular versus plural, and plain synonyms change nothing. A value containing all words of the required one passes: "dark olive" for olive, "super soft touch" for soft. Other wording fails: an abbreviation or code is not the word behind it, and "olive green" is no "bottle green". Before comparing numbers, convert units. Boundaries belong to their limits. A range passes only when all of it sits inside the limit.
- Decisive is the attribute keyed by the condition's own property, literally or via a plain synonym. If its reading differs, the entry is out, whatever other keys or the heading suggest. Keys about other properties, related ones included, tell you nothing about the condition (a lid's colour is no guide to the pot's colour). Wren appends [evidence] lines to opened listings: the keys named like a condition's property.
- A listing with no attribute for that property can still pass: the wanted setting counts when another attribute (a features or styles list, say) holds it, or the heading or blurb states it. Numbers of another kind do not count (a length is not a volume), and a lone figure with no word on what it measures proves nothing.
- When a condition's value is part of the category label itself (a season, or a target audience, inside the label), the category satisfies that condition.
- Instead of stating a value, a condition may refer to another listing: "matching the X of the <shop> listing that has <property>", "within N of the X on ...'s page". Deal with such conditions first. Search for that listing by category, maker, seller or any of its words. Open its complete page with the one-listing tool or the comparison tool (a result row will not do). If that page sends you on to another listing for the property, find that one too and open it likewise. Whatever page comes last supplies the value. The order is valid only after every page along the chain has been opened. Unless the wanted value is that other listing's own value, the condition shuts out just the listings with exactly that value. By itself it is never grounds for buying nothing.
- If the entry you plan to buy points to a further page about itself, read that page before buying.
- When one requested product must suit another (a charger for a phone, say), select the two jointly, by the listings' own text: a compatibility list, or a series or model code on both pages. If no accessory satisfying its own conditions suits the chosen device, change to another device that satisfies its conditions and that such an accessory does suit; until that change has been tried, never end without buying.
- Honour exclusions and closed lists. Mere wishes ("ideally", "would be nice") serve only to break ties.
- Never buy anything the shop marks as sold out. Price times quantity, added up across the basket, may not exceed the spending limit; matching it exactly is fine.

PICKING THE LISTING
- Cheapest wanted (lowest price or total)? Then the order must hold the lowest-priced entry of ALL fitting ones. Assemble that whole set by paging, since rows are often unsorted by price; for a short result Wren fetches the next pages itself. Then work upward from the lowest price and take the first candidate that fits completely. When a search accepts a maximum price, give it one: list the category below your current best pick's price, which leaves only cheaper candidates to examine.
- Asked for the extreme (highest or lowest) of a measure? Rank every fitting entry by it and take the end asked for. An entry that leaves the measure out never wins.
- Every other case: any listing satisfying all conditions is correct. Do not burn steps looking for a better one.
- Once an opened entry satisfies everything asked of a product, cart it straight away (for cheapest, most or least picks, once the comparison is done) and move to the next product. Shops often sell out a line soon after it is carted; adding early leaves steps for a replacement.
- Two or more distinct products mean a separate basket line apiece, each at its requested quantity.

WHEN THINGS MOVE
- Mid-run, the customer or the shop may change something: the spending limit, a condition gained or removed, a price, a listing gone. Such news arrives inside tool results, often as a refusal at checkout. Notes matter whichever tool's reply carries them, and the ledger always shows what applies. After a refusal the run continues: take in every change, repair the cart, check out again.
- Just before paying, look again at each chosen entry: prices and availability can shift quietly.

FINISHING
- The final cart must hold exactly the chosen lines (take out the rest). Look at it once, then place the checkout call on its own.
- Buy nothing only after a genuine search turned up no listing satisfying everything. A shortage of steps is no such proof: pay then for the strongest cart you hold. Phrase that ending the way the request asks; keep any remark brief and name no listing.
- Each reply must include at least one tool call. Keep your words few: what you know, then your next move."""

INTAKE = """Your job: convert one customer's shopping request into a brief for a buying agent. Read the request together with the shop's tool list. Reply with ONE JSON object, compact, without indentation, and nothing besides it. Its keys:

"items": a list with an element for every separate product the customer wants bought. Each element is {"name": the sort of product, worded as the request words it, "shelf": when some tool enumerates labels, the shop's own category label for this product, copied letter for letter; failing that, the category the request itself names for it ("a tent under camping gear" yields "camping gear"); and otherwise "", "quantity": units of that single listing (an integer; 1 if no number is given; a pair or set sold as one listing is 1), "musts": a list of strings in the form "property: constraint", one per firm condition, each naming the precise value, bound, exclusion or permitted set}.
- Call a property by the request's own word: "meant for an orchard occasion" yields "occasion: orchard"; "a 9-inch tray" yields "tray width: 9 inch". Do not substitute an adjacent property.
- A must drawing its value from a different listing is ONE must keeping that listing's full description, for example "strap colour: unlike the strap colour of the satchel Brindle sells with a brass buckle". Everything identifying that other entry, whether its seller, its type, or a following "with <property> <value> and <property> <value>" clause, remains part of the one must; never split it off as a separate must.
- Price, budget, stock and quantity: never musts.
- Background in a request (who uses the product, where it goes, an earlier failure, the occasion) is no must, except where it spells out a property's value or limit directly. Product names can include such words too, like a season or a target group; those words are part of "name" and never a must.
- When the request treats a use or occasion as a feature the product itself has ("boots for town wear", "a lamp designed for parties", "a scarf, and it is for a graduation", "the occasion must be travel"), it IS a must, filed under that property ("occasion: town"), except when the shop's category label already states it. One particular person's event, or a person or setting the product is meant for ("to take to her graduation", "for my neighbour"), remains background. So does how it ought to fit, feel or look when used.
- Several products in a single "and" phrase (one "I want ..." carrying one quantity, often matching one department of the shop) form ONE item. Split them into several items only if each one gets a quantity or conditions of its own in the request.
"budget": the customer's ceiling on the total, in the request's unit (a number); otherwise null.
"budget_unit": that unit's name (such as "cents"); otherwise null.
"budget_scope": "total" (whole order) or "each" (per item); null without a limit.
"in_stock_only": true, except when the request clearly allows items that cannot currently be had.
"pick": "cheapest" if the lowest price or smallest total is wanted; "most" or "least" for an extreme of some measure; "any" otherwise. Remarks about a tight budget set no pick rule.
"pick_measure": with "most"/"least", the measure pushed up or down; otherwise "".
"messages": an integer, the messages the customer lets the agent send; null if unstated.
"steps": the steps, moves or tool calls the request allows (an integer); null if it sets none.
"lookups": those musts, verbatim, whose value must first come from another listing the request refers to; [] if none.
"rules": all instructions about working method or about finishing, 12 words at most apiece.
"wishes": things the customer would merely welcome.
"ask_first": things the request explicitly says must be checked with the customer before choosing; [] if none.
"process": steps demanded beyond locating and purchasing the correct products, like obtaining the customer's approval, telling or showing them something, confirming something with them, reading something beforehand, using a specific tool. Each is {"step": the action, 15 words at most, "when": "choose" if due before the products are picked, "order" if due just ahead of paying, "end" if due just ahead of a no-purchase close, "messages": count of customer messages it uses, 0 for none}. Use [] when none is wanted. Not steps: conditions on items; the budget; stock; caps on messages and steps; how to finish if nothing fits.
"quiet_decline": true if alternatives must not be suggested or named when nothing fits; false if alternatives are requested; null if silent.
"fits": pairs {"item": number, "with": number}, one wherever the request says a product has to work with, fit or suit some OTHER product it asks for (the one the agent ends up choosing); [] otherwise.
"roles": an object assigning the precise tool per job, null where no tool does it. Jobs: "search"; "open" (displays one listing's attributes or properties; if there are two single-listing tools, take the one described as showing attributes, properties or specifications); "offer" (the current offer of one listing, price, stock and the like, when that is a tool other than "open"); "compare" (shows several listings completely); "ask" (sends the customer a message); "cart" (alters the cart); "cart_view" (displays the cart); "buy" (pays for the order); "refuse" (closes the run with no purchase).
"buy_args": the fixed argument values that make the "buy" tool pay (a mode setting, for a tool that has one); {} if nothing.
"refuse_args": the fixed argument values that make the "refuse" tool close with no purchase; {} if nothing.
"""

ASK_PLAN = """You plan which questions a shopping agent will ask a customer. A customer holds a few firm preferences left out of the request, each tied to a single trait of a single wanted product. Only a question naming exactly that property draws a preference out. If the customer has no opinion on it, the reply is "don't mind"; if they never thought of it, the reply is puzzled. Naming a property therefore always uses up part of a message.

Inputs: the request; the items and what the request fixes for each; and, item by item, the keys found on opened listings of its category. Each key is numbered and shows how many listings have it, plus sample values. Hidden wishes can also involve traits that no key in this list names.

Priority for choosing properties:
0. Anything the request tells you to check with the customer (under ASK FIRST, when present), each named by the property involved.
1. Hinted properties. Requests often tell a little story, and it may nudge toward a hidden preference while leaving its content open. Look for sentences about: something the customer can't assess or hands over to you; a person whose taste rules; how the product looks, or where it will go; who will use or wear it; what is inside it. Each such sentence yields one trait plus the product it concerns; keep them in the sequence the story uses.
When some key in the item's list fits the hint, use that key's name for the trait. A hint about the wearer: the gender or audience key. A hint about looks: the colour, pattern or finish key appearing on the bulk of that product's entries. Traits that none of the product's entries show go behind its top three most common keys.
2. Next, the listed keys, ranked. Put first the properties that people shopping for this product mostly choose by. Typical ones: type or style; form or fit; size, capacity or dimensions; material; colour or pattern. Which of these leads depends on the product. Favour keys that the bulk of the item's entries carry. Once at least five entries have been examined, avoid keys that turned up just once or twice.
Leave out: any trait whose value the request has already fixed for that item (meaning a condition mentions it; a key whose values merely echo a required value, e.g. a "finish" key that shows the required coating, is NOT fixed); any alias of a property already picked; keys where each value bundles several buying options; IDs; model names; warranties and other logistics. Note that pack sizes, counts and per-pack quantities are ordinary properties. A key tagged [not the stated ...] differs from that condition's property, even if the names overlap. Rank each item separately: hinted properties first, listed keys after.
"label": just the plain name of the property. If it comes from a listed key, write out every word of that key: "strap length" for strapLength, "seat height" for seat_height, never trimmed to "length" or "height". Nothing but the name: without sample values, brackets or another property.
At most <<HOWMANY>> picks per item, best first. Your reply: a single JSON object, nothing more: {"ask": [{"item": N, "label": "...", "hint": "up to eight words of the story that led to this pick, or \"\" if none"}]}, every product's choices listed best first."""

ANSWER_READ = """Questions from a shopping agent have been answered by the customer. Go through every answer. Your reply: a single JSON object, nothing more:
{"needs": [{"item": N, "need": "property: condition", "points": true if the value has to be looked up on a different listing, false if the answer states it}], "drop": [{"item": N, "property": "a condition the customer gives up"}], "free": [{"item": N, "property": "..."}], "crowded": [question numbers; include a question when its answer objects to bundled topics or wants them put one by one; leave out answers that merely failed to understand or that ask for plainer words], "budget": the new overall cap (a number), or null}
- "needs": each binding preference an answer expresses, with its precise value or limit and unit: "at least 2 kg", "round", "not leather". Use the property the answer itself mentions, even if a different one was asked about. Several answers on one property make a single need. Quote the value verbatim, nouns and units kept: "at least 4 drawers", never just "at least 4". The value holds only what a listing could show, running from the wanted setting or its limit word through to its end, minus any framing the answer wraps round it: "round" out of "matching round", "settled, with round" or "round and nothing else"; "at least 2 kg" out of "equal to at least 2 kg".
- Some answers give the value by comparison: like, unlike or near the value on another listing. That listing is a product of some shop; a few of its properties pick it out. That is ONE need with "points": true, under the property the answer names. Copy the reply's wording for that value with no edits, starting at the comparison word and running through the full description of the other product, any mention of a further entry included. Never shorten or rephrase it. What describes that other product never becomes a need of this item.
- "free": for each question, the property the customer is indifferent about.
- Homely or roundabout phrasing that names a setting still states that setting. Phrasing that will not bargain over a setting (no swapping it, surrendering it or altering it) counts as wanting that setting; put "not X" only when the reply turns down X itself."""

NOTICE_READ = """A note appeared during a shopping agent's run; it may come from the customer or from the shop. Your reply: a single JSON object, nothing more:
{"kind": "change" if the note alters the task (spending limit, conditions, a price, availability, or the order being refused), "error" if it only says a call was malformed or rejected, otherwise "other", "budget": the newly set total spending cap (a number, in the unit the request uses), else null, "add": [{"item": N, "need": "property: condition"}], "drop": [{"item": N, "need": "the condition dropped"}], "gone": [ids of listings reported unavailable], "sold_out": true if it reports that a picked listing can't be bought any more (gone or sold out), named or not, "price_moved": true if it reports a price change, "summary": "a brief sentence"}
To replace a condition, drop the old one and add the new one. Include every change the note contains."""

BASKET_REVIEW_TEXT = """You give a shopping agent's cart its final review, just before checkout: going by the records shown, is it exactly right?
How to read the records:
- A condition is met when the listing's attribute for that property shows the value. Spelling of keys is irrelevant; one member of a list suffices; letter case, singular or plural form and plain synonyms do not matter; a value including all the words of the required one passes ("dark olive" for olive); units are converted; limits include their boundary; ranges must fall entirely within.
- An attribute keyed to the condition's very property decides, and any other value there fails the listing. An attribute on a different trait, even a close one (a lid's colour versus the pot's colour), decides nothing.
- Without such an attribute, the wanted value still counts when another attribute, the title or the description gives it. Numbers of another kind count for nothing (a length is no volume), and so does a figure whose wording leaves open what it measures.
- A condition named by the category label is met by that category. And each listing's category label must match the requested product as a whole.
- A requirement to differ from another listing's setting holds unless the records show that very setting.
Every product asked for must get a cart line of its own, holding the quantity asked for. The cart may hold nothing besides, and nothing shown as out of stock. Price times units, summed over every line, must stay within the spending cap. Of several records of a listing, the newest wins.
Cheapest, most or least picks: the cart is also wrong when some other listing on display clearly meets every condition and outdoes the cart's pick under that rule.
Your reply: a single JSON object, nothing more: {"ok": true/false, "problems": ["a brief, specific problem", ...], "better": [ids that should take over a cart line, strongest first]}. Set ok to false only if the records clearly show a problem or no record speaks to some condition."""

MATCH_RULES = """an attribute settles a property if its key names it (key spelling is irrelevant; one list member suffices; synonyms and unit conversions count, as does a value including all the words of the required one, such as "dark olive" for olive; limits include their boundary). Keys for other properties, related ones too (a lid's colour versus the pot's colour), settle nothing. With no attribute naming the property, the wanted value still counts when another attribute, the title or the description gives it; a bare figure that leaves unsaid what it measures does not count. A condition carried by the category label itself is met by that category, and the label must match the item. A demand to differ from another listing's setting holds unless the records reveal exactly that setting. Listings that are out of stock or over budget do not fit."""

NOFIT_REVIEW = ("""A shopping agent intends to finish without buying, believing that no listing fits. Go through the listing records on display and test them against the conditions. How to read them: """ + MATCH_RULES + """
SEARCH ROW records were seen only as result rows: their page and its attributes are unread, so on any trait the row leaves out they neither pass nor fail.
Your reply: a single JSON object, nothing more: {"fits": {"<number of the item>": [up to 4 ids of listings that look like a full fit, by price, lowest first]}, "open": [up to 6 ids of SEARCH ROW listings, of a wanted kind and affordable, whose full page might show a fit; most promising first], "reason": "one sentence"}""")

POINTER_SORT = """A customer's request was split into firm conditions on a product. Some conditions take their value from a different listing that the request points to; the words that identify that listing (its type, its shop, the property values given after it with "with ...") sometimes end up wrongly read as conditions on the product being bought. Going by the request's wording, which of the OTHER CONDITIONS below merely identify that other listing? Your reply: a single JSON object, nothing more: {"describe_other": [those conditions, copied exactly; empty if each one is a real condition on the product]}"""

CHAIN_TRACE = """Checkout is next for a shopping agent. Some conditions give no value of their own: they point to a different listing. And that listing's page may pass the reader on again, to a further listing, for the same property. The purchase only counts if the agent has fully opened every listing on each such trail. Using the records below, walk every trail link by link, starting from the condition's own wording and continuing through what each page says. FULL PAGE: the agent opened that page. SEARCH ROW: the listing was only met in a list of results; nobody read its page.
Your reply: a single JSON object, nothing more: {"chains": [{"need": the condition, verbatim, "links": [{"id": id of the record below that is this link, or "" if none is, "about": the words the condition (or the previous page) uses for this listing, "find": "when the link is still missing: a brief search, the seller's name plus the settings given"}], "done": true if the trail ends at a FULL PAGE that directly states the value of the property (even when the condition asks for some other or unlike value, as that page's value is the yardstick), otherwise false, "value": the value of the property as written on that final page, or ""}]}
To be a link, a record must agree with the whole description: the seller or brand, the product type, AND each property value mentioned. Thus "a twin pack of 6" excludes the same shop's listing with 6 sachets but no twin pack. Lacking any entry that agrees on everything, treat the link as still missing, even when that seller offers similar goods, and fill its "find" with a short query: the seller's name and the settings the description mentions.
List a trail's links in sequence, ending at the first one that is missing or not a FULL PAGE: what it says is still unknown."""

STEP_CHECK = """A shopping agent's next move: <<MOVE>>. The customer wants the extra steps below done along the way. For each step due before <<MOVE>>: has it been carried out for the cart as it stands? Decide from the numbered LOG of what the agent did. Only a LOG line showing the step performed makes it done. For a step on the basket or the purchase, that line must follow the latest basket change, and the reply logged there must agree to it. Asking about product properties completes no step.
Where a step is outstanding, name the one shop call that should do it next: a tool from AVAILABLE TOOLS, with arguments. Customer messages use the messaging tool and cover that step only, in one brief plain sentence or question; leave out any product property or value the step does not call for. If nothing due is outstanding, return no call. Likewise when the messages or steps left are too few for the missing step.
Your reply: a single JSON object, nothing more: {"steps": [{"step": as listed, "done": true/false, "lines": [numbers of the LOG lines proving it]}], "call": null or {"tool": "...", "args": {...}}, "hold": any change a reply demands before the agent goes on, else "", "needs": [{"item": N, "need": "property: condition"} for every firm condition such a reply states]}"""

STANDIN_SEARCH = ("""A shopping agent's cart has a line that just sold out. Search the other listing records shown for possible replacements, reading them this way: """ + MATCH_RULES + """
Your reply: a single JSON object, nothing more: {"fits": [{"id": the listing's id, "price": numeric price taken from its record} for every listing that seems to satisfy every condition on the item of the line that sold out, 6 at most], "reason": "one sentence"}""")

RESULT_LAYOUT = """Here is what a shop tool returned. Locate its parts, spelled exactly as keys appear: in the result; in the tool's parameters. Your reply: a single JSON object, nothing more: {"rows": the key whose value is the listing array, "count": key of the overall match total (not the rows on this page), "next": key whose value fetches more rows, "next_arg": which tool parameter that value goes into, "price": per-row price key, "stock": per-row key saying if it can be bought now}. Any of count, next, next_arg, price and stock may be null when absent."""

FINISH_MODES = """Depending on the arguments it gets, the shop tool TOOLNAME will either check out or close without buying. Its schema: SCHEMA
Your reply: a single JSON object, nothing more: {"buy_args": fixed argument values selecting checkout, "refuse_args": fixed argument values selecting the close without buying} (include only the arguments that pick the mode)."""
# ---------------------------------------------------------------------------------------------------- patterns
def any_word(words: tuple, flags: int = 0, edges: bool = True) -> Any:
    """A compiled pattern that finds any of `words` (regex pieces), between word edges unless `edges` is off. Callers
    only ask whether there is a hit, so the order of the words does not matter."""
    body = "(?:" + "|".join(words) + ")"
    return re.compile(r"\b" + body + r"\b" if edges else body, flags)


_MODEL_CODE = "[A-Z][A-Za-z]{0,5}" + r" ?\d{3,6}"
_PART_CODE = r"[A-Z]{1,4}\d{3,6}"
CODE_TOKEN = re.compile(r"\b" + _MODEL_CODE + r"\b")
OWN_PART = re.compile(r"[\[(]\s*(?:part\s+)?(" + _PART_CODE + r")\s*[\])]", re.I)
MODEL_NUMBER = re.compile(r"\bmodel\s*(?:no\.?|number|#)\s*[:.]?\s*(" + _MODEL_CODE + r")\b", re.I)
ALIAS_PHRASE = re.compile(r"\b(?:" + "|".join(("also known as", "also sold as", "sold also as")) + r")\s+([^.;]+)", re.I)
WORKS_PHRASE = re.compile(r"\b(?<!not )(?<!never )(?:compatible with|works with|fits)\s*:?\s+([^.;]+)", re.I)
PART_ALSO_BEFORE = re.compile(r"\bpart\s+" + _PART_CODE + r"\s+also\s+$", re.I)
PART_ALSO_FITS = re.compile(r"\bpart\s+(" + _PART_CODE + r")\s+also fits\s*:?\s*([^.;]+)", re.I)
PART_POINTER = re.compile(r"\b(?:read|see|details|more)\s+(?:about|under|at|on)?\s*(?:the\s+)?part\s+(" + _PART_CODE + r")", re.I)

# keys that name no property of the product itself (identifiers, media, warranty, variants)
NOT_A_PROPERTY = any_word(("condition", "photos?", "images?", "links?", "url", "video", r"warrant\w*", "barcodes?", "ean",
                           "upc", "asin", "codes?", "ids?", "sku", "models?", "variants?", "options?"))
# keys whose values bundle several features at once (unless the key is about a type, style, kind or shape)
_KIND_WORDS = ("shapes?", "kinds?", "styles?", "types?")
_BUNDLE_WORDS = (r"misc\w*", "extras?", r"accessor\w*", r"includ\w*", "descriptions?", "specifications?", "specs?",
                 "highlights?", "functions?", "features?")
BUNDLE_KEY = re.compile("^(?!.*" + any_word(_KIND_WORDS).pattern + ").*" + any_word(_BUNDLE_WORDS).pattern)
FENCE = re.compile(r"^```[A-Za-z]*\s*(.*?)\s*```$", re.S)
CAMEL_EDGE = re.compile(r"([a-z0-9])([A-Z])")
WORD_GLUE = re.compile(r"[\s_.\-]+")
ES_PLURAL = re.compile(r"(?:ch|sh|ss|x|z)es$")
SIGNED_NUMBER = re.compile(r"\s*-?\d+(\.\d+)?\s*")
PLAIN_NUMBER = re.compile(r"\s*\d+(\.\d+)?\s*")
WHOLE_TEXT = re.compile(r"\s*\d+(\.0+)?\s*")
ANY_DIGIT = re.compile(r"\d")
THREE_DIGITS = re.compile(r"\d{3}")
SPACE_RUN = re.compile(r"\s+")
ALNUM = re.compile(r"[a-z0-9]+")
ALNUM3 = re.compile(r"[a-z0-9]{3,}")
ALPHA3 = re.compile(r"[a-z]{3,}")
HAS_LETTER = re.compile(r"[a-z]", re.I)
EXPECTED_STEP = re.compile(r"expected (\d+)")
# words of a value that is a bound, an exclusion or a pointer rather than a plain setting
_BOUND_TERMS = ("or", "up to", "within", "between", "more", "less", "above", "below", "over", "under", "at least",
                "at most", "unlike", "other than", "no", "not")
POINTER_WORDS = any_word(("same as", "unlike", "other than", "shown", "page", "listing"), re.I)
COMPARATIVE = any_word(_BOUND_TERMS, re.I)
SEARCHABLE_EXCLUDED = any_word(_BOUND_TERMS + ("shown", "page", "listing", "alongside"), re.I)
MEASURE_VALUE = re.compile(r"^\s*[-x /.,\d]+\s*[a-z]{0,6}\s*$")
LABEL_CUT = re.compile(r"\s+(?:and|or)\s+|[/,;(]")
PLURAL_TAIL = re.compile(r"(?<=[a-z]{3})s\b")
OUR_TAIL = re.compile(r"our\b")
UNIT_KEY = any_word(("units?", "count", "quantity", "qty"))
REMOVAL_MODE = any_word(("take", "clear", "drop", "delet", "remov"), re.I, edges=False)
SELLER_KEY = any_word(("merchant", "maker", "brand", "vendor", "store", "shop", "seller"), re.I, edges=False)
MONEY_WORDS = any_word(("money", "spend", "budget", "amount", "cost", "price"), re.I, edges=False)
CAP_WORDS = any_word(("up_to", "upto", "cap", "limit", "below", "under", "most", "max"), re.I, edges=False)
_QUERY_TOKENS = ("words?", "terms?", "text", "search", "keywords?", "phrase", "query", "q")
QUERY_FIELD = re.compile("|".join(("(?:^|_)(?:" + "|".join(_QUERY_TOKENS) + ")(?:_|$)", "keyword", "phrase", "query")),
                         re.I)
# for listing a shelf under a price: a tool's ceiling and shelf parameters, and the mode option that does it
CEILING_PARAM = re.compile("(?:^|_)(?:" + "|".join(("up_?to", "under", "ceiling", "maximum", "max", "cap")) + ")(?:_|$)", re.I)
NOT_A_CEILING = any_word(("step", "item", "num", "size", "count", "page", "row", "result"), re.I, edges=False)
SHELF_PARAM = re.compile("|".join(("aisle", "shelf", "label", "categor", "department", "section",
                                   r"(?<!age)(?:^|_)group(?:_name|_label)?$")), re.I)
SHELF_MODE = any_word(("structured", "list", "categor", "brows", "facet", "filter", "field"), re.I, edges=False)
# quantities: a figure followed by a short unit, and a value giving one as its minimum
_AMOUNT = r"\d[\d.,]*\s*[a-z\"']{1,12}"
WITH_UNIT = re.compile(_AMOUNT, re.I)
LOWER_LIMIT = re.compile(r"\b(?:min(?:imum)?|no (?:fewer|less) than|at least)\s+(" + _AMOUNT + r")\s*$", re.I)
AND_WORD = re.compile(r"\band\b", re.I)
NAME_JOINT = re.compile(r"\s*(?:\band\b|,)\s*", re.I)
PLURAL_NOUN = re.compile(r"[a-z]{3}(?<![su])s$", re.I)
OUR_ENDING = re.compile("our$")
SEPARATORS = re.compile("[,/;|]")
DIGIT_RUN = re.compile(r"\d+")
QUOTED_ID = re.compile(r'"([0-9A-Za-z][0-9A-Za-z_.:\-]{5,40})"')
FILLER = ("and", "for", "the", "with")
LINK_STOPWORDS = frozenset(("the", "and", "with", "from", "that", "comes", "listing", "page", "its", "for"))
UNREAD_VALUE_WORDS = frozenset(("not", "no", "least", "most", "under", "over", "unlike", "different", "min", "max", "minimum",
                                "maximum"))
NO_REPAIR = object()


# ---------------------------------------------------------------------------------------------------- small helpers
def as_json_text(value: Any) -> str:
    """Compact JSON, or str() for what JSON cannot hold."""
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        return str(value)


def shorten(value: Any, cap: int, level: int = 0) -> Any:
    """Every string inside `value` cut to `cap` characters; nesting below eight levels becomes '...'."""
    if level > 7:
        return "..."
    deeper = level + 1
    if isinstance(value, list):
        return [shorten(entry, cap, deeper) for entry in value]
    if isinstance(value, dict):
        return dict((key, shorten(entry, cap, deeper)) for key, entry in value.items())
    if isinstance(value, str) and len(value) > cap:
        return value[:cap] + "..."
    return value


def snippet(value: Any, cap: int, limit: int) -> str:
    return as_json_text(shorten(value, cap))[:limit]


def _decoded(text: str) -> Any:
    try:
        return json.loads(text)
    except ValueError:
        return NO_REPAIR


def parse_object(text: Any) -> dict | None:
    """The JSON object a model answered with: a fenced block is unwrapped, and when the whole text is not JSON the
    span from the first '{' to the last '}' is tried. Anything but an object reads as None."""
    if not isinstance(text, str):
        return None
    body = text.strip()
    fenced = FENCE.match(body)
    body = fenced.group(1) if fenced else body
    parsed = _decoded(body)
    if parsed is NO_REPAIR:
        start, stop = body.find("{"), body.rfind("}")
        parsed = _decoded(body[start:stop + 1]) if 0 <= start < stop else None
    return parsed if isinstance(parsed, dict) else None


def split_words(key: Any) -> str:
    """'lidColour', 'lid_colour' and 'Lid-Colour' all read 'lid colour'."""
    return WORD_GLUE.sub(" ", CAMEL_EDGE.sub(r"\1 \2", str(key))).strip().lower()


def snake_key(key: Any) -> str:
    return split_words(key).replace(" ", "_")


def _singular(word: str) -> str:
    return word[:-1] if word.endswith("s") and len(word) > 3 else word


def stems(text: Any) -> frozenset:
    """The words of `text` with a plural 's' taken off words longer than three letters."""
    return frozenset(map(_singular, split_words(text).split()))


def long_stems(text: Any) -> set:
    return {word for word in stems(text) if len(word) >= 4}


def roots(text: Any) -> set:
    """Singular word roots of three letters or more, without the small joining words."""
    found = set()
    for word in split_words(text).split():
        if ES_PLURAL.search(word):
            word = word[:-2]
        elif word.endswith("s") and not word.endswith("ss") and len(word) > 3:
            word = word[:-1]
        if len(word) >= 3 and word not in FILLER:
            found.add(word)
    return found


def name_forms(text: Any) -> frozenset:
    """A name's words spelt one way: no plural 's' on words longer than three letters, -our as -or ("colours" is
    "color")."""
    forms = set()
    for word in split_words(text).split():
        if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
            word = word[:-1]
        forms.add(OUR_ENDING.sub("or", word))
    return frozenset(forms)


def plain_words(text: Any) -> set:
    """A text's words with list separators taken for spaces."""
    return set(SEPARATORS.sub(" ", split_words(text)).split())


def same_property(prop: frozenset, key: frozenset) -> bool:
    """True when a condition's property and an attribute key refer to one thing: the key holds every stem of the
    property, or a key of at least two stems sits entirely within the property."""
    if prop and key:
        return prop <= key or (len(key) >= 2 and key <= prop)
    return False


def lifo_nodes(root: Any) -> Any:
    """Every value inside `root`, itself first, in the order a last-in-first-out walk over dict values and list
    items reaches them (a generator: the walk goes on only as far as the caller reads)."""
    pending = [root]
    while pending:
        node = pending.pop()
        yield node
        if isinstance(node, dict):
            pending.extend(node.values())
        elif isinstance(node, list):
            pending.extend(node)


def listed(value: Any) -> list:
    """A list as it is; one non-empty string or object as a list of one; anything else as no entries."""
    if isinstance(value, list):
        return value
    return [value] if isinstance(value, (str, dict)) and value else []


def whole_number(value: Any) -> int | None:
    """An integer given as an int, an integral float or a digit string; None for anything else (and for bools)."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else None
    if isinstance(value, str) and WHOLE_TEXT.fullmatch(value):
        return int(float(value))
    return None


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def counted(value: Any) -> bool:
    return whole_number(value) is not None and not isinstance(value, str)


def looks_like_id(value: Any, named: bool = False) -> bool:
    """A listing reference: 3 to 64 characters, no space, and a digit unless its field is called an id."""
    if not isinstance(value, str):
        return False
    core = value.strip()
    if " " in core or not 3 <= len(core) <= 64:
        return False
    return named or ANY_DIGIT.search(value) is not None


def guess_id_key(rows: list) -> Any:
    """The field of these rows that holds their reference: every value looks like one and no two rows share it.
    A field named like an id wins, then the longer values."""
    choice, rank = None, None
    for key in rows[0]:
        called_id = "id" in str(key).lower()
        column = [row.get(key) for row in rows]
        if any(not looks_like_id(cell, called_id) for cell in column):
            continue
        column = [cell.strip() for cell in column]
        if len(rows) > 1 and len(column) != len(set(column)):
            continue
        here = (called_id, sum(map(len, column)) / len(column))
        if rank is None or here > rank:
            choice, rank = key, here
    return choice


def _names_ref(entry: Any, ident: str) -> bool:
    return isinstance(entry, dict) and ident in [str(x).strip() for x in entry.values() if isinstance(x, str)]


def holds_id(node: Any, ident: str, depth: int = 0) -> bool:
    """Whether a listing reference shows as a field value anywhere in a result (six levels deep at most)."""
    if depth > 5:
        return False
    if isinstance(node, dict):
        return any(holds_id(inner, ident, depth + 1) for inner in node.values())
    if isinstance(node, list):
        return any(_names_ref(entry, ident) for entry in node) or any(holds_id(inner, ident, depth + 1) for inner in node)
    return False


def _scalars(block: dict) -> dict:
    return {key: cell for key, cell in block.items() if isinstance(cell, (str, int, float)) and not isinstance(cell, bool)}


def attribute_block(node: Any) -> dict | None:
    """The listing's attribute table: its inner object with the most scalar fields (two at least)."""
    if not isinstance(node, dict):
        return None
    tables = [_scalars(inner) for inner in node.values() if isinstance(inner, dict) and len(inner) >= 2]
    best = None
    for table in tables:
        if len(table) >= 2 and len(table) > len(best or ()):
            best = table
    return best


def tool_line(fn: dict) -> str:
    return f"- {fn['name']}: {fn.get('description', '')} | parameters: {as_json_text(fn.get('parameters') or {})}"


def _cache_marked(message: dict) -> dict:
    copy_ = dict(message)
    text = copy_.get("content")
    if isinstance(text, str) and text:
        copy_["content"] = [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]
    return copy_


def with_cache_marks(model: str, messages: list) -> list:
    """Anthropic models get the first and the last message marked for prompt caching (a copy; the chat is kept)."""
    if not model.startswith("anthropic/") or len(messages) <= 1:
        return messages
    ends = (0, len(messages) - 1)
    return [_cache_marked(message) if index in ends else dict(message) for index, message in enumerate(messages)]


# ---------------------------------------------------------------------------------------------------- schemas
def _is_text(value: Any) -> bool:
    return isinstance(value, str)


def schema_types(schema: dict) -> list:
    declared = schema.get("type")
    if _is_text(declared):
        return [declared]
    return list(filter(_is_text, declared or []))


def fields_of(schema: Any) -> dict:
    return (schema or {}).get("properties") or {}


def text_fields(fields: dict) -> list:
    """Parameters that take free text (a string without a list of allowed values)."""
    return [key for key, spec in fields.items()
            if isinstance(spec, dict) and "string" in schema_types(spec) and "enum" not in spec]


def first_field(fields: dict, kind: str) -> Any:
    for key, spec in fields.items():
        if isinstance(spec, dict) and kind in schema_types(spec):
            return key
    return None


def choice_field(fields: dict) -> Any:
    """The first parameter that chooses between two or more fixed values (a mode)."""
    for key, spec in fields.items():
        if isinstance(spec, dict) and isinstance(spec.get("enum"), list) and len(spec["enum"]) >= 2:
            return key
    return None


_TYPE_TESTS = {
    "string": _is_text,
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": is_number,
    "boolean": lambda v: isinstance(v, bool),
    "array": lambda v: isinstance(v, list),
    "object": lambda v: isinstance(v, dict),
    "null": lambda v: v is None,
}


def matches_type(value: Any, kind: str) -> bool:
    test = _TYPE_TESTS.get(kind)
    return True if test is None else test(value)


def _numeric_text(value: Any) -> bool:
    return isinstance(value, str) and SIGNED_NUMBER.fullmatch(value) is not None


def _repair_integer(value: Any) -> Any:
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return int(float(value)) if _numeric_text(value) else NO_REPAIR


def _repair_number(value: Any) -> Any:
    if not _numeric_text(value):
        return NO_REPAIR
    number = float(value)
    return int(number) if number.is_integer() else number


def _repair_string(value: Any) -> Any:
    return str(value) if is_number(value) else NO_REPAIR


def _repair_boolean(value: Any) -> Any:
    word = value.strip().lower() if isinstance(value, str) else ""
    return word == "true" if word in ("true", "false") else NO_REPAIR


def _repair_array(value: Any) -> Any:
    return NO_REPAIR if value is None or isinstance(value, list) else [value]


_REPAIRS = {"integer": _repair_integer, "number": _repair_number, "string": _repair_string,
            "boolean": _repair_boolean, "array": _repair_array}


def _check_choice(value: Any, options: list, where: str, faults: list) -> Any:
    if isinstance(value, str):
        wanted = value.strip().lower()
        for option in options:
            if isinstance(option, str) and option.lower() == wanted:
                value = option
                break
    if value not in options:
        faults.append(f"{where} must be one of {options}")
    return value


_NUMBER_LIMITS = (("minimum", lambda value, bound: value < bound, "is under the minimum"),
                  ("maximum", lambda value, bound: value > bound, "is over the maximum"))


def _check_object(value: dict, schema: dict, where: str, faults: list) -> dict:
    fields = schema.get("properties")
    if not isinstance(fields, dict):
        fields = {}
    if schema.get("additionalProperties") is False:
        faults.extend(f"{where}.{name} is not an argument of this tool" for name in [n for n in value if n not in fields])
    for name in schema.get("required") or []:
        if name not in value:
            faults.append(f"{where}.{name} is missing but required")
    out = {}
    for name, entry in value.items():
        out[name] = conform(entry, fields[name], f"{where}.{name}", faults) if name in fields else entry
    return out


def conform(value: Any, schema: Any, where: str, faults: list) -> Any:
    """`value` checked against a JSON schema and repaired where that is safe. Every problem left is appended to
    `faults` as a short sentence the model can act on."""
    if not isinstance(schema, dict):
        return value
    kinds = schema_types(schema)
    fits = lambda candidate: any(matches_type(candidate, kind) for kind in kinds)
    if kinds and not fits(value):
        for kind in kinds:
            repaired = _REPAIRS[kind](value) if kind in _REPAIRS else NO_REPAIR
            if repaired is not NO_REPAIR:
                value = repaired
                break
        if not fits(value):
            faults.append(f"{where} must be of type {'/'.join(kinds)}")
            return value
    if isinstance(schema.get("enum"), list) and value not in schema["enum"]:
        value = _check_choice(value, schema["enum"], where, faults)
    if isinstance(value, str):
        if isinstance(schema.get("maxLength"), int) and len(value) > schema["maxLength"]:
            value = value[:schema["maxLength"]]
        if isinstance(schema.get("minLength"), int) and len(value) < schema["minLength"]:
            faults.append(f"{where} is shorter than allowed")
    if is_number(value):
        for key, breaks, words in _NUMBER_LIMITS:
            bound = schema.get(key)
            if isinstance(bound, (int, float)) and breaks(value, bound):
                faults.append(f"{where} {words} {bound}")
    if isinstance(value, list):
        if isinstance(schema.get("maxItems"), int) and len(value) > schema["maxItems"]:
            faults.append(f"{where} takes no more than {schema['maxItems']} entries")
        if isinstance(schema.get("minItems"), int) and len(value) < schema["minItems"]:
            faults.append(f"{where} takes at least {schema['minItems']} entries")
        if isinstance(schema.get("items"), dict):
            value = [conform(entry, schema["items"], f"{where}[{index}]", faults) for index, entry in enumerate(value)]
    if isinstance(value, dict):
        value = _check_object(value, schema, where, faults)
    return value


def checked_args(given: dict, schema: Any) -> dict | None:
    """Arguments checked against a tool's schema, or None when any problem is left."""
    faults: list = []
    args = conform(given, schema, "args", faults)
    return None if faults else args


# ---------------------------------------------------------------------------------------------------- fit codes
def model_codes(text: Any) -> list:
    """Model or series codes in a text (an upper-case letter, up to five more letters, an optional space and three to
    six digits), lower-cased."""
    found = CODE_TOKEN.findall(str(text or ""))
    return [SPACE_RUN.sub(" ", code).strip().lower() for code in found]


def compat_marks(text: Any) -> dict:
    """What a listing's own words say about compatibility: its own part code, the model codes it carries, the codes
    it says it works with, per part code the codes another listing says that part also fits, and the part code it
    sends a reader to for more."""
    text = str(text or "")
    carried = [code for pattern in (MODEL_NUMBER, ALIAS_PHRASE) for hit in pattern.finditer(text)
               for code in model_codes(hit.group(1))]
    works = [code for hit in WORKS_PHRASE.finditer(text)
             if PART_ALSO_BEFORE.search(text[max(0, hit.start() - 40):hit.start()]) is None
             for code in model_codes(hit.group(1))]
    extra = [(hit.group(1).upper(), model_codes(hit.group(2))) for hit in PART_ALSO_FITS.finditer(text)]
    own, pointer = OWN_PART.search(text), PART_POINTER.search(text)
    return {"own": own.group(1).upper() if own else "", "codes": carried, "works": works, "also": extra,
            "pointer": pointer.group(1).upper() if pointer else ""}



# Stores can write a listing as text instead of a record: "field: value" parts joined by some separator.
LINE_KEY_LOWER = r"[a-z_][A-Za-z0-9_]*(?: [a-z_][A-Za-z0-9_]*){0,2}"
LINE_KEY_ANY = r"[A-Za-z_][A-Za-z0-9_]*(?: [A-Za-z_][A-Za-z0-9_]*){0,2}"
LINE_JOINS = (r"\s\|\s", r"\s*\n\s*", r"\s*\t\s*", r"\s;\s", "\\s[\u2022\u00b7]\\s")
LINE_WORDS = {"True": True, "False": False, "None": None, "true": True, "false": False, "null": None}
UNSHOWN_NEED = re.compile(r"^\s*(?:arrival|arrive|arrives|arriving|delivered by|delivery date|delivery by|needed by|deadline|orderable)\b",
                          re.I)


def line_value(text: str, field: str = "") -> Any:
    """One field of a text listing as the value a record would hold: a word, a number, a dict of pairs, or the text."""
    if text in LINE_WORDS:
        return LINE_WORDS[text]
    if re.fullmatch(r"-?\d+", text):
        return int(text)
    if re.fullmatch(r"-?\d+\.\d+", text):
        return float(text)
    shapes = [(r";\s*(?=[^=;]{1,40}=)", "=")]
    if re.search("fact|attrib|spec|propert|feature", field, re.I):
        shapes += [(r",\s*(?=[^=,]{1,40}=)", "="), (r";\s*(?=[^:;]{1,40}:\s)", ":")]
    for gap, sign in shapes:
        halves = [piece.split(sign, 1) for piece in re.split(gap, text)]
        if len(halves) >= 2 and all(len(half) == 2 and half[0].strip() for half in halves):
            return {name.strip(): value.strip() for name, value in halves}
    return text


def line_record(text: str, join: str, key: str) -> dict | None:
    record: dict = {}
    for piece in re.split(f"{join}(?={key}\\s*[:=](?:\\s|$))", text):
        match = re.match(f"({key})\\s*[:=](?:\\s(.*))?$", piece, re.S)
        if not match:
            return None
        record[match.group(1)] = line_value((match.group(2) or "").strip(), match.group(1))
    return record if len(record) >= 4 and any(not isinstance(value, str) for value in record.values()) else None


def listing_from_text(text: Any) -> dict | None:
    """The record a text listing stands for, or None for any other text (a reply, a note)."""
    if not isinstance(text, str) or len(text) < 20:
        return None
    body = text.strip()
    if body.startswith("{") and body.endswith("}"):
        try:
            parsed = json.loads(body)
        except ValueError:
            return None
        return parsed if isinstance(parsed, dict) else None
    for join in LINE_JOINS:
        if len(re.findall(join, body)) >= 3:
            for key in (LINE_KEY_LOWER, LINE_KEY_ANY):
                record = line_record(body, join, key)
                if record:
                    return record
    return None


def records_from_text(node: Any) -> Any:
    """A tool result with every text listing turned into its record; everything else stays as it came."""
    if isinstance(node, dict):
        return {key: records_from_text(value) for key, value in node.items()}
    if isinstance(node, list):
        return [records_from_text(value) for value in node]
    record = listing_from_text(node)
    return node if record is None else record


# ---------------------------------------------------------------------------------------------------- the pieces
class Want:
    """One product the shopper asked for: what it is called, the store label it sits under, how many units, the firm
    requirements read from the request, and what the shopper added, gave up or said they do not mind later."""

    def __init__(self, called: str, aisle: str, units: int, firm: list):
        self.called, self.aisle, self.units, self.firm = called, aisle, units, firm
        self.heard, self.lifted, self.unfussed = [], [], []

    def needs(self) -> list:
        return [need for need in self.firm + self.heard if need not in self.lifted]


class Link:
    """Both proxy routes of one episode: model calls (with pauses before a retry and a spend tally) and store calls
    (with the step counter the store expects). Failures go into the episode log instead of raising."""

    step = 0
    store_broke = False
    model_broke = False
    waited = 0
    spent = 0.0
    model_calls = 0

    def __init__(self, binding: dict, finished: bool):
        self.binding, self.finished, self.trace = binding, finished, []
        self.opened_at = time.monotonic()

    def log(self, kind: str, **fields) -> None:
        entry = {"kind": kind}
        entry.update(fields)
        self.trace.append(entry)

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.opened_at

    @property
    def room(self) -> float:
        """Seconds left for model work before the time the episode keeps in reserve."""
        return EPISODE_LIMIT - CLOSING_RESERVE - self.elapsed

    # -- model ------------------------------------------------------------------------------------------------------
    def consult(self, messages: list, models: list, max_tokens: int, tools: list | None = None) -> dict | None:
        """The first usable reply from these models in turn, or None. Each model counts as one call however often a
        passing failure is retried."""
        for model in models:
            if self.model_calls >= MODEL_CALL_LIMIT:
                break
            if self.room < CALL_MINIMUM:
                self.log("call_skipped", clock=round(self.elapsed, 1))
                break
            self.model_calls += 1
            reply, status = self._tries(model, messages, max_tokens, tools)
            if reply is not None:
                return reply
            if status == 402:
                self.model_broke = True
                break
        return None

    def _tries(self, model: str, messages: list, max_tokens: int, tools: list | None) -> tuple:
        attempt = 0
        reply, status = self._post(model, messages, max_tokens, tools)
        while reply is None and self._pause_allowed(status, attempt):
            base = RETRY_PAUSES[attempt]
            self.waited += base
            attempt += 1
            time.sleep(base * (0.8 + 0.4 * random.random()))
            reply, status = self._post(model, messages, max_tokens, tools)
        return reply, status

    def _pause_allowed(self, status: Any, attempt: int) -> bool:
        passing = status is None or status in (402, 408, 425, 429) or (isinstance(status, int) and status >= 500)
        tries = 1 if status in (None, 408) else len(RETRY_PAUSES)
        if not passing or attempt >= tries or self.waited + RETRY_PAUSES[attempt] > PAUSE_BUDGET:
            return False
        return self.room >= CALL_MINIMUM + RETRY_PAUSES[attempt] * 1.2

    def _post(self, model: str, messages: list, max_tokens: int, tools: list | None) -> tuple:
        body = {"model": model, "messages": with_cache_marks(model, messages), "max_tokens": max_tokens,
                "temperature": 0}
        if tools:
            body.update(tools=tools, tool_choice="auto")
        result = line_within(self.room).post_verbose("/inference/chat/completions", body)
        if isinstance(result.data, dict) and result.data.get("choices"):
            return self._first_choice(model, result.data)
        return None, self._failure_status(model, result)

    def _first_choice(self, model: str, data: dict) -> tuple:
        cost = (data.get("usage") or {}).get("cost")
        if isinstance(cost, (int, float)):
            self.spent += float(cost)
        message = data["choices"][0].get("message") or {}
        if message.get("content") or message.get("tool_calls"):
            return message, 200
        self.log("model_error", model=model, status=0)
        return None, 0

    def _failure_status(self, model: str, result: Any) -> Any:
        error, data = result.error or {}, result.data
        status = error.get("status")
        if isinstance(data, dict) and isinstance(data.get("error"), dict):
            status = whole_number(data["error"].get("code"))
            error = {"body": data["error"].get("message")}
        self.log("model_error", model=model, status=status, body=str(error.get("body") or "")[:200])
        return status

    def ask_for_object(self, prompt: str, models: list, max_tokens: int = 1800) -> dict | None:
        """One prompt to these models in turn until one answers with a JSON object."""
        question = [{"role": "user", "content": prompt}]
        for model in models:
            found = parse_object((self.consult(question, [model], max_tokens) or {}).get("content"))
            if found is not None or self.store_broke or self.model_broke or self.model_calls >= MODEL_CALL_LIMIT:
                return found
        return None

    # -- store ------------------------------------------------------------------------------------------------------
    def store_call(self, name: str, args: dict) -> dict | None:
        """One store step. A conflict over the step number is resolved (up to four sends); a finished or quarantined
        episode, a bad request and an unpaid account end the call with None."""
        payload = dict(self.binding)
        payload["step"] = self.step + 1
        payload["action"] = {"name": name, "args": args}
        for _ in range(4):
            result = STORE_LINE.post_verbose("/environment/call", payload)
            if isinstance(result.data, dict):
                return self._accepted(result.data)
            again = self._after_failure(result.error or {})
            if again is False:
                return None
            if again:
                payload["step"] = self.step + 1
        return None

    def _accepted(self, data: dict) -> dict:
        if "observation" in data:
            data["observation"] = records_from_text(data["observation"])
        self.step = data["step"] if isinstance(data.get("step"), int) else self.step + 1
        if data.get("finished"):
            self.finished = True
        return data

    def _after_failure(self, error: dict) -> bool | None:
        """True: send again at the corrected step; None: send the same again; False: give up."""
        status = error.get("status")
        body = str(error.get("body") or "")
        self.log("store_error", status=status, body=body[:200])
        if status in (400, 401, 402, 403, 404, 422):
            if status == 402:
                self.store_broke = True
            return False
        if status != 409:
            return None
        lowered = body.lower()
        if any(word in lowered for word in ("finish", "quarantin", "finaliz")):
            self.finished = True
            return False
        expected = EXPECTED_STEP.search(body)
        if expected:
            self.step = int(expected.group(1)) - 1
            return True
        if "reuse" in lowered or "different action" in lowered:
            self.step += 1
            return True
        return False


class Toolbox:
    """The store's tools: their schemas, the job the request reading gave each one, and what calls have taught about
    them (which views show attributes, result layouts, which arguments take listing references, side-by-side size)."""

    side_ok = 0
    side_bad = 0

    def __init__(self, offered: Any):
        self.entries = [tool for tool in offered or [] if isinstance(tool, dict)
                        and isinstance(tool.get("function"), dict) and isinstance(tool["function"].get("name"), str)]
        self.params = {}
        for tool in self.entries:
            spec = tool["function"].get("parameters")
            self.params[tool["function"]["name"]] = spec if isinstance(spec, dict) else {"type": "object",
                                                                                         "properties": {}}
        self.jobs, self.shows_attrs, self.layouts, self.id_params = {}, {}, {}, set()

    def fields(self, tool: Any) -> dict:
        return fields_of(self.params.get(tool or ""))

    def single_param(self, tool: str | None) -> str | None:
        """The one free-text parameter a tool takes to name a listing, when that is all it requires."""
        if not tool:
            return None
        schema = self.params.get(tool) or {}
        texts = text_fields(fields_of(schema))
        required = schema.get("required") or []
        target = next((key for key in required if key in texts), None) or (texts[0] if len(texts) == 1 else None)
        if target and all(key == target for key in required):
            return target
        return None

    @property
    def endings(self) -> set:
        return {self.jobs.get("buy"), self.jobs.get("refuse")} - {None}

    @property
    def page_views(self) -> set:
        views = set()
        for choices in (("offer", "open"), ("compare",)):
            tool = next(filter(None, map(self.jobs.get, choices)), None)
            if tool:
                views.add(tool)
        return views

    @property
    def offer_view(self) -> str | None:
        return self.jobs.get("offer") or self.jobs.get("open")

    @property
    def attr_view(self) -> str | None:
        """Name of the one-listing tool to use for attributes: whichever was seen showing them, otherwise 'open' if
        it has not been seen without them."""
        for tool, shows in self.shows_attrs.items():
            if shows and self.single_param(tool):
                return tool
        tool = self.jobs.get("open")
        if self.shows_attrs.get(tool or "") is not False and self.single_param(tool):
            return tool
        return None

    @property
    def side_bare(self) -> bool:
        """True when the comparison view shows (or probably shows) no attributes."""
        seen = self.shows_attrs.get(self.jobs.get("compare") or "")
        if seen is None:
            return self.shows_attrs.get(self.attr_view or "") is True or bool(self.jobs.get("offer"))
        return seen is False

    @property
    def side_param(self) -> str | None:
        schema = self.params.get(self.jobs.get("compare") or "") or {}
        target = first_field(fields_of(schema), "array")
        if target and all(key == target for key in schema.get("required") or []):
            return target
        return None

    @property
    def side_room(self) -> int:
        """Upper bound on listings per comparison call: the schema's maxItems when given, otherwise what earlier
        calls taught."""
        limit = (self.fields(self.jobs.get("compare")).get(self.side_param or "") or {}).get("maxItems")
        if isinstance(limit, int):
            return limit
        known_good = self.side_ok
        if self.side_bad:
            return max(2, min(self.side_bad - 1, max(known_good, 2)))
        return max(4, known_good)

    def labels(self) -> set:
        """Every enum option anywhere in the tools' schemas, lower-cased (store labels the tools list)."""
        found: set = set()
        pending = [(schema, 0) for schema in self.params.values()]
        while pending:
            node, depth = pending.pop()
            if depth > 6 or not isinstance(node, dict):
                continue
            found.update(option.strip().lower() for option in node.get("enum") or [] if isinstance(option, str))
            pending += [(inner, depth + 1) for inner in node.values() if isinstance(inner, dict)]
        return found

    def sets_units(self, tool: str) -> bool:
        """True when this cart tool writes a line's count outright rather than adding to it (it offers no mode
        choice and allows a count of 0)."""
        specs = [spec for spec in fields_of(self.params.get(tool, {})).values() if isinstance(spec, dict)]
        if any(isinstance(spec.get("enum"), list) and len(spec["enum"]) >= 2 for spec in specs):
            return False
        return any("integer" in schema_types(spec) and spec.get("minimum") == 0 for spec in specs)

    def cart_params(self, tool: str) -> tuple:
        """(listing parameter, count parameter, mode parameter) of the cart tool."""
        schema = self.params.get(tool) or {}
        fields = fields_of(schema)
        texts = text_fields(fields)
        used = [key for named, key in self.id_params if named == tool and key in texts]
        target = (used[0] if used else None) or next((key for key in schema.get("required") or [] if key in texts), None)
        if not target and len(texts) == 1:
            target = texts[0]
        return target, first_field(fields, "integer"), choice_field(fields)


class Task:
    """What the request asks for, as read once at the start and changed by the shopper or the store later."""

    policy: Any = None
    money_cap: Any = None
    msg_cap = 0
    msgs_kept = 0

    def __init__(self, query: str, step_limit: int):
        self.query, self.step_limit = query, step_limit
        self.reading, self.wants, self.duties = {}, [], []
        self.pointers, self.pointer_values, self.pairs = [], {}, []
        self.adopted: dict = {}

    @property
    def pick(self) -> Any:
        return self.reading.get("pick")

    def live_needs(self) -> list:
        return [(number, need) for number, want in enumerate(self.wants, 1) for need in want.needs()]

    def want_for(self, number: Any) -> Want:
        return self.wants[min(max(1, whole_number(number) or 1), len(self.wants)) - 1]


class Track:
    """The episode's running record: counters and flags, notices and messages, the chat, and what checks decided."""

    turns = 0
    empty_turns = 0
    msgs_sent = 0
    duty_msgs = 0
    interviewed = False
    epoch = 0
    news_named = False
    news_sold = False
    decline_reviewed = False
    gone_holds = 0
    refused_step = -1
    decline_blocks = 0
    decline_pushed = False
    final_cart_offered = False
    confirmed_pick: Any = None
    auto_bought = False
    checkouts = 0
    chain_rounds = 0
    chain_holds = 0
    extra_pages = 0
    wider_said = False
    line_hold_count = 0
    fit_hold_count = 0
    link_search_count = 0
    extra_question_used = False
    lists_on = True

    def __init__(self):
        self.deed_log, self.deed_steps, self.history = [], [], []
        self.news, self.pending_news, self.recent_calls, self.recent_searches = [], [], [], []
        self.candidates, self.hint_order, self.paged_results = [], [], []
        self.duties_cleared, self.read_notes, self.reviewed_keys = set(), set(), set()
        self.sellout_lines, self.cheaper_done = set(), set()
        self.candidates_by_item, self.review_verdicts, self.read_memo, self.link_search_memo = {}, {}, {}, {}
        self.pointer_codes, self.ceilings_listed, self.single_product_items = set(), set(), set()
        self.list_misses: dict = {}
        self.round = Round({})


class Stock:
    """Every listing the store has shown: the merged row per reference, the full record of those whose attribute
    table was seen, the order they first appeared in, and what is known about their availability and views."""

    id_key: Any = None
    price_field: Any = None
    stock_field: Any = None

    def __init__(self):
        self.rows, self.pages, self.order = {}, {}, []
        self.unavailable, self.opened = set(), set()
        self.gone, self.offer_read_at, self.keys_by_tool, self.keys_by_listing = {}, {}, {}, {}

    def ref(self, node: Any, in_list: bool = False, depth: int = 0) -> str | None:
        """The reference of a listing record, or None when `node` is not one."""
        if not isinstance(node, dict) or len(node) < 3:
            return None
        if self.id_key and looks_like_id(node.get(self.id_key), True):
            return node[self.id_key].strip()
        if not (in_list or depth == 0 or any(isinstance(inner, dict) for inner in node.values())):
            return None
        column = guess_id_key([node])
        value = node[column].strip() if column else None
        if value is not None and ("id" in str(column).lower() or value in self.rows):
            return value
        return None

    def rows_shape(self, node: list) -> bool:
        """Whether a list holds listing rows; learns the reference field from them the first time."""
        rows = [entry for entry in node if isinstance(entry, dict) and len(entry) >= 3]
        if len(rows) * 2 < len(node) or not rows:
            return False
        if self.id_key and all(looks_like_id(row.get(self.id_key), True) for row in rows):
            return True
        found = guess_id_key(rows)
        if found:
            self.id_key = found
        return bool(found)

    def take_in(self, observation: Any) -> None:
        """Merge every listing record inside a store result into the books (seven levels deep at most)."""
        pending = [(observation, 0, False)]
        while pending:
            node, depth, in_list = pending.pop()
            if depth > 6:
                continue
            if isinstance(node, list):
                as_rows = self.rows_shape(node)
                pending.extend((child, depth + 1, as_rows) for child in reversed(node))
            elif isinstance(node, dict):
                ident = self.ref(node, in_list, depth)
                if ident:
                    self._merge(ident, node)
                else:
                    pending.extend((child, depth + 1, False) for child in reversed(list(node.values())))

    def _merge(self, ident: str, node: dict) -> None:
        row = self.rows.get(ident)
        if row is None:
            self.order.append(ident)
            row = self.rows[ident] = {}
        went = any(row.get(key) is True and cell is False for key, cell in node.items())
        back = any(row.get(key) is False and cell is True for key, cell in node.items())
        if went:
            self.unavailable.add(ident)
        elif back:
            self.unavailable.discard(ident)
        row.update(node)
        if attribute_block(node):
            self.pages[ident] = dict(row)
        elif ident in self.pages:
            self.pages[ident].update(node)

    def _listing_rows(self, entries: list) -> list:
        found = []
        for entry in entries:
            ident = self.ref(entry, True) if isinstance(entry, dict) else None
            if ident:
                found.append((ident, entry))
        return found

    @staticmethod
    def _row_lists(inner: Any) -> list:
        if isinstance(inner, list):
            return [inner]
        if isinstance(inner, dict):
            return [deeper for deeper in inner.values() if isinstance(deeper, list)]
        return []

    def records(self, observation: Any) -> list:
        """(reference, record) for the listings a result shows: the result itself, its lists, or the lists one
        object down."""
        if isinstance(observation, list):
            return self._listing_rows(observation)
        if not isinstance(observation, dict):
            return []
        ident = self.ref(observation)
        if ident:
            return [(ident, observation)]
        return [pair for inner in observation.values() for entries in self._row_lists(inner)
                for pair in self._listing_rows(entries)]

    def record(self, ident: str) -> Any:
        return self.pages.get(ident) or self.rows.get(ident)

    def price(self, ident: str) -> float | None:
        cell = (self.rows.get(ident) or {}).get(self.price_field) if self.price_field else None
        if isinstance(cell, dict):
            cell = next(filter(is_number, cell.values()), None)
        if isinstance(cell, str) and PLAIN_NUMBER.fullmatch(cell):
            cell = float(cell)
        return float(cell) if is_number(cell) else None

    def sold_out(self, ident: str) -> bool:
        return ident in self.unavailable or ident in self.gone

    def buyable(self, ident: str) -> bool:
        if self.sold_out(ident):
            return False
        cell = (self.rows.get(ident) or {}).get(self.stock_field) if self.stock_field else None
        return cell is not False and not (is_number(cell) and cell <= 0)

    def remember_view(self, tool: str, observation: Any, epoch: int) -> None:
        """Book what fields this tool returns and what each listing has returned through it, for the current
        epoch."""
        for ident, entry in self.records(observation):
            self.keys_by_tool.setdefault(tool, set()).update(set(entry))
            stamp = self.keys_by_listing.get(ident)
            earlier = stamp[1] if stamp is not None and stamp[0] == epoch else set()
            self.keys_by_listing[ident] = (epoch, earlier | set(entry))

    def nothing_new(self, tool: str, ident: str, epoch: int) -> bool:
        """True when calling this tool for the listing again could reveal nothing new in the current epoch."""
        fields = self.keys_by_tool.get(tool)
        if not fields or ident in self.gone:
            return False
        when, seen = self.keys_by_listing.get(ident, (-1, set()))
        return when == epoch and fields <= seen

    def text_of(self, ident: str) -> str:
        entry = self.record(ident)
        return snippet(entry, 400, 3000) if entry else "(no record kept)"


class Basket:
    """The cart as this episode knows it: the lines its own cart calls left (reference -> units) and, when it is
    newer than the last change, the store's own view of the cart."""

    view: Any = None
    viewed_at = -1
    changed_at = -1
    touched = False
    last_reply = ""

    def __init__(self):
        self.lines, self.removed, self.best_lines = {}, set(), []

    @property
    def fresh(self) -> bool:
        return self.view is not None and self.viewed_at >= self.changed_at

    def mark(self) -> str:
        return as_json_text(sorted(self.lines.items()))

    def refs(self, known: dict) -> list:
        if not self.fresh:
            return list(self.lines)
        dumped = as_json_text(self.view)
        return [ident for ident in known if ident in dumped]

    def units(self, ident: str) -> int:
        """Units of a line: the store's own cart view when it is fresh and names one count for it, else the book."""
        if self.fresh:
            for node in lifo_nodes(self.view):
                if not isinstance(node, dict) or ident not in [cell.strip() for cell in node.values()
                                                               if isinstance(cell, str)]:
                    continue
                counts = [whole_number(cell) for key, cell in node.items()
                          if not isinstance(cell, str) and counted(cell) and UNIT_KEY.search(split_words(key))]
                if len(counts) == 1:
                    return counts[0]
        return self.lines.get(ident, 1)

    def describe(self) -> str:
        if self.fresh:
            return snippet(self.view, 300, 3000)
        return "Wren's tally of the cart from its cart calls (listing: units): " + as_json_text(self.lines)

    @staticmethod
    def removing(args: dict, fields: dict) -> bool:
        def later_option(key: str, value: Any) -> bool:
            options = (fields.get(key) or {}).get("enum")
            return isinstance(options, list) and len(options) >= 2 and value in options[1:]

        return any(later_option(key, value) for key, value in args.items())

    @staticmethod
    def _named_lines(args: dict, fields: dict, known: dict, loose: Any) -> tuple:
        """(reference, count) for each listing a cart call names, and whether it lists whole cart lines."""
        named: list = []
        whole = False
        for key, value in args.items():
            if isinstance(value, str):
                if value in known:
                    named.append((value, loose))
                continue
            if not isinstance(value, list):
                continue
            shape = (fields.get(key) or {}).get("items") or {}
            whole = whole or (not value and isinstance(shape, dict)
                              and (shape.get("type") == "object" or "properties" in shape))
            for line in value:
                if isinstance(line, dict):
                    whole = True
                    ident = next((cell for cell in line.values() if isinstance(cell, str) and cell in known), None)
                    count = next((whole_number(cell) for cell in line.values() if counted(cell)), 1)
                    if ident:
                        named.append((ident, count))
                elif isinstance(line, str) and line in known:
                    named.append((line, loose))
        return named, whole

    def apply(self, args: dict, fields: dict, known: dict, step: int, sets_units: bool) -> None:
        """Book one accepted cart call. A list of line objects replaces the whole cart; otherwise each named listing
        gains, loses or is set to the call's count, as the tool's schema says it works."""
        self.touched = True
        self.changed_at = step
        removing = self.removing(args, fields)
        loose = next((whole_number(value) for value in args.values() if counted(value)), None)
        named, whole = self._named_lines(args, fields, known, loose)
        if whole:
            self.lines = {ident: count for ident, count in named if count and count > 0}
            return
        for ident, count in named:
            if removing and first_field(fields, "integer") is not None:
                have = self.lines.get(ident, 0) - (count or 1)
            elif removing:
                have = self.lines.get(ident, 0) - count if count else 0
            elif sets_units:
                have = 1 if count is None else count
            else:
                have = self.lines.get(ident, 0) + (1 if count is None else count)
            if have > 0:
                self.lines[ident] = have
            else:
                self.lines.pop(ident, None)



# ---------------------------------------------------------------------------------------------------- the task
STAGES = ("choose", "order", "end")
STAGE_WORDS = {"choose": "before the products are chosen", "order": "with the order (Wren sees to it as you check out)",
               "end": "with an ending that buys nothing (Wren sees to it then)"}


class Duty:
    """One extra step the request asks for: what it is, its stage (choose, order or end), and the customer
    messages it costs."""

    def __init__(self, text: str, stage: str, messages: int):
        self.text, self.stage, self.messages = text, stage, messages


class _TaskPart:
    """Reading the request once, and the ledger and status lines the store loop sees on every turn."""

    @property
    def steps_left(self) -> int:
        return self.task.step_limit - self.link.step

    @property
    def free_messages(self) -> int:
        return self.task.msg_cap - self.track.msgs_sent - max(0, self.task.msgs_kept - self.track.duty_msgs)

    @property
    def cart_epoch(self) -> str:
        return self.basket.mark() + f"#{self.track.epoch}"

    @property
    def closing_steps(self) -> int:
        """Steps kept for the end: two when a message is still due with the order, else one."""
        if self.track.msgs_sent >= self.task.msg_cap:
            return 1
        speaks = any(duty.stage == "order" and duty.messages > 0 for duty in self.task.duties)
        return 2 if speaks and self.duty_due("order") else 1

    def review_models(self, limit: float = SPEND_CALM) -> list:
        spent = self.link.spent
        if spent <= limit and spent + STRONG_HEADROOM <= SPEND_CEILING:
            return [STRONG_MODEL, QUICK_MODEL]
        return [QUICK_MODEL] if spent <= SPEND_CEILING else []

    def duty_due(self, stage: str) -> bool:
        if (stage, self.cart_epoch) in self.track.duties_cleared:
            return False
        return any(duty.stage == stage for duty in self.task.duties)

    # -- reading the request ----------------------------------------------------------------------------------------
    @staticmethod
    def _reading_usable(reading: dict) -> bool:
        jobs = reading.get("roles")
        return bool(listed(reading.get("items"))) and isinstance(jobs, dict) and bool(jobs.get("buy"))

    @staticmethod
    def _shelf_shared(reading: dict) -> bool:
        shelves = [split_words(str(entry.get("shelf") or "")) for entry in listed(reading.get("items"))
                   if isinstance(entry, dict)]
        shelves = list(filter(None, shelves))
        return len(set(shelves)) < len(shelves)

    def _intake_prompt(self) -> str:
        opening = self.task.policy.get("opening")
        parts = [INTAKE, "\nTHE SHOP'S TOOLS:\n", "\n".join(tool_line(tool["function"]) for tool in self.tools.entries),
                 "\n\nTHE REQUEST:\n", self.task.query]
        if opening is not None:
            parts += ["\n\nWHAT THE SHOP SHOWED FIRST:\n", snippet(opening, 1500, None)]
        return "".join(parts)

    def _first_reading(self, prompt: str) -> dict:
        reading = self.link.ask_for_object(prompt, [QUICK_MODEL], 4000) or {}
        if self.link.model_broke:
            return reading
        if not self._reading_usable(reading):
            return self.link.ask_for_object(prompt, [STRONG_MODEL, QUICK_MODEL], 2500) or reading
        if self._shelf_shared(reading):
            second = self.link.ask_for_object(prompt, [STRONG_MODEL], 2500) or {}
            if self._reading_usable(second):
                return second
        return reading

    def read_task(self) -> None:
        reading = self._first_reading(self._intake_prompt())
        self.task.reading = reading
        named = reading.get("roles") if isinstance(reading.get("roles"), dict) else {}
        self.tools.jobs = {job: tool for job, tool in named.items() if isinstance(tool, str) and tool in self.tools.params}
        for key in ("buy_args", "refuse_args"):
            reading[key] = reading[key] if isinstance(reading.get(key), dict) else {}
        self._settle_finish_modes(reading)
        self._take_items(reading)
        self._take_numbers(reading)

    def _settle_finish_modes(self, reading: dict) -> None:
        """One tool both orders and declines: make sure the two argument sets that pick the mode differ."""
        tool = self.tools.jobs.get("buy")
        if not tool or tool != self.tools.jobs.get("refuse"):
            return
        if reading["buy_args"] and reading["refuse_args"] and reading["buy_args"] != reading["refuse_args"]:
            return
        head, rest = FINISH_MODES.split("TOOLNAME", 1)
        middle, tail = rest.split("SCHEMA", 1)
        prompt = "".join((head, tool, middle, as_json_text(self.tools.params.get(tool)), tail))
        modes = self.link.ask_for_object(prompt, [QUICK_MODEL, STRONG_MODEL], 1500) or {}
        pair = (modes.get("buy_args"), modes.get("refuse_args"))
        if all(isinstance(side, dict) for side in pair) and pair[0] != pair[1]:
            reading["buy_args"], reading["refuse_args"] = pair

    def _take_items(self, reading: dict) -> None:
        for entry in listed(reading.get("items")):
            if isinstance(entry, dict):
                units = whole_number(entry.get("quantity"))
                firm = [str(must) for must in listed(entry.get("musts")) if must and not UNSHOWN_NEED.search(str(must))]
                self.task.wants.append(Want(str(entry.get("name") or "the product"), str(entry.get("shelf") or ""),
                                            units if units and units > 0 else 1, firm))
        renumber = self._merge_shelf_twins()
        self.task.wants = self.task.wants or [Want("the requested product", "", 1, [])]
        size = len(self.task.wants)
        for pair in listed(reading.get("fits")):
            if not isinstance(pair, dict):
                continue
            part, main = (renumber.get(whole_number(pair.get(key)) or 0) for key in ("item", "with"))
            if part and main and part != main and (part, main) not in self.task.pairs:
                if 1 <= min(part, main) and max(part, main) <= size:
                    self.task.pairs.append((part, main))
        for lookup in listed(reading.get("lookups")):
            need = self._lookup_need(str(lookup).strip())
            number = next((n for n, want in enumerate(self.task.wants, 1) if need in want.firm), 1)
            if need and (number, need) not in self.task.pointers:
                self.task.pointers.append((number, need))
        self.split_pointer_words()

    def _lookup_need(self, text: str) -> str:
        """The requirement a lookup names: the same text, or the first requirement on the same property."""
        if any(text in want.firm for want in self.task.wants):
            return text
        prop = split_words(text.split(":")[0])
        for want in self.task.wants:
            for must in want.firm:
                if split_words(must.split(":")[0]) == prop:
                    return must
        return text

    @staticmethod
    def _musts_clash(ours: list, theirs: list) -> bool:
        pairs = [(one.partition(":"), other.partition(":")) for one in ours for other in theirs]
        return any(split_words(a[0]) == split_words(b[0]) and split_words(a[2]) != split_words(b[2]) for a, b in pairs)

    def _shelf_host(self, kept: list, want: Want) -> Want | None:
        if not want.aisle.strip():
            return None
        for other in kept:
            if split_words(other.aisle) == split_words(want.aisle) and not self._musts_clash(other.firm, want.firm):
                return other
        return None

    def _merge_shelf_twins(self) -> dict:
        """Two items under one store label whose requirements do not clash are one item; returns old -> new number."""
        kept: list = []
        renumber: dict = {}
        for number, want in enumerate(self.task.wants, 1):
            host = self._shelf_host(kept, want)
            if host is None:
                kept.append(want)
                renumber[number] = len(kept)
                continue
            renumber[number] = kept.index(host) + 1
            host.firm += [must for must in want.firm if must not in host.firm]
            self.link.log("merged_items", into=host.called, dropped=want.called)
        self.task.wants = kept
        return renumber

    def _take_numbers(self, reading: dict) -> None:
        cap = reading.get("budget")
        cap = float(cap) if isinstance(cap, str) and PLAIN_NUMBER.fullmatch(cap) else cap
        if is_number(cap) and cap > 0:
            self.task.money_cap = cap
        allowed = whole_number(reading.get("messages"))
        if allowed is None or allowed < 0:
            allowed = 3 if self.tools.jobs.get("ask") else self.task.msg_cap
        self.task.msg_cap = allowed
        steps = whole_number(reading.get("steps"))
        if steps is not None and 10 <= steps < self.task.step_limit:
            self.task.step_limit = steps
        for entry in listed(reading.get("process")):
            text = str(entry.get("step") or "").strip() if isinstance(entry, dict) else ""
            if text:
                stage = str(entry.get("when") or "").strip().lower()
                self.task.duties.append(Duty(text, stage if stage in STAGES else "order",
                                             max(0, whole_number(entry.get("messages")) or 0)))
        if self.tools.jobs.get("ask"):
            spoken = [duty.messages for duty in self.task.duties if duty.stage != "choose" and duty.messages]
            self.task.msgs_kept = min(max(spoken), self.task.msg_cap) if spoken else 0

    def split_pointer_words(self) -> None:
        """Words that only describe the other listing a requirement points at are folded into that requirement."""
        for number, want in enumerate(self.task.wants, 1):
            pointing = [need for n, need in self.task.pointers if n == number and need in want.firm]
            others = [must for must in want.firm if must not in pointing]
            moved = self._describing_other(want, pointing, others) if pointing and others else []
            if not moved:
                continue
            head = pointing[0]
            joined = f"{head} (that listing: {'; '.join(moved)})"
            want.firm = [joined if must == head else must for must in want.firm if must not in moved]
            self.task.pointers = [(n, joined if n == number and need == head else need) for n, need in self.task.pointers]
            self.link.log("pointer_split", item=number, moved=moved)

    def _describing_other(self, want: Want, pointing: list, others: list) -> list:
        prompt = "".join((POINTER_SORT, "\n\nTHE REQUEST:\n", self.task.query, f"\n\nTHE PRODUCT: {want.called}\n",
                          "CONDITIONS THAT POINT AT ANOTHER LISTING:\n", "\n".join(pointing), "\nOTHER CONDITIONS:\n",
                          "\n".join(others)))
        answer = self.link.ask_for_object(prompt, [STRONG_MODEL, QUICK_MODEL], 1500) or {}
        flagged = {str(entry).strip() for entry in listed(answer.get("describe_other"))}
        return [must for must in others if must in flagged]

    # -- what the loop sees -----------------------------------------------------------------------------------------
    def _names_other_item(self, want: Want, value: str) -> bool:
        value_stems = stems(value)
        return any(long_stems(other.called) & value_stems for other in self.task.wants if other is not want)

    def only_in_words(self, number: int, need: str) -> bool:
        """A requirement in the shopper's words that no attribute key of this item's opened listings names."""
        if need in [n for _, n in self.task.pointers] or need in [str(x) for x in listed(self.task.reading.get("lookups"))]:
            return False
        prop, _, value = need.partition(":")
        if not prop.strip() or len(ALNUM.findall(value.lower())) < 4 or POINTER_WORDS.search(value):
            return False
        want = self.task.wants[number - 1]
        if self._names_other_item(want, value):
            return False
        key = self.shelf_key()
        tables = [attribute_block(entry) or {} for entry in self.stock.pages.values()
                  if not key or self.owns(want, str(entry.get(key) or ""))]
        tables = [table for table in tables if table]
        if len(tables) < 3:
            return False
        prop_stems = stems(prop)
        return not any(stems(split_words(name)) & prop_stems for table in tables for name in table)

    def ledger_text(self) -> str:
        lines = [self._item_line(number, want) for number, want in enumerate(self.task.wants, 1)]
        lines += self._reading_lines()
        lines += [f"step requested, due {STAGE_WORDS[duty.stage]}: {duty.text}" for duty in self.task.duties]
        source = self._attribute_source_line()
        if source:
            lines.append(source)
        return "\n".join(map("- {}".format, lines))

    def _reading_lines(self) -> list:
        reading = self.task.reading
        lines = []
        if self.task.money_cap is not None:
            scope = "in total" if reading.get("budget_scope") != "each" else "per item"
            lines.append(f"budget: at most {self.task.money_cap} {reading.get('budget_unit') or ''} {scope}".replace("  ", " "))
        pick = reading.get("pick")
        if pick == "cheapest":
            lines.append("choice rule: the cheapest fitting listing (lowest price)")
        elif pick == "most" or pick == "least":
            lines.append(f"choice rule: the {pick} {reading.get('pick_measure') or ''} of the fitting listings".replace("  ", " "))
        else:
            lines.append("choice rule: any fully fitting listing")
        if reading.get("in_stock_only", True) is not False:
            lines.append("in stock only")
        lines.extend(map("wish (only breaks ties): {}".format, listed(reading.get("wishes"))))
        lines.extend(map("rule from the request: {}".format, listed(reading.get("rules"))))
        if reading.get("quiet_decline") is True:
            lines.append("if nothing fits: close without naming or suggesting any listing")
        return lines

    def _item_line(self, number: int, want: Want) -> str:
        live = want.needs()
        worded = [need for need in live if self.only_in_words(number, need)]
        shown = [need + (f" (value from the page it points at: {self.task.pointer_values[need]})"
                         if need in self.task.pointer_values else "") for need in live if need not in worded]
        pieces = [f"item {number}: {want.called}" if len(self.task.wants) > 1 else f"item: {want.called}"]
        if want.aisle:
            pieces.append(f" (shop's label: {want.aisle})")
        pieces.append(f", quantity {want.units}; conditions: ")
        pieces.append("; ".join(shown) if shown else "just the kind of product")
        if worded:
            pieces.append("; worded as the customer put it and absent from every listing's attributes (they decide nothing): "
                          + "; ".join(worded))
        return "".join(pieces)

    def _attribute_source_line(self) -> str:
        facts = self.tools.attr_view
        if not facts:
            return ""
        if self.tools.shows_attrs.get(facts) is not True and not self.tools.jobs.get("offer"):
            return ""
        views = self.tools.page_views
        bare = sorted(tool for tool in views | {self.tools.jobs.get("offer") or ""}
                      if tool and self.tools.shows_attrs.get(tool) is False)
        pages = sorted(tool for tool in views if tool != facts)
        return "".join((f"listing attributes come from {facts}",
                        f"; {', '.join(bare)} show none" if bare else "",
                        f"; a listing's page (as meant by page chains and page checks) is what {' or '.join(pages)} shows"
                        if pages else ""))

    def _urgency(self, left: int, bought: int) -> str:
        if left <= self.closing_steps + 1:
            return (" FINAL STEPS: with a cart line for every item, check out now. Ending with no purchase needs "
                    "proof from your searches that some item lacks any fitting listing; running low on steps is no such "
                    "proof.")
        if left <= 6 or (bought == 0 and left <= len(self.task.wants) + 5):
            return " Few steps remain: set the cart to the lines you chose and check out right away."
        if bought:
            return ""
        if left <= 12:
            return (" Steps are running low: settle what you will buy, cart it, and complete checkout soon, "
                    "within a few steps.")
        if not self.basket.touched and self.link.step >= self.task.step_limit // 2:
            return (" Half the step budget is spent with nothing in the cart: give every item its "
                    "fitting listing in the cart now, since sell-outs happen.")
        return ""

    def _clock_note(self, left: int) -> str:
        if self.link.elapsed < HURRY_FROM or left <= self.closing_steps + 1:
            return ""
        return (" THE CLOCK HAS NEARLY RUN OUT: set the cart to the lines you chose and finish checkout inside two "
                "more turns; a run that overruns its time scores nothing.")

    def status_text(self) -> str:
        left, bought = self.steps_left, self.track.checkouts
        blocks = [f"[Wren] {self.link.step} of {self.task.step_limit} steps used, {left} to go."
                  + self._urgency(left, bought) + self._clock_note(left)]
        live = {need for _, need in self.task.live_needs()}
        unresolved = [need for _, need in self.task.pointers if need in live and need not in self.task.pointer_values]
        if unresolved and not bought:
            blocks.append("[resolve first: for each, open the listing it points to and any listing that page sends "
                          "you on to; the final page holds the value] " + " | ".join(unresolved[:3]))
        if self.track.candidates:
            blocks.append("[review suggestions] " + ", ".join(self.track.candidates))
        plan = self.pairing_plan() if self.task.pairs and not bought else ""
        if plan:
            blocks.append(plan)
        if self.track.news:
            blocks.append("[recent changes] " + " | ".join(self.track.news[-4:]))
        blocks.append("[ledger]\n" + self.ledger_text())
        return "\n".join(blocks)



# ---------------------------------------------------------------------------------------------------- listings
class KeySlot:
    """One attribute key seen under one store label: on how many listings, the distinct values, value per listing."""

    def __init__(self):
        self.count, self.values, self.per = 0, [], {}

    def note(self, ident: str, text: str) -> None:
        self.count += 1
        self.per[ident] = text
        if text not in self.values:
            self.values.append(text)

    def absorb(self, other: "KeySlot") -> None:
        for ident, text in other.per.items():
            self.per.setdefault(ident, text)
            if text not in self.values:
                self.values.append(text)
        self.count = len(self.per)


class ShelfGroup:
    """The opened listings under one store label and their attribute keys."""

    def __init__(self):
        self.count, self.slots = 0, {}

    def add(self, ident: str, table: dict) -> None:
        self.count += 1
        done: set = set()
        for field, cell in table.items():
            name = split_words(field)
            if name and name not in done:
                done.add(name)
                if name not in self.slots:
                    self.slots[name] = KeySlot()
                self.slots[name].note(ident, str(cell).strip().lower())


class _ListingPart:
    """Which field names a listing's kind, the attribute keys seen per item, and reading what a store result says
    besides its listings."""

    # -- the field naming a listing's kind --------------------------------------------------------------------------
    @staticmethod
    def _shelf_rank(seen: set, present: int, labels: set) -> tuple:
        listed_share = sum(1 for cell in seen if cell.strip().lower() in labels) / len(seen)
        return -listed_share, len(seen), -present, -(sum(map(len, seen)) / len(seen))

    def shelf_key(self) -> Any:
        """The row field that names a listing's kind: on most rows, few distinct short values, ideally the labels
        the tools list."""
        present: dict = {}
        values: dict = {}
        for row in self.stock.rows.values():
            for key, cell in row.items():
                if not isinstance(cell, str) or not 0 < len(cell) <= 40 or THREE_DIGITS.search(cell):
                    continue
                values.setdefault(key, set()).add(cell)
                present[key] = present.get(key, 0) + 1
        floor = 0.8 * max(1, len(self.stock.rows))
        labels = self.tools.labels()
        ranked = [(self._shelf_rank(seen, present[key], labels), key) for key, seen in values.items()
                  if present[key] >= floor and len(seen) <= 40]
        return min(ranked, key=lambda pair: pair[0])[1] if ranked else None

    def shelf_of(self, ident: str, key: Any = None) -> str:
        return str((self.stock.rows.get(ident) or {}).get(key) or "")

    # -- items and the kinds of listing -----------------------------------------------------------------------------
    @staticmethod
    def _label_score(words: str, label_roots: set, want: Want) -> float:
        if words == split_words(want.aisle or ""):
            return 3.0
        if words == split_words(want.called):
            return 2.0
        own = roots(want.aisle or "") | roots(want.called)
        return len(label_roots & own) / max(1, len(label_roots | own))

    def item_for_label(self, label: str) -> int:
        """The item (1-based) a store label belongs to, 0 for none."""
        if not label:
            return 0
        if label in self.task.adopted:
            return self.task.adopted[label]
        words, label_roots = split_words(label), roots(label)
        scores = [self._label_score(words, label_roots, want) for want in self.task.wants]
        best = 0
        for number, score in enumerate(scores, 1):
            if score > (scores[best - 1] if best else 0.0):
                best = number
        return best

    def owns(self, want: Want, label: str) -> bool:
        if label and want in self.task.wants:
            return self.item_for_label(label) == self.task.wants.index(want) + 1
        return False

    def adopt_stray_labels(self) -> None:
        """Several items, opened listings under labels no item claims, and just one item left with no label of its
        own: those labels are taken to be that item's."""
        key = self.shelf_key()
        if len(self.task.wants) < 2 or not key:
            return
        labels = {str(entry.get(key) or "") for entry in self.stock.pages.values()} - {""}
        owners = {label: self.item_for_label(label) for label in labels}
        strays = [label for label, owner in owners.items() if not owner]
        claimed = {owner for label, owner in owners.items() if label not in self.task.adopted}
        orphans = set(range(1, len(self.task.wants) + 1)) - claimed
        if strays and len(orphans) == 1:
            number = orphans.pop()
            self.task.adopted.update(dict.fromkeys(strays, number))
            self.link.log("labels_adopted", item=number, labels=strays)

    def home_shelf(self, number: int) -> str:
        """The label (as split_words writes it) an item's listings belong under: the one the item is called by when
        listings carry it, else the commonest among its opened listings, or among its rows when none is opened."""
        key = self.shelf_key()
        if not key:
            return ""
        want = self.task.wants[number - 1]
        for pool in (self.stock.pages, self.stock.rows):
            tally: dict = {}
            for entry in pool.values():
                label = str(entry.get(key) or "")
                if label and self.owns(want, label):
                    words = split_words(label)
                    tally[words] = tally.get(words, 0) + 1
            if tally:
                named = [words for words in map(split_words, (want.aisle, want.called)) if words in tally]
                return named[0] if named else max(tally, key=tally.get)
        return ""

    def _wanted_forms(self) -> list:
        return [(split_words(want.aisle or ""), split_words(want.called), long_stems(want.aisle or "") | long_stems(want.called))
                for want in self.task.wants]

    def _label_wanted(self, label: str, forms: list) -> bool:
        words, label_stems = split_words(label), long_stems(label)
        if any(words in (shelf, name) or label_stems & stem_set for shelf, name, stem_set in forms):
            return True
        return bool(self.item_for_label(label))

    def item_refs(self, number: int) -> list:
        key = self.shelf_key()
        if not key:
            return []
        want = self.task.wants[number - 1]
        return list(filter(lambda ident: self.owns(want, self.shelf_of(ident, key)), self.stock.order))

    def opened_count(self, number: int) -> int:
        key = self.shelf_key()
        if not key:
            return 0
        want = self.task.wants[number - 1]
        return len([entry for entry in self.stock.pages.values()
                    if self.owns(want, str(entry.get(key) or "")) and attribute_block(entry)])

    def item_lanes(self) -> list:
        """Listing references grouped by store label, the groups of the requested kinds first by size."""
        key = self.shelf_key()
        groups: dict = {}
        for ident in self.stock.order:
            groups.setdefault(str(self.stock.rows[ident].get(key) or "") if key else "", []).append(ident)
        forms = self._wanted_forms()
        pool = [refs for label, refs in groups.items() if self._label_wanted(label, forms)] or list(groups.values())
        pool.sort(key=len, reverse=True)
        return pool[:max(1, len(self.task.wants))]

    def attribute_census(self) -> list:
        """(label, key, listings carrying it, up to four example values) for the attribute keys of opened
        listings, per store label of a requested kind, most carried first."""
        key = self.shelf_key()
        groups: dict = {}
        for ident in self.stock.order:
            entry = self.stock.pages.get(ident)
            if not entry:
                continue
            label = str(entry.get(key) or "listings") if key else "listings"
            if label not in groups:
                groups[label] = ShelfGroup()
            groups[label].add(ident, attribute_block(entry) or {})
        forms = self._wanted_forms()
        chosen = {label for label in groups if self._label_wanted(label, forms)}
        ordered = sorted((pair for pair in groups.items() if not chosen or pair[0] in chosen),
                         key=lambda pair: -pair[1].count)
        return [(label, name, slot.count, [cell[:40] for cell in slot.values[:4]])
                for label, group in ordered for name, slot in self._useful_keys(group.slots)[:14]]

    def _useful_keys(self, slots: dict) -> list:
        kept: list = []
        for name, slot in sorted(slots.items(), key=lambda pair: (-pair[1].count, len(pair[0]))):
            cells = list(slot.per.values())
            if not cells or self._noise_key(cells):
                continue
            host = next((spec for other, spec in kept if self.twin_keys(other, spec, name, slot)), None)
            if host is None:
                kept.append((name, slot))
            else:
                host.absorb(slot)
        kept.sort(key=lambda pair: -pair[1].count)
        return kept

    @staticmethod
    def _noise_key(cells: list) -> bool:
        """Long free text, lists of options, or values that are all different and share no words."""
        mean = sum(map(len, cells)) / len(cells)
        if mean > 40 or sum(" / " in cell for cell in cells) * 2 >= len(cells):
            return True
        if len(cells) < 4 or len(set(cells)) != len(cells) or mean <= 12:
            return False
        lonely = [cell for cell in cells
                  if not set(ALPHA3.findall(cell)) & set(ALPHA3.findall(" ".join(o for o in cells if o is not cell)))]
        return (len(cells) - len(lonely)) * 2 < len(cells)

    @staticmethod
    def _values_agree(left: str, right: str) -> bool:
        if left == right:
            return True
        a, b = ALNUM.findall(left.lower()), ALNUM.findall(right.lower())
        short, long_ = sorted((a, b), key=len) if len(a) != len(b) else (a, b)
        return bool(short) and set(short) <= set(long_) and len(long_) <= 2 * len(short) + 2

    @classmethod
    def twin_keys(cls, name: str, spec: KeySlot, other: str, slot: KeySlot) -> bool:
        """Two keys sharing a word that hold the same values on the listings both appear on."""
        mine, theirs = set(name.split()), set(other.split())
        if not (mine and theirs and mine & theirs):
            return False
        both = [ident for ident in spec.per
                if ident in slot.per and spec.per[ident].strip() and slot.per[ident].strip()]
        same = len([ident for ident in both if cls._values_agree(spec.per[ident], slot.per[ident])])
        if mine <= theirs or theirs <= mine:
            if both:
                return same * 2 > len(both)
            return bool((set(spec.values) & set(slot.values)) - {""})
        return len(both) >= 2 and same == len(both)

    def other_property_note(self, label: str, name: str, cells: list) -> str:
        """A key sharing words with a requirement's property whose values never show the required value."""
        number = self.item_for_label(label)
        if number:
            shown = set(ALNUM3.findall(" ".join(cells).lower()))
            for must in self.task.wants[number - 1].firm:
                prop, _, value = must.partition(":")
                if COMPARATIVE.search(value) or not prop.strip() or split_words(prop) == name:
                    continue
                words = set(ALNUM3.findall(value.lower()))
                if stems(prop) & stems(name) and words and not words <= shown:
                    return f" [not the stated {prop.strip()}: a different property]"
        return ""

    # -- result layouts and paging ----------------------------------------------------------------------------------
    @staticmethod
    def _row_lists_in(observation: Any) -> list:
        if not isinstance(observation, dict):
            return []
        return [key for key, cell in observation.items()
                if isinstance(cell, list) and len(cell) >= 2 and all(isinstance(row, dict) for row in cell)]

    def layout_of(self, tool: str, observation: Any) -> dict | None:
        """Where a list result keeps its rows, total, next-page value, price and stock (asked once per tool)."""
        if tool in self.tools.layouts:
            return self.tools.layouts[tool]
        lists = self._row_lists_in(observation)
        if not lists:
            return None
        self.tools.layouts[tool] = None
        sample = {key: (cell[:3] if key in lists else cell) for key, cell in observation.items()}
        prompt = "".join((RESULT_LAYOUT, "\n\nTHE TOOL: ", tool, ", with parameters ",
                          as_json_text(self.tools.params.get(tool) or {}), "\n\nITS RESULT (first rows only):\n",
                          snippet(sample, 120, 4000)))
        answer = self.link.ask_for_object(prompt, [QUICK_MODEL, STRONG_MODEL], 2500) or {}
        rows_key = answer.get("rows") if answer.get("rows") in lists else None
        if not rows_key:
            return None
        first = observation[rows_key][0]
        layout = {"rows": rows_key}
        for part, where in (("count", observation), ("next", observation), ("next_arg", self.tools.fields(tool)),
                            ("price", first), ("stock", first)):
            named = answer.get(part)
            layout[part] = named if isinstance(named, str) and named in where else None
        total = observation[layout["count"]] if layout["count"] else None
        if layout["count"] and (isinstance(total, bool) or not isinstance(total, int)):
            layout["count"] = None
        self.tools.layouts[tool] = layout
        self.stock.price_field = self.stock.price_field or layout["price"]
        self.stock.stock_field = self.stock.stock_field or layout["stock"]
        self.link.log("layout", tool=tool, shape=layout)
        return layout

    def page_state(self, tool: str, observation: Any) -> tuple | None:
        """(total, rows shown, next-page parameter, its value) for a result that can be paged."""
        layout = self.layout_of(tool, observation)
        if not layout or not all((layout["count"], layout["next"], layout["next_arg"])) or not isinstance(observation, dict):
            return None
        rows, total, follow = (observation.get(layout[part]) for part in ("rows", "count", "next"))
        if not (isinstance(rows, list) and rows) or isinstance(total, bool) or not isinstance(total, int):
            return None
        if follow is None or (isinstance(follow, (str, int)) and not isinstance(follow, bool)):
            return total, len(rows), layout["next_arg"], follow
        return None

    def _in_price_order(self, observation: Any) -> bool:
        prices = [self.stock.price(ident) for ident, _ in self.stock.records(observation)]
        if len(prices) < 3 or None in prices:
            return False
        return all(map(lambda low, high: low <= high, prices, prices[1:]))

    def _pages_to_read(self, tool: str, args: dict, observation: Any) -> Any:
        """None (nothing to do), a text explaining why the rest matters, or (total, shown, param, value, pages)."""
        state = self.page_state(tool, observation) if self.task.pick in ("cheapest", "most", "least") else None
        if not state:
            return None
        total, shown, param, follow = state
        given = args.get(param)
        later = given not in (None, 0, 1) if isinstance(given, int) and not isinstance(given, bool) else bool(
            str(given or "").strip())
        if follow is None or later or total <= shown or (self.task.pick == "cheapest" and self._in_price_order(observation)):
            return None
        pages = -(-(total - shown) // shown)
        spare = self.steps_left - self.closing_steps
        if (self.task.pick == "cheapest" and pages <= WHOLE_RESULT_PAGES and spare - pages >= 12
                and self.track.extra_pages + pages <= WHOLE_RESULT_TOTAL):
            return total, shown, param, follow, pages
        if pages > 2 and self.track.extra_pages + 2 <= SELF_PAGES and spare - 2 >= 12:
            pages = 2
        if pages <= 2 and self.track.extra_pages + pages <= SELF_PAGES and spare - pages >= 12:
            return total, shown, param, follow, pages
        return (f"\n[Wren] {total} listings match this search and {shown} are shown, not sorted by price. The "
                f"choice rule weighs every fit: narrow the search by the condition's value, or ask for more pages "
                f"with {param}={as_json_text(follow)}.")

    def read_more_pages(self, tool: str, args: dict, observation: Any) -> str:
        """Cheapest/most/least rules need the whole fitting set: page on through a first search result here (a few
        pages, if steps allow), or tell the model why the remaining rows matter."""
        self.track.paged_results = []
        plan = self._pages_to_read(tool, args, observation)
        if not isinstance(plan, tuple):
            return plan or ""
        total, shown, param, follow, pages = plan
        texts = []
        while len(texts) < pages:
            call = checked_args({**args, param: follow}, self.tools.params.get(tool))
            reply = self.call_store(tool, call) if call is not None else None
            if reply is None or self.link.finished:
                break
            self.track.extra_pages += 1
            seen = reply.get("observation")
            self.observe(tool, seen)
            self.track.paged_results.append(seen)
            shown_text = self.show(seen)
            self.track.read_memo[tool + as_json_text(call)] = (self.track.epoch, self.link.step, shown_text)
            texts.append(f"{param}={as_json_text(follow)}: {shown_text}")
            after = self.page_state(tool, seen)
            follow = after[3] if after else None
            if follow is None:
                break
        self.link.log("paged_on", tool=tool, read=len(texts))
        if not texts:
            return ""
        tail = (f"\n[Wren] Of {total} matching listings, Wren has read {shown * (len(texts) + 1)} rows; to see the "
                f"others, narrow the search by the condition's value or continue paging.") if follow is not None else ""
        return "\n[Wren fetched this search's next pages itself; one step per page]\n" + "\n".join(texts) + tail

    def ignored_price_cap(self, tool: str, args: dict, observation: Any) -> str:
        """A search given a maximum price that still returned dearer rows."""
        fields = self.tools.fields(tool)

        def about_money(key: str) -> bool:
            spec = fields.get(key) if isinstance(fields.get(key), dict) else {}
            return MONEY_WORDS.search(f"{key} {spec.get('description') or ''}") is not None

        caps = sorted((key for key, value in args.items()
                       if key in fields and is_number(value) and CAP_WORDS.search(key) and about_money(key)),
                      key=args.get)
        if not caps or not self.stock.price_field:
            return ""
        cap = args[caps[0]]
        dearer = [ident for ident, _ in self.stock.records(observation) if (self.stock.price(ident) or 0) > cap]
        if not dearer:
            return ""
        others = {key: [option for option in fields[key]["enum"] if option != value] for key, value in args.items()
                  if isinstance(value, str) and isinstance((fields.get(key) or {}).get("enum"), list)}
        hint = ""
        if any(others.values()):
            switches = ", ".join(f"{key}={as_json_text(options[0])}" for key, options in others.items() if options)
            hint = f"; to get only cheaper entries, list the category with {switches} and the same {caps[0]}"
        return f"\n[Wren] The search ignored {caps[0]}={cap:g}: {len(dearer)} rows here are dearer than it{hint}."

    def cheapest_rows_note(self, observations: list) -> str:
        if self.task.pick != "cheapest" or not self.stock.price_field:
            return ""
        seen = {ident for observation in observations for ident, _ in self.stock.records(observation)}
        priced = sorted((self.stock.price(ident), ident) for ident in seen
                        if self.stock.buyable(ident) and self.stock.price(ident) is not None)
        if len(priced) < 2:
            return ""
        return "\n[Wren] rows for sale, lowest price first: " + ", ".join(f"{i} ({p:g})" for p, i in priced[:6])

    # -- what a result says ---------------------------------------------------------------------------------------
    @staticmethod
    def show(observation: Any) -> str:
        text = snippet(observation, FIELD_CHARS, None)
        return text[:RESULT_CHARS] + "...(cut)" if len(text) > RESULT_CHARS else text

    def observe(self, tool: str, observation: Any) -> str:
        """Book a result's listings and read any notice in it; a change is also queued for the loop."""
        self.stock.take_in(observation)
        lines = self.notice_lines(observation)
        meaning = self.read_notice(lines) if lines else ""
        if meaning:
            self.track.pending_news.append(f"(from {tool}) {meaning}")
        return meaning

    def notice_lines(self, observation: Any) -> list:
        """Sentences a result carries outside its listings (five words or more; six one level down)."""
        if not isinstance(observation, dict) or self.stock.ref(observation):
            return []
        lines = []
        for cell in observation.values():
            if isinstance(cell, str):
                lines += [cell] if len(cell.split()) >= 5 else []
            elif isinstance(cell, list):
                if cell and all(isinstance(entry, str) for entry in cell):
                    lines += [entry for entry in cell if len(entry.split()) >= 5]
            elif isinstance(cell, dict) and not self.stock.ref(cell, False, 1):
                lines += [inner for inner in cell.values() if isinstance(inner, str) and len(inner.split()) >= 6]
        return [line for line in lines if line not in self.track.read_notes]

    def _notice_answer(self, lines: list) -> dict:
        self.track.read_notes.update(lines)
        prompt = "".join((NOTICE_READ, "\n\nTHE TASK AS IT STANDS:\n", self.ledger_text(), "\n\nTHE NOTE:\n",
                          "\n".join(lines)))
        return self.link.ask_for_object(prompt, [QUICK_MODEL, FALLBACK_MODEL], 2500) or {}

    def _drop_need(self, entry: dict) -> str:
        want = self.task.want_for(entry.get("item"))
        target = str(entry["need"]).strip()
        live = want.needs()
        hit = [need for need in live if need == target]
        if not hit:
            prop = split_words(target.split(":")[0])
            hit = [need for need in live if split_words(need.split(":")[0]) == prop][:1]
        want.lifted.extend(hit or [target])
        return f"no longer needed: {target}"

    def _add_need(self, entry: dict) -> str:
        want = self.task.want_for(entry.get("item"))
        need = str(entry["need"]).strip()
        if need in want.lifted:
            del want.lifted[want.lifted.index(need)]
        if need not in want.firm + want.heard:
            want.heard.append(need)
        return f"newly needed: {need}"

    def read_notice(self, lines: list) -> str:
        """What a notice changes: budget, requirements, availability, prices. Applied to the books here."""
        answer = self._notice_answer(lines)
        gone = [str(entry) for entry in listed(answer.get("gone")) if isinstance(entry, (str, int))]
        self.stock.unavailable.update(ident for ident in gone if ident in self.stock.rows)
        self.track.news_sold = bool(gone) or answer.get("sold_out") is True
        self.track.news_named = any(ident in self.stock.rows for ident in gone)
        if self.track.news_sold and len(lines) == 1:
            self.track.sellout_lines.update(lines)
        if answer.get("kind") not in (None, "change"):
            return ""
        changed = []
        cap = answer.get("budget")
        if is_number(cap) and cap > 0 and cap != self.task.money_cap:
            self.task.money_cap = cap
            changed.append(f"spending limit now {cap}")

        def wanted(key: str) -> list:
            return [entry for entry in listed(answer.get(key)) if isinstance(entry, dict) and str(entry.get("need") or "").strip()]

        drops, adds = wanted("drop"), wanted("add")
        changed += [self._drop_need(entry) for entry in drops]
        changed += [self._add_need(entry) for entry in adds]
        if gone:
            changed.append("no longer for sale: " + ", ".join(gone))
        if answer.get("price_moved") is True:
            changed.append("a price moved")
        summary = str(answer.get("summary") or "").strip()
        if self.task.pick == "cheapest":
            if drops or gone or self.track.news_sold or any(text.startswith("spending limit now") for text in changed):
                self.track.wider_said = False
                self.track.cheaper_done = set()
        if changed or summary:
            self.track.epoch += 1
            self.track.recent_calls = []
            self.track.news.append(summary or "; ".join(changed))
        return (summary + (" (" + "; ".join(changed) + ")" if changed else "")).strip()

    @staticmethod
    def spoken(observation: Any) -> str:
        """The words of a reply: every string of two words or more (five levels deep), else its JSON."""
        if isinstance(observation, str):
            return observation
        found = []
        pending = [(observation, 0)]
        while pending:
            node, depth = pending.pop()
            if depth > 4:
                continue
            if isinstance(node, str):
                if len(node.split()) >= 2:
                    found.append(node)
                continue
            children = list(node.values()) if isinstance(node, dict) else node if isinstance(node, list) else []
            pending.extend((inner, depth + 1) for inner in reversed(children))
        return " ".join(found)[:800] if found else as_json_text(observation)[:400]

    def _opened_in(self, observation: Any) -> list:
        refs: list = []
        pending = [(observation, 0)]
        while pending:
            node, depth = pending.pop()
            if depth > 6:
                continue
            if isinstance(node, dict):
                ident = self.stock.ref(node, False, depth)
                if ident and attribute_block(node) and ident not in refs:
                    refs.append(ident)
                    continue
                children = list(node.values())
            elif isinstance(node, list):
                children = node
            else:
                continue
            pending.extend((inner, depth + 1) for inner in reversed(children))
        return refs

    def evidence_note(self, observation: Any) -> str:
        """Which attribute keys of each opened listing in the result carry a condition's property in their name."""
        refs = self._opened_in(observation)
        needs = self.task.live_needs()
        if not refs or not needs:
            return ""
        tagged = len(self.task.wants) > 1
        lines = []
        for ident in refs[:12]:
            table = attribute_block(self.stock.pages.get(ident) or {}) or {}
            parts = []
            for number, need in needs[:8]:
                prop = need.split(":", 1)[0].strip()
                hits = [f"{key}={str(cell)[:50]}" for key, cell in table.items() if same_property(stems(prop), stems(key))]
                if hits:
                    parts.append("{}{} -> {}".format(f"(item {number}) " if tagged else "", prop, ", ".join(hits[:3])))
            if parts:
                lines.append(f"{ident}: " + "; ".join(parts))
        if not lines:
            return ""
        return ("[evidence: keys named like a condition's property; a key on a different property (a separate part "
                "or aspect) settles nothing, and a key named otherwise can still hold the same property] "
                + " | ".join(lines))



# ---------------------------------------------------------------------------------------------------- opening, asking
class Round:
    """What the questions run on, from the interview through to the one extra question: per item the planned
    properties, answers that added a need, questions spent and property names covered; and follow-ups queued."""

    def __init__(self, lanes: dict):
        self.lanes = lanes
        self.found = dict.fromkeys(lanes, 0)
        self.spent = dict.fromkeys(lanes, 0)
        self.asked = {number: set() for number in lanes}
        self.queued: list = []


def _error_shown(observation: Any) -> bool:
    return isinstance(observation, dict) and bool(observation.get("error"))


class _AskPart:
    """Opening listings in bulk, and the interview: which properties to ask about, the questions, the answers."""

    def call_store(self, tool: str, args: dict) -> dict | None:
        """One store step; a single-listing or side-by-side view that worked marks its listings as opened."""
        data = self.link.store_call(tool, args)
        if data is not None and tool in self.tools.page_views and not _error_shown(data.get("observation")):
            self.stock.opened.update(cell.strip() for value in args.values()
                                     for cell in (value if isinstance(value, list) else [value]) if isinstance(cell, str))
        return data

    def _open_one(self, tool: str, param: str, ident: str) -> dict | None:
        args = checked_args({param: ident}, self.tools.params.get(tool))
        return None if args is None else self.call_store(tool, args)

    def open_for_attributes(self, refs: list, keep: int) -> list:
        """Open listings one by one with the attribute view until only `keep` steps are left."""
        tool = self.tools.attr_view
        param = self.tools.single_param(tool)
        if not (tool and param):
            return []
        texts = []
        for ident in [i for i in refs if i in self.stock.rows and i not in self.stock.pages and not self.stock.sold_out(i)]:
            if self.steps_left <= keep or self.link.finished or self.tools.shows_attrs.get(tool) is False:
                break
            reply = self._open_one(tool, param, ident)
            if reply is None:
                break
            seen = reply.get("observation")
            self.observe(tool, seen)
            self.stock.remember_view(tool, seen, self.track.epoch)
            if isinstance(seen, dict) and self.stock.ref(seen):
                self.tools.shows_attrs[tool] = self.tools.shows_attrs.get(tool, False) or bool(attribute_block(seen))
            texts.append(f"{ident}: " + self.show(seen)[:1500])
        if texts:
            self.link.log("opened_pages", n=len(texts))
        return texts

    def _bulk_picks(self, cap: int) -> list:
        lanes = self.item_lanes()
        share = max(1, cap // max(1, len(lanes)))
        picks = [ident for refs in lanes for ident in [i for i in refs if i not in self.stock.pages][:share]]
        return picks[:cap]

    def open_in_bulk(self) -> None:
        """Before the interview: open the first listings of each requested kind, side by side when the store can."""
        tool = self.tools.jobs.get("compare")
        target = self.tools.side_param if tool else None
        if not target or self.tools.side_bare:
            self.open_singly()
            return
        if self.steps_left <= 12:
            return
        spec = self.tools.fields(tool)[target]
        picks = self._bulk_picks(spec.get("maxItems") if isinstance(spec.get("maxItems"), int) else 4)
        args = checked_args({target: picks}, self.tools.params.get(tool)) if len(picks) >= 2 else None
        if args is None:
            return
        before = len(self.stock.pages)
        reply = self.call_store(tool, args)
        if reply is None:
            return
        seen = reply.get("observation")
        self.observe(tool, seen)
        self.stock.remember_view(tool, seen, self.track.epoch)
        rows = self.stock.records(seen)
        if rows:
            self.tools.shows_attrs[tool] = self.tools.shows_attrs.get(tool, False) or any(attribute_block(r) for _, r in rows)
        self.link.log("bulk_open", ids=picks, opened=len(self.stock.pages) - before)
        if rows and self.tools.shows_attrs.get(tool) is False:
            self.open_singly()
        elif len(self.stock.pages) == before and len(picks) > 2 and self.steps_left > 12:
            again = checked_args({target: picks[:2]}, self.tools.params.get(tool))
            reply = self.call_store(tool, again) if again is not None else None
            if reply is not None:
                self.observe(tool, reply.get("observation"))

    def _lane_pool(self, refs: list, wanted: set, each: int) -> tuple:
        opened = len([ident for ident in refs if ident in self.stock.pages])
        pool = [ident for ident in refs if ident not in self.stock.pages and self.stock.buyable(ident)]
        if self.task.pick == "cheapest":
            pool.sort(key=self._price_or_inf)
        pool.sort(key=lambda ident: -len(wanted & set(ALNUM3.findall(as_json_text(self.stock.rows.get(ident) or {}).lower()))))
        return pool[:max(0, each - opened)], max(0, 4 - opened)

    def open_singly(self) -> None:
        tool = self.tools.attr_view
        if not tool or not self.tools.single_param(tool):
            return
        wanted = {word for _, need in self.task.live_needs() for word in ALNUM3.findall(need.split(":", 1)[-1].lower())}
        each = BULK_PER_ITEM if len(self.task.wants) <= 2 else 4
        lanes = [self._lane_pool(refs, wanted, each) for refs in self.item_lanes()]
        first = [lane[rank] for rank in range(4) for lane, head in lanes if rank < min(head, len(lane))]
        more = [lane[rank] for rank in range(each) for lane, head in lanes if head <= rank < len(lane)]
        cap = max(5, 2 * len(lanes) + 2)
        read = self.open_for_attributes(first[:min(cap, 8)], 22)
        self.open_for_attributes((first[cap:] + more)[:max(0, 10 - len(read))], 30)

    def _price_or_inf(self, ident: str) -> float:
        price = self.stock.price(ident)
        return float("inf") if price is None else price

    # -- the interview ----------------------------------------------------------------------------------------------
    @property
    def interview_due(self) -> bool:
        if self.track.interviewed or not self.tools.jobs.get("ask"):
            return False
        if self.free_messages <= 0 or self.steps_left <= 8 or self.link.elapsed > INTERVIEW_LATEST:
            return False
        return len(self.stock.pages) >= 3 * len(self.task.wants) + 1 or self.link.step >= max(8, self.task.step_limit // 4)

    def message_args(self, text: str) -> dict | None:
        """Arguments for the messaging tool: the text in its free-text field, the first option of any other required
        choice. None when the tool needs something else."""
        schema = self.tools.params.get(self.tools.jobs.get("ask") or "") or {}
        fields = fields_of(schema)
        texts = text_fields(fields)
        required = schema.get("required") or []
        target = next((key for key in required if key in texts), None) or (texts[0] if texts else None)
        if not target:
            return None
        args = {target: text}
        for key in [key for key in required if key not in args]:
            options = (fields.get(key) or {}).get("enum")
            if not isinstance(options, list) or not options:
                return None
            args[key] = options[0]
        return checked_args(args, schema)

    def run_interview(self) -> str:
        """Put questions to the shopper about the properties most likely to hide a firm preference, as long as
        messages, steps and time allow, and return what was learned for the loop."""
        self.track.interviewed = True
        if self.free_messages <= 0:
            return ""
        self.open_in_bulk()
        left = self.task.msg_cap - self.track.msgs_sent
        state = self.track.round = Round(self.plan_questions(max(left, LIST_LENGTH) if self.track.lists_on else left))
        exchanges: list = []
        learned: list = []
        while self.free_messages > 0 and not self.link.finished and self.steps_left > 6 and self.link.elapsed < ASKING_LATEST:
            batch = self._ask_next(state, learned)
            if not batch:
                break
            exchanges += batch
        return self._interview_summary(state, exchanges, learned)

    def _ask_next(self, state: Round, learned: list, floor: int = 6) -> list:
        """One question sent and its answer booked into the items (each new firm need goes on `learned`). Returns the
        exchange, or [] when no question went out."""
        chosen = self._question_ready(state)
        if chosen is None:
            return []
        number, labels, with_examples, text = chosen
        batch = self.send_messages([(number, labels[0], text)], floor)
        if not batch:
            return []
        state.spent[number] = state.spent.get(number, 0) + 1
        before = [len(want.heard) for want in self.task.wants]
        answer = self.hear(batch, [QUICK_MODEL, STRONG_MODEL])
        if len(labels) > 1:
            answer.pop("free", None)
        self.take_answers(answer, learned)
        fresh = [need for want, size in zip(self.task.wants, before) for need in want.heard[size:]]
        for index, size in enumerate(before):
            if len(self.task.wants[index].heard) > size:
                state.found[index + 1] = state.found.get(index + 1, 0) + 1
        crowded = 1 in [whole_number(entry) for entry in listed(answer.get("crowded"))]
        self._update_plan(state, number, labels, with_examples, fresh, crowded)
        return batch

    def _question_ready(self, state: Round) -> tuple | None:
        """(item, labels, with examples, text) of the next question the messaging tool can carry whole: a queued
        follow-up, else the next planned labels. A label that fits in no question is passed over, a few at most."""
        for _ in range(LIST_LENGTH):
            state.queued = [entry for entry in state.queued if not self.answered_before(entry[0], entry[1][0])]
            if state.queued:
                number, labels, with_examples = state.queued.pop(0)
            else:
                (number, labels), with_examples = self._next_labels(state), False
            if not labels:
                return None
            fitted, text = self._trimmed_question(number, labels, with_examples)
            if with_examples and not fitted:
                fitted, text = self._trimmed_question(number, labels, False)
            if fitted:
                return number, fitted, with_examples, text
            state.asked.setdefault(number, set()).add(split_words(labels[0]))
            self.link.log("question_too_long", item=number, label=labels[0])
        return None

    def _trimmed_question(self, number: int, labels: list, with_examples: bool) -> tuple:
        """The most labels, taken from the front, whose question comes through the messaging tool's schema uncut
        (a text over its length cap would lose its end), and that question; ([], "") when even one label is too much."""
        kept = list(labels)
        while kept:
            text = self.question_for(number, kept, with_examples)
            if text in (self.message_args(text) or {}).values():
                return kept, text
            kept.pop()
        return [], ""

    def _update_plan(self, state: Round, number: int, labels: list, with_examples: bool, fresh: list, crowded: bool) -> None:
        """What an answer means for the plan. One label: it is covered, and a plea for something narrower when the
        answer gave nothing gets a follow-up. A list: such a plea ends lists for good and its first label goes alone;
        otherwise the plan turns round the labels the answer showed to be done."""
        covered = state.asked.setdefault(number, set())
        pleads = crowded and not fresh
        if len(labels) == 1:
            covered.add(split_words(labels[0]))
            if pleads:
                self._queue_narrower(state, number, labels, with_examples)
        elif pleads:
            self.track.lists_on = False
            self.link.log("lists_ended", item=number)
            state.queued.insert(0, (number, labels[:1], False))
        else:
            self._turn_list(state, number, labels, fresh)

    def _queue_narrower(self, state: Round, number: int, labels: list, with_examples: bool) -> None:
        """A follow-up after a plea to ask less at once: the same question about one of the products the item's name
        joins, else the label's last word alone, else the label with example values."""
        if number not in self.track.single_product_items and self.joined_products(number):
            self.track.single_product_items.add(number)
            self.link.log("single_product", item=number, said=self.item_as_spoken(number))
            state.queued.insert(0, (number, labels, with_examples))
            return
        label = labels[0]
        tail = label.split()[-1]
        if len(label.split()) >= 2 and split_words(tail) not in state.asked[number]:
            state.queued.insert(0, (number, [tail], False))
        elif not with_examples and len(self.examples(number, label)) >= 2:
            state.queued.insert(0, (number, labels, True))

    def _turn_list(self, state: Round, number: int, labels: list, fresh: list) -> None:
        """After a list question. The shopper takes up the first property on it they care about, and treats a list with
        none of those like a property they do not mind. So a need closes its label and all in front of it, and what
        stood behind moves to the head of the plan. A reply with no need closes the first label alone and flips the
        rest, so that one the shopper is easy about stops shielding the others; every second miss in a row, the
        labels this list left out go first."""
        answered = self._answered_label(labels, fresh) if fresh else -1
        misses = self.track.list_misses
        misses[number] = 0 if answered >= 0 else misses.get(number, 0) + 1
        state.asked[number].update(map(split_words, labels[:max(answered, 0) + 1]))
        ahead = labels[answered + 1:] if answered >= 0 else labels[:0:-1]
        others = [label for label in state.lanes.get(number) or [] if label not in labels]
        unnamed_first = misses[number] > 0 and misses[number] % 2 == 0
        state.lanes[number] = others + ahead if unnamed_first else ahead + others
        self.link.log("list_turned", item=number, settled=labels[:answered + 1], ahead=ahead[:4])

    def _answered_label(self, labels: list, fresh: list) -> int:
        """The position in `labels` of the first property an answer took up, -1 for none. A fresh need is matched by
        the words of its property; if no label has any of them, by the words of its value against what opened
        listings show under each label."""
        values: dict = {}
        for _, name, _, cells in self.attribute_census():
            bag = values.setdefault(split_words(name), set())
            for cell in cells:
                bag |= roots(cell)
        places = []
        for need in fresh:
            prop, _, value = need.partition(":")
            overlap = [len(name_forms(label) & name_forms(prop)) for label in labels]
            if not any(overlap):
                overlap = [len(values.get(split_words(label), set()) & roots(value)) for label in labels]
            if any(overlap):
                places.append(overlap.index(max(overlap)))
        return min(places, default=-1)

    def answered_before(self, number: int, label: str) -> bool:
        """True when the shopper has already stated what they want for this property, comparing names with
        same_property as name_forms spells them."""
        forms = name_forms(label)
        return any(same_property(forms, name_forms(need.split(":", 1)[0])) for need in self.task.wants[number - 1].heard)

    def _interview_summary(self, state: Round, exchanges: list, learned: list) -> str:
        hits = [number for number, count in state.found.items() if count > 0]
        searched = self.search_new_values(hits) if hits else ""
        if learned:
            self.track.epoch += 1
        lines = ["Wren's interviewer put these questions to the customer (one step per question):\n"
                 + "\n".join(f"Question: {q}\nAnswer: {a}" for _, q, a in exchanges)]
        lines.append("\nFirm conditions learned from the answers: " + ("; ".join(learned) if learned else "none"))
        lines.append(". These apply to every pick; where needed, run fresh searches using the new values.")
        pointing = [need for _, need in self.task.pointers if any(need in line for line in learned)]
        if pointing:
            lines.append(" These draw their value from another listing: " + " | ".join(pointing) + ". Track it down and "
                         "fully open its page, then each listing that page sends you to; the final page gives the "
                         "value, and the order is valid only after every one is open.")
        if searched:
            lines.append("\n" + searched)
        if self.track.pending_news:
            lines.append("\n[notes that came in meanwhile] " + " | ".join(self.track.pending_news))
            self.track.pending_news = []
        return "".join(lines)

    def extra_question(self) -> str:
        """Just before a checkout: one more planned question, on a message the order does not need. The interview
        leaves alone every message kept for a step of the request, those kept for an ending included, so a tight cap
        can leave a hidden need unasked; once buying, messages kept for the ending are free. Returns the text for the
        loop when the answer adds a requirement (the checkout then waits), else ''."""
        if self.track.extra_question_used or not self.tools.jobs.get("ask") or self.track.msgs_sent >= self.task.msg_cap:
            return ""
        if any(duty.stage == "order" and duty.messages > 0 for duty in self.task.duties):
            return ""
        if self.steps_left < EXTRA_QUESTION_STEPS or self.link.elapsed > EXTRA_QUESTION_LATEST or not self.review_models():
            return ""
        self.track.extra_question_used = True
        state = self.track.round
        if not state.lanes:
            state = self.track.round = Round(self.plan_questions(LIST_LENGTH if self.track.lists_on else 1))
        learned: list = []
        cap_before = self.task.money_cap
        batch = self._ask_next(state, learned, EXTRA_QUESTION_STEPS - 2)
        if not batch:
            self.link.log("extra_question", unsent="nothing left to ask, or no message went out")
            return ""
        if self.task.money_cap != cap_before:
            learned.append(f"spending limit now {self.task.money_cap}")
        number, question, answer = batch[0]
        self.link.log("extra_question", q=question, a=answer[:200], learned=learned)
        if not learned:
            return ""
        self.track.epoch += 1
        return self._extra_summary(number, question, answer, learned)

    def _extra_summary(self, number: int, question: str, answer: str, learned: list) -> str:
        text = (f"Before the checkout Wren put one further question to the customer (it used one step):\nQuestion: "
                f"{question}\nAnswer: {answer}\nFirm conditions learned from this answer: " + "; ".join(learned)
                + ". The checkout did not happen: test each cart line against them (where needed, run fresh searches "
                  "using the new values), repair the cart, then check out.")
        pointing = [need for _, need in self.task.pointers if any(need in line for line in learned)]
        if pointing:
            text += (" These draw their value from another listing: " + " | ".join(pointing) + ". Track it down and fully open "
                     "its page, then each listing that page sends you to; the final page gives the value.")
        searched = self.search_new_values([number])
        return text + "\n" + searched if searched else text

    def _next_labels(self, state: Round) -> tuple:
        """The item to ask about next and its labels still to ask: items without a found need first, then the least
        questioned (not counted when the cap is tight and lists have ended), then the order of the story's hints."""
        place = {number: index for index, number in enumerate(self.track.hint_order)}
        tight = self.task.msg_cap <= TIGHT_MESSAGE_CAP and not self.track.lists_on
        order = sorted(state.lanes, key=lambda n: (state.found.get(n, 0) > 0, 0 if tight else state.spent.get(n, 0),
                                                   place.get(n, len(place) + n)))
        for number in order:
            fresh = [label for label in state.lanes[number] if split_words(label) not in state.asked[number]
                     and not self.answered_before(number, label)]
            if fresh:
                return number, fresh[:LIST_LENGTH if self.track.lists_on else 1]
        return 0, []

    def question_for(self, number: int, labels: list, with_examples: bool = False) -> str:
        said = self.item_as_spoken(number)
        if len(labels) > 1:
            return "What " + ", ".join(labels[:-1]) + f" or {labels[-1]} would you like for the {said}?"
        examples = self.examples(number, labels[0]) if with_examples else []
        ending = f", for example {examples[0]} or {examples[1]}?" if examples else "?"
        return f"What {labels[0]} would you like for the {said}" + ending

    def joined_products(self, number: int) -> list:
        """The products an item's name lists with "and" or commas, when each one is a plural noun ("hooks and rails");
        [] for one product whose name happens to hold an "and" ("hat and scarf set", "his and hers mugs")."""
        called = self.task.wants[number - 1].called
        if AND_WORD.search(called) is None:
            return []
        parts = [piece for piece in map(str.strip, NAME_JOINT.split(called)) if piece]
        return parts if len(parts) >= 2 and all(map(PLURAL_NOUN.search, parts)) else []

    def item_as_spoken(self, number: int) -> str:
        """The item as a question words it. Shoppers read a name joining plural nouns with "and" as several questions
        at once and one joined with "or" as one, so it is said with "or"; once they have asked for one thing at a
        time, only the product most of the item's rows name."""
        want = self.task.wants[number - 1]
        home = self.home_shelf(number)
        if home and number not in self.track.single_product_items:
            return home
        parts = self.joined_products(number)
        if not parts:
            return want.called
        if number not in self.track.single_product_items:
            return AND_WORD.sub("or", want.called)
        key = self.shelf_key()
        rows = [row for row in self.stock.rows.values() if not key or self.owns(want, str(row.get(key) or ""))]
        said = [roots(" ".join(cell for field, cell in row.items() if isinstance(cell, str) and field != key)) for row in rows]
        counts = [len([words for words in said if roots(part.split()[-1]) & words]) for part in parts]
        return parts[counts.index(max(counts))]

    def examples(self, number: int, label: str) -> list:
        """Two short values seen under this key that the request does not already name."""
        wanted = split_words(label)
        cells = next((row[3] for row in self.attribute_census() if row[1] == wanted), None)
        if cells is None:
            return []
        settled = " ".join(self.task.wants[number - 1].firm).lower()
        return [cell for cell in cells if cell and cell not in settled and len(cell) <= 30][:2]

    def send_messages(self, questions: list, floor: int = 6) -> list:
        tool = self.tools.jobs["ask"]
        sent = []
        for number, _, text in questions:
            if self.link.finished or self.steps_left <= floor or self.track.msgs_sent >= self.task.msg_cap:
                break
            args = self.message_args(text)
            reply = self.call_store(tool, args) if args else None
            if reply is None:
                break
            self.track.msgs_sent += 1
            answer = reply.get("observation")
            words = self.spoken(answer)
            sent.append((number, text, words))
            self.track.deed_log.append(f"step {self.link.step}: sent the customer {text!r}; they replied {words[:300]!r} "
                                       f"(cart at that time: {self.basket.mark()})")
            self.track.deed_steps.append(self.link.step)
            self.link.log("question", q=text, a=words[:300])
            if _error_shown(answer):
                break
        return sent

    def hear(self, sent: list, models: list) -> dict:
        items = "\n".join(f"item {n}: {want.called}" for n, want in enumerate(self.task.wants, 1))
        pairs = "\n".join(f"{index}. (item {n})\nQuestion: {q}\nAnswer: {a}" for index, (n, q, a) in enumerate(sent, 1))
        prompt = "".join((ANSWER_READ, "\n\nTHE ITEMS:\n", items, "\n\nTHE EXCHANGE:\n", pairs))
        return self.link.ask_for_object(prompt, list(models), 2500) or {}

    def _note_need(self, entry: dict, learned: list) -> None:
        number = min(max(1, whole_number(entry.get("item")) or 1), len(self.task.wants))
        want = self.task.wants[number - 1]
        need = str(entry["need"]).strip()
        if need not in want.heard:
            want.heard.append(need)
            learned.append(f"item {number}: {need}" if len(self.task.wants) > 1 else need)
        if entry.get("points") is True and (number, need) not in self.task.pointers:
            self.task.pointers.append((number, need))

    def take_answers(self, answer: dict, learned: list) -> None:
        """Book what the shopper said: new firm needs, needs given up, properties they do not mind, a new budget."""
        for entry in listed(answer.get("needs")):
            if isinstance(entry, dict) and str(entry.get("need") or "").strip():
                self._note_need(entry, learned)
        for entry in listed(answer.get("drop")):
            if isinstance(entry, dict) and str(entry.get("property") or "").strip():
                want = self.task.want_for(entry.get("item"))
                prop = split_words(str(entry["property"]).split(":")[0])
                hit = [need for need in want.needs() if split_words(need.split(":")[0]) == prop]
                want.lifted.extend(hit)
                learned.extend("given up: " + need for need in hit)
        for entry in listed(answer.get("free")):
            if isinstance(entry, dict) and entry.get("property"):
                self.task.want_for(entry.get("item")).unfussed.append(str(entry["property"]))
        cap = answer.get("budget")
        if is_number(cap) and cap > 0:
            self.task.money_cap = cap

    # -- planning the questions -------------------------------------------------------------------------------------
    def item_key_rows(self, number: int, table: list) -> list:
        """This item's attribute keys, most carried first, without the ones the request settles or that are not
        properties; keys bundling several features go last."""
        settled = [stems(must.split(":", 1)[0]) for must in self.task.wants[number - 1].firm]
        measured = {}
        for row in table:
            measured[row[1]] = len([cell for cell in row[3] if MEASURE_VALUE.match(cell)]) * 2 > len(row[3])
        mine = [row for row in table if self.item_for_label(row[0]) == number]
        if not mine and len(self.task.wants) == 1:
            mine = list(table)
        mine.sort(key=lambda row: (-row[2], measured.get(row[1], False), -len(row[3])))
        keys = [row[1] for row in mine]
        keys = [key for key in keys if not NOT_A_PROPERTY.search(key)
                and not any(same_property(prop, stems(key)) for prop in settled)]
        bundled = [BUNDLE_KEY.search(key) is not None for key in keys]
        return [key for key, flag in zip(keys, bundled) if not flag] + [key for key, flag in zip(keys, bundled) if flag]

    def plan_questions(self, count: int) -> dict:
        """Per item, the properties to ask about in order: the strong model's picks, guarded so a widely carried key
        is not crowded out of the first three by a weak pick, then the remaining keys."""
        self.adopt_stray_labels()
        table = self.attribute_census()
        settled = "\n".join(f"item {number}: {want.called}; already settled by the request: " + ("; ".join(want.firm) or "nothing")
                            for number, want in enumerate(self.task.wants, 1))
        parts = [ASK_PLAN.replace("<<HOWMANY>>", str(max(1, count))), "\n\nTHE REQUEST:\n", self.task.query, "\n\nTHE ITEMS:\n", settled]
        first = [str(entry) for entry in listed(self.task.reading.get("ask_first")) if str(entry).strip()]
        if first:
            parts += ["\n\nASK FIRST (the request wants these checked with the customer):\n", "\n".join(first)]
        parts += ["\n\nATTRIBUTE KEYS ON OPENED LISTINGS:\n", "\n".join(self._key_listing(table)) or "(no listing read in full yet)"]
        plan = self.link.ask_for_object("".join(parts), [STRONG_MODEL, QUICK_MODEL], 3500) or {}
        lanes = {number: [] for number in range(1, len(self.task.wants) + 1)}
        hinted = {number: set() for number in lanes}
        for pick in listed(plan.get("ask")):
            if isinstance(pick, dict):
                self._take_pick(pick, lanes, hinted)
        for number in lanes:
            lanes[number] = self._guarded_lane(number, table, lanes[number], hinted[number])
        self.link.log("question_plan", ask=shorten(plan.get("ask"), 120), lanes={n: lane[:8] for n, lane in lanes.items()})
        return lanes

    def _take_pick(self, pick: dict, lanes: dict, hinted: dict) -> None:
        number = min(max(1, whole_number(pick.get("item") or 1) or 1), len(self.task.wants))
        label = LABEL_CUT.split(str(pick.get("label") or ""))[0].strip().rstrip("?.")
        if not label or len(label.split()) > 5:
            return
        if split_words(label) not in map(split_words, lanes[number]):
            lanes[number].append(label)
        if str(pick.get("hint") or "").strip():
            hinted[number].add(split_words(label))
            if number not in self.track.hint_order:
                self.track.hint_order.append(number)

    def _key_listing(self, table: list) -> list:
        listing = []
        last = None
        for index, (label, name, carried, cells) in enumerate(table, 1):
            if label != last:
                listing.append(f"under category {label!r}:")
            last = label
            bundle = " [bundles several features: ask about it last]" if BUNDLE_KEY.search(name) else ""
            listing.append(f"  {index}. {name} (carried by {carried}): " + " | ".join(cells)
                           + self.other_property_note(label, name, cells) + bundle)
        return listing

    def _lane_shares(self, number: int, table: list) -> dict:
        opened = max(1, self.opened_count(number))
        share: dict = {}
        for label, key, carried, _ in table:
            if self.item_for_label(label) == number:
                share[key] = max(share.get(key, 0.0), carried / opened)
        return share

    @staticmethod
    def _plain(text: str) -> str:
        return OUR_TAIL.sub("or", PLURAL_TAIL.sub("", split_words(text)))

    def _guarded_lane(self, number: int, table: list, picks: list, hinted: set) -> list:
        keys = self.item_key_rows(number, table)
        share = self._lane_shares(number, table)

        def spelled(label: str) -> str:
            plain = self._plain(label)
            return next((key for key in share if self._plain(key) == plain), label)

        def spread_of(label: str) -> float:
            if split_words(label) in share:
                return share[split_words(label)]
            near = [value for key, value in share.items()
                    if same_property(stems(label), stems(key)) or same_property(stems(key), stems(label))]
            return max(near) if near else 0.0

        strong = {split_words(spelled(label)) for label in hinted}
        lane: list = []
        for pick in picks:
            if split_words(spelled(pick)) not in map(split_words, lane):
                lane.append(spelled(pick))
        for key in [key for key in keys if not BUNDLE_KEY.search(key)][:KEY_GUARDS]:
            words = split_words(key)
            top = [split_words(other) for other in lane[:3]]
            if words in top:
                continue
            weak = next((index for index, other in enumerate(lane[:3]) if top[index] not in strong
                         and spread_of(other) + GUARD_MARGIN <= share.get(key, 0.0)), None)
            if weak is None and len(lane) >= 3:
                continue
            placed = [other for other in lane if split_words(other) != words]
            if weak is None:
                placed.append(key)
            elif len(self.task.wants) > 1:
                bumped, placed[weak] = placed[weak], key
                placed.insert(min(3, len(placed)), bumped)
            else:
                placed.insert(min(len(placed), weak), key)
            kept_top = {split_words(other) for other in placed[:3]}
            if not any(name in strong and name not in kept_top for name in top):
                lane = placed
        for key in keys:
            if split_words(key) not in map(split_words, lane):
                lane.append(key)
        return ([label for label in lane if not BUNDLE_KEY.search(split_words(label))]
                + [label for label in lane if BUNDLE_KEY.search(split_words(label))])

    # -- searching the new values -----------------------------------------------------------------------------------
    def _free_text_field(self, key: str) -> bool:
        """A text field the searches filled with varied, several-word wording (not one of the store's labels)."""
        searches = self.track.recent_searches
        if len({str(entry.get(key)) for entry in searches if entry.get(key)}) < 2:
            return False
        if not any(len(str(entry.get(key)).split()) >= 2 for entry in searches):
            return False
        shelf = self.shelf_key()
        labels = {str(row.get(shelf) or "").strip().lower() for row in self.stock.rows.values()} - {""} if shelf else set()
        return all(str(entry.get(key) or "").strip().lower() not in labels for entry in searches)

    @staticmethod
    def _names_a_shelf(value: str, shelves: set) -> bool:
        if split_words(value) in shelves:
            return True
        return any(long_stems(value) & long_stems(shelf) for shelf in shelves if shelf)

    def search_shape(self) -> tuple | None:
        """(tool, fixed arguments, text field) for a search worded by values, modelled on the latest worded
        search the loop made."""
        tool = self.tools.jobs.get("search")
        schema = self.tools.params.get(tool or "") or {}
        fields, required = fields_of(schema), schema.get("required") or []
        texts = text_fields(fields)
        named = [key for key in texts if QUERY_FIELD.search(snake_key(key))]
        for args in self.track.recent_searches[::-1]:
            candidates = [key for key in named or texts if isinstance(args.get(key), str) and HAS_LETTER.search(args[key])]
            if not named:
                candidates = list(filter(self._free_text_field, candidates))
            if candidates and tool:
                return tool, self._fixed_search_args(args, candidates[0], fields, required), candidates[0]
        return None

    def _fixed_search_args(self, args: dict, field: str, fields: dict, required: list) -> dict:
        choice = lambda key: isinstance((fields.get(key) or {}).get("enum"), list)
        shelves = {split_words(want.aisle or "") for want in self.task.wants} | {split_words(want.called) for want in self.task.wants}
        shelf = self.shelf_key()
        if shelf:
            shelves |= {split_words(str(row.get(shelf) or "")) for row in self.stock.rows.values()}
        fixed = {key: value for key, value in args.items() if key != field and (choice(key) or key in required)
                 and (key in required or not isinstance(value, str) or not self._names_a_shelf(value, shelves))}
        worded = [entry for entry in self.track.recent_searches
                  if isinstance(entry.get(field), str) and HAS_LETTER.search(entry[field])]
        for key in [key for key in fixed if choice(key)]:
            used = [as_json_text(entry[key]) for entry in worded if key in entry]
            if used:
                common = max(used, key=used.count)
                fixed[key] = next(entry[key] for entry in worded if key in entry and as_json_text(entry[key]) == common)
        return fixed

    def search_or_mode_shape(self) -> tuple | None:
        """What a search by values copies: search_shape(), or, after nothing but wordless listings of a category, a
        shape worked out from the schema alone."""
        shape = self.search_shape()
        tool = self.tools.jobs.get("search")
        if shape or not tool or not self.track.recent_searches:
            return shape
        return self._worded_mode_shape(tool, self.track.recent_searches[-1])

    def _worded_mode_shape(self, tool: str, last: dict) -> tuple | None:
        """(tool, fixed arguments, query field) from the schema alone, None if it leaves any doubt. The mode is the
        one option named after words or a query; failing that, the other side of a required choice of two."""
        schema = self.tools.params.get(tool) or {}
        fields, required = fields_of(schema), schema.get("required") or []
        named = [key for key in text_fields(fields) if QUERY_FIELD.search(snake_key(key))]
        modes = []
        choices = 0
        for key, spec in fields.items():
            options = spec.get("enum") if isinstance(spec, dict) else None
            if not isinstance(options, list) or len(options) < 2:
                continue
            choices += 1
            worded = [option for option in options if isinstance(option, str) and QUERY_FIELD.search(option)]
            unused = [option for option in options if option != last.get(key)]
            if len(worded) == 1:
                modes.append((key, worded[0]))
            elif len(unused) == 1 and len(options) == 2 and key in required:
                modes.append((key, unused[0]))
        if len(named) != 1 or len(modes) > 1 or (choices and not modes):
            return None
        fixed = {key: last[key] for key in required if key != named[0] and key in last}
        fixed.update(modes)
        return tool, fixed, named[0]

    def value_terms(self, number: int) -> list:
        """One pair per requirement of the item that wording can find: the text to search, and True if it came from an
        answer. The text is the plain value; for a lower limit from an answer it is the figure with its unit, since
        fitting listings tend to show that figure. Other limits, negations and pointers are left out."""
        heard = self.task.wants[number - 1].heard
        pointing = {need for _, need in self.task.pointers}
        terms = []
        for owner, need in self.task.live_needs():
            if owner != number or need in pointing or ":" not in need:
                continue
            value = need.split(":", 1)[1].strip()
            limit = LOWER_LIMIT.search(value)
            if limit is None and value and not SEARCHABLE_EXCLUDED.search(value):
                terms.append((value, need in heard))
            elif limit is not None and need in heard:
                terms.append((limit.group(1).strip(), True))
        return terms

    @staticmethod
    def _term_attempts(want: Want, terms: list, fresh: list) -> list:
        """The wordings to search, up to two: all values, a new value of one word or a number with its unit in front,
        then the same without the last value the request stated. A lone value of that kind is first tried with the
        item's name."""
        lead = [term for term in fresh if len(term.split()) == 1 or WITH_UNIT.fullmatch(term)]
        terms = lead + [term for term in terms if term not in lead]
        if len(terms) == 1:
            return [[terms[0], want.called], terms] if lead else [terms]
        stated = [index for index, term in enumerate(terms) if term not in fresh]
        skip = stated[-1] if stated else 0
        return [terms, terms[:skip] + terms[skip + 1:]]

    def search_new_values(self, numbers: list) -> str:
        """After new answers, search each item's values together (once more without one value the request already
        stated) and read the first rows of the right kind."""
        shape = self.search_or_mode_shape()
        if not shape or self.steps_left < 22:
            return ""
        tool, fixed, field = shape
        notes: list = []
        allowance = [8]
        for number in numbers:
            pairs = self.value_terms(number)
            fresh = [term for term, heard in pairs if heard]
            if fresh and allowance[0] > 2:
                wordings = self._term_attempts(self.task.wants[number - 1], [term for term, _ in pairs], fresh)
                self._search_terms(tool, fixed, field, number, wordings, notes, allowance)
        if not notes:
            return ""
        self.link.log("value_lookup", seen=[note[:200] for note in notes])
        return ("[Wren searched each such item's condition values at once and looked at the first rows of "
                "the wanted kind] "
                + "\n".join(notes))

    def _search_terms(self, tool: str, fixed: dict, field: str, number: int, wordings: list, notes: list,
                      allowance: list) -> None:
        for words in filter(None, wordings[:2]):
            args = checked_args({**fixed, field: " ".join(words)}, self.tools.params.get(tool))
            if args is None or self.link.finished:
                return
            reply = self.call_store(tool, args)
            allowance[0] -= 1
            if reply is None:
                return
            observation = reply.get("observation")
            self.observe(tool, observation)
            rows = self._fitting_kind_rows(number, observation)
            notes.append(f"{as_json_text(args.get(field))}: " + (", ".join(rows[:6]) or "no row of the right kind"))
            if rows:
                opened = self.open_for_attributes([i for i in rows if i not in self.stock.pages][:min(3, allowance[0])],
                                                  self.closing_steps + 16)
                allowance[0] -= len(opened)
                if opened:
                    notes.append("read in full: " + " | ".join(opened)[:2400])
                return

    def _fitting_kind_rows(self, number: int, observation: Any) -> list:
        """Buyable rows of a result that belong to the item and its budget, those under the item's home label first."""
        want = self.task.wants[number - 1]
        key = self.shelf_key()
        refs = [ident for ident, _ in self.stock.records(observation)]
        cap = self.task.money_cap if isinstance(self.task.money_cap, (int, float)) else None
        if cap is not None and refs and all(price is not None and price > 10 * cap for price in map(self.stock.price, refs)):
            cap = None
        rows = [ident for ident in refs
                if self.stock.buyable(ident) and (not key or self.owns(want, self.shelf_of(ident, key)))
                and (cap is None or (self.stock.price(ident) or 0) <= cap)]
        home = self.home_shelf(number)
        if key and home:
            rows.sort(key=lambda ident: split_words(self.shelf_of(ident, key)) != home)
        return rows



# ---------------------------------------------------------------------------------------------------- guards
UNIT_FAULT_MARK = " is held at "


class GateRun:
    """One pass of the step gate before an order or a decline."""

    nudged = False

    def __init__(self, stage: str, due: list, tool_lines: list, endings: set):
        self.stage, self.due, self.tool_lines, self.endings = stage, due, tool_lines, endings
        self.said: list = []


class _GuardPart:
    """Everything that can hold a checkout or a decline back, and the steps the request wants done on the way."""

    def ends_in_order(self, tool: str, args: dict) -> bool:
        return self._finishes(tool, args, "buy")

    def ends_in_decline(self, tool: str, args: dict) -> bool:
        return self._finishes(tool, args, "refuse")

    def _finishes(self, tool: str, args: dict, job: str) -> bool:
        if tool != self.tools.jobs.get(job):
            return False
        fixed = self.task.reading.get(job + "_args")
        return all(args.get(name) == value for name, value in (fixed.items() if isinstance(fixed, dict) else ()))

    def other_listings(self, skip: list, limit: int = 30) -> str:
        order = self.stock.order[::-1]
        if self.task.pick == "cheapest":
            order.sort(key=lambda ident: (ident not in self.stock.pages, self._price_or_inf(ident)))
        lines: list = []
        for ident in order:
            if ident in skip or self.stock.sold_out(ident):
                continue
            lines.append(snippet(self.stock.record(ident), 160, 900))
            if len(lines) >= limit:
                break
        return "\n".join(lines)

    # -- steps the request asks for ---------------------------------------------------------------------------------
    def duty_gate(self, stage: str) -> tuple:
        """Before an order ('order') or a decline ('end'): let a model check the steps the request wants done by
        then against the record and make the calls still missing. Returns (go on, what was done or said)."""
        due = [duty for duty in self.task.duties if duty.stage == stage]
        if not due or self.link.finished or not self.duty_due(stage):
            return True, ""
        endings = self.tools.endings
        run = GateRun(stage, due, [tool_line(tool["function"]) for tool in self.tools.entries
                                   if tool["function"]["name"] not in endings], endings)
        for _ in range(DUTY_ROUNDS):
            outcome = self._gate_step(run)
            if outcome == "stop":
                break
            if outcome is not None:
                return outcome
        if run.said:
            return True, "Wren first completed the extra steps the request wants:\n" + "\n".join(run.said)
        return True, ""

    def _gate_step(self, run: GateRun) -> Any:
        """One round: None to go on, 'stop' to leave the gate, or the gate's (go on, text) result."""
        order = [STRONG_MODEL, QUICK_MODEL] if run.nudged else [QUICK_MODEL, STRONG_MODEL]
        answer = self.link.ask_for_object(self._duty_prompt(run), order, 3500)
        if answer is None:
            return "stop"
        learned, hold = [], ""
        if run.said:
            self.take_answers({"needs": answer.get("needs")}, learned)
            hold = str(answer.get("hold") or "").strip()
        call = answer.get("call") if isinstance(answer.get("call"), dict) else None
        self.link.log("duty_check", stage=run.stage, call=shorten(call, 200), steps=shorten(answer.get("steps"), 120),
                      hold=hold[:200], told=learned)
        if learned or hold:
            self.track.epoch += 1
            done = "Wren first completed the extra steps the request wants:\n" + "\n".join(run.said) + "\n" if run.said else ""
            change = "; ".join(learned + [hold] if hold else learned)
            return False, done + f"The customer's reply changes the task: {change}. Adjust the cart to it, then check out again."
        if not call:
            if self._nudge(run, answer):
                return None
            self.track.duties_cleared.add((run.stage, self.cart_epoch))
            return "stop"
        tool = str(call.get("tool") or "")
        if tool not in self.tools.params or tool in run.endings:
            return "stop"
        asks = tool == self.tools.jobs.get("ask")
        if self.steps_left < 2 or (asks and self.track.msgs_sent >= self.task.msg_cap):
            return "stop"
        args = checked_args(call["args"] if isinstance(call.get("args"), dict) else {}, self.tools.params[tool])
        if args is None:
            return "stop"
        if asks:
            text = next((value for value in args.values() if isinstance(value, str)), "")
            batch = self.send_messages([(1, "", text)], 1)
            if not batch:
                return "stop"
            self.track.duty_msgs += 1
            run.said.append(f"Question: {text}\nAnswer: {batch[0][2]}")
            return None
        reply = self.call_store(tool, args)
        if reply is None:
            return "stop"
        seen = reply.get("observation")
        self.stock.take_in(seen)
        self.track.deed_log.append(f"step {self.link.step}: called {tool} {as_json_text(args)}; it returned "
                                   f"{self.show(seen)[:300]} (cart at that time: {self.basket.mark()})")
        self.track.deed_steps.append(self.link.step)
        run.said.append(f"{tool} {as_json_text(args)} => {self.show(seen)[:400]}")
        return None

    def _nudge(self, run: GateRun, answer: dict) -> bool:
        """The model named no call but reports a step without a record line that proves it: say so in the record
        and ask once more (with the strong model first)."""
        if run.said or run.nudged or not self.tools.jobs.get("ask"):
            return False
        if self.track.msgs_sent >= self.task.msg_cap or self.steps_left < 2:
            return False
        unproven = self._unproven_steps(run, answer)
        if not unproven:
            return False
        run.nudged = True
        self.link.log("duty_unproven", steps=shorten(unproven, 120))
        named = "; ".join(str(entry.get("step") or "") for entry in unproven)[:200]
        self.track.deed_log.append(
            f"(Wren: after the cart's latest change at step {self.basket.changed_at}, no line proves {named} done; "
            f"{self.task.msg_cap - self.track.msgs_sent} message(s) and {self.steps_left} steps remain to spare, so "
            f"say which call would do it)")
        self.track.deed_steps.append(-1)
        return True

    def _duty_prompt(self, run: GateRun) -> str:
        ordering = run.stage == "order"
        record = "\n".join(f"{index}. {line}" for index, line in enumerate(self.track.deed_log, 1)) or "(empty so far)"
        parts = [STEP_CHECK.replace("<<MOVE>>", "place the order" if ordering else "end without a purchase"),
                 "\n\nTHE REQUEST:\n", self.task.query, "\n\nSTEPS DUE:\n", "\n".join(f"- {duty.text}" for duty in run.due),
                 "\n\nTASK LEDGER:\n", self.ledger_text(), "\n\nCURRENT CART: ", self.basket.describe(),
                 "\n\nLOG (every message and step call up to now):\n", record]
        if self.track.news:
            parts += ["\n\nSHOP NOTES:\n", "\n".join(self.track.news[-4:])]
        parts.append(f"\n\nMessages remaining: {max(0, self.task.msg_cap - self.track.msgs_sent)}; steps remaining: "
                     f"{self.steps_left} (the {'order' if ordering else 'ending'} itself uses one)")
        parts += ["\n\nAVAILABLE TOOLS:\n", "\n".join(run.tool_lines)]
        return "".join(parts)

    def _unproven_steps(self, run: GateRun, answer: dict) -> list:
        """Steps the model calls done that no record line backs up. When a step that sends a message belongs to the
        order, only lines logged after the latest cart change count."""
        speaks = run.stage == "order" and any(duty.messages > 0 for duty in run.due)
        steps = self.track.deed_steps
        proof = {line for line in range(1, len(self.track.deed_log) + 1)
                 if not speaks or (line <= len(steps) and steps[line - 1] > self.basket.changed_at)}
        reported = filter(lambda entry: isinstance(entry, dict), listed(answer.get("steps")))
        return [entry for entry in reported if entry.get("done") is not True
                or all((whole_number(line) or 0) not in proof for line in listed(entry.get("lines")))]

    # -- the cart review --------------------------------------------------------------------------------------------
    def review_key(self, refs: list) -> str:
        return "|".join(sorted(refs)) + f"#{self.track.epoch}"

    def review_cart(self, refs: list | None = None, limit: float = SPEND_CALM) -> str:
        """A strong-model check of the cart (or of a planned set of lines) against the records. Returns a hold
        message when it finds the cart wrong, else ''."""
        planned = refs is not None
        refs = refs if planned else self.basket.refs(self.stock.rows)
        key = self.review_key(refs)
        if key in self.track.reviewed_keys or not self.review_models(limit) or not refs:
            return ""
        self.track.reviewed_keys.add(key)
        parts = [BASKET_REVIEW_TEXT, "\n\nTHE REQUEST:\n", self.task.query, "\n\nCONDITIONS IN FORCE:\n", self.ledger_text()]
        if planned:
            units = "; ".join(f"item {n}: quantity {want.units}" for n, want in enumerate(self.task.wants, 1))
            parts += ["\n\nPLANNED CART (treat each listing below as sitting in the cart at the quantity its item requests: ",
                      units, "):\n", ", ".join(refs)]
        else:
            parts += ["\n\nTHE CART:\n", self.basket.describe()]
        flags = ["SOLD OUT (no longer buyable, says the shop). " if self.stock.sold_out(i) else "" for i in refs]
        parts += ["\n\nTHE CART LISTINGS' RECORDS:\n",
                  "\n".join(f"{i}: {flag}{self.stock.text_of(i)}" for i, flag in zip(refs, flags))]
        ranked = self.task.pick in ("cheapest", "most", "least")
        if ranked:
            parts += ["\n\nOTHER LISTINGS ON RECORD (newest first):\n", self.other_listings(refs, 15)]
        answer = self._review_verdict("".join(parts), limit, planned or ranked or self.track.decline_reviewed)
        self.link.log("basket_review", cart=refs, got=answer)
        rejected = bool(answer) and answer.get("ok") is False
        self.track.review_verdicts[key] = bool(answer) and not rejected
        if not rejected:
            return ""
        self.track.candidates = [ident for ident in self.track.candidates if ident not in refs]
        problems = [str(entry) for entry in listed(answer.get("problems"))][:5] or ["see the conditions"]
        better = [str(entry) for entry in listed(answer.get("better"))
                  if str(entry) in self.stock.rows and not self.stock.sold_out(str(entry))][:4]
        text = "A second look at this cart found problems: " + "; ".join(problems)
        if better:
            text += ". Listings that look better: " + ", ".join(better)
        return text + "; repair the cart, or check the records again and point out where the review errs, then check out."

    def _review_verdict(self, prompt: str, limit: float, full_panel: bool) -> dict | None:
        """The review's answer. The full panel of models decides a planned cart, a ranked pick and any cart after a
        decline review. Otherwise, with two models to hand, the quick one reads first and the strong one is called
        only to hear a rejection out."""
        models = self.review_models(limit)
        answer = None
        if len(models) > 1 and not full_panel:
            answer = self.link.ask_for_object(prompt, [QUICK_MODEL], 3000)
            if answer:
                second = answer.get("ok") is False and STRONG_MODEL in self.review_models(limit)
                models = [STRONG_MODEL] if second else []
        if not models or self.link.model_broke:
            return answer
        return self.link.ask_for_object(prompt, models, 3000) or answer

    # -- requirements that point at another listing -----------------------------------------------------------------
    def _chain_records(self) -> list:
        newest = self.stock.order[::-1]
        opened = [self.stock.record(ident) if ident in self.stock.opened else None for ident in newest]
        records = ["FULL PAGE " + snippet(entry, 300, 1400) for entry in opened if entry][:45]
        for ident, entry in zip(newest, opened):
            if len(records) >= 60:
                break
            if not entry:
                records.append("SEARCH ROW " + snippet(self.stock.rows.get(ident), 80, 300))
        return records

    def follow_pointers(self) -> tuple:
        """Resolve every condition whose value sits on another listing: trace the chain of pages and open those not
        yet open. Returns (text for the loop, whether to hold the checkout)."""
        live = {need for _, need in self.task.live_needs()}
        pending = [need for _, need in self.task.pointers if need in live and need not in self.task.pointer_values]
        tool = self.tools.offer_view
        param = self.tools.single_param(tool)
        seen: list = []
        lost: list = []
        while pending and self.track.chain_rounds < (6 if self.steps_left > 15 else 4) and self.link.spent < SPEND_CEILING:
            self.track.chain_rounds += 1
            prompt = "".join((CHAIN_TRACE, "\n\nCONDITIONS VALUED BY ANOTHER LISTING:\n", "\n".join(pending),
                              "\n\nTHE RECORDS (newest first):\n", "\n".join(self._chain_records())))
            models = [QUICK_MODEL, STRONG_MODEL] if self.link.spent + STRONG_HEADROOM <= SPEND_CEILING else [QUICK_MODEL]
            answer = self.link.ask_for_object(prompt, models, 4500) or {}
            self.link.log("chain_round", pending=pending, got=shorten(answer, 150))
            reach, lost, gaps = self._read_chains(answer, pending)
            pending = [need for need in pending if need not in self.task.pointer_values]
            searchable = tool and param and self.steps_left > 12 and self.track.link_search_count < 3
            if not reach and gaps and searchable:
                reach, known = self.find_linked(*gaps[0])
                if known:
                    seen.append(f"{known}: already open, it is the named shop's page with the description's values; "
                                f"read the value there")
                    lost = lost[1:]
            if not (reach and tool and param) or self.steps_left < len(reach) + 4 or self.link.finished:
                break
            for ident in reach[:max(0, min(3, 6 - len(seen)))]:
                reply = self._open_one(tool, param, ident)
                if reply is None or self.link.finished:
                    break
                self.observe(tool, reply.get("observation"))
                seen.append(f"{ident}: " + self.show(reply.get("observation"))[:1500])
        return self._chain_report(seen, lost)

    def _chain_report(self, seen: list, lost: list) -> tuple:
        text = ""
        if seen:
            text += "[Wren opened the pages that conditions point at]\n" + "\n".join(seen) + "\n"
        if self.task.pointer_values:
            values = "; ".join(f"{need} => {value}" for need, value in self.task.pointer_values.items())
            text += f"[values taken from the pages conditions point at] {values}\n"
        hold = bool(lost) and self.track.chain_holds < (3 if self.steps_left > 15 else 2) and self.steps_left > 5
        if hold:
            self.track.chain_holds += 1
            text += ("Checkout is only valid after every page a condition points to is fully open (search rows are "
                     "not enough), including each further listing a page sends you to. Still missing: "
                     + " | ".join(lost[:3]) + ". Look it up by the shop's name plus the values in its description "
                     "(other listings of that kind from the same shop do not qualify); open the shop's row whose page "
                     "carries those values, use the value on that page, then check out.")
        return text, hold

    def _read_chains(self, answer: dict, pending: list) -> tuple:
        """Book every chain the model resolved; return (rows to open, what is still lost, (about, search words))."""
        reach: list = []
        lost: list = []
        gaps: list = []
        for chain in filter(lambda entry: isinstance(entry, dict), listed(answer.get("chains"))):
            said = str(chain.get("need") or "").strip()
            need = said if said in pending else None
            need = need or (pending[0] if len(pending) == 1 else None)
            if not need:
                continue
            links = [link for link in listed(chain.get("links")) if isinstance(link, dict)]
            refs = [str(link.get("id") or "").strip() for link in links]
            all_open = bool(links) and set(refs) <= self.stock.opened
            if all_open and (chain.get("done") is True and str(chain.get("value") or "").strip()
                             or not str(links[-1].get("find") or "").strip()):
                if chain.get("done") is True and str(chain.get("value") or "").strip():
                    self.task.pointer_values[need] = str(chain["value"]).strip()
                else:
                    self.task.pointer_values[need] = str(chain.get("value") or "").strip() or f"as stated on {refs[-1]}"
                self.track.epoch += 1
                continue
            closed = [ident for ident in refs if ident in self.stock.rows and ident not in self.stock.opened]
            if closed:
                reach += [ident for ident in closed if ident not in reach]
                continue
            gap = next((link for link, ident in zip(links, refs) if ident not in self.stock.rows), {})
            about = str(gap.get("about") or "") or "the listing it points at"
            words = str(gap.get("find") or "").strip()
            lost.append(f"{need} => still to find: {about}" + (f" (search words: {words})" if words else ""))
            gaps.append((about, words))
        return reach, lost, gaps

    def _shop_names(self, ident: str) -> list:
        row = self.stock.rows.get(ident) or {}
        return [str(cell).strip() for key, cell in row.items()
                if isinstance(cell, str) and 2 < len(cell.strip()) <= 40 and SELLER_KEY.search(str(key))]

    @staticmethod
    def _word_in(word: str, text: str) -> bool:
        return re.search(r"(?<![a-z0-9])" + re.escape(word) + r"(?![a-z0-9])", text) is not None

    def seller_rows(self, about: str, refs: list) -> list:
        """Unopened listings whose shop or brand the description names."""
        lowered = about.lower()
        return [ident for ident in refs if ident not in self.stock.opened
                and any(self._word_in(name.lower(), lowered) for name in self._shop_names(ident))]

    def opened_shop_pages(self, about: str) -> list:
        lowered = about.lower()
        return [ident for ident in self.stock.opened
                if any(self._word_in(name.lower(), lowered) for name in self._shop_names(ident))]

    def _shop_words(self, about: str) -> set:
        words: set = set()
        for ident in self.seller_rows(about, list(self.stock.rows)) + self.opened_shop_pages(about):
            for key, cell in (self.stock.rows.get(ident) or {}).items():
                if isinstance(cell, str) and len(cell.strip()) <= 40 and SELLER_KEY.search(str(key)):
                    words |= set(ALNUM3.findall(cell.lower()))
        return words

    @staticmethod
    def _leaf_text(entry: Any) -> str:
        return " ".join(str(node) for node in lifo_nodes(entry) if not isinstance(node, (dict, list))).lower()

    def find_linked(self, about: str, words: str) -> tuple:
        """Locate the listing a chain describes. First choice: a page of that shop, already open, that shows the
        description's values (the second value returned); otherwise a search by shop and values. Returns (rows to
        open, known page)."""
        shop_words = self._shop_words(about)
        source = (words or about).lower()
        values = {word for word in ALNUM.findall(source) if len(word) >= 3 or word.isdigit()}
        values = values - shop_words - LINK_STOPWORDS - {"of"}
        for ident in self.opened_shop_pages(about):
            text = self._leaf_text(self.stock.record(ident) or {})
            if values and all(self._word_in(value, text) for value in values):
                self.link.log("chain_page_known", id=ident)
                return [], ident
        known = self._link_search(about, words)
        kind = set(ALNUM3.findall(source)) - shop_words - LINK_STOPWORDS

        def same_kind(refs: list) -> list:
            if not kind:
                return list(refs)
            return [ident for ident in refs
                    if kind & set(ALNUM3.findall(as_json_text(self.stock.rows.get(ident) or {}).lower()))]

        return (same_kind(known) or same_kind(self.seller_rows(about, self.stock.order[::-1])))[:2], None

    def _link_search(self, about: str, words: str) -> list:
        memo = " ".join(words.lower().split())
        shape = self.search_shape()
        if memo and memo in self.track.link_search_memo:
            self.track.link_search_count += 1
            return self.seller_rows(about, self.track.link_search_memo[memo])
        if not (shape and words):
            return []
        tool, fixed, field = shape
        args = checked_args({**fixed, field: words}, self.tools.params.get(tool))
        if args is None:
            return []
        self.track.link_search_count += 1
        reply = self.call_store(tool, args)
        if reply is None:
            return []
        observation = reply.get("observation")
        self.observe(tool, observation)
        self.track.link_search_memo[memo] = [ident for ident, _ in self.stock.records(observation)]
        found = self.seller_rows(about, self.track.link_search_memo[memo])
        self.link.log("chain_search", words=words[:80], rows=found[:4])
        return found

    # -- sell-outs and re-reads ------------------------------------------------------------------------------------
    def reread_cart_offers(self) -> str:
        """Re-read the offer of each cart line not read since the last refusal or two steps ago."""
        tool = self.tools.offer_view
        param = self.tools.single_param(tool)
        if not (tool and param):
            return ""
        since = max(self.link.step - 2, self.track.refused_step)
        stale = [ident for ident in self.basket.refs(self.stock.rows)
                 if self.stock.offer_read_at.get(ident, -99) < since and ident not in self.stock.gone]
        if not stale or self.steps_left < len(stale) + 3:
            return ""
        lines = []
        for ident in stale:
            args = checked_args({param: ident}, self.tools.params.get(tool))
            if args is None:
                break
            before = dict(self.stock.rows.get(ident) or {})
            reply = self.call_store(tool, args)
            if reply is None or self.link.finished:
                break
            observation = reply.get("observation")
            self.stock.offer_read_at[ident] = self.link.step
            self.observe(tool, observation)
            moved = [key for key, cell in (self.stock.rows.get(ident) or {}).items()
                     if key in before and before[key] != cell and not isinstance(cell, (dict, list))]
            if moved:
                lines.append(f"{ident} has changed ({', '.join(moved[:4])}): {snippet(observation, 300, 900)}")
        self.link.log("offers_reread", ids=stale, changed=len(lines))
        if lines:
            self.track.epoch += 1
        return "\n".join(lines)

    def sellout_hint(self) -> str:
        """Cart lines newly known to be sold out: book them as gone and point at fitting stand-ins."""
        lost = [ident for ident in self.basket.refs(self.stock.rows)
                if ident not in self.stock.gone and self.stock.sold_out(ident)]
        if not lost:
            return ""
        self.stock.gone.update(dict.fromkeys(lost, self.link.step))
        self.track.candidates = [ident for ident in self.track.candidates if ident not in lost]
        self.track.epoch += 1
        picks = self.stand_ins(lost)
        text = (f"[Wren] Now sold out: {', '.join(lost)} (the shop shows it as unbuyable). It will not return, so do "
                f"not open it again.")
        if not picks:
            return text + (" Take that line out and put the strongest remaining listing that fits in its place; if "
                           "none you have read fits, search for one.")
        if self.steps_left <= 6:
            return text + (f" Few steps remain: in this turn, two calls: add {picks[0]} at the requested quantity; "
                           f"remove the sold-out line. Then check out. Further read listings that look like fits: "
                           f"{', '.join(picks[1:]) or 'none'}.")
        if self.task.pick == "cheapest":
            return text + (f" Stand-ins for that item among listings you have read, by price, lowest first: "
                           f"{', '.join(picks)}. Under the cheapest rule, do one search priced below {picks[0]} first "
                           f"(with the search's maximum price, where it has one). Skip it only if you already know every "
                           f"fitting listing. Then swap in the lowest-priced fit for the sold-out line.")
        return text + (f" Stand-ins for that item among listings you have read: {', '.join(picks)}. Now, in one "
                       f"turn: the sold-out line goes out; {picks[0]} goes in, at the quantity requested. Then check "
                       f"out.")

    def find_sold_line(self) -> str:
        """A refusal or notice says a cart line sold out without naming it: with one line that is the one; with
        several, re-read their offers."""
        refs = [ident for ident in self.basket.refs(self.stock.rows) if ident not in self.stock.gone]
        if not refs or any(map(self.stock.sold_out, refs)):
            return ""
        if len(refs) > 1:
            return self.reread_cart_offers()
        self.stock.unavailable.add(refs[0])
        return ""

    def stand_ins(self, lost: list) -> list:
        if not self.review_models():
            return []
        records = []
        for ident in self.stock.order[::-1]:
            if ident in lost or ident in self.stock.gone or self.stock.sold_out(ident):
                continue
            records.append(snippet(self.stock.record(ident), 200, 1200))
            if len(records) == 25:
                break
        if not records:
            return []
        prompt = "".join((STANDIN_SEARCH, "\n\nTHE REQUEST:\n", self.task.query, "\n\nCONDITIONS IN FORCE:\n",
                          self.ledger_text(), "\n\nTHE SOLD-OUT CART LINE:\n",
                          "\n".join(f"{ident}: {self.stock.text_of(ident)}" for ident in lost), "\n\nTHE CART NOW:\n",
                          self.basket.describe(), "\n\nOTHER LISTING RECORDS (newest first):\n", "\n".join(records)))
        answer = self.link.ask_for_object(prompt, self.review_models()[::-1], 3500)
        fits: list = []
        for entry in listed((answer or {}).get("fits")):
            ident = str(entry.get("id") if isinstance(entry, dict) else entry)
            price = entry.get("price") if isinstance(entry, dict) else None
            usable = ident in self.stock.rows and ident not in lost and ident not in self.stock.gone
            if usable and ident not in self.basket.refs(self.stock.rows) and ident not in [ref for _, ref in fits]:
                fits.append((float(price) if is_number(price) else float("inf"), ident))
        ordered = sorted(fits, key=lambda pair: pair[0]) if self.task.pick == "cheapest" else fits
        self.link.log("standins", lost=lost, fits=fits)
        return [ident for _, ident in ordered][:4]

    # -- before a decline -------------------------------------------------------------------------------------------
    def decline_review(self) -> str:
        """Before the first decline: a strong-model look for listings that may fit after all (opening a few rows the
        model asks for when it names none)."""
        if self.track.decline_reviewed or not self.review_models():
            return ""
        self.track.decline_reviewed = True
        answer = self._decline_review_call()
        named = self._review_fits(answer)
        if not named:
            rows = [str(entry) for entry in listed((answer or {}).get("open"))]
            rows = [ident for ident in rows if ident in self.stock.rows and ident not in self.stock.opened
                    and not self.stock.sold_out(ident)]
            if rows and self.open_review_rows(rows):
                answer = self._decline_review_call()
                named = self._review_fits(answer)
        if not named:
            return ""
        self.track.candidates_by_item = named
        self.track.candidates = [ident for refs in named.values() for ident in refs]
        listing = "; ".join(f"item {number}: {', '.join(refs)}" for number, refs in named.items())
        reason = f" ({answer.get('reason')})" if answer.get("reason") else ""
        return (f"An independent review thinks these listings could fit: {listing}{reason}. Open them and buy "
                f"whatever truly fits; close without buying only if every one of them fails.")

    def _decline_review_call(self) -> dict | None:
        records = []
        for ident in self.stock.order[::-1]:
            if self.stock.sold_out(ident):
                continue
            page = self.stock.pages.get(ident)
            records.append("FULL PAGE " + snippet(page, 200, 1200) if page else "SEARCH ROW " + snippet(self.stock.rows.get(ident), 120, 500))
            if len(records) == 30:
                break
        if not records:
            return None
        prompt = "".join((NOFIT_REVIEW, "\n\nTHE REQUEST:\n", self.task.query, "\n\nCONDITIONS IN FORCE:\n",
                          self.ledger_text(), "\n\nLISTING RECORDS (newest first):\n", "\n".join(records)))
        models = self.review_models()
        answer = self.link.ask_for_object(prompt, [QUICK_MODEL] if models else [], 4000)
        if STRONG_MODEL in models and not self.link.model_broke and not self._review_fits(answer):
            answer = self.link.ask_for_object(prompt, [STRONG_MODEL], 4000) or answer
        self.link.log("decline_review", got=answer)
        return answer

    def _review_fits(self, answer: dict | None) -> dict:
        fits = (answer or {}).get("fits")
        if not isinstance(fits, dict):
            return {}
        usable = lambda ident: ident in self.stock.rows and not self.stock.sold_out(ident)
        named = {str(number): [ident for ident in map(str, listed(refs)) if usable(ident)][:4] for number, refs in fits.items()}
        return {number: refs for number, refs in named.items() if refs}

    def open_review_rows(self, rows: list) -> bool:
        side, joiner = self.tools.side_param, self.tools.jobs.get("compare")
        tool = self.tools.attr_view
        param = self.tools.single_param(tool)
        if side and joiner and not self.tools.side_bare:
            plan = [(joiner, {side: rows[:self.tools.side_room]})]
        elif tool and param:
            plan = [(tool, {param: ident}) for ident in rows[:2]]
        else:
            return False
        if self.steps_left < len(plan) + 5:
            return False
        before = len(self.stock.pages)
        for name, given in plan:
            args = checked_args(given, self.tools.params.get(name))
            if args is None:
                return False
            reply = self.call_store(name, args)
            if reply is None or self.link.finished:
                break
            self.observe(name, reply.get("observation"))
        self.link.log("review_open", rows=rows, opened=len(self.stock.pages) - before)
        return len(self.stock.pages) > before

    # -- cart calls made by the episode itself ----------------------------------------------------------------------
    def _line_list_call(self, fields: dict, lines: dict) -> Any:
        """A cart tool that takes the whole cart as a list of line objects: one call, or None when it cannot."""
        for key, spec in fields.items():
            shape = spec.get("items") if isinstance(spec, dict) and "array" in schema_types(spec) else None
            if not isinstance(shape, dict) or not isinstance(shape.get("properties"), dict):
                continue
            inner = shape["properties"]
            ref_key = next(iter(text_fields(inner)), None)
            units_key = first_field(inner, "integer")
            if not ref_key:
                return None
            return [{key: [dict([(ref_key, ident)] + ([(units_key, units)] if units_key else []))
                           for ident, units in lines.items()]}]
        return False

    def cart_plan_calls(self, lines: dict, drop: list) -> list | None:
        """The cart calls that leave exactly `lines` (reference -> units) after taking `drop` out; None when the cart
        tool cannot be driven that way."""
        tool = self.tools.jobs.get("cart") or ""
        fields = self.tools.fields(tool)
        whole = self._line_list_call(fields, lines)
        if whole is not False:
            return whole
        target, count, mode = self.tools.cart_params(tool)
        if not target:
            return None
        sets = self.tools.sets_units(tool)
        calls = []
        for ident in drop:
            have = self.basket.lines.get(ident, 1)
            if mode:
                call = {target: ident, mode: fields[mode]["enum"][1], **({count: have} if count else {})}
            elif count and sets:
                call = {target: ident, count: 0}
            elif count and isinstance(fields[count].get("minimum"), int) and fields[count]["minimum"] <= -have:
                call = {target: ident, count: -have}
            else:
                return None
            calls.append(call)
        in_cart = self.basket.refs(self.stock.rows)
        for ident, want in lines.items():
            have = self.basket.lines.get(ident, 0) if ident in in_cart else 0
            if have != want:
                if have and not sets:
                    return None
                calls.append({target: ident, **({mode: fields[mode]["enum"][0]} if mode else {}),
                              **({count: want} if count else {})})
        return calls

    def _kind_of(self, key: Any) -> Any:
        if not key:
            return lambda ident: ident
        return lambda ident: str((self.stock.rows.get(ident) or {}).get(key) or "").strip().lower()

    def _confirmed_plan(self) -> tuple | None:
        """(lines, drop, calls) that put the reviewed listings in the cart, or None."""
        kind = self._kind_of(self.shelf_key())
        plan: dict = {}
        single = 1 if len(self.task.wants) == 1 else None
        for number, refs in self.track.candidates_by_item.items():
            digits = DIGIT_RUN.search(str(number))
            index = int(digits.group()) if digits else single
            choice = next((ident for ident in refs if not self.stock.sold_out(ident)), None)
            if choice and index and 1 <= index <= len(self.task.wants) and kind(choice) not in set(map(kind, plan)):
                plan[choice] = self.task.wants[index - 1].units
        if self.track.confirmed_pick not in plan:
            return None
        now = self.basket.refs(self.stock.rows)
        planned_kinds = set(map(kind, plan))
        keep = {ident: self.basket.lines.get(ident, 1) for ident in now
                if ident not in plan and not self.stock.sold_out(ident) and kind(ident) not in planned_kinds}
        drop = [ident for ident in now if ident not in plan and ident not in keep]
        if len(plan) + len(set(map(kind, keep))) < len(self.task.wants):
            return None
        lines = {**keep, **plan}
        calls = self.cart_plan_calls(lines, drop)
        if not calls or len(calls) + 1 > self.steps_left:
            return None
        return lines, drop, calls

    def buy_confirmed(self) -> str:
        """After a review named a fitting listing and a second check confirmed it: set the cart to the reviewed
        lines and check out in place of the decline the loop asked for."""
        cart, buy = self.tools.jobs.get("cart"), self.tools.jobs.get("buy")
        pick = self.track.confirmed_pick
        if self.track.auto_bought or not pick or self.stock.sold_out(pick) or not cart or not buy:
            return ""
        self.track.auto_bought = True
        prepared = self._confirmed_plan()
        if prepared is None:
            return ""
        lines, drop, calls = prepared
        refs = list(lines)
        self.open_for_attributes(refs, len(calls) + 3)
        self.follow_part_pointers(refs, len(calls) + 3)
        self.review_cart(refs, SPEND_CEILING)
        if not self.track.review_verdicts.get(self.review_key(refs)):
            return ""
        done = []
        for given in calls:
            args = checked_args(given, self.tools.params.get(cart))
            reply = self.call_store(cart, args) if args is not None else None
            if reply is None or self.link.finished:
                return ""
            self.book_cart_call(cart, args)
            self.observe(cart, reply.get("observation"))
            done.append(as_json_text(args))
        order_args = checked_args(dict(self.task.reading.get("buy_args") or {}), self.tools.params.get(buy))
        if order_args is None:
            return ""
        replies = self._confirmed_checkout(buy, order_args, refs, lines, drop)
        if not replies:
            return ""
        return (f"Not made (Wren acted instead): an independent check confirmed {pick}, so Wren set the cart "
                f"({'; '.join(done)}) and checked out. The shop replied: " + " | after that: ".join(replies))

    def _confirmed_checkout(self, buy: str, args: dict, refs: list, lines: dict, drop: list) -> list:
        replies: list = []
        for attempt in (0, 1):
            if attempt:
                self.reread_cart_offers()
                self.review_cart(refs, SPEND_CEILING)
                verdict = self.track.review_verdicts.get(self.review_key(refs))
                if not verdict or self.steps_left < 1 or self.link.finished or any(map(self.stock.sold_out, lines)):
                    break
            go, said = self.duty_gate("order")
            if said:
                replies.append(said)
            if not go:
                break
            reply = self.call_store(buy, args)
            self.link.log("confirmed_buy", lines=lines, drop=drop, reply=shorten(reply, 200))
            if reply is None:
                break
            self.track.checkouts += 1
            if not reply.get("finished"):
                self.track.refused_step = self.link.step
            self.observe(buy, reply.get("observation"))
            replies.append(self.show(reply.get("observation")))
            if self.link.finished or reply.get("finished"):
                break
        return replies

    def book_cart_call(self, tool: str, args: dict) -> None:
        fields = fields_of(self.tools.params.get(tool, {}))
        self.basket.apply(args, fields, self.stock.rows, self.link.step, self.tools.sets_units(tool))

    # -- fit between parts and devices ------------------------------------------------------------------------------
    def compat_index(self) -> tuple:
        """Per listing its compatibility marks, and per part code the codes other listings say it also fits."""
        marks: dict = {}
        notes: dict = {}
        for ident in self.stock.order:
            entry = self.stock.record(ident) or {}
            plain = [str(cell) for cell in entry.values() if isinstance(cell, str)]
            nested = [f"{key} {cell}" for block in entry.values() if isinstance(block, dict)
                      for key, cell in block.items() if isinstance(cell, str)]
            marks[ident] = compat_marks(" ".join(plain) + " " + ". ".join(nested))
            for part, codes in marks[ident]["also"]:
                notes.setdefault(part, set()).update(codes)
        return marks, notes

    @staticmethod
    def fits_codes(ident: str, marks: dict, notes: dict) -> set:
        mark = marks.get(ident) or {}
        found = set(mark.get("works") or [])
        if mark.get("own"):
            found |= notes.get(mark["own"], set())
        return found

    def follow_part_pointers(self, refs: list, keep: int) -> None:
        """A cart line whose text sends the reader to a part code no seen listing carries: search that code, once per
        code, and open what it finds if steps allow. It feeds the books, not the model. Multi-line carts only."""
        shape = self.search_or_mode_shape()
        if not shape or max(len(self.task.wants), len(refs)) < 2:
            return
        tool, fixed, field = shape
        for ident in refs:
            marks = self.compat_index()[0]
            code = (marks.get(ident) or {}).get("pointer")
            carried = {mark.get("own") for mark in marks.values()}
            if not code or code in self.track.pointer_codes or code in carried:
                continue
            if self.steps_left <= keep or self.link.finished:
                return
            self.track.pointer_codes.add(code)
            args = checked_args({**fixed, field: code}, self.tools.params.get(tool))
            reply = self.call_store(tool, args) if args is not None else None
            if reply is None:
                return
            self.observe(tool, reply.get("observation"))
            self.link.log("part_pointer", line=ident, code=code)
            found = [other for other, mark in self.compat_index()[0].items() if mark.get("own") == code]
            self.open_for_attributes(found, max(keep, self.closing_steps + 6))

    def pairing_gaps(self) -> list:
        """Cart parts whose own words say they work with other models than the cart's device."""
        if not self.task.pairs:
            return []
        marks, notes = self.compat_index()
        cart = self.basket.refs(self.stock.rows)
        faults = []
        for part, main in self.task.pairs:
            devices = [ident for ident in cart if ident in self.item_refs(main)]
            parts = [ident for ident in cart if ident in self.item_refs(part)]
            codes = set(marks.get(devices[0], {}).get("codes") or []) if len(devices) == 1 and parts else set()
            for line in parts if codes else []:
                works = self.fits_codes(line, marks, notes)
                if works and not works & codes:
                    faults.append(self._misfit_text(line, part, devices[0], works, codes, marks, notes, cart))
        return faults

    def _misfit_text(self, line: str, part: int, device: str, works: set, codes: set, marks: dict, notes: dict,
                     cart: list) -> str:
        good = [ident for ident in self.item_refs(part) if self.fits_codes(ident, marks, notes) & codes
                and not self.stock.sold_out(ident) and ident not in cart][:4]
        text = (f"{line} ({self.task.wants[part - 1].called}) is made for {', '.join(sorted(works)[:4])}, not for "
                f"{device} ({', '.join(sorted(codes))})")
        if good:
            return text + f"; parts of that kind already seen that do suit it: {', '.join(good)}"
        return text + "; no part of that kind seen so far suits it, so search for one or change the device: " + self.pairing_plan(short=True)

    def pairing_plan(self, short: bool = False) -> str:
        """Per device item, the model codes its candidate parts name, most named first, and the devices seen
        carrying each code."""
        if not self.task.pairs:
            return ""
        marks, notes = self.compat_index()
        lines = []
        for main in sorted({main for _, main in self.task.pairs}):
            tally: dict = {}
            for part in [part for part, owner in self.task.pairs if owner == main]:
                for ident in filter(lambda ident: not self.stock.sold_out(ident), self.item_refs(part)):
                    for code in self.fits_codes(ident, marks, notes):
                        per = tally.setdefault(code, {})
                        per[part] = per.get(part, 0) + 1
            ranked = sorted(tally.items(), key=lambda pair: (-len(pair[1]), -sum(pair[1].values())))[:4]
            pieces = [self._code_piece(code, per, main, marks) for code, per in ranked]
            if pieces:
                lines.append(f"model codes named by parts for the {self.task.wants[main - 1].called} => " + "; ".join(pieces))
        if not lines:
            return ""
        joined = " | ".join(lines)
        if short:
            return joined
        return ("[fit plan: part X goes with device Y when X's wording carries Y's model code; another sales name of "
                "Y also counts. Searching by code does not work: browse the device's category instead and compare the "
                "codes its rows show] " + joined)

    def _code_piece(self, code: str, per: dict, main: int, marks: dict) -> str:
        devices = [ident for ident in self.item_refs(main) if code in (marks.get(ident, {}).get("codes") or [])][:2]
        counts = ", ".join(f"{n} {self.task.wants[p - 1].called}" for p, n in sorted(per.items()))
        seen = f" (devices seen: {', '.join(devices)})" if devices else " (no device seen yet)"
        return f"{code}: {counts}{seen}"

    # -- cart lines against the request -----------------------------------------------------------------------------
    def basket_gaps(self) -> list:
        """What the cart's lines get wrong: a missing or doubled item, wrong units, a label short of the item's full
        name, a kind not asked for, a sold-out line, the budget."""
        self.adopt_stray_labels()
        key = self.shelf_key()
        refs = [ident for ident in self.basket.refs(self.stock.rows) if (self.basket.units(ident) or 0) > 0]
        if not (refs and key):
            return []
        label = {ident: self.shelf_of(ident, key) for ident in refs}
        known = {number for number, want in enumerate(self.task.wants, 1)
                 if any(self.owns(want, str(row.get(key) or "")) for row in self.stock.rows.values())}
        owner = {ident: [n for n, want in enumerate(self.task.wants, 1) if self.owns(want, label[ident])] for ident in refs}
        faults = [fault for number, want in enumerate(self.task.wants, 1) if number in known
                  for fault in self._item_faults(number, want, refs, owner, label)]
        if len(known) == len(self.task.wants):
            faults += [f"{ident} ({label[ident] or 'unknown kind'}) is a kind the customer did not ask for: remove it"
                       for ident in refs if not owner[ident]]
        faults += [f"{ident} is sold out" for ident in refs if self.stock.sold_out(ident)]
        return faults + self._budget_faults(refs)

    def _item_faults(self, number: int, want: Want, refs: list, owner: dict, label: dict) -> list:
        mine = [ident for ident in refs if number in owner[ident]]
        faults = []
        if not mine:
            faults.append(f"item {number} ({want.called}) is missing from the cart")
        elif len([ident for ident in mine if owner[ident] == [number]]) > 1:
            faults.append(f"item {number} ({want.called}) has {len(mine)} lines ({', '.join(mine)}); keep just one")
        for ident in mine:
            have = self.basket.units(ident)
            if owner[ident] == [number] and have != want.units:
                faults.append(f"{ident}{UNIT_FAULT_MARK}{have} unit(s) in the cart, but item {number} ({want.called}) needs {want.units}")
            fuller = self.fuller_shelf(want, label[ident])
            if fuller:
                faults.append(f"{ident} sits under {label[ident]}, yet the shop also has {fuller}, a label giving the "
                              f"whole name of item {number} ({want.called}): take this line from there")
        return faults

    def fuller_shelf(self, want: Want, label: str) -> str:
        """A label among the rows that holds every word of an item's several-word name, when the given label does not
        (a label that is the item's own shelf is left alone)."""
        name = roots(want.called)
        key = self.shelf_key()
        if len(name) < 2 or name <= roots(label) or not key or (want.aisle and roots(label) == roots(want.aisle)):
            return ""
        labels = sorted({str(row.get(key) or "") for row in self.stock.rows.values()})
        return next(filter(lambda seen: seen and name <= roots(seen), labels), "")

    def _budget_faults(self, refs: list) -> list:
        cap = self.task.money_cap
        prices = {ident: self.stock.price(ident) for ident in refs}
        if not isinstance(cap, (int, float)) or None in prices.values():
            return []
        if self.task.reading.get("budget_scope") == "each":
            return [f"{ident} costs {prices[ident]:g}, above the per-item limit of {cap:g}"
                    for ident in refs if cap < prices[ident] <= 10 * cap]
        total = sum(prices[ident] * self.basket.units(ident) for ident in refs)
        return [f"the total of {total:g} goes over the spending limit of {cap:g}"] if cap + 1e-09 < total <= 10 * cap else []

    def unit_fixes(self) -> list:
        """(line, units held, units wanted) for lines that are the only line of their item but hold the wrong count."""
        key = self.shelf_key()
        if not key:
            return []
        out = []
        for ident in self.basket.refs(self.stock.rows):
            have = self.basket.units(ident) or 0
            owners = [want for want in self.task.wants if self.owns(want, self.shelf_of(ident, key))]
            if len(owners) != 1 or have <= 0:
                continue
            want = owners[0]
            mine = [other for other in self.basket.refs(self.stock.rows)
                    if (self.basket.lines.get(other, 1) or 0) > 0 and self.owns(want, self.shelf_of(other, key))]
            if len(mine) == 1 and have != want.units and not self.stock.sold_out(ident):
                out.append((ident, have, want.units))
        return out

    def set_units(self, wrong: list) -> list:
        tool = self.tools.jobs.get("cart") or ""
        target, count, mode = self.tools.cart_params(tool)
        if not (tool and target and count):
            return []
        calls: list = []
        for ident, have, want in wrong:
            if self.tools.sets_units(tool):
                calls.append({target: ident, count: want})
            elif mode:
                options = [str(option) for option in self.tools.fields(tool)[mode]["enum"]]
                off = next((option for option in options if REMOVAL_MODE.search(option)), options[1])
                on = next((option for option in options if option != off), options[0])
                calls.extend(({target: ident, mode: off, count: have}, {target: ident, mode: on, count: want}))
            else:
                return []
        if len(calls) + 3 > self.steps_left:
            return []
        done = []
        for given in calls:
            args = checked_args(given, self.tools.params.get(tool) or {})
            reply = self.call_store(tool, args) if args is not None else None
            if reply is None or self.link.finished:
                break
            self.book_cart_call(tool, args)
            self.basket.last_reply = as_json_text(reply.get("observation"))
            self.observe(tool, reply.get("observation"))
            done.append(as_json_text(args) + " => " + self.show(reply.get("observation"))[:300])
        self.link.log("units_set", calls=done)
        return done

    def wider_search_hint(self) -> str:
        """Once, for a cheapest pick with steps to spare: hold the checkout for a few more searches."""
        if self.track.wider_said or self.task.pick != "cheapest" or self.steps_left < 14 + 2 * len(self.task.wants):
            return ""
        self.track.wider_said = True
        priced = [f"{ident} at {self.stock.price(ident):g}" for ident in self.basket.refs(self.stock.rows)
                  if self.stock.price(ident) is not None]
        listing = f" ({'; '.join(priced)})" if priced else ""
        return (f"Not made yet (no step spent). Choice rule: cheapest; {self.steps_left} steps left. A search can "
                f"look at just part of a listing, so one on a condition's value turns up only some of the fitting "
                f"listings; rows also come unsorted by price. So before checkout: per item, run a few extra "
                f"searches with other names for the product: synonyms, a plain one-word name, title words of "
                f"fitting listings seen so far. Only buyable rows cheaper than the cart's line{listing} matter: open the "
                f"lowest-priced ones to see their attributes; if one fits, put it in place of the line; then check "
                f"out.")

    def cheaper_peeks(self) -> list:
        """Cheapest rule: per cart line, open the two lowest-priced rows of the same kind that cost less, are
        buyable and are still unopened (at most CHEAP_PEEKS in all; once per cart and epoch)."""
        if self.task.pick != "cheapest":
            return []
        mark, key = self.cart_epoch, self.shelf_key()
        if mark in self.track.cheaper_done or not key:
            return []
        self.track.cheaper_done.add(mark)
        cart = self.basket.refs(self.stock.rows)
        self.list_shelves_below(cart)
        picks: list = []
        for line in cart:
            price = self.stock.price(line)
            kind = self.shelf_of(line, key).strip().lower()
            if price is None or not kind:
                continue
            pool = sorted((ident for ident in self.stock.order
                           if ident not in cart and ident not in self.stock.pages and self.stock.buyable(ident)
                           and self.shelf_of(ident, key).strip().lower() == kind and self._price_or_inf(ident) < price),
                          key=lambda ident: self.stock.price(ident) or 0.0)
            picks.extend(ident for ident in pool[:2] if ident not in picks)
        return self.open_for_attributes(picks[:min(6, CHEAP_PEEKS * max(1, len(cart)))], self.closing_steps + 4)

    def _shelf_ceiling_tool(self) -> tuple | None:
        """(tool, fixed arguments, shelf field, ceiling field) of the listing tool taking a shelf and a price ceiling,
        None when there is no such tool or its mode is unclear."""
        for tool, schema in self.tools.params.items():
            fields = fields_of(schema)
            shelves = [key for key in text_fields(fields) if SHELF_PARAM.search(snake_key(key))]
            ceilings = [key for key, spec in fields.items() if isinstance(spec, dict)
                        and {"integer", "number"} & set(schema_types(spec))
                        and CEILING_PARAM.search(snake_key(key)) and not NOT_A_CEILING.search(key)]
            if len(shelves) == 1 and len(ceilings) == 1:
                fixed = self._listing_modes(fields, shelves[0])
                return None if fixed is None else (tool, fixed, shelves[0], ceilings[0])
        return None

    def _listing_modes(self, fields: dict, shelf: str) -> dict | None:
        """Each mode of the tool set as the latest listing of a shelf set it (a search with no words in it), else to
        the one option named after listing; None when a mode stays open."""
        def wordless(args: dict) -> bool:
            for key, value in args.items():
                free_text = key != shelf and "enum" not in (fields[key] if isinstance(fields.get(key), dict) else ())
                if free_text and isinstance(value, str) and value.strip() and QUERY_FIELD.search(snake_key(key)):
                    return False
            return True

        fixed: dict = {}
        for key, spec in fields.items():
            options = spec.get("enum") if isinstance(spec, dict) else None
            if isinstance(options, list) and len(options) >= 2:
                used = [args[key] for args in self.track.recent_searches if shelf in args and key in args and wordless(args)]
                named = [option for option in options if isinstance(option, str) and SHELF_MODE.search(option)]
                choice = used[-1] if used and used[-1] is not None else (named[0] if len(named) == 1 else None)
                if choice is None:
                    return None
                fixed[key] = choice
        return fixed

    def list_shelves_below(self, cart: list) -> None:
        """Cheapest rule: where the shop can list a shelf under a price, list each cart line's shelf just below that
        line's price, and read the second page too when the rows spill onto one."""
        found = self._shelf_ceiling_tool()
        key = self.shelf_key()
        if not found or not key:
            return
        tool, fixed, shelf, ceiling = found
        for line in cart:
            price = self.stock.price(line)
            label = self.shelf_of(line, key)
            if price is None or price < 2 or not label or (line, price) in self.track.ceilings_listed:
                continue
            if self.steps_left <= self.closing_steps + 5 or self.link.finished:
                return
            self.track.ceilings_listed.add((line, price))
            below = int(price) - 1 if float(price).is_integer() else int(price)
            args = checked_args({**fixed, shelf: label, ceiling: below}, self.tools.params.get(tool))
            reply = self.call_store(tool, args) if args is not None else None
            if reply is None:
                return
            seen = reply.get("observation")
            self.observe(tool, seen)
            self.link.log("shelf_listed", line=line, price=price)
            self._read_second_page(tool, args, seen)

    def _read_second_page(self, tool: str, args: dict, seen: Any) -> None:
        state = self.page_state(tool, seen)
        if not state or state[3] is None or not state[1] < state[0] <= 2 * state[1]:
            return
        if self.steps_left <= self.closing_steps + 5 or self._in_price_order(seen):
            return
        call = checked_args({**args, state[2]: state[3]}, self.tools.params.get(tool))
        reply = self.call_store(tool, call) if call is not None else None
        if reply is not None:
            self.observe(tool, reply.get("observation"))

    # -- a cart short of an item, at a forced finish ----------------------------------------------------------------
    def uncovered_items(self) -> list:
        """The items a short cart leaves out: fewer lines than items, none under the item's labels, and at least one
        seen listing under them."""
        key = self.shelf_key()
        refs = [ident for ident in self.basket.refs(self.stock.rows) if (self.basket.units(ident) or 0) > 0]
        if not key or len(refs) >= len(self.task.wants):
            return []
        seen = [str(row.get(key) or "") for row in self.stock.rows.values()]
        held = [self.shelf_of(ident, key) for ident in refs]

        def covers(want: Want, labels: list) -> bool:
            return any(self.owns(want, label) for label in labels)

        return [number for number, want in enumerate(self.task.wants, 1) if covers(want, seen) and not covers(want, held)]

    def states_every_need(self, ident: str, number: int) -> bool:
        """True when the listing's attribute table spells out every requirement of the item in full. Bounds and
        exclusions stay unread, which makes them fail this test."""
        table = attribute_block(self.stock.pages.get(ident) or {}) or {}

        def spells(need: str) -> bool:
            prop, _, value = need.partition(":")
            wanted = plain_words(value)
            if not wanted or wanted & UNREAD_VALUE_WORDS:
                return False
            forms = name_forms(prop)
            return any(same_property(forms, name_forms(key)) and wanted <= plain_words(str(cell)) for key, cell in table.items())

        return all(spells(need) for owner, need in self.task.live_needs() if owner == number)

    def _affordable_extra(self, pick: str, want: Want) -> bool:
        """Whether a listing taken for a missing item stays inside the spending limit."""
        cap = self.task.money_cap
        if not isinstance(cap, (int, float)):
            return True
        price = self.stock.price(pick) or 0.0
        if self.task.reading.get("budget_scope") == "each":
            return price <= cap + 1e-09
        held = sum((self.stock.price(ident) or 0.0) * (self.basket.units(ident) or 0)
                   for ident in self.basket.refs(self.stock.rows))
        return price * want.units + held <= cap + 1e-09

    def fill_missing_items(self) -> bool:
        """Before a forced order: an uncovered item gets the lowest-priced opened listing spelling out its needs.
        False when one stays uncovered: a short order is worse than ending with no purchase."""
        tool = self.tools.jobs.get("cart") or ""
        paired = {number for pair in self.task.pairs for number in pair}
        for number in self.uncovered_items():
            want = self.task.wants[number - 1]
            stated = [ident for ident in self.item_refs(number) if ident in self.stock.pages
                      and ident not in self.basket.lines and self.stock.buyable(ident)
                      and self.stock.price(ident) is not None and self.states_every_need(ident, number)]
            pick = None if number in paired else min(stated, key=lambda ident: self.stock.price(ident) or 0.0, default=None)
            if pick is None or not self._affordable_extra(pick, want):
                return False
            lines = {ident: self.basket.lines[ident] for ident in self.basket.refs(self.stock.rows) if ident in self.basket.lines}
            lines[pick] = want.units
            calls = self.cart_plan_calls(lines, [])
            if not calls or len(calls) + 1 > self.steps_left:
                return False
            for given in calls:
                args = checked_args(given, self.tools.params.get(tool))
                if args is None or self.call_store(tool, args) is None:
                    return False
                self.book_cart_call(tool, args)
            self.link.log("line_filled", item=number, line=pick)
        return True

    def decline_possible(self) -> bool:
        tool = self.tools.jobs.get("refuse")
        if not tool:
            return False
        given = self.quiet_args(tool, dict(self.task.reading.get("refuse_args") or {}))
        return checked_args(given, self.tools.params.get(tool)) is not None

    def quiet_args(self, tool: str, args: dict) -> dict:
        """A decline's free-text fields get one short sentence that names no listing (unless the request asked for
        alternatives and the text names none)."""
        schema = self.tools.params.get(tool) or {}
        out = dict(args)
        for key, spec in fields_of(schema).items():
            if not isinstance(spec, dict) or "string" not in schema_types(spec) or "enum" in spec:
                continue
            value = out.get(key)
            if value is None and key not in (schema.get("required") or []):
                continue
            text = str(value or "")
            names = self.task.reading.get("quiet_decline") is not False or any(ident in text for ident in self.stock.rows)
            if names or not text:
                out[key] = ("Sorry, nothing in the store meets all of your requirements, so I am not buying "
                            "anything.")[:spec.get("maxLength", 200)]
        return out



# ---------------------------------------------------------------------------------------------------- the loop
class Ticket:
    """One tool call from the model on its way through the checks: its tool, arguments and what the checks found."""

    side: Any = None
    param: Any = None
    ending = False
    mark = ""
    trimmed = ""

    def __init__(self, tool: Any, raw: Any):
        self.tool, self.raw = tool, raw
        self.args: dict = {}
        self.readers: set = set()


class _LoopPart:
    """The store loop: what the model is offered each turn, how its calls are screened, run and answered, and how
    the episode is closed when the loop stops."""

    def tools_offered(self) -> list:
        """The tools for this turn: never the messaging tool; near the end only the finishing ones (and the cart)."""
        jobs, left, closing = self.tools.jobs, self.steps_left, self.closing_steps
        endings = self.tools.endings
        offered = [tool for tool in self.tools.entries if tool["function"]["name"] != jobs.get("ask")]
        if left <= closing:
            keep = set(endings)
        elif left <= closing + 3:
            keep = endings | {jobs.get("cart"), jobs.get("cart_view")}
        elif self.track.checkouts == 0 and left <= len(self.task.wants) + 5:
            keep = endings | {jobs.get("cart"), jobs.get("cart_view"), jobs.get("offer")}
        else:
            return offered
        if not endings:
            return offered
        return [tool for tool in offered if tool["function"]["name"] in keep] or offered

    def fix_listing_refs(self, tool: str, args: dict) -> tuple:
        """A listing named by its title where a reference belongs is swapped for the reference when one listing's
        text starts that way. Returns (arguments, problem to report or '')."""
        jobs = self.tools.jobs
        if not self.stock.id_key or tool not in {jobs.get(job) for job in ("cart", "open", "offer", "compare")} - {None}:
            return args, ""
        free = [key for key, spec in self.tools.fields(tool).items() if isinstance(spec, dict)
                and "enum" not in spec and ("string" in schema_types(spec) or "array" in schema_types(spec))]
        if len(free) != 1:
            return args, ""
        if tool != jobs.get("cart") and (tool, free[0]) not in self.tools.id_params:
            return args, ""
        key = free[0]
        if key not in args:
            return dict(args), ""
        given = args[key]
        fixed = list(map(self._ref_for, given if isinstance(given, list) else [given]))
        problem = next((note for _, note in fixed if note), "")
        if problem:
            return args, problem
        values = [value for value, _ in fixed]
        return {**args, key: values if isinstance(given, list) else values[0]}, ""

    def _ref_for(self, value: Any) -> tuple:
        if isinstance(value, str) and value.strip() in self.stock.rows:
            return value.strip(), ""
        if not isinstance(value, str) or looks_like_id(value, True) or len(value.strip()) < 8:
            return value, ""
        start = value.strip().lower()
        hits = [ident for ident, row in self.stock.rows.items()
                if any(key != self.stock.id_key and isinstance(cell, str) and cell.strip().lower().startswith(start)
                       for key, cell in row.items())]
        if len(hits) == 1:
            return hits[0], ""
        if not hits:
            return value, f"{value!r} is no listing id; pass the id shown in the result rows (their {self.stock.id_key} field)"
        return value, f"the text {value!r} appears on several listings ({', '.join(hits[:6])}); pass a single listing id"

    def merge_single_opens(self, calls: list) -> list:
        """When the comparison view shows attributes, fold a turn's separate one-listing opens into a single
        comparison call."""
        opener, joiner = self.tools.jobs.get("open"), self.tools.jobs.get("compare")
        param, side = self.tools.single_param(opener), self.tools.side_param
        if not all((opener, joiner, param, side)) or opener == joiner or self.tools.side_bare:
            return calls
        picks = []
        for index, call in enumerate(calls):
            fn = call.get("function") or {}
            args = parse_object(fn["arguments"]) if isinstance(fn.get("arguments"), str) else None
            if fn.get("name") != opener or not args or set(args) != {param}:
                continue
            ident = str(args.get(param) or "").strip()
            if ident in self.stock.rows and ident not in self.stock.gone:
                picks.append((index, ident))
        picks = picks[:self.tools.side_room]
        if len(picks) < 2:
            return calls
        merged_at = dict(picks)
        joined = {"name": joiner, "arguments": as_json_text({side: list(merged_at.values())})}
        out = []
        for index, call in enumerate(calls):
            if index == picks[0][0]:
                out.append({"id": call["id"], "type": "function", "function": joined})
            else:
                out.append({"id": call["id"], "joined": True} if index in merged_at else call)
        self.link.log("opens_joined", ids=list(merged_at.values()))
        return out

    def answer_calls(self, calls: list) -> list:
        calls = self.merge_single_opens(calls)
        return [(call["id"], self.answer_call(call, len(calls))) for call in calls]

    def answer_call(self, call: dict, count: int) -> str:
        """Screen one call, run it if it passes (or hold a finishing call back), and return the text the model reads
        as its result."""
        if call.get("joined"):
            return "Already covered: Wren opened these listings jointly, in one step; see the comparison result above."
        fn = call.get("function") or {}
        ticket = Ticket(fn.get("name"), fn.get("arguments"))
        for stage in (self._refuse_early, self._parse_ticket, self._hold_before_run, self._repeat_note,
                      self._finish_holds):
            text = stage(ticket, count)
            if text:
                return text
        reply = self.call_store(ticket.tool, ticket.args)
        if reply is None:
            return "The call did not go through (network or server trouble). You can try again."
        return self._after_run(ticket, reply)

    def _refuse_early(self, ticket: Ticket, count: int) -> str:
        if self.link.finished:
            return "Not made: the run has already ended."
        if ticket.tool == self.tools.jobs.get("ask"):
            return "Not made (no step spent): every message to the customer goes through Wren's interviewer."
        allowed = sorted({offer["function"]["name"] for offer in self.tools_offered()})
        if ticket.tool not in self.tools.params:
            return f"Not made (no step spent): no such tool. Available: {', '.join(allowed)}."
        if ticket.tool not in allowed:
            return "Not made (no step spent): too few steps remain for that; use one of: " + ", ".join(allowed) + "."
        return ""

    def _parse_ticket(self, ticket: Ticket, count: int) -> str:
        raw = ticket.raw
        args = raw if isinstance(raw, dict) else (parse_object(raw) if (raw or "").strip() else {})
        if args is None:
            return "Not made (no step spent): the arguments were not a single JSON object."
        faults: list = []
        args = conform(args, self.tools.params[ticket.tool], "args", faults)
        if faults:
            return "Not made (no step spent): " + "; ".join(faults[:6])
        args, wrong = self.fix_listing_refs(ticket.tool, args)
        if wrong:
            return f"Not made (no step spent): {wrong}."
        if ticket.tool == self.tools.jobs.get("cart"):
            args = self._count_in_removal(ticket.tool, args)
        if ticket.tool == self.tools.jobs.get("compare"):
            ticket.side = self.tools.side_param
        listed_refs = args.get(ticket.side) if ticket.side else None
        room = self.tools.side_room
        if isinstance(listed_refs, list) and len(listed_refs) > room and self.tools.side_bad:
            ticket.trimmed = (f"\n[Wren] Only the first {room} listings went: side by side this shop shows {room} or "
                              f"fewer. Still to show: {', '.join(map(str, listed_refs[room:]))}.")
            args = {**args, ticket.side: listed_refs[:room]}
        ticket.args = args
        ticket.param = self.tools.single_param(ticket.tool)
        return ""

    def _count_in_removal(self, tool: str, args: dict) -> dict:
        """A removal of a held line with no count gets the units held as its count: the store would take off one."""
        fields = self.tools.fields(tool)
        count = first_field(fields, "integer")
        if not count or count in args or not Basket.removing(args, fields):
            return args
        ident = next((value for value in args.values() if isinstance(value, str) and value in self.basket.lines), None)
        if not ident:
            return args
        held = self.basket.lines[ident]
        return {**args, count: min(held, fields[count].get("maximum", held))}

    def _hold_before_run(self, ticket: Ticket, count: int) -> str:
        jobs, args, param = self.tools.jobs, ticket.args, ticket.param
        if param and ticket.tool in (jobs.get("open"), jobs.get("offer")) and str(args.get(param)) in self.stock.gone:
            return (f"Not made (no step spent): {args[param]} has been sold out since step "
                    f"{self.stock.gone[str(args[param])]} "
                    f"and stays that way. Its cart line must go to some other listing that fits.")
        if ticket.tool == jobs.get("cart") and any((whole_number(v) or 0) < 0 for v in args.values() if not isinstance(v, str)):
            view = as_json_text(self.basket.view) if self.basket.viewed_at >= self.basket.changed_at else ""
            absent = [value for value in args.values() if isinstance(value, str) and value in self.stock.rows
                      and value not in self.basket.lines and value not in view and value not in self.basket.last_reply]
            if absent:
                return (f"Not made (no step spent): nothing can come off, because {absent[0]} is absent from "
                        f"the cart. Cart now: {self.basket.describe()}")
        ending = self.ends_in_order(ticket.tool, args) or self.ends_in_decline(ticket.tool, args)
        if ending:
            return "Not made (no step spent): first read the results of your other calls, then make the finishing call alone." if count > 1 else ""
        if self.steps_left <= self.closing_steps and (jobs.get("buy") or jobs.get("refuse")):
            return ("Not made (no step spent): the steps left are held back for the finish: checkout (after the "
                    "customer approves the cart, if that is due) or a no-purchase ending.")
        return ""

    def _repeat_note(self, ticket: Ticket, count: int) -> str:
        tool, args = ticket.tool, ticket.args
        ticket.ending = self.ends_in_order(tool, args) or self.ends_in_decline(tool, args)
        ticket.mark = tool + as_json_text(args)
        ticket.readers = {self.tools.jobs.get(job) for job in ("search", "open", "offer", "compare")} - {None}
        if not ticket.ending and tool != self.tools.jobs.get("cart") and ticket.mark in self.track.recent_calls:
            return "Not made (no step spent): you made this exact call a moment ago and nothing has changed; its result is above."
        cached = self.track.read_memo.get(ticket.mark)
        if tool in ticket.readers and cached and cached[0] == self.track.epoch:
            return (f"Not made (no step spent): repeat of step {cached[1]}'s call. No change since then; the "
                    f"result it gave:\n" + cached[2][:3000])
        if ticket.param and tool in ticket.readers and tool != self.tools.jobs.get("compare"):
            if self.stock.nothing_new(tool, str(args.get(ticket.param)), self.track.epoch):
                return (f"Not made (no step spent): this view of {args[ticket.param]} has nothing new to show "
                        f"after the last change. Its record so far: " + self.stock.text_of(str(args[ticket.param])))
        refs = args.get(ticket.side) if ticket.side else None
        if isinstance(refs, list) and refs:
            fresh = [ident for ident in refs if not self.stock.nothing_new(tool, str(ident), self.track.epoch)]
            if not fresh:
                records = "\n".join(f"{ident}: {self.stock.text_of(str(ident))}" for ident in refs)
                return ("Not made (no step spent): this view has nothing new to show for these listings after "
                        "the last change. Their records so far:\n" + records[:4000])
            if len(fresh) < len(refs):
                left_out = ", ".join(str(ident) for ident in refs if ident not in fresh)
                ticket.trimmed += f"\n[Wren] Left out {left_out}: fully shown before, and no change since then."
                ticket.args = {**args, ticket.side: fresh}
                ticket.mark = tool + as_json_text(ticket.args)
        return ""

    def _finish_holds(self, ticket: Ticket, count: int) -> str:
        tool = ticket.tool
        if ticket.ending and self.steps_left > 2:
            hold = self._checkout_hold() if self.ends_in_order(tool, ticket.args) else self._decline_hold()
            if hold:
                return hold
        if self.ends_in_decline(tool, ticket.args):
            hold = self._late_decline_hold() or self._push_search()
            if hold:
                return hold
            done = self.buy_confirmed()
            if done:
                return done + (" [episode finished]" if self.link.finished else "")
            ticket.args = self.quiet_args(tool, ticket.args)
        if self.ends_in_order(tool, ticket.args):
            news = self.extra_question()
            if news:
                return f"Not made yet. {news}\nCurrent cart: {self.basket.describe()}"
        if ticket.ending and self.steps_left >= 2:
            go, said = self.duty_gate("order" if self.ends_in_order(tool, ticket.args) else "end")
            if not go:
                return f"Not made yet. {said}\nCurrent cart: {self.basket.describe()}"
            if said:
                ticket.trimmed = "\n[Wren] " + said + ticket.trimmed
        return ""

    def _push_search(self) -> str:
        if self.track.candidates or self.track.decline_pushed or self.steps_left <= self.task.step_limit // 3:
            return ""
        self.track.decline_pushed = True
        return (f"Not made yet (no step spent). With {self.steps_left} steps left, try searches you have not "
                f"run yet before you end. Go through each requested item's category page by page, passing back the "
                f"token or offset the shop gives. Query single value words, and other names for the property. Open "
                f"whichever listings could fit, and only then end without a purchase.")

    def _checkout_hold(self) -> str:
        """Before a checkout runs: the episode's own checks, in order. Returns the hold text, or '' to go on."""
        cart_now = "\nCurrent cart: "
        wider = self.wider_search_hint()
        if wider:
            return wider + cart_now + self.basket.describe()
        reread = self.reread_cart_offers()
        if self.track.news_sold:
            self.find_sold_line()
        change = self._look_at_cart()
        if change:
            return ("Not made yet (no step spent). The cart view brought news: " + change + cart_now
                    + self.basket.describe() + "\nTake the change in, adjust the cart if it needs it, then check out.")
        swap = self.sellout_hint()
        gone = []
        if self.track.gone_holds < 2:
            gone = [ident for ident in self.basket.refs(self.stock.rows) if ident in self.stock.gone]
        if swap or gone:
            self.track.gone_holds += 0 if swap else 1
            reason = swap or f"{', '.join(gone)} sold out while in the cart: swap that line before anything else."
            return "Not made yet (no step spent). " + reason + cart_now + self.basket.describe()
        chain, hold = self.follow_pointers()
        if hold:
            return "Not made yet (no step spent). " + chain + cart_now + self.basket.describe()
        closing = self.closing_steps
        misfit = self.pairing_gaps() if self.track.fit_hold_count < 2 and self.steps_left > closing + 3 else []
        if misfit:
            self.track.fit_hold_count += 1
            self.link.log("misfit_hold", faults=misfit[:4])
            return ("Not made yet (no step spent). The cart's device and the part chosen for it do not go "
                    "together: " + "; ".join(misfit[:3]) + ". Swap the part (or else the device) and check out."
                    + cart_now
                    + self.basket.describe())
        faults = self.basket_gaps() if self.track.line_hold_count < 2 and self.steps_left > self.closing_steps + 2 else []
        if faults and all(UNIT_FAULT_MARK in fault for fault in faults):
            fixed = self.set_units(self.unit_fixes())
            faults = self.basket_gaps() if fixed else faults
            if fixed and not faults:
                self.track.line_hold_count += 1
                return ("Not made yet (no step spent). Wren corrected the cart's quantities itself: " + "; ".join(fixed)
                        + ". Should the shop have rejected one of these, repair that line, then check out." + cart_now
                        + self.basket.describe())
        if faults:
            self.track.line_hold_count += 1
            self.link.log("lines_hold", faults=faults[:6])
            return ("Not made yet (no step spent). The cart does not yet match the request: " + "; ".join(faults[:6])
                    + ". Repair them with the cart tool, adding or removing units the way that tool does it; then "
                    "check out."
                    + cart_now
                    + self.basket.describe())
        refs = self.basket.refs(self.stock.rows)
        read = self.open_for_attributes(refs, self.closing_steps + (1 if len(refs) >= 2 else 3))
        self.follow_part_pointers(refs, self.closing_steps + 1)
        read += self.cheaper_peeks()
        verdict = self.review_cart()
        if verdict:
            seen = ""
            if read:
                seen = "Wren opened these listings to see their attributes for the review, one step apiece:\n" + "\n".join(read) + "\n"
            return "Not made yet. " + chain + seen + verdict + cart_now + self.basket.describe()
        if reread and self.link.spent > SPEND_CALM:
            return ("Not made yet (no step spent). Wren has just re-read the cart's listings:\n" + reread + "\nIf "
                    "each line is still in stock and priced within the spending limit, check out again; otherwise, "
                    "first replace the line that is not.")
        return ""

    def _look_at_cart(self) -> str:
        """View the cart once after it changed (when there is a view tool and steps to spare); returns what a notice
        on it changed, or ''."""
        view_tool = self.tools.jobs.get("cart_view")
        stale = self.basket.touched and self.basket.viewed_at < self.basket.changed_at
        if not (stale and view_tool and self.steps_left > 3):
            return ""
        args = checked_args({}, self.tools.params.get(view_tool))
        reply = self.call_store(view_tool, args) if args is not None else None
        if reply is None:
            return ""
        self.basket.view = reply.get("observation")
        self.basket.viewed_at = self.link.step
        change = self.observe(view_tool, self.basket.view)
        if change:
            self.track.pending_news = []
        return change

    def _decline_hold(self) -> str:
        if self.steps_left <= 5:
            return ""
        review = self.decline_review()
        if review:
            return "Not made yet (no step spent). " + review
        if not self.track.candidates or self.steps_left <= 4 or self.track.decline_blocks > 1:
            return ""
        if not self.track.decline_blocks:
            self.track.decline_blocks = 1
            self.link.log("decline_held", leads=list(self.track.candidates))
            return ("Not made yet (no step spent). The review found listings that seem to fit: "
                    + ", ".join(self.track.candidates) + ". Open the unread ones, cart the strongest fit under the "
                    "choice rule, then check out. A no-purchase ending is allowed only if each of them plainly fails a "
                    "condition: then name it in your text and repeat the ending call.")
        self.track.decline_blocks = 2
        first = self.track.candidates[0]
        self.review_cart([first], SPEND_CEILING)
        if not self.track.review_verdicts.get(self.review_key([first])) or first not in self.track.candidates:
            return ""
        self.track.confirmed_pick = first
        self.link.log("decline_held", confirmed=first)
        return (f"Not made yet (no step spent). A separate check of the records for {first} says it satisfies "
                f"all conditions. Cart it, at the quantity requested, and check out. Ending with no purchase is right "
                f"only when its record plainly violates a condition.")

    def _late_decline_hold(self) -> str:
        """A decline in the last steps while the episode once built a full cart: offer that cart back instead."""
        if self.steps_left > 5 or self.track.final_cart_offered:
            return ""
        lines = [ident for ident in self.basket.best_lines if not self.stock.sold_out(ident)]
        in_cart = self.basket.refs(self.stock.rows)
        fresh = self.basket.fresh
        missing = [ident for ident in lines if ident not in in_cart or (ident in self.basket.removed and not fresh)]
        if len(lines) < len(self.task.wants) or len(missing) + 1 > self.steps_left:
            return ""
        self.track.final_cart_offered = True
        self.link.log("late_decline", lines=lines, missing=missing)
        redo = (f"Return {', '.join(missing)} to the cart (each at its requested quantity) and check out." if missing
                else "Buy what the cart holds now.")
        return ("Not made yet (no step spent). A no-purchase ending earns zero if the shop does have a fitting "
                "listing; being short of steps is no evidence that it lacks one. " + redo + " Ending without buying is "
                "only right once your searches have shown a requested item with no listing that meets its conditions; "
                "after that, repeat the ending call.")

    def _after_run(self, ticket: Ticket, reply: dict) -> str:
        """Book a call that ran and write what the model reads: the result, pages read on, notes and hints."""
        jobs, tool, args = self.tools.jobs, ticket.tool, ticket.args
        observation = reply.get("observation")
        refs = args.get(ticket.side) if ticket.side else None
        if isinstance(refs, list) and len(refs) > 1:
            dumped = as_json_text(observation)
            if any(str(ident) in dumped for ident in refs):
                self.tools.side_ok = max(self.tools.side_ok, len(refs))
            else:
                self.tools.side_bad = min(self.tools.side_bad or 999, len(refs))
                ticket.trimmed += (f"\n[Wren] Comparing {len(refs)} listings at once gave no results at all: put at "
                                   f"most {self.tools.side_room} in a call.")
        self.track.recent_calls = self.track.recent_calls[-2:] + [ticket.mark]
        for key, value in args.items():
            cells = value if isinstance(value, list) else [value]
            if cells and all(isinstance(cell, str) and cell in self.stock.rows for cell in cells):
                self.tools.id_params.add((tool, key))
        self.stock.take_in(observation)
        if tool == jobs.get("compare"):
            rows = self.stock.records(observation)
            if rows:
                self.tools.shows_attrs[tool] = self.tools.shows_attrs.get(tool, False) or any(attribute_block(r) for _, r in rows)
        if tool in ticket.readers and tool != jobs.get("search"):
            self.stock.remember_view(tool, observation, self.track.epoch)
        plain = {named for job, named in jobs.items() if job != "search"}
        searchy = tool not in plain and not ticket.ending
        paged = self.read_more_pages(tool, args, observation) if searchy else ""
        self._learn_single_view(tool, observation)
        if tool == jobs.get("cart"):
            self._after_cart_call(tool, args, observation)
        if tool == jobs.get("search") and self.stock.records(observation):
            self.track.recent_searches = self.track.recent_searches[-7:] + [dict(args)]
        if tool == jobs.get("cart_view"):
            self.basket.view, self.basket.viewed_at = observation, self.link.step
        if self.ends_in_order(tool, args):
            self.track.checkouts += 1
            if not reply.get("finished"):
                self.track.refused_step = self.link.step
        base = self.show(observation) + paged + ticket.trimmed
        if searchy:
            base += self.cheapest_rows_note([observation] + self.track.paged_results)
            base += self.ignored_price_cap(tool, args, observation)
        notes = [self.evidence_note(observation)]
        if tool == jobs.get("cart") and any(whole_number(v) == 0 for v in args.values() if not isinstance(v, str)):
            if any(isinstance(v, str) and v in self.stock.rows and holds_id(observation, v) for v in args.values()):
                notes.append("[Wren] That line is still in the cart after a count of 0, because this cart tool adds "
                             "the given count to the line; a negative count removes units.")
        notes += self._news_after_run(tool, args, reply, observation)
        if tool in ticket.readers:
            self.track.read_memo[ticket.mark] = (self.track.epoch, self.link.step, base)
        text = base + "".join("\n" + note for note in notes if note)
        return text + " [episode finished]" if reply.get("finished") else text

    def _news_after_run(self, tool: str, args: dict, reply: dict, observation: Any) -> list:
        """Notices in a result, a refusal that names a sell-out, sold-out cart lines, and notes queued meanwhile."""
        notes = []
        self.track.news_sold = False
        lines = self.notice_lines(observation)
        meaning = self.read_notice(lines) if lines else ""
        if meaning:
            notes.append(f"[note] {meaning}. Your picks must reflect this change by checkout time.")
        buying = self.ends_in_order(tool, args)
        refused = buying and not reply.get("finished")
        again = refused and any(line in self.spoken(observation) for line in self.track.sellout_lines)
        loose = self.track.news_sold and (refused or (not self.track.news_named and not buying))
        if (loose or again) and self.basket.refs(self.stock.rows) and not self.link.finished:
            seen = self.find_sold_line()
            if seen:
                notes.append("[cart lines checked anew once the shop refused]\n" + seen)
        self.track.news_sold = False
        notes.append(self.sellout_hint())
        if self.track.pending_news:
            notes.append("[notes from Wren's own calls] " + " | ".join(self.track.pending_news))
            self.track.pending_news = []
        return notes

    def _learn_single_view(self, tool: str, observation: Any) -> None:
        """A single-listing view that ran: does it show attributes? An 'open' tool that does not hands its job to
        another single-listing tool; a view without attributes becomes the 'offer' view."""
        jobs = self.tools.jobs
        busy = tuple(jobs.get(job) for job in ("cart", "buy", "refuse", "ask"))
        listing = self.stock.ref(observation) if isinstance(observation, dict) else None
        if not self.tools.single_param(tool) or tool in busy or not listing:
            return
        shows = self.tools.shows_attrs.get(tool, False) or bool(attribute_block(observation))
        self.tools.shows_attrs[tool] = shows
        if not shows and tool == jobs.get("open"):
            skip = busy + (jobs.get("search"),)
            other = next((name for name in self.tools.params if name != tool and self.tools.single_param(name)
                          and self.tools.shows_attrs.get(name) is not False and name not in skip), None)
            if other:
                jobs["open"] = other
        if not shows and tool != jobs.get("open") and not jobs.get("offer"):
            jobs["offer"] = tool
        if tool == self.tools.offer_view:
            self.stock.offer_read_at[listing] = self.link.step

    def _after_cart_call(self, tool: str, args: dict, observation: Any) -> None:
        self.book_cart_call(tool, args)
        self.basket.last_reply = as_json_text(observation)
        self.track.recent_calls = []
        counts = [whole_number(value) for value in args.values() if counted(value)]
        for value in args.values() if counts else ():
            if isinstance(value, str) and value in self.stock.rows:
                if counts[0] > 0:
                    self.basket.removed.discard(value)
                elif not holds_id(observation, value):
                    self.basket.removed.add(value)
        lines = [ident for ident in self.basket.refs(self.stock.rows)
                 if not self.stock.sold_out(ident) and ident not in self.basket.removed]
        if len(lines) >= len(self.basket.best_lines):
            self.basket.best_lines = lines

    # -- closing --------------------------------------------------------------------------------------------------
    def force_finish(self, again: bool = False) -> str:
        """Finish without asking the model: buy a held cart, or else decline. With `again`, a checkout the shop
        refuses is passed back to the loop to repair."""
        if self.link.finished or self.link.store_broke or self.steps_left <= 0:
            return ""
        fresh = self.basket.fresh
        holding = bool(self.basket.refs(self.stock.rows)) if fresh else bool(self.basket.lines)
        whole = self._cart_made_whole() if holding and len(self.task.wants) >= 2 else True
        if self.link.finished or self.steps_left <= 0:
            return ""
        buy = self.tools.jobs.get("buy")
        if whole and buy and (holding or (self.basket.touched and not fresh and self.steps_left >= 2)):
            outcome = self._forced_order(buy, again)
            if outcome is not None:
                return outcome
        refuse = self.tools.jobs.get("refuse")
        return self._forced_decline(refuse) if refuse else ""

    def _cart_made_whole(self) -> bool:
        """Before a forced finish of a cart for several items: fill its gaps, or decide against ordering it (only
        when a decline is open), and read the lines of a cart that goes ahead."""
        if not self.fill_missing_items() and self.decline_possible():
            self.link.log("short_cart_declined", lines=self.basket.refs(self.stock.rows))
            return False
        if len(self.basket.refs(self.stock.rows)) >= 2:
            self.open_for_attributes(self.basket.refs(self.stock.rows), self.closing_steps)
            if self.steps_left >= 3 and self.link.room > 5:
                self.follow_part_pointers(self.basket.refs(self.stock.rows), self.closing_steps)
        return True

    def _forced_order(self, buy: str, again: bool) -> str | None:
        """None: go on to a decline; a string: the result to return."""
        args = checked_args(dict(self.task.reading.get("buy_args") or {}), self.tools.params.get(buy))
        if args is None:
            return None
        if self.steps_left >= 2:
            self.duty_gate("order")
        if self.link.finished or self.steps_left <= 0:
            return ""
        reply = self.call_store(buy, args)
        self.link.log("final_order", reply=shorten(reply, 200))
        if self.link.finished or self.steps_left <= 0:
            return ""
        if reply is None or not again:
            return None
        self.track.checkouts += 1
        self.track.refused_step = self.link.step
        self.observe(buy, reply.get("observation"))
        return "the shop refused the checkout: " + self.show(reply.get("observation"))

    def _forced_decline(self, refuse: str) -> str:
        args = checked_args(self.quiet_args(refuse, dict(self.task.reading.get("refuse_args") or {})),
                            self.tools.params.get(refuse))
        if args is None:
            return ""
        if self.steps_left >= 2:
            self.duty_gate("end")
        if self.link.finished or self.steps_left <= 0:
            return ""
        reply = self.call_store(refuse, args)
        self.link.log("final_decline", reply=shorten(reply, 200))
        if reply is None or self.link.finished:
            return ""
        self.observe(refuse, reply.get("observation"))
        return "the shop refused the no-purchase ending: " + self.show(reply.get("observation"))

    def trim_history(self) -> None:
        """Past CHAT_CHARS, the older half of the tool results is cut to 400 characters plus the listings they named."""
        history = self.track.history
        if sum(len(as_json_text(message.get("content"))) for message in history) <= CHAT_CHARS:
            return
        results = [message for message in history if message.get("role") == "tool"]
        for message in results[:max(1, len(results) // 2)]:
            content = message.get("content") or ""
            if len(content) > 600:
                named = [ident for ident in dict.fromkeys(QUOTED_ID.findall(content)) if ident in self.stock.rows]
                message["content"] = content[:400] + f"...(older result trimmed; listings it named: {', '.join(named[:20])})"

    def run(self) -> list:
        if self.link.finished:
            return self.link.trace
        self.read_task()
        self.link.log("reading", brief=self.task.reading)
        opening = self.task.policy.get("opening")
        intro = ["THE CUSTOMER'S REQUEST:\n" + self.task.query]
        if opening is not None:
            self.stock.take_in(opening)
            intro.append("\n\nWHAT THE SHOP SHOWS FIRST:\n" + self.show(opening))
        intro.append("\n\nBegin by searching for each item in its category and opening several listings in full, so the "
                     "interviewer can see which properties those listings carry.\n\n" + self.status_text())
        self.track.history = [{"role": "system", "content": PLAYBOOK}, {"role": "user", "content": "".join(intro)}]
        self.loop(TURN_LIMIT, SPEND_CEILING)
        if not self.link.finished:
            refused = self.force_finish(again=True)
            stuck = self.link.finished or self.link.store_broke or self.link.model_broke or self.steps_left <= 1
            if refused and not stuck:
                self.track.history.append({"role": "user", "content": "[Wren] Wren tried to end the run and "
                                           + refused + "\nApply what the shop says, then finish.\n" + self.status_text()})
                self.loop(self.track.turns + 10, SPEND_CEILING + 0.01)
            if not self.link.finished:
                self.force_finish()
        self.link.log("done", finished=self.link.finished, steps=self.link.step, cost=round(self.link.spent, 5),
                      calls=self.link.model_calls)
        return self.link.trace

    def _loop_open(self, turn_cap: int, spend_cap: float) -> bool:
        if self.link.finished or self.link.store_broke or self.steps_left <= 0:
            return False
        if self.track.turns >= turn_cap or self.link.spent >= spend_cap:
            return False
        if self.link.elapsed < TURN_LATEST:
            return True
        return not self.basket.refs(self.stock.rows) and self.link.room > BARE_CART_ROOM

    def loop(self, turn_cap: int, spend_cap: float) -> None:
        idle = 0
        while self._loop_open(turn_cap, spend_cap):
            if self.interview_due:
                summary = self.run_interview()
                if summary:
                    self.track.history.append({"role": "user", "content": summary + "\n" + self.status_text()})
                continue
            self.trim_history()
            self.track.turns += 1
            message = self.link.consult(self.track.history, [QUICK_MODEL, FALLBACK_MODEL], 3000, self.tools_offered())
            if message is None:
                self.track.empty_turns += 1
                if self.link.model_broke or self.link.model_calls >= MODEL_CALL_LIMIT or self.track.empty_turns > 2:
                    break
                continue
            calls = self._tool_calls(message)
            content = message.get("content")
            content = content if isinstance(content, str) else (as_json_text(content) if content else "")
            if calls:
                idle = 0
                self._take_turn(content, calls)
                continue
            idle += 1
            self.track.history += [{"role": "assistant", "content": content or "(no call)"},
                                   {"role": "user", "content": "Continue by calling one of the tools.\n" + self.status_text()}]
            if idle >= 3:
                break

    def _take_turn(self, content: str, calls: list) -> None:
        self.track.history.append({"role": "assistant", "content": content or None, "tool_calls": calls})
        if self._ask_before_commit(calls):
            self.track.history += [{"role": "tool", "tool_call_id": call["id"],
                                    "content": "Not made (no step spent): the customer is being asked first."} for call in calls]
            summary = self.run_interview() or "The interviewer found nothing to ask."
            self.track.history.append({"role": "user", "content": summary + "\n" + self.status_text()})
            return
        self.link.log("loop_turn", text=content[:300], calls=[call["function"] for call in calls])
        replies = self.answer_calls(calls)
        last = len(replies) - 1
        for index, (call_id, text) in enumerate(replies):
            tail = "\n" + self.status_text() if index == last else ""
            self.track.history.append({"role": "tool", "tool_call_id": call_id, "content": text + tail})

    def _tool_calls(self, message: dict) -> list:
        calls = []
        for number, call in enumerate(message.get("tool_calls") or []):
            if not isinstance(call, dict):
                continue
            fn = call["function"] if isinstance(call.get("function"), dict) else {}
            raw = fn.get("arguments")
            parsed = raw if isinstance(raw, dict) else (parse_object(raw) if isinstance(raw, str) and raw.strip() else {})
            arguments = as_json_text(parsed if isinstance(parsed, dict) else {})
            calls.append({"id": str(call.get("id") or f"k{self.track.turns}_{number}"), "type": "function",
                          "function": {"name": str(fn.get("name") or ""), "arguments": arguments}})
        return calls

    def _ask_before_commit(self, calls: list) -> bool:
        """The model wants to cart or finish before the interview ran: run the interview first."""
        if self.track.interviewed or not self.tools.jobs.get("ask") or self.free_messages <= 0:
            return False
        if self.link.elapsed > INTERVIEW_LATEST:
            return False
        committing = {self.tools.jobs.get(job) for job in ("cart", "buy", "refuse")} - {None}
        return self.steps_left > 8 and any(call["function"]["name"] in committing for call in calls)


class Shopper(_LoopPart, _GuardPart, _AskPart, _ListingPart, _TaskPart):
    """One episode: the components and the parts that work on them."""

    def __init__(self, problem: dict):
        environment = problem["environment"]
        binding = dict(environment["binding"])
        policy = environment["policy_view"]
        query = str(policy.get("query") or "")
        self.tools = Toolbox(policy.get("tools"))
        limit = whole_number(policy.get("max_steps"))
        if limit is None and isinstance(policy.get("max_steps"), float):
            limit = int(policy["max_steps"])
        self.task = Task(query, limit if limit and limit > 0 else 30)
        self.task.policy = policy
        self.link = Link(binding, bool(policy.get("finished")))
        self.stock, self.basket, self.track = Stock(), Basket(), Track()


def agent_main(problem_data: dict) -> list:
    try:
        episode = Shopper(problem_data)
    except Exception as exc:
        return [{"kind": "crash", "error": f"{type(exc).__name__}: {exc}"[:400]}]
    try:
        return episode.run()
    except Exception as exc:
        episode.link.log("crash", error=f"{type(exc).__name__}: {exc}"[:400])
        try:
            if not episode.link.finished:
                episode.force_finish()
        except Exception as again:
            episode.link.log("crash", error=f"{type(again).__name__}: {again}"[:400])
        return episode.link.trace


__all__ = ["agent_main"]
