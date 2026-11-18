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

LOOP_CALLS = 36             # model calls of the main loop, and of the whole episode: passing episodes stay under these,
ALL_CALLS = 46              # and past them the 300 s limit ends the episode with no order at all
SPEND_SOFT = 0.055
SPEND_HARD = 0.078
# waits before resending a failed model call; an episode has 300 s in all, so at most 45 s go to waiting
WAITS = (4, 10, 20, 35)
WAIT_TOTAL = 45
SHOW_CHARS = 320
ROW_CHARS = 160             # a search row's text in the conversation: enough to choose what to open, and it is resent every turn
OBS_LIMIT = 15000
HISTORY_LIMIT = 100000      # characters of conversation before old results shrink: long ones slow each call toward the 300 s limit
GATE_ROUNDS = 3             # store calls the process gate may make before one checkout or ending
AUTO_PAGES = 6              # further search pages Kyo reads itself for a cheapest, most or least pick
CHEAPER_OPENS = 3           # cheaper rows of a cart line's kind whose attributes Kyo reads before a cheapest checkout
EARLY_TRAILS = 3            # chain reads while the agent works (after a page view), so a chain's value reaches the ledger
COMPARE_FIRST = 5           # listings sent side by side when the store declares no limit; a declared limit starts at half (stores cap below it)
WIDEN_EACH = 3              # listings of each item whose attributes the question planner should have seen
FOLLOW_STEPS = 14           # most steps Kyo spends following the pages requirements point at itself
FOLLOW_ROUNDS = 10          # rounds (a search, a next page or a further side-by-side read) Kyo spends on the links of those chains
FOLLOW_ROWS = 6             # likely rows of one search result Kyo reads while looking for a chain's link
LONG_STEPS = 70             # a stated step limit this long: after LATE_PLAIN main-loop turns, the light model's turns go
LATE_PLAIN = 26             # without hidden reasoning (several times faster), so the long many-item tasks finish in time;
                            # earlier turns keep it (without it the model gives up on searches far more often)

# one attempt per call (think() resends): a stuck call costs 50 s, not minutes, inside the 300 s
_llm = ProxyClient(timeout=50, max_retries=1, retry_delay=1)
_env = ProxyClient(timeout=95, max_retries=3, retry_delay=1)


GUIDE = """Your name here is Kyo. You shop on behalf of a single customer inside an online store, and your only way to do anything is to call that store's tools.

SCORING
You get 1 for a finish that is exactly right and 0 for anything else. Right means one of two things. Either you check out an order that holds each wanted product at the asked quantity, with every product satisfying all conditions still active at checkout time, for no more than the budget. Or, if nothing in the store can satisfy all of it, you close the episode through the no-purchase tool, phrased the way the request wants.

CLOSING OUT
- The cart should hold the selected lines and nothing more (delete extras); inspect it once, and make checkout a standalone call.
- Ending without buying is allowed only after genuine searching shows no listing satisfies everything. Being low on steps does not show that; in that case check out the best cart you have. Word that ending as the request says, and if a remark is needed, keep it brief and mention no listing. If the ledger states that the order must hold the closest listing when nothing satisfies everything, no such ending exists: cart the closest listing for each item and check out (Kyo does this when you call the ending).
- Never answer without a tool call in it. Any prose stays short: the facts you hold, followed by what you do next.

WHEN THINGS SHIFT MID-TASK
- The customer or the store may alter things along the way: a new budget, a condition added or removed, a price moving, a listing selling out. Such notes come inside tool results, frequently as a refusal at checkout; a note counts no matter which tool's result carries it, and the ledger keeps the conditions currently active. A refusal does not end the episode: apply all changes, repair the cart and check out once more.
- Prices and stock can change without notice, so re-read each chosen listing's record shortly before checkout.

PICKING AMONG QUALIFIERS
- Asked for the cheapest (smallest price, lowest total)? Then the order needs the lowest-priced listing out of EVERY qualifying one. Gather the complete qualifying set by paging (rows often come in no price order, and Kyo itself fetches further pages of a short result), go through candidates from lowest price upward, and keep the first one that qualifies fully. If search accepts a maximum price, set it under the price of your current best so that only cheaper candidates remain.
- A most-or-least request means ranking every qualifying listing by that measure and going with the top (or bottom) one; listings silent on the measure are out of the running.
- With no such pick rule, any listing that satisfies everything is correct, so spend no steps looking for a better one.
- Add an item to the cart the moment you have opened a listing that satisfies all its conditions (for a cheapest, most or least pick, once the qualifying set is compared), then go on to the next. Carted lines often go out of stock shortly after, and carting early leaves you steps to swap them.
- Several different products mean one cart line apiece, each at its asked quantity.

TOOLS AND SEARCH
- Thinking is free; each call to a tool uses up one of a limited number of steps. This store has its own tool names, parameter names and output formats, so study each description and schema and pass precisely the parameter names and permitted values it defines.
- Expect restrictive search: it may filter by category, its keyword mode may look at only the opening words you type, it may return only a handful of rows per page, and further pages need a token or offset. Whenever your answer hinges on the full set, check the reported count against what you have actually seen and fetch more pages using exactly the token or offset you were handed. Short keyword strings that lead with the desired value and then the noun do best. If a filtered query comes back empty, suspect the filter parameter or its value; try again with the label spelled the way rows display it, or switch to another filter parameter.
- A result row is only a summary. Stores divide a record in different ways: one view may give the attributes (properties), another may give just the offer and the description, and a multi-listing view might give either one. Once Kyo has learnt which tool shows attributes, the ledger names it. Decide every property condition from those attributes, so read them for each candidate you might purchase; a single call covering several listings is a step saver if it shows what you need. Do not reopen a listing you already read unless something has changed.
- You never write to the customer. Kyo's interviewer asks the customer on your behalf; replies show up in the conversation and get entered into the ledger.
- Some requests also want actions in between, such as an approval, a message, a thing to verify or to read beforehand; the ledger lists them. The ones due right before ordering or right before closing are done by Kyo the moment you call checkout or the ending tool, so do not hold off for them and never skip buying on their account. Handle the rest with the store's tools, sending questions about products through Kyo's interviewer.

DOES A LISTING QUALIFY
- Product type comes from the listing's category label. Only a label that names the wanted product as a whole qualifies; an adjacent label never does.
- A property condition holds when the listing's attribute for that property gives the value. Ignore how the key is written (spacing, underscores, camelCase, prefixes); for a list split by commas or slashes, a single matching entry suffices. Compare values by wording: capitals, plural versus singular and everyday synonyms do not matter, and a value whose words include the wanted value's words qualifies ("deep teal" covers teal, "extra roomy fit" covers roomy). Different wording fails: a code or abbreviation is not the word behind it, and "teal green" is not "sea green". Convert units before comparing figures; a limit includes its boundary value, and a range qualifies only if all of it sits within the limit.
- If some attribute key names the condition's property (literally or by a plain synonym), that attribute rules: should it give another value, the listing fails no matter what the title or other keys say. A key about some other property, related or not (the colour of a handle says nothing about the body colour), rules on nothing for that condition. After each opened listing Kyo appends [evidence] lines listing the keys whose names fit a condition.
- If no attribute covers the property, you may accept the wanted value when it appears under a different attribute (say, a styles or features list that contains it) or in the title or description. You may not accept a figure of another sort (weight is not capacity) or a bare number whose text does not tell what it measures.
- A category label that itself carries the value (a season or an audience in the label) satisfies that condition by itself.
- Respect closed lists and exclusions. Anything phrased as a wish ("ideally", "would be nice") only breaks ties.
- A sold-out item is never bought. The order total, meaning price times quantity added up across all lines, must stay at or under the budget; hitting it exactly is fine.

CONDITIONS THAT POINT ELSEWHERE
- Instead of a value, a condition may refer to a different listing ("the same X as the listing sold by <shop> with <property>", "within N of the X on the page of ..."). Settle it before anything else: track down that listing (search by its category, brand, shop or wording) and open its full page with the one-item or the side-by-side tool, since a row from search will not do. If that page sends you on to yet another listing for the property, locate and open that too, likewise; whichever page comes last fixes the value. An order only counts if you opened every page along that chain. What the value means for your purchase depends on the condition's own relation word. Same or matching: buy exactly that value. Unlike or different: anything else qualifies, only listings carrying that very value are excluded, so this alone never justifies ending without buying. Suits, goes well with, compatible: buy the partner value that the last page gives for its own value (it may say so directly or state the store's pairing rule, in which case look up the page's own value in it), and never the page's own value. When the chain is finished the ledger shows the value Kyo read, and its pages then count as opened: do not search for them or open them again.
- Should the listing you mean to buy refer you elsewhere for more details on itself, visit that page before purchasing.
- Accessories and the devices they serve (or any product that must fit another) are selected as a pair, judged by what the listings themselves say: a compatibility list, or a model or series code printed on both pages. Suppose every accessory satisfying its own conditions is incompatible with the device you settled on: then move to a different qualifying device that such an accessory does fit. Until that swap has been attempted, ending with no purchase is off the table.
- Some conditions apply only in one case, depending on what a different wanted product ends up having. Choose that other product first, read its record, and enforce the condition only if its case holds; otherwise it demands nothing. A compatibility list may cover a series with a pattern (a letter like x meaning any digit) and list exceptions: a model matching the pattern is covered unless it appears among the exceptions."""


BRIEF = """An agent will shop for someone; you turn that person's request, together with the store's tool list, into a plan. Answer with ONE JSON object (compact, no indentation, nothing else) whose keys are these:

"roles": maps each job to the exact tool that performs it, with null for a job no tool covers: "search", "open" (reveals one listing's attributes or properties; given two single-listing tools, pick the one whose description promises attributes, properties or specifications), "offer" (reveals one listing's live offer, price and stock for instance, if a separate tool from "open" does that), "compare" (several listings in full), "ask" (messages the shopper), "cart" (modifies the cart), "cart_view" (displays the cart), "buy" (checks out), "refuse" (ends with no purchase).
"keyword_search": describes free-text searching as {"tool": the exact tool name, "words": the parameter receiving the text, "fixed": whatever other arguments a text search on that tool requires, a mode for example (exact allowed values), otherwise {}}; null when no tool takes free text.
"buy_args": arguments that stay fixed so that the "buy" tool performs checkout (if that tool takes a mode argument, that mode); otherwise {}.
"refuse_args": arguments that stay fixed so that the "refuse" tool finishes with no purchase; otherwise {}.
"budget": numeric ceiling on what the shopper spends overall, in whatever unit the request uses; null when absent.
"budget_unit": name of that unit written out (for example "cents"); null when absent.
"budget_scope": "total" for a cap on the entire order, "each" for a cap applying to every item separately, null with no cap.
"in_stock_only": set true, except when the request plainly lets unavailable items through.
"pick": "cheapest" for a request after the lowest price or smallest total, "most" or "least" for one after the extreme of a measure, "any" in every other case. Saying funds are tight does not make a pick rule, nor does guidance on which listing to offer if nothing fits.
"pick_measure": with "most"/"least", the measure pushed up or down; otherwise "".
"if_nothing_fits": "closest" when, failing a full match, the request wants the nearest listing put forward, "decline" when it wants the task closed without buying, null if silent.
"closest_rule": in 25 words or fewer, the request's own way of choosing that closest listing; otherwise "".
"quiet_decline": true when the request forbids suggesting or naming alternatives if nothing fits, false when it wants them, null if silent.
"messages": integer count of messages the agent is permitted to send the shopper; null if not stated.
"steps": integer count of steps, moves or tool calls permitted; null otherwise.
"process": any steps, apart from finding and buying the right items, that the request obliges the agent to take (getting the shopper's approval, telling or showing the shopper something, checking something with them, reading something first, using a particular tool), each {"step": the action in 15 words or fewer, "when": "choose" (prior to choosing products), "order" (immediately before checkout) or "end" (immediately before finishing with no purchase), "messages": the number of shopper messages it consumes (0 if none)}; [] if none is asked. Not steps: requirements on items, budget, stock, the caps on messages and steps, and the handling of a no-fit outcome.
"ask_first": points the request explicitly wants confirmed with the shopper ahead of choosing; empty if none.
"rules": each instruction the request gives on working method or on finishing, 12 words maximum apiece.
"wishes": nice-to-haves the shopper mentions without insisting.
"items": a list holding, for every separate product the shopper wants purchased, an object {"name": what sort of product it is, phrased as the request phrases it, "shelf": its category label exactly as some tool lists the store's labels, or "" when no tool lists them, "quantity": integer count of units of that one listing (default 1; a pair or a set sold as a single listing is 1), "musts": that product's firm requirements, written "property: constraint" each, with the precise value, limit, exclusion or closed list. Call the property by the term the request itself uses (so "for a garden party" becomes "occasion: garden party" and "a 30-litre tank" becomes "tank capacity: 30 litre"), not by a neighbouring property. If the value has to come from a different listing, produce ONE must containing the request's whole description of that listing in its own relation words, copied verbatim and never turned into "same as" or any other paraphrase ("strap colour: different from the strap colour of the watch Torvik sells with a steel case"); whatever describes that other listing (its shop, its kind, a trailing "with <property> <value> and <property> <value>") remains inside that single must and is never a must of this product}. Keep the order's overall budget, stock and quantity out of musts; a price cap the request places on one item alone does belong among that item's musts ("price: under 25", say). When a requirement holds only in a case decided by what another item of the order turns out to have (unless, except when, only if, in case), it is ONE must on the product it concerns that keeps the case in the request's words ("cushion colour: grey, unless the sofa's frame is oak"); do not drop the case and do not hand it to the other item. Where a must names the model of a vehicle or device the product has to suit, that model code stays in it. The surrounding story (the intended user, where it will stand, earlier mishaps, the occasion behind it) is context rather than a must, except where it outright gives a property's value or limit; and words that are part of the product's name (a season, audience or use named together with the product) are never musts: they go in "name". Products named in one phrase joined by "and" (a single "I want ..." with a single quantity, frequently the name of one store department) make ONE item; create separate items only if each product gets its own quantity or its own requirements.
"lookups": those musts (copied exactly) whose value is to be read beforehand from another listing the request refers to; otherwise []. Musts concerning a further item of this very order (it has to fit or suit whatever gets purchased) are not lookups.
"""

PLAN = """Role: pick shopper questions for a shopping agent.
Facts: shoppers keep few unwritten firm wishes; each = one property of one requested product. Only a question naming exactly that property of exactly that product surfaces it; only its first-named property gets answered -> one question spends one property.
Input: request; items, each with what request already pins down; per item category, numbered attribute keys from opened listings (listing count per key, sample values). Hidden wishes nearly always sit on one listed key of their item.
Rules:
1. First: anything request says to confirm with shopper (shown under ASK FIRST when present), each named by its property.
2. Next, hinted properties. Stories hint a wish yet withhold its value: line on what shopper can't judge or leaves to you, whose taste rules, looks or surroundings, intended user or wearer, contents. Per line, story order: one property + target item, named by that item's listed key. Wearer hint -> gender. Looks hint -> colour, unless line concerns a facet another listed key names (shape, pattern, finish).
3. Then remaining listed keys, ranked: colour top if request leaves it open; next, keys most listings of that item carry with differing values; skip keys on just one or two listings.
4. Multiple items: every item's hinted properties first; afterwards alternate items, least-pinned first. Items bound to another product or shopper's car/device (must fit/suit) seldom hide wishes -> their turn comes after others.
5. Exclude: properties request already pins for that item; duplicates of chosen properties (keys differing only by spelling/prefix = same); keys listing purchase variants (mixed sizes, colours, pack counts; often "options"); identifiers; brand; model; condition; warranty; other logistics.
6. "label" = plain property name; listed key -> all its words spelled plainly (lidColour -> "lid colour", rim_finish -> "rim finish"; never trimmed to "colour"/"finish"); name only: no sample values, brackets, second property.
7. Max COUNT, strongest first.
Output: ONE JSON object and nothing else: {"ask": [{"item": item number, "label": "...", "hint": "up to 8 source words from story, or an empty string"}]}"""


HEAR = """Role: extract what a shopper's replies to an agent's questions establish.
Rules:
1. "needs" holds each firm preference a reply states, exact value or limit plus unit ("at least 30 cm", "oval", "not wicker"). Use the property the reply itself names, even if it differs from the one asked; several replies on one property form one need. Copy the value verbatim with its nouns and units ("at least 3 shelves", never "at least 3").
2. A reply may set the value by reference to another listing (like, unlike or close to the value on some shop's product, identified by a few of its properties). That is ONE need on the named property with "points": true; its value is the reply's exact words from the comparison words through the end of that description, including anything said about a listing that page may name; no trimming or rewording. Properties used to describe the other product are not needs of this item.
3. "free" lists the asked property of each question the shopper said they don't care about.
4. Colloquial or indirect phrasing that conveys a value still conveys it.
Output: ONE JSON object and nothing else:
{"needs": [{"item": item number, "need": "property: constraint", "points": true if the value comes from another listing, else false}], "drop": [{"item": item number, "property": "requirement the shopper has abandoned"}], "free": [{"item": item number, "property": "..."}], "budget": revised overall cap as a number, or null}"""


NOTE = """Role: sort out a message that appeared during the shopping run (sent by the shopper or by the store).
Rules:
1. Capture all changes it carries.
2. A requirement replaced by another counts as removing the old one and adding the new one.
Output: ONE JSON object and nothing else:
{"kind": "change" (task altered: budget, requirements, a price, availability, the order refused) | "error" (merely says a call was rejected or badly formed) | "other" (all else), "budget": revised overall cap as a number in the request's unit, else null, "add": [{"item": item number, "need": "property: constraint"}], "drop": [{"item": item number, "need": "the dropped requirement, verbatim from the list above"}], "gone": [ids of listings reported unavailable], "sold_out": true if a picked listing is reported gone, sold out or unavailable (named or not), "price_moved": true if a price is reported changed, "summary": "a brief sentence"}"""


CHECK = """Role: final gate before checkout; judge from the shown records whether the cart is exactly right.
Rules:
1. Settle each property by the attribute whose key is that property, however spelled: a single list entry is enough; case and singular/plural differences do not matter; plain synonyms count; a value that includes the wanted words passes ("deep teal" satisfies teal); convert units; bounds are inclusive; a range has to sit fully within the limit. If that attribute says something different, the listing fails.
2. An attribute about a neighbouring property (handle colour versus body colour) settles nothing.
3. If no attribute covers the property, accept the wanted value when another attribute, the title or the description states it; not a number of a different kind (weight is no capacity) and not a bare number that doesn't say what it measures.
4. Whatever the category label states is satisfied by it. Price rules go by that listing's own price. The category label has to match the whole requested item.
5. An "unlike listing X's value" rule passes unless the records show the same value. A "suits / pairs with listing X's value" rule needs the partner value X's page names for itself (shown in the ledger), not X's own value.
6. A rule that applies only in some case is checked against the record of the cart line of the other item it hinges on, and is void if that case is false.
7. A compatibility list can cover a model family by pattern (x for any digit, say) with named exceptions; a pattern-matching model is covered unless named as an exception.
8. Cart: one separate line per requested item at the requested quantity; no extra lines; no line shown out of stock; price x quantity summed across lines stays within budget. A listing's latest record overrides older ones.
9. Pick rule cheapest, most or least: scan the other shown listings too; one that plainly satisfies everything and beats the cart's pick on that rule makes the cart wrong.
10. ok=false only for a fault the records plainly show, or for a requirement no record speaks to.
Output: ONE JSON object and nothing else: {"ok": true or false, "problems": ["brief specific issue", ...], "better": [ids of replacement listings for a cart line, strongest first]}"""


READING = """attribute keyed to that very property decides (any key spelling; one list entry suffices; synonyms, unit conversions ok; value containing wanted words passes, e.g. "deep teal" for teal; bounds inclusive); key for a neighbouring property (handle colour vs body colour) decides nothing; no attribute for it -> wanted value stated under another attribute, title or description counts (bare number not saying what it measures: never); category label's own statement satisfies that requirement; price rules use listing's own price; category label must match item; "unlike listing X's value" passes unless records show that same value; "suits / pairs with listing X's value" needs partner value X's page names for itself (ledger shows it), never X's own value; case-dependent rule judged against record of cart line or chosen listing of the other item it hinges on, void when case fails; compatibility list may cover model family by pattern (x = any digit, say) with named exceptions: pattern-matching model covered unless named exception; out-of-stock or over-budget listings never fit."""

AUDIT = """Role: second opinion before the agent quits with no purchase because it believes no listing qualifies; test each shown listing record against the requirements. How to read records: """ + READING + """
Rule: a record marked ROW is only a search result row; its page (attributes) is unread, so it can neither pass nor fail on a property the row doesn't show.
Output: ONE JSON object and nothing else: {"fits": {"item number": [up to 4 ids seemingly meeting all requirements, lowest price first]}, "open": [up to 6 ROW ids of a wanted kind and within budget whose page might prove a fit, likeliest first], "reason": "a single sentence"}"""

FITS = """Role: a requirement's value was just looked up on a different listing's page; recheck the OPENED listing records below against the full current requirement set. How to read records: """ + READING + """
Output: ONE JSON object and nothing else: {"fits": {"item number": [up to 3 ids of listings seemingly meeting all that item's requirements, ordered best-first by the pick rule]}}"""

CLOSEST = """Role: no listing surely fits everything, so the agent will order the nearest ones; per requested item, pick from the records below the listing of that product kind with the fewest unmet requirements (availability and the budget count as requirements as well). How to read records: """ + READING + """
Rules:
1. A ROW record (search result row only) may be chosen only if no OPENED record of that kind exists.
2. Ties on misses: the request's own closest rule, else the cheapest.
3. A line already in the cart stays unless a record clearly misses fewer.
4. One line per item that any record can serve.
Output: ONE JSON object and nothing else: {"lines": [{"item": item number, "id": listing id, "misses": count of unmet requirements}], "budget_blocks": true if each requested item has a recorded listing satisfying everything for it except price, yet no combination of those fits the budget (only the budget prevents a fitting order), else false}"""

UNMIX = """Role: requirement untangler. Some firm product requirements take value from another listing request mentions; phrases identifying it (its kind, shop, properties added with "with ...") get misread as requirements on purchased item.
Rule: per request wording, flag OTHER REQUIREMENTS below that only describe that other listing.
Output: ONE JSON object and nothing else: {"describe_other": [the flagged requirements, verbatim; [] if all are genuine product requirements]}"""

TRAIL = """Role: chain tracer before checkout. Some requirements borrow their value from another listing; a page may redirect to a further listing for that property. Ordering allowed only once agent has fully opened every listing on every chain.
Rules:
1. Start at requirement's own description; walk link by link via pages' own wording, using records below (OPENED = agent opened page; ROW = search-result row only, page unread).
2. Valid link = listing whose own record shows values its description gives. Same-shop or same-kind listing with other values is not that link; never follow its page.
3. Request holds each requirement's relation in shopper's words; those words override differing wording below.
Output: ONE JSON object and nothing else: {"chains": [{"need": requirement number below, "links": [record ids in chain order so far; stop at first link lacking OPENED page], "unmatched": [final link's described values its record lacks; any such value = wrong listing, exclude from "links"], "done": true only if final link is OPENED page stating property value itself, no redirect onward for that property (redirecting page = middle link, despite own value), else false, "about": if not done, max 20 words: how requirement or final page identifies next listing, else "", "find": if not done, compact keyword query for next listing (shop name + few telling described values; add product kind only when vague), else "", "own": final page's value for its own listing, or "", "partner": value page pairs with own (stated outright, or via page's pairing rule keyed by own value), or "", "value": purchased product's required value per requirement's relation word, as meant in shopper's dialect: own value for same/matching and for only-that-value wording ("nothing but that one"); "anything but " followed by own value for unlike/different; partner for suits/goes well with/compatible (own value when no partner named); "" if not done}]}"""

GATE = """Role: process gate. Agent about to STAGE; request demands steps below en route.
Rules:
1. Per step owed beforehand: complete for current cart? Decide from numbered RECORD of agent actions.
2. Complete = some RECORD line performs it; cart/order steps additionally need that line after latest cart change, with reply approving. Property questions never complete steps.
3. Step missing -> return single next store call performing it: TOOLS tool + arguments. Shopper messages go via messaging tool: one brief plain sentence/question on that step only; mention no product property/value step doesn't need.
4. No call when all owed steps complete, or remaining messages/steps can't achieve missing one.
Output: ONE JSON object and nothing else: {"steps": [{"step": the step, "done": true or false, "lines": [proving RECORD line numbers]}], "call": {"tool": "...", "args": {...}} or null, "hold": change a reply demands before continuing, or "", "needs": [{"item": item number, "need": "property: constraint"} per firm requirement such reply lays down]}"""

SWAP = """Role: one cart item's listing is now sold out; search the remaining listing records shown for substitutes for that line. How to read records: """ + READING + """
Output: ONE JSON object and nothing else: {"fits": [{"id": listing id, "price": numeric price as recorded} per listing seemingly meeting all requirements of the sold-out line's item, max 6], "reason": "a single sentence"}"""

SHAPE = """Role: map the parts of the store tool result below, using the exact keys found in the result and in the tool's parameters.
Output: ONE JSON object and nothing else: {"rows": key whose value is the listing array, "count": key giving the overall match total (not the shown total) or null, "next": key whose value requests the next page or null, "next_arg": parameter of the tool that accepts that value or null, "price": per-row key carrying price or null, "stock": per-row key telling if it is purchasable right now or null}"""


import dataclasses


def _dump(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        return str(value)


def _clip(value: Any, limit: int, depth: int = 0) -> Any:
    if depth > 7:
        return "..."
    if isinstance(value, str):
        return value[:limit] + "..." if len(value) > limit else value
    deeper = depth + 1
    if isinstance(value, dict):
        return dict(zip(value, (_clip(v, limit, deeper) for v in value.values())))
    if isinstance(value, list):
        return list(map(lambda v: _clip(v, limit, deeper), value))
    return value


def _json_attempts(body: str):
    """Texts worth parsing, best first: the whole body, then its outermost braces, then those without trailing commas."""
    yield body
    left, right = body.find("{"), body.rfind("}")
    if 0 <= left < right:
        part = body[left:right + 1]
        yield part
        yield re.sub(r",\s*([}\]])", r"\1", part)


def _json_obj(text: Any) -> dict[str, Any] | None:
    if not isinstance(text, str):
        return None
    body = text.strip()
    fence = re.match(r"^```[A-Za-z]*\s*(.*?)\s*```$", body, re.S)
    if fence:
        body = fence.group(1)
    for attempt in _json_attempts(body):
        try:
            got = json.loads(attempt)
        except ValueError:
            continue
        return got if isinstance(got, dict) else None
    return None


def _words(key: str) -> str:
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(key))
    return " ".join(re.split(r"[\s_.\-]+", spaced)).strip().lower()


def _singular(word: str) -> str:
    return word[:-1] if len(word) > 3 and word.endswith("s") else word


def _stem(text: str) -> frozenset[str]:
    return frozenset(map(_singular, _words(text).split()))


def _names(prop: frozenset[str], key: frozenset[str]) -> bool:
    if prop and key:
        return prop <= key or (len(key) >= 2 and key <= prop)
    return False


def _kinds(schema: dict[str, Any]) -> list[str]:
    kind = schema.get("type")
    pool = [kind] if isinstance(kind, str) else (kind or [])
    return [k for k in pool if isinstance(k, str)]


def _plain_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


_KIND_TESTS = {
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": _plain_number,
    "boolean": lambda v: isinstance(v, bool),
    "array": lambda v: isinstance(v, list),
    "object": lambda v: isinstance(v, dict),
    "null": lambda v: v is None,
}


def _fits_kind(value: Any, kind: str) -> bool:
    test = _KIND_TESTS.get(kind)
    return test is None or test(value)


_UNBENT = object()


def _numeric_text(value: Any, whole: bool) -> Any:
    if not (isinstance(value, str) and re.fullmatch(r"\s*-?\d+(\.\d+)?\s*", value)):
        return _UNBENT
    number = float(value)
    return int(number) if whole or number.is_integer() else number


def _bend_integer(value: Any) -> Any:
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return _numeric_text(value, True)


def _bend_boolean(value: Any) -> Any:
    word = value.strip().lower() if isinstance(value, str) else None
    return _UNBENT if word not in ("true", "false") else word == "true"


# one repair per schema kind; each gives _UNBENT when the value cannot be bent that way
_BENDERS = {
    "integer": _bend_integer,
    "number": lambda v: _numeric_text(v, False),
    "string": lambda v: str(v) if _plain_number(v) else _UNBENT,
    "boolean": _bend_boolean,
    "array": lambda v: _UNBENT if isinstance(v, list) or v is None else [v],
}


def _fits_any(value: Any, kinds: list[str]) -> bool:
    return any(_fits_kind(value, k) for k in kinds)


def _bend(value: Any, kinds: list[str]) -> Any:
    tries = (_BENDERS[k](value) for k in kinds if k in _BENDERS)
    return next((got for got in tries if got is not _UNBENT), value)


def _enum_fix(value: Any, options: list[Any], where: str, faults: list[str]) -> Any:
    if isinstance(value, str):
        low = value.strip().lower()
        value = next((o for o in options if isinstance(o, str) and o.lower() == low), value)
    if value not in options:
        faults.append(f"{where}: allowed values are {options}")
    return value


_NUMBER_EDGES = (
    ("minimum", lambda v, e: v < e, "is below"),
    ("maximum", lambda v, e: v > e, "is above"),
)
_COUNT_EDGES = (
    ("maxItems", lambda n, e: n > e, "allows at most {} entries"),
    ("minItems", lambda n, e: n < e, "needs at least {} entries"),
)


def _text_limits(value: str, schema: dict[str, Any], where: str, faults: list[str]) -> str:
    cap, floor = schema.get("maxLength"), schema.get("minLength")
    if isinstance(cap, int) and len(value) > cap:
        value = value[:cap]
    if isinstance(floor, int) and len(value) < floor:
        faults.append(f"{where} is too short")
    return value


def _list_limits(value: list[Any], schema: dict[str, Any], where: str, faults: list[str]) -> list[Any]:
    size = len(value)
    faults.extend(f"{where} " + words.format(schema[key]) for key, over, words in _COUNT_EDGES
                  if isinstance(schema.get(key), int) and over(size, schema[key]))
    inner = schema.get("items")
    if not isinstance(inner, dict):
        return value
    return [_shape(v, inner, f"{where}[{i}]", faults) for i, v in enumerate(value)]


def _dict_limits(value: dict[str, Any], schema: dict[str, Any], where: str, faults: list[str]) -> dict[str, Any]:
    props = schema.get("properties")
    if not isinstance(props, dict):
        props = {}
    if schema.get("additionalProperties") is False:
        faults.extend([f"{where}.{n} is not an allowed argument" for n in value if n not in props])
    faults.extend(f"{where}.{n} is required" for n in (schema.get("required") or []) if n not in value)
    return {k: (_shape(v, props[k], f"{where}.{k}", faults) if k in props else v) for k, v in value.items()}


def _shape(value: Any, schema: Any, where: str, faults: list[str]) -> Any:
    """Push a value toward its JSON schema, noting whatever cannot be repaired."""
    if not isinstance(schema, dict):
        return value
    kinds = _kinds(schema)
    if kinds and not _fits_any(value, kinds):
        value = _bend(value, kinds)
        if not _fits_any(value, kinds):
            faults.append(f"{where} should be {'/'.join(kinds)}")
            return value
    options = schema.get("enum")
    if isinstance(options, list) and value not in options:
        value = _enum_fix(value, options, where, faults)
    if isinstance(value, str):
        value = _text_limits(value, schema, where, faults)
    if _plain_number(value):
        faults.extend(f"{where} {words} {schema[key]}" for key, over, words in _NUMBER_EDGES
                      if isinstance(schema.get(key), (int, float)) and over(value, schema[key]))
    if isinstance(value, list):
        value = _list_limits(value, schema, where, faults)
    if isinstance(value, dict):
        value = _dict_limits(value, schema, where, faults)
    return value


def _items(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return [value] if isinstance(value, (str, dict)) and value else []


def _count(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    whole_float = isinstance(value, float) and value.is_integer()
    whole_text = isinstance(value, str) and re.fullmatch(r"\s*\d+(\.0+)?\s*", value) is not None
    return int(float(value)) if whole_float or whole_text else None


def _id_like(value: Any, named: bool = False) -> bool:
    if isinstance(value, str):
        core = value.strip()
        if 3 <= len(core) <= 64 and " " not in core:
            return named or bool(re.search(r"\d", value))
    return False


def _id_score(rows: list[dict[str, Any]], key: Any) -> tuple[bool, float] | None:
    named = "id" in str(key).lower()
    raw = [row.get(key) for row in rows]
    if not all(_id_like(v, named) for v in raw):
        return None
    values = [v.strip() for v in raw]
    if len(rows) > 1 and len(set(values)) < len(values):
        return None
    return named, sum(map(len, values)) / len(values)


def _learn_id(rows: list[dict[str, Any]]) -> str | None:
    """Pick the row-identifying key: id-looking strings, distinct across rows, preferring 'id' names then longer values."""
    scored = [(score, key) for key in rows[0] for score in (_id_score(rows, key),) if score is not None]
    return max(scored, key=lambda pair: pair[0])[1] if scored else None


def _in_rows(node: Any, ident: str, depth: int = 0) -> bool:
    """True when some list in the result holds a record carrying this id (e.g. a cart still listing the line)."""
    pending = [(node, depth)]
    while pending:
        here, level = pending.pop()
        if level > 5:
            continue
        if isinstance(here, list):
            if any(isinstance(v, dict) and any(ident == str(x).strip() for x in v.values() if isinstance(x, str)) for v in here):
                return True
            pending.extend((kid, level + 1) for kid in here)
        elif isinstance(here, dict):
            pending.extend((kid, level + 1) for kid in here.values())
    return False


def _scalar_fields(block: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in block.items() if isinstance(v, (str, int, float)) and not isinstance(v, bool)}


def _fact_block(node: Any) -> dict[str, Any] | None:
    if not isinstance(node, dict):
        return None
    blocks = [_scalar_fields(v) for v in node.values() if isinstance(v, dict) and len(v) >= 2]
    blocks = [b for b in blocks if len(b) >= 2]
    return max(blocks, key=len) if blocks else None


@dataclasses.dataclass(eq=False)
class Item:
    name: str
    shelf: str
    qty: int
    musts: list[str]
    told: list[str] = dataclasses.field(default_factory=list, init=False)
    dropped: list[str] = dataclasses.field(default_factory=list, init=False)
    free: list[str] = dataclasses.field(default_factory=list, init=False)

class Kyo:
    # per-episode state and its starting value; a type here means a fresh empty one for each episode
    _START: tuple[tuple[str, Any], ...] = (
        ("plain_loop", False), ("reasoning_refused", False), ("step", 0), ("broke", False), ("no_credit", False),
        ("waited", 0), ("misses", 0), ("cost", 0.0), ("calls", 0), ("loop_calls", 0), ("trace", list), ("brief", dict),
        ("roles", dict), ("items", list), ("budget", None), ("msg_limit", 0), ("msg_used", 0), ("process", list),
        ("reserve", 0), ("gate_msgs", 0), ("gate_ok", set), ("done_log", list), ("done_steps", list), ("asked", False),
        ("qa", list), ("rows", dict), ("full", dict), ("seen_order", list), ("heard", set), ("notes", list),
        ("version", 0), ("cart_touched", False), ("cart_at", -1), ("cart_seen_at", -1), ("cart_view", None),
        ("cart_lines", dict), ("checked", set), ("audited", False), ("offer_seen", dict), ("gone", dict),
        ("unavailable", set), ("sold_lines", set), ("note_sold", False), ("held", 0), ("id_args", set),
        ("cart_said", ""), ("refused_at", -1), ("leads", list), ("refuse_blocks", 0), ("pushed", False),
        ("built", list), ("dropped", set), ("late_held", False), ("lead_items", dict), ("verdicts", dict),
        ("confirmed", None), ("completed", False), ("id_key", None), ("facts_from", dict), ("queued", list),
        ("buys", 0), ("refusals", 0), ("recent", list), ("chat", list), ("opened", set), ("refs", list),
        ("ref_values", dict), ("ref_early", dict), ("note_flags", dict), ("next_find", dict), ("follow_reads", 0),
        ("closest_tries", 0), ("nothing_fits", False), ("followed", set), ("empty_filters", list), ("blind", dict),
        ("filters_ok", set), ("trails", 0), ("early_trails", 0), ("trail_blocks", 0), ("compare_ok", 0),
        ("compare_bad", 0), ("tool_keys", dict), ("view_keys", dict), ("read_cache", dict), ("pages_read", 0),
        ("shapes", dict), ("price_key", None), ("stock_key", None), ("cheaper_marks", set), ("wider_held", False),
        ("paged_obs", list),
    )

    def __init__(self, problem: dict[str, Any]):
        env = problem["environment"]
        self.binding = dict(env["binding"])
        self.view = env["policy_view"]
        self.request = str(self.view.get("query") or "")
        self.tools = list(filter(self._usable_tool, self.view.get("tools") or []))
        self.schemas = dict(map(self._tool_schema, self.tools))
        self.max_steps = self._step_cap(self.view.get("max_steps"))
        for attr, start in self._START:
            setattr(self, attr, start() if isinstance(start, type) else start)
        self.finished = bool(self.view.get("finished"))

    @staticmethod
    def _usable_tool(spec: Any) -> bool:
        fn = spec.get("function") if isinstance(spec, dict) else None
        return isinstance(fn, dict) and isinstance(fn.get("name"), str)

    @staticmethod
    def _tool_schema(spec: dict[str, Any]) -> tuple[str, Any]:
        fn = spec["function"]
        params = fn.get("parameters")
        return fn["name"], (params if isinstance(params, dict) else {"type": "object", "properties": {}})

    @staticmethod
    def _step_cap(raw: Any) -> int:
        n = _count(raw)
        if n is None and isinstance(raw, float):
            n = int(raw)
        return 30 if not n or n <= 0 else n

    # ---------- plumbing ----------
    def left(self) -> int:
        return self.max_steps - self.step

    def msgs_free(self) -> int:
        """Messages left to the interviewer once the ones held back for steps at the order or the ending are set aside."""
        held_back = self.reserve - self.gate_msgs
        return self.msg_limit - self.msg_used - (held_back if held_back > 0 else 0)

    def cart_mark(self) -> str:
        return _dump(sorted(self.cart_lines.items()))

    def page_tools(self) -> set[str]:
        """Tools showing a listing's own page (offer view if the store has one, else single-listing view) plus compare."""
        roles = self.roles
        return set(filter(None, (roles.get("offer") or roles.get("open"), roles.get("compare"))))

    def facts_tool(self) -> str | None:
        """Single-listing tool that shows attributes: one already seen doing so, else the brief's attributes view."""
        for name, ok in self.facts_from.items():
            if not (ok and self.single_target(name)):
                continue
            if name:
                return name
            break
        fallback = self.roles.get("open")
        if self.facts_from.get(fallback or "") is False or not self.single_target(fallback):
            return None
        return fallback

    def compare_facts(self) -> bool | None:
        """True/False once the side-by-side view has shown (or not shown) attributes; None before it is tried."""
        return self.facts_from.get(self.roles.get("compare") or "")

    def compare_bare(self) -> bool:
        """Side-by-side view cannot be used to read attributes: it showed none, or, untried, a separate attributes
        view exists (one that showed them, or one apart from the offer view)."""
        seen = self.compare_facts()
        if seen is not None:
            return seen is False
        return self.facts_from.get(self.facts_tool() or "") is True or bool(self.roles.get("offer"))

    def gate_due(self, stage: str) -> bool:
        # a step due at this stage and the gate not yet passed for the cart in its present state
        if all(p["when"] != stage for p in self.process):
            return False
        return (stage, f"{self.cart_mark()}#{self.version}") not in self.gate_ok

    def tail(self) -> int:
        """Steps kept for the finish: checkout, plus one while a step due at the order still needs a message."""
        if not any(p["when"] == "order" and p["messages"] > 0 for p in self.process):
            return 1
        return 2 if self.msg_used < self.msg_limit and self.gate_due("order") else 1

    def _pause_for(self, status: Any, tries: int) -> float | None:
        """Wait before resending a failed call, or None when it is not resent. Credit, rate-limit, timeout and server
        failures are resent; a timeout already spent the client's 50 s, so it goes once more at most."""
        if status in (None, 408):
            cap = 1
        elif status in (402, 425, 429) or (isinstance(status, int) and status >= 500):
            cap = len(WAITS)
        else:
            cap = 0  # any other failure is final
        if tries >= cap or self.waited + WAITS[tries] > WAIT_TOTAL:
            return None
        base = WAITS[tries]
        pause = base * (0.8 + 0.4 * random.random())
        self.waited += base
        return pause

    def _attempts(self, model: str, messages: list[dict[str, Any]], max_tokens: int, tools: list[dict[str, Any]] | None,
                  plain: bool, tries: int = 0) -> tuple[dict[str, Any] | None, Any]:
        """One model's call, resent (recursively) while its failure allows; `tries` counts the waits taken so far."""
        message, status = self.call_model(model, messages, max_tokens, tools, plain)
        if message is None:
            if status == 400 and plain and not self.reasoning_refused:
                # route rejects the reasoning setting: resend without it, and drop it from later calls too
                self.reasoning_refused = True
                return self._attempts(model, messages, max_tokens, tools, plain, tries)
            pause = self._pause_for(status, tries)
            if pause is not None:
                time.sleep(pause)
                return self._attempts(model, messages, max_tokens, tools, plain, tries + 1)
        return message, status

    def think(self, messages: list[dict[str, Any]], models: list[str], max_tokens: int, tools: list[dict[str, Any]] | None = None,
              plain: bool = False) -> dict[str, Any] | None:
        for model in models:
            if self.calls >= ALL_CALLS:
                break
            self.calls += 1
            message, status = self._attempts(model, messages, max_tokens, tools, plain)
            if message is None and status != 402:
                continue
            if message is None:
                self.no_credit = True
            return message
        return None

    def _model_body(self, model: str, messages: list[dict[str, Any]], max_tokens: int, tools: list[dict[str, Any]] | None,
                    plain: bool) -> dict[str, Any]:
        sent = messages
        if model.startswith("anthropic/") and len(messages) > 1:
            ends = {0, len(messages) - 1}
            sent = [
                {**m, "content": [{"type": "text", "text": m["content"], "cache_control": {"type": "ephemeral"}}]}
                if i in ends and isinstance(m.get("content"), str) and m["content"] else dict(m)
                for i, m in enumerate(messages)
            ]
        body: dict[str, Any] = {"model": model, "messages": sent, "max_tokens": max_tokens, "temperature": 0}
        if plain and model == LIGHT and not self.reasoning_refused:
            body["reasoning"] = {"enabled": False}
        if tools:
            body.update(tools=tools, tool_choice="auto")
        return body

    def _llm_failed(self, model: str, status: Any, body: Any = None) -> tuple[None, Any]:
        record: dict[str, Any] = {"kind": "llm_fail", "model": model, "status": status}
        if body is not None:
            record["body"] = str(body or "")[:200]
        self.trace.append(record)
        return None, status

    def call_model(self, model: str, messages: list[dict[str, Any]], max_tokens: int, tools: list[dict[str, Any]] | None,
                   plain: bool = False) -> tuple[dict[str, Any] | None, Any]:
        """One inference call: the reply message, or None and the failure's status (0 for a reply without content)."""
        result = _llm.post_verbose("/inference/chat/completions", self._model_body(model, messages, max_tokens, tools, plain))
        data = result.data
        if not (isinstance(data, dict) and data.get("choices")):
            inner = data.get("error") if isinstance(data, dict) else None
            if isinstance(inner, dict):
                # error delivered inside a 200 reply, carrying the provider's own code
                status, why = _count(inner.get("code")), inner.get("message")
            else:
                err = result.error or {}
                status, why = err.get("status"), err.get("body")
            return self._llm_failed(model, status, why or "")
        spent = (data.get("usage") or {}).get("cost")
        if isinstance(spent, (int, float)):
            self.cost += float(spent)
        message = data["choices"][0].get("message") or {}
        if not (message.get("content") or message.get("tool_calls")):
            return self._llm_failed(model, 0)
        return message, 200

    def ask_json(self, prompt: str, models: list[str], max_tokens: int = 1800) -> dict[str, Any] | None:
        for model in models:
            reply = self.think([{"role": "user", "content": prompt}], [model], max_tokens)
            got = _json_obj((reply or {}).get("content"))
            if got is not None or self.broke or self.no_credit or self.calls >= ALL_CALLS:
                return got
        return None

    def _env_payload(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        return {**self.binding, "step": self.step + 1, "action": {"name": name, "args": args}}

    def _env_retry(self, result: Any) -> bool:
        """Log a failed environment call; whether it goes again (the step number already corrected), False ends it."""
        err = result.error or {}
        status, body = err.get("status"), str(err.get("body") or "")
        self.trace.append({"kind": "env_fail", "status": status, "body": body[:200]})
        if status in (400, 401, 402, 403, 404, 422):
            if status == 402:
                self.broke = True
            return False
        if status != 409:
            return True
        low = body.lower()
        if any(word in low for word in ("finish", "quarantin", "finaliz")):
            self.finished = True
            return False
        expected = re.search(r"expected (\d+)", body)
        if expected:
            corrected = int(expected.group(1)) - 1
        elif "reuse" in low or "different action" in low:
            corrected = self.step + 1
        else:
            return False
        self.step = corrected
        return True

    def _env_read(self, name: str, args: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        reported = data.get("step")
        self.step = reported if isinstance(reported, int) else self.step + 1
        if data.get("finished"):
            self.finished = True
        seen = data.get("observation")
        if isinstance(seen, dict) and seen.get("error") or name not in self.page_tools():
            return data
        # listings whose own page the agent has opened (a search row does not count): only those the answer shows,
        # since a refused call answers without them
        shown = _dump(seen)
        for value in args.values():
            for item in value if isinstance(value, list) else (value,):
                if isinstance(item, str) and item.strip() in shown:
                    self.opened.add(item.strip())
        return data

    def act(self, name: str, args: dict[str, Any]) -> dict[str, Any] | None:
        payload = self._env_payload(name, args)
        # the generator posts lazily, so each resend carries the step number corrected just before it
        for result in (_env.post_verbose("/environment/call", payload) for _ in range(4)):
            if isinstance(result.data, dict):
                return self._env_read(name, args, result.data)
            if not self._env_retry(result):
                break
            payload["step"] = self.step + 1
        return None

    # ---------- reading the request ----------
    def _rq_prompt(self) -> str:
        parts = [BRIEF, "\nSTORE TOOLS:\n"]
        parts.append("\n".join(map(lambda fn: f"- {fn['name']}: {fn.get('description', '')} | parameters: {_dump(fn.get('parameters') or {})}",
                                    [t["function"] for t in self.tools])))
        parts += ["\n\nTHE SHOPPER'S REQUEST:\n", self.request]
        if self.view.get("opening") is not None:
            parts += ["\n\nWHAT THE STORE SAID AT THE START:\n", _dump(_clip(self.view.get("opening"), 1500))]
        return "".join(parts)

    def _rq_needs_heavy(self, brief: dict) -> bool:
        if self.no_credit:
            return False
        roles = brief.get("roles")
        return not (_items(brief.get("items")) and isinstance(roles, dict) and roles.get("buy"))

    def _rq_split_modes(self, brief: dict) -> None:
        """One tool both checks out and declines: learn the arguments that choose each mode."""
        buy = self.roles.get("buy")
        clash = bool(buy) and buy == self.roles.get("refuse")
        settled = all((brief["buy_args"], brief["refuse_args"])) and brief["buy_args"] != brief["refuse_args"]
        if not clash or settled:
            return
        ask = (f"The store tool {buy} can complete the purchase or close the episode with no purchase; its arguments decide which. Schema: "
               + _dump(self.schemas.get(buy)) + "\nReply with a single JSON object only: {\"buy_args\": the constant arguments that make it complete the purchase, \"refuse_args\": the constant arguments that make it close with no purchase} (just the arguments that select the mode).")
        fixed = self.ask_json(ask, [LIGHT, HEAVY], 600) or {}
        b_args, r_args = fixed.get("buy_args"), fixed.get("refuse_args")
        if isinstance(b_args, dict) and isinstance(r_args, dict) and b_args != r_args:
            brief.update(buy_args=b_args, refuse_args=r_args)

    def _rq_items(self, brief: dict) -> None:
        def build(entry: dict) -> Item:
            wanted = _count(entry.get("quantity"))
            return Item(str(entry.get("name") or "the product"), str(entry.get("shelf") or ""),
                        wanted if wanted and wanted > 0 else 1, list(map(str, filter(None, _items(entry.get("musts"))))))
        self.items.extend(build(e) for e in _items(brief.get("items")) if isinstance(e, dict))
        if not self.items:
            self.items = [Item("the requested product", "", 1, [])]

    def _rq_lookups(self, brief: dict) -> None:
        def settle(text: str) -> str:
            pool = [m for item in self.items for m in item.musts]
            if text in pool:
                return text
            # copied loosely: the must on the same property stands for it
            key = _words(text.split(":")[0])
            same = [m for m in pool if _words(m.split(":")[0]) == key]
            return same[0] if same else text
        for text in (settle(str(raw).strip()) for raw in _items(brief.get("lookups"))):
            owners = [n for n, item in enumerate(self.items, 1) if text in item.musts]
            ref = (owners[0] if owners else 1, text)
            if text and ref not in self.refs:
                self.refs.append(ref)

    def _rq_budget(self, value: Any) -> None:
        numeric = isinstance(value, str) and re.fullmatch(r"\s*\d+(\.\d+)?\s*", value)
        amount = float(value) if numeric else value
        if isinstance(amount, (int, float)) and not isinstance(amount, bool) and amount > 0:
            self.budget = amount

    def _rq_messages(self, value: Any) -> None:
        limit = _count(value)
        if limit is None or limit < 0:
            if self.roles.get("ask"):
                self.msg_limit = 3
            return
        self.msg_limit = limit

    def _rq_steps(self, value: Any) -> None:
        steps = _count(value)
        if steps is not None and 10 <= steps < self.max_steps:
            self.max_steps = steps

    def _rq_process(self, brief: dict) -> None:
        def shape(entry: dict) -> dict:
            when = str(entry.get("when") or "").strip().lower()
            return {"step": str(entry.get("step") or "").strip(), "when": when if when in ("choose", "end") else "order",
                    "messages": max(0, _count(entry.get("messages")) or 0)}
        shaped = (shape(e) for e in _items(brief.get("process")) if isinstance(e, dict))
        self.process.extend(entry for entry in shaped if entry["step"])
        if self.roles.get("ask"):
            # messages the order and ending steps take, one at least for each of the two that needs any
            talk = [p["messages"] for p in self.process if p["messages"] and p["when"] != "choose"]
            self.reserve = min(max(talk), self.msg_limit) if talk else 0

    def read_request(self) -> None:
        prompt = self._rq_prompt()
        # the light model reads the request (room for its thinking); the heavy one only when that reading misses an item or the buy tool
        brief = self.ask_json(prompt, [LIGHT], 4000) or {}
        if self._rq_needs_heavy(brief):
            brief = self.ask_json(prompt, [HEAVY, LIGHT], 2500) or brief
        self.brief = brief
        roles = brief.get("roles")
        self.roles = {k: v for k, v in (roles.items() if isinstance(roles, dict) else ()) if isinstance(v, str) and v in self.schemas}
        for key in ("buy_args", "refuse_args"):
            if not isinstance(brief.get(key), dict):
                brief[key] = {}
        self._rq_split_modes(brief)
        self._rq_items(brief)
        self._rq_lookups(brief)
        self.unmix()
        for key, normalise in (("budget", self._rq_budget), ("messages", self._rq_messages), ("steps", self._rq_steps)):
            normalise(brief.get(key))
        self.plain_loop = self.max_steps >= LONG_STEPS
        self._rq_process(brief)

    def _unmix_item(self, number: int, item: Item) -> None:
        pointers = [need for n, need in self.refs if n == number and need in item.musts]
        rest = [m for m in item.musts if m not in pointers]
        if not pointers or not rest:
            return
        prompt = (UNMIX + "\n\nTHE SHOPPER'S REQUEST:\n" + self.request + f"\n\nTHE PRODUCT: {item.name}\nREQUIREMENTS THAT REFER TO A DIFFERENT LISTING:\n"
                  + "\n".join(pointers) + "\nOTHER REQUIREMENTS:\n" + "\n".join(rest))
        named = {str(x).strip() for x in _items((self.ask_json(prompt, [HEAVY, LIGHT], 500) or {}).get("describe_other"))}
        moved = list(filter(named.__contains__, rest))
        if not moved:
            return
        head = pointers[0]
        joined = f"{head} (that listing: {'; '.join(moved)})"
        item.musts = [joined if m == head else m for m in item.musts if m not in moved]
        self.refs = [(n, joined) if (n, need) == (number, head) else (n, need) for n, need in self.refs]
        self.trace.append({"kind": "unmix", "item": number, "moved": moved})

    def unmix(self) -> None:
        """Requirements misread as the product's own that really describe the referenced listing are folded into that reference."""
        for number, item in enumerate(self.items, 1):
            self._unmix_item(number, item)

    def _lg_items(self) -> list[str]:
        many = len(self.items) > 1
        notes = ((self.ref_values, "(value resolved by following its referenced pages, per its relation: {})"),
                 (self.ref_early, "(partial value from the referenced pages so far, per its relation: {})"))

        def annotate(m: str) -> str:
            return next((f"{m} {form.format(table[m])}" for table, form in notes if m in table), m)

        def describe(number: int, item: Item) -> str:
            needs = "; ".join(annotate(m) for m in item.musts + item.told if m not in item.dropped)
            bits = [f"item {number}: {item.name}" if many else f"item: {item.name}",
                    f" (the store shelves it as: {item.shelf})" if item.shelf else "",
                    f", {item.qty} wanted; must have: ", needs or "nothing past being this kind of product",
                    "; the shopper is fine with any value of: " + ", ".join(item.free) if item.free else ""]
            return "".join(bits)
        return [describe(n, it) for n, it in enumerate(self.items, 1)]

    def _lg_choice(self) -> list[str]:
        brief, out = self.brief, []
        if self.budget is not None:
            scope = {"each": "for each item"}.get(brief.get("budget_scope"), "in total")
            out.append(f"spending limit: no more than {self.budget} {brief.get('budget_unit') or ''} {scope}".replace("  ", " "))
        pick = brief.get("pick")
        out.append("choose: the fitting listing with the lowest price" if pick == "cheapest"
                   else f"choose: among fitting listings, the one with the {pick} {brief.get('pick_measure') or ''}".replace("  ", " ") if pick in ("most", "least")
                   else "choose: any listing that meets everything")
        if brief.get("in_stock_only", True) is not False:
            out.append("only listings in stock")
        out += [f"preference (only to break a tie): {w}" for w in _items(brief.get("wishes"))]
        out += [f"the request sets this rule: {r}" for r in _items(brief.get("rules"))]
        return out

    def _lg_endings(self) -> list[str]:
        brief = self.brief
        out = []
        if self.closest_mode():
            rule = str(brief.get("closest_rule") or "").strip()
            out.append("when no listing satisfies everything: closing with no purchase scores nothing; buy the nearest match for every item, i.e. the listing that fails the fewest requirements (stock and budget included)" + (f" ({rule})" if rule else ""))
        elif brief.get("quiet_decline") is True:
            out.append("when no listing fits: finish without naming or proposing a listing")
        moments = {"choose": "before choosing", "order": "at the order (Kyo does it when you check out)", "end": "at an ending without a purchase (Kyo does it then)"}
        return out + [f"a step the request wants, {moments[e['when']]}: {e['step']}" for e in self.process]

    def _lg_sources(self) -> list[str]:
        facts = self.facts_tool()
        if not facts or not (self.facts_from.get(facts) is True or self.roles.get("offer")):
            return []
        shown = self.page_tools()
        bare = sorted(t for t in shown | {self.roles.get("offer") or ""} if t and self.facts_from.get(t) is False)
        pages = sorted(t for t in shown if t != facts)
        text = f"a listing's attributes are shown by {facts}"
        if bare:
            text += f"; {', '.join(bare)} show none"
        if pages:
            text += f"; to open a listing's own page (needed for a page check or for following a chain of pages) use {' or '.join(pages)}"
        return [text]

    def ledger(self) -> str:
        sections = (self._lg_items, self._lg_choice, self._lg_endings, self._lg_sources)
        return "\n".join("- " + p for build in sections for p in build())

    def _st_nudge(self, left: int) -> str:
        unbought = self.buys == 0
        rules = (
            (lambda: left <= self.tail() + 1,
             lambda: " FINAL STEPS: place the order immediately; for any item lacking a cart line, Kyo adds its nearest-matching listing." if self.closest_mode() else
             " FINAL STEPS: place the order now if every item has a cart line; close without buying only if searching proved some item has no suitable listing (a lack of remaining steps proves nothing)."),
            (lambda: left <= 6 or (left <= len(self.items) + 5 and unbought),
             lambda: " Only a few steps remain: put the chosen lines in the cart and check out right away."),
            (lambda: left <= len(self.items) + 7 and unbought and self.cart_lines and self.lines_missing() == 0,
             lambda: " Each item already has a line in the cart: check out now, so steps remain to react if the store announces a change at checkout."),
            (lambda: unbought and left <= 12,
             lambda: " Remaining steps are thinning: lock in choices, load the cart and place the order within a handful of steps."),
            (lambda: unbought and not self.cart_touched and self.step >= self.max_steps // 2,
             lambda: " You are past the halfway mark with an empty cart: put a suitable listing for each item in it now (stock may still run out)."),
        )
        return next((say() for when, say in rules if when()), "")

    def _st_plan(self, left: int) -> str:
        if self.buys or left <= self.tail() + 6:
            return ""
        questions = max(0, self.msgs_free()) if not self.asked else 0
        talk = f"{questions} go to the interviewer's questions, " if questions else ""
        return (f"\n[step plan] {left} steps left: " + talk
                + f"keep {self.tail() + len(self.items) + 3} to fill the cart, place the order and handle any change announced when ordering; spend the remainder searching for and verifying items.")

    def _st_lookups(self) -> str:
        live = {need for _, need in self.needs_now()}
        pending = [need for _, need in self.refs
                   if need in live and not (need in self.ref_values or need in self.ref_early)]
        if not pending or self.buys:
            return ""
        return "\n[resolve first: visit the listing each one references, then any listing that page references in turn; the final page gives the value, interpreted through the requirement's relation word] " + " | ".join(pending[:3])

    def status(self) -> str:
        left = self.left()
        pieces = [f"[Kyo] step {self.step} of {self.max_steps} used, {left} left.", self._st_nudge(left), self._st_plan(left), self._st_lookups()]
        if self.leads:
            pieces.append("\n[candidates to review] " + ", ".join(self.leads))
        if self.notes:
            pieces.append("\n[changes up to now] " + " | ".join(self.notes[-4:]))
        pieces.append("\n[ledger]\n" + self.ledger())
        return "".join(pieces)

    # ---------- listings ----------
    def ident(self, node: Any, in_list: bool = False, depth: int = 0) -> str | None:
        """Return the listing id of a record via the learned id field, or None when the node is not a listing."""
        if not isinstance(node, dict) or len(node) < 3:
            return None
        key = self.id_key
        if not (key and _id_like(node.get(key), True)):
            may_learn = in_list or depth == 0 or any(isinstance(v, dict) for v in node.values())
            key = _learn_id([node]) if may_learn else None
            if not key or not ("id" in str(key).lower() or node[key].strip() in self.rows):
                return None
        return node[key].strip()

    def learn_rows(self, node: list[Any]) -> bool:
        rows = list(filter(lambda n: isinstance(n, dict) and len(n) >= 3, node))
        if not rows or 2 * len(rows) < len(node):
            return False
        known = self.id_key
        fits_known = bool(known) and all(_id_like(r.get(known), True) for r in rows)
        found = None if fits_known else _learn_id(rows)
        if found:
            self.id_key = found
        return fits_known or bool(found)

    def _ab_record(self, ident: str, node: dict) -> None:
        record = self.rows.get(ident)
        if record is None:
            self.seen_order.append(ident)
            record = self.rows[ident] = {}
        # a flag of the listing's own record flipping true -> false reads as "can no longer be bought"
        went_off = went_on = False
        for key, value in node.items():
            before = record.get(key)
            went_off = went_off or (before is True and value is False)
            went_on = went_on or (before is False and value is True)
        if went_off:
            self.unavailable.add(ident)
        elif went_on:
            self.unavailable.discard(ident)
        record.update(node)
        if _fact_block(node):
            self.full[ident] = dict(record)
        elif ident in self.full:
            self.full[ident].update(node)

    def absorb(self, observation: Any) -> None:
        pending: list[tuple[Any, int, bool]] = [(observation, 0, False)]
        while pending:
            node, depth, in_list = pending.pop()
            if depth > 6:
                continue
            if isinstance(node, list):
                as_rows = self.learn_rows(node)
                pending.extend((v, depth + 1, as_rows) for v in reversed(node))
            elif isinstance(node, dict):
                ident = self.ident(node, in_list, depth)
                if ident:
                    self._ab_record(ident, node)
                else:
                    pending.extend((v, depth + 1, False) for v in reversed(list(node.values())))

    def labels(self) -> set[str]:
        found: set[str] = set()
        pending = [(schema, 0) for schema in self.schemas.values()]
        while pending:
            schema, depth = pending.pop()
            if depth > 6 or not isinstance(schema, dict):
                continue
            found.update(option.strip().lower() for option in schema.get("enum") or [] if isinstance(option, str))
            pending.extend((value, depth + 1) for value in schema.values() if isinstance(value, dict))
        return found

    def category_key(self) -> str | None:
        counts: dict[str, set[str]] = {}
        present: dict[str, int] = {}
        short = lambda v: isinstance(v, str) and 0 < len(v) <= 40 and not re.search(r"\d{3}", v)
        for key, value in ((k, v) for row in self.rows.values() for k, v in row.items() if short(v)):
            counts.setdefault(key, set()).add(value)
            present[key] = present.get(key, 0) + 1
        floor = 0.8 * max(1, len(self.rows))
        known = self.labels()

        def rank(key: str) -> tuple[float, int, int, float]:
            values = counts[key]
            listed = sum(v.strip().lower() in known for v in values) / len(values)
            return (-listed, len(values), -present[key], -(sum(map(len, values)) / len(values)))

        return min((k for k in counts if present[k] >= floor and len(counts[k]) <= 40), key=rank, default=None)

    def _wanted_test(self):
        """A test for a category label: does it name (or share a long word with) a requested item's shelf or name."""
        wanted = [(_words(item.shelf or ""), _words(item.name), {w for w in _stem(item.shelf or "") | _stem(item.name) if len(w) >= 4}) for item in self.items]

        def hit(label: str) -> bool:
            word = _words(label)
            long_words = {w for w in _stem(label) if len(w) >= 4}
            return any(word in entry[:2] or long_words & entry[2] for entry in wanted)

        return hit

    def _usable_keys(self, keys: dict[str, dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
        def plain(values: list[str]) -> bool:
            mean = sum(map(len, values)) / len(values)
            if mean > 40 or sum(" / " in v for v in values) * 2 >= len(values):
                return False
            return not (len(values) >= 4 and len(set(values)) == len(values) and mean > 12)

        kept: list[tuple[str, dict[str, Any]]] = []
        for name, slot in sorted(keys.items(), key=lambda kv: -kv[1]["n"]):
            values = list(slot["per"].values())
            if not values or not plain(values):
                continue
            if any(name.startswith(other + " ") and slot["per"] == spec["per"] for other, spec in kept):
                continue
            kept.append((name, slot))
        return kept

    def key_table(self) -> list[tuple[str, str, int, list[str]]]:
        """(category, key name, listings carrying it, sample values), most carried first per category."""
        cat_key = self.category_key()
        groups: dict[str, dict[str, Any]] = {}
        for ident, record in ((i, self.full.get(i)) for i in self.seen_order):
            if not record:
                continue
            label = str(record.get(cat_key) or "listings") if cat_key else "listings"
            group = groups.setdefault(label, {"n": 0, "keys": {}})
            group["n"] += 1
            firsts: dict[str, Any] = {}
            for key, value in (_fact_block(record) or {}).items():
                firsts.setdefault(_words(key), value)
            firsts.pop("", None)
            for name, value in firsts.items():
                slot = group["keys"].setdefault(name, {"n": 0, "values": [], "per": {}})
                slot["n"] += 1
                text = slot["per"][ident] = str(value).strip().lower()
                if text not in slot["values"]:
                    slot["values"].append(text)
        hit = self._wanted_test()
        groups = {label: group for label, group in groups.items() if hit(label)} or groups
        table: list[tuple[str, str, int, list[str]]] = []
        for label, group in sorted(groups.items(), key=lambda kv: -kv[1]["n"]):
            table.extend((label, name, slot["n"], [v[:40] for v in slot["values"][:4]]) for name, slot in self._usable_keys(group["keys"])[:14])
        return table

    def item_groups(self) -> list[list[str]]:
        """Listing ids per category, keeping the categories that match a requested item (largest first)."""
        cat_key = self.category_key()
        label_of = (lambda ident: str(self.rows[ident].get(cat_key) or "")) if cat_key else (lambda ident: "")
        groups: dict[str, list[str]] = {}
        for ident in self.seen_order:
            groups.setdefault(label_of(ident), []).append(ident)
        hit = self._wanted_test()
        pool = [ids for label, ids in groups.items() if hit(label)] or list(groups.values())
        return sorted(pool, key=len, reverse=True)[: max(1, len(self.items))]

    def single_target(self, tool: str | None) -> str | None:
        if not tool:
            return None
        schema = self.schemas.get(tool) or {}
        required = schema.get("required") or []
        texts = [k for k, v in (schema.get("properties") or {}).items() if isinstance(v, dict) and "string" in _kinds(v) and "enum" not in v]
        target = next((k for k in required if k in texts), None) or (texts[0] if len(texts) == 1 else None)
        return target if target and all(k == target for k in required) else None

    def price_of(self, ident: str) -> float | None:
        """The listing's price as its newest record shows it, under the price key the model named for result rows."""
        if not self.price_key:
            return None
        number = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool)
        value = (self.rows.get(ident) or {}).get(self.price_key)
        if isinstance(value, dict):
            value = next(filter(number, value.values()), None)
        elif isinstance(value, str) and re.fullmatch(r"\s*\d+(\.\d+)?\s*", value):
            return float(value)
        return float(value) if number(value) else None

    def buyable(self, ident: str) -> bool:
        """Not sold out by any record: no note or refusal named it, and its stock flag (the key the model named) is not no."""
        if self.sold_out(ident):
            return False
        if not self.stock_key:
            return True
        value = (self.rows.get(ident) or {}).get(self.stock_key)
        if value is False:
            return False
        return isinstance(value, bool) or not isinstance(value, (int, float)) or not value <= 0

    def records_in(self, observation: Any) -> list[tuple[str, dict[str, Any]]]:
        """The listing records a result carries: the result itself, or the rows of its lists (two levels deep)."""
        def listed(rows: list[Any]) -> list[tuple[str, dict[str, Any]]]:
            out: list[tuple[str, dict[str, Any]]] = []
            for row in filter(lambda r: isinstance(r, dict), rows):
                found_id = self.ident(row, True)
                if found_id:
                    out.append((found_id, row))
            return out

        if isinstance(observation, list):
            return listed(observation)
        if not isinstance(observation, dict):
            return []
        own = self.ident(observation)
        if own:
            return [(own, observation)]
        found: list[tuple[str, dict[str, Any]]] = []
        for value in observation.values():
            if isinstance(value, list):
                found.extend(listed(value))
                continue
            if not isinstance(value, dict):
                continue
            single = self.ident(value)
            found.extend([(single, value)] if single else [])
            for inner in (v for v in value.values() if isinstance(v, list)):
                found.extend(listed(inner))
        return found

    def note_view(self, name: str, observation: Any) -> None:
        """Fields each viewing tool shows, and the fields a listing has shown since the last change (a view that
        would show nothing new is skipped)."""
        pairs = [(ident, set(record)) for ident, record in self.records_in(observation)]
        if not pairs:
            return
        self.tool_keys.setdefault(name, set()).update(*(keys for _, keys in pairs))
        for ident, keys in pairs:
            version, have = self.view_keys.get(ident, (-1, set()))
            self.view_keys[ident] = (self.version, keys | have if version == self.version else keys)

    def shown_already(self, name: str, ident: str) -> bool:
        keys = self.tool_keys.get(name)
        if not keys or ident in self.gone:
            return False
        version, have = self.view_keys.get(ident, (-1, set()))
        return version == self.version and keys <= have

    # ---------- result pages ----------
    def unwrapped(self, observation: Any) -> Any:
        """A result with notes beside it: the part that holds the rows."""
        if not isinstance(observation, dict):
            return observation
        if any(isinstance(v, list) and v and all(isinstance(r, dict) for r in v) for v in observation.values()):
            return observation
        holders = [v for v in observation.values() if isinstance(v, dict) and any(isinstance(x, list) for x in v.values())]
        return holders[0] if len(holders) == 1 else observation

    def read_shape(self, name: str, observation: Any) -> dict[str, Any] | None:
        """Which keys of a tool's row-list result hold the rows, the match count, the next-page value (and the parameter
        taking it), a row's price and its can-be-bought flag: named once per tool by the light model, checked against the result."""
        if name in self.shapes:
            return self.shapes[name]
        observation = self.unwrapped(observation)
        if not isinstance(observation, dict):
            return None
        lists = [k for k, v in observation.items() if isinstance(v, list) and len(v) >= 2 and all(isinstance(r, dict) for r in v)]
        if not lists:
            return None
        self.shapes[name] = None
        schema = self.schemas.get(name) or {}
        sample = {k: v[:3] if k in lists else v for k, v in observation.items()}
        prompt = "".join((SHAPE, "\n\nTOOL: ", name, " | parameters: ", _dump(schema),
                          "\n\nRESULT (first rows only):\n", _dump(_clip(sample, 120))[:4000]))
        got = self.ask_json(prompt, [LIGHT, HEAVY], 1500) or {}
        rows = got.get("rows")
        if rows not in lists or not rows:
            return None
        first = observation[rows][0]
        homes = {"count": observation, "next": observation, "next_arg": schema.get("properties") or {}, "price": first, "stock": first}
        shape: dict[str, Any] = {"rows": rows}
        for field, home in homes.items():
            key = got.get(field)
            shape[field] = key if isinstance(key, str) and key in home else None
        if shape["count"]:
            tally = observation[shape["count"]]
            if isinstance(tally, bool) or not isinstance(tally, int):
                shape["count"] = None
        self.shapes[name] = shape
        self.price_key = self.price_key or shape["price"]
        self.stock_key = self.stock_key or shape["stock"]
        self.trace.append({"kind": "shape", "tool": name, "shape": shape})
        return shape

    def paging(self, name: str, observation: Any) -> tuple[int, int, str, Any] | None:
        """(matches reported, rows shown, the parameter asking for the next page, its value or None on the last page)
        by the tool's named result shape; None when the result lacks those parts."""
        shape = self.read_shape(name, observation)
        observation = self.unwrapped(observation)
        if not (shape and all(shape[k] for k in ("count", "next", "next_arg"))) or not isinstance(observation, dict):
            return None
        rows, count, value = (observation.get(shape[k]) for k in ("rows", "count", "next"))
        whole = lambda x: isinstance(x, int) and not isinstance(x, bool)
        if not (isinstance(rows, list) and rows and whole(count)):
            return None
        if value is None or (isinstance(value, (str, int)) and not isinstance(value, bool)):
            return count, len(rows), shape["next_arg"], value
        return None

    def price_sorted(self, observation: Any) -> bool:
        prices = [self.price_of(i) for i, _ in self.records_in(observation)]
        if len(prices) < 3 or None in prices:
            return False
        return all(map(float.__le__, prices, prices[1:]))

    @staticmethod
    def _paging_itself(given: Any) -> bool:
        """True when the call already asked for a later page (a page number past the first, or any cursor text)."""
        if isinstance(given, int) and not isinstance(given, bool):
            return given not in (None, 0, 1)
        return bool(str(given or "").strip())

    def _read_pages(self, name: str, args: dict[str, Any], param: str, cursor: Any, budget: int):
        """Walk the remaining pages of one search, yielding (cursor, shown text) for each page read."""
        while budget > 0 and cursor is not None:
            budget -= 1
            faults: list[str] = []
            call = _shape({**args, param: cursor}, self.schemas.get(name), "args", faults)
            if faults:
                return
            reply = self.act(name, call)
            if reply is None or self.finished:
                return
            self.pages_read += 1
            seen = reply.get("observation")
            self.observe(name, seen)
            self.paged_obs.append(seen)
            shown_text = self.show(seen, ROW_CHARS)
            self.read_cache[name + _dump(call)] = (self.version, self.step, shown_text)
            yield cursor, shown_text
            nxt = self.paging(name, seen)
            cursor = nxt[3] if nxt else None

    def more_pages(self, name: str, args: dict[str, Any], observation: Any) -> str:
        """When a cheapest/most/least pick meets a search with hidden matches, a few remaining pages are fetched here
        (one step each, no model turn); a long tail is only reported, so the agent can tighten the search."""
        self.paged_obs = []
        pick = self.brief.get("pick")
        if pick not in ("cheapest", "most", "least") or not (plan := self.paging(name, observation)):
            return ""
        count, shown, param, value = plan
        if value is None or self._paging_itself(args.get(param)) or count <= shown:
            return ""
        if pick == "cheapest" and self.price_sorted(observation):
            return ""
        pages = -(-(count - shown) // shown)
        too_long = (lambda: pages > 2, lambda: self.pages_read + pages > AUTO_PAGES,
                    lambda: self.left() - self.tail() - pages < 12)
        if any(limit() for limit in too_long):
            return (f"\n[Kyo] {count} listings matched this search but only {shown} are shown, unsorted by price. Every fitting listing counts toward the pick: "
                    f"tighten the search (use the requirement's value as the words) or fetch the next page with {param}={_dump(value)}.")
        texts = [f"{param}={_dump(at)}: {body}" for at, body in self._read_pages(name, args, param, value, pages)]
        self.trace.append({"kind": "pages", "tool": name, "read": len(texts)})
        if not texts:
            return ""
        return "\n[Kyo fetched the remaining pages of this search on its own, one step per page]\n" + "\n".join(texts)

    def cheapest_here(self, observations: list[Any]) -> str:
        """Cheapest pick: buyable rows of these results (and pages read on), listed in price order."""
        if self.brief.get("pick") != "cheapest" or not self.price_key:
            return ""
        found = {i for seen in observations for i, _ in self.records_in(seen)}
        priced = sorted(filter(lambda pair: pair[0] is not None, ((self.price_of(i), i) for i in found if self.buyable(i))))
        if len(priced) < 2:
            return ""
        return "\n[Kyo] buyable rows here, cheapest first: " + ", ".join(f"{i} ({p:g})" for p, i in priced[:6])

    def wider_hold(self) -> str:
        """One-time pause before a cheapest checkout with steps to spare: title-only keyword search misses fitting
        listings named in other words."""
        if self.wider_held:
            return ""
        if self.brief.get("pick") != "cheapest":
            return ""
        if self.left() < 14 + 2 * len(self.items):
            return ""
        self.wider_held = True
        listed = "; ".join(f"{i} at {p:g}" for i, p in ((i, self.price_of(i)) for i in self.cart_ids()) if p is not None)
        cart_line = f" ({listed})" if listed else ""
        return (f"Not run yet (no step used). {self.left()} steps remain and the pick is the cheapest. Store search looks at titles only,"
                " so searching for a requirement's value (which often sits only in the attributes) surfaces few fitting listings, unsorted by price."
                " Before you check out, search again for each item using other names for the product (a plain one-word name, synonyms,"
                " words taken from fitting titles already seen) and consider only buyable rows cheaper than the cart's line"
                + cart_line + "; inspect the lowest-priced of them closely, replace a cart line with any that qualifies, and only then place the order.")

    def label_hint(self, args: dict[str, Any]) -> str:
        """Empty search result: list the category labels rows have shown (a filter matches only that spelling), unless
        the search already used one."""
        cat_key = self.category_key()
        if not cat_key:
            return ""
        label_of = lambda ident: (self.rows.get(ident) or {}).get(cat_key)
        counts: dict[str, int] = {}
        for label in filter(lambda lab: isinstance(lab, str) and lab.strip(), map(label_of, self.seen_order)):
            counts[label] = counts.get(label, 0) + 1
        if not counts:
            return ""
        if any(isinstance(v, str) and v in counts for v in args.values()):
            return ""
        own = (label_of(ids[0]) if ids else None for ids in self.item_groups())
        order = list(dict.fromkeys(lab for lab in own if isinstance(lab, str) and lab in counts))
        order.extend(lab for lab, _ in sorted(counts.items(), key=lambda kv: -kv[1]) if lab not in order)
        return ("\n[Kyo] Nothing came back. Filters match only values spelled exactly as result rows spell them; category labels seen on rows so far: "
                + ", ".join(map(repr, order[:8])) + ".")

    def row_keys_of(self, value: str) -> set[str]:
        """Record keys whose text, in any listing seen so far, equals this value (case and spacing aside)."""
        squash = lambda text: " ".join(text.lower().split())
        want = squash(value)
        found: set[str] = set()
        for row in self.rows.values():
            found.update(k for k, v in row.items() if isinstance(v, str) and squash(v) == want)
        return found

    def learn_blind(self) -> None:
        """An empty-result filter value that listings do show (under some field) marks that parameter blind to the field."""
        hits = ((entry, self.row_keys_of(entry[2])) for entry in list(self.empty_filters))
        for entry, keys in filter(lambda pair: pair[1], hits):
            self.blind[(entry[0], entry[1])] = min(keys)
            self.empty_filters.remove(entry)

    def blind_call(self, name: str, args: dict[str, Any]) -> str:
        def blocked(param: str, value: Any) -> str:
            key = self.blind.get((name, param))
            if not (key and isinstance(value, str) and value.strip()):
                return ""
            keys = self.row_keys_of(value)
            return key if not keys or keys == {key} else ""
        hit = next(((p, k) for p, v in args.items() for k in (blocked(p, v),) if k), None)
        if hit is None:
            return ""
        param, key = hit
        return (f"Not run (no step used): {param} came back empty before for a value listings carry as their {key!r}, so that field is invisible"
                f" to this parameter. Use plain keywords to reach such a listing instead: its {key}, what kind of product it is, and any telling values.")

    def _open_view(self, tool: str, raw: dict[str, Any], schema: Any) -> tuple[bool, Any]:
        """Shape and run one view call, absorbing and noting what it shows; (False, None) when it could not run."""
        faults: list[str] = []
        call = _shape(raw, schema, "args", faults)
        if faults:
            return False, None
        reply = self.act(tool, call)
        if reply is None:
            return False, None
        seen = reply.get("observation")
        self.observe(tool, seen)
        self.note_view(tool, seen)
        return True, seen

    def _fill_joined(self, todo: list[str], many: str, joiner: str, keep: int) -> list[str]:
        room = self.compare_room()
        texts: list[str] = []
        for chunk in (todo[s:s + room] for s in range(0, len(todo), room)):
            if self.left() <= keep or self.finished:
                break
            ok, seen = self._open_view(joiner, {many: chunk}, self.schemas.get(joiner))
            if not ok:
                break
            texts.append(", ".join(chunk) + ": " + self.show(seen)[:4000])
        if texts:
            self.trace.append({"kind": "fill_facts", "n": len(todo), "joined": True})
        return texts

    def _fill_single(self, todo: list[str], keep: int) -> list[str]:
        tool = self.facts_tool()
        target = self.single_target(tool)
        if not tool or not target:
            return []
        texts: list[str] = []
        pending = iter(todo)
        while True:
            ident = next(pending, None)
            if ident is None or self.left() <= keep or self.finished or self.facts_from.get(tool) is False:
                break
            ok, seen = self._open_view(tool, {target: ident}, self.schemas.get(tool))
            if not ok:
                break
            if isinstance(seen, dict) and self.ident(seen):
                self.facts_from[tool] = self.facts_from.get(tool, False) or bool(_fact_block(seen))
            texts.append(f"{ident}: " + self.show(seen)[:1500])
        if texts:
            self.trace.append({"kind": "fill_facts", "n": len(texts)})
        return texts

    def fill_facts(self, ids: list[str], keep: int) -> list[str]:
        """Read unseen attributes of listings about to be judged, leaving `keep` steps: several per step in a
        side-by-side view that shows attributes, else one listing per step."""
        todo = [i for i in ids if i in self.rows and i not in self.full and not self.sold_out(i)]
        many, joiner = self.compare_target(), self.roles.get("compare")
        if len(todo) >= 2 and many and joiner and self.compare_facts() is True:
            return self._fill_joined(todo, many, joiner, keep)
        return self._fill_single(todo, keep)

    def cheaper_rows(self) -> list[str]:
        """Cheapest checkout: unread buyable rows of a cart line's kind priced under it are read (a few, cheapest
        first) for the final check to weigh."""
        if self.brief.get("pick") != "cheapest":
            return []
        mark, cat_key = self.cart_mark() + f"#{self.version}", self.category_key()
        if not cat_key or mark in self.cheaper_marks:
            return []
        self.cheaper_marks.add(mark)
        cart = self.cart_ids()
        # a side-by-side view with attributes reads several in one step
        if self.compare_target() and self.roles.get("compare") and self.compare_facts() is True:
            per, limit = 3, self.compare_room()
        else:
            per, limit = 2, CHEAPER_OPENS
        kind_of = lambda ident: str((self.rows.get(ident) or {}).get(cat_key) or "").strip().lower()
        sort_price = lambda ident: self.price_of(ident) if self.price_of(ident) is not None else float("inf")

        def under(line: str) -> list[str]:
            price, kind = self.price_of(line), kind_of(line)
            if price is None or not kind:
                return []
            tests = (lambda i: i not in cart, lambda i: i not in self.full, self.buyable,
                     lambda i: kind_of(i) == kind, lambda i: sort_price(i) < price)
            pool = [i for i in self.seen_order if all(test(i) for test in tests)]
            return sorted(pool, key=lambda i: self.price_of(i) or 0.0)[:per]

        picks: list[str] = []
        for fresh in map(under, cart):
            picks += [i for i in fresh if i not in picks]
        return self.fill_facts(picks[:limit], self.tail() + 4)

    def observe(self, name: str, observation: Any) -> str:
        """Absorb what one of Kyo's own calls returned and handle any shopper note inside; the result is the note's effect."""
        self.absorb(observation)
        found = self.loose_lines(observation)
        if not found:
            return ""
        meaning = self.take_note(found)
        self.queued.extend([f"(seen on {name}) {meaning}" + self.after_change()] if meaning else [])
        return meaning

    @staticmethod
    def _interleave(lanes: list[list[str]]) -> list[str]:
        """Round-robin over the lanes: every lane's first entry, then every lane's second, and so on."""
        out: list[str] = []
        depth = 0
        while True:
            tier = [lane[depth] for lane in lanes if depth < len(lane)]
            if not tier:
                return out
            out += tier
            depth += 1

    def widen(self) -> None:
        """Compare extra listings in one side-by-side step, so the attribute-key tally is not drawn from just two or three."""
        tool = self.roles.get("compare")
        schema = self.schemas.get(tool or "") or {}
        props = schema.get("properties") or {}
        arrays = [k for k in props if isinstance(props[k], dict) and "array" in _kinds(props[k])]
        target = arrays[0] if arrays else None
        usable = bool(tool and target) and set(schema.get("required") or []) <= {target}
        # without attributes on the side-by-side view the planner sees no keys: read attributes singly instead
        if not usable or self.compare_bare():
            self.widen_singly()
            return
        if self.left() <= 12:
            return
        room = self.compare_room()
        lanes = [[i for i in ids if i not in self.full][: max(0, WIDEN_EACH - sum(i in self.full for i in ids))]
                 for ids in self.item_groups()]
        queue = self._interleave(lanes)
        cap = min(len(queue), room * max(1, len(lanes)))
        start = 0
        while start < cap:
            picks, start = queue[start:start + room], start + room
            if len(picks) < 2 or self.left() <= 12:
                return
            outcome = self._widen_batch(tool, target, schema, picks)
            if outcome == "single":
                # rows but no attributes in this view: fall back to single reads
                self.widen_singly()
            if outcome != "more":
                return

    def _widen_batch(self, tool: str, target: str, schema: dict[str, Any], picks: list[str]) -> str:
        """One side-by-side call for widen: "more" to go on, "single" to switch to single reads, "stop" otherwise."""
        before = len(self.full)
        ok, seen = self._open_view(tool, {target: picks}, schema)
        if not ok:
            return "stop"
        rows = self.records_in(seen)
        record = {"kind": "widen", "ids": picks, "opened": len(self.full) - before}
        if rows:
            self.facts_from[tool] = self.facts_from.get(tool, False) or any(_fact_block(r) for _, r in rows)
            self.compare_ok = max(self.compare_ok, len(rows))
        else:
            self.compare_bad = min(self.compare_bad or 999, len(picks))
        self.trace.append(record)
        return "single" if rows and self.facts_from.get(tool) is False else "more"

    def widen_singly(self) -> None:
        """Read a few more listings' attributes one at a time (no side-by-side view with attributes)."""
        tool = self.facts_tool()
        target = self.single_target(tool)
        if not tool or not target:
            return
        by_price = lambda i: self.price_of(i) if self.price_of(i) is not None else float("inf")

        def lane(ids: list[str]) -> list[str]:
            room = max(0, min(2, 8 - sum(i in self.full for i in ids)))
            pool = [i for i in ids if i not in self.full and self.buyable(i)]
            if self.brief.get("pick") == "cheapest":
                pool = sorted(pool, key=by_price)
            return pool[:room]

        # up to two per item group, taken in turn; 22 steps stay for the rest of the episode
        self.fill_facts(self._interleave([lane(ids) for ids in self.item_groups()])[:5], 22)

    def ready_to_ask(self) -> bool:
        if self.asked or not self.roles.get("ask"):
            return False
        if self.msgs_free() <= 0 or self.left() <= 8:
            return False
        browsed_enough = len(self.full) >= 3 * len(self.items) + 1
        return browsed_enough or self.step >= max(8, self.max_steps // 4)

    def message_args(self, text: str) -> dict[str, Any] | None:
        schema = self.schemas.get(self.roles.get("ask") or "") or {}
        props = schema.get("properties") or {}
        required = schema.get("required") or []
        free_text = [key for key, spec in props.items() if isinstance(spec, dict) and "string" in _kinds(spec) and "enum" not in spec]
        slot = next(filter(free_text.__contains__, required), None) or (free_text[0] if free_text else None)
        if not slot:
            return None
        filled: dict[str, Any] = {slot: text}
        for key in (k for k in required if k not in filled):
            choices = (props.get(key) or {}).get("enum")
            if not (isinstance(choices, list) and choices):
                return None
            filled[key] = choices[0]
        faults: list[str] = []
        shaped = _shape(filled, schema, "args", faults)
        return shaped if not faults else None

    def _item_no(self, raw: Any) -> int:
        """1-based item number from a model field, clamped to the items there are."""
        return min(max(1, _count(raw) or 1), len(self.items))

    def _plan_prompt(self, count: int, table: list[tuple[str, str, int, list[str]]]) -> str:
        settled = "\n".join(f"item {n}: {it.name} - already fixed by the request: " + ("; ".join(it.musts) or "none")
                            for n, it in enumerate(self.items, 1))
        rows: list[str] = []
        before = [None] + [row[0] for row in table[:-1]]
        for idx, ((label, name, carried, values), previous) in enumerate(zip(table, before), 1):
            if label != previous:
                rows.append(f"group {label!r}:")
            rows.append(f"  {idx}. {name} (on {carried}): " + " | ".join(values))
        # one pick more than the messages allow, so a pick cut as naming two properties leaves enough
        parts = [PLAN.replace("COUNT", str(count + 1)), "\n\nSHOPPER'S REQUEST:\n", self.request, "\n\nITEMS TO BUY:\n", settled]
        first = [str(x) for x in _items(self.brief.get("ask_first")) if str(x).strip()]
        if first:
            parts += ["\n\nASK FIRST (the request wants these checked with the shopper):\n", "\n".join(first)]
        parts += ["\n\nPROPERTY KEYS ON LISTINGS OPENED SO FAR:\n", "\n".join(rows) or "(no listing opened in full yet)"]
        return "".join(parts)

    def _select_picks(self, plan: dict[str, Any]) -> list[tuple[int, str]]:
        chosen: list[tuple[int, str]] = []
        for pick in filter(lambda p: isinstance(p, dict), _items(plan.get("ask"))):
            number = self._item_no(pick.get("item") or 1)
            # one property per question: a label naming two is cut to the first
            label = re.split(r"\s+(?:and|or)\s+|[/,;(]", str(pick.get("label") or ""))[0].strip().rstrip("?.")
            if not label or len(label.split()) > 5:
                continue
            if any(n == number and _words(label) == _words(other) for n, other in chosen):
                continue
            chosen.append((number, label))
        return chosen

    def _interview_summary(self, sent: list[tuple[int, str, str]], told: list[str]) -> str:
        exchange = "\n".join(f"Q: {q}\nA: {a}" for _, q, a in sent)
        out = ("Kyo's interviewer put these questions to the shopper (one step each):\n" + exchange
               + "\nFirm requirements these answers add: " + ("; ".join(told) or "none")
               + ". Every pick must meet them (run new searches with the new values where needed).")
        pointed = [need for _, need in self.refs if any(need in t for t in told)]
        if pointed:
            followed = self.follow()
            out += ("\n" + followed) if followed else ""
            pointed = [need for need in pointed if need in self.unresolved()]
        if pointed:
            out += (" Their value comes from another listing: " + " | ".join(pointed) + ". Locate that listing, open its full page,"
                    " then open each listing that page points to; the final page holds the value, and the order only counts once every one of them is opened.")
        if self.queued:
            out += "\n[store notes that came in meanwhile] " + " | ".join(self.queued)
            self.queued = []
        return out

    def interview(self) -> str:
        self.asked = True
        count = self.msgs_free()
        if count <= 0:
            return ""
        self.widen()
        # plan
        plan = self.ask_json(self._plan_prompt(count, self.key_table()), [HEAVY, LIGHT], 1800) or {}
        # select
        picks = self._select_picks(plan)
        self.trace.append({"kind": "plan", "ask": _clip(plan.get("ask"), 120)})
        # send: every message goes to a different pick (the same property asked twice only gets the same answer)
        sent = self.send([(n, label, self.question(n, label)) for n, label in picks[:count]])
        if not sent:
            return ""
        # hear + apply
        told: list[str] = []
        self.take_heard(self.hear(sent), told)
        self.version += 1 if told else 0
        return self._interview_summary(sent, told)

    def question(self, number: int, label: str) -> str:
        return f"What {label} would you like for the {self.items[number - 1].name}?"

    def _deliver(self, tool: str, number: int, text: str) -> tuple[tuple[int, str, str], bool] | None:
        """One message to the shopper; None when it could not go out, else (record, whether the reply was an error)."""
        args = self.message_args(text)
        reply = self.act(tool, args) if args else None
        if reply is None:
            return None
        self.msg_used += 1
        answer = reply.get("observation")
        words = self.spoken(answer)
        self.done_log.append(f"step {self.step}: asked the shopper {text!r} -> answer {words[:300]!r} (cart at that point: {self.cart_mark()})")
        self.done_steps.append(self.step)
        self.trace.append({"kind": "ask", "q": text, "a": words[:300]})
        return (number, text, words), bool(isinstance(answer, dict) and answer.get("error"))

    def send(self, questions: list[tuple[int, str, str]], floor: int = 6) -> list[tuple[int, str, str]]:
        tool = self.roles["ask"]
        sent: list[tuple[int, str, str]] = []
        pending = iter(questions)
        failed = False
        while not failed:
            nxt = next(pending, None)
            if nxt is None or self.finished or self.left() <= floor or self.msg_used >= self.msg_limit:
                break
            outcome = self._deliver(tool, nxt[0], nxt[2])
            if outcome is None:
                break
            record, failed = outcome
            sent.append(record)
        return sent

    def _heard_need(self, entry: dict[str, Any], told: list[str], many: bool) -> None:
        need = str(entry.get("need") or "").strip()
        if not need:
            return
        number = self._item_no(entry.get("item"))
        known = self.items[number - 1].told
        if need not in known:
            known.append(need)
            told.append(f"item {number}: {need}" if many else need)
        if entry.get("points") is True and (number, need) not in self.refs:
            self.refs.append((number, need))

    def _heard_drop(self, entry: dict[str, Any], told: list[str], many: bool) -> None:
        if not str(entry.get("property") or "").strip():
            return
        item = self.items[self._item_no(entry.get("item")) - 1]
        gone = _words(str(entry["property"]).split(":")[0])
        hit = [m for m in item.musts + item.told if m not in item.dropped and _words(m.split(":")[0]) == gone]
        item.dropped.extend(hit)
        told.extend(f"dropped {m}" for m in hit)

    def _heard_free(self, entry: dict[str, Any], told: list[str], many: bool) -> None:
        if entry.get("property"):
            self.items[self._item_no(entry.get("item")) - 1].free.append(str(entry["property"]))

    def take_heard(self, heard: dict[str, Any], told: list[str]) -> None:
        """Record what the answers stated; `told` gathers the new requirements for the summary."""
        many = len(self.items) > 1
        for key, handle in (("needs", self._heard_need), ("drop", self._heard_drop), ("free", self._heard_free)):
            for entry in _items(heard.get(key)):
                if isinstance(entry, dict):
                    handle(entry, told, many)
        budget = heard.get("budget")
        if isinstance(budget, (int, float)) and not isinstance(budget, bool) and budget > 0:
            self.budget = budget

    def hear(self, sent: list[tuple[int, str, str]], models: tuple[str, ...] = (HEAVY, LIGHT)) -> dict[str, Any]:
        roster = "\n".join(f"item {n}: {it.name}" for n, it in enumerate(self.items, 1))
        exchange = "\n".join(f"{i}. [item {n}] Q: {q}\nA: {a}" for i, (n, q, a) in enumerate(sent, 1))
        return self.ask_json(HEAR + "\n\nITEMS BEING BOUGHT:\n" + roster + "\n\nQUESTIONS PUT TO THE SHOPPER, WITH ANSWERS:\n" + exchange, list(models), 1500) or {}

    def _gate_prompt(self, stage: str, steps: list[dict[str, Any]], tools: list[str]) -> str:
        record = "\n".join(f"{i}. {line}" for i, line in enumerate(self.done_log, 1)) or "(empty so far)"
        notes = ("\n\nRECENT STORE NOTES:\n" + "\n".join(self.notes[-4:])) if self.notes else ""
        return "".join([
            GATE.replace("STAGE", "check out" if stage == "order" else "finish without buying anything"),
            "\n\nSHOPPER'S REQUEST:\n", self.request,
            "\n\nSTEPS THE REQUEST WANTS AT THIS POINT:\n", "\n".join(f"- {p['step']}" for p in steps),
            "\n\nREQUIREMENTS LEDGER:\n", self.ledger(),
            "\n\nCURRENT CART: ", self.cart_text(),
            "\n\nRECORD (shopper messages and step calls made so far):\n", record,
            notes,
            f"\n\nMESSAGES REMAINING: {max(0, self.msg_limit - self.msg_used)}   STEPS REMAINING: {self.left()} (one of them goes to the {'order' if stage == 'order' else 'ending'} itself)",
            "\n\nTOOLS:\n", "\n".join(tools)])

    def _gate_unproven(self, stage: str, steps: list[dict[str, Any]], got: dict[str, Any]) -> list[dict[str, Any]]:
        """Steps the model lists that no record line proves (at the order, a talking step needs a line after the last cart change)."""
        talk = stage == "order" and any(p["messages"] > 0 for p in steps)
        proof = {i for i in range(1, len(self.done_log) + 1)
                 if not talk or (i <= len(self.done_steps) and self.done_steps[i - 1] > self.cart_at)}

        def proven(e: dict[str, Any]) -> bool:
            return e.get("done") is True and any((_count(n) or 0) in proof for n in _items(e.get("lines")))
        return [e for e in _items(got.get("steps")) if isinstance(e, dict) and not proven(e)]

    def _gate_decide(self, stage: str, steps: list[dict[str, Any]], tools: list[str], said: list[str], nudged: bool) -> tuple[str, Any]:
        """One model turn of the gate -> (verdict, payload): stop | changed | nudge | clear | call."""
        # the light model decides; after a nudge, the heavy one
        got = self.ask_json(self._gate_prompt(stage, steps, tools), [HEAVY, LIGHT] if nudged else [LIGHT, HEAVY], 2000)
        if got is None:
            return "stop", None
        told: list[str] = []
        hold = ""
        if said:
            # only a reply to the gate's own call is read here (earlier answers are the interviewer's)
            self.take_heard({"needs": got.get("needs")}, told)
            hold = str(got.get("hold") or "").strip()
        call = got.get("call") if isinstance(got.get("call"), dict) else None
        self.trace.append({"kind": "gate", "stage": stage, "call": _clip(call, 200), "steps": _clip(got.get("steps"), 120), "hold": hold[:200], "told": told})
        if told or hold:
            return "changed", told + [hold] * bool(hold)
        if call:
            return "call", call
        unproven = self._gate_unproven(stage, steps, got)
        if unproven and not said and not nudged and self.roles.get("ask") and self.msg_used < self.msg_limit and self.left() >= 2:
            return "nudge", unproven
        return "clear", None

    def _gate_nudge(self, unproven: list[dict[str, Any]]) -> None:
        self.trace.append({"kind": "gate_unproven", "steps": _clip(unproven, 120)})
        missing = "; ".join(str(e.get("step") or "") for e in unproven)[:200]
        self.done_log.append(f"(Kyo: nothing in the record since the cart last changed at step {self.cart_at} proves " + missing
                             + f" done; {self.msg_limit - self.msg_used} message(s) and {self.left()} steps remain, so give the call that does it)")
        self.done_steps.append(-1)

    def _gate_act(self, call: dict[str, Any], ending: set[str], said: list[str]) -> bool:
        """Carry out the model's store call; False ends the gate."""
        name = str(call.get("tool") or "")
        ask = self.roles.get("ask")
        if name not in self.schemas or name in ending:
            return False
        if self.left() < 2 or (name == ask and self.msg_used >= self.msg_limit):
            return False
        faults: list[str] = []
        raw = call.get("args")
        args = _shape(raw if isinstance(raw, dict) else {}, self.schemas[name], "args", faults)
        if faults:
            return False
        if name == ask:
            text = next((v for v in args.values() if isinstance(v, str)), "")
            batch = self.send([(1, "", text)], 1)
            if batch:
                self.gate_msgs += 1
                said.append(f"Q: {text}\nA: {batch[0][2]}")
            return bool(batch)
        return self._gate_store_call(name, args, said)

    def _gate_store_call(self, name: str, args: dict[str, Any], said: list[str]) -> bool:
        reply = self.act(name, args)
        if reply is None:
            return False
        seen = reply.get("observation")
        self.absorb(seen)
        self.done_log.append(f"step {self.step}: {name} {_dump(args)} -> {self.show(seen)[:300]} (cart then: {self.cart_mark()})")
        self.done_steps.append(self.step)
        said.append(f"{name} {_dump(args)} -> {self.show(seen)[:400]}")
        return True

    def gate(self, stage: str) -> tuple[bool, str]:
        """Right before a checkout ("order") or an ending without a purchase ("end"): the steps the request asks for
        there, done through store calls the model picks (shopper messages, pages, any store tool) until it sees none
        missing for the current cart. Returns whether to go on, and what to tell the agent."""
        steps = [p for p in self.process if p["when"] == stage]
        if not steps or self.finished or not self.gate_due(stage):
            return True, ""
        ending = {self.roles.get("buy"), self.roles.get("refuse")} - {None}
        tools = [f"- {t['function']['name']}: {t['function'].get('description', '')} | parameters: {_dump(t['function'].get('parameters') or {})}"
                 for t in self.tools if t["function"]["name"] not in ending]
        said: list[str] = []
        nudged = False
        rounds_left = GATE_ROUNDS
        while rounds_left > 0:
            rounds_left -= 1
            verdict, payload = self._gate_decide(stage, steps, tools, said, nudged)
            if verdict == "changed":
                self.version += 1
                head = (self._gate_preface(said) + "\n") if said else ""
                return False, head + "The shopper's answer changes the task: " + "; ".join(payload) + ". Adjust the cart to match, then check out again."
            if verdict == "nudge":
                nudged = True
                self._gate_nudge(payload)
                continue
            if verdict == "clear":
                self.gate_ok.add((stage, self.cart_mark() + f"#{self.version}"))
            if verdict != "call" or not self._gate_act(payload, ending, said):
                break
        return True, self._gate_preface(said)

    def _gate_preface(self, said: list[str]) -> str:
        return ("Kyo first carried out the steps the request asks for at this point:\n" + "\n".join(said)) if said else ""

    def spoken(self, observation: Any) -> str:
        if isinstance(observation, str):
            return observation
        # depth-first, values in their own order: phrases of two or more words, at most four levels down
        found: list[str] = []
        stack: list[tuple[Any, int]] = [(observation, 0)]
        while stack:
            node, depth = stack.pop()
            if depth > 4:
                continue
            if isinstance(node, str):
                if len(node.split()) >= 2:
                    found.append(node)
                continue
            kids = list(node.values()) if isinstance(node, dict) else node if isinstance(node, list) else []
            stack.extend((kid, depth + 1) for kid in reversed(kids))
        if not found:
            return _dump(observation)[:400]
        return " ".join(found)[:800]

    # ---------- notes inside tool results ----------
    def _loose_from(self, value: Any):
        if isinstance(value, str):
            if len(value.split()) >= 5:
                yield value
        elif isinstance(value, list):
            if value and all(isinstance(v, str) for v in value):
                yield from (v for v in value if len(v.split()) >= 5)
        elif isinstance(value, dict) and not self.ident(value, False, 1):
            yield from (inner for inner in value.values() if isinstance(inner, str) and len(inner.split()) >= 6)

    def loose_lines(self, observation: Any) -> list[str]:
        if not isinstance(observation, dict) or self.ident(observation):
            return []
        fresh = (line for value in observation.values() for line in self._loose_from(value))
        return [line for line in fresh if line not in self.heard]

    def _note_item(self, entry: dict) -> Any:
        slot = _count(entry.get("item")) or 1
        return self.items[min(max(1, slot), len(self.items)) - 1]

    @staticmethod
    def _note_entries(raw: Any) -> list[dict]:
        return [e for e in _items(raw) if isinstance(e, dict) and str(e.get("need") or "").strip()]

    def _drop_match(self, live: list[str], target: str, log: dict) -> list[str]:
        exact = [m for m in live if m == target]
        if exact:
            return exact
        head = target.split(":")[0]
        same = next((m for m in live if _words(m.split(":")[0]) == _words(head)), None)
        if same is not None:
            return [same]
        # named in other words ("collar type" for "collar"): the requirement whose property shares its words
        stem = _stem(head)
        log["gone"] = stem
        near = (m for m in live if _names(stem, _stem(m.split(":")[0])) or _names(_stem(m.split(":")[0]), stem))
        return [m for m in near][:1]

    def _note_budget(self, budget: Any, log: dict) -> None:
        fits = isinstance(budget, (int, float)) and not isinstance(budget, bool) and budget > 0
        if fits and budget != self.budget:
            self.budget = budget
            self.note_flags["budget"] = True
            log["changed"].append(f"budget now {budget}")

    def _note_drop(self, raw: Any, log: dict) -> None:
        targets = [(self._note_item(entry), str(entry["need"]).strip()) for entry in self._note_entries(raw)]
        if not targets:
            return
        for item, target in targets:
            live = [m for m in item.musts + item.told if m not in item.dropped]
            item.dropped.extend(self._drop_match(live, target, log) or [target])
        self.note_flags["dropped"] = True
        log["changed"].extend(f"dropped {target}" for _, target in targets)

    def _note_add(self, raw: Any, log: dict) -> None:
        for entry in self._note_entries(raw):
            item = self._note_item(entry)
            need = str(entry["need"]).strip()
            try:
                item.dropped.remove(need)
            except ValueError:
                pass
            item.told += [] if need in item.musts + item.told else [need]
            log["changed"].append(f"added {need}")

    def _note_gone(self, _raw: Any, log: dict) -> None:
        if log["gone"]:
            log["changed"].append("unavailable: " + ", ".join(log["gone"]))

    def _note_price(self, moved: Any, log: dict) -> None:
        if moved is True:
            log["changed"].append("a price changed")

    def take_note(self, lines: list[str]) -> str:
        self.heard.update(lines)
        prompt = NOTE + "\n\nITEMS AND REQUIREMENTS NOW:\n" + self.ledger() + "\n\nTHE NOTE:\n" + "\n".join(lines)
        got = self.ask_json(prompt, [LIGHT, SPARE], 1200) or {}
        gone = [str(g) for g in _items(got.get("gone")) if isinstance(g, (str, int))]
        self.unavailable.update(g for g in gone if g in self.rows)
        self.note_sold = bool(gone) or got.get("sold_out") is True
        if self.note_sold and len(lines) == 1:
            # a note that is only the sell-out is kept: the same words later mean the same thing
            self.sold_lines.update(lines)
        if got.get("kind") not in (None, "change"):
            return ""
        self.note_flags = {"dropped": False, "price": got.get("price_moved") is True, "budget": False}
        log: dict = {"changed": [], "gone": gone}
        steps = (
            ("budget", self._note_budget),
            ("drop", self._note_drop),
            ("add", self._note_add),
            (None, self._note_gone),
            ("price_moved", self._note_price),
        )
        for field, apply in steps:
            apply(got.get(field) if field else None, log)
        changed = log["changed"]
        summary = str(got.get("summary") or "").strip()
        if changed or summary:
            self.version += 1
            self.recent = []
            self.notes.append(summary or "; ".join(changed))
        tail = f" ({'; '.join(changed)})" if changed else ""
        return (summary + tail).strip()

    def after_change(self) -> str:
        # hints after a change note: dropped needs or moved prices can alter a cheapest/most/least choice; a lower budget can strand the cart
        flags, self.note_flags = self.note_flags, {}
        pick = self.brief.get("pick")
        carted = {i: q for i, q in self.cart_lines.items() if q > 0}
        prices = [self.price_of(i) for i in carted]
        priced = bool(carted) and None not in prices
        total = None
        if flags.get("budget") and priced and self.budget is not None and self.brief.get("budget_scope") != "each":
            total = sum(p * q for p, q in zip(prices, carted.values()))
        notes = (
            (flags.get("dropped") and pick in ("cheapest", "most", "least"),
             lambda: f"One requirement no longer applies; listings rejected only for it now qualify and could take the {pick} pick, so review them before you buy."),
            (flags.get("price") and pick == "cheapest",
             lambda: "Prices have changed: check the carted price again, since a different qualifying listing could now cost less."),
            (total is not None and total > self.budget,
             lambda: f"The cart now totals {total:g}, above the new budget of {self.budget:g}: replace lines with cheaper fitting listings."),
        )
        out = [say() for hit, say in notes if hit]
        return " " + " ".join(out) if out else ""

    def needs_now(self) -> list[tuple[int, str]]:
        return [(number, need) for number, item in enumerate(self.items, 1)
                for need in item.musts + item.told if need not in item.dropped]

    def _fact_ids(self, observation: Any) -> list[str]:
        # depth-first: a dict that names a listing with a fact block is taken whole (its insides are not searched)
        ids: list[str] = []
        stack: list[tuple[Any, int]] = [(observation, 0)]
        while stack:
            node, depth = stack.pop()
            if depth > 6:
                continue
            if isinstance(node, dict):
                ident = self.ident(node, False, depth)
                if ident and _fact_block(node) and ident not in ids:
                    ids.append(ident)
                    continue
                kids = list(node.values())
            elif isinstance(node, list):
                kids = node
            else:
                continue
            stack.extend((kid, depth + 1) for kid in reversed(kids))
        return ids

    def _evidence_line(self, ident: str, needs: list[tuple[int, str]], many: bool) -> str:
        block = _fact_block(self.full.get(ident) or {}) or {}
        props = ((number, need.split(":", 1)[0].strip()) for number, need in needs)
        parts = [
            ("(item %d) " % number if many else "") + f"{prop} -> " + ", ".join(hits[:3])
            for number, prop in props
            if (hits := [f"{k}={str(v)[:50]}" for k, v in block.items() if _names(_stem(prop), _stem(k))])
        ]
        return f"{ident}: " + "; ".join(parts) if parts else ""

    def evidence(self, observation: Any) -> str:
        ids = self._fact_ids(observation)
        needs = self.needs_now()
        if not (ids and needs):
            return ""
        many = len(self.items) > 1
        lines = [line for line in (self._evidence_line(i, needs[:8], many) for i in ids[:12]) if line]
        if not lines:
            return ""
        lead = "[evidence: record keys that mention a requirement's property name; a key about some other part or aspect settles nothing, and a differently named key can still carry the property] "
        return lead + " | ".join(lines)

    # ---------- checks ----------
    def _cart_fresh(self) -> bool:
        # a cart view older than the last cart change would show lines that are gone
        return self.cart_view is not None and self.cart_seen_at >= self.cart_at

    def cart_text(self) -> str:
        views = {
            True: lambda: _dump(_clip(self.cart_view, 300))[:3000],
            False: lambda: "lines by Kyo's record of the cart calls (listing: units): " + _dump(self.cart_lines),
        }
        return views[self._cart_fresh()]()

    def cart_ids(self) -> list[str]:
        views = {
            True: lambda: [i for i in self.rows if i in _dump(self.cart_view)],
            False: lambda: list(self.cart_lines),
        }
        return views[self._cart_fresh()]()

    def record_of(self, ident: str) -> str:
        record = self.full.get(ident) or self.rows.get(ident)
        return record and _dump(_clip(record, 400))[:3000] or "(no record seen)"

    def others(self, skip: list[str], limit: int = 30) -> str:
        order = self.seen_order[::-1]
        if self.brief.get("pick") == "cheapest":
            # listings whose attributes were read first, each group cheapest first
            def rank(i: str) -> tuple[bool, float]:
                price = self.price_of(i)
                return i not in self.full, price if price is not None else float("inf")
            order.sort(key=rank)
        kept = (i for i in order if i not in skip and not self.sold_out(i))
        picked = [i for _, i in zip(range(max(limit, 1)), kept)]
        return "\n".join(_dump(_clip(self.full.get(i) or self.rows.get(i), 160))[:900] for i in picked)

    def check_key(self, ids: list[str]) -> str:
        return "#".join(("|".join(sorted(ids)), str(self.version)))

    def _check_prompt(self, ids: list[str], lead: bool) -> str:
        sections = [CHECK, "\n\nREQUEST:\n", self.request, "\n\nREQUIREMENTS IN FORCE:\n", self.ledger()]
        if lead:
            quantity = "; ".join(f"item {n}: quantity {it.qty}" for n, it in enumerate(self.items, 1))
            sections += ["\n\nCART (not yet added; assess these listings as though they sat in the cart at the asked quantities: ", quantity, "):\n", ", ".join(ids)]
        else:
            sections += ["\n\nCART:\n", self.cart_text()]
        flags = ("", "SOLD OUT (the store says it can no longer be bought). ")
        records = (f"{i}: {flags[bool(self.sold_out(i))]}{self.record_of(i)}" for i in ids)
        sections += ["\n\nRECORDS OF THE CART'S LISTINGS:\n", "\n".join(records)]
        if self.brief.get("pick") in ("cheapest", "most", "least"):
            sections += ["\n\nOTHER LISTINGS SEEN (newest first):\n", self.others(ids, 15)]
        return "".join(sections)

    def _check_reply(self, got: Any, ids: list[str]) -> str:
        self.leads = [i for i in self.leads if i not in ids]
        problems = [str(p) for p in _items(got.get("problems"))][:5]
        better = [str(b) for b in _items(got.get("better")) if str(b) in self.rows and not self.sold_out(str(b))][:4]
        said = (
            (True, "A careful review says this cart is not right yet: " + "; ".join(problems or ["see the requirements"])),
            (bool(better), ". Listings that look better: " + ", ".join(better)),
            (True, ". Correct the cart, or re-read the records to prove the review mistaken, before checking out."),
        )
        return "".join(text for keep, text in said if keep)

    def check_cart(self, ids: list[str] | None = None, limit: float = SPEND_SOFT) -> str:
        lead = ids is not None
        if not lead:
            ids = self.cart_ids()
        key = self.check_key(ids)
        reply = ""
        if key not in self.checked and not self.cost > limit and ids:
            self.checked.add(key)
            got = self.ask_json(self._check_prompt(ids, lead), [HEAVY, LIGHT], 1500)
            self.trace.append({"kind": "check", "cart": ids, "got": got})
            rejected = bool(got) and got.get("ok") is False
            self.verdicts[key] = bool(got) and not rejected
            if rejected:
                reply = self._check_reply(got, ids)
        return reply

    def pending_refs(self) -> list[str]:
        open_needs = {need for _, need in self.needs_now()}.difference(self.ref_values)
        return [need for _, need in self.refs if need in open_needs]

    def _send(self, name: str | None, raw: dict[str, Any]) -> dict[str, Any] | None:
        """Shape one call's arguments for its tool and make it; None when the arguments do not fit or no reply came."""
        faults: list[str] = []
        args = _shape(raw, self.schemas.get(name), "args", faults)
        if faults:
            return None
        return self.act(name, args)

    def _trail_room(self) -> bool:
        """Whether another chain read fits: a few reads plus one per early read and per follow read, under the hard spend."""
        limit = 4 + self.early_trails + self.follow_reads
        return self.trails < limit and self.cost < SPEND_HARD

    def _trail_lines(self, early: bool):
        """Record lines for the chain read, newest first: opened pages in full, other rows briefly (not on an early read)."""
        for ident in reversed(self.seen_order):
            record = None
            if ident in self.opened:
                record = self.full.get(ident) or self.rows.get(ident)
            if record:
                yield "OPENED " + _dump(_clip(record, 400))[:2200]
            elif not early:
                yield "ROW " + _dump(_clip(self.rows.get(ident), 80))[:300]

    def _trail_read(self, pending: list[str], early: bool) -> Any:
        """One model read of the chains: which pages each pending requirement still needs, or its value."""
        cap = 30 if early else 60
        records = [line for _, line in zip(range(cap), self._trail_lines(early))]
        numbered = "\n".join(f"{n}. {need}" for n, need in enumerate(pending, 1))
        prompt = (TRAIL + "\n\nTHE SHOPPER'S REQUEST:\n" + self.request + "\n\nREQUIREMENTS WHOSE VALUE COMES FROM ANOTHER LISTING (by number):\n" + numbered
                  + "\n\nRECORDS (most recent first):\n" + "\n".join(records))
        # the light model gets room for a whole answer (a cut reply escalates to the far dearer heavy model)
        models = [LIGHT] if early else [LIGHT, HEAVY]
        got = self.ask_json(prompt, models, 3500) or {}
        self.trace.append({"kind": "trail", "pending": pending, "got": _clip(got, 150)})
        return got

    @staticmethod
    def _trail_need(chain: dict[str, Any], pending: list[str]) -> str | None:
        """The pending requirement a chain answers: by its number, else by its words, else the only one pending."""
        number = _count(chain.get("need"))
        if number and 1 <= number <= len(pending):
            pick = pending[number - 1]
        else:
            label = str(chain.get("need") or "").strip()
            pick = next((p for p in pending if p == label), None)
        return pick or (pending[0] if len(pending) == 1 else None)

    def _chain_complete(self, chain: dict[str, Any], links: list, ids: list[str]) -> bool:
        """A chain stands when the model calls it done, nothing is unmatched and its last page and every named page are open."""
        if chain.get("done") is not True or not links or _items(chain.get("unmatched")):
            return False
        return all(i in self.opened for i in [ids[-1]] + [i for i in ids if i])

    def _trail_decide(self, got: Any, pending: list[str], found: dict[str, str]) -> tuple[list[str], list[str]]:
        """Apply one read: settle finished chains into `found`, note what to search next; return the closed pages to
        open and the requirements whose next listing nobody has found."""
        reach: list[str] = []
        lost: list[str] = []
        for chain in _items(got.get("chains")):
            if not isinstance(chain, dict):
                continue
            need = self._trail_need(chain, pending)
            if not need:
                continue
            links = _items(chain.get("links"))
            ids = [str((l.get("id") if isinstance(l, dict) else l) or "").strip() for l in links]
            find = str(chain.get("find") or "").strip()
            if find:
                self.next_find[need] = find
            value = str(chain.get("value") or "").strip()
            if self._chain_complete(chain, links, ids) and value:
                found[need] = value
                self.version += 1
                continue
            closed = [i for i in ids if i in self.rows and i not in self.opened]
            if not closed:
                lost.append(f"{need} -> find {str(chain.get('about') or '') or 'the listing it points at'}")
                continue
            reach.extend([i for i in closed if i not in reach])
        return reach, lost

    def _trail_page(self, tool: str, target: str, ident: str, seen: list[str]) -> bool:
        """Open one named page with the page view and note what it showed; False when the run should stop opening."""
        reply = self._send(tool, {target: ident})
        if reply is None or self.finished:
            return False
        self.observe(tool, reply.get("observation"))
        seen.append(f"{ident}: " + self.show(reply.get("observation"))[:1500])
        return True

    def _trail_text(self, early: bool, seen: list[str], found: dict[str, str], known: dict[str, str],
                    lost: list[str]) -> tuple[str, bool]:
        """What the chain work showed, and (at checkout) whether a page still to be found holds the order back."""
        parts: list[str] = []
        if seen:
            parts.append("[Kyo opened the pages that requirements refer to]\n" + "\n".join(seen) + "\n")
        if early:
            fresh = [f"{k} -> {v}" for k, v in found.items() if known.get(k) != v]
            if fresh:
                parts.append("[Kyo followed the chain of pages; the product must have] " + "; ".join(fresh) + "\n")
            return "".join(parts), False
        if self.ref_values:
            parts.append("[values taken from the pages that requirements refer to] "
                         + "; ".join(f"{k} -> {v}" for k, v in self.ref_values.items()) + "\n")
        hold = bool(lost) and self.trail_blocks < 2 and self.left() > 5
        if hold:
            self.trail_blocks += 1
            parts.append("An order is accepted only once each page a requirement refers to has been opened in full (a search result row does not count), including every further listing such a page refers on to. Not yet found and opened: "
                         + " | ".join(lost[:3]) + ". Search for it by its shop, brand or wording, open its page, use the value it gives, and only then check out.")
        return "".join(parts), hold

    def trail(self, early: bool = False, settle: bool = False) -> tuple[str, bool]:
        """Before a checkout, every page in the chain of a requirement whose value sits on another listing gets opened:
        Kyo opens links it can name, and a link not yet found holds the checkout back (at most twice). Returns the text
        and the hold flag. With `early` it only reads opened pages so a finished chain reaches the ledger in time."""
        pending = self.pending_refs()
        # an early read stays provisional (more pages may open); the checkout read settles it
        found = self.ref_values if settle or not early else self.ref_early
        known = dict(found)
        tool = self.offer_tool()
        target = self.single_target(tool)
        seen: list[str] = []
        lost: list[str] = []
        while pending and self._trail_room():
            self.trails += 1
            got = self._trail_read(pending, early)
            reach, lost = self._trail_decide(got, pending, found)
            pending = [p for p in pending if p not in found]
            if early or not (reach and tool and target) or self.left() < len(reach) + 4 or self.finished:
                break
            batch = reach[: max(0, min(3, 6 - len(seen)))]
            all(self._trail_page(tool, target, ident, seen) for ident in batch)
        return self._trail_text(early, seen, found, known, lost)

    def unresolved(self) -> list[str]:
        """Requirements pointing at another listing with no value from their chain yet, provisional ones included."""
        early = self.ref_early
        return [need for need in self.pending_refs() if need not in early]

    def keyword_call(self, words: str) -> tuple[str, dict[str, Any]] | None:
        """A free-words search: the brief's tool, words parameter and fixed arguments; else the search tool when one
        free-text parameter is all it needs."""
        spec = self.brief.get("keyword_search")
        if not isinstance(spec, dict):
            spec = {}
        tool = spec.get("tool") if spec.get("tool") in self.schemas else self.roles.get("search")
        schema = self.schemas.get(tool or "") or {}
        props = schema.get("properties") or {}
        required = schema.get("required") or []
        texts = [k for k, v in props.items() if isinstance(v, dict) and "string" in _kinds(v) and "enum" not in v]
        param = (spec.get("words") if spec.get("words") in texts else None) \
            or next((k for k in required if k in texts), None) \
            or (texts[0] if len(texts) == 1 else None)
        if not (tool and param):
            return None
        fixed = spec.get("fixed")
        args = {k: v for k, v in fixed.items() if k in props} if isinstance(fixed, dict) else {}
        args[param] = words
        if not all(k in args for k in required):
            return None
        faults: list[str] = []
        shaped = _shape(args, schema, "args", faults)
        return (tool, shaped) if not faults else None

    def chain_budget(self) -> int:
        """Steps the follower may spend: the stated limit less the questions still to ask, the cart, the checkout and
        a few searches and reads per requested item, clamped to the follower's own cap."""
        reserve = [0 if self.asked else max(0, self.msgs_free())]
        reserve += [self.tail(), 3, 4 * len(self.items)]
        return sorted((0, self.left() - sum(reserve), FOLLOW_STEPS))[1]

    def _batch_plan(self, rows: list[str]) -> tuple[list[tuple[str, dict[str, Any]]], Any, Any]:
        """The calls that open these rows: one side-by-side call when that view shows the pages' words, else the first
        two with the page view; plus the side-by-side parameter and tool."""
        many, joiner = self.compare_target(), self.roles.get("compare")
        page = self.offer_tool()
        target = self.single_target(page)
        if many and joiner and not self.compare_bare():
            return [(joiner, {many: rows[: self.compare_room()]})], many, joiner
        if page and target:
            return [(page, {target: i}) for i in rows[:2]], many, joiner
        return [], many, joiner

    def _batch_learn(self, name: str, args: dict[str, Any], seen: Any, many: Any) -> None:
        """Learn the side-by-side view's real room from what one call showed."""
        asked = args.get(many)
        if not isinstance(asked, list):
            return
        shown = _dump(seen)
        hits = [i for i in asked if str(i) in shown]
        if not hits:
            self.compare_bad = min(self.compare_bad or 999, len(asked))
            return
        self.compare_ok = max(self.compare_ok, len(hits))
        self.facts_from[name] = self.facts_from.get(name, False) or any(_fact_block(r) for _, r in self.records_in(seen))

    def open_batch(self, rows: list[str]) -> int:
        """Open listings' pages within the store's real side-by-side room, or the first two one by one. Returns the
        steps used."""
        plan, many, joiner = self._batch_plan(rows)
        # the steps used are the calls made before the first one that could not be made
        return next((k for k, (name, raw) in enumerate(plan) if not self._batch_step(name, raw, many, joiner)), len(plan))

    def _batch_step(self, name: str, raw: dict[str, Any], many: Any, joiner: Any) -> bool:
        """Make one opening call and learn from it; False when it could not be made."""
        if self.finished:
            return False
        faults: list[str] = []
        args = _shape(raw, self.schemas.get(name), "args", faults)
        if faults:
            return False
        reply = self.act(name, args)
        if reply is None:
            return False
        seen = reply.get("observation")
        self.observe(name, seen)
        self.note_view(name, seen)
        if name == joiner:
            self._batch_learn(name, args, seen, many)
        return True

    def _follow_go(self, start: int, budget: int, rounds: int) -> bool:
        """Whether another follow round fits: a chain still open, rounds and steps left, spend under the soft line."""
        if self.finished or not self.unresolved() or rounds >= FOLLOW_ROUNDS:
            return False
        return self.step - start + 1 <= budget and self.cost < SPEND_SOFT

    def _follow_decide(self) -> str | None:
        """Read the chains once more; None when none is left open, '' when the read named nothing to find, else the
        words for the next listing."""
        self.follow_reads += 1
        self.trail(early=True, settle=True)
        needs = self.unresolved()
        if not needs:
            return None
        return next((self.next_find[n] for n in needs if self.next_find.get(n)), "")

    def _follow_pool(self, query: str, seen: Any) -> list[str]:
        """Unopened, in-stock rows of a result sharing the most of the query's words, best first."""
        words = _stem(query)
        scored: list[tuple[str, int]] = []
        for ident, rec in self.records_in(seen):
            if ident in self.opened or self.sold_out(ident):
                continue
            text = " ".join(str(v) for v in rec.values() if isinstance(v, str) and len(v) <= 200)
            scored.append((ident, len(words & _stem(text))))
        scored.sort(key=lambda pair: -pair[1])
        floor = max(1, (scored[0][1] if scored else 0) - 1)
        return [ident for ident, n in scored if n >= floor][:FOLLOW_ROWS]

    def _follow_slot(self, tried: dict, query: str) -> dict[str, Any] | None:
        """The rows to open for a query: what is left of its earlier result, else a fresh search (or that result's next
        page when its likely rows are all read). None when no search can be made."""
        key = _stem(query)
        slot = tried.get(key)
        if slot:
            # another query may have opened some of these rows meanwhile
            slot["pool"] = [i for i in slot["pool"] if i not in self.opened]
            if slot["pool"]:
                return slot
            plan = self.paging(slot["tool"], slot["seen"])
            if not plan or plan[3] is None:
                return None
            call = (slot["tool"], {**slot["args"], plan[2]: plan[3]})
        else:
            call = self.keyword_call(query)
            if not call:
                return None
        reply = self.act(call[0], call[1])
        if reply is None:
            return None
        seen = reply.get("observation")
        self.observe(call[0], seen)
        tried[key] = {"tool": call[0], "args": call[1], "seen": seen, "pool": self._follow_pool(query, seen)}
        return tried[key]

    def _follow_apply(self, slot: dict[str, Any], start: int, budget: int) -> None:
        """Open the next chunk of a slot's rows that fits side by side; rows the store did not open go back first."""
        if not slot["pool"] or self.step - start + 1 > budget:
            return
        room = 2 if self.compare_bare() else self.compare_room()
        chunk, rest = slot["pool"][:room], slot["pool"][room:]
        slot["pool"] = rest
        self.open_batch(chunk)
        slot["pool"] = [i for i in chunk if i not in self.opened] + slot["pool"]

    def _follow_report(self, used: int, before: dict[str, str]) -> str:
        """What following showed the agent: steps, values read, chains still open."""
        new = {k: v for k, v in self.ref_values.items() if before.get(k) != v}
        if not used and not new:
            return ""
        text = f"[Kyo itself followed the pages that requirements refer to ({used} steps)]"
        if new:
            text += (" Values found along the way, per each requirement's relation: " + "; ".join(f"{k} -> {v}" for k, v in new.items())
                     + ". All pages of these chains are open, so the order's page check is satisfied: do not search for or reopen them; look for the product that has that value.")
        left = self.unresolved()
        if left:
            text += " Still open: " + " | ".join(f"{n} (find next: {self.next_find.get(n) or 'the listing named on its last page'})" for n in left[:2]) + "."
        return text

    def follow(self) -> str:
        """Kyo follows requirement chains itself, round by round: a light read names the next listing (or the value),
        Kyo searches for it and opens its likely rows side by side, all within the steps the stated limit leaves."""
        if self.finished or not self.unresolved() or not self.keyword_call("x"):
            return ""
        budget = self.chain_budget()
        start = self.step
        before = dict(self.ref_values)
        opened_before = set(self.opened)
        tried: dict[frozenset[str], dict[str, Any]] = {}
        rounds = misses = 0
        while self._follow_go(start, budget, rounds):
            rounds += 1
            query = self._follow_decide()
            if query is None:
                break
            if not query:
                # nothing to go on: one more read, then the agent takes over
                misses += 1
                if misses >= 2:
                    break
                continue
            slot = self._follow_slot(tried, query)
            if slot is None:
                break
            self._follow_apply(slot, start, budget)
        if self.unresolved() and self.step > start:
            # the last pages opened may have closed a chain
            self.follow_reads += 1
            self.trail(early=True, settle=True)
        used = self.step - start
        self.followed |= self.opened - opened_before
        self.trace.append({"kind": "follow", "steps": used, "rounds": rounds, "values": dict(self.ref_values)})
        return self._follow_report(used, before)

    def offer_tool(self) -> str | None:
        """The tool that re-reads a listing's live offer: the offer view, else the single-listing view."""
        offer = self.roles.get("offer")
        return offer if offer else self.roles.get("open")

    def _reread_stale(self) -> list[str]:
        """Cart lines whose offer was last read before the latest refusal or two steps back, minus the gone ones."""
        since = max(self.step - 2, self.refused_at)
        return [i for i in self.cart_ids() if self.offer_seen.get(i, -99) < since and i not in self.gone]

    def _reread_line(self, tool: str, target: str, ident: str, lines: list[str]) -> bool:
        """Re-read one cart line's offer and note any plain field that moved; False when re-reading should stop."""
        prior = dict(self.rows.get(ident) or {})
        reply = self._send(tool, {target: ident})
        if reply is None or self.finished:
            return False
        observation = reply.get("observation")
        self.offer_seen[ident] = self.step
        self.observe(tool, observation)
        # compare stored records, so a re-read wrapped together with a note still shows its change
        moved = [k for k, v in (self.rows.get(ident) or {}).items()
                 if k in prior and prior[k] != v and not isinstance(v, (dict, list))]
        if moved:
            lines.append(f"{ident} has changed ({', '.join(moved[:4])}): {_dump(_clip(observation, 300))[:900]}")
        return True

    def reread_offers(self) -> str:
        """Re-read every cart line with the offer view right before checkout, since stock and price can move silently."""
        tool = self.offer_tool()
        target = self.single_target(tool)
        if not (tool and target):
            return ""
        stale = self._reread_stale()
        if not stale or self.left() < len(stale) + 3:
            return ""
        lines: list[str] = []
        all(self._reread_line(tool, target, ident, lines) for ident in stale)
        self.trace.append({"kind": "reread", "ids": stale, "changed": len(lines)})
        if lines:
            self.version += 1
        return "\n".join(lines)

    def sold_out(self, ident: str) -> bool:
        """Gone by a note or refusal, or its own record's availability flag flipped off."""
        if ident in self.unavailable:
            return True
        return ident in self.gone

    def swap_hint(self) -> str:
        """Cart lines newly seen as sold out: mark them, then name stand-ins already read (once per line)."""
        lost = list(filter(lambda i: i not in self.gone and self.sold_out(i), self.cart_ids()))
        if not lost:
            return ""
        self.gone.update(dict.fromkeys(lost, self.step))
        self.leads = list(filter(lambda i: i not in lost, self.leads))
        self.version += 1
        picks = self.replacements(lost)
        head = f"[Kyo] Sold out now: {', '.join(lost)} (the store marks it as unbuyable). It is gone for good, so do not open it again."
        if not picks:
            return head + " Take that line out of the cart and put in the best other fitting listing; if nothing you have read fits, search for one."
        first, joined = picks[0], ", ".join(picks)
        if self.left() <= 6:
            tail = f" Few steps are left: in this turn add {first} in the quantity asked for and delete the unavailable line (two calls), then go to checkout. Other read listings that look like fits: {', '.join(picks[1:]) or 'none'}."
        elif self.brief.get("pick") == "cheapest":
            tail = f" Read listings that look like fits for that item, lowest price first: {joined}. The item must be the cheapest, so unless the full set of fitting listings is already known, run one search priced below {first} (via the search's price ceiling, when it offers one) before replacing the sold-out line with the cheapest fitting listing."
        else:
            tail = f" Read listings that look like fits for that item: {joined}. This turn, drop the sold-out line from the cart and put in {first} at the requested quantity, then check out."
        return head + tail

    def find_sold(self) -> str:
        """A refusal reported a sold-out pick without naming it: one live cart line must be it; with several,
        re-read them in the offer view and return what that showed."""
        live = [i for i in self.cart_ids() if i not in self.gone]
        if any(map(self.sold_out, live)):
            return ""
        if not live:
            return ""
        if len(live) > 1:
            return self.reread_offers()
        self.unavailable.add(live[0])
        return ""

    def _newest_unsold(self, skip=()) -> list[str]:
        """Seen listings, newest first, minus the given ones, the gone ones and the sold-out ones."""
        return [i for i in self.seen_order[::-1] if i not in skip and i not in self.gone and not self.sold_out(i)]

    def replacements(self, lost: list[str]) -> list[str]:
        if self.cost > SPEND_SOFT:
            return []
        records = [_dump(_clip(self.full.get(i) or self.rows.get(i), 200))[:1200] for i in self._newest_unsold(lost)[:25]]
        if not records:
            return []
        parts = [SWAP, "REQUEST:\n" + self.request, "REQUIREMENTS IN FORCE:\n" + self.ledger(),
                 "SOLD-OUT CART LINE:\n" + "\n".join(f"{i}: {self.record_of(i)}" for i in lost),
                 "CART NOW:\n" + self.cart_text(), "OTHER LISTING RECORDS SEEN (newest first):\n" + "\n".join(records)]
        got = self.ask_json("\n\n".join(parts), [HEAVY, LIGHT], 1200)
        fits: list[tuple[float, str]] = []
        taken: set[str] = set()
        for entry in _items((got or {}).get("fits")):
            boxed = isinstance(entry, dict)
            ident = str(entry.get("id") if boxed else entry)
            blocked = ident not in self.rows or ident in lost or ident in self.gone or ident in self.cart_ids() or ident in taken
            if blocked:
                continue
            price = entry.get("price") if boxed else None
            numeric = isinstance(price, (int, float)) and not isinstance(price, bool)
            fits.append((float(price) if numeric else float("inf"), ident))
            taken.add(ident)
        # the model picks fits well but orders them by price poorly, so cheapest mode sorts (stable) by price
        order = sorted(fits, key=lambda f: f[0]) if self.brief.get("pick") == "cheapest" else fits
        self.trace.append({"kind": "swap", "lost": lost, "fits": fits})
        return [i for _, i in order][:4]

    def _lead_with(self, named: dict[str, list[str]]) -> str:
        """Make the named fits the current leads; return their 'item k: a, b' listing."""
        self.lead_items = named
        self.leads = [i for ids in named.values() for i in ids]
        return "; ".join(f"item {k}: {', '.join(v)}" for k, v in named.items())

    def fits_now(self) -> str:
        """Just after a chain value is learned: which opened listings now meet every requirement (light model first)."""
        if self.cost > SPEND_SOFT:
            return ""
        opened = [i for i in self.seen_order[::-1] if i in self.full and not self.sold_out(i)][:20]
        if not opened:
            return ""
        body = "\n".join(_dump(_clip(self.full[i], 200))[:1200] for i in opened)
        prompt = "\n\n".join([FITS, "REQUEST:\n" + self.request, "REQUIREMENTS IN FORCE:\n" + self.ledger(),
                              "OPENED LISTING RECORDS (newest first):\n" + body])
        got = self.ask_json(prompt, [LIGHT, HEAVY], 1200)
        self.trace.append({"kind": "fits", "got": got})
        named = self.audit_fits(got)
        return ("With that value, these opened listings look like they satisfy every requirement: " + self._lead_with(named)
                + ". Put the pick-rule winner in the cart unless its record violates a requirement.\n") if named else ""

    def audit(self) -> str:
        if self.audited or self.cost > SPEND_SOFT:
            return ""
        self.audited = True
        got = self.audit_call()
        named = self.audit_fits(got)
        # nothing named: rows never opened may still fit, so open the promising ones once and ask again
        unread = [] if named else [r for r in map(str, _items((got or {}).get("open")))
                                   if r in self.rows and r not in self.opened and not self.sold_out(r)]
        if unread and self.open_rows(unread):
            got = self.audit_call()
            named = self.audit_fits(got)
        if not named:
            return ""
        why = got.get("reason")
        return ("A careful review thinks these listings may fit: " + self._lead_with(named) + (f" ({why})" if why else "")
                + ". Open each and buy the one that really fits; finish without buying only if all of them fail.")

    def _record_line(self, ident: str) -> str:
        """An OPENED record when its page was read, else the shorter ROW record."""
        page = self.full.get(ident)
        if page:
            return "OPENED " + _dump(_clip(page, 200))[:1200]
        return "ROW " + _dump(_clip(self.rows.get(ident), 120))[:500]

    def audit_call(self) -> dict[str, Any] | None:
        records = [self._record_line(i) for i in self.seen_order[::-1] if not self.sold_out(i)][:30]
        if not records:
            return None
        prompt = "\n\n".join([AUDIT, "REQUEST:\n" + self.request, "REQUIREMENTS IN FORCE:\n" + self.ledger(),
                              "LISTING RECORDS SEEN (newest first):\n" + "\n".join(records)])
        got = self.ask_json(prompt, [HEAVY, LIGHT], 1500)
        self.trace.append({"kind": "audit", "got": got})
        return got

    def audit_fits(self, got: dict[str, Any] | None) -> dict[str, list[str]]:
        fits = (got or {}).get("fits")
        if not isinstance(fits, dict):
            return {}
        usable = lambda s: s in self.rows and not self.sold_out(s)
        named = dict((str(k), list(filter(usable, map(str, _items(v))))[:4]) for k, v in fits.items())
        return dict(filter(lambda kv: kv[1], named.items()))

    def _shaped_act(self, name: str, raw: dict[str, Any]) -> tuple[str, Any, Any]:
        """Fit raw args to the tool's schema and send them: ('bad', ...) on a schema fault, ('stop', ...) when the
        store answered nothing or the episode ended, else ('ok', args, reply)."""
        faults: list[str] = []
        args = _shape(raw, self.schemas.get(name), "args", faults)
        if faults:
            return "bad", args, None
        reply = self.act(name, args)
        return ("stop" if reply is None or self.finished else "ok"), args, reply

    def _open_plan(self, rows: list[str]) -> list[tuple[Any, dict[str, Any]]] | None:
        """One compare call over the rows when the store allows it, else single opens of the first two; None if neither."""
        many, joiner = self.compare_target(), self.roles.get("compare")
        tool = self.facts_tool()
        target = self.single_target(tool)
        if many and joiner and not self.compare_bare():
            return [(joiner, {many: rows[: self.compare_room()]})]
        if not (tool and target):
            return None
        return [(tool, {target: r}) for r in rows[:2]]

    def open_rows(self, rows: list[str]) -> bool:
        """Open search-only rows: all at once through the compare tool when possible, else the first two singly."""
        plan = self._open_plan(rows)
        if plan is None or self.left() < len(plan) + 5:
            return False
        before = len(self.full)
        queue = list(plan)
        while queue:
            name, raw = queue.pop(0)
            status, _, reply = self._shaped_act(name, raw)
            if status == "bad":
                return False
            if status == "stop":
                break
            self.observe(name, reply.get("observation"))
        self.trace.append({"kind": "audit_open", "rows": rows, "opened": len(self.full) - before})
        return len(self.full) > before

    def cart_calls(self, lines: dict[str, int], drop: list[str]) -> list[dict[str, Any]] | None:
        """Cart-tool arguments that leave exactly `lines` and none of `drop`, worked out from the schema; None if impossible."""
        name = self.roles.get("cart") or ""
        schema = self.schemas.get(name) or {}
        props = schema.get("properties") or {}
        typed = lambda s, kind: isinstance(s, dict) and kind in _kinds(s)
        plain = lambda s: typed(s, "string") and "enum" not in s
        bulk = next(((k, s["items"]["properties"]) for k, s in props.items()
                     if typed(s, "array") and isinstance(s.get("items"), dict) and isinstance(s["items"].get("properties"), dict)), None)
        if bulk:
            key, inner = bulk
            id_key = next((k for k, s in inner.items() if plain(s)), None)
            count_key = next((k for k, s in inner.items() if typed(s, "integer")), None)
            if not id_key:
                return None
            return [{key: [dict([(id_key, i)] + ([(count_key, n)] if count_key else [])) for i, n in lines.items()]}]
        texts = [k for k, s in props.items() if plain(s)]
        named_arg = next((k for t, k in self.id_args if t == name and k in texts), None)
        needed = next((k for k in schema.get("required") or [] if k in texts), None)
        target = named_arg or needed or (texts[0] if len(texts) == 1 else None)
        count = next((k for k, s in props.items() if typed(s, "integer")), None)
        mode = next((k for k, s in props.items() if isinstance(s, dict) and isinstance(s.get("enum"), list) and len(s["enum"]) >= 2), None)
        if not target:
            return None
        sets = lambda: self.cart_sets(name)

        def removal(ident: str) -> dict[str, Any] | None:
            have = self.cart_lines.get(ident, 1)
            if mode:
                return {target: ident, mode: props[mode]["enum"][1], **({count: have} if count else {})}
            if not count:
                return None
            if sets():
                return {target: ident, count: 0}
            floor = props[count].get("minimum")
            return {target: ident, count: -have} if isinstance(floor, int) and floor <= -have else None

        calls: list[dict[str, Any]] = [removal(ident) for ident in drop]
        if any(call is None for call in calls):
            return None
        for ident, want in lines.items():
            have = self.cart_lines.get(ident, 0) if ident in self.cart_ids() else 0
            if have == want:
                continue
            if have and not sets():
                return None
            extra = {}
            if mode:
                extra[mode] = props[mode]["enum"][0]
            if count:
                extra[count] = want
            calls.append({target: ident, **extra})
        return calls

    def _can_end_with_order(self) -> bool:
        return all(self.roles.get(role) for role in ("cart", "buy"))

    def closest_mode(self) -> bool:
        if self.brief.get("if_nothing_fits") != "closest":
            return False
        return self._can_end_with_order()

    def fallback_mode(self) -> bool:
        """True unless the request asks to decline: before a no-purchase ending, the nearest listings are re-read and
        ordered if nothing is missed (paid orders must satisfy all), otherwise the ending is kept."""
        if self.brief.get("if_nothing_fits") == "decline":
            return False
        return self._can_end_with_order()

    def lines_missing(self) -> int:
        """How many requested items still have no live cart line."""
        live = list(filter(lambda kv: kv[1] > 0 and not self.sold_out(kv[0]), self.cart_lines.items()))
        gap = len(self.items) - len(live)
        return gap if gap > 0 else 0

    def closest_due(self) -> bool:
        """Items still lack a line and the steps are nearly gone: put the closest listings forward now."""
        missing = self.lines_missing()
        checks = (self.fallback_mode, lambda: not self.finished, lambda: self.closest_tries == 0,
                  lambda: missing > 0, lambda: self.left() <= missing + self.tail() + 1)
        return all(check() for check in checks)

    def closest_lines(self) -> tuple[dict[str, int], bool]:
        """Per requested item, the closest listing as a model reads the records seen, plus whether that order falls
        short (a line misses something, an item has no line, or only the budget blocks a fitting order)."""
        def rank(i: str):
            price = self.price_of(i)
            return (i not in self.full, price if price is not None else float("inf"))
        records = [self._record_line(i) for i in sorted(self.seen_order, key=rank)][:40]
        if not records:
            return {}, False
        sold = ", ".join(i for i in self.seen_order if self.sold_out(i)) or "none"
        prompt = "\n\n".join([CLOSEST, "REQUEST:\n" + self.request, "REQUIREMENTS IN FORCE:\n" + self.ledger(),
                              "CART NOW: " + self.cart_text(), "SOLD OUT: " + sold, "LISTING RECORDS:\n" + "\n".join(records)])
        got = self.ask_json(prompt, [LIGHT, HEAVY], 3500) or {}
        self.trace.append({"kind": "closest", "got": _clip(got, 150)})
        usable = [ln for ln in _items(got.get("lines")) if isinstance(ln, dict) and str(ln.get("id") or "") in self.rows]
        slot = lambda ln: min(max(1, _count(ln.get("item")) or 1), len(self.items))
        plan: dict[str, int] = {str(ln["id"]): self.items[slot(ln) - 1].qty for ln in usable}
        worst = max((_count(ln.get("misses")) or 0 for ln in usable), default=0)
        short = worst > 0 or len(plan) < len(self.items) or got.get("budget_blocks") is True
        return plan, short

    def _closest_vetoed(self, plan: dict[str, int], short: bool) -> bool:
        """Not asked for the nearest listings: keep the no-purchase ending unless that order meets all within budget."""
        budget = self.budget if isinstance(self.budget, (int, float)) and self.brief.get("budget_scope") != "each" else None
        prices = [self.price_of(i) for i in plan]
        total = None
        if plan and None not in prices:
            total = sum(p * n for p, n in zip(prices, plan.values()))
        over = budget is not None and total is not None and total > budget
        if not (short or not plan or over):
            return False
        self.nothing_fits = True
        self.trace.append({"kind": "closest_declined", "short": short, "total": total, "budget": budget})
        return True

    def _apply_cart(self, cart: str, calls: list[dict[str, Any]]) -> list[str]:
        """Run the cart calls the remaining steps allow (keeping one for checkout); return each sent argument dump."""
        sent: list[str] = []
        for raw in calls[: max(0, self.left() - 1)]:
            status, args, reply = self._shaped_act(cart, raw)
            if status == "stop":
                break
            if status == "ok":
                self.note_cart(cart, args)
                self.observe(cart, reply.get("observation"))
                sent.append(_dump(args))
        return sent

    def _checkout_closest(self, buy: str, plan: dict[str, int], asked: bool, said: str) -> str:
        """Gate, then send the ending tool for the closest cart; returns the note for the agent."""
        done = "Kyo set the cart to the closest listings."
        if self.finished or self.left() < 1:
            return done
        go, gated = self.gate("order")
        if not go:
            return "Kyo set the cart to the closest listings (" + said + "). " + gated
        if self.finished or self.left() < 1:
            return done
        schema = self.schemas.get(buy) or {}
        free = [k for k, v in (schema.get("properties") or {}).items() if isinstance(v, dict) and "string" in _kinds(v) and "enum" not in v]
        order = dict(self.brief.get("buy_args") or {})
        if free and asked:
            # asked to be told: say it in the ending tool's free text
            order[free[0]] = "Nothing met every requirement, so this order holds the closest listings."
        faults: list[str] = []
        order = _shape(order, schema, "args", faults)
        if faults:
            return "Kyo set the cart to the closest listings (" + said + "); check out now."
        reply = self.act(buy, order)
        self.trace.append({"kind": "order_closest", "lines": plan, "reply": _clip(reply, 200)})
        if reply is None:
            return "Kyo set the cart to the closest listings (" + said + "); the checkout failed, try again."
        self.buys += 1
        if not reply.get("finished"):
            self.refused_at = self.step
        self.observe(buy, reply.get("observation"))
        return ("Not run (Kyo acted instead): nothing fit, so the nearest matches were used: Kyo filled the cart (" + said
                + ") and placed the order. Store reply: " + self.show(reply.get("observation")))

    def order_closest(self) -> str:
        """Fallback order: cart the nearest listing per item and check out; returns a note, or "" if no action was possible."""
        cart, buy = self.roles.get("cart"), self.roles.get("buy")
        blocked = not self.fallback_mode() or self.finished or self.closest_tries >= 2 or self.left() < 1
        if blocked:
            return ""
        self.closest_tries += 1
        plan, short = self.closest_lines()
        asked = self.closest_mode()
        if not asked and self._closest_vetoed(plan, short):
            return ""
        # no model answer: whatever is already carted is the order
        plan = plan or {i: n for i, n in self.cart_lines.items() if n > 0}
        if not plan:
            return ""
        now = [i for i, n in self.cart_lines.items() if n > 0]
        said = "; ".join(self._apply_cart(cart, self.cart_calls(plan, [i for i in now if i not in plan]) or []))
        return self._checkout_closest(buy, plan, asked, said)

    def complete_cart(self) -> str:
        """Agent stopped short of buying after a reviewed pick was confirmed: Kyo builds that cart itself and orders it."""
        tools = (self.roles.get("cart"), self.roles.get("buy"))
        if self.completed or not self.confirmed or self.sold_out(self.confirmed) or not all(tools):
            return ""
        self.completed = True
        state: dict[str, Any] = {"cart": tools[0], "buy": tools[1]}
        # each step reads and extends the shared state; the first one that fails leaves the order to the agent
        for step in (self._cc_plan, self._cc_lines, self._cc_vet, self._cc_fill, self._cc_order):
            if not step(state):
                return ""
        return ("Not run (Kyo acted instead): " + self.confirmed + " passed the second-opinion check; Kyo filled the cart ("
                + "; ".join(state["done"]) + ") and placed the order. The store answered: " + " | then: ".join(state["replies"]))

    def _cc_plan(self, state: dict[str, Any]) -> bool:
        """Step 1: one reviewed, in-stock listing per requested item, at most one per category."""
        cat_key = self.category_key()

        def kind(ident: str) -> str:
            if not cat_key:
                return ident
            return str((self.rows.get(ident) or {}).get(cat_key) or "").strip().lower()

        def leads():
            for number, ids in self.lead_items.items():
                slot = _count(number)
                pick = next((i for i in ids if not self.sold_out(i)), None)
                if pick and slot and 0 < slot <= len(self.items):
                    yield pick, self.items[slot - 1].qty

        plan: dict[str, int] = {}
        kinds: set[str] = set()
        for pick, qty in leads():
            label = kind(pick)
            if label not in kinds:
                plan[pick] = qty
                kinds.add(label)
        state.update(kind=kind, plan=plan, kinds=kinds)
        return self.confirmed in plan

    def _cc_lines(self, state: dict[str, Any]) -> bool:
        """Step 2: cart lines of other items stay, the rest go; the whole request must be covered and fit the steps left."""
        plan, kind, kinds = state["plan"], state["kind"], state["kinds"]
        others = [i for i in self.cart_ids() if i not in plan]
        stays = [not self.sold_out(i) and kind(i) not in kinds for i in others]
        keep = {i: self.cart_lines.get(i, 1) for i, kept in zip(others, stays) if kept}
        drop = [i for i, kept in zip(others, stays) if not kept]
        if len(plan) + len({kind(i) for i in keep}) < len(self.items):
            return False
        lines = {**keep, **plan}
        calls = self.cart_calls(lines, drop)
        state.update(lines=lines, drop=drop, calls=calls)
        return bool(calls) and len(calls) + 1 <= self.left()

    def _cc_passes(self, lines: dict[str, int]) -> bool:
        self.check_cart(list(lines), SPEND_HARD)
        return bool(self.verdicts.get(self.check_key(list(lines))))

    def _cc_vet(self, state: dict[str, Any]) -> bool:
        """Step 3: acting costs no model call; a cart not yet passed gets one check, allowed up to the hard spend cap."""
        self.fill_facts(list(state["lines"]), len(state["calls"]) + 3)
        return self._cc_passes(state["lines"])

    def _cc_shaped(self, tool: str, args: Any) -> tuple[Any, bool]:
        faults: list[str] = []
        shaped = _shape(args, self.schemas.get(tool), "args", faults)
        return shaped, not faults

    def _cc_send(self, cart: str, raw: Any) -> str | bool | None:
        """One cart call of the plan: its dumped args when sent, False when it failed, None when no call is left."""
        if raw is None:
            return None
        args, ok = self._cc_shaped(cart, raw)
        reply = self.act(cart, args) if ok else None
        if reply is None or self.finished:
            return False
        self.note_cart(cart, args)
        self.observe(cart, reply.get("observation"))
        return _dump(args)

    def _cc_fill(self, state: dict[str, Any]) -> bool:
        """Step 4: send the cart calls, then shape the checkout arguments."""
        done: list[str] = []
        pending = iter(state["calls"])
        while (sent := self._cc_send(state["cart"], next(pending, None))) is not None:
            if sent is False:
                return False
            done.append(sent)
        state["done"] = done
        state["buy_args"], ok = self._cc_shaped(state["buy"], dict(self.brief.get("buy_args") or {}))
        return ok

    def _cc_checkout(self, state: dict[str, Any], replies: list[str]) -> bool:
        """One checkout attempt; True when the store refused it and a retry may follow."""
        go, said = self.gate("order")
        if said:
            replies.append(said)
        if not go:
            return False
        buy = state["buy"]
        reply = self.act(buy, state["buy_args"])
        self.trace.append({"kind": "completed", "lines": state["lines"], "drop": state["drop"], "reply": _clip(reply, 200)})
        if reply is None:
            return False
        self.buys += 1
        if not reply.get("finished"):
            self.refused_at = self.step
        self.observe(buy, reply.get("observation"))
        replies.append(self.show(reply.get("observation")))
        return not (self.finished or reply.get("finished"))

    def _cc_order(self, state: dict[str, Any]) -> bool:
        """Step 5: check out; a refused checkout (a price, budget or stock change) is retried once after re-reading the
        lines and a fresh check."""
        lines = state["lines"]
        replies: list[str] = []
        if self._cc_checkout(state, replies):
            self.reread_offers()
            if (self._cc_passes(lines) and self.left() >= 1 and not self.finished
                    and not any(self.sold_out(i) for i in lines)):
                self._cc_checkout(state, replies)
        state["replies"] = replies
        return bool(replies)

    # ---------- tool use ----------
    def _ends_with(self, role: str, fixed_key: str, name: str, args: dict[str, Any]) -> bool:
        """A call to the closing tool of this role carrying every argument the brief fixes for it."""
        if name != self.roles.get(role):
            return False
        fixed = self.brief.get(fixed_key)
        return not isinstance(fixed, dict) or all(args.get(k) == v for k, v in fixed.items())

    def is_buy(self, name: str, args: dict[str, Any]) -> bool:
        return self._ends_with("buy", "buy_args", name, args)

    def is_refuse(self, name: str, args: dict[str, Any]) -> bool:
        return self._ends_with("refuse", "refuse_args", name, args)

    def offered(self) -> list[dict[str, Any]]:
        left, tail = self.left(), self.tail()
        fresh, size = self.buys == 0, len(self.items)
        ends = {self.roles.get("buy"), self.roles.get("refuse")} - {None}
        cart_pair = {self.roles.get("cart"), self.roles.get("cart_view")}
        # (when, tools added to the closing ones); the first rung that holds narrows the menu. The last rung: every item
        # has a line, so the first checkout goes now and a change the store announces there still leaves steps to act on
        ladder = (
            (lambda: left <= tail, set()),
            (lambda: left <= tail + 3, cart_pair),
            (lambda: left <= size + 5 and fresh, cart_pair | {self.roles.get("offer")}),
            (lambda: left <= size + 7 and fresh and bool(self.cart_lines) and self.lines_missing() == 0, cart_pair),
        )
        extra = next((more for holds, more in ladder if holds()), None)
        ask = self.roles.get("ask")
        tools = [self.cart_explained(t) for t in self.tools if t["function"]["name"] != ask]
        if extra is None or not ends:
            return tools
        allowed = ends | extra
        return [t for t in tools if t["function"]["name"] in allowed] or tools

    def quiet(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        schema = self.schemas.get(name) or {}
        needed = schema.get("required") or []
        always = self.brief.get("quiet_decline") is not False
        decline = "Sorry, nothing in the store meets all of your requirements, so I am not buying anything."
        out = dict(args)
        for key, spec in (schema.get("properties") or {}).items():
            free_text = isinstance(spec, dict) and "string" in _kinds(spec) and "enum" not in spec
            if not free_text or (out.get(key) is None and key not in needed):
                continue
            text = str(out.get(key) or "")
            if not text or always or any(ident in text for ident in self.rows):
                out[key] = decline[: spec.get("maxLength", 200)]
        return out

    def _id_for(self, value: Any) -> tuple[Any, str]:
        """One argument value as a listing id: kept, matched from a title start, or a reason it cannot be used."""
        if not isinstance(value, str):
            return value, ""
        bare = value.strip()
        if bare in self.rows:
            return bare, ""
        if _id_like(value, True) or len(bare) < 8:
            return value, ""
        low = bare.lower()

        def carries(row: dict[str, Any]) -> bool:
            texts = (v for k, v in row.items() if k != self.id_key and isinstance(v, str))
            return any(t.strip().lower().startswith(low) for t in texts)

        hits = [ident for ident, row in self.rows.items() if carries(row)]
        if not hits:
            return value, f"{value!r} does not name a listing; send the {self.id_key} value shown in the result rows instead"
        if len(hits) > 1:
            return value, f"{value!r} matches more than one listing ({', '.join(hits[:6])}); send a single listing id"
        return hits[0], ""

    def resolve_ids(self, name: str, args: dict[str, Any]) -> tuple[dict[str, Any], str]:
        """Swap a title for its listing id when exactly one known row matches; else explain, so no step is wasted."""
        takers = {self.roles.get(r) for r in ("cart", "open", "offer", "compare")} - {None}
        if not self.id_key or name not in takers:
            return args, ""
        props = (self.schemas.get(name) or {}).get("properties") or {}
        free = [k for k, v in props.items() if isinstance(v, dict) and "enum" not in v and {"string", "array"} & set(_kinds(v))]
        # only the tool's one free argument, and only where it is known to take ids (the cart, or a call seen carrying one)
        if len(free) != 1:
            return args, ""
        key = free[0]
        if name != self.roles.get("cart") and (name, key) not in self.id_args:
            return args, ""
        if key not in args:
            return dict(args), ""
        raw = args[key]
        many = isinstance(raw, list)
        fixed = [self._id_for(v) for v in (raw if many else [raw])]
        problem = next((p for _, p in fixed if p), "")
        if problem:
            return args, problem
        values = [v for v, _ in fixed]
        return {**args, key: values if many else values[0]}, ""

    def cart_count(self, name: str | None) -> str | None:
        """Name of the cart tool's unit-count (integer) argument."""
        for key, spec in ((self.schemas.get(name or "") or {}).get("properties") or {}).items():
            if isinstance(spec, dict) and "integer" in _kinds(spec):
                return key
        return None

    def cart_explained(self, tool: dict[str, Any]) -> dict[str, Any]:
        """Tool spec handed to the model, with the cart count's meaning spelled out."""
        fn = tool["function"]
        if fn["name"] != self.roles.get("cart"):
            return tool
        count = self.cart_count(fn["name"])
        if not count:
            return tool
        if self.cart_sets(fn["name"]):
            said = "total units this listing's cart line should end up with (not an index): 3 means the line holds exactly 3, 0 removes the line"
        else:
            said = "number of units of this listing to add or remove"
        params = dict(fn.get("parameters") or {})
        props = dict(params.get("properties") or {})
        props[count] = dict(props[count], description=said)
        return {**tool, "function": {**fn, "parameters": {**params, "properties": props}}}

    @staticmethod
    def _sole(found: list[Any]) -> tuple[bool, Any]:
        return (True, found[0]) if len(found) == 1 else (False, None)

    def item_of(self, ident: str) -> Item | None:
        """Which requested item this listing belongs to (via its category), or None if unclear."""
        if len(self.items) == 1:
            return self.items[0]
        cat_key = self.category_key()
        label = str((self.rows.get(ident) or {}).get(cat_key) or "") if cat_key else ""
        if not label:
            return None
        said = _words(label)
        one, item = self._sole([it for it in self.items if said in (_words(it.shelf or ""), _words(it.name))])
        if one:
            return item

        def long(words: frozenset[str]) -> set[str]:
            return {w for w in words if len(w) >= 4}

        mine = long(_stem(label))
        return self._sole([it for it in self.items if mine & long(_stem(it.shelf or "") | _stem(it.name))])[1]

    def cart_guard(self, name: str, args: dict[str, Any]) -> tuple[dict[str, Any], str, str]:
        """For set-style cart tools: force the requested quantity; refuse a 0 for a listing not in the cart.
        Gives back (args, refusal, remark)."""
        count = self.cart_count(name)
        guarded = name == self.roles.get("cart") and count and self.cart_sets(name) and isinstance(args.get(count), int)
        ident = next((v for v in args.values() if isinstance(v, str) and v in self.rows), None) if guarded else None
        if not ident:
            return args, "", ""
        sent = args[count]
        if sent <= 0:
            if ident in self.cart_lines:
                return args, "", ""
            want = self.item_of(ident)
            ending = f" ({want.qty})." if want else "."
            return args, f"Not run (no step used): {ident} has no cart line, and {count} sets the line's total, so 0 changes nothing. Pass the amount you need{ending}", ""
        want = self.item_of(ident)
        if not want or sent == want.qty:
            return args, "", ""
        return {**args, count: want.qty}, "", f"\n[Kyo] Sent {count}={want.qty}: it is this line's quantity, and the request asks for {want.qty} of {want.name}."

    def fix_quantities(self) -> list[str]:
        """Before ordering, correct any cart line whose units differ from the request (one step per line)."""
        name = self.roles.get("cart")
        count, target = self.cart_count(name), self.single_id_arg(name)
        if not (name and count and target and self.cart_sets(name)):
            return []

        def due():
            # read lazily: each line is judged against the steps left after the previous fix
            for ident, have in list(self.cart_lines.items()):
                want = self.item_of(ident)
                if want and have != want.qty and self.left() > self.tail() + 1 and not self.finished:
                    yield ident, have, want.qty

        done: list[str] = []
        for ident, have, qty in due():
            args, ok = self._cc_shaped(name, {target: ident, count: qty})
            if not ok:
                continue
            reply = self.act(name, args)
            if reply is None:
                break
            self.note_cart(name, args)
            self.observe(name, reply.get("observation"))
            done.append(f"{ident}: {have} -> {qty}")
        return done

    def single_id_arg(self, name: str | None) -> str | None:
        """The cart tool's parameter taking the listing id."""
        schema = self.schemas.get(name or "") or {}
        texts = [k for k, s in (schema.get("properties") or {}).items() if isinstance(s, dict) and "string" in _kinds(s) and "enum" not in s]
        # seen carrying an id, else required, else the only text parameter
        for source in ((k for t, k in self.id_args if t == name), schema.get("required") or []):
            hit = next((k for k in source if k in texts), None)
            if hit:
                return hit
        return texts[0] if len(texts) == 1 else None

    def cart_sets(self, name: str) -> bool:
        """True for a cart tool that sets totals: its count allows 0 and it has no add/remove mode switch."""
        specs = [s for s in (self.schemas.get(name, {}).get("properties") or {}).values() if isinstance(s, dict)]
        if any(isinstance(s.get("enum"), list) and len(s["enum"]) >= 2 for s in specs):
            return False
        return any("integer" in _kinds(s) and s.get("minimum") == 0 for s in specs)

    def _lines_sent(self, spec: dict[str, Any], args: dict[str, Any]) -> tuple[list[tuple[str, int | None]], bool]:
        """The (listing, count) pairs a cart call names, and whether it replaces the whole cart (a list of line objects)."""
        def number(values: Any, default: int | None) -> int | None:
            return next((_count(v) for v in values if _count(v) is not None and not isinstance(v, str)), default)

        loose = number(args.values(), None)
        lines: list[tuple[str, int | None]] = []
        whole = False
        for key, value in args.items():
            if isinstance(value, str):
                if value in self.rows:
                    lines.append((value, loose))
                continue
            if not isinstance(value, list):
                continue
            shape = (spec.get(key) or {}).get("items") or {}
            if not value and isinstance(shape, dict) and (shape.get("type") == "object" or "properties" in shape):
                whole = True
            for line in value:
                if isinstance(line, str):
                    if line in self.rows:
                        lines.append((line, loose))
                elif isinstance(line, dict):
                    whole = True
                    ident = next((v for v in line.values() if isinstance(v, str) and v in self.rows), None)
                    if ident:
                        lines.append((ident, number(line.values(), 1)))
        return lines, whole

    def note_cart(self, name: str, args: dict[str, Any]) -> None:
        self.cart_touched = True
        self.cart_at = self.step
        spec = self.schemas.get(name, {}).get("properties") or {}
        lines, whole = self._lines_sent(spec, args)
        if whole:
            self.cart_lines = {i: c for i, c in lines if c and c > 0}
            return
        # a value past an enum's first option is a take-off mode
        removal = False
        for key, value in args.items():
            options = (spec.get(key) or {}).get("enum")
            removal = removal or (isinstance(options, list) and len(options) >= 2 and value in options[1:])
        operations = {
            "remove": lambda have, count: have - count if count else 0,
            "set": lambda have, count: 1 if count is None else count,
            "add": lambda have, count: have + (1 if count is None else count),
        }
        apply = operations["remove" if removal else "set" if self.cart_sets(name) else "add"]
        for ident, count in lines:
            have = apply(self.cart_lines.get(ident, 0), count)
            if have > 0:
                self.cart_lines[ident] = have
            else:
                self.cart_lines.pop(ident, None)

    def show(self, observation: Any, chars: int = SHOW_CHARS) -> str:
        text = _dump(_clip(observation, chars))
        head = text[:OBS_LIMIT]
        return head + "...(cut)" if head != text else text

    def compare_target(self) -> str | None:
        schema = self.schemas.get(self.roles.get("compare") or "") or {}
        arrays = (k for k, v in (schema.get("properties") or {}).items() if isinstance(v, dict) and "array" in _kinds(v))
        target = next(arrays, None)
        return target if target and all(k == target for k in schema.get("required") or []) else None

    def compare_room(self) -> int:
        """Max listings per compare call, from the schema cap or the store's observed behaviour."""
        spec = ((self.schemas.get(self.roles.get("compare") or "") or {}).get("properties") or {}).get(self.compare_target() or "") or {}
        cap = spec.get("maxItems")
        limit = cap if isinstance(cap, int) and cap > 0 else None
        # what the store has shown wins over its schema: a refused count halves the room, a taken one holds it up
        floors = {True: lambda: max(2, (self.compare_bad + 1) // 2),
                  False: lambda: max(2, limit // 2) if limit else COMPARE_FIRST}
        room = max(self.compare_ok, floors[bool(self.compare_bad)]())
        return room if not limit else min(room, limit)

    def join_opens(self, calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Merge a turn's single-listing opens into one compare call to save steps."""
        opener, joiner = self.roles.get("open"), self.roles.get("compare")
        target = self.single_target(opener)
        many = self.compare_target()
        # only into a side-by-side view that shows what the opens would (attributes)
        if not (opener and joiner and target and many) or opener == joiner or self.compare_bare():
            return calls

        def opened(call: dict[str, Any]) -> str | None:
            fn = call.get("function") or {}
            raw = fn.get("arguments")
            args = _json_obj(raw) if isinstance(raw, str) else None
            ident = str((args or {}).get(target) or "").strip()
            fits = fn.get("name") == opener and args and set(args) == {target} and ident in self.rows and ident not in self.gone
            return ident if fits else None

        found = ((index, opened(call)) for index, call in enumerate(calls))
        picks = [(index, ident) for index, ident in found if ident is not None][: self.compare_room()]
        if len(picks) < 2:
            return calls
        ids = [ident for _, ident in picks]
        lead, folded = picks[0][0], {index for index, _ in picks[1:]}

        def recast(index: int, call: dict[str, Any]) -> dict[str, Any]:
            if index == lead:
                return {"id": call["id"], "type": "function", "function": {"name": joiner, "arguments": _dump({many: list(ids)})}}
            return {"id": call["id"], "joined": True} if index in folded else call

        out = [recast(index, call) for index, call in enumerate(calls)]
        self.trace.append({"kind": "joined", "ids": ids})
        return out

    def run_calls(self, calls: list[dict[str, Any]]) -> list[tuple[str, str]]:
        # each call flows as one record through the ordered stages below; the first stage that produces a reply
        # text settles the call (the last stage always does)
        calls = self.join_opens(calls)
        others = {tool for job, tool in self.roles.items() if job != "search"}
        stages = (self._cs_admit, self._cs_parse, self._cs_squeeze, self._cs_vet, self._cs_cart_guard, self._cs_ending,
                  self._cs_seen, self._cs_buy_review, self._cs_refuse_review, self._cs_late_decline, self._cs_push,
                  self._cs_refuse_wrap, self._cs_gate, self._cs_execute, self._cs_compare_count, self._cs_record,
                  self._cs_roles, self._cs_cart_after, self._cs_body, self._cs_tail)
        results: list[tuple[str, str]] = []
        for call in calls:
            rec = _CallRec(call, others, len(calls))
            results.append((rec.cid, next(text for text in map(lambda stage: stage(rec), stages) if text is not None)))
        return results

    def _cs_admit(self, rec: "_CallRec") -> str | None:
        rec.allowed = {t["function"]["name"] for t in self.offered()}
        if rec.call.get("joined"):
            return "Covered by the side-by-side result above: Kyo read these listings as a group in a single step."
        fn = rec.call.get("function") or {}
        rec.name, rec.raw = fn.get("name"), fn.get("arguments")
        menu = ", ".join(sorted(rec.allowed))
        refusals = (
            (lambda: self.finished, "Not run: the episode is over."),
            (lambda: rec.name == self.roles.get("ask"), "Not run (no step used): all messages to the shopper go through Kyo's interviewer."),
            (lambda: rec.name not in self.schemas, f"Not run (no step used): unknown tool. Tools: {menu}."),
            (lambda: rec.name not in rec.allowed, "Not run (no step used): not enough steps remain for that; available now: " + menu + "."),
        )
        return next((said for test, said in refusals if test()), None)

    def _cs_parse(self, rec: "_CallRec") -> str | None:
        args = rec.raw if isinstance(rec.raw, dict) else (_json_obj(rec.raw) if (rec.raw or "").strip() else {})
        if args is None:
            return "Not run (no step used): the arguments did not form a single JSON object."
        faults: list[str] = []
        args = _shape(args, self.schemas[rec.name], "args", faults)
        if faults:
            return "Not run (no step used): " + "; ".join(faults[:6])
        rec.args, wrong = self.resolve_ids(rec.name, args)
        return "Not run (no step used): " + wrong + "." if wrong else None

    def _cs_squeeze(self, rec: "_CallRec") -> None:
        rec.many = self.compare_target() if rec.name == self.roles.get("compare") else None
        listed = rec.args.get(rec.many) if rec.many else None
        if isinstance(listed, list) and len(listed) > self.compare_room():
            room = self.compare_room()
            rec.note = f"\n[Kyo] Sent the first {room} listings only (one side-by-side call takes {room} here). Not shown yet: {', '.join(map(str, listed[room:]))}."
            rec.args = {**rec.args, rec.many: listed[:room]}
        rec.target = self.single_target(rec.name)
        return None

    def _cs_vet(self, rec: "_CallRec") -> str | None:
        name, args, target = rec.name, rec.args, rec.target
        picked = str(args.get(target)) if target else None
        if target and name in (self.roles.get("open"), self.roles.get("offer")) and picked in self.gone:
            return f"Not run (no step used): {args[target]} sold out at step {self.gone[picked]} and stays unavailable. Swap its cart line for a different listing that fits."
        if name != self.roles.get("cart") or not any((_count(v) or 0) < 0 for v in args.values() if not isinstance(v, str)):
            return None
        view = _dump(self.cart_view) if self.cart_seen_at >= self.cart_at else ""
        absent = next((v for v in args.values() if isinstance(v, str) and v in self.rows and v not in self.cart_lines
                       and v not in view and v not in self.cart_said), None)
        return None if absent is None else f"Not run (no step used): {absent} has no cart line, so there is nothing to remove. Cart now: " + self.cart_text()

    def _cs_cart_guard(self, rec: "_CallRec") -> str | None:
        rec.args, refusal, remark = self.cart_guard(rec.name, rec.args)
        if refusal:
            return refusal
        rec.note += remark
        return None

    def _cs_ending(self, rec: "_CallRec") -> str | None:
        rec.ending = self.is_buy(rec.name, rec.args) or self.is_refuse(rec.name, rec.args)
        if rec.ending:
            return "Not run (no step used): read your other results first, then send the finishing call alone." if rec.total > 1 else None
        if self.left() <= self.tail() and (self.roles.get("buy") or self.roles.get("refuse")):
            return "Not run (no step used): the remaining steps are reserved for the checkout (plus the order-time step the request calls for, if any) or for ending without a purchase."
        return None

    def _cs_seen(self, rec: "_CallRec") -> str | None:
        name, args, target, many = rec.name, rec.args, rec.target, rec.many
        rec.mark = name + _dump(args)
        if not rec.ending and name != self.roles.get("cart") and rec.mark in self.recent:
            return "Not run (no step used): this exact call was just made and nothing has changed; see its result above."
        rec.readers = {self.roles.get(r) for r in ("search", "open", "offer", "compare")} - {None}
        reading = name in rec.readers
        cached = self.read_cache.get(rec.mark)
        if reading and cached and cached[0] == self.version:
            return f"Not run (no step used): the same call at step {cached[1]} returned this, and nothing has changed since:\n" + cached[2][:3000]
        single = bool(target) and reading and name != self.roles.get("compare")
        if single and str(args.get(target)) in self.followed and str(args.get(target)) not in self.gone:
            return f"Not run (no step used): Kyo opened {args[target]} when it traced a requirement's chain, so it is already open. Its record: " + self.record_of(str(args[target]))
        blind = "" if name in rec.others else self.blind_call(name, args)
        if blind:
            return blind
        if single and self.shown_already(name, str(args.get(target))):
            return f"Not run (no step used): {args[target]} has already given all this view offers, and nothing changed since. Its record: " + self.record_of(str(args[target]))
        listed = args.get(many) if many else None
        if not (isinstance(listed, list) and listed):
            return None
        keep = [i for i in listed if not self.shown_already(name, str(i))]
        if not keep:
            return ("Not run (no step used): these listings already gave all this view offers, and nothing changed since. Their records:\n"
                    + "\n".join(f"{i}: {self.record_of(str(i))}" for i in listed)[:4000])
        if len(keep) < len(listed):
            rec.note += f"\n[Kyo] Left out {', '.join(str(i) for i in listed if i not in keep)}: shown in full already, nothing changed since."
            rec.args = {**args, many: keep}
            rec.mark = name + _dump(rec.args)
        return None

    def _cs_peek_cart(self) -> str:
        # look at the cart once before checking out; a change found there holds the checkout
        faults: list[str] = []
        view_args = _shape({}, self.schemas.get(self.roles["cart_view"]), "args", faults)
        view = None if faults else self.act(self.roles["cart_view"], view_args)
        if view is None:
            return ""
        self.cart_view = view.get("observation")
        self.cart_seen_at = self.step
        change = self.observe(self.roles["cart_view"], self.cart_view)
        if not change:
            return ""
        self.queued = []
        return "Not run yet (no step used). The cart view revealed a change: " + change + "\nCart now: " + self.cart_text() + "\nApply the change, fix the cart if needed, then check out."

    def _cs_buy_review(self, rec: "_CallRec") -> str | None:
        if not (rec.ending and self.left() > 2 and self.is_buy(rec.name, rec.args)):
            return None
        wider = self.wider_hold()
        if wider:
            return wider + "\nCart now: " + self.cart_text()
        fixed = self.fix_quantities()
        if fixed:
            rec.note += "\n[Kyo] Set the requested quantities: " + "; ".join(fixed)
        reread = self.reread_offers()
        if self.cart_touched and self.cart_seen_at < self.cart_at and self.roles.get("cart_view") and self.left() > 3:
            held = self._cs_peek_cart()
            if held:
                return held
        swap = self.swap_hint()
        # a sold-out cart line holds the checkout at most twice, so a wrong cart record cannot stall the episode
        gone = [i for i in self.cart_ids() if i in self.gone] if self.held < 2 else []
        if swap or gone:
            if not swap:
                self.held += 1
            return "Not run yet (no step used). " + (swap or f"{', '.join(gone)} in the cart is sold out: replace that line first.") + "\nCart now: " + self.cart_text()
        chain, hold = self.trail()
        if hold:
            return "Not run yet (no step used). " + chain + "\nCart now: " + self.cart_text()
        # the check judges attributes: read the cart lines' own (and, for a cheapest pick, a few cheaper rows')
        read = self.fill_facts(self.cart_ids(), self.tail() + 3) + self.cheaper_rows()
        verdict = self.check_cart()
        if verdict:
            seen = "For the check Kyo read the attributes of these listings (a step apiece):\n" + "\n".join(read) + "\n" if read else ""
            return "Not run yet. " + chain + seen + verdict + "\nCart now: " + self.cart_text()
        if reread and self.cost > SPEND_SOFT:
            return "Not run yet (no step used). Kyo has just re-read the listings in the cart:\n" + reread + "\nRetry the checkout once each line is still in stock and priced within budget; if one is not, replace it first."
        return None

    def _cs_refuse_review(self, rec: "_CallRec") -> str | None:
        if not (rec.ending and self.left() > 2) or self.is_buy(rec.name, rec.args) or self.left() <= 3:
            return None
        # a requirement whose value sits on another listing's page: read the opened chain first, since the value it
        # gives (often not the last page's own) can make listings fit that seemed not to
        if self.pending_refs() and self.early_trails <= EARLY_TRAILS:
            self.early_trails += 1
            chain, _ = self.trail(early=True)
            if chain and self.left() > 4:
                return "Not run yet (no step used). " + chain + self.fits_now() + "Run a new search using this value together with the other requirements and buy a fitting listing; end without a purchase only if nothing fits."
        review = "" if self.closest_mode() else self.audit()
        if review:
            return "Not run yet (no step used). " + review
        if not self.leads or self.left() <= 3:
            return None
        if self.refuse_blocks == 0:
            self.refuse_blocks = 1
            self.trace.append({"kind": "refuse_blocked", "leads": list(self.leads)})
            return "Not run yet (no step used). The review turned up listings that look like a fit: " + ", ".join(self.leads) + ". Open them where needed, add the one that fits best to the cart (following the pick rule) and check out. Only if every one of them plainly breaks a requirement may you end without a purchase; name the failing requirement in your text and resend the ending call."
        if self.refuse_blocks != 1:
            return None
        self.refuse_blocks = 2
        first = self.leads[0]
        self.check_cart([first], SPEND_HARD)
        if not (self.verdicts.get(self.check_key([first])) and first in self.leads):
            return None
        self.confirmed = first
        self.trace.append({"kind": "refuse_blocked", "confirmed": first})
        return f"Not run yet (no step used). An independent check of {first}'s records found it satisfies every requirement. Put it in the cart at the requested quantity and check out; end without a purchase only when its record plainly violates a requirement."

    def _cs_late_decline(self, rec: "_CallRec") -> str | None:
        if not self.is_refuse(rec.name, rec.args) or self.left() > 5 or self.late_held:
            return None
        # too late for the review: a decline now usually throws away a built cart only because time ran out
        lines = [i for i in self.built if not self.sold_out(i)]
        fresh = self.cart_view is not None and self.cart_seen_at >= self.cart_at
        missing = [i for i in lines if i not in self.cart_ids() or (i in self.dropped and not fresh)]
        if len(lines) < len(self.items) or len(missing) + 1 > self.left():
            return None
        self.late_held = True
        self.trace.append({"kind": "late_decline_held", "lines": lines, "missing": missing})
        redo = f"Put {', '.join(missing)} into the cart again at the requested quantities and then check out." if missing else "Check out the cart as it is."
        return ("Not run yet (no step used). If the store holds a fitting listing, ending without a purchase scores nothing, and a shortage of steps is no proof that none exists. "
                + redo + " Only end without a purchase when your searching showed that a requested item has no listing meeting its requirements; in that case resend the ending call.")

    def _cs_push(self, rec: "_CallRec") -> str | None:
        if not self.is_refuse(rec.name, rec.args) or self.leads or self.pushed or self.left() <= self.max_steps // 3:
            return None
        self.pushed = True
        return f"Not run yet (no step used). {self.left()} steps remain, so before ending try searches you have not used yet: browse each requested item's category and page on with the token or offset the store gives back, query single-word values and other names for the property, and open listings that could fit. Only then end without a purchase."

    def _cs_refuse_wrap(self, rec: "_CallRec") -> str | None:
        if not self.is_refuse(rec.name, rec.args):
            return None
        done = self.complete_cart() or self.order_closest()
        if done:
            return done + (" [episode finished]" if self.finished else "")
        rec.args = self.quiet(rec.name, rec.args)
        return None

    def _cs_gate(self, rec: "_CallRec") -> str | None:
        if not rec.ending or self.left() < 2:
            return None
        go, said = self.gate("order" if self.is_buy(rec.name, rec.args) else "end")
        if not go:
            return "Not run yet. " + said + "\nCart now: " + self.cart_text()
        if said:
            rec.note = "\n[Kyo] " + said + rec.note
        return None

    def _cs_execute(self, rec: "_CallRec") -> str | None:
        rec.pages_before = len(self.opened)
        rec.reply = self.act(rec.name, rec.args)
        if rec.reply is None:
            return "The call failed (network or server). You may try again."
        rec.obs = rec.reply.get("observation")
        return None

    def _cs_compare_count(self, rec: "_CallRec") -> None:
        listed = rec.args.get(rec.many) if rec.many else None
        if not (isinstance(listed, list) and len(listed) > 1):
            return None
        shown = _dump(rec.obs)
        sent = list(dict.fromkeys(str(i) for i in listed))
        got = [i for i in sent if i in shown]
        if len(got) == len(sent):
            self.compare_ok = max(self.compare_ok, len(sent))
        elif not got:
            self.compare_bad = min(self.compare_bad or 999, len(sent))
            rec.note += f"\n[Kyo] The side-by-side view showed nothing for {len(sent)} listings: send at most {self.compare_room()} per call."
        else:
            missing = [i for i in sent if i not in got]
            if missing == sent[len(got):]:
                # the first ones answered and the rest dropped: what it answered is all it takes per call
                self.compare_ok = max(self.compare_ok, len(got))
                self.compare_bad = min(self.compare_bad or 999, len(got) + 1)
            rec.note += f"\n[Kyo] The side-by-side view answered {len(got)} of the {len(sent)} listings sent. Not shown: {', '.join(missing)}."
        return None

    def _cs_record(self, rec: "_CallRec") -> None:
        name, obs = rec.name, rec.obs
        self.recent = (self.recent + [rec.mark])[-3:]
        listed = lambda v: v if isinstance(v, list) else [v]
        self.id_args.update((name, k) for k, v in rec.args.items() if listed(v) and all(isinstance(x, str) and x in self.rows for x in listed(v)))
        self.absorb(obs)
        if name == self.roles.get("compare"):
            rows = self.records_in(obs)
            if rows:
                self.facts_from[name] = self.facts_from.get(name, False) or any(_fact_block(r) for _, r in rows)
        if name in rec.readers - {self.roles.get("search")}:
            self.note_view(name, obs)
        return None

    def _cs_roles(self, rec: "_CallRec") -> None:
        name, obs = rec.name, rec.obs
        # only a search result is paged: every other job is plain
        rec.plain = {tool for job, tool in self.roles.items() if job != "search"}
        rec.wide = name not in rec.plain and not rec.ending
        rec.paged = self.more_pages(name, rec.args, obs) if rec.wide else ""
        listing = self._cs_listing_of(name, obs)
        if listing:
            self._cs_learn_views(name, obs, listing)
        return None

    def _cs_listing_of(self, name: Any, obs: Any) -> Any:
        acting = ("cart", "buy", "refuse", "ask")
        if not self.single_target(name) or any(name == self.roles.get(job) for job in acting):
            return None
        return self.ident(obs) if isinstance(obs, dict) else None

    def _cs_learn_views(self, name: Any, obs: Any, listing: Any) -> None:
        # which single-listing views carry the attribute block: a bare "open" hands its job to one that may
        self.facts_from[name] = self.facts_from.get(name, False) or bool(_fact_block(obs))
        bare = not self.facts_from[name]
        skip = lambda: {self.roles.get(job) for job in ("cart", "buy", "refuse", "ask", "search")}
        if bare and name == self.roles.get("open"):
            other = next(filter(lambda t: t != name and self.single_target(t) and self.facts_from.get(t) is not False
                                and t not in skip(), self.schemas), None)
            if other:
                self.roles["open"] = other
        if bare and name != self.roles.get("open") and not self.roles.get("offer"):
            self.roles["offer"] = name
        if name == self.offer_tool():
            self.offer_seen[listing] = self.step

    def _cs_cart_after(self, rec: "_CallRec") -> None:
        for role, after in (("cart", self._cs_after_cart), ("cart_view", self._cs_after_view)):
            if rec.name == self.roles.get(role):
                after(rec)
        if not self.is_buy(rec.name, rec.args):
            return None
        self.buys += 1
        self.refused_at = self.refused_at if rec.reply.get("finished") else self.step
        return None

    def _cs_after_view(self, rec: "_CallRec") -> None:
        self.cart_view, self.cart_seen_at = rec.obs, self.step

    def _cs_after_cart(self, rec: "_CallRec") -> None:
        self.note_cart(rec.name, rec.args)
        self.cart_said = _dump(rec.obs)
        rec.note += "\n[Kyo] Cart by Kyo's record now (listing: units): " + (_dump(self.cart_lines) if self.cart_lines else "empty")
        self.recent = []
        # a zero or negative count takes lines off (unless the reply still lists them); the fullest cart built so far
        # is kept, to be offered back when a late decline would throw it away
        counts = [c for c in map(_count, (v for v in rec.args.values() if not isinstance(v, str))) if c is not None]
        first = counts[0] if counts else None
        named = [v for v in rec.args.values() if isinstance(v, str) and v in self.rows] if counts else []
        for value in named:
            if first > 0:
                self.dropped.discard(value)
            elif not _in_rows(rec.obs, value):
                self.dropped.add(value)
        lines = [i for i in self.cart_ids() if not (self.sold_out(i) or i in self.dropped)]
        self.built = lines if len(lines) >= len(self.built) else self.built

    def _cs_filter_watch(self, name: Any, args: dict[str, Any], obs: Any) -> str:
        texts = {k: v for k, v in args.items() if isinstance(v, str) and v.strip()}
        if self.records_in(obs):
            self.filters_ok.update((name, k, v.lower()) for k, v in texts.items())
            return ""
        hint = self.label_hint(args)
        # the one text filter an empty result can be blamed on (no number filter, every other value found rows
        # before): read again once listings show its value
        suspects = [k for k, v in texts.items() if (name, k, v.lower()) not in self.filters_ok]
        lone = (name, suspects[0], texts[suspects[0]]) if len(suspects) == 1 else None
        if lone and len(texts) == len(args) and lone not in self.empty_filters:
            self.empty_filters.append(lone)
        return hint

    def _cs_body(self, rec: "_CallRec") -> None:
        name, args, obs = rec.name, rec.args, rec.obs
        pieces = [self.show(obs, ROW_CHARS if rec.wide else SHOW_CHARS), rec.paged, rec.note]
        if rec.wide:
            pieces += [self.cheapest_here([obs] + self.paged_obs), self._cs_filter_watch(name, args, obs)]
        self.learn_blind()
        rec.base = "".join(pieces)
        extras = [self.evidence(obs)]
        on_cart = name == self.roles.get("cart")
        if on_cart and any(_count(v) == 0 for v in args.values() if not isinstance(v, str)) \
                and any(isinstance(v, str) and v in self.rows and _in_rows(obs, v) for v in args.values()):
            extras.append("[Kyo] The line is still in the cart after a count of 0: this tool adds the count to the existing line, so units come off only with a negative count.")
        rec.text = rec.base + "".join("\n" + piece for piece in extras if piece)
        return None

    def _cs_tail(self, rec: "_CallRec") -> str:
        name, obs, reply = rec.name, rec.obs, rec.reply
        self.note_sold = False
        lines = self.loose_lines(obs)
        meaning = self.take_note(lines) if lines else None
        if meaning:
            rec.text += "\n[note read] " + meaning + ". Apply this change to your picks before checking out." + self.after_change()
        if self.is_buy(name, rec.args) and not reply.get("finished") and (self.note_sold or any(line in self.spoken(obs) for line in self.sold_lines)):
            seen = self.find_sold()
            if seen:
                rec.text += "\n[Kyo re-read the cart's lines after the refusal]\n" + seen
        swap = self.swap_hint()
        if swap:
            rec.text += "\n" + swap
        if self.queued:
            rec.text += "\n[notes seen on Kyo's own calls] " + " | ".join(self.queued)
            self.queued = []
        if (name in self.page_tools() and len(self.opened) > rec.pages_before and self.early_trails < EARLY_TRAILS
                and self.pending_refs() and self.cost < SPEND_SOFT and not self.finished):
            self.early_trails += 1
            chain, _ = self.trail(early=True)
            if chain:
                rec.text += "\n" + chain + self.fits_now()
        if name in rec.readers:
            self.read_cache[rec.mark] = (self.version, self.step, rec.base)
        return rec.text + (" [episode finished]" if reply.get("finished") else "")

    def finish(self, again: bool = False) -> str:
        """End the episode in up to three tries (nearest listings, checkout, ending without a purchase); the first
        try that settles it gives the text. With `again`, a refused checkout comes back as the store's reply for the
        agent to deal with, instead of being followed by an ending without a purchase."""
        if self.finished or self.broke or self.left() <= 0:
            return ""
        settled = (said for said in (end(again) for end in (self._end_closest, self._end_buy, self._end_refuse)) if said is not None)
        return next(settled, "")

    def _end_closest(self, again: bool) -> str | None:
        if not (self.fallback_mode() and self.lines_missing() > 0 and self.closest_tries == 0):
            return None
        done = self.order_closest()
        if self.finished or self.left() <= 0:
            return ""
        return done if done and again else None

    def _end_buy(self, again: bool) -> str | None:
        buy = self.roles.get("buy")
        fresh = self.cart_view is not None and self.cart_seen_at >= self.cart_at
        holding = bool(self.cart_ids()) if fresh else bool(self.cart_lines)
        # a cart short of a line is never paid; when the nearest listings fall short too, the ending without a purchase can be
        if not buy or not (holding or (self.cart_touched and not fresh and self.left() >= 2)) or (self.nothing_fits and self.lines_missing() > 0):
            return None
        faults: list[str] = []
        args = _shape(dict(self.brief.get("buy_args") or {}), self.schemas.get(buy), "args", faults)
        if faults:
            return None
        # a reply that adds a requirement does not stop a forced finish: no steps are left to act on it
        if self.left() >= 2:
            self.gate("order")
        if self.finished or self.left() <= 0:
            return ""
        reply = self.act(buy, args)
        self.trace.append({"kind": "forced_buy", "reply": _clip(reply, 200)})
        if self.finished or self.left() <= 0:
            return ""
        if reply is None or not again:
            return None
        self.buys += 1
        self.refused_at = self.step
        self.observe(buy, reply.get("observation"))
        return "the checkout was refused: " + self.show(reply.get("observation"))

    def _end_refuse(self, again: bool) -> str | None:
        refuse = self.roles.get("refuse")
        if not refuse:
            return None
        faults: list[str] = []
        args = _shape(self.quiet(refuse, dict(self.brief.get("refuse_args") or {})), self.schemas.get(refuse), "args", faults)
        if faults:
            return None
        if self.left() >= 2:
            self.gate("end")
        if self.finished or self.left() <= 0:
            return ""
        reply = self.act(refuse, args)
        self.trace.append({"kind": "forced_refuse", "reply": _clip(reply, 200)})
        if reply is None or self.finished:
            return None
        self.observe(refuse, reply.get("observation"))
        return "ending without a purchase was refused: " + self.show(reply.get("observation"))

    def trim(self) -> None:
        if sum(len(_dump(m.get("content"))) for m in self.chat) <= HISTORY_LIMIT:
            return
        results = [m for m in self.chat if m.get("role") == "tool"]
        for message in results[: max(1, len(results) // 2)]:
            content = message.get("content") or ""
            if len(content) <= 600:
                continue
            found = dict.fromkeys(re.findall(r'"([A-Za-z0-9][A-Za-z0-9_\-:.]{5,40})"', content))
            ids = [i for i in found if i in self.rows][:20]
            message["content"] = content[:400] + f"...(older result shortened; listings in it: {', '.join(ids)})"

    def run(self) -> list[dict[str, Any]]:
        if self.finished:
            return self.trace
        # phases: brief + opening -> main rounds -> (retry round, final ending) while the episode is still open
        self.chat = [{"role": "system", "content": GUIDE}, {"role": "user", "content": self._phase_intro()}]
        self.work(LOOP_CALLS, SPEND_HARD)
        for phase in (self._phase_retry, self.finish):
            if self.finished:
                break
            phase()
        self.trace.append({"kind": "end", "finished": self.finished, "steps": self.step, "cost": round(self.cost, 5), "calls": self.calls})
        return self.trace

    def _phase_intro(self) -> str:
        self.read_request()
        self.trace.append({"kind": "brief", "brief": self.brief})
        opening = self.view.get("opening")
        if opening is not None:
            self.absorb(opening)
        parts = ["SHOPPER'S REQUEST:\n" + self.request]
        if opening is not None:
            parts.append("WHAT THE STORE SHOWS FIRST:\n" + self.show(opening))
        followed = self.follow()
        if followed:
            parts.append(followed)
        parts.append("First search every item's category and open a number of listings in full, so the interviewer learns which properties these listings have.\n\n" + self.status())
        return "\n\n".join(parts)

    def _phase_retry(self) -> None:
        turned = self.finish(again=True)
        # the store turned the ending down (a change from the shopper, a sold-out line): a short second round deals with it
        if not turned or self.finished or self.broke or self.no_credit or self.left() <= 1:
            return
        self.chat.append({"role": "user", "content": "[Kyo] Kyo tried to end the episode and " + turned + "\nApply what the store says, then finish.\n" + self.status()})
        self.work(self.loop_calls + 10, SPEND_HARD + 0.01)

    def work(self, loop_cap: int, spend_cap: float) -> None:
        # each round runs the turn stages in order; a stage answers None (go on), "next" (new round) or "stop"
        turn: dict[str, Any] = {"idle": 0}
        stages = (self._turn_interview, self._turn_closest, self._turn_model, self._turn_dispatch)
        while not self.finished and not self.broke and self.left() > 0 and self.loop_calls < loop_cap and self.cost < spend_cap:
            if next((said for said in map(lambda stage: stage(turn), stages) if said is not None), None) == "stop":
                break

    def _turn_interview(self, turn: dict[str, Any]) -> str | None:
        if not self.ready_to_ask():
            return None
        summary = self.interview()
        if summary:
            self.chat.append({"role": "user", "content": summary + "\n" + self.status()})
        return "next"

    def _turn_closest(self, turn: dict[str, Any]) -> str | None:
        if not self.closest_due():
            return None
        done = self.order_closest()
        if self.finished:
            return "stop"
        if done:
            self.chat.append({"role": "user", "content": done + "\n" + self.status()})
        return None

    def _turn_model(self, turn: dict[str, Any]) -> str | None:
        self.trim()
        self.loop_calls += 1
        turn["message"] = self.think(self.chat, [LIGHT, SPARE], 3000, self.offered(), plain=self.plain_loop and self.loop_calls > LATE_PLAIN)
        if turn["message"] is not None:
            return None
        self.misses += 1
        return "stop" if self.no_credit or self.calls >= ALL_CALLS or self.misses > 2 else "next"

    def _turn_call(self, number: int, call: dict[str, Any]) -> dict[str, Any]:
        fn = call.get("function") if isinstance(call.get("function"), dict) else {}
        raw = fn.get("arguments")
        parsed = raw if isinstance(raw, dict) else (_json_obj(raw) if isinstance(raw, str) and raw.strip() else {})
        return {"id": str(call.get("id") or f"k{self.loop_calls}_{number}"), "type": "function",
                "function": {"name": str(fn.get("name") or ""), "arguments": _dump(parsed if isinstance(parsed, dict) else {})}}

    def _turn_dispatch(self, turn: dict[str, Any]) -> str:
        message = turn["message"]
        calls = [self._turn_call(number, call) for number, call in enumerate(message.get("tool_calls") or []) if isinstance(call, dict)]
        content = message.get("content")
        content = content if isinstance(content, str) else (_dump(content) if content else "")
        turn["idle"] = 0 if calls else turn["idle"] + 1
        if not calls:
            self.chat += [{"role": "assistant", "content": content or "(no call)"},
                          {"role": "user", "content": "Call one of the tools to continue.\n" + self.status()}]
            return "next" if turn["idle"] < 3 else "stop"
        self.chat.append({"role": "assistant", "content": content or None, "tool_calls": calls})
        replies = self._turn_ask_first(calls) or self._turn_run(content, calls)
        self.chat.extend(replies)
        return "next"

    def _turn_ask_first(self, calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        # a cart or ending call before the shopper was ever asked: hold the whole turn and run the interview
        committing = {self.roles.get(job) for job in ("cart", "buy", "refuse")} - {None}
        due = not self.asked and self.roles.get("ask") and self.msgs_free() > 0
        if not (due and any(c["function"]["name"] in committing for c in calls) and self.left() > 8):
            return []
        self.chat.extend({"role": "tool", "tool_call_id": c["id"], "content": "Not run (no step used): the shopper is asked first."} for c in calls)
        summary = self.interview()
        return [{"role": "user", "content": (summary or "The interviewer had nothing to ask.") + "\n" + self.status()}]

    def _turn_run(self, content: str, calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        self.trace.append({"kind": "turn", "text": content[:300], "calls": [c["function"] for c in calls]})
        replies = self.run_calls(calls)
        tags = [""] * (len(replies) - 1) + ["\n" + self.status()] if replies else []
        return [{"role": "tool", "tool_call_id": cid, "content": text + tag} for (cid, text), tag in zip(replies, tags)]


class _CallRec:
    """One tool call on its way through Kyo.run_calls' stages: what was asked, what the stages decided, and the
    store's reply once it ran."""

    def __init__(self, call: dict[str, Any], others: set, total: int) -> None:
        self.call = call
        self.cid = call["id"]
        self.others = others
        self.total = total
        self.allowed: set = set()
        self.name: Any = None
        self.raw: Any = None
        self.args: dict[str, Any] = {}
        self.note = ""
        self.many: Any = None
        self.target: Any = None
        self.ending = False
        self.mark = ""
        self.readers: set = set()
        self.pages_before = 0
        self.reply: Any = None
        self.obs: Any = None
        self.plain: set = set()
        self.wide = False
        self.paged = ""
        self.text = ""
        self.base = ""


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

