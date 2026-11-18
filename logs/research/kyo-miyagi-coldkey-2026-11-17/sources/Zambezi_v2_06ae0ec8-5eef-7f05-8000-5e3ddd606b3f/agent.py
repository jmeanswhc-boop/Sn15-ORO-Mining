from __future__ import annotations

import collections
import itertools
import json
import re
import time
from typing import Any, Callable

from src.agent.proxy_client import ProxyClient

MAIN_MODEL = "anthropic/claude-haiku-5.5"
BACKUP_MODEL = "openai/gpt-6-luna"
SETUP_MODEL = "anthropic/claude-haiku-5.5"
CHEAP_MODEL = "openai/gpt-oss-120b"
LAST_MODEL = "anthropic/claude-sonnet-5.5"   # only when every cheaper seat has failed
READ_MODEL = "anthropic/claude-haiku-5.5"

TASK_SECONDS = 265.0
COST_SOFT = 0.06            # spend per episode beyond which checks move to the cheaper LLM
COST_STOP = 0.08            # spend per episode at which the episode is wound up (raised a little per extra product)
COST_PER_PRODUCT = 0.01
EVENT_RESERVE = 3
SHORTLIST = 3
JUDGED = 16
WRAP_UP_SECONDS = 70.0
FORCE_SECONDS = 28.0
OBS_CHARS = 16000
MAX_TURNS = 80
SWEEP_LAST = False

SYSTEM_PROMPT = """Role:
You act as a careful buying assistant for a single shopper inside a simulated shop, and the tools supplied are your only means of acting. Grading is all or nothing. A task earns its point only when the finish is completely correct: the purchased item satisfies each condition written in the request, each hidden need the shopper discloses once asked, and each change announced while you are busy; the number of units and the spending limit are respected; where the request calls for the cheapest (or best) pick, that pick is made; and each working rule the request sets is obeyed. Should no item in the shop be able to satisfy all of that, the correct ending is to decline, done precisely as the request instructs.

Input:
Your first user message brings the shopper's request, any opening observation the shop gives, a first reading of the request prepared by a helper, and a status line. Tool results come back as text; they can end with a status line and with "[note]" lines added by the code around you, and those must be read and obeyed.

Rules:
The shop
1. Any tool call spends one step, including calls that fail or come back with an error, and steps run out. Reasoning costs nothing; steps cost.
2. Calls that do not depend on each other's results may travel together in one reply (say, opening several listings' details at once). They execute in the order given. A purchase or a decline always travels alone.
3. Where some tool shows complete details for several listings in a single call, use it rather than opening listings separately, since that spends fewer steps.
4. Each shop names its tools, parameters and result fields in its own way. Study the tool descriptions and their JSON schemas, and pass exactly the parameter names and permitted values declared there. Let the first results teach you where each fact lives (price, stock, detailed properties).
5. Early on, learn the category labels this shop puts on the requested kind of product (they appear on listings). If similar labels sit side by side, choose the one whose name fits the request's wording taken as a whole, then search and buy inside it.
6. Searches may be limited: a ceiling on rows or pages shown, ordering by id instead of relevance or price, every word mandatory, filters valid in a single mode only. Set the reported total against the rows really visible, and keep tightening (more precise words, category or price filters, later pages) until the whole relevant set has passed before your eyes.
The shopper and what they keep back
7. The shopper has unwritten needs and discloses one only when a message asks about that very property. Only the FIRST thing a message asks receives an answer, so each message covers exactly ONE property, phrased as "Which <property> would you like the <product> to have?", with the property spelled as the listings spell it, the product named by its plain noun from the request, and no example values. Never give instructions, never mention lists, labels, positions or rules, and never ask two things at once.
8. With a request for several different products, each question must state its product; leaving the product open bewilders the shopper. Distribute the questions among the products.
9. Should the shopper say they did not follow, or want your message explained, simplified or rephrased, do not explain and do not resend it: they most probably do not care about that property. Turn to some other property next, naming the product.
10. Answers such as "don't mind", "no preference" or "anything is fine" mean the property is open. An answer giving a preference or a limit becomes a firm condition from then on, worded exactly as given, numeric limits included.
11. Conditions come only from the request's statements and from what the shopper reveals or alters. Do not derive one from the story (an age group, a gender, a size, who will use it); such details are not conditions and no listing fails on account of them. If such a detail could matter, it may hide a need, so ask about it.
12. If a planner is running, it chooses and sends your questions for you the moment you have opened the details of two or three listings of the requested kind (for each kind, when the request names several different products), so open them among your first moves. The answers will then appear in the conversation along with what they mean; make every revealed need a firm condition and search again with it plus the stated values.
13. Messaging the shopper is reserved for questions (and for answering a question the shopper put to you). Never write to report what you found or bought, or that you are declining; the finishing tool conveys that.
14. If you must do the asking yourself, first open the details of two or three listings, then ask about properties the story points to (who it is for, how it must look, where or how it will be used), and after that about properties the listings carry with differing values among candidates. Skip anything the request already decides, and stay inside the message allowance.
Working rules set by the request
15. A request may prescribe how you work: how many listings to inspect or compare before deciding, what to verify, what to tell the shopper, what a remark must contain, how to decline. Such rules are graded as strictly as conditions, so carry out each one literally and completely. A rule to inspect or open N listings means N listings that fit in full, each opened through the details tool.
16. If the shopper asks you something (to confirm a change, for instance), a reply goes out for you while messages remain; the change itself is binding on you.
17. Mere wishes ("ideally", "would be nice", "if possible") are not conditions: they can only settle a choice between items that already satisfy everything, and they never override a cheapest rule.
18. Should the request leave open a point only the shopper can decide (no price limit given, or two wishes that might clash, where the shopper has to say which counts more), ask about it before you decide.
Changes along the way
19. During your work the shopper may alter things: a fresh budget (larger or smaller), a condition removed, swapped or added, a price increase, an item running out. A single shopper line may hold several such changes. These lines can surface inside any tool result (an extra field of sentences) or arrive with a purchase attempt. Each change applies from the moment it appears: test your choice and your backups against it again, and if the request seeks the cheapest or best, pick anew from the updated acceptable set.
20. An order attempt can be REFUSED. The result then carries the reason in some field (a refusal, complaints, objections or similar), even when the inner result looks successful (accepted: true, placed: true, executed: true). A purchase that succeeds closes the episode, so while the episode keeps going nothing has been bought, no matter what the result says. The episode goes on and the cart stays as it was. Read the refusal, apply every change it mentions, take out of the cart whatever is no longer acceptable (a sold-out item, say), put in the next verified fit, look at stock and price once more, and try the purchase again.
21. Stock and prices may also shift without warning. Hence stock and price are checked anew just before each purchase attempt; once current price times quantity goes over the current budget, the item no longer fits.
Judging a listing (this is what the grade rests on)
22. A requirement is met only when the listing's own property for it states the value. Property keys vary in spelling (spaces, dots, underscores, prefixes), and the same property may appear under several keys; any key that names that property counts. Values match by meaning: one element of a comma-separated list is enough; singular and plural are equivalent; near synonyms count; capitalisation is irrelevant. A word that merely shares its opening letters does not count, nor does a wider or different class (for example girls is not women, plain unisex is not female).
23. Fall back on the title or description only when the listing lacks any property key for that condition. When the listing's own property states something different, it fails even if its title or description carries the value.
24. A value stored under some other, related property does not count (a length field cannot satisfy a style condition; the lining's material cannot satisfy the material). An options or variant field is the exception: it names the variant the listing sells, so it can state a property the listing has no key for (a colour, say); where the listing has that key and its options give a value for that same property that the key does not list, the two disagree and the requirement is not met; options that pick one of the key's own values, or speak of something else, leave the key to decide.
25. Numbers are compared numerically after unit conversion; the boundary value itself satisfies a limit; a measured range (a spec that spans values) qualifies only if every part of it meets the limit. Variants on offer are different: when a listing sells several sizes, lengths or capacities to choose from, it meets a limit as soon as one offered variant does.
26. Respect exclusions ("leave out anything with X") and closed lists ("A or B and nothing else").
27. The requested kind of product is satisfied by the listing's category; never demand a property that echoes the category's name. The item must really be that kind: where the shop assigns categories, it must belong to the one whose label fits the request's wording as a whole (a request for "A or B" matches a category called "A & B", not a narrower neighbour called "B"). When the request wants goods on hand it must be in stock, and price times quantity may not exceed the budget (being equal is fine).
28. When a listing has options or variants (sizes, colours, versions) with ids, prices or stock of their own, treat each option as its own candidate: the option's own value settles each condition, its own price and stock are what count, and the cart line must identify the selected option in the form the cart tool's schema expects.
29. A property listing several choices that include the required one (a comma list) is a plain match, not conflicting evidence. Evidence conflicts only when the listing's own keys for that property give incompatible values. In that case alone, and only when no cheapest or best rule decides, lean toward a candidate with consistent evidence.
Cheapest or best
30. When the request asks for the cheapest, the order must be the lowest-priced item among ALL items that meet every requirement; one fitting item seen is not sufficient. Use price filters when a tool offers them, lower the price limit until every remaining row is visible, rank the visible candidates by price yourself, and verify them in ascending price; the first one that fully fits, with every cheaper one checked and rejected, is the answer. Hold the following fits in reserve as backups.
31. For a best request on some measure (most, largest, longest, highest and so forth), weigh ALL fitting items on that measure in the same way and take the extreme one. An item whose listing states no figure for the measure cannot be chosen as best.
Finishing
32. Maintain a ranked set of verified fits with a backup or more, as your first choice might run out at checkout.
33. Before every order attempt: the cart holds exactly the lines the request needs (usually one: the chosen item at the required quantity; one line per item when the request asks for several different items) and nothing else (remove any other line), and that after your latest cart change you have looked up the chosen item's stock and price again and found it available and affordable.
34. Declining is rarely correct: most tasks have an item in the shop that satisfies every condition, and a mistaken decline earns 0. Decline only after searching thoroughly, and only if for each candidate you can cite a stated condition, a revealed need, the budget or stock that its own records plainly miss. Uncertainty, an absent nice-to-have or a condition you merely presumed does not justify a decline; buy the best verified fit instead. Carry out the request's decline instructions to the letter. If the request says not to offer substitutes, the decline remark must not name, describe or point to any listing: no ids, no titles, no prices.
35. Ration your steps: send your first purchase attempt while seven steps or more remain, leaving room to recover from a refusal. Once only 7 steps are left, only cart, stock-check and finishing tools can still be used. Settle once; do not swap cart contents back and forth, because each cart change spends a step.

Output:
Each reply carries at least one tool call. Keep any text short: the conditions as they now stand, the verified fits with price and stock, and what you will do next."""

SETUP_PROMPT = """Role:
You turn a shopper's request and a shop's tool list into a structured summary that a buying assistant works from.

Input:
The shop's tools come after "Store tools:"; the request comes after "Shopper request:", possibly followed by an "Opening observation:".

Rules:
1. Account for each sentence of the request: it is a condition, the priority, a soft want, a rule, a budget, a hint, or mere story. When torn about whether a sentence is a rule, file it under "rules".
2. Fill every key listed under Output, using the exact tool names from the list wherever a tool is named.

Output:
A single compact JSON object (no indentation) with exactly these keys and no other text:
- "product": what kind of product is wanted, using the request's wording (naming each one if several different items are wanted).
- "quantity": units of a single listing to buy, as an integer, defaulting to 1 if the request asks for no more. Anything ordinarily sold as one thing, like a pair of shoes or a set, counts as 1.
- "items": for a request covering several DIFFERENT products, one object for each, shaped {"product": its short name taken from the request, "quantity": units of it (integer)}; an empty list when only one kind is wanted.
- "requirements": strings, one per firm condition the request puts on the product, each shaped "property: constraint" and carrying the shopper's exact value, limit, exclusion or closed list. If several different products are wanted, prefix each with the short name of its product, as "<product> <property>: constraint". Do not include price, stock, quantity, budgets or mere wishes here, and do not restate the product kind or category as a requirement: words that are part of the category's name (a season, an occasion or an audience in the category name) are covered by the category and are not a separate property requirement. A condition whose value is to be read from another listing is one requirement, "<property>: as stated on the listing for the <kind> sold by <seller>"; the details that identify that other listing (whatever properties are named for it) describe that listing, not the product, so they are never requirements.
- "price_cap": the top total price as a number, or null if none is given.
- "currency": the unit of price the request uses, or null.
- "in_stock_only": true if the request accepts only goods actually available or in stock.
- "priority": "cheapest" only when the request explicitly asks for the cheapest or lowest-priced fitting option; "best" only when it explicitly asks for the best by some measure; "none" in every other case. Comments that funds are short or that each penny matters do not ask for the cheapest; they merely explain the price cap.
- "priority_detail": with "best", the measure that defines it; otherwise "".
- "message_budget": how many messages the shopper permits, as an integer, or null if unstated.
- "step_budget": how many moves or steps the request permits, as an integer, or null if unstated.
- "rules": each procedural instruction the request gives the assistant (how many listings to inspect, what to verify or report, how to decline, what to avoid, what to include), copied word for word. Statements about what the shopper intends to do personally are story, not rules.
- "soft_wants": what the shopper would like without insisting (worded as ideally, preferably, would be nice, if possible). Never put these in "requirements".
- "lookups": when the request gives a condition's value not outright but as whatever some other listing states (naming that listing by its kind, its seller and a few of its details), one object per such listing: {"kind": its kind of product, "seller": the seller's name as written, "details": the details named for it}; [] otherwise. Those details only identify the other listing: never turn them into requirements for the product to buy; the requirement is only the property whose value that listing gives.
- "ask_first": points only the shopper can settle, to be asked before deciding: say the budget when there is no price limit, which of two wishes counts more when the request admits both may be impossible (in that case also place both wishes in soft_wants rather than requirements), or anything the request says to check with the shopper. An empty list if nothing applies.
- "decline_no_names": true if the request's decline rule forbids offering, naming or suggesting substitutes or alternatives; false if it asks for something to be named; null if it sets no such rule.
- "roles": an object giving, for each of the roles "search", "details", "view", "message", "cart", "cart_view", "order", "decline", "browse", "batch_view", the exact tool name from the list, or null when no tool suits. "message" writes to the shopper; "cart" puts things into or takes them out of the cart; "order" buys or checks out; "decline" closes the task with no purchase; "browse" lists one category's listings (often with price or brand filters); "batch_view" opens several listings in one call. One tool may fill several roles.
- "store_args": how to call the store's lookup tools, as {"search": {"tool", "query", "page", "fixed"}, "browse": {"tool", "category", "max_price", "brand", "page", "fixed"}, "view": {"tool", "id"}, "batch_view": {"tool", "ids"}}: each value names the tool's own parameter for that purpose (null when it has none), and "fixed" maps any other parameter to the value that selects that use (a mode switch, say); null for a lookup the store lacks.
- "finish_values": null when buying and declining are separate tools. When one tool does both and a parameter with fixed allowed values picks which, give {"order": the allowed value that buys the cart, "decline": the allowed value that ends the task with nothing bought}, each copied exactly from the schema.
"""

INTERPRET_PROMPT = """Role:
You support a buying assistant by working out what the shopper has just said.

Input:
Below follow "Shopper request (context only):", "Requirements so far:", perhaps "Earlier exchanges with the shopper (oldest first):" and "The assistant asked:", and last "The shopper now said:".

Rules:
1. If several different products are wanted, prefix each condition in "needs" and "requirements_now" with the short name of its product, as "<product> <property>: constraint" (shoppers often phrase it as "<property> of the <product kind>").
2. One line can contain several changes; report all of them. A condition swapped for another is both dropped (the former) and a new need (the latter).
3. When the shopper picks which of two wishes counts more, the favoured wish becomes a need. Any budget the shopper names belongs in new_budget. A simple wish ("ideally", "would be nice") is not a need.
4. Take the shopper's words literally and in good faith. Homely, figurative or indirect phrasing that names a value means the shopper definitely wants it; treat a value as excluded only when the shopper clearly says to steer clear of it. A number with a comparison (at most, at least, under, over, no more than) is a limit, kept with its unit.
5. A fresh budget overrides the previous one, larger or smaller, even if the shopper presents it as good news.
6. Name each need by the property the shopper's own words name, not by the one the assistant asked about: an answer about "weight" to a question on load capacity is a weight condition, and an answer naming a broader or different property keeps that property.
7. If the shopper merely says they don't mind, have no preference, or anything will do, put the property that was asked about into no_preference. Such an answer touches only that asked property: it never removes a stated condition, a price limit or a rule, and if the assistant's message asked about no property it alters nothing.

Output:
A single JSON object with exactly these keys and no other text:
- "meaning": one brief sentence saying what the shopper intends.
- "needs": newly stated firm conditions on the product, each shaped "property: constraint" with the precise value, limit, exclusion or closed list the shopper used.
- "no_preference": properties the shopper declared they are indifferent to.
- "dropped": earlier conditions the shopper has let go.
- "new_budget": the fresh top total price as a number, or null.
- "new_quantity": if the shopper changes the number of units, {"product": which product it concerns (or "" if only one kind is wanted), "quantity": the new figure as an integer}; null otherwise.
- "unavailable": true if the shopper reports that an item can no longer be bought (sold out, gone, out of stock, button disabled); false otherwise.
- "requirements_now": every firm condition on the product as things stand after all the shopper has said, assembled from the requirements so far and the shopper's explicit words, each shaped "property: constraint". Never insert a condition drawn from the story or from your own interpretation of the request: only values the request gives as conditions, or the shopper gives in an answer, qualify. Keep every earlier requirement unless the shopper dropped or replaced it; for a replaced value keep only the newest one; add every newly revealed need. Exclude price, stock, quantity and the product category.
- "lookups": when the shopper gives a need's value not outright but as whatever some other listing states (naming that listing by its kind, its seller and a few of its details), one object per such listing: {"kind": its kind of product, "seller": the seller's name as written, "details": the details named for it}; [] otherwise. Those details only identify the other listing: never turn them into needs for the product to buy.
- "unclear": true if the shopper says they did not follow or grasp the assistant's message, or wants it explained, simplified, rephrased or repeated; false otherwise.
- "reply": if the shopper puts a question to the assistant or wants something confirmed, a brief plain answer to send back that addresses it and states the change itself (confirming the new value, say); otherwise "". It must be "" whenever "unclear" is true, since asking to have the assistant's own question explained is not a question needing an answer."""

PLAN_PROMPT = """Role:
You choose the questions a buying assistant puts to a shopper. The shopper keeps hidden needs, each being one property's value, disclosed only when a message asks about precisely that property; a message gets an answer only about its first question, and asking about anything else just earns a courteous "don't mind".

Input:
Below follow "Shopper request:", "Requirements so far:", perhaps a note that several different products are wanted, perhaps "Already asked (with the shopper's answers):", perhaps "Must be learned from the shopper:", and "Properties in the listings' details seen so far (name, listing count, example values):".

Rules:
1. Select QUESTION_COUNT distinct properties (fewer only if not enough useful ones exist), strongest candidate first.
2. Top priority: anything in the "Must be learned from the shopper" block not yet asked; ask it in plain words. For the budget, ask for the most they are willing to spend. For a choice between two wishes, state both values together with their properties, such as "If one listing can't have both, would you rather have the <value> <property> or the <value> <property>?".
3. After that: properties the listing details below truly contain, favouring (a) those present in most listings with values that vary across them, of whatever sort (material, appearance, construction or operation, intended user, technical specifications), because a property sharing one value across every listing seen makes a weak question; and (b) a property the request's story points toward (who will use it, how it must look, where or how it will be used).
4. Avoid logistics (warranty, condition, pack size, years) and values that only list everything on offer (all sizes or variants in one string), and never ask about the brand, maker, seller or shop. A size, colour or similar option held by a listing, or by one of its options, as its own single value is a normal property.
5. Do not select a property the request already decides (its stated conditions, price, stock, quantity). Do not ask again about a property already covered for the same product; a question the shopper failed to follow counts as covered, so go on to another, unless that question did not name the product. Do not use two questions on near-identical properties (two sorts of style, two spellings of the same property), and skip properties absent from every listing. Name properties as the listings name them.
6. If several different products are wanted, each question concerns ONE of them and the questions are spread among the products: favour a product asked about less so far, and take its properties from listings of that product's kind (the menu is grouped by listing labels where present).
7. Phrase each question as "Which <property> would you like the <product> to have?", using the property's name exactly as in the listings (spaces replacing underscores or dots) and the product's plain noun from the request ("the rug", "the curtains"; never with adjectives, materials, colours or other property values). No sample values, no additional words, no second question. A point to be learned first (rule 2) is asked plainly in a single short sentence.

Output:
A single JSON object and no other text: {"questions": [{"product": "...", "property": "...", "question": "..."}, ...]}."""

SWEEP_NOTE = """

Extra rule: your list ends with a sweep question. Phrase it as "Do you have a requirement for the <product>'s <property>, <property> or <property>?", covering two or three further promising properties of ONE product that none of the other questions touch (spelled as the listings spell them, the product by its plain noun). Its "property" field gives them separated by ", "."""

AUDIT_PROMPT = """Role:
You review a buying assistant that is on the point of telling the shopper the shop has nothing suitable. A mistaken decline earns 0.

Input:
Below follow "Shopper request:", "Current requirements:" and "Listing records seen (newest first):".

Rules:
1. Compare the listing records with the conditions.
2. A condition holds when the listing's own property for it shows the value; any key spelling, a single entry of a comma list, singular/plural and near synonyms are all acceptable, whereas a wider or different class is not.
3. The title or description is usable only when the listing has no key for that property.
4. Numbers are compared after unit conversion, and the boundary satisfies a limit.
5. The price has to fit the budget and the item may not appear out of stock.
6. Only the listed conditions matter, never one presumed from the story; with several different products wanted, each listing is assessed only against the conditions of its own kind.
7. Include every listing that might fit; omit one only when its own records plainly miss a listed condition.

Output:
A single JSON object and no other text: {"possible_fits": [ids of listings that appear to meet every requirement, best first, at most 5], "reason": "one sentence"}."""

FINISH_MODE_PROMPT = """Role:
You read one tool of an online shop for a buying assistant.

Input:
The tool's name, description and parameters follow "Tool:".

Rules:
1. One parameter takes fixed allowed values, and they decide how the task ends.
2. Name the value that buys what is in the cart and the value that ends the task with nothing bought, each copied exactly from the allowed values.

Output:
A single JSON object and no other text: {"order": "...", "decline": "..."}."""

_llm = ProxyClient(timeout=70, max_retries=1, retry_delay=1)
_env = ProxyClient(timeout=95, max_retries=2, retry_delay=1)

def any_of(*parts) -> str:
    return "|".join(parts)


def word_group(*parts) -> str:
    return r"\b(" + any_of(*parts) + r")\b"


FACT_TABLE_KEY = re.compile(any_of("fact", "propert", "trait", "spec", "attribut", "feature", "detail", "characteristic"), re.I)
# refusal wording that means the item cannot be had any more
GONE_WORDS = re.compile(any_of(
    "sold out", "out of stock", "unavailable", "no longer", "zero left", "none left", "all gone", "is gone", "it's gone",
    "now gone", "gone from stock", "dried up", "ran dry", "run dry", "cleared out", "cleared from", "bought the last",
    "can't be ordered", "cannot be ordered", "grayed out", "greyed out", "hit zero", "no units", "none remaining",
    r"no .{0,12}remaining", "been sold", "been taken", "snapped up", "in stock any ?more", "not in stock", "isn't in stock",
    "no stock", r"stock.{0,20}\bzero\b", "last one"), re.I)
PRICE_TALK = re.compile(any_of("price", "cost", "charg", "priced", "bumped", "gone up", "went up", "come down", "dropped", "cents",
                               "dearer", "cheaper"), re.I)
# names that describe the listing itself, kept away from the question menu
SELLER_WORDS = ("brand", "maker", "manufacturer", "seller", "vendor", "merchant", "headline", "heading", "sku", "url", "title",
                "made by", "sold by", "blurb", "tagline", "caption", "teaser", "slogan")
WHOLE_META = ("make", "shop", "shop name", "store", "store name", "outlet", "company", "category", "department", "shelf",
              "description", r"(overview|description|summary)" + r"( text)?", "notes?", "image", "images", "rating", "reviews?",
              "review count", "link", "options?", "variants?", "id", "name", "price", "amount", "cost", "stock", "on hand", "in stock")
META_NAMES = re.compile(word_group(*SELLER_WORDS) + "|^(" + any_of(*WHOLE_META) + ")$", re.I)
ALLOWANCE_SPENT = re.compile(any_of("budget_exceeded", "quota", "allowance", "used up", r"no more\b.{0,40}" + r"\b(may|can|allowed|left)",
                                    "limit reached", "reached the limit", "maximum (number|reached)", "already used"), re.I)
IDENT_KEY = re.compile(any_of("id", "ref", "sku", "code", "number", "key", "listing", "product"), re.I)
OPTION_KEY = re.compile(any_of("option", "variant", "choice", "config", "version", "edition", "sku", "model"), re.I)
KIND_KEY = re.compile(any_of("categor", "department", "^dept$", "shelf", "aisle", "section", "^kind$", "product.?(type|family|group|kind)",
                             "item.?(type|kind)", "offer.?kind", "^family$", "^group$"), re.I)
FIRST_PERSON = re.compile(word_group("I", "I'm", "I'd", "I've", "I'll", "my", "me", "mine"))
ASKING = re.compile(r"\?|" + word_group("confirm", "let me know", "tell me", "acknowledge", "reply", "answer", "get back to me"), re.I)
TELL_RULE = re.compile(word_group("tell", "inform", "let (me|them|the shopper) know", "update", "confirm", "report", "mention",
                                  "explain", "message"), re.I)
FINISH_TOOL_NAME = re.compile(any_of("final", "confirm", "complete", "checkout", "pay", "place", "submit", "commit", "token"), re.I)
PENDING_KEY = re.compile(any_of("token", "pending", "await", "confirm", r"next.?step", r"checkout.?id", "session"), re.I)
ID_SHAPE = re.compile(r"^[A-Za-z0-9]" + r"[A-Za-z0-9_\-:.]{3,63}$")
DIGIT = re.compile(r"\d")
CODE_TEXT = re.compile(r"[A-Za-z0-9_\-]+")

NEUTRAL_DECLINE = "Sorry, nothing in the store meets all of your requirements, so I am not ordering anything."
NO_STEP = "Not run (no step used)"
HELD_BACK = "Not run yet (no step used). "


def dump(value) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        text = str(value)
    return text if len(text) <= OBS_CHARS else text[:OBS_CHARS] + "...(cut)"


def clip_strings(value, limit, depth=0) -> Any:
    if depth > 8:
        return None
    if isinstance(value, str):
        return value[:limit] + "..." if len(value) > limit else value
    deeper = depth + 1
    if isinstance(value, list):
        return [clip_strings(child, limit, deeper) for child in value]
    if isinstance(value, dict):
        return dict(zip(value, (clip_strings(child, limit, deeper) for child in value.values())))
    return value


def is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def coded(text) -> bool:
    return bool(ID_SHAPE.match(text)) and any(ch.isdigit() for ch in text)


def holds_words(value, depth=0) -> bool:
    if depth > 4:
        return False
    if isinstance(value, str):
        return any(ch.isalpha() for ch in value) and not coded(value)
    if isinstance(value, dict):
        value = list(value.values())
    if not isinstance(value, list):
        return False
    return any(holds_words(child, depth + 1) for child in value)


def gather_sentences(value, out, depth=0) -> None:
    if isinstance(value, str):
        if depth <= 4 and len(value.split()) >= 4:
            out.append(value)
    elif isinstance(value, list) and depth <= 4:
        for child in value[:10]:
            gather_sentences(child, out, depth + 1)


def harvest_ids(value, out) -> None:
    pending = [(value, 0)]
    while pending:
        node, depth = pending.pop()
        if depth > 6:
            continue
        if isinstance(node, list):
            pending.extend((child, depth + 1) for child in node)
        elif isinstance(node, dict):
            for key, child in node.items():
                if isinstance(child, str) and "id" in key.lower() and ID_SHAPE.match(child):
                    out.add(child)
                else:
                    pending.append((child, depth + 1))


def awaits_next_step(observation) -> bool:
    sentences = []
    gather_sentences([v for v in observation.values() if isinstance(v, (list, str))], sentences)
    if any(len(sentence.split()) >= 8 for sentence in sentences):
        return False
    keys = list(observation) + [inner for v in observation.values() if isinstance(v, dict) for inner in v]
    return any(PENDING_KEY.search(str(key)) for key in keys)


def sentence_list(value) -> bool:
    return isinstance(value, list) and bool(value) and all(isinstance(x, str) for x in value)


def event_split(observation) -> tuple[Any, list[str]]:
    # a result may arrive with the shopper's lines beside it, as {<result>, [sentences]}
    if isinstance(observation, dict) and len(observation) == 2:
        said = [v for v in observation.values() if sentence_list(v)]
        rest = [v for v in observation.values() if not sentence_list(v)]
        if len(said) == 1 and len(rest) == 1:
            return rest[0], said[0]
    return observation, []


def error_text(observation) -> str:
    inner = event_split(observation)[0]
    if not isinstance(inner, dict):
        return ""
    if isinstance(inner.get("error"), str):
        return inner["error"]
    # a short result that carries no data, only a sentence, is an error whatever its key
    if inner and len(inner) <= 3 and not any(isinstance(v, (dict, list)) for v in inner.values()):
        texts = [v for v in inner.values() if isinstance(v, str) and len(v.split()) >= 4]
        if texts and not any(isinstance(v, str) and coded(v) for v in inner.values()):
            return texts[0]
    return ""


def refusal_lines(observation) -> list[str]:
    said = event_split(observation)[1]
    if said:
        return said
    if not isinstance(observation, dict):
        return []
    lines = []
    gather_sentences([v for v in observation.values() if isinstance(v, (list, str))], lines)
    return lines


def line_naming(args, ident) -> str:
    for value in args.values():
        if not isinstance(value, list):
            continue
        hit = next((line for line in value if isinstance(line, dict) and ident in line.values()), None)
        if hit is not None:
            return dump(hit)
    return ""


def ids_in_line_list(args) -> list[str] | None:
    def names_listing(pair) -> bool:
        key, value = pair
        return isinstance(value, str) and coded(value) and bool(IDENT_KEY.search(str(key)))

    # a call that names its listing outright carries choices in its lists, not cart lines
    if any(map(names_listing, args.items())):
        return None
    tables = [v for v in args.values() if isinstance(v, list) and v and all(isinstance(x, dict) for x in v)]
    leading = (next((pair[1] for pair in line.items() if names_listing(pair)), None) for table in tables for line in table)
    return [ident for ident in leading if ident] or None


def mode_signature(properties, args) -> str:
    return dump({k: v for k, v in args.items() if isinstance(properties.get(k), dict) and "enum" in properties[k]})


def json_object_in(text) -> dict[str, Any] | None:
    if not isinstance(text, str):
        return None
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[A-Za-z]*", "", text).strip()
        if text.endswith("```"):
            text = text[:-3].strip()
    attempts = [text]
    first, last = text.find("{"), text.rfind("}")
    if first >= 0 and last > first:
        attempts.append(text[first : last + 1])
    for attempt in attempts:
        try:
            value = json.loads(attempt)
        except ValueError:
            continue
        return value if isinstance(value, dict) else None
    # several objects separated by prose: keep the last whole top-level one
    decoder = json.JSONDecoder()
    chosen = None
    cursor = first
    while 0 <= cursor < len(text):
        try:
            value, stop = decoder.raw_decode(text, cursor)
        except ValueError:
            cursor = text.find("{", cursor + 1)
            continue
        balanced = text.count("{", 0, cursor) == text.count("}", 0, cursor)
        if balanced and isinstance(value, dict):
            chosen = value
        cursor = text.find("{", stop)
    return chosen


KIND_TESTS = {
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": is_number,
    "boolean": lambda v: isinstance(v, bool),
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "null": lambda v: v is None,
}


class SchemaGuard:
    @staticmethod
    def kinds(schema) -> Any:
        declared = schema.get("type")
        return [declared] if isinstance(declared, str) else declared

    @staticmethod
    def fits(value, kind) -> bool:
        test = KIND_TESTS.get(kind) if isinstance(kind, str) else None
        return True if test is None else test(value)

    @classmethod
    def coerce(cls, value, schema) -> Any:
        kinds = cls.kinds(schema)
        if not kinds:
            return value
        if any(cls.fits(value, kind) for kind in kinds):
            whole_float = isinstance(value, float) and value.is_integer()
            return int(value) if "integer" in kinds and whole_float else value
        for kind in kinds:
            try:
                converted = cls.convert(value, kind)
            except (TypeError, ValueError):
                continue
            if converted is not None:
                return converted[0]
        return value

    @staticmethod
    def convert(value, kind) -> tuple[Any] | None:
        if kind == "integer":
            if isinstance(value, float) and value.is_integer():
                return (int(value),)
            if isinstance(value, str) and re.fullmatch(r"\s*-?\d+(\.0+)?\s*", value):
                return (int(float(value)),)
        if kind == "number" and isinstance(value, str) and re.fullmatch(r"\s*-?\d+(\.\d+)?\s*", value):
            number = float(value)
            return (int(number) if number.is_integer() else number,)
        if kind == "string" and is_number(value):
            return (str(value),)
        if kind == "boolean" and isinstance(value, str) and value.lower() in ("true", "false"):
            return (value.lower() == "true",)
        return None

    @classmethod
    def check(cls, value, schema, path) -> tuple[Any, list[str]]:
        if not isinstance(schema, dict):
            return value, []
        value = cls.coerce(value, schema)
        kinds = cls.kinds(schema)
        if kinds and not any(cls.fits(value, kind) for kind in kinds):
            return value, [f"{path}: expected {'/'.join(kinds)}"]
        issues = []
        value = cls.check_choices(value, schema, path, issues)
        if isinstance(value, str):
            if "minLength" in schema and len(value) < schema["minLength"]:
                issues.append(f"{path}: shorter than {schema['minLength']} characters")
            if "maxLength" in schema and len(value) > schema["maxLength"]:
                value = value[: schema["maxLength"]]
        if is_number(value):
            cls.check_bounds(value, schema, path, issues)
        if isinstance(value, list):
            value = cls.check_list(value, schema, path, issues)
        if isinstance(value, dict):
            value = cls.check_object(value, schema, path, issues)
        return value, issues

    @staticmethod
    def check_choices(value, schema, path, issues) -> Any:
        if "enum" in schema and value not in schema["enum"]:
            if isinstance(value, str):
                wanted = value.strip().lower()
                value = next((o for o in schema["enum"] if isinstance(o, str) and o.lower() == wanted), value)
            if value not in schema["enum"]:
                issues.append(f"{path}: must be one of {schema['enum']}")
        if "const" in schema and value != schema["const"]:
            issues.append(f"{path}: must equal {schema['const']!r}")
        return value

    @staticmethod
    def check_bounds(value, schema, path, issues) -> None:
        if "minimum" in schema and value < schema["minimum"]:
            issues.append(f"{path}: below minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            issues.append(f"{path}: above maximum {schema['maximum']}")
        low, high = schema.get("exclusiveMinimum"), schema.get("exclusiveMaximum")
        if "exclusiveMinimum" in schema and isinstance(low, (int, float)) and value <= low:
            issues.append(f"{path}: must be above {low}")
        if "exclusiveMaximum" in schema and isinstance(high, (int, float)) and value >= high:
            issues.append(f"{path}: must be below {high}")

    @classmethod
    def check_list(cls, value, schema, path, issues) -> list[Any]:
        if "minItems" in schema and len(value) < schema["minItems"]:
            issues.append(f"{path}: needs at least {schema['minItems']} items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            issues.append(f"{path}: allows at most {schema['maxItems']} items")
        if not isinstance(schema.get("items"), dict):
            return value
        repaired = []
        for position, element in enumerate(value):
            element, found = cls.check(element, schema["items"], f"{path}[{position}]")
            repaired.append(element)
            issues.extend(found)
        return repaired

    @classmethod
    def check_object(cls, value, schema, path, issues) -> dict[str, Any]:
        properties = schema.get("properties") or {}
        issues.extend(f"{path}.{name}: required" for name in schema.get("required") or [] if name not in value)
        if schema.get("additionalProperties") is False:
            issues.extend(f"{path}.{name}: not an allowed argument" for name in list(value) if name not in properties)
        repaired = {}
        for name, element in value.items():
            if name in properties:
                element, found = cls.check(element, properties[name], f"{path}.{name}")
                issues.extend(found)
            repaired[name] = element
        return repaired


class Clock:
    def __init__(self) -> None:
        self.started = time.monotonic()
        self.cost = 0.0

    def seconds_left(self) -> float:
        return TASK_SECONDS - (time.monotonic() - self.started)

    def elapsed(self) -> float:
        return time.monotonic() - self.started


class LlmDesk:
    def __init__(self, clock, trace) -> None:
        self.clock = clock
        self.trace = trace

    @staticmethod
    def cache_marked(model, messages) -> list[dict[str, Any]]:
        if not model.startswith("anthropic/") or len(messages) <= 2:
            return messages
        marked = [dict(m) for m in messages]
        for position in (0, len(marked) - 1):
            content = marked[position].get("content")
            if isinstance(content, str) and content:
                marked[position]["content"] = [{"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}]
        return marked

    def usable(self, data) -> dict[str, Any] | None:
        if not (isinstance(data, dict) and data.get("choices")):
            return None
        spent = (data.get("usage") or {}).get("cost")
        if isinstance(spent, (int, float)):
            self.clock.cost += float(spent)
        message = data["choices"][0].get("message") or {}
        return message if message.get("content") or message.get("tool_calls") else None

    def complete(self, messages, tools, models, max_tokens, brief=False) -> dict[str, Any] | None:
        for model in models:
            left = self.clock.seconds_left()
            if left < FORCE_SECONDS - 10:
                return None
            _llm.timeout = int(max(8, min(75, left - FORCE_SECONDS + 15)))
            body = {"model": model, "messages": self.cache_marked(model, messages), "temperature": 0, "max_tokens": max_tokens}
            if tools:
                body.update(tools=tools, tool_choice="auto")
            if "gpt-oss" in model:
                # this LLM will not answer with its reasoning off
                body["reasoning"] = {"effort": "low"}
            elif brief and not tools:
                # a yes/no judgement needs little hidden reasoning, and that reasoning was most of a judge call's cost
                body["reasoning"] = {"effort": "low"}
            result = _llm.post_verbose("/inference/chat/completions", body)
            message = self.usable(result.data)
            if message is not None:
                return message
            self.trace.append({"kind": "inference_failure", "model": model, "error": str(result.error)[:300] if result.error else "empty"})
        return None

    def post(self, body) -> tuple[Any, Any]:
        # one raw completion for the engines' own LLM chains: (completion or None, HTTP status or None)
        left = self.clock.seconds_left()
        if left < FORCE_SECONDS:
            return None, None
        _llm.timeout = int(max(8, min(60, left - FORCE_SECONDS)))
        result = _llm.post_verbose("/inference/chat/completions", body)
        data = result.data if isinstance(result.data, dict) else None
        spent = ((data or {}).get("usage") or {}).get("cost")
        if isinstance(spent, (int, float)):
            self.clock.cost += float(spent)
        status = (result.error or {}).get("status") if isinstance(result.error, dict) else None
        if data is None:
            self.trace.append({"kind": "inference_failure", "model": body.get("model"), "error": str(result.error)[:300]})
        return data, status

    def chat(self, messages, model=None, **params) -> str:
        body = {"model": model or MAIN_MODEL, "messages": messages, **params}
        if "gpt-oss" in body["model"]:
            body["reasoning"] = {"effort": "low"}
        data, status = self.post(body)
        message = ((data or {}).get("choices") or [{}])[0].get("message") or {}
        if not message.get("content"):
            raise RuntimeError(f"no reply from {body['model']} ({status})")
        return message["content"]

    def ask_json(self, prompt, max_tokens=1500, models=None, brief=False, keep=False) -> dict[str, Any] | None:
        chosen = models or [MAIN_MODEL, CHEAP_MODEL, LAST_MODEL]
        if self.clock.cost > COST_SOFT and not keep:
            chosen = [m for m in chosen if m != LAST_MODEL] or chosen
        conversation = [{"role": "user", "content": prompt}]
        answer = self.complete(conversation, None, chosen, max_tokens, brief) or {}
        return json_object_in(answer.get("content") or "")


class WorldDesk:
    def __init__(self, binding, policy, clock, trace) -> None:
        self.binding = binding
        self.clock = clock
        self.trace = trace
        self.tools = [t for t in policy.get("tools") or [] if isinstance(t, dict) and isinstance(t.get("function"), dict)]
        self.schemas = {t["function"]["name"]: t["function"].get("parameters") or {"type": "object", "properties": {}} for t in self.tools}
        self.max_steps = int(policy.get("max_steps") or 30)
        self.finished = bool(policy.get("finished"))
        self.step = 0
        self.dead = False
        self.roles = {}
        self.capped = set()

    def steps_left(self) -> int:
        return self.max_steps - self.step

    def over(self) -> bool:
        return self.finished or self.dead

    def role(self, name) -> str | None:
        return self.roles.get(name)

    def roles_of(self, tool) -> set[str]:
        return {role for role, assigned in self.roles.items() if assigned == tool}

    def tool_names(self, tools=None) -> list[str]:
        return [t["function"]["name"] for t in (self.tools if tools is None else tools)]

    def call(self, actions) -> dict[str, Any] | None:
        single = len(actions) == 1
        payload = {**self.binding, "step": self.step + 1}
        payload["action" if single else "actions"] = actions[0] if single else actions
        tries = 3
        while tries > 0:
            tries -= 1
            remaining = self.clock.seconds_left()
            if remaining < 2:
                break
            _env.timeout = int(max(3, min(95, remaining - 1)))
            outcome = _env.post_verbose("/environment/call", payload)
            if isinstance(outcome.data, dict):
                return self.accept(outcome.data, len(actions))
            if not self.retry_after(outcome.error or {}, payload):
                break
        return None

    def accept(self, data, count) -> dict[str, Any]:
        reported = data.get("step")
        self.step = reported if isinstance(reported, int) else self.step + count
        if data.get("finished"):
            self.finished = True
        return data

    def retry_after(self, error, payload) -> bool:
        status = error.get("status")
        body = str(error.get("body") or "")
        self.trace.append({"kind": "env_failure", "status": status, "body": body[:300]})
        if status == 402:
            self.dead = True
            return False
        if status == 409:
            if "finished" in body:
                self.finished = True
                return False
            expected = re.search(r"expected (\d+)", body)
            if not expected:
                return False
            self.step = int(expected.group(1)) - 1
            payload["step"] = self.step + 1
            return True
        return status not in (403, 422, 500, 503, 504)


class PropertyTally:
    def __init__(self) -> None:
        self.counts = {}
        self.distinct = {}
        self.samples = {}
        self.tables = {}
        self.fact_sets = 0

    @staticmethod
    def short(value) -> bool:
        return isinstance(value, str) and 0 < len(value) <= 70

    def visit(self, value, depth, key="", group="") -> None:
        if depth > 6:
            return
        if isinstance(value, list):
            for child in value:
                self.visit(child, depth + 1, key, group)
            return
        if not isinstance(value, dict):
            return
        # a kind label (category, shelf...) sorts the facts so several products get separate menus
        group = next((v.strip().lower() for k, v in value.items() if self.short(v) and KIND_KEY.search(str(k))), "") or group
        pairs = [(k, v) for k, v in value.items() if self.short(v)]
        has_inner_table = any(isinstance(v, dict) and FACT_TABLE_KEY.search(str(k)) for k, v in value.items())
        named_table = len(pairs) >= 2 and (FACT_TABLE_KEY.search(key) or OPTION_KEY.search(key))
        if named_table or (len(pairs) >= 5 and not has_inner_table):
            self.take_row(pairs, key, group)
        for name, child in value.items():
            if isinstance(child, (dict, list)):
                self.visit(child, depth + 1, str(name), group)

    def take_row(self, pairs, key, group) -> None:
        taken = 0
        for raw_key, text in pairs:
            if DIGIT.search(text) and CODE_TEXT.fullmatch(text) and ("-" in text or "_" in text):
                continue
            name = re.sub(r"[_.]+", " ", str(raw_key)).strip().lower()
            if META_NAMES.search(name) or KIND_KEY.search(name):
                continue
            tally = self.counts.setdefault(group, {})
            tally[name] = tally.get(name, 0) + 1
            lowered = text.lower()
            self.distinct.setdefault(group, {}).setdefault(name, set()).add(lowered)
            bucket = self.samples.setdefault(group, {}).setdefault(name, [])
            if lowered not in bucket and len(bucket) < 5:
                bucket.append(lowered)
            taken += 1
        # an options block of one listing is not a separate listing
        if taken >= 2 and not OPTION_KEY.search(key):
            self.fact_sets += 1
            self.tables[group] = self.tables.get(group, 0) + 1

    @staticmethod
    def settled(name, group, decided) -> bool:
        # the request or the shopper already settled this property for this kind of listing
        for head in decided:
            head = re.sub(r"[_.]+", " ", str(head)).strip().lower()
            if head != name and not head.endswith(" " + name):
                continue
            rest = head[: len(head) - len(name)].split()
            if not rest or not group or any(word[:4] == part[:4] for word in rest for part in group.split()):
                return True
        return False

    def ranked(self, group, limit, decided=()) -> list[str]:
        counts = self.counts.get(group, {})
        distinct = self.distinct.get(group, {})
        samples = self.samples.get(group, {})
        open_names = [name for name in counts if not self.settled(name, group, decided)]
        order = sorted(open_names, key=lambda name: (len(distinct.get(name, ())) < 2, -counts[name]))[:limit]
        return [f"- {name}: seen in {counts[name]} listings, {len(distinct.get(name, ()))} distinct values: " + " | ".join(samples.get(name, []))
                for name in order]

    def fold_unlabelled(self) -> None:
        merged = {}
        pooled_values = self.distinct.setdefault("", {})
        pooled_samples = self.samples.setdefault("", {})
        for group, tally in self.counts.items():
            for name, seen in tally.items():
                merged[name] = merged.get(name, 0) + seen
                pooled_values.setdefault(name, set()).update(self.distinct[group][name])
                bucket = pooled_samples.setdefault(name, [])
                for text in list(self.samples[group][name]):
                    if text not in bucket and len(bucket) < 5:
                        bucket.append(text)
        self.counts[""] = merged

    def render(self, several, decided=()) -> str:
        groups = [g for g in sorted(self.counts, key=lambda g: -self.tables.get(g, 0)) if g and self.tables.get(g, 0) >= 2]
        if len(groups) < 2 or not several:
            self.fold_unlabelled()
            return "\n".join(self.ranked("", 28, decided))
        lines = []
        for group in groups[:4]:
            lines.append(f"Listings with the label \"{group}\" ({self.tables[group]} carry details):")
            lines += self.ranked(group, 20, decided)
        if self.counts.get(""):
            lines.append("Listings that carry no label:")
            lines += self.ranked("", 10, decided)
        return "\n".join(lines)


def label_summary(records) -> str:
    counts = {}

    def visit(value, depth) -> None:
        if depth > 6:
            return
        if isinstance(value, list):
            for child in value:
                visit(child, depth + 1)
            return
        if not isinstance(value, dict):
            return
        if len(value) >= 4:
            for key, text in value.items():
                if not isinstance(text, str) or not 0 < len(text) <= 40:
                    continue
                if DIGIT.search(text) and CODE_TEXT.fullmatch(text):
                    continue
                bucket = counts.setdefault(str(key), {})
                bucket[text] = bucket.get(text, 0) + 1
        for child in value.values():
            if isinstance(child, (dict, list)):
                visit(child, depth + 1)

    for record in records:
        visit(record, 0)
    lines = []
    for key, bucket in counts.items():
        if sum(bucket.values()) < 4 or not 2 <= len(bucket) <= 8 or max(bucket.values()) < 2:
            continue
        ordered = sorted(bucket.items(), key=lambda pair: -pair[1])
        lines.append(f"- {key}: " + " | ".join(f"{text} [x{seen}]" for text, seen in ordered))
    return "\n".join(lines[:12])


def row_ident(row) -> str | None:
    return next((v for k, v in row.items() if isinstance(v, str) and "id" in str(k).lower() and ID_SHAPE.match(v)), None)


def flat_fields(row) -> dict[str, Any]:
    # a small nested value (a price with its currency, say) is read as dotted fields of the row
    fields = {}
    for key, value in row.items():
        if isinstance(value, dict) and len(value) <= 4:
            fields.update((f"{key}.{inner}", v) for inner, v in value.items() if not isinstance(v, (dict, list)))
        elif not isinstance(value, (dict, list)):
            fields[str(key)] = value
    return fields


def shop_rows(records) -> dict[str, dict[str, Any]]:
    rows = {}

    def visit(value, depth) -> None:
        if depth > 6:
            return
        if isinstance(value, list):
            for child in value:
                visit(child, depth + 1)
            return
        if not isinstance(value, dict):
            return
        ident = row_ident(value)
        if ident:
            fields = flat_fields(value)
            if any(is_number(v) for v in fields.values()):
                merged = rows.pop(ident, {})
                merged.update((k, v) for k, v in fields.items() if v is not None)
                rows[ident] = merged
        for child in value.values():
            if isinstance(child, (dict, list)):
                visit(child, depth + 1)

    for record in records:
        visit(record, 0)
    return rows


def common_field(rows, test) -> list[str]:
    counts = {}
    for row in rows:
        for key, value in row.items():
            if test(value):
                counts[key] = counts.get(key, 0) + 1
    return [key for key, seen in sorted(counts.items(), key=lambda pair: -pair[1]) if seen * 2 >= len(rows)]


def cart_lines_in(observation) -> list[dict[str, Any]] | None:
    # the cart's own lines: a list of records that each name a listing; an empty list alone means an empty cart
    lists = []

    def visit(value, depth) -> None:
        if depth > 4:
            return
        if isinstance(value, dict):
            for child in value.values():
                visit(child, depth + 1)
        elif isinstance(value, list):
            lists.append(value)
            for child in value:
                if isinstance(child, dict):
                    visit(child, depth + 1)

    visit(observation, 0)
    named = [v for v in lists if v and all(isinstance(x, dict) and row_ident(x) for x in v)]
    if named:
        return named[0]
    tables = [v for v in lists if all(isinstance(x, dict) for x in v)]
    return [] if len(tables) == 1 and not tables[0] else None


def item_records(value, item, depth, holder, in_options, found) -> None:
    if depth > 6 or len(found) >= 40:
        return
    if isinstance(value, list):
        for child in value:
            item_records(child, item, depth + 1, holder, in_options, found)
    elif isinstance(value, dict):
        if item in [v for v in value.values() if isinstance(v, str)]:
            # an option is shown inside the listing that owns it
            shown = holder if in_options and holder is not None and holder is not value else value
            text = dump(clip_strings(shown, 900))[:6000]
            detailed = any(isinstance(v, dict) and len(v) >= 2 for v in shown.values())
            if all(text != seen for seen, _ in found):
                found.append((text, detailed))
            return
        if any(isinstance(v, str) and IDENT_KEY.search(str(k)) and coded(v) for k, v in value.items()):
            holder, in_options = value, False
        for key, child in value.items():
            item_records(child, item, depth + 1, holder, in_options or bool(OPTION_KEY.search(str(key))), found)


class Ledger:
    def __init__(self, policy) -> None:
        self.query = str(policy.get("query") or "")
        self.plan = {}
        self.records = []
        self.seen_ids = set()
        self.opened = set()
        self.cart_ids = []
        self.cart_lines = {}
        self.cart_counts = {}
        self.add_template = None
        self.cart_changed_at = -1
        self.gone = set()
        self.last_cart_view = None
        self.add_signature = None
        self.cart_used = False
        self.order_attempts = 0
        self.refusals = 0
        self.refused_at = -1
        self.refused_lines = {}
        self.searched_at = -1
        self.budget_changed_at = -1
        self.order_checked = False
        self.stated = []
        self.revealed = []
        self.free = []
        self.dropped = []
        self.current = None
        self.budget_now = None
        self.version = 0
        self.heard = set()
        self.said = []
        self.messages_used = 0
        self.message_budget = None
        self.questions_sent = []
        self.pending_reply = ""
        self.last_unclear = False
        self.lookups = []
        self.pointed = []

    def note_refusal(self, step) -> None:
        self.refusals += 1
        self.refused_at = step
        self.order_checked = False

    def add_lookups(self, entries) -> list[str]:
        added = []
        for entry in entries if isinstance(entries, list) else []:
            seller = str(entry.get("seller") or "").strip() if isinstance(entry, dict) else ""
            if not seller or len(self.lookups) >= 3 or any(seller.lower() == known["seller"].lower() for known in self.lookups):
                continue
            self.lookups.append({"kind": str(entry.get("kind") or "").strip(), "seller": seller,
                                 "details": str(entry.get("details") or "").strip(), "found": None, "follow": 0})
            added.append(seller)
        return added

    def items(self) -> list[tuple[str, int]]:
        raw = self.plan.get("items")
        found = []
        for entry in raw if isinstance(raw, list) else []:
            if isinstance(entry, str) and entry.strip():
                entry = {"product": entry, "quantity": 1}
            if not isinstance(entry, dict) or not str(entry.get("product") or "").strip():
                continue
            count = entry.get("quantity")
            if isinstance(count, str) and count.strip().isdigit():
                count = int(count.strip())
            found.append((str(entry["product"]).strip(), int(count) if is_number(count) and count >= 1 else 1))
        return found if len(found) >= 2 else []

    def requirements(self) -> str:
        if self.current is None:
            listed = [f"{r} (from the request)" for r in self.stated if r not in self.dropped]
            listed += [f"{r} (from the shopper)" for r in self.revealed if r not in self.dropped]
        else:
            listed = list(self.current)
        pieces = ["; ".join(listed) if listed else "as the request says"]
        if self.budget_now is not None:
            pieces.append(f"total no higher than {self.budget_now} {self.plan.get('currency') or ''}".rstrip())
        products = self.items()
        counts = ", ".join(f"{name} x{count}" for name, count in products)
        pieces.append("products and counts: " + counts if products else f"quantity {self.plan.get('quantity') or 1}")
        if self.dropped:
            pieces.append("dropped by the shopper: " + "; ".join(self.dropped))
        if self.free:
            pieces.append("the shopper has no preference on: " + ", ".join(self.free))
        return "; ".join(pieces)

    def records_of(self, item) -> list[str]:
        found = []
        for record in self.records:
            item_records(record, item, 0, None, False, found)
        # the newest full details record always goes in, then the two newest records of any kind
        detailed = [index for index, (_, full) in enumerate(found) if full]
        keep = sorted(set(detailed[-1:]) | set(range(max(0, len(found) - 2), len(found))))
        return [found[index][0] for index in keep]

    def other_details(self, item, limit=3) -> list[str]:
        # a requirement may point to another listing, so the listings it led to and the newest other opened ones go along
        shown, found, taken = [], [], set()
        for record in reversed(self.records):
            inner = event_split(record)[0] if isinstance(record, dict) else record
            if not isinstance(inner, dict):
                continue
            if not row_ident(inner):
                inner = next((v for v in inner.values() if isinstance(v, dict) and row_ident(v)), inner)
            ident = row_ident(inner)
            if not ident or ident == item or ident in taken or ident not in self.opened:
                continue
            if not any(isinstance(v, dict) and len(v) >= 2 for v in inner.values()):
                continue
            taken.add(ident)
            (found if ident in self.pointed else shown).append(dump(clip_strings(inner, 400))[:700])
        return found[:3][::-1] + shown[:limit][::-1]

    def property_menu(self) -> tuple[str, int]:
        tally = PropertyTally()
        for record in self.records:
            tally.visit(record, 0)
        held = [entry for entry in self.stated + self.revealed + list(self.current or []) if entry not in self.dropped]
        decided = [entry.split(":")[0] for entry in held] + list(self.free)
        return tally.render(bool(self.items()), decided), tally.fact_sets

    def row_fields(self) -> tuple[dict[str, dict[str, Any]], str | None, str | None, str | None]:
        rows = shop_rows(self.records)
        sample = list(rows.values())
        prices = common_field(sample, is_number)
        if len(prices) > 1 and self.last_cart_view is not None:
            # the price is the figure a cart line repeats for its listing
            lines = cart_lines_in(self.last_cart_view) or []
            shared = [key for key in prices if any(key in flat_fields(line) for line in lines)]
            prices = shared or prices
        if len(prices) > 1:
            prices = [key for key in prices if "price" in key.lower()] or prices
        stocks = common_field(sample, lambda v: isinstance(v, bool))
        kinds = [key for key in common_field(sample, PropertyTally.short) if KIND_KEY.search(key)]
        return rows, (prices or [None])[0], (stocks or [None])[0], (kinds or [None])[0]

    def wanted_units(self) -> int:
        planned = self.plan.get("quantity")
        return planned if isinstance(planned, int) and not isinstance(planned, bool) and planned > 0 else 1

    def affordable(self, total) -> bool:
        return not (is_number(self.budget_now) and self.budget_now > 0 and is_number(total) and total > self.budget_now)

    def cart_total(self, fields=None) -> Any:
        rows, price, _, _ = fields or self.row_fields()
        total = 0
        for ident in self.cart_ids:
            value = (rows.get(ident) or {}).get(price) if price else None
            if not is_number(value):
                return None
            total += value * self.cart_counts.get(ident, 1)
        return total

    def candidates(self, below=None, opened=None, fields=None) -> list[tuple[Any, str]]:
        rows, price, stock, kind = fields or self.row_fields()
        if not price:
            return []
        held = rows.get(self.cart_ids[-1]) if self.cart_ids else None
        shelf = (held or {}).get(kind) if kind else None
        if kind and not shelf:
            labels = [rows[i].get(kind) for i in self.opened if i in rows and rows[i].get(kind)]
            shelf = max(set(labels), key=labels.count) if labels else None
        units = self.wanted_units()
        pool = []
        for ident, row in rows.items():
            value = row.get(price)
            if not is_number(value) or ident in self.gone or (stock and row.get(stock) is False):
                continue
            if shelf and row.get(kind) and row.get(kind) != shelf:
                continue
            if below is not None and value >= below or not self.affordable(value * units):
                continue
            if opened is not None and (ident in self.opened) != opened:
                continue
            pool.append((value, ident))
        return sorted(pool)

    def cheaper_unopened(self) -> list[tuple[str, Any]]:
        if self.plan.get("priority") != "cheapest" or not self.cart_ids:
            return []
        fields = self.row_fields()
        rows, price = fields[0], fields[1]
        held = (rows.get(self.cart_ids[-1]) or {}).get(price) if price else None
        if not is_number(held):
            return []
        ranked = [pair for pair in self.candidates(held, False, fields) if pair[1] not in self.cart_ids]
        return [(ident, value) for value, ident in ranked[:5]]

    @staticmethod
    def cart_target(properties, args) -> str | None:
        return next((v for k, v in args.items() if isinstance(v, str) and "enum" not in (properties.get(k) or {}) and v.strip() and " " not in v.strip()), None)

    @staticmethod
    def count_field(properties) -> str | None:
        return next((k for k, spec in properties.items() if isinstance(spec, dict) and spec.get("type") == "integer" and "enum" not in spec), None)

    def units_after(self, properties, args, target) -> int | None:
        field = self.count_field(properties)
        if field is None:
            return None
        value = args.get(field, properties[field].get("default", 1))
        if not is_number(value):
            return None
        floor = properties[field].get("minimum")
        # a count allowed below zero adds to the line; otherwise it sets the line
        if is_number(floor) and floor < 0:
            return self.cart_counts.get(target, 0) + int(value)
        return int(value)

    def is_removal(self, properties, args) -> bool:
        if self.add_signature is not None and mode_signature(properties, args) != self.add_signature:
            return True
        target = self.cart_target(properties, args)
        units = self.units_after(properties, args, target) if target else None
        return units is not None and units <= 0

    def note_cart_call(self, properties, args, step) -> None:
        removal = self.is_removal(properties, args)
        if self.add_signature is None and not removal:
            self.add_signature = mode_signature(properties, args)
        self.cart_changed_at = step
        listed = ids_in_line_list(args)
        if listed is not None:
            # a cart tool taking whole line lists: either the new basket or lines to take out
            if not removal:
                self.cart_ids = listed
                self.cart_lines = {ident: line_naming(args, ident) for ident in listed}
            else:
                self.cart_ids = [i for i in self.cart_ids if i not in listed]
            return
        target = self.cart_target(properties, args)
        if not target:
            return
        remaining = [i for i in self.cart_ids if i != target]
        if removal:
            self.cart_ids = remaining
            self.cart_counts.pop(target, None)
            return
        self.cart_ids = remaining + [target]
        self.cart_counts[target] = self.units_after(properties, args, target) or 1
        self.add_template = dict(args)

    def sync_cart(self, observation) -> None:
        lines = cart_lines_in(observation)
        if lines is None:
            return
        price = self.row_fields()[1]
        idents = []
        for line in lines:
            ident = row_ident(line)
            whole = [v for k, v in flat_fields(line).items() if k != price and isinstance(v, int) and not isinstance(v, bool)]
            self.cart_counts[ident] = whole[0] if len(whole) == 1 else self.cart_counts.get(ident, 1)
            idents.append(ident)
        self.cart_counts = {ident: self.cart_counts[ident] for ident in idents}
        self.cart_ids = idents


def text_shown(verdicts, needs, listings) -> dict[str, dict[str, Any]]:
    # the judge asks for a fact per need; a listing whose own text holds every need's words, with no fact against it, is taken as a fit
    for listing in listings:
        verdict = verdicts.get(listing.get("id"))
        if verdict and verdict.get("fits") is False and not verdict.get("binding") and needs and all(
                Need(n.get("name"), n.get("value")).judge(listing) == 1 for n in needs):
            verdict.update(fits=True, why="every need's words are in the listing's own text", passed=len(needs))
    return verdicts


def bound_needs(engine) -> list[dict[str, Any]]:
    # the search reader's amount limits go to the judge as its bound fields, so the judge settles plain amounts without another reading
    needs = []
    for requirement, need in zip(engine.requirements, engine.needs):
        entry = {"name": requirement.get("name"), "value": requirement.get("value")}
        if need.bound:
            (amount, unit), rest = need.forms[0], need.forms[1:]
            entry.update(bound=need.bound, amount=amount, unit=unit, forms=[{"value": v, "unit": u} for v, u in rest])
        needs.append(entry)
    return needs


class SetupAdvisor:
    CART_WORDS = {"cart", "basket", "bag", "trolley", "stash", "hold", "line", "lines", "keep", "selection"}

    def __init__(self, llm, world, ledger, clock, trace, opening) -> None:
        self.llm = llm
        self.world = world
        self.ledger = ledger
        self.clock = clock
        self.trace = trace
        self.opening = opening

    def read_request(self) -> None:
        world, ledger = self.world, self.ledger
        described = []
        for tool in world.tools:
            spec = tool["function"]
            described.append(f"- {spec['name']} :: {spec.get('description', '')} :: parameters {dump(spec.get('parameters') or {})}")
        fixed = SETUP_PROMPT + "\n\nStore tools:\n" + "\n".join(described)
        varying = "Shopper request:\n" + ledger.query
        if self.opening is not None:
            varying += "\n\nOpening observation:\n" + dump(self.opening)
        parts = [{"type": "text", "text": fixed, "cache_control": {"type": "ephemeral"}}, {"type": "text", "text": varying}]
        conversation = [{"role": "user", "content": parts}]
        plan = {}
        # the cheap reading first; a stronger LLM only when the reading misses what everything later needs
        for models, room in (([SETUP_MODEL, BACKUP_MODEL], 4000), ([CHEAP_MODEL], 4000), ([LAST_MODEL], 4000)):
            if plan and self.complete_plan(plan) or self.clock.seconds_left() < 200:
                break
            answer = self.llm.complete(conversation, None, models, room)
            read = json_object_in((answer or {}).get("content") or "") or {}
            if self.complete_plan(read) or not plan:
                plan = read
        ledger.plan = plan
        self.read_caps(plan)
        self.assign_roles(plan)
        self.take_numbers(plan)

    def complete_plan(self, plan) -> bool:
        roles = plan.get("roles") if isinstance(plan.get("roles"), dict) else {}
        named = {role for role, tool in roles.items() if isinstance(tool, str) and tool in self.world.schemas}
        items = plan.get("items")
        listed = isinstance(items, list) and all(isinstance(i, dict) and str(i.get("product") or "").strip() and is_number(i.get("quantity")) for i in items)
        return bool(str(plan.get("product") or "").strip()) and "search" in named and bool(named & {"order", "decline"}) and listed

    def read_caps(self, plan) -> None:
        # the shopper's own caps on messages and tool calls, read by an LLM; never parsed from the text by code
        if is_number(plan.get("message_budget")) and is_number(plan.get("step_budget")):
            return
        prompt = ("How many messages may the assistant send the shopper, and how many tool calls or steps may it use, as the request below states them?"
                  ' Reply with JSON only: {"message_cap": integer or null, "tool_call_cap": integer or null} (null when the request states none).\n\n'
                  "Request:\n" + self.ledger.query)
        caps = self.llm.ask_json(prompt, 1500, [READ_MODEL, CHEAP_MODEL]) or {}
        if not is_number(plan.get("message_budget")) and is_number(caps.get("message_cap")):
            plan["message_budget"] = caps["message_cap"]
        if not is_number(plan.get("step_budget")) and is_number(caps.get("tool_call_cap")):
            plan["step_budget"] = caps["tool_call_cap"]

    def assign_roles(self, plan) -> None:
        named = plan.get("roles") if isinstance(plan.get("roles"), dict) else {}
        self.world.roles = {role: tool for role, tool in named.items() if isinstance(tool, str) and tool in self.world.schemas}
        if self.world.roles:
            return
        self.world.roles = self.guess_roles()
        self.trace.append({"kind": "guessed_roles", "roles": self.world.roles})

    def take_numbers(self, plan) -> None:
        ledger, world = self.ledger, self.world
        allowance = plan.get("message_budget")
        if is_number(allowance) and allowance >= 0:
            ledger.message_budget = int(allowance)
        ledger.stated = list(map(str, filter(None, plan.get("requirements") or [])))
        ledger.add_lookups(plan.get("lookups"))
        ledger.budget_now = plan.get("price_cap")
        cap = plan.get("step_budget")
        if is_number(cap) and 0 < cap < world.max_steps:
            world.max_steps = int(cap)

    @staticmethod
    def name_words(name) -> set[str]:
        spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name)
        return {piece.lower() for piece in re.split(r"[^A-Za-z0-9]+", spaced) if piece}

    def single_text_field(self, name) -> bool:
        properties = (self.world.schemas.get(name) or {}).get("properties") or {}
        return len([k for k, v in properties.items() if isinstance(v, dict) and v.get("type") == "string" and "enum" not in v]) == 1

    def role_matches(self, role, words, name) -> bool:
        cart = self.CART_WORDS
        if role == "decline":
            return (bool(words & {"decline", "refuse", "reject", "abandon", "dismiss", "quit", "giveup"})
                    or any(w.startswith(("declin", "refus")) for w in words) or {"give", "up"} <= words or "nothing" in words)
        if role == "cart_view":
            return bool(words & (cart | {"items", "order"})) and bool(words & {"view", "show", "list", "review", "status", "current", "my", "see", "contents", "summary"})
        if role == "cart":
            return bool(words & cart) or bool(words & {"add", "put", "set", "remove", "rewrite", "update"}) and bool(words & (cart | {"items", "order", "all"}))
        if role == "order":
            return bool(words & {"checkout", "purchase", "order", "buy", "commit", "confirm", "place", "settle", "pay", "execute", "finalize", "submit"})
        if role == "message":
            return bool(words & {"message", "shopper", "buyer", "customer", "client", "ask", "chat", "note", "contact", "write", "talk"}) and self.single_text_field(name)
        if role == "details":
            return bool(words & {"detail", "details", "inspect", "open", "read", "view", "fetch", "get", "datasheet", "info", "spec", "specs", "describe", "examine"})
        return bool(words & {"search", "find", "browse", "locate", "hunt", "query", "lookup", "catalog", "seek", "list"})

    def guess_roles(self) -> dict[str, str]:
        guessed = {}
        for role in ("decline", "cart_view", "cart", "order", "message", "details", "search"):
            free = [name for name in self.world.tool_names() if name not in guessed.values()]
            pick = next((name for name in free if self.role_matches(role, self.name_words(name), name)), None)
            if pick is not None:
                guessed[role] = pick
        return guessed

    def opening_brief(self) -> str:
        plan, ledger = self.ledger.plan, self.ledger
        if not plan:
            return ""
        lines = ["Draft reading of the request (compare it with the request itself):"]
        if plan.get("product"):
            lines.append(f"- product wanted: {plan.get('product')}")
        products = ledger.items()
        lines += [f"- product to buy: {name}, count {count} (its own cart line)" for name, count in products]
        if not products:
            lines.append(f"- quantity: {plan.get('quantity') or 1}")
        lines += [f"- requirement from the request: {requirement}" for requirement in plan.get("requirements") or []]
        if plan.get("price_cap") is not None:
            lines.append(f"- total price limit: {plan.get('price_cap')} {plan.get('currency') or ''}".rstrip())
        if plan.get("in_stock_only"):
            lines.append("- in-stock items only")
        priority = plan.get("priority")
        if priority in ("cheapest", "best"):
            measure = f" ({plan.get('priority_detail')})" if plan.get("priority_detail") else ""
            lines.append(f"- pick the {priority}{measure} of all items that fit")
        else:
            lines.append("- no cheapest or best rule found: any item meeting every requirement within budget is correct, so spend no steps chasing the lowest price (unless the request says so)")
        lines += [f"- optional wish, NOT a requirement (tie-breaker only): {want}" for want in plan.get("soft_wants") or []]
        lines += [f"- ask the shopper before choosing: {point}" for point in plan.get("ask_first") or []]
        lines += [f"- procedure rule: {rule}" for rule in plan.get("rules") or []]
        if ledger.message_budget is not None:
            lines.append(f"- message allowance: {ledger.message_budget}")
        lines.append(f"- step allowance: {self.world.max_steps}")
        hints = [h for h in plan.get("hint_properties") or [] if isinstance(h, str)]
        if hints:
            lines.append("- properties the story hints at (ask these first, one per message): " + ", ".join(hints))
        return "\n".join(lines)


class InterpretAdvisor:
    SKIP_WORDS = ("and", "the", "for", "set", "with", "pair", "pack", "of")

    def __init__(self, llm, world, ledger, clock) -> None:
        self.llm = llm
        self.world = world
        self.ledger = ledger
        self.clock = clock

    def hear(self, lines, question) -> str:
        ledger = self.ledger
        fresh = [line for line in lines if line not in ledger.heard]
        if not fresh:
            return ""
        ledger.heard.update(fresh)
        # a late change still binds the order, so it is read while any time is left
        if self.clock.seconds_left() < FORCE_SECONDS + 15:
            return ""
        sections = [INTERPRET_PROMPT, "Shopper request (context only):\n" + ledger.query, "Requirements so far:\n" + ledger.requirements()]
        if ledger.said:
            sections.append("Earlier exchanges with the shopper (oldest first):\n" + "\n".join(ledger.said[-12:]))
        if question:
            sections.append("The assistant asked:\n" + question)
        sections.append("The shopper now said:\n" + "\n".join(fresh))
        ledger.said.append((f"assistant asked: {question} => " if question else "") + "shopper said: " + " / ".join(fresh))
        result = self.llm.ask_json("\n\n".join(sections), 1500, [READ_MODEL, CHEAP_MODEL, LAST_MODEL])
        if not result:
            return ""
        notes = self.apply_changes(result)
        if notes:
            ledger.version += 1
        notes += self.follow_up(result, fresh, question)
        meaning = str(result.get("meaning") or "")
        return "Meaning: " + meaning + (" (" + "; ".join(notes) + ")" if notes else "")

    def apply_changes(self, result) -> list[str]:
        ledger = self.ledger
        notes = []
        now = result.get("requirements_now")
        if isinstance(now, list) and all(isinstance(r, str) for r in now):
            ledger.current = [r for r in now if r.strip()]
        for field, kept, label in (("needs", ledger.revealed, "new requirement"), ("no_preference", ledger.free, ""),
                                   ("dropped", ledger.dropped, "dropped")):
            for entry in result.get(field) or []:
                if not isinstance(entry, str) or not entry or entry in kept:
                    continue
                kept.append(entry)
                if label:
                    notes.append(f"{label}: {entry}")
        notes += [f"the value comes from the listing sold by {seller}: open it" for seller in ledger.add_lookups(result.get("lookups"))]
        budget = result.get("new_budget")
        if is_number(budget) and budget > 0 and budget != ledger.budget_now:
            ledger.budget_now = budget
            ledger.budget_changed_at = self.world.step
            notes.append(f"new budget: {budget}")
            total = ledger.cart_total()
            if is_number(total) and total > budget:
                notes.append(f"the cart now totals {total}, above the new budget: take the line out, search again under {budget}"
                             " (cheaper listings that meet every requirement may exist) and buy the cheapest fit; a smaller budget is no reason to decline")
        notes += self.apply_quantity(result.get("new_quantity"))
        if result.get("unavailable"):
            if len(ledger.cart_ids) == 1:
                ledger.gone.add(ledger.cart_ids[0])
            notes.append("one item can no longer be bought: take it out of the cart and never order it")
        return notes

    def apply_quantity(self, change) -> list[str]:
        if isinstance(change, int) and not isinstance(change, bool):
            change = {"product": "", "quantity": change}
        if not isinstance(change, dict):
            return []
        wanted = change.get("quantity")
        if not isinstance(wanted, int) or isinstance(wanted, bool) or wanted < 1:
            return []
        plan = self.ledger.plan
        if not self.ledger.items():
            plan["quantity"] = wanted
            return [f"new quantity: {wanted}"]
        product = str(change.get("product") or "").strip().lower()
        entries = plan.get("items") if isinstance(plan.get("items"), list) else []
        for entry in entries:
            if not isinstance(entry, dict) or not product:
                continue
            label = str(entry.get("product") or "").lower()
            if product in label or label in product:
                entry["quantity"] = wanted
                return [f"new quantity: {entry.get('product')} x{wanted}"]
        return []

    def names_a_product(self, question) -> bool:
        words = {w.rstrip("s") for w in str(question).lower().replace("?", " ").split()}
        for product, _ in self.ledger.items():
            for word in product.lower().split():
                if len(word) >= 3 and word not in self.SKIP_WORDS and word.rstrip("s") in words:
                    return True
        return False

    def follow_up(self, result, fresh, question) -> list[str]:
        ledger = self.ledger
        reply = str(result.get("reply") or "").strip()
        unclear = result.get("unclear") is True and question is not None
        ledger.last_unclear = unclear
        if unclear and ledger.items() and not self.names_a_product(question):
            # the question left open which product it meant, so the property may still matter
            return ["the shopper lost track because the question did not name its product: if messages remain, ask about the same property once more and name the product"]
        if unclear:
            return ["the shopper did not follow that question: neither explain nor resend it; that property is most likely open, so if messages remain ask about another one (naming the product)"]
        if reply and self.world.role("message") and ASKING.search(" ".join(fresh)):
            ledger.pending_reply = reply
        return []


class VerifyAdvisor:
    def __init__(self, llm, world, ledger, clock, trace) -> None:
        self.llm = llm
        self.world = world
        self.ledger = ledger
        self.clock = clock
        self.trace = trace
        self.verdicts = {}
        self.opened_at = {}
        self.passed = set()
        self.passed_ever = []
        self.cart_checked = set()
        self.qty_warned = set()

    def key_of(self, item, line="") -> str:
        return f"{item}|{line}|{self.ledger.version}"

    def check_cart(self) -> str:
        # every cart line is checked, each one once per state of the requirements
        for item in list(self.ledger.cart_ids):
            verdict = self.check_item(item, "order", self.ledger.cart_lines.get(item, ""))
            if verdict:
                return verdict
        return ""

    def passes(self, item, purpose="order") -> bool:
        line = self.ledger.cart_lines.get(item, "")
        key = self.key_of(item, line)
        self.check_item(item, purpose, line)
        if key not in self.verdicts:
            # no time left to check: an earlier pass that nothing has contradicted still counts
            return item in self.passed_ever
        return key in self.passed

    def failed(self, item) -> bool:
        return bool(self.verdicts.get(self.key_of(item, self.ledger.cart_lines.get(item, ""))))

    def known(self, item) -> str:
        key = self.key_of(item, self.ledger.cart_lines.get(item, ""))
        if key in self.passed:
            return "passed the check"
        return "failed the check" if self.verdicts.get(key) else ""

    def passed_items(self) -> list[str]:
        return [item for item in self.passed_ever if not self.failed(item)]

    def check_item(self, item, purpose, line="") -> str:
        ledger = self.ledger
        if self.clock.seconds_left() < WRAP_UP_SECONDS + 20:
            return ""
        key = self.key_of(item, line)
        # a failure may stem from a listing not opened yet, so it gets one more look once others are opened
        if key in self.verdicts and not (self.verdicts[key] and 0 <= self.opened_at.get(key, -1) < len(ledger.opened)):
            return self.verdicts[key]
        self.opened_at[key] = -1 if key in self.opened_at else len(ledger.opened)
        records = ledger.records_of(item)
        if not records:
            return ""
        listing = self.card_of(item) if hasattr(self, "card_of") else None
        needs = self.needs_for(item) if listing else []
        if not needs:
            return ""
        # the judge engine: code cells first, then the cheap LLM chain; only a failure the listing's own facts show binds
        result = text_shown(judge_listings(self.ask, needs, [listing], quantity=1), needs, [listing]).get(item) or {}
        self.judged[item] = result
        self.trace.append({"kind": "verify", "item": item, "purpose": purpose, "result": result})
        verdict = ""
        if result.get("fits") is False and result.get("binding"):
            verdict = (f"A separate check finds that {item} fails a requirement: {result.get('why')}."
                       " Pick another fully verified fit unless you are sure the check is mistaken.")
        elif result.get("fits") is True:
            self.passed.add(key)
            if item not in self.passed_ever:
                self.passed_ever.append(item)
        self.verdicts[key] = verdict
        return verdict

    def needs_for(self, item) -> list[dict[str, Any]]:
        ledger = self.ledger
        product = (getattr(self, "product_of", None) or {}).get(item)
        line = next((l for l in getattr(self, "lines", None) or [] if l.get("category") == product), None)
        if line is None and len(getattr(self, "lines", None) or []) == 1:
            line = self.lines[0]
        if line is not None:
            needs = list(line.get("needs") or [{"name": r.get("name"), "value": r.get("value")} for r in line["record"].get("requirements") or [] if isinstance(r, dict)])
            # what the shopper said after the engines ran still counts
            needs += needs_from_ledger([r for r in ledger.revealed[getattr(self, "revealed_before", 0):] if r not in ledger.dropped])
            return needs
        entries = [r for r in ledger.stated + ledger.revealed if r not in ledger.dropped]
        return needs_from_ledger(entries, "", [name for name, _ in ledger.items()])

    def quantity_warning(self, name, args) -> str:
        ledger = self.ledger
        wanted = ledger.plan.get("quantity")
        if not isinstance(wanted, int) or isinstance(wanted, bool) or wanted < 1:
            return ""
        properties = self.world.schemas[name].get("properties") or {}
        if ledger.add_signature is not None and mode_signature(properties, args) != ledger.add_signature:
            return ""
        target = ledger.cart_target(properties, args)
        units = ledger.units_after(properties, args, target) if target else None
        if units is None:
            return ""
        products = ledger.items()
        if products:
            if units not in {count for _, count in products}:
                return f"this request calls for {', '.join(f'{n} x{q}' for n, q in products)}, yet the line would hold {units}."
            return ""
        if units != wanted:
            return f"this request calls for {wanted} units of one listing, yet the line would hold {units}."
        return ""

    def cart_gate(self, name, args, signature) -> str:
        ledger = self.ledger
        properties = self.world.schemas[name].get("properties") or {}
        # taking a line out needs no check
        if ledger.is_removal(properties, args):
            return ""
        if self.world.steps_left() > 3:
            target = ledger.cart_target(properties, args)
            if target is None and ids_in_line_list(args):
                target = ids_in_line_list(args)[-1]
            if target and f"{target}|{ledger.version}" not in self.cart_checked:
                self.cart_checked.add(f"{target}|{ledger.version}")
                verdict = self.check_item(target, "put in the cart", line_naming(args, target))
                if verdict:
                    return NO_STEP + ". " + verdict + " Repeat the identical call if you are sure it fits."
        warning = self.quantity_warning(name, args)
        if warning and signature not in self.qty_warned:
            self.qty_warned.add(signature)
            return NO_STEP + ": " + warning + " Repeat the identical call if you are sure."
        return ""


class PlanAdvisor:
    def __init__(self, llm, world, ledger, clock) -> None:
        self.llm = llm
        self.world = world
        self.ledger = ledger
        self.clock = clock
        self.active = True
        self.asked = []
        self.sets_at_last_ask = -1
        self.misses = 0
        self.retry_at = 0

    def limit(self) -> int:
        budget = self.ledger.message_budget
        return 5 if budget is None else budget

    def message_args(self, text) -> dict[str, Any] | None:
        schema = self.world.schemas.get(self.world.role("message") or "") or {}
        properties = schema.get("properties") or {}
        required = list(schema.get("required") or [])
        texts = [k for k, v in properties.items()
                 if isinstance(v, dict) and (v.get("type") == "string" or "string" in (v.get("type") or [])) and "enum" not in v]
        field = next((k for k in required if k in texts), None) or (texts[0] if len(texts) == 1 else None)
        if not field or any(k != field for k in required):
            return None
        return {field: text}

    def message_call(self, call_id, text) -> dict[str, Any] | None:
        tool = self.world.role("message")
        args = self.message_args(text)
        if not args:
            return None
        args, problems = SchemaGuard.check(args, self.world.schemas[tool], "args")
        if problems:
            return None
        return {"id": call_id, "type": "function", "function": {"name": tool, "arguments": json.dumps(args)}}

    def worth_planning(self, fact_sets, urgent, force) -> bool:
        if not (force or urgent) and fact_sets < 4 and self.world.step < 8:
            return False
        # a later round waits for more listings so it can use earlier answers
        if self.asked and not (force or urgent or self.ledger.last_unclear) and fact_sets < self.sets_at_last_ask + 2:
            return False
        return force or fact_sets >= self.retry_at

    def draft(self, force=False) -> tuple[list[dict[str, Any]], list[tuple[str, str]], int, int] | None:
        world, ledger = self.world, self.ledger
        limit = self.limit()
        if not world.role("message") or ledger.messages_used >= limit:
            self.active = False
            return None
        menu, fact_sets = ledger.property_menu()
        urgent = bool(ledger.plan.get("ask_first")) and not self.asked
        if not self.worth_planning(fact_sets, urgent, force):
            return None
        urgent = urgent and not force
        if world.steps_left() <= (6 if force else 10) or self.clock.seconds_left() < WRAP_UP_SECONDS + 40:
            self.active = False
            return None
        products = ledger.items()
        # one product: all remaining questions now; several: one per round
        count = 1 if products else max(1, limit - ledger.messages_used)
        result = self.llm.ask_json(self.plan_prompt(count, limit, urgent, products, menu), 2000, [READ_MODEL, CHEAP_MODEL, LAST_MODEL])
        planned = [q for q in (result or {}).get("questions") or [] if isinstance(q, dict) and str(q.get("question") or "").strip()]
        if urgent and fact_sets < 4:
            planned = planned[:1]
        planned = [q for q in planned if str(q.get("question")).strip().lower() not in ledger.questions_sent][:count]
        calls, labels = [], []
        for index, entry in enumerate(planned):
            question = str(entry.get("question")).strip()
            call = self.message_call(f"ask_{ledger.messages_used + index + 1}_{world.step}", question)
            if call is None:
                continue
            calls.append(call)
            product = str(entry.get("product") or "").strip()
            labels.append(((product + " " if product and products else "") + str(entry.get("property")), question))
        if calls:
            return calls, labels, fact_sets, limit
        # nothing usable: try once more after another listing is opened
        self.misses += 1
        self.retry_at = fact_sets + 1
        if self.misses >= 2:
            self.active = False
        return None

    def plan_prompt(self, count, limit, urgent, products, menu) -> str:
        ledger = self.ledger
        prompt = PLAN_PROMPT.replace("QUESTION_COUNT", str(count)) + "\n\nShopper request:\n" + ledger.query
        if SWEEP_LAST and limit >= 2 and ledger.messages_used + count == limit and not urgent:
            prompt += SWEEP_NOTE
        prompt += "\n\nRequirements so far:\n" + ledger.requirements()
        if products:
            prompt += "\n\nSeveral different products requested (say which one each question is about): " + "; ".join(name for name, _ in products)
        if self.asked:
            prompt += "\n\nAlready asked (with the shopper's answers):\n" + "\n".join(self.asked)
        if ledger.plan.get("ask_first"):
            prompt += "\n\nMust be learned from the shopper: " + "; ".join(str(h) for h in ledger.plan.get("ask_first") or [])
        return prompt + "\n\nProperties in the listings' details seen so far (name, listing count, example values):\n" + (menu or "(nothing seen yet)")

    def record(self, labels, replies, fact_sets, limit) -> None:
        for (label, question), (_, text) in zip(labels, replies):
            answer = text.split("\n[status]")[0][:400]
            self.asked.append(f"- {label}: we asked \"{question}\" => {answer}")
        self.sets_at_last_ask = fact_sets
        self.active = self.ledger.messages_used < limit

    def reply_call(self, text) -> dict[str, Any] | None:
        world, ledger = self.world, self.ledger
        if not world.role("message") or ledger.messages_used >= self.limit() or world.steps_left() <= 4:
            return None
        if self.clock.seconds_left() < WRAP_UP_SECONDS:
            return None
        return self.message_call(f"reply_{world.step}", text)


class AuditAdvisor:
    def __init__(self, llm, world, ledger, clock, trace) -> None:
        self.llm = llm
        self.world = world
        self.ledger = ledger
        self.clock = clock
        self.trace = trace
        self.seen = None

    def look(self) -> tuple[list[str], str]:
        ledger = self.ledger
        stamp = (len(ledger.records), ledger.version)
        if self.seen is not None and self.seen[0] == stamp:
            return self.seen[1]
        if self.clock.seconds_left() < WRAP_UP_SECONDS + 30:
            return [], ""
        blocks = []
        used = 0
        for record in reversed(ledger.records):
            text = dump(clip_strings(record, 160))
            if used + len(text) <= 16000:
                blocks.append(text)
                used += len(text)
        if not blocks:
            return [], ""
        prompt = "\n\n".join([AUDIT_PROMPT, "Shopper request:\n" + ledger.query, "Current requirements:\n" + ledger.requirements(),
                                "Listing records seen (newest first):\n" + "\n".join(blocks)])
        # this seat reasons before it answers: the room covers both
        result = self.llm.ask_json(prompt, 3000, brief=True)
        self.trace.append({"kind": "audit", "result": result})
        fits = [str(x) for x in (result or {}).get("possible_fits") or [] if isinstance(x, (str, int))][:5]
        self.seen = (stamp, (fits, str((result or {}).get("reason") or "")))
        return self.seen[1]

    def review(self) -> str:
        fits, why = self.look()
        if not fits:
            return ""
        reason = f" ({why})" if why else ""
        text = ("A separate check believes these listings could meet every requirement: " + ", ".join(fits) + reason
                + ". Check them (details, stock, price) and buy the right one; decline only when every one of them really fails.")
        if self.ledger.order_attempts:
            text += " No purchase has happened: an order that succeeds ends the episode, and this episode is still going, so a decline now leaves the shopper empty-handed."
        return text

    def scrub(self, args, name) -> dict[str, Any]:
        rule = self.ledger.plan.get("decline_no_names")
        if rule is False:
            return args
        seen = self.ledger.seen_ids
        mentions = any(token and token in seen for value in args.values() if isinstance(value, str)
                       for token in re.split(r"[\s.,;:!?()\[\]{}<>\"'`]+", value))
        if not (rule or mentions):
            return args
        properties = (self.world.schemas.get(name) or {}).get("properties") or {}
        cleaned = {}
        for key, value in args.items():
            spec = properties.get(key) or {}
            cleaned[key] = NEUTRAL_DECLINE[: spec.get("maxLength", len(NEUTRAL_DECLINE))] if isinstance(value, str) and "enum" not in spec else value
        return cleaned


class Finisher:
    def __init__(self, llm, world, ledger, clock, trace) -> None:
        self.llm = llm
        self.world = world
        self.ledger = ledger
        self.clock = clock
        self.trace = trace
        self.modes = None

    def mode_field(self, name) -> tuple[str, list[str]] | None:
        properties = (self.world.schemas.get(name) or {}).get("properties") or {}
        for key, spec in properties.items():
            values = [v for v in (spec.get("enum") or []) if isinstance(v, str)] if isinstance(spec, dict) else []
            if len(values) >= 2:
                return key, values
        return None

    def finish_modes(self) -> dict[str, str]:
        # one finishing tool for both endings: the setup reading names the two values, else elimination, else one short question
        if self.modes is not None:
            return self.modes
        self.modes = {}
        roles = self.world.roles
        tool = roles.get("order") or roles.get("decline")
        found = self.mode_field(tool) if tool else None
        named = self.ledger.plan.get("finish_values")
        named = named if isinstance(named, dict) else {}
        merged = roles.get("order") == roles.get("decline") or (not roles.get("decline") and (named or not self.ledger.plan))
        if not found or not merged:
            return self.modes
        field, values = found
        pick = {end: next((v for v in values if v.lower() == str(named.get(end) or "").strip().lower()), None) for end in ("order", "decline")}
        if len(values) == 2:
            for end, other in (("order", "decline"), ("decline", "order")):
                if pick[end] is None and pick[other] is not None:
                    pick[end] = next(v for v in values if v != pick[other])
        if pick["order"] is None or pick["decline"] is None or pick["order"] == pick["decline"]:
            pick = self.ask_modes(tool, values) or pick
        if pick.get("order") and pick.get("decline") and pick["order"] != pick["decline"]:
            self.modes = {"tool": tool, "field": field, "order": pick["order"], "decline": pick["decline"]}
            roles.setdefault("order", tool)
            roles.setdefault("decline", tool)
        self.trace.append({"kind": "finish_modes", "modes": self.modes})
        return self.modes

    def ask_modes(self, tool, values) -> dict[str, str] | None:
        spec = next((t["function"] for t in self.world.tools if t["function"]["name"] == tool), {})
        prompt = FINISH_MODE_PROMPT + "\n\nTool:\n" + dump({"name": tool, "description": spec.get("description", ""), "parameters": spec.get("parameters") or {}})
        answer = self.llm.ask_json(prompt, 1500, [READ_MODEL, CHEAP_MODEL, LAST_MODEL]) or {}
        pick = {end: next((v for v in values if v.lower() == str(answer.get(end) or "").strip().lower()), None) for end in ("order", "decline")}
        return pick if all(pick.values()) else None

    def is_decline(self, name, args) -> bool:
        roles = self.world.roles
        modes = self.finish_modes()
        if modes and name == modes["tool"]:
            return args.get(modes["field"]) == modes["decline"]
        return name == roles.get("decline") and roles.get("order") != name

    def neutral_args(self, name, text, declining=False) -> dict[str, Any]:
        schema = self.world.schemas.get(name) or {}
        properties = schema.get("properties") or {}
        modes = self.finish_modes()
        args = {}
        if modes and name == modes["tool"]:
            args[modes["field"]] = modes["decline" if declining else "order"]
        for key in schema.get("required") or []:
            spec = properties.get(key) or {}
            declared = spec.get("type")
            if key in args:
                continue
            if "enum" in spec and spec["enum"]:
                args[key] = spec["enum"][0]
            elif declared == "string" or "string" in (declared or []):
                args[key] = text[: spec.get("maxLength", len(text))]
            elif declared == "boolean":
                args[key] = True
            elif declared in ("integer", "number"):
                args[key] = max(1, spec.get("minimum", 1))
        if len(properties) == 1:
            for key, spec in properties.items():
                if key not in args and isinstance(spec, dict) and spec.get("type") == "string" and "enum" not in spec:
                    args[key] = text[: spec.get("maxLength", len(text))]
        return args

    def decline_action(self) -> dict[str, Any]:
        decline = self.world.role("decline")
        return {"name": decline, "args": self.neutral_args(decline, NEUTRAL_DECLINE, declining=True)}

    def send(self, action) -> dict[str, Any] | None:
        reply = self.world.call([action])
        self.trace.append({"kind": "forced_finish", "action": action, "reply": reply})
        return reply

    def force(self) -> None:
        world = self.world
        if world.over() or world.steps_left() <= 0 or self.clock.seconds_left() < 2:
            return
        order, decline = world.role("order"), world.role("decline")
        if order and self.ledger.cart_ids:
            action = {"name": order, "args": self.neutral_args(order, "Order the cart.")}
        elif decline:
            action = self.decline_action()
        else:
            return
        reply = self.send(action)
        observation = (reply or {}).get("observation")
        refused = observation if isinstance(observation, dict) else {}
        pieces = [v for v in refused.values() if isinstance(v, (list, str))]
        pieces += [x for v in refused.values() if isinstance(v, dict) for x in v.values() if isinstance(x, (list, str))]
        lines = []
        gather_sentences(pieces, lines)
        said = " ".join(lines)
        open_reply = bool(reply) and not reply.get("finished") and bool(lines)
        if (open_reply and order and action["name"] == order and self.ledger.cart_ids
                and PRICE_TALK.search(said) and not GONE_WORDS.search(said) and world.steps_left() > 0):
            # only the price moved: the cart is still good, so order once more
            reply = self.send(action)
        if reply and not reply.get("finished") and isinstance(reply.get("observation"), dict) and lines and decline and world.steps_left() > 0:
            self.send(self.decline_action())


class ShopClosed(Exception):
    pass


class StoreDesk:
    # the engines' store contract (search, browse, view, view_many, ask, chat, steps_left) spoken through the world's own tools
    def __init__(self, driver) -> None:
        self.driver = driver
        self.world = driver.world
        self.limit = None
        self.uses = self.read_uses(driver.ledger.plan.get("store_args"))
        self.spare = self.spare_viewers()
        self.offer = None
        if "batch_view" in self.uses:
            self.view_many = self.open_many
            spec = ((self.world.schemas[self.uses["batch_view"]["tool"]].get("properties") or {}).get(self.uses["batch_view"]["ids"]) or {})
            self.batch = int(spec["maxItems"]) if is_number(spec.get("maxItems")) and spec["maxItems"] >= 1 else 12

    def read_uses(self, given) -> dict[str, dict[str, Any]]:
        # each lookup's tool and parameter names, kept only where the tool's schema has them
        given = given if isinstance(given, dict) else {}
        uses = {}
        if not (isinstance(given.get("batch_view"), dict) and given["batch_view"].get("tool") in self.world.schemas):
            # a tool whose one required input is a list of strings opens several listings at once
            for tool, schema in self.world.schemas.items():
                required = schema.get("required") or []
                spec = (schema.get("properties") or {}).get(required[0]) if len(required) == 1 else None
                if isinstance(spec, dict) and spec.get("type") == "array" and (spec.get("items") or {}).get("type") == "string":
                    given = {**given, "batch_view": {"tool": tool, "ids": required[0]}}
                    break
        for use, slots in (("search", ("query", "page")), ("browse", ("category", "max_price", "brand", "page")), ("view", ("id",)), ("batch_view", ("ids",))):
            entry = given.get(use) if isinstance(given.get(use), dict) else {}
            tool = entry.get("tool") if entry.get("tool") in self.world.schemas else self.world.role(use if use != "view" else "details") or self.world.role(use)
            if not tool:
                continue
            properties = (self.world.schemas[tool].get("properties") or {})
            chosen = {slot: entry.get(slot) for slot in slots if entry.get(slot) in properties}
            fixed = entry.get("fixed") if isinstance(entry.get("fixed"), dict) else {}
            chosen["fixed"] = {k: v for k, v in fixed.items() if k in properties and k not in chosen.values()}
            main = slots[0]
            if main not in chosen:
                free = [k for k, v in properties.items() if isinstance(v, dict) and "enum" not in v and k not in chosen["fixed"]
                        and (v.get("type") in ("string", "array") or "string" in (v.get("type") or []))]
                required = [k for k in (self.world.schemas[tool].get("required") or []) if k in free]
                pick = (required or free)[:1]
                if not pick:
                    continue
                chosen[main] = pick[0]
            if use == "search" and "page" not in chosen:
                # a search tool with one free parameter besides its words takes the page marker there
                rest = [k for k, v in properties.items() if isinstance(v, dict) and "enum" not in v and k != chosen[main] and k not in chosen["fixed"]]
                if len(rest) == 1:
                    chosen["page"] = rest[0]
            chosen["tool"] = tool
            uses[use] = chosen
        return uses

    def ready(self) -> bool:
        return "search" in self.uses and ("view" in self.uses or "batch_view" in self.uses)

    def refresh(self, ident) -> dict[str, Any]:
        if not self.offer or self.steps_left <= 0:
            return {}
        args = {**self.offer["fixed"], self.offer["id"]: ident}
        node = self.node_of(self.driver.code_call(self.offer["tool"], args), ident)
        return self.driver.card(ident, node) if node else {}

    def spare_viewers(self) -> list[tuple[str, str]]:
        # other tools that take nothing but one listing id: a store may keep the offer and the properties apart
        used = {entry["tool"] for entry in self.uses.values()}
        taken = set()
        for role in ("search", "message", "cart", "cart_view", "order", "decline", "browse"):
            taken.add(self.world.role(role))
        spare = []
        for tool, schema in self.world.schemas.items():
            properties = schema.get("properties") or {}
            required = schema.get("required") or []
            if tool in used or tool in taken or len(properties) != 1 or list(properties) != required:
                continue
            spec = properties[required[0]]
            if isinstance(spec, dict) and spec.get("type") == "string" and "enum" not in spec:
                spare.append((tool, required[0]))
        return spare

    @property
    def steps_left(self) -> int:
        left = self.world.steps_left() - 1
        if self.limit is not None:
            left = min(left, self.limit - self.world.step)
        return max(0, left)

    @property
    def categories(self) -> list[str]:
        return []

    def use(self, name, extra) -> Any:
        if self.steps_left <= 0 or self.world.over():
            raise ShopClosed()
        entry = self.uses[name]
        args = {**entry["fixed"], **{entry[slot]: value for slot, value in extra.items() if entry.get(slot) and value is not None}}
        args, problems = SchemaGuard.check(args, self.world.schemas[entry["tool"]], "args")
        if problems:
            self.driver.trace.append({"kind": "store_args", "use": name, "args": args, "problems": problems})
            raise TypeError(f"{name}: {problems}")
        return self.driver.code_call(entry["tool"], args)

    def search(self, query, page=None) -> dict[str, Any]:
        return self.result(self.use("search", {"query": query, "page": page}), "search")

    def browse(self, category, page=None, max_price=None, brand=None) -> dict[str, Any]:
        if "browse" not in self.uses:
            return self.search(category, page)
        if max_price is not None and not self.uses["browse"].get("max_price"):
            raise TypeError("no price filter")
        return self.result(self.use("browse", {"category": category, "page": page, "max_price": max_price, "brand": brand}), "browse")

    def view(self, ident) -> dict[str, Any]:
        observation = self.use("view", {"id": ident})
        node = self.node_of(observation, ident)
        if node:
            self.driver.ledger.opened.add(ident)
        card = self.driver.card(ident, node) if node else {}
        if node and not card.get("facts") and self.spare:
            # an open without properties: the store's other id-only tool is tried once, and kept for every open if it shows them
            spare, self.spare = self.spare, []
            for tool, slot in spare:
                if self.steps_left <= 0:
                    break
                other = self.node_of(self.driver.code_call(tool, {slot: ident}), ident)
                fuller = self.driver.card(ident, other) if other else {}
                if fuller.get("facts"):
                    # the first tool still holds the offer itself (stock, price): picks are checked there before the cart
                    self.offer, self.uses["view"] = self.uses["view"], {"tool": tool, "id": slot, "fixed": {}}
                    self.driver.stage_notes.append(f"Listing properties come from {tool}; {self.world.role('details') or 'the other details tool'} shows none,"
                                                   f" so open candidates with {tool}.")
                    return fuller
        return card

    def open_many(self, ids) -> dict[str, dict[str, Any]]:
        # a store may quietly answer nothing above some batch size: the size that came back empty is halved and asked again
        ids, cards = list(ids), {}
        while ids and self.steps_left > 0:
            chunk = ids[:self.batch]
            observation = self.use("batch_view", {"ids": chunk})
            found = {ident: self.driver.card(ident, node) for ident in chunk for node in [self.node_of(observation, ident)] if node}
            if not found and len(chunk) > 1:
                self.batch = max(1, len(chunk) // 2)
                continue
            cards.update(found)
            self.driver.ledger.opened.update(found)
            ids = ids[len(chunk):]
        return cards

    def ask(self, message) -> str:
        if self.steps_left <= 0 or self.world.over():
            raise ShopClosed()
        args = self.driver.planner.message_args(message)
        if not args:
            raise TypeError("no message tool")
        observation = self.driver.code_call(self.world.role("message"), args)
        texts = []

        def visit(value) -> None:
            if isinstance(value, str) and value.strip() and value.strip() != message.strip():
                texts.append(value.strip())
            elif isinstance(value, dict):
                for child in value.values():
                    visit(child)
            elif isinstance(value, list):
                for child in value:
                    visit(child)

        visit(observation)
        return " ".join(texts)

    def chat(self, messages, model=None, **params) -> str:
        return self.driver.llm.chat(messages, model, **params)

    @staticmethod
    def node_of(observation, ident) -> dict[str, Any] | None:
        found = []

        def visit(value, depth) -> None:
            if depth > 6:
                return
            if isinstance(value, dict):
                if any(v == ident for v in value.values()):
                    found.append(value)
                for child in value.values():
                    visit(child, depth + 1)
            elif isinstance(value, list):
                for child in value:
                    visit(child, depth + 1)

        visit(observation, 0)
        return max(found, key=len) if found else None

    def result(self, observation, use) -> dict[str, Any]:
        # rows are the longest list of records that name a listing; the count is the largest whole number beside them,
        # and the one other plain value beside them is the next page's marker
        body = event_split(observation)[0] if isinstance(observation, dict) else observation
        if not isinstance(body, dict) or error_text(body):
            return {"rows": [], "count": None, "next": None}
        lists = [(k, v) for k, v in body.items() if isinstance(v, list) and any(isinstance(x, dict) and row_ident(x) for x in v)]
        key, raw = max(lists, key=lambda pair: len(pair[1])) if lists else (None, [])
        rows = [self.driver.card(row_ident(x), x) for x in raw if isinstance(x, dict) and row_ident(x)]
        plain = {k: v for k, v in body.items() if k != key and not isinstance(v, (dict, list, bool))}
        numbers = {k: v for k, v in plain.items() if is_number(v)}
        count_key = max(numbers, key=lambda k: numbers[k]) if numbers else None
        count = numbers[count_key] if count_key and (numbers[count_key] >= len(rows) or not rows) else None
        if not lists and count is None:
            count = 0 if any(isinstance(v, list) for v in body.values()) else None
        rest = [k for k in plain if k != count_key]
        marker = body.get(rest[0]) if len(rest) == 1 and self.uses[use].get("page") else None
        return {"rows": rows, "count": int(count) if is_number(count) else count, "next": marker if marker not in ("", 0) else None}


class Driver:
    def __init__(self, clock, llm, world, ledger, trace, opening, setup, interpreter, planner, verifier, auditor, finisher) -> None:
        self.clock = clock
        self.llm = llm
        self.world = world
        self.ledger = ledger
        self.trace = trace
        self.opening = opening
        self.setup = setup
        self.interpreter = interpreter
        self.planner = planner
        self.verifier = verifier
        self.auditor = auditor
        self.finisher = finisher
        self.messages = []
        self.recent = []
        self.decline_checks = 0
        self.cheaper_checks = 0
        self.research_checks = 0
        self.early_bounced = False
        self.notes = {}
        self.swaps = 0
        self.rescues = 0
        self.reorders = 0
        self.fills = 0
        self.budget_checks = 0
        self.page_seen = {}
        self.cards = {}
        self.lines = []
        self.product_of = {}
        self.judged = {}
        self.picks = {}
        self.stage_notes = []
        self.basket_holds = 0
        self.decline_holds = 0
        self.restocked = set()
        self.verifier.cards = self.cards
        self.verifier.product_of = self.product_of
        self.verifier.lines = self.lines
        self.verifier.judged = self.judged
        self.verifier.card_of = self.card_of
        self.verifier.ask = chain_ask(self.llm.post)

    def status(self) -> str:
        world, ledger = self.world, self.ledger
        left = world.steps_left()
        sent = f"{ledger.messages_used} messages sent" + ("" if ledger.message_budget is None else f" of {ledger.message_budget}")
        text = (f"[status] {world.step} of {world.max_steps} steps used, {left} remaining; {sent}; "
                f"roughly {int(max(0, self.clock.seconds_left() - FORCE_SECONDS))} s to go.")
        text += "\n[current requirements] " + ledger.requirements()
        text += self.pointer_note()
        text += self.cheapest_note()
        if ledger.refusals and not world.finished and world.step - ledger.refused_at >= 2:
            # a refused order is easily mistaken for a placed one
            text += (f" NO PURCHASE HAS HAPPENED: the order attempt made {world.step - ledger.refused_at} steps ago left the episode running."
                     " When the cart holds fitting items inside the current budget, call the order tool now; otherwise repair the cart and then order"
                     " (decline only when nothing in the store fits any more).")
        if left <= 2:
            text += " FINAL STEPS: order now when the cart holds a verified fit, else decline."
        elif left <= 5 or self.clock.seconds_left() < WRAP_UP_SECONDS:
            text += " WRAP UP NOW: set the cart to the verified line(s) the request needs, check stock again, then order (or decline when nothing fits)."
        elif ledger.order_attempts == 0 and left <= 14:
            text += " Steps are getting scarce: soon put your best verified fit in the cart and order; searches and detail reads stop once 7 steps remain, and repairing a refused order takes about 4 steps."
        return text

    def offered_tools(self) -> list[dict[str, Any]]:
        world, ledger = self.world, self.ledger
        order, decline, cart = world.role("order"), world.role("decline"), world.role("cart")
        left = world.steps_left()
        names = None
        if left <= 1:
            names = {order, decline}
        elif left <= 3:
            names = {order, decline, cart}
        elif left <= 7 and (ledger.order_attempts == 0 or ledger.refusals > 0):
            names = {order, decline, cart, world.role("cart_view"), world.role("view")}
        tools = world.tools
        if names is not None and order and cart:
            # an unassigned tool that completes checkout stays on offer
            assigned = set(world.roles.values())
            names |= {name for name in world.tool_names() if name not in assigned and FINISH_TOOL_NAME.search(name)}
            narrowed = [t for t in world.tools if t["function"]["name"] in names]
            tools = narrowed or tools
        if self.planner.active and world.role("message"):
            tools = [t for t in tools if t["function"]["name"] != world.role("message")]
        if world.capped:
            tools = [t for t in tools if t["function"]["name"] not in world.capped] or tools
        return tools

    def post_replies(self, replies) -> None:
        for index, (call_id, text) in enumerate(replies):
            if index == len(replies) - 1:
                text += "\n" + self.status()
            self.messages.append({"role": "tool", "tool_call_id": call_id, "content": text})

    def ask_round(self, force=False) -> bool:
        drafted = self.planner.draft(force)
        if drafted is None:
            return False
        calls, labels, fact_sets, limit = drafted
        self.messages.append({"role": "assistant", "content": "Questions for the shopper now: " + ", then ".join(label for label, _ in labels) + ".", "tool_calls": calls})
        self.ledger.last_unclear = False
        replies = self.run_calls(calls, planned=True)
        self.post_replies(replies)
        self.planner.record(labels, replies, fact_sets, limit)
        return True

    def answer_shopper(self, text) -> bool:
        call = self.planner.reply_call(text)
        if call is None:
            return False
        self.messages.append({"role": "assistant", "content": "Replying to the shopper's question.", "tool_calls": [call]})
        self.post_replies(self.run_calls([call], planned=True))
        return True

    def screen(self, call, total, allowed, finishing, planned, queued) -> Any:
        world, ledger = self.world, self.ledger
        message_tool = world.role("message")
        function = call.get("function") or {}
        name = function.get("name")
        raw = function.get("arguments")
        if name not in world.schemas:
            return f"{NO_STEP}: '{name}' is not an available tool. Available: {', '.join(sorted(allowed))}."
        if not planned and name not in allowed:
            if name == message_tool and self.planner.active:
                return f"{NO_STEP}: shopper questions are planned and sent for you once you have opened a few listings' details."
            return f"{NO_STEP}: not enough steps remain for that; the only usable tools now are " + ", ".join(sorted(allowed)) + "."
        if isinstance(raw, dict):
            args = raw
        elif (raw or "").strip():
            args = json_object_in(raw or "{}")
            if args is None:
                return f"{NO_STEP}: the arguments are not a valid JSON object."
        else:
            args = {}
        args, problems = SchemaGuard.check(args, world.schemas[name], "args")
        if problems:
            return f"{NO_STEP}: argument errors: " + "; ".join(problems[:6])
        if name in finishing and total > 1:
            return f"{NO_STEP}: send an order or a decline on its own, once you have the results of your other calls."
        if name == message_tool:
            spoken = " ".join(str(v) for v in args.values() if isinstance(v, str))
            if not planned and "?" not in spoken and not any(TELL_RULE.search(str(rule)) for rule in ledger.plan.get("rules") or []):
                return f"{NO_STEP}: shopper messages are for questions only, and this one asks nothing. Keep findings, picks and plans out of messages; the finishing tool carries them."
            if spoken.strip().lower() in ledger.questions_sent:
                return f"{NO_STEP}: this very message went out before and its answer is above. Ask something else, or move on."
            if ledger.message_budget is not None and ledger.messages_used + queued >= ledger.message_budget:
                return f"{NO_STEP}: no messages are left in the allowance. Do not write to the shopper again."
        signature = name + dump(args)
        if name not in finishing and name not in (message_tool, world.role("cart")) and signature in self.recent:
            return f"{NO_STEP}: this same call was just made and nothing has changed; its result is above. Alter the arguments or take another action."
        if name == world.role("cart"):
            refusal = self.verifier.cart_gate(name, args, signature)
            if refusal:
                return refusal
        return name, args

    def order_gate(self) -> str:
        world, ledger, clock = self.world, self.ledger, self.clock
        problems = self.basket_problems()
        # a cart whose lines show no failure in their own facts is held once; a second hold only pushes the loop into declining
        holds = 2 if any((self.judged.get(i) or {}).get("binding") or (self.cards.get(i) or {}).get("in_stock") is False for i in ledger.cart_ids) else 1
        if problems and self.basket_holds < holds and world.steps_left() > 2:
            self.basket_holds += 1
            return HELD_BACK + "The basket check holds the order: " + "; ".join(problems) + ". Fix these, then order."
        missing = self.missing_products() if ledger.items() else []
        if missing and self.fills < 1 and world.steps_left() > 2:
            # an order without a line for every listed product cannot pass
            self.fills += 1
            filled = self.fill_basket() if clock.seconds_left() > WRAP_UP_SECONDS + 10 else ""
            if filled:
                return HELD_BACK + filled
            return (HELD_BACK + "The cart has no line for " + ", ".join(name for name, _ in missing) + ", and an order missing a listed product"
                    " fails. Put the best fitting listing of each missing product in the cart at its count, then order.")
        # after a refusal an earlier backup may have lost its place as cheapest fit
        if (ledger.plan.get("priority") == "cheapest" and ledger.refused_at > ledger.searched_at and self.research_checks < 1
                and world.steps_left() > 10 and clock.seconds_left() > WRAP_UP_SECONDS + 20):
            self.research_checks += 1
            return (HELD_BACK + "The previous order was refused and the request wants the cheapest fitting item. An earlier backup is only right"
                    " when nothing cheaper fits now, so search again before ordering: use the stated requirement words and the category, a price"
                    " limit a little above the refused item's price if the search allows one, and further pages. Open the details of cheaper"
                    " candidates, put the cheapest one meeting every requirement in the cart, then order.")
        if self.cheaper_checks < 2 and world.steps_left() > 10 and clock.seconds_left() > WRAP_UP_SECONDS + 20:
            cheaper = ledger.cheaper_unopened()
            if cheaper:
                self.cheaper_checks += 1
                return (HELD_BACK + "The request wants the cheapest fitting item, yet these cheaper listings from your results were never opened with the details tool: "
                        + ", ".join(f"{ident} ({price})" for ident, price in cheaper)
                        + ". Open their details (several per reply); if one meets every requirement, put the cheapest such listing in the cart instead, and if none does, order again.")
        if not ledger.order_checked and world.steps_left() > 2 and clock.seconds_left() > WRAP_UP_SECONDS:
            ledger.order_checked = True
            verdict = self.verifier.check_cart()
            if verdict:
                return HELD_BACK + verdict
            return (HELD_BACK + "Checklist before ordering: (1) the cart has exactly the lines the request needs (normally one: the chosen item at the"
                    " required quantity) and nothing more; (2) since your last cart change you have re-read this item's stock and price with a tool"
                    " that shows them, it is in stock, and price times quantity fits the current budget; (3) its own properties satisfy every stated"
                    " requirement, every need the shopper revealed and every announced change; (4) if the cheapest or best was requested, no better"
                    " fitting item exists among all you could find; (5) every procedural rule of the request has been followed. Check anything still"
                    " open now; otherwise call the order tool again.")
        return ""

    def closest_hold(self) -> str:
        if not self.lines or self.decline_holds >= 1 or self.world.steps_left() <= 3:
            return ""
        self.decline_holds += 1
        before = list(self.ledger.cart_ids)
        self.cart_picks(closest=True)
        if not self.ledger.cart_ids:
            return ""
        added = [i for i in self.ledger.cart_ids if i not in before]
        return (HELD_BACK + "Races almost never want a decline while something close exists. The cart now holds the closest fit of each product"
                + (f" (code added {', '.join(added)})" if added else "") + ". Order it, unless every candidate of a product fails a need its own facts state.")

    def decline_gate(self) -> str:
        # each decline attempt first gets a separate look for listings that might fit
        if self.decline_checks >= 2 or self.world.steps_left() <= 4 or self.clock.seconds_left() <= WRAP_UP_SECONDS + 30:
            return ""
        self.decline_checks += 1
        audit = self.auditor.review()
        if audit:
            return HELD_BACK + audit
        if self.decline_checks == 1:
            return (HELD_BACK + "Make sure before declining: you searched in several ways (other words, filters, pages), you read every candidate's own"
                    " properties, and each one fails a stated requirement, a revealed need, the budget or stock. Close synonyms and any key spelling"
                    " count, and a requirement you only inferred from the story does not. If something fits, buy it; if nothing does, call the decline tool again.")
        return ""

    def finish_gate(self, runnable) -> str:
        call_id, name, args = runnable[0]
        if self.finisher.is_decline(name, args):
            order = self.proven_order() or self.audited_order()
            if order is not None:
                runnable[0] = (call_id, order["name"], order["args"])
                self.notes[call_id] = ("Code step: this decline was sent as an order instead, because every cart line either passed the separate check"
                                       " or was named a possible fit by the decline review, with no line failing a requirement in its own facts.")
                return ""
            held = self.rescue()
            if held:
                return held
            order, done = self.reorder_refused()
            if order is not None:
                runnable[0] = (call_id, order["name"], order["args"])
                self.notes[call_id] = ("Code step: this decline was sent as an order of the refused cart instead. A refusal is not a reason to decline;"
                                       " ordering the same lines again is." + (f" Cart steps used:\n{done}" if done else ""))
                return ""
            held = self.budget_research() or self.closest_hold() or self.decline_gate()
            if not held:
                runnable[0] = (call_id, name, self.auditor.scrub(args, name))
            return held
        if name != self.world.role("order"):
            return ""
        return self.order_gate()

    def reorder_refused(self) -> tuple[dict[str, Any] | None, str]:
        # a decline right after a refused order: in race 4 every such decline failed, while ordering the same cart again mostly passed
        world, ledger = self.world, self.ledger
        lines = ledger.refused_lines
        order = world.role("order")
        if not order or not lines or self.reorders >= 1 or world.steps_left() < 2 or ledger.refused_at < 0:
            return None, ""
        if any(item in ledger.gone for item in lines) or len(lines) < len(ledger.items()):
            return None, ""
        rows, price, _, _ = ledger.row_fields()
        values = [(rows.get(item) or {}).get(price) for item in lines] if price else []
        if values and all(is_number(v) for v in values) and not ledger.affordable(sum(v * lines[i] for v, i in zip(values, lines))):
            return None, ""
        actions = [self.put_action(item, 0) for item in ledger.cart_ids if item not in lines]
        actions += [self.put_action(item, units) for item, units in lines.items()
                    if item not in ledger.cart_ids or ledger.cart_counts.get(item, 1) != units]
        if None in actions or len(actions) + 1 > world.steps_left():
            return None, ""
        self.reorders += 1
        done = self.run_code_actions(actions) if actions else ""
        return {"name": order, "args": self.finisher.neutral_args(order, "Order the cart.")}, done

    def proven_order(self) -> dict[str, Any] | None:
        world, ledger = self.world, self.ledger
        order = world.role("order")
        if not order or not ledger.cart_ids or self.swaps >= 2 or world.steps_left() < 1:
            return None
        # a basket short of a wanted product fails however good its lines are
        if any(item in ledger.gone for item in ledger.cart_ids) or len(ledger.cart_ids) < len(ledger.items()):
            return None
        if not all(self.verifier.passes(item) for item in ledger.cart_ids):
            return None
        self.swaps += 1
        return {"name": order, "args": self.finisher.neutral_args(order, "Order the cart.")}

    def audited_order(self) -> dict[str, Any] | None:
        # a full cart the decline review itself names as possible fits, none failing its own facts, is ordered: races almost never want a decline
        world, ledger = self.world, self.ledger
        order = world.role("order")
        if not order or not ledger.cart_ids or self.swaps >= 2 or world.steps_left() < 1:
            return None
        if any(item in ledger.gone for item in ledger.cart_ids) or len(ledger.cart_ids) < len(ledger.items()):
            return None
        if ledger.items() and self.missing_products():
            return None
        for ident in ledger.cart_ids:
            if ident not in self.judged:
                self.verifier.check_item(ident, "order", ledger.cart_lines.get(ident, ""))
        if any((self.judged.get(i) or {}).get("binding") or (self.cards.get(i) or {}).get("in_stock") is False for i in ledger.cart_ids):
            return None
        fits = set(self.auditor.look()[0])
        if not all(i in fits or self.verifier.passes(i) for i in ledger.cart_ids):
            return None
        self.swaps += 1
        return {"name": order, "args": self.finisher.neutral_args(order, "Order the cart.")}

    def rescue(self) -> str:
        # a decline with a checked fit on hand: code puts that fit in the cart instead
        world, ledger = self.world, self.ledger
        if self.rescues >= 2 or world.steps_left() < 3 or self.clock.seconds_left() < WRAP_UP_SECONDS + 30:
            return ""
        self.rescues += 1
        if ledger.items():
            return self.basket_rescue() or self.fill_basket()
        fits = self.auditor.look()[0]
        pool = [i for i in dict.fromkeys(self.verifier.passed_items() + fits) if i not in ledger.gone and i not in ledger.cart_ids]
        if ledger.plan.get("priority") != "best":
            fields = ledger.row_fields()
            rows, price = fields[0], fields[1]
            pool.sort(key=lambda i: (rows.get(i) or {}).get(price) if is_number((rows.get(i) or {}).get(price)) else float("inf"))
        for item in pool[:3]:
            if self.verifier.passes(item):
                return self.put_in_cart(item, f"{item} passed the separate check against every current requirement", "Do not decline:")
        return ""

    def basket_rescue(self) -> str:
        # several products: each cart line that fails the check gives way to a checked fit carrying the same label
        world, ledger = self.world, self.ledger
        failing = [item for item in ledger.cart_ids if not self.verifier.passes(item)]
        if not failing:
            return ""
        rows, _, _, kind = ledger.row_fields()
        pool = [i for i in dict.fromkeys(self.verifier.passed_items() + self.auditor.look()[0]) if i not in ledger.gone and i not in ledger.cart_ids]
        actions, named = [], []
        for item in failing[:2]:
            label = (rows.get(item) or {}).get(kind) if kind else None
            for other in [i for i in pool if label and (rows.get(i) or {}).get(kind) == label][:2]:
                units = ledger.cart_counts.get(item, 1)
                swap = [self.put_action(item, 0), self.put_action(other, units)]
                if None in swap or self.verifier.failed(other) or not self.verifier.passes(other):
                    continue
                actions += swap
                named.append(f"{other} in place of {item}")
                pool.remove(other)
                break
        if not actions or len(actions) + 1 > world.steps_left():
            return ""
        done = self.run_code_actions(actions)
        return ("Do not decline: some cart lines failed the separate check, and checked fits with the same label were found, so code"
                f" swapped them ({'; '.join(named)}). These steps were used:\n{done}\nNow call the order tool.")

    @staticmethod
    def label_fits(name, label) -> bool:
        words = [w for w in re.split(r"[^a-z0-9]+", str(name).lower()) if len(w) >= 3]
        parts = [w for w in re.split(r"[^a-z0-9]+", str(label).lower()) if len(w) >= 3]
        return any(w[:4] == part[:4] for w in words for part in parts)

    def missing_products(self, fields=None) -> list[tuple[str, int]]:
        ledger = self.ledger
        rows, _, _, kind = fields or ledger.row_fields()
        if not kind:
            return []
        covered = {(rows.get(i) or {}).get(kind) for i in ledger.cart_ids}
        return [(name, units) for name, units in ledger.items() if not any(label and self.label_fits(name, label) for label in covered)]

    def fill_basket(self) -> str:
        # a basket decline almost never passed in races 3 and 4, so each product with no cart line gets the cheapest opened listing of its label that passes the check
        world, ledger = self.world, self.ledger
        fields = ledger.row_fields()
        rows, price, stock, kind = fields
        if not price or not kind or self.clock.cost > COST_STOP:
            return ""
        total = sum((rows.get(i) or {}).get(price) * ledger.cart_counts.get(i, 1) for i in ledger.cart_ids if is_number((rows.get(i) or {}).get(price)))
        actions, named, checks = [], [], 0
        for name, units in self.missing_products(fields):
            pool = sorted((row.get(price), ident) for ident, row in rows.items()
                          if ident in ledger.opened and ident not in ledger.gone and is_number(row.get(price))
                          and not (stock and row.get(stock) is False) and self.label_fits(name, row.get(kind) or ""))
            pick = next((ident for _, ident in pool if ident in self.verifier.passed_items()), None)
            for _, ident in pool:
                if pick or checks >= 4:
                    break
                if self.verifier.failed(ident):
                    continue
                checks += 1
                # the order itself is checked again on the stronger LLM, so these looks may use the cheaper one
                if self.verifier.passes(ident, "buy"):
                    pick = ident
            action = self.put_action(pick, units) if pick else None
            if action is None:
                return ""
            actions.append(action)
            named.append(f"{pick} for the {name}")
            total += rows[pick][price] * units
        if not actions or len(actions) + 1 > world.steps_left() or not ledger.affordable(total):
            return ""
        done = self.run_code_actions(actions)
        return ("Every product the request lists now has a cart line: the missing ones got an opened listing that passed the separate check,"
                f" put in by code ({'; '.join(named)}). These steps were used:\n{done}\nNow call the order tool; do not decline.")

    def budget_research(self) -> str:
        world, ledger = self.world, self.ledger
        if self.budget_checks >= 1 or ledger.budget_changed_at <= ledger.searched_at or world.steps_left() <= 8:
            return ""
        if self.clock.seconds_left() <= WRAP_UP_SECONDS + 30:
            return ""
        self.budget_checks += 1
        return (HELD_BACK + f"The budget changed to {ledger.budget_now} after your last search, so the store has not been searched under it."
                " Search again (the requirement words and the category, a price filter if the search offers one, further pages), open the"
                " cheapest candidates, and buy one that meets every requirement within the new budget. A smaller budget is no reason to decline.")

    def put_action(self, ident, units) -> dict[str, Any] | None:
        world, ledger = self.world, self.ledger
        tool = world.role("cart")
        if not tool:
            return None
        schema = world.schemas[tool]
        properties = schema.get("properties") or {}
        count = ledger.count_field(properties)
        if ledger.add_template:
            # the shopper's own earlier add is the surest pattern for this cart tool
            args = dict(ledger.add_template)
            target = ledger.cart_target(properties, args)
            field = next((k for k, v in args.items() if v == target), None)
        else:
            if any(isinstance(spec, dict) and "enum" in spec for k, spec in properties.items() if k in (schema.get("required") or [])):
                return None
            texts = [k for k, spec in properties.items() if isinstance(spec, dict) and spec.get("type") == "string" and "enum" not in spec]
            field = next((k for k in texts if IDENT_KEY.search(k)), None) or (texts[0] if len(texts) == 1 else None)
            args = {}
        if not field:
            return None
        args[field] = ident
        if count:
            floor = properties[count].get("minimum")
            adds = is_number(floor) and floor < 0
            args[count] = units - ledger.cart_counts.get(ident, 0) if adds else units
            if adds and args[count] == 0:
                return None
        args, problems = SchemaGuard.check(args, schema, "args")
        return None if problems else {"name": tool, "args": args}

    def cart_plan(self, item) -> list[dict[str, Any]] | None:
        ledger = self.ledger
        actions = []
        for ident in ledger.cart_ids:
            if ident == item:
                continue
            removal = self.put_action(ident, 0)
            if removal is None:
                return None
            actions.append(removal)
        if item not in ledger.cart_ids or ledger.cart_counts.get(item, 1) != ledger.wanted_units():
            adding = self.put_action(item, ledger.wanted_units())
            if adding is None:
                return None
            actions.append(adding)
        return actions

    def put_in_cart(self, item, why, lead) -> str:
        world = self.world
        actions = self.cart_plan(item)
        if not actions or len(actions) + 1 > world.steps_left():
            return HELD_BACK + f"{lead} {why}. Put it in the cart as the only line, at the required quantity, then call the order tool."
        done = self.run_code_actions(actions)
        return (f"{lead} {why}, so code set the cart to it (these steps were used):\n{done}\n"
                "Now call the order tool. Decline only if that order is refused and nothing else in the store fits.")

    def run_code_actions(self, actions) -> str:
        reply = self.world.call(actions)
        self.trace.append({"kind": "code_cart", "step": self.world.step, "actions": actions, "reply": reply})
        if reply is None:
            return "(the cart call did not go through)"
        observations = reply.get("observations") if isinstance(reply.get("observations"), list) else [reply.get("observation")]
        lines = []
        for action, observation in zip(actions, observations):
            note = self.absorb(action["name"], action["args"], observation, reply)
            lines.append(f"{action['name']} {dump(action['args'])} -> {dump(observation)[:1500]}" + (f" [note] {note}" if note else ""))
        return "\n".join(lines)

    def basket_rows_note(self) -> str:
        ledger = self.ledger
        products = ledger.items()
        if not products or self.world.steps_left() <= 12:
            return ""
        rows, price, stock, kind = ledger.row_fields()
        if not price or not kind:
            return ""
        groups = {}
        for ident, row in rows.items():
            value, label = row.get(price), row.get(kind)
            if not is_number(value) or not isinstance(label, str) or ident in ledger.gone or (stock and row.get(stock) is False):
                continue
            groups.setdefault(label, []).append((value, ident))
        marks = []
        for name, _ in products:
            for label, pool in groups.items():
                if not self.label_fits(name, label):
                    continue
                read = [value for value, ident in pool if ident in ledger.opened]
                # rows carry no details, so a cheaper row nobody opened may still be the fit
                unread = sorted((value, ident) for value, ident in pool if ident not in ledger.opened and ident not in ledger.cart_ids
                                and (not read or value < min(read)))[:3]
                if unread:
                    marks.append(f"{name}: " + ", ".join(f"{ident} {value}" for value, ident in unread))
        if not marks:
            return ""
        return ("\n[cheaper rows not opened yet, per product] " + "; ".join(marks)
                + " (rows show no details: open these as well, besides searching each product by its requirement words, before settling on a dearer listing)")

    def cheapest_note(self) -> str:
        ledger = self.ledger
        if ledger.items():
            return self.basket_rows_note()
        if ledger.plan.get("priority") != "cheapest":
            return ""
        pool = ledger.candidates()[:6]
        if not pool:
            return ""
        marks = []
        for value, ident in pool:
            state = self.verifier.known(ident)
            if ident in ledger.cart_ids:
                state = "in the cart" + (", " + state if state else "")
            marks.append(f"{ident} {value} ({state or ('details read' if ident in ledger.opened else 'details not read')})")
        return "\n[cheapest rows seen so far, in stock and within budget, lowest first] " + "; ".join(marks)

    def paging_note(self, name, args, observation) -> str:
        world, ledger = self.world, self.ledger
        if ledger.plan.get("priority") != "cheapest" or "search" not in world.roles_of(name) or world.steps_left() <= 15:
            return ""
        inner = event_split(observation)[0]
        if not isinstance(inner, dict):
            return ""
        shown = cart_lines_in(inner) or []
        counts = [v for v in inner.values() if isinstance(v, int) and not isinstance(v, bool)]
        total = max(counts) if counts else None
        if not total or total > 72 or len(shown) >= total:
            return ""
        seen = self.page_seen.setdefault(name + dump({k: v for k, v in args.items() if not is_number(v)}), set())
        seen.update(row_ident(row) for row in shown)
        if len(seen) >= total:
            return ""
        return (f"This search matched {total} listings and {len(seen)} of them have been shown. Rows do not come in price order, so the"
                " cheapest fit may sit on a page not yet seen: fetch the remaining pages before choosing.")

    def wind_up(self) -> None:
        ledger = self.ledger
        if self.lines and not self.world.over():
            self.cart_picks(closest=True)
            covered = {self.product_of.get(i) for i in ledger.cart_ids}
            missing = [line.get("category") for line in self.lines if line.get("category") not in covered]
            if missing and len(ledger.cart_ids) < len(self.lines) and len(self.lines) > 1 and self.finisher.world.role("decline"):
                # an order without every listed product cannot pass, so a partial basket is not forced through
                self.trace.append({"kind": "partial_basket", "missing": missing})
                self.finisher.send(self.finisher.decline_action())
                return
        if not ledger.cart_ids and not ledger.items() and self.world.steps_left() >= 2 and self.clock.seconds_left() > 4:
            pool = [i for i in self.verifier.passed_items() if i not in ledger.gone]
            if ledger.plan.get("priority") != "best":
                pool = [i for _, i in ledger.candidates() if i in pool] or pool
            actions = self.cart_plan(pool[0]) if pool else None
            if actions and len(actions) < self.world.steps_left():
                self.run_code_actions(actions)
        self.finisher.force()

    def run_calls(self, calls, planned=False) -> list[tuple[str, str]]:
        world = self.world
        allowed = set(world.tool_names(self.offered_tools()))
        finishing = {world.role("order"), world.role("decline")} - {None}
        replies = []
        runnable = []
        queued = 0
        for call in calls:
            outcome = self.screen(call, len(calls), allowed, finishing, planned, queued)
            if isinstance(outcome, str):
                replies.append((call["id"], outcome))
                continue
            name, args = outcome
            if name == world.role("message"):
                queued += 1
            runnable.append((call["id"], name, args))
        if runnable and runnable[0][1] in finishing:
            held = self.finish_gate(runnable)
            if held:
                return replies + [(runnable[0][0], held)]
        elif runnable and finishing:
            keep = max(0, world.steps_left() - 1)
            if keep < len(runnable):
                replies += [(call_id, f"{NO_STEP}: just one step is left, and it has to be the order (or the decline).") for call_id, _, _ in runnable[keep:]]
                runnable = runnable[:keep]
        for start in range(0, len(runnable), 16):
            self.run_chunk(runnable[start : start + 16], replies)
        position = {c["id"]: i for i, c in enumerate(calls)}
        replies.sort(key=lambda r: position.get(r[0], 0))
        return replies

    def run_chunk(self, chunk, replies) -> None:
        world = self.world
        if world.over() or world.steps_left() <= 0:
            replies += [(call_id, "Not run: the episode has ended or no steps remain.") for call_id, _, _ in chunk]
            return
        chunk = chunk[: world.steps_left()]
        actions = [{"name": name, "args": args} for _, name, args in chunk]
        before = world.step
        reply = world.call(actions)
        self.trace.append({"kind": "env", "step": before + 1, "actions": actions, "reply": reply})
        if reply is None:
            replies += [(call_id, "The call did not go through (network or server). Try it again.") for call_id, _, _ in chunk]
            return
        observations = reply.get("observations") if isinstance(reply.get("observations"), list) else [reply.get("observation")]
        for index, (call_id, name, args) in enumerate(chunk):
            if index >= len(observations):
                replies.append((call_id, "Not run: the episode ended before this call."))
                continue
            note = " ".join(n for n in (self.notes.pop(call_id, ""), self.absorb(name, args, observations[index], reply)) if n)
            text = dump(observations[index])
            if index == len(observations) - 1 and reply.get("finished"):
                text += " [episode over]"
            if note:
                text += "\n[note] " + note
            replies.append((call_id, text))

    def book(self, name, args, observation) -> None:
        world, ledger = self.world, self.ledger
        hits = world.roles_of(name)
        if "message" in hits:
            ledger.messages_used += 1
            sent = " ".join(str(v) for v in args.values() if isinstance(v, str)).strip().lower()
            failed = isinstance(observation, dict) and isinstance(observation.get("error"), str)
            # a failed message may be sent again later
            if sent and not failed:
                ledger.questions_sent.append(sent)
        if "order" in hits:
            ledger.order_attempts += 1
        if "cart" in hits:
            ledger.cart_used = True
            ledger.order_checked = False
        resets = hits & {"cart", "message", "order", "decline"}
        self.recent = [] if resets else (self.recent + [name + dump(args)])[-2:]
        ledger.records.append(observation)
        if "cart" in hits and not error_text(observation):
            ledger.note_cart_call(world.schemas[name].get("properties") or {}, args, world.step)
        if hits & {"cart", "cart_view"}:
            # the cart's own report of its lines overrides what the calls suggest
            ledger.sync_cart(observation)
        if "search" in hits:
            ledger.searched_at = world.step
        if hits & {"details", "view"}:
            ledger.opened.update(v for v in args.values() if isinstance(v, str))
            self.track_pointer(args, observation)
        if hits & {"details", "view", "batch_view"}:
            ledger.opened.update(v for value in args.values() if isinstance(value, list) for v in value if isinstance(v, str))
        if hits & {"details", "view", "batch_view"} and not error_text(observation):
            # what an open shows (stock, price, facts) reaches the cards the judge and the gates read, whoever made the call
            named = [v for value in args.values() for v in (value if isinstance(value, list) else [value]) if isinstance(v, str)]
            for ident in named:
                node = StoreDesk.node_of(observation, ident)
                if node:
                    self.card(ident, node)
        if "cart_view" in hits:
            ledger.last_cart_view = observation

    def track_pointer(self, args, observation) -> None:
        ledger = self.ledger
        inner = event_split(observation)[0]
        ident = next((v for v in args.values() if isinstance(v, str)), None)
        if not ledger.lookups or not ident or not isinstance(inner, dict) or error_text(observation) or ident in ledger.pointed:
            return
        texts = set()

        def gather(value, depth) -> None:
            if isinstance(value, str) and len(value) <= 60:
                texts.add(value.strip().lower())
            elif isinstance(value, dict) and depth < 3:
                for child in value.values():
                    gather(child, depth + 1)

        gather(inner, 0)
        lookup = next((entry for entry in ledger.lookups if entry["seller"].lower() in texts), None)
        if lookup is None:
            # a listing opened soon after a pointed one is most likely the next link of its chain
            lookup = next((entry for entry in ledger.lookups if entry["found"] and entry["follow"] < 3), None)
            if lookup is None:
                return
            lookup["follow"] += 1
        else:
            # verdicts given before the pointed listing was read are judged again
            if lookup["found"] is None:
                ledger.version += 1
            # one seller may have several listings; the details tell which one is meant
            lookup["found"] = ", ".join(filter(None, [lookup["found"], ident]))
        ledger.pointed.append(ident)

    def pointer_note(self) -> str:
        ledger = self.ledger
        if not ledger.lookups or self.world.steps_left() <= 16:
            return ""
        notes = []
        for entry in ledger.lookups:
            name = f"the {entry['kind'] or 'listing'} sold by {entry['seller']}" + (f" (the one with {entry['details']}: these details only identify that listing, they are not values for the item to buy)" if entry["details"] else "")
            if entry["found"] is None:
                notes.append(f"a requirement takes its value from {name}: find it (search its kind together with the seller's name), open its details and read the value there before choosing")
            elif entry["follow"] < 3:
                notes.append(f"for {name}, listings of that seller opened so far: {entry['found']} (if none has those details, open another of that seller's listings); if its details pass the value on to yet another listing, find and open that one too, since the last listing of such a chain states the value; and if a listing of the chain states a store rule that pairs one value with another, the requirement takes the value paired with the one found")
        return "\n[pointer] " + "; ".join(notes) + "." if notes else ""

    def allowance_note(self, name, observation) -> str:
        world, ledger = self.world, self.ledger
        problem = observation.get("error") if name == world.role("message") else error_text(observation)
        if not isinstance(problem, str) or not problem or world.roles_of(name) & {"order", "decline", "cart", "cart_view"}:
            return ""
        if not ALLOWANCE_SPENT.search(problem):
            return ""
        if name == world.role("message"):
            used = ledger.messages_used
            ledger.message_budget = used if ledger.message_budget is None else min(ledger.message_budget, used)
        world.capped.add(name)
        return ("This tool's allowance is spent and it is no longer offered. Do not call it again: take prices and stock from search"
                " results or the cart view, and decide with what you already know.")

    def refusal_notes(self, lines) -> list[str]:
        ledger = self.ledger
        ledger.note_refusal(self.world.step)
        notes = ["THE ORDER WAS REFUSED; the episode goes on and the cart is as it was. Apply every change the refusal names, repair"
                 " the cart (drop any line no longer acceptable, such as a sold-out item), re-check the stock and price of the new pick,"
                 " and order again."]
        wording = " ".join(lines)
        if PRICE_TALK.search(wording) and not GONE_WORDS.search(wording):
            notes.append("This refusal does not say the item is gone: if it only reports a new price (or another change the item still"
                         " satisfies), the item can still be bought. Read its current price again; if price times quantity still fits the"
                         " budget and it still meets every requirement, order it again at once. A price change is never a reason to decline.")
        return notes

    def observe(self, name, args, observation, refused) -> str:
        if isinstance(observation, (dict, list)):
            harvest_ids(observation, self.ledger.seen_ids)
        if not isinstance(observation, dict):
            return ""
        notes = [self.allowance_note(name, observation), self.paging_note(name, args, observation)]
        if refused:
            notes += self.refusal_notes(refused)
        elif len(observation) >= 2 and name != self.world.role("message"):
            lines = []
            for value in observation.values():
                if isinstance(value, list) and value and all(isinstance(v, str) for v in value):
                    gather_sentences(value, lines)
            if lines:
                notes.append("This result holds a line from the shopper announcing a change. Apply it: " + " | ".join(lines[:4]))
        return " ".join(note for note in notes if note)

    def order_followup(self, name, observation, reply, refused) -> str:
        world = self.world
        if name != world.role("order") or name == world.role("decline") or reply.get("finished"):
            return ""
        if refused:
            return ""
        if isinstance(observation, dict) and awaits_next_step(observation):
            return "The order needs one more step (it hands over a token or asks for confirmation): take that step now with the tool that completes checkout. No purchase happens until the episode ends."
        # the episode did not end, so the order was refused, failed or is waiting
        self.ledger.note_refusal(world.step)
        return ("THE ORDER CALL LEFT THE EPISODE RUNNING, SO NO PURCHASE HAS HAPPENED, whatever the result claims (a successful order ends the episode)."
                " If the result asks for another step (a confirmation, say, with a token or code it supplies), take it now. If the order was refused or"
                " failed, read why, apply every change it names, repair the cart, re-check the pick's stock and price, and order again; a changed price"
                " alone does not rule an item out while price times quantity stays within the budget.")

    def shopper_lines(self, name, args, observation, refused) -> tuple[list[str], str | None]:
        world = self.world
        heard = []
        if name == world.role("message"):
            if isinstance(observation, str):
                heard = [observation]
            elif isinstance(observation, (dict, list)) and not (isinstance(observation, dict) and "error" in observation):
                if isinstance(observation, dict):
                    gather_sentences([v for v in observation.values() if isinstance(v, (str, list))], heard)
                if not heard and holds_words(observation):
                    # a terse or structured answer still counts as the shopper's reply
                    heard = [dump(observation)[:600]]
            return heard, " ".join(str(v) for v in args.values() if isinstance(v, str))
        if not isinstance(observation, dict):
            return heard, None
        wrapped = len(observation) <= 3 and any(isinstance(v, (dict, list)) for v in observation.values())
        for value in observation.values():
            if isinstance(value, list) and value and all(isinstance(v, str) for v in value):
                gather_sentences(value, heard)
            elif not isinstance(value, str):
                continue
            elif refused and len(value.split()) >= 4:
                heard.append(value)
            elif wrapped and len(value.split()) >= 6 and (FIRST_PERSON.search(value) or name == world.role("order")):
                heard.append(value)
        return heard, None

    def absorb(self, name, args, observation, reply) -> str:
        world = self.world
        self.book(name, args, observation)
        # a finish that leaves the episode running was refused; its sentences say why
        finishing = bool(world.roles_of(name) & {"order", "decline"}) and not reply.get("finished")
        if finishing and not self.finisher.is_decline(name, args) and self.ledger.cart_ids:
            self.ledger.refused_lines = {ident: self.ledger.cart_counts.get(ident, 1) for ident in self.ledger.cart_ids}
        refused = refusal_lines(observation) if finishing else []
        if finishing and isinstance(observation, dict) and "refusal" in observation and not refused:
            refused = [dump(observation["refusal"])]
        notes = [self.observe(name, args, observation, refused), self.order_followup(name, observation, reply, refused)]
        heard, question = self.shopper_lines(name, args, observation, bool(refused))
        if heard:
            notes.append(self.interpreter.hear(heard, question))
        notes.append(self.restock_swap())
        return " ".join(n for n in notes if n)

    def restock_swap(self) -> str:
        # a cart line that turns out sold out gives way at once to the best checked fit of its product that is still on the shelf
        ledger = self.ledger
        if getattr(self, "swapping", False) or self.world.steps_left() < 4:
            return ""
        sold = [i for i in ledger.cart_ids if (self.cards.get(i) or {}).get("in_stock") is False and i not in self.restocked]
        actions, named = [], []
        for item in sold:
            self.restocked.add(item)
            product = self.product_of.get(item)
            pool = [i for i, v in self.judged.items() if v.get("fits") is True and i in ledger.opened and i not in ledger.cart_ids and i not in ledger.gone
                    and (self.cards.get(i) or {}).get("in_stock") is not False and (self.product_of.get(i) == product or len(self.lines) <= 1)]
            if ledger.plan.get("priority") == "cheapest":
                pool.sort(key=lambda i: (self.cards.get(i) or {}).get("price") or 0)
            swap = [self.put_action(item, 0), self.put_action(pool[0], ledger.cart_counts.get(item, 1))] if pool else [None]
            if None not in swap:
                actions += swap
                named.append(f"{pool[0]} in place of {item}")
        if not actions or len(actions) + 2 > self.world.steps_left():
            return ""
        self.swapping = True
        try:
            done = self.run_code_actions(actions)
        finally:
            self.swapping = False
        return f"Code step: a cart line is sold out, so code swapped it for a checked fit ({'; '.join(named)}):\n{done}"

    def next_message(self) -> dict[str, Any] | None:
        return self.llm.complete(self.messages, self.offered_tools(), [MAIN_MODEL, BACKUP_MODEL, MAIN_MODEL], 2500)

    def bounce_early(self, content, calls) -> bool:
        # with too few listings opened, planned questions would be guesses
        if self.early_bounced or self.ledger.property_menu()[1] >= 3 or self.world.steps_left() <= 12:
            return False
        if self.clock.seconds_left() <= WRAP_UP_SECONDS + 60:
            return False
        self.early_bounced = True
        self.messages.append({"role": "assistant", "content": content or None, "tool_calls": calls})
        held = (f"{NO_STEP}: before you choose, open the details of two or three more listings of the requested kind"
                " (several per reply, favouring ones unlike those already seen); the shopper questions are planned from what they show.")
        self.post_replies([(c["id"], held) for c in calls])
        return True

    def card(self, ident, node) -> dict[str, Any]:
        # one listing in the engines' row form: the price as the file reads prices elsewhere, stock from a flag or from words saying it is gone
        fields = flat_fields(node)
        numbers = {k: v for k, v in fields.items() if is_number(v)}
        priced = [k for k in numbers if "price" in k.lower()] or (list(numbers) if len(numbers) == 1 else [])
        price = numbers[priced[0]] if priced else None
        flags = [v for v in fields.values() if isinstance(v, bool)]
        gone = any(isinstance(v, str) and len(v) <= 40 and GONE_WORDS.search(v) for v in fields.values())
        stock = False if gone else flags[0] if len(flags) == 1 else None
        kind = next((v for k, v in fields.items() if isinstance(v, str) and KIND_KEY.search(k) and len(v) <= 80), None)
        row = {k: v for k, v in fields.items() if isinstance(v, str) and v != ident}
        row.update(listing_from_record(ident, node, price, stock, kind))
        known = self.cards.get(ident) or {}
        if len(row.get("facts") or {}) < len(known.get("facts") or {}):
            row["facts"] = known["facts"]
        self.cards[ident] = {**known, **{k: v for k, v in row.items() if v not in (None, "", {})}}
        return dict(self.cards[ident])

    def card_of(self, item) -> dict[str, Any] | None:
        for record in reversed(self.ledger.records):
            node = StoreDesk.node_of(record, item)
            if node:
                return self.card(item, node)
        return self.cards.get(item)

    def code_call(self, name, args) -> Any:
        reply = self.world.call([{"name": name, "args": args}])
        self.trace.append({"kind": "stage", "step": self.world.step, "name": name, "args": args, "reply": reply})
        if reply is None:
            return None
        observations = reply.get("observations") if isinstance(reply.get("observations"), list) else [reply.get("observation")]
        note = self.absorb(name, args, observations[0] if observations else None, reply)
        if note:
            self.stage_notes.append(note)
        return observations[0] if observations else None

    def stages(self) -> None:
        # ask, then search each product in its own share of steps, then judge each product's leaders and cart the picks
        world = self.world
        store = StoreDesk(self)
        if not store.ready() or world.steps_left() < 8:
            return
        asked = ask_engine({"request": self.ledger.query}, store)
        self.trace.append({"kind": "ask_stage", "record": asked.get("record"), "lines": len(asked.get("lines") or []), "refused": asked.get("refused")})
        lines = [line for line in asked.get("lines") or [] if isinstance(line, dict) and isinstance(line.get("record"), dict)]
        if not lines:
            return
        self.lines[:] = lines
        # the ask stage read the products against the shop's own categories: its lines are the basket from here on
        self.ledger.plan["items"] = [{"product": line.get("category"), "quantity": line["record"].get("quantity") or 1} for line in lines] if len(lines) > 1 else []
        # the planner stays on only when the ask stage could not reach the shopper
        self.planner.active = self.ledger.messages_used == 0
        prefix = len(lines) > 1
        self.ledger.current = [f"{line.get('category')} " * prefix + f"{r.get('name')}: {r.get('value')} (from the {'shopper' if r.get('source') == 'hidden_need' else 'request'})"
                               for line in lines for r in line["record"].get("requirements") or [] if isinstance(r, dict)]
        self.verifier.revealed_before = len(self.ledger.revealed)
        for refused in asked.get("refused") or []:
            if is_number(refused.get("length")):
                SEEN["limit"] = min(SEEN["limit"], int(refused["length"]) - 1)
        # room after the stages for the cart, the order and one refusal or event to repair, in proportion to the task's step allowance
        finish = len(lines) + 2 + max(EVENT_RESERVE, world.max_steps // 8)
        engines = []
        for position, line in enumerate(lines):
            share = max(2, (world.steps_left() - finish) // (len(lines) - position))
            store.limit = world.step + share
            engine = Engine({"record": line["record"], "seed": line}, store)
            try:
                engine.run()
            except Exception as problem:
                self.trace.append({"kind": "search_stage", "line": position, "error": f"{type(problem).__name__}: {problem}"[:200]})
            store.limit = None
            engines.append(engine)
            for ident in engine.rows:
                self.product_of.setdefault(ident, line.get("category"))
        ask = self.verifier.ask
        for line, engine in zip(lines, engines):
            record = line["record"]
            ranked = engine.pool()
            # every opened row is judged (two cheap batches at most); unopened rows only when nothing was opened
            leaders = [i for i in ranked if i in engine.viewed][:JUDGED] or ranked[:SHORTLIST]
            listings = [self.cards.get(i) or {**engine.rows[i], "id": i} for i in leaders]
            needs = add_bounds(ask, bound_needs(engine)) if engine.requirements else []
            line["needs"] = needs
            verdicts = text_shown(judge_listings(ask, needs, listings, quantity=record.get("quantity") or 1), needs, listings) if listings and needs else {}
            for ident, verdict in verdicts.items():
                self.judged[ident] = verdict
            pick = choose_listing(verdicts, listings, record.get("quantity") or 1, bool(record.get("cheapest"))) if verdicts else None
            if not needs and leaders:
                pick = min(leaders, key=lambda i: listing_cost(self.cards.get(i) or engine.rows[i], record.get("quantity") or 1)) if record.get("cheapest") else leaders[0]
            for _ in range(2):
                # where the offer lives apart from the properties, a pick's stock is read from the offer before it reaches the cart
                if not pick or not verdicts or store.refresh(pick).get("in_stock") is not False:
                    break
                self.ledger.gone.add(pick)
                verdicts[pick] = self.judged[pick] = {**verdicts.get(pick, {}), "fits": False, "binding": True, "why": "sold out on its offer"}
                pick = choose_listing(verdicts, listings, record.get("quantity") or 1, bool(record.get("cheapest")))
            self.picks[line.get("category")] = {"pick": pick, "ranked": ranked[:6], "quantity": record.get("quantity") or 1}
            self.trace.append({"kind": "judge_stage", "product": line.get("category"), "pick": pick, "verdicts": verdicts})
        self.cart_picks()

    def cart_picks(self, closest=False) -> None:
        # the verified pick of each product; at the end, a product still missing takes its closest ranked fit instead
        actions = []
        covered = {self.product_of.get(i) for i in self.ledger.cart_ids}
        for product, chosen in self.picks.items():
            choice = chosen["pick"]
            if not choice and closest and product not in covered:
                choice = next((i for i in chosen["ranked"] if i not in self.ledger.gone and not (self.judged.get(i) or {}).get("binding")), None)
            if choice and choice not in self.ledger.cart_ids and product not in covered:
                action = self.put_action(choice, chosen["quantity"])
                if action:
                    actions.append(action)
        if actions and len(actions) + 2 <= self.world.steps_left():
            self.stage_notes.append("code put the picks in the cart:\n" + self.run_code_actions(actions))

    def stage_brief(self) -> str:
        if not self.lines:
            return ""
        parts = ["[engines] Code already asked the shopper, searched every product and checked the leaders against their needs:"]
        for line in self.lines:
            record = line["record"]
            chosen = self.picks.get(line.get("category")) or {}
            needs = "; ".join(f"{r.get('name')}: {r.get('value')}" for r in record.get("requirements") or [] if isinstance(r, dict)) or "none stated"
            ranked = ", ".join(f"{i} ({(self.cards.get(i) or {}).get('price')})" for i in chosen.get("ranked") or [])
            pick = chosen.get("pick")
            status = f"carted {pick}, which meets every need" if pick and pick in self.ledger.cart_ids else "no listing confirmed yet"
            parts.append(f"- {line.get('category')} x{record.get('quantity') or 1} (needs: {needs}): {status}; ranked candidates: {ranked or 'none'}")
        parts.append("Finish from here: for a product with no confirmed listing, open the ranked candidates or search again; keep one cart line per product"
                     " at its count within the budget; then order. Do not ask the shopper again unless a message is left and a need is still unknown.")
        if self.stage_notes:
            parts.append("Notes from those steps: " + " ".join(self.stage_notes)[-3000:])
        return "\n".join(parts)

    def basket_problems(self) -> list[str]:
        ledger = self.ledger
        if not self.lines:
            return []
        products = [{"name": line.get("category"), "noun": line.get("category"), "quantity": line["record"].get("quantity") or 1} for line in self.lines]
        names = [p["name"] for p in products]
        cart = []
        for ident in ledger.cart_ids:
            if ident not in self.judged:
                # a line nobody has judged yet is judged now, so the basket is never ordered on trust
                self.verifier.check_item(ident, "order", ledger.cart_lines.get(ident, ""))
            card = self.cards.get(ident) or self.card_of(ident) or {}
            verdict = self.judged.get(ident) or {}
            product = self.product_of.get(ident) or (names[0] if len(names) == 1 else None)
            cart.append({"id": ident, "product": product, "quantity": ledger.cart_counts.get(ident, 1), "price": card.get("price"),
                         "verified": verdict.get("fits") if isinstance(verdict.get("fits"), bool) else None,
                         "binding": verdict.get("why") if verdict.get("binding") else "", "in_stock_problem": card.get("in_stock") is False})
        cheaper = {}
        for line in self.lines:
            record = line["record"]
            if not record.get("cheapest"):
                continue
            mine = [c for c in cart if c["product"] == line.get("category")]
            fits = [i for i, v in self.judged.items() if v.get("fits") is True and self.product_of.get(i) == line.get("category") and i not in ledger.gone]
            costs = {i: listing_cost(self.cards.get(i) or {}, record.get("quantity") or 1) for i in fits}
            if mine and costs and is_number(mine[-1]["price"]):
                best = min(costs, key=costs.get)
                if best != mine[-1]["id"] and costs[best] < mine[-1]["price"] * (record.get("quantity") or 1):
                    cheaper[line.get("category")] = {"fit": best}
        budget = ledger.budget_now if is_number(ledger.budget_now) and ledger.budget_changed_at >= 0 else self.lines[0]["record"].get("budget")
        prices = sorted(c["price"] * (c["quantity"] or 1) for c in cart if is_number(c["price"]))
        if is_number(budget) and prices and prices[len(prices) // 2] > 5 * budget:
            # a budget told in whole money against a store that prices in cents
            budget = budget * 100
        return basket_check(cart, products, budget, cheaper)

    def opening_message(self) -> str:
        user = "Shopper request:\n" + self.ledger.query
        if self.opening is not None:
            user += "\n\nOpening observation:\n" + dump(self.opening)
        brief = self.setup.opening_brief()
        if brief:
            user += "\n\n" + brief
        staged = self.stage_brief()
        if staged:
            user += "\n\n" + staged
        return user + "\n\n" + self.status()

    def run(self) -> list[dict[str, Any]]:
        world, ledger = self.world, self.ledger
        if world.finished:
            return self.trace
        self.setup.read_request()
        self.finisher.finish_modes()
        self.trace.append({"kind": "plan", "plan": ledger.plan, "roles": world.roles})
        try:
            self.stages()
        except Exception as problem:
            self.trace.append({"kind": "stage_crash", "error": f"{type(problem).__name__}: {problem}"[:500]})
        user = self.opening_message()
        if self.opening is not None:
            ledger.records.append(self.opening)
            harvest_ids(self.opening, ledger.seen_ids)
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]
        self.loop()
        if not world.over():
            self.wind_up()
        self.trace.append({"kind": "end", "finished": world.finished, "steps": world.step, "cost": round(self.clock.cost, 5), "seconds": round(self.clock.elapsed(), 1)})
        return self.trace

    def turn_calls(self, message, counter) -> tuple[list[dict[str, Any]], int]:
        calls = []
        for call in message.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            counter += 1
            function = call.get("function") or {}
            arguments = function.get("arguments")
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments or {})
            calls.append({"id": call.get("id") or f"call_{counter}", "type": "function", "function": {"name": function.get("name"), "arguments": arguments}})
        return calls, counter

    def loop(self) -> None:
        world, ledger = self.world, self.ledger
        idle = failures = counter = 0
        for _ in range(MAX_TURNS):
            if world.over() or world.steps_left() <= 0:
                return
            if self.clock.seconds_left() < FORCE_SECONDS:
                self.wind_up()
                return
            if self.clock.cost > COST_STOP + COST_PER_PRODUCT * max(0, len(self.lines) - 1):
                self.trace.append({"kind": "spend_stop", "cost": round(self.clock.cost, 5)})
                self.wind_up()
                return
            if ledger.pending_reply:
                text, ledger.pending_reply = ledger.pending_reply, ""
                if self.answer_shopper(text):
                    continue
            if self.planner.active and self.ask_round():
                continue
            message = self.next_message()
            if message is None:
                failures += 1
                if failures >= 2:
                    self.wind_up()
                    return
                continue
            calls, counter = self.turn_calls(message, counter)
            content = message.get("content") or ""
            if not isinstance(content, str):
                content = dump(content)
            committing = {world.role("cart"), world.role("order"), world.role("decline")} - {None}
            if self.planner.active and any(c["function"]["name"] in committing for c in calls):
                if self.bounce_early(content, calls) or self.ask_round(force=True):
                    continue
            if not calls:
                idle += 1
                self.messages.append({"role": "assistant", "content": content or "(no tool call)"})
                self.messages.append({"role": "user", "content": "Each reply has to call at least one of the supplied tools. Carry on.\n" + self.status()})
                if idle >= 3:
                    self.wind_up()
                    return
                continue
            idle = 0
            self.trace.append({"kind": "model", "content": content[:2000], "calls": [c["function"] for c in calls]})
            self.messages.append({"role": "assistant", "content": content or None, "tool_calls": calls})
            self.post_replies(self.run_calls(calls))



# ==== ask engine (V1, frozen ready/lorocV1_ask-a1728300.py; seats renamed ASK_MODEL / ASK_BACKUP_MODEL) ====
ASK_MODEL = "google/gemini-3.8-flash"
ASK_BACKUP_MODEL = "anthropic/claude-sonnet-5.5"
SEAT_PARAMS: dict[str, dict[str, Any]] = {
    ASK_MODEL: {"reasoning": {"effort": "minimal"}},
    ASK_BACKUP_MODEL: {"reasoning": {"effort": "low"}},
}
# the reading LLM reasons before it answers, so each seat leaves room for both
OUTPUT_TOKENS = {"request": 3000, "hint": 1500, "reply": 1500}
LOOK_LISTINGS = 8
MENU_SIZE = 24
SAMPLE_VALUES = 3
RECORD_LIMIT = 24
STEP_SPARE = 1
SEARCH_STEPS = 4
COMMIT_STEPS = 2
EVENT_STEPS = 3
CHEAPEST_STEPS = 2
LOOK_MINIMUM = 2
LOOKUP_STEPS = 3
FIND_LISTINGS = 3
BATCH_LISTINGS = 12
FIRST_PICKS = 2
LLM_ROUNDS = 2
LLM_WAIT = 4

REQUEST_GUIDE = """# Shopping assistant - this step: read the customer's opening request
- You read one request a customer sent to an online shop and return its plain facts as JSON.
# Fields
- "products": one entry per distinct product the customer wants, in the order named. Shop category names often join two words with "and" or "&"; keep such a name whole as one product, and split only where the request gives each part its own quantity or conditions:
  - "name": a short noun phrase for the product, the way a shop would label it (two to four words).
  - "search": the words to type into the shop's search box to find it: the product noun, plus at most two of the values the customer fixed.
  - "stated": every condition the request fixes for that product, each written "property: value" (for example "material: wool" or "price: at most 1500 cents"). Leave out stock, delivery and how to choose among fits. When the request ties a value to another listing (one a named seller sells), the attributes that single out that listing describe it, not this product: they are never stated conditions.
  - "quantity": how many units of it the customer wants, as an integer (1 when not said).
- "messages": how many messages the customer lets you send, as an integer, or null when the request does not say.
- "cheapest": true when the customer wants the lowest price or lowest total among the fits, else false.
- "budget": the most the whole order may cost, as a number in the money units the request uses (for example 2950 for "2950¢", 150 for "$150"), or null when there is no cap.
- "budget_unit": "cents" when that cap is given in cents or pence, "whole" when it is given in whole currency (dollars, pounds, euros), or null when there is no cap.
- "in_stock": true when the customer wants only items that are in stock and ready to ship, else false.
- "story": the background the customer gives (who uses it, where, when, why, what they will notice) in one short sentence, or "" when there is none.
# Answer
- ONE JSON object and nothing else."""

HINT_GUIDE = """# Shopping assistant - this step: choose which properties to ask the customer about
- The customer keeps back a wish or two about a product and states it only when a question names that exact property.
- For each product (P1, P2 ...) you get the properties seen on the shop's listings for it, with how many listings carry each and some sample values.
# Rules
- Rank up to eight properties per product, the one most likely to hold an unstated wish first.
- A property the request already settles holds no unstated wish: leave it out. Leave out identifiers, titles and prices too.
- Lean on what the background implies without saying it: who uses the item, where or how it is used, how it should look or feel, what went wrong with the old one.
- A property such a detail points at goes in "story" as well as in "ranked". With no such detail, "story" stays empty.
- Use property names exactly as listed; two spellings of one property count once.
- Keys that name one property in different words (a short and a long name, two words for one thing) go together in "same", one list per property, each key written as listed.
# Answer
- ONE JSON object and nothing else: {"P1": {"ranked": ["property", ...], "story": ["property", ...], "same": [["property", "property"], ...]}, "P2": {...}}"""

REPLY_GUIDE = """# Shopping assistant - this step: read the customer's reply
- You get the question that was sent, the customer's reply, and the property names the shop's listings carry for each product (P1, P2 ...).
# Rules
- Write down each wish the reply states. A reply can state none, one or several.
- "product": the label of the product the wish is about.
- "kind": "value" when the reply sets a property to a value; "listing" when the value has to be read off another listing the reply names.
- For "value": "property" is the property the reply names (the listed name that matches it, when one does); "value" is the value exactly as written; "relation" is "is", "not", "at_most" or "at_least". When the value carries its own comparison, that comparison is the relation, whatever words frame the sentence. A comparison word before a value that is no number makes it "is". Figures of speech that only stress how firm the wish is are no value.
- For "listing": "property" as above; "value" is the words that name that other listing; "relation" is "is", or "not" when the reply wants a value different from that listing's; "search" is two or three words to find that listing with the shop's search: the seller or brand name it gives, then the product noun.
- Indifference, leaving it to the shop, a plea to ask again more simply or one point at a time, or anything else without a wish gives an empty list.
- "split" is true when the reply asks for fewer things per question (one question at a time, pick one, too many asks); a reply that only says it did not understand is not that.
# Answer
- ONE JSON object and nothing else: {"wishes": [{"product": "P1", "kind": "value", "property": "", "value": "", "relation": "", "search": ""}], "split": false}"""

RELATION_WORDS = {"is": "", "not": "not ", "at_most": "at most ", "at_least": "at least "}


def words(text: Any) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(text).lower())


def plain(name: Any) -> str:
    return " ".join(words(str(name).replace("_", " ")))


def content_words(name: Any) -> set[str]:
    return {word for word in words(plain(name)) if len(word) > 2}


def decode_object(text: str) -> dict[str, Any] | None:
    found = re.search(r"\{.*\}", text or "", re.S)
    if not found:
        return None
    try:
        value = json.loads(found.group(0))
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def ask_llm(shop: Any, prompt: str, seat: str) -> dict[str, Any] | None:
    for round_number in range(LLM_ROUNDS):
        failed = False
        for model in (ASK_MODEL, ASK_BACKUP_MODEL):
            try:
                text = shop.chat([{"role": "user", "content": prompt}], model=model, temperature=0, max_tokens=OUTPUT_TOKENS[seat], **SEAT_PARAMS[model])
            except Exception:
                failed = True
                continue
            answer = decode_object(text if isinstance(text, str) else "")
            if answer is not None:
                return answer
        if not failed or round_number == LLM_ROUNDS - 1:
            break
        time.sleep(LLM_WAIT)
    return None


def packed_pairs(text: str) -> list[tuple[str, str]]:
    return re.findall(r'([A-Za-z][\w \-]*?)\s*=\s*"([^"]*)"', text)


def leaf_facts(listing: Any, into: dict[str, str] | None = None, depth: int = 0) -> dict[str, str]:
    facts = {} if into is None else into
    if isinstance(listing, dict):
        for key, value in listing.items():
            if isinstance(value, str) and len(packed_pairs(value)) >= 2:
                for name, inner in packed_pairs(value):
                    if inner and len(inner) <= 60:
                        facts.setdefault(name.strip(), inner)
            elif isinstance(value, (dict, list)):
                if depth < 3:
                    leaf_facts(value, facts, depth + 1)
            elif value not in (None, "") and depth > 0 and len(str(value)) <= 60 and len(str(value).split()) <= 5:
                facts.setdefault(str(key), str(value))
        if depth == 0 and not facts:
            for key, value in listing.items():
                if not isinstance(value, (dict, list, bool)) and value not in (None, "") and len(str(value)) <= 40:
                    facts[str(key)] = str(value)
    elif isinstance(listing, list):
        for item in listing:
            if isinstance(item, dict) and len(item) == 2 and {"name", "value"} <= set(item):
                if item["value"] not in (None, "") and len(str(item["value"])) <= 60:
                    facts.setdefault(str(item["name"]), str(item["value"]))
            elif depth < 3:
                leaf_facts(item, facts, depth + 1)
    return facts


def full_text(listing: Any) -> str:
    if isinstance(listing, dict):
        return " ".join(full_text(value) for value in listing.values())
    if isinstance(listing, list):
        return " ".join(full_text(value) for value in listing)
    return "" if listing is None else str(listing)


def row_ids(result: Any) -> list[str]:
    rows = result.get("rows") if isinstance(result, dict) else result
    ids = []
    for row in rows or []:
        ident = (row.get("id") or row.get("listing_id") or row.get("product_id")) if isinstance(row, dict) else row
        if ident and str(ident) not in ids:
            ids.append(str(ident))
    return ids


def menu_order(listings: list[dict[str, str]], settled: set[str]) -> list[tuple[str, int, list[str]]]:
    seen: dict[str, dict[str, int]] = {}
    for facts in listings:
        for key, value in facts.items():
            seen.setdefault(key, {})
            seen[key][value] = seen[key].get(value, 0) + 1
    ranked = []
    for key, values in seen.items():
        name = plain(key)
        if not name or name in settled:
            continue
        carriers = sum(values.values())
        distinct = len(values)
        if all(re.fullmatch(r"[a-z]{0,3}[-_]?[0-9a-f]{8,}", value.lower()) for value in values):
            continue
        if distinct == carriers and sum(len(value.split()) for value in values) >= 4 * distinct:
            continue
        weight = carriers * min(distinct, 4)
        samples = [value for value, _ in sorted(values.items(), key=lambda pair: -pair[1])][:SAMPLE_VALUES]
        ranked.append((weight, key, carriers, samples))
    ranked.sort(key=lambda entry: -entry[0])
    unique: list[tuple[str, int, list[str]]] = []
    names: set[str] = set()
    for _, key, carriers, samples in ranked:
        if plain(key).replace(" ", "") not in names:
            names.add(plain(key).replace(" ", ""))
            unique.append((key, carriers, samples))
    return unique


def merged_order(first: list[str], story: list[str], spread: list[str]) -> list[str]:
    known = {plain(key): key for key in spread}
    order = [known[plain(key)] for key in first + story if plain(key) in known]
    return list(dict.fromkeys(order + spread))


QUESTION_CHARS = 70


def example_text(samples: list[str], count: int = 2) -> str:
    shown = [value for value in samples if len(value) <= 24][:count]
    return f", such as {' or '.join(shown)}" if shown else ""


def question_text(product: str, key: str, samples: list[str], basket: bool) -> str:
    # short questions draw more answers: examples stay only while the question stays short
    bare = f"What {plain(key)} do you need for the {product}?"
    if basket:
        return bare
    for count in (2, 1):
        worded = f"What {plain(key)} do you need for the {product}{example_text(samples, count)}?"
        if len(worded) <= QUESTION_CHARS:
            return worded
    return bare


class Product:
    def __init__(self, label: str, name: str, search: str, stated: list[str], quantity: int = 1) -> None:
        self.label = label
        self.quantity = quantity
        self.name = name
        self.search = search
        self.stated = stated
        self.listings: dict[str, dict[str, str]] = {}
        self.menu: list[tuple[str, int, list[str]]] = []
        self.queue: list[str] = []
        self.asked: list[str] = []
        self.bare: list[str] = []
        self.same: list[set[str]] = []
        self.told = 0
        self.searches: list[dict[str, Any]] = []
        self.rows: dict[str, dict[str, Any]] = {}
        self.shelf: str | None = None
        self.strength = 0.0

    def called(self, basket: bool) -> str:
        return (self.shelf or self.name) if basket else self.name

    def samples(self, key: str) -> list[str]:
        return next((samples for name, _, samples in self.menu if name == key), [])


class AskEngine:
    def __init__(self, task: dict[str, Any], shop: Any) -> None:
        self.task = task
        self.shop = shop
        self.request = str(task.get("request") or "")
        self.products: list[Product] = []
        self.messages = 0
        self.story = ""
        self.record: list[str] = []
        self.sent = 0
        self.split = False
        self.trace: list[dict[str, Any]] = []
        self.texts: dict[str, str] = {}
        self.facts: dict[str, dict[str, str]] = {}
        self.labels: dict[str, set[str]] = {}
        self.view_steps = 1.0
        self.cheapest = False
        self.budget: float | None = None
        self.in_stock = False
        self.budget_unit: str | None = None
        self.told_lines: list[tuple[Product, str, str, str]] = []
        self.said: list[str] = []
        self.pages: dict[str, dict[str, Any]] = {}
        self.refused: list[dict[str, Any]] = []

    def search(self, product: Product, query: str, page: Any = None) -> Any:
        result = self.shop.search(query, page=page) if page is not None else self.shop.search(query)
        rows = (result.get("rows") if isinstance(result, dict) else result) or []
        count = result.get("count") if isinstance(result, dict) else None
        product.searches.append({"query": query, "count": count, "next": result.get("next") if isinstance(result, dict) else None, "ids": row_ids(result)})
        for row in rows:
            if isinstance(row, dict):
                ident = row.get("id") or row.get("listing_id") or row.get("product_id")
                if ident:
                    product.rows.setdefault(str(ident), dict(row))
        if not rows and count is None:
            self.refused.append({"query": query, "length": len(query)})
        return result

    def open_listing(self, ident: str) -> dict[str, str]:
        if ident not in self.facts:
            before = self.steps_left()
            listing = self.shop.view(ident)
            self.view_steps = max(self.view_steps, before - self.steps_left())
            self.keep_listing(ident, listing)
        return self.facts[ident]

    def open_listings(self, ids: list[str]) -> None:
        fresh = [ident for ident in ids if ident not in self.facts]
        if not fresh:
            return
        if not callable(getattr(self.shop, "view_many", None)):
            for ident in fresh:
                self.open_listing(ident)
            return
        for start in range(0, len(fresh), BATCH_LISTINGS):
            chunk = fresh[start:start + BATCH_LISTINGS]
            before = self.steps_left()
            result = self.shop.view_many(chunk)
            self.view_steps = max(1, before - self.steps_left()) / len(chunk)
            listings = result.items() if isinstance(result, dict) else zip(chunk, result or [])
            for ident, listing in listings:
                self.keep_listing(str(ident), listing)

    def keep_listing(self, ident: str, listing: Any) -> None:
        self.facts[ident] = {key: value for key, value in leaf_facts(listing).items() if value != ident}
        self.texts[ident] = full_text(listing)
        self.labels[ident] = {" ".join(words(value)) for value in (listing.values() if isinstance(listing, dict) else []) if isinstance(value, str) and len(value) <= 40}
        self.pages[ident] = {key: value for key, value in listing.items() if not isinstance(value, (dict, list))} if isinstance(listing, dict) else {}

    def steps_left(self) -> int:
        try:
            return int(self.shop.steps_left)
        except Exception:
            return 40

    def steps_free(self) -> int:
        later = (SEARCH_STEPS + 1) * len(self.products) + COMMIT_STEPS + EVENT_STEPS + (CHEAPEST_STEPS if self.cheapest else 0)
        return self.steps_left() - later

    def read_request(self) -> None:
        reading = ask_llm(self.shop, REQUEST_GUIDE + "\n\n# Request\n" + self.request, "request") or {}
        for spot, item in enumerate(reading.get("products") or []):
            if isinstance(item, dict) and item.get("name"):
                stated = [str(line) for line in item.get("stated") or [] if line]
                amount = item.get("quantity")
                self.products.append(Product(f"P{spot + 1}", str(item["name"]), str(item.get("search") or item["name"]), stated, int(amount) if isinstance(amount, (int, float)) and amount > 0 else 1))
        if not self.products:
            self.products.append(Product("P1", "item", " ".join(words(self.request)[:6]), []))
        allowance = reading.get("messages")
        self.messages = int(allowance) if isinstance(allowance, (int, float)) and allowance > 0 else 2
        self.story = str(reading.get("story") or "")
        self.cheapest = reading.get("cheapest") is True
        self.budget = reading.get("budget") if isinstance(reading.get("budget"), (int, float)) else None
        self.in_stock = reading.get("in_stock") is True
        self.budget_unit = reading.get("budget_unit") if reading.get("budget_unit") in ("cents", "whole") else None
        self.trace.append({"kind": "request", "products": [p.name for p in self.products], "messages": self.messages, "story": bool(self.story)})

    def look(self) -> None:
        for spot, product in enumerate(self.products):
            room = (self.steps_free() - self.messages - STEP_SPARE - LOOKUP_STEPS) // (len(self.products) - spot)
            result = self.search(product, product.search)
            ids = row_ids(result)
            if len(ids) < LOOK_LISTINGS and room > LOOK_LISTINGS * self.view_steps and isinstance(result, dict) and result.get("next") is not None:
                more = self.search(product, product.search, result.get("next"))
                ids += [ident for ident in row_ids(more) if ident not in ids]
            if len(ids) < LOOK_MINIMUM * 2 and plain(product.name) != plain(product.search):
                more = self.search(product, product.name)
                ids += [ident for ident in row_ids(more) if ident not in ids]
            if callable(getattr(self.shop, "view_many", None)):
                self.open_listings(ids[:LOOK_LISTINGS])
                product.listings.update({ident: self.facts[ident] for ident in ids[:LOOK_LISTINGS] if ident in self.facts})
            for ident in ids:
                share = min(LOOK_LISTINGS, max(LOOK_MINIMUM, int((room - 1) // self.view_steps)))
                if len(product.listings) >= share or self.steps_left() - self.view_steps < self.messages + STEP_SPARE:
                    break
                product.listings[ident] = self.open_listing(ident)
            settled = {plain(line.split(":")[0]) for line in product.stated if ":" in line}
            product.menu = menu_order(list(product.listings.values()), settled)[:MENU_SIZE]
            product.shelf = self.shelf_of(product, product.rows)
            self.trace.append({"kind": "look", "product": product.label, "opened": len(product.listings), "menu": [key for key, _, _ in product.menu]})

    def rank(self) -> None:
        blocks = []
        for product in self.products:
            lines = [f"- {key} ({carriers} listings): {', '.join(samples)}" for key, carriers, samples in product.menu]
            stated = "; ".join(product.stated) or "nothing"
            blocks.append(f"## {product.label}: {product.name}\n- already stated: {stated}\n" + "\n".join(lines))
        answer = None
        basket = len(self.products) > 1
        if (basket or self.story) and any(product.menu for product in self.products):
            answer = ask_llm(self.shop, HINT_GUIDE + "\n\n# Request\n" + self.request + "\n\n# Products\n" + "\n\n".join(blocks), "hint")
        for product in self.products:
            pick = (answer or {}).get(product.label) or {}
            story = [str(key) for key in pick.get("story") or []] if isinstance(pick, dict) else []
            ranked = [str(key) for key in pick.get("ranked") or []] if isinstance(pick, dict) else []
            groups = pick.get("same") if isinstance(pick, dict) and isinstance(pick.get("same"), list) else []
            product.same = [{plain(key) for key in group} for group in groups if isinstance(group, list) and len(group) > 1]
            # the ranker's own first picks lead for one product too: on race-5 labels they held the hidden key more often than the story picks
            product.queue = merged_order(ranked[:FIRST_PICKS], story, [key for key, _, _ in product.menu])
            top = next((carriers for key, carriers, _ in product.menu if product.queue and key == product.queue[0]), 0)
            product.strength = (1.0 if story else 0.0) + top / max(1, len(product.listings))
            self.trace.append({"kind": "rank", "product": product.label, "ranked": ranked, "story": story, "queue": list(product.queue)})

    def next_asks(self) -> list[tuple[Product, str]]:
        waiting = [product for product in self.products if product.queue and not product.told] or [product for product in self.products if product.queue]
        if not waiting:
            return []
        waiting.sort(key=lambda product: (product.told > 0, len(product.asked), -product.strength))
        return [(waiting[0], waiting[0].queue[0])]

    def converse(self) -> None:
        while self.sent < self.messages and self.steps_left() > STEP_SPARE and (self.steps_free() > 0 or self.sent < len(self.products)):
            asks = self.next_asks()
            if not asks:
                break
            for product, key in asks:
                product.queue.remove(key)
                product.asked.append(key)
            basket = len(self.products) > 1
            product, key = asks[0]
            if key in product.bare:
                shared = any(plain(key) in {plain(other_key) for other_key, _, _ in other.menu} for other in self.products if other is not product)
                message = f"What {plain(key)} do you need{' for the ' + product.called(basket) if shared else ''}?"
            else:
                message = question_text(product.called(basket), key, product.samples(key), basket)
            reply = self.shop.ask(message)
            self.sent += 1
            told = self.read_reply(message, str(reply or ""), [product for product, _ in asks])
            # a property the shop spells several ways is asked once: its other spellings leave the queue
            product.queue = [other for other in product.queue if not any(plain(key) in group and plain(other) in group for group in product.same)]
            if not told and not self.split:
                # an open answer covers the keys whose words hold the asked key's words or sit inside them
                asked = content_words(key)
                product.queue = [other for other in product.queue if not (asked and content_words(other) and (content_words(other) <= asked or asked <= content_words(other)))]
            shared = any(plain(key) in {plain(other_key) for other_key, _, _ in other.menu} for other in self.products if other is not product)
            bare = f"What {plain(key)} do you need{' for the ' + product.called(basket) if shared else ''}?"
            # a plea to ask more simply earns one bare re-ask, never the same words again
            if not told and self.split and key not in product.bare and bare != message:
                product.bare.append(key)
                product.queue.insert(0, key)
            self.trace.append({"kind": "ask", "message": message, "reply": str(reply or "")[:300], "told": told})

    def read_reply(self, message: str, reply: str, involved: list[Product]) -> list[str]:
        catalogue = "\n".join(f"- {product.label} ({product.name}): {', '.join(key for key, _, _ in product.menu)}" for product in self.products)
        answer = ask_llm(self.shop, REPLY_GUIDE + f"\n\n# Question\n{message}\n\n# Reply\n{reply}\n\n# Listing properties\n{catalogue}", "reply") or {}
        self.split = answer.get("split") is True
        told = []
        for wish in answer.get("wishes") or []:
            if not isinstance(wish, dict) or not wish.get("property") or not wish.get("value"):
                continue
            product = next((item for item in self.products if item.label == wish.get("product")), involved[0])
            prop = str(wish["property"])
            value = str(wish["value"])
            if wish.get("kind") == "listing":
                value = self.value_from_listing(product, prop, value, str(wish.get("search") or "")) or value
                entry = f"{prop}: {RELATION_WORDS.get(str(wish.get('relation')), '')}{value}"
            else:
                entry = f"{prop}: {RELATION_WORDS.get(str(wish.get('relation')), '')}{value}"
            if len(self.products) > 1:
                entry = f"{product.name} - {entry}"
            product.told += 1
            self.told_lines.append((product, prop, entry.split(": ", 1)[-1] if ": " in entry else value, reply))
            words_entry = f"{product.name} - {prop}, in the shopper's words: {reply}" if len(self.products) > 1 else f"{prop}, in the shopper's words: {reply}"
            if words_entry not in self.said:
                self.said.append(words_entry)
            if entry not in self.record:
                self.record.append(entry)
                told.append(entry)
            said = set(words(plain(prop)))
            product.queue = [key for key in product.queue if not (set(words(plain(key))) <= said or said <= set(words(plain(key))))]
        return told

    def value_from_listing(self, product: Product, prop: str, named: str, search: str) -> str | None:
        target = set(words(named))
        if not target:
            return None
        noun = set(words(product.name)) | set(words(prop))
        names = [token for token in words(search) if token in target and token not in noun] or [token for token in words(named) if len(token) > 3 and token not in set(words(self.request)) and token not in noun][:1]
        best = self.closest_listing(target, names)
        if (best is None or not self.owns(best, names)) and self.steps_free() > STEP_SPARE + self.messages - self.sent:
            for query in dict.fromkeys(filter(None, [search.strip(), " ".join(names)])):
                ids = [ident for ident in row_ids(self.search(product, query)) if ident not in self.facts]
                for ident in ids[:FIND_LISTINGS]:
                    if self.steps_left() <= STEP_SPARE:
                        break
                    self.open_listing(ident)
                best = self.closest_listing(target, names)
                if (best is not None and self.owns(best, names)) or self.steps_left() <= STEP_SPARE + 1:
                    break
        if best is None:
            return None
        return next((value for key, value in self.facts[best].items() if plain(key) == plain(prop)), None)

    def owns(self, ident: str, names: list[str]) -> bool:
        return not names or " ".join(names) in self.labels.get(ident, set())

    def closest_listing(self, target: set[str], names: list[str]) -> str | None:
        spread = {token: sum(1 for text in self.texts.values() if token in set(words(text))) for token in target}
        def weight(token: str) -> float:
            return 1.0 / spread[token] if spread[token] else 0.0
        whole = sum(weight(token) for token in target)
        best, rank = None, (False, 0.0)
        for ident, text in self.texts.items():
            have = set(words(text))
            if names and not all(name in have for name in names):
                continue
            mark = (self.owns(ident, names), sum(weight(token) for token in target & have))
            if mark > rank:
                best, rank = ident, mark
        if best is None or (not names and rank[1] < 0.5 * whole):
            return None
        return best

    def store_budget(self) -> float | None:
        if self.budget is None or self.budget_unit is None:
            return None
        if self.budget_unit == "cents":
            return self.budget
        prices = sorted(float(row["price"]) for product in self.products for row in product.rows.values() if isinstance(row.get("price"), (int, float)))
        if not prices:
            return None
        middle = prices[len(prices) // 2]
        return self.budget * 100 if middle > self.budget * 5 else self.budget

    def shelf_of(self, product: Product, listings: dict[str, dict[str, Any]]) -> str | None:
        if product.shelf:
            return product.shelf
        try:
            names = [str(name) for name in (self.shop.categories or [])]
        except Exception:
            names = []
        wanted = set(words(product.name))
        if names and wanted:
            best = max(names, key=lambda name: len(wanted & set(words(name))) / max(1, len(set(words(name)) | wanted)))
            if wanted & set(words(best)):
                return best
        seen: dict[str, int] = {}
        for entry in listings.values():
            label = entry.get("category") or entry.get("department")
            if isinstance(label, str) and label:
                seen[label] = seen.get(label, 0) + 1
        if not seen:
            return None
        label, count = max(seen.items(), key=lambda pair: pair[1])
        return label if count * 2 > sum(seen.values()) else None

    def finish(self) -> dict[str, Any]:
        entries = list(self.record) + self.said
        for product in self.products:
            for line in product.stated:
                entry = f"{product.name} - {line}" if len(self.products) > 1 else line
                if entry not in entries:
                    entries.append(entry)
        lines = []
        for spot, product in enumerate(self.products):
            listings = {}
            for ident in list(product.rows) + [ident for ident in self.facts if ident not in product.rows and any(ident in search["ids"] for search in product.searches)]:
                entry = dict(product.rows.get(ident) or {})
                entry.update(self.pages.get(ident) or {})
                entry.update({"id": ident, "opened": ident in self.facts, "facts": dict(self.facts.get(ident) or {})})
                if ident in self.texts:
                    entry["description"] = self.texts[ident]
                listings[ident] = entry
            requirements = [{"name": line.split(":", 1)[0].strip(), "value": line.split(":", 1)[1].strip(), "source": "request"} for line in product.stated if ":" in line and not plain(line.split(":", 1)[0]).startswith(("price", "quantity", "budget"))]
            requirements += [{"name": prop, "value": value, "source": "hidden_need", "said": reply} for owner, prop, value, reply in self.told_lines if owner is product]
            shelf = self.shelf_of(product, listings)
            spec = {"category": shelf or product.name, "quantity": product.quantity, "budget": self.store_budget(), "in_stock": self.in_stock,
                    "requirements": requirements, "cheapest": self.cheapest, "products_in_order": len(self.products)}
            lines.append({"line": spot, "category": product.name, "shelf": shelf, "record": spec, "searches": product.searches, "listings": listings})
        return {"record": entries[:RECORD_LIMIT], "lines": lines, "refused": self.refused}


def ask_engine(task: dict, shop) -> dict:
    engine = AskEngine(task, shop)
    closed = globals().get("ShopClosed") or RuntimeError
    try:
        engine.read_request()
        engine.look()
        engine.rank()
        engine.converse()
    except closed:
        engine.trace.append({"kind": "closed"})
    except Exception as problem:
        engine.trace.append({"kind": "error", "error": str(problem)[:200]})
    return engine.finish()


# ==== search engine (V2, frozen ready/lorocV2_search-9f65c818.py; shares words() with the ask engine) ====
READER_LLM = "anthropic/claude-haiku-5.5"
BACKUP_LLM = "openai/gpt-6-luna"
MAX_OPTION_SETS = 4
MAX_QUERIES = 14
MAX_PROBES = 6
MAX_PAGES = 6
FULL_PAGES = 9
BATCH = 8
RETRIES = 3
SPARE = 15
# the longest query the world took or refused, kept across the items of one run
SEEN = {"limit": 200}
# a number and the unit text written straight after it, read from store fact values
AMOUNT = re.compile(r"(\d+(?:\.\d+)?)\s*([^\d\s,;/()]*)")

READER_PROMPT = """You prepare a shopper's needs for an online shop's keyword search and for checking listing facts.
The search returns listings whose text holds every query word, so query words must be ones a listing itself would carry.
The shopper's own words find the right listings most often, so keep them; never swap a value for a different one.
For the record below, reply with the product's plain noun as listing titles write it, and with one entry per requirement, in the record's order:
- "query": search phrases for the wanted value, without the noun: each allowed value as the record words it first, then at most one other spelling of it; [] only when the requirement rules values out or sets an upper or lower limit
- When the record allows several values, give each its own query and its own option, all inside that requirement's one entry: exactly one entry per requirement.
- "options": each value that satisfies the requirement as the record words it, plus other spellings of that same value (plural, hyphen, abbreviation); never a neighbouring or similar value
- "excluded": true only when the requirement's value itself says those values are unwanted
- "pointer": true when the value does not state the wanted value but refers to another listing to take it from (a seller's listing to match or differ from); its query and options are then []
- "bound": "at_least" or "at_most" when the value sets a lower or upper limit, "equal" when it names an exact amount, otherwise null
- "value", "unit": the amount and its unit as stated, otherwise null
- "forms": the same amount in each other unit or spelling a listing might use, as [{"value": number, "unit": text}]
Reply with JSON only: {"noun": "...", "needs": [{"query": [...], "options": [...], "excluded": false, "pointer": false, "bound": null, "value": null, "unit": null, "forms": []}]}

Record:
"""




def key_words(key):
    return words(re.sub(r"([a-z])([A-Z])", r"\1 \2", str(key)))


def stem(word):
    return word[:-1] if len(word) > 3 and word.endswith("s") else word


def present(term, bag):
    return any(w == term or w.startswith(term) for w in bag)


def unit_text(unit):
    # units are compared as written: case, dots and a plural "s" aside
    text = str(unit or "").strip().lower().replace(".", "")
    if text[:1] in ('"', "'"):
        return text[:1]
    return text[:-1] if len(text) > 2 and text.endswith("s") else text


def amount(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def row_text(row):
    # every text the row carries, nested tables as "key value"; the row's own id is no listing text
    parts = []
    for key, value in row.items():
        if value == row.get("id") or isinstance(value, bool):
            continue
        if isinstance(value, dict):
            parts += [f"{k} {v}" for k, v in value.items() if not isinstance(v, (dict, list))]
        elif isinstance(value, list):
            parts += [str(v) for v in value if isinstance(v, (str, int, float)) and not isinstance(v, bool)]
        elif isinstance(value, str):
            parts.append(value)
    return " ".join(parts)


def strong(value):
    # a bare number or a single letter matches too much listing text to prove a value
    return all(len(w) > 1 and not w.isdigit() for w in words(value))


def facts_of(row):
    facts = row.get("facts")
    return facts if isinstance(facts, dict) else {}


def json_in(text):
    start, end = str(text or "").find("{"), str(text or "").rfind("}")
    try:
        return json.loads(text[start:end + 1]) if start >= 0 < end else None
    except ValueError:
        return None


class Need:
    def __init__(self, name, value, read=None):
        self.name = str(name or "").lower()
        self.value = str(value or "")
        self.read = isinstance(read, dict)
        read = read if self.read else {}
        self.excluded = read.get("excluded") is True
        # a value that only names another listing holds no words a fitting listing carries: it neither searches nor rules out
        self.pointer = read.get("pointer") is True
        bound = read.get("bound")
        forms = [(amount(read.get("value")), unit_text(read.get("unit")))]
        forms += [(amount(f.get("value")), unit_text(f.get("unit"))) for f in read.get("forms") or [] if isinstance(f, dict)]
        self.forms = [(v, u) for v, u in forms if v is not None]
        self.bound = bound if bound in ("at_least", "at_most", "equal") and self.forms else None
        # the record's own wording stays an option: a reading adds spellings, it never replaces the shopper's words
        own = " ".join(words(value))
        options = [" ".join(words(o)) for o in read.get("options") or [] if words(o)]
        self.options = list(dict.fromkeys([own] + options if own else options))
        query = [" ".join(words(q)) for q in read.get("query") or [] if words(q)]
        self.query = list(dict.fromkeys([own] + query if own else query))
        if self.pointer:
            self.options, self.query, self.bound = [], [], None

    def query_options(self):
        if self.excluded or self.bound in ("at_least", "at_most"):
            return []
        return [q for q in self.query if q]

    def own_key(self, row):
        # the fact whose name holds every longer word of the need's name, with the fewest other words
        name = {w for w in words(self.name) if len(w) > 2}
        keys = [k for k in facts_of(row) if name and name <= set(key_words(k))]
        return min(keys, key=lambda k: len(set(key_words(k)) - name)) if keys else None

    def judge(self, row):
        # 1 shown, 0 unknown, -1 contradicted
        if self.pointer:
            return 0
        if self.bound in ("at_least", "at_most"):
            return self.judge_amount(row)
        bag = {stem(w) for w in words(row_text(row))}
        shown = any(all(present(stem(w), bag) for w in words(option)) for option in self.options)
        if self.excluded:
            return -1 if shown else 0
        if shown:
            return 1
        if self.bound == "equal":
            return self.judge_amount(row)
        # without a reading nothing is ruled out: the words may be the shopper's, not the listing's
        return -1 if self.read and self.own_key(row) else 0

    def judge_amount(self, row):
        key = self.own_key(row)
        found = AMOUNT.search(str(facts_of(row).get(key, "")).lower()) if key else None
        if not found:
            return 0
        have, unit = float(found.group(1)), unit_text(found.group(2))
        wanted = next((v for v, u in self.forms if u == unit), None)
        if wanted is None:
            return 0
        if self.bound == "equal":
            return 1 if abs(have - wanted) <= 0.02 * max(abs(wanted), 1e-9) else -1
        return 1 if (have >= wanted if self.bound == "at_least" else have <= wanted) else -1


def reader_messages(record):
    requirements = [{"name": r.get("name"), "value": r.get("value")} for r in record.get("requirements") or [] if isinstance(r, dict)]
    return [{"role": "user", "content": READER_PROMPT + json.dumps({"category": record.get("category"), "requirements": requirements})}]


def reading_from(reply, record):
    # a reading that does not cover every requirement in order is no reading
    read = json_in(reply)
    needs = read.get("needs") if isinstance(read, dict) else None
    count = len([r for r in record.get("requirements") or [] if isinstance(r, dict)])
    return read if isinstance(needs, list) and len(needs) == count else None


class Engine:
    def __init__(self, task, shop):
        self.record = task.get("record") or {}
        self.shop = shop
        self.category = str(self.record.get("category") or "").strip()
        self.requirements = [r for r in self.record.get("requirements") or [] if isinstance(r, dict)]
        self.needs = [Need(r.get("name"), r.get("value")) for r in self.requirements]
        self.noun_words = words(self.category)
        self.cheapest = bool(self.record.get("cheapest"))
        self.rows = {}
        self.order = []
        self.viewed = set()
        self.credit = {}
        self.counts = {}
        self.sent = set()
        self.and_search = True
        self.sent_text = ""
        self.capped = False
        self.best = (-1, frozenset(), frozenset(), [])
        self.reading = task.get("reading")
        self.seed(task.get("seed"))

    def seed(self, line):
        # what an earlier stage already searched and opened: kept as rows, never asked for again
        if not isinstance(line, dict):
            return
        for search in line.get("searches") or []:
            if isinstance(search, dict) and words(search.get("query")):
                self.sent.add(" ".join(words(search.get("query"))))
        for ident, listing in (line.get("listings") or {}).items():
            if not isinstance(listing, dict):
                continue
            row = {**listing, "id": listing.get("id") or ident}
            self.take({"rows": [row]})
            if listing.get("opened"):
                self.viewed.add(row["id"])

    def adopt(self, read):
        self.needs = [Need(r.get("name"), r.get("value"), n) for r, n in zip(self.requirements, read["needs"])]
        if words(read.get("noun")):
            self.noun_words = words(read.get("noun"))

    def steps(self):
        try:
            return int(self.shop.steps_left)
        except Exception:
            return 0

    def call(self, tool, *args, **kwargs):
        # a busy store answers 429 now and then; wait and ask again, but never past the item's steps
        for attempt in range(RETRIES):
            try:
                return getattr(self.shop, tool)(*args, **kwargs)
            except TypeError:
                raise
            except Exception as error:
                if type(error).__name__ == "ShopClosed" or attempt == RETRIES - 1:
                    raise
                time.sleep(2 * (attempt + 1))

    def read_needs(self):
        if not self.requirements:
            return
        given = reading_from(json.dumps(self.reading), self.record) if self.reading else None
        if given:
            self.adopt(given)
            return
        messages = reader_messages(self.record)
        for llm in (READER_LLM, READER_LLM, BACKUP_LLM):
            try:
                reply = self.shop.chat(messages, model=llm, max_tokens=900, temperature=0, reasoning={"enabled": False})
            except Exception as error:
                if type(error).__name__ == "ShopClosed":
                    raise
                continue
            read = reading_from(reply, self.record)
            if read:
                self.reading = read
                self.adopt(read)
                return

    def noun(self):
        return " ".join(self.noun_words)

    def take(self, found, covers=()):
        if not isinstance(found, dict):
            return []
        got = []
        for row in found.get("rows") or []:
            ident = row.get("id")
            if not ident:
                continue
            if ident not in self.rows:
                self.order.append(ident)
            merged = dict(self.rows.get(ident) or {})
            merged.update({k: v for k, v in row.items() if v is not None})
            self.rows[ident] = merged
            self.credit.setdefault(ident, set()).update(covers)
            got.append(ident)
        return got

    def note_count(self, covers, count):
        # an all-words search never finds more by adding words; if it does, query words are no proof
        self.counts[covers] = count
        for other, seen in self.counts.items():
            if (other < covers and seen < count) or (covers < other and count < seen):
                self.and_search = False

    def plan(self):
        noun = self.noun()
        wanted = [i for i, need in enumerate(self.needs) if need.query_options()]
        plans = []
        for size in range(len(wanted), 0, -1):
            for subset in itertools.combinations(wanted, size):
                choices = itertools.product(*[self.needs[i].query_options() for i in subset])
                for values in itertools.islice(choices, MAX_OPTION_SETS):
                    credit = frozenset(i for i, value in zip(subset, values) if strong(value))
                    plans.append((frozenset(subset), credit, " ".join(list(values) + [noun])))
        return plans[:MAX_QUERIES - 1] + [(frozenset(), frozenset(), noun)]

    def shorten(self, terms):
        # the noun's leading words go first (the shelf check still holds them), then trailing words
        terms = list(terms)
        while len(" ".join(terms)) > SEEN["limit"] and len(terms) > 1:
            cut = next((w for w in self.noun_words[:-1] if w in terms), None)
            terms.remove(cut) if cut else terms.pop()
        return terms

    def ask(self, text):
        # an answer without rows and without a count is a refusal, not an empty result
        try:
            found = self.call("search", text)
        except Exception as error:
            if type(error).__name__ == "ShopClosed":
                raise
            return {}
        if not isinstance(found, dict) or (not found.get("rows") and not isinstance(found.get("count"), int)):
            return None
        return found

    def search(self, covers, credit, query, reserve, probe=False):
        terms = self.shorten(dict.fromkeys(words(query)))
        if not terms or self.steps() <= reserve or " ".join(terms) in self.sent:
            return None
        self.sent.add(" ".join(terms))
        found = self.ask(" ".join(terms))
        if found is None:
            # the world refused the query (its length, mostly): remember the limit and ask again shorter
            SEEN["limit"] = min(SEEN["limit"], len(" ".join(terms)) - 1)
            terms = self.shorten(terms)
            if self.steps() <= reserve or len(" ".join(terms)) > SEEN["limit"]:
                return None
            self.sent.add(" ".join(terms))
            found = self.ask(" ".join(terms))
            if found is None:
                return None
        if not found:
            return None
        self.sent_text = " ".join(terms)
        got = self.take(found, credit)
        count = found.get("count") if isinstance(found, dict) else None
        if isinstance(count, int) and count > len(found.get("rows") or []) and not found.get("next"):
            self.capped = True
        if probe:
            return found if got else None
        if got and len(credit) > self.best[0]:
            noun = set(self.noun_words)
            self.best = (len(credit), credit, covers, [w for w in terms if w not in noun])
        self.note_count(covers, count if isinstance(count, int) else len(got))
        return found if got else None

    def page_on(self, found, query, credit, reserve):
        # a loose search can match hundreds of rows: a few pages, then the rest of the share goes elsewhere
        page, pages = found.get("next"), 1
        count, shown = found.get("count"), len(found.get("rows") or []) or 1
        # a small result is read whole; the allowance is in pages, since a store may show only a few rows a page
        most = float("inf") if isinstance(count, int) and -(-count // shown) <= FULL_PAGES else MAX_PAGES
        while page and self.steps() > reserve and pages < most:
            pages += 1
            found = self.call("search", query, page=page)
            self.take(found, credit)
            page = found.get("next") if isinstance(found, dict) else None

    def probe(self, reserve):
        # a world that shows one page only: open other windows of the shelf with words its titles use
        if not self.capped:
            return
        _, credit, covers, values = self.best
        used = {stem(w) for w in values + self.noun_words}
        counts = collections.Counter(stem(w) for i in self.rows if self.on_shelf(i)
                                     for w in set(words(self.rows[i].get("title")))
                                     if len(w) > 3 and not w.isdigit())
        tried = 0
        for word, _ in counts.most_common():
            if tried >= MAX_PROBES or self.steps() <= reserve:
                return
            if word in used:
                continue
            if not self.cheapest and sum(1 for i in self.pool() if self.full(i)) >= 10:
                return
            tried += 1
            self.search(covers, credit, " ".join(values + [word] + self.noun_words[-1:]), reserve, probe=True)

    def browse(self, reserve):
        if not self.category or self.steps() <= reserve:
            return
        budget = self.record.get("budget")
        quantity = self.record.get("quantity") or 1
        cap = budget / quantity if isinstance(budget, (int, float)) and budget > 0 else None
        try:
            found = self.call("browse", self.category, max_price=cap)
        except TypeError:
            found = self.call("browse", self.category)
        self.take(found)
        page = found.get("next") if isinstance(found, dict) else None
        while self.cheapest and page and self.steps() > reserve:
            try:
                found = self.call("browse", self.category, page=page, max_price=cap)
            except TypeError:
                found = self.call("browse", self.category, page=page)
            self.take(found)
            page = found.get("next") if isinstance(found, dict) else None

    def fits_money(self, ident):
        row = self.rows[ident]
        if row.get("in_stock") is False:
            return False
        budget, price = self.record.get("budget"), row.get("price")
        quantity = self.record.get("quantity") or 1
        return not (isinstance(budget, (int, float)) and isinstance(price, (int, float)) and price * quantity > budget)

    def on_shelf(self, ident):
        label = str(self.rows[ident].get("category") or "").lower()
        return not label or not self.category or label == self.category.lower()

    def marks(self, ident):
        row = self.rows[ident]
        credited = self.credit.get(ident, set()) if self.and_search else set()
        out = []
        for i, need in enumerate(self.needs):
            mark = need.judge(row)
            if mark == 0 and i in credited:
                mark = 1
            out.append(mark)
        return out

    def score(self, ident):
        marks = self.marks(ident)
        price = self.rows[ident].get("price")
        price = price if isinstance(price, (int, float)) else float("inf")
        return (marks.count(-1), -marks.count(1), price if self.cheapest else 0, marks.count(0), self.order.index(ident))

    def pool(self):
        return sorted((i for i in self.rows if self.fits_money(i) and self.on_shelf(i)), key=self.score)

    def full(self, ident):
        return all(m == 1 for m in self.marks(ident))

    def absorb(self, ident, listing):
        self.viewed.add(ident)
        if isinstance(listing, dict):
            self.rows[ident] = {**self.rows[ident], **{k: v for k, v in listing.items() if v is not None}}
            # a listing that lacks a value its search words claimed: this world's search words are no proof
            if any(self.needs[i].judge(self.rows[ident]) == -1 for i in self.credit.get(ident, ())):
                self.and_search = False

    def view_leaders(self):
        many = callable(getattr(self.shop, "view_many", None))
        while self.steps() > 0:
            fresh = [i for i in self.pool()[:10] if i not in self.viewed]
            if not fresh:
                return
            if many:
                chosen = sorted(fresh, key=lambda i: 0 in self.marks(i), reverse=True)[:BATCH]
                listings = self.call("view_many", chosen)
                if isinstance(listings, list):
                    listings = {row.get("id"): row for row in listings if isinstance(row, dict)}
                for ident in chosen:
                    self.absorb(ident, (listings or {}).get(ident))
                continue
            ident = next((i for i in fresh if 0 in self.marks(i)), fresh[0])
            self.absorb(ident, self.call("view", ident))

    def run(self):
        start = self.steps()
        # with steps to spare, the plain noun goes first, so the item holds rows even if the reading below fails
        if start >= SPARE and not self.reading:
            self.search(frozenset(), frozenset(), self.noun(), 0)
        self.read_needs()
        numeric = any(need.bound in ("at_least", "at_most") for need in self.needs)
        reserve = start // 3 if numeric else min(2, start // 5)
        done = set()
        every = frozenset(i for i, need in enumerate(self.needs) if need.query_options())
        for covers, credit, query in self.plan():
            if any(covers < d for d in done) and len(self.pool()) >= 10:
                continue
            found = self.search(covers, credit, query, reserve)
            if found is None:
                continue
            if self.cheapest or (isinstance(found.get("count"), int) and found["count"] <= 3 * len(found.get("rows") or [])):
                self.page_on(found, self.sent_text, credit, reserve)
            if covers:
                done.add(covers)
            if not self.cheapest and sum(1 for i in self.pool() if self.full(i)) >= 10:
                break
        self.probe(reserve)
        # a cheapest pick needs the cheap end of the shelf unless a search already held every wanted value
        if len(self.pool()) < 10 or (self.cheapest and not (every and every in done)):
            self.browse(reserve)
        self.view_leaders()
        return self.pool()[:10]


def search_engine(task, shop):
    engine = Engine(task, shop)
    try:
        candidates = engine.run()
    except Exception:
        # ShopClosed past the item's steps, or a store tool that fails: keep what the item found
        candidates = engine.pool()[:10]
    return {"candidates": candidates}


# ==== judge engine (V3, frozen ready/lorocV3_judge-6824264b.py) ====
# V3 judge engine (race 6), frozen from race6/start/lorocV3_touchStone.py; the agent carries this same section between its markers.
# Interface:
#   ask = chain_ask(post)          post(body) -> (chat completion dict or None, HTTP status or None); gpt-oss-120b -> deepseek-v4.1-flash -> Haiku,
#                                  one retry on 429/5xx, reasoning effort low for gpt-oss
#   listing = listing_from_record(item_id, record_node, price, in_stock, category)   -> {id, title, facts, price, in_stock, category, note?, description?}
#   needs = needs_from_ledger(["property: constraint", ...], product, products, skip)  -> [{name, value}] for that product
#   verdicts = judge_listings(ask, needs, listings, quantity, budget, in_stock_only, context)
#              -> {id: {fits: True/False/None, binding: bool, why: str, passed: int}}; binding = failure quoted from the listing's own facts
#   needs = add_bounds(ask, needs)  one call; amount needs gain bound/amount/unit/forms so judge_listings can settle plain "<number> <unit>" facts in code
#   choose_listing(verdicts, listings, quantity, cheapest) -> id or None;  pick_one(ask, needs, listings) -> id or None
#   basket_check(lines, products, budget, cheaper) -> [fix notes]; lines {id, product, quantity, price, verified, binding, in_stock_problem},
#              products {name, noun, quantity}, cheaper {product: {fit, unopened}}; an empty list means the basket may be ordered


# ---- judge core: the same text runs alone on the arena judge bench ----
JUDGE_CHAIN = ["openai/gpt-oss-120b", "deepseek/deepseek-v4.1-flash", "anthropic/claude-haiku-5.5"]
CHECK_BATCH = 8

CHECK_PROMPT = """You check shop listings against what a shopper needs. JSON only.
Return {"verdicts": {"<listing id>": {"fits": true or false, "failed_need": number of the first need it fails or null, "quote": the listing's own fact text showing the failure, copied exactly, or ""}}} with every listing id given.
A listing fits only if every need is stated by one of its own facts: any key that names the property counts (spelling, spaces, dots, underscores,
prefixes), and when several keys describe the property, one passing is enough. The fact's value must hold the whole required value: spelling variants,
singular/plural and a longer value that contains every required word all pass; a value holding only part of a combined value, a different word of
similar meaning, or a neighbouring class fails. A need may read "at least N unit", "at most N unit", "X or Y" (any one) or "not X"; convert units before
comparing, and a limit's edge passes. Title and description never stand in for a missing fact: a need no fact states fails, with an empty quote.
A note saying a property is taken from another listing sends you to that listing's record under OTHER LISTINGS when one is given."""

PICK_ONE_PROMPT = """One of these shop listings is right for the shopper's needs; the others each miss something. JSON only: {"choice": "<listing id>"}.
Judge on the listings' own properties ("facts"); values match by sense; convert units; a limit's edge passes."""


def squash(text: Any) -> str:
    return "".join(re.findall(r"[a-z0-9]+", str(text).lower()))


def same_key(a: Any, b: Any) -> bool:
    a, b = squash(a), squash(b)
    return a == b or (len(a) > 3 and len(b) > 3 and (a in b or b in a))


def plain_match(have: Any, want: Any) -> bool:
    # code accepts only a plain match; anything looser goes to the LLM
    return squash(have) == squash(want) or any(squash(part) == squash(want) for part in re.split(r"[,/;|]", str(have)) if part.strip())


BOUND_PROMPT = """Some shopper needs below set an amount. For each such need give its limit, JSON only:
{"needs": [{"n": need number, "bound": "at_least" | "at_most" | "equal", "value": number, "unit": the unit as written or "", "forms": [{"value": number, "unit": text}]}]}
"forms" gives the same amount in each other unit or spelling a listing might print. Leave out every need that sets no amount."""
PLAIN_AMOUNT = re.compile(r"\s*(-?\d+(?:\.\d+)?)\s*([^\d\s,;/()]*)\s*")


def add_bounds(ask: Callable[[str, str], Any], needs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # one small reading turns stated amounts into limits code can compare; units come from the reading, never from a table
    if not any(re.search(r"\d", str(n.get("value"))) and not n.get("bound") for n in needs):
        return needs
    reply = ask(BOUND_PROMPT, need_lines(needs)) or {}
    out = [dict(n) for n in needs]
    for entry in reply.get("needs") or []:
        number = entry.get("n") if isinstance(entry, dict) else None
        if isinstance(number, int) and 1 <= number <= len(out) and not out[number - 1].get("bound"):
            out[number - 1].update(bound=entry.get("bound"), amount=entry.get("value"), unit=entry.get("unit"), forms=entry.get("forms") or [])
    return out


def number_cell(listing: dict[str, Any], need: dict[str, Any]) -> tuple[str | None, str]:
    # code decides only a plain "<number> <unit>" fact whose unit text matches one of the need's forms; anything else stays with the LLM
    bound = need.get("bound")
    forms = [(need.get("amount"), need.get("unit"))] + [(f.get("value"), f.get("unit")) for f in need.get("forms") or [] if isinstance(f, dict)]
    forms = [(float(v), squash(u)) for v, u in forms if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if bound not in ("at_least", "at_most", "equal") or not forms:
        return None, ""
    verdict, quote = None, ""
    for key, value in (listing.get("facts") or {}).items():
        found = PLAIN_AMOUNT.fullmatch(str(value)) if same_key(key, need.get("name")) else None
        if not found:
            continue
        have, unit = float(found.group(1)), squash(found.group(2))
        for want, want_unit in forms:
            if want_unit != unit:
                continue
            ok = have >= want if bound == "at_least" else have <= want if bound == "at_most" else abs(have - want) < 1e-9
            if ok:
                return "pass", str(value)
            verdict, quote = "fail", str(value)
    return verdict, quote


def code_cells(listing: dict[str, Any], needs: list[dict[str, Any]]) -> list[bool]:
    facts = listing.get("facts") or {}
    return [any(same_key(key, need.get("name")) and plain_match(value, need.get("value")) for key, value in facts.items()) for need in needs]


def listing_lines(listing: dict[str, Any]) -> dict[str, Any]:
    shown = {"id": listing.get("id"), "title": listing.get("title"), "facts": listing.get("facts")}
    if listing.get("note"):
        shown["note"] = listing["note"]
    if not listing.get("facts"):
        shown["description"] = str(listing.get("description") or "")[:600]
    return shown


def quoted_in(listing: dict[str, Any], quote: str) -> bool:
    wanted = squash(quote)
    facts = listing.get("facts") or {}
    return len(wanted) >= 2 and any(wanted in squash(value) or wanted in squash(f"{key} {value}") for key, value in facts.items())


def need_lines(needs: list[dict[str, Any]]) -> str:
    return "\n".join(f"{n}. {need.get('name')}: {need.get('value')}" for n, need in enumerate(needs, 1))


def listing_cost(listing: dict[str, Any], quantity: int) -> float:
    price = listing.get("price")
    return price * quantity if isinstance(price, (int, float)) and not isinstance(price, bool) else float("inf")


def judge_listings(ask: Callable[[str, str], Any], needs: list[dict[str, Any]], listings: list[dict[str, Any]], quantity: int = 1,
                   budget: Any = None, in_stock_only: bool = False, context: str = "") -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    unsure = []
    for listing in listings:
        cost = listing_cost(listing, quantity)
        if in_stock_only and listing.get("in_stock") is False:
            out[listing["id"]] = {"fits": False, "binding": True, "why": "it is out of stock", "passed": 0}
            continue
        if isinstance(budget, (int, float)) and cost != float("inf") and cost > budget:
            out[listing["id"]] = {"fits": False, "binding": True, "why": f"price x quantity {cost:g} is over the budget {budget:g}", "passed": 0}
            continue
        cells = code_cells(listing, needs)
        numbers = [number_cell(listing, need) for need in needs]
        cells = [c or numbers[i][0] == "pass" for i, c in enumerate(cells)]
        failed = next((i for i, c in enumerate(cells) if not c and numbers[i][0] == "fail"), None)
        if failed is not None:
            need = needs[failed]
            out[listing["id"]] = {"fits": False, "binding": True, "passed": sum(cells),
                                  "why": f"{need.get('name')}: {need.get('value')} (its own fact reads \"{numbers[failed][1]}\")"}
            continue
        out[listing["id"]] = {"fits": True if all(cells) else None, "binding": False, "why": "", "passed": sum(cells)}
        if not all(cells):
            unsure.append(listing)
    for start in range(0, len(unsure), CHECK_BATCH):
        chunk = unsure[start:start + CHECK_BATCH]
        user = ("NEEDS:\n" + need_lines(needs) + ("\n\nOTHER LISTINGS:\n" + context if context else "")
                + "\n\nLISTINGS:\n" + json.dumps([listing_lines(c) for c in chunk]))
        reply = ask(CHECK_PROMPT, user) or {}
        verdicts = reply.get("verdicts") if isinstance(reply.get("verdicts"), dict) else reply.get("fits") if isinstance(reply.get("fits"), dict) else {}
        for listing in chunk:
            verdict = verdicts.get(listing["id"])
            verdict = {"fits": verdict} if isinstance(verdict, bool) else verdict
            if not (isinstance(verdict, dict) and isinstance(verdict.get("fits"), bool)):
                continue
            entry = out[listing["id"]]
            if verdict["fits"]:
                entry["fits"] = True
                continue
            number = verdict.get("failed_need")
            need = needs[number - 1] if isinstance(number, int) and 1 <= number <= len(needs) else None
            quote = str(verdict.get("quote") or "").strip()
            # a failure the listing's own fact shows is binding; one without such a quote stays advisory
            entry.update(fits=False, binding=bool(quote) and quoted_in(listing, quote),
                         why=(f"{need.get('name')}: {need.get('value')}" if need else "a need")
                         + (f" (its own fact reads \"{quote}\")" if quote else " (no fact states it)"))
    return out


def choose_listing(verdicts: dict[str, dict[str, Any]], listings: list[dict[str, Any]], quantity: int, cheapest: bool) -> str | None:
    fits = [c for c in listings if (verdicts.get(c["id"]) or {}).get("fits") is True]
    if not fits:
        return None
    # the cheapest fit when the request asks for it; otherwise the surest fit, then the cheaper one
    if cheapest:
        return min(fits, key=lambda c: listing_cost(c, quantity))["id"]
    return min(fits, key=lambda c: (-(verdicts[c["id"]].get("passed") or 0), listing_cost(c, quantity)))["id"]


def pick_one(ask: Callable[[str, str], Any], needs: list[dict[str, Any]], listings: list[dict[str, Any]]) -> str | None:
    user = "NEEDS:\n" + need_lines(needs) + "\n\nLISTINGS:\n" + json.dumps([listing_lines(c) for c in listings])
    choice = (ask(PICK_ONE_PROMPT, user) or {}).get("choice")
    return choice if choice in {c["id"] for c in listings} else None
FACT_KEY = re.compile(r"fact|propert|trait|spec|attribut|feature|detail|characteristic", re.I)
NAME_KEY = re.compile(r"title|name|heading|caption|headline", re.I)
NOTE_KEY = re.compile(r"note|know|remark|caveat", re.I)
BUSY = (429, 500, 502, 503, 504)


def first_object(text: Any) -> dict[str, Any] | None:
    text = str(text or "")
    start = text.find("{")
    while start >= 0:
        try:
            found = json.JSONDecoder().raw_decode(text[start:])[0]
            if isinstance(found, dict):
                return found
        except ValueError:
            pass
        start = text.find("{", start + 1)
    return None


def chain_ask(post: Callable[[dict[str, Any]], tuple[Any, Any]], chain: list[str] | None = None, max_tokens: int = 4000) -> Callable[[str, str], Any]:
    # post(body) -> (chat completion dict or None, HTTP status or None); a busy provider gets one retry, then the next LLM
    def ask(system: str, user: str) -> Any:
        for model in chain or JUDGE_CHAIN:
            for attempt in range(2):
                body = {"model": model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                        "temperature": 0, "max_tokens": max_tokens}
                if "gpt-oss" in model:
                    body["reasoning"] = {"effort": "low"}
                data, status = post(body)
                message = ((data.get("choices") or [{}])[0].get("message") or {}) if isinstance(data, dict) else {}
                found = first_object(message.get("content"))
                if found is not None:
                    return found
                if attempt or status not in BUSY:
                    break
                time.sleep(1.5)
        return None
    return ask


def listing_from_record(item: str, node: dict[str, Any], price: Any = None, in_stock: Any = None, category: Any = None) -> dict[str, Any]:
    tables = [v for k, v in node.items() if isinstance(v, dict) and FACT_KEY.search(str(k))]
    tables = tables or [v for v in node.values() if isinstance(v, dict) and sum(1 for x in v.values() if isinstance(x, str) and 0 < len(x) <= 70) >= 3]
    best = max(tables, key=len) if tables else {}
    facts = {str(k): str(v) for k, v in best.items() if isinstance(v, (str, int, float)) and not isinstance(v, bool)}
    title = next((v for k, v in node.items() if isinstance(v, str) and NAME_KEY.search(str(k))), "")
    note = " ".join(v for k, v in node.items() if isinstance(v, str) and NOTE_KEY.search(str(k)) and v.strip())
    listing = {"id": item, "title": title, "facts": facts, "price": price, "in_stock": in_stock, "category": category}
    if note:
        listing["note"] = note
    if not facts:
        listing["description"] = max((v for v in node.values() if isinstance(v, str)), key=len, default="")
    return listing


def needs_from_ledger(entries: list[str], product: str = "", products: list[str] | None = None, skip: list[str] | None = None) -> list[dict[str, str]]:
    # "property: constraint" strings; with several products each opens with its product's name
    products, skip = products or [], [s.lower() for s in skip or [] if s]
    needs = []
    for entry in entries:
        head, sep, value = str(entry).partition(":")
        if not sep or not value.strip() or any(mark in str(entry).lower() for mark in skip):
            continue
        name = head.strip()
        owners = [p for p in products if name.lower().startswith(p.lower())]
        if owners:
            owner = max(owners, key=len)
            if owner != product:
                continue
            name = name[len(owner):].strip() or name
        needs.append({"name": name, "value": value.strip()})
    return needs


def basket_check(lines: list[dict[str, Any]], products: list[dict[str, Any]], budget: Any = None,
                 cheaper: dict[str, dict[str, Any]] | None = None) -> list[str]:
    # lines: {id, product, quantity, price, verified, binding, in_stock_problem}; products: {name, noun, quantity}
    problems: list[str] = []
    total, priced = 0.0, True
    cheaper = cheaper or {}
    for product in products:
        label = product.get("noun") or product.get("name") or "item"
        mine = [line for line in lines if line.get("product") == product.get("name")]
        if not mine:
            problems.append(f"no cart line yet for the {label}")
            continue
        if len(mine) > 1:
            problems.append(f"{len(mine)} lines for the {label} ({', '.join(str(m['id']) for m in mine)}); keep only one")
        line = mine[-1]
        want, have = product.get("quantity") or 1, line.get("quantity")
        if isinstance(have, (int, float)) and have != want:
            problems.append(f"{line['id']} is set to {have} but the {label} is wanted {want} times")
        price = line.get("price")
        if isinstance(price, (int, float)) and not isinstance(price, bool):
            total += price * (have if isinstance(have, (int, float)) else want)
        else:
            priced = False
        if line.get("binding"):
            problems.append(str(line["binding"]))
            continue
        if line.get("verified") is False:
            problems.append(f"{line['id']} is not confirmed to meet every need of the {label}; check its own facts or choose a verified fit")
        if line.get("in_stock_problem"):
            problems.append(f"{line['id']} shows as not in stock; re-check it or choose an in-stock fit")
        swap = cheaper.get(product.get("name")) or {}
        if swap.get("fit"):
            problems.append(f"for the {label}, {swap['fit']} also meets every need and costs less than {line['id']}: swap to it")
        elif swap.get("unopened"):
            problems.append(f"for the {label}, cheaper in-stock rows {', '.join(swap['unopened'])} were never opened: open them first")
    if len(products) > 1:
        names = {p.get("name") for p in products}
        problems += [f"{line['id']} matches none of the wanted products; take it out of the cart" for line in lines if line.get("product") not in names]
    if isinstance(budget, (int, float)) and not isinstance(budget, bool) and priced and lines and total > budget:
        problems.append(f"the lines add up to {total:g} against the budget {budget:g}: search for cheaper fits product by product (a lower"
                        " total is the fix, never a decline)")
    return problems
# ---- end judge core ----


def assemble(problem_data) -> Driver:
    clock = Clock()
    environment = problem_data["environment"]
    binding = dict(environment["binding"])
    policy = environment["policy_view"]
    trace = []
    world = WorldDesk(binding, policy, clock, trace)
    llm = LlmDesk(clock, trace)
    ledger = Ledger(policy)
    opening = policy.get("opening")
    return Driver(clock, llm, world, ledger, trace, opening,
                  SetupAdvisor(llm, world, ledger, clock, trace, opening),
                  InterpretAdvisor(llm, world, ledger, clock),
                  PlanAdvisor(llm, world, ledger, clock),
                  VerifyAdvisor(llm, world, ledger, clock, trace),
                  AuditAdvisor(llm, world, ledger, clock, trace),
                  Finisher(llm, world, ledger, clock, trace))


def agent_main(problem_data) -> list[dict[str, Any]]:
    driver = assemble(problem_data)
    try:
        return driver.run()
    except Exception as exc:
        driver.trace.append({"kind": "crash", "error": f"{type(exc).__name__}: {exc}"[:500]})
        try:
            if not driver.world.over():
                driver.finisher.force()
        except Exception as again:
            driver.trace.append({"kind": "crash", "error": f"{type(again).__name__}: {again}"[:500]})
        return driver.trace


__all__ = ["agent_main"]
