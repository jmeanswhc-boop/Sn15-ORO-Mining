from __future__ import annotations

import json
import re
import time
from typing import Any

from src.agent.proxy_client import ProxyClient

MAIN_MODEL = "anthropic/claude-haiku-5.5"
BACKUP_MODEL = "openai/gpt-6-luna"
SETUP_MODEL = "anthropic/claude-sonnet-5.5"
JUDGE_MODEL = "anthropic/claude-sonnet-5.5"
READ_MODEL = "anthropic/claude-haiku-5.5"

TASK_SECONDS = 265.0
COST_SOFT = 0.06            # spend per episode beyond which checks move to the cheaper LLM
COST_STOP = 0.08            # spend per episode at which the episode is wound up
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
- "requirements": strings, one per firm condition the request puts on the product, each shaped "property: constraint" and carrying the shopper's exact value, limit, exclusion or closed list. If several different products are wanted, prefix each with the short name of its product, as "<product> <property>: constraint". Do not include price, stock, quantity, budgets or mere wishes here, and do not restate the product kind or category as a requirement: words that are part of the category's name (a season, an occasion or an audience in the category name) are covered by the category and are not a separate property requirement.
- "price_cap": the top total price as a number, or null if none is given.
- "currency": the unit of price the request uses, or null.
- "in_stock_only": true if the request accepts only goods actually available or in stock.
- "priority": "cheapest" only when the request explicitly asks for the cheapest or lowest-priced fitting option; "best" only when it explicitly asks for the best by some measure; "none" in every other case. Comments that funds are short or that each penny matters do not ask for the cheapest; they merely explain the price cap.
- "priority_detail": with "best", the measure that defines it; otherwise "".
- "message_budget": how many messages the shopper permits, as an integer, or null if unstated.
- "step_budget": how many moves or steps the request permits, as an integer, or null if unstated.
- "rules": each procedural instruction the request gives the assistant (how many listings to inspect, what to verify or report, how to decline, what to avoid, what to include), copied word for word. Statements about what the shopper intends to do personally are story, not rules.
- "soft_wants": what the shopper would like without insisting (worded as ideally, preferably, would be nice, if possible). Never put these in "requirements".
- "ask_first": points only the shopper can settle, to be asked before deciding: say the budget when there is no price limit, which of two wishes counts more when the request admits both may be impossible (in that case also place both wishes in soft_wants rather than requirements), or anything the request says to check with the shopper. An empty list if nothing applies.
- "decline_no_names": true if the request's decline rule forbids offering, naming or suggesting substitutes or alternatives; false if it asks for something to be named; null if it sets no such rule.
- "roles": an object giving, for each of the roles "search", "details", "view", "message", "cart", "cart_view", "order", "decline", the exact tool name from the list, or null when no tool suits. "message" writes to the shopper; "cart" puts things into or takes them out of the cart; "order" buys or checks out; "decline" closes the task with no purchase.
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
- "unclear": true if the shopper says they did not follow or grasp the assistant's message, or wants it explained, simplified, rephrased or repeated; false otherwise.
- "reply": if the shopper puts a question to the assistant or wants something confirmed, a brief plain answer to send back that addresses it and states the change itself (confirming the new value, say); otherwise "". It must be "" whenever "unclear" is true, since asking to have the assistant's own question explained is not a question needing an answer."""

VERIFY_PROMPT = """Role:
You are the final inspector before a buying assistant submits an order. The task earns its point only if the ordered item satisfies each condition given below.

Input:
Below follow "Shopper request:", "Current requirements (stated, revealed and changed):", the item under review with "Records of this item seen in the store (oldest first):", and possibly its cart line, the latest cart view and the labels that recur on this store's listings.

Rules:
1. Base the judgement solely on the records provided. Where records conflict (over price or stock, say), the most recent one prevails.
2. A condition holds only when the item's own property for it shows the value. Keys vary in spelling (spaces, dots, underscores, prefixes) and one property may appear under several keys; whichever key names it counts. Values are compared by meaning: one entry of a comma list suffices, singular/plural and near synonyms count, capitalisation is irrelevant. A near synonym names the same thing in other words; a value that differs in degree or amount is a different value, however close (water resistant is not waterproof, semi-gloss is not gloss, none is not a little). A requested value is a value, not a limit: unless the request words it as a bound (at most, or less, at least, or more), more or less of it does not meet it, even when it seems better. A word that just looks alike, or a wider/different class, does not count.
3. The title or description is usable only if the item has no property key for that condition. When its own property states something else, the item fails even if the title carries the value; if the item's own property for a condition gives another value, that condition fails, whatever the title may suggest. A value stored under some other property does not count, even when its wording carries the wanted value and the item has no key of the requested name: another key never stands in for the property the requirement names. An options or variant field is the exception: it names the variant this listing sells, so where the item has no key of the requested name, a value the options give for it (a colour, say) counts as shown; where the item has that key and the options give a value for that same property that the key does not list, the two disagree and the condition is not shown. Options that pick one of the key's own values, or that speak of something else (a size, a number), leave the key to decide.
4. Numbers are compared after unit conversion; the boundary satisfies a limit; a measured range qualifies only if every part of it meets the limit, while a set of variants on offer (sizes, lengths or capacities the buyer picks from) qualifies once one offered variant meets it. Exclusions and closed lists are binding. A closed list (values named with "and no other", "only", or the like) admits the named values however they are spelled or phrased, and nothing beside them: a neighbouring grade of the same property (a "medium" next to a named "small", say) lies outside the list.
5. The item must be the requested kind of product; if listings carry a category label, its label must be the one fitting the request's wording as a whole, not a narrower or adjacent one.
6. Price times quantity may not exceed the budget (equality is fine). A price field without a unit is in the same units as the budget figure, so compare the two numbers as they stand; never rescale a price because it looks too low or too high. An item the records show as out of stock fails.
7. Assess only the conditions listed below; never introduce one of your own from the story (an age group, a gender, a size, who will use it). A simple wish ("ideally", "would be nice") marked as not required is not a condition. If several different products are wanted, assess the item only against the conditions of its own kind: a condition prefixed with a different product's name does not concern it, and the quantity is the one for its kind.
8. Answer fits=false only for a condition the records plainly miss, or that no record speaks to at all; a missing record of stock or price is never grounds to fail an item. Neither is a condition tied to another listing whose own record is absent here: judge the item's other conditions, and fail it on that one only when the other listing's record is provided and disagrees.

Output:
A single JSON object and no other text: {"item": the item's id, "fits": true or false, "problems": [short strings, each naming a requirement that is not met or not shown]}."""

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
            elif brief:
                # a yes/no judgement needs little hidden reasoning, and that reasoning was most of a judge call's cost
                body["reasoning"] = {"effort": "low"}
            result = _llm.post_verbose("/inference/chat/completions", body)
            message = self.usable(result.data)
            if message is not None:
                return message
            self.trace.append({"kind": "inference_failure", "model": model, "error": str(result.error)[:300] if result.error else "empty"})
        return None

    def ask_json(self, prompt, max_tokens=700, models=None, brief=False, keep=False) -> dict[str, Any] | None:
        chosen = models or [JUDGE_MODEL, MAIN_MODEL]
        if chosen[0] == JUDGE_MODEL and self.clock.cost > COST_SOFT and not keep:
            chosen = [READ_MODEL, JUDGE_MODEL]
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

    def ranked(self, group, limit) -> list[str]:
        counts = self.counts.get(group, {})
        distinct = self.distinct.get(group, {})
        samples = self.samples.get(group, {})
        order = sorted(counts, key=lambda name: (len(distinct.get(name, ())) < 2, -counts[name]))[:limit]
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

    def render(self, several) -> str:
        groups = [g for g in sorted(self.counts, key=lambda g: -self.tables.get(g, 0)) if g and self.tables.get(g, 0) >= 2]
        if len(groups) < 2 or not several:
            self.fold_unlabelled()
            return "\n".join(self.ranked("", 28))
        lines = []
        for group in groups[:4]:
            lines.append(f"Listings with the label \"{group}\" ({self.tables[group]} carry details):")
            lines += self.ranked(group, 14)
        if self.counts.get(""):
            lines.append("Listings that carry no label:")
            lines += self.ranked("", 10)
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

    def note_refusal(self, step) -> None:
        self.refusals += 1
        self.refused_at = step
        self.order_checked = False

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
        # a requirement may point to another listing, so the newest other opened listings go along
        shown, taken = [], set()
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
            shown.append(dump(clip_strings(inner, 400))[:700])
            if len(shown) >= limit:
                break
        return shown[::-1]

    def property_menu(self) -> tuple[str, int]:
        tally = PropertyTally()
        for record in self.records:
            tally.visit(record, 0)
        return tally.render(bool(self.items())), tally.fact_sets

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
        answer = self.llm.complete(conversation, None, [SETUP_MODEL, BACKUP_MODEL], 1500)
        plan = json_object_in((answer or {}).get("content") or "") or {}
        if not isinstance(plan.get("roles"), dict) and self.clock.seconds_left() > 200:
            # one more try: nothing later works well without this reading
            answer = self.llm.complete(conversation, None, [SETUP_MODEL, MAIN_MODEL], 2000)
            plan = json_object_in((answer or {}).get("content") or "") or plan
        ledger.plan = plan
        self.assign_roles(plan)
        self.take_numbers(plan)

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
        result = self.llm.ask_json("\n\n".join(sections), 1500, [READ_MODEL, JUDGE_MODEL])
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
        sections = [VERIFY_PROMPT, "Shopper request:\n" + ledger.query,
                    "Current requirements (stated, revealed and changed):\n" + ledger.requirements(),
                    f"Item to {purpose}: {item}\nRecords of this item seen in the store (oldest first):\n" + "\n".join(records)]
        others = ledger.other_details(item)
        if others:
            sections.append("Other listings opened in the store (oldest first). Use one only where a requirement names or points to that listing;"
                            " a value it states then counts as shown:\n" + "\n".join(others))
        if line:
            sections.append("Cart line for it (if it names an option or variant, judge that option by its own values, price and stock): " + line)
        if ledger.last_cart_view is not None:
            sections.append("Latest cart view:\n" + dump(ledger.last_cart_view)[:2000])
        labels = label_summary(ledger.records)
        if labels:
            sections.append("Labels that recur on this store's listings, with counts (the item's category label must match the request's wording as a whole, not a neighbouring label):\n" + labels)
        # the check that decides an order stays on the stronger LLM whatever has been spent
        result = self.llm.ask_json("\n\n".join(sections), 1200, brief=True, keep=purpose == "order")
        self.trace.append({"kind": "verify", "item": item, "result": result})
        verdict = ""
        if result and result.get("fits") is False:
            problems = [str(p) for p in result.get("problems") or []][:5]
            verdict = (f"A separate check finds that {item} fails at least one requirement: " + "; ".join(problems)
                       + ". Pick another fully verified fit unless you are sure the check is mistaken.")
        elif result and result.get("fits") is True:
            self.passed.add(key)
            if item not in self.passed_ever:
                self.passed_ever.append(item)
        self.verdicts[key] = verdict
        return verdict

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
        result = self.llm.ask_json(self.plan_prompt(count, limit, urgent, products, menu), 2000, [READ_MODEL, JUDGE_MODEL])
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
        result = self.llm.ask_json(prompt, 1200, brief=True)
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
        answer = self.llm.ask_json(prompt, 300, [READ_MODEL, JUDGE_MODEL]) or {}
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
        self.budget_checks = 0
        self.page_seen = {}

    def status(self) -> str:
        world, ledger = self.world, self.ledger
        left = world.steps_left()
        sent = f"{ledger.messages_used} messages sent" + ("" if ledger.message_budget is None else f" of {ledger.message_budget}")
        text = (f"[status] {world.step} of {world.max_steps} steps used, {left} remaining; {sent}; "
                f"roughly {int(max(0, self.clock.seconds_left() - FORCE_SECONDS))} s to go.")
        text += "\n[current requirements] " + ledger.requirements()
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
            order = self.proven_order()
            if order is not None:
                runnable[0] = (call_id, order["name"], order["args"])
                self.notes[call_id] = ("Code step: this decline was sent as an order instead, because every cart line passed the separate check"
                                       " against the current requirements.")
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
            held = self.budget_research() or self.decline_gate()
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

    def rescue(self) -> str:
        # a decline with a checked fit on hand: code puts that fit in the cart instead
        world, ledger = self.world, self.ledger
        if self.rescues >= 2 or world.steps_left() < 3 or self.clock.seconds_left() < WRAP_UP_SECONDS + 30:
            return ""
        self.rescues += 1
        if ledger.items():
            return self.basket_rescue()
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

    def cheapest_note(self) -> str:
        ledger = self.ledger
        if ledger.plan.get("priority") != "cheapest" or ledger.items():
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
        if not total or total > 48 or len(shown) >= total:
            return ""
        seen = self.page_seen.setdefault(name + dump({k: v for k, v in args.items() if not is_number(v)}), set())
        seen.update(row_ident(row) for row in shown)
        if len(seen) >= total:
            return ""
        return (f"This search matched {total} listings and {len(seen)} of them have been shown. Rows do not come in price order, so the"
                " cheapest fit may sit on a page not yet seen: fetch the remaining pages before choosing.")

    def wind_up(self) -> None:
        ledger = self.ledger
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
        if "cart_view" in hits:
            ledger.last_cart_view = observation

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
        return " ".join(n for n in notes if n)

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

    def opening_message(self) -> str:
        user = "Shopper request:\n" + self.ledger.query
        if self.opening is not None:
            user += "\n\nOpening observation:\n" + dump(self.opening)
        brief = self.setup.opening_brief()
        if brief:
            user += "\n\n" + brief
        return user + "\n\n" + self.status()

    def run(self) -> list[dict[str, Any]]:
        world, ledger = self.world, self.ledger
        if world.finished:
            return self.trace
        self.setup.read_request()
        self.finisher.finish_modes()
        self.trace.append({"kind": "plan", "plan": ledger.plan, "roles": world.roles})
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
            if self.clock.cost > COST_STOP:
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
