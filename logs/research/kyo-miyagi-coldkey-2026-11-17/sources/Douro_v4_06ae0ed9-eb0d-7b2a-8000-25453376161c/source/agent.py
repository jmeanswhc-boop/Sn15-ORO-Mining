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
SETUP_MODEL = "anthropic/claude-sonnet-5.5"
JUDGE_MODEL = "anthropic/claude-sonnet-5.5"
READ_MODEL = "anthropic/claude-haiku-5.5"
ALT_MODEL = "google/gemini-3.8-flash"
SETUP_BACKUP = "openai/gpt-oss-120b"
CHEAP_JUDGES = ["openai/gpt-oss-120b", "deepseek/deepseek-v4.1-flash", "anthropic/claude-haiku-5.5"]

TASK_SECONDS = 265.0
WRAP_UP_SECONDS = 70.0
FORCE_SECONDS = 28.0
OBS_CHARS = 16000
HISTORY_CHARS = 35000
COMPACT_BLOCK = 20000
KEEP_RECENT = 6
MAX_TURNS = 80
RETRY_PAUSES = (1.5, 3.0, None)
RUNAWAY_USD = 0.08
SPEND_FINISH_USD = 0.06
SHELF_HOLDS = 2
OPEN_BEFORE_ASK = 5
DECLINE_SEARCHES = 8
ENGINE_JUDGE = True
SEARCH_ENGINE = True
TABLE_SWAP = False

SYSTEM_PROMPT = """Role: you act as a scrupulous purchasing agent for one customer of a simulated shop, and calling the tools you were given is the only thing you can do. Scoring is binary. The point is earned only by a perfect end state: the purchased product honours each condition the customer wrote, each hidden wish they disclose under questioning, and each change they make mid-task; units and spending limit are right; when they want the lowest price (or the top choice by some measure) that is what you picked; and each instruction in their text was obeyed. Should no product in the shop satisfy all of that, the correct ending is a refusal to buy, carried out exactly as their text prescribes.

Input: the customer's text, a draft breakdown of it, a status line, and whatever your tool calls return.

Rules:
1. Any tool call, failed or erroring ones included, burns one step out of a fixed allowance. Reasoning costs nothing; steps are scarce.
2. Independent calls (e.g. opening several offers' detail pages together) may share a reply and run in sequence. A checkout or a decline always goes out on its own.
3. Tools differ per shop in names, arguments and output layout. Read each description and JSON schema closely and use precisely the argument names and allowed values it lists. Let your first outputs teach you which tool exposes which facts (cost, availability, detailed attributes).
4. Find out soon which category labels the shop attaches to the wanted product type (offers display them). Among adjacent labels, the correct one is the label agreeing with the customer's phrasing taken whole; search and buy inside it.
5. Searches may be crippled: a cap on visible rows or pages, ordering by id rather than relevance or cost, all words mandatory, filters valid in only one mode. Hold the stated total up against the rows actually shown and keep narrowing (more precise words, category or price filters, further pages) until nothing relevant remains unseen.
6. The customer has unwritten wishes and discloses one only when a message asks about that attribute. Only the FIRST point of any message gets answered, so each message covers precisely ONE attribute: a brief, simple, warm question to the customer that names it (one or two sample values allowed). No commands, no talk of lists, labels, positions or rules, never two questions in one.
7. If they answer that it doesn't matter, they have no preference or anything goes, that attribute is open. If they name a preference or a limit, it binds from then on, verbatim (numeric limits too).
8. With a planner active, questions get chosen and sent for you once you have viewed details of two or three offers of the wanted type, so do that early. Replies then show up together with their meaning; treat each disclosed wish as binding and search again combining it with the written values.
9. When asking falls to you: view details of two or three offers first, next ask about attributes the story implies (the user of the item, the look it must have, the place or manner of use), then about attributes the offers carry that vary across candidates. Skip anything the text already decides, and stay inside the message allowance.
10. The text can prescribe working methods: how many offers to inspect or compare before deciding, what to verify, what to tell the customer, what a remark must contain, how to decline. They are scored like conditions; carry each one out literally and fully. Looking at or opening N offers means N fully suitable offers, each opened through the details tool.
11. A question from the customer to you (say, confirming a change) gets answered automatically while messages remain; the change binds you regardless.
12. Things merely hoped for ("ideally", "would be nice", "if possible") are not conditions: they only break ties among offers that already pass everything, and never outrank a lowest-price instruction. The story around the request (who uses the item, where, when and why) is background: it never adds a condition of its own, and no offer is ruled out for clashing with it.
13. Where the text leaves open something only the customer knows (for instance there is no price limit, or two wishes may clash and the customer must rank them), ask before deciding.
14. Mid-task changes happen: a fresh spending limit (up or down), a condition removed, swapped or added, a price increase, a product gone. A single customer line can bundle several. Such lines can hide in any tool output (an extra field of sentences) or ride along with a checkout attempt. Each binds from then on: test your choice and reserves against it, and if lowest price or top choice was requested, choose again from the new passing set.
15. Checkout can be REFUSED: the output then contains a "refusal" field, even if the inner output looks fine. Nothing ends and the cart stays as it was. Read the refusal, apply every change in it, take out of the cart whatever no longer passes (e.g. a sold-out product), put in the next verified candidate, recheck availability and cost, and try checkout again.
16. Products can also vanish or get pricier with no warning, hence availability and cost are rechecked just before each checkout; a product whose current cost times units exceeds the current limit has stopped qualifying.
17. Judging an offer decides your score. A condition holds only if the offer's own attribute for it shows the value. Key spelling varies (spaces, dots, underscores, prefixes) and one attribute may appear under several keys; any key naming it qualifies. Values match by sense: a single item of a comma-separated list suffices; singular equals plural; near synonyms qualify; letter case is irrelevant. Shared word beginnings are no match, nor is a wider or other class (girls is not women; plain unisex is not female).
18. Fall back on title or description only if the offer has no key at all for that condition. Should its own attribute say otherwise, it fails even when title or description shows the value. A condition that no offer in the shop states anywhere (not under any key, title or description) cannot be checked here, so on its own it never rules an offer out.
19. Values under some other, related attribute do not qualify (a length field cannot satisfy a style condition; lining material cannot satisfy material).
20. Compare numbers numerically after unit conversion; limits are inclusive; a range qualifies only if all of it respects the limit.
21. Obey exclusions ("leave out anything with X") and closed sets ("A or B and nothing else").
22. The offer's category label decides the product type. Buy only inside the category whose label matches the customer's phrasing taken as a whole (asking for "A or B" points to a category named "A & B", not to a narrower neighbour named "B"); an offer filed under any other category does not count, however close it is. Inside the right category, never demand an attribute that repeats the category name, and never reject an offer because its title or a type field of its own names another kind of item. It must be available when the text wants goods on hand, and cost times units may not exceed the limit (exactly equal is fine).
23. An attribute value listing several options, one of them the required one (a comma list), is a clean match and not conflicting evidence. Evidence conflicts only when the offer's own keys for that attribute give different, incompatible values; only then, and only with no lowest-price or top-choice rule to decide, favour the candidate with clean evidence.
24. Lowest price requested: the purchase must be the cheapest product among ALL products meeting every condition. Seeing one suitable product proves nothing. Use price filters when available, drop the price ceiling until all remaining rows show, sort visible candidates by cost yourself and verify upward from the cheapest; the first full fit, with every cheaper one examined and rejected, wins. Keep the following fits as reserves.
25. Top choice requested by some measure (most, largest, longest, highest and so on): weigh ALL suitable products on that measure likewise and take the extreme. An offer stating no value for the measure cannot win.
26. Maintain a ranked set of verified fits with a reserve, since the favourite may sell out during checkout.
27. Before any checkout: one cart line per wanted product (a single line when one product is wanted), each the chosen offer in its required units, nothing else (remove other lines); after your last cart change you reread its availability and cost, and it is available and within the limit.
28. Decline only when an exhaustive search leaves nothing that meets every condition, following the text's decline rule to the letter. If the text forbids offering substitutes, the decline remark must not name, describe or hint at any offer: no ids, no titles, no prices.
29. Pacing: send the first checkout while seven or more steps remain so a refusal can still be repaired. At 7 remaining steps only cart, stock-check and finishing tools stay usable. Commit once; flipping the cart back and forth wastes a step per change.

Output: a reply always includes one or more tool calls, with brief text: current conditions, verified fits with cost and availability, and your next action."""

SETUP_PROMPT = """Role: you analyse a customer's text and the shop's tool catalogue on behalf of a purchasing agent.

Input: the catalogue appears right after these instructions; the customer's text (and maybe an opening observation) follows it.

Rules:
1. Reply with ONE JSON object on a single line (no indentation), nothing else, using exactly these keys.
2. "product": the product type wanted, phrased as in the text. When the text names the range, category or shelf it wants, the product is that name as written, never a looser word the story uses for it.
3. "quantity": units of one offer to buy (an integer; 1 unless several are asked for). Things normally sold as a single unit, like a pair of shoes or a set, are quantity 1. "quantity_span": when quantity is above 1, the exact stretch of the text that asks for that many units, copied character for character; "" otherwise.
4. "products": when the text asks for more than one kind of product in one order (each named on its own, usually with its own units), a list with one entry per kind; one product type whose name joins two words with "and" or "&" (a single shelf name) is one kind; a listing the text only points at to read a value off it (a reference) is never a product to buy. Entries: {"product": its type as phrased, "quantity": its units, "requirements": its own conditions in the format below, "search": the words to type into the shop's search box to find it (its product noun plus at most two values the text fixes for it)}; a list with a single entry otherwise. "requirements": a list of strings, one per firm written condition on the product (with several products, start each with its product, as "<product> - property: constraint"), each formatted "property: constraint" and keeping the customer's exact value, limit, exclusion or closed list; a wish for anything except a value is an exclusion, written "property: not <value>", never the value itself. Price, stock, quantity, budgets and mere wishes do not belong here, nor does the product type or category, nor anything only the story implies (who will use it, what will be done with it later), nor the attributes that only single out another listing the text points at (those belong under "references"): words forming part of the category name (a season, an occasion or an audience inside it) are covered by the category and are not separate conditions.
5. "price_cap": the total spending ceiling as a number in the request's own currency unit, copied as written and never converted to another unit (a ceiling written as 53 of a unit is 53), or null if absent.
5b. "references": conditions whose value is not given in the text but has to be read off a different listing, which the text identifies by its seller and a few of its attributes. One entry each: {"product": the wanted product it constrains, "property": the attribute whose value must be read, "range": the category of that other listing, "seller": the seller named, "identify": the attributes that single it out, "tolerance": the allowed difference as written, or ""}. An empty list if none. Keep such a condition out of "requirements" until its value is known. The attributes the text gives to single out that other listing belong under "identify" only and are never requirements of the product to buy.
6. "currency": the money unit used in the text, or null. "price_cap_unit": "cents" when the cap is written in cents or pence, "whole" when it is written in whole currency (dollars, pounds, euros), null when there is no cap.
7. "in_stock_only": true if the text accepts only goods actually available or on hand.
8. "priority": "cheapest" only for an explicit demand for the cheapest or lowest-priced suitable option; "best" only for an explicit demand for the best by some measure; "none" otherwise. Notes that cash is short or every penny matters merely justify the cap and do not demand the cheapest.
9. "priority_detail": the measure behind "best"; "" in all other cases.
10. "message_cap": the most messages the customer lets the agent send them, as an integer, or null when the text sets no such limit.
11. "tool_call_cap": the most tool calls, steps, actions or moves the text allows the agent, as an integer, or null when the text sets no such limit.
12. "rules": every procedural instruction for the agent (how many offers to inspect, what to verify or report, how to decline, what to avoid, what to include), quoted verbatim. Statements about what the customer will do personally are story, not rules.
13. "soft_wants": things wanted without strict need (worded ideally, preferably, would be nice, if possible); never put them under "requirements".
14. "ask_first": things only the customer can settle that should be asked before choosing, e.g. the budget when no ceiling is given, which of two wishes wins when the text says both may be impossible (then put both wishes in soft_wants too, not in requirements), or anything the text says to confirm with the customer. Write each entry as the one product property it settles, in a few words (the property's plain name, not the story's own phrasing), or as the budget or the two-wish choice; a general offer to answer questions, or a question about the story itself, is no entry (put a property it points at under "hint_properties"). An empty list if none.
15. Every sentence must be accounted for: a requirement, the priority, a soft want, a rule, a budget, a hint, or mere story. When unsure whether something is a rule, list it under "rules".
16. "decline_no_names": true when the decline rule forbids offering, naming or suggesting substitutes or alternatives; false when it asks for something to be named; null when no decline rule exists.
17. "roles": an object mapping each role to an exact tool name from the catalogue, or null where nothing fits:
  "search", "details", "view", "message", "cart", "cart_view", "order", "decline", "browse".
  "details" = the tool that returns one listing's full attribute sheet (its properties or specifications); "view" = a lighter tool for one listing's price and stock; when only one such tool exists, it is "details".
  "message" = the tool for messaging the customer; "cart" = the tool adding to or removing from the cart; "order" = the tool that purchases or checks out; "decline" = the tool that closes the task without a purchase.
17a. "search_query": the exact name of the search tool's argument that takes free search words (keywords typed by a shopper), or null when the search tool has none.
17b. "browse" = the tool that lists the listings of one shop category (range), when one exists. "browse_args": the exact argument names of that tool, {"category": the argument naming the category, "max_price": the argument setting a price ceiling, "brand": the argument naming a brand or maker, "page": the argument taking a continuation token}, each null when the tool has no such argument; {} when there is no browse tool.
18. "finish_values": for the order tool and the decline tool, each argument that only accepts a fixed list of values, mapped to the allowed value that carries out that tool's own job (buying for "order", ending with no purchase for "decline"), read from the tool's and the argument's descriptions; written {"order": {"<argument>": "<value>"}, "decline": {"<argument>": "<value>"}}, with {} where no such argument exists.
19. "cart_values": the same for the cart tool: each argument with a fixed list of values, mapped to the value that puts an item in and the value that takes it out, written {"add": {"<argument>": "<value>"}, "remove": {"<argument>": "<value>"}}; {} where no such argument exists.
20. "hint_properties": up to three product properties the story is about without naming them, each as a short property name (a part of the item, how it fits or attaches, what sets its look, where it has to go); an empty list when the story points at none. Never repeat a written requirement here.
21. "story": the background the customer gives (who uses it, where, when, why, what they will notice) in one short sentence, or "" when there is none.

Output: that single-line JSON object.
"""

INTERPRET_PROMPT = """Role: you decode a customer's latest words for a purchasing agent.

Input: below come the customer's original text, the conditions so far, earlier exchanges, the agent's question if one was asked, and the new customer words.

Rules:
1. "meaning": one brief sentence on what the customer intends.
2. "needs": newly disclosed firm conditions on the product, each formatted "property: constraint", keeping the exact value, limit, exclusion or closed list given. The property is the one the customer's own words name, in full as they write it: when a line names a property and then gives a value, that named property takes the value, and the linking words between them never add another property; when they answer a question about one attribute by setting another, the condition belongs to the attribute they set, never to the one the question named.
3. "no_preference": attributes the customer declared themselves indifferent about.
4. "dropped": former conditions the customer has given up.
5. "new_budget": the updated total spending ceiling as a number, or null.
6. "unavailable": true if the customer reports that a product can no longer be purchased (sold out, gone, out of stock, button disabled); false otherwise.
7. "requirements_now": the complete current list of firm product conditions after everything said so far, rebuilt from the original text, previous customer lines and this one, each formatted "property: constraint" (with several products wanted, "<product> - property: constraint", naming the product the customer or the question was about). Retain each earlier condition unless dropped or replaced; for a replaced value keep only the latest; append every newly disclosed need. Exclude price, stock, quantity and the product category.
8. "asked_us": true only when the customer's new words put a question to the agent or ask the agent to confirm something; false otherwise. A plea to say it again, simpler, shorter or one point at a time is no question to the agent (asked_us false): it means the customer holds no wish on the attribute just asked, so list that attribute under "no_preference". "reply": when asked_us is true, a brief simple answer to return that addresses it and states the change in substance (e.g. confirming the new value); else "".
9. One line may contain several changes; list them all. A swapped condition counts as dropped (the old one) and as a need (the new one). If the customer says which of two wishes matters more, the winning wish is a need. Any budget the customer gives is new_budget. A mere hope ("ideally", "would be nice") is not a need.
10. Read literally yet generously. Folksy, figurative or roundabout wording naming a value means a firm wish for that value; treat a value as excluded only on a clear instruction to avoid it. A number with a comparison (at most, at least, under, over, no more than) is a limit and keeps its unit; when the value itself carries its comparison, that comparison is the condition, whatever words frame the sentence around it. A comparison on a value that is no number (a word such as a shape, material, type or size name) only makes that value required: write "property: value" without the comparison. A new budget supersedes the previous one, higher or lower, even if announced as good news.
11. When the customer merely says it doesn't matter, they have no preference or anything works, put the attribute that was asked about into no_preference.
12. "released": the attributes the new words themselves name while setting their condition aside (telling the agent to drop, leave off or stop minding that attribute); [] when the words only express indifference, which never gives up a condition the request wrote down.
13. "references": when the new words tie an attribute's value to a different listing (the figure some named seller's offer carries, possibly by way of a further offer that one refers to), one entry each: {"property": the attribute of the wanted product, "relation": how it must compare as written (same as, different from, within a stated margin), "range": the category of that other listing, "seller": the seller named, "identify": the attributes that single that listing out}. The identifying attributes describe the other listing only and are never conditions on the wanted product; such a condition goes here, not under "needs". [] if none.

Output: ONE JSON object holding exactly these keys, nothing else."""

VERIFY_PROMPT = """Role: you are the final gate before a purchasing agent checks out. The point is earned only if the purchased product satisfies every condition listed below.

Input: the customer's text, the current conditions, the product id, the records of it seen in the shop, and possibly the newest cart view plus labels recurring across offers.

Rules:
1. Base the judgement only on the records provided.
2. A condition holds only if the product's own attribute for it shows the value. Key spellings vary (spaces, dots, underscores, prefixes) and one attribute may sit under several keys; any key naming it qualifies. Values match by sense: one item of a comma list suffices, singular/plural and near synonyms qualify, letter case is irrelevant. Lookalike words and wider or other classes do not qualify.
3. Title or description only counts when the product has no key for that condition. If its own attribute says something else, it fails even when the title shows the value.
4. Values under some other, related attribute do not qualify.
5. Numbers: convert units first; limits are inclusive; a range qualifies only if all of it respects the limit.
6. Exclusions and closed sets must be obeyed.
7. It must be the wanted product type. Where offers carry category labels, only the category whose label matches the customer's phrasing taken as a whole qualifies; an offer filed under any other category fails the type check, a narrower or neighbouring one included. An offer in that category passes the type check even when its title, description or a type attribute of its own names a different kind of item.
8. If the product's own attribute for a condition gives another value, that condition fails, whatever the title implies.
9. Cost times units may not exceed the budget (exactly equal is fine). If a record shows it out of stock, it fails. When one product is wanted, this item is to be the cart's only line, so other lines in the cart view are about to be replaced and never count against the budget.
10. A mere hope ("ideally", "would be nice") marked as not required is no condition.
11. With conflicting records (about price or stock, for example), the newest wins.
12. Use fits=false only for a condition the records plainly fail or that no record mentions at all; a missing record of stock or price is never grounds for failing.
13. When the order holds several products, judge the item only against the conditions of the product it is (by its category), and read the spending limit as covering the whole cart, not this item alone.
14. Sort every problem into one of two kinds. "contradicted": the product's own record states something for that condition and it is not the wanted value (another value, the wrong category, over budget, out of stock). "not_shown": nothing in the records speaks to that condition either way.

Output: ONE JSON object, nothing else: {"item": the product id, "fits": true or false, "problems": [{"condition": the condition exactly as listed, "kind": "contradicted" or "not_shown", "note": a few words}]}."""

PLAN_PROMPT = """Role: you decide which questions a purchasing agent puts to the customer. The customer carries hidden wishes; each one is the value of a single product attribute and surfaces only when a message asks about exactly that attribute. Only a message's first point is answered; asking about anything else yields a polite "don't mind".

Input: the customer's text, the conditions so far, the questions already put, items flagged ASK THE CUSTOMER FIRST, and the attributes found in the offers' details.

Rules:
1. Select up to QUESTION_COUNT distinct attributes, strongest candidates first.
2. First come items flagged below as ASK THE CUSTOMER FIRST that have not been asked; ask them plainly. For the budget, ask for their maximum spend. For a choice between two wishes, name both values with their attributes, e.g. "If no single listing offers both, do you prefer the <value> <property> or the <value> <property>?".
3. Then attributes the offers' details below truly contain, favouring (a) an attribute the story implies indirectly (who will own, wear or use it, how it should look, where or how it gets used) and (b) attributes whose values differ most across the offers, measures and sizes included (height, width, size, capacity, pack type), over logistics (warranty, condition, years) and over attributes that every offer shares.
4. Never choose an attribute the text already decides (its written conditions, price, stock, quantity), never repeat one already asked, never spend two questions on near-identical attributes (two kinds of style, two spellings of the same attribute), and skip attributes no offer has.
5. Use attribute names exactly as the offers spell them.
6. An attribute question names the attribute the way the offers name it and nothing more, in at most twelve words, for example "What <attribute> would you like?". Give no sample values and never offer a choice between two values. No instructions, no lists, no statements, no second question.
7. Mark each question "kind": "must_learn" for an item flagged ASK THE CUSTOMER FIRST, "attribute" otherwise.
8. When the order holds several products, give each question its "product" and close it with the product, as in "What <attribute> would you like for the <product>?"; spread the questions over the products whose details were seen.
9. An earlier answer that says the customer did not follow, wants it simpler or shorter, or wants one point at a time is this shop's way of saying the customer holds no wish on that attribute: never ask that attribute again in any wording; move on to another attribute.

Output: ONE JSON object, nothing else: {"questions": [{"property": "...", "kind": "attribute" or "must_learn", "product": "... or empty", "question": "..."}, ...]}."""

AUDIT_PROMPT = """Role: a purchasing agent is ready to tell the customer the shop has nothing suitable; you double-check that.

Input: the customer's text, the current conditions and the offer records seen.

Rules:
1. Compare each record with the conditions.
2. A condition holds when the offer's own attribute for it shows the value (any key spelling, one item of a comma list, singular/plural and near synonyms all qualify; a wider or other class does not).
3. Title or description only counts when the offer has no key for that attribute.
4. Numbers are compared after unit conversion, with inclusive limits.
5. The cost must fit the budget and the product must not appear out of stock.
6. The category label settles the product type: an offer in the wanted category counts as that type whatever its title or its own type attribute calls it.

Output: ONE JSON object, nothing else: {"possible_fits": [ids of offers that seem to meet every condition, strongest first, 5 at most], "reason": "a single sentence"}."""

_llm = ProxyClient(timeout=70, max_retries=1, retry_delay=1)
_env = ProxyClient(timeout=95, max_retries=2, retry_delay=1)

_ID_HINT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-:.]{3,63}$")
_SKU_LIKE = re.compile(r"[A-Za-z0-9_\-]+")
_DECLINE_TEXT = "Sorry, nothing in the store meets all of your requirements, so I am not ordering anything."
ORDER_WORDING = "Order: buy and pay for the items in the cart and check out."
DECLINE_WORDING = "Decline: cancel and end the task without buying anything."
REMOVE_WORDING = "Remove: take the item out of the cart, discard or delete it."
ADD_WORDING = "Add: put the item into the cart with the wanted number of units."


def _trim(value: Any, limit: int, depth: int = 0) -> Any:
    if depth > 8:
        return None
    if isinstance(value, str):
        return value[:limit] + "..." if len(value) > limit else value
    if isinstance(value, dict):
        return {key: _trim(part, limit, depth + 1) for key, part in value.items()}
    if isinstance(value, list):
        return [_trim(part, limit, depth + 1) for part in value]
    return value


def _shrink(value: Any, depth: int = 0) -> Any:
    # older results keep ids, figures, short facts and shopper lines; long prose is cut
    if depth > 8:
        return None
    if isinstance(value, str):
        return value[:100] + "..." if len(value) > 120 else value
    if isinstance(value, dict):
        return {key: _shrink(part, depth + 1) for key, part in value.items()}
    if isinstance(value, list):
        if value and all(isinstance(part, str) for part in value):
            return value
        return [_shrink(part, depth + 1) for part in value]
    return value


def _unpaged(observation: dict[str, Any]) -> tuple[int, int] | None:
    # a result whose own count exceeds the rows it lists, with no non-empty continuation value beside them
    rows = next((part for part in observation.values() if isinstance(part, list) and part and all(isinstance(row, dict) for row in part)), None)
    if not rows:
        return None
    counts = [part for part in observation.values() if isinstance(part, int) and not isinstance(part, bool) and part > len(rows)]
    onward = [part for part in observation.values() if isinstance(part, str) and part.strip() and len(part) <= 80 and " " not in part.strip()]
    if not counts or onward:
        return None
    return max(counts), len(rows)


def _dump(value: Any) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        text = str(value)
    return text if len(text) <= OBS_CHARS else text[:OBS_CHARS] + "...(cut)"


def _decode_object(text: Any) -> dict[str, Any] | None:
    if not isinstance(text, str):
        return None
    body = text.strip()
    if body.startswith("```"):
        body = re.sub(r"^```[A-Za-z]*", "", body).strip()
        if body.endswith("```"):
            body = body[:-3].strip()
    try:
        value = json.loads(body)
        return value if isinstance(value, dict) else None
    except ValueError:
        pass
    left, right = body.find("{"), body.rfind("}")
    if 0 <= left < right:
        try:
            value = json.loads(body[left : right + 1])
            return value if isinstance(value, dict) else None
        except ValueError:
            return None
    return None


def _plain_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _short_sentences(value: Any, sink: list[str], depth: int = 0) -> None:
    if depth > 4:
        return
    if isinstance(value, str):
        if len(value.split()) >= 4:
            sink.append(value)
    elif isinstance(value, list):
        for part in value[:10]:
            _short_sentences(part, sink, depth + 1)


def _gather_ids(value: Any, sink: set[str], depth: int = 0) -> None:
    if depth > 6:
        return
    if isinstance(value, dict):
        for key, part in value.items():
            if isinstance(part, str) and "id" in key.lower() and _ID_HINT.match(part):
                sink.add(part)
            else:
                _gather_ids(part, sink, depth + 1)
    elif isinstance(value, list):
        for part in value:
            _gather_ids(part, sink, depth + 1)


def _nested_dicts(value: Any, depth: int = 0):
    if depth > 6:
        return
    if isinstance(value, list):
        for part in value:
            yield from _nested_dicts(part, depth + 1)
    elif isinstance(value, dict):
        yield value
        for part in value.values():
            if isinstance(part, (dict, list)):
                yield from _nested_dicts(part, depth + 1)


def _code_like(text: str) -> bool:
    return bool(re.search(r"\d", text)) and bool(_SKU_LIKE.fullmatch(text))


def _code_only(observation: dict[str, Any]) -> bool:
    # a reply with no sentence at all but a status code (one snake_case token) delivered nothing
    texts: list[str] = []
    _short_sentences(list(observation.values()), texts)
    codes = [part for found in [observation] + [v for v in observation.values() if isinstance(v, dict)] for part in found.values() if isinstance(part, str) and "_" in part and " " not in part.strip()]
    return not texts and bool(codes)


def _string_lines(observation: dict[str, Any]) -> list[str]:
    found: list[str] = []
    for value in observation.values():
        if isinstance(value, list) and value and all(isinstance(part, str) for part in value):
            _short_sentences(value, found)
    return found


class Clock:
    def __init__(self) -> None:
        self.started = time.monotonic()
        self.cut = False

    def left(self) -> float:
        left = TASK_SECONDS - (time.monotonic() - self.started)
        # a runaway spend counts as a short clock, so the usual wrap-up path takes over
        return min(left, WRAP_UP_SECONDS - 1) if self.cut else left

    def elapsed(self) -> float:
        return time.monotonic() - self.started


class LLMDesk:
    def __init__(self, clock: Clock, trace: list[dict[str, Any]]):
        self.clock = clock
        self.trace = trace
        self.cost = 0.0
        self.spend: dict[str, list[float]] = {}
        self.dry = False
        self.refusing: set[str] = set()

    @staticmethod
    def _with_cache_marks(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        marked = [dict(m) for m in messages]
        for spot in (0, len(marked) - 1):
            text = marked[spot].get("content")
            if isinstance(text, str) and text:
                marked[spot]["content"] = [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]
        return marked

    def ask(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, models: list[str], max_tokens: int, role: str = "loop", effort: str | None = None, reasoning: dict[str, Any] | None = None) -> dict[str, Any] | None:
        refused: set[str] = set()
        for model in models:
            for pause in RETRY_PAUSES:
                remaining = self.clock.left()
                if remaining < FORCE_SECONDS - 10:
                    return None
                _llm.timeout = int(max(8, min(75, remaining - FORCE_SECONDS + 15)))
                cached = model.startswith("anthropic/") and len(messages) > 2
                request: dict[str, Any] = {"model": model, "messages": self._with_cache_marks(messages) if cached else messages, "temperature": 0, "max_tokens": max_tokens}
                if tools:
                    request.update(tools=tools, tool_choice="auto")
                if reasoning:
                    request["reasoning"] = reasoning
                elif effort and (model.startswith("anthropic/") or "gpt-oss" in model):
                    request["reasoning"] = {"effort": effort}
                outcome = _llm.post_verbose("/inference/chat/completions", request)
                answer = self._take(outcome.data, role)
                if answer is not None:
                    return answer
                if self._ran_dry(outcome.data):
                    # the reply came back and spent its whole allowance without an answer; the same request repeats that, so try the next LLM
                    self.trace.append({"kind": "inference_failure", "model": model, "error": "ran out of tokens before answering"})
                    break
                self.trace.append({"kind": "inference_failure", "model": model, "error": str(outcome.error)[:300] if outcome.error else "empty"})
                status = (outcome.error or {}).get("status") if isinstance(outcome.error, dict) else None
                if status in (402, 403):
                    refused.add(model)
                # only a busy or failing upstream is worth the same request again; anything else goes to the next LLM
                if status not in (429, 500, 502, 503, 504) or pause is None or self.clock.left() < FORCE_SECONDS + 30:
                    break
                time.sleep(pause)
        self.refusing |= refused
        if len(self.refusing) >= 2:
            # two different LLMs refused the key: no further judgement can be bought, so code has to finish
            self.dry = True
        return None

    @staticmethod
    def _ran_dry(data: Any) -> bool:
        if not isinstance(data, dict) or not data.get("choices"):
            return False
        choice = data["choices"][0]
        message = choice.get("message") or {}
        return choice.get("finish_reason") == "length" and not (message.get("content") or message.get("tool_calls"))

    def _take(self, data: Any, role: str) -> dict[str, Any] | None:
        if not isinstance(data, dict) or not data.get("choices"):
            return None
        spent = (data.get("usage") or {}).get("cost")
        if isinstance(spent, (int, float)):
            self.cost += float(spent)
            tally = self.spend.setdefault(role, [0, 0.0])
            tally[0] += 1
            tally[1] += float(spent)
            if self.cost >= RUNAWAY_USD and not self.clock.cut:
                self.clock.cut = True
                self.trace.append({"kind": "runaway", "cost": round(self.cost, 5), "role": role})
        message = data["choices"][0].get("message") or {}
        if message.get("content") or message.get("tool_calls"):
            return message
        return None

    def ask_json(self, prompt: str, max_tokens: int = 700, models: list[str] | None = None, role: str = "judge", effort: str | None = None) -> dict[str, Any] | None:
        message = self.ask([{"role": "user", "content": prompt}], None, models or CHEAP_JUDGES + [JUDGE_MODEL], max_tokens, role, effort)
        return _decode_object((message or {}).get("content") or "")


class WorldDesk:
    def __init__(self, clock: Clock, trace: list[dict[str, Any]], environment: dict[str, Any]):
        self.clock = clock
        self.trace = trace
        self.binding = dict(environment["binding"])
        self.policy = environment["policy_view"]
        self.max_steps = int(self.policy.get("max_steps") or 30)
        self.step = 0
        self.finished = bool(self.policy.get("finished"))
        self.dead = False

    @property
    def query(self) -> str:
        return str(self.policy.get("query") or "")

    @property
    def opening(self) -> Any:
        return self.policy.get("opening")

    def steps_left(self) -> int:
        return self.max_steps - self.step

    def over(self) -> bool:
        return self.finished or self.dead or self.steps_left() <= 0

    def call(self, actions: list[dict[str, Any]]) -> dict[str, Any] | None:
        payload: dict[str, Any] = dict(self.binding)
        payload["step"] = self.step + 1
        payload.update({"action": actions[0]} if len(actions) == 1 else {"actions": actions})
        attempt = 0
        while attempt < 3:
            attempt += 1
            remaining = self.clock.left()
            if remaining < 2:
                return None
            _env.timeout = int(max(3, min(95, remaining - 1)))
            outcome = _env.post_verbose("/environment/call", payload)
            if isinstance(outcome.data, dict):
                return self._accept(outcome.data, len(actions))
            error = outcome.error or {}
            status = error.get("status")
            text = str(error.get("body") or "")
            self.trace.append({"kind": "env_failure", "status": status, "body": text[:300]})
            if status == 402:
                self.dead = True
                return None
            if status != 409:
                if status in (403, 422, 500, 503, 504):
                    return None
                continue
            if "finished" in text:
                self.finished = True
                return None
            expected = re.search(r"expected (\d+)", text)
            if not expected:
                return None
            self.step = int(expected.group(1)) - 1
            payload["step"] = self.step + 1
        return None

    def _accept(self, data: dict[str, Any], count: int) -> dict[str, Any]:
        self.step = data["step"] if isinstance(data.get("step"), int) else self.step + count
        if data.get("finished"):
            self.finished = True
        return data


class SchemaGuard:
    def __init__(self, policy: dict[str, Any]):
        self.tools = [tool for tool in policy.get("tools") or [] if isinstance(tool, dict) and isinstance(tool.get("function"), dict)]
        self.schemas = {tool["function"]["name"]: tool["function"].get("parameters") or {"type": "object", "properties": {}} for tool in self.tools}

    def listed_values(self) -> list[str]:
        # the string choices the tools' own schemas list (category names, among others)
        found = []
        for tool in self.tools:
            for spec in ((tool["function"].get("parameters") or {}).get("properties") or {}).values():
                if isinstance(spec, dict):
                    found.extend(value for value in spec.get("enum") or (spec.get("items") or {}).get("enum") or [] if isinstance(value, str))
        return found
    def knows(self, name: Any) -> bool:
        return name in self.schemas

    def props(self, name: str | None) -> dict[str, Any]:
        return (self.schemas.get(name or "") or {}).get("properties") or {}

    @staticmethod
    def matches(value: Any, kind: str) -> bool:
        if kind == "null":
            return value is None
        if kind == "string":
            return isinstance(value, str)
        if kind == "boolean":
            return isinstance(value, bool)
        if kind == "integer":
            return isinstance(value, int) and not isinstance(value, bool)
        if kind == "number":
            return _plain_number(value)
        if kind == "array":
            return isinstance(value, list)
        if kind == "object":
            return isinstance(value, dict)
        return True

    @staticmethod
    def kinds_of(schema: dict[str, Any]) -> list[str] | None:
        kinds = schema.get("type")
        return [kinds] if isinstance(kinds, str) else kinds

    @staticmethod
    def takes_text(spec: dict[str, Any]) -> bool:
        return spec.get("type") == "string" or "string" in (spec.get("type") or [])

    def convert(self, value: Any, schema: dict[str, Any]) -> Any:
        kinds = self.kinds_of(schema)
        if not kinds:
            return value
        if any(self.matches(value, kind) for kind in kinds):
            whole_float = isinstance(value, float) and value.is_integer()
            return int(value) if "integer" in kinds and whole_float else value
        for kind in kinds:
            try:
                if kind == "integer" and isinstance(value, float) and value.is_integer():
                    return int(value)
                if kind == "integer" and isinstance(value, str) and re.fullmatch(r"\s*-?\d+(\.0+)?\s*", value):
                    return int(float(value))
                if kind == "number" and isinstance(value, str) and re.fullmatch(r"\s*-?\d+(\.\d+)?\s*", value):
                    figure = float(value)
                    return int(figure) if figure.is_integer() else figure
                if kind == "string" and _plain_number(value):
                    return str(value)
                if kind == "boolean" and isinstance(value, str) and value.lower() in ("true", "false"):
                    return value.lower() == "true"
            except (TypeError, ValueError):
                continue
        return value

    def validate(self, value: Any, schema: Any, path: str) -> tuple[Any, list[str]]:
        if not isinstance(schema, dict):
            return value, []
        value = self.convert(value, schema)
        kinds = self.kinds_of(schema)
        if kinds and not any(self.matches(value, kind) for kind in kinds):
            return value, [f"{path}: expected {'/'.join(kinds)}"]
        issues: list[str] = []
        options = schema.get("enum")
        if "enum" in schema and value not in options:
            if isinstance(value, str):
                wanted = value.strip().lower()
                value = next((option for option in options if isinstance(option, str) and option.lower() == wanted), value)
            if value not in options:
                issues.append(f"{path}: must be one of {options}")
        if "const" in schema and value != schema["const"]:
            issues.append(f"{path}: must equal {schema['const']!r}")
        if isinstance(value, str):
            if "minLength" in schema and len(value) < schema["minLength"]:
                issues.append(f"{path}: shorter than {schema['minLength']} characters")
            if "maxLength" in schema and len(value) > schema["maxLength"]:
                value = value[: schema["maxLength"]]
        if _plain_number(value):
            low, high = schema.get("exclusiveMinimum"), schema.get("exclusiveMaximum")
            if "minimum" in schema and value < schema["minimum"]:
                issues.append(f"{path}: below minimum {schema['minimum']}")
            if "maximum" in schema and value > schema["maximum"]:
                issues.append(f"{path}: above maximum {schema['maximum']}")
            if "exclusiveMinimum" in schema and isinstance(low, (int, float)) and value <= low:
                issues.append(f"{path}: must be above {low}")
            if "exclusiveMaximum" in schema and isinstance(high, (int, float)) and value >= high:
                issues.append(f"{path}: must be below {high}")
        if isinstance(value, list):
            if "minItems" in schema and len(value) < schema["minItems"]:
                issues.append(f"{path}: needs at least {schema['minItems']} items")
            if "maxItems" in schema and len(value) > schema["maxItems"]:
                issues.append(f"{path}: allows at most {schema['maxItems']} items")
            element = schema.get("items")
            if isinstance(element, dict):
                repaired_list = []
                for position, part in enumerate(value):
                    part, found = self.validate(part, element, f"{path}[{position}]")
                    repaired_list.append(part)
                    issues += found
                value = repaired_list
        if isinstance(value, dict):
            known = schema.get("properties") or {}
            for field in schema.get("required") or []:
                if field not in value:
                    issues.append(f"{path}.{field}: required")
            if schema.get("additionalProperties") is False:
                for field in list(value):
                    if field not in known:
                        issues.append(f"{path}.{field}: not an allowed argument")
            repaired = {}
            for field, part in value.items():
                if field in known:
                    part, found = self.validate(part, known[field], f"{path}.{field}")
                    issues += found
                repaired[field] = part
            value = repaired
        return value, issues

    def check_args(self, name: str, args: Any) -> tuple[Any, list[str]]:
        return self.validate(args, self.schemas[name], "args")

    def enum_signature(self, name: str, args: dict[str, Any]) -> str:
        known = self.props(name)
        return _dump({key: value for key, value in args.items() if isinstance(known.get(key), dict) and "enum" in known[key]})

    def message_args(self, name: str | None, text: str) -> dict[str, Any] | None:
        schema = self.schemas.get(name or "") or {}
        known = schema.get("properties") or {}
        required = list(schema.get("required") or [])
        free_text = [key for key, spec in known.items() if isinstance(spec, dict) and self.takes_text(spec) and "enum" not in spec]
        field = next((key for key in required if key in free_text), None) or (free_text[0] if len(free_text) == 1 else None)
        if not field or any(key != field for key in required):
            return None
        return {field: text}

    def describe(self, name: str) -> str:
        return next((str(tool["function"].get("description") or "") for tool in self.tools if tool["function"].get("name") == name), "")

    def pick_enum(self, name: str, key: str, text: str, chosen: dict[str, Any] | None = None) -> Any:
        # a fixed choice is never free text: the value the setup reading named for this job, else the value whose
        # describing sentence shares most words with this action's own wording
        spec = self.props(name).get(key) or {}
        options = spec.get("enum") or []
        picked = (chosen or {}).get(key)
        if picked in options:
            return picked
        said = _words(text)
        notes = [line for line in re.split(r"(?<=[.;,:])\s+|\s+or\s+", f"{self.describe(name)} {spec.get('description') or ''}") if line.strip()]

        def fit(option: Any) -> int:
            # the option's own words count most; the words of the clause that names it break ties
            own = _words(option)
            near = set().union(*[_words(line) for line in notes if own & _words(line)]) if notes else set()
            own, near = {word for word in own if len(word) > 3}, {word for word in near if len(word) > 3}
            return 3 * len(own & said) + len(near & said)

        return max(options, key=fit) if options else None

    def counts(self, name: str) -> list[str]:
        return [key for key, spec in self.props(name).items() if isinstance(spec, dict) and spec.get("type") in ("integer", "number")]

    def adds_units(self, name: str, learned: bool | None) -> bool:
        # whether the cart tool adds the units sent to the line (a negative count takes units out) or sets the line to them
        if learned is not None:
            return learned
        return any(((self.props(name).get(key) or {}).get("minimum") or 0) < 0 for key in self.counts(name))

    def zero_count(self, name: str, args: dict[str, Any], learned: bool | None = None) -> bool:
        # a call that empties a line: zero on a cart that sets units, a negative count on one that adds them
        adding = self.adds_units(name, learned)
        for key in self.counts(name):
            value = args.get(key)
            minimum = (self.props(name).get(key) or {}).get("minimum")
            if _plain_number(value) and (value < 0 or (value == 0 and not adding and (minimum is None or minimum <= 0))):
                return True
        return False

    def idle_count(self, name: str, args: dict[str, Any], learned: bool | None = None) -> bool:
        return self.adds_units(name, learned) and any(args.get(key) == 0 for key in self.counts(name))

    def removal_args(self, name: str, ident: str, chosen: dict[str, Any] | None = None, units: int = 1, learned: bool | None = None) -> dict[str, Any] | None:
        # the call that takes one listing out of the cart, built from the schema alone
        known = self.props(name)
        texts = [key for key, spec in known.items() if isinstance(spec, dict) and self.takes_text(spec) and "enum" not in spec]
        counts = [key for key in self.counts(name) if ((known.get(key) or {}).get("minimum") or 0) <= 0]
        modes = [key for key, spec in known.items() if isinstance(spec, dict) and spec.get("enum")]
        if len(texts) != 1 or not (counts or modes):
            return None
        args: dict[str, Any] = {texts[0]: ident}
        adding = self.adds_units(name, learned)
        counts = [key for key in self.counts(name) if adding or ((known.get(key) or {}).get("minimum") or 0) <= 0]
        if counts and not modes:
            minimum = (known.get(counts[0]) or {}).get("minimum")
            args[counts[0]] = max(-max(1, units), minimum) if adding and _plain_number(minimum) else (-max(1, units) if adding else 0)
        else:
            for key in modes:
                args[key] = self.pick_enum(name, key, REMOVE_WORDING, chosen)
            for key in self.counts(name):
                if key in (self.schemas.get(name) or {}).get("required", []):
                    args[key] = max(1, (known.get(key) or {}).get("minimum", 1))
        return args

    def add_args(self, name: str, ident: str, units: int, chosen: dict[str, Any] | None = None) -> dict[str, Any] | None:
        # the call that puts a listing in the cart at the wanted units, built from the schema alone
        known = self.props(name)
        texts = [key for key, spec in known.items() if isinstance(spec, dict) and self.takes_text(spec) and "enum" not in spec]
        if len(texts) != 1:
            return None
        args: dict[str, Any] = {texts[0]: ident}
        for key, spec in known.items():
            if not isinstance(spec, dict) or key in args:
                continue
            if spec.get("enum"):
                args[key] = self.pick_enum(name, key, ADD_WORDING, chosen)
            elif key in self.counts(name):
                args[key] = max(units, spec.get("minimum", 1) or 1)
        missing = [key for key in (self.schemas.get(name) or {}).get("required", []) if key not in args]
        return None if missing else args

    def filler_args(self, name: str, text: str, chosen: dict[str, Any] | None = None, every_choice: bool = False, job: str | None = None) -> dict[str, Any]:
        schema = self.schemas.get(name) or {}
        known = schema.get("properties") or {}
        args: dict[str, Any] = {}
        # a choice the setup reading named, or any fixed choice of a finishing tool, is sent even when the schema leaves it optional
        wanted = list(schema.get("required") or []) + [key for key, spec in known.items() if isinstance(spec, dict) and spec.get("enum") and (key in (chosen or {}) or every_choice)]
        for key in dict.fromkeys(wanted):
            spec = known.get(key) or {}
            if "enum" in spec and spec["enum"]:
                args[key] = self.pick_enum(name, key, job or text, chosen)
            elif self.takes_text(spec):
                args[key] = text[: spec.get("maxLength", len(text))]
            elif spec.get("type") == "boolean":
                args[key] = True
            elif spec.get("type") in ("integer", "number"):
                args[key] = max(1, spec.get("minimum", 1))
        if len(known) == 1:
            for key, spec in known.items():
                if key not in args and isinstance(spec, dict) and spec.get("type") == "string" and "enum" not in spec:
                    args[key] = text[: spec.get("maxLength", len(text))]
        return args


class Ledger:
    def __init__(self) -> None:
        self.plan: dict[str, Any] = {}
        self.seen_ids: set[str] = set()
        self.stated: list[str] = []
        self.revealed: list[str] = []
        self.free: list[str] = []
        self.dropped: list[str] = []
        self.loose: list[frozenset[str]] = []
        self.looked_up: list[tuple[frozenset[str], list[str]]] = []
        self.budget_now: Any = None
        self.current: list[str] | None = None
        self.version = 0
        self.heard: set[str] = set()
        self.said: list[str] = []
        self.records: list[Any] = []
        self.cart_ids: list[str] = []
        self.cart_units: dict[str, int] = {}
        self.taken_out: dict[str, int] = {}
        self.adds_units: bool | None = None
        self.last_cart_view: Any = None
        self.add_signature: str | None = None
        self.shelf = Shelf()

    def sync_cart(self, observation: Any) -> None:
        # a cart result that lists its lines is the truth about the cart; the tracked order of additions is kept for what remains
        holders = [observation] + ([part for part in observation.values() if isinstance(part, dict)] if isinstance(observation, dict) else [])
        known = set(self.shelf.rows) | self.seen_ids
        for holder in holders:
            if not isinstance(holder, dict):
                continue
            lists = [part for part in holder.values() if isinstance(part, list) and all(isinstance(row, dict) for row in part)]
            filled = [part for part in lists if part]
            if not lists or len(filled) > 1:
                continue
            rows = filled[0] if filled else []
            found = []
            units: dict[str, int] = {}
            price_key = max(self.shelf.price_votes, key=lambda name: self.shelf.price_votes[name]) if self.shelf.price_votes else None
            for row in rows:
                ident = next((part for part in row.values() if isinstance(part, str) and part in known), None) or _row_id(row)
                if not ident:
                    break
                found.append(ident)
                # a line's units: its one whole-number field besides the price the listing row also shows
                whole = [key for key, part in row.items() if isinstance(part, int) and not isinstance(part, bool) and key != price_key and part != (self.shelf.rows.get(ident) or {}).get(key)]
                if len(whole) == 1:
                    units[ident] = row[whole[0]]
            else:
                self.cart_ids = [ident for ident in self.cart_ids if ident in found] + [ident for ident in found if ident not in self.cart_ids]
                self.cart_units = units
                return

    def remember(self, value: Any) -> None:
        self.records.append(value)

    def note_ids(self, value: Any) -> None:
        _gather_ids(value, self.seen_ids)

    def summary(self) -> str:
        if self.current is None:
            parts = [f"{r} (stated)" for r in self.stated if r not in self.dropped]
            parts.extend(f"{r} (revealed by the shopper)" for r in self.revealed if r not in self.dropped)
        else:
            parts = list(self.current)
        # a "same as that listing" condition gives way to the value read off that listing
        parts = [part for part in parts if not any(_same_property(part, [prop]) and any(name in part.lower() for name in names) for prop, names in self.looked_up)]
        text = "; ".join(parts) or "see the request"
        if self.budget_now is not None:
            unit = "cents" if self.plan.get("price_cap_unit") == "cents" and self.budget_now == self.plan.get("price_cap") else ""
            text += f"; total price at most {self.budget_now} {unit or self.plan.get('currency') or ''}".rstrip()
        wanted = _basket(self.plan)
        if wanted:
            text += "; one order of: " + ", ".join(f"{entry['quantity']} x {entry['product']}" for entry in wanted)
        else:
            text += f"; quantity {self.plan.get('quantity') or 1}"
        if self.dropped:
            text += "; no longer required: " + "; ".join(self.dropped)
        if self.free:
            text += "; shopper does not care about: " + ", ".join(self.free)
        return text

    def records_of(self, item: str) -> list[str]:
        hits: list[str] = []

        def visit(value: Any, depth: int) -> None:
            if depth > 6 or len(hits) >= 8:
                return
            if isinstance(value, dict):
                if any(isinstance(part, str) and part == item for part in value.values()):
                    text = _dump(_trim(value, 400))[:2500]
                    if text not in hits:
                        hits.append(text)
                    return
                for part in value.values():
                    visit(part, depth + 1)
            elif isinstance(value, list):
                for part in value:
                    visit(part, depth + 1)

        for record in self.records:
            visit(record, 0)
        return hits[-4:]

    def property_menu(self, products: list[str] | None = None) -> tuple[str, int]:
        # in one order of several products each product gets its own menu, so no product's listings crowd out another's
        if products and self.shelf.range_key():
            parts, total = [], 0
            for product in products:
                menu, rich = self.single_menu(product)
                total += rich
                if menu:
                    parts.append(f"For the {product}:\n{menu}")
            return "\n".join(parts), total
        # one product: its own range when the shop labels ranges, so listings opened for a lookup or by mistake stay out
        own = self.single_menu(str(self.plan.get("product") or "")) if self.plan.get("product") else ("", 0)
        if own[0]:
            return own
        # before the product's own range is seen, listings opened for a lookup elsewhere must not stand in for it
        lookups = {self.shelf.main_range(name, strict=True) for entry in self.plan.get("references") or [] if isinstance(entry, dict) for name in [str(entry.get("range") or "")] + [str(old) for old in entry.get("ranges") or []]} - {None}
        return self.tally_menu(True, None, lookups) if lookups else self.single_menu(None)

    def single_menu(self, product: str | None) -> tuple[str, int]:
        # attribute maps nested inside a listing come first; the listing's own row fields (seller, title, range) are not attributes
        menu = self.tally_menu(True, product)
        if menu[1] or not self.shelf.opened:
            return menu
        return self.tally_menu(False, product)

    def tally_menu(self, nested_only: bool, product: str | None = None, skip: set[str] | None = None) -> tuple[str, int]:
        counts: dict[str, int] = {}
        examples: dict[str, list[str]] = {}
        values: dict[str, set[str]] = {}
        rich = 0
        label_wanted = self.shelf.main_range(product, strict=True) if product else None
        if product and not label_wanted:
            return "", 0

        range_key = self.shelf.range_key() if product or skip else None
        row_fields = self.shelf.listing_keys

        def visit(value: Any, depth: int, owner: str | None) -> None:
            nonlocal rich
            if depth > 6:
                return
            if isinstance(value, dict):
                # a map that carries a listing's own id is that listing's row, under whatever key the shop names the id
                own = next((part for part in value.values() if isinstance(part, str) and part in self.shelf.rows), None) or _row_id(value)
                if own:
                    owner = self.shelf.range_of(own) or (str(value.get(range_key)) if range_key and value.get(range_key) else None)
                elif range_key and isinstance(value.get(range_key), str):
                    owner = value[range_key].strip().lower()
                facts = [(key, part) for key, part in value.items() if isinstance(part, str) and 0 < len(part) <= 70]
                mine = (not label_wanted or owner == label_wanted) and not (skip and owner in skip)
                if mine and len(facts) >= (3 if nested_only else 5) and not (nested_only and own):
                    rich += 1
                    for key, part in facts:
                        if _code_like(part) and ("-" in part or "_" in part):
                            continue
                        label = re.sub(r"[_.]+", " ", str(key)).strip().lower()
                        # a field the search rows show too belongs to the listing (its text, seller, maker), not to the product's choices
                        if key in row_fields:
                            continue
                        counts[label] = counts.get(label, 0) + 1
                        values.setdefault(label, set()).add(part.lower())
                        pool = examples.setdefault(label, [])
                        if part.lower() not in pool and len(pool) < 5:
                            pool.append(part.lower())
                for part in value.values():
                    if isinstance(part, (dict, list)):
                        visit(part, depth + 1, owner)
            elif isinstance(value, list):
                for part in value:
                    visit(part, depth + 1, owner)

        for record in self.records:
            visit(record, 0, None)
        # a field whose value differs on every listing names that listing, not a choice the shopper makes
        kept = [label for label in counts if not (counts[label] >= 4 and len(values[label]) == counts[label])]
        # a field many listings carry with several different values splits the candidates best; one shared value splits nothing
        # a field that is different on most listings it appears on is nearly a listing's name, so it ranks lower
        top = sorted(kept, key=lambda label: -(counts[label] * min(len(values[label]), 5) * (0.4 if counts[label] >= 3 and len(values[label]) >= 0.6 * counts[label] else 1)))[:12 if product else 28]
        return "\n".join(f"- {label} (in {counts[label]} listings, {len(values[label])} different values): " + " | ".join(examples.get(label, [])) for label in top), rich

    def recurring_labels(self) -> str:
        tally: dict[str, dict[str, int]] = {}
        for found in (d for record in self.records for d in _nested_dicts(record)):
            if len(found) < 4:
                continue
            for key, value in found.items():
                if isinstance(value, str) and 0 < len(value) <= 40 and not _code_like(value):
                    column = tally.setdefault(str(key), {})
                    column[value] = column.get(value, 0) + 1
        rows = []
        for key, column in tally.items():
            if sum(column.values()) < 4 or not 2 <= len(column) <= 8 or max(column.values()) < 2:
                continue
            ordered = sorted(column.items(), key=lambda pair: -pair[1])
            rows.append(f"- {key}: " + " | ".join(f"{value} (x{n})" for value, n in ordered))
        return "\n".join(rows[:12])


def _basket(plan: dict[str, Any]) -> list[dict[str, Any]]:
    # several kinds of product in one order; a single entry is an ordinary request
    entries = []
    for entry in plan.get("products") or []:
        if isinstance(entry, dict) and str(entry.get("product") or "").strip():
            units = entry.get("quantity")
            units = units if isinstance(units, int) and not isinstance(units, bool) and units > 0 else 1
            entries.append({"product": str(entry["product"]).strip(), "quantity": units, "requirements": [str(r) for r in entry.get("requirements") or [] if r]})
    return entries if len(entries) >= 2 else []


def _words(value: Any) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", str(value).lower()))


def _content_words(value: Any) -> set[str]:
    return {word for word in _words(value) if len(word) > 2}


def _need_property(need: Any) -> frozenset[str]:
    # "<product> - property: constraint" or "property: constraint" names its property before the colon
    head = str(need).partition(":")[0]
    return frozenset(_words(head.rpartition(" - ")[2]))


def _identifies(need: str, references: Any) -> bool:
    # a condition whose property and value both sit in a looked-up listing's identifying attributes describes that listing, not the item to buy
    value = _tokens(str(need).partition(":")[2])
    marks = [_tokens(found) for entry in references or [] if isinstance(entry, dict) for found in [entry.get("identify")] + list(entry.get("identified") or [])]
    return bool(value) and any(f" {value} " in f" {mark} " and _need_property(need) & set(mark.split()) for mark in marks)


def _same_property(need: Any, props: Any) -> bool:
    own = _need_property(need)
    return any(prop and own and (prop <= own or own <= prop) for prop in props)


def _split_need(need: str) -> tuple[set[str], set[str]]:
    head, _, tail = str(need).partition(":")
    return _words(head), _words(tail)


def _row_id(found: dict[str, Any]) -> str | None:
    loose = None
    for key, part in found.items():
        if isinstance(part, str) and _ID_HINT.match(part):
            name = str(key).lower()
            if name == "id":
                return part
            # "id" must be a word of the key (item_id, productId), not letters inside one, and a loose id carries a digit
            if "id" in _words(re.sub(r"([a-z])([A-Z])", r"\1 \2", str(key))) and any(ch.isdigit() for ch in part) and loose is None:
                loose = part
    return loose


class Shelf:
    # per listing id: its plain row fields, its facts, and whether its page was opened
    def __init__(self) -> None:
        self.rows: dict[str, dict[str, Any]] = {}
        self.facts: dict[str, dict[str, str]] = {}
        self.opened: set[str] = set()
        self.price_votes: dict[str, int] = {}
        self.range_memo: tuple[int, str | None] = (-1, None)
        self.listing_keys: set[str] = set()
        self.figures: dict[str, dict[str, float]] = {}
        self.stock_field: str | None = None
        self.stock_kind = ""
        self.stock_out: list[Any] = []

    def take(self, observation: Any, opened: bool) -> None:
        for found in _nested_dicts(observation):
            ident = _row_id(found)
            if not ident:
                continue
            flat = self.rows.setdefault(ident, {})
            flat.update({key: part for key, part in found.items() if not isinstance(part, (dict, list))})
            if not opened:
                self.listing_keys.update(key for key, part in found.items() if not isinstance(part, (dict, list)))
            known = self.facts.setdefault(ident, {})
            for inner in _nested_dicts(found):
                if inner is not found and _row_id(inner) not in (None, ident):
                    continue
                known.update({str(key): part for key, part in inner.items() if isinstance(part, str) and 0 < len(part) <= 200})
                self.figures.setdefault(ident, {}).update({str(key): part for key, part in inner.items() if _plain_number(part)})
            if opened:
                self.opened.add(ident)

    def learn_price(self, cart_view: Any) -> None:
        # the price field is the row field whose figure the cart repeats for the same listing
        for found in _nested_dicts(cart_view):
            ident = _row_id(found)
            row = self.rows.get(ident or "")
            if not row:
                continue
            figures = [part for part in found.values() if _plain_number(part) and part > 1]
            for key, part in row.items():
                if _plain_number(part) and part in figures:
                    self.price_votes[key] = self.price_votes.get(key, 0) + 1

    def sole_field(self, numeric: bool) -> str | None:
        # without a cart to learn from, a field only counts when it is the one field of its type most rows carry
        if len(self.rows) < 3:
            return None
        cover: dict[str, int] = {}
        for row in self.rows.values():
            for key, part in row.items():
                if (_plain_number(part) if numeric else isinstance(part, bool)):
                    cover[key] = cover.get(key, 0) + 1
        common = [key for key, n in cover.items() if n >= 0.8 * len(self.rows)]
        return common[0] if len(common) == 1 else None

    def price(self, ident: str) -> float | None:
        key = max(self.price_votes, key=lambda name: self.price_votes[name]) if self.price_votes else self.sole_field(True)
        value = (self.rows.get(ident) or {}).get(key) if key else None
        return float(value) if _plain_number(value) else None

    def sold_out(self, ident: str) -> bool:
        if self.stock_field:
            value = (self.rows.get(ident) or {}).get(self.stock_field)
            if self.stock_kind == "count" and _plain_number(value):
                return value <= 0
            if isinstance(value, bool):
                return value in [mark for mark in self.stock_out if isinstance(mark, bool)]
            if isinstance(value, str):
                return value.strip().lower() in [str(mark).strip().lower() for mark in self.stock_out if isinstance(mark, str)]
        key = self.sole_field(False)
        return key is not None and (self.rows.get(ident) or {}).get(key) is False

    def range_key(self) -> str | None:
        if self.range_memo[0] != len(self.rows):
            self.range_memo = (len(self.rows), self.find_range_key())
        return self.range_memo[1]

    def find_range_key(self) -> str | None:
        # the label every row carries with the fewest different values is the shop's range (category) label
        if len(self.rows) < 4:
            return None
        tally: dict[str, set[str]] = {}
        for row in self.rows.values():
            for key, part in row.items():
                if isinstance(part, str) and 0 < len(part) <= 60 and not _code_like(part):
                    tally.setdefault(key, set()).add(part.strip().lower())
        cover = {key: sum(1 for row in self.rows.values() if isinstance(row.get(key), str) and 0 < len(row[key]) <= 60) for key in tally}
        usable = [key for key in tally if cover[key] >= 0.8 * len(self.rows) and len(tally[key]) < len(self.rows)]
        if not usable:
            return None
        return min(usable, key=lambda key: (len(tally[key]), -cover[key]))

    def range_of(self, ident: str) -> str | None:
        key = self.range_key()
        value = (self.rows.get(ident) or {}).get(key) if key else None
        return value.strip().lower() if isinstance(value, str) else None

    def main_range(self, product: str = "", strict: bool = False) -> str | None:
        # the range label with the widest word overlap with the wanted product (a word and its plural count as one), else the range of most opened listings
        labels = {self.range_of(ident) for ident in self.rows} - {None}
        wanted = {word for word in _words(product) if len(word) > 2}

        def same(word: str, other: str) -> bool:
            return word == other or other in (word + "s", word + "es") or word in (other + "s", other + "es")

        tail = re.findall(r"[a-z0-9]+", product.lower())[-1:]

        def overlap(label: str) -> tuple[int, int, int]:
            # whole words, a plural counting as its singular; then the head noun (last word) agreeing; then fewest words beyond the product
            have = {word for word in _words(label) if len(word) > 2}
            hits = sum(1 for word in wanted if any(same(word, other) for other in have))
            head = re.findall(r"[a-z0-9]+", label.lower())[-1:]
            agrees = int(bool(tail and head and same(tail[0], head[0])))
            extra = sum(1 for other in have if not any(same(word, other) for word in wanted))
            return hits, agrees, -extra

        scored = sorted(((overlap(label), label) for label in labels), reverse=True)
        if scored and scored[0][0][0] > 0:
            return scored[0][1]
        if strict:
            return None
        counts: dict[str, int] = {}
        for ident in self.opened:
            label = self.range_of(ident)
            if label:
                counts[label] = counts.get(label, 0) + 1
        return max(counts, key=lambda label: counts[label]) if counts else None

    def unopened(self, label: str | None, cap: float | None, quantity: int, below: float | None = None) -> list[tuple[float, str]]:
        found = []
        for ident in self.rows:
            if ident in self.opened or (label and self.range_of(ident) != label):
                continue
            cost = self.price(ident)
            if cost is None or self.sold_out(ident) or (cap is not None and _plain_number(cap) and cost * quantity > cap) or (below is not None and cost >= below):
                continue
            found.append((cost, ident))
        return sorted(found)

    def carried(self, need: str) -> bool:
        # does any opened listing state this property under some key of its own
        wanted, _ = _split_need(need)
        wanted = {word for word in wanted if len(word) > 2}
        if not wanted:
            return True
        return any(wanted & _words(key) for ident in self.opened for key in (self.facts.get(ident) or {}))

    def literal(self, ident: str, need: str) -> bool:
        # every word of the wanted value sits inside the value of a key that names the property
        wanted, value = _split_need(need)
        wanted = {word for word in wanted if len(word) > 2}
        if not wanted or not value:
            return False
        return any(wanted & _words(key) and value <= _words(part) for key, part in (self.facts.get(ident) or {}).items())


class SetupAdvisor:
    def __init__(self, llm: LLMDesk, world: WorldDesk, guard: SchemaGuard):
        self.llm = llm
        self.world = world
        self.guard = guard

    def read(self) -> dict[str, Any]:
        catalogue = "\n".join(
            f"- {tool['function']['name']}: {tool['function'].get('description', '')} | arguments: {_dump(tool['function'].get('parameters') or {})}"
            for tool in self.guard.tools
        )
        fixed_part = SETUP_PROMPT + "\n\nTOOLS:\n" + catalogue
        task_part = "REQUEST:\n" + self.world.query
        if self.world.opening is not None:
            task_part += "\n\nOPENING OBSERVATION:\n" + _dump(self.world.opening)
        parts = [{"type": "text", "text": fixed_part, "cache_control": {"type": "ephemeral"}}, {"type": "text", "text": task_part}]
        best: dict[str, Any] = {}
        # the cheap reader reads the request; a reading without the core roles leaves the loop blind, so the stronger LLM and then another vendor get the same request
        for model in (READ_MODEL, SETUP_BACKUP, SETUP_MODEL, ALT_MODEL):
            # the cheap reader writes the whole reading out without reasoning first, and a four-product basket runs past 4000 tokens, so it gets the longest allowance
            message = self.llm.ask([{"role": "user", "content": parts}], None, [model], 2000 if model == SETUP_MODEL else 8000 if model == READ_MODEL else 4000, "setup", "low" if model in (SETUP_MODEL, SETUP_BACKUP) else None)
            plan = _decode_object((message or {}).get("content") or "") or {}
            if self.usable(plan) and (model != READ_MODEL or self.complete(plan)):
                return plan
            best = best or plan
            if self.llm.dry:
                break
        return best

    def usable(self, plan: dict[str, Any]) -> bool:
        roles = plan.get("roles") if isinstance(plan.get("roles"), dict) else {}
        return all(self.guard.knows(roles.get(role)) for role in ("search", "cart", "order"))

    def complete(self, plan: dict[str, Any]) -> bool:
        # the cheap reader's reading stands only when every part the loop leans on came back in shape
        products = plan.get("products")
        if not plan.get("product") or not isinstance(products, list) or not products:
            return False
        if not all(isinstance(entry, dict) and entry.get("product") and isinstance(entry.get("requirements"), list) for entry in products):
            return False
        if not all(_plain_number(entry.get("quantity")) and entry["quantity"] >= 1 for entry in products):
            return False
        # two products whose names together make up one category the shop's tools list are one shelf read as two kinds
        names = [_content_words(entry["product"]) for entry in products]
        if any(first and second and first != second and (first | second) <= _content_words(label) for label in self.guard.listed_values() for first in names for second in names):
            return False
        roles = plan["roles"]
        values = plan.get("finish_values") if isinstance(plan.get("finish_values"), dict) else {}
        if roles.get("order") == roles.get("decline") and not all(isinstance(values.get(job), dict) and values.get(job) for job in ("order", "decline")):
            # one tool both buys and refuses: without a value per job the finish cannot be told apart
            return False
        if not all(isinstance(plan.get(key), list) for key in ("references", "rules", "ask_first")):
            return False
        return True


class InterpretAdvisor:
    def __init__(self, llm: LLMDesk, ledger: Ledger, clock: Clock, query: str):
        self.llm = llm
        self.ledger = ledger
        self.clock = clock
        self.query = query

    def read(self, lines: list[str], question: str | None) -> tuple[str, str]:
        book = self.ledger
        fresh = [line for line in lines if line not in book.heard]
        if not fresh:
            return "", ""
        book.heard.update(fresh)
        if self.clock.left() < WRAP_UP_SECONDS:
            return "", ""
        sections = [INTERPRET_PROMPT, "Input - the shopper's request (context only):\n" + self.query, "Input - requirements gathered so far:\n" + book.summary()]
        if book.said:
            sections.append("Input - earlier exchanges with the shopper, oldest first:\n" + "\n".join(book.said[-12:]))
        if question:
            sections.append("Input - the agent's question:\n" + question)
        sections.append("Input - what the shopper says now:\n" + "\n".join(fresh))
        book.said.append((f"asked: {question} -> " if question else "") + "shopper: " + " / ".join(fresh))
        verdict = self.llm.ask_json("\n\n".join(sections), 3000, [READ_MODEL, JUDGE_MODEL], "interpret")
        if not verdict:
            return "", ""
        reply = str(verdict.get("reply") or "").strip() if verdict.get("asked_us") is True else ""
        return self._apply(verdict), reply

    def _apply(self, verdict: dict[str, Any]) -> str:
        book = self.ledger
        changes = []
        for key in ("needs", "requirements_now"):
            # a reply that maps properties to values instead of listing them still names conditions
            if isinstance(verdict.get(key), dict):
                verdict[key] = [f"{prop}: {value}" for prop, value in verdict[key].items()]
        for entry in verdict.get("references") or []:
            if not isinstance(entry, dict) or not str(entry.get("seller") or "").strip() or not str(entry.get("property") or "").strip():
                continue
            # a value read off another listing joins the request's own lookups; the attributes that find that listing stay out of the conditions
            # with several products, a lookup named without its product belongs to the one its range names
            owner = next((item["product"] for item in _basket(book.plan) if _words(item["product"]) & _words(entry.get("range"))), None)
            entry = {**entry, "product": str(entry.get("product") or owner or book.plan.get("product") or ""), "tolerance": str(entry.get("relation") or "")}
            known = next((old for old in book.plan.get("references") or [] if isinstance(old, dict) and _same_property(str(old.get("property")), [_need_property(str(entry["property"]))]) and str(entry["seller"]).strip().lower() in [str(old.get("seller") or "").strip().lower()] + list(old.get("hops") or [])), None)
            if known is not None:
                # the same lookup told again: its wording may identify the listing better, but it is no second lookup
                known.setdefault("identified", []).append(entry.get("identify"))
            else:
                book.plan.setdefault("references", []).append(entry)
            # needs told in the same breath as the lookup may only describe that listing; the listing itself settles it once found
            (known if known is not None else entry).setdefault("with_needs", []).extend(item for item in verdict.get("needs") or [] if isinstance(item, str))
            need = f"{entry['property']}: {entry.get('relation') or 'same as'} the value on the {entry.get('range')} listing sold by {entry['seller']} with {entry.get('identify')}"
            marks = [_need_property(part) for part in str(entry.get("identify") or "").split(",")]
            # a lookup told again adds no second pointer condition and no second instruction
            pointer = [need] if known is None else []
            verdict["needs"] = [item for item in verdict.get("needs") or [] if isinstance(item, str) and not _same_property(item, marks)] + pointer
            # needs an earlier reply drew from the same listing's identifying attributes leave as well
            book.revealed[:] = [item for item in book.revealed if not _identifies(item, [entry])]
            if isinstance(verdict.get("requirements_now"), list):
                verdict["requirements_now"] = [item for item in verdict["requirements_now"] if isinstance(item, str) and not _same_property(item, marks)] + [need]
            if known is not None:
                continue
            changes.append(f"new lookup: search the {entry.get('range')} range, open the listings sold by {entry['seller']}, take the one with {entry.get('identify')}, read its {entry['property']} (following any further offer it refers to) and judge the {entry['property']} by it; {entry.get('identify')} are not conditions on the item you buy")
        # a condition the request states holds until the shopper names a new value for that property; indifference never lifts it
        for named in list(verdict.get("needs") or []) + list(verdict.get("released") or []):
            if isinstance(named, str) and _need_property(named) and _need_property(named) not in book.loose:
                book.loose.append(_need_property(named))
        firm = [need for need in book.stated if not _same_property(need, book.loose)]
        held = [_need_property(need) for need in firm]
        latest = verdict.get("requirements_now")
        if isinstance(latest, list) and all(isinstance(entry, str) for entry in latest):
            # the attributes that single out a looked-up listing are no conditions on the item to buy, unless the reading stated them
            latest = [entry for entry in latest if entry in book.stated or not _identifies(entry, book.plan.get("references"))]
            book.current = [entry for entry in latest if entry.strip() and not _same_property(entry, held)] + firm
        verdict["no_preference"] = [prop for prop in verdict.get("no_preference") or [] if not _same_property(prop, held)]
        verdict["dropped"] = [gone for gone in verdict.get("dropped") or [] if not _same_property(gone, held)]
        for need in verdict.get("needs") or []:
            if isinstance(need, str) and need and need not in book.revealed:
                book.revealed.append(need)
                changes.append(f"new requirement: {need}")
        for prop in verdict.get("no_preference") or []:
            if isinstance(prop, str) and prop and prop not in book.free:
                book.free.append(prop)
        for gone in verdict.get("dropped") or []:
            if isinstance(gone, str) and gone and gone not in book.dropped:
                book.dropped.append(gone)
                changes.append(f"dropped: {gone}")
        budget = verdict.get("new_budget")
        if _plain_number(budget) and budget > 0 and budget != book.budget_now:
            book.budget_now = budget
            changes.append(f"new budget: {budget}")
        if verdict.get("unavailable"):
            changes.append("an item became unavailable; remove it from the cart and do not order it")
        if changes:
            book.version += 1
        tail = " (" + "; ".join(changes) + ")" if changes else ""
        return "Meaning: " + str(verdict.get("meaning") or "") + tail


class PlanAdvisor:
    def __init__(self, llm: LLMDesk, ledger: Ledger, query: str):
        self.llm = llm
        self.ledger = ledger
        self.query = query

    def questions(self, count: int, asked: list[str], menu: str) -> list[dict[str, Any]]:
        must_learn = self.ledger.plan.get("ask_first")
        sections = [PLAN_PROMPT.replace("QUESTION_COUNT", str(count)), "Input - the shopper's request:\n" + self.query, "Input - requirements gathered so far:\n" + self.ledger.summary()]
        if asked:
            sections.append("Input - questions already asked, with the shopper's answers:\n" + "\n".join(asked))
        if must_learn:
            sections.append("ASK THE CUSTOMER FIRST: " + "; ".join(str(entry) for entry in must_learn or []))
        hints = [str(entry) for entry in self.ledger.plan.get("hint_properties") or [] if isinstance(entry, str)]
        if hints:
            sections.append("Input - properties the story points at without naming them (strong candidates when the listings carry them): " + ", ".join(hints))
        sections.append("Input - properties in the listings' details seen so far (name, how many listings, example values):\n" + (menu or "(none seen yet)"))
        answer = self.llm.ask_json("\n\n".join(sections), 3000, [READ_MODEL, JUDGE_MODEL], "plan")
        return [entry for entry in (answer or {}).get("questions") or [] if isinstance(entry, dict) and str(entry.get("question") or "").strip()]


STOCK_PROMPT = """Role: you read a few listing rows from an online shop and find the field that tells whether a listing can be bought now.

Rules:
1. "field": the exact field name that shows availability or stock, or null when no field does.
2. "kind": "flag" when the field is true/false, "count" when it is a number of units on hand, "text" when it is words.
3. "unavailable": for a flag, the one value (true or false) that means it cannot be bought now; for text, every wording that means it cannot be bought now, both those in the rows and the usual ones a shop of this kind writes; [] for a count.

Output: ONE JSON object and nothing else: {"field": null, "kind": "", "unavailable": []}"""

VERIFY_LINES_PROMPT = VERIFY_PROMPT.rsplit("Output:", 1)[0] + """Several items follow, one per cart line; judge each one on its own against the conditions of the product it is.

Output: ONE JSON object, nothing else: {"items": [{"item": the product id, "fits": true or false, "problems": [{"condition": the condition exactly as listed, "kind": "contradicted" or "not_shown", "note": a few words}]}]}, one entry per item."""


class VerifyAdvisor:
    def __init__(self, llm: LLMDesk, ledger: Ledger, clock: Clock, trace: list[dict[str, Any]], query: str):
        self.llm = llm
        self.ledger = ledger
        self.clock = clock
        self.trace = trace
        self.query = query
        self.memo: dict[str, str] = {}
        self.passed: dict[str, None] = {}
        self.table: VerdictTable | None = None
        self.judged: set[str] = set()
        self.soft: dict[str, int] = {}

    def check(self, item: str, purpose: str) -> str:
        if self.clock.left() < WRAP_UP_SECONDS + 20:
            return ""
        book = self.ledger
        memo_key = f"{item}|{book.version}"
        if memo_key in self.memo and (purpose != "ORDER" or self.table is None or memo_key in self.judged):
            return self.memo[memo_key]
        records = book.records_of(item)
        if not records:
            return ""
        if self.table is not None:
            # the table judges every step short of the order; the judge LLM is kept for the order itself
            verdict = self.table_check(item, purpose)
            if verdict is not None:
                self.memo[memo_key] = verdict
                return verdict
        sections = [
            VERIFY_PROMPT,
            "Input - the shopper's request:\n" + self.query,
            "Input - current requirements (stated, revealed and changed):\n" + book.summary(),
            f"Input - the item to {purpose}: {item}\nInput - records of this item seen in the shop, oldest first:\n" + "\n".join(records),
        ]
        if book.last_cart_view is not None:
            sections.append("Input - latest cart view:\n" + _dump(book.last_cart_view)[:3000])
        labels = book.recurring_labels()
        if labels:
            sections.append("Input - labels that recur on this shop's listings, with counts (the item's category label has to be the one matching the request's wording as a whole, not a neighbouring one):\n" + labels)
        # the item check is the judge's seat at every spend level
        result = self.llm.ask_json("\n\n".join(sections), 3000, role="verify", effort="low")
        return self.settle(item, result, memo_key)

    def check_lines(self, items: list[str], purpose: str) -> str:
        # every cart line in one judge call; a line the table already settles, or one judged before, takes no seat
        if len(items) <= 1 or self.clock.left() < WRAP_UP_SECONDS + 20:
            return " ".join(verdict for verdict in (self.check(item, purpose) for item in items) if verdict)
        book = self.ledger
        verdicts: dict[str, str] = {}
        asks: list[str] = []
        for item in items:
            memo_key = f"{item}|{book.version}"
            if memo_key in self.memo and (self.table is None or memo_key in self.judged):
                verdicts[item] = self.memo[memo_key]
                continue
            if not book.records_of(item):
                verdicts[item] = ""
                continue
            if self.table is not None:
                verdict = self.table_check(item, purpose)
                if verdict is not None:
                    self.memo[memo_key] = verdict
                    verdicts[item] = verdict
                    continue
            asks.append(item)
        if asks:
            sections = [
                VERIFY_LINES_PROMPT,
                "Input - the shopper's request:\n" + self.query,
                "Input - current requirements (stated, revealed and changed):\n" + book.summary(),
            ]
            sections.extend(f"Input - the item to {purpose}: {item}\nInput - records of this item seen in the shop, oldest first:\n" + "\n".join(book.records_of(item)) for item in asks)
            if book.last_cart_view is not None:
                sections.append("Input - latest cart view:\n" + _dump(book.last_cart_view)[:3000])
            labels = book.recurring_labels()
            if labels:
                sections.append("Input - labels that recur on this shop's listings, with counts (the item's category label has to be the one matching the request's wording as a whole, not a neighbouring one):\n" + labels)
            result = self.llm.ask_json("\n\n".join(sections), max(3000, 700 * len(asks) + 500), role="verify", effort="low") or {}
            each = {str(entry.get("item")): entry for entry in result.get("items") or [] if isinstance(entry, dict)}
            for item in asks:
                verdicts[item] = self.settle(item, each.get(item), f"{item}|{book.version}")
        return " ".join(verdict for verdict in verdicts.values() if verdict)

    def settle(self, item: str, result: dict[str, Any] | None, memo_key: str) -> str:
        self.judged.add(memo_key)
        problems = self.blocking(item, result) if result and result.get("fits") is False else []
        # an item held only on conditions its record does not show, never on a contrary value, is remembered as the fallback to a refusal
        kinds = [str(problem.get("kind") or "contradicted").lower() if isinstance(problem, dict) else "contradicted" for problem in (result or {}).get("problems") or []]
        if result and result.get("fits") is False and kinds and all(kind.startswith("not") for kind in kinds):
            self.soft[item] = len(kinds)
        else:
            self.soft.pop(item, None)
        self.trace.append({"kind": "verify", "item": item, "result": result, "holds": problems})
        verdict = ""
        if problems:
            self.passed.pop(item, None)
            verdict = f"An independent check says {item} does NOT meet every requirement: " + "; ".join(problems[:5]) + ". Unless you are certain the check is wrong, choose another fully verified fit."
        elif result:
            self.passed.pop(item, None)
            self.passed[item] = None
        self.memo[memo_key] = verdict
        return verdict

    def table_check(self, item: str, purpose: str) -> str | None:
        table = self.table
        shelf = self.ledger.shelf
        product = self.product_of(item)
        found = table.row(item, product)
        if found["unknown"]:
            # the other open listings of the range ride along in the same call
            others = [ident for ident in table.pool(product, bool(_basket(self.ledger.plan))) if ident != item and table.row(ident, product)["unknown"]][:4]
            table.resolve([item] + others, product)
            found = table.row(item, product)
        # a fail quoting the listing's own other value under the need's key is as firm as one found in code
        code_fails = [(need, evidence) for need, evidence in found["fail"] if (item, need.text) not in table.settled or (item, need.text) in table.contrary]
        self.trace.append({"kind": "table", "item": item, "purpose": purpose, "fail": [n.text for n, _ in found["fail"]], "open": [n.text for n, _ in found["unknown"]]})
        if purpose == "ORDER":
            # a cell the listing's own table fails needs no judge; anything else goes to the judge
            if code_fails:
                self.passed.pop(item, None)
                return table.hold_text(item, code_fails)
            return None
        if found["fail"]:
            self.passed.pop(item, None)
            return table.hold_text(item, found["fail"])
        if not found["unknown"]:
            self.passed.pop(item, None)
            self.passed[item] = None
        return ""

    def product_of(self, item: str) -> str:
        entries = _basket(self.ledger.plan)
        if not entries:
            return str(self.ledger.plan.get("product") or "")
        shelf = self.ledger.shelf
        label = shelf.range_of(item)
        for entry in entries:
            if label and shelf.main_range(entry["product"], strict=True) == label:
                return entry["product"]
        return ""

    def blocking(self, item: str, result: dict[str, Any]) -> list[str]:
        # only evidence against the item stops it; a condition no opened listing states at all cannot be checked in this shop
        shelf = self.ledger.shelf
        kept = []
        for problem in result.get("problems") or []:
            if isinstance(problem, dict):
                need = str(problem.get("condition") or problem.get("requirement") or "")
                kind = str(problem.get("kind") or "contradicted").lower()
                note = str(problem.get("note") or "")
            else:
                need, kind, note = str(problem), "contradicted", ""
            if kind.startswith("not") and not shelf.carried(need) and not (self.table is not None and self.table.carried(need)):
                continue
            if kind.startswith("not") and self.table is not None and self.table.unbacked(need):
                continue
            if kind.startswith("not") and any(_same_property(need, [prop]) and any(name in need.lower() for name in names) for prop, names in self.ledger.looked_up):
                continue
            if need and shelf.literal(item, need):
                continue
            kept.append(f"{need} ({note})" if note else need)
        if not kept and not result.get("problems"):
            kept.append("the check found it unfit")
        return kept


class AuditAdvisor:
    def __init__(self, llm: LLMDesk, ledger: Ledger, clock: Clock, trace: list[dict[str, Any]], query: str):
        self.llm = llm
        self.ledger = ledger
        self.clock = clock
        self.trace = trace
        self.query = query
        self.done = False
        self.fits: list[str] = []

    def review(self) -> str:
        if self.done or self.clock.left() < WRAP_UP_SECONDS + 30:
            return ""
        self.done = True
        picked: list[str] = []
        used = 0
        for record in self.ledger.records[::-1]:
            text = _dump(_shrink(record))
            if used + len(text) <= 16000:
                picked.append(text)
                used += len(text)
        if not picked:
            return ""
        sections = [
            AUDIT_PROMPT,
            "Input - the shopper's request:\n" + self.query,
            "Input - current requirements:\n" + self.ledger.summary(),
            "Input - listing records seen, newest first:\n" + "\n".join(picked),
        ]
        # a refusal is never sent unreviewed, whatever has been spent
        result = self.llm.ask_json("\n\n".join(sections), 3000, role="audit", effort="low")
        self.trace.append({"kind": "audit", "result": result})
        candidates = [str(x) for x in (result or {}).get("possible_fits") or [] if isinstance(x, (str, int))][:5]
        self.fits = candidates
        if not candidates:
            return ""
        why = f" ({result.get('reason')})" if result.get("reason") else ""
        return "An independent check thinks these listings may meet every requirement: " + ", ".join(candidates) + why + ". Verify them (details, stock, price) and buy the right one; decline only if each one truly fails."


class Advisors:
    def __init__(self, setup: SetupAdvisor, interpret: InterpretAdvisor, plan: PlanAdvisor, verify: VerifyAdvisor, audit: AuditAdvisor):
        self.setup = setup
        self.interpret = interpret
        self.plan = plan
        self.verify = verify
        self.audit = audit


NEED_PROMPT = """Role: you rewrite a purchasing agent's conditions as records that code can hold up against a listing's own property table.

Input: the shopper's request, the numbered conditions taken from it so far, what is no longer required or does not matter, for an order of several products their names, and maybe the property keys the shop's opened listings carry.

Rules:
1. Give every condition at least one record, carrying its number. A condition about two properties gets two records under the same number.
2. "property": the attribute the condition is about, in the short plain form a listing's table would use as its key. When the listings' keys are given, use the key under which such values appear there, spelled as they spell it, even when the condition words it differently.
3. "relation", exactly one of: "is" (this value has to be shown), "one_of" (any of several values will do), "is_not" (this value must not be shown), "none_of" (none of these values may be shown), "at_most", "at_least", "between" (numeric limits), "other" (anything else, such as a value to be read off another listing). A comparison word placed before a value that is no number makes it "is" with that value alone.
4. "values": the values as written, each a short phrase; for a numeric limit, the limit as written with its unit.
5. "number": the numeric limit as a plain number ("between": the lower end), "high": the upper end for "between", each null when there is none; "unit": the unit written beside the number, "" when there is none.
6. "product": for an order of several products, the listed product the condition belongs to, copied exactly; "" otherwise.
7. "hard": false only for a mere wish; true for everything else.
8. The list can miss something. A value the request itself demands of the product's own properties, which no numbered condition covers (also one tucked inside the product phrase, such as the occasion, material or packaging it names together with the product), gets a record of its own with "n": 0 and "condition": a short "property: value" line. The price, the budget, stock, the quantity and the product type itself are never records, and nothing listed as no longer required or not mattering comes back.

Output: ONE JSON object and nothing else: {"needs": [{"n": the condition's number or 0, "condition": "", "product": "", "property": "", "relation": "", "values": [], "number": null, "high": null, "unit": "", "hard": true}]}."""

REF_PROMPT = """Role: a purchasing agent has to read one property's value off another listing; you read the listings opened for that lookup.

Input: the shopper's request, the lookup (the property to read, and the range, seller and attributes that single out the listing) and the records of the opened listings by that seller.

Rules:
1. Choose the listing whose own attributes match the lookup's identifying attributes; when none of them matches, leave "listing" empty.
2. When that listing's text says the property is to be taken from another listing, describe that other listing under "next" and leave "value" empty: "seller" is the name of whoever sells it as the listing writes it (never left empty when the listing names one), "range" is its product type as a shop category (never an attribute value such as a size, age or capacity), "identify" its distinguishing attributes.
3. Otherwise give the value the request asks for under "value", and under "quote" the exact words of the record it rests on, copied character for character. Usually that is the listing's own value for the property (its key and value). When the request asks not for that listing's own value but for the one a rule stated on the listing pairs with it, apply the rule to the listing's own value: give the paired value and quote the rule.
4. Never guess: with neither a shown value nor a pointer, leave both empty.

Output: ONE JSON object and nothing else: {"listing": "", "value": "", "quote": "", "next": null or {"range": "", "seller": "", "identify": ""}}."""

CELL_PROMPT = """Role: a purchasing agent keeps a table of listings against conditions; code has filled the clear cells and you settle the open ones.

Input: the shopper's request, the listings' records, and the open cells, each naming one listing and one condition.

Rules:
1. Each listing is judged on its own record only.
2. "pass" when the listing's own attribute for the condition carries the value. Key spellings vary, and singular or plural, close synonyms and one entry of a list all count.
3. "fail" when the listing's own attribute for the condition carries another value, or a number outside the limit once units are converted. An attribute that is only related (a lining for a material, a length for a style) settles nothing.
4. With no attribute for the condition at all, the title, caption or description may settle it; when nothing in the record touches it, answer "not_shown".
5. "quote": the exact words of the record the answer rests on, copied character for character (a key and its value, or a stretch of the title or description); "" for "not_shown".

Output: ONE JSON object and nothing else: {"cells": [{"cell": the cell number, "verdict": "pass" or "fail" or "not_shown", "quote": ""}]}."""


def _norm_word(word: str) -> str:
    return word[:-1] if len(word) > 3 and word.endswith("s") and not word.endswith("ss") else word


def _key_words(text: Any) -> frozenset[str]:
    spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(text))
    return frozenset(_norm_word(word) for word in re.findall(r"[a-z0-9]+", spaced.lower()))


def _value_parts(text: str) -> list[frozenset[str]]:
    # a value field may list several values; each entry stands on its own
    return [_key_words(part) for part in re.split(r"\s*(?:,|/|;|\|| & )\s*", text) if part.strip()]


def _figures(text: str) -> list[tuple[float, str]]:
    return [(float(number), unit.lower()) for number, unit in re.findall(r"(\d+(?:\.\d+)?)\s*([A-Za-z]*)", text)]


def _unit_of(numbers: list[tuple[float, str]]) -> str | None:
    # a span writes its unit once ("10-20 cm"); two different units in one value cannot be compared without converting
    units = {unit for _, unit in numbers if unit}
    return None if len(units) > 1 else (units.pop() if units else "")


def _same_unit(shown: str | None, wanted: str) -> bool:
    # a limit written without a unit takes the listing's own; otherwise the unit must agree, a plural ending aside
    if shown is None:
        return False
    return not wanted or shown == wanted or shown in (wanted + "s", wanted + "es") or wanted in (shown + "s", shown + "es")


def _tokens(text: Any) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(text).lower()))


class Need:
    def __init__(self, text: str, entry: dict[str, Any]):
        self.text = text
        self.product = str(entry.get("product") or "")
        self.prop = _key_words(entry.get("property") or text.partition(":")[0].rpartition(" - ")[2])
        self.relation = str(entry.get("relation") or "other").lower()
        self.values = [str(value) for value in entry.get("values") or [] if str(value).strip()]
        self.low = float(entry["number"]) if _plain_number(entry.get("number")) else None
        self.high = float(entry["high"]) if _plain_number(entry.get("high")) else None
        self.unit = str(entry.get("unit") or "").strip().lower()
        self.hard = entry.get("hard") is not False
        self.extra = entry.get("n") == 0


class VerdictTable:
    # candidate x need = pass / fail / unknown; code fills what the listing's own table decides, one cheap call settles the rest
    def __init__(self, llm: LLMDesk, ledger: Ledger, clock: Clock, trace: list[dict[str, Any]], query: str):
        self.llm = llm
        self.ledger = ledger
        self.clock = clock
        self.trace = trace
        self.query = query
        self.read_for: tuple[str, ...] | None = None
        self.needs: list[Need] = []
        self.settled: dict[tuple[str, str], tuple[str, str]] = {}
        self.contrary: set[tuple[str, str]] = set()
        self.asked: set[tuple[str, str]] = set()

    def conditions(self) -> list[str]:
        book = self.ledger
        if book.current is not None:
            return [entry for entry in book.current if entry not in book.dropped]
        return [entry for entry in book.stated + book.revealed if entry not in book.dropped]

    def current_needs(self) -> list[Need]:
        wanted = tuple(self.conditions())
        book = self.ledger
        keys = self.shop_keys()
        # the reading is redone once the opened listings show their own keys, so the needs take the shop's names
        state = wanted + ("|",) + tuple(book.dropped) + ("|",) + tuple(book.free) + ("|", str(len(keys) >= 6))
        if state == self.read_for:
            return self.needs
        self.read_for = state
        products = [entry["product"] for entry in _basket(self.ledger.plan)]
        sections = [NEED_PROMPT, "Input - the shopper's request:\n" + self.query, "Input - conditions:\n" + ("\n".join(f"{n}. {text}" for n, text in enumerate(wanted, 1)) or "(none listed)")]
        if book.dropped or book.free:
            sections.append("Input - no longer required or not mattering: " + "; ".join(book.dropped + book.free))
        if keys:
            sections.append("Input - property keys on the opened listings, most common first: " + ", ".join(keys))
        if products:
            sections.append("Input - products of the order: " + "; ".join(products))
        answer = self.llm.ask_json("\n\n".join(sections), 4000, [READ_MODEL, JUDGE_MODEL], "needs") if self.clock.left() > WRAP_UP_SECONDS else None
        made: list[Need] = []
        covered: set[int] = set()
        for entry in (answer or {}).get("needs") or []:
            if not isinstance(entry, dict) or not isinstance(entry.get("n"), int) or not 0 <= entry["n"] <= len(wanted):
                continue
            if entry["n"] == 0:
                # a demand of the request the list missed joins unless a listed condition already speaks for its property
                text = str(entry.get("condition") or "").strip()
                if ":" in text and not _identifies(text, book.plan.get("references")) and not any(_key_words(entry.get("property")) == need.prop for need in made) and not _same_property(text, [_need_property(old) for old in wanted + tuple(book.dropped) + tuple(book.free)]):
                    made.append(Need(text, entry))
                continue
            covered.add(entry["n"])
            made.append(Need(wanted[entry["n"] - 1], entry))
        # a condition the reader skipped stays in the table as one only the cell reader can settle
        made += [Need(text, {}) for n, text in enumerate(wanted, 1) if n not in covered]
        # a condition only the reply reader's rebuilt list carries, neither stated in the reading nor revealed by the shopper, is held like one found in the story
        backed = [_need_property(old) for old in book.stated + book.revealed]
        for need in made:
            if need.text in wanted and not _same_property(need.text, backed):
                need.extra = True
        self.needs = made
        self.trace.append({"kind": "needs", "needs": [[need.text, sorted(need.prop), need.relation, need.values, need.low, need.high, need.unit, need.hard] for need in made]})
        return made

    def own_contrary(self, ident: str, need: Need, quote: str) -> bool:
        wanted = set().union(*(_key_words(value) for value in need.values)) if need.values else set()
        for key, part in (self.ledger.shelf.facts.get(ident) or {}).items():
            if _key_words(key) == need.prop and _tokens(part) and _tokens(part) in quote:
                if not (_key_words(part) & wanted) or need.relation in ("at_most", "at_least", "between"):
                    return True
        return False

    def unbacked(self, text: str) -> bool:
        # a story condition no opened listing shows is no ground to hold an item
        return any(need.extra and not self.shown_somewhere(need) for need in self.needs if need.text == text or _same_property(text, [_need_property(need.text)]))

    def shown_somewhere(self, need: Need) -> bool:
        # a demand the reader found in the request's story counts only once the shop shows that value under that key somewhere
        shelf = self.ledger.shelf
        return any(self.judge_value(need, key, str(part))[0] == "pass" for ident in shelf.opened for key, part in (shelf.facts.get(ident) or {}).items() if _key_words(key) == need.prop)

    def carried(self, text: str) -> bool:
        # the shop states this need's property when some opened listing has the key the need reader named for it
        shelf = self.ledger.shelf
        props = [need.prop for need in self.needs if need.prop and (need.text == text or _same_property(text, [_need_property(need.text)]))]
        return any(_key_words(key) == prop for prop in props for ident in shelf.opened for key in shelf.facts.get(ident) or {})

    def shop_keys(self) -> list[str]:
        shelf = self.ledger.shelf
        counts: dict[str, int] = {}
        for ident in shelf.opened:
            for key in shelf.facts.get(ident) or {}:
                if key not in shelf.listing_keys:
                    counts[key] = counts.get(key, 0) + 1
        return sorted(counts, key=lambda key: -counts[key])[:40]

    def needs_for(self, product: str) -> list[Need]:
        # a need code cannot compare (a value to be read off another listing) is the judge's, never a table cell
        needs = [need for need in self.current_needs() if need.hard and need.relation != "other" and need.prop and (not need.extra or self.shown_somewhere(need))]
        if not _basket(self.ledger.plan) or not product:
            return needs
        own = _words(product)
        return [need for need in needs if not need.product or _words(need.product) & own or _words(need.text.partition(" - ")[0]) & own]

    def record_text(self, ident: str) -> str:
        return _tokens(" ".join(self.ledger.records_of(ident)))

    def cell(self, ident: str, need: Need) -> tuple[str, str]:
        if (ident, need.text) in self.settled:
            return self.settled[(ident, need.text)]
        shelf = self.ledger.shelf
        facts = shelf.facts.get(ident) or {}
        figures = shelf.figures.get(ident) or {}
        if need.relation == "other" or not need.prop:
            return "unknown", ""
        exact = [(key, str(part)) for key, part in list(facts.items()) + list(figures.items()) if _key_words(key) == need.prop]
        if not exact:
            # with no key of its own the title or description may still settle it; that reading is the cell reader's
            return "unknown", ""
        verdicts = [self.judge_value(need, key, value) for key, value in exact]
        for verdict in ("fail", "pass"):
            hits = [evidence for kind, evidence in verdicts if kind == verdict]
            # keys naming the same property may disagree; any key showing the value is enough, and a lone contrary key fails only when none shows it
            if verdict == "fail" and any(kind == "pass" for kind, _ in verdicts):
                continue
            if hits:
                return verdict, hits[0]
        return "unknown", ""

    @staticmethod
    def judge_value(need: Need, key: str, value: str) -> tuple[str, str]:
        evidence = f"{key}: {value}"
        parts = _value_parts(value)
        wanted = [_key_words(item) for item in need.values if _key_words(item)]
        shown = [words for words in wanted if any(words <= part for part in parts)]
        if need.relation in ("is", "one_of"):
            if shown:
                return "pass", evidence
            numbers = _figures(value)
            if need.low is not None and numbers and _same_unit(_unit_of(numbers), need.unit):
                return ("pass" if any(number == need.low for number, _ in numbers) else "fail"), evidence
            return "unknown", evidence
        if need.relation in ("is_not", "none_of"):
            return ("fail" if shown else "pass"), evidence
        if need.relation in ("at_most", "at_least", "between") and need.low is not None:
            numbers = _figures(value)
            if not numbers or not _same_unit(_unit_of(numbers), need.unit):
                return "unknown", evidence
            if need.relation == "at_most":
                fits = all(number <= need.low for number, _ in numbers)
            elif need.relation == "at_least":
                fits = all(number >= need.low for number, _ in numbers)
            else:
                fits = need.high is not None and all(need.low <= number <= need.high for number, _ in numbers)
            return ("pass" if fits else "fail"), evidence
        return "unknown", evidence

    def row(self, ident: str, product: str) -> dict[str, list[tuple[Need, str]]]:
        found: dict[str, list[tuple[Need, str]]] = {"pass": [], "fail": [], "unknown": []}
        for need in self.needs_for(product):
            kind, evidence = self.cell(ident, need)
            found[kind].append((need, evidence))
        return found

    def resolve(self, idents: list[str], product: str) -> None:
        # one call for every open cell of these listings; an answer whose quote is not in the record stays open
        if self.clock.left() < WRAP_UP_SECONDS + 20:
            return
        cells: list[tuple[str, Need]] = []
        for ident in idents:
            for need, _ in self.row(ident, product)["unknown"]:
                if (ident, need.text) not in self.asked and len(cells) < 24:
                    cells.append((ident, need))
        if not cells:
            return
        listings = list(dict.fromkeys(ident for ident, _ in cells))
        sections = [CELL_PROMPT, "Input - the shopper's request:\n" + self.query]
        sections.append("Input - records of the listings:\n" + "\n".join(f"{ident}:\n" + "\n".join(self.ledger.records_of(ident)[-2:]) for ident in listings))
        sections.append("Input - open cells:\n" + "\n".join(f"{n}. listing {ident} | condition: {need.text}" for n, (ident, need) in enumerate(cells, 1)))
        # the cheap reader may reason before it answers; a cut reply settles nothing, so it gets room for every cell
        answer = self.llm.ask_json("\n\n".join(sections), 4000, [READ_MODEL, JUDGE_MODEL], "cells")
        shelf = self.ledger.shelf
        for ident, need in cells:
            self.asked.add((ident, need.text))
        for entry in (answer or {}).get("cells") or []:
            if not isinstance(entry, dict) or not isinstance(entry.get("cell"), int) or not 1 <= entry["cell"] <= len(cells):
                continue
            ident, need = cells[entry["cell"] - 1]
            verdict = str(entry.get("verdict") or "").lower()
            quote = _tokens(entry.get("quote") or "")
            if verdict == "not_shown":
                # a condition no opened listing states anywhere cannot be checked in this shop, so it rules nothing out
                self.settled[(ident, need.text)] = ("fail", f"nothing on its record shows {need.text}") if shelf.carried(need.text) or self.carried(need.text) else ("pass", "no listing states it")
            elif verdict == "pass" and need.relation in ("is", "one_of") and need.values and not any(_key_words(value) <= {_norm_word(word) for word in quote.split()} for value in need.values if _key_words(value)):
                # a pass by sense alone, with none of the wanted words in what it quotes, stays open for the judge
                continue
            elif verdict == "fail" and not self.own_contrary(ident, need, quote):
                # a fail must quote the listing's own value under the need's key, and that value must share no wanted word
                continue
            elif verdict in ("pass", "fail") and quote and quote in self.record_text(ident):
                self.settled[(ident, need.text)] = (verdict, str(entry.get("quote"))[:120])
                if verdict == "fail":
                    self.contrary.add((ident, need.text))
        self.trace.append({"kind": "cells", "asked": len(cells), "settled": sum(1 for ident, need in cells if (ident, need.text) in self.settled)})

    def pool(self, product: str, strict: bool) -> list[str]:
        shelf = self.ledger.shelf
        label = shelf.main_range(product, strict=strict) if product else None
        return [ident for ident in shelf.opened if not shelf.sold_out(ident) and (not label or shelf.range_of(ident) == label)]

    def pick(self, product: str, units: int, cap: float | None, strict: bool, allow_open: bool = False, skip: set[str] | None = None) -> str | None:
        # the cheapest listing of the range in stock and within budget that no cell fails (and, unless allowed, none is open)
        shelf = self.ledger.shelf
        ranked = []
        for ident in self.pool(product, strict):
            cost = shelf.price(ident)
            if ident in (skip or set()) or cost is None or (cap is not None and cost * units > cap):
                continue
            found = self.row(ident, product)
            if found["fail"] or (found["unknown"] and not allow_open):
                continue
            ranked.append((len(found["unknown"]), cost, ident))
        return min(ranked)[2] if ranked else None

    def hold_text(self, ident: str, fails: list[tuple[Need, str]]) -> str:
        return f"The requirement table shows {ident} does NOT meet: " + "; ".join(f"{need.text} ({evidence})" for need, evidence in fails[:4]) + ". Choose a listing that meets every requirement."


ORDER_CHECKLIST = "Not run yet (no step used). Pre-order check: (1) the cart holds one line per wanted product (exactly one line when one product is wanted), each at its required quantity, and nothing else; (2) after your last cart change you re-read this item's stock and price with a tool that shows them, and it is in stock and price times quantity is within the current budget; (3) its own properties meet every stated requirement, every need the shopper revealed and every change announced; (4) if the cheapest or best was asked for, no better fitting item exists among everything you could find; (5) every procedural rule in the request has been followed. If anything is unchecked, check it now; otherwise call the order tool again."
DECLINE_CHECKLIST = "Not run yet (no step used). Before declining, be sure: you searched in several ways (different words, filters, pages), you checked every candidate's own properties, and each one fails a stated requirement, a revealed need, the budget or stock. Remember that close synonyms and any key spelling count. If something does fit, buy it instead; if nothing fits, call the decline tool again."
REFUSAL_NOTE = "THE ORDER WAS REFUSED and the episode continues; the cart is unchanged. Apply every change in the refusal, fix the cart (remove any line that is no longer acceptable, such as a sold-out item), re-check stock and price of the new pick, and order again."


ASK_MODEL = "google/gemini-3.8-flash"
ASK_BACKUP_MODEL = "anthropic/claude-haiku-5.5"
SEAT_PARAMS: dict[str, dict[str, Any]] = {
    ASK_MODEL: {"reasoning": {"effort": "minimal"}},
    ASK_BACKUP_MODEL: {},
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
- Indifference, leaving it to the shop, a plea to ask again more simply or one point at a time, or anything else without a wish gives an empty list. A reply that leaves the asked property open states no wish, even when it adds a remark about the customer's day, mood or what they are after in general.
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
        self.pointers: list[tuple[Product, str, str, str]] = []
        self.resolve_pointers = True

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

    def adopt(self, products: list[tuple[str, str, list[str], int]], messages: int, story: str, cheapest: bool, in_stock: bool) -> None:
        for spot, (name, search, stated, quantity) in enumerate(products):
            self.products.append(Product(f"P{spot + 1}", name, search or name, stated, quantity))
        self.messages = messages
        self.story = story
        self.cheapest = cheapest
        self.in_stock = in_stock
        self.trace.append({"kind": "request", "products": [p.name for p in self.products], "messages": self.messages, "story": bool(story)})

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
            product.queue = merged_order(ranked[:FIRST_PICKS] if basket else [], story, [key for key, _, _ in product.menu])
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
            if not told and self.split and key not in product.bare:
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
                found = self.value_from_listing(product, prop, value, str(wish.get("search") or "")) if self.resolve_pointers else None
                if found is None:
                    self.pointers.append((product, prop, reply, message))
                value = found or value
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
        range_of = getattr(self.shop, "range_of", None)
        for ident in listings:
            label = range_of(ident) if callable(range_of) else None
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


CAPS_PROMPT = """Role: you read a customer's request to an online shop for its two limits.

Rules:
1. "message_cap": the most messages the customer lets the agent send them, as an integer, or null when the request sets no such limit.
2. "tool_call_cap": the most tool calls, steps, actions or moves the request allows the agent, as an integer, or null when the request sets no such limit.

Output: ONE JSON object and nothing else: {"message_cap": null, "tool_call_cap": null}"""


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


class SearchNeed:
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
        self.needs = [SearchNeed(r.get("name"), r.get("value")) for r in self.requirements]
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
        self.needs = [SearchNeed(r.get("name"), r.get("value"), n) for r, n in zip(self.requirements, read["needs"])]
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


class AskStop(Exception):
    pass


class AgentShop:
    # the ask engine's shop: every call goes through the driver's own tools, so the ledger and shelf fill as they would for the loop
    def __init__(self, driver: "Driver") -> None:
        self.driver = driver
        self.fresh_labels: list[str] = []
        self.query_field: str | None = None
        self.batch = self.find_batch()
        if not self.batch:
            # without a tool that opens many listings in one step, the engine opens them one by one within its share
            self.view_many = None

    def find_batch(self) -> tuple[str, str, int] | None:
        # a tool whose one required argument is a list of ids; of the mapped tools only a details or view tool can be one
        guard = self.driver.guard
        mapped = {tool for role, tool in self.driver.roles.items() if role not in ("details", "view")}
        for name, schema in guard.schemas.items():
            known = schema.get("properties") or {}
            required = list(schema.get("required") or [])
            if name in mapped or len(required) != 1:
                continue
            spec = known.get(required[0]) or {}
            items = spec.get("items") if isinstance(spec.get("items"), dict) else {}
            if spec.get("type") == "array" and items.get("type") in (None, "string"):
                return name, required[0], int(spec.get("maxItems") or BATCH_LISTINGS)
        return None

    @property
    def steps_left(self) -> int:
        return self.driver.world.steps_left()

    @property
    def categories(self) -> list[str]:
        # the range labels on the rows of the searches since the last reading, which the stage makes once per product after its searches;
        # it picks the label sharing the product name's words, else the range most of its listings carry
        labels, self.fresh_labels = list(dict.fromkeys(self.fresh_labels)), []
        return labels

    def range_of(self, ident: str) -> str | None:
        return self.driver.ledger.shelf.range_of(ident)

    def text_fields(self, name: str) -> list[str]:
        guard = self.driver.guard
        schema = guard.schemas.get(name) or {}
        known = schema.get("properties") or {}
        free_text = [key for key, spec in known.items() if isinstance(spec, dict) and guard.takes_text(spec) and "enum" not in spec]
        # the search words go where the setup reading or an earlier answered search put them, then a required text field, then the rest
        named = [self.query_field, self.driver.plan.get("search_query")] if name == self.driver.role("search") else []
        first = [key for key in named + list(schema.get("required") or []) if isinstance(key, str) and key in free_text]
        return list(dict.fromkeys(first + free_text))

    def text_args(self, name: str, text: str, field: str | None = None) -> dict[str, Any] | None:
        guard = self.driver.guard
        known = (guard.schemas.get(name) or {}).get("properties") or {}
        field = field or next(iter(self.text_fields(name)), None)
        if not field:
            return None
        args = {key: value for key, value in guard.filler_args(name, text).items() if key != field}
        args[field] = text[: (known.get(field) or {}).get("maxLength", len(text))]
        args, issues = guard.check_args(name, args)
        return None if issues else args

    def run(self, name: str | None, args: dict[str, Any] | None) -> Any:
        if not name or args is None:
            raise AskStop()
        if self.driver.world.over() or self.driver.clock.left() < WRAP_UP_SECONDS:
            raise AskStop()
        answer = self.driver.world.call([{"name": name, "args": args}])
        if not answer:
            raise AskStop()
        observation = answer.get("observation")
        self.driver.absorb(name, args, observation)
        return observation

    def search(self, query: str, page: Any = None) -> dict[str, Any]:
        name = self.driver.role("search")
        fields = self.text_fields(name) if name else []
        observation = self.run(name, self.text_args(name, query) if name else None)
        if not self.query_field and len(fields) > 1 and not any(_row_id(found) for found in _nested_dicts(observation)):
            # nothing came back: the words may have gone to a filter field, so the next text field gets them once
            retry = self.run(name, self.text_args(name, query, fields[1]))
            if any(_row_id(found) for found in _nested_dicts(retry)):
                self.query_field, observation = fields[1], retry
        elif any(_row_id(found) for found in _nested_dicts(observation)) and fields:
            self.query_field = self.query_field or fields[0]
        rows: list[dict[str, Any]] = []
        for found in _nested_dicts(observation):
            ident = _row_id(found)
            if ident and ident not in {row["id"] for row in rows}:
                rows.append({**{key: part for key, part in found.items() if not isinstance(part, (dict, list))}, "id": ident})
        totals = [part for part in (observation.values() if isinstance(observation, dict) else []) if _plain_number(part) and part >= len(rows)]
        shelf = self.driver.ledger.shelf
        self.fresh_labels.extend(label for label in (shelf.range_of(row["id"]) for row in rows) if label)
        return {"rows": rows, "count": int(min(totals)) if totals else len(rows), "next": None}

    def detail_tool(self) -> str | None:
        return self.driver.role("details") or self.driver.role("view")

    def view(self, ident: str) -> Any:
        name = self.detail_tool()
        args = self.text_args(name, ident) if name else None
        if args is None and self.batch:
            return self.view_many([ident]).get(ident)
        return self.run(name, args)

    def view_many(self, ids: list[str]) -> dict[str, Any]:
        if not self.batch:
            return {}
        found: dict[str, Any] = {}
        waiting = list(ids)
        while waiting and self.batch:
            name, field, most = self.batch
            chunk = waiting[:most]
            observation = self.run(name, {field: chunk})
            self.driver.ledger.shelf.take(observation, True)
            got = {}
            for part in _nested_dicts(observation):
                ident = _row_id(part)
                if ident in chunk and ident not in got:
                    got[ident] = part
            if got:
                found.update(got)
                waiting = waiting[len(chunk):]
            elif len(chunk) > 1:
                # an empty answer to a long list: the tool takes fewer ids than its schema says, so the size is learned
                self.batch = (name, field, max(1, len(chunk) // 2))
            else:
                # not even one id came back: it is no batch view, so the engine opens listings one by one within its share
                self.batch = None
                self.view_many = None
        return found

    def ask(self, text: str) -> str:
        name = self.driver.role("message")
        observation = self.run(name, self.driver.guard.message_args(name, text) if name else None)
        heard: list[str] = []
        if isinstance(observation, str):
            heard = [observation]
        elif isinstance(observation, dict) and "error" not in observation:
            _short_sentences([part for part in observation.values() if isinstance(part, (str, list))], heard)
        reply = " ".join(heard) if heard else _dump(observation)
        self.driver.ledger.said.append(f"asked: {text} -> shopper: {reply}")
        self.driver.ledger.heard.update(heard)
        return reply

    def chat(self, messages: list[dict[str, Any]], model: str | None = None, temperature: float = 0, max_tokens: int = 700, reasoning: dict[str, Any] | None = None) -> str:
        answer = self.driver.llm.ask(messages, None, [model or ASK_MODEL], max_tokens, "ask", reasoning=reasoning)
        if not answer or not isinstance(answer.get("content"), str):
            raise RuntimeError("no answer")
        return answer["content"]


class ShopClosed(Exception):
    pass


class SearchShop:
    # the search engine's shop for one product: the driver's tools, rows in the engine's fields, and the product's share of steps
    def __init__(self, driver: "Driver", share: int) -> None:
        self.driver = driver
        self.base = AgentShop(driver)
        self.stop = driver.world.steps_left() - share
        if not self.base.batch:
            self.view_many = None

    @property
    def steps_left(self) -> int:
        return max(0, self.driver.world.steps_left() - self.stop)

    def row(self, ident: str, node: dict[str, Any]) -> dict[str, Any]:
        shelf = self.driver.ledger.shelf
        return listing_from_record(ident, node, shelf.price(ident), not shelf.sold_out(ident), shelf.range_of(ident))

    def guard(self, tool: str, *args: Any, **kwargs: Any) -> Any:
        if self.steps_left <= 0:
            raise ShopClosed()
        try:
            return getattr(self.base, tool)(*args, **kwargs)
        except AskStop:
            raise ShopClosed()

    def search(self, query: str, page: Any = None) -> dict[str, Any]:
        found = self.guard("search", query)
        return {"rows": [self.row(row["id"], row) for row in found.get("rows") or []], "count": found.get("count"), "next": None}

    def browse(self, category: str, page: Any = None, max_price: Any = None, brand: Any = None) -> dict[str, Any]:
        driver = self.driver
        name = driver.role("browse")
        named = driver.plan.get("browse_args") if isinstance(driver.plan.get("browse_args"), dict) else {}
        known = driver.guard.props(name)
        fields = {job: field for job, field in named.items() if isinstance(field, str) and field in known}
        if not name or "category" not in fields:
            return {}
        # the category as the shop's own rows spell it
        key = driver.ledger.shelf.range_key()
        spelled = next((row[key].strip() for row in driver.ledger.shelf.rows.values() if key and isinstance(row.get(key), str) and row[key].strip().lower() == str(category).strip().lower()), category)
        args: dict[str, Any] = {fields["category"]: spelled}
        for job, value in (("max_price", max_price), ("brand", brand), ("page", page)):
            if value is not None and job in fields:
                spec = known.get(fields[job]) or {}
                args[fields[job]] = int(value) if job == "max_price" and spec.get("type") == "integer" else value
        args, issues = driver.guard.check_args(name, args)
        if issues:
            return {}
        if self.steps_left <= 0:
            raise ShopClosed()
        try:
            found = self.base.run(name, args)
        except AskStop:
            raise ShopClosed()
        rows = []
        for part in _nested_dicts(found):
            ident = _row_id(part)
            if ident and ident not in {row["id"] for row in rows}:
                rows.append(self.row(ident, part))
        top = found if isinstance(found, dict) else {}
        totals = [part for part in top.values() if _plain_number(part) and part >= len(rows)]
        # the continuation token sits in the result field that shares a word with the paging argument's name
        token_words = set(key_words(fields["page"])) if "page" in fields else set()
        token = next((part for key, part in top.items() if isinstance(part, (str, int)) and not isinstance(part, bool) and token_words & set(key_words(key))), None)
        return {"rows": rows, "count": int(min(totals)) if totals else len(rows), "next": token if rows else None}

    def view(self, ident: str) -> dict[str, Any]:
        found = self.guard("view", ident)
        node = next((part for part in _nested_dicts(found) if _row_id(part) == ident), found if isinstance(found, dict) else {})
        return self.row(ident, node)

    def view_many(self, ids: list[str]) -> dict[str, Any]:
        found = self.guard("view_many", ids)
        return {ident: self.row(ident, node) for ident, node in found.items()}

    def chat(self, messages: list[dict[str, Any]], model: str | None = None, **params: Any) -> str:
        return self.base.chat(messages, model=model, max_tokens=params.get("max_tokens", 900), reasoning=params.get("reasoning"))


class SearchStage:
    # the search between asking and judging: the search engine runs once per product within its share of the steps, seeded with the ask stage's work
    LOOP_MARGIN = 4

    def __init__(self, driver: "Driver") -> None:
        self.driver = driver
        self.candidates: dict[str, list[str]] = {}

    def run(self) -> None:
        driver = self.driver
        engine = driver.ask_engine
        # the engine reaches a shelf's cheap end through a browse tool, which this driver does not map yet
        if not SEARCH_ENGINE or not driver.role("search") or driver.clock.left() < WRAP_UP_SECONDS + 60:
            return
        lines = engine.finish()["lines"] if engine is not None else []
        entries = driver.wanted_lines()
        budget = driver.store_budget()
        for spot, entry in enumerate(entries):
            room = driver.world.steps_left() - driver.commit_reserve() - self.LOOP_MARGIN
            share = room // (len(entries) - spot)
            if share < 2 or driver.clock.left() < WRAP_UP_SECONDS + 40:
                break
            product = entry["product"]
            shelf = driver.product_shelf.get(product) or driver.ledger.shelf.main_range(product, strict=bool(driver.basket)) or product
            record = {"category": shelf, "quantity": entry["quantity"], "budget": budget, "in_stock": driver.plan.get("in_stock_only") is True,
                      "requirements": driver.judge.needs(product), "cheapest": driver.plan.get("priority") == "cheapest"}
            line = lines[spot] if spot < len(lines) else {}
            shop = SearchShop(driver, share)
            seed = {"searches": line.get("searches") or [],
                    "listings": {ident: {**shop.row(ident, driver.judge.record_node(ident)), "opened": ident in driver.ledger.shelf.opened} for ident in line.get("listings") or {}}}
            try:
                found = search_engine({"record": record, "seed": seed}, shop)
            except Exception as problem:
                driver.trace.append({"kind": "search_stage_error", "error": f"{type(problem).__name__}: {problem}"[:300]})
                continue
            self.candidates[product] = [ident for ident in found.get("candidates") or [] if isinstance(ident, str)]
        driver.trace.append({"kind": "search_stage", "candidates": self.candidates, "steps": driver.world.step})
        if self.candidates:
            shelf = driver.ledger.shelf
            notes = ["Candidates found so far by the search stage (a starting point, not a verdict: search further where cheaper or better fits may exist, and order only after every line is verified):"]
            for product, ids in self.candidates.items():
                notes.append(f"- {product}: " + ", ".join(f"{ident} ({shelf.price(ident)}{', opened' if ident in shelf.opened else ''})" for ident in ids[:6]))
            driver.ask_note = (driver.ask_note + "\n" if driver.ask_note else "") + "\n".join(notes)


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


class JudgeStage:
    # the judge of cart lines and per-product picks: the code requirement table first, then the judge engine's code cells and strict check
    def __init__(self, driver: "Driver") -> None:
        self.driver = driver
        self.ask = chain_ask(self.post)
        self.bounds: dict[str, list[dict[str, Any]]] = {}
        self.memo: dict[str, dict[str, Any]] = {}

    def post(self, body: dict[str, Any]) -> tuple[Any, Any]:
        clock = self.driver.clock
        if clock.left() < FORCE_SECONDS + 5:
            return None, None
        _llm.timeout = int(max(8, min(60, clock.left() - FORCE_SECONDS)))
        outcome = _llm.post_verbose("/inference/chat/completions", body)
        if isinstance(outcome.data, dict):
            self.driver.llm._take(outcome.data, "judge")
            return outcome.data, 200
        return None, (outcome.error or {}).get("status") if isinstance(outcome.error, dict) else None

    def needs(self, product: str) -> list[dict[str, str]]:
        book, driver = self.driver.ledger, self.driver
        entries = book.current if book.current is not None else [entry for entry in book.stated + book.revealed if entry not in book.dropped]
        # a pointer condition gives way to the value read off its listing, as in the requirement summary
        entries = [entry for entry in entries if not any(_same_property(entry, [prop]) and any(name in entry.lower() for name in names) for prop, names in book.looked_up)]
        names = [entry["product"] for entry in driver.basket]
        open_sellers = [str(entry.get("seller")) for entry in driver.references() if not entry.get("value") and not entry.get("dropped")]
        found = needs_from_ledger(entries, product if names else "", names, open_sellers)
        needs = [{"name": need["name"].lstrip("- ").strip() or need["name"], "value": need["value"]} for need in found]
        # stated amounts become limits once per product and ledger version, so code compares plain figures
        key = f"{product}|{book.version}"
        if key not in self.bounds:
            self.bounds[key] = add_bounds(self.ask, needs) if self.driver.clock.left() > WRAP_UP_SECONDS else needs
        return self.bounds[key]

    def record_node(self, ident: str) -> dict[str, Any]:
        # a shop can split one listing over several tools (price and stock in one, the property sheet in another): every record of it is merged, the newest value winning
        merged: dict[str, Any] = {}
        for record in self.driver.ledger.records:
            for found in _nested_dicts(record):
                if _row_id(found) == ident:
                    merged.update({key: value for key, value in found.items() if value not in (None, "", [], {})})
        return merged

    def listing(self, ident: str) -> dict[str, Any]:
        shelf = self.driver.ledger.shelf
        return listing_from_record(ident, self.record_node(ident), shelf.price(ident), not shelf.sold_out(ident), shelf.range_of(ident))

    def verdicts(self, product: str, idents: list[str], units: int) -> dict[str, dict[str, Any]]:
        version = self.driver.ledger.version
        fresh = [ident for ident in idents if f"{ident}|{version}" not in self.memo]
        if fresh and self.driver.clock.left() > WRAP_UP_SECONDS:
            judged = judge_listings(self.ask, self.needs(product), [self.listing(ident) for ident in fresh], units)
            for ident in fresh:
                self.memo[f"{ident}|{version}"] = judged.get(ident) or {"fits": None}
        return {ident: self.memo.get(f"{ident}|{version}") or {"fits": None} for ident in idents}

    def units_of(self, product: str) -> int:
        return next((entry["quantity"] for entry in self.driver.wanted_lines() if entry["product"] == product), 1)

    def order_holds(self, lines: list[str]) -> str:
        driver = self.driver
        verify, table = driver.advisors.verify, driver.advisors.verify.table
        version = driver.ledger.version
        groups: dict[str, list[str]] = {}
        for ident in lines[:4]:
            groups.setdefault(driver.line_product(ident) or "", []).append(ident)
        holds, unsure = [], []
        for product, idents in groups.items():
            judged = []
            for ident in idents:
                found = table.row(ident, product) if table is not None else {"fail": []}
                firm = [(need, evidence) for need, evidence in found["fail"] if (ident, need.text) not in table.settled or (ident, need.text) in table.contrary] if table is not None else []
                if firm:
                    verify.passed.pop(ident, None)
                    holds.append(table.hold_text(ident, firm))
                else:
                    judged.append(ident)
            for ident, verdict in self.verdicts(product, judged, self.units_of(product)).items():
                if verdict.get("fits") is True:
                    verify.passed[ident] = None
                    verify.memo[f"{ident}|{version}"] = ""
                elif verdict.get("fits") is False and verdict.get("binding"):
                    verify.passed.pop(ident, None)
                    text = f"The judge holds {ident}: it misses {verdict.get('why') or 'a need'} - this is the listing's own fact, so choose another listing."
                    verify.memo[f"{ident}|{version}"] = text
                    holds.append(text)
                else:
                    # a failure the judge could not quote from the listing goes to the line check, which reads the whole listing, instead of holding the order
                    unsure.append(ident)
        if unsure:
            holds.append(verify.check_lines(unsure, "ORDER"))
        return " ".join(hold for hold in holds if hold)

    def pick(self, product: str, units: int, skip: set[str]) -> str | None:
        # the shortlist: the cheapest in-stock opened listings of the product's range
        driver, shelf = self.driver, self.driver.ledger.shelf
        label = driver.product_shelf.get(product) or shelf.main_range(product, strict=bool(driver.basket))
        pool = [ident for ident in shelf.opened if ident not in skip and not shelf.sold_out(ident) and (not label or shelf.range_of(ident) == label) and shelf.price(ident) is not None]
        found = [ident for ident in driver.search_stage.candidates.get(product, []) if ident in pool]
        pool = (found + sorted((ident for ident in pool if ident not in found), key=lambda ident: shelf.price(ident) or 0))[:4]
        if not pool:
            return None
        judged = self.verdicts(product, pool, units)
        return choose_listing(judged, [self.listing(ident) for ident in pool], units, driver.plan.get("priority") == "cheapest")


class Driver:
    def __init__(self, clock: Clock, trace: list[dict[str, Any]], llm: LLMDesk, world: WorldDesk, guard: SchemaGuard, ledger: Ledger, advisors: Advisors):
        self.clock = clock
        self.trace = trace
        self.llm = llm
        self.world = world
        self.guard = guard
        self.ledger = ledger
        self.advisors = advisors
        self.plan: dict[str, Any] = {}
        self.roles: dict[str, str] = {}
        self.message_budget: int | None = None
        self.messages_used = 0
        self.order_attempts = 0
        self.refusals = 0
        self.order_checked = False
        self.firm_holds: set[str] = set()
        self.read_at: dict[str, int] = {}
        self.carted_at: dict[str, int] = {}
        self.reread_asked: set[str] = set()
        self.detail_hold = False
        self.sheet_seen: dict[str, list[bool]] = {}
        self.search_numbers: list[float] = []
        self.price_note = False
        self.decline_checked = False
        self.cart_used = False
        self.planner_on = True
        self.used_needles: set[str] = set()
        self.used_phrases: list[frozenset[str]] = []
        self.search_rows: list[tuple[int, list[str]]] = []
        self.dry_search = False
        self.compacted = 0
        self.paging_noted = False
        self.asked: list[str] = []
        self.qty_warned: set[str] = set()
        self.cart_checked: set[str] = set()
        self.recent: list[str] = []
        self.pending_reply = ""
        self.messages: list[dict[str, Any]] = []
        self.price_holds = 0
        self.decline_holds = 0
        self.searches = 0
        self.search_holds = 0
        self.searched_budget: Any = None
        self.lane_asks: dict[int, int] = {}
        self.asking = False
        self.table_swaps: set[int] = set()
        self.table_tried = False
        self.budget_hold = False
        self.basket: list[dict[str, Any]] = []
        self.ask_stage_on = False
        self.ask_note = ""
        self.ask_engine: AskEngine | None = None
        self.product_shelf: dict[str, str | None] = {}
        self.search_stage = SearchStage(self)
        self.judge = JudgeStage(self)
        self.gate_holds = 0
        self.cheaper_named: set[str] = set()

    @property
    def finished(self) -> bool:
        return self.world.finished

    @property
    def dead(self) -> bool:
        return self.world.dead

    def role(self, name: str) -> str | None:
        return self.roles.get(name)

    def message_limit(self) -> int:
        return 5 if self.message_budget is None else self.message_budget

    def finishers(self) -> set[str]:
        return {self.role("order"), self.role("decline")} - {None}

    # reading of the request

    def adopt_plan(self, plan: dict[str, Any]) -> None:
        self.plan = plan
        self.ledger.plan = plan
        named = plan.get("roles") if isinstance(plan.get("roles"), dict) else {}
        self.roles = {role: tool for role, tool in named.items() if isinstance(tool, str) and self.guard.knows(tool)}
        allowance = plan.get("message_cap")
        if _plain_number(allowance) and allowance >= 0:
            self.message_budget = int(allowance)
        self.ledger.stated = [str(entry) for entry in plan.get("requirements") or [] if entry]
        self.basket = _basket(plan)
        if self.basket and not self.ledger.stated:
            self.ledger.stated = [f"{entry['product']} - {need}" for entry in self.basket for need in entry["requirements"]]
        if not self.ledger.stated:
            # a single product may carry its conditions only in its own entry
            self.ledger.stated = [str(need) for entry in plan.get("products") or [] if isinstance(entry, dict) for need in entry.get("requirements") or [] if need]
        self.ledger.stated = [need for need in self.ledger.stated if not _identifies(need, plan.get("references"))]
        span = str(plan.get("quantity_span") or "").strip().lower()
        if self.quantity() > 1 and (not span or span not in self.world.query.lower()):
            plan["quantity"] = 1
        self.ledger.budget_now = plan.get("price_cap")
        moves = plan.get("tool_call_cap")
        if _plain_number(moves) and 0 < moves < self.world.max_steps:
            self.world.max_steps = int(moves)

    def read_caps(self, plan: dict[str, Any]) -> None:
        def usable(value: Any) -> bool:
            return _plain_number(value) and value > 0

        if usable(plan.get("message_cap")) and usable(plan.get("tool_call_cap")):
            return
        caps = self.llm.ask_json(CAPS_PROMPT + "\n\nRequest:\n" + self.world.query, 1500, [ASK_MODEL, MAIN_MODEL], "caps") or {}
        for key in ("message_cap", "tool_call_cap"):
            if not usable(plan.get(key)) and usable(caps.get(key)):
                plan[key] = int(caps[key])

    def ask_stage(self) -> None:
        if not self.role("message") or not self.role("search") or self.messages_used >= self.message_limit():
            return
        entries = self.basket or [{"product": str(self.plan.get("product") or "item"), "quantity": self.quantity(), "requirements": list(self.ledger.stated)}]
        readings = [entry for entry in self.plan.get("products") or [] if isinstance(entry, dict)]
        products = []
        for spot, entry in enumerate(entries):
            search = str(readings[spot].get("search") or "") if spot < len(readings) else ""
            stated = [need.partition(":")[0].rpartition(" - ")[2].strip() + ":" + need.partition(":")[2] for need in entry["requirements"] if ":" in need]
            products.append((entry["product"], search or entry["product"], stated, entry["quantity"]))
        engine = AskEngine({"request": self.world.query}, AgentShop(self))
        # a pointer's listing can point on to another listing; the driver's lookup follows those hops, so the stage only hands it over
        engine.resolve_pointers = False
        engine.adopt(products, self.message_limit() - self.messages_used, str(self.plan.get("story") or ""), self.plan.get("priority") == "cheapest", self.plan.get("in_stock_only") is True)
        self.ask_stage_on = True
        try:
            engine.look()
            self.join_shared_shelves(engine)
            engine.rank()
            engine.converse()
        except AskStop:
            engine.trace.append({"kind": "stopped"})
        except Exception as problem:
            engine.trace.append({"kind": "error", "error": f"{type(problem).__name__}: {problem}"[:300]})
        finally:
            self.ask_stage_on = False
        self.ask_engine = engine
        self.trace.append({"kind": "ask_stage", "steps": self.world.step, "sent": engine.sent, "trace": engine.trace})
        self.take_asks(engine)

    def join_shared_shelves(self, engine: AskEngine) -> None:
        # two products found on one shop range whose name holds both their names are one product read as two
        for first in list(engine.products):
            for second in list(engine.products):
                if first is second or first not in engine.products or second not in engine.products or not first.shelf or first.shelf != second.shelf:
                    continue
                if not ((_content_words(first.name) | _content_words(second.name)) <= _content_words(first.shelf)):
                    continue
                engine.products.remove(second)
                first.stated += [need for need in second.stated if need not in first.stated]
                first.listings.update(second.listings)
                first.rows.update(second.rows)
                first.searches += second.searches
                first.quantity = max(first.quantity, second.quantity)
                self.trace.append({"kind": "joined_products", "kept": first.name, "joined": second.name, "shelf": first.shelf})
                old_names = (first.name, second.name)
                first.name = first.shelf
                self.basket = [entry for entry in self.basket if entry["product"] not in old_names[1:]]
                for entry in self.basket:
                    if entry["product"] == old_names[0]:
                        entry["product"] = first.name
                        entry["quantity"] = first.quantity
                self.ledger.stated = [need.replace(old_names[1] + " - ", first.name + " - ", 1).replace(old_names[0] + " - ", first.name + " - ", 1) for need in self.ledger.stated]
        for spot, product in enumerate(engine.products):
            product.label = f"P{spot + 1}"
        if not any(step.get("kind") == "joined_products" for step in self.trace):
            return
        if len(self.basket) < 2 and len(engine.products) == 1:
            # one product left: an ordinary request, its conditions without the product prefix
            product = engine.products[0]
            self.basket = []
            self.plan["product"] = product.name
            self.plan["quantity"] = product.quantity
            self.ledger.stated = list(dict.fromkeys(need.split(" - ", 1)[1] if need.startswith(product.name + " - ") else need for need in self.ledger.stated))
        # the plan the readers rebuild the requirements from follows the joined products
        self.plan["products"] = [{"product": entry["product"], "quantity": entry["quantity"], "requirements": [need for need in self.ledger.stated if not self.basket or need.startswith(entry["product"] + " - ")]} for entry in (self.basket or [{"product": self.plan["product"], "quantity": self.plan.get("quantity") or 1}])]
        self.plan["requirements"] = list(self.ledger.stated)

    def settled_shelf(self, product: Product) -> str | None:
        shelf = self.ledger.shelf
        labels = [label for label in (shelf.range_of(ident) for ident in product.rows) if label]
        if not labels:
            return product.shelf
        wanted = _content_words(product.name)
        named = [label for label in dict.fromkeys(labels) if wanted & _content_words(label)]
        if named:
            return max(named, key=lambda label: len(wanted & _content_words(label)) / max(1, len(wanted | _content_words(label))))
        label = max(set(labels), key=labels.count)
        return label if labels.count(label) * 2 > len(labels) else product.shelf

    def take_asks(self, engine: AskEngine) -> None:
        # told values join the revealed conditions; a pointer the stage could not read goes to the lookup machinery with the shopper's own words
        pointed = {(id(product), prop) for product, prop, _, _ in engine.pointers}
        several = len(engine.products) > 1
        for owner, prop, value, _ in engine.told_lines:
            if (id(owner), prop) in pointed:
                continue
            need = f"{owner.name} - {prop}: {value}" if several else f"{prop}: {value}"
            if need not in self.ledger.revealed:
                self.ledger.revealed.append(need)
                self.ledger.version += 1
        for _, _, reply, question in engine.pointers:
            self.ledger.heard.discard(reply)
            before = len(self.plan.get("references") or [])
            self.advisors.interpret.read([reply], question)
            for entry in (self.plan.get("references") or [])[before:]:
                if isinstance(entry, dict):
                    entry.setdefault("said", reply)
        # the shop's range field is read again from every row seen by now; a range fixed while one product's rows alone were known can name a seller
        self.product_shelf = {product.name: self.settled_shelf(product) for product in engine.products}
        self.asked.extend(step["message"] for step in engine.trace if step.get("kind") == "ask")
        # the planner stays only as a fallback for a stage that never reached the shopper
        self.planner_on = engine.sent == 0
        self.ask_note = self.ask_summary(engine)

    def ask_summary(self, engine: AskEngine) -> str:
        shelf = self.ledger.shelf
        lines = [f"ALREADY DONE before this conversation ({self.world.step} steps used):"]
        for product in engine.products:
            queries = ", ".join(repr(entry["query"]) for entry in product.searches) or "none"
            lines.append(f"- {product.name} (shop range {product.shelf or 'unknown'}): searched {queries}")
            for ident in list(product.rows)[:8]:
                row = {key: part for key, part in shelf.rows.get(ident, {}).items() if isinstance(part, (str, int, float)) and len(str(part)) <= 80}
                extra = {key: part for key, part in (shelf.facts.get(ident) or {}).items() if key not in row}
                state = "opened, properties: " + _dump(_trim(extra, 40))[:300] if ident in shelf.opened else "not opened"
                lines.append(f"  - {ident}: {_dump(row)[:160]} [{state}]")
        for step in engine.trace:
            if step.get("kind") == "ask":
                lines.append(f"- asked the shopper: {step['message']} -> {step['reply']}")
        lines.append(f"Messages to the shopper used: {self.messages_used} of {self.message_limit()}. The shopper's answers are already in the requirements; do not ask again.")
        return "\n".join(lines)

    def first_reading(self) -> str:
        plan = self.plan
        if not plan:
            return ""
        lines = ["A first reading of the request (check it against the request itself):"]
        if plan.get("product"):
            lines.append(f"- product: {plan.get('product')}")
        if self.basket:
            lines.append("- ONE ORDER OF SEVERAL PRODUCTS: the cart must end with one line per product below, each at its own quantity, and the budget covers the whole cart")
            for entry in self.basket:
                lines.append(f"- {entry['quantity']} x {entry['product']}" + (": " + "; ".join(entry["requirements"]) if entry["requirements"] else ""))
        else:
            lines.append(f"- quantity: {plan.get('quantity') or 1}")
        lines.extend(f"- stated requirement: {entry}" for entry in plan.get("requirements") or [])
        if plan.get("price_cap") is not None:
            lines.append(f"- price cap (total): {plan.get('price_cap')} {plan.get('currency') or ''}".rstrip())
        if plan.get("in_stock_only"):
            lines.append("- only items in stock / on hand")
        goal = plan.get("priority")
        if goal not in ("cheapest", "best"):
            lines.append("- no cheapest/best rule was found: any item that meets every requirement within the budget is right, so do not spend steps hunting for the lowest price (unless the request itself says otherwise)")
        else:
            detail = f" ({plan.get('priority_detail')})" if plan.get("priority_detail") else ""
            lines.append(f"- choose the {goal}{detail} among all fitting items")
        for entry in self.references():
            lines.append(f"- LOOK UP FIRST: the {entry.get('property')} for the {entry.get('product')} is set by the {entry.get('range')} listing sold by {entry.get('seller')} with {entry.get('identify')}; search that range, open the listings of that seller, take the one that matches, read its {entry.get('property')}, then require it" + (f" (allowed difference: {entry.get('tolerance')})" if entry.get("tolerance") else ""))
        lines.extend(f"- nice to have, NOT required (only a tie-breaker): {entry}" for entry in plan.get("soft_wants") or [])
        lines.extend(f"- must be learned from the shopper first: {entry}" for entry in plan.get("ask_first") or [])
        lines.extend(f"- rule: {entry}" for entry in plan.get("rules") or [])
        if self.message_budget is not None:
            lines.append(f"- messages to the shopper allowed: {self.message_budget}")
        lines.append(f"- steps allowed: {self.world.max_steps}")
        hints = [entry for entry in plan.get("hint_properties") or [] if isinstance(entry, str)]
        if hints:
            lines.append("- properties the story may hint at (ask about these first, one per message): " + ", ".join(hints))
        return "\n".join(lines)

    def status(self) -> str:
        left = self.world.steps_left()
        pieces = [f"[status] steps used {self.world.step} of {self.world.max_steps}, {left} left"]
        if self.message_budget is None:
            pieces.append(f"messages sent {self.messages_used}")
        else:
            pieces.append(f"messages sent {self.messages_used} of {self.message_budget}")
        pieces.append(f"about {int(max(0, self.clock.left() - FORCE_SECONDS))} s left")
        text = "; ".join(pieces) + "."
        text += "\n[requirements now] " + self.ledger.summary() + self.step_plan()
        if self.planner_on and self.role("message") and left > 14 and self.messages_used < self.message_limit():
            gaps = self.open_gap()
            if gaps:
                text += "\n[before the questions] the shopper's questions go out once enough listings are opened; open the details of these first (a tool that takes several ids at once saves steps): " + "; ".join(f"{product} ({opened} opened): {', '.join(ids)}" for product, opened, ids in gaps)
        if left <= 2:
            return text + " LAST STEPS: order now if the cart holds a verified fit, otherwise decline."
        if self.llm.cost >= SPEND_FINISH_USD:
            return text + " FINISH NOW, the task's LLM allowance is nearly spent: keep only the verified line(s) in the cart, re-check stock, then order (or decline if nothing fits)."
        if left <= 5 or self.clock.left() < WRAP_UP_SECONDS:
            return text + " FINISH NOW: fix the cart to one verified line, then order right away (or decline if nothing fits); keep the remaining steps for repairing a refused order."
        if self.order_attempts == 0 and left <= 14:
            return text + " Steps are running low: within the next few steps put your best verified fit in the cart and order; searching and reading details end when 7 steps remain, and a refused order needs about 4 steps to repair."
        return text

    def reading_open(self) -> bool:
        # whether a tool that opens a listing is still offered at this step
        offered = {tool["function"]["name"] for tool in self.offered_tools()}
        return bool({self.role("details"), self.role("view")} & offered - {None})

    def offered_tools(self) -> list[dict[str, Any]]:
        left = self.world.steps_left()
        order, decline, cart = self.role("order"), self.role("decline"), self.role("cart")
        narrow: set[str | None] | None = None
        if left <= 1:
            narrow = {order, decline}
        elif left <= 2 and self.order_attempts == 0 and self.ledger.cart_ids:
            # the first checkout is often refused with a change; it goes out while one step remains to send it again
            narrow = {order, decline}
        elif left <= 3:
            narrow = {order, decline, cart}
        elif left <= self.commit_reserve() and (self.order_attempts == 0 or self.refusals > 0):
            # the last steps stay for the order and one repair of it (a line can sell out at checkout), not for rereading
            narrow = {order, decline, cart} | ({self.role("cart_view"), self.role("view")} if left > 5 else set())
        tools = self.guard.tools
        if narrow is not None and order and cart:
            kept = [tool for tool in self.guard.tools if tool["function"]["name"] in narrow]
            if kept:
                tools = kept
        messenger = self.role("message")
        if self.planner_on and messenger:
            tools = [tool for tool in tools if tool["function"]["name"] != messenger]
        return tools

    # conversation helpers

    def post_tool_replies(self, replies: list[tuple[str, str]], asked_labels: list[tuple[str, str]] | None = None) -> None:
        last = len(replies) - 1
        for position, (call_id, text) in enumerate(replies):
            if position == last:
                text += "\n" + self.status()
            self.messages.append({"role": "tool", "tool_call_id": call_id, "content": text})
            if asked_labels is not None and position < len(asked_labels):
                prop, question = asked_labels[position]
                heard = text.split("\n[status]")[0][:400]
                self.asked.append(f"- {prop}: asked \"{question}\" -> {heard}")

    def message_call(self, call_id: str, args: dict[str, Any]) -> dict[str, Any]:
        return {"id": call_id, "type": "function", "function": {"name": self.role("message"), "arguments": json.dumps(args)}}

    def ask_planned(self, force: bool = False) -> bool:
        messenger = self.role("message")
        limit = self.message_limit()
        if not messenger or self.messages_used >= limit:
            self.planner_on = False
            return False
        products = [str(entry["product"]) for entry in self.basket] if self.basket else None
        menu, rich = self.ledger.property_menu(products)
        urgent = bool(self.plan.get("ask_first")) and not self.asked
        # a hidden wish is a property whose values differ across the range, which shows only once enough of the range is opened
        late = self.world.step >= min(16, int(self.world.max_steps * 0.35))
        if not (force or self.asking or urgent or late or (rich >= 4 and not self.open_gap())):
            return False
        urgent = urgent and not force
        # once a forced round has begun, the rest of the allowance follows it round after round
        self.asking = self.asking or force
        if self.world.steps_left() <= 10 or self.clock.left() < WRAP_UP_SECONDS + 40:
            self.planner_on = False
            return False
        count = limit - self.messages_used
        if not menu and not urgent:
            # nothing opened yet: keep the questions for later rather than asking blind
            return False
        planned = self.advisors.plan.questions(count + len(products or []), self.asked, menu)
        planned = self.grounded(planned, menu) + self.uncovered(planned, products)
        # one question per round: each answer reaches the next plan, and an unstated allowance is learnt from the shop's own refusal
        planned = self.spread(planned, products)
        if products:
            planned.sort(key=lambda entry: self.lane_asks.get(self.lane(entry, products), 0))
        planned = planned[:1]
        calls: list[dict[str, Any]] = []
        labels: list[tuple[str, str]] = []
        for position, entry in enumerate(planned):
            wording = self.plain_question(entry)
            args = self.guard.message_args(messenger, wording)
            if not args:
                continue
            args, issues = self.guard.check_args(messenger, args)
            if issues:
                continue
            calls.append(self.message_call(f"ask_{self.messages_used + position + 1}_{self.world.step}", args))
            labels.append((str(entry.get("property")), wording))
            if products:
                lane = self.lane(entry, products)
                self.lane_asks[lane] = self.lane_asks.get(lane, 0) + 1
        if not calls:
            self.planner_on = False
            return False
        topics = ", then ".join(prop for prop, _ in labels)
        self.messages.append({"role": "assistant", "content": "I will ask the shopper about " + topics + ".", "tool_calls": calls})
        self.post_tool_replies(self.run_calls(calls, planned=True), labels)
        if self.messages_used >= self.message_limit():
            self.planner_on = False
        return True

    @staticmethod
    def grounded(planned: list[dict[str, Any]], menu: str) -> list[dict[str, Any]]:
        # an attribute question must name an attribute the menu carries; the shopper only answers about a listed property
        labels = [line[2:].split(" (in ")[0] for line in menu.splitlines() if line.startswith("- ")]
        kept = []
        for entry in planned:
            name = re.sub(r"[_.]+", " ", str(entry.get("property") or "")).strip().lower()
            if str(entry.get("kind") or "attribute") == "must_learn" or not labels or name in labels:
                kept.append(entry)
        return kept or planned

    def uncovered(self, planned: list[dict[str, Any]], products: list[str] | None) -> list[dict[str, Any]]:
        # a product the plan left without any question gets the best-splitting attribute of its own menu
        if not products:
            return []
        named = [_words(entry.get("product")) for entry in planned]
        asked = " ".join(self.asked).lower()
        extra = []
        for product in products:
            if any(words & _words(product) for words in named):
                continue
            menu, _ = self.ledger.single_menu(product)
            for line in menu.splitlines():
                label = line[2:].split(" (in ")[0]
                values = line.split("listings, ")[-1].split(" different")[0]
                if values.isdigit() and int(values) >= 2 and label not in asked and not ({w for w in _words(label) if len(w) > 2} & _words(self.ledger.summary())):
                    extra.append({"property": label, "kind": "attribute", "product": product, "question": ""})
                    break
        return extra

    @staticmethod
    def lane(entry: dict[str, Any], products: list[str]) -> int:
        words = _words(entry.get("product"))
        if not words or str(entry.get("kind") or "attribute") == "must_learn":
            return -1
        return max(range(len(products)), key=lambda spot: len(words & _words(products[spot])))

    def spread(self, planned: list[dict[str, Any]], products: list[str] | None) -> list[dict[str, Any]]:
        # one order of several products: take the questions in turn per product, so each product's hidden wish gets a chance
        if not products:
            return planned
        lanes: dict[int, list[dict[str, Any]]] = {}
        for entry in planned:
            lanes.setdefault(self.lane(entry, products), []).append(entry)
        order = lanes.pop(-1, [])
        while any(lanes.values()):
            for lane in sorted(lanes):
                if lanes[lane]:
                    order.append(lanes[lane].pop(0))
        return order

    def open_gap(self) -> list[tuple[str, int, list[str]]]:
        # per wanted product: how many listings of its range are opened, and the unopened in-stock ones still to open
        shelf = self.ledger.shelf
        products = [str(entry["product"]) for entry in self.basket] if self.basket else [str(self.plan.get("product") or "")]
        target = OPEN_BEFORE_ASK if not self.basket else max(4, OPEN_BEFORE_ASK - 3)
        gaps = []
        for product in products:
            label = shelf.main_range(product, strict=bool(self.basket))
            if not label:
                continue
            opened = sum(1 for ident in shelf.opened if shelf.range_of(ident) == label)
            waiting = [ident for ident in shelf.rows if ident not in shelf.opened and shelf.range_of(ident) == label and not shelf.sold_out(ident)]
            if opened < min(target, opened + len(waiting)):
                gaps.append((product, opened, waiting[:target - opened]))
        return gaps

    def needle_hint(self) -> str:
        # a search that brought no new listing: every word must match and few rows show, so point at single words
        # from the wanted values of the lines not yet in the cart that no search has used so far
        shelf, book = self.ledger.shelf, self.ledger
        wanted = [str(entry["product"]) for entry in self.basket] or [str(self.plan.get("product") or "")]
        held = {shelf.range_of(ident) for ident in book.cart_ids}
        needs = book.current if book.current is not None else [need for need in book.stated + book.revealed if need not in book.dropped]
        picks: list[str] = []
        pairs: list[str] = []
        for product in wanted:
            label = shelf.main_range(product, strict=bool(self.basket))
            if self.basket and (not label or label in held):
                continue
            own = _words(product)
            for need in needs:
                head, _, rest = str(need).partition(" - ")
                if rest and ":" in rest and ":" not in head:
                    if self.basket and shelf.main_range(head, strict=True) != label:
                        continue
                    need = rest
                _, values = _split_need(need)
                ranked = [word for word in sorted(values, key=len, reverse=True) if len(word) >= 4 and not word.isdigit() and word not in own]
                fresh = [word for word in ranked if word not in self.used_needles and word not in picks]
                if fresh:
                    picks.append(fresh[0])
                # a filter may ignore its words, so a value word counts as tried only once it went out with the product's own name alone
                head_word = (re.findall(r"[a-z0-9]+", product.lower()) or [""])[-1]
                for word in ranked[:1]:
                    phrase = f"{word} {head_word}".strip()
                    if frozenset(_words(phrase)) not in self.used_phrases and phrase not in pairs:
                        pairs.append(phrase)
        if not picks and not pairs:
            return ""
        text = "That search added no new listing. Here every search word must match a listing, few rows show, and a filter may ignore its words."
        if pairs:
            text += " In the plain word search, send one wanted value word with the product's name and nothing else; not tried yet: " + ", ".join(f"'{phrase}'" for phrase in pairs[:4]) + "."
        if picks:
            text += " Single value words not searched yet: " + ", ".join(picks[:4]) + "."
        return text

    def plain_question(self, entry: dict[str, Any]) -> str:
        # an attribute question is built in code: one attribute, the shop's own name for it, and the product only in a several-product order
        wording = str(entry.get("question")).strip()
        if str(entry.get("kind") or "attribute") == "must_learn":
            return wording
        label = re.sub(r"[_.]+", " ", str(entry.get("property") or "")).strip()
        if not label:
            return wording
        product = str(entry.get("product") or "").strip() if self.basket else ""
        return f"What {label} would you like for the {product}?" if product else f"What {label} would you like?"

    def send_reply(self, text: str) -> bool:
        messenger = self.role("message")
        if not messenger or self.messages_used >= self.message_limit() or self.world.steps_left() <= 4 or self.clock.left() < WRAP_UP_SECONDS:
            return False
        args = self.guard.message_args(messenger, text)
        if not args:
            return False
        args, issues = self.guard.check_args(messenger, args)
        if issues:
            return False
        call = self.message_call(f"reply_{self.world.step}", args)
        self.messages.append({"role": "assistant", "content": "I will answer the shopper's question.", "tool_calls": [call]})
        self.post_tool_replies(self.run_calls([call], planned=True))
        return True

    # call screening

    def quantity_warning(self, name: str, args: dict[str, Any]) -> str:
        if self.basket:
            return ""
        wanted = self.plan.get("quantity")
        if isinstance(wanted, bool) or not isinstance(wanted, int) or wanted < 1:
            return ""
        known = self.guard.props(name)
        signature = self.guard.enum_signature(name, args)
        if self.ledger.add_signature is not None and signature != self.ledger.add_signature:
            return ""
        for key, spec in known.items():
            if isinstance(spec, dict) and spec.get("type") == "integer" and "enum" not in spec:
                asked_for = args.get(key, spec.get("default", 1))
                if isinstance(asked_for, int) and asked_for != wanted:
                    return f"the request needs a quantity of {wanted} of one listing, but this call asks for {asked_for}."
        return ""

    def cart_gate(self, name: str, args: dict[str, Any]) -> str:
        known = self.guard.props(name)
        signature = self.guard.enum_signature(name, args)
        if self.guard.idle_count(name, args, self.ledger.adds_units):
            return "A count of 0 changes nothing with this cart tool: it adds the units sent to the line, and a negative count takes units out."
        if self.guard.zero_count(name, args, self.ledger.adds_units):
            return ""
        if self.ledger.add_signature is None or signature == self.ledger.add_signature:
            target = next((v for k, v in args.items() if isinstance(v, str) and "enum" not in (known.get(k) or {}) and v.strip() and " " not in v.strip()), None)
            if target and f"{target}|{self.ledger.version}" not in self.cart_checked:
                self.cart_checked.add(f"{target}|{self.ledger.version}")
                # taking a listing out and putting it back with nothing learned in between only spends steps
                if self.ledger.taken_out.get(target) == self.ledger.version and self.world.steps_left() <= 12:
                    return f"You took {target} out of the cart earlier and nothing new is known since; switching back and forth spends the steps the order needs. Keep the current cart and order it, or choose a different listing."
                return self.advisors.verify.check(target, "ADD TO THE CART")
        return ""

    def screen(self, call: dict[str, Any], allowed: set[str], planned: bool, count: int, queued_messages: int) -> tuple[str | None, Any]:
        function = call.get("function") or {}
        name = function.get("name")
        raw = function.get("arguments")
        messenger, cart = self.role("message"), self.role("cart")
        finishing = self.finishers()
        if not self.guard.knows(name):
            return f"Not run (no step used): '{name}' is not one of the tools. Use one of: {', '.join(sorted(allowed))}.", None
        if not planned and name not in allowed:
            if name == messenger and self.planner_on:
                reason = "questions to the shopper are planned and sent for you after you have opened a few listings' details"
            else:
                reason = "too few steps are left for this; only these tools can be used now: " + ", ".join(sorted(allowed))
            return f"Not run (no step used): {reason}.", None
        if isinstance(raw, dict):
            args = raw
        elif (raw or "").strip():
            args = _decode_object(raw or "{}")
            if args is None:
                return "Not run (no step used): the arguments were not a valid JSON object.", None
        else:
            args = {}
        args, issues = self.guard.check_args(name, args)
        if issues:
            return "Not run (no step used): invalid arguments: " + "; ".join(issues[:6]), None
        if name in {self.role("cart"), self.role("details"), self.role("view")} - {None}:
            args = self.plain_ids(name, args)
        if name in finishing and count > 1:
            return "Not run (no step used): an order or a decline must be sent alone, after you have seen the results of your other calls.", None
        if name == messenger and self.message_budget is not None and self.messages_used + queued_messages >= self.message_budget:
            return "Not run (no step used): the message budget is used up. Do not message the shopper again.", None
        signature = name + _dump(args)
        if signature in self.recent and name not in finishing and name not in (messenger, cart):
            return "Not run (no step used): you just made this exact call and nothing has changed since; its result is above. Change the arguments or do something else.", None
        if name == cart:
            if self.world.steps_left() > 3:
                verdict = self.cart_gate(name, args)
                if verdict:
                    return "Not run (no step used). " + verdict + " If you are certain it fits, send the same call again.", None
            warning = self.quantity_warning(name, args)
            if warning and signature not in self.qty_warned:
                self.qty_warned.add(signature)
                return "Not run (no step used): " + warning + " If you are sure, send the same call again.", None
        return None, (name, args)

    def intent(self, name: str, args: dict[str, Any]) -> str | None:
        # one tool may both buy and refuse; its own fixed-choice value says which, read first by the setup's finish values,
        # then by which job's wording that value fits
        order, decline = self.role("order"), self.role("decline")
        if order != decline:
            return "order" if name == order else "decline" if name == decline else None
        if name != order:
            return None
        votes = []
        for key, value in args.items():
            if not (self.guard.props(name).get(key) or {}).get("enum"):
                continue
            jobs = {job: self.guard.pick_enum(name, key, text, self.finish_values(job)) for job, text in (("order", ORDER_WORDING), ("decline", DECLINE_WORDING))}
            if jobs["order"] != jobs["decline"]:
                votes += [job for job, option in jobs.items() if option == value]
        return votes[0] if len(set(votes)) == 1 else None

    def complete_finish(self, name: str, args: dict[str, Any], job: str) -> dict[str, Any]:
        # a merged finish sent without its fixed choice gets the value for the job, even when the schema leaves it optional
        if self.role("order") != self.role("decline"):
            return args
        filled = dict(args)
        for key, spec in self.guard.props(name).items():
            if isinstance(spec, dict) and spec.get("enum") and key not in filled:
                filled[key] = self.guard.pick_enum(name, key, ORDER_WORDING if job == "order" else DECLINE_WORDING, self.finish_values(job))
        return filled

    def gate_finish(self, name: str, args: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
        job = self.intent(name, args) or ("order" if self.ledger.cart_ids else "decline")
        args = self.complete_finish(name, args, job)
        buying = job == "order"
        # a lookup still open at the finish is settled or dropped before the cart is judged
        self.follow_references()
        if buying:
            gate = self.basket_gate()
            if gate:
                return gate, args
            stale = [line for line in self.ledger.cart_ids if self.carted_at.get(line, -1) > self.read_at.get(line, -1) and line not in self.reread_asked]
            details = self.role("details")
            if stale and details and self.reading_open() and self.world.steps_left() > len(stale) + 2:
                # the shop can change a listing after it was read; the line is read again once before it is ordered
                self.reread_asked.update(stale)
                return f"Not run yet (no step used for the order). The cart changed after these lines were last read, and the shop may have changed them since: open {', '.join(stale)} with {details} again, check them against every requirement, then order.", args
            if not self.order_checked and self.world.steps_left() > 2 and self.clock.left() > WRAP_UP_SECONDS:
                self.order_checked = True
                verdict = self.judge.order_holds(self.ledger.cart_ids)
                if verdict:
                    return "Not run yet (no step used). " + verdict, args
                swapped = self.table_swap()
                if swapped:
                    return swapped, args
                if self.world.steps_left() > 5:
                    return ORDER_CHECKLIST, args
            elif self.world.steps_left() > 8 and self.clock.left() > WRAP_UP_SECONDS + 30:
                # a line the check still holds gets one more hold while there is room to replace it; an order sent over a hold lost every time
                verify = self.advisors.verify
                held = [line for line in self.ledger.cart_ids[:4] if verify.memo.get(f"{line}|{self.ledger.version}") and line not in self.firm_holds]
                if held:
                    self.firm_holds.update(held)
                    shelf = self.ledger.shelf
                    unread = [ident for line in held if shelf.range_of(line) for _, ident in shelf.unopened(shelf.range_of(line), self.ledger.budget_now, 1 if self.basket else self.quantity())[:4]] if self.reading_open() else []
                    reasons = " ".join(verify.memo[f"{line}|{self.ledger.version}"] for line in held)
                    return "Not run yet (no step used). The check still holds the cart: " + reasons + " Replace the held line with a listing of the same range that shows the value" + (": open these unopened ones first: " + ", ".join(dict.fromkeys(unread)) if unread else "") + ". Order the held line only if none fits.", args
            hint = self.cheaper_left()
            return (("Not run yet (no step used). " + hint) if hint else None), args
        bought = self.table_buy()
        if bought:
            return bought, args
        # adding one line and ordering take two steps; a fit found late is still bought
        if self.decline_checked and self.world.steps_left() >= 2:
            bought = self.buy_instead()
            if bought:
                return bought, args
        budget = self.ledger.budget_now
        if not self.budget_hold and self.searches and _plain_number(budget) and budget != self.searched_budget and self.world.steps_left() > 8:
            # the shop was searched under an older budget; listings that fit only the new one were never shown
            self.budget_hold = True
            return f"Not run yet (no step used). The budget is now {budget}, and no search has run under it. Search again with it as the price bound (and the other wanted values), open the new candidates and judge them before refusing.", args
        if self.searches < DECLINE_SEARCHES and self.world.steps_left() > 12 and self.search_holds < 2:
            # a refusal needs the shop searched in several ways first; few searches mean most listings were never shown
            self.search_holds += 1
            return f"Not run yet (no step used). Only {self.searches} searches so far. Before refusing, search in other ways: single words from each wanted value, the category filter, price bounds and further pages; open the new candidates and judge them.", args
        if not self.decline_checked and self.world.steps_left() > 6 and self.clock.left() > WRAP_UP_SECONDS + 30:
            self.decline_checked = True
            audit = self.advisors.audit.review()
            return ("Not run yet (no step used). " + audit) if audit else DECLINE_CHECKLIST, args
        bought = self.soft_buy()
        if bought:
            return bought, args
        proof = self.decline_gap()
        if proof:
            return "Not run yet (no step used). " + proof, args
        closest = self.closest_buy()
        if closest:
            return closest, args
        kept = self.order_sound_cart()
        if kept:
            return kept, args
        return None, self.scrub_decline(args, name)

    def order_sound_cart(self) -> str:
        # a refusal almost never scores in these worlds: a cart that already holds one in-stock line per product, at its quantity and within the budget, is ordered instead
        book = self.ledger
        if not book.cart_ids or self.world.steps_left() < 1 or self.clock.left() < FORCE_SECONDS:
            return ""
        problems, _, entries, _ = self.basket_problems()
        if problems or any(book.cart_units.get(ident) is None for ident in book.cart_ids) and len(entries) > 1:
            return ""
        lines = [(ident, book.cart_units.get(ident) or next((entry["quantity"] for entry in entries if entry["product"] == self.line_product(ident)), 1)) for ident in book.cart_ids]
        self.trace.append({"kind": "order_sound_cart", "lines": lines})
        return self.order_lines(lines, "The cart already held one in-stock line per wanted product within the budget, so it was ordered instead of a refusal")

    def passed_fits(self, product: str = "") -> list[str]:
        # listings the item check passed that are still in stock and within the budget, best first
        shelf, verify = self.ledger.shelf, self.advisors.verify
        label = shelf.main_range(product, strict=True) if product else None
        cap = self.ledger.budget_now if _plain_number(self.ledger.budget_now) else None
        fits = []
        for ident in reversed(list(verify.passed)):
            if shelf.sold_out(ident) or (label and shelf.range_of(ident) != label):
                continue
            price = shelf.price(ident)
            if not self.basket and cap is not None and price is not None and price * self.quantity() > cap:
                continue
            if verify.memo.get(f"{ident}|{self.ledger.version}"):
                continue
            fits.append(ident)
        if self.plan.get("priority") == "cheapest":
            fits.sort(key=lambda ident: shelf.price(ident) if shelf.price(ident) is not None else float("inf"))
        return fits

    def plausible(self, product: str) -> str | None:
        # for a basket line nothing proved: the cheapest opened listing of its range that no check has held against
        shelf, verify = self.ledger.shelf, self.advisors.verify
        label = shelf.main_range(product, strict=True)
        pool = [ident for ident in shelf.opened if label and shelf.range_of(ident) == label and not shelf.sold_out(ident) and not verify.memo.get(f"{ident}|{self.ledger.version}")]
        pool.sort(key=lambda ident: shelf.price(ident) if shelf.price(ident) is not None else float("inf"))
        return pool[0] if pool else None

    def table_swap(self) -> str:
        # the cheapest listing every cell passes replaces a dearer cart line, once per state of the requirements
        table = self.advisors.verify.table
        if table is None or not TABLE_SWAP or self.basket or self.plan.get("priority") == "best" or self.world.steps_left() < 4 or self.clock.left() < WRAP_UP_SECONDS + 20:
            return ""
        if self.plan.get("soft_wants") and self.plan.get("priority") != "cheapest":
            return ""
        shelf, verify = self.ledger.shelf, self.advisors.verify
        line = self.ledger.cart_ids[-1] if self.ledger.cart_ids else None
        cost = shelf.price(line) if line else None
        if cost is None or self.ledger.version in self.table_swaps:
            return ""
        self.table_swaps.add(self.ledger.version)
        product = str(self.plan.get("product") or "")
        cap = self.ledger.budget_now if _plain_number(self.ledger.budget_now) else None
        cheaper = [ident for ident in table.pool(product, False) if ident != line and (shelf.price(ident) or cost) < cost]
        table.resolve([ident for ident in cheaper if table.row(ident, product)["unknown"]][:6], product)
        pick = table.pick(product, self.quantity(), cap, False, skip={line})
        if not pick or (shelf.price(pick) or cost) >= cost:
            return ""
        if verify.check(pick, "ORDER") or pick not in verify.passed:
            return ""
        self.trace.append({"kind": "table_swap", "from": line, "to": pick})
        return self.order_lines([(pick, self.quantity())], "The requirement table found a cheaper listing that meets every requirement, and the judge confirmed it, so the cart was switched to it and ordered")

    def table_buy(self) -> str:
        # a decline needs every candidate failed by its own record; the cheapest listing no cell fails is judged and bought instead
        table = self.advisors.verify.table
        if table is None or self.table_tried or self.world.steps_left() < 4 or self.clock.left() < WRAP_UP_SECONDS + 20:
            return ""
        self.table_tried = True
        shelf, verify = self.ledger.shelf, self.advisors.verify
        entries = self.basket or [{"product": str(self.plan.get("product") or ""), "quantity": self.quantity()}]
        cap = self.ledger.budget_now if _plain_number(self.ledger.budget_now) else None
        lines: list[tuple[str, int]] = []
        for entry in entries:
            product, strict = entry["product"], bool(self.basket)
            pool = table.pool(product, strict)
            table.resolve([ident for ident in pool if table.row(ident, product)["unknown"]][:8], product)
            skip: set[str] = set()
            chosen = None
            for _ in range(2):
                pick = table.pick(product, entry["quantity"], None if self.basket else cap, strict, allow_open=True, skip=skip)
                if not pick:
                    break
                if not verify.check(pick, "ORDER") and pick in verify.passed:
                    chosen = pick
                    break
                skip.add(pick)
            if not chosen:
                return ""
            lines.append((chosen, entry["quantity"]))
        if cap is not None and sum((shelf.price(ident) or 0) * units for ident, units in lines) > cap:
            return ""
        self.trace.append({"kind": "table_buy", "lines": lines})
        return self.order_lines(lines, "Every check passed for this listing, so the decline became its purchase")

    def soft_buy(self) -> str:
        # a refusal scores nothing; a listing the judge held only because some condition is not shown on it is the better bet
        if self.basket or self.world.steps_left() < 2 or self.clock.left() < WRAP_UP_SECONDS + 10:
            return ""
        shelf, verify = self.ledger.shelf, self.advisors.verify
        cap = self.ledger.budget_now if _plain_number(self.ledger.budget_now) else None
        units = self.quantity()
        pool = []
        label = shelf.main_range(str(self.plan.get("product") or ""), strict=True)
        for ident, count in verify.soft.items():
            price = shelf.price(ident)
            if shelf.sold_out(ident) or (cap is not None and price is not None and price * units > cap) or (label and shelf.range_of(ident) != label):
                continue
            if verify.table is not None and verify.table.row(ident, verify.product_of(ident))["fail"]:
                continue
            pool.append((count, price if price is not None else float("inf"), ident))
        if not pool:
            return ""
        pick = min(pool)[2]
        self.trace.append({"kind": "soft_buy", "item": pick})
        return self.order_lines([(pick, units)], "Every candidate was held only on a condition its record does not show, none on a contrary value, so the closest one was bought instead of a refusal")

    def buy_instead(self) -> str:
        # a decline repeated while a checked fit is on hand is a purchase of that fit, carted and ordered in code
        if self.basket:
            lines = []
            proven = 0
            held = self.ledger.cart_ids
            for entry in self.basket:
                fits = self.passed_fits(entry["product"])
                label = self.ledger.shelf.main_range(entry["product"], strict=True)
                # a line already in the cart that no check holds against stands for its product as well as a passed fit
                carted = [ident for ident in held if label and self.ledger.shelf.range_of(ident) == label and not self.advisors.verify.memo.get(f"{ident}|{self.ledger.version}")]
                # the refusal audit's own picks for this product's range stand in when nothing was carted or passed
                audited = [ident for ident in self.advisors.audit.fits if label and self.ledger.shelf.range_of(ident) == label and not self.ledger.shelf.sold_out(ident) and not self.advisors.verify.memo.get(f"{ident}|{self.ledger.version}")]
                pick = fits[0] if fits else carted[-1] if carted else audited[0] if audited else self.plausible(entry["product"])
                proven += bool(fits or carted or audited)
                if pick:
                    lines.append((pick, entry["quantity"]))
            if not proven:
                return ""
        else:
            fits = self.passed_fits()
            if not fits:
                # one check of the refusal audit's first pick, the judgement an order needs anyway
                audited = [ident for ident in self.advisors.audit.fits if self.ledger.records_of(ident) and not self.ledger.shelf.sold_out(ident)][:1]
                for ident in audited:
                    self.advisors.verify.check(ident, "ORDER")
                fits = self.passed_fits()
            if not fits:
                return ""
            lines = [(fits[0], self.quantity())]
        return self.order_lines(lines, "A checked fit was on hand, so the decline became this purchase")

    def order_lines(self, lines: list[tuple[str, int]], why: str) -> str:
        cart, order = self.role("cart"), self.role("order")
        if not cart or not order:
            return ""
        chosen = [ident for ident, _ in lines]
        drops = [self.guard.removal_args(cart, ident, self.cart_values("remove"), self.ledger.cart_units.get(ident, 1), self.ledger.adds_units) for ident in self.ledger.cart_ids if ident not in chosen]
        adds = [self.guard.add_args(cart, ident, units, self.cart_values("add")) for ident, units in lines if ident not in self.ledger.cart_ids]
        actions = [{"name": cart, "args": args} for args in drops + adds]
        if None in drops + adds or self.world.steps_left() < len(actions) + 1:
            return ""
        if actions:
            answer = self.world.call(actions)
            self.trace.append({"kind": "buy_instead", "actions": actions, "reply": answer})
            if not answer:
                return ""
            observations = answer.get("observations") if isinstance(answer.get("observations"), list) else [answer.get("observation")]
            for action, observation in zip(actions, observations):
                self.absorb(cart, action["args"], observation)
        finish = self.complete_finish(order, self.guard.filler_args(order, "Order the cart.", self.finish_values("order"), True, ORDER_WORDING), "order")
        answer = self.world.call([{"name": order, "args": finish}])
        self.trace.append({"kind": "buy_instead", "actions": [{"name": order, "args": finish}], "reply": answer})
        if answer and isinstance(answer.get("observation"), dict):
            self.absorb(order, finish, answer["observation"])
        return f"{why}: {', '.join(chosen)}. " + ("The order went through." if self.world.over() else "The order did not go through; read the result above and fix the cart.")

    def clear_stray(self) -> str:
        # take the extra lines out in code, by the cart tool's own removal call, while the steps allow it
        cart = self.role("cart")
        wanted = max(1, len(self.basket))
        extra = self.ledger.cart_ids[:-wanted]
        actions = [{"name": cart, "args": args} for args in (self.guard.removal_args(cart, ident, self.cart_values("remove"), self.ledger.cart_units.get(ident, 1), self.ledger.adds_units) for ident in extra) if cart and args]
        if not actions or len(actions) < len(extra) or self.world.steps_left() < len(actions) + 3:
            return ""
        answer = self.world.call(actions)
        self.trace.append({"kind": "clear_stray", "actions": actions, "reply": answer})
        if not answer:
            return ""
        observations = answer.get("observations") if isinstance(answer.get("observations"), list) else [answer.get("observation")]
        for action, observation in zip(actions, observations):
            self.absorb(cart, action["args"], observation)
        left = self.ledger.cart_ids
        return f"Extra cart lines {', '.join(extra)} were taken out; the cart now holds {', '.join(left) or 'nothing'}. Re-check it and order again."

    def wanted_lines(self) -> list[dict[str, Any]]:
        return self.basket or [{"product": str(self.plan.get("product") or ""), "quantity": self.quantity()}]

    def store_budget(self) -> Any:
        budget = self.ledger.budget_now
        shelf = self.ledger.shelf
        prices = sorted(price for price in (shelf.price(ident) for ident in shelf.rows) if price is not None)
        if not _plain_number(budget) or not prices:
            return budget if _plain_number(budget) else None
        # a cap written in whole currency against a shop whose prices are whole cents; a budget changed later carries no unit of its own
        unit = self.plan.get("price_cap_unit") if budget == self.plan.get("price_cap") else None
        if unit == "cents":
            return budget
        whole = unit in ("whole", None)
        cents = all(float(price).is_integer() for price in prices)
        return budget * 100 if whole and cents and prices[len(prices) // 2] > budget * 5 else budget

    def commit_reserve(self) -> int:
        # one cart step per product, a cart view, the order, and room for a refused order
        return max(7, len(self.wanted_lines()) + 5)

    def line_product(self, ident: str) -> str | None:
        entries = self.wanted_lines()
        shelf = self.ledger.shelf
        label = shelf.range_of(ident)
        if len(entries) == 1:
            # one product: a line from another range than the one its name picks out belongs to no wanted product
            wanted = self.product_shelf.get(entries[0]["product"]) or shelf.main_range(entries[0]["product"], strict=True)
            return entries[0]["product"] if not (label and wanted) or label == wanted else None
        for entry in entries:
            if label and (self.product_shelf.get(entry["product"]) or shelf.main_range(entry["product"], strict=True)) == label:
                return entry["product"]
        return None

    def basket_gate(self) -> str:
        # before any order, in code: one line per product at its quantity and nothing else, every line in stock, the whole total within the budget
        problems, lines, entries, cap = self.basket_problems()
        if not problems:
            return self.cheaper_hold(lines)
        self.trace.append({"kind": "basket_gate", "problems": problems})
        fixed = self.basket_repair(entries, lines, cap)
        if fixed:
            return fixed
        if self.world.steps_left() <= 3 or self.clock.left() < WRAP_UP_SECONDS or self.gate_holds >= 4:
            return ""
        self.gate_holds += 1
        return "Not run yet (no step used for the order). The basket check found: " + "; ".join(problems) + ". The order needs exactly one in-stock line per product at its quantity, nothing else, within the whole budget. Fix the cart, then order."

    def basket_problems(self) -> tuple[list[str], dict[str, list[str]], list[dict[str, Any]], float | None]:
        shelf, book = self.ledger.shelf, self.ledger
        entries = self.wanted_lines()
        cap = self.store_budget()
        lines: dict[str, list[str]] = {entry["product"]: [] for entry in entries}
        stray = []
        for ident in book.cart_ids:
            owner = self.line_product(ident)
            if owner in lines:
                lines[owner].append(ident)
            else:
                stray.append(ident)
        problems = []
        for entry in entries:
            held = lines[entry["product"]]
            if not held:
                problems.append(f"no line for the {entry['product']}")
            elif len(held) > 1:
                problems.append(f"{len(held)} lines for the {entry['product']} ({', '.join(held)}), one wanted")
            for ident in held:
                units = book.cart_units.get(ident)
                if units is not None and units != entry["quantity"]:
                    problems.append(f"{ident} holds {units} units, {entry['quantity']} wanted")
        if stray:
            problems.append(f"lines that belong to no wanted product: {', '.join(stray)}")
        gone = [ident for ident in book.cart_ids if shelf.sold_out(ident)]
        if gone:
            problems.append(f"out of stock: {', '.join(gone)}")
        wanted = {entry["product"]: entry["quantity"] for entry in entries}
        total = sum((shelf.price(ident) or 0) * book.cart_units.get(ident, wanted.get(self.line_product(ident) or "", 1)) for ident in book.cart_ids)
        if cap is not None and total > cap:
            problems.append(f"the cart totals {total:g}, over the budget of {cap:g}")
        return problems, lines, entries, cap

    def basket_repair(self, entries: list[dict[str, Any]], lines: dict[str, list[str]], cap: float | None) -> str:
        # each product keeps a sound line it already has, else takes the judge's pick; the cheapest picks replace lines when the total is over
        shelf, verify = self.ledger.shelf, self.advisors.verify
        chosen: list[tuple[str, int]] = []
        for entry in entries:
            kept = [ident for ident in lines[entry["product"]] if not shelf.sold_out(ident) and not verify.memo.get(f"{ident}|{self.ledger.version}")]
            pick = kept[-1] if kept else self.judge.pick(entry["product"], entry["quantity"], set())
            if not pick:
                return ""
            chosen.append((pick, entry["quantity"]))
        if cap is not None and sum((shelf.price(ident) or 0) * units for ident, units in chosen) > cap:
            cheapest = [(self.judge.pick(entry["product"], entry["quantity"], set()), entry["quantity"]) for entry in entries]
            if any(ident is None for ident, _ in cheapest) or sum((shelf.price(ident) or 0) * units for ident, units in cheapest) > cap:
                return ""
            chosen = [(str(ident), units) for ident, units in cheapest]
        if any(ident in self.ledger.cart_units and self.ledger.cart_units[ident] != units for ident, units in chosen):
            # a cart tool may add units rather than set them, so a wrong count is left to the loop to fix
            return ""
        fresh = [ident for ident, _ in chosen if ident not in self.ledger.cart_ids]
        if fresh and self.judge.order_holds(fresh):
            return ""
        self.trace.append({"kind": "basket_repair", "lines": chosen})
        return self.order_lines(chosen, "The basket check fixed the cart to one line per product")

    def cheaper_hold(self, lines: dict[str, list[str]]) -> str:
        # for the cheapest, a line with a cheaper listing the table fully passes is named once
        if self.plan.get("priority") != "cheapest" or self.world.steps_left() <= self.commit_reserve():
            return ""
        shelf = self.ledger.shelf
        notes = []
        for entry in self.wanted_lines():
            for ident in lines[entry["product"]]:
                pick = self.judge.pick(entry["product"], entry["quantity"], {ident})
                if pick and pick not in self.cheaper_named and (shelf.price(pick) or float("inf")) < (shelf.price(ident) or 0):
                    self.cheaper_named.add(pick)
                    notes.append(f"{pick} is cheaper than {ident} for the {entry['product']} and meets every requirement in the table")
        return ("Not run yet (no step used for the order). " + "; ".join(notes) + ". Check it and swap it in if it fits, then order.") if notes else ""

    def closest_buy(self) -> str:
        # a refusal almost never scores: unless every reachable candidate of some product fails on its own quoted fact, the closest fit is bought
        # it needs only the cart and order calls, and late in the task the judge reuses its earlier verdicts, so it runs until the forced finish
        if self.world.steps_left() < len(self.wanted_lines()) + 1 or self.clock.left() < FORCE_SECONDS:
            return ""
        shelf = self.ledger.shelf
        cap = self.store_budget()
        lines: list[tuple[str, int]] = []
        for entry in self.wanted_lines():
            product, units = entry["product"], entry["quantity"]
            label = self.product_shelf.get(product) or shelf.main_range(product, strict=bool(self.basket))
            pool = [ident for ident in shelf.opened if not shelf.sold_out(ident) and (not label or shelf.range_of(ident) == label) and shelf.price(ident) is not None]
            pool = sorted(pool, key=lambda ident: shelf.price(ident) or 0)[:6]
            judged = self.judge.verdicts(product, pool, units)
            open_ones = [ident for ident in pool if judged[ident].get("fits") is True] or [ident for ident in pool if not judged[ident].get("binding")]
            if not open_ones:
                return ""
            lines.append((max(open_ones, key=lambda ident: (judged[ident].get("fits") is True, judged[ident].get("passed") or 0, -(shelf.price(ident) or 0))), units))
        if cap is not None and sum((shelf.price(ident) or 0) * units for ident, units in lines) > cap:
            return ""
        self.trace.append({"kind": "closest_buy", "lines": lines})
        return self.order_lines(lines, "No listing was ruled out by its own facts, so the closest fit was bought instead of a refusal")

    def learn_stock(self) -> None:
        # which field tells availability, and which of its values mean sold out, read once from the rows seen
        shelf = self.ledger.shelf
        rows = [row for row in shelf.rows.values() if row][:6]
        if not rows or shelf.stock_field or self.clock.left() < WRAP_UP_SECONDS + 40:
            return
        sample = "\n".join(_dump({key: part for key, part in row.items() if len(str(part)) <= 80}) for row in rows)
        answer = self.llm.ask_json(STOCK_PROMPT + "\n\nRows:\n" + sample, 1500, [ASK_MODEL, MAIN_MODEL], "stock") or {}
        field = answer.get("field")
        if isinstance(field, str) and any(field in row for row in rows):
            shelf.stock_field = field
            shelf.stock_kind = str(answer.get("kind") or "")
            # a flag's one sold-out value may come back bare instead of in a list
            marks = answer.get("unavailable")
            marks = marks if isinstance(marks, list) else [] if marks is None else [marks]
            shelf.stock_out = [mark for mark in marks if isinstance(mark, (str, bool))]
            if len(shelf.rows) >= 4 and sum(shelf.sold_out(ident) for ident in shelf.rows) >= 0.8 * len(shelf.rows):
                # a reading that empties the shelf is a misreading
                shelf.stock_field = None
        self.trace.append({"kind": "stock", "answer": answer, "field": shelf.stock_field, "sold_out_marks": list(shelf.stock_out)})

    def step_plan(self) -> str:
        entries = self.wanted_lines()
        room = self.world.steps_left() - self.commit_reserve()
        if len(entries) < 2 or room <= 0:
            return ""
        return f"\n[step plan] about {room // len(entries)} steps per product for searching and checking, then {self.commit_reserve()} steps stay for the cart and the order."

    def stray_lines(self) -> str:
        # more cart lines than wanted products: the older ones go, by the cart tool's own removal call
        wanted = max(1, len(self.basket))
        lines = self.ledger.cart_ids
        if len(lines) <= wanted:
            return ""
        extra = lines[:-wanted]
        cart = self.role("cart")
        calls = [f"{cart} {json.dumps(args)}" for args in (self.guard.removal_args(cart, ident, self.cart_values("remove"), self.ledger.cart_units.get(ident, 1), self.ledger.adds_units) for ident in extra) if cart and args]
        how = (" Remove them with: " + "; ".join(calls) + ".") if calls else " Remove them with the cart tool."
        return f"The cart holds {len(lines)} lines but the order wants {wanted}: {', '.join(lines)}. Only the chosen item{'s' if wanted > 1 else ''} may stay; {', '.join(extra)} must go first.{how}"

    def plain_ids(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        # an id argument sent with a title around it is the one known id inside it
        known = set(self.ledger.shelf.rows) | self.ledger.seen_ids
        fixed = dict(args)
        for key, value in args.items():
            spec = self.guard.props(name).get(key) or {}
            if isinstance(value, str) and value not in known and "enum" not in spec:
                inside = {token for token in re.split(r"[\s,;()\[\]\"'`]+", value) if token in known}
                if len(inside) == 1:
                    fixed[key] = inside.pop()
        return fixed

    def references(self) -> list[dict[str, Any]]:
        return [entry for entry in self.plan.get("references") or [] if isinstance(entry, dict) and str(entry.get("seller") or "").strip()]

    def follow_references(self) -> str:
        # a value read off another listing is followed in code: the lookup listing is read, a pointer it carries moves the lookup on, and the value found becomes a stated condition
        shelf, book = self.ledger.shelf, self.ledger
        notes = [note for note in [self.cap_references()] if note]
        for entry in self.references():
            if entry.get("value") or entry.get("dropped"):
                continue
            if self.clock.left() < WRAP_UP_SECONDS + 30:
                continue
            seller = str(entry.get("seller")).strip().lower()
            found = [ident for ident in shelf.opened if any(isinstance(part, str) and part.strip().lower() == seller for part in (shelf.facts.get(ident) or {}).values())]
            if not found or set(found) <= set(entry.get("read") or []):
                continue
            entry["read"] = found
            sections = [
                REF_PROMPT,
                "Input - the shopper's request:\n" + self.world.query,
                f"Input - the lookup: the {entry.get('property')} of the {entry.get('range')} listing sold by {entry.get('seller')} with {entry.get('identify')}" + (f"\nThe shopper's own words for it: {entry['said']}" if entry.get("said") else ""),
                "Input - records of the opened listings by that seller:\n" + "\n".join(f"{ident}:\n" + "\n".join(book.records_of(ident)[-1:]) for ident in found[:6]),
            ]
            answer = self.llm.ask_json("\n\n".join(sections), 3000, [READ_MODEL, JUDGE_MODEL], "reference") or {}
            ident = str(answer.get("listing") or "").strip()
            if ident not in found:
                continue
            record = _tokens(" ".join(book.records_of(ident)))
            for told in entry.pop("with_needs", []):
                # a need told with the lookup that this listing shows itself, key and value, was the listing's description
                value = _tokens(told.partition(":")[2])
                if value and f" {value} " in f" {record} " and _need_property(told) <= set(record.split()):
                    entry.setdefault("identified", []).append(told)
                    book.revealed[:] = [item for item in book.revealed if item != told]
                    if book.current is not None:
                        book.current = [item for item in book.current if not (_same_property(item, [_need_property(told)]) and _tokens(item.partition(":")[2]) == value)]
            onward = answer.get("next") if isinstance(answer.get("next"), dict) else None
            value = str(answer.get("value") or "").strip()
            quote = _tokens(answer.get("quote") or "")
            names = [str(entry.get("seller")).strip().lower()]
            if onward and _tokens(onward.get("seller")) and _tokens(onward.get("seller")) in record:
                # a pointer counts only when the listing itself names the next seller
                entry.setdefault("hops", []).append(names[0])
                # every range the lookup passed through stays a lookup range, so its listings never stand in for the product
                entry.setdefault("ranges", []).append(str(entry.get("range") or ""))
                # the attributes that found each listing on the way stay known, so none of them turns into a condition on the item
                entry.setdefault("identified", []).append(entry.get("identify"))
                entry.update(range=str(onward.get("range") or entry.get("range")), seller=str(onward.get("seller")).strip(), identify=onward.get("identify") or "", read=[])
                self.trace.append({"kind": "reference_hop", "from": ident, "to": entry["seller"]})
                notes.append(f"Listing {ident} says its {entry['property']} is to be taken from the {entry['range']} listing sold by {entry['seller']} with {entry['identify']}; search that range, open that seller's listings and read the {entry['property']} there.")
            elif value and quote and quote in record and _tokens(value) in quote:
                # the value decides the order, so the judge LLM reads the chosen listing once more with the request before it stands
                check = self.llm.ask_json("\n\n".join(sections[:3] + ["Input - records of the opened listings by that seller:\n" + f"{ident}:\n" + "\n".join(book.records_of(ident)[-1:])]), 3000, [SETUP_BACKUP, JUDGE_MODEL], "reference", "low") or {}
                second, said_quote = str(check.get("value") or "").strip(), _tokens(check.get("quote") or "")
                if second and said_quote and said_quote in record and _tokens(second) in said_quote:
                    value = second
                    answer["quote"] = check.get("quote")
                entry["value"] = value
                relation = str(entry.get("tolerance") or "").lower()
                said = f"not {value}" if "differ" in relation or relation.startswith("not") else value
                if relation and not any(word in relation for word in ("same", "match", "differ", "not", "equal")):
                    said += f" (allowed difference: {entry.get('tolerance')})"
                text = f"{str(entry.get('property')).strip()}: {said}"
                if self.basket and entry.get("product"):
                    text = f"{entry['product']} - {text}"
                book.stated.append(text)
                if book.current is not None:
                    book.current.append(text)
                book.looked_up.append((_need_property(str(entry.get("property"))), names + list(entry.get("hops") or [])))
                # the shopper's pointer marked this property as theirs to set; the value read for it now stands as a stated condition
                book.loose[:] = [prop for prop in book.loose if prop != _need_property(text)]
                self.trace.append({"kind": "reference_value", "from": ident, "condition": text})
                notes.append(f"Lookup done: from listing {ident} (\"{str(answer.get('quote') or '')[:160]}\") the item must have {entry['property']} {said}; search for and judge by that value, not by other values on that listing. This settles the condition that pointed at that listing.")
        return " ".join(notes)

    def cap_references(self) -> str:
        # a lookup gets its product's share of the steps left when it opened; past that, or past the last free steps, its condition stays unchecked instead of holding every candidate
        notes = []
        for entry in self.references():
            if entry.get("value") or entry.get("dropped"):
                continue
            if "opened_at" not in entry:
                entry["opened_at"] = self.world.step
                entry["share"] = max(4, (self.world.steps_left() - self.commit_reserve()) // max(1, len(self.wanted_lines())))
            spent = self.world.step - entry["opened_at"]
            if len(entry.get("hops") or []) < 5 and spent <= entry["share"] and self.world.steps_left() >= min(15, self.world.max_steps // 3):
                continue
            entry["dropped"] = True
            self.ledger.looked_up.append((_need_property(str(entry.get("property"))), [str(entry.get("seller")).strip().lower()] + list(entry.get("hops") or [])))
            self.trace.append({"kind": "reference_dropped", "seller": entry.get("seller"), "spent": spent})
            notes.append(f"The lookup of the {entry.get('property')} could not be finished within its share of the steps, so that condition stays unchecked: stop following it, choose among the candidates by the other conditions and order.")
        return " ".join(notes)

    def seller_rows(self, observation: Any) -> str:
        # rows that carry a referenced seller's name in any of their own fields
        notes = []
        for entry in self.references():
            name = str(entry.get("seller")).strip().lower()
            hits = []
            for found in _nested_dicts(observation):
                ident = _row_id(found)
                if ident and any(isinstance(part, str) and part.strip().lower() == name for part in found.values()):
                    hits.append(ident)
            if hits:
                notes.append(f"listings sold by {entry.get('seller')} in this result: " + ", ".join(dict.fromkeys(hits)) + f" (open them to find the one with {entry.get('identify')})")
        return " ".join(notes)

    def quantity(self) -> int:
        wanted = self.plan.get("quantity")
        return wanted if isinstance(wanted, int) and not isinstance(wanted, bool) and wanted > 0 else 1

    def cheaper_left(self) -> str:
        # under a cheapest rule, no cheaper same-range listing within budget may stay unjudged
        # a hold needs room to act on it: opening, judging and swapping a line take several steps
        if self.plan.get("priority") != "cheapest" or not self.ledger.cart_ids or self.price_holds >= SHELF_HOLDS:
            return ""
        # opening one listing, swapping its line and ordering take about four steps
        if self.world.steps_left() <= 4 or self.clock.left() < WRAP_UP_SECONDS + 20:
            return ""
        shelf = self.ledger.shelf
        verify = self.advisors.verify
        units = 1 if self.basket else self.quantity()
        cap = self.ledger.budget_now if _plain_number(self.ledger.budget_now) else None
        lines = self.ledger.cart_ids[-len(self.basket):] if self.basket else self.ledger.cart_ids[-1:]
        judged: list[tuple[float, str]] = []
        unread: list[tuple[float, str]] = []
        for pick in lines:
            cost = shelf.price(pick)
            if cost is None:
                continue
            label = shelf.range_of(pick)

            def cheaper(ident: str) -> bool:
                value = shelf.price(ident)
                return ident not in lines and value is not None and value < cost and not shelf.sold_out(ident) and (not label or shelf.range_of(ident) == label) and (cap is None or value * units <= cap)

            # a cheaper listing the loop opened but skipped goes to the checker before it is dropped
            for _, ident in sorted((shelf.price(i), i) for i in shelf.opened if cheaper(i) and i not in verify.passed)[:1]:
                verify.check(ident, "BUY INSTEAD")
            judged += sorted((shelf.price(i) or 0.0, i) for i in verify.passed if cheaper(i))
            if self.world.steps_left() > 5:
                # with few steps left only the cheapest couple are worth opening
                unread += shelf.unopened(label, cap, units, below=cost)[:6 if self.world.steps_left() > 10 else 2]
        if not judged and not unread:
            # nothing cheaper is known: when the search can bound its numbers and never went below the cart lines, one capped search is owed
            search = self.role("search")
            numeric = any(isinstance(spec, dict) and spec.get("type") in ("number", "integer") for spec in self.guard.props(search).values()) if search else False
            priced = [(pick, shelf.price(pick)) for pick in lines if shelf.price(pick) is not None]
            lowest = min(cost for _, cost in priced) if priced else 0
            # a page number is far below any price; a bound near the cart price is a capped search
            if not numeric or not priced or self.world.steps_left() <= 10 or any(lowest / 10 <= number < lowest for number in self.search_numbers):
                return ""
            self.price_holds += 1
            return "The request asks for the cheapest fit, and no listing priced below the cart line" + ("s" if len(priced) > 1 else "") + " (" + ", ".join(f"{pick} at {cost:g}" for pick, cost in priced) + ") has been seen. Search its range again with the wanted value words and the search's price bound set just under that price, open the new listings and judge them; cart a cheaper one only if it meets every requirement. If none does, send the order again."
        self.price_holds += 1
        parts = []
        if judged:
            parts.append("these listings passed the independent requirement check and are priced below the cart line of their range: " + ", ".join(f"{i} ({p:g})" for p, i in judged[:4]) + " (the shop's category label settles what kind of item a listing is, so a title or type field naming another kind does not rule it out)")
        if unread:
            parts.append("these listings of the same range within the budget are priced below the cart line and were never opened: " + ", ".join(f"{i} ({p:g})" for p, i in unread[:6]))
        return "The request asks for the cheapest fit, and " + "; ".join(parts) + ". Open and judge the cheaper ones, cart the cheapest that meets every requirement, then order. If each of them fails, send the order again."

    def decline_gap(self) -> str:
        # a refusal stands only when no audited fit survives the check and no in-budget listing of the range is left unopened
        if self.decline_holds >= SHELF_HOLDS or self.world.steps_left() <= 4 or self.clock.left() < WRAP_UP_SECONDS + 30:
            return ""
        shelf = self.ledger.shelf
        survivors = []
        judged = 0
        for ident in self.advisors.audit.fits[:3]:
            if ident not in self.advisors.verify.passed and self.ledger.records_of(ident) and not judged:
                judged += 1
                self.advisors.verify.check(ident, "ORDER")
            if ident in self.advisors.verify.passed:
                survivors.append(ident)
        wanted = [entry["product"] for entry in self.basket] or [str(self.plan.get("product") or "")]
        unread = [i for name in wanted for _, i in shelf.unopened(shelf.main_range(name), self.ledger.budget_now, 1 if self.basket else self.quantity())[:3 if self.basket else 6]]
        # the rows of the most selective search matched the wanted words, even when they show no category
        cap = self.ledger.budget_now if _plain_number(self.ledger.budget_now) else None
        labels = {shelf.main_range(name) for name in wanted} | {None}
        for _, shown in sorted(self.search_rows, key=lambda entry: entry[0]):
            fresh = [ident for ident in shown if ident not in shelf.opened and ident not in unread and shelf.range_of(ident) in labels and not shelf.sold_out(ident) and not (cap is not None and shelf.price(ident) is not None and shelf.price(ident) > cap)]
            if fresh:
                unread += fresh[:4]
                break
        if not self.reading_open():
            # the last steps no longer offer a tool that opens listings, so unopened ones cannot hold a refusal
            unread = []
        if not survivors and not unread:
            return ""
        self.decline_holds += 1
        parts = []
        if survivors:
            parts.append("these listings passed the requirement check, so the shop does have a fit: " + ", ".join(survivors) + "; cart the best of them and order it")
        if unread:
            parts.append("these listings of the same range within the budget were never opened: " + ", ".join(unread) + "; open them before deciding")
        return "Before a refusal: " + "; ".join(parts) + ". Decline only if every one of them fails a requirement."

    def scrub_decline(self, args: dict[str, Any], name: str) -> dict[str, Any]:
        rule = self.plan.get("decline_no_names")
        if rule is False:
            return args
        leaks = any(
            token and token in self.ledger.seen_ids
            for value in args.values() if isinstance(value, str)
            for token in re.split(r"[\s.,;:!?()\[\]{}<>\"'`]+", value)
        )
        if not (rule or leaks):
            return args
        known = self.guard.props(name)
        scrubbed = {}
        for key, value in args.items():
            spec = known.get(key) or {}
            scrubbed[key] = _DECLINE_TEXT[: spec.get("maxLength", len(_DECLINE_TEXT))] if isinstance(value, str) and "enum" not in spec else value
        return scrubbed

    # execution

    def run_calls(self, calls: list[dict[str, Any]], planned: bool = False) -> list[tuple[str, str]]:
        replies: list[tuple[str, str]] = []
        allowed = {tool["function"]["name"] for tool in self.offered_tools()}
        finishing = self.finishers()
        messenger = self.role("message")
        ready: list[tuple[str, str, dict[str, Any]]] = []
        queued_messages = 0
        for call in calls:
            refusal, accepted = self.screen(call, allowed, planned, len(calls), queued_messages)
            if refusal is not None:
                replies.append((call["id"], refusal))
                continue
            name, args = accepted
            if name == messenger:
                queued_messages += 1
            ready.append((call["id"], name, args))
        if ready and ready[0][1] in finishing:
            call_id, name, args = ready[0]
            held, args = self.gate_finish(name, args)
            if held is not None:
                replies.append((call_id, held))
                return replies
            ready[0] = (call_id, name, args)
        elif ready and finishing:
            # before the first order attempt two steps stay free: the order and one repair of a refused order
            keep = 2 if self.order_attempts == 0 and self.ledger.cart_ids else 1
            room = max(0, self.world.steps_left() - keep)
            if room < len(ready):
                replies.extend((call_id, f"Not run (no step used): the last {keep} step{'s are' if keep > 1 else ' is'} kept for the order" + (" and one repair of it, so order now." if keep > 1 else " (or the decline).")) for call_id, _, _ in ready[room:])
                ready = ready[:room]
        for offset in range(0, len(ready), 16):
            self.execute(ready[offset : offset + 16], replies)
        position = {call["id"]: index for index, call in enumerate(calls)}
        replies.sort(key=lambda pair: position.get(pair[0], 0))
        return replies

    def execute(self, batch: list[tuple[str, str, dict[str, Any]]], replies: list[tuple[str, str]]) -> None:
        if self.world.over():
            replies.extend((call_id, "Not run: the episode is over or no steps are left.") for call_id, _, _ in batch)
            return
        batch = batch[: self.world.steps_left()]
        actions = [{"name": name, "args": args} for _, name, args in batch]
        first_step = self.world.step + 1
        answer = self.world.call(actions)
        self.trace.append({"kind": "env", "step": first_step, "actions": actions, "reply": answer})
        if answer is None:
            replies.extend((call_id, "The call failed (network or server). Try again.") for call_id, _, _ in batch)
            return
        observations = answer.get("observations") if isinstance(answer.get("observations"), list) else [answer.get("observation")]
        for index, (call_id, name, args) in enumerate(batch):
            if index < len(observations):
                text, note = self.absorb(name, args, observations[index])
                if index == len(observations) - 1 and answer.get("finished"):
                    text += " [the episode is finished]"
                if name == self.role("cart") and not answer.get("finished"):
                    # lines the loop itself takes out in this batch are left to its own calls
                    leaving = {value for _, later, later_args in batch[index + 1 :] if later == name and self.guard.zero_count(name, later_args, self.ledger.adds_units) for value in later_args.values() if isinstance(value, str)}
                    swapped = self.drop_replaced(args, leaving)
                    note = f"{note} {swapped}".strip() if swapped else note
                replies.append((call_id, text + "\n[note] " + note if note else text))
            else:
                replies.append((call_id, "Not run: the episode finished before this call."))

    def drop_replaced(self, args: dict[str, Any], leaving: set[str] | None = None) -> str:
        # one line per wanted product: a line the loop adds replaces an older line of the same product, taken out in code at once
        book, cart = self.ledger, self.role("cart")
        added = next((value for value in args.values() if isinstance(value, str) and value in book.cart_ids), None)
        if not added or not cart or self.world.steps_left() < 3 or self.guard.zero_count(cart, args, book.adds_units):
            return ""
        owner = self.line_product(added)
        older = [ident for ident in book.cart_ids if ident != added and owner is not None and self.line_product(ident) == owner and ident not in (leaving or set())]
        calls = [self.guard.removal_args(cart, ident, self.cart_values("remove"), book.cart_units.get(ident, 1), book.adds_units) for ident in older]
        if not older or None in calls or any(self.guard.check_args(cart, call)[1] for call in calls):
            return ""
        actions = [{"name": cart, "args": call} for call in calls]
        answer = self.world.call(actions)
        self.trace.append({"kind": "drop_replaced", "added": added, "removed": older, "reply": answer})
        if not answer:
            return ""
        observations = answer.get("observations") if isinstance(answer.get("observations"), list) else [answer.get("observation")]
        for action, observation in zip(actions, observations):
            self.absorb(cart, action["args"], observation)
        return f"{added} replaces {', '.join(older)} for the {owner}: the older line was taken out in code (one step), so the cart keeps one line per product."

    def book_cart(self, name: str, args: dict[str, Any]) -> None:
        known = self.guard.props(name)
        signature = self.guard.enum_signature(name, args)
        book = self.ledger
        removing = self.guard.zero_count(name, args, book.adds_units)
        if self.guard.idle_count(name, args, book.adds_units):
            return
        if book.add_signature is None and not removing:
            book.add_signature = signature
        for key, value in args.items():
            spec = known.get(key) if isinstance(known.get(key), dict) else {}
            if isinstance(value, str) and "enum" not in spec and value.strip() and " " not in value.strip():
                if signature == book.add_signature and not removing:
                    book.cart_ids = [entry for entry in book.cart_ids if entry != value] + [value]
                else:
                    book.cart_ids = [entry for entry in book.cart_ids if entry != value]
                    book.taken_out[value] = book.version

    def absorb(self, name: str, args: dict[str, Any], observation: Any) -> tuple[str, str]:
        messenger, cart = self.role("message"), self.role("cart")
        finishing = self.finishers()
        if name == messenger:
            self.messages_used += 1
        if name in finishing and self.intent(name, args) == "order":
            self.order_attempts += 1
        if name == cart:
            self.cart_used = True
            self.order_checked = False
        # when each listing was last read and last carted, so an order never rests on a reading older than the cart line
        touched = next((value for value in args.values() if isinstance(value, str) and value in self.ledger.shelf.rows), None)
        if touched and name == cart:
            self.carted_at[touched] = self.world.step
        if touched and name == self.role("details"):
            self.read_at[touched] = self.world.step
        self.recent = [] if name in (cart, messenger) or name in finishing else (self.recent + [name + _dump(args)])[-2:]
        self.ledger.remember(observation)
        before = len(self.ledger.shelf.rows)
        self.ledger.shelf.take(observation, name in {self.role("details"), self.role("view")} - {None})
        if name in {self.role("details"), self.role("view")} - {None} and self.role("view") and self.role("details") != self.role("view"):
            # the tool whose results carry an attribute sheet is the details tool, whatever the setup reading said
            sheet = any(not _row_id(found) and len([part for part in found.values() if isinstance(part, str) and 0 < len(part) <= 70]) >= 3 for found in _nested_dicts(observation))
            self.sheet_seen.setdefault(name, []).append(sheet)
            details, view = self.role("details"), self.role("view")
            if details in self.sheet_seen and view in self.sheet_seen and not any(self.sheet_seen[details]) and all(self.sheet_seen[view]):
                self.roles["details"], self.roles["view"] = view, details
        if name == self.role("search"):
            self.searches += 1
            self.searched_budget = self.ledger.budget_now
            self.search_numbers.extend(float(value) for value in args.values() if _plain_number(value))
            self.used_needles |= set().union(*[_words(value) for value in args.values() if isinstance(value, str)])
            self.used_phrases.extend(frozenset(_words(value)) for value in args.values() if isinstance(value, str))
            shown = list(dict.fromkeys(ident for found in _nested_dicts(observation) for ident in [_row_id(found)] if ident))
            # the search's own total of matches, when it reports one, tells how selective its words were
            totals = [part for part in (observation.values() if isinstance(observation, dict) else []) if isinstance(part, int) and not isinstance(part, bool) and part >= len(shown)]
            if shown:
                self.search_rows.append((min(totals) if totals else len(shown), shown))
            self.dry_search = len(self.ledger.shelf.rows) == before
        target = next((value for value in args.values() if isinstance(value, str) and (value in self.ledger.cart_units or value in self.ledger.shelf.rows)), None) if name == cart else None
        held_before = self.ledger.cart_units.get(target or "")
        if name == cart:
            self.book_cart(name, args)
        if name in (cart, self.role("cart_view")):
            self.ledger.sync_cart(observation)
        if target and held_before:
            # a line that already held units shows whether the tool adds the units sent or sets the line to them
            sent = next((args[key] for key in self.guard.counts(name) if _plain_number(args.get(key))), None)
            after = self.ledger.cart_units.get(target, 0)
            if sent and after == held_before + sent and after != sent:
                self.ledger.adds_units = True
            elif sent is not None and after == sent and after != held_before + sent:
                self.ledger.adds_units = False
        if name == self.role("cart_view"):
            self.ledger.last_cart_view = observation
            self.ledger.shelf.learn_price(observation)
        note = self.observe(name, observation)
        view = self.role("view")
        if name == self.role("details") and view and view != name and not self.price_note:
            opened = next((value for value in args.values() if isinstance(value, str) and value in self.ledger.shelf.rows), None)
            if opened and self.ledger.shelf.price(opened) is None:
                # a sheet without price or stock leaves both to the lighter tool
                self.price_note = True
                note = f"{note} This sheet shows no price or stock; {view} shows them, so read the candidates' prices and stock with {view} before choosing.".strip()
        if name == self.role("search") and self.dry_search:
            hint = self.needle_hint()
            if hint:
                note = f"{note} {hint}".strip()
        lookup = self.seller_rows(observation) if name == self.role("search") else ""
        if lookup:
            note = f"{note} {lookup}".strip()
        if name == self.role("details"):
            followed = self.follow_references()
        else:
            followed = self.cap_references()
        if followed:
            note = f"{note} {followed}".strip()
        heard: list[str] = []
        if name == messenger:
            if isinstance(observation, str):
                heard = [observation]
            elif isinstance(observation, dict) and "error" not in observation:
                _short_sentences([v for v in observation.values() if isinstance(v, (str, list))], heard)
            question = " ".join(str(v) for v in args.values() if isinstance(v, str))
        else:
            question = None
            if isinstance(observation, dict):
                for value in observation.values():
                    if isinstance(value, list) and value and all(isinstance(v, str) for v in value):
                        _short_sentences(value, heard)
                    elif isinstance(value, str) and len(value.split()) >= 4 and self.refused(name, observation):
                        heard.append(value)
        if heard and not (name == messenger and self.ask_stage_on):
            meaning, reply = self.advisors.interpret.read(heard, question)
            if reply and self.role("message"):
                self.pending_reply = reply
            if meaning:
                note = f"{note} {meaning}" if note else meaning
        return _dump(observation), note

    def refused(self, name: str, observation: dict[str, Any]) -> bool:
        # a finish that left the task open and carries the shopper's sentences was turned down, whatever its key is called
        return "refusal" in observation or (name in self.finishers() and not self.world.finished and bool(_string_lines(observation)))

    def observe(self, name: str, observation: Any) -> str:
        notes = []
        if isinstance(observation, dict):
            self.ledger.note_ids(observation)
            if name == self.role("message") and _code_only(observation):
                if name == self.role("message") or self.message_budget is not None:
                    self.message_budget = self.messages_used if self.message_budget is None else min(self.message_budget, self.messages_used)
                notes.append("That budget is used up; do not call this tool again.")
            if self.refused(name, observation):
                self.refusals += 1
                self.order_checked = False
                notes.append(REFUSAL_NOTE)
            elif len(observation) >= 2 and name != self.role("message"):
                lines = _string_lines(observation)
                if lines:
                    notes.append("This result carries a message from the shopper (a change). Apply it: " + " | ".join(lines[:4]))
            if name == self.role("search") and not self.paging_noted:
                gap = _unpaged(observation)
                if gap:
                    self.paging_noted = True
                    notes.append(f"This search reports {gap[0]} matches but shows {gap[1]} and offers no further page; to see the rest, narrow it (more precise words, a filter or price bounds) instead of guessing a page value.")
        elif isinstance(observation, list):
            self.ledger.note_ids(observation)
        return " ".join(notes)

    # finishing

    def cart_values(self, job: str) -> dict[str, Any]:
        found = self.plan.get("cart_values")
        values = found.get(job) if isinstance(found, dict) else None
        return values if isinstance(values, dict) else {}

    def finish_values(self, job: str) -> dict[str, Any]:
        found = self.plan.get("finish_values")
        values = found.get(job) if isinstance(found, dict) else None
        return values if isinstance(values, dict) else {}

    def force_finish(self) -> None:
        if self.world.over() or self.clock.left() < 2:
            return
        order, decline = self.role("order"), self.role("decline")
        if order and not self.ledger.cart_ids and self.world.steps_left() >= 2:
            fits = self.passed_fits() if not self.basket else []
            if fits and self.order_lines([(fits[0], self.quantity())], "forced finish"):
                if self.world.over():
                    return
            # with nothing passed, the listing held only on unshown conditions beats a refusal here as well
            if not fits and self.soft_buy() and self.world.over():
                return
        if order and self.cart_used and self.ledger.cart_ids:
            action = {"name": order, "args": self.guard.filler_args(order, "Order the cart.", self.finish_values("order"), True, ORDER_WORDING)}
        elif decline:
            action = {"name": decline, "args": self.guard.filler_args(decline, _DECLINE_TEXT, self.finish_values("decline"), True, DECLINE_WORDING)}
        else:
            return
        answer = self.world.call([action])
        self.trace.append({"kind": "forced_finish", "action": action, "reply": answer})
        refused = bool(answer) and not answer.get("finished") and isinstance(answer.get("observation"), dict) and self.refused(action["name"], answer["observation"])
        if refused and decline and self.world.steps_left() > 0:
            action = {"name": decline, "args": self.guard.filler_args(decline, _DECLINE_TEXT, self.finish_values("decline"), True, DECLINE_WORDING)}
            answer = self.world.call([action])
            self.trace.append({"kind": "forced_finish", "action": action, "reply": answer})

    # main loop

    def compact_history(self) -> None:
        # older results are compacted a block at a time, so the cached prefix holds for many turns between passes
        if sum(len(str(m.get("content") or "")) for m in self.messages) < HISTORY_CHARS:
            return
        edge = len(self.messages) - KEEP_RECENT
        if edge <= self.compacted or sum(len(str(m.get("content") or "")) for m in self.messages[self.compacted:edge]) < COMPACT_BLOCK:
            return
        start, self.compacted = self.compacted, edge
        reader = json.JSONDecoder()
        for spot, message in list(enumerate(self.messages[:edge]))[start:]:
            text = message.get("content")
            if message.get("role") != "tool" or not isinstance(text, str) or len(text) < 1200:
                continue
            rest = text.split("\n[status]")[0]
            try:
                body, end = reader.raw_decode(rest)
            except ValueError:
                self.messages[spot] = dict(message, content=rest[:1200] + "...(cut)")
                continue
            self.messages[spot] = dict(message, content=_dump(_shrink(body)) + rest[end:])

    def opening_message(self) -> str:
        opening = self.world.opening
        text = "SHOPPER REQUEST:\n" + self.world.query
        if opening is not None:
            text += "\n\nOPENING OBSERVATION:\n" + _dump(opening)
        reading = self.first_reading()
        if reading:
            text += "\n\n" + reading
        if self.ask_note:
            text += "\n\n" + self.ask_note
        text += "\n\n" + self.status()
        if opening is not None:
            self.ledger.remember(opening)
            self.ledger.note_ids(opening)
        return text

    def normalise_calls(self, message: dict[str, Any], counter: int) -> tuple[list[dict[str, Any]], int]:
        calls = []
        for raw in message.get("tool_calls") or []:
            if not isinstance(raw, dict):
                continue
            counter += 1
            function = raw.get("function") or {}
            arguments = function.get("arguments")
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments or {})
            calls.append({"id": raw.get("id") or f"call_{counter}", "type": "function", "function": {"name": function.get("name"), "arguments": arguments}})
        return calls, counter

    def turn_loop(self) -> None:
        idle = failures = counter = 0
        for _ in range(MAX_TURNS):
            if self.world.over():
                return
            if self.clock.left() < FORCE_SECONDS or self.clock.cut:
                # past the runaway spend line no further LLM turn is taken; code finishes with what the cart holds
                self.force_finish()
                return
            if self.pending_reply:
                text, self.pending_reply = self.pending_reply, ""
                if self.send_reply(text):
                    continue
            if self.planner_on and self.ask_planned():
                continue
            self.compact_history()
            message = self.llm.ask(self.messages, self.offered_tools(), [MAIN_MODEL, BACKUP_MODEL, MAIN_MODEL], 4000)
            if message is None and self.llm.dry:
                self.force_finish()
                return
            if message is None:
                failures += 1
                if failures >= 2:
                    self.force_finish()
                    return
                continue
            calls, counter = self.normalise_calls(message, counter)
            content = message.get("content") or ""
            if not isinstance(content, str):
                content = _dump(content)
            committing = {self.role("cart"), self.role("order"), self.role("decline")} - {None}
            if self.planner_on and any(call["function"]["name"] in committing for call in calls):
                if self.ask_planned(force=True):
                    continue
                details = self.role("details")
                if details and not self.messages_used and not self.detail_hold and self.world.steps_left() > 10:
                    # the questions need some listings' own properties; a cart or a finish before any question skips the shopper's hidden wishes
                    self.detail_hold = True
                    self.messages.append({"role": "assistant", "content": content or None, "tool_calls": calls})
                    self.post_tool_replies([(call["id"], f"Not run (no step used): no question has reached the shopper yet, and questions need some listings' own properties first. Open the full details of two or three candidates with {details}; the questions then go out for you.") for call in calls])
                    continue
            if calls:
                idle = 0
                self.trace.append({"kind": "model", "content": content[:2000], "calls": [call["function"] for call in calls]})
                self.messages.append({"role": "assistant", "content": content or None, "tool_calls": calls})
                self.post_tool_replies(self.run_calls(calls))
                continue
            idle += 1
            self.messages.append({"role": "assistant", "content": content or "(no tool call)"})
            self.messages.append({"role": "user", "content": "Every reply must call at least one of the supplied tools. Continue.\n" + self.status()})
            if idle >= 3:
                self.force_finish()
                return

    def run(self) -> list[dict[str, Any]]:
        if self.world.finished:
            return self.trace
        plan = self.advisors.setup.read()
        self.read_caps(plan)
        self.adopt_plan(plan)
        self.trace.append({"kind": "plan", "plan": self.plan, "roles": self.roles})
        self.ask_stage()
        self.learn_stock()
        self.search_stage.run()
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": self.opening_message()}]
        self.turn_loop()
        if not self.world.finished and not self.world.dead:
            self.force_finish()
        self.trace.append({"kind": "end", "finished": self.world.finished, "steps": self.world.step, "cost": round(self.llm.cost, 5), "spend": {k: [v[0], round(v[1], 5)] for k, v in self.llm.spend.items()}, "seconds": round(self.clock.elapsed(), 1)})
        return self.trace


def assemble(problem_data: dict[str, Any]) -> Driver:
    clock = Clock()
    trace: list[dict[str, Any]] = []
    world = WorldDesk(clock, trace, problem_data["environment"])
    guard = SchemaGuard(world.policy)
    llm = LLMDesk(clock, trace)
    ledger = Ledger()
    advisors = Advisors(
        SetupAdvisor(llm, world, guard),
        InterpretAdvisor(llm, ledger, clock, world.query),
        PlanAdvisor(llm, ledger, world.query),
        VerifyAdvisor(llm, ledger, clock, trace, world.query),
        AuditAdvisor(llm, ledger, clock, trace, world.query),
    )
    if ENGINE_JUDGE:
        advisors.verify.table = VerdictTable(llm, ledger, clock, trace, world.query)
    return Driver(clock, trace, llm, world, guard, ledger, advisors)


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    driver = assemble(problem_data)
    try:
        return driver.run()
    except Exception as exc:
        driver.trace.append({"kind": "crash", "error": f"{type(exc).__name__}: {exc}"[:500]})
        try:
            if not driver.finished and not driver.dead:
                driver.force_finish()
        except Exception as again:
            driver.trace.append({"kind": "crash", "error": f"{type(again).__name__}: {again}"[:500]})
        return driver.trace


__all__ = ["agent_main"]
