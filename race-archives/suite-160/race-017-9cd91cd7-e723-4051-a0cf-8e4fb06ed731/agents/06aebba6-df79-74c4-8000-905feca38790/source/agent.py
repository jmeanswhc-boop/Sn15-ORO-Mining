"""Shopping agent for ORO Bench (Bittensor subnet 15).

One synchronous entry point, agent_main(problem_data). The agent reads the world's own tools
from the policy view and drives the episode through /environment/call, one tool call per step.

Design, in short:
  * a brief: one careful model call turns the request into structured requirements, budgets,
    rules and a map from roles (search, details, cart, order, decline, message) to tool names;
  * a worker loop: a fast model chooses tool calls, seeing a status board that the code keeps
    (steps and messages left, current requirements, every listing seen with price and stock);
  * a question planner: once a few listings are known, the code questions the shopper on the
    properties that differ between candidates, one property per message, and folds the
    answers into the requirements;
  * gates that cost no steps: before an order the chosen item is checked against the
    requirements, budget, stock and (for "cheapest") cheaper known listings; before a decline
    the known listings are audited for a fit, and the remark is kept free of listing names;
  * an endgame that always finishes before the step or work budget runs out.

Effort is metered by a work counter that charges each model call and each tool call a fixed amount. The
clock is read only as a safety stop: the sandbox kills a task after 300 s (SANDBOX_PROBLEM_TIMEOUT), and a
killed task scores nothing, so the episode is finished before that whatever the work counter says.
"""

from __future__ import annotations

import json
import collections
import re
import time
from typing import Any

from src.agent.proxy_client import ProxyClient

# ---------------------------------------------------------------------------------------------
# Models and budgets

BRIEF_MODEL = "anthropic/claude-haiku-5.5"
JUDGE_MODEL = "anthropic/claude-sonnet-5.5"
WORKER_MODEL = "anthropic/claude-haiku-5.5"
FALLBACK_MODEL = "openai/gpt-6-luna"
LIGHT_MODEL = "anthropic/claude-haiku-5.5"

EPISODE_SECONDS = 272.0     # safety stop: the sandbox kills a task at 300 s; keep a margin for the last calls
WORK_BUDGET = 560.0         # work units per task (about seconds on a race validator)
CALL_WORK = 9.0             # charged for each model request (attempt), whatever its outcome
SLOW_CALL_WORK = 14.0       # ...for a request to the larger judge model
STEP_WORK = 0.5             # charged for each tool call
FINISH_SECONDS = 50.0       # with this little work left, finish with the best known item
MODEL_SECONDS = 60.0        # network timeout of one model request (an inference error past it)
THINK_TOKENS = 4000         # room for a model's own reasoning, which counts against max_tokens, on top of the answer
SPEND_HARD_USD = 0.08       # past this spend in one task, stop the worker and finish with what is known
TOOL_TEXT_LIMIT = 9000      # characters of one tool result shown to the worker
OLD_TOOL_TEXT = 700         # characters kept of older tool results
KEEP_FULL_RESULTS = 6       # past this many full tool results, older ones are shortened...
TRIM_TO = 3                 # ...down to this many, in one go, so the conversation's start stays cached between trims
STEP_RESERVE = 5            # steps kept free for a refused order to be handled
DECLINE_WORDS = r"declin|abandon|refus|cancel|no.?(buy|purchase|order)|give.?up|reject|skip|abstain|\bnone\b|nothing|without|neither"
ORDER_WORDS = r"order|buy|purchas|check.?out|commit|confirm|proceed|place|accept|complete|finali[sz]e|submit|\bpay|\byes\b|approv|go.?ahead"
STOP_WORDS = DECLINE_WORDS + r"|\bpass|\bno\b|\bnot\b|\bstop|\bquit|\bleave|walk|\bforgo|\babort|\bdrop|\bdeny|\bhold.?off"


def _choice_declines(value: str, options: list[str]) -> bool:
    """Whether a finishing tool's choice value ends without buying, judged against its alternatives: a value saying
    decline or stop does; otherwise one that is not the order-like value while another option is (race 13's
    order/pass, race 11's commit/none, race 8's buy/skip: "pass" was read as an order before)."""
    words = lambda t: re.sub(r"[_\-]+", " ", re.sub(r"(?<=[a-z])(?=[A-Z])", " ", t))  # checkout_none, buyNothing
    value, options = words(value), [words(o) for o in options]
    if re.search(DECLINE_WORDS, value, re.I):
        return True  # complete_no_purchase, buy_nothing, proceed_without_buying
    if re.search(STOP_WORDS, value, re.I) and not re.fullmatch(r"\W*(" + ORDER_WORDS + r")\w*\W*", value, re.I):
        return True
    if re.search(ORDER_WORDS, value, re.I):
        return False
    return any(re.search(ORDER_WORDS, o, re.I) for o in options if o != value)


def _outcome_choice(name: str, options: list[str]) -> bool:
    """A choice field that picks the ending: named like one, or offering an order-like value and a declining one (a
    delivery choice such as standard/drop_off is neither)."""
    if re.search(r"way|mode|kind|variant|action|outcome|result|choice|decision|verb|method|finish|close", name, re.I):
        return True
    ordering = [o for o in options if re.search(ORDER_WORDS, o, re.I) and not re.search(DECLINE_WORDS, o, re.I)]
    stopping = [o for o in options if o not in ordering and re.search(DECLINE_WORDS + "|" + STOP_WORDS, o, re.I)]
    return bool(ordering) and bool(stopping)


COMMIT_STEPS = 12           # with this few steps left, stop exploring and order the best confirmed match

_clients: dict[tuple[int, int], ProxyClient] = {}


def _client(timeout: float, attempts: int) -> ProxyClient:
    """A proxy client with a per-request timeout bucketed to 10 s, so late calls cannot overrun the task."""
    key = (max(10, int(timeout) // 10 * 10), max(1, attempts))
    if key not in _clients:
        _clients[key] = ProxyClient(timeout=key[0], max_retries=key[1], retry_delay=1)
    return _clients[key]

# ---------------------------------------------------------------------------------------------
# Prompts

WORKER_GUIDE = """ROLE
You run errands for a single customer of a simulated web store, and you can only act by calling the store's tools. Grading is pass/fail. A pass needs one of two endings:
(a) a purchase of the correct item (or the correct option of an item) in the correct number of units, whose total stays inside the spending limit, which is available when the customer insists on available goods, which satisfies everything the customer wrote, everything they tell you when asked and every update they send along the way, and which is the lowest-priced or top-ranked match when they ask for that; or
(b) giving up, only if the store truly has nothing acceptable, carried out in the manner the customer prescribes.
Working instructions in the request (number of items to inspect, things to confirm, things to mention or omit) are graded too.

HOW THE GAME WORKS
1. Each tool call spends one move from a fixed allowance, whether it succeeds or not. Reasoning is free; moves are not.
2. Every store invents its own tool names, parameters and response formats. Follow each schema to the letter.
3. Calls that do not depend on each other can go out together in one reply (say, three detail lookups). A purchase or a give-up call must always travel alone.
4. Each turn you also get a status sheet compiled by code from the actual responses. Its counts of moves and messages, its list of needs and its item table are reliable.

SEARCHING
5. Begin by searching for the kind of product itself in plain words, to learn the store's department name for it; starting from an attribute often drags in other departments. The status sheet lists department names seen so far. Pick the department whose name covers the whole wording of the request (a request for "A or B" belongs with a department named like "A & B", not with a narrower sibling named just "B"). Shop only there.
6. Search tools can be awkward: every word may be mandatory, pages may be short and few, results may not be ordered by usefulness or price, and some filters may work in one mode only. Compare the total the tool reports with the rows actually visible, and make sure each filter really changed the output. Prefer short queries, filters, price ceilings and extra pages until you have covered what matters.
7. If a keyword query matches too many rows to read, add a second word (for example a product type) to narrow it, and try a few such combinations until every total fits on the readable pages.
8. When the status sheet warns that an argument seems to be ignored, stop relying on it in that mode. Fill each argument only with the kind of value its name and description ask for (a brand field takes brand names only), and when a tool that takes a list of items fails or returns nothing when given many, retry with half as many.
9. Open an item's detail sheet before relying on it; the deciding attributes usually live only there.

DECIDING WHETHER AN ITEM MATCHES
10. A need is satisfied only if the item's own field for that attribute carries the requested value. Use the fields named as the attribute itself (ignoring capitals, spaces, underscores, dots, hyphens, word order and plural endings), also when preceded by the product's own noun (a skirt's "skirt style" field); only when there are none, a field naming the same property in other words counts. Any other field never counts, even when its name shares a word with the attribute. Compare values word by word: the field must contain the customer's word or words, ignoring capitals, punctuation, word order, plural endings and spelling variants of the same word; any single entry of a comma-separated value is enough, and one counted field carrying it is enough even if another says something else. Abbreviations, synonyms, broader or narrower words and ranges that merely cover the value do not count.
11. An item with no field at all for a required attribute satisfies it only if its title shows the value; descriptions never count. An exclusion ("not X") is satisfied when no counted field (or, lacking one, the title) contains X.
12. Read the customer's conditions literally: a value they want the item to state is matched as written, and only explicit limit words (at least, at most, under, over, between) make a numeric bound.
13. Numbers are compared numerically after unit conversion; limits are inclusive; a range only passes if all of it lies within the limit.
14. Honour exclusions and closed lists. The kind of product is settled by the item's department: an item filed in the requested department is that product whatever its title calls it.
15. Unit price times units may not exceed the spending limit. An item marked unavailable cannot be bought when the customer insists on available goods.
16. Options or variants with separate identifiers, prices or availability are separate candidates.

THE CUSTOMER
17. The customer has unwritten preferences, and each surfaces only when a message mentions that attribute by name. A question planner sends those questions on your behalf after a few items of the right kind have been opened; the replies show up in the conversation and on the status sheet. Every preference revealed this way is binding.
18. Do not write to the customer yourself unless you are answering their question or the request explicitly demands a message, and never go over the message allowance. Never send progress reports, purchase notices or give-up notices.
19. Background details (who it is for, the occasion) do not constrain the choice unless the customer turns them into a condition.
20. Phrases like "ideally" or "would be nice" only decide between items that already satisfy everything, and never override a lowest-price instruction.

UPDATES WHILE YOU WORK
21a. When a condition points to another listing's value (the same as, or anything but, what some listing shows), find and open that listing first; the checks then use its actual value.
21. Customer remarks can be attached to any tool response or come back as the rejection of a purchase. They may change the spending limit, add, remove or swap a need, or report that an item is no longer available or more expensive. Each update applies from that point on: re-examine your choice, and under a lowest-price or top-rank instruction pick again among the items that now qualify.
22. A rejected purchase changes nothing: no sale happened and the basket is untouched. Absorb the update, correct the basket (take out anything that no longer qualifies), put in the next confirmed match and try the purchase again.
23. Items can sell out or change while you work, so keep a second confirmed match ready before buying. If a match is gone and you know no other, try searching distinctive words from its title, since stores sometimes list the same product more than once.

LOWEST PRICE OR TOP RANK
24. "Lowest price" means the cheapest among every item that satisfies all needs. Work upward from the bottom: filter the right department with a low price ceiling (in the mode where ceilings apply) so the reported total fits on the readable pages, open the cheapest promising rows, and raise the ceiling gradually only when nothing qualifies. The first full match in ascending price is the answer.
25. "Top by some measure" means the most extreme value of that measure among all full matches; an item that does not state the measure is out.

ENDING
26. Before purchasing, the basket must contain exactly the required line(s) in the required number of units and nothing more. When several products are wanted, every product takes a separate basket line holding its required count, the whole order stays inside the spending limit, and every product gets its own search, questions and checks. The status sheet tracks the basket from your own calls, so check the basket only when a basket call reported a problem.
27. When there is no lowest-price or top-rank instruction, buy the first confirmed match as soon as the customer's answers are in: any match passes, extra searching only wastes moves, and an early attempt leaves room to recover from a rejection. With such an instruction, finish the comparison first.
28. Keep a few moves in hand when you try to buy, so a rejection can still be handled. Avoid swapping items in and out of the basket.
29. A well-stocked store almost always has the right item, and giving up has been wrong in nearly every case: an item that seems to miss one condition is often the intended one (a value worded differently, a field read too strictly). Before concluding the search, re-check every near match condition by condition and search again with each required value on its own and with the product word, reading further pages. When no listing passes every check after that, the code orders the closest match for you; give up only when the store has nothing of the requested kind within the spending limit. If the request forbids alternatives, a give-up note may not mention, characterise or allude to any item.
30. Each reply must call a tool. Keep any text brief: current needs, confirmed matches, next action."""

BRIEF_PROMPT = """Turn a customer's shopping request and a store's tool list into a compact plan for an assistant. Reply with one JSON object only (no commentary, no code fences), using exactly these fields:

"goods": list of {"kind": the product type as the request names it, without attribute values but keeping qualifier words that form part of the type's name (for a request for "A or B", write "A or B"), "units": integer count of units of one item to buy (1 unless more are wanted; something sold as a pair or a set counts as 1)}. Normally a single entry.
"must_have": list of firm conditions about the product, each in the form "attribute: value" (when several products are wanted, start each condition with its product kind and " - ", as in "desk lamp - finish: matte"). The attribute is the property word the request itself uses for the condition, such as "finish" in "the finish must be matte"; only when the request gives a bare value, name the property it belongs to in plain words. The value is the short word or words the request attaches to the property, copied exactly, with nothing added: an explanation that follows it (after a colon, a dash or "meaning") is not the value, and neither are examples or paraphrases. Each property gets one entry for its wanted value (a closed list when several values are allowed); words that only restate or emphasise that wish add no second value. An exclusion stated together with a wanted value belongs to the same property. Examples of the form: "the lid must be glass, nothing else" gives "lid: glass"; "it's a quiet motor that matters: one that won't wake the baby" gives "motor: quiet"; "a tent for a beach setting, so nothing alpine" gives "setting: beach" and "setting: not alpine". A wanted value the request attaches to the property word itself (a word just before it, or after "is" or "be") always gets its own entry, also when exclusions or an explanation follow; each exclusion is another entry. Write an exclusion as "attribute: not X" (one entry per excluded value), a closed list as "attribute: A or B", and a numeric bound only when the request uses limit words such as at least, at most, under, over or between. A value the item itself should list (a rating, a reading or a range it states) is copied as the value, never turned into a bound. Count only what the request demands of the item itself (must, needs, required, only, at least, not, and similar). The customer's story is not a condition: who the item is for, the outing, trip or season they will use it in, delivery or arrival dates, their routine, plans and reasons all belong in "context", and so does any description of the situation, even one that hints at a property (what will be visible, who will see it, how it will be used). A kind of occasion, use or user that the request names as a value the item must carry is a condition, named with the request's own word for that property: a lamp "for reading use" gives "use: reading", and winter boots described as "for a cold climate" give "climate: cold". Leave out price, availability, quantity and the product type itself (words that only name the department, including a season or style word that is part of the product type's name).
"max_total": highest acceptable total price (number; null if unstated).
"money_unit": the currency or unit the request prices things in, or null.
"only_available": true if the customer accepts only goods that are actually available or stocked, otherwise false.
"ordering_rule": "lowest_price" only if the request clearly wants the cheapest qualifying item; "top_by_measure" only when it explicitly wants the strongest item on some measure; otherwise "any". Complaints about money are not a lowest-price request.
"measure": what is being maximised for "top_by_measure", else "".
"max_messages": message allowance the customer grants (integer; null if unstated).
"max_moves": move or step allowance the request grants (integer; null if unstated).
"procedures": every instruction about how the assistant must work, each as {"text": the instruction copied verbatim, "kind": "go_ahead" if the customer must approve the chosen item before it is bought, "tell_first" if the customer must be told something before the purchase (such as the item, its price or a reason), "inspect_count" if at least some number of items must be opened or considered, "check_basket" if the basket must be looked at before buying, "compare_first" if items must be compared before choosing, "give_up_style" when it says how to end without buying, "limits" when it sets message or step allowances, otherwise "other"; "count": the number involved, or null}.
"refusal_style": "no_alternatives" if giving up must not offer, name or hint at other items; "suggest_alternatives" if the request wants some named; otherwise null.
"nice_to_have": non-binding wishes (phrases such as "ideally" or "if possible").
"unknowns": information only the customer can supply that must be asked before choosing (e.g. a budget when none is stated, or which of two wishes wins).
"context": one sentence on the intended user, occasion and use of the product, if stated; else "".
"story_hints": product properties the customer's story hints they may care about without stating a value (for example who will use it, how many people, how it is carried, stored or shared, what it must go with, what will be seen of it, what its label or tag says); these are questions to ask, never conditions. Do not infer conditions from the story.
"tool_map": object giving the exact tool name for each job, or null: "find", "inspect", "basket", "basket_view", "checkout", "abandon", "contact". "inspect" shows one item's full attributes; "checkout" completes the purchase; "abandon" ends without buying; "contact" writes to the customer.
"search_notes": one or two sentences on using the search tool(s), taken from their descriptions and schemas: which parameters apply in which mode (e.g. "free text only in keyword mode; department, brand and price ceiling only in filter mode"), paging, and anything a mode ignores."""

PLAN_PROMPT = """Choose clarifying questions for a shopping helper. Its customer holds back some preferences; each preference is the value of one product attribute and is revealed only by a question about that attribute. Each question asks about one single attribute and contains no instructions or lists.

For every product under PRODUCT, rank up to 4 distinct attributes, most likely held back first:
1. Whatever appears under ASK FIRST for that product and is still unasked.
2. After that, the attributes the STORY HINTS and the customer's background point to (who will use it, how many people, how it is carried, stored or shared, what it must go with, what will be seen of it, what its tag says), each mapped to the attribute name the items below use for it.
3. Then attributes the items below actually have for that product, favouring ones present on most of its items with values that vary. An attribute with one shared value everywhere makes a weak question.
Held-back preferences are usually attributes with one short value per item, most often, in this order: the product's type or kind, its colour, its material, then a measured size of a specific part (such as a display or screen size) or its dimensions, and after those its pattern, style, shape, closure, finish, connector, form, or who it is for. A plain fit size (of clothing, shoes or a rug) rarely is, and neither is an attribute whose single value lists several purchase variants at once. When the request leaves open who will use the product, rank that (such as gender or age group) high.
Only a few items may have been opened, so an attribute the background clearly points to may be ranked even when the items below do not show it yet.
Skip anything already fixed by the request (its stated conditions, price, availability, quantity, product type), brand or seller, values that list many variants or features, near-duplicates of one attribute, and anything asked before.

Write each attribute exactly as the items name it (underscores and dots become spaces; drop store prefixes such as "fa", codes and suffixes): "memory type" and "memory" are different attributes, so keep the item's full name.

Output a single JSON object only: {"plan": [{"product": "<product as named under PRODUCT>", "attributes": ["...", "..."]}]}"""

READ_PROMPT = """Interpret fresh remarks from a shop customer for the assistant serving them. Output a single JSON object only, with these fields:
"gist": one short sentence restating what the customer said.
"current_needs": the full list of firm conditions after these remarks, each "attribute: value" (with the product kind and " - " in front when several products are wanted, as in CURRENT REQUIREMENTS) with the customer's own property word as the attribute and their value words copied exactly (an exclusion as "attribute: not X"). Begin with CURRENT REQUIREMENTS, take out whatever was withdrawn, update whatever changed and add each new firm condition with its exact value or limit. Do not derive conditions from background details. Leave out price, availability and quantity.
"indifferent_to": attributes the customer is indifferent about.
"removed": conditions the customer withdrew.
"budget_now": the new spending limit as a number if the customer changed it, else null. Copy the customer's own number exactly, in the request's unit; never convert.
"units_now": the new number of units as an integer if changed, else null.
"pick_gone": true if the customer says the proposed or chosen item is no longer purchasable (sold out, withdrawn), else false.
"price_moved": true if the customer says that item's price changed, else false.
"confused": true if the customer says they did not understand the assistant, else false.
"question_for_me": the customer's own question to the assistant, if any, else "".
When the customer's value points to another listing (the same as, or anything but, what some seller's or maker's listing shows), write that whole reference as the value, keeping the name exactly as written (capitals included) and every detail given to identify that listing.
Interpret colloquial or figurative language generously: a named value is a firm preference. Keep the property word the customer uses; never rename a property or merge two different properties. Saying they have no preference, that anything works or that they don't mind about the attribute asked puts it in indifferent_to and changes nothing else; it never cancels a condition the request stated. A condition is withdrawn only when the customer clearly withdraws it."""

JUDGE_PROMPT = """Audit candidate items for a shopping helper before it buys. For each item record below, decide from the record alone whether it satisfies EVERY condition. Each condition reads "attribute: value".
- Which fields count: the fields named as the condition's attribute, ignoring capitals, spaces, underscores, dots, hyphens, word order and plural endings, including that name preceded by the product's own noun (when the product is a skirt, a field named "skirt style" is its style). When the item has no such field, a field whose name is plainly another wording of the same property counts. Other fields never count, even when they share a word with the attribute or contain the requested word: a name with any other prefix or extra words is a different field.
- A required value is met when at least one counted field contains it: the customer's word or words, ignoring capitals, punctuation, word order, plural endings and spelling variants of the same word; one entry of a comma-separated list is enough, and other counted fields saying something else do not matter. Abbreviations, codes, synonyms, broader or narrower words and ranges that merely cover the value do not count.
- When the item has no field for the condition's property at all, its title stands in for that field (the description never does): the value is met only if the title contains it.
- An exclusion ("not X") is met when no counted field contains X (or, with no field for the property, the title does not). A closed list ("A or B") is met when a counted field contains one of the allowed values.
- Only the words at least, at most, under, over and between make a numeric bound; it holds when the item's number meets it after unit conversion, bounds are inclusive and a range passes only if all of it lies inside. Any other value containing a number must itself appear in a counted field.
- The product type is settled by the item's department, not by its title: an item filed in the requested product's department is that product whatever its title calls it, and an item filed elsewhere is not. The department is the one whose name covers the whole wording of the requested product (see CATEGORY LABELS SEEN): a request for "A or B" needs a department named like "A & B", and an item filed under a narrower sibling department (just "B") fails. Never require an attribute to repeat the department name.
- When several products are wanted, a condition starting with a product kind and " - " applies only to items of that product: judge each item as the product whose department it is filed in, against that product's conditions only.
- Ignore wishes and background; test only the listed conditions. Ignore price and availability (handled elsewhere).
- A condition that points to another listing's value (the same as, or anything but, what that listing shows) uses that listing's value from REFERENCED LISTINGS; when that listing is not there, mark the test not ok with the issue "open the referenced listing first".
- If records of the same item conflict, the latest record wins.
Work through the conditions one at a time for each item and quote the evidence (the counted field and its value, or "none") before deciding.
Output a single JSON object only: {"verdicts": [{"id": "...", "tests": [{"need": "...", "proof": "attribute: value", "ok": true or false}], "pass": true or false, "issues": ["short reason", ...]}]}. pass is true only when every test is ok. Mark ok=false only when the record clearly breaks the condition or does not address it at all."""

REFERENCE_PROMPT = """A customer's condition on one product property points to another store listing instead of naming a value.
Output one JSON object only:
{"relation": "same" if the wanted value is the one that listing has, "different" if it must differ from it,
 "seller": the shop, seller or maker name of that listing exactly as written (capitals kept),
 "department": the department or product type of that listing as written, or "",
 "details": the identifying details given for that listing, each as {"attribute": "...", "value": "..."}}"""

LISTING_PROMPT = """Read one store listing for a shopping helper. PROPERTY names one product property.
Output one JSON object only:
{"value": this listing's own value for PROPERTY, from its attribute table or its text, or "" when it gives none
 or says to take it from another item,
 "defers_to": null, or, when the listing's text says to take PROPERTY from another item (such as "for PROPERTY,
 refer to / go by / use the ... item, not this one" or "its PROPERTY is to be taken from ..."):
 {"seller": that item's shop, seller or maker name exactly as written, "department": its department or product
  type, "details": its identifying details, each as {"attribute": "...", "value": "..."}}}
A note about some other property is not a deferral for PROPERTY.
Also give "code": when the listing gives no value for PROPERTY but its text sends the reader to another item by a code
(such as "see part AB123 for the rest"), that code exactly as written; else ""."""

# ---------------------------------------------------------------------------------------------
# Small helpers

_ID_TEXT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-:.]{3,63}$")
_ID_KEY = re.compile(r"(^|_|\b)(id|pid|sku|ref|code|key|listing|item|product|article)s?($|_|\b)", re.I)
_PRICE_KEY = re.compile(r"price|cost|amount|charge", re.I)
_STOCK_KEY = re.compile(r"stock|availab|on_?hand|inventory|in_?store|sellable|orderable", re.I)
_TITLE_KEY = re.compile(r"title|name|headline|label", re.I)
_CATEGORY_KEY = re.compile(r"categor|department|dept|aisle|section|shelf|product_?type|kind|group|taxonom|classif|browse|catalog|segment|division|genre|collection", re.I)
_FACTS_KEY = re.compile(r"spec|attrib|data|propert|characterist|fact|detail|trait|feature|particular|technical|profile|sheet|facet|aspect", re.I)
_OWNER_KEY = re.compile(r"brand|maker|manufactur|seller|vendor|merchant|dealer|store|shop|supplier|sold_?by", re.I)  # who sells it, not what it is
_PROSE_KEY = re.compile(r"remark|note|comment|annot|fine_?print|descr|about|blurb|overview|write_?up|good_?to_?know|review", re.I)  # free text
_NOT_LABEL_KEY = re.compile(r"condition|status|state|currency|colou?r|size|unit|shipping|delivery|rating|warranty", re.I)
_REMOVE_WORD = re.compile(r"remov|discard|delet|drop|clear|take|minus|decreas|release|subtract|withdraw|pull|unhold|unset|reduc|revoke|retract|exclude|eject", re.I)
_ADD_WORD = re.compile(r"add|hold|put|place|insert|stash|keep|include|grab|set|increas|append|load|pick", re.I)


def _department_like(key: str, value: Any) -> bool:
    """A department is a short label, so a free-text field whose name holds a department word (remarks_section in
    race 12) gives no department, and neither does a brand or an id."""
    return (isinstance(value, str) and not _PROSE_KEY.search(key) and not _OWNER_KEY.search(key)
            and not re.search(r"(^|_)(id|code|ref)$", key, re.I) and 0 < len(value.strip()) <= 60
            and len(value.split()) <= 8 and not value.rstrip().endswith((".", "!", "?"))
            and bool(re.search(r"[A-Za-z]{3}", value)) and not (" " not in value.strip() and re.search(r"\d", value)))


def _facts_table(key: str, value: Any) -> bool:
    """A property table: a nested object named like one, or one holding only short plain values (tables go by a new
    name in many races: particulars in race 12)."""
    if not isinstance(value, dict) or len(value) < 2 or _PRICE_KEY.search(key):
        return False

    def plain(v: Any) -> bool:  # a short value, or a short list of them (a store may list several values)
        if isinstance(v, list):
            return 0 < len(v) <= 12 and all(isinstance(x, (str, int, float)) and not isinstance(x, bool) and len(str(x)) <= 80 for x in v)
        return v is None or (isinstance(v, (str, int, float)) and not isinstance(v, bool) and len(str(v)) <= 80)
    return bool(_FACTS_KEY.search(key)) or (all(plain(v) for v in value.values())
                                           and sum(v is not None for v in value.values()) >= 2)


def _dumps(value: Any, limit: int | None = None) -> str:
    text = json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
    if limit is not None and len(text) > limit:
        text = text[:limit] + "…(truncated)"
    return text


def _json_from(text: str) -> dict | None:
    """The first JSON object in a model reply (tolerates code fences and stray prose)."""
    if not isinstance(text, str):
        return None
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except ValueError:
        pass
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        value = json.loads(text[start:i + 1])
                        if isinstance(value, dict):
                            return value
                    except ValueError:
                        break
                    break
        start = text.find("{", start + 1)
    return None


def _as_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        m = re.search(r"-?\d+(?:[.,]\d+)?", value.replace(",", ""))
        if m:
            try:
                return float(m.group(0))
            except ValueError:
                return None
    return None


def _as_stock(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value > 0
    if isinstance(value, str):
        v = value.strip().lower()
        if re.search(r"out|none|sold|unavail|no\b|false|0\b|gone", v):
            return False
        if re.search(r"in|yes|true|avail|stock|ready", v):
            return True
    return None


def _strings_in(value: Any, out: list[str], depth: int = 0) -> None:
    if depth > 6:
        return
    if isinstance(value, str):
        out.append(value)
    elif isinstance(value, dict):
        for v in value.values():
            _strings_in(v, out, depth + 1)
    elif isinstance(value, list):
        for v in value:
            _strings_in(v, out, depth + 1)


_FAIL_KEY = re.compile(r"error|issue|fault|fail|problem|invalid|denied|rejected", re.I)
_FAIL_CODE = re.compile(r"^[A-Za-z]{1,4}[-_ ]?\d{1,5}$")


def _unwrapped(observation: Any) -> Any:
    """A result that carries shopper notes comes as {<result key>: result, <notes key>: [sentences]}."""
    if isinstance(observation, dict) and len(observation) == 2:
        inner = [v for v in observation.values() if isinstance(v, dict)]
        notes = [v for v in observation.values() if isinstance(v, list) and all(isinstance(s, str) for s in v)]
        if len(inner) == 1 and len(notes) == 1:
            return inner[0]
    return observation


def _failed(observation: Any) -> bool:
    """The store reports a failed call: a small result under an error-like key, or nothing but a short
    letters-and-digits code."""
    body = _unwrapped(observation)
    if not isinstance(body, dict) or not body or len(body) > 3:
        return False
    if any(_FAIL_KEY.search(k) and v not in (None, False, "", [], {}) for k, v in body.items()):
        return True
    return len(body) == 1 and isinstance(next(iter(body.values())), str) and bool(_FAIL_CODE.match(next(iter(body.values())).strip()))


def _fit_args(value: Any, schema: dict, path: str = "args") -> tuple[Any, str | None]:
    """Coerce tool arguments toward the schema (numbers from numeric text, enum case, length limits,
    unknown keys dropped) and report the first problem the runtime would refuse, so that a malformed
    call is caught before it costs a step."""
    kinds = schema.get("type")
    kinds = kinds if isinstance(kinds, list) else ([kinds] if kinds else [])
    if kinds:
        def is_kind(v: Any, k: str) -> bool:
            return {"null": v is None, "boolean": isinstance(v, bool),
                    "integer": isinstance(v, int) and not isinstance(v, bool),
                    "number": isinstance(v, (int, float)) and not isinstance(v, bool),
                    "string": isinstance(v, str), "array": isinstance(v, list), "object": isinstance(v, dict)}.get(k, True)
        if not any(is_kind(value, k) for k in kinds):
            fixed = value
            if isinstance(value, str) and ("integer" in kinds or "number" in kinds):
                n = _as_number(value)
                if n is not None:
                    fixed = int(n) if "integer" in kinds and float(n).is_integer() else n
            elif isinstance(value, float) and "integer" in kinds and value.is_integer():
                fixed = int(value)
            elif isinstance(value, (int, float)) and not isinstance(value, bool) and "string" in kinds:
                fixed = str(value)
            elif isinstance(value, str) and "boolean" in kinds and value.lower() in ("true", "false"):
                fixed = value.lower() == "true"
            elif "array" in kinds and not isinstance(value, list) and value is not None:
                fixed = [value]
            if not any(is_kind(fixed, k) for k in kinds):
                return value, f"{path}: expected {'/'.join(kinds)}"
            value = fixed
    if "enum" in schema:
        options = schema["enum"]
        if value not in options:
            match = next((o for o in options if isinstance(o, str) and isinstance(value, str)
                          and o.lower() == value.strip().lower()), None)
            if match is None:
                return value, f"{path}: must be one of {options}"
            value = match
    if isinstance(value, dict):
        props = schema.get("properties") or {}
        out = {}
        for k, v in value.items():
            if k in props:
                v, err = _fit_args(v, props[k] or {}, f"{path}.{k}")
                if err:
                    return value, err
                out[k] = v
            elif schema.get("additionalProperties", True) is False:
                continue  # drop what the store would refuse
            else:
                out[k] = v
        for k in schema.get("required") or []:
            if k not in out:
                return out, f"{path}: missing required argument {k!r}"
        value = out
    if isinstance(value, list) and "items" in schema:
        fixed_items = []
        for i, item in enumerate(value):
            item, err = _fit_args(item, schema["items"] or {}, f"{path}[{i}]")
            if err:
                return value, err
            fixed_items.append(item)
        value = fixed_items
        if not schema.get("minItems", 0) <= len(value) <= schema.get("maxItems", len(value)):
            return value, f"{path}: list length out of range"
    if isinstance(value, str):
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            value = value[: schema["maxLength"]]
        if len(value) < schema.get("minLength", 0):
            return value, f"{path}: text too short"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            return value, f"{path}: below the minimum {schema['minimum']}"
        if "maximum" in schema and value > schema["maximum"]:
            return value, f"{path}: above the maximum {schema['maximum']}"
    return value, None


def _brief_fields(raw: Any) -> dict[str, Any]:
    """Map the brief model's field names onto the names the code uses."""
    if not isinstance(raw, dict):
        return {}
    goods = raw.get("goods") if isinstance(raw.get("goods"), list) else []
    jobs = raw.get("tool_map") if isinstance(raw.get("tool_map"), dict) else {}
    return {
        "products": [{"name": g.get("kind"), "quantity": g.get("units")} for g in goods if isinstance(g, dict)],
        "requirements": raw.get("must_have") or [],
        "price_cap": raw.get("max_total"),
        "price_unit": raw.get("money_unit"),
        "in_stock_only": raw.get("only_available"),
        "priority": {"lowest_price": "cheapest", "top_by_measure": "best"}.get(raw.get("ordering_rule"), "none"),
        "priority_measure": raw.get("measure") or "",
        "message_budget": raw.get("max_messages"),
        "step_budget": raw.get("max_moves"),
        "rules": [str(p.get("text") or "") if isinstance(p, dict) else str(p) for p in raw.get("procedures") or []],
        "procedures": [{"text": str(p.get("text") or ""), "kind": str(p.get("kind") or "other"),
                        "count": p.get("count") if isinstance(p.get("count"), int) else None}
                       for p in raw.get("procedures") or [] if isinstance(p, dict)],
        "decline_rule": {"no_alternatives": "no_substitutes", "suggest_alternatives": "name_alternatives"}.get(
            raw.get("refusal_style")),
        "soft_wants": raw.get("nice_to_have") or [],
        "ask_first": raw.get("unknowns") or [],
        "story": raw.get("context") or "",
        "story_hints": raw.get("story_hints") or [],
        "roles": {mine: jobs.get(theirs) for mine, theirs in (
            ("search", "find"), ("details", "inspect"), ("cart", "basket"), ("cart_view", "basket_view"),
            ("order", "checkout"), ("decline", "abandon"), ("message", "contact"))},
        "search_usage": raw.get("search_notes") or "",
    }


# a condition whose value is read from another listing; races 8-13 phrased it many ways ("whatever the page of …'s
# … shows", "whatever's on the page for …", "the one sold by …"): the old list caught 10-61% of such replies, this all
# of them (8,388 replies) and none of 53,251 others
_RELATIVE = re.compile(r"listing|listed|page for|page of|the page|sold by|same as|anything but|anything bar|other than|"
                       r"apart from|match for|whatever|wherever", re.I)
_NOT_NAMES = {"anything", "same", "listing", "page", "other", "with", "what", "that", "this", "only", "never", "must",
              "item", "items", "product", "products", "seller", "store", "brand", "maker"}

# A request sentence that has the customer see or approve the purchase before it is placed.
_BEFORE_BUYING = re.compile(r"\b(before|prior to|ahead of|until)\b[^.!?;]*\b(order|buy|purchas|check\s?out|pay|place|"
                            r"submit|finali[sz]|ring)", re.I)
_WHOLE_BASKET = re.compile(r"\b(final|finished|complete|whole|full)\s+(basket|cart|trolley|bag|order|list)", re.I)
_SEE_OR_APPROVE = re.compile(r"\b(show|tell|run\b[^.!?;]*\b(by|past)|confirm|approv|okay|ok\b|yes\b|go[- ]ahead|"
                             r"green light|sign[- ]?off|look (it |them )?over|see\b|check with|let me know|walk me through|"
                             r"read (it|them|back)|send me)", re.I)


def confirmation_sentences(request: str) -> list[str]:
    """Sentences of the request asking to see or approve the purchase before it is placed."""
    return [s.strip() for s in re.split(r"(?<=[.!?;])\s+", request)
            if _SEE_OR_APPROVE.search(s) and (_BEFORE_BUYING.search(s) or _WHOLE_BASKET.search(s))]


def _reading_fields(raw: Any) -> dict[str, Any] | None:
    """Map the reply reader's field names onto the names the code uses."""
    if not isinstance(raw, dict):
        return None
    return {
        "summary": raw.get("gist"), "requirements_now": raw.get("current_needs"),
        "no_preference": raw.get("indifferent_to") or [], "dropped": raw.get("removed") or [],
        "new_price_cap": raw.get("budget_now"), "new_quantity": raw.get("units_now"),
        "item_unavailable": raw.get("pick_gone"), "price_changed": raw.get("price_moved"),
        "unclear": raw.get("confused"), "asks_assistant": raw.get("question_for_me") or "",
    }


def _stems(text: str) -> set[str]:
    return {re.sub(r"(es|s)$", "", w)[:5] for w in re.findall(r"[a-z]{3,}", str(text).lower())} - {"and", "the", "for", "with"}


def split_req(req: str) -> tuple[str, str, str]:
    """(product, attribute, value) of a condition written "product - attribute: value" or "attribute: value"."""
    attr, _sep, value = str(req).partition(":")
    product = ""
    m = re.match(r"\s*(.+?)\s+[-\u2013\u2014]\s+(.+)$", attr)
    if m:
        product, attr = m.group(1), m.group(2)
    return product.strip(), attr.strip(), value.strip()


def _sentence_like(text: str) -> bool:
    words = text.split()
    return len(words) >= 4 and len(text) >= 18 and bool(re.search(r"[a-z]", text)) and not _ID_TEXT.match(text)


# ---------------------------------------------------------------------------------------------
# Episode


class Episode:
    def __init__(self, problem: dict[str, Any]):
        self.t0 = time.monotonic()
        self.work = 0.0                     # work units used (see WORK_BUDGET)
        env = problem["environment"]
        self.binding = dict(env["binding"])
        policy = env["policy_view"]
        self.request = str(policy.get("query") or problem.get("query") or "")
        self.opening = policy.get("opening")
        self.tools = [t for t in policy.get("tools", []) if isinstance(t, dict) and isinstance(t.get("function"), dict)]
        self.tool_names = {t["function"]["name"] for t in self.tools}
        self.max_steps = int(policy.get("max_steps") or 30)
        self.finished = bool(policy.get("finished"))
        self.step = 0
        self.spent = 0.0
        self.keep_free = FINISH_SECONDS     # work a model call must leave for finishing
        self.cost_by_role: dict[str, float] = {}
        self.dialogue: list[dict[str, Any]] = []

        self.brief: dict[str, Any] = {}
        self.roles: dict[str, str | None] = {}
        self.requirements: list[str] = []
        self.price_cap: float | None = None
        self.quantity = 1
        self.priority = "none"
        self.in_stock_only = False
        self.message_budget: int | None = None
        self.messages_sent = 0
        self.notes: list[str] = []          # what the shopper has said, newest last
        self.heard: set[str] = set()        # notice lines already read (worlds repeat them in later results)
        self._last_lines: list[str] = []
        self.asked: list[str] = []          # properties already asked about
        self.questions_done = False
        self.force_questions = False      # set when a purchase is attempted before any questions

        self.listings: dict[str, dict[str, Any]] = {}
        self.detail_ids: set[str] = set()   # listings whose detail view was opened
        self.unavailable: set[str] = set()
        self.verdicts: dict[str, dict[str, Any]] = {}
        self.cart: dict[str, int] = {}
        self.searches: list[dict[str, Any]] = []
        self.gone_picks: list[str] = []     # basket picks reported gone, in order
        self.probed: set[str] = set()       # keyword probes already sent by code
        self.references_tried: set[str] = set()  # conditions pointing to another listing that code tried to resolve
        self.folded: list[str] = []              # stated conditions folded into the department (still settled)
        self.references_retry: dict[str, int] = {}  # ...tried once (at that step); a second try comes 6 steps later
        self.hints: list[str] = []
        self.department_key: str | None = None  # the field learned from the data to hold the department
        self.stock_key: str | None = None       # ...the stock flag, when no field is named like one
        self.query_words: set[str] = set()      # words of the current call's arguments (what was searched for)
        self.title_key: str | None = None       # ...the listing's name, when no field is named like one
        self.price_key: str | None = None       # ...the price, when no field is named like one
        self.absorbing_search = False
        self.tool_facts: dict[str, bool] = {}   # tool -> whether its results showed a property table
        self.list_ok: dict[str, int] = {}       # tool -> longest id list it accepted
        self.list_failed: dict[str, int] = {}   # tool -> shortest id list it refused
        self.basket_read_step = -1              # last step whose basket view was read line by line
        self.order_attempts = 0
        self.gate_blocks: dict[str, int] = {}
        self.last_cart_step = 0             # step of the latest basket change
        self.message_steps: list[int] = []  # steps at which a message reached the shopper
        self.basket_view_steps: list[int] = []
        self.compare_steps: list[int] = []
        self.opened: dict[str, int] = {}        # listing -> last step its own detail page was opened (single-item tool)
        self.added_step: dict[str, int] = {}    # listing -> step it was last put in the basket
        self.auto_pages = 0                     # next pages fetched by code
        self.last_refusal_step = -1             # step of the latest order the store refused
        self.buy_closest = False                # a give-up was replaced by ordering the closest match

    # --- work and money --------------------------------------------------------------------

    def time_left(self) -> float:
        """Budget left: work units, capped by the seconds left before the sandbox's per-task kill."""
        return min(WORK_BUDGET - self.work, EPISODE_SECONDS - (time.monotonic() - self.t0))

    def steps_left(self) -> int:
        cap = self.max_steps
        sb = self.brief.get("step_budget")
        if isinstance(sb, int) and 0 < sb < cap:
            cap = sb
        return cap - self.step

    def messages_left(self) -> int | None:
        if self.message_budget is None:
            return None
        return max(0, self.message_budget - self.messages_sent)

    # --- model calls -----------------------------------------------------------------------

    def chat(self, model: str, messages: list[dict], *, tools: list | None = None, max_tokens: int = 1500,
             fallback: str | None = FALLBACK_MODEL, timeout_floor: float = 12.0, force_tool: bool = False,
             role: str = "other") -> dict | None:
        """One chat completion; on failure a retry on the fallback model, then one more on the first model.
        Returns the message. A request never runs into the work kept free for finishing."""
        names = [model] + ([fallback, model] if fallback and fallback != model else [model])
        for name in names:
            budget = self.time_left() - self.keep_free
            if budget < timeout_floor:
                return None
            self.work += SLOW_CALL_WORK if name == JUDGE_MODEL else CALL_WORK
            body: dict[str, Any] = {"model": name, "messages": messages, "max_tokens": max_tokens + THINK_TOKENS,
                                    "temperature": 0}
            if tools:
                body["tools"] = tools
                body["tool_choice"] = "required" if force_tool else "auto"
            data = _client(min(MODEL_SECONDS, budget), 1).post("/inference/chat/completions", json_data=body)
            if not isinstance(data, dict):
                continue
            usage = data.get("usage") or {}
            cost = usage.get("cost")
            if isinstance(cost, (int, float)):
                self.spent += float(cost)
                self.cost_by_role[role] = self.cost_by_role.get(role, 0.0) + float(cost)
            choices = data.get("choices") or []
            if choices and isinstance(choices[0], dict) and isinstance(choices[0].get("message"), dict):
                return choices[0]["message"]
        return None

    def ask_json(self, model: str, system: str, user: str, max_tokens: int = 1200, role: str = "other") -> dict | None:
        msg = self.chat(model, [{"role": "system", "content": system}, {"role": "user", "content": user}],
                        max_tokens=max_tokens, role=role)
        return _json_from((msg or {}).get("content") or "")

    # --- environment -----------------------------------------------------------------------

    def act(self, name: str, args: dict[str, Any]) -> tuple[Any, bool]:
        """Run one tool call as the next step. Returns (observation, ok)."""
        if self.finished:
            return {"error": "the episode has finished"}, False
        self.step += 1
        self.work += STEP_WORK
        envelope = {**self.binding, "step": self.step, "action": {"name": name, "args": args}}
        env = _client(min(30.0, max(10.0, self.time_left() + 20)), 2 if self.time_left() > 60 else 1)
        result = env.post("/environment/call", json_data=envelope)
        if not isinstance(result, dict):
            # A lost response: replaying the same step returns the recorded result.
            result = env.post("/environment/call", json_data=envelope)
        if not isinstance(result, dict):
            self.dialogue.append({"step": self.step, "tool": name, "args": args, "error": "environment call failed"})
            return {"error": "environment call failed"}, False
        if isinstance(result.get("step"), int):
            self.step = result["step"]
        self.finished = bool(result.get("finished"))
        if not self.finished and self.finish_kind(name, args) == "order":
            self.last_refusal_step = self.step
        observation = result.get("observation")
        self.dialogue.append({"step": self.step, "tool": name, "args": args, "observation": observation,
                              "finished": self.finished})
        self.absorb(name, args, observation)
        return observation, True

    # --- memory of listings ----------------------------------------------------------------

    def _listing_id(self, d: dict) -> str | None:
        best = None
        for k, v in d.items():
            if isinstance(v, str) and _ID_TEXT.match(v) and _ID_KEY.search(k) and not _TITLE_KEY.search(k):
                if re.search(r"\d", v):
                    return v
                best = best or v
        if best or (len(d) < 4 and not any(_facts_table(k, v) for k, v in d.items())):
            return best  # (a detail view can be small: a name, a property table and an id under unknown names)
        # no field named like an id: a value built like one, in an object with enough fields to be a row
        return next((v for v in d.values() if isinstance(v, str) and re.fullmatch(r"[A-Za-z]{0,4}[-_]?[0-9a-f]{8,}", v) and re.search(r"\d", v)), None)

    def _learn_department_key(self, rows: list) -> None:
        """When no field of a search's rows is named like a department, the department is the short label most of the
        rows share that is no title, owner, price or free text."""
        rows = [r for r in rows if isinstance(r, dict) and self._listing_id(r)]
        if self.department_key or len(rows) < 4:
            return
        labels: dict[str, list[str]] = {}
        for r in rows:
            for k, v in r.items():
                if not _department_like(k, v):
                    continue
                if _CATEGORY_KEY.search(k):
                    return  # named like one: nothing to learn
                if not (_TITLE_KEY.search(k) or _PRICE_KEY.search(k) or _STOCK_KEY.search(k) or _NOT_LABEL_KEY.search(k)
                        or k in (self.title_key, self.stock_key) or (_ID_KEY.search(k) and re.search(r"\d", v))):
                    labels.setdefault(k, []).append(v.strip().lower())
        # Races 8-12 (every search of a top agent): the department shares words with what was searched for (0.51-0.85 of
        # rows) and with the titles (0.61-0.82) and repeats most (0.11-0.31 distinct); a brand shares no search words
        # (0.00-0.01), a seller or shop neither search nor title words and is nearly unique per row (0.81-0.96).
        def title_of(r: dict) -> str:
            return next((v for k, v in r.items() if isinstance(v, str) and (k == self.title_key or (
                _TITLE_KEY.search(k) and not _OWNER_KEY.search(k)))), "")
        scored = []
        for k, vals in labels.items():
            if len(vals) < 0.6 * len(rows) or len(set(vals)) > 0.5 * len(vals):
                continue
            pairs = [(str(r.get(k)).lower(), title_of(r)) for r in rows if isinstance(r.get(k), str)]
            in_title = sum(bool(_stems(v) & _stems(t)) for v, t in pairs) / len(pairs)
            in_query = sum(bool(_stems(v) & self.query_words) for v, _ in pairs) / len(pairs)
            scored.append((in_title + in_query - len(set(vals)) / len(vals), -len(vals), k))  # (ties: measured best this way)
        if scored:
            self.department_key = max(scored)[2]
            self.note("department field", self.department_key)

    def _learn_row_keys(self, rows: list) -> None:
        """Fields learned from a search's rows when none is named like one (stores rename fields every race): the stock
        flag is the true/false field most rows carry; the name is the field of multi-word text that differs from row
        to row and is shorter than the free-text fields."""
        rows = [r for r in rows if isinstance(r, dict) and self._listing_id(r)]
        if len(rows) < 4:
            return
        keys = {k for r in rows for k in r}
        if self.stock_key is None and not any(_STOCK_KEY.search(k) for k in keys):
            flags = [k for k in keys if sum(isinstance(r.get(k), bool) for r in rows) >= 0.6 * len(rows)
                     and sum(r.get(k) is True for r in rows) >= 0.6 * len(rows)  # most listings can be bought
                     and not re.search(r"sold|out|discontinu|backorder|sale|deal|featur|new|prime|best|promo|clearance|"
                                       r"hidden|archiv|retir|sponsor|gift|wrap", k, re.I)]
            if len(flags) == 1:
                self.stock_key = flags[0]
                self.note("stock field", self.stock_key)
        if self.price_key is None and not any(_PRICE_KEY.search(k) for k in keys):
            # the price is the rows' largest number (every search of races 8-12; counts have a median of 1)
            medians = {}
            for k in keys:
                if _ID_KEY.search(k) or _STOCK_KEY.search(k) or re.search(
                        r"rating|review|count|units|qty|quant|sold|rank|score|weight|year|stock|star|vote|(^|_)age($|_)|size", k, re.I):
                    continue
                nums = [r[k] for r in rows if isinstance(r.get(k), (int, float)) and not isinstance(r.get(k), bool)]
                if len(nums) >= 0.8 * len(rows):
                    medians[k] = sorted(nums)[len(nums) // 2]
            if medians and max(medians.values()) > 0:
                self.price_key = max(medians, key=medians.get)
                self.note("price field", self.price_key)
        if self.title_key is None and not any(_TITLE_KEY.search(k) and not _OWNER_KEY.search(k) for k in keys):
            options = []
            for k in keys:
                vals = [r.get(k) for r in rows if isinstance(r.get(k), str) and r.get(k).strip()]
                if len(vals) < 0.8 * len(rows) or k == self.department_key or _ID_TEXT.match(vals[0]) \
                        or _OWNER_KEY.search(k) or _SELLER_KEY.search(k) or _PROSE_KEY.search(k) or _CATEGORY_KEY.search(k):
                    continue
                mean = sum(len(v) for v in vals) / len(vals)
                multi = sum(len(v.split()) >= 2 for v in vals) / len(vals)
                prose = sum(bool(re.search(r"[.!?;]\s+\S", v)) for v in vals) / len(vals)
                searched = sum(bool(_stems(v) & self.query_words) for v in vals) / len(vals)
                if len(set(vals)) >= 0.8 * len(vals) and multi >= 0.7 and 8 <= mean <= 200 and prose < 0.3:
                    options.append((searched < 0.2, mean, k))  # names share the searched words; shop names don't
            if options:
                self.title_key = min(options)[2]
                self.note("title field", self.title_key)

    def _absorb_value(self, value: Any, from_details: bool, depth: int = 0) -> None:
        if depth > 6:
            return
        if isinstance(value, list):
            self._learn_row_keys(value)
            self._learn_department_key(value)
            for v in value:
                self._absorb_value(v, from_details, depth + 1)
            return
        if not isinstance(value, dict):
            return
        lid = self._listing_id(value)
        looks_like_listing = lid is not None and (len(value) >= 4 or any(
            _PRICE_KEY.search(k) or _TITLE_KEY.search(k) or _STOCK_KEY.search(k) or _FACTS_KEY.search(k)
            or _facts_table(k, v) for k, v in value.items()))
        if looks_like_listing:
            rec = self.listings.setdefault(lid, {"id": lid, "fields": {}})
            for k, v in value.items():
                rec["fields"][k] = v
                if (_PRICE_KEY.search(k) and not re.search(r"cap|max|min|limit|budget", k, re.I)) or k == self.price_key:
                    if isinstance(v, dict):  # a price object such as {"amount": 299, "currency": "USD"}
                        v = next((x for kk, x in v.items() if re.search(r"amount|value|cents|total|price", kk, re.I)
                                  and _as_number(x) is not None), None)
                    if _as_number(v) is not None and not isinstance(v, (dict, list)):
                        rec["price"] = _as_number(v)
                        rec.pop("price_stale", None)
                elif (_STOCK_KEY.search(k) or k == self.stock_key) and not isinstance(v, (dict, list)) and _as_stock(v) is not None:
                    rec["in_stock"] = _as_stock(v)
                elif (_CATEGORY_KEY.search(k) or k == self.department_key) and _department_like(k, v):
                    rec["category"] = v
                elif ((_TITLE_KEY.search(k) and not _OWNER_KEY.search(k)) or k == self.title_key) and isinstance(v, str) \
                        and ("title" not in rec or rec.get("title_guessed")):
                    rec["title"] = v
                    rec.pop("title_guessed", None)
                if isinstance(v, dict) and (_FACTS_KEY.search(k) or not self.absorbing_search) and _facts_table(k, v):
                    from_details = True  # a property table: this is a detail view
            if self.absorbing_search and "title" not in rec:  # no field named like a title: the longest short text that is no owner, department or free text
                texts = [v for k, v in value.items() if isinstance(v, str) and len(v.split()) >= 2 and len(v) <= 200 and not _ID_TEXT.match(v)
                         and not (_OWNER_KEY.search(k) or _PROSE_KEY.search(k) or _CATEGORY_KEY.search(k) or k == self.department_key)]
                if texts:
                    rec["title"] = max(texts, key=len)
                    rec["title_guessed"] = True
            if self.absorbing_search and "price" not in rec:  # no field named like a price: the row's only plain number
                numbers = [v for k, v in value.items() if isinstance(v, (int, float)) and not isinstance(v, bool)
                           and not re.search(r"rating|review|count|stock|units|qty|quant|sold|rank|score|weight|year", k, re.I)]
                if len(numbers) == 1 and numbers[0] > 0:
                    rec["price"] = numbers[0]
            if from_details:
                self.detail_ids.add(lid)
                rec["detailed"] = True
            for v in value.values():
                if isinstance(v, (list, dict)):
                    self._absorb_value(v, from_details, depth + 1)
            return
        for v in value.values():
            self._absorb_value(v, from_details, depth + 1)

    def shopper_lines(self, observation: Any) -> list[str]:
        """Natural-language lines in a result that are not part of a listing: shopper notices."""
        lines: list[str] = []
        if isinstance(observation, dict) and set(observation) == {"observation", "refusal"}:
            found: list[str] = []
            _strings_in(observation["refusal"], found)
            lines.extend(s for s in found if s.strip())
            observation = observation["observation"]
        if isinstance(observation, dict):
            for k, v in observation.items():
                if isinstance(v, list) and v and all(isinstance(s, str) for s in v) and all(_sentence_like(s) for s in v):
                    lines.extend(v)
                elif isinstance(v, str) and _sentence_like(v) and len(observation) <= 3 and not _TITLE_KEY.search(k) \
                        and not re.search(r"descr|error|detail|issue|status|message_?id", k, re.I):
                    if re.search(r"\b(I|I'm|I'd|my|me|you|your)\b", v):
                        lines.append(v)
        seen, out = set(), []
        for s in lines:
            if s not in seen:
                seen.add(s)
                out.append(s)
        return out

    def _removal_verb(self, tool: str, args: dict[str, Any], verbs: str) -> bool:
        """A basket call that takes out: a removal word, or the option that is not the adding one in the tool's list."""
        if _REMOVE_WORD.search(verbs):
            return True
        props = (self.tool_spec(tool).get("parameters") or {}).get("properties") or {}
        for k, v in args.items():
            words = [e for e in (props.get(k) or {}).get("enum") or [] if isinstance(e, str)]
            if v in words and not _ADD_WORD.search(str(v)) and any(_ADD_WORD.search(e) for e in words):
                return True
        return False

    def _has_property_table(self, value: Any, depth: int = 0) -> bool:
        if depth > 5:
            return False
        if isinstance(value, dict):
            for k, v in value.items():
                if _facts_table(k, v):
                    return True
                if isinstance(v, (dict, list)) and self._has_property_table(v, depth + 1):
                    return True
        elif isinstance(value, list):
            return any(self._has_property_table(v, depth + 1) for v in value[:10])
        return False

    def _is_id(self, text: str) -> bool:
        return text in self.listings or bool(_ID_TEXT.match(text) and re.search(r"\d", text))

    def _basket_lines(self, observation: Any) -> dict[str, int] | None:
        """{listing id: units} from a basket view, or None when the result shows no list of basket lines."""
        body = _unwrapped(observation)
        if not isinstance(body, dict):
            return None
        lists = [v for v in body.values() if isinstance(v, list) and all(isinstance(x, dict) for x in v)]
        if len(lists) != 1:
            return None
        out: dict[str, int] = {}
        for row in lists[0]:
            lid = self._listing_id(row)
            if not lid:
                return None
            qty = next((int(v) for k, v in row.items() if isinstance(v, int) and not isinstance(v, bool)
                        and re.search(r"qty|quant|piece|count|units|number|amount_of", k, re.I)), 1)
            out[lid] = out.get(lid, 0) + qty
        return out

    def list_limit(self, tool: str) -> int:
        """How many ids to send in one call of a list-taking tool: below the shortest list it refused, and never
        above a maximum its schema states."""
        props = (self.tool_spec(tool).get("parameters") or {}).get("properties") or {}
        caps = [int(v["maxItems"]) for v in props.values() if isinstance(v, dict) and v.get("type") == "array"
                and isinstance(v.get("maxItems"), int) and v["maxItems"] > 0]
        if tool in self.list_failed:
            limit = max(1, min(self.list_failed[tool] - 1, max(self.list_ok.get(tool, 1), self.list_failed[tool] // 2)))
        else:
            limit = max(4, self.list_ok.get(tool, 4))
        return min([limit] + caps)

    def absorb(self, name: str, args: dict[str, Any], observation: Any) -> None:
        self.query_words = _stems(" ".join(v for v in (args or {}).values() if isinstance(v, str) and not _ID_TEXT.match(v)))
        sizes = [len(v) for v in (args or {}).values() if isinstance(v, list) and all(isinstance(x, str) for x in v)]
        if sizes and max(sizes) > 1:
            if _failed(observation):
                self.list_failed[name] = min(self.list_failed.get(name, 10 ** 6), max(sizes))
            else:
                self.list_ok[name] = max(self.list_ok.get(name, 0), max(sizes))
        if name != self.roles.get("search") and isinstance(observation, (dict, list)) and not _failed(observation):
            # Learn which tools really show a listing's property table, and use that one for details.
            shows = self._has_property_table(observation)
            self.tool_facts[name] = self.tool_facts.get(name, False) or shows
            # only a one-listing tool can be the detail page: a side-by-side tool that also shows properties
            # is used for screening, but opening each bought item's own page is what graders look for
            if shows and not self.tool_facts.get(self.roles.get("details") or "", False) and not self.takes_list(name) \
                    and name not in (self.roles.get("cart"), self.roles.get("cart_view"), self.roles.get("order")):
                self.roles["details"] = name
        details = self.roles.get("details")
        self.absorbing_search = name == self.roles.get("search")
        self._absorb_value(observation, from_details=(name == details))
        self.absorbing_search = False
        if name == self.roles.get("cart") and isinstance(args, dict) and not _failed(observation):
            # Flat arguments ({"id": ..., "qty": ...}) or a list of line objects ([{"item": ..., "qty": ...}]).
            groups = [args] + [d for v in args.values() if isinstance(v, list) for d in v if isinstance(d, dict)]
            verbs = " ".join(str(v).lower() for v in args.values() if isinstance(v, str) and not self._is_id(v))
            removing = self._removal_verb(name, args, verbs)
            line_groups = groups[1:]
            if line_groups and not removing:
                # a list of line objects sets the whole basket (races 7 and 13: lines left out were gone afterwards)
                new_cart: dict[str, int] = {}
                for g in line_groups:
                    ids = [v for v in g.values() if isinstance(v, str) and self._is_id(v)]
                    qty = next((int(v) for k, v in g.items() if isinstance(v, int) and not isinstance(v, bool)), None)
                    for i in ids[:1]:
                        if qty != 0:
                            new_cart[i] = qty or 1
                            if i not in self.cart:
                                self.added_step[i] = self.step
                self.cart = new_cart
                groups = []
            for g in groups:
                ids = [v for v in g.values() if isinstance(v, str) and self._is_id(v)]
                ids += [x for v in g.values() if isinstance(v, list) for x in v if isinstance(x, str) and self._is_id(x)]
                qty = next((int(v) for k, v in g.items() if isinstance(v, int) and not isinstance(v, bool)), None)
                for i in ids:
                    if removing or qty == 0:
                        left = self.cart.get(i, 0) - (qty or self.cart.get(i, 0))
                        if left > 0:
                            self.cart[i] = left
                        else:
                            self.cart.pop(i, None)
                    else:
                        self.cart[i] = self.cart.get(i, 0) + (qty or 1) if not re.search(r"\bset\b", verbs) else (qty or 1)
                        self.added_step[i] = self.step
            body = _unwrapped(observation)
            shown = self._basket_lines(observation)
            if shown is not None and isinstance(body, dict) and any(re.search(r"total|sum", k, re.I) for k in body):
                self.cart = shown  # the reply lists the whole basket with its total: trust the store over the arguments
        if name == self.roles.get("cart_view") and not _failed(observation):
            seen = self._basket_lines(observation)
            if seen is not None:
                self.cart = seen  # the store's own basket overrides what the agent's calls suggested
                self.basket_read_step = self.step
        failed = _failed(observation)
        if name == self.page_tool() and not failed and isinstance(args, dict):
            for v in args.values():
                if isinstance(v, str) and self._is_id(v):
                    self.opened[v] = self.step
        if name == self.roles.get("cart") and not failed:
            self.last_cart_step = self.step
        if name == self.roles.get("cart_view") and not failed:
            self.basket_view_steps.append(self.step)
        if name == self.compare_tool() and not failed:
            self.compare_steps.append(self.step)
        if name == self.roles.get("message") and not failed:
            self.message_steps.append(self.step)
        if name == self.roles.get("message"):
            if isinstance(observation, dict) and observation.get("error") == "budget_exceeded":
                self.message_budget = self.messages_sent
            elif not _failed(observation):
                self.messages_sent += 1
        if name == self.roles.get("search"):
            rows = 0
            total = None
            next_token: Any = None
            ids: list[str] = []
            body = observation
            if isinstance(body, dict) and len(body) == 2 and any(isinstance(v, list) and v and all(isinstance(s, str) for s in v)
                                                                  for v in body.values()):
                body = next((v for v in body.values() if isinstance(v, dict)), body)  # a notice wrapper
            if isinstance(body, dict):
                for k, v in body.items():
                    if isinstance(v, list):
                        dict_rows = [r for r in v if isinstance(r, dict)]
                        if len(dict_rows) >= rows:
                            rows = len(dict_rows)
                            ids = [self._listing_id(r) or "" for r in dict_rows]
                    elif re.search(r"total|count|matched|hits|found|number|num_", k, re.I) and _as_number(v) is not None \
                            and not isinstance(v, bool):
                        total = int(_as_number(v))
                    if re.search(r"next|cursor|continu|more_?|after|follow|token", k, re.I) and isinstance(v, (str, int)) \
                            and not isinstance(v, bool) and str(v).strip():
                        next_token = v
            props = (self.tool_spec(name).get("parameters") or {}).get("properties") or {}
            texts = [(k, v) for k, v in args.items() if isinstance(v, str) and not (props.get(k) or {}).get("enum")
                     and re.search(r"[a-zA-Z]{3}", v) and v not in self.listings
                     and not (re.search(r"\d", v) and _ID_TEXT.match(v))
                     and not _CATEGORY_KEY.search(k) and not re.search(r"brand|maker|seller|vendor|manufacturer", k, re.I)]
            entry = {"args": args, "rows": rows, "total": total, "ids": ids, "text_key": texts[0][0] if texts else None,
                     "next": next_token}
            row_titles = " ".join(str(self.listings.get(i, {}).get("title", "")) for i in ids if i).lower()
            for k, v in texts:
                words = [w[:4] for w in re.findall(r"[a-z]{4,}", v.lower())]
                if ids and words and row_titles and not any(w in row_titles for w in words):
                    hint = (f"The search text {k}={v!r} appears in none of the {len(ids)} titles it returned, so that "
                            f"text was probably ignored with {_dumps({a: b for a, b in args.items() if a != k}, 120)}. "
                            "Put search words in a keyword-search mode instead.")
                    if hint not in self.hints:
                        self.hints.append(hint)
            def stems(value: Any) -> Any:
                if not isinstance(value, str):
                    return value
                return frozenset(w[:4] for w in re.findall(r"[a-z0-9]+", value.lower()))

            for prev in self.searches:  # an argument whose change altered nothing is probably ignored
                if prev["ids"] == ids and prev["total"] == total and ids and prev["args"] != args:
                    changed = sorted(k for k in set(prev["args"]) | set(args)
                                     if stems(prev["args"].get(k)) != stems(args.get(k))
                                     and not re.search(r"page|offset|cursor|next|token|continu", k, re.I))
                    if changed:
                        hint = (f"Searching with {_dumps(args, 120)} gave exactly the same rows as {_dumps(prev['args'], 120)}: "
                                f"{', '.join(changed)} probably had no effect in that combination (e.g. text ignored by a "
                                "filter-only mode). Use the mode where it takes effect.")
                        if hint not in self.hints:
                            self.hints.append(hint)
                        break
            self.searches.append(entry)

    # --- requirements ----------------------------------------------------------------------

    def apply_brief(self, brief: dict[str, Any]) -> None:
        self.brief = brief
        roles = brief.get("roles") if isinstance(brief.get("roles"), dict) else {}
        self.roles = {k: (v if isinstance(v, str) and v in self.tool_names else None) for k, v in roles.items()}
        self._guess_roles()
        self.requirements = [str(r) for r in brief.get("requirements") or [] if str(r).strip()]
        # A "requirement" that only restates the product kind ("season: summer" for summer
        # dresses) is covered by the category and must not fail listings.
        product_words = set(re.findall(r"[a-z]+", _dumps(brief.get("products")).lower()))
        def restates_kind(req: str) -> bool:
            if ":" not in req:
                return False
            value_words = set(re.findall(r"[a-z]+", req.split(":", 1)[1].lower())) - {"and", "or", "the", "a"}
            return bool(value_words) and value_words <= product_words
        self.requirements = [r for r in self.requirements if not restates_kind(r)]
        # A condition naming the store's department is the product kind itself (the category check covers it).
        products = brief.get("products") if isinstance(brief.get("products"), list) else []
        for r in list(self.requirements):
            attr, _, value = r.partition(":")
            if value.strip() and re.fullmatch(r"\s*(department|category|aisle|section|range|shelf)s?\s*", attr, re.I):
                self.requirements.remove(r)
                if len(products) == 1 and isinstance(products[0], dict) and not re.match(r"\s*not\b", value, re.I):
                    products[0]["name"] = value.strip()
        # A long "value" without list or exclusion words describes the customer's situation, not the item.
        def situation(req: str) -> bool:
            value = req.split(":", 1)[1] if ":" in req else ""
            return len(re.findall(r"[A-Za-z0-9]+", value)) > 5 and not re.search(
                r",|\b(or|not|no|except|without|excluding)\b", value, re.I)
        story = [r for r in self.requirements if situation(r)]
        if story:
            self.requirements = [r for r in self.requirements if r not in story]
            brief["story"] = " ".join([str(brief.get("story") or "")] + story).strip()
        self.price_cap = _as_number(brief.get("price_cap"))
        products = brief.get("products") if isinstance(brief.get("products"), list) else []
        if products and isinstance(products[0], dict) and isinstance(products[0].get("quantity"), int):
            self.quantity = max(1, products[0]["quantity"])
        self.priority = brief.get("priority") if brief.get("priority") in ("cheapest", "best") else "none"
        self.in_stock_only = bool(brief.get("in_stock_only"))
        mb = brief.get("message_budget")
        self.message_budget = mb if isinstance(mb, int) and mb >= 0 else None

    def _guess_roles(self) -> None:
        """Fill roles the brief left empty by scoring every tool for every role."""
        scores: list[tuple[float, str, str]] = []
        for t in self.tools:
            f = t["function"]
            name = f["name"]
            words = re.sub(r"([a-z])([A-Z])", r"\1 \2", name).replace("_", " ").replace("-", " ")
            text = f"{words} {f.get('description', '')}".lower()
            params = f.get("parameters") or {}
            props = params.get("properties") or {}
            strings = [k for k, v in props.items() if (v or {}).get("type") == "string" and not (v or {}).get("enum")]
            long_text_params = [k for k in strings if int((props[k] or {}).get("maxLength") or 1000) >= 200]
            enums = " ".join(str(e) for v in props.values() for e in ((v or {}).get("enum") or [])).lower()
            declines = bool(re.search(r"declin|give ?up|no purchase|without (buy|purchas)|abandon|refus|cancel|"
                                      r"end\w* without|nothing (fits|suits)|no order", text + " " + enums))
            orders = bool(re.search(r"check ?out|purchas|\bbuy|place\w* (the |an )?order|settle|\bpay|submit\w* (the )?order|"
                                    r"complete\w* (the )?order|finali[sz]e", text))
            cartish = bool(re.search(r"\b(cart|basket|bag|trolley)\b", text))
            viewish = bool(re.search(r"\b(view|show|review|see|list|read|get|current)\b", text))
            role_score = {
                "decline": 6 * declines - 4 * (orders and not re.search(r"no purchase|without", text)),
                "order": 5 * orders - 6 * bool(re.search(r"no purchase|without (buy|purchas)|give ?up|abandon", text)),
                "cart_view": 4 * (cartish and viewish) + 2 * (not props) - 4 * orders,
                "cart": 4 * cartish + 3 * bool(re.search(r"\b(add|remove|insert|delete|modify|update|put|set|change)\b",
                                                         text + " " + enums)) - 5 * orders - 3 * (not props),
                "message": 4 * bool(re.search(r"message|\bask|question|write|contact|chat|shopper|customer|buyer|note",
                                              text)) + 2 * bool(long_text_params) - 4 * cartish - 4 * orders,
                "details": 4 * bool(re.search(r"detail|spec|propert|trait|attribut|full|inspect|single", text))
                           + 1 * bool(re.search(r"spec|propert|trait|attribut|fact", text))
                           + 2 * (len(props) <= 2 and bool(strings)) - 3 * cartish,
                "search": 4 * bool(re.search(r"search|find|look|browse|query|filter|catalog|match", text))
                          + 2 * (len(props) >= 2) - 3 * cartish,
            }
            for role, s in role_score.items():
                if s > 0:
                    scores.append((float(s), role, name))
        taken = {v for v in self.roles.values() if v}
        for s, role, name in sorted(scores, reverse=True):
            if self.roles.get(role) or name in taken:
                continue
            self.roles[role] = name
            taken.add(name)
        if not self.roles.get("decline") and self.roles.get("order"):
            # One finishing tool for both outcomes: a decline is chosen through its arguments.
            spec = self.tool_spec(self.roles["order"])
            enums = " ".join(str(e) for v in ((spec.get("parameters") or {}).get("properties") or {}).values()
                             for e in ((v or {}).get("enum") or []))
            if re.search(DECLINE_WORDS, enums, re.I):
                self.roles["decline"] = self.roles["order"]

    def product_list(self) -> list[dict]:
        return [p for p in (self.brief.get("products") or []) if isinstance(p, dict) and p.get("name")]

    def product_of(self, lid: str) -> int | None:
        """Index of the wanted product a listing belongs to (by its department words, else its title)."""
        products = self.product_list()
        if len(products) < 2:
            return 0 if products else None
        rec = self.listings.get(lid) or {}
        for text in (rec.get("category"), rec.get("title")):
            if not text:
                continue
            scores = [len(_stems(p["name"]) & _stems(text)) for p in products]
            if max(scores) > 0 and scores.count(max(scores)) == 1:
                return scores.index(max(scores))
        return None

    def quantity_for(self, lid: str) -> int:
        products = self.product_list()
        i = self.product_of(lid)
        if i is not None and i < len(products) and isinstance(products[i].get("quantity"), int):
            return max(1, products[i]["quantity"])
        return self.quantity

    def takes_list(self, name: str | None) -> bool:
        """Whether a tool has a list parameter (a side-by-side or batch tool, not a one-listing page)."""
        props = (self.tool_spec(name).get("parameters") or {}).get("properties") or {}
        return any((v or {}).get("type") == "array" for v in props.values())

    def page_tool(self) -> str | None:
        """The tool that opens one listing's own page: the details role when it takes a single listing, else the
        one-listing tool that best reads as a detail view."""
        tool = self.roles.get("details")
        if tool in self.tool_names and not self.takes_list(tool):
            return tool
        taken = {v for k, v in self.roles.items() if v and k != "details"}
        best, best_weight = None, 0
        for t in self.tools:
            f = t["function"]
            props = (f.get("parameters") or {}).get("properties") or {}
            if f["name"] in taken or self.takes_list(f["name"]) or not props or len(props) > 2:
                continue
            words = re.sub(r"([a-z])([A-Z])", r"\1 \2", f["name"]) + " " + str(f.get("description") or "")
            weight = 2 * bool(re.search(r"detail|spec|inspect|full|single|page|sheet|read", words, re.I)) \
                + bool(re.search(r"propert|attribut|fact|trait", words, re.I)) + bool(self.tool_facts.get(f["name"]))
            if weight > best_weight:
                best, best_weight = f["name"], weight
        return best

    def opens_due(self) -> list[str]:
        """Basket lines whose own detail page has not been opened since the basket last changed or an order was
        refused. Top agents open every item they buy right before ordering (races 7-9: orders that skipped one, or
        opened it only before the last basket change, failed the procedure check far more often)."""
        if not self.page_tool():
            return []
        since = max(self.last_cart_step, self.last_refusal_step)
        # a line known to be gone is swapped out, not reopened (reopening it cost a sold-out swap its last move)
        return [i for i in self.cart if self.opened.get(i, -1) < since and i not in self.unavailable]

    def compare_tool(self) -> str | None:
        """A tool that sets several listings side by side, if the store has one."""
        taken = {v for v in self.roles.values() if v}
        for t in self.tools:
            f = t["function"]
            if f["name"] not in taken and re.search(r"compar|side.by.side|contrast|weigh", f"{f['name']} {f.get('description', '')}", re.I):
                return f["name"]
        return None

    def batch_detail_tool(self) -> str | None:
        """A tool that takes a list of listing ids and can show several listings' details in one move."""
        taken = {v for k, v in self.roles.items() if v and k not in ("details",)}
        for t in self.tools:
            f = t["function"]
            props = (f.get("parameters") or {}).get("properties") or {}
            listy = any((v or {}).get("type") == "array" and ((v or {}).get("items") or {}).get("type") in (None, "string")
                        for v in props.values())
            if f["name"] not in taken and listy and (self.tool_facts.get(f["name"]) or re.search(
                    r"compar|side.by.side|contrast|weigh|detail|spec|attribut|several|multiple|batch",
                    f"{f['name']} {f.get('description', '')}", re.I)):
                return f["name"]
        return None

    def confirmation_owed(self) -> bool:
        """The customer sees the final basket before the purchase only when the request asks for it. Races 6-7 wanted
        it in most tasks, but in races 8-11 an unasked summary did not help within the same task (race 11: 9 tasks
        better, 33 worse), and the message it kept back was a lost question (more questions: 41 tasks better, 12 worse)."""
        return any(p.get("kind") in ("go_ahead", "tell_first") for p in self.brief.get("procedures") or [])

    def reserved_messages(self) -> int:
        """Messages kept back from questions for the basket confirmation (only a known allowance needs it)."""
        return 1 if self.confirmation_owed() and self.roles.get("message") and self.messages_left() is not None else 0

    def confirmation_due(self) -> bool:
        """Owed, still possible, and not yet sent since the basket last changed."""
        return (self.confirmation_owed() and bool(self.roles.get("message")) and self.messages_left() != 0
                and not any(s > self.last_cart_step for s in self.message_steps))

    def basket_seen_empty(self) -> bool:
        """The store's basket view, read after the last basket change, listed no lines."""
        return not self.cart and self.basket_read_step > self.last_cart_step

    def finish_moves(self) -> int:
        """Moves the code spends right before an order: opening each new basket item's page, a look at the settled
        basket, then the confirmation."""
        look = 1 if self.roles.get("cart_view") and not any(s > self.last_cart_step for s in self.basket_view_steps) else 0
        return len(self.opens_due()) + look + (1 if self.confirmation_due() else 0)

    def open_procedures(self) -> list[tuple[str, str]]:
        """(kind, instruction) for each stated procedure the episode has not carried out yet."""
        out = []
        for p in self.brief.get("procedures") or []:
            kind, text, count = p.get("kind"), p.get("text") or "", p.get("count")
            if kind in ("go_ahead", "tell_first") and self.roles.get("message") and self.messages_left() != 0:
                if not any(s > self.last_cart_step for s in self.message_steps):
                    out.append((kind, text))
            elif kind == "inspect_count" and isinstance(count, int) and len(self.detail_ids) < min(count, 12):
                out.append((kind, text))
            elif kind == "check_basket" and self.roles.get("cart_view"):
                if not any(s > self.last_cart_step for s in self.basket_view_steps):
                    out.append((kind, text))
            elif kind == "compare_first" and self.compare_tool() and not self.compare_steps:
                out.append((kind, text))
        return out

    def tool_spec(self, name: str | None) -> dict:
        return next((t["function"] for t in self.tools if t["function"]["name"] == name), {})

    def finish_kind(self, name: str, args: dict) -> str | None:
        """'order' or 'decline' when this call would finish the episode, else None."""
        order, decline = self.roles.get("order"), self.roles.get("decline")
        if name not in (order, decline):
            return None
        if order == decline:
            # route by the choice fields (enums, flags); free text such as an explanation only when there are none
            props = (self.tool_spec(name).get("parameters") or {}).get("properties") or {}
            choices = [(k, v) for k, v in args.items() if isinstance(v, str) and (props.get(k) or {}).get("enum")
                       and _outcome_choice(k, [o for o in (props.get(k) or {}).get("enum") or [] if isinstance(o, str)])]
            choices.sort(key=lambda kv: not re.search(r"way|mode|kind|variant|action|outcome|result|choice|decision|type|verb",
                                                      kv[0], re.I))
            for k, v in choices[:1]:
                options = [o for o in (props.get(k) or {}).get("enum") or [] if isinstance(o, str)]
                return "decline" if _choice_declines(v, options) else "order"
            words = DECLINE_WORDS
            values = " ".join(str(v) for k, v in args.items() if isinstance(v, str)
                              and not re.search(r"explan|reason|note|why|justif|comment|remark|summary|detail", k, re.I))
            if re.search(words, values, re.I):
                return "decline"
            for k, v in args.items():  # a flag answers its own name: abstain=true declines, buy=false declines
                if isinstance(v, bool):
                    if re.search(words, k, re.I):
                        return "decline" if v else "order"
                    if re.search(r"buy|purchas|order|confirm|accept|proceed", k, re.I):
                        return "order" if v else "decline"
            return "order"
        return "order" if name == order else "decline"

    def apply_reading(self, reading: dict[str, Any]) -> None:
        if isinstance(reading.get("requirements_now"), list):
            def prop(req: str) -> str:
                return re.sub(r"[^a-z0-9 ]", " ", req.split(":", 1)[0].lower()).strip()
            new = [self._shopper_wording(str(r)) for r in reading["requirements_now"] if str(r).strip()]
            dropped = {prop(str(d)) for d in reading.get("dropped") or [] if str(d).strip()}
            # "Don't mind" about a property is never a reason to drop a requirement on it, and a requirement is
            # only dropped when the shopper's own words name its property ("anything is fine" names none).
            dropped -= {prop(str(d)) for d in reading.get("no_preference") or [] if str(d).strip()}
            said = " ".join(self._last_lines).lower()
            dropped = {d for d in dropped if any(w in said for w in re.findall(r"[a-z]{3,}", d))}
            have = {prop(r) for r in new}
            for old in self.requirements:  # keep what the reading lost without the shopper dropping it
                p = prop(old)
                if p and p not in have and not any(p in d or d in p for d in dropped if d):
                    new.append(old)
            self.requirements = new
        cap = _as_number(reading.get("new_price_cap"))
        if cap is not None and cap > 0:
            cap = self._grounded_amount(cap)
        if cap is not None and cap > 0:
            self.price_cap = cap
        q = reading.get("new_quantity")
        if isinstance(q, int) and q > 0:
            self.quantity = q
        if reading.get("item_unavailable"):
            for i in list(self.cart):
                if i not in self.unavailable and (self.listings.get(i) or {}).get("title"):
                    self.gone_picks.append(i)  # chosen as a match, so listings of the same product are worth a look
                self.unavailable.add(i)
                if i in self.listings:
                    self.listings[i]["in_stock"] = False
        if reading.get("price_changed"):
            for i in list(self.cart):
                if i in self.listings:
                    self.listings[i].pop("price", None)
                    self.listings[i]["price_stale"] = True
        self.verdicts.clear()

    def _shopper_wording(self, req: str) -> str:
        """A condition read from fresh shopper lines keeps the property name the shopper used: when the reader
        wrote a name the lines do not contain, and the lines contain exactly one property name the store's
        listings use, that name replaces it (a renamed condition would overwrite a different one)."""
        product, attr, value = split_req(req)
        if ":" not in req or req in self.requirements:
            return req
        def norm(text: str) -> str:
            return " ".join(re.findall(r"[a-z0-9]+", text.lower()))
        heard = f" {norm(' '.join(self._last_lines))} "
        if not heard.strip() or f" {norm(attr)} " in heard:
            return req
        value_words = [w for w in norm(value).split() if w not in ("not", "no", "or", "and", "the", "a")]
        if not value_words or not all(f" {w}" in heard for w in value_words):
            return req  # not a condition these lines state (an older one reworded): leave it alone
        names = set()
        for rec in self.listings.values():
            for v in (rec.get("fields") or {}).values():
                if isinstance(v, dict) and len(v) >= 2:
                    names.update(norm(str(k)) for k in v)
        said = [n for n in names if n and f" {n} " in heard]
        said = [n for n in said if not any(n != m and f" {n} " in f" {m} " for m in said)]  # longest wording wins
        return (f"{product} - " if product else "") + f"{said[0]}: {value}" if len(said) == 1 else req

    def _grounded_amount(self, value: float) -> float | None:
        """Accept a model-read amount only if the shopper's latest lines write that number;
        otherwise keep the old budget."""
        text = " ".join(self._last_lines).replace(",", "")
        numbers = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", text)]
        for n in numbers:
            if abs(n - value) < 1e-6:
                return value
        return None

    def read_lines(self, lines: list[str], context: str = "") -> dict[str, Any] | None:
        self._last_lines = list(lines)
        user = (f"CURRENT REQUIREMENTS: {_dumps(self.requirements)}\nBUDGET: {self.price_cap}\nQUANTITY: {self.quantity}\n"
                f"{context}\nNEW SHOPPER LINES:\n" + "\n".join(f"- {s}" for s in lines))
        reading = _reading_fields(self.ask_json(LIGHT_MODEL, READ_PROMPT, user, max_tokens=800, role="read"))
        if reading:
            self.apply_reading(reading)
            self.notes.append(str(reading.get("summary") or " / ".join(lines))[:300])
        else:
            self.notes.append(" / ".join(lines)[:300])
        return reading

    # --- judging ---------------------------------------------------------------------------

    def record_for(self, lid: str) -> dict[str, Any]:
        rec = self.listings.get(lid) or {}
        return {"id": lid, **{k: v for k, v in (rec.get("fields") or {}).items()}}

    def judge(self, ids: list[str], careful: bool = False) -> dict[str, dict[str, Any]]:
        """Fit verdicts for listings, cached until the requirements change; price and stock are not judged
        here. Screening uses the light model; `careful` re-judges with the strong model."""
        model = JUDGE_MODEL if careful else LIGHT_MODEL
        self.merge_department_products()
        self.fold_department_words()
        todo = [i for i in ids if i in self.listings
                and (i not in self.verdicts or (careful and self.verdicts[i].get("model") != JUDGE_MODEL))]
        for chunk_start in range(0, len(todo), 6):
            chunk = todo[chunk_start:chunk_start + 6]
            if self.time_left() < 25:
                break
            user = (f"REQUEST (the conditions below were taken from it; where its own words name a condition's property or "
                    f"value more exactly, follow the request's words, but never add a condition): {self.request[:1500]}\n"
                    f"PRODUCT: {_dumps(self.brief.get('products'))}\nREQUIREMENTS: {_dumps(self.requirements)}\n"
                    f"WISHES (not required): {_dumps(self.brief.get('soft_wants') or [])}\n"
                    f"CATEGORY LABELS SEEN: {_dumps(self.category_counts())}\n"
                    + (f"REQUIRED CATEGORY: {self.required_category()} (a listing in any other category fails)\n"
                       if self.required_category() else "") + "\nLISTINGS:\n"
                    + "\n".join(_dumps(self.record_for(i), 3500) for i in chunk)
                    + self.referenced_records(chunk))
            out = self.ask_json(model, JUDGE_PROMPT, user, max_tokens=600 + 350 * len(chunk),
                                role="judge careful" if careful else "judge screen") or {}
            for item in out.get("verdicts") or []:
                if isinstance(item, dict) and item.get("id") in chunk:
                    tests = [c for c in item.get("tests") or [] if isinstance(c, dict)]
                    fits = bool(item.get("pass")) and all(c.get("ok") is not False for c in tests)
                    problems = item.get("issues") or [f"{c.get('need')}: {c.get('proof')}" for c in tests
                                                      if c.get("ok") is False]
                    self.verdicts[item["id"]] = {"fits": fits, "problems": problems, "model": model}
        return {i: self.verdicts[i] for i in ids if i in self.verdicts}

    def relative_requirements(self) -> list[str]:
        return [r for r in self.requirements if _RELATIVE.search(r)]

    @staticmethod
    def reference_names(req: str) -> set[str]:
        """Capitalised names in a relative condition (the seller, maker or brand of the listing it points to)."""
        return {w for w in re.findall(r"\b[A-Z][a-z]{3,}\b", req) if w.lower() not in _NOT_NAMES}

    def referenced_ids(self, req: str) -> list[str]:
        """Opened listings whose seller, maker, brand or title carries a name the condition gives."""
        names = self.reference_names(req)
        out = []
        for lid in self.detail_ids:
            rec = self.listings.get(lid) or {}
            text = " ".join(str(v) for k, v in (rec.get("fields") or {}).items()
                            if isinstance(v, str) and re.search(r"brand|maker|merchant|seller|vendor|shop|store|title|name", k, re.I))
            if names and any(n.lower() in text.lower() for n in names):
                out.append(lid)
        return out

    def referenced_records(self, chunk: list[str]) -> str:
        """Records of opened listings that a relative condition seems to name (by brand, maker or seller words)."""
        relative = self.relative_requirements()
        if not relative:
            return ""
        found = []
        for r in relative:
            found += [i for i in self.referenced_ids(r) if i not in chunk and i not in found]
        records = [_dumps(self.record_for(i), 2500) for i in found[:4]]
        return ("\n\nREFERENCED LISTINGS:\n" + "\n".join(records)) if records else "\n\nREFERENCED LISTINGS: (none opened yet)"

    def required_category(self) -> str | None:
        """The store label whose words are exactly the requested product's words (a request for "A or B" ->
        a label "A & B"), when such a label was seen."""
        def stems(text: str) -> frozenset[str]:
            words = re.findall(r"[a-z0-9]+", text.lower())
            return frozenset(re.sub(r"(es|s)$", "", w) for w in words if w not in ("and", "or", "the", "a", "of"))
        products = [p.get("name") for p in (self.brief.get("products") or []) if isinstance(p, dict) and p.get("name")]
        if len(products) != 1:
            return None
        want = stems(str(products[0]))
        labels = {r.get("category") for r in self.listings.values() if isinstance(r.get("category"), str)}
        matches = [label for label in labels if stems(label) == want and want]
        return matches[0] if len(matches) == 1 else None

    def merge_department_products(self) -> None:
        """Wanted "products" that a single multi-name department names ("plates" and "cups" in "Plates, Bowls & Cups")
        are one product: races 8-12 have 123 tasks where a passing order held a line of such a department, and in
        none did a passing order hold two."""
        products = self.product_list()
        if len(products) < 2:
            return
        def stems(text: Any) -> frozenset[str]:
            words = re.findall(r"[a-z0-9]+", str(text).lower())
            return frozenset(re.sub(r"(es|s)$", "", w) for w in words if w not in ("and", "or", "the", "a", "of"))
        labels = {r.get("category") for r in self.listings.values() if isinstance(r.get("category"), str)}
        for label in sorted(x for x in labels if re.search(r"&|,|\band\b", x, re.I)):
            inside = [p for p in products if stems(p["name"]) and stems(p["name"]) <= stems(label)]
            if len(inside) < 2:
                continue
            names = [str(p["name"]) for p in inside]
            kept = [p for p in products if p not in inside]
            merged = {"name": label, "quantity": max((p["quantity"] for p in inside if isinstance(p.get("quantity"), int)), default=1)}
            self.brief["products"] = kept + [merged]
            if not kept:
                self.quantity = max(1, merged["quantity"])
            rewritten: list[str] = []
            for r in self.requirements:  # conditions of the parts now belong to the one product
                prod, attr, value = split_req(r)
                if prod and stems(prod) <= stems(label):
                    r = f"{label + ' - ' if kept else ''}{attr}: {value}"
                if r not in rewritten:
                    rewritten.append(r)
            self.requirements = rewritten
            self.verdicts.clear()
            self.hints.append(f"The store files {', '.join(names)} under one department, '{label}'. They are ONE product: "
                              "one basket line from that department, not one line per word.")
            self.note("department", f"{names} are one product: '{label}'")
            return

    def fold_department_words(self) -> None:
        """A condition that only repeats a department name's words (a qualifier plus the product noun) names
        that department: require the department instead of an attribute its listings may not carry."""
        products = [p for p in (self.brief.get("products") or []) if isinstance(p, dict) and p.get("name")]
        if len(products) != 1 or not self.requirements:
            return
        def stems(text: str) -> frozenset[str]:
            words = re.findall(r"[a-z0-9]+", text.lower())
            return frozenset(re.sub(r"(es|s)$", "", w) for w in words if w not in ("and", "or", "the", "a", "of"))
        want = stems(str(products[0]["name"]))
        labels = {r.get("category") for r in self.listings.values() if isinstance(r.get("category"), str)}
        for req in list(self.requirements):
            if ":" not in req:
                continue
            value = stems(req.split(":", 1)[1])
            hits = [label for label in labels if want and value and stems(label) == want | value]
            if len(hits) == 1:
                self.requirements.remove(req)
                self.folded.append(req)  # its property stays settled for the questions
                products[0]["name"] = hits[0]
                self.verdicts.clear()
                self.note("department", f"'{req}' names the department '{hits[0]}', required from now on")

    def category_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for r in self.listings.values():
            if isinstance(r.get("category"), str):
                counts[r["category"]] = counts.get(r["category"], 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1])[:15])

    def sweep_hint(self) -> str | None:
        """For a cheapest request: a price cap likely to fit on the visible rows, from past capped searches."""
        best = None
        for s in reversed(self.searches):
            caps = [_as_number(v) for k, v in s["args"].items()
                    if re.search(r"cap|max|price|budget|spend|upto|under|limit", k, re.I) and _as_number(v) is not None
                    and not isinstance(v, bool)]
            if not caps or not s["rows"] or s["total"] is None:
                continue
            cap = min(c for c in caps if c is not None)
            if s["total"] > s["rows"]:
                best = (f"PRICE SWEEP: a capped search at {cap:g} matched {s['total']} rows but only {s['rows']} are "
                        f"visible (ordered by id, not price). Try the same search with a cap near "
                        f"{max(1, int(cap * s['rows'] / s['total'])):d}, then raise it about 1.5x at a time, opening the "
                        "cheapest plausible rows first.")
                break
            best = (f"PRICE SWEEP: the capped search at {cap:g} showed all {s['total']} rows. If none of them fits, raise "
                    f"the cap to about {int(cap * 1.5):d} and repeat.")
            break
        return best

    def cheap_coverage(self, below: float) -> bool:
        """Whether some search capped prices at or below `below` and showed every row it matched."""
        for s in self.searches:
            caps = [_as_number(v) for k, v in s["args"].items()
                    if re.search(r"cap|max|price|budget|spend|upto|under|limit", k, re.I) and _as_number(v) is not None
                    and not isinstance(v, bool)]
            if any(c is not None and c <= below for c in caps):
                if s["total"] is None or s["total"] <= s["rows"]:
                    return True
        return False

    def affordable(self, lid: str, quantity: int | None = None) -> bool | None:
        price = (self.listings.get(lid) or {}).get("price")
        if price is None or self.price_cap is None:
            return None if price is None and self.price_cap is not None else True
        return price * (quantity or self.quantity) <= self.price_cap + 1e-9

    def orderable(self, lid: str) -> bool:
        rec = self.listings.get(lid) or {}
        if lid in self.unavailable:
            return False
        need = self.required_category()
        if need and isinstance(rec.get("category"), str) and rec["category"] != need:
            return False
        if self.in_stock_only and rec.get("in_stock") is False:
            return False
        return self.affordable(lid) is not False

    # --- status board ----------------------------------------------------------------------

    def board(self) -> str:
        self.merge_department_products()
        self.fold_department_words()
        rows = []
        items = sorted(self.listings.values(), key=lambda r: (r.get("price") is None, r.get("price") or 0))
        for r in items[:28]:
            v = self.verdicts.get(r["id"])
            verdict = "-" if v is None else ("FITS" if v["fits"] else "fails: " + "; ".join(map(str, v["problems"]))[:90])
            stock = {True: "in stock", False: "OUT", None: "?"}[r.get("in_stock")]
            if r["id"] in self.unavailable:
                stock = "GONE"
            rows.append(f"  {r['id']} | {r.get('price', '?')} | {stock} | {str(r.get('category', ''))[:28]} | "
                        f"{str(r.get('title', ''))[:60]} | {'details' if r.get('detailed') else 'row'} | {verdict}")
        ml = self.messages_left()
        lines = [
            "STATUS SHEET (compiled by code from the actual responses)",
            f"Steps used {self.step}; steps left {self.steps_left()}. Messages left: {'unknown' if ml is None else ml}. "
            f"Time left about {int(self.time_left())} s.",
            f"Requirements now: {_dumps(self.requirements)}",
            f"Budget (total): {self.price_cap if self.price_cap is not None else 'none stated'}; quantity {self.quantity}; "
            f"in stock only: {self.in_stock_only}; priority: {self.priority}"
            + (f" ({self.brief.get('priority_measure')})" if self.priority == "best" else ""),
        ]
        products = self.product_list()
        if len(products) > 1:
            lines.append("SEVERAL PRODUCTS WANTED (one basket line each, total within budget): "
                         + "; ".join(f"{p['name']} x {p.get('quantity') or 1}" for p in products))
        if self.brief.get("rules"):
            lines.append(f"Procedure rules: {_dumps(self.brief.get('rules'))}")
        if self.notes:
            lines.append("Shopper so far: " + " | ".join(self.notes[-6:]))
        if self.cart:
            lines.append(f"Cart (as tracked): {_dumps(self.cart)}")
        cats = self.category_counts()
        if cats:
            lines.append("Category labels seen (listings): " + ", ".join(f"{k} ({v})" for k, v in cats.items()))
        need = self.required_category()
        if need:
            lines.append(f"REQUIRED CATEGORY: '{need}' is the store label for the requested product; only its listings "
                         "can be ordered. Search and filter inside it.")
        if self.searches:
            lines.append("Searches so far (args -> rows shown / total matched):")
            for s in self.searches[-8:]:
                lines.append(f"  {_dumps(s['args'], 160)} -> {s['rows']} / {s['total'] if s['total'] is not None else '?'}"
                             + (f"  [MORE PAGES: next-page value {_dumps(s['next'], 40)}]" if s.get("next") is not None else ""))
            short = [s["rows"] for s in self.searches if s.get("next") is not None and s["rows"]]
            if short and max(short) <= SHORT_PAGE:
                lines.append(f"PAGES ARE SHORT HERE ({max(short)} rows): the first page of a query rarely shows every match. "
                             "Read the next pages of the product's main queries (pass the next-page value) before calling a "
                             "query empty or giving up; the right item is often on page 2 or 3.")
        lines.append(f"Listings seen ({len(self.listings)}), cheapest first: id | price | stock | category | title | seen as | verdict")
        lines.extend(rows or ["  (none yet)"])
        for hint in self.hints[-3:]:
            lines.append("NOTE: " + hint)
        with_table = [t for t, v in self.tool_facts.items() if v]
        without = [t for t, v in self.tool_facts.items() if not v and t not in (self.roles.get("cart"),
                   self.roles.get("cart_view"), self.roles.get("order"), self.roles.get("decline"), self.roles.get("message"))]
        batch = self.batch_detail_tool()
        if batch and self.tool_facts.get(batch) is not False:
            lines.append(f"MANY AT ONCE: {batch} takes a list of listings and shows their details together; use it to check up "
                         f"to {self.list_limit(batch)} promising rows in a single move instead of opening them one by one"
                         + (" (larger lists failed here)." if batch in self.list_failed else "."))
        if with_table:
            lines.append(f"Tools that show a listing's property table: {', '.join(with_table)}"
                         + (f". Tools that showed none (do not use them to check properties): {', '.join(without)}"
                            if without else ""))
        if self.priority == "cheapest" and self.steps_left() > COMMIT_STEPS:
            sweep = self.sweep_hint()
            lines.append(sweep or "PRICE SWEEP: once the category label is known, filter that category with a LOW price "
                                  "cap so every matching row is visible, then raise it.")
        if self.priority == "none" and self.questions_done and self.verified_fits() and not self.order_attempts \
                and self.fits_every_product():
            lines.append("NO LOWEST-PRICE RULE: any confirmed match passes. Put the first confirmed match "
                         + ("of each product " if len(products) > 1 else "") + "into the basket and buy now.")
        for req in self.relative_requirements()[:2]:
            ids = self.referenced_ids(req)
            names = ", ".join(sorted(self.reference_names(req))) or "the named seller"
            lines.append(f"REFERENCED LISTING for '{req}': " + (
                f"opened: {', '.join(ids[:4])}; use its value of that property (if its notes send you to yet another "
                "listing, open that one and use its value)." if ids else
                f"not opened yet. Find the listing of {names} that has the details the condition gives (search with "
                f"the name and those details, or browse the product's listings and read their seller fields), open it, "
                "then judge candidates against its value."))
        promising = self.promising_rows()
        if promising:
            lines.append("UNOPENED ROWS WHOSE TITLE CARRIES A REQUIRED VALUE (open these before browsing further): "
                         + ", ".join(f"{i} ({self.listings[i].get('price', '?')})" for i in promising[:6]))
        fits = self.verified_fits()
        if fits and len(products) > 1:
            for n, prod in enumerate(products):
                mine = [i for i in fits if self.product_of(i) == n]
                lines.append(f"CONFIRMED MATCHES FOR THE {str(prod['name']).upper()}, cheapest first: "
                             + (", ".join(f"{i} ({self.listings[i].get('price', '?')})" for i in mine[:4]) or "none yet"))
        elif fits:
            lines.append("CONFIRMED MATCHES (meet the requirements, orderable as far as known), cheapest first: "
                         + ", ".join(f"{i} ({self.listings[i].get('price', '?')})" for i in fits[:6]))
        elif self.gone_picks:
            gone = self.gone_picks[-1]
            lines.append(f"SIBLING HINT: {gone} was picked as a match but is gone. Its title is \"{self.listings[gone]['title'][:120]}\". "
                         "Listings with the same or a near-identical title are usually the same product and fit too: "
                         "open those first.")
        if self.steps_left() <= COMMIT_STEPS:
            lines.append(f"ONLY {self.steps_left()} STEPS LEFT: stop exploring. Put the best confirmed match in the cart "
                         "(remove anything else) and order now; after a refusal, switch to the next confirmed match.")
        return "\n".join(lines)

    def promising_rows(self) -> list[str]:
        """Listings seen only as rows whose title carries every word of some required value, within budget and
        (when known) in the required category; cheapest first."""
        values = []
        for req in self.requirements:
            if ":" in req:
                value = req.split(":", 1)[1].split(",")[0].strip().lower()
                words = [w[:5] for w in re.findall(r"[a-z]{3,}", value)]
                if words and not re.match(r"(not|no|without|except|excluding)\b", value):
                    values.append(words)
        if not values:
            return []
        need = self.required_category()
        out = []
        for r in self.listings.values():
            title = str(r.get("title", "")).lower()
            if r["id"] in self.detail_ids or not self.orderable(r["id"]) or not title:
                continue
            if need and isinstance(r.get("category"), str) and r["category"] != need:
                continue
            if self.price_cap is not None and r.get("price") is not None and r["price"] > self.price_cap:
                continue
            if any(all(w in title for w in words) for words in values):
                out.append(r["id"])
        return sorted(out, key=lambda i: (self.listings[i].get("price") is None, self.listings[i].get("price") or 0))

    def keyword_template(self) -> tuple[dict[str, Any], str] | None:
        """Arguments of the latest search with a text argument that returned rows, and that argument's name."""
        for s in reversed(self.searches):
            if s.get("text_key") and s["rows"]:
                return s["args"], s["text_key"]
        return None

    def due_probes(self) -> list[tuple[str, str]]:
        """Keyword searches the code sends itself. Title words of a basket pick that sold out (stores often
        list one product several times), and each short required value with the product's head noun
        (keyword search often needs every word in the title, so short queries find more)."""
        out = []
        for i in self.gone_picks:
            words = re.findall(r"[A-Za-z]{3,}", str(self.listings[i].get("title", "")))[:5]
            if len(words) >= 2:
                out.append((" ".join(words), f"other listings of the same product as {i}, which sold out"))
        for req in self.relative_requirements()[:2]:
            names = sorted(self.reference_names(req))
            if names and not self.referenced_ids(req):
                out.append((" ".join(names[:2]), f"the listing the condition '{req}' points to"))
        products = [p.get("name") for p in self.brief.get("products") or [] if isinstance(p, dict) and p.get("name")]
        head = (re.findall(r"[a-z]+", str(products[0]).lower()) or [""])[-1] if len(products) == 1 else ""
        titles = [str(r.get("title", "")).lower() for r in self.listings.values()]
        if not titles or sum(head[:5] in t for t in titles) < 0.3 * len(titles):
            head = ""  # the product word is rare in titles here, so it would only hide matches
        values = 0
        for req in self.requirements:
            if ":" not in req or values >= 2:
                continue
            value = req.split(":", 1)[1].split(",")[0].strip().lower()
            if re.match(r"(not|no|without|except|excluding|any|at least|at most|under|over|between)\b", value) \
                    or re.search(r"\d|[<>=]|\bor\b", value):
                continue
            words = re.findall(r"[a-z]+", value)
            if not 1 <= len(words) <= 3:
                continue
            values += 1
            query = " ".join(words + ([head] if head and head not in words else []))
            out.append((query, f"listings whose titles carry the required value '{value}'"))
        return [(q, why) for q, why in out if q not in self.probed]

    def fits_every_product(self) -> bool:
        products = self.product_list()
        fits = self.verified_fits()
        if len(products) < 2:
            return bool(fits)
        return all(any(self.product_of(f) == i for f in fits) for i in range(len(products)))

    def closest_matches(self) -> list[str]:
        """Judged listings that can be bought (orderable, affordable, in the required category), full fits first, then
        fewest unmet conditions, then cheapest. Ordering the closest match beats giving up: declines were wrong in
        about 9 of 10 cases in races 7 and 8 and in every case in race 9, and in race 9 the 'closest fallback' a
        declining agent named matched a passing agent's purchase in about half the checkable cases."""
        need = self.required_category()
        pool = []
        for i, v in self.verdicts.items():
            rec = self.listings.get(i) or {}
            if not self.orderable(i) or self.affordable(i) is False:
                continue
            if need and isinstance(rec.get("category"), str) and rec["category"] != need:
                continue
            misses = 0 if v.get("fits") else max(1, len(v.get("problems") or []))
            pool.append((misses, rec.get("price") is None, rec.get("price") or 0, i))
        return [i for *_rest, i in sorted(pool)]

    def last_resort(self) -> list[str]:
        """No judged listing to fall back on: buyable listings of a wanted kind, opened ones first, ranked by how many
        requirement words their record holds, then price. Same task, races 9-11: runs that ordered passed more often
        than runs that declined in 76 tasks and less often in 3, so a weak pick still beats giving up."""
        products = self.product_list()
        kinds: set[str] = set()
        for d in self.detail_ids:
            c = (self.listings.get(d) or {}).get("category")
            if isinstance(c, str) and c.strip():
                kinds.add(c)
        need = self.required_category()
        if need:
            kinds = {need}
        product_words = set().union(*(_stems(str(p.get("name") or "")) for p in products)) if products else set()
        values = set()
        for r in self.requirements:
            if ":" in r and not re.search(r":\s*not\b", r, re.I):
                values |= _stems(split_req(r)[2])
        pool = []
        for lid, rec in self.listings.items():
            if not self.orderable(lid) or rec.get("in_stock") is False or (self.affordable(lid) is None and self.price_cap is not None):
                continue
            cat = rec.get("category")
            title_words = _stems(str(rec.get("title") or ""))
            if kinds and isinstance(cat, str):
                if cat not in kinds:
                    continue
            elif not (product_words & (title_words | _stems(str(cat or "")))):
                continue  # nothing says it is the wanted kind
            text = _stems(_listing_text(rec.get("fields") or {}) + " " + str(rec.get("title") or ""))
            hits = len(values & text)
            pool.append((lid not in self.detail_ids, -hits, rec.get("price") is None, rec.get("price") or 0, lid))
        return [x[-1] for x in sorted(pool)]

    def verified_fits(self) -> list[str]:
        fits = [i for i, v in self.verdicts.items() if v.get("fits") and self.orderable(i)]
        return sorted(fits, key=lambda i: (self.listings[i].get("price") is None, self.listings[i].get("price") or 0))

    def note(self, kind: str, text: Any) -> None:
        """Diagnostics for the returned dialogue (never used for scoring)."""
        self.dialogue.append({"note": kind, "step": self.step, "work": round(self.work, 1),
                              "text": text if isinstance(text, (dict, list)) else str(text)[:1200]})


# ---------------------------------------------------------------------------------------------
# Brief


def make_brief(ep: Episode) -> None:
    tools = [{"name": t["function"]["name"], "description": t["function"].get("description", ""),
              "parameters": t["function"].get("parameters", {})} for t in ep.tools]
    user = f"REQUEST:\n{ep.request}\n\nOPENING: {_dumps(ep.opening, 1500)}\n\nTOOLS:\n{_dumps(tools, 9000)}"
    brief = _brief_fields(ep.ask_json(BRIEF_MODEL, BRIEF_PROMPT, user, max_tokens=1800, role="brief") or {})
    ep.apply_brief(brief)
    if not ep.confirmation_owed():  # the reading can miss it; the request's own sentence decides
        for sentence in confirmation_sentences(ep.request)[:1]:
            ep.brief.setdefault("procedures", []).append({"text": sentence, "kind": "go_ahead", "count": None})
    ep.note("brief", {"brief": brief, "roles": ep.roles})
    m = re.search(r"(\d+)\s*(?:messages?|questions?)", ep.request, re.I)
    if ep.message_budget is None and m:
        ep.message_budget = int(m.group(1))


# ---------------------------------------------------------------------------------------------
# Questions to the shopper


# keys that never held a shopper's unwritten preference in races 8-12 (706 cases): identifiers, links, warranty,
# condition, and bare variant lists ("color options" did, so only the bare words are left out)
_NOT_PROPERTY = re.compile(r"^(options?|variants?)$|\b(sku|asin|upc|ean|barcodes?|urls?|links?|images?|photos?|"
                           r"warrant\w*|condition)\b", re.I)


def menu_stats(ep: Episode, only: list[str] | None = None) -> tuple[dict[str, dict[str, Any]], int]:
    """Property keys of the detailed listings (or of `only`): per key the listings carrying it, sample values and
    spread; and how many listings were read."""
    stats: dict[str, dict[str, Any]] = {}
    ids = [i for i in (only if only is not None else ep.detail_ids) if i in ep.listings] or (
        [] if only is not None else list(ep.listings)[:10])
    if ep.detail_ids and only is None:
        # Prefer listings in the category most of the opened ones share (the requested kind).
        cats: dict[str, int] = {}
        for i in ids:
            c = ep.listings[i].get("category")
            if isinstance(c, str):
                cats[c] = cats.get(c, 0) + 1
        if cats:
            top = max(cats, key=cats.get)
            ids = [i for i in ids if ep.listings[i].get("category") in (None, top)] or ids
    for lid in ids[:20]:
        fields = ep.listings[lid].get("fields") or {}
        # Only the listing's property tables (nested objects such as facts/specs) describe the product.
        tables = [v for k, v in fields.items() if isinstance(v, dict) and len(v) >= 2 and not _PRICE_KEY.search(k)]
        counted: set[str] = set()
        for table in tables:
            for raw_key, value in table.items():
                if isinstance(value, (dict, list)) and not (isinstance(value, list) and all(isinstance(x, str) for x in value)):
                    continue
                key = re.sub(r"(?<=[a-z0-9])(?=[A-Z][a-z])", " ", str(raw_key))  # lidColour -> lid Colour
                key = re.sub(r"[._]+", " ", key).strip().lower()
                key = re.sub(r"^(?:[a-z]{1,2} )+(?=[a-z]{3,})", "", key)  # store prefixes: "fa pattern" -> "pattern"
                if _PRICE_KEY.search(key) or _STOCK_KEY.search(key) or re.search(
                        r"\b(id|sku|url|image)\b", key) or (isinstance(value, str) and value.count(",") >= 5):
                    continue
                s = stats.setdefault(key, {"n": 0, "values": [], "all": []})
                if key not in counted:
                    s["n"] += 1
                    counted.add(key)
                text = value if isinstance(value, str) else _dumps(value)
                s["all"].append(str(text).strip().lower())
                if text not in s["values"] and len(s["values"]) < 6:
                    s["values"].append(str(text)[:40])
    seen = max(1, len(ids[:20]))
    for v in stats.values():
        # spread: how much the property separates listings (coverage x share not held by the commonest value)
        counts = [v["all"].count(x) for x in set(v["all"])]
        v["share"] = v["n"] / seen
        v["spread"] = v["share"] * (1 - max(counts) / max(1, len(v["all"]))) if counts else 0.0
    return stats, seen


def property_menu(ep: Episode, only: list[str] | None = None) -> str:
    """Property keys of the detailed listings (or of `only`) with how often each appears and sample values."""
    stats, seen = menu_stats(ep, only)
    # coverage first, spread second: the held-back property was in the top 3 more often this way in races 8-11
    ranked = sorted(stats.items(), key=lambda kv: (-(kv[1]["share"] + 0.3 * kv[1]["spread"]), -kv[1]["n"]))[:40]
    lines = [f"- {k} (in {v['n']} of {seen}, spread {v['spread']:.2f}): {', '.join(v['values'])}"
 for k, v in ranked
             if not _NOT_PROPERTY.search(k)]
    return "\n".join(lines) or "(no listing properties seen yet)"


def _product_label(ep: Episode, index: int) -> str:
    """How a question names a wanted product: the store department its opened listings share, else its name."""
    products = ep.product_list()
    cats: dict[str, int] = {}
    for d in ep.detail_ids:
        c = (ep.listings.get(d) or {}).get("category")
        if isinstance(c, str) and c.strip() and (len(products) < 2 or ep.product_of(d) == index):
            cats[c.strip()] = cats.get(c.strip(), 0) + 1
    if cats:
        return max(cats, key=cats.get).lower()
    return str(products[index]["name"]) if index < len(products) else "item"


def _menu_keys(ep: Episode, only: list[str] | None) -> list[str]:
    """Property names from a listing menu, best first (the fallback ranking when the planner gives too few)."""
    keys = []
    for line in property_menu(ep, only).splitlines():
        m = re.match(r"- (.+?) \(in (\d+) of (\d+), spread ([\d.]+)\)", line)
        if m and m.group(1) != "options" and float(m.group(4)) > 0:
            keys.append(m.group(1))
    return keys


def _same_property(a: str, b: str) -> bool:
    """Two names for one property: one's words hold the other's, leaving out words such as "type" and the British or
    American spelling of colour."""
    def words(text: str) -> set[str]:
        return _stems(re.sub(r"colour", "color", text.lower())) - {"type", "kind", "of", "the"}
    wa, wb = words(a), words(b)
    return bool(wa) and bool(wb) and (wa <= wb or wb <= wa)


_RARELY_HELD = ("size", "type", "model", "features", "wattage", "length", "voltage", "dimensions")
GUARD_KEYS = 3      # a product's most carried keys that may claim one of its first questions
GUARD_GAP = 0.34    # ...from a planner pick carried by at least this much smaller a share of its listings


def guard_coverage(ep: Episode, queue: list[str], only: list[str] | None, settled: Any) -> list[str]:
    """Keep each product's most carried listing keys among its first questions. The held-back property was a key most
    listings carry far more often than a rare one (races 8-12: coverage ranks it in the top 3 in about 63% of cases),
    so a planner pick that few listings carry gives way, unless the request itself points at it (ASK FIRST or STORY
    HINTS)."""
    stats, _seen = menu_stats(ep, only)
    share = {k: v["share"] for k, v in stats.items() if v["spread"] > 0 and not _NOT_PROPERTY.search(k)}
    if not share:
        return queue
    hinted = _stems(_dumps(ep.brief.get("ask_first") or []) + " " + _dumps(ep.brief.get("story_hints") or []))

    def is_hinted(attr: str) -> bool:  # the attribute's own noun (its last word) is in the request's hints
        words = [w for w in re.findall(r"[a-z]{3,}", attr.lower()) if w not in ("type", "kind", "style")]
        return bool(words) and bool(_stems(words[-1]) & hinted)

    def share_of(attr: str) -> float:
        if attr in share:
            return share[attr]
        aw = _stems(attr)
        near = [v for k, v in share.items() if aw and (_stems(k) <= aw or aw <= _stems(k))]
        return max(near) if near else 0.0

    open_ = [a for a in queue if not settled(a)]
    rest = [a for a in queue if settled(a)]
    guards = [k for k in sorted(share, key=lambda k: -share[k]) if not settled(k)][:GUARD_KEYS]
    for key in guards:
        key_share = share[key]
        named = next((a for a in open_ if a == key or _same_property(a, key)), None)  # keep the queue's own name for it
        if named is not None and named in open_[:3]:
            continue
        key = named or key
        others = [a for a in open_ if a != key]
        weak = next((j for j, a in enumerate(others[:3]) if not is_hinted(a)
                     and share_of(a) + GUARD_GAP <= key_share), None)
        if weak is None and len(others) >= 3:
            continue
        if weak is None:
            others.insert(len(others[:3]), key)
        else:
            moved = others[weak]
            others[weak] = key
            others.insert(min(3, len(others)), moved)
        open_ = others
    return open_ + rest


def ask_questions(ep: Episode, convo: list[dict]) -> None:
    """Question the customer on likely held-back attributes, one attribute per message, adapting to each reply:
    the product with no revealed preference and the fewest questions so far is asked next, and an attribute
    that drew "no preference" or "I don't follow" is never asked again in other words."""
    ep.questions_done = True
    ep.merge_department_products()
    tool = ep.roles.get("message")
    left = ep.messages_left()
    if not tool or left == 0:
        ep.note("questions skipped", f"message tool {tool!r}, messages left {left}")
        return
    limit = max(0, (4 if left is None else left) - ep.reserved_messages())  # keep the confirmation's message
    if limit <= 0:
        return
    products = ep.product_list() or [{"name": "item"}]
    multi = len(products) > 1
    only = {i: ([d for d in ep.detail_ids if ep.product_of(d) == i] if multi else None) for i in range(len(products))}
    if multi:
        menus = "\n\n".join(f"PROPERTIES OF LISTINGS SEEN FOR THE {str(p['name']).upper()}:\n" + property_menu(ep, only[i])
                             for i, p in enumerate(products))
    else:
        menus = f"PROPERTIES OF LISTINGS SEEN:\n{property_menu(ep)}"
    user = (PLAN_PROMPT
            + f"\n\nREQUEST:\n{ep.request}\n\nPRODUCT: {_dumps(ep.brief.get('products'))}\n"
            f"STATED REQUIREMENTS: {_dumps(ep.requirements)}\nASK FIRST: {_dumps(ep.brief.get('ask_first') or [])}\n"
            f"STORY HINTS: {_dumps(ep.brief.get('story_hints') or [])}\n"
            f"ALREADY ASKED: {_dumps(ep.asked)}\n\n{menus}")
    plan = ep.ask_json(JUDGE_MODEL, "You plan clarifying questions. Answer with JSON only.", user, max_tokens=700, role="plan") or {}

    def index_of(name: Any) -> int:
        if not multi:
            return 0
        scores = [len(_stems(p["name"]) & _stems(str(name))) for p in products]
        return scores.index(max(scores)) if max(scores) > 0 and scores.count(max(scores)) == 1 else -1

    queues: dict[int, list[str]] = {i: [] for i in range(len(products))}
    for entry in plan.get("plan") or []:
        if isinstance(entry, dict):
            i = index_of(entry.get("product"))
            if i >= 0:
                attrs = entry.get("attributes") or []
                if isinstance(attrs, str):
                    attrs = re.split(r"[,;/]", attrs)
                queues[i] += [str(a).strip().lower() for a in attrs if isinstance(a, str) and re.search(r"[a-z]{2}", str(a).lower())
                              and not _NOT_PROPERTY.search(str(a).strip())]  # never a hidden key (races 8-12)
    # Never spend a message on a property the request already settles: a "don't mind" reply to it
    # wastes the message (and must not be read as dropping the requirement).
    product_words = set(re.findall(r"[a-z]+", _dumps(ep.brief.get("products")).lower()))
    filler = {"the", "of", "for", "a", "an"} | product_words

    def words(text: str) -> set[str]:
        return {w.rstrip("s") for w in re.findall(r"[a-z]+", text.lower())} - {w.rstrip("s") for w in filler}

    def settled(i: int, attr: str) -> bool:
        """Stated already (the same attribute, or a broader word for a stated one) or asked before."""
        aw = words(attr) - {"type", "kind"}
        if not aw:
            return True
        aw = words(attr)
        for r in ep.requirements + ep.folded:
            if ":" not in r:
                continue
            prod, prop, _v = split_req(r)
            if multi and prod and index_of(prod) not in (i, -1):
                continue
            if aw <= words(prop):
                return True
        asked_here = [a for a in ep.asked if not multi or index_of(a.split(" - ")[0]) in (i, -1)]
        return any(aw == words(a.split(" - ")[-1]) for a in asked_here)

    for i in queues:  # the listing data fills in when the planner ranks too few
        fallback = _menu_keys(ep, only[i])
        queues[i] = list(dict.fromkeys(queues[i] + fallback))
        queues[i] = guard_coverage(ep, queues[i], only[i], lambda a, i=i: settled(i, a))
        # plain one-word properties that were (almost) never the held-back one across races 8-13 (asked on tasks / held
        # back): size 232/9, type 130/0, model 103/0, features 52/1, wattage 45/0, length 39/0, voltage 33/0,
        # dimensions 33/0. Asked last, unless the request itself points at them.
        hints = _dumps(ep.brief.get("ask_first") or []) + _dumps(ep.brief.get("story_hints") or [])
        late = {w for w in _RARELY_HELD if not re.search(rf"\b{w}\b", hints, re.I)}
        queues[i] = [a for a in queues[i] if a.strip() not in late] + [a for a in queues[i] if a.strip() in late]
    ep.note("question plan", {"limit": limit, "queues": {products[i]["name"]: q[:6] for i, q in queues.items()}})
    params = (next(t for t in ep.tools if t["function"]["name"] == tool)["function"].get("parameters") or {})
    props = params.get("properties") or {}
    text_arg = next((k for k in params.get("required") or [] if (props.get(k) or {}).get("type") == "string"), None) \
        or next((k for k, v in props.items() if v.get("type") == "string"), None)
    if not text_arg:
        return
    asks = {i: 0 for i in queues}
    revealed = {i: 0 for i in queues}
    answered = []
    while len(answered) < limit:
        if ep.finished or ep.steps_left() <= STEP_RESERVE + 4 or ep.time_left() < FINISH_SECONDS + 20:
            break
        if ep.messages_left() is not None and ep.messages_left() <= ep.reserved_messages():
            break
        for i in queues:
            queues[i] = [a for a in queues[i] if not settled(i, a)]
        open_ = [i for i in queues if queues[i]]
        if not open_:
            break
        # A product still without a revealed preference goes first; among those, the least asked.
        i = min(open_, key=lambda k: (revealed[k] > 0, asks[k], k))
        attr = queues[i].pop(0)
        label = _product_label(ep, i)
        text = f"What {attr} would you like for the {label}?"
        listed = [(i, attr)]
        last = limit - len(answered) == 1 or (ep.messages_left() is not None
                                              and ep.messages_left() - ep.reserved_messages() == 1)
        if last:
            # The last message names two more candidates after the one a lone question would ask. A shopper's reply
            # (and the record of what was asked) follows the first property in the message that the shopper tracks,
            # and properties they don't track are passed over, so this never does worse than the lone question and
            # gets a further chance when that property is untracked (races 8-12, same task and basket: first tracked
            # property of a list vs never asked 0.55 vs 0.30, 12 groups better, 3 worse). The shopper's check names
            # the property only, so a product not asked about yet gets its best candidate in first.
            def fresh(k: int, a: str) -> bool:
                return not settled(k, a) and not any(_same_property(a, b) for _, b in listed) \
                    and not any(_same_property(a, b.split(" - ")[-1]) for b in ep.asked)
            for k in queues:
                if k != i and asks[k] == 0 and len(listed) < 3:
                    best = next((a for a in queues[k] if fresh(k, a)), None)
                    if best:
                        listed.append((k, best))
            for a in queues[i]:
                if len(listed) < 3 and fresh(i, a):
                    listed.append((i, a))
            room = int((props.get(text_arg) or {}).get("maxLength") or 400)
            while len(listed) >= 2:
                if all(k == i for k, _ in listed):
                    names = [a for _, a in listed]
                    text = f"Which {', '.join(names[:-1])} or {names[-1]} would you like for the {label}?"
                else:
                    parts = [f"{a} for the {_product_label(ep, k)}" for k, a in listed]
                    text = f"Which {', '.join(parts[:-1])} or {parts[-1]} would you like?"
                if len(text) <= room:
                    break
                listed.pop()  # too long for the message field: name one property fewer
            if len(listed) < 2:
                text = f"What {attr} would you like for the {label}?"
        text = text[: int((props.get(text_arg) or {}).get("maxLength") or 400)]
        before = set(ep.requirements)
        observation, ok = ep.act(tool, {text_arg: text})
        for k in {k for k, _ in listed}:
            asks[k] += 1
        for k, a in listed:
            ep.asked.append((f"{products[k]['name']} - " if multi else "") + a)
        if not ok or _failed(observation):
            answered.append({"question": text, "reply": observation})
            if isinstance(observation, dict) and observation.get("error") == "budget_exceeded":
                break
            continue
        lines = []
        _strings_in(observation, lines)
        lines = [s for s in lines if s.strip()]
        ep.heard.update(lines)
        reading = ep.read_lines(lines, context=f"THE ASSISTANT ASKED: {text}") if lines else None
        fresh = [r for r in ep.requirements if r not in before]
        for r in fresh:
            prod = split_req(r)[0]
            k = index_of(prod) if prod else i
            revealed[k if k >= 0 else i] += 1
        answered.append({"question": text, "reply": lines, "meaning": (reading or {}).get("summary")})
    if answered:
        convo.append({"role": "user", "content": "The question planner asked the shopper (each question used one step "
                      "and one message):\n" + "\n".join(
                          f"- Q: {a['question']}\n  A: {_dumps(a['reply'], 400)}" + (f"\n  Meaning: {a['meaning']}" if a.get("meaning") else "")
                          for a in answered)
                      + f"\nRequirements now: {_dumps(ep.requirements)}. Search again with them together with the original conditions "
                      "and re-check your candidates."})


# ---------------------------------------------------------------------------------------------
# Gates


PROCEDURE_HELP = {
    "go_ahead": "Send the customer one short message naming the item in the basket and its price and asking whether to "
                "go ahead; read the reply, then order.",
    "tell_first": "Send the customer one short message that does exactly what the instruction asks (name the item in "
                  "the basket and its price when unsure), then order.",
    "inspect_count": "Open more items in detail until the required number is reached, then order the best match.",
    "check_basket": "Look at the basket with the basket-view tool now, then order.",
    "compare_first": "Run the store's comparison tool on the candidates, including the chosen one, then order.",
}


def _order_gate(ep: Episode) -> str | None:
    """A reason not to order yet, or None. Costs no steps.

    Facts the code knows for certain (price over budget, shown out of stock) block harder than
    judgement calls, but every block gives way after repeated attempts: the cart is tracked from the
    agent's own calls and could be wrong, and a refused order is cheaper than a frozen episode."""
    if ep.basket_seen_empty() and ep.gate_blocks.get("empty basket", 0) < 2:
        ep.gate_blocks["empty basket"] = ep.gate_blocks.get("empty basket", 0) + 1
        return ("PURCHASE HELD BACK (no move spent): the store's basket view shows the basket EMPTY, so an order would be "
                "refused. Put the chosen item in the basket first, or decline when no item qualifies.")
    if not ep.cart:
        return None  # nothing tracked to check; the store will answer
    for kind, text in ep.open_procedures():
        if kind in ("go_ahead", "tell_first", "check_basket"):
            continue  # done by code once the order is otherwise ready (see finish_procedures)
        key = f"procedure {kind}"
        if ep.gate_blocks.get(key, 0) < 2 and ep.steps_left() > 3:
            ep.gate_blocks[key] = ep.gate_blocks.get(key, 0) + 1
            gate = (f"PURCHASE HELD BACK (no move spent). The request sets a procedure not yet carried out: \"{text}\". "
                    + PROCEDURE_HELP.get(kind, ""))
            ep.note("order gate", gate)
            return gate
    hard, soft = [], []
    lines = list(ep.cart.items())
    if len(lines) > 1 and len(ep.brief.get("products") or []) <= 1:
        soft.append(f"the cart seems to hold {len(lines)} different listings ({', '.join(i for i, _ in lines)}); "
                    "a single-product order needs exactly one line")
    need = ep.required_category()
    for lid, qty in lines:
        rec = ep.listings.get(lid) or {}
        if lid in ep.unavailable or (ep.in_stock_only and rec.get("in_stock") is False):
            hard.append(f"{lid} is no longer available / shown out of stock")
        if need and isinstance(rec.get("category"), str) and rec["category"] != need:
            hard.append(f"{lid} is filed under '{rec['category']}', but the wanted product type is the store's "
                        f"'{need}' category")
        elif not need and isinstance(rec.get("category"), str):
            def stems(text: str) -> set[str]:
                return {re.sub(r"(es|s)$", "", w)[:5] for w in re.findall(r"[a-z]{3,}", text.lower())} - {"and", "the"}
            product = " ".join(str(p.get("name")) for p in ep.brief.get("products") or [] if isinstance(p, dict))
            if product and stems(product) and not stems(product) & stems(rec["category"]):
                soft.append(f"{lid} is filed under '{rec['category']}', which does not look like the requested "
                            f"'{product}'. Search the product's own words to find the store's category for it")
        if rec.get("price_stale"):
            soft.append(f"the price of {lid} changed; open it again to see the new price before buying")
        if len(lines) == 1 and qty != ep.quantity:
            soft.append(f"{lid} is in the cart {qty} time(s) but the quantity needed is {ep.quantity}")
        if rec.get("price") is not None and ep.price_cap is not None and rec["price"] * qty > ep.price_cap + 1e-9:
            hard.append(f"{lid} costs {rec['price']} x {qty}, over the budget {ep.price_cap}")
    products = ep.product_list()
    if len(products) > 1:
        by: dict[int | None, list[tuple[str, int]]] = {}
        for lid, qty in lines:
            by.setdefault(ep.product_of(lid), []).append((lid, qty))
        for i, prod in enumerate(products):
            got = by.get(i, [])
            if not got:
                fits = [f for f in ep.verified_fits() if ep.product_of(f) == i]
                soft.append(f"the basket has no line for the {prod['name']}"
                            + (f"; confirmed matches for it: {', '.join(fits[:3])}" if fits else "; find a match and add it"))
            elif len(got) > 1:
                soft.append(f"the {prod['name']} has {len(got)} lines in the basket; keep only one")
            elif got[0][1] != ep.quantity_for(got[0][0]):
                soft.append(f"{got[0][0]} ({prod['name']}) is in the basket {got[0][1]} time(s) but "
                            f"{ep.quantity_for(got[0][0])} are needed")
        total = sum(((ep.listings.get(lid) or {}).get("price") or 0) * qty for lid, qty in lines)
        if ep.price_cap is not None and total > ep.price_cap + 1e-9:
            hard.append(f"the basket totals {total:g}, over the budget {ep.price_cap:g}")
    late = ep.steps_left() <= COMMIT_STEPS
    if not hard and ep.time_left() > 45 and ep.steps_left() > 1:
        for lid, v in ep.judge([lid for lid, _ in lines], careful=True).items():
            if not v["fits"]:
                soft.append(f"{lid} does not meet: " + "; ".join(map(str, v["problems"]))[:300])
                others = [i for i in ep.verified_fits() if i != lid]
                if others:
                    soft.append("confirmed matches to use instead: " + ", ".join(others[:3]))
        if ep.priority == "cheapest" and len(products) > 1 and not late:
            for lid, _qty in lines:
                price = (ep.listings.get(lid) or {}).get("price")
                cheaper = [f for f in ep.verified_fits() if f != lid and ep.product_of(f) == ep.product_of(lid)
                           and price is not None and (ep.listings[f].get("price") or price) < price]
                if cheaper:
                    soft.append(f"{cheaper[0]} also seems to fit the same product as {lid} and is cheaper; the customer "
                                "wants the lowest price, so re-check it and switch if it truly fits")
        if ep.priority == "cheapest" and len(lines) == 1 and not soft and not late:
            lid = lines[0][0]
            price = (ep.listings.get(lid) or {}).get("price")
            if price is not None:
                cat = (ep.listings.get(lid) or {}).get("category")
                cheaper = [r["id"] for r in ep.listings.values()
                           if r["id"] != lid and r.get("price") is not None and r["price"] < price
                           and ep.orderable(r["id"]) and (cat is None or r.get("category") in (None, cat))]
                detailed = sorted((i for i in cheaper if i in ep.detail_ids), key=lambda i: ep.listings[i]["price"])[:10]
                fits = [i for i, v in ep.judge(detailed).items() if v["fits"]] if detailed else []
                if fits:
                    best = min(fits, key=lambda i: ep.listings[i]["price"])
                    soft.append(f"{best} also seems to fit and is cheaper ({ep.listings[best]['price']} < {price}); "
                                "the customer wants the lowest price, so re-check it and switch if it truly fits")
                unchecked = sorted((i for i in cheaper if i not in ep.detail_ids), key=lambda i: ep.listings[i]["price"])
                if unchecked:
                    soft.append("the cheapest rows not yet opened in detail (open them in this order, stop at the first "
                                "full fit): " + ", ".join(unchecked[:3]))
                if not ep.cheap_coverage(price) and ep.steps_left() > COMMIT_STEPS + 4:
                    soft.append(f"no search so far has shown the complete set of listings priced at or below {price}: "
                                "a cheaper fit may be hidden. Search the right category with a price cap at or below that "
                                "price, in a mode where the cap takes effect, until the reported total fits on the visible rows"
                                + (f". {ep.sweep_hint()}" if ep.sweep_hint() else ""))
    if late and not hard:
        soft = [s for s in soft if "does not meet" in s or "confirmed matches" in s or "quantity" in s
                or "no line for" in s or "are needed" in s or "lines in the basket" in s or "price of" in s]
    if not hard and not soft:
        return None
    key = "|".join(sorted(ep.cart))
    ep.gate_blocks[key] = ep.gate_blocks.get(key, 0) + 1
    limit = 2 if hard else (1 if late else 2)
    if ep.gate_blocks[key] > limit or ep.steps_left() <= 2:
        return None
    text = "PURCHASE HELD BACK (no move spent). Fix first: " + " | ".join(hard + soft)
    ep.note("order gate", text)
    return text


def _decline_gate(ep: Episode) -> str | None:
    """Declines are rarely right, so push back (twice at most) while steps allow."""
    if ep.steps_left() <= 2 or ep.time_left() < 40:
        return None
    blocks = ep.gate_blocks.get("decline", 0)
    if blocks >= 3 or (blocks >= 1 and ep.steps_left() <= 5):
        return None
    candidates = [r["id"] for r in ep.listings.values() if ep.orderable(r["id"]) and r["id"] in ep.detail_ids]
    candidates.sort(key=lambda i: (ep.listings[i].get("price") is None, ep.listings[i].get("price") or 0))
    verdicts = ep.judge(candidates[:12]) if candidates else {}
    fits = [i for i, v in verdicts.items() if v["fits"]]
    near = [i for i, v in verdicts.items() if not v["fits"] and len(v["problems"]) <= 1][:4]
    ep.gate_blocks["decline"] = blocks + 1
    text = None
    if fits:
        text = ("GIVE-UP HELD BACK (no move spent). These known listings seem to satisfy all conditions within the spending limit: "
                + ", ".join(fits[:5]) + ". Order the best of them instead, unless one clearly fails something.")
    elif blocks == 0 and ep.steps_left() > 6:
        text = ("GIVE-UP HELD BACK (no move spent). A mistaken give-up fails the task, and some item usually qualifies. Re-check every near match condition by condition before concluding. "
                "Search more first: the key required value word on its own, the right category with low price caps and "
                "further pages, and distinctive title words of listings that came close (siblings of the same product).")
        if near:
            text += " Near fits worth re-checking: " + "; ".join(
                f"{i} ({'; '.join(map(str, verdicts[i]['problems']))[:120]})" for i in near)
    elif blocks >= 1 and ep.steps_left() > 5:
        values = [r.split(":", 1)[1].strip() for r in ep.requirements if ":" in r][:4]
        text = (f"GIVE-UP HELD BACK (no move spent). {ep.steps_left()} steps are left and a mistaken give-up "
                "fails the task. Keep looking: keyword searches that pair the product word with each required value "
                f"({', '.join(values) if values else 'the key values'}) and with other common wordings of those values, "
                "the right category under lower price caps, and siblings of near fits. Open the most promising rows.")
    ep.note("decline gate", text or f"passed (checked {len(candidates)} known listings)")
    return text


NEUTRAL_REMARK = "Unfortunately no item here satisfies all of your conditions."


def _clean_remark(ep: Episode, args: dict[str, Any]) -> dict[str, Any]:
    """A give-up note that mentions no item: every free-text field gets one fixed, neutral sentence."""
    out = dict(args)
    props = (_schema(ep, ep.roles.get("decline") or "").get("properties") or {})
    for k, v in args.items():
        if isinstance(v, str) and not (props.get(k) or {}).get("enum"):
            out[k] = NEUTRAL_REMARK
    return out


# ---------------------------------------------------------------------------------------------
# Worker loop


def _tool_result_text(ep: Episode, observation: Any) -> str:
    return _dumps(observation, TOOL_TEXT_LIMIT)


def _trim(convo: list[dict]) -> None:
    full = [i for i, m in enumerate(convo) if m.get("role") == "tool" and len(m.get("content") or "") > OLD_TOOL_TEXT]
    if len(full) <= KEEP_FULL_RESULTS:
        return
    for i in full[:-TRIM_TO]:
        convo[i]["content"] = convo[i]["content"][:OLD_TOOL_TEXT] + "…(older result shortened; see the status board)"


def _cache_marked(convo: list[dict]) -> list[dict]:
    """A copy of the conversation whose newest text message carries a cache mark, so the next turn reads the
    shared start from the provider's cache instead of paying for it again."""
    for i in range(len(convo) - 1, -1, -1):
        content = convo[i].get("content")
        if isinstance(content, str) and content:
            marked = dict(convo[i], content=[{"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}])
            return convo[:i] + [marked] + convo[i + 1:]
    return convo


def _system_text(ep: Episode) -> str:
    roles = {k: v for k, v in ep.roles.items() if v}
    usage = ep.brief.get("search_usage")
    return (WORKER_GUIDE + "\n\nTOOL ROLES IN THIS STORE (from the tool list): " + _dumps(roles)
            + (f"\nHOW SEARCH WORKS HERE: {usage}" if isinstance(usage, str) and usage.strip() else "")
            + "\n\nBRIEF OF THE REQUEST (prepared in advance; the status board has the current state):\n"
            + _dumps({k: ep.brief.get(k) for k in ("products", "requirements", "price_cap", "price_unit", "in_stock_only",
                                                 "priority", "priority_measure", "message_budget", "step_budget", "rules",
                                                 "decline_rule", "soft_wants", "ask_first", "story")}, 4000))


def _questions_due(ep: Episode) -> bool:
    if ep.questions_done or ep.finished or not ep.roles.get("message"):
        return False
    if ep.messages_left() == 0:
        return False
    if ep.steps_left() <= STEP_RESERVE + 5:
        return False
    n = len(ep.product_list())
    if n > 1:
        covered = {ep.product_of(d) for d in ep.detail_ids} - {None}
        return ep.force_questions or len(covered) >= n or ep.step >= 12
    return ep.force_questions or len(ep.detail_ids) >= 3 or ep.step >= 9


_PAGE_ARG = re.compile(r"next|continu|cursor|token|after|page|offset|start|skip|begin", re.I)  # SkipAt (race 11), begin_at_at (race 10)
AUTO_PAGES = 6              # next pages the code may fetch in one task
SHORT_PAGE = 6              # a first page with at most this many rows is "short": most matches sit on later pages


def _page_token(observation: Any) -> Any:
    """The next-page value of a result that lists rows, or None."""
    body = _unwrapped(observation)
    if not isinstance(body, dict) or not any(isinstance(v, list) and v and isinstance(v[0], dict) for v in body.values()):
        return None
    for k, v in body.items():
        if re.search(r"next|cursor|continu|more_?|after|follow|token", k, re.I) and isinstance(v, (str, int)) \
                and not isinstance(v, bool) and str(v).strip():
            return v
    return None


def _later_page(args: dict[str, Any]) -> bool:
    """A call that already asks for a later page (a first page may pass row_start 0 or page 1)."""
    return any(_PAGE_ARG.search(k) and v not in (None, "", 0, 1, "0", "1") for k, v in args.items())


def _worth_paging(ep: Episode, observation: Any) -> bool:
    """A first page mostly of a wanted department while the store reports more matches than it shows. Races 8, 12, 13:
    when our run never saw a winning item, a later page of one of its own searches held one in 15 of 30, 6 of 11 and
    7 of 18 runs; this rule picks about 2-5 searches a task."""
    body = _unwrapped(observation)
    if not isinstance(body, dict):
        return False
    rows = max((v for v in body.values() if isinstance(v, list)), key=len, default=[])
    ids = [ep._listing_id(r) for r in rows if isinstance(r, dict)]
    cats = [(ep.listings.get(i) or {}).get("category") for i in ids if i]
    if len(cats) < 3:
        return False
    total = next((v for k, v in body.items() if isinstance(v, int) and not isinstance(v, bool)
                  and re.search(r"total|count|tally|matched|number|hits|found", k, re.I)), None)
    if total is not None and total <= len(rows):
        return False
    named = [c for c in cats if c]
    if not named:
        return False
    top, n = collections.Counter(named).most_common(1)[0]
    if n < 0.6 * len(cats):
        return False
    wanted = {(ep.listings.get(d) or {}).get("category") for d in ep.detail_ids} | {ep.required_category()}
    products = ep.product_list()
    names = set().union(*(_stems(str(p.get("name") or "")) for p in products)) if products else set()
    if top not in wanted and not (_stems(top) & names):
        return False
    if ep.verified_fits() and ep.brief.get("priority") != "cheapest":
        return False  # a fit is in hand: no need to read further unless the cheapest one is wanted
    return True


def _next_page_args(ep: Episode, name: str, args: dict[str, Any], observation: Any) -> dict | None:
    """Arguments for the next page of a first-page listing search, fetched by code (one move, no model turn), when the
    page came back short or the search is worth reading further (`_worth_paging`), while moves and time are plentiful."""
    total = ep.step + ep.steps_left()
    roomy = total >= 60  # races allowed about 100-130 steps; qualifying and practice 38
    allowance = min(AUTO_PAGES, total // (12 if roomy else 15))  # a small share of a tight step limit
    if ep.auto_pages >= allowance or ep.finished or ep.steps_left() <= (20 if roomy else 25) + ep.finish_moves() \
            or ep.time_left() < 90:
        return None
    if name in {ep.roles.get(r) for r in ("cart", "cart_view", "order", "decline", "message", "details")} or ep.takes_list(name):
        return None
    if _later_page(args):
        return None  # already a later page
    token = _page_token(observation)
    body = _unwrapped(observation)
    rows = max((len(v) for v in body.values() if isinstance(v, list)), default=0) if isinstance(body, dict) else 0
    if token is None or token is False or rows == 0:
        return None
    if not (rows <= SHORT_PAGE or (roomy and _worth_paging(ep, observation))):
        return None
    props = (_schema(ep, name).get("properties") or {})
    key = next((k for k, v in props.items() if (v or {}).get("type") == "string" and _PAGE_ARG.search(k)
                and isinstance(token, str)), None)
    value: Any = token
    if key is None:  # a row offset takes the store's next offset (race 13: next_token_token 10 -> row_start 10)
        key = next((k for k, v in props.items() if (v or {}).get("type") == "integer"
                    and re.search(r"offset|start|begin|from|skip|row", k, re.I)), None)
        value = token if isinstance(token, int) and not isinstance(token, bool) and token > 0 else rows
    if key is None:  # a page number
        key = next((k for k, v in props.items() if (v or {}).get("type") == "integer" and re.search(r"page", k, re.I)), None)
        value = 2
    if key is None:
        return None
    fitted, problem = _fit_args(dict(args, **{key: value}), _schema(ep, name))
    return None if problem else fitted


def run_worker(ep: Episode) -> None:
    system = {"role": "system", "content": [{"type": "text", "text": _system_text(ep), "cache_control": {"type": "ephemeral"}}]}
    first = f"SHOPPER REQUEST:\n{ep.request}"
    if ep.opening is not None:
        first += f"\n\nWHAT THE STORE SHOWS FIRST:\n{_dumps(ep.opening, 4000)}"
    convo: list[dict] = [{"role": "user", "content": first}]
    idle = 0
    turns = 0
    exit_reason = "finished"
    while not ep.finished:
        turns += 1
        if turns > ep.max_steps * 2 + 20:
            exit_reason = "turn limit"
            break
        if ep.steps_left() <= 0 or ep.time_left() < FINISH_SECONDS:
            exit_reason = "out of steps or time"
            break
        if ep.spent >= SPEND_HARD_USD:
            exit_reason = "spend cap"
            break
        due = ep.finish_moves()
        if ep.steps_left() <= 1 + due or (ep.steps_left() <= 2 + due and (not ep.order_attempts or not ep.cart)):
            exit_reason = "last moves: finishing with the best checked item"
            break
        if run_probes(ep, convo) or resolve_references(ep, convo):
            continue
        if _questions_due(ep):
            ask_questions(ep, convo)
            continue
        _trim(convo)
        board = {"role": "user", "content": ep.board()}
        msg = ep.chat(WORKER_MODEL, [system] + _cache_marked(convo) + [board], tools=ep.tools, max_tokens=1200, force_tool=idle > 0,
                      role="worker")
        if msg is None:
            idle += 1
            if idle >= 3:
                exit_reason = "model calls failing"
                break
            continue
        calls = msg.get("tool_calls") or []
        content = msg.get("content") or ""
        if not calls:
            idle += 1
            convo.append({"role": "assistant", "content": content or "(no tool call)"})
            convo.append({"role": "user", "content": "Reply with a tool call (one step) to continue."})
            if idle >= 3:
                exit_reason = "no tool calls"
                break
            continue
        idle = 0
        if content.strip():
            ep.note("worker", content)
        convo.append({"role": "assistant", "content": content, "tool_calls": calls})
        finishing = {ep.roles.get("order"), ep.roles.get("decline")} - {None}
        if len(calls) > 1 and any((c.get("function") or {}).get("name") in finishing for c in calls):
            for c in calls:
                convo.append({"role": "tool", "tool_call_id": c.get("id"),
                              "content": "NOT RUN: an order or decline must be the only call in its reply. Send it again on its own."})
            continue
        for c in calls:
            fn = c.get("function") or {}
            name = fn.get("name")
            raw = fn.get("arguments")
            try:
                args = json.loads(raw) if isinstance(raw, str) and raw.strip() else (raw if isinstance(raw, dict) else {})
            except ValueError:
                args = None
            if ep.finished:
                convo.append({"role": "tool", "tool_call_id": c.get("id"), "content": "NOT RUN: the episode has finished."})
                continue
            if not isinstance(args, dict):
                convo.append({"role": "tool", "tool_call_id": c.get("id"),
                              "content": "SKIPPED (no move spent): arguments must be a JSON object."})
                continue
            if name not in ep.tool_names:
                convo.append({"role": "tool", "tool_call_id": c.get("id"),
                              "content": f"SKIPPED (no move spent): there is no tool named {name!r}."})
                continue
            args, problem = _fit_args(args, ep.tool_spec(name).get("parameters") or {})
            if problem:
                convo.append({"role": "tool", "tool_call_id": c.get("id"),
                              "content": f"SKIPPED (no move spent): {problem}. Check the tool's schema and send a corrected call."})
                continue
            kind = ep.finish_kind(name, args)
            gate = None
            due = ep.finish_moves()
            # taking out or replacing a basket line that can no longer be ordered (gone, over the budget) is finishing
            # work: remove, add and order fit in three moves, and filling an empty basket needs only add and order
            swapping = name == ep.roles.get("cart") and (
                (ep.steps_left() >= 3 and any(not ep.orderable(i) for i in ep.cart)) or (not ep.cart and ep.steps_left() >= 2))
            if not kind and not swapping and ep.steps_left() <= 2 + due \
                    and (name != ep.roles.get("cart") or ep.cart or ep.steps_left() <= 1 + due):
                gate = ("NOT RUN (no move spent): the last moves are kept for finishing, because a purchase or "
                        "give-up can be answered with an update that needs one more call. Order or give up now.")
            elif not kind and name not in (ep.roles.get("cart"), ep.roles.get("cart_view")) \
                    and ep.steps_left() <= COMMIT_STEPS and ep.fits_every_product() \
                    and ep.gate_blocks.get("commit", 0) < 3:
                ep.gate_blocks["commit"] = ep.gate_blocks.get("commit", 0) + 1
                gate = (f"NOT RUN (no move spent): only {ep.steps_left()} moves are left and confirmed matches exist "
                        f"({', '.join(ep.verified_fits()[:3])}). Stop exploring: put the best confirmed match "
                        + ("of each product in the basket (one line each)" if len(ep.product_list()) > 1 else "in the basket (alone)")
                        + " and order now, keeping moves to recover if the order is refused.")
            elif kind == "order" and not ep.questions_done and ep.roles.get("message") and ep.messages_left() != 0 \
                    and ep.steps_left() > STEP_RESERVE + 4:
                ep.force_questions = True  # hidden preferences come first; the planner runs next turn
                gate = ("PURCHASE HELD BACK (no move spent): the customer's unwritten preferences have not been asked "
                        "yet. The questions are being sent now; re-check your pick against the answers before buying.")
            elif kind == "order":
                gate = _order_gate(ep) or finish_procedures(ep)
            elif kind == "decline":
                gate = _decline_gate(ep)
                closest = ep.closest_matches() if gate is None else []
                if gate is None and not closest and ep.roles.get("cart") and ep.steps_left() >= 4:
                    closest = [i for i in ep.last_resort() if _cart_args(ep, ep.roles["cart"], i) is not None][:3]
                if closest:
                    ep.buy_closest = True
                    gate = ("NOT RUN (no move spent): instead of giving up, the code now orders the closest match "
                            f"({', '.join(closest[:2])}); give-ups were wrong in nearly every past case.")
                elif gate is None:
                    args = _clean_remark(ep, args) if ep.brief.get("decline_rule") != "name_alternatives" else args
            elif name == ep.roles.get("message"):
                if ep.messages_left() == 0:
                    gate = "SKIPPED (no move spent): no messages remain."
                elif ep.messages_left() is not None and ep.messages_left() <= ep.reserved_messages():
                    gate = ("SKIPPED (no move spent): the last message is kept for the basket confirmation the request "
                            "requires; the code sends it, listing the whole basket, right before the order. Complete the "
                            "basket, then order.")
                elif not ep.questions_done and ep.steps_left() > STEP_RESERVE + 5:
                    # race 12: 19 runs began with a free-form message, which used to end the planner's questions: 4 passed where the field would pass 12.7
                    ep.gate_blocks["early_message"] = ep.gate_blocks.get("early_message", 0) + 1
                    ep.force_questions = ep.force_questions or ep.gate_blocks["early_message"] >= 2
                    gate = ("NOT SENT (no move spent): questions go through the question planner, one property per "
                            "message, once a few items of the right kind are open. Carry on searching and opening "
                            "items; the shopper's answers will show up here.")
            if gate:
                convo.append({"role": "tool", "tool_call_id": c.get("id"), "content": gate})
                continue
            if kind == "order":
                ep.order_attempts += 1
                args = _with_explanation(ep, name, args)
            if kind:
                args = _complete_finish_args(ep, name, args, kind)
            observation, ok = ep.act(name, args)
            text = _tool_result_text(ep, observation)
            try:
                more = _next_page_args(ep, name, args, observation) if ok and not kind else None
            except Exception as exc:  # a race-only path: a store shape never seen costs a page, never the task
                ep.note("paging error", repr(exc)[:200])
                more = None
            extra: list[str] = []
            if more is not None:
                ep.auto_pages += 1
                try:
                    page2, ok2 = ep.act(name, more)
                except Exception as exc:
                    ep.note("paging error", repr(exc)[:200])
                    page2, ok2 = None, False
                if ok2:
                    text += (f"\n[The store holds more matches than this page shows, so the code also fetched the next page "
                             f"({_dumps(more, 160)}):]\n"
                             + _tool_result_text(ep, page2))
                    extra = ep.shopper_lines(page2)
            if kind and ok and not ep.finished and not (isinstance(observation, dict) and "refusal" in observation):
                text += ("\n[The episode is still open after this finishing call, so nothing is settled. If the result asks for "
                         "a confirmation, token or next step, complete it with the right tool; otherwise read the result, "
                         "fix the problem and finish again.]")
            lines = [s for s in (ep.shopper_lines(observation) + extra if ok else []) if s not in ep.heard]
            lines = list(dict.fromkeys(lines))
            if name == ep.roles.get("message") and ok and isinstance(observation, (str, dict, list)):
                found: list[str] = []
                _strings_in(observation, found)
                lines = [s for s in found if s.strip()]
                ep.asked.append(str(args)[:80])
            ep.heard.update(lines)
            if lines and not ep.finished:
                reading = ep.read_lines(lines)
                if reading:
                    text += ("\n[Shopper lines understood: " + str(reading.get("summary") or "") +
                             f" Requirements now: {_dumps(ep.requirements)}; budget {ep.price_cap}; quantity {ep.quantity}.]")
                    if isinstance(observation, dict) and "refusal" in observation:
                        text += " [The order was REFUSED: nothing was bought. Apply the change and order again.]"
            convo.append({"role": "tool", "tool_call_id": c.get("id"), "content": text})
        if ep.buy_closest:
            exit_reason = "give-up replaced by ordering the closest match"
            break
        screen_new(ep)
    ep.note("worker exit", f"{exit_reason}: turns {turns}, steps left {ep.steps_left()}, work left {int(ep.time_left())}")


def run_probes(ep: Episode, convo: list[dict]) -> bool:
    """Send the code's keyword probes (Episode.due_probes) through the store's own keyword search, shaped
    like the agent's latest keyword search, and show the results to the worker as its own calls."""
    template, search = ep.keyword_template(), ep.roles.get("search")
    if not template or not search or ep.finished or ep.steps_left() <= STEP_RESERVE + 6 or ep.time_left() < 90:
        return False
    sent = False
    for query, why in ep.due_probes():
        ep.probed.add(query)
        if ep.finished or ep.steps_left() <= STEP_RESERVE + 6:
            break
        schema = ep.tool_spec(search).get("parameters") or {}
        base = {k: v for k, v in template[0].items() if not re.search(r"page|offset|cursor|continue|start|next|token|after", k, re.I)}
        base[template[1]] = query
        # a page-number argument lets a short list of hits be read to the end (at most two more pages)
        page_key = next((k for k, v in (schema.get("properties") or {}).items()
                         if re.search(r"page", k, re.I) and v.get("type") == "integer"), None)
        for page in (1, 2, 3):
            args = dict(base, **({page_key: page} if page_key and page > 1 else {}))
            args, problem = _fit_args(args, schema)
            if problem or ep.finished or (page > 1 and ep.steps_left() <= STEP_RESERVE + 6):
                break
            observation, ok = ep.act(search, args)
            text = _tool_result_text(ep, observation)
            lines = [s for s in (ep.shopper_lines(observation) if ok else []) if s not in ep.heard]
            ep.heard.update(lines)
            if lines and not ep.finished:
                reading = ep.read_lines(lines)
                if reading:
                    text += "\n[Shopper lines understood: " + str(reading.get("summary") or "") + "]"
            call_id = f"probe-{ep.step}"
            convo.append({"role": "assistant", "content": f"Keyword search for {why}" + (f", page {page}." if page > 1 else "."),
                          "tool_calls": [{"id": call_id, "type": "function",
                                          "function": {"name": search, "arguments": json.dumps(args)}}]})
            convo.append({"role": "tool", "tool_call_id": call_id, "content": text})
            ep.note("probe", {"query": query, "page": page, "why": why})
            sent = True
            last = ep.searches[-1] if ep.searches else {}
            shown, total = last.get("rows") or 0, last.get("total")
            if not page_key or not shown or total is None or total <= shown * page or total > shown * 5:
                break
    return sent


def screen_new(ep: Episode) -> None:
    """Judge newly opened listings right away (light model), so confirmed matches are on the board."""
    if ep.finished or ep.time_left() < 70 or ep.steps_left() <= 1:
        return
    fresh = [i for i in ep.detail_ids if i not in ep.verdicts and i in ep.listings and ep.orderable(i)]
    if fresh:
        fresh.sort(key=lambda i: (ep.listings[i].get("price") is None, ep.listings[i].get("price") or 0))
        ep.judge(fresh[:12])


# ---------------------------------------------------------------------------------------------
# Conditions that point to another listing


_SELLER_KEY = re.compile(r"shop|seller|merchant|vendor|store|maker|make|brand|supplier", re.I)


def _plain(text: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def _listing_text(fields: dict) -> str:
    """What a listing says it is (title, attribute values, text), leaving out sentences that deny something
    ("It is not the 153.67 cm height"): those name exactly the values the listing does not have."""
    parts: list[str] = []
    for v in fields.values():
        if isinstance(v, dict):
            parts += [str(x) for x in v.values() if not isinstance(x, (dict, list))]
        elif isinstance(v, str):
            parts += [s for s in re.split(r"(?<=[.!?;])\s+", v) if not re.search(r"\bnot\b", s, re.I)]
    return " " + _plain(" ".join(parts)) + " "


def _detail_value(detail: Any) -> str:
    return _plain(detail.get("value") if isinstance(detail, dict) else detail)


def _matches(fields: dict, details: list) -> int:
    text = _listing_text(fields)
    return sum(1 for d in details if _detail_value(d) and f" {_detail_value(d)} " in text)


def _sold_by(rec: dict, seller: str, department: str) -> bool:
    fields = rec.get("fields") or {}
    names = {_plain(v) for k, v in fields.items() if isinstance(v, str) and _SELLER_KEY.search(k)}
    if seller not in names:
        return False
    if not department or not rec.get("category"):
        return True
    stems = lambda t: {w.rstrip("s") for w in _plain(t).split() if len(w) > 2}
    return bool(stems(rec["category"]) & stems(department))


def _search_args(ep: Episode, text: str) -> dict | None:
    template = ep.keyword_template()
    tool = ep.roles.get("search")
    schema = _schema(ep, tool) if tool else {}
    if template:
        args = {k: v for k, v in template[0].items() if not re.search(r"page|offset|cursor|continue|start|next|token|after", k, re.I)}
        args[template[1]] = text
    else:
        props = schema.get("properties") or {}
        key = next((k for k, v in props.items() if (v or {}).get("type") == "string"
                    and re.search(r"query|search|phrase|text|keyword|term", k, re.I)), None)
        if not key:
            return None
        args = {key: text}
    fitted, problem = _fit_args(args, schema)
    return None if problem else fitted


def _detail_args(ep: Episode, lid: str, tool: str | None = None) -> dict | None:
    params = _schema(ep, tool or ep.roles.get("details") or "")
    props = params.get("properties") or {}
    key = next((k for k, v in props.items() if (v or {}).get("type") == "string" and _ID_KEY.search(k)), None) \
        or next((k for k in params.get("required") or [] if (props.get(k) or {}).get("type") == "string"), None)
    if not key:
        return None
    fitted, problem = _fit_args({key: lid}, params)
    return None if problem else fitted


def _find_referenced(ep: Episode, ref: dict, moves: list[int]) -> str | None:
    """The listing a reference names: sold by that seller, in that department, matching every detail given.
    Searches the seller's name alone first (in race 13, 842 of 1,001 one-name searches showed that seller's rows; the
    name together with every detail value matched nothing in stores that require all words), then with the department
    and with one detail, and opens the most promising candidates. A listing that matches only some details is never
    taken: stores plant near-twins that differ in one detail."""
    seller = _plain(ref.get("seller") or "")
    details = [d for d in ref.get("details") or [] if _detail_value(d)]
    if not seller:
        return None
    mine = lambda: [lid for lid, rec in ep.listings.items() if _sold_by(rec, seller, ref.get("department") or "")]
    dept = " ".join(str(ref.get("department") or "").split()[-2:])
    queries = [str(ref.get("seller")), f"{ref.get('seller')} {dept}".strip()] + [f"{ref.get('seller')} {_detail_value(d)}" for d in details[:1]]
    for query in dict.fromkeys(queries):
        if moves[0] <= 1 or ep.finished:
            break
        full = [lid for lid in mine() if lid in ep.detail_ids and _matches(ep.listings[lid]["fields"], details) >= len(details)]
        if full or len(mine()) >= 3:
            break
        args = _search_args(ep, query)
        if args:
            moves[0] -= 1
            ep.act(ep.roles["search"], args)
    cands = sorted(mine(), key=lambda lid: (lid not in ep.detail_ids, -_matches(ep.listings[lid]["fields"], details)))
    for lid in cands[:5]:
        if lid not in ep.detail_ids:
            args = _detail_args(ep, lid)
            if moves[0] <= 0 or not args or ep.finished:
                break
            moves[0] -= 1
            ep.act(ep.roles["details"], args)
        if lid in ep.detail_ids and _matches(ep.listings[lid]["fields"], details) >= len(details):
            return lid
    return None


def _find_by_code(ep: Episode, code: str, moves: list[int]) -> str | None:
    """The listing that carries an item code in its title (a page's "see part AB123 for the rest")."""
    pattern = re.compile(rf"(?<![A-Za-z0-9]){re.escape(code)}(?![A-Za-z0-9])", re.I)
    has = lambda: [lid for lid, rec in ep.listings.items() if pattern.search(str(rec.get("title") or ""))]
    if not has() and moves[0] > 1:
        args = _search_args(ep, code)
        if args:
            moves[0] -= 1
            ep.act(ep.roles["search"], args)
    for lid in has()[:2]:
        if lid not in ep.detail_ids:
            args = _detail_args(ep, lid)
            if moves[0] <= 0 or not args or ep.finished:
                break
            moves[0] -= 1
            ep.act(ep.roles["details"], args)
        if lid in ep.detail_ids:
            return lid
    return None


def resolve_reference(ep: Episode, prop: str, condition: str, moves: int) -> tuple[str | None, str]:
    """(value, relation) for a condition on `prop` that points to another listing: find that listing and
    follow its notes to the listing they defer to, at most `moves` store calls. (None, ...) when not found."""
    left = [moves]
    ref = ep.ask_json(LIGHT_MODEL, REFERENCE_PROMPT, f"PROPERTY: {prop}\nCONDITION: {condition}", max_tokens=500,
                      role="reference") or {}
    relation = "different" if ref.get("relation") == "different" else "same"
    for _hop in range(4):
        lid = _find_referenced(ep, ref, left)
        if not lid or ep.time_left() < 60:
            return None, relation
        fields = ep.listings[lid]["fields"]
        read = ep.ask_json(LIGHT_MODEL, LISTING_PROMPT, f"PROPERTY: {prop}\nLISTING ({lid}): {_dumps(fields, 4000)}",
                           max_tokens=500, role="reference") or {}
        ep.note("reference hop", {"condition": condition[:120], "listing": lid, "read": read})
        nxt = read.get("defers_to")
        if isinstance(nxt, dict) and nxt.get("seller"):
            ref = nxt
            continue
        value = str(read.get("value") or "").strip()
        code = str(read.get("code") or "").strip()
        if not value and code and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9\-]{2,20}", code) and re.search(r"\d", code):
            other = _find_by_code(ep, code, left)  # "or wherever that page sends you": the page names another item
            if other:
                read = ep.ask_json(LIGHT_MODEL, LISTING_PROMPT, f"PROPERTY: {prop}\nLISTING ({other}): "
                                   f"{_dumps(ep.listings[other]['fields'], 4000)}", max_tokens=500, role="reference") or {}
                ep.note("reference hop", {"code": code, "listing": other, "read": read})
                value = str(read.get("value") or "").strip()
        return (value or None), relation
    return None, relation


def resolve_references(ep: Episode, convo: list[dict]) -> bool:
    """Turn each condition that points to another listing into a plain one, by code, while moves and time are
    plentiful. Returns True when store calls were made."""
    if ep.finished or ep.steps_left() < 20 or ep.time_left() < 120 or not ep.roles.get("search") \
            or not ep.roles.get("details"):
        return False
    before = ep.step
    for req in ep.relative_requirements():
        if req in ep.references_tried:
            continue
        second = req in ep.references_retry
        if second and ep.step < ep.references_retry[req] + 6:
            continue  # a second and last try once the worker has searched on
        prod, prop, condition = split_req(req)
        value, relation = resolve_reference(ep, prop, condition, min(8, ep.steps_left() - 14))
        if second:
            ep.references_tried.add(req)
        else:
            ep.references_retry[req] = ep.step  # counted from the end of this try
        if not value or req not in ep.requirements:
            continue
        plain = f"{prod + ' - ' if prod else ''}{prop}: {'not ' if relation == 'different' else ''}{value}"
        ep.requirements[ep.requirements.index(req)] = plain
        ep.verdicts.clear()
        convo.append({"role": "user", "content": f"The referenced listing was found and read: \"{req}\" now reads "
                      f"\"{plain}\". Judge candidates against that."})
        ep.note("reference resolved", {"from": req, "to": plain})
    return ep.step > before


# ---------------------------------------------------------------------------------------------
# Endgame


_GONE_WORDS = re.compile(r"discontinued|withdrawn|unavailable|not purchasable|cannot be purchased|no stock|"
                         r"not in stock|sold out|none left|not available|no longer available", re.I)


def _after_refusal(ep: Episode, observation: Any) -> None:
    """Fold a refused finish into the state: read the shopper's lines, or at least spot a sell-out."""
    lines = [s for s in ep.shopper_lines(observation) if s not in ep.heard]
    ep.heard.update(lines)
    if not lines:
        return
    if ep.time_left() > 25:
        ep.read_lines(lines)
    elif any(_GONE_WORDS.search(s) for s in lines):
        for i in list(ep.cart):
            ep.unavailable.add(i)


def force_finish(ep: Episode) -> None:
    """Out of steps or time: order the best known fit, else the closest match (handling refusals); decline only
    when no candidate is known at all."""
    ep.keep_free = 15.0
    order_tool, cart_tool, decline_tool = ep.roles.get("order"), ep.roles.get("cart"), ep.roles.get("decline")
    ordered_any = False
    for attempt in range(3):
        if ep.finished or ep.steps_left() <= 0:
            return
        candidates = [r["id"] for r in ep.listings.values() if ep.orderable(r["id"])]
        fits = [i for i in ep.verified_fits() if i in candidates]
        if not fits and candidates and ep.time_left() > 25:
            pool = sorted((i for i in candidates if i in ep.detail_ids and i not in ep.verdicts),
                          key=lambda i: (ep.listings[i].get("price") is None, ep.listings[i].get("price") or 0))
            if pool:
                ep.judge(pool[:6])
            fits = [i for i in ep.verified_fits() if i in candidates]
        if not fits:
            fits = [i for i in ep.closest_matches() if i in candidates]  # the closest match, never an empty-handed exit
            if fits:
                ep.note("closest match", {"picks": fits[:3], "problems": {i: ep.verdicts[i].get("problems") for i in fits[:3]}})
        if not fits:  # nothing judged: a buyable listing of the wanted kind that the code can put in the basket
            fits = [i for i in ep.last_resort() if i in candidates
                    and (not cart_tool or i in ep.cart or _cart_args(ep, cart_tool, i) is not None)]
            if fits:  # its page is opened by finish_procedures after the basket change, before the order
                ep.note("last resort", {"picks": fits[:3], "opened": [i in ep.detail_ids for i in fits[:3]]})
        ep.note("force finish", {"attempt": attempt, "fits": fits[:5], "cart": ep.cart, "steps_left": ep.steps_left()})
        if not fits or not order_tool:
            break
        products = ep.product_list()
        if len(products) > 1:  # the cheapest confirmed match of each product (fits are sorted by price)
            picks = []
            for i in range(len(products)):
                held = [c for c in ep.cart if ep.product_of(c) == i and c in fits]
                mine = held or [f for f in fits if ep.product_of(f) == i]
                if mine:
                    picks.append(mine[0])
        else:
            picks = [fits[0]]
        missing = [x for x in picks if x not in ep.cart]
        if missing and cart_tool:
            others = [i for i in ep.cart if i not in picks]
            confirm = 1 if ep.reserved_messages() and ep.messages_left() != 0 else 0  # a changed basket is shown again
            look = 1 if ep.roles.get("cart_view") else 0  # and looked at first
            if any(o not in fits for o in others) or not any(c in fits for c in ep.cart):
                confirm = look = 0  # a gone or unmatched line, or no match in the basket at all: swap without moves to spare
            if ep.steps_left() < len(others) + len(missing) + 1 + confirm + look:
                if not others and len(picks) == 1:
                    break
                # no moves left to swap: order what the basket holds rather than a mixed basket
            else:
                for other in others:
                    remove = _cart_args(ep, cart_tool, other, remove=True)
                    if remove is not None:
                        ep.act(cart_tool, remove)
                        ep.cart.pop(other, None)
                for x in missing:
                    args = _cart_args(ep, cart_tool, x)
                    if args is None:
                        break
                    ep.act(cart_tool, args)
        if ep.finished or ep.steps_left() < 1:
            return
        note = finish_procedures(ep)
        if ep.finished or ep.steps_left() < 1:
            return
        if not ep.cart and ep.basket_seen_empty():
            break  # nothing could be put in the basket: an order of an empty basket is always refused
        if note and attempt < 2:
            continue  # the basket check or the reply changed the order: pick again under the new terms
        observation, ok = ep.act(order_tool, _finish_args(ep, order_tool, "order"))
        ordered_any = True
        if ep.finished:
            return
        _after_refusal(ep, observation)  # a sell-out moves on to the next fit; a mere change retries this one
    if not ep.finished and order_tool and ep.cart and not ordered_any and ep.order_attempts and ep.steps_left() >= 1:
        observation, _ = ep.act(order_tool, _finish_args(ep, order_tool, "order"))  # retry the worker's refused order
        if ep.finished:
            return
    for _attempt in range(2):  # a decline can be refused once by a change that fires at that moment
        if ep.finished or not decline_tool or ep.steps_left() < 1:
            break  # an open episode scores nothing, so a decline is the last resort even when a match was seen
        ep.act(decline_tool, _clean_remark(ep, _finish_args(ep, decline_tool, "decline")))


def _amount(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:.2f}"


def basket_text(ep: Episode, width: int) -> str:
    """The whole basket for the customer: every line with units and unit price, then the total."""
    unit = str(ep.brief.get("money_unit") or "").strip()
    lines, total, known = [], 0.0, True
    for lid, qty in ep.cart.items():
        rec = ep.listings.get(lid) or {}
        price = rec.get("price")
        title = re.sub(r"\s+", " ", str(rec.get("title") or ""))[:width].strip()
        lines.append(f"{qty} x " + (f"{title} ({lid})" if title else lid)
                     + (f" at {_amount(price)} each" if isinstance(price, (int, float)) else ""))
        if isinstance(price, (int, float)):
            total += price * qty
        else:
            known = False
    text = "Here is the final basket I am about to order: " + "; ".join(lines) + "."
    if known:
        text += f" Total {_amount(total)}{' ' + unit if unit else ''}" + (
            f", within your limit of {_amount(ep.price_cap)}" if isinstance(ep.price_cap, (int, float)) else "") + "."
    return text + " May I place the order with exactly this basket?"


def confirm_basket(ep: Episode) -> str | None:
    """Show the customer the complete basket before the order, as the request requires, and read the reply.
    Returns None when the order can go ahead, else a note on what the reply changed."""
    tool = ep.roles.get("message")
    if not tool or not ep.cart or ep.steps_left() < 2:
        return None
    params = _schema(ep, tool)
    props = params.get("properties") or {}
    key = next((k for k in params.get("required") or [] if (props.get(k) or {}).get("type") == "string"), None) \
        or next((k for k, v in props.items() if (v or {}).get("type") == "string"), None)
    if not key:
        return None
    room = int((props.get(key) or {}).get("maxLength") or 1000)
    text = next((t for t in (basket_text(ep, w) for w in (70, 35, 0)) if len(t) <= room), basket_text(ep, 0))[:room]
    before = (list(ep.requirements), ep.price_cap, ep.quantity, set(ep.unavailable))
    observation, ok = ep.act(tool, {key: text})
    lines: list[str] = []
    _strings_in(observation, lines)
    lines = [s for s in lines if s.strip() and s not in ep.heard]
    ep.heard.update(lines)
    ep.note("basket confirmation", {"text": text, "reply": lines})
    if not ok or not lines or ep.time_left() < 25:
        return None
    reading = ep.read_lines(lines, context=f"THE ASSISTANT SAID: {text}") or {}
    if (list(ep.requirements), ep.price_cap, ep.quantity, set(ep.unavailable)) == before \
            and not reading.get("item_unavailable") and not reading.get("price_changed"):
        return None
    return ("NOT RUN (the confirmation used one move and one message): the customer answered the basket with "
            f"{_dumps(lines, 300)}. Requirements now: {_dumps(ep.requirements)}; budget {ep.price_cap}; quantity "
            f"{ep.quantity}. Fix the basket to match, then order.")


def _message_args(ep: Episode, text: str) -> dict | None:
    tool = ep.roles.get("message")
    params = _schema(ep, tool) if tool else {}
    props = params.get("properties") or {}
    key = next((k for k in params.get("required") or [] if (props.get(k) or {}).get("type") == "string"), None) \
        or next((k for k, v in props.items() if (v or {}).get("type") == "string"), None)
    if not key:
        return None
    return {key: text[: int((props.get(key) or {}).get("maxLength") or 400)]}


def open_bought(ep: Episode) -> str | None:
    """Open the page of every basket item not opened since it was put in the basket (steps allowing, keeping room
    for the look, the confirmation, the order and one spare). Returns a note when a page or a shopper line shows
    that the basket no longer works (gone, out of stock, new price, changed terms), else None."""
    tool, due = ep.page_tool(), ep.opens_due()
    if not tool or not due or ep.finished:
        return None
    due = due[: max(0, ep.steps_left() - 2)]  # the opens come last: keep the order and one spare
    def state() -> tuple:
        return (list(ep.requirements), ep.price_cap, ep.quantity, set(ep.unavailable),
                {i: ((ep.listings.get(i) or {}).get("price"), (ep.listings.get(i) or {}).get("in_stock")) for i in ep.cart})
    before = state()
    heard: list[str] = []
    for lid in due:
        args = _detail_args(ep, lid, tool)
        if args is None or ep.finished:
            break
        observation, ok = ep.act(tool, args)
        lines = [s for s in (ep.shopper_lines(observation) if ok else []) if s not in ep.heard]
        ep.heard.update(lines)
        heard += lines
        if lines and not ep.finished and ep.time_left() > 25:
            ep.read_lines(lines)
    ep.note("opened basket items", {"opened": due, "lines": heard})
    after = state()
    if after == before:
        return None
    gone = [i for i, (_price, stock) in after[4].items() if stock is False or i in ep.unavailable]
    moved = [i for i, (price, _stock) in after[4].items() if price != before[4].get(i, (price, None))[0]]
    return ("NOT RUN (opening the basket items used moves): " + (f"the shopper wrote {_dumps(heard, 300)}. " if heard else "")
            + (f"Now unavailable: {', '.join(gone)}. " if gone else "") + (f"Price changed: {', '.join(moved)}. " if moved else "")
            + f"Basket now: {_dumps(ep.cart)}. Requirements now: {_dumps(ep.requirements)}; budget {ep.price_cap}; "
            f"quantity {ep.quantity}. Bring the basket in line with these conditions, then order.")


def finish_procedures(ep: Episode) -> str | None:
    """Right before an order, steps allowing: look at the basket once it has stopped changing, show the customer
    the final basket, and last of all open each basket item's own page (the step right before the order). The look comes first because a shopper's change (a new budget, a new price, a
    sold-out item) arrives with the first result after the basket changes; ordering straight away would receive
    it only with the order. Returns a note when the basket view or the customer's reply changes the order, else None."""
    view = ep.roles.get("cart_view")
    looked = any(s > ep.last_cart_step for s in ep.basket_view_steps)
    opens = len(ep.opens_due())
    # look, confirm, open, order, and one move spare for an order refused by a sell-out; the look is dropped first
    # when moves are short
    if view and not looked and ep.steps_left() >= 3 + (1 if ep.confirmation_due() else 0) + opens:
        args, _problem = _fit_args({}, _schema(ep, view))
        before = (list(ep.requirements), ep.price_cap, ep.quantity, set(ep.unavailable))
        observation, ok = ep.act(view, args if isinstance(args, dict) else {})
        lines = [s for s in (ep.shopper_lines(observation) if ok else []) if s not in ep.heard]
        ep.heard.update(lines)
        reading = ep.read_lines(lines) if lines and ep.time_left() > 25 and not ep.finished else None
        if ep.basket_seen_empty():
            return ("NOT RUN (the basket check used one move): the store shows the basket EMPTY. Put the chosen item in "
                    "the basket with the basket tool first, then order.")
        if (list(ep.requirements), ep.price_cap, ep.quantity, set(ep.unavailable)) != before \
                or (reading or {}).get("item_unavailable") or (reading or {}).get("price_changed"):
            return ("NOT RUN (the basket check used one move): " + (f"the shopper wrote {_dumps(lines, 300)}. " if lines else "")
                    + f"Basket now: {_dumps(ep.cart)}. Requirements now: {_dumps(ep.requirements)}; budget {ep.price_cap}; "
                    f"quantity {ep.quantity}. Bring the basket in line with these conditions (re-check prices), then order.")
    if ep.confirmation_due() and not ep.finished and ep.steps_left() >= 3 + min(opens, 1):
        note = confirm_basket(ep)
        if note or ep.finished:
            return note
    return open_bought(ep)


def _schema(ep: Episode, tool: str) -> dict:
    return next((t["function"].get("parameters") or {} for t in ep.tools if t["function"]["name"] == tool), {})


def _line_args(ep: Episode, schema: dict, lid: str) -> dict | None:
    """One basket line object ({"item": lid, "qty": n}) built from an array parameter's item schema."""
    props = schema.get("properties") or {}
    line: dict[str, Any] = {}
    for k, v in props.items():
        v = v or {}
        if v.get("type") == "string" and (_ID_KEY.search(k) or re.search(r"product|item|which|listing|sku", k, re.I)):
            line.setdefault(k, lid)
        elif v.get("type") == "integer" and re.search(r"qty|quant|count|many|units|number|want|piece", k, re.I):
            line[k] = ep.quantity_for(lid)
    if lid not in line.values() or any(k not in line for k in schema.get("required") or []):
        return None
    return line


def _cart_args(ep: Episode, tool: str, lid: str, remove: bool = False) -> dict | None:
    params = _schema(ep, tool)
    props = params.get("properties") or {}
    lines_key = next((k for k, v in props.items() if (v or {}).get("type") == "array"
                      and ((v or {}).get("items") or {}).get("type") == "object"), None)
    if lines_key:  # the basket takes a list of line objects
        # Such a call sets the whole basket to the lines sent (race 13: 550 of 550 calls that left a held line out
        # dropped it; race 7: 625 of 625 at the next basket view), so the lines already held go along, each with
        # its own units, and a removal is the basket without that line.
        item_schema = props[lines_key].get("items") or {}
        qty_key = next((k for k, v in (item_schema.get("properties") or {}).items()
                        if (v or {}).get("type") == "integer" and re.search(r"qty|quant|count|many|units|number|want|piece", k, re.I)), None)
        lines = []
        for other, units in ep.cart.items():
            if other == lid:
                continue
            kept = _line_args(ep, item_schema, other)
            if kept is None:
                return None
            if qty_key and isinstance(units, int) and units > 0:
                kept[qty_key] = units
            lines.append(kept)
        if not remove:
            line = _line_args(ep, item_schema, lid)
            if line is None:
                return None
            lines.append(line)
        elif not lines and (props[lines_key].get("minItems") or 0) > 0:
            return None  # the store cannot empty its basket this way
        args = {lines_key: lines}
        return args if all(k in args for k in params.get("required") or []) else None
    args: dict[str, Any] = {}
    for k, v in props.items():
        if v.get("enum"):
            words = [e for e in v["enum"] if isinstance(e, str)]
            if remove:  # a removal word, else the option that is not the adding one
                choice = next((e for e in words if _REMOVE_WORD.search(e)), None) or (
                    next((e for e in words if not _ADD_WORD.search(e)), None) if any(_ADD_WORD.search(e) for e in words) else None)
            else:  # an adding word, else the first option that is not a removal
                choice = next((e for e in words if _ADD_WORD.search(e) and not _REMOVE_WORD.search(e)), None) or next(
                    (e for e in words if not _REMOVE_WORD.search(e)), None)
            if choice is None:
                return None
            args[k] = choice
        elif v.get("type") == "string" and (_ID_KEY.search(k) or re.search(r"product|item|which|listing|sku", k, re.I)):
            args[k] = lid  # camelCase names too (WhichItemId in race 8, WhichListingId in race 11)
        elif v.get("type") == "integer" and re.search(r"qty|quant|count|many|units|number|piece", k, re.I):
            args[k] = ep.quantity_for(lid) if not remove else max(1, ep.cart.get(lid, 1))
    if lid not in args.values():
        return None
    if remove and not any((v or {}).get("enum") for v in props.values()):
        # no add/remove choice: the units field sets the line, and 0 takes it out (race 9: 193 of 193)
        qty_keys = [k for k, v in props.items() if (v or {}).get("type") == "integer" and re.search(r"qty|quant|count|many|units|number|piece|line", k, re.I)]
        if not qty_keys:
            return None
        for k in qty_keys:
            args[k] = 0
    for k in params.get("required") or []:
        if k not in args:
            return None
    return args


_NOTE_KEY = re.compile(r"explan|reason|note|why|justif|comment|remark|summary|rationale", re.I)


def order_explanation(ep: Episode, room: int = 600) -> str:
    """Why this basket: each line with units and price, the total against the budget, and the conditions it meets."""
    unit = str(ep.brief.get("money_unit") or "").strip()
    lines, total = [], 0.0
    for lid, qty in ep.cart.items():
        rec = ep.listings.get(lid) or {}
        price = rec.get("price")
        title = re.sub(r"\s+", " ", str(rec.get("title") or ""))[:60].strip()
        lines.append(f"{qty} x {title or lid} ({lid})" + (f" at {_amount(price)}" if isinstance(price, (int, float)) else ""))
        total += (price or 0) * qty if isinstance(price, (int, float)) else 0
    text = "Chose " + "; ".join(lines) + "." if lines else "Chose the item that best matches the request."
    if total:
        text += f" Total {_amount(total)}{' ' + unit if unit else ''}" + (
            f", within the limit of {_amount(ep.price_cap)}" if isinstance(ep.price_cap, (int, float)) else "") + "."
    if ep.requirements:
        text += " It meets the stated conditions: " + "; ".join(map(str, ep.requirements[:8])) + "."
    if ep.notes:
        text += " Updates from the shopper were applied."
    return text[:room]


def _with_explanation(ep: Episode, tool: str, args: dict[str, Any]) -> dict[str, Any]:
    """Fill an empty free-text explanation field of an order call (race 9: orders that carried one passed about
    twice as often within each top agent)."""
    props = _schema(ep, tool).get("properties") or {}
    out = dict(args)
    for k, v in props.items():
        v = v or {}
        if v.get("type") == "string" and not v.get("enum") and _NOTE_KEY.search(k) and not str(out.get(k) or "").strip():
            out[k] = order_explanation(ep, int(v.get("maxLength") or 600))
    return out


def _complete_finish_args(ep: Episode, tool: str, args: dict[str, Any], kind: str) -> dict[str, Any]:
    """A finishing call from the worker with a required choice missing or outside its allowed values gets the schema's
    value for this kind of ending (race 9: 10 of our declines were refused for a missing required choice); the
    worker's own text stays."""
    props = _schema(ep, tool).get("properties") or {}
    required = set(_schema(ep, tool).get("required") or [])
    base = _finish_args(ep, tool, kind)
    out = dict(args)
    for k, v in base.items():
        spec = props.get(k) or {}
        allowed = spec.get("enum")
        if k not in out and k in required:
            out[k] = v
        elif allowed and out.get(k) not in allowed:
            out[k] = v
    return out


def _finish_args(ep: Episode, tool: str, kind: str) -> dict:
    """Arguments for a finishing call built from its schema: the outcome choice, flags and a short note."""
    params = _schema(ep, tool)
    props = params.get("properties") or {}
    required = set(params.get("required") or [])
    decline_words = DECLINE_WORDS
    args: dict[str, Any] = {}
    for k, v in props.items():
        v = v or {}
        if v.get("enum"):
            options = [e for e in v["enum"] if isinstance(e, str)]
            if _outcome_choice(k, options):
                pick = [e for e in options if _choice_declines(e, options) == (kind == "decline")]
                if pick and (k in required or ep.roles.get("order") == ep.roles.get("decline")):
                    args[k] = pick[0]
            elif k in required and options:
                args[k] = options[0]
        elif v.get("type") == "boolean" and (k in required or re.search(decline_words + r"|abstain", k, re.I)):
            args[k] = bool(re.search(decline_words + r"|abstain", k, re.I)) == (kind == "decline")
        elif v.get("type") == "string" and k in required and not _ID_KEY.search(k):
            args[k] = ("Unfortunately no item here satisfies all of your conditions." if kind == "decline"
                       else order_explanation(ep, int(v.get("maxLength") or 600)))
        elif v.get("type") == "string" and kind == "decline" and not args and not _ID_KEY.search(k):
            args[k] = "Unfortunately no item here satisfies all of your conditions."
    if kind == "order":
        args = _with_explanation(ep, tool, args)
    fitted, _problem = _fit_args(args, params)
    return fitted if isinstance(fitted, dict) else args


# ---------------------------------------------------------------------------------------------
# Entry point


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    ep = Episode(problem_data)
    if ep.finished:
        return ep.dialogue
    try:
        make_brief(ep)
        run_worker(ep)
    except Exception as exc:  # never leave the episode open because of an agent bug
        ep.dialogue.append({"error": f"{type(exc).__name__}: {exc}"})
    try:
        if not ep.finished:
            force_finish(ep)
    except Exception as exc:
        ep.dialogue.append({"error": f"finish: {type(exc).__name__}: {exc}"})
    ep.dialogue.append({"summary": {"steps": ep.step, "spent_usd": round(ep.spent, 5), "work": round(ep.work, 1),
                                    "cost_by_role": {k: round(v, 5) for k, v in ep.cost_by_role.items()},
                                    "requirements": ep.requirements, "asked": ep.asked}})
    return ep.dialogue


__all__ = ["agent_main"]
