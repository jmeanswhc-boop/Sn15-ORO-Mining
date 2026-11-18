from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

from src.agent.proxy_client import ProxyClient

MAIN_MODEL = "anthropic/claude-haiku-5.5"
BACKUP_MODEL = "openai/gpt-6-luna"
SETUP_MODEL = "anthropic/claude-sonnet-5.5"
JUDGE_MODEL = "anthropic/claude-sonnet-5.5"
READ_MODEL = "anthropic/claude-haiku-5.5"

TASK_SECONDS = 265.0
WRAP_UP_SECONDS = 70.0
FORCE_SECONDS = 28.0
COST_SOFT = 0.032
COST_HARD = 0.036
COST_STOP = 0.038
COST_CAP = 0.042
OBS_CHARS = 16000
MAX_TURNS = 80

SYSTEM_PROMPT = """You act for one shopper in a simulated store, and the tools you are given are the only way you can do anything.

GRADING
Check: how this task is marked. Then: know it gets 1 or 0, and 1 only for a finish with nothing wrong in it. The bought item has to satisfy each requirement the shopper wrote, each hidden need they disclose once asked, and each change they announce while you are busy; the quantity and the budget have to be correct; when the request wants the cheapest (or best), that is what you buy; and every rule written in the request is obeyed.
Check: no listing in the store can satisfy all of that. Then: declining is the correct finish, done in precisely the manner the request describes.

HOW THE STORE BEHAVES
Check: you are about to use a tool. Then: count it as one step, even when it fails or comes back with an error; steps run out, while thinking on your side costs nothing.
Check: some calls do not need one another's results (opening the details of a few listings together, say). Then: you may put them in a single reply, where they run one after another; an order or a decline, however, always travels alone.
Check: a new store. Then: expect its tool names, arguments and result layouts to be its own; study each tool description and JSON schema, send exactly the argument names and permitted values they define, and work out from the first results which tool reveals which facts (price, stock, detailed properties).
Check: the category labels on listings. Then: learn early which label the store gives the requested kind of product; if neighbouring labels exist, the right one is the label whose words match the request taken as a whole, and you search and buy inside it.
Check: a search tool that may hold things back (only so many rows or pages visible, ordering by id instead of relevance or price, all words required, filters that work in one mode only). Then: set the reported total against the rows you really see, and keep narrowing (sharper words, category or price filters, further pages) until the entire relevant set has passed before you.

THE SHOPPER'S HIDDEN NEEDS
Check: the shopper keeps needs they never wrote down. Then: remember a need comes out only when a message asks about that property, and every message is answered solely about the FIRST thing in it; so ask about exactly ONE property per message, as a plain question of at most 15 words that names the property the way the listings write it ("Which <property> would you like for the <product>?"). Never offer "A or B", never send a statement, give no instructions, mention no lists, labels, positions or rules, and never bundle two questions. When the shopper says they did not understand, ask about a different property instead of rewording the same one.
Check: an answer says they don't mind, have no preference, or anything is fine. Then: that property is free.
Check: an answer names a preference or a limit. Then: from that point on it is a hard requirement, worded exactly as given (numeric limits too).
Check: a planner is available. Then: open the details of two or three listings of the requested kind in your opening moves, since that is when the questions get chosen and sent on your behalf; the shopper's answers come back in the conversation together with what they mean, and every need the shopper reveals binds you from then on, so search again with it joined to the stated values.
Check: you have to do the asking yourself. Then: open the details of two or three listings first, then ask about properties the story points to (whom it is for, how it should look, where or how it gets used), and after that about properties the listings carry that differ between candidates. Skip anything the request already settles, and stay inside the message budget.

RULES ABOUT HOW YOU WORK
Check: the request lays down rules for your procedure (how many listings to view or compare before choosing, what to check, what to tell the shopper, what to write in a remark, how to decline). Then: they are graded just like requirements, so carry out each one literally and fully; a rule to look at or open N listings means N listings that fit completely, each opened with the details tool.
Check: the shopper puts a question to you (to confirm a change, for instance). Then: it gets answered for you while messages are left; the change itself binds you either way.
Check: something the shopper would only like ("ideally", "would be nice", "if possible"). Then: it is no requirement; let it decide only between items that already meet everything, and never let it beat a cheapest rule.
Check: the request leaves open something only the shopper can tell you (no price limit at all, or two wishes that might not both be possible so the shopper has to say which counts more). Then: ask and find out before you choose.

WHEN THINGS CHANGE
Check: the shopper alters something mid-task (a budget moved up or down, a requirement dropped, replaced or added, a price rising, an item selling out). Then: know that one shopper line may carry several changes and may turn up inside any tool result (an extra field holding sentences) or alongside an order attempt; every change binds from that moment, so test your pick and your backups against it again, and if the request wants the cheapest or best, choose afresh from the new acceptable set.
Check: an order attempt comes back with a "refusal" field (even if the inner result looks like success). Then: the order was REFUSED, the episode goes on and the cart is as before; read the refusal, apply each change it names, take out of the cart whatever is no longer acceptable (a sold-out item, for example), add the next verified fit, look at stock and price again, and order once more.
Check: an item may sell out or change price with no warning. Then: look at stock and price again right before each order attempt; an item whose current price times quantity exceeds the current budget has stopped fitting.

JUDGING A LISTING (the score depends on this)
Check: whether a listing meets a requirement. Then: it does only when the listing's own property for it gives the value. Keys are spelled in many ways (spaces, dots, underscores, prefixes) and one property may show up under several keys; any key naming that property counts. Values are matched by sense: a single element of a comma-separated list is enough, singular equals plural, close synonyms are accepted, case is ignored. A word that merely opens with the same letters is not a match, and neither is a wider or different class (girls is not women, plain unisex is not female).
Check: the listing has a property key for that requirement. Then: its value decides, and the listing fails if that value differs even when the title or description says the wanted value; only a listing with no such key at all lets the title or description serve as evidence.
Check: the value sits under some other, related property. Then: it does not count (a length field cannot satisfy a style requirement, the material of the lining cannot satisfy the material).
Check: a number. Then: compare it as a number once the units agree; a limit includes its edge value; a range passes only if all of it meets the limit.
Check: an exclusion ("leave out anything with X") or a closed list ("A or B and nothing else"). Then: honour it.
Check: the requested kind of product. Then: the listing's category settles it, and no property needs to repeat the category's name. The item must still be that kind: where the store labels listings with categories, it belongs in the category whose label matches the request's wording as a whole (asking for "A or B" points to a category called "A & B", not to a narrower neighbour called "B"). When the request is for things on hand, an item out of stock fails, and price times quantity has to stay inside the budget (equal passes).
Check: a property value that lists several options, the required one among them (a comma list). Then: that is a plain match, not mixed evidence; evidence is mixed only when the listing's own keys for the property give different values that cannot both hold, and only in that case, with no cheapest or best rule to decide, favour a candidate whose evidence is clean.

CHEAPEST OR BEST
Check: the request asks for the cheapest. Then: the order has to be the lowest-priced item among ALL items meeting every requirement, and spotting one fitting item proves nothing. Use price filters where a tool offers them, push the price limit down until every remaining row is visible, sort the visible candidates by price yourself, and verify them from the cheapest up; the first that fits completely, with every cheaper one examined and rejected, is the answer. Keep the following fits as backups.
Check: the request asks for the best on some measure (most, largest, longest, highest and the like). Then: weigh ALL fitting items on that measure in the same way and take the extreme one; a listing that states no value for the measure cannot win.

FINISHING
Check: your choice so far. Then: hold a ranked list of verified fits with one backup at least (for each product, when the order has several), because the first pick may be sold out by checkout.
Check: an order attempt is next. Then: the cart must contain one line for each product the request wants (usually exactly one line), each the chosen item at its required quantity, and nothing more (take out any other line), and since your last cart change you must have read the chosen item's stock and price again and found it in stock and within budget.
Check: you are leaning towards a decline. Then: decline only once a thorough search turns up nothing that meets every requirement, and do it exactly as the request's decline rule says; if the request forbids offering substitutes, the decline remark may not name, describe or point at any listing: no ids, no titles, no prices.
Check: your remaining steps. Then: get the first order attempt in while seven or more steps are left, so that a refusal can still be dealt with; once 7 steps remain, only cart, stock-check and finishing tools are still offered. Make up your mind once rather than swapping the cart back and forth, because each cart change costs a step.
Reply: with one or more tool calls every time, and only short text: the requirements as they stand, the verified fits with price and stock, and your next move."""

ORDER_MEANING = "the tool that buys or checks out"
DECLINE_MEANING = "the tool that closes the task without a purchase"

SETUP_PROMPT = """Study the shopper's request and the store's tool list further down.
Check: the shape of your answer. Then: write ONE compact JSON object (no indentation), with nothing around it, holding exactly the keys below.
Check: "products". Then: give a list with one object for each different kind of product the request wants bought in this order (usually one; two or three when the request plainly asks for several different products together). Each object holds:
  "product": the kind of product, in the request's own words;
  "quantity": how many units of a single listing to order, as an integer (1 unless the request asks for several); a thing normally sold as one item, a pair of shoes or a set for instance, counts as 1;
  "quantity_quote": the exact words of the request that state that number of units, or "" when it is 1;
  "requirements": one object per firm stated requirement on that product: {"property": the property in the request's words, "rule": the constraint with the shopper's exact value, limit, exclusion or closed list, "op": "is" | "is_not" | "at_least" | "at_most" | "other", "values": [each value exactly as written]}. Price, stock, quantity, budgets and mere wishes stay out of it, and so does the product kind or category: words that form part of the category's name (a season, an occasion or an audience inside the category name) are covered by the category and are not an extra property requirement. A sentence that only tells a story (why the shopper buys, what happened before) is not a requirement.
Check: a requirement whose value is set by a different listing in the store (the request points to that listing through its category, its seller and a few of its values, and wants one of its properties matched). Then: still list it under the product's requirements with op "other", and also add it to "references": {"product": the product it is for, "property": the property whose value must match, "category": the category named, "seller": the shop or brand named, "known": {each property: value the request gives to identify that listing}, "tolerance": the allowed difference in the request's words, or ""}; any remark about where that listing may lead is only story.
Check: "hint_properties". Then: list the product properties that a story sentence points to without stating a value (a remark about a shoe's tip points to toe style, a tall window to length, photos to colour or look, a child to an age property), most likely first, up to three for each product; when the order holds several products write each as "<product>: <property>". Leave out properties the requirements already settle.
Check: "price_cap". Then: the largest total price as a number, or null if the request states none.
Check: "currency". Then: the price unit the request uses, or null.
Check: "in_stock_only". Then: true if the request wants only items that are really on hand or in stock.
Check: "priority". Then: "cheapest" only if the request plainly asks for the cheapest or lowest-priced fitting option, "best" only if it plainly asks for the best by some measure, and "none" in every other case. Saying that money is tight or that every penny counts does not ask for the cheapest; such remarks only explain the price cap.
Check: "priority_detail". Then: for "best", the measure that "best" refers to; in every other case "".
Check: "message_budget". Then: how many messages the shopper lets you send, as an integer, or null when the request does not say.
Check: "step_budget". Then: how many moves or steps the request permits, as an integer, or null when the request does not say.
Check: "rules". Then: list, copied word for word, each procedural rule the request sets for the assistant (how many listings to view, what to check or report, how to decline, what to avoid, what to include). Lines about what the shopper will do on their own are story and not rules.
Check: "soft_wants". Then: list what the shopper would like without strictly needing it (said with ideally, preferably, would be nice, if possible), and keep those out of "requirements".
Check: "ask_first". Then: list what the agent can only get from the shopper and ought to ask about before choosing, such as the budget when the request names no price limit, which of two wishes weighs more when the request says both may not be possible (in that case both wishes also go into soft_wants and not into requirements), or anything the request says to confirm with the shopper. Leave it empty when there is nothing.
Check: every sentence of the request. Then: make sure it ends up somewhere: a requirement, the priority, a soft want, a rule, a budget, a hint, or plain story; a sentence that may or may not be a rule goes into "rules".
Check: "decline_no_names". Then: true if the request's rule for declining forbids offering, naming or suggesting substitutes or alternatives, false if it asks you to name something, null if the request has no such rule.
Check: "roles". Then: give an object that maps each role to the tool filling it, with tool names copied exactly from the list, or null where no tool fits:
  "search", "details", "view", "message", "cart", "cart_view", "order", "decline".
  "message" means the tool that sends the shopper a message, "cart" the tool that puts items into or takes them out of the cart, "order" """ + ORDER_MEANING + """, and "decline" """ + DECLINE_MEANING + """.
Check: "finish_choices". Then: if one tool both orders and declines through a field with fixed choices, give {"order": {"<field>": "<the choice that buys>"}, "decline": {"<field>": "<the choice that closes without buying>"}}, names and choices copied from that tool; otherwise {}.
Reply: with that single JSON object only.
"""

INTERPRET_PROMPT = """A shopper has just said something, and a shopping assistant needs to know what it means.
Check: the format. Then: write ONE JSON object and nothing more, using these keys.
Check: "meaning". Then: one short sentence on what the shopper means.
Check: "needs". Then: a list of new firm requirements on the product, each shaped "property: constraint" with the exact value, limit, exclusion or closed list the shopper gave.
Check: "no_preference". Then: a list of properties the shopper says they do not care about.
Check: "dropped". Then: a list of earlier requirements the shopper no longer needs.
Check: "new_budget". Then: the new highest total price as a number, or null.
Check: "unavailable". Then: true if the shopper says an item can no longer be bought (sold out, gone, out of stock, button disabled), otherwise false.
Check: "requirements_now". Then: the full list of firm product requirements as they stand after everything the shopper has said up to now, worked out again from the request, the earlier shopper lines and this one, each as "property: constraint". Every earlier requirement stays unless the shopper dropped or replaced it; for a replaced value only the newest survives; every newly revealed need is added. Price, stock, quantity and the product category do not belong here.
Check: "understood". Then: false when the shopper says they could not follow or understand the question and asks for it again in other words; otherwise true.
Check: "asked_us". Then: true only when the shopper's new line puts a direct question to the assistant or asks it to confirm something; a stated preference, limit or change is not a question.
Check: "reply". Then: if the shopper asks the assistant something or wants a confirmation, a short plain answer to send back that responds and states the change in substance (confirming the new value, say); in any other case "".
Check: the line holds several changes. Then: report all of them; a requirement swapped for another counts as dropped (the old one) and as a new need (the new one).
Check: the shopper says which of two wishes matters more. Then: the wish they prefer is a need.
Check: the shopper names a budget. Then: it is new_budget, and it replaces the old one whether higher or lower, even if the shopper calls it good news.
Check: a mere wish ("ideally", "would be nice"). Then: it is not a need.
Check: the wording. Then: read the shopper literally and kindly; homely, figurative or indirect phrasing that names a value means the shopper firmly wants that value, and a value counts as excluded only when the shopper clearly says to avoid it. A number with a comparison (at most, at least, under, over, no more than) is a limit and keeps its unit.
Check: the shopper only says they don't mind, have no preference, or anything works. Then: place the property that was asked about in no_preference.
Check: the order holds several products (the requirements so far are labelled "<product> - ..."). Then: label every need and every entry of requirements_now the same way, with the product the shopper's words or the question are about.
Reply: with the JSON object alone."""

VERIFY_PROMPT = """You are the last check before a shopping assistant places an order, and the task scores 1 only if the ordered item meets every requirement listed below; judge the item from the records shown and from nothing else.
Check: whether a requirement is met. Then: it is met when the item's own property for it gives the value. Keys are spelled many ways (spaces, dots, underscores, prefixes) and a property may sit under several keys; any key naming that property counts. Values match by sense: one element of a comma list suffices, singular/plural and close synonyms are fine, case is ignored, and the wanted value standing as whole words inside a longer value counts too ("white" inside "off white", "mandarin" inside "mandarin collar"). A word that only looks similar does not count.
Check: the title or description. Then: it counts only when the item has no property key for that requirement; if the item's own property says something else, the item fails even when the title says the value.
Check: a value under a different, related property. Then: it does not count.
Check: numbers. Then: compare them once units agree; limits include the edge; a range passes only if the whole range meets the limit.
Check: exclusions and closed lists. Then: they must hold.
Check: the kind of product. Then: the item must be what was asked for; where listings carry a category label, it must be the category whose label matches the request's wording as a whole, not a narrower or neighbouring one.
Check: the item's own property for a requirement gives a different value. Then: that requirement fails, whatever the title suggests.
Check: price and stock. Then: price times quantity must stay within the budget (equal is fine), and an item the records show as out of stock fails.
Check: a mere wish ("ideally", "would be nice") marked as not required. Then: it is not a requirement.
Check: records that disagree (on price or stock, for example). Then: the newest record wins.
Check: each requirement in turn. Then: give it "pass" when a record shows the value, "fail" only when the item's own property for it clearly gives another value (or the price, stock or kind of product rules it out), and "unproven" when no record says anything about it; a missing record is never a fail.
Reply: with ONE JSON object and nothing more: {"item": <the item's id>, "checks": [{"requirement": "...", "fact": "at most eight words of the record you relied on, or \"\"", "verdict": "pass" | "fail" | "unproven"}], "fits": true unless some check is "fail"}."""

PLAN_PROMPT = """You choose the questions a shopping assistant sends to a shopper. Each hidden need of the shopper is the value of one product property and comes out only when a message asks about exactly that property; a message is answered only about the first thing it asks, and questions on anything else draw a polite "don't mind". Pick up to QUESTION_COUNT different properties, the most promising first.
Check: the list below has things MUST BE LEARNED FROM THE SHOPPER that nobody has asked yet. Then: put them first and ask them plainly; for the budget, ask the most they want to spend; for a choice between two wishes, name both values with their properties, for example "Should no single listing offer both, which would you pick: the <value> <property> or the <value> <property>?".
Check: what comes next. Then: choose properties from the MENU below only, copying each name exactly as the menu writes it. Put first a property that a STORY HINT names or that the request's story points to indirectly (whom the item is for or who will use it, how it has to look, where or how it gets used); then follow the menu's order, which already ranks the properties carried by many listings with values that differ, preferring properties that say what the product is like (who it is made for, material, pattern, shape, type or style, closure, length, size of the product itself, function) over logistics and listing variants (warranty, condition, lists of available sizes or colours offered, option codes, years).
Check: a property the request already settles (its stated requirements, the price, stock, quantity), one already asked, one the shopper did not understand, or a near-duplicate of another pick (two sorts of style, two spellings of one property). Then: leave it out.
Check: the order holds several products. Then: pick properties of each product, and give the product with each pick.
Reply: with ONE JSON object and nothing more: {"questions": [{"product": "the product it is about", "property": "the menu name", "question": "only for a MUST BE LEARNED item: one short plain sentence; otherwise \"\""}, ...]}."""

AUDIT_PROMPT = """A shopping assistant is on the point of telling the shopper that nothing in the store fits; test the listing records below against the requirements first.
Check: whether a listing meets a requirement. Then: it does when the listing's own property for it gives the value (any key spelling, one element of a comma list, singular/plural and close synonyms are all fine; a wider or different class is not), and the title or description counts only when the listing has no key for that property.
Check: numbers, price and stock. Then: compare numbers once units agree with limits including the edge, keep the price within the budget, and reject an item shown out of stock.
Reply: with ONE JSON object and nothing more: {"possible_fits": [ids of up to 5 listings that look as if they meet every requirement, strongest first], "reason": "a single sentence"}."""

DECLINE_TEXT = "Sorry, nothing in the store meets all of your requirements, so I am not ordering anything."

_llm = ProxyClient(timeout=70, max_retries=1, retry_delay=1)
_env = ProxyClient(timeout=95, max_retries=2, retry_delay=1)

_ID_HINT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-:.]{3,63}$")
_SKU_LIKE = re.compile(r"[A-Za-z0-9_\-]+")
_DECLINE_SPLIT = re.compile(r"[\s.,;:!?()\[\]{}<>\"'`]+")

STOP = "stop"
NEXT = "next"


# ---- text and JSON helpers ----

def clip_strings(value: Any, limit: int, depth: int = 0) -> Any:
    if depth > 8:
        return None
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + "..."
    if isinstance(value, dict):
        return {key: clip_strings(item, limit, depth + 1) for key, item in value.items()}
    if isinstance(value, list):
        return [clip_strings(item, limit, depth + 1) for item in value]
    return value


def lean_copy(value: Any, limit: int, depth: int = 0) -> Any:
    if depth > 8:
        return None
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + "..."
    if isinstance(value, dict):
        return {key: lean_copy(item, limit, depth + 1) for key, item in value.items() if item not in (None, "", [], {})}
    if isinstance(value, list):
        return [lean_copy(item, limit, depth + 1) for item in value]
    return value


def to_json_text(value: Any) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        text = str(value)
    if len(text) > OBS_CHARS:
        text = text[:OBS_CHARS] + "...(cut)"
    return text


def read_json_object(text: str) -> dict[str, Any] | None:
    if not isinstance(text, str):
        return None
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[A-Za-z]*", "", text).strip()
        if text.endswith("```"):
            text = text[:-3].strip()
    try:
        found = json.loads(text)
        return found if isinstance(found, dict) else None
    except ValueError:
        pass
    left = text.find("{")
    right = text.rfind("}")
    if left >= 0 and right > left:
        try:
            found = json.loads(text[left : right + 1])
            return found if isinstance(found, dict) else None
        except ValueError:
            return None
    return None


def gather_sentences(value: Any, out: list[str], depth: int = 0) -> None:
    if depth > 4:
        return
    if isinstance(value, str):
        if len(value.split()) >= 4:
            out.append(value)
    elif isinstance(value, list):
        for item in value[:10]:
            gather_sentences(item, out, depth + 1)


def id_field(key: Any, value: Any) -> bool:
    # a listing id sits under a key that has "id" as a word of its own (id, item_id, productId), and it holds a digit;
    # "width" or "kid flat type" with a word value are product facts
    parts = {part.lower() for part in re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+", str(key))}
    return isinstance(value, str) and "id" in parts and bool(_ID_HINT.match(value)) and any(char.isdigit() for char in value)


def gather_ids(value: Any, out: set[str], depth: int = 0) -> None:
    if depth > 6:
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if id_field(key, item):
                out.add(item)
            else:
                gather_ids(item, out, depth + 1)
    elif isinstance(value, list):
        for item in value:
            gather_ids(item, out, depth + 1)


def all_strings(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(isinstance(entry, str) for entry in value)


def has_string_type(spec: dict[str, Any]) -> bool:
    return spec.get("type") == "string" or "string" in (spec.get("type") or [])


# ---- schema checks ----

def type_matches(value: Any, kind: str) -> bool:
    if kind == "string":
        return isinstance(value, str)
    if kind == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if kind == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if kind == "boolean":
        return isinstance(value, bool)
    if kind == "object":
        return isinstance(value, dict)
    if kind == "array":
        return isinstance(value, list)
    if kind == "null":
        return value is None
    return True


def type_list(schema: dict[str, Any]) -> Any:
    kinds = schema.get("type")
    return [kinds] if isinstance(kinds, str) else kinds


def coerce_to_schema(value: Any, schema: dict[str, Any]) -> Any:
    kinds = type_list(schema)
    if not kinds:
        return value
    if any(type_matches(value, kind) for kind in kinds):
        whole_float = isinstance(value, float) and value.is_integer()
        return int(value) if "integer" in kinds and whole_float else value
    for kind in kinds:
        if kind == "integer":
            if isinstance(value, float) and value.is_integer():
                return int(value)
            if isinstance(value, str) and re.fullmatch(r"\s*-?\d+(\.0+)?\s*", value):
                return int(float(value))
        if kind == "number" and isinstance(value, str) and re.fullmatch(r"\s*-?\d+(\.\d+)?\s*", value):
            number = float(value)
            return int(number) if number.is_integer() else number
        if kind == "string" and isinstance(value, (int, float)) and not isinstance(value, bool):
            return str(value)
        if kind == "boolean" and isinstance(value, str) and value.lower() in ("true", "false"):
            return value.lower() == "true"
    return value


def validate(value: Any, schema: dict[str, Any], path: str) -> tuple[Any, list[str]]:
    if not isinstance(schema, dict):
        return value, []
    problems: list[str] = []
    value = coerce_to_schema(value, schema)
    kinds = type_list(schema)
    if kinds and not any(type_matches(value, kind) for kind in kinds):
        return value, [f"{path}: expected {'/'.join(kinds)}"]
    if "enum" in schema and value not in schema["enum"]:
        if isinstance(value, str):
            for option in schema["enum"]:
                if isinstance(option, str) and option.lower() == value.strip().lower():
                    value = option
                    break
        if value not in schema["enum"]:
            problems.append(f"{path}: must be one of {schema['enum']}")
    if "const" in schema and value != schema["const"]:
        problems.append(f"{path}: must equal {schema['const']!r}")
    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            problems.append(f"{path}: shorter than {schema['minLength']} characters")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            value = value[: schema["maxLength"]]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        low, high = schema.get("exclusiveMinimum"), schema.get("exclusiveMaximum")
        if "minimum" in schema and value < schema["minimum"]:
            problems.append(f"{path}: below minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            problems.append(f"{path}: above maximum {schema['maximum']}")
        if "exclusiveMinimum" in schema and isinstance(low, (int, float)) and value <= low:
            problems.append(f"{path}: must be above {low}")
        if "exclusiveMaximum" in schema and isinstance(high, (int, float)) and value >= high:
            problems.append(f"{path}: must be below {high}")
    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            problems.append(f"{path}: needs at least {schema['minItems']} items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            problems.append(f"{path}: allows at most {schema['maxItems']} items")
        if isinstance(schema.get("items"), dict):
            checked = []
            for position, entry in enumerate(value):
                entry, sub = validate(entry, schema["items"], f"{path}[{position}]")
                checked.append(entry)
                problems.extend(sub)
            value = checked
    if isinstance(value, dict):
        properties = schema.get("properties") or {}
        for name in schema.get("required") or []:
            if name not in value:
                problems.append(f"{path}.{name}: required")
        if schema.get("additionalProperties") is False:
            for name in list(value):
                if name not in properties:
                    problems.append(f"{path}.{name}: not an allowed argument")
        rebuilt = {}
        for name, entry in value.items():
            if name in properties:
                entry, sub = validate(entry, properties[name], f"{path}.{name}")
                problems.extend(sub)
            rebuilt[name] = entry
        value = rebuilt
    return value, problems


def enum_signature(properties: dict[str, Any], args: dict[str, Any]) -> str:
    return to_json_text({key: value for key, value in args.items() if isinstance(properties.get(key), dict) and "enum" in properties[key]})


def message_arguments(schema: dict[str, Any], text: str) -> dict[str, Any] | None:
    properties = schema.get("properties") or {}
    required = list(schema.get("required") or [])
    free_text = [key for key, spec in properties.items() if isinstance(spec, dict) and has_string_type(spec) and "enum" not in spec]
    target = next((key for key in required if key in free_text), None)
    if target is None and len(free_text) == 1:
        target = free_text[0]
    if not target or any(key != target for key in required):
        return None
    return {target: text}


def filler_arguments(schema: dict[str, Any], text: str) -> dict[str, Any]:
    properties = schema.get("properties") or {}
    filled: dict[str, Any] = {}
    for key in schema.get("required") or []:
        spec = properties.get(key) or {}
        if "enum" in spec and spec["enum"]:
            filled[key] = spec["enum"][0]
        elif has_string_type(spec):
            filled[key] = text[: spec.get("maxLength", len(text))]
        elif spec.get("type") == "boolean":
            filled[key] = True
        elif spec.get("type") in ("integer", "number"):
            filled[key] = max(1, spec.get("minimum", 1))
    if len(properties) == 1:
        for key, spec in properties.items():
            if key not in filled and isinstance(spec, dict) and spec.get("type") == "string" and "enum" not in spec:
                filled[key] = text[: spec.get("maxLength", len(text))]
    return filled


def add_cache_marks(messages: list[dict[str, Any]], model: str) -> list[dict[str, Any]]:
    if not model.startswith("anthropic/") or len(messages) <= 2:
        return messages
    copied = [dict(message) for message in messages]
    for position in (0, len(copied) - 1):
        content = copied[position].get("content")
        if isinstance(content, str) and content:
            copied[position]["content"] = [{"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}]
    return copied


def tool_call(call_id: str, name: str, args: dict[str, Any]) -> dict[str, Any]:
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


# ---- the state record ----

@dataclass
class Session:
    started: float
    binding: dict[str, Any]
    policy: dict[str, Any]
    tools: list[dict[str, Any]]
    schemas: dict[str, Any]
    max_steps: int
    finished: bool
    step: int = 0
    dead: bool = False
    messages_used: int = 0
    message_budget: int | None = None
    order_attempts: int = 0
    refusals: int = 0
    order_checked: bool = False
    decline_checked: bool = False
    order_due: bool = False
    order_tried: bool = False
    cart_used: bool = False
    audited: bool = False
    planner_on: bool = True
    budget_now: Any = None
    current: list[str] | None = None
    last_cart_view: Any = None
    add_signature: str | None = None
    pending_reply: str = ""
    ledger_version: int = 0
    cost: float = 0.0
    idle: int = 0
    failures: int = 0
    call_counter: int = 0
    seen_ids: set[str] = field(default_factory=set)
    heard: set[str] = field(default_factory=set)
    qty_warned: set[str] = field(default_factory=set)
    cart_checked: set[str] = field(default_factory=set)
    stated: list[str] = field(default_factory=list)
    revealed: list[str] = field(default_factory=list)
    free: list[str] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)
    cart_ids: list[str] = field(default_factory=list)
    records: list[Any] = field(default_factory=list)
    asked: list[str] = field(default_factory=list)
    said: list[str] = field(default_factory=list)
    recent: list[str] = field(default_factory=list)
    verdicts: dict[str, str] = field(default_factory=dict)
    unproven: dict[str, list[str]] = field(default_factory=dict)
    details: list[Any] = field(default_factory=list)
    row_keys: set[str] = field(default_factory=set)
    unclear: list[str] = field(default_factory=list)
    unclear_questions: list[str] = field(default_factory=list)
    asked_keys: list[str] = field(default_factory=list)
    asked_products: list[str] = field(default_factory=list)
    widened: dict[str, int] = field(default_factory=dict)
    settled_products: list[str] = field(default_factory=list)
    passed: list[str] = field(default_factory=list)
    rows: dict[str, dict[str, Any]] = field(default_factory=dict)
    totals: list[tuple[str, str, Any]] = field(default_factory=list)
    totals_noted: bool = False
    paging_noted: bool = False
    page_values: set[str] = field(default_factory=set)
    search_calls: list[dict[str, Any]] = field(default_factory=list)
    cheap_held: bool = False
    noted: set[str] = field(default_factory=set)
    roles: dict[str, str] = field(default_factory=dict)
    plan: dict[str, Any] = field(default_factory=dict)
    trace: list[dict[str, Any]] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)


def new_session(started: float, problem_data: dict[str, Any]) -> Session:
    environment = problem_data["environment"]
    binding = dict(environment["binding"])
    policy = environment["policy_view"]
    tools = [tool for tool in policy.get("tools") or [] if isinstance(tool, dict) and isinstance(tool.get("function"), dict)]
    schemas = {}
    for tool in tools:
        function = tool["function"]
        schemas[function["name"]] = function.get("parameters") or {"type": "object", "properties": {}}
    return Session(
        started=started,
        binding=binding,
        policy=policy,
        tools=tools,
        schemas=schemas,
        max_steps=int(policy.get("max_steps") or 30),
        finished=bool(policy.get("finished")),
    )


def steps_left(state: Session) -> int:
    return state.max_steps - state.step


def request_text(state: Session) -> str:
    return str(state.policy.get("query") or "")


def message_limit(state: Session) -> int:
    return 3 if state.message_budget is None else state.message_budget


def finishing_tools(state: Session) -> set[str]:
    return {state.roles.get("order"), state.roles.get("decline")} - {None}


def as_count(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


# ---- request reading ----

def setup_content(state: Session) -> list[dict[str, Any]]:
    listing = []
    for tool in state.tools:
        function = tool["function"]
        listing.append(f"- {function['name']}: {function.get('description', '')} | arguments: {to_json_text(function.get('parameters') or {})}")
    fixed = SETUP_PROMPT + "\n\nTOOLS:\n" + "\n".join(listing)
    varying = "REQUEST:\n" + request_text(state)
    if state.policy.get("opening") is not None:
        varying += "\n\nOPENING OBSERVATION:\n" + to_json_text(state.policy.get("opening"))
    return [{"type": "text", "text": fixed, "cache_control": {"type": "ephemeral"}}, {"type": "text", "text": varying}]


def found_in_request(state: Session, quote: Any, share: float = 1.0) -> bool:
    # a phrase the reading quotes has to occur in the request itself (case and spacing aside); with share < 1 it is
    # enough that most of its words do
    words = re.findall(r"[a-z0-9]+", str(quote or "").lower())
    if not words:
        return False
    text = request_text(state).lower()
    if share >= 1.0:
        return " ".join(words) in " ".join(re.findall(r"[a-z0-9]+", text))
    present = set(re.findall(r"[a-z0-9]+", text))
    return sum(word in present for word in words) >= share * len(words)


def spec_products(state: Session, plan: dict[str, Any]) -> list[dict[str, Any]]:
    # the reading's product list, each quantity above 1 kept only when its quoted words are in the request, and each
    # requirement only when most of its quoted words are
    products = []
    for entry in plan.get("products") or []:
        if not isinstance(entry, dict) or not str(entry.get("product") or "").strip():
            continue
        quantity = entry.get("quantity")
        if not (isinstance(quantity, int) and not isinstance(quantity, bool) and quantity >= 1):
            quantity = 1
        if quantity > 1 and not found_in_request(state, entry.get("quantity_quote")):
            quantity = 1
        needs = []
        for need in entry.get("requirements") or []:
            if not isinstance(need, dict) or not str(need.get("property") or "").strip():
                continue
            if need.get("quote") and not found_in_request(state, need.get("quote"), 0.6):
                continue
            needs.append(need)
        products.append({"product": str(entry["product"]).strip(), "quantity": quantity, "requirements": needs})
    return products


def need_line(need: dict[str, Any]) -> str:
    rule = str(need.get("rule") or "").strip()
    values = [str(value) for value in need.get("values") or [] if str(value).strip()]
    missing = [value for value in values if value.lower() not in rule.lower()]
    return f"{need.get('property')}: {rule}" + (f" ({', '.join(missing)})" if missing else "")


def absorb_setup(state: Session, plan: dict[str, Any]) -> None:
    products = spec_products(state, plan)
    if products:
        plan["products"] = products
        plan["product"] = " + ".join(entry["product"] for entry in products)
        plan["quantity"] = products[0]["quantity"] if len(products) == 1 else None
        many = len(products) > 1
        plan["requirements"] = [(f"{entry['product']} - " if many else "") + need_line(need) for entry in products for need in entry["requirements"]]
    state.plan = plan
    roles = plan.get("roles")
    if isinstance(roles, dict):
        state.roles = {role: tool for role, tool in roles.items() if isinstance(tool, str) and tool in state.schemas}
    else:
        state.roles = {}
    budget = plan.get("message_budget")
    if as_count(budget) and budget >= 0:
        state.message_budget = int(budget)
    state.stated = [str(entry) for entry in plan.get("requirements") or [] if entry]
    state.budget_now = plan.get("price_cap")
    if plan and state.budget_now is None and state.roles.get("message"):
        earlier = [str(entry) for entry in plan.get("ask_first") or [] if entry]
        plan["ask_first"] = ["the highest amount the shopper is willing to pay, because the request sets no price limit (ignore this if another entry here says the same)"] + earlier
    allowed_steps = plan.get("step_budget")
    if as_count(allowed_steps) and 0 < allowed_steps < state.max_steps:
        state.max_steps = int(allowed_steps)


def reading_brief(state: Session) -> str:
    plan = state.plan
    if not plan:
        return ""
    out = ["Draft summary of the request (verify it against the request text itself):"]
    products = plan.get("products") or []
    if len(products) > 1:
        out.append(f"- this order holds {len(products)} different products, one cart line each, each meeting its own requirements:")
        out.extend(f"  - {entry['product']}: {entry['quantity']} unit(s)" for entry in products)
    else:
        if plan.get("product"):
            out.append(f"- product wanted: {plan.get('product')}")
        out.append(f"- units to order: {plan.get('quantity') or 1}")
    out.extend(f"- requirement from the request: {entry}" for entry in plan.get("requirements") or [])
    if plan.get("price_cap") is not None:
        out.append(f"- total budget: {plan.get('price_cap')} {plan.get('currency') or ''}".rstrip())
    if plan.get("in_stock_only"):
        out.append("- in-stock (on hand) items only")
    priority = plan.get("priority")
    if priority not in ("cheapest", "best"):
        out.append("- the request names no cheapest or best rule, so any item that satisfies every requirement inside the budget is correct; do not burn steps chasing the lowest price (unless the request itself asks for that)")
    else:
        detail = plan.get("priority_detail")
        out.append(f"- pick the {priority}{f' ({detail})' if detail else ''} out of all fitting items")
    out.extend(f"- optional wish, NOT a requirement (tie-breaker only): {want}" for want in plan.get("soft_wants") or [])
    out.extend(f"- learn this from the shopper before choosing: {entry}" for entry in plan.get("ask_first") or [])
    out.extend(f"- procedural rule: {rule}" for rule in plan.get("rules") or [])
    if state.message_budget is not None:
        out.append(f"- messages you may send the shopper: {state.message_budget}")
    out.append(f"- step limit: {state.max_steps}")
    hints = [hint for hint in plan.get("hint_properties") or [] if isinstance(hint, str)]
    if hints:
        out.append("- properties the story seems to point at (ask these first, one per message): " + ", ".join(hints))
    for ref in plan.get("references") or []:
        if not isinstance(ref, dict) or not ref.get("property"):
            continue
        known = ", ".join(f"{key}: {value}" for key, value in (ref.get("known") or {}).items()) if isinstance(ref.get("known"), dict) else ""
        out.append(f"- a value given by reference: first find the listing in the category '{ref.get('category')}' sold by '{ref.get('seller')}'"
                   + (f" that has {known}" if known else "") + f" (search that category, keep rows from that seller, open them and match those values); read its {ref.get('property')}, and from then on treat it as a concrete requirement"
                   + (f" ({ref.get('tolerance')})" if ref.get("tolerance") else "") + f" for the {ref.get('product') or 'item'}; that listing may itself be bought only when it sits in the wanted product's own category and meets every other requirement, otherwise it just supplies the value.")
    return "\n".join(out)


def units_text(plan: dict[str, Any]) -> str:
    products = plan.get("products") or []
    if len(products) > 1:
        return ", ".join(f"{entry['quantity']} x {entry['product']}" for entry in products)
    return str(plan.get("quantity") or 1)


def requirements_line(state: Session) -> str:
    if state.current is None:
        entries = [f"{entry} (stated)" for entry in state.stated if entry not in state.dropped]
        entries += [f"{entry} (revealed by the shopper)" for entry in state.revealed if entry not in state.dropped]
    else:
        entries = list(state.current)
    text = "; ".join(entries) if entries else "as written in the request"
    if state.budget_now is not None:
        text += f"; whole order costs no more than {state.budget_now} {state.plan.get('currency') or ''}".rstrip()
    text += "; units " + units_text(state.plan)
    if state.dropped:
        text += "; dropped by the shopper: " + "; ".join(state.dropped)
    if state.free:
        text += "; shopper is indifferent to: " + ", ".join(state.free)
    return text


def status_line(state: Session, seconds: float, rushed: bool) -> str:
    left = steps_left(state)
    sent = f"shopper messages {state.messages_used}"
    if state.message_budget is not None:
        sent += f" out of {state.message_budget}"
    head = f"[progress] {state.step} of {state.max_steps} steps spent, {left} remaining; {sent}; roughly {int(max(0, seconds - FORCE_SECONDS))} s to go."
    text = head + "\n[current needs] " + requirements_line(state)
    if left <= 2:
        return text + " Final steps: if the cart holds a verified fit, order it now; if not, decline."
    if rushed:
        return text + " Wrap up now: trim the cart to a single verified line, check its stock again, then order (or decline when nothing fits)."
    if state.cost > COST_STOP:
        return text + " The task's spending limit is near: put your best verified fit in the cart now and order it (or decline if nothing fits)."
    if state.order_attempts == 0 and left <= 14:
        return text + " Steps are getting scarce: put your best verified fit in the cart and order within the next few moves; search and detail tools go away at 7 steps left, and repairing a refused order takes roughly 4 steps."
    return text


def offered_tools(state: Session) -> list[dict[str, Any]]:
    left = steps_left(state)
    roles = state.roles
    order, decline, cart = roles.get("order"), roles.get("decline"), roles.get("cart")
    if left <= 1:
        keep = {order, decline}
    elif left <= 3:
        keep = {order, decline, cart}
    elif (left <= 7 and (state.order_attempts == 0 or state.refusals > 0)) or state.cost > COST_STOP:
        keep = {order, decline, cart, roles.get("cart_view"), roles.get("view")}
    else:
        keep = None
    offered = state.tools
    if keep is not None and order and cart:
        offered = [tool for tool in state.tools if tool["function"]["name"] in keep] or offered
    lost_last = bool(state.asked_keys) and state.asked_keys[-1] in state.unclear
    if (state.planner_on or lost_last) and roles.get("message"):
        # questions are the interview's job; after a reply the shopper did not understand, the loop does not reword
        # the same question on its own (a direct question from the shopper is answered through the relay)
        offered = [tool for tool in offered if tool["function"]["name"] != roles.get("message")]
    return offered


# ---- record digests ----

def records_mentioning(records: list[Any], item: str) -> list[str]:
    hits: list[str] = []

    def visit(node: Any, depth: int) -> None:
        if depth > 6 or len(hits) >= 8:
            return
        if isinstance(node, list):
            for child in node:
                visit(child, depth + 1)
            return
        if not isinstance(node, dict):
            return
        if item in [child for child in node.values() if isinstance(child, str)]:
            text = to_json_text(clip_strings(node, 300))[:1600]
            if text not in hits:
                hits.append(text)
            return
        for child in node.values():
            visit(child, depth + 1)

    for record in records:
        visit(record, 0)
    return hits[-3:]


def fact_maps(observation: Any) -> list[dict[str, str]]:
    # the product facts of one details result: the nested map of short values when the store sends one, else the
    # short top-level values; one map per listing when the result holds several
    out: list[dict[str, str]] = []

    def short_values(node: dict[str, Any]) -> dict[str, str]:
        return {str(key): str(value) for key, value in node.items()
                if isinstance(value, (str, int, float)) and not isinstance(value, bool) and 0 < len(str(value)) <= 70}

    def visit(node: Any, depth: int) -> None:
        if depth > 4:
            return
        if isinstance(node, list):
            for child in node:
                visit(child, depth + 1)
            return
        if not isinstance(node, dict):
            return
        nested = [short_values(child) for child in node.values() if isinstance(child, dict)]
        nested = [entry for entry in nested if len(entry) >= 2]
        if nested and any(isinstance(child, str) for child in node.values()):
            out.append(max(nested, key=len))
            return
        if sum(isinstance(child, (str, int, float)) for child in node.values()) >= 4 and not nested:
            out.append({"__top__": "1", **short_values(node)})
            return
        for child in node.values():
            if isinstance(child, (dict, list)):
                visit(child, depth + 1)

    visit(observation, 0)
    return out


def stems(text: Any) -> set[str]:
    return {word[:-1] if len(word) > 3 and word.endswith("s") else word for word in words_of(text)}


def observation_product(state: Session, observation: Any) -> str | None:
    # the product of a several-product order that an opened listing belongs to: the one whose name shares the most
    # words with the listing's own text (its category and title); None when no name or two names fit equally
    names = [str(entry.get("product") or "") for entry in state.plan.get("products") or []]
    if len(names) < 2:
        return None
    seen: set[str] = set()

    def visit(node: Any, depth: int) -> None:
        if depth > 3:
            return
        children = node.values() if isinstance(node, dict) else node if isinstance(node, list) else []
        for child in children:
            if isinstance(child, str) and len(child) <= 200:
                seen.update(stems(child))
            elif isinstance(child, (dict, list)):
                visit(child, depth + 1)

    visit(observation, 0)
    scores = sorted(((len(stems(name) & seen) / max(1, len(stems(name))), name) for name in names), reverse=True)
    if not scores[0][0] or (len(scores) > 1 and scores[1][0] == scores[0][0]):
        return None
    return scores[0][1]


def question_menu(state: Session, product: str | None = None) -> list[tuple[str, int, list[str]]]:
    # properties of the opened listings worth asking about: carried by many of them, with values that differ,
    # neither nearly unique per listing nor a twin of a key already kept, and not settled by a stated need; with a
    # product named, only its own listings and its own stated needs count
    holders: dict[str, int] = {}
    values: dict[str, set[str]] = {}
    total = 0
    for observation in state.details:
        if product and observation_product(state, observation) != product:
            continue
        for facts in fact_maps(observation):
            top = facts.pop("__top__", None)
            total += 1
            for key, value in facts.items():
                if top and key in state.row_keys:
                    continue
                if re.search(r"\d", value) and _SKU_LIKE.fullmatch(value) and ("-" in value or "_" in value):
                    continue
                label = re.sub(r"[_.]+", " ", key).strip().lower()
                # a store prefix of one or two letters is dropped, and a run-together key of many words is not asked
                # in that form
                label = re.sub(r"^[a-z]{1,2} (?=[a-z]{3})", "", label)
                if any(len(word) >= 14 for word in label.split()):
                    continue
                holders[label] = holders.get(label, 0) + 1
                values.setdefault(label, set()).add(value.strip().lower()[:40])
    settled: list[set[str]] = []
    for entry in state.plan.get("products") or []:
        if product and str(entry.get("product") or "") != product:
            continue
        for need in entry.get("requirements") or []:
            names = {word for word in words_of(need.get("property")) if len(word) >= 3}
            if names:
                settled.append(names)
    total = max(1, total)
    scored = []
    for label in holders:
        mine = {word for word in words_of(label) if len(word) >= 3}
        # a key is settled only when its name and a stated property's name hold the same words, one inside the other
        # ("pack type" settles "type of pack", not "packaging type")
        if mine and any(mine <= names or names <= mine for names in settled):
            continue
        spread = len(values[label])
        weight = (holders[label] / total) * min(spread, 4)
        if holders[label] >= 4 and spread >= 0.9 * holders[label]:
            weight *= 0.4
        scored.append((weight, label))
    scored.sort(key=lambda pair: -pair[0])
    kept: list[tuple[str, int, list[str]]] = []
    for weight, label in scored:
        if any(values[label] == values[other] and words_of(label) & words_of(other) for other, _, _ in kept):
            continue
        kept.append((label, holders[label], sorted(values[label])[:5]))
    return kept[:16]


def recurring_labels(records: list[Any]) -> str:
    tally: dict[str, dict[str, int]] = {}

    def visit(node: Any, depth: int) -> None:
        if depth > 6:
            return
        if isinstance(node, list):
            for child in node:
                visit(child, depth + 1)
            return
        if not isinstance(node, dict):
            return
        if len(node) >= 4:
            for key, child in node.items():
                if not isinstance(child, str) or not 0 < len(child) <= 40:
                    continue
                if re.search(r"\d", child) and _SKU_LIKE.fullmatch(child):
                    continue
                counts = tally.setdefault(str(key), {})
                counts[child] = counts.get(child, 0) + 1
        for child in node.values():
            if isinstance(child, (dict, list)):
                visit(child, depth + 1)

    for record in records:
        visit(record, 0)
    out = []
    for key, counts in tally.items():
        if sum(counts.values()) < 4 or not 2 <= len(counts) <= 8 or max(counts.values()) < 2:
            continue
        ranked = sorted(counts.items(), key=lambda pair: -pair[1])
        out.append(f"- {key}: " + " | ".join(f"{value} (x{times})" for value, times in ranked))
    return "\n".join(out[:12])


# ---- judge prompts and their answers ----

def interpret_prompt(state: Session, fresh: list[str], question: str | None) -> str:
    parts = [INTERPRET_PROMPT, "THE SHOPPER'S REQUEST (for context):\n" + request_text(state), "REQUIREMENTS SO FAR:\n" + requirements_line(state)]
    if state.said:
        parts.append("EARLIER EXCHANGES WITH THE SHOPPER (oldest first):\n" + "\n".join(state.said[-12:]))
    if question:
        parts.append("THE ASSISTANT ASKED:\n" + question)
    parts.append("THE SHOPPER NOW SAID:\n" + "\n".join(fresh))
    return "\n\n".join(parts)


def need_key(entry: Any) -> str:
    return str(entry).split(":", 1)[0].strip().lower()


def absorb_interpretation(state: Session, result: dict[str, Any], answering: bool = False) -> str:
    changes = []
    # a reply that only says the shopper does not mind about the asked property changes nothing they revealed to an
    # earlier question: a need given there leaves only through an explicit change
    neutral = answering and bool(result.get("no_preference")) and not result.get("needs") and not result.get("new_budget")
    held = []
    for need in state.revealed[::-1] if neutral else []:
        if need not in state.dropped and all(need_key(need) != need_key(other) for other in held):
            held.insert(0, need)
    if held:
        # nor is a revealed property listed as one the shopper does not mind
        keys = {need_key(need) for need in held}
        result["dropped"] = [gone for gone in result.get("dropped") or [] if need_key(gone) not in keys and gone not in held]
        result["no_preference"] = [prop for prop in result.get("no_preference") or [] if need_key(prop) not in keys]
    latest = result.get("requirements_now")
    if isinstance(latest, list) and all(isinstance(entry, str) for entry in latest):
        state.current = [entry for entry in latest if entry.strip()]
        if held:
            state.current = [entry for entry in state.current if need_key(entry) not in keys] + held
    for need in result.get("needs") or []:
        if isinstance(need, str) and need and need not in state.revealed:
            state.revealed.append(need)
            changes.append(f"added need: {need}")
    for prop in result.get("no_preference") or []:
        if isinstance(prop, str) and prop and prop not in state.free:
            state.free.append(prop)
    for gone in result.get("dropped") or []:
        if isinstance(gone, str) and gone and gone not in state.dropped:
            state.dropped.append(gone)
            changes.append(f"no longer needed: {gone}")
    budget = result.get("new_budget")
    if as_count(budget) and budget > 0 and budget != state.budget_now:
        state.budget_now = budget
        changes.append(f"budget now {budget}")
    if result.get("unavailable"):
        changes.append("something is no longer for sale; take it out of the cart and do not buy it")
    if changes:
        state.ledger_version += 1
    answer = str(result.get("reply") or "").strip()
    if answer and state.roles.get("message") and result.get("asked_us") is True and result.get("understood") is not False:
        state.pending_reply = answer
    summary = "Shopper meant: " + str(result.get("meaning") or "")
    return summary + (" (" + "; ".join(changes) + ")" if changes else "")


def verify_prompt(state: Session, item: str, purpose: str, records: list[str]) -> str:
    parts = [
        VERIFY_PROMPT,
        "THE SHOPPER'S REQUEST:\n" + request_text(state),
        "CURRENT REQUIREMENTS (stated, revealed and changed):\n" + requirements_line(state)
        + ("\nThis order holds several products: judge the item only against the requirements of the product it is (the ones labelled with that product), and the budget against the whole order." if len(state.plan.get("products") or []) > 1 else ""),
        f"THE ITEM TO {purpose}: {item}\nWHAT THE STORE HAS SHOWN ABOUT IT, OLDEST FIRST:\n" + "\n".join(records),
    ]
    if state.last_cart_view is not None:
        parts.append("LATEST CART VIEW:\n" + to_json_text(state.last_cart_view)[:3000])
    labels = recurring_labels(state.records)
    if labels:
        parts.append("LABEL VALUES REPEATED ACROSS THIS STORE'S LISTINGS (with counts; the item's category has to be the label matching the request's wording as a whole, never a neighbour):\n" + labels)
    return "\n\n".join(parts)


def failed_check_text(item: str, problems: list[str]) -> str:
    return f"A separate review finds that {item} misses at least one requirement: " + "; ".join(problems[:5]) + ". Pick another fully verified fit unless you are sure the review is mistaken."


def words_of(text: Any) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", str(text or "").lower()))


def item_facts(records: list[Any], item: str) -> dict[str, str]:
    # every short value the store has shown for this item, top level and one map down, under its key
    found: dict[str, str] = {}

    def take(node: dict[str, Any]) -> None:
        for key, value in node.items():
            if isinstance(value, (str, int, float)) and not isinstance(value, bool) and len(str(value)) <= 200:
                found[str(key)] = str(value)
            elif isinstance(value, dict):
                for inner, deeper in value.items():
                    if isinstance(deeper, (str, int, float)) and not isinstance(deeper, bool) and len(str(deeper)) <= 200:
                        found[str(inner)] = str(deeper)

    def visit(node: Any, depth: int) -> None:
        if depth > 6:
            return
        if isinstance(node, list):
            for child in node:
                visit(child, depth + 1)
        elif isinstance(node, dict):
            if item in [child for child in node.values() if isinstance(child, str)]:
                take(node)
                return
            for child in node.values():
                visit(child, depth + 1)

    for record in records:
        visit(record, 0)
    return found


def property_names(state: Session, requirement: str) -> set[str]:
    # the words of the property a review line is about: the longest known property name inside it, else its head
    said = words_of(requirement)
    known = [str(need.get("property") or "") for entry in state.plan.get("products") or [] for need in entry.get("requirements") or []]
    known += [line.split(":", 1)[0] for line in (state.current or []) + state.revealed if ":" in line]
    matches = [{word for word in words_of(prop) if len(word) >= 3} for prop in known]
    matches = [names for names in matches if names and names <= said]
    if matches:
        return max(matches, key=len)
    return {word for word in words_of(requirement.split(":", 1)[0]) if len(word) >= 3} if ":" in requirement else set()


def carried_elsewhere(state: Session, item: str, requirement: str) -> bool:
    # does an opened listing other than this item hold a key naming the requirement's property (the words before
    # the colon, or the whole text when there is none)?
    said = words_of(requirement)
    loose = [need for entry in state.plan.get("products") or [] for need in entry.get("requirements") or [] if need.get("op") == "other"]
    if any({word for word in words_of(need.get("property")) if len(word) >= 3} <= said for need in loose if words_of(need.get("property"))):
        # a requirement code cannot pin down (one given by reference, say) is left to the review
        return False
    names = property_names(state, requirement)
    if not names:
        return False
    for observation in state.details:
        mine = item in [value for value in (observation.values() if isinstance(observation, dict) else []) if isinstance(value, str)]
        if mine:
            continue
        for facts in fact_maps(observation):
            if any(names <= words_of(key) or (len(names) == 1 and names & words_of(key)) for key in facts if key != "__top__"):
                return True
    return False


def literal_pass(facts: dict[str, str], need: dict[str, Any]) -> bool:
    # a stated value counts when all its words stand as whole words in the item's value under a key that names
    # the property (one shared word of three letters or more), the way the store writes its values
    if need.get("op") != "is":
        return False
    names = {word for word in words_of(need.get("property")) if len(word) >= 3}
    related = [key for key in facts if names & words_of(key)]
    # a key that names the whole property outranks keys sharing only a word with it: when one exists, only it counts
    exact = [key for key in related if names and names <= words_of(key)]
    same = [key for key in exact if words_of(key) == names]
    for key in same or exact or related:
        value = facts[key]
        held = words_of(value)
        if any(words_of(wanted) and words_of(wanted) <= held for wanted in need.get("values") or []):
            return True
    return False


def plan_prompt(state: Session, count: int, menu: str, target: str | None = None) -> str:
    parts = [
        PLAN_PROMPT.replace("QUESTION_COUNT", str(count)),
        "THE SHOPPER'S REQUEST:\n" + request_text(state),
        "REQUIREMENTS SO FAR:\n" + requirements_line(state),
    ]
    if state.asked:
        parts.append("ALREADY ASKED (with what the shopper said):\n" + "\n".join(state.asked))
    must_learn = state.plan.get("ask_first")
    if must_learn:
        parts.append("MUST BE LEARNED FROM THE SHOPPER: " + "; ".join(str(entry) for entry in must_learn or []))
    hints = [hint for hint in state.plan.get("hint_properties") or [] if isinstance(hint, str)]
    if hints:
        parts.append("STORY HINTS (properties a story detail points to): " + ", ".join(hints))
    if state.unclear:
        parts.append("NOT UNDERSTOOD BY THE SHOPPER (never ask these again): " + ", ".join(state.unclear))
    if target:
        parts.append("THIS QUESTION IS ABOUT: " + target + " (the menu below holds its listings only; the other products get their turn in later questions)")
    elif len(state.plan.get("products") or []) > 1:
        parts.append("PRODUCTS IN THIS ORDER: " + ", ".join(entry["product"] for entry in state.plan["products"]))
    parts.append("MENU: properties found in the opened listings' details (name, how many listings carry it, sample values):\n" + (menu or "(nothing seen yet)"))
    return "\n\n".join(parts)


def audit_prompt(state: Session) -> str | None:
    blocks: list[str] = []
    used = 0
    for record in state.records[::-1]:
        text = to_json_text(clip_strings(record, 160))
        if used + len(text) <= 12000:
            blocks.append(text)
            used += len(text)
    if not blocks:
        return None
    parts = [AUDIT_PROMPT, "THE SHOPPER'S REQUEST:\n" + request_text(state), "CURRENT REQUIREMENTS:\n" + requirements_line(state), "LISTING RECORDS SEEN (newest first):\n" + "\n".join(blocks)]
    return "\n\n".join(parts)


def audit_message(result: dict[str, Any] | None) -> str:
    candidates = [str(entry) for entry in (result or {}).get("possible_fits") or [] if isinstance(entry, (str, int))][:5]
    if not candidates:
        return ""
    reason = f" ({result.get('reason')})" if result.get("reason") else ""
    return "A separate review suspects these listings could satisfy every requirement: " + ", ".join(candidates) + reason + ". Check them (details, stock, price) and buy the correct one; decline only once each of them really fails."


def quantity_problem(state: Session, name: str, args: dict[str, Any]) -> str:
    wanted = state.plan.get("quantity")
    if isinstance(wanted, bool) or not isinstance(wanted, int) or wanted < 1:
        return ""
    properties = state.schemas[name].get("properties") or {}
    if state.add_signature is not None and enum_signature(properties, args) != state.add_signature:
        return ""
    for key, spec in properties.items():
        if not isinstance(spec, dict) or spec.get("type") != "integer" or "enum" in spec:
            continue
        asked_for = args.get(key, spec.get("default", 1))
        if isinstance(asked_for, int) and asked_for != wanted:
            return f"this call asks for {asked_for}, while the request needs {wanted} units of one listing."
    return ""


def scrub_decline(state: Session, args: dict[str, Any], name: str) -> dict[str, Any]:
    rule = state.plan.get("decline_no_names")
    if rule is False:
        return args
    names_listing = any(
        token and token in state.seen_ids
        for value in args.values() if isinstance(value, str)
        for token in _DECLINE_SPLIT.split(value)
    )
    if not rule and not names_listing:
        return args
    properties = (state.schemas.get(name) or {}).get("properties") or {}
    scrubbed = {}
    for key, value in args.items():
        spec = properties.get(key) or {}
        scrubbed[key] = DECLINE_TEXT[: spec.get("maxLength", len(DECLINE_TEXT))] if isinstance(value, str) and "enum" not in spec else value
    return scrubbed


def paging_notes(state: Session, observation: dict[str, Any], args: dict[str, Any]) -> list[str]:
    # a page token is only good when a result offered it: the top-level text values of results are the tokens on
    # offer; a result that holds more matches than rows but leaves its token field empty cannot be paged further
    rows = next((value for value in observation.values() if isinstance(value, list)), None)
    if rows is None:
        # a refused call that repeats an earlier good search with one value added: that value was the page guess
        offered = state.page_values
        guessed = [key for key, value in args.items() if isinstance(value, str) and value not in offered
                   and any(earlier == {other: item for other, item in args.items() if other != key} for earlier in state.search_calls)]
        if guessed and not offered:
            return [f"The {guessed[0]} value was not one the store offered: no search result so far has offered a next-page value, so pages cannot be opened that way. Reach other listings with other words, another mode or other filters."]
        return []
    state.search_calls.append(dict(args))
    state.page_values.update(value for value in observation.values() if isinstance(value, str) and 0 < len(value) <= 40)
    total = max((value for value in observation.values() if isinstance(value, int) and not isinstance(value, bool)), default=0)
    empty = [key for key, value in observation.items() if value is None]
    if empty and total > len(rows) and not state.paging_noted:
        state.paging_noted = True
        return [f"This result holds {total} matches but its {empty[0]} field is empty, so this list cannot be paged further. Do not guess page values; reach other listings with other or more specific words, another mode or other filters."]
    return []


def observation_notes(state: Session, name: str, observation: Any, args: dict[str, Any] | None = None) -> str:
    if isinstance(observation, list):
        gather_ids(observation, state.seen_ids)
        return ""
    if not isinstance(observation, dict):
        return ""
    gather_ids(observation, state.seen_ids)
    notes = []
    message_tool = state.roles.get("message")
    if observation.get("error") == "budget_exceeded":
        if state.message_budget is not None:
            state.message_budget = min(state.message_budget, state.messages_used)
        elif name == message_tool:
            state.message_budget = state.messages_used
        notes.append("This budget is exhausted; stop calling this tool.")
    if name == state.roles.get("search") and not state.totals_noted and len(state.totals) >= 2:
        filters, words, total = state.totals[-1]
        if total is not None and words and any(f == filters and w and w != words and n == total for f, w, n in state.totals[:-1]):
            # the same filters with other words gave the very same count: in this mode the words change nothing
            state.totals_noted = True
            notes.append(f"This search returned exactly the same total ({total}) as an earlier search with other words under the same filters, so in this mode the words are probably ignored; search by words in the mode that uses them, or page through this list (lowering a price filter if the request wants the cheapest).")
    if name == state.roles.get("search"):
        notes.extend(paging_notes(state, observation, args or {}))
    if "refusal" in observation:
        state.refusals += 1
        state.order_checked = False
        notes.append("ORDER REFUSED: the episode goes on and the cart is as it was. Apply each change the refusal names, repair the cart (drop any line that is no longer acceptable, a sold-out item for example), check the new pick's stock and price again, and place the order once more.")
    elif len(observation) >= 2 and name != message_tool:
        lines: list[str] = []
        for value in observation.values():
            if all_strings(value):
                gather_sentences(value, lines)
        if lines:
            notes.append("The shopper speaks inside this result (a change), so act on it: " + " | ".join(lines[:4]))
    return " ".join(notes)


def shopper_lines(name: str, observation: Any, args: dict[str, Any], message_tool: str | None) -> tuple[list[str], str | None]:
    heard: list[str] = []
    if name != message_tool:
        if isinstance(observation, dict):
            refused = "refusal" in observation
            for value in observation.values():
                if all_strings(value):
                    gather_sentences(value, heard)
                elif refused and isinstance(value, str) and len(value.split()) >= 4:
                    heard.append(value)
        return heard, None
    if isinstance(observation, str):
        heard = [observation]
    elif isinstance(observation, dict) and "error" not in observation:
        gather_sentences([value for value in observation.values() if isinstance(value, (str, list))], heard)
    return heard, " ".join(str(value) for value in args.values() if isinstance(value, str))


def line_ids(args: dict[str, Any]) -> list[str] | None:
    # a cart call that sends the whole cart as a list of lines (one object per line): the listing ids it holds, in
    # order; None when the call names items one at a time
    lines = [value for value in args.values() if isinstance(value, list) and value and all(isinstance(entry, dict) for entry in value)]
    if not lines and not any(isinstance(value, list) and not value for value in args.values()):
        return None
    found: list[str] = []
    for entry in (lines[0] if lines else []):
        found.extend(value for value in entry.values() if isinstance(value, str) and " " not in value.strip() and any(char.isdigit() for char in value) and value not in found)
    return found


def bare_values(args: dict[str, Any], properties: dict[str, Any]) -> list[str]:
    found = []
    for key, value in args.items():
        spec = properties.get(key) if isinstance(properties.get(key), dict) else {}
        if isinstance(value, str) and "enum" not in spec and value.strip() and " " not in value.strip():
            found.append(value)
    return found


def book_result(state: Session, name: str, args: dict[str, Any], observation: Any) -> None:
    roles = state.roles
    cart_tool, message_tool = roles.get("cart"), roles.get("message")
    if name == message_tool:
        state.messages_used += 1
    if name == roles.get("order"):
        state.order_attempts += 1
    if name == cart_tool:
        state.cart_used = True
        state.order_checked = False
    if name in (cart_tool, message_tool) or name in finishing_tools(state):
        state.recent = []
    else:
        state.recent = (state.recent + [name + to_json_text(args)])[-2:]
    state.records.append(observation)
    other_roles = {roles.get("search"), cart_tool, message_tool, roles.get("cart_view"), roles.get("order"), roles.get("decline")}
    if name in (roles.get("details"), roles.get("view")) and isinstance(observation, (dict, list)):
        state.details.append(observation)
    elif name not in other_roles and isinstance(observation, (dict, list)) and any(len(facts) >= 3 for facts in fact_maps(observation)):
        # a tool that shows several listings side by side opens them just as the details tool does
        state.details.append(observation)
    if name == roles.get("search") and isinstance(observation, dict):
        properties = state.schemas[name].get("properties") or {}
        wordy = {key for key, value in args.items() if isinstance(value, str) and "enum" not in (properties.get(key) or {})}
        text = " ".join(str(args[key]) for key in sorted(wordy))
        filters = to_json_text({key: value for key, value in sorted(args.items()) if key not in wordy})
        # the reported total is the largest whole number at the top of the result (a page offset is smaller)
        count = max((value for value in observation.values() if isinstance(value, int) and not isinstance(value, bool)), default=None)
        state.totals.append((filters, text.lower(), count))
        # the keys a search row carries at its top level (seller, brand, title, price ...) describe the listing,
        # not the product, so they never make a question topic
        for value in observation.values():
            if isinstance(value, list):
                for row in value:
                    if isinstance(row, dict):
                        state.row_keys.update(str(key) for key in row)
                        row_id = next((child for key, child in row.items() if id_field(key, child)), None)
                        if row_id:
                            state.rows[row_id] = row
    if name == cart_tool:
        properties = state.schemas[name].get("properties") or {}
        signature = enum_signature(properties, args)
        if state.add_signature is None:
            state.add_signature = signature
        adding = signature == state.add_signature
        listed = line_ids(args)
        if listed is not None:
            # the call sends the whole cart: it becomes exactly these listings
            state.cart_ids = listed
        for value in bare_values(args, properties) if listed is None else []:
            state.cart_ids = [entry for entry in state.cart_ids if entry != value] + ([value] if adding else [])
    if name == roles.get("cart_view"):
        state.last_cart_view = observation


def normalise_calls(state: Session, raw_calls: list[Any]) -> list[dict[str, Any]]:
    calls = []
    for raw in raw_calls:
        if not isinstance(raw, dict):
            continue
        state.call_counter += 1
        function = raw.get("function") or {}
        arguments = function.get("arguments")
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments or {})
        calls.append({"id": raw.get("id") or f"call_{state.call_counter}", "type": "function", "function": {"name": function.get("name"), "arguments": arguments}})
    return calls


# ---- the shell: the only place that reads the clock or talks to the LLM and the world ----

class Shell:
    def __init__(self, problem_data: dict[str, Any]):
        self.state = new_session(time.monotonic(), problem_data)

    def seconds_left(self) -> float:
        return TASK_SECONDS - (time.monotonic() - self.state.started)

    def elapsed(self) -> float:
        return time.monotonic() - self.state.started

    def status(self) -> str:
        left = steps_left(self.state)
        seconds = self.seconds_left()
        rushed = left > 2 and (left <= 5 or self.seconds_left() < WRAP_UP_SECONDS)
        return status_line(self.state, seconds, rushed)

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, models: list[str], max_tokens: int) -> dict[str, Any] | None:
        state = self.state
        for model in models:
            body: dict[str, Any] = {"model": model, "messages": add_cache_marks(messages, model), "temperature": 0, "max_tokens": max_tokens}
            if tools:
                body["tools"] = tools
                body["tool_choice"] = "auto"
            if model == JUDGE_MODEL or model == SETUP_MODEL or not tools:
                # single-answer calls think briefly: hidden reasoning was most of the long setup replies' cost, and on
                # the reader it used up the whole token allowance of decline audits, leaving them empty
                body["reasoning"] = {"effort": "low"}
            # a server-side hiccup (5xx, rate limit, empty reply) often clears within seconds: the same LLM is tried
            # again twice, with a short pause, before the next one in the list takes over
            for attempt in range(3):
                remaining = self.seconds_left()
                if remaining < FORCE_SECONDS - 10:
                    return None
                _llm.timeout = int(max(8, min(75, remaining - FORCE_SECONDS + 15)))
                result = _llm.post_verbose("/inference/chat/completions", body)
                data = result.data
                if isinstance(data, dict) and data.get("choices"):
                    spent = (data.get("usage") or {}).get("cost")
                    if isinstance(spent, (int, float)):
                        state.cost += float(spent)
                    reply = data["choices"][0].get("message") or {}
                    if reply.get("content") or reply.get("tool_calls"):
                        return reply
                state.trace.append({"kind": "inference_failure", "model": model, "error": str(result.error)[:300] if result.error else "empty"})
                status = (result.error or {}).get("status") if isinstance(result.error, dict) else None
                if status == 400 and body.pop("reasoning", None) is not None:
                    # a provider that refuses the reasoning setting gets the same request without it
                    continue
                passing = result.error is not None and (status is None or status == 429 or (isinstance(status, int) and status >= 500))
                if not passing or attempt == 2 or self.seconds_left() < FORCE_SECONDS + 25:
                    break
                time.sleep(1.5 * (attempt + 1))
        return None

    def judge(self, prompt: str, max_tokens: int = 700, models: list[str] | None = None) -> dict[str, Any] | None:
        reviewing = models is None
        models = models or [JUDGE_MODEL, MAIN_MODEL]
        if reviewing and self.state.cost > COST_HARD:
            # near the cap the optional reviews stop; the loop and the reading of shopper lines go on
            return None
        if self.state.cost > COST_SOFT:
            # past the soft spend line the remaining reviews run on the cheaper reader, so no task nears the cap
            models = [model for model in models if model != JUDGE_MODEL] or [MAIN_MODEL]
        reply = self.chat([{"role": "user", "content": prompt}], None, models, max_tokens)
        return read_json_object((reply or {}).get("content") or "")

    def world(self, actions: list[dict[str, Any]]) -> dict[str, Any] | None:
        state = self.state
        payload: dict[str, Any] = dict(state.binding)
        payload["step"] = state.step + 1
        if len(actions) == 1:
            payload["action"] = actions[0]
        else:
            payload["actions"] = actions
        for _ in range(3):
            remaining = self.seconds_left()
            if remaining < 2:
                return None
            _env.timeout = int(max(3, min(95, remaining - 1)))
            result = _env.post_verbose("/environment/call", payload)
            data = result.data
            if isinstance(data, dict):
                state.step = data["step"] if isinstance(data.get("step"), int) else state.step + len(actions)
                if data.get("finished"):
                    state.finished = True
                return data
            error = result.error or {}
            code = error.get("status")
            text = str(error.get("body") or "")
            state.trace.append({"kind": "env_failure", "status": code, "body": text[:300]})
            if code == 402:
                state.dead = True
                return None
            if code == 409:
                if "finished" in text:
                    state.finished = True
                    return None
                expected = re.search(r"expected (\d+)", text)
                if not expected:
                    return None
                state.step = int(expected.group(1)) - 1
                payload["step"] = state.step + 1
                continue
            if code in (403, 422, 500, 503, 504):
                return None
        return None


# ---- operations that need the shell ----

def post_replies(shell: Shell, replies: list[tuple[str, str]]) -> list[str]:
    texts = []
    for position, (call_id, text) in enumerate(replies):
        if position == len(replies) - 1:
            text += "\n" + shell.status()
        shell.state.messages.append({"role": "tool", "tool_call_id": call_id, "content": text})
        texts.append(text)
    return texts


def read_shopper(shell: Shell, lines: list[str], question: str | None) -> str:
    state = shell.state
    fresh = [line for line in lines if line not in state.heard]
    if not fresh:
        return ""
    state.heard.update(fresh)
    if shell.seconds_left() < WRAP_UP_SECONDS:
        return ""
    prompt = interpret_prompt(state, fresh, question)
    state.said.append((f"we asked: {question} -> " if question else "") + "shopper said: " + " / ".join(fresh))
    result = shell.judge(prompt, 1500, [READ_MODEL, JUDGE_MODEL])
    if not result:
        return ""
    if question and result.get("understood") is False:
        state.unclear_questions.append(question)
    return absorb_interpretation(state, result, question is not None)


def check_item(shell: Shell, item: str, purpose: str) -> str:
    state = shell.state
    if shell.seconds_left() < WRAP_UP_SECONDS + 20:
        return ""
    key = f"{item}|{state.ledger_version}"
    if key in state.verdicts:
        return state.verdicts[key]
    records = records_mentioning(state.records, item)
    if not records:
        return ""
    result = shell.judge(verify_prompt(state, item, purpose, records), 1400)
    state.trace.append({"kind": "verify", "item": item, "result": result})
    if not result:
        state.verdicts[key] = ""
        return ""
    checks = [entry for entry in result.get("checks") or [] if isinstance(entry, dict)]
    facts = item_facts(state.records, item)
    shown = [need for entry in state.plan.get("products") or [] for need in entry.get("requirements") or [] if literal_pass(facts, need)]

    def settled(check: dict[str, Any]) -> bool:
        # a fail the store's own words contradict (the value stands in the item's value for that property) is no fail
        said = words_of(check.get("requirement"))
        return any({word for word in words_of(need.get("property")) if len(word) >= 3} & said for need in shown)

    fails = [str(entry.get("requirement") or "") for entry in checks if entry.get("verdict") == "fail" and not settled(entry)]
    if not checks and result.get("fits") is False:
        fails = [str(problem) for problem in result.get("problems") or []] or ["a requirement"]
    unproven = [str(entry.get("requirement") or "") for entry in checks if entry.get("verdict") == "unproven" and not settled(entry)]
    for entry in checks:
        # a pass read from some other key, while this item has no key for the property and other opened listings
        # do, is the related-key reading the store does not accept
        if entry.get("verdict") != "pass" or settled(entry):
            continue
        point = str(entry.get("requirement") or "")
        names = property_names(state, point)
        quoted = str(entry.get("fact") or "").split(":", 1)[0] if ":" in str(entry.get("fact") or "") else ""
        own = [key for key in facts if names and names <= words_of(key)]
        if names and quoted and not names <= words_of(quoted) and not own and carried_elsewhere(state, item, point):
            fails.append(f"{point} (shown only under a related key, while other listings here state it)")
    # silence on a property this store does state for other opened listings is a miss, not an open point: such an
    # item rarely turns out right; silence on a property no opened listing states cannot be checked here and stays open
    shown_elsewhere = [point for point in unproven if carried_elsewhere(state, item, point)]
    fails += [f"{point} (its records never state it, while other listings here do)" for point in shown_elsewhere]
    products = state.plan.get("products") or []
    for need in products[0].get("requirements") or [] if len(products) == 1 else []:
        # an exclusion or a closed list holds only where the item's own key gives an allowed value: silence proves
        # nothing once other listings here state that property
        if not (need.get("op") == "is_not" or (need.get("op") == "is" and len(need.get("values") or []) >= 2)):
            continue
        names = {word for word in words_of(need.get("property")) if len(word) >= 3}
        own = [key for key in facts if names and (names <= words_of(key) or (len(names) == 1 and names & words_of(key)))]
        point = f"{need.get('property')}: {need.get('rule') or ', '.join(str(value) for value in need.get('values') or [])}"
        if names and not own and carried_elsewhere(state, item, point) and not any(names <= words_of(fail) for fail in fails):
            fails.append(f"{point} (its records never state {need.get('property')}, while other listings here do, so the rule is not shown to hold)")
    state.unproven[item] = [point for point in unproven if point not in shown_elsewhere]
    verdict = failed_check_text(item, fails) if fails else ""
    state.verdicts[key] = verdict
    if not verdict and checks and result.get("fits") is not False and item not in state.passed:
        state.passed.append(item)
    return verdict


def check_cart(shell: Shell) -> str:
    # every line of the cart is judged, since an order of several products fails on any one of them
    verdicts = [check_item(shell, item, "ORDER") for item in shell.state.cart_ids]
    return " ".join(verdict for verdict in verdicts if verdict)


def audit_before_decline(shell: Shell) -> str:
    state = shell.state
    # an item the full review already passed under the current requirements answers the decline by itself
    cleared = [item for item in state.passed if state.verdicts.get(f"{item}|{state.ledger_version}") == ""][:2]
    if cleared and not state.audited:
        state.audited = True
        return ("A separate review already checked " + ", ".join(cleared) + " against every current requirement, stock and price included, and found "
                + ("it fits" if len(cleared) == 1 else "they fit") + ". Declining would be wrong while such an item exists: put it in the cart and order it.")
    if state.audited or shell.seconds_left() < WRAP_UP_SECONDS + 30 or state.cost > COST_CAP:
        return ""
    state.audited = True
    prompt = audit_prompt(state)
    if prompt is None:
        return ""
    # the audit only proposes candidates (the review then checks them), so the cheaper reader goes first; the larger
    # LLM answers only when the reader returns nothing
    result = shell.judge(prompt, 2000, [READ_MODEL, JUDGE_MODEL])
    state.trace.append({"kind": "audit", "result": result})
    message = audit_message(result)
    # each suspected fit gets the full review; one that nothing contradicts means the decline would be wrong
    named = [str(entry) for entry in (result or {}).get("possible_fits") or [] if isinstance(entry, (str, int))][:1]
    passing = [item for item in named if records_mentioning(state.records, item) and not check_item(shell, item, "BUY INSTEAD OF DECLINING")]
    if passing:
        return ("A separate review checked " + ", ".join(passing) + " against every requirement and found nothing that rules "
                + ("it" if len(passing) == 1 else "them") + " out. Declining would be wrong while such an item exists: check its stock and price, put it in the cart and order it.")
    return message


def compare_tool(state: Session) -> tuple[str, str, int] | None:
    # a tool that takes one list of listing ids and shows them side by side (often given the view role)
    used = {tool for role, tool in state.roles.items() if tool and role not in ("view", "details")}
    for name, schema in state.schemas.items():
        properties = schema.get("properties") or {}
        if name in used or len(properties) > 2:
            continue
        lists = [key for key, spec in properties.items() if isinstance(spec, dict) and spec.get("type") == "array" and (spec.get("items") or {}).get("type") == "string"]
        if len(lists) == 1:
            spec = properties[lists[0]]
            return name, lists[0], int(spec.get("maxItems") or 4)
    return None


def widen_details(shell: Shell, product: str | None) -> bool:
    # the hidden need's property shows up among the menu's differing values only once several listings of that
    # product are open: before its first question, code opens unopened listed rows of it, a few per step, up to eight
    state = shell.state
    found = compare_tool(state)
    if not found or steps_left(state) <= 18 or shell.seconds_left() < WRAP_UP_SECONDS + 60:
        return False
    if state.widened.get(product or "", 0) >= 2:
        return False
    opened: set[str] = set()
    for observation in state.details:
        if not product or observation_product(state, observation) == product:
            gather_ids(observation, opened)
    if len(opened) >= 8:
        state.widened[product or ""] = 2
        return False
    name, key, size = found
    fresh = [row_id for row_id, row in state.rows.items()
             if row_id not in opened and row_id not in state.cart_ids and (not product or observation_product(state, row) == product)]
    fresh = fresh[:max(2, min(size, 4, 8 - len(opened)))]
    if len(fresh) < 2:
        state.widened[product or ""] = 2
        return False
    args, problems = validate({key: fresh}, state.schemas[name], "args")
    if problems:
        state.widened[product or ""] = 2
        return False
    state.widened[product or ""] = state.widened.get(product or "", 0) + 1
    call = tool_call(f"widen_{state.step}", name, args)
    state.trace.append({"kind": "widen", "product": product, "ids": fresh})
    state.messages.append({"role": "assistant", "content": "Opening more listings" + (f" of the {product}" if product else "") + " side by side before asking the shopper.", "tool_calls": [call]})
    post_replies(shell, execute_calls(shell, [call], planned=True))
    return True


def run_interview(shell: Shell, force: bool = False) -> bool:
    state = shell.state
    tool = state.roles.get("message")
    limit = message_limit(state)
    if not tool or state.messages_used >= limit:
        state.planner_on = False
        return False
    products = state.plan.get("products") or []
    target = None
    names = [str(entry.get("product") or "") for entry in products] or [str(state.plan.get("product") or "item")]
    # a product seldom hides more than one need (live race 2: about 3 of 119 episodes had two on one product), so
    # once a reply has revealed a need for it, the remaining questions go to the others first
    open_names = [name for name in names if name not in state.settled_products] or names
    if len(products) > 1:
        # in a several-product order the questions take the products in turn, the least asked first, so a hidden
        # need on any of them can come out; a product with no opened listing yet waits
        for name in sorted(open_names, key=lambda name: (state.asked_products.count(name), names.index(name))):
            if question_menu(state, name):
                target = name
                break
    if widen_details(shell, target):
        return True
    ranked = question_menu(state, target)
    menu = "\n".join(f"- {label} (in {count} listings): " + " | ".join(samples) for label, count, samples in ranked)
    fact_sets = sum(len(fact_maps(observation)) for observation in state.details)
    urgent = bool(state.plan.get("ask_first")) and not state.asked
    # questions wait until candidates have been opened: the first cart or finish move, four opened listings, or the
    # point where steps start to run short
    if not (force or urgent or fact_sets >= 4 or steps_left(state) <= 14):
        return False
    urgent = urgent and not force
    if steps_left(state) <= 10 or shell.seconds_left() < WRAP_UP_SECONDS + 40:
        state.planner_on = False
        return False
    count = limit - state.messages_used
    result = shell.judge(plan_prompt(state, count, menu, target), 2000, [READ_MODEL, JUDGE_MODEL])
    # only properties the opened listings carry are asked; a story hint only moves such a property forward
    known = {label: samples for label, _, samples in ranked}
    single_product = str(products[0]["product"]) if len(products) == 1 else str(state.plan.get("product") or "item")
    picks = []
    for entry in (result or {}).get("questions") or []:
        if not isinstance(entry, dict):
            continue
        prop = re.sub(r"[_.]+", " ", str(entry.get("property") or "")).strip().lower()
        written = str(entry.get("question") or "").strip()
        if prop in known and prop not in state.unclear and prop not in state.asked_keys:
            # one property per message, named the way the listings name it, as a short plain question
            product = target or (str(entry.get("product") or "").strip() if len(products) > 1 else single_product) or single_product
            # the question says what is needed and gives two values the listings show, so even a shopper who
            # follows only concrete questions can answer it
            shown = [value for value in known[prop] if 0 < len(value) <= 24 and len(value.split()) <= 3 and "," not in value and not words_of(value) <= words_of(prop)]
            pair = next(((first, second) for first in shown for second in shown[::-1]
                         if not words_of(first) <= words_of(second) and not words_of(second) <= words_of(first)), None)
            example = f", such as {pair[0]} or {pair[1]}" if pair else ""
            text = f"What {prop} do you need for the {product}{example}?"
            picks.append({"property": prop, "question": text, "product": product})
        elif written and urgent and not state.asked:
            picks.append({"property": str(entry.get("property") or "the first thing to learn"), "question": written})
    # a story-hinted property the listings carry goes first, whatever order the planner gave
    hinted = [str(hint).strip().lower().rsplit(":", 1)[-1].strip() for hint in state.plan.get("hint_properties") or [] if str(hint).strip()]
    def hint_rank(pick: dict[str, Any]) -> int:
        names = words_of(pick.get("property"))
        return 0 if any(names and (names <= words_of(hint) or words_of(hint) <= names) for hint in hinted) else 1
    picks.sort(key=hint_rank)
    # one question per round: the reply is read before the next pick, so a reveal or a reply the shopper did not
    # understand shapes what is asked next
    picks = picks[:1]
    calls = []
    topics = []
    for position, pick in enumerate(picks):
        question = str(pick.get("question")).strip()
        args = message_arguments(state.schemas.get(tool) or {}, question)
        if not args:
            continue
        args, problems = validate(args, state.schemas[tool], "args")
        if problems:
            continue
        calls.append(tool_call(f"ask_{state.messages_used + position + 1}_{state.step}", tool, args))
        topics.append((str(pick.get("property")), question))
        state.asked_products.append(str(pick.get("product") or target or single_product))
    if not calls:
        state.planner_on = False
        return False
    state.messages.append({"role": "assistant", "content": "Asking the shopper about " + ", next ".join(topic for topic, _ in topics) + ".", "tool_calls": calls})
    known_needs = len(state.revealed)
    texts = post_replies(shell, execute_calls(shell, calls, planned=True))
    if len(state.revealed) > known_needs:
        state.settled_products.append(state.asked_products[-1] if len(products) > 1 and state.asked_products else names[0])
    for (topic, question), text in zip(topics, texts):
        answer = text.split("\n[progress]")[0][:400]
        state.asked_keys.append(topic.lower())
        if question in state.unclear_questions:
            # the shopper could not follow it: that topic is set aside and the next round asks a different one
            state.unclear.append(topic.lower())
            answer += " (not understood: do not ask this again)"
        state.asked.append(f"- {topic}: we asked \"{question}\"; reply: {answer}")
    if state.messages_used >= limit:
        state.planner_on = False
    return True


def relay_answer(shell: Shell, text: str) -> bool:
    state = shell.state
    tool = state.roles.get("message")
    if not tool or state.messages_used >= message_limit(state) or steps_left(state) <= 4 or shell.seconds_left() < WRAP_UP_SECONDS:
        return False
    args = message_arguments(state.schemas.get(tool) or {}, text)
    if not args:
        return False
    args, problems = validate(args, state.schemas[tool], "args")
    if problems:
        return False
    call = tool_call(f"reply_{state.step}", tool, args)
    state.messages.append({"role": "assistant", "content": "Replying to the shopper's question.", "tool_calls": [call]})
    post_replies(shell, execute_calls(shell, [call], planned=True))
    return True


def finish_arguments(state: Session, tool: str, declining: bool, text: str) -> dict[str, Any]:
    # code-filled arguments for the order or the decline; when one tool does both through a field with fixed choices,
    # the setup reading of the tool names the choice for each, and the first choice orders, the last declines otherwise
    schema = state.schemas.get(tool) or {}
    args = filler_arguments(schema, text)
    chosen = (state.plan.get("finish_choices") or {}) if isinstance(state.plan.get("finish_choices"), dict) else {}
    picked = chosen.get("decline" if declining else "order")
    picked = picked if isinstance(picked, dict) else {}
    described = " ".join(str((entry.get("function") or {}).get("description") or "") for entry in state.tools if (entry.get("function") or {}).get("name") == tool)
    described = (described + " " + " ".join(str(spec.get("description") or "") for spec in (schema.get("properties") or {}).values() if isinstance(spec, dict))).lower()
    for key, spec in (schema.get("properties") or {}).items():
        choices = [str(value) for value in spec.get("enum") or []] if isinstance(spec, dict) else []
        if key not in args or len(choices) < 2:
            continue
        if str(picked.get(key)) in choices:
            args[key] = str(picked[key])
            continue
        # without the setup reading: the words the description puts right after each choice are set against the two
        # role meanings the setup prompt gives; the choice reading most like declining declines
        def leaning(value: str) -> int:
            found = re.search(r"\b" + re.escape(value.lower()) + r"\b", described)
            after = described[found.end() : found.end() + 60] if found else ""
            ends = [after.find(other.lower()) for other in choices if other != value and after.find(other.lower()) >= 0]
            said = {word for word in stems(value + " " + (after[: min(ends)] if ends else after)) if len(word) >= 3}
            return len(said & stems(DECLINE_MEANING)) - len(said & stems(ORDER_MEANING))
        scores = {value: leaning(value) for value in choices}
        if len(set(scores.values())) > 1:
            args[key] = (max if declining else min)(choices, key=lambda value: scores[value])
        else:
            args[key] = choices[-1 if declining else 0]
    return args


def order_step(shell: Shell) -> str | None:
    state = shell.state
    if not state.order_due:
        return None
    state.order_due = False
    order = state.roles.get("order")
    reply = shell.world([{"name": order, "args": finish_arguments(state, order, False, "Order the cart.")}])
    state.trace.append({"kind": "passed_cart_order", "reply": reply})
    return STOP if state.finished else None


def forced_finish(shell: Shell) -> None:
    state = shell.state
    if state.finished or state.dead or steps_left(state) <= 0 or shell.seconds_left() < 2:
        return
    order, decline = state.roles.get("order"), state.roles.get("decline")
    if order and state.cart_used:
        action = {"name": order, "args": finish_arguments(state, order, False, "Order the cart.")}
    elif decline:
        action = {"name": decline, "args": finish_arguments(state, decline, True, DECLINE_TEXT)}
    else:
        return
    reply = shell.world([action])
    state.trace.append({"kind": "forced_finish", "action": action, "reply": reply})
    refused = bool(reply) and not reply.get("finished") and isinstance(reply.get("observation"), dict) and "refusal" in reply["observation"]
    if refused and action["name"] == order and steps_left(state) > 1:
        # a refused order is often accepted when sent again (a price or stock notice that the cart survives): one
        # more try comes before giving up on the cart
        reply = shell.world([action])
        state.trace.append({"kind": "forced_finish", "action": action, "reply": reply})
        refused = bool(reply) and not reply.get("finished") and isinstance(reply.get("observation"), dict) and "refusal" in reply["observation"]
    if refused and decline and steps_left(state) > 0:
        action = {"name": decline, "args": finish_arguments(state, decline, True, DECLINE_TEXT)}
        reply = shell.world([action])
        state.trace.append({"kind": "forced_finish", "action": action, "reply": reply})


def screen_call(shell: Shell, call: dict[str, Any], context: dict[str, Any]) -> tuple[str | None, dict[str, Any] | None]:
    state = shell.state
    allowed = context["allowed"]
    finishing = context["finishing"]
    message_tool, cart_tool = state.roles.get("message"), state.roles.get("cart")
    function = call.get("function") or {}
    name = function.get("name")
    if name not in state.schemas:
        return f"Skipped, no step spent: there is no tool called '{name}'. Available: {', '.join(sorted(allowed))}.", None
    if not context["planned"] and name not in allowed:
        if name == message_tool and state.planner_on:
            return "Skipped, no step spent: shopper questions are chosen and sent automatically once you have opened the details of a few listings.", None
        return "Skipped, no step spent: not enough steps remain for that; the tools still usable are: " + ", ".join(sorted(allowed)) + ".", None
    raw = function.get("arguments") or ""
    args = read_json_object(raw) if raw.strip() else {}
    if args is None:
        return "Skipped, no step spent: the arguments could not be read as a JSON object.", None
    args, problems = validate(args, state.schemas[name], "args")
    if problems:
        return "Skipped, no step spent: the arguments break the schema: " + "; ".join(problems[:6]), None
    if name in finishing and context["count"] > 1:
        return "Skipped, no step spent: send an order or a decline by itself, once the results of your other calls are in.", None
    if name == message_tool and state.message_budget is not None and state.messages_used + context["messages"] >= state.message_budget:
        return "Skipped, no step spent: no messages to the shopper are left. Stop messaging them.", None
    signature = name + to_json_text(args)
    if signature in state.recent and name not in finishing and name not in (message_tool, cart_tool):
        return "Skipped, no step spent: this is the same call you just made and nothing has changed; its result is above. Use different arguments or a different move.", None
    if name == cart_tool:
        if steps_left(state) > 3:
            properties = state.schemas[name].get("properties") or {}
            if state.add_signature is None or enum_signature(properties, args) == state.add_signature:
                listed = line_ids(args)
                if listed is not None:
                    # a whole-cart call: the first listing not yet in the cart is the one to review
                    target = next((value for value in listed if value not in state.cart_ids and f"{value}|{state.ledger_version}" not in state.cart_checked), None)
                else:
                    target = next((value for key, value in args.items() if isinstance(value, str) and "enum" not in (properties.get(key) or {}) and value.strip() and " " not in value.strip()), None)
                checked_key = f"{target}|{state.ledger_version}"
                if target and checked_key not in state.cart_checked:
                    state.cart_checked.add(checked_key)
                    verdict = check_item(shell, target, "ADD TO THE CART")
                    if verdict:
                        return "Skipped, no step spent. " + verdict + " Repeat the identical call if you are sure it fits.", None
                    open_points = [point for point in state.unproven.get(target) or [] if point]
                    if open_points and target not in state.noted:
                        state.noted.add(target)
                        return ("Skipped, no step spent. A separate review finds nothing that rules " + target + " out, but its records never state: "
                                + "; ".join(open_points[:3]) + ". If an opened item states those and fits, prefer it; otherwise repeat the identical call."), None
        warning = quantity_problem(state, name, args)
        if warning and signature not in state.qty_warned:
            state.qty_warned.add(signature)
            return "Skipped, no step spent: " + warning + " Repeat the identical call if you are sure.", None
    if name == message_tool:
        context["messages"] += 1
    return None, args


def cheaper_unopened(state: Session) -> list[tuple[float, str]]:
    # under a cheapest rule: listed rows of the cart item's own department, cheaper than it, never opened
    if state.plan.get("priority") != "cheapest" or not state.rows or not state.cart_ids:
        return []
    rows = list(state.rows.values())
    numeric: dict[str, int] = {}
    labels: dict[str, set[str]] = {}
    for row in rows:
        for key, value in row.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                numeric[key] = numeric.get(key, 0) + 1
            elif isinstance(value, str) and 0 < len(value) <= 60:
                labels.setdefault(key, set()).add(value)
    if not numeric:
        return []
    # the price is the number every row carries; the department the short label with the fewest different values
    price_key = max(numeric, key=numeric.get)
    shared = [key for key, seen in labels.items() if sum(1 for row in rows if key in row) >= 0.8 * len(rows)]
    group_key = min(shared, key=lambda key: len(labels[key]), default=None)
    opened: set[str] = set()
    for observation in state.details:
        gather_ids(observation, opened)
    found: list[tuple[float, str]] = []
    for item in state.cart_ids:
        mine = state.rows.get(item)
        if not mine or not isinstance(mine.get(price_key), (int, float)):
            continue
        for row_id, row in state.rows.items():
            price = row.get(price_key)
            if row_id in opened or row_id in state.cart_ids or not isinstance(price, (int, float)) or price >= mine[price_key]:
                continue
            if group_key and row.get(group_key) != mine.get(group_key):
                continue
            found.append((float(price), row_id))
    return sorted(set(found))[:5]


def gate_finish(shell: Shell, name: str, args: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
    state = shell.state
    ready_to_check = name == state.roles.get("order") and not state.order_checked and steps_left(state) > 2
    if ready_to_check and shell.seconds_left() > WRAP_UP_SECONDS:
        state.order_checked = True
        verdict = check_cart(shell)
        if verdict:
            return "On hold, no step spent. " + verdict, args
        cheaper = cheaper_unopened(state)
        if cheaper and not state.cheap_held and steps_left(state) > 8:
            # the cheapest rule is judged on every listed candidate: cheaper rows of the same department that were
            # never opened are looked at before the order, once
            state.cheap_held = True
            listed = ", ".join(f"{row_id} ({price:g})" for price, row_id in cheaper)
            return ("On hold, no step spent. The request wants the cheapest fitting item, and these cheaper listings of the same department were never opened: "
                    + listed + ". Open them (the cheapest first); if one meets every requirement, buy it instead, otherwise order your pick again."), args
        return "On hold, no step spent. Before ordering, confirm: (1) the cart has one line per product the request wants (usually one), each the chosen item at its required quantity, and nothing else; (2) since your last cart change you have read this item's stock and price again with a tool that shows them, and it is in stock with price times quantity inside the current budget; (3) its own properties satisfy every stated requirement, every need the shopper revealed and every announced change; (4) if the cheapest or best was requested, no better fitting item exists among all you could find; (5) every procedural rule of the request has been obeyed. Check whatever is still open now; otherwise send the order call again.", args
    if name != state.roles.get("decline"):
        return None, args
    if not state.decline_checked and steps_left(state) > 6 and shell.seconds_left() > WRAP_UP_SECONDS + 30:
        state.decline_checked = True
        audit = audit_before_decline(shell)
        if audit:
            return "On hold, no step spent. " + audit, args
        return "On hold, no step spent. Before you decline, make sure that you searched in several ways (other words, filters, pages), that you read each candidate's own properties, and that every one fails a stated requirement, a revealed need, the budget or stock. Close synonyms and any spelling of a key do count. If something fits, buy it; if nothing does, send the decline call again.", args
    cart = state.cart_ids
    # a second decline while the cart's one line is an item the full review passed under the current requirements is
    # answered by ordering that cart, once
    if (not state.order_tried and len(state.plan.get("products") or []) <= 1 and len(cart) == 1 and cart[0] in state.passed
            and state.verdicts.get(f"{cart[0]}|{state.ledger_version}") == "" and state.roles.get("order") and steps_left(state) > 2):
        state.order_tried = True
        state.order_due = True
        return "On hold, no step spent. The cart holds " + cart[0] + ", which the full review passed against every current requirement, so code orders it now.", args
    return None, scrub_decline(state, args, name)


def deliver(shell: Shell, chunk: list[tuple[str, str, dict[str, Any]]], replies: list[tuple[str, str]]) -> None:
    state = shell.state
    message_tool = state.roles.get("message")
    actions = [{"name": name, "args": args} for _, name, args in chunk]
    first_step = state.step + 1
    reply = shell.world(actions)
    state.trace.append({"kind": "env", "step": first_step, "actions": actions, "reply": reply})
    if reply is None:
        replies.extend((call_id, "The world call did not go through (network or server). Try it again.") for call_id, _, _ in chunk)
        return
    observations = reply.get("observations")
    if not isinstance(observations, list):
        observations = [reply.get("observation")]
    last = len(observations) - 1
    for position, (call_id, name, args) in enumerate(chunk):
        if position > last:
            replies.append((call_id, "Skipped: the episode ended before this call ran."))
            continue
        observation = observations[position]
        book_result(state, name, args, observation)
        note = observation_notes(state, name, observation, args)
        heard, question = shopper_lines(name, observation, args, message_tool)
        if heard:
            meaning = read_shopper(shell, heard, question)
            if meaning:
                note = f"{note} {meaning}" if note else meaning
        # the loop sees a lean copy (empty fields dropped, long texts cut; search rows shorter than opened details);
        # the reviews read the full records
        text = to_json_text(lean_copy(observation, 110 if name == state.roles.get("search") else 600))
        if position == last and reply.get("finished"):
            text += " [episode over]"
        if note:
            text += "\n[agent note] " + note
        replies.append((call_id, text))


def execute_calls(shell: Shell, calls: list[dict[str, Any]], planned: bool = False) -> list[tuple[str, str]]:
    state = shell.state
    finishing = finishing_tools(state)
    context = {"allowed": {tool["function"]["name"] for tool in offered_tools(state)}, "finishing": finishing, "planned": planned, "count": len(calls), "messages": 0}
    replies: list[tuple[str, str]] = []
    runnable: list[tuple[str, str, dict[str, Any]]] = []
    for call in calls:
        refusal, args = screen_call(shell, call, context)
        if refusal is not None:
            replies.append((call["id"], refusal))
        else:
            runnable.append((call["id"], call["function"]["name"], args))
    if runnable and runnable[0][1] in finishing:
        call_id, name, args = runnable[0]
        held, args = gate_finish(shell, name, args)
        if held is not None:
            replies.append((call_id, held))
            return replies
        runnable[0] = (call_id, name, args)
    elif runnable and finishing:
        room = max(0, steps_left(state) - 1)
        if room < len(runnable):
            replies.extend((call_id, "Skipped, no step spent: only one step is left, and that step belongs to the order (or the decline).") for call_id, _, _ in runnable[room:])
            runnable = runnable[:room]
    for start in range(0, len(runnable), 16):
        chunk = runnable[start : start + 16]
        if state.finished or state.dead or steps_left(state) <= 0:
            replies.extend((call_id, "Skipped: the episode has ended or the steps are gone.") for call_id, _, _ in chunk)
            continue
        deliver(shell, chunk[: steps_left(state)], replies)
    position_of = {call["id"]: position for position, call in enumerate(calls)}
    return sorted(replies, key=lambda entry: position_of.get(entry[0], 0))


# ---- phases ----

def setup_phase(shell: Shell) -> None:
    state = shell.state
    reply = shell.chat([{"role": "user", "content": setup_content(state)}], None, [SETUP_MODEL, BACKUP_MODEL], 2600)
    absorb_setup(state, read_json_object((reply or {}).get("content") or "") or {})
    state.trace.append({"kind": "plan", "plan": state.plan, "roles": state.roles})


def opening_phase(shell: Shell) -> None:
    state = shell.state
    opening = state.policy.get("opening")
    sections = ["THE SHOPPER'S REQUEST:\n" + request_text(state)]
    if opening is not None:
        sections.append("OPENING OBSERVATION:\n" + to_json_text(opening))
    brief = reading_brief(state)
    if brief:
        sections.append(brief)
    sections.append(shell.status())
    if opening is not None:
        state.records.append(opening)
        gather_ids(opening, state.seen_ids)
    state.messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": "\n\n".join(sections)}]


def deadline_step(shell: Shell) -> str | None:
    state = shell.state
    if state.finished or state.dead or steps_left(state) <= 0:
        return STOP
    if shell.seconds_left() < FORCE_SECONDS:
        forced_finish(shell)
        return STOP
    if (state.cost > COST_STOP and state.cart_used) or state.cost > COST_CAP:
        # the task's spending limit: with something in the cart the code orders it, with nothing it declines; no more
        # LLM turns are spent
        state.trace.append({"kind": "spend_stop", "cost": round(state.cost, 4)})
        forced_finish(shell)
        return STOP
    return None


def relay_step(shell: Shell) -> str | None:
    state = shell.state
    if not state.pending_reply:
        return None
    text, state.pending_reply = state.pending_reply, ""
    return NEXT if relay_answer(shell, text) else None


def interview_step(shell: Shell) -> str | None:
    return NEXT if shell.state.planner_on and run_interview(shell) else None


def model_step(shell: Shell) -> str:
    state = shell.state
    reply = shell.chat(state.messages, offered_tools(state), [MAIN_MODEL, BACKUP_MODEL, MAIN_MODEL], 2500)
    if reply is None:
        state.failures += 1
        if state.failures < 2:
            return NEXT
        forced_finish(shell)
        return STOP
    calls = normalise_calls(state, reply.get("tool_calls") or [])
    content = reply.get("content") or ""
    if not isinstance(content, str):
        content = to_json_text(content)
    committing = {state.roles.get("cart")} | finishing_tools(state)
    committing.discard(None)
    if state.planner_on and any(call["function"]["name"] in committing for call in calls) and run_interview(shell, force=True):
        return NEXT
    if not calls:
        state.idle += 1
        state.messages.append({"role": "assistant", "content": content or "(no tools called)"})
        state.messages.append({"role": "user", "content": "Each reply needs at least one call to the supplied tools. Go on.\n" + shell.status()})
        if state.idle < 3:
            return NEXT
        forced_finish(shell)
        return STOP
    state.idle = 0
    state.trace.append({"kind": "model", "content": content[:2000], "calls": [call["function"] for call in calls]})
    state.messages.append({"role": "assistant", "content": content or None, "tool_calls": calls})
    post_replies(shell, execute_calls(shell, calls))
    return NEXT


TURN_STEPS = (deadline_step, order_step, relay_step, interview_step, model_step)


def turn_phase(shell: Shell) -> None:
    for _ in range(MAX_TURNS):
        outcome = None
        for step in TURN_STEPS:
            outcome = step(shell)
            if outcome is not None:
                break
        if outcome == STOP:
            return


def closing_phase(shell: Shell) -> None:
    state = shell.state
    if not state.finished and not state.dead:
        forced_finish(shell)
    state.trace.append({"kind": "end", "finished": state.finished, "steps": state.step, "cost": round(state.cost, 5), "seconds": round(shell.elapsed(), 1)})


PHASES = (setup_phase, opening_phase, turn_phase, closing_phase)


def drive(shell: Shell) -> list[dict[str, Any]]:
    if not shell.state.finished:
        for phase in PHASES:
            phase(shell)
    return shell.state.trace


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    shell = Shell(problem_data)
    try:
        return drive(shell)
    except Exception as exc:
        trace = shell.state.trace
        trace.append({"kind": "crash", "error": f"{type(exc).__name__}: {exc}"[:500]})
        try:
            if not shell.state.finished and not shell.state.dead:
                forced_finish(shell)
        except Exception as again:
            trace.append({"kind": "crash", "error": f"{type(again).__name__}: {again}"[:500]})
        return trace


__all__ = ["agent_main"]
