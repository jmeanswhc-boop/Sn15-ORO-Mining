from __future__ import annotations

import json
import re
import threading
from contextlib import contextmanager
from typing import Any, Iterator

from src.agent.proxy_client import ProxyClient

PILOT_MODEL = "anthropic/claude-haiku-5.5"
RESERVE_MODEL = "openai/gpt-6-luna"
BRIEFING_MODEL = "anthropic/claude-sonnet-5.5"
REFEREE_MODEL = "anthropic/claude-sonnet-5.5"
HEARING_MODEL = "anthropic/claude-haiku-5.5"
SECOND_VENDOR = "google/gemini-3.8-flash"     # stands behind the referee so a failed judge call does not fall to the small model
SPEND_STOP = 0.085                             # USD per episode; past it the episode is finished from code (the run key has a cap)
CHECK_MODEL = "anthropic/claude-opus-5.5"      # price row used for unknown models in the cost estimate
# USD per million tokens (fresh input, cached input, output); used only when a reply carries no cost figure
TOKEN_PRICES = {"anthropic/claude-haiku-5.5": (0.10, 0.01, 0.50), "anthropic/claude-sonnet-5.5": (2.0, 0.10, 10.0),
                "anthropic/claude-opus-5.5": (4.0, 0.20, 20.0), "openai/gpt-6-luna": (0.10, 0.01, 0.50)}

RUN_ALLOWANCE = 265.0
WIND_DOWN_AT = 70.0
CLOSE_NOW_AT = 28.0
RESULT_TEXT_CAP = 16000
TURN_LIMIT = 80
TICK_LENGTH = 0.25

PILOT_GUIDE = """You are a careful, thorough shopping assistant working for a single shopper in a simulated store, and the supplied tools are your only way to act. Scoring is 1 or 0. You earn 1 only when your finish is completely correct: the ordered item satisfies every requirement the shopper wrote, every hidden need the shopper discloses when asked, and every change the shopper announces during your work; quantity and budget are correct; when the request calls for the cheapest (or best) choice, that is the one you make; and each rule stated in the request is obeyed. When no item in the store can satisfy all of it, the correct finish is to decline, precisely as the request describes.

THE WORLD
- Each tool call costs one step, failed calls and calls returning errors included, and steps are limited. Thinking costs nothing; steps do.
- When no call needs another call's output, you may place several tool calls in a single reply (for example, opening several listings' details together). They are executed in order. An order or a decline must never share a reply with other calls.
- When one tool reveals the details of several listings per call (a compare tool), inspect candidates with it in groups: several listings for one step.
- Each store has its own tool names, arguments and result shapes. Study the tool descriptions and JSON schemas closely and use precisely the argument names and allowed values they define. Work out from the first results which tool reveals which facts (price, stock, detailed properties).
- Discover early which category labels the store gives the requested kind of product (they are shown on listings). Where neighbouring labels exist, pick the label that matches the wording of the request as a whole; do your searching and buying inside it.
- Search tools may be restricted: a cap on visible rows or pages, results ordered by id instead of relevance or price, every word required, filters working in one mode only. Set the reported total against the rows you can really see, and narrow down (more specific words, category or price filters, further pages) until the whole relevant set has been seen.

THE SHOPPER AND HIDDEN NEEDS
- The shopper holds needs they never wrote. A need is disclosed only when a message asks about that property. Only the FIRST thing a message asks gets an answer, so each message must ask about exactly ONE property, as a brief, plain, friendly question to the shopper naming the property (one or two example values are fine). Give no instructions, mention no lists, labels, positions or rules, and never ask two things together.
- When a reply says they don't mind, have no preference, or anything is fine, that property is free. When a reply states a preference or a limit, it is a hard requirement from then on, exactly as phrased (numeric limits included).
- If a planner is available, your questions are picked and sent for you once you have opened the details of two or three listings of the requested kind, so do that among your first moves. The shopper's answers then show up in the conversation with their meaning; treat every disclosed need as a hard requirement and search again with it together with the stated values.
- If the asking falls to you: first open the details of two or three listings, then ask about properties the story points to (who it is for, how it must look, where or how it will be used), and after that about properties the listings carry that vary between candidates. Never ask about anything the request already settles, and never go beyond the message budget.

RULES OF PROCEDURE
- The request may lay down rules on how to work: how many listings to view or compare before choosing, what to verify, what to tell the shopper, what to write in a remark, how to decline. They are graded just like requirements: carry out each one literally and in full. A rule to view or open N listings means N listings that fit completely, each opened with the details tool.
- When the shopper asks you something (for example to confirm a change), it gets answered for you while messages remain; treat the change itself as binding.
- What the shopper would merely like ("ideally", "would be nice", "if possible") is not a requirement: use it only to decide between items that already satisfy everything, and never above a cheapest rule.
- When the request leaves open something only the shopper can tell you (for example no price limit, or two wishes that may not both be achievable where the shopper must say which matters more), learn it by asking before you choose.

CHANGES
- During your work the shopper may change things: a new budget (higher or lower), a requirement removed, swapped or added, a price rising, an item selling out. A single shopper line can hold several changes. Shopper lines can show up inside any tool result (an extra field containing sentences) or come attached to an order attempt. Every change binds from that moment on: check your pick and your backups against it again, and when the request asks for the cheapest or best, choose anew from the new acceptable set.
- An order attempt may be REFUSED. In that case the result holds a "refusal" field, even when the inner result appears successful. The episode goes on and the cart stays as it was. Read the refusal, apply each change it states, take out of the cart whatever is no longer acceptable (for example a sold-out item), add the next verified fit, check stock and price again, and order again.
- Items may also sell out or change price without notice. For that reason stock and price have to be checked again just before each order attempt; an item whose current price multiplied by quantity exceeds the current budget no longer fits.

HOW TO JUDGE A LISTING (this decides the score)
- A requirement counts as met only when the listing's own property for it states the value. Property keys are spelled in different ways (spaces, dots, underscores, prefixes) and one property may show up under several keys; any key naming that property counts. Values match on meaning: a single element of a comma-separated list suffices; singular and plural are equal; close synonyms count; case is irrelevant. A word that only begins the same way does not count, and a broader or different class does not count (for example girls is not women, plain unisex is not female).
- Treat the title or description as evidence only when the listing carries no property key at all for that requirement. When the listing's own property states something else, the listing fails even if the title or description states the value.
- A value under some other, related property does not count (a length field does not meet a style requirement, the lining's material does not meet the material).
- Numbers: compare them numerically after converting units; limits include the edge value; a range counts only when the entire range satisfies the limit.
- Honour exclusions ("leave out anything with X") and closed lists ("A or B and nothing else").
- The listing's category covers the requested kind of product; never require some property to repeat the category's name. The item needs to be the requested kind of product: when the store tags listings with categories, it needs to be in the category whose label matches the request's wording as a whole (a request for "A or B" matches a category called "A & B", not a narrower neighbour called "B"). Stock is required when the request asks for things on hand, and price multiplied by quantity needs to be within the budget (equal is allowed).
- A property whose value names several options, the required one among them (a comma list), simply matches; that is not mixed evidence. Evidence is mixed only when the listing's own keys for the property give different, incompatible values. Only in that case, and only when no cheapest or best rule settles it, favour a candidate with clean evidence.

CHEAPEST OR BEST
- When the request asks for the cheapest, you must order the lowest-priced item among ALL items satisfying every requirement. Spotting one fitting item is not enough. Apply price filters when a tool provides them, drop the price limit until every remaining row is visible, sort the visible candidates by price yourself, and verify them from the lowest price up; the first that fits completely, with each cheaper one checked and rejected, is the answer. Hold the next fitting ones as backups.
- When the request asks for the best by some measure (most, largest, longest, highest, and so on), compare ALL fitting items on that measure in the same way and choose the extreme. An item whose listing gives no value for that measure cannot win.

FINISHING
- Maintain a ranked list of verified fits with at least one backup, since your first pick may sell out at checkout.
- Before each order attempt: the cart contains exactly one line, the chosen item at the required quantity, and nothing more (remove every other line); you have read the chosen item's stock and price again after your last cart change, and it is in stock and within budget.
- Decline only once a thorough search turns up nothing that satisfies every requirement. A decline scores 0 whenever some listing could have satisfied the stated requirements and the revealed needs, so when the choice is between ordering a listing that satisfies them and declining, order. Follow the request's declining rule exactly. When the request says not to offer substitutes, the decline remark must not name, describe or point to any listing: no ids, no titles, no prices.
- Budget your steps: make your first order attempt while at least seven steps are left, so a refusal can still be dealt with. Once 7 steps remain, only cart, stock-check and finishing tools are still available. Decide once: do not switch the cart back and forth, as every cart change uses a step.
- Every reply needs at least one tool call. Keep the reply text brief: your current requirements, the verified fits with price and stock, and your next move."""

BRIEFING_GUIDE = """Read the shopper's request and the store's tool list given below. Return ONE compact JSON object (no indentation) and nothing besides it, holding these keys:

"product": the kind of product asked for, in the words of the request.
"quantity": how many units of one listing to order (an integer; 1 when the request does not ask for more than one). An item normally sold as a single unit, such as a pair of shoes or a set, counts as quantity 1.
"requirements": a list of strings, one for each firm requirement stated on the product, each written as "property: constraint" carrying the shopper's exact value, limit, exclusion or closed list. Leave out price, stock, quantity, budgets and mere wishes, and do not repeat the product kind or category as a requirement; delivery dates, arrival times and shipping speed are never product requirements, because the store lists no such facts: words forming part of the category's name (a season, an occasion or an audience in the category name) are covered by the category and are not a separate property requirement. A requirement is only what the request states plainly as a condition on the product, a property whose value the shopper spells out; what the story only implies (a situation that makes some property matter, a look or a feel mentioned in passing) is not a requirement and belongs in "hints".
"hints": a list of objects {"property": ..., "likely": ..., "why": ...}: the product properties the story of the request points to without giving a value, named the way a listing would name them (for example "water resistance" when the shopper will swim with the watch, "pile height" when a door must open over the rug), with the value the story suggests in "likely" (or "" when it suggests none) and the request's own words in "why". Hints are what to ask the shopper about first and what to favour among items that already fit; they never justify rejecting an item. Empty when the request is plain.
"price_cap": the highest total price as a number, or null when none is given.
"currency": the price unit used by the request, or null.
"in_stock_only": true when the request wants only items really on hand or in stock.
"priority": "cheapest" only when the request explicitly asks for the cheapest or lowest-priced fitting option; "best" only when it explicitly asks for the best by some measure; otherwise "none". Comments that money is tight or that every penny matters do not ask for the cheapest; they only explain the price cap.
"priority_detail": for "best", the measure that defines "best"; otherwise "".
"message_budget": how many messages the shopper lets you send, as an integer, or null when not given.
"step_budget": how many moves or steps the request allows, as an integer, or null when not given.
"rules": a list of every procedural rule the request sets for the assistant (how many listings to view, what to verify or tell, how to decline, what to avoid, what to include), quoted word for word. Comments about what the shopper will do personally are story, not rules.
"soft_wants": a list of things the shopper would like without strictly requiring them (phrased as ideally, preferably, would be nice, if possible). Keep them out of "requirements".
"ask_first": a list of things the agent can learn only from the shopper and should ask about before choosing: for example the budget when no price limit is given, which of two wishes matters more when the request says both may not be possible (then also put both wishes in soft_wants, not in requirements), or anything the request says to check with the shopper. Empty when there are none.
Each sentence of the request must end up somewhere: a requirement, the priority, a soft want, a rule, a budget, a hint, or pure story. When in doubt whether a sentence is a rule, put it in "rules".
"decline_no_names": true when the request's declining rule forbids offering, naming or suggesting substitutes or alternatives; false when it asks you to name something; null when the request gives no such rule.
"roles": an object giving the tool for each role, with the exact tool names from the list, or null where no tool fits:
  "search", "details", "view", "message", "cart", "cart_view", "order", "decline".
  "message" is the tool that sends the shopper a message. "cart" is the tool that adds to or takes out of the cart. "order" is the tool that buys or checks out. "decline" is the tool that ends the task without a purchase.
"finish_args": when a single tool both buys and ends without buying, an object {"order": {...}, "decline": {...}} holding the argument values that pick each, for example {"order": {"mode": "buy"}, "decline": {"mode": "stop"}}; null otherwise.
"""

HEARING_GUIDE = """You help a shopping assistant work out what a shopper has just said. Return ONE JSON object and nothing besides it, holding these keys:
"meaning": a single short sentence stating what the shopper means.
"needs": a list of new firm requirements on the product, each written as "property: constraint" with the exact value, limit, exclusion or closed list the shopper gave.
"no_preference": a list of properties the shopper said they do not care about.
"dropped": a list of earlier requirements the shopper no longer needs.
"new_budget": the new highest total price as a number, or null.
"unavailable": true when the shopper says an item can no longer be purchased (sold out, gone, out of stock, button disabled), else false.
"requirements_now": the full list of firm requirements on the product as they stand after all the shopper has said so far, worked out again from the request, the earlier shopper lines and this one, each as "property: constraint". Keep each earlier requirement unless the shopper dropped or replaced it; for a replaced value keep only the latest one; add each newly revealed need. Leave out price, stock, quantity and the product category, and never add a requirement taken from the request's story or from your own interpretation of the situation: only what the request states plainly and what the shopper has said.
"reply": when the shopper asks the assistant something or asks for a confirmation, a brief, plain reply to send back that answers it and names the change in substance (for example confirming the new value); otherwise "".
A single line can hold several changes: report each of them. A requirement swapped for another is both dropped (the old one) and a new need (the new one). When the shopper says which of two wishes matters more, the favoured wish is a need. When the shopper gives a budget, it is new_budget. A mere wish ("ideally", "would be nice") is not a need. Read the shopper literally and generously. Folksy, figurative or indirect wording that names a value means the shopper firmly wants that value; count a value as excluded only when the shopper clearly says to avoid it. A number with a comparison (at most, at least, under, over, no more than) is a limit, kept together with its unit. A new budget replaces the previous one whether higher or lower, even when the shopper calls it good news. When the shopper only says they don't mind, have no preference, or anything works, put the property that was asked about in no_preference."""

REFEREE_GUIDE = """You are the last check before a shopping assistant places an order. The task scores 1 only when the ordered item satisfies every requirement below. Judge the item ONLY on the records shown, applying these reading rules:
- A requirement counts as met only when the item's own property for it states the value. Property keys are spelled in different ways (spaces, dots, underscores, prefixes) and one property may show up under several keys; any key naming that property counts. Values match on meaning: a single element of a comma list suffices, singular/plural and close synonyms count, case is irrelevant. A word that merely looks similar or a broader/different class does not count.
- The title or description counts only when the item carries no property key for that requirement. When its own property says something else, it fails even if the title states the value.
- A value under some other, related property does not count.
- Numbers: compare after converting units; limits include the edge; a range counts only when the entire range satisfies the limit.
- Exclusions and closed lists have to be honoured.
- It needs to be the requested kind of product; when listings carry a category label, it needs to be the category whose label matches the wording of the request as a whole, not a narrower or neighbouring one.
- When the item's own property for a requirement gives a different value, the requirement fails, even if the title hints otherwise.
- Price multiplied by quantity needs to be within the budget (equal is fine). When the records show it out of stock, it fails.
- A mere wish ("ideally", "would be nice") marked as not required is not a requirement.
When records conflict (for example on price or stock), the latest record counts.
Return ONE JSON object and nothing besides it: {"item": the item's id, "fits": true or false, "problems": [short strings, each naming a requirement that is not met or not shown]}. Set fits=false only for a requirement the records clearly fail, or one no record covers at all; do not fail an item because a stock or price record is missing."""

QUESTIONS_GUIDE = """You plan the questions a shopping assistant puts to a shopper. The shopper holds hidden needs: each one is a value of a single property of the product, disclosed only when a message asks about that exact property; only the first thing a message asks gets an answer, and questions on anything else get a polite "don't mind". Pick up to QUESTION_COUNT distinct properties, the most promising first:
0. first, anything listed below as MUST BE LEARNED FROM THE SHOPPER that has not yet been asked; ask it plainly. For the budget, ask the most they want to spend. For a choice between two wishes, name both values with their properties, for example "If one listing can't have both, would you rather have the <value> <property> or the <value> <property>?";
1. after that, properties the listings' details below really carry, favouring (a) a property the request's story hints at indirectly (who the item is for or who will wear or use it, how it must look in photos or to others, where or how it will be used), and (b) properties describing what the product is like whose values differ across the listings (who it is made for, pattern, shape, type or style variant, closure, length, function) over logistics (warranty, condition, pack size, variant or size lists, years). When CORE PROPERTIES are listed below (carried by most of the listings opened), pick only from them.
Never choose a property the request already settles (its stated requirements, the price, stock, quantity), never repeat a property already asked, never use two questions on near-duplicate properties (two kinds of style, two spellings of one property), and skip properties no listing carries. Name properties the way the listings name them. Phrase each question as a single short, plain, friendly sentence to the shopper that names that one property in the listings' own words for it and may offer one or two example values from the listings, for example "Do you have a preference for the <property>, such as <value> or <value>?". No instructions, no lists, no second question.
Return ONE JSON object and nothing besides it: {"questions": [{"property": "...", "question": "..."}, ...]}."""

RECHECK_GUIDE = """A shopping assistant is on the point of telling the shopper that nothing in the store fits. Test the listing records below against the requirements, applying these reading rules: a requirement is met when the listing's own property for it states the value (any key spelling, a single element of a comma list, singular/plural and close synonyms all count; a broader or different class does not); the title or description counts only when the listing has no key for that property; numbers are compared after unit conversion, limits include the edge; the price needs to be within the budget and the item must not appear out of stock.
Return ONE JSON object and nothing besides it: {"possible_fits": [ids of listings that seem to satisfy every requirement, best first, at most 5], "reason": "one sentence"}."""

_model_link = ProxyClient(timeout=70, max_retries=1, retry_delay=1)
_store_link = ProxyClient(timeout=95, max_retries=2, retry_delay=1)

def _trim(value: Any, limit: int, depth: int = 0) -> Any:
    """Shorten every string inside a nested structure to `limit` characters; anything nested too deep turns into None."""
    if depth <= 8:
        nxt = depth + 1
        if isinstance(value, str):
            if len(value) > limit:
                return value[:limit] + "..."
            return value
        if isinstance(value, dict):
            return dict((key, _trim(inner, limit, nxt)) for key, inner in value.items())
        if isinstance(value, list):
            return list(map(lambda inner: _trim(inner, limit, nxt), value))
        return value
    return None


_ID_FORM = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-:.]{3,63}$")


def _one_line(value: Any) -> str:
    """Compact single-line JSON of `value` (falling back to str), capped at RESULT_TEXT_CAP characters."""
    try:
        rendered = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        rendered = str(value)
    return rendered if len(rendered) <= RESULT_TEXT_CAP else rendered[:RESULT_TEXT_CAP] + "...(cut)"


def _qa_dict_or_none(raw: str) -> tuple[bool, dict[str, Any] | None]:
    """Parse `raw` as JSON: (parsed?, the object when it is a dict)."""
    try:
        decoded = json.loads(raw)
    except ValueError:
        return False, None
    return True, (decoded if isinstance(decoded, dict) else None)


def _json_in(text: str) -> dict[str, Any] | None:
    """Pull a JSON object out of a model reply, tolerating code fences and chatter around the braces."""
    if isinstance(text, str):
        body = text.strip()
        if body.startswith("```"):
            body = re.sub(r"^```[A-Za-z]*", "", body).strip()
            if body.endswith("```"):
                body = body[:-3].strip()
        parsed, found = _qa_dict_or_none(body)
        if parsed:
            return found
        opening, closing = body.find("{"), body.rfind("}")
        if opening >= 0 and closing > opening:
            return _qa_dict_or_none(body[opening : closing + 1])[1]
    return None


_qa_TYPE_TESTS = {
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "null": lambda v: v is None,
}


def _is_type(value: Any, kind: str) -> bool:
    """JSON-schema type test; an unknown type name accepts anything."""
    test = _qa_TYPE_TESTS.get(kind)
    return True if test is None else test(value)


def _qa_kinds(schema: dict[str, Any]) -> Any:
    declared = schema.get("type")
    return [declared] if isinstance(declared, str) else declared


def _qa_coerce_one(value: Any, kind: str) -> tuple[bool, Any]:
    """Try converting `value` into a single schema type; (converted?, result)."""
    numeric = isinstance(value, (int, float)) and not isinstance(value, bool)
    if kind == "integer":
        if isinstance(value, float) and value.is_integer():
            return True, int(value)
        if isinstance(value, str) and re.fullmatch(r"\s*-?\d+(\.0+)?\s*", value):
            return True, int(float(value))
    elif kind == "number":
        if isinstance(value, str) and re.fullmatch(r"\s*-?\d+(\.\d+)?\s*", value):
            as_float = float(value)
            return True, (int(as_float) if as_float.is_integer() else as_float)
    elif kind == "string":
        if numeric:
            return True, str(value)
    elif kind == "boolean":
        if isinstance(value, str) and value.lower() in ("true", "false"):
            return True, value.lower() == "true"
    return False, None


def _cast(value: Any, schema: dict[str, Any]) -> Any:
    """Nudge a model-supplied value toward the declared type (string digits to numbers, numbers to text, and so on)."""
    kinds = _qa_kinds(schema)
    if not kinds:
        return value
    if not any(_is_type(value, k) for k in kinds):
        for kind in kinds:
            try:
                ok, converted = _qa_coerce_one(value, kind)
            except (TypeError, ValueError):
                continue
            if ok:
                return converted
        return value
    whole_float = isinstance(value, float) and value.is_integer()
    return int(value) if "integer" in kinds and whole_float else value


def _qa_number_issues(value: Any, schema: dict[str, Any], path: str) -> list[str]:
    issues: list[str] = []
    if "minimum" in schema and value < schema["minimum"]:
        issues.append(f"{path}: is less than the lowest allowed value {schema['minimum']}")
    if "maximum" in schema and value > schema["maximum"]:
        issues.append(f"{path}: is more than the highest allowed value {schema['maximum']}")
    floor = schema.get("exclusiveMinimum")
    if isinstance(floor, (int, float)) and value <= floor:
        issues.append(f"{path}: has to be greater than {floor}")
    ceiling = schema.get("exclusiveMaximum")
    if isinstance(ceiling, (int, float)) and value >= ceiling:
        issues.append(f"{path}: has to be smaller than {ceiling}")
    return issues


def _validate(value: Any, schema: dict[str, Any], path: str) -> tuple[Any, list[str]]:
    """Coerce and check `value` against a tool-argument schema; gives back the repaired value and readable complaints."""
    if not isinstance(schema, dict):
        return value, []
    value = _cast(value, schema)
    kinds = _qa_kinds(schema)
    if kinds and not any(_is_type(value, k) for k in kinds):
        return value, [f"{path}: should be of type {'/'.join(kinds)}"]
    issues: list[str] = []
    if "enum" in schema:
        allowed = schema["enum"]
        if value not in allowed and isinstance(value, str):
            wanted = value.strip().lower()
            value = next((choice for choice in allowed if isinstance(choice, str) and choice.lower() == wanted), value)
        if value not in allowed:
            issues.append(f"{path}: has to be one of {allowed}")
    if "const" in schema and value != schema["const"]:
        issues.append(f"{path}: has to be exactly {schema['const']!r}")
    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            issues.append(f"{path}: needs {schema['minLength']} characters or more")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            value = value[: schema["maxLength"]]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        issues += _qa_number_issues(value, schema, path)
    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            issues.append(f"{path}: must hold {schema['minItems']} entries or more")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            issues.append(f"{path}: may hold no more than {schema['maxItems']} entries")
        element_schema = schema.get("items")
        if isinstance(element_schema, dict):
            repaired = []
            for position, element in enumerate(value):
                element, found = _validate(element, element_schema, f"{path}[{position}]")
                repaired.append(element)
                issues += found
            value = repaired
    if isinstance(value, dict):
        declared = schema.get("properties") or {}
        issues += [f"{path}.{key}: missing but required" for key in schema.get("required") or [] if key not in value]
        if schema.get("additionalProperties") is False:
            issues += [f"{path}.{key}: is not an argument this tool accepts" for key in list(value) if key not in declared]
        rebuilt = {}
        for key, element in value.items():
            if key in declared:
                element, found = _validate(element, declared[key], f"{path}.{key}")
                issues += found
            rebuilt[key] = element
        value = rebuilt
    return value, issues


def _gather_lines(value: Any, out: list[str], depth: int = 0) -> None:
    """Append sentence-like strings (four words or more) to `out`, looking into at most ten list entries per level."""
    if depth <= 4:
        if isinstance(value, list):
            for entry in value[:10]:
                _gather_lines(entry, out, depth + 1)
        elif isinstance(value, str) and len(value.split()) >= 4:
            out.append(value)


def _gather_ids(value: Any, out: set[str], depth: int = 0) -> None:
    """Add to `out` every id-shaped string held under a key mentioning 'id', walking dicts and lists."""
    if depth > 6:
        return
    below = depth + 1
    if isinstance(value, list):
        for child in value:
            _gather_ids(child, out, below)
        return
    if not isinstance(value, dict):
        return
    for key, entry in value.items():
        looks_like_id = isinstance(entry, str) and "id" in key.lower() and _ID_FORM.match(entry)
        if not looks_like_id:
            _gather_ids(entry, out, below)
            continue
        out.add(entry)


_SPELLINGS: dict[str, str] = {}
for _pair in "colour color / flavour flavor / grey gray / fibre fiber / metre meter / litre liter / centre center / jewellery jewelry / aluminium aluminum / tyre tire / armour armor / favourite favorite".split("/"):
    for _w in _pair.split():
        _SPELLINGS[_w] = _pair.split()[-1]
_AUDIENCES = [set(part.split()) for part in "women woman womens female females ladies lady / men man mens male males gents / girls girl / boys boy / kids kid children child".split("/")]
_STOPWORDS = set("a an the of and with must be is are has have having include includes including exactly only should that it its any made from".split())
_BIGGER_WORDS = "more|bigger|larger|higher|longer|wider|taller|heavier"
_SMALLER_WORDS = "less|smaller|fewer|lower|shorter|narrower|lighter"
_HEADLINE_KEY = re.compile(r"(?i)title|heading|name|option|variant")
_CATEGORY_KEY = re.compile(r"(?i)categor|aisle|shelf|department|section")
_NON_FACT_KEY = re.compile(r"(?i)price|cost|budget|stock|quantit|deliver|arriv|ship|categor|^(?:product|item|kind)(?: type)?$")


def _word_list(text: Any) -> list[str]:
    """Lower-case words of three letters or more, splitting camelCase."""
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(text or ""))
    return [w for w in re.findall(r"[a-z0-9]+", spaced.lower()) if len(w) > 2]


def _tokens(text: Any, join: bool) -> list[str]:
    """Normalised words for comparison: lowered, possessive 's removed, British spellings mapped to one form, and with
    `join` a number glued to the unit after it ('30 mm' reads as '30mm')."""
    source = "" if text is None else str(text)
    lowered = re.sub(r"'s\b", "", source.lower())
    if join:
        lowered = re.sub(r"(\d)\s+(?=[a-z%])", r"\1", lowered)
    words = re.findall(r"[a-z0-9]+", lowered)
    return [_SPELLINGS[w] if w in _SPELLINGS else w for w in words]


def _singular(word: str) -> str:
    """Crude plural stripping: berries -> berry, boxes -> box, shoes -> shoe; 'glass' stays."""
    size = len(word)
    if size > 3 and word.endswith("ies"):
        return word[:-3] + "y"
    if size > 4 and word.endswith(("ches", "shes", "xes", "sses")):
        return word[:-2]
    return word[:-1] if size > 3 and word.endswith("s") and not word.endswith("ss") else word


def _qa_shared_start(first: str, second: str) -> int:
    """How many leading characters two words have in common."""
    count = 0
    while count < min(len(first), len(second)) and first[count] == second[count]:
        count += 1
    return count


def _qa_one_slip(longer: str, shorter: str) -> bool:
    """True when `shorter` becomes `longer` through at most one changed or inserted character (walking left to right)."""
    same_size = len(longer) == len(shorter)
    a = b = slips = 0
    while a < len(longer) and b < len(shorter):
        if longer[a] != shorter[b]:
            slips += 1
            if same_size:
                b += 1
        else:
            b += 1
        a += 1
    return slips + (len(longer) - a) + (len(shorter) - b) <= 1


def _alike(one: str, other: str) -> bool:
    """Treat two words as the same when they differ only by number or word form (tunik/tunic, stripe/striped,
    rectangle/rectangular, women/female); a '-less' antonym or a different figure never counts."""
    one, other = _singular(one), _singular(other)
    if one == other:
        return True
    if any(one in group and other in group for group in _AUDIENCES):
        return True
    if re.search(r"\d", one + other):
        return False
    short, longer = sorted((one, other), key=len)
    gap = len(longer) - len(short)
    if longer.startswith(short):
        return len(short) >= 4 and gap <= 3
    if _qa_shared_start(one, other) >= max(5, len(short) - 2):
        return True
    return len(short) >= 5 and gap <= 1 and _qa_one_slip(longer, short)


def _says(value: str, text: str) -> bool:
    """The text carries the value when each meaningful word of the value matches some word of the text, or when the
    value squeezed of spaces appears inside the squeezed text ('particle board' within 'particleboard')."""
    for join in (True, False):
        needed = [w for w in _tokens(value, join) if w not in _STOPWORDS]
        if not needed:
            continue
        pool = _tokens(text, join)
        if all(any(_alike(w, h) for h in pool) for w in needed):
            return True
    squeezed = re.sub(r"[^a-z0-9]+", "", value.lower())
    if len(squeezed) < 8:
        return False
    return squeezed in re.sub(r"[^a-z0-9]+", "", str(text).lower())


def _label_words(label: str) -> list[str]:
    """Singular words of a fact label, skipping leading one- or two-letter fragments while more words follow."""
    parts = _tokens(label, False)
    skip = 0
    while len(parts) - skip > 1 and len(parts[skip]) <= 2:
        skip += 1
    return list(map(_singular, parts[skip:]))


def _key_names(label: str, prop: str, spare: set[str]) -> bool:
    """Does a fact label stand for the property? It must contain the property's words (words of the product itself
    may be missing) plus no more than two others, and must contain 'type' if the property does: 'top material' is a
    material label, 'system memory' is not a memory-type label."""
    label_set = set(_label_words(label))
    prop_set = set(_label_words(prop))
    essential = prop_set - {"type"}
    if essential and not essential <= label_set:
        essential = essential - spare
        prop_set = prop_set - spare
    if not essential or not essential <= label_set:
        return False
    needs_type_word = "type" in prop_set and "type" not in label_set
    if needs_type_word and label_set != essential:
        return False
    return len(label_set - prop_set) <= 2


def _limits_of(value: str) -> tuple[float | None, float | None] | None:
    """Read a numeric bound out of a requirement: '50 cm or longer' gives (50, None), 'no more than 750 g' gives
    (None, 750), '2-5 kg' gives (2, 5); None when no bound is stated."""
    phrase = value.lower().replace(",", "")
    figures = list(map(float, re.findall(r"\d+(?:\.\d+)?", phrase)))
    if not figures:
        return None
    first = figures[0]
    is_range = len(figures) >= 2 and re.search(r"between|\d\s*(?:to|-|and)\s*\d", phrase) and not re.search(r"up to|at (?:most|least)", phrase)
    if is_range:
        pair = figures[:2]
        return (min(pair), max(pair))
    checks = (
        (r"\bno(?:t)?\s+(?:%s|over|above)\b" % _BIGGER_WORDS, (None, first)),
        (r"\bno(?:t)?\s+(?:%s|under|below)\b" % _SMALLER_WORDS, (first, None)),
        (r"at least|or (?:%s|above|over)|minimum|\bmin\b|(?:%s) than|\babove\b|\bover\b|>=|\d\+" % (_BIGGER_WORDS, _BIGGER_WORDS), (first, None)),
        (r"at most|up to|or (?:%s|below|under)|maximum|\bmax\b|(?:%s) than|\bunder\b|\bbelow\b|within|<=" % (_SMALLER_WORDS, _SMALLER_WORDS), (None, first)),
    )
    return next((bound for pattern, bound in checks if re.search(pattern, phrase)), None)


def _numbers_in(text: str) -> list[tuple[float, str]]:
    """Every figure in the text with the (singular) unit written right after it, possibly empty."""
    cleaned = str(text).lower().replace(",", "")
    found = re.finditer(r"(?<![a-z0-9.])(\d+(?:\.\d+)?)\s*([a-z%\"]*)", cleaned)
    return [(float(hit.group(1)), _singular(hit.group(2))) for hit in found]


def _qa_fact_pairs(sheet: dict[str, Any]) -> list[tuple[str, str]]:
    """(label, text) for every plain filled-in value on the sheet and in its first nested facts dict."""
    nested: dict[str, Any] = {}
    for inner in sheet.values():
        if isinstance(inner, dict):
            nested = inner
            break
    pairs = []
    for label, raw in list(sheet.items()) + list(nested.items()):
        if isinstance(raw, bool) or not isinstance(raw, (str, int, float)):
            continue
        if re.fullmatch(r"(?i)\s*(?:n/?a|none|not specified|unknown|null|-|)\s*", str(raw)):
            continue
        pairs.append((str(label), str(raw)))
    return pairs


def _qa_bound_verdict(value: str, bound: tuple[float | None, float | None], texts: list[str], has_named: bool) -> str:
    low, high = bound
    wanted_unit = next((u for _, u in _numbers_in(value) if u), "")
    readings = [pair for t in texts for pair in _numbers_in(t)]
    for figure, _ in readings:
        if (low is None or figure >= low) and (high is None or figure <= high):
            return "yes"
    same_scale = [figure for figure, unit in readings if unit == wanted_unit or not unit or not wanted_unit]
    return "no" if same_scale and has_named else "unshown"


def _verdict_for(sheet: dict[str, Any], prop: str, value: str, spare: set[str]) -> tuple[str, str]:
    """Mirror of how the store judges one requirement (v3 line): only a fact labelled for the property counts, with the
    title or variant options standing in when no such fact exists; a labelled fact saying something else beats the
    title; an excluded value found there fails; a bound passes on a figure inside it and fails on a comparable figure
    outside it. Gives ('yes' | 'no' | 'unshown', the evidence that was read)."""
    pairs = _qa_fact_pairs(sheet)
    named = [(label, text) for label, text in pairs if not _HEADLINE_KEY.search(label) and _key_names(label, prop, spare)]
    if named:
        texts = [text for _, text in named]
    else:
        texts = [text for label, text in pairs if _HEADLINE_KEY.search(label) and not _CATEGORY_KEY.search(label)]
    evidence = "; ".join(f"{label}: {text[:60]}" for label, text in named[:2]) or "no fact named for it"
    bound = _limits_of(value)
    negated = re.match(r"^(?:not|no|without|anything but|anything except|except|excluding|avoid)\s+(.+)$", value)
    if bound is not None:
        return _qa_bound_verdict(value, bound, texts, bool(named)), evidence
    if negated:
        hit = any(_says(negated.group(1), t) for t in texts)
        return ("no" if hit else "yes"), evidence
    alternatives = [part for part in re.split(r"\s+or\s+|(?<=[a-z])\s*/\s*(?=[a-z])|,\s*(?!\d)", value) if part.strip()] or [value]
    for part in alternatives:
        if any(_says(part, t) for t in texts):
            return "yes", evidence
    if not named:
        return "unshown", evidence
    value_units = {u for _, u in _numbers_in(value) if u}
    if value_units:
        for t in texts:
            if any(u and u not in value_units for _, u in _numbers_in(t)):
                return "unshown", evidence
    return "no", evidence


def _price_of(record: dict[str, Any]) -> float | None:
    """First numeric price/cost field of a record, ignoring totals."""
    for label, amount in record.items():
        name = str(label)
        if not re.search(r"(?i)price|cost", name) or re.search(r"(?i)total", name):
            continue
        if isinstance(amount, (int, float)) and not isinstance(amount, bool):
            return float(amount)
    return None


def _stock_of(record: dict[str, Any]) -> bool | None:
    """Stock flag from the first stock/availability field holding a bool or a count."""
    for label, amount in record.items():
        if not re.search(r"(?i)stock|avail", str(label)):
            continue
        if isinstance(amount, bool):
            return amount
        if isinstance(amount, (int, float)):
            return amount > 0
    return None


def _category_of(record: dict[str, Any]) -> str:
    """The shelf/category text of a record, or an empty string."""
    for label, shelf in record.items():
        if _CATEGORY_KEY.search(str(label)) and isinstance(shelf, str) and shelf.strip():
            return shelf
    return ""


def _qd_crash_entry(err: Exception) -> dict[str, Any]:
    """Trace record for an exception escaping the episode."""
    return {"kind": "crash", "error": f"{type(err).__name__}: {err}"[:500]}




class Shopper:
    def __init__(self, problem_data: dict[str, Any]):
        self.ticks = 0
        environment = problem_data["environment"]
        self.link = dict(environment["binding"])
        self.view = environment["policy_view"]
        self.tool_specs = [t for t in self.view.get("tools") or [] if isinstance(t, dict) and isinstance(t.get("function"), dict)]
        self.arg_specs = {t["function"]["name"]: t["function"].get("parameters") or {"type": "object", "properties": {}} for t in self.tool_specs}
        self.step_cap = int(self.view.get("max_steps") or 30)
        self.steps_used = 0
        self.done = bool(self.view.get("finished"))
        self.out_of_funds = False
        self.notes_sent = 0
        self.note_allowance: int | None = None
        self.orders_tried = 0
        self.refused_times = 0
        self.order_vetted = False
        self.refusal_vetted = False
        self.basket_touched = False
        self.known_ids: set[str] = set()
        self.given: list[str] = []
        self.told: list[str] = []
        self.indifferent: list[str] = []
        self.released: list[str] = []
        self.budget: Any = None
        self.heard_lines: set[str] = set()
        self.basket: list[str] = []
        self.seen: list[Any] = []
        self.basket_view: Any = None
        self.cart_edit_step = 0
        self.basket_seen_step = -1
        self.basket_modes: dict[str, str] = {}
        self.adding_mode: str | None = None
        self.questions_log: list[str] = []
        self.planning = True
        self.rechecked = False
        self.count_warned: set[str] = set()
        self.reply_queued = ""
        self.needs_now: list[str] | None = None
        self.verdict_cache: dict[str, str] = {}
        self.needs_rev = 0
        self.cart_vetted: set[str] = set()
        self.chat_log: list[str] = []
        self.last_calls: list[str] = []
        self.role_map: dict[str, str] = {}
        self.reading_plan: dict[str, Any] = {}
        self.spent = 0.0
        self.closing_args: dict[str, dict[str, Any]] = {}
        self.fact_sheets: dict[str, dict[str, Any]] = {}
        self.search_count = 0
        self.search_watch = False
        self.listing_rows: dict[str, dict[str, Any]] = {}
        self.rule_flagged: set[str] = set()
        self.shown_fits = ""
        self.warned_once: set[str] = set()
        self.key_coverage: dict[str, dict[str, Any]] = {}
        self.log: list[dict[str, Any]] = []
        self.convo: list[dict[str, Any]] = []
        self.call_log: list[tuple[str, dict[str, Any], Any]] = []

    def seconds_left(self) -> float:
        """Run time still available, estimated from counted ticks rather than any clock."""
        used = self.ticks * TICK_LENGTH
        return RUN_ALLOWANCE - used

    @contextmanager
    def counting(self) -> Iterator[None]:
        """While the wrapped network call runs, a background thread adds one tick per TICK_LENGTH; one more tick is
        charged when it ends."""
        halt = threading.Event()

        def _qa_tick_loop() -> None:
            while True:
                if halt.wait(TICK_LENGTH):
                    break
                self.ticks += 1
        ticker = threading.Thread(target=_qa_tick_loop, daemon=True)
        ticker.start()
        try:
            yield
        finally:
            halt.set()
            ticker.join()
            self.ticks += 1

    def steps_remaining(self) -> int:
        """Store steps not yet spent."""
        return self.step_cap - self.steps_used

    def call_models(self, convo: list[dict[str, Any]], tool_specs: list[dict[str, Any]] | None, models: list[str], max_tokens: int) -> dict[str, Any] | None:
        """Ask each model in turn until one returns text or tool calls; None when all fail or time is nearly gone."""
        for model in models:
            remaining = self.seconds_left()
            if remaining < CLOSE_NOW_AT - 10:
                return None
            _model_link.timeout = int(max(8, min(75, remaining - CLOSE_NOW_AT + 15)))
            cached = model.startswith("anthropic/") and len(convo) > 2
            request: dict[str, Any] = {
                "model": model,
                "messages": self._qa_cache_marked(convo) if cached else convo,
                "temperature": 0,
                "max_tokens": max_tokens,
            }
            if tool_specs:
                request.update(tools=tool_specs, tool_choice="auto")
            with self.counting():
                reply = _model_link.post_verbose("/inference/chat/completions", request)
            payload = reply.data
            if isinstance(payload, dict) and payload.get("choices"):
                self.spent += self._priced(model, payload.get("usage") or {})
                answer = payload["choices"][0].get("message") or {}
                if answer.get("content") or answer.get("tool_calls"):
                    return answer
            failure = str(reply.error)[:300] if reply.error else "empty"
            self.log.append({"kind": "inference_failure", "model": model, "error": failure})
        return None

    @staticmethod
    def _priced(model: str, usage: dict[str, Any]) -> float:
        """The cost of one reply: the figure the reply carries, else an estimate from its token counts."""
        if isinstance(usage.get("cost"), (int, float)):
            return float(usage["cost"])
        fresh, cached, out = TOKEN_PRICES.get(model, TOKEN_PRICES[CHECK_MODEL])
        prompt = usage.get("prompt_tokens") or 0
        hit = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
        return (max(0, prompt - hit) * fresh + hit * cached + (usage.get("completion_tokens") or 0) * out) / 1e6

    def store_call(self, actions: list[dict[str, Any]]) -> dict[str, Any] | None:
        """Send one or several actions to the store as the next step, retrying up to three times on recoverable
        failures (a step-number clash resyncs the counter first)."""
        request: dict[str, Any] = dict(self.link)
        request["step"] = self.steps_used + 1
        if len(actions) != 1:
            request["actions"] = actions
        else:
            request["action"] = actions[0]
        attempts = 0
        while attempts < 3:
            attempts += 1
            remaining = self.seconds_left()
            if remaining < 2:
                return None
            _store_link.timeout = int(max(3, min(95, remaining - 1)))
            with self.counting():
                reply = _store_link.post_verbose("/environment/call", request)
            if isinstance(reply.data, dict):
                return self._qa_absorb_reply(reply.data, len(actions))
            failure = reply.error or {}
            code = failure.get("status")
            detail = str(failure.get("body") or "")
            self.log.append({"kind": "env_failure", "status": code, "body": detail[:300]})
            if code == 402:
                self.out_of_funds = True
                return None
            if code == 409:
                if "finished" in detail:
                    self.done = True
                    return None
                clash = re.search(r"expected (\d+)", detail)
                if not clash:
                    return None
                self.steps_used = int(clash.group(1)) - 1
                request["step"] = self.steps_used + 1
                continue
            if code in (403, 422, 500, 503, 504):
                return None
        return None

    def brief_once(self) -> None:
        """One reading of the request by the briefing model; keeps the roles, closing modes, budgets and stated needs."""
        catalogue = "\n".join(
            "- {}: {} | arguments: {}".format(fn["name"], fn.get("description", ""), _one_line(fn.get("parameters") or {}))
            for fn in (entry["function"] for entry in self.tool_specs)
        )
        fixed_part = BRIEFING_GUIDE + "\n\nTOOLS:\n" + catalogue
        live_part = "REQUEST:\n" + str(self.view.get("query") or "")
        first_obs = self.view.get("opening")
        if first_obs is not None:
            live_part += "\n\nOPENING OBSERVATION:\n" + _one_line(first_obs)
        packet = [{"type": "text", "text": fixed_part, "cache_control": {"type": "ephemeral"}}, {"type": "text", "text": live_part}]
        answer = self.call_models([{"role": "user", "content": packet}], None, [BRIEFING_MODEL, RESERVE_MODEL], 1500)
        plan = _json_in((answer or {}).get("content") or "") or {}
        self.reading_plan = plan
        raw_roles = plan.get("roles")
        mapped: dict[str, Any] = {}
        if isinstance(raw_roles, dict):
            for role, tool in raw_roles.items():
                if isinstance(tool, str) and tool in self.arg_specs:
                    mapped[role] = tool
        self.role_map = mapped
        finishers = plan.get("finish_args")
        if isinstance(finishers, dict):
            kept: dict[str, Any] = {}
            for mode, extra in finishers.items():
                if mode in ("order", "decline") and isinstance(extra, dict):
                    kept[mode] = extra
            self.closing_args = kept
        self._roles_by_shape()
        allowance = plan.get("message_budget")
        if self._qb_is_count(allowance) and allowance >= 0:
            self.note_allowance = int(allowance)
        self.given = [str(need) for need in (plan.get("requirements") or []) if need]
        self.budget = plan.get("price_cap")
        if plan and self.budget is None and self.role_map.get("message"):
            earlier = [str(topic) for topic in (plan.get("ask_first") or []) if topic]
            plan["ask_first"] = ["how much the shopper is willing to pay at most, because the request gives no price ceiling (leave this out if it already appears below)", *earlier]
        cap = plan.get("step_budget")
        if self._qb_is_count(cap) and 0 < cap < self.step_cap:
            self.step_cap = int(cap)

    # ---- tool roles read from the tools' own shapes (a second opinion on the briefing's role map)

    @staticmethod
    def _arg_kind(spec: Any) -> str:
        """choice (enum/const/boolean), count (integer), text (free string), lines (a list of objects holding a text and
        an integer, or such a list inside an object), texts (a list of strings), else ''."""
        if not isinstance(spec, dict):
            return ""
        kind = spec.get("type")
        if spec.get("enum") or "const" in spec or kind == "boolean":
            return "choice"
        if kind == "integer":
            return "count"
        if kind == "string":
            return "text"
        if kind == "array" and isinstance(spec.get("items"), dict):
            inner = spec["items"]
            if inner.get("type") == "string":
                return "texts"
            parts = {Shopper._arg_kind(v) for v in (inner.get("properties") or {}).values()}
            return "lines" if {"text", "count"} <= parts else ""
        if kind == "object":
            parts = {Shopper._arg_kind(v) for v in (spec.get("properties") or {}).values()}
            return "lines" if "lines" in parts else ""
        return ""

    def _tool_shapes(self) -> dict[str, dict[str, Any]]:
        shapes = {}
        for fn in (entry["function"] for entry in self.tool_specs):
            params = fn.get("parameters") or {}
            props = params.get("properties") if isinstance(params.get("properties"), dict) else {}
            shapes[fn["name"]] = {"kinds": {k: self._arg_kind(v) for k, v in props.items()}, "props": props,
                                  "must": set(params.get("required") or []) & set(props),
                                  "said": str(fn.get("description") or ""),
                                  "words": set(_word_list(fn["name"].replace("_", " ") + " " + str(fn.get("description") or "")))}
        return shapes

    def _roles_by_shape(self) -> None:
        """Fill or correct the cart, finishing and message roles from the tools' argument shapes: the cart tool takes
        listing lines (or one required id text beside a count); a tool whose only required argument is a two-value choice
        ends the task both ways, the value whose stretch of the description shares more words with the cart tool
        ordering; two argument-free tools sharing a description sentence no other tool has are the two finishing tools,
        the one sharing more cart words ordering; the message tool takes far longer text than a listing id. A model
        reading of the right shape stands; only missing or misshapen roles are replaced."""
        shapes = self._tool_shapes()
        if not shapes:
            return
        changed: dict[str, Any] = {}
        carts = [n for n, sh in shapes.items() if "lines" in sh["kinds"].values()
                 or (sum(1 for a, k in sh["kinds"].items() if k == "text" and a in sh["must"]) == 1 and "count" in sh["kinds"].values())]
        cart = self.role_map.get("cart") if self.role_map.get("cart") in carts else (carts[0] if len(carts) == 1 else None)
        if cart and self.role_map.get("cart") != cart:
            changed["cart"] = cart
            self.role_map["cart"] = cart
        cart_words = shapes[cart]["words"] if cart else set()
        common = {w for w in set().union(*(sh["words"] for sh in shapes.values())) if 2 * sum(w in sh["words"] for sh in shapes.values()) > len(shapes)}
        marks = cart_words - common
        both = []
        for name, sh in shapes.items():
            picks = [a for a in sh["must"] if sh["kinds"].get(a) == "choice"]
            rest = [k for a, k in sh["kinds"].items() if a not in picks]
            values = sh["props"][picks[0]].get("enum") if len(picks) == 1 else None
            if len(sh["must"]) == 1 and picks and isinstance(values, list) and len(values) == 2 and all(k == "text" for k in rest):
                side = self._choice_side(sh["said"], [str(v) for v in values], marks)
                if side:
                    both.append((name, picks[0], side))
        if len(both) == 1:
            name, arg, (buy, stop) = both[0]
            for job, value in (("order", buy), ("decline", stop)):
                if self.role_map.get(job) != name or (self.closing_args.get(job) or {}).get(arg) != value:
                    changed[job] = {"tool": name, arg: value}
                self.role_map[job] = name
                self.closing_args[job] = {arg: value}
        elif not (self.role_map.get("order") and self.role_map.get("decline")):
            bare = [n for n, sh in shapes.items() if not sh["must"] and set(sh["kinds"].values()) <= {"text"} and n != cart]
            pairs = []
            for i, one in enumerate(bare):
                for other in bare[i + 1:]:
                    mine = {frozenset(_word_list(x)) for x in re.split(r"[.!?]+", shapes[one]["said"]) if len(_word_list(x)) > 1}
                    theirs = {frozenset(_word_list(x)) for x in re.split(r"[.!?]+", shapes[other]["said"]) if len(_word_list(x)) > 1}
                    if mine & theirs:
                        pairs.append((one, other))
            if len(pairs) == 1:
                one, other = pairs[0]
                score = lambda n: len(shapes[n]["words"] & marks)
                if score(one) != score(other):
                    buy, stop = (one, other) if score(one) > score(other) else (other, one)
                    for job, tool in (("order", buy), ("decline", stop)):
                        if not self.role_map.get(job):
                            self.role_map[job] = tool
                            changed[job] = tool
        if not self.role_map.get("message"):
            id_room = next((sh["props"][a].get("maxLength") for n, sh in shapes.items() if n == cart for a, k in sh["kinds"].items() if k == "text"), None)
            wordy = [n for n, sh in shapes.items() if len(sh["kinds"]) == 1 and "text" in sh["kinds"].values() and sh["must"]
                     and isinstance(id_room, int) and (next(iter(sh["props"].values())).get("maxLength") or 10 ** 6) > 2 * id_room]
            if len(wordy) == 1:
                self.role_map["message"] = wordy[0]
                changed["message"] = wordy[0]
        self.log.append({"kind": "roles_by_shape", "changed": changed, "roles": dict(self.role_map)})

    @staticmethod
    def _choice_side(said: str, values: list[str], marks: set[str]) -> tuple[str, str] | None:
        """Of two choice values the description names: (the one whose stretch of text, up to the next value, shares more
        of the cart tool's own words, the other); None when the description does not tell them apart."""
        low = said.lower()
        spots = []
        for value in values:
            hit = re.search(r"\b%s\b" % re.escape(value.lower()), low)
            if not hit:
                return None
            spots.append((hit.start(), hit.end(), value))
        spots.sort()
        reach = {}
        for i, (_, end, value) in enumerate(spots):
            stop = spots[i + 1][0] if i + 1 < len(spots) else len(low)
            reach[value] = len(set(_word_list(low[end:stop])) & marks)
        first, second = values
        if reach[first] == reach[second]:
            return None
        return (first, second) if reach[first] > reach[second] else (second, first)

    def briefing_note(self) -> str:
        """The briefing model's reading, as a bullet list for the loop's opening message ('' when there is none)."""
        plan = self.reading_plan
        if not plan:
            return ""
        out = ["Initial reading of the request (verify it against the request text):"]
        if plan.get("product"):
            out.append("- product: %s" % (plan.get("product"),))
        out.append("- quantity: %s" % (plan.get("quantity") or 1,))
        out.extend("- firm requirement from the request: %s" % (need,) for need in plan.get("requirements") or [])
        if plan.get("price_cap") is not None:
            out.append(("- total price ceiling: %s %s" % (plan.get("price_cap"), plan.get("currency") or "")).rstrip())
        if plan.get("in_stock_only"):
            out.append("- only listings that are in stock / available now")
        rank = plan.get("priority")
        if rank not in ("cheapest", "best"):
            out.append("- the request sets no cheapest/best rule: every item meeting all requirements inside the budget is acceptable, so spend no steps chasing a lower price (unless the request says so itself)")
        else:
            extra = plan.get("priority_detail")
            out.append("- pick the %s%s of all the items that fit" % (rank, " (%s)" % (extra,) if extra else ""))
        out.extend("- a wish, NOT a requirement (use only to break a tie): %s" % (wish,) for wish in plan.get("soft_wants") or [])
        out.extend("- has to be asked of the shopper first: %s" % (topic,) for topic in plan.get("ask_first") or [])
        out.extend("- rule: %s" % (rule,) for rule in plan.get("rules") or [])
        if self.note_allowance is not None:
            out.append("- messages you may send the shopper: %s" % (self.note_allowance,))
        out.append("- step allowance: %s" % (self.step_cap,))
        hinted = [name for name in plan.get("hint_properties") or [] if isinstance(name, str)]
        if hinted:
            out.append("- properties the story may point to (ask about them first, a single one per message): " + ", ".join(hinted))
        return "\n".join(out)

    def status_text(self) -> str:
        """The running status line appended to tool results: steps, messages, time, current needs and the end-game nudge."""
        remaining = self.steps_remaining()
        sent = f"messages sent {self.notes_sent}" + (f" of {self.note_allowance}" if self.note_allowance is not None else "")
        clock_left = int(max(0, self.seconds_left() - CLOSE_NOW_AT))
        body = f"[status] steps used {self.steps_used} of {self.step_cap}, {remaining} left; {sent}; about {clock_left} s left."
        body += "\n[requirements now] " + self.needs_book()
        hints = self.story_hints()
        if hints:
            listed = "; ".join(p if not v else f"{p} (probably {v})" for p, v in hints)
            body += f"\n[story hints] {listed}: put these questions first, then favour the fit that agrees; a hint is never grounds to reject an item or to decline."
        body += self.fit_summary()
        if remaining <= 2:
            return body + " FINAL STEPS: if the cart holds a fit, order it now; decline only if no listing fits at all."
        if remaining <= 5 or self.seconds_left() < WIND_DOWN_AT:
            return body + " WRAP UP NOW: reduce the cart to a single verified line, check its stock again, then order (or decline when nothing fits)."
        if self.orders_tried == 0 and remaining <= 14:
            return body + " Few steps remain: in the next few steps cart your best verified fit and order it; searching and opening details stop once 7 steps are left, and repairing a refused order takes about 4 steps."
        return body

    def needs_book(self) -> str:
        """The requirements as they stand, plus budget, quantity, dropped needs and don't-care properties, in one line."""
        if self.needs_now is None:
            entries = [f"{need} (stated)" for need in self.given if need not in self.released]
            entries.extend(f"{need} (revealed by the shopper)" for need in self.told if need not in self.released)
        else:
            entries = list(self.needs_now)
        pieces = ["; ".join(entries) if entries else "see the request"]
        if self.budget is not None:
            pieces.append(f"total price at most {self.budget} {self.reading_plan.get('currency') or ''}".rstrip())
        pieces.append(f"quantity {self.reading_plan.get('quantity') or 1}")
        if self.released:
            pieces.append("no longer required: " + "; ".join(self.released))
        if self.indifferent:
            pieces.append("shopper does not care about: " + ", ".join(self.indifferent))
        return "; ".join(pieces)

    def read_intent(self, name: str, args: dict[str, Any]) -> str | None:
        """Classify a finishing call as 'order' or 'decline' (None for any other tool); a shared finishing tool is
        told apart by the closing arguments the briefing found, else by its first enum value."""
        buy_tool = self.role_map.get("order")
        quit_tool = self.role_map.get("decline")
        if name != buy_tool and name != quit_tool:
            return None
        if buy_tool != quit_tool:
            return "decline" if name != buy_tool else "order"
        for mode in ("decline", "order"):
            pattern = self.closing_args.get(mode) or {}
            if pattern and not any(args.get(key) != wanted for key, wanted in pattern.items()):
                return mode
        fields = (self.arg_specs.get(name) or {}).get("properties") or {}
        chosen = next(((field, spec) for field, spec in fields.items() if isinstance(spec, dict) and spec.get("enum") and args.get(field) is not None), None)
        if chosen is None:
            return "order"
        return "order" if args.get(chosen[0]) == chosen[1]["enum"][0] else "decline"

    def note_down(self, name: str, observation: Any) -> None:
        """Remember each opened listing (any record carrying a facts block) and the rows of result lists; a list that
        mostly lacks facts counts as one search."""
        if not isinstance(observation, dict):
            return
        body = observation
        pair = len(observation) == 2
        for part in observation.values():
            if pair and isinstance(part, dict):
                body = part
        if not isinstance(body, dict):
            return
        facts = next(filter(self._qb_flat, body.values()), None)
        key = self._qb_ident(body)
        if key and facts is not None:
            self.fact_sheets[key] = self._qb_compact(body)
        rows = next((part for part in body.values() if isinstance(part, list) and part and all(isinstance(x, dict) for x in part)), None)
        if rows is None or name == self.role_map.get("cart_view"):
            return
        with_facts = 0
        for row in rows:
            row_id = self._qb_ident(row)
            if not row_id:
                continue
            self.listing_rows[row_id] = {
                k: v for k, v in row.items()
                if isinstance(v, (str, int, float, bool)) and (not isinstance(v, str) or len(v) <= 160)
            }
            if any(self._qb_flat(v) for v in row.values()):
                self.fact_sheets[row_id] = self._qb_compact(row)
                with_facts += 1
        if 2 * with_facts < len(rows):
            self.search_count += 1

    def sheets_with(self, need: str) -> tuple[list[str], list[str]]:
        """Split opened listings by a 'property: value' need: those whose own fact for that property (or, lacking such a
        fact, whose text) names the value, and those whose fact for it names something else."""
        prop, _, value = need.partition(":")
        prop_words = [w for w in self._qb_words(prop.lower()) if w != "type"]
        value_words = self._qb_words(value.lower())
        ranged = bool(re.search(r"\d", value)) and bool(re.search(r"(?i)most|least|under|over|up to|between", value))
        if ranged or not prop_words or not value_words:
            return [], []
        hits: list[str] = []
        misses: list[str] = []
        for ident, record in self.fact_sheets.items():
            block = next((v for v in record.values() if isinstance(v, dict)), {})
            matching = []
            for fact_key in block:
                key_words = self._qb_words(str(fact_key).lower().replace("_", " "))
                if all(w in key_words for w in prop_words):
                    matching.append(fact_key)
            if matching:
                candidates = [str(block[k]).lower() for k in matching]
            else:
                candidates = [" ".join(str(v) for v in record.values() if isinstance(v, str)).lower()]
            if any(set(value_words) <= set(self._qb_words(c)) for c in candidates):
                hits.append(ident)
            elif matching:
                misses.append(ident)
        return hits, misses

    def story_hints(self) -> list[tuple[str, str]]:
        """Up to four (property, likely value) pairs the request's story hints at without stating, minus properties already
        settled as requirements or don't-cares. They steer questions and tie-breaks only, never rejection (hunting them as
        if required cost the story pack's pass 6b five tasks)."""
        settled_sets = [set(_label_words(s)) for s in [p for p, _ in self.needs_text()] + list(self.indifferent)]
        picked: list[tuple[str, str]] = []
        seen_props: set[str] = set()
        for entry in self.reading_plan.get("hints") or []:
            if isinstance(entry, dict):
                prop = str(entry.get("property")).strip().lower()
                likely = str(entry.get("likely") or "").strip().lower()
            else:
                prop = str(entry or "").strip().lower()
                likely = ""
            core = set(_label_words(prop)) - {"type"}
            if not core:
                continue
            covered = False
            for words in settled_sets:
                trimmed = words - {"type"}
                if core <= words or (trimmed and trimmed <= core):
                    covered = True
                    break
            if covered or prop in seen_props:
                continue
            seen_props.add(prop)
            picked.append((prop, likely))
        return picked[:4]

    def favoured(self, fits: list[tuple[str, float | None]]) -> str:
        """Pick among proven fits: the first (cheapest) whose facts confirm a hinted value, unless a cheapest/best rule
        applies, in which case simply the first."""
        cheapest = fits[0][0]
        if self.reading_plan.get("priority") in ("cheapest", "best"):
            return cheapest                                          # an explicit ranking rule beats every hint
        product = self._qb_product()
        for candidate, _ in fits:
            sheet = self.fact_sheets.get(candidate) or {}
            for prop, likely in self.story_hints():
                if likely and _verdict_for(sheet, prop, likely, product)[0] == "yes":
                    return candidate
        return cheapest

    def cart_call_args(self, ident: str) -> dict[str, Any]:
        """Build the cart tool's arguments for adding one listing at the wanted quantity, guided only by its schema."""
        fields = (self.arg_specs.get(self.role_map.get("cart") or "") or {}).get("properties") or {}
        wanted = self.reading_plan.get("quantity")
        if not (isinstance(wanted, int) and not isinstance(wanted, bool) and wanted > 0):
            wanted = 1
        built: dict[str, Any] = {}
        id_slot_used = False
        for field, spec in fields.items():
            if not isinstance(spec, dict):
                continue
            declared = spec.get("type")
            as_list = declared if isinstance(declared, list) else []
            if spec.get("enum") and "enum" in spec:
                built[field] = self._qb_add_choice([str(v) for v in spec["enum"]])
            elif declared in ("integer", "number") or "integer" in as_list or "number" in as_list:
                built[field] = max(wanted, spec.get("minimum", 1))
            elif not id_slot_used and (declared == "string" or "string" in as_list):
                built[field] = ident
                id_slot_used = True
        return built

    def buy_instead(self) -> tuple[str, dict[str, Any], str] | None:
        """Turn a late decline into a purchase: declining while a fit exists earns 0 (suite 160 race 1: 0 of 381 declines
        scored, 1,199 orders did). Once the fits were named and the loop still declines, or steps run short, order the cart
        if it already holds the fit, otherwise cart the preferred fit."""
        fits = self.confirmed()
        why = "whose own facts state every current requirement"
        late = self.refusal_vetted or f"decline|{self.needs_rev}" in self.rule_flagged or self.steps_remaining() <= 6
        if late and not fits:
            fits = self.likely_fits()                                   # no proof anywhere: a listing that contradicts nothing still beats declining
            why = "whose own facts contradict no current requirement (no opened listing states all of them, and declining earns nothing)"
        order_tool = self.role_map.get("order")
        cart_tool = self.role_map.get("cart")
        blocked = not fits or not order_tool or self.steps_remaining() <= 1 or self.seconds_left() < CLOSE_NOW_AT
        if blocked or not late:
            return None
        pick = self.favoured(fits)
        cost = next((p for i, p in fits if i == pick), None)
        label = pick if cost is None else f"{pick} ({cost:g})"
        if self.basket == [pick]:
            return (order_tool, self.bland_args(order_tool, "Order the cart.", "order"), f"Your decline was held back: declining earns 0 when a fitting listing is available, and the cart already has {label}, {why}, inside the budget and not marked out of stock. The order went out in its place.")
        if not cart_tool or self.steps_remaining() <= 2:
            return None
        return (cart_tool, self.cart_call_args(pick), f"Your decline was held back: declining earns 0 when a fitting listing is available, and {label} is the opened listing {why}, inside the budget and not marked out of stock. It has been added to the cart in its place: drop any other line, check its stock and price again, and order it.")

    def likely_fits(self) -> list[tuple[str, float | None]]:
        """Opened listings on the main shelf, within budget and not shown sold out, where no current requirement reads
        'no' (each is yes or unshown); most proven requirements first, then by price. Only for choosing between such a
        listing and a decline, since in suite 160 race 1 no decline scored (0 of 381)."""
        pairs = self.needs_text()
        product = self._qb_product()
        shelf = self.main_category()
        ranked = []
        for ident, sheet in self.fact_sheets.items():
            if not self._qb_on_shelf(sheet, shelf):
                continue
            verdicts = [_verdict_for(sheet, prop, value, product)[0] for prop, value in pairs]
            if "no" not in verdicts:
                ranked.append((ident, self.newest_price(ident), sum(1 for v in verdicts if v == "yes")))
        ranked.sort(key=lambda entry: (-entry[2], entry[1] is None, entry[1] or 0.0))
        return [(ident, price) for ident, price, _ in ranked]

    def needs_text(self) -> list[tuple[str, str]]:
        """Current requirements reduced to (property, value) pairs that the store's fact rule can check."""
        if self.needs_now is None:
            source = [need for need in self.given + self.told if need not in self.released]
        else:
            source = list(self.needs_now)
        pairs = []
        for entry in source:
            left, colon, right = str(entry).partition(":")
            left = left.strip().lower()
            right = re.sub(r"\s*\([^)]*\)", "", right.strip().lower()).strip(" .")
            usable = colon and left and right
            if usable and _NON_FACT_KEY.search(left) is None and re.search(r"(?i)\bbrand\b", left) is None:
                pairs.append((left, right))
        return pairs

    def within_cap(self, record: dict[str, Any]) -> bool:
        """Affordable at the wanted quantity (when a numeric cap exists and a price is known) and not shown out of stock."""
        unit = _price_of(record)
        count = self.reading_plan.get("quantity")
        if not (isinstance(count, int) and not isinstance(count, bool) and count > 0):
            count = 1
        ceiling = self.budget
        over = self._qb_is_count(ceiling) and unit is not None and unit * count > ceiling + 1e-9
        return not over and _stock_of(record) is not False

    def confirmed(self) -> list[tuple[str, float | None]]:
        """Opened listings on the main shelf, within budget and not shown sold out, whose facts state every current
        requirement by the store's rule; cheapest first (the ranked fits of the v3 line)."""
        pairs = self.needs_text()
        if not pairs:
            return []
        product = self._qb_product()
        shelf = self.main_category()
        proven = [
            (ident, self.newest_price(ident))
            for ident, sheet in self.fact_sheets.items()
            if self._qb_on_shelf(sheet, shelf) and all(_verdict_for(sheet, prop, value, product)[0] == "yes" for prop, value in pairs)
        ]
        proven.sort(key=lambda entry: (entry[1] is None, entry[1] or 0.0))
        return proven

    def main_category(self) -> str:
        """The shelf most opened listings sit on, ranked first by word overlap with the product name, then by count."""
        product = self._qb_product()
        tally: dict[str, int] = {}
        for sheet in self.fact_sheets.values():
            shelf = _category_of(sheet)
            if shelf:
                tally[shelf] = tally.get(shelf, 0) + 1
        if not tally:
            return ""
        return max(tally, key=lambda shelf: (len(product & set(map(_singular, _tokens(shelf, False)))), tally[shelf]))

    def newest_price(self, item: str) -> float | None:
        """The latest price seen for the item in any record (cart or bag views after a price change count too)."""
        for record in self.seen[::-1]:
            price = self._qb_price_in(record, item, 0)
            if price is not None:
                return price
        return _price_of(self.fact_sheets.get(item) or self.listing_rows.get(item) or {})

    def rule_review(self, item: str, adding: bool) -> str:
        """Check a listing against the store's own fact sheet right before it is carted or ordered. A fact recorded for a
        property with a different value outweighs any wording in the title (the v3 line reads facts before committing)."""
        own_sheet = self.fact_sheets.get(item)
        if own_sheet is not None:
            return self._qc_fact_breaches(item, own_sheet)
        must_open = adding and self.fact_sheets and self.role_map.get("details") and self.steps_remaining() > 8
        return (f"{item} has not had its details opened yet; the store judges each listing on its own facts, and only "
                f"the details reveal them. Open the details before going on.") if must_open else ""

    def cheaper_hint(self, item: str) -> str:
        """For a cheapest-first request, just before ordering (the v3 line searches for a cheaper fit in code): name the
        opened fits priced under the pick, then the unopened, cheaper listings from the pick's shelf."""
        if self.reading_plan.get("priority") == "cheapest":
            pick_cost = self.newest_price(item)
            if pick_cost is not None:
                pieces = self._qc_cheaper_pieces(item, pick_cost)
                if pieces:
                    return "This request wants the cheapest item that fits: " + "; ".join(pieces) + ". Place the order for the lowest-priced one that fits."
        return ""

    def order_hint(self, item: str) -> str:
        """One pre-order remark per item and requirement revision: a fact clash first, else a cheaper-fit nudge."""
        if item:
            stamp = f"{item}|{self.needs_rev}|order"
            if stamp not in self.rule_flagged:
                self.rule_flagged.add(stamp)
                return self.rule_review(item, False) or self.cheaper_hint(item)
        return ""

    def refusal_hint(self) -> str:
        """Push back once per requirement revision against a decline while proven fits or unexplored leads remain."""
        stamp = f"decline|{self.needs_rev}"
        too_late = stamp in self.rule_flagged or self.steps_remaining() <= 3 or self.seconds_left() < WIND_DOWN_AT
        if too_late:
            return ""
        proven = self.confirmed()
        if proven:
            self.rule_flagged.add(stamp)
            shown = ", ".join(i if p is None else f"{i} ({p:g})" for i, p in proven[:5])
            return ("Going by the store's own facts, these opened listings state every current requirement, fit the budget and are not marked out of stock: "
                    + shown + ". Re-check the strongest one (details, stock, price), put it in the cart and order it; decline only when every one of them really fails.")
        leads = self._qc_decline_leads() if self.steps_remaining() > 6 else []
        if not leads:
            return ""
        self.rule_flagged.add(stamp)
        return ("No opened listing states every current requirement so far, yet before you decline: " + "; ".join(leads)
                + ". Decline only after each of these has failed.")

    def fit_summary(self) -> str:
        """Report the proven fits to the loop each time that set changes (the v3 line keeps a ranked list plus a backup)."""
        proven = self.confirmed()
        roster = ", ".join(i if p is None else f"{i} ({p:g})" for i, p in proven[:5])
        previous = self.shown_fits
        if roster == previous:
            return ""
        self.shown_fits = roster
        if not proven:
            if not previous:
                return ""
            return "\n[fits by the store's facts] at the moment no opened listing states every current requirement in its own facts."
        parts = ["\n[fits by the store's facts] opened listings whose own facts state every current requirement, inside the budget and in stock, cheapest first: " + roster + "."]
        if len(proven) == 1:
            if self.orders_tried == 0:
                parts.append(" Just one so far: a second fit would serve as the fallback should this one sell out or change price.")
        elif self.story_hints():
            liked = self.favoured(proven)
            if liked != proven[0][0]:
                parts.append(f" {liked} lines up with the story's hint: favour it unless the shopper's own words or a cheapest rule point elsewhere.")
        return "".join(parts)

    def main_labels(self) -> list[str]:
        """Core labels worth asking about: carried by half or more of the opened listings (never fewer than two), not yet
        settled, asked or ignored, and not logistics; word-valued ones come before figure-valued ones (race 1 hid 33 needs,
        29 of them with word values and 32 of them such labels; Leen, Clockwerk and L.MODRIC aimed 77-78% of their
        questions there, enjoy only 46%)."""
        tally: dict[str, int] = {}
        numeric: dict[str, int] = {}
        spellings: dict[str, dict[str, int]] = {}
        blank = re.compile(r"(?i)n/?a|none|not specified|unknown|null|-")
        logistics = re.compile(r"(?i)option|condition|warrant|model|title|categor|brand|price|stock|seller|sku|origin|series|assembl|care|instruction|dimension")

        def stem(label: str) -> str:
            return " ".join(w for w in label.split() if w != "type")

        for sheet in self.fact_sheets.values():
            facts: dict[Any, Any] = {}
            for candidate in sheet.values():
                if isinstance(candidate, dict):
                    facts = candidate
                    break
            for key, val in facts.items():
                if re.search(r"\d", str(val)):
                    numbered = " ".join(w for w in _label_words(str(key)) if w != "type")
                    numeric[numbered] = numeric.get(numbered, 0) + 1
            present: set[str] = set()
            for key, val in facts.items():
                cleaned = str(val).strip()
                if cleaned and not blank.fullmatch(cleaned):
                    present.add(" ".join(_label_words(str(key))))
            for root in {stem(label) for label in present}:
                if root and not logistics.search(root):
                    tally[root] = tally.get(root, 0) + 1
            for label in present:
                bucket = spellings.setdefault(stem(label), {})
                bucket[label] = bucket.get(label, 0) + 1
        threshold = max(2, (len(self.fact_sheets) + 1) // 2)
        covered = [prop for prop, _ in self.needs_text()]
        covered += [entry[2:].split(": asked", 1)[0] for entry in self.questions_log]
        covered += list(self.indifferent)
        picked: list[str] = []
        for root in sorted(tally, key=lambda r: (numeric.get(r, 0) * 2 > tally[r], -tally[r])):
            root_words = set(root.split())
            if tally[root] >= threshold and root_words and not self._qc_label_taken(root_words, covered):
                picked.append(max(spellings[root], key=lambda form: (spellings[root][form], "type" in form.split())))
        return picked[:8]

    def keep_main(self, planned: list[dict[str, Any]], urgent: bool) -> list[dict[str, Any]]:
        """Planned questions that already target a core label stay as written; any other question is swapped for the next
        unused core label (a must-learn-first question, such as a missing budget, holds its slot)."""
        core = self.main_labels()
        if not core:
            return planned
        hint_sets = [set(_label_words(p)) - {"type"} for p, _ in self.story_hints()]
        word_sets = [set(_label_words(str(q.get("property") or "") + " " + str(q.get("question") or ""))) for q in planned]
        claimed: list[str | None] = []
        for pos, words in enumerate(word_sets):
            if urgent and pos == 0:
                claimed.append(None)
                continue
            claimed.append(next((c for c in core if c not in claimed and set(c.split()) - {"type"} <= words), None))
        kept: list[dict[str, Any]] = []
        for pos, question in enumerate(planned):
            pinned = (urgent and pos == 0) or any(h and h <= word_sets[pos] for h in hint_sets)
            if not pinned and claimed[pos] is None:
                spare = next((c for c in core if c not in claimed), None)
                if spare is None:
                    continue
                claimed[pos] = spare
                question = {"property": spare, "question": f"Do you have any preference for the {spare}?"}
            kept.append(question)
        return kept

    def unread_picks(self) -> list[str]:
        """Affordable, never-opened listings from the opened listings' shelf: the ones whose own text mentions more required
        values come first, then lower prices (Clockwerk reads candidates rather than trusting their titles)."""
        shelf = self.main_category()
        wanted = [value for _, value in self.needs_text()]
        ranked = []
        for ident, row in self.listing_rows.items():
            if ident in self.fact_sheets or not self.within_cap(row):
                continue
            row_shelf = _category_of(row) if shelf else ""
            if shelf and row_shelf and row_shelf != shelf:
                continue
            blob = " ".join(str(v) for v in row.values() if isinstance(v, str))
            matched = len([v for v in wanted if _limits_of(v) is None and _says(v, blob)])
            cost = _price_of(row)
            ranked.append((-matched, float("inf") if cost is None else cost, ident))
        ranked.sort()
        return [ident if cost == float("inf") else f"{ident} ({cost:g})" for _, cost, ident in ranked[:5]]

    def paging_hint(self, name: str, args: dict[str, Any], observation: Any) -> str:
        """Like Clockwerk, page through every match of a search before rephrasing it: report how many are still unseen."""
        if name != self.role_map.get("search"):
            return ""
        body = observation
        if isinstance(observation, dict) and len(observation) == 2:
            body = next((v for v in observation.values() if isinstance(v, dict)), observation)
        if not isinstance(body, dict):
            return ""
        pager = re.compile(r"(?i)page|offset|first|start|limit|skip|token|cursor|after|next|further")
        match_count = None
        for key, val in body.items():
            if isinstance(val, int) and not isinstance(val, bool) and not pager.search(str(key)):
                match_count = val
                break
        if match_count is None:
            return ""
        schema_props = (self.arg_specs.get(name) or {}).get("properties") or {}
        free_terms = [str(v) for k, v in args.items() if isinstance(v, str) and not pager.search(str(k)) and not (schema_props.get(k) or {}).get("enum")]
        query_key = name + _one_line({k: v for k, v in args.items() if not pager.search(str(k))})
        record = self.key_coverage.setdefault(query_key, {"label": " ".join(free_terms), "total": match_count, "seen": set()})
        record["total"] = match_count
        rows: list[Any] = []
        for val in body.values():
            if isinstance(val, list) and val and all(isinstance(x, dict) for x in val):
                rows = val
                break
        for row in rows:
            for key, val in row.items():
                if isinstance(val, str) and "id" in key.lower() and _ID_FORM.match(val):
                    record["seen"].add(val)
                    break
        seen_count = len(record["seen"])
        remaining = match_count - seen_count
        if remaining > 0 and match_count <= 60:
            return f"This search found {match_count} listings; {seen_count} have been seen so far and {remaining} more sit on its later pages."
        return ""

    def ask_referee(self, prompt: str, max_tokens: int = 700, models: list[str] | None = None) -> dict[str, Any] | None:
        """One plain JSON-returning model call, defaulting to the referee then the pilot model."""
        lineup = models if models else [REFEREE_MODEL, SECOND_VENDOR, PILOT_MODEL]
        answer = self.call_models([{"role": "user", "content": prompt}], None, lineup, max_tokens) or {}
        return _json_in(answer.get("content") or "")

    def listen(self, lines: list[str], question: str | None) -> str:
        """Interpret new shopper lines with the hearing model and fold the result into the requirement state."""
        unheard = [ln for ln in lines if ln not in self.heard_lines]
        if not unheard:
            return ""
        self.heard_lines.update(unheard)
        if self.seconds_left() < WIND_DOWN_AT:
            return ""
        sections = [HEARING_GUIDE, "\n\nTHE SHOPPER'S REQUEST (context only):\n" + str(self.view.get("query") or ""),
                    "\n\nREQUIREMENTS KNOWN SO FAR:\n" + self.needs_book()]
        if self.chat_log:
            sections.append("\n\nPREVIOUS EXCHANGES WITH THE SHOPPER (oldest first):\n" + "\n".join(self.chat_log[-12:]))
        if question:
            sections.append("\n\nTHE ASSISTANT'S QUESTION:\n" + question)
        sections.append("\n\nWHAT THE SHOPPER JUST SAID:\n" + "\n".join(unheard))
        self.chat_log.append(("asked: " + question + " -> " if question else "") + "shopper: " + " / ".join(unheard))
        heard = self.ask_referee("".join(sections), 1500, [HEARING_MODEL, REFEREE_MODEL])
        if not heard:
            return ""
        if question:
            # an answer to our own question can add needs, never drop or replace what is already required
            heard["dropped"] = []
            current = heard.get("requirements_now")
            if isinstance(current, list) and self.needs_now:
                heard["requirements_now"] = current + [r for r in self.needs_now if r not in current]
        changes = self._qc_absorb_hearing(heard)
        if changes:
            self.needs_rev += 1
        answer = str(heard.get("reply") or "").strip()
        if answer and self.role_map.get("message") and any("?" in ln for ln in unheard):
            self.reply_queued = answer
        gist = "Meaning: " + str(heard.get("meaning") or "")
        return gist + " (" + "; ".join(changes) + ")" if changes else gist

    def records_for(self, item: str) -> list[str]:
        """Up to the last four distinct store records (dicts) naming this item, found in a bounded walk of what was seen."""
        hits: list[str] = []
        pending: list[tuple[Any, int]] = []
        for record in self.seen:
            pending.append((record, 0))
            while pending:
                node, level = pending.pop()
                if level > 6 or len(hits) >= 8:
                    continue
                if isinstance(node, dict):
                    if item in [v for v in node.values() if isinstance(v, str)]:
                        flat = _one_line(_trim(node, 400))[:2500]
                        if flat not in hits:
                            hits.append(flat)
                        continue
                    children = list(node.values())
                elif isinstance(node, list):
                    children = list(node)
                else:
                    continue
                pending.extend((child, level + 1) for child in reversed(children))
        return hits[-4:]

    def vet_basket(self) -> str:
        """Referee check of the cart lines before an order (newest first, at most three); the first objection wins."""
        for item in list(reversed(self.basket))[:3]:
            objection = self.vet_listing(item, "ORDER")
            if objection:
                return objection
        return ""

    def vet_listing(self, item: str, purpose: str) -> str:
        """Ask the referee whether a listing meets every requirement; cache the verdict per requirement revision."""
        if self.seconds_left() < WIND_DOWN_AT + 20:
            return ""
        slot = f"{item}|{self.needs_rev}"
        if slot in self.verdict_cache:
            return self.verdict_cache[slot]
        evidence = self.records_for(item)
        if not evidence:
            return ""
        blocks = [REFEREE_GUIDE,
                  "\n\nTHE SHOPPER'S REQUEST:\n" + str(self.view.get("query") or ""),
                  "\n\nREQUIREMENTS IN FORCE (stated, revealed and changed):\n" + self.needs_book(),
                  f"\n\nITEM TO {purpose}: {item}\nSTORE RECORDS SEEN FOR THIS ITEM (oldest first):\n" + "\n".join(evidence)]
        if self.basket_view is not None:
            blocks.append("\n\nMOST RECENT CART VIEW:\n" + _one_line(self.basket_view)[:3000])
        recurring = self.label_rows()
        if recurring:
            blocks.append("\n\nLABELS RECURRING ACROSS THIS STORE'S LISTINGS (with counts; the item's category label has to match the request's wording as a whole, not a nearby label):\n" + recurring)
        ruling = self.ask_referee("".join(blocks), 1200)
        self.log.append({"kind": "verify", "item": item, "result": ruling})
        verdict = ""
        if ruling and ruling.get("fits") is False:
            faults = [str(p) for p in ruling.get("problems") or []][:5]
            verdict = (f"A separate check finds that {item} does NOT satisfy every requirement: " + "; ".join(faults)
                       + ". Unless you are sure the check is mistaken, pick another fully verified fit.")
        self.verdict_cache[slot] = verdict
        return verdict

    def tools_now(self) -> list[dict[str, Any]]:
        """The tool list offered this turn: narrowed to finishing moves as steps run out, minus messaging while planning."""
        remaining = self.steps_remaining()
        roles = self.role_map
        order_tool, cart_tool = roles.get("order"), roles.get("cart")
        allowed: set[str] | None
        if remaining <= 1:
            allowed = {order_tool, roles.get("decline")}
        elif remaining <= 3:
            allowed = {order_tool, roles.get("decline"), cart_tool}
        elif remaining <= 7 and (self.orders_tried == 0 or self.refused_times > 0):
            allowed = {order_tool, roles.get("decline"), cart_tool, roles.get("cart_view"), roles.get("view")}
        else:
            allowed = None
        offered = self.tool_specs
        if allowed is not None and order_tool and cart_tool:
            offered = [spec for spec in self.tool_specs if spec["function"]["name"] in allowed] or self.tool_specs
        chat_tool = roles.get("message")
        if self.planning and chat_tool:
            offered = [spec for spec in offered if spec["function"]["name"] != chat_tool]
        return offered

    def property_digest(self) -> tuple[str, int]:
        """Walk every record seen so far and list the fact names that recur in fact-rich blocks
        (five or more short text values), each with up to five sample values; also count those blocks."""
        tally: dict[str, int] = {}
        examples: dict[str, list[str]] = {}
        groups = [0]

        def _qd_codeish(raw: str) -> bool:
            # identifiers such as "ab-12_x" carry no property meaning
            return bool(re.search(r"\d", raw)) and bool(re.fullmatch(r"[A-Za-z0-9_\-]+", raw)) and ("-" in raw or "_" in raw)

        def walk(node: Any, level: int) -> None:
            if level > 6:
                return
            if isinstance(node, list):
                for child in node:
                    walk(child, level + 1)
                return
            if not isinstance(node, dict):
                return
            brief = [(field, raw) for field, raw in node.items() if isinstance(raw, str) and 0 < len(raw) <= 70]
            if len(brief) >= 5:
                groups[0] += 1
                for field, raw in brief:
                    if _qd_codeish(raw):
                        continue
                    label = re.sub(r"[_.]+", " ", str(field)).strip().lower()
                    tally[label] = tally.get(label, 0) + 1
                    pool = examples.setdefault(label, [])
                    lowered = raw.lower()
                    if len(pool) < 5 and lowered not in pool:
                        pool.append(lowered)
            for child in node.values():
                if isinstance(child, (dict, list)):
                    walk(child, level + 1)

        for rec in self.seen:
            walk(rec, 0)
        leaders = sorted(tally, key=lambda label: -tally[label])[:28]
        digest = "\n".join(f"- {label} [found in {tally[label]} listings]: " + " | ".join(examples.get(label, [])) for label in leaders)
        return digest, groups[0]

    def label_rows(self) -> str:
        """Short text fields shared by many records whose values form a small, repeating set,
        shown with how often each value turned up (at most twelve fields)."""
        freq: dict[str, dict[str, int]] = {}

        def walk(node: Any, level: int) -> None:
            if level > 6:
                return
            if isinstance(node, list):
                for child in node:
                    walk(child, level + 1)
                return
            if not isinstance(node, dict):
                return
            if len(node) >= 4:
                for field, raw in node.items():
                    if not isinstance(raw, str) or not 0 < len(raw) <= 40:
                        continue
                    if re.search(r"\d", raw) and re.fullmatch(r"[A-Za-z0-9_\-]+", raw):
                        continue
                    per = freq.setdefault(str(field), {})
                    per[raw] = per.get(raw, 0) + 1
            for child in node.values():
                if isinstance(child, (dict, list)):
                    walk(child, level + 1)

        for rec in self.seen:
            walk(rec, 0)
        rows = [
            f"- {field}: " + " | ".join(f"{val} ({n} times)" for val, n in sorted(per.items(), key=lambda kv: -kv[1]))
            for field, per in freq.items()
            if sum(per.values()) >= 4 and 2 <= len(per) <= 8 and max(per.values()) >= 2
        ]
        return "\n".join(rows[:12])

    def note_args(self, text: str) -> dict[str, Any] | None:
        """Arguments for the message tool carrying `text`, or None when its schema needs more than one free-text slot."""
        spec = self.arg_specs.get(self.role_map.get("message") or "") or {}
        fields = spec.get("properties") or {}
        needed = list(spec.get("required") or [])
        free_text = []
        for field, desc in fields.items():
            if isinstance(desc, dict) and (desc.get("type") == "string" or "string" in (desc.get("type") or [])) and "enum" not in desc:
                free_text.append(field)
        slot = None
        for field in needed:
            if field in free_text:
                slot = field
                break
        if not slot:
            slot = free_text[0] if len(free_text) == 1 else None
        if not slot or any(field != slot for field in needed):
            return None
        return {slot: text}

    def send_questions(self, force: bool = False) -> bool:
        """Plan the shopper questions with a model, send them, and log the answers; True when anything was sent."""
        tool = self.role_map.get("message")
        cap = 3 if self.note_allowance is None else self.note_allowance
        if tool and self.notes_sent < cap:
            menu, fact_sets = self.property_digest()
            pressing = bool(self.reading_plan.get("ask_first")) and not self.questions_log
            too_early = len(self.fact_sheets) < 4 and self.steps_used < min(10, self.step_cap // 3)
            if too_early and not pressing and not force:
                return False
            pressing = pressing and not force
            if not (self.steps_remaining() <= 10 or self.seconds_left() < WIND_DOWN_AT + 40):
                room = cap - self.notes_sent
                result = self.ask_referee(self._qd_question_prompt(room, menu), 2000, [HEARING_MODEL, REFEREE_MODEL])
                thin = pressing and fact_sets < 4
                drafted = [q for q in (result or {}).get("questions") or [] if isinstance(q, dict) and str(q.get("question") or "").strip()]
                if thin:
                    drafted = drafted[:1]
                drafted = self.keep_main(drafted, pressing)[:room]
                outgoing: list[dict[str, Any]] = []
                topics: list[tuple[str, str]] = []
                for pos, entry in enumerate(drafted):
                    wording = str(entry.get("question")).strip()
                    payload = self.note_args(wording)
                    if not payload:
                        continue
                    payload, issues = _validate(payload, self.arg_specs[tool], "args")
                    if issues:
                        continue
                    outgoing.append({"id": f"ask_{self.notes_sent + pos + 1}_{self.steps_used}", "type": "function", "function": {"name": tool, "arguments": json.dumps(payload)}})
                    topics.append((str(entry.get("property")), wording))
                if outgoing:
                    self.convo.append({"role": "assistant", "content": "Now putting questions to the shopper on " + ", then ".join(t[0] for t in topics) + ".", "tool_calls": outgoing})
                    answers = self.handle_calls(outgoing, planned=True)
                    self._qd_post_replies(answers)
                    for pos, (_, body) in enumerate(answers):
                        if pos < len(topics):
                            heard = body.split(chr(10) + "[status]")[0][:400]
                            self.questions_log.append(f"- {topics[pos][0]}: asked \"{topics[pos][1]}\" -> {heard}")
                    if self.notes_sent >= cap or not thin:
                        self.planning = self.notes_sent < cap and thin
                    return True
        self.planning = False
        return False

    def answer_back(self, text: str) -> bool:
        """Reply to a question the shopper put to us, if the message budget and the clock allow."""
        tool = self.role_map.get("message")
        cap = 3 if self.note_allowance is None else self.note_allowance
        blocked = not tool or self.notes_sent >= cap or self.steps_remaining() <= 4 or self.seconds_left() < WIND_DOWN_AT
        payload = None if blocked else self.note_args(text)
        if not payload:
            return False
        payload, issues = _validate(payload, self.arg_specs[tool], "args")
        if issues:
            return False
        outgoing = {"id": f"reply_{self.steps_used}", "type": "function", "function": {"name": tool, "arguments": json.dumps(payload)}}
        self.convo.append({"role": "assistant", "content": "Replying to the shopper's question.", "tool_calls": [outgoing]})
        self._qd_post_replies(self.handle_calls([outgoing], planned=True))
        return True

    def count_check(self, name: str, args: dict[str, Any]) -> str:
        """Warn when a cart call's quantity differs from the quantity the request asks for."""
        target = self.reading_plan.get("quantity")
        if isinstance(target, bool) or not isinstance(target, int) or target < 1:
            return ""
        fields = self.arg_specs[name].get("properties") or {}
        mode = _one_line({k: v for k, v in args.items() if isinstance(fields.get(k), dict) and "enum" in fields[k]})
        if self.adding_mode is not None and mode != self.adding_mode:
            return ""
        numeric = [(k, d) for k, d in fields.items() if isinstance(d, dict) and d.get("type") == "integer" and "enum" not in d]
        for field, desc in numeric:
            given = args.get(field, desc.get("default", 1))
            if isinstance(given, int) and given != target:
                return f"the request calls for {target} units of a single listing, yet this call asks for {given}."
        return ""

    def recheck_before_decline(self) -> str:
        """One-time second opinion over everything read, before the agent is allowed to decline."""
        if self.rechecked or self.seconds_left() < WIND_DOWN_AT + 30:
            return ""
        self.rechecked = True
        evidence: list[str] = []
        used = [0]

        def _qd_take(chunk: str, ceiling: int) -> None:
            if used[0] + len(chunk) <= ceiling:
                evidence.append(chunk)
                used[0] += len(chunk)

        for key in reversed(list(self.fact_sheets)):
            _qd_take(_one_line(self.fact_sheets[key]), 30000)
        for rec in reversed(self.seen):
            _qd_take(_one_line(_trim(rec, 160)), 36000)
        if not evidence:
            return ""
        brief = "\n\n".join([
            RECHECK_GUIDE,
            "THE SHOPPER'S REQUEST, AS WRITTEN:\n" + str(self.view.get("query") or ""),
            "REQUIREMENTS AS THEY STAND NOW:\n" + self.needs_book(),
            "LISTING RECORDS READ (most recent first):\n" + "\n".join(evidence),
        ])
        result = self.ask_referee(brief, 1200)
        self.log.append({"kind": "audit", "result": result})
        hits = [str(x) for x in (result or {}).get("possible_fits") or [] if isinstance(x, (str, int))][:5]
        if not hits:
            return ""
        why = f" ({result.get('reason')})" if result.get("reason") else ""
        return ("A separate review believes these listings might satisfy every requirement: " + ", ".join(hits) + why
                + ". Confirm them (details, stock, price) and purchase the right one; decline only when every one of them really fails.")

    def bland_args(self, name: str, text: str, kind: str | None = None) -> dict[str, Any]:
        """Fill a tool's required arguments with safe defaults, putting `text` in free-text slots."""
        spec = self.arg_specs.get(name) or {}
        fields = spec.get("properties") or {}
        filled: dict[str, Any] = dict(self.closing_args.get(kind) or {}) if kind else {}
        for field in [f for f in spec.get("required") or [] if f not in filled]:
            desc = fields.get(field) or {}
            kind_of = desc.get("type")
            if kind_of == "string" or "string" in (kind_of or []):
                filled[field] = text[: desc.get("maxLength", len(text))]
            elif "enum" in desc and desc["enum"]:
                filled[field] = desc["enum"][0]
            elif kind_of == "boolean":
                filled[field] = True
            elif kind_of in ("integer", "number"):
                filled[field] = max(1, desc.get("minimum", 1))
        if len(fields) == 1:
            for field, desc in fields.items():
                if field not in filled and isinstance(desc, dict) and desc.get("type") == "string" and "enum" not in desc:
                    filled[field] = text[: desc.get("maxLength", len(text))]
        return filled

    def plain_refusal(self, args: dict[str, Any], name: str) -> dict[str, Any]:
        """Swap decline text for a neutral sentence when it must not name listings or it mentions a known id."""
        policy = self.reading_plan.get("decline_no_names")
        if policy is False:
            return args
        splitter = r"[\s.,;:!?()\[\]{}<>\"'`]+"
        names_listing = any(
            piece and piece in self.known_ids
            for value in args.values() if isinstance(value, str)
            for piece in re.split(splitter, value)
        )
        if not (policy or names_listing):
            return args
        neutral = "Sorry, nothing in the store meets all of your requirements, so I am not ordering anything."
        fields = (self.arg_specs.get(name) or {}).get("properties") or {}
        out: dict[str, Any] = {}
        for field, value in args.items():
            desc = fields.get(field) or {}
            out[field] = neutral[: desc.get("maxLength", len(neutral))] if isinstance(value, str) and "enum" not in desc else value
        return out

    def close_from_code(self) -> None:
        """End the episode without the model: cart a likely fit if the cart is empty, then order or decline."""
        if self.done or self.out_of_funds or self.steps_remaining() <= 0 or self.seconds_left() < 2:
            return
        order_tool, decline_tool, cart_tool = self.role_map.get("order"), self.role_map.get("decline"), self.role_map.get("cart")
        apology = "Sorry, nothing in the store meets all of your requirements, so I am not ordering anything."
        candidates = []
        if order_tool and cart_tool and not self.basket and self.steps_remaining() >= 2 and self.seconds_left() > 8:
            candidates = self.confirmed() or self.likely_fits()
        if candidates:
            pick = self.favoured(candidates)
            got = self.store_call([{"name": cart_tool, "args": self.cart_call_args(pick)}])
            self.log.append({"kind": "forced_cart", "item": pick, "reply": got})
            faulted = isinstance(got.get("observation"), dict) and any(re.search(r"fault|error|issue", str(k)) for k in got["observation"]) if got is not None else True
            if not faulted:
                self.basket_touched = True
                self.basket = [pick]
        if order_tool and self.basket_touched:
            got = self._qd_finish({"name": order_tool, "args": self.bland_args(order_tool, "Order the cart.", "order")})
        elif decline_tool:
            got = self._qd_finish({"name": decline_tool, "args": self.bland_args(decline_tool, apology, "decline")})
        else:
            return
        if not got or got.get("finished"):
            return
        if isinstance(got.get("observation"), dict) and "refusal" in got["observation"] and decline_tool and self.steps_remaining() > 0:
            self._qd_finish({"name": decline_tool, "args": self.bland_args(decline_tool, apology, "decline")})

    def absorb_result(self, name: str, observation: Any) -> str:
        """Record ids from a tool result and return any note the model must see about it."""
        if isinstance(observation, list):
            _gather_ids(observation, self.known_ids)
            return ""
        if not isinstance(observation, dict):
            return ""
        _gather_ids(observation, self.known_ids)
        remarks: list[str] = []
        message_tool = self.role_map.get("message")
        if observation.get("error") == "budget_exceeded":
            if name == message_tool or self.note_allowance is not None:
                self.note_allowance = self.notes_sent if self.note_allowance is None else min(self.note_allowance, self.notes_sent)
            remarks.append("This budget is exhausted; stop calling this tool.")
        if "refusal" in observation:
            self.refused_times += 1
            self.order_vetted = False
            remarks.append(
                "THE ORDER WAS REFUSED, but the episode goes on and the cart was left as it was. Carry out every change the refusal "
                "states, repair the cart (take out any line that no longer qualifies, for example a sold-out item), re-check the "
                "new pick's stock and price, and then place the order again."
            )
        elif len(observation) >= 2 and name != message_tool:
            said: list[str] = []
            for value in observation.values():
                if isinstance(value, list) and value and all(isinstance(v, str) for v in value):
                    _gather_lines(value, said)
            if said:
                remarks.append("A message from the shopper (a change) came with this result; act on it: " + " | ".join(said[:4]))
        return " ".join(remarks)

    def handle_calls(self, calls: list[dict[str, Any]], planned: bool = False) -> list[tuple[str, str]]:
        """Screen the model's tool calls, gate any order/decline, run the rest in batches, and answer each call id in request order."""
        gate: dict[str, Any] = {
            "allowed": {t["function"]["name"] for t in self.tools_now()},
            "finishing": {self.role_map.get("order"), self.role_map.get("decline")} - {None},
            "msg": self.role_map.get("message"),
            "cart": self.role_map.get("cart"),
            "planned": planned,
            "count": len(calls),
            "queued_msgs": 0,
        }
        out: list[tuple[str, str]] = []
        queue: list[tuple[str, str, dict[str, Any]]] = []
        for call in calls:
            cid = call["id"]
            verdict, payload = self._qe_screen(call, gate)
            if verdict == "skip":
                out.append((cid, payload))
            else:
                queue.append((cid,) + payload)
        closers = gate["finishing"]
        if queue and queue[0][1] in closers and self._qe_finisher(queue, out):
            return out
        if queue and closers and queue[0][1] not in closers:
            queue = self._qe_reserve_last_step(queue, out)
        for offset in range(0, len(queue), 16):
            self._qe_run_batch(queue[offset : offset + 16], out, gate)
        rank = {c["id"]: i for i, c in enumerate(calls)}
        return sorted(out, key=lambda pair: rank.get(pair[0], 0))

    def play(self) -> list[dict[str, Any]]:
        """Drive the model turn by turn until the episode ends, then make sure it is closed."""
        if self.done:
            return self.log
        self.brief_once()
        self.log.append({"kind": "plan", "plan": self.reading_plan, "roles": self.role_map})
        self.convo = [{"role": "system", "content": PILOT_GUIDE}, {"role": "user", "content": self._qd_opening_message()}]
        silent_turns = 0
        misses = 0
        issued = [0]
        turns = 0
        while turns < TURN_LIMIT and not (self.done or self.out_of_funds or self.steps_remaining() <= 0):
            turns += 1
            if self.seconds_left() < CLOSE_NOW_AT or self.spent > SPEND_STOP:
                self.close_from_code()
                break
            if self.reply_queued:
                pending, self.reply_queued = self.reply_queued, ""
                if self.answer_back(pending):
                    continue
            if self.planning and self.send_questions():
                continue
            message = self.call_models(self.convo, self.tools_now(), [PILOT_MODEL, RESERVE_MODEL, PILOT_MODEL], 2500)
            if message is None:
                misses += 1
                if misses < 2:
                    continue
                self.close_from_code()
                break
            calls = self._qd_normalise_calls(message, issued)
            said = message.get("content") or ""
            said = said if isinstance(said, str) else _one_line(said)
            binding = {self.role_map.get("cart"), self.role_map.get("order"), self.role_map.get("decline")} - {None}
            if self.planning and any(c["function"]["name"] in binding for c in calls) and self.send_questions(force=True):
                continue
            if calls:
                silent_turns = 0
                self.log.append({"kind": "model", "content": said[:2000], "calls": [c["function"] for c in calls]})
                self.convo.append({"role": "assistant", "content": said or None, "tool_calls": calls})
                self._qd_post_replies(self.handle_calls(calls))
                continue
            silent_turns += 1
            self.convo.append({"role": "assistant", "content": said or "(no tool was called)"})
            self.convo.append({"role": "user", "content": "Each reply has to call one or more of the tools provided. Carry on.\n" + self.status_text()})
            if silent_turns >= 3:
                self.close_from_code()
                break
        if not (self.done or self.out_of_funds):
            self.close_from_code()
        self.log.append({"kind": "end", "finished": self.done, "steps": self.steps_used, "cost": round(self.spent, 5), "seconds": round(self.ticks * TICK_LENGTH, 1)})
        return self.log

    @staticmethod
    def _qa_cache_marked(convo: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Shallow copy of the conversation whose first and last text messages carry an ephemeral cache mark."""
        copied = list(map(dict, convo))
        for spot in {0, len(copied) - 1}:
            text = copied[spot].get("content")
            if isinstance(text, str) and text:
                copied[spot]["content"] = [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]
        return copied

    def _qa_absorb_reply(self, data: dict[str, Any], count: int) -> dict[str, Any]:
        """Bring the step counter and finished flag in line with a successful store reply."""
        reported = data.get("step")
        self.steps_used = reported if isinstance(reported, int) else self.steps_used + count
        if data.get("finished"):
            self.done = True
        return data

    @staticmethod
    def _qb_is_count(value: Any) -> bool:
        """True for a real int or float (a bool does not count)."""
        return not isinstance(value, bool) and isinstance(value, (int, float))

    @staticmethod
    def _qb_flat(value: Any) -> bool:
        """A non-empty dict holding only plain scalar values (a facts block)."""
        return isinstance(value, dict) and len(value) > 0 and all(isinstance(x, (str, int, float)) for x in value.values())

    @staticmethod
    def _qb_ident(mapping: dict[str, Any]) -> Any:
        """The first id-shaped string under a key mentioning 'id', else None."""
        for key, value in mapping.items():
            if isinstance(value, str) and "id" in key.lower() and _ID_FORM.match(value):
                return value
        return None

    @staticmethod
    def _qb_compact(record: dict[str, Any]) -> dict[str, Any]:
        return {key: (value if isinstance(value, dict) else _trim(value, 160)) for key, value in record.items()}

    @staticmethod
    def _qb_words(text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", text)

    def _qb_product(self) -> set[str]:
        return set(map(_singular, _tokens(self.reading_plan.get("product") or "", False)))

    @staticmethod
    def _qb_add_choice(options: list[str]) -> str:
        """The enum option that adds to the cart; failing that, the first one that does not remove; else the first."""
        for option in options:
            if re.search(r"(?i)add|hold|put|keep|plus|insert|reserve|increase|take", option):
                return option
        for option in options:
            if not re.search(r"(?i)remov|releas|drop|minus|delet|discard|clear|decreas", option):
                return option
        return options[0]

    def _qb_on_shelf(self, sheet: dict[str, Any], shelf: str) -> bool:
        """The sheet sits on the main shelf (or names none) and passes the budget and stock check."""
        own = _category_of(sheet)
        if shelf and own and own != shelf:
            return False
        return self.within_cap(sheet)

    @staticmethod
    def _qb_price_in(node: Any, item: str, depth: int) -> float | None:
        """Depth-first search (at most six levels) for a record naming the item and carrying a price; that price or None."""
        if depth > 6:
            return None
        if isinstance(node, dict):
            if any(isinstance(v, str) and v == item for v in node.values()):
                price = _price_of(node)
                if price is not None:
                    return price
            children = list(node.values())
        elif isinstance(node, list):
            children = node
        else:
            return None
        for child in children:
            price = Shopper._qb_price_in(child, item, depth + 1)
            if price is not None:
                return price
        return None

    def _qc_fact_breaches(self, item: str, own_sheet: dict[str, Any]) -> str:
        """Requirements the given fact sheet contradicts, worded for the loop (empty when nothing clashes)."""
        spare_words = set(map(_singular, _tokens(self.reading_plan.get("product") or "", False)))
        clashes: list[str] = []
        for need_prop, need_value in self.needs_text():
            outcome, stated = _verdict_for(own_sheet, need_prop, need_value, spare_words)
            if outcome != "no":
                continue
            clashes.append(f"{need_prop}: {need_value} (the listing's own fact reads {stated})")
        if clashes:
            return (f"Judged on the store's own facts, {item} misses: " + "; ".join(clashes[:4])
                    + ". The store takes each requirement from the listing's fact for that property, never from the title or a similar-sounding word.")
        return ""

    def _qc_cheaper_pieces(self, item: str, pick_cost: float) -> list[str]:
        pieces: list[str] = []
        lower_fits = []
        for ident, cost in self.confirmed():
            if ident != item and cost is not None and cost < pick_cost:
                lower_fits.append(f"{ident} ({cost:g})")
        if lower_fits and self.steps_remaining() > 4:
            pieces.append(f"these opened listings satisfy every current requirement on the store's own facts and are priced below {item} ({pick_cost:g}): " + ", ".join(lower_fits[:4]))
        shelf = _category_of(self.fact_sheets.get(item) or self.listing_rows.get(item) or {})
        unopened = []
        if shelf:
            for ident, row in self.listing_rows.items():
                if ident in self.fact_sheets or _category_of(row) != shelf:
                    continue
                cost = _price_of(row)
                if cost is not None and cost < pick_cost and self.within_cap(row):
                    unopened.append((cost, ident))
        unopened.sort()
        if unopened and self.steps_remaining() > 9:
            listed = ", ".join(f"{ident} ({cost:g})" for cost, ident in unopened[:5])
            pieces.append("lower-priced listings from the same shelf showed up in results but were never opened: " + listed + "; open those, lowest price first, before you order")
        return pieces

    def _qc_decline_leads(self) -> list[str]:
        leads: list[str] = []
        never_opened = self.unread_picks()
        if never_opened:
            leads.append("these listings appeared in results but were never opened: " + ", ".join(never_opened) + "; open them, because their facts settle it and their titles do not")
        partial: list[str] = []
        for cover in self.key_coverage.values():
            if len(partial) == 3:
                break
            if cover["label"] and len(cover["seen"]) < cover["total"] <= 60:
                partial.append(f"'{cover['label']}' ({cover['total']} matched, {len(cover['seen'])} seen)")
        if partial:
            leads.append("these searches still hold matches you have not looked at: " + ", ".join(partial) + "; page through the rest of their results")
        return leads

    @staticmethod
    def _qc_label_taken(root_words: set[str], covered: list[str]) -> bool:
        for other in covered:
            other_words = set(_label_words(other))
            if root_words <= other_words:
                return True
            trimmed = other_words - {"type"}
            if trimmed and trimmed <= root_words:
                return True
        return False

    def _qc_absorb_hearing(self, heard: dict[str, Any]) -> list[str]:
        """Apply the hearing model's verdict to the requirement state; return the change notes in their fixed order."""
        changes: list[str] = []
        current = heard.get("requirements_now")
        if isinstance(current, list) and all(isinstance(r, str) for r in current):
            self.needs_now = [r for r in current if r.strip()]
        added = heard.get("needs") or []
        for need in added:
            if isinstance(need, str) and need and need not in self.told:
                self.told.append(need)
                changes.append(f"new requirement: {need}")
        for prop in heard.get("no_preference") or []:
            if isinstance(prop, str) and prop and prop not in self.indifferent:
                self.indifferent.append(prop)
        for drop in heard.get("dropped") or []:
            if isinstance(drop, str) and drop and drop not in self.released:
                self.released.append(drop)
                changes.append(f"dropped: {drop}")
        new_cap = heard.get("new_budget")
        cap_ok = isinstance(new_cap, (int, float)) and not isinstance(new_cap, bool) and new_cap > 0
        if cap_ok and new_cap != self.budget:
            self.budget = new_cap
            changes.append(f"new budget: {new_cap}")
        if heard.get("unavailable"):
            changes.append("one item is no longer available: take it out of the cart and do not order it")
        for need in heard.get("needs") or []:
            if not (isinstance(need, str) and ":" in need):
                continue
            stating, _ = self.sheets_with(need)
            if stating:
                changes.append(f"opened listings stating {need.strip()}: " + ", ".join(stating[:6]))
            elif self.fact_sheets:
                changes.append(f"none of the opened listings states {need.strip()} yet: search for it")
        if (bool(heard.get("dropped")) or cap_ok) and self.reading_plan.get("priority") == "cheapest":
            changes.append("the set of fitting items has shifted: search again (items ruled out earlier may qualify now) and pick the cheapest fit afresh before ordering")
        return changes

    def _qd_question_prompt(self, room: int, menu: str) -> str:
        """Assemble the brief for the question planner."""
        parts = [QUESTIONS_GUIDE.replace("QUESTION_COUNT", str(room)), "THE SHOPPER'S REQUEST, AS WRITTEN:\n" + str(self.view.get("query") or "")]
        parts.append("REQUIREMENTS KNOWN SO FAR:\n" + self.needs_book())
        hints = self.story_hints()
        if hints:
            parts.append(
                "HINTS FROM THE STORY (properties the request alludes to without naming a value; put these first, and phrase each "
                "with the wording the listings use for it): " + "; ".join(f"{p} (probably {v})" if v else p for p, v in hints)
            )
        if self.questions_log:
            parts.append("QUESTIONS ALREADY SENT (and the shopper's answers):\n" + "\n".join(self.questions_log))
        wanted_first = self.reading_plan.get("ask_first")
        if wanted_first:
            parts.append("MUST BE LEARNED FROM THE SHOPPER: " + "; ".join(str(h) for h in wanted_first or []))
        parts.append("PROPERTIES FOUND IN THE LISTING DETAILS SO FAR (name, how many listings, sample values):\n" + (menu or "(nothing seen yet)"))
        core = self.main_labels()
        if core:
            parts.append("CORE PROPERTIES (present on most opened listings and neither settled nor asked yet; restrict questions to these): " + ", ".join(core))
        return "\n\n".join(parts)

    def _qd_post_replies(self, replies: list[tuple[str, str]]) -> None:
        """Put tool replies into the conversation, the status line riding on the final one."""
        final = len(replies) - 1
        for pos, (call_id, body) in enumerate(replies):
            self.convo.append({"role": "tool", "tool_call_id": call_id, "content": body + "\n" + self.status_text() if pos == final else body})

    def _qd_finish(self, action: dict[str, Any]) -> Any:
        """Send one finishing action and trace it."""
        reply = self.store_call([action])
        self.log.append({"kind": "forced_finish", "action": action, "reply": reply})
        return reply

    def _qd_opening_message(self) -> str:
        """First user turn: the request, the opening observation, the briefing and the status line."""
        start = self.view.get("opening")
        pieces = ["THE SHOPPER'S REQUEST:\n" + str(self.view.get("query") or "")]
        if start is not None:
            pieces.append("WHAT THE STORE SHOWED FIRST:\n" + _one_line(start))
        extra = self.briefing_note()
        if extra:
            pieces.append(extra)
        pieces.append(self.status_text())
        if start is not None:
            self.seen.append(start)
            _gather_ids(start, self.known_ids)
        return "\n\n".join(pieces)

    @staticmethod
    def _qd_normalise_calls(message: dict[str, Any], issued: list[int]) -> list[dict[str, Any]]:
        """Turn the model's tool calls into well-formed call records; issued[0] numbers the ones lacking an id."""
        tidy: list[dict[str, Any]] = []
        for raw in message.get("tool_calls") or []:
            if isinstance(raw, dict):
                issued[0] += 1
                fn = raw.get("function") or {}
                arguments = fn.get("arguments")
                if not isinstance(arguments, str):
                    arguments = json.dumps(fn.get("arguments") or {})
                tidy.append({"id": raw.get("id") or f"call_{issued[0]}", "type": "function", "function": {"name": fn.get("name"), "arguments": arguments}})
        return tidy

    def _qe_enum_signature(self, tool: str, args: dict[str, Any]) -> tuple[dict[str, Any], str]:
        """Schema properties of a tool plus a compact key built from its enum-typed (mode) arguments only."""
        props = self.arg_specs[tool].get("properties") or {}
        modal = {}
        for key, val in args.items():
            if isinstance(props.get(key), dict) and "enum" in props[key]:
                modal[key] = val
        return props, _one_line(modal)

    @staticmethod
    def _qe_parse_args(raw: Any) -> dict[str, Any] | None:
        """Turn the model's argument payload into a dict; None when the JSON text is broken."""
        if isinstance(raw, dict):
            return raw
        payload = raw or ""
        if not payload.strip():
            return {}
        return _json_in(payload)

    def _qe_cart_review(self, tool: str, args: dict[str, Any]) -> str | None:
        """First look at a cart add for a product in this needs revision: return an objection or None."""
        props, mode_key = self._qe_enum_signature(tool, args)
        if self.adding_mode is not None and mode_key != self.adding_mode:
            return None
        product = None
        for key, val in args.items():
            if not isinstance(val, str) or "enum" in (props.get(key) or {}):
                continue
            bare = val.strip()
            if bare and " " not in bare:
                product = val
                break
        targets = [product] if product else self._listed_ids(args)
        for target in targets[:3]:
            tag = f"{target}|{self.needs_rev}"
            if tag in self.cart_vetted:
                continue
            self.cart_vetted.add(tag)
            objection = self.rule_review(target, True) or self.vet_listing(target, "ADD TO THE CART")
            if objection:
                return objection
        return None

    @staticmethod
    def _listed_ids(args: dict[str, Any]) -> list[str]:
        """Listing ids named inside list arguments of a cart call (one per line object or plain entry)."""
        found: list[str] = []
        for value in args.values():
            if not isinstance(value, list):
                continue
            for line in value:
                cells = list(line.values()) if isinstance(line, dict) else [line]
                ident = next((c for c in cells if isinstance(c, str) and c.strip() and " " not in c.strip()), None)
                if ident and ident not in found:
                    found.append(ident)
        return found

    @staticmethod
    def _cart_lines_shown(observation: Any) -> list[str] | None:
        """The cart as the store itself lists it in a reply (ids of the listed lines), or None when no list of lines shows."""
        stack = [observation]
        while stack:
            node = stack.pop()
            if not isinstance(node, dict):
                continue
            for value in node.values():
                if isinstance(value, list) and all(isinstance(v, dict) for v in value):
                    ids = []
                    for line in value:
                        ident = next((v for k, v in line.items() if isinstance(v, str) and "id" in str(k).lower() and _ID_FORM.match(v)), None)
                        if ident is None:
                            break
                        ids.append(ident)
                    else:
                        return ids
                elif isinstance(value, dict):
                    stack.append(value)
        return None

    def _made_up_page(self, tool: str, args: dict[str, Any]) -> str:
        """A call repeating an earlier call to the same tool with one text argument added reads a later page; that text
        has to be a value one of this tool's replies handed out, never one the model made up."""
        if tool != self.role_map.get("search"):
            return ""
        same_tool = [(a, o) for t, a, o in self.call_log if t == tool]
        for past_args, observation in reversed(same_tool):
            added = [k for k in args if k not in past_args]
            if len(added) != 1 or any(args.get(k) != v for k, v in past_args.items()):
                continue
            value = args[added[0]]
            if not isinstance(value, str) or any(value in _one_line(o) for _, o in same_tool):
                return ""
            offered = [v for v in (observation.values() if isinstance(observation, dict) else []) if isinstance(v, str) and v.strip() and " " not in v and len(v) <= 40]
            return (f"'{value}' for {added[0]} was never handed out by this search, so it cannot open a later page. "
                    + (f"Values the last reply gave: {', '.join(offered[:6])}. " if offered else "That reply offered no next page. ")
                    + "Pass the next-page value exactly as a reply gave it.")
        return ""

    def _qe_screen(self, call: dict[str, Any], gate: dict[str, Any]) -> tuple[str, Any]:
        """Check one requested call. Gives ('skip', reason) or ('go', (name, args))."""
        fn = call.get("function") or {}
        tool = fn.get("name")
        usable = gate["allowed"]
        if tool not in self.arg_specs:
            return "skip", f"Skipped (no step spent): '{tool}' is not an available tool. Pick from: {', '.join(sorted(usable))}."
        if not gate["planned"] and tool not in usable:
            if tool == gate["msg"] and self.planning:
                why = "shopper questions get planned and sent on your behalf once you have read the details of a few listings"
            else:
                why = "not enough steps remain for that; the tools still usable right now are: " + ", ".join(sorted(usable))
            return "skip", f"Skipped (no step spent): {why}."
        parsed = self._qe_parse_args(fn.get("arguments"))
        if parsed is None:
            return "skip", "Skipped (no step spent): the arguments could not be read as a JSON object."
        parsed, issues = _validate(parsed, self.arg_specs[tool], "args")
        if issues:
            return "skip", "Skipped (no step spent): bad arguments: " + "; ".join(issues[:6])
        closers = gate["finishing"]
        if tool in closers and gate["count"] > 1:
            return "skip", "Skipped (no step spent): send an order or a decline on its own, only after the results of your other calls have come back."
        is_msg = tool == gate["msg"]
        is_cart = tool == gate["cart"]
        if is_msg and self.note_allowance is not None and self.notes_sent + gate["queued_msgs"] >= self.note_allowance:
            return "skip", "Skipped (no step spent): no messages are left in the budget. Stop writing to the shopper."
        made_up = self._made_up_page(tool, parsed)
        if made_up:
            return "skip", "Skipped (no step spent): " + made_up
        fingerprint = tool + _one_line(parsed)
        if tool not in closers and not is_msg and not is_cart and fingerprint in self.last_calls:
            return "skip", "Skipped (no step spent): this identical call was just made and nothing changed since; its result is shown above. Use different arguments or take another action."
        if is_cart:
            if self.steps_remaining() > 3:
                objection = self._qe_cart_review(tool, parsed)
                if objection:
                    return "skip", "Skipped (no step spent). " + objection + " If you are sure it is a fit, repeat the exact same call."
            caution = self.count_check(tool, parsed)
            if caution and fingerprint not in self.count_warned:
                self.count_warned.add(fingerprint)
                return "skip", "Skipped (no step spent): " + caution + " If you are confident, repeat the exact same call."
        if is_msg and not gate["planned"]:
            spoken = " ".join(str(v) for v in parsed.values() if isinstance(v, str))
            if "?" not in spoken and fingerprint not in self.warned_once:
                self.warned_once.add(fingerprint)
                return "skip", "Skipped (no step spent): each message to the shopper should pose a single question about a single property; statements or confirmations just burn the message budget. If a rule in the request demands this exact message, repeat the same call."
        if is_msg:
            gate["queued_msgs"] += 1
        return "go", (tool, parsed)

    def _unread_hold(self) -> str:
        """Before an order, once per cart state: a cart line whose own fact sheet was never read cannot be judged."""
        unread = [item for item in self.basket if item not in self.fact_sheets]
        stamp = f"unread|{self.cart_edit_step}|{','.join(self.basket)}"
        if not unread or stamp in self.warned_once or self.steps_remaining() <= len(unread) + 2:
            return ""
        self.warned_once.add(stamp)
        return ("The full details of " + ", ".join(unread) + " were never opened, so none of its properties is confirmed. "
                "Open them and test every requirement before ordering; if they fit, order again.")

    def _qe_order_gate(self, cid: str) -> tuple[str, str] | None:
        """Hold the first order attempt for a review; None lets it through."""
        if self.order_vetted or self.steps_remaining() <= 2 or self.seconds_left() <= WIND_DOWN_AT:
            return None
        hint = self.order_hint(self.basket[-1] if self.basket else "")
        if hint:
            return cid, "Held for now (no step spent). " + hint
        self.order_vetted = True
        objection = self.vet_basket()
        if objection:
            return cid, "Held for now (no step spent). " + objection
        checklist = (
            "Held for now (no step spent). Confirm before ordering: (1) the cart contains a single line, the chosen item at the needed quantity, and nothing more; "
            "(2) since your most recent cart change you re-read this item's stock and price with a tool that reports them, it is in stock, and price multiplied by quantity fits the current budget; "
            "(3) the item's own properties satisfy every stated requirement, every need the shopper disclosed and every announced change; "
            "(4) when the cheapest or best was requested, no better matching item exists among all you were able to find; "
            "(5) every procedural rule in the request has been respected. Check anything still open now; otherwise call the order tool once more."
        )
        return cid, checklist

    def _qe_decline_gate(self, cid: str) -> str | None:
        """Push back on a decline while there is room to search more; None means let it go."""
        hint = self.refusal_hint()
        if hint:
            return "Held for now (no step spent). " + hint
        if self.refusal_vetted and not self.search_watch and self.steps_remaining() > 12 and self.search_count < 8 and self.seconds_left() > WIND_DOWN_AT + 30:
            self.search_watch = True
            return (f"Held for now (no step spent). There are still {self.steps_remaining()} steps and just {self.search_count} searches so far. "
                    "Try more searches first: the wanted value as sellers would phrase it or as a fact value, the shelf or category listing, later pages; "
                    "open the details of candidates not yet read and re-read those already opened. Decline only if every candidate still fails.")
        if self.refusal_vetted or self.steps_remaining() <= 6 or self.seconds_left() <= WIND_DOWN_AT + 30:
            return None
        self.refusal_vetted = True
        audit = self.recheck_before_decline()
        if audit:
            return "Held for now (no step spent). " + audit
        return ("Held for now (no step spent). Make sure before declining: you searched several ways (other words, filters, pages), you read each candidate's own properties, "
                "and every one fails a stated requirement, a disclosed need, the budget or stock. Near synonyms and any key spelling count as matches. "
                "If something fits, buy it; if nothing does, call the decline tool once more.")

    def _qe_finisher(self, queue: list[tuple[str, str, dict[str, Any]]], replies: list[tuple[str, str]]) -> bool:
        """Review a finishing call at the head of the queue; True means stop and return the replies now."""
        cid, tool, params = queue[0]
        intent = self.read_intent(tool, params)
        if intent == "order":
            unread = self._unread_hold()
            if unread:
                replies.append((cid, "Held for now (no step spent). " + unread))
                return True
            held = self._qe_order_gate(cid)
            if held is not None:
                replies.append(held)
                return True
        if intent == "decline":
            swap = self.buy_instead()
            if swap is not None:
                alt_tool, alt_args, alt_text = swap
                self.log.append({"kind": "instead_of_decline", "tool": alt_tool, "args": alt_args})
                replies.append((cid, alt_text))
                queue[0] = (cid, alt_tool, alt_args)
                intent = self.read_intent(alt_tool, alt_args)
        if intent != "decline":
            return False
        pushback = self._qe_decline_gate(cid)
        if pushback is not None:
            replies.append((cid, pushback))
            return True
        queue[0] = (cid, tool, self.plain_refusal(params, tool))
        return False

    def _qe_reserve_last_step(self, queue: list[tuple[str, str, dict[str, Any]]], replies: list[tuple[str, str]]) -> list[tuple[str, str, dict[str, Any]]]:
        """Keep the final step free for the order or decline by dropping overflow calls."""
        spare = self.steps_remaining() - 1
        if spare >= len(queue):
            return queue
        room = max(0, spare)
        replies.extend((cid, "Skipped (no step spent): just the final step remains, and it has to be the order (or the decline).") for cid, _t, _a in queue[room:])
        return queue[:room]

    def _qe_heard(self, tool: str, args: dict[str, Any], observation: Any, msg_tool: str | None) -> tuple[list[str], str | None]:
        """Collect free-text lines that may come from the shopper, plus the question we asked (if any)."""
        lines: list[str] = []
        if tool == msg_tool:
            if isinstance(observation, str):
                lines = [observation]
            elif isinstance(observation, dict) and "error" not in observation:
                _gather_lines(list(v for v in observation.values() if isinstance(v, (str, list))), lines)
            return lines, " ".join(str(v) for v in args.values() if isinstance(v, str))
        if not isinstance(observation, dict):
            return lines, None
        for chunk in observation.values():
            if isinstance(chunk, list):
                if chunk and all(isinstance(v, str) for v in chunk):
                    _gather_lines(chunk, lines)
            elif isinstance(chunk, str) and len(chunk.split()) >= 4 and ("refusal" in observation):
                lines.append(chunk)
        return lines, None

    def _qe_track_cart(self, tool: str, args: dict[str, Any]) -> None:
        """Update the believed basket after a cart edit ran."""
        props, mode_key = self._qe_enum_signature(tool, args)
        if self.adding_mode is None:
            self.adding_mode = mode_key
        for key, val in args.items():
            schema = props.get(key) if isinstance(props.get(key), dict) else {}
            if not isinstance(val, str) or "enum" in schema:
                continue
            if not val.strip() or " " in val.strip():
                continue
            self.basket_modes[val] = mode_key
            remaining = [item for item in self.basket if item != val]
            self.basket = remaining + [val] if mode_key == self.adding_mode else remaining
        listed = self._listed_ids(args)
        if listed and not any(isinstance(v, str) and v.strip() and " " not in v.strip() for v in args.values()):
            self.basket = listed   # a call naming its lines in a list sets the cart to those lines
        self.cart_edit_step = self.steps_used

    def _qe_digest(self, tool: str, args: dict[str, Any], observation: Any, gate: dict[str, Any]) -> str:
        """Book-keep one executed call and build the text the model sees for it."""
        msg_tool, cart_tool, closers = gate["msg"], gate["cart"], gate["finishing"]
        if tool == msg_tool:
            self.notes_sent += 1
        if self.read_intent(tool, args) == "order":
            self.orders_tried += 1
        if tool == cart_tool:
            self.basket_touched = True
            self.order_vetted = False
        if tool in (cart_tool, msg_tool) or tool in closers:
            self.last_calls = []
        else:
            self.last_calls = (self.last_calls + [tool + _one_line(args)])[-2:]
        self.seen.append(observation)
        self.call_log.append((tool, dict(args), observation))
        self.note_down(tool, observation)
        if tool == cart_tool:
            self._qe_track_cart(tool, args)
        if tool == self.role_map.get("cart_view"):
            self.basket_view = observation
        if tool in (cart_tool, self.role_map.get("cart_view")):
            shown = self._cart_lines_shown(observation)
            if shown is not None:
                self.basket = shown   # the store's own listing of the cart wins over our tracking
        absorbed = self.absorb_result(tool, observation)
        paging = self.paging_hint(tool, args, observation)
        remark = " ".join(piece for piece in (absorbed, paging) if piece)
        lines, asked = self._qe_heard(tool, args, observation, msg_tool)
        if lines:
            gist = self.listen(lines, asked)
            if gist:
                remark = f"{remark} {gist}" if remark else gist
        return remark

    def _qe_run_batch(self, batch: list[tuple[str, str, dict[str, Any]]], replies: list[tuple[str, str]], gate: dict[str, Any]) -> None:
        """Send one batch of calls to the store and turn each observation into a reply."""
        if self.done or self.out_of_funds or self.steps_remaining() <= 0:
            replies.extend((cid, "Not executed: the episode has ended or there are no steps left.") for cid, _t, _a in batch)
            return
        batch = batch[: self.steps_remaining()]
        actions = [{"name": tool, "args": params} for _c, tool, params in batch]
        step_no = self.steps_used + 1
        reply = self.store_call(actions)
        self.log.append({"kind": "env", "step": step_no, "actions": actions, "reply": reply})
        if reply is None:
            replies.extend((cid, "The call did not go through (network or server problem). Please retry.") for cid, _t, _a in batch)
            return
        many = reply.get("observations")
        results = many if isinstance(many, list) else [reply.get("observation")]
        last = len(results) - 1
        for pos, (cid, tool, params) in enumerate(batch):
            if pos > last:
                replies.append((cid, "Not executed: the episode ended before this call ran."))
                continue
            seen_now = results[pos]
            remark = self._qe_digest(tool, params, seen_now, gate)
            shown = _one_line(seen_now)
            if pos == last and reply.get("finished"):
                shown += " [the episode is finished]"
            if remark:
                shown += "\n[note] " + remark
            replies.append((cid, shown))


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Entry point: run one shopping episode and always hand back its trace, even after a crash."""
    episode = Shopper(problem_data)
    try:
        return episode.play()
    except Exception as failure:
        episode.log.append(_qd_crash_entry(failure))
    try:
        if not (episode.done or episode.out_of_funds):
            episode.close_from_code()
    except Exception as second:
        episode.log.append(_qd_crash_entry(second))
    return episode.log


__all__ = ["agent_main"]
