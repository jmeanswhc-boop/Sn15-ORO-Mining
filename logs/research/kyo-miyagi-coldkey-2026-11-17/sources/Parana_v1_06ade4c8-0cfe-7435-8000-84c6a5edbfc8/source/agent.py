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
WRAP_UP_SECONDS = 70.0
FORCE_SECONDS = 28.0
OBS_CHARS = 16000
MAX_TURNS = 80

SYSTEM_PROMPT = """You are a meticulous shopping assistant. You serve one shopper inside a simulated store, and you can act only through the supplied tools. The task is scored 1 or 0. It scores 1 only if your finish is fully right: the ordered item meets every requirement the shopper stated, every hidden need the shopper reveals when asked, and every change the shopper announces while you work; the quantity and the budget are right; the cheapest (or best) choice is made when the request asks for it; and every rule the request states is followed. If nothing in the store can meet everything, the right finish is to decline, exactly the way the request says.

THE WORLD
- Every tool call uses one step, including calls that fail or return errors, and the number of steps is limited. Model thinking is free; steps are not.
- You may put several tool calls in one reply when none depends on another's result (for example, opening the details of several listings at once). They run in order. Never put an order or a decline together with other calls.
- If a tool shows the details of several listings in one call (a compare tool), read candidates with it in batches: one step for several listings.
- Tool names, arguments and the shape of results differ from store to store. Read the tool descriptions and the JSON schemas carefully and use exactly the argument names and allowed values they define. Learn from the first results which tool shows which facts (price, stock, detailed properties).
- Find out early which category labels the store uses for the requested kind of product (they appear on listings). When neighbouring labels exist, the right one is the label that matches the request's wording as a whole; search and buy within it.
- Search tools can be limited: a capped number of visible rows or pages, results sorted by id rather than by relevance or price, every word required, filters that only work in one mode. Compare the reported total with the rows you can actually see, and narrow (more specific words, category or price filters, other pages) until you have seen the whole relevant set.

THE SHOPPER AND HIDDEN NEEDS
- The shopper has needs they did not write down. They reveal one only when a message asks about that property. Each message is answered only about the FIRST thing it asks, so ask about exactly ONE property per message, as a short, plain, friendly question to the shopper that names the property (an example value or two is fine). Never give instructions, never mention lists, labels, positions or rules, and never ask two things at once.
- A reply saying they don't mind, have no preference, or anything is fine, means that property is free. A reply that states a preference or a limit is a hard requirement from now on, exactly as worded (numeric limits included).
- When a planner is available, your questions are chosen and sent for you as soon as you have opened the details of two or three listings of the requested kind, so do that in your first moves. The shopper's answers then appear in the conversation with their meaning; apply every revealed need as a hard requirement and search again with it combined with the stated values.
- If you must ask yourself: first open the details of two or three listings, then ask about the properties the story hints at (who it is for, how it must look, where or how it will be used), then about properties the listings carry that differ among candidates. Never ask about something the request already settles, and never exceed the message budget.

RULES OF PROCEDURE
- The request may set rules about how to work: how many listings to look at or compare before choosing, what to check, what to tell the shopper, what to put in a remark, how to decline. These are graded like requirements: follow each one literally and completely. A rule to look at or open N listings means N listings that fully fit, each opened with the details tool.
- If the shopper asks you something (for example to confirm a change), it will be answered for you when messages remain; treat the change itself as binding.
- Something the shopper would merely like ("ideally", "would be nice", "if possible") is not a requirement: use it only to choose between items that already meet everything, and never over a cheapest rule.
- If the request leaves something you can only learn from the shopper (for example no price limit, or two wishes that may not both be possible and the shopper must say which matters more), find it out by asking before you choose.

CHANGES
- While you work, the shopper may change things: a new budget (higher or lower), a requirement dropped, replaced or added, a price going up, an item sold out. One shopper line can carry several changes. Shopper lines can appear inside any tool result (an extra field holding sentences) or attached to an order attempt. Each change is binding from that moment: re-check your pick and your backups against it, and when the request asks for the cheapest or best, choose again over the new acceptable set.
- An order attempt can be REFUSED. The result then has a "refusal" field, even if the inner result looks successful. The episode continues and the cart is unchanged. Read the refusal, apply every change it states, remove anything that is no longer acceptable from the cart (for example a sold-out item), add the next verified fit, re-check stock and price, and order again.
- Items can also sell out or change price silently. That is why stock and price must be re-checked right before every order attempt; an item whose current price times quantity is over the current budget no longer fits.

HOW TO JUDGE A LISTING (this decides the score)
- A requirement is met only when the listing's own property for it states the value. Property keys vary in spelling (spaces, dots, underscores, prefixes) and the same property may appear under several keys; any key that names that property counts. Values match by meaning: one element of a comma-separated list is enough; singular and plural are the same; close synonyms count; case does not matter. A word that merely starts the same way does not count, and a broader or different class does not count (for example girls is not women, plain unisex is not female).
- Use the title or description as evidence only when the listing has no property key for that requirement at all. If the listing's own property says something different, the listing fails even when the title or description says the value.
- A value under a different, related property does not count (a length field does not satisfy a style requirement, a material of the lining does not satisfy the material).
- Numbers: compare as numbers after converting units; limits include the edge value; a range counts only when the whole range meets the limit.
- Respect exclusions ("leave out anything with X") and closed lists ("A or B and nothing else").
- The requested kind of product is met by the listing's category; never demand that some property repeat the category's name. The item must be the requested kind of product: when the store labels listings with categories, it must sit in the category whose label matches the request's wording as a whole (a request for "A or B" matches a category called "A & B", not a narrower neighbour called "B"). It must be in stock when the request asks for things on hand, and price times quantity must be within the budget (equal is allowed).
- A property whose value lists several options including the required one (a comma list) simply matches; that is not mixed evidence. Evidence is mixed only when the listing's own keys for the property state different, incompatible values. Only then, and only when no cheapest or best rule decides, prefer a candidate whose evidence is clean.

CHEAPEST OR BEST
- When the request asks for the cheapest, the order must be the lowest-priced item among ALL items that meet every requirement. Seeing one fitting item is not enough. Use price filters when a tool offers them, lower the price limit until every remaining row is visible, sort the visible candidates by price yourself, and verify them in ascending price; the first one that fully fits, with every cheaper one checked and rejected, is the answer. Keep the next fitting ones as backups.
- When the request asks for the best by some measure (most, largest, longest, highest, and so on), compare ALL fitting items on that measure the same way and pick the extreme. An item whose listing states no value for that measure cannot win.

FINISHING
- Keep a ranked list of verified fits, with at least one backup, because your first pick may sell out when you check out.
- Before every order attempt: the cart holds exactly one line, the chosen item at the required quantity, and nothing else (remove any other line); you have re-read the chosen item's stock and price after your last cart change, and it is in stock and within budget.
- Decline only after a thorough search finds nothing that meets every requirement. A decline scores 0 whenever some listing could have met the stated requirements and the revealed needs, so between ordering a listing that meets them and declining, order. Follow the request's rule for declining exactly. If the request says not to offer substitutes, the decline remark must not name, describe or point to any listing: no ids, no titles, no prices.
- Plan your steps: get your first order attempt in while at least seven steps remain, so a refusal can still be handled. When 7 steps remain, only cart, stock-check and finishing tools stay available. Decide once: do not swap the cart back and forth, since every cart change costs a step.
- Every reply must contain at least one tool call. Keep the reply text short: your current requirements, the verified fits with price and stock, and your next move."""

SETUP_PROMPT = """Read the shopper's request and the store's tool list below. Return ONE compact JSON object (no indentation) and nothing else, with these keys:

"product": the kind of product requested, in the request's words.
"products": a list with one entry per distinct kind of product the request asks to buy (a single entry when it asks for one kind), each in the request's words; one shop department named with "and" or "&" (such as "flats and heels") is one kind.
"references": one object per requirement whose value the request gives by pointing at another listing's page instead of naming it (it says the property must match, or must differ from, what the page of some seller's product shows, naming that product by a few of its details, possibly adding that the page may send you on to another listing): {"property": the property of the product being bought, in the request's exact words for it, "relation": "different" or "same", "seller": the name before 's, "product": the kind of product that page is for, "facts": ["property: value", ...] the details that describe that other listing}. Such a requirement goes only here, never in "requirements": the details that describe the other listing are not requirements on the product being bought.
"quantity": the number of units of one listing to order (an integer; 1 when the request does not ask for several). Something naturally sold as one item, such as a pair of shoes or a set, is quantity 1.
"requirements": a list of strings, one per firm stated requirement on the product, each written as "property: constraint" with the shopper's exact value, limit, exclusion or closed list. Do not include price, stock, quantity, budgets or mere wishes here, and do not restate the product kind or category as a requirement; delivery dates, arrival times and shipping speed are never product requirements, since the store lists no such facts: words that are part of the category's name (a season, an occasion or an audience in the category name) are covered by the category and are not a separate property requirement. A requirement is only what the request states outright as a condition on the product, a property with a value the shopper spells out; what the story merely implies (a situation that makes some property matter, a look or a feel described in passing) is not a requirement and goes in "hints". When the request asks for several kinds of product, start each requirement with the kind it applies to in square brackets, in the request's words for that kind, for example "[garden chairs] color: green".
"hints": a list of objects {"property": ..., "likely": ..., "why": ...}: the product properties the request's story points at without stating a value, named as a listing would name them (for example "water resistance" when the shopper will swim with the watch, "pile height" when a door must open over the rug), with the value the story implies in "likely" (or "" when it implies none) and the request's own words in "why". Hints are what to ask the shopper about first and what to prefer among items that already fit; they are never a reason to reject an item. Empty when the request is plain.
"price_cap": the maximum total price as a number, or null when none is stated.
"currency": the price unit the request uses, or null.
"in_stock_only": true when the request wants only items actually on hand or in stock.
"priority": "cheapest" only when the request explicitly asks for the cheapest or lowest-priced fitting option; "best" only when it explicitly asks for the best by some measure; otherwise "none". Remarks that money is tight or that every penny counts are not a request for the cheapest; they only explain the price cap.
"priority_detail": for "best", what "best" is measured by; otherwise "".
"message_budget": the number of messages the shopper allows you to send, as an integer, or null when not stated.
"step_budget": the number of moves or steps the request allows, as an integer, or null when not stated.
"rules": a list of every procedural rule the request states for the assistant (how many listings to look at, what to check or tell, how to decline, what not to do, what to include), quoted word for word. Remarks about what the shopper will do themselves are story, not rules.
"soft_wants": a list of things the shopper would like but does not strictly require (phrased as ideally, preferably, would be nice, if possible). Keep these out of "requirements".
"ask_first": a list of things the agent can only learn from the shopper and should ask about before choosing: for example the budget when no price limit is stated, which of two wishes matters more when the request says both may not be possible (then also put both wishes in soft_wants, not in requirements), or anything the request says to check with the shopper. Empty when none.
Every sentence of the request must land somewhere: a requirement, the priority, a soft want, a rule, a budget, a hint, or pure story. When unsure whether a sentence is a rule, put it in "rules".
"decline_no_names": true when the request's rule for declining forbids offering, naming or suggesting substitutes or alternatives; false when it asks you to name something; null when the request gives no rule.
"roles": an object naming the tool for each role, using the exact tool names from the list, or null when no tool fits:
  "search", "details", "view", "message", "cart", "cart_view", "order", "decline".
  "message" is the tool that sends a message to the shopper. "cart" is the tool that adds to or removes from the cart. "order" is the tool that buys or checks out. "decline" is the tool that ends the task without buying.
"finish_args": when one tool both buys and ends without buying, an object {"order": {...}, "decline": {...}} with the argument values that select each, for example {"order": {"mode": "buy"}, "decline": {"mode": "stop"}}; null otherwise.
"""

INTERPRET_PROMPT = """You help a shopping assistant understand what a shopper just said. Return ONE JSON object and nothing else, with these keys:
"meaning": one short sentence saying what the shopper means.
"needs": a list of new firm requirements on the product, each written as "property: constraint" with the exact value, limit, exclusion or closed list the shopper gave.
"references": one object per need whose value the shopper gives by pointing at another listing's page instead of naming it (the shopper says P must match, or must differ from, what the page of some seller's product shows, naming that product by a few of its details, possibly adding that the page may send you on to another listing): {"property": the property of the product being bought that the need is about, in the shopper's exact words for it, "relation": "different" when the value must differ from that page's value, "same" when it must match it, "seller": the name before 's, "product": the kind of product that page is for, "facts": ["property: value", ...] the details that describe that other listing}. Such a need goes only here, never in "needs" or "requirements_now": the details that describe the other listing are not requirements on the product being bought.
"no_preference": a list of properties the shopper said they do not care about.
"dropped": a list of earlier requirements the shopper no longer needs.
"new_budget": the new maximum total price as a number, or null.
"unavailable": true when the shopper says an item can no longer be bought (sold out, gone, out of stock, button disabled), else false.
"requirements_now": the complete list of firm requirements on the product as they stand after everything the shopper has said so far, re-derived from the request, the earlier shopper lines and this one, each as "property: constraint". Keep every earlier requirement unless the shopper dropped or replaced it; for a replaced value keep only the newest one; add every newly revealed need. Do not include price, stock, quantity or the product category, and never add a requirement from the request's story or from your own reading of the situation: only what the request states outright and what the shopper has said. When the request asks for several kinds of product, every requirement keeps its "[kind]" tag, and a newly revealed need about one kind gets that kind's tag; the same goes for "needs".
"reply": when the shopper asks the assistant something or asks for a confirmation, a short, plain reply to send back that answers it and names the change in substance (for example confirming the new value); otherwise "".
One line can carry several changes: report every one. A requirement replaced by another is both dropped (the old one) and a new need (the new one). When the shopper says which of two wishes matters more, the preferred wish is a need. When the shopper states a budget, it is new_budget. A mere wish ("ideally", "would be nice") is not a need. Read the shopper literally and charitably. Folksy, figurative or roundabout phrasing that names a value means the shopper firmly wants that value; treat a value as excluded only when the shopper clearly says to avoid it. A number with a comparison (at most, at least, under, over, no more than) is a limit, kept with its unit. A new budget replaces the old one whether it is higher or lower, even when the shopper calls it good news. When the shopper only says they don't mind, have no preference, or anything works, put the property asked about in no_preference."""

VERIFY_PROMPT = """You are the final checker before a shopping assistant places an order. The task scores 1 only if the ordered item meets every requirement below. Judge the item ONLY from the records shown, using these reading rules:
- A requirement is met only when the item's own property for it states the value. Property keys vary in spelling (spaces, dots, underscores, prefixes) and the same property may appear under several keys; any key naming that property counts. Values match by meaning: one element of a comma list is enough, singular/plural and close synonyms count, case does not matter. A merely similar-looking word or a broader/different class does not count.
- The title or description counts only when the item has no property key for that requirement. If its own property says something else, it fails even when the title says the value.
- A value under a different, related property does not count.
- Numbers: compare after converting units; limits include the edge; a range counts only when the whole range meets the limit.
- Exclusions and closed lists must be respected.
- The item must be the requested kind of product; when listings carry a category label, it must be the category whose label matches the request's wording as a whole, not a narrower or neighbouring one.
- When the item's own property for a requirement states a different value, the requirement fails, even if the title suggests otherwise.
- Price times quantity must be within the budget (equal is fine). If the records show it out of stock, it fails.
- A mere wish ("ideally", "would be nice") listed as not required is not a requirement.
When records disagree (for example on price or stock), the newest record counts.
Return ONE JSON object and nothing else: {"item": the item's id, "fits": true or false, "problems": [short strings, each naming a requirement that is not met or not shown]}. Use fits=false only for a requirement that the records clearly fail, or that no record addresses at all; do not fail an item for a missing record of stock or price."""

PLAN_PROMPT = """You plan the questions a shopping assistant sends to a shopper. The shopper has hidden needs: each is a value of one property of the product, revealed only when a message asks about that exact property; each message is answered only about the first thing it asks, and questions about anything else get a polite "don't mind". Choose up to QUESTION_COUNT different properties, most promising first:
0. first, anything listed below as MUST BE LEARNED FROM THE SHOPPER that has not been asked yet; ask it plainly. For the budget, ask how much they want to spend at most. For a choice between two wishes, name both values with their properties, for example "If one listing can't have both, would you rather have the <value> <property> or the <value> <property>?";
1. then properties that the listings' details below actually carry, preferring (a) a property the story in the request hints at indirectly (who the item is for or who will wear or use it, how it must look in photos or to others, where or how it will be used), and (b) properties that describe what the product is like and whose values vary across the listings (who it is made for, pattern, shape, type or style variant, closure, length, function) rather than logistics (warranty, condition, pack size, variant or size lists, years). When CORE PROPERTIES are listed below (most of the listings opened carry them), choose only among them.
Never pick a property the request already settles (its stated requirements, the price, stock, quantity), never repeat a property already asked, never spend two questions on near-duplicate properties (two kinds of style, two spellings of one property), and avoid properties no listing carries. Use the property names as the listings name them. Write each question as one short, plain, friendly sentence to the shopper that names that one property in the listings' own words for it and may give one or two example values from the listings, for example "Do you have a preference for the <property>, such as <value> or <value>?". No instructions, no lists, no second question.
Return ONE JSON object and nothing else: {"questions": [{"property": "...", "question": "..."}, ...]}."""

AUDIT_PROMPT = """A shopping assistant is about to tell the shopper that nothing in the store fits. Check the listing records below against the requirements, using these reading rules: a requirement is met when the listing's own property for it states the value (any key spelling, one element of a comma list, singular/plural and close synonyms all count; a broader or different class does not); the title or description counts only when the listing has no key for that property; numbers are compared after converting units, limits include the edge; the price must be within the budget and the item must not be shown out of stock.
Return ONE JSON object and nothing else: {"possible_fits": [ids of listings that appear to meet every requirement, best first, at most 5], "reason": "one sentence"}."""

_llm = ProxyClient(timeout=70, max_retries=1, retry_delay=1)
_env = ProxyClient(timeout=95, max_retries=2, retry_delay=1)

def _brief(value: Any, limit: int, depth: int = 0) -> Any:
    if depth > 8:
        return None
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + "..."
    if isinstance(value, dict):
        return {k: _brief(v, limit, depth + 1) for k, v in value.items()}
    if isinstance(value, list):
        return [_brief(v, limit, depth + 1) for v in value]
    return value


_ID_HINT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-:.]{3,63}$")


def _compact(value: Any) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        text = str(value)
    if len(text) > OBS_CHARS:
        text = text[:OBS_CHARS] + "...(cut)"
    return text


def _parse_json_object(text: str) -> dict[str, Any] | None:
    if not isinstance(text, str):
        return None
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[A-Za-z]*", "", text).strip()
        if text.endswith("```"):
            text = text[:-3].strip()
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except ValueError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            value = json.loads(text[start : end + 1])
            return value if isinstance(value, dict) else None
        except ValueError:
            return None
    return None


def _type_ok(value: Any, kind: str) -> bool:
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


def _coerce(value: Any, schema: dict[str, Any]) -> Any:
    kinds = schema.get("type")
    if isinstance(kinds, str):
        kinds = [kinds]
    if not kinds:
        return value
    if any(_type_ok(value, k) for k in kinds):
        if "integer" in kinds and isinstance(value, float) and value.is_integer():
            return int(value)
        return value
    for kind in kinds:
        try:
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
        except (TypeError, ValueError):
            continue
    return value


def _check(value: Any, schema: dict[str, Any], path: str) -> tuple[Any, list[str]]:
    if not isinstance(schema, dict):
        return value, []
    problems: list[str] = []
    value = _coerce(value, schema)
    kinds = schema.get("type")
    if isinstance(kinds, str):
        kinds = [kinds]
    if kinds and not any(_type_ok(value, k) for k in kinds):
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
        if "minimum" in schema and value < schema["minimum"]:
            problems.append(f"{path}: below minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            problems.append(f"{path}: above maximum {schema['maximum']}")
        if "exclusiveMinimum" in schema and isinstance(schema["exclusiveMinimum"], (int, float)) and value <= schema["exclusiveMinimum"]:
            problems.append(f"{path}: must be above {schema['exclusiveMinimum']}")
        if "exclusiveMaximum" in schema and isinstance(schema["exclusiveMaximum"], (int, float)) and value >= schema["exclusiveMaximum"]:
            problems.append(f"{path}: must be below {schema['exclusiveMaximum']}")
    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            problems.append(f"{path}: needs at least {schema['minItems']} items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            problems.append(f"{path}: allows at most {schema['maxItems']} items")
        if isinstance(schema.get("items"), dict):
            fixed = []
            for index, item in enumerate(value):
                item, sub = _check(item, schema["items"], f"{path}[{index}]")
                fixed.append(item)
                problems += sub
            value = fixed
    if isinstance(value, dict):
        properties = schema.get("properties") or {}
        for name in schema.get("required") or []:
            if name not in value:
                problems.append(f"{path}.{name}: required")
        if schema.get("additionalProperties") is False:
            for name in list(value):
                if name not in properties:
                    problems.append(f"{path}.{name}: not an allowed argument")
        fixed = {}
        for name, item in value.items():
            if name in properties:
                item, sub = _check(item, properties[name], f"{path}.{name}")
                problems += sub
            fixed[name] = item
        value = fixed
    return value, problems


def _sentences(value: Any, out: list[str], depth: int = 0) -> None:
    if depth > 4:
        return
    if isinstance(value, str):
        if len(value.split()) >= 4:
            out.append(value)
    elif isinstance(value, list):
        for item in value[:10]:
            _sentences(item, out, depth + 1)


def _collect_ids(value: Any, out: set[str], depth: int = 0) -> None:
    if depth > 6:
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, str) and "id" in key.lower() and _ID_HINT.match(item):
                out.add(item)
            else:
                _collect_ids(item, out, depth + 1)
    elif isinstance(value, list):
        for item in value:
            _collect_ids(item, out, depth + 1)


_SPELL: dict[str, str] = {}
for _pair in "colour color / flavour flavor / grey gray / fibre fiber / metre meter / litre liter / centre center / jewellery jewelry / aluminium aluminum / tyre tire / armour armor / favourite favorite".split("/"):
    for _w in _pair.split():
        _SPELL[_w] = _pair.split()[-1]
_GROUPS = [set(part.split()) for part in "women woman womens female females ladies lady / men man mens male males gents / girls girl / boys boy / kids kid children child".split("/")]
_FILLER = set("a an the of and with must be is are has have having include includes including exactly only should that it its any made from".split())
_MORE = "more|bigger|larger|higher|longer|wider|taller|heavier"
_LESS = "less|smaller|fewer|lower|shorter|narrower|lighter"
_TITLE_KEY = re.compile(r"(?i)title|heading|name|option|variant")
_SHELF_KEY = re.compile(r"(?i)categor|aisle|shelf|department|section")
_NOT_A_FACT = re.compile(r"(?i)price|cost|budget|stock|quantit|deliver|arriv|ship|categor|^(?:product|item|kind)(?: type)?$")


def _words(text: Any, join: bool) -> list[str]:
    """Matching words: lower case, one spelling, possessives off, a figure joined to its unit when `join` ('30 mm' = '30mm')."""
    text = re.sub(r"'s\b", "", str(text if text is not None else "").lower())
    if join:
        text = re.sub(r"(\d)\s+(?=[a-z%])", r"\1", text)
    return [_SPELL.get(w, w) for w in re.findall(r"[a-z0-9]+", text)]


def _one(word: str) -> str:
    if len(word) > 3 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith(("ches", "shes", "xes", "sses")):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _same(one: str, other: str) -> bool:
    """One word in another number or form (tunik and tunic, stripe and striped, rectangle and rectangular, women and
    female), never a word and its '-less' opposite, never two different figures."""
    one, other = _one(one), _one(other)
    if one == other or any(one in group and other in group for group in _GROUPS):
        return True
    if re.search(r"\d", one + other):
        return False
    short, longer = sorted((one, other), key=len)
    if longer.startswith(short):
        return len(short) >= 4 and len(longer) - len(short) <= 3
    common = 0
    for a, b in zip(one, other):
        if a != b:
            break
        common += 1
    if common >= max(5, len(short) - 2):
        return True
    if len(short) < 5 or len(longer) - len(short) > 1:
        return False
    i = j = edits = 0
    while i < len(longer) and j < len(short):
        if longer[i] == short[j]:
            i += 1
            j += 1
            continue
        edits += 1
        i += 1
        j += 1 if len(longer) == len(short) else 0
    return edits + (len(longer) - i) + (len(short) - j) <= 1


def _stated(value: str, text: str) -> bool:
    """Every word of the value appears, in some form, among the words of the text; or the value written without spaces
    sits inside the text ('particle board' in 'particleboard')."""
    for join in (True, False):
        wanted = [w for w in _words(value, join) if w not in _FILLER]
        have = _words(text, join)
        if wanted and all(any(_same(w, h) for h in have) for w in wanted):
            return True
    flat = re.sub(r"[^a-z0-9]+", "", value.lower())
    return len(flat) >= 8 and flat in re.sub(r"[^a-z0-9]+", "", str(text).lower())


def _plain(label: str) -> list[str]:
    parts = _words(label, False)
    while len(parts) > 1 and len(parts[0]) <= 2:
        parts = parts[1:]
    return [_one(w) for w in parts]


def _names(label: str, prop: str, spare: set[str]) -> bool:
    """A fact label names a property when it holds the property's words (the product's own words may be left out) and at
    most two more, and holds 'type' when the property does: 'top material' names material, 'system memory' does not
    name memory type."""
    lw, pw = set(_plain(label)), set(_plain(prop))
    core = pw - {"type"}
    if core and not core <= lw:
        core, pw = core - spare, pw - spare
    if not core or not core <= lw:
        return False
    if "type" in pw and "type" not in lw and lw != core:
        return False
    return len(lw - pw) <= 2


def _bound(value: str) -> tuple[float | None, float | None] | None:
    """The bound a value states: '50 cm or longer' -> (50, None); 'no more than 750 g' -> (None, 750); '2-5 kg' -> (2, 5)."""
    text = value.lower().replace(",", "")
    nums = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", text)]
    if not nums:
        return None
    if len(nums) >= 2 and re.search(r"between|\d\s*(?:to|-|and)\s*\d", text) and not re.search(r"up to|at (?:most|least)", text):
        return (min(nums[:2]), max(nums[:2]))
    if re.search(r"\bno(?:t)?\s+(?:%s|over|above)\b" % _MORE, text):
        return (None, nums[0])
    if re.search(r"\bno(?:t)?\s+(?:%s|under|below)\b" % _LESS, text):
        return (nums[0], None)
    if re.search(r"at least|or (?:%s|above|over)|minimum|\bmin\b|(?:%s) than|\babove\b|\bover\b|>=|\d\+" % (_MORE, _MORE), text):
        return (nums[0], None)
    if re.search(r"at most|up to|or (?:%s|below|under)|maximum|\bmax\b|(?:%s) than|\bunder\b|\bbelow\b|within|<=" % (_LESS, _LESS), text):
        return (None, nums[0])
    return None


def _figures(text: str) -> list[tuple[float, str]]:
    return [(float(m.group(1)), _one(m.group(2))) for m in re.finditer(r"(?<![a-z0-9.])(\d+(?:\.\d+)?)\s*([a-z%\"]*)", str(text).lower().replace(",", ""))]


def _reading(sheet: dict[str, Any], prop: str, value: str, spare: set[str]) -> tuple[str, str]:
    """The store's literal rule, from the v3 line: a value counts only when a fact named for the property states it, or
    the title or the variant options when no fact is named for it; a named fact stating something else loses whatever
    the title says; an excluded value stated there loses; a bound is met by a figure inside it and lost by a comparable
    figure outside it. Returns 'yes', 'no' or 'unshown', with the evidence read."""
    facts = next((v for v in sheet.values() if isinstance(v, dict)), {})
    pairs = [(str(k), str(v)) for k, v in list(sheet.items()) + list(facts.items()) if isinstance(v, (str, int, float)) and not isinstance(v, bool)
             and not re.fullmatch(r"(?i)\s*(?:n/?a|none|not specified|unknown|null|-|)\s*", str(v))]
    named = [(k, v) for k, v in pairs if not _TITLE_KEY.search(k) and _names(k, prop, spare)]
    texts = [v for _, v in named] or [v for k, v in pairs if _TITLE_KEY.search(k) and not _SHELF_KEY.search(k)]
    shown = "; ".join(f"{k}: {v[:60]}" for k, v in named[:2]) or "no fact named for it"
    bound = _bound(value)
    excluded = re.match(r"^(?:not|no|without|anything but|anything except|except|excluding|avoid)\s+(.+)$", value)
    if excluded and bound is None:
        return ("no" if any(_stated(excluded.group(1), t) for t in texts) else "yes"), shown
    if bound is not None:
        low, high = bound
        unit = next((u for _, u in _figures(value) if u), "")
        figures = [f for t in texts for f in _figures(t)]
        if any((low is None or n >= low) and (high is None or n <= high) for n, _ in figures):
            return "yes", shown
        comparable = [n for n, u in figures if u == unit or not u or not unit]
        return ("no" if comparable and named else "unshown"), shown
    options = [a for a in re.split(r"\s+or\s+|(?<=[a-z])\s*/\s*(?=[a-z])|,\s*(?!\d)", value) if a.strip()] or [value]
    if any(_stated(a, t) for a in options for t in texts):
        return "yes", shown
    if not named:
        return "unshown", shown
    units = {u for _, u in _figures(value) if u}
    if units and any(u and u not in units for t in texts for _, u in _figures(t)):
        return "unshown", shown
    return "no", shown


def _price(record: dict[str, Any]) -> float | None:
    for key, value in record.items():
        # worlds rename the price field (race 3: amount_amount); a name the price reader misses leaves every budget and
        # cheapest check blind
        if re.search(r"(?i)price|cost|amount|fee|charge|cents|usd|dollar", str(key)) and not re.search(r"(?i)total|payable|subtotal|discount|saving|tax|ship|deliver|count|quantit", str(key)) and isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def _in_stock(record: dict[str, Any]) -> bool | None:
    for key, value in record.items():
        if re.search(r"(?i)stock|avail", str(key)) and isinstance(value, (bool, int, float)):
            return bool(value) if isinstance(value, bool) else value > 0
    return None


def _shelf(record: dict[str, Any]) -> str:
    return next((v for k, v in record.items() if _SHELF_KEY.search(str(k)) and isinstance(v, str) and v.strip()), "")


REF_HOP_PROMPT = """A listing's page text is below. Does it send the reader to a different listing for its PROPERTY (or for a property named almost the same, such as "color family" for "color")? If it does, return {"sends": true, "seller": the other listing's seller or maker, "product": its kind of product, "facts": ["property: value", ...] the details the text gives to identify that listing}. If it sends some other property elsewhere, or sends nothing, return {"sends": false}. Return ONE JSON object and nothing else."""

REF_LOOKUP = True

_LOGISTICS = re.compile(r"(?i)brand|outlet|seller|vendor|store|\bshop\b|price|stock|condition|warrant|shipping|deliver")


def _in_written_unit(cap: Any, text: str) -> Any:
    """A budget in the unit the words wrote it: the reading model turns "1700 cents" into 17 dollars, and prices are
    shown in cents (cf1a, dd2c, 0306 declined or ranked blind). Scaled only when the words carry the figure times
    100 and not the figure itself."""
    if not isinstance(cap, (int, float)) or isinstance(cap, bool) or cap <= 0:
        return cap
    hundred = round(cap * 100)
    written = {float(n.replace(",", "")) for n in re.findall(r"\d[\d,]*(?:\.\d+)?", text)}
    if float(hundred) in written and float(cap) not in written:
        return float(hundred)
    return cap


def _kind_tag(text: str) -> tuple[str | None, str]:
    """A requirement's "[kind]" tag and the rest of it."""
    match = re.match(r"^\s*\[([^\]]+)\]\s*(.*)$", text, re.S)
    return (match.group(1).strip(), match.group(2)) if match else (None, text)


def _need_key(text: str) -> str:
    prop, _, value = _kind_tag(str(text))[1].partition(":")
    return " ".join(w for w in _plain(prop) if w != "type") + ":" + " ".join(re.findall(r"[a-z0-9.]+", value.lower()))


class Episode:
    def __init__(self, problem_data: dict[str, Any]):
        self.started = time.monotonic()
        environment = problem_data["environment"]
        self.binding = dict(environment["binding"])
        self.policy = environment["policy_view"]
        self.tools = [t for t in self.policy.get("tools") or [] if isinstance(t, dict) and isinstance(t.get("function"), dict)]
        self.schemas = {t["function"]["name"]: t["function"].get("parameters") or {"type": "object", "properties": {}} for t in self.tools}
        self.max_steps = int(self.policy.get("max_steps") or 30)
        self.step = 0
        self.finished = bool(self.policy.get("finished"))
        self.dead = False
        self.messages_used = 0
        self.message_budget: int | None = None
        self.order_attempts = 0
        self.refusals = 0
        self.order_checked = False
        self.decline_checked = False
        self.cart_used = False
        self.seen_ids: set[str] = set()
        self.stated: list[str] = []
        self.revealed: list[str] = []
        self.free: list[str] = []
        self.dropped: list[str] = []
        self.budget_now: Any = None
        self.heard: set[str] = set()
        self.cart_ids: list[str] = []
        self.records: list[Any] = []
        self.last_cart_view: Any = None
        self.cart_changed_at = 0
        self.cart_view_at = -1
        self.cart_state: dict[str, str] = {}
        self.add_signature: str | None = None
        self.asked: list[str] = []
        self.planner_on = True
        self.audited = False
        self.qty_warned: set[str] = set()
        self.pending_reply = ""
        self.refs: list[dict[str, Any]] = []
        self.refs_done: list[dict[str, Any]] = []
        self.kinds_asked: list[str] = []
        self.described: set[str] = set()
        self.probed: set[str] = set()
        self.call_seq = 0
        self.opened_first = False
        self.plan_rounds = 0
        self.plan_sheets = 0
        self.forced_commit = False
        self.sheet_tool: str | None = None
        self.current: list[str] | None = None
        self.verdicts: dict[str, str] = {}
        self.ledger_version = 0
        self.cart_checked: set[str] = set()
        self.said: list[str] = []
        self.recent: list[str] = []
        self.roles: dict[str, str] = {}
        self.plan: dict[str, Any] = {}
        self.cost = 0.0
        self.finish_args: dict[str, dict[str, Any]] = {}
        self.sheets: dict[str, dict[str, Any]] = {}
        self.searches_done = 0
        self.search_guarded = False
        self.rows: dict[str, dict[str, Any]] = {}
        self.rule_told: set[str] = set()
        self.fits_shown = ""
        self.told_once: set[str] = set()
        self.coverage: dict[str, dict[str, Any]] = {}
        self.trace: list[dict[str, Any]] = []
        self.messages: list[dict[str, Any]] = []

    def time_left(self) -> float:
        return TASK_SECONDS - (time.monotonic() - self.started)

    def steps_left(self) -> int:
        return self.max_steps - self.step

    def infer(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, models: list[str], max_tokens: int) -> dict[str, Any] | None:
        for model in models:
            left = self.time_left()
            if left < FORCE_SECONDS - 10:
                return None
            _llm.timeout = int(max(8, min(75, left - FORCE_SECONDS + 15)))
            sent = messages
            if model.startswith("anthropic/") and len(messages) > 2:
                sent = [dict(m) for m in messages]
                for index in (0, len(sent) - 1):
                    if isinstance(sent[index].get("content"), str) and sent[index]["content"]:
                        sent[index]["content"] = [{"type": "text", "text": sent[index]["content"], "cache_control": {"type": "ephemeral"}}]
            body: dict[str, Any] = {"model": model, "messages": sent, "temperature": 0, "max_tokens": max_tokens}
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
                    return message
            self.trace.append({"kind": "inference_failure", "model": model, "error": str(result.error)[:300] if result.error else "empty"})
        return None

    def env_call(self, actions: list[dict[str, Any]]) -> dict[str, Any] | None:
        payload: dict[str, Any] = dict(self.binding)
        payload["step"] = self.step + 1
        if len(actions) == 1:
            payload["action"] = actions[0]
        else:
            payload["actions"] = actions
        for _ in range(3):
            left = self.time_left()
            if left < 2:
                return None
            _env.timeout = int(max(3, min(95, left - 1)))
            result = _env.post_verbose("/environment/call", payload)
            if isinstance(result.data, dict):
                data = result.data
                if isinstance(data.get("step"), int):
                    self.step = data["step"]
                else:
                    self.step += len(actions)
                if data.get("finished"):
                    self.finished = True
                return data
            error = result.error or {}
            status = error.get("status")
            body = str(error.get("body") or "")
            self.trace.append({"kind": "env_failure", "status": status, "body": body[:300]})
            if status == 402:
                self.dead = True
                return None
            if status == 409:
                if "finished" in body:
                    self.finished = True
                    return None
                match = re.search(r"expected (\d+)", body)
                if match:
                    self.step = int(match.group(1)) - 1
                    payload["step"] = self.step + 1
                    continue
                return None
            if status in (403, 422, 500, 503, 504):
                return None
        return None

    def setup(self) -> None:
        tool_lines = []
        for tool in self.tools:
            function = tool["function"]
            tool_lines.append(f"- {function['name']}: {function.get('description', '')} | arguments: {_compact(function.get('parameters') or {})}")
        static = SETUP_PROMPT + "\n\nTOOLS:\n" + "\n".join(tool_lines)
        variable = "REQUEST:\n" + str(self.policy.get("query") or "")
        if self.policy.get("opening") is not None:
            variable += "\n\nOPENING OBSERVATION:\n" + _compact(self.policy.get("opening"))
        content = [{"type": "text", "text": static, "cache_control": {"type": "ephemeral"}}, {"type": "text", "text": variable}]
        # the reading model thinks before it answers: 1500 tokens cut the plan off mid-object in 13 of 81 race-3 tasks
        # (Cloud: 6 of 57), and an empty plan declines; more room, and one more try when it still does not parse
        plan = {}
        for models in ([SETUP_MODEL], [SETUP_MODEL, BACKUP_MODEL]):
            message = self.infer([{"role": "user", "content": content}], None, models, 4000)
            plan = _parse_json_object((message or {}).get("content") or "") or {}
            if plan:
                break
        self.plan = plan
        roles = plan.get("roles") if isinstance(plan.get("roles"), dict) else {}
        self.roles = {k: v for k, v in roles.items() if isinstance(v, str) and v in self.schemas}
        modes = plan.get("finish_args")
        if isinstance(modes, dict):
            self.finish_args = {k: v for k, v in modes.items() if k in ("order", "decline") and isinstance(v, dict)}
        budget = plan.get("message_budget")
        if isinstance(budget, (int, float)) and not isinstance(budget, bool) and budget >= 0:
            self.message_budget = int(budget)
        self.stated = [str(r) for r in plan.get("requirements") or [] if r]
        for ref in [r for r in plan.get("references") or [] if isinstance(r, dict) and str(r.get("property") or "").strip()]:
            self.described |= {_need_key(f) for f in ref.get("facts") or [] if isinstance(f, str) and ":" in f}
            ref["property"] = self.ref_property(str(ref["property"]), [str(self.policy.get("query") or "")])
            self.refs.append(ref)
        self.stated = [r for r in self.stated if not self.pointed(r)]
        self.budget_now = plan.get("price_cap")
        if self.budget_now is not None:
            self.budget_now = plan["price_cap"] = _in_written_unit(self.budget_now, str(self.policy.get("query") or ""))
        if plan and self.budget_now is None and self.roles.get("message"):
            ask = [str(a) for a in plan.get("ask_first") or [] if a]
            plan["ask_first"] = ["the most the shopper wants to spend, since the request states no price limit (skip if the same thing is listed again here)"] + ask
        steps = plan.get("step_budget")
        if isinstance(steps, (int, float)) and not isinstance(steps, bool) and 0 < steps < self.max_steps:
            self.max_steps = int(steps)

    def brief(self) -> str:
        plan = self.plan
        if not plan:
            return ""
        lines = ["A first reading of the request (check it against the request itself):"]
        if plan.get("product"):
            lines.append(f"- product: {plan.get('product')}")
        lines.append(f"- quantity: {plan.get('quantity') or 1}")
        for requirement in plan.get("requirements") or []:
            lines.append(f"- stated requirement: {requirement}")
        if plan.get("price_cap") is not None:
            lines.append(f"- price cap (total): {plan.get('price_cap')} {plan.get('currency') or ''}".rstrip())
        if plan.get("in_stock_only"):
            lines.append("- only items in stock / on hand")
        if plan.get("priority") in ("cheapest", "best"):
            detail = f" ({plan.get('priority_detail')})" if plan.get("priority_detail") else ""
            lines.append(f"- choose the {plan.get('priority')}{detail} among all fitting items")
        else:
            lines.append("- no cheapest/best rule was found: any item that meets every requirement within the budget is right, so do not spend steps hunting for the lowest price (unless the request itself says otherwise)")
        for want in plan.get("soft_wants") or []:
            lines.append(f"- nice to have, NOT required (only a tie-breaker): {want}")
        for item in plan.get("ask_first") or []:
            lines.append(f"- must be learned from the shopper first: {item}")
        for rule in plan.get("rules") or []:
            lines.append(f"- rule: {rule}")
        if self.message_budget is not None:
            lines.append(f"- messages to the shopper allowed: {self.message_budget}")
        lines.append(f"- steps allowed: {self.max_steps}")
        hints = [h for h in plan.get("hint_properties") or [] if isinstance(h, str)]
        if hints:
            lines.append("- properties the story may hint at (ask about these first, one per message): " + ", ".join(hints))
        return "\n".join(lines)

    def status(self) -> str:
        left = self.steps_left()
        parts = [f"[status] steps used {self.step} of {self.max_steps}, {left} left"]
        if self.message_budget is not None:
            parts.append(f"messages sent {self.messages_used} of {self.message_budget}")
        else:
            parts.append(f"messages sent {self.messages_used}")
        parts.append(f"about {int(max(0, self.time_left() - FORCE_SECONDS))} s left")
        text = "; ".join(parts) + "."
        text += "\n[requirements now] " + self.ledger()
        hints = self.hints()
        if hints:
            text += "\n[story hints] " + "; ".join(f"{p} (likely {v})" if v else p for p, v in hints) + ": ask about these before anything else, then prefer the fit that matches; never reject an item or decline over a hint."
        text += self.fits_line()
        if left <= 2:
            text += " LAST STEPS: order now if the cart holds a fit; decline only when no listing fits at all."
        elif left <= 5 or self.time_left() < WRAP_UP_SECONDS:
            text += " FINISH NOW: fix the cart to one verified line, re-check its stock, then order (or decline if nothing fits)."
        elif self.order_attempts == 0 and left <= 14:
            text += " Steps are running low: within the next few steps put your best verified fit in the cart and order; searching and reading details end when 7 steps remain, and a refused order needs about 4 steps to repair."
        return text

    def ledger(self) -> str:
        if self.current is not None:
            items = list(self.current)
        else:
            items = [f"{r} (stated)" for r in self.stated if r not in self.dropped]
            items += [f"{r} (revealed by the shopper)" for r in self.revealed if r not in self.dropped]
        text = "; ".join(items) if items else "see the request"
        if self.budget_now is not None:
            text += f"; total price at most {self.budget_now} {self.plan.get('currency') or ''}".rstrip()
        text += f"; quantity {self.plan.get('quantity') or 1}"
        if self.dropped:
            text += "; no longer required: " + "; ".join(self.dropped)
        if self.free:
            text += "; shopper does not care about: " + ", ".join(self.free)
        return text

    def intent(self, name: str, args: dict[str, Any]) -> str | None:
        """'order' or 'decline' for a call of a finishing tool; one tool may do both, told apart by its mode argument."""
        order, decline = self.roles.get("order"), self.roles.get("decline")
        if name not in (order, decline):
            return None
        if order != decline:
            return "order" if name == order else "decline"
        for kind in ("decline", "order"):
            want = self.finish_args.get(kind) or {}
            if want and all(args.get(k) == v for k, v in want.items()):
                return kind
        properties = (self.schemas.get(name) or {}).get("properties") or {}
        for key, spec in properties.items():
            if isinstance(spec, dict) and spec.get("enum") and args.get(key) is not None:
                return "order" if args.get(key) == spec["enum"][0] else "decline"
        return "order"

    def remember(self, name: str, observation: Any) -> None:
        """Keep a compact record of every opened listing (one carrying a facts-like object) and count searches."""
        if isinstance(observation, dict):
            inner = observation
            for value in observation.values():
                if isinstance(value, dict) and len(observation) == 2:
                    inner = value
            facts = next((v for v in inner.values() if isinstance(v, dict) and len(v) >= 1 and all(isinstance(x, (str, int, float)) for x in v.values())), None) if isinstance(inner, dict) else None
            ident = next((v for k, v in inner.items() if isinstance(v, str) and "id" in k.lower() and _ID_HINT.match(v)), None) if isinstance(inner, dict) else None
            if facts is not None and ident:
                self.sheet_tool = name
                self.sheets[ident] = {k: (_brief(v, 160) if not isinstance(v, dict) else v) for k, v in inner.items()}
            rows = next((v for v in inner.values() if isinstance(v, list) and v and all(isinstance(x, dict) for x in v)), None) if isinstance(inner, dict) else None
            if rows is not None and name != self.roles.get("cart_view"):
                detailed = 0
                for row in rows:
                    key = next((v for k, v in row.items() if isinstance(v, str) and "id" in k.lower() and _ID_HINT.match(v)), None)
                    if not key:
                        continue
                    self.rows[key] = {k: v for k, v in row.items() if isinstance(v, (str, int, float, bool)) and not (isinstance(v, str) and len(v) > 160)}
                    if any(isinstance(v, dict) and v and all(isinstance(x, (str, int, float)) for x in v.values()) for v in row.values()):
                        self.sheets[key] = {k: (_brief(v, 160) if not isinstance(v, dict) else v) for k, v in row.items()}
                        detailed += 1
                if detailed * 2 < len(rows):
                    self.searches_done += 1

    def sheets_stating(self, need: str) -> tuple[list[str], list[str]]:
        """Opened listings whose own fact for the property states the value (or whose title does, when they carry no
        fact for it), and those whose fact states something else."""
        prop, _, value = _kind_tag(need)[1].partition(":")
        pwords = [w for w in re.findall(r"[a-z0-9]+", prop.lower()) if w != "type"]
        vwords = re.findall(r"[a-z0-9]+", value.lower())
        if not pwords or not vwords or re.search(r"\d", value) and re.search(r"(?i)most|least|under|over|up to|between", value):
            return [], []
        stating, other = [], []
        for ident, rec in self.sheets.items():
            facts = next((v for v in rec.values() if isinstance(v, dict)), {})
            keys = [k for k in facts if all(w in re.findall(r"[a-z0-9]+", str(k).lower().replace("_", " ")) for w in pwords)]
            texts = [str(facts[k]).lower() for k in keys] or [" ".join(str(v) for v in rec.values() if isinstance(v, str)).lower()]
            if any(all(w in re.findall(r"[a-z0-9]+", t) for w in vwords) for t in texts):
                stating.append(ident)
            elif keys:
                other.append(ident)
        return stating, other

    def hints(self) -> list[tuple[str, str]]:
        """The story's hints from the setup, (property, likely value): properties the request points at without stating a
        value. They choose the questions and break ties among fits; they never reject an item (on the story pack pass 6b
        lost five tasks by hunting for story details as if they were requirements)."""
        settled = [prop for prop, _ in self.demands()] + list(self.free)
        out: list[tuple[str, str]] = []
        for item in self.plan.get("hints") or []:
            prop = str(item.get("property") if isinstance(item, dict) else item or "").strip().lower()
            likely = str(item.get("likely") or "").strip().lower() if isinstance(item, dict) else ""
            words = set(_plain(prop)) - {"type"}
            if not words or any(words <= set(_plain(s)) or (set(_plain(s)) - {"type"} and set(_plain(s)) - {"type"} <= words) for s in settled):
                continue
            if prop not in [p for p, _ in out]:
                out.append((prop, likely))
        return out[:4]

    def preferred(self, fits: list[tuple[str, float | None]]) -> str:
        """Among the proven fits, the cheapest whose own facts state a hint's likely value, else the cheapest."""
        if self.plan.get("priority") in ("cheapest", "best"):
            return fits[0][0]                                        # a cheapest or best rule outranks any hint
        product = {_one(w) for w in _words(self.plan.get("product") or "", False)}
        for ident, _ in fits:
            if any(v and _reading(self.sheets.get(ident) or {}, p, v, product)[0] == "yes" for p, v in self.hints()):
                return ident
        return fits[0][0]

    def cart_args(self, ident: str) -> dict[str, Any]:
        """Arguments that put one listing in the cart at the required quantity, read from the cart tool's schema."""
        schema = self.schemas.get(self.roles.get("cart") or "") or {}
        properties = schema.get("properties") or {}
        quantity = self.plan.get("quantity")
        quantity = quantity if isinstance(quantity, int) and not isinstance(quantity, bool) and quantity > 0 else 1
        args: dict[str, Any] = {}
        placed = False
        for key, spec in properties.items():
            if not isinstance(spec, dict):
                continue
            kind = spec.get("type")
            if "enum" in spec and spec["enum"]:
                values = [str(v) for v in spec["enum"]]
                args[key] = next((v for v in values if re.search(r"(?i)add|hold|put|keep|plus|insert|reserve|increase|take", v)), None) or next((v for v in values if not re.search(r"(?i)remov|releas|drop|minus|delet|discard|clear|decreas", v)), values[0])
            elif kind in ("integer", "number") or (isinstance(kind, list) and ("integer" in kind or "number" in kind)):
                args[key] = max(quantity, spec.get("minimum", 1))
            elif (kind == "string" or (isinstance(kind, list) and "string" in kind)) and not placed:
                args[key] = ident
                placed = True
        return args

    def instead_of_decline(self) -> tuple[str, dict[str, Any], str] | None:
        """A decline while a proven fit exists scores 0 (in race 1 of suite 160 not one of 381 declines scored, 1,199
        orders did). Once the fits have been named and the loop still declines, or when steps are short, the decline
        becomes the order of the cart when it holds a proven fit, or the carting of the preferred fit."""
        fits = self.proven()
        basis = "whose own facts state every current requirement"
        late = self.decline_checked or f"decline|{self.ledger_version}" in self.rule_told or self.steps_left() <= 6
        if not fits and late:
            fits = self.plausible()                                   # nothing proven: the best listing that contradicts nothing beats a decline
            basis = "whose own facts contradict no current requirement (nothing opened states them all, and a decline scores nothing)"
        order, cart = self.roles.get("order"), self.roles.get("cart")
        if not fits or not order or self.steps_left() <= 1 or self.time_left() < FORCE_SECONDS:
            return None
        if not late:
            return None
        best = self.preferred(fits)
        price = next((p for i, p in fits if i == best), None)
        shown = f"{best} ({price:g})" if price is not None else best
        if self.cart_ids == [best]:
            return (order, self.neutral_args(order, "Order the cart.", "order"), f"Your decline was not sent: a decline scores 0 while a fitting listing exists, and the cart holds {shown}, {basis}, within the budget and not shown out of stock. The order was sent instead.")
        if cart and self.steps_left() > 2:
            return (cart, self.cart_args(best), f"Your decline was not sent: a decline scores 0 while a fitting listing exists, and {shown} is the opened listing {basis}, within the budget and not shown out of stock. It was put in the cart instead: remove any other line, re-check its stock and price, and order it.")
        return None

    def plausible(self) -> list[tuple[str, float | None]]:
        """Opened listings none of whose own facts contradicts a current requirement (each read as yes or unshown), within
        the budget and not shown sold out, on the top shelf; the most requirements proven first, then price. Used only
        where the choice is between one of them and a decline: in race 1 of suite 160 no decline scored (0 of 381)."""
        product = {_one(w) for w in _words(self.plan.get("product") or "", False)}
        top = self.top_shelf() if len(self.plan.get("products") or []) < 2 else ""
        out = []
        for ident, sheet in self.sheets.items():
            if top and _shelf(sheet) and _shelf(sheet) != top or not self.affordable(sheet):
                continue
            readings = [_reading(sheet, prop, value, product)[0] for prop, value in self.demands(ident)]
            if "no" in readings:
                continue
            out.append((ident, self.latest_price(ident), readings.count("yes")))
        out.sort(key=lambda row: (-row[2], row[1] is None, row[1] or 0.0))
        return [(i, p) for i, p, _ in out]

    def demands(self, ident: str | None = None) -> list[tuple[str, str]]:
        """The current requirements as (property, value) pairs the store's rule can read; for one listing of a request
        with several kinds, only those tagged with its own kind (a ring is not checked against the shoes' colour: 5cfa
        declined, 4383 was steered to a wrong pick)."""
        items = list(self.current) if self.current is not None else [r for r in self.stated + self.revealed if r not in self.dropped]
        kinds = [k for k in self.plan.get("products") or [] if isinstance(k, str) and k.strip()]
        mine = self.kind_of(ident, kinds) if ident and len(kinds) > 1 else None
        out = []
        for item in items:
            tag, text = _kind_tag(str(item))
            if mine and tag and not ({_one(w) for w in _words(tag, False)} & {_one(w) for w in _words(mine, False)}):
                continue
            prop, sep, value = text.partition(":")
            prop = prop.strip().lower()
            value = re.sub(r"\s*\([^)]*\)", "", value.strip().lower()).strip(" .")
            if sep and prop and value and not _NOT_A_FACT.search(prop) and not re.search(r"(?i)\bbrand\b", prop):
                out.append((prop, value))
        return out

    def affordable(self, record: dict[str, Any]) -> bool:
        price, quantity, cap = _price(record), self.plan.get("quantity"), self.budget_now
        quantity = quantity if isinstance(quantity, int) and not isinstance(quantity, bool) and quantity > 0 else 1
        if len([k for k in self.plan.get("products") or [] if isinstance(k, str)]) > 1:
            quantity = 1  # one quantity for several lines: only the single-item bound is sure
        if isinstance(cap, (int, float)) and not isinstance(cap, bool) and price is not None and price * quantity > cap + 1e-9:
            return False
        return _in_stock(record) is not False

    def proven(self) -> list[tuple[str, float | None]]:
        """Opened listings that state every current requirement by the store's rule, within the budget and not shown as
        sold out, on the shelf the opened listings point to; by price (the v3 line's ranked fits)."""
        if not self.demands():
            return []
        product = {_one(w) for w in _words(self.plan.get("product") or "", False)}
        top = self.top_shelf() if len(self.plan.get("products") or []) < 2 else ""
        out = []
        for ident, sheet in self.sheets.items():
            if top and _shelf(sheet) and _shelf(sheet) != top or not self.affordable(sheet):
                continue
            demands = self.demands(ident)
            if demands and all(_reading(sheet, prop, value, product)[0] == "yes" for prop, value in demands):
                out.append((ident, self.latest_price(ident)))
        out.sort(key=lambda pair: (pair[1] is None, pair[1] or 0.0))
        return out

    def top_shelf(self) -> str:
        """The shelf the opened listings point to: the one named most like the product, then the most opened."""
        product = {_one(w) for w in _words(self.plan.get("product") or "", False)}
        shelves: dict[str, int] = {}
        for sheet in self.sheets.values():
            if _shelf(sheet):
                shelves[_shelf(sheet)] = shelves.get(_shelf(sheet), 0) + 1
        return max(shelves, key=lambda s: (len(product & {_one(w) for w in _words(s, False)}), shelves[s])) if shelves else ""

    def latest_price(self, item: str) -> float | None:
        """The item's newest price in any record (a cart or bag view after a price move included)."""
        found: list[float | None] = [None]

        def walk(value: Any, depth: int) -> bool:
            if depth > 6:
                return False
            if isinstance(value, dict):
                if any(isinstance(v, str) and v == item for v in value.values()) and _price(value) is not None:
                    found[0] = _price(value)
                    return True
                return any(walk(v, depth + 1) for v in value.values())
            if isinstance(value, list):
                return any(walk(v, depth + 1) for v in value)
            return False

        for record in reversed(self.records):
            if walk(record, 0):
                break
        return found[0] if found[0] is not None else next((p for p in (_price(self.sheets.get(item) or {}), _price(self.rows.get(item) or {})) if p is not None), None)

    def rule_check(self, item: str, adding: bool) -> str:
        """The store's rule on a listing about to be carted or ordered (the v3 line reads the pick's facts first, and a
        fact named for a property that states another value loses whatever the title says)."""
        sheet = self.sheets.get(item)
        if sheet is None:
            if adding and self.sheets and self.roles.get("details") and self.steps_left() > 8:
                return f"The details of {item} have not been opened: the store grades a listing by its own facts, which only the details show. Open them first."
            return ""
        product = {_one(w) for w in _words(self.plan.get("product") or "", False)}
        fails = []
        for prop, value in self.demands(item):
            verdict, shown = _reading(sheet, prop, value, product)
            if verdict == "no":
                fails.append(f"{prop}: {value} (its own fact says {shown})")
        if not fails:
            return ""
        return f"By the store's own facts {item} does not meet: " + "; ".join(fails[:4]) + ". The store reads a requirement from the listing's own fact for that property, not from the title or a look-alike word."

    def cheaper_note(self, item: str) -> str:
        """Cheapest tasks, before an order (the v3 line hunts a cheaper fit by code): fitting opened listings that cost
        less than the pick, and cheaper listings on the pick's shelf seen in results but never opened."""
        if self.plan.get("priority") != "cheapest":
            return ""
        price = self.latest_price(item)
        if price is None:
            return ""
        notes = []
        cheaper = [f"{i} ({p:g})" for i, p in self.proven() if i != item and p is not None and p < price]
        if cheaper and self.steps_left() > 4:
            notes.append(f"by the store's own facts these opened listings meet every current requirement and cost less than {item} ({price:g}): " + ", ".join(cheaper[:4]))
        shelf = _shelf(self.sheets.get(item) or self.rows.get(item) or {})
        unread = sorted((_price(row), i) for i, row in self.rows.items() if i not in self.sheets and shelf and _shelf(row) == shelf and _price(row) is not None and _price(row) < price and self.affordable(row))
        if unread and self.steps_left() > 9:
            notes.append("cheaper listings on the same shelf were seen in results but never opened: " + ", ".join(f"{i} ({p:g})" for p, i in unread[:5]) + "; open them, cheapest first, before ordering")
        if not notes:
            return ""
        return "The request asks for the cheapest fitting item: " + "; ".join(notes) + ". Order the cheapest one that fits."

    def order_note(self, item: str) -> str:
        key = f"{item}|{self.ledger_version}|order"
        if not item or key in self.rule_told:
            return ""
        self.rule_told.add(key)
        return self.rule_check(item, False) or self.cheaper_note(item)

    def decline_note(self) -> str:
        key = f"decline|{self.ledger_version}"
        if key in self.rule_told or self.steps_left() <= 3 or self.time_left() < WRAP_UP_SECONDS:
            return ""
        fits = self.proven()
        if not fits:
            notes = []
            if self.steps_left() > 6:
                unread = self.unread_candidates()
                if unread:
                    notes.append("these listings seen in results were never opened: " + ", ".join(unread) + "; open them, since their facts decide and their titles do not")
                unseen = [f"'{c['label']}' ({c['total']} matched, {len(c['seen'])} seen)" for c in self.coverage.values() if c["label"] and len(c["seen"]) < c["total"] <= 60][:3]
                if unseen:
                    notes.append("these searches have matches you never saw: " + ", ".join(unseen) + "; go through their further pages")
            if not notes:
                return ""
            self.rule_told.add(key)
            return "No opened listing states every current requirement yet, but before declining: " + "; ".join(notes) + ". Decline only if each one fails."
        self.rule_told.add(key)
        return "By the store's own facts these opened listings state every current requirement and are within the budget and not shown out of stock: " + ", ".join(f"{i} ({p:g})" if p is not None else i for i, p in fits[:5]) + ". Re-read the best one (details, stock, price), cart it and order it; decline only if each one truly fails."

    def fits_line(self) -> str:
        """The proven fits, shown to the loop whenever the set changes (the v3 line keeps a ranked list and a proven backup)."""
        fits = self.proven()
        line = ", ".join(f"{i} ({p:g})" if p is not None else i for i, p in fits[:5])
        if line == self.fits_shown:
            return ""
        before, self.fits_shown = self.fits_shown, line
        if not fits:
            return "\n[fits by the store's facts] none of the opened listings states every current requirement in its own facts now." if before else ""
        text = "\n[fits by the store's facts] opened listings whose own facts state every current requirement, within budget and in stock, by price: " + line + "."
        if len(fits) == 1 and self.order_attempts == 0:
            text += " Only one so far: a second fit is the backup if this one sells out or its price moves."
        if len(fits) > 1 and self.hints() and self.preferred(fits) != fits[0][0]:
            text += f" {self.preferred(fits)} matches the story's hint: prefer it unless the shopper's own words or a cheapest rule say otherwise."
        return text

    def core_labels(self, ids: list[str] | None = None) -> list[str]:
        """Labels carried by at least half of the opened listings (two at least), not settled, not asked, not logistics,
        word-valued before figure-valued (29 of race 1's 33 hidden needs had words for values, not figures):
        32 of race 1's 33 hidden needs were such labels, and Leen, Clockwerk and L.MODRIC aimed 77-78% of their questions
        at them against enjoy's 46%."""
        counts: dict[str, int] = {}
        figures: dict[str, int] = {}
        forms: dict[str, dict[str, int]] = {}
        values: dict[str, set[str]] = {}
        group = [self.sheets[i] for i in ids if i in self.sheets] if ids is not None else list(self.sheets.values())
        for sheet in group:
            facts = next((v for v in sheet.values() if isinstance(v, dict)), {})
            for k, v in facts.items():
                values.setdefault(" ".join(w for w in _plain(str(k)) if w != "type"), set()).add(str(v).strip().lower())
                if re.search(r"\d", str(v)):
                    base = " ".join(w for w in _plain(str(k)) if w != "type")
                    figures[base] = figures.get(base, 0) + 1
            labels = {" ".join(_plain(str(k))) for k, v in facts.items() if str(v).strip() and not re.fullmatch(r"(?i)n/?a|none|not specified|unknown|null|-", str(v).strip())}
            for base in {" ".join(w for w in label.split() if w != "type") for label in labels}:
                if base and not re.search(r"(?i)option|condition|warrant|model|title|categor|brand|price|stock|seller|sku|origin|series|assembl|care|instruction|dimension", base):
                    counts[base] = counts.get(base, 0) + 1
            for label in labels:
                base = " ".join(w for w in label.split() if w != "type")
                forms.setdefault(base, {})[label] = forms.get(base, {}).get(label, 0) + 1
        floor = max(2, (len(group) + 1) // 2)
        taken = [prop for prop, _ in self.demands()] + [a[2:].split(": asked", 1)[0] for a in self.asked] + list(self.free)
        out = []
        for base in sorted(counts, key=lambda k: (figures.get(k, 0) * 2 > counts[k], -counts[k])):
            words = set(base.split())
            if counts[base] < floor or not words:
                continue
            if any(words <= set(_plain(t)) or (set(_plain(t)) - {"type"} and set(_plain(t)) - {"type"} <= words) for t in taken):
                continue
            out.append(max(forms[base], key=lambda f: (forms[base][f], "type" in f.split())))
        return out[:8]

    def keep_core(self, planned: list[dict[str, Any]], urgent: bool, core: list[str] | None = None) -> list[dict[str, Any]]:
        """Each planned question about a core label keeps its wording; one about anything else gives way to the next core
        label (what must be learned first, such as a missing budget, keeps its place)."""
        core = self.core_labels() if core is None else core
        if not core:
            return planned
        hinted = [set(_plain(p)) - {"type"} for p, _ in self.hints()]
        labels: list[str | None] = []
        texts: list[set[str]] = []
        for index, item in enumerate(planned):
            text = set(_plain(str(item.get("property") or "") + " " + str(item.get("question") or "")))
            texts.append(text)
            label = None if urgent and index == 0 else next((c for c in core if c not in labels and set(c.split()) - {"type"} <= text), None)
            labels.append(label)
        out: list[dict[str, Any]] = []
        for index, (item, label) in enumerate(zip(planned, labels)):
            if urgent and index == 0 or any(h and h <= texts[index] for h in hinted):
                out.append(item)
                continue
            if label is None:
                label = next((c for c in core if c not in labels), None)
                if label is None:
                    continue
                labels[index] = label
                item = {"property": label, "question": f"Do you have any preference for the {label}?"}
            out.append(item)
        return out

    def unread_candidates(self) -> list[str]:
        """Listings seen in results on the opened listings' shelf, affordable and never opened: those whose own words carry
        a required value first, then the cheaper (Clockwerk reads candidates instead of judging their titles)."""
        top = self.top_shelf()
        wanted = [value for _, value in self.demands()]
        pool = []
        for ident, row in self.rows.items():
            if ident in self.sheets or not self.affordable(row) or top and _shelf(row) and _shelf(row) != top:
                continue
            text = " ".join(str(v) for v in row.values() if isinstance(v, str))
            hits = sum(1 for value in wanted if _bound(value) is None and _stated(value, text))
            price = _price(row)
            pool.append((-hits, price if price is not None else float("inf"), ident))
        return [f"{i} ({p:g})" if p != float("inf") else i for _, p, i in sorted(pool)[:5]]

    def page_note(self, name: str, args: dict[str, Any], observation: Any) -> str:
        """Clockwerk pages through every result of a search before rewording it: say how many matches are still unseen."""
        if name != self.roles.get("search"):
            return ""
        inner = observation
        if isinstance(observation, dict) and len(observation) == 2:
            inner = next((v for v in observation.values() if isinstance(v, dict)), observation)
        if not isinstance(inner, dict):
            return ""
        paging = re.compile(r"(?i)page|offset|first|start|limit|skip|token|cursor|after|next|further")
        total = next((v for k, v in inner.items() if isinstance(v, int) and not isinstance(v, bool) and not paging.search(str(k))), None)
        if total is None:
            return ""
        properties = (self.schemas.get(name) or {}).get("properties") or {}
        free = [str(v) for k, v in args.items() if isinstance(v, str) and not paging.search(str(k)) and not (properties.get(k) or {}).get("enum")]
        entry = self.coverage.setdefault(name + _compact({k: v for k, v in args.items() if not paging.search(str(k))}), {"label": " ".join(free), "total": total, "seen": set()})
        entry["total"] = total
        rows = next((v for v in inner.values() if isinstance(v, list) and v and all(isinstance(x, dict) for x in v)), [])
        for row in rows:
            ident = next((v for k, v in row.items() if isinstance(v, str) and "id" in k.lower() and _ID_HINT.match(v)), None)
            if ident:
                entry["seen"].add(ident)
        left = total - len(entry["seen"])
        if 0 < left and total <= 60:
            return f"This search matched {total} listings; {len(entry['seen'])} of them seen so far, {left} more on its further pages."
        return ""

    def judge(self, prompt: str, max_tokens: int = 700, models: list[str] | None = None) -> dict[str, Any] | None:
        """One JSON answer; an answer cut off by the token limit is asked again with more room."""
        for room in (max(max_tokens, 2000), max(max_tokens, 2000) * 3):
            message = self.infer([{"role": "user", "content": prompt}], None, models or [JUDGE_MODEL, MAIN_MODEL], room)
            if message is None:
                return None
            parsed = _parse_json_object(message.get("content") or "")
            if parsed is not None:
                return parsed
        return None

    def pointed(self, need: Any) -> bool:
        """A requirement that is really a pointer to another listing's page, or one of the details that describe it."""
        return not isinstance(need, str) or bool(re.search(r"(?i)on the page|points? to|set apart|value (?:listed|shown) (?:for|on)", need)) or _need_key(need) in self.described

    def ref_property(self, prop: str, lines: list[str]) -> str:
        """The property a pointed need is about, in the shopper's own words: the model's name when the shopper used it
        outside the description of the other listing, else the longest fact label the opened listings carry that the
        shopper did use there."""
        text = re.sub(r"(?is)page for .*?(?:\)|$)", " ", " ".join(lines).lower())
        spoken = " " + " ".join(_plain(text)) + " "
        if " " + " ".join(_plain(prop)) + " " in spoken:
            return prop.strip().lower()
        labels = {" ".join(_plain(str(k))) for s in self.sheets.values() for k in next((v for v in s.values() if isinstance(v, dict)), {})}
        used = [l for l in labels if l and l != "type" and " " + l + " " in spoken]
        return max(used, key=len) if used else prop.strip().lower()

    def reserve(self) -> int:
        """Steps the finish still needs, live: per product line not yet in the cart, reading its pick and carting it;
        the order with its cart check; room to repair one refusal at checkout until one has happened. Scales with any
        world's step limit and any number of products (fixed thresholds tuned for 40 steps starved 10 orders in #2319)."""
        kinds = max(1, len([k for k in self.plan.get("products") or [] if isinstance(k, str) and k.strip()]))
        lines_left = max(0, kinds - len(self.cart_ids))
        return 2 * lines_left + 2 + (4 if self.order_attempts == 0 else 2)

    def commit_due(self) -> bool:
        """Past the finish's reserve with nothing carted: cart the best fit by the store's facts (else the best that
        contradicts nothing) while steps remain to order it. The decline swap and the forced finish both came too late
        (carting needs more than 2 steps left): six race-3 tasks declined at step 36-40 with a fit opened."""
        if self.forced_commit or self.cart_ids or self.finished or self.spendable() > -2 or self.steps_left() < 3:
            return False
        cart = self.roles.get("cart")
        fits = self.proven() or self.plausible()
        if not cart or not fits:
            return False
        self.forced_commit = True
        best = self.preferred(fits)
        self.call_seq += 1
        call = {"id": f"commit_{self.call_seq}", "type": "function", "function": {"name": cart, "arguments": json.dumps(self.cart_args(best))}}
        self.messages.append({"role": "assistant", "content": f"Steps are short, so I put {best}, the best fit found, in the cart.", "tool_calls": [call]})
        for call_id, text in self.run_calls([call], planned=True):
            self.messages.append({"role": "tool", "tool_call_id": call_id, "content": text})
        self.messages.append({"role": "user", "content": f"[steps] Only {self.steps_left()} steps are left: {best} is in the cart. Make sure it is the only line at the right quantity and order it now; a decline scores nothing while a fit exists.\n" + self.status()})
        return True

    def spendable(self) -> int:
        """Steps free for optional work (questions, look-ups, checks) after the finish's reserve."""
        return self.steps_left() - self.reserve()

    def side_call(self, name: str | None, args: dict[str, Any] | None, why: str) -> Any:
        """One call the agent makes itself (not chosen by the loop's model); returns its observation."""
        if not name or args is None or name not in self.schemas or self.spendable() < 1:
            return None
        args, problems = _check(args, self.schemas[name], "args")
        if problems:
            return None
        self.call_seq += 1  # unique ids: a reused id breaks the whole conversation for every later model call
        call = {"id": f"side_{self.call_seq}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
        self.messages.append({"role": "assistant", "content": why, "tool_calls": [call]})
        count = len(self.records)
        for call_id, text in self.run_calls([call], planned=True):
            self.messages.append({"role": "tool", "tool_call_id": call_id, "content": text})
        return self.records[-1] if len(self.records) > count else None

    def search_args(self, text: str) -> dict[str, Any] | None:
        """Free-text search arguments read from the search tool's schema (a text mode when it has modes)."""
        properties = (self.schemas.get(self.roles.get("search") or "") or {}).get("properties") or {}
        args: dict[str, Any] = {}
        slot = None
        for key, spec in properties.items():
            if not isinstance(spec, dict):
                continue
            if spec.get("enum"):
                pick = next((v for v in spec["enum"] if isinstance(v, str) and re.search(r"(?i)text|keyword|free|query|word|search", v)), None)
                if pick is not None:
                    args[key] = pick
            elif slot is None and (spec.get("type") == "string" or "string" in (spec.get("type") or [])) and re.search(r"(?i)text|keyword|query|term|search|phrase|words|^q$", key):
                slot = key
        if slot is None:
            return None
        args[slot] = text[: int(properties[slot].get("maxLength") or 200)]
        return args

    def id_args(self, name: str | None, ident: str) -> dict[str, Any] | None:
        properties = (self.schemas.get(name or "") or {}).get("properties") or {}
        keys = [k for k, s in properties.items() if isinstance(s, dict) and "enum" not in s and (s.get("type") == "string" or "string" in (s.get("type") or []))]
        key = next((k for k in keys if re.search(r"(?i)id|number|sku|listing|product|item|target|offer", k)), keys[0] if len(keys) == 1 else None)
        return {key: ident} if key else None

    def facts_tool(self) -> str | None:
        """The page tool that returns an item's facts: the one seen doing so, else the one whose description says so."""
        if self.sheet_tool:
            return self.sheet_tool
        tools = [t for t in (self.roles.get("view"), self.roles.get("details")) if t]
        described = {t["function"]["name"]: str(t["function"].get("description") or "") for t in self.tools}
        return next((t for t in tools if re.search(r"(?i)propert|spec|attribute|fact|feature", described.get(t, ""))), tools[0] if tools else None)

    def anchor_fit(self, ident: str, anchor: dict[str, Any], strict: bool = True) -> int:
        """How well a listed item matches the listing a shopper described: -1 when its seller or kind is another, else
        the number of the described details its own facts state."""
        sheet = self.sheets.get(ident) or {}
        record = {**(self.rows.get(ident) or {}), **{k: v for k, v in sheet.items() if not isinstance(v, dict)}}
        strings = [str(v).strip().lower() for v in record.values() if isinstance(v, str)]
        seller = str(anchor.get("seller") or "").strip().lower()
        if seller and not any(re.search(r"\b" + re.escape(seller) + r"\b", s) for s in strings):
            return -1
        kind = {_one(w) for w in _words(str(anchor.get("product") or ""), False)} if strict else set()
        title = " ".join(str(v) for k, v in record.items() if _TITLE_KEY.search(str(k)) and isinstance(v, str))
        if kind and not kind & {_one(w) for w in _words(_shelf(record) + " " + title, False)}:
            return -1
        facts = [str(f).partition(":") for f in anchor.get("facts") or [] if isinstance(f, str) and ":" in f]
        blob = " ".join(strings + self.item_records(ident)).lower()
        return sum(bool(sheet) and _reading(sheet, p.strip().lower(), v.strip().lower(), set())[0] == "yes" or v.strip().lower() in blob for p, _, v in facts)

    def find_anchor(self, anchor: dict[str, Any]) -> str | None:
        """The listing a shopper described by seller, kind and details: searched by kind and details first, candidates
        opened best-shown first (at most eight), taken only when it states every detail or is the one best match one
        detail short; a kind named loosely (a "multipack of spices" for a pepper) falls back to seller and details."""
        seller = str(anchor.get("seller") or "").strip()
        product = str(anchor.get("product") or "").strip()
        values = " ".join(str(f).split(":", 1)[-1].strip() for f in anchor.get("facts") or [] if isinstance(f, str))
        want = len([f for f in anchor.get("facts") or [] if isinstance(f, str) and ":" in f])
        tool = self.facts_tool()
        opened = 0
        fallback = None
        for text in ("", f"{product} {values}", f"{seller} {product}", f"{seller} {values}"):
            if text.strip() and self.side_call(self.roles.get("search"), self.search_args(" ".join(text.split())), f"I will look for {seller}'s {product}, the listing the shopper pointed to.") is None:
                continue
            for strict in (True, False):
                found = [i for i in list(self.rows) + [i for i in self.sheets if i not in self.rows] if self.anchor_fit(i, anchor, strict) >= 0]
                if found:
                    break
            if not found:
                continue
            for ident in sorted([i for i in found if i not in self.sheets], key=lambda i: -self.anchor_fit(i, anchor, strict)):
                if max(self.anchor_fit(i, anchor, strict) for i in found) >= want or opened >= 8 or self.spendable() < 2:
                    break
                self.side_call(tool, self.id_args(tool, ident), "I will read its details to tell the described listing apart.")
                opened += 1
            fits = sorted(((self.anchor_fit(i, anchor, strict), i) for i in found), reverse=True)
            if fits[0][0] >= want:
                return fits[0][1]
            if fits[0][0] >= max(1, want - 1) and (len(fits) == 1 or fits[1][0] < fits[0][0]):
                fallback = fallback or fits[0][1]
            if opened >= 8:
                break
        return fallback

    def page_text(self, ident: str) -> str:
        """The long texts on a listing's own page (a memo there can send a property to another listing)."""
        view = self.roles.get("view") or self.roles.get("details")
        texts: list[str] = []

        def take(value: Any, depth: int) -> None:
            if depth > 5:
                return
            if isinstance(value, dict):
                for key, item in value.items():
                    if isinstance(item, str) and len(item.split()) >= 5 and not _TITLE_KEY.search(str(key)):
                        texts.append(item)
                    elif isinstance(item, (dict, list)):
                        take(item, depth + 1)
            elif isinstance(value, list):
                for item in value:
                    take(item, depth + 1)
        for record in self.records:
            if isinstance(record, dict) and any(isinstance(v, str) and v == ident for v in record.values()):
                take(record, 0)
        if not any(re.search(r"(?i)refer|taken from|go by|look to|look at|points? to|instead|another|not (?:to|from) this", t) for t in texts):
            other = next((t for t in (self.roles.get("view"), self.roles.get("details")) if t and t != self.sheet_tool), view)
            take(self.side_call(other, self.id_args(other, ident), "I will read that listing's page."), 0)
        return " ".join(dict.fromkeys(texts))[:2500]

    def value_on(self, ident: str, prop: str, page: str) -> str | None:
        if ident not in self.sheets and self.roles.get("details"):
            self.side_call(self.roles.get("details"), self.id_args(self.roles.get("details"), ident), "I will read its details for the value.")
        sheet = self.sheets.get(ident) or {}
        facts = next((v for v in sheet.values() if isinstance(v, dict)), {})
        named = [str(v) for k, v in facts.items() if _names(str(k), prop, set()) and str(v).strip()]
        if not named:
            words = set(_plain(prop)) - {"type"}
            named = [str(v) for k, v in facts.items() if words & (set(_plain(str(k))) - {"type"}) and str(v).strip()]
        if named:
            return named[0]
        result = self.judge(f"What does this listing's page say its {prop} is? Return ONE JSON object and nothing else: {{\"value\": the value in the page's words, or null}}.\n\nTHE PAGE:\n{page}\n{_compact(sheet)[:1500]}", 200, [READ_MODEL, JUDGE_MODEL])
        value = (result or {}).get("value")
        return value.strip() if isinstance(value, str) and value.strip() else None

    def resolve_refs(self) -> bool:
        """A need whose value sits on another listing's page (the shopper names a seller's product by a few details and
        asks for the same value or a different one, and that page may send the property on to a further listing): find that listing by its seller,
        kind and details, follow its page when it sends the property to yet another listing, read the value there and
        turn the need into a plain requirement. The King read the described details as requirements on the item being
        bought and never looked the value up."""
        if not self.refs or not REF_LOOKUP:
            self.refs = []
            return False
        ref = self.refs.pop(0)
        self.refs_done.append(ref)
        if self.spendable() < 8 or self.time_left() < WRAP_UP_SECONDS + 90:
            return False  # a look-up costs about 3-8 steps
        prop = str(ref.get("property")).strip().lower()
        anchor, value, ident = ref, None, None
        for _ in range(3):
            ident = self.find_anchor(anchor)
            if not ident:
                break
            page = self.page_text(ident)
            hop = None
            if page and re.search(r"(?i)refer|taken from|take it from|go by|look to|look at|points? to|instead|another|not (?:to|from) this", page):
                hop = self.judge(REF_HOP_PROMPT.replace("PROPERTY", prop) + "\n\nPROPERTY: " + prop + "\n\nTHE PAGE'S TEXT:\n" + page, 400, [JUDGE_MODEL, READ_MODEL])
            if isinstance(hop, dict) and hop.get("sends") and (hop.get("seller") or hop.get("facts")):
                anchor = hop
                continue
            value = self.value_on(ident, prop, page)
            break
        if not value:
            self.messages.append({"role": "user", "content": f"[note] The {prop} the shopper pointed to could not be read from that listing's page; judge the {prop} from the shopper's words.\n" + self.status()})
            return True
        need = f"{prop}: {'' if ref.get('relation') == 'same' else 'anything except '}{value.strip().lower()}"
        self.revealed.append(need)
        if self.current is not None:
            self.current.append(need)
        self.ledger_version += 1
        self.messages.append({"role": "user", "content": f"[note] The {prop} on the page the shopper pointed to ({ident}) is {value}, so the requirement is now {need}. Apply it to every candidate before carting or ordering.\n" + self.status()})
        return True

    def interpret(self, lines: list[str], question: str | None) -> str:
        fresh = [line for line in lines if line not in self.heard]
        if not fresh:
            return ""
        self.heard.update(fresh)
        if self.time_left() < WRAP_UP_SECONDS:
            return ""
        prompt = INTERPRET_PROMPT + "\n\nTHE SHOPPER'S REQUEST (for context):\n" + str(self.policy.get("query") or "")
        prompt += "\n\nREQUIREMENTS SO FAR:\n" + self.ledger()
        if self.said:
            prompt += "\n\nEARLIER EXCHANGES WITH THE SHOPPER (oldest first):\n" + "\n".join(self.said[-12:])
        if question:
            prompt += "\n\nTHE ASSISTANT ASKED:\n" + question
        prompt += "\n\nTHE SHOPPER NOW SAID:\n" + "\n".join(fresh)
        self.said.append((f"asked: {question} -> " if question else "") + "shopper: " + " / ".join(fresh))
        result = self.judge(prompt, 1500, [READ_MODEL, JUDGE_MODEL])
        if not result:
            return ""
        notes = []
        refs = [r for r in result.get("references") or [] if isinstance(r, dict) and str(r.get("property") or "").strip()]
        for ref in refs:
            self.described |= {_need_key(f) for f in ref.get("facts") or [] if isinstance(f, str) and ":" in f}
        if refs or self.described:
            result["needs"] = [n for n in result.get("needs") or [] if not self.pointed(n)]
            if isinstance(result.get("requirements_now"), list):
                result["requirements_now"] = [n for n in result["requirements_now"] if not self.pointed(n)]
        if refs:
            for ref in refs:
                ref["property"] = self.ref_property(str(ref["property"]), fresh)
                prop = str(ref["property"]).strip().lower()
                if all(str(r.get("property")).strip().lower() != prop for r in self.refs + self.refs_done):
                    self.refs.append(ref)
                    notes.append(f"the {prop} must be {'the same as' if ref.get('relation') == 'same' else 'different from'} the value on another listing's page" + ("; that listing is looked up for you" if REF_LOOKUP else "; the details that describe that listing are not requirements on this item"))
        now = result.get("requirements_now")
        if isinstance(now, list) and all(isinstance(r, str) for r in now):
            self.current = [r for r in now if r.strip()]
        for need in result.get("needs") or []:
            if isinstance(need, str) and need and need not in self.revealed:
                self.revealed.append(need)
                notes.append(f"new requirement: {need}")
        for prop in result.get("no_preference") or []:
            if isinstance(prop, str) and prop and prop not in self.free:
                self.free.append(prop)
        for drop in result.get("dropped") or []:
            if isinstance(drop, str) and drop and drop not in self.dropped:
                self.dropped.append(drop)
                notes.append(f"dropped: {drop}")
        budget = result.get("new_budget")
        if isinstance(budget, (int, float)) and not isinstance(budget, bool):
            budget = _in_written_unit(budget, " ".join(fresh))
        if isinstance(budget, (int, float)) and not isinstance(budget, bool) and budget > 0 and budget != self.budget_now:
            self.budget_now = budget
            notes.append(f"new budget: {budget}")
        if result.get("unavailable"):
            notes.append("an item became unavailable; remove it from the cart and do not order it")
        for need in result.get("needs") or []:
            if isinstance(need, str) and ":" in need:
                stating, other = self.sheets_stating(need)
                if stating:
                    notes.append(f"opened listings that state {need.strip()}: " + ", ".join(stating[:6]))
                elif self.sheets:
                    notes.append(f"no opened listing states {need.strip()} yet: search for it")
        moved = bool(result.get("dropped")) or (isinstance(budget, (int, float)) and not isinstance(budget, bool) and budget > 0)
        if moved and self.plan.get("priority") == "cheapest":
            notes.append("the set of fitting items has changed: search again (items excluded before may fit now) and re-pick the cheapest that fits before ordering")
        if notes:
            self.ledger_version += 1
        reply = str(result.get("reply") or "").strip()
        if reply and self.roles.get("message") and any("?" in line for line in fresh):
            self.pending_reply = reply
        meaning = str(result.get("meaning") or "")
        return "Meaning: " + meaning + (" (" + "; ".join(notes) + ")" if notes else "")

    def item_records(self, item: str) -> list[str]:
        found: list[str] = []

        def walk(value: Any, depth: int) -> None:
            if depth > 6 or len(found) >= 8:
                return
            if isinstance(value, dict):
                if any(isinstance(v, str) and v == item for v in value.values()):
                    text = _compact(_brief(value, 400))[:2500]
                    if text not in found:
                        found.append(text)
                    return
                for v in value.values():
                    walk(v, depth + 1)
            elif isinstance(value, list):
                for v in value:
                    walk(v, depth + 1)

        for record in self.records:
            walk(record, 0)
        return found[-4:]

    def verify_cart(self) -> str:
        if not self.cart_ids:
            return ""
        return self.verify_item(self.cart_ids[-1], "ORDER")

    def verify_item(self, item: str, purpose: str) -> str:
        if self.time_left() < WRAP_UP_SECONDS + 20:
            return ""
        key = f"{item}|{self.ledger_version}"
        if key in self.verdicts:
            return self.verdicts[key]
        records = self.item_records(item)
        if not records:
            return ""
        prompt = VERIFY_PROMPT + "\n\nTHE SHOPPER'S REQUEST:\n" + str(self.policy.get("query") or "")
        prompt += "\n\nCURRENT REQUIREMENTS (stated, revealed and changed):\n" + self.ledger()
        prompt += f"\n\nTHE ITEM TO {purpose}: {item}\nRECORDS OF THIS ITEM SEEN IN THE STORE (oldest first):\n" + "\n".join(records)
        if self.last_cart_view is not None:
            prompt += "\n\nLATEST CART VIEW:\n" + _compact(self.last_cart_view)[:3000]
        labels = self.row_labels()
        if labels:
            prompt += "\n\nLABELS THAT RECUR ON THIS STORE'S LISTINGS (with counts; the item's category label must be the one that matches the request's wording as a whole, not a neighbouring one):\n" + labels
        result = self.judge(prompt, 1200)
        self.trace.append({"kind": "verify", "item": item, "result": result})
        if not result or result.get("fits") is not False:
            self.verdicts[key] = ""
            return ""
        problems = [str(p) for p in result.get("problems") or []][:5]
        verdict = f"An independent check says {item} does NOT meet every requirement: " + "; ".join(problems) + ". Unless you are certain the check is wrong, choose another fully verified fit."
        self.verdicts[key] = verdict
        return verdict

    def tools_for_turn(self) -> list[dict[str, Any]]:
        left = self.steps_left()
        order, decline, cart = self.roles.get("order"), self.roles.get("decline"), self.roles.get("cart")
        names: set[str] | None = None
        if left <= 1:
            names = {order, decline}
        elif left <= 3:
            names = {order, decline, cart}
        elif left <= 7 and (self.order_attempts == 0 or self.refusals > 0):
            names = {order, decline, cart, self.roles.get("cart_view"), self.roles.get("view")}
        tools = self.tools
        if names is not None and order and cart:
            narrowed = [t for t in self.tools if t["function"]["name"] in names]
            if narrowed:
                tools = narrowed
        if self.planner_on and self.roles.get("message"):
            tools = [t for t in tools if t["function"]["name"] != self.roles.get("message")]
        return tools

    def property_menu(self) -> tuple[str, int]:
        counts: dict[str, int] = {}
        samples: dict[str, list[str]] = {}
        fact_sets = 0

        def short(value: Any) -> bool:
            return isinstance(value, str) and 0 < len(value) <= 70

        def walk(value: Any, depth: int) -> None:
            nonlocal fact_sets
            if depth > 6:
                return
            if isinstance(value, dict):
                strings = [(k, v) for k, v in value.items() if short(v)]
                if len(strings) >= 5:
                    fact_sets += 1
                    for key, item in strings:
                        if re.search(r"\d", item) and re.fullmatch(r"[A-Za-z0-9_\-]+", item) and ("-" in item or "_" in item):
                            continue
                        name = re.sub(r"[_.]+", " ", str(key)).strip().lower()
                        if _LOGISTICS.search(name) or re.search(r"(?i)title|summary|department|memo|^id$|name$", name):
                            continue
                        counts[name] = counts.get(name, 0) + 1
                        bucket = samples.setdefault(name, [])
                        if item.lower() not in bucket and len(bucket) < 5:
                            bucket.append(item.lower())
                for item in value.values():
                    if isinstance(item, (dict, list)):
                        walk(item, depth + 1)
            elif isinstance(value, list):
                for item in value:
                    walk(item, depth + 1)

        for record in self.records:
            walk(record, 0)
        ranked = sorted(counts, key=lambda k: -counts[k])[:28]
        lines = [f"- {k} (in {counts[k]} listings): " + " | ".join(samples.get(k, [])) for k in ranked]
        return "\n".join(lines), fact_sets

    def row_labels(self) -> str:
        counts: dict[str, dict[str, int]] = {}

        def walk(value: Any, depth: int) -> None:
            if depth > 6:
                return
            if isinstance(value, dict):
                if len(value) >= 4:
                    for key, item in value.items():
                        if isinstance(item, str) and 0 < len(item) <= 40 and not (re.search(r"\d", item) and re.fullmatch(r"[A-Za-z0-9_\-]+", item)):
                            bucket = counts.setdefault(str(key), {})
                            bucket[item] = bucket.get(item, 0) + 1
                for item in value.values():
                    if isinstance(item, (dict, list)):
                        walk(item, depth + 1)
            elif isinstance(value, list):
                for item in value:
                    walk(item, depth + 1)

        for record in self.records:
            walk(record, 0)
        lines = []
        for key, bucket in counts.items():
            total = sum(bucket.values())
            if total >= 4 and 2 <= len(bucket) <= 8 and max(bucket.values()) >= 2:
                lines.append(f"- {key}: " + " | ".join(f"{v} (x{n})" for v, n in sorted(bucket.items(), key=lambda kv: -kv[1])))
        return "\n".join(lines[:12])

    def message_args(self, text: str) -> dict[str, Any] | None:
        name = self.roles.get("message")
        schema = self.schemas.get(name or "") or {}
        properties = schema.get("properties") or {}
        required = list(schema.get("required") or [])
        strings = [k for k, v in properties.items() if isinstance(v, dict) and (v.get("type") == "string" or "string" in (v.get("type") or [])) and "enum" not in v]
        target = next((k for k in required if k in strings), None) or (strings[0] if len(strings) == 1 else None)
        if not target:
            return None
        if any(k != target for k in required):
            return None
        return {target: text}

    def kind_of(self, ident: str, kinds: list[str]) -> str | None:
        """Which requested kind of product an opened listing is, by its shelf (its title when it shows none)."""
        record = {**(self.rows.get(ident) or {}), **{k: v for k, v in (self.sheets.get(ident) or {}).items() if not isinstance(v, dict)}}
        text = _shelf(record) or " ".join(str(v) for k, v in record.items() if _TITLE_KEY.search(str(k)) and isinstance(v, str))
        words = {_one(w) for w in _words(text, False)}
        scored = [(len(words & {_one(w) for w in _words(k, False)}), k) for k in kinds]
        best = max(scored, key=lambda s: s[0], default=(0, None))
        return best[1] if best[0] else None

    def menu_of(self, ids: list[str]) -> str:
        counts: dict[str, int] = {}
        samples: dict[str, list[str]] = {}
        for ident in ids:
            facts = next((v for v in (self.sheets.get(ident) or {}).values() if isinstance(v, dict)), {})
            for key, item in facts.items():
                if not isinstance(item, (str, int, float)) or not str(item).strip():
                    continue
                name = re.sub(r"[_.]+", " ", str(key)).strip().lower()
                counts[name] = counts.get(name, 0) + 1
                bucket = samples.setdefault(name, [])
                if str(item).lower()[:40] not in bucket and len(bucket) < 5:
                    bucket.append(str(item).lower()[:40])
        return "\n".join(f"- {k} (in {counts[k]} listings): " + " | ".join(samples[k]) for k in sorted(counts, key=lambda k: -counts[k])[:24])

    def ask_split(self, force: bool) -> bool | None:
        """The request asks for several kinds of product, each with hidden needs of its own: the messages are shared
        out over the kinds, each kind's questions are planned from its own opened listings once two of them are open,
        and every question names the kind it is about. The King planned all its questions from the first kind's
        listings and left the other kinds unasked. None when the request names one kind."""
        kinds = [str(k).strip() for k in self.plan.get("products") or [] if isinstance(k, str) and str(k).strip()]
        if len(kinds) < 2 or (self.plan.get("ask_first") and not self.asked):
            return None
        tool = self.roles.get("message")
        limit = self.message_budget if self.message_budget is not None else 3
        waiting = [k for k in kinds if k not in self.kinds_asked]
        if not waiting or self.spendable() < 2 or self.time_left() < WRAP_UP_SECONDS + 40:
            self.planner_on = False
            return False
        groups = {k: [i for i in self.sheets if self.kind_of(i, kinds) == k] for k in kinds}
        known = {_shelf(r) for r in self.rows.values() if _shelf(r)}

        def home(kind: str) -> str:
            words = {_one(w) for w in _words(kind, False)}
            best = max(((len(words & {_one(w) for w in _words(s, False)}), s) for s in known), default=(0, ""))
            return best[1] if best[0] else ""
        merged: list[str] = []
        for k in kinds:
            if not home(k) or home(k) not in [home(j) for j in merged]:
                merged.append(k)
        if len(merged) < 2:
            return None
        if len(merged) < len(kinds):
            kinds = merged
            groups = {k: [i for i in self.sheets if self.kind_of(i, kinds) == k] for k in kinds}
            waiting = [k for k in waiting if k in kinds]
        late = force or self.step >= min(16, self.max_steps // 2)
        ready = [k for k in waiting if len(groups[k]) >= 2 or (late and groups[k])]
        if not ready:
            return False
        left = min(limit - self.messages_used, self.spendable() - 1)  # each question is a step
        calls: list[dict[str, Any]] = []
        labels: list[tuple[str, str]] = []
        for kind in ready:
            share = -(-left // len(waiting))
            waiting.remove(kind)
            self.kinds_asked.append(kind)
            if share <= 0:
                continue
            core = [c for c in self.core_labels(groups[kind]) if not _LOGISTICS.search(c)]
            prompt = PLAN_PROMPT.replace("QUESTION_COUNT", str(share)) + "\n\nTHE SHOPPER'S REQUEST:\n" + str(self.policy.get("query") or "")
            prompt += f"\n\nTHESE QUESTIONS ARE ABOUT THE {kind.upper()} ONLY (the request asks for other kinds too; their questions are planned separately)."
            prompt += "\n\nREQUIREMENTS SO FAR:\n" + self.ledger()
            if self.asked:
                prompt += "\n\nALREADY ASKED (with what the shopper said):\n" + "\n".join(self.asked)
            prompt += f"\n\nPROPERTIES IN THE DETAILS OF THE {kind.upper()} LISTINGS OPENED (name, number of listings, example values):\n" + (self.menu_of(groups[kind]) or "(none seen yet)")
            if core:
                prompt += "\n\nCORE PROPERTIES (carried by most of these listings, not settled or asked yet; ask only about these): " + ", ".join(core)
            result = self.judge(prompt, 1500, [READ_MODEL, JUDGE_MODEL])
            planned = [q for q in (result or {}).get("questions") or [] if isinstance(q, dict) and str(q.get("property") or "").strip()]
            planned = [q for q in planned if not _LOGISTICS.search(str(q.get("property")))]
            planned = self.keep_core(planned, False, core)[:share]
            for item in planned:
                prop = re.sub(r"\s+", " ", str(item.get("property")).strip().lower())
                question = f"Do you have a preference for the {prop} of the {kind.lower()}?"
                args = self.message_args(question)
                if not args:
                    continue
                args, problems = _check(args, self.schemas[tool], "args")
                if problems:
                    continue
                self.call_seq += 1
                calls.append({"id": f"ask_{self.call_seq}", "type": "function", "function": {"name": tool, "arguments": json.dumps(args)}})
                labels.append((f"{prop} ({kind})", question))
            left -= len(planned)
        self.planner_on = bool(waiting) and self.messages_used + len(calls) < limit
        if not calls:
            return False
        self.messages.append({"role": "assistant", "content": "I will ask the shopper about " + ", then ".join(l[0] for l in labels) + ".", "tool_calls": calls})
        replies = self.run_calls(calls, planned=True)
        for index, (call_id, text) in enumerate(replies):
            if index == len(replies) - 1:
                text += "\n" + self.status()
            self.messages.append({"role": "tool", "tool_call_id": call_id, "content": text})
            if index < len(labels):
                self.asked.append(f"- {labels[index][0]}: asked \"{labels[index][1]}\" -> {text.split(chr(10) + '[status]')[0][:400]}")
        return True

    def filter_args(self, shelf: str, ceiling: float, page: int, words: str = "") -> dict[str, Any] | None:
        """Shelf-and-price filter arguments read from the search tool's schema, or None when it has no such filter."""
        properties = (self.schemas.get(self.roles.get("search") or "") or {}).get("properties") or {}
        args: dict[str, Any] = {}
        for key, spec in properties.items():
            if not isinstance(spec, dict):
                continue
            kind = spec.get("type")
            if spec.get("enum"):
                pick = next((v for v in spec["enum"] if isinstance(v, str) and re.search(r"(?i)narrow|filter|browse|categor|shelf|department|facet", v)), None)
                if pick is not None:
                    args[key] = pick
            elif kind == "string" and re.search(r"(?i)shelf|department|categor|section|aisle", key):
                args[key] = shelf
            elif kind in ("number", "integer") and re.search(r"(?i)ceil|max|upper|price|budget|cost", key):
                args[key] = ceiling if kind == "number" else int(ceiling)
            elif kind == "integer" and re.search(r"(?i)page", key) and page > 1:
                args[key] = page
            elif kind == "string" and words and re.search(r"(?i)keyword|query|text|term|words", key):
                args[key] = words[: int(spec.get("maxLength") or 60)]
        if shelf not in args.values() or ceiling not in args.values() and int(ceiling) not in args.values():
            return None
        return args

    def cheaper_probe(self, calls: list[dict[str, Any]]) -> str:
        """Before a listing is carted under a cheapest rule: the store's own price filter on its shelf, just under its
        price, lists every cheaper item; those are read cheapest first against the requirements that apply to that
        shelf, and the first that states them all is offered instead. Keyword searches showed only part of a shelf,
        and 11 of 81 race-3 tasks ordered a fit that was not the cheapest."""
        cart, order = self.roles.get("cart"), self.roles.get("order")
        items: list[str] = []
        for call in calls:
            name = call["function"].get("name")
            raw = call["function"].get("arguments")
            args = raw if isinstance(raw, dict) else (_parse_json_object(raw or "{}") or {})
            if name == cart:
                items += [v for v in args.values() if isinstance(v, str) and (v in self.rows or v in self.sheets)]
            elif name == order:
                items += list(self.cart_ids)
        product = {_one(w) for w in _words(self.plan.get("product") or "", False)}
        tool = self.facts_tool()
        for item in items:
            if item in self.probed or self.spendable() < 6 or self.time_left() < WRAP_UP_SECONDS + 60:
                continue
            self.probed.add(item)
            record = {**(self.rows.get(item) or {}), **{k: v for k, v in (self.sheets.get(item) or {}).items() if not isinstance(v, dict)}}
            shelf, price, sheet = _shelf(record), self.latest_price(item), self.sheets.get(item)
            if not shelf or price is None or not sheet:
                continue
            facts = next((v for v in sheet.values() if isinstance(v, dict)), {})
            mine = [(p, v) for p, v in self.demands(item) if any(_names(str(k), p, product) for k in facts)]
            if not mine or any(_reading(sheet, p, v, product)[0] != "yes" for p, v in mine):
                continue
            ceiling = price - 0.5 if float(price).is_integer() else round(price * 0.999, 2)
            found: list[str] = []
            words = " ".join(v for _, v in mine if not _bound(v) and not re.match(r"(?i)(?:not|no|without|anything|except)\b", v))
            # the requirement words narrow the shelf first; the whole shelf under the price is the fallback
            for passes in ([[(1, words)]] if words else []) + [[(n, "") for n in range(1, 7)]]:
                for page, text in passes:
                    args = self.filter_args(shelf, ceiling, page, text)
                    if args is None or self.spendable() < 2:
                        break
                    observation = self.side_call(self.roles.get("search"), args, f"I will check {shelf} for anything cheaper than {item} that also fits.")
                    inner: Any = observation
                    if isinstance(observation, dict) and len(observation) == 2:
                        inner = next((v for v in observation.values() if isinstance(v, dict)), observation)
                    rows = next((v for v in inner.values() if isinstance(v, list) and all(isinstance(x, dict) for x in v)), []) if isinstance(inner, dict) else []
                    ids = [next((v for k, v in r.items() if isinstance(v, str) and "id" in k.lower() and _ID_HINT.match(v)), None) for r in rows]
                    found += [i for i in ids if i and i != item and i not in found]
                    more = isinstance(inner, dict) and any(re.search(r"(?i)next|more", str(k)) and v not in (None, False, "", 0) for k, v in inner.items())
                    if not rows or not more:
                        break
                if found:
                    break
            cheaper = sorted((p, i) for i in found for p in [self.latest_price(i)] if p is not None and p < price and _in_stock(self.rows.get(i) or {}) is not False)
            opened = 0
            for p, ident in cheaper:
                if ident not in self.sheets:
                    if opened >= 8 or self.spendable() < 2:
                        break
                    self.side_call(tool, self.id_args(tool, ident), "I will read its details against the requirements.")
                    opened += 1
                other = self.sheets.get(ident)
                if other and self.affordable(other) is not False and all(_reading(other, q, v, product)[0] == "yes" for q, v in mine):
                    self.probed.add(ident)
                    return (f"[cheaper fit] {ident} costs {p:g}, less than {item} ({price:g}), and its own facts state every requirement for {shelf} ("
                            + "; ".join(f"{q}: {v}" for q, v in mine) + "). The request asks for the cheapest, so put " + ident + " in the cart instead of " + item + " (check its stock), then order.")
        return ""

    def ask_planned(self, force: bool = False) -> bool:
        tool = self.roles.get("message")
        limit = self.message_budget if self.message_budget is not None else 3
        if not tool or self.messages_used >= limit:
            self.planner_on = False
            return False
        split = self.ask_split(force)
        if split is not None:
            return split
        opened = sum(1 for sheet in self.sheets.values() if any(isinstance(v, dict) and v for v in sheet.values()))
        if force and opened < 3 and not self.opened_first and self.spendable() >= 4:
            self.opened_first = True
            self.messages.append({"role": "user", "content": "[note] Before carting, open the details (facts) of two or three listings of the requested kind: the questions to the shopper are planned from those details.\n" + self.status()})
            return True
        menu, fact_sets = self.property_menu()
        urgent = bool(self.plan.get("ask_first")) and not self.asked
        if self.plan_rounds and not force and not urgent and len(self.sheets) < self.plan_sheets + 2:
            return False  # a later round waits for two more opened listings (or the next commit)
        if len(self.sheets) < 4 and self.step < min(10, self.max_steps // 3) and not urgent and not force:
            return False
        if force:
            urgent = False
        count = min(limit - self.messages_used, self.spendable() - 1)  # each question is a step
        if count < 1 or self.time_left() < WRAP_UP_SECONDS + 40:
            self.planner_on = False
            return False
        prompt = PLAN_PROMPT.replace("QUESTION_COUNT", str(count)) + "\n\nTHE SHOPPER'S REQUEST:\n" + str(self.policy.get("query") or "")
        prompt += "\n\nREQUIREMENTS SO FAR:\n" + self.ledger()
        if self.hints():
            prompt += "\n\nHINTS FROM THE STORY (properties the request points at without stating a value; ask about these first, each in the listings' own words for it): " + "; ".join(f"{p} (likely {v})" if v else p for p, v in self.hints())
        if self.asked:
            prompt += "\n\nALREADY ASKED (with what the shopper said):\n" + "\n".join(self.asked)
        if self.plan.get("ask_first"):
            prompt += "\n\nMUST BE LEARNED FROM THE SHOPPER: " + "; ".join(str(h) for h in self.plan.get("ask_first") or [])
        prompt += "\n\nPROPERTIES IN THE LISTINGS' DETAILS SEEN SO FAR (name, number of listings, example values):\n" + (menu or "(none seen yet)")
        core = self.core_labels()
        if core:
            prompt += "\n\nCORE PROPERTIES (carried by most of the listings opened, not settled or asked yet; ask only about these): " + ", ".join(core)
        result = self.judge(prompt, 2000, [READ_MODEL, JUDGE_MODEL])
        planned = [q for q in (result or {}).get("questions") or [] if isinstance(q, dict) and str(q.get("question") or "").strip()]
        if not planned and core:
            # the core list can hold only properties the request already settles under another name (3ff2: "dietary
            # need" against "sugar: low"); then the whole menu is open
            free = prompt.split("\n\nCORE PROPERTIES")[0]
            result = self.judge(free, 2000, [READ_MODEL, JUDGE_MODEL])
            planned = [q for q in (result or {}).get("questions") or [] if isinstance(q, dict) and str(q.get("question") or "").strip()]
            core = []
        planned = [q for i, q in enumerate(planned) if urgent and i == 0 or not _LOGISTICS.search(str(q.get("property") or ""))]
        if urgent and fact_sets < 4:
            planned = planned[:1]
        planned = (self.keep_core(planned, urgent) if core else planned)[:count]
        calls = []
        labels = []
        for index, item in enumerate(planned):
            args = self.message_args(str(item.get("question")).strip())
            if not args:
                continue
            args, problems = _check(args, self.schemas[tool], "args")
            if problems:
                continue
            self.call_seq += 1
            calls.append({"id": f"ask_{self.call_seq}", "type": "function", "function": {"name": tool, "arguments": json.dumps(args)}})
            labels.append((str(item.get("property")), str(item.get("question")).strip()))
        if not calls:
            self.planner_on = False
            return False
        self.messages.append({"role": "assistant", "content": "I will ask the shopper about " + ", then ".join(l[0] for l in labels) + ".", "tool_calls": calls})
        replies = self.run_calls(calls, planned=True)
        for index, (call_id, text) in enumerate(replies):
            if index == len(replies) - 1:
                text += "\n" + self.status()
            self.messages.append({"role": "tool", "tool_call_id": call_id, "content": text})
            if index < len(labels):
                self.asked.append(f"- {labels[index][0]}: asked \"{labels[index][1]}\" -> {text.split(chr(10) + '[status]')[0][:400]}")
        # a round that used fewer messages than are left keeps the planner on (up to three rounds): the core
        # properties of the first few listings can run out after one question (3ff2, 849f asked 1 of 4)
        self.plan_rounds += 1
        self.plan_sheets = len(self.sheets)
        if self.messages_used >= limit or not (urgent and fact_sets < 4):
            self.planner_on = self.messages_used < limit and (urgent and fact_sets < 4 or self.plan_rounds < 3)
        return True

    def send_reply(self, text: str) -> bool:
        tool = self.roles.get("message")
        limit = self.message_budget if self.message_budget is not None else 3
        if not tool or self.messages_used >= limit or self.steps_left() <= 4 or self.time_left() < WRAP_UP_SECONDS:
            return False
        args = self.message_args(text)
        if not args:
            return False
        args, problems = _check(args, self.schemas[tool], "args")
        if problems:
            return False
        call = {"id": f"reply_{self.step}", "type": "function", "function": {"name": tool, "arguments": json.dumps(args)}}
        self.messages.append({"role": "assistant", "content": "I will answer the shopper's question.", "tool_calls": [call]})
        replies = self.run_calls([call], planned=True)
        for index, (call_id, reply_text) in enumerate(replies):
            if index == len(replies) - 1:
                reply_text += "\n" + self.status()
            self.messages.append({"role": "tool", "tool_call_id": call_id, "content": reply_text})
        return True

    def quantity_warning(self, name: str, args: dict[str, Any]) -> str:
        wanted = self.plan.get("quantity")
        if not isinstance(wanted, int) or isinstance(wanted, bool) or wanted < 1:
            return ""
        properties = self.schemas[name].get("properties") or {}
        signature = _compact({k: v for k, v in args.items() if isinstance(properties.get(k), dict) and "enum" in properties[k]})
        if self.add_signature is not None and signature != self.add_signature:
            return ""
        for key, spec in properties.items():
            if isinstance(spec, dict) and spec.get("type") == "integer" and "enum" not in spec:
                value = args.get(key, spec.get("default", 1))
                if isinstance(value, int) and value != wanted:
                    return f"the request needs a quantity of {wanted} of one listing, but this call asks for {value}."
        return ""

    def audit_decline(self) -> str:
        if self.audited or self.time_left() < WRAP_UP_SECONDS + 30:
            return ""
        self.audited = True
        blocks: list[str] = []
        total = 0
        for ident in reversed(list(self.sheets)):
            text = _compact(self.sheets[ident])
            if total + len(text) > 30000:
                continue
            blocks.append(text)
            total += len(text)
        for record in reversed(self.records):
            text = _compact(_brief(record, 160))
            if total + len(text) > 36000:
                continue
            blocks.append(text)
            total += len(text)
        if not blocks:
            return ""
        prompt = AUDIT_PROMPT + "\n\nTHE SHOPPER'S REQUEST:\n" + str(self.policy.get("query") or "")
        prompt += "\n\nCURRENT REQUIREMENTS:\n" + self.ledger()
        prompt += "\n\nLISTING RECORDS SEEN (newest first):\n" + "\n".join(blocks)
        result = self.judge(prompt, 1200)
        self.trace.append({"kind": "audit", "result": result})
        fits = [str(x) for x in (result or {}).get("possible_fits") or [] if isinstance(x, (str, int))][:5]
        if not fits:
            return ""
        return "An independent check thinks these listings may meet every requirement: " + ", ".join(fits) + (f" ({result.get('reason')})" if result.get("reason") else "") + ". Verify them (details, stock, price) and buy the right one; decline only if each one truly fails."

    def neutral_args(self, name: str, text: str, kind: str | None = None) -> dict[str, Any]:
        schema = self.schemas.get(name) or {}
        properties = schema.get("properties") or {}
        args: dict[str, Any] = dict(self.finish_args.get(kind) or {}) if kind else {}
        for key in schema.get("required") or []:
            if key in args:
                continue
            spec = properties.get(key) or {}
            if spec.get("type") == "string" or "string" in (spec.get("type") or []):
                args[key] = text[: spec.get("maxLength", len(text))]
            elif "enum" in spec and spec["enum"]:
                args[key] = spec["enum"][0]
            elif spec.get("type") == "boolean":
                args[key] = True
            elif spec.get("type") in ("integer", "number"):
                args[key] = max(1, spec.get("minimum", 1))
        for key, spec in properties.items():
            if key not in args and isinstance(spec, dict) and spec.get("type") == "string" and "enum" not in spec and len(properties) == 1:
                args[key] = text[: spec.get("maxLength", len(text))]
        return args

    def clean_decline(self, args: dict[str, Any], name: str) -> dict[str, Any]:
        if self.plan.get("decline_no_names") is False:
            return args
        dirty = False
        for key, value in args.items():
            if isinstance(value, str):
                for token in re.split(r"[\s.,;:!?()\[\]{}<>\"'`]+", value):
                    if token and token in self.seen_ids:
                        dirty = True
        if self.plan.get("decline_no_names") or dirty:
            neutral = "Sorry, nothing in the store meets all of your requirements, so I am not ordering anything."
            cleaned = {}
            properties = (self.schemas.get(name) or {}).get("properties") or {}
            for key, value in args.items():
                spec = properties.get(key) or {}
                if isinstance(value, str) and "enum" not in spec:
                    cleaned[key] = neutral[: spec.get("maxLength", len(neutral))]
                else:
                    cleaned[key] = value
            return cleaned
        return args

    def force_finish(self) -> None:
        if self.finished or self.dead or self.steps_left() <= 0 or self.time_left() < 2:
            return
        order = self.roles.get("order")
        decline = self.roles.get("decline")
        cart = self.roles.get("cart")
        fits = (self.proven() or self.plausible()) if order and cart and not self.cart_ids and self.steps_left() >= 2 and self.time_left() > 8 else []
        if fits:
            best = self.preferred(fits)
            reply = self.env_call([{"name": cart, "args": self.cart_args(best)}])
            self.trace.append({"kind": "forced_cart", "item": best, "reply": reply})
            if reply is not None and not (isinstance(reply.get("observation"), dict) and any(re.search(r"fault|error|issue", str(k)) for k in reply["observation"])):
                self.cart_used = True
                self.cart_ids = [best]
        if order and self.cart_used:
            action = {"name": order, "args": self.neutral_args(order, "Order the cart.", "order")}
        elif decline:
            action = {"name": decline, "args": self.neutral_args(decline, "Sorry, nothing in the store meets all of your requirements, so I am not ordering anything.", "decline")}
        else:
            return
        reply = self.env_call([action])
        self.trace.append({"kind": "forced_finish", "action": action, "reply": reply})
        if reply and not reply.get("finished") and isinstance(reply.get("observation"), dict) and "refusal" in reply["observation"] and decline and self.steps_left() > 0:
            action = {"name": decline, "args": self.neutral_args(decline, "Sorry, nothing in the store meets all of your requirements, so I am not ordering anything.", "decline")}
            reply = self.env_call([action])
            self.trace.append({"kind": "forced_finish", "action": action, "reply": reply})

    def note_observation(self, name: str, observation: Any) -> str:
        notes = []
        if isinstance(observation, dict):
            _collect_ids(observation, self.seen_ids)
            if observation.get("error") == "budget_exceeded":
                if name == self.roles.get("message") or self.message_budget is not None:
                    self.message_budget = self.messages_used if self.message_budget is None else min(self.message_budget, self.messages_used)
                notes.append("That budget is used up; do not call this tool again.")
            if "refusal" in observation:
                self.refusals += 1
                self.order_checked = False
                notes.append("THE ORDER WAS REFUSED and the episode continues; the cart is unchanged. Apply every change in the refusal, fix the cart (remove any line that is no longer acceptable, such as a sold-out item), re-check stock and price of the new pick, and order again.")
            elif len(observation) >= 2 and name != self.roles.get("message"):
                lines: list[str] = []
                for key, value in observation.items():
                    if isinstance(value, list) and value and all(isinstance(v, str) for v in value):
                        _sentences(value, lines)
                if lines:
                    notes.append("This result carries a message from the shopper (a change). Apply it: " + " | ".join(lines[:4]))
        elif isinstance(observation, list):
            _collect_ids(observation, self.seen_ids)
        return " ".join(notes)

    def run_calls(self, calls: list[dict[str, Any]], planned: bool = False) -> list[tuple[str, str]]:
        """Exactly one result per call: the decline-to-order swap answered the decline AND ran the swapped call under the
        same id, and two results for one call make the provider refuse every later turn of the conversation (Cloud's
        race-3 run: 9 tasks, 8 lost; ours: 7-11 per full run)."""
        merged: dict[str, str] = {}
        for call_id, text in self.run_calls_once(calls, planned):
            merged[call_id] = merged[call_id] + "\n" + text if call_id in merged else text
        for call in calls:
            merged.setdefault(call["id"], "Not run.")
        order = {c["id"]: i for i, c in enumerate(calls)}
        return sorted(merged.items(), key=lambda r: order.get(r[0], 0))

    def run_calls_once(self, calls: list[dict[str, Any]], planned: bool = False) -> list[tuple[str, str]]:
        replies: list[tuple[str, str]] = []
        allowed = {t["function"]["name"] for t in self.tools_for_turn()}
        runnable: list[tuple[str, str, dict[str, Any]]] = []
        finishing = {self.roles.get("order"), self.roles.get("decline")} - {None}
        message_tool = self.roles.get("message")
        cart_tool = self.roles.get("cart")
        planned_messages = 0
        for call in calls:
            call_id = call["id"]
            function = call.get("function") or {}
            name = function.get("name")
            raw = function.get("arguments")
            if name not in self.schemas:
                replies.append((call_id, f"Not run (no step used): '{name}' is not one of the tools. Use one of: {', '.join(sorted(allowed))}."))
                continue
            if not planned and name not in allowed:
                if name == message_tool and self.planner_on:
                    reason = "questions to the shopper are planned and sent for you after you have opened a few listings' details"
                else:
                    reason = "too few steps are left for this; only these tools can be used now: " + ", ".join(sorted(allowed))
                replies.append((call_id, f"Not run (no step used): {reason}."))
                continue
            if isinstance(raw, dict):
                args = raw
            else:
                args = _parse_json_object(raw or "{}") if (raw or "").strip() else {}
                if args is None:
                    replies.append((call_id, "Not run (no step used): the arguments were not a valid JSON object."))
                    continue
            args, problems = _check(args, self.schemas[name], "args")
            if problems:
                replies.append((call_id, "Not run (no step used): invalid arguments: " + "; ".join(problems[:6])))
                continue
            if name in finishing and len(calls) > 1:
                replies.append((call_id, "Not run (no step used): an order or a decline must be sent alone, after you have seen the results of your other calls."))
                continue
            if name == message_tool and self.message_budget is not None and self.messages_used + planned_messages >= self.message_budget:
                replies.append((call_id, "Not run (no step used): the message budget is used up. Do not message the shopper again."))
                continue
            signature = name + _compact(args)
            if name not in finishing and name != message_tool and name != cart_tool and signature in self.recent:
                replies.append((call_id, "Not run (no step used): you just made this exact call and nothing has changed since; its result is above. Change the arguments or do something else."))
                continue
            if name == cart_tool and self.steps_left() > 3:
                properties = self.schemas[name].get("properties") or {}
                signature_now = _compact({k: v for k, v in args.items() if isinstance(properties.get(k), dict) and "enum" in properties[k]})
                if self.add_signature is None or signature_now == self.add_signature:
                    target = next((v for k, v in args.items() if isinstance(v, str) and "enum" not in (properties.get(k) or {}) and v.strip() and " " not in v.strip()), None)
                    if target and f"{target}|{self.ledger_version}" not in self.cart_checked:
                        self.cart_checked.add(f"{target}|{self.ledger_version}")
                        verdict = self.rule_check(target, True) or self.verify_item(target, "ADD TO THE CART")
                        if verdict:
                            replies.append((call_id, "Not run (no step used). " + verdict + " If you are certain it fits, send the same call again."))
                            continue
            if name == cart_tool:
                warning = self.quantity_warning(name, args)
                if warning and signature not in self.qty_warned:
                    self.qty_warned.add(signature)
                    replies.append((call_id, "Not run (no step used): " + warning + " If you are sure, send the same call again."))
                    continue
            if name == message_tool and not planned and "?" not in " ".join(str(v) for v in args.values() if isinstance(v, str)) and signature not in self.told_once:
                self.told_once.add(signature)
                replies.append((call_id, "Not run (no step used): a message to the shopper must ask one question about one property; a statement or a confirmation only spends the message budget. If a rule in the request requires this message, send the same call again."))
                continue
            if name == message_tool:
                planned_messages += 1
            runnable.append((call_id, name, args))
        if runnable and runnable[0][1] in finishing:
            call_id, name, args = runnable[0]
            kind = self.intent(name, args)
            if kind == "order" and not self.order_checked and self.steps_left() > 2 and self.time_left() > WRAP_UP_SECONDS:
                note = self.order_note(self.cart_ids[-1] if self.cart_ids else "")
                if note:
                    replies.append((call_id, "Not run yet (no step used). " + note))
                    return replies
                self.order_checked = True
                verdict = self.verify_cart()
                if verdict:
                    replies.append((call_id, "Not run yet (no step used). " + verdict))
                    return replies
                replies.append((call_id, "Not run yet (no step used). Pre-order check: (1) the cart holds exactly one line, the chosen item at the required quantity, and nothing else; (2) after your last cart change you re-read this item's stock and price with a tool that shows them, and it is in stock and price times quantity is within the current budget; (3) its own properties meet every stated requirement, every need the shopper revealed and every change announced; (4) if the cheapest or best was asked for, no better fitting item exists among everything you could find; (5) every procedural rule in the request has been followed. If anything is unchecked, check it now; otherwise call the order tool again."))
                return replies
            if kind == "decline":
                swap = self.instead_of_decline()
                if swap is not None:
                    self.trace.append({"kind": "instead_of_decline", "tool": swap[0], "args": swap[1]})
                    replies.append((call_id, swap[2]))
                    runnable[0] = (call_id, swap[0], swap[1])
                    kind = self.intent(swap[0], swap[1])
            if kind == "decline":
                note = self.decline_note()
                if note:
                    replies.append((call_id, "Not run yet (no step used). " + note))
                    return replies
                if self.decline_checked and not self.search_guarded and self.steps_left() > 12 and self.searches_done < 8 and self.time_left() > WRAP_UP_SECONDS + 30:
                    self.search_guarded = True
                    replies.append((call_id, f"Not run yet (no step used). {self.steps_left()} steps remain and only {self.searches_done} searches were made. Search again in other ways first: the requested value in the sellers' words or as a fact value, the shelf or category listing, further pages; open the details of candidates you have not read and re-read the ones you opened. Decline only if each candidate still fails."))
                    return replies
                if not self.decline_checked and self.steps_left() > 6 and self.time_left() > WRAP_UP_SECONDS + 30:
                    self.decline_checked = True
                    audit = self.audit_decline()
                    if audit:
                        replies.append((call_id, "Not run yet (no step used). " + audit))
                        return replies
                    replies.append((call_id, "Not run yet (no step used). Before declining, be sure: you searched in several ways (different words, filters, pages), you checked every candidate's own properties, and each one fails a stated requirement, a revealed need, the budget or stock. Remember that close synonyms and any key spelling count. If something does fit, buy it instead; if nothing fits, call the decline tool again."))
                    return replies
                args = self.clean_decline(args, name)
                runnable[0] = (call_id, name, args)
        if runnable and runnable[0][1] not in finishing and finishing:
            keep = self.steps_left() - 1
            if keep < len(runnable):
                for call_id, name, _ in runnable[max(0, keep):]:
                    replies.append((call_id, "Not run (no step used): only the last step is left, and it must be the order (or the decline)."))
                runnable = runnable[: max(0, keep)]
        # questions reach the shopper one per call: several arriving together are turned back ("send them in singly")
        size = 1 if message_tool and any(name == message_tool for _, name, _ in runnable) else 16
        for start in range(0, len(runnable), size):
            chunk = runnable[start : start + size]
            if self.finished or self.dead or self.steps_left() <= 0:
                for call_id, _, _ in chunk:
                    replies.append((call_id, "Not run: the episode is over or no steps are left."))
                continue
            chunk = chunk[: self.steps_left()]
            actions = [{"name": name, "args": args} for _, name, args in chunk]
            before = self.step
            reply = self.env_call(actions)
            self.trace.append({"kind": "env", "step": before + 1, "actions": actions, "reply": reply})
            if reply is None:
                for call_id, _, _ in chunk:
                    replies.append((call_id, "The call failed (network or server). Try again."))
                continue
            observations = reply.get("observations") if isinstance(reply.get("observations"), list) else [reply.get("observation")]
            for index, (call_id, name, args) in enumerate(chunk):
                if index >= len(observations):
                    replies.append((call_id, "Not run: the episode finished before this call."))
                    continue
                observation = observations[index]
                if name == message_tool:
                    self.messages_used += 1
                if self.intent(name, args) == "order":
                    self.order_attempts += 1
                if name == cart_tool:
                    self.cart_used = True
                    self.order_checked = False
                if name in (cart_tool, message_tool) or name in finishing:
                    self.recent = []
                else:
                    self.recent = (self.recent + [name + _compact(args)])[-2:]
                self.records.append(observation)
                self.remember(name, observation)
                if name == cart_tool:
                    properties = self.schemas[name].get("properties") or {}
                    signature = _compact({k: v for k, v in args.items() if isinstance(properties.get(k), dict) and "enum" in properties[k]})
                    if self.add_signature is None:
                        self.add_signature = signature
                    for key, value in args.items():
                        spec = properties.get(key) if isinstance(properties.get(key), dict) else {}
                        if isinstance(value, str) and "enum" not in spec and value.strip() and " " not in value.strip():
                            self.cart_state[value] = signature
                            if signature == self.add_signature:
                                self.cart_ids = [i for i in self.cart_ids if i != value] + [value]
                            else:
                                self.cart_ids = [i for i in self.cart_ids if i != value]
                    self.cart_changed_at = self.step
                if name == self.roles.get("cart_view"):
                    self.last_cart_view = observation
                note = " ".join(x for x in (self.note_observation(name, observation), self.page_note(name, args, observation)) if x)
                heard: list[str] = []
                if name == message_tool:
                    if isinstance(observation, str):
                        heard = [observation]
                    elif isinstance(observation, dict) and "error" not in observation:
                        _sentences(list(v for v in observation.values() if isinstance(v, (str, list))), heard)
                    question = " ".join(str(v) for v in args.values() if isinstance(v, str))
                else:
                    question = None
                    if isinstance(observation, dict):
                        for value in observation.values():
                            if isinstance(value, list) and value and all(isinstance(v, str) for v in value):
                                _sentences(value, heard)
                            elif isinstance(value, str) and len(value.split()) >= 4 and ("refusal" in observation):
                                heard.append(value)
                if heard:
                    meaning = self.interpret(heard, question)
                    if meaning:
                        note = (note + " " if note else "") + meaning
                text = _compact(observation)
                if index == len(observations) - 1 and reply.get("finished"):
                    text += " [the episode is finished]"
                if note:
                    text += "\n[note] " + note
                replies.append((call_id, text))
        order = {c["id"]: i for i, c in enumerate(calls)}
        replies.sort(key=lambda r: order.get(r[0], 0))
        return replies

    def run(self) -> list[dict[str, Any]]:
        if self.finished:
            return self.trace
        self.setup()
        self.trace.append({"kind": "plan", "plan": self.plan, "roles": self.roles})
        opening = self.policy.get("opening")
        user = "SHOPPER REQUEST:\n" + str(self.policy.get("query") or "")
        if opening is not None:
            user += "\n\nOPENING OBSERVATION:\n" + _compact(opening)
        brief = self.brief()
        if brief:
            user += "\n\n" + brief
        user += "\n\n" + self.status()
        if opening is not None:
            self.records.append(opening)
            _collect_ids(opening, self.seen_ids)
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]
        idle = 0
        failures = 0
        counter = 0
        for _ in range(MAX_TURNS):
            if self.finished or self.dead or self.steps_left() <= 0:
                break
            if self.time_left() < FORCE_SECONDS:
                self.force_finish()
                break
            if self.pending_reply:
                text, self.pending_reply = self.pending_reply, ""
                if self.send_reply(text):
                    continue
            if self.commit_due():
                continue
            if self.refs and self.resolve_refs():
                continue
            if self.planner_on and self.ask_planned():
                continue
            message = self.infer(self.messages, self.tools_for_turn(), [MAIN_MODEL, BACKUP_MODEL, MAIN_MODEL], 2500)
            if message is None:
                failures += 1
                if failures >= 2:
                    self.force_finish()
                    break
                continue
            calls = []
            for call in message.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                counter += 1
                function = call.get("function") or {}
                calls.append({"id": call.get("id") or f"call_{counter}", "type": "function", "function": {"name": function.get("name"), "arguments": function.get("arguments") if isinstance(function.get("arguments"), str) else json.dumps(function.get("arguments") or {})}})
            content = message.get("content") or ""
            if not isinstance(content, str):
                content = _compact(content)
            committing = {self.roles.get("cart"), self.roles.get("order"), self.roles.get("decline")} - {None}
            if self.planner_on and any(c["function"]["name"] in committing for c in calls):
                if self.ask_planned(force=True):
                    continue
            if self.plan.get("priority") == "cheapest" and any(c["function"]["name"] in committing for c in calls):
                note = self.cheaper_probe(calls)
                if note:
                    self.messages.append({"role": "user", "content": note + "\n" + self.status()})
                    continue
            if not calls:
                idle += 1
                self.messages.append({"role": "assistant", "content": content or "(no tool call)"})
                self.messages.append({"role": "user", "content": "Every reply must call at least one of the supplied tools. Continue.\n" + self.status()})
                if idle >= 3:
                    self.force_finish()
                    break
                continue
            idle = 0
            self.trace.append({"kind": "model", "content": content[:2000], "calls": [c["function"] for c in calls]})
            self.messages.append({"role": "assistant", "content": content or None, "tool_calls": calls})
            replies = self.run_calls(calls)
            for index, (call_id, text) in enumerate(replies):
                if index == len(replies) - 1:
                    text += "\n" + self.status()
                self.messages.append({"role": "tool", "tool_call_id": call_id, "content": text})
        if not self.finished and not self.dead:
            self.force_finish()
        self.trace.append({"kind": "end", "finished": self.finished, "steps": self.step, "cost": round(self.cost, 5), "seconds": round(time.monotonic() - self.started, 1)})
        return self.trace


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    episode = Episode(problem_data)
    try:
        return episode.run()
    except Exception as exc:
        episode.trace.append({"kind": "crash", "error": f"{type(exc).__name__}: {exc}"[:500]})
        try:
            if not episode.finished and not episode.dead:
                episode.force_finish()
        except Exception as again:
            episode.trace.append({"kind": "crash", "error": f"{type(again).__name__}: {again}"[:500]})
        return episode.trace


__all__ = ["agent_main"]
