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
OBS_CHARS = 9000
MAX_TURNS = 80
ROW_CHARS = 260
NOTE_CHARS = 600
SPEND_SOFT = 0.050
SPEND_HARD = 0.060
SPEND_STOP = 0.075

SYSTEM_PROMPT = """You are a careful shopping assistant working for one shopper in a simulated store. You act only through the supplied tools. The episode is graded 1 or 0, and it earns 1 only when the finish is exactly right: every requested product is ordered once, at its quantity, and each ordered listing meets every requirement the shopper stated, every hidden need the shopper reveals when asked, and every change the shopper announces along the way; the total stays within the budget; the cheapest (or best) fitting listing is chosen when the request asks for that; and every procedural rule in the request is followed. If the store truly cannot satisfy the request, the right finish is a decline made exactly as the request says.

HOW THE STORE WORKS
- Each tool call costs one step, failed or refused calls included, and steps are limited. Thinking is free; steps are not.
- Several independent calls may go in one reply (for example opening several listings' details); they run in order. An order or a decline always goes alone.
- If a tool shows the details of several listings in one call (a compare tool), read candidates with it in batches: one step for many listings.
- Tool names, argument names and result shapes change from store to store. Read the schemas and use exactly the names and allowed values they define. Learn from the first results which tool shows which facts: a details tool usually shows a listing's property facts, another tool may show its current price, stock and a seller note.
- Learn the store's category labels for each requested product from the listings, and stay inside the label that matches the request's wording as a whole.
- Search results can be capped (only a few pages visible), ordered by id instead of price or relevance, and every search word may be required; some modes ignore some arguments (words in a filter mode, filters in a word mode). Words can match property values, so adding a required value to the search words narrows to listings that carry it. Category and price-cap filters shrink a large set until every remaining row is visible. Compare the reported match count with what you have actually seen, and keep narrowing (more specific words, category or price filters, further pages) until you have seen the whole relevant set. Row descriptions often state facts: use them to choose which listings to open. Spell values in search words the way the listings' facts spell them (a unit written out in full in the facts should be written out in full in the search).
- Listings with the same title and price from different sellers are separate listings whose facts can differ: open a twin before giving up on a product, and after a sell-out a twin is often the next best pick.

SEVERAL PRODUCTS IN ONE REQUEST
- A request can ask for two or more different products, each with its own quantity and its own requirements. Treat each as a separate line: find, verify and cart each one. The cart at the order must hold exactly one line per requested product, each at its quantity, and nothing else. A requirement belongs only to the product it was stated or revealed for.

THE SHOPPER AND HIDDEN NEEDS
- The shopper holds back some needs and reveals one only when a message asks about that very property (of that product, when there are several products). Questions are planned and sent for you once you have opened the details of a few listings of each requested product, so open those first.
- A question that touches two or more properties the shopper keeps track of is not answered (the shopper asks for one question at a time); a question about a property that is not one of their needs gets a "don't mind" or a puzzled "say it simpler" reply. Neither reply means anything except that this property is free: never rephrase or repeat it.
- Every revealed value is a hard requirement from then on, exactly as worded (limits included). A property the shopper does not care about is free.

SELLER NOTES AND CROSS-REFERENCES
- A seller note can name some other listing as the source of one of this listing's properties. Then this listing's own fact for that property does not count: its real value is the other listing's value. Do not rely on such a listing for that property unless you open the referenced listing and its value fits.
- A shopper can define a wanted value by another listing. When the value must stand apart from, differ from or not equal the value of another seller's listing (including a listing that one's own note refers to), the wanted value must be DIFFERENT from that value. When it must match another listing, it is the SAME value. In both cases find the referenced listing (seller and brand names can be searched together with the product words; the attributes given only identify that other listing and are NOT requirements on what you buy), read its value (following its seller note if it points elsewhere), then require a different value or the same value. Never decline because no listing of the product you buy carries the referenced listing's attributes.

WORKING RULES FROM THE REQUEST
- The request may set rules on how to work (how many listings to compare, what to check, what to say, how to decline). Follow each literally; they are graded like requirements. A rule to look at or open N listings means N listings that fully fit, each opened with the details tool.
- If the request leaves something only the shopper can settle (for example two wishes that may not both be possible), find it out by asking before you choose.
- If the shopper asks you something, it is answered for you while messages remain; treat any change they confirm as binding.
- Wishes ("ideally", "if possible", "would be nice") are not requirements; use them only to choose among listings that already fit, never over a cheapest rule.

CHANGES WHILE YOU WORK
- The shopper may change the budget, drop or replace a requirement, or report an item sold out. A line of shopper text can arrive inside any tool result or with a refused order or decline. Every change binds from then on: re-check your picks and backups. After a requirement is dropped or the budget rises, listings you set aside only for that reason are back in play; when the request asks for the cheapest or best, choose again over the new set of fitting listings.
- An order can be REFUSED: the result carries a refusal and the episode continues with the cart unchanged. Apply every change it states, take out anything no longer acceptable, put in the next verified fit, re-check stock and price, and order again.
- Putting a listing in the cart can sell it out without any message. After your last cart change, re-check every cart line's stock with a tool that shows stock before ordering; a sold-out line must be replaced by the next verified fit for that product.

HOW A LISTING IS JUDGED (this decides the grade)
- A requirement is met only when the listing's own fact for that property states the value. Keys vary in spelling (spaces, underscores, prefixes) and one property may sit under several keys; any key naming the property counts. Values match by meaning: an element of a comma list counts, singular/plural and close synonyms count, a more specific wording of the same value counts. A combined value naming several different things ("<a> + <b>") is not the same as one of them alone, and a broader, narrower or different class does not count (for example girls is not women, plain unisex is not female).
- Use the title or description as evidence only when the listing has no fact for that property at all. When its own fact says something else, it fails, whatever the title says.
- Never reject a fitting listing for a reason neither the request nor the shopper stated (who it seems made for, its size range, how it looks, what you guess the shopper would prefer): only stated requirements, revealed needs and the category count, and story details are not requirements.
- A value under a different, related property does not count (the material of one part does not settle the material of the whole), but a product-specific key for the same property (for example "<product> style" for a style requirement) does count.
- Numbers are compared as numbers after converting units; limits include the edge; a range counts only when all of it meets the limit.
- Respect exclusions and closed lists. The requested kind of product is met by the listing's category; never demand that some property repeat the category's name, except as follows. The listing must be the requested kind (its category label), in stock, and price times quantity must keep the total within the budget. When the category label joins several kinds ("<kind A> and <kind B>") and the request asks for only one of them, the listing must also state that kind in one of its facts (a type, kind or style key) or in its title.
- Mixed evidence exists only when the listing's own keys for one property disagree; then prefer a clean candidate when no cheapest or best rule decides.

CHEAPEST OR BEST
- Cheapest means the lowest-priced listing among ALL that meet every requirement; seeing one fitting listing is not enough. Narrow with category and price-cap filters until every remaining row is visible, sort by price yourself and verify upward from the cheapest; the first that fully fits, after every cheaper one was checked and rejected, is the answer. Keep the next fits as backups.
- Best by a measure means the extreme value among all fitting listings; a listing that states no value for the measure cannot win.

FINISHING
- Keep verified fits with backups for every product: a pick can sell out at checkout.
- Before each order attempt: one cart line per requested product at its quantity and nothing else; every line's stock and price re-read after your last cart change; every line fits its own product's requirements; cheapest/best rules respected; all procedural rules done.
- Decline only after a real search finds nothing that fits; a decline scores 0 whenever some listing could have fitted, so between ordering a fit and declining, order. In a request for several products, one hard product is a reason to search harder for it, never a reason to give up on the whole order. Follow the request's decline rule; if substitutes must not be offered, the decline text names no listing at all.
- Place the first order attempt while at least seven steps remain so a refusal can be repaired. Near the end only cart, stock-check and finishing tools stay available. Do not swap the cart back and forth.
- Every reply must call at least one tool. Keep your text short: current requirements per product, verified fits with price and stock, next move."""

SETUP_PROMPT = """Read the shopper's request and the store's tool list below. Return ONE compact JSON object (no indentation) and nothing else, with these keys:

"lines": one object per distinct product the shopper wants bought, in the order the request names them: {"product": the kind of product in the request's words, "quantity": units of one listing to order for it (integer; 1 unless several units of that product are asked for; a pair or a set sold as one item is 1), "requirements": the firm stated requirements on THAT product only, each as "property: constraint" with the shopper's exact value, limit, exclusion or closed list; a value defined by another listing stays as written, for example "property: same as the <listing described>" or "property: different from the <listing described>"}. Price, stock, quantity, budgets, delivery and wishes are never product requirements, and neither is the product kind itself (words that are part of the category's name are covered by the category). Only conditions the request states outright go here; a situation the story merely describes goes in "hints". An occasion or use the request names as what the product itself is for ("these are for a graduation ceremony", "one meant for a rooftop dinner") is a requirement such as "occasion: graduation"; a passing setting of the story ("I usually carry it on the bus") is not.
"product": the first line's product. "quantity": the first line's quantity. "requirements": the first line's requirements.
"hints": a list of objects {"product": which line it concerns, "property": ..., "likely": ..., "why": ...}: properties the story points at without stating a value, named as a listing would name them, with the implied value in "likely" ("" when none) and the request's own words in "why". Hints choose what to ask first and break ties; they never reject a listing. Empty when the request is plain.
"price_cap": the maximum total price for the whole order as a number, or null when none is stated.
"currency": the price unit the request uses, or null.
"in_stock_only": true when the request wants only items actually on hand.
"priority": "cheapest" only when the request explicitly asks for the cheapest or lowest-priced fitting option; "best" only when it explicitly asks for the best by some measure; otherwise "none". Saying money is tight is not a cheapest request.
"priority_detail": for "best", the measure; otherwise "".
"message_budget": how many messages the shopper allows, as an integer, or null.
"step_budget": how many moves or steps the request allows, as an integer, or null.
"rules": every procedural rule the request sets for the assistant, quoted word for word. What the shopper will do themselves is story, not a rule.
"soft_wants": things the shopper would like but does not require (ideally, preferably, if possible).
"ask_first": things only the shopper can settle and that must be asked before choosing (for example which of two wishes matters more when the request says both may not be possible; then both wishes also go in soft_wants). Empty when none.
Every sentence of the request must land somewhere: a line's requirement, the priority, a soft want, a rule, a budget, a hint, or story. When unsure whether a sentence is a rule, put it in "rules".
"decline_no_names": true when the request's decline rule forbids naming or suggesting substitutes; false when it asks you to name something; null when it gives no rule.
"roles": an object naming the tool for each role, with exact tool names from the list, or null when no tool fits: "search", "details" (shows one listing's property facts), "view" (shows one listing's current price and stock, and its seller note if any), "message" (sends a message to the shopper), "cart" (adds to or removes from the cart), "cart_view", "order" (buys or checks out), "decline" (ends without buying).
"finish_args": when one tool both buys and ends without buying, an object {"order": {...}, "decline": {...}} with the argument values that select each; null otherwise.
"""

INTERPRET_PROMPT = """You help a shopping assistant read what a shopper just said. Return ONE JSON object and nothing else, with these keys:
"meaning": one short sentence with what the shopper means.
"needs": new firm requirements on a product, each as "property: constraint" with the exact value, limit, exclusion or closed list given. When the request asks for several products, start each with the product it concerns in square brackets, for example "[<product>] property: constraint" (the shopper's line usually names the product; a question about one product is answered about that product).
"no_preference": properties the shopper said they do not care about (the property the assistant asked about, when the reply only says they don't mind or anything works). Such a reply never cancels a requirement the request itself states, even when the question happened to touch it: it frees only properties that are not already required.
"too_many": true when the shopper asks for one question at a time or for fewer things per message, else false.
"puzzled": true when the shopper says they did not understand and wants it said more simply, else false. Such a reply reveals no need; it means the property asked about is not one of their needs.
"dropped": earlier requirements the shopper no longer needs.
"new_budget": the new maximum total price as a number, or null.
"unavailable": true when the shopper says an item can no longer be bought (sold out, gone, none left), else false.
"requirements_now": the complete list of firm requirements as they stand after everything said so far, re-derived from the request, the earlier shopper lines and this one, each as "property: constraint" (with the "[<product>] " prefix when the request asks for several products). Keep every earlier requirement unless dropped or replaced; for a replaced value keep only the newest; add every newly revealed need. No price, stock, quantity or category here, and nothing from the story or your own reading: only what the request states outright and what the shopper said.
"lookups": a list of strings, one per requirement that depends on another listing (see below), describing which listing must be found, for example "<seller>'s <product> with <attributes>: read its <property>".
"reply": when the shopper asks the assistant a real question about the order or asks for a confirmation of a change, a short plain reply that answers it and names the change in substance; otherwise "". Never write a reply when the shopper only says they did not understand, asks you to say it more simply or differently, or asks for one question at a time: such a line is a complete answer (the property asked about is not one of their needs) and must not be rephrased or repeated.
One line can carry several changes: report every one. A replaced requirement is both dropped (old) and a new need (new). When the shopper says which of two wishes matters more, that wish is a need. A stated budget is new_budget, higher or lower. A wish is not a need. Read figurative or folksy wording charitably: a named value is wanted firmly unless the shopper clearly says to avoid it. A number with a comparison word is a limit with its unit.
A value defined by another listing, worded as standing apart from, differing from or not equalling the value of another seller's listing (including a listing that one's own note refers to), means the wanted value must DIFFER from the value that other listing shows for this property (or from the listing its seller note refers to for this property). Write it as "property: different from the <property> of <seller>'s <product> (<attributes>)" and add a lookup for it; the attributes only identify that other listing and are never requirements of their own. A value worded as matching another listing is written "property: same as ..." with a lookup too."""

VERIFY_PROMPT = """You are the last check before a shopping assistant orders. The order earns credit only if the listing meets every requirement of the product line it is bought for. Judge it ONLY from the records shown, with these reading rules:
- A requirement is met only when the listing's own fact for that property states the value. Any key spelling naming the property counts and one property may appear under several keys. Values match by meaning: one element of a comma list is enough; singular/plural, close synonyms and a more specific wording of the same value count; case does not matter. A merely similar-looking word, a combined value naming several different things ("<a> + <b>" is not <a>), or a broader/narrower/different class does not count.
- The title or description counts only when the listing has no fact for that property. If its own fact says something else, it fails even when the title agrees.
- A value under a different, related property does not count; but a product-specific key for the same property (for example "<product> style" for style) is a key for that property. When one key for the property states the value and another key for it states something else, that is mixed evidence, not a failure: return fits=true and note "mixed evidence" in problems.
- If the listing's seller note says a property must be taken from another listing, the listing's own fact for that property does not count; unless the records show the referenced listing's value fits, that requirement is not met.
- "different from X" requirements are met by any stated value other than X.
- Numbers are compared after converting units; limits include the edge; a range counts only if all of it meets the limit.
- Exclusions and closed lists must be respected.
- The listing must be the requested kind of product for its line: when listings carry category labels, the label must match that product's wording as a whole, not a narrower or neighbouring label; when the label joins several kinds and the line asks for only one of them, a fact or the title must state that kind.
- A requirement whose value is part of the category label itself (for example a season or an audience named in the category) is met by the category.
- Price times the line's quantity must fit the budget (equal is fine); if the newest record shows it out of stock, it fails.
- When records disagree on price or stock, the newest counts. Wishes are not requirements.
Return ONE JSON object and nothing else: {"item": the listing's id, "line": the product it is for, "fits": true or false, "problems": [short strings, each naming a requirement not met or not shown]}. Use fits=false only for a requirement the records clearly fail or that no record addresses; never fail it for a missing stock or price record."""

PLAN_PROMPT = """You plan the questions a shopping assistant sends to a shopper. The shopper has hidden needs: each is a value of one property of the product (of one of the products, when several are requested), revealed only when a message asks about that exact property; a message that touches two or more of the shopper's properties is not answered, and questions about anything else get a polite "don't mind". Choose up to QUESTION_COUNT different properties, most promising first:
0. first, anything listed below as MUST BE LEARNED FROM THE SHOPPER that has not been asked yet; ask it plainly. For a choice between two wishes, name both values with their properties, for example "If one listing can't have both, would you rather have the <value> <property> or the <value> <property>?";
1. then properties that the listings' details below actually carry, preferring (a) a property the story in the request hints at indirectly (who the item is for or who will wear or use it, how it must look in photos or to others, where or how it will be used), and (b) properties that describe what the product is like and whose values vary across the listings (who it is made for, pattern, shape, type or style variant, closure, length, function) rather than logistics (warranty, condition, pack size, variant or size lists, years). Never ask about the brand, the seller, the store, the listing's title or name, its id or its price: they are never the shopper's needs. When CORE PROPERTIES are listed below (most of the listings opened carry them), choose only among them.
2. when several products are requested, give every product a question before any product gets a second one, and say which product each question is for.
Never pick a property the request already settles (its stated requirements, the price, stock, quantity), never repeat a property already asked, never spend two questions on near-duplicate properties (two kinds of style, two spellings of one property, a general and a specific name for one property), and avoid properties no listing carries. Use the property names as the listings name them (spaces instead of underscores).
Return ONE JSON object and nothing else: {"questions": [{"product": "...", "property": "...", "question": "..."}, ...]}."""

AUDIT_PROMPT = """A shopping assistant is about to tell the shopper that nothing in the store fits. Check the listing records below against the requirements of each requested product with these reading rules: a requirement is met when the listing's own fact for that property states the value (any key spelling, one element of a comma list, singular/plural and close synonyms count; a combined value of several things, or a broader or different class, does not); the title or description counts only when the listing has no fact for that property; a property the seller note sends to another listing is not shown by this listing; numbers are compared after converting units and limits include the edge; the price must fit the budget and the listing must not be shown out of stock.
Return ONE JSON object and nothing else: {"possible_fits": [ids of listings that appear to meet every requirement of their product, best first, at most 5], "reason": "one sentence"}."""

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
    """How a listing's facts read for one requirement: a value counts when a fact named for the property states it, or
    the title or variant options when no fact is named for it; a named fact stating something else wins over the
    title; an excluded value stated there fails; a bound is met by a figure inside it and failed by a comparable
    figure outside it. Returns 'yes', 'no' or 'unshown', with the evidence read."""
    facts = next((v for v in sheet.values() if isinstance(v, dict)), {})
    pairs = [(str(k), str(v)) for k, v in list(sheet.items()) + list(facts.items()) if isinstance(v, (str, int, float)) and not isinstance(v, bool)
             and not re.fullmatch(r"(?i)\s*(?:n/?a|none|not specified|unknown|null|-|)\s*", str(v))]
    named = [(k, v) for k, v in pairs if not _TITLE_KEY.search(k) and _names(k, prop, spare)]
    texts = [v for _, v in named] or [v for k, v in pairs if _TITLE_KEY.search(k) and not _SHELF_KEY.search(k)]
    shown = "; ".join(f"{k}: {v[:60]}" for k, v in named[:2]) or "no fact named for it"
    bound = _bound(value)
    excluded = re.match(r"^(?:not|no|without|anything but|anything except|anything other than|except|excluding|avoid|any (?:[a-z]+ )?(?:except|but))\s+(.+)$", value)
    if excluded and bound is None:
        if any(_stated(value, t) for t in texts):
            return "yes", shown                                      # the fact states the negated value itself ("not available", "no sugar")
        term = excluded.group(1)
        head = re.escape((_words(term, False) or [term])[0])
        for t in texts:
            if _stated(term, t):
                if re.search(r"(?i)\b" + head + r"[\s-]*free\b", t) or re.search(r"(?i)\b(?:no|not|non|without)[\s-]+" + head, t):
                    continue                                          # the excluded thing is stated as absent ("sugar free", "no sugar")
                return "no", shown
        if re.fullmatch(r"[a-z]+(?:able|ible)", term.strip()):
            absent = named and any(re.search(r"(?i)\b(?:no|not|none|without|n/?a)\b", t) for t in texts)
            return ("yes" if absent else "unshown"), shown            # an absence ("not available") is not shown by a concrete value: left to the model check
        return "yes", shown
    if bound is not None:
        low, high = bound
        unit = next((u for _, u in _figures(value) if u), "")
        verdicts = []
        for t in texts:
            found = [(n, u) for n, u in _figures(t) if not unit or not u or _same(u, unit)]
            if not found:
                continue
            fits = [(low is None or n >= low) and (high is None or n <= high) for n, _ in found]
            ranged = re.search(r"\d\s*(?:-|to|\u2013)\s*\d", t)
            verdicts.append(all(fits) if ranged else any(fits))     # a range meets a limit only when all of it does
        if any(verdicts):
            return "yes", shown
        return ("no" if verdicts and named else "unshown"), shown
    options = [a for a in re.split(r"\s+or\s+|(?<=[a-z])\s*/\s*(?=[a-z])|,\s*(?!\d)", value) if a.strip()] or [value]
    matching = [t for t in texts if any(_stated(a, t) for a in options)]
    if matching:
        for text in matching:
            parts = [p for p in re.split(r"\s*\+\s*", text) if p.strip()]
            if len(parts) < 2 or all(any(_stated(p, a) for a in options) for p in parts):
                return "yes", shown                                 # a value joined with '+' names several things: it matches only a requirement naming them all
        return "no", shown
    if not named:
        return "unshown", shown
    wanted = [n for n, _ in _figures(value)]
    if wanted and any(_figures(t) and all(not u for _, u in _figures(t)) and [n for n, _ in _figures(t)] == wanted for t in texts):
        return "yes", shown                                          # exactly the same number(s), written without a unit
    units = {u for _, u in _figures(value) if u}
    if units and any(u and u not in units for t in texts for _, u in _figures(t)):
        return "unshown", shown
    return "no", shown


_PRICE_KEYS = (re.compile(r"(?i)price|cost"), re.compile(r"(?i)amount|fee|charge|\busd\b|cents|dollars"))
_NOT_PRICE = re.compile(r"(?i)total|payable|sum|count|qty|quantit|units|points|discount|saving|sold|ship|deliver|tax")
_REF_PRICE = re.compile(r"(?i)original|was_|msrp|compare|list_?price|before|regular")


def _price(record: dict[str, Any]) -> float | None:
    """A listing's unit price: a key named for price or cost, else for an amount or charge (stores rename it); a
    reference price (original, list) only when nothing else is shown."""
    for strict in (True, False):
        for index, pattern in enumerate(_PRICE_KEYS):
            for key, value in record.items():
                name = str(key)
                if not pattern.search(name) or _NOT_PRICE.search(name) or (strict and _REF_PRICE.search(name)) or (index and re.search(r"(?i)number", name)):
                    continue
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    return float(value)
    return None


def _in_stock(record: dict[str, Any]) -> bool | None:
    for key, value in record.items():
        name = str(key)
        if re.search(r"(?i)low|restock|back.?in|threshold|alert|warn|date|eta|until", name):
            continue
        if isinstance(value, str):
            if not re.search(r"(?i)stock|avail|inventor|on.?hand|status|left|remain", name):
                continue
            text = value.strip().lower()
            if re.search(r"sold.?out|out.?of.?stock|unavailab|not available|no stock|none left|discontinu|\b0\s*(?:left|available|in stock)", text):
                return False
            if re.search(r"in.?stock|^available|^yes$|^true$|\b[1-9]\d*\s*(?:left|available|in stock)", text):
                return True
            continue
        if not isinstance(value, (bool, int, float)):
            continue
        flag = bool(value) if isinstance(value, bool) else value > 0
        if re.search(r"(?i)sold.?out|out.?of.?stock|unavailab", name):
            return not flag
        if re.search(r"(?i)stock|avail|on.?hand|in.?store|inventor|left|remain", name):
            return flag
    return None


def _faulty(observation: Any) -> bool:
    """A tool result that reports a failure (an error, fault or refusal key with content)."""
    return isinstance(observation, dict) and any(re.search(r"(?i)fault|error|refus", str(k)) and v for k, v in observation.items())


def _shelf(record: dict[str, Any]) -> str:
    return next((v for k, v in record.items() if _SHELF_KEY.search(str(k)) and isinstance(v, str) and v.strip()), "")


_REDIRECT_CUE = re.compile(r"(?i)not (?:[a-z]+ )?(?:to |from |by |on )?this (?:item|listing|one|product)|rather than (?:to |from |on )?this (?:item|listing|one|product)|(?:comes?|taken|read) from (?:that|another|a different|the other) (?:product|listing|item)|from (?:another|a different) (?:product|listing|item)|instead of this (?:item|listing|one|product)")
_REDIRECT_PROP = [
    re.compile(r"(?i)\bfor (?:its |the |this item's )?([a-z][a-z0-9 /&-]{1,40}?)\s*,\s*(?:refer|go|take|look|use|check|see|read|consult|rely)"),
    re.compile(r"(?i)\b(?:its |the )?([a-z][a-z0-9 /&-]{1,40}?) (?:is|are|should be|must be|has to be) (?:to be )?(?:taken|read|copied|sourced|checked|found) (?:from|on|in)"),
    re.compile(r"(?i)\b([a-z][a-z0-9 /&-]{1,40}?) (?:is not taken|does not come|is not read) from this"),
    re.compile(r"(?i)\bthe ([a-z][a-z0-9 /&-]{1,40}?) (?:comes|is to come) from"),
]


def _redirected_props(text: str) -> list[str]:
    """Properties a seller note says come from some other listing (a property name next to a pointer away from this
    listing)."""
    if not isinstance(text, str) or not _REDIRECT_CUE.search(text):
        return []
    found: list[str] = []
    for pattern in _REDIRECT_PROP:
        for match in pattern.finditer(text):
            prop = re.sub(r"^(?:and|but|while|its|the)\s+", "", match.group(1).strip().lower())
            if prop and len(prop.split()) <= 5 and prop not in found:
                found.append(prop)
    return found


_ORDER_WORD = re.compile(r"(?i)check.?out|order|buy|purchase|place|commit|confirm|pay|settle|complete|submit")
_DECLINE_WORD = re.compile(r"(?i)cancel|declin|refus|stop|abandon|give.?up|quit|reject|abort|skip|none|nothing|no.?buy")


def _finish_choice(values: list[str], kind: str | None) -> str:
    """The enum value of a finishing tool that means ordering or declining (by its own wording), else the first."""
    if kind == "decline":
        pick = next((v for v in values if _DECLINE_WORD.search(v)), None)
    elif kind == "order":
        pick = next((v for v in values if _ORDER_WORD.search(v) and not _DECLINE_WORD.search(v)), None)
    else:
        pick = None
    return pick or values[0]


def _trim(value: Any, depth: int = 0, in_rows: bool = False) -> Any:
    """A lighter copy of a tool result for the running conversation: long texts in result rows are cut, single records
    keep more (a seller note must stay readable)."""
    if depth > 8:
        return None
    if isinstance(value, str):
        limit = ROW_CHARS if in_rows else NOTE_CHARS
        return value if len(value) <= limit else value[:limit] + "..."
    if isinstance(value, dict):
        return {k: _trim(v, depth + 1, in_rows) for k, v in value.items()}
    if isinstance(value, list):
        rows = len(value) >= 3 and all(isinstance(v, dict) for v in value)
        return [_trim(v, depth + 1, in_rows or rows) for v in value]
    return value


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
        self.lines: list[dict[str, Any]] = []
        self.stock_seen: dict[str, tuple[int, bool]] = {}
        self.added_at: dict[str, int] = {}
        self.redirects: dict[str, list[str]] = {}
        self.queue: list[dict[str, str]] = []
        self.plan_gap: list[int] = []
        self.planned_once = 0
        self.reasked: set[str] = set()
        self.last_flags: dict[str, Any] = {}
        self.lookups: list[str] = []
        self.stock_rounds = 0
        self.stock_unshown: set[str] = set()
        self.capped_below: dict[str, float] = {}
        self.sweep_told: set[str] = set()
        self.ignored_told: set[str] = set()
        self.cap_template: dict[str, Any] | None = None
        self.laddered: set[str] = set()
        self.guard_ok = ""
        self.cart_removed: set[str] = set()
        self.cart_counts: dict[str, int] = {}
        self.vetoed: set[str] = set()
        self.readd_told: set[str] = set()

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
        message = self.infer([{"role": "user", "content": content}], None, [SETUP_MODEL, BACKUP_MODEL], 3200)
        plan = _parse_json_object((message or {}).get("content") or "") or {}
        if not isinstance(plan.get("roles"), dict) or not isinstance(plan.get("lines"), list):
            self.trace.append({"kind": "setup_retry", "got": str((message or {}).get("content") or "")[:300]})
            again = self.infer([{"role": "user", "content": content}], None, [SETUP_MODEL, MAIN_MODEL], 4000)
            retry = _parse_json_object((again or {}).get("content") or "") or {}
            if isinstance(retry.get("roles"), dict):
                plan = retry
        self.plan = plan
        roles = plan.get("roles") if isinstance(plan.get("roles"), dict) else {}
        self.roles = {k: v for k, v in roles.items() if isinstance(v, str) and v in self.schemas}
        modes = plan.get("finish_args")
        if isinstance(modes, dict):
            self.finish_args = {k: v for k, v in modes.items() if k in ("order", "decline") and isinstance(v, dict)}
        budget = plan.get("message_budget")
        if isinstance(budget, (int, float)) and not isinstance(budget, bool) and budget >= 0:
            self.message_budget = int(budget)
        lines: list[dict[str, Any]] = []
        for item in plan.get("lines") or []:
            if not isinstance(item, dict) or not str(item.get("product") or "").strip():
                continue
            quantity = item.get("quantity")
            quantity = int(quantity) if isinstance(quantity, (int, float)) and not isinstance(quantity, bool) and quantity >= 1 else 1
            lines.append({"product": str(item["product"]).strip(), "quantity": quantity, "requirements": [str(r) for r in item.get("requirements") or [] if r]})
        if not lines:
            quantity = plan.get("quantity")
            quantity = int(quantity) if isinstance(quantity, (int, float)) and not isinstance(quantity, bool) and quantity >= 1 else 1
            lines = [{"product": str(plan.get("product") or "the product"), "quantity": quantity, "requirements": [str(r) for r in plan.get("requirements") or [] if r]}]
        self.lines = lines
        plan["product"], plan["quantity"] = lines[0]["product"], lines[0]["quantity"]
        multi = len(lines) > 1
        self.stated = [(f"[{line['product']}] {r}" if multi else r) for line in lines for r in line["requirements"]]
        self.budget_now = plan.get("price_cap")
        steps = plan.get("step_budget")
        if isinstance(steps, (int, float)) and not isinstance(steps, bool) and 0 < steps < self.max_steps:
            self.max_steps = int(steps)

    def brief(self) -> str:
        plan = self.plan
        if not plan:
            return ""
        lines = ["A first reading of the request (check it against the request itself):"]
        if len(self.lines) > 1:
            lines.append(f"- {len(self.lines)} different products are requested; the order must hold one cart line for each:")
        for index, line in enumerate(self.lines, 1):
            lines.append(f"- product {index}: {line['product']} (quantity {line['quantity']})")
            for requirement in line["requirements"]:
                lines.append(f"  - stated requirement: {requirement}")
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
        if self.cost > SPEND_HARD and left > 5:
            text += " SPENDING LIMIT NEAR: stop exploring; put the best verified fit for every product in the cart, re-check stock, and order now."
        if left <= 2:
            text += " LAST STEPS: order now if the cart holds a fit; decline only when no listing fits at all."
        elif left <= 5 or self.time_left() < WRAP_UP_SECONDS:
            text += " FINISH NOW: fix the cart to one verified line, re-check its stock, then order (or decline if nothing fits)."
        elif self.order_attempts == 0 and left <= 14:
            text += " Steps are running low: within the next few steps put your best verified fit in the cart and order; searching and reading details end when 7 steps remain, and a refused order needs about 4 steps to repair."
            if left <= 12:
                picks = []
                for line in range(max(1, len(self.lines))):
                    found = self.proven(line) or self.plausible(line)
                    if found:
                        picks.append(self.preferred(found, line))
                if picks:
                    text += " Best candidates found so far: " + ", ".join(picks) + ". Cart them and order now unless one is known to fail."
        return text

    def ledger(self) -> str:
        if self.current is not None:
            items = list(self.current)
        else:
            items = [f"{r} (stated)" for r in self.stated if r not in self.dropped]
            items += [f"{r} (revealed by the shopper)" for r in self.revealed if r not in self.dropped]
        covered = []
        for line in range(max(1, len(self.lines))):
            shelf = self.top_shelf(line) if self.sheets else ""
            for item in items:
                value = item.split("]", 1)[-1].split(":", 1)[-1].split("(")[0].strip()
                if value and self.category_says(line, value) and item not in covered:
                    covered.append(item)
                    items = [f"{i} (already stated by the category '{shelf}': every listing in it meets this)" if i == item else i for i in items]
        text = "; ".join(items) if items else "see the request"
        if self.budget_now is not None:
            text += f"; total price at most {self.budget_now} {self.plan.get('currency') or ''}".rstrip()
        if len(self.lines) > 1:
            text += "; one cart line per product: " + ", ".join(f"{line['product']} x{line['quantity']}" for line in self.lines)
        else:
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
                values = [str(v) for v in spec["enum"]]
                if str(args.get(key)) == _finish_choice(values, "decline") and _finish_choice(values, "decline") != _finish_choice(values, "order"):
                    return "decline"
                return "order"
        return "order"

    def note_listing(self, record: dict[str, Any]) -> None:
        """Stock seen for a listing (with the step it was seen at) and the properties its seller note sends elsewhere."""
        ident = next((v for k, v in record.items() if isinstance(v, str) and "id" in k.lower() and _ID_HINT.match(v)), None)
        if not ident:
            return
        stock = _in_stock(record)
        if stock is not None:
            self.stock_seen[ident] = (self.step, stock)
        for key, value in record.items():
            if isinstance(value, str) and len(value) > 30 and not _TITLE_KEY.search(str(key)):
                for prop in _redirected_props(value):
                    bucket = self.redirects.setdefault(ident, [])
                    if prop not in bucket:
                        bucket.append(prop)

    def remember(self, name: str, observation: Any) -> None:
        """Keep a compact record of every opened listing (one carrying a facts-like object) and count searches."""
        if isinstance(observation, dict):
            for candidate in [observation] + [v for v in observation.values() if isinstance(v, dict)]:
                self.note_listing(candidate)
                for value in candidate.values():
                    if isinstance(value, list):
                        for row in value[:40]:
                            if isinstance(row, dict):
                                self.note_listing(row)
            inner = observation
            for value in observation.values():
                if isinstance(value, dict) and len(observation) == 2:
                    inner = value
            facts = next((v for v in inner.values() if isinstance(v, dict) and len(v) >= 1 and all(isinstance(x, (str, int, float)) for x in v.values())), None) if isinstance(inner, dict) else None
            ident = next((v for k, v in inner.items() if isinstance(v, str) and "id" in k.lower() and _ID_HINT.match(v)), None) if isinstance(inner, dict) else None
            if facts is not None and ident:
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
        prop, _, value = need.partition(":")
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

    def property_words(self, text: str, line: int | None = None) -> set[str]:
        """A property name's words without 'type' and without the product's own words, each kept when nothing else is
        left."""
        product = self.line_words(line if line is not None and line >= 0 else 0) if self.lines else set()
        full = {_one(w) for w in _plain(text)}
        base = (full - {"type"}) or full
        return (base - product) or base

    def same_property(self, one: str, other: str, line: int | None = None) -> bool:
        """Two property names name the same property when their words match once 'type' and the product's own words are
        set aside ('pack size' is not 'size', 'toe style' is not 'style')."""
        a, b = self.property_words(one, line), self.property_words(other, line)
        return bool(a) and a == b

    def near_stated(self, label: str, values: list[str], line: int | None) -> bool:
        """A label whose name is a narrower or broader form of a stated property's name and whose listings' values share
        a word with that stated value: the request already settles it."""
        mine = self.property_words(label, line)
        seen = {_one(w) for v in values for w in _words(v, False)}
        multi = len(self.lines) > 1
        for prop, value in self.demands(line if multi and line is not None and line >= 0 else None, everything=True):
            theirs = self.property_words(prop, line)
            if not mine or not theirs or mine == theirs or not (mine < theirs or theirs < mine):
                continue
            words = {_one(w) for w in _words(value, False)} - _FILLER - mine - theirs
            if words & seen:
                return True
        return False

    def settled_for(self, line: int | None) -> list[str]:
        """Property names already settled for one product (its requirements, the questions asked about it and the
        properties it was said to be free on); every product's when there is one."""
        multi = len(self.lines) > 1
        scope = line if multi and line is not None and line >= 0 else None
        taken = [prop for prop, _ in self.demands(scope, everything=True)]
        for asked in self.asked:
            label = asked[2:].split(": asked", 1)[0]
            prop, _, owner = label.partition(" of the ")
            if scope is not None and owner and self.line_index(owner) != scope:
                continue
            taken.append(prop if owner else label)
        taken += [f for f in self.free if scope is None or self.line_index(f) == scope]
        return taken

    def hint_items(self) -> list[tuple[str, str, str]]:
        """Story hints from the setup as (property, likely value, product): properties the request points at without a
        value, kept while that product has not settled them."""
        multi = len(self.lines) > 1
        out: list[tuple[str, str, str]] = []
        for item in self.plan.get("hints") or []:
            prop = str(item.get("property") if isinstance(item, dict) else item or "").strip().lower()
            likely = str(item.get("likely") or "").strip().lower() if isinstance(item, dict) else ""
            product = str(item.get("product") or "").strip() if isinstance(item, dict) and multi else ""
            line = self.line_index(product) if product else None
            if not set(_plain(prop)) - {"type"} or any(self.same_property(prop, t, line) for t in self.settled_for(line)):
                continue
            if (prop, product) not in [(p, q) for p, _, q in out]:
                out.append((prop, likely, product if line is not None and line >= 0 else ""))
        return out[:4]

    def hints(self) -> list[tuple[str, str]]:
        """Story hints as (property, likely value). They steer questions and break ties among fits; they never reject a
        listing."""
        return [(p, v) for p, v, _ in self.hint_items()]

    def line_index(self, text: Any, tie: Any = None) -> int:
        """The requested product a phrase (a tag, a shelf label, a title) belongs to: the line whose product words it
        shares most (a second phrase breaks ties), -1 when neither shares any."""
        words = [_one(w) for w in _words(text, False)]
        extra = [_one(w) for w in _words(tie, False)] if tie else []
        best, score = -1, (0, 0)
        for index, line in enumerate(self.lines):
            product = [_one(w) for w in _words(line["product"], False) if w not in _FILLER]
            hit = (sum(1 for w in product if any(_same(w, x) for x in words)), sum(1 for w in product if any(_same(w, x) for x in extra)))
            if hit > score:
                best, score = index, hit
        return best

    def shelf_of(self, item: str) -> str:
        return _shelf(self.rows.get(item) or {}) or _shelf(self.sheets.get(item) or {})

    def item_line(self, item: str) -> int:
        """Which requested product a listing is for (always 0 for a one-product request)."""
        if len(self.lines) <= 1:
            return 0
        record = dict(self.rows.get(item) or {}, **(self.sheets.get(item) or {}))
        title = " ".join(str(v) for k, v in record.items() if isinstance(v, str) and _TITLE_KEY.search(str(k)))
        return self.line_index(self.shelf_of(item), title)

    def line_qty(self, item: str) -> int:
        index = self.item_line(item)
        if 0 <= index < len(self.lines):
            return self.lines[index]["quantity"]
        quantity = self.plan.get("quantity")
        return quantity if isinstance(quantity, int) and not isinstance(quantity, bool) and quantity > 0 else 1

    def line_words(self, line: int) -> set[str]:
        product = self.lines[line]["product"] if 0 <= line < len(self.lines) else self.plan.get("product") or ""
        return {_one(w) for w in _words(product, False)}

    def redirected(self, ident: str, prop: str) -> bool:
        """True when the listing's seller note sends this property to another listing."""
        wanted = set(_plain(prop)) - {"type"}
        for other in self.redirects.get(ident, []):
            have = set(_plain(other)) - {"type"}
            if wanted and have and (wanted <= have or have <= wanted):
                return True
        return False

    def value_keys(self, prop: str, value: str) -> set[str]:
        """Keys that state this value on other opened listings, used when no key is named for the property (a
        requirement written 'type: x' whose value the store files under a differently named key)."""
        found: set[str] = set()
        for sheet in self.sheets.values():
            facts = next((v for v in sheet.values() if isinstance(v, dict)), {})
            if any(_names(str(k), prop, set()) for k in facts):
                return set()                                      # the store does name this property: the normal rule applies
            for key, text in facts.items():
                if isinstance(text, (str, int, float)) and not isinstance(text, bool) and _stated(value, str(text)) and not _TITLE_KEY.search(str(key)):
                    found.add(str(key))
        return found

    def read(self, ident: str, prop: str, value: str, product: set[str]) -> str:
        if value.startswith("~"):
            return "unshown"                                          # relative to another listing or owned by no line: only the model check reads it
        sheet = self.sheets.get(ident) or {}
        verdict = _reading(sheet, prop, value, product)[0]
        if verdict == "yes" and self.redirected(ident, prop):
            return "unshown"
        if verdict in ("yes", "unshown") and _bound(value) is None:
            facts = next((v for v in sheet.values() if isinstance(v, dict)), {})
            if not any(_names(str(k), prop, set()) for k in facts):
                keys = self.value_keys(prop, value)
                carried = [str(facts[k]) for k in facts if str(k) in keys]
                if keys and carried:
                    return "yes" if any(_stated(value, t) for t in carried) else "no"   # the key that carries this value elsewhere decides here
                if keys and verdict == "yes":
                    return "unshown"
        return verdict

    def preferred(self, fits: list[tuple[str, float | None]], line: int = 0) -> str:
        """Among the proven fits, the cheapest whose own facts state a hint's likely value, else the cheapest."""
        if self.plan.get("priority") in ("cheapest", "best"):
            return fits[0][0]                                        # a cheapest or best rule outranks any hint
        product = self.line_words(line)
        for ident, _ in fits:
            if any(v and self.read(ident, p, v, product) == "yes" for p, v in self.hints()):
                return ident
        return fits[0][0]

    def cart_args(self, ident: str, remove: bool = False, keep: list[str] | None = None) -> dict[str, Any]:
        """Arguments that put one listing in the cart at the required quantity (or take it out), read from the cart
        tool's schema: flat id + count or mode, or a list of line objects. A list sets the whole cart, so it also
        carries the lines kept (by default the cart's lines not known to be sold out)."""
        schema = self.schemas.get(self.roles.get("cart") or "") or {}
        properties = schema.get("properties") or {}
        quantity = self.line_qty(ident)
        taking = r"(?i)remov|releas|drop|minus|delet|discard|clear|decreas"
        moded = any(isinstance(spec, dict) and spec.get("enum") for spec in properties.values())
        numeric = lambda kind: kind in ("integer", "number") or (isinstance(kind, list) and ("integer" in kind or "number" in kind))
        args: dict[str, Any] = {}
        placed = False
        for key, spec in properties.items():
            if not isinstance(spec, dict):
                continue
            kind = spec.get("type")
            if "enum" in spec and spec["enum"]:
                values = [str(v) for v in spec["enum"]]
                if remove:
                    args[key] = next((v for v in values if re.search(taking, v)), values[0])
                else:
                    args[key] = next((v for v in values if re.search(r"(?i)add|hold|put|keep|plus|insert|reserve|increase|take", v)), None) or next((v for v in values if not re.search(taking, v)), values[0])
            elif numeric(kind):
                args[key] = 0 if remove and not moded else max(quantity, spec.get("minimum", 1))
            elif (kind == "string" or (isinstance(kind, list) and "string" in kind)) and not placed:
                args[key] = ident
                placed = True
            elif kind == "array" and isinstance(spec.get("items"), dict) and not placed:
                kept = keep if keep is not None else [i for i in self.cart_ids if self.stock_seen.get(i, (0, True))[1] is not False]
                wanted = [i for i in kept if i != ident] + ([] if remove else [ident])
                lines = []
                for listing in wanted:
                    line: dict[str, Any] = {}
                    for inner, part in (spec["items"].get("properties") or {}).items():
                        part = part if isinstance(part, dict) else {}
                        if numeric(part.get("type")):
                            line[inner] = max(self.line_qty(listing), part.get("minimum", 1))
                        elif part.get("type") == "string" and not any(v == listing for v in line.values()):
                            line[inner] = listing
                    if any(v == listing for v in line.values()):
                        lines.append(line)
                if lines:
                    args[key] = lines
                    placed = True
        return args

    def instead_of_decline(self) -> tuple[str, dict[str, Any], str] | None:
        """A decline scores nothing while a proven fit exists. Once the fits were named and the loop still declines,
        or steps are short, the decline becomes the order of a cart holding a proven fit, or the carting of the
        preferred fit."""
        late = self.decline_checked or f"decline|{self.ledger_version}" in self.rule_told or self.steps_left() <= 6
        order, cart = self.roles.get("order"), self.roles.get("cart")
        refused = self.vetoed & self.cart_removed                    # listings the veto carted and the loop took out again

        def strong(found: list[tuple[str, float | None]]) -> list[tuple[str, float | None]]:
            return [f for f in found if f[0] not in refused]

        def weak(found: list[tuple[str, float | None]]) -> list[tuple[str, float | None]]:
            return [] if refused else [f for f in found if f[0] not in self.cart_removed]   # an unproven pick is offered once, never one the loop took out

        if len(self.lines) > 1:
            if not late or not order or self.steps_left() <= 1 or self.time_left() < FORCE_SECONDS:
                return None
            covered = {self.item_line(i) for i in self.cart_ids if self.stock_seen.get(i, (0, True))[1] is not False}
            for line in range(len(self.lines)):
                if line in covered:
                    continue
                fits = strong(self.proven(line)) or weak(self.plausible(line)) or (weak(self.row_fits(line)) if self.steps_left() <= 6 else [])
                if fits and cart and self.steps_left() > 2:
                    best = self.preferred(fits, line)
                    self.vetoed.add(best)
                    return (cart, self.cart_args(best), f"Your decline was not sent: a decline scores 0 while fitting listings exist. {best} is the opened listing that best fits {self.lines[line]['product']} and it was put in the cart instead; make sure every product has its line, re-check stock and price, and order.")
                return None
            gone = [i for i in self.cart_ids if self.stock_seen.get(i, (0, True))[1] is False]
            if self.cart_ids and not gone:
                return (order, self.neutral_args(order, "Order the cart.", "order"), "Your decline was not sent: every requested product has a fitting line in the cart, and a decline scores 0. The order was sent instead.")
            kept = [i for i in self.cart_ids if i not in gone]
            if gone and kept and cart and self.steps_left() > 2:
                args = self.cart_args(kept[-1], keep=kept[:-1]) if self.list_cart() else self.cart_args(gone[0], remove=True)
                if args and not _check(args, self.schemas[cart], "args")[1]:
                    return (cart, args, f"Your decline was not sent: every requested product has an in-stock fitting line in the cart, but {gone[0]} is sold out, so it was taken out. Re-check the cart and order.")
            return None
        fits = strong(self.proven())
        basis = "whose own facts state every current requirement"
        if not fits and late:
            fits = weak(self.plausible())                             # nothing proven: the best listing that contradicts nothing beats a decline
            basis = "whose own facts contradict no current requirement (nothing opened states them all, and a decline scores nothing)"
        if not fits and late and self.steps_left() <= 6:
            fits = weak(self.row_fits())
            basis = "whose result row states every current requirement (it was never opened, but a decline scores nothing)"
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
            self.vetoed.add(best)
            return (cart, self.cart_args(best, keep=[]), f"Your decline was not sent: a decline scores 0 while a fitting listing exists, and {shown} is the opened listing {basis}, within the budget and not shown out of stock. It was put in the cart instead: remove any other line, re-check its stock and price, and order it.")
        return None

    def row_fits(self, line: int = 0) -> list[tuple[str, float | None]]:
        """Listings seen only as result rows (never opened) on the product's shelf, affordable and not shown sold out,
        whose row text states every current requirement: the last resort before a decline."""
        demands = self.demands(line)
        if not demands:
            return []
        top = self.top_shelf(line)
        out = []
        for ident, row in self.rows.items():
            if ident in self.sheets or (len(self.lines) > 1 and self.item_line(ident) != line) or self.stock_seen.get(ident, (0, True))[1] is False:
                continue
            if top and _shelf(row) and _shelf(row) != top or not self.affordable(row, ident):
                continue
            text = " ".join(str(v) for v in row.values() if isinstance(v, str))
            if all(_bound(value) is None and _stated(value, text) for _, value in demands):
                out.append((ident, _price(row)))
        out.sort(key=lambda pair: (pair[1] is None, pair[1] or 0.0))
        return out

    def plausible(self, line: int = 0) -> list[tuple[str, float | None]]:
        """Opened listings none of whose own facts contradicts a current requirement (each read yes or unshown),
        within budget, not shown sold out, on the product's shelf; most requirements proven first, then price. Used
        only when the choice is between one of them and a decline."""
        demands = self.demands(line)
        product = self.line_words(line)
        top = self.top_shelf(line)
        out = []
        for ident, sheet in self.sheets.items():
            if len(self.lines) > 1 and self.item_line(ident) != line:
                continue
            if top and self.shelf_of(ident) and self.shelf_of(ident) != top or not self.affordable(dict(self.rows.get(ident) or {}, **sheet), ident):
                continue
            if self.stock_seen.get(ident, (0, True))[1] is False:
                continue
            readings = [self.read(ident, prop, value, product) for prop, value in demands]
            if "no" in readings:
                continue
            out.append((ident, self.latest_price(ident), readings.count("yes")))
        out.sort(key=lambda row: (-row[2], row[1] is None, row[1] or 0.0))
        return [(i, p) for i, p, _ in out]

    def demands(self, line: int | None = None, everything: bool = False) -> list[tuple[str, str]]:
        """The current requirements as (property, value) pairs the store's rule can read; with a line, only that
        product's (a requirement carries its product as a '[product]' tag when several products are requested)."""
        items = list(self.current) if self.current is not None else [r for r in self.stated + self.revealed if r not in self.dropped]
        multi = len(self.lines) > 1
        out = []
        for item in items:
            text = str(item)
            tag = re.match(r"^\s*\[([^\]]+)\]\s*(.*)$", text)
            owner = 0 if not multi else -1
            unowned = False
            if tag:
                owner = self.line_index(tag.group(1)) if multi else 0
                if owner < 0 and multi:
                    owner = next((i for i in range(len(self.lines)) if self.top_shelf(i) and self.line_index_words(tag.group(1), self.top_shelf(i))), -1)
                text = tag.group(2)
            if multi and owner < 0:
                unowned = True                                         # no product matches the tag: no line may count it as proven
            if line is not None and owner != line and not unowned:
                continue
            prop, sep, value = text.partition(":")
            prop = prop.strip().lower()
            value = re.sub(r"\s*\([^)]*\)", "", value.strip().lower()).strip(" .")
            if re.match(r"(?i)(?:different from|other than|unlike|same as|matching)\b", value) and not everything:
                value = "~" + value                                   # relative to another listing: kept, but the code cannot prove it
            elif unowned and line is not None and not everything:
                value = "~" + value
            if sep and prop and value and not _NOT_A_FACT.search(prop) and not re.search(r"(?i)\bbrand\b", prop):
                if owner >= 0 and not everything and not value.startswith("~") and self.category_says(owner, value, prop):
                    continue                                          # the product's own category label already states it
                out.append((prop, value))
        return out

    def line_index_words(self, phrase: str, label: str) -> bool:
        words = [_one(w) for w in _words(label, False) if w not in _FILLER]
        return any(any(_same(_one(w), x) for x in words) for w in _words(phrase, False) if w not in _FILLER)

    def category_says(self, line: int, value: str, prop: str | None = None) -> bool:
        """True when the product's category label itself spells out the value (a season or audience that is part of
        the category's name) and no opened listing of that product states another value for the property; never for a
        label that joins several kinds, where the kind must be stated."""
        if _bound(value) is not None or re.match(r"(?i)(?:not|no|without)\b", value):
            return False
        shelf = self.top_shelf(line) if self.sheets else ""
        if not shelf or re.search(r"(?i)\band\b|&|,", shelf) or not _stated(value, shelf):
            return False
        if prop:
            for ident, sheet in self.sheets.items():
                if len(self.lines) > 1 and self.item_line(ident) != line:
                    continue
                facts = next((v for v in sheet.values() if isinstance(v, dict)), {})
                for key, text in facts.items():
                    if _names(str(key), prop, set()) and isinstance(text, (str, int, float)) and not _stated(value, str(text)):
                        return False                                  # some listing on the shelf says something else: the shelf does not settle it
        return True

    def affordable(self, record: dict[str, Any], ident: str | None = None) -> bool:
        price, cap = _price(record), self.budget_now
        quantity = self.line_qty(ident) if ident else 1
        if isinstance(cap, (int, float)) and not isinstance(cap, bool) and price is not None and price * quantity > cap + 1e-9:
            return False
        return _in_stock(record) is not False

    def proven(self, line: int = 0) -> list[tuple[str, float | None]]:
        """Opened listings of one requested product that state every current requirement by the store's rule, within
        the budget and not shown as sold out, on the product's shelf; by price."""
        demands = self.demands(line)
        if not demands:
            return []
        product = self.line_words(line)
        top = self.top_shelf(line)
        out = []
        for ident, sheet in self.sheets.items():
            if len(self.lines) > 1 and self.item_line(ident) != line:
                continue
            if self.stock_seen.get(ident, (0, True))[1] is False:
                continue
            if top and self.shelf_of(ident) and self.shelf_of(ident) != top or not self.affordable(dict(self.rows.get(ident) or {}, **sheet), ident):
                continue
            if all(self.read(ident, prop, value, product) == "yes" for prop, value in demands):
                out.append((ident, self.latest_price(ident)))
        out.sort(key=lambda pair: (pair[1] is None, pair[1] or 0.0))
        return out

    def top_shelf(self, line: int = 0) -> str:
        """The shelf a product's opened listings point to: the one named most like the product, then the most opened."""
        product = self.line_words(line)
        shelves: dict[str, int] = {}
        for ident in self.sheets:
            if len(self.lines) > 1 and self.item_line(ident) != line:
                continue
            shelf = self.shelf_of(ident)
            if shelf:
                shelves[shelf] = shelves.get(shelf, 0) + 1
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
        if found[0] is not None:
            return found[0]
        price = _price(self.sheets.get(item) or {})
        return price if price is not None else _price(self.rows.get(item) or {})

    def rule_check(self, item: str, adding: bool) -> str:
        """The facts rule applied to a listing about to be carted or ordered: a fact named for a property that states
        another value wins over the title; a property the seller note sends elsewhere is not proven by the listing."""
        sheet = self.sheets.get(item)
        if sheet is None:
            if adding and self.sheets and self.roles.get("details") and self.steps_left() > 8:
                return f"The details of {item} have not been opened: the store grades a listing by its own facts, which only the details show. Open them first."
            return ""
        line = self.item_line(item)
        if len(self.lines) > 1 and line < 0:
            return ""
        product = self.line_words(line)
        fails = []
        moved = []
        for prop, value in self.demands(line):
            if value.startswith("~"):
                continue
            verdict, shown = _reading(sheet, prop, value, product)
            if verdict == "no" and (len(value.split()) > 4 or ("," in value and _bound(value) is None)):
                verdict = "unshown"                                 # a long free-text requirement is left to the model check
            if verdict == "no":
                fails.append(f"{prop}: {value} (its own fact says {shown})")
            elif self.redirected(item, prop):
                moved.append(prop)
        shelf, top = self.shelf_of(item), self.top_shelf(line)
        if shelf and top and shelf != top:
            words = self.line_words(line)
            overlap = lambda label: len(words & {_one(w) for w in _words(label, False)})
            if overlap(top) > overlap(shelf):
                fails.append(f"category: {item} sits in '{shelf}', but the request's wording as a whole matches the category '{top}'; the store grades the category label, so pick from '{top}'")
        if not fails and not moved:
            return ""
        text = ""
        if fails:
            text = f"Your own reading of its facts (a pre-check, not a store refusal): {item} does not meet " + "; ".join(fails[:4]) + ". A requirement is read from the listing's own fact for that property, not from the title or a look-alike word."
        if moved:
            text += (" " if text else "") + f"The seller note of {item} says its " + ", ".join(moved) + " must be taken from another listing, so its own fact for that does not count; prefer a listing whose own facts state it, or open the referenced listing first."
        return text

    def cheaper_note(self, item: str) -> str:
        """Cheapest requests, before an order: opened fitting listings cheaper than the pick, cheaper unopened rows on
        its shelf, and a capped search not yet made below its price."""
        if self.plan.get("priority") != "cheapest":
            return ""
        price = self.latest_price(item)
        if price is None:
            return ""
        notes = []
        cheaper = [f"{i} ({p:g})" for i, p in self.proven(self.item_line(item)) if i != item and p is not None and p < price]
        if cheaper and self.steps_left() > 8:
            notes.append(f"by the store's own facts these opened listings meet every current requirement and cost less than {item} ({price:g}): " + ", ".join(cheaper[:4]))
        shelf = self.shelf_of(item)
        unread = sorted((_price(row), i) for i, row in self.rows.items() if i not in self.sheets and shelf and _shelf(row) == shelf and _price(row) is not None and _price(row) < price and self.affordable(row, i))
        if unread and self.steps_left() > 9:
            notes.append("cheaper listings on the same shelf were seen in results but never opened: " + ", ".join(f"{i} ({p:g})" for p, i in unread[:5]) + "; open them, cheapest first, before ordering")
        cap = self.cap_arg()
        if cap and shelf and item not in self.sweep_told and self.steps_left() > 10 and not any(shelf.lower() in label and low <= price for label, low in self.capped_below.items()):
            self.sweep_told.add(item)
            notes.append(f"search results are not sorted by price and only some rows are visible, so cheaper fitting listings may never have been shown: search the '{shelf}' shelf with {cap} set just below {price:g} (and the required values as words if that mode allows), then open the cheapest rows first")
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
        fits = [f for line in range(len(self.lines) or 1) for f in self.proven(line)] if len(self.lines) > 1 else self.proven()
        if len(self.lines) > 1 and fits and any(not self.proven(line) for line in range(len(self.lines))):
            fits = []
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
        """The proven fits, shown to the loop whenever the set changes (per product when several are requested)."""
        if len(self.lines) > 1:
            parts = []
            for line in range(len(self.lines)):
                found = self.proven(line)
                parts.append(f"{self.lines[line]['product']}: " + (", ".join(f"{i} ({p:g})" if p is not None else i for i, p in found[:4]) or "none yet"))
            text = "; ".join(parts)
            if text == self.fits_shown:
                return ""
            self.fits_shown = text
            return "\n[fits by the store's facts, per product] " + text + "."
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

    def core_labels(self, line: int | None = None) -> list[str]:
        """Labels carried by at least half of a product's opened listings (two at least), not settled, not asked and
        not logistics; word-valued labels before figure-valued ones."""
        counts: dict[str, int] = {}
        figures: dict[str, int] = {}
        forms: dict[str, dict[str, int]] = {}
        chosen = [sheet for ident, sheet in self.sheets.items() if line is None or len(self.lines) <= 1 or self.item_line(ident) == line]
        for sheet in chosen:
            facts = next((v for v in sheet.values() if isinstance(v, dict)), {})
            for k, v in facts.items():
                if re.search(r"\d", str(v)):
                    base = " ".join(w for w in _plain(str(k)) if w != "type")
                    figures[base] = figures.get(base, 0) + 1
            labels = {" ".join(_plain(str(k))) for k, v in facts.items() if str(v).strip() and not re.fullmatch(r"(?i)n/?a|none|not specified|unknown|null|-", str(v).strip())}
            for base in {" ".join(w for w in label.split() if w != "type") for label in labels}:
                if base and not re.search(r"(?i)option|condition|warrant|model|title|categor|brand|price|stock|seller|sku|assembl|care|instruction|dimension", base):
                    counts[base] = counts.get(base, 0) + 1
            for label in labels:
                base = " ".join(w for w in label.split() if w != "type")
                forms.setdefault(base, {})[label] = forms.get(base, {}).get(label, 0) + 1
        floor = max(2, (len(chosen) + 1) // 2)
        covered: set[str] = set()
        values_by_base: dict[str, list[str]] = {}
        for sheet in chosen:
            facts = next((v for v in sheet.values() if isinstance(v, dict)), {})
            for k, v in facts.items():
                base = " ".join(w for w in _plain(str(k)) if w != "type")
                if isinstance(v, (str, int, float)) and not isinstance(v, bool) and str(v).strip():
                    values_by_base.setdefault(base, []).append(str(v))
        for base, values in values_by_base.items():
            if values and sum(1 for v in values if self.category_says(line if line is not None else 0, v)) * 2 >= len(values):
                covered.add(base)                                     # the category label already answers it (a season named in the aisle)
        taken = self.settled_for(line)
        out = []
        for base in sorted(counts, key=lambda k: (figures.get(k, 0) * 2 > counts[k], -counts[k])):
            words = set(base.split())
            if counts[base] < floor or not words or base in covered:
                continue
            if any(self.same_property(base, t, line) for t in taken) or self.near_stated(base, values_by_base.get(base, []), line):
                continue
            out.append(max(forms[base], key=lambda f: (forms[base][f], "type" in f.split())))
        return out[:8]

    def keep_core(self, planned: list[dict[str, Any]], urgent: bool) -> list[dict[str, Any]]:
        """Each planned question about a core label keeps its wording; one about anything else gives way to the next core
        label (what must be learned first, such as a missing budget, keeps its place)."""
        core = self.core_labels()
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
                label = next((c for c in core if c not in labels and not self.settled_by_request(c, "")), None)
                if label is None:
                    out.append(item)                                  # core labels used up: the planner's own choice stays
                    continue
                labels[index] = label
                item = {"property": label, "question": f"Do you have any preference for the {label}?"}
            out.append(item)
        return out

    def unread_candidates(self) -> list[str]:
        """Listings seen in results on the products' shelves, affordable and never opened: those whose own words carry
        a required value first, then the cheaper."""
        tops = {self.top_shelf(line) for line in range(len(self.lines) or 1)} - {""}
        wanted = [value for _, value in self.demands()]
        pool = []
        for ident, row in self.rows.items():
            if ident in self.sheets or not self.affordable(row, ident) or tops and _shelf(row) and _shelf(row) not in tops:
                continue
            text = " ".join(str(v) for v in row.values() if isinstance(v, str))
            hits = sum(1 for value in wanted if _bound(value) is None and _stated(value, text))
            price = _price(row)
            pool.append((-hits, price if price is not None else float("inf"), ident))
        return [f"{i} ({p:g})" if p != float("inf") else i for _, p, i in sorted(pool)[:5]]

    def cap_arg(self) -> str | None:
        """The search tool's numeric price-cap argument, if it has one."""
        properties = (self.schemas.get(self.roles.get("search") or "") or {}).get("properties") or {}
        for key, spec in properties.items():
            if not isinstance(spec, dict):
                continue
            kind = spec.get("type")
            numeric = kind in ("integer", "number") or (isinstance(kind, list) and ("integer" in kind or "number" in kind))
            text = f"{key} {spec.get('description') or ''}"
            if numeric and re.search(r"(?i)price|cap|max|ceiling|budget|bound|limit|spend|under|upto|up_to|cost", text) and not re.search(r"(?i)page|offset|count|num_?results|size|min", str(key)):
                return key
        return None

    def page_note(self, name: str, args: dict[str, Any], observation: Any) -> str:
        """For a search: how many of its matches are still unseen on further pages."""
        if name != self.roles.get("search"):
            return ""
        cap = self.cap_arg()
        if cap and isinstance(args.get(cap), (int, float)) and not isinstance(args.get(cap), bool):
            label = " ".join(str(v) for k, v in args.items() if isinstance(v, str) and k != cap).lower()
            self.capped_below[label] = min(float(args[cap]), self.capped_below.get(label, float("inf")))
            listed = observation if not (isinstance(observation, dict) and len(observation) == 2) else next((v for v in observation.values() if isinstance(v, dict)), observation)
            found = next((v for v in listed.values() if isinstance(v, list) and v and all(isinstance(x, dict) for x in v)), []) if isinstance(listed, dict) else []
            if found and not any(_price(r) is not None and _price(r) > float(args[cap]) + 1e-9 for r in found):
                self.cap_template = dict(args)
        inner = observation
        if isinstance(observation, dict) and len(observation) == 2:
            inner = next((v for v in observation.values() if isinstance(v, dict)), observation)
        if not isinstance(inner, dict):
            return ""
        paging = re.compile(r"(?i)page|offset|first|start|limit|skip|token|cursor|after|next|further|bookmark|resume|continu|marker")
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
        notes = []
        mode = _compact({k: v for k, v in args.items() if (properties.get(k) or {}).get("enum")})
        if cap and isinstance(args.get(cap), (int, float)) and not isinstance(args.get(cap), bool) and f"cap|{mode}" not in self.ignored_told:
            over = [r for r in rows if isinstance(r, dict) and _price(r) is not None and _price(r) > float(args[cap]) + 1e-9]
            if over:
                self.ignored_told.add(f"cap|{mode}")
                notes.append(f"This search returned rows priced above the {cap} you set, so this mode ignores the price cap: filter by price yourself or use the mode that applies it.")
        if notes:
            return " ".join(notes)
        if self.plan.get("priority") == "cheapest" and cap and isinstance(args.get(cap), (int, float)) and not isinstance(args.get(cap), bool) and total > max(30, 3 * len(rows)):
            return f"This price-capped search matched {total} listings, more than can be seen, and rows are not sorted by price: to reach the cheapest ones, search again with {cap} lowered (about {max(1, int(float(args[cap]) / 2))}) until the matches fit on the visible pages, then open the cheapest rows first."
        if 0 < left and total <= 66 and self.plan.get("priority") == "cheapest":
            return f"This search matched {total} listings; {len(entry['seen'])} seen so far, {left} more on its further pages. Rows are not sorted by price: page through to the end before picking the cheapest, then open the cheapest rows first."
        return ""                                                     # without a cheapest rule, paging advice only spends steps

    def judge(self, prompt: str, max_tokens: int = 700, models: list[str] | None = None) -> dict[str, Any] | None:
        models = models or [JUDGE_MODEL, MAIN_MODEL]
        if self.cost > SPEND_SOFT:
            models = [m for m in models if m != JUDGE_MODEL] or [MAIN_MODEL]
        message = self.infer([{"role": "user", "content": prompt}], None, models, max_tokens)
        return _parse_json_object((message or {}).get("content") or "")

    def interpret(self, lines: list[str], question: str | None, notices: list[str] | None = None) -> str:
        notices = [line for line in notices or [] if line not in self.heard]
        fresh = [line for line in lines if line not in self.heard]
        if not fresh and not notices:
            return ""
        self.heard.update(fresh + notices)
        if self.time_left() < WRAP_UP_SECONDS:
            return ""
        prompt = INTERPRET_PROMPT + "\n\nTHE SHOPPER'S REQUEST (for context):\n" + str(self.policy.get("query") or "")
        prompt += "\n\nREQUIREMENTS SO FAR:\n" + self.ledger()
        if self.said:
            prompt += "\n\nEARLIER EXCHANGES WITH THE SHOPPER (oldest first):\n" + "\n".join(self.said[-12:])
        if question:
            prompt += "\n\nTHE ASSISTANT ASKED:\n" + question
        prompt += "\n\nTHE SHOPPER NOW SAID:\n" + ("\n".join(fresh) or "(nothing new)")
        if notices:
            prompt += "\n\nCHANGE NOTICES THAT CAME WITH IT (these can drop or change requirements):\n" + "\n".join(notices)
        self.said.append((f"asked: {question} -> " if question else "") + "shopper: " + " / ".join(fresh + notices))
        result = self.judge(prompt, 1500, [READ_MODEL, JUDGE_MODEL])
        if not result:
            return ""
        notes = []
        self.last_flags = {"too_many": bool(result.get("too_many")), "puzzled": bool(result.get("puzzled")), "needs": [n for n in result.get("needs") or [] if isinstance(n, str) and n]}
        for look in result.get("lookups") or []:
            if isinstance(look, str) and look.strip() and look not in self.lookups:
                self.lookups.append(look)
                notes.append(f"to apply a requirement that depends on another listing, find it: {look.strip()} (search the seller's name with the product words, open its details and its seller note, read the value, then require a different value)")
        if question:
            result["dropped"] = [d for d in result.get("dropped") or [] if isinstance(d, str)] if notices else []   # a reply to our own question never drops a requirement; a notice sent with it can
        now = result.get("requirements_now")
        if isinstance(now, list) and all(isinstance(r, str) for r in now):
            current = [r for r in now if r.strip()]
            dropped_now = [str(d).lower() for d in (result.get("dropped") or []) if isinstance(d, str)] + [d.lower() for d in self.dropped]
            for stated in self.stated + [n for n in self.revealed if n not in self.stated]:
                core = stated.split("]", 1)[-1].split(":", 1)[-1].strip().lower()
                kept = any(core and core in r.lower() for r in current)
                gone = any(core and (core in d or d in stated.lower()) for d in dropped_now if d)
                if not kept and not gone:
                    current.append(stated)
            self.current = current
        for need in result.get("needs") or []:
            if isinstance(need, str) and need and need not in self.revealed:
                self.revealed.append(need)
                if self.current is not None and need not in self.current:
                    self.current.append(need)
                notes.append(f"new requirement: {need}")
        for prop in result.get("no_preference") or []:
            if isinstance(prop, str) and prop and prop not in self.free:
                self.free.append(prop)
        for drop in result.get("dropped") or []:
            if isinstance(drop, str) and drop and drop not in self.dropped:
                self.dropped.append(drop)
                notes.append(f"dropped: {drop}")
        budget = result.get("new_budget")
        if isinstance(budget, (int, float)) and not isinstance(budget, bool) and budget > 0 and budget != self.budget_now:
            self.budget_now = budget
            notes.append(f"new budget: {budget}")
        if result.get("unavailable"):
            notes.append("an item became unavailable; remove it from the cart and do not order it")
            if len(self.cart_ids) == 1:
                self.stock_seen[self.cart_ids[0]] = (self.step, False)
        for need in result.get("needs") or []:
            if isinstance(need, str) and ":" in need:
                stating, other = self.sheets_stating(need)
                if stating:
                    notes.append(f"opened listings that state {need.strip()}: " + ", ".join(stating[:6]))
                elif self.sheets:
                    notes.append(f"no opened listing states {need.strip()} yet: search for it")
        moved = bool(result.get("dropped")) or (isinstance(budget, (int, float)) and not isinstance(budget, bool) and budget > 0)
        if moved:
            self.ledger_version += 1
            refit = [f"{i} ({p:g})" if p is not None else i for line in range(max(1, len(self.lines))) for i, p in self.proven(line)[:3]]
            if self.plan.get("priority") == "cheapest":
                notes.append("the set of fitting items has changed: listings set aside only for the dropped requirement or the old budget may fit now; search again and re-pick the cheapest that fits before ordering" + (f" (opened listings that fit now, by price: {', '.join(refit)})" if refit else ""))
            elif refit:
                notes.append("opened listings that fit the changed requirements: " + ", ".join(refit))
        if notes:
            self.ledger_version += 1
        reply = str(result.get("reply") or "").strip()
        if reply and self.roles.get("message") and any("?" in line for line in fresh) and not result.get("puzzled") and not result.get("too_many") and not result.get("no_preference") and not result.get("needs"):
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
                    text = _compact(_brief(value, 400))[:1500]
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
        return found[-3:]

    def verify_cart(self) -> str:
        for item in list(self.cart_ids)[-4:]:
            verdict = self.verify_item(item, "ORDER")
            if verdict:
                return verdict
        return ""

    def code_proven(self, item: str) -> bool:
        """The listing's own facts state every current requirement of its line (no cross-listing requirement, no seller
        note sending a property elsewhere, not shown sold out, on the line's shelf): the model check adds nothing."""
        line = self.item_line(item)
        if len(self.lines) > 1 and line < 0:
            return False
        items = list(self.current) if self.current is not None else [r for r in self.stated + self.revealed if r not in self.dropped]
        if any(re.search(r"(?i)different from|other than|same as|matching", str(r)) for r in items):
            return False
        if self.redirects.get(item) or self.stock_seen.get(item, (0, True))[1] is False or item not in self.sheets:
            return False
        top, shelf = self.top_shelf(line), self.shelf_of(item)
        if top and shelf and shelf != top:
            return False
        demands = self.demands(line)
        if not demands or any(value.startswith("~") or _bound(value) is not None or re.search(r"\d", value) or re.match(r"(?i)(?:not|no|without|any|anything|except|excluding|avoid)\b", value) for _, value in demands):
            return False                                              # limits, numbers and exclusions are left to the model check
        raw = [str(r).split("]", 1)[-1] for r in items]
        if len([r for r in raw if ":" in r]) > len(self.demands(None, everything=True)) or any(":" not in r for r in raw if r.strip()):
            return False                                              # some requirement (a brand, a kind, an unparsed one) is not read by the code
        price, cap = self.latest_price(item), self.budget_now
        if isinstance(cap, (int, float)) and not isinstance(cap, bool) and (price is None or price * self.line_qty(item) > cap + 1e-9):
            return False
        product = self.line_words(line)
        if not all(self.read(item, prop, value, product) == "yes" for prop, value in demands):
            return False
        facts = next((v for v in (self.sheets.get(item) or {}).values() if isinstance(v, dict)), {})
        for prop, value in demands:
            named = [str(text) for key, text in facts.items() if _names(str(key), prop, set()) and isinstance(text, (str, int, float))]
            want = [_one(w) for w in _words(value, False) if w not in _FILLER]
            if not named or not want or not any(set(want) <= {_one(w) for w in _words(t, False)} for t in named):
                return False                                          # exact words under a key named for the property, nothing fuzzy
        return True

    def verify_item(self, item: str, purpose: str) -> str:
        if self.time_left() < WRAP_UP_SECONDS + 20:
            return ""
        if self.code_proven(item) or self.cost > SPEND_SOFT:
            return ""                                                 # proven by the facts, or past the soft spend mark (a weaker checker over-holds)
        key = f"{item}|{self.ledger_version}"
        if key in self.verdicts:
            return self.verdicts[key]
        records = self.item_records(item)
        if not records:
            return ""
        prompt = VERIFY_PROMPT + "\n\nTHE SHOPPER'S REQUEST:\n" + str(self.policy.get("query") or "")
        prompt += "\n\nCURRENT REQUIREMENTS (stated, revealed and changed):\n" + self.ledger()
        line = self.item_line(item)
        if len(self.lines) > 1:
            target = self.lines[line]["product"] if line >= 0 else "the product it best matches"
            prompt += f"\n\nTHIS LISTING IS FOR THE PRODUCT LINE: {target} (judge it only against that product's requirements)"
        prompt += f"\n\nTHE ITEM TO {purpose}: {item} (quantity {self.line_qty(item)})\nRECORDS OF THIS ITEM SEEN IN THE STORE (oldest first):\n" + "\n".join(records)
        if self.last_cart_view is not None:
            prompt += "\n\nLATEST CART VIEW:\n" + _compact(self.last_cart_view)[:3000]
        labels = self.row_labels()
        if labels:
            prompt += "\n\nLABELS THAT RECUR ON THIS STORE'S LISTINGS (with counts; the item's category label must be the one that matches the request's wording as a whole, not a neighbouring one):\n" + labels
        result = self.judge(prompt, 1200, [JUDGE_MODEL, MAIN_MODEL])
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
                        if _TITLE_KEY.search(str(key)) or _SHELF_KEY.search(str(key)) or re.search(r"(?i)brand|seller|outlet|store|shop|company|maker|headline|summary|description|^id$|_id$", str(key)):
                            continue
                        name = re.sub(r"[_.]+", " ", str(key)).strip().lower()
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

    def price_ladder(self, item: str) -> str:
        """Cheapest requests, before ordering a pick: repeat the last price-capped search with the cap just under the
        pick's price and show any cheaper rows (one step)."""
        search, cap, template = self.roles.get("search"), self.cap_arg(), self.cap_template
        if self.plan.get("priority") != "cheapest" or not search or not cap or not template or self.steps_left() < 10 or self.time_left() < WRAP_UP_SECONDS + 30:
            return ""
        price = self.latest_price(item)
        shelf = self.shelf_of(item)
        key = (shelf or item).lower()
        if price is None or key in self.laddered:
            return ""
        self.laddered.add(key)
        spec = ((self.schemas.get(search) or {}).get("properties") or {}).get(cap) or {}
        integer = spec.get("type") == "integer" or (isinstance(spec.get("type"), list) and "integer" in spec["type"])
        below = (int(price) - 1) if integer else round(price - 0.01, 2)
        if below <= 0:
            return ""
        paging = re.compile(r"(?i)page|offset|first|start|skip|token|cursor|after|next|further|bookmark|resume|continu|marker")
        args = {k: v for k, v in template.items() if not paging.search(str(k))}
        args[cap] = below
        args, problems = _check(args, self.schemas[search], "args")
        if problems:
            return ""
        before = self.step
        reply = self.env_call([{"name": search, "args": args}])
        self.trace.append({"kind": "price_ladder", "step": before + 1, "args": args, "reply": reply})
        if reply is None:
            return ""
        observation = reply.get("observation")
        self.records.append(observation)
        self.remember(search, observation)
        self.page_note(search, args, observation)                      # the capped search is on record: the sweep advice is not repeated
        self.sweep_told.add(item)
        heard: list[str] = []
        if isinstance(observation, dict):
            for value in observation.values():
                if isinstance(value, list) and value and all(isinstance(v, str) for v in value):
                    _sentences(value, heard)
        meaning = self.interpret(heard, None) if heard else ""
        said = (" The shopper also said something: " + meaning) if meaning else ""
        listed = observation if not (isinstance(observation, dict) and len(observation) == 2) else next((v for v in observation.values() if isinstance(v, dict)), observation)
        rows = next((v for v in listed.values() if isinstance(v, list) and v and all(isinstance(x, dict) for x in v)), []) if isinstance(listed, dict) else []
        cheaper = []
        open_fits = {i for i, _ in self.plausible(self.item_line(item))}
        for row in rows:
            ident = next((v for k, v in row.items() if isinstance(v, str) and "id" in k.lower() and _ID_HINT.match(v)), None)
            if not ident or ident == item or (ident in self.sheets and ident not in open_fits) or (shelf and _shelf(row) and _shelf(row) != shelf):
                continue                                                # an opened listing whose facts contradict a requirement is out
            if _price(row) is not None and _price(row) < price and _in_stock(row) is not False:
                title = next((str(v) for k, v in row.items() if isinstance(v, str) and _TITLE_KEY.search(str(k))), "")
                cheaper.append((_price(row), ident, title[:70]))
        if not cheaper:
            return said.strip()
        cheaper.sort()
        total = next((v for k, v in listed.items() if isinstance(v, int) and not isinstance(v, bool) and not paging.search(str(k))), len(rows)) if isinstance(listed, dict) else len(rows)
        shown = "; ".join(f"{i} ({p:g}) {t}" for p, i, t in cheaper[:10])
        return f"Price check before ordering {item} ({price:g}), done for you with one step: the same capped search under {below:g} lists {total} listing(s), cheaper rows that could fit: {shown}. Open the ones that could fit (cheapest first) and order the cheapest that fully fits; order {item} only if each of them fails." + said

    def cart_moves(self, name: str, args: dict[str, Any]) -> list[tuple[str, bool]]:
        """(listing id, adding?) for each line a cart call touches: flat id + count or add/remove mode, or a list of
        line objects (possibly a JSON string)."""
        properties = (self.schemas.get(name) or {}).get("properties") or {}
        moves: list[tuple[str, bool]] = []
        for value in args.values():
            lines = value
            if isinstance(lines, str) and lines.strip().startswith("["):
                try:
                    lines = json.loads(lines)
                except ValueError:
                    continue
            if isinstance(lines, list):
                for line in lines:
                    if isinstance(line, dict):
                        ident = next((v for v in line.values() if isinstance(v, str) and _ID_HINT.match(v) and " " not in v), None)
                        count = next((v for v in line.values() if isinstance(v, (int, float)) and not isinstance(v, bool)), 1)
                        if ident:
                            moves.append((ident, bool(count and count > 0)))
        if moves:
            return moves
        target = next((v for k, v in args.items() if isinstance(v, str) and "enum" not in (properties.get(k) or {}) and v.strip() and " " not in v.strip()), None)
        if not target:
            return []
        count = next((v for v in args.values() if isinstance(v, (int, float)) and not isinstance(v, bool)), None)
        mode = _compact({k: v for k, v in args.items() if isinstance(properties.get(k), dict) and "enum" in properties[k]})
        removing = count == 0 or bool(re.search(r"(?i)remov|releas|drop|minus|delet|discard|clear|decreas", mode))
        return [(target, not removing)]

    def list_cart(self) -> bool:
        """The cart tool takes a list of line objects: such a call sets the whole cart, not one line."""
        properties = (self.schemas.get(self.roles.get("cart") or "") or {}).get("properties") or {}
        return any(isinstance(spec, dict) and spec.get("type") == "array" for spec in properties.values())

    def track_nested_cart(self, args: dict[str, Any]) -> None:
        """Cart tools that take a list of lines ({listing id, count} objects, possibly sent as a JSON string): the
        list is the whole cart, so the tracked cart becomes its lines with a positive count."""
        for value in args.values():
            if isinstance(value, str) and value.strip().startswith("["):
                try:
                    value = json.loads(value)
                except ValueError:
                    continue
            if not isinstance(value, list):
                continue
            kept = []
            for line in value:
                if not isinstance(line, dict):
                    continue
                ident = next((v for v in line.values() if isinstance(v, str) and _ID_HINT.match(v) and " " not in v), None)
                count = next((v for v in line.values() if isinstance(v, (int, float)) and not isinstance(v, bool)), 1)
                if ident and count and count > 0 and ident not in kept:
                    kept.append(ident)
                    self.cart_counts[ident] = int(count)
            for ident in kept:
                if ident not in self.cart_ids:
                    self.added_at[ident] = self.step
            self.cart_removed.update(i for i in self.cart_ids if i not in kept and i in self.vetoed)   # a veto pick left out of the list was taken out
            self.cart_ids = kept
            self.cart_changed_at = self.step

    def sync_cart_view(self, observation: Any) -> None:
        """Take the cart's lines from a cart view (ids of rows with a positive count), when the view lists rows."""
        inner = observation
        if isinstance(observation, dict) and len(observation) == 2:
            inner = next((v for v in observation.values() if isinstance(v, dict)), observation)
        if not isinstance(inner, dict):
            return
        rows = next((v for v in inner.values() if isinstance(v, list) and all(isinstance(x, dict) for x in v)), None)
        if rows is None:
            return
        found = []
        for row in rows:
            ident = next((v for k, v in row.items() if isinstance(v, str) and "id" in k.lower() and _ID_HINT.match(v)), None)
            count = next((v for k, v in row.items() if isinstance(v, (int, float)) and not isinstance(v, bool) and re.search(r"(?i)qty|quantit|count|units|amount_total|number", str(k))), 1)
            if ident and count:
                found.append(ident)
        if found or rows == []:
            for ident in found:
                if ident not in self.cart_ids:
                    self.added_at.setdefault(ident, self.step)
            self.cart_ids = found

    def view_args(self, item: str) -> dict[str, Any] | None:
        name = self.roles.get("view")
        schema = self.schemas.get(name or "") or {}
        properties = schema.get("properties") or {}
        required = list(schema.get("required") or [])
        strings = [k for k, v in properties.items() if isinstance(v, dict) and (v.get("type") == "string" or "string" in (v.get("type") or [])) and "enum" not in v]
        target = next((k for k in required if k in strings), None) or (strings[0] if len(strings) == 1 else None)
        if not target or any(k != target for k in required):
            return None
        args, problems = _check({target: item}, schema, "args")
        return None if problems else args

    def stock_guard(self) -> str:
        """Before an order: every cart line's stock must have been seen after the last cart change (a listing can sell
        out silently when it goes into the cart). Unseen lines are re-read here with the tool that shows stock."""
        view = self.roles.get("view")
        if not view or not self.cart_ids or self.finished or self.dead or self.time_left() < WRAP_UP_SECONDS / 2:
            return ""
        strict = [i for i in self.cart_ids if self.stock_seen.get(i, (-1, None))[0] <= self.cart_changed_at]
        loose = [i for i in self.cart_ids if self.stock_seen.get(i, (-1, None))[0] <= self.added_at.get(i, -1)]
        stale = [i for i in (strict if self.steps_left() > len(strict) + 6 else loose) if i not in self.stock_unshown]
        stale = stale[:4]
        if not stale or self.steps_left() <= len(stale) + 3:
            return ""
        actions = []
        for item in stale:
            args = self.view_args(item)
            if args is None:
                return ""
            actions.append({"name": view, "args": args})
        self.stock_rounds += 1
        before = self.step
        reply = self.env_call(actions)
        self.trace.append({"kind": "stock_guard", "step": before + 1, "actions": actions, "reply": reply})
        if reply is None:
            return ""
        observations = reply.get("observations") if isinstance(reply.get("observations"), list) else [reply.get("observation")]
        heard: list[str] = []
        shown = []
        for item, observation in zip(stale, observations):
            self.records.append(observation)
            self.remember(view, observation)
            if isinstance(observation, dict):
                for value in observation.values():
                    if isinstance(value, list) and value and all(isinstance(v, str) for v in value):
                        _sentences(value, heard)
            seen = self.stock_seen.get(item)
            state = "in stock" if seen and seen[0] > before and seen[1] else ("OUT OF STOCK" if seen and seen[1] is False else "stock not shown")
            if state == "stock not shown":
                self.stock_unshown.add(item)                          # this view does not show its stock: reported once, not re-read again
            price = self.latest_price(item)
            shown.append(f"{item}: {state}" + (f", price {price:g}" if price is not None else "") + (f" (its record: {_compact(_trim(observation))[:400]})" if state == "stock not shown" else ""))
        meaning = self.interpret(heard, None) if heard else ""
        gone = [i for i in stale if self.stock_seen.get(i, (0, True))[1] is False]
        cap = self.budget_now
        prices = [(self.latest_price(i), self.line_qty(i)) for i in self.cart_ids]
        total = sum(p * q for p, q in prices if p is not None)
        over = isinstance(cap, (int, float)) and not isinstance(cap, bool) and all(p is not None for p, _ in prices) and total > cap + 1e-9
        if over:
            meaning = (meaning + " " if meaning else "") + f"The cart now totals {total:g}, over the budget of {cap:g}: replace a line with a cheaper verified fit."
        if not gone and not meaning and not any("stock not shown" in line for line in shown):
            self.guard_ok = "Stock and price were just re-read for you after your last cart change: " + "; ".join(shown) + ". "
            return ""
        text = "Stock re-check after your last cart change (done for you, it used steps): " + "; ".join(shown) + "."
        if meaning:
            text += " The shopper also said something: " + meaning
        if gone:
            text += " " + ", ".join(gone) + " sold out: " + ("send the cart again without it and with the next verified fit for that product (a cart call lists the whole cart: include every line you keep)" if self.list_cart() else "take it out of the cart (or set its count to 0) and put in the next verified fit for that product") + ", then order again."
        else:
            text += " Apply that change before ordering."
        return text

    def sheets_per_line(self) -> list[int]:
        lines = max(1, len(self.lines))
        per_line = [0] * lines
        for ident in self.sheets:
            line = self.item_line(ident)
            if 0 <= line < lines:
                per_line[line] += 1
        return per_line

    def ready_to_ask(self) -> bool:
        lines = max(1, len(self.lines))
        per_line = self.sheets_per_line()
        if min(per_line) >= 2 and len(self.sheets) >= min(4, 2 * lines + 1):
            return True
        if lines > 1 and min(per_line) == 0 and self.steps_left() > 16:
            return False                                              # a product with nothing opened would get no question yet
        return self.step >= min(12, self.max_steps // 3)

    def line_menu(self, line: int) -> tuple[str, list[str]]:
        """The property names the opened listings of one product carry (count and example values), and its core
        labels."""
        counts: dict[str, int] = {}
        samples: dict[str, list[str]] = {}
        for ident, sheet in self.sheets.items():
            if len(self.lines) > 1 and self.item_line(ident) != line:
                continue
            facts = next((v for v in sheet.values() if isinstance(v, dict)), {})
            for key, value in facts.items():
                if not isinstance(value, (str, int, float)) or isinstance(value, bool) or not str(value).strip():
                    continue
                name = re.sub(r"[_.]+", " ", str(key)).strip().lower()
                counts[name] = counts.get(name, 0) + 1
                bucket = samples.setdefault(name, [])
                if str(value).lower()[:40] not in bucket and len(bucket) < 4:
                    bucket.append(str(value).lower()[:40])
        ranked = sorted(counts, key=lambda k: -counts[k])[:22]
        menu = "\n".join(f"- {k} (in {counts[k]} listings): " + " | ".join(samples.get(k, [])) for k in ranked)
        return menu, self.core_labels(line)

    def settled_by_request(self, prop: str, product: str) -> bool:
        """True when the request already settles this property: a stated requirement of the product names the same
        property, or the opened listings show a stated value under this very key, or (when no listing carries this very
        key) under a narrower or broader one, for example a requirement written as a type whose value sits under a style
        key."""
        line = self.line_index(product) if product and len(self.lines) > 1 else None
        props, values = [], []
        for item in self.stated:
            tag = re.match(r"^\s*\[([^\]]+)\]\s*(.*)$", item)
            if tag and product and self.line_index(tag.group(1)) != self.line_index(product):
                continue
            name, _, value = (tag.group(2) if tag else item).partition(":")
            props.append(name.strip().lower())
            value = value.strip().lower()
            if value and _bound(value) is None:
                values.extend(a.strip() for a in re.split(r"\s+or\s+|(?<=[a-z])\s*/\s*(?=[a-z])|,\s*(?!\d)", value) if len(a.strip()) >= 3)
        if any(p and self.same_property(prop, p, line) for p in props):
            return True
        if not values:
            return False
        wanted = set(_plain(prop)) - {"type"}
        exact = carried = near = False
        for sheet in self.sheets.values():
            facts = next((v for v in sheet.values() if isinstance(v, dict)), {})
            for key, value in facts.items():
                have = set(_plain(str(key))) - {"type"}
                stated = any(_stated(v, str(value)) for v in values)
                if wanted and have == wanted:
                    exact, carried = exact or stated, True
                elif wanted and have and (wanted <= have or have <= wanted) and stated:
                    near = True
        return exact or (near and not carried)

    def plan_questions(self, count: int, urgent: bool) -> list[dict[str, str]]:
        multi = len(self.lines) > 1
        prompt = PLAN_PROMPT.replace("QUESTION_COUNT", str(count)) + "\n\nTHE SHOPPER'S REQUEST:\n" + str(self.policy.get("query") or "")
        prompt += "\n\nREQUIREMENTS SO FAR:\n" + self.ledger()
        if self.hint_items():
            prompt += "\n\nHINTS FROM THE STORY (properties the request points at without stating a value; ask about these first, each in the listings' own words for it): " + "; ".join((f"{p} of the {q}" if q else p) + (f" (likely {v})" if v else "") for p, v, q in self.hint_items())
        if self.asked:
            prompt += "\n\nALREADY ASKED (with what the shopper said):\n" + "\n".join(self.asked)
        if self.plan.get("ask_first"):
            prompt += "\n\nMUST BE LEARNED FROM THE SHOPPER: " + "; ".join(str(h) for h in self.plan.get("ask_first") or [])
        if multi:
            for line in range(len(self.lines)):
                menu, core = self.line_menu(line)
                prompt += f"\n\nPRODUCT {line + 1}: {self.lines[line]['product']}\nPROPERTIES IN ITS OPENED LISTINGS (name, number of listings, example values):\n" + (menu or "(none opened yet)")
                if core:
                    prompt += "\nCORE PROPERTIES OF THIS PRODUCT: " + ", ".join(core)
        else:
            menu, _ = self.property_menu()
            prompt += "\n\nPROPERTIES IN THE LISTINGS' DETAILS SEEN SO FAR (name, number of listings, example values):\n" + (menu or "(none seen yet)")
            core = self.core_labels()
            if core:
                prompt += "\n\nCORE PROPERTIES (carried by most of the listings opened, not settled or asked yet; ask only about these): " + ", ".join(core)
        result = self.judge(prompt, 2000, [READ_MODEL, JUDGE_MODEL])
        done = {a.split(": asked", 1)[0][2:].strip().lower() for a in self.asked}
        planned: list[dict[str, str]] = []
        for index, item in enumerate((result or {}).get("questions") or []):
            if not isinstance(item, dict):
                continue
            prop = re.sub(r"[_.]+", " ", re.sub(r"\s+", " ", str(item.get("property") or ""))).strip(" ?.").lower()
            if not prop or len(prop) > 50:
                continue
            product = ""
            if multi:
                line = self.line_index(str(item.get("product") or "") + " " + str(item.get("question") or ""))
                product = self.lines[line]["product"] if line >= 0 else ""
            label = f"{prop} of the {product}" if product else prop
            if label in done or any(label == q["label"] for q in planned):
                continue
            if not (urgent and index == 0) and self.settled_by_request(prop, product):
                continue
            if urgent and index == 0 and str(item.get("question") or "").strip():
                question = str(item.get("question")).strip()
            elif product:
                question = f"For the {product}, do you have a preference for the {prop}?"
            else:
                question = f"Do you have a preference for the {prop}?"
            planned.append({"product": product, "property": prop, "question": question, "label": label})
        if not multi and planned:
            kept = self.keep_core([{"property": q["property"], "question": q["question"]} for q in planned], urgent)
            planned = [{"product": "", "property": str(k.get("property") or ""), "question": str(k.get("question") or "").replace("Do you have any preference", "Do you have a preference"), "label": str(k.get("property") or "")} for k in kept if str(k.get("question") or "").strip()]
        return self.shape_queue(planned, count, urgent)

    def shape_queue(self, planned: list[dict[str, str]], count: int, urgent: bool) -> list[dict[str, str]]:
        """Drop near-duplicate properties of one product (and ones already asked); products take turns."""
        def core_words(text: str) -> set[str]:
            return set(_plain(text)) - {"type"}

        def near(one: str, other: str) -> bool:
            a, b = core_words(one), core_words(other)
            return bool(a and b and (a <= b or b <= a))

        asked = [(a.split(": asked", 1)[0][2:].strip().lower()) for a in self.asked]
        head: list[dict[str, str]] = []
        groups: dict[str, list[dict[str, str]]] = {}
        order: list[str] = []
        for index, item in enumerate(planned):
            if urgent and index == 0:
                head.append(item)
                continue
            product = item.get("product", "")
            mine = groups.setdefault(product, [])
            if product not in order:
                order.append(product)
            if any(near(q["property"], item["property"]) for q in mine):
                continue
            if any(near(item["label"], done) for done in asked):
                continue
            mine.append(item)
        mixed: list[dict[str, str]] = []
        while any(groups[p] for p in order):
            for p in order:
                if groups[p]:
                    mixed.append(groups[p].pop(0))
        return (head + mixed)[: count + 1]

    def ask_planned(self, force: bool = False) -> bool:
        """Questions go one per step: plan a short ranked list, send the next one, read the reply, and re-ask once in
        the plainest words when the shopper says the message held too much."""
        tool = self.roles.get("message")
        limit = self.message_budget if self.message_budget is not None else 3
        if not tool or self.messages_used >= limit:
            self.planner_on = False
            return False
        if self.steps_left() <= 10 or self.time_left() < WRAP_UP_SECONDS + 40:
            self.planner_on = False
            return False
        urgent = bool(self.plan.get("ask_first")) and not self.asked
        if not force and not urgent and not self.ready_to_ask():
            return False
        if self.queue and self.plan_gap and all(self.sheets_per_line()[line] > 0 for line in self.plan_gap):
            self.queue, self.plan_gap = [q for q in self.queue if q["label"] in self.reasked], []   # a product that had nothing opened now has listings: plan again (a queued re-ask goes first)
        if not self.queue:
            if self.planned_once >= 3:
                self.planner_on = False
                return False
            left = limit - self.messages_used
            gap = [line for line, count in enumerate(self.sheets_per_line()) if count == 0] if len(self.lines) > 1 and not urgent else []
            if gap and self.plan_gap and self.steps_left() > 14 and not force:
                return False                                          # the kept messages wait for that product's listings
            reserve = min(len(gap), left - 1) if gap and self.steps_left() > 14 else 0
            self.queue = self.plan_questions(left - reserve, urgent)
            if urgent:
                self.queue = self.queue[:1]                           # planned before any listing was read: only the head
            elif reserve:
                self.queue = self.queue[: left - reserve]
            self.plan_gap = gap if reserve else []
            self.planned_once += 1
            if not self.queue:
                if self.plan_gap:
                    return False
                self.planner_on = False
                return False
        item = self.queue.pop(0)
        args = self.message_args(item["question"])
        if not args:
            self.planner_on = False
            return False
        args, problems = _check(args, self.schemas[tool], "args")
        if problems:
            self.planner_on = False
            return False
        call = {"id": f"ask_{self.messages_used + 1}_{self.step}", "type": "function", "function": {"name": tool, "arguments": json.dumps(args)}}
        self.messages.append({"role": "assistant", "content": f"I will ask the shopper about the {item['label']}.", "tool_calls": [call]})
        self.last_flags = {}
        replies = self.run_calls([call], planned=True)
        for call_id, text in replies:
            self.messages.append({"role": "tool", "tool_call_id": call_id, "content": text + "\n" + self.status()})
            self.asked.append(f"- {item['label']}: asked \"{item['question']}\" -> {text.split(chr(10) + '[status]')[0][:300]}")
        if self.last_flags.get("too_many") and item["label"] not in self.reasked and limit - self.messages_used >= 2:
            self.reasked.add(item["label"])
            plain = f"What {item['property']} would you like" + (f" for the {item['product']}" if item.get("product") else "") + "?"
            self.queue.insert(0, {"product": item.get("product", ""), "property": item["property"], "question": plain, "label": item["label"]})
        if self.messages_used >= limit:
            self.planner_on = False
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
        properties = self.schemas[name].get("properties") or {}
        target = next((v for k, v in args.items() if isinstance(v, str) and "enum" not in (properties.get(k) or {}) and v.strip() and " " not in v.strip()), None)
        wanted = self.line_qty(target) if target else self.plan.get("quantity")
        if not isinstance(wanted, int) or isinstance(wanted, bool) or wanted < 1:
            return ""
        signature = _compact({k: v for k, v in args.items() if isinstance(properties.get(k), dict) and "enum" in properties[k]})
        if self.add_signature is not None and signature != self.add_signature:
            return ""
        for key, spec in properties.items():
            if isinstance(spec, dict) and spec.get("type") == "integer" and "enum" not in spec:
                value = args.get(key, spec.get("default", 1))
                if isinstance(value, int) and value not in (wanted, 0):
                    return f"the request needs a quantity of {wanted} of this product, but this call asks for {value}."
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
            if "enum" in spec and spec["enum"]:
                args[key] = _finish_choice([str(v) for v in spec["enum"]], kind)  # an enum takes one of its own values, never free text
            elif spec.get("type") == "string" or "string" in (spec.get("type") or []):
                args[key] = text[: spec.get("maxLength", len(text))]
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

    def force_repair(self) -> None:
        """Before a forced order (no model calls): a line known to be sold out leaves the cart when its product keeps a
        line (a kept one or its best remaining fit), and every product without a line gets its best remaining fit."""
        cart = self.roles.get("cart")
        if not cart or self.steps_left() < 2 or self.time_left() < 8:
            return
        multi = len(self.lines) > 1
        product_of = (lambda i: self.item_line(i)) if multi else (lambda i: 0)
        gone = [i for i in self.cart_ids if self.stock_seen.get(i, (0, True))[1] is False]
        kept = [i for i in self.cart_ids if i not in gone]
        covered = {product_of(i) for i in kept}
        added: dict[int, str] = {}
        for line in range(max(1, len(self.lines))):
            found = [] if line in covered else [f for f in (self.proven(line) or self.plausible(line)) if f[0] not in gone]
            if found:
                added[line] = self.preferred(found, line)
        dropped = [i for i in gone if product_of(i) in covered or product_of(i) in added]
        if not added and not dropped:
            return
        if self.list_cart():
            basket = kept + [i for i in gone if i not in dropped] + list(added.values())
            args = self.cart_args(basket[-1], keep=basket[:-1]) if basket else {}
            if args and not _check(args, self.schemas[cart], "args")[1]:
                reply = self.env_call([{"name": cart, "args": args}])
                self.trace.append({"kind": "forced_cart", "actions": [{"name": cart, "args": args}], "reply": reply})
                if reply is not None and not _faulty(reply.get("observation")):
                    self.cart_used = True
                    self.cart_ids = basket
            return
        actions, idents = [], []
        for ident in dropped:
            args = self.cart_args(ident, remove=True)
            if args and not _check(args, self.schemas[cart], "args")[1]:
                actions.append({"name": cart, "args": args})
                idents.append((ident, False))
        for best in added.values():
            args = self.cart_args(best)
            if args and not _check(args, self.schemas[cart], "args")[1]:
                actions.append({"name": cart, "args": args})
                idents.append((best, True))
        actions = actions[: max(0, self.steps_left() - 1)]
        if not actions:
            return
        reply = self.env_call(actions)
        self.trace.append({"kind": "forced_cart", "actions": actions, "reply": reply})
        if reply is None:
            return
        observations = reply.get("observations") if isinstance(reply.get("observations"), list) else [reply.get("observation")] * len(actions)
        for (ident, adding), observation in zip(idents, observations):
            if _faulty(observation):
                continue
            self.cart_used = True
            self.cart_ids = [i for i in self.cart_ids if i != ident] + ([ident] if adding else [])

    def force_finish(self) -> None:
        if self.finished or self.dead or self.steps_left() <= 0 or self.time_left() < 2:
            return
        order = self.roles.get("order")
        decline = self.roles.get("decline")
        if order and self.steps_left() >= 2 and self.time_left() > 8:
            self.force_repair()
        sorry = "Sorry, nothing in the store meets all of your requirements, so I am not ordering anything."
        multi = len(self.lines) > 1
        missing = multi and bool(self.cart_ids) and any(line not in {self.item_line(i) for i in self.cart_ids} for line in range(len(self.lines)))
        if order and self.cart_used and not missing:
            action = {"name": order, "args": self.neutral_args(order, "Order the cart.", "order")}
        elif decline:
            action = {"name": decline, "args": self.neutral_args(decline, sorry, "decline")}
        else:
            return

        def turned_down(reply: dict[str, Any] | None) -> bool:
            observation = (reply or {}).get("observation")
            return bool(reply) and not reply.get("finished") and isinstance(observation, dict) and ("refusal" in observation or _faulty(observation))

        reply = self.env_call([action])
        self.trace.append({"kind": "forced_finish", "action": action, "reply": reply})
        refused = turned_down(reply)
        if refused and action["name"] == order and self.cart_ids and self.steps_left() >= 3 and self.time_left() > 8:
            view = self.roles.get("view")
            checks = [{"name": view, "args": a} for a in (self.view_args(i) for i in self.cart_ids) if a][: max(0, self.steps_left() - 3)] if view else []
            if checks:
                seen = self.env_call(checks)                          # stock is read, not guessed from the refusal's wording
                self.trace.append({"kind": "forced_stock", "actions": checks, "reply": seen})
                for observation in (seen or {}).get("observations") or [(seen or {}).get("observation")]:
                    self.remember(view, observation)
            elif len(self.cart_ids) == 1:
                said = " ".join(str(x) for x in (reply["observation"].get("refusal") or []))
                if re.search(r"(?i)sold.?out|out of stock|all gone|it'?s gone|gone from|dried up|run dry|ran dry|none left|nowt left|none to be had|none remaining|no units|no more left|bought the last|every last one|cleared (?:out|from)|hit zero|ran out|run out|snapped up|unavailab|no longer (?:available|in stock|sold)", said):
                    self.stock_seen[self.cart_ids[0]] = (self.step, False)
            self.force_repair()
            if self.cart_ids and self.steps_left() >= 1 and not any(self.stock_seen.get(i, (0, True))[1] is False for i in self.cart_ids):
                reply = self.env_call([action])                       # one more try with the repaired cart (or after a change notice)
                self.trace.append({"kind": "forced_finish", "action": action, "reply": reply})
                refused = turned_down(reply)
        if refused and decline and self.steps_left() > 0:
            action = {"name": decline, "args": self.neutral_args(decline, sorry, "decline")}
            reply = self.env_call([action])
            self.trace.append({"kind": "forced_finish", "action": action, "reply": reply})

    def note_observation(self, name: str, observation: Any) -> str:
        notes = []
        gone = [i for i in self.cart_ids if self.stock_seen.get(i, (0, True))[1] is False and self.stock_seen.get(i, (0, True))[0] == self.step]
        if gone:
            notes.append("OUT OF STOCK NOW: " + ", ".join(gone) + " (in your cart). " + ("Send the cart again without it and with the next verified fit for that product (a cart call lists the whole cart: include every line you keep) before ordering." if self.list_cart() else "Take it out and put in the next verified fit for that product before ordering."))
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
        replies: list[tuple[str, str]] = []
        lead: dict[str, str] = {}
        allowed = {t["function"]["name"] for t in self.tools_for_turn()}
        runnable: list[tuple[str, str, dict[str, Any]]] = []
        finishing = {self.roles.get("order"), self.roles.get("decline")} - {None}
        message_tool = self.roles.get("message")
        cart_tool = self.roles.get("cart")
        planned_messages = 0
        cart_held = False
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
                adds = [i for i, adding in self.cart_moves(name, args) if adding and f"{i}|{self.ledger_version}" not in self.cart_checked]
                verdicts = []
                for target in adds[:3]:
                    self.cart_checked.add(f"{target}|{self.ledger_version}")
                    verdict = self.rule_check(target, True) or self.verify_item(target, "ADD TO THE CART")
                    if verdict:
                        verdicts.append(verdict)
                if verdicts:
                    cart_held = True
                    replies.append((call_id, "Not run (no step used). Your own pre-check (not the store) flags this: " + " ".join(verdicts) + " If you are certain it fits, send the same call again and it will go through."))
                    continue
            if name == cart_tool and not planned:
                moves = self.cart_moves(name, args)
                covered = {self.item_line(i) for i in self.cart_ids if i not in {m for m, adding in moves if not adding}}
                back = [i for i, adding in moves if adding and i in self.cart_removed and i not in self.readd_told and self.item_line(i) in covered]
                if back and self.steps_left() > 2:
                    cart_held = True
                    self.readd_told.update(back)
                    replies.append((call_id, "Not run (no step used): you removed " + ", ".join(back) + " earlier and are now putting it back. Swapping the cart back and forth costs steps you need for checkout. Decide once: keep the current cart (re-check its stock, then order), or send this same call again if you are sure."))
                    continue
            if name == cart_tool:
                warning = self.quantity_warning(name, args)
                if warning and signature not in self.qty_warned:
                    cart_held = True
                    self.qty_warned.add(signature)
                    replies.append((call_id, "Not run (no step used): " + warning + " If you are sure, send the same call again."))
                    continue
            if name == message_tool and not planned and signature not in self.told_once:
                said = set(_plain(" ".join(str(v) for v in args.values() if isinstance(v, str))))
                repeat = next((a.split(": asked", 1)[0][2:] for a in self.asked if (set(_plain(a.split(": asked", 1)[0][2:])) - {"type", "the", "of"}) and (set(_plain(a.split(": asked", 1)[0][2:])) - {"type", "the", "of"}) <= said), None)
                if repeat:
                    self.told_once.add(signature)
                    replies.append((call_id, f"Not run (no step used): the shopper already answered about the {repeat}; a 'don't mind' or 'say it simpler' reply means it is not one of their needs. Do not ask about it again; ask about a different property, or carry on shopping."))
                    continue
            if name == message_tool and not planned and "?" not in " ".join(str(v) for v in args.values() if isinstance(v, str)) and signature not in self.told_once:
                self.told_once.add(signature)
                replies.append((call_id, "Not run (no step used): a message to the shopper must ask one question about one property; a statement or a confirmation only spends the message budget. If a rule in the request requires this message, send the same call again."))
                continue
            if name == message_tool:
                planned_messages += 1
            runnable.append((call_id, name, args))
        if cart_held and any(n == cart_tool for _, n, _ in runnable):
            for call_id, _, _ in [r for r in runnable if r[1] == cart_tool]:
                replies.append((call_id, "Not run (no step used): held back because another cart change in this same batch was flagged above; cart changes sent together go through together. Settle the flagged one, then send them again."))
            runnable = [r for r in runnable if r[1] != cart_tool]
        if runnable and runnable[0][1] in finishing:
            call_id, name, args = runnable[0]
            kind = self.intent(name, args)
            if kind == "order" and self.stock_rounds < 3:
                guard = self.stock_guard()
                if guard:
                    replies.append((call_id, "Not run yet (no step used for the order). " + guard))
                    return replies
            if kind == "order" and self.cart_ids and len(self.lines) <= 1 and self.steps_left() > len(self.cart_ids) + 4:
                ladder = self.price_ladder(self.cart_ids[-1])
                if ladder:
                    replies.append((call_id, "Not run yet (no step used for the order). " + ladder))
                    return replies
            if kind == "order" and not self.order_checked and self.steps_left() > len(self.cart_ids) + 4 and self.time_left() > WRAP_UP_SECONDS:
                note = self.order_note(self.cart_ids[-1] if self.cart_ids else "")
                if note:
                    replies.append((call_id, "Not run yet (no step used). " + note))
                    return replies
                self.order_checked = True
                verdict = self.verify_cart()
                if verdict:
                    replies.append((call_id, "Not run yet (no step used). " + verdict))
                    return replies
                if len(self.lines) > 1:
                    shape = "the cart holds exactly one line for each requested product (" + ", ".join(f"{line['product']} x{line['quantity']}" for line in self.lines) + ") and nothing else"
                else:
                    shape = f"the cart holds exactly one line, the chosen item at quantity {self.lines[0]['quantity'] if self.lines else 1}, and nothing else"
                lead_note, self.guard_ok = self.guard_ok, ""
                checks = [shape]
                if not lead_note:
                    checks.append("after your last cart change every line's stock and price were re-read with a tool that shows them: in stock, and the total within the current budget")
                checks += ["each line's own properties meet every stated requirement, every need the shopper revealed and every change announced for that product",
                           "if the cheapest or best was asked for, no better fitting item exists among everything you could find",
                           "every procedural rule in the request has been followed"]
                listed = "; ".join(f"({n}) {c}" for n, c in enumerate(checks, 1))
                replies.append((call_id, f"Not run yet (no step used). {lead_note}Pre-order check: {listed}. If anything is unchecked, check it now; otherwise call the order tool again."))
                return replies
            if kind == "decline":
                swap = self.instead_of_decline()
                if swap is not None and _check(swap[1], self.schemas.get(swap[0]) or {}, "args")[1]:
                    swap = None                                       # the swapped call would not pass the tool's schema
                if swap is not None:
                    self.trace.append({"kind": "instead_of_decline", "tool": swap[0], "args": swap[1]})
                    lead[call_id] = swap[2]                           # one result per call id: the note goes in front of the swapped call's result
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
        for start in range(0, len(runnable), 16):
            chunk = runnable[start : start + 16]
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
                failed_cart = name == cart_tool and _faulty(observation)
                if name == cart_tool and not failed_cart:
                    self.cart_used = True
                    self.order_checked = False
                if name in (cart_tool, message_tool) or name in finishing:
                    self.recent = []
                else:
                    self.recent = (self.recent + [name + _compact(args)])[-2:]
                self.records.append(observation)
                self.remember(name, observation)
                if name == cart_tool and not failed_cart:
                    for ident, adding in self.cart_moves(name, args):
                        if not adding:
                            self.cart_removed.add(ident)
                    self.track_nested_cart(args)
                    properties = self.schemas[name].get("properties") or {}
                    signature = _compact({k: v for k, v in args.items() if isinstance(properties.get(k), dict) and "enum" in properties[k]})
                    if self.add_signature is None:
                        self.add_signature = signature
                    for key, value in args.items():
                        spec = properties.get(key) if isinstance(properties.get(key), dict) else {}
                        if isinstance(value, str) and "enum" not in spec and value.strip() and " " not in value.strip() and not value.strip().startswith(("[", "{")):
                            self.cart_state[value] = signature
                            count = next((v for k, v in args.items() if isinstance(v, int) and not isinstance(v, bool)), None)
                            if signature == self.add_signature and count != 0:
                                self.cart_ids = [i for i in self.cart_ids if i != value] + [value]
                                self.added_at[value] = self.step
                            else:
                                self.cart_ids = [i for i in self.cart_ids if i != value]
                    self.cart_changed_at = self.step
                if name == self.roles.get("cart_view"):
                    self.last_cart_view = observation
                    self.sync_cart_view(observation)
                note = " ".join(x for x in (self.note_observation(name, observation), self.page_note(name, args, observation)) if x)
                if name == cart_tool and not failed_cart and self.list_cart():
                    held = ", ".join(f"{i} x{self.cart_counts.get(i, '?')}" + (f" (the request asks for {self.line_qty(i)})" if self.cart_counts.get(i) not in (None, self.line_qty(i)) else "") for i in self.cart_ids)
                    note = (note + " " if note else "") + "A call to this cart tool sets the whole cart (lines left out are removed); it now holds: " + (held or "nothing") + "."
                heard: list[str] = []
                notices: list[str] = []
                if name == message_tool:
                    if isinstance(observation, str):
                        heard = [observation]
                    elif isinstance(observation, dict) and "error" not in observation:
                        _sentences([v for v in observation.values() if isinstance(v, str)], heard)
                        _sentences([v for v in observation.values() if isinstance(v, list)], notices)
                        if not heard:
                            heard, notices = notices, []                   # the reply itself came as lines
                    question = " ".join(str(v) for v in args.values() if isinstance(v, str))
                else:
                    question = None
                    if isinstance(observation, dict):
                        for value in observation.values():
                            if isinstance(value, list) and value and all(isinstance(v, str) for v in value):
                                _sentences(value, heard)
                            elif isinstance(value, str) and len(value.split()) >= 4 and ("refusal" in observation):
                                heard.append(value)
                if heard or notices:
                    meaning = self.interpret(heard, question, notices)
                    if meaning:
                        note = (note + " " if note else "") + meaning
                text = _compact(_trim(observation))
                if index == len(observations) - 1 and reply.get("finished"):
                    text += " [the episode is finished]"
                if note:
                    text += "\n[note] " + note
                if call_id in lead:
                    text = lead.pop(call_id) + "\n" + text
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
            if self.time_left() < FORCE_SECONDS or self.cost > SPEND_STOP:
                self.force_finish()
                break
            if self.pending_reply:
                text, self.pending_reply = self.pending_reply, ""
                if self.send_reply(text):
                    continue
            if self.planner_on and self.ask_planned():
                continue
            message = self.infer(self.messages, self.tools_for_turn(), [MAIN_MODEL, BACKUP_MODEL, MAIN_MODEL], 1600)
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
