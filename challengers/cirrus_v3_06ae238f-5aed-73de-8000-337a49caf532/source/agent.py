from __future__ import annotations

import json
import re
import time
from typing import Any

from src.agent.proxy_client import ProxyClient

_FIT_LINE_FILLER = " |:" + "".join(str(d) for d in range(10)) + "product"

DRIVER = "anthropic/claude-haiku-5.5"
SPARE = "openai/gpt-6-luna"
STRONG = "anthropic/claude-sonnet-5.5"
LIGHT = "anthropic/claude-haiku-5.5"

EPISODE_SECONDS = 265.0
CLOSING_SECONDS = 70.0
LAST_SECONDS = 28.0
VIEW_LIMIT = 16000
TURN_LIMIT = 80
SOFT_SPEND = 0.03
HARD_SPEND = 0.048
CLOSE_SPEND = 0.044
STOP_SPEND = 0.048
PROMPT_CHARS = 240 * 1000
RATES = {"anthropic/claude-haiku-5.5": (0.10, 0.50), "anthropic/claude-sonnet-5.5": (2.0, 10.0)}

DRIVER_RULES = """You shop for one person in a simulated online store, and everything you do is a call to one of the tools you are given. The episode is graded pass or fail. It passes only when your finish is completely correct: what you buy satisfies everything the request states, every preference the shopper discloses when you ask, and every update the shopper sends while you work; the units bought and the total spend are right; when the request wants the cheapest or the best, nothing better that also satisfies everything exists; and every working instruction in the request was obeyed. When the store truly has nothing that satisfies everything, ending without a purchase, in the manner the request describes, is the correct finish.

STEPS AND TOOLS
- Each tool call costs one step, even a call that fails, and the step allowance is fixed. Thinking costs nothing.
- Calls that do not depend on each other's results may share one reply (for instance opening three listings); they run in the given order. A purchase, or an ending without one, always goes alone in its reply.
- When a tool returns the details of several listings at once, read candidates with it in groups.
- Each store names its tools, arguments and result fields in its own way. Use only argument names and values the schemas allow, and learn from the first results where price, stock and detailed properties appear.

SEARCHING
- In a search phrase, lead with the two words that best identify the product: some stores read only the first two or three words.
- Learn which category labels the store gives the requested kind of product. When similar labels exist, the right one is the label matching the request's wording as a whole; shop inside it.
- Use the store's own filters (category, price ceiling, maker) whenever the search tool offers them.
- Result lists may be capped or ordered by id. Compare any reported total with the rows you actually received, and keep narrowing or paging until you have seen the relevant set. Continue a result list only with a page number inside the allowed range or a continuation value the store itself returned; never make one up.

ASKING THE SHOPPER
- The shopper holds preferences the request does not mention. One comes out only when a message asks about that property, and only the first thing a message asks gets an answer. So every message is one short, friendly question about one property that names it (one or two example values are fine). No instructions, no talk of lists, labels or rules, never two questions in one message.
- An answer such as any, no preference or doesn't matter leaves that property open. An answer naming a value or a limit makes it a firm requirement from then on, worded as given.
- While the planner is active, questions are written and sent for you once you have opened two or three listings of the right kind, so open some early. Treat each disclosed preference as firm and search again with it.
- If you write questions yourself: open a few listings first, then ask about what the story points to (who uses it, where, how it must look), then about properties that differ between candidates. Never ask about something the request already decides, and stay within the message allowance.

THE REQUEST'S OWN RULES
- Instructions about how to work (how many listings to compare or open, what to check, what to tell the shopper, how to end without buying) are graded like requirements. Follow each literally and completely. Looking at N listings means N fully suitable listings, each opened with the details tool.
- What the shopper would merely like (ideally, if possible, nice to have) is no requirement; it only breaks ties among items that already satisfy everything, and never overrides a cheapest rule.
- If something can only come from the shopper (no price limit was given, or two wishes may conflict and the shopper must rank them), ask before choosing.

UPDATES AND REFUSALS
- The shopper can change things at any time: a new budget, a requirement removed, replaced or added, a price rise, a sell-out. Updates arrive as sentences inside tool results or with a refused purchase, and one line can hold several. Each binds from that moment: re-check the pick and the backups, and redo a cheapest or best choice over the new set.
- A purchase attempt can be refused: its result then has a "refusal" field even when the rest looks fine. The episode goes on and the cart stays as it was. Apply every change in the refusal, take out whatever is no longer acceptable, put in the next verified fit, re-check its stock and price, and try again.
- Stock and prices can also change silently, so re-read them just before each purchase attempt.

DECIDING WHETHER A LISTING FITS
- A requirement is met only when the listing's own property for it states the value. Property names vary (spaces, dots, underscores, prefixes) and one property can appear under several names; any of them counts. Values match by meaning: one entry of a comma list is enough, singular equals plural, case is ignored, near-synonyms count. A word that merely begins the same way does not count, nor does a wider or different group (girls is not women).
- Title and description count only when the listing has no property for that requirement at all. If its own property says otherwise, the listing fails whatever the title says. A value under a different, merely related property does not count.
- Numbers are compared after unit conversion; limits include the boundary; a range passes only if all of it lies inside the limit. Respect exclusions and closed lists.
- The listing's category settles the product kind; no property needs to repeat it. The listing must sit in the category whose label matches the request's wording as a whole. It must be in stock when the request wants items on hand, and price times quantity must stay within the budget (equal is fine).
- A property listing several options, the needed one among them, simply matches. Evidence is mixed only when the listing's own fields state values that cannot both hold; only then, and only when no cheapest or best rule decides, prefer a candidate with clean evidence.

CHEAPEST AND BEST
- Cheapest means the lowest price among ALL listings that satisfy everything, not the first fit found. Use price ceilings or filters, sort candidates by price yourself, and check them from the cheapest up; the first that fully fits, with every cheaper one checked and rejected, is the answer. Keep the next fits as backups.
- Best by some measure (most, largest, longest and so on) works the same way: compare all fitting listings on that measure. A listing stating no value for it cannot win.

SHOPPING LISTS
- A request can name several products. Each one is its own search with its own requirements and quantity, and the cart must end with one line per product at that product's quantity and nothing else.
- Lists need more steps: build the cart early and place the order while enough steps remain to recover from one refusal.

FINISHING
- Keep a ranked set of verified fits with at least one backup per product, because a pick can sell out at checkout.
- Before each purchase attempt: the cart holds exactly the planned lines at the planned quantities and nothing else, and you re-read each line's stock and price after your last cart change.
- End without buying only after a thorough search found nothing that satisfies everything. Such an ending scores 0 whenever some listing would have satisfied the stated requirements and the disclosed preferences, so when a fitting listing exists, buy it. Follow the request's rule for that ending exactly; if it forbids offering alternatives, the closing text names no listing, id, title or price.
- Make the first purchase attempt while at least seven steps remain. From seven steps left, only cart, cart-view, stock-check and finishing tools stay available. Decide once; every cart change costs a step.
- Every reply must call at least one tool. Keep reply text short: current requirements, verified fits with price and stock, next move."""

READ_REQUEST = """You prepare a shopping request for an assistant that will carry it out in an online store. Read the request and the tool list, then answer with a single JSON object on one line, without commentary, using exactly these keys:

"items": one entry per distinct product the shopper wants bought, in the order the request gives them: {"product": the kind of product in the request's words, "quantity": units of that one listing to buy, "needs": [...]}. A request for a single product has one entry.
  - "quantity" counts only what the buying words ask for ("two of", "3 units"; a pair or a set is 1). Numbers of people, guests, days, rooms or uses in the story are never a quantity. Default 1.
  - "needs" lists each firm condition the request puts on that product as "property: value" with the shopper's value. Write numeric limits with symbols: "property: >= 50 cm", "property: <= 2 kg", "property: 10-20 l". Write an exclusion as "property: not X" and a closed choice as "property: X or Y". Leave out price, budget, stock, quantity, delivery or shipping, and the product kind or category itself (words belonging to the category's name are covered by the category). Only conditions the request states outright belong here; what the story merely suggests goes to "hints".
"hints": [{"property": ..., "likely": ..., "why": ...}] for properties the story points at without stating a value (for example water resistance when the shopper will swim with it), named as a listing would name them, with the implied value or "" and the request's own words. Hints steer questions and tie-breaks only and never disqualify a listing. Empty for a plain request.
"price_cap": the most the shopper will pay in total, as a number, or null.
"currency": the money unit of the request, or null.
"in_stock_only": true when the shopper wants only items on hand.
"priority": "cheapest" only when the request explicitly asks for the lowest-priced fitting choice (for a list, the cheapest basket), "best" only when it explicitly asks for the best by some measure, else "none". Saying money is tight is no cheapest request.
"priority_detail": for "best", the measure; otherwise "".
"message_budget": how many messages the shopper allows, as an integer, or null.
"step_budget": how many steps or moves the request allows, as an integer, or null.
"rules": every working instruction the request gives the assistant (how many listings to compare, what to check or report, how to end without buying, what to avoid), copied word for word. What the shopper will do personally is story, not a rule.
"soft_wants": what the shopper would like but does not require (ideally, preferably, if possible). Never repeat these under needs.
"ask_first": what must be learned from the shopper before choosing, for example the spending limit when none is given, or which of two wishes wins when the request says both may not be possible (then put both wishes under soft_wants). Empty when nothing.
"before_order_message": only when the request explicitly requires a message to the shopper before the order is placed (for example confirming the final cart or stating the total), that duty in a few words; otherwise "". Questions about preferences are not such a duty.
"decline_no_names": true when the rule for ending without a purchase forbids suggesting or naming alternatives, false when it asks for them, null when there is no such rule.
"roles": the exact tool name for each job, or null: "search", "details" (shows one listing's properties), "view", "message" (writes to the shopper), "cart" (puts items into or takes them out of the cart), "cart_view", "order" (buys or checks out), "decline" (ends without buying).
"finish_args": when one tool both buys and ends without buying, {"order": {...}, "decline": {...}} holding the argument values that select each; otherwise null.
Every sentence of the request must land somewhere: a need, the priority, a soft want, a rule, a budget, a hint or plain story. When unsure whether something is a rule, list it under rules."""

READ_SHOPPER = """A shopping assistant needs to know what the shopper's latest words change. Answer with one JSON object and nothing else, using these keys:
"meaning": one short sentence restating what the shopper means.
"needs": newly stated firm conditions, each {"for": the product it applies to, named as in the requirement list, "need": "property: value"}. Write numeric limits with symbols (">= 50 cm", "<= 2 kg", "10-20 l"), exclusions as "not X", closed choices as "X or Y".
"no_preference": properties the shopper said do not matter.
"dropped": earlier conditions the shopper no longer requires, written as in the requirement list.
"new_budget": a new total spending limit as a number, or null.
"unavailable": true when the shopper says an item can no longer be bought (sold out, gone, removed), else false.
"requirements_now": the complete current set of firm conditions after everything said so far, each {"for": product, "need": "property: value"}: keep earlier conditions unless dropped or replaced (for a replaced one keep only the newest value) and add every new one. No price, stock, quantity or category entries, and nothing drawn from the story or from your own guesses: only what the request states and what the shopper has said.
"reply": when the shopper asked the assistant something or wants a confirmation, a short plain answer naming the change; otherwise "".
"understood": false when the shopper's words show the question was not understood or ask for it to be put again or more simply (and nothing else was said); otherwise true.
How to read: a single line may contain several changes, so report each. A replaced condition is both dropped (the old value) and new (the new value). If the shopper says which of two wishes matters more, the preferred one becomes a need. A stated budget is new_budget, higher or lower. A wish (ideally, would be nice) is not a need. Read the shopper literally and generously: colourful or indirect phrasing that points to a value means that value is wanted, and a value is excluded only when the shopper clearly rejects it. When the shopper only says anything is fine, put the property that was asked about under no_preference."""

CHECK_PICK = """You are the last check before a shopping assistant commits to a listing. Decide from the records below only, with the store's reading rules:
- A requirement passes only if the listing's own property for it states the value. Property names may be spelled several ways and may repeat; any of them counts. Values match by meaning: one entry of a comma list is enough, singular equals plural, near-synonyms count, case is ignored. A look-alike word or a wider or different group does not pass.
- Title and description count only when the listing has no property for that requirement at all; a property saying something else fails the requirement even when the title says the value. A related but different property does not count.
- Numbers are compared after unit conversion, limits include the boundary, and a range passes only if all of it lies inside the limit. Exclusions and closed lists apply.
- The listing must be the requested kind of product; when listings carry category labels, its label must be the one matching the request's wording as a whole.
- Price times quantity must not exceed the budget (equal passes). Records showing it out of stock fail it. When records disagree, the newest wins.
- Wishes marked as not required are not requirements.
Answer with one JSON object and nothing else: {"item": the listing id, "fits": true or false, "problems": [short strings naming each requirement the records clearly fail or never address]}. Never fail a listing only because no record shows its stock or price."""

ASK_PLAN = """You pick the single next question a shopping assistant sends to a shopper. The shopper keeps some preferences to themselves; each is the value of one product property, and it comes out only when a message asks about exactly that property. Some replies come back garbled or ask for a repeat; then the same property is asked again in plainer, shorter words.
Choose ONE property, in this order:
1. A property listed under ASK AGAIN (the shopper did not understand it last time).
2. Anything under MUST LEARN FIRST that has not been asked. For a missing spending limit, ask the most they want to spend. For two wishes that may conflict, name both and ask which matters more.
3. Otherwise a property from CANDIDATE PROPERTIES for the product named there: prefer one whose values differ across the listings, and one the story points at (who uses it, where, how it must look). Skip logistics such as warranty, condition, model codes, seller, pack or size lists.
Never pick something already decided (stated requirements, price, stock, quantity) or already asked and answered, including other spellings of it. Use the listings' own name for the property.
Write the question in at most twelve plain words, naming the property and the product, for example "Any preferred <property> for the <product>?". No example values, no instructions, no second question.
Answer with one JSON object and nothing else: {"property": "...", "for": "the product", "question": "..."}; use {"property": ""} when nothing is worth asking."""

SECOND_LOOK = """A shopping assistant is about to end without buying because it believes nothing fits. Check the opened listings below against the current requirements with the store's reading rules: a requirement passes when the listing's own property states the value (any spelling of the property name, one entry of a comma list, singular or plural and near-synonyms all count; a wider or different group does not); the title counts only when no property covers the requirement; numbers are compared after unit conversion and limits include the boundary; the price must fit the budget and the listing must not show as out of stock.
Answer with one JSON object and nothing else: {"possible_fits": [ids of listings that seem to meet every requirement, best first, at most 5], "reason": "one sentence"}."""

_llm = ProxyClient(timeout=70, max_retries=1, retry_delay=1)
_env = ProxyClient(timeout=95, max_retries=2, retry_delay=1)

_IDISH = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-:.]{3,63}$")
_MONEY = re.compile(r"^\s*[^\d\s]{0,3}\s*(\d[\d,]*(?:\.\d+)?)\s*[A-Za-z]{0,3}\s*$")
_SHELF_CUE = re.compile(r"(?i)categor|shelf|depart|aisle|section")
_STOCK_CUE = re.compile(r"(?i)stock|avail|hand")
_CURSOR_CUE = re.compile(r"(?i)page|cursor|token|bookmark|resume")
_TOTAL_CUE = re.compile(r"(?i)total|count|matched|found")
_FAULT_CUE = re.compile(r"(?i)error|fault|invalid")
_LOWER_CUE = re.compile(r">|\bat least\b|\bor more\b|\bminimum\b|\bover\b")
_UPPER_CUE = re.compile(r"<|\bat most\b|\bup to\b|\bor less\b|\bmaximum\b")
_NOT_PRODUCT = re.compile(r"(?i)price|budget|stock|quantit|categor")
_EXCLUDED = re.compile(r"^(?:not|no|without|except|excluding)\s+(.+)$")
_CHOICE_SPLIT = re.compile(r"\s+or\s+|(?<=[a-z])\s*/\s*(?=[a-z])|,\s*(?!\d)")
_ADD_CUE = re.compile(r"(?i)add|hold|put|plus|insert")
_DROP_CUE = re.compile(r"(?i)remov|releas|drop|delet|clear")
_BLANK = re.compile(r"(?i)\s*(?:n/?a|none|null|unknown|-)?\s*")
_BUY_CUE = re.compile(r"(?i)\bbuy|purchase|check.?out|commit|\border\b")
_STOP_CUE = re.compile(r"(?i)without|cancel|refus|decline|no purchase")
_ASK_CUE = re.compile(r"(?i)messag|shopper|customer|question|ask")
_REMOVE_CUE = re.compile(r"(?i)remov|drop|delete|takes? .{0,12}(?:out|off)")
_LOOK_CUE = re.compile(r"(?i)search|find|look|query|scan")
_MINOR_UNIT = re.compile(r"(?i)\bcents?\b")


def _shorten(node: Any, width: int, level: int = 0) -> Any:
    if level > 8:
        return None
    if isinstance(node, str):
        return node[:width] + "..." if len(node) > width else node
    if isinstance(node, list):
        return [_shorten(part, width, level + 1) for part in node]
    if isinstance(node, dict):
        return {name: _shorten(part, width, level + 1) for name, part in node.items()}
    return node


def _flat(node: Any) -> str:
    try:
        out = json.dumps(node, ensure_ascii=False, separators=(",", ":"), sort_keys=False)
    except (TypeError, ValueError):
        out = str(node)
    return out if len(out) <= VIEW_LIMIT else out[:VIEW_LIMIT] + "...(cut)"


def _object_in(text: Any, last: bool = True) -> dict[str, Any] | None:
    if not isinstance(text, str):
        return None
    body = text.strip()
    if body.startswith("```"):
        body = re.sub(r"^```\w*", "", body).rstrip("`").strip()
    pieces = [body]
    left, right = body.find("{"), body.rfind("}")
    if 0 <= left < right:
        pieces.append(body[left:right + 1])
    for piece in pieces + (list(reversed(_outer_objects(body))) if last else []):
        try:
            found = json.loads(piece)
        except ValueError:
            continue
        if isinstance(found, dict):
            return found
    return None


def _outer_objects(text: str) -> list[str]:
    spans: list[str] = []
    depth, start, quoted, escaped = 0, -1, False, False
    for pos, ch in enumerate(text):
        if quoted:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                quoted = False
        elif ch == '"' and depth > 0:
            quoted = True
        elif ch == "{":
            if depth == 0:
                start = pos
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0:
                spans.append(text[start:pos + 1])
    return spans


def _is_kind(value: Any, kind: str) -> bool:
    if isinstance(value, bool):
        return kind == "boolean"
    if kind == "integer":
        return isinstance(value, int)
    if kind == "number":
        return isinstance(value, (int, float))
    if kind == "string":
        return isinstance(value, str)
    if kind == "object":
        return isinstance(value, dict)
    if kind == "array":
        return isinstance(value, list)
    if kind == "null":
        return value is None
    return kind != "boolean"


def _kinds_of(spec: dict[str, Any]) -> list[str]:
    declared = spec.get("type")
    if isinstance(declared, str):
        return [declared]
    return [k for k in declared if isinstance(k, str)] if isinstance(declared, list) else []


def _nudge(value: Any, kinds: list[str]) -> Any:
    if not kinds:
        return value
    if any(_is_kind(value, k) for k in kinds):
        if "integer" in kinds and isinstance(value, float) and value.is_integer():
            return int(value)
        return value
    text = value.strip() if isinstance(value, str) else None
    for kind in kinds:
        if kind in ("integer", "number"):
            if isinstance(value, float) and value.is_integer():
                return int(value)
            if text is not None and re.fullmatch(r"-?\d+(?:\.\d+)?", text):
                number = float(text)
                if number.is_integer():
                    return int(number)
                if kind == "number":
                    return number
        elif kind == "string" and isinstance(value, (int, float)) and not isinstance(value, bool):
            return str(value)
        elif kind == "boolean" and text is not None and text.lower() in ("true", "false"):
            return text.lower() == "true"
    return value


def _range_issues(number: float, spec: dict[str, Any], where: str) -> list[str]:
    out = []
    low, high = spec.get("minimum"), spec.get("maximum")
    if isinstance(low, (int, float)) and not isinstance(low, bool) and number < low:
        out.append(f"{where}: below minimum {low}")
    if isinstance(high, (int, float)) and not isinstance(high, bool) and number > high:
        out.append(f"{where}: above maximum {high}")
    floor, ceiling = spec.get("exclusiveMinimum"), spec.get("exclusiveMaximum")
    if isinstance(floor, (int, float)) and not isinstance(floor, bool) and number <= floor:
        out.append(f"{where}: must be above {floor}")
    if isinstance(ceiling, (int, float)) and not isinstance(ceiling, bool) and number >= ceiling:
        out.append(f"{where}: must be below {ceiling}")
    return out


def _gate(value: Any, spec: Any, where: str) -> tuple[Any, list[str]]:
    if not isinstance(spec, dict):
        return value, []
    kinds = _kinds_of(spec)
    value = _nudge(value, kinds)
    if kinds and not any(_is_kind(value, k) for k in kinds):
        return value, [f"{where}: expected {'/'.join(kinds)}"]
    issues: list[str] = []
    options = spec.get("enum")
    if isinstance(options, list) and value not in options:
        hit = None
        if isinstance(value, str):
            hit = next((o for o in options if isinstance(o, str) and o.lower() == value.strip().lower()), None)
        if hit is None:
            issues.append(f"{where}: must be one of {options}")
        else:
            value = hit
    if "const" in spec and value != spec["const"]:
        issues.append(f"{where}: must equal {spec['const']!r}")
    if isinstance(value, str):
        shortest, longest = spec.get("minLength"), spec.get("maxLength")
        if isinstance(shortest, int) and len(value) < shortest:
            issues.append(f"{where}: shorter than {shortest} characters")
        if isinstance(longest, int) and len(value) > longest:
            value = value[:longest]
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        issues += _range_issues(value, spec, where)
    elif isinstance(value, list):
        fewest, most = spec.get("minItems"), spec.get("maxItems")
        if isinstance(fewest, int) and len(value) < fewest:
            issues.append(f"{where}: needs at least {fewest} items")
        if isinstance(most, int) and len(value) > most:
            issues.append(f"{where}: allows at most {most} items")
        inner = spec.get("items")
        if isinstance(inner, dict):
            mended = []
            for pos, entry in enumerate(value):
                entry, more = _gate(entry, inner, f"{where}[{pos}]")
                mended.append(entry)
                issues += more
            value = mended
    elif isinstance(value, dict):
        fields = spec.get("properties") or {}
        issues += [f"{where}.{name}: required" for name in spec.get("required") or [] if name not in value]
        if spec.get("additionalProperties") is False:
            issues += [f"{where}.{name}: not an allowed argument" for name in value if name not in fields]
        mended_obj = {}
        for name, entry in value.items():
            if name in fields:
                entry, more = _gate(entry, fields[name], f"{where}.{name}")
                issues += more
            mended_obj[name] = entry
        value = mended_obj
    return value, issues


def _lines_from(node: Any, out: list[str], level: int = 0) -> None:
    if level > 4:
        return
    if isinstance(node, str):
        if len(node.split()) >= 4:
            out.append(node)
    elif isinstance(node, list):
        for part in node[:10]:
            _lines_from(part, out, level + 1)


def _gather_ids(node: Any, out: set[str], level: int = 0) -> None:
    if level > 6:
        return
    if isinstance(node, dict):
        for name, part in node.items():
            if isinstance(part, str) and "id" in str(name).lower() and _IDISH.match(part):
                out.add(part)
            else:
                _gather_ids(part, out, level + 1)
    elif isinstance(node, list):
        for part in node:
            _gather_ids(part, out, level + 1)


def _money(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        found = _MONEY.match(value)
        if found and re.search(r"\d", value):
            try:
                return float(found.group(1).replace(",", ""))
            except ValueError:
                return None
    return None


def _idlike(value: Any) -> bool:
    return isinstance(value, str) and bool(_IDISH.match(value)) and any(ch.isdigit() for ch in value)


def _option_meaning(option: str, description: str, others: list[str]) -> str:
    text = description.lower()
    spot = re.search(r"\b" + re.escape(option.lower()) + r"\b([^;.]{0,60})", text)
    tail = spot.group(1) if spot else ""
    for other in others:
        if other.lower() != option.lower():
            cut = re.search(r"\b" + re.escape(other.lower()) + r"\b", tail)
            if cut:
                tail = tail[:cut.start()]
    return option + " " + tail


def _finish_split(options: list[str], note: str) -> tuple[list[str], list[str]]:
    stops = [o for o in options if _STOP_CUE.search(_option_meaning(o, note, options)) or _STOP_CUE.search(o)]
    buys = [o for o in options if o not in stops and (_BUY_CUE.search(o) or len(options) == 2)]
    return stops, buys


def _unwrap(observation: Any) -> tuple[Any, list[str]]:
    if not isinstance(observation, dict):
        return observation, []
    if "refusal" in observation and isinstance(observation.get("observation"), (dict, list)):
        said: list[str] = []
        refusal = observation.get("refusal")
        _lines_from(refusal if isinstance(refusal, list) else [refusal], said)
        return observation["observation"], said
    inner = [v for v in observation.values() if isinstance(v, dict)]
    rest = [v for v in observation.values() if not isinstance(v, dict)]
    if len(inner) == 1 and rest and all(isinstance(v, list) and all(isinstance(x, str) for x in v) for v in rest):
        said = []
        for part in rest:
            _lines_from(part, said)
        return inner[0], said
    return observation, []


def _facts_of(record: Any) -> dict[str, Any] | None:
    if not isinstance(record, dict):
        return None
    for part in record.values():
        if isinstance(part, dict) and part and all(isinstance(x, (str, int, float)) or x is None for x in part.values()):
            return part
        if isinstance(part, list) and len(part) >= 2 and all(isinstance(x, dict) and len(x) == 2 for x in part):
            table: dict[str, Any] = {}
            for entry in part:
                left, right = list(entry.values())
                if isinstance(left, str) and isinstance(right, (str, int, float)):
                    table[left] = right
            if len(table) >= 2:
                return table
    return None


def _tokens(text: Any, glue: bool = False) -> list[str]:
    raw = re.sub(r"'s\b", "", str(text if text is not None else "").lower())
    if glue:
        raw = re.sub(r"(\d)\s+(?=[a-z%])", r"\1", raw)
    return re.findall(r"[a-z0-9]+", raw)


def _base(word: str) -> str:
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith(("ches", "shes", "sses", "xes")):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def _one_edit(short: str, longer: str) -> bool:
    if len(short) == len(longer):
        return sum(1 for x, y in zip(short, longer) if x != y) <= 1
    return any(longer[:cut] + longer[cut + 1:] == short for cut in range(len(longer)))


def _akin(left: str, right: str) -> bool:
    a, b = _base(left), _base(right)
    if a == b:
        return True
    if any(ch.isdigit() for ch in a + b):
        return False
    short, longer = (a, b) if len(a) <= len(b) else (b, a)
    if longer.startswith(short):
        return len(short) >= 4 and len(longer) - len(short) <= 3
    shared = 0
    while shared < len(short) and short[shared] == longer[shared]:
        shared += 1
    if shared >= max(5, len(short) - 2):
        return True
    if len(short) < 5 or len(longer) - len(short) > 1:
        return False
    return _one_edit(short, longer)


def _letters_placed(words: list[str], have: list[str], text: str) -> bool:
    for at, letter in enumerate(words):
        if len(letter) != 1 or letter.isdigit() or len(words) < 2:
            continue
        partners = [words[j] for j in (at - 1, at + 1) if 0 <= j < len(words)]
        if any(ch.isdigit() for p in partners for ch in p):
            continue
        beside = [have[j + d] for j, h in enumerate(have) if h == letter for d in (-1, 1) if 0 <= j + d < len(have)]
        if not any(_akin(p, b) for p in partners for b in beside) and "".join(words) not in re.sub(r"[^a-z0-9]+", "", str(text).lower()):
            return False
    return True


def _says(value: str, text: str, placed: bool = True) -> bool:
    for glue in (True, False):
        words = _tokens(value, glue)
        wanted = [w for w in words if len(w) > 2 or w.isdigit()] or words
        have = _tokens(text, glue)
        if wanted and all(any(_akin(w, h) for h in have) for w in wanted) and (not placed or _letters_placed(words, have, text)):
            return True
    squeezed = re.sub(r"[^a-z0-9]+", "", str(value).lower())
    return len(squeezed) >= 8 and squeezed in re.sub(r"[^a-z0-9]+", "", str(text).lower())


def _label_words(label: str) -> list[str]:
    parts = _tokens(label)
    while len(parts) > 1 and len(parts[0]) <= 2:
        parts = parts[1:]
    return [_base(w) for w in parts]


def _covers(label: str, prop: str, spare: set[str]) -> bool:
    have, want = set(_label_words(label)), set(_label_words(prop))
    core = want - {"type"}
    if core and not core <= have:
        core, want = core - spare, want - spare
    if not core or not core <= have:
        return False
    if "type" in want and "type" not in have and have != core:
        return False
    return len(have - want) <= 2


def _limits(value: str) -> tuple[float | None, float | None] | None:
    text = value.lower().replace(",", "")
    numbers = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", text)]
    if not numbers:
        return None
    upper, lower = _UPPER_CUE.search(text), _LOWER_CUE.search(text)
    if len(numbers) >= 2 and not upper and not lower and re.search(r"\bbetween\b|\d\s*(?:-|to)\s*\d", text):
        return min(numbers[:2]), max(numbers[:2])
    if upper:
        return None, numbers[0]
    if lower:
        return numbers[0], None
    return None


def _figures(text: str) -> list[tuple[float, str]]:
    found = re.finditer(r"(?<![a-z0-9.])(\d+(?:\.\d+)?)\s*([a-z%\"]*)", str(text).lower().replace(",", ""))
    return [(float(m.group(1)), _base(m.group(2))) for m in found]


def _judge_fact(pairs: list[tuple[str, str]], titles: list[str], prop: str, value: str, spare: set[str]) -> tuple[str, str]:
    named = [(k, v) for k, v in pairs if _covers(k, prop, spare)]
    texts = [v for _, v in named] or titles
    shown = "; ".join(f"{k}: {v[:60]}" for k, v in named[:2]) or "no field named for it"
    window = _limits(value)
    barred = _EXCLUDED.match(value)
    if barred and window is None:
        return ("no" if any(_says(barred.group(1), t, placed=False) for t in texts) else "yes"), shown
    if window is not None:
        low, high = window
        unit = next((u for _, u in _figures(value) if u), "")
        seen = [f for t in texts for f in _figures(t)]
        if any((low is None or n >= low) and (high is None or n <= high) for n, _ in seen):
            return "yes", shown
        comparable = [n for n, u in seen if u == unit or not u or not unit]
        return ("no" if comparable and named else "unshown"), shown
    choices = [c for c in _CHOICE_SPLIT.split(value) if c.strip()] or [value]
    if any(_says(c, t) for c in choices for t in texts):
        return "yes", shown
    if not named:
        return "unshown", shown
    units = {u for _, u in _figures(value) if u}
    if units and any(u and u not in units for t in texts for _, u in _figures(t)):
        return "unshown", shown
    return "no", shown


def _states_exactly(pairs: list[tuple[str, str]], prop: str, value: str, spare: set[str]) -> bool:
    named = [(k, v) for k, v in pairs if _covers(k, prop, spare)]
    if not named:
        return False
    if _limits(value) is not None or _EXCLUDED.match(value):
        return _judge_fact(named, [], prop, value, spare)[0] == "yes"
    own = set(_label_words(prop)) | {_base(w) for w in spare}
    choices = [c for c in _CHOICE_SPLIT.split(value) if c.strip()] or [value]
    entries = [e for _, text in named for e in _CHOICE_SPLIT.split(text.lower()) if e.strip()]
    for entry in entries:
        have = [w for w in _tokens(entry) if _base(w) not in own]
        for choice in choices:
            want = [w for w in _tokens(choice) if _base(w) not in own]
            if have and want and all(any(_akin(a, b) for b in want) for a in have) and all(any(_akin(b, a) for a in have) for b in want):
                return True
    return False


class Lens:
    def __init__(self) -> None:
        self.stats: dict[str, dict[str, Any]] = {}
        self.votes: dict[str, int] = {}
        self.rows_seen = 0
        self.id_key: str | None = None
        self.price_key: str | None = None
        self.stock_key: str | None = None
        self.shelf_key: str | None = None
        self.title_key: str | None = None
        self.shelf_values: set[str] = set()

    def note_shelf_value(self, value: Any) -> None:
        if isinstance(value, str) and value.strip():
            self.shelf_values.add(value.strip().lower())

    def absorb(self, rows: list[Any]) -> None:
        for row in rows:
            if not isinstance(row, dict):
                continue
            self.rows_seen += 1
            for key, value in row.items():
                bucket = self.stats.setdefault(str(key), {"n": 0, "nums": set(), "bools": 0, "strs": set(), "ids": 0, "spaced": 0, "sizes": []})
                bucket["n"] += 1
                if isinstance(value, bool):
                    bucket["bools"] += 1
                    continue
                amount = _money(value)
                if amount is not None:
                    bucket["nums"].add(amount)
                if isinstance(value, str):
                    bucket["strs"].add(value.strip().lower())
                    bucket["sizes"].append(len(value))
                    if " " in value.strip():
                        bucket["spaced"] += 1
                    if _idlike(value):
                        bucket["ids"] += 1
                    if value.strip().lower() in self.shelf_values:
                        self.votes[str(key)] = self.votes.get(str(key), 0) + 1
        self.relearn()

    def relearn(self) -> None:
        total = max(1, self.rows_seen)
        stats = self.stats
        keys = sorted(stats)
        idish = [k for k in keys if stats[k]["ids"] >= 0.9 * stats[k]["n"] and stats[k]["n"] >= 0.8 * total]
        if idish:
            self.id_key = max(idish, key=lambda k: (len(stats[k]["strs"]), "id" in k.lower(), -len(k)))
        numeric = [k for k in keys if k != self.id_key and len(stats[k]["nums"]) >= 1 and stats[k]["n"] >= 0.6 * total
                   and len(stats[k]["sizes"]) <= stats[k]["n"] - len(stats[k]["nums"]) + len(stats[k]["sizes"])]
        if numeric:
            self.price_key = max(numeric, key=lambda k: (len(stats[k]["nums"]), max(stats[k]["nums"]), k))
        flags = [k for k in keys if stats[k]["bools"] >= 0.6 * stats[k]["n"] and stats[k]["n"] >= 0.5 * total]
        if flags:
            self.stock_key = max(flags, key=lambda k: (bool(_STOCK_CUE.search(k)), stats[k]["bools"], k))
        wordy = [k for k in keys if k != self.id_key and stats[k]["sizes"] and stats[k]["spaced"] >= 0.5 * len(stats[k]["sizes"])
                 and len(stats[k]["strs"]) >= 0.5 * len(stats[k]["sizes"])]
        if wordy:
            self.title_key = min(wordy, key=lambda k: (self._median(k) < 10, self._median(k), k))
        labels = [k for k in keys if k not in (self.id_key, self.title_key) and stats[k]["sizes"]
                  and max(stats[k]["sizes"]) <= 60 and len(stats[k]["sizes"]) >= 0.7 * total]
        if labels:
            self.shelf_key = max(labels, key=lambda k: (self.votes.get(k, 0) > 0, bool(_SHELF_CUE.search(k)),
                                                       len(stats[k]["sizes"]) / max(1, len(stats[k]["strs"])), k))

    def _median(self, key: str) -> float:
        sizes = sorted(self.stats[key]["sizes"])
        return float(sizes[len(sizes) // 2]) if sizes else 0.0

    def ident(self, record: Any) -> str | None:
        if not isinstance(record, dict):
            return None
        if self.id_key and _idlike(record.get(self.id_key)):
            return record[self.id_key]
        for key in sorted(record, key=lambda k: ("id" not in str(k).lower(), str(k))):
            value = record[key]
            if "id" in str(key).lower() and _idlike(value):
                return value
        return None

    def price(self, record: Any) -> float | None:
        if not isinstance(record, dict):
            return None
        if self.price_key and self.price_key in record:
            return _money(record[self.price_key])
        return None

    def stock(self, record: Any) -> bool | None:
        if not isinstance(record, dict):
            return None
        if self.stock_key and isinstance(record.get(self.stock_key), bool):
            return record[self.stock_key]
        for key in sorted(record):
            if _STOCK_CUE.search(str(key)) and isinstance(record[key], (bool, int)) and not isinstance(record[key], float):
                return bool(record[key])
        return None

    def shelf(self, record: Any) -> str:
        if not isinstance(record, dict):
            return ""
        if self.shelf_key and isinstance(record.get(self.shelf_key), str):
            return record[self.shelf_key].strip()
        for key in sorted(record):
            if _SHELF_CUE.search(str(key)) and isinstance(record[key], str) and record[key].strip():
                return record[key].strip()
        return ""

    def title(self, record: Any) -> str:
        if not isinstance(record, dict):
            return ""
        if self.title_key and isinstance(record.get(self.title_key), str):
            return record[self.title_key]
        texts = sorted((v for v in record.values() if isinstance(v, str) and " " in v and len(v) <= 240), key=len)
        return texts[0] if texts else ""


class Episode:
    def __init__(self, problem_data: dict[str, Any]):
        self.started = time.monotonic()
        world = problem_data["environment"]
        self.binding = dict(world["binding"])
        self.policy = world["policy_view"]
        self.tools = [t for t in self.policy.get("tools") or [] if isinstance(t, dict) and isinstance(t.get("function"), dict)]
        self.schemas = {t["function"]["name"]: t["function"].get("parameters") or {"type": "object", "properties": {}} for t in self.tools}
        self.max_steps = int(self.policy.get("max_steps") or 30)
        self.step = 0
        self.finished = bool(self.policy.get("finished"))
        self.dead = False
        self.lens = Lens()
        self.plan: dict[str, Any] = {}
        self.roles: dict[str, str] = {}
        self.finish_args: dict[str, dict[str, Any]] = {}
        self.wants: list[dict[str, Any]] = []
        self.free: list[str] = []
        self.budget_now: Any = None
        self.message_budget: int | None = None
        self.messages_used = 0
        self.order_attempts = 0
        self.refusals = 0
        self.taken_out: set[str] = set()
        self.affirmed: set[str] = set()
        self.order_checked = False
        self.decline_checked = False
        self.swaps = 0
        self.cart_writes = 0
        self.basket: dict[str, int] = {}
        self.cart_marks: dict[str, int] = {}
        self.basket_seen = False
        self.add_value: str | None = None
        self.seen_ids: set[str] = set()
        self.cursors: set[str] = set()
        self.heard: set[str] = set()
        self.records: list[Any] = []
        self.rows: dict[str, dict[str, Any]] = {}
        self.sheets: dict[str, dict[str, Any]] = {}
        self.offers: dict[str, dict[str, Any]] = {}
        self.searches = 0
        self.search_nudged = False
        self.asks: list[dict[str, Any]] = []
        self.unclear: dict[str, Any] | None = None
        self.planner_on = True
        self.planner_idle = 0
        self.planner_quiet = False
        self.ask_pause = False
        self.last_understood = True
        self.line_clear: dict[str, bool] = {}
        self.swapped_in: set[str] = set()
        self.gone_mark: dict[str, int] | None = None
        self.switch_args: dict[str, dict[str, Any]] = {}
        self.result_sets: dict[str, str] = {}
        self.audited = False
        self.bounced: set[str] = set()
        self.pending_reply = ""
        self.verdicts: dict[str, str] = {}
        self.version = 0
        self.said: list[str] = []
        self.recent: list[str] = []
        self.cost = 0.0
        self.last_turn_cost = 0.0
        self.duty = ""
        self.duty_done = False
        self.duty_phase = False
        self.duty_basket: dict[str, Any] = {}
        self.coverage: dict[str, dict[str, Any]] = {}
        self.fits_shown = ""
        self.trace: list[dict[str, Any]] = []
        self.messages: list[dict[str, Any]] = []

    def time_left(self) -> float:
        return EPISODE_SECONDS - (time.monotonic() - self.started)

    def steps_left(self) -> int:
        return self.max_steps - self.step

    def seats(self, strong: bool) -> list[str] | None:
        if self.cost >= HARD_SPEND:
            return None
        if strong and self.cost < SOFT_SPEND:
            return [STRONG, LIGHT]
        return [LIGHT, STRONG] if self.cost < SOFT_SPEND else [LIGHT]

    def call_model(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, models: list[str], max_tokens: int) -> dict[str, Any] | None:
        for model in models:
            budget = max_tokens
            for attempt in range(2):
                left = self.time_left()
                if left < LAST_SECONDS - 10:
                    return None
                _llm.timeout = int(max(8, min(75, left - LAST_SECONDS + 15)))
                sent = messages
                if model.startswith("anthropic/") and len(messages) > 2:
                    sent = [dict(m) for m in messages]
                    for pos in (0, len(sent) - 1):
                        if isinstance(sent[pos].get("content"), str) and sent[pos]["content"]:
                            sent[pos]["content"] = [{"type": "text", "text": sent[pos]["content"], "cache_control": {"type": "ephemeral"}}]
                body: dict[str, Any] = {"model": model, "messages": sent, "temperature": 0, "max_tokens": budget}
                if tools:
                    body["tools"] = tools
                    body["tool_choice"] = "auto"
                result = _llm.post_verbose("/inference/chat/completions", body)
                data = result.data
                if isinstance(data, dict) and data.get("choices"):
                    usage = data.get("usage") or {}
                    spent = self.bill(model, usage)
                    self.cost += spent
                    if tools:
                        self.last_turn_cost = spent
                    message = data["choices"][0].get("message") or {}
                    if message.get("content") or message.get("tool_calls"):
                        return message
                    self.trace.append({"kind": "model_empty", "model": model})
                    if data["choices"][0].get("finish_reason") == "length" and attempt == 0 and self.time_left() > LAST_SECONDS + 25:
                        budget = min(6000, int(budget * 1.6))
                        continue
                    break
                error = result.error or {}
                status = error.get("status")
                self.trace.append({"kind": "model_failure", "model": model, "status": status, "error": str(error.get("body") or "")[:300]})
                if status in (402, 403):
                    return None
                if (status is None or status in (200, 429, 502, 503, 529)) and attempt == 0 and self.time_left() > LAST_SECONDS + 25:
                    time.sleep(3.0 if status in (429, 529) else 1.0)
                    continue
                break
        return None

    def bill(self, model: str, usage: dict[str, Any]) -> float:
        if isinstance(usage.get("cost"), (int, float)) and not isinstance(usage.get("cost"), bool):
            return float(usage["cost"])
        prompt, output = usage.get("prompt_tokens"), usage.get("completion_tokens")
        if not isinstance(prompt, (int, float)) or not isinstance(output, (int, float)):
            return 0.004
        rate_in, rate_out = RATES.get(model, (2.0, 10.0))
        if prompt > 100 * 1000:
            rate_in *= 5
        details = usage.get("prompt_tokens_details") if isinstance(usage.get("prompt_tokens_details"), dict) else {}
        cached = details.get("cached_tokens") if isinstance(details.get("cached_tokens"), (int, float)) else 0
        written = details.get("cache_write_tokens") if isinstance(details.get("cache_write_tokens"), (int, float)) else 0
        plain = max(0, prompt - cached - written)
        return (plain * rate_in + cached * rate_in * 0.1 + written * rate_in * 1.25 + output * rate_out) / 1e6

    def closing(self) -> bool:
        return self.cost >= CLOSE_SPEND

    def act(self, actions: list[dict[str, Any]]) -> dict[str, Any] | None:
        payload: dict[str, Any] = dict(self.binding)
        payload["step"] = self.step + 1
        if len(actions) == 1:
            payload["action"] = actions[0]
        else:
            payload["actions"] = actions
        waits = 0
        for _ in range(5):
            left = self.time_left()
            if left < 2:
                return None
            _env.timeout = int(max(3, min(95, left - 1)))
            result = _env.post_verbose("/environment/call", payload)
            if isinstance(result.data, dict):
                data = result.data
                self.step = data["step"] if isinstance(data.get("step"), int) else self.step + len(actions)
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
                expected = re.search(r"expected (\d+)", body)
                if expected:
                    self.step = int(expected.group(1)) - 1
                    payload["step"] = self.step + 1
                    continue
                return None
            if status is None or status in (429, 502, 503, 504):
                if waits >= 2 or self.time_left() < 12:
                    return None
                waits += 1
                time.sleep(1.5 * waits)
                continue
            return None
        return None

    def setup(self) -> None:
        listing = []
        for tool in self.tools:
            spec = tool["function"]
            listing.append(f"- {spec['name']}: {spec.get('description', '')} | arguments: {_flat(spec.get('parameters') or {})}")
        fixed = READ_REQUEST + "\n\nTOOLS:\n" + "\n".join(listing)
        varying = "REQUEST:\n" + str(self.policy.get("query") or "")
        if self.policy.get("opening") is not None:
            varying += "\n\nOPENING OBSERVATION:\n" + _flat(self.policy.get("opening"))
        content = [{"type": "text", "text": fixed, "cache_control": {"type": "ephemeral"}}, {"type": "text", "text": varying}]
        reply = self.call_model([{"role": "user", "content": content}], None, [STRONG, SPARE], 1800)
        if reply is None and self.time_left() > EPISODE_SECONDS - 60:
            time.sleep(4.0)
            reply = self.call_model([{"role": "user", "content": content}], None, [STRONG, SPARE], 1800)
        plan = _object_in((reply or {}).get("content") or "") or {}
        self.plan = plan
        roles = plan.get("roles") if isinstance(plan.get("roles"), dict) else {}
        self.roles = {k: v for k, v in sorted(roles.items()) if isinstance(v, str) and v in self.schemas}
        modes = plan.get("finish_args")
        if isinstance(modes, dict):
            self.finish_args = {k: v for k, v in modes.items() if k in ("order", "decline") and isinstance(v, dict)}
            self.trim_finish_args()
        if self.roles.get("message") and isinstance(plan.get("before_order_message"), str):
            self.duty = plan["before_order_message"].strip()[:300]
        allowance = plan.get("message_budget")
        if isinstance(allowance, (int, float)) and not isinstance(allowance, bool) and allowance >= 0:
            self.message_budget = int(allowance)
        for entry in plan.get("items") or []:
            if not isinstance(entry, dict):
                continue
            count = entry.get("quantity")
            count = int(count) if isinstance(count, (int, float)) and not isinstance(count, bool) and count >= 1 else 1
            needs = [str(n) for n in entry.get("needs") or [] if isinstance(n, (str, int, float)) and str(n).strip()]
            self.wants.append({"product": str(entry.get("product") or "").strip(), "qty": count, "stated": needs, "revealed": [], "dropped": [], "now": None})
        if not self.wants:
            self.wants.append({"product": str(plan.get("product") or "").strip(), "qty": 1, "stated": [], "revealed": [], "dropped": [], "now": None})
        self.budget_now = _money(plan.get("price_cap")) if plan.get("price_cap") is not None else None
        self.budget_now = self.literal_budget(self.budget_now, plan)
        if isinstance(self.budget_now, float) and self.budget_now.is_integer():
            self.budget_now = int(self.budget_now)
        if plan and self.budget_now is None and self.roles.get("message"):
            first = [str(a) for a in plan.get("ask_first") or [] if a]
            plan["ask_first"] = ["the most the shopper is willing to spend, since the request gives no price limit (skip if this already appears below)"] + first
        allowed_steps = plan.get("step_budget")
        if isinstance(allowed_steps, (int, float)) and not isinstance(allowed_steps, bool) and 0 < allowed_steps < self.max_steps:
            self.max_steps = int(allowed_steps)
        search = self.roles.get("search")
        for spec in ((self.schemas.get(search) or {}).get("properties") or {}).values() if search else []:
            if isinstance(spec, dict) and isinstance(spec.get("enum"), list) and len(spec["enum"]) >= 5:
                for option in spec["enum"]:
                    self.lens.note_shelf_value(option)

    def trim_finish_args(self) -> None:
        for kind in list(self.finish_args):
            fields = (self.schemas.get(self.roles.get(kind) or "") or {}).get("properties") or {}
            if not fields:
                continue
            kept: dict[str, Any] = {}
            for key, value in self.finish_args[kind].items():
                entry = fields.get(key)
                options = entry.get("enum") if isinstance(entry, dict) else None
                if not isinstance(options, list):
                    kept[key] = value
                    continue
                same = [o for o in options if str(o).lower() == str(value).lower()]
                if same:
                    kept[key] = same[0]
            if kept:
                self.finish_args[kind] = kept
            else:
                del self.finish_args[kind]

    def literal_budget(self, cap: float | None, plan: dict[str, Any]) -> float | None:
        if cap is None or cap <= 0:
            return cap
        query = str(self.policy.get("query") or "")
        written = [float(v.replace(",", "")) for v in re.findall(r"\d[\d,]*(?:\.\d+)?", query)]
        if not _MINOR_UNIT.search(query) or any(abs(v - cap) < 1e-6 for v in written):
            return cap
        if any(abs(v - cap * 100) < 0.5 for v in written):
            plan["currency"] = _MINOR_UNIT.search(query).group(0).lower()
            return float(round(cap * 100))
        return cap

    def is_list(self) -> bool:
        return len(self.wants) > 1

    def overview(self) -> str:
        plan = self.plan
        if not plan:
            return ""
        out = ["How the request was read (compare it with the request itself):"]
        for pos, want in enumerate(self.wants):
            head = f"- product {pos + 1}: {want['product'] or 'see the request'}, quantity {want['qty']}" if self.is_list() else f"- product: {want['product'] or 'see the request'}; quantity {want['qty']}"
            out.append(head)
            out += [f"  - stated need: {n}" for n in want["stated"]]
        if self.budget_now is not None:
            out.append(f"- total price cap: {self.budget_now} {plan.get('currency') or ''}".rstrip())
        if plan.get("in_stock_only"):
            out.append("- in-stock items only")
        if plan.get("priority") in ("cheapest", "best"):
            extra = f" ({plan.get('priority_detail')})" if plan.get("priority_detail") else ""
            out.append(f"- pick the {plan.get('priority')}{extra} among everything that fits")
        else:
            out.append("- no cheapest or best rule: any listing meeting every requirement within the budget is right, so do not hunt for lower prices unless the request says so")
        out += [f"- wish only, not required (tie-breaker): {w}" for w in plan.get("soft_wants") or []]
        out += [f"- learn from the shopper first: {a}" for a in plan.get("ask_first") or []]
        out += [f"- rule: {r}" for r in plan.get("rules") or []]
        if self.message_budget is not None:
            out.append(f"- messages to the shopper allowed: {self.message_budget}")
        out.append(f"- steps allowed: {self.max_steps}")
        if self.is_list():
            out.append("- this is a shopping list: one cart line per product, each at its own quantity")
        return "\n".join(out)

    def needs_of(self, pos: int) -> list[str]:
        want = self.wants[pos]
        if want["now"] is not None:
            return list(want["now"])
        return [n for n in want["stated"] + want["revealed"] if n not in want["dropped"]]

    def ledger(self) -> str:
        parts = []
        for pos, want in enumerate(self.wants):
            needs = "; ".join(self.needs_of(pos)) or "only what the request says"
            if self.is_list():
                parts.append(f"product {pos + 1} ({want['product']}, quantity {want['qty']}): {needs}")
            else:
                parts.append(f"{needs}; quantity {want['qty']}")
        text = " | ".join(parts)
        if self.budget_now is not None:
            text += f"; total price at most {self.budget_now} {self.plan.get('currency') or ''}".rstrip()
        dropped = [d for want in self.wants for d in want["dropped"]]
        if dropped:
            text += "; no longer required: " + "; ".join(dropped)
        if self.free:
            text += "; the shopper does not mind: " + ", ".join(self.free)
        return text

    def demands(self, pos: int) -> list[tuple[str, str]]:
        out = []
        for need in self.needs_of(pos):
            prop, sep, value = str(need).partition(":")
            prop = prop.strip().lower()
            value = re.sub(r"\s*\([^)]*\)", "", value.strip().lower()).strip(" .")
            if sep and prop and value and not _NOT_PRODUCT.search(prop) and "brand" not in prop.split():
                out.append((prop, value))
        return out

    def product_words(self, pos: int) -> set[str]:
        return {_base(w) for w in _tokens(self.wants[pos]["product"]) if len(w) > 2}

    def hints(self) -> list[tuple[str, str]]:
        settled = [p for pos in range(len(self.wants)) for p, _ in self.demands(pos)] + list(self.free)
        out: list[tuple[str, str]] = []
        for entry in self.plan.get("hints") or []:
            prop = str(entry.get("property") if isinstance(entry, dict) else entry or "").strip().lower()
            likely = str(entry.get("likely") or "").strip().lower() if isinstance(entry, dict) else ""
            words = set(_label_words(prop)) - {"type"}
            if not words:
                continue
            if any(words <= set(_label_words(s)) or (set(_label_words(s)) - {"type"} and set(_label_words(s)) - {"type"} <= words) for s in settled):
                continue
            if prop not in [p for p, _ in out]:
                out.append((prop, likely))
        return out[:4]

    def price_of(self, ident: str) -> float | None:
        for record in reversed(self.records):
            found = self._find(record, ident, 0)
            if found is not None and self.lens.price(found) is not None:
                return self.lens.price(found)
        for source in (self.offers, self.rows, self.sheets):
            if ident in source and self.lens.price(source[ident]) is not None:
                return self.lens.price(source[ident])
        return None

    def stock_of(self, ident: str) -> bool | None:
        for record in reversed(self.records):
            found = self._find(record, ident, 0)
            if found is not None and self.lens.stock(found) is not None:
                return self.lens.stock(found)
        return None

    def shelf_of(self, ident: str) -> str:
        for source in (self.sheets, self.offers, self.rows):
            label = self.lens.shelf(source.get(ident))
            if label:
                return label
        return ""

    def _find(self, node: Any, ident: str, level: int) -> dict[str, Any] | None:
        if level > 6:
            return None
        if isinstance(node, dict):
            if self.lens.ident(node) == ident:
                return node
            for part in node.values():
                if isinstance(part, (dict, list)):
                    hit = self._find(part, ident, level + 1)
                    if hit is not None:
                        return hit
        elif isinstance(node, list):
            for part in node:
                hit = self._find(part, ident, level + 1)
                if hit is not None:
                    return hit
        return None

    def shelf_for(self, pos: int) -> str:
        counts: dict[str, int] = {}
        for ident in self.sheets:
            label = self.shelf_of(ident)
            if label:
                counts[label] = counts.get(label, 0) + 2
        for ident in self.rows:
            label = self.lens.shelf(self.rows[ident])
            if label:
                counts[label] = counts.get(label, 0) + 1
        if not counts:
            return ""
        words = self.product_words(pos)

        def overlap(label: str) -> int:
            return sum(1 for w in words if any(_akin(w, x) for x in _tokens(label) if len(x) > 2))

        best = max(sorted(counts), key=lambda label: (overlap(label), counts[label]))
        if overlap(best) > 0:
            return best
        if not self.is_list():
            sheet_counts = {k: v for k, v in counts.items() if v >= 2}
            return max(sorted(sheet_counts), key=lambda label: sheet_counts[label]) if sheet_counts else ""
        return ""

    def belongs(self, ident: str, pos: int) -> bool:
        if not self.is_list():
            target = self.shelf_for(0)
            label = self.shelf_of(ident)
            return not (target and label and label.lower() != target.lower())
        target = self.shelf_for(pos)
        label = self.shelf_of(ident)
        if target and label:
            return label.lower() == target.lower()
        words = self.product_words(pos)
        text = label + " " + self.lens.title(self.sheets.get(ident) or self.rows.get(ident) or self.offers.get(ident) or {})
        return any(any(_akin(w, x) for x in _tokens(text) if len(x) > 2) for w in words)

    def owner(self, ident: str) -> int | None:
        if not self.is_list():
            return 0
        owners = [pos for pos in range(len(self.wants)) if self.belongs(ident, pos)]
        return owners[0] if len(owners) == 1 else None

    def affordable(self, ident: str, pos: int) -> bool:
        price, cap = self.price_of(ident), self.budget_now
        qty = self.wants[pos]["qty"]
        if isinstance(cap, (int, float)) and not isinstance(cap, bool) and price is not None and price * qty > cap + 1e-9:
            return False
        return self.stock_of(ident) is not False

    def fact_pairs(self, ident: str) -> tuple[list[tuple[str, str]], list[str]]:
        sheet = self.sheets.get(ident) or {}
        facts = _facts_of(sheet) or {}
        pairs = []
        for key, value in list(sheet.items()) + list(facts.items()):
            if isinstance(value, (str, int, float)) and not isinstance(value, bool) and not _BLANK.fullmatch(str(value)):
                pairs.append((str(key), str(value)))
        title_key = self.lens.title_key
        titles = [str(sheet.get(title_key) or "")] if title_key and sheet.get(title_key) else []
        if not titles:
            row = self.rows.get(ident) or {}
            if title_key and row.get(title_key):
                titles = [str(row[title_key])]
        titles += [v for k, v in pairs if re.search(r"(?i)option|variant", k)]
        named_pairs = [(k, v) for k, v in pairs if k != title_key]
        return named_pairs, [t for t in titles if t]

    def reading(self, ident: str, prop: str, value: str, pos: int) -> tuple[str, str]:
        pairs, titles = self.fact_pairs(ident)
        return _judge_fact(pairs, titles, prop, value, self.product_words(pos))

    def proven(self, pos: int) -> list[tuple[str, float | None]]:
        wanted = self.demands(pos)
        if not wanted:
            return []
        out = []
        for ident in sorted(self.sheets):
            if not self.belongs(ident, pos) or not self.affordable(ident, pos):
                continue
            if all(self.reading(ident, p, v, pos)[0] == "yes" for p, v in wanted):
                out.append((ident, self.price_of(ident)))
        out.sort(key=lambda pair: (pair[1] is None, pair[1] or 0.0, pair[0]))
        return out

    def plausible(self, pos: int) -> list[tuple[str, float | None]]:
        wanted = self.demands(pos)
        ranked = []
        for ident in sorted(self.sheets):
            if not self.belongs(ident, pos) or not self.affordable(ident, pos):
                continue
            marks = [self.reading(ident, p, v, pos)[0] for p, v in wanted]
            if "no" in marks:
                continue
            if any(m == "unshown" and not self.mentions(ident, v) for m, (_, v) in zip(marks, wanted)):
                continue
            ranked.append((ident, self.price_of(ident), marks.count("yes")))
        ranked.sort(key=lambda row: (-row[2], row[1] is None, row[1] or 0.0, row[0]))
        return [(i, p) for i, p, _ in ranked]

    def mentions(self, ident: str, value: str) -> bool:
        if _limits(value) is not None or _EXCLUDED.match(value):
            return False
        sheet = self.sheets.get(ident) or {}
        texts = [str(v) for v in (_facts_of(sheet) or {}).values()] + [str(v) for v in sheet.values() if isinstance(v, str)]
        choices = [c for c in re.split(r"\s+or\s+|,\s*(?!\d)", value) if c.strip()] or [value]
        return any(_says(c, t) for c in choices for t in texts if t)

    def favourite(self, fits: list[tuple[str, float | None]], pos: int) -> str:
        if self.plan.get("priority") in ("cheapest", "best"):
            return fits[0][0]
        for ident, _ in fits:
            if any(v and self.reading(ident, p, v, pos)[0] == "yes" for p, v in self.hints()):
                return ident
        return fits[0][0]

    def unread(self, pos: int) -> list[str]:
        target = self.shelf_for(pos)
        wanted = [v for _, v in self.demands(pos)]
        pool = []
        for ident, row in sorted(self.rows.items()):
            if ident in self.sheets or not self.affordable(ident, pos):
                continue
            label = self.lens.shelf(row)
            if target and label and label.lower() != target.lower():
                continue
            if self.is_list() and not self.belongs(ident, pos):
                continue
            text = " ".join(str(v) for v in row.values() if isinstance(v, str))
            hits = sum(1 for v in wanted if _limits(v) is None and _says(v, text))
            price = self.price_of(ident)
            pool.append((-hits, price if price is not None else float("inf"), ident))
        pool.sort()
        return [f"{i} ({p:g})" if p != float("inf") else i for _, p, i in pool[:5]]

    def cart_spec(self) -> dict[str, Any]:
        return self.schemas.get(self.roles.get("cart") or "") or {}

    def cart_shape(self) -> tuple[str, dict[str, Any]]:
        spec = self.cart_spec()
        fields = spec.get("properties") or {}
        for name in sorted(fields):
            entry = fields[name]
            if isinstance(entry, dict) and "array" in _kinds_of(entry) and isinstance(entry.get("items"), dict):
                inner = entry["items"].get("properties") or {}
                text_field = next((k for k in sorted(inner, key=lambda k: ("id" not in k.lower(), k)) if "string" in _kinds_of(inner[k] or {}) and "enum" not in inner[k]), None)
                count_field = next((k for k in sorted(inner) if set(_kinds_of(inner[k] or {})) & {"integer", "number"}), None)
                if text_field:
                    return "array", {"field": name, "id": text_field, "count": count_field}
        verb = next((k for k in sorted(fields) if isinstance(fields[k], dict) and isinstance(fields[k].get("enum"), list) and fields[k]["enum"]), None)
        required = list(spec.get("required") or [])
        text_field = next((k for k in sorted(fields, key=lambda k: (k not in required, "id" not in k.lower(), k))
                           if isinstance(fields[k], dict) and "string" in _kinds_of(fields[k]) and "enum" not in fields[k]), None)
        count_field = next((k for k in sorted(fields) if isinstance(fields[k], dict) and set(_kinds_of(fields[k])) & {"integer", "number"} and "enum" not in fields[k]), None)
        if verb:
            return "verb", {"verb": verb, "id": text_field, "count": count_field}
        return "set", {"id": text_field, "count": count_field}

    def verb_options(self, verb: str) -> tuple[list[str], list[str]]:
        tool = next((t["function"] for t in self.tools if t["function"]["name"] == self.roles.get("cart")), {})
        note = str(tool.get("description") or "") + " " + str(((self.cart_spec().get("properties") or {}).get(verb) or {}).get("description") or "")
        options = [str(v) for v in (self.cart_spec().get("properties") or {}).get(verb, {}).get("enum") or []]
        removing = [v for v in options if _DROP_CUE.search(v) or _REMOVE_CUE.search(_option_meaning(v, note, options))]
        return options, removing

    def add_word(self, verb: str) -> str:
        options, removing = self.verb_options(verb)
        if self.add_value in options:
            return self.add_value
        return next((v for v in options if _ADD_CUE.search(v)), None) or next((v for v in options if v not in removing), options[0] if options else "")

    def drop_word(self, verb: str) -> str | None:
        options, removing = self.verb_options(verb)
        adding = self.add_word(verb)
        return next((v for v in removing if v != adding), None)

    def cart_args(self, ident: str, qty: int) -> dict[str, Any] | None:
        tool = self.roles.get("cart")
        if not tool:
            return None
        shape, parts = self.cart_shape()
        if not parts.get("id"):
            return None
        if shape == "array":
            lines = dict(self.basket)
            if qty > 0:
                lines[ident] = qty
            else:
                lines.pop(ident, None)
            rows = []
            for item, count in lines.items():
                row: dict[str, Any] = {parts["id"]: item}
                if parts.get("count"):
                    row[parts["count"]] = count
                rows.append(row)
            args: dict[str, Any] = {parts["field"]: rows}
        elif shape == "verb":
            if qty > 0:
                args = {parts["verb"]: self.add_word(parts["verb"]), parts["id"]: ident}
                if parts.get("count"):
                    args[parts["count"]] = qty
            else:
                word = self.drop_word(parts["verb"])
                if not word:
                    return None
                args = {parts["verb"]: word, parts["id"]: ident}
        else:
            args = {parts["id"]: ident}
            if parts.get("count"):
                args[parts["count"]] = qty
            elif qty <= 0:
                return None
        args, problems = _gate(args, self.cart_spec(), "args")
        if problems:
            self.trace.append({"kind": "cart_args_refused", "problems": problems[:4]})
            return None
        return args

    def cart_effect(self, args: dict[str, Any]) -> dict[str, int] | None:
        shape, parts = self.cart_shape()
        if shape == "array":
            rows = args.get(parts.get("field") or "")
            if not isinstance(rows, list):
                return None
            out: dict[str, int] = {}
            for row in rows:
                if isinstance(row, dict) and isinstance(row.get(parts["id"]), str):
                    count = row.get(parts.get("count") or "", 1)
                    out[row[parts["id"]]] = int(count) if isinstance(count, (int, float)) and not isinstance(count, bool) else 1
            return out
        target = args.get(parts.get("id") or "")
        if not isinstance(target, str) or not target.strip():
            return None
        count = args.get(parts.get("count") or "", 1)
        count = int(count) if isinstance(count, (int, float)) and not isinstance(count, bool) else 1
        new = dict(self.basket)
        if shape == "verb":
            verb = args.get(parts["verb"])
            adding = verb == self.add_word(parts["verb"]) or (not _DROP_CUE.search(str(verb)) and verb != self.drop_word(parts["verb"]))
            if adding:
                new[target] = max(1, count)
            else:
                new.pop(target, None)
        elif count <= 0:
            new.pop(target, None)
        else:
            new[target] = count
        return new

    def read_basket(self, payload: Any) -> bool:
        if not isinstance(payload, dict):
            return False
        for value in payload.values():
            if isinstance(value, dict) and self.read_basket(value):
                return True
        for value in payload.values():
            if isinstance(value, list) and all(isinstance(x, dict) for x in value):
                lines: dict[str, int] = {}
                for row in value:
                    ident = self.lens.ident(row)
                    if not ident:
                        return False
                    price = self.lens.price_key
                    counts = [v for k, v in sorted(row.items()) if k != price and isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= 99]
                    lines[ident] = counts[0] if counts else 1
                self.basket = lines
                self.basket_seen = True
                return True
        return False

    def fits_text(self) -> str:
        chunks = []
        for pos in range(len(self.wants)):
            fits = self.proven(pos)
            line = ", ".join(f"{i} ({p:g})" if p is not None else i for i, p in fits[:5])
            if self.is_list():
                chunks.append(f"product {pos + 1}: " + (line or "none yet"))
            elif line:
                chunks.append(line)
        return " | ".join(chunks)

    def fits_line(self) -> str:
        line = self.fits_text()
        if line == self.fits_shown:
            return ""
        before, self.fits_shown = self.fits_shown, line
        if not line.replace("none yet", "").strip(_FIT_LINE_FILLER):
            return "\n[fits by the listings' own fields] none of the opened listings states every current requirement now." if before else ""
        text = "\n[fits by the listings' own fields] opened listings whose fields state every current requirement, within budget and not sold out, cheapest first: " + line + "."
        if not self.is_list():
            fits = self.proven(0)
            if len(fits) == 1 and self.order_attempts == 0:
                text += " Just one so far: find a second as the backup in case this one sells out or its price moves."
            if len(fits) > 1 and self.hints() and self.favourite(fits, 0) != fits[0][0]:
                text += f" {self.favourite(fits, 0)} matches the story's hint: prefer it unless the shopper's words or a cheapest rule say otherwise."
        return text

    def status(self) -> str:
        text = self.status_base()
        if self.reserved():
            text += "\n- one message is kept for what the request requires before the order: " + self.duty
        if self.closing():
            text += "\n- this task's spending limit is nearly reached: finish now (fix the cart only if needed, then order, or end without buying as the request says)."
        return text

    def status_base(self) -> str:
        left = self.steps_left()
        parts = [f"[status] step {self.step} of {self.max_steps}, {left} left"]
        if self.message_budget is not None:
            parts.append(f"messages used {self.messages_used} of {self.message_budget}")
        else:
            parts.append(f"messages used {self.messages_used}")
        text = "; ".join(parts) + "."
        text += "\n[requirements now] " + self.ledger()
        hints = self.hints()
        if hints:
            text += "\n[story hints] " + "; ".join(f"{p} (likely {v})" if v else p for p, v in hints) + ": ask about these first and prefer the fit that matches; a hint never rules a listing out."
        if self.basket_seen or self.cart_writes:
            text += "\n[cart] " + (", ".join(f"{i} x{q}" for i, q in self.basket.items()) or "empty")
        if self.plan.get("priority") == "cheapest" and self.rows and self.lens.price_key is None:
            text += "\n[prices] the code could not find a price field in this store's rows: compare prices yourself from the results."
        text += self.fits_line()
        reserve = 14 + 4 * (len(self.wants) - 1)
        if left <= 2:
            text += " LAST STEPS: order now if the cart holds a fit; end without buying only when nothing fits at all."
        elif left <= 5 or self.time_left() < CLOSING_SECONDS:
            text += " FINISH NOW: set the cart to the planned lines, re-check stock, then order (or end without buying if nothing fits)."
        elif self.order_attempts == 0 and left <= reserve:
            text += " Steps are running out: within the next few steps put the best verified fit" + (" for each product" if self.is_list() else "") + " in the cart and order; searching and reading details stop at 7 steps left, and repairing a refused order takes about 4 steps."
        return text

    def note_fixed(self, item: str, adding: bool, pos: int) -> str:
        if item not in self.sheets:
            if adding and self.sheets and self.roles.get("details") and self.steps_left() > 8:
                return f"The details of {item} have not been opened yet: the store judges a listing by its own fields, which only its details show. Open them first."
            return ""
        misses = []
        for prop, value in self.demands(pos):
            verdict, shown = self.reading(item, prop, value, pos)
            if verdict == "no":
                misses.append(f"{prop}: {value} (its own field says {shown})")
        if not misses:
            return ""
        return f"Read literally, {item}'s own fields do not state: " + "; ".join(misses[:4]) + ". The store reads a requirement from the listing's own field for that property, not from the title or a similar-looking word; if the field means the same thing in other words, it still counts."

    def cheaper_hint(self, item: str, pos: int) -> str:
        if self.plan.get("priority") != "cheapest":
            return ""
        price = self.price_of(item)
        if price is None:
            return ""
        notes = []
        lower = [f"{i} ({p:g})" for i, p in self.proven(pos) if i != item and p is not None and p < price]
        if lower and self.steps_left() > 4:
            notes.append(f"these opened listings state every current requirement and cost less than {item} ({price:g}): " + ", ".join(lower[:4]))
        target = self.shelf_of(item)
        unseen = sorted((self.price_of(i), i) for i, row in self.rows.items() if i not in self.sheets and target and self.lens.shelf(row) == target
                        and self.price_of(i) is not None and self.price_of(i) < price and self.affordable(i, pos))
        if unseen and self.steps_left() > 9:
            notes.append("cheaper listings in the same category appeared in results but were never opened: " + ", ".join(f"{i} ({p:g})" for p, i in unseen[:5]) + "; open them from the cheapest up before ordering")
        if not notes:
            return ""
        return "The request asks for the cheapest fitting choice: " + "; ".join(notes) + ". Order the cheapest one that fits."

    def over_budget(self, lines: dict[str, int], verb: str = "costs") -> str:
        cap = self.budget_now
        if not isinstance(cap, (int, float)) or isinstance(cap, bool) or cap <= 0 or not lines:
            return ""
        total = 0.0
        for ident, count in lines.items():
            price = self.price_of(ident)
            if price is None:
                return ""
            total += price * count
        if total <= cap + 1e-9:
            return ""
        scale = [p for p in (self.lens.price(row) for row in self.rows.values()) if p is not None]
        if not scale or min(scale) > cap:
            return ""
        shown = ", ".join(f"{i} at {self.price_of(i):g} x{q}" for i, q in lines.items())
        return f"the cart {verb} {total:g} in all ({shown}), over the current budget of {cap:g}"

    def swap_actions(self, old: str, new: str, qty: int) -> list[dict[str, Any]] | None:
        tool = self.roles.get("cart")
        if not tool:
            return None
        shape, parts = self.cart_shape()
        if shape == "array":
            held = dict(self.basket)
            self.basket = {k: v for k, v in held.items() if k != old}
            args = self.cart_args(new, qty)
            self.basket = held
            return [{"name": tool, "args": args}] if args is not None else None
        first, second = self.cart_args(old, 0), self.cart_args(new, qty)
        if first is None or second is None:
            return None
        return [{"name": tool, "args": first}, {"name": tool, "args": second}]

    def exact_fit(self, ident: str, pos: int) -> bool:
        wanted = self.demands(pos)
        pairs, _ = self.fact_pairs(ident)
        return bool(wanted) and all(_states_exactly(pairs, p, v, self.product_words(pos)) for p, v in wanted)

    def bind_cheaper(self) -> str:
        if self.steps_left() <= 6 or self.time_left() < CLOSING_SECONDS + 20:
            return ""
        for ident, count in list(self.basket.items()):
            pos = self.owner(ident)
            price = self.price_of(ident)
            key = f"bind|{pos}|{self.version}"
            if pos is None or price is None or key in self.bounced:
                continue
            shelf = self.shelf_of(ident).lower()
            cheaper = [(i, p) for i, p in self.proven(pos) if i not in self.basket and i not in self.taken_out and p is not None and p < price - 1e-9
                       and self.stock_of(i) is not False and not self.verdicts.get(f"{i}|{self.version}")
                       and (not shelf or self.shelf_of(i).lower() in ("", shelf)) and self.exact_fit(i, pos)]
            if not cheaper:
                continue
            self.bounced.add(key)
            best, low = cheaper[0]
            if self.check_item(best, "SWAP INTO THE CART IN PLACE OF " + ident, pos) or f"{best}|{self.version}" not in self.affirmed:
                continue
            actions = self.swap_actions(ident, best, self.wants[pos]["qty"])
            if not actions or self.steps_left() < len(actions) + 3:
                continue
            before = self.step
            reply = self.act(actions)
            self.trace.append({"kind": "bind_cheaper", "step": before + 1, "out": ident, "in": best, "reply": reply})
            if reply is None:
                return ""
            observations = reply.get("observations") if isinstance(reply.get("observations"), list) else [reply.get("observation")]
            texts = [self.digest(action["name"], action["args"], observation, False)[:600] for action, observation in zip(actions, observations)]
            used = self.step - before
            lead = (f"The order was not sent; the code used {used} step" + ("" if used == 1 else "s") + " on the cart instead. The request asks for the cheapest fitting choice, and "
                    + f"{best} ({low:g}) is an opened listing on the same shelf whose own fields state exactly every current requirement for less than {ident} ({price:g}), and a separate check agrees. ")
            if best in self.basket and ident not in self.basket:
                return lead + "The code swapped it into the cart in place of " + ident + ": " + " | ".join(texts) + ". Re-check it, then order."
            holds = ", ".join(f"{i} x{q}" for i, q in sorted(self.basket.items())) or "nothing"
            return lead + "The code tried to swap it in for " + ident + ": " + " | ".join(texts) + ". The cart now holds " + holds + ". Fix the cart, then order."
        return ""

    def basket_issue(self) -> str:
        if not (self.basket_seen or self.cart_writes):
            return ""
        if not self.basket:
            return "The cart is empty: put the chosen listing" + ("s" if self.is_list() else "") + " in it before ordering."
        problems = []
        if len(self.basket) > len(self.wants):
            problems.append(f"the cart has {len(self.basket)} lines but the request wants {len(self.wants)} product" + ("s" if self.is_list() else ""))
        claimed: dict[int, list[str]] = {}
        for ident, count in self.basket.items():
            pos = self.owner(ident)
            if pos is None:
                continue
            claimed.setdefault(pos, []).append(ident)
            if count != self.wants[pos]["qty"]:
                problems.append(f"{ident} is in the cart {count} times but the request wants {self.wants[pos]['qty']}")
        if self.is_list():
            missing = [str(pos + 1) for pos in range(len(self.wants)) if pos not in claimed]
            if missing and len(self.basket) < len(self.wants):
                problems.append("no cart line yet for product " + ", ".join(missing))
            doubled = [str(pos + 1) for pos, idents in sorted(claimed.items()) if len(idents) > 1]
            if doubled:
                problems.append("more than one line for product " + ", ".join(doubled))
        return ("Cart check: " + "; ".join(problems) + ".") if problems else ""

    def order_hint(self) -> str:
        key = f"order|{self.version}|{_flat(sorted(self.basket.items()))}"
        if key in self.bounced:
            return ""
        self.bounced.add(key)
        issue = self.basket_issue()
        if issue:
            return issue
        for ident in list(self.basket):
            pos = self.owner(ident)
            if pos is None:
                continue
            note = self.note_fixed(ident, False, pos) or self.cheaper_hint(ident, pos)
            if note:
                return note
        return ""

    def cart_stands(self) -> bool:
        if not self.basket or self.basket_issue():
            return False
        if self.gone_mark is not None and self.basket == self.gone_mark:
            return False
        owners = [self.owner(ident) for ident in self.basket]
        if None in owners or len(set(owners)) < len(self.wants):
            return False
        if self.over_budget(self.basket):
            return False
        for ident, pos in zip(self.basket, owners):
            if pos is None or not self.affordable(ident, pos) or self.verdicts.get(f"{ident}|{self.version}"):
                return False
            for prop, value in self.demands(pos):
                hard = _limits(value) is not None or bool(_EXCLUDED.match(value))
                if hard and self.reading(ident, prop, value, pos)[0] == "no":
                    return False
        return True

    def orderable(self, pos: int) -> list[tuple[str, float | None, int, bool]]:
        demanded = self.demands(pos)
        wanted = [(p, v) for p, v in demanded if len(_tokens(v)) <= 8]
        described = len(wanted) < len(demanded)
        out = []
        for ident in sorted(self.sheets):
            if not self.belongs(ident, pos) or not self.affordable(ident, pos):
                continue
            marks = [self.reading(ident, p, v, pos)[0] for p, v in wanted]
            if "no" in marks:
                continue
            if any(m != "yes" and (_limits(v) is not None or _EXCLUDED.match(v)) for m, (_, v) in zip(marks, wanted)):
                continue
            pairs, _ = self.fact_pairs(ident)
            spare = self.product_words(pos)
            if any(_limits(v) is not None and not any(_covers(k, p, spare) for k, _ in pairs) for p, v in wanted):
                continue
            out.append((ident, self.price_of(ident), marks.count("yes"), not described and all(m == "yes" for m in marks)))
        return out

    def proof_basket(self) -> dict[int, str] | None:
        cheap = self.plan.get("priority") == "cheapest"
        picks: dict[int, str] = {}
        for pos in range(len(self.wants)):
            rows = self.orderable(pos)
            if cheap:
                rows.sort(key=lambda r: (r[1] is None, r[1] or 0.0, not r[3], r[0]))
            else:
                rows.sort(key=lambda r: (r[0] not in self.basket, not r[3], -r[2], r[1] is None, r[1] or 0.0, r[0]))
            chosen = None
            checks = 0
            for ident, _, _, stated in rows:
                if stated or f"{ident}|{self.version}" in self.affirmed:
                    chosen = ident
                    break
                if checks < 2 and not self.verdicts.get(f"{ident}|{self.version}"):
                    checks += 1
                    self.check_item(ident, "ORDER", pos)
                    if f"{ident}|{self.version}" in self.affirmed:
                        chosen = ident
                        break
            if chosen is None:
                return None
            picks[pos] = chosen
        lines = {ident: self.wants[pos]["qty"] for pos, ident in picks.items()}
        if len(lines) < len(self.wants) or self.over_budget(lines):
            return None
        return picks

    def drop_action(self, ident: str) -> dict[str, Any] | None:
        args = self.cart_args(ident, 0)
        if args is not None:
            return {"name": self.roles["cart"], "args": args}
        remover = self.roles.get("remove")
        fields = (self.schemas.get(remover or "") or {}).get("properties") or {}
        key = next((k for k, v in sorted(fields.items()) if isinstance(v, dict) and "string" in _kinds_of(v)), None)
        if not remover or not key:
            return None
        args, problems = _gate({key: ident}, self.schemas[remover], "args")
        return None if problems else {"name": remover, "args": args}

    def basket_actions(self, want: dict[str, int]) -> list[dict[str, Any]] | None:
        tool = self.roles.get("cart")
        shape, _ = self.cart_shape()
        if want == self.basket:
            return []
        if shape == "array":
            held = dict(self.basket)
            self.basket = dict(want)
            last = sorted(want)[-1]
            args = self.cart_args(last, want[last])
            self.basket = held
            return None if args is None else [{"name": tool, "args": args}]
        out = []
        for ident in sorted(self.basket):
            if ident not in want or (shape == "verb" and self.basket[ident] != want[ident]):
                action = self.drop_action(ident)
                if action is None:
                    return None
                out.append(action)
        for ident in sorted(want):
            if self.basket.get(ident) == want[ident]:
                continue
            args = self.cart_args(ident, want[ident])
            if args is None:
                return None
            out.append({"name": tool, "args": args})
        return out

    def proof_order(self) -> tuple[str, dict[str, Any] | None] | None:
        key = f"proof|{self.version}"
        order, cart = self.roles.get("order"), self.roles.get("cart")
        if key in self.bounced or not order or not cart or self.finished or self.time_left() < LAST_SECONDS + 15:
            return None
        args = self.neutral_args(order, "Order the cart.", "order")
        if not (order != self.roles.get("decline") or (bool(self.finish_args.get("order")) and self.intent(order, args) == "order")):
            return None
        picks = self.proof_basket()
        if picks is None:
            return None
        want = {ident: self.wants[pos]["qty"] for pos, ident in picks.items()}
        actions = self.basket_actions(want)
        if actions is None or self.steps_left() < len(actions) + 2:
            return None
        self.bounced.add(key)
        shown = ", ".join(f"{ident} x{qty}" + (f" ({self.price_of(ident):g})" if self.price_of(ident) is not None else "") for ident, qty in sorted(want.items()))
        lead = ("Your ending without a purchase was not sent. Every product has an opened listing that is in stock, fits the budget, "
                "and whose own fields state each current requirement (or a separate check confirmed the ones they leave open): " + shown + ". ")
        if actions:
            before = self.step
            reply = self.act(actions)
            self.trace.append({"kind": "proof_cart", "step": before + 1, "actions": actions, "reply": reply})
            if reply is None:
                return lead + "Setting the cart failed (network or server); check the cart.", None
            observations = reply.get("observations") if isinstance(reply.get("observations"), list) else [reply.get("observation")]
            texts = [self.digest(action["name"], action["args"], observation, False)[:500] for action, observation in zip(actions, observations)]
            lead += "The code set the cart: " + " | ".join(texts) + ". "
            if self.basket != want or self.finished:
                return lead + "The cart does not match the plan yet: fix it, then order.", None
        return lead + "The order follows.", args

    def instead_of_decline(self) -> tuple[str, dict[str, Any], str] | None:
        late = self.decline_checked or f"decline|{self.version}" in self.bounced or self.steps_left() <= 6
        order, cart = self.roles.get("order"), self.roles.get("cart")
        if not late or not order or self.swaps >= 2 or self.steps_left() < 1 or self.time_left() < LAST_SECONDS:
            return None
        args = self.neutral_args(order, "Order the cart.", "order")
        trusted = order != self.roles.get("decline") or (bool(self.finish_args.get("order")) and self.intent(order, args) == "order")
        if trusted and self.cart_stands():
            self.swaps += 1
            return (order, args, "Your ending without a purchase was not sent: it scores 0 while a fitting listing exists, the cart already holds a line for every product, and no cart line fails a stated number, an exclusion, the budget or stock. The order was sent instead.")
        if self.steps_left() <= 1:
            return None
        picks: list[tuple[int, str, float | None, str]] = []
        rejected = {i for i in self.swapped_in if i not in self.basket}
        for pos in range(len(self.wants)):
            fits, basis = [f for f in self.proven(pos) if f[0] not in rejected], "whose own fields state every current requirement"
            if not fits:
                fits, basis = [f for f in self.plausible(pos) if f[0] not in rejected], "whose own fields contradict no current requirement (no opened listing states them all, and ending without a purchase scores nothing)"
            if fits:
                best = self.favourite(fits, pos)
                picks.append((pos, best, next((p for i, p in fits if i == best), None), basis))
        if not picks:
            return None
        covered = {pos for ident in self.basket for pos in [self.owner(ident)] if pos is not None}
        for pos, best, price, basis in picks:
            if pos in covered or not cart or self.steps_left() <= 2:
                continue
            args = self.cart_args(best, self.wants[pos]["qty"])
            if args is None:
                return None
            shown = f"{best} ({price:g})" if price is not None else best
            self.swaps += 1
            self.swapped_in.add(best)
            return (cart, args, f"Your ending without a purchase was not sent: it scores 0 while a fitting listing exists, and {shown} is an opened listing {basis}, within the budget and not shown as sold out. It was put in the cart instead: check the cart holds only the planned lines, re-check stock and price, and order.")
        return None

    def held_note(self) -> str:
        if not self.basket or self.steps_left() <= 2 or self.time_left() < LAST_SECONDS + 10:
            return ""
        key = f"held|{self.version}|{_flat(sorted(self.basket.items()))}"
        if key in self.bounced:
            return ""
        self.bounced.add(key)
        lines = ", ".join(f"{i} x{q}" for i, q in sorted(self.basket.items()))
        return f"The cart still holds {lines}, and nothing has been bought. Ending without a purchase scores 0 whenever those lines meet the request, so if they do, order the cart. End without buying only if a cart line clearly fails a stated requirement, a disclosed preference, the budget or stock."

    def decline_hint(self) -> str:
        key = f"decline|{self.version}"
        if key in self.bounced or self.steps_left() <= 3 or self.time_left() < CLOSING_SECONDS:
            return ""
        notes = []
        for pos in range(len(self.wants)):
            label = f"product {pos + 1}: " if self.is_list() else ""
            fits = self.proven(pos)
            if fits:
                notes.append(label + "these opened listings state every current requirement within the budget and are not shown as sold out: " + ", ".join(f"{i} ({p:g})" if p is not None else i for i, p in fits[:5]) + "; re-read the best one, cart it and order it")
                continue
            if self.steps_left() > 6:
                waiting = self.unread(pos)
                if waiting:
                    notes.append(label + "these listings from the results were never opened, and an unopened listing is unknown rather than failing: " + ", ".join(waiting) + "; open them, cheapest first, since their fields decide")
        if self.steps_left() > 6:
            unseen = [f"'{c['label']}' ({c['total']} matched, {len(c['seen'])} seen)" for _, c in sorted(self.coverage.items()) if c["label"] and len(c["seen"]) < c["total"] <= 60][:3]
            if unseen:
                notes.append("these searches have matches you have not seen: " + ", ".join(unseen) + "; go through their further pages")
        if not notes:
            return ""
        self.bounced.add(key)
        return "Before ending without a purchase: " + " | ".join(notes) + ". End without buying only if every one of them fails."

    def page_hint(self, name: str, args: dict[str, Any], payload: Any) -> str:
        if name != self.roles.get("search") or not isinstance(payload, dict):
            return ""
        total = next((v for k, v in sorted(payload.items()) if isinstance(v, int) and not isinstance(v, bool) and _TOTAL_CUE.search(str(k))), None)
        if total is None:
            return ""
        props = (self.schemas.get(name) or {}).get("properties") or {}
        steady = {k: v for k, v in sorted(args.items()) if not isinstance(v, (int, float)) and not _CURSOR_CUE.search(str(k))}
        words = [str(v) for k, v in steady.items() if isinstance(v, str) and not (props.get(k) or {}).get("enum")]
        entry = self.coverage.setdefault(name + _flat(steady), {"label": " ".join(words), "total": total, "seen": set()})
        entry["total"] = total
        rows = next((v for v in payload.values() if isinstance(v, list) and v and all(isinstance(x, dict) for x in v)), [])
        for row in rows:
            ident = self.lens.ident(row)
            if ident:
                entry["seen"].add(ident)
        remaining = total - len(entry["seen"])
        if 0 < remaining and total <= 60:
            return f"This search matched {total} listings; {len(entry['seen'])} seen so far, {remaining} more on further pages."
        return ""

    def echo_hint(self, name: str, args: dict[str, Any], payload: Any) -> str:
        if name != self.roles.get("search") or not isinstance(payload, dict):
            return ""
        rows = next((v for v in payload.values() if isinstance(v, list) and v and all(isinstance(x, dict) for x in v)), [])
        idents = [self.lens.ident(row) or "" for row in rows]
        if len(idents) < 3 or not all(idents):
            return ""
        key = "|".join(idents)
        asked = _flat(dict(sorted(args.items())))
        earlier = self.result_sets.get(key)
        if earlier is None:
            self.result_sets[key] = asked
            return ""
        if earlier == asked:
            return ""
        return "These are exactly the listings an earlier search with different arguments returned (" + earlier[:160] + "), so the arguments that differ had no effect here; check the tool description for which arguments each search mode reads."

    def ask_light(self, prompt: str, max_tokens: int, strong: bool) -> dict[str, Any] | None:
        models = self.seats(strong)
        if not models:
            return None
        reply = self.call_model([{"role": "user", "content": prompt}], None, models, max_tokens)
        return _object_in((reply or {}).get("content") or "")

    def item_index(self, label: Any, fallback: int | None = None) -> int:
        if not self.is_list():
            return 0
        words = {_base(w) for w in _tokens(label) if len(w) > 2}
        scored = [(sum(1 for w in words if any(_akin(w, x) for x in self.product_words(pos))), -pos) for pos in range(len(self.wants))]
        best = max(range(len(self.wants)), key=lambda pos: scored[pos])
        if scored[best][0] > 0:
            return best
        if fallback is not None:
            return fallback
        return self.asks[-1]["pos"] if self.asks else 0

    def interpret(self, lines: list[str], question: str | None) -> str:
        fresh = [line for line in lines if line not in self.heard]
        if not fresh:
            if question is not None:
                self.last_understood = all(self.line_clear.get(line, True) for line in lines)
            return ""
        self.heard.update(fresh)
        if self.time_left() < CLOSING_SECONDS:
            return ""
        prompt = READ_SHOPPER + "\n\nTHE SHOPPER'S REQUEST (context):\n" + str(self.policy.get("query") or "")
        if self.is_list():
            prompt += "\n\nPRODUCTS ON THE LIST: " + "; ".join(f"{pos + 1}. {w['product']}" for pos, w in enumerate(self.wants))
        prompt += "\n\nREQUIREMENTS SO FAR:\n" + self.ledger()
        if self.said:
            prompt += "\n\nEARLIER EXCHANGES (oldest first):\n" + "\n".join(self.said[-12:])
        if question:
            prompt += "\n\nTHE ASSISTANT ASKED:\n" + question
        prompt += "\n\nTHE SHOPPER NOW SAID:\n" + "\n".join(fresh)
        self.said.append((f"asked: {question} -> " if question else "") + "shopper: " + " / ".join(fresh))
        result = self.ask_light(prompt, 1500, False)
        if not result:
            return ""
        clear = result.get("understood") is not False
        if question is not None:
            for line in fresh:
                self.line_clear[line] = clear
            self.last_understood = clear
            if not clear:
                return "The reply took no position on that property: treat it as no preference and move on to a different property."
        if question is not None and self.free and not self.is_list():
            asked = {_base(w) for w in _tokens(question)}
            declined = [set(_label_words(f.partition(":")[0])) for f in self.free]
            held = set(self.needs_of(0))

            def aside(entry: Any) -> bool:
                need = self.loose_need(entry)[0]
                words = set(_label_words(need.partition(":")[0]))
                return bool(words) and need not in held and words in declined and not words <= asked
            result["needs"] = [e for e in result.get("needs") or [] if not aside(e)]
            if isinstance(result.get("requirements_now"), list):
                result["requirements_now"] = [e for e in result["requirements_now"] if not aside(e)]
        notes = []
        now = result.get("requirements_now")
        if isinstance(now, list) and now:
            grouped: dict[int, list[str]] = {pos: [] for pos in range(len(self.wants))}
            usable = True
            for entry in now:
                if isinstance(entry, dict) and isinstance(entry.get("need"), str) and entry["need"].strip():
                    grouped[self.item_index(entry.get("for"))].append(entry["need"].strip())
                elif isinstance(entry, str) and entry.strip():
                    grouped[self.item_index("")].append(entry.strip())
                else:
                    usable = False
            if usable:
                for pos, needs in grouped.items():
                    self.wants[pos]["now"] = needs
        for entry in result.get("dropped") or []:
            drop, owner = self.loose_need(entry)
            if drop:
                for pos in self.drop_targets(drop, owner):
                    if self.forget_need(pos, drop):
                        notes.append(f"dropped: {drop}" + (f" (product {pos + 1})" if self.is_list() else ""))
        for entry in result.get("needs") or []:
            need, owner = self.loose_need(entry)
            if need:
                pos = self.item_index(owner or "")
                want = self.wants[pos]
                if want["now"] is not None and need not in want["now"]:
                    want["now"].append(need)
                if need not in want["revealed"]:
                    want["revealed"].append(need)
                    notes.append(f"new requirement: {need}" + (f" (product {pos + 1})" if self.is_list() else ""))
                    stating = [i for i in sorted(self.sheets) if self.belongs(i, pos) and ":" in need and self.reading(i, need.partition(":")[0].strip().lower(), need.partition(":")[2].strip().lower(), pos)[0] == "yes"]
                    if stating:
                        notes.append(f"opened listings that state {need.strip()}: " + ", ".join(stating[:6]))
                    elif self.sheets:
                        notes.append(f"no opened listing states {need.strip()} yet: search for it")
        for entry in result.get("no_preference") or []:
            prop, _ = self.loose_need(entry)
            if prop and prop not in self.free:
                self.free.append(prop)
        budget = result.get("new_budget")
        if isinstance(budget, (int, float)) and not isinstance(budget, bool) and budget > 0 and budget != self.budget_now:
            self.budget_now = budget
            notes.append(f"new budget: {budget}")
        if result.get("unavailable"):
            notes.append("an item became unavailable; take it out of the cart and do not order it")
            self.gone_mark = dict(self.basket)
        moved = bool(result.get("dropped")) or (isinstance(budget, (int, float)) and not isinstance(budget, bool) and budget > 0)
        if moved and self.plan.get("priority") == "cheapest":
            notes.append("the set of fitting items has changed: search again (items excluded before may fit now) and re-pick the cheapest fit before ordering")
        if notes:
            self.version += 1
        reply = str(result.get("reply") or "").strip()
        if reply and self.roles.get("message") and any("?" in line for line in fresh):
            self.pending_reply = reply
        return "Meaning: " + str(result.get("meaning") or "") + (" (" + "; ".join(notes) + ")" if notes else "")

    def loose_need(self, entry: Any) -> tuple[str, str | None]:
        if isinstance(entry, str):
            return entry.strip(), None
        if isinstance(entry, dict):
            text = next((entry.get(k) for k in ("need", "property", "condition", "value") if isinstance(entry.get(k), str) and entry.get(k).strip()), "")
            owner = entry.get("for") if isinstance(entry.get("for"), str) else None
            return str(text).strip(), owner
        return "", None

    def same_need(self, need: str, drop: str) -> bool:
        have_prop, _, have_value = need.partition(":")
        drop_prop, _, drop_value = drop.partition(":")
        have_words = set(_label_words(have_prop)) - {"type"}
        drop_words = set(_label_words(drop_prop)) - {"type"}
        if not have_words or not drop_words or not (have_words <= drop_words or drop_words <= have_words):
            return False
        drop_value, have_value = drop_value.strip().lower(), have_value.strip().lower()
        return not drop_value or _says(drop_value, have_value) or _says(have_value, drop_value)

    def drop_targets(self, drop: str, owner: str | None) -> list[int]:
        if not self.is_list():
            return [0]
        holders = [pos for pos in range(len(self.wants)) if any(self.same_need(n, drop) for n in self.needs_of(pos))]
        if owner:
            named = self.item_index(owner, -1)
            if named >= 0:
                return [named]
        return holders if holders else []

    def forget_need(self, pos: int, drop: str) -> bool:
        want = self.wants[pos]
        before = self.needs_of(pos)
        for key in ("stated", "revealed"):
            want[key] = [n for n in want[key] if not self.same_need(n, drop)]
        if want["now"] is not None:
            want["now"] = [n for n in want["now"] if not self.same_need(n, drop)]
        if drop not in want["dropped"]:
            want["dropped"].append(drop)
        return self.needs_of(pos) != before

    def records_about(self, item: str) -> list[str]:
        found: list[str] = []
        for record in self.records:
            self._collect(record, item, found, 0)
        return found[-4:]

    def _collect(self, node: Any, item: str, found: list[str], level: int) -> None:
        if level > 6 or len(found) >= 8:
            return
        if isinstance(node, dict):
            if any(isinstance(v, str) and v == item for v in node.values()):
                text = _flat(_shorten(node, 400))[:2500]
                if text not in found:
                    found.append(text)
                return
            for part in node.values():
                self._collect(part, item, found, level + 1)
        elif isinstance(node, list):
            for part in node:
                self._collect(part, item, found, level + 1)

    def check_item(self, item: str, purpose: str, pos: int | None) -> str:
        if self.time_left() < CLOSING_SECONDS + 20:
            return ""
        key = f"{item}|{self.version}"
        if key in self.verdicts:
            return self.verdicts[key]
        records = self.records_about(item)
        if not records:
            return ""
        prompt = CHECK_PICK + "\n\nTHE SHOPPER'S REQUEST:\n" + str(self.policy.get("query") or "")
        if pos is not None and self.is_list():
            prompt += f"\n\nTHIS LISTING IS FOR PRODUCT {pos + 1} ({self.wants[pos]['product']}, quantity {self.wants[pos]['qty']}). ITS REQUIREMENTS: " + ("; ".join(self.needs_of(pos)) or "only the product kind")
            if self.budget_now is not None:
                prompt += f"; the whole list must cost at most {self.budget_now} in total"
        else:
            prompt += "\n\nCURRENT REQUIREMENTS (stated, revealed and changed):\n" + self.ledger()
        prompt += f"\n\nTHE LISTING TO {purpose}: {item}\nRECORDS OF THIS LISTING SEEN IN THE STORE (oldest first):\n" + "\n".join(records)
        labels = self.label_census()
        if labels:
            prompt += "\n\nLABELS THAT RECUR ON THIS STORE'S LISTINGS (with counts; the listing's category label must be the one that matches the request's wording as a whole, not a neighbouring one):\n" + labels
        result = self.ask_light(prompt, 1200, False)
        self.trace.append({"kind": "verify", "item": item, "result": result})
        if result and result.get("fits") is True:
            self.affirmed.add(key)
        if not result or result.get("fits") is not False:
            self.verdicts[key] = ""
            return ""
        problems = [str(p) for p in result.get("problems") or []][:5]
        verdict = f"A separate check says {item} does NOT meet every requirement: " + "; ".join(problems) + ". Unless you are sure the check is wrong, choose another verified fit."
        self.verdicts[key] = verdict
        return verdict

    def label_census(self) -> str:
        tally: dict[str, dict[str, int]] = {}
        for ident, row in sorted(self.rows.items()):
            for key, value in sorted(row.items()):
                if isinstance(value, str) and 0 < len(value) <= 40 and not _idlike(value):
                    bucket = tally.setdefault(str(key), {})
                    bucket[value] = bucket.get(value, 0) + 1
        lines = []
        for key in sorted(tally):
            bucket = tally[key]
            if sum(bucket.values()) >= 4 and 2 <= len(bucket) <= 8 and max(bucket.values()) >= 2:
                lines.append(f"- {key}: " + " | ".join(f"{v} (x{n})" for v, n in sorted(bucket.items(), key=lambda kv: (-kv[1], kv[0]))))
        return "\n".join(lines[:12])

    def tools_now(self) -> list[dict[str, Any]]:
        return self.tools

    def allowed_names(self) -> set[str]:
        left = self.steps_left()
        order, decline, cart = self.roles.get("order"), self.roles.get("decline"), self.roles.get("cart")
        names: set[Any] | None = None
        if left <= 1:
            names = {order, decline}
        elif left <= 3:
            names = {order, decline, cart}
        elif (left <= 7 and (self.order_attempts == 0 or self.refusals > 0)) or self.closing():
            names = {order, decline, cart, self.roles.get("cart_view"), self.roles.get("view")}
        allowed = {t["function"]["name"] for t in self.tools}
        if names is not None and order and cart:
            allowed = (allowed & names) or allowed
        message = self.roles.get("message")
        if message and (self.duty_phase or (self.basket and self.messages_left() > 0)) and self.duty_owed() and left > 1:
            allowed.add(message)
        elif message and self.planner_on:
            allowed.discard(message)
        return allowed

    def duty_owed(self) -> bool:
        return bool(self.duty and self.roles.get("message")) and (not self.duty_done or self.duty_basket != dict(self.basket))

    def reserved(self) -> int:
        return 1 if self.duty_owed() else 0

    def messages_left(self) -> int:
        return (self.message_budget if self.message_budget is not None else 99) - self.messages_used

    def cart_summary(self) -> str:
        lines, total = [], 0.0
        for ident, qty in self.basket.items():
            each = self.price_of(ident)
            lines.append(f"{ident} x {qty}" + (f" at {each:g} each" if isinstance(each, (int, float)) else ""))
            total += (each or 0.0) * qty
        return "Before I place the order, here is the final cart: " + "; ".join(lines) + (f". Total: {total:.2f}." if total else ".")

    def trim_history(self) -> None:
        size = sum(len(m["content"]) for m in self.messages if isinstance(m.get("content"), str))
        if size <= PROMPT_CHARS:
            return
        for entry in self.messages[2:max(2, len(self.messages) - 12)]:
            text = entry.get("content")
            if entry.get("role") == "tool" and isinstance(text, str) and len(text) > 600:
                entry["content"] = text[:400] + " ... (the rest of this older result was removed to save space; call the tool again if you need it)"
        self.trace.append({"kind": "trimmed", "before": size})

    def asked_props(self) -> list[str]:
        return [a["prop"] for a in self.asks if a["clear"]]

    def property_menu(self, pos: int, strict: bool) -> tuple[list[str], int]:
        settled = [p for p, _ in self.demands(pos)] + [a["prop"] for a in self.asks if a["pos"] == pos] + list(self.free)
        values: dict[str, set[str]] = {}
        carriers: dict[str, int] = {}
        opened = 0
        for ident in sorted(self.sheets):
            if not self.belongs(ident, pos):
                continue
            opened += 1
            if strict and any(self.reading(ident, p, v, pos)[0] == "no" for p, v in self.demands(pos)):
                continue
            facts = _facts_of(self.sheets[ident]) or {}
            for key, value in sorted(facts.items()):
                text = str(value).strip().lower()
                if not text or _BLANK.fullmatch(text) or len(text) > 70:
                    continue
                label = " ".join(_tokens(key))
                values.setdefault(label, set()).add(text)
                carriers[label] = carriers.get(label, 0) + 1
        ranked = []
        for label in sorted(values):
            words = set(_label_words(label)) - {"type"}
            if not words or any(words <= set(_label_words(t)) for t in settled if t):
                continue
            ranked.append((-(carriers[label] * min(len(values[label]), 4)), label))
        ranked.sort()
        lines = [f"- {label} (in {carriers[label]} listings): " + " | ".join(sorted(values[label])[:5]) for _, label in ranked[:10]]
        return lines, opened

    def splitting(self, wide: bool = True) -> tuple[str, int, int]:
        counts = [sum(1 for a in self.asks if a["pos"] == p) for p in range(len(self.wants))]
        order = sorted(range(len(self.wants)), key=lambda p: (counts[p], p))
        if self.unclear is not None:
            order = [self.unclear["pos"]] + [p for p in order if p != self.unclear["pos"]]
        first_opened = 0
        for rank, pos in enumerate(order if wide else order[:1]):
            lines, opened = self.property_menu(pos, True)
            if not lines:
                lines, opened = self.property_menu(pos, False)
            if rank == 0:
                first_opened = opened
            if lines:
                return "\n".join(lines), opened, pos
        return "", first_opened, order[0]

    def ask_planned(self, force: bool = False) -> bool:
        tool = self.roles.get("message")
        limit = (self.message_budget if self.message_budget is not None else 3) - self.reserved()
        if not tool or self.messages_used >= limit:
            self.planner_on = False
            return False
        if self.steps_left() <= 4 + 2 * len(self.wants) or self.time_left() < CLOSING_SECONDS + 40:
            self.planner_on = False
            return False
        if not force and (self.planner_quiet or self.steps_left() <= 8 + 3 * len(self.wants)):
            return False
        if self.ask_pause and not force:
            self.ask_pause = False
            return False
        menu, opened, focus = self.splitting(force or self.step >= self.max_steps // 2)
        urgent = bool(self.plan.get("ask_first")) and not self.asks
        if not (force or urgent or self.unclear is not None):
            if opened < 2 and self.step < min(10, self.max_steps // 3):
                return False
            if not menu and self.step < self.max_steps // 2:
                return False
        prompt = ASK_PLAN + "\n\nTHE SHOPPER'S REQUEST:\n" + str(self.policy.get("query") or "")
        if self.is_list():
            prompt += "\n\nPRODUCTS ON THE LIST: " + "; ".join(f"{p + 1}. {w['product']}" for p, w in enumerate(self.wants))
        prompt += "\n\nREQUIREMENTS SO FAR:\n" + self.ledger()
        if self.hints():
            prompt += "\n\nSTORY HINTS (properties the request points at without a value): " + "; ".join(f"{p} (likely {v})" if v else p for p, v in self.hints())
        if self.asks:
            prompt += "\n\nALREADY ASKED:\n" + "\n".join(f"- {a['prop']} ({self.wants[a['pos']]['product']}): \"{a['question']}\" -> {a['answer']}" + ("" if a["clear"] else " (no position taken: no preference, do not ask again)") for a in self.asks)
        if self.unclear is not None:
            prompt += f"\n\nASK AGAIN: {self.unclear['prop']} for the {self.wants[self.unclear['pos']]['product']} (the shopper did not follow \"{self.unclear['question']}\")"
        if urgent:
            prompt += "\n\nMUST LEARN FIRST: " + "; ".join(str(h) for h in self.plan.get("ask_first") or [])
        prompt += f"\n\nCANDIDATE PROPERTIES for the {self.wants[focus]['product'] or 'product'} (from its opened listings; values seen):\n" + (menu or "(no listing opened yet)")
        result = self.ask_light(prompt, 900, False) or {}
        prop = str(result.get("property") or "").strip()
        question = str(result.get("question") or "").strip()
        if (not prop or not question) and force:
            spare = self.unasked_pick()
            if spare is not None:
                prop, question, focus = spare
                result = {"for": self.wants[focus]["product"]}
        if not prop or not question:
            self.planner_idle += 1
            if self.planner_idle >= 2:
                self.planner_quiet = True
            return False
        args = self.message_args(question)
        if not args:
            self.planner_on = False
            return False
        args, problems = _gate(args, self.schemas[tool], "args")
        if problems:
            self.planner_on = False
            return False
        pos = self.item_index(result.get("for"), focus)
        call = {"id": f"ask_{self.messages_used + 1}_{self.step}", "type": "function", "function": {"name": tool, "arguments": json.dumps(args)}}
        self.messages.append({"role": "assistant", "content": f"Asking the shopper about {prop}.", "tool_calls": [call]})
        version = self.version
        self.last_understood = True
        replies = self.run_calls([call], planned=True)
        answer = ""
        for call_id, text in replies:
            answer = text
            self.messages.append({"role": "tool", "tool_call_id": call_id, "content": text + "\n" + self.status()})
        clear = self.last_understood
        self.asks.append({"prop": prop, "pos": pos, "question": question, "answer": answer.split("\n[note]")[0][:240], "clear": clear})
        self.unclear = None
        self.planner_idle = 0
        if clear and self.version != version:
            self.ask_pause = True
        if self.messages_used >= limit:
            self.planner_on = False
        return True

    def unasked_pick(self) -> tuple[str, str, int] | None:
        counts = [sum(1 for a in self.asks if a["pos"] == p) for p in range(len(self.wants))]
        for pos in sorted(range(len(self.wants)), key=lambda p: (counts[p], p)):
            for strict in (True, False):
                lines, _ = self.property_menu(pos, strict)
                for line in lines:
                    label = line[2:].split(" (in ", 1)[0].strip()
                    if label:
                        product = self.wants[pos]["product"] or "item"
                        return label, f"Any preferred {label} for the {product}?", pos
        return None

    def evidence_note(self) -> str:
        tool = self.roles.get("message")
        limit = (self.message_budget if self.message_budget is not None else 3) - self.reserved()
        left = limit - self.messages_used
        if not tool or self.planner_on or left <= 0 or "evidence" in self.bounced:
            return ""
        if self.steps_left() <= 4 + 2 * len(self.wants) or self.time_left() < CLOSING_SECONDS + 40:
            return ""
        self.bounced.add("evidence")
        examples = []
        for pos in range(len(self.wants)):
            lines, _ = self.property_menu(pos, False)
            label = lines[0][2:].split(" (in ", 1)[0].strip() if lines else ""
            if label:
                examples.append(f"{label} for the {self.wants[pos]['product'] or 'item'}")
        hint = (" (for example: " + "; ".join(examples[:3]) + ")") if examples else ""
        return (f"{left} of the {limit} messages to the shopper are unused, and the shopper may hold preferences nobody has asked about yet{hint}. "
                "An order placed before those are heard can fail. Ask one short question per message, each about one property of one cart line and naming that product, then order.")

    def message_args(self, text: str) -> dict[str, Any] | None:
        spec = self.schemas.get(self.roles.get("message") or "") or {}
        fields = spec.get("properties") or {}
        required = list(spec.get("required") or [])
        texts = [k for k, v in fields.items() if isinstance(v, dict) and "string" in _kinds_of(v) and "enum" not in v]
        target = next((k for k in required if k in texts), None) or (texts[0] if len(texts) == 1 else None)
        if not target or any(k != target for k in required):
            return None
        return {target: text}

    def send_reply(self, text: str) -> bool:
        tool = self.roles.get("message")
        limit = (self.message_budget if self.message_budget is not None else 3) - self.reserved()
        if not tool or self.messages_used >= limit or self.steps_left() <= 4 or self.time_left() < CLOSING_SECONDS:
            return False
        args = self.message_args(text)
        if not args:
            return False
        args, problems = _gate(args, self.schemas[tool], "args")
        if problems:
            return False
        call = {"id": f"reply_{self.step}", "type": "function", "function": {"name": tool, "arguments": json.dumps(args)}}
        self.messages.append({"role": "assistant", "content": "Answering the shopper's question.", "tool_calls": [call]})
        replies = self.run_calls([call], planned=True)
        for pos, (call_id, reply_text) in enumerate(replies):
            if pos == len(replies) - 1:
                reply_text += "\n" + self.status()
            self.messages.append({"role": "tool", "tool_call_id": call_id, "content": reply_text})
        return True

    def quantity_issue(self, name: str, args: dict[str, Any]) -> str:
        shape, parts = self.cart_shape()
        if shape == "array":
            lines = self.cart_effect(args) or {}
            if len(lines) > len(self.wants):
                return f"this cart would hold {len(lines)} lines, but the request has {len(self.wants)} product" + ("s." if self.is_list() else ".")
            for ident, count in lines.items():
                pos = self.owner(ident)
                if pos is not None and count != self.wants[pos]["qty"]:
                    return f"{ident} would be in the cart {count} times, but the request wants {self.wants[pos]['qty']}."
            return ""
        if not parts.get("count") or not parts.get("id"):
            return ""
        if shape == "verb" and args.get(parts["verb"]) != self.add_word(parts["verb"]):
            return ""
        count = args.get(parts["count"], ((self.cart_spec().get("properties") or {}).get(parts["count"]) or {}).get("default", 1))
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            return ""
        target = args.get(parts["id"])
        pos = self.owner(target) if isinstance(target, str) else None
        allowed = {self.wants[pos]["qty"]} if pos is not None else {w["qty"] for w in self.wants}
        if count not in allowed:
            return f"the request wants {' or '.join(str(q) for q in sorted(allowed))} of this listing, but this call asks for {count}."
        return ""

    def cart_target(self, args: dict[str, Any]) -> list[str]:
        shape, parts = self.cart_shape()
        if shape == "array":
            before = set(self.basket)
            return [i for i in (self.cart_effect(args) or {}) if i not in before]
        if shape == "verb" and args.get(parts.get("verb") or "") != self.add_word(parts["verb"]):
            return []
        if shape == "set":
            count = args.get(parts.get("count") or "", 1)
            if isinstance(count, (int, float)) and not isinstance(count, bool) and count <= 0:
                return []
        target = args.get(parts.get("id") or "")
        return [target] if isinstance(target, str) and target.strip() and " " not in target.strip() else []

    def second_look(self) -> str:
        if self.audited or self.time_left() < CLOSING_SECONDS + 30:
            return ""
        self.audited = True
        blocks: list[str] = []
        size = 0
        for ident in reversed(list(self.sheets)):
            record = dict(self.sheets[ident])
            price = self.price_of(ident)
            if price is not None:
                record["_price_seen"] = price
            if self.stock_of(ident) is not None:
                record["_in_stock_seen"] = self.stock_of(ident)
            text = _flat(_shorten(record, 200))
            if size + len(text) > 14000:
                continue
            blocks.append(text)
            size += len(text)
        if not blocks:
            return ""
        prompt = SECOND_LOOK + "\n\nTHE SHOPPER'S REQUEST:\n" + str(self.policy.get("query") or "")
        prompt += "\n\nCURRENT REQUIREMENTS:\n" + self.ledger()
        prompt += "\n\nOPENED LISTINGS (newest first):\n" + "\n".join(blocks)
        result = self.ask_light(prompt, 1200, False)
        self.trace.append({"kind": "audit", "result": result})
        fits = [str(x) for x in (result or {}).get("possible_fits") or [] if isinstance(x, (str, int))][:5]
        if not fits:
            return ""
        reason = f" ({result.get('reason')})" if result and result.get("reason") else ""
        return "A separate check thinks these listings may meet every requirement: " + ", ".join(fits) + reason + ". Verify them (details, stock, price) and buy the right one; end without buying only if each one truly fails."

    def neutral_args(self, name: str, text: str, kind: str | None = None) -> dict[str, Any]:
        spec = self.schemas.get(name) or {}
        fields = spec.get("properties") or {}
        args: dict[str, Any] = dict(self.finish_args.get(kind) or {}) if kind else {}
        for key in spec.get("required") or []:
            if key in args:
                continue
            entry = fields.get(key) or {}
            kinds = _kinds_of(entry)
            if isinstance(entry.get("enum"), list) and entry["enum"]:
                args[key] = entry["enum"][0]
            elif "string" in kinds:
                args[key] = text[: entry.get("maxLength", len(text))]
            elif "boolean" in kinds:
                args[key] = True
            elif "integer" in kinds or "number" in kinds:
                args[key] = max(1, entry.get("minimum", 1))
        if len(fields) == 1:
            key, entry = next(iter(fields.items()))
            if key not in args and isinstance(entry, dict) and "string" in _kinds_of(entry) and "enum" not in entry:
                args[key] = text[: entry.get("maxLength", len(text))]
        return args

    def scrub_decline(self, args: dict[str, Any], name: str) -> dict[str, Any]:
        if self.plan.get("decline_no_names") is False:
            return args
        leaked = any(token and token in self.seen_ids for value in args.values() if isinstance(value, str) for token in re.split(r"[\s.,;:!?()\[\]{}<>\"'`]+", value))
        if not (self.plan.get("decline_no_names") or leaked):
            return args
        plain = "Sorry, nothing in the store meets all of your requirements, so I am not buying anything."
        fields = (self.schemas.get(name) or {}).get("properties") or {}
        out = {}
        for key, value in args.items():
            entry = fields.get(key) or {}
            out[key] = plain[: entry.get("maxLength", len(plain))] if isinstance(value, str) and "enum" not in entry else value
        return out

    def intent(self, name: str, args: dict[str, Any]) -> str | None:
        order, decline = self.roles.get("order"), self.roles.get("decline")
        if name not in (order, decline):
            return None
        if order != decline:
            return "order" if name == order else "decline"
        for kind in ("decline", "order"):
            wanted = self.finish_args.get(kind) or {}
            if wanted and all(args.get(k) == v for k, v in wanted.items()):
                return kind
        if self.finish_args:
            return None
        note = next((str(t["function"].get("description") or "") for t in self.tools if t["function"]["name"] == name), "")
        fields = (self.schemas.get(name) or {}).get("properties") or {}
        for key, entry in sorted(fields.items()):
            if isinstance(entry, dict) and entry.get("enum") and args.get(key) is not None:
                options = [str(o) for o in entry["enum"]]
                stops, buys = _finish_split(options, note)
                if len(stops) != 1:
                    return None
                value = str(args.get(key))
                return "decline" if value in stops else "order" if value in buys else None
        return "order"

    def force_finish(self) -> None:
        if self.finished or self.dead or self.steps_left() <= 0 or self.time_left() < 2:
            return
        order, decline, cart = self.roles.get("order"), self.roles.get("decline"), self.roles.get("cart")
        if order and cart and not self.basket and self.time_left() > 8:
            for pos in range(len(self.wants)):
                if self.steps_left() < 2:
                    break
                fits = self.proven(pos) or self.plausible(pos)
                if not fits:
                    continue
                best = self.favourite(fits, pos)
                args = self.cart_args(best, self.wants[pos]["qty"])
                if args is None:
                    continue
                reply = self.act([{"name": cart, "args": args}])
                self.trace.append({"kind": "forced_cart", "item": best, "reply": reply})
                if reply is not None and not self.failed(reply.get("observation")):
                    self.cart_writes += 1
                    self.basket = self.cart_effect(args) or self.basket
                    payload, _ = _unwrap(reply.get("observation"))
                    self.read_basket(payload)
        message = self.roles.get("message")
        if order and self.basket and self.reserved() and message and self.messages_left() > 0 and self.steps_left() >= 2 and self.time_left() > 6:
            args = self.message_args(self.cart_summary())
            if args is not None:
                reply = self.act([{"name": message, "args": args}])
                self.trace.append({"kind": "forced_duty", "args": args, "reply": reply})
                self.messages_used += 1
                self.duty_done = True
                self.duty_basket = dict(self.basket)
                if self.finished or self.dead or self.steps_left() <= 0:
                    return
        if order and self.basket:
            action = {"name": order, "args": self.neutral_args(order, "Order the cart.", "order")}
        elif decline:
            action = {"name": decline, "args": self.neutral_args(decline, "Sorry, nothing in the store meets all of your requirements, so I am not buying anything.", "decline")}
        else:
            return
        if self.steps_left() <= 0:
            return
        reply = self.act([action])
        self.trace.append({"kind": "forced_finish", "action": action, "reply": reply})
        if reply and not reply.get("finished") and isinstance(reply.get("observation"), dict) and "refusal" in reply["observation"] and decline and self.steps_left() > 0:
            action = {"name": decline, "args": self.neutral_args(decline, "Sorry, nothing in the store meets all of your requirements, so I am not buying anything.", "decline")}
            reply = self.act([action])
            self.trace.append({"kind": "forced_finish", "action": action, "reply": reply})

    def failed(self, observation: Any) -> bool:
        if not isinstance(observation, dict):
            return False
        if observation.get("accepted") is False or observation.get("ok") is False:
            return True
        return any(_FAULT_CUE.search(str(k)) for k in observation) or (isinstance(observation.get("error"), str) and bool(observation.get("error")))

    def remember(self, name: str, observation: Any) -> None:
        payload, _ = _unwrap(observation)
        if name in (self.roles.get("cart"), self.roles.get("cart_view")):
            self.read_basket(payload)
            return
        if not isinstance(payload, dict):
            return
        if name == self.roles.get("search"):
            self.searches += 1
            for key, value in sorted(payload.items()):
                if isinstance(value, str) and value.strip() and " " not in value.strip() and len(value) <= 64:
                    self.cursors.add(value.strip())
        rows = next((v for v in payload.values() if isinstance(v, list) and v and all(isinstance(x, dict) for x in v)), None)
        if rows is not None:
            self.lens.absorb(rows)
            for row in rows:
                ident = self.lens.ident(row)
                if not ident:
                    continue
                self.rows[ident] = {k: v for k, v in row.items() if isinstance(v, (str, int, float, bool)) and not (isinstance(v, str) and len(v) > 160)}
                if _facts_of(row) is not None:
                    self.sheets[ident] = {k: (_shorten(v, 160) if not isinstance(v, (dict, list)) else v) for k, v in row.items()}
            return
        ident = self.lens.ident(payload)
        if not ident:
            return
        if _facts_of(payload) is not None:
            self.sheets[ident] = {k: (_shorten(v, 160) if not isinstance(v, (dict, list)) else v) for k, v in payload.items()}
        else:
            self.offers[ident] = {k: v for k, v in payload.items() if isinstance(v, (str, int, float, bool))}

    def notes_for(self, name: str, observation: Any) -> str:
        notes = []
        if isinstance(observation, dict):
            _gather_ids(observation, self.seen_ids)
            if observation.get("error") == "budget_exceeded":
                if name == self.roles.get("message") or self.message_budget is not None:
                    self.message_budget = self.messages_used if self.message_budget is None else min(self.message_budget, self.messages_used)
                notes.append("That allowance is used up; do not call this tool again.")
            if "refusal" in observation:
                self.refusals += 1
                self.order_checked = False
                notes.append("THE ORDER WAS REFUSED and the episode goes on; the cart is unchanged. Apply every change in the refusal, fix the cart (take out any line that is no longer acceptable, such as a sold-out item), re-check stock and price of the new pick, and order again.")
            elif name != self.roles.get("message"):
                _, said = _unwrap(observation)
                if said:
                    notes.append("This result carries a message from the shopper (a change). Apply it: " + " | ".join(said[:4]))
        elif isinstance(observation, list):
            _gather_ids(observation, self.seen_ids)
        return " ".join(notes)

    def shopper_lines(self, name: str, observation: Any) -> list[str]:
        heard: list[str] = []
        if name == self.roles.get("message"):
            if isinstance(observation, str) and observation.strip():
                heard = [observation.strip()]
            elif isinstance(observation, dict) and "error" not in observation:
                for value in observation.values():
                    if isinstance(value, str) and value.strip():
                        heard.append(value.strip())
                    elif isinstance(value, list):
                        heard += [v.strip() for v in value if isinstance(v, str) and v.strip()]
            return heard
        if isinstance(observation, dict):
            _, said = _unwrap(observation)
            heard += said
            if "refusal" not in observation:
                for value in observation.values():
                    if isinstance(value, list) and value and all(isinstance(v, str) for v in value):
                        _lines_from(value, heard)
        seen = set()
        return [h for h in heard if not (h in seen or seen.add(h))]

    def screen(self, call_id: str, name: Any, raw: Any, planned: bool, allowed: set[str], batch: int, queued_messages: int) -> tuple[str | None, dict[str, Any] | None]:
        message_tool, cart_tool = self.roles.get("message"), self.roles.get("cart")
        finishing = {self.roles.get("order"), self.roles.get("decline")} - {None}
        if name not in self.schemas:
            return f"Not run (no step used): '{name}' is not one of the tools. Use one of: {', '.join(sorted(allowed))}.", None
        if not planned and name not in allowed:
            if name == message_tool and self.planner_on:
                return "Not run (no step used): questions to the shopper are planned and sent for you once you have opened a few listings' details.", None
            reason = "this task's spending limit is nearly reached" if self.closing() and self.steps_left() > 7 else "too few steps are left for this"
            return "Not run (no step used): " + reason + "; only these tools can be used now: " + ", ".join(sorted(allowed)) + ".", None
        if isinstance(raw, dict):
            args = raw
        else:
            text = raw if isinstance(raw, str) else ""
            parsed = _object_in(text, last=False) if text.strip() else {}
            if parsed is None:
                return "Not run (no step used): the arguments were not a valid JSON object.", None
            args = parsed
        args, problems = _gate(args, self.schemas[name], "args")
        if problems:
            return "Not run (no step used): invalid arguments: " + "; ".join(problems[:6]), None
        if name in finishing and batch > 1:
            return "Not run (no step used): an order or an ending without purchase must be sent alone, after you have seen the results of your other calls.", None
        if name == message_tool and not self.duty_phase and self.reserved() and self.messages_left() - queued_messages == 1:
            if not self.basket:
                return "Not run (no step used): the last message is kept for what the request requires before the order (" + self.duty + "); send it once the final cart is in place.", None
            self.duty_phase = True
        if name == message_tool and self.message_budget is not None and self.messages_used + queued_messages >= self.message_budget:
            return "Not run (no step used): the message allowance is used up. Do not message the shopper again.", None
        signature = name + _flat(args)
        if name not in finishing and name not in (message_tool, cart_tool) and signature in self.recent:
            return "Not run (no step used): you just made this exact call and nothing has changed since; its result is above. Change the arguments or do something else.", None
        if name == self.roles.get("search"):
            for key, value in sorted(args.items()):
                if isinstance(value, str) and _CURSOR_CUE.search(str(key)) and value.strip() and value.strip() not in self.cursors:
                    return f"Not run (no step used): '{key}' only takes a continuation value that an earlier result of this search returned, and '{value}' was never returned. Leave it out for the first page.", None
        if name == cart_tool and self.steps_left() > 3:
            for target in self.cart_target(args):
                pos = self.owner(target)
                key = f"cart|{target}|{self.version}"
                if key in self.bounced:
                    continue
                self.bounced.add(key)
                verdict = self.note_fixed(target, True, pos if pos is not None else 0) or self.check_item(target, "ADD TO THE CART", pos)
                if verdict:
                    return "Not run (no step used). " + verdict + " If you are certain it fits, send the same call again.", None
        if name == cart_tool and self.steps_left() > 3 and self.cart_target(args):
            added = self.cart_target(args)
            slots = {self.owner(i) for i in added} - {None}
            projected = {i: q for i, q in (self.cart_effect(args) or {}).items() if i in added or self.owner(i) is None or self.owner(i) not in slots}
            issue = self.over_budget(projected, "would cost")
            key = f"budget|{signature}|{self.version}"
            if issue and key not in self.bounced:
                self.bounced.add(key)
                return "Not run (no step used): with this change " + issue + ". Choose listings that keep the whole cart within the budget. If you are sure, send the same call again.", None
        if name == cart_tool:
            warning = self.quantity_issue(name, args)
            key = "qty|" + signature
            if warning and key not in self.bounced:
                self.bounced.add(key)
                return "Not run (no step used): " + warning + " If you are sure, send the same call again.", None
        if name == message_tool and not planned and not self.duty_phase and not (self.duty_owed() and self.basket) and "?" not in " ".join(str(v) for v in args.values() if isinstance(v, str)):
            key = "say|" + signature
            if key not in self.bounced:
                self.bounced.add(key)
                return "Not run (no step used): a message to the shopper should ask one question about one property; a statement or confirmation only uses up the allowance. If a rule in the request requires this message, send the same call again.", None
        return None, args

    def run_calls(self, calls: list[dict[str, Any]], planned: bool = False) -> list[tuple[str, str]]:
        parts: dict[str, list[str]] = {}
        allowed = self.allowed_names()
        runnable: list[tuple[str, str, dict[str, Any]]] = []
        finishing = {self.roles.get("order"), self.roles.get("decline")} - {None}
        message_tool, cart_tool = self.roles.get("message"), self.roles.get("cart")
        queued = 0
        for call in calls:
            call_id = call["id"]
            spec = call.get("function") or {}
            refusal, args = self.screen(call_id, spec.get("name"), spec.get("arguments"), planned, allowed, len(calls), queued)
            if refusal is not None or args is None:
                parts.setdefault(call_id, []).append(refusal or "Not run (no step used).")
                continue
            if spec.get("name") == message_tool:
                queued += 1
            runnable.append((call_id, spec["name"], args))
        if runnable and runnable[0][1] in finishing:
            call_id, name, args = runnable[0]
            kind = self.intent(name, args)
            if kind == "order" and self.plan.get("priority") == "cheapest":
                bound = self.bind_cheaper()
                if bound:
                    parts.setdefault(call_id, []).append(bound)
                    return self.assemble(calls, parts)
            if kind == "order" and self.steps_left() > 2:
                issue = self.over_budget(self.basket)
                key = f"budget|order|{self.version}|{_flat(sorted(self.basket.items()))}"
                if issue and key not in self.bounced:
                    self.bounced.add(key)
                    parts.setdefault(call_id, []).append("Not run yet (no step used). Right now " + issue + ". Bring the cart within the budget (a cheaper verified fit), then order.")
                    return self.assemble(calls, parts)
            if kind == "order":
                thin = self.evidence_note()
                if thin:
                    parts.setdefault(call_id, []).append("Not run yet (no step used). " + thin)
                    return self.assemble(calls, parts)
            if kind == "order" and not self.order_checked and self.steps_left() > 2 and self.time_left() > CLOSING_SECONDS:
                note = self.order_hint()
                if note:
                    parts.setdefault(call_id, []).append("Not run yet (no step used). " + note)
                    return self.assemble(calls, parts)
                self.order_checked = True
                for ident in list(self.basket)[:3]:
                    verdict = self.check_item(ident, "ORDER", self.owner(ident))
                    if verdict:
                        parts.setdefault(call_id, []).append("Not run yet (no step used). " + verdict)
                        return self.assemble(calls, parts)
                lines = "one line per product on the list, each at that product's quantity" if self.is_list() else "exactly one line, the chosen listing at the required quantity"
                parts.setdefault(call_id, []).append(f"Not run yet (no step used). Before the order: (1) the cart holds {lines}, and nothing else; (2) after your last cart change you re-read stock and price with a tool that shows them, everything is in stock and the total is within the current budget; (3) each listing's own properties meet every stated requirement, every revealed need and every announced change; (4) if the cheapest or best was asked for, nothing better that fits exists among everything you could find; (5) every working rule in the request has been followed. Check whatever is unchecked; otherwise call the order tool again.")
                return self.assemble(calls, parts)
            if kind == "decline":
                proof = self.proof_order()
                if proof is not None:
                    parts.setdefault(call_id, []).append(proof[0])
                    if proof[1] is None or self.finished or self.steps_left() <= 0:
                        return self.assemble(calls, parts)
                    runnable[0] = (call_id, self.roles["order"], proof[1])
                    kind = "order"
            if kind == "decline":
                swap = self.instead_of_decline()
                if swap is not None:
                    self.trace.append({"kind": "instead_of_decline", "tool": swap[0], "args": swap[1]})
                    parts.setdefault(call_id, []).append(swap[2])
                    runnable[0] = (call_id, swap[0], swap[1])
                    kind = self.intent(swap[0], swap[1])
            key = "duty|order|" + _flat(sorted(self.basket.items()))
            if kind == "order" and self.duty_owed() and message_tool and self.steps_left() > 2 and self.messages_left() > 0 and key not in self.bounced:
                self.bounced.add(key)
                self.duty_phase = True
                parts.setdefault(call_id, []).append("Not run yet (no step used). The request requires this before the order: " + self.duty + ". Send that message to the shopper now with " + message_tool + " (name each listing in the cart with its quantity, and the total), read the reply, then order.")
                return self.assemble(calls, parts)
            if kind == "decline":
                note = self.decline_hint()
                if note:
                    parts.setdefault(call_id, []).append("Not run yet (no step used). " + note)
                    return self.assemble(calls, parts)
                if self.decline_checked and not self.search_nudged and self.steps_left() > 12 and self.searches < 8 and self.time_left() > CLOSING_SECONDS + 30:
                    self.search_nudged = True
                    parts.setdefault(call_id, []).append(f"Not run yet (no step used). {self.steps_left()} steps remain and only {self.searches} searches were made. Search in other ways first: the needed value in the sellers' words or as a property value, the category listing, further pages; open candidates you have not read and re-read the ones you opened. End without buying only if every candidate still fails.")
                    return self.assemble(calls, parts)
                if not self.decline_checked and self.steps_left() > 6 and self.time_left() > CLOSING_SECONDS + 30:
                    self.decline_checked = True
                    second = self.second_look()
                    if second:
                        parts.setdefault(call_id, []).append("Not run yet (no step used). " + second)
                        return self.assemble(calls, parts)
                    parts.setdefault(call_id, []).append("Not run yet (no step used). Before ending without a purchase, be sure: you searched in several ways (other words, filters, pages), you checked each candidate's own properties, and each one fails a stated requirement, a revealed need, the budget or stock. Near-synonyms and any spelling of a property name count. If something fits, buy it; if nothing does, call the ending tool again.")
                    return self.assemble(calls, parts)
                held = self.held_note()
                if held:
                    parts.setdefault(call_id, []).append("Not run yet (no step used). " + held)
                    return self.assemble(calls, parts)
                runnable[0] = (call_id, runnable[0][1], self.scrub_decline(runnable[0][2], runnable[0][1]))
        if runnable and runnable[0][1] in finishing and self.intent(runnable[0][1], runnable[0][2]) == "order" and not self.finished:
            checked = self.reread_cart()
            if checked is not None:
                parts.setdefault(runnable[0][0], []).append(checked[0])
                if checked[1] or self.finished or self.steps_left() <= 0:
                    return self.assemble(calls, parts)
        if runnable and runnable[0][1] not in finishing and finishing:
            keep = self.steps_left() - 1
            if keep < len(runnable):
                for call_id, _, _ in runnable[max(0, keep):]:
                    parts.setdefault(call_id, []).append("Not run (no step used): only the last step is left, and it must be the order (or the ending without purchase).")
                runnable = runnable[: max(0, keep)]
        for start in range(0, len(runnable), 16):
            chunk = runnable[start:start + 16]
            if self.finished or self.dead or self.steps_left() <= 0:
                for call_id, _, _ in chunk:
                    parts.setdefault(call_id, []).append("Not run: the episode is over or no steps are left.")
                continue
            dropped = chunk[self.steps_left():]
            chunk = chunk[: self.steps_left()]
            for call_id, _, _ in dropped:
                parts.setdefault(call_id, []).append("Not run: no steps are left for this call.")
            actions = [{"name": name, "args": args} for _, name, args in chunk]
            before = self.step
            reply = self.act(actions)
            self.trace.append({"kind": "env", "step": before + 1, "actions": actions, "reply": reply})
            if reply is None:
                for call_id, _, _ in chunk:
                    parts.setdefault(call_id, []).append("The call failed (network or server). Try again.")
                continue
            observations = reply.get("observations") if isinstance(reply.get("observations"), list) else [reply.get("observation")]
            for pos, (call_id, name, args) in enumerate(chunk):
                if pos >= len(observations):
                    parts.setdefault(call_id, []).append("Not run: the episode finished before this call.")
                    continue
                observation = observations[pos]
                parts.setdefault(call_id, []).append(self.digest(name, args, observation, pos == len(observations) - 1 and bool(reply.get("finished"))))
        return self.assemble(calls, parts)

    def mark_cart(self, effect: dict[str, int]) -> None:
        for ident in sorted(set(effect) | set(self.basket)):
            if effect.get(ident) != self.basket.get(ident):
                self.cart_marks[ident] = len(self.records)
            if ident in self.basket and not effect.get(ident):
                self.taken_out.add(ident)

    def stale_lines(self) -> list[str]:
        out = []
        for ident in self.basket:
            since = self.records[self.cart_marks.get(ident, 0):]
            if not any(self.lens.stock(found) is not None for record in since for found in [self._find(record, ident, 0)] if found is not None):
                out.append(ident)
        return out

    def details_args(self, ident: str) -> dict[str, Any] | None:
        spec = self.schemas.get(self.roles.get("details") or "") or {}
        fields = spec.get("properties") or {}
        texts = [k for k, v in sorted(fields.items()) if isinstance(v, dict) and "string" in _kinds_of(v) and "enum" not in v]
        required = list(spec.get("required") or [])
        target = next((k for k in required if k in texts), None) or (texts[0] if len(texts) == 1 else None)
        if not target or any(k != target for k in required):
            return None
        args, problems = _gate({target: ident}, spec, "args")
        return None if problems else args

    def reread_cart(self) -> tuple[str, bool] | None:
        tool = self.roles.get("details")
        if "reread|blind" in self.bounced or not tool or not self.basket or self.lens.stock_key is None or self.time_left() < CLOSING_SECONDS:
            return None
        stale = self.stale_lines()
        if not stale or len(stale) > 3 or self.steps_left() < len(stale) + 2:
            return None
        actions = []
        for ident in stale:
            args = self.details_args(ident)
            if args is None:
                return None
            actions.append({"name": tool, "args": args})
        before, version = self.step, self.version
        reply = self.act(actions)
        self.trace.append({"kind": "reread", "step": before + 1, "items": stale, "reply": reply})
        if reply is None:
            return None
        observations = reply.get("observations") if isinstance(reply.get("observations"), list) else [reply.get("observation")]
        texts = [self.digest(tool, action["args"], observation, False)[:900] for action, observation in zip(actions, observations)]
        before_set = set(stale)
        unread = [ident for ident in sorted(before_set) if ident in self.stale_lines()]
        if unread:
            self.bounced.add("reread|blind")
        gone = [ident for ident in stale if self.stock_of(ident) is False]
        head = "Before the order, the code re-read the cart lines changed since their last stock reading: " + " | ".join(texts)
        if gone:
            self.version += 1
            return head + "\n[note] THE ORDER WAS NOT SENT: " + ", ".join(gone) + " no longer shows as in stock. Take it out, put in the next verified fit, re-check it, then order.", True
        if self.version != version:
            return head + "\n[note] THE ORDER WAS NOT SENT: the shopper's words above change the request. Apply the change to the cart, then order.", True
        dear = self.over_budget(self.basket)
        if dear:
            return head + "\n[note] THE ORDER WAS NOT SENT: after the re-read " + dear + ". Bring the cart within the budget, then order.", True
        if unread:
            return head + "\n[note] The re-read showed no stock field for " + ", ".join(unread) + "; the order follows.", False
        return head + "\n[note] Every re-read cart line is still in stock; the order follows.", False

    def assemble(self, calls: list[dict[str, Any]], parts: dict[str, list[str]]) -> list[tuple[str, str]]:
        out = []
        for call in calls:
            texts = parts.get(call["id"])
            out.append((call["id"], "\n".join(texts) if texts else "Not run (no step used)."))
        return out

    def digest(self, name: str, args: dict[str, Any], observation: Any, last_and_finished: bool) -> str:
        message_tool, cart_tool = self.roles.get("message"), self.roles.get("cart")
        finishing = {self.roles.get("order"), self.roles.get("decline")} - {None}
        if name == message_tool:
            self.messages_used += 1
            if self.duty_phase:
                self.duty_done = True
                self.duty_basket = dict(self.basket)
                self.duty_phase = False
        if self.intent(name, args) == "order":
            self.order_attempts += 1
        if name in (cart_tool, message_tool) or name in finishing:
            self.recent = []
        else:
            self.recent = (self.recent + [name + _flat(args)])[-2:]
        self.records.append(observation)
        if name == cart_tool:
            self.order_checked = False
            if not self.failed(observation):
                self.cart_writes += 1
                shape, parts = self.cart_shape()
                if shape == "verb" and self.add_value is None and self.cart_target(args):
                    self.add_value = str(args.get(parts["verb"]))
                effect = self.cart_effect(args)
                if effect is not None:
                    self.mark_cart(effect)
                    self.basket = effect
        self.remember(name, observation)
        if name == self.roles.get("search"):
            for key, value in sorted(args.items()):
                if isinstance(value, str) and _SHELF_CUE.search(str(key)):
                    self.lens.note_shelf_value(value)
        inner = _unwrap(observation)[0]
        note = " ".join(x for x in (self.notes_for(name, observation), self.page_hint(name, args, inner), self.echo_hint(name, args, inner)) if x)
        heard = self.shopper_lines(name, observation)
        question = " ".join(str(v) for v in args.values() if isinstance(v, str)) if name == message_tool else None
        if heard:
            meaning = self.interpret(heard, question)
            if meaning:
                note = (note + " " if note else "") + meaning
        text = _flat(observation)
        if last_and_finished:
            text += " [the episode is finished]"
        if note:
            text += "\n[note] " + note
        return text

    def run(self) -> list[dict[str, Any]]:
        if self.finished:
            return self.trace
        self.setup()
        self.trace.append({"kind": "plan", "plan": self.plan, "roles": self.roles})
        self.infer_roles()
        if not self.plan:
            self.shop_without_models()
            return self.trace
        opening = self.policy.get("opening")
        user = "SHOPPER REQUEST:\n" + str(self.policy.get("query") or "")
        if opening is not None:
            user += "\n\nOPENING OBSERVATION:\n" + _flat(opening)
            self.records.append(opening)
            _gather_ids(opening, self.seen_ids)
        overview = self.overview()
        if overview:
            user += "\n\n" + overview
        user += "\n\n" + self.status()
        self.messages = [{"role": "system", "content": DRIVER_RULES}, {"role": "user", "content": user}]
        idle = failures = counter = 0
        for _ in range(TURN_LIMIT):
            if self.finished or self.dead or self.steps_left() <= 0:
                break
            if self.time_left() < LAST_SECONDS:
                self.force_finish()
                break
            if self.pending_reply:
                text, self.pending_reply = self.pending_reply, ""
                if self.send_reply(text):
                    continue
            if self.planner_on and self.ask_planned():
                continue
            if self.cost + max(self.last_turn_cost, 0.002) >= STOP_SPEND:
                self.trace.append({"kind": "spend_stop", "cost": round(self.cost, 5)})
                self.force_finish()
                break
            self.trim_history()
            message = self.call_model(self.messages, self.tools_now(), [DRIVER, SPARE, DRIVER], 2500)
            if message is None:
                failures += 1
                if failures >= 4 or self.time_left() < CLOSING_SECONDS + 20:
                    self.force_finish()
                    break
                time.sleep(min(6.0, 2.0 * failures))
                continue
            failures = 0
            calls = []
            for call in message.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                counter += 1
                spec = call.get("function") or {}
                arguments = spec.get("arguments") if isinstance(spec.get("arguments"), str) else json.dumps(spec.get("arguments") or {})
                calls.append({"id": call.get("id") or f"call_{counter}", "type": "function", "function": {"name": spec.get("name"), "arguments": arguments}})
            seen_ids: set[str] = set()
            for call in calls:
                if call["id"] in seen_ids:
                    counter += 1
                    call["id"] = f"{call['id']}_{counter}"
                seen_ids.add(call["id"])
            content = message.get("content") or ""
            if not isinstance(content, str):
                content = _flat(content)
            committing = {self.roles.get("cart"), self.roles.get("order"), self.roles.get("decline")} - {None}
            if self.planner_on and any(c["function"]["name"] in committing for c in calls):
                if self.ask_planned(force=True):
                    continue
            if not calls:
                idle += 1
                self.messages.append({"role": "assistant", "content": content or "(no tool call)"})
                self.messages.append({"role": "user", "content": "Every reply must call at least one of the tools. Continue.\n" + self.status()})
                if idle >= 3:
                    self.force_finish()
                    break
                continue
            idle = 0
            self.trace.append({"kind": "model", "content": content[:2000], "calls": [c["function"] for c in calls]})
            self.messages.append({"role": "assistant", "content": content or None, "tool_calls": calls})
            replies = self.run_calls(calls)
            for pos, (call_id, text) in enumerate(replies):
                if pos == len(replies) - 1:
                    text += "\n" + self.status()
                self.messages.append({"role": "tool", "tool_call_id": call_id, "content": text})
        if not self.finished and not self.dead:
            self.force_finish()
        self.trace.append({"kind": "end", "finished": self.finished, "steps": self.step, "cost": round(self.cost, 5), "seconds": round(time.monotonic() - self.started, 1)})
        return self.trace


    def infer_roles(self) -> None:
        found: dict[str, str] = {}
        finisher: str | None = None
        lookers: list[tuple[bool, int, str]] = []
        for tool in self.tools:
            spec = tool["function"]
            name = spec["name"]
            text = name + " " + str(spec.get("description") or "")
            schema = spec.get("parameters") or {}
            fields = {k: v for k, v in sorted((schema.get("properties") or {}).items()) if isinstance(v, dict)}
            required = list(schema.get("required") or [])
            options = {k: [str(o) for o in v["enum"]] for k, v in fields.items() if isinstance(v.get("enum"), list)}
            free = [k for k, v in fields.items() if "string" in _kinds_of(v) and "enum" not in v]
            counts = [k for k, v in fields.items() if set(_kinds_of(v)) & {"integer", "number"}]
            arrays = [k for k, v in fields.items() if "array" in _kinds_of(v)]
            note = str(spec.get("description") or "")
            switch, buy_value, stop_value = None, None, None
            for key, opts in options.items():
                stops, buys = _finish_split(opts, note)
                if len(opts) == 2 and len(stops) == 1 and buys and not counts and not arrays:
                    switch, buy_value, stop_value = key, buys[0], stops[0]
                    break
            if switch:
                finisher = finisher or name
                if not self.switch_args:
                    self.switch_args = {"order": {switch: buy_value}, "decline": {switch: stop_value}}
                if "order" not in self.finish_args and "decline" not in self.finish_args:
                    self.finish_args = {"order": {switch: buy_value}, "decline": {switch: stop_value}}
            elif _STOP_CUE.search(text) and not counts and not arrays and not any(k in required for k in free):
                found.setdefault("decline", name)
            elif _BUY_CUE.search(text) and not counts and not arrays and not free:
                found.setdefault("order", name)
            elif arrays or any(k in required for k in free) and any(k in required for k in counts) or (free and any(_ADD_CUE.search(o) for opts in options.values() for o in opts)) \
                    or (options and counts and any(k in required for k in free) and not _LOOK_CUE.search(text)):
                found.setdefault("cart", name)
            elif len(free) == 1 and len(fields) == 1 and _REMOVE_CUE.search(text):
                found.setdefault("remove", name)
            elif len(free) == 1 and len(fields) == 1 and _ASK_CUE.search(text):
                found.setdefault("message", name)
            elif not fields:
                found.setdefault("cart_view", name)
            elif len(fields) == 1 and len(free) == 1:
                found.setdefault("details", name)
            elif free or options:
                lookers.append((not _LOOK_CUE.search(text), len(lookers), name))
        if finisher:
            found["order"] = found["decline"] = finisher
        if lookers:
            found.setdefault("search", min(lookers)[2])
        for role, name in sorted(found.items()):
            self.roles.setdefault(role, name)
        self.trim_finish_args()
        if not self.finish_args and self.switch_args and self.roles.get("order") == self.roles.get("decline") == finisher:
            self.finish_args = {k: dict(v) for k, v in self.switch_args.items()}

    def request_numbers(self) -> tuple[int, list[str], float | None]:
        query = str(self.policy.get("query") or "")
        cap = None
        money = re.search(r"[$\u20ac\u00a3]\s*(\d+(?:\.\d+)?)", query) or re.search(r"(?i)(\d+(?:\.\d+)?)\s*(?:cents|dollars)", query)
        if money:
            cap = float(money.group(1))
        pieces = re.findall(r"[$\u20ac\u00a3]?[A-Za-z0-9][A-Za-z0-9'.-]*|[,.;:!?]", query)
        qty, start = 1, None
        for pos, piece in enumerate(pieces):
            if re.fullmatch(r"\d{1,2}", piece) and 0 < int(piece) <= 50 and pos + 1 < len(pieces) and re.fullmatch(r"[A-Za-z][A-Za-z'-]*", pieces[pos + 1]):
                qty, start = int(piece), pos + 1
                break
            if piece.lower() in ("a", "an", "one") and start is None and pos + 1 < len(pieces):
                start = pos + 1
                break
        phrase: list[str] = []
        for piece in pieces[start:] if start is not None else []:
            if not re.fullmatch(r"[A-Za-z]+", piece):
                if phrase and not re.search(r"[A-Za-z0-9]", piece):
                    break
                continue
            if len(piece) < 4:
                if phrase:
                    break
                continue
            phrase.append(piece.lower())
            if len(phrase) == 2:
                break
        return qty, phrase, cap

    def shop_without_models(self) -> None:
        qty, phrase, cap = self.request_numbers()
        if not self.wants:
            self.wants = [{"product": " ".join(phrase), "qty": qty, "stated": [], "revealed": [], "dropped": [], "now": None}]
        self.wants[0]["qty"] = qty
        search, order, decline = self.roles.get("search"), self.roles.get("order"), self.roles.get("decline")
        query_words = set(_tokens(self.policy.get("query") or ""))
        rows: list[dict[str, Any]] = []
        tries: list[Any] = []
        shelf_args = self.shelf_search(search) if search else None
        if shelf_args:
            tries.append(shelf_args)
        tries += [" ".join(phrase)] + ([phrase[-1]] if len(phrase) > 1 else [])
        tries += [t for t in self.anchored_words() if t not in tries]
        searched = 0
        for words in tries:
            if searched >= 6:
                break
            searched += 1
            if not search or not words or self.finished or self.steps_left() <= 3:
                break
            args = words if isinstance(words, dict) else self.search_args(search, words)
            if args is None:
                continue
            reply = self.act([{"name": search, "args": args}])
            if reply is None:
                break
            observation = reply.get("observation")
            self.records.append(observation)
            self.remember(search, observation)
            payload, _ = _unwrap(observation)
            found = next((v for v in payload.values() if isinstance(v, list) and v and all(isinstance(x, dict) for x in v)), None) if isinstance(payload, dict) else None
            if found:
                rows = found
                break
        prices = [self.lens.price(r) for r in rows if self.lens.price(r) is not None]
        if cap is not None and prices and min(prices) > 10 * cap:
            cap *= 100
        ranked = []
        for row in rows:
            ident = self.lens.ident(row)
            price = self.lens.price(row)
            if not ident or self.lens.stock(row) is False or (cap is not None and price is not None and price > cap):
                continue
            overlap = len(query_words & set(_tokens(self.lens.title(row) or _flat(row))))
            ranked.append((-overlap, price if price is not None else float("inf"), ident))
        ranked.sort()
        picks = [i for _, _, i in ranked]
        if not picks or not order:
            if decline and not self.finished and self.steps_left() > 0:
                self.act([{"name": decline, "args": self.neutral_args(decline, "Sorry, nothing in the store meets all of your requirements, so I am not buying anything.", "decline")}])
            return
        current = None
        for _ in range(4):
            if self.finished or self.steps_left() <= 1:
                return
            if current is None:
                if not picks:
                    break
                current = picks.pop(0)
                if not self.code_cart(current, self.wants[0]["qty"]):
                    current = None
                    continue
            reply = self.act([{"name": order, "args": self.neutral_args(order, "Order the cart.", "order")}])
            if reply is None or self.finished:
                return
            observation = reply.get("observation")
            if not (isinstance(observation, dict) and "refusal" in observation):
                if self.failed(_unwrap(observation)[0]):
                    current = None
                continue
            _, said = _unwrap(observation)
            numbers = [int(n) for n in re.findall(r"(?<![\w.$-])(\d{1,2})(?![\w%-]|\.\d)", " ".join(said)) if 0 < int(n) <= 50]
            if len(numbers) >= 2 and self.wants[0]["qty"] in numbers and numbers[0] != self.wants[0]["qty"]:
                self.wants[0]["qty"] = numbers[0]
                self.code_uncart(current)
                if not self.code_cart(current, numbers[0]):
                    current = None
                continue
            self.code_uncart(current)
            current = None
        if decline and not self.finished and self.steps_left() > 0:
            self.act([{"name": decline, "args": self.neutral_args(decline, "Sorry, nothing in the store meets all of your requirements, so I am not buying anything.", "decline")}])

    def anchored_words(self) -> list[str]:
        pieces = re.findall(r"[A-Za-z0-9][A-Za-z0-9'-]*|[,.;:!?]", str(self.policy.get("query") or ""))
        found: list[str] = []
        for pos, piece in enumerate(pieces):
            if not (piece.lower() in ("a", "an", "one", "the") or piece.isdigit()):
                continue
            words = []
            for nxt in pieces[pos + 1:pos + 5]:
                if not re.fullmatch(r"[A-Za-z]+", nxt):
                    if words or not re.search(r"[A-Za-z0-9]", nxt):
                        break
                    continue
                if len(nxt) < 4:
                    if words:
                        break
                    continue
                words.append(nxt.lower())
            for word in reversed(words):
                if word not in found:
                    found.append(word)
        return found

    def shelf_search(self, tool: str) -> dict[str, Any] | None:
        schema = self.schemas.get(tool) or {}
        fields = schema.get("properties") or {}
        wanted = [w for w in _tokens(self.policy.get("query") or "") if len(w) >= 4]
        best: tuple[int, int, str, str] | None = None
        for key, spec in sorted(fields.items()):
            options = spec.get("enum") if isinstance(spec, dict) else None
            if not isinstance(options, list) or len(options) < 5:
                continue
            for option in options:
                words = [w for w in _tokens(option) if len(w) >= 4]
                score = sum(1 for w in words if any(_akin(w, x) for x in wanted))
                if score and (best is None or (score, -len(str(option))) > (best[0], best[1])):
                    best = (score, -len(str(option)), key, str(option))
        if best is None:
            return None
        args: dict[str, Any] = {best[2]: best[3]}
        for key, spec in sorted(fields.items()):
            if key != best[2] and key in (schema.get("required") or []) and isinstance(spec, dict) and isinstance(spec.get("enum"), list) and spec["enum"]:
                args[key] = next((o for o in spec["enum"] if _SHELF_CUE.search(str(o))), spec["enum"][-1])
        args, problems = _gate(args, schema, "args")
        return None if problems else args

    def search_args(self, tool: str, words: str) -> dict[str, Any] | None:
        schema = self.schemas.get(tool) or {}
        fields = schema.get("properties") or {}
        args: dict[str, Any] = {}
        free = [k for k, v in sorted(fields.items()) if isinstance(v, dict) and "string" in _kinds_of(v) and "enum" not in v and not _CURSOR_CUE.search(k) and not _SHELF_CUE.search(k)]
        if not free:
            return None
        required = list(schema.get("required") or [])
        for key, spec in sorted(fields.items()):
            if key in required and isinstance(spec, dict) and isinstance(spec.get("enum"), list) and spec["enum"]:
                args[key] = next((o for o in spec["enum"] if re.search(r"(?i)text|word|phrase|free", str(o))), spec["enum"][0])
        chosen = " ".join(str(v).lower() for v in args.values())

        def rank(key: str) -> tuple[bool, bool, bool, str]:
            named = bool(chosen) and (key.lower() in chosen or any(part in key.lower() for part in chosen.split() if len(part) > 2))
            return (not named, key not in required, not re.search(r"(?i)word|text|phrase|query|^q$", key), key)

        args[min(free, key=rank)] = words
        args, problems = _gate(args, schema, "args")
        return None if problems else args

    def code_cart(self, ident: str, qty: int) -> bool:
        tool = self.roles.get("cart")
        args = self.cart_args(ident, qty) if tool else None
        if args is None:
            return False
        reply = self.act([{"name": tool, "args": args}])
        if reply is None:
            return False
        observation = reply.get("observation")
        self.records.append(observation)
        if self.failed(_unwrap(observation)[0]):
            return False
        self.basket = self.cart_effect(args) or self.basket
        self.cart_writes += 1
        return True

    def code_uncart(self, ident: str | None) -> None:
        if not ident or self.finished or self.steps_left() <= 2:
            return
        remover = self.roles.get("remove")
        if remover:
            fields = (self.schemas.get(remover) or {}).get("properties") or {}
            key = next((k for k, v in sorted(fields.items()) if isinstance(v, dict) and "string" in _kinds_of(v)), None)
            args, problems = _gate({key: ident} if key else {}, self.schemas.get(remover) or {}, "args")
            if key and not problems:
                self.act([{"name": remover, "args": args}])
                self.basket.pop(ident, None)
                return
        args = self.cart_args(ident, 0)
        if args is not None and self.roles.get("cart"):
            self.act([{"name": self.roles["cart"], "args": args}])
            self.basket.pop(ident, None)


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
