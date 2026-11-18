"""lotus kernel (c9): a light conductor model walks the store through the world's own tools, one turn per step; the
code keeps the sheet (what is wanted, what was told, the cart, what was seen, what is left) and shows it after every
result; the code also decides when the shopper is interviewed, proves the request's process steps before an ending,
asks a heavier model to check the cart before an order and the evidence before a decline, refuses calls that cannot
change anything without spending a step, and always finishes the episode itself when the conductor does not.

Budgets are counted, not only timed: a fixed number of conductor turns and of model calls, a soft spend past which the
helpers stop and a hard spend past which the loop stops, plus the task's wall clock.

problem_data["environment"] gives the binding and the policy view (query, opening, tools, max_steps). Each world tool
call is one step through /environment/call.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

from src.agent.proxy_client import ProxyClient

LIGHT = ["anthropic/claude-haiku-5.5", "openai/gpt-6-luna"]      # the conductor and the quick readings
HEAVY = ["anthropic/claude-haiku-5.5", "openai/gpt-6-luna"]      # the request, the interview plan, the cart check, the audit: the same
                                                                 # model without thinking (its thinking overruns a 1600-token answer in half the calls; a larger model costs 5-9x)

_TASK_SECONDS = 250.0          # the network gives each task 300 s
_CLOSE_SECONDS = 35.0          # below this the episode is closed by code
_WALL_SECONDS = 290.0          # nothing may still be waiting at this point
_CALL_SECONDS = 80.0           # longest wait for one model reply
_RETRY_PAUSES = (0, 5, 10, 20)   # pauses before each round of model calls when every model failed (seconds)
_OBS_CHARS = 3500              # newest observation shown to the conductor
_OLD_CHARS = 500               # older observations, cut down (the sheet keeps what was learned)
_HISTORY_CHARS = 150000        # the conversation is trimmed past this
_AUTO_PAGES = 6                # result pages the code reads itself per episode
_CHEAPER_OPENS = 3             # cheaper rows opened before a cheapest checkout
_TRAILS = 4                    # chain readings per episode

_LOOP_TURNS = 42               # conductor turns per episode
_EXTRA_TURNS = 10              # more turns after a refused forced checkout
_ALL_CALLS = 60                # model calls of any kind per episode
_MISSES = 3                    # consecutive conductor turns without a reply that end the loop
_IDLE = 3                      # consecutive conductor turns without a tool call that end the loop
_SPEND_SOFT = 0.028            # past this the helpers (checks, audits, readings) stop
_SPEND_HARD = 0.038            # past this the loop stops and the code finishes (a run's key must cover every task)

_proxy = ProxyClient(timeout=80, max_retries=2)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _fold(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def _short(value: Any, cap: int) -> str:
    """JSON text of an observation with long strings trimmed, then cut to cap characters."""

    def trim(node: Any) -> Any:
        if isinstance(node, str):
            return node if len(node) <= 260 else node[:260] + "..."
        if isinstance(node, dict):
            return {k: trim(v) for k, v in node.items()}
        if isinstance(node, list):
            return [trim(v) for v in node[:40]]
        return node
    text = json.dumps(trim(value), ensure_ascii=False, separators=(",", ":"))
    return text if len(text) <= cap else text[:cap] + "...(cut)"


def _json_object(text: Any) -> dict[str, Any] | None:
    if isinstance(text, dict):
        return text
    if not isinstance(text, str):
        return None
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except (ValueError, TypeError):
        match = re.search(r"\{.*\}", text, re.S)
        if match:
            try:
                parsed = json.loads(match.group(0))
                return parsed if isinstance(parsed, dict) else None
            except (ValueError, TypeError):
                return None
    return None


def _form(name: str, description: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"type": "function", "function": {"name": name, "description": description,
                                             "parameters": {"type": "object", "properties": properties, "required": required}}}


def _type_fits(value: Any, kind: Any) -> bool:
    kinds = kind if isinstance(kind, list) else [kind]
    for k in kinds:
        if k is None or k == "null" and value is None:
            return True
        if k == "string" and isinstance(value, str):
            return True
        if k == "integer" and isinstance(value, int) and not isinstance(value, bool):
            return True
        if k == "number" and isinstance(value, (int, float)) and not isinstance(value, bool):
            return True
        if k == "boolean" and isinstance(value, bool):
            return True
        if k == "array" and isinstance(value, list):
            return True
        if k == "object" and isinstance(value, dict):
            return True
    return False


def _repair(value: Any, schema: dict[str, Any]) -> Any:
    """Plain type mix-ups fixed before checking: "3" for an integer, 3 for a string, "true" for a boolean, an enum
    value in another case."""

    kind = schema.get("type")
    kinds = kind if isinstance(kind, list) else [kind]
    if isinstance(value, str):
        if "integer" in kinds and re.fullmatch(r"-?\d+", value.strip()):
            return int(value.strip())
        if "number" in kinds and re.fullmatch(r"-?\d+(\.\d+)?", value.strip()):
            return float(value.strip())
        if "boolean" in kinds and value.strip().lower() in ("true", "false"):
            return value.strip().lower() == "true"
        if "enum" in schema and value not in schema["enum"]:
            same = [e for e in schema["enum"] if isinstance(e, str) and e.lower() == value.strip().lower()]
            if same:
                return same[0]
    if isinstance(value, (int, float)) and not isinstance(value, bool) and "string" in kinds and "number" not in kinds and "integer" not in kinds:
        return str(value)
    if isinstance(value, float) and value.is_integer() and "integer" in kinds:
        return int(value)
    return value


def schema_problems(args: dict[str, Any], schema: dict[str, Any], where: str = "") -> tuple[dict[str, Any], list[str]]:
    """The arguments with plain type mix-ups repaired, and what still breaks the schema."""

    problems: list[str] = []
    props = schema.get("properties") or {}
    out: dict[str, Any] = {}
    for name in schema.get("required") or []:
        if name not in args or args[name] is None:
            problems.append(f"{where}{name} is required")
    for name, value in args.items():
        spec = props.get(name)
        if spec is None:
            if schema.get("additionalProperties") is False or props:
                problems.append(f"{where}{name} is not a parameter of this tool (parameters: {', '.join(props) or 'none'})")
            continue
        value = _repair(value, spec)
        if spec.get("type") is not None and not _type_fits(value, spec.get("type")):
            problems.append(f"{where}{name} needs type {spec.get('type')}")
        if "enum" in spec and value not in spec["enum"]:
            problems.append(f"{where}{name} takes only the values {spec['enum']}")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if "minimum" in spec and value < spec["minimum"]:
                problems.append(f"{where}{name} may not be below {spec['minimum']}")
            if "maximum" in spec and value > spec["maximum"]:
                problems.append(f"{where}{name} may not exceed {spec['maximum']}")
        if isinstance(value, str):
            if "maxLength" in spec and len(value) > spec["maxLength"]:
                value = value[: spec["maxLength"]]
            if "minLength" in spec and len(value) < spec["minLength"]:
                problems.append(f"{where}{name} needs at least {spec['minLength']} characters")
        if isinstance(value, list):
            if "maxItems" in spec and len(value) > spec["maxItems"]:
                problems.append(f"{where}{name} may hold at most {spec['maxItems']} items")
            if "minItems" in spec and len(value) < spec["minItems"]:
                problems.append(f"{where}{name} needs at least {spec['minItems']} items")
        if isinstance(value, dict) and spec.get("type") == "object":
            value, inner = schema_problems(value, spec, f"{where}{name}.")
            problems += inner
        out[name] = value
    return out, problems


# ---------------------------------------------------------------------------
# Prompts of the kernel (the request reading, the interview and the checks come with their own)
# ---------------------------------------------------------------------------

BRIEF_SYSTEM = """You read one shopper's request to an online-store assistant and the list of the store's tools, and you
write down what the assistant must get right, as a brief for a helper that never sees the request again.

Products: one entry per product the shopper wants, with its quantity and its conditions. The quantity is how many
units of one listing to buy: 1 unless the request states a number of units; "a pair" of shoes, gloves or socks, "a
set" or "a pack" is one unit, since the store sells them as one listing. A condition is "property: constraint" in the request's own property word (colour: red; length: at most
2 m; material: not wicker). A constraint is something a store could list for the property: a word or two, a number
with its unit, a limit, a yes or no. The shopper's story is context, not a condition, unless it states such a value
outright; how the product should sit, feel, look or serve, and the place, company or frequency of its use, stay context.
Two products joined by "and" inside one phrase are one product. A product the shopper names only to describe another
(a thing that must fit it) is not a product to buy. When a store tool has a fixed list of shelf or category names, copy
the one the product belongs to as its shelf.

Money: the total allowed, in the number and unit the request uses, and whether it covers everything together.
Stock: whether sold-out listings are excluded (true unless the request says otherwise).
Pick rule: how to choose among listings that fit: cheapest, most or least of a named measure, or any.
Allowances: how many messages the shopper allows and how many steps, when the request states them.
Lookups: a condition whose value the request does not state but points to somewhere in the store (the same value as
on some other listing's page): the property, the product it belongs to, and the request's own words for where to look.
Rules: anything else that binds the order (a seller to avoid, a delivery wish) in one short line each.
Ask first: true when the shopper asks to be asked before anything is chosen, or says there is something to find out.
Process steps: things the shopper asks the assistant to do as part of the order, each with when it is due (choose =
before a listing is chosen or carted, order = before the order is sent, end = at the very end) and how many messages
it takes (0 when it needs no message). A wish to be asked about needs is not a process step: it is ask first.
Quiet decline: true when the shopper says what to do, or not to do, when nothing fits; null otherwise.
Roles: which tool searches, opens one listing, compares several, messages the shopper, adds to the cart, shows the
cart, places the order, and ends without an order; null for a role no tool does. When one tool both orders and ends
without an order, give the fixed arguments that make it do each (buy_args, refuse_args)."""

BRIEF_FORM = _form("order_brief", "The brief for the helpers.", {
    "items": {"type": "array", "items": {"type": "object", "properties": {
        "name": {"type": "string"}, "shelf": {"type": ["string", "null"]}, "quantity": {"type": "integer"},
        "musts": {"type": "object", "additionalProperties": {"type": "string"}}}, "required": ["name", "quantity", "musts"]}},
    "budget": {"type": ["number", "null"]}, "unit": {"type": ["string", "null"]}, "budget_covers_all": {"type": "boolean"},
    "in_stock_only": {"type": "boolean"},
    "pick": {"type": ["string", "null"], "enum": ["cheapest", "most", "least", "any", None]}, "pick_measure": {"type": ["string", "null"]},
    "messages": {"type": ["integer", "null"]}, "steps": {"type": ["integer", "null"]},
    "lookups": {"type": "array", "items": {"type": "object", "properties": {"property": {"type": "string"}, "item": {"type": "integer"}, "where": {"type": "string"}}, "required": ["property", "item", "where"]}},
    "rules": {"type": "array", "items": {"type": "string"}},
    "ask_first": {"type": "boolean"},
    "process": {"type": "array", "items": {"type": "object", "properties": {"text": {"type": "string"}, "when": {"type": "string", "enum": ["choose", "order", "end"]}, "messages": {"type": "integer"}}, "required": ["text", "when", "messages"]}},
    "quiet_decline": {"type": ["boolean", "null"]},
    "roles": {"type": "object", "properties": {r: {"type": ["string", "null"]} for r in ("search", "open", "compare", "ask", "cart", "cart_view", "buy", "refuse")}},
    "buy_args": {"type": ["object", "null"]}, "refuse_args": {"type": ["object", "null"]}},
    ["items", "budget", "in_stock_only", "pick", "messages", "steps", "lookups", "rules", "ask_first", "process", "roles"])

PLAN_SYSTEM = """A shopper keeps a few fixed wishes about the products they asked an online-store assistant for, each
about one property of one product, and tells one only when asked about that exact property: a property they have no
view on gets "don't mind", one they never thought about gets a puzzled reply, so every question spends a message. You plan which properties to ask about,
from the menu of attributes the store's listings of each product actually carry (with sample values), the request and
the shopper's story.

Order: a product the shopper asked to be asked about first; then properties the story hints at (who will use it, where
it goes or must fit, what it is for, what it is measured or planned against); a measurement the request already states
puts the product's other measurements (its sibling dimensions, weight, capacity) next; then the listed attributes
ranked by what a buyer of that kind of product chooses between first.
Skip what the request already settles, a property that repeats another in other words,
variant lists, identifiers, codes, warranty and shipping. One property per question; the label is the attribute's full
plain name as a buyer would say it, never shortened to one word of it. For several products, every product's hinted
properties come first, then the rest in turns. Give exactly the number of questions asked for."""

PLAN_FORM = _form("question_plan", "The properties to ask about, in order.", {
    "questions": {"type": "array", "items": {"type": "object", "properties": {"item": {"type": "integer"}, "property": {"type": "string"}}, "required": ["item", "property"]}}},
    ["questions"])

STOCK_SYSTEM = """A store's listing row has a field saying whether the listing can be bought now. Given one value of that field, say whether it means it can be bought now."""

STOCK_FORM = _form("stock_value", "Whether this value means the listing can be bought now.", {"can_buy": {"type": "boolean"}}, ["can_buy"])

HEAR_SYSTEM = """An online-store assistant asked a shopper questions, one message at a time, about the products the shopper wants
(a message may name one property or several), and you read the replies. For each reply write what it settles. A need is a property the reply itself names, with the value
word for word including units, limits ("at least", "at most", "not") and alternatives; a reply that leaves the property
open, says it does not matter, or asks you to choose, settles nothing. A reply that points to a value found elsewhere in
the store (the same as on some other listing) is a pointer, with the reply's own words for where to look. A reply that
drops a condition names the property dropped; a reply that frees a property says it no longer matters; a reply that
states a new total allowed gives the new budget. Crowded: the shopper complains that the message asked about too many
things at once or wants the questions to come singly; a reply that only says it did not follow or understand the
question is not crowded and settles nothing. Restated: the reply only repeats what the request already said."""

HEAR_FORM = _form("replies_read", "What the replies settle.", {
    "needs": {"type": "array", "items": {"type": "object", "properties": {"item": {"type": "integer"}, "property": {"type": "string"}, "value": {"type": "string"}}, "required": ["item", "property", "value"]}},
    "pointers": {"type": "array", "items": {"type": "object", "properties": {"item": {"type": "integer"}, "property": {"type": "string"}, "where": {"type": "string"}}, "required": ["item", "property", "where"]}},
    "dropped": {"type": "array", "items": {"type": "string"}},
    "freed": {"type": "array", "items": {"type": "string"}},
    "budget": {"type": ["number", "null"]},
    "crowded": {"type": "array", "items": {"type": "integer"}, "description": "numbers of the questions the shopper called crowded or unclear"},
    "restated": {"type": "array", "items": {"type": "integer"}, "description": "numbers of the questions answered only by repeating the request"}},
    ["needs", "pointers", "dropped", "freed", "budget", "crowded", "restated"])

SHAPE_SYSTEM = """You read one raw result an online-store tool returned, holding a list of listing rows, and name the row keys
that hold the listing's price, its availability (sold out or not), and the key (if any) that holds the token or number
for the next page, with the name of the tool parameter that takes it. Name only keys that are in the rows shown; null otherwise."""

SHAPE_FORM = _form("rows_shape", "The keys of the listing rows.", {
    "price_key": {"type": ["string", "null"]}, "stock_key": {"type": ["string", "null"]},
    "count_key": {"type": ["string", "null"], "description": "the result's key holding the total number of matches, outside the rows"},
    "next_key": {"type": ["string", "null"], "description": "the result's key holding the next page token or number, outside the rows"},
    "next_param": {"type": ["string", "null"]}}, ["price_key", "stock_key", "count_key", "next_key", "next_param"])

NOTE_SYSTEM = """A line of text came back inside an online-store result (a note from the shopper, or the store's own words about a
call). You say what it is: a change the shopper makes to the order, an error or refusal from the store, or something
else. For a change: a new total allowed, conditions added (property and value, in the line's own words), conditions
dropped (property), listings now sold out or otherwise not buyable (ids as written), a price that moved. For a store
refusal naming a listing, give its id. Keep the summary to one short line."""

NOTE_FORM = _form("note_read", "What the line means.", {
    "kind": {"type": "string", "enum": ["change", "error", "other"]},
    "budget": {"type": ["number", "null"]},
    "added": {"type": "array", "items": {"type": "object", "properties": {"item": {"type": "integer"}, "property": {"type": "string"}, "value": {"type": "string"}}, "required": ["item", "property", "value"]}},
    "dropped": {"type": "array", "items": {"type": "string"}},
    "gone": {"type": "array", "items": {"type": "string"}},
    "sold_out": {"type": "boolean"}, "price_moved": {"type": "boolean"},
    "summary": {"type": "string"}}, ["kind", "budget", "added", "dropped", "gone", "sold_out", "price_moved", "summary"])

TRAIL_SYSTEM = """A shopper's condition points to a value found somewhere in the store: on the page of another listing, described by
where to look. From the records shown (OPENED pages with their attributes, ROWS seen only in results), follow each
pointer: name the chain of listing ids that leads to the page holding the value (the page pointed at, and any page it
sends you on to), say whether the chain is complete (every link opened) and, when it is, the value found, word for
word. A link not among the records is unknown."""

TRAIL_FORM = _form("pointer_trail", "Each pointer's chain.", {
    "chains": {"type": "array", "items": {"type": "object", "properties": {
        "pointer": {"type": "integer"}, "links": {"type": "array", "items": {"type": "string"}}, "done": {"type": "boolean"}, "value": {"type": ["string", "null"]}},
        "required": ["pointer", "links", "done", "value"]}}}, ["chains"])

CHECK_SYSTEM = """Before an online-store assistant places an order, you read each cart line's newest store record against the
shopper's request and the conditions now in force for its product, from the records shown. For each line say whether
it is a wanted kind of product (where the store files listings under categories the category decides, whatever the
title says) and which wanted product (1 for the first product listed on the sheet), give every condition of that
product its verdict — met: a record states the wanted value; other: a record states a different value (quote it);
silent: no record speaks to it — and say whether a record shows the line sold out. Name the other listings shown that satisfy every condition for a product: the replacement
for a line that fails, or, under a pick rule, one that beats the cart line on that rule (cheaper under cheapest; a
larger or a smaller measure under most or least).
Reading: an attribute named for the property decides; one list element is enough; units are converted; edges are
inclusive; a key named for a different property decides nothing. Without such a key, the value under another
attribute, the title or the description counts, but a number of another kind does not. A pointer condition counts as
met unless the records show that same value. Quantities, totals, messages and the steps the shopper asked for are not
yours to judge."""

def _silent(problem: str) -> bool:
    """A problem that only says a record is silent (nothing shown, confirmed or on record), or speaks of messages: not
    a failed condition under the check's contract."""

    p = _fold(problem)
    return ("not shown" in p or "cannot be confirmed" in p or "cannot be verified" in p or "no record" in p or "not confirmed" in p
            or "not verified" in p or "messages left" in p)


CHECK_FORM = _form("cart_check", "What the records state about each cart line.", {
    "lines": {"type": "array", "items": {"type": "object", "properties": {
        "id": {"type": "string"}, "wanted_kind": {"type": "boolean", "description": "the line is one of the wanted kinds of product"},
        "product": {"type": "integer", "description": "which wanted product: 1 for the first product on the sheet, 2 for the second"},
        "conditions": {"type": "array", "items": {"type": "object", "properties": {
            "property": {"type": "string"}, "stated": {"type": "string", "description": "the value a record states, word for word; empty when silent"},
            "verdict": {"type": "string", "enum": ["met", "other", "silent"]}}, "required": ["property", "stated", "verdict"]}},
        "sold_out_shown": {"type": "boolean"}}, "required": ["id", "wanted_kind", "product", "conditions", "sold_out_shown"]}},
    "better": {"type": "array", "items": {"type": "string"}, "description": "ids of listings shown that satisfy every condition for a product and would replace a failing line, or beat the cart line under the pick rule"}},
    ["lines", "better"])

AUDIT_SYSTEM = """An online-store assistant is about to end a session without ordering. From the store records shown (opened
listings with their attributes, and rows seen only in results), say for each wanted product which listings, if any,
satisfy every condition now in force (opened ones only), and which rows seen only in results are worth opening because
they could satisfy them. An attribute named for the property decides; a key named for a different property decides
nothing; a condition no record speaks to does not disqualify a listing; a pointer condition counts as met unless the
records show that same value. Name ids exactly as shown."""

AUDIT_FORM = _form("decline_audit", "Fits and rows worth opening.", {
    "fits": {"type": "array", "items": {"type": "object", "properties": {"item": {"type": "integer"}, "ids": {"type": "array", "items": {"type": "string"}}}, "required": ["item", "ids"]}},
    "open": {"type": "array", "items": {"type": "string"}}}, ["fits", "open"])

VERDICT_SYSTEM = """You check one store listing for a shopper against numbered conditions, from the listing's own store record.
For each condition say met, unmet or unknown. Met: the listing's attribute for that property states the value (any
spelling of the attribute's name; one element of a list is enough; the same word or a plain synonym; numbers compared
as numbers after converting units, a limit including its edge, a range wholly inside the limit), or, only when no
attribute is named for that property, the title or the description states it. Unmet: the attribute for that property
states a different value, or the title or description states a different one while no attribute states the wanted
one; for a condition that excludes a value, unmet when the listing states that value. Unknown: nothing states a value
for that property, or it could only be inferred. An attribute named for another property decides nothing. A condition
naming what the product must fit or suit (a vehicle, a model, a device, another chosen product) is unmet when the
record lists what it fits and the named one is not among them. Say whether the listing is the kind of product wanted: when the store files listings under categories, the category decides,
whatever the title says; without categories, the title does. Quote the deciding words (ten at most)."""

VERDICT_FORM = _form("listing_verdict", "One listing against the conditions.", {
    "is_item": {"type": "boolean"}, "in_stock": {"type": ["boolean", "null"]},
    "checks": {"type": "array", "items": {"type": "object", "properties": {
        "condition": {"type": "integer"}, "verdict": {"type": "string", "enum": ["met", "unmet", "unknown"]}, "quote": {"type": "string"}},
        "required": ["condition", "verdict"]}}}, ["is_item", "in_stock", "checks"])

GATE_SYSTEM = """A shopper asked an online-store assistant to do certain things as part of the order. You decide, from the record of
what was sent and done so far, which of the steps due now are already done, and for one step not yet done, the single
store call that does it (a message to the shopper, or a store tool call), or null when every step is done. A step is
done only when a record line shows it; a message step is done only by a message that carries what the step asks for,
sent after the cart last changed when the step belongs to the order. Name the record lines that prove each step."""

GATE_FORM = _form("process_gate", "The process steps' state.", {
    "done": {"type": "array", "items": {"type": "object", "properties": {"step": {"type": "integer"}, "lines": {"type": "array", "items": {"type": "integer"}}}, "required": ["step", "lines"]}},
    "call": {"type": ["object", "null"], "properties": {"tool": {"type": "string"}, "args": {"type": "object"}}}}, ["done", "call"])

CONDUCT_SYSTEM = """You are the assistant of one shopper in an online store, and you act only by calling the store's tools.
Each tool call is one step of the shopper's step allowance. After every result the code appends a sheet: what the
shopper wants (every condition in force), what the shopper has told you, the cart as the store holds it, what is not
buyable, and what is left. Trust the sheet over your memory. The session is graded as a whole: a purchase counts
only when it holds every wanted product at its quantity, each satisfying every condition in force, within the money
allowed; otherwise the session must end with the store's no-purchase tool the way the shopper asked.

The store: read each tool's parameters and use its argument names exactly. Searches can be narrow (a shelf filter, a
keyword mode, small pages, a page token); compare the total a result reports with the rows it shows and page on when
the answer depends on every row. Use short queries that lead with the value wanted and the product word; an empty
filtered search usually means a wrong filter value, not an empty store. Rows are summaries: the attributes live in the
listing's own page, and the sheet names the tool that shows them. Never open again a listing whose record you already
have unless the sheet says it changed. Messages to the shopper are sent by the code at the right moment; the steps
the shopper asked for at the order or the ending are done by the code when you call the ending tool, so never wait for
them and never end without a purchase because of them.

Fit: the listing's shelf or category decides its kind. For a condition, an attribute named for that
property decides: its spelling is free, one element of a list is enough, a synonym or a value containing the wanted
one counts, a different word does not; units are converted, limits are inclusive, a range must lie wholly inside the
wanted one. A key named for a different property decides nothing. Without a key for the property, the value under
another attribute, the title or the description counts, but a number of another kind does not. A condition whose
value points to another listing's page is met only after that page was opened and its value applied. A listing that
sends you to another page must be looked at. When one product must work with another, choose from what the pages
say, and change the other product before ending without a purchase. Wishes break ties only. No sold-out line; the
total within the money allowed.

Choosing: under a cheapest rule the pick is the lowest-priced of every listing that fits, so page through the results,
check candidates in rising price and use a price cap parameter when the search has one; under a most or least rule
compare every fitting listing on that measure, and one that does not state it cannot win; otherwise the first fit
will do. Put each product's chosen listing in the cart as soon as its page shows it fits (a sold-out line shows up on
carting); one line per product.

Changes: the shopper's words inside any result count from then on; the session goes on after a refused order; re-read
the cart lines before ordering when anything changed. Finishing: the cart holds exactly the chosen lines; view the cart
once; call the ordering tool alone in a reply of its own. End without a purchase only after real searches found
nothing that fits, with a short text. One tool per reply at the ending. A reply with no tool call does nothing."""


# ---------------------------------------------------------------------------
# The episode
# ---------------------------------------------------------------------------

_RECORD_KEY = re.compile(r"^([A-Za-z_][A-Za-z0-9_.]*)(?:: |=)(.*)$", re.S)
_PAIR_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_. -]*=")
_ATTR_LINE = re.compile(r"^([A-Za-z_][A-Za-z0-9_. -]{0,40}?)(?:: ?|=)(.*)$", re.S)
_ID_NAME = re.compile(r"(^|_)id$|^ref|code$|sku|number$", re.I)


def _typed(text: str) -> Any:
    if text == "None":
        return None
    if text in ("True", "False"):
        return text == "True"
    if re.fullmatch(r"-?\d+", text):
        return int(text)
    if re.fullmatch(r"-?\d+\.\d+", text):
        return float(text)
    return text


def _pairs(text: str) -> Any:
    """'a=b; c=d' as a dict; any other text as it is."""

    pieces = text.split("; ")
    if not _PAIR_KEY.match(pieces[0]):
        return text
    out: dict[str, str] = {}
    last = None
    for piece in pieces:
        if _PAIR_KEY.match(piece):
            last, value = piece.split("=", 1)
            last = last.strip()
            out[last] = value.strip()
        elif last is not None:
            out[last] += "; " + piece
    return out


def _attributes(value: Any) -> Any:
    """A list that reads as attributes (name and value objects, [name, value] pairs, 'name: value' lines) as a dict;
    anything else as it is."""

    if not isinstance(value, list) or len(value) < 2:
        return value
    pairs: list[tuple[str, Any]] = []
    for item in value:
        if isinstance(item, dict) and len(item) == 2:
            name, content = list(item.values())
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            name, content = item
        elif isinstance(item, str) and _ATTR_LINE.match(item):
            name, content = _ATTR_LINE.match(item).groups()
        else:
            return value
        if not isinstance(name, str) or not name or re.search(r"\d", name) or isinstance(content, (dict, list)):
            return value
        pairs.append((name.strip(), content.strip() if isinstance(content, str) else content))
    return dict(pairs)


def _record_text(text: str) -> Any:
    """'key: value' pieces joined by one separator, as a dict; a piece that opens no key belongs to the one before it.
    Three keys at least, and the first piece must open one."""

    if text.startswith("|") and text.endswith("|"):
        text = text.strip("| ")
    for sep in (" | ", "\n", "; ", ", "):
        if sep not in text:
            continue
        record: dict[str, Any] = {}
        last = None
        for part in text.split(sep):
            match = _RECORD_KEY.match(part)
            if match:
                last = match.group(1)
                record[last] = match.group(2)
            elif last is None:
                record = {}
                break
            else:
                record[last] += sep + part
        if len(record) >= 3 and any(_ID_NAME.search(k) or re.fullmatch(r"\S+", str(v)) for k, v in record.items()) and sum(" " not in k for k in record) == len(record):
            return record
    return None


def self_record(block: dict[str, Any]) -> bool:
    """A sub-object that describes the listing itself: a group of its fields, not its attribute table (whose values are all text)."""

    return any(not isinstance(v, str) for v in block.values()) or any(_ID_NAME.search(str(k)) for k in block)


def unstring(value: Any) -> Any:
    """A store's records read as plain objects whatever way it writes them: as text ('key: value | key: value', one per
    line, JSON), as columns and rows, or grouped under sub-objects; prices as amount objects; attribute blocks as lists.
    Everything else stays as it is."""

    if isinstance(value, list):
        return [unstring(v) for v in value]
    if isinstance(value, dict):
        lists = {k: v for k, v in value.items() if isinstance(v, list) and v}
        names = next((k for k, v in lists.items() if all(isinstance(x, str) for x in v)), None)
        rows = next((k for k, v in lists.items() if k != names and all(isinstance(x, list) for x in v)), None)
        if names and rows and all(len(r) == len(value[names]) for r in value[rows]):
            value = {**{k: v for k, v in value.items() if k not in (names, rows)}, rows: [dict(zip(value[names], r)) for r in value[rows]]}
        amount = [k for k, v in value.items() if re.search(r"amount|value", str(k), re.I) and isinstance(v, (int, float)) and not isinstance(v, bool)]
        if len(amount) == 1 and len(value) <= 3 and any(re.search(r"currency|unit", str(k), re.I) for k in value):
            return value[amount[0]]
        # (text fields of an object stay text; a string is a record only as a list item or as the one payload of a result)
        others = [v for v in value.values() if not (isinstance(v, list) and all(isinstance(x, str) for x in v))]
        out = {}
        for k, v in value.items():
            if isinstance(v, str):
                out[k] = unstring(v) if (len(others) == 1 and v is others[0]) or v[:1] == "{" else v
            elif isinstance(v, list) and len(others) == 1 and all(isinstance(x, str) for x in v):
                out[k] = v
            else:
                inner = unstring(v)
                out[k] = _attributes(v) if inner == v else inner
        if len(out) == 1:
            key, inner = next(iter(out.items()))
            if isinstance(value[key], str) and isinstance(inner, dict):
                return inner
        groups = [v for v in out.values() if isinstance(v, dict) and self_record(v)]
        if len(groups) >= 2 and not any(isinstance(v, str) and _ID_NAME.search(str(k)) for k, v in out.items()) and any(any(_ID_NAME.search(str(k)) and isinstance(v, str) for k, v in g.items()) for g in groups):
            flat: dict[str, Any] = {}
            for k, v in out.items():
                if isinstance(v, dict) and self_record(v):
                    flat.update(v)
                else:
                    flat[k] = v
            return flat
        return out
    if not isinstance(value, str):
        return value
    text = value.strip()
    if text[:1] in "[{" and text[-1:] in "]}":
        try:
            return unstring(json.loads(text))
        except ValueError:
            return value
    record = _record_text(value.strip("\n"))
    if record is None:
        return value
    out: dict[str, Any] = {}
    for k, v in record.items():
        parsed = _pairs(v)
        if parsed == v and v[:1] == "{":
            try:
                parsed = json.loads(v)
            except ValueError:
                parsed = v
        out[k] = parsed if parsed != v else _typed(v)
    return out


class Kernel:
    def __init__(self, problem: dict[str, Any]) -> None:
        self.started = time.monotonic()
        environment = problem["environment"]
        self.binding = dict(environment.get("binding") or {})
        self.view = environment.get("policy_view") or {}
        self.request = str(self.view.get("query") or "")
        self.tools = [t for t in self.view.get("tools") or [] if isinstance(t, dict) and (t.get("function") or {}).get("name")]
        self.schemas = {t["function"]["name"]: (t["function"].get("parameters") or {"type": "object", "properties": {}}) for t in self.tools}
        self.max_steps = int(self.view.get("max_steps") or 30)
        self.step = 0
        self.finished = bool(self.view.get("finished"))
        self.trace: list[dict[str, Any]] = []
        self.cost = 0.0
        self.calls = 0                      # model calls of any kind
        self.turns = 0                      # conductor turns
        self.call_costs: dict[str, float] = {}
        self.spent_out = False
        self.broke = False                  # the run's key is out of credit: no model call can be made
        # the sheet
        self.items: list[dict[str, Any]] = []     # name, quantity, musts (property: constraint), told, free
        self.budget: float | None = None
        self.unit: str | None = None
        self.in_stock_only = True
        self.pick: str | None = None              # cheapest / most / least / None, with pick_measure
        self.pick_measure: str | None = None
        self.msg_limit: int | None = None
        self.messages_sent = 0
        self.step_cap: int | None = None
        self.process: list[dict[str, Any]] = []   # the request's own steps: text, when (choose / order / end), messages, done
        self.lookups: list[dict[str, Any]] = []   # conditions whose value is on another listing's page: property, item, where, value
        self.rules: list[str] = []
        self.ask_first = False
        self.quiet_decline: bool | None = None
        self.roles: dict[str, dict[str, Any]] = {}   # search / open / compare / ask / cart / cart_view / buy / refuse -> tool, args
        self.version = 0                          # bumps whenever what is wanted changes
        self.cart: dict[str, int] = {}            # the cart as the code books it: listing id -> quantity
        self.cart_at = -1                         # the step at which the cart last changed
        self.records: list[tuple[str, dict[str, Any], Any]] = []
        self.seen: dict[str, dict[str, Any]] = {}   # listing id -> what the rows and pages showed (title, price, stock)
        self.gone: set[str] = set()               # sold out
        self.refused: set[str] = set()            # the store refused them (a cart add or an order named them)
        self.notes: list[str] = []                # changes the shopper announced, in order
        self.asked: list[tuple[str, str]] = []    # (question, reply)
        self.refusal_step: int | None = None        # the step of the last refused checkout
        self.held: dict[str, int] = {}            # how often each kind of hold was used
        self.done_log: list[str] = []             # the record the process gate reads: messages sent, calls made
        self.full: dict[str, dict[str, Any]] = {}   # listing id -> its own opened record (the newest)
        self.raw_rows: list[dict[str, Any]] = []   # result rows as they came, for re-reading them under a corrected id key
        self.id_key: str | None = None            # the row key that holds the listing id
        self.category_key: str | None = None      # the row key whose values are the store's category labels
        self.asked_once = False                   # the interview was held
        self.stock_words: dict[str, bool | None] = {}
        self.bundle_ok = False                    # one property per message
        self.beacon_done = False                  # the diagnostic search was sent
        self.gate_msgs = 0                        # messages the process gate sent (they count against the reserve, not the questions)
        self.widened = False
        self.heard_text = ""                      # what the interview settled, as shown to the conductor
        self.price_key: str | None = None
        self.stock_key: str | None = None
        self.next_key: str | None = None
        self.count_key: str | None = None
        self.pages_read = 0
        self.trails = 0
        self.trail_holds = 0
        self.seen_notes: set[str] = set()         # loose lines already read
        self.note_reads = 0
        self.record_keys: set[str] = set()        # keys the rows and the attribute blocks use (their strings are listing data)
        self.row_fields: set[str] = set()         # keys the listing rows use (a page's copy of them names the listing, not a property)
        self.attr_tool: str | None = None         # the tool whose record of a listing holds its attribute table
        self.turn_cart_ids: set[str] = set()      # listings a cart call of the current conductor reply already touched
        self.compare_room: int | None = None      # ids a compare call takes at most (the schema's, or learned from a call that returned nothing)
        self.sold_note: str | None = None         # a note that was nothing but a sell-out: the same words mean the same later
        self.results: dict[str, tuple[int, Any]] = {}   # call signature -> (version, observation), for reads that cannot change
        self.recent: list[str] = []               # the last call signatures
        self.evidence_text = ""
        self.last_pages: list[Any] = []
        self.verdicts: dict[str, dict[str, Any]] = {}   # listing id -> the code check's verdict at a version
        self.search_shape: tuple[str, str, dict[str, Any]] | None = None   # (search tool, text parameter, fixed args) of a search that showed rows
        self.heard_searched: set[str] = set()
        self.next_param: str | None = None
        self.shape_asked = False
        self.gate_ok: set[tuple[str, str, int]] = set()   # (stage, cart mark, version) the process gate passed
        self.checks: dict[tuple[str, int], dict[str, Any]] = {}   # (cart ids, version) -> the heavy check's verdict
        self.leads: list[str] = []                # listings the audit found fitting
        self.confirmed: str | None = None         # a lead an independent check passed
        self.audits = 0
        self.built: dict[str, int] = {}           # the fullest cart booked so far
        self.viewed_at = -1                       # the step of the last cart view
        self.reread: set[tuple[str, int]] = set() # (listing, refusal step) re-read before the next order

    # -- clocks and budgets ---------------------------------------------------------------------
    def seconds_left(self) -> float:
        return _TASK_SECONDS - (time.monotonic() - self.started)

    def wall_left(self) -> float:
        return _WALL_SECONDS - (time.monotonic() - self.started)

    def wait_for_model(self) -> float:
        closing = self.seconds_left() - _CLOSE_SECONDS + 15.0
        return max(4.0, min(_CALL_SECONDS, closing if closing > 0 else self.wall_left() - 20.0, self.wall_left() - 20.0))

    def steps_left(self) -> int:
        return self.max_steps - self.step

    def msgs_left(self) -> int | None:
        return None if self.msg_limit is None else max(0, self.msg_limit - self.messages_sent)

    def reserve(self) -> int:
        """Steps kept for the ending: the ending call, and one more while a process step at the order still needs a
        message."""

        return 2 if any(p.get("when") == "order" and p.get("messages") for p in self.process) else 1

    def helpers_may_run(self) -> bool:
        return self.cost < _SPEND_SOFT and self.calls < _ALL_CALLS and not self.broke

    # -- inference -----------------------------------------------------------------------------
    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, models: list[str], max_tokens: int,
             forced: str | None = None, think: bool = True) -> dict[str, Any] | None:
        if self.broke or self.calls >= _ALL_CALLS:
            return None
        if self.cost >= _SPEND_HARD:
            self.spent_out = True
            return None
        head = str(messages[0].get("content"))[:80] if messages else ""
        if self.cost + self.call_costs.get(head, 0.0) > _SPEND_HARD + 0.001:
            # (the same kind of call cost this much last time: it would carry the task past its limit)
            self.spent_out = True
            return None
        self.calls += 1
        for pause in _RETRY_PAUSES:
            if pause:
                # (every model refused or stalled: a pause, then the same round again, while the task has the time)
                if self.seconds_left() < pause + _CLOSE_SECONDS + 30 or self.wall_left() < pause + 45:
                    break
                self.trace.append({"inference_pause": pause})
                time.sleep(pause)
            message = self.chat_round(messages, tools, models, max_tokens, forced, think)
            if message is not None or self.broke:
                return message
        return None

    def chat_round(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, models: list[str], max_tokens: int,
                   forced: str | None, think: bool) -> dict[str, Any] | None:
        head = str(messages[0].get("content"))[:80] if messages else ""
        for model in models:
            if self.seconds_left() < 12 or self.wall_left() < 25:
                return None
            _proxy.timeout = int(self.wait_for_model())
            # (a model that reasons before answering needs room for it in the answer budget: a short limit comes back empty;
            #  the larger models of the family always reason and take no required answer function)
            always_thinks = model.startswith("anthropic/") and not model.startswith("anthropic/claude-haiku")
            thinks = think or always_thinks
            room = max(max_tokens, 1600) if thinks else max_tokens
            body: dict[str, Any] = {"model": model, "messages": messages, "temperature": 0, "max_tokens": room}
            if tools:
                body["tools"] = tools
                body["tool_choice"] = {"type": "function", "function": {"name": forced}} if forced and not always_thinks else "auto"
            if model.startswith("anthropic/") and messages and isinstance(messages[0].get("content"), str):
                body["messages"] = [{**messages[0], "content": [{"type": "text", "text": messages[0]["content"], "cache_control": {"type": "ephemeral"}}]}] + messages[1:]
                last = max((i for i, m in enumerate(body["messages"]) if i > 0 and isinstance(m.get("content"), str) and m["content"]), default=None)
                if last is not None and len(body["messages"]) > 2:
                    body["messages"] = body["messages"][:last] + [{**body["messages"][last], "content": [{"type": "text", "text": body["messages"][last]["content"], "cache_control": {"type": "ephemeral"}}]}] + body["messages"][last + 1:]
            if not thinks and model.startswith("anthropic/"):
                body["reasoning"] = {"enabled": False}
            try:
                data = _proxy.post("/inference/chat/completions", json_data=body)
            except Exception as error:  # noqa: BLE001
                text = repr(error)
                self.trace.append({"inference_error": model, "error": text[:200]})
                if "402" in text:
                    self.broke = True
                    return None
                if "reasoning" not in body:
                    continue
                body.pop("reasoning")
                try:
                    data = _proxy.post("/inference/chat/completions", json_data=body)
                except Exception as again:  # noqa: BLE001
                    self.trace.append({"inference_error": model, "error": repr(again)[:200]})
                    continue
            if not isinstance(data, dict) or not data.get("choices"):
                self.trace.append({"inference_error": model, "error": str(data)[:200]})
                continue
            spent = (data.get("usage") or {}).get("cost")
            if isinstance(spent, (int, float)) and not isinstance(spent, bool):
                self.cost += spent
                self.call_costs[head] = float(spent)
            message = (data["choices"][0] or {}).get("message") or {}
            if message.get("tool_calls") or message.get("content"):
                return message
        return None

    def read(self, system: str, user: str, form: dict[str, Any], max_tokens: int = 1200, models: list[str] | None = None,
             think: bool | None = None) -> dict[str, Any]:
        """A reader's structured answer through a function call it is told to make."""

        name = form["function"]["name"]
        system = system + f"\n\nAnswer by calling {name} once."
        short = (max_tokens <= 400) if think is None else not think
        message = self.chat([{"role": "system", "content": system}, {"role": "user", "content": user}], [form], models or LIGHT, max_tokens,
                            name if short else None, think=not short)
        if not message:
            return {}
        for call in message.get("tool_calls") or []:
            parsed = _json_object(((call or {}).get("function") or {}).get("arguments"))
            if parsed:
                return parsed
        return _json_object(message.get("content")) or {}

    # -- the world -----------------------------------------------------------------------------
    def world(self, actions: list[dict[str, Any]]) -> list[Any]:
        """Run calls as consecutive steps in one envelope; the observations in order (fewer when the session ended)."""

        if self.finished or not actions:
            return []
        envelope: dict[str, Any] = {**self.binding, "step": self.step + 1}
        if len(actions) == 1:
            envelope["action"] = actions[0]
        else:
            envelope["actions"] = actions
        reply = None
        for _ in range(3):
            if self.wall_left() < 4:
                break
            _proxy.timeout = int(max(3.0, min(60.0, self.wall_left() - 2.0)))
            try:
                reply = _proxy.post("/environment/call", json_data=envelope)
            except Exception as error:  # noqa: BLE001
                text = repr(error)
                self.trace.append({"world_error": text[:200]})
                expected = re.search(r"expected (?:step )?(\d+)", text)
                if expected:
                    # (the store counts the steps: the envelope is sent again at the step it expects)
                    self.step = int(expected.group(1)) - 1
                    envelope["step"] = self.step + 1
                elif re.search(r"finish|finaliz|quarantin", text, re.I):
                    self.finished = True
                    return []
                reply = None
            if isinstance(reply, dict):
                break
            time.sleep(1.0)
        if not isinstance(reply, dict):
            return []
        if isinstance(reply.get("step"), int):
            self.step = int(reply["step"])
        else:
            self.step += len(actions)
        self.finished = bool(reply.get("finished"))
        many = reply.get("observations")
        observations = [unstring(o) for o in (many if isinstance(many, list) else [reply.get("observation")])]
        for action, observation in zip(actions, observations):
            self.records.append((action["name"], action["args"], observation))
            self.trace.append({"step": self.step, "action": action, "observation": observation, "finished": self.finished})
        return observations

    # -- roles ----------------------------------------------------------------------------------
    def role_tool(self, role: str) -> str | None:
        return (self.roles.get(role) or {}).get("tool")

    def role_args(self, role: str) -> dict[str, Any]:
        return dict((self.roles.get(role) or {}).get("args") or {})

    def is_role(self, role: str, name: str, args: dict[str, Any]) -> bool:
        """The call does the role: its tool, and the role's fixed arguments (when any) as given."""

        spec = self.roles.get(role)
        if not spec or spec.get("tool") != name:
            return False
        fixed = spec.get("args") or {}
        return all(args.get(key) == value for key, value in fixed.items())

    def ending(self, name: str, args: dict[str, Any]) -> str | None:
        """'buy' or 'refuse' when the call ends the episode."""

        for role in ("buy", "refuse"):
            if self.is_role(role, name, args):
                return role
        return None

    def ending_args(self, role: str) -> dict[str, Any]:
        """The ending call's arguments: the role's fixed ones, the required ones filled."""

        tool = self.role_tool(role)
        args = self.role_args(role)
        props = (self.schemas.get(tool) or {}).get("properties") or {}
        for name in (self.schemas.get(tool) or {}).get("required") or []:
            if name in args:
                continue
            spec = props.get(name) or {}
            kind = spec.get("type")
            kinds = kind if isinstance(kind, list) else [kind]
            if "enum" in spec:
                args[name] = spec["enum"][0]
            elif "string" in kinds:
                args[name] = "done" if role == "buy" else "Nothing in the store meets every condition, so I am not ordering."
            elif "integer" in kinds or "number" in kinds:
                args[name] = spec.get("minimum", 0)
            elif "boolean" in kinds:
                args[name] = True
            elif "array" in kinds:
                args[name] = []
            elif "object" in kinds:
                args[name] = {}
        if role == "refuse":
            # (a quiet decline text in the first free string parameter, so the ending says what it is)
            text_param = next((k for k, p in props.items() if isinstance(p, dict) and p.get("type") == "string" and "enum" not in p and k not in args), None)
            if text_param:
                args[text_param] = "Nothing in the store meets every condition, so I am not ordering."
        args, _ = schema_problems(args, self.schemas.get(tool) or {})
        return args

    # -- the cart book -------------------------------------------------------------------------
    def ids_in(self, args: Any) -> list[str]:
        """Listing ids named by the arguments: values that are ids of listings seen."""

        found: list[str] = []
        def walk(node: Any) -> None:
            if isinstance(node, str):
                if node in self.seen and node not in found:
                    found.append(node)
            elif isinstance(node, dict):
                for v in node.values():
                    walk(v)
            elif isinstance(node, list):
                for v in node:
                    walk(v)
        walk(args)
        return found

    def book_cart(self, name: str, args: dict[str, Any]) -> None:
        """The cart book after a cart call: a whole-cart array replaces it; an enum parameter whose second value is
        used means a removal; an integer with minimum 0 and no mode sets the line; otherwise the call adds."""

        props = (self.schemas.get(name) or {}).get("properties") or {}
        ids = self.ids_in(args)
        if not ids:
            # (the store took the call, so the string argument that is not a mode names the listing)
            ids = [v for k, v in args.items() if isinstance(v, str) and isinstance(props.get(k), dict) and "enum" not in props[k] and len(v) >= 4]
            for rid in ids:
                self.seen.setdefault(rid, {})
        before = dict(self.cart)
        arrays = [k for k, v in args.items() if isinstance(v, list) and isinstance(props.get(k), dict) and props[k].get("type") == "array"]
        if arrays and all(isinstance(x, dict) for x in args[arrays[0]]):
            self.cart = {}
            for line in args[arrays[0]]:
                line_ids = self.ids_in(line)
                count = next((v for v in line.values() if isinstance(v, int) and not isinstance(v, bool)), 1)
                if line_ids and count > 0:
                    self.cart[line_ids[0]] = count
        elif ids:
            modes = [(k, v) for k, v in args.items() if isinstance(props.get(k), dict) and "enum" in props[k] and isinstance(v, str)]
            counts = [(k, v) for k, v in args.items() if isinstance(v, int) and not isinstance(v, bool) and isinstance(props.get(k), dict)
                      and props[k].get("type") == "integer"]
            count = counts[0][1] if counts else 1
            removal = any(props[k]["enum"].index(v) == 1 for k, v in modes if v in props[k]["enum"] and len(props[k]["enum"]) >= 2)
            sets = bool(counts) and not modes and props[counts[0][0]].get("minimum") == 0
            for rid in ids:
                if removal or (sets and count <= 0):
                    self.cart.pop(rid, None)
                elif sets:
                    self.cart[rid] = count
                else:
                    self.cart[rid] = self.cart.get(rid, 0) + max(1, count)
        if self.cart != before:
            self.cart_at = self.step

    def cart_from_view(self, observation: Any, whole: bool = True) -> None:
        """The cart book replaced by what a cart view shows, when it shows lines with seen ids."""

        lines: list[tuple[str, int]] = []
        def walk(node: Any) -> None:
            if isinstance(node, dict):
                ids = [v for v in node.values() if isinstance(v, str) and v in self.seen]
                price = self.price_of(ids[0]) if ids else None
                ints = {k: v for k, v in node.items() if isinstance(v, int) and not isinstance(v, bool) and 0 < v < 1000
                        and not (price is not None and abs(v - price) < 1e-9)}      # (the line's price is never its count)
                counts = sorted(ints.values())      # (the smallest remaining integer: a count is small, a code or an amount is not)
                zeros = [v for v in node.values() if isinstance(v, int) and not isinstance(v, bool) and v == 0]
                if ids:
                    # (a line with a zero and no positive count is gone: the store answered a removal with the count left)
                    lines.append((ids[0], counts[0] if counts else 0 if zeros else 1))
                    return
                for v in node.values():
                    walk(v)
            elif isinstance(node, list):
                for v in node:
                    walk(v)
        walk(observation)
        text = _fold(observation)
        if lines:
            # (a cart call answers for the lines it touched: those are set or dropped, the rest stand; a view shows the whole cart)
            new = {rid: n for rid, n in lines if n > 0} if whole else {**{r: n for r, n in self.cart.items() if r not in dict(lines)}, **{rid: n for rid, n in lines if n > 0}}
            if new != self.cart:
                self.cart = new
                self.cart_at = self.step
        elif re.search(r"\b(empty|no items|nothing)\b", text) and not re.search(r"\b(error|invalid|unknown)\b", text):
            if self.cart:
                self.cart = {}
                self.cart_at = self.step

    # -- the record book ------------------------------------------------------------------------
    @staticmethod
    def dict_lists(observation: Any, least: int = 2) -> list[list[dict[str, Any]]]:
        """Every list of `least` or more dicts inside the observation, outer first."""

        found: list[list[dict[str, Any]]] = []
        def walk(node: Any) -> None:
            if isinstance(node, list):
                if len(node) >= least and all(isinstance(x, dict) for x in node):
                    found.append(node)
                for v in node:
                    walk(v)
            elif isinstance(node, dict):
                for v in node.values():
                    walk(v)
        walk(observation)
        return found

    @staticmethod
    def scalar_count(node: Any) -> int:
        return sum(1 for v in node.values() if isinstance(v, (str, int, float, bool)) and v not in ("", None)) if isinstance(node, dict) else 0

    @classmethod
    def facts_of(cls, record: Any) -> dict[str, Any]:
        """The attributes block of a listing record: the nested dict with the most scalar values (two at least)."""

        best: dict[str, Any] = {}
        def walk(node: Any, depth: int) -> None:
            nonlocal best
            if isinstance(node, dict):
                if depth > 0 and cls.scalar_count(node) >= 2 and cls.scalar_count(node) > cls.scalar_count(best):
                    best = node
                for v in node.values():
                    walk(v, depth + 1)
            elif isinstance(node, list):
                for v in node:
                    walk(v, depth + 1)
        walk(record, 0)
        if not best and isinstance(record, dict):
            # (a flat record: its short text values that are not links; ids, titles and numbers are not attributes)
            flat = {str(k): v for k, v in record.items() if isinstance(v, str) and 0 < len(v) <= 60 and not v.startswith("http")
                    and not re.search(r"name|title", str(k), re.I) and not re.search(r"(^|_)id$|_id_|^id_|identifier", str(k), re.I)}
            if len(flat) >= 2:
                best = flat
        return {str(k): v for k, v in best.items() if isinstance(v, (str, int, float, bool)) and v not in ("",)}

    @classmethod
    def attributes_of(cls, record: Any) -> dict[str, Any]:
        """The nested attribute block of a record (the dict with the most scalar values, two at least); empty for a
        flat page."""

        best: dict[str, Any] = {}
        def walk(node: Any, depth: int) -> None:
            nonlocal best
            if isinstance(node, dict):
                if depth > 0 and cls.scalar_count(node) >= 2 and cls.scalar_count(node) > cls.scalar_count(best):
                    best = node
                for v in node.values():
                    walk(v, depth + 1)
            elif isinstance(node, list):
                for v in node:
                    walk(v, depth + 1)
        walk(record, 0)
        return {str(k): v for k, v in best.items() if isinstance(v, (str, int, float, bool)) and v not in ("",)}

    def facts(self, record: Any) -> dict[str, Any]:
        """A record's attributes: its nested block; where the store shows attributes in a table, a flat page carries
        none; otherwise the page's short text values that are not the rows' own fields (a page repeats the row's brand,
        shelf and seller: those name the listing, not a property to choose by)."""

        block = self.attributes_of(record)
        if block:
            return block
        if self.attr_tool is not None:
            return {}
        return {k: v for k, v in self.facts_of(record).items() if k not in self.row_fields}

    def stock_flag(self, value: Any) -> bool | None:
        """A row's stock field as can be bought or not: a number by its size, a word by one reading per distinct word."""

        if isinstance(value, bool) or value is None:
            return value
        if isinstance(value, (int, float)):
            return value > 0
        text = _fold(value)
        if text not in self.stock_words and self.helpers_may_run():
            said = self.read(STOCK_SYSTEM, "Value: " + text[:80], STOCK_FORM, 60, LIGHT).get("can_buy")
            self.stock_words[text] = said if isinstance(said, bool) else None
        return self.stock_words.get(text)

    def learn_rows(self, observation: Any) -> list[str]:
        """Listing rows from a result: the id key is learned from the key whose values are later passed to the
        opening tools, before that from a key named for an id; the category key from the key whose values are the
        store's shelf labels. Returns the new ids."""

        new: list[str] = []
        lists = self.dict_lists(observation)
        # (a single-row result counts once the id key is known, or when a key is named for an id)
        lists += [rows for rows in self.dict_lists(observation, 1) if len(rows) == 1 and (isinstance(rows[0].get(self.id_key), str) if self.id_key
                  else any(isinstance(v, str) and v and re.search(r"(^|_)id($|_)|identifier|^sku$|_sku$|^code$", k, re.I) for k, v in rows[0].items()))]
        for rows in lists:
            keys = [k for k in rows[0] if all(isinstance(r.get(k), str) and r.get(k) for r in rows)]
            if not keys:
                continue
            if self.id_key not in rows[0]:
                unique = [k for k in keys if len({r[k] for r in rows}) == len(rows)]
                named = [k for k in unique if re.search(r"(^|_)id($|_)|identifier|^sku$|_sku$|^code$", k, re.I)]
                guess = named[0] if named else next((k for k in unique if all(len(r[k]) <= 40 and " " not in r[k] for r in rows)), None)
                if not guess:
                    continue
                key = guess
            else:
                key = self.id_key
            self.raw_rows.extend(rows[:60])
            for r in rows[:60]:
                self.record_keys.update(str(k) for k in r)
                self.row_fields.update(str(k) for k in r)
            if not self.shape_asked and self.helpers_may_run():
                self.shape_asked = True
                shape = self.read(SHAPE_SYSTEM, "Result rows:\n" + _short(rows[:3], 2500) + "\n\nTool parameters:\n" + self.tool_list()[:2000], SHAPE_FORM, 300, LIGHT)
                for field in ("price_key", "stock_key"):
                    value = shape.get(field)
                    if isinstance(value, str) and value in rows[0]:
                        setattr(self, field, value)
                top = observation if isinstance(observation, dict) else {}
                for field in ("count_key", "next_key"):
                    value = shape.get(field)
                    if isinstance(value, str) and value in top:
                        setattr(self, field, value)
                if isinstance(shape.get("next_param"), str) and self.next_key:
                    self.next_param = shape["next_param"]
                self.trace.append({"rows_shape": {"id": key, "price": self.price_key, "stock": self.stock_key, "next": self.next_key}})
            if self.category_key is None:
                labels = self.enum_labels()
                wanted = set()
                for it in self.items:
                    wanted |= set(self.stem(w) for w in re.findall(r"[a-z0-9]+", _fold(it.get("name")) + " " + _fold(it.get("shelf"))) if len(w) > 2)
                usable = []
                for k in keys:
                    values = [r[k] for r in rows]
                    distinct = set(_fold(v) for v in values)
                    if k != key and len(distinct) <= 40 and all(len(v) <= 60 for v in values):
                        usable.append((k, values, distinct))
                for k, values, distinct in usable:
                    # (the key whose values are the shelf labels a tool's schema lists)
                    if labels and sum(1 for v in values if _fold(v) in labels) >= 0.8 * len(values):
                        self.category_key = k
                        break
                if self.category_key is None:
                    for k, values, distinct in usable:
                        # (no such list: the key whose values share a word with a wanted item on most rows, and repeat)
                        if wanted and sum(1 for v in values if {self.stem(w) for w in re.findall(r"[a-z0-9]+", _fold(v))} & wanted) * 2 >= len(values) and len(distinct) < len(values):
                            self.category_key = k
                            break
            for r in rows:
                rid = r.get(key)
                if not isinstance(rid, str) or not rid:
                    continue
                row = self.seen.setdefault(rid, {})
                row["row"] = r
                price = r.get(self.price_key) if self.price_key else None
                if isinstance(price, (int, float)) and not isinstance(price, bool):
                    row["price"] = float(price)
                elif isinstance(price, str) and re.fullmatch(r"\s*[^\d]{0,3}\d[\d,]*(\.\d+)?\s*[a-zA-Z]{0,5}\s*", price):
                    row["price"] = float(re.sub(r"[^\d.]", "", price))
                stock = self.stock_flag(r.get(self.stock_key)) if self.stock_key else None
                if isinstance(stock, bool):
                    if stock and row.get("stock") is False:
                        self.gone.discard(rid)      # (back in stock: a later row says so)
                    row["stock"] = stock
                    if not stock:
                        self.gone.add(rid)
                if self.category_key and isinstance(r.get(self.category_key), str):
                    row["category"] = r[self.category_key]
                title = next((v for k2, v in r.items() if isinstance(v, str) and k2 != key and re.search(r"name|title", k2, re.I)), None)
                if title:
                    row["title"] = title
                if rid not in new and "opened" not in row:
                    new.append(rid)
            if self.id_key is None:
                self.id_key = key
        return new

    @staticmethod
    def stem(word: str) -> str:
        return re.sub(r"(ies|es|s)$", "", word) if len(word) > 3 else word

    def enum_labels(self) -> set[str]:
        labels: set[str] = set()
        for schema in self.schemas.values():
            for p in (schema.get("properties") or {}).values():
                if isinstance(p, dict):
                    for e in p.get("enum") or []:
                        if isinstance(e, str):
                            labels.add(_fold(e))
        return labels

    def fix_id_key(self, args: dict[str, Any]) -> None:
        """An opening call's argument that is a row value under another key than the guessed id key: that key is the
        id key, and the rows are read again under it."""

        values = set()
        def walk(node: Any) -> None:
            if isinstance(node, str):
                values.add(node)
            elif isinstance(node, list):
                for v in node:
                    walk(v)
            elif isinstance(node, dict):
                for v in node.values():
                    walk(v)
        walk(args)
        if not values or any(v in self.seen and (self.seen[v].get("row") is not None) for v in values):
            return
        for row in self.raw_rows:
            for k, v in row.items():
                if isinstance(v, str) and v in values and k != self.id_key and sum(1 for r in self.raw_rows if r.get(k) == v) == 1:
                    # (the value names one row only under that key: an id, not a shelf label or a brand)
                    old = self.id_key
                    self.id_key = k
                    for r in self.raw_rows:
                        if isinstance(r.get(k), str) and r[k]:
                            entry = self.seen.setdefault(r[k], {})
                            entry["row"] = r
                            if self.category_key and isinstance(r.get(self.category_key), str):
                                entry["category"] = r[self.category_key]
                            title = next((t for k2, t in r.items() if isinstance(t, str) and k2 != k and re.search(r"name|title", k2, re.I)), None)
                            if title:
                                entry["title"] = title
                    if old:
                        for r in self.raw_rows:
                            stale = r.get(old)
                            if isinstance(stale, str) and stale in self.seen and stale not in self.full and stale not in self.cart:
                                self.seen.pop(stale, None)
                    self.trace.append({"id_key": {"was": old, "now": k}})
                    return

    def compare_param(self) -> str | None:
        compare = self.role_tool("compare")
        schema = self.schemas.get(compare) or {}
        return next((k for k, p in (schema.get("properties") or {}).items() if isinstance(p, dict) and p.get("type") == "array"), None) if compare else None

    def room_for_compare(self) -> int:
        if self.compare_room:
            return self.compare_room
        param = self.compare_param()
        spec = ((self.schemas.get(self.role_tool("compare")) or {}).get("properties") or {}).get(param) or {} if param else {}
        return spec.get("maxItems") if isinstance(spec.get("maxItems"), int) and spec["maxItems"] >= 1 else 4

    def learn_compare_room(self, name: str, args: dict[str, Any], found: list[str]) -> str:
        """A compare call that brought back no record for several ids: the store takes fewer per call; the next calls
        are cut to half that many (one at least)."""

        if not self.is_role("compare", name, args) or found:
            return ""
        ids = self.ids_in(args)
        if len(ids) < 2:
            return ""
        room = max(1, len(ids) // 2)
        if self.compare_room is None or room < self.compare_room:
            self.compare_room = room
            self.trace.append({"compare_room": room})
        return f"\n[compare] no listing record came back for {len(ids)} ids: this store's compare takes at most {self.compare_room} listings in one call; open them {self.compare_room} at a time."

    def learn_pages(self, args: dict[str, Any], observation: Any, name: str | None = None) -> list[str]:
        """The own record of each listing the call named, when the result holds it: the dict that carries the id,
        merged over what other tools gave of the same listing (a store may split a page over tools: text on one, the
        attribute table on another). The tool whose record holds an attribute table is remembered as the attributes
        tool; each listing remembers which tools have read it."""

        found: list[str] = []
        for rid in self.ids_in(args):
            record = self.own_record(observation, rid)
            if record is not None:
                kept = self.full.get(rid)
                self.full[rid] = {**kept, **record} if isinstance(kept, dict) else dict(record)
                self.record_keys.update(str(k) for k in self.facts_of(record))
                row = self.seen.setdefault(rid, {})
                row["opened"] = self.step
                if name:
                    row["by"] = sorted(set(row.get("by") or []) | {name})
                    if self.attr_tool is None and self.attributes_of(record):
                        self.attr_tool = name
                found.append(rid)
        return found

    def own_record(self, observation: Any, rid: str) -> dict[str, Any] | None:
        """The innermost dict that holds the id as one of its values."""

        best: dict[str, Any] | None = None
        def walk(node: Any) -> None:
            nonlocal best
            if isinstance(node, dict):
                if any(v == rid for v in node.values() if isinstance(v, str)):
                    best = node
                for v in node.values():
                    walk(v)
            elif isinstance(node, list):
                for v in node:
                    walk(v)
        walk(observation)
        return best

    # -- the interview -------------------------------------------------------------------------
    def item_of(self, rid: str) -> int:
        """Which wanted item a listing belongs to: by its category label's words against the items' names and shelves;
        the first item when nothing matches."""

        label = _fold((self.seen.get(rid) or {}).get("category"))
        if label and len(self.items) > 1:
            words = {self.stem(w) for w in re.findall(r"[a-z0-9]+", label)}
            best, score = 1, 0
            for k, item in enumerate(self.items, 1):
                own = {self.stem(w) for w in re.findall(r"[a-z0-9]+", _fold(item.get("name")) + " " + _fold(item.get("shelf")))}
                hit = len(words & own)
                if hit > score:
                    best, score = k, hit
            return best
        return 1

    def key_menu(self) -> dict[int, list[tuple[str, int, list[str]]]]:
        """Per item, the attribute keys its opened listings carry, with carry counts and sample values; keys that are
        free text, variant lists, identifiers, twins of another key, or already settled are left out."""

        groups: dict[int, list[dict[str, Any]]] = {}
        for rid, record in self.full.items():
            facts = self.facts(record)
            if facts:
                groups.setdefault(self.item_of(rid), []).append(facts)
        menu: dict[int, list[tuple[str, int, list[str]]]] = {}
        for k, blocks in groups.items():
            settled = {_fold(p) for p in list(self.items[k - 1].get("musts") or {}) + list(self.items[k - 1].get("told") or {}) + list(self.items[k - 1].get("free") or [])}
            carry: dict[str, list[str]] = {}
            for facts in blocks:
                for key, v in facts.items():
                    carry.setdefault(key, []).append(str(v))
            kept: list[tuple[str, int, list[str]]] = []
            for key, values in sorted(carry.items(), key=lambda kv: (-len(kv[1]), kv[0])):
                name = _fold(key.replace("_", " "))
                if not name or name in settled or any(name in p or p in name for p in settled if len(p) > 3):
                    continue
                distinct = list(dict.fromkeys(values))
                mean = sum(len(v) for v in values) / len(values)
                if mean > 40 or sum(1 for v in values if " / " in v or ", " in v) * 2 >= len(values):
                    continue
                if len(values) >= 4 and len(distinct) == len(values) and (mean > 12 or all(re.search(r"\d", v) and " " not in v for v in values)):
                    continue        # (identifiers: every listing its own value, long or code-like)
                if any(_fold(other.replace("_", " ")) != name and (name in _fold(other.replace("_", " ")) or _fold(other.replace("_", " ")) in name)
                       and carry[other] == values and len(other) < len(key) for other in carry):
                    continue
                if len(values) >= 4 and max(values.count(v) for v in distinct) * 100 >= 85 * len(values):
                    continue        # (one value on nearly every listing: nothing to choose between)
                measured = sum(1 for v in values if re.match(r"^\s*[\d.,/x -]+\s*[a-z\"']{0,6}\s*$", v)) * 2 > len(values)
                kept.append((measured, key, len(values), distinct[:4]))
            kept.sort(key=lambda x: -x[2])
            menu[k] = [(key, count, samples) for _, key, count, samples in kept[:10]]
        return menu

    def questions_left(self) -> int:
        """Messages the interview may use: the allowance less what was sent, less what the request's own steps still
        need (those not yet proven done)."""

        limit = self.msg_limit if self.msg_limit is not None else 4      # (no stated allowance: four questions)
        reserve = sum(p["messages"] for p in self.process if not p.get("done"))
        return max(0, limit - self.messages_sent - max(0, reserve - self.gate_msgs))

    def interview_due(self) -> bool:
        if self.asked_once or not self.roles.get("ask") or self.questions_left() <= 0 or self.steps_left() <= 8 or self.broke:
            return False
        # (while a listing tool is still untried the count takes attribute tables only: a text page is not a record to
        #  plan questions from, and widen finds the table tool before the plan)
        tried = {t for row in self.seen.values() for t in (row.get("by") or [])}
        untried = [t for t in self.listing_tools() if t not in tried]
        opened = sum(1 for rid in self.full if (self.attributes_of(self.full[rid]) if (self.attr_tool or untried) else self.facts(self.full[rid])))
        return opened >= 3 * len(self.items) + 1 or (opened >= 1 and self.step >= 4) or self.step >= max(8, self.max_steps // 4)

    def open_with(self, tool: str, rids: list[str]) -> list[str]:
        """One call of a listing tool on the ids (through its string id parameter, or its array one), absorbed; the
        ids whose record came back from it."""

        schema = self.schemas.get(tool) or {}
        props = schema.get("properties") or {}
        role = next((r for r in ("open", "compare") if self.role_tool(r) == tool), None)
        args = self.clean_args(tool, self.role_args(role)) if role else {}
        array = next((k for k, p in props.items() if isinstance(p, dict) and p.get("type") == "array"), None)
        single = next((k for k, p in props.items() if isinstance(p, dict) and p.get("type") == "string" and "enum" not in p and k not in args), None)
        if array:
            args[array] = rids[: max(1, self.room_for_compare())]
        elif single:
            args[single] = rids[0]
        else:
            return []
        args, problems = schema_problems(args, schema)
        if problems:
            return []
        observations = self.world([{"name": tool, "args": args}])
        if not observations:
            return []
        self.absorb(tool, args, observations[0])
        return [r for r in rids if r in self.full and tool in ((self.seen.get(r) or {}).get("by") or [])]

    def find_attr_tool(self) -> None:
        """The attributes tool, found by opening one listing with each listing tool not yet tried (a store may show the
        text on one tool and the attribute table on another)."""

        if self.attr_tool or not self.seen:
            return
        tried = {t for row in self.seen.values() for t in (row.get("by") or [])}
        samples = [rid for rid in self.full if rid not in self.gone] + [rid for rid, row in self.seen.items() if rid not in self.full and row.get("row") and rid not in self.gone]
        if not samples:
            return
        for tool in self.listing_tools():      # (the compare tool first: one call covers several)
            if tool in tried or self.attr_tool or self.steps_left() <= 16 or self.finished:
                continue
            self.open_with(tool, samples[: self.room_for_compare()] if self.takes_array(tool) else samples[:1])

    def listing_tools(self) -> list[str]:
        """Tools that may show one listing: the compare and open roles first, then any other tool outside the roles
        whose parameters are a single string (an id) — a store may give the attribute table a tool of its own that the
        brief named for no role."""

        taken = {self.role_tool(r) for r in ("search", "ask", "cart", "cart_view", "buy", "refuse")} - {None}
        out = [t for t in (self.role_tool("compare"), self.role_tool("open")) if t]
        for t in self.tools:
            name = t["function"]["name"]
            props = (t["function"].get("parameters") or {}).get("properties") or {}
            strings = [k for k, p in props.items() if isinstance(p, dict) and p.get("type") == "string" and "enum" not in p]
            if name not in taken and name not in out and len(props) <= 2 and len(strings) == 1:
                out.append(name)
        return out

    def takes_array(self, tool: str) -> bool:
        props = (self.schemas.get(tool) or {}).get("properties") or {}
        return any(isinstance(p, dict) and p.get("type") == "array" for p in props.values())

    def widen(self) -> None:
        """Before the plan: the attribute pages the menu is built from. The attributes tool is found first; then the
        listings without a table are opened with it (those the conductor already looked at first, then two unread rows
        per item). Where no tool shows a table: one compare call across the items, or up to five single pages."""

        if self.widened:
            return
        self.widened = True
        self.find_attr_tool()
        if self.attr_tool:
            queue = [rid for rid in self.full if not self.attributes_of(self.full[rid]) and rid not in self.gone]
            per_item: dict[int, int] = {}
            for rid, row in self.seen.items():
                if rid not in self.full and rid not in self.gone and row.get("row"):
                    k = self.item_of(rid)
                    if per_item.get(k, 0) < 2:
                        per_item[k] = per_item.get(k, 0) + 1
                        queue.append(rid)
            opened = 0
            while queue and opened < 6 and self.steps_left() > 16 and not self.finished:
                batch = queue[: max(1, self.room_for_compare())] if self.takes_array(self.attr_tool) else queue[:1]
                queue = queue[len(batch):]
                opened += len(self.open_with(self.attr_tool, batch)) or 1
            return
        unread: dict[int, list[str]] = {}
        for rid, row in self.seen.items():
            if rid not in self.full and rid not in self.gone and row.get("row"):
                unread.setdefault(self.item_of(rid), []).append(rid)
        if not unread:
            return
        compare = self.role_tool("compare")
        if compare and self.steps_left() > 12:
            schema = self.schemas.get(compare) or {}
            param = next((k for k, p in (schema.get("properties") or {}).items() if isinstance(p, dict) and p.get("type") == "array"), None)
            if param:
                room = self.room_for_compare()
                picks: list[str] = []
                while len(picks) < room and any(unread.values()):
                    for k in list(unread):
                        if unread[k] and len(picks) < room:
                            picks.append(unread[k].pop(0))
                args = self.clean_args(compare, self.role_args("compare"))
                args[param] = picks
                observations = self.world([{"name": compare, "args": args}])
                if observations:
                    self.absorb(compare, args, observations[0])
                    if any(self.facts(self.full.get(r) or {}) for r in picks):
                        return
        opener = self.role_tool("open")
        if not opener:
            return
        param = next((k for k, p in ((self.schemas.get(opener) or {}).get("properties") or {}).items() if isinstance(p, dict) and p.get("type") == "string"), None)
        if not param:
            return
        opened = 0
        for k in list(unread):
            for rid in unread[k][:2]:
                if opened >= 5 or self.steps_left() <= 22 or self.finished:
                    return
                args = self.clean_args(opener, self.role_args("open"))
                args[param] = rid
                observations = self.world([{"name": opener, "args": args}])
                if observations:
                    self.absorb(opener, args, observations[0])
                opened += 1

    def question_text(self, item: int, label: str, samples: list[str] | None = None, labels: list[str] | None = None) -> str:
        name = self.items[item - 1]["name"] if 1 <= item <= len(self.items) else "the product"
        if labels and len(labels) > 1:
            return f"For the {name}, what {', '.join(labels[:-1])} or {labels[-1]} would you like?"
        text = f"What {label} would you like for the {name}?"
        if samples:
            text = text[:-1] + ", for example " + " or ".join(samples[:2]) + "?"
        return text

    @staticmethod
    def first_property(label: str) -> str:
        """A planned label cut to its first property: at ' and ', ' or ', ',', ';', '(' and to five words."""

        cut = re.split(r"\s+(?:and|or)\s+|[,;(/]", str(label or ""), maxsplit=1)[0].strip().strip(".:?")
        return " ".join(cut.split()[:5]).lower()

    def send(self, text: str, floor: int = 6) -> str | None:
        """One message to the shopper, within the allowance and keeping `floor` steps; the reply's text, None when not sent."""

        spec = self.roles.get("ask")
        if not spec or self.finished or self.steps_left() <= floor:
            return None
        if self.msg_limit is not None and self.messages_sent >= self.msg_limit:
            return None
        props = (self.schemas.get(spec["tool"]) or {}).get("properties") or {}
        param = next((k for k, p in props.items() if isinstance(p, dict) and p.get("type") == "string" and "enum" not in p), None)
        if not param:
            return None
        args, problems = schema_problems({**(spec.get("args") or {}), param: text}, self.schemas[spec["tool"]])
        if problems:
            return None
        observations = self.world([{"name": spec["tool"], "args": args}])
        if not observations:
            return None
        self.messages_sent += 1
        reply = self.reply_text(observations[0])
        self.asked.append((text, reply))
        self.done_log.append(f"message sent at step {self.step}: {text[:160]} -> {reply[:200]}")
        return reply

    @staticmethod
    def reply_text(observation: Any) -> str:
        """The shopper's words in a message result: its string values, longest first."""

        texts: list[tuple[str, str]] = []
        def walk(node: Any, key: str) -> None:
            if isinstance(node, str):
                texts.append((key, node))
            elif isinstance(node, dict):
                for k, v in node.items():
                    walk(v, str(k))
            elif isinstance(node, list):
                for v in node:
                    walk(v, key)
        walk(observation, "")
        if len(texts) == 1:
            return texts[0][1][:600]
        spoken = [t for _, t in texts if " " in t.strip()]
        return " ".join(sorted(spoken or [t for _, t in texts], key=len, reverse=True)[:3])[:600]

    def beacon(self) -> None:
        """Once, when the interview has not sent a question by the first cart or ending call (or by step 6): one call
        of the tool the episode opened with, its words carrying the code's state, so the network's record of the
        episode shows why the shopper was not asked (the code's own trace is not kept there)."""

        if self.beacon_done or self.finished or not self.records or self.steps_left() <= 4:
            return
        self.beacon_done = True
        tool = self.role_tool("search") or self.records[0][0]
        first_args = next((a for n, a, _ in self.records if n == tool), {})
        props = (self.schemas.get(tool) or {}).get("properties") or {}
        param = next((k for k, p in props.items() if isinstance(p, dict) and p.get("type") == "string" and "enum" not in p), None)
        if not param:
            return
        name = self.items[0]["name"] if self.items else "item"
        state = (f"{name} a{int(bool(self.roles.get('ask')))} r{len(self.roles)} q{self.questions_left()} m{self.msg_limit} "
                 f"p{sum(int(p.get('messages') or 0) for p in self.process)} s{self.steps_left()} c{int(self.cost * 1000)} n{self.calls} "
                 f"b{int(self.broke)} i{int(self.asked_once)} k{len(self.full)}")
        args, problems = schema_problems({k: v for k, v in first_args.items() if k != param}, self.schemas[tool])
        args[param] = state
        self.trace.append({"beacon": state})
        self.world([{"name": tool, "args": args}])

    def hold_interview(self) -> str:
        """The one interview: the menu, the plan, the first questions back to back, one reading of the replies, then the
        last message as a repair or the next property. Returns the summary for the conductor."""

        self.asked_once = True
        self.widen()
        free = self.questions_left()
        count = free + 1
        menu = self.key_menu()
        menu_text = "\n".join(f"item {k} ({self.items[k - 1]['name']}): " + "; ".join(f"{key} e.g. {', '.join(samples[:3])}" for key, _, samples in keys) for k, keys in menu.items()) or "(no attributes read yet)"
        items_text = "\n".join(f"item {k}: {it['name']} | settled: " + ("; ".join(f"{p}: {c}" for p, c in {**it['musts'], **it['told']}.items()) or "nothing") + (" | the shopper asked to be asked about this first" if self.ask_first else "")
                               for k, it in enumerate(self.items, 1))
        plan = self.read(PLAN_SYSTEM, f"Request:\n{self.request}\n\nProducts:\n{items_text}\n\nAttribute menu:\n{menu_text}\n\nQuestions wanted: {count}", PLAN_FORM, 700, HEAVY, think=False)
        planned: list[tuple[int, str]] = []
        for q in plan.get("questions") or []:
            if not isinstance(q, dict):
                continue
            label = self.first_property(q.get("property"))
            item = q.get("item") if isinstance(q.get("item"), int) and 1 <= q.get("item") <= len(self.items) else 1
            if label and (item, label) not in planned and not any(_fold(p) == label for p in list(self.items[item - 1]["musts"]) + list(self.items[item - 1]["told"])):
                planned.append((item, label))
        if not planned:
            for k, keys in menu.items():
                for key, _, _ in keys[:3]:
                    planned.append((k, _fold(key.replace("_", " "))))
        self.trace.append({"interview": {"count": count, "menu": {k: [key for key, _, _ in v] for k, v in menu.items()}, "planned": planned}})
        # (one question per message, its reply read before the next is chosen: every question carries one property, so
        #  a reply that did not follow it settles nothing and the next message goes to the next property — of another
        #  product first once this one has yielded a wish; no question is repeated. The menu's remaining keys follow the
        #  plan, so the messages never run out of properties)
        for k, keys in menu.items():
            for key, _, _ in keys:
                if (k, _fold(key.replace("_", " "))) not in planned:
                    planned.append((k, _fold(key.replace("_", " "))))
        sent: list[tuple[int, int, str, str]] = []     # (question number, item, label, reply)
        heard_items: set[int] = set()
        asked_together: dict[int, list[str]] = {}       # per item, the labels already named in a bundled message
        def settled(item: int, label: str) -> bool:
            return any(_fold(p) == label for p in list(self.items[item - 1]["told"]) + list(self.items[item - 1]["musts"]))
        while planned and self.questions_left() > 0:
            item, label = next(((i, l) for i, l in planned if i not in heard_items), planned[0])
            if settled(item, label):
                planned.remove((item, label))
                continue
            # (while the shopper answers them, a message names the product's open properties together, the ones not yet
            #  named first; a reply that calls the message crowded ends that, and the rest go one per message)
            labels = [label]
            if self.bundle_ok:
                open_labels = [l for i, l in planned if i == item and not settled(i, l)]
                named = asked_together.get(item) or []
                labels = ([l for l in open_labels if l not in named] + [l for l in open_labels if l in named])[:8]
            text = self.question_text(item, label, labels=labels)
            reply = self.send(text)
            if reply is None:
                break
            sent.append((len(sent) + 1, item, label, reply))
            before = dict(self.items[item - 1]["told"])
            heard = self.hear([sent[-1]], HEAVY, questions={len(sent): text} if len(labels) > 1 else None)
            if len(labels) > 1 and len(sent) in (heard.get("crowded") or []):
                self.bundle_ok = False
                continue
            if len(labels) > 1:
                asked_together.setdefault(item, []).extend(l for l in labels if l not in asked_together.get(item, []))
                for l in labels:        # (a bundle with no wish in the reply is kept for a repeat, behind the unasked ones)
                    if (item, l) in planned:
                        planned.remove((item, l)); planned.append((item, l))
            else:
                planned.remove((item, label))
            if self.items[item - 1]["told"] != before or any(l["item"] == item for l in self.lookups):
                heard_items.add(item)
        if not sent:
            self.beacon()
        lines = [f"[shopper asked] Q{n} ({self.items[item - 1]['name']}, {label}): {reply[:160]}" for n, item, label, reply in sent]
        told = "; ".join(f"item {k} {p}: {v}" for k, it in enumerate(self.items, 1) for p, v in it["told"].items())
        if told:
            lines.append("[conditions now in force from the shopper] " + told)
        if self.lookups:
            lines.append("[pointers] " + " | ".join(f"item {l['item']} {l['property']}: {l['where']}" for l in self.lookups if not l.get("value")))
        searched = self.heard_searches()
        if searched:
            lines.append(searched)
        self.heard_text = "\n".join(lines)
        return self.heard_text

    def hear(self, sent: list[tuple[int, int, str, str]], models: list[str], questions: dict[int, str] | None = None) -> dict[str, Any]:
        """One reading of the replies; what they settle goes onto the sheet. `questions` gives the text of a message
        that was not a label question (a process step's message)."""

        text = "\n".join(f"Q{n} (item {item}: {self.items[item - 1]['name']}" + (f"; property asked: {label}" if not (questions or {}).get(n) else "") + f"): {(questions or {}).get(n) or self.question_text(item, label)}\nA{n}: {reply}"
                         for n, item, label, reply in sent)
        heard = self.read(HEAR_SYSTEM, f"Request:\n{self.request}\n\nQuestions and replies:\n{text}", HEAR_FORM, 800, models, think=False)
        changed = False
        for need in heard.get("needs") or []:
            if isinstance(need, dict) and need.get("property") and need.get("value"):
                item = need.get("item") if isinstance(need.get("item"), int) and 1 <= need.get("item") <= len(self.items) else 1
                prop, value = str(need["property"]).strip()[:60], str(need["value"]).strip()[:120]
                if not re.search(r"[a-z0-9]", value, re.I) or re.fullmatch(r"[<(\[]?\s*(?:unknown|none|n/?a|null|unspecified)\s*[>)\]]?", value, re.I):
                    continue        # (a placeholder is not a value)
                if self.items[item - 1]["told"].get(prop) != value:
                    self.items[item - 1]["told"][prop] = value
                    changed = True
        for pointer in heard.get("pointers") or []:
            if isinstance(pointer, dict) and pointer.get("property") and pointer.get("where"):
                item = pointer.get("item") if isinstance(pointer.get("item"), int) and 1 <= pointer.get("item") <= len(self.items) else 1
                prop = str(pointer["property"]).strip()[:60]
                if any(l["item"] == item and _fold(l["property"]) == _fold(prop) for l in self.lookups):
                    continue
                self.lookups.append({"property": prop, "item": item, "where": str(pointer["where"]).strip()[:300], "value": None})
                changed = True
        for prop in heard.get("dropped") or []:
            for it in self.items:
                for p in [p for p in it["musts"] if _fold(p) == _fold(prop)]:
                    it["musts"].pop(p); changed = True
                for p in [p for p in it["told"] if _fold(p) == _fold(prop)]:
                    it["told"].pop(p); changed = True
        for prop in heard.get("freed") or []:
            if isinstance(prop, str) and prop.strip():
                for it in self.items:
                    if _fold(prop) not in [_fold(f) for f in it["free"]]:
                        it["free"].append(prop.strip()[:60])
        budget = heard.get("budget")
        if isinstance(budget, (int, float)) and not isinstance(budget, bool) and budget > 0 and budget != self.budget:
            self.budget = float(budget)
            changed = True
        if changed:
            self.version += 1
        return heard

    def absorb(self, name: str, args: dict[str, Any], observation: Any) -> None:
        """What a store result teaches the record book: rows, opened records (an opening call's argument may also
        correct the id key; a search's words never do)."""

        self.learn_rows(observation)
        if self.is_role("open", name, args) or self.is_role("compare", name, args) or self.is_role("cart", name, args):
            self.fix_id_key(args)
        self.learn_pages(args, observation, name)

    # -- prices, records and cart calls ---------------------------------------------------------
    def price_of(self, rid: str) -> float | None:
        row = self.seen.get(rid) or {}
        if isinstance(row.get("price"), float):
            return row["price"]
        record = self.full.get(rid) or {}
        value = record.get(self.price_key) if self.price_key else None
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        return None

    def buyable(self, rid: str) -> bool:
        return rid not in self.gone and rid not in self.refused and (self.seen.get(rid) or {}).get("stock") is not False

    def record_text(self, rid: str, cap: int = 1500) -> str:
        record = self.full.get(rid)
        if record is None:
            row = (self.seen.get(rid) or {}).get("row")
            return "ROW " + _short(row, cap) if row else rid
        return ("SOLD OUT " if rid in self.gone else "") + _short(record, cap)

    def cart_state(self) -> str:
        return json.dumps(self.cart, sort_keys=True)

    def cart_calls(self, picks: dict[str, int]) -> list[dict[str, Any]]:
        """The cart tool's calls that turn the cart book into picks: removals of lines not wanted, adds of lines missing."""

        tool = self.role_tool("cart")
        if not tool:
            return []
        schema = self.schemas.get(tool) or {}
        props = schema.get("properties") or {}
        fixed = self.role_args("cart")
        array = next((k for k, p in props.items() if isinstance(p, dict) and p.get("type") == "array"), None)
        if array:
            inner = ((props[array].get("items") or {}).get("properties") or {}) if isinstance(props[array].get("items"), dict) else {}
            id_name = next((k for k, p in inner.items() if isinstance(p, dict) and p.get("type") == "string"), "id")
            n_name = next((k for k, p in inner.items() if isinstance(p, dict) and p.get("type") == "integer"), "quantity")
            return [{"name": tool, "args": {**fixed, array: [{id_name: rid, n_name: n} for rid, n in picks.items()]}}]
        id_param = next((k for k, p in props.items() if isinstance(p, dict) and p.get("type") == "string" and "enum" not in p), None)
        if not id_param:
            return []
        count = next((k for k, p in props.items() if isinstance(p, dict) and p.get("type") == "integer"), None)
        mode = next((k for k, p in props.items() if isinstance(p, dict) and "enum" in p and len(p["enum"]) >= 2 and k not in fixed), None)
        calls: list[dict[str, Any]] = []
        for rid in list(self.cart):
            if rid not in picks:
                args = {**fixed, id_param: rid}
                if mode:
                    args[mode] = props[mode]["enum"][1]
                elif count and props[count].get("minimum") == 0:
                    args[count] = 0
                else:
                    continue
                calls.append({"name": tool, "args": args})
        for rid, n in picks.items():
            if self.cart.get(rid) == n:
                continue
            args = {**fixed, id_param: rid}
            if mode:
                args[mode] = props[mode]["enum"][0]
            if count:
                sets = props[count].get("minimum") == 0 and not mode
                if not sets and self.cart.get(rid, 0) > n and mode:
                    # (a line holding more than wanted: the line goes out and comes back with the wanted count)
                    calls.append({"name": tool, "args": {**fixed, id_param: rid, mode: props[mode]["enum"][1]}})
                    args[count] = n
                else:
                    args[count] = n if sets or rid not in self.cart else max(1, n - self.cart.get(rid, 0))
            calls.append({"name": tool, "args": args})
        return calls

    def reread_lines(self, ids: list[str], keep: int) -> list[str]:
        """Lines read again after a refusal (a sold-out or changed line shows there), the fresh read merged over what
        the other tools gave of the listing."""

        kept = {r: self.full.pop(r, None) for r in ids if r in self.full}
        opened = self.open_ids(ids, keep=keep)
        for r, old in kept.items():
            if isinstance(old, dict):
                self.full[r] = {**old, **(self.full.get(r) or {})}
        return opened

    def open_ids(self, ids: list[str], keep: int) -> list[str]:
        """Listings opened by code, through the compare tool when it exists else one by one, keeping `keep` steps."""

        opened: list[str] = []
        ids = [r for r in ids if r in self.seen and r not in self.full and r not in self.gone]
        compare = self.role_tool("compare")
        if compare and ids:
            param = self.compare_param()
            if param:
                room = self.room_for_compare()
                while ids and self.steps_left() > keep and not self.finished:
                    batch, ids = ids[:room], ids[room:]
                    args = {**self.role_args("compare"), param: batch}
                    observations = self.world([{"name": compare, "args": args}])
                    if not observations:
                        break
                    self.absorb(compare, args, observations[0])
                    opened += [r for r in batch if r in self.full]
                return opened
        opener = self.role_tool("open")
        param = next((k for k, p in ((self.schemas.get(opener) or {}).get("properties") or {}).items() if isinstance(p, dict) and p.get("type") == "string"), None) if opener else None
        if not param:
            return opened
        for rid in ids:
            if self.steps_left() <= keep or self.finished:
                break
            args = {**self.role_args("open"), param: rid}
            observations = self.world([{"name": opener, "args": args}])
            if not observations:
                break
            self.absorb(opener, args, observations[0])
            if rid in self.full:
                opened.append(rid)
        return opened

    # -- the process gate ---------------------------------------------------------------------
    def process_gate(self, stage: str) -> tuple[bool, str]:
        """The request's own steps due at this stage, proven from the record or done by one call each; at most three
        rounds. Returns (go, text): go False holds the ending with the text."""

        # (a step due at the end is due at the order too: the order ends the episode)
        due = [p for p in self.process if p.get("when") == stage or (stage == "order" and p.get("when") == "end")]
        mark = (stage, self.cart_state(), self.version)
        if not due or mark in self.gate_ok or not self.helpers_may_run():
            return True, ""
        ending = {self.role_tool("buy"), self.role_tool("refuse")} - {None}
        tools = "\n".join(f"- {t['function']['name']}: parameters {json.dumps(t['function'].get('parameters') or {}, ensure_ascii=False)[:400]}" for t in self.tools if t["function"]["name"] not in ending)
        nudge = ""
        for round_no in range(3):
            record = "\n".join(f"{i + 1}. {line}" for i, line in enumerate(self.done_log)) or "(nothing sent or done yet)"
            steps = "\n".join(f"step {i + 1} ({p['when']}, {p['messages']} message(s)): {p['text']}" for i, p in enumerate(due))
            user = (f"Request:\n{self.request}\n\n{self.sheet()}\n\nSteps due now:\n{steps}\n\nRecord:\n{record}\n\nStore tools:\n{tools}"
                    + (f"\n\nNote: {nudge}" if nudge else ""))
            verdict = self.read(GATE_SYSTEM, user, GATE_FORM, 2000, HEAVY if nudge else LIGHT)
            if not verdict:
                if self.hold("gate_unread") <= 1:
                    return False, ("held: the request's own steps could not be verified by the code (" + "; ".join(p["text"][:80] for p in due)
                                   + "). Do what they ask yourself, then finish again.")
                self.gate_ok.add(mark)
                return True, ""
            proven: set[int] = set()
            for d in verdict.get("done") or []:
                if not isinstance(d, dict) or not isinstance(d.get("step"), int) or not 1 <= d["step"] <= len(due):
                    continue
                lines = [n for n in d.get("lines") or [] if isinstance(n, int) and 1 <= n <= len(self.done_log)]
                step = due[d["step"] - 1]
                if not lines:
                    continue
                if stage == "order" and step.get("messages"):
                    # (a message at the order counts only when sent after the cart last changed)
                    stamps = [int(m.group(1)) for n in lines for m in [re.search(r"at step (\d+)", self.done_log[n - 1])] if m]
                    if not stamps or max(stamps) < self.cart_at:
                        continue
                proven.add(d["step"] - 1)
                step["done"] = lines
            call = verdict.get("call") if isinstance(verdict.get("call"), dict) else None
            if len(proven) == len(due) and not call:
                self.gate_ok.add(mark)
                return True, ""
            if not call:
                if nudge:
                    self.gate_ok.add(mark)       # (the heavy reader found nothing more to do: the steps stand as proven)
                    return True, ""
                nudge = "a step claimed done has no record line after the cart last changed, or no proof at all: give the one call that does it, or the proof."
                continue
            name, args = call.get("tool"), call.get("args") if isinstance(call.get("args"), dict) else {}
            if name not in self.schemas or name in ending or self.steps_left() <= 1:
                nudge = "that call is not one of the store's tools usable here; give another, or null."
                continue
            if self.is_role("ask", name, args):
                text = next((v for v in args.values() if isinstance(v, str) and len(v) > 3), "")
                reply = self.send(text, floor=1)
                if reply is None:
                    return True, ""
                self.gate_msgs += 1
                nudge = ""
                heard = self.hear([(1, 1, "", reply)], LIGHT, {1: text}) if self.helpers_may_run() else {}
                if heard.get("needs") or heard.get("pointers") or heard.get("dropped") or heard.get("budget"):
                    return False, f"held: the shopper's reply changes the task ({reply[:200]}). Fix the cart for it, then finish again."
                continue
            args, problems = schema_problems(args, self.schemas[name])
            if problems:
                nudge = "the call's arguments break the tool's schema: " + "; ".join(problems[:3])
                continue
            observations = self.world([{"name": name, "args": args}])
            if not observations:
                return True, ""
            self.done_log.append(f"call made at step {self.step}: {name} {json.dumps(args, ensure_ascii=False)[:160]} -> {_short(observations[0], 200)}")
            self.after(name, args, observations[0])
            nudge = ""
        self.gate_ok.add(mark)
        return True, ""

    # -- the cart check and the audit ----------------------------------------------------------
    def verify_cart(self, picks: dict[str, int] | None = None, forced: bool = False) -> dict[str, Any]:
        """The heavy reader's verdict on the cart (or on planned picks): cached per cart and version."""

        cart = picks if picks is not None else self.cart
        key = (json.dumps(cart, sort_keys=True), self.version)
        if key in self.checks:
            return self.checks[key]
        if not (self.helpers_may_run() or (forced and self.cost < _SPEND_HARD and not self.broke)):
            return {"ok": True, "problems": [], "better": [], "skipped": True}
        lines = "\n".join(f"{rid} x{n}" + (f" at {self.price_of(rid):g}" if self.price_of(rid) is not None else "") + ": " + self.record_text(rid, 1100) for rid, n in cart.items())
        others = ""
        if self.pick:
            rest = [r for r in self.seen if r not in cart and self.buyable(r)]
            rest.sort(key=lambda r: (r not in self.full, self.price_of(r) if self.price_of(r) is not None else 1e12))
            others = "\n\nOther listings seen (opened first, cheapest first):\n" + "\n".join(f"{r}" + (f" at {self.price_of(r):g}" if self.price_of(r) is not None else "") + ": " + self.record_text(r, 450) for r in rest[:8])
        verdict = self.read(CHECK_SYSTEM, f"Request:\n{self.request}\n\n{self.sheet()}\n\nCart:\n{lines or '(empty)'}{others}", CHECK_FORM, 900, HEAVY, think=False)
        if not verdict:
            # (no reading came back: not cached, so the next attempt asks again; the caller holds once on it)
            return {"ok": False, "problems": [], "better": [], "unread": True}
        verdict = self.check_verdict(cart, verdict)
        self.checks[key] = verdict
        self.trace.append({"cart_check": {"cart": cart, "verdict": verdict}})
        return verdict

    def check_verdict(self, cart: dict[str, int], read: dict[str, Any]) -> dict[str, Any]:
        """The verdict from the reader's structured reading: a line that is not a wanted kind of product, a condition
        some record states a different value for, a line shown sold out. The arithmetic is the code's own: each
        product's quantity in the cart (the reader's product number for a line, else the category match) and the total
        against the allowance. Under a pick rule a better listing counts only when it is shown, buyable, not in the
        cart and, under cheapest, priced below the line it would replace."""

        problems: list[str] = []
        lines = {l["id"]: l for l in (read.get("lines") or []) if isinstance(l, dict) and isinstance(l.get("id"), str)}
        owner: dict[str, int] = {}
        for rid, n in cart.items():
            line = lines.get(rid) or {}
            product = line.get("product")
            owner[rid] = product if isinstance(product, int) and 1 <= product <= len(self.items) else self.item_of(rid)
            if line.get("wanted_kind") is False:
                problems.append(f"{rid}: not the kind of product wanted")
            for c in (line.get("conditions") or []):
                if isinstance(c, dict) and c.get("verdict") == "other" and isinstance(c.get("property"), str) and c.get("stated"):
                    problems.append(f"{rid}: {c['property'][:40]} is {str(c['stated'])[:80]}")
                    if len(problems) >= 6:
                        break
            if line.get("sold_out_shown") is True:
                problems.append(f"{rid}: shown sold out")
        for k, item in enumerate(self.items, 1):
            have = sum(n for rid, n in cart.items() if owner.get(rid) == k)
            want = item.get("quantity") or 1
            if have < want:
                problems.append(f"item {k} ({item.get('name')}): {have} of {want} in the cart")
            elif have > want and sum(1 for rid in cart if owner.get(rid) == k) > 1:
                problems.append(f"item {k} ({item.get('name')}): {have} of {want} in the cart on several lines; keep one listing of it")
        if self.budget is not None and cart and all(self.price_of(r) is not None for r in cart):
            total = sum(self.price_of(r) * n for r, n in cart.items())
            if total > self.budget + 1e-9:
                problems.append(f"the total {total:g} exceeds the allowed {self.budget:g}")
        better: list[str] = []
        failing = {owner[rid] for rid in cart if any(p.startswith(rid + ":") for p in problems)}
        if self.pick or failing:
            for x in read.get("better") or []:
                if not (isinstance(x, str) and x in self.seen and x not in cart and self.buyable(x)):
                    continue
                if self.pick == "cheapest" and self.item_of(x) not in failing:
                    # (the line it would replace fits: only a cheaper listing beats it)
                    mine = [self.price_of(r) for r in cart if owner.get(r) == self.item_of(x) and self.price_of(r) is not None]
                    if self.price_of(x) is None or not mine or self.price_of(x) >= min(mine):
                        continue
                if self.item_of(x) not in failing and not self.pick:
                    continue
                better.append(x)
            if better and not failing:
                problems.append("under the pick rule these shown listings would do better: " + ", ".join(better[:5]))
        return {"ok": not problems, "problems": problems[:6], "better": better[:5]}

    def audit(self) -> dict[str, Any]:
        """The heavy reader's audit before a decline: fits per item among opened listings, rows worth opening."""

        opened = sorted(self.full, key=lambda r: -(self.seen.get(r) or {}).get("opened", 0))
        rows = [r for r in self.seen if r not in self.full and self.buyable(r)][:30]
        text = "\n".join(f"OPENED {r}" + (f" at {self.price_of(r):g}" if self.price_of(r) is not None else "") + ": " + self.record_text(r, 600) for r in opened[:16])
        text += "\n" + "\n".join(f"ROW {r}: " + _short((self.seen[r] or {}).get("row"), 220) for r in rows[:16])
        verdict = self.read(AUDIT_SYSTEM, f"Request:\n{self.request}\n\n{self.sheet()}\n\nRecords:\n{text}", AUDIT_FORM, 800, HEAVY, think=False)
        self.audits += 1
        fits: dict[int, list[str]] = {}
        for f in verdict.get("fits") or []:
            if isinstance(f, dict) and isinstance(f.get("item"), int):
                fits[f["item"]] = [x for x in f.get("ids") or [] if isinstance(x, str) and x in self.full and self.buyable(x)]
        to_open = [x for x in verdict.get("open") or [] if isinstance(x, str) and x in self.seen and x not in self.full and self.buyable(x)][:6]
        self.trace.append({"audit": {"fits": fits, "open": to_open}})
        return {"fits": fits, "open": to_open}

    def complete_cart(self) -> str | None:
        """The ending done by code when a confirmed lead exists: the lead carted for its item, the planned cart checked
        once more, the gate, the checkout; one retry after a refusal. Returns the text for the conductor."""

        lead = self.confirmed
        if not lead or not self.role_tool("buy") or not self.role_tool("cart"):
            return None
        item = self.item_of(lead)
        picks = {rid: n for rid, n in self.cart.items() if self.item_of(rid) != item}
        picks[lead] = self.items[item - 1]["quantity"] if 1 <= item <= len(self.items) else 1
        verdict = self.verify_cart(picks, forced=True)
        if not verdict.get("ok"):
            return None
        tried = False
        for attempt in range(2):
            if self.finished or self.steps_left() < len(self.cart_calls(picks)) + 1:
                break
            for call in self.cart_calls(picks):
                observations = self.world([call])
                if not observations:
                    return None
                self.after(call["name"], call["args"], observations[0])
            if self.cart != picks:
                self.cart = dict(picks)
                self.cart_at = self.step
            go, text = self.process_gate("order")
            if not go:
                return text
            if self.steps_left() < 1:
                break
            observations = self.world([{"name": self.role_tool("buy"), "args": self.ending_args("buy")}])
            self.trace.append({"complete_cart": {"attempt": attempt, "picks": picks, "result": _short(observations[0] if observations else None, 300)}})
            tried = True
            if self.finished:
                return "Not run: the code carted the fitting listing and placed the order instead."
            if not observations:
                return None
            self.after(self.role_tool("buy"), self.ending_args("buy"), observations[0])
            if any(not self.buyable(r) for r in picks) or not self.verify_cart(picks, forced=True).get("ok"):
                return "the code tried to order the fitting listing and the store refused it; read the refusal above and fix the cart."
            self.reread_lines(list(picks), keep=1)
            self.checks.clear()
        if not tried:
            return None
        return "the code tried to order the fitting listing and the store refused it; read the refusal above and fix the cart."

    def quiet_text(self, args: dict[str, Any]) -> dict[str, Any]:
        """A decline's free text replaced by one fixed sentence unless the request explicitly asks for an explanation."""

        tool = self.role_tool("refuse")
        props = (self.schemas.get(tool) or {}).get("properties") or {}
        for k, p in props.items():
            if isinstance(p, dict) and p.get("type") == "string" and "enum" not in p and k not in self.role_args("refuse"):
                text = str(args.get(k) or "")
                if self.quiet_decline is not False or self.ids_in({"t": text}) or re.search(r"\b(p|r|l|sku)-[0-9a-f]{6,}\b", text, re.I):
                    args[k] = "Nothing in the store meets every condition, so I am not ordering."
        return args

    # -- evidence, notes, pointers, paging ------------------------------------------------------
    def evidence(self, ids: list[str]) -> str:
        """After an open: per listing, the attribute keys named for a condition's property, with their values; a key
        named for another property decides nothing, and that is said."""

        lines = ["[evidence] (a key named for the property decides; a key for another property decides nothing)"]
        for rid in ids[:6]:
            facts = self.facts(self.full.get(rid) or {})
            item = self.items[self.item_of(rid) - 1] if self.items else {"musts": {}, "told": {}}
            props = list(item.get("musts") or {}) + list(item.get("told") or {})
            parts = []
            for prop in props:
                words = {self.stem(w) for w in re.findall(r"[a-z0-9]+", _fold(prop)) if len(w) > 2}
                hits = [f"{k}={str(v)[:60]}" for k, v in facts.items() if words and words <= {self.stem(w) for w in re.findall(r"[a-z0-9]+", _fold(k.replace('_', ' ')))}]
                parts.append(f"{prop}: " + ("; ".join(hits[:3]) if hits else "no key named for it"))
            price = self.price_of(rid)
            lines.append(f"{rid}" + (f" at {price:g}" if price is not None else "") + (" SOLD OUT" if not self.buyable(rid) else "") + " | " + (" | ".join(parts) or "no condition to check"))
        return "\n".join(lines)

    def loose_lines(self, observation: Any) -> list[str]:
        """Sentences inside a result that are not listing data: strings of five words or more (six inside nested
        records), outside the rows learned."""

        out: list[str] = []
        def is_record(node: dict[str, Any]) -> bool:
            if self.id_key and isinstance(node.get(self.id_key), str):
                return True
            return any(isinstance(v, str) and v in self.seen for v in node.values())
        def walk(node: Any, inside: bool) -> None:
            if isinstance(node, dict):
                record = inside or is_record(node)
                for k, v in node.items():
                    if isinstance(v, str):
                        # (inside a record, a string under a key the rows or the attribute blocks use is listing data;
                        #  a string under another key, such as a shopper's note attached to the result, is a line)
                        if (not record or str(k) not in self.record_keys) and len(v.split()) >= 5 and v not in out:
                            out.append(v[:400])
                    else:
                        walk(v, record)
            elif isinstance(node, list):
                for v in node:
                    if isinstance(v, str):
                        if not inside and len(v.split()) >= 5 and v not in out:
                            out.append(v[:400])
                    else:
                        walk(v, inside)
        walk(observation, False)
        return out[:2]

    def take_notes(self, name: str, args: dict[str, Any], observation: Any) -> str:
        """Loose lines in a result read once each; a change goes onto the sheet and is echoed to the conductor."""

        lines = [l for l in self.loose_lines(observation) if l not in self.seen_notes or (self.sold_note and _fold(l) == _fold(self.sold_note))]
        if not lines or self.note_reads >= 8:
            return ""
        echoes: list[str] = []
        for line in lines:
            if self.sold_note and _fold(line) == _fold(self.sold_note) and line in self.seen_notes:
                self.mark_sold(args)
                echoes.append("[note read] a line is sold out again: " + line[:120])
                continue
            self.seen_notes.add(line)
            if not self.helpers_may_run():
                echoes.append("[note] " + line[:200])
                continue
            self.note_reads += 1
            note = self.read(NOTE_SYSTEM, f"Request:\n{self.request}\n\n{self.sheet()}\n\nThe call: {name} {json.dumps(args, ensure_ascii=False)[:200]}\n\nThe line:\n{line}", NOTE_FORM, 800, LIGHT)
            if not note:
                echoes.append("[note] " + line[:200])
                continue
            changed = False
            budget = note.get("budget")
            if isinstance(budget, (int, float)) and not isinstance(budget, bool) and budget > 0 and budget != self.budget:
                self.budget = float(budget); changed = True
            for add in note.get("added") or []:
                if isinstance(add, dict) and add.get("property") and add.get("value"):
                    item = add.get("item") if isinstance(add.get("item"), int) and 1 <= add.get("item") <= len(self.items) else 1
                    prop, value = str(add["property"]).strip()[:60], str(add["value"]).strip()[:120]
                    if self.items[item - 1]["told"].get(prop) != value:
                        self.items[item - 1]["told"][prop] = value; changed = True
            for prop in note.get("dropped") or []:
                for it in self.items:
                    for p in [p for p in it["musts"] if _fold(p) == _fold(prop)]:
                        it["musts"].pop(p); changed = True
                    for p in [p for p in it["told"] if _fold(p) == _fold(prop)]:
                        it["told"].pop(p); changed = True
            gone = [g for g in note.get("gone") or [] if isinstance(g, str) and g in self.seen]
            for g in gone:
                if note.get("kind") == "error" and not note.get("sold_out"):
                    self.refused.add(g)
                else:
                    self.gone.add(g)
                changed = True
            if note.get("sold_out") and not gone:
                if self.mark_sold(args):
                    changed = True
                if note.get("kind") == "change" and not note.get("added") and not note.get("dropped") and budget is None:
                    self.sold_note = line
            if note.get("kind") == "change" or changed:
                self.version += 1
                self.recent.clear()
                for kind in ("code_fits", "leads", "code_order"):
                    # (what is wanted changed: the holds before a decline start over, so the code may order again)
                    self.held.pop(kind, None)
                summary = str(note.get("summary") or line)[:160]
                self.notes.append(summary)
                echoes.append("[note read] " + summary + ". Apply this change to your picks before checking out.")
            elif note.get("kind") == "error":
                echoes.append("[store note] " + str(note.get("summary") or line)[:160])
        return "\n".join(echoes)

    def mark_sold(self, args: dict[str, Any]) -> bool:
        """An unnamed sell-out pinned to the listing the call named, else to the lone cart line."""

        ids = [r for r in self.ids_in(args) if r in self.cart] or list(self.cart)
        if len(ids) == 1:
            self.gone.add(ids[0])
            return True
        return False

    def trail(self) -> str | None:
        """Before a checkout with pointers unresolved: a chain reading over the records; links seen only as rows are
        opened by code; a resolved chain puts its value on the sheet; a link nobody found holds the checkout twice."""

        open_lookups = [l for l in self.lookups if not l.get("value")]
        if not open_lookups or self.trails >= _TRAILS or not self.helpers_may_run():
            return None
        opened = sorted(self.full, key=lambda r: -(self.seen.get(r) or {}).get("opened", 0))[:40]
        rows = [r for r in self.seen if r not in self.full][:20]
        records = "\n".join(f"OPENED {r}: " + self.record_text(r, 700) for r in opened) + "\n" + "\n".join(f"ROW {r}: " + _short((self.seen[r] or {}).get("row"), 250) for r in rows)
        pointers = "\n".join(f"pointer {i + 1} (item {l['item']}, {l['property']}): {l['where']}" for i, l in enumerate(open_lookups))
        verdict = self.read(TRAIL_SYSTEM, f"Request:\n{self.request}\n\nPointers:\n{pointers}\n\nRecords:\n{records}", TRAIL_FORM, 1800, LIGHT)
        self.trails += 1
        to_open: list[str] = []
        missing: list[str] = []
        for chain in verdict.get("chains") or []:
            if not isinstance(chain, dict) or not isinstance(chain.get("pointer"), int) or not 1 <= chain["pointer"] <= len(open_lookups):
                continue
            lookup = open_lookups[chain["pointer"] - 1]
            links = [x for x in chain.get("links") or [] if isinstance(x, str)]
            if chain.get("done") and chain.get("value") and links and all(x in self.full for x in links):
                lookup["value"] = str(chain["value"]).strip()[:120]
                item = self.items[lookup["item"] - 1]
                item["told"][lookup["property"]] = lookup["value"]
                self.version += 1
                self.trace.append({"pointer_resolved": {"property": lookup["property"], "value": lookup["value"], "links": links}})
                continue
            to_open += [x for x in links if x in self.seen and x not in self.full and x not in to_open]
            if not links:
                missing.append(lookup["property"])
        if to_open and self.steps_left() > self.reserve() + 3:
            self.open_ids(to_open[:3], keep=self.reserve() + 2)
            return self.trail()
        if missing and self.trail_holds < 2:
            self.trail_holds += 1
            return ("held: the value of " + ", ".join(missing) + " is on another listing's page (see the sheet): find that page, open it, apply its value, then finish.")
        return None

    def page_on(self, name: str, args: dict[str, Any], observation: Any) -> str:
        """Under a cheapest / most / least rule, the rest of a search's pages: read by code when short, else the exact
        next-page argument is handed to the conductor."""

        if self.pick not in ("cheapest", "most", "least") or not self.next_key or not isinstance(observation, dict):
            return ""
        token = observation.get(self.next_key)
        if token in (None, "", False, 0) or (self.next_param and args.get(self.next_param) not in (None, "", 1)):
            return ""
        shown = sum(len(rows) for rows in self.dict_lists(observation, 1) if self.id_key and isinstance(rows[0].get(self.id_key), str))
        total = observation.get(self.count_key) if self.count_key else None
        per_page = max(1, shown)
        rest = int((total - shown + per_page - 1) // per_page) if isinstance(total, (int, float)) and not isinstance(total, bool) and total > shown else 1
        if rest > 2 or self.pages_read + rest > _AUTO_PAGES or self.steps_left() - self.reserve() - rest < 12 or not self.next_param:
            return (f"\n[more results] {total if total is not None else 'more'} matches in all, {shown} shown; the next page takes {self.next_param or 'the next-page parameter'} = {json.dumps(token)}. "
                    "Narrow the search, or page through when the pick depends on every row.")
        lines = []
        self.last_pages = []
        for _ in range(rest):
            if token in (None, "", False, 0) or self.finished or self.steps_left() - self.reserve() < 12:
                break
            page_args = {**args, self.next_param: token}
            observations = self.world([{"name": name, "args": page_args}])
            if not observations:
                break
            self.pages_read += 1
            page = observations[0]
            self.last_pages.append(page)
            self.absorb(name, page_args, page)
            lines.append(f"page with {self.next_param}={json.dumps(token)}: " + _short(page, 1800))
            token = page.get(self.next_key) if isinstance(page, dict) else None
        return ("\n[next pages read by the code]\n" + "\n".join(lines)) if lines else ""

    def rows_by_price(self, observation: Any) -> str:
        if self.pick != "cheapest":
            return ""
        ids = [r for obs in [observation] + list(self.last_pages) for rows in self.dict_lists(obs, 1) for r in [x.get(self.id_key) for x in rows] if isinstance(r, str) and r in self.seen]
        self.last_pages = []
        wanted = {self.stem(w) for it in self.items for w in re.findall(r"[a-z0-9]+", _fold(it.get("name")) + " " + _fold(it.get("shelf"))) if len(w) > 2}
        def of_kind(r: str) -> bool:
            label = (self.seen.get(r) or {}).get("category")
            return not label or not wanted or bool({self.stem(w) for w in re.findall(r"[a-z0-9]+", _fold(label))} & wanted)
        priced = sorted({r for r in ids if self.buyable(r) and self.price_of(r) is not None and of_kind(r)}, key=lambda r: self.price_of(r))
        if not priced:
            return ""
        return "\n[buyable rows here, cheapest first] " + "; ".join(f"{r} {self.price_of(r):g}" + (" (opened)" if r in self.full else "") for r in priced[:10])

    def title_to_id(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """A string argument that is a seen listing's title (not an id) replaced by the id."""

        titles = {_fold(row.get("title")): rid for rid, row in self.seen.items() if row.get("title")}
        props = (self.schemas.get(name) or {}).get("properties") or {}
        out = dict(args)
        for k, v in args.items():
            if isinstance(v, str) and v not in self.seen and _fold(v) in titles and isinstance(props.get(k), dict) and "enum" not in props[k]:
                out[k] = titles[_fold(v)]
            elif isinstance(v, list) and isinstance(props.get(k), dict):
                out[k] = [titles.get(_fold(x), x) if isinstance(x, str) and x not in self.seen else x for x in v]
        return out

    def cheaper_rows(self) -> None:
        """Before a cheapest checkout: per cart line, up to two unread buyable rows of its category priced below it are
        opened (three in all), so the check can weigh them."""

        if self.pick != "cheapest" or not self.cart:
            return
        picks: list[str] = []
        for rid in self.cart:
            mine = self.price_of(rid)
            if mine is None:
                continue
            cat = (self.seen.get(rid) or {}).get("category")
            cheaper = [r for r, row in self.seen.items() if r not in self.full and self.buyable(r) and self.price_of(r) is not None and self.price_of(r) < mine
                       and (not cat or row.get("category") == cat)]
            cheaper.sort(key=lambda r: self.price_of(r))
            picks += [r for r in cheaper[:2] if r not in picks]
        picks = picks[:_CHEAPER_OPENS]
        if picks and self.steps_left() > self.reserve() + 4:
            self.open_ids(picks, keep=self.reserve() + 4)

    # -- the code check of one listing ---------------------------------------------------------
    def conditions_of(self, item: int) -> list[tuple[str, str]]:
        it = self.items[item - 1] if 1 <= item <= len(self.items) else {"musts": {}, "told": {}}
        return list((it.get("musts") or {}).items()) + list((it.get("told") or {}).items())

    def judge(self, rid: str) -> dict[str, Any] | None:
        """The light reader's verdict on one opened listing against its item's conditions, cached per version; a
        condition it calls unknown counts as met when the wanted value's words stand in the record."""

        known = self.verdicts.get(rid)
        if known and known.get("version") == self.version:
            return known
        if rid not in self.full or not self.helpers_may_run():
            return known
        item = self.item_of(rid)
        conds = self.conditions_of(item)
        text = self.record_text(rid, 1500)
        user = (f"Product wanted: {self.items[item - 1]['name']}" + (f" (shelf: {self.items[item - 1].get('shelf')})" if self.items[item - 1].get("shelf") else "")
                + "\n\nConditions:\n" + ("\n".join(f"{i + 1}. {p}: {v}" for i, (p, v) in enumerate(conds)) or "(none)") + f"\n\nRecord:\n{text}")
        read = self.read(VERDICT_SYSTEM, user, VERDICT_FORM, 500, LIGHT, think=False)
        if not read:
            return known
        folded = _fold(text)
        problems: list[str] = []
        unmet = False
        states = {}
        for c in read.get("checks") or []:
            if isinstance(c, dict) and isinstance(c.get("condition"), int) and 1 <= c["condition"] <= len(conds):
                states[c["condition"]] = c.get("verdict")
        for i, (p, v) in enumerate(conds, 1):
            state = states.get(i, "unknown")
            if state == "unknown" and _fold(v) and _fold(v) in folded:
                state = "met"
            if state == "unmet":
                problems.append(f"{p} not met"); unmet = True
            elif state != "met":
                problems.append(f"{p} not shown")
        is_item = read.get("is_item") is not False
        if not is_item:
            problems.insert(0, "not the kind of product wanted")
        if read.get("in_stock") is False:
            self.gone.add(rid)
        verdict: dict[str, Any] = {}
        verdict["version"] = self.version
        verdict["item"] = item
        verdict["fits"] = is_item and not problems
        verdict["unmet"] = unmet or not is_item
        verdict["problems"] = problems[:5]
        self.verdicts[rid] = verdict
        return verdict

    def check_lines(self, ids: list[str]) -> str:
        lines = []
        for rid in ids[:4]:
            v = self.judge(rid)
            if not v:
                continue
            price = self.price_of(rid)
            lines.append(f"[check] {rid}" + (f" at {price:g}" if price is not None else "") + (" SOLD OUT" if not self.buyable(rid) else "")
                         + (f" (item {v['item']})" if len(self.items) > 1 else "") + ": " + ("fits every condition" if v["fits"] else "; ".join(v["problems"])))
        return "\n".join(lines)

    def judge_item(self, item: int, limit: int = 4) -> None:
        """The code's verdicts for its own ending: the cheapest opened buyable listings of the item not yet judged at
        this version, a few at a time (the conductor is shown evidence lines only, never these verdicts)."""

        pending = [r for r in self.full if self.buyable(r) and self.item_of(r) == item and (self.verdicts.get(r) or {}).get("version") != self.version]
        pending.sort(key=lambda r: self.price_of(r) if self.price_of(r) is not None else 1e12)
        for r in pending[:limit]:
            self.judge(r)

    def affordable(self, rid: str, n: int, room: float | None) -> bool:
        """Whether n of the listing fit in the room the allowance leaves (unknown price or no allowance: yes)."""

        price = self.price_of(rid)
        return room is None or price is None or price * n <= room + 1e-9

    def room_for(self, item: int, picks: dict[str, int]) -> float | None:
        """What the allowance leaves for an item once the other items' picks are paid for; None without an allowance
        or with a pick of unknown price."""

        if self.budget is None:
            return None
        others = 0.0
        for rid, n in picks.items():
            if self.item_of(rid) != item:
                price = self.price_of(rid)
                if price is None:
                    return None
                others += price * n
        return self.budget - others

    def replace_unmet(self) -> str | None:
        """Every cart line judged; a line whose verdict says a stated condition is not met is held once, with the
        fitting listing already read named as the swap — or, when none is read and steps allow, after one code search
        per unmet condition (its value and the product word), the cheapest new rows of the item opened and judged.
        Returns the hold text, or None when no line is unmet or nothing better was found (the order proceeds)."""

        for r in list(self.cart):
            if r in self.full and self.helpers_may_run():
                self.judge(r)
        unmet = [r for r in self.cart if (self.verdicts.get(r) or {}).get("version") == self.version and self.verdicts[r].get("unmet")]
        silent = [r for r in self.cart if r not in unmet and (self.verdicts.get(r) or {}).get("version") == self.version and not self.verdicts[r].get("fits")
                  and any(p.endswith("not shown") for p in self.verdicts[r].get("problems") or [])]
        if not (unmet or silent) or self.steps_left() <= self.reserve() + 4 or self.hold("unmet_line") > 1:
            return None         # (near the end the order stands: a hold now would leave no step to act on it)
        lines = []
        for r in unmet + silent:
            item = self.item_of(r)
            problems = [p for p in (self.verdicts[r].get("problems") or []) if p.endswith("not met") or p.startswith("not the kind") or p.endswith("not shown")]
            room = self.room_for(item, {k: v for k, v in self.cart.items() if k != r})
            fits = [f for f in self.fits_for(item, room) if f not in self.cart]
            if not fits and r in unmet and self.search_shape and self.steps_left() > self.reserve() + 14:
                name, param, fixed = self.search_shape
                conds = dict(self.conditions_of(item))
                before = set(self.seen)
                for p in problems[:2]:
                    value = conds.get(p[: -len(" not met")]) if p.endswith(" not met") else None
                    if not value:
                        continue
                    figure = re.search(r"\d[\d.,]*\s*[a-zA-Z]{0,6}", str(value))
                    plain = figure.group(0).strip() if figure else _fold(value)
                    query = f"{plain} {self.items[item - 1]['name']}".strip()[:80]
                    sargs, bad = schema_problems({**fixed, param: query}, self.schemas[name])
                    if bad or self.steps_left() <= self.reserve() + 6:
                        continue
                    observations = self.world([{"name": name, "args": sargs}])
                    if not observations:
                        break
                    self.absorb(name, sargs, observations[0])
                    self.trace.append({"unmet_search": query})
                new = [x for x in self.seen if x not in before and x not in self.full and self.buyable(x) and self.item_of(x) == item and self.affordable(x, self.items[item - 1].get("quantity") or 1, room)]
                new.sort(key=lambda x: self.price_of(x) if self.price_of(x) is not None else 1e12)
                if new and self.steps_left() > self.reserve() + 4:
                    self.open_ids(new[:3], keep=self.reserve() + 2)
                    self.judge_item(item, limit=4)
                    fits = [f for f in self.fits_for(item, room) if f not in self.cart]
            if fits:
                lines.append(f"{r} {'fails a stated condition' if r in unmet else 'does not show a stated condition'} ({'; '.join(problems[:2]) or 'see its record'}); {fits[0]} satisfies every condition read: put it in the cart instead")
        if not lines:
            return None
        return "held before ordering by the code's reading of the cart lines: " + ". ".join(lines) + ". Then order again."

    def fits_for(self, item: int, room: float | None = None) -> list[str]:
        """Opened listings the code check passed for the item at this version, within the room, cheapest first."""

        self.judge_item(item)
        n = self.items[item - 1].get("quantity") or 1 if 1 <= item <= len(self.items) else 1
        ids = [r for r, v in self.verdicts.items() if v.get("fits") and v.get("version") == self.version and v.get("item") == item and self.buyable(r) and self.affordable(r, n, room)]
        return sorted(ids, key=lambda r: self.price_of(r) if self.price_of(r) is not None else 1e12)

    def best_for(self, item: int, room: float | None = None) -> str | None:
        """The listing to order for an item when nothing fits: the kind wanted, the fewest problems, no unmet
        condition if any such exists, cheapest first; never one the room cannot cover."""

        fits = self.fits_for(item, room)
        if fits:
            return fits[0]
        n = self.items[item - 1].get("quantity") or 1 if 1 <= item <= len(self.items) else 1
        pool = [(r, v) for r, v in self.verdicts.items() if v.get("item") == item and self.buyable(r) and "not the kind of product wanted" not in v.get("problems", []) and self.affordable(r, n, room)]
        if pool:
            pool.sort(key=lambda rv: (rv[1].get("unmet", True), len(rv[1].get("problems") or []), self.price_of(rv[0]) if self.price_of(rv[0]) is not None else 1e12))
            return pool[0][0]
        # (no listing of the item was judged: the cheapest row seen under its name, since an order of what was found
        #  beats ending with nothing)
        wanted = {self.stem(w) for w in re.findall(r"[a-z0-9]+", _fold(self.items[item - 1].get("name")) + " " + _fold(self.items[item - 1].get("shelf")))}
        rows = []
        for r in self.seen:
            if r in self.verdicts or not self.buyable(r) or self.price_of(r) is None or self.item_of(r) != item or not self.affordable(r, n, room):
                continue
            label = {self.stem(w) for w in re.findall(r"[a-z0-9]+", _fold((self.seen.get(r) or {}).get("category")))}
            if len(self.items) == 1 or label & wanted:
                rows.append(r)
        rows.sort(key=lambda r: self.price_of(r))
        return rows[0] if rows else None

    def best_picks(self) -> dict[str, int]:
        """A full cart from the code's verdicts: the cart's own lines kept where they fit, the best listing for every
        item left; empty when some item has nothing at all."""

        picks: dict[str, int] = {}
        for k, it in enumerate(self.items, 1):
            mine = [r for r in self.cart if self.item_of(r) == k and self.buyable(r)]
            kept = next((r for r in mine if (self.verdicts.get(r) or {}).get("fits") and self.verdicts[r].get("version") == self.version), None)
            rid = kept or self.best_for(k) or (mine[0] if mine else None)
            if not rid:
                return {}
            picks[rid] = it.get("quantity") or 1
        # (over the allowance: each product re-picked within the room the others leave, the dearest line first)
        def total() -> float | None:
            return None if any(self.price_of(r) is None for r in picks) else sum(self.price_of(r) * q for r, q in picks.items())
        if self.budget is not None and picks and total() is not None and total() > self.budget + 1e-9:
            for rid in sorted(picks, key=lambda r: -self.price_of(r) * picks[r]):
                k = self.item_of(rid)
                rest = {r: q for r, q in picks.items() if r != rid}
                other = self.best_for(k, self.room_for(k, rest))
                if other and other != rid:
                    picks = {**rest, other: picks[rid]}
                if total() is not None and total() <= self.budget + 1e-9:
                    break
        return picks

    def learn_search_shape(self, name: str, args: dict[str, Any], new_rows: int) -> None:
        if not self.is_role("search", name, args) or new_rows <= 0:
            return
        props = (self.schemas.get(name) or {}).get("properties") or {}
        text = [k for k, v in args.items() if isinstance(v, str) and re.search(r"[a-z]", v, re.I) and isinstance(props.get(k), dict) and props[k].get("type") == "string" and "enum" not in props[k]]
        if len(text) == 1:
            self.search_shape = (name, text[0], {k: v for k, v in args.items() if k != text[0] and not (isinstance(v, int) and not isinstance(v, bool))})

    def heard_searches(self, limit: int = 3) -> str:
        """One code search per value the shopper stated, in the shape of a search that showed rows: the value and the
        product word; the rows go into the record book and the cheapest are named."""

        if not self.search_shape:
            return ""
        name, param, fixed = self.search_shape
        lines = []
        for k, it in enumerate(self.items, 1):
            for prop, value in (it.get("told") or {}).items():
                key = f"{k}|{prop}|{value}"
                if key in self.heard_searched or len(lines) >= limit or self.steps_left() <= self.reserve() + 8 or self.finished:
                    continue
                self.heard_searched.add(key)
                figure = re.search(r"\d[\d.,]*\s*[a-zA-Z]{0,6}", str(value))
                plain = figure.group(0).strip() if figure else _fold(value)      # (a stated figure with its unit, else the value's words)
                query = f"{plain} {it['name']}".strip()[:80]
                args, problems = schema_problems({**fixed, param: query}, self.schemas[name])
                if problems:
                    continue
                before = set(self.seen)
                observations = self.world([{"name": name, "args": args}])
                if not observations:
                    break
                self.absorb(name, args, observations[0])
                new = [r for r in self.seen if r not in before]
                priced = sorted((r for r in new if self.price_of(r) is not None and self.buyable(r)), key=lambda r: self.price_of(r))
                lines.append(f"[code searched] '{query}' for the stated {prop}: {len(new)} new rows" + (": " + "; ".join(f"{r} {self.price_of(r):g}" for r in priced[:6]) if priced else ""))
        return "\n".join(lines)

    # -- the sheet ----------------------------------------------------------------------------
    def sheet(self) -> str:
        lines = ["[sheet]"]
        for k, item in enumerate(self.items, 1):
            musts = "; ".join(f"{p}: {c}" for p, c in (item.get("musts") or {}).items()) or "no stated condition"
            told = "; ".join(f"{p}: {c}" for p, c in (item.get("told") or {}).items())
            lines.append(f"item {k}: {item.get('name')} x{item.get('quantity') or 1} | conditions: {musts}" + (f" | told by the shopper: {told}" if told else ""))
        if not self.items:
            lines.append("item: the requested product")
        if self.budget is not None:
            lines.append(f"budget: {self.budget:g}" + (f" {self.unit}" if self.unit else "") + " for everything together")
        if self.pick:
            lines.append(f"pick rule: {self.pick}" + (f" {self.pick_measure}" if self.pick_measure else ""))
        for l in self.lookups:
            lines.append(f"lookup for item {l['item']}, {l['property']}: " + (f"resolved = {l['value']}" if l.get("value") else f"the value is {l['where']}; find that page, open it, apply the value"))
        for r in self.rules:
            lines.append("rule: " + r)
        for p in self.process:
            lines.append(f"the shopper asks ({p['when']}): {p['text']}" + (" [done]" if p.get("done") else ""))
        if self.in_stock_only:
            lines.append("sold-out listings are never ordered")
        cart = ", ".join(f"{rid} x{n}" for rid, n in self.cart.items()) or "empty"
        lines.append(f"cart (as booked): {cart}")
        if self.gone or self.refused:
            lines.append("not buyable (sold out or refused by the store): " + ", ".join(sorted(self.gone | self.refused)[:12]))
        if self.notes:
            lines.append("changes so far: " + " | ".join(self.notes[-4:]))
        roles = ", ".join(f"{r} = {s.get('tool')}" for r, s in self.roles.items() if s.get("tool"))
        if roles:
            lines.append("tools by role: " + roles)
        if self.attr_tool:
            lines.append(f"listing attributes (the keys conditions are judged by) are shown by {self.attr_tool}: open a listing with it before judging it")
        else:
            singles = self.listing_tools()
            if len(singles) > 1:
                lines.append("which tool shows a listing's attribute table is not known yet: open one listing with each of " + ", ".join(singles) + " once")
        left = self.steps_left()
        msgs = self.msgs_left()
        lines.append(f"steps left: {left}" + (f" | messages left: {msgs}" if msgs is not None else ""))
        if left <= self.reserve() + 1:
            lines.append("NEXT CALL MUST FINISH: order the cart with the ordering tool, alone.")
        elif left <= 6:
            lines.append("few steps left: put the chosen listing in the cart and finish.")
        elif left <= 12 and not self.cart:
            lines.append("the cart is still empty: choose and cart now.")
        elif left <= self.max_steps // 2 and not self.cart:
            lines.append("half the steps are gone and the cart is empty.")
        return "\n".join(lines)

    # -- the loop ------------------------------------------------------------------------------
    def tool_list(self) -> str:
        return "\n".join(f"- {t['function']['name']}: {t['function'].get('description') or ''} parameters {json.dumps(t['function'].get('parameters') or {}, ensure_ascii=False)[:700]}" for t in self.tools)

    def setup(self) -> None:
        """The request read once into the sheet: products, conditions, money, allowances, lookups, process steps and
        the tools' roles. The heavy reader repeats it only when the light one named no product or no ordering tool."""

        user = "Request:\n" + self.request + "\n\nStore tools:\n" + self.tool_list()
        brief = self.read(BRIEF_SYSTEM, user, BRIEF_FORM, 4000, LIGHT, think=False)
        if (not self.take_brief(brief) or not self.roles.get("ask")) and not self.broke:
            note = ("\n\nNote: the tool that places the order and the tool that ends without an order are not the tool that puts a listing "
                    "in the cart, nor the tools that search, open or compare listings, nor the one that messages the shopper.")
            again = self.read(BRIEF_SYSTEM, user + note, BRIEF_FORM, 2500, HEAVY[1:] + HEAVY[:1], think=False)
            if self.take_brief(again):
                brief = again
        self.trace.append({"notes": {"brief": brief, "roles": self.roles, "items": self.items, "process": self.process}})

    def take_brief(self, brief: dict[str, Any]) -> bool:
        """The sheet filled from a brief; True when it names a product and an ordering tool."""

        if not isinstance(brief, dict):
            return False
        items = []
        for x in brief.get("items") or []:
            if not isinstance(x, dict) or not str(x.get("name") or "").strip():
                continue
            musts = {str(k).strip(): str(v).strip() for k, v in (x.get("musts") or {}).items() if isinstance(x.get("musts"), dict) and str(v).strip()}
            quantity = x.get("quantity")
            items.append({"name": str(x["name"]).strip()[:80], "shelf": str(x["shelf"]).strip()[:60] if x.get("shelf") else None,
                          "quantity": quantity if isinstance(quantity, int) and not isinstance(quantity, bool) and quantity > 0 else 1,
                          "musts": musts, "told": {}, "free": []})
        merged: list[dict[str, Any]] = []
        for it in items:
            # (one entry per product: a second entry naming the same product or shelf joins the first)
            twin = next((m for m in merged if (_fold(m["name"]) == _fold(it["name"]) or (it.get("shelf") and m.get("shelf") and _fold(m["shelf"]) == _fold(it["shelf"])))
                         and not any(_fold(p) == _fold(q) and _fold(v) != _fold(w) for p, v in m["musts"].items() for q, w in it["musts"].items())), None)
            if twin is None:
                merged.append(it)
            else:
                twin["quantity"] = max(twin["quantity"], it["quantity"])
                for p, v in it["musts"].items():
                    twin["musts"].setdefault(p, v)
        self.items = merged[:6] or [{"name": "the requested product", "shelf": None, "quantity": 1, "musts": {}, "told": {}, "free": []}]
        budget = brief.get("budget")
        self.budget = float(budget) if isinstance(budget, (int, float)) and not isinstance(budget, bool) and budget > 0 else None
        self.unit = str(brief["unit"]).strip()[:30] if brief.get("unit") else None
        self.in_stock_only = brief.get("in_stock_only") is not False
        self.pick = brief.get("pick") if brief.get("pick") in ("cheapest", "most", "least") else None
        self.pick_measure = str(brief["pick_measure"]).strip()[:40] if self.pick in ("most", "least") and brief.get("pick_measure") else None
        self.lookups = [{"property": str(l["property"]).strip(), "item": l["item"] if isinstance(l.get("item"), int) and 1 <= l["item"] <= len(self.items) else 1,
                         "where": str(l.get("where") or "").strip()[:300], "value": None}
                        for l in brief.get("lookups") or [] if isinstance(l, dict) and l.get("property")]
        for l in self.lookups:
            # (a condition that only restates the lookup is the lookup, not a value to match)
            musts = self.items[l["item"] - 1]["musts"]
            for p in [p for p in musts if _fold(p) == _fold(l["property"])]:
                musts.pop(p)
        self.rules = [str(r).strip()[:200] for r in brief.get("rules") or [] if str(r).strip()][:8]
        self.ask_first = bool(brief.get("ask_first"))
        self.process = [{"text": str(p["text"]).strip()[:300], "when": p["when"], "messages": p["messages"] if isinstance(p.get("messages"), int) and p["messages"] >= 0 else 0, "done": []}
                        for p in brief.get("process") or [] if isinstance(p, dict) and p.get("text") and p.get("when") in ("choose", "order", "end")][:6]
        self.quiet_decline = brief.get("quiet_decline") if isinstance(brief.get("quiet_decline"), bool) else None
        self.roles = {}
        given = brief.get("roles") if isinstance(brief.get("roles"), dict) else {}
        ending_args = {k: (brief.get(k) if isinstance(brief.get(k), dict) else given.get(k) if isinstance(given.get(k), dict) else None) for k in ("buy_args", "refuse_args")}
        for role, tool in given.items():
            if role in ("search", "open", "compare", "ask", "cart", "cart_view", "buy", "refuse") and tool in self.schemas:
                fixed = ending_args["buy_args"] if role == "buy" else ending_args["refuse_args"] if role == "refuse" else None
                self.roles[role] = {"tool": tool, "args": self.clean_args(tool, fixed)}
        if self.roles.get("buy") and self.roles.get("refuse") and self.roles["buy"]["tool"] == self.roles["refuse"]["tool"] and self.roles["buy"]["args"] == self.roles["refuse"]["args"]:
            # (one tool for both endings with no telling argument: its first enum parameter tells them apart, the
            #  first value for the order and the second for the decline; without one the decline is dropped)
            props = (self.schemas.get(self.roles["buy"]["tool"]) or {}).get("properties") or {}
            mode = next((k for k, p in props.items() if isinstance(p, dict) and isinstance(p.get("enum"), list) and len(p["enum"]) >= 2 and k not in self.roles["buy"]["args"]), None)
            if mode:
                self.roles["buy"]["args"] = {**self.roles["buy"]["args"], mode: props[mode]["enum"][0]}
                self.roles["refuse"]["args"] = {**self.roles["refuse"]["args"], mode: props[mode]["enum"][1]}
                self.trace.append({"ending_args_guessed": {mode: props[mode]["enum"][:2]}})
            else:
                self.roles.pop("refuse")
        messages = brief.get("messages")
        if isinstance(messages, int) and not isinstance(messages, bool) and messages >= 0:
            self.msg_limit = messages
        elif self.roles.get("ask"):
            self.msg_limit = 3
        steps = brief.get("steps")
        if isinstance(steps, int) and not isinstance(steps, bool) and 10 <= steps < self.max_steps:
            self.step_cap = steps
            self.max_steps = steps
        buy = (self.roles.get("buy") or {}).get("tool")
        clash = [r for r in ("cart", "cart_view", "open", "search", "ask", "compare") if buy and (self.roles.get(r) or {}).get("tool") == buy]
        if clash:
            # (the ordering tool named as another role's tool: the brief is wrong about the endings; the heavy reader
            #  repeats it, and a repeat that keeps the clash loses the ordering role rather than the cart role)
            self.trace.append({"roles_clash": {"buy": buy, "also": clash}})
            self.roles.pop("buy", None)
            if (self.roles.get("refuse") or {}).get("tool") == buy:
                self.roles.pop("refuse", None)
        return bool(items) and bool(self.roles.get("buy"))

    def clean_args(self, tool: str, fixed: Any) -> dict[str, Any]:
        """A role's fixed arguments kept only where the tool's schema has the parameter, enum values only from the enum."""

        props = (self.schemas.get(tool) or {}).get("properties") or {}
        out: dict[str, Any] = {}
        for k, v in (fixed if isinstance(fixed, dict) else {}).items():
            p = props.get(k)
            if not isinstance(p, dict):
                continue
            v = _repair(v, p)
            if "enum" in p and v not in p["enum"]:
                continue
            if "enum" not in p and isinstance(v, str):
                continue        # (a free text is a sample, not a discriminator: the call's own words go there)
            out[k] = v
        return out

    def tools_offered(self) -> list[dict[str, Any]]:
        """The store's tools the conductor may call now: at the tail only the ending tools."""

        left = self.steps_left()
        if left <= self.reserve():
            keep = {self.role_tool("buy"), self.role_tool("refuse")} - {None}
            return [t for t in self.tools if t["function"]["name"] in keep] or self.tools
        return self.tools

    def run(self) -> list[dict[str, Any]]:
        if self.finished:
            return self.trace
        self.setup()
        self.work(_LOOP_TURNS)
        self.finish()
        if not self.finished and self.refusal_step is not None and self.steps_left() > 0 and not self.broke:
            # (a forced checkout the store refused: the conductor gets a few more turns to fix the cart)
            self.work(_EXTRA_TURNS)
            self.finish()
        self.trace.append({"notes": {"end": True, "steps": self.step, "cost": round(self.cost, 5), "calls": self.calls, "stop": getattr(self, "stop", None),
                                     "seconds": round(time.monotonic() - self.started, 1), "cart": self.cart, "items": self.items,
                                     "budget": self.budget, "version": self.version, "changes": self.notes[-6:]}})
        return self.trace

    def work(self, turns: int) -> None:
        if not hasattr(self, "messages"):
            opening = self.view.get("opening")
            user = "Shopper's request:\n" + self.request
            if opening is not None:
                user += "\n\nOpening observation:\n" + _short(opening, _OBS_CHARS)
                self.records.append(("opening", {}, opening))
            self.messages: list[dict[str, Any]] = [{"role": "system", "content": CONDUCT_SYSTEM}, {"role": "user", "content": user + "\n\n" + self.sheet()}]
        misses = idle = 0
        self.stop = "turns"
        for _ in range(turns):
            if self.finished or self.steps_left() <= 0 or self.broke:
                self.stop = "finished" if self.finished else "steps" if self.steps_left() <= 0 else "broke"
                break
            if self.seconds_left() < _CLOSE_SECONDS or self.cost >= _SPEND_HARD:
                self.stop = "time" if self.seconds_left() < _CLOSE_SECONDS else "spend"
                break
            self.trim()
            if self.interview_due():
                summary = self.hold_interview()
                if self.finished:
                    break
                self.messages.append({"role": "user", "content": (summary or "(the shopper was asked; no reply came)") + "\n" + self.sheet()})
                self.heard_text = ""
            elif not self.asked_once and ((self.step >= 12 and self.full) or (self.step >= 6 and (not self.roles.get("ask") or self.questions_left() <= 0))):
                # (a beacon only when the interview cannot come: no ask tool or no message left, or records by step 12 and still no interview)
                self.beacon()
            self.turns += 1
            message = self.chat(self.messages, self.tools_offered(), LIGHT, 3000)
            if message is None:
                misses += 1
                if misses >= _MISSES or self.spent_out:
                    self.stop = "spend" if self.spent_out else "misses"
                    break
                continue
            misses = 0
            calls = [c for c in message.get("tool_calls") or [] if isinstance(c, dict)]
            if not calls:
                idle += 1
                self.messages.append({"role": "assistant", "content": message.get("content") or "..."})
                self.messages.append({"role": "user", "content": "Call one of the store's tools now.\n" + self.sheet()})
                if idle >= _IDLE:
                    self.stop = "idle"
                    break
                continue
            idle = 0
            self.messages.append({"role": "assistant", "content": message.get("content") or None, "tool_calls": [
                {"id": c.get("id") or f"c{i}", "type": "function", "function": {"name": (c.get("function") or {}).get("name"),
                 "arguments": (c.get("function") or {}).get("arguments") if isinstance((c.get("function") or {}).get("arguments"), str) else json.dumps((c.get("function") or {}).get("arguments") or {})}}
                for i, c in enumerate(calls)]})
            results = self.carry_calls(self.messages[-1]["tool_calls"])
            for index, (call_id, text) in enumerate(results):
                if index == len(results) - 1:
                    text += "\n" + self.sheet()
                self.messages.append({"role": "tool", "tool_call_id": call_id, "content": text})

    def trim(self) -> None:
        """Older tool results cut down once the conversation is long; the sheet carries what was learned."""

        if sum(len(str(m.get("content") or "")) for m in self.messages) < _HISTORY_CHARS:
            return
        tool_turns = [m for m in self.messages if m.get("role") == "tool" and isinstance(m.get("content"), str) and len(m["content"]) > 600 and not m.get("_cut")]
        for m in tool_turns[: max(1, len(tool_turns) // 2)]:
            ids = ", ".join(self.ids_in(_json_object(m["content"]) or {}) or re.findall(r"\b(p-[0-9a-f]{8,})\b", m["content"])[:20])
            m["content"] = m["content"][:_OLD_CHARS] + "...(older result cut; listings it held: " + ids + ")"
            m["_cut"] = True

    def carry_calls(self, calls: list[dict[str, Any]]) -> list[tuple[str, str]]:
        """Every call guarded, the runnable ones sent in order; a finishing call must be the only one."""

        replies: list[tuple[str, str]] = []
        runnable: list[tuple[str, str, dict[str, Any]]] = []
        alone = len(calls) == 1
        self.turn_cart_ids = set()
        for call in calls:
            function = call.get("function") or {}
            name = function.get("name")
            args = _json_object(function.get("arguments")) or {}
            args, refused = self.guard(name, args, alone)
            if refused:
                replies.append((call["id"], "Not run (no step used): " + refused))
                self.trace.append({"held": name, "reason": refused[:300]})
                continue
            runnable.append((call["id"], name, args))
        for call_id, name, args in runnable:
            if self.finished:
                replies.append((call_id, "Not run: the session is over."))
                continue
            observations = self.world([{"name": name, "args": args}])
            if not observations:
                replies.append((call_id, "Not run: the store did not answer this call."))
                continue
            replies.append((call_id, self.after(name, args, observations[0])))
        position = {c["id"]: i for i, c in enumerate(calls)}
        return sorted(replies, key=lambda r: position.get(r[0], 0))

    def guard(self, name: str, args: dict[str, Any], alone: bool) -> tuple[dict[str, Any], str | None]:
        """The call as it will run, or the reason it is not run (no step is spent on it)."""

        if name not in self.schemas:
            return args, f"'{name}' is not one of the store's tools."
        args, problems = schema_problems(args if isinstance(args, dict) else {}, self.schemas[name])
        if problems:
            return args, "invalid arguments: " + "; ".join(problems[:5]) + "."
        kind = self.ending(name, args)
        if kind and not alone:
            return args, "an order or a decline must be sent alone, after the results of your other calls."
        if not kind and not self.is_role("ask", name, args):
            args = self.title_to_id(name, args)
            signature = name + json.dumps(args, sort_keys=True)
            ids = self.ids_in(args)
            if (self.is_role("open", name, args) or self.is_role("compare", name, args)) and ids and all(r in self.gone for r in ids):
                return args, f"{', '.join(ids)} is sold out: its page cannot help; choose another listing."
            if self.is_role("compare", name, args) and self.compare_param() and isinstance(args.get(self.compare_param()), list):
                args[self.compare_param()] = list(dict.fromkeys(args[self.compare_param()]))
                if len(args[self.compare_param()]) > self.room_for_compare():
                    return args, f"open at most {self.room_for_compare()} listings in one compare call (the store returns nothing for more)."
            cached = self.results.get(signature)
            if cached and cached[0] == self.version and not self.is_role("cart", name, args) and not self.is_role("cart_view", name, args):
                return args, "this exact read was made since anything last changed; its result is in the conversation and the sheet." + (" The record: " + " || ".join(f"{r}: {self.record_text(r, 1200)}" for r in ids[:2]) if ids else "")
            if (self.is_role("open", name, args) or self.is_role("compare", name, args)) and ids and all(r in self.full and name in ((self.seen.get(r) or {}).get("by") or []) and (self.seen.get(r) or {}).get("opened", -1) >= self.cart_at for r in ids) and not any((r, self.refusal_step) not in self.reread and self.refusal_step is not None for r in ids):
                return args, f"{', '.join(ids)} is already open with this tool and nothing changed since; the record: " + " || ".join(f"{r}: {self.record_text(r, 1200)}" for r in ids[:2])
            if signature in self.recent and not self.is_role("cart", name, args) and not (self.is_role("cart_view", name, args) and self.cart_at > self.viewed_at):
                return args, "this exact call was just made; its result is above."
        if not kind and self.steps_left() <= self.reserve():
            return args, "the last steps are kept for finishing: order the cart now with the ordering tool."
        if self.is_role("ask", name, args):
            if not self.asked_once and self.questions_left() > 0 and self.steps_left() > 8 and not self.broke:
                summary = self.hold_interview()
                self.heard_text = ""
                return args, "messages are sent by the code: the shopper was interviewed now instead; read the replies, re-check your pick against every condition now in force, then go on.\n" + summary
            if self.msg_limit is not None and self.messages_sent >= self.msg_limit:
                return args, "the shopper allows no more messages."
            return args, ("messages are sent by the code: the shopper's needs were asked in the interview, and the steps the shopper asked for at the "
                          "order or the ending are done by the code when you call the ending tool. Go on with the store's tools.")
        if (kind or self.is_role("cart", name, args)) and not self.asked_once and self.roles.get("ask") and self.questions_left() > 0 and self.steps_left() > 8 and not self.broke:
            summary = self.hold_interview()
            self.heard_text = ""
            return args, "the shopper is asked first; read the replies, re-check your pick against every condition now in force, then go on.\n" + summary
        if (kind or self.is_role("cart", name, args)) and not self.asked_once:
            self.beacon()
        if self.is_role("cart", name, args) and self.ids_in(args):
            repeated = [r for r in self.ids_in(args) if r in self.turn_cart_ids]
            if repeated:
                return args, f"{', '.join(repeated)} was already changed in the cart by a call of this same reply; view the cart and decide once."
            self.turn_cart_ids.update(self.ids_in(args))
        if self.is_role("cart_view", name, args) and self.viewed_at >= 0 and self.viewed_at >= self.cart_at:
            return args, "the cart has not changed since it was last viewed: " + (self.cart_state() if self.cart else "empty")
        if self.is_role("cart", name, args) and self.ids_in(args) and any(not self.buyable(r) for r in self.ids_in(args)):
            return args, f"{', '.join(r for r in self.ids_in(args) if not self.buyable(r))} is sold out or was refused by the store: choose another listing."
        if self.is_role("cart", name, args) and not self.cart:
            go, text = self.process_gate("choose")
            if not go:
                return args, text
        if kind == "buy":
            return self.before_order(args)
        if kind == "refuse":
            return self.before_decline(args)
        return args, None

    def hold(self, kind: str) -> int:
        self.held[kind] = self.held.get(kind, 0) + 1
        return self.held[kind]

    def before_order(self, args: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
        """The holds before an order, none spending a conductor step: an empty cart, sold lines, lines never read,
        the cart view after a change, the heavy check, the process gate."""

        if not self.cart:
            return args, "the cart is empty: put the chosen listing of each wanted product in the cart first."
        sold = [r for r in self.cart if not self.buyable(r)]
        if sold and self.hold("sold_line") <= 2:
            return args, f"{', '.join(sold)} in the cart is sold out or was refused: replace it with a fitting listing, then order."
        unread = [r for r in self.cart if r not in self.full]
        if unread and self.steps_left() > self.reserve() + 2:
            self.open_ids(unread, keep=self.reserve() + 1)
        if self.refusal_step is not None:
            stale = [r for r in self.cart if (r, self.refusal_step) not in self.reread]
            if stale and self.steps_left() > self.reserve() + 2:
                # (after a refusal the cart lines are read again once: a sold-out or changed line shows there)
                self.reread_lines(stale, keep=self.reserve() + 1)
                for r in stale:
                    self.reread.add((r, self.refusal_step))
        view = self.role_tool("cart_view")
        if view and self.cart_at > self.viewed_at and self.steps_left() > self.reserve() + 1:
            vargs, problems = schema_problems(self.role_args("cart_view"), self.schemas[view])
            if not problems:
                observations = self.world([{"name": view, "args": vargs}])
                self.viewed_at = self.step
                if observations:
                    self.cart_from_view(observations[0])
                    if not self.cart:
                        return args, "the store's cart is empty (see the cart view): put the chosen listing in the cart first."
        held_unmet = self.replace_unmet()
        if held_unmet:
            return args, held_unmet
        if self.pick == "cheapest" and self.hold("wider") == 1 and self.steps_left() >= 14 + 2 * len(self.items):
            return args, ("held once before a cheapest checkout: search the item under other product names and page through the results, and open "
                          "any buyable row priced below the cart line that could fit; then order the cheapest listing that satisfies every condition.")
        self.cheaper_rows()
        held_trail = self.trail()
        if held_trail:
            return args, held_trail
        verdict = self.verify_cart()
        if verdict.get("unread"):
            if self.hold("check_unread") <= 1:
                return args, "held: the cart could not be checked by the code; re-read each cart line's record against every condition in force yourself, then order again."
        elif not verdict.get("ok"):
            held = self.hold("check")
            if held <= 2:
                better = ", ".join(verdict.get("better") or [])
                return args, ("held before ordering by a closer reading of the records: " + "; ".join(verdict.get("problems") or ["the cart does not satisfy every condition"])
                              + (f". Listings shown that would do better: {better}" if better else "") + ". Fix the cart (or show the reading is wrong by re-reading the records), then order again.")
            if held == 3:
                # (the conductor kept the cart through two holds: the code tries the better listing itself, once)
                for rid in verdict.get("better") or []:
                    item = self.item_of(rid)
                    picks = {r: n for r, n in self.cart.items() if self.item_of(r) != item}
                    picks[rid] = self.items[item - 1]["quantity"] if 1 <= item <= len(self.items) else 1
                    if self.buyable(rid) and self.verify_cart(picks, forced=True).get("ok"):
                        self.confirmed = rid
                        text = self.complete_cart()
                        if text:
                            return args, text
                        break
        go, text = self.process_gate("order")
        if not go:
            return args, text
        return args, None

    def before_decline(self, args: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
        """The holds before a decline: the audit, the leads, an independent check of the first lead, the built cart,
        one more search, the code's own completion; then the quiet text and the ending gate."""

        left = self.steps_left()
        if left <= 5 and self.role_tool("buy") and self.role_tool("cart"):
            # (a cart holding a line for every product — the one in the cart, else the fullest built earlier — is
            #  ordered, not abandoned: running short of steps shows nothing about the store)
            full = next((c for c in (self.cart, self.built) if c and len({self.item_of(r) for r in c if self.buyable(r)}) >= len(self.items)), None)
            if full:
                text = self.order_best(dict(full) if full is not self.cart else None)
                if text:
                    return args, text
        code_fits = [r for k in range(1, len(self.items) + 1) for r in self.fits_for(k)[:2]]
        if code_fits and self.hold("code_fits") == 1:
            return args, ("held: the code's check passed these opened listings for every condition: " + ", ".join(code_fits[:6])
                          + ". Put the fitting one of each product in the cart and order; end without an order only when no product can be fitted.")
        if left > 5 and self.audits == 0 and self.helpers_may_run() and not code_fits:
            found = self.audit()
            if found["open"] and self.steps_left() > 6:
                self.open_ids(found["open"], keep=5)
                if self.helpers_may_run():
                    found = self.audit()
            self.leads = [r for ids in found["fits"].values() for r in ids if r not in self.leads]
        if code_fits and not self.leads:
            self.leads = [r for r in code_fits if r not in self.leads]
        lead_holds = self.hold("leads") if self.leads else 0
        if lead_holds == 1:
            return args, "held: these opened listings read as fitting every condition: " + ", ".join(self.leads[:4]) + ". Put the fitting one of each product in the cart and order."
        if lead_holds == 2 and self.helpers_may_run():
            lead = self.leads[0]
            item = self.item_of(lead)
            picks = {rid: n for rid, n in self.cart.items() if self.item_of(rid) != item}
            picks[lead] = self.items[item - 1]["quantity"] if 1 <= item <= len(self.items) else 1
            verdict = self.verify_cart(picks, forced=True)
            if verdict.get("ok"):
                self.confirmed = lead
                return args, f"held: an independent check passed {lead} for every condition. Cart it and order (a decline is right only when nothing in the store can fit)."
        if left <= 5 and self.built and not self.cart and self.hold("built") == 1 and len({self.item_of(r) for r in self.built}) >= len(self.items):
            return args, "held: a full cart was built earlier (" + ", ".join(f"{r} x{n}" for r, n in self.built.items()) + "); restore it and order rather than ending with nothing."
        if not self.leads and left > self.max_steps // 3 and self.hold("search_more") == 1:
            return args, "held: before ending without an order, search the item under other product names and browse its category, paging through the results."
        if self.confirmed and self.hold("complete") == 1:
            text = self.complete_cart()
            if text:
                return args, text
        if self.role_tool("buy") and self.role_tool("cart") and self.hold("code_order") == 1:
            # (an order of the best listing found for each product beats ending with nothing: the code carts and orders)
            text = self.order_best()
            if text:
                return args, text
        args = self.quiet_text(args)
        go, text = self.process_gate("end")
        if not go:
            return args, text
        return args, None

    def order_best(self, picks: dict[str, int] | None = None) -> str | None:
        """The best picks (or the given lines) carted and ordered by code; the text for the conductor, None when nothing
        could be done."""

        picks = picks or self.best_picks()
        if not picks or self.finished:
            return None
        calls = self.cart_calls(picks)
        if self.steps_left() < len(calls) + 1:
            return None
        for call in calls:
            observations = self.world([call])
            if not observations or self.finished:
                return None
            self.after(call["name"], call["args"], observations[0])
        if self.cart != picks:
            self.cart = dict(picks)
            self.cart_at = self.step
        go, text = self.process_gate("order")
        if not go:
            return text
        if self.steps_left() < 1:
            return None
        buy_args = self.ending_args("buy")
        observations = self.world([{"name": self.role_tool("buy"), "args": buy_args}])
        self.trace.append({"order_best": {"picks": picks, "result": _short(observations[0] if observations else None, 300)}})
        if self.finished:
            return "Not run: the code carted the best listing found for each product and placed the order instead."
        if not observations:
            return None
        # (refused: the refusal's words are read — a revised allowance, a line gone — and, when they changed what is
        #  in force, the picks are made again under it and ordered once more)
        version = self.version
        self.after(self.role_tool("buy"), buy_args, observations[0])
        if self.version != version:
            again = self.best_picks()
            calls = self.cart_calls(again) if again and again != self.cart else []
            if calls and self.steps_left() >= len(calls) + 1:
                for call in calls:
                    observations = self.world([call])
                    if not observations or self.finished:
                        return None
                    self.after(call["name"], call["args"], observations[0])
                if self.cart != again:
                    self.cart = dict(again)
                    self.cart_at = self.step
                observations = self.world([{"name": self.role_tool("buy"), "args": buy_args}])
                self.trace.append({"order_best": {"picks": again, "result": _short(observations[0] if observations else None, 300), "repick": True}})
                if self.finished:
                    return "Not run: the code re-picked the cart under the store's reply and placed the order instead."
                if observations:
                    self.after(self.role_tool("buy"), buy_args, observations[0])
        return "the code carted the best listings found and tried to order; the store refused (read why above); fix the cart and order."

    def after(self, name: str, args: dict[str, Any], observation: Any) -> str:
        """The result as the conductor sees it, with the sheet's bookkeeping done."""

        kind = self.ending(name, args)
        if self.is_role("ask", name, args):
            self.messages_sent += 1
            question = next((v for v in args.values() if isinstance(v, str)), "")
            self.asked.append((question[:200], _short(observation, 300)))
            self.done_log.append(f"message sent at step {self.step}: {question[:160]} -> {_short(observation, 200)}")
        if self.is_role("cart", name, args):
            self.book_cart(name, args)
            if len(self.cart) >= len(self.built) and len({self.item_of(r) for r in self.cart}) >= len({self.item_of(r) for r in self.built}):
                self.built = dict(self.cart)
        if self.is_role("cart_view", name, args):
            self.viewed_at = self.step
        if self.is_role("cart_view", name, args) or self.is_role("cart", name, args):
            self.cart_from_view(observation, whole=self.is_role("cart_view", name, args))
        signature = name + json.dumps(args, sort_keys=True)
        self.recent = (self.recent + [signature])[-3:]
        extra = ""
        if not self.is_role("ask", name, args) and not kind:
            before = set(self.full)
            before_seen = set(self.seen)
            self.absorb(name, args, observation)
            opened = [r for r in self.ids_in(args) if r in self.full and (r not in before or self.full[r] is not None)]
            if opened and (self.is_role("open", name, args) or self.is_role("compare", name, args)):
                self.results[signature] = (self.version, observation)
                extra += "\n" + self.evidence(opened)
            extra += self.learn_compare_room(name, args, opened)
            if self.is_role("search", name, args):
                self.learn_search_shape(name, args, len(self.seen) - len(before_seen))
                extra += self.page_on(name, args, observation)
                extra += self.rows_by_price(observation)
        text = _short(observation, _OBS_CHARS)
        if self.finished:
            text += " [the session is over]"
        elif kind == "buy":
            self.refusal_step = self.step
            text = "THE ORDER WAS NOT PLACED: the store refused it and the session is still open. Read why, fix the cart, order again.\n" + text
        elif kind == "refuse":
            text = "THE DECLINE WAS NOT ACCEPTED: the session is still open. Read the result and go on.\n" + text
        if not self.is_role("ask", name, args):
            note = self.take_notes(name, args, observation)
            if note:
                extra += "\n" + note
        text += extra
        if self.heard_text:
            text += "\n" + self.heard_text
            self.heard_text = ""
        return text

    # -- the code's own ending ----------------------------------------------------------------
    def finish(self) -> None:
        """The episode ended by code when the conductor did not: a checkout whenever the cart book holds a line, else
        the decline, each tried twice at most."""

        if self.finished or self.steps_left() <= 0:
            return
        buy, refuse = self.role_tool("buy"), self.role_tool("refuse")
        if buy and not self.cart and self.confirmed and self.steps_left() >= 3:
            self.complete_cart()
            if self.finished:
                return
        if buy and self.role_tool("cart") and len({self.item_of(r) for r in self.cart if self.buyable(r)}) < len(self.items):
            # (nothing, or not every product, in the cart: the best listing the code saw for each product goes in,
            #  since an order of the best found beats ending with nothing)
            picks = self.best_picks()
            calls = self.cart_calls(picks) if picks else []
            if picks and self.steps_left() >= len(calls) + 1:
                for call in calls:
                    observations = self.world([call])
                    if not observations or self.finished:
                        break
                    self.after(call["name"], call["args"], observations[0])
                if self.cart != picks:
                    self.cart = dict(picks)
                    self.cart_at = self.step
                self.trace.append({"finish_picks": picks})
        if buy and self.cart:
            for attempt in range(3):
                if self.finished or self.steps_left() <= 0 or self.wall_left() < 4:
                    return
                if self.steps_left() >= 2:
                    self.process_gate("order")
                args = self.ending_args("buy")
                observations = self.world([{"name": buy, "args": args}])
                self.trace.append({"finish": "buy", "cart": dict(self.cart), "result": _short(observations[0] if observations else None, 300)})
                if self.finished or not observations:
                    return
                # (refused: the refusal's words are read — a revised allowance, a line gone — and the cart re-picked
                #  under what is now in force before the next try)
                self.after(buy, args, observations[0])
                repicked = False
                if self.role_tool("cart") and attempt < 2:
                    picks = self.best_picks()
                    calls = self.cart_calls(picks) if picks and picks != self.cart else []
                    if calls and self.steps_left() >= len(calls) + 1:
                        for call in calls:
                            observations = self.world([call])
                            if not observations or self.finished:
                                return
                            self.after(call["name"], call["args"], observations[0])
                        if self.cart != picks:
                            self.cart = dict(picks)
                            self.cart_at = self.step
                        self.trace.append({"finish_repick": picks})
                        repicked = True
                if not repicked and attempt >= 1:
                    return          # (the same cart refused twice: the conductor gets its second round, as before)
            return
        if refuse:
            for _ in range(2):
                if self.finished or self.steps_left() <= 0 or self.wall_left() < 4:
                    return
                if self.steps_left() >= 2:
                    self.process_gate("end")
                args = self.quiet_text(self.ending_args("refuse"))
                observations = self.world([{"name": refuse, "args": args}])
                self.trace.append({"finish": "refuse", "result": _short(observations[0] if observations else None, 300)})
                if self.finished or not observations:
                    return


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    kernel = Kernel(problem_data)
    try:
        return kernel.run()
    except Exception as error:  # noqa: BLE001
        kernel.trace.append({"crash": repr(error)[:400]})
        try:
            kernel.finish()
        except Exception as again:  # noqa: BLE001
            kernel.trace.append({"crash": repr(again)[:400]})
        return kernel.trace


__all__ = ["agent_main"]
