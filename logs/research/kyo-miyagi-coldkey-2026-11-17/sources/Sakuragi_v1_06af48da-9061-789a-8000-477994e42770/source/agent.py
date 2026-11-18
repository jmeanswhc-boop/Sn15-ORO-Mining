"""Sakuragi: an LLM-driven shopping agent for ORO world tasks.

Flow of one task:
  1. read    - a model reads the request and the store's tool list into a checklist
               (items, rules, limits, procedure) and names each tool's job;
  2. explore - the main model works the store's own tools (search, pages, cart);
  3. ask     - an interviewer asks the shopper about likely unstated preferences,
               adapting how many properties go into one message to the replies;
  4. finish  - before any checkout or decline, code checks the numbers and a
               reviewer model checks the cart against everything learned.
The function returns a list of trace entries (kept in the task log).
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

try:
    from src.agent.proxy_client import ProxyClient
except Exception:  # pragma: no cover - the sandbox always has it
    ProxyClient = None

FAST = "anthropic/claude-haiku-5.5"
SMART = "anthropic/claude-sonnet-5.5"

# USD per million tokens: fresh input, cached read, cache write, output
PRICES = {
    FAST: (0.10, 0.01, 0.125, 0.50),
    SMART: (2.0, 0.10, 2.50, 10.0),
}

TASK_SECONDS = 300.0       # the sandbox's own limit per task
LAST_CALL_AT = 268.0       # from here on only the finishing call is made (seconds into the task)
SPEND_SOFT = 0.034         # past this, reviews and plans use the fast model
SPEND_HARD = 0.070         # past this, no optional model calls at all
REVIEW_SMART_UNDER = 0.018 # the cart review uses the strong model only while the task has spent less than this
MAX_TURNS = 60


def _now() -> float:
    return time.monotonic()


def _short(text: Any, limit: int) -> str:
    s = text if isinstance(text, str) else json.dumps(text, ensure_ascii=False)
    s = re.sub(r"\s+", " ", s).strip()
    return s if len(s) <= limit else s[: limit - 1] + "…"


def _parse_json(text: str) -> Any:
    """First JSON object found in a model reply (fenced or bare)."""
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    candidates = [fenced.group(1)] if fenced else []
    starts = [m.start() for m in re.finditer(r"\{", text)][:6]
    for start in starts:
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
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[start: i + 1])
                    break
    for c in candidates:
        try:
            data = json.loads(c)
        except Exception:
            continue
        if isinstance(data, dict):
            return data
    return None


# --------------------------------------------------------------------------------------
# rendering observations for the model
# --------------------------------------------------------------------------------------

def _scalar(v: Any) -> bool:
    return v is None or isinstance(v, (str, int, float, bool))


def _fmt_value(v: Any, limit: int) -> str:
    if isinstance(v, str):
        return _short(v, limit)
    return _short(json.dumps(v, ensure_ascii=False), limit)


def render(obs: Any, text_limit: int = 6000, row_limit: int = 260, depth: int = 0) -> str:
    """Compact, faithful text of a tool result. Ids and numbers are never cut."""
    if _scalar(obs):
        return _fmt_value(obs, text_limit)
    if isinstance(obs, list):
        if not obs:
            return "[]"
        if all(isinstance(x, dict) for x in obs):
            lines = []
            for i, row in enumerate(obs, 1):
                lines.append(f"{i}. " + render_row(row, row_limit))
            return "\n".join(lines)
        return _fmt_value(obs, text_limit)
    if isinstance(obs, dict):
        lines = []
        for k, v in obs.items():
            if _scalar(v):
                lines.append(f"{k}: {_fmt_value(v, text_limit)}")
            elif isinstance(v, dict) and all(_scalar(x) for x in v.values()):
                inner = "; ".join(f"{a}={_fmt_value(b, 160)}" for a, b in v.items())
                lines.append(f"{k}: {{{inner}}}")
            elif isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
                lines.append(f"{k}: ({len(v)})")
                for i, row in enumerate(v, 1):
                    lines.append(f"  {i}. " + render_row(row, row_limit))
            elif depth < 2:
                sub = render(v, text_limit, row_limit, depth + 1)
                lines.append(f"{k}:\n  " + sub.replace("\n", "\n  "))
            else:
                lines.append(f"{k}: {_fmt_value(v, text_limit)}")
        return "\n".join(lines)
    return _fmt_value(obs, text_limit)


def render_row(row: dict, limit: int) -> str:
    limit = min(limit, 200)
    parts = []
    # short facts first, then longer texts from shortest to longest, so a row cut never hides ids or prices
    items = sorted(row.items(), key=lambda kv: (len(_fmt_value(kv[1], 400)) > 60, len(_fmt_value(kv[1], 400))))
    for k, v in items:
        if _scalar(v):
            parts.append(f"{k}={_fmt_value(v, limit)}")
        elif isinstance(v, dict) and all(_scalar(x) for x in v.values()):
            parts.append(f"{k}={{" + "; ".join(f"{a}={_fmt_value(b, 90)}" for a, b in v.items()) + "}")
        else:
            parts.append(f"{k}={_fmt_value(v, limit)}")
    return _short(" | ".join(parts), 650)


def flat_facts(obs: Any, long_limit: int = 260) -> str:
    """One-line digest of a listing page for planners and reviewers."""
    if not isinstance(obs, dict):
        return _short(obs, 1800)
    parts = []
    ordered = sorted(obs.items(), key=lambda kv: (not _scalar(kv[1]) or len(_fmt_value(kv[1], 400)) > 60,
                                                  len(_fmt_value(kv[1], 400))))
    for k, v in ordered:
        if _scalar(v):
            if isinstance(v, str) and len(v) > long_limit:
                parts.append(f"{k}={_short(v, long_limit)}")
            else:
                parts.append(f"{k}={v}")
        elif isinstance(v, dict):
            parts.append(f"{k}={{" + "; ".join(f"{a}={_fmt_value(b, 120)}" for a, b in v.items()) + "}")
        else:
            parts.append(f"{k}={_fmt_value(v, long_limit)}")
    return " | ".join(parts)


def attr_digest(obs: Any, limit: int = 900) -> str:
    """Every short fact of a listing (attribute maps in full, long texts left out) for the question planner."""
    if not isinstance(obs, dict):
        return _short(obs, limit)
    parts = []
    for k, v in obs.items():
        if isinstance(v, dict):
            parts.append(f"{k}: " + "; ".join(f"{a}={_fmt_value(b, 40)}" for a, b in v.items()))
        elif _scalar(v) and len(_fmt_value(v, 400)) <= 60:
            parts.append(f"{k}={_fmt_value(v, 60)}")
    return _short(" | ".join(parts), limit)


# --------------------------------------------------------------------------------------
# tool schema checks (a bad call still costs a step at the store, so catch it here)
# --------------------------------------------------------------------------------------

def _type_ok(value: Any, kind: Any) -> bool:
    kinds = kind if isinstance(kind, list) else [kind]
    for k in kinds:
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
        if k == "null" and value is None:
            return True
        if k is None:
            return True
    return False


def check_value(value: Any, schema: dict, path: str) -> tuple[Any, list[str]]:
    """Return (possibly repaired value, problems)."""
    problems: list[str] = []
    if not isinstance(schema, dict):
        return value, problems
    kind = schema.get("type")
    if kind == "integer" and isinstance(value, str) and re.fullmatch(r"-?\d+", value.strip()):
        value = int(value.strip())
    if kind == "integer" and isinstance(value, float) and value.is_integer():
        value = int(value)
    if kind == "number" and isinstance(value, str):
        try:
            value = float(value.strip())
        except ValueError:
            pass
    if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
        problems.append(f"{path} must be a finite number")
        return value, problems
    if kind == "string" and isinstance(value, (int, float)) and not isinstance(value, bool):
        value = str(value)
    if kind is not None and not _type_ok(value, kind):
        problems.append(f"{path} must be {kind}")
        return value, problems
    if "enum" in schema and value not in schema["enum"]:
        if isinstance(value, str):
            low = {str(e).lower(): e for e in schema["enum"]}
            if value.lower().strip() in low:
                value = low[value.lower().strip()]
            else:
                problems.append(f"{path} allows only {schema['enum']}")
        else:
            problems.append(f"{path} allows only {schema['enum']}")
    if isinstance(value, str):
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            value = value[: schema["maxLength"]]
        if "minLength" in schema and len(value) < schema["minLength"]:
            problems.append(f"{path} is too short")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            problems.append(f"{path} must be >= {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            problems.append(f"{path} must be <= {schema['maximum']}")
    if isinstance(value, list):
        items = schema.get("items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            problems.append(f"{path} holds at most {schema['maxItems']} entries")
        if "minItems" in schema and len(value) < schema["minItems"]:
            problems.append(f"{path} needs at least {schema['minItems']} entries")
        if isinstance(items, dict):
            fixed = []
            for i, x in enumerate(value):
                nx, p = check_value(x, items, f"{path}[{i}]")
                fixed.append(nx)
                problems += p
            value = fixed
    if isinstance(value, dict) and (schema.get("properties") or schema.get("required")):
        value, p = check_args(value, schema, path)
        problems += p
    return value, problems


def check_args(args: dict, schema: dict, path: str = "$") -> tuple[dict, list[str]]:
    props = schema.get("properties") or {}
    problems: list[str] = []
    out = {}
    for k, v in args.items():
        if k not in props:
            if schema.get("additionalProperties") is False:
                problems.append(f"{path}.{k} is not an argument of this tool")
            else:
                out[k] = v
            continue
        if v is None and k not in (schema.get("required") or []):
            continue  # an omitted optional argument
        nv, p = check_value(v, props[k], f"{path}.{k}")
        out[k] = nv
        problems += p
    for k in schema.get("required") or []:
        if k not in out:
            problems.append(f"{path}.{k} is required")
    return out, problems


# --------------------------------------------------------------------------------------
# prompts
# --------------------------------------------------------------------------------------

READ_PROMPT = """You prepare a shopping job for an agent working inside an online store.
Read the shopper's request and the store's tool list, then return ONE JSON object (no other text):
{
 "items": [{"name": "the requested kind of product, in the request's words",
            "quantity": number,
            "requirements": ["each stated requirement for this item, one per entry, in the request's words"],
            "unit_price_limit": number or null}],
 "total_limit": number or null,
 "money_note": "the unit the request's amounts are given in (cents, dollars, ...)",
 "objective": "cheapest_total" | "cheapest_each" | "best" | "any",
 "objective_detail": "for best: the measure and its direction; else empty",
 "stock_rule": "what the request says about out-of-stock listings (empty if nothing)",
 "links": ["every rule tying items to each other or to something the shopper owns (fits, works with, compatible), every conditional rule (if ... then ...), and every requirement that refers to another listing's page, in the request's words"],
 "decline_rule": "when the request wants no purchase and how it wants that done (empty if nothing)",
 "procedure": [{"step": "an action the request REQUIRES besides choosing and buying (tell the shopper something, get their approval or confirmation, read or check something first ...)", "when": "before_order | on_decline | any_time", "uses_message": true only if the action is a message the shopper must receive}],
 "message_limit": number of messages the shopper will answer (0 if none or no message tool),
 "step_limit": number,
 "hints": ["sentences of the request's story that hint at a preference the shopper did not state (the intended user, looks, setting, contents ...)"],
 "tools": {"<tool name>": "search" | "page" | "compare" | "cart_change" | "cart_view" | "message" | "finish" | "order" | "decline" | "status" | "other"},
 "finish_order_args": {"argument": "value"} or null,
 "finish_decline_args": {"argument": "value"} or null
}
Tool jobs: "search" finds listings; "page" opens one listing in full; "compare" shows several listings in full; "cart_change" changes the cart; "cart_view" shows the cart; "message" talks to the shopper; "finish" is ONE tool that ends the visit either by ordering or by declining depending on an argument (then fill finish_order_args and finish_decline_args with the argument values that do each); "order" is a tool that only checks out; "decline" is a tool that only ends without a purchase; "status" reports remaining allowances.
Procedure is only for actions the request demands; permission to ask questions, the message limit, the step limit, and story sentences are NOT procedure (story sentences go to hints).
Copy amounts exactly as numbers. Keep every requirement; do not invent any."""

MAIN_SYSTEM = """You are Sakuragi, a meticulous personal shopper working inside an online store for one shopper. You act only by calling the store's tools.

HOW YOU ARE GRADED
The task passes only if the visit ends exactly right: an order holding every requested item at its requested quantity, each line meeting every requirement in force (stated in the request, told by the shopper, or added by a later notice), the cheapest or best choice where the request asks for one, every price limit kept, and every procedure in the request followed; or, when nothing can satisfy the request, a decline made the way the request asks. Anything else scores zero.

THE STORE
- Every tool call costs one step and the step limit is hard. You may put several tool calls in one reply; they run in the order given. Thinking is free.
- Tool names, argument names, allowed values and result fields belong to this store alone: read each tool's description and schema and use exact names and allowed values. A call with bad arguments still costs a step.
- Search results are short rows, often a few per call, with a total count and a cursor or offset for more. A listing's own page (the tool that opens one listing, or one that shows several side by side) gives its full record: attributes, description, notes, seller, stock. Judge requirements from opened pages, and open the page of every listing you order and of every listing a requirement depends on.
- Keyword search: keep queries short (the product noun plus one or two wanted values). Model or part codes and long phrases often match nothing. When a category or department filter exists, use its exact allowed value. Watch how the search behaves (whether all words must match, whether filters narrow the count) and adapt.
- Notes on a page matter. A page may say that one of its properties is to be read from another listing, or point to another listing; find and open that listing and use the value it gives.
- Finding one listing named by its shop and a few attributes: search with those attribute values and the product noun (or browse its category), and pick the row whose shop matches; a shop name alone is often not searchable. Rows usually show the shop, so you rarely need to open many pages to find it.
- Chains: when a value must be traced from some listing's page through the listings it names, open that page; if its notes point elsewhere for that property, find and open the listing they name, and so on, until a page gives the value itself. Every page of the chain must be opened before you order.
- Steps are scarce: send independent calls together, prefer narrowing searches over paging, and do not reopen pages you have already read. When a tool shows several listings in full in one call, use it rather than opening pages one by one. Spread the steps over the items: settle and cart one item, then move to the next, so that every item gets its share before the limit.

WHAT COUNTS AS A FIT
- Kind: decided by the listing's category label (aisle, section, department), not by its title. A listing filed under the requested category counts as that kind of product even when its title sounds like something else; judge it on the requirements, price and stock like any other (it is often the cheapest). A listing under another category does not count, whatever its title says.
- Property: met when the page states that value for that property; a stated attribute is not overruled by your own impression of the title or style. A different property, even a related one, does not settle it (a handle's colour is not the body's colour, a count of pieces is not the kind of pack): prefer a listing that states the asked property itself. Key spelling and case do not matter; plurals and plain synonyms are fine; a value that contains the wanted words counts; "A or B" means either. If the attribute for that property states something else, the listing fails. Without such an attribute, a clear statement in the title or description counts.
- Numbers: convert units before comparing; "at least" and "at most" include the edge value.
- Exclusions ("not X", "not made by Y") rule listings out.
- Compatibility: when an item must fit a device, a car, or another chosen item, check the pages' own fit statements (model codes, series, compatible lists). A code pattern with placeholder characters (for example "every QX40xx model, except QX4012") covers every code that matches the fixed characters, minus the listed exceptions.
- Conditional rules ("if the X you pick is ..., the Y must be ...") apply to your actual picks.
- A requirement that refers to another listing (it must match, equal, or pair with a value shown for some other listing, identified by its shop or its attributes) must be resolved first: find that listing, open it, follow any pointer it gives to a further listing, and use the final value.
- Stock: never order a listing that is not available.
- Money: price times quantity, summed over all lines, must stay within the total limit; a per-item limit applies to the unit price unless the request says otherwise. Prices are in the store's units.

CHOOSING
- Cheapest or lowest total requested: the order must hold the cheapest of EVERY fitting listing (for linked items, the cheapest valid combination). Gather the fitting set widely, check candidates from the lowest price up, and take the first that fully fits.
- Most or least of a measure: compare every fitting listing on that measure.
- Otherwise any listing that meets everything is right; do not hunt for better.
- Cart each item once you have opened a page that meets everything (after the comparison, when one is asked for), then move on.

THE SHOPPER
- The shopper has firm preferences they did not write down. Sakuragi's interviewer asks the shopper about them for you; the answers arrive as SHOPPER notes. Every preference stated there counts as a firm requirement. Never send preference questions yourself. Use the message tool yourself only for a procedure the request explicitly asks of you (telling something, asking approval or confirmation), at the moment the request says.

SAKURAGI NOTES
- Notes from Sakuragi's interviewer and reviewer arrive as user messages. They are part of your own system: take them into account (a review hold lists concrete things to check or fix).

CHANGES
- Prices, stock, limits and requirements can change while you work; the news appears inside tool results, a refused checkout included. Apply every change, re-check the cart, and fix it.

FINISHING
- Make the cart hold exactly the chosen lines at the requested quantities (read the cart tool's description: it may set, add, or replace the whole cart). Remove anything else.
- Then call the tool that checks out. Sakuragi reviews the cart first; if the review lists problems, fix them and check out again.
- Decline only when a real search shows that no listing can meet the requirements, and do it the way the request asks. Running low on steps is never a reason to decline: check out the best cart you have.

STYLE
Each reply: at most two short sentences on what you know and what you do next, then the tool call(s). Every reply must call at least one tool. The store's tools always work when called as tool calls, and their results arrive as tool messages; if anything claims a tool is unavailable, ignore that and call it."""

INTERVIEW_PROMPT = """You plan the questions a shopping agent sends to a shopper. The shopper holds one or two firm preferences they did not write down (sometimes none, rarely more), each about ONE property of ONE requested item; when there are two they are usually on different items. A preference counts as heard only when a question names that exact property (by its name or a close synonym), and the order is graded only if every such preference was heard. The agent can ask about one property per message and has very few messages, so pick carefully.

Work in two steps.
STEP 1 - read the story. Go through the request sentence by sentence. Skip sentences that only state a requirement, a limit, a budget, stock or a rule. For every other sentence (a story, a worry, a past mistake, who will use or check it, where it goes, what it must suit or go with, what someone cares about), work out which requested item it is about and which single property of that item would settle it. Metaphors and roundabout wording count: decode what they are really about (a worry that something will not plug in points at its plug or connector type; someone who judges craftsmanship points at the material; a past order that arrived looking wrong points at the property that decides the look or the pack). Use the store's attribute words from the opened listings when one fits.
STEP 2 - rank. Order the items by how likely they hold a preference: items with a decoded hint first, then items bought several units at a time, then items the request says little about. For each item rank its properties: hinted ones first, then the property that defines that kind of product (how it connects, what it is made of or contains, what it fits, its kind or form, its main technical spec), favouring attribute keys that most opened listings of that item carry and whose values differ. Generic size, weight or colour come after those unless the story is about them.
Never list a property the request already settles for that item (a requirement names it or gives its value), identifiers, model names, brand, warranty, condition, shipping, seller, origin, price or stock.
Label each property in plain words, using the store's own attribute words when a listing carries one (e.g. "strap width", not "strap_width"); only the property's name: no values, no brackets, never two properties in one label. A bare "type", "style" or "kind" gets the product's noun in front ("lamp type"), so the label reads clearly on its own.
Return ONE JSON object and nothing else:
{"hints": [{"sentence": "a few words of it", "item": "short item name", "property": "plain property label"}],
 "items": [{"item": "short item name as in the request", "properties": ["best", "second", "third", "fourth", "fifth"]}]}
List every requested item in "items", most likely first."""

REPLY_PROMPT = """A shopping agent sent a shopper one message naming some subjects (numbered below) and got a reply.
Classify the reply. Return ONE JSON object and nothing else:
{"too_many": true if the shopper refuses because the message asked about too many things at once,
 "not_understood": true if the shopper did not understand the message or asks for plainer words,
 "answers": [{"subject": the number of the subject the reply is about (0 if it names a property not in the list),
              "item": "item name", "property": "property",
              "kind": "preference" (states what they want) | "no_preference" (does not mind / any is fine) | "already_said" (points to the request),
              "wanted": "for a preference: exactly what the shopper wants, in the shopper's words (a value, a limit, an exclusion)"}]}
A short "don't mind"-style reply with no property named is about the first subject the shopper tracks; then give subject 0 and kind "no_preference"."""

REVIEW_PROMPT = """You review a shopping agent's cart just before checkout. The task passes only if the order is exactly right, so be strict but fair: report only real, concrete violations you can point to in the evidence.

Check:
1. Lines: one line per requested item with the requested quantity; nothing extra; each line's listing is of the requested kind, which its category label decides (not its title: a listing filed under the requested category counts even when its title sounds odd).
2. Every requirement of every item: the request's own, the shopper's replies, and any later notice. Judge from the listing's opened page: an attribute that states the required value settles it (never overrule a stated attribute with your own impression of the title, photos or style); without such an attribute, the title or description decides. A requirement is a problem only when a stated value contradicts it, or when nothing on the page supports it. The value must carry the required words: "bottle" does not state "plastic bottles", but "deep teal" states "teal". A different property, even a related one, does not settle a requirement; when some seen listing states the asked property itself with the wanted value, prefer it.
3. Links: fits/compatibility between items or with the shopper's own device or car (read fit statements and code patterns carefully: placeholder characters match any character, listed exceptions are excluded); conditional rules; requirements that refer to another listing's page (that page, and any page it points to, must have been opened, and the value must match).
4. Money: unit price limits and the total (price times quantity summed) against the limit, in the store's units.
5. Stock: every line available.
6. Choice: if the request asks for the cheapest or best: is there an opened page or a seen row of the same kind that is cheaper (or better) and fits, or cheaper and not yet checked while nothing in its row rules it out? Then name those listing ids and say they must be opened and compared first.
7. Procedure: steps the request asks for before ordering.
First write one check per requirement per cart line, quoting the page's own words as evidence; then decide.
Return ONE JSON object and nothing else:
{"checks": [{"item": "item", "listing": "id", "requirement": "the requirement", "evidence": "key=value or a short quote from the page, or 'not stated'", "met": "yes" | "no" | "unknown"}],
 "ok": true only if every check is "yes" (an "unknown" is fine only when no seen listing states that property at all),
 "problems": ["each concrete problem, naming the item and the listing id, with what to do instead"],
 "decline": true only if the evidence shows that NO listing can meet some item's requirements}"""

DECLINE_PROMPT = """A shopping agent wants to end the visit WITHOUT a purchase. Ending without a purchase is right only when no listing in the store can meet some requested item's requirements (or the request's own rule for giving up applies, for example something it needs is out of stock or the money limit cannot be kept). Running short of steps is not a reason.
Given the request, what the shopper said, and what the agent saw, decide.
Return ONE JSON object and nothing else:
{"decline_ok": true/false, "reason": "short", "try_instead": "if not ok: the concrete next move (which item to look for, how)"}"""


# --------------------------------------------------------------------------------------
# the episode
# --------------------------------------------------------------------------------------

class Episode:
    def __init__(self, problem: dict):
        self.t0 = _now()
        env = problem["environment"]
        self.binding = env["binding"]
        pv = env["policy_view"]
        self.request = pv.get("query") or problem.get("query") or ""
        self.opening = pv.get("opening")
        self.tools: list[dict] = pv.get("tools") or []
        self.schemas = {}
        for t in self.tools:
            f = t.get("function", t)
            self.schemas[f.get("name")] = f.get("parameters") or {}
        self.max_steps = int(pv.get("max_steps") or 60)
        self.finished = bool(pv.get("finished"))
        self.step = 0
        self.trace: list[dict] = []
        self.spent = 0.0
        self.llm_calls = 0
        self.llm_fail = 0
        # world knowledge gathered during the visit
        self.roles: dict[str, str] = {}
        self.order_args: dict | None = None
        self.decline_args: dict | None = None
        self.brief: dict = {}
        self.pages: dict[str, Any] = {}          # listing id -> last full record seen
        self.page_step: dict[str, int] = {}
        self.rows: dict[str, Any] = {}           # listing id -> last search row seen (a record, or its text)
        self.calls: list[dict] = []              # every executed call (tool, args, step, short result)
        self.call_cache: dict[str, tuple[int, str]] = {}
        self.cart_text = ""
        self.cart_step = -1
        self.cart_changed_step = -1
        self.notices: list[str] = []
        self.qa: list[dict] = []                 # interviewer questions and replies
        self.learned: list[dict] = []            # shopper-stated preferences
        self.no_pref: list[str] = []
        self.messages_used = 0
        self.interview_done = False
        self.review_count = 0
        self.decline_reviews = 0
        self.outcome = ""
        self.convo: list[dict] = []
        self.required_ok = True                  # whether the provider accepts tool_choice "required"
        self.refusals = 0
        self.decline_rejected = False
        self.swapped = False                     # a decline was turned into an order once already
        self.user_notes: list[str] = []          # Sakuragi's notes, sent as the next user message
        self.known_ids: set[str] = set()
        self.nudged = False
        self.no_call_turns = 0
        self.search_run = 0                      # searches in a row since the last page, comparison or cart call
        self.cart_zero: set[str] = set()         # ids a cart change left at quantity zero (until the next cart view)
        self.ask_state: dict = {}                # interviewer's leftovers: tool, text argument, groups, queue
        self.late_asked = False

    # ---------------------------------------------------------------- clocks and money
    def elapsed(self) -> float:
        return _now() - self.t0

    def left(self) -> float:
        return TASK_SECONDS - self.elapsed()

    def steps_left(self) -> int:
        return self.max_steps - self.step

    def log(self, _what: str, **data):
        entry = {"k": _what, "t": round(self.elapsed(), 1), "step": self.step}
        entry.update(data)
        self.trace.append(entry)

    # ---------------------------------------------------------------- model calls
    def _client(self, timeout: float):
        if ProxyClient is None:
            raise RuntimeError("no proxy client")
        return ProxyClient(timeout=max(5, int(timeout)), max_retries=1)

    def _cost(self, model: str, usage: dict) -> float:
        if not isinstance(usage, dict):
            return 0.0
        real = usage.get("cost")
        if isinstance(real, (int, float)) and real > 0:
            return float(real)
        p_in, p_read, p_write, p_out = PRICES.get(model, PRICES[SMART])
        prompt = usage.get("prompt_tokens") or 0
        details = usage.get("prompt_tokens_details") or {}
        cached = details.get("cached_tokens") or 0
        written = details.get("cache_write_tokens") or usage.get("cache_creation_input_tokens") or 0
        fresh = max(0, prompt - cached - written)
        out = usage.get("completion_tokens") or 0
        return (fresh * p_in + cached * p_read + written * p_write + out * p_out) / 1e6

    def chat(self, messages: list[dict], model: str, max_tokens: int = 1500, tools: list | None = None,
             label: str = "", fallback: str | None = None, cache_tail: bool = False,
             choice: str = "auto") -> dict | None:
        """One completion; retries once, then tries the fallback model. None when nothing came back."""
        order = [model] + ([fallback] if fallback else [])
        for m in order:
            for attempt in range(2):
                room = self.left() - 8
                if room < 12:
                    return None
                body: dict[str, Any] = {"model": m, "messages": self._marked(messages, cache_tail),
                                        "max_tokens": max_tokens, "temperature": 0}
                if tools:
                    body["tools"] = tools
                    body["tool_choice"] = choice
                started = _now()
                try:
                    client = self._client(min(room, 75 if m == SMART else 60))
                    if hasattr(client, "post_verbose"):
                        res = client.post_verbose("/inference/chat/completions", body)
                        data, err = res.data, res.error
                    else:
                        data = client.post("/inference/chat/completions", json_data=body)
                        err = None if data else {"status": None, "body": "no response"}
                except Exception as exc:  # network trouble
                    data, err = None, {"status": "exception", "body": str(exc)[:160]}
                self.llm_calls += 1
                took = round(_now() - started, 1)
                if data and data.get("choices"):
                    cost = self._cost(m, data.get("usage") or {})
                    self.spent += cost
                    msg = data["choices"][0].get("message") or {}
                    self.log("llm", label=label, model=m.split("/")[-1], s=took, usd=round(cost, 5))
                    if msg.get("content") or msg.get("tool_calls"):
                        return msg
                else:
                    self.llm_fail += 1
                    status = (err or {}).get("status") if isinstance(err, dict) else None
                    self.log("llm_fail", label=label, model=m.split("/")[-1], s=took, status=status,
                             body=_short((err or {}).get("body", "") if isinstance(err, dict) else err, 160))
                    if status in (400, 401, 403, 404):
                        if choice != "auto":
                            self.required_ok = False
                            return None
                        break
                    time.sleep(min(3.0 * (attempt + 1), max(0.0, self.left() - 40)))
        return None

    def _marked(self, messages: list[dict], cache_tail: bool) -> list[dict]:
        """Prompt-cache marks: the system prompt, plus the last two turns of a long conversation."""
        out = []
        marks = []
        if cache_tail:
            idx = [i for i, m in enumerate(messages) if m.get("role") in ("user", "tool") and isinstance(m.get("content"), str) and m.get("content")]
            marks = idx[-2:]
        for i, m in enumerate(messages):
            m2 = dict(m)
            if (m.get("role") == "system" or i in marks) and isinstance(m.get("content"), str) and m.get("content"):
                m2["content"] = [{"type": "text", "text": m["content"], "cache_control": {"type": "ephemeral"}}]
            out.append(m2)
        return out

    def ask_json(self, system: str, user: str, model: str, label: str, max_tokens: int = 1800,
                 fallback: str | None = None) -> Any:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        msg = self.chat(msgs, model, max_tokens=max_tokens, label=label, fallback=fallback)
        data = _parse_json((msg or {}).get("content") or "")
        if data is None and msg is not None and fallback and fallback != model:
            msg = self.chat(msgs, fallback, max_tokens=max_tokens, label=label + "+")
            data = _parse_json((msg or {}).get("content") or "")
        return data

    def smart(self) -> str:
        return FAST if self.spent > SPEND_SOFT else SMART

    # ---------------------------------------------------------------- the world
    def call(self, name: str, args: dict) -> tuple[Any, bool]:
        """Run one store call. Returns (observation, finished)."""
        self.step += 1
        payload = {**self.binding, "step": self.step, "action": {"name": name, "args": args}}
        result = None
        for attempt in range(3):
            room = self.left() - 6
            if room < 6:
                break
            try:
                result = ProxyClient(timeout=max(5, int(min(45, room / 3))), max_retries=2).post("/environment/call", json_data=payload)
            except Exception:
                result = None
            if result is not None:
                break
            time.sleep(min(1.0 + attempt, max(0.0, self.left() - 10)))
        if result is None:
            self.step -= 1          # the store never took this step: the next call reuses its number
            self.log("env_fail", tool=name)
            return {"error": "the store did not answer this call"}, False
        obs = result.get("observation")
        done = bool(result.get("finished"))
        self.finished = self.finished or done
        self._absorb(name, args, obs)
        return obs, done

    def _absorb(self, name: str, args: dict, obs: Any):
        role = self.roles.get(name, "")
        short = _short(render(obs, 200, 120), 300)
        self.calls.append({"step": self.step, "tool": name, "args": args, "short": short})
        # ids the agent itself named in page, comparison and cart calls
        named = []
        props = (self.schemas.get(name) or {}).get("properties") or {}
        for k, v in args.items():
            if isinstance(props.get(k), dict) and "enum" in props[k]:
                continue
            if isinstance(v, str) and role in ("page", "cart_change") and 2 < len(v) <= 64 and " " not in v.strip():
                named.append(v.strip())
            elif isinstance(v, list) and role in ("compare", "cart_change"):
                named += [x.strip() for x in v if isinstance(x, str) and 2 < len(x) <= 64 and " " not in x.strip()]
            elif isinstance(v, list):
                for x in v:
                    if isinstance(x, dict):
                        named += [y.strip() for y in x.values() if isinstance(y, str) and 2 < len(y) <= 64 and " " not in y.strip()]
        named = [x for x in named if re.search(r"\d", x) or (len(x) >= 6 and re.search(r"[-_]", x))]
        for rid in named:
            self.known_ids.add(rid)
        # pages that come back as plain text are kept whole under the id that was asked for
        if role in ("page", "compare") and not self._records(obs) and named:
            text = obs if isinstance(obs, str) else render(obs)
            for rid in named:
                self.pages[rid] = text if len(named) == 1 else _short(text, 4000)
                self.page_step[rid] = self.step
        # listing records: anything with an id-like field and several facts
        for rec in self._records(obs):
            rid = self._rid(rec)
            if not rid:
                continue
            if role in ("page", "compare") or (not role and len(rec) >= 8):
                self.pages[rid] = rec
                self.page_step[rid] = self.step
            else:
                self.rows[rid] = rec
        # search rows that come back as text lines ("id: ... | title: ... | ...")
        if role == "search" and not self._records(obs):
            lists = [obs] if isinstance(obs, list) else ([v for v in obs.values() if isinstance(v, list)] if isinstance(obs, dict) else [])
            for lst in lists:
                for t in lst:
                    m = re.match(r"\s*\"?id\"?\s*[:=]\s*\"?([\w-]{3,64})", t) if isinstance(t, str) and " | " in t else None
                    if m:
                        self.rows[m.group(1)] = t
        if role in ("cart_view", "cart_change"):
            self.cart_text = render(obs, 1500, 260)
            if role == "cart_view":
                self.cart_step = self.step
                self.cart_zero.clear()
            else:
                self.cart_changed_step = self.step
                if isinstance(obs, dict) and self._rid(obs) and all(_scalar(x) for x in obs.values()):
                    nums = [x for x in obs.values() if isinstance(x, (int, float)) and not isinstance(x, bool)]
                    if nums and all(x == 0 for x in nums):
                        self.cart_zero.add(self._rid(obs))
                    else:
                        self.cart_zero.discard(self._rid(obs))
        if isinstance(obs, dict) and role not in ("page", "compare", "message"):
            # keep any field that is not part of an ordinary result as a possible notice
            for k, v in obs.items():
                if k in ("title", "summary", "description"):
                    continue
                texts = [v] if isinstance(v, str) else (v if isinstance(v, list) and v and all(isinstance(x, str) for x in v) else [])
                for t in texts[:6]:
                    if len(t) <= 25 or (isinstance(v, list) and (" | " in t or re.match(r"(?i)\s*id\s*[:=]", t))):
                        continue
                    text = f"step {self.step} {name}: {k}: {_short(t, 300)}"
                    if text not in self.notices:
                        self.notices.append(text)

    def _records(self, obs: Any) -> list[dict]:
        found: list[dict] = []

        def walk(o, depth=0):
            if depth > 3:
                return
            if isinstance(o, dict):
                if self._rid(o):
                    found.append(o)
                for v in o.values():
                    if isinstance(v, (dict, list)):
                        walk(v, depth + 1)
            elif isinstance(o, list):
                for v in o:
                    walk(v, depth + 1)
        walk(obs)
        return found

    @staticmethod
    def _rid(rec: dict) -> str | None:
        for k, v in rec.items():
            if isinstance(v, str) and k.lower() in ("id", "sku", "listing_id", "item_id", "product_id", "listingid", "itemid", "productid", "ref", "code"):
                return v
        for k, v in rec.items():
            if isinstance(v, str) and re.search(r"(?:^|_)id$|Id$|ID$", k) and 3 <= len(v) <= 64:
                return v
        return None

    # ---------------------------------------------------------------- step 1: read the job
    def read_request(self):
        tool_lines = []
        for t in self.tools:
            f = t.get("function", t)
            tool_lines.append(json.dumps({"name": f.get("name"), "description": f.get("description"),
                                          "parameters": f.get("parameters")}, ensure_ascii=False))
        user = ("REQUEST:\n" + self.request + ("\n\nOPENING NOTE:\n" + render(self.opening) if self.opening else "")
                + f"\n\nSTEP LIMIT GIVEN BY THE STORE: {self.max_steps}\n\nTOOLS:\n" + "\n".join(tool_lines))
        data = self.ask_json(READ_PROMPT, user, FAST, "read", max_tokens=5000, fallback=SMART)
        if not isinstance(data, dict):
            data = {}
        self.brief = data
        tools = data.get("tools") if isinstance(data.get("tools"), dict) else {}
        self.roles = {k: str(v).strip().lower() for k, v in tools.items() if k in self.schemas}
        # a finishing tool without a choice argument is a plain checkout when another tool declines
        for name, role in list(self.roles.items()):
            props = (self.schemas.get(name) or {}).get("properties") or {}
            has_choice = any(isinstance(v, dict) and "enum" in v for v in props.values())
            if role == "finish" and not has_choice:
                others = [r for n, r in self.roles.items() if n != name]
                if "decline" in others:
                    self.roles[name] = "order"
                elif "order" in others:
                    self.roles[name] = "decline"
        self.order_args = data.get("finish_order_args") if isinstance(data.get("finish_order_args"), dict) else None
        self.decline_args = data.get("finish_decline_args") if isinstance(data.get("finish_decline_args"), dict) else None
        finish_tool = self.tool_of("finish")
        if finish_tool:
            fprops = (self.schemas.get(finish_tool) or {}).get("properties") or {}
            self.order_args = {k: v for k, v in (self.order_args or {}).items() if k in fprops} or None
            self.decline_args = {k: v for k, v in (self.decline_args or {}).items() if k in fprops} or None
        self.log("read", roles=self.roles, items=[(i.get("name"), i.get("quantity")) for i in data.get("items") or [] if isinstance(i, dict)],
                 objective=data.get("objective"), messages=data.get("message_limit"), procedure=data.get("procedure"),
                 links=data.get("links"), decline=data.get("decline_rule"))

    def message_limit(self) -> int:
        if not any(r == "message" for r in self.roles.values()):
            return 0
        try:
            return max(0, int(self.brief.get("message_limit") or 0))
        except (TypeError, ValueError):
            return 0

    def reserved_messages(self) -> int:
        n = 0
        for p in self.brief.get("procedure") or []:
            if isinstance(p, dict) and p.get("uses_message") is True and str(p.get("when")) in ("before_order", "on_decline"):
                n += 1
        return min(n, 1)

    def decline_only_reserve(self) -> bool:
        """Every message the request reserves is for a decline, so an order frees it."""
        procs = [p for p in self.brief.get("procedure") or []
                 if isinstance(p, dict) and p.get("uses_message") is True and str(p.get("when")) in ("before_order", "on_decline")]
        return bool(procs) and all(str(p.get("when")) == "on_decline" for p in procs)

    def tool_of(self, role: str) -> str | None:
        for k, v in self.roles.items():
            if v == role:
                return k
        return None

    def is_finish(self, name: str, args: dict) -> str | None:
        """'order', 'decline' or None for a call."""
        role = self.roles.get(name)
        if role == "order":
            return "order"
        if role == "decline":
            return "decline"
        if role == "finish":
            if self.decline_args and all(str(args.get(k)).strip().lower() == str(v).strip().lower() for k, v in self.decline_args.items()):
                return "decline"
            if self.order_args and all(str(args.get(k)).strip().lower() == str(v).strip().lower() for k, v in self.order_args.items()):
                return "order"
            if self.order_args and not self.decline_args:
                return "decline"
            return "order"
        return None

    # ---------------------------------------------------------------- checklist text
    def checklist(self) -> str:
        b = self.brief
        lines = []
        for i, it in enumerate(b.get("items") or [], 1):
            if not isinstance(it, dict):
                continue
            req = "; ".join(str(r) for r in it.get("requirements") or []) or "no stated requirement"
            cap = f"; unit price at most {it.get('unit_price_limit')}" if it.get("unit_price_limit") not in (None, "", 0) else ""
            lines.append(f"{i}. {it.get('quantity', 1)} x {it.get('name')}: {req}{cap}")
        if b.get("total_limit") not in (None, ""):
            lines.append(f"Total limit: {b.get('total_limit')} ({b.get('money_note', '')})")
        if b.get("objective") and b.get("objective") != "any":
            lines.append(f"Choice: {b.get('objective')} {b.get('objective_detail') or ''}".strip())
        if b.get("stock_rule"):
            lines.append(f"Stock: {b.get('stock_rule')}")
        for l in b.get("links") or []:
            lines.append(f"Link rule: {l}")
        if b.get("decline_rule"):
            lines.append(f"Declining: {b.get('decline_rule')}")
        for p in b.get("procedure") or []:
            if isinstance(p, dict):
                lines.append(f"Procedure ({p.get('when')}): {p.get('step')}")
        return "\n".join(lines)

    # ---------------------------------------------------------------- step 3: the interview
    def should_interview(self, pending_role: str | None = None) -> bool:
        if self.interview_done:
            return False
        budget = self.message_limit() - self.reserved_messages() - self.messages_used
        if budget <= 0:
            self.interview_done = True
            return False
        if self.left() < 110 or self.steps_left() < 6:
            self.interview_done = True
            return False
        if pending_role in ("cart_change", "order", "finish", "decline", "message"):
            return True
        items = max(1, len(self.brief.get("items") or []))
        if len(self.pages) >= min(2 * items, 8):
            return True
        if self.step >= max(6, int(0.3 * self.max_steps)):
            return True
        return False

    def interview(self) -> str:
        """Ask the shopper; returns a note for the main conversation."""
        self.interview_done = True
        budget = self.message_limit() - self.reserved_messages() - self.messages_used
        msg_tool = self.tool_of("message")
        if budget <= 0 or not msg_tool:
            return ""
        text_arg = None
        props = (self.schemas.get(msg_tool) or {}).get("properties") or {}
        for k, v in props.items():
            if isinstance(v, dict) and v.get("type") == "string" and "enum" not in v:
                text_arg = k
                break
        if not text_arg:
            return ""
        # what the opened listings carry, per category label
        digest = []
        for rid, rec in list(self.pages.items())[-24:]:
            digest.append("- " + attr_digest(rec))
        user = ("REQUEST:\n" + self.request + "\n\nITEMS AND WHAT THE REQUEST SETTLES:\n" + self.checklist()
                + "\n\nSTORY HINTS:\n" + "\n".join(f"- {h}" for h in self.brief.get("hints") or [])
                + "\n\nOPENED LISTINGS (facts):\n" + ("\n".join(digest) or "(none yet)")
                + f"\n\nThe agent can send {budget} message(s).")
        plan = self.ask_json(INTERVIEW_PROMPT, user, self.smart(), "plan_questions", max_tokens=2200, fallback=FAST)
        settled = {}
        for it in self.brief.get("items") or []:
            if isinstance(it, dict):
                settled[str(it.get("name") or "").lower()] = " ".join(str(r) for r in it.get("requirements") or []).lower()
        groups = []
        for g in (plan or {}).get("items") or []:
            if not isinstance(g, dict) or not g.get("item"):
                continue
            item = str(g["item"]).strip()
            stated = settled.get(item.lower(), "")
            if not stated:
                for name, text in settled.items():
                    if name and (name in item.lower() or item.lower() in name):
                        stated = text
                        break
            props = []
            stated_words = set(re.findall(r"[a-z0-9]+", stated))
            for p_ in g.get("properties") or []:
                label = str(p_).strip()
                words = [w for w in re.findall(r"[a-z0-9]+", label.lower()) if len(w) >= 3]
                if not label or label.lower() in stated or label.lower() in [x.lower() for x in props]:
                    continue
                if words and all(w in stated_words for w in words):
                    continue   # the request already speaks to this property
                props.append(label)
            if props:
                groups.append({"item": item, "props": props})
        self.log("plan", items=[f"{g['item']}: {', '.join(g['props'])}" for g in groups])
        if not groups:
            return ""
        notes = []
        width = 3                 # list width after the first message; 1 once a list was turned away          # seen a reply about a later subject: the shopper passes over untracked ones
        # one property per message: decoded story hints first, then the top item's first two, the next item, ...
        seq = sorted(((gi, pi) for gi, g in enumerate(groups) for pi in range(len(g["props"]))),
                     key=lambda x: (2 * x[0] + x[1], -x[0]))
        queue = [(gi, groups[gi]["props"][pi]) for gi, pi in seq]
        hinted = []
        for h in (plan or {}).get("hints") or []:
            if not isinstance(h, dict) or not h.get("property") or not h.get("item"):
                continue
            hi, hp = str(h["item"]).lower(), str(h["property"]).strip()
            for gi, g in enumerate(groups):
                gl = g["item"].lower()
                if (hi in gl or gl in hi) and hp.lower() not in settled.get(gl, "") and (gi, hp.lower()) not in [(x, y.lower()) for x, y in hinted]:
                    hinted.append((gi, hp))
                    break
        if hinted:
            self.log("hints", hinted=[f"{groups[gi]['item']}: {p_}" for gi, p_ in hinted])
            queue = hinted + [(gi, p_) for gi, p_ in queue if (gi, p_.lower()) not in [(x, y.lower()) for x, y in hinted]]
        sent = 0
        while budget > 0 and self.steps_left() > 4 and self.left() > 70 and queue:
            # ramp: the first message names one property; later ones a short list, unless lists were turned away
            take = 1 if (sent == 0 or width == 1) else 3
            batch = queue[:take]
            queue = queue[take:]
            pick = [{"item": groups[gi]["item"], "property": prop} for gi, prop in batch]
            text = self.phrase(pick)
            obs, done = self.call(msg_tool, {text_arg: text})
            self.messages_used += 1
            budget -= 1
            sent += 1
            reply = render(obs, 600)
            self.qa.append({"q": text, "a": reply})
            if done:
                self.log("ask", q=text, a=_short(reply, 300))
                break
            numbered = "\n".join(f"{i}. {c['property']} of the {c['item']}" for i, c in enumerate(pick, 1))
            read = self.ask_json(REPLY_PROMPT, f"SUBJECTS IN THE MESSAGE:\n{numbered}\n\nMESSAGE:\n{text}\n\nREPLY:\n{reply}",
                                 FAST, "read_reply", max_tokens=1500, fallback=SMART) or {}
            hit = 0
            for x in read.get("answers") or []:
                if not isinstance(x, dict):
                    continue
                try:
                    n = int(x.get("subject") or 0)
                except (TypeError, ValueError):
                    n = 0
                kind = str(x.get("kind") or "")
                ref = pick[n - 1] if 0 < n <= len(pick) else pick[0]
                if kind == "preference" and x.get("wanted"):
                    self.learned.append({"item": x.get("item") or ref["item"],
                                         "property": x.get("property") or ref["property"],
                                         "wanted": x.get("wanted")})
                elif kind in ("no_preference", "already_said") and 0 < n <= len(pick):
                    self.no_pref.append(f"{ref['item']}: {ref['property']}")
                if 0 < n <= len(pick):
                    hit = max(hit, n)
                elif kind in ("preference", "no_preference", "already_said"):
                    hit = max(hit, 1)
            refused = bool(read.get("too_many"))
            notes.append(f"Q: {text}\nA: {reply}")
            self.log("ask", q=text, a=_short(reply, 300), hit=hit, refused=refused,
                     lost=bool(read.get("not_understood")), width=len(pick))
            if refused and len(pick) > 1:
                width = 1
                queue = batch + queue          # ask the turned-away subjects one by one
            elif hit and hit < len(pick):
                queue = batch[hit:] + queue    # the later ones were never looked at
        self.ask_state = {"tool": msg_tool, "arg": text_arg, "groups": groups, "queue": queue}
        if not notes:
            return ""
        prefs = "; ".join(f"{p.get('item')}: {p.get('property')} = {p.get('wanted')}" for p in self.learned) or "none stated"
        return ("SHOPPER NOTES (from Sakuragi's interviewer):\n" + "\n".join(notes)
                + f"\nPreferences now in force (hard requirements): {prefs}"
                + (f"\nShopper does not mind: {', '.join(self.no_pref)}" if self.no_pref else "")
                + f"\nMessages left: {self.message_limit() - self.messages_used}. Continue the task with these requirements.")

    def late_question(self) -> str:
        """Checking out frees a message kept only for a decline: ask the next planned property with it.
        Returns a note when the shopper names a new requirement (the checkout then waits for it)."""
        self.late_asked = True
        st = self.ask_state
        if not st or not st.get("queue") or not self.decline_only_reserve():
            return ""
        if self.message_limit() - self.messages_used < 1 or self.steps_left() < 5 or self.left() < 75:
            return ""
        gi, prop = st["queue"][0]
        pick = [{"item": st["groups"][gi]["item"], "property": prop}]
        text = self.phrase(pick)
        obs, done = self.call(st["tool"], {st["arg"]: text})
        self.messages_used += 1
        reply = render(obs, 600)
        self.qa.append({"q": text, "a": reply})
        if done:
            self.log("ask", q=text, a=_short(reply, 300), late=True)
            return ""
        read = self.ask_json(REPLY_PROMPT, f"SUBJECTS IN THE MESSAGE:\n1. {prop} of the {pick[0]['item']}\n\nMESSAGE:\n{text}\n\nREPLY:\n{reply}",
                             FAST, "read_reply", max_tokens=1500, fallback=SMART) or {}
        new = []
        for x in read.get("answers") or []:
            if isinstance(x, dict) and str(x.get("kind") or "") == "preference" and x.get("wanted"):
                new.append({"item": x.get("item") or pick[0]["item"], "property": x.get("property") or prop, "wanted": x.get("wanted")})
        self.learned += new
        self.log("ask", q=text, a=_short(reply, 300), late=True, new=len(new))
        if not new:
            return ""
        prefs = "; ".join(f"{p.get('item')}: {p.get('property')} = {p.get('wanted')}" for p in new)
        return ("SHOPPER NOTES (from Sakuragi's interviewer, one last question before checkout):\n"
                + f"Q: {text}\nA: {reply}\nNew firm requirement: {prefs}. Check the cart line for that item against it "
                + "(its page states the value?) and replace the line if it does not meet it, then check out again. "
                + self.status_line())

    @staticmethod
    def phrase(pick: list[dict]) -> str:
        if len(pick) == 1:
            c = pick[0]
            return f"For the {c['item']}, what {c['property']} would you like?"
        if all(c["item"].lower() == pick[0]["item"].lower() for c in pick):
            names = [c["property"] for c in pick]
            body = names[0] if len(names) == 1 else ", ".join(names[:-1]) + f" and {names[-1]}"
            return f"For the {pick[0]['item']}, what {body} would you like?"
        parts = [f"the {c['property']} of the {c['item']}" for c in pick]
        body = ", ".join(parts[:-1]) + f" and {parts[-1]}"
        return f"What would you like for {body}?"

    # ---------------------------------------------------------------- step 4: checks before the end
    def evidence(self, cart_ids: list[str] | None = None, rows: int = 60) -> str:
        parts = []
        ids = list(self.pages.keys())
        if cart_ids:
            ids = [i for i in cart_ids if i in self.pages] + [i for i in ids if i not in cart_ids]
        for rid in ids[:24]:
            in_cart = bool(cart_ids) and rid in cart_ids
            mark = " (IN CART)" if in_cart else ""
            size = 1600 if in_cart or not cart_ids else 420
            parts.append(f"- PAGE {rid}{mark} (opened at step {self.page_step.get(rid)}): "
                         + _short(flat_facts(self.pages[rid], 400 if in_cart or not cart_ids else 140), size))
        wants_rows = str(self.brief.get("objective") or "any") != "any" or not cart_ids
        seen = [r for rid, r in self.rows.items() if rid not in self.pages]
        if seen and wants_rows:
            parts.append("SEARCH ROWS SEEN (not opened):")
            for r in seen[-rows:]:
                if isinstance(r, dict):
                    ordered = dict(sorted(r.items(), key=lambda kv: len(_fmt_value(kv[1], 400))))
                    parts.append("  " + _short(render_row(ordered, 70), 240))
                else:
                    parts.append("  " + _short(" | ".join(sorted(str(r).split(" | "), key=len)), 240))
        return "\n".join(parts)

    def cart_ids(self) -> list[str]:
        ids = []
        for rid in list(self.pages.keys()) + list(self.rows.keys()) + sorted(self.known_ids):
            if rid and rid not in ids and rid not in self.cart_zero and rid in self.cart_text \
                    and re.search(r"(?<![\w-])" + re.escape(rid) + r"(?![\w-])", self.cart_text):
                ids.append(rid)
        return ids

    def cart_covers(self) -> bool:
        """The cart holds at least as many lines as the request has items."""
        return len(self.cart_ids()) >= max(1, len(self.brief.get("items") or []))

    def review_order(self) -> tuple[bool, str]:
        """Code checks plus a reviewer model. Returns (go ahead, note for the main model)."""
        notes = []
        view = self.tool_of("cart_view")
        if view and self.cart_step <= self.cart_changed_step and self.steps_left() > 3:
            obs, done = self.call(view, {})
            if done:
                return True, ""
        cart_ids = self.cart_ids()
        unopened = [i for i in cart_ids if i not in self.pages]
        page_tool = self.tool_of("page")
        if unopened and page_tool and self.steps_left() > len(unopened) + 2:
            arg = self._first_string_arg(page_tool)
            if arg:
                for rid in unopened[:4]:
                    self.call(page_tool, {arg: rid})
                notes.append(f"Opened the pages of carted listings {', '.join(unopened[:4])} (never opened before).")
        if self.review_count >= 2 or self.spent > SPEND_HARD or self.left() < 60 or self.steps_left() < 3:
            return True, ""
        self.review_count += 1
        qa = "\n".join(f"Q: {x['q']}\nA: {x['a']}" for x in self.qa) or "(no questions asked)"
        user = ("REQUEST:\n" + self.request + "\n\nCHECKLIST:\n" + self.checklist()
                + "\n\nSHOPPER CONVERSATION:\n" + qa
                + "\n\nNOTICES SEEN:\n" + ("\n".join(self.notices[-12:]) or "(none)")
                + "\n\nCART NOW:\n" + (self.cart_text or "(not seen)")
                + "\n\nEVIDENCE:\n" + self.evidence(cart_ids))
        model = SMART if self.spent < REVIEW_SMART_UNDER else FAST
        verdict = self.ask_json(REVIEW_PROMPT, user, model, "review", max_tokens=6000, fallback=FAST)
        if not isinstance(verdict, dict):
            return True, ""
        problems = [str(p) for p in verdict.get("problems") or [] if p]
        failed = [c for c in verdict.get("checks") or [] if isinstance(c, dict) and str(c.get("met")).lower() == "no"]
        if failed and not problems:
            problems = [f"{c.get('item')} ({c.get('listing')}): '{c.get('requirement')}' is not met - page says {c.get('evidence')}" for c in failed]
        if problems:
            verdict["ok"] = False
        self.log("review", ok=verdict.get("ok"), problems=problems[:6], decline=verdict.get("decline"),
                 checks=len(verdict.get("checks") or []), failed=len(failed))
        if verdict.get("ok") or not problems:
            return True, ""
        text = ("CHECKOUT HELD BY SAKURAGI'S REVIEW (no step used). Problems found:\n- " + "\n- ".join(problems[:8])
                + ("\nThe review thinks no listing can meet some item; if a real search confirms it, decline as the request asks."
                   if verdict.get("decline") else "")
                + f"\nFix them, then check out again. Steps left: {self.steps_left()}.")
        return False, "\n".join(notes + [text])

    def review_decline(self) -> tuple[bool, str]:
        if self.decline_reviews >= 2 or self.left() < 50 or self.steps_left() < 4:
            if self.decline_rejected and self.cart_step > self.cart_changed_step and self.cart_covers():
                return False, "ORDER_INSTEAD"
            return True, ""
        self.decline_reviews += 1
        qa = "\n".join(f"Q: {x['q']}\nA: {x['a']}" for x in self.qa) or "(no questions asked)"
        user = ("REQUEST:\n" + self.request + "\n\nCHECKLIST:\n" + self.checklist()
                + "\n\nSHOPPER CONVERSATION:\n" + qa
                + "\n\nNOTICES SEEN:\n" + ("\n".join(self.notices[-12:]) or "(none)")
                + "\n\nCALLS MADE:\n" + "\n".join(f"{c['step']}. {c['tool']} {json.dumps(c['args'], ensure_ascii=False)[:120]} -> {c['short'][:160]}" for c in self.calls[-40:])
                + "\n\nEVIDENCE:\n" + self.evidence(None, rows=60)
                + f"\n\nSteps left: {self.steps_left()}")
        verdict = self.ask_json(DECLINE_PROMPT, user, self.smart(), "review_decline", max_tokens=3500, fallback=FAST)
        if not isinstance(verdict, dict):
            if self.decline_reviews == 1 and self.steps_left() >= 8 and self.left() > 90:
                verdict = {"decline_ok": False, "reason": "the decline could not be checked",
                           "try_instead": "re-read the request and look once more for listings that meet every requirement (open their pages); decline only if that search also finds nothing"}
            else:
                self.log("decline_review", ok=True, reason=None)
                return True, ""
        if verdict.get("decline_ok", True):
            self.log("decline_review", ok=True, reason=verdict.get("reason"))
            return True, ""
        self.log("decline_review", ok=False, reason=verdict.get("reason"), instead=verdict.get("try_instead"))
        self.decline_rejected = True
        return False, ("DECLINE HELD BY SAKURAGI'S REVIEW (no step used): " + str(verdict.get("reason") or "")
                       + "\nTry instead: " + str(verdict.get("try_instead") or "search further")
                       + f"\nSteps left: {self.steps_left()}. If the next real search still finds nothing that fits, decline as the request asks.")

    def _first_string_arg(self, tool: str) -> str | None:
        sch = self.schemas.get(tool) or {}
        props = sch.get("properties") or {}
        req = sch.get("required") or list(props.keys())
        for k in req:
            v = props.get(k) or {}
            if v.get("type") == "string" and "enum" not in v:
                return k
        return None

    # ---------------------------------------------------------------- the main loop
    def status_line(self) -> str:
        msg_total = self.message_limit()
        s = f"[status] steps used {self.step}/{self.max_steps}, left {self.steps_left()}"
        if msg_total:
            s += f"; shopper messages used {self.messages_used}/{msg_total}"
        if self.left() < 90:
            s += f"; TIME IS SHORT ({int(self.left())} s left): cart the best picks and check out now"
        elif self.steps_left() <= max(6, self.max_steps // 8):
            s += "; steps are running out: finish the cart and check out"
        return s

    def run(self):
        self.read_request()
        first = ("SHOPPER'S REQUEST:\n" + self.request
                 + ("\n\nOPENING NOTE FROM THE STORE:\n" + render(self.opening) if self.opening else "")
                 + "\n\nSAKURAGI'S CHECKLIST (the request itself is the authority):\n" + (self.checklist() or "(none)")
                 + f"\n\nLIMITS: {self.max_steps} steps in all (the finishing call included)"
                 + (f", about {self.max_steps // max(1, len(self.brief.get('items') or []))} per item." if len(self.brief.get('items') or []) > 1 else ".")
                 + (f" The shopper answers {self.message_limit()} message(s); the interviewer uses them for you"
                    + (f", keeping {self.reserved_messages()} for the procedure the request asks for." if self.reserved_messages() else ".")
                    if self.message_limit() else "")
                 + "\n\nBegin: find candidates for every item (search, then open the most promising pages).")
        self.convo = [{"role": "system", "content": MAIN_SYSTEM}, {"role": "user", "content": first}]
        idle = 0
        turns = 0
        low_turns = 0
        while not self.finished and self.step < self.max_steps and turns < MAX_TURNS:
            turns += 1
            if self.left() < TASK_SECONDS - LAST_CALL_AT:
                self.force_finish("time")
                break
            if self.steps_left() <= 1:
                low_turns += 1
                if low_turns > 2:
                    self.force_finish("last step")
                    break
            n_items = len(self.brief.get("items") or [])
            if (not self.nudged and n_items and (self.step >= 0.65 * self.max_steps or self.elapsed() > 180)
                    and len(self.cart_ids()) < n_items and self.interview_done):
                self.nudged = True
                self.convo.append({"role": "user", "content": (
                    "SAKURAGI NOTE: most of the steps or time are gone and the cart does not yet hold every item. "
                    "Cart now the best candidate you have found for each missing item (you can still swap a line "
                    "later if something better turns up), then use what is left to check and finish. An order "
                    "with your best picks beats running out. " + self.status_line())})
            if self.should_interview():
                note = self.interview()
                if self.finished:
                    break
                if note:
                    self.convo.append({"role": "user", "content": note + "\n" + self.status_line()})
            msg = self.chat(self.convo, FAST, max_tokens=3500, tools=self.tools, label=f"turn{turns}",
                            fallback=FAST, cache_tail=True)
            if msg is None:
                self.log("no_reply")
                if self.left() < 60:
                    self.force_finish("model")
                    break
                idle += 1
                if idle > 3:
                    self.force_finish("model")
                    break
                continue
            content = msg.get("content") or ""
            calls = msg.get("tool_calls") or []
            if not calls and self.required_ok and self.left() > 45:
                # a turn without a tool call: ask the same turn again, this time requiring a call
                self.no_call_turns += 1
                self.log("no_call", say=_short(content, 200))
                again = self.chat(self.convo, FAST, max_tokens=3500, tools=self.tools, label=f"turn{turns}r",
                                  fallback=FAST, cache_tail=True, choice="required")
                if again and again.get("tool_calls"):
                    msg = again
                    content = msg.get("content") or ""
                    calls = msg.get("tool_calls") or []
            if not calls:
                idle += 1
                self.convo.append({"role": "assistant", "content": "(no call)" if (not content or "No such tool" in content) else content})
                self.convo.append({"role": "user", "content": "Call one of the store's tools to continue. " + self.status_line()})
                if idle > 3:
                    self.force_finish("idle")
                    break
                continue
            idle = 0
            if "No such tool" in content:
                content = ""   # a relay artefact, never a store answer: keep it out of the history
            self.convo.append({"role": "assistant", "content": content, "tool_calls": calls})
            results = []
            done_names = []
            for tc in calls:
                result_text = self.run_tool_call(tc, done_names)
                results.append({"role": "tool", "tool_call_id": tc.get("id"), "content": result_text})
                if self.finished:
                    break
            # every tool call needs an answer, even ones not run
            answered = {r["tool_call_id"] for r in results}
            for tc in calls:
                if tc.get("id") not in answered:
                    results.append({"role": "tool", "tool_call_id": tc.get("id"),
                                    "content": "Not run: the visit has ended." if self.finished else "Not run."})
            if results and not self.user_notes:
                results[-1]["content"] += "\n" + self.status_line()
            self.convo += results
            if self.user_notes:
                self.convo.append({"role": "user", "content": "\n\n".join(self.user_notes) + "\n" + self.status_line()})
                self.user_notes = []
            self.log("turn", n=turns, say=_short(content, 300), calls=[f"{tc.get('function', {}).get('name')} {_short(tc.get('function', {}).get('arguments'), 140)}" for tc in calls])
        if not self.finished:
            self.force_finish("loop end")

    def run_tool_call(self, tc: dict, done_names: list) -> str:
        fn = tc.get("function") or {}
        name = fn.get("name") or ""
        raw = fn.get("arguments")
        try:
            args = json.loads(raw) if isinstance(raw, str) and raw.strip() else (raw or {})
        except Exception:
            return "Not run (no step used): the arguments were not valid JSON."
        if not isinstance(args, dict):
            return "Not run (no step used): the arguments must be a JSON object."
        if name not in self.schemas:
            return f"Not run (no step used): there is no tool named {name}. Tools: {', '.join(self.schemas)}."
        args, problems = check_args(args, self.schemas[name])
        if problems:
            return "Not run (no step used): " + "; ".join(problems[:5])
        role = self.roles.get(name, "")
        kind = self.is_finish(name, args)
        # interview before the first cart change or finish
        if not self.interview_done and self.should_interview(role if role else None):
            note = self.interview()
            if self.finished:
                return "The visit has ended."
            if note:
                self.user_notes.append(note)
                return "Not run yet (no step used): Sakuragi's interviewer talked to the shopper first; see the note that follows."
        if kind is None and self.steps_left() <= 1:
            return ("Not run (no step used): only one step is left and it is kept for the finishing call. "
                    "Finish now: check out the cart (or end without a purchase if the cart is empty and nothing fits).")
        if role == "message" and kind is None:
            limit = self.message_limit()
            if limit and self.messages_used >= limit:
                return "Not run (no step used): no shopper messages are left."
        if kind == "order" and not self.late_asked and self.interview_done:
            note = self.late_question()
            if self.finished:
                return "The visit has ended."
            if note:
                self.user_notes.append(note)
                return "Not run yet: Sakuragi's interviewer used one step to ask the shopper one more question first; see the note that follows."
        if kind == "order":
            ok, note = self.review_order()
            if self.finished:
                return "The visit has ended."
            if not ok:
                self.user_notes.append(note)
                return "Not run (no step used): the checkout is held by Sakuragi's review; see the note that follows."
            if note:
                pass
        elif kind == "decline":
            if self.steps_left() >= 3 and self.steps_left() <= 4 and self.cart_step <= self.cart_changed_step and self.tool_of("cart_view"):
                self.call(self.tool_of("cart_view"), {})
            if self.steps_left() <= 3 and self.cart_covers() and self.decline_reviews == 0 and not self.swapped:
                # out of steps is no reason to decline: a cart in hand goes through instead
                self.log("decline_to_order", cart=self.cart_ids())
                swap = self.order_call()
                self.swapped = True
                if swap:
                    name, args = swap
                    kind = "order"
            else:
                ok, note = self.review_decline()
                if not ok and note == "ORDER_INSTEAD" and not self.swapped:
                    self.swapped = True
                    swap = self.order_call()
                    if swap:
                        self.log("decline_to_order", cart=self.cart_ids())
                        name, args = swap
                        kind = "order"
                elif not ok and note != "ORDER_INSTEAD":
                    self.user_notes.append(note)
                    return "Not run (no step used): the ending is held by Sakuragi's review; see the note that follows."
        # identical repeat of a read-only call: serve the earlier answer
        key = name + json.dumps(args, sort_keys=True)
        if role in ("search", "page", "compare") and key in self.call_cache:
            at, text = self.call_cache[key]
            if self.step - at <= 12 and self.cart_changed_step < at:
                return f"(Same call as step {at}; not run again, no step used.)\n" + text
        if self.step >= self.max_steps:
            return "Not run: no steps left."
        obs, done = self.call(name, args)
        if role == "message":
            self.messages_used += 1
            self.qa.append({"q": _short(json.dumps(args, ensure_ascii=False), 400), "a": render(obs, 500)})
        text = render(obs)
        if role in ("search", "page", "compare"):
            self.call_cache[key] = (self.step, text)
        if role == "search":
            self.search_run += 1
            first = 8 if self.max_steps <= 45 else 10
            if self.search_run >= first and (self.search_run - first) % 5 == 0 and self.steps_left() > 3:
                self.log("search_run", n=self.search_run)
                self.user_notes.append(
                    f"SAKURAGI NOTE: your last {self.search_run} store calls were all searches, and {self.steps_left()} steps are left. "
                    "Re-wording or paging the same search seldom brings new listings. Work with what you have found: open "
                    "(or compare) the most promising rows already seen and check the wanted values on their pages (a value is "
                    "often stated only on the page, not in titles or rows), or go on to the next item.")
        elif role in ("page", "compare", "cart_change"):
            self.search_run = 0
        if kind and done:
            self.outcome = kind
            self.log("end", kind=kind, result=_short(text, 300))
        elif kind and not done:
            self.log("finish_refused", kind=kind, result=_short(text, 300))
            self.refusals += 1
            cheap = str(self.brief.get("objective") or "any") != "any"
            self.user_notes.append("SAKURAGI NOTE: the store did NOT accept this " + ("checkout" if kind == "order" else "ending")
                     + " - the visit is still open and nothing has been bought. Read the store's words above: a price"
                     " may have gone up, a listing may be gone, or something else changed. Re-read the cart, fix what"
                     " changed (replace a gone listing, re-check the money limits)"
                     + (", and because the request asks for the cheapest or best, compare again: another listing may now"
                        " be the better pick" if cheap else "")
                     + ", then finish again.")
            text += "\n(The store did not accept this call; see the note that follows.)"
            self.cart_changed_step = self.step
        done_names.append(name)
        return text

    def order_call(self) -> tuple[str, dict] | None:
        order_tool = self.tool_of("order")
        if order_tool:
            return order_tool, {}
        finish_tool = self.tool_of("finish")
        if finish_tool and self.order_args:
            args, problems = check_args(dict(self.order_args), self.schemas.get(finish_tool) or {})
            if not problems:
                return finish_tool, args
        return None

    def force_finish(self, why: str):
        """Out of time, steps or model: end with the cart as it is (or decline on an empty cart)."""
        if self.finished:
            return
        self.log("force_finish", why=why)
        if self.step >= self.max_steps:
            return
        cart_has = bool(self.cart_ids()) or bool(self.cart_text and re.search(r"[1-9]", self.cart_text))
        order_tool = self.tool_of("order")
        finish_tool = self.tool_of("finish")
        decline_tool = self.tool_of("decline")
        attempts = []
        if cart_has or (not decline_tool and not self.decline_args):
            if order_tool:
                attempts.append((order_tool, {}))
            if finish_tool and self.order_args:
                attempts.append((finish_tool, dict(self.order_args)))
        else:
            if decline_tool:
                attempts.append((decline_tool, {}))
            if finish_tool and self.decline_args:
                attempts.append((finish_tool, dict(self.decline_args)))
        for name, args in attempts:
            if self.step >= self.max_steps or self.finished:
                break
            sch = self.schemas.get(name) or {}
            for k in sch.get("required") or []:
                if k not in args:
                    v = (sch.get("properties") or {}).get(k) or {}
                    if v.get("type") == "string" and "enum" not in v:
                        args[k] = "Done."
            args, problems = check_args(args, sch)
            if problems:
                continue
            obs, done = self.call(name, args)
            self.log("end", kind="forced", tool=name, result=_short(render(obs), 200))
            if done:
                break


def agent_main(problem_data: dict) -> list:
    ep = None
    try:
        ep = Episode(problem_data)
        ep.run()
    except Exception as exc:  # never crash: try to end the visit cleanly
        if ep is not None:
            ep.log("crash", error=_short(repr(exc), 400))
            try:
                ep.force_finish("crash")
            except Exception as exc2:
                ep.log("crash2", error=_short(repr(exc2), 200))
        else:
            return [{"k": "crash", "error": _short(repr(exc), 400)}]
    ep.log("summary", steps=ep.step, max_steps=ep.max_steps, llm_calls=ep.llm_calls, llm_fail=ep.llm_fail,
           usd=round(ep.spent, 4), messages=ep.messages_used, pages=len(ep.pages), outcome=ep.outcome,
           learned=ep.learned)
    return ep.trace


__all__ = ["agent_main"]
