"""S1 decoy shopper for ORO v3 worlds (rebuilt baseline).

A fast model drives the store's own tools, one reply at a time. The code around it:
  * reads the request and the tool list once: what is wanted, basket lines, limits, which tool does what;
  * learns hidden wishes by asking about one store label at a time, in the store's own spelling;
  * fact-checks a listing before it is carted and the cart before an order, and audits a decline;
  * on unranked single-item tasks sends the first order with a checked second choice in the cart, so check-out
    surprises land on that one and the real pick stays clean (the decoy);
  * keeps steps in reserve, validates arguments so a bad call costs nothing, and finishes the episode itself when
    steps or seconds run out.
"""
from __future__ import annotations

import json
import re
import time
from os import getenv
from typing import Any

from src.agent.proxy_client import ProxyClient

_env = ProxyClient(timeout=60, max_retries=1)
_llm = ProxyClient(timeout=60, max_retries=1)

# ================================================================ settings
OPENROUTER = (getenv("INFERENCE_PROVIDER") or "").strip().lower() == "openrouter"
TEE_CHAIN = ("zai-org/GLM-5.2-TEE", "moonshotai/Kimi-K2.6-TEE", "deepseek-ai/DeepSeek-V3.2-TEE")
# Fast chain: drives the tools, reads replies, checks listings. Strong chain: reads the request, plans questions,
# rates a measure and resolves pointers to other listings. About $2.7 per 90 tasks.
FAST = ("anthropic/claude-haiku-5.5", "openai/gpt-6-luna", "z-ai/glm-5.2") if OPENROUTER else TEE_CHAIN
STRONG = ("anthropic/claude-sonnet-5.5", "openai/gpt-6-sol", "anthropic/claude-haiku-5.5") if OPENROUTER else TEE_CHAIN
# The request reading decides everything after it and is one call per task: the strongest model reads it.
READER = ("anthropic/claude-opus-5.5", "anthropic/claude-sonnet-5.5", "openai/gpt-6-sol") if OPENROUTER else TEE_CHAIN

SECONDS = 262.0       # of the 300 s each problem allows
WRAP_UP_AT = 24.0     # seconds left when code ends the episode itself
RESERVE = 7           # steps held back for cart, checks and the order
SHOW = 7000           # longest tool result passed to the driver
TURNS = 90

INTENT_ORDER = ("order", "decline", "cart_add", "cart_remove", "search", "filter")
INTENTS = {
    "order": r"order|purchase|buy|check.?out|commit|confirm|accept|settle|pay",
    "decline": r"declin|refus|reject|abandon|cancel|abort|quit|give|skip|none|no.?purchase",
    "cart_add": r"add|put|include|insert|plus|hold|keep|increase",
    "cart_remove": r"remov|delet|drop|discard|release|minus|take|decrease|out",
    "search": r"search|query|keyword|phrase|text|word|typed|free",
    "filter": r"filter|facet|field|categor|department|aisle|shelf",
}
# words in a tool's name or description that tell which part it plays, for parts the request reading left open
TOOL_CUES = {
    "search": ("search", "find", "look", "query", "browse", "hunt"),
    "details": ("detail", "view", "open", "show", "full", "spec", "fact", "propert"),
    "cart": ("cart", "basket", "bag", "hold", "keep"),
    "order": ("checkout", "check out", "purchase", "buy", "order", "settle", "pay", "commit", "end"),
    "decline": ("decline", "give up", "no purchase", "refuse", "close", "end"),
    "message": ("message", "ask", "question", "customer", "shopper", "chat"),
}
ID_KEY = r"id|item|product|article|listing|sku|target|which"
QTY_KEY = r"qty|quant|amount|count|units|how|many|number"
GENERIC_WORDS = frozenset({"size", "type", "material", "color", "colour", "style", "length", "width", "height", "weight",
                           "pack", "number", "count", "value", "level", "per"})
NON_VALUE = re.compile(r"(any|anything|none|n/?a|no preference)\W*|(unspecified|not specified|not stated|unknown|to be asked)\b.*")
NOTHING_FITS = "I'm sorry - no listing in this store meets everything you asked for, so I'm not placing an order."

# ================================================================ wording
JSON_ONLY = "You work inside an automated shopping agent. Reply with a single JSON object and nothing else."

DRIVER_BRIEF = """You act for one shopper in a simulated store and can act only through the tools listed. The result \
is all or nothing: it counts only when the order is right in every respect - each requirement in the request, each \
wish the shopper reveals when asked, each change announced along the way, the right quantity within the budget, the \
cheapest (or best) fitting item when the request ranks, and each instruction the request gives about how to work. \
When no listing satisfies everything, the right result is a decline made the way the request asks.

USING THE TOOLS
- Each call uses one step, a failed one too; steps are few, thinking costs nothing. Calls that do not depend on each \
other (say, opening several listings) can share one reply; an order or a decline is always sent on its own.
- Every store names its tools, arguments and result fields its own way: follow the descriptions and schemas and use \
only the argument names and allowed values they give. A paging argument can be an offset (rows to skip) instead of a \
page number; when the store returns a next-page value, send that.
- Text search tends to require all words and also looks inside listing facts: search the required values on their \
own and with one or two product words; short queries work best.
- Find out early under which category label the store keeps the wanted product (listings show it). Whatever a title \
says, a listing in that category is the wanted kind; between similar labels, take the one whose whole name matches \
the request. Leave that category only when nothing inside fits.
- Compare the total a search reports with the rows shown; narrow the query or page on until the relevant listings \
are all seen.
- If one tool returns several listings in full side by side, open listings with it: one step, many records.
- A requirement tied to another listing ("the same X as the one sold by ...", "within N of the X shown for ...") is \
read on that listing first; the system looks these up and puts the value into the requirements.

DECIDING WHETHER A LISTING FITS
- A requirement holds when the listing's own fact for that property gives the value. Fact keys are spelled loosely \
(spaces, dots, underscores, prefixes, colour/color) and a property may sit under more than one key: it is enough that \
one states the value. Compare values by meaning: one item of a comma list, singular or plural, a close synonym, or a \
narrower kind containing the wanted word ("coil" is met by "3 single coil", "neck" by "scoop neck"); listings using \
the shopper's own word come first. A wider or different class is not a match.
- Use the title or description only when no fact key covers the property; a fact saying otherwise rules the listing \
out, title or not. When the shopper names a vague property (type, style, kind), a related key giving the value word \
for word counts.
- Numbers: convert units before comparing; "at least" and "at most" include the bound; a closed list ("A or B, \
nothing else") allows only those. Price times quantity stays within the budget; stock has to be there whenever the \
request asks for availability.

CHEAPEST AND BEST
- Cheapest: the order is the lowest price among EVERY fitting listing. Without a price filter, walk the product's \
category page by page so every price is seen; check candidates from the lowest price up; the first full fit (all \
cheaper ones checked and failed) wins; keep the next fits as backups.
- Best by a measure (most, biggest, longest, lowest X ...): compare every fit on that measure and take the extreme; a \
listing that does not state the measure cannot win.

THE SHOPPER
- The system plans and sends the questions to the shopper for you once two or three listings have been opened; never \
message the shopper yourself. A revealed wish is a hard requirement from then on; "don't mind" leaves the property \
open. Search again with revealed values.
- The shopper can change things while you work (budget, requirements, an item no longer available). Such news may \
come inside any tool result or as an order refusal; each change binds from then on: check your pick and backups again.

ENDING
- Keep a ranked list of fits with a backup. Before ordering, the cart holds only the chosen item at the right \
quantity, and its stock and price were read again after the last cart change.
- Send the first order while seven or more steps remain so a refusal can still be dealt with. With seven steps left \
only the cart, details and finishing tools remain.
- Decline only after several different searches found no fit; follow the request's rule for a decline; if it forbids \
substitutes, the remark names no listing, id, title or price.
- Every reply calls at least one tool. Keep text brief: requirements in force, ranked fits, next move."""

READ_PROMPT = """Below are a shopper's request and the store's tools. Return ONE JSON object:
{
 "product": the general kind of product wanted, using the request's words,
 "requirements": [{"property": the request's own name for it, "value": the required value, "relation": "is" | "at least" | "at most" | "one of"}],
 "budget": the spending limit as a number exactly as written (0 when none), "budget_text": the words that set it,
 "units": how many separate units to buy (1 unless more are asked for; a pack size the item comes in - "comes as 4 pcs", "a set of 6" - is a requirement, not units),
 "basket": [] for a single kind of product; when the request wants SEVERAL different products together, one entry each:
           {"product": its kind, "units": its units, "requirements": [as above, for that product alone]},
 "rank": "cheapest" | "best" | "none", "measure": for best, the quantity to maximise or minimise and which way,
 "message_limit": how many messages the shopper allows (number; -1 when not said),
 "step_limit": the most tool calls or steps the request allows (number; 0 when not said),
 "rules": [each instruction about how the assistant has to work, copied word for word],
 "learn_first": [anything only the shopper can tell before a choice is made, e.g. the budget when none is given],
 "quiet_decline": true when a decline may not name or suggest any listing,
 "pointers": [one per requirement whose value has to be read off ANOTHER listing the request refers to:
              {"property": what to read there, "clue": words that find that listing (its shop or brand, category and the
               properties given for it), "relation": how the wanted value relates ("same", "within 3", "at least"...)}],
 "searches": [four short queries: the required values alone, and with one or two product words],
 "tools": {"search": tool, "filter": tool or "", "details": tool, "cart": tool, "cart_view": tool or "", "order": tool,
           "decline": tool or "", "message": tool or ""}
}
Requirements are the details of the product itself (material, colour, amount with unit, pattern, shape, style, type, \
what it has to fit ...); alternatives stay one requirement ("A or B" -> relation "one of", value "A or B"); an occasion \
or use is a requirement only as a short store value ("formal", "wedding"), never as the story of one event. Budget, \
stock, category and the reason for buying are not requirements, and words already in the category's name are covered \
by the category. Saying money is tight does not ask for the cheapest. With several products, "product", \
"requirements" and "units" describe the first one, the budget covers the whole basket, and "cheapest" means the \
lowest total.

REQUEST:
{request}

TOOLS:
{tools}"""

QUESTION_PROMPT = """A shopper keeps back the wanted value of one or two product properties and gives it only when a \
message asks about that property by name; a message may name several. Rank up to {n} properties worth asking about, \
best first.
Request: {request}
Already required: {known}
Already asked: {asked}
To learn from the shopper first: {first}
Fact labels the store uses on listings like these (most frequent first), with sample values:
{labels}
Start from the side remarks in the request that hint at what matters beyond the stated needs (a habit, a worry, an \
earlier problem, who will wear or use it, where or how it will be seen). Reading labels or tags hints at material; \
looking good in photos or beside other things hints at pattern, colour and shape; the wearer or user hints at gender, \
age or size; fitting a body part, a device or a space hints at size and fit. Put what the hints point to first. With \
no hints, rank the labels: first those carried by most listings that describe the product itself (its kind or \
variant, make-up, main component, shape, fit, closure, finish, colour, size class, function) and whose values vary \
between listings, then the others. Pick only from the labels shown; never one already required or asked, never the \
measure the request ranks by, never two spellings of one property, never logistics (seller, warranty, condition, \
stock, price, option lists, model numbers, years).
Return JSON {{"hints": [...], "ask": [{{"property": label, "question": one short friendly question naming only that property, with one or two sample values}}]}}."""

FIT_PROMPT = """A shopping assistant is about to cart or order a listing. Test each candidate against all the shopper \
requires at this moment.
Product: {item}
Requirements:
{reqs}
For each candidate: kind is Y when it is the wanted product (a listing kept in the product's category counts whatever \
its title says), else N. Then, per requirement, find the facts whose label names that property (labels may use \
spaces, underscores or dots; more than one label may name it):
 Y - the label that names the property most exactly gives the value or one with the same meaning (singular/plural, a \
synonym, a list containing it; for "at least"/"at most" an amount on the allowed side; for "one of" any listed value); \
when several labels fit equally well, one giving the value is enough; Y as well when the property name is vague \
(type, style, kind) or no label names it and a closely related label gives the value word for word;
 T - no label names the property but the title plainly states the value;
 N - the most exact label gives another value;
 ? - nothing says.
Copy the label and its value exactly as written.
Return JSON {{"verdicts": [{{"number": n, "kind": "Y"|"N", "checks": [{{"req": number, "label": label, "value": value, "ok": "Y"|"T"|"N"|"?"}}]}}]}}.

CANDIDATES:
{cands}"""

NEWS_PROMPT = """Keep the list of what a shopper wants current.
Product: {item}
Requirements now: {reqs}
Budget now: {budget}
{context}
New words from the shopper or the store:
{text}
Return JSON {{"meaning": one short sentence, "adds": [{{"property": name, "value": bare value, "relation": "is"|"at least"|"at most"|"one of"}}], \
"drops": [properties withdrawn or replaced], "budget": the new limit exactly as written or 0, "budget_words": those \
words, "quantity": a new number of units or 0, "unavailable": true when an item can no longer be bought, \
"no_preference": true when the shopper does not mind}}.
A value given without a property belongs to the property it naturally describes. A changed value: drop the old \
property and add the new value. A "don't mind" answer changes nothing else."""

MEASURE_PROMPT = """The shopper will take, among listings meeting every requirement, the one doing best on: {measure}.
For each candidate give the figure its facts (else its options, else its title) state for that quantity, as one \
number in one shared unit (convert where needed); null when not stated. Say whether the best is the largest ("max") \
or the smallest ("min").
Return JSON {{"direction": "max"|"min", "values": [one number or null per candidate, in order]}}.

CANDIDATES:
{cands}"""

POINTER_PROMPT = """Which listing below is the one described as: {clue}? Then read its value for '{prop}'. Return \
JSON {{"number": n, or 0 when none matches, "value": that value exactly as the listing gives it, or ""}}.
{cands}"""

NOTE_PROMPT = """A seller note on a listing says one of its properties is to be read from a different listing.
Note: {note}
Return JSON {{"property": the property's name, "shop": the other listing's shop or brand, "kind": its kind of \
product, "details": any other words identifying it}}."""

NOTE_PICK_PROMPT = """Which listing below is: {target}? Read its value for '{prop}'. Return JSON {{"number": n or 0, \
"value": the value as written, or ""}}.
{cands}"""


# ================================================================ small helpers
def read_json(text: Any) -> Any:
    if not isinstance(text, str):
        return None
    body = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    for chunk in (body, body[body.find("{"):body.rfind("}") + 1] if 0 <= body.find("{") < body.rfind("}") else None):
        if chunk is None:
            break
        try:
            return json.loads(chunk)
        except ValueError:
            continue
    return None


def squeeze(value: Any, limit: int = 600) -> str:
    s = json.dumps(value, ensure_ascii=False, default=str, separators=(",", ":"))
    return s[:limit] + "..." if len(s) > limit else s


def tokens(text: Any) -> list:
    return re.findall(r"[a-z0-9]+", str(text or "").lower())


def stems(text: Any) -> set:
    return {w.rstrip("s") for w in tokens(text)}


def alike(a: str, b: str) -> bool:
    """One word in two spellings: equal once a plural s is dropped, or the same first four letters (colour/color)."""
    a, b = a.rstrip("s"), b.rstrip("s")
    return a == b or min(len(a), len(b)) >= 4 and a[:4] == b[:4]


def same_property(a: str, b: str) -> bool:
    x = [w for w in tokens(a) if len(w) > 2 and w not in ("type", "kind")]
    y = [w for w in tokens(b) if len(w) > 2 and w not in ("type", "kind")]
    if not (x and y):
        return False
    return all(any(alike(m, t) for t in y) for m in x) or all(any(alike(m, t) for t in x) for m in y)


def core_words(label: str) -> list:
    return [w for w in tokens(label) if w not in ("type", "types", "kind", "of", "the")]


def spellings_match(a: str, b: str) -> bool:
    """Two store labels for one property: the same words in other spellings ("pack type" / "packaging type"), or a
    shared specific word ("roast" / "coffee roast", "display" / "display size")."""
    ca, cb = core_words(a), core_words(b)
    if ca and len(ca) == len(cb) and all(any(alike(m, t) for t in cb) for m in ca):
        return True
    specific = lambda ws: {w.rstrip("s")[:6] for w in ws if len(w) >= 4 and w not in GENERIC_WORDS}
    return bool(specific(ca) & specific(cb))


def conform(args: dict, props: dict) -> tuple:
    """Arguments held against the schema: (cleaned arguments, problems found)."""
    clean, problems = {}, []
    for name, value in (args or {}).items():
        spec = props.get(name)
        if spec is None:
            problems.append(f"unknown argument '{name}'")
            continue
        kind = spec.get("type")
        try:
            if kind == "integer":
                value = int(round(float(value)))
                if "maximum" in spec:
                    value = min(value, int(spec["maximum"]))
                if "minimum" in spec:
                    value = max(value, int(spec["minimum"]))
            elif kind == "number":
                value = float(value)
                value = int(value) if value == int(value) else value
            elif kind == "string":
                value = str(value)[: int(spec["maxLength"])] if "maxLength" in spec else str(value)
            elif kind == "boolean":
                value = value.strip().lower() in ("true", "yes", "1") if isinstance(value, str) else bool(value)
        except (TypeError, ValueError):
            problems.append(f"'{name}' must be {kind}")
            continue
        allowed = spec.get("enum")
        if allowed and value not in allowed:
            same = [e for e in allowed if str(e).strip().lower() == str(value).strip().lower()]
            if not same:
                problems.append(f"'{name}' must be one of {allowed[:12]}")
                continue
            value = same[0]
        clean[name] = value
    return clean, problems


ID_SHAPE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-:.]{2,63}")


def find_id(x: dict) -> Any:
    """A listing's id, told by the value's shape (short, no spaces, a digit in it), an id-like key preferred; None for
    records that are not listings (a listing carries a number or a facts map besides its id)."""
    shaped = [(k, v) for k, v in x.items() if isinstance(v, str) and ID_SHAPE.fullmatch(v) and re.search(r"\d", v)]
    has_body = any(isinstance(v, (int, float)) and not isinstance(v, bool) for v in x.values()) or \
        any(isinstance(v, dict) for v in x.values())
    if not shaped or not has_body:
        return None
    keyed = [v for k, v in shaped if re.search(r"(^|[_.\s-])(id|ref|sku|pid|key|code)s?$|^id|pid|sku", k.lower())]
    if keyed:
        return keyed[0]
    return shaped[0][1] if len(shaped) == 1 else None


def facts_of(x: dict) -> Any:
    """The flat map of a record's facts: its first non-empty dict whose values are all plain."""
    return next((v for v in x.values() if isinstance(v, dict) and v
                 and all(not isinstance(z, (dict, list)) for z in v.values())), None)


def available(card: dict) -> bool:
    s = card.get("stock")
    if isinstance(s, str):
        return not re.search(r"\b(out|no|none|unavailable|sold|false|0)\b", s.lower())
    return s is not False


def says_literally(card: dict, need: dict) -> bool:
    want = " ".join(tokens(need.get("value")))
    if not want:
        return False
    pattern = r"(?<![a-z0-9])" + re.escape(want) + r"(?![a-z0-9])"
    return any(re.search(pattern, " ".join(tokens(v))) for v in (card.get("facts") or {}).values())


def errored(obs: Any) -> bool:
    return isinstance(obs, dict) and "error" in json.dumps(obs).lower()


# ================================================================ time and models
class Timer:
    def __init__(self) -> None:
        self.began = time.monotonic()

    def remaining(self) -> float:
        return SECONDS - (time.monotonic() - self.began)


class Models:
    """Chat calls down a chain of providers; reasoning switched off where allowed, kept low where required."""

    def __init__(self, timer: Timer) -> None:
        self.timer = timer
        self.log: list = []
        self.effort = {m: {"effort": "low"} for m in READER + STRONG + FAST if re.match(r"anthropic/claude-(sonnet|opus)", m)}

    @staticmethod
    def with_cache(messages: list) -> list:
        """Anthropic prompt caching: mark the system message and the newest user/tool message, so the conversation so
        far is read from the cache on the next turn."""
        out = [dict(m) for m in messages]
        marked = lambda text: [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]
        if out and out[0].get("role") == "system" and isinstance(out[0].get("content"), str):
            out[0]["content"] = marked(out[0]["content"])
        for i in range(len(out) - 1, 0, -1):
            m = out[i]
            if m.get("role") in ("user", "tool") and isinstance(m.get("content"), str) and m["content"]:
                m["content"] = marked(m["content"])
                break
        return out

    def ask(self, messages: list, chain: tuple, max_tokens: int, tools: Any = None) -> Any:
        for model in chain:
            if self.timer.remaining() < 8:
                return None
            anthropic = model.startswith("anthropic/")
            tries = 0
            while tries < 3:
                tries += 1
                body: dict = {"model": model, "temperature": 0, "max_tokens": max_tokens,
                              "messages": self.with_cache(messages) if anthropic else messages}
                if model in self.effort:
                    body["reasoning"] = self.effort[model]
                elif not anthropic:
                    body["reasoning"] = {"enabled": False}
                if tools:
                    body["tools"], body["tool_choice"] = tools, "auto"
                began = time.monotonic()
                res = _llm.post_verbose("/inference/chat/completions", json_data=body)
                code = res.error.get("status") if isinstance(res.error, dict) else None
                self.log.append({"model": model, "s": round(time.monotonic() - began, 1), "ok": res.ok, "status": code})
                if res.ok:
                    msg = (((res.data or {}).get("choices") or [{}])[0]).get("message") or {}
                    if msg.get("content") or msg.get("tool_calls"):
                        return msg
                    break                                  # an empty answer: next provider
                if code == 400 and model not in self.effort and "reasoning" in str(res.error).lower():
                    self.effort[model] = {"effort": "low"}  # this model insists on reasoning
                    continue
                spare = self.timer.remaining() > 20
                if code in (408, 429) and tries < 3 and spare:
                    time.sleep(1.5 * 2 ** (tries - 1))
                    continue
                if code in (500, 502, 503, 504) and tries < 2 and spare:
                    time.sleep(0.8)
                    continue
                break
        return None

    def json(self, prompt: str, chain: tuple = STRONG, max_tokens: int = 900, accept=None) -> Any:
        convo = [{"role": "system", "content": JSON_ONLY}, {"role": "user", "content": prompt}]
        good = lambda o: o is not None and (accept is None or accept(o))
        msg = self.ask(convo, chain, 2 * max_tokens + 1200)
        obj = read_json((msg or {}).get("content"))
        if not good(obj) and msg is not None and len(chain) > 1:   # a malformed answer: once more, next provider
            msg = self.ask(convo, chain[1:], 2 * max_tokens + 1200)
            obj = read_json((msg or {}).get("content"))
        return obj if good(obj) else None


# ================================================================ the store's tools
class Toolbox:
    """The tool schemas, which tool plays which part, and calls built by code for a part."""

    def __init__(self, tools: list) -> None:
        self.specs: dict = {}
        for t in tools:
            fn = t.get("function") if "function" in t else t
            if isinstance(fn, dict) and fn.get("name"):
                self.specs[fn["name"]] = fn
        self.offer = [{"type": "function", "function": fn} for fn in self.specs.values()]
        self.roles: dict = {}
        self.many = self._side_by_side()

    def params(self, tool: Any) -> dict:
        return ((self.specs.get(tool) or {}).get("parameters") or {}).get("properties") or {}

    def needed(self, tool: str) -> list:
        return ((self.specs.get(tool) or {}).get("parameters") or {}).get("required") or []

    def enum(self, tool: str, arg: str) -> list:
        spec = self.params(tool).get(arg) or {}
        return list(spec.get("enum") or []) if isinstance(spec, dict) else []

    def adopt(self, named: Any) -> None:
        """Roles from the request reading, then a guess from tool names and descriptions for the parts left open."""
        if isinstance(named, dict):
            self.roles = {k: v for k, v in named.items() if isinstance(v, str) and v in self.specs}
        for role, cues in TOOL_CUES.items():
            if role in self.roles:
                continue
            for name, spec in self.specs.items():
                text = (name + " " + str(spec.get("description", ""))).lower().replace("_", " ")
                if any(c in text for c in cues):
                    self.roles[role] = name
                    break

    def intent(self, tool: Any, args: dict) -> str:
        """The part a call plays; a shared tool is told apart by the allowed value it was given."""
        props = self.params(tool)
        for role in INTENT_ORDER:
            if any(isinstance(v, str) and re.search(INTENTS[role], v.lower()) and props.get(k, {}).get("enum")
                   for k, v in args.items()):
                return role
        for role, t in self.roles.items():
            if t == tool:
                return "cart_add" if role == "cart" else role
        return ""

    def fill_required(self, tool: str, intent: str, args: dict) -> dict:
        """Required options the call left out, filled from their allowed values by what the call is for."""
        pat = INTENTS.get(intent)
        for a in self.needed(tool):
            spec = self.params(tool).get(a) or {}
            if a in args or not spec.get("enum"):
                continue
            args[a] = next((v for v in spec["enum"] if pat and re.search(pat, str(v).lower())), spec["enum"][0])
        return args

    def build(self, intent: str, values: dict) -> tuple:
        """(tool, args) for a call made by code (cart, details, order, decline, message), checked and completed."""
        tool = self.roles.get("cart" if intent in ("cart_add", "cart_remove") else intent)
        if not tool or tool not in self.specs:
            return None, None
        props = self.params(tool)
        texts = [a for a, p in props.items() if p.get("type") == "string" and not p.get("enum")]
        nums = [a for a, p in props.items() if p.get("type") in ("integer", "number")]
        args: dict = {}
        if "id" in values:
            key = next((a for a in texts if re.search(ID_KEY, a.lower())), texts[0] if texts else None)
            if key:
                args[key] = values["id"]
        if "qty" in values and nums:
            args[next((a for a in nums if re.search(QTY_KEY, a.lower())), nums[0])] = values["qty"]
        if "text" in values and texts:
            args[texts[0]] = values["text"]
        listed = next((a for a, p in props.items() if p.get("type") == "array" and isinstance(p.get("items"), dict)
                       and isinstance(p["items"].get("properties"), dict)), None)
        if listed and "id" in values and not any(re.search(ID_KEY, a.lower()) for a in args):
            inner = props[listed]["items"]["properties"]    # a cart set from a list of lines replaces its contents
            itexts = [a for a, q in inner.items() if q.get("type") == "string" and not q.get("enum")]
            inums = [a for a, q in inner.items() if q.get("type") in ("integer", "number")]
            row: dict = {}
            key = next((a for a in itexts if re.search(ID_KEY, a.lower())), itexts[0] if itexts else None)
            if key:
                row[key] = values["id"]
            if inums:
                row[next((a for a in inums if re.search(QTY_KEY + "|want", a.lower())), inums[0])] = values.get("qty") or 1
            args[listed] = [] if intent == "cart_remove" else [row]
        pat = INTENTS.get(intent)
        if pat:                                              # options naming the part (add/remove, commit/refuse)
            for a, p in props.items():
                pick = next((v for v in p.get("enum") or [] if re.search(pat, str(v).lower())), None)
                if pick is not None:
                    args[a] = pick
        args, _ = conform(args, props)
        return tool, self.fill_required(tool, intent, args)

    def _side_by_side(self) -> str:
        """A tool that returns several listings in full from one list of ids."""
        for name, spec in self.specs.items():
            body = spec.get("parameters") or spec
            props = body.get("properties") or {}
            arrays = [a for a, q in props.items() if isinstance(q, dict) and q.get("type") == "array"
                      and (q.get("items") or {}).get("type") == "string"]
            text = (name + " " + str(spec.get("description", ""))).lower().replace("_", " ")
            if len(arrays) == 1 and set(body.get("required") or []) <= set(arrays) and \
                    re.search(r"compar|side by side|several|match|many|bulk", text):
                return name
        return ""

    def page_param(self, tool: str) -> str:
        props = self.params(tool).items()
        offset = next((a for a, s in props if isinstance(s, dict) and s.get("type") == "integer"
                       and re.search(r"first|offset|start|skip|page|from|cursor|index|at$", a.lower())), "")
        return offset or next((a for a, s in props if isinstance(s, dict) and s.get("type") == "string"
                               and re.search(r"bookmark|cursor|token|resume|continu|onward|next", a.lower())), "")

    def with_page(self, tool: str, args: dict, arg: str, nxt: Any) -> dict:
        """Arguments for one page: a string cursor goes only when the store handed one out (none for page one)."""
        if nxt in (0, None, "") and (self.params(tool).get(arg) or {}).get("type") == "string":
            return {k: v for k, v in args.items() if k != arg}
        return {**args, arg: nxt}


# ================================================================ what the store has shown
class Shelf:
    """Listing rows from lists, full records with facts, store notices already seen, and the searches made."""

    def __init__(self, tools: Toolbox) -> None:
        self.tools = tools
        self.seen: dict = {}       # id -> latest row from a list
        self.full: dict = {}       # id -> latest full record
        self.noticed: set = set()
        self.searches: list = []
        self.cursors: set = set()  # next-page values the store handed out

    def knows(self, pid: Any) -> bool:
        return pid in self.seen or pid in self.full

    def absorb(self, obs: Any) -> list:
        """Store the rows and records in a result; return the shopper/store sentences that are new."""
        found: list = []

        def walk(x: Any, depth: int) -> None:
            if depth > 6:
                return
            if isinstance(x, dict):
                pid = find_id(x)
                if pid and facts_of(x) is not None:
                    self.full[pid] = {**(self.full.get(pid) or {}), **x}
                elif pid:
                    self.seen[pid] = {**self.seen.get(pid, {}), **x}
                    if pid in self.full:           # a list or the cart may show a fresher price or stock
                        self.full[pid] = {**self.full[pid], **{k: v for k, v in x.items() if isinstance(v, (int, float, bool))}}
                for v in x.values():
                    walk(v, depth + 1)
            elif isinstance(x, list):
                if x and all(isinstance(v, str) for v in x):
                    found.extend(v for v in x if len(v.split()) >= 4 and v not in self.noticed)
                    return
                for v in x:
                    walk(v, depth + 1)

        walk(obs, 0)
        if isinstance(obs, dict) and isinstance(obs.get("refusal"), str):
            found.append(obs["refusal"])
        self.noticed.update(found)
        return found

    def card(self, pid: str) -> dict:
        """One listing in plain terms: title, category, price, stock, facts, seller notes."""
        r = {**self.seen.get(pid, {}), **self.full.get(pid, {})}
        textual = [k for k, v in r.items() if isinstance(v, str)
                   and not re.search(r"categ|aisle|shelf|brand|maker|company|vendor|seller", k.lower())]
        own = [k for k in textual if not re.search(r"shop|brand|seller|store|vendor|make|maker", k.lower())]  # not a shop name
        title_key = next((k for k in own if k.lower() in ("title", "name")), None) \
            or next((k for k in own if re.search(r"title|heading|headline", k.lower())), None) \
            or next((k for k in own if "name" in k.lower()), None)
        numbers = [(k, v) for k, v in r.items() if isinstance(v, (int, float)) and not isinstance(v, bool)]
        price = next((v for k, v in numbers if re.search(r"price|cost", k.lower())), None)
        if price is None:                          # some stores say "amount" for the price (not "amount_total")
            price = next((v for k, v in numbers if re.fullmatch(r"(unit_?)?amount|amount_?each|charge|fee", k.lower())), None)
        return {
            "id": pid,
            "title": r.get(title_key, "") if title_key else "",
            "category": next((v for k, v in r.items() if isinstance(v, str)
                              and re.search(r"categ|aisle|shelf|department|section", k.lower())), ""),
            "price": price,
            "stock": next((v for k, v in r.items() if re.search(r"stock|avail|on_?hand|in_?hand|sellable", k.lower())), True),
            "facts": facts_of(r) or {},
            "notes": " ".join(str(v) for k, v in r.items() if isinstance(v, str) and v
                              and re.search(r"note|remark|seller_?comment", k.lower())),
            "paged": pid in self.full,
        }

    def category(self, pid: str) -> str:
        return str(self.card(pid).get("category") or "").lower()

    def price(self, pid: str) -> Any:
        return self.card(pid).get("price")

    def text_param(self, search: dict) -> str:
        return next((a for a, v in search["args"].items() if isinstance(v, str) and len(v) > 1
                     and v not in self.tools.enum(search["name"], a)), "")

    def log_search(self, tool: str, args: dict, obs: Any) -> dict:
        """Remember a search: the listings it showed and where its next page starts."""
        text = obs if isinstance(obs, str) else json.dumps(obs)
        ids = [p for p in re.findall(r"p-[0-9a-z]{6,}|[A-Za-z0-9_-]{6,}", text) if p in self.seen][:60]
        ids = list(dict.fromkeys(ids))
        nxt = total = None
        keys = obs if isinstance(obs, dict) else {}
        pageish = lambda d: any(re.search(r"next|onward|bookmark|cursor|continu|match|total|count|found", str(k).lower()) for k in d)
        if not pageish(keys):                     # a result may wrap its listings and paging one level down
            keys = next((v for v in keys.values() if isinstance(v, dict) and pageish(v)), keys)
        for k, v in keys.items():
            if re.search(r"next|onward|bookmark|cursor|continu", k.lower()) and v not in (None, "", False):
                nxt = v                                # a next offset, or a cursor to send back
            elif isinstance(v, int) and not isinstance(v, bool) and re.search(r"match|total|count|found", k.lower()):
                total = v
        at = args.get(self.tools.page_param(tool))
        if nxt is None and total and isinstance(at, int) and ids and at + len(ids) < total \
                and not any(re.search(r"next", str(k).lower()) for k in keys):
            nxt = at + len(ids)                        # no next-page field: the offset past the rows shown
        if isinstance(nxt, str):
            self.cursors.add(nxt)
        self.searches.append({"name": tool, "args": dict(args), "ids": ids, "next": nxt, "total": total})
        return self.searches[-1]


class Line:
    """One product of a basket request."""

    def __init__(self, item: str, units: int, needs: list) -> None:
        self.item, self.units, self.needs = item, units, needs
        self.marks: dict = {}
        self.pick: Any = None      # the listing carted for this line
        self.asked: list = []
        self.start: Any = None     # step at which work on the line began
        self.tried = False
        self.filled = False
        self.guess = False         # carted as the closest match, not a confirmed fit


# ================================================================ judging listings
class Judging:
    def need_text(self, r: dict) -> str:
        rel = str(r.get("relation") or "is").lower()
        return f"{r['property']}: {rel + ' ' if rel in ('at least', 'at most', 'one of') else ''}{r['value']}"

    def rank(self) -> str:
        return str(self.reading.get("rank") or "none").lower()

    def ranks(self) -> bool:
        return self.rank() in ("cheapest", "best")

    def within_budget(self, card: dict) -> bool:
        p = card.get("price")
        return not (self.cap and isinstance(p, (int, float)) and p * self.units > self.cap + 1e-9)

    def judge(self, pids: list) -> None:
        pids = [p for p in dict.fromkeys(pids) if p][:8]
        if not pids:
            return
        for p in pids:                         # a seller note may send a property to another listing: read it there
            self.follow_seller_note(p)
        reqs = "\n".join(f"{i + 1}. {self.need_text(r)}" for i, r in enumerate(self.needs)) or "(only the kind of product)"
        listed = []
        for n, p in enumerate(pids):
            c = self.shelf.card(p)
            listed.append(f"{n + 1}. {c['title']} | category: {c['category']} | facts: {squeeze(c['facts'], 900)}"
                         + (f" | seller notes: {c['notes'][:300]}" if c.get("notes") else ""))
        got = self.models.json(FIT_PROMPT.format(item=self.item, reqs=reqs, cands="\n".join(listed)), FAST,
                               150 + 60 * len(pids), accept=lambda o: isinstance(o, dict) and isinstance(o.get("verdicts"), list)) or {}
        for v in got.get("verdicts") or []:
            try:
                n = int(v.get("number"))
            except (TypeError, ValueError, AttributeError):
                continue
            if 1 <= n <= len(pids):
                self.marks[pids[n - 1]] = self.score(self.shelf.card(pids[n - 1]), v)

    def score(self, card: dict, verdict: dict) -> str:
        """Y / T / N / ? from the per-requirement checks; a fact Y stands only when its quote is really in the facts."""
        if verdict.get("kind") == "N":
            return "N"
        norm = lambda s: re.sub(r"[\s_.]+", " ", str(s)).strip().lower()
        facts = {norm(k): str(x).lower() for k, x in (card.get("facts") or {}).items()}
        title = str(card.get("title") or "").lower()
        per: dict = {}
        for c in verdict.get("checks") or []:
            try:
                i = int(c.get("req"))
            except (TypeError, ValueError, AttributeError):
                continue
            said, value = c.get("ok"), str(c.get("value") or "").lower().strip()
            in_title = bool(value) and value in title
            if said == "Y":
                held = facts.get(norm(c.get("label") or ""))
                said = "Y" if held is not None and value and (value in held or held in value) else ("T" if in_title else "?")
            elif said == "T":
                said = "T" if in_title else "?"
            per[i] = "Y" if per.get(i) == "Y" else said
        shelf_words = set(tokens(card.get("category")))
        for i, r in enumerate(self.needs, 1):
            vw = set(tokens(r.get("value")))
            if vw and vw <= shelf_words:
                per[i] = "Y"                       # the category's own name says it
        self.literal_hits[card["id"]] = sum(1 for r in self.needs if says_literally(card, r))
        got = [per.get(i) for i in range(1, len(self.needs) + 1)]
        if "N" in got:
            return "N"
        if all(g in ("Y", "T") for g in got):
            return "T" if "T" in got else "Y"
        return "?"

    def ranked_fits(self) -> list:
        """Confirmed fits, best first: fact-confirmed before title-confirmed, then by what the request ranks by."""
        cards = [self.shelf.card(p) for p, m in self.marks.items() if m in ("Y", "T") and p not in self.dead]
        cards = [c for c in cards if available(c) and self.within_budget(c)]
        rank = self.rank()
        if rank == "best" and len(cards) > 1:
            self.rate_measure([c["id"] for c in cards])

            def by_measure(c: dict) -> tuple:
                v = self.measured.get(c["id"])
                return self.marks[c["id"]] != "Y", v is None, -(v or 0) if self.measure_dir == "max" else (v or 0)
            cards.sort(key=by_measure)
        elif rank == "cheapest":
            cards.sort(key=lambda c: (self.marks[c["id"]] != "Y", c.get("price") or 0))
        else:
            # unranked: a typical item of the wanted kind (named in its title, else early in the store's own order) is
            # the safest bet for wishes never revealed
            kind = {w.rstrip("s") for w in tokens(self.item) if len(w) > 2 and w not in ("and", "for", "the", "with")}
            order = {p: i for i, p in enumerate(list(self.shelf.seen) + [p for p in self.shelf.full if p not in self.shelf.seen])}
            cards.sort(key=lambda c: (self.marks[c["id"]] != "Y", -self.literal_hits.get(c["id"], 0),
                                      -len(kind & stems(c.get("title"))), order.get(c["id"], 10 ** 6)))
        return [c["id"] for c in cards]

    def rate_measure(self, pids: list) -> None:
        todo = [p for p in pids if p not in self.measured][:10]
        if not todo:
            return
        cands = "\n".join(f"{n + 1}. {self.shelf.card(p)['title']} | facts: {squeeze(self.shelf.card(p)['facts'], 600)}"
                          for n, p in enumerate(todo))
        got = self.models.json(MEASURE_PROMPT.format(measure=self.reading.get("measure") or "the stated measure", cands=cands),
                               STRONG, 300, accept=lambda o: isinstance(o, dict) and isinstance(o.get("values"), list)) or {}
        if got.get("direction") in ("max", "min"):
            self.measure_dir = got["direction"]
        values = got.get("values") or []
        for i, p in enumerate(todo):
            try:
                self.measured[p] = float(values[i])
            except (TypeError, ValueError, IndexError):
                self.measured[p] = None


# ================================================================ the shopper
class Asking:
    def asked_here(self) -> list:
        return self.lines[self.at].asked if self.is_basket() else self.asked_props

    def take_news(self, text: str, source: str, about: str = "") -> str:
        """Read a shopper reply or store notice into the requirements, budget and units; returns a line for the driver."""
        self.heard.append(f"{source}: {text}")
        got = self.models.json(NEWS_PROMPT.format(item=self.item, reqs=squeeze(self.needs, 1200), budget=self.cap or "none",
                                                  context=f"The assistant had just asked about: {about}" if about else "",
                                                  text=text[:3000]), FAST, 600, accept=lambda o: isinstance(o, dict)) or {}
        self.last_news = got
        indifferent = bool(got.get("no_preference"))
        drops = set() if indifferent and source == "shopper" else {str(d).lower() for d in got.get("drops") or []}
        changed = reworded = False                 # reworded: this product's requirements changed
        if drops:
            kept = self.without(self.needs, drops)
            changed = reworded = len(kept) != len(self.needs)
            self.needs = kept
            for i, ln in enumerate(self.lines):
                if i != self.at and self.is_basket():
                    rest = self.without(ln.needs, drops)
                    if len(rest) != len(ln.needs):
                        ln.needs, ln.marks, changed = rest, {}, True
        for add in got.get("adds") or []:
            if indifferent or not (isinstance(add, dict) and add.get("property") and add.get("value")):
                continue
            prop = str(add["property"]).lower()
            owner = next((i for i, ln in enumerate(self.lines) if i != self.at and same_property(prop, ln.item)),
                         None) if self.is_basket() else None
            if owner is not None:                  # it belongs to another product of the basket
                ln = self.lines[owner]
                ln.needs = [r for r in ln.needs if str(r["property"]).lower() != prop] + [add]
                ln.marks = {}
            else:
                self.needs = [r for r in self.needs if str(r["property"]).lower() != prop] + [add]
                reworded = True
            changed = True
        cap = self.number_in(got.get("budget"), got.get("budget_words"), text)
        if cap > 0 and cap != self.cap:
            self.cap, changed = cap, True
        try:
            units = int(got.get("quantity") or 0)
        except (TypeError, ValueError):
            units = 0
        if units > 0 and units != self.units:
            self.units, changed = units, True
        if got.get("unavailable") and self.cart and source != "store refusal":
            self.dead.update(self.cart)
        if reworded:                               # budget and units are checked live; verdicts depend on requirements
            self.was_fit |= {k for k, v in self.marks.items() if v in ("Y", "T")}
            self.marks = {k: v for k, v in self.marks.items() if v == "N" and not drops}
            self.recheck = True
        line = f"[news] {got.get('meaning') or text[:200]} "
        if changed:
            line += (f"Requirements in force: {[self.need_text(r) for r in self.needs]}; budget {self.cap or 'none'}; "
                     f"units {self.units}.")
        return line.strip()

    @staticmethod
    def without(needs: list, drops: set) -> list:
        """Requirements left after the shopper withdraws some: by exact name, else by a close name ("closure" for
        "shoes closure type")."""
        names = [str(r["property"]).lower() for r in needs]
        gone = set()
        for d in drops:
            exact = {i for i, n in enumerate(names) if n == d}
            gone |= exact or {i for i, n in enumerate(names) if same_property(n, d)}
        return [r for i, r in enumerate(needs) if i not in gone]

    def refresh_fits(self) -> str:
        """After the requirements changed: check the opened listings of this product again at once (earlier fits first,
        then the newest), so the driver works from fits under the requirements now in force."""
        if not self.recheck:
            return ""
        self.recheck = False
        if self.steps_left() <= 2 or self.timer.remaining() < 45:
            return ""
        mine = [p for p in self.shelf.full if p not in self.marks and p not in self.dead
                and (not self.is_basket() or self.line_of(p) == self.at)]
        mine.sort(key=lambda p: (p not in self.was_fit, -list(self.shelf.full).index(p)))
        if not mine:
            return ""
        self.judge(mine[:8])
        return (f"\n[system] the requirements changed; opened listings checked again against those now in force "
                f"({[self.need_text(r) for r in self.needs]}). Confirmed fits now: {self.ranked_fits()[:4]}.")

    def store_labels(self) -> str:
        """The fact labels of opened listings of the wanted kind, most common first, with sample values."""
        pool = [p for p in self.shelf.full if self.line_of(p) == self.at] if self.is_basket() else list(self.shelf.full)
        kind = {w.rstrip("s") for w in tokens(self.item) if len(w) > 2}
        pool = [p for p in pool if kind & stems(self.shelf.card(p).get("category"))] or pool
        groups: dict = {}
        for pid in pool:
            for key, val in (self.shelf.card(pid)["facts"] or {}).items():
                spaced = re.sub(r"[_.]+", " ", re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(key)))   # FurnitureType
                ws = tokens(spaced)
                core = [w for w in ws if w not in ("type", "types", "fa", "general")]
                if not core:
                    continue
                g = groups.setdefault(" ".join(core), {"n": 0, "names": [], "values": []})
                g["n"] += 1
                name = " ".join(w for w in ws if w not in ("fa", "general"))
                if name not in g["names"] and len(g["names"]) < 3:
                    g["names"].append(name)
                if val not in g["values"] and len(g["values"]) < 5:
                    g["values"].append(val)
        top = sorted(groups.values(), key=lambda g: -g["n"])[:28]
        return "\n".join(f"- {g['names'][0]} ({g['n']}): {', '.join(map(str, g['values']))}" for g in top)

    def covered(self, prop: str) -> bool:
        """Already required or already asked, under any spelling."""
        mine = [w for w in tokens(prop) if len(w) > 2 and w not in ("type", "kind")]
        for other in [r["property"] for r in self.needs] + self.asked_here():
            theirs = [w for w in tokens(other) if len(w) > 2 and w not in ("type", "kind")]
            if mine and theirs and all(any(alike(m, t) for t in theirs) for m in mine):
                return True
        return False

    def add_pick(self, picks: list, name: str) -> None:
        """A new property to ask about, or another store spelling of one already picked: the shopper's reader knows a
        property only by its own label, so the spellings share one question (two at most)."""
        if self.is_basket() and any(i != self.at and same_property(name, ln.item) for i, ln in enumerate(self.lines)):
            return                                 # names another basket product ("ram" beside a "ram modules" line)
        for first, spellings in picks:
            if any(s.lower() == name.lower() for s in spellings):
                return
            if spellings_match(name, first):
                if len(spellings) < 2:
                    spellings.append(name)
                return
        picks.append((name, [name]))

    def rank_picks(self) -> list:
        """The working product's labels ranked for questions: [(store label, [its spellings in this store])]."""
        request = f"[ask only about this product of the basket: {self.item}] {self.request}" if self.is_basket() else self.request
        labels = self.store_labels()
        got = self.models.json(QUESTION_PROMPT.format(n=12, request=request, known=[self.need_text(r) for r in self.needs],
                                                      asked=self.asked_here(), first=self.reading.get("learn_first") or "-",
                                                      labels=labels or "(none seen yet)"),
                               STRONG, 1100, accept=lambda o: isinstance(o, dict) and isinstance(o.get("ask"), list)) or {}
        names = re.findall(r"^- (.+?) \(\d+\):", labels, re.M)
        measure = str(self.reading.get("measure") or "") if self.ranks() else ""
        picks: list = []
        for q in got.get("ask") or []:
            if not (isinstance(q, dict) and q.get("property") and q.get("question")):
                continue
            prop = str(q["property"]).strip()
            exact = [l for l in names if l.lower() == prop.lower()] or [l for l in names if same_property(prop, l)]
            prop = exact[0] if exact else prop
            if not (measure and same_property(prop, measure)) and not self.covered(prop):
                self.add_pick(picks, prop)
        for name in names:                        # topped up from the store's own labels
            if len(picks) >= 12:
                break
            if self.covered(name) or (measure and same_property(name, measure)) or \
                    re.search(r"option|model|warranty|condition|seller|year|asin|code", name):
                continue
            self.add_pick(picks, name)
        return picks

    def send_question(self, spellings: list) -> str:
        """One message naming one property (in every store spelling) for the working product; the reply is read in."""
        listed = " / ".join(spellings)
        item = (self.item or "item").strip()
        tool, args = self.tools.build("message", {"text": f"Which {listed} would you prefer for the {item}?"})
        if not tool:
            return ""
        obs = self.act(tool, args)
        self.msgs_left -= 1
        self.asked_props.extend(s for s in spellings if s not in self.asked_props)
        if self.is_basket():
            self.lines[self.at].asked.extend(spellings)
        texts = self.shelf.absorb(obs)
        reply = obs if isinstance(obs, str) else " ".join(texts) or squeeze(obs, 800)
        return f"Asked about {listed} ({item}): {self.take_news(reply, 'shopper', about=listed)}"

    def question_round(self, force: bool = False) -> str:
        """Pick and send the questions still allowed now; returns what was learned, for the driver."""
        if self.asking_done or self.msgs_left <= 0 or not self.tools.roles.get("message"):
            return ""
        if self.steps_left() <= RESERVE + 3 or self.timer.remaining() < 70:
            self.asking_done = not self.is_basket() and bool(self.shelf.full)
            return ""
        if self.is_basket():
            return self.basket_round(force)
        if not self.shelf.full or (not force and len(self.shelf.full) < 3 and self.step < 8):
            return ""                             # plan only from the labels of opened listings
        self.asking_done = True
        allowed = min(self.msgs_left, self.steps_left() - RESERVE - 2)
        picks = self.rank_picks()
        if not picks:                             # nothing to ask yet: one more try later
            self.empty_plans += 1
            self.asking_done = self.empty_plans >= 2
            return ""
        # The reader scores only the properties it holds back, and is surest when a message names a single one by its
        # exact label (measured: a lone label is revealed every time, a list of six about 40%). So each message names
        # the next pick; a reveal may settle later picks, which are then skipped.
        notes = []
        for _ in range(allowed):
            while picks and self.covered(picks[0][0]):
                picks.pop(0)
            if not picks or self.msgs_left <= 0:
                break
            note = self.send_question(picks.pop(0)[1])
            if not note:
                break
            notes.append(note)
        return "\n".join(notes) + self.refresh_fits()

    def basket_round(self, force: bool) -> str:
        """A basket: the shopper may hold a wish back on any of its products. Once every line has opened listings (or
        before a commit), each line's labels are ranked and the questions take turns between the lines, best label of
        each line first; a reveal is filed under the line asked about."""
        seen = lambda i: any(self.line_of(p) == i for p in self.shelf.full)
        if not force and not all(seen(i) for i in range(len(self.lines))):
            return ""
        self.asking_done = True
        home = self.at
        plans = {}
        for i in range(len(self.lines)):
            self.on_line(i)
            plans[i] = self.rank_picks()
        queue = [(i, plans[i][d]) for d in range(max(map(len, plans.values()), default=0))
                 for i in range(len(self.lines)) if d < len(plans[i])]
        allowed = min(self.msgs_left, self.steps_left() - RESERVE - 2)
        notes, touched = [], []
        for i, (label, spellings) in queue:
            if len(notes) >= allowed or self.msgs_left <= 0:
                break
            self.on_line(i)
            if self.covered(label):
                continue
            note = self.send_question(spellings)
            if not note:
                break
            notes.append(note)
            if self.recheck and i not in touched:
                touched.append(i)
        tail = ""
        for i in touched:                         # lines whose requirements changed: their opened listings again
            self.on_line(i)
            self.recheck = True
            tail += self.refresh_fits()
        self.on_line(home)
        return "\n".join(notes) + tail

# ================================================================ basket requests
class Basket:
    def is_basket(self) -> bool:
        return len(self.lines) > 1

    def park(self) -> None:
        """Save the working requirements, units and marks into the current basket line."""
        if self.is_basket():
            ln = self.lines[self.at]
            ln.needs, ln.units, ln.marks = self.needs, self.units, self.marks

    def on_line(self, i: int) -> None:
        """Make basket line i the working one without starting its step clock (planning, asking, looking ahead)."""
        if not self.is_basket() or i == self.at:
            return
        self.park()
        self.at = i
        ln = self.lines[i]
        self.needs, self.units, self.marks, self.item = ln.needs, ln.units, ln.marks, ln.item

    def switch_to(self, i: int) -> None:
        if not self.is_basket() or i == self.at:
            return
        self.on_line(i)
        if self.lines[i].start is None:
            self.lines[i].start = self.step

    def look_ahead(self) -> str:
        """Once, early in a basket: every line without opened listings gets one search and one read of its top results,
        so each product's labels are known before the questions and each line has early fits."""
        if not self.is_basket() or "ahead" in self.once or not self.first_text_search():
            return ""
        todo = [i for i, ln in enumerate(self.lines) if not ln.filled and not any(self.line_of(p) == i for p in self.shelf.full)]
        if not todo:
            self.once.add("ahead")
            return ""
        if self.steps_left() < RESERVE + 8 + 2 * len(todo) or self.timer.remaining() < 120:
            return ""
        self.once.add("ahead")
        home, notes = self.at, []
        for i in todo:
            self.on_line(i)
            if self.seed_line():
                notes.append(f"'{self.item}': {self.ranked_fits()[:3]}")
        self.on_line(home)
        return ("[system] looked ahead for the other basket lines; confirmed fits so far: " + "; ".join(notes)) if notes else ""

    def line_of(self, pid: str) -> int:
        """The basket line a listing belongs to: the one whose product its category and title name best."""
        card = self.shelf.card(pid)
        text = stems(f"{card.get('category')} {card.get('title')}")
        score = lambda ln: len(text & {w.rstrip("s") for w in tokens(ln.item) if len(w) > 2})
        best = max(range(len(self.lines)), key=lambda i: (score(self.lines[i]), i == self.at))
        return best if score(self.lines[best]) else self.at

    def carry_lines(self, tool: str, args: dict) -> dict:
        """A cart tool whose lines replace the cart has to carry the basket lines already carted too."""
        key = next((a for a, v in args.items() if isinstance(v, list) and all(isinstance(z, dict) for z in v)), None)
        if key is None:
            return args
        rows = [dict(z) for z in args[key]]
        named = {v for z in rows for v in z.values() if isinstance(v, str)}
        for ln in self.lines:
            if ln.pick and ln.pick not in named and self.cart.get(ln.pick):
                t, extra = self.tools.build("cart_add", {"id": ln.pick, "qty": self.cart[ln.pick]})
                if t == tool and isinstance(extra.get(key), list):
                    rows += extra[key]
        return {**args, key: rows}

    def settle_lines(self) -> None:
        """After a cart change each line takes the confirmed fit of its kind now carted; the next open line becomes
        the current one."""
        self.park()
        held = [p for p, q in self.cart.items() if q]
        for i, ln in enumerate(self.lines):
            mine = [p for p in held if self.line_of(p) == i and ln.marks.get(p) in ("Y", "T")]
            if mine:
                ln.pick = mine[0]
            elif not (ln.pick in held and (ln.guess or ln.marks.get(ln.pick) in ("Y", "T"))):
                ln.pick = None
        if self.lines[self.at].pick:
            nxt = next((i for i, ln in enumerate(self.lines) if not ln.pick), None)
            if nxt is not None:
                self.switch_to(nxt)

    def fill_open_lines(self) -> str:
        """Each line still without a fit gets its closest opened listing of its kind (unsure before rejected); returns
        what was carted, or '' when some line has nothing to offer."""
        self.park()
        order = {"Y": 0, "T": 1, "?": 2, None: 3, "N": 4}
        picks = []
        for i, ln in enumerate(self.lines):
            if ln.pick:
                continue
            mine = [p for p in self.shelf.full if self.line_of(p) == i and p not in self.dead and available(self.shelf.card(p))]
            mine.sort(key=lambda p: (order.get(ln.marks.get(p), 3), self.shelf.price(p) or 0))
            if not mine or ln.marks.get(mine[0]) == "N":
                return ""
            picks.append((i, mine[0]))
        if not picks or self.steps_left() < 3:
            return ""
        for i, pid in picks:
            tool, args = self.tools.build("cart_add", {"id": pid, "qty": self.lines[i].units})
            if not tool:
                return ""
            if errored(self.act(tool, self.carry_lines(tool, args))):
                return ""
            self.cart[pid] = self.lines[i].units
            self.lines[i].pick, self.lines[i].guess = pid, True
        self.order_cleared = True                 # a deliberate guess: let the order through
        return ", ".join(f"{self.lines[i].item}: {pid}" for i, pid in picks)

    def line_share(self) -> int:
        """Steps one line may use: what is left after the reserve and the questions, split evenly."""
        return max(6, (self.max_steps - RESERVE - 4) // max(1, len(self.lines)))

    def basket_note(self) -> str:
        if not self.is_basket():
            return ""
        parts = []
        for i, ln in enumerate(self.lines):
            state = f"chosen {ln.pick}" if ln.pick else ("IN PROGRESS" if i == self.at else "open")
            parts.append(f"{i + 1}. {ln.item} x{self.units if i == self.at else ln.units}: {state}")
        cur = self.lines[self.at]
        left = self.line_share() - (self.step - (self.step if cur.start is None else cur.start))
        return ("; BASKET (each line goes into the cart, then a single order; the budget covers the total): "
                + " | ".join(parts) + f"; roughly {max(0, left)} steps remain for the current line")

    def close_spent_line(self) -> str:
        """When the current line has used its share: cart its best confirmed fit, or leave it open, and move on."""
        if not self.is_basket() or self.steps_left() <= RESERVE + 2:
            return ""
        ln = self.lines[self.at]
        if ln.pick:
            return ""
        if ln.start is None:
            ln.start = self.step
        if self.step - ln.start < self.line_share():
            return ""
        fits = self.ranked_fits()
        note = ""
        if fits:
            tool, args = self.tools.build("cart_add", {"id": fits[0], "qty": self.units})
            if tool and not errored(self.act(tool, self.carry_lines(tool, args))):
                self.cart[fits[0]] = self.units
                note = f"carted its best confirmed fit {fits[0]}"
        if not note:
            ln.tried = True
            note = "no confirmed fit yet; it stays open for a later pass"
        was = self.at
        self.settle_lines()
        if self.at == was:                        # not moved on by a filled line: take the next open one
            nxt = next((i for i, x in enumerate(self.lines) if not x.pick and not x.tried and i != self.at), None)
            if nxt is None:
                nxt = next((i for i, x in enumerate(self.lines) if not x.pick and i != self.at), None)
            if nxt is not None:
                self.switch_to(nxt)
        self.lines[self.at].start = self.step
        return f"[system] basket line '{ln.item}' used up its share of steps: {note}. Now working on '{self.item}'."

    def seed_line(self) -> str:
        """Once per line: one search for it (the wanted value first; its shelf when the tool has shelves), then open
        and check the top results together, so the line has confirmed fits within a few steps."""
        ln = self.lines[self.at] if self.is_basket() else None
        if ln is None or ln.filled or self.steps_left() < RESERVE + 6 or self.timer.remaining() < 90:
            return ""
        base = self.first_text_search()
        if not base:
            return ""
        ln.filled = True
        item = self.item or ""
        kind = (tokens(item) or [""])[-1]
        args = self.fresh_args(base)
        on_shelf = self.choose_shelf(base["name"], args, stems(item))
        # inside the product's shelf the value alone finds most (titles often lack the product word); without one the
        # value leads and one product word follows
        vw = re.findall(r"[a-z0-9]+", self.short_value().lower())
        if on_shelf and vw:
            query = " ".join(vw[:3])
        elif vw and kind not in vw:
            query = " ".join((vw + [kind])[:2])
        else:
            query = " ".join(tokens(item)[:2])
        args[self.shelf.text_param(base)] = query
        ids = [p for p in self.run_search(base["name"], args)["ids"]
               if p not in self.shelf.full and p not in self.dead and self.line_of(p) == self.at]
        if self.rank() == "cheapest":
            ids.sort(key=lambda p: self.shelf.price(p) or 0)
        self.open_listings(ids[:4])
        todo = [p for p in self.shelf.full if p not in self.marks and self.line_of(p) == self.at]
        if todo:
            self.judge(todo[:8])
        return (f"[system] searched '{query}' for basket line '{item}' and checked the top results. Confirmed fits for "
                f"this line: {self.ranked_fits()[:3]}. Cart the best one (the cart keeps the other lines), then carry on.")


# ================================================================ reading more of the store
class Reading:
    def first_text_search(self) -> Any:
        return next((x for x in self.shelf.searches if self.shelf.text_param(x)), None)

    def fresh_args(self, search: dict) -> dict:
        page = self.tools.page_param(search["name"])
        return {k: v for k, v in search["args"].items() if k != page}

    def run_search(self, tool: str, args: dict) -> dict:
        obs = self.act(tool, args)
        self.shelf.absorb(obs)
        return self.shelf.log_search(tool, args, obs)

    def choose_shelf(self, tool: str, args: dict, wanted: set) -> bool:
        """Set each long list of allowed values (a shelf) to the entry naming the wanted product, if one does."""
        hit = False
        for a, spec in self.tools.params(tool).items():
            vals = [str(v) for v in spec.get("enum") or []] if isinstance(spec, dict) else []
            if len(vals) <= 4 or not wanted:
                continue
            best = max(vals, key=lambda v: len(stems(v) & wanted))
            if stems(best) & wanted:
                args[a], hit = best, True
        return hit

    def short_value(self) -> str:
        return next((str(r.get("value")) for r in self.needs if len(tokens(r.get("value"))) <= 2), "")

    def value_query(self) -> str:
        """The store's own reading of the request: the wanted value, then the product word (two words)."""
        value, kind = self.short_value(), (tokens(self.item) or [""])[-1]
        return " ".join((tokens(value) + [kind])[:2]) if value and kind and kind not in tokens(value) else ""

    def maker_fields(self, pid: str, pattern: str, limit: int) -> str:
        r = {**self.shelf.seen.get(pid, {}), **self.shelf.full.get(pid, {})}
        return squeeze({k: v for k, v in r.items() if re.search(pattern, str(k).lower())}, limit)

    def open_listings(self, pids: list) -> None:
        """Read several full records: one step with a side-by-side tool, else one step each."""
        pids = [p for p in dict.fromkeys(pids) if p][:6]
        many = self.tools.many
        if many and len(pids) > 1:
            body = self.tools.specs[many].get("parameters") or self.tools.specs[many]
            arg = next(a for a, q in (body.get("properties") or {}).items() if isinstance(q, dict) and q.get("type") == "array")
            obs = self.act(many, {arg: pids[:body["properties"][arg].get("maxItems") or 6]})
            self.shelf.absorb(obs)
            if not (isinstance(obs, dict) and re.search(r"error|oops|no known|not found|invalid", json.dumps(obs)[:300].lower())):
                return
            pids = [p for p in pids if p not in self.shelf.full][:3]   # one unknown id spoils the call: one by one
        for p in pids:
            tool, args = self.tools.build("details", {"id": p})
            if tool:
                self.shelf.absorb(self.act(tool, args))

    def resolve_pointers(self) -> str:
        """Requirements that point at another listing: find it, read its value, make the requirement concrete."""
        todo = [x for x in self.reading.get("pointers") or [] if isinstance(x, dict) and x.get("property")
                and x.get("clue") and not x.get("done")]
        if not todo or self.steps_left() < RESERVE + 8 or self.timer.remaining() < 100:
            return ""
        base = self.first_text_search()
        if not base:
            return ""
        notes = []
        for ptr in todo[:2]:
            ptr["done"] = True
            clue, prop = str(ptr["clue"]), str(ptr["property"])
            shops = re.findall(r"\b[A-Z][a-zA-Z]{3,}\b", clue)
            args = self.fresh_args(base)
            args[self.shelf.text_param(base)] = " ".join((shops[:1] + tokens(clue))[:2]) if shops else " ".join(tokens(clue)[:2])
            ids = self.run_search(base["name"], args)["ids"][:5]
            if not ids:
                continue
            self.open_listings(ids)
            cands = "\n".join(f"{n + 1}. {self.shelf.card(p)['title']} | seller/brand: "
                              f"{self.maker_fields(p, r'make|brand|seller|shop|vendor|sold', 160)} | facts: "
                              f"{squeeze(self.shelf.card(p)['facts'], 700)}" for n, p in enumerate(ids))
            got = self.models.json(POINTER_PROMPT.format(clue=clue, prop=prop, cands=cands), STRONG, 300,
                                   accept=lambda o: isinstance(o, dict)) or {}
            val = str(got.get("value") or "").strip()
            if not val:
                continue
            rel = str(ptr.get("relation") or "same").strip()
            new = {"property": prop, "value": val if rel.lower() in ("same", "is", "") else f"{rel} of {val}", "relation": "is"}
            swap = lambda needs: [new if same_property(str(r.get("property")), prop) else r for r in needs]
            if self.is_basket():
                self.park()
                for ln in self.lines:
                    ln.needs = swap(ln.needs)
                self.needs = self.lines[self.at].needs
            else:
                self.needs = swap(self.needs)
            self.marks = {k: v for k, v in self.marks.items() if v == "N"}
            notes.append(f"{prop} = {new['value']} (read from the listing described as: {clue})")
        return ("[system] requirements that point at another listing, now resolved: " + "; ".join(notes)) if notes else ""

    def follow_seller_note(self, pid: str) -> None:
        """A seller note like 'for sleeves, refer to the Koshivexka plain jumpsuit, not this item': find that listing,
        read the property there and keep it as this listing's value (two notes per task at most; steps are scarce)."""
        note = self.shelf.card(pid).get("notes") or ""
        if not note or pid in self.notes_followed or len(self.notes_followed) >= 2 or \
                not re.search(r"refer to|judged from|taken from|see the|same as|as on|look at", note, re.I):
            return
        if self.steps_left() < RESERVE + 6 or self.timer.remaining() < 100:
            return
        base = self.first_text_search()
        if not base:
            return
        self.notes_followed.add(pid)
        got = self.models.json(NOTE_PROMPT.format(note=note), FAST, 200,
                               accept=lambda o: isinstance(o, dict) and o.get("property")) or {}
        if not got.get("shop") and not got.get("kind"):
            return
        args = self.fresh_args(base)
        args[self.shelf.text_param(base)] = " ".join(tokens(str(got.get("shop") or got.get("kind")))[:2])
        self.choose_shelf(base["name"], args, stems(got.get("kind")))
        ids = [p for p in self.run_search(base["name"], args)["ids"] if p != pid][:4]
        if not ids:
            return
        self.open_listings(ids)
        cands = "\n".join(f"{n + 1}. {self.shelf.card(p)['title']} | shop: {self.maker_fields(p, r'shop|brand|seller', 120)}"
                          f" | facts: {squeeze(self.shelf.card(p)['facts'], 600)}" for n, p in enumerate(ids))
        target = f"{got.get('shop')} {got.get('kind')} {got.get('details') or ''}"
        pick = self.models.json(NOTE_PICK_PROMPT.format(target=target, prop=got["property"], cands=cands), STRONG, 250,
                                accept=lambda o: isinstance(o, dict)) or {}
        val = str(pick.get("value") or "").strip()
        if val and pid in self.shelf.full:
            facts = next((v for v in self.shelf.full[pid].values() if isinstance(v, dict)), None)
            if facts is not None:
                facts[str(got["property"])] = val
            self.marks.pop(pid, None)

    def commonest_category(self, pids: list) -> str:
        counts: dict = {}
        for p in pids:
            c = self.shelf.category(p)
            counts[c] = counts.get(c, 0) + 1
        return max(counts, key=counts.get) if counts else ""

    def promising_unopened(self, k: int) -> list:
        """Unopened rows whose text names a wanted value, in the category opened most so far."""
        top = self.commonest_category(list(self.shelf.full))
        same_cat = lambda p: not top or self.shelf.category(p) == top
        wanted = {w for r in self.needs for w in tokens(r.get("value")) if len(w) > 2}
        hits = lambda p: len(wanted & set(tokens(squeeze(self.shelf.seen[p], 1200))))
        rows = [p for p in self.shelf.seen if p not in self.shelf.full and p not in self.dead and hits(p) and same_cat(p)]
        if rows:
            return sorted(rows, key=hits, reverse=True)[:k]
        recent = [p for x in reversed(self.shelf.searches) for p in x["ids"]]    # the value may sit only in facts
        return [p for p in dict.fromkeys(recent) if p not in self.shelf.full and p not in self.dead and same_cat(p)][:k]

    def variant_sweep(self, pid: str) -> str:
        """Best-by-measure, once before ordering: variants of the pick (titles sharing three or more words) often
        differ in just that measure; open and check the unopened ones; a better confirmed fit, or ''."""
        if "variants" in self.once or self.rank() != "best":
            return ""
        self.once.add("variants")
        mine = set(tokens(self.shelf.card(pid).get("title")))
        room = max(0, min(3, self.steps_left() - 8))
        sibs = [p for p in self.shelf.seen if p not in self.shelf.full and p not in self.dead and p != pid
                and len(mine & set(tokens(self.shelf.card(p).get("title")))) >= 3][:room]
        self.open_listings(sibs)
        if sibs:
            self.judge(sibs)
        fits = self.ranked_fits()
        return fits[0] if fits and fits[0] != pid else ""

    def price_sweep(self, pid: str) -> str:
        """Cheapest, once before ordering: read further pages of the search that showed the pick (and of the value
        query), then open and check cheaper listings in price order; a cheaper confirmed fit, or ''."""
        if "late_price" in self.once or self.rank() != "cheapest":
            return ""
        self.once.add("late_price")
        price = self.shelf.price(pid)
        texted = [x for x in self.shelf.searches if self.shelf.text_param(x)]
        pool = [x for x in texted if pid in x["ids"]] or sorted(texted, key=lambda x: -(x["total"] or len(x["ids"])))
        if price is None or not pool:
            return ""
        last = pool[0]
        page, textarg, own = self.tools.page_param(last["name"]), self.shelf.text_param(last), self.value_query()
        starts = [last]
        if textarg and own and own != str(last["args"].get(textarg, "")).lower():
            starts.append({"name": last["name"], "args": {**self.fresh_args(last), textarg: own},
                           "next": 0 if page else None, "ids": [], "total": None})
        pages = 5
        for cur in starts:
            while pages > 0 and page and cur["next"] is not None and self.steps_left() >= 12 and self.timer.remaining() >= 90:
                cur = self.run_search(cur["name"], self.tools.with_page(cur["name"], cur["args"], page, cur["next"]))
                pages -= 1
                if not cur["ids"]:
                    break
        cat = self.shelf.category(pid)
        cheaper = lambda p: (self.shelf.price(p) or 10 ** 9) < price
        room = max(0, min(3, self.steps_left() - 8))
        cands = sorted((p for p in self.shelf.seen if p not in self.shelf.full and p not in self.dead and p != pid
                        and cheaper(p) and available(self.shelf.card(p)) and (not cat or self.shelf.category(p) == cat)),
                       key=lambda p: self.shelf.price(p) or 0)[:room]
        self.open_listings(cands)
        todo = [p for p in self.shelf.full if p not in self.marks and cheaper(p)]
        if todo:
            self.judge(todo[:8])
        best = [p for p in self.ranked_fits() if p != pid and cheaper(p)]
        return best[0] if best else ""

    def early_sweep(self) -> str:
        """Ranked single-item tasks, once, after the driver's first text search: page the value query, then open and
        check the cheapest (or the value query's own) unopened listings of the main category."""
        rank = self.rank()
        if "early" in self.once or rank not in ("cheapest", "best"):
            return ""
        texted = [x for x in self.shelf.searches if self.shelf.text_param(x)]
        if not texted or self.steps_left() < 18 or self.timer.remaining() < 120:
            return ""
        self.once.add("early")
        base = texted[0]
        page, ta, own = self.tools.page_param(base["name"]), self.shelf.text_param(base), self.value_query()
        if not (page and ta and own):
            return ""
        cur = {"name": base["name"], "args": {**self.fresh_args(base), ta: own}, "next": 0, "ids": []}
        for _ in range(4):
            if cur["next"] is None or self.steps_left() < 16:
                break
            cur = self.run_search(cur["name"], self.tools.with_page(cur["name"], cur["args"], page, cur["next"]))
            if not cur["ids"]:
                break
        cat = self.commonest_category([p for x in self.shelf.searches for p in x["ids"]])
        pool = [p for p in self.shelf.seen if p not in self.shelf.full and p not in self.dead
                and available(self.shelf.card(p)) and self.within_budget(self.shelf.card(p))
                and (not cat or self.shelf.category(p) == cat)]
        if rank == "cheapest":
            cands = sorted(pool, key=lambda p: self.shelf.price(p) or 0)[:3]
        else:                                     # the measure sits in the facts: the value query's rows, store order
            mine = [p for x in self.shelf.searches if x["args"].get(ta) == own for p in x["ids"]]
            cands = [p for p in dict.fromkeys(mine) if p in pool][:4]
        self.open_listings(cands)
        todo = [p for p in self.shelf.full if p not in self.marks]
        if todo:
            self.judge(todo[:8])
        if rank == "best":
            return (f"[system sweep] read '{own}' page by page and checked more listings. Confirmed fits, best first by "
                    f"{self.reading.get('measure') or 'the measure'}: {self.ranked_fits()[:4]}. Before ordering, leave no "
                    f"listing unchecked that might do better.")
        return (f"[system price sweep] read '{own}' page by page and checked the cheapest listings. Confirmed fits, "
                f"cheapest first: {[(f, self.shelf.price(f)) for f in self.ranked_fits()[:4]]}. Before ordering, leave no "
                f"cheaper listing unchecked that might fit.")


# ================================================================ gates before cart, order and decline
class Gates:
    def tools_now(self) -> list:
        if self.steps_left() > RESERVE:
            return self.tools.offer
        keep = {self.tools.roles.get(r) for r in ("cart", "cart_view", "details", "order", "decline")}
        return [t for t in self.tools.offer if t["function"]["name"] in keep] or self.tools.offer

    def screen(self, name: str, args: dict) -> str:
        """Why a driver's call should not run (no step spent), or '' to run it."""
        if name not in self.tools.specs:
            return f"'{name}' is not a tool; pick one of {sorted(self.tools.specs)}."
        if name not in {t["function"]["name"] for t in self.tools_now()}:
            return "Too few steps remain for that; only the cart, details and finishing tools are open now."
        if name == self.tools.roles.get("message"):
            return "The system plans and sends shopper questions itself; carry on with the store."
        intent = self.tools.intent(name, args)
        key = name + json.dumps(args, sort_keys=True)
        if key in self.refused:
            return "The store already refused this exact call; change something."
        if intent not in ("order", "decline", "cart_add", "cart_remove") and key in self.last_calls:
            return "You just made this same call and its result is above; try something else."
        page = self.tools.page_param(name)
        if page in args and (self.tools.params(name).get(page) or {}).get("type") == "string" \
                and args[page] not in self.shelf.cursors:
            return (f"'{page}' takes only a next-page value the store itself returned (so far: "
                    f"{sorted(self.shelf.cursors)[-3:] or 'none'}); when a result gives no next page there is none - narrow "
                    f"the query or change the shelf instead.")
        if intent != "cart_add":
            return ""
        flat = list(args.values()) + [v for x in args.values() if isinstance(x, list) for z in x if isinstance(z, dict)
                                      for v in z.values()]
        ids = [v for v in flat if isinstance(v, str) and self.shelf.knows(v)]
        if self.is_basket():                      # judged for its own line; lines already carted pass
            ids = [v for v in ids if v not in {ln.pick for ln in self.lines}] or ids[:1]
            if ids:
                self.switch_to(self.line_of(ids[0]))
        pid = ids[0] if ids else None
        if pid and pid not in self.shelf.full and pid not in self.warned:
            self.warned.add(pid)
            return f"Open {pid}'s details first so its facts can be checked."
        if pid and pid in self.shelf.full:
            if pid not in self.marks:
                self.judge([pid])
            mark = self.marks.get(pid)
            if mark == "N":
                return f"Cart check: {pid} misses a requirement. Take a confirmed fit instead: {self.ranked_fits()[:3]}."
            if mark == "?" and pid not in self.warned:
                self.warned.add(pid)
                return (f"Cart check: {pid} does not clearly meet every requirement. Prefer a confirmed fit "
                        f"{self.ranked_fits()[:3]}, or repeat the same call if you are certain.")
        return ""

    def order_gate(self) -> str:
        """The cart holds exactly the chosen fit at the right units (a basket: every line, within the budget)."""
        if self.is_basket() and self.steps_left() > 3 and self.timer.remaining() >= 40:
            self.order_cleared = False            # a basket is checked on every order attempt
        if self.order_cleared or self.steps_left() <= 2 or self.timer.remaining() < 60:
            return ""
        self.order_cleared = True
        held = {p: q for p, q in self.cart.items() if q}
        if self.is_basket():
            self.settle_lines()
            missing = [f"{ln.item} x{ln.units}" for ln in self.lines if not ln.pick]
            if missing:
                return (f"Order check: the basket still has no confirmed fit for {missing}. Find, check and cart those as "
                        f"well (the cart keeps the lines already chosen), then order once.")
            total = sum((self.shelf.price(p) or 0) * q for p, q in held.items())
            if self.cap and total > self.cap + 1e-9:
                return f"Order check: the basket total {total} is above the budget {self.cap}; pick cheaper fits."
            return ""
        if len(held) != 1:
            return f"Order check: the cart has to hold exactly one line; it holds {held}. Fix the cart, then order."
        pid, q = next(iter(held.items()))
        if q != self.units:
            return f"Order check: {pid} is carted {q} time(s) but the shopper wants {self.units}. Fix that, then order."
        if pid in self.shelf.full and pid not in self.marks:
            self.judge([pid])
        if self.marks.get(pid) in ("N", "?"):
            return (f"Order check: {pid} does not clearly meet every requirement ({self.marks.get(pid)}). Confirmed fits: "
                    f"{self.ranked_fits()[:3]}.")
        if not self.within_budget(self.shelf.card(pid)):
            return (f"Order check: {pid} now costs {self.shelf.price(pid)} x {self.units}, above the budget {self.cap}. "
                    f"Confirmed fits within the budget: {self.ranked_fits()[:3]}.")
        better = self.variant_sweep(pid)
        if better:
            self.order_cleared = False
            return (f"Order check: {better} is a confirmed fit that does better on {self.reading.get('measure') or 'the measure'} "
                    f"than {pid}. Put it in the cart instead, then order.")
        cheaper = self.price_sweep(pid)
        if cheaper:
            self.order_cleared = False
            return (f"Order check: a cheaper confirmed fit exists: {cheaper} ({self.shelf.price(cheaper)} vs "
                    f"{self.shelf.price(pid)}). Put it in the cart in place of {pid}, then order.")
        return ""

    def decline_gate(self) -> str:
        """At most two audits: a confirmed fit, an unchecked opened listing, or an unopened row naming wanted values may
        still qualify."""
        if self.steps_left() <= 5 or self.timer.remaining() < 70:
            return ""
        if self.is_basket():                      # a basket is declined only after every line was looked for
            self.lines[self.at].tried = True
            nxt = next((i for i, ln in enumerate(self.lines) if not ln.pick and not ln.tried), None)
            if nxt is not None and self.steps_left() > 8:
                self.switch_to(nxt)
                return (f"Decline held: basket line '{self.lines[nxt].item}' has not been searched yet. Work on it now "
                        f"(the line you could not fill stays open); decline only if a line truly has no fit.")
            guessed = self.fill_open_lines()      # a doubtful line can still pass; a decline never does
            if guessed:
                return f"Decline held: every basket line now has its closest match in the cart ({guessed}). Order the basket now."
        fits = self.ranked_fits()
        if fits:                                  # the fact check reads more surely than the driver
            return (f"Decline held: the fact check confirmed {fits[:3]} meet every requirement (a wanted value inside a "
                    f"longer listing value counts, e.g. 'coil' in '3 single coil'). Order the first; decline only if the "
                    f"store refuses it and nothing else fits.")
        if self.decline_reviews >= 2:
            return ""
        self.decline_reviews += 1
        unchecked = [p for p in self.shelf.full if p not in self.marks][:6]
        if unchecked:
            self.judge(unchecked)
        fits = self.ranked_fits()
        if fits:
            return (f"Review before declining: these listings look like they meet every requirement: {fits[:4]}. Check "
                    f"again and order one, or decline if they truly fail.")
        room = self.steps_left() - 6              # opening, then cart and order, still fit
        if self.decline_reviews == 1 and room >= 1:
            rows = self.promising_unopened(min(4, room))
            if rows:
                return (f"Review before declining: these listings from your searches were never opened and their text "
                        f"mentions what is wanted: {[(p, str(self.shelf.card(p)['title'])[:70]) for p in rows]}. Open "
                        f"their details so their facts can be checked, then order a fit or decline.")
        return ""

    def decoy(self, order_tool: str, order_args: dict) -> Any:
        """The first order goes with only a confirmed runner-up in the cart, so check-out events land on it. Returns
        what the driver is shown, or None when the decoy does not apply (the driver's own order then runs)."""
        if self.decoy_used or self.ranks() or self.is_basket() or self.steps_left() < 7 or self.timer.remaining() < 60:
            return None
        self.decoy_used = True
        held = {p: q for p, q in self.cart.items() if q}
        if len(held) != 1:
            return None
        best = next(iter(held))
        spares = lambda: [p for p in self.ranked_fits() if p != best and self.marks.get(p) == "Y"]
        options = spares()
        if not options:
            unchecked = [p for p in self.shelf.full if p not in self.marks and p != best][:4]
            if unchecked:
                self.judge(unchecked)
                options = spares()
        if not options:
            return None
        runner = options[0]
        for intent, pid, qty in (("cart_remove", best, held[best]), ("cart_add", runner, self.units)):
            tool, args = self.tools.build(intent, {"id": pid, "qty": qty})
            if not tool:
                return None
            self.act(tool, args)
            self.cart[pid] = 0 if intent == "cart_remove" else qty
        obs = self.act(order_tool, order_args)
        if self.done:
            return f"[decoy] Ordered the confirmed runner-up {runner}; the order went through. {squeeze(obs, 300)}"
        texts = self.shelf.absorb(obs)
        heard = self.take_news(" ".join(texts) or squeeze(obs, 600), "store refusal") if texts else ""
        heard += self.refresh_fits()
        self.dead.add(runner)                     # whatever the refusal says, the decoy has done its job
        log = [f"[decoy] The first order carried the runner-up {runner} and was refused: {heard or squeeze(obs, 400)}"]
        for intent, pid, qty in (("cart_remove", runner, self.units), ("cart_add", best, self.units)):
            if intent == "cart_add" and (best in self.dead or self.marks.get(best) not in ("Y", "T")
                                         or not self.within_budget(self.shelf.card(best))):
                log.append(f"{best} no longer fits after the changes; choose again from {self.ranked_fits()[:3]}.")
                break
            tool, args = self.tools.build(intent, {"id": pid, "qty": qty})
            if tool:
                self.act(tool, args)
                self.cart[pid] = 0 if intent == "cart_remove" else qty
        else:
            log.append(f"The cart holds {best} again; read its stock and price again, then order.")
        self.order_cleared = False
        return "\n".join(log)


# ================================================================ the episode
class Episode(Judging, Asking, Basket, Reading, Gates):
    def __init__(self, problem: dict) -> None:
        env = problem.get("environment") or {}
        view = env.get("policy_view") or {}
        self.binding = env.get("binding") or {}
        self.request = str(view.get("query") or problem.get("query") or "")
        self.opening = view.get("opening")
        self.max_steps = int(view.get("max_steps") or 30)
        self.tools = Toolbox([t for t in view.get("tools") or [] if isinstance(t, dict)])
        self.shelf = Shelf(self.tools)
        self.timer = Timer()
        self.models = Models(self.timer)
        self.step = 0
        self.done = False
        self.reading: dict = {}
        self.item = ""
        self.needs: list = []
        self.cap = 0.0
        self.units = 1
        self.lines: list = []          # a basket: one Line per product
        self.at = 0                    # the basket line being worked on
        self.msgs_left = 0
        self.asked_props: list = []
        self.heard: list = []
        self.last_news: dict = {}
        self.asking_done = False
        self.empty_plans = 0
        self.marks: dict = {}          # listing id -> Y | T | N | ?
        self.literal_hits: dict = {}   # listing id -> requirements its facts state in the shopper's words
        self.measured: dict = {}
        self.measure_dir = "max"
        self.cart: dict = {}           # listing id -> units
        self.dead: set = set()         # listings sold out or used up by the decoy
        self.warned: set = set()
        self.refused: set = set()
        self.last_calls: list = []
        self.notes_followed: set = set()
        self.once: set = set()
        self.decoy_used = False
        self.order_cleared = False
        self.decline_reviews = 0
        self.recheck = False           # requirements changed: opened listings are checked again
        self.was_fit: set = set()
        self.chat: list = []
        self.trace: list = []

    # ------------------------------------------------ the environment
    def act(self, name: str, args: dict) -> Any:
        if self.done or self.step >= self.max_steps:
            return None
        step = self.step + 1
        for _ in range(3):
            res = _env.post_verbose("/environment/call", json_data={**self.binding, "step": step, "action": {"name": name, "args": args}})
            if res.ok:
                data = res.data or {}
                self.step = int(data.get("step") or step)
                self.done = bool(data.get("finished"))
                obs = data.get("observation")
                self.trace.append({"step": self.step, "tool": name, "args": args, "obs": squeeze(obs, 400)})
                return obs
            body = str(res.error or "")
            m = re.search(r"expected (\d+)", body)
            if m and int(m.group(1)) != step:     # out of step with the store: take its count
                step = int(m.group(1))
                self.step = step - 1
                continue
            if "finished" in body:
                self.done = True
                return None
            time.sleep(0.5)
        return None

    def steps_left(self) -> int:
        return self.max_steps - self.step

    # ------------------------------------------------ the request
    def number_in(self, value: Any, words_: Any, source: Any = None) -> float:
        """A figure the model read, kept only when the text really says it."""
        said = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", str(self.request if source is None else source).replace(",", ""))]
        try:
            v = float(value or 0)
        except (TypeError, ValueError):
            v = 0.0
        if v > 0 and v in said:
            return v
        for x in re.findall(r"\d+(?:\.\d+)?", str(words_ or "").replace(",", "")):
            if float(x) > 0 and float(x) in said:
                return float(x)
        return 0.0

    def read_request(self) -> None:
        listing = "\n".join(squeeze({"name": n, "description": s.get("description", ""), "parameters": s.get("parameters", {})}, 1600)
                            for n, s in self.tools.specs.items())
        got = self.models.json(READ_PROMPT.replace("{request}", self.request).replace("{tools}", listing), READER, 1500,
                               accept=lambda o: isinstance(o, dict) and "product" in o) or {}
        self.reading = got
        self.tools.adopt(got.get("tools"))
        self.item = str(got.get("product") or "")
        stated = lambda r: isinstance(r, dict) and r.get("property") and r.get("value") \
            and not NON_VALUE.fullmatch(str(r["value"]).strip().lower())      # "toe style: unspecified" binds nothing
        self.needs = [r for r in got.get("requirements") or [] if stated(r)
                      and not re.search(r"categor|department|budget|price|stock", str(r["property"]).lower())]
        self.cap = self.number_in(got.get("budget"), got.get("budget_text"))
        units = lambda x: max(1, min(int(x or 1), 50))
        try:
            self.units = units(got.get("units"))
        except (TypeError, ValueError):
            self.units = 1
        self.lines = []
        for entry in got.get("basket") or []:
            if not isinstance(entry, dict) or not entry.get("product"):
                continue
            try:
                n = units(entry.get("units"))
            except (TypeError, ValueError):
                n = 1
            needs = [r for r in entry.get("requirements") or [] if stated(r)]
            self.lines.append(Line(item=str(entry["product"]), units=n, needs=needs))
        if len(self.lines) < 2:
            self.lines = []
        else:
            self.at = 0
            self.needs, self.units, self.item = self.lines[0].needs, self.lines[0].units, self.lines[0].item
        stated = int(self.number_in(got.get("step_limit"), None))   # kept only when the request says that figure
        if 10 <= stated < self.max_steps:
            self.max_steps = stated
        try:
            limit = int(got.get("message_limit"))
        except (TypeError, ValueError):
            limit = -1
        self.msgs_left = (limit if limit >= 0 else 2) if self.tools.roles.get("message") else 0

    def priority_note(self) -> str:
        if self.ranks():
            return f"{self.reading.get('rank')} {self.reading.get('measure') or ''}".strip()
        return "none (any confirmed fit within the budget is right; a lower price earns nothing)"

    def status(self) -> str:
        fits = self.ranked_fits()
        text = (f"[status] steps left {self.steps_left()} of {self.max_steps}; seconds left {int(self.timer.remaining())}; "
                f"messages left {self.msgs_left}; budget {self.cap or 'none'}; units {self.units}; requirements "
                f"{[self.need_text(r) for r in self.needs]}; priority {self.priority_note()}; cart {self.cart or 'empty'}; "
                f"confirmed fits (best first) {fits[:4]}{self.basket_note()}")
        if not fits and RESERVE + 2 < self.steps_left() <= 16:
            rows = self.promising_unopened(3)
            if rows:
                text += f"; nothing confirmed yet - open these unopened listings before searching again: {rows}"
        return text

    def say(self, text: str) -> None:
        self.chat.append({"role": "user", "content": text + "\n" + self.status()})

    # ------------------------------------------------ the driver's calls
    def track_cart(self, intent: str, args: dict, obs: Any) -> None:
        ok = not errored(obs)
        setter = next((v for v in args.values() if isinstance(v, list) and all(isinstance(z, dict) for z in v)), None)
        pid = next((v for v in args.values() if isinstance(v, str) and self.shelf.knows(v)), None)
        qty = next((v for v in args.values() if isinstance(v, int) and not isinstance(v, bool)), 1)
        if setter is not None and ok:             # the lines given replace the cart
            self.cart = {}
            for z in setter:
                zid = next((v for v in z.values() if isinstance(v, str) and self.shelf.knows(v)), None)
                if zid:
                    self.cart[zid] = next((v for v in z.values() if isinstance(v, int) and not isinstance(v, bool)), 1)
        elif pid and ok:
            have = self.cart.get(pid, 0)
            self.cart[pid] = have + qty if intent == "cart_add" else max(0, have - qty)
        self.order_cleared = False
        if self.is_basket() and ok:
            self.settle_lines()

    def recheck_prices(self) -> str:
        """Cheapest, after an order refusal: prices may have moved; read them again and look for a backup."""
        held = [p for p, q in self.cart.items() if q]
        if self.last_news.get("unavailable"):     # sold out: that pick is gone; a price change is not
            self.dead.update(held)
        for p in [x for x in held if x not in self.dead] + [f for f in self.ranked_fits() if f not in held][:2]:
            if self.steps_left() > 6:
                tool, args = self.tools.build("details", {"id": p})
                if tool:
                    self.shelf.absorb(self.act(tool, args))
        cheaper = sorted((p for p in self.shelf.full if p not in self.marks and p not in self.dead),
                         key=lambda p: self.shelf.price(p) or 0)[:4]
        if cheaper:
            self.judge(cheaper)
        if len(self.ranked_fits()) < 2:           # no backup left: the cheapest unopened rows within the budget
            cats = {self.shelf.category(p) for p in held}
            room = max(0, min(3, self.steps_left() - 7))
            more = sorted((p for p in self.shelf.seen if p not in self.shelf.full and p not in self.dead
                           and self.within_budget(self.shelf.card(p)) and available(self.shelf.card(p))
                           and (not cats or self.shelf.category(p) in cats)), key=lambda p: self.shelf.price(p) or 0)[:room]
            self.open_listings(more)
            if more:
                self.judge(more)
        return (f"\nPrices read again after the refusal. Confirmed fits now, cheapest first, at current prices: "
                f"{[(f, self.shelf.price(f)) for f in self.ranked_fits()[:4]]}; order the first unless the refusal rules it out.")

    def execute(self, calls: list) -> list:
        replies = []
        for c in calls:
            cid = c.get("id") or f"call_{len(self.trace)}"
            fn = c.get("function") or {}
            name, raw = fn.get("name"), fn.get("arguments")
            args = raw if isinstance(raw, dict) else (read_json(raw or "{}") if str(raw or "").strip() else {})
            skip = lambda why, held=False: replies.append((cid, ("Held back, no step spent: " if held else
                                                                  "Skipped, no step spent: ") + why))
            if not isinstance(args, dict):
                skip("the arguments were not a JSON object.")
                continue
            if name not in self.tools.specs:
                skip(f"'{name}' is not a tool.")
                continue
            args, problems = conform(args, self.tools.params(name))
            intent = self.tools.intent(name, args)
            args = self.tools.fill_required(name, intent, args)
            missing = [a for a in self.tools.needed(name) if a not in args]
            if problems or missing:
                skip("; ".join(problems + [f"missing required {m}" for m in missing]) + ".")
                continue
            if intent in ("order", "decline") and len(calls) > 1:
                skip("an order or a decline has to be sent alone.")
                continue
            why = self.screen(name, args)
            if why:
                skip(why)
                continue
            if intent == "order":
                why = self.order_gate()
                if why:
                    skip(why, held=True)
                    continue
                shown = self.decoy(name, args)
                if shown is not None:
                    replies.append((cid, shown))
                    continue
            if intent == "decline":
                why = self.decline_gate()
                if why:
                    skip(why, held=True)
                    continue
                if self.reading.get("quiet_decline"):
                    for a, v in list(args.items()):
                        if isinstance(v, str) and not self.tools.params(name).get(a, {}).get("enum"):
                            args[a] = NOTHING_FITS
            if self.is_basket() and intent in ("cart_add", "cart_remove"):
                args = self.carry_lines(name, args)
            obs = self.act(name, args)
            texts = self.shelf.absorb(obs)        # rows first, so the search log sees this result's listings
            if intent in ("search", "filter"):
                self.shelf.log_search(name, args, obs)
            key = name + json.dumps(args, sort_keys=True)
            if isinstance(obs, dict) and any("error" in str(k).lower() for k in obs):
                self.refused.add(key)
            self.last_calls = (self.last_calls + [key])[-3:]
            if intent in ("cart_add", "cart_remove"):
                self.track_cart(intent, args, obs)
            note = ""
            if texts:
                note = self.take_news(" ".join(texts), "store refusal" if intent == "order" else "store notice")
                if intent == "order" and self.rank() == "cheapest":
                    note += self.recheck_prices()
                note += self.refresh_fits()
            text = squeeze(obs, SHOW) + (" [the episode is over]" if self.done else "")
            replies.append((cid, text + ("\n" + note if note else "")))
        return replies

    def wrap_up(self) -> None:
        """Out of steps, time or patience: cart what can be carted and order, else decline."""
        if self.done or self.steps_left() <= 0:
            return
        if self.is_basket() and any(not ln.pick for ln in self.lines) and self.steps_left() >= 3:
            self.settle_lines()
            self.fill_open_lines()                # order the basket with its closest matches
        fits = self.ranked_fits()
        can_order = bool(self.tools.roles.get("order"))
        if not any(self.cart.values()) and fits and self.steps_left() >= 2 and can_order:
            tool, args = self.tools.build("cart_add", {"id": fits[0], "qty": self.units})
            if tool:
                self.act(tool, args)
                self.cart[fits[0]] = self.units
        if any(self.cart.values()) and can_order:
            tool, args = self.tools.build("order", {})
        else:
            tool, args = self.tools.build("decline", {"text": NOTHING_FITS})
        if not tool:
            return
        self.act(tool, args)
        if not self.done and self.steps_left() > 0 and self.tools.roles.get("decline"):
            tool, args = self.tools.build("decline", {"text": NOTHING_FITS})
            if tool:
                self.act(tool, args)

    # ------------------------------------------------ the loop
    def commits(self, calls: list) -> bool:
        for c in calls:
            fn = c.get("function") or {}
            raw = fn.get("arguments")
            args = raw if isinstance(raw, dict) else read_json(raw or "{}")
            if self.tools.intent(fn.get("name"), args if isinstance(args, dict) else {}) in ("cart_add", "order", "decline"):
                return True
        return False

    def run(self) -> None:
        self.read_request()
        if self.opening is not None:
            self.shelf.absorb(self.opening)
        brief = (f"SHOPPER REQUEST:\n{self.request}\n\nHOW THE SYSTEM READ IT (check it against the request): product "
                 f"{self.item}; requirements {[self.need_text(r) for r in self.needs]}; budget {self.cap or 'none'}; units "
                 f"{self.units}; priority {self.reading.get('rank') or 'none'} {self.reading.get('measure') or ''}; rules "
                 f"{self.reading.get('rules') or []}; suggested searches {self.reading.get('searches') or []}; tool roles "
                 f"{self.tools.roles}.")
        if self.opening is not None:
            brief += "\n\nFIRST VIEW OF THE STORE:\n" + squeeze(self.opening, 2000)
        self.chat = [{"role": "system", "content": DRIVER_BRIEF}, {"role": "user", "content": brief + "\n\n" + self.status()}]
        idle = failures = 0
        for _ in range(TURNS):
            if self.done or self.steps_left() <= 0:
                break
            if self.timer.remaining() < WRAP_UP_AT:
                self.wrap_up()
                break
            learned = self.question_round()
            if learned:
                self.say("[questions sent by the system]\n" + learned)
            extra = self.resolve_pointers() or (self.early_sweep() if not self.is_basket()
                                                else self.look_ahead() or self.close_spent_line() or self.seed_line())
            if extra:
                self.say(extra)
            msg = self.models.ask(self.chat, FAST, 2500, tools=self.tools_now())
            if msg is None:
                failures += 1
                if failures >= 2:
                    self.wrap_up()
                    break
                continue
            calls = [c for c in msg.get("tool_calls") or [] if isinstance(c, dict)]
            if self.commits(calls):
                learned = self.question_round(force=True)      # ask before committing to anything
                if learned:
                    self.say("[questions sent by the system before you commit]\n" + learned)
                    continue
            if not calls:
                idle += 1
                self.chat.append({"role": "assistant", "content": msg.get("content") or "(no tool call)"})
                self.say("Each reply has to call a tool. Go on.")
                if idle >= 3:
                    self.wrap_up()
                    break
                continue
            idle = 0
            self.chat.append({"role": "assistant", "content": msg.get("content") or None,
                              "tool_calls": [{"id": c.get("id") or f"call_{i}", "type": "function", "function": c.get("function")}
                                             for i, c in enumerate(calls)]})
            replies = self.execute(calls)
            for i, (cid, text) in enumerate(replies):
                tail = "\n" + self.status() if i == len(replies) - 1 else ""
                self.chat.append({"role": "tool", "tool_call_id": cid, "content": text + tail})
            fresh = [p for p in list(self.shelf.full)[-6:] if p not in self.marks]
            if len(fresh) >= 2 and self.steps_left() > 3:
                self.judge(fresh)                 # keep the confirmed-fits list current for the driver
        if not self.done:
            self.wrap_up()
        self.trace.append({"verdicts": self.marks, "requirements": self.needs, "budget": self.cap, "probed": self.decoy_used,
                           "calls": len(self.models.log), "failed_calls": sum(1 for c in self.models.log if not c["ok"])})


def agent_main(problem_data: dict) -> list:
    episode = Episode(problem_data)
    try:
        episode.run()
    except Exception as exc:  # never leave the episode open without trying to finish it
        episode.trace.append({"error": f"{type(exc).__name__}: {exc}"})
        try:
            episode.wrap_up()
        except Exception:
            pass
    return episode.trace
