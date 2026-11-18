from __future__ import annotations

import json
import random
import re
import threading
import time
from typing import Any

from src.agent.proxy_client import ProxyClient

LIGHT = "anthropic/claude-haiku-5.5"
HEAVY = "anthropic/claude-sonnet-5.5"
SPARE = "openai/gpt-6-luna"

SEATS: dict[str, list[str]] = {
    "toolmap": [LIGHT, HEAVY],
    "reader": [HEAVY, LIGHT],
    "shape": [LIGHT, HEAVY],
    "screen": [LIGHT, SPARE],
    "planner": [HEAVY, LIGHT],
    "listener": [LIGHT, HEAVY],
    "listener_strict": [HEAVY, LIGHT],
    "linker": [HEAVY, LIGHT],
    "pointer": [LIGHT, HEAVY],
    "seeker": [LIGHT, SPARE],
    "judge": [LIGHT, HEAVY],
    "checker": [HEAVY, LIGHT],
    "notes": [LIGHT, SPARE],
    "units": [LIGHT, HEAVY],
}
OUTPUT_TOKENS: dict[str, int] = {
    "toolmap": 1500, "reader": 3000, "shape": 700, "screen": 900, "planner": 1600, "listener": 600, "listener_strict": 700,
    "pointer": 1600, "linker": 1800, "seeker": 500, "judge": 2600, "checker": 1800, "notes": 700, "units": 600,
}
HEAVY_SEATS = {"reader", "planner", "checker", "linker"}

TIME_LIMIT_SECONDS = 300
WRAP_UP_SECONDS = 240
WAIT_STEPS = (3, 8, 16)
WAIT_BUDGET_SECONDS = 40
LLM_CALLS_CAP = 40
SPEND_SOFT = 0.05
SPEND_HARD = 0.07

DEFAULT_MESSAGES = 5
CLOSE_RESERVE = 9
SURVEY_OPENS = 6
SURVEY_OPENS_EACH = 3
HUNT_OPENS = 8
HUNT_OPENS_EACH = 4
HUNT_QUERIES = 3
POINTER_HOPS = 6
CLOSE_ROUNDS = 3
RECORD_CHARS = 1400
ROW_CHARS = 260

_llm = ProxyClient(timeout=50, max_retries=1, retry_delay=1)
_env = ProxyClient(timeout=95, max_retries=3, retry_delay=1)

ROLE = "You help a shopping agent that works for one shopper inside an online store simulation."

MATCH_RULES = """- A store property speaks to a requirement when its key names the requirement's property (spacing, underscores, case and plural do not matter).
  - The value meets it when it carries the wanted value in plain words; case, plural, obvious synonyms and small misspellings do not matter, one entry of a list is enough, and a longer value that contains the wanted words still counts ("modern classic" meets modern).
  - A different word is a different value; an abbreviation is not the word it shortens.
  - Convert units before comparing numbers; "at most" and "at least" include the edge.
- A property about another part of the product (a strap's colour for a case's colour) says nothing.
- When no property key names it, the value written in the title or description is only "likely"; a bare number with no unit or name says nothing.
- Nothing in the record about a requirement means "unknown", never "against"."""

TOOLMAP_GUIDE = ROLE + """
# Name the store's tools by job
You get the store's tools (name, description, parameters). Return ONE JSON object and nothing else:
{"search": {"tool": "...", "query": "the free-text parameter for search words", "page": "the parameter that asks for a further page (token, cursor or offset) or null", "price_max": "a parameter capping the price, or null", "category": "a parameter taking a category or department label, or null", "browse": {fixed parameters that make the tool list a category instead of matching words, or {} when the category works with words}, "fixed": {parameter: value} for required parameters with one sensible value (a mode meaning plain text search), else {}},
 "details": {"tool": "...", "id": "the listing id parameter"} for the tool that shows one listing's properties, attributes or specifications,
 "page": {"tool": "...", "id": "..."} or null, for another single-listing tool (offer, price, stock, remarks, description),
 "many": {"tool": "...", "ids": "the list parameter"} or null, for a tool that shows several listings at once,
 "message": {"tool": "...", "text": "..."} for writing to the shopper,
 "cart": {"tool": "...", "id": "...", "qty": "... or null", "verb": "the parameter choosing add or remove, or null", "add": "its value that adds", "remove": "its value that removes", "lines": "a parameter taking a whole list of lines, or null", "line_id": "id key inside one line or null", "line_qty": "quantity key inside one line or null"},
 "remove": {"tool": "...", "id": "..."} or null, only when removing uses its own tool,
 "cart_view": {"tool": "..."} or null,
 "buy": {"tool": "...", "args": {fixed arguments that make it place the order}},
 "refuse": {"tool": "...", "args": {fixed arguments that make it end without an order}}}
Use exact names from the schemas. One tool may do two jobs with different fixed arguments."""

READER_GUIDE = ROLE + """
# Read the shopper's request
Return ONE JSON object and nothing else:
{"products": [{"name": "the product in the request's words", "noun": "two or three words naming it plainly", "units": integer (1 when unstated),
   "musts": [{"property": "the property in plain words", "want": "the exact value, limit, exclusion or closed list, word for word", "points": true when the value must be read from another listing}],
   "queries": ["2 or 3 short store searches for it, product noun first"]}],
 "budget": total limit as a number in the request's own unit or null, "budget_scope": "order" or "product" or null,
 "pick": "cheapest" | "most" | "least" | "any", "measure": "for most or least, else \\"\\"",
 "fits": [{"part": product index from 0, "host": product index, "how": "the request's words on how they must fit"}],
 "ifs": [{"when": "the condition on one product, in the request's words", "then": "what another product must then be, in the request's words", "product": index of the product the condition is about, "other": index of the product that must change}],
 "messages": the number of messages the shopper allows or null, "steps": the number of steps allowed or null,
 "when_nothing_fits": "decline" | "closest" | "",
 "rules": ["any other instruction on how to work or finish, at most 20 words each"]}
- One entry per distinct product. Things joined in one phrase with one quantity are one product.
- A must is firm: a stated value, limit, exclusion or closed list. Price, budget and stock are never musts. Story and wishes are not musts. A must never just restates what the product is (its name or kind).
- Name a must's property with the request's own word for it ("pack type" when the request says pack type); keep the value as written.
- A must whose value comes from another listing keeps that whole description in "want" ("same as the colour of the kettle sold by X") and has "points": true; its "property" is the property of this product that the sentence is about (the occasion, the colour), never a word like "type" or "value".
- A request to buy the lowest price or total gives pick "cheapest".
- A rule that ties two products (the second product's value depends on what the first one has) goes in "ifs", and a rule that one product must suit, fit or work with another goes in "fits"; neither is a must."""

SHAPE_GUIDE = """A store search tool returned the result below. Name its parts by their exact keys. Return ONE JSON object and nothing else:
{"rows": the key of the list of listings, "id": the key of a listing's id inside a row, "price": the key of the price inside a row or null, "stock": the key saying whether it can be bought now, or null, "title": the key of the listing's name, "category": the key of the listing's category or department label or null, "count": the key of the total number of matches or null, "next": the key holding the value that asks for the next page or null}"""

SCREEN_GUIDE = ROLE + """
# Which search rows are the product wanted
For each product below, list the ids of rows that could be that product. Drop only rows that are clearly something else (an accessory, a part, or an unrelated item that merely shares a word); when a row might be the product, keep it. The shopper's wants can widen what the product is (a wanted form or format can make rows the product even when their name differs from the product's name); keep rows that match them.
Return ONE JSON object and nothing else: {"keep": {"product index": [ids, most likely first]}}"""

PLANNER_GUIDE = ROLE + """
# Choose what to ask the shopper
The shopper holds a few firm wishes they never wrote down, each on one property of one product. A question finds one only when it names that exact property, and only the first property a question names is answered, so each question spends one message on one property.
Work in two steps.
1. Hints. Read the request sentence by sentence. List every sentence that points at a wish without giving its value: where or how the product will be used, stored, carried or placed, who will use it, what it must go with, something the shopper is unsure about, asks you to ask about, or leaves to you. Name the one property each hint is about; use a listed key when one covers it, even if the listed listings show only one value for it (they are a small sample).
2. Ask. Put the hinted properties first, in the order of the hints. Fill the rest with listed keys that most listings carry and whose values differ; keys seen once or with one value come after those.
Leave out what the request settles, ids, keys listing purchase variants or options, warranty, delivery and shop logistics.
Copy listed keys exactly as shown. Also, for each settled requirement whose property is clearly the same thing as a listed key under another name, give that key.
Return ONE JSON object and nothing else: {"hints": [{"sentence": "a few words of it", "property": "..."}], "ask": {"product index": ["key", ...]}, "same": [{"product": index, "property": "the requirement's property", "key": "the listed key"}]} with up to LIMIT keys per product."""

LISTENER_GUIDE = ROLE + """
# Read one reply
The agent asked the shopper about one property of one product. Return ONE JSON object and nothing else:
{"need": {"property": "the property the reply names", "want": "the wish stated plainly, value or limit with units", "quote": "the reply's words that carry it", "points": true when the value is to be read from another listing} or null,
 "no_need": true when the reply gives no wish on it (does not mind, leaves it to the agent, does not follow, asks to hear it again or one at a time),
 "drop": ["a property the shopper gives up"], "budget": a new total limit as a number or null}
- "want" states the wish plainly as it is meant: "suede" for "I will not give up suede"; "anything other than lagoon" for "keep it off lagoon"; "at least 3 shelves"; keep relation words (not, other than, at least, at most, same as) only when they are part of the wish itself.
- "quote" copies the reply's own words that carry the wish, word for word.
- Folksy wording that names a value still names it. A value given by pointing at another listing keeps the whole description of that listing in "want"."""

POINTER_GUIDE = ROLE + """
# Find the listing a requirement points to
A requirement takes its value from another listing. From its words and the records below, say where that listing is.
Return ONE JSON object and nothing else:
{"seller": "the shop or seller named, exact spelling, or \\"\\"", "kind": "the kind of product, 1 to 3 words", "clues": ["other words that pick it out"], "property": "the property whose value is wanted", "found": "the id of a record below that is that listing, or \\"\\"", "value": "that listing's value for the property when a record below shows it, else \\"\\"", "next": "when that record sends the reader to yet another listing for this property, that listing's description, else \\"\\"", "rewrite": "when the value is known, the requirement restated with that value in place of the description of the other listing, keeping words such as not, other than, same as, at most; else \\"\\""}
A record is that listing only when its own words carry every clue (seller and the properties named)."""

SEEKER_GUIDE = ROLE + """
# Write store searches
Write short searches (2 to 5 words, product noun included) that should surface listings meeting the needs below; use the wanted values as words. Avoid repeating the searches already done.
Return ONE JSON object and nothing else: {"queries": ["...", "..."]}"""

JUDGE_GUIDE = ROLE + """
# Does each listing meet each requirement
Reading rules:
""" + MATCH_RULES + """
For every listing and every requirement give one verdict: "met" (a property says so), "likely" (only the title or description says so), "against" or "unknown", with "quote": the exact words of the record that decide it (copied, at most 12 words), "" for unknown.
Also say "kind": false when the listing is not the product wanted at all.
Return ONE JSON object and nothing else: {"listings": {"id": {"kind": true or false, "needs": {"requirement number": {"v": "met|likely|against|unknown", "quote": "..."}}}}}"""

CHECKER_GUIDE = ROLE + """
# Check the cart before the order
Reading rules:
""" + MATCH_RULES + """
For each cart line and each requirement of its product write a row. A row is false only when the record of that very listing shows a value that breaks the requirement; quote that value exactly.
Return ONE JSON object and nothing else: {"rows": [{"line": id, "need": requirement number, "page_value": "exact words from the record or \\"\\"", "met": true | false | null}], "missing": [product indexes with no cart line]}"""

NOTES_GUIDE = ROLE + """
# A note arrived while shopping
Return ONE JSON object and nothing else:
{"kind": "change" | "refusal" | "other", "budget": a new total limit as a number or null, "add": [{"product": index, "property": "...", "want": "..."}], "drop": [{"product": index, "property": "..."}], "gone": [listing ids said to be gone], "sold_out": true when it says something chosen is gone or sold out, "price_moved": true when it says a price changed, "summary": "one short sentence"}"""

LINKER_GUIDE = ROLE + """
# Pick one listing per product so the products go together
You get the products, the request's rules that link them, and for each product its candidate listings, best first. Rules are of two kinds: one product must suit or fit another (read model numbers, "works with" lines, sizes, ports and series in titles, descriptions and properties), and if-then rules (when the listing picked for one product has a value, another product must have a given value).
Pick for every product the earliest candidate that keeps every rule true with the other picks. A rule holds when the records show it holds; a fit rule also holds when no record says anything about it. A listing whose own words exclude the other pick (an "except" model, a different size or series) breaks the rule.
When an if-then rule is triggered by a pick and no candidate of the other product meets it, also name what that product must have.
Return ONE JSON object and nothing else: {"choice": {"product index": "listing id"}, "checks": [{"rule": "...", "holds": true | false, "quote": "the words that decide it"}], "needs": [{"product": index, "property": "...", "want": "..."}]}"""

UNITS_GUIDE = """Each listing below gives one measurement, possibly in different units. Convert all of them to one common unit.
Return ONE JSON object and nothing else: {"unit": "the common unit", "values": {"listing id": number or null}}"""


def as_text(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        return str(value)


def shorten(value: Any, limit: int, depth: int = 0) -> Any:
    if depth > 7:
        return "..."
    if isinstance(value, str):
        return value[:limit] + "..." if len(value) > limit else value
    if isinstance(value, list):
        return [shorten(v, limit, depth + 1) for v in value]
    if isinstance(value, dict):
        return {k: shorten(v, limit, depth + 1) for k, v in value.items()}
    return value


def read_json(text: Any) -> dict[str, Any] | None:
    if not isinstance(text, str):
        return None
    body = text.strip()
    fenced = re.match(r"^```[A-Za-z]*\s*(.*?)\s*```$", body, re.S)
    if fenced:
        body = fenced.group(1)
    for candidate in (body, body[body.find("{"): body.rfind("}") + 1] if "{" in body else ""):
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return [value] if isinstance(value, (str, dict)) and value else []


def as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and re.fullmatch(r"\s*\d+(\.0+)?\s*", value):
        return int(float(value))
    return None


def as_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        found = re.fullmatch(r"\s*[^\d\-]{0,3}\s*(-?\d[\d,]*(?:\.\d+)?)\s*[A-Za-z]{0,4}\s*", value)
        if found:
            return float(found.group(1).replace(",", ""))
    return None


def measured(value: Any) -> tuple[float, str] | None:
    found = re.search(r'(-?\d[\d,]*(?:\.\d+)?)\s*("|[a-zA-Z]+)?', str(value))
    if not found:
        return None
    return float(found.group(1).replace(",", "")), (found.group(2) or "").lower()


def plain(key: Any) -> str:
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(key))
    return re.sub(r"[\s_.\-]+", " ", spaced).strip().lower()


def stems(text: Any) -> frozenset[str]:
    return frozenset(w[:-1] if len(w) > 3 and w.endswith("s") else w for w in plain(text).split())


def same_property(wanted: str, key: str) -> bool:
    a, b = stems(wanted), stems(key)
    if not a or not b:
        return False
    return a == b or a <= b or (len(b) >= 2 and b <= a)


def words_of(text: Any) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", str(text).lower())


def quote_in(quote: str, record: str) -> bool:
    q = words_of(quote)
    if not q:
        return False
    body = " ".join(words_of(record))
    if " ".join(q) in body:
        return True
    have = set(body.split())
    hits = sum(1 for w in q if w in have)
    return hits >= max(1, round(0.8 * len(q)))


def quote_value(quote: str) -> str:
    text = str(quote)
    if ":" in text:
        text = text.rsplit(":", 1)[1]
    return " ".join(words_of(text))


def close_text(a: str, b: str) -> bool:
    import difflib
    return bool(a) and bool(b) and len(a) >= 5 and difflib.SequenceMatcher(None, a, b).ratio() >= 0.88


def type_names(schema: dict[str, Any]) -> list[str]:
    kind = schema.get("type")
    if isinstance(kind, str):
        return [kind]
    return [k for k in kind or [] if isinstance(k, str)]


def matches_type(value: Any, kind: str) -> bool:
    checks = {
        "string": lambda v: isinstance(v, str),
        "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
        "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
        "boolean": lambda v: isinstance(v, bool),
        "array": lambda v: isinstance(v, list),
        "object": lambda v: isinstance(v, dict),
        "null": lambda v: v is None,
    }
    return checks.get(kind, lambda v: True)(value)


def coerce(value: Any, kind: str) -> tuple[bool, Any]:
    if kind == "integer" and isinstance(value, float) and value.is_integer():
        return True, int(value)
    if kind in ("integer", "number") and isinstance(value, str) and re.fullmatch(r"\s*-?\d+(\.\d+)?\s*", value):
        number = float(value)
        return True, int(number) if kind == "integer" or number.is_integer() else number
    if kind == "string" and isinstance(value, (int, float)) and not isinstance(value, bool):
        return True, str(value)
    if kind == "boolean" and isinstance(value, str) and value.strip().lower() in ("true", "false"):
        return True, value.strip().lower() == "true"
    if kind == "array" and value is not None and not isinstance(value, list):
        return True, [value]
    return False, value


def conform(value: Any, schema: Any) -> Any:
    if not isinstance(schema, dict):
        return value
    kinds = type_names(schema)
    if kinds and not any(matches_type(value, k) for k in kinds):
        for kind in kinds:
            done, value = coerce(value, kind)
            if done:
                break
    allowed = schema.get("enum")
    if isinstance(allowed, list) and value not in allowed and isinstance(value, str):
        same = [o for o in allowed if isinstance(o, str) and o.lower() == value.strip().lower()]
        value = same[0] if same else value
    if isinstance(value, str) and isinstance(schema.get("maxLength"), int):
        value = value[: schema["maxLength"]]
    if isinstance(value, list) and isinstance(schema.get("items"), dict):
        value = [conform(v, schema["items"]) for v in value]
    if isinstance(value, dict):
        fields = fields_of(schema)
        value = {k: conform(v, fields[k]) if k in fields else v for k, v in value.items()}
    return value


def fields_of(schema: dict[str, Any] | None) -> dict[str, Any]:
    found = (schema or {}).get("properties")
    return found if isinstance(found, dict) else {}


def looks_like_id(value: Any, named: bool = False) -> bool:
    if isinstance(value, int) and not isinstance(value, bool):
        return named
    if not isinstance(value, str):
        return False
    text = value.strip()
    if not 3 <= len(text) <= 64 or " " in text:
        return False
    return named or bool(re.search(r"\d", text))


def choose_id_key(rows: list[dict[str, Any]]) -> str | None:
    best: tuple[tuple[bool, float], str] | None = None
    for key in rows[0]:
        named = "id" in str(key).lower()
        values = [row.get(key) for row in rows]
        if not all(looks_like_id(v, named) for v in values):
            continue
        values = [str(v).strip() for v in values]
        if len(rows) > 1 and len(set(values)) < len(values):
            continue
        score = (named, sum(len(v) for v in values) / len(values))
        if best is None or score > best[0]:
            best = (score, key)
    return best[1] if best else None


def row_lists(node: Any, depth: int = 0) -> list[tuple[str, list[dict[str, Any]]]]:
    found: list[tuple[str, list[dict[str, Any]]]] = []
    if depth > 4 or not isinstance(node, dict):
        return found
    for key, value in node.items():
        if isinstance(value, list) and value and all(isinstance(v, dict) for v in value):
            found.append((key, value))
        elif isinstance(value, dict):
            found.extend(row_lists(value, depth + 1))
    return found


def property_block(node: Any) -> dict[str, Any] | None:
    if not isinstance(node, dict):
        return None
    best: dict[str, Any] | None = None
    for value in node.values():
        if not isinstance(value, dict) or len(value) < 2:
            continue
        scalars = {k: v for k, v in value.items() if isinstance(v, (str, int, float)) and not isinstance(v, bool)}
        if len(scalars) >= 2 and (best is None or len(scalars) > len(best)):
            best = scalars
    return best


def is_sentence(value: Any) -> bool:
    return isinstance(value, str) and len(value.split()) >= 3 and not looks_like_id(value)


def split_notes(observation: Any) -> tuple[Any, list[str], bool]:
    if not isinstance(observation, dict):
        return observation, [], False
    notes: list[str] = []
    refused = False
    inner: Any = observation
    if isinstance(observation.get("refusal"), (list, str)):
        refused = True
        notes.extend(x for x in as_list(observation["refusal"]) if isinstance(x, str))
        inner = observation.get("observation", {k: v for k, v in observation.items() if k != "refusal"})
    elif len(observation) == 2:
        keys = list(observation)
        for a, b in ((keys[0], keys[1]), (keys[1], keys[0])):
            lines = observation[a]
            if isinstance(lines, list) and lines and all(is_sentence(x) for x in lines) and not isinstance(observation[b], str):
                notes.extend(lines)
                inner = observation[b]
                break
    if isinstance(inner, dict) and inner is not observation:
        deeper, more, again = split_notes(inner)
        notes.extend(more)
        refused = refused or again
        inner = deeper
    return inner, notes, refused


def spoken_text(observation: Any) -> str:
    if isinstance(observation, str):
        return observation
    found: list[str] = []

    def walk(node: Any, depth: int) -> None:
        if depth > 4:
            return
        if isinstance(node, str) and len(node.split()) >= 2:
            found.append(node)
        elif isinstance(node, dict):
            for value in node.values():
                walk(value, depth + 1)
        elif isinstance(node, list):
            for value in node:
                walk(value, depth + 1)

    walk(observation, 0)
    return " ".join(found)[:800] if found else as_text(observation)[:400]


def failed_call(observation: Any) -> bool:
    if observation is None:
        return True
    if isinstance(observation, dict):
        if observation.get("error") or observation.get("issue"):
            return True
        if len(observation) == 1:
            value = next(iter(observation.values()))
            if isinstance(value, str) and len(value.split()) >= 3 and not row_lists(observation):
                return True
    return False


def short_query(text: str, limit: int = 6) -> str:
    return " ".join(re.findall(r"[\w\-\.']+", str(text))[:limit])


def need_label(need: dict[str, Any]) -> str:
    said = str(need.get("quote") or "").strip()
    return f"{need.get('property', '')}: {need.get('want', '')}" + (f' (shopper\'s words: "{said}")' if said else "")


class Product:
    def __init__(self, index: int, name: str, noun: str, units: int, musts: list[dict[str, Any]], queries: list[str]):
        self.index = index
        self.name = name
        self.noun = noun or name
        self.units = units
        self.musts = musts
        self.told: list[dict[str, Any]] = []
        self.dropped: set[str] = set()
        self.queries = queries
        self.asked: set[str] = set()
        self.quiet: set[str] = set()
        self.kind_ids: list[str] = []
        self.not_kind: set[str] = set()

    def needs(self) -> list[dict[str, Any]]:
        return [n for n in self.musts + self.told if plain(n.get("property")) not in self.dropped]

    def settled(self) -> set[str]:
        return {plain(n.get("property")) for n in self.needs()} | self.asked | self.quiet


class Store:
    def __init__(self, binding: dict[str, Any], max_steps: int, finished: bool, trace: list[dict[str, Any]]):
        self.binding = binding
        self.max_steps = max_steps
        self.step = 0
        self.finished = finished
        self.log = trace
        self.memo: dict[str, Any] = {}

    def left(self) -> int:
        return self.max_steps - self.step

    def call(self, name: str, args: dict[str, Any]) -> tuple[Any, bool]:
        payload = dict(self.binding)
        payload["step"] = self.step + 1
        payload["action"] = {"name": name, "args": args}
        for _ in range(4):
            result = _env.post_verbose("/environment/call", payload)
            if isinstance(result.data, dict):
                data = result.data
                self.step = data["step"] if isinstance(data.get("step"), int) else self.step + 1
                if data.get("finished"):
                    self.finished = True
                self.log.append({"kind": "call", "step": self.step, "tool": name, "args": args})
                return data.get("observation"), True
            failure = result.error or {}
            status = failure.get("status")
            text = str(failure.get("body") or "")
            self.log.append({"kind": "env_fail", "status": status, "body": text[:200]})
            if status == 409:
                low = text.lower()
                if any(w in low for w in ("finish", "quarantin", "finaliz")):
                    self.finished = True
                    return None, False
                expected = re.search(r"expected (\d+)", text)
                if expected:
                    self.step = int(expected.group(1)) - 1
                elif "reuse" in low or "different action" in low:
                    self.step += 1
                else:
                    return None, False
                payload["step"] = self.step + 1
                continue
            if status in (400, 401, 402, 403, 404, 413, 422):
                return {"error": text[:300]}, False
            time.sleep(1 + random.random())
        return None, False


class Brains:
    def __init__(self, trace: list[dict[str, Any]], started: float):
        self.log = trace
        self.started = started
        self.cost = 0.0
        self.calls = 0
        self.waited = 0.0
        self.no_credit = False
        self.lock = threading.Lock()

    def seat_chain(self, seat: str) -> list[str]:
        chain = SEATS[seat]
        if self.cost >= SPEND_SOFT and seat in HEAVY_SEATS:
            chain = [m for m in chain if m != HEAVY] or chain
        return chain

    def ask(self, seat: str, system: str, user: str) -> dict[str, Any] | None:
        for model in self.seat_chain(seat):
            with self.lock:
                if self.calls >= LLM_CALLS_CAP or self.no_credit or self.cost >= SPEND_HARD:
                    return None
                self.calls += 1
            text = self.post(model, [{"role": "system", "content": system}, {"role": "user", "content": user}], OUTPUT_TOKENS[seat], seat)
            parsed = read_json(text)
            if parsed is not None:
                return parsed
        return None

    def post(self, model: str, messages: list[dict[str, Any]], max_tokens: int, seat: str) -> str | None:
        sent = [dict(m) for m in messages]
        if model.startswith("anthropic/"):
            content = sent[0].get("content")
            if isinstance(content, str) and len(content) > 1500:
                sent[0]["content"] = [{"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}]
        body: dict[str, Any] = {"model": model, "messages": sent, "max_tokens": max_tokens, "temperature": 0}
        attempt = 0
        while True:
            result = _llm.post_verbose("/inference/chat/completions", body)
            data, failure = result.data, result.error or {}
            if isinstance(data, dict) and data.get("choices"):
                usage = data.get("usage") or {}
                with self.lock:
                    if isinstance(usage.get("cost"), (int, float)):
                        self.cost += float(usage["cost"])
                    else:
                        self.cost += (usage.get("prompt_tokens", 0) * (3e-6 if model == HEAVY else 1e-6) + usage.get("completion_tokens", 0) * (15e-6 if model == HEAVY else 5e-6))
                self.log.append({"kind": "llm", "seat": seat, "model": model, "cost": round(self.cost, 5)})
                return (data["choices"][0].get("message") or {}).get("content")
            status = failure.get("status")
            if isinstance(data, dict) and isinstance(data.get("error"), dict):
                status = as_int(data["error"].get("code"))
            self.log.append({"kind": "llm_fail", "seat": seat, "model": model, "status": status})
            if status == 402:
                self.no_credit = True
                return None
            passing = status is None or status in (408, 425, 429, 529) or (isinstance(status, int) and status >= 500)
            if not passing or attempt >= len(WAIT_STEPS) or self.waited + WAIT_STEPS[attempt] > WAIT_BUDGET_SECONDS:
                return None
            if time.monotonic() - self.started + WAIT_STEPS[attempt] > WRAP_UP_SECONDS:
                return None
            self.waited += WAIT_STEPS[attempt]
            time.sleep(WAIT_STEPS[attempt] * (0.8 + 0.4 * random.random()))
            attempt += 1

    def both(self, first: tuple[str, str, str], second: tuple[str, str, str]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        out: list[dict[str, Any] | None] = [None, None]

        def run(slot: int, job: tuple[str, str, str]) -> None:
            try:
                out[slot] = self.ask(*job)
            except Exception:
                out[slot] = None

        worker = threading.Thread(target=run, args=(1, second))
        worker.start()
        run(0, first)
        worker.join(70)
        return out[0], out[1]


class Episode:
    def __init__(self, problem: dict[str, Any]):
        environment = problem["environment"]
        view = environment["policy_view"]
        self.started = time.monotonic()
        self.log: list[dict[str, Any]] = []
        self.request = str(view.get("query") or problem.get("query") or "")
        self.opening = view.get("opening")
        self.tool_specs = [t["function"] for t in view.get("tools") or [] if isinstance(t, dict) and isinstance(t.get("function"), dict) and isinstance(t["function"].get("name"), str)]
        self.params = {t["name"]: t.get("parameters") if isinstance(t.get("parameters"), dict) else {"type": "object", "properties": {}} for t in self.tool_specs}
        steps = as_int(view.get("max_steps")) or 30
        self.store = Store(dict(environment["binding"]), steps, bool(view.get("finished")), self.log)
        self.brains = Brains(self.log, self.started)
        self.tools: dict[str, Any] = {}
        self.plan: dict[str, Any] = {}
        self.products: list[Product] = []
        self.budget: float | None = None
        self.pick = "cheapest"
        self.measure = ""
        self.linked: dict[int, str] = {}
        self.paged: set[str] = set()
        self.size_cache: dict[str, float | None] = {}
        self.msg_cap = DEFAULT_MESSAGES
        self.msg_used = 0
        self.shape: dict[str, Any] = {}
        self.rows: dict[str, dict[str, Any]] = {}
        self.props: dict[str, dict[str, Any]] = {}
        self.pages: dict[str, dict[str, Any]] = {}
        self.order: list[str] = []
        self.gone: set[str] = set()
        self.price_override: dict[str, float] = {}
        self.queries_done: set[str] = set()
        self.next_tokens: dict[str, Any] = {}
        self.inbox: list[str] = []
        self.heard: list[str] = []
        self.verdicts: dict[str, dict[str, dict[str, Any]]] = {}
        self.cart: dict[str, int] = {}
        self.cart_of: dict[str, int] = {}
        self.many_cap = 2
        self.prop_tool: str | None = None
        self.qa: list[tuple[str, str]] = []
        self.stuck: set[str] = set()
        self.loose = False

    def elapsed(self) -> float:
        return time.monotonic() - self.started

    def late(self) -> bool:
        return self.elapsed() > WRAP_UP_SECONDS or self.store.finished

    def steps_left(self) -> int:
        return self.store.left()

    def note(self, kind: str, **fields: Any) -> None:
        self.log.append({"kind": kind, **fields})

    def tool_ok(self, entry: Any, *args: str) -> bool:
        if not isinstance(entry, dict) or entry.get("tool") not in self.params:
            return False
        fields = fields_of(self.params[entry["tool"]])
        return all(entry.get(a) in fields for a in args if entry.get(a) is not None)

    def map_tools(self, got: dict[str, Any] | None) -> None:
        got = as_dict(got)
        for job, needs in (("search", ("query",)), ("details", ("id",)), ("page", ("id",)), ("many", ("ids",)), ("message", ("text",)), ("cart_view", ()), ("remove", ("id",))):
            entry = got.get(job)
            if self.tool_ok(entry, *needs) and all(as_dict(entry).get(n) for n in needs):
                self.tools[job] = entry
        cart = got.get("cart")
        if isinstance(cart, dict) and cart.get("tool") in self.params:
            fields = fields_of(self.params[cart["tool"]])
            if cart.get("lines") in fields or cart.get("id") in fields:
                self.tools["cart"] = cart
        for job in ("buy", "refuse"):
            entry = got.get(job)
            if isinstance(entry, dict) and entry.get("tool") in self.params:
                fields = fields_of(self.params[entry["tool"]])
                entry["args"] = {k: v for k, v in as_dict(entry.get("args")).items() if k in fields}
                self.tools[job] = entry
        if "search" in self.tools:
            fields = fields_of(self.params[self.tools["search"]["tool"]])
            self.tools["search"]["fixed"] = {k: v for k, v in as_dict(self.tools["search"].get("fixed")).items() if k in fields}
            used = {self.tools["search"]["query"], *self.tools["search"]["fixed"]}
            if self.tools["search"].get("page") not in fields or self.tools["search"]["page"] in used:
                self.tools["search"]["page"] = None
            if self.tools["search"].get("price_max") not in fields or self.tools["search"]["price_max"] in used:
                self.tools["search"]["price_max"] = None
            if self.tools["search"].get("category") not in fields or self.tools["search"]["category"] in used:
                self.tools["search"]["category"] = None
            self.tools["search"]["browse"] = {k: v for k, v in as_dict(self.tools["search"].get("browse")).items() if k in fields}
        self.prop_tool = (self.tools.get("details") or {}).get("tool")
        self.note("tools", map={k: v.get("tool") for k, v in self.tools.items()}, search=self.tools.get("search"), cart=self.tools.get("cart"))

    def fill_required(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        schema = self.params.get(name) or {}
        fields = fields_of(schema)
        for key in schema.get("required") or []:
            if key in args or key not in fields:
                continue
            spec = fields[key]
            if isinstance(spec.get("enum"), list) and spec["enum"]:
                args[key] = spec["enum"][0]
            elif "default" in spec:
                args[key] = spec["default"]
        return {k: conform(v, fields.get(k)) for k, v in args.items()}

    def call(self, name: str, args: dict[str, Any]) -> tuple[Any, list[str], bool]:
        if self.store.finished or self.steps_left() <= 0:
            return None, [], False
        args = self.fill_required(name, dict(args))
        key = name + as_text(args)
        observation, ok = self.store.call(name, args)
        inner, notes, refused = split_notes(observation)
        for text in notes:
            if text not in self.heard:
                self.heard.append(text)
                self.inbox.append(text)
        self.soak(name, inner)
        if ok:
            self.store.memo[key] = inner
        return inner, notes, refused

    def soak(self, name: str, inner: Any) -> None:
        if not isinstance(inner, dict):
            return
        for _, rows in row_lists(inner):
            key = self.shape.get("id") if self.shape.get("id") in rows[0] else choose_id_key(rows)
            if not key:
                continue
            for row in rows:
                ident = str(row.get(key, "")).strip()
                if not ident:
                    continue
                merged = self.rows.setdefault(ident, {})
                merged.update({k: v for k, v in row.items() if not isinstance(v, (dict, list))})
                if ident not in self.order:
                    self.order.append(ident)
                block = property_block(row)
                if block and len(block) >= 2 and name != (self.tools.get("search") or {}).get("tool"):
                    self.props[ident] = block

    def take_page(self, ident: str, inner: Any, tool: str) -> bool:
        if not isinstance(inner, dict) or failed_call(inner):
            return False
        page = self.pages.setdefault(ident, {})
        page.update({k: v for k, v in inner.items() if not isinstance(v, dict)})
        block = property_block(inner)
        if block:
            self.props[ident] = block
            if self.prop_tool != tool and tool != (self.tools.get("details") or {}).get("tool"):
                self.prop_tool = self.prop_tool or tool
        merged = self.rows.setdefault(ident, {})
        for k, v in inner.items():
            if not isinstance(v, (dict, list)):
                merged.setdefault(k, v)
        if ident not in self.order:
            self.order.append(ident)
        return bool(block)

    def learn_shape(self, inner: Any) -> None:
        if self.shape.get("rows") or not isinstance(inner, dict):
            return
        lists = row_lists(inner)
        if not lists:
            return
        rows_key, rows = max(lists, key=lambda kv: len(kv[1]))
        sample = shorten({k: (v[:3] if k == rows_key else v) for k, v in inner.items()}, 160)
        got = as_dict(self.brains.ask("shape", SHAPE_GUIDE, as_text(sample)))
        first = rows[0]
        shape = {"rows": rows_key, "id": got.get("id") if got.get("id") in first else choose_id_key(rows)}
        for part in ("price", "stock", "title", "category"):
            shape[part] = got.get(part) if got.get(part) in first else None
        for part in ("count", "next"):
            shape[part] = got.get(part) if got.get(part) in inner else None
        if shape["price"] is None:
            numeric = [k for k, v in first.items() if as_number(v) is not None and not isinstance(v, bool) and k != shape["id"]]
            shape["price"] = numeric[0] if len(numeric) == 1 else None
        if shape["stock"] is None:
            flags = [k for k, v in first.items() if isinstance(v, bool)]
            shape["stock"] = flags[0] if len(flags) == 1 else None
        if shape["title"] is None:
            texts = [k for k, v in first.items() if isinstance(v, str) and len(v.split()) >= 2]
            shape["title"] = min(texts, key=lambda k: len(str(first[k]))) if texts else None
        self.shape = shape
        self.note("shape", shape=shape)

    def price(self, ident: str) -> float | None:
        if ident in self.price_override:
            return self.price_override[ident]
        key = self.shape.get("price")
        for source in (self.pages.get(ident) or {}, self.rows.get(ident) or {}):
            if key and as_number(source.get(key)) is not None:
                return as_number(source.get(key))
        return None

    def in_stock(self, ident: str) -> bool:
        if ident in self.gone:
            return False
        key = self.shape.get("stock")
        for source in (self.pages.get(ident) or {}, self.rows.get(ident) or {}):
            if key and isinstance(source.get(key), bool):
                return source[key]
        return True

    def title(self, ident: str) -> str:
        row = self.rows.get(ident) or {}
        key = self.shape.get("title")
        return str(row.get(key) or "") if key else ""

    def record(self, ident: str, limit: int = RECORD_CHARS) -> str:
        head: dict[str, Any] = {}
        if ident in self.props:
            head["PROPERTIES"] = self.props[ident]
        if ident in self.price_override:
            head["PRICE_NOW"] = self.price_override[ident]
        if ident in self.gone:
            head["SOLD_OUT"] = True
        rest = {k: v for k, v in (self.rows.get(ident) or {}).items()}
        rest.update(self.pages.get(ident) or {})
        rest = dict(sorted(rest.items(), key=lambda kv: len(str(kv[1]))))
        text = as_text(shorten(head, 160))
        room = max(300, limit - len(text))
        return text + " " + as_text(shorten(rest, 200))[:room]

    def search(self, words: str, more: bool = False, label: str | None = None) -> list[str]:
        entry = self.tools.get("search")
        if not entry or self.steps_left() <= CLOSE_RESERVE:
            return []
        if label is not None:
            if not entry.get("category"):
                return []
            return self.run_search(entry, words, more, {**entry.get("browse", {}), entry["category"]: label}, "@" + label.lower())
        return self.run_search(entry, words, more, {}, "")

    def run_search(self, entry: dict[str, Any], words: str, more: bool, extra: dict[str, Any], tag: str) -> list[str]:
        limit = as_dict(fields_of(self.params[entry["tool"]]).get(entry["query"])).get("maxLength")
        words = short_query(words, 8)
        if isinstance(limit, int):
            while len(words) > limit and " " in words:
                words = words.rsplit(" ", 1)[0]
        key = tag + words.lower()
        args = dict(entry.get("fixed") or {})
        args.update(extra)
        if words:
            args[entry["query"]] = words
        if extra and entry.get("price_max") and self.budget is not None and len(self.products) == 1:
            args[entry["price_max"]] = self.budget / max(1, self.products[0].units)
        if more:
            token = self.next_tokens.get(key)
            page = entry.get("page") or self.page_param(token)
            if token is None or not page:
                return []
            args[page] = token
        elif key in self.queries_done:
            return []
        self.queries_done.add(key)
        inner, _, _ = self.call(entry["tool"], args)
        self.learn_shape(inner)
        found: list[str] = []
        if isinstance(inner, dict):
            rows = inner.get(self.shape.get("rows")) if self.shape.get("rows") in inner else None
            if rows is None:
                lists = row_lists(inner)
                rows = max(lists, key=lambda kv: len(kv[1]))[1] if lists else []
            idk = self.shape.get("id")
            found = [str(r.get(idk)).strip() for r in rows if isinstance(r, dict) and idk and r.get(idk) is not None]
            nxt = inner.get(self.shape.get("next")) if self.shape.get("next") else None
            self.next_tokens[key] = nxt if nxt not in (None, "", False) and found else None
        self.note("search", words=words, tag=tag, more=more, found=len(found))
        return found

    def page_param(self, token: Any) -> str | None:
        entry = self.tools.get("search") or {}
        fields = fields_of(self.params.get(entry.get("tool"), {}))
        used = {entry.get("query"), entry.get("price_max"), entry.get("category"), *as_dict(entry.get("fixed")), *as_dict(entry.get("browse"))}
        kind = "integer" if isinstance(token, int) and not isinstance(token, bool) else "string"
        free = [k for k, v in fields.items() if k not in used and isinstance(v, dict) and (kind in type_names(v) or (kind == "integer" and "number" in type_names(v))) and "enum" not in v]
        if kind == "integer":
            free = [k for k in free if "maximum" in fields[k] or "minimum" in fields[k]] or free
        return free[0] if len(free) == 1 else None

    def open_many(self, ids: list[str]) -> int:
        ids = [i for i in dict.fromkeys(ids) if i not in self.props]
        opened = 0
        many = self.tools.get("many")
        use_many = many and self.prop_tool == many["tool"]
        while ids and self.steps_left() > CLOSE_RESERVE and not self.late():
            if use_many and len(ids) > 1:
                chunk, ids = ids[: self.many_cap], ids[self.many_cap:]
                inner, _, _ = self.call(many["tool"], {many["ids"]: chunk})
                got = 0
                for _, rows in row_lists(as_dict(inner)):
                    for row in rows:
                        for ident in chunk:
                            if ident in [str(v).strip() for v in row.values() if isinstance(v, (str, int))]:
                                got += self.take_page(ident, row, many["tool"])
                if not got:
                    self.many_cap = max(1, self.many_cap // 2)
                    if self.many_cap == 1:
                        use_many = False
                    ids = chunk + ids
                opened += got
                continue
            ident = ids.pop(0)
            opened += self.open_one(ident)
        return opened

    def open_one(self, ident: str) -> bool:
        order = [self.prop_tool] if self.prop_tool else []
        for job in ("details", "page"):
            entry = self.tools.get(job)
            if entry and entry["tool"] not in order:
                order.append(entry["tool"])
        for tool in order:
            if self.steps_left() <= CLOSE_RESERVE:
                return False
            job = next((j for j in ("details", "page") if (self.tools.get(j) or {}).get("tool") == tool), None)
            if job is None:
                continue
            inner, _, _ = self.call(tool, {self.tools[job]["id"]: ident})
            if self.take_page(ident, inner, tool):
                self.prop_tool = tool
                return True
            if self.prop_tool == tool and not failed_call(inner):
                return False
        return False

    def read_page(self, ident: str) -> None:
        entry = self.tools.get("page") or self.tools.get("details")
        if entry and self.steps_left() > 2:
            inner, _, _ = self.call(entry["tool"], {entry["id"]: ident})
            self.take_page(ident, inner, entry["tool"])

    def message(self, text: str) -> str | None:
        entry = self.tools.get("message")
        if not entry or self.msg_used >= self.msg_cap or self.steps_left() <= CLOSE_RESERVE:
            return None
        inner, _, _ = self.call(entry["tool"], {entry["text"]: text})
        self.msg_used += 1
        if inner is None or (isinstance(inner, dict) and inner.get("error")):
            self.msg_cap = self.msg_used
            return None
        if isinstance(inner, dict):
            for key in ("value", "reply", "answer", "text", "message"):
                if isinstance(inner.get(key), str):
                    return inner[key]
            strings = [v for v in inner.values() if isinstance(v, str) and len(v.split()) >= 2]
            if strings:
                return max(strings, key=len)
        return spoken_text(inner)

    def cart_args(self, ident: str, units: int, adding: bool) -> tuple[str, dict[str, Any]] | None:
        entry = self.tools.get("cart")
        if not entry:
            return None
        if entry.get("lines"):
            lines = dict(self.cart)
            if adding:
                lines[ident] = units
            else:
                lines.pop(ident, None)
            id_key, qty_key = entry.get("line_id") or "id", entry.get("line_qty") or "quantity"
            return entry["tool"], {entry["lines"]: [{id_key: i, qty_key: q} for i, q in lines.items()]}
        if not adding and self.tools.get("remove"):
            rem = self.tools["remove"]
            return rem["tool"], {rem["id"]: ident}
        args: dict[str, Any] = {entry["id"]: ident}
        qty = entry.get("qty") or self.qty_param()
        if qty:
            args[qty] = units
        if entry.get("verb"):
            value = entry.get("add") if adding else entry.get("remove")
            if value is None:
                return None
            args[entry["verb"]] = value
        elif not adding:
            return None
        return entry["tool"], args

    def qty_param(self) -> str | None:
        entry = self.tools.get("cart") or {}
        fields = fields_of(self.params.get(entry.get("tool"), {}))
        free = [k for k, v in fields.items() if k not in (entry.get("id"), entry.get("verb")) and isinstance(v, dict) and "integer" in type_names(v) and "enum" not in v]
        return free[0] if len(free) == 1 else None

    def put(self, ident: str, units: int) -> bool:
        plan = self.cart_args(ident, units, True)
        if not plan or self.steps_left() <= 2:
            return False
        inner, _, _ = self.call(*plan)
        if failed_call(inner):
            return False
        self.cart[ident] = units
        return True

    def take_out(self, ident: str) -> bool:
        plan = self.cart_args(ident, self.cart.get(ident, 1), False)
        if not plan or self.steps_left() <= 2:
            return False
        inner, _, _ = self.call(*plan)
        if failed_call(inner):
            return False
        self.cart.pop(ident, None)
        return True

    def view_cart(self) -> None:
        entry = self.tools.get("cart_view")
        if entry and self.steps_left() > 3:
            self.call(entry["tool"], {})

    def close(self, buying: bool) -> tuple[bool, list[str]]:
        entry = self.tools.get("buy" if buying else "refuse")
        if not entry:
            return False, []
        inner, notes, refused = self.call(entry["tool"], dict(entry.get("args") or {}))
        if self.store.finished:
            return True, notes
        if not notes and isinstance(inner, dict):
            notes = [v for v in inner.values() if is_sentence(v)]
        return False, notes

    def tools_digest(self) -> str:
        out = []
        for spec in self.tool_specs:
            out.append({"name": spec["name"], "description": str(spec.get("description") or "")[:400], "parameters": shorten(self.params[spec["name"]], 200)})
        return as_text(out)

    def read_request(self) -> None:
        user_map = "TOOLS:\n" + self.tools_digest()
        user_read = "REQUEST:\n" + self.request + ("\n\nOPENING NOTE:\n" + as_text(self.opening) if self.opening else "")
        mapped, plan = self.brains.both(("toolmap", TOOLMAP_GUIDE, user_map), ("reader", READER_GUIDE, user_read))
        self.map_tools(mapped)
        if not self.tools.get("search") or not self.tools.get("buy"):
            again = self.brains.ask("toolmap", TOOLMAP_GUIDE, user_map + "\n\nName every job; the last answer was missing some.")
            self.map_tools(again)
        plan = as_dict(plan) or as_dict(self.brains.ask("reader", READER_GUIDE, user_read))
        self.plan = plan
        for index, item in enumerate(as_list(plan.get("products"))[:6]):
            item = as_dict(item)
            name = str(item.get("name") or "").strip()
            if not name:
                continue
            musts = [m for m in as_list(item.get("musts")) if isinstance(m, dict) and m.get("property") and m.get("want")]
            queries = [str(q) for q in as_list(item.get("queries")) if isinstance(q, str) and q.strip()] or [name]
            self.products.append(Product(len(self.products), name, str(item.get("noun") or name), max(1, as_int(item.get("units")) or 1), musts, queries[:3]))
        if not self.products:
            self.products.append(Product(0, short_query(self.request, 5), short_query(self.request, 3), 1, [], [short_query(self.request, 5)]))
        self.budget = as_number(plan.get("budget"))
        self.pick = str(plan.get("pick") or "cheapest")
        self.measure = str(plan.get("measure") or "").strip() if self.pick in ("most", "least") else ""
        cap = as_int(plan.get("messages"))
        if cap is not None and 0 <= cap <= 12:
            self.msg_cap = cap
        stated = as_int(plan.get("steps"))
        if stated and 0 < stated < self.store.max_steps:
            self.store.max_steps = stated
        self.note("read", products=[(p.name, p.units, [need_label(m) for m in p.musts]) for p in self.products], budget=self.budget, pick=self.pick, msgs=self.msg_cap)

    def screen(self, ids: list[str], owner: Product | None = None) -> None:
        fresh = [i for i in dict.fromkeys(ids) if all(i not in p.kind_ids and i not in p.not_kind for p in self.products)]
        if not fresh:
            return
        if len(self.products) == 1 and len(fresh) <= 2:
            self.products[0].kind_ids.extend(fresh)
            return
        rows = {i: shorten({k: v for k, v in (self.rows.get(i) or {}).items() if isinstance(v, (str, int, float))}, 90) for i in fresh[:40]}
        wanted = {str(p.index): p.name + ("; shopper wants " + "; ".join(f"{n.get('property', '')}: {n.get('want', '')}" for n in p.needs() if not n.get("points")) if p.needs() else "") for p in self.products}
        got = as_dict(self.brains.ask("screen", SCREEN_GUIDE, "PRODUCTS:\n" + as_text(wanted) + "\n\nROWS:\n" + as_text(rows)))
        keep = as_dict(got.get("keep"))
        if not keep:
            for p in ([owner] if owner else self.products):
                p.kind_ids.extend(i for i in fresh if i not in p.kind_ids)
            return
        self.note("screen", rows=len(rows), keep={k: len(as_list(v)) for k, v in keep.items()})
        for p in self.products:
            chosen = [str(i) for i in as_list(keep.get(str(p.index))) if str(i) in rows]
            p.kind_ids.extend(i for i in chosen if i not in p.kind_ids)
            p.not_kind.update(i for i in rows if i not in chosen)

    def cheap_first(self, ids: list[str], p: Product | None = None) -> list[str]:
        wanted = [set(words_of(n.get("want"))) for n in (p.needs() if p else []) if not n.get("points")]
        wanted = [w for w in wanted if w and len(w) <= 4]

        def hits(i: str) -> int:
            text = set(words_of(as_text(self.rows.get(i) or {})))
            return sum(1 for w in wanted if w <= text)

        return sorted(ids, key=lambda i: (not self.in_stock(i), -hits(i), self.price(i) if self.price(i) is not None else 1e12))

    def survey(self) -> None:
        each = SURVEY_OPENS if len(self.products) == 1 else SURVEY_OPENS_EACH
        for p in self.products:
            if self.late():
                return
            found: list[str] = []
            for words in p.queries[:2 if len(self.products) == 1 else 1]:
                found.extend(self.search(words))
            self.screen(found, p)
            if not any(self.affordable(i, p) for i in p.kind_ids):
                self.screen(self.browse(p, pages=1), p)
            pool = [i for i in p.kind_ids if i not in self.props]
            pool = sorted(self.cheap_first(pool, p), key=lambda i: not self.affordable(i, p))
            self.open_many(pool[:each])

    def menu(self, p: Product) -> list[tuple[str, int, int, list[str]]]:
        cover: dict[str, int] = {}
        values: dict[str, list[str]] = {}
        shown: dict[str, str] = {}
        opened = [i for i in p.kind_ids if i in self.props] or [i for i in self.props if i not in p.not_kind]
        for ident in opened:
            for key, value in self.props[ident].items():
                name = plain(key)
                if not name or len(name) > 40:
                    continue
                shown.setdefault(name, name)
                cover[name] = cover.get(name, 0) + 1
                text = str(value).strip().lower()[:40]
                if text and text not in values.setdefault(name, []):
                    values[name].append(text)
        rows = [(shown[k], cover[k], len(values.get(k, [])), values.get(k, [])[:4]) for k in cover]
        rows.sort(key=lambda r: -(r[1] * min(r[2], 6)))
        return rows

    def plan_questions(self) -> dict[int, list[str]]:
        menus = {p.index: self.menu(p) for p in self.products}
        limit = max(3, self.msg_cap)
        lines = []
        for p in self.products:
            lines.append(f"PRODUCT {p.index}: {p.name} (units {p.units})")
            lines.append("  settled by the request: " + ("; ".join(need_label(n) for n in p.needs()) or "nothing"))
            for key, cover, distinct, samples in menus[p.index][:28]:
                lines.append(f"  key '{key}': on {cover} listings, {distinct} values, e.g. {samples}")
        got = as_dict(self.brains.ask("planner", PLANNER_GUIDE.replace("LIMIT", str(limit)), "REQUEST:\n" + self.request + "\n\n" + "\n".join(lines)))
        for item in as_list(got.get("same")):
            item = as_dict(item)
            index = as_int(item.get("product"))
            if index is None or not 0 <= index < len(self.products):
                continue
            key = plain(item.get("key"))
            if key not in [k for k, _, _, _ in menus[index]]:
                continue
            for need in self.products[index].musts:
                if plain(need.get("property")) == plain(item.get("property")) and plain(need.get("property")) != key:
                    self.note("rename", product=index, old=need.get("property"), new=key)
                    need["property"] = key
        ranked: dict[int, list[str]] = {}
        asks = as_dict(got.get("ask"))
        for p in self.products:
            listed = [k for k, _, _, _ in menus[p.index]]
            mine = []
            for key in as_list(asks.get(str(p.index))):
                if not isinstance(key, str) or not key.strip():
                    continue
                name = plain(key)
                exact = next((k for k in listed if k == name), None) or next((k for k in listed if same_property(name, k)), None)
                mine.append(exact or name)
            for key in listed:
                if key not in mine:
                    mine.append(key)
            ranked[p.index] = [k for k in dict.fromkeys(mine) if not any(same_property(k, s) for s in p.settled())]
        return ranked

    def question(self, p: Product, key: str) -> str:
        if len(self.products) == 1:
            return f"What {key} should the {p.noun} have?"
        return f"For the {p.noun}, what {key} should it have?"

    def interview(self) -> None:
        if not self.tools.get("message") or self.msg_cap <= 0:
            return
        ranked = self.plan_questions()
        turn = 0
        while self.msg_used < self.msg_cap and not self.late() and self.steps_left() > CLOSE_RESERVE + 4:
            open_products = [p for p in self.products if any(not any(same_property(k, s) for s in p.settled()) for k in ranked.get(p.index, []))]
            if not open_products:
                return
            open_products.sort(key=lambda p: (len(p.told), len(p.asked), (p.index - turn) % len(self.products)))
            p = open_products[0]
            key = next(k for k in ranked[p.index] if not any(same_property(k, s) for s in p.settled()))
            p.asked.add(plain(key))
            reply = self.message(self.question(p, key))
            turn += 1
            if reply is None:
                return
            self.qa.append((f"{p.noun} / {key}", reply))
            self.listen(p, key, reply)
            self.read_inbox()

    def listen(self, p: Product, key: str, reply: str) -> None:
        prompt = f"PRODUCT: {p.name}\nASKED ABOUT: {key}\nREPLY: {reply}"
        got = as_dict(self.brains.ask("listener", LISTENER_GUIDE, prompt))
        need = as_dict(got.get("need"))
        if need.get("want") and not quote_in(str(need.get("quote") or ""), reply):
            again = as_dict(self.brains.ask("listener_strict", LISTENER_GUIDE, prompt + "\n\nThe quote must be the reply's own words."))
            if as_dict(again.get("need")).get("want"):
                got, need = again, as_dict(again.get("need"))
        repeat = need.get("want") and any(" ".join(words_of(n.get("want")))[-60:] == " ".join(words_of(need.get("want")))[-60:] for x in self.products for n in x.needs())
        if need.get("want") and not got.get("no_need") and not repeat:
            need["property"] = str(need.get("property") or key)
            need["points"] = bool(need.get("points")) and len(words_of(need.get("want"))) >= 8
            said = " ".join(words_of(reply))
            if " ".join(words_of(need["property"])) not in said:
                named = [k for k, _, _, _ in self.menu(p) if " ".join(words_of(k)) in said]
                if named:
                    need["property"] = min(named, key=lambda k: (said.find(" ".join(words_of(k))), -len(k)))
            need["told"] = True
            p.told.append(need)
            self.note("told", product=p.index, need=need_label(need), points=bool(need.get("points")))
        else:
            p.quiet.add(plain(key))
        for prop in as_list(got.get("drop")):
            if isinstance(prop, str):
                p.dropped.add(plain(prop))
        if as_number(got.get("budget")) is not None:
            self.budget = as_number(got.get("budget"))

    def read_inbox(self) -> dict[str, Any]:
        if not self.inbox:
            return {}
        texts, self.inbox = self.inbox[:], []
        products = {str(p.index): p.name for p in self.products}
        cart = {i: self.title(i)[:60] for i in self.cart}
        got = as_dict(self.brains.ask("notes", NOTES_GUIDE, "PRODUCTS: " + as_text(products) + "\nCART: " + as_text(cart) + "\nNOTES:\n" + "\n".join(texts)))
        if as_number(got.get("budget")) is not None:
            self.budget = as_number(got.get("budget"))
        for add in as_list(got.get("add")):
            add = as_dict(add)
            index = as_int(add.get("product")) or 0
            if 0 <= index < len(self.products) and add.get("want"):
                self.products[index].told.append({"property": str(add.get("property") or ""), "want": str(add["want"]), "told": True})
        for drop in as_list(got.get("drop")):
            drop = as_dict(drop)
            index = as_int(drop.get("product")) or 0
            if 0 <= index < len(self.products) and drop.get("property"):
                self.products[index].dropped.add(plain(drop["property"]))
        for ident in as_list(got.get("gone")):
            if isinstance(ident, str) and ident in self.rows:
                self.gone.add(ident)
        self.note("notes", texts=texts, read=shorten(got, 120))
        return got

    def pointer_needs(self) -> list[tuple[Product, dict[str, Any]]]:
        return [(p, n) for p in self.products for n in p.needs() if (n.get("points") or len(words_of(n.get("want"))) >= 12) and not n.get("resolved")]

    def follow_pointers(self) -> None:
        for p, need in self.pointer_needs():
            text = str(need.get("want"))
            searched: set[str] = set()
            seller_hint = ""
            for _ in range(POINTER_HOPS + 1):
                if self.late() or self.steps_left() <= CLOSE_RESERVE + 3:
                    return
                opened = [i for i in self.order if i in self.pages or i in self.props]
                low = str(seller_hint or "").lower()
                mine = [i for i in opened if low and low in as_text(self.rows.get(i) or {}).lower()]
                pick = list(dict.fromkeys(mine[-6:] + opened[-6:]))
                seen = {i: self.record(i, 1600) for i in pick}
                got = as_dict(self.brains.ask("pointer", POINTER_GUIDE, f"REQUIREMENT: {need.get('property')}: {text}\n\nRECORDS:\n" + as_text(seen)))
                self.note("pointer_read", got=shorten(got, 80), seen=len(seen))
                if got.get("found") in self.rows and got.get("value") and not got.get("next"):
                    need["want"] = f"{need.get('want')} [that listing's {need.get('property')}: {got['value']}]"
                    need["resolved"] = True
                    need["points"] = False
                    self.note("pointer", need=need_label(need), source=got["found"])
                    break
                if got.get("next") and got.get("found") in self.rows and str(got["next"]) != text:
                    text = str(got["next"])
                    searched = set()
                    continue
                seller = str(got.get("seller") or "").strip()
                kind = str(got.get("kind") or "").strip()
                seller_hint = seller
                if not seller and not kind:
                    break
                found: list[str] = []
                for words in (f"{kind} {seller}".strip(), seller):
                    if words and words.lower() not in searched:
                        searched.add(words.lower())
                        found.extend(self.search(words))
                pool = found or list(self.order)
                if seller:
                    low = seller.lower()
                    pool = [i for i in pool if low in as_text(self.rows.get(i) or {}).lower()]
                clues = [str(c).lower() for c in as_list(got.get("clues")) if isinstance(c, str)]
                pool.sort(key=lambda i: -sum(1 for c in clues if c and c in as_text(self.rows.get(i) or {}).lower()))
                target = [i for i in dict.fromkeys(pool) if i not in self.pages and i not in self.props][:3]
                if not target and seller:
                    more = [i for i in self.search(seller, more=True) if low in as_text(self.rows.get(i) or {}).lower()]
                    target = [i for i in more if i not in self.pages and i not in self.props][:3]
                if not target:
                    break
                self.open_many(target)

    def judge(self, p: Product, ids: list[str]) -> None:
        needs = p.needs()
        ids = [i for i in ids if i not in self.verdicts or self.verdicts[i].get("_n") != len(needs)]
        if not ids:
            return
        if not needs:
            for i in ids:
                self.verdicts[i] = {"_n": 0}
            return
        listing = "\n".join(f"{n}: {need_label(x)}" for n, x in enumerate(needs))
        for start in range(0, len(ids), 6):
            chunk = ids[start:start + 6]
            records = {i: self.record(i) for i in chunk}
            got = as_dict(self.brains.ask("judge", JUDGE_GUIDE, f"PRODUCT: {p.name}\nREQUIREMENTS:\n{listing}\n\nLISTINGS:\n" + as_text(records)))
            for ident in chunk:
                entry = as_dict(as_dict(got.get("listings")).get(ident))
                result: dict[str, Any] = {"_n": len(needs)}
                if entry.get("kind") is False:
                    p.not_kind.add(ident)
                for n in range(len(needs)):
                    v = as_dict(as_dict(entry.get("needs")).get(str(n)))
                    verdict = str(v.get("v") or "unknown")
                    quote = str(v.get("quote") or "")
                    if verdict in ("met", "likely", "against") and not quote_in(quote, records[ident]):
                        verdict = "unknown"
                    result[str(n)] = {"v": verdict, "quote": quote}
                self.verdicts[ident] = result
        self.settle_verdicts(p)

    def settle_verdicts(self, p: Product) -> None:
        met: dict[str, set[str]] = {}
        for ident in p.kind_ids:
            for k, x in (self.verdicts.get(ident) or {}).items():
                if k != "_n" and x.get("v") == "met":
                    met.setdefault(k, set()).add(quote_value(x.get("quote", "")))
        needs = p.needs()
        for ident in p.kind_ids:
            for k, x in (self.verdicts.get(ident) or {}).items():
                if k == "_n" or x.get("v") != "against":
                    continue
                value = quote_value(x.get("quote", ""))
                index = as_int(k)
                want = " ".join(words_of(needs[index].get("want"))) if index is not None and index < len(needs) else ""
                if value in met.get(k, set()) or any(close_text(value, m) for m in met.get(k, set())):
                    x["v"] = "met"
                elif want and len(want.split()) <= 3 and (set(want.split()) <= set(value.split()) or close_text(value, want)):
                    x["v"] = "met"

    def standing(self, ident: str) -> tuple[int, int]:
        v = self.verdicts.get(ident) or {}
        against = sum(1 for k, x in v.items() if k != "_n" and x.get("v") == "against")
        weak = sum(2 if x.get("v") == "unknown" else 1 for k, x in v.items() if k != "_n" and x.get("v") in ("unknown", "likely"))
        return against, weak

    def readable(self, p: Product) -> set[str]:
        out: set[str] = set()
        for ident in p.kind_ids:
            for k, x in (self.verdicts.get(ident) or {}).items():
                if k != "_n" and x.get("v") in ("met", "against"):
                    out.add(k)
        return out

    def fits(self, p: Product) -> list[str]:
        out = []
        readable = self.readable(p)
        for ident in p.kind_ids:
            if ident in p.not_kind or ident not in self.verdicts or not self.in_stock(ident):
                continue
            against, _ = self.standing(ident)
            if any(k in readable and x.get("v") == "unknown" for k, x in self.verdicts[ident].items() if k != "_n"):
                continue
            price = self.price(ident)
            if against:
                continue
            if self.budget is not None and price is not None and price * p.units > self.budget:
                continue
            out.append(ident)
        return self.rank(out)

    def size_text(self, ident: str) -> str | None:
        props = self.props.get(ident) or {}
        key = next((k for k in props if plain(k) == plain(self.measure)), None) or next((k for k in props if same_property(self.measure, k)), None)
        return str(props[key]) if key is not None else None

    def sizes(self, ids: list[str]) -> dict[str, float]:
        texts = {i: t for i in ids if (t := self.size_text(i)) and measured(t)}
        units = {measured(t)[1] for t in texts.values()}
        if len(units) <= 1:
            return {i: measured(t)[0] for i, t in texts.items()}
        todo = {i: t for i, t in texts.items() if i not in self.size_cache}
        if todo:
            got = as_dict(as_dict(self.brains.ask("units", UNITS_GUIDE, as_text(todo))).get("values"))
            for i in todo:
                self.size_cache[i] = as_number(got.get(i))
        return {i: self.size_cache[i] for i in texts if self.size_cache.get(i) is not None}

    def rank(self, ids: list[str]) -> list[str]:
        size = self.sizes(ids) if self.measure else {}

        def order(i: str) -> tuple:
            price = self.price(i) if self.price(i) is not None else 1e12
            if self.measure:
                value = size.get(i)
                return (value is None, -(value or 0) if self.pick == "most" else (value or 0), min(self.standing(i)[1], 2), price)
            return (min(self.standing(i)[1], 2), price)
        return sorted(ids, key=order)

    def hunt(self, p: Product, opens: int) -> None:
        needs = p.needs()
        done = sorted(self.queries_done)[-8:]
        queries: list[str] = []
        if needs:
            got = as_dict(self.brains.ask("seeker", SEEKER_GUIDE, f"PRODUCT: {p.name}\nNEEDS:\n" + "\n".join(need_label(n) for n in needs) + "\nDONE: " + as_text(done)))
            queries = [q for q in as_list(got.get("queries")) if isinstance(q, str)][:HUNT_QUERIES]
        queries += [q for q in p.queries if q.lower() not in self.queries_done]
        found: list[str] = []
        rank: dict[str, int] = {}
        for words in queries[:HUNT_QUERIES]:
            if self.steps_left() <= CLOSE_RESERVE + 4 or self.late():
                break
            rows = self.search(words)
            for place, ident in enumerate(rows):
                rank[ident] = min(rank.get(ident, place), place)
            found.extend(rows)
        if self.pick == "cheapest" and self.steps_left() > CLOSE_RESERVE + 8 and p.queries:
            found.extend(self.search(p.queries[0], more=True))
        found.extend(self.browse(p))
        self.screen(found, p)
        pool = [i for i in self.cheap_first(p.kind_ids, p) if i not in self.props and self.in_stock(i) and self.affordable(i, p)]
        aimed = sorted((i for i in pool if i in rank), key=lambda i: rank[i])
        front = aimed[:max(2, opens - (3 if self.pick == "cheapest" else 2))]
        self.open_many((front + [i for i in pool if i not in front])[:opens])
        self.judge(p, [i for i in p.kind_ids if i in self.props or i in self.pages])

    def affordable(self, ident: str, p: Product) -> bool:
        price = self.price(ident)
        return self.budget is None or price is None or price * p.units <= self.budget

    def browse(self, p: Product, pages: int = 3) -> list[str]:
        key = self.shape.get("category")
        if not key or not (self.tools.get("search") or {}).get("category"):
            return []
        labels: dict[str, int] = {}
        for ident in p.kind_ids:
            label = (self.rows.get(ident) or {}).get(key)
            if isinstance(label, str) and label.strip():
                labels[label] = labels.get(label, 0) + 1
        if not labels:
            return []
        label = max(labels, key=labels.get)
        found = self.search("", label=label)
        for _ in range(pages - 1):
            if self.steps_left() <= CLOSE_RESERVE + 8 or not found:
                break
            more = self.search("", more=True, label=label)
            if not more:
                break
            found.extend(more)
        return found

    def choose(self, strict: bool = True) -> dict[int, str]:
        choice: dict[int, str] = {}
        for p in self.products:
            fits = self.fits(p) or ([] if strict else self.loose_fits(p))
            linked = self.linked.get(p.index)
            if linked in fits:
                choice[p.index] = linked
            elif fits:
                choice[p.index] = fits[0]
        return choice

    def link_rules(self) -> list[str]:
        rules = []
        names = {p.index: p.name for p in self.products}
        for item in as_list(self.plan.get("fits")):
            item = as_dict(item)
            part, host = as_int(item.get("part")), as_int(item.get("host"))
            if part in names and host in names and part != host:
                rules.append(f"product {part} ({names[part]}) must suit product {host} ({names[host]}): {item.get('how', '')}")
        for item in as_list(self.plan.get("ifs")):
            item = as_dict(item)
            one, other = as_int(item.get("product")), as_int(item.get("other"))
            if one in names and other in names and item.get("when") and item.get("then"):
                rules.append(f"if product {one} ({names[one]}): {item.get('when')} -> then product {other} ({names[other]}): {item.get('then')}")
        return rules

    def link(self, choice: dict[int, str], again: bool = True) -> dict[int, str]:
        rules = self.link_rules()
        if len(self.products) < 2 or not rules or self.late():
            return choice
        pools: dict[int, list[str]] = {}
        for p in self.products:
            pool = self.fits(p) or self.loose_fits(p)
            first = [choice[p.index]] if p.index in choice else []
            pools[p.index] = list(dict.fromkeys(first + pool))[:4]
        if self.tools.get("page") and (self.tools.get("page") or {}).get("tool") != self.prop_tool:
            reads = [i for k in sorted(pools) for i in pools[k][:3] if i not in self.paged]
            for ident in reads[:10]:
                if self.steps_left() <= CLOSE_RESERVE + 4 or self.late():
                    break
                self.paged.add(ident)
                self.read_page(ident)
        products = {str(p.index): {"name": p.name, "needs": [need_label(n) for n in p.needs()], "candidates": {i: self.record(i, 900) for i in pools[p.index]}} for p in self.products}
        got = as_dict(self.brains.ask("linker", LINKER_GUIDE, "RULES:\n" + "\n".join(rules) + "\n\nPRODUCTS:\n" + as_text(products)))
        picked = as_dict(got.get("choice"))
        out = dict(choice)
        for p in self.products:
            ident = str(picked.get(str(p.index)) or "")
            if ident in pools[p.index]:
                out[p.index] = ident
                self.linked[p.index] = ident
        self.note("link", rules=rules, before=choice, after=out, checks=as_list(got.get("checks"))[:6], needs=as_list(got.get("needs"))[:3])
        added = []
        for item in as_list(got.get("needs"))[:2]:
            item = as_dict(item)
            index = as_int(item.get("product"))
            if index is None or not 0 <= index < len(self.products) or not item.get("property") or not item.get("want"):
                continue
            p = self.products[index]
            if any(same_property(str(item["property"]), str(n.get("property"))) for n in p.needs()):
                continue
            p.musts.append({"property": str(item["property"]), "want": str(item["want"])})
            added.append(p)
        if added and again and self.steps_left() > CLOSE_RESERVE + 10 and not self.late():
            for p in added:
                self.linked.pop(p.index, None)
                self.judge(p, [i for i in p.kind_ids if i in self.props or i in self.pages])
                self.hunt(p, 4)
            fresh = self.choose() or self.choose(strict=False)
            for index, ident in out.items():
                fresh.setdefault(index, ident)
            return self.link(fresh, again=False)
        return out

    def loose_fits(self, p: Product) -> list[str]:
        pool = [i for i in p.kind_ids if i in self.verdicts and i not in p.not_kind and self.in_stock(i) and self.affordable(i, p)]
        if not self.loose:
            pool = [i for i in pool if self.standing(i)[0] == 0]
        return sorted(self.rank(pool), key=lambda i: self.standing(i)[0])

    def ensure_cart(self, choice: dict[int, str]) -> None:
        wanted = {choice[p.index]: p.units for p in self.products if p.index in choice}
        for ident in list(self.cart):
            if ident not in wanted:
                self.take_out(ident)
        for ident, units in sorted(wanted.items(), key=lambda kv: self.price(kv[0]) or 0):
            if self.cart.get(ident) != units:
                if ident in self.cart:
                    self.take_out(ident)
                if self.put(ident, units):
                    for p in self.products:
                        if choice.get(p.index) == ident:
                            self.cart_of[ident] = p.index

    def after_change(self) -> bool:
        got = self.read_inbox()
        changed = bool(got.get("add") or got.get("drop") or got.get("gone") or got.get("budget") is not None)
        if got.get("sold_out") and not got.get("gone"):
            for ident in list(self.cart):
                self.gone.add(ident)
            changed = True
        if got.get("price_moved"):
            for ident in list(self.cart):
                self.read_page(ident)
            changed = True
        for ident in list(self.cart):
            if ident in self.gone and not self.take_out(ident):
                self.cart.pop(ident, None)
                self.stuck.add(ident)
        if got.get("add"):
            for p in self.products:
                self.judge(p, [i for i in p.kind_ids if i in self.verdicts])
        return changed

    def check_cart(self) -> None:
        lines = {i: self.record(i, 1200) for i in self.cart}
        if not lines:
            return
        needs = {str(p.index): {"name": p.name, "needs": {str(n): need_label(x) for n, x in enumerate(p.needs())}} for p in self.products}
        mapping = {i: self.cart_of.get(i, 0) for i in self.cart}
        got = as_dict(self.brains.ask("checker", CHECKER_GUIDE, "REQUEST:\n" + self.request + "\n\nPRODUCTS AND REQUIREMENTS:\n" + as_text(needs) + "\n\nCART LINES (listing -> product index):\n" + as_text(mapping) + "\n\nRECORDS:\n" + as_text(lines)))
        bad: set[str] = set()
        for row in as_list(got.get("rows")):
            row = as_dict(row)
            ident = str(row.get("line") or "")
            if row.get("met") is False and ident in lines and quote_in(str(row.get("page_value") or ""), lines[ident]):
                bad.add(ident)
                need = as_int(row.get("need"))
                v = self.verdicts.setdefault(ident, {"_n": 0})
                if need is not None:
                    v[str(need)] = {"v": "against", "quote": str(row.get("page_value"))}
        self.note("check", bad=sorted(bad))
        if bad:
            choice = self.choose() or self.choose(strict=False)
            if choice:
                self.ensure_cart(choice)

    def finish_order(self) -> bool:
        for _ in range(CLOSE_ROUNDS):
            if self.store.finished:
                return True
            if not self.cart:
                choice = self.choose() or self.choose(strict=False)
                if not choice:
                    return False
                self.ensure_cart(choice)
                if not self.cart:
                    return False
            if self.over_budget():
                choice = self.choose() or self.choose(strict=False) or self.closest()
                self.ensure_cart(choice)
                if not self.cart or self.over_budget():
                    return False
            done, notes = self.close(True)
            if done:
                return True
            for text in notes:
                if text not in self.heard:
                    self.heard.append(text)
                    self.inbox.append(text)
            if not notes:
                self.view_cart()
            self.after_change()
            choice = self.choose() or self.choose(strict=False)
            if choice:
                self.ensure_cart(choice)
        return self.store.finished

    def over_budget(self) -> bool:
        if self.budget is None:
            return False
        total = 0.0
        for ident, units in self.cart.items():
            price = self.price(ident)
            if price is not None:
                total += price * units
        return total > self.budget + 1e-6

    def decline(self) -> None:
        for _ in range(2):
            if self.store.finished:
                return
            done, notes = self.close(False)
            if done:
                return
            for text in notes:
                if text not in self.heard:
                    self.heard.append(text)
                    self.inbox.append(text)
            self.after_change()
            choice = self.choose()
            if choice:
                self.ensure_cart(choice)
                if self.finish_order():
                    return

    def closest(self) -> dict[int, str]:
        choice: dict[int, str] = {}
        for p in self.products:
            pool = [i for i in p.kind_ids if i in self.verdicts and i not in p.not_kind and self.in_stock(i) and self.affordable(i, p)]
            if pool:
                choice[p.index] = min(pool, key=lambda i: (self.standing(i)[0], self.price(i) or 1e12))
        return choice

    def play(self) -> list[dict[str, Any]]:
        if self.store.finished:
            return self.log
        self.read_request()
        self.survey()
        self.interview()
        self.follow_pointers()
        for p in self.products:
            if self.late():
                break
            self.judge(p, [i for i in p.kind_ids if i in self.props or i in self.pages])
            each = HUNT_OPENS if len(self.products) == 1 else HUNT_OPENS_EACH
            if not self.fits(p) or p.told or self.pick == "cheapest" or self.measure:
                self.hunt(p, each)
        choice = self.choose()
        missing = [p for p in self.products if p.index not in choice]
        for p in missing:
            if self.late() or self.steps_left() <= CLOSE_RESERVE + 4:
                break
            p.queries = [p.noun] + p.queries
            self.hunt(p, 4)
        choice = self.choose()
        self.note("pool", products=[{"p": p.index, "kind": len(p.kind_ids), "not": len(p.not_kind), "opened": sum(1 for i in p.kind_ids if i in self.props), "fits": [(i, self.price(i), self.standing(i)) for i in self.fits(p)[:4]], "judged": {i: {k: v.get("v") for k, v in self.verdicts[i].items() if k != "_n"} for i in p.kind_ids[:10] if i in self.verdicts}} for p in self.products], choice=choice, judged_all={p.index: {i: {k: (v.get("v"), v.get("quote", "")[:40]) for k, v in self.verdicts[i].items() if k != "_n"} for i in p.kind_ids if i in self.verdicts} for p in self.products})
        if len(choice) < len(self.products):
            self.loose = True
            loose = self.choose(strict=False)
            hard_no = [p for p in self.products if p.index not in choice and not any(self.standing(i)[0] == 0 for i in self.loose_fits(p))]
            if str(self.plan.get("when_nothing_fits") or "") == "decline" and hard_no or not loose:
                self.note("decline", missing=[p.index for p in self.products if p.index not in choice])
                self.decline()
                self.wrap_up()
                return self.log
            for index, ident in loose.items():
                choice.setdefault(index, ident)
            self.loose = True
        choice = self.link(choice)
        self.ensure_cart(choice)
        self.view_cart()
        if self.after_change():
            self.ensure_cart(self.choose() or choice)
        if not self.late() and self.steps_left() > 3:
            self.check_cart()
        if not self.finish_order():
            self.wrap_up()
        return self.log

    def wrap_up(self) -> None:
        if self.store.finished:
            return
        if not self.cart:
            choice = self.choose() or self.closest()
            for p in self.products:
                ident = choice.get(p.index)
                if ident:
                    self.put(ident, p.units)
        if self.cart:
            done, _ = self.close(True)
            if done:
                return
        self.close(False)


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        episode = Episode(problem_data)
    except Exception as exc:
        return [{"kind": "crash", "error": f"{type(exc).__name__}: {exc}"[:400]}]
    try:
        return episode.play()
    except Exception as exc:
        episode.log.append({"kind": "crash", "error": f"{type(exc).__name__}: {exc}"[:400]})
        try:
            episode.wrap_up()
        except Exception as again:
            episode.log.append({"kind": "crash", "error": f"{type(again).__name__}: {again}"[:400]})
        return episode.log


__all__ = ["agent_main"]
