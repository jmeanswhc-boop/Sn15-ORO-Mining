
from __future__ import annotations

import json
import math
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from src.agent.proxy_client import ProxyClient

HAIKU, SONNET = "anthropic/claude-haiku-5.5", "anthropic/claude-sonnet-5.5"
PRICE = {HAIKU: (0.1, 0.5, 0.01, 0.125), SONNET: (2.0, 10.0, 0.2, 2.5)}
TASK_MONEY, JUDGE_MONEY = 0.046, 0.015
WALK_S, TASK_S = 240.0, 290.0
RATE, BURST = 1.0, 5
TRACE_T0 = [time.monotonic()]


def trace_row(trace: list, kind: str, **fields: Any) -> None:
    return None

"""Every model seat in one place: the seat table (model and reasoning effort), then each seat's prompt and its output schema, in the order the walk calls them. Code elsewhere only fills the inputs."""

SEATS = {
    "MAP": (HAIKU, "low"),
    "INTAKE": (HAIKU, "high"),
    "INTAKE_TIE": (SONNET, "low"),
    "RECORD": (HAIKU, "medium"),
    "STORY": (SONNET, "low"),
    "READ": (HAIKU, "low"),
    "CHAIN": (HAIKU, "medium"),
    "JUDGE": (SONNET, "low"),
    "AUDIT": (SONNET, "low"),
    "ACTOR": (HAIKU, "medium"),
    "ROWS": (HAIKU, "low"),
    "MAP_ALT": (SONNET, "low"),
    "INTAKE_ALT": (SONNET, "low"),
}

MAP_TOOLS = """You map a shop's tools to roles for a shopping agent. You read only the tool list below and answer with one JSON object.

Roles, each null when no tool fits:
- search: free-text product search. query_arg = the free-text argument. page_arg = the argument that asks for further results (a token, a page number or a row offset), page_kind = "token", "page" or "offset". fixed = other required arguments with the value that selects plain text search (an enum value copied exactly), else {}. price_arg = an argument that caps the price, category_arg = an argument that limits results to one shelf or category.
- browse: a tool that lists a shelf or price range without free text (same fields as search; query_arg null).
- facts: a tool that returns the attribute table of one product. id_arg = its product id argument.
- page: a tool that returns the product page (description, small print, remarks) of one product. id_arg as above. The same tool may be both facts and page.
- compare: a tool that shows several products at once. ids_arg = its list argument, most = the largest number of ids it takes (null if not stated).
- cart_write: the tool that changes the basket. mode = "per_item" when one call adds or removes units of one product, "replace" when one call sets the whole basket from a list. For per_item: id_arg, qty_arg, op_arg (null if none), add_value and remove_value (enum values copied exactly, null if none). For replace: list_arg, item_id_key, item_qty_key. fixed = other required arguments with their value, else {}.
- cart_view: a tool that shows the basket.
- order: the call that buys the basket: tool and args (fixed values copied exactly, e.g. an enum value meaning order or commit).
- decline: the call that refuses the request: tool and args.
- message: the tool that sends a message to the shopper: tool and text_arg.

Copy tool names, argument names and enum values exactly as written. Example of the form (not of any shop): {"search": {"tool": "find", "query_arg": "q", "page_arg": "after", "page_kind": "token", "fixed": {}, "price_arg": null, "category_arg": null}, "browse": null, "facts": {"tool": "specs", "id_arg": "pid"}, "page": {"tool": "specs", "id_arg": "pid"}, "compare": null, "cart_write": {"tool": "basket", "mode": "per_item", "id_arg": "pid", "qty_arg": "n", "op_arg": "op", "add_value": "add", "remove_value": "drop", "list_arg": null, "item_id_key": null, "item_qty_key": null, "fixed": {}}, "cart_view": {"tool": "show_basket"}, "order": {"tool": "finish", "args": {"choice": "buy"}}, "decline": {"tool": "finish", "args": {"choice": "refuse"}}, "message": {"tool": "tell", "text_arg": "text"}}"""

MAP_FIELDS = """You read one search result and one product record of a shop and name what each field holds. Answer with one JSON object.

row = keys of a search result row: id (the product id used by other tools), title, price, stock (in-stock flag), category (shelf name), seller, brand, text (description shown in the row). result = top-level keys of the search result: rows (the list of products), count (the total number of matches, null if none), next (the value to send back for further results, null if none). record = keys of the product record: facts (the attribute table: a key holding key-value pairs, or null when the record is one text), small_print (fine print, remarks or notes), description. Use null for anything not present. When rows or the record are plain text, give null keys and set text_rows true.

Example of the form: {"row": {"id": "code", "title": "name", "price": "cost", "stock": "available", "category": "shelf", "seller": "shop", "brand": null, "text": "about"}, "result": {"rows": "items", "count": "total", "next": "cursor"}, "record": {"facts": "specs", "small_print": "notes", "description": "about"}, "text_rows": false}"""

ROWS = """You read one search result of a shop whose rows the code could not split, and list the products it shows. Answer with one JSON object only.

Input: RESULT, the search result exactly as the shop returned it (a long result is cut at a row or sentence end, and the count of characters left out is stated). A task scores 1 only when the exact products are bought, so every id is copied character for character.
1. Find every product the result lists, in the order shown.
2. id: the product's identifier exactly as written, the value other tools take to open or buy it (not its title, and not a model number inside its text).
3. title: the product's name as written. price: its price as the number written, null when none is shown. in_stock: true or false when the result says whether it can be bought now, else null.
4. next: the value the result gives for asking for further results (a cursor, token or page value), copied exactly; null when none is shown.
A value not written in RESULT is null; never invent one.
Form (not any shop's data): {"rows": [{"id": "<id as written>", "title": "<name as written>", "price": 1250, "in_stock": true}], "next": null}"""

ROWS_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["rows", "next"],
    "properties": {
        "rows": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "title", "price", "in_stock"],
                "properties": {"id": {"type": "string"}, "title": {"type": "string"}, "price": {"type": ["number", "null"]}, "in_stock": {"type": ["boolean", "null"]}},
            },
        },
        "next": {"type": ["string", "null"]},
    },
}

INTAKE = """You read one shopping request and write the brief a shop assistant works from. Answer with one JSON object only.

1. lines: one line per product the shopper wants bought. Count lines by quantities and requirements: products joined by "and" in one phrase with one quantity are one line; a product gets its own line only when it has its own quantity or its own requirements. Never add a product the request does not ask to buy. "a pair" of a thing sold in pairs is quantity 1.
2. For each line write item (the product phrase as the request names it), category (the shelf or category name when the request names one, else null), qty (a whole number; 1 when the request gives none), and wants: every requirement the request states for that product. A want has key (the attribute words as written), op (is, one_of, not, at_least, at_most, between, ref), values (the request's value words exactly: every alternative for one_of, the number and its unit as written for numbers) and quote (the request's exact words for it).
   - "up to X" about a value or a quantity is at_most X; "goes up to X" or "supports up to X" about what the product can do is at_least X.
   - A value restated in other words ("X, so ones where K is Y", "X, nice and Y") is one want with the first value only.
   - When the request names the value and then explains it after a colon or dash ("the X kind: A or B"), the want's value is X, not A or B. Only when the attribute alone stands before the colon ("colour: A or B") are the words after it the values.
   - A requirement taken from another listing ("the same as the one sold by …", "whatever the page of … says") is op ref; quote the whole pointer.
3. objective per line: "cheapest" only when the request asks for the lowest price, "best" (with best_key and best_dir max or min) when it asks for the most or least of an attribute, else "any". A sentence about budget is a cap, not an objective. "Lowest in total" makes total_objective "cheapest" and every line "cheapest".
4. cap_each per line when a price limit is for one unit of that product; total_cap when the limit is for the whole order. Write the number only (160, not "160 cents"), in the request's unit.
5. relations only when the request states them: fits (a line must fit, suit or be compatible with another line or with an identifier the request names; give the identifier), same or different (two lines share or differ in an attribute), if_then (if line a's attribute is X then line b's attribute must be Y; give both).
6. in_stock: true when the request rules out sold-out items. messages and steps: the numbers the request allows, null if not stated.
7. procedures: every instruction about how to act (ask before buying, report first, wait for approval, put every question in one message, explain a decline), each quoted. decline_rule: what the request says to do when nothing fits, quoted, else null.
8. story: every sentence about the shopper's situation, use, people or places rather than a requirement, copied whole, each with the line number it is about (null when about no single line).
9. checks: attributes the shopper says they will check or confirm themselves, with the value when one is given (null otherwise), each with its line.

Copy words exactly and never invent a value. Form (not any real request): {"lines": [{"item": "<item>", "category": null, "qty": 2, "objective": "cheapest", "best_key": null, "best_dir": null, "cap_each": null, "wants": [{"key": "<attribute>", "op": "is", "values": ["<value>"], "quote": "<request words>"}]}], "total_objective": "cheapest", "total_cap": 500, "in_stock": true, "messages": 3, "steps": 40, "relations": [], "procedures": [], "decline_rule": null, "story": [{"text": "<sentence>", "line": 1}], "checks": [{"line": 1, "key": "<attribute>", "value": null}]}"""

INTAKE_TIE = """Two readings of one shopping request disagree. Read the request yourself, decide which reading is right on the number of lines, the quantities and the caps (or correct both), and answer with the corrected brief as one JSON object in the same form as the readings."""

RECORD = """You judge shop records against a shopper's requirements. Answer with one JSON object only.

The grader reads the store's facts for each attribute, so prove every requirement with the store's own words, quoted exactly.
1. Evidence order: (a) own key: the key named exactly as the requirement's attribute, or that name after the item's name ("<item> <attribute>"), spelling of separators, case and singular/plural aside; (b) a sibling key: another key holding the same attribute (its last word is the attribute word, or it stands beside the own key with the same value, or, with no own key, it holds values of the same kind); a key beside the own key with a different value is another attribute; (c) options (not a list of variants on sale); (d) title; (e) the first sentence of the small print; (f) the description. The first place that holds a fact decides; when it says another value, later places are not used. Numbers count only from (a) or (b).
2. Label: PASS when a deciding fact says the required value, FAIL when it says only other values, UNKNOWN when nothing says it. Two spellings of one key: either saying it passes. Guesses, near values, wider classes and values only offered as choices are not proof.
3. Words: the requirement's words must stand in order inside one item of the fact (items split by commas, semicolons or slashes; words by spaces, underscores or hyphens; "a+b" is one item). Singular/plural, separators, case, regional spelling and an abbreviation in brackets beside the word are the same. Derived forms (<A> vs <A>ed), joined compounds, synonyms of other words and bare abbreviations are not the same, except an abbreviation the store itself uses as this attribute's value (see STORE USE). Extra words before or after are fine when they narrow the value ("<B> <A>" passes "<A>"); a different word in the middle fails ("<A> <C> <B>" does not pass "<A> <B>").
4. Lists pass only when they hold the value. A bound on a number passes only when the whole range is inside it; an exact value passes when one end of a range is that value. Convert units before arithmetic; read a key's first number. A number written in the requirement's unit beats a converted one.
5. "not X": PASS when the deciding fact says a value without X, FAIL when it holds X ("pu leather" holds "leather"), UNKNOWN when no fact says the attribute. A number bound with no number fact is UNKNOWN.
6. FAIL only when the record clearly shows another value; a record you cannot read for the attribute is UNKNOWN, never FAIL.
7. "fits" (op fits, values = identifiers of the product it must fit): PASS when the record's own words name one of those identifiers or a family that covers it ("every <code>xx model" covers <code>10, unless an exception names it), FAIL when the record names fit identifiers and none covers them, UNKNOWN when the record says nothing about fit. Quote the words that name the identifier.
STORE USE gives, per requirement, how the other opened records of this shelf write the attribute; use it to tell the store's vocabulary for that attribute.
For every record and every requirement number give label, where (own_key, sibling, options, title, small_print, description, none) and quote (the record's exact words that decide, empty when UNKNOWN).
Form: {"verdicts": [{"id": "<record id>", "want": 1, "label": "PASS", "where": "own_key", "quote": "<exact words>"}]}"""

RECORD_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["verdicts"],
    "properties": {
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "want", "label", "where", "quote"],
                "properties": {
                    "id": {"type": "string"},
                    "want": {"type": "integer"},
                    "label": {"type": "string", "enum": ["PASS", "FAIL", "UNKNOWN"]},
                    "where": {"type": "string", "enum": ["own_key", "sibling", "options", "title", "small_print", "description", "none"]},
                    "quote": {"type": "string"},
                },
            },
        }
    },
}

CHAIN = """A shopper's requirement takes its value from another listing. You read the pointer and the pages found so far and answer with one JSON object only.
- If one of the PAGES is the listing the pointer describes (every described value matches: seller, kind of product, stated values), say which and read what it gives: the value of the attribute the pointer asks for. When that page itself points on to another listing, give that pointer as next. When the page gives a pairing rule ("X goes with Y", "for X use Y"), the value is what the rule pairs with the shopper's case, not the page's own value.
- If none matches, say which words to search for to find it (seller name, kind of product, a described value), up to three short searches.
Form: {"match": "<page id or null>", "value": "<value or null>", "next": "<pointer text or null>", "search": ["<words>"]}"""

CHAIN_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["match", "value", "next", "search"],
    "properties": {"match": {"type": ["string", "null"]}, "value": {"type": ["string", "null"]}, "next": {"type": ["string", "null"]}, "search": {"type": "array", "items": {"type": "string"}}},
}

STORY = """A shopper's request has story sentences that may hint at needs they did not state. For each sentence, decide which attribute keys of the CANDIDATES (of the line the sentence is about) the sentence describes as a property of the product itself: at most two, best first, or none.
Branches: photos, looks, matching -> pattern, colour, finish; label, feel, washing -> material; fit, comfort, movement -> fit, size, length; someone else uses or wears it -> gender, age; weather, season -> material, lining; where it sits, how it connects -> mounting, connection; what it is for, the occasion or the part it plays -> style, use, occasion; carrying, storing -> packaging, capacity; speed, power -> that number's key; how it is made -> construction; the space it goes in -> its dimension keys. A sentence saying the shopper has not decided, does not mind, or wants to see the choices on some aspect of the product points at that aspect's key: give that key. A sentence that only says why something new is needed, or that repeats a requirement the request already states, gives none. Pick only keys from the CANDIDATES list of that line.
Form (two keys for one sentence are two picks with the same sentence number): {"picks": [{"sentence": 1, "line": 1, "key": "<candidate key or null>"}]}"""

STORY_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["picks"],
    "properties": {
        "picks": {
            "type": "array",
            "items": {"type": "object", "additionalProperties": False, "required": ["sentence", "line", "key"], "properties": {"sentence": {"type": "integer"}, "line": {"type": "integer"}, "key": {"type": ["string", "null"]}}},
        }
    },
}

READ = """You read one shopper reply to the assistant's message and say what it tells. Answer with one JSON object only.
kind: "told" (the shopper states a requirement), "no_preference" (they do not mind), "not_understood" (they did not follow, ask to repeat or say it more simply), "one_at_a_time" (they ask for one question at a time or refuse several at once), "other".
For told give every requirement stated: line (the number of the line whose product the reply names; the asked line when it names none), key (the attribute as the reply words it), op (is, one_of, not, at_least, at_most, between, ref), values (the reply's value words exactly; numbers with their unit), quote (the reply's exact words). op ref only when the reply names another product's page, listing, seller or record as the source of the value ("whatever is on the page of <seller>'s listing of <item>"); quote that whole pointer. A value said directly is never ref, whatever the verb: "<attribute> has got to match <value>" is op is with value <value>.
notices: any sentence of the reply that changes the order (budget, a requirement dropped, stock), quoted.
Form: {"kind": "told", "needs": [{"line": 1, "key": "<attribute>", "op": "is", "values": ["<value>"], "quote": "<reply words>"}], "notices": []}"""

READ_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["kind", "needs", "notices"],
    "properties": {
        "kind": {"type": "string", "enum": ["told", "no_preference", "not_understood", "one_at_a_time", "other"]},
        "needs": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["line", "key", "op", "values", "quote"],
                "properties": {
                    "line": {"type": "integer"},
                    "key": {"type": "string"},
                    "op": {"type": "string", "enum": ["is", "one_of", "not", "at_least", "at_most", "between", "ref"]},
                    "values": {"type": "array", "items": {"type": "string"}},
                    "quote": {"type": "string"},
                },
            },
        },
        "notices": {"type": "array", "items": {"type": "string"}},
    },
}

NOTICE = """Store or shopper notices change a shopping order. For each notice say what it changes now. change: "drop" (a requirement no longer applies; key), "cap" (new total price cap; value as a number), "cap_each" (new cap for one line; line, value), "add" (a new requirement; line, key, op, values), "stock" (an item is gone; id when given), "none". Copy keys and numbers exactly.
Form: {"changes": [{"line": 1, "change": "drop", "key": "<attribute>", "op": null, "values": [], "value": null, "id": null}]}"""

NOTICE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["changes"],
    "properties": {
        "changes": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["line", "change", "key", "op", "values", "value", "id"],
                "properties": {
                    "line": {"type": ["integer", "null"]},
                    "change": {"type": "string", "enum": ["drop", "cap", "cap_each", "add", "stock", "none"]},
                    "key": {"type": ["string", "null"]},
                    "op": {"type": ["string", "null"]},
                    "values": {"type": "array", "items": {"type": "string"}},
                    "value": {"type": ["number", "null"]},
                    "id": {"type": ["string", "null"]},
                },
            },
        }
    },
}

JUDGE = """You make the final check of a basket before it is ordered. A task scores 1 only when the finish is exactly right: every stated and disclosed requirement met, the cheapest or best option where asked, quantities and caps right, every stated procedure followed.
The record's shelf is the store's own kind of product and decides the kind: a record on the line's shelf is the right kind even when its title, a type fact or its description names another product word, and a record on a neighbouring shelf is not. NEVER reject a record on the line's shelf for its kind. Read every line's pick record whole, the requirements as they stand now (notices override the request), the cheaper or better records already seen, and judge with these rules: own key first, then sibling keys, options, title, small print, description; when the own key is written in two spellings (the same words with a space or an underscore, or in another case) and one of them states the value, the record passes; numbers only from keys; the requirement's words in order inside one fact item; derived forms only when no exact form exists; "not X" and number bounds need a fact; a record that cannot be read for a requirement does not pass it. For each line say ok, or the problem and which seen record to use instead (its id) when one fits better. List pointers: identifiers of other products the picks' own words point to ("see part …", "refer to …", "fits: …", "compatible part …").
Form: {"order": true, "lines": [{"line": 1, "ok": true, "problem": "", "use": null}], "pointers": ["<identifier>"]}"""

JUDGE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["order", "lines", "pointers"],
    "properties": {
        "order": {"type": "boolean"},
        "pointers": {"type": "array", "items": {"type": "string"}},
        "lines": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["line", "ok", "problem", "use"],
                "properties": {"line": {"type": "integer"}, "ok": {"type": "boolean"}, "problem": {"type": "string"}, "use": {"type": ["string", "null"]}},
            },
        },
    },
}

AUDIT = """The assistant wants to decline a shopping request because no basket passes. Check the evidence: records opened and judged, records only seen as rows. Allow the decline only when every line was searched and its likely candidates were opened; otherwise name up to four seen ids to open first. Form: {"decline": false, "open": ["<id>"]}"""

AUDIT_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["decline", "open"], "properties": {"decline": {"type": "boolean"}, "open": {"type": "array", "items": {"type": "string"}}}}

class Pace:

    def __init__(self) -> None:
        self.lock, self.tokens, self.at = threading.Lock(), 1.0, time.monotonic()

    def wait(self) -> None:
        while True:
            with self.lock:
                now = time.monotonic()
                self.tokens = min(float(BURST), self.tokens + (now - self.at) * RATE)
                self.at = now
                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return
                need = (1.0 - self.tokens) / RATE
            time.sleep(need)


class Env:

    def __init__(self, problem_data: dict, pace: Pace, trace: list) -> None:
        environment = problem_data["environment"]
        self.binding, self.view = dict(environment["binding"]), environment["policy_view"]
        self.step, self.finished, self.over = 0, bool(self.view.get("finished")), False
        self.shut: set = set()
        self.durations: list[float] = []
        self.max_steps = int(self.view.get("max_steps") or 0)
        self.pace, self.trace, self.client = pace, trace, ProxyClient(timeout=60, max_retries=2, retry_delay=1)
        self.t0 = time.monotonic()

    def left(self) -> int:
        return self.max_steps - self.step

    def elapsed(self) -> float:
        return time.monotonic() - self.t0

    def per_call(self) -> float:
        recent = self.durations[-10:]
        return sum(recent) / len(recent) if recent else 2.0

    def call(self, name: str, args: dict) -> dict:
        if self.finished or self.over or self.left() <= 0:
            return {"observation": None, "finished": self.finished, "error": "no step left" if not self.finished else "finished"}
        if name in self.shut:
            return {"observation": None, "finished": False, "error": "tool shut: its budget is spent"}
        payload = {**self.binding, "step": self.step + 1, "action": {"name": name, "args": args}}
        for attempt in range(5):
            if TASK_S + 8 - self.elapsed() < 3:
                trace_row(self.trace, "call_fail", step=payload["step"], tool=name, status="time")
                break
            self.client.timeout = max(3, min(60, int(TASK_S + 8 - self.elapsed())))
            self.pace.wait()
            sent = time.monotonic()
            result = self.client.post_verbose("/environment/call", payload)
            data = result.data
            if isinstance(data, dict):
                self.durations.append(time.monotonic() - sent)
                self.step = data["step"] if isinstance(data.get("step"), int) else payload["step"]
                self.finished = bool(data.get("finished"))
                obs = data.get("observation")
                error = obs.get("error") if isinstance(obs, dict) and isinstance(obs.get("error"), str) else None
                if error and "budget" in error.lower() and "exceed" in error.lower():
                    if self.left() <= 0:
                        self.over = True
                    else:
                        self.shut.add(name)
                        trace_row(self.trace, "meter_spent", tool=name, step=self.step)
                trace_row(self.trace, "call", step=self.step, tool=name, args=args, obs=obs, finished=self.finished)
                refusal = data.get("refusal")
                return {"observation": obs, "finished": self.finished, "error": error, "refusal": refusal if isinstance(refusal, list) else [refusal] if isinstance(refusal, str) else []}
            failure = result.error or {}
            status, body = failure.get("status"), str(failure.get("body") or "")
            trace_row(self.trace, "call_fail", step=payload["step"], tool=name, status=status, body=body[:300])
            if status == 409:
                low = body.lower()
                if any(word in low for word in ("finish", "quarantin", "finaliz")):
                    self.finished = True
                    break
                expected = re.search(r"expected (\d+)", body)
                if expected:
                    self.step = int(expected.group(1)) - 1
                else:
                    self.step += 1
                payload["step"] = self.step + 1
                continue
            if status in (400, 401, 402, 403, 404, 422):
                break
            time.sleep(min(8.0, 0.5 * 2**attempt) * (0.8 + 0.4 * random.random()))
        return {"observation": None, "finished": self.finished, "error": "transport"}

def parse_json(text: Any) -> Any:
    if isinstance(text, (dict, list)):
        return text
    if not isinstance(text, str):
        return None
    try:
        return json.loads(text)
    except ValueError:
        start, end = text.find("{"), text.rfind("}")
        if 0 <= start < end:
            try:
                return json.loads(text[start : end + 1])
            except ValueError:
                return None
    return None


class Brain:

    def __init__(self, env: Env, pace: Pace, trace: list) -> None:
        self.env, self.pace, self.trace = env, pace, trace
        self.spent, self.judged, self.clients, self.lock = 0.0, False, {}, threading.Lock()

    def client(self) -> ProxyClient:
        limit = int(max(8, min(60, TASK_S - self.env.elapsed() - 8)))
        if limit not in self.clients:
            self.clients[limit] = ProxyClient(timeout=limit, max_retries=1, retry_delay=1)
        return self.clients[limit]

    def room(self, seat: str) -> bool:
        if seat in ("JUDGE", "AUDIT"):
            return True
        return self.spent + (0.0 if self.judged else JUDGE_MONEY) < TASK_MONEY

    def price(self, model: str, usage: dict) -> float:
        upstream = (usage.get("cost_details") or {}).get("upstream_inference_cost")
        if isinstance(usage.get("cost"), (int, float)):
            return float(usage["cost"]) + (float(upstream) if isinstance(upstream, (int, float)) else 0.0)
        fresh, out, read, write = PRICE[model]
        details = usage.get("prompt_tokens_details") or {}
        cached, written = int(details.get("cached_tokens") or 0), int(usage.get("cache_write_tokens") or details.get("cache_write_tokens") or 0)
        prompt = int(usage.get("prompt_tokens") or 0)
        return ((prompt - cached - written) * fresh + cached * read + written * write + int(usage.get("completion_tokens") or 0) * out) / 1e6

    def send(self, model: str, effort: str, messages: list, schema: dict | None, name: str) -> tuple[str | None, str]:
        body: dict[str, Any] = {"model": model, "messages": messages, "reasoning": {"effort": effort}}
        if schema and schema.get("properties"):
            body["response_format"] = {"type": "json_schema", "json_schema": {"name": name, "strict": True, "schema": schema}}
        for attempt in range(3):
            if TASK_S - self.env.elapsed() < 10:
                return None, "time"
            self.pace.wait()
            result = self.client().post_verbose("/inference/chat/completions", body)
            data = result.data
            if isinstance(data, dict) and data.get("choices"):
                usage = data.get("usage") or {}
                cost = self.price(model, usage)
                with self.lock:
                    self.spent += cost
                choice = data["choices"][0]
                message = choice.get("message") or {}
                trace_row(self.trace, "model", seat=name, model=model, effort=effort, cost=round(cost, 5), usage=usage, finish=choice.get("finish_reason"), reply=message.get("content"))
                return message.get("content"), str(choice.get("finish_reason") or "")
            failure = result.error or {}
            status = failure.get("status")
            trace_row(self.trace, "model_fail", seat=name, model=model, status=status, body=str(failure.get("body") or "")[:300])
            if status not in (408, 425, 429, 500, 502, 503, 504, 529, None):
                return None, f"http {status}"
            time.sleep(min(6.0, 1.0 * 2**attempt) * (0.8 + 0.4 * random.random()))
        return None, "failed"

    def ask(self, seat: str, fixed: str, task: str, now: str, schema: dict, check=None) -> dict | None:
        model, effort = SEATS[seat]
        if not self.room(seat):
            if seat not in ("JUDGE", "AUDIT"):
                trace_row(self.trace, "money_stop", seat=seat, spent=round(self.spent, 4))
                return None
        if seat in ("JUDGE", "AUDIT") and self.spent >= TASK_MONEY:
            model, effort = HAIKU, "high"
        user = ([{"type": "text", "text": task, "cache_control": {"type": "ephemeral"}}] if task else []) + [{"type": "text", "text": now}]
        messages = [{"role": "system", "content": [{"type": "text", "text": fixed, "cache_control": {"type": "ephemeral"}}]}, {"role": "user", "content": user}]
        for turn in range(2):
            text, finish = self.send(model, effort, messages, schema, seat.lower())
            if text is None and turn == 0 and finish not in ("time",):
                other = SONNET if model == HAIKU else HAIKU
                text, finish = self.send(other, "low" if other == SONNET else "medium", messages, schema, seat.lower())
            answer = parse_json(text)
            problem = "the reply was not one JSON object" if not isinstance(answer, dict) else (check(answer) if check else None)
            if problem is None:
                if seat == "JUDGE":
                    self.judged = True
                return answer
            trace_row(self.trace, "reask", seat=seat, problem=problem)
            if text is None:
                return None
            messages = messages + [{"role": "assistant", "content": text}, {"role": "user", "content": f"Your answer cannot be used: {problem}. Answer again with only the JSON object."}]
        return None

def tool_props(tool: dict) -> dict:
    fn = tool.get("function", tool)
    return ((fn.get("parameters") or fn.get("input_schema") or {}).get("properties")) or {}


def words(text: Any) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", str(text).lower())


def kv_text(text: str) -> dict:
    found: dict = {}
    for part in re.split(r"\s*\|\s*|\n+", text):
        key, sep, value = part.partition(":")
        if sep and 0 < len(key.strip()) <= 40 and value.strip():
            inner = [pair.partition("=") for pair in value.split(";")]
            if sum(1 for k, eq, _ in inner if eq and k.strip()) >= 2:
                found.setdefault(key.strip(), {k.strip(): v.strip() for k, eq, v in inner if eq and k.strip()})
            else:
                found.setdefault(key.strip(), value.strip())
    return found


def cut_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    for pattern in (r"\n|\},(?=\s?\{)|\",(?=\s?\")", r"[.!?](?=\s)", r"\s\|\s|,\s", r"\s"):
        ends = [m.end() for m in re.finditer(pattern, text[:limit]) if m.end() > limit // 2]
        if ends:
            break
    end = ends[-1] if ends else limit
    return text[:end].rstrip() + f" …({len(text) - end} more characters left out)"


def raw_of(obs: Any) -> str:
    return obs if isinstance(obs, str) else json.dumps(obs, ensure_ascii=False, default=str)


class Store:

    def __init__(self, env: Env, brain: Brain, trace: list) -> None:
        self.env, self.brain, self.trace = env, brain, trace
        self.tools = {fn["name"]: t for t in env.view.get("tools") or [] if isinstance(t, dict) for fn in [t.get("function", t)] if isinstance(fn, dict) and fn.get("name")}
        self.roles: dict = {}
        self.fields: dict = {}
        self.cards: dict[str, dict] = {}
        self.records: dict[str, dict] = {}
        self.pages: dict[str, Any] = {}
        self.notices: list[str] = []
        self.tried: set = set()
        self.written: set = set()

    def learn_tools(self, seat: str = "MAP") -> None:
        listing = "\n".join(f"- {name}: {t.get('function', t).get('description', '')} | arguments: {json.dumps(tool_props(t), sort_keys=True)}" for name, t in sorted(self.tools.items()))

        def check(answer: dict) -> str | None:
            for role, spec in answer.items():
                if isinstance(spec, dict) and spec.get("tool") and spec["tool"] not in self.tools:
                    return f"{role}.tool {spec['tool']!r} is not a tool name"
            return None if isinstance(answer.get("search"), dict) else "no search tool named"

        answer = self.brain.ask(seat, MAP_TOOLS, "", "TOOLS:\n" + listing, {"type": "object"}, check) or {}
        for role, spec in answer.items():
            if isinstance(spec, dict) and spec.get("tool") in self.tools:
                props = tool_props(self.tools[spec["tool"]])
                for key in ("query_arg", "page_arg", "price_arg", "category_arg", "id_arg", "ids_arg", "qty_arg", "op_arg", "list_arg", "text_arg"):
                    if spec.get(key) and spec[key] not in props:
                        spec[key] = None
                self.roles[role] = spec
        trace_row(self.trace, "roles", roles=self.roles)

    def fit_args(self, tool: str, args: dict) -> dict:
        props, fitted = tool_props(self.tools.get(tool, {})), {}
        for key, value in args.items():
            spec = props.get(key)
            if spec is None or value is None:
                continue
            kind = spec.get("type")
            if spec.get("enum") and isinstance(value, str):
                match = [e for e in spec["enum"] if str(e).lower() == value.lower()]
                value = match[0] if match else value
            if kind in ("integer", "number") and isinstance(value, str) and re.fullmatch(r"-?\d+(\.\d+)?", value.strip()):
                value = int(float(value)) if kind == "integer" else float(value)
            if kind == "string" and isinstance(value, str) and spec.get("maxLength"):
                value = value[: int(spec["maxLength"])]
            fitted[key] = value
        return fitted

    def act(self, tool: str, args: dict) -> dict:
        fitted = self.fit_args(tool, args)
        result = self.env.call(tool, fitted)
        self.file_notices(result.get("observation"))
        for text in result.get("refusal") or []:
            if isinstance(text, str) and text.strip() and text not in self.notices:
                self.notices.append(text)
        if result.get("refusal"):
            trace_row(self.trace, "refusal", tool=tool, refusal=result["refusal"])
        self.watch(tool, fitted, result)
        return result

    def watch(self, tool: str, args: dict, result: dict) -> None:
        roles = [r for r, spec in self.roles.items() if isinstance(spec, dict) and spec.get("tool") == tool]
        error = str(result.get("error") or "")
        if any(r not in self.tried for r in roles) and re.search(r"invalid|unknown|schema|argument", error.lower()):
            trace_row(self.trace, "handoff", reason="schema error on the first call of a role", roles=roles, tool=tool, args=args, error=error)
        self.tried.update(roles)
        if error or result.get("observation") is None:
            return
        spec = self.roles.get("cart_write") or {}
        if "cart_write" in roles and spec.get("mode") == "replace" and spec.get("list_arg"):
            self.written = {str(i.get(spec.get("item_id_key") or "id")) for i in args.get(spec["list_arg"]) or [] if isinstance(i, dict)}
        elif "cart_write" in roles:
            removing = bool(spec.get("op_arg")) and args.get(spec["op_arg"]) == spec.get("remove_value")
            (self.written.discard if removing else self.written.add)(str(args.get(spec.get("id_arg") or "id")))
        if "cart_view" in roles:
            shown = raw_of(result["observation"])
            missing = sorted(i for i in self.written if i not in shown)
            if missing:
                trace_row(self.trace, "handoff", reason="cart view does not show an id written", missing=missing, view=shown)

    def file_notices(self, obs: Any) -> None:
        if isinstance(obs, dict):
            for key, value in obs.items():
                if isinstance(value, list) and value and all(isinstance(v, str) for v in value) and not self.looks_rows(value):
                    for text in value:
                        if len(words(text)) >= 5 and text not in self.notices:
                            self.notices.append(text)

    def looks_rows(self, items: list) -> bool:
        return sum(1 for v in items if len(kv_text(v)) >= 2) >= max(1, len(items) // 2)

    def rows_list(self, obs: Any) -> tuple[list, dict]:
        best, top = [], {}
        if isinstance(obs, dict):
            for key, value in obs.items():
                if isinstance(value, list) and len(value) > len(best) and (all(isinstance(v, dict) for v in value) or self.looks_rows(value)):
                    best = value
                elif isinstance(value, dict):
                    inner, _ = self.rows_list(value)
                    if len(inner) > len(best):
                        best = inner
                elif not isinstance(value, (list, dict)):
                    top[key] = value
        elif isinstance(obs, list):
            best = obs
        return best, top

    def learn_fields(self, obs: Any) -> None:
        rows, top = self.rows_list(obs)
        if not rows or self.fields:
            return
        sample = rows[0] if isinstance(rows[0], dict) else {"text": rows[0]}
        shown = json.dumps({"result_top_level": top, "first_row": sample}, ensure_ascii=False, sort_keys=True)[:3000]
        answer = self.brain.ask("MAP", MAP_FIELDS, "", "SEARCH RESULT:\n" + shown, {"type": "object"}) or {}
        self.fields = answer if isinstance(answer, dict) else {}
        trace_row(self.trace, "fields", fields=self.fields)

    def card_of(self, row: Any) -> dict | None:
        f = (self.fields.get("row") or {}) if isinstance(self.fields.get("row"), dict) else {}
        data = row if isinstance(row, dict) else kv_text(row) if isinstance(row, str) else {}
        if not data:
            return None

        def pick(role: str, kinds: tuple) -> Any:
            key = f.get(role)
            if key in data:
                return data[key]
            for k, v in data.items():
                if isinstance(v, kinds) and role in k.lower():
                    return v
            if role == "price":
                numbers = [v for v in data.values() if isinstance(v, (int, float)) and not isinstance(v, bool)]
                return numbers[0] if len(numbers) == 1 else None
            if role == "stock":
                flags = [v if isinstance(v, bool) else v.strip().lower() == "true" for v in data.values() if isinstance(v, bool) or (isinstance(v, str) and v.strip().lower() in ("true", "false"))]
                return flags[0] if len(flags) == 1 else None
            return None

        ident = pick("id", (str,))
        if not isinstance(ident, str) or " " in ident.strip():
            ident = next((v for v in data.values() if isinstance(v, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_\-:.]{3,40}", v) and re.search(r"\d", v)), None)
        if not ident:
            return None
        price = pick("price", (int, float, str))
        if isinstance(price, str):
            m = re.search(r"\d+(?:\.\d+)?", price.replace(",", ""))
            price = float(m.group(0)) if m else None
        stock = pick("stock", (bool, str))
        if isinstance(stock, str):
            stock = None if not stock.strip() else not re.search(r"\b(no|false|out|0)\b", stock.lower())
        text = " | ".join(f"{k}: {v}" for k, v in data.items() if isinstance(v, (str, int, float)) and v != ident) if isinstance(row, dict) else str(row)
        return {
            "id": ident.strip(),
            "title": str(pick("title", (str,)) or ""),
            "price": price,
            "stock": stock,
            "category": str(pick("category", (str,)) or ""),
            "seller": str(pick("seller", (str,)) or ""),
            "text": text,
            "facts": {},
            "small": "",
            "raw": row,
        }

    def search(self, query: str, more: Any = None, role: str = "search", extra: dict | None = None) -> dict:
        spec = self.roles.get(role) or {}
        if not spec.get("tool"):
            return {"cards": [], "count": None, "next": None, "error": "no tool"}
        args = dict(spec.get("fixed") or {})
        if spec.get("query_arg") and query:
            args[spec["query_arg"]] = query
        if more is not None and spec.get("page_arg"):
            args[spec["page_arg"]] = more
        args.update(extra or {})
        result = self.act(spec["tool"], args)
        obs = result.get("observation")
        self.learn_fields(obs)
        rows, top = self.rows_list(obs)
        cards = [c for c in (self.card_of(r) for r in rows) if c]
        read_next = None
        if not cards and not result.get("error") and obs is not None and (rows or len(raw_of(obs)) > 200):
            trace_row(self.trace, "handoff", reason="search result not empty but no card read", query=query, chars=len(raw_of(obs)))
            read, read_next = self.read_rows(obs)
            cards = [c for c in (self.card_of(r) for r in read) if c]
        for card in cards:
            self.cards.setdefault(card["id"], card)
        res = self.fields.get("result") or {}
        count = top.get(res.get("count")) if isinstance(res, dict) else None
        token = top.get(res.get("next")) if isinstance(res, dict) else None
        if token is None and read_next is not None:
            token = int(read_next) if spec.get("page_kind") in ("page", "offset") and read_next.isdigit() else read_next
        if token is None and spec.get("page_arg") and spec.get("page_kind") == "token":
            others = [v for k, v in top.items() if k != res.get("count") and isinstance(v, (str, int)) and not isinstance(v, bool) and v != count]
            token = others[0] if len(others) == 1 else None
        if spec.get("page_kind") == "offset" and isinstance(token, (int, float)) and token == 0:
            token = None
        if spec.get("page_kind") == "page" and token is None and cards and spec.get("page_arg"):
            token = (more or 1) + 1
        if spec.get("page_kind") == "offset" and token is None and cards and spec.get("page_arg"):
            token = (more or 0) + len(cards)
        if spec.get("page_kind") == "offset" and isinstance(token, (int, float)) and isinstance(count, (int, float)) and token >= count:
            token = None
        return {"cards": cards, "count": count if isinstance(count, (int, float)) else None, "next": token, "error": result.get("error")}

    def read_rows(self, obs: Any) -> tuple[list[dict], str | None]:
        raw = raw_of(obs)

        def check(answer: dict) -> str | None:
            if not isinstance(answer.get("rows"), list):
                return "rows is not a list"
            lost = [r.get("id") for r in answer["rows"] if isinstance(r, dict) and str(r.get("id") or "").strip() not in raw]
            return f"these ids are not written in RESULT: {lost[:5]}" if lost else None

        answer = self.brain.ask("ROWS", ROWS, "", "RESULT:\n" + cut_text(raw, 12000), ROWS_SCHEMA, check) or {}
        numbers, rows = set(numbers_in(raw)), []
        for r in answer.get("rows") or []:
            ident = str(r.get("id") or "").strip() if isinstance(r, dict) else ""
            if ident and ident in raw and all(x["id"] != ident for x in rows):
                price = r.get("price") if isinstance(r.get("price"), (int, float)) and float(r["price"]) in numbers else None
                rows.append({"id": ident, "title": str(r.get("title") or ""), "price": price, "in_stock": r["in_stock"] if isinstance(r.get("in_stock"), bool) else None})
        starts = sorted((raw.find(r["id"]), k) for k, r in enumerate(rows))
        for (at, k), until in zip(starts, [s for s, _ in starts[1:]] + [len(raw)]):
            rows[k]["text"] = cut_text(raw[at + len(rows[k]["id"]) : until], 800)
        token = answer.get("next")
        token = token if isinstance(token, str) and token.strip() and token in raw else None
        trace_row(self.trace, "rows", answered=len(answer.get("rows") or []), kept=[r["id"] for r in rows], next=token)
        return rows, token

    def read_card(self, ident: str, want_page: bool = False) -> dict | None:
        card = self.cards.setdefault(ident, {"id": ident, "title": "", "price": None, "stock": None, "category": "", "seller": "", "text": "", "facts": {}, "small": "", "raw": None})
        for role in ("facts", "page") if want_page else ("facts",):
            spec = self.roles.get(role) or self.roles.get("page" if role == "facts" else "facts")
            if not spec or not spec.get("tool") or (role == "page" and spec.get("tool") == (self.roles.get("facts") or {}).get("tool") and ident in self.records):
                continue
            result = self.act(spec["tool"], {spec.get("id_arg") or "id": ident})
            obs = result.get("observation")
            if result.get("error") or obs is None:
                continue
            self.absorb(card, obs, role)
            if role == "facts" and (self.roles.get("page") or {}).get("tool") == spec["tool"]:
                self.pages[card["id"]] = obs
        return card

    def absorb(self, card: dict, obs: Any, role: str) -> None:
        rec = self.fields.get("record") or {}
        data = obs if isinstance(obs, dict) else kv_text(obs) if isinstance(obs, str) else {}
        if isinstance(obs, dict) and isinstance(obs.get("observation"), (dict, str)):
            data = obs["observation"] if isinstance(obs["observation"], dict) else kv_text(obs["observation"])
        facts = data.get(rec.get("facts")) if isinstance(rec, dict) and rec.get("facts") else None
        if not isinstance(facts, dict):
            facts = next((v for v in data.values() if isinstance(v, dict) and len(v) >= 2), None)
        if isinstance(facts, dict):
            card["facts"].update({str(k): str(v) for k, v in facts.items() if v is not None})
        elif isinstance(obs, str):
            card["facts"].update(data)
        small = data.get(rec.get("small_print")) if isinstance(rec, dict) and rec.get("small_print") else None
        if isinstance(small, str):
            card["small"] = small
        elif not (isinstance(rec, dict) and rec.get("small_print")):
            known = {card.get("title"), card.get("category"), card.get("seller")}
            card["small"] = cut_text(" | ".join(f"{k}: {v}" for k, v in data.items() if isinstance(v, str) and v not in known and len(v) > 3), 1500)
        fresh = self.card_of({k: v for k, v in data.items() if not isinstance(v, dict)}) if isinstance(obs, dict) else None
        if fresh:
            for key in ("title", "price", "stock", "category", "seller"):
                if fresh.get(key) not in (None, ""):
                    card[key] = fresh[key]
            card["text"] = fresh["text"] or card["text"]
        raw = raw_of(obs)
        if not card["facts"] and len(raw) > 200:
            if not card.get("raw_text"):
                trace_row(self.trace, "handoff", reason="record without readable facts", id=card["id"], role=role, chars=len(raw))
            if raw[:200] not in card.get("raw_text", ""):
                card["raw_text"] = cut_text((card.get("raw_text", "") + "\n" + raw).strip(), 6000)
        if role == "page":
            self.pages[card["id"]] = obs
        else:
            self.records[card["id"]] = obs

def norm_text(text: Any) -> str:
    return " " + " ".join(re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", str(text).lower())) + " "


def numbers_in(text: Any) -> list[float]:
    return [float(n.replace(",", "")) for n in re.findall(r"\d[\d,]*(?:\.\d+)?", str(text))]


class Ledger:

    def __init__(self, request: str) -> None:
        self.request = request
        self.lines: list[dict] = []
        self.total_cap: float | None = None
        self.total_objective = "any"
        self.in_stock = True
        self.messages_left: int | None = None
        self.relations: list[dict] = []
        self.procedures: list[str] = []
        self.decline_rule: str | None = None
        self.story: list[dict] = []
        self.notes: list[str] = []
        self.refusals = 0

    def line(self, n: int) -> dict:
        return self.lines[n - 1]

    def wants(self, n: int) -> list[dict]:
        return [w for w in self.line(n)["wants"] if not w.get("dropped")]

    def state(self, env: Env) -> str:
        rows = [f"steps left {env.left()}, messages left {self.messages_left}, total cap {self.total_cap}"]
        for n, line in enumerate(self.lines, 1):
            wants = "; ".join(f"{w['key']} {w['op']} {'/'.join(map(str, w['values']))}" for w in self.wants(n)) or "-"
            rows.append(f"line {n}: {line['qty']} x {line['item']} ({line['objective']}) wants: {wants} | pick {line.get('pick')} | standby {line.get('standby')}")
        return "\n".join(rows)

def brief_problem(brief: dict, request: str) -> str | None:
    lines = brief.get("lines")
    if not isinstance(lines, list) or not lines:
        return "no lines"
    written = set(numbers_in(request))
    text = norm_text(request)
    for n, line in enumerate(lines, 1):
        if not isinstance(line, dict) or not isinstance(line.get("qty"), int) or line["qty"] < 1:
            return f"line {n} has no whole quantity"
        for want in line.get("wants") or []:
            quote = norm_text(want.get("quote", ""))
            if quote.strip() and quote not in text:
                return f"line {n} quote {want.get('quote')!r} is not the request's words"
    for cap in [brief.get("total_cap")] + [l.get("cap_each") for l in lines]:
        if isinstance(cap, (int, float)) and float(cap) not in written:
            return f"cap {cap} is not a number the request writes"
    return None


def same_shape(a: dict, b: dict) -> bool:

    def shape(x):
        return (len(x.get("lines") or []), [l.get("qty") for l in x.get("lines") or []], x.get("total_cap"), [l.get("cap_each") for l in x.get("lines") or []])

    return shape(a) == shape(b)


def read_request(brain: Brain, request: str, trace: list) -> dict | None:
    check = lambda answer: brief_problem(answer, request)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first, second = pool.map(lambda _: brain.ask("INTAKE", INTAKE, "", "REQUEST:\n" + request, {"type": "object"}, check), range(2))
    if first and second and same_shape(first, second):
        return first
    if first and second:
        tie = brain.ask("INTAKE_TIE", INTAKE + "\n\n" + INTAKE_TIE, "", f"REQUEST:\n{request}\n\nREADING A:\n{json.dumps(first, ensure_ascii=False)}\n\nREADING B:\n{json.dumps(second, ensure_ascii=False)}", {"type": "object"}, check)
        trace_row(trace, "intake_tie", a=first, b=second, chosen=tie)
        return tie or first
    return first or second


def load_brief(ledger: Ledger, brief: dict) -> None:
    for line in brief.get("lines") or []:
        wants = [dict(w, source="stated", values=[str(v) for v in (w.get("values") or [])]) for w in line.get("wants") or [] if isinstance(w, dict)]
        ledger.lines.append(
            {
                "item": str(line.get("item") or ""),
                "category": line.get("category"),
                "qty": int(line.get("qty") or 1),
                "objective": line.get("objective") if line.get("objective") in ("any", "cheapest", "best") else "any",
                "best_key": line.get("best_key"),
                "best_dir": line.get("best_dir"),
                "cap_each": line.get("cap_each"),
                "wants": wants,
                "verdicts": {},
                "pick": None,
                "standby": None,
                "seen": [],
                "asked": [],
                "told": [],
            }
        )
    ledger.total_cap = brief.get("total_cap") if isinstance(brief.get("total_cap"), (int, float)) else None
    ledger.total_objective = brief.get("total_objective") or "any"
    ledger.in_stock = brief.get("in_stock") is not False
    ledger.messages_left = brief.get("messages") if isinstance(brief.get("messages"), int) else None
    ledger.relations = [r for r in brief.get("relations") or [] if isinstance(r, dict)]
    ledger.procedures = [str(p) for p in brief.get("procedures") or []]
    ledger.decline_rule = brief.get("decline_rule")
    ledger.story = [s for s in brief.get("story") or [] if isinstance(s, dict) and s.get("text")]
    ledger.checks = [c for c in brief.get("checks") or [] if isinstance(c, dict)]

class Engine:

    def __init__(self, trace: list) -> None:
        self.trace = trace
        self.seen: list[dict] = []
        self.tier_votes: list[bool] = []
        self.facts: dict[str, Any] = {"or": None, "tiers": None, "hidden": None, "numbers": None, "first_words": None, "capped": None, "price_sorted": None}

    def note(self, query: str, cards: list[dict], count: Any, page: int) -> None:
        qwords = [w for w in words(query) if w not in FUNCTION_WORDS]
        rows = [norm_text(c.get("title", "") + " " + c.get("text", "") + " " + c.get("category", "")) for c in cards]
        matched = [sum(1 for w in qwords if any(f" {v} " in r for v in variants(w))) for r in rows]
        prices = [c["price"] for c in cards if isinstance(c.get("price"), (int, float))]
        self.seen.append({"q": " ".join(qwords), "words": qwords, "ids": [c["id"] for c in cards], "matched": matched, "count": count, "page": page, "prices": prices})
        f = self.facts
        if len(qwords) >= 2 and rows:
            partial = sum(1 for m in matched if 0 < m < len(qwords))
            if partial:
                f["or"] = True
            elif f["or"] is None and all(m == len(qwords) or m == 0 for m in matched) and page == 0:
                f["or"] = False
            if len(matched) >= 4 and partial:
                half = len(matched) // 2
                self.tier_votes.append(sum(matched[:half]) / half > sum(matched[half:]) / (len(matched) - half))
                if len(self.tier_votes) >= 3:
                    share = sum(self.tier_votes) / len(self.tier_votes)
                    f["tiers"] = True if share >= 0.7 else False if share <= 0.3 else None
        if qwords and rows:
            blind = sum(1 for m in matched if m == 0)
            if blind:
                f["hidden"] = True
        if page == 0 and cards:
            bare = " ".join(w for w in qwords if not w.replace(".", "").isdigit())
            numeric = bare != " ".join(qwords)
            for s in self.seen[:-1]:
                s_bare = " ".join(w for w in s["words"] if not w.replace(".", "").isdigit())
                if s["page"] == 0 and s["ids"] and s_bare == bare and (s_bare != s["q"]) != numeric:
                    f["numbers"] = s["ids"] != [c["id"] for c in cards]
                    break
        if page == 0 and len(qwords) >= 2:
            for s in self.seen[:-1]:
                same = lambda a, b: bool(variants(a) & variants(b))
                if s["page"] == 0 and s["ids"] and s["ids"] == [c["id"] for c in cards] and not (len(s["words"]) == len(qwords) and all(same(a, b) for a, b in zip(s["words"], qwords))):
                    k = next((i for i, (a, b) in enumerate(zip(s["words"], qwords)) if not same(a, b)), min(len(s["words"]), len(qwords)))
                    extra = (qwords if len(qwords) > len(s["words"]) else s["words"])[k:]
                    rare = [w for w in extra if sum(1 for r in rows if any(f" {v} " in r for v in variants(w))) * 2 < len(rows)]
                    if 1 <= k < max(len(s["words"]), len(qwords)) and rare:
                        f["first_words"] = k if not isinstance(f["first_words"], int) else min(f["first_words"], k)
        firsts = {s["q"]: s["count"] for s in self.seen if s["page"] == 0 and isinstance(s["count"], (int, float))}
        if len(firsts) >= 3:
            f["capped"] = len(set(firsts.values())) == 1
        if len(prices) >= 4:
            f["price_sorted"] = all(a <= b for a, b in zip(prices, prices[1:]))
        trace_row(self.trace, "engine", query=query, page=page, matched=matched[:12], facts=f)

    def tier_dropped(self, query: str, cards: list[dict]) -> bool:
        qwords = [w for w in words(query) if w not in FUNCTION_WORDS]
        if not (self.facts["or"] and self.facts["tiers"]) or len(qwords) < 2 or not cards:
            return False
        last = norm_text(cards[-1].get("title", "") + " " + cards[-1].get("text", ""))
        return sum(1 for w in qwords if any(f" {v} " in last for v in variants(w))) < len(qwords)

    def searchable(self, value: str) -> str:
        if self.facts["numbers"] is False:
            return " ".join(p for p in value.split() if not re.fullmatch(r"[\d.,]+", p))
        return value

FUNCTION_WORDS = {"a", "an", "the", "and", "or", "of", "for", "with", "in", "on", "to", "by", "x"}


def variants(word: str) -> set[str]:
    w = word.lower()
    out = {w}
    if w.endswith("ies") and len(w) > 4:
        out.add(w[:-3] + "y")
    elif w.endswith("es") and len(w) > 4 and w[-3] in "sxz":
        out.add(w[:-2])
    elif w.endswith("s") and len(w) > 3 and not w.endswith("ss"):
        out.add(w[:-1])
    else:
        out.add(w + "s")
    return out


def unpossessed(text: Any) -> str:
    return re.sub(r"['’]s\b", "", str(text))


def item_words(line: dict) -> list[str]:
    found = []
    for text in (line.get("item") or "", line.get("category") or ""):
        for w in words(text):
            if w not in FUNCTION_WORDS and len(w) > 2 and not w.replace(".", "").isdigit() and w not in found:
                found.append(w)
    return found


def kind_words(line: dict) -> list[str]:
    held = {x for want in line.get("wants") or [] if not want.get("dropped") and want.get("op") in ("is", "one_of") for v in want.get("values") or [] for w in words(unpossessed(v)) for x in variants(w)}
    item = item_words({"item": line.get("item")})
    named = [w for w in item_words({"category": line.get("category")}) if w not in item and not variants(w) & held]
    return ([w for w in item if not variants(w) & held] or item) + named


def value_words(line: dict, wants: list[dict]) -> list[tuple[int, str]]:
    found = []
    for k, want in enumerate(wants):
        if want.get("op") in ("is", "one_of", "at_least", "at_most", "between"):
            for value in want.get("values") or []:
                text = " ".join(str(value).split())
                if text and all(w.replace(".", "").isdigit() for w in words(text)):
                    unit = " ".join(w for w in key_words(want.get("key", "")).split() if w not in FUNCTION_WORDS)
                    text = f"{text} {unit}".strip()
                if text and all(text.lower() != v.lower() for _, v in found):
                    found.append((k, text))
    return found


class Gather:

    def __init__(self, store: Store, ledger: Ledger, trace: list, engine: Engine | None = None) -> None:
        self.store, self.ledger, self.trace = store, ledger, trace
        self.engine = engine or Engine(trace)
        self.done_queries: set = set()
        self.cursors: dict[tuple, dict] = {}
        self.hits: dict[int, dict[str, int]] = {}
        self.cover: dict[int, dict[str, set]] = {}
        self.capped: set = set()
        self.stop_at = 10**9

    def queries(self, line: dict, values: list[str]) -> list[tuple[str, bool]]:
        spec = self.store.roles.get("search") or {}
        props = tool_props(self.store.tools.get(spec.get("tool"), {}))
        limit = int((props.get(spec.get("query_arg") or "", {}) or {}).get("maxLength") or 60)
        head, value_held = self.head(line), set(item_words(line)) - set(kind_words(line))
        phrase = " ".join(w for w in words(unpossessed(line.get("item") or "")) if w not in FUNCTION_WORDS and w not in value_held)[:limit].strip()
        k = self.engine.facts.get("first_words")
        if isinstance(k, int) and k >= 1:
            return self.short_queries(line, values, head, k, limit)
        out = []
        texts = [self.engine.searchable(v) for _, v in values]
        wants_n = len({k for k, _ in values})
        joined = " ".join(dict.fromkeys(w for v in texts for w in v.split()))
        if wants_n >= 2 and joined:
            text = joined if head in set(words(joined)) else f"{joined} {head}".strip()
            out.append((text[:limit].rsplit(" ", 1)[0] if len(text) > limit else text, True))
        item_set = {x for w in item_words(line) for x in variants(w)}
        for value in texts:
            if not value.strip():
                continue
            vwords = words(value)
            text = value if head and head in variants(vwords[-1] if vwords else "") | set(vwords) else f"{value} {head}".strip()
            if text != value and (len(vwords) >= 2 or set(vwords) & item_set):
                out.append((value[:limit], True))
            if len(text) > limit:
                text = value[:limit]
            out.append((text, True))
        out.append((phrase or head, False))
        return [q for q in out if q[0]]

    def run(self, n: int, pages: int, resume: bool = False) -> list[str]:
        line = self.ledger.line(n)
        values = value_words(line, self.ledger.wants(n))
        hits = self.hits.setdefault(n, {})
        cover = self.cover.setdefault(n, {})
        later = sum(1 for k in range(n + 1, len(self.ledger.lines) + 1) if not self.ledger.line(k)["seen"])
        budget = max(3, (self.store.env.left() - self.reserve_steps()) // (1 + later))
        self.stop_at = self.store.env.step + budget
        if not line["seen"]:
            phrase = [q for q in self.queries(line, []) if not q[1]]
            for text, _ in phrase[:1]:
                if (text, n) not in self.done_queries:
                    self.done_queries.add((text, n))
                    result = self.store.search(text)
                    self.engine.note(text, result["cards"], result["count"], 0)
                    for card in result["cards"]:
                        if card["id"] not in line["seen"]:
                            line["seen"].append(card["id"])
                    self.advance(n, text, result, values)
            if line["seen"] and self.file_category(n, line):
                values = value_words(line, self.ledger.wants(n))
        for text, by_value in self.queries(line, values):
            if (text, n) not in self.done_queries:
                self.done_queries.add((text, n))
                self.read(n, text, by_value, values, pages)
        self.browse(n, line, values, pages, hits)
        self.widen(n, line, values, hits, pages)
        if resume:
            self.resume(n, within_share=True)
        if self.engine.facts.get("capped"):
            self.capped.add(n)
        kept = [i for i in line["seen"] if self.keep(line, self.store.cards[i])]
        order = sorted(kept, key=lambda i: self.score(self.store.cards[i], values, hits.get(i, 10**6), line["seen"].index(i), line, cover.get(i, set())))
        trace_row(self.trace, "gather", line=n, values=values, seen=len(line["seen"]), kept=len(kept), capped=n in self.capped, order=order[:12])
        return order

    def read(self, n: int, text: str, by_value: bool, values: list, pages: int, cap: float | None = None) -> None:
        line = self.ledger.line(n)
        hits, cover = self.hits.setdefault(n, {}), self.cover.setdefault(n, {})
        covered = {k for k, v in values if set(words(self.engine.searchable(v))) <= set(words(text))} if by_value else set()
        first = (n, text) not in self.cursors
        for _ in range(pages):
            cur = self.cursors.get((n, text))
            if (cur and cur["ended"] and not first) or self.out_of_steps():
                break
            page = cur["page"] if cur else 0
            result = self.store.search(text, cur["next"] if cur else None)
            self.engine.note(text, result["cards"], result["count"], page)
            if page == 0 and not result["cards"] and by_value:
                self.advance(n, text, result, values, by_value)
                self.retry_empty(n, line, text, values)
                break
            new = [c["id"] for c in result["cards"] if c["id"] not in line["seen"]]
            for rank, card in enumerate(result["cards"]):
                if card["id"] not in line["seen"]:
                    line["seen"].append(card["id"])
                if by_value:
                    hits[card["id"]] = min(hits.get(card["id"], 10**6), page * len(result["cards"]) + rank)
                    cover.setdefault(card["id"], set()).update(covered)
            cur = self.advance(n, text, result, values, by_value)
            token = result["next"]
            shown_value = any(f" {' '.join(words(v))} " in norm_text(c["text"] + " " + c["title"]) for c in result["cards"] for _, v in values)
            if cap is not None and cur["sorted"]:
                cur["ended"] = cur["ended"] or (isinstance(cur["last"], (int, float)) and cur["last"] > cap)
                why = "search ended or past the cap" if cur["ended"] else None
            else:
                why = (
                    "no new rows"
                    if not new
                    else (
                        "no next page"
                        if token is None
                        else "value not shown" if (by_value and page >= 2 and not shown_value and not self.engine.facts["hidden"]) else "tier dropped" if self.engine.tier_dropped(text, result["cards"]) else None
                    )
                )
            if why:
                trace_row(self.trace, "page_stop", line=n, query=text, page=page, why=why)
                break

    def advance(self, n: int, text: str, result: dict, values: list, by_value: bool = False) -> dict:
        line = self.ledger.line(n)
        cur = self.cursors.setdefault((n, text), {"next": None, "page": 0, "rows": 0, "held": 0, "sorted": True, "last": None, "value": False, "ids": None})
        cards = result["cards"]
        ids = [c["id"] for c in cards]
        prices = ([cur["last"]] if cur["last"] is not None else []) + [c["price"] for c in cards if isinstance(c.get("price"), (int, float))]
        cur["held"] += sum(1 for c in cards if self.keep(line, self.store.cards.get(c["id"], c)) and any(f" {' '.join(words(v))} " in norm_text(c["text"] + " " + c["title"]) for _, v in values))
        count, again = result["count"], ids == cur["ids"]
        cur.update(
            next=result["next"], page=cur["page"] + 1, rows=cur["rows"] + len(cards), value=cur["value"] or by_value, ids=ids, sorted=cur["sorted"] and all(a <= b for a, b in zip(prices, prices[1:])), last=prices[-1] if prices else None
        )
        cur["ended"] = not cards or again or cur["next"] is None or (isinstance(count, (int, float)) and cur["rows"] >= count and not self.engine.facts.get("capped"))
        return cur

    def resume(self, n: int, within_share: bool = False) -> list[str] | None:
        line = self.ledger.line(n)
        live = [(text, cur) for (m, text), cur in self.cursors.items() if m == n and not cur["ended"]]
        if not live:
            return None
        cheapest = line.get("objective") == "cheapest"
        text, cur = max(live, key=lambda tc: (cheapest and tc[1]["sorted"], tc[1]["held"] / max(1, tc[1]["rows"]), -tc[1]["page"]))
        values, seen, page = value_words(line, self.ledger.wants(n)), set(line["seen"]), cur["page"]
        if not within_share:
            self.stop_at = 10**9
        self.read(n, text, cur["value"], values, 1, self.cover_cap(line))
        if cur["page"] == page:
            return None
        hits, cover = self.hits.get(n, {}), self.cover.get(n, {})
        new = [i for i in line["seen"] if i not in seen and self.keep(line, self.store.cards[i])]
        trace_row(self.trace, "dig_page", line=n, query=text, page=page, new=len(new), ended=cur["ended"])
        return sorted(new, key=lambda i: self.score(self.store.cards[i], values, hits.get(i, 10**6), line["seen"].index(i), line, cover.get(i, set())))

    def cover_cap(self, line: dict) -> float | None:
        return self.line_cap(line) if self.engine.facts.get("price_sorted") else None

    def line_cap(self, line: dict) -> float | None:
        cap = line.get("cap_each")
        if not isinstance(cap, (int, float)) and isinstance(self.ledger.total_cap, (int, float)):
            cap = self.ledger.total_cap / max(1, line["qty"])
        return cap if isinstance(cap, (int, float)) else None

    def walk_over(self) -> bool:
        env = self.store.env
        return env.left() <= self.reserve_steps() or env.elapsed() > WALK_S or env.elapsed() + (self.reserve_steps() + 1) * env.per_call() > TASK_S - 10

    def out_of_steps(self) -> bool:
        return self.walk_over() or self.store.env.step >= self.stop_at

    def retry_empty(self, n: int, line: dict, text: str, values: list[str]) -> None:
        head = self.head(line)
        ws = [w for w in words(text) if w not in FUNCTION_WORDS and not (head and w in variants(head))]
        digits = [w for w in ws if w.replace(".", "").isdigit()]
        letters = [w for w in ws if not w.replace(".", "").isdigit()]
        tries = []
        if digits and letters:
            tries += [" ".join(digits + [head]).strip(), " ".join(letters + [head]).strip()]
        if len(letters) >= 2:
            tries.append(" ".join(letters[:-1] + [head]).strip())
        for alt in tries:
            if not alt or (alt, n) in self.done_queries or self.out_of_steps():
                continue
            self.done_queries.add((alt, n))
            result = self.store.search(alt)
            self.engine.note(alt, result["cards"], result["count"], 0)
            for card in result["cards"]:
                if card["id"] not in line["seen"]:
                    line["seen"].append(card["id"])
            self.advance(n, alt, result, values)
            trace_row(self.trace, "retry_empty", line=n, query=text, alt=alt, rows=len(result["cards"]))
            if result["cards"]:
                return

    def widen(self, n: int, line: dict, values: list[str], hits: dict, pages: int) -> None:
        shelf = self.listed_shelf(line)
        rows = [self.store.cards[i] for i in line["seen"] if self.store.cards.get(i, {}).get("category") == shelf] if shelf else []
        head_words = item_words(line)
        head = self.head(line)
        for _, value in values:
            vw = [w for w in words(value) if w not in FUNCTION_WORDS]
            if not vw or any(hits.get(c["id"], 10**6) < 10**6 for c in rows if any(f" {w} " in norm_text(c.get("text", "") + " " + c.get("title", "")) for w in vw)):
                continue
            near: dict[str, int] = {}
            for c in rows:
                text = norm_text(c.get("title", "") + " " + c.get("text", ""))
                if any(f" {w} " in text for w in vw if not w.replace(".", "").isdigit()) or (vw and f" {' '.join(vw)} " in text):
                    for w in set(text.split()):
                        if len(w) > 3 and w not in FUNCTION_WORDS and w not in vw and not w.replace(".", "").isdigit() and w not in {x for h in head_words for x in variants(h)}:
                            near[w] = near.get(w, 0) + 1
            for w, _ in sorted(near.items(), key=lambda kv: (-kv[1], kv[0]))[:2]:
                text = f"{w} {head}".strip()
                if (text, n) in self.done_queries or self.out_of_steps():
                    continue
                self.done_queries.add((text, n))
                result = self.store.search(text)
                self.engine.note(text, result["cards"], result["count"], 0)
                for card in result["cards"]:
                    if card["id"] not in line["seen"]:
                        line["seen"].append(card["id"])
                self.advance(n, text, result, values)
                trace_row(self.trace, "widen", line=n, value=value, word=w)

    def short_queries(self, line: dict, values: list[str], head: str, k: int, limit: int) -> list[tuple[str, bool]]:
        seen_text = [norm_text((self.store.cards.get(i) or {}).get("title", "") + " " + (self.store.cards.get(i) or {}).get("text", "")) for i in line.get("seen") or []]
        rarity = lambda w: sum(1 for t in seen_text if f" {w} " in t)
        out = []
        pool = [w for _, v in values for w in words(self.engine.searchable(v)) if w not in FUNCTION_WORDS]
        if len({k for k, _ in values}) >= 2 and pool:
            out.append((" ".join(sorted(dict.fromkeys(pool), key=lambda w: (rarity(w), pool.index(w)))[:k])[:limit], True))
        for _, value in values:
            vw = [w for w in words(self.engine.searchable(value)) if w not in FUNCTION_WORDS]
            if not vw:
                continue
            chosen = sorted(vw, key=lambda w: (rarity(w), vw.index(w)))[: max(1, k - 1)]
            chosen = [w for w in vw if w in chosen]
            text = " ".join(chosen + ([head] if head and head not in chosen and len(chosen) < k else []))
            out.append((text[:limit], True))
        out.append((head, False))
        return out

    def head(self, line: dict) -> str:
        candidates = kind_words(line)
        if not candidates:
            return ""
        seen = [self.store.cards.get(i) or {} for i in line.get("seen") or []]
        counts = {w: 0 for w in candidates}
        for field in ("title", "category"):
            for card in seen:
                text = norm_text(card.get(field, ""))
                for w in candidates:
                    counts[w] += any(f" {v} " in text for v in variants(w))
            if any(counts.values()):
                break
        shelf = self.shelf(line)
        nouns = [w for w in kind_words(dict(line, category=None)) if shelf and any(variants(w) & variants(x) for x in words(shelf))]
        if nouns:
            rows = [(norm_text(c.get("category", "")) == norm_text(shelf), norm_text(c.get("title", "") + " " + c.get("category", ""))) for c in seen]
            carry = lambda w, on: sum(1 for here, t in rows if here == on and any(f" {v} " in t for v in variants(w)))
            word = min(nouns, key=lambda w: (carry(w, False) > carry(w, True), -counts[w], -nouns.index(w)))
        else:
            best = max(candidates, key=lambda w: (counts[w], candidates.index(w)))
            word = best if counts[best] else candidates[-1]
        spelled: dict[str, int] = {}
        for i in line.get("seen") or []:
            card = self.store.cards.get(i) or {}
            for form in sorted(variants(word)):
                spelled[form] = spelled.get(form, 0) + norm_text(card.get("category", "") + " " + card.get("title", "")).count(f" {form} ")
        return max(spelled, key=lambda f: (spelled[f], f == word, f)) if spelled and max(spelled.values()) else word

    def shelf(self, line: dict) -> str | None:
        order, names, within = self.shelf_terms(line)
        item, value_held = names(line.get("item") or ""), set(item_words(line)) - set(kind_words(line))
        need = [w for w in item if not w.replace(".", "").isdigit() and w not in value_held]
        allowed = item + [w for want in line.get("wants") or [] if not want.get("dropped") and want.get("op") in ("is", "one_of") for v in want.get("values") or [] for w in names(v)]
        return self.named_shelf(line) or next((c for c in order if need and within(need, names(c)) and within(names(c), allowed)), None)

    def shelf_terms(self, line: dict) -> tuple:
        counts: dict[str, int] = {}
        for i in line.get("seen") or []:
            cat = (self.store.cards.get(i) or {}).get("category") or ""
            if cat:
                counts[cat] = counts.get(cat, 0) + 1
        names = lambda text: [w for w in words(unpossessed(text)) if w not in FUNCTION_WORDS]
        within = lambda part, whole: all(any(variants(w) & variants(x) for x in whole) for w in part)
        return sorted(counts, key=lambda c: (-counts[c], c)), names, within

    def named_shelf(self, line: dict) -> str | None:
        order, names, within = self.shelf_terms(line)
        named = names(line.get("category") or "")
        core = [w for w in named if any(within([w], names(c)) for c in order)]
        same = lambda c, ws: ws and within(names(c), ws) and within(ws, names(c))
        return next((c for c in order if same(c, named)), None) or next((c for c in order if same(c, core)), None)

    def file_category(self, n: int, line: dict) -> bool:
        _, names, within = self.shelf_terms(line)
        phrase = " ".join(names(line.get("category") or ""))
        if not phrase or within(phrase.split(), names(line.get("item") or "")) or self.named_shelf(line) or any(w.get("source") == "category" for w in line["wants"]):
            return False
        line["wants"].append({"key": "type", "op": "is", "values": [phrase], "quote": str(line.get("category")), "source": "category"})
        trace_row(self.trace, "category_want", line=n, phrase=phrase)
        return True

    def listed_shelf(self, line: dict) -> str | None:
        bound = self.shelf(line)
        if bound:
            return bound
        counts: dict[str, int] = {}
        for i in line["seen"]:
            card = self.store.cards.get(i) or {}
            cat = card.get("category") or ""
            text = norm_text(card.get("title", "") + " " + cat)
            if cat and any(f" {v} " in text for w in kind_words(line) for v in variants(w)):
                counts[cat] = counts.get(cat, 0) + 1
        return max(counts, key=counts.get) if counts else None

    def browse(self, n: int, line: dict, values: list[str], pages: int, hits: dict) -> None:
        spec = self.store.roles.get("browse") or {}
        shelf = self.listed_shelf(line)
        if not spec.get("tool") or not shelf or not spec.get("category_arg"):
            return
        cap = self.line_cap(line)
        key = ("browse", shelf, cap)
        if key in self.done_queries:
            return
        self.done_queries.add(key)
        extra = {spec["category_arg"]: shelf, **({spec["price_arg"]: cap} if spec.get("price_arg") and isinstance(cap, (int, float)) else {})}
        token = None
        for page in range(max(2, pages) * 3):
            if self.out_of_steps():
                break
            result = self.store.search("", token, role="browse", extra=extra)
            new = [c["id"] for c in result["cards"] if c["id"] not in line["seen"]]
            for card in result["cards"]:
                if card["id"] not in line["seen"]:
                    line["seen"].append(card["id"])
            token = result["next"]
            if not new or token is None:
                break

    def keep(self, line: dict, card: dict) -> bool:
        text = norm_text(" ".join([card.get("title", ""), card.get("category", ""), card.get("text", "")]))
        bound, shelf = self.shelf(line), self.listed_shelf(line)
        has_item = any(f" {v} " in text for w in kind_words(line) for v in variants(w)) or (shelf is not None and card.get("category") == shelf)
        elsewhere = bool(bound and card.get("category")) and norm_text(card["category"]) != norm_text(bound)
        cap = line.get("cap_each")
        over = isinstance(cap, (int, float)) and isinstance(card.get("price"), (int, float)) and card["price"] > cap
        sold = self.ledger.in_stock and card.get("stock") is False
        return has_item and not elsewhere and not over and not sold

    def score(self, card: dict, values: list, best: int, position: int, line: dict, covered: set) -> tuple:
        text = norm_text(card.get("title", "") + " " + card.get("text", ""))
        shown = {k for k, v in values if f" {' '.join(words(v))} " in text}
        level = len(shown | covered)
        price = card["price"] if isinstance(card.get("price"), (int, float)) else float("inf")
        if line.get("objective") == "cheapest":
            return (-level, price, position)
        return (-level, best, position, price)

    def reserve_steps(self) -> int:
        lines = max(1, len(self.ledger.lines))
        roles = self.store.roles
        cart = 1 if (roles.get("cart_write") or {}).get("mode") == "replace" else lines
        pages = lines if (roles.get("page") or {}).get("tool") not in (None, (roles.get("facts") or {}).get("tool")) else 0
        return (lines + pages + cart + 2) + (lines + 2 * cart + 1) + 1 + (self.ledger.messages_left or 0)

def key_words(key: str) -> str:
    return norm_text(re.sub(r"(?<=[a-z])(?=[A-Z])", " ", str(key)))


def key_form(words_: list[str]) -> str:
    out = []
    for w in words_:
        w = sorted(variants(w))[0]
        out.append(w[:-3] + "or" if len(w) >= 5 and w.endswith("our") else w[:-2] + "er" if len(w) >= 5 and w.endswith("re") and w[-3] not in "aeiouy" else w)
    return " ".join(out)


def own_keys(card: dict, want: dict, line: dict) -> list[str]:
    target = key_form(key_words(want.get("key", "")).split())
    items = {v for w in item_words(line) for v in variants(w)}
    found = []
    for key in card.get("facts") or {}:
        kw = key_words(key).split()
        if target and target in (key_form(kw), key_form([w for w in kw if w not in items])):
            found.append(key)
    return found


def fact_items(value: str) -> list[str]:
    return [p for p in re.split(r"[,;/]", str(value)) if p.strip()]


def holds(item: str, value: str) -> bool:
    have = norm_text(item)
    need = value_variants(value)
    return any(v in have for v in need)


def value_variants(value: str) -> list[str]:
    ws = words(value)
    if not ws:
        return []
    forms = [" " + " ".join(ws) + " "]
    last = sorted(variants(ws[-1]) - {ws[-1]})
    forms += [" " + " ".join(ws[:-1] + [alt]) + " " for alt in last]
    return forms


def code_verdict(card: dict, want: dict, line: dict) -> tuple[str, str, str] | None:
    keys = own_keys(card, want, line)
    op, values = want.get("op"), [str(v) for v in want.get("values") or []]
    shelf = str(card.get("category") or "")
    if op in ("is", "one_of") and shelf and any(holds(unpossessed(shelf), unpossessed(v)) for v in values):
        return ("PASS", "shelf", shelf)
    if not keys or op not in ("is", "one_of", "not") or not values:
        return None
    facts = card["facts"]
    for key in keys:
        for item in fact_items(facts[key]):
            if "+" in item:
                if any(norm_text(item) == form for v in values for form in value_variants(v)):
                    return ("FAIL" if op == "not" else "PASS", "own_key", f"{key}: {facts[key]}")
                return None
            if any(holds(item, v) for v in values):
                return ("FAIL" if op == "not" else "PASS", "own_key", f"{key}: {facts[key]}")
    if len(keys) == 1:
        value = str(facts[keys[0]])
        if len(fact_items(value)) == 1 and len(words(value)) <= 4 and not re.search(r"\d", value + " ".join(values)):
            shared = set(words(value)) & {w for v in values for w in words(v)}
            if not shared:
                return ("PASS" if op == "not" else "FAIL", "own_key", f"{keys[0]}: {value}")
    return None


def store_use(cards: list[dict], want: dict, line: dict) -> str:
    with_key, holding, examples = 0, 0, []
    for card in cards:
        keys = own_keys(card, want, line)
        if keys:
            with_key += 1
            value = str(card["facts"][keys[0]])
            holding += any(holds(i, v) for i in fact_items(value) for v in want.get("values") or [])
            if value not in examples and len(examples) < 6:
                examples.append(value[:60])
    return f"{with_key} of {len(cards)} opened records have the key; {holding} write the value; values seen: {examples}" + (" (few records)" if len(cards) < 4 else "")


def card_view(card: dict) -> str:
    small = str(card.get("small") or "")
    first = re.split(r"(?<=[.!?])\s+", small)[0] if small else ""
    text = str(card.get("text") or "")
    lead = " ".join(re.split(r"(?<=[.!?])\s+", text)[:3])
    facts = "; ".join(f"{k}: {v}" for k, v in (card.get("facts") or {}).items())
    return (
        f"ID {card['id']} | title: {card.get('title')} | shelf: {card.get('category')} | price: {card.get('price')} | in stock: {card.get('stock')}\n"
        f"  facts: {facts or ('(none read) the record as written: ' + card['raw_text'] if card.get('raw_text') else '(none)')}\n  small print: {first}\n  row text: {lead}"
        + (f" (+{len(text) - len(lead)} chars)" if len(text) > len(lead) else "")
    )


class Judge:

    def __init__(self, brain: Brain, store: Store, ledger: Ledger, trace: list) -> None:
        self.brain, self.store, self.ledger, self.trace = brain, store, ledger, trace

    def file(self, n: int, ident: str, k: Any, verdict: tuple, open_pairs: set | None = None) -> None:
        verdicts, wants = self.ledger.line(n)["verdicts"], self.ledger.wants(n)
        tag = want_tag(wants[k - 1]) if isinstance(k, int) and 1 <= k <= len(wants) else None
        kept = verdicts.get(ident, {}).get(tag)
        if tag is None or kept or (open_pairs is not None and (ident, k) not in open_pairs):
            trace_row(self.trace, "judge_drop", line=n, id=ident, want=k, verdict=verdict, kept=kept)
            return
        verdicts.setdefault(ident, {})[tag] = verdict

    def judge(self, n: int, ids: list[str]) -> None:
        line, wants = self.ledger.line(n), self.ledger.wants(n)
        verdicts = line["verdicts"]
        pending: dict[str, list[int]] = {}
        for i in dict.fromkeys(ids):
            got = verdicts.setdefault(i, {})
            for k, want in enumerate(wants, 1):
                if want_tag(want) in got or want.get("op") == "ref":
                    continue
                cv = code_verdict(self.store.cards[i], want, line)
                if cv:
                    self.file(n, i, k, cv)
                else:
                    pending.setdefault(i, []).append(k)
        if not pending:
            return
        open_pairs = {(i, k) for i, ks in pending.items() for k in ks}
        shelf = [self.store.cards[i] for i in line["verdicts"] if self.store.cards.get(i, {}).get("facts")]
        asked = sorted({k for ks in pending.values() for k in ks})
        reqs = "\n".join(f"{k}. {wants[k - 1]['key']} {wants[k - 1]['op']} {' | '.join(wants[k - 1]['values'])}  (STORE USE: {store_use(shelf, wants[k - 1], line)})" for k in asked)
        task = f"ITEM: {line['item']}\nREQUIREMENTS:\n{reqs}"
        now = "RECORDS:\n" + "\n".join(card_view(self.store.cards[i]) + f"\n  requirements to judge: {', '.join(map(str, ks))}" for i, ks in pending.items())
        answer = self.brain.ask("RECORD", RECORD, task, now, RECORD_SCHEMA) or {}
        for v in answer.get("verdicts") or []:
            i, k = str(v.get("id")), v.get("want") if isinstance(v.get("want"), int) else None
            label, quote = v.get("label"), str(v.get("quote") or "")
            if (i, k) in open_pairs and label in ("PASS", "FAIL") and norm_text(quote).strip() and norm_text(quote) not in norm_text(card_view(self.store.cards[i])):
                label = "UNKNOWN"
            self.file(n, i, k, (label, v.get("where") or "none", quote), open_pairs)
        trace_row(self.trace, "judge", line=n, verdicts={i: verdicts.get(i) for i in ids})

    def status(self, n: int, ident: str) -> str:
        got = self.ledger.line(n)["verdicts"].get(ident, {})
        labels = [(want, "UNKNOWN" if want.get("op") == "ref" else got.get(want_tag(want), ("UNKNOWN",))[0]) for want in self.ledger.wants(n)]
        if any(l == "FAIL" for _, l in labels):
            return "FAIL"
        if all(l == "PASS" for _, l in labels):
            return "PASS"
        if all(l == "PASS" or w.get("op") in ("not", "at_least", "at_most", "between", "ref") for w, l in labels):
            return "WEAK"
        return "UNKNOWN"


def want_tag(want: dict) -> str:
    return f"{want.get('key')}|{want.get('op')}|{'/'.join(map(str, want.get('values') or []))}"

EVIDENCE_RANK = {"own_key": 0, "sibling": 1, "options": 1, "shelf": 1, "title": 2, "small_print": 3, "description": 3, "none": 4}


def number_of(text: Any) -> float | None:
    found = numbers_in(text)
    return found[0] if found else None


class Picker:

    def __init__(self, brain: Brain, store: Store, ledger: Ledger, judge: Judge, gather: Gather, trace: list) -> None:
        self.brain, self.store, self.ledger, self.judge, self.gather, self.trace = brain, store, ledger, judge, gather, trace

    def strength(self, n: int, ident: str) -> int:
        got = self.ledger.line(n)["verdicts"].get(ident, {})
        return max([EVIDENCE_RANK.get(v[1], 4) for v in got.values() if v[0] == "PASS"] or [4])

    def ranked(self, n: int) -> list[str]:
        line = self.ledger.line(n)
        cards = [i for i in line["verdicts"] if self.ok_card(line, self.store.cards.get(i, {})) and i not in line.get("vetoed", ())]
        passing = [i for i in cards if self.judge.status(n, i) == "PASS"] or [i for i in cards if self.judge.status(n, i) == "WEAK"]
        price = lambda i: self.store.cards[i]["price"] if isinstance(self.store.cards[i].get("price"), (int, float)) else float("inf")
        if line["objective"] == "best" and line.get("best_key"):
            want = {"key": line["best_key"]}

            def best_value(i):
                keys = own_keys(self.store.cards[i], want, line)
                value = number_of(self.store.cards[i]["facts"][keys[0]]) if keys else None
                return value if value is not None else (float("-inf") if line.get("best_dir") != "min" else float("inf"))

            return sorted(passing, key=lambda i: ((-best_value(i)) if line.get("best_dir") != "min" else best_value(i), price(i)))
        if line["objective"] == "cheapest":
            return sorted(passing, key=lambda i: (price(i), self.strength(n, i)))
        return sorted(passing, key=lambda i: (self.strength(n, i), price(i)))

    def ok_card(self, line: dict, card: dict) -> bool:
        price = card.get("price") if isinstance(card.get("price"), (int, float)) else None
        total = self.ledger.total_cap
        return self.gather.keep(line, card) and not (price is not None and isinstance(total, (int, float)) and price * line["qty"] > total)

    def fit_total(self) -> None:
        total = self.ledger.total_cap
        if not isinstance(total, (int, float)):
            return
        price = lambda i: self.store.cards[i]["price"] if i and isinstance(self.store.cards[i].get("price"), (int, float)) else 0
        for _ in range(12):
            lines = self.ledger.lines
            if sum(price(l.get("pick")) * l["qty"] for l in lines) <= total:
                return
            options = []
            for n, line in enumerate(lines, 1):
                cheaper = [i for i in self.ranked(n) if price(i) < price(line.get("pick"))]
                if cheaper:
                    best = min(cheaper, key=price)
                    options.append(((price(line["pick"]) - price(best)) * line["qty"], n, best))
            if not options:
                return
            _, n, best = max(options)
            self.ledger.line(n)["pick"] = best

    def choose(self, n: int) -> None:
        order = self.ranked(n)
        line = self.ledger.line(n)
        if line.get("pinned") in order:
            order = [line["pinned"]] + [i for i in order if i != line["pinned"]]
        line["pick"], line["standby"] = (order[0] if order else None), (order[1] if len(order) > 1 else None)

    def open_and_judge(self, n: int, ids: list[str], page: bool = False) -> None:
        fresh = [i for i in ids if i not in self.store.records]
        compare = self.store.roles.get("compare") or {}
        if len(fresh) > 1 and compare.get("tool") and compare.get("ids_arg"):
            most = compare.get("most") or 4
            for k in range(0, len(fresh), int(most)):
                if self.gather.walk_over():
                    break
                result = self.store.act(compare["tool"], {compare["ids_arg"]: fresh[k : k + int(most)]})
                rows, _ = self.store.rows_list(result.get("observation"))
                for row in rows:
                    card = self.store.card_of(row)
                    if card and card["id"] in self.store.cards and isinstance(row, (dict, str)):
                        self.store.absorb(self.store.cards[card["id"]], row, "facts")
        for i in fresh:
            if not self.store.cards.get(i, {}).get("facts") and not self.gather.walk_over():
                self.store.read_card(i, page)
        self.judge.judge(n, [i for i in ids if self.store.cards.get(i, {}).get("facts") or i in self.store.records])

    def follow_chain(self, n: int, want: dict) -> None:
        pointer = want.get("quote") or " ".join(want.get("values") or [])
        searched: set = set()
        for hop in range(4):
            pages = [i for i in self.store.records][-8:]
            view = "\n".join(card_view(self.store.cards[i]) for i in pages if i in self.store.cards)
            answer = self.brain.ask("CHAIN", CHAIN, f"POINTER: {pointer}\nATTRIBUTE: {want.get('key')}", "PAGES:\n" + (view or "(none yet)"), CHAIN_SCHEMA) or {}
            if answer.get("match") and answer.get("value"):
                if answer.get("next"):
                    pointer = answer["next"]
                    continue
                want.update(op="is", values=[answer["value"]], source="chain")
                trace_row(self.trace, "chain", line=n, want=want)
                return
            for text in (answer.get("search") or [])[:3]:
                if text in searched or self.store.env.left() <= self.gather.reserve_steps():
                    continue
                searched.add(text)
                result = self.store.search(text)
                for card in result["cards"][:4]:
                    if card["id"] not in self.store.records and self.store.env.left() > self.gather.reserve_steps():
                        self.store.read_card(card["id"], True)
        trace_row(self.trace, "chain_open", line=n, want=want)

def same_key(a: str, b: str) -> bool:
    canon = lambda k: sorted(sorted(variants(w))[0] for w in key_words(k).split())
    return bool(key_words(a).strip()) and canon(a) == canon(b)


class Asker:

    def __init__(self, brain: Brain, store: Store, ledger: Ledger, trace: list) -> None:
        self.brain, self.store, self.ledger, self.trace = brain, store, ledger, trace
        self.queue: dict[int, list[str]] = {}
        self.bundles = None

    def table(self, n: int) -> list[tuple[str, int, int]]:
        line = self.ledger.line(n)
        held: dict[str, dict[str, str]] = {}
        names: dict[str, str] = {}
        for i in line["verdicts"]:
            card = self.store.cards.get(i, {})
            stated = {k for w in line["wants"] for k in own_keys(card, w, line)}
            for key, value in (card.get("facts") or {}).items():
                kw = key_words(key)
                if kw.strip() in ("options", "option") or key in stated or not kw.strip():
                    continue
                names.setdefault(kw, key)
                held.setdefault(kw, {}).setdefault(i, norm_text(value))
        order = sorted(held, key=lambda kw: (-len(held[kw]), -len(set(held[kw].values())), kw))
        kept: list[str] = []
        for kw in order:
            twin = next((k for k in kept if len(set(held[k]) & set(held[kw])) >= 2 and all(held[k][c] == held[kw][c] for c in set(held[k]) & set(held[kw]))), None)
            if twin is None:
                kept.append(kw)
        return [(names[kw], len(held[kw]), len(set(held[kw].values()))) for kw in kept]

    def plan(self) -> None:
        tables = {n: self.table(n) for n in range(1, len(self.ledger.lines) + 1)}
        story = self.ledger.story
        lifted: dict[int, list[str]] = {n: [] for n in tables}
        checks = [
            {"text": f"The shopper will check the {c['key']} themselves.", "line": c.get("line")} for c in getattr(self.ledger, "checks", []) if isinstance(c, dict) and c.get("key") and not c.get("value")
        ]
        story = checks + story
        if story:
            listing = "\n".join(f"line {n}: {self.ledger.line(n)['item']} | CANDIDATES: {[r[0] for r in rows[:12]]}" for n, rows in tables.items())
            sentences = "\n".join(f"{k}. (line {s.get('line')}) {s['text']}" for k, s in enumerate(story, 1))
            answer = self.brain.ask("STORY", STORY, "LINES:\n" + listing, "SENTENCES:\n" + sentences, STORY_SCHEMA) or {}
            for pick in sorted(answer.get("picks") or [], key=lambda p: p.get("sentence") or 0):
                n, key = pick.get("line"), pick.get("key")
                if n in tables and key and any(key == r[0] for r in tables[n]) and key not in lifted[n]:
                    lifted[n].append(key)
        for n, rows in tables.items():
            self.queue[n] = lifted[n] + [r[0] for r in rows if r[0] not in lifted[n]]
        self.bundles = self.ledger.messages_left == 3
        trace_row(self.trace, "ask_plan", queue=self.queue, bundles=self.bundles)

    def order(self) -> list[int]:
        told = lambda n: bool(self.ledger.line(n)["told"])
        return sorted(self.queue, key=lambda n: (told(n), len(self.ledger.line(n)["asked"]), n))

    def name(self, key: str, n: int, drop: set | None = None) -> str:
        items = {v for w in item_words(self.ledger.line(n)) for v in variants(w)}
        kept = [w for w in key_words(key).split() if w not in items and w not in (drop or set())]
        return " ".join(kept) or key_words(key).strip()

    def text(self, n: int, keys: list[str]) -> str:
        item = self.ledger.line(n)["item"]
        names = [self.name(k, n) for k in keys]
        if len(names) == 1:
            return f"What {names[0]} would you like for the {item}?"
        return f"What {', '.join(names[:-1])} or {names[-1]} would you like for the {item}?"

    def say(self, text: str) -> str | None:
        spec = self.store.roles.get("message") or {}
        if not spec.get("tool") or not self.ledger.messages_left:
            return None
        result = self.store.act(spec["tool"], {spec.get("text_arg") or "text": text})
        self.ledger.messages_left -= 1
        if result.get("error") and ("budget" in result["error"].lower() or "shut" in result["error"].lower()):
            self.ledger.messages_left = 0
            return None
        obs = result.get("observation")
        return obs if isinstance(obs, str) else json.dumps(obs, ensure_ascii=False) if obs is not None else None

    def read(self, n: int, asked: list[str], reply: str) -> dict:
        lines = "\n".join(f"line {k}: {l['item']}" for k, l in enumerate(self.ledger.lines, 1))
        answer = self.brain.ask("READ", READ, "LINES:\n" + lines, f"ASKED (line {n}): {asked}\nREPLY: {reply}", READ_SCHEMA) or {"kind": "other", "needs": [], "notices": []}
        reply_words = norm_text(reply)
        answer["needs"] = [d for d in answer.get("needs") or [] if all(norm_text(v).strip() and norm_text(v) in reply_words for v in d.get("values") or [])]
        names = [r[0] for r in self.table(n)] + list(asked)
        for need in answer["needs"]:
            said = key_words(need.get("key") or "")
            if said.strip() and said not in reply_words:
                near = [c for c in names if key_words(c).strip() and key_words(c) in reply_words and {v for w in key_words(c).split() for v in variants(w)} & {v for w in said.split() for v in variants(w)}]
                if near:
                    need["key"] = max(near, key=lambda c: len(key_words(c)))
        standing = [w for line in self.ledger.lines for w in line["wants"] if not w.get("dropped")]
        same_values = lambda w, d: sorted(norm_text(v) for v in w.get("values") or []) == sorted(norm_text(v) for v in d.get("values") or [])
        repeat = lambda d: key_words(d.get("key") or "") not in reply_words and any(key_words(w.get("key") or "").strip() and key_words(w["key"]) in reply_words and same_values(w, d) for w in standing)
        answer["needs"] = [d for d in answer["needs"] if not repeat(d)]
        for need in answer["needs"]:
            if need.get("op") == "ref" and len(words(need.get("quote") or "")) <= len(words(" ".join(need.get("values") or []))) + 4:
                need["op"] = "is"
        return answer

    def answered(self, n: int, keys: list[str], reply: str, needs: list[dict]) -> int:
        told = [i for i, k in enumerate(keys) if any(same_key(k, d.get("key") or "") for d in needs)]
        if told:
            return max(told)
        said = norm_text(reply)
        return next((i for i, k in enumerate(keys) if norm_text(self.name(k, n)).strip() and norm_text(self.name(k, n)) in said), 0)

    def ask_all(self, regather) -> None:
        if self.bundles is None:
            self.plan()
        while self.ledger.messages_left and any(self.queue.values()) and self.store.env.left() > 4 and self.store.env.elapsed() < WALK_S:
            n = next((k for k in self.order() if self.queue[k]), None)
            if n is None:
                break
            keys = self.queue[n][:4] if self.bundles else self.queue[n][:1]
            question = self.text(n, keys)
            reply = self.say(question)
            if reply is None:
                break
            answer = self.read(n, keys, reply)
            line = self.ledger.line(n)
            line["asked"].append(keys[0])
            trace_row(self.trace, "asked", line=n, keys=keys, question=question, reply=reply, read=answer)
            self.ledger.notes += answer.get("notices") or []
            if answer["kind"] == "told":
                for need in answer["needs"]:
                    m = need.get("line") if isinstance(need.get("line"), int) and 1 <= need["line"] <= len(self.ledger.lines) else n
                    target = self.ledger.line(m)
                    target["wants"].append({"key": need["key"], "op": need["op"], "values": need["values"], "quote": need.get("quote"), "source": "disclosed"})
                    target["told"].append(need["key"])
                    self.queue[m] = [k for k in self.queue[m] if not same_key(k, need["key"])]
                    regather(m)
                if self.bundles:
                    self.queue[n] = [k for k in self.queue[n] if k not in keys[: self.answered(n, keys, reply, answer["needs"]) + 1]]
                else:
                    self.queue[n] = self.queue[n][1:]
            elif answer["kind"] == "one_at_a_time" or (answer["kind"] == "not_understood" and len(keys) > 1):
                if len(keys) > 1:
                    self.bundles = False
                    continue
                others = {w for k in self.queue[n][1:6] for w in key_words(k).split()}
                narrow = self.name(keys[0], n, others)
                if self.ledger.messages_left and narrow and narrow != self.name(keys[0], n):
                    reply = self.say(f"What {narrow} would you like for the {line['item']}?")
                    if reply:
                        again = self.read(n, keys, reply)
                        for need in again.get("needs") or [] if again.get("kind") == "told" else []:
                            line["wants"].append({"key": need["key"], "op": need["op"], "values": need["values"], "quote": need.get("quote"), "source": "disclosed"})
                            line["told"].append(need["key"])
                            regather(n)
                self.queue[n] = self.queue[n][1:]
            elif self.bundles:
                self.queue[n] = [k for k in self.queue[n] if k not in keys[: self.answered(n, keys, reply, []) + 1]]
            else:
                self.queue[n] = self.queue[n][1:]

class Closer:

    def __init__(self, brain: Brain, store: Store, ledger: Ledger, picker: Picker, trace: list) -> None:
        self.brain, self.store, self.ledger, self.picker, self.trace = brain, store, ledger, picker, trace
        self.applied: set = set()
        self.in_cart: dict[str, int] = {}

    def apply_notices(self) -> bool:
        fresh = [t for t in self.store.notices + self.ledger.notes if t not in self.applied]
        self.applied.update(fresh)
        if not fresh:
            return False
        lines = "\n".join(f"line {n}: {l['item']} wants {[(w['key'], w['op'], w['values']) for w in self.ledger.wants(n)]}" for n, l in enumerate(self.ledger.lines, 1))
        answer = self.brain.ask("READ", NOTICE, "LINES:\n" + lines, "NOTICES:\n" + "\n".join(fresh), NOTICE_SCHEMA) or {}
        changed = False
        for c in answer.get("changes") or []:
            n = c.get("line") if isinstance(c.get("line"), int) and 1 <= c["line"] <= len(self.ledger.lines) else None
            if c["change"] == "drop" and c.get("key"):
                for m in ([n] if n else range(1, len(self.ledger.lines) + 1)):
                    for w in self.ledger.line(m)["wants"]:
                        if set(key_words(w["key"]).split()) & set(key_words(c["key"]).split()):
                            w["dropped"], changed = True, True
            elif c["change"] == "cap" and isinstance(c.get("value"), (int, float)):
                self.ledger.total_cap, changed = c["value"], True
            elif c["change"] == "cap_each" and n and isinstance(c.get("value"), (int, float)):
                self.ledger.line(n)["cap_each"], changed = c["value"], True
            elif c["change"] == "add" and n and c.get("key"):
                self.ledger.line(n)["wants"].append({"key": c["key"], "op": c.get("op") or "is", "values": c.get("values") or [], "source": "notice"})
                changed = True
            elif c["change"] == "stock" and c.get("id") in self.store.cards:
                self.store.cards[c["id"]]["stock"], changed = False, True
        trace_row(self.trace, "notices", notices=fresh, changes=answer.get("changes"))
        return changed

    def spare(self, calls: int) -> bool:
        env = self.store.env
        return env.left() > calls and env.elapsed() + (calls + 1) * env.per_call() < TASK_S - 5

    def write_cart(self) -> None:
        spec = self.store.roles.get("cart_write") or {}
        want = {}
        for line in self.ledger.lines:
            if line.get("pick"):
                want[line["pick"]] = want.get(line["pick"], 0) + line["qty"]
        if not spec.get("tool"):
            return
        fixed = dict(spec.get("fixed") or {})
        if spec.get("mode") == "replace" and spec.get("list_arg"):
            items = [{spec.get("item_id_key") or "id": i, spec.get("item_qty_key") or "qty": q} for i, q in want.items()]
            self.store.act(spec["tool"], {**fixed, spec["list_arg"]: items})
        else:
            qkey = spec.get("qty_arg") or "qty"
            qschema = tool_props(self.store.tools.get(spec["tool"], {})).get(qkey) or {}
            by_op = bool(spec.get("op_arg")) and spec.get("remove_value") is not None
            by_zero = not by_op and isinstance(qschema, dict) and qschema.get("minimum") == 0
            stale = [i for i, q in self.in_cart.items() if want.get(i) != q]
            if stale and not (by_op or by_zero):
                trace_row(self.trace, "handoff", reason="cart write cannot remove an item", items=stale, spec=spec)
            for ident in stale:
                if (by_op or by_zero) and self.spare(1):
                    remove = {spec["op_arg"]: spec["remove_value"], qkey: self.in_cart[ident]} if by_op else {qkey: 0}
                    self.store.act(spec["tool"], {**fixed, spec.get("id_arg") or "id": ident, **remove})
            for ident, qty in want.items():
                if self.in_cart.get(ident) != qty and self.spare(1):
                    self.store.act(spec["tool"], {**fixed, spec.get("id_arg") or "id": ident, spec.get("qty_arg") or "qty": qty, **({spec["op_arg"]: spec["add_value"]} if spec.get("op_arg") and spec.get("add_value") is not None else {})})
        self.in_cart = want
        for ident in want:
            if self.spare(2):
                self.store.read_card(ident)
        view = self.store.roles.get("cart_view") or {}
        if view.get("tool") and self.spare(1):
            self.store.act(view["tool"], {})

    def judge_basket(self) -> dict:
        parts = []
        for n, line in enumerate(self.ledger.lines, 1):
            pick = line.get("pick")
            better = [i for i in self.picker.ranked(n) if i != pick][:3]
            wants = "; ".join(f"{w['key']} {w['op']} {w['values']}" for w in self.ledger.wants(n))
            parts.append(
                f"LINE {n}: {line['qty']} x {line['item']} ({line['objective']}) cap {line.get('cap_each')}\n  requirements now: {wants}\n"
                f"  PICK: {card_view(self.store.cards[pick]) if pick else '(none)'}\n  OTHER PASSING: " + " || ".join(card_view(self.store.cards[i]) for i in better)
            )
        task = f"REQUEST:\n{self.ledger.request}\n\nTOTAL CAP: {self.ledger.total_cap}\nNOTICES: {self.store.notices + self.ledger.notes}"
        verdict = self.brain.ask("JUDGE", JUDGE, task, "\n\n".join(parts), JUDGE_SCHEMA)
        if verdict is None:
            trace_row(self.trace, "judge_missing", spent=round(self.brain.spent, 4))
        return verdict or {"order": True, "lines": [], "pointers": []}

    def duties(self, pointers: list[str]) -> None:
        for line in self.ledger.lines:
            if line.get("pick") and line["pick"] not in self.store.pages and self.spare(2):
                self.store.read_card(line["pick"], True)
        for name in pointers[:4]:
            if self.spare(3):
                self.store.search(name)

    def order(self) -> bool:
        spec = self.store.roles.get("order") or {}
        if not spec.get("tool"):
            return False
        result = self.store.act(spec["tool"], dict(spec.get("args") or {}))
        trace_row(self.trace, "order", result=result)
        return bool(result.get("finished"))

    def decline(self) -> bool:
        spec = self.store.roles.get("decline") or {}
        if not spec.get("tool"):
            return False
        result = self.store.act(spec["tool"], dict(spec.get("args") or {}))
        trace_row(self.trace, "decline", result=result)
        return bool(result.get("finished"))

    def audit(self) -> list[str]:
        keep = lambda c: any(self.picker.gather.keep(l, c) for l in self.ledger.lines)
        row = lambda i, c: f"{i} {c.get('title', '')[:60]} | shelf: {c.get('category')} | price: {c.get('price')} | in stock: {c.get('stock')}"
        opened = [row(i, self.store.cards[i]) for i in self.store.records]
        rows = [row(i, c) for i, c in self.store.cards.items() if i not in self.store.records and keep(c)][:30]
        wants = "\n".join(f"line {n}: {l['item']} {[(w['key'], w['op'], w['values']) for w in self.ledger.wants(n)]}" for n, l in enumerate(self.ledger.lines, 1))
        answer = self.brain.ask("AUDIT", AUDIT, "REQUIREMENTS:\n" + wants, "OPENED:\n" + "\n".join(opened) + "\nROWS ONLY:\n" + "\n".join(rows), AUDIT_SCHEMA) or {"decline": True, "open": []}
        return [] if answer.get("decline") else [i for i in answer.get("open") or [] if i in self.store.cards and keep(self.store.cards[i])][:4]

class Agent:

    def __init__(self, problem_data: dict) -> None:
        self.trace: list = []
        TRACE_T0[0] = time.monotonic()
        self.pace = Pace()
        self.env = Env(problem_data, self.pace, self.trace)
        self.brain = Brain(self.env, self.pace, self.trace)
        self.store = Store(self.env, self.brain, self.trace)
        request = self.env.view.get("query") or ""
        if self.env.view.get("opening") is not None:
            request += "\n\n" + json.dumps(self.env.view["opening"], ensure_ascii=False)
        self.ledger = Ledger(request)
        self.gather = Gather(self.store, self.ledger, self.trace)
        self.judge = Judge(self.brain, self.store, self.ledger, self.trace)
        self.picker = Picker(self.brain, self.store, self.ledger, self.judge, self.gather, self.trace)
        self.asker = Asker(self.brain, self.store, self.ledger, self.trace)
        self.closer = Closer(self.brain, self.store, self.ledger, self.picker, self.trace)

    def run(self) -> list:
        with ThreadPoolExecutor(max_workers=2) as pool:
            tools = pool.submit(self.store.learn_tools)
            brief = pool.submit(read_request, self.brain, self.ledger.request, self.trace)
            tools.result()
            got = brief.result()
        roles = self.store.roles
        missing = lambda: [r for r, ok in (("search", roles.get("search")), ("order", roles.get("order")), ("open", roles.get("facts") or roles.get("page") or roles.get("compare"))) if not ok]
        if missing():
            trace_row(self.trace, "handoff", reason="tool map without " + " or ".join(missing()), roles=roles)
            self.store.learn_tools("MAP_ALT")
        if not got:
            trace_row(self.trace, "handoff", reason="no brief")
            request = self.ledger.request
            got = self.brain.ask("INTAKE_ALT", INTAKE, "", "REQUEST:\n" + request, {"type": "object"}, lambda a: brief_problem(a, request))
        if not got or {"search", "order"} & set(missing()):
            trace_row(self.trace, "handoff", reason="decline: " + ("no brief" if not got else "tool map without " + " or ".join(missing())), roles=roles)
            self.settle()
            return self.end()
        load_brief(self.ledger, got)
        for n in range(1, len(self.ledger.lines) + 1):
            self.look(n, pages=2, opens=6)
        for n in range(1, len(self.ledger.lines) + 1):
            if any(w.get("op") == "ref" and not w.get("followed") for w in self.ledger.wants(n)):
                self.look(n, pages=2, opens=4)
        if self.ledger.messages_left and self.env.elapsed() < WALK_S:
            self.asker.ask_all(lambda m: self.look(m, pages=2, opens=4))
        self.closer.apply_notices()
        for n in range(1, len(self.ledger.lines) + 1):
            line = self.ledger.line(n)
            pages = 3 if (n in self.gather.capped or line["objective"] != "any") else 2
            self.look(n, pages=pages, opens=6 if line["objective"] != "any" else 3)
        return self.finish()

    def look(self, n: int, pages: int, opens: int) -> None:
        if self.env.left() <= self.gather.reserve_steps() or self.env.elapsed() > WALK_S:
            self.picker.choose(n)
            return
        line = self.ledger.line(n)
        for want in [w for w in self.ledger.wants(n) if w.get("op") == "ref" and not w.get("followed")]:
            want["followed"] = True
            self.picker.follow_chain(n, want)
        order = self.gather.run(n, pages, resume=bool(line["verdicts"]) and not self.passing(n))
        opened = lambda i: i in self.store.records or i in self.store.pages
        fresh = [i for i in order if not opened(i)][:opens]
        judged = [i for i in line["verdicts"] if opened(i)]
        self.picker.open_and_judge(n, judged + [i for i in fresh if i not in judged])
        self.picker.choose(n)
        if line["objective"] == "cheapest" and line.get("pick"):
            cheaper = [i for i in order if not opened(i) and isinstance(self.store.cards[i].get("price"), (int, float)) and self.store.cards[i]["price"] <= (self.store.cards[line["pick"]].get("price") or 0)][:4]
            if cheaper:
                self.picker.open_and_judge(n, cheaper)
                self.picker.choose(n)

    def finish(self) -> list:
        audited, judged = False, 0
        for attempt in range(4):
            if self.env.finished:
                break
            for n in range(1, len(self.ledger.lines) + 1):
                self.picker.choose(n)
            if attempt == 0:
                self.dig()
            self.picker.fit_total()
            if not all(l.get("pick") for l in self.ledger.lines):
                extra = self.closer.audit() if self.closer.spare(4) and not audited else []
                audited = True
                if extra:
                    for n in range(1, len(self.ledger.lines) + 1):
                        self.picker.open_and_judge(n, extra)
                    continue
                self.closer.decline()
                break
            self.closer.write_cart()
            self.closer.apply_notices()
            if judged < 2 and self.env.elapsed() < TASK_S - 30:
                judged += 1
                verdict = self.closer.judge_basket()
                vetoed = False
                for row in verdict.get("lines") or []:
                    n = row.get("line")
                    if isinstance(n, int) and 1 <= n <= len(self.ledger.lines) and not row.get("ok"):
                        line = self.ledger.line(n)
                        use = row.get("use")
                        if use in self.store.cards and self.judge.status(n, use) in ("PASS", "WEAK") and use != line["pick"] and self.closer.spare(2 * len(self.ledger.lines) + 2):
                            line.setdefault("vetoed", set()).add(line["pick"])
                            line["pinned"] = use
                            vetoed = True
                        trace_row(self.trace, "veto", line=n, pick=line["pick"], problem=row.get("problem"), use=use, taken=vetoed)
                if vetoed:
                    continue
                self.closer.duties(verdict.get("pointers") or [])
            if self.closer.order():
                break
            self.closer.apply_notices()
            self.ledger.refusals += 1
            for line in self.ledger.lines:
                if line.get("pick") and self.closer.spare(2 * len(self.ledger.lines) + 2):
                    self.store.read_card(line["pick"])
        self.settle()
        return self.end()

    def passing(self, n: int) -> bool:
        pick = self.ledger.line(n).get("pick")
        return bool(pick) and self.judge.status(n, pick) == "PASS"

    def dig(self) -> None:
        digging = [n for n in range(1, len(self.ledger.lines) + 1) if not self.passing(n)]
        while digging and not self.gather.walk_over():
            for n in list(digging):
                fresh = None if self.gather.walk_over() else self.gather.resume(n)
                if fresh is None:
                    digging.remove(n)
                    continue
                self.picker.open_and_judge(n, fresh)
                self.picker.choose(n)
                trace_row(self.trace, "dig", line=n, opened=fresh, pick=self.ledger.line(n).get("pick"), passing=self.passing(n))
                if self.passing(n):
                    digging.remove(n)

    def settle(self) -> None:
        if not (self.env.finished or self.env.over) and self.env.left() > 0:
            if self.ledger.lines and all(l.get("pick") for l in self.ledger.lines) and self.closer.in_cart:
                self.closer.order()
            if not self.env.finished and not self.closer.decline():
                trace_row(self.trace, "handoff", reason="no finish call ended the episode", roles=self.store.roles)

    def end(self) -> list:
        trace_row(self.trace, "end", finished=self.env.finished, steps=self.env.step, spent=round(self.brain.spent, 4), state=self.ledger.state(self.env))
        return self.trace

    def rescue(self, exc: Exception) -> list:
        self.trace.append({"kind": "crash", "error": f"{type(exc).__name__}: {exc}"[:400]})
        for close in ([self.finish] if self.ledger.lines else []) + [self.settle]:
            if self.env.finished or self.env.over or self.env.left() <= 0:
                break
            try:
                close()
            except Exception as again:
                self.trace.append({"kind": "crash", "error": f"{type(again).__name__}: {again}"[:400]})
        return self.trace


def agent_main(problem_data: dict) -> list:
    agent = None
    try:
        agent = Agent(problem_data)
        return agent.run()
    except Exception as exc:
        if agent is None:
            return [{"kind": "crash", "error": f"{type(exc).__name__}: {exc}"[:400]}]
        try:
            return agent.rescue(exc)
        except Exception:
            return agent.trace
