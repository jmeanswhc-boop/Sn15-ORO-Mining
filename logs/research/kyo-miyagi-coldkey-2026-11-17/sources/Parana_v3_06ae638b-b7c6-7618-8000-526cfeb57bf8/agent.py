"""Shopping agent for ORO world sessions (session3-2 build of a team agent).

A setup pass turns the request and the tool schemas into items, requirements, budgets, procedure steps and tool roles.
The session then searches and screens rows, opens listings, judges them against every requirement (values taken from
other listings are traced through the pages that point to them), asks the shopper about likely hidden properties,
keeps the procedure the request demands, and finishes by ordering the cheapest fitting basket or declining.
"""
from __future__ import annotations
import json
import re
import time
from difflib import SequenceMatcher
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from src.agent.proxy_client import ProxyClient
PARSE_LLM = 'anthropic/claude-sonnet-5.5'
SCHEME_LLM = 'anthropic/claude-sonnet-5.5'
CONFIRM_LLM = 'anthropic/claude-sonnet-5.5'
FAST_LLM = 'anthropic/claude-haiku-5.5'
BACKUP_LLM = 'openai/gpt-6-luna'
TASK_SECS = 255.0
CONCLUDE_SECS = 45.0
CHARS_FACT = 2200
RECORD_ROW_CHARS = 260
LOOSE_COST = 0.03
COST_CAP = 0.05
_SERIES = re.compile('\\b([A-Z]{1,5}\\d{1,3})-\\d{2,}\\b')
OPS_REF = {'same as reference': 'is', 'different from reference': 'is not', 'counterpart of reference': 'is', 'at least reference': 'at least', 'at most reference': 'at most'}
_llm = ProxyClient(timeout=75, max_retries=1, retry_delay=1)
_env = ProxyClient(timeout=90, max_retries=2, retry_delay=1)
_MEMORY_WORLD: dict[str, Any] = {}

def _compact(value: Any, limit: int) -> str:
    text: Any = json.dumps(value, ensure_ascii=False, separators=(',', ':'))
    _rv1: Any = text if len(text) <= limit else text[:limit] + '…'
    return _rv1

def _find_seq_of_mappings(obs: Any, depth: int=0) -> list[dict[str, Any]] | None:
    if isinstance(obs, list) and obs and all((isinstance(r, dict) for r in obs)):
        return obs
    if isinstance(obs, dict) and depth < 2:
        best: Any = None
        for v in iter(obs.values()):
            found: Any = _find_seq_of_mappings(v, depth + 1)
            if found and (best is None or len(found) > len(best)):
                best: Any = found
        return best
    return None

def _js_in(text: str) -> Any:
    if not bool(text):
        return None
    text: Any = text.strip()
    fences: Any = re.findall('```(?:json)?\\s*(.*?)```', text, re.S)
    if bool(fences):
        text: Any = fences[-1].strip()
    for opener, closer in iter((('{', '}'), ('[', ']'))):
        start: Any = text.find(opener)
        if start < 0:
            continue
        last: Any = None
        while start >= 0:
            got: Any = _js_one(text, start, opener, closer)
            if got is None:
                break
            last, end = got
            start = text.find(opener, end)
        if last is not None:
            return last
    try:
        _rv3: Any = json.loads(text)
        return _rv3
    except Exception:
        return None

def _js_one(text: str, start: int, opener: str, closer: str) -> Any:
    if True:
        depth: int = 0
        in_str: bool = False
        esc: bool = False
        for i in range(start, len(text)):
            ch: Any = text[i]
            if bool(in_str):
                if bool(esc):
                    esc: bool = False
                elif ch == '\\':
                    esc: bool = True
                elif ch == '"':
                    in_str: bool = False
                continue
            if ch == '"':
                in_str: bool = True
            elif ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return (json.loads(text[start:i + 1]), i + 1)
                    except Exception:
                        return None
    return None

def _near(a: Any, b: Any) -> bool:
    if _norm(a) == _norm(b):
        return True
    ta, tb = (_pieces_of_text(a), _pieces_of_text(b))
    _rv4: Any = bool(ta) and ta == tb
    return _rv4

def _norm(text: Any) -> str:
    text: Any = str(text or '').lower().replace('&', ' and ').replace('_', ' ').replace('-', ' ')
    _rv5: Any = re.sub('\\bcolour', 'color', re.sub('[^a-z0-9 ]+', ' ', text).strip())
    return _rv5

def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        _rv6: float = float(value)
        return _rv6
    if isinstance(value, str):
        m: Any = re.search('-?\\d+(?:\\.\\d+)?', value.replace(',', ''))
        if bool(m):
            try:
                _rv7: float = float(m.group(0))
                return _rv7
            except Exception:
                return None
    return None

def _parallel(jobs: list, workers: int=3) -> list:
    if len(jobs) <= 1:
        _rv8: list = [_safe(j) for j in jobs]
        return _rv8
    with ThreadPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
        _rv9: list = list(pool.map(_safe, jobs))
        return _rv9

def _pieces_of_text(text: Any) -> set[str]:
    out: set = set()
    for t in iter(_norm(text).split()):
        if len(t) > 1:
            if len(t) > 3 and t.endswith('s'):
                t: Any = t[:-1]
            out.add(t)
    return out

def _roots(text: Any) -> set[str]:
    _rv10: set = {t[:4] for t in _pieces_of_text(text)}
    return _rv10

def _safe(job: Any) -> Any:
    try:
        _rv11: Any = job()
        return _rv11
    except Exception:
        return None

def _sentence_is(value: Any) -> bool:
    _rv12: Any = isinstance(value, str) and len(value) > 24 and (' ' in value.strip())
    return _rv12

def _similar(a: Any, b: Any) -> float:
    ta, tb = (_pieces_of_text(a), _pieces_of_text(b))
    _rv13: Any = 0.0 if not ta or not tb else len(ta & tb) / len(ta | tb)
    return _rv13

def _tokens2(text: Any) -> set[str]:
    _rv14: set = {t[:-1] if len(t) > 3 and t.endswith('s') else t for t in _norm(text).split()}
    return _rv14

def _unwrap(obs: Any) -> tuple[Any, list[str]]:
    lines: list[str] = []
    if isinstance(obs, str):
        _rv15: tuple = (obs, lines)
        return _rv15
    if not isinstance(obs, dict):
        _rv16: tuple = (obs, lines)
        return _rv16
    inner: dict[str, Any] = {}
    dict_siblings: float = sum((1 for v in obs.values() if isinstance(v, dict)))
    record: bool = any((isinstance(v, str) and v.strip() and (' ' not in v.strip()) for v in obs.values()))
    for k, v in iter(obs.items()):
        if isinstance(v, list) and v and all((_sentence_is(x) for x in v)):
            lines.extend((str(x) for x in v))
        elif _sentence_is(v) and dict_siblings == 1 and (len(obs) <= 3) and (not record):
            lines.append(str(v))
        else:
            inner[k] = v
    if lines and len(inner) == 1 and isinstance(list(inner.values())[0], dict):
        inner: Any = list(inner.values())[0]
    _rv17: tuple = (inner, lines)
    return _rv17

class LLM:

    def __init__(self, episode: 'Episode'):
        self.ep = episode
        self.price_tag = 0.0
        self.invokes = 0
        self.misfires = 0
        self.blank = False
        self.rescued = 0

    def chat(self, model: str, system: str, user: str, max_tokens: int) -> str | None:
        if self.price_tag >= LOOSE_COST and model in (SCHEME_LLM, CONFIRM_LLM, PARSE_LLM) and (model != FAST_LLM):
            model: Any = FAST_LLM
        left: Any = self.ep.time_remaining()
        urgent: bool = bool(getattr(self.ep, 'urgent', False))
        if left < (6 if urgent else CONCLUDE_SECS - 12):
            return None
        _llm.timeout = int(max(4, min(20, left - 3))) if urgent else int(max(10, min(45, left - CONCLUDE_SECS + 20)))
        if bool(model.startswith('anthropic/')):
            sys_content: Any = [dict(type='text', text=system, cache_control=dict(type='ephemeral'))]
        else:
            sys_content: Any = system
        body: dict = dict(model=model, messages=[dict(role='system', content=sys_content), dict(role='user', content=user)], temperature=0, max_tokens=max_tokens)
        self.invokes += 1
        self.blank = False
        result: Any = _llm.post_verbose('/inference/chat/completions', body)
        for attempt in range(2):
            error: Any = result.error if isinstance(result.error, dict) else {}
            if isinstance(result.data, dict) and result.data.get('choices'):
                break
            if error.get('status') not in (429, 500, 502, 503, 504) or self.ep.time_remaining() < CONCLUDE_SECS + 10:
                break
            time.sleep(2.0 * (attempt + 1))
            result: Any = _llm.post_verbose('/inference/chat/completions', body)
        data: Any = result.data
        if isinstance(data, dict) and data.get('choices'):
            usage: Any = data.get('usage') or {}
            if isinstance(usage.get('cost'), (int, float)):
                self.price_tag += float(usage['cost'])
            message: Any = data['choices'][0].get('message') or {}
            content: Any = message.get('content')
            if isinstance(content, list):
                content: str = ''.join((str(p.get('text') or '') for p in content if isinstance(p, dict)))
            if bool(content):
                _rv18: str = str(content)
                return _rv18
            self.blank = not message.get('tool_calls')
        self.misfires += 1
        self.ep.remark2('llm_failure', model=model, error=str(getattr(result, 'error', ''))[:200])
        return None

    def json(self, models: list[str], system: str, user: str, max_tokens: int) -> Any:
        max_tokens = max(2 * max_tokens, 1200)
        for model in iter(models):
            text: Any = self.chat(model, system, user, max_tokens)
            parsed: Any = _js_in(text or '')
            if parsed is not None:
                return parsed
            if self.blank and model.startswith('anthropic/'):
                args: Any = self.forced_once(model, system, user, ANSWER_TOOL, max_tokens)
                got: Any = args.get('answer') if isinstance(args, dict) else None
                got: Any = _js_in(got) if isinstance(got, str) else got
                if got is not None:
                    self.rescued += 1
                    self.ep.remark2('llm_rescued', model=model)
                    return got
        return None

    def forced_once(self, model: str, system: str, user: str, tool: dict[str, Any], max_tokens: int) -> dict[str, Any] | None:
        left: Any = self.ep.time_remaining()
        urgent: bool = bool(getattr(self.ep, 'urgent', False))
        if left < (6 if urgent else CONCLUDE_SECS - 12):
            return None
        _llm.timeout = int(max(4, min(20, left - 3))) if urgent else int(max(10, min(45, left - CONCLUDE_SECS + 20)))
        head: Any = [dict(type='text', text=system, cache_control=dict(type='ephemeral'))] if model.startswith('anthropic/') else system
        body: dict = dict(model=model, messages=[dict(role='system', content=head), dict(role='user', content=user)], temperature=0, max_tokens=max_tokens, tools=[dict(type='function', function=tool)], tool_choice=dict(type='function', function=dict(name=tool['name'])))
        self.invokes += 1
        result: Any = _llm.post_verbose('/inference/chat/completions', body)
        data: Any = result.data
        if isinstance(data, dict) and data.get('choices'):
            usage: Any = data.get('usage') or {}
            if isinstance(usage.get('cost'), (int, float)):
                self.price_tag += float(usage['cost'])
            message: Any = data['choices'][0].get('message') or {}
            for placed in iter(message.get('tool_calls') or []):
                args: Any = (placed.get('function') or {}).get('arguments') if isinstance(placed, dict) else None
                args: Any = _js_in(args) if isinstance(args, str) else args
                if isinstance(args, dict) and args:
                    return args
        self.ep.remark2('llm_failure', model=model, error='no function call ' + str(getattr(result, 'error', ''))[:160])
        return None
READ_BRIEF = 'Before an assistant starts shopping, you lay out its task. You receive the shopper\'s message and the shop\'s tools (written as OpenAI function schemas). Answer with exactly one JSON object and nothing else.\n\nTop-level keys of that object:\n* "items" - a list with one object per distinct product to buy (usually a single one). Each item object has:\n  - "name": what the product is called, a short noun phrase taken from the request (for a list entry, that entry\'s wording);\n  - "category": the product kind or shop category, as the request words it;\n  - "quantity": how many listings to buy (an integer);\n  - "unit_price_cap": the most the request lets ONE unit of this item cost (a price limit for this item alone, separate from the total), as a plain number in the request\'s unit, else null;\n  - "requirements": the firm demands on this item\'s own properties (never price, stock, quantity or budget), each written as {"property": ..., "op": ..., "value": ..., "reference": ...}. The "property" is named the way a listing page would name it; the "value" copies the request\'s value, bound or list (for "one of", separate the values with " | "); "reference" is null except for the five reference ops below;\n  - "best": the measure to optimise for this item alone when the request names one for it, else null;\n  - "hints": names of properties the story suggests the shopper quietly cares about (who will wear it -> a gender or an age group; what goes inside a computer -> the processor); an empty list when there are none;\n  - "search_terms": 2 to 4 search-box queries, the sharpest first. Each one is short: the product noun plus one or two value words (a colour, a material, a figure with its unit such as 16gb or 5v). Never a property name by itself, never a word that only identifies a referenced listing; the final query is the bare product noun.\n* "total_cap": the highest total the shopper accepts, as a plain number in the request\'s unit (in cents when the request counts in cents), else null.\n* "cheapest": true only if the shopper asks for the lowest price or the cheapest total; otherwise false. A "cheapest" that appears only inside the instruction for when nothing fits belongs to "no_fit_rule", not here.\n* "best": null, or the measure by which the shopper wants the best one when it applies to every item (a measure for one item goes in that item\'s "best").\n* "message_budget" and "call_budget": the number of shopper messages and of tool calls/steps the request allows, as integers, or null. When the request limits them in more than one place, take the strictest ("no questions at all" makes message_budget 0).\n* "in_stock_only": true when only listings that can be bought right now are wanted.\n* "ask_about": subjects the request tells the assistant to raise with the shopper, in its words (empty list if none).\n* "fit_rules": [{"accessory": item name, "host": item name}] for each "make sure X fits Y" instruction (empty list if none).\n* "conditions": demands that hold for one item only depending on how another item turns out, each {"if": {"item", "property", "values": [...]}, "then": {"item", "property", "op", "value"} or null, "otherwise": the same shape or null} (empty list if none). Ops allowed here: is, is not, one of, at least, at most.\n* "decline_rule": what to do when nothing qualifies, as the request says it, or null.\n* "no_fit_rule": what the request says to do when no listing meets every requirement: {"action": "decline" (buy nothing), "order_closest" (buy the listing that comes closest), "order_fitting_lines_only" (buy only the items that fully fit) or "not said", "count_as_miss": what the request says to count as shortfalls besides the product\'s own requirements, from "stock" and "budget" (empty list if none), "price_slack": how much dearer than the cheapest of the closest ones is still fine, as a fraction (0.1 for 10%), or null, "tie_break": "cheapest", "dearest" or null, "words": the request\'s words for this, copied, or null}. Read the meaning, not the look: "put forward" or "go for" a listing means buying it; "just say so" or "walk away" means buying nothing.\n* "basket_checks": demands on spending or on the items taken together, each {"what": "total" (the sum spent), "same" (every item shares the value of a field), "distinct" (no two items share it), "share" (spend as a part of the total or of the budget) or "compare" (spend on one item against spend on another), "item": the item it is limited to, or null for every item, "other": for "compare", the item compared with, else null, "of": for "share", "total" or "budget", else null, "field": the listing field it is about (seller, brand, colour, ... ; null for total and share), "op": "at least" or "at most" (for total, share and compare, else null), "value": a number in the request\'s unit, a share as a fraction such as 0.25, for "compare" how many times the other item\'s spend (1 for equal), or null}. A plain spending ceiling stays in "total_cap" only. Empty list if none.\n* "adjustments": changes to how many of an item to buy, or whether to buy it at all, that depend on what is found, each {"when": {"subject": "total" (the basket total), "price" (one unit of the item) or a property name, "item": item name or null, "op": "is", "is not", "one of", "at least", "at most", "under" or "over", "value": ...}, "then": {"item": item name, "quantity": a number} or {"item": item name, "drop": true}, "otherwise": the same shape as "then", or null}. Empty list if none.\n* "closing_note": what the request wants said in the final note or explanation when the task ends, in its words, or null. This is never a message to the shopper.\n* "other_rules": every other instruction in the request that no key above holds, copied word for word (empty list if none).\n* "procedure": the steps the request imposes on the way the job is done, each {"moment": "start|before_order|before_decline|before_finish", "kind": "message|tool", "tool": tool name or null, "goal": what must happen in the request\'s words, "messages": the number of shopper messages it uses (0 for a tool)}. This holds approvals or confirmations to get, things to show or tell the shopper, tools to use, an order of steps. It never holds product demands, budgets, caps on messages or steps, the nothing-fits case (that is "no_fit_rule"; its "say so" goes in the closing note, never in a message) or what the final note must say (that is "closing_note"). A check made inside the shop (stock, availability, a value) is kind "tool" with 0 messages unless the request wants the outcome reported. Empty list if none.\n* "tools": which schema tool does what, with exact names:\n  - "search_text": {"name", "query" (the keyword argument), "page" (the argument that moves on to more results - page number, row offset or a token from the last result - or null), "page_kind" ("number", "offset", "token" or null), "mode" ({"arg", "value"} that switches the tool to keyword search, or null)};\n  - "search_fields": the same keys for searching by fields, plus "category", "brand" and "price_max" arguments (each an argument name or null), or null when there is no such search;\n  - "open": {"name", "id"} for the tool that returns a whole listing;\n  - "facts": {"name", "id"} for a separate property-table tool, or null;\n  - "compare": {"name", "ids"} for a tool returning several listings in full, or null;\n  - "cart_add": {"name", "id", "qty" (or null), "mode" ({"arg", "add", "remove"} when one argument chooses adding or removing, else null)};\n  - "cart_set": {"name", "lines", "id", "qty"} for a tool that replaces the whole cart with a list of lines ("id" and "qty" are the keys inside one line), or null;\n  - "cart_view": {"name"} or null;\n  - "checkout": {"name", "args" ({argument: value} or null)} - the final ordering tool;\n  - "decline": {"name", "remark" (its free-text argument or null), "args" ({argument: value} or null)} - the final tool that ends without buying;\n  - "message": {"name", "text"} - the tool that writes to the shopper;\n  - "status": {"name"} for a tool that reports what is left, or null.\n\nThe "op" of a requirement is one of: "is", "is not", "one of", "contains", "at least", "at most", or, for a value taken from another listing, "same as reference", "different from reference", "counterpart of reference", "at least reference", "at most reference".\n\nReading the request:\n- Only firm demands become requirements. Wishes ("ideally", "it would be nice") and story details (who uses it, where it goes) do not; note what they suggest in "hints". Words that merely name the product or its category ("kids\'" in "kids\' rain boots", "gaming" inside a category name) belong in "name"/"category". But when the request sets a value for a named property ("in the <value> style", "with <value> as its <property>", "a <property> of <value>"), that is a requirement on the property even if the value sounds like a product type, and the category remains the product named.\n- "3 x lamps where the shade is linen and the height is not tall" is a single item, quantity 3, carrying two requirements.\n- A count for what one listing holds (a box, set or pack of N pieces) is a requirement on that pack size; "quantity" is how many listings to purchase.\n- A remark that only spells out a requirement\'s consequence ("zip front only, so nothing with buttons or toggles") stays ONE requirement (closure: zip front); do not invent properties for what it excludes.\n- Either-or values are "one of" with "A | B"; "not X" and "anything except X" are "is not"; "at least N units" is "at least" with value "N units". Keep units and spelling as written.\n- Values tied to ANOTHER listing\'s page (one identified by its category, seller and a few attribute values): wanting the value that page states -> "same as reference"; anything except it -> "different from reference"; the value the page or the shop pairs with it, the one that belongs with it -> "counterpart of reference"; no lower / no higher than it -> "at least reference" / "at most reference". Describe the page the request names even if that page merely forwards the property to yet another listing (the assistant follows such forwarding itself). In "reference" put {"category", "seller", "attributes" ({property: value} that identify the page), "property" (what to read there), "count" (how many of that seller\'s listings the request says qualify, or null), "pick" (the rule for choosing among them, such as "whichever is wider", or null)}. Attributes hold plain identifying values only, never the comparison. Those identifying values describe THAT page alone: never turn them into requirements on the item being bought or into its search terms. Invented case: "for the colour, steer clear of whatever Fenwick\'s garden hose page (length 20 m, material rubber) states" gives exactly one requirement {"property": "colour", "op": "different from reference", "value": "", "reference": {"category": "garden hoses", "seller": "Fenwick", "attributes": {"length": "20 m", "material": "rubber"}, "property": "colour"}} and nothing on length or material.\n- A demand on one item that applies only depending on another item ("if X\'s P ends up V, then Y\'s Q must be W"; "Y\'s Q must be W unless X\'s P is V") goes into "conditions": in the first form "then" applies when X\'s P is one of the values, in the second "otherwise" applies when it is not. Never into "requirements" or "procedure".\n- When the request names a shop category ("the category is garden tools"), copy those words into "category".\n- A total stated in cents makes "total_cap" that number of cents.\n- Tool and argument names come verbatim from the schemas. If ONE tool ends the task both ways, an argument choosing between buying the cart and leaving empty-handed, then "checkout" and "decline" both name that tool, each with "args" set to the argument value that selects it (exactly as the schema spells it). A final tool that closes without buying is "decline"; the one that checks out the cart is "checkout". If a single search tool has an enum argument switching between keyword and field search, both "search_text" and "search_fields" name it, each with its mode value.'
SIFT_BRIEF = 'Search-result triage for a shopping assistant. Every numbered row gives an id, a title, a short text, a category and a price. For each row decide whether the listing might still satisfy the requirements once its full property page is opened, using only what the row itself shows. If the row is silent about a requirement, it might still satisfy it (that is a "maybe"). If the row\'s own words plainly contradict a required value, it is a "no". If the row itself shows the required values, it is a "yes". The row\'s category label settles what kind of product it is: a row filed under the requested category is the right kind no matter what its title suggests, so never reject a row for its kind. Reply with one JSON object only: {"yes": [row numbers], "no": [row numbers]}; any row in neither list is a "maybe".'
JUDGE_BRIEF = 'Decide, for every listing shown, whether its own records satisfy each numbered requirement (R1, R2, ...).\n\nReply format - one JSON object, no comments: {"verdicts": [{"id": ..., "failed": [...], "unsure": [...], "weak": [...], "loose": [...]}, ...]}, one entry per listing, each list holding R numbers, and any empty list left out.\n- failed: the records plainly contradict the requirement.\n- unsure: nothing in the records speaks to it.\n- weak: it is satisfied only through the title, the description or an options field, not through a property or a seller note.\n- loose: it is satisfied by a synonym, a wider term or a unit conversion rather than by the value\'s own words (for instance \'female\' standing in for a required \'girls\').\nAn entry with neither failed nor unsure means the listing satisfies everything asked.\n\nHow to judge\n1. Look first at the property table. Any key whose name means the property counts, however it is spelled (spaces, underscores, prefixes, short forms), and one property can be spread over several keys. If such a key exists, its value rules: another value is a failure even when the title claims the right one.\n2. Values are compared by sense. One item of a comma list suffices; singular and plural are equal; near synonyms and evident short forms count; case is irrelevant. Numbers are compared as numbers after converting units; a bound includes its own edge; a range passes a bound only when the whole range passes.\n3. Only when no key covers the property may the seller\'s notes, the options/variants field, the title or the description provide the value. If none of them states it, mark it unsure. A value filed under some other, related property is no evidence (a lining material is not the material; a handle is not a lock type).\n4. "is not" and "different from" fail if the listing\'s property shows the excluded value. "one of" needs any listed value, "contains" needs the value among those given. A requirement that some feature be missing (value none / no / without) holds when no record mentions the feature.\n5. The category label decides what kind of product it is: carrying the requested category makes it the right kind whatever the title. Only a listing without a label can fail as an accessory, spare part or case.\n6. Compatibility with another product (named by model names or codes) holds only when this listing\'s own text lists one of those names among what it fits. Where the other product\'s page names a family of compatible part codes, a listing whose own part code (often bracketed in its title) begins with that family holds. A sentence that some OTHER part fits a product speaks for the listing whose title bears that part\'s code, not for the page that contains it; "related_pages" attached to a listing are such pages, and their statements about this listing\'s part code count for it. A mere pointer to another page proves nothing.\n7. If the seller notes send one of this listing\'s properties to ANOTHER listing (go by / take the value of some other listing), the listing\'s own value for it no longer counts and a requirement on it is unsure, unless a related page attached to the listing states the value. Forwarding of properties that no requirement concerns is irrelevant.\n8. STORE FIELDS, when given, show per required property the keys and values this shop\'s already-read listings of the category use. If some of them hold the exact required value in that field, a listing whose field shows a different value fails, even a broader or related one; a value carried by none of the read listings may still be matched through a clear synonym. When STORE FIELDS show listings of this kind carrying a key named plainly as the property, a listing without such a key has no value for it: a key naming one part of the product (a handle material, a lining material, a strap colour) does not stand in for it, so the requirement is unsure.\n9. Judge exactly the listed requirements and nothing else.'
SCHEME_BRIEF = 'Your task: choose the questions an assistant will put to a shopper. Each held-back requirement of the shopper is the value of a single property of one requested product, and the shopper discloses it only to a message that names that very property (or a plain synonym). Only the FIRST property named in a message gets answered; asking about something the shopper doesn\'t care about returns "no preference" and burns the message.\nPer requested item you see its name, requirements already known, story hints, and a property menu listing what its candidate listings actually carry (how many listings carry each property, plus sample values).\nReturn exactly QUESTION_COUNT questions, strongest first: {"questions": [{"item": item name as given, "property": property name spelled as on the menu (words with spaces)}, ...]}.\nOrder of preference:\n(a) Properties the story or the hints aim at - whatever the request says the shopper will tell when asked. Who will wear or use it suggests an age group or a gender; how it fastens suggests a closure; what is inside suggests a processor; how it looks suggests colour, pattern or style. Pick the menu name closest to each hint.\n(b) After that, first the property that divides this product into kinds that cannot replace each other in use (what it runs on, where on the vehicle or body it goes, the form or packaging it comes in, its shape); then a property measured in figures with a unit (a dimension such as depth, width or height, a size, a capacity, a weight) - such a property earns a question of its own whenever the menu has one; then material, look, type or style, the intended user, the headline technical figure. Prefer properties that many candidates carry (see the listings count). A property that shows a single value on the listings read so far is still a good pick: the shopper\'s value is often one that few listings carry. Weak picks: an option or variant selector, identifiers, model numbers, warranty or shipping remarks.\n(c) Never a property a known requirement already settles, never price, stock or quantity, never the same property twice (two spellings count as one). Distribute questions across items by how uncertain each is; an item with strong hints receives the first one.'
CLUE_BRIEF = 'Help a shopping assistant decide what to ask its shopper. The shopper keeps some requirements back; each is the value of ONE property of one requested product and is told only to a question about exactly that property. You receive the request and, per requested item, its known requirements and a property menu (properties its candidate listings carry, with counts and sample values).\nAnswer with one JSON object only:\n{"hinted": [{"item": "the item name as given", "property": "the menu property name, spelled exactly", "why": "the request\'s words that point at it"}],\n "junk": ["menu properties that are no real product trait a shopper would state: option or variant selectors, identifiers, model numbers, warranty, shipping or packaging remarks, repeats of a known requirement"]}\nPut into "hinted" only properties that the request\'s story, or the subjects it says to raise, point at (a remark about who wears it -> a gender or age-group property; "how it fastens" -> a closure; what sits inside a computer -> a processor; "how it looks" -> colour, pattern or style; a subject the request says to raise -> its property). At most 3, best first; an empty list when the request hints at nothing. Never include a property a known requirement already settles.'
RESPONSE_BRIEF = 'Classify the shopper\'s answer to the assistant\'s question (which concerned one property of one product) and extract what it says.\nReturn one JSON object with these keys:\n"kind" - one of need, repeat, free, split, unclear, other (see below);\n"property" - the property the answer concerns: the name the shopper uses if they name one (which may differ from what was asked), otherwise the asked property;\n"op" - for a direct value one of "is", "one of", "at least", "at most", "is not"; for a value tied to another page one of "same as reference", "at least reference", "at most reference", "different from reference", "counterpart of reference";\n"value" - the value, bound or list stated (list values joined with \' | \');\n"reference" - null, or {"category", "seller", "attributes": {property: value}, "property" (what to read on that page), "count" (how many listings it says qualify, or null), "pick" (the rule for choosing among several, or null)};\n"budget" - null, or the new total the shopper says they can spend, as a number;\n"answer" - "" or, if the shopper asked the assistant something, a short plain reply;\n"steps" - [] or the procedural steps the shopper now demands (an approval to get, something to show or tell them, a tool to use before ordering), each {"moment": "before_order|before_decline|before_finish", "kind": "message|tool", "tool": null, "goal": what must happen, "messages": 1}.\nThe kinds:\nneed = the shopper insists on a value, bound, exclusion or list, in whatever words (settled, firm, nothing but, only X will do, anything bar X, X or Y and nothing else). "Anything bar X" / "anything other than X" (the shopper accepts the rest) means op "is not", value X; refusing everything other than X ("I won\'t settle for / accept / take anything other than X", "nothing other than X will do", "can\'t do without X") means op "is", value X. When they point to ANOTHER listing\'s page (named by category, seller and attribute values): the value stated there -> "same as reference" with "reference" filled; anything but that value -> "different from reference"; the value the page pairs with its own (its counterpart, what goes with it) -> "counterpart of reference"; no lower or no higher than that value -> "at least reference" or "at most reference".\nrepeat = rather than addressing the asked property, the shopper restates a requirement already in KNOWN NEEDS (same property, same value).\nfree = no preference at all (doesn\'t mind, leaves it open, anything is fine, never thought about it).\nsplit = they want one question at a time, the questions taken in turn, or only the first part answered.\nunclear = they did not follow the question: they ask for it again, more simply, more gently or once more, or say it made no sense to them. A request to repeat or rephrase is unclear, never split.\nother = anything else (a greeting, a counter-question, a change of budget).\nExplicit bound words decide: when the answer says "at most N", "at least N", "no more than N" or "under N", those words set op and value even if the sentence around them is oddly built (e.g. "nothing that falls short of at most 34 inches" -> op "at most", value "34 inches").'
EVENT_BRIEF = 'A shop tool\'s result carried a line from the shopper while the assistant was busy. Describe what has changed as one JSON object: {"changes": [ ... ]}, each change being {"kind", "budget", "property", "op", "value", "item", "note"} (plus "step" for kind step).\nKinds and what to fill:\nbudget - the shopper can now spend a different amount (more or less): "budget" is that number exactly as written, in the line\'s own unit (cents stay cents; do not convert).\ndrop - something is no longer required ("let X go", "never mind X", "drop X"): "property" names it.\nadd - a fresh requirement: "property", "op" (is, is not, at least, at most or one of) and "value".\nprice - the shopper reports that the chosen listing now costs something else.\nstock - the shopper reports that the chosen listing can no longer be bought.\nstep - the shopper wants a step in how the job is done (an approval, something shown or told to them, a tool used before ordering): "step" = {"moment": "before_order|before_decline|before_finish", "kind": "message|tool", "tool": null, "goal": what must happen, "messages": 1}.\nother - anything else.\nUnused fields are null; "item" names the requested product concerned (or null); "note" is one short sentence.'
REFER_BRIEF = 'The assistant is tracing a reference and has opened the page below; it wants this page\'s value for one property.\nReturn one JSON object: {"value": ..., "paired": ..., "redirect": ...}\n- "value": the page\'s own value as it is written, or null.\n- "paired": the value this page couples with that value, or null.\n- "redirect": null, or {"category", "seller", "attributes": {property: value}, "property"} when the page sends the WANTED property to another listing.\nWhere values can be found: the property table, the seller\'s notes, and - only when no table key names the property - the options field, the title or the description. The notes may instead forward the property to ANOTHER listing (identified by category, seller and attribute values): then answer with the redirect and leave "value" null. Forwarding of some other property is not a redirect for this one. A value the notes state outright wins. A sentence that only enumerates the shop\'s pairs does not reveal which value is this page\'s own. "paired" is taken only from a pairing the page itself states - from a list of pairs, the pair that contains this page\'s own value (pairs work both ways) - and is null when the page states none or its own value is unknown.'
PHRASE_BRIEF = 'Write search-box queries for a shopping assistant. Shops differ: some return only listings holding every word of the query, others return listings holding any of its words, ranked by how many they hold; only the first rows of the results can be read. Either way short queries do best, led by the most distinctive required value: a long query with many values floods an any-word shop and empties an every-word one. You are given the product, its requirements, the queries tried so far and some listings already seen (with whether they met the requirements). Reply with one JSON object only: {"queries": ["...", ...]} holding up to 5 new queries of 1-3 words, best first: a required value spelled as the listings seen write it (keep their unit spelling, such as 2.5mm or 5w) plus the product\'s own noun as this shop\'s titles word it; then that value with a second required value; then other words this shop uses for that product or value; never repeat a query already tried.'
OPTION_BRIEF = 'A shop tool that ends a shopping task needs an argument stating how it ends. You get the tool\'s description, the arguments it still needs (with the values each accepts) and the goal. Reply with one JSON object only: {"<argument name>": "<the accepted value that achieves the goal>", ...}.'
DOOR_BRIEF = 'You are auditing the procedure. At a given moment of the task, check the assistant\'s process against what the request demands.\nYou receive: the request; the procedural steps it demands; the moment (what the assistant is about to do); the basket right now; the allowances left; the shop\'s tools (excluding those that end the task or change the cart, which the assistant runs itself); and all calls made so far with their results.\nFind the demanded steps that are due NOW - before the assistant moves on - and have not been done for the present state. Rules: an approval or confirmation is valid only for the basket exactly as it is now (one obtained for an earlier basket does not count); a step already done for this state is not due again; a step that belongs to a different moment is not due.\nReturn one JSON object, nothing else: {"actions": [{"tool": exact tool name, "args": {argument: value, ...}, "why": the demanded step it performs}]} in order and at most two, or {"actions": []} when nothing is due.\nWriting a shopper message: it must be self-contained - for a basket, name each line (quantity, title, price) and the total, then state plainly what you are asking. Product-property questions never belong here, and the last remaining message is only ever spent on a demanded step. Checks inside the shop (stock, availability, a value) are done with the shop\'s reading tools, not by message, unless the request wants to be told; because the assistant reads every listing it buys, such a check is due only if the basket\'s listings are still unread.'
RANK_BRIEF = 'Every listing below already meets the requirements; the shopper wants the top one on a stated measure (a value to make as large or as small as possible). Take each listing\'s figure for that measure from its properties, its title or its description, converting units where needed so that the figures are comparable. Return one JSON object only: {"order": [all listing ids, best first], "values": {id: the figure as read}}. Listings with no readable figure come last; on equal figures, the cheaper listing comes first.'
INSPECTION_BRIEF = 'The assistant is about to abandon one product. Check its search first. Given: the product, its requirements, the budget, the queries already used, and all listings encountered - those it READ (with their properties, notes, and the requirements it scored as failed or unaddressed) and rows it only saw in search results.\nReturn one JSON object only: {"fits_after_all": [...], "unopened": [...], "queries": [...]}.\n- fits_after_all: ids of READ listings that in fact satisfy every requirement, judged from their own records (a property\'s value decides; a different value fails; a property no record states counts as not met).\n- unopened: up to 4 ids of rows not read yet that most probably satisfy the requirements, most promising first.\n- queries: up to 2 fresh keyword queries of 1-3 words likely to surface fitting listings, none of them already used.\nLists may be empty.'
COND_BRIEF = 'A condition set by the shopper has to be tested on one listing: does the listing\'s value for the named property belong to the given values (same meaning, units converted), judging only from its own records? Return one JSON object only: {"holds": true/false, "value": the listing\'s value as written, or null}. If none of its records states that property, return "holds": false and "value": null.'
MATCH_BRIEF = 'The assistant wants one particular listing, described by its category, its seller and some attribute values; candidates and their records follow. Return one JSON object only, without explanation: {"id": id of the candidate fitting that description, or null}. To fit, the seller must be the same, the category label must name the described kind, and each described attribute must agree with the candidate\'s records (same sense, units converted). For a candidate shown without its property table, its seller notes, title and description are its records, so notes stating the described values make it fit. With a PICK rule several candidates can fit: apply that rule to the fitting ones and return its choice.'
CONFIRM_BRIEF = 'Final gate before ordering. Using only the listing\'s own records, test it against EVERY requirement listed.\nRules:\n(1) Met when the listing\'s own property for the requirement states the value - any key naming the property; comma lists, synonyms, short forms, singular/plural and converted units all count. When STORE FIELDS show listings of this kind carrying a key named plainly as the property, a key naming one part of the product (a handle material, a lining material, a strap colour) does not stand in for it: without the plain key the requirement is unmet.\n(2) The seller\'s notes, options field, title or description may supply the value only when no property key covers it; a requirement nothing states is unmet; a value stored under a different property is no evidence.\n(3) Compatibility holds when this listing names the other product among what it fits, or when the other product\'s page names a family of compatible part codes that this listing\'s own part code begins with.\n(4) If the notes hand the property over to another listing, this listing\'s own value does not satisfy it.\n(5) Exclusions and closed lists are binding.\n(6) A listing carrying the requested category label is the right kind of product whatever its title says.\n(7) Price times quantity must not exceed the budget (same unit as the prices shown); a listing the records show cannot be bought fails.\nJudge exactly the listed requirements, adding none.\nReturn one JSON object: {"fits": true/false, "problems": [each requirement that is plainly unmet or that no record addresses]}.'
LABEL_BRIEF = 'Listings in this shop sit under category labels; the shopper wants one kind of product. Among the labels observed, return all those that ARE that kind of product: a label whose meaning covers the request\'s wording as a whole, broader labels that include it too (e.g. "sweaters and cardigans" when a cardigan is wanted). Neighbouring kinds stay out - accessories, parts, cases for the product, or a narrower or different type. Return one JSON object only: {"labels": [labels copied exactly]}, empty when nothing matches.'
MAP_BRIEF = 'Here is a JSON record returned by a shop tool. Tell which key of it serves each role below (exact key name, or null if none does), as one JSON object keyed by role: "id" = identifier of the listing; "title" = its name; "text" = its descriptive text; "price" = what it costs; "stock" = whether it can be purchased; "category" = its category label; "brand" = its brand; "seller" = the shop selling it; "facts" = the object that is its property table; "notes" = free-text remarks from the seller; "total" = a cart\'s overall price; "lines" = a cart\'s line list; "next" = marker or token for the following page of results; "count" = number of matching results overall.'

DIRECTOR_BRIEF = 'You make the final call in a shopping session that is about to end. For at least one product line the assistant found no listing that meets every requirement it understood, so its built-in rule would now end the task without buying. Before that happens, you decide how the task ends.\n\nThe shopper\'s own words are the authority: the request, their replies to the assistant\'s questions, and any later lines. Work out what the shopper wants done in exactly this situation: they may say what to do when nothing fully fits, which shortfalls are tolerable, how budget and stock count, how many lines to buy, or what to tell them. Follow such instructions literally, numbers included. If they give no instruction for this situation, decline. Declining is not a safe choice when their words do say what to do: it fails the task as surely as a wrong order. Facts that could not be confirmed, a referenced page that was not found, or a fit that could not be checked are shortfalls to count, not reasons to give up.\n\nThe task can end in only two ways: ordering (the basket is bought) or declining (nothing is bought). There is no way to show the shopper a listing and wait for approval, so when the shopper\'s words ask you to put forward, suggest, offer or go for a listing in this situation, that means ordering it, with the explanation telling them how it falls short.\n\nEach requirement is marked with its source. Requirements from the shopper\'s replies or later lines are as binding as the request itself. A requirement from the assistant\'s own reading of the request may be mis-stated; where that reading disagrees with the request text, the request text wins.\n\nEach product line shows the listings examined so far: price (in the shop\'s unit), whether it can be bought now (can_buy), whether its page was opened, the requirements it was found to fail (fails) or could not confirm (unsure), and other failed checks such as fit or conditions. An unopened listing has not been checked. Choose only listing ids shown for that line. Count quantity when you add up prices.\n\nWork in this order: first copy the shopper\'s words that say what to do when nothing fully fits (or "none"); then apply them line by line to the listings shown, keeping for each line a listing that meets everything when there is one; only when there are no such words, decline.\n\nAnswer with one JSON object: {"instruction": the shopper\'s words for this situation, copied, or "none", "decision": "order" or "decline", "lines": [{"item": product line name, "listing": listing id or null}], "explanation": one or two plain sentences for the shopper saying what is bought and where it misses, or why nothing is bought, "reasoning": one short sentence naming the shopper\'s words that decided this}. Keep the whole answer brief.'
REPAIR_BRIEF = 'A call to a shop tool is missing arguments, was rejected because of its arguments, or did not have the effect the assistant wanted. You get the tool\'s name, description and JSON schema, the arguments the assistant used, the error or the evidence of what went wrong, and what the assistant is trying to do. Return the full corrected arguments that do exactly what the assistant is trying to do and satisfy the schema. Argument names can mislead: trust the evidence over the name. A count of 0 normally adds nothing or removes. Keep every listing id as given; never invent one. Never answer with an argument set listed as already tried with no effect; change what the evidence shows did not work. For each argument you added or changed, name its meaning in this call: "quantity" (how many units), "position" (a slot or line index), "id", "choice", "text" or "other". Answer with one JSON object: {"args": {...}, "roles": {argument name: meaning}}.'

PREORDER_BRIEF = 'The assistant is about to order the basket below. Some rules in the shopper\'s request have no automatic check in the assistant\'s code; they are listed. Check the basket against those rules only: everything else was checked already. If the basket breaks one of them, fix it using only the listings shown for each product line: a different listing for a line, leaving a line out, or buying nothing. Answer with one JSON object: {"ok": true or false, "decision": "order" or "decline" (when ok is false), "lines": [{"item": product line name, "listing": listing id or null}] (when ok is false and decision is order: the whole basket to order instead), "reasoning": one short sentence}.'

RULE_MAP_BRIEF = 'An assistant\'s plan for a shopping task is shown, with rules from the request that were not yet put into the plan. For each rule, decide whether it changes something the plan already has: a top-level field (message_budget, call_budget, cheapest, best, in_stock_only, basket_checks, adjustments, conditions, fit_rules, closing_note, procedure, no_fit_rule) or one item (its requirements, best, unit_price_cap or quantity). A rule can overrule what the plan took from elsewhere in the request: the more specific or stricter instruction wins ("no questions at all" makes message_budget 0 even when the request also says how many messages it can manage). Use the shapes the plan already uses and only the values each field allows; a rule those shapes cannot express exactly stays in other_rules. A rule that changes nothing in the plan, or only repeats something the plan already holds, is dropped if repeated and otherwise kept in other_rules. Answer with one JSON object: {"set": {field: new value}, "item_changes": [{"item": item name, "add_requirements": [requirement objects], "best": text or null, "unit_price_cap": number or null, "quantity": number or null}], "other_rules": [rules still not mapped, word for word]}.'
NOTE_BRIEF = 'Write the closing note that ends a shopping task, for the shopper. It must do what the shopper asked of it, using only the facts given. Two sentences at most, plain words. Answer with one JSON object: {"note": the text}.'

MAPPABLE = ('message_budget', 'call_budget', 'cheapest', 'best', 'in_stock_only', 'basket_checks', 'adjustments', 'conditions', 'fit_rules', 'closing_note', 'procedure', 'no_fit_rule')

def _rule_map_message(ask: str, plan: dict) -> str:
    shown: dict = {k: v for k, v in plan.items() if k != 'tools'}
    return ''.join(("REQUEST:\n", ask, '\n\nPLAN SO FAR:\n', json.dumps(shown, ensure_ascii=False)[:9000], '\n\nRULES NOT YET IN THE PLAN:\n- ', '\n- '.join((str(r) for r in plan.get('other_rules') or []))))

def _apply_rule_map(plan: dict, answer: Any) -> dict:
    if not isinstance(answer, dict):
        return plan
    for k, v in (answer.get('set') or {}).items() if isinstance(answer.get('set'), dict) else []:
        if k in MAPPABLE and v is not None:
            plan[k] = v
    for ch in answer.get('item_changes') or []:
        if not isinstance(ch, dict) or not ch.get('item'):
            continue
        items: list = [i for i in plan.get('items') or [] if isinstance(i, dict)]
        if not items:
            continue
        it: dict = max(items, key=lambda i: SequenceMatcher(None, str(i.get('name') or '').lower(), str(ch['item']).lower()).ratio())
        for r in ch.get('add_requirements') or []:
            if isinstance(r, dict) and r.get('property'):
                it.setdefault('requirements', []).append(r)
        for k in ('best', 'unit_price_cap', 'quantity'):
            if ch.get(k) is not None:
                it[k] = ch[k]
    if isinstance(answer.get('other_rules'), list):
        plan['other_rules'] = [str(r) for r in answer['other_rules'] if r]
    return plan

def _clean_plan(plan: dict) -> dict:
    """Keep only basket checks and adjustments the code can evaluate; anything malformed goes back to other_rules (the director reads those)."""
    good: list = []
    for c in plan.get('basket_checks') or []:
        what: str = str((c or {}).get('what') or '').lower() if isinstance(c, dict) else ''
        op: str = str(c.get('op') or '').lower() if isinstance(c, dict) else ''
        num: bool = isinstance(c, dict) and _number(c.get('value')) is not None
        ok: bool = (what in ('same', 'distinct') and bool(c.get('field'))) or (what in ('total', 'share') and ('least' in op or 'most' in op) and num) or (what == 'compare' and ('least' in op or 'most' in op) and bool(c.get('item')) and bool(c.get('other')))
        if ok:
            good.append(c)
        else:
            plan.setdefault('other_rules', []).append('basket rule: ' + json.dumps(c, ensure_ascii=False)[:200])
    plan['basket_checks'] = good
    keep: list = []
    for a in plan.get('adjustments') or []:
        w: Any = a.get('when') if isinstance(a, dict) else None
        acts: list = [x for x in (a.get('then'), a.get('otherwise')) if isinstance(x, dict)] if isinstance(a, dict) else []
        ok = isinstance(w, dict) and str(w.get('op') or '').lower() in ('is', 'is not', 'one of', 'at least', 'at most', 'under', 'over') and bool(acts) and all((x.get('item') and (x.get('drop') or _number(x.get('quantity')) is not None) for x in acts))
        if ok:
            keep.append(a)
        else:
            plan.setdefault('other_rules', []).append('adjustment: ' + json.dumps(a, ensure_ascii=False)[:200])
    plan['adjustments'] = keep
    return plan

SOURCE_LABEL = {'request': "the assistant's reading of the request", 'shopper': "the shopper's reply or a later line", 'condition': 'a condition in the request'}

def _director_message(ask: str, heard: list, dialogue: list, cap: Any, steps: int, reason: str, blocks: list, rules: list | None=None) -> str:
    talk: list = [''.join(('Q: ', str(d.get('asked') or '')[:300], '\n   A: ', str(d.get('reply') or '')[:300])) for d in dialogue[-8:]]
    return ''.join(("REQUEST (the shopper's own words):\n", ask,
                    '\n\nQUESTIONS THE ASSISTANT ASKED AND THE SHOPPER\'S REPLIES:\n', '\n'.join(talk) if talk else '(none)',
                    '\n\nLATER LINES FROM THE SHOPPER:\n- ', '\n- '.join((h[:220] for h in heard)) if heard else '(none)',
                    '\n\nRULES IN THE REQUEST THAT THE ASSISTANT HAS NO AUTOMATIC CHECK FOR:\n- ' + '\n- '.join((str(r)[:220] for r in rules)) if rules else '',
                    '\n\nTOTAL BUDGET AS THE ASSISTANT UNDERSTOOD IT: ', format(cap, '') if cap is not None else 'none stated',
                    '\nSTEPS LEFT: ', str(steps), ' (each basket line costs a step and ordering costs one more)',
                    '\nWHY THE ASSISTANT WAS ABOUT TO END WITHOUT BUYING: ', (reason or 'no listing met every requirement')[:300],
                    '\n\nPRODUCT LINES:\n', _compact(blocks, 16000)))

class Offer:

    def __init__(self, id_: str):
        self.ident = id_
        self.title = ''
        self.text = ''
        self.cost_of: float | None = None
        self.stock: bool | None = None
        self.category = ''
        self.brand = ''
        self.seller = ''
        self.data_points: dict[str, Any] = {}
        self.remarks = ''
        self.unfolded = False
        self.leaf_read = False
        self.raw: dict[str, Any] = {}
        self.sift = 'maybe'

    def entry2(self, limit: int=CHARS_FACT) -> dict[str, Any]:
        rec: dict[str, Any] = dict(id=self.ident, title=self.title[:160])
        if bool(self.category):
            rec['category'] = self.category
        if bool(self.brand):
            rec['brand'] = self.brand
        if bool(self.seller):
            rec['seller'] = self.seller
        if self.cost_of is not None:
            rec['price'] = self.cost_of
        if self.stock is not None:
            rec['can_buy'] = self.stock
        if bool(self.data_points):
            rec['properties'] = self.data_points
        if bool(self.remarks):
            rec['seller_notes'] = self.remarks[:600]
        if bool(self.text):
            rec['description'] = self.text[:500] if not self.data_points else self.text[:400]
        text: Any = json.dumps(rec, ensure_ascii=False)
        if len(text) > limit and self.data_points:
            rec['properties'] = dict(list(self.data_points.items())[:max(6, int(len(self.data_points) * limit / len(text)))])
        if len(json.dumps(rec, ensure_ascii=False)) > limit and rec.get('description'):
            rec['description'] = rec['description'][:200]
        return rec
ANSWER_TOOL = dict(name='give_answer', description='Hand over the answer the instructions ask for.', parameters=dict(type='object', properties=dict(answer=dict(type='object', description='The JSON object the instructions describe, exactly as it would be written as the reply.')), required=['answer']))
_NOT_ASKED = {'options', 'option', 'model', 'condition', 'warranty', 'brand', 'seller', 'delivery', 'care', 'instructions', 'sku', 'id', 'title', 'number', 'code'}

def _close(a: str, b: str) -> bool:
    if a == b:
        return True
    _rv19: Any = False if min(len(a), len(b)) < 5 or abs(len(a) - len(b)) > 1 else SequenceMatcher(None, a, b).ratio() >= 0.85
    return _rv19

def _inside_range(required: str, written: str) -> bool:
    span: Any = re.search('(\\d+(?:\\.\\d+)?)\\s*(?:-|–|to)\\s*(\\d+(?:\\.\\d+)?)', required)
    figures: Any = re.findall('\\d+(?:\\.\\d+)?', written)
    if not span or len(figures) != 1:
        return False
    low, high = sorted((float(span.group(1)), float(span.group(2))))
    _rv20: bool = low <= float(figures[0]) <= high
    return _rv20

def _spelled(words: Any) -> frozenset:
    _rv21: Any = frozenset(words)
    return _rv21

def _within(want: Any, have: Any) -> bool:
    _rv22: bool = all((any((_close(w, h) for h in have)) for w in want))
    return _rv22

class Requirement:

    def __init__(self, prop: str, op: str, value: str, reference: dict | None=None, origin: str='request'):
        self.prop = str(prop or '').strip()
        self.op = str(op or 'is').strip().lower()
        self.value = str(value or '').strip()
        self.reference = reference if isinstance(reference, dict) else None
        self.origin = origin

    def text(self) -> str:
        _rv23: Any = ''.join((format(self.prop, ''), ': ', format(self.op, ''), ' (still to be looked up)')) if self.op in OPS_REF and self.reference else ''.join((format(self.prop, ''), ': ', format(self.op, ''), ' ', format(self.value, '')))
        return _rv23

    def key(self) -> str:
        _rv24: Any = _norm(self.prop)
        return _rv24

class Article:

    def __init__(self, spec: dict[str, Any], index: int):
        self.index = index
        self.name = str(spec.get('name') or spec.get('category') or 'the product').strip()
        self.category = str(spec.get('category') or self.name).strip()
        qty: Any = spec.get('quantity')
        self.amount = int(qty) if isinstance(qty, (int, float)) and (not isinstance(qty, bool)) and (qty >= 1) else 1
        self.requirements: list[Requirement] = []
        for r in iter(spec.get('requirements') or []):
            if isinstance(r, dict) and r.get('property'):
                self.requirements.append(Requirement(r.get('property'), r.get('op'), r.get('value'), r.get('reference')))
        self.unit_cap = _number(spec.get('unit_price_cap'))
        self.best = spec.get('best') if isinstance(spec.get('best'), str) and spec.get('best') else None
        self.hints = [str(h) for h in spec.get('hints') or [] if h]
        self.words2 = [str(t) for t in spec.get('search_terms') or [] if t]
        if not bool(self.words2):
            self.words2 = [self.category]
        self.noun = self.words2[-1]
        self.pool: dict[str, Offer] = {}
        self.rulings: dict[str, dict[str, set[str]]] = {}
        self.for_judged: dict[str, set[str]] = {}
        self.queried: list[str] = []
        self.freed: list[str] = []
        self.lookup_done = 0
        self.exhausted = False
        self.chosen: Offer | None = None
        self.labels_category: dict[str, int] = {}
        self.ok_labels: set[str] | None = None
        self.cursors: dict[str, dict[str, Any]] = {}

    def active(self) -> list[Requirement]:
        _rv25: list = [r for r in self.requirements if not (r.reference and r.op in OPS_REF)]
        return _rv25

    def req_proses(self) -> list[str]:
        _rv26: list = [r.text() for r in self.active()]
        return _rv26

    def exactness(self, listing: Offer) -> int:
        if not bool(listing.data_points):
            return 0
        elements: set[frozenset[str]] = set()
        for v in iter(listing.data_points.values()):
            for e in iter(re.split('[,;/|]', str(v))):
                t: Any = frozenset(_pieces_of_text(e))
                if bool(t):
                    elements.add(t)
        score: int = 0
        for r in iter(self.active()):
            if r.op not in ('is', 'one of') or not r.value:
                continue
            wants: list = []
            for _it27 in r.value.split('|'):
                if _pieces_of_text(_it27):
                    wants.append(frozenset(_pieces_of_text(_it27)))
            if any((w in elements for w in wants)):
                score += 1
        return score

    def matches(self, include_unsure: bool=False, cheapest: bool=False, strong_only: bool=False) -> list[Offer]:
        sure, unsure = ([], [])
        for lid, listing in iter(self.pool.items()):
            if not bool(listing.unfolded):
                continue
            v: Any = self.rulings.get(lid)
            if v is None or v['failed']:
                continue
            if strong_only and v.get('weak'):
                continue
            if self.for_judged.get(lid, set()) >= set(self.req_proses()):
                (unsure if v['unsure'] else sure).append(listing)
        key: Any = lambda l: l.cost_of if l.cost_of is not None else 1e+18
        if bool(cheapest):
            sure.sort(key=lambda l: (key(l), -self.exactness(l)))
        else:
            sure.sort(key=lambda l: (-self.exactness(l), len(self.rulings[l.ident].get('weak') or ()), key(l)))
        unsure.sort(key=lambda l: (len(self.rulings[l.ident]['unsure']), key(l)))
        _rv28: Any = sure if sure or not include_unsure else unsure
        return _rv28
_FIGURE = re.compile('(\\d+(?:\\.\\d+)?)\\s*([a-zA-Z]+|\\")?')
_JOINERS = {'to', 'and', 'or', 'x', 'by', 'of', 'for', 'per'}

def _doubt(req: Requirement, listing: Offer) -> bool:
    value: Any = req.value or ''
    if req.op == 'is' and ',' in value and (not re.search('\\bor\\b', value)):
        text: Any = _tokens2(' '.join([listing.title or '', listing.text or '', ' '.join((str(v) for v in listing.data_points.values()))]))
        if not all((_tokens2(p) <= text for p in value.split(',') if _tokens2(p))):
            return True
    return False

def _own(listing: Offer, prop: str) -> list[str]:
    want: Any = _tokens2(prop)
    out: list = []
    for k, v in iter(listing.data_points.items()):
        label: Any = _tokens2(re.sub('(?<=[a-z0-9])(?=[A-Z])', ' ', str(k)).replace('_', ' ').replace('-', ' '))
        if label and (label == want or want <= label or _near(' '.join(sorted(label)), prop) or _near(str(k).replace('_', ' '), prop)):
            out.append(str(v))
    return out

def _proven(req: Requirement, listing: Offer) -> bool:
    own: Any = _own(listing, req.prop)
    if not own or not req.value:
        return False
    if req.op in ('at least', 'at most'):
        want: Any = _FIGURE.search(req.value)
        if not bool(want):
            return False
        unit: Any = _units_in(req.value)
        for f in iter(own):
            for n, u in iter(_FIGURE.findall(f)):
                if u and u.lower() not in _JOINERS and unit and (not any((_same_unit(_unit(u), x) for x in unit))):
                    continue
                n: float = float(n)
                if n >= float(want.group(1)) - 1e-09 if req.op == 'at least' else n <= float(want.group(1)) + 1e-09:
                    return True
        return False
    alts: list = []
    for _it29 in req.value.split('|'):
        if _pieces_of_text(_it29):
            alts.append(_spelled(_pieces_of_text(_it29)))
    elems: list = []
    for _it31 in own:
        for _it30 in re.split('[,;/|+、]', _it31):
            if _pieces_of_text(_it30):
                elems.append(_spelled(_pieces_of_text(_it30)))
    if req.op == 'is not':
        _rv32: Any = bool(elems) and (not any((_within(a, e) for a in alts for e in elems)))
        return _rv32
    _rv33: Any = any((_within(a, e) and len(e) <= len(a) + 1 for a in alts for e in elems)) if req.op in ('is', 'one of', 'contains') else False
    return _rv33

def _same_unit(a: str, b: str) -> bool:
    _rv34: Any = a == b or (min(len(a), len(b)) >= 2 and (a.startswith(b) or b.startswith(a)))
    return _rv34

def _scaled(a: str, b: str) -> bool:
    _rv35: Any = len(a) == len(b) == 2 and a[-1] == b[-1] and (a[0] != b[0])
    return _rv35

def _unit(word: str) -> str:
    w: Any = word.lower()
    _rv36: Any = w[:-2] if len(w) > 4 and w.endswith('es') else w[:-1] if len(w) > 3 and w.endswith('s') else w
    return _rv36

def _units_in(text: str) -> set[str]:
    _rv37: set = {_unit(u) for _, u in _FIGURE.findall(text) if u and u.lower() not in _JOINERS}
    return _rv37

class Wiring:

    def article_named(self, name: Any) -> Article | None:
        name: str = str(name or '')
        if not name or not self.items:
            return None
        best: Any = max(self.items, key=lambda it: max(_similar(it.name, name), _similar(it.category, name)))
        _rv38: Any = best if max(_similar(best.name, name), _similar(best.category, name)) >= 0.2 else None
        return _rv38

    def bundle_unfold(self) -> bool:
        _rv39: Any = bool(self.jobs2.get('compare')) and (not getattr(self, 'off_compare', False)) and (not self.compare_thin)
        return _rv39

    def buyable(self, item: Article) -> list[Offer]:
        table: Any = self.slot_grid(item)
        _rv40: list = [f for f in item.matches(cheapest=self.cheapest) if f.stock is not False and (not self.slot_fails(item, f, table)) and (not (getattr(item, 'unit_cap', None) is not None and f.cost_of is not None and f.cost_of > item.unit_cap + 1e-06))]
        return _rv40

    def invoke(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        try:
            return self.invoke_fixed(name, args)
        finally:
            self.call_intent = ''

    def invoke_fixed(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        schema: Any = self.specs.get(name) or {}
        props: Any = schema.get('properties') or {}
        args = dict(args)
        for k, v in iter((getattr(self, 'arg_patch', {}).get(name) or {}).items()):
            if k in props and k not in args:
                args[k] = v
        missing: list = [k for k in schema.get('required') or [] if k not in args]
        if missing and not getattr(self, 'repairing', False):
            fixed: Any = self.repair_args(name, args, 'missing required properties: ' + ', '.join(missing))
            if isinstance(fixed, dict):
                args = fixed
        out: dict[str, Any] = self.invoke_once(name, args)
        msg: Any = self.jobs2.get('message') or {}
        if name == msg.get('name') and out.get('ok'):
            self.dialogue = getattr(self, 'dialogue', []) + [dict(asked=str(args.get(msg.get('text') or '') or '')[:400], reply=_compact(out.get('obs'), 400))]
        err: str = str(out.get('error') or '')
        if out.get('ok') or not err or getattr(self, 'repairing', False) or self.done2:
            return out
        if not re.search('argument|required|propert|schema|invalid|must be|expected|not allowed', err, re.I):
            return out
        fixed: Any = self.repair_args(name, args, err)
        if not isinstance(fixed, dict) or fixed == args:
            return out
        return self.invoke_once(name, fixed)

    def repair_args(self, name: str, args: dict[str, Any], error: str) -> dict[str, Any] | None:
        self.repairing = True
        try:
            schema: Any = self.specs.get(name) or {}
            user: str = ''.join(('TOOL NAME: ', name, '\nWHAT IT DOES: ', self.instrument_description(name), '\nSCHEMA: ', _compact(schema, 3000),
                                 '\nARGUMENTS SO FAR: ', _compact(args, 1500), '\nERROR OR WHAT IS MISSING: ', error[:400],
                                 '\nWHAT THE ASSISTANT IS TRYING TO DO: ', getattr(self, 'call_intent', '') or 'use this tool for its stated purpose'))
            self.urgent = self.time_remaining() < CONCLUDE_SECS
            try:
                answer: Any = self.llm.json([FAST_LLM, BACKUP_LLM], REPAIR_BRIEF, user, 400)
            finally:
                self.urgent = False
            self.remark2('arg_repair', tool=name, error=error[:160], answer=_compact(answer, 300))
            if not isinstance(answer, dict) or not isinstance(answer.get('args'), dict):
                return None
            props: Any = schema.get('properties') or {}
            fixed: dict[str, Any] = {k: v for k, v in answer['args'].items() if k in props}
            for k, v in iter(args.items()):
                if isinstance(v, str) and v.startswith('p-') and k in fixed and fixed[k] != v:
                    fixed[k] = v
            roles: Any = answer.get('roles') if isinstance(answer.get('roles'), dict) else {}
            cart: Any = self.jobs2.get('cart_add') or {}
            for k, role in iter(roles.items()):
                if k not in fixed:
                    continue
                role = str(role).lower()
                if role == 'quantity' and cart.get('name') == name and not cart.get('qty'):
                    cart['qty'] = k
                elif role in ('choice', 'text', 'other') and k not in args:
                    self.arg_patch = dict(getattr(self, 'arg_patch', {}))
                    self.arg_patch[name] = dict(self.arg_patch.get(name) or {}, **{k: fixed[k]})
            return fixed
        finally:
            self.repairing = False

    def invoke_once(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = dict(obs=None, lines=[], error=None, ok=False)
        if self.done2 or self.dead or name not in self.specs:
            out['error'] = 'unavailable'
            return out
        finishing: bool = name in {(self.jobs2.get(r) or {}).get('name') for r in ('checkout', 'decline')}
        if self.moves_remaining() <= 0 or (self.moves_remaining() <= 1 and (not finishing)):
            out['error'] = 'the step allowance is used up'
            return out
        payload: dict[str, Any] = dict(self.link)
        payload['step'] = self.move + 1
        payload['action'] = dict(name=name, args=args)
        for _ in range(3):
            left: Any = self.time_remaining()
            if left < 3:
                out['error'] = 'no time'
                return out
            _env.timeout = int(max(4, min(90, left - 1)))
            result: Any = _env.post_verbose('/environment/call', payload)
            data: Any = result.data
            if isinstance(data, dict) and ('observation' in data or 'finished' in data):
                self.move = data['step'] if isinstance(data.get('step'), int) else self.move + 1
                if bool(data.get('finished')):
                    self.done2 = True
                obs, lines = _unwrap(data.get('observation'))
                for line in iter(lines):
                    if line not in self.handled_rows:
                        self.pending_rows.append(line)
                out['obs'] = obs
                out['lines'] = lines
                if isinstance(obs, dict) and isinstance(obs.get('error'), str) and (len(obs) <= 2):
                    out['error'] = obs['error']
                else:
                    out['ok'] = True
                    add: Any = self.jobs2.get('cart_add') or {}
                    if name == add.get('name') and isinstance(args.get(add.get('id')), str):
                        self.cart_sent = dict(getattr(self, 'cart_sent', {}), **{args[add['id']]: dict(args)})
                self.remark2('call', tool=name, args=_compact(args, 160), result=_compact(obs, 200), lines=lines[:3])
                entry: dict = dict(step=self.move, tool=name, args=_compact(args, 300 if name == (self.jobs2.get('message') or {}).get('name') else 120), result=_compact(obs, 160))
                if bool(lines):
                    entry['shopper_lines'] = [l[:200] for l in lines[:3]]
                if name == (self.jobs2.get('message') or {}).get('name'):
                    entry['basket_then'] = [list(x) for x in self.trolley_sig()]
                self.past.append(entry)
                return out
            error: Any = result.error or {}
            status: Any = error.get('status') if isinstance(error, dict) else None
            body: Any = str(error.get('body') if isinstance(error, dict) else error)[:300]
            self.remark2('env_failure', status=status, body=body)
            if status == 402:
                self.dead = True
                out['error'] = 'payment'
                return out
            if status == 409:
                if 'finished' in body:
                    self.done2 = True
                    out['error'] = 'finished'
                    return out
                m: Any = re.search('expected (\\d+)', body)
                if bool(m):
                    self.move = int(m.group(1)) - 1
                    payload['step'] = self.move + 1
                    continue
                out['error'] = 'conflict'
                return out
            if status in (403, 422):
                out['error'] = ''.join(('refused ', format(status, '')))
                return out
        out['error'] = 'env failure'
        return out

    def mapping(self, kind: str, sample: dict[str, Any]) -> dict[str, str]:
        key: Any = kind + ':' + ','.join(sorted(sample.keys()))
        if key in self.maps:
            _rv43: Any = self.maps[key]
            return _rv43
        for known_key, known in iter(self.maps.items()):
            if known_key.startswith(kind + ':') and known:
                present: float = sum((1 for k in known.values() if k in sample))
                if present >= max(2, int(0.6 * len(known))):
                    self.maps[key] = known
                    return known
        shown: dict = {}
        for _it41, _it42 in sample.items():
            shown[_it41] = _it42 if not isinstance(_it42, (dict, list)) else _compact(_it42, 160)
        for k, v in iter(list(shown.items())):
            if isinstance(v, str) and len(v) > 160:
                shown[k] = v[:160] + '…'
        answer: Any = self.llm.json([FAST_LLM, BACKUP_LLM], MAP_BRIEF, json.dumps(shown, ensure_ascii=False), 700)
        result: dict[str, str] = {}
        if isinstance(answer, dict):
            for role, k in iter(answer.items()):
                if isinstance(k, str) and k in sample:
                    result[role] = k
        if 'facts' not in result:
            for k, v in iter(sample.items()):
                if isinstance(v, dict) and v and all((isinstance(x, (str, int, float)) or x is None for x in v.values())):
                    result['facts'] = k
                    break
        if 'id' not in result:
            for k, v in iter(sample.items()):
                if isinstance(v, str) and re.fullmatch('[A-Za-z]{0,3}-?[0-9a-f]{8,}', v):
                    result['id'] = k
                    break
        if 'price' not in result:
            for k, v in iter(sample.items()):
                if isinstance(v, (int, float)) and (not isinstance(v, bool)) and ('price' in k.lower()):
                    result['price'] = k
                    break
        self.maps[key] = result
        self.remember()
        return result

    def merge(self, item: Article, listing: Offer, opened: bool) -> Offer:
        known: Any = item.pool.get(listing.ident)
        if known is None:
            item.pool[listing.ident] = listing
            known: Any = listing
        else:
            for attr in iter(('title', 'text', 'category', 'brand', 'seller', 'remarks')):
                if getattr(listing, attr) and (opened or not getattr(known, attr)):
                    setattr(known, attr, getattr(listing, attr))
            if listing.cost_of is not None:
                known.cost_of = listing.cost_of
            if listing.stock is not None:
                known.stock = listing.stock
            if bool(listing.data_points):
                if listing.data_points != known.data_points:
                    item.for_judged.pop(known.ident, None)
                known.data_points = listing.data_points
            if listing.remarks and listing.remarks != known.remarks and opened:
                item.for_judged.pop(known.ident, None)
            known.leaf_read = known.leaf_read or listing.leaf_read
            known.raw = listing.raw or known.raw
        if bool(opened):
            known.unfolded = True
            self.unfolded_leaves.add(known.ident)
        if bool(known.category):
            item.labels_category[known.category] = item.labels_category.get(known.category, 0) + 1
        return known

    def moves_remaining(self) -> int:
        _rv44: Any = self.top_moves - self.move
        return _rv44

    def offer_from(self, rec: dict[str, Any], kind: str) -> Offer | None:
        m: Any = self.mapping(kind, rec)
        idk: Any = m.get('id')
        if not idk or not isinstance(rec.get(idk), (str, int)):
            return None
        lid: str = str(rec[idk])
        listing: Any = Offer(lid)
        listing.raw = rec
        if m.get('title') and isinstance(rec.get(m['title']), str):
            listing.title = rec[m['title']]
        if m.get('text') and isinstance(rec.get(m['text']), str):
            listing.text = rec[m['text']]
        if bool(m.get('price')):
            listing.cost_of = _number(rec.get(m['price']))
        if m.get('stock') and isinstance(rec.get(m['stock']), bool):
            listing.stock = rec[m['stock']]
        if m.get('category') and isinstance(rec.get(m['category']), str):
            listing.category = rec[m['category']]
        if m.get('brand') and isinstance(rec.get(m['brand']), str):
            listing.brand = rec[m['brand']]
        if m.get('seller') and isinstance(rec.get(m['seller']), str):
            listing.seller = rec[m['seller']]
        if m.get('facts') and isinstance(rec.get(m['facts']), dict):
            listing.data_points = {str(k): v for k, v in rec[m['facts']].items() if v not in (None, '')}
        if m.get('notes') and isinstance(rec.get(m['notes']), str):
            listing.remarks = rec[m['notes']]
        return listing

    def remark2(self, kind: str, **fields: Any) -> None:
        entry: dict = dict(kind=kind, step=self.move, t=round(time.monotonic() - self.started, 1))
        entry.update(fields)
        self.log.append(entry)

    def remember(self) -> None:
        _MEMORY_WORLD[self.world_label] = dict(maps=self.maps, compare_limit=self.compare_cap, thin_compare=self.compare_thin)

    def spent(self) -> bool:
        _rv45: bool = self.llm.price_tag >= COST_CAP
        return _rv45

    def time_remaining(self) -> float:
        _rv46: Any = TASK_SECS - (time.monotonic() - self.started)
        return _rv46

    def _trim(self, role: str, arg: str, value: str) -> str:
        props: Any = self._spec(role).get('properties') or {}
        limit: Any = (props.get(arg) or {}).get('maxLength')
        if isinstance(limit, int) and len(value) > limit:
            value: Any = value[:limit].rsplit(' ', 1)[0] if ' ' in value[:limit] else value[:limit]
        return value

    def _trim_params(self, prop: dict[str, Any], value: str) -> str:
        limit: Any = prop.get('maxLength')
        _rv47: Any = value[:limit] if isinstance(limit, int) and len(value) > limit else value
        return _rv47

class Groundwork:

    def _enum_val(self, role: str, arg: str, value: Any) -> Any:
        props: Any = self._spec(role).get('properties') or {}
        enum: Any = (props.get(arg) or {}).get('enum')
        if not isinstance(enum, list) or not enum or (not isinstance(value, str)):
            return value
        by: dict = {}
        for _it48 in enum:
            if isinstance(_it48, str):
                by[_norm(_it48)] = _it48
        if not bool(by):
            return value
        if _norm(value) in by:
            _rv49: Any = by[_norm(value)]
            return _rv49
        best: Any = max(by, key=lambda e: _similar(e, value))
        _rv50: Any = by[best] if _similar(best, value) >= 0.5 else None
        return _rv50

    def instrument_description(self, name: str) -> str:
        for t in iter(self.instruments):
            fn: Any = t.get('function') if isinstance(t.get('function'), dict) else t
            if isinstance(fn, dict) and fn.get('name') == name:
                _rv51: str = str(fn.get('description') or '')
                return _rv51
        return ''

    def _leaf_mode(self, role: str) -> str | None:
        spec: Any = self.jobs2.get(role) or {}
        page: Any = spec.get('page')
        if not bool(page):
            return None
        if self._param_sort(role, page) == 'integer':
            kind: Any = str(spec.get('page_kind') or '').lower()
            name: Any = page.lower()
            if kind == 'offset' or (kind != 'number' and ('offset' in name or 'start' in name or 'skip' in name)):
                return 'offset'
            return 'index'
        return 'token'

    def _match_phrase(self, item: Article, role: str, arg: str, query: str) -> str:
        words: Any = query.split()
        cat: Any = _pieces_of_text(item.category) | _pieces_of_text(item.name)
        values: list = []
        for _it52 in words:
            if not (_pieces_of_text(_it52) and _pieces_of_text(_it52) <= cat):
                values.append(_it52)
        rest: list = []
        for _it53 in words:
            if _it53 not in values:
                rest.append(_it53)
        query: str = ' '.join(values + rest)
        props: Any = self._spec(role).get('properties') or {}
        limit: Any = (props.get(arg) or {}).get('maxLength')
        if not isinstance(limit, int) or len(query) <= limit:
            return query
        kept: list = list(values)
        for w in iter(rest):
            if len(' '.join(kept + [w])) <= limit:
                kept.append(w)
        _rv54: Any = self._trim(role, arg, ' '.join(kept) if kept else query)
        return _rv54

    def _param_sort(self, role: str, arg: str | None) -> str | None:
        if not bool(arg):
            return None
        props: Any = self._spec(role).get('properties') or {}
        _rv55: Any = (props.get(arg) or {}).get('type')
        return _rv55

    def pinned_params(self, role: str) -> dict[str, Any]:
        spec: Any = self.jobs2.get(role) or {}
        schema: Any = self._spec(role)
        props: Any = schema.get('properties') or {}
        out: dict[str, Any] = {}
        for k, v in iter((spec.get('args') or {}).items() if isinstance(spec.get('args'), dict) else []):
            if k not in props or v in (None, ''):
                continue
            enum: Any = props[k].get('enum')
            if isinstance(enum, list) and enum and (v not in enum):
                best: Any = max(enum, key=lambda e: _similar(str(e), str(v)))
                if _similar(str(best), str(v)) < 0.5:
                    continue
                v: Any = best
            out[k] = v
        missing: list = []
        for _it56 in schema.get('required') or []:
            if _it56 not in out and isinstance((props.get(_it56) or {}).get('enum'), list):
                missing.append(_it56)
        if bool(missing):
            cached: Any = getattr(self, 'option_stash', {}).get(role)
            if cached is None:
                goal: Any = 'buy what the cart now holds' if role == 'checkout' else 'close the task with no purchase'
                user: str = ''.join(('TOOL NAME: ', format(spec.get('name'), ''), '\nWHAT IT DOES: ', format(self.instrument_description(spec.get('name') or ''), ''), '\nARGUMENTS STILL NEEDED: ', format(json.dumps({k: props[k] for k in missing}, ensure_ascii=False), ''), '\nAIM: ', format(goal, '')))
                answer: Any = self.llm.json([FAST_LLM, BACKUP_LLM], OPTION_BRIEF, user, 300)
                cached: Any = answer if isinstance(answer, dict) else {}
                if bool(cached):
                    self.option_stash = dict(getattr(self, 'option_stash', {}), **{role: cached})
            for k in iter(missing):
                enum: Any = props[k]['enum']
                v: Any = cached.get(k)
                if v in enum:
                    out[k] = v
        return out

    def setup(self) -> bool:
        tool_lines: list = []
        for t in iter(self.instruments):
            fn: Any = t.get('function') if isinstance(t.get('function'), dict) else t
            tool_lines.append(dict(name=fn.get('name'), description=fn.get('description', ''), parameters=fn.get('parameters')))
        user: Any = 'REQUEST:\n' + self.ask2
        if self.opening is not None:
            user += '\n\nOPENING OBSERVATION:\n' + _compact(self.opening, 1500)
        user += '\n\nTOOLS:\n' + json.dumps(tool_lines, ensure_ascii=False)
        plan: Any = self.llm.json([PARSE_LLM, BACKUP_LLM], READ_BRIEF, user, 2500)
        if not isinstance(plan, dict) or not isinstance(plan.get('tools'), dict):
            plan: Any = self.llm.json([PARSE_LLM, FAST_LLM], READ_BRIEF, user, 2500)
        if not isinstance(plan, dict):
            self.remark2('setup_failed')
            return False
        if plan.get('other_rules'):
            mapped: Any = self.llm.json([PARSE_LLM, FAST_LLM], RULE_MAP_BRIEF, _rule_map_message(self.ask2, plan), 1500)
            plan = _apply_rule_map(plan, mapped)
            self.remark2('rule_map', answer=_compact(mapped, 400))
        plan = _clean_plan(plan)
        self.scheme = plan
        roles: Any = plan.get('tools') if isinstance(plan.get('tools'), dict) else {}
        self.jobs2 = {}
        for role, spec in iter(roles.items()):
            if isinstance(spec, dict) and isinstance(spec.get('name'), str) and (spec['name'] in self.specs):
                self.jobs2[role] = spec
        self.vet_jobs()
        for role in iter(('search_text', 'open', 'checkout', 'decline', 'message')):
            if role not in self.jobs2:
                self.remark2('role_missing', role=role)
        items: Any = plan.get('items') if isinstance(plan.get('items'), list) else []
        self.items = [Article(s, i) for i, s in enumerate(items) if isinstance(s, dict)]
        if not bool(self.items):
            self.items = [Article(dict(name='the product', category='', search_terms=[self.ask2[:60]]), 0)]
        for item in iter(self.items):
            leaked: set[tuple[str, str]] = set()
            for r in iter(item.requirements):
                if r.reference and isinstance(r.reference.get('attributes'), dict):
                    for k, v in iter(r.reference['attributes'].items()):
                        leaked.add((_norm(k), _norm(v)))
            if bool(leaked):
                item.requirements = [r for r in item.requirements if r.reference or (_norm(r.prop), _norm(r.value)) not in leaked]
                item.words2 = [t for t in item.words2 if not any((v and v in _norm(t) for _, v in leaked))] or [item.category]
            name_words: Any = _pieces_of_text(item.category)
            item.requirements = [r for r in item.requirements if r.reference or r.op not in ('is',) or (not (_pieces_of_text(r.value) and _pieces_of_text(r.value) <= name_words))]
            for r in iter(item.requirements):
                if r.op in ('is', 'one of', 'contains') and r.value:
                    for v in iter(r.value.split('|')):
                        v: Any = v.strip()
                        forms: Any = [v] + ([v.replace('-', ' ')] if '-' in v else [])
                        for form in iter(forms):
                            if form and len(form) <= 40 and (not any((_norm(form) == _norm(t) for t in item.words2))):
                                item.words2.append(form)
        self.cap = _number(plan.get('total_cap'))
        rule: Any = plan.get('no_fit_rule')
        self.no_fit = rule if isinstance(rule, dict) else {}
        self.other_rules = [str(r) for r in plan.get('other_rules') or [] if r]
        self.closing_note = str(plan.get('closing_note') or '').strip()
        self.basket_checks = [c for c in plan.get('basket_checks') or [] if isinstance(c, dict) and c.get('what')]
        self.adjustments = [a for a in plan.get('adjustments') or [] if isinstance(a, dict) and isinstance(a.get('when'), dict)]
        self.cheapest = bool(plan.get('cheapest'))
        self.best = plan.get('best') if isinstance(plan.get('best'), str) and plan.get('best') else None
        mb: Any = plan.get('message_budget')
        self.note_allowance = int(mb) if isinstance(mb, (int, float)) and (not isinstance(mb, bool)) and (mb >= 0) else None
        cb: Any = plan.get('call_budget')
        if isinstance(cb, (int, float)) and (not isinstance(cb, bool)) and (0 < cb < self.top_moves):
            self.top_moves = int(cb)
        self.in_stock_solely = bool(plan.get('in_stock_only', True))
        self.query_about = [str(a) for a in plan.get('ask_about') or [] if a]
        self.match_rules = [r for r in plan.get('fit_rules') or [] if isinstance(r, dict)]
        self.conditions = self.conditions_read(plan.get('conditions'))
        self.routine = [st for st in plan.get('procedure') or [] if isinstance(st, dict) and st.get('goal')][:6]
        for st in iter(self.routine):
            st['moment'] = str(st.get('moment') or 'before_finish').lower()
            st['kind'] = 'message' if str(st.get('kind') or '').lower() == 'message' else 'tool'
        self.confirm_initial = any((st['kind'] == 'message' and st['moment'] in ('before_order', 'before_finish') for st in self.routine))
        self.remark2('plan', items=[(i.name, i.amount, i.req_proses(), i.hints) for i in self.items], cap=self.cap, cheapest=self.cheapest, messages=self.note_allowance, steps=self.top_moves, roles={k: v.get('name') for k, v in self.jobs2.items()})
        return True

    def _spec(self, role: str) -> dict[str, Any]:
        spec: Any = self.jobs2.get(role) or {}
        _rv57: Any = self.specs.get(spec.get('name') or '', {})
        return _rv57

    def vet_jobs(self) -> None:
        for role in iter(('cart_add', 'cart_set')):
            spec: Any = self.jobs2.get(role)
            if not bool(spec):
                continue
            props: Any = (self.specs.get(spec['name']) or {}).get('properties') or {}
            lists: list = []
            for _it58, _it59 in props.items():
                if isinstance(_it59, dict) and _it59.get('type') == 'array' and isinstance(_it59.get('items'), dict) and (_it59['items'].get('type') == 'object'):
                    lists.append(_it58)
            if bool(lists):
                line: Any = lists[0]
                line_props: Any = props[line]['items'].get('properties') or {}
                idk: Any = spec.get('id') if role == 'cart_set' and spec.get('id') in line_props else next((k for k, v in line_props.items() if isinstance(v, dict) and v.get('type') == 'string'), None)
                qk: Any = spec.get('qty') if role == 'cart_set' and spec.get('qty') in line_props else next((k for k, v in line_props.items() if isinstance(v, dict) and v.get('type') == 'integer'), None)
                if bool(idk):
                    self.jobs2['cart_set'] = dict(name=spec['name'], lines=line, id=idk, qty=qk)
                if role == 'cart_add':
                    self.jobs2.pop('cart_add', None)
                continue
            if role == 'cart_add':
                if spec.get('id') not in props:
                    spec['id'] = next((k for k, v in props.items() if isinstance(v, dict) and v.get('type') == 'string'), None)
                if spec.get('qty') and spec['qty'] not in props:
                    spec['qty'] = next((k for k, v in props.items() if isinstance(v, dict) and v.get('type') == 'integer'), None)
                if not bool(spec.get('id')):
                    self.jobs2.pop('cart_add', None)
        for have, other in iter((('checkout', 'decline'), ('decline', 'checkout'))):
            spec: Any = self.jobs2.get(have)
            if not spec or other in self.jobs2:
                continue
            schema: Any = self.specs.get(spec['name']) or {}
            props: Any = schema.get('properties') or {}
            if any((isinstance((props.get(k) or {}).get('enum'), list) for k in schema.get('required') or [])):
                self.jobs2[other] = dict(name=spec['name'])

class Hunting:

    def contenders(self, item: Article) -> list[Offer]:
        cap: Any = self.cap
        out: list = []
        for listing in iter(item.pool.values()):
            if listing.stock is False and self.in_stock_solely:
                continue
            if cap is not None and listing.cost_of is not None and (listing.cost_of * item.amount > cap + 1e-06):
                continue
            if not bool(self.ok_category(item, listing)):
                continue
            out.append(listing)
        out.sort(key=lambda l: (0 if l.sift == 'yes' else 2 if l.sift == 'no' else 1, l.cost_of if l.cost_of is not None else 1e+18))
        return out

    def explore(self, item: Article, steps: int, final: bool=False) -> None:
        stop_at: Any = self.moves_remaining() - steps
        terms: list = list(item.words2)
        term_index: Any = min(item.lookup_done, len(terms))
        want_fits: Any = 2 if not (self.cheapest or self.best) else 3 if self.cheapest else 4
        while self.moves_remaining() > max(stop_at, self.end_keep_back()) and self.time_remaining() > CONCLUDE_SECS + 10 and (not self.spent()):
            self.handle_rows()
            cands: Any = self.contenders(item)
            self.sift(item, cands)
            cands: Any = self.contenders(item)
            unopened: list = []
            for _it61 in cands:
                if not _it61.unfolded:
                    unopened.append(_it61)
            self.judge(item, [l for l in cands if l.unfolded])
            fits: Any = item.matches()
            if fits and (not self.cheapest) and (len(fits) >= want_fits):
                break
            if fits and self.cheapest:
                best: Any = fits[0].cost_of if fits[0].cost_of is not None else 1e+18
                unopened: list = [l for l in unopened if l.cost_of is not None and l.cost_of < best]
                if not unopened and best < 1e+17 and self.jobs2.get('search_fields') and self.jobs2['search_fields'].get('price_max') and (getattr(item, 'below_sweep', None) != best) and (self.moves_remaining() > max(stop_at, self.end_keep_back()) + 2):
                    item.below_sweep = best
                    fields: dict = dict(category=item.category, price_max=max(0.0, best - 0.01))
                    self.search(item, fields=fields, max_pages=3, budget=max(stop_at, self.end_keep_back()))
                    continue
                if not unopened and getattr(item, 'capped', False) and (not getattr(item, 'narrowed', False)) and (self.moves_remaining() > max(stop_at, self.end_keep_back()) + 2):
                    item.narrowed = True
                    values: list = []
                    for _it60 in item.active():
                        if _it60.op in ('is', 'one of') and _it60.value:
                            values.append(_it60.value.split('|')[0].strip())
                    if bool(values):
                        for q in iter((' '.join(values + [item.category]), ' '.join(values))):
                            if self.moves_remaining() <= max(stop_at, self.end_keep_back()) + 1:
                                break
                            self.search(item, query=q, max_pages=2, budget=max(stop_at, self.end_keep_back()))
                        continue
                if not bool(unopened):
                    main: Any = next((q for q in terms if item.cursors.get('search_text:' + q) and (not item.cursors['search_text:' + q].get('done'))), None)
                    if main and self.moves_remaining() > max(stop_at, self.end_keep_back()) + 2:
                        before: Any = self.move
                        self.search(item, query=main, max_pages=1, budget=max(stop_at, self.end_keep_back()) + 1)
                        if self.move > before:
                            continue
                if not unopened and item.lookup_done >= min(len(terms), 2):
                    break
            if bool(unopened):
                batch: Any = unopened[:self.compare_cap if self.bundle_unfold() else 4 if self.compare_thin else 2]
                got: Any = self.unfold_bundle(item, [l.ident for l in batch], floor=max(stop_at, self.end_keep_back()))
                if not bool(got):
                    break
                continue
            needs: list = []
            for _it62 in item.active():
                if _it62.origin == 'shopper' and _it62.value and (_it62.op in ('is', 'one of')):
                    needs.append(_it62.value.split('|')[0].strip())
            stated: list = []
            for _it63 in item.active():
                if _it63.origin != 'shopper' and _it63.value and (_it63.op in ('is', 'one of')):
                    stated.append(_it63.value.split('|')[0].strip())
            if needs and (not getattr(item, 'want_lookup_done', False)):
                item.want_lookup_done = True
                for q in iter(([''.join((format(item.category, ''), ' ', format(stated[0], ''), ' ', format(needs[0], '')))] if stated else []) + [''.join((format(item.category, ''), ' ', format(needs[0], ''))), needs[0]]):
                    if self.moves_remaining() <= max(stop_at, self.end_keep_back()):
                        break
                    self.search(item, query=q, max_pages=3, budget=max(stop_at, self.end_keep_back()))
                continue
            if term_index < len(terms):
                query: Any = terms[term_index]
                term_index += 1
                pages: Any = 6 if len(self.items) == 1 else 3
                new: Any = self.search(item, query=query, max_pages=pages, budget=max(stop_at, self.end_keep_back()))
                if new == 0 and term_index >= len(terms):
                    item.exhausted = True
                continue
            if bool(self.jobs2.get('search_fields')):
                fields: dict = dict(category=item.category)
                if self.cap is not None:
                    fields['price_max'] = self.cap / max(item.amount, 1)
                sig: Any = 'search_fields:' + json.dumps(fields, sort_keys=True)
                if not bool((item.cursors.get(sig) or {}).get('done')):
                    self.search(item, fields=fields, max_pages=2, budget=max(stop_at, self.end_keep_back()))
                    continue
            resumed: bool = False
            for q in iter(terms[:2]):
                sig: Any = 'search_text:' + q
                if not bool((item.cursors.get(sig) or {}).get('done')):
                    if self.search(item, query=q, max_pages=2, budget=max(stop_at, self.end_keep_back())) > 0:
                        resumed: bool = True
                        break
            if bool(resumed):
                continue
            if not getattr(item, 'suggested', False) and self.moves_remaining() > max(stop_at, self.end_keep_back()) + 1:
                item.suggested = True
                extra: Any = self.queries_suggest(item)
                if bool(extra):
                    item.words2 += extra
                    terms += extra
                    continue
            item.exhausted = True
            break

    def fetch_data_points(self, lid: str) -> dict[str, Any] | None:
        tool: Any = self.jobs2.get('facts')
        if not bool(tool):
            return None
        res: Any = self.invoke(tool['name'], {tool['id']: lid})
        if not res['ok'] or not isinstance(res['obs'], dict):
            return None
        m: Any = self.mapping('facts', res['obs'])
        fk: Any = m.get('facts')
        if fk and isinstance(res['obs'].get(fk), dict):
            _rv64: dict = {str(k): v for k, v in res['obs'][fk].items() if v not in (None, '')}
            return _rv64
        _rv65: Any = {str(k): v for k, v in res['obs'].items() if v not in (None, '')} if all((isinstance(v, (str, int, float)) or v is None for v in res['obs'].values())) else {}
        return _rv65

    def labels_resolve(self, item: Article) -> None:
        labels: list = sorted({l.category for l in item.pool.values() if l.category})
        if not labels or item.ok_labels is not None:
            return
        by_norm: dict = {}
        for _it66 in labels:
            by_norm[_norm(_it66)] = _it66
        if len(labels) == 1:
            item.ok_labels = set(by_norm) if _similar(labels[0], item.category) >= 0.3 or _norm(item.category) in _norm(labels[0]) or _norm(labels[0]) in _norm(item.category) else set()
            if not bool(item.ok_labels):
                item.ok_labels = None
            return
        user: str = ''.join(('PRODUCT WANTED: ', format(item.name, ''), ' (request calls the category: ', format(item.category, ''), ')\nFROM THE REQUEST: ', format(self.ask2[:600], ''), '\nCATEGORY LABELS IN THIS STORE: ', format(json.dumps(labels, ensure_ascii=False), '')))
        answer: Any = self.llm.json([FAST_LLM, BACKUP_LLM], LABEL_BRIEF, user, 500)
        chosen: set[str] = set()
        if isinstance(answer, dict) and isinstance(answer.get('labels'), list):
            for l in iter(answer['labels']):
                if isinstance(l, str) and _norm(l) in by_norm:
                    chosen.add(_norm(l))
        if bool(chosen):
            item.ok_labels = chosen
            self.remark2('labels', item=item.name, labels=sorted(chosen))

    def ok_category(self, item: Article, listing: Offer) -> bool:
        if not listing.category or not item.category:
            return True
        label: Any = _norm(listing.category)
        if bool(item.ok_labels):
            _rv67: bool = label in item.ok_labels
            return _rv67
        want: Any = _norm(item.category)
        _rv68: Any = True if label == want or label in want or want in label else _similar(label, want) >= 0.5
        return _rv68

    def queries_suggest(self, item: Article) -> list[str]:
        seen: Any = sorted(item.pool.values(), key=lambda l: (not l.unfolded, l.ident))[:14]
        rows: list = []
        for _it69 in seen:
            rows.append(dict(title=_it69.title[:110], fits=_it69.ident in {f.ident for f in item.matches(include_unsure=True)} if _it69.unfolded else None))
        tried: list = sorted({k.split(':', 1)[1] for k in item.cursors if k.startswith('search_text:')})
        limit: Any = (self._spec('search_text').get('properties') or {}).get((self.jobs2.get('search_text') or {}).get('query') or '', {}).get('maxLength')
        user: Any = ''.join(('PRODUCT WANTED: ', format(item.name, ''), ' (category: ', format(item.category, ''), ')\nITS REQUIREMENTS:\n- ')) + ('\n- '.join(item.req_proses()) or 'none') + ''.join(('\nQUERIES ALREADY USED: ', format(json.dumps(tried, ensure_ascii=False), ''), '\nLISTINGS ENCOUNTERED: ', format(json.dumps(rows, ensure_ascii=False), ''))) + (''.join(('\nLONGEST ALLOWED QUERY: ', format(limit, ''), ' characters')) if limit else '')
        answer: Any = self.llm.json([FAST_LLM, BACKUP_LLM], PHRASE_BRIEF, user, 700)
        out: list[str] = []
        if isinstance(answer, dict):
            for q in iter(answer.get('queries') or []):
                q: Any = str(q).strip()
                if q and len(q) <= 60 and (_norm(q) not in {_norm(t) for t in tried + item.words2 + out}):
                    out.append(q)
        self.remark2('suggested', item=item.name, queries=out[:5])
        _rv70: Any = out[:5]
        return _rv70

    def read_data_points(self, item: Article, lid: str) -> Offer | None:
        known: Any = item.pool.get(lid)
        facts: Any = self.fetch_data_points(lid)
        if facts is None or (known is None and (not facts)):
            return None
        listing: Any = Offer(lid)
        listing.data_points = facts
        _rv71: Any = self.merge(item, listing, True)
        return _rv71

    def search(self, item: Article, query: str | None=None, fields: dict[str, Any] | None=None, max_pages: int=1, budget: int | None=None) -> int:
        role: Any = 'search_text' if query is not None else 'search_fields'
        spec: Any = self.jobs2.get(role)
        if not bool(spec):
            return 0
        name: Any = spec['name']
        new: int = 0
        page_mode: Any = self._leaf_mode(role)
        sig: Any = role + ':' + (query if query is not None else json.dumps(fields or {}, sort_keys=True))
        cursor: Any = item.cursors.get(sig)
        if cursor and cursor.get('done'):
            return 0
        token: Any = cursor.get('token') if cursor else None
        first: Any = cursor.get('page', 0) + 1 if cursor else 1
        if first > 1 and page_mode is None:
            return 0
        done: bool = False
        page: Any = first - 1
        for page in range(first, first + max_pages):
            if self.moves_remaining() <= (budget if budget is not None else 0) or self.time_remaining() < CONCLUDE_SECS:
                page -= 1
                break
            args: dict[str, Any] = {}
            if spec.get('mode') and isinstance(spec['mode'], dict) and spec['mode'].get('arg'):
                args[spec['mode']['arg']] = spec['mode'].get('value')
            if query is not None:
                if not bool(spec.get('query')):
                    return 0
                args[spec['query']] = self._match_phrase(item, role, spec['query'], query)
            else:
                for key, value in iter((fields or {}).items()):
                    arg: Any = spec.get(key)
                    if arg and value not in (None, ''):
                        value: Any = self._enum_val(role, arg, value)
                        if value is None:
                            item.cursors[sig] = dict(page=0, token=None, done=True)
                            return new
                        args[arg] = self._trim(role, arg, value) if isinstance(value, str) else value
            if page > 1:
                if page_mode == 'index':
                    args[spec['page']] = page
                elif page_mode == 'offset':
                    args[spec['page']] = (page - 1) * max(self.leaf_extent, 1)
                elif page_mode == 'token' and token:
                    args[spec['page']] = token
                else:
                    done: bool = True
                    page -= 1
                    break
                top: Any = ((self._spec(role).get('properties') or {}).get(spec['page']) or {}).get('maximum')
                if isinstance(args[spec['page']], int) and isinstance(top, (int, float)) and (args[spec['page']] > top):
                    item.capped = True
                    done: bool = True
                    page -= 1
                    break
            res: Any = self.invoke(name, args)
            if not bool(res['ok']):
                self.remark2('search_error', error=res['error'])
                done: bool = True
                break
            obs: Any = res['obs']
            rows: Any = _find_seq_of_mappings(obs) or []
            token: Any = None
            total: Any = None
            if isinstance(obs, dict):
                for k, v in iter(obs.items()):
                    if isinstance(v, str) and 0 < len(v) <= 20 and (' ' not in v) and (not isinstance(v, bool)):
                        token: Any = v
                numbers: list = []
                for _it72 in obs.values():
                    if isinstance(_it72, int) and (not isinstance(_it72, bool)):
                        numbers.append(_it72)
                total: Any = max(numbers) if numbers else None
            if bool(rows):
                self.leaf_extent = max(self.leaf_extent, len(rows))
            for rec in iter(rows):
                listing: Any = self.offer_from(rec, 'row')
                if listing is None:
                    continue
                if listing.ident not in item.pool:
                    new += 1
                self.merge(item, listing, opened=False)
            item.lookup_done += 1
            if item.ok_labels is None and item.index != 99:
                self.labels_resolve(item)
            if not rows or (total is not None and page * max(len(rows), 1) >= total) or len(rows) < self.leaf_extent:
                done: bool = True
                break
            if page_mode == 'token' and (not token):
                if total is not None and total > page * len(rows):
                    item.capped = True
                done: bool = True
                break
        item.cursors[sig] = dict(page=max(page, 0), token=token, done=done or page_mode is None)
        return new

    def sift(self, item: Article, listings: list[Offer]) -> None:
        todo: list = []
        for _it73 in listings:
            if not _it73.unfolded and _it73.sift == 'maybe' and (not getattr(_it73, 'screened', False)):
                todo.append(_it73)
        if len(todo) < 8 or not item.req_proses():
            return
        rows: list = []
        batch: Any = todo[:30]
        for n, l in enumerate(batch, 1):
            rows.append(dict(n=n, title=l.title[:140], text=(l.text or '')[:RECORD_ROW_CHARS], category=l.category, price=l.cost_of))
        user: Any = ''.join(('PRODUCT WANTED: ', format(item.category, ''), '\nITS REQUIREMENTS:\n- ')) + '\n- '.join(item.req_proses()) + '\n\nROWS:\n' + json.dumps(rows, ensure_ascii=False)
        answer: Any = self.llm.json([FAST_LLM, BACKUP_LLM], SIFT_BRIEF, user, 1200)
        for l in iter(batch):
            l.screened = True
        if isinstance(answer, dict):
            for verdict in iter(('no', 'yes')):
                for n in iter(answer.get(verdict) or []):
                    if isinstance(n, (int, str)) and str(n).strip().isdigit() and (1 <= int(n) <= len(batch)):
                        batch[int(n) - 1].sift = verdict

    def unfold_bundle(self, item: Article | None, ids: list[str], floor: int=0) -> list[Offer]:
        ids: list = [i for i in ids if i]
        opened: list[Offer] = []
        if not bool(ids):
            return opened
        compare: Any = self.jobs2.get('compare') if not getattr(self, 'off_compare', False) else None
        facts_tool: Any = self.jobs2.get('facts')
        open_tool: Any = self.jobs2.get('open')
        if compare and (not getattr(self, 'compare_vetted', False)):
            self.compare_vetted = True
            most: Any = ((self._spec('compare').get('properties') or {}).get(compare.get('ids') or '') or {}).get('maxItems')
            if isinstance(most, int) and 2 <= most < self.compare_cap:
                self.compare_cap = most
        pos: int = 0
        while pos < len(ids) and self.moves_remaining() > 0 and (self.time_remaining() > CONCLUDE_SECS):
            if self.compare_thin and facts_tool and (item is not None):
                if floor and self.moves_remaining() - 1 < floor:
                    break
                got: Any = self.read_data_points(item, ids[pos])
                pos += 1
                if got is not None:
                    opened.append(got)
                continue
            if compare and len(ids) - pos >= 2:
                if floor and self.moves_remaining() - 1 < floor:
                    break
                chunk: Any = ids[pos:pos + self.compare_cap]
                res: Any = self.invoke(compare['name'], {compare['ids']: chunk})
                if res['ok'] and _find_seq_of_mappings(res['obs']):
                    rows: list = []
                    for _it74 in (self.offer_from(rec, 'detail') for rec in _find_seq_of_mappings(res['obs']) or []):
                        if _it74 is not None:
                            rows.append(_it74)
                    thin: Any = bool(facts_tool) and bool(rows) and (not any((l.data_points for l in rows)))
                    if thin and (not self.compare_thin):
                        self.compare_thin = True
                        self.remember()
                        self.remark2('thin_compare', tool=compare['name'], facts_tool=facts_tool.get('name'))
                    for listing in iter(rows):
                        listing.leaf_read = True
                        self.unfolded_leaves.add(listing.ident)
                        if thin and item is not None:
                            self.merge(item, listing, False)
                            continue
                        listing.unfolded = True
                        opened.append(self.merge(item, listing, True) if item else listing)
                    if thin and item is not None:
                        for listing in iter(rows):
                            if floor and self.moves_remaining() - 1 < floor or self.time_remaining() < CONCLUDE_SECS:
                                break
                            got: Any = self.read_data_points(item, listing.ident)
                            if got is not None:
                                opened.append(got)
                    pos += len(chunk)
                    continue
                if res['error'] or (res['ok'] and (not _find_seq_of_mappings(res['obs']))):
                    if self.compare_cap > 2 and len(chunk) > 2:
                        self.compare_cap = 4 if self.compare_cap > 4 else 2
                        self.remember()
                        continue
                    self.off_compare = True
                    compare: Any = None
                    continue
            if not bool(open_tool):
                break
            cost: Any = 2 if facts_tool else 1
            if floor and self.moves_remaining() - cost < floor:
                break
            lid: Any = ids[pos]
            pos += 1
            res: Any = self.invoke(open_tool['name'], {open_tool['id']: lid})
            if not res['ok'] or not isinstance(res['obs'], dict):
                continue
            rec: Any = res['obs']
            listing: Any = self.offer_from(rec, 'detail')
            if listing is None:
                listing: Any = Offer(lid)
                listing.raw = rec
            listing.leaf_read = True
            if not listing.data_points and facts_tool and (self.moves_remaining() > 0):
                listing.data_points = self.fetch_data_points(lid) or {}
            if listing.ident != lid:
                listing.ident = lid
            listing.unfolded = True
            opened.append(self.merge(item, listing, True) if item else listing)
            self.unfolded_leaves.add(lid)
        return opened

    def up_warm(self, item: Article, allowance: int | None=None) -> None:
        if self.moves_remaining() <= self.end_keep_back() + 2 or 'message' not in self.jobs2:
            return
        stop_at: Any = self.moves_remaining() - (allowance if allowance is not None else self.moves_remaining())
        floor: Any = max(stop_at, self.end_keep_back() + 2)
        target: Any = 12 if self.bundle_unfold() else 5

        def spare() -> list[Offer]:
            out: list = []
            for _it75 in item.pool.values():
                if not _it75.unfolded and self.ok_category(item, _it75) and (_it75.stock is not False):
                    out.append(_it75)
            out.sort(key=lambda l: (l.sift == 'no', l.sift != 'yes', l.cost_of if l.cost_of is not None else 1e+18))
            return out
        while self.moves_remaining() > floor and self.time_remaining() > CONCLUDE_SECS + 20 and (not self.spent()):
            opened: list = []
            for _it76 in item.pool.values():
                if _it76.unfolded and self.ok_category(item, _it76):
                    opened.append(_it76)
            want: Any = target - len(opened)
            if want <= 0:
                return
            todo: Any = spare()
            if len(todo) < 2 and item.lookup_done < len(item.words2) + 1:
                before: int = len(item.pool)
                self.search(item, query=item.category, max_pages=1, budget=floor)
                todo: Any = spare()
                if len(item.pool) == before and (not todo):
                    return
            if not bool(todo):
                return
            chunk: Any = min(want, self.compare_cap if self.bundle_unfold() else 4 if self.compare_thin else 1)
            got: Any = self.unfold_bundle(item, [l.ident for l in todo[:chunk]], floor=floor)
            if not bool(got):
                return
            self.judge(item, [l for l in self.contenders(item) if l.unfolded])

class Weighing:

    def all_judge(self) -> None:
        _parallel([lambda it=it: self.judge(it, [l for l in self.contenders(it) if l.unfolded]) for it in self.items])

    def best_rank(self, item: Article, cands: list[Offer]) -> list[Offer]:
        measure: Any = getattr(item, 'best', None) or self.best
        if not measure or len(cands) < 2:
            return cands
        cached: Any = getattr(self, 'best_purchase', {}).get(item.index)
        ids: list = []
        for _it78 in cands:
            ids.append(_it78.ident)
        if cached is None or cached[0] != ids:
            rows: list = []
            for _it77 in cands[:12]:
                rows.append(dict(id=_it77.ident, title=_it77.title[:120], price=_it77.cost_of, properties=_it77.data_points, description=str(_it77.text or '')[:300]))
            user: Any = ''.join(('MEASURE TO OPTIMISE: ', format(measure, ''), '\nPRODUCT: ', format(item.name, ''), '\nCANDIDATE LISTINGS: ')) + _compact(rows, 9000)
            answer: Any = self.llm.json([SCHEME_LLM, BACKUP_LLM], RANK_BRIEF, user, 600)
            order: Any = [str(x) for x in answer.get('order') or []] if isinstance(answer, dict) else []
            self.remark2('best_rank', measure=measure, order=order[:12])
            cached: tuple = (ids, order)
            if bool(order):
                self.best_purchase = dict(getattr(self, 'best_purchase', {}))
                self.best_purchase[item.index] = cached
        order: Any = cached[1]
        pos: dict = {}
        for _it79, _it80 in enumerate(order):
            pos[_it80] = _it79
        _rv81: list = sorted(cands, key=lambda l: pos.get(l.ident, len(pos) + ids.index(l.ident)))
        return _rv81

    def bind_match(self, item: Article, host_item: Article, host: Offer) -> None:
        page: str = ' '.join((x for x in [host.title[:120], (host.text or '')[:300], (host.remarks or '')[:150]] if x))
        item.requirements = [r for r in item.requirements if r.origin != 'fit']
        item.requirements.append(Requirement('compatibility', 'is', ''.join(('matches the ', format(host_item.name, ''), ' picked, whose page says: ', format(page, ''))), origin='fit'))
        code: Any = re.compile('\\b[A-Z]{1,6}-?\\d{3,}[A-Z0-9]*\\b')
        titled: set = {c for it in self.items for l in it.pool.values() for c in code.findall(l.title or '')}
        prefixes: set = set()
        for _it84 in item.pool.values():
            for _it83 in code.findall(_it84.title or ''):
                prefixes.add(re.match('[A-Z]+', _it83).group(0))
        wanted: list[str] = []
        for l in iter(self.contenders(item)):
            if not l.unfolded or (item.rulings.get(l.ident) or {}).get('failed'):
                continue
            for c in iter(code.findall(' '.join([l.text or '', l.remarks or '']))):
                if c not in titled and c not in wanted and (re.match('[A-Z]+', c).group(0) in prefixes):
                    wanted.append(c)
        host_text: str = ' '.join([host.text or '', host.remarks or '', json.dumps(host.data_points, ensure_ascii=False)])
        heads: set = {h for it in self.items if it is not host_item for l in it.pool.values() for h in _SERIES.findall(l.title or '')}
        series: list = []
        for c in iter(re.findall('\\b[A-Z]{1,5}\\d{1,3}\\b', host_text)):
            family: Any = re.match('[A-Z]+', c).group(0)
            if c not in series and c not in _SERIES.findall(host.title or '') and (c in heads or any((re.match('[A-Z]+', h).group(0) == family for h in heads))):
                series.append(c)
        own: list[str] = []
        for c in iter(re.findall('\\b[A-Z]{2,6}-?\\d{2,6}[A-Z0-9]*\\b', ' '.join([host.title or '', host_text]))):
            if c not in own and c not in series and c not in titled - set(code.findall(host.title or '')):
                own.append(c)
        for c in iter(series[:2] + own[:2] + [w for w in wanted if w not in series and w not in own][:1]):
            if self.moves_remaining() <= self.end_keep_back() + 2 or self.time_remaining() < CONCLUDE_SECS + 10:
                break
            query: Any = ''.join((format(c, ''), ' ', format(item.category, ''))) if c in series or c in own else c
            self.search(item, query=query, max_pages=1, budget=self.end_keep_back())
        item.match_series = series[:2]
        if series and self.moves_remaining() > self.end_keep_back() + 2:
            mine: list = []
            for _it82 in self.contenders(item):
                if not _it82.unfolded and self.series_in(_it82, series):
                    mine.append(_it82)
            if bool(mine):
                self.unfold_bundle(item, [l.ident for l in mine[:self.compare_cap if self.bundle_unfold() else 3]], floor=self.end_keep_back() + 1)
        self.remark2('fit_bound', item=item.name, host=host.ident, series=series[:2], host_codes=own[:2], looked_up=wanted[:1])
        self.judge(item, [l for l in self.contenders(item) if l.unfolded])
        if bool(series):
            fit_req: Any = item.requirements[-1].text()
            for l in iter(self.contenders(item)):
                v: Any = item.rulings.get(l.ident)
                if v and l.unfolded and self.series_in(l, series):
                    for key in iter(('unsure', 'weak', 'failed')):
                        if key in v:
                            v[key].discard(fit_req)

    def conditions_apply(self, item: Article, chosen: dict[int, Offer]) -> None:
        mine: list = []
        for _it85 in self.conditions:
            if any((d is item for _, d, _ in _it85['branches'])):
                mine.append(_it85)
        if not bool(mine):
            return
        wanted: list = []
        for cond in iter(mine):
            pick: Any = chosen.get(cond['src'].index)
            holds: Any = self.holds_condition(cond, pick) if pick is not None else None
            if holds is None:
                continue
            for when, dst, b in iter(cond['branches']):
                if dst is item and when == holds:
                    wanted.append(Requirement(b['property'], b.get('op') or 'is', b['value'], origin='condition'))
        old: list = sorted((r.text() for r in item.requirements if r.origin == 'condition'))
        if old == sorted((r.text() for r in wanted)):
            return
        item.requirements = [r for r in item.requirements if r.origin != 'condition'] + wanted
        self.remark2('condition_applied', item=item.name, reqs=[r.text() for r in wanted])
        self.judge(item, [l for l in self.contenders(item) if l.unfolded])
        if not self.buyable(item) and self.moves_remaining() - self.end_keep_back() >= 3 and (self.time_remaining() > CONCLUDE_SECS + 20):
            item.exhausted = False
            self.explore(item, self.moves_remaining() - self.end_keep_back(), final=True)

    def conditions_read(self, raw: Any) -> list[dict[str, Any]]:
        out: list = []
        for c in iter(raw if isinstance(raw, list) else []):
            cond: Any = c.get('if') if isinstance(c, dict) else None
            if not isinstance(cond, dict) or not cond.get('property'):
                continue
            src: Any = self.article_named(cond.get('item'))
            values: Any = cond.get('values')
            values: Any = [str(v) for v in values if v] if isinstance(values, list) else [str(values)] if values else []
            branches: list = []
            for key, when in iter((('then', True), ('otherwise', False))):
                b: Any = c.get(key)
                if isinstance(b, dict) and b.get('property') and b.get('value'):
                    dst: Any = self.article_named(b.get('item'))
                    if dst is not None and dst is not src:
                        branches.append((when, dst, b))
            if src is None or not values or (not branches):
                continue
            out.append(dict(src=src, prop=str(cond['property']), values=values, branches=branches))
            for _, dst, b in iter(branches):
                noun: Any = dst.noun
                for v in iter(str(b['value']).split('|')):
                    v: Any = v.strip()
                    if v and len(v) <= 40 and (not any((_norm(v) in _norm(t) for t in dst.words2))):
                        dst.words2.append(''.join((format(v, ''), ' ', format(noun, '')))[:60])
        if bool(out):
            self.remark2('conditions', rules=[(c['src'].name, c['prop'], c['values'], [(w, d.name, b.get('property'), b.get('value')) for w, d, b in c['branches']]) for c in out])
        return out

    def confirm(self, item: Article, listing: Offer) -> bool:
        key: tuple = (item.index, listing.ident, tuple(item.req_proses()), json.dumps(listing.data_points, sort_keys=True, default=str))
        cache: Any = self.__dict__.setdefault('verified', {})
        if key not in cache:
            cache[key] = self._confirm(item, listing)
        _rv86: Any = cache[key]
        return _rv86

    def _confirm(self, item: Article, listing: Offer, strict: bool=False) -> bool:
        reqs: Any = item.req_proses()
        user: Any = ''.join(('PRODUCT WANTED: ', format(item.category, ''), ' (quantity ', format(item.amount, ''), ')\nBUDGET FOR ALL LINES COMBINED: ', format(self.cap if self.cap is not None else 'none stated', ''), '\nITS REQUIREMENTS:\n- ')) + ('\n- '.join(reqs) if reqs else '(nothing besides the product kind, the budget and being in stock)') + '\n\nLISTING:\n' + json.dumps(listing.entry2(3500), ensure_ascii=False)
        table: Any = self.slot_grid(item)
        if bool(table):
            user = user + "\n\nSTORE FIELDS (taken from this store's listings read until now):\n" + self.slot_rows(table)
        answer: Any = self.llm.json([CONFIRM_LLM, FAST_LLM], CONFIRM_BRIEF, user, 800)
        if not isinstance(answer, dict):
            _rv88: bool = not strict
            return _rv88
        if strict and answer.get('fits') is not True:
            return False
        problems: list = []
        for _it87 in answer.get('problems') or []:
            problems.append(str(_it87))
        self.remark2('verify', id=listing.ident, fits=answer.get('fits'), problems=problems[:4])
        if answer.get('fits') is False and problems:
            v: Any = item.rulings.setdefault(listing.ident, dict(failed=set(), unsure=set()))
            for p in iter(problems):
                best: Any = max(reqs, key=lambda r: _similar(r, p)) if reqs else None
                v['failed'].add(best if best and _similar(best, p) > 0.2 else p)
            return False
        return True

    def holds_condition(self, cond: dict[str, Any], listing: Offer) -> bool | None:
        cache: Any = self.__dict__.setdefault('held_cache', {})
        key: tuple = (id(cond), listing.ident, json.dumps(listing.data_points, sort_keys=True, default=str))
        if key in cache:
            _rv92: Any = cache[key]
            return _rv92
        own: list = []
        for _it90, _it91 in listing.data_points.items():
            if _near(str(_it90).replace('_', ' '), cond['prop']) or _similar(str(_it90).replace('_', ' '), cond['prop']) >= 0.5:
                own.append(str(_it91))
        holds: bool | None = None
        if bool(own):
            text: Any = _tokens2(' '.join(own))
            wants: list = []
            for _it89 in cond['values']:
                if _tokens2(_it89):
                    wants.append(_tokens2(_it89))
            if any((w <= text for w in wants)):
                holds: bool = True
            elif not any((w & text for w in wants)):
                holds: bool = False
        if holds is None:
            user: Any = ''.join(('PROPERTY TO TEST: ', format(cond['prop'], ''), '\nACCEPTED VALUES: ', format(json.dumps(cond['values'], ensure_ascii=False), ''), '\n\nTHE LISTING:\n')) + json.dumps(listing.entry2(2500), ensure_ascii=False)
            answer: Any = self.llm.json([FAST_LLM, BACKUP_LLM], COND_BRIEF, user, 300)
            if isinstance(answer, dict) and isinstance(answer.get('holds'), bool):
                holds: Any = answer['holds']
        cache[key] = holds
        self.remark2('condition', src=cond['src'].name, listing=listing.ident, prop=cond['prop'], holds=holds)
        return holds

    def judge(self, item: Article, listings: list[Offer], force: bool=False) -> None:
        self.label_restatements_drop(item)
        reqs: Any = item.req_proses()
        if not bool(reqs):
            for l in iter(listings):
                if l.unfolded and l.ident not in item.rulings:
                    item.rulings[l.ident] = dict(failed=set(), unsure=set())
                    item.for_judged[l.ident] = set()
            return
        todo: list = []
        for _it96 in listings:
            if _it96.unfolded and (not item.for_judged.get(_it96.ident, set()) >= set(reqs)):
                todo.append(_it96)
        if not todo or (self.time_remaining() < CONCLUDE_SECS and (not force)) or self.time_remaining() < 10:
            return
        table: Any = self.slot_grid(item)
        fields: Any = "\n\nSTORE FIELDS (taken from this store's listings read until now):\n" + self.slot_rows(table) if table else ''
        batches, prompts = ([], [])
        for start in range(0, len(todo), 4):
            batch: Any = todo[start:start + 4]
            records: list = []
            for l in iter(batch):
                rec: Any = l.entry2()
                related: Any = self.related_leaves(item, l)
                if bool(related):
                    rec['related_pages'] = related
                records.append(rec)
            batches.append(batch)
            prompts.append(''.join(('PRODUCT WANTED: ', format(item.category, ''), '\nREQUIREMENTS TO CHECK ONE BY ONE:\n')) + '\n'.join((''.join(('R', format(k, ''), ': ', format(r, ''))) for k, r in enumerate(reqs, 1))) + fields + '\n\nLISTINGS:\n' + json.dumps(records, ensure_ascii=False))
        self.urgent = bool(force) and self.time_remaining() < CONCLUDE_SECS
        try:
            answers: Any = _parallel([lambda u=u: self.llm.json([FAST_LLM, BACKUP_LLM], JUDGE_BRIEF, u, 3000) for u in prompts])
        finally:
            self.urgent = False
        for batch, answer in zip(batches, answers):
            got: dict = {}
            if isinstance(answer, dict) and isinstance(answer.get('verdicts'), list):
                for v in iter(answer['verdicts']):
                    if isinstance(v, dict) and v.get('id') in item.pool:
                        got[v['id']] = v
            for l in iter(batch):
                v: Any = got.get(l.ident)
                if v is None:
                    continue
                failed: set = set()
                for _it93 in v.get('failed') or []:
                    if _it93:
                        failed.add(str(_it93))
                unsure: set = set()
                for _it94 in v.get('unsure') or []:
                    if _it94:
                        unsure.add(str(_it94))
                loose: set = set()
                for _it95 in v.get('loose') or []:
                    if _it95:
                        loose.add(str(_it95))
                weak: Any = {str(x) for x in v.get('weak') or [] if x} | loose

                def canon(names: set[str]) -> set[str]:
                    out: set = set()
                    for n in iter(names):
                        m: Any = re.fullmatch('\\s*R?(\\d+)\\s*', n)
                        if m and 1 <= int(m.group(1)) <= len(reqs):
                            out.add(reqs[int(m.group(1)) - 1])
                            continue
                        best: Any = max(reqs, key=lambda r: _similar(r, n)) if reqs else None
                        if best and _similar(best, n) > 0.2:
                            out.add(best)
                        else:
                            out.add(n)
                    return out
                item.rulings[l.ident] = dict(failed=canon(failed) | self.slot_fails(item, l, table), unsure=canon(unsure), weak=canon(weak), loose=canon(loose))
                item.for_judged[l.ident] = set(reqs)
                v: Any = item.rulings[l.ident]
                for r in iter(item.active()):
                    t: Any = r.text()
                    if t not in v['failed'] and t not in v['unsure'] and (t not in v['loose']) and self.unstated(item, l, r):
                        v['unsure'].add(t)
                    if bool(_proven(r, l)):
                        v['failed'].discard(t)
                        v['unsure'].discard(t)
                    elif t not in v['failed'] and _doubt(r, l):
                        v['unsure'].add(t)

    def label_restatements_drop(self, item: Article) -> None:
        if not bool(item.ok_labels):
            return
        label_words: Any = set().union(*(_pieces_of_text(l) for l in item.ok_labels))
        read: list = []
        for _it97 in item.pool.values():
            if _it97.unfolded and _it97.data_points and self.ok_category(item, _it97):
                read.append(_it97)
        if len(read) < 3:
            return
        keep: list = []
        for r in iter(item.requirements):
            words: Any = _pieces_of_text(r.value)
            if r.origin == 'request' and r.op == 'is' and words and (words <= label_words) and (not any((_near(k, r.prop) or _similar(str(k).replace('_', ' '), r.prop) >= 0.5 for l in read for k in l.data_points))):
                self.remark2('label_restatement', item=item.name, req=r.text())
                continue
            keep.append(r)
        item.requirements = keep

    def match_pairs(self) -> list[tuple[Article, Article]]:
        out: list = []
        for rule in iter(self.match_rules):
            acc, host = (str(rule.get('accessory') or ''), str(rule.get('host') or ''))
            if not acc or not host or len(self.items) < 2:
                continue
            a: Any = max(self.items, key=lambda it: _similar(it.name, acc))
            h: Any = max(self.items, key=lambda it: _similar(it.name, host))
            if a is not h and _similar(a.name, acc) >= 0.2 and (_similar(h.name, host) >= 0.2) and ((a, h) not in out):
                out.append((a, h))
        return out

    def related_leaves(self, item: Article, listing: Offer, limit: int=2) -> list[dict[str, Any]]:
        code: Any = re.compile('\\b[A-Z]{1,6}-?\\d{3,}[A-Z0-9]*\\b')
        own_text: str = ' '.join([listing.text or '', listing.remarks or ''])
        own_codes: set = set(code.findall(listing.title or ''))
        mentioned: Any = set(code.findall(own_text)) - own_codes
        out: list = []
        pools: Any = [item.pool] + [it.pool for it in self.items if it is not item]
        seen: set[str] = {listing.ident}
        for pool in iter(pools):
            for other in iter(pool.values()):
                if other.ident in seen or len(out) >= limit:
                    continue
                titled: set = set(code.findall(other.title or ''))
                names_it: bool = bool(own_codes & set(code.findall(' '.join([other.text or '', other.remarks or '']))))
                if titled & mentioned or names_it:
                    seen.add(other.ident)
                    out.append(dict(id=other.ident, title=other.title[:120], text=' '.join([other.text or '', other.remarks or ''])[:400]))
        return out

    @staticmethod
    def series_in(listing: Offer, series: list[str]) -> bool:
        _rv98: bool = any((h in series for h in _SERIES.findall(listing.title or '')))
        return _rv98

    def slot_fails(self, item: Article, listing: Offer, table: list[dict[str, Any]] | None=None) -> set[str]:
        out: set = set()
        for row in iter(self.slot_grid(item) if table is None else table):
            own: list = []
            for _it99 in listing.data_points:
                if _near(_it99, row['prop']):
                    own.append(_it99)
            if not own or not row['writers'] - {listing.ident}:
                continue
            elems: list = []
            for _it101 in own:
                for _it100 in re.split('[,;/|+、]', str(listing.data_points[_it101])):
                    if _pieces_of_text(_it100):
                        elems.append(_spelled(_pieces_of_text(_it100)))
            if elems and (not any((_within(w, e) for w in row['wants'] for e in elems))) and (not _inside_range(row['req'], ' '.join((str(listing.data_points[k]) for k in own)))):
                out.add(row['req'])
        return out

    def slot_grid(self, item: Article) -> list[dict[str, Any]]:
        read: list = []
        for _it104 in item.pool.values():
            if _it104.unfolded and _it104.data_points and self.ok_category(item, _it104):
                read.append(_it104)
        if len(read) < 3:
            _rv105: list = []
            return _rv105
        out: list = []
        for r in iter(item.active()):
            if r.op not in ('is', 'one of') or not r.value or re.search('\\d', r.value):
                continue
            wants: list = []
            for _it103 in r.value.split('|'):
                if _pieces_of_text(_it103):
                    wants.append(frozenset(_pieces_of_text(_it103)))
            if not bool(wants):
                continue
            keys: set[str] = set()
            values: dict[str, int] = {}
            carriers, writers = (0, set())
            for l in iter(read):
                hit: list = []
                for _it102 in l.data_points:
                    if _near(_it102, r.prop):
                        hit.append(_it102)
                if not bool(hit):
                    continue
                carriers += 1
                keys.update(hit)
                for k in iter(hit):
                    for e in iter(re.split('[,;/|]', str(l.data_points[k]))):
                        e: str = ' '.join(e.lower().split())
                        if bool(e):
                            values[e] = values.get(e, 0) + 1
                            if any((w <= frozenset(_pieces_of_text(e)) for w in wants)):
                                writers.add(l.ident)
            if bool(carriers):
                out.append(dict(req=r.text(), prop=r.prop, keys=sorted(keys), carriers=carriers, read=len(read), values=values, wants=wants, writers=writers))
        return out

    def slot_rows(self, table: list[dict[str, Any]]) -> str:
        lines: list = []
        for row in iter(table):
            seen: str = ' | '.join((''.join((format(v, ''), ' (', format(n, ''), ')')) for v, n in sorted(row['values'].items(), key=lambda x: -x[1])[:8]))
            lines.append(''.join(('- ', format(row['req'], ''), ': key ', format(', '.join((repr(k) for k in row['keys'][:3])), ''), ' appears on ', format(row['carriers'], ''), ' of ', format(row['read'], ''), ' listings read; values found: ', format(seen, ''), '; ', format(len(row['writers']), ''), ' listing(s) carry the required value itself')))
        _rv106: str = '\n'.join(lines)
        return _rv106

    @staticmethod
    def unstated(item: Article, listing: Offer, req: Requirement) -> bool:
        if req.op not in ('is', 'one of') or req.origin == 'fit' or (not req.value):
            return False
        common: Any = _tokens2(item.name) | _tokens2(item.category)
        alts: list = []
        for _it107 in ({x for x in _tokens2(v) - common if not (x.isalpha() and len(x) <= 3)} for v in req.value.split('|')):
            if _it107:
                alts.append(_it107)
        if not bool(alts):
            return False
        text: Any = _tokens2(' '.join([listing.title or '', listing.text or '', listing.remarks or '', listing.category or '', ' '.join((''.join((format(k, ''), ' ', format(v, ''))) for k, v in listing.data_points.items()))]))
        _rv108: bool = not any((a & text for a in alts))
        return _rv108

class Dialogue:

    def append_move(self, step: dict[str, Any]) -> None:
        if not step.get('goal') or len(self.routine) >= 8:
            return
        step: dict = dict(step)
        step['moment'] = str(step.get('moment') or 'before_finish').lower()
        step['kind'] = 'message' if str(step.get('kind') or '').lower() == 'message' else 'tool'
        if any((_similar(step['goal'], st.get('goal')) >= 0.6 for st in self.routine)):
            return
        self.routine.append(step)
        self.confirm_initial = self.confirm_initial or (step['kind'] == 'message' and step['moment'] in ('before_order', 'before_finish'))
        self.remark2('step_added', step=step)

    def apply_response(self, item: Article, prop: str, answer: dict[str, Any] | None) -> None:
        if not bool(answer):
            return
        for step in iter(answer.get('steps') or []):
            if isinstance(step, dict):
                self.append_move(step)
        kind: Any = str(answer.get('kind') or '').lower()
        budget: Any = _number(answer.get('budget'))
        if budget is not None and budget > 0:
            self.cap = budget
            self.remark2('budget', cap=budget)
        if kind == 'need' and answer.get('value') or (kind == 'need' and answer.get('reference')):
            p: str = str(answer.get('property') or prop)
            owner: Any = re.match('(.+?) (?:of|for) (?:the |your |my )?(.+)$', p)
            if owner and len(self.items) > 1:
                fit: Any = lambda it: max(_similar(owner.group(2), it.name), _similar(owner.group(2), it.category))
                other: Any = max(self.items, key=fit)
                if fit(other) >= 0.6:
                    if other is not item:
                        self.remark2('need_rerouted', said=p, to=other.name)
                    item, p = (other, owner.group(1).strip())
            req: Any = Requirement(p, answer.get('op') or 'is', answer.get('value') or '', answer.get('reference'), origin='shopper')
            if bool(req.reference):
                req.reference['property'] = req.prop
            if req.value and any((r.origin == 'shopper' and _norm(r.value) == _norm(req.value) and (r.op == req.op) for r in item.requirements)):
                return
            item.requirements = [r for r in item.requirements if not (r.key() == req.key() and r.origin == 'shopper')]
            item.requirements.append(req)
            if req.reference and req.op in OPS_REF:
                self.reference_resolve(item, req)
            elif req.op in ('is', 'one of', 'contains') and req.value:
                for v in str(req.value).split(' | ')[:2]:
                    words: str = ''.join((v.strip(), ' ', format(item.noun, '')))[:60]
                    if len(v) <= 40 and not any((_norm(words) == _norm(w) for w in item.words2)):
                        item.words2.insert(min(item.lookup_done, len(item.words2)), words)
        elif kind == 'free':
            item.freed.append(_norm(str(answer.get('property') or prop)))

    def apply_response_kept(self, item: Article, prop: str, answer: dict[str, Any] | None) -> None:
        left: Any = self.note_allowance - self.notes_dispatched if self.note_allowance is not None else 2
        self.floor_ref = max(0, left) + self.end_keep_back()
        try:
            self.apply_response(item, prop, answer)
        finally:
            self.floor_ref = 0

    def can_note(self) -> bool:
        if 'message' not in self.jobs2:
            return False
        _rv109: Any = False if self.note_allowance is not None and self.notes_dispatched >= self.note_allowance else self.moves_remaining() > 0
        return _rv109

    def can_query(self) -> bool:
        _rv110: Any = False if not bool(self.can_note()) else self.note_allowance is None or self.notes_dispatched < self.note_allowance - self.parked_notes()
        return _rv110

    def confirm_trolley(self) -> bool:
        if not self.can_note() or self.moves_remaining() < 2:
            return False
        spec: Any = self.jobs2['message']
        parts, total = ([], 0.0)
        for lid, qty in iter(self.trolley_sig()):
            item: Any = self.basket_article.get(lid)
            listing: Any = item.pool.get(lid) if item else None
            title: Any = re.sub('\\s+', ' ', (listing.title if listing else lid) or lid).strip()[:70]
            price: Any = listing.cost_of if listing and listing.cost_of is not None else None
            if price is not None:
                total += price * qty
            parts.append(''.join((format(qty, ''), ' x ', format(title, ''))) + (''.join((' (', format(price, 'g'), ' each)')) if price is not None else ''))
        text: Any = 'Here is the basket I am about to order: ' + '; '.join(parts) + (''.join(('. Total ', format(total, 'g'), '.')) if total else '.') + ' Should I go ahead and check it out? A yes from you confirms it.'
        text: Any = self._trim('message', spec['text'], text)
        res: Any = self.invoke(spec['name'], {spec['text']: text})
        self.notes_dispatched += 1
        self.sig_confirmed = self.trolley_sig()
        self.remark2('confirm_asked', basket=self.sig_confirmed, ok=res['ok'], obs=_compact(res['obs'], 160))
        return True

    def decided_traits(self, item: Article) -> list[str]:
        out: list = []
        for c in iter(getattr(self, 'conditions', [])):
            if c['src'] is item:
                out.append(c['prop'])
            out += [str(b['property']) for _, d, b in c['branches'] if d is item]
        if any((a is item for a, _ in self.match_pairs())):
            out.append('compatibility')
        _rv111: list = list(dict.fromkeys(out))
        return _rv111

    def door(self, moment: str, intent: str) -> bool | None:
        if self.door_invokes.get(moment, 0) >= 3 or self.moves_remaining() < 2 or self.time_remaining() < CONCLUDE_SECS:
            return False
        self.door_invokes[moment] = self.door_invokes.get(moment, 0) + 1
        ending: Any = {(self.jobs2.get(r) or {}).get('name') for r in ('checkout', 'decline', 'cart_add', 'cart_set')} - {None}
        tools: list = []
        for _it112 in self.specs:
            if _it112 not in ending:
                tools.append(dict(name=_it112, description=self.instrument_description(_it112), parameters=self.specs[_it112]))
        basket: list = []
        for lid, qty in iter(self.trolley_sig()):
            item: Any = self.basket_article.get(lid)
            l: Any = item.pool.get(lid) if item else None
            basket.append({'id': lid, 'quantity': qty, 'title': (l.title if l else '')[:120], 'price': l.cost_of if l else None, 'for': item.name if item else None})
        total: float = sum(((b['price'] or 0) * b['quantity'] for b in basket))
        left_msgs: Any = self.note_allowance - self.notes_dispatched if self.note_allowance is not None else None
        user: Any = 'REQUEST:\n' + self.ask2 + '\n\nPROCEDURE THE REQUEST REQUIRES:\n' + json.dumps(self.routine, ensure_ascii=False) + ''.join(('\n\nMOMENT: ', format(moment, ''), ' (next the assistant will ', format(intent, ''), ')')) + '\n\nBASKET NOW: ' + json.dumps(dict(lines=basket, total=total), ensure_ascii=False) + ''.join(('\nMESSAGES REMAINING: ', format(left_msgs if left_msgs is not None else 'no limit', ''), '; STEPS REMAINING: ', format(self.moves_remaining(), ''))) + '\n\nTOOLS:\n' + json.dumps(tools, ensure_ascii=False) + '\n\nCALLS SO FAR (with results, earliest first):\n' + json.dumps(self.past[-30:], ensure_ascii=False)
        answer: Any = self.llm.json([SCHEME_LLM, BACKUP_LLM], DOOR_BRIEF, user, 900)
        if not isinstance(answer, dict):
            self.remark2('gate_failed', moment=moment)
            return None
        ran: bool = False
        acts: list = []
        for _it113 in (answer.get('actions') or [])[:2]:
            if isinstance(_it113, dict):
                acts.append(_it113)
        msg_tool: Any = (self.jobs2.get('message') or {}).get('name')
        for k, act in enumerate(acts):
            if act.get('tool') not in self.specs or act.get('tool') in ending:
                continue
            name: Any = act['tool']
            if name != msg_tool:
                due: Any = max(sum((1 for a in acts[k + 1:] if a.get('tool') == msg_tool)), self.pending_notes())
                if self.moves_remaining() < 2 + due:
                    self.remark2('gate_step_skipped', moment=moment, tool=name, steps=self.moves_remaining(), due=due)
                    continue
            schema: Any = self.specs[name]
            props: Any = schema.get('properties') or {}
            args: Any = {k: v for k, v in (act.get('args') or {}).items() if k in props} if isinstance(act.get('args'), dict) else {}
            for k, v in iter(list(args.items())):
                enum: Any = (props.get(k) or {}).get('enum')
                if isinstance(enum, list) and enum and (v not in enum):
                    args.pop(k)
                elif isinstance(v, str):
                    args[k] = self._trim_params(props.get(k) or {}, v)
            if any((k not in args for k in schema.get('required') or [])):
                continue
            message: bool = name == (self.jobs2.get('message') or {}).get('name')
            if message and (not self.can_note()):
                continue
            if self.moves_remaining() < 2:
                break
            res: Any = self.invoke(name, args)
            if bool(message):
                self.notes_dispatched += 1
                self.sig_confirmed = self.trolley_sig()
            self.remark2('gate_step', moment=moment, tool=name, why=str(act.get('why') or '')[:160], ok=res['ok'])
            ran: bool = True
        if not bool(ran):
            self.remark2('gate_clear', moment=moment)
        return ran

    def handle_rows(self) -> bool:
        lines: list = []
        for _it116 in self.pending_rows:
            if _it116 not in self.handled_rows:
                lines.append(_it116)
        self.pending_rows = []
        if not bool(lines):
            return False
        for l in iter(lines):
            self.handled_rows.add(l)
        user: Any = 'LINES:\n- ' + '\n- '.join(lines) + '\n\nREQUESTED ITEMS: ' + json.dumps([i.name for i in self.items])
        answer: Any = self.llm.json([FAST_LLM, BACKUP_LLM], EVENT_BRIEF, user, 700)
        self.remark2('event', lines=[l[:160] for l in lines], read=_compact(answer, 300))
        changed: bool = False
        if not isinstance(answer, dict):
            return False
        for ch in iter(answer.get('changes') or []):
            if not isinstance(ch, dict):
                continue
            kind: Any = str(ch.get('kind') or '').lower()
            target: Any = None
            if bool(ch.get('item')):
                target: Any = max(self.items, key=lambda it: _similar(it.name, ch.get('item')))
                if _similar(target.name, ch.get('item')) < 0.2 and len(self.items) > 1:
                    target: Any = None
            if kind == 'budget' and _number(ch.get('budget')):
                budget: Any = _number(ch.get('budget'))
                in_lines: list = []
                for _it115 in re.findall('\\d[\\d,]*(?:\\.\\d+)?', ' '.join(lines)):
                    in_lines.append(float(_it115.replace(',', '')))
                if budget not in in_lines:
                    for alt in iter((budget * 100, budget / 100)):
                        if alt in in_lines:
                            budget: Any = alt
                            break
                    else:
                        if in_lines and self.cap is not None:
                            near: list = []
                            for _it114 in in_lines:
                                if self.cap / 20 <= _it114 <= self.cap * 20:
                                    near.append(_it114)
                            if bool(near):
                                budget: Any = near[0]
                self.cap = budget
                changed: bool = True
            elif kind == 'drop' and ch.get('property'):
                for item in iter([target] if target else self.items):
                    before: int = len(item.requirements)
                    item.requirements = [r for r in item.requirements if _similar(r.prop, ch['property']) < 0.5]
                    if len(item.requirements) != before:
                        changed: bool = True
                        for v in iter(item.rulings.values()):
                            v['failed'] = {f for f in v['failed'] if _similar(f.split(':')[0], ch['property']) < 0.5}
            elif kind == 'add' and ch.get('property') and ch.get('value'):
                for item in iter([target] if target else self.items[:1]):
                    item.requirements.append(Requirement(ch['property'], ch.get('op') or 'is', ch['value'], None, origin='shopper'))
                    changed: bool = True
            elif kind in ('price', 'stock'):
                changed: bool = True
                self.refresh_basket_articles()
            elif kind == 'step' and isinstance(ch.get('step'), dict):
                self.append_move(ch['step'])
        return changed

    def hints_read(self) -> None:
        blocks: list = []
        for item in iter(self.items):
            blocks.append(dict(item=item.name, category=item.category, known_requirements=item.req_proses(), hints=item.hints, property_menu=[e['property'] for e in self.trait_catalogue(item)]))
        user: Any = 'REQUEST:\n' + self.ask2 + '\n\nTOPICS THE REQUEST SAYS TO BRING UP WITH THE SHOPPER: ' + json.dumps(self.query_about) + '\n\nITEMS:\n' + json.dumps(blocks, ensure_ascii=False)
        answer: Any = self.llm.json([SCHEME_LLM, FAST_LLM], SCHEME_BRIEF, user, 500)
        self.hinted: list[tuple[Article, str]] = []
        self.junk: set[str] = set()
        if not isinstance(answer, dict):
            return
        for q in iter(answer.get('hinted') or []):
            if not isinstance(q, dict) or not q.get('property'):
                continue
            target: Any = max(self.items, key=lambda it: _similar(it.name, q.get('item') or it.name))
            prop: Any = re.sub('\\s+', ' ', str(q['property']).replace('_', ' ')).strip().lower()
            if prop and (not any((it is target and _near(p, prop) for it, p in self.hinted))):
                self.hinted.append((target, prop))
        self.junk = {re.sub('\\s+', ' ', str(j).replace('_', ' ')).strip().lower() for j in answer.get('junk') or [] if j}
        self.remark2('hints', hinted=[(it.name, p) for it, p in self.hinted], junk=sorted(self.junk)[:12])

    def multi_scheme(self, count: int, taken: Any) -> list[tuple[Article, str]]:
        queues: dict[int, list[str]] = {}
        for _it118 in self.items:
            queues[_it118.index] = []
        first: list[int] = []
        for it, pr in iter(self.scheduled):
            if it.index in queues and (not taken(it, pr)) and (not any((_near(pr, q) for q in queues[it.index]))):
                queues[it.index].append(pr)
                if it.index not in first:
                    first.append(it.index)
        for it in iter(self.items):
            menu: list = []
            for _it117 in self.trait_catalogue(it):
                if _it117['listings'] >= 2 or _it117['of'] <= 2:
                    menu.append(_it117)
            q: Any = queues[it.index]
            top: Any = next((e['property'] for e in menu if e['listings'] >= 3), None)
            if top and (not taken(it, top)) and (not any((_near(top, x) for x in q[:2]))):
                q[:] = [x for x in q if not _near(x, top)]
                q.insert(min(1, len(q)), top)
            often: Any = self.often_kept(it, taken)[:1]
            if often and (not any((_near(often[0], x) for x in q[:2]))):
                q[:] = [x for x in q if not _near(x, often[0])]
                q.insert(min(1, len(q)), often[0])
            for e in iter(menu):
                if not taken(it, e['property']) and (not any((_near(e['property'], x) for x in q))):
                    q.append(e['property'])
            if it.index not in first:
                first.append(it.index)
        told: set = set()
        for _it119 in self.items:
            if any((r.origin == 'shopper' for r in _it119.requirements)):
                told.add(_it119.index)
        by_index: dict = {}
        for _it120 in self.items:
            by_index[_it120.index] = _it120
        turn: list = sorted(first, key=lambda i: (i in told, len(by_index[i].queried), first.index(i)))
        plan: list[tuple[Article, str]] = []
        while len(plan) < count and any((queues[i] for i in turn)):
            for i in iter(turn):
                if queues[i] and len(plan) < count:
                    plan.append((by_index[i], queues[i].pop(0)))
        return plan

    def parked_notes(self) -> int:
        if 'message' not in self.jobs2 or not self.routine:
            return 0

        def need(moments: tuple[str, ...]) -> int:
            _rv121: float = sum((max(1, int(st.get('messages') or 1)) for st in self.routine if st['kind'] == 'message' and st['moment'] in moments))
            return _rv121
        held: Any = need(('before_order', 'before_finish'))
        if self.note_allowance is not None:
            held: Any = min(held, max(0, self.note_allowance - self.notes_dispatched))
        return held

    def pending_notes(self) -> int:
        _rv122: Any = 0 if self.basket2 and self.sig_confirmed == self.trolley_sig() else self.parked_notes()
        return _rv122

    def query2(self, item: Article, prop: str, narrow: bool=False) -> dict[str, Any] | None:
        spec: Any = self.jobs2['message']
        noun: Any = item.name
        text: str = ''.join(('Which ', format(prop, ''), ' would you like the ', format(noun, ''), ' to have?'))
        if bool(narrow):
            seen: list = []
            for l in iter(item.pool.values()):
                for k, v in iter(l.data_points.items()):
                    if _near(re.sub('(?<=[a-z0-9])(?=[A-Z])', ' ', str(k)).replace('_', ' '), prop) and str(v) not in seen:
                        seen.append(str(v)[:30])
            examples: str = ' or '.join(seen[:2])
            text: Any = ''.join(('Just this one thing: for the ', format(noun, ''), ', what ', format(prop, ''), ' do you need')) + (''.join((', for example ', format(examples, ''), '?')) if examples else '?')
        text: Any = self._trim('message', spec['text'], text)
        res: Any = self.invoke(spec['name'], {spec['text']: text})
        self.notes_dispatched += 1
        if not bool(narrow):
            item.queried.append(_norm(prop))
        if not bool(res['ok']):
            return None
        obs: Any = res['obs']
        reply: Any = None
        if isinstance(obs, str):
            reply: Any = obs
        elif isinstance(obs, dict):
            strings: list = []
            for _it123 in obs.values():
                if isinstance(_it123, str):
                    strings.append(_it123)
            reply: Any = max(strings, key=len) if strings else _compact(obs, 400)
        if not bool(reply):
            return None
        known: list = []
        for _it124 in item.requirements:
            if _it124.origin == 'shopper' and _it124.value:
                known.append(''.join((format(_it124.prop, ''), ': ', format(_it124.op, ''), ' ', format(_it124.value, ''))))
        user: str = ''.join(('PRODUCT: ', format(noun, ''), '\nPROPERTY THE QUESTION WAS ABOUT: ', format(prop, ''), '\nQUESTION SENT: ', format(text, ''), '\nKNOWN NEEDS: ', format(json.dumps(known), ''), "\nTHE SHOPPER'S REPLY: ", format(reply, '')))
        answer: Any = self.llm.json([FAST_LLM, BACKUP_LLM, SCHEME_LLM], RESPONSE_BRIEF, user, 600)
        if isinstance(answer, dict) and str(answer.get('kind') or '').lower() == 'need':
            named: Any = self.trait_named(item, reply, [prop, str(answer.get('property') or '')], str(answer.get('value') or ''))
            if bool(named):
                answer['property'] = named
        self.remark2('reply', item=noun, prop=prop, reply=reply[:200], read=_compact(answer, 300))
        _rv125: Any = answer if isinstance(answer, dict) else None
        return _rv125

    def scheme_queries(self, count: int, model: str=SCHEME_LLM) -> list[tuple[Article, str]]:
        if count <= 0:
            _rv128: list = []
            return _rv128
        plan: list[tuple[Article, str]] = []

        def taken(item: Article, prop: str) -> bool:
            if any((_near(prop, a) for a in item.queried)):
                return True
            if any((_near(prop, d) or _similar(prop, d) >= 0.5 for d in self.decided_traits(item))):
                return True
            _rv126: Any = True if any((_near(prop, r.prop) for r in item.requirements)) else any((it is item and _near(prop, p) for it, p in plan))
            return _rv126
        if not bool(getattr(self, 'scheduled', None)):
            blocks: list = []
            for item in iter(self.items):
                blocks.append(dict(item=item.name, category=item.category, known_requirements=item.req_proses(), decided_by_the_request=self.decided_traits(item), hints=item.hints, already_asked=item.queried, property_menu=self.trait_catalogue(item)))
            total: Any = self.note_allowance if self.note_allowance is not None else count
            user: Any = 'REQUEST:\n' + self.ask2 + '\n\nTOPICS THE REQUEST SAYS TO BRING UP WITH THE SHOPPER: ' + json.dumps(self.query_about) + '\n\nITEMS:\n' + json.dumps(blocks, ensure_ascii=False) + ''.join(('\n\nQUESTION_COUNT = ', format(max(total, count), '')))
            answer: Any = self.llm.json([model, FAST_LLM if model != FAST_LLM else BACKUP_LLM], SCHEME_BRIEF, user, 700)
            ordered: list[tuple[Article, str]] = []
            if isinstance(answer, dict) and isinstance(answer.get('questions'), list):
                for q in iter(answer['questions']):
                    if not isinstance(q, dict) or not q.get('property'):
                        continue
                    target: Any = max(self.items, key=lambda it: _similar(it.name, q.get('item') or it.name))
                    prop: Any = re.sub('\\s+', ' ', str(q['property']).replace('_', ' ')).strip().lower()
                    if prop and (not any((it is target and _near(prop, p) for it, p in ordered))):
                        ordered.append((target, prop))
            self.scheduled = ordered
            self.remark2('plan_questions', questions=[(it.name, p) for it, p in ordered])
        if len(self.items) > 1:
            _rv129: Any = self.multi_scheme(count, taken)
            return _rv129
        for item, prop in iter(self.scheduled):
            if len(plan) >= count:
                break
            if not bool(taken(item, prop)):
                plan.append((item, prop))
        if count >= 3:
            top_n: Any = 2 if len(self.items) == 1 else 1
            for item in iter(self.items):
                tops: Any = [e['property'] for e in self.trait_catalogue(item) if e['listings'] >= 3][:top_n]
                for k, prop in enumerate(tops):
                    if bool(taken(item, prop)):
                        continue
                    if len(plan) < count:
                        plan.append((item, prop))
            plan: Any = plan[:count]
        for k, prop in enumerate(self.often_kept(self.items[0], taken)[:2]):
            if len(plan) < count and (not any((it is self.items[0] and _near(p, prop) for it, p in plan))):
                plan.append((self.items[0], prop))
        plan: Any = plan[:count]
        if len(plan) < count:
            menus: dict = {}
            for _it127 in self.items:
                menus[_it127.index] = [e for e in self.trait_catalogue(_it127) if e['listings'] >= 2 or e['of'] <= 2]
            order: list = sorted(self.items, key=lambda it: (sum((1 for h, _ in plan if h is it)) + len(it.queried), it.index))
            while len(plan) < count:
                added: bool = False
                for item in iter(order):
                    while menus[item.index]:
                        prop: Any = menus[item.index].pop(0)['property']
                        if bool(taken(item, prop)):
                            continue
                        plan.append((item, prop))
                        added: bool = True
                        break
                    if len(plan) >= count:
                        break
                if not bool(added):
                    break
        _rv130: Any = plan[:count]
        return _rv130

    def trait_catalogue(self, item: Article) -> list[dict[str, Any]]:
        opened: list = []
        for _it132 in item.pool.values():
            if _it132.unfolded and self.ok_category(item, _it132):
                opened.append(_it132)
        alive: list = []
        for _it133 in opened:
            if not (item.rulings.get(_it133.ident) or {}).get('failed'):
                alive.append(_it133)
        if len(alive) >= 4:
            opened: Any = alive
        per: list[dict[str, str]] = []
        for l in iter(opened):
            f: dict[str, str] = {}
            for k, v in iter(l.data_points.items()):
                key: Any = re.sub('(?<=[a-z0-9])(?=[A-Z])', ' ', str(k))
                key: Any = re.sub('\\s+', ' ', key.replace('_', ' ').replace('.', ' ').replace('-', ' ')).strip().lower()
                if key and set(key.split()) <= _NOT_ASKED:
                    continue
                if key and key not in f:
                    f[key] = str(v).strip().lower()[:60]
            per.append(f)
        counts: dict[str, int] = {}
        for f in iter(per):
            for k in iter(f):
                counts[k] = counts.get(k, 0) + 1
        keys: list = sorted(counts, key=lambda k: (-counts[k], len(k), k))
        alias: dict[str, str] = {}
        for i, a in enumerate(keys):
            if a in alias:
                continue
            for b in iter(keys[i + 1:]):
                if b in alias:
                    continue
                both: list = []
                for _it131 in per:
                    if a in _it131 and b in _it131:
                        both.append(_it131)
                if len(both) >= 2 and sum((1 for f in both if f[a] == f[b])) >= 0.8 * len(both):
                    alias[b] = a
        fixed: Any = {r.key() for r in item.requirements} | set(item.queried)
        menu: list = []
        for key in iter(keys):
            if key in alias:
                continue
            names: Any = [key] + [b for b, a in alias.items() if a == key]
            if any((_norm(n) in fixed or any((_similar(n, f) >= 0.67 for f in fixed)) for n in names)):
                continue
            carriers: float = sum((1 for f in per if any((n in f for n in names))))
            values: list[str] = []
            for f in iter(per):
                v: Any = next((f[n] for n in names if n in f), None)
                if v and v not in values:
                    values.append(v)
            weight: Any = carriers * (0.5 if len(values) == 1 and carriers >= 3 else 1.0)
            distinct: int = len({next((f[n] for n in names if n in f), None) for f in per} - {None})
            if carriers >= 4 and distinct >= carriers:
                weight *= 0.5
            menu.append({'property': key, 'listings': carriers, 'of': len(opened), 'values': values[:3], **(dict(same_as=names[1:4]) if len(names) > 1 else {}), '_w': weight})
        menu.sort(key=lambda e: (-e['_w'], e['property']))
        for e in iter(menu):
            e.pop('_w', None)
        _rv134: Any = menu[:40]
        return _rv134

    def trait_named(self, item: Article, reply: str, asked: list[str], value: str) -> str | None:
        text: Any = ' ' + _norm(reply) + ' '
        if any((a and _norm(a) and (' ' + _norm(a) + ' ' in text) for a in asked)):
            return None
        keys: set[str] = set()
        for l in iter(item.pool.values()):
            for k in iter(l.data_points):
                key: Any = _norm(re.sub('(?<=[a-z0-9])(?=[A-Z])', ' ', str(k)))
                if len(key) >= 4 and (not _pieces_of_text(key) & _pieces_of_text(value)) and (not any((_near(key, a) for a in asked if a))):
                    keys.add(key)
        hits: list = []
        for _it135 in keys:
            if ' ' + _it135 + ' ' in text:
                hits.append(_it135)
        hits: list = [k for k in hits if not any((k != o and ' ' + k + ' ' in ' ' + o + ' ' for o in hits))]
        _rv136: Any = hits[0] if len(hits) == 1 else None
        return _rv136

    def trolley_sig(self) -> tuple:
        _rv137: tuple = tuple(sorted(((lid, int(q or 0)) for lid, q in self.basket2.items() if q)))
        return _rv137

class Tracing:

    def reference_resolve(self, item: Article, req: Requirement, hop: int=0) -> None:
        ref: Any = req.reference or {}
        floor: Any = max(4, getattr(self, 'floor_ref', 0))
        if hop > 7 or self.moves_remaining() < floor or self.time_remaining() < CONCLUDE_SECS + 20:
            return
        seller: Any = str(ref.get('seller') or '').strip()
        category: Any = str(ref.get('category') or '').strip()
        attrs: Any = ref.get('attributes') if isinstance(ref.get('attributes'), dict) else {}
        target_prop: Any = str(ref.get('property') or req.prop).strip()
        scratch: Any = Article(dict(name=category or 'listing', category=category, search_terms=[]), 99)
        attr_words: str = ' '.join((str(v) for v in list(attrs.values())[:3] if v))
        queries: list = []
        if bool(seller):
            queries.append((seller + ' ' + attr_words).strip())
            if bool(category):
                queries.append(seller + ' ' + category)
            queries.append(seller)
        elif bool(category):
            queries.append((category + ' ' + attr_words).strip())

        def seller_ok(l: Offer) -> bool:
            _rv138: Any = True if not bool(seller) else _similar(l.seller, seller) >= 0.5 or _similar(l.brand, seller) >= 0.5 or _norm(seller) in _norm(l.title + ' ' + l.text + ' ' + l.remarks)
            return _rv138

        def overlap(l: Offer) -> int:
            words: Any = _pieces_of_text(attr_words)
            if not bool(words):
                return 0
            text: Any = _pieces_of_text(l.title + ' ' + l.text + ' ' + l.remarks + ' ' + json.dumps(l.data_points))
            _rv139: int = len(words & text)
            return _rv139

        def attrs_in_text(l: Offer) -> bool:
            if not bool(attrs):
                return False
            text: Any = _tokens2(' '.join([l.title or '', l.remarks or '', l.text or '', ' '.join((''.join((format(k, ''), ' ', format(v, ''))) for k, v in l.data_points.items()))]))
            _rv140: bool = all((_tokens2(v) and _tokens2(v) <= text for v in attrs.values() if str(v).strip()))
            return _rv140

        def attrs_match(l: Offer) -> bool:
            if not attrs or not l.unfolded:
                return False
            facts: list = []
            for _it141, _it142 in l.data_points.items():
                facts.append((_norm(re.sub('(?<=[a-z0-9])(?=[A-Z])', ' ', str(_it141))), _norm(_it142)))
            for k, v in iter(attrs.items()):
                want: Any = _norm(v)
                if not want or not any(((_near(fk, k) or _similar(fk, k) >= 0.5) and fv == want for fk, fv in facts)):
                    return False
            return True

        def best_described(listings: list[Offer]) -> Offer | None:
            values: list = []
            for _it143 in attrs.values():
                if str(_it143).strip() and _tokens2(_it143):
                    values.append(_tokens2(_it143))
            if not bool(values):
                return None
            scored: list = []
            for l in iter(listings):
                if not bool(seller_ok(l)):
                    continue
                text: Any = _tokens2(' '.join([l.title or '', l.remarks or '', ' '.join((str(v) for v in l.data_points.values()))]))
                scored.append((sum((1 for v in values if v <= text)), l))
            scored.sort(key=lambda x: -x[0])
            if not scored or scored[0][0] < max(1, -(-3 * len(values) // 5)) or (len(scored) > 1 and scored[1][0] == scored[0][0]):
                return None
            self.remark2('reference_best_described', page=scored[0][1].ident, stated=scored[0][0], of=len(values))
            _rv144: Any = scored[0][1]
            return _rv144
        found: list[Offer] = []
        for q in iter(queries):
            if self.moves_remaining() < 4:
                break
            for _ in range(2):
                if self.search(scratch, query=q, max_pages=1, budget=3) == 0:
                    break
                found: list = []
                for _it145 in scratch.pool.values():
                    if seller_ok(_it145) and (not category or self.ok_category(scratch, _it145)):
                        found.append(_it145)
                if found and (not attrs or any((overlap(l) > 0 or attrs_in_text(l) for l in found))):
                    break
            if len(found) >= 2 or (found and overlap(found[0]) > 0):
                break
        if not bool(found):
            self.remark2('reference_unfound', seller=seller, category=category)
            return
        found.sort(key=lambda l: (-overlap(l), not _similar(l.seller, seller) >= 0.5, l.cost_of or 0))
        page: Any = None
        tried: int = 0
        pick: Any = str(ref.get('pick') or '').strip()
        count: Any = ref.get('count') if isinstance(ref.get('count'), int) else None
        if pick or (count or 0) > 1:
            for start in range(0, min(len(found), 3 * self.compare_cap), self.compare_cap):
                if self.moves_remaining() < 3:
                    break
                for l in iter(self.unfold_bundle(None, [x.ident for x in found[start:start + self.compare_cap]])):
                    scratch.pool[l.ident] = l
            records: list = []
            for _it146 in found:
                if _it146.ident in scratch.pool and scratch.pool[_it146.ident].unfolded:
                    records.append(scratch.pool[_it146.ident].entry2(1200))
            if bool(records):
                user: Any = 'DESCRIPTION: category ' + category + '; seller ' + seller + '; attributes ' + json.dumps(attrs, ensure_ascii=False) + (''.join(('\nPICK: ', format(pick, ''))) if pick else '') + (''.join(('\nHOW MANY LISTINGS QUALIFY: ', format(count, ''))) if count else '') + '\n\nCANDIDATES:\n' + json.dumps(records, ensure_ascii=False)
                match: Any = self.llm.json([FAST_LLM, BACKUP_LLM], MATCH_BRIEF, user, 600)
                if isinstance(match, dict) and match.get('id') in scratch.pool:
                    page: Any = scratch.pool[match['id']]
            found: Any = [] if page is not None else found
        single: Any = not self.jobs2.get('compare') or getattr(self, 'off_compare', False)
        size: Any = 1 if single else self.compare_cap
        seen: list[Offer] = []
        while found and tried < (5 if single else 3) and (page is None) and (self.moves_remaining() >= 3):
            batch: Any = found[:size]
            found: Any = found[size:]
            tried += 1
            opened: Any = self.unfold_bundle(None, [l.ident for l in batch])
            for l in iter(opened):
                scratch.pool[l.ident] = l
            now: list = []
            for _it147 in batch:
                if _it147.ident in scratch.pool and scratch.pool[_it147.ident].unfolded:
                    now.append(scratch.pool[_it147.ident])
            exact: list = []
            for _it148 in now:
                if attrs_match(_it148):
                    exact.append(_it148)
            if bool(exact):
                page: Any = exact[0]
                break
            seen += now
            if single and found and (len(seen) < 3):
                continue
            shown: Any = seen if single else now
            records: list = []
            for _it149 in shown:
                records.append(_it149.entry2(1600))
            seen: list = []
            if not bool(records):
                continue
            user: Any = 'DESCRIPTION: category ' + category + '; seller ' + seller + '; attributes ' + json.dumps(attrs, ensure_ascii=False) + '\n\nCANDIDATES:\n' + json.dumps(records, ensure_ascii=False)
            match: Any = self.llm.json([FAST_LLM, BACKUP_LLM], MATCH_BRIEF, user, 600)
            worded: list = []
            for _it150 in shown:
                if attrs_in_text(_it150):
                    worded.append(_it150)
            if isinstance(match, dict) and match.get('id') in scratch.pool:
                page: Any = scratch.pool[match['id']]
            elif len(worded) == 1:
                page: Any = worded[0]
            elif len(records) == 1 and (not found):
                page: Any = scratch.pool[records[0]['id']]
            else:
                page: Any = best_described(shown)
        if page is None:
            self.remark2('reference_unmatched', seller=seller, category=category)
            return
        paired: bool = req.op == 'counterpart of reference'

        def read_page() -> Any:
            user: Any = ''.join(('PROPERTY NEEDED: ', format(target_prop, ''), '\n\nLISTING PAGE:\n')) + json.dumps(page.entry2(3000), ensure_ascii=False)
            _rv151: Any = self.llm.json([FAST_LLM, BACKUP_LLM], REFER_BRIEF, user, 600)
            return _rv151
        thin: Any = self.compare_thin and (not page.data_points)
        if thin and (not page.remarks) and (self.moves_remaining() >= 3):
            page.data_points, thin = (self.fetch_data_points(page.ident) or {}, False)
        read: Any = read_page()
        if thin and self.moves_remaining() >= 3 and isinstance(read, dict) and (not read.get('redirect')) and (not read.get('value') or paired):
            page.data_points = self.fetch_data_points(page.ident) or {}
            if bool(page.data_points):
                read: Any = read_page()
        tried_pages: set = {page.ident}
        while not (isinstance(read, dict) and (read.get('value') or read.get('paired') or read.get('redirect'))) and len(tried_pages) < 4 and (self.time_remaining() > CONCLUDE_SECS + 15):
            others: list = []
            for _it152 in scratch.pool.values():
                if _it152.unfolded and _it152.ident not in tried_pages and seller_ok(_it152):
                    others.append(_it152)
            if not bool(others):
                break
            others.sort(key=lambda l: (_norm(target_prop) not in _norm(l.remarks or ''), -overlap(l)))
            page: Any = others[0]
            tried_pages.add(page.ident)
            self.remark2('reference_next_page', page=page.ident)
            read: Any = read_page()
        self.remark2('reference', hop=hop, page=page.ident, read=_compact(read, 200))
        redirect: Any = read.get('redirect') if isinstance(read, dict) else None
        if isinstance(redirect, dict) and redirect:
            about: str = str(redirect.get('property') or target_prop)
            if not (_near(about, target_prop) or _similar(about, target_prop) >= 0.5 or _norm(target_prop) in _norm(about) or (_norm(about) in _norm(target_prop))):
                redirect: Any = None
                if not read.get('value') and thin and (not page.data_points) and (self.moves_remaining() >= 3):
                    page.data_points = self.fetch_data_points(page.ident) or {}
                    again: Any = read_page() if page.data_points else None
                    if isinstance(again, dict):
                        read['value'] = again.get('value') or read.get('value')
                        read['paired'] = again.get('paired') or read.get('paired')
                if not bool(read.get('value')):
                    own: Any = next((v for k, v in page.data_points.items() if _near(str(k).replace('_', ' '), target_prop) or _similar(str(k).replace('_', ' '), target_prop) >= 0.5), None)
                    if own not in (None, ''):
                        read['value'] = str(own)
                self.remark2('reference_other_redirect', about=about, wanted=target_prop, value=read.get('value'))
        if isinstance(redirect, dict) and redirect:
            req.reference = redirect
            req.reference['property'] = target_prop
            self.reference_resolve(item, req, hop + 1)
            return
        value: Any = read.get('paired' if paired else 'value') if isinstance(read, dict) else None
        if bool(value):
            req.value = str(value)
            req.op = OPS_REF.get(req.op, 'is')
            req.reference = None
            if req.op == 'is' and len(req.value) <= 40 and item.words2 and (not any((_norm(req.value) in _norm(t) for t in item.words2))):
                item.words2.insert(min(item.lookup_done, len(item.words2)), ''.join((format(req.value, ''), ' ', format(item.noun, '')))[:60])
        elif paired and isinstance(read, dict) and read.get('value'):
            self.remark2('pair_unstated', page=page.ident, value=read.get('value'))

class Checkout:

    def basket_append(self, item: Article, listing: Offer) -> bool:
        if not self.jobs2.get('cart_add') and self.jobs2.get('cart_set'):
            lines: dict = dict(self.basket2)
            lines[listing.ident] = item.amount
            if not bool(self.set_basket(lines)):
                return False
            self.basket2 = lines
            self.basket_article[listing.ident] = item
            return True
        spec: Any = self.jobs2.get('cart_add')
        if not bool(spec):
            return False
        args: dict[str, Any] = {spec['id']: listing.ident}
        qty: Any = item.amount
        self.call_intent = ''.join(('add listing ', listing.ident, ' to the cart; the shopper wants ', str(qty), ' unit(s) of it'))
        if bool(spec.get('qty')):
            props: Any = self._spec('cart_add').get('properties') or {}
            maximum: Any = (props.get(spec['qty']) or {}).get('maximum')
            if isinstance(maximum, int) and qty > maximum:
                qty: Any = maximum
            args[spec['qty']] = qty
        mode: Any = spec.get('mode') if isinstance(spec.get('mode'), dict) else None
        if mode and mode.get('arg') and (mode.get('add') is not None):
            args[mode['arg']] = mode['add']
        res: Any = self.invoke(spec['name'], args)
        if bool(res['ok']):
            self.basket2[listing.ident] = qty
            self.basket_article[listing.ident] = item
            self.cart_verify(item, listing, args, res.get('obs'))
            return True
        self.remark2('cart_add_failed', error=res['error'], obs=_compact(res['obs'], 200))
        return False

    def cart_line(self, obs: Any, lid: str, depth: int=0) -> Any:
        if depth > 6:
            return None
        if isinstance(obs, dict):
            if lid in [v for v in obs.values() if isinstance(v, str)]:
                return obs
            for v in iter(obs.values()):
                got: Any = self.cart_line(v, lid, depth + 1)
                if got is not None:
                    return got
        elif isinstance(obs, list):
            for v in iter(obs):
                got: Any = self.cart_line(v, lid, depth + 1)
                if got is not None:
                    return got
        return None

    def cart_verify(self, item: Article, listing: Offer, args: dict[str, Any], seen: Any=None) -> None:
        if getattr(self, 'cart_trusted', False) or getattr(self, 'cart_checks', 0) >= 3:
            return
        args = getattr(self, 'cart_sent', {}).get(listing.ident) or args
        want: int = int(self.basket2.get(listing.ident, item.amount) or 1)
        line: Any = self.cart_line(seen, listing.ident)
        view: Any = self.jobs2.get('cart_view') or {}
        shown: Any = seen
        if line is None:
            if not view.get('name') or view.get('name') not in self.specs or self.moves_remaining() < 3:
                return
            self.cart_checks = getattr(self, 'cart_checks', 0) + 1
            got: Any = self.invoke(view['name'], self.pinned_params('cart_view'))
            if not got.get('ok'):
                return
            shown = got.get('obs')
            line = self.cart_line(shown, listing.ident)
        if line is not None:
            nums: list = [v for k, v in line.items() if isinstance(v, int) and (not isinstance(v, bool)) and re.search('qty|quant|count|unit|held|piece|amount|number', str(k), re.I)]
            if want in nums or (not nums):
                spec0: Any = self.jobs2.get('cart_add') or {}
                if want in nums and (not spec0.get('qty')):
                    for k, v in iter(args.items()):
                        if k != spec0.get('id') and isinstance(v, int) and (not isinstance(v, bool)) and (v == want):
                            spec0['qty'] = k
                            self.remark2('cart_qty_learned', arg=k)
                            self.cart_trusted = True
                            self.resend_lines(listing.ident)
                            break
                self.cart_trusted = True
                return
            problem: str = ''.join(('the call returned ok, but the cart now shows this line for the listing: ', _compact(line, 300), '; the shopper wants ', str(want), ' unit(s)'))
        else:
            problem = ''.join(('the call returned ok, but the cart afterwards does not contain the listing at all: ', _compact(shown, 400)))
        spec: Any = self.jobs2.get('cart_add') or {}
        tried: list = getattr(self, 'cart_tried', [])
        if args not in tried:
            tried = tried + [dict(args)]
            self.cart_tried = tried
        problem += '; argument sets already tried with no effect: ' + _compact(tried, 600)
        self.call_intent = ''.join(('add listing ', listing.ident, ' to the cart so that it holds ', str(want), ' unit(s) of it'))
        fixed: Any = self.repair_args(spec.get('name') or '', args, problem)
        self.call_intent = ''
        if not isinstance(fixed, dict) or fixed == args or fixed in tried:
            fixed = self.vary_args(spec.get('name') or '', args, want, tried)
        if not isinstance(fixed, dict) or self.moves_remaining() < 2:
            return
        again: Any = self.invoke_once(spec['name'], fixed)
        self.remark2('cart_fixed', args=_compact(fixed, 160), ok=again.get('ok'))
        if again.get('ok'):
            q: Any = spec.get('qty')
            if q and q in fixed and isinstance(fixed[q], int):
                self.basket2[listing.ident] = fixed[q]
            self.cart_verify(item, listing, fixed, again.get('obs'))

    def vary_args(self, name: str, args: dict[str, Any], want: int, tried: list) -> dict[str, Any] | None:
        props: Any = (self.specs.get(name) or {}).get('properties') or {}
        idk: Any = (self.jobs2.get('cart_add') or {}).get('id')
        for k, p in iter(props.items()):
            kind: Any = p.get('type') if isinstance(p, dict) else None
            if k == idk or not ({'integer', 'number'} & set(kind if isinstance(kind, list) else [kind])):
                continue
            lo: int = p['minimum'] if isinstance(p.get('minimum'), int) else 0
            hi: int = p['maximum'] if isinstance(p.get('maximum'), int) else max(want, 1) + 3
            for c in [want, 1] + list(range(lo, hi + 1)):
                alt: dict[str, Any] = dict(args, **{k: c})
                if lo <= c <= hi and alt != args and alt not in tried:
                    self.remark2('cart_vary', arg=k, value=c)
                    return alt
        return None

    def resend_lines(self, done: str) -> None:
        spec: Any = self.jobs2.get('cart_add') or {}
        q: Any = spec.get('qty')
        for lid, n in iter(list(self.basket2.items())):
            sent: Any = getattr(self, 'cart_sent', {}).get(lid) or {}
            if lid == done or not q or sent.get(q) == n or self.moves_remaining() < 2:
                continue
            again: Any = self.invoke_once(spec['name'], dict(sent, **{spec['id']: lid, q: n}))
            self.remark2('cart_resent', listing=lid, ok=again.get('ok'))

    def basket_remove(self, listing_id: str) -> bool:
        if not self.jobs2.get('cart_add') and self.jobs2.get('cart_set'):
            lines: dict = {}
            for _it153, _it154 in self.basket2.items():
                if _it153 != listing_id:
                    lines[_it153] = _it154
            if not bool(self.set_basket(lines)):
                return False
            self.basket2 = lines
            self.basket_article.pop(listing_id, None)
            return True
        spec: Any = self.jobs2.get('cart_add')
        if not bool(spec):
            return False
        args: dict[str, Any] = {spec['id']: listing_id}
        self.call_intent = ''.join(('take listing ', listing_id, ' out of the cart completely'))
        mode: Any = spec.get('mode') if isinstance(spec.get('mode'), dict) else None
        if mode and mode.get('arg') and (mode.get('remove') is not None):
            args[mode['arg']] = mode['remove']
            if bool(spec.get('qty')):
                args[spec['qty']] = self.basket2.get(listing_id, 1)
        elif bool(spec.get('qty')):
            props: Any = self._spec('cart_add').get('properties') or {}
            minimum: Any = (props.get(spec['qty']) or {}).get('minimum')
            args[spec['qty']] = 0 if not isinstance(minimum, int) or minimum <= 0 else minimum
        res: Any = self.invoke(spec['name'], args)
        if bool(res['ok']):
            self.basket2.pop(listing_id, None)
            self.basket_article.pop(listing_id, None)
            return True
        return False

    def write_note(self, buying: bool, note: str) -> str:
        if not getattr(self, 'closing_note', '') or self.time_remaining() < 10:
            return note
        lines: list = [dict(item=it.name, listing=it.chosen.title[:90], unit_price=it.chosen.cost_of, quantity=self.basket2.get(it.chosen.ident, it.amount), seller=it.chosen.seller or None)
                       for it in self.items if it.chosen is not None and it.chosen.ident in self.basket2] if buying else []
        user: str = ''.join(('WHAT THE SHOPPER WANTS THE NOTE TO SAY: ', self.closing_note, '\nOUTCOME: ', 'ordering the basket below' if buying else 'nothing is bought',
                             '\nBASKET: ', _compact(lines, 2500), '\nOTHER POINTS TO KEEP: ', note[:500] or '(none)'))
        self.urgent = self.time_remaining() < CONCLUDE_SECS
        try:
            answer: Any = self.llm.json([FAST_LLM, BACKUP_LLM], NOTE_BRIEF, user, 400)
        finally:
            self.urgent = False
        text: str = str(answer.get('note') or '').strip() if isinstance(answer, dict) else ''
        self.remark2('closing_note', text=text[:200])
        return ' '.join((x for x in (note.strip(), text) if x)) or note

    def close_purchase(self, note: str='') -> bool:
        spec: Any = self.jobs2.get('checkout')
        if not bool(spec):
            return False
        if getattr(self, 'other_rules', None) and (not getattr(self, 'preorder_done', False)) and self.basket2 and (self.time_remaining() > 12):
            self.preorder_done = True
            if bool(self.preorder_check()):
                return bool(self.done2)
        if self.routine and self.basket2:
            ran: Any = self.door('before_order', 'order the basket shown below')
            if ran and self.pending_notes() and (self.moves_remaining() >= 2):
                ran: Any = self.door('before_order', 'order the basket shown below') or ran
            if ran is not False and self.confirm_initial and (self.trolley_sig() != self.sig_confirmed):
                ran: Any = self.confirm_trolley() or ran
            if ran and (self.handle_rows() or self.cost_of_or_stock_changed()):
                self.remark2('procedure_changed')
                return False
        params: dict[str, Any] = self.pinned_params('checkout')
        note = self.write_note(True, note)
        rem: Any = (self.jobs2.get('decline') or {}).get('remark')
        cprops: Any = self._spec('checkout').get('properties') or {}
        if note and rem and rem in cprops and rem not in params and (cprops.get(rem) or {}).get('type') in (None, 'string'):
            params[rem] = self._trim('checkout', rem, note)
        sig: Any = tuple(sorted(((k, v) for k, v in self.basket2.items())))
        fails: Any = getattr(self, 'checkout_fails', {})
        if fails.get(sig, 0) >= 2:
            self.remark2('checkout_skipped_same_basket', basket=self.basket2)
            return False
        self.call_intent = 'buy what the cart now holds'
        res: Any = self.invoke(spec['name'], params)
        if not self.done2 and (not res.get('lines')):
            self.checkout_fails = dict(fails, **{sig: fails.get(sig, 0) + 1})
            for lid, it in iter(list(self.basket_article.items())):
                if lid in self.basket2 and lid in it.pool and (not getattr(self, 'cart_trusted', False)):
                    self.cart_verify(it, it.pool[lid], {(self.jobs2.get('cart_add') or {}).get('id') or 'item': lid}, None)
                    break
        if bool(self.done2):
            self.remark2('ordered', cart=self.basket2)
            return True
        self.remark2('order_refused', obs=_compact(res['obs'], 200), lines=res['lines'][:3])
        return False

    def cost_of_or_stock_changed(self) -> bool:
        for item in iter(self.items):
            l: Any = item.chosen
            if l is None:
                continue
            if l.stock is False:
                return True
            if self.cap is not None and l.cost_of is not None and (sum(((it.chosen.cost_of or 0) * it.amount for it in self.items if it.chosen)) > self.cap + 1e-06):
                return True
        return False

    def end_keep_back(self) -> int:
        _rv155: Any = self.keep_back_override if getattr(self, 'keep_back_override', None) is not None else len(self.items) + 6 + (2 if self.jobs2.get('facts') and (not self.jobs2.get('compare')) else 0) + min(4, sum((1 for st in self.routine if st['moment'] != 'start')) + (1 if self.confirm_initial else 0))
        return _rv155

    def force_close(self) -> None:
        if bool(self.done2):
            return
        for item in iter(self.items):
            if item.chosen is None or item.chosen.stock is False:
                fits: Any = self.buyable(item)
                item.chosen = fits[0] if fits else None
        ready: Any = all((it.chosen for it in self.items)) and bool(self.items)
        if ready and self.cap is not None:
            ready: bool = sum(((it.chosen.cost_of or 0) * it.amount for it in self.items)) <= self.cap + 1e-06
        if bool(ready):
            stale: list = []
            for _it156 in self.basket2:
                if not any((it.chosen.ident == _it156 for it in self.items)):
                    stale.append(_it156)
            adds: list = []
            for _it157 in self.items:
                if _it157.chosen.ident not in self.basket2:
                    adds.append(_it157)
            if self.moves_remaining() >= len(stale) + len(adds) + 1:
                for lid in iter(stale):
                    self.basket_remove(lid)
                for item in iter(adds):
                    self.basket_append(item, item.chosen)
                if bool(self.close_purchase()):
                    return
        if not self.done2 and self.kept_cart_holds():
            self.remark2('forced_cart_kept', cart=self.basket2)
            if bool(self.close_purchase()):
                return
        if not bool(self.done2):
            have: dict = {it.index: it.chosen for it in self.items if it.chosen is not None and it.chosen.stock is not False}
            gone: list = [it for it in self.items if it.index not in have]
            if gone and self.act_on_no_fit(have, gone):
                return
            self.refuse('forced')

    def inspection(self, missing: list[Article]) -> bool:
        did: bool = False
        self.keep_back_override = len(self.items) + 3 + max(self.parked_notes(), 1 if self.confirm_initial else 0)
        try:
            for item in iter(missing):
                if self.time_remaining() < CONCLUDE_SECS + 20:
                    break
                cards: list = []
                pool: list = []
                for _it159 in item.pool.values():
                    if self.ok_category(item, _it159) and _it159.stock is not False and (not (self.cap is not None and _it159.cost_of is not None and (_it159.cost_of * item.amount > self.cap + 1e-06))):
                        pool.append(_it159)
                for l in iter(sorted(pool, key=lambda l: (not l.unfolded, l.cost_of if l.cost_of is not None else 1e+18))[:40]):
                    card: dict[str, Any] = dict(id=l.ident, title=l.title[:110], price=l.cost_of, read=l.unfolded)
                    if bool(l.unfolded):
                        v: Any = item.rulings.get(l.ident) or {}
                        card['failed'] = sorted(v.get('failed') or [])[:3]
                        card['unsure'] = sorted(v.get('unsure') or [])[:3]
                        card['properties'] = dict(list(l.data_points.items())[:16])
                        if bool(l.remarks):
                            card['notes'] = l.remarks[:200]
                    else:
                        card['text'] = (l.text or '')[:160]
                    cards.append(card)
                if not bool(cards):
                    continue
                tried: list = sorted({k.split(':', 1)[1] for k in item.cursors if k.startswith('search_text:')})
                user: Any = ''.join(('PRODUCT WANTED: ', format(item.name, ''), ' (category: ', format(item.category, ''), '; quantity ', format(item.amount, ''), ')\nBUDGET FOR ALL LINES COMBINED: ', format(self.cap if self.cap is not None else 'none stated', ''), '\nITS REQUIREMENTS:\n- ')) + ('\n- '.join(item.req_proses()) or 'none') + ''.join(('\nQUERIES ALREADY USED: ', format(json.dumps(tried, ensure_ascii=False), ''), '\n\nLISTINGS ENCOUNTERED:\n')) + _compact(cards, 14000)
                answer: Any = self.llm.json([SCHEME_LLM, FAST_LLM], INSPECTION_BRIEF, user, 1500)
                if not isinstance(answer, dict):
                    continue
                self.remark2('audit', item=item.name, answer=_compact(answer, 300))
                for lid in iter([str(x) for x in answer.get('fits_after_all') or []][:2]):
                    l: Any = item.pool.get(lid)
                    if l is None or not l.unfolded or self.slot_fails(item, l):
                        continue
                    if bool(self._confirm(item, l, strict=True)):
                        item.rulings[l.ident] = dict(failed=set(), unsure=set(), weak=set(), loose=set())
                        item.for_judged[l.ident] = set(item.req_proses())
                        did: bool = True
                fresh: Any = [str(x) for x in answer.get('unopened') or [] if str(x) in item.pool and (not item.pool[str(x)].unfolded)][:4]
                if fresh and self.moves_remaining() - self.end_keep_back() >= 1:
                    if bool(self.unfold_bundle(item, fresh, floor=self.end_keep_back())):
                        did: bool = True
                for q in iter([str(x).strip() for x in answer.get('queries') or [] if str(x).strip()][:2]):
                    if self.moves_remaining() - self.end_keep_back() < 3 or self.time_remaining() < CONCLUDE_SECS + 15:
                        break
                    if _norm(q) in {_norm(t) for t in tried} or self.search(item, query=q[:60], max_pages=1, budget=self.end_keep_back()) <= 0:
                        continue
                    rows: list = []
                    for _it158 in self.contenders(item):
                        if not _it158.unfolded:
                            rows.append(_it158)
                    self.sift(item, rows)
                    rows: Any = [l for l in self.contenders(item) if not l.unfolded and l.sift != 'no'][:2]
                    if rows and self.unfold_bundle(item, [l.ident for l in rows], floor=self.end_keep_back()):
                        did: bool = True
                self.judge(item, [l for l in self.contenders(item) if l.unfolded])
        finally:
            self.keep_back_override = None
        return did

    def place_purchase(self, attempt: int=0) -> None:
        self.purchase_rounds = getattr(self, 'purchase_rounds', 0) + 1
        if attempt > 2 or self.done2 or self.purchase_rounds > 8:
            return
        for lid in iter(list(self.basket2)):
            if not any((it.chosen and it.chosen.ident == lid for it in self.items)):
                self.basket_remove(lid)
        for item in iter(self.items):
            if item.chosen and item.chosen.ident not in self.basket2:
                if not bool(self.basket_append(item, item.chosen)):
                    item.rulings.setdefault(item.chosen.ident, dict(failed=set(), unsure=set()))['failed'].add('cart refused')
                    item.chosen = None
        if any((it.chosen is None for it in self.items)):
            self.select_and_close()
            return
        if self.moves_remaining() > 2 + self.pending_notes() and self.items and self.items[0].chosen:
            first: Any = self.items[0].chosen
            self.unfold_bundle(self.items[0], [first.ident])
        changed: Any = self.handle_rows()
        for item in iter(self.items):
            if item.chosen is not None and item.chosen.ident not in item.for_judged:
                self.judge(item, [item.chosen])
                if bool((item.rulings.get(item.chosen.ident) or {}).get('failed')):
                    self.remark2('chosen_failed', id=item.chosen.ident, failed=sorted(item.rulings[item.chosen.ident]['failed'])[:3])
                    changed: bool = True
        if changed or self.cost_of_or_stock_changed():
            self.remark2('rechoose', cart=self.basket2)
            for item in iter(self.items):
                self.judge(item, [l for l in self.contenders(item) if l.unfolded])
            self.rechoose()
            if bool(self.done2):
                return
            self.place_purchase(attempt + 1)
            return
        if bool(self.close_purchase()):
            return
        changed: Any = self.handle_rows()
        if not bool(changed):
            self.refresh_basket_articles()
            self.handle_rows()
        self.rechoose()
        if not bool(self.done2):
            self.place_purchase(attempt + 1)

    def rechoose(self) -> None:
        for item in iter(self.items):
            self.judge(item, [l for l in self.contenders(item) if l.unfolded])
            if not [f for f in item.matches() if f.stock is not False] or self.cheapest:
                item.want_lookup_done = False
                item.below_sweep = None
                share: Any = max(2, self.moves_remaining() - self.end_keep_back())
                if share >= 3 and self.time_remaining() > CONCLUDE_SECS + 20:
                    self.explore(item, share, final=True)
        for item in iter(self.items):
            fits: list = []
            for _it160 in item.matches(cheapest=self.cheapest):
                if _it160.stock is not False:
                    fits.append(_it160)
            if not bool(fits):
                item.chosen = None
                continue
            if bool(self.cheapest):
                item.chosen = fits[0]
            elif item.chosen is None or item.chosen.ident not in [f.ident for f in fits]:
                item.chosen = fits[0]
        picks: dict = {}
        for _it162 in self.items:
            if _it162.chosen is not None:
                picks[_it162.index] = _it162.chosen
        for dst in iter({d.index: d for c in getattr(self, 'conditions', []) for _, d, _ in c['branches']}.values()):
            self.conditions_apply(dst, picks)
            ok: list = []
            for _it161 in self.buyable(dst):
                ok.append(_it161.ident)
            if dst.chosen is not None and dst.chosen.ident not in ok:
                dst.chosen = dst.pool[ok[0]] if ok else None
        if any((it.chosen is None for it in self.items)):
            self.select_and_close()
            return
        if self.cap is not None:
            total: float = sum(((it.chosen.cost_of or 0) * it.amount for it in self.items))
            if total > self.cap + 1e-06:
                self.select_and_close()

    def refresh_basket_articles(self) -> None:
        for lid, item in iter(list(self.basket_article.items())):
            if self.moves_remaining() <= 2:
                return
            opened: Any = self.unfold_bundle(item, [lid])
            self.remark2('refreshed', id=lid, price=item.pool[lid].cost_of if lid in item.pool else None, stock=item.pool[lid].stock if lid in item.pool else None)

    def director_view(self, per: int=12) -> list[dict[str, Any]]:
        blocks: list = []
        for item in iter(self.items):
            try:
                table: Any = self.slot_grid(item)
            except Exception:
                table = None
            rulings: Any = item.rulings
            pool: list = [l for l in item.pool.values() if self.ok_category(item, l)]
            pool.sort(key=lambda l: (not l.unfolded, len((rulings.get(l.ident) or {}).get('failed') or ()), len((rulings.get(l.ident) or {}).get('unsure') or ()), l.stock is False, l.cost_of if l.cost_of is not None else 1e+18))
            cards: list = []
            for l in iter(pool[:per]):
                card: dict[str, Any] = dict(id=l.ident, title=l.title[:100 if per >= 8 else 70], price=l.cost_of, can_buy=l.stock, opened=l.unfolded)
                if bool(l.unfolded):
                    v: Any = rulings.get(l.ident) or {}
                    card['fails'] = sorted(v.get('failed') or [])[:6]
                    card['unsure'] = sorted(v.get('unsure') or [])[:6]
                    try:
                        extra: Any = sorted(self.slot_fails(item, l, table))[:4]
                    except Exception:
                        extra = []
                    if bool(extra):
                        card['other_checks_failed'] = extra
                cards.append(card)
            reqs: list = [''.join((r.text(), '  [from ', SOURCE_LABEL.get(r.origin, r.origin), ']')) for r in item.active()]
            if getattr(item, 'unit_cap', None) is not None:
                reqs.append(''.join(('price for one unit: at most ', format(item.unit_cap, ''), "  [from the assistant's reading of the request]")))
            try:
                sure: list = [f.ident for f in self.buyable(item)][:6]
            except Exception:
                sure = []
            blocks.append(dict(item=item.name, category=item.category, quantity=item.amount, requirements=reqs, listings_confirmed_to_meet_everything=sure, listings=cards))
        return blocks

    def director_ending(self, reason: str) -> bool:
        rounds: int = getattr(self, 'director_rounds', 0)
        if rounds >= 2 or self.done2 or not self.items or not self.jobs2.get('checkout'):
            return False
        if self.moves_remaining() < 2 or self.time_remaining() < 8:
            return False
        self.director_rounds = rounds + 1
        heard: list = sorted(self.handled_rows)[:12]
        per: int = 12
        blocks: Any = self.director_view(per)
        while per > 3 and len(json.dumps(blocks, ensure_ascii=False, separators=(',', ':'))) > 15000:
            per -= 2
            blocks = self.director_view(per)
        user: str = _director_message(self.ask2, heard, getattr(self, 'dialogue', []), self.cap, self.moves_remaining(), reason or getattr(self, 'refuse_reason', ''), blocks, getattr(self, 'other_rules', []))
        self.urgent = self.time_remaining() < CONCLUDE_SECS
        try:
            answer: Any = self.llm.json([FAST_LLM] if self.time_remaining() < 15 else [SCHEME_LLM, FAST_LLM], DIRECTOR_BRIEF, user, 2500)
        finally:
            self.urgent = False
        self.remark2('director', round=self.director_rounds, answer=_compact(answer, 500))
        if not isinstance(answer, dict):
            return False
        self.director_note = str(answer.get('explanation') or '').strip()
        if str(answer.get('decision') or '').lower() != 'order':
            return False
        picks: dict = {}
        for row in iter(answer.get('lines') or []):
            if not isinstance(row, dict) or not row.get('listing'):
                continue
            lid: str = str(row['listing']).strip()
            owners: list = [it for it in self.items if lid in it.pool]
            if not bool(owners):
                continue
            owner: Any = max(owners, key=lambda it: _similar(it.name, row.get('item') or ''))
            picks[owner.index] = owner.pool[lid]
        if not bool(picks):
            return False
        for item in iter(self.items):
            item.chosen = picks.get(item.index)
        for lid in iter(list(self.basket2)):
            if not any((it.chosen is not None and it.chosen.ident == lid for it in self.items)):
                self.basket_remove(lid)
        for item in iter(self.items):
            if item.chosen is not None and item.chosen.ident not in self.basket2:
                if self.moves_remaining() < 2 or not self.basket_append(item, item.chosen):
                    item.chosen = None
        if not bool(self.basket2):
            return False
        if bool(self.close_purchase(note=self.director_note)):
            return True
        if not self.done2 and self.handle_rows():
            return self.director_ending('the order came back with new lines from the shopper; decide again with them')
        return bool(self.done2)

    def item_named(self, name: Any) -> Any:
        if not name or not self.items:
            return None
        best: Any = max(self.items, key=lambda it: _similar(it.name, name))
        return best if _similar(best.name, name) >= 0.2 or len(self.items) == 1 else None

    def listing_field(self, l: Offer, field: Any) -> Any:
        f: str = str(field or '').lower()
        if not f:
            return None
        if 'price' in f or 'cost' in f:
            return l.cost_of
        if 'seller' in f or 'shop' in f or 'store' in f or 'vendor' in f:
            return (l.seller or '').strip().lower() or None
        if 'brand' in f or 'make' in f or 'maker' in f:
            return (l.brand or '').strip().lower() or None
        for k, v in iter(l.data_points.items()):
            if _near(k, f):
                return str(v).strip().lower()
        return None

    def holds(self, x: Any, op: str, value: Any) -> bool:
        op = str(op or 'is').lower()
        num: Any = _number(value)
        if any((w in op for w in ('least', 'over', 'above', 'more', 'under', 'below', 'less', 'most'))):
            xv: Any = x if isinstance(x, (int, float)) else _number(x)
            if xv is None or num is None:
                return False
            if 'least' in op:
                return xv >= num - 1e-06
            if 'most' in op:
                return xv <= num + 1e-06
            return xv > num if any((w in op for w in ('over', 'above', 'more'))) else xv < num
        vals: list = [str(v).strip().lower() for v in (value if isinstance(value, list) else str(value or '').split('|'))]
        hit: bool = x is not None and any((v and (v == str(x).lower() or v in str(x).lower()) for v in vals))
        return not hit if 'not' in op else hit

    def apply_adjustments(self, picks: dict[int, Offer], qty: dict[int, int], drop: set) -> None:
        for adj in iter(sorted(getattr(self, 'adjustments', []), key=lambda a: str((a.get('when') or {}).get('subject') or '').lower() == 'total')):
            w: Any = adj.get('when') or {}
            subject: str = str(w.get('subject') or '').lower()
            if subject == 'total':
                x: Any = sum(((picks[i].cost_of or 0) * qty[i] for i in picks if i not in drop))
            else:
                it: Any = self.item_named(w.get('item'))
                l: Any = picks.get(it.index) if it is not None else None
                x = None if l is None else (l.cost_of if subject in ('price', 'unit price', 'cost') else self.listing_field(l, subject))
            act: Any = adj.get('then') if self.holds(x, w.get('op'), w.get('value')) else adj.get('otherwise')
            if not isinstance(act, dict):
                continue
            target: Any = self.item_named(act.get('item'))
            if target is None:
                continue
            q: Any = act.get('quantity')
            if act.get('drop') or (isinstance(q, (int, float)) and (not isinstance(q, bool)) and q <= 0):
                drop.add(target.index)
            elif isinstance(q, (int, float)) and (not isinstance(q, bool)):
                qty[target.index] = int(q)

    def basket_ok(self, picks: dict[int, Offer], qty: dict[int, int], drop: set) -> bool:
        live: list = [(it, picks[it.index]) for it in self.items if it.index in picks and it.index not in drop]
        total: float = sum(((l.cost_of or 0) * qty.get(it.index, it.amount) for it, l in live))
        for c in iter(getattr(self, 'basket_checks', [])):
            what: str = str(c.get('what') or '').lower()
            val: Any = _number(c.get('value'))
            scope: Any = self.item_named(c.get('item')) if c.get('item') else None
            spend: Any = None if scope is None or scope.index not in picks or scope.index in drop else (picks[scope.index].cost_of or 0) * qty.get(scope.index, scope.amount)
            if what == 'compare':
                other: Any = self.item_named(c.get('other'))
                if spend is not None and other is not None and other.index in picks and other.index not in drop:
                    theirs: float = (picks[other.index].cost_of or 0) * qty.get(other.index, other.amount)
                    if not self.holds(spend, c.get('op') or 'at most', (val if val is not None else 1.0) * theirs):
                        return False
                continue
            if what == 'total' and val is not None:
                if not self.holds(total if scope is None else spend or 0, c.get('op') or 'at most', val):
                    return False
            elif what in ('same', 'distinct'):
                vals: list = [v for v in (self.listing_field(l, c.get('field')) for it, l in live) if v not in (None, '')]
                if what == 'same' and len(set(vals)) > 1:
                    return False
                if what == 'distinct' and len(set(vals)) < len(vals):
                    return False
            elif what == 'share' and val is not None and total > 0:
                part: float = val if val <= 1 else val / 100.0
                base: float = self.cap if 'budget' in str(c.get('of') or '').lower() and self.cap else total
                spends: list = [spend] if scope is not None and spend is not None else [(l.cost_of or 0) * qty.get(it.index, it.amount) for it, l in live]
                if not all((self.holds(x, c.get('op') or 'at most', part * base) for x in spends)):
                    return False
        return True

    def solve_basket(self, chosen: dict[int, Offer]) -> Any:
        opts: list = []
        for it in iter(self.items):
            first: Any = chosen.get(it.index)
            cands: list = ([first] if first is not None else []) + [c for c in self.buyable(it) if c is not first]
            if not bool(cands):
                return None
            opts.append(cands[:5])
        combos: list = [[]]
        for o in iter(opts):
            combos = [c + [x] for c in combos for x in o][:4000]
        best: Any = None
        for combo in iter(combos):
            picks: dict = {it.index: l for it, l in zip(self.items, combo)}
            qty: dict = {it.index: it.amount for it in self.items}
            drop: set = set()
            self.apply_adjustments(picks, qty, drop)
            if not self.basket_ok(picks, qty, drop):
                continue
            total: float = sum(((picks[i].cost_of or 0) * qty[i] for i in picks if i not in drop))
            if self.cap is not None and total > self.cap + 1e-06:
                continue
            rank: int = sum((o.index(l) for o, l in zip(opts, combo)))
            key: Any = (total, rank) if self.cheapest else (rank, total)
            if best is None or key < best[0]:
                best = (key, picks, qty, drop)
        return None if best is None else best[1:]

    def preorder_check(self) -> bool:
        lines: list = []
        for it in iter(self.items):
            l: Any = it.chosen
            lines.append(dict(item=it.name, quantity=self.basket2.get(l.ident, it.amount) if l is not None else 0, listing=None if l is None else dict(id=l.ident, title=l.title[:100], price=l.cost_of, seller=l.seller or None, brand=l.brand or None, properties=dict(list(l.data_points.items())[:10]))))
        user: str = ''.join(("REQUEST (the shopper's own words):\n", self.ask2, '\n\nRULES WITH NO AUTOMATIC CHECK:\n- ', '\n- '.join(self.other_rules),
                             '\n\nBASKET ABOUT TO BE ORDERED:\n', _compact(lines, 4000), '\n\nLISTINGS SEEN FOR EACH PRODUCT LINE:\n', _compact(self.director_view(6), 9000)))
        answer: Any = self.llm.json([SCHEME_LLM, FAST_LLM] if self.time_remaining() > 25 else [FAST_LLM], PREORDER_BRIEF, user, 1200)
        self.remark2('preorder_check', answer=_compact(answer, 300))
        if not isinstance(answer, dict) or answer.get('ok') is not False:
            return False
        if str(answer.get('decision') or '').lower() == 'decline':
            self.refuse('a rule in the request rules out the basket')
            return True
        picks: dict = {}
        for row in iter(answer.get('lines') or []):
            if isinstance(row, dict) and row.get('listing'):
                owners: list = [it for it in self.items if str(row['listing']) in it.pool]
                if bool(owners):
                    owner: Any = max(owners, key=lambda it: _similar(it.name, row.get('item') or ''))
                    picks[owner.index] = owner.pool[str(row['listing'])]
        return bool(picks) and self.execute_basket(picks, str(answer.get('reasoning') or ''))

    def closest_pick(self, item: Article, picks: dict[int, Offer]) -> tuple[Any, set]:
        rule: Any = getattr(self, 'no_fit', {}) or {}
        counted: str = ' '.join((str(x).lower() for x in rule.get('count_as_miss') or []))
        others: float = sum(((picks[it.index].cost_of or 0) * it.amount for it in self.items if it is not item and it.index in picks))
        rows: list = []
        for l in iter(item.pool.values()):
            if not l.unfolded or not self.ok_category(item, l):
                continue
            v: Any = item.rulings.get(l.ident)
            if v is None:
                continue
            need: set = set(item.req_proses())
            short: set = {f for f in v.get('failed') or set() if f in need}
            if 'cart refused' in (v.get('failed') or set()):
                continue
            if l.stock is False:
                if 'stock' not in counted:
                    continue
                short.add('can be bought now')
            price: Any = l.cost_of
            over: bool = price is not None and ((getattr(item, 'unit_cap', None) is not None and price > item.unit_cap + 1e-06) or (self.cap is not None and others + price * item.amount > self.cap + 1e-06))
            if bool(over):
                if 'budget' not in counted and 'price' not in counted:
                    continue
                short.add('within budget')
            rows.append((len(short), l.stock is False, len([u for u in v.get('unsure') or () if u in need]) + len(need - set(item.for_judged.get(l.ident, set()))), price if price is not None else 1e+18, l, short))
        if not bool(rows):
            return (None, set())
        fewest: int = min((r[0] for r in rows))
        top: list = [r for r in rows if r[0] == fewest]
        tie: str = str(rule.get('tie_break') or '').lower()
        if 'dear' in tie or 'expens' in tie or 'high' in tie:
            pick: Any = max(top, key=lambda r: (not r[1], r[3] if r[3] < 1e+18 else -1))
        else:
            pick = min(top, key=lambda r: (r[1], r[3], r[2]))
        return (pick[4], pick[5])

    def execute_basket(self, picks: dict[int, Offer], note: str) -> bool:
        for item in iter(self.items):
            item.chosen = picks.get(item.index)
        for lid in iter(list(self.basket2)):
            if not any((it.chosen is not None and it.chosen.ident == lid for it in self.items)):
                self.basket_remove(lid)
        for item in iter(self.items):
            if item.chosen is not None and item.chosen.ident not in self.basket2:
                if self.moves_remaining() < 2 or not self.basket_append(item, item.chosen):
                    item.chosen = None
        if not bool(self.basket2):
            return False
        return bool(self.close_purchase(note=note)) or bool(self.done2)

    def act_on_no_fit(self, chosen: dict[int, Offer], missing: list[Article]) -> bool:
        action: str = str((getattr(self, 'no_fit', {}) or {}).get('action') or '').lower()
        if action not in ('order_closest', 'order_fitting_lines_only') or getattr(self, 'no_fit_rounds', 0) >= 2 or self.moves_remaining() < 2:
            return False
        self.no_fit_rounds = getattr(self, 'no_fit_rounds', 0) + 1
        picks: dict[int, Offer] = dict(chosen)

        def unjudged(it: Article) -> list:
            need: set = set(it.req_proses())
            return [l for l in it.pool.values() if l.unfolded and self.ok_category(it, l) and (not it.for_judged.get(l.ident, set()) >= need)][:10]
        late: list = [(it, unjudged(it)) for it in missing]
        late = [(it, ls) for it, ls in late if ls]
        if late and self.time_remaining() > 12:
            _parallel([lambda it=it, ls=ls: self.judge(it, ls, force=True) for it, ls in late])
            self.remark2('late_judged', lines={it.name: len(ls) for it, ls in late})
        said: list = []
        if action == 'order_closest':
            host_of: dict = {a.index: h for a, h in self.match_pairs()}
            for item in iter(sorted(missing, key=lambda it: (it.index in host_of, it.index))):
                host: Any = host_of.get(item.index)
                if host is not None and host.index in picks and self.time_remaining() > 20:
                    self.bind_match(item, host, picks[host.index])
                    seen: list = [l for l in item.pool.values() if l.unfolded and self.ok_category(item, l)]
                    if bool(seen):
                        self.judge(item, seen[:10], force=True)
                pick, short = self.closest_pick(item, picks)
                if pick is not None:
                    picks[item.index] = pick
                    said.append(''.join((item.name, ': ', pick.title[:70], ' (falls short on: ', ', '.join(sorted(short))[:140] or 'nothing confirmed', ')')))
        if not bool(picks):
            return False
        left: list = [it.name for it in self.items if it.index not in picks]
        note: str = ''.join(('No listing meets every requirement for ', ', '.join((it.name for it in missing)), '. ',
                             'Ordered the closest instead: ' + '; '.join(said) + '. ' if said else '',
                             'Left out: ' + ', '.join(left) + '.' if left else ''))
        self.remark2('no_fit_acted', action=action, picks={str(k): v.ident for k, v in picks.items()}, left=left)
        if bool(self.execute_basket(picks, note.strip())):
            return True
        if not self.done2 and self.handle_rows():
            self.select_and_close()
        return bool(self.done2)

    def refuse(self, reason: str='') -> bool:
        if not getattr(self, 'in_director', False):
            self.in_director = True
            try:
                if bool(self.director_ending(reason)):
                    return self.done2
            finally:
                self.in_director = False
        if bool(self.done2):
            return True
        spec: Any = self.jobs2.get('decline')
        if not bool(spec):
            return False
        if self.routine and (not getattr(self, 'refuse_gated', False)):
            self.refuse_gated = True
            self.door('before_decline', 'close the task with no purchase')
        args: dict[str, Any] = self.pinned_params('decline')
        remark: Any = spec.get('remark')
        note: str = getattr(self, 'director_note', '') or "No listing on offer satisfies all of the request's requirements."
        note = self.write_note(False, note)
        if remark and (remark in (self._spec('decline').get('required') or []) or getattr(self, 'director_note', '') or getattr(self, 'closing_note', '')) and remark in (self._spec('decline').get('properties') or {}):
            args[remark] = self._trim('decline', remark, note)
        res: Any = self.invoke(spec['name'], args)
        self.remark2('declined', reason=reason or self.refuse_reason)
        if not self.done2 and (not getattr(self, 'refuse_rejected', False)) and (self.moves_remaining() >= 1):
            self.refuse_rejected = True
            self.remark2('decline_refused', obs=_compact(res['obs'], 200), lines=res['lines'][:3])
            if self.handle_rows() and self.moves_remaining() >= len(self.items) + 2:
                self.select_and_close()
            if not bool(self.done2):
                self.invoke(spec['name'], args)
        _rv163: Any = self.done2
        return _rv163

    def select_and_close(self) -> None:
        self.handle_rows()
        self.all_judge()
        chosen: dict[int, Offer] = {}
        pairs: Any = self.match_pairs()
        hosts: set = set()
        for _it166, _it167 in pairs:
            hosts.add(_it167.index)
        accessories: dict = {}
        for _it168, _it169 in pairs:
            accessories[_it168.index] = _it169
        conditions: Any = getattr(self, 'conditions', [])
        sources: set = set()
        for _it170 in conditions:
            sources.add(_it170['src'].index)
        dependents: set = set()
        for _it172 in conditions:
            for _it171, _it173, _it171 in _it172['branches']:
                dependents.add(_it173.index)
        tops: list = []
        for _it174 in self.items:
            if _it174.index not in accessories and _it174.index not in dependents:
                tops.append((_it174, (self.best_rank(_it174, self.buyable(_it174)) or [None])[0]))
        _parallel([lambda it=it, c=c: self.confirm(it, c) for it, c in tops if c is not None])
        for item in iter(sorted(self.items, key=lambda it: (it.index in accessories or it.index in dependents, it.index not in hosts and it.index not in sources, it.index))):
            if item.index in accessories and accessories[item.index].index in chosen:
                self.bind_match(item, accessories[item.index], chosen[accessories[item.index].index])
            if item.index in dependents:
                self.conditions_apply(item, chosen)
            tries: int = 0
            for cand in iter(self.best_rank(item, self.buyable(item))):
                if tries >= 3 or self.time_remaining() < CONCLUDE_SECS:
                    break
                tries += 1
                if bool(self.confirm(item, cand)):
                    chosen[item.index] = cand
                    break
            if item.index not in chosen:
                rest: Any = self.best_rank(item, self.buyable(item))
                more: bool = not getattr(self, 'final_chance', False) and self.time_remaining() > CONCLUDE_SECS + 20 and self.moves_remaining() - self.end_keep_back() >= 3
                if rest and (tries >= 3 or self.time_remaining() < CONCLUDE_SECS) and (not more):
                    chosen[item.index] = rest[0]
                elif rest and tries >= 3:
                    item.gate_failed = [l.ident for l in rest[:3]]
            if item.index in accessories and item.index not in chosen:
                host: Any = accessories[item.index]
                for other in iter(self.buyable(host)[:3]):
                    if chosen.get(host.index) is other or self.time_remaining() < CONCLUDE_SECS:
                        continue
                    self.bind_match(item, host, other)
                    got: Any = self.buyable(item)
                    if bool(got):
                        chosen[host.index] = other
                        chosen[item.index] = got[0]
                        break
            if item.index in dependents and item.index not in chosen:
                for cond in iter([c for c in conditions if c['src'].index in chosen and any((d is item for _, d, _ in c['branches']))]):
                    src: Any = cond['src']
                    if any((h is src and a.index in chosen for a, h in pairs)) or self.time_remaining() < CONCLUDE_SECS:
                        continue
                    now: Any = self.holds_condition(cond, chosen[src.index])
                    for other in iter(self.buyable(src)[:4]):
                        if other is chosen[src.index] or self.holds_condition(cond, other) == now:
                            continue
                        was: Any = chosen[src.index]
                        chosen[src.index] = other
                        self.conditions_apply(item, chosen)
                        got: Any = self.buyable(item)
                        if got and self.confirm(src, other):
                            chosen[item.index] = got[0]
                            break
                        chosen[src.index] = was
                        self.conditions_apply(item, chosen)
                    if item.index in chosen:
                        break
        missing: list = []
        for _it175 in self.items:
            if _it175.index not in chosen:
                missing.append(_it175)
        if missing and (not getattr(self, 'final_chance', False)) and (self.time_remaining() > CONCLUDE_SECS + 20) and (not self.spent()):
            self.final_chance = True
            self.keep_back_override = len(self.items) + 3 + max(self.parked_notes(), 1 if self.confirm_initial else 0)
            try:
                for it in iter(missing):
                    if self.moves_remaining() - self.end_keep_back() >= 3:
                        it.exhausted = False
                        self.explore(it, self.moves_remaining() - self.end_keep_back(), final=True)
            finally:
                self.keep_back_override = None
            if any((self.buyable(it) for it in missing)):
                self.select_and_close()
                return
        if missing and (not getattr(self, 'audited', False)) and (self.time_remaining() > CONCLUDE_SECS + 25) and (not self.spent()):
            self.audited = True
            if self.inspection(missing) and all((self.buyable(it) for it in missing)):
                self.select_and_close()
                return
        if missing and self.act_on_no_fit(chosen, missing):
            return
        if bool(missing):
            self.refuse_reason = 'nothing that fits was found for ' + ', '.join((it.name for it in missing))
            self.remark2('no_fit', items=[it.name for it in missing], pools={it.name: len(it.pool) for it in missing})
            self.refuse(self.refuse_reason)
            return
        if getattr(self, 'basket_checks', []) or getattr(self, 'adjustments', []):
            solved: Any = self.solve_basket(chosen)
            if solved is None:
                self.refuse_reason = 'no combination of the listings found meets the rules about the whole basket'
                self.remark2('basket_unsolved', checks=self.basket_checks, adjustments=self.adjustments)
                self.refuse(self.refuse_reason)
                return
            picks, qty, drop = solved
            for it in iter(self.items):
                it.amount = qty.get(it.index, it.amount)
            self.remark2('basket_solved', picks={str(k): v.ident for k, v in picks.items()}, qty={str(k): v for k, v in qty.items()}, drop=sorted(drop))
            if bool(self.execute_basket({k: v for k, v in picks.items() if k not in drop}, '')):
                return
            if not self.done2 and self.handle_rows() and getattr(self, 'solve_rounds', 0) < 2:
                self.solve_rounds = getattr(self, 'solve_rounds', 0) + 1
                self.select_and_close()
            return
        if self.cap is not None:

            def total() -> float:
                _rv164: float = sum(((chosen[it.index].cost_of or 0) * it.amount for it in self.items))
                return _rv164
            guard: int = 0
            while total() > self.cap + 1e-06 and guard < 10:
                guard += 1
                moved: bool = False
                for it in iter(sorted(self.items, key=lambda i: -((chosen[i.index].cost_of or 0) * i.amount))):
                    fits: list = []
                    for _it165 in it.matches(cheapest=True):
                        if _it165.stock is not False:
                            fits.append(_it165)
                    pos: Any = [f.ident for f in fits].index(chosen[it.index].ident) if chosen[it.index].ident in [f.ident for f in fits] else -1
                    if 0 <= pos < len(fits) - 1:
                        chosen[it.index] = fits[pos + 1]
                        moved: bool = True
                        break
                if not bool(moved):
                    break
            if total() > self.cap + 1e-06:
                self.refuse_reason = 'even the cheapest fitting combination costs more than the budget'
                self.remark2('over_budget', total=total(), cap=self.cap)
                self.refuse(self.refuse_reason)
                return
        for item in iter(self.items):
            item.chosen = chosen[item.index]
        self.place_purchase()

    def set_basket(self, lines: dict[str, int]) -> bool:
        spec: Any = self.jobs2.get('cart_set') or {}
        props: Any = self._spec('cart_set').get('properties') or {}
        line_props: Any = ((props.get(spec.get('lines') or '') or {}).get('items') or {}).get('properties') or {}
        maximum: Any = (line_props.get(spec.get('qty') or '') or {}).get('maximum')
        rows: list = []
        for lid, qty in iter(lines.items()):
            row: dict = {spec['id']: lid}
            if bool(spec.get('qty')):
                row[spec['qty']] = min(qty, maximum) if isinstance(maximum, int) else qty
            rows.append(row)
        res: Any = self.invoke(spec['name'], {spec['lines']: rows})
        if not bool(res['ok']):
            self.remark2('cart_set_failed', error=res['error'], obs=_compact(res['obs'], 200))
        _rv176: Any = res['ok']
        return _rv176

class Session(Wiring, Groundwork, Hunting, Weighing, Dialogue, Tracing, Checkout):

    def __init__(self, problem_data: dict[str, Any]):
        self.started = time.monotonic()
        environment: Any = problem_data['environment']
        self.link = dict(environment['binding'])
        self.terms_view = environment['policy_view']
        self.ask2 = str(self.terms_view.get('query') or '')
        self.opening = self.terms_view.get('opening')
        self.instruments = [t for t in self.terms_view.get('tools') or [] if isinstance(t, dict)]
        self.specs: dict[str, dict[str, Any]] = {}
        for t in iter(self.instruments):
            fn: Any = t.get('function') if isinstance(t.get('function'), dict) else t
            if isinstance(fn, dict) and fn.get('name'):
                self.specs[fn['name']] = fn.get('parameters') or dict(type='object', properties={})
        self.top_moves = int(self.terms_view.get('max_steps') or 30)
        self.move = 0
        self.done2 = bool(self.terms_view.get('finished'))
        self.dead = False
        self.log: list[dict[str, Any]] = []
        self.llm = LLM(self)
        self.jobs2: dict[str, Any] = {}
        self.items: list[Article] = []
        self.cap: float | None = None
        self.cheapest = False
        self.best: str | None = None
        self.confirm_initial = False
        self.routine: list[dict[str, Any]] = []
        self.past: list[dict[str, Any]] = []
        self.door_invokes: dict[str, int] = {}
        self.sig_confirmed: tuple | None = None
        self.note_allowance: int | None = None
        self.notes_dispatched = 0
        self.in_stock_solely = True
        self.query_about: list[str] = []
        self.match_rules: list[dict[str, Any]] = []
        self.pending_rows: list[str] = []
        self.handled_rows: set[str] = set()
        self.basket2: dict[str, int] = {}
        self.basket_article: dict[str, Article] = {}
        self.maps: dict[str, dict[str, str]] = {}
        self.compare_cap = 4
        self.leaf_extent = 0
        self.world_label = '|'.join(sorted(self.specs))
        memo: Any = _MEMORY_WORLD.get(self.world_label) or {}
        self.maps = dict(memo.get('maps') or {})
        self.compare_cap = int(memo.get('compare_limit') or 4)
        self.compare_thin = bool(memo.get('thin_compare'))
        self.unfolded_leaves: set[str] = set()
        self.refuse_reason = ''
        self.mode = 'full'

    def often_kept(self, item: Article, taken: Any) -> list[str]:
        menu: list = []
        for _it177 in self.trait_catalogue(item):
            if _it177['listings'] >= 2:
                menu.append(_it177['property'])
        out: list = []
        for kind in iter((('material',), ('pattern',), ('color', 'colour'))):
            hit: Any = next((m for m in menu if m in kind), None) or next((m for m in menu if set(m.split()) & set(kind)), None)
            if hit and (not taken(item, hit)) and (hit not in out):
                out.append(hit)
        return out

    def drive(self) -> list[dict[str, Any]]:
        try:
            if not bool(self.setup()):
                self.force_close()
                _rv178: Any = self.log
                return _rv178
            self.work()
        except Exception as exc:
            self.remark2('exception', error=repr(exc)[:300])
        try:
            if not bool(self.done2):
                self.force_close()
        except Exception as exc:
            self.remark2('exception', error=repr(exc)[:300])
        self.remark2('done', cost=round(self.llm.price_tag, 4), llm_calls=self.llm.invokes, steps=self.move)
        _rv179: Any = self.log
        return _rv179

    def work(self) -> None:
        items: Any = self.items
        n: int = len(items)
        questions: Any = (self.note_allowance if self.note_allowance is not None else 3) - self.parked_notes()
        if 'message' not in self.jobs2 or questions < 0:
            questions: int = 0
        self.floor_ref = questions + (1 if questions else 0) + self.end_keep_back() + 2 * n
        for item in iter(items):
            for req in iter(list(item.requirements)):
                if req.reference and req.op in OPS_REF:
                    self.reference_resolve(item, req)
        self.floor_ref = 0
        free: Any = self.moves_remaining() - questions - (1 if questions else 0) - self.end_keep_back()
        per_item: Any = max(2, free // max(n, 1)) if free > 0 else 0
        if any((st['moment'] == 'start' for st in self.routine)):
            self.door('start', 'begin the task (start searching the store)')
            self.handle_rows()
            questions: Any = max(0, min(questions, self.note_allowance - self.notes_dispatched - self.parked_notes() if self.note_allowance is not None else questions))
        for k, item in enumerate(items):
            share: Any = per_item if k < n - 1 else max(per_item, self.moves_remaining() - questions - 1 - self.end_keep_back())
            if share <= 0:
                break
            before: Any = self.moves_remaining()
            self.explore(item, max(2, share // 2))
            self.up_warm(item, max(0, share - (before - self.moves_remaining())))
        if questions > 0 and self.can_query():
            plan: Any = self.scheme_queries(questions)
            while plan:
                item, prop = plan.pop(0)
                if not self.can_query() or self.moves_remaining() <= self.end_keep_back():
                    break
                if any((_similar(r.prop, prop) >= 0.67 for r in item.requirements)):
                    continue
                before: int = len(item.requirements)
                answer: Any = self.query2(item, prop)
                if isinstance(answer, dict) and str(answer.get('kind') or '').lower() == 'split' and self.can_query() and (self.moves_remaining() > self.end_keep_back()):
                    answer: Any = self.query2(item, prop, narrow=True)
                self.apply_response_kept(item, prop, answer)
                self.handle_rows()
                if len(item.requirements) != before and plan:
                    left: Any = min(len(plan), self.note_allowance - self.notes_dispatched - self.parked_notes() if self.note_allowance is not None else len(plan))
                    plan: Any = self.scheme_queries(left)
            while self.can_query() and self.moves_remaining() > self.end_keep_back() + 1 and (self.time_remaining() > CONCLUDE_SECS + 30):
                more: Any = self.scheme_queries(1)
                if not bool(more):
                    break
                item, prop = more[0]
                answer: Any = self.query2(item, prop)
                self.apply_response_kept(item, prop, answer)
                self.handle_rows()
        for item in iter(items):
            for req in iter(list(item.requirements)):
                if req.reference and req.op in OPS_REF:
                    self.floor_ref = self.end_keep_back() + 2
                    self.reference_resolve(item, req)
                    self.floor_ref = 0
        if self.mode == 'ask':
            return
        self.all_judge()
        order: list = sorted(items, key=lambda it: (bool(it.matches()), -len(it.pool)))
        for k, item in enumerate(order):
            share: Any = max(2, (self.moves_remaining() - self.end_keep_back()) // max(len(order) - k, 1))
            self.explore(item, share, final=True)
        self.select_and_close()

    def kept_cart_holds(self) -> bool:
        if not self.basket2 or not self.items or self.moves_remaining() < 1:
            return False
        lines: dict = {}
        for _it180 in self.basket2:
            lines[_it180] = self.basket_article.get(_it180)
        if {it.index for it in lines.values() if it} != {it.index for it in self.items}:
            return False
        total: float = 0.0
        for lid, item in iter(lines.items()):
            listing: Any = item.pool.get(lid) if item else None
            if listing is None or listing.stock is False or (item.rulings.get(lid) or {}).get('failed') or (lid not in item.rulings):
                return False
            total += (listing.cost_of or 0) * self.basket2[lid]
        _rv181: Any = self.cap is None or total <= self.cap + 1e-06
        return _rv181

def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    _rv182: Any = Session(problem_data).drive()
    return _rv182
__all__ = ['agent_main']
