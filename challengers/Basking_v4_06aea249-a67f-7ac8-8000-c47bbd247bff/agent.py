"""Sortie: a shopping assistant for ORO store worlds in which code leads and models advise.

One sortie carries one task. It reads the request and the tool list once and assigns tools to jobs, hunts the store by
keywords and by required values, opens postings, rules on them property by property, asks the shopper about the
properties most likely to carry an unwritten stipulation, chases stipulations that point at other listings, and ends
with a purchase, the nearest posting or a decline, whichever the request permits.
"""
from __future__ import annotations
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from src.agent.proxy_client import ProxyClient
INTERPRET_ENGINE = 'anthropic/claude-sonnet-5.5'
BLUEPRINT_ENGINE = 'anthropic/claude-sonnet-5.5'
VALIDATE_ENGINE = 'anthropic/claude-sonnet-5.5'
QUICK_ENGINE = 'anthropic/claude-haiku-5.5'
STANDBY_ENGINE = 'openai/gpt-6-luna'
ASSIGNMENT_SECS = 255.0
ENFOLD_SECS = 45.0
DATUM_GLYPHS = 2200
LINE_GLYPHS = 260
MILD_OUTLAY = 0.03
OUTLAY_LID = 0.05
_TOKEN = re.compile('\\b[A-Za-z]{2,}-?\\d{3,}[A-Za-z]?\\b|\\b[A-Za-z]{2,}\\d*-\\d{2,}\\b')
_RANGE = re.compile('\\b([A-Z]{1,5}\\d{1,3})-\\d{2,}\\b')
_INDICATOR = re.compile('\\b(?:more|details?|info(?:rmation)?|see|read|look)\\b[^.;:!?]{0,24}?\\b(?:part|model|item|listing|product|page|code)\\s*(?:no\\.?|number|#)?\\s*([A-Za-z0-9][A-Za-z0-9-]{2,})', re.I)
ALLUSION_OPERATORS = {'same as reference': 'is', 'different from reference': 'is not', 'counterpart of reference': 'is', 'at least reference': 'at least', 'at most reference': 'at most', 'within reference': 'at least'}
_model = ProxyClient(timeout=75, max_retries=1, retry_delay=1)
_habitat = ProxyClient(timeout=90, max_retries=2, retry_delay=1)
_NON_PETITIONED = {'option', 'variant'}
_SPHERE_REMEMBRANCE: dict[str, Any] = {}

class _Text:
    """Text normalisation and comparison."""

    @staticmethod
    def _is_statement(worth: Any) -> bool:
        return isinstance(worth, str) and len(worth) > 24 and (' ' in worth.strip())

    @staticmethod
    def _digit(worth: Any) -> float | None:
        if isinstance(worth, bool):
            return None
        if isinstance(worth, (int, float)):
            return float(worth)
        if isinstance(worth, str):
            mm = re.search('-?\\d+(?:\\.\\d+)?', worth.replace(',', ''))
            if mm:
                try:
                    return float(mm.group(0))
                except Exception:
                    return None
        return None

    @staticmethod
    def _dense(worth: Any, boundary: int) -> str:
        wording = json.dumps(worth, ensure_ascii=False, separators=(',', ':'))
        return wording[:boundary] + '…' if not len(wording) <= boundary else wording

    @staticmethod
    def _akin(xa: Any, xb: Any) -> float:
        (tok_a, tok_b) = (_Text._counters(xa), _Text._counters(xb))
        if not tok_a or not tok_b:
            return 0.0
        return len(tok_a & tok_b) / len(tok_a | tok_b)

    @staticmethod
    def _nigh(xa: Any, xb: Any) -> bool:
        if _Text._normalize(xa) == _Text._normalize(xb):
            return True
        (tok_a, tok_b) = (_Text._counters(xa), _Text._counters(xb))
        return bool(tok_a) and tok_b == tok_a

    @staticmethod
    def _bases(wording: Any) -> set[str]:
        return {tl[:4] for tl in _Text._counters(wording)}

    @staticmethod
    def _vocables(wording: Any) -> set[str]:
        return {tl if not (len(tl) > 3 and tl.endswith('s')) else tl[:-1] for tl in _Text._normalize(wording).split()}

    @staticmethod
    def _counters(wording: Any) -> set[str]:
        away = set()
        for tl in _Text._normalize(wording).split():
            if len(tl) > 1:
                if len(tl) > 3 and tl.endswith('s'):
                    tl = tl[:-1]
                away.add(tl)
        return away

    @staticmethod
    def _normalize(wording: Any) -> str:
        wording = str(wording or '').lower().replace('&', ' and ').replace('_', ' ').replace('-', ' ')
        return re.sub('[^a-z0-9 ]+', ' ', wording).strip()

class _Schema:
    """Fitting values to argument schemas."""

    @staticmethod
    def _operand_standard(details: dict[str, Any]) -> Any:
        genus = details.get('type')
        if isinstance(genus, list):
            genus = next((kk for kk in genus if 'null' != kk), None)
        if isinstance(details.get('enum'), list) and details['enum']:
            return details['enum'][0]
        if genus in ('integer', 'number'):
            return _Schema._conform_worth(details, 1)
        if 'boolean' == genus:
            return False
        if 'string' == genus:
            return _Schema._conform_worth(details, 'n/a')
        return _LETGO if not ('array' == genus and (not details.get('minItems'))) else []

    @staticmethod
    def _conform_worth(details: dict[str, Any], worth: Any) -> Any:
        genus = details.get('type')
        if isinstance(genus, list):
            genus = next((kk for kk in genus if 'null' != kk), None)
        enum = details.get('enum')
        if isinstance(enum, list) and enum:
            if worth in enum:
                return worth
            through = {_Text._normalize(err): err for err in enum if isinstance(err, str)}
            if isinstance(worth, str) and through:
                finest = max(through, key=lambda err: _Text._akin(err, worth))
                if _Text._normalize(worth) in through:
                    return through[_Text._normalize(worth)]
                if _Text._akin(finest, worth) >= 0.5:
                    return through[finest]
            return _LETGO
        if genus in ('integer', 'number'):
            cnt = _Text._digit(worth)
            if cnt is None:
                return _LETGO
            (bottom_i, top_i) = (details.get('minimum'), details.get('maximum'))
            cnt = cnt if not isinstance(bottom_i, (int, float)) else max(cnt, bottom_i)
            cnt = cnt if not isinstance(top_i, (int, float)) else min(cnt, top_i)
            return cnt if not 'integer' == genus else int(round(cnt))
        if 'string' == genus:
            worth = (str(worth) if not (isinstance(worth, list) and worth) else str(worth[0])) if not isinstance(worth, str) else worth
            boundary = details.get('maxLength')
            if isinstance(boundary, int) and len(worth) > boundary:
                worth = worth[:boundary] if not ' ' in worth[:boundary] else worth[:boundary].rsplit(' ', 1)[0]
            return worth if not (isinstance(details.get('minLength'), int) and len(worth) < details['minLength']) else _LETGO
        if 'boolean' == genus:
            if isinstance(worth, bool):
                return worth
            return {'true': True, 'false': False}.get(str(worth).strip().lower(), _LETGO)
        if 'array' == genus:
            worth = [worth] if not isinstance(worth, list) else worth
            return worth if not isinstance(details.get('maxItems'), int) else worth[:details['maxItems']]
        return worth

class _Shape:
    """Reading the shape of model replies and store observations."""

    @staticmethod
    def _unbox(observation: Any) -> tuple[Any, list[str]]:
        strands: list[str] = []
        if isinstance(observation, str):
            return (observation, strands)
        if not isinstance(observation, dict):
            return (observation, strands)
        inside: dict[str, Any] = {}
        lookup_brethren = sum((1 for vv in observation.values() if isinstance(vv, dict)))
        ledger = any((isinstance(vv, str) and vv.strip() and (' ' not in vv.strip()) for vv in observation.values()))
        for (kk, vv) in observation.items():
            if isinstance(vv, list) and vv and all((_Text._is_statement(xx) for xx in vv)):
                strands.extend((str(xx) for xx in vv))
            elif not (_Text._is_statement(vv) and 1 == lookup_brethren and (len(observation) <= 3) and (not ledger)):
                inside[kk] = vv
            else:
                strands.append(str(vv))
        if strands and len(inside) == 1 and isinstance(list(inside.values())[0], dict):
            inside = list(inside.values())[0]
        return (inside, strands)

    @staticmethod
    def _discover_sequence_of_lookups(observation: Any, deepness: int=0) -> list[dict[str, Any]] | None:
        if isinstance(observation, list) and observation and all((isinstance(stp, dict) for stp in observation)):
            return observation
        if isinstance(observation, dict) and 2 > deepness:
            finest = None
            for vv in observation.values():
                discovered = _Shape._discover_sequence_of_lookups(vv, deepness + 1)
                if discovered and (finest is None or len(discovered) > len(finest)):
                    finest = discovered
            return finest
        return None

    @staticmethod
    def _allusion_imprint(allusion: Any) -> tuple:
        if not isinstance(allusion, dict):
            return ()
        qualities = {} if not isinstance(allusion.get('attributes'), dict) else allusion.get('attributes')
        return (_Text._normalize(allusion.get('seller')), _Text._normalize(allusion.get('category')), tuple(sorted((_Text._normalize(vv) for vv in qualities.values() if vv))))

    @staticmethod
    def _vetoed_trolley(observation: Any) -> bool:
        if not isinstance(observation, dict) or len(observation) != 1:
            return False
        just = next(iter(observation.values()))
        return isinstance(just, str) and (not re.search('\\b(ok|okay|success\\w*|added|done|accepted|applied|updated|in (the |your )?(cart|basket))\\b', just, re.I))

    @staticmethod
    def _json_inside(wording: str) -> Any:
        if not wording:
            return None
        wording = wording.strip()
        barrier = re.search('```(?:json)?\\s*(.*?)```', wording, re.S)
        if barrier:
            wording = barrier.group(1).strip()
        for (unfolder, tighter) in (('{', '}'), ('[', ']')):
            start = wording.find(unfolder)
            if 0 > start:
                continue
            deepness = 0
            inside_string = False
            quoted = False
            for idx in range(start, len(wording)):
                chr_ = wording[idx]
                if inside_string:
                    if quoted:
                        quoted = False
                    elif '\\' == chr_:
                        quoted = True
                    elif '"' == chr_:
                        inside_string = False
                    continue
                if '"' == chr_:
                    inside_string = True
                elif unfolder == chr_:
                    deepness += 1
                elif tighter == chr_:
                    deepness -= 1
                    if 0 == deepness:
                        try:
                            return json.loads(wording[start:idx + 1])
                        except Exception:
                            break
        try:
            return json.loads(wording)
        except Exception:
            return None

class _Chronometer:
    """Elapsed seconds of the task, tallied by a background thread in short ticks rather than read from a clock."""
    TICK = 0.2

    def __init__(self) -> None:
        self._ticks = 0
        self._over = threading.Event()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        while not self._over.wait(self.TICK):
            self._ticks += 1

    @property
    def elapsed(self) -> float:
        return self._ticks * self.TICK

    def halt(self) -> None:
        self._over.set()

    @staticmethod
    def rest(seconds: float) -> None:
        threading.Event().wait(seconds)


class _Pool:
    """Running independent model calls side by side."""

    @staticmethod
    def _secure(chore: Any) -> Any:
        try:
            return chore()
        except Exception:
            return None

    @staticmethod
    def _concurrent(chores: list, labourers: int=3) -> list:
        if len(chores) <= 1:
            return [_Pool._secure(jdx) for jdx in chores]
        with ThreadPoolExecutor(max_workers=min(labourers, len(chores))) as reservoir:
            return list(reservoir.map(_Pool._secure, chores))
_LETGO = object()

class Seer:

    def __init__(self, sortie: 'Episode'):
        self.sortie_ref = sortie
        self.expense = 0.0
        self.dispatches = 0
        self.breakdowns = 0

    def converse(self, machine: str, framework: str, client: str, upper_counters: int) -> str | None:
        if MILD_OUTLAY <= self.expense and machine in (BLUEPRINT_ENGINE, VALIDATE_ENGINE, INTERPRET_ENGINE) and (QUICK_ENGINE != machine):
            machine = QUICK_ENGINE
        leftover = self.sortie_ref.duration_leftover()
        if leftover < ENFOLD_SECS - 12:
            return None
        _model.timeout = int(max(10, min(45, leftover - ENFOLD_SECS + 20)))
        if not machine.startswith('anthropic/'):
            core_matter = framework
        else:
            core_matter: Any = [{'type': 'text', 'text': framework, 'cache_control': {'type': 'ephemeral'}}]
        main_text = {'model': machine, 'messages': [{'role': 'system', 'content': core_matter}, {'role': 'user', 'content': client}], 'temperature': 0, 'max_tokens': upper_counters}
        self.dispatches += 1
        consequence = _model.post_verbose('/inference/chat/completions', main_text)
        for essay in range(2):
            error = {} if not isinstance(consequence.error, dict) else consequence.error
            if isinstance(consequence.data, dict) and consequence.data.get('choices'):
                break
            if error.get('status') not in (429, 500, 502, 503, 504) or self.sortie_ref.duration_leftover() < ENFOLD_SECS + 10:
                break
            _Chronometer.rest(2.0 * (essay + 1))
            consequence = _model.post_verbose('/inference/chat/completions', main_text)
        data = consequence.data
        if isinstance(data, dict) and data.get('choices'):
            consumption = data.get('usage') or {}
            if isinstance(consumption.get('cost'), (int, float)):
                self.expense += float(consumption['cost'])
            missive = data['choices'][0].get('message') or {}
            matter = missive.get('content')
            if isinstance(matter, list):
                matter = ''.join((str(pp.get('text') or '') for pp in matter if isinstance(pp, dict)))
            if matter:
                return str(matter)
        self.breakdowns += 1
        self.sortie_ref.jot('llm_failure', model=machine, error=str(getattr(consequence, 'error', ''))[:200])
        return None

    def json(self, machines: list[str], framework: str, client: str, upper_counters: int) -> Any:
        for machine in machines:
            wording = self.converse(machine, framework, client, upper_counters)
            interpreted = _Shape._json_inside(wording or '')
            if interpreted is not None:
                return interpreted
        return None
UPTAKE_INSTRUCTION = """Setting: a simulated store and a single shopping task. You receive two things, the shopper's request text and the store's tool list (written as OpenAI function schemas), and from them you build the briefing a shopping assistant will work from. Your answer must be exactly ONE JSON object with no prose around it; its shape is shown at the end, after the guidance.

GUIDANCE FOR INTERPRETING THE REQUEST

1. Firm demands only. A requirement is something the request insists on. Wishes (ideally, would be nice) and background from the story (who the product is for, where it is going to live) are NOT requirements: record what such remarks hint at in "hints". Words that merely label what sort of product it is, or its category (the men's, women's, kids' of "men's rain capes"; "gaming" inside a category name), belong in "name"/"category", not in "requirements". By contrast, when the request gives a value to a property it names, phrased like "in the <value> style", like "a <property> of <value>", or like "with <value> as its <property>", you have a genuine requirement on that property, though the value may sound like a product kind; the category remains the product named in the request.

2. Item lists. "2 x lamps where the finish is brass and the height is not short" is a single item: quantity 2, two requirements (finish is brass; height is not short).

3. Pack sizes. If the request states how much one listing holds (N pieces in a pack, set or box; a pack that contains N), that number constrains the size of the pack or set. "quantity" is different: it counts the listings to buy.

4. Restatements. When the request merely spells out what a requirement already implies ("laces only, so no velcro and no buckles"), that is ONE requirement (fastening: laces), not several. Do not create properties for whatever it excludes.

5. Operator wording. Alternatives ("A or B") for one property: value "A | B", op "one of". Exclusions ("not X", "anything but X"): op "is not". "at least N units" becomes op "at least" with value "N units". Every value keeps its unit and its spelling exactly as written.

6. References to another listing. Sometimes a requirement is tied to ANOTHER listing's page, which the request identifies by category, seller and a few attribute values. Pick the op from what the shopper wants relative to that page:
   - the very value the page states: "same as reference";
   - any value but the page's: "different from reference";
   - not the page's value itself but its partner value (what the store or that page pairs with it, or what is compatible with it, suits it or works with it): "counterpart of reference";
   - a value not below, or not above, the page's: "at least reference", or "at most reference";
   - a value permitted to differ from the page's figure by some amount, either way: "within reference", with that amount in "tolerance".
   Describe in "reference" the page the request itself names (category, seller, the attributes it names, and the property to read there), even if that page only forwards the property to yet another listing; the assistant follows such forwarding on its own. When, by the request, more than one listing of that seller qualifies and a selection rule is given (the larger one, the cheaper one, the one having more of something), put that rule in "pick" and that number in "count". "attributes" carries just plain identifying values, and never the comparison. Identifying values like these belong to THAT other listing alone, so they must not turn into requirements for the requested item, nor into words to search for it. Made-up case: "the colour must not be whatever Brinkley's rope listing (length 30 m, material nylon) shows" yields exactly one requirement, {"property": "colour", "value": "", "op": "different from reference", "reference": {"property": "colour", "attributes": {"material": "nylon", "length": "30 m"}, "category": "ropes", "seller": "Brinkley"}}, and nothing at all about length or material.

7. Cross-item conditions. A requirement on one item that applies only once another item turns out a particular way goes into "conditions", never into "requirements" or "procedure". Two shapes occur: "in case X's P ends up V, Y's Q has to be W" fills "then", which applies when X's P is among the values; "Y's Q has to be W, but not if X's P is V" fills "otherwise", which applies when it is not.

8. Category wording. "category" means the product kind. When the request spells out a store category (say "the category is kitchen appliances"), copy those exact words.

9. Money. A total stated in cents gives "total_cap" in cents. All amounts (total_cap and price_cap alike) are in the single money unit the request counts in; when that unit is cents, convert any figure given in dollars or spelled out in words to cents ("twelve dollars forty" is 1240).

10. Tools. Tool names and argument names come verbatim from the schemas. If ONE tool finishes the task both ways, through an argument whose value decides between purchasing the cart and leaving with nothing, both "checkout" and "decline" carry that tool's name, and the "args" of each hold the value of that argument which selects it (spelling as in the schema's own list). Where a tool's description marks it final and says it closes with no purchase, it is "decline"; the tool that checks the cart out is "checkout". For a single search tool whose enum argument toggles between keyword search and field search, list it under "search_text" and under "search_fields" too, each one with its own mode value.

SHAPE OF THE ANSWER (the // remarks explain fields and are not part of the JSON)

{
 "items": [  // an entry for every distinct product the shopper intends to purchase; most requests have just one
   {"name": "the product as a plain noun phrase, named the way the request names it (for an item of a list, in that item's words)",
    "quantity": 1,
    "category": "the requested store category or product kind, in the wording of the request",
    "requirements": [ // each firm demand the request makes on a property of THIS item; price, budget, stock and quantity never belong here
      {"property": "the name a listing would give this property", "op": "at least|is|one of|is not|contains|at most|counterpart of reference|within reference|different from reference|same as reference|at most reference|at least reference",
       "value": "the value, limit or list as the request states it, word for word (with 'one of', alternatives joined by ' | ')",
       "reference": null  // filled only when the op ends in 'reference': {"property": "property to read on the referenced page", "relation": "the phrase the request uses for how the wanted value stands to that page's value", "attributes": {"property": "value", ...}, "seller": "...", "category": "...", "pick": null or how to choose among several (e.g. "whichever is wider"), "count": null or how many listings, per the description, fit, "tolerance": null or, with 'within reference' only, the amount and its unit}
      }],
    "price_cap": null,  // a ceiling on what ONE unit of this item may cost (on its own price, not on the order total), as a bare number in the request's money unit; null if there is none
    "hints": ["names of properties the story suggests the shopper privately cares about (e.g. talk of who is going to wear it hints at an age group or a gender; talk of what sits in a computer hints at the processor); [] when nothing is hinted"],
    "search_terms": ["2-4 search-box queries, most specific first. A listing has to match every word, so every query stays short: one or two value words added to the product noun (a material, a colour, a number with unit, e.g. 5v or 16gb). No property names by themselves, and no values whose only role is to point at a referenced listing. The product noun on its own comes last"]
   }],
 "message_budget": null,            // integer: the number of messages the shopper permits; else null
 "call_budget": null,               // integer: the number of tool calls / steps the request permits; else null
 "total_cap": 12345 or null,        // ceiling on the whole order as a bare number, in the request's money unit (cents if the request talks in cents)
 "cheapest": false,                 // set true only when the lowest price / cheapest / cheapest total among the fitting listings is asked for (a tie-break for near misses when nothing fits is not this)
 "best": null,                      // null, or the measure by which the request wants the best listing
 "in_stock_only": true,             // the request only wants listings that can really be bought
 "ask_about": ["topics that, per the request, the assistant should bring up with the shopper; worded as the request words them; [] if none"],
 "procedure": [                     // demands on HOW the task is done: obtain an approval or confirmation, show or tell the shopper something, use some tool, keep some order of steps. Not product requirements, not budgets, not limits on messages or steps, and not the no-fit rule (say so, offer a near miss), which is what decline_rule / nothing_fits are for. [] if none
   {"goal": "the required outcome in the request's own terms", "moment": "before_decline|start|before_finish|before_order", "tool": "null or the tool's name", "kind": "tool|message", "messages": 1}  // "messages" = how many shopper messages the step uses (0 for a tool step); checking something in the store itself (availability, stock, a value) is a "tool" step (kind "tool", 0 messages), except when the request wants to hear the result
 ],
 "fit_rules": [{"host": "name of the host item", "accessory": "name of the accessory item"}],   // one per rule of the form 'make sure X fits Y'; [] if none
 "conditions": [{"if": {"values": ["..."], "property": "...", "item": "which item"}, "then": {"op": "is not|at least|one of|is|at most", "value": "...", "item": "which item", "property": "..."} or null, "otherwise": null or an object of the then-form}],   // requirements of one item that depend on the outcome for another item; [] if none
 "nothing_fits": null,              // "nearest" if the request, should no listing meet all requirements, wants the closest listing bought, put forward or recommended; "decline" if in that case it wants to end with no purchase; null if it says nothing about it
 "decline_rule": null,              // null, or what the request asks for when nothing fits (e.g. only report it, no substitutes)
 "tools": {
   "search_text": {"query": "keyword argument", "name": "tool name", "mode": null or {"value": "...", "arg": "argument choosing keyword search"},
                   "page_kind": "offset|token|number or null", "page": "null or the argument for moving on to more results (page number, row offset, or token from the last result)"},
   "search_fields": null or {"name": "tool searching by the fields category/brand/price (possibly the same tool in field mode)", "brand": "argument or null", "price_max": "argument or null", "category": "argument or null", "mode": null or {"value": "...", "arg": "..."}, "page": "argument or null", "page_kind": "offset|token|number or null"},
   "open": {"id": "id argument", "name": "tool giving one complete listing"},
   "facts": null or {"id": "argument", "name": "dedicated tool for a listing's property table, where the store offers one"},
   "compare": null or {"ids": "list argument", "name": "tool giving several complete listings"},
   "cart_add": {"qty": "quantity argument or null", "id": "argument", "name": "tool that adds a listing to the cart",
                "mode": null or {"remove": "value meaning remove", "add": "value meaning add", "arg": "argument switching between adding and removing"}},
   "cart_set": null or {"lines": "list argument", "name": "tool that replaces all cart contents with a given list of lines, where the store offers one", "qty": "key of the quantity within a line", "id": "key of the listing id within a line"},
   "cart_view": null or {"name": "tool displaying the cart"},
   "checkout": {"name": "final tool that places the order", "args": null or {"argument": "value"}},
   "decline": {"args": null or {"argument": "value"}, "name": "final tool that ends with no purchase", "remark": "text argument or null"},
   "message": {"text": "text argument", "name": "tool for writing to the shopper"},
   "status": null or {"name": "tool that reports the allowances left"}
 }
}"""
FILTER_INSTRUCTION = """Reply with ONE JSON object only: {"no": [row numbers], "yes": [row numbers]}. A row you put in neither list is treated as "maybe".

Context: a shopping assistant ran a search, and you triage the numbered result rows. Every row shows an id, a title, a short text, a category and a price. For each one, decide whether the listing COULD satisfy the requirements after its full property page gets read.
- Use only what is visible in the row itself.
- The row is silent about some requirement: it could still satisfy it, so "maybe".
- The row's own words plainly contradict a required value: "no".
- The row itself states the required values: "yes".
- The store's category label tells what kind of product each row is. Any row labelled with the requested category counts as the right kind, regardless of the impression its title gives, so the kind of product is never a reason for "no"."""
EVALUATE_INSTRUCTION = """Task: for each store listing supplied, decide from that listing's own records whether it satisfies the shopper's requirements, which are numbered R1, R2, and so on.

Answer with ONE JSON object, no remarks around it:
{"verdicts": [{"id": "...", "failed": ["R numbers clearly contradicted by the records"], "unsure": ["R numbers on which every record is silent"], "weak": ["R numbers backed by nothing but an options field, the description or the title, with neither a property nor a seller note behind them"], "loose": ["R numbers matched only via a synonym or a wider term, not by the very words of the value (e.g. 'female' where 'girls' is required); neither converting units nor a series covering a code makes it loose"]}, ...]}
Give one verdict per listing; omit lists that would have nothing in them. When a listing has neither "failed" nor "unsure", it satisfies all the requirements shown.

Rules:
1. Judge the listed requirements and nothing else; never make up new ones.
2. Where the value comes from. Look at the property table first: if a property that covers the requirement exists, its value settles the matter, and a different value there fails the requirement whatever the title claims. Only if NO property key covers the requirement may a value from the title, the description, the options/variants field or the seller's notes count. If no record states it anywhere, the requirement counts as unaddressed ("unsure"). Values filed under a different but related property are ignored (a handle says nothing of the lock type; a lining's material is not the product's material).
3. When a value matches. A requirement is satisfied if the value appears in the listing's own property for that requirement. Key spellings vary (prefixes, underscores, spaces, abbreviations): any key naming the property counts, and a property may sit under several keys. Matching is about meaning: in a comma-separated list, a single element is enough; plural equals singular; close synonyms and obvious abbreviations count; case is ignored; a store value built from the required value's words plus qualifiers satisfies it, so 'matte black' counts as 'black' and 'faux suede' counts as 'suede'; values with units match after unit conversion (numbers compare numerically, a limit includes its edge value, and a range satisfies a limit only if the entire range does).
4. Operators. A requirement that some feature be absent (value none, no or without) is satisfied when no record mentions the feature. "contains": the value must be among those stated. "one of": any listed value will do. Exclusions ("is not", "different from") fail when the excluded value is what the listing's property states.
5. STORE FIELDS, if the input has them, tell for each required property which keys and values appear in this store's listings of the category that have already been read. Suppose some of them hold exactly the required value in that field: in that case any listing whose field holds anything else fails, broader or related values included (adding qualifiers to the required words does not make it anything else). A value that no read listing has can still be satisfied through a clear synonym.
6. Product kind. The store's category label settles it: when a listing's label is the requested category, the listing is the right kind, never mind its title. Failing as a case, accessory or spare part rather than the product is possible only for a listing that has no label.
7. Compatibility. A requirement to work with another product (given by model names or codes) is satisfied only if one of those names appears in this listing's own text as something it fits or works with. There is a second route: the other product's page may list a series (family) of part codes that are compatible; then a listing satisfies the requirement if its own part code, often shown in brackets in the title, starts with that series. Text claiming that some OTHER part fits a product speaks for the listing whose title bears that part's code, and not for the page holding the text. Such pages are shown in a listing's "related_pages"; if such a page says something about the part code of this very listing, that statement counts for this listing. A bare link to another page proves nothing.
8. Redirected properties. Seller notes may hand one of the listing's properties to ANOTHER listing (go by, or take, another listing's value for it). In that case disregard the value this listing gives itself for that property: any requirement on that property becomes "unsure", except where a related page displayed along with the listing gives the value. Redirecting a property that no requirement mentions changes nothing."""
BLUEPRINT_INSTRUCTION = """Return JSON in exactly this form, with QUESTION_COUNT entries ordered best first:
{"questions": [{"item": "the item's name exactly as you were given it", "property": "the property name in the menu's exact spelling (space-separated words)"}, ...]}

Background: a shopping assistant is about to message a shopper, and you decide what it asks. Each hidden requirement of the shopper is a single value of ONE property of one requested product. The shopper discloses it only when a message asks about precisely that property (by name or through an obvious synonym), and answers only about the FIRST property a message names. Asking about something they have no opinion on just gets "no preference" and wastes the message.

What you receive: the request itself; for every requested item, its name, its already-known requirements, story hints, and a property menu, i.e. the properties its candidate listings really carry, each with the number of listings that carry it and sample values.

Priorities:
1. Properties the hints or the story point to, meaning things the request says the shopper will reveal when asked. Examples: "how it holds on" suggests a closure; who wears or uses it suggests an age group or a gender; what is inside suggests a processor; "how it looks" suggests colour, pattern or style. Each hint is mapped onto whichever menu property name is nearest.
2. After those, properties that buyers of this kind of product usually have opinions on (material, appearance, intended user, size or fit, style or type, main technical spec, a variant such as a flavour), preferring ones carried by many candidate listings (the "listings" count) and whose values vary from listing to listing. Weak choices: identifiers and model numbers, notes on shipping or warranty, pseudo-properties (variant or option selectors), and a property whose value is the same everywhere.
3. Exclusions: never a property that a known requirement already fixes; never price, stock or quantity; never one property twice, where two spellings of a property are still one.
4. Allocate questions among items in proportion to each item's uncertainty, and give the first question to an item whose hints are strong."""
SIGNAL_INSTRUCTION = """Purpose: find which menu properties the request's story points at, so a shopping assistant knows what to ask. Background: every hidden requirement of the shopper is a single value of ONE property, belonging to one requested product, and the shopper discloses it only to a question about precisely that property. Per requested item you receive its known requirements and a property menu (properties found on its candidate listings, with counts and sample values), plus the request itself.

Rules for "hinted":
* Include only properties that the story of the request, or the subjects it says the assistant should raise, point to. Examples: "how it holds on" suggests a closure; a mention of who will wear it suggests an age-group or gender property; "how it looks" suggests colour, pattern or style; what sits in a computer suggests a processor; a subject the request says to raise suggests its own property.
* Order best first; at most 3; [] when the request contains no such hint.
* Never include a property that a known requirement already fixes.

Answer with ONE JSON object and nothing more:
{"junk": ["menu properties that are no real product trait a shopper would state: identifiers, model numbers, option or variant selectors, notes about packaging, shipping or warranty, repeats of a known requirement"],
 "hinted": [{"why": "the words of the request pointing to it", "item": "the item's name exactly as you were given it", "property": "the property name in the menu's exact spelling"}]}"""
RETORT_INSTRUCTION = """Situation: a shopping assistant asked the shopper about one property of one product (sometimes two), and the shopper answered. Classify the answer and pull out what it says.

Kinds of answer (field "kind"):
* "free": no preference; they don't mind, it is open, anything goes, they never considered it.
* "split": they want the message divided, its subjects asked one at a time, or only its first part answered (it touched several subjects).
* "unclear": the message lost them, or they ask for it again or in simpler words.
* "other": anything else, e.g. a greeting, a question back, a budget change.
* "repeat": rather than answering about the property asked, they restate a requirement already present in KNOWN NEEDS (same property, same value).
* "need": they insist on a value, limit, exclusion or list (phrasing varies: firm, only X will do, X or Y and nothing else, nothing but, anything bar X, settled). Treat "other than X" or "anything bar X" as value X with op "is not". If they point at ANOTHER listing's page (identified by category, seller and attribute values), fill "reference" and choose the op: that page's very value gives "same as reference"; anything except it gives "different from reference"; what goes with that page's value, i.e. its counterpart, the value the page pairs with it, gives "counterpart of reference"; no higher or no lower than the page's value gives "at most reference" or "at least reference".

Which "property": the one the shopper's own reply names, in their words, if it names one (this may differ from what was asked). If it names none, the property asked (if two were asked, whichever one the value is for).

Limit words versus plain values:
- Explicit limit wording such as "under N", "no more than N", "at most N" or "at least N" fixes the op and the value even inside an odd-sounding sentence; "I can't do anything short of at most 34 inches" still means op "at most" and value "34 inches".
- When there are no limit words, a value they insist on is op "is", no matter which figure of speech it comes in. Someone who won't look beyond a value, or won't settle for anything else, wants exactly that value, numbers included (op "is", value "50 cm").

Reply with ONE JSON object and nothing more:
{"kind": "free|other|need|unclear|repeat|split",
 "property": "what the shopper's statement is about: their wording, or the asked property if that is what they mean",
 "value": "the value, limit or list they gave (separate the values of a list with ' | ')",
 "op": "at most|is|is not|one of|at least|counterpart of reference|at most reference|within reference|same as reference|different from reference|at least reference",
 "reference": null or {"property": "property to read on that page", "attributes": {"property": "value"}, "seller": "...", "category": "...", "tolerance": null or how far above or below that page's figure is allowed, "relation": "the phrase the shopper uses for how the wanted value stands to that page's value", "count": null or how many listings fit, by their account, "pick": null or how to choose among several},
 "answer": "" or a brief, plain reply should the shopper have asked the assistant something,
 "budget": null or a number: a new total spend the shopper names,
 "steps": [] or, for process demands the shopper now makes (obtain an approval, show or tell them something, use a tool prior to ordering), a list of {"goal": "the required outcome", "moment": "before_finish|before_decline|before_order", "tool": null, "kind": "message|tool", "messages": 1}}"""
INCIDENT_INSTRUCTION = """Mid-task, a store tool result came back carrying a line written by the shopper. Work out what that line changes for the shopping assistant.

Kinds of change:
* "price": according to the shopper, the chosen listing now costs something different.
* "stock": the chosen listing, says the shopper, can no longer be bought.
* "add": a requirement that is new.
* "drop": some property is no longer required ("drop X", "never mind X", "let X go").
* "budget": the shopper's spending limit moved, up or down. Copy the figure as the line writes it and keep the line's unit (no conversion: cents remain cents).
* "step": a process demand from the shopper about how the task is to be done (an approval to get, a tool to use before the order, something to show or tell them); describe it in "step" as {"goal": "the required outcome", "kind": "message|tool", "moment": "before_finish|before_order|before_decline", "tool": null, "messages": 1}.
* "other": anything else.

Return ONE JSON object:
{"changes": [{"kind": "price|other|step|add|stock|budget|drop",
  "note": "a single short sentence",
  "op": "at least|is not|one of|at most|is", "value": null or what the new requirement's value is,
  "property": null or the property that the added or dropped requirement concerns,
  "item": null or the requested product it applies to,
  "budget": null or, as a number, the new ceiling on the total}]}"""
ALLUDE_INSTRUCTION = """A shopping assistant is following a reference and has opened one store listing page; you read that page for it. It needs this page's value for one given property.

Where that value can be:
(a) the page's property table, or the seller's notes;
(b) not on this page at all, because the notes send that property to ANOTHER listing, which they identify by its category, seller and some attribute values.
The notes may also state which value of the property goes with which: either one pair covering the value of this page, or every pair the store has.

How to answer:
- Notes that forward the WANTED property to another listing: give "redirect" and no value. Notes forwarding some different property are not a redirect for this one.
- Notes that state the value outright: give that value.
- Otherwise: read the property table, where any key naming the property counts; if no key does, use the options field, the description or the title when one of them states it.
- If a sentence just lists the store's pairs, it does not reveal this page's own value.
- "paired" requires a pairing stated by the page: out of a list of pairs, take the pair that contains this page's own value (a pair can be read either way). It is null when the page states no pairing or its own value is not known.
- "wanted" depends on the shopper's RELATION: for the same value, this page's own value; for something going with, suiting or compatible with the page's value, the paired value, provided the page states a pairing.

Return ONE JSON object:
{"redirect": null or {"property": "...", "attributes": {"property": "value"}, "seller": "...", "category": "null unless the notes give one"}, "wanted": "the value called for by the RELATION, as explained above", "paired": "null, or the value this page pairs with its own", "value": "null, or the page's own value as written there"}"""
SEARCHTERM_INSTRUCTION = """Write new search-box queries for a store, on behalf of a shopping assistant.

How the search box behaves: a listing comes up only if every word of the query is found in its title, brand, category or property values (stems count as matches; very short words are dropped). Results always arrive in the same order, a few rows per page, so a good query is narrow enough that most of what it returns meets the requirements.

You are given: the product, its requirements, the queries tried so far, and a sample of listings already seen, each marked with whether it met the requirements.

Each new query is 1-3 words: the product noun worded the way this store's titles put it, combined with a required value written the way listings write it, or other words this store uses for the product or the value. Never repeat a query that was tried. Give no more than 5, best first.

Output: ONE JSON object with nothing else, {"queries": ["...", ...]}."""
OPERANDS_INSTRUCTION = """Fix a store tool call that the tool will not accept. Input: the tool's schema, name and description; the call's current arguments; the job the tool does here; and the store's refusal if one came back.

Rules:
1. Leave the existing arguments as they are, except where the refusal says they are wrong.
2. Fill in whatever is missing with the value that serves this job: a count of 1 if the arguments show no other, results from page one, and whichever option suits the job.
3. Stick to argument names and values the schema allows.

Output ONE JSON object only: {"args": {every argument of this call, complete}}."""
SELECTION_INSTRUCTION = """You pick the argument values for a store tool that finishes a shopping task in more than one way. Given: the tool's description, the arguments still missing together with the values each one permits, and the goal. Output ONE JSON object only, mapping each missing argument to the permitted value that achieves the goal: {"<argument name>": "<permitted value>", ...}."""
BARRIER_INSTRUCTION = """Output first: ONE JSON object and nothing else, either {"actions": [{"why": "which demanded step this carries out", "tool": "the tool's exact name", "args": {"argument": "value", ...}}]} with at most 2 actions in the order to run them, or {"actions": []} if no step is due.

Your role: at one point in a task, compare what a shopping assistant has done with what the shopper's request demands about the process. You receive the request, the procedural steps it demands, the moment, i.e. the next thing the assistant will do, the current basket, the remaining allowances, the store's tools (except those that change the cart or end the task, which the assistant runs itself), and every call made so far together with its result.

Which steps are due: those demanded steps that belong NOW, before the assistant proceeds, and are not yet done for the state things are in now.
- An approval or confirmation covers only the basket exactly as it now is; approval of a previous basket is worthless now.
- Steps completed for this state already are not due a second time.
- Steps that belong to a different moment are not due.

How to carry them out:
- A message to the shopper must make sense by itself: when it concerns a basket, give each line (quantity, title, price) along with the total, and state clearly what you ask the shopper for.
- No questions about product properties belong here, and the final message may only go to a demanded step.
- Store-side checks (stock, availability, a value) are made via the store's reading tools and never a message, except when the request wants to be told. Because the assistant reads every listing it buys anyway, such a check is needed only for basket listings that have not been read."""
ORDERBY_INSTRUCTION = """Every listing given meets the shopper's requirements; the shopper wants the best of them by one measure (some value wanted as large, or as small, as it can be). Find each listing's value for the measure in its properties, its title or its description; where units differ, convert them so the values can be compared.

Output ONE JSON object only: {"values": {"id": "the value you read"}, "order": ["id", ...]}. "order" lists all listing ids, best first. Unreadable values go at the end; for equal values, the cheaper listing comes first."""
INSPECTION_INSTRUCTION = """A shopping assistant is about to give up on one product. Before it does, go over its search once more.

Input: the budget, the product and its requirements, queries tried so far, and every listing it saw. READ listings carry their properties and notes, plus which requirements the assistant marked as failed or unaddressed; the other listings were seen only as search-result rows.

Output ONE JSON object only (each list may be empty):
{"queries": ["up to 2 fresh keyword queries of 1-3 words that are likely to bring up fitting listings; never one that was tried"],
 "unopened": ["up to 4 ids of rows not yet read that most probably meet the requirements, best first"],
 "fits_after_all": ["READ listings (by id) that do in fact satisfy every requirement, going by their own records: what a property's value says settles it, a different value fails, and a property that no record states is not met"]}"""
CLAUSE_INSTRUCTION = """The shopper set a condition, and a shopping assistant is checking one listing against it. Using only the listing's own records, decide whether the listing's value of the named property is one of the values supplied (meaning counts, not wording; convert units).

Output ONE JSON object only: {"holds": true or false, "value": "the value as the listing writes it, or null"}. If no record states the property, give "holds" false and "value" null."""
CORRESPOND_INSTRUCTION = """Goal: among the candidates below, shown with their records, find the one listing that fits a description (a seller, some attribute values and a category). A shopping assistant is looking for it.

A candidate matches only if all of these hold:
- the seller is the described seller;
- its category label names the described kind;
- every described attribute agrees with its records (meaning counts, not wording; units converted).
For a candidate without a shown property table, the title, description and seller notes serve as its records, so notes that state the described values make it a match. A candidate is no match if some described value is missing from its records or its records give another value. In that case answer null; do not settle for the nearest candidate, as the assistant will keep looking. If a PICK rule is given, more than one candidate may match: apply the rule to the matching ones and answer the one it selects.

Output ONE JSON object only, without explanation: {"id": "id of the matching listing, or null when no candidate matches"}."""
VALIDATE_INSTRUCTION = """Final gate: a shopping assistant is about to order. Test the listing below against EVERY listed requirement, using the listing's own records and these rules:
1. The listing's own property for a requirement states the value: met. Every key that names the property counts; comma lists, abbreviations, synonyms, plural or singular forms and unit conversions count as well.
2. If no property key covers it, the title, description, options field or seller's notes may state the value instead.
3. Stated nowhere: unmet.
4. A value filed under another property is irrelevant.
5. Compatibility with another product: satisfied if this listing names that product as something it fits, or if this listing's own part code begins with a series of compatible part codes named on the other product's page.
6. If the notes point a property to another listing's value, this listing's own value for that property does not meet the requirement.
7. Exclusions and closed lists have to be respected.
8. The product kind is right whenever the category label of the listing is the requested category, title notwithstanding.
9. Price multiplied by quantity has to stay within the budget (in the same unit as the prices shown).
10. If the records show the listing cannot be bought, it fails.
Check only the listed requirements; add none of your own.

Output ONE JSON object: {"problems": ["every requirement that is clearly unmet or that no record addresses"], "fits": true or false}."""
CAPTION_INSTRUCTION = """Output ONE JSON object only: {"labels": ["each label copied exactly as given", ...]}, with [] when no label qualifies.

A shopper asked for a single kind of product, and a store files its listings under category labels. Among the labels seen, select every one that IS that kind of product: a label whose meaning matches what the request's wording says as a whole, including a broader label that contains it (for a cardigan, "sweaters and cardigans" qualifies). Exclude neighbouring kinds such as parts, accessories, cases for the product, or a narrower or different type."""
ATLAS_INSTRUCTION = """Below is a JSON record that came back from a store tool. Tell which of its keys fills each role listed; write the key exactly as it appears, or null if none fills that role. Output ONE JSON object:
{"facts": "object holding the listing's property table", "notes": "free-text notes the seller wrote about it", "seller": "the shop or seller offering it", "brand": "brand", "category": "category label", "stock": "whether one can buy it", "price": "price", "text": "descriptive text", "title": "name of the listing", "id": "listing identifier", "count": "total number of results that match", "next": "token or marker leading to the next page of results", "lines": "list of lines in a cart", "total": "total price of a cart"}"""

class Posting:

    def __init__(self, ident_b: str):
        self.id = ident_b
        self.title = ''
        self.wording = ''
        self.tariff: float | None = None
        self.inventory: bool | None = None
        self.classification = ''
        self.marque = ''
        self.merchant = ''
        self.particulars: dict[str, Any] = {}
        self.jottings = ''
        self.unfolded = False
        self.folio_peruse = False
        self.untreated: dict[str, Any] = {}
        self.sieve = 'maybe'

    def ledger(self, boundary: int=DATUM_GLYPHS) -> dict[str, Any]:
        cell: dict[str, Any] = {'id': self.id, 'title': self.title[:160]}
        if self.classification:
            cell['category'] = self.classification
        if self.marque:
            cell['brand'] = self.marque
        if self.merchant:
            cell['seller'] = self.merchant
        if self.tariff is not None:
            cell['price'] = self.tariff
        if self.inventory is not None:
            cell['can_buy'] = self.inventory
        if self.particulars:
            (characteristics, spotted) = ({}, set())
            for (kk, vv) in self.particulars.items():
                imprint = (_Text._normalize(re.sub('(?<=[a-z0-9])(?=[A-Z])', ' ', str(kk))), json.dumps(vv, sort_keys=True, default=str))
                if imprint not in spotted:
                    spotted.add(imprint)
                    characteristics[kk] = vv
            cell['properties'] = characteristics
        if self.jottings:
            cell['seller_notes'] = self.jottings[:600]
        if self.wording:
            cell['description'] = self.wording[:400] if self.particulars else self.wording[:500]
        wording = json.dumps(cell, ensure_ascii=False)
        if len(wording) > boundary and self.particulars:
            characteristics = cell['properties']
            cell['properties'] = dict(list(characteristics.items())[:max(6, int(len(characteristics) * boundary / len(wording)))])
        if len(json.dumps(cell, ensure_ascii=False)) > boundary and cell.get('description'):
            cell['description'] = cell['description'][:200]
        return cell

class Stipulation:

    def __init__(self, aspect: str, operator_: str, worth: str, citation: dict | None=None, provenance: str='request'):
        self.aspect = str(aspect or '').strip()
        self.operator_ = str(operator_ or 'is').strip().lower()
        self.worth = str(worth or '').strip()
        self.citation = None if not isinstance(citation, dict) else citation
        self.provenance = provenance

    def wording(self) -> str:
        if self.operator_ in ALLUSION_OPERATORS and self.citation:
            return f'{self.aspect}: {self.operator_} (not yet resolved)'
        return f'{self.aspect}: {self.operator_} {self.worth}'

    def clef(self) -> str:
        return _Text._normalize(self.aspect)

class Article:

    def __init__(self, details: dict[str, Any], index: int):
        self.index = index
        self.moniker = str(details.get('name') or details.get('category') or 'the product').strip()
        self.classification = str(details.get('category') or self.moniker).strip()
        units_n = details.get('quantity')
        self.headcount = 1 if not (isinstance(units_n, (int, float)) and (not isinstance(units_n, bool)) and (1 <= units_n)) else int(units_n)
        self.stipulations: list[Stipulation] = []
        for stp in details.get('requirements') or []:
            if isinstance(stp, dict) and stp.get('property'):
                self.stipulations.append(Stipulation(stp.get('property'), stp.get('op'), stp.get('value'), stp.get('reference')))
        self.signals = [str(xh) for xh in details.get('hints') or [] if xh]
        self.tariff_lid = _Text._digit(details.get('price_cap'))
        if self.tariff_lid is not None and 0 >= self.tariff_lid:
            self.tariff_lid = None
        self.expressions = [str(tl) for tl in details.get('search_terms') or [] if tl]
        if not self.expressions:
            self.expressions = [self.classification]
        self.substantive = self.expressions[-1]
        self.reservoir: dict[str, Posting] = {}
        self.findings: dict[str, dict[str, set[str]]] = {}
        self.evaluated_per: dict[str, set[str]] = {}
        self.petitioned: list[str] = []
        self.loosed: list[str] = []
        self.search_over_with = 0
        self.spent_up = False
        self.selected: Posting | None = None
        self.classification_captions: dict[str, int] = {}
        self.captions_ok: set[str] | None = None
        self.pointer_poss: dict[str, dict[str, Any]] = {}
        self.inquiry_numbers: dict[str, set[str]] = {}
        self.worth_numbers: set[str] = set()
        self.worth_entire = False

    def inquiry_subsequent(self, inquiry: str) -> None:
        if inquiry and (not any((_Text._normalize(inquiry) == _Text._normalize(tl) for tl in self.expressions))):
            self.expressions.insert(min(self.search_over_with, len(self.expressions)), inquiry[:60])

    def atop_lid(self, posting: Posting) -> bool:
        return self.tariff_lid is not None and posting.tariff is not None and (posting.tariff > self.tariff_lid + 1e-06)

    def running(self) -> list[Stipulation]:
        return [stp for stp in self.stipulations if not (stp.citation and stp.operator_ in ALLUSION_OPERATORS)]

    def demand_wordings(self) -> list[str]:
        return [stp.wording() for stp in self.running()]

    def enfolded(self, posting: Posting) -> int:
        cnt = 0
        for stp in self.running():
            if stp.operator_ not in ('is', 'one of') or not stp.worth:
                continue
            cravings = [frozenset(_Text._counters(ww)) for ww in stp.worth.split('|') if _Text._counters(ww)]
            native = [vv for (kk, vv) in posting.particulars.items() if _Text._nigh(str(kk).replace('-', ' '), stp.aspect) or _Text._akin(str(kk).replace('-', ' ').replace('_', ' '), stp.aspect) >= 0.5]
            if not cravings or not native:
                continue
            constituents = {frozenset(_Text._counters(err)) for vv in native for err in re.split('[,;/|]', str(vv)) if _Text._counters(err)}
            if not any((ww in constituents for ww in cravings)):
                cnt += 1
        return cnt

    def literalness(self, posting: Posting) -> int:
        if not posting.particulars:
            return 0
        constituents: set[frozenset[str]] = set()
        for vv in posting.particulars.values():
            for err in re.split('[,;/|]', str(vv)):
                tl = frozenset(_Text._counters(err))
                if tl:
                    constituents.add(tl)
        points = 0
        for stp in self.running():
            if stp.operator_ not in ('is', 'one of') or not stp.worth:
                continue
            cravings = [frozenset(_Text._counters(ww)) for ww in stp.worth.split('|') if _Text._counters(ww)]
            if any((ww in constituents for ww in cravings)):
                points += 1
        return points

    def conforms(self, contain_hesitant: bool=False, thriftiest: bool=False, robust_just: bool=False) -> list[Posting]:
        (assured, hesitant) = ([], [])
        for (cap, posting) in self.reservoir.items():
            if not posting.unfolded:
                continue
            vv = self.findings.get(cap)
            if vv is None or vv['failed']:
                continue
            if self.atop_lid(posting):
                continue
            if robust_just and vv.get('weak'):
                continue
            if self.evaluated_per.get(cap, set()) >= set(self.demand_wordings()):
                (assured if not (vv['unsure'] or self.vocable_lax(cap)) else hesitant).append(posting)
        clef = lambda pst: 1e+18 if not pst.tariff is not None else pst.tariff
        if not thriftiest:
            assured.sort(key=lambda pst: (-self.literalness(pst), len(self.findings[pst.id].get('weak') or ()), clef(pst)))
        else:
            assured.sort(key=lambda pst: (self.enfolded(pst) > 0, clef(pst)))
        hesitant.sort(key=lambda pst: (len(self.findings[pst.id]['unsure']) + len(self.vocable_lax(pst.id)), clef(pst)))
        if assured or not contain_hesitant:
            return assured
        return hesitant

    def vocable_lax(self, cap: str) -> list[str]:
        vv = self.findings.get(cap) or {}
        wordings = set(vv.get('loose') or ())
        if not wordings:
            return []
        posting = self.reservoir.get(cap)
        return [stp.wording() for stp in self.running() if stp.wording() in wordings and stp.operator_ in ('is', 'one of') and stp.worth and (not re.search('\\d', stp.worth)) and ('fit' != stp.provenance) and (not (posting is not None and self.parts_vocables(posting, stp)))]

    @staticmethod
    def parts_vocables(posting: Posting, demand: Stipulation) -> bool:
        native = [str(vv) for (kk, vv) in posting.particulars.items() if _Text._nigh(str(kk), demand.aspect) or _Text._akin(str(kk).replace('_', ' '), demand.aspect) >= 0.5]
        vocables = set() if not native else set().union(*(_Text._counters(vv) for vv in native))
        return any((_Text._counters(substitute) & vocables for substitute in demand.worth.split('|')))

class _Ordering:
    """Choosing what to buy and placing the order."""

    def reselect(self) -> None:
        for article in self.items:
            self.evaluate(article, [pst for pst in self.aspirants(article) if pst.unfolded])
            if not [xf for xf in article.conforms() if xf.inventory is not False] or self.thriftiest:
                article.necessity_search_over_with = False
                article.scan_all_beneath = None
                part = max(2, self.paces_leftover() - self.terminus_fallback())
                if 3 <= part and self.duration_leftover() > ENFOLD_SECS + 20:
                    self.survey(article, part, terminal=True)
        for article in self.items:
            conforms = [xf for xf in article.conforms(thriftiest=self.thriftiest) if xf.inventory is not False]
            if not conforms:
                article.selected = None
                continue
            if self.thriftiest:
                article.selected = conforms[0]
            elif article.selected is None or article.selected.id not in [xf.id for xf in conforms]:
                article.selected = conforms[0]
        elects = {art.index: art.selected for art in self.items if art.selected is not None}
        for sink in {xd.index: xd for xc in getattr(self, 'clauses', []) for (_, xd, _) in xc['branches']}.values():
            self.carry_out_clauses(sink, elects)
            good = [xf.id for xf in self.obtainable(sink)]
            if sink.selected is not None and sink.selected.id not in good:
                sink.selected = None if not good else sink.reservoir[good[0]]
        if any((art.selected is None for art in self.items)):
            self.select_with_conclude()
            return
        if self.lid is not None:
            grand_total = sum(((art.selected.tariff or 0) * art.headcount for art in self.items))
            if grand_total > self.lid + 1e-06:
                self.select_with_conclude()

    def tariff_or_inventory_altered(self) -> bool:
        for article in self.items:
            pst = article.selected
            if pst is None:
                continue
            if pst.inventory is False:
                return True
            if self.lid is not None and pst.tariff is not None and (sum(((art.selected.tariff or 0) * art.headcount for art in self.items if art.selected)) > self.lid + 1e-06):
                return True
        return False

    def locate_purchase(self, essay: int=0) -> None:
        self.purchase_laps = getattr(self, 'purchase_laps', 0) + 1
        if 2 < essay or self.concluded or 8 < self.purchase_laps:
            return
        for cap in list(self.trolley):
            if not any((art.selected and cap == art.selected.id for art in self.items)):
                self.trolley_withdraw(cap)
        for article in self.items:
            if article.selected and article.selected.id not in self.trolley:
                if not self.trolley_append(article, article.selected):
                    if getattr(self, 'trolley_defective', False):
                        self.jot('cart_faulty')
                        self.forgo('the cart tool refuses its arguments')
                        return
                    article.findings.setdefault(article.selected.id, {'failed': set(), 'unsure': set()})['failed'].add('cart refused')
                    article.selected = None
        if any((art.selected is None for art in self.items)):
            self.select_with_conclude()
            return
        if self.page_initial():
            for article in self.items:
                pst = article.selected
                if pst is not None and (not pst.folio_peruse) and (self.paces_leftover() > 2 + self.outstanding_messages()):
                    self.unfold_lot(article, [pst.id], folio=True)
        elif self.paces_leftover() > 2 + self.outstanding_messages() and self.items and self.items[0].selected:
            initial = self.items[0].selected
            self.unfold_lot(self.items[0], [initial.id])
        altered = self.knob_strands()
        for article in self.items:
            if article.selected is not None and article.selected.id not in article.evaluated_per:
                self.evaluate(article, [article.selected])
                if (article.findings.get(article.selected.id) or {}).get('failed'):
                    self.jot('chosen_failed', id=article.selected.id, failed=sorted(article.findings[article.selected.id]['failed'])[:3])
                    altered = True
        if altered or self.tariff_or_inventory_altered():
            self.jot('rechoose', cart=self.trolley)
            for article in self.items:
                self.evaluate(article, [pst for pst in self.aspirants(article) if pst.unfolded])
            self.reselect()
            if self.concluded:
                return
            self.locate_purchase(essay + 1)
            return
        if self.conclude_purchase():
            return
        altered = self.knob_strands()
        if not altered:
            self.update_trolley_items()
            self.knob_strands()
        self.reselect()
        if not self.concluded:
            self.locate_purchase(essay + 1)

    def select_with_conclude(self) -> None:
        self.knob_strands()
        if self.thriftiest and (not getattr(self, 'standbys_over_with', False)):
            self.peruse_standbys()
        self.evaluate_whole()
        selected: dict[int, Posting] = {}
        couplings = self.conform_couplings()
        lodgers = {xh.index for (_, xh) in couplings}
        extras = {xa.index: xh for (xa, xh) in couplings}
        clauses = getattr(self, 'clauses', [])
        fountains = {xc['src'].index for xc in clauses}
        dependants = {xd.index for xc in clauses for (_, xd, _) in xc['branches']}
        summits = [(art, (self.order_by_finest(art, self.obtainable(art)) or [None])[0]) for art in self.items if art.index not in extras and art.index not in dependants]
        _Pool._concurrent([lambda art=art, xc=xc: self.validate(art, xc) for (art, xc) in summits if xc is not None])
        for article in sorted(self.items, key=lambda art: (art.index in extras or art.index in dependants, art.index not in lodgers and art.index not in fountains, art.index)):
            if article.index in extras and extras[article.index].index in selected:
                self.fasten_conform(article, extras[article.index], selected[extras[article.index].index])
            if article.index in dependants:
                self.carry_out_clauses(article, selected)
            tests = 0
            for hopeful in self.order_by_finest(article, self.obtainable(article)):
                if 3 <= tests or self.duration_leftover() < ENFOLD_SECS:
                    break
                tests += 1
                if self.validate(article, hopeful):
                    selected[article.index] = hopeful
                    break
            if article.index not in selected:
                balance = self.order_by_finest(article, self.obtainable(article))
                if balance and (3 <= tests or self.duration_leftover() < ENFOLD_SECS):
                    selected[article.index] = balance[0]
            if article.index in extras and article.index not in selected:
                lodger = extras[article.index]
                for different in self.obtainable(lodger)[:3]:
                    if selected.get(lodger.index) is different or self.duration_leftover() < ENFOLD_SECS:
                        continue
                    self.fasten_conform(article, lodger, different)
                    gained = self.obtainable(article)
                    if gained:
                        selected[lodger.index] = different
                        selected[article.index] = gained[0]
                        break
            if article.index in dependants and article.index not in selected:
                for clause in [xc for xc in clauses if xc['src'].index in selected and any((xd is article for (_, xd, _) in xc['branches']))]:
                    fount = clause['src']
                    if any((xh is fount and xa.index in selected for (xa, xh) in couplings)) or self.duration_leftover() < ENFOLD_SECS:
                        continue
                    presently = self.clause_persists(clause, selected[fount.index])
                    for different in self.obtainable(fount)[:4]:
                        if different is selected[fount.index] or self.clause_persists(clause, different) == presently:
                            continue
                        were = selected[fount.index]
                        selected[fount.index] = different
                        self.carry_out_clauses(article, selected)
                        gained = self.obtainable(article)
                        if gained and self.validate(fount, different):
                            selected[article.index] = gained[0]
                            break
                        selected[fount.index] = were
                        self.carry_out_clauses(article, selected)
                    if article.index in selected:
                        break
        lacking = [art for art in self.items if art.index not in selected]
        if lacking and (not getattr(self, 'latest_odds', False)) and (self.duration_leftover() > ENFOLD_SECS + 20) and (not self.expended()):
            self.latest_odds = True
            self.fallback_overrule = len(self.items) + 3 + max(self.withheld_messages(), 0 if not self.ratify_initial else 1)
            try:
                for art in lacking:
                    if self.paces_leftover() - self.terminus_fallback() >= 3:
                        art.spent_up = False
                        self.survey(art, self.paces_leftover() - self.terminus_fallback(), terminal=True)
            finally:
                self.fallback_overrule = None
            if any((self.obtainable(art) for art in lacking)):
                self.select_with_conclude()
                return
        if lacking and (not getattr(self, 'inspected', False)) and (self.duration_leftover() > ENFOLD_SECS + 25) and (not self.expended()):
            self.inspected = True
            if self.inspection(lacking) and all((self.obtainable(art) for art in lacking)):
                self.select_with_conclude()
                return
        self.nigh_elects = dict(selected)
        if lacking:
            self.forgo_grounds = 'no fitting listing for ' + ', '.join((art.moniker for art in lacking))
            self.jot('no_fit', items=[art.moniker for art in lacking], pools={art.moniker: len(art.reservoir) for art in lacking})
            self.forgo(self.forgo_grounds)
            return
        if self.lid is not None:

            def grand_total() -> float:
                return sum(((selected[art.index].tariff or 0) * art.headcount for art in self.items))
            sentry = 0
            while grand_total() > self.lid + 1e-06 and 10 > sentry:
                sentry += 1
                migrated = False
                for art in sorted(self.items, key=lambda idx: -((selected[idx.index].tariff or 0) * idx.headcount)):
                    conforms = [xf for xf in art.conforms(thriftiest=True) if xf.inventory is not False]
                    pos = -1 if not selected[art.index].id in [xf.id for xf in conforms] else [xf.id for xf in conforms].index(selected[art.index].id)
                    if 0 <= pos < len(conforms) - 1:
                        selected[art.index] = conforms[pos + 1]
                        migrated = True
                        break
                if not migrated:
                    break
            if grand_total() > self.lid + 1e-06:
                self.forgo_grounds = 'the cheapest fitting combination is over the budget'
                self.jot('over_budget', total=grand_total(), cap=self.lid)
                self.forgo(self.forgo_grounds)
                return
        for article in self.items:
            article.selected = selected[article.index]
        self.locate_purchase()

class _Closing:
    """Ending the task: orders, the nearest posting, declines and forced finishes."""

    def oblige_conclude(self) -> None:
        if self.concluded:
            return
        for article in self.items:
            if article.selected is None or article.selected.inventory is False:
                conforms = self.obtainable(article)
                article.selected = None if not conforms else conforms[0]
        set_up = all((art.selected for art in self.items)) and bool(self.items)
        if set_up and self.lid is not None:
            set_up = sum(((art.selected.tariff or 0) * art.headcount for art in self.items)) <= self.lid + 1e-06
        if set_up:
            musty = [cap for cap in self.trolley if not any((cap == art.selected.id for art in self.items))]
            joins = [art for art in self.items if art.selected.id not in self.trolley]
            if self.paces_leftover() >= len(musty) + len(joins) + 1:
                for cap in musty:
                    self.trolley_withdraw(cap)
                for article in joins:
                    self.trolley_append(article, article.selected)
                if self.conclude_purchase():
                    return
        if not self.concluded:
            self.forgo('forced')

    def obtainable(self, article: Article) -> list[Posting]:
        matrix = self.column_matrix(article)
        return [xf for xf in article.conforms(thriftiest=self.thriftiest) if xf.inventory is not False and (not self.column_falls_short(article, xf, matrix))]

    def terminus_fallback(self) -> int:
        if getattr(self, 'fallback_overrule', None) is not None:
            return self.fallback_overrule
        appearance = 2 if not self.page_initial() else min(len(self.items), 3)
        return len(self.items) + 6 + (0 if not (self.functions.get('facts') and (not self.functions.get('compare'))) else appearance) + min(4, sum((1 for state_s in self.protocol if state_s['moment'] != 'start')) + (0 if not self.ratify_initial else 1)) + (0 if not (self.thriftiest and (not getattr(self, 'standbys_over_with', False))) else 1)

    def forgo(self, grounds: str='') -> bool:
        details = self.functions.get('decline')
        if self.nighest_ok and self.purchase_nighest():
            return True
        if not self.nighest_ok and (not getattr(self, 'forgo_voiced', False)) and self.purchase_hesitant():
            return True
        if self.concluded:
            return True
        if not details:
            return False
        if self.protocol and (not getattr(self, 'forgo_barrier_checked', False)):
            self.forgo_barrier_checked = True
            self.barrier_check('before_decline', 'end the task without buying anything')
        operands: dict[str, Any] = self.stable_operands('decline')
        aside = details.get('remark')
        if aside and aside in (self._outline('decline').get('required') or []):
            operands[aside] = self._clip('decline', aside, 'Nothing available meets every requirement of the request.')
        result = self.dispatch(details['name'], operands)
        self.jot('declined', reason=grounds or self.forgo_grounds)
        if not self.concluded and (not getattr(self, 'forgo_vetoed', False)) and (self.paces_leftover() >= 1):
            self.forgo_vetoed = True
            self.jot('decline_refused', obs=_Text._dense(result['obs'], 200), lines=result['lines'][:3])
            if self.knob_strands() and self.paces_leftover() >= len(self.items) + 2:
                self.select_with_conclude()
            if not self.concluded:
                for cap in list(self.trolley):
                    if self.paces_leftover() > 1:
                        self.trolley_withdraw(cap)
                self.dispatch(details['name'], operands)
        return self.concluded

    def purchase_hesitant(self) -> bool:
        if getattr(self, 'hesitant_tested', False) or getattr(self, 'nighest_tested', False) or (not self.functions.get('checkout')):
            return False
        self.hesitant_tested = True
        elects: dict[int, Posting] = {}
        for article in self.items:
            ordered_by = self.order_by_finest(article, self.obtainable(article))
            if not ordered_by:
                matrix = self.column_matrix(article)
                ordered_by = [pst for pst in article.conforms(contain_hesitant=True, thriftiest=self.thriftiest) if pst.inventory is not False and (not self.column_falls_short(article, pst, matrix)) and (not article.vocable_lax(pst.id)) and ('cart refused' not in (article.findings.get(pst.id) or {}).get('failed', ()))]
            if not ordered_by:
                return False
            elects[article.index] = ordered_by[0]
        grand_total = sum(((elects[art.index].tariff or 0) * art.headcount for art in self.items))
        if self.lid is not None and grand_total > self.lid + 1e-06:
            return False
        self.jot('unsure_order', picks={art.moniker: elects[art.index].id for art in self.items})
        self.nigh_elects = elects
        return self.purchase_nighest()

    def purchase_nighest(self) -> bool:
        if getattr(self, 'nighest_tested', False) or not self.items or (not self.functions.get('checkout')):
            return False
        self.nighest_tested = True
        granted = getattr(self, 'nigh_elects', None) or {}
        elects: dict[int, Posting] = {}
        for article in self.items:
            elect = granted.get(article.index)
            if elect is None:
                ordered_by = self.order_by_finest(article, self.obtainable(article))
                elect = self.nighest_shortfall(article) if not ordered_by else ordered_by[0]
            if elect is not None:
                elects[article.index] = elect
        if not elects:
            return False
        musty = [cap for cap in self.trolley if not any((cap == pp.id for pp in elects.values()))]
        joins = [art for art in self.items if art.index in elects and elects[art.index].id not in self.trolley]
        self.jot('nearest_order', picks={art.moniker: elects[art.index].id for art in self.items if art.index in elects})
        for cap in musty:
            if self.paces_leftover() > len(joins) + 1:
                self.trolley_withdraw(cap)
        for art in joins:
            if self.paces_leftover() < 2:
                break
            art.selected = elects[art.index]
            self.trolley_append(art, art.selected)
        if not self.trolley or self.paces_leftover() < 1:
            return False
        if self.conclude_purchase() or self.concluded:
            return True
        self.nighest_laps = getattr(self, 'nighest_laps', 0) + 1
        if self.knob_strands() and 2 >= self.nighest_laps and (self.paces_leftover() >= len(self.items) + 1):
            self.nighest_tested = False
            self.nigh_elects = None
            self.select_with_conclude()
            return self.concluded
        return False

    def nighest_shortfall(self, article: Article) -> Posting | None:
        matrix = self.column_matrix(article)
        (finest, finest_clef) = (None, None)
        for pst in article.reservoir.values():
            if not pst.unfolded or pst.id not in article.findings or (not self.classification_ok(article, pst)):
                continue
            vv = article.findings[pst.id]
            fell_short = set(vv.get('failed') or ())
            if 'cart refused' in fell_short:
                continue
            curt = len(fell_short | self.column_falls_short(article, pst, matrix)) + (pst.inventory is False) + article.atop_lid(pst)
            if self.lid is not None and pst.tariff is not None and (pst.tariff * article.headcount > self.lid + 1e-06):
                curt += 1
            clef = (curt, len(vv.get('unsure') or ()), 1e+18 if not pst.tariff is not None else pst.tariff)
            if finest_clef is None or finest_clef > clef:
                (finest, finest_clef) = (pst, clef)
        return finest

    def conclude_purchase(self) -> bool:
        details = self.functions.get('checkout')
        if not details:
            return False
        if self.protocol and self.trolley:
            went = self.barrier_check('before_order', 'check out the basket below as the order')
            if went and self.outstanding_messages() and (self.paces_leftover() >= 2):
                went = self.barrier_check('before_order', 'check out the basket below as the order') or went
            if went is not False and self.ratify_initial and (self.pannier_imprint() != self.ratified_imprint):
                went = self.ratify_pannier() or went
            if went and (self.knob_strands() or self.tariff_or_inventory_altered()):
                self.jot('procedure_changed')
                return False
        self.exhibit_indicated_components()
        result = self.dispatch(details['name'], self.stable_operands('checkout'))
        if self.concluded:
            self.jot('ordered', cart=self.trolley)
            return True
        self.jot('order_refused', obs=_Text._dense(result['obs'], 200), lines=result['lines'][:3])
        return False

class _CartKeeping:
    """The cart: adding, removing and keeping track of lines."""

    def peruse_standbys(self) -> None:
        self.standbys_over_with = True
        if not all((self.obtainable(art) for art in self.items)):
            return
        for article in self.items:
            if self.paces_leftover() - self.terminus_fallback() < 1 or self.duration_leftover() < ENFOLD_SECS + 20:
                break
            conforms = self.obtainable(article)
            edge = (None if not conforms else conforms[0].tariff) if not len(conforms) > 1 else conforms[1].tariff
            lines = [pst for pst in self.aspirants(article) if not pst.unfolded and 'no' != pst.sieve and (pst.inventory is not False) and (pst.tariff is not None) and (edge is None or edge >= pst.tariff) and (not (self.lid is not None and pst.tariff * article.headcount > self.lid + 1e-06))]
            lines.sort(key=lambda pst: pst.tariff)
            if not lines:
                continue
            gained = self.unfold_lot(article, [pst.id for pst in lines[:self.collate_boundary]], minimum=self.terminus_fallback())
            self.jot('backups', item=article.moniker, read=[pst.id for pst in gained], bound=edge)
            if gained:
                self.evaluate(article, gained)

    def tentative_trolley(self) -> None:
        if getattr(self, 'tentative_over_with', False) or self.trolley or (not (self.functions.get('cart_add') or self.functions.get('cart_set'))):
            return
        if self.paces_leftover() - self.terminus_fallback() < 3 or self.duration_leftover() < ENFOLD_SECS + 30:
            self.tentative_over_with = True
            return
        self.evaluate_whole()
        for article in self.items:
            ordered_by = self.order_by_finest(article, self.obtainable(article))
            if not ordered_by:
                continue
            elect = ordered_by[0]
            if self.lid is not None and elect.tariff is not None and (elect.tariff * article.headcount > self.lid + 1e-06):
                continue
            self.tentative_over_with = True
            if self.trolley_append(article, elect):
                self.jot('provisional_cart', item=article.moniker, id=elect.id)
            return

    def trolley_withdraw(self, posting_ident: str) -> bool:
        if not self.functions.get('cart_add') and self.functions.get('cart_set'):
            strands = {kk: vv for (kk, vv) in self.trolley.items() if posting_ident != kk}
            if not self.fix_trolley(strands):
                return False
            self.trolley = strands
            self.trolley_article.pop(posting_ident, None)
            return True
        details = self.functions.get('cart_add')
        if not details:
            return False
        operands: dict[str, Any] = {details['id']: posting_ident}
        fashion = None if not isinstance(details.get('mode'), dict) else details.get('mode')
        if fashion and fashion.get('arg') and (fashion.get('remove') is not None):
            operands[fashion['arg']] = fashion['remove']
            if details.get('qty'):
                operands[details['qty']] = self.trolley.get(posting_ident, 1)
        elif details.get('qty'):
            characteristics = self._outline('cart_add').get('properties') or {}
            nadir = (characteristics.get(details['qty']) or {}).get('minimum')
            operands[details['qty']] = nadir if not (not isinstance(nadir, int) or 0 >= nadir) else 0
        result = self.dispatch(details['name'], operands)
        if result['ok']:
            self.trolley.pop(posting_ident, None)
            self.trolley_article.pop(posting_ident, None)
            return True
        return False

    def trolley_append(self, article: Article, posting: Posting) -> bool:
        if not self.functions.get('cart_add') and self.functions.get('cart_set'):
            strands = dict(self.trolley)
            strands[posting.id] = article.headcount
            if not self.fix_trolley(strands):
                return False
            self.trolley = strands
            self.trolley_article[posting.id] = article
            return True
        details = self.functions.get('cart_add')
        if not details:
            return False
        operands: dict[str, Any] = {details['id']: posting.id}
        units_n = article.headcount
        if details.get('qty'):
            characteristics = self._outline('cart_add').get('properties') or {}
            apex = (characteristics.get(details['qty']) or {}).get('maximum')
            if isinstance(apex, int) and apex < units_n:
                units_n = apex
            operands[details['qty']] = units_n
        fashion = None if not isinstance(details.get('mode'), dict) else details.get('mode')
        if fashion and fashion.get('arg') and (fashion.get('add') is not None):
            operands[fashion['arg']] = fashion['add']
        result = self.dispatch(details['name'], operands)
        if result['ok'] and (not _Shape._vetoed_trolley(result['obs'])):
            self.trolley[posting.id] = units_n
            self.trolley_article[posting.id] = article
            return True
        self.trolley_defective = bool(result.get('bad_args'))
        self.jot('cart_add_failed', error=result['error'], obs=_Text._dense(result['obs'], 200))
        return False

    def fix_trolley(self, strands: dict[str, int]) -> bool:
        details = self.functions.get('cart_set') or {}
        characteristics = self._outline('cart_set').get('properties') or {}
        strand_characteristics = ((characteristics.get(details.get('lines') or '') or {}).get('items') or {}).get('properties') or {}
        apex = (strand_characteristics.get(details.get('qty') or '') or {}).get('maximum')
        lines = []
        for (cap, units_n) in strands.items():
            line = {details['id']: cap}
            if details.get('qty'):
                line[details['qty']] = units_n if not isinstance(apex, int) else min(units_n, apex)
            lines.append(line)
        result = self.dispatch(details['name'], {details['lines']: lines})
        if not result['ok']:
            self.jot('cart_set_failed', error=result['error'], obs=_Text._dense(result['obs'], 200))
        return result['ok']

    def update_trolley_items(self) -> None:
        for (cap, article) in list(self.trolley_article.items()):
            if self.paces_leftover() <= 2:
                return
            unfolded = self.unfold_lot(article, [cap], folio=True)
            self.jot('refreshed', id=cap, price=None if not cap in article.reservoir else article.reservoir[cap].tariff, stock=None if not cap in article.reservoir else article.reservoir[cap].inventory)

class _Conditions:
    """Conditional stipulations and products that must fit each other."""

    @staticmethod
    def inside_sequence_run(posting: Posting, sequence_run: list[str]) -> bool:
        return any((xh in sequence_run for xh in _RANGE.findall(posting.title or '')))

    def fasten_conform(self, article: Article, lodger_article: Article, lodger: Posting) -> None:
        folio = ' '.join((xx for xx in [lodger.title[:120], (lodger.wording or '')[:300], (lodger.jottings or '')[:150]] if xx))
        article.stipulations = [stp for stp in article.stipulations if 'fit' != stp.provenance]
        article.stipulations.append(Stipulation('compatibility', 'is', f'works with the {lodger_article.moniker} chosen, whose page reads: {folio}', provenance='fit'))
        token_code = re.compile('\\b[A-Z]{1,6}-?\\d{3,}[A-Z0-9]*\\b')
        entitled = {xc for art in self.items for pst in art.reservoir.values() for xc in token_code.findall(pst.title or '')}
        openers = {re.match('[A-Z]+', xc).group(0) for pst in article.reservoir.values() for xc in token_code.findall(pst.title or '')}
        craved: list[str] = []
        for pst in self.aspirants(article):
            if not pst.unfolded or (article.findings.get(pst.id) or {}).get('failed'):
                continue
            for xc in token_code.findall(' '.join([pst.wording or '', pst.jottings or ''])):
                if xc not in entitled and xc not in craved and (re.match('[A-Z]+', xc).group(0) in openers):
                    craved.append(xc)
        lodger_wording = ' '.join([lodger.wording or '', lodger.jottings or '', json.dumps(lodger.particulars, ensure_ascii=False)])
        prefixes = {xh for art in self.items if art is not lodger_article for pst in art.reservoir.values() for xh in _RANGE.findall(pst.title or '')}
        sequence_run = []
        for xc in re.findall('\\b[A-Z]{1,5}\\d{1,3}\\b', lodger_wording):
            lineage = re.match('[A-Z]+', xc).group(0)
            if xc not in sequence_run and xc not in _RANGE.findall(lodger.title or '') and (xc in prefixes or any((re.match('[A-Z]+', xh).group(0) == lineage for xh in prefixes))):
                sequence_run.append(xc)
        for xc in sequence_run[:2] + [ww for ww in craved if ww not in sequence_run][:3]:
            if self.paces_leftover() <= self.terminus_fallback() + 2 or self.duration_leftover() < ENFOLD_SECS + 10:
                break
            inquiry = xc if not xc in sequence_run else f'{xc} {article.classification}'
            self.search(article, inquiry=inquiry, upper_folios=1, purse=self.terminus_fallback())
        article.conform_sequence_run = sequence_run[:2]
        if sequence_run and self.paces_leftover() > self.terminus_fallback() + 2:
            my_own = [pst for pst in self.aspirants(article) if not pst.unfolded and self.inside_sequence_run(pst, sequence_run)]
            if my_own:
                self.unfold_lot(article, [pst.id for pst in my_own[:3 if not self.lot_unfold() else self.collate_boundary]], minimum=self.terminus_fallback() + 1)
        self.jot('fit_bound', item=article.moniker, host=lodger.id, series=sequence_run[:2], looked_up=craved[:3])
        self.evaluate(article, [pst for pst in self.aspirants(article) if pst.unfolded])
        if sequence_run:
            conform_demand = article.stipulations[-1].wording()
            for pst in self.aspirants(article):
                vv = article.findings.get(pst.id)
                if vv and pst.unfolded and self.inside_sequence_run(pst, sequence_run):
                    for clef in ('unsure', 'weak', 'failed'):
                        if clef in vv:
                            vv[clef].discard(conform_demand)

    def conform_couplings(self) -> list[tuple[Article, Article]]:
        away = []
        for canon in self.conform_canons:
            (runsum, lodger) = (str(canon.get('accessory') or ''), str(canon.get('host') or ''))
            if not runsum or not lodger or len(self.items) < 2:
                continue
            xa = max(self.items, key=lambda art: _Text._akin(art.moniker, runsum))
            xh = max(self.items, key=lambda art: _Text._akin(art.moniker, lodger))
            if xa is not xh and _Text._akin(xa.moniker, runsum) >= 0.2 and (_Text._akin(xh.moniker, lodger) >= 0.2) and ((xa, xh) not in away):
                away.append((xa, xh))
        return away

    def carry_out_clauses(self, article: Article, selected: dict[int, Posting]) -> None:
        my_own = [xc for xc in self.clauses if any((xd is article for (_, xd, _) in xc['branches']))]
        if not my_own:
            return
        craved = []
        for clause in my_own:
            elect = selected.get(clause['src'].index)
            persists = None if not elect is not None else self.clause_persists(clause, elect)
            if persists is None:
                continue
            for (whenever, sink, xb) in clause['branches']:
                if sink is article and persists == whenever:
                    craved.append(Stipulation(xb['property'], xb.get('op') or 'is', xb['value'], provenance='condition'))
        earlier = sorted((stp.wording() for stp in article.stipulations if 'condition' == stp.provenance))
        if earlier == sorted((stp.wording() for stp in craved)):
            return
        article.stipulations = [stp for stp in article.stipulations if 'condition' != stp.provenance] + craved
        self.jot('condition_applied', item=article.moniker, reqs=[stp.wording() for stp in craved])
        self.evaluate(article, [pst for pst in self.aspirants(article) if pst.unfolded])
        if not self.obtainable(article) and self.paces_leftover() - self.terminus_fallback() >= 3 and (self.duration_leftover() > ENFOLD_SECS + 20):
            article.spent_up = False
            self.survey(article, self.paces_leftover() - self.terminus_fallback(), terminal=True)

    def clause_persists(self, clause: dict[str, Any], posting: Posting) -> bool | None:
        hoard = self.__dict__.setdefault('held_cache', {})
        clef = (id(clause), posting.id, json.dumps(posting.particulars, sort_keys=True, default=str))
        if clef in hoard:
            return hoard[clef]
        native = [str(vv) for (kk, vv) in posting.particulars.items() if _Text._nigh(str(kk).replace('_', ' '), clause['prop']) or _Text._akin(str(kk).replace('_', ' '), clause['prop']) >= 0.5]
        persists: bool | None = None
        if native:
            wording = _Text._vocables(' '.join(native))
            cravings = [_Text._vocables(vv) for vv in clause['values'] if _Text._vocables(vv)]
            if any((wording >= ww for ww in cravings)):
                persists = True
            elif not any((ww & wording for ww in cravings)):
                persists = False
        if persists is None:
            client = f"PROPERTY: {clause['prop']}\nVALUES: {json.dumps(clause['values'], ensure_ascii=False)}\n\nLISTING:\n" + json.dumps(posting.ledger(2500), ensure_ascii=False)
            rejoinder = self.model.json([QUICK_ENGINE, STANDBY_ENGINE], CLAUSE_INSTRUCTION, client, 300)
            if isinstance(rejoinder, dict) and isinstance(rejoinder.get('holds'), bool):
                persists = rejoinder['holds']
        hoard[clef] = persists
        self.jot('condition', src=clause['src'].moniker, listing=posting.id, prop=clause['prop'], holds=persists)
        return persists

    def peruse_clauses(self, untreated: Any) -> list[dict[str, Any]]:
        away = []
        for xc in [] if not isinstance(untreated, list) else untreated:
            clause = None if not isinstance(xc, dict) else xc.get('if')
            if not isinstance(clause, dict) or not clause.get('property'):
                continue
            fount = self.article_titled(clause.get('item'))
            values = clause.get('values')
            values = ([] if not values else [str(values)]) if not isinstance(values, list) else [str(vv) for vv in values if vv]
            limbs = []
            for (clef, whenever) in (('then', True), ('otherwise', False)):
                xb = xc.get(clef)
                if isinstance(xb, dict) and xb.get('property') and xb.get('value'):
                    sink = self.article_titled(xb.get('item'))
                    if sink is not None and sink is not fount:
                        limbs.append((whenever, sink, xb))
            if fount is None or not values or (not limbs):
                continue
            away.append({'src': fount, 'prop': str(clause['property']), 'values': values, 'branches': limbs})
            for (_, sink, xb) in limbs:
                substantive = sink.substantive
                for vv in str(xb['value']).split('|'):
                    vv = vv.strip()
                    if vv and len(vv) <= 40 and (not any((_Text._normalize(vv) in _Text._normalize(tl) for tl in sink.expressions))):
                        sink.expressions.append(f'{vv} {substantive}'[:60])
        if away:
            self.jot('conditions', rules=[(xc['src'].moniker, xc['prop'], xc['values'], [(ww, xd.moniker, xb.get('property'), xb.get('value')) for (ww, xd, xb) in xc['branches']]) for xc in away])
        return away

    def article_titled(self, moniker: Any) -> Article | None:
        moniker = str(moniker or '')
        if not moniker or not self.items:
            return None
        finest = max(self.items, key=lambda art: max(_Text._akin(art.moniker, moniker), _Text._akin(art.classification, moniker)))
        return None if not max(_Text._akin(finest.moniker, moniker), _Text._akin(finest.classification, moniker)) >= 0.2 else finest

class _References:
    """Stipulations whose value is read off another listing."""

    def exhibit_indicated_components(self) -> None:
        details = self.functions.get('search_text')
        if not details or not details.get('query') or (not self.trolley):
            return
        exhibited = self.__dict__.setdefault('parts_shown', set())
        entitlements = ' '.join((pst.title or '' for art in self.items for pst in art.reservoir.values()))
        token_codes: list[str] = []
        for cap in self.trolley:
            pst = next((art.reservoir[cap] for art in self.items if cap in art.reservoir), None)
            if pst is None:
                continue
            for mm in _INDICATOR.finditer(' '.join([pst.wording or '', pst.jottings or ''])):
                token_code = mm.group(1).strip('-')
                if len(token_code) < 4 or not re.search('\\d', token_code) or token_code.lower() in exhibited or (token_code in token_codes):
                    continue
                if re.search('(?<![A-Za-z0-9])' + re.escape(token_code) + '(?![A-Za-z0-9])', pst.title or '', re.I):
                    continue
                if re.search('(?<![A-Za-z0-9])' + re.escape(token_code) + '(?![A-Za-z0-9])', entitlements, re.I):
                    exhibited.add(token_code.lower())
                    continue
                token_codes.append(token_code)
        if not token_codes or self.paces_leftover() < 2 + self.outstanding_messages():
            return
        operands: dict[str, Any] = {}
        if isinstance(details.get('mode'), dict) and details['mode'].get('arg'):
            operands[details['mode']['arg']] = details['mode'].get('value')
        operands[details['query']] = self._clip('search_text', details['query'], ' '.join(token_codes[:5]))
        result = self.dispatch(details['name'], operands)
        exhibited.update((xc.lower() for xc in token_codes[:5]))
        self.jot('pointed_parts', codes=token_codes[:5], rows=len(_Shape._discover_sequence_of_lookups(result['obs']) or []))

    def _correspond_folio(self, jotpad: Article, discovered: list[Posting], allusion: dict, merchant: str, classification: str, qualities: dict, qualities_correspond: Any, qualities_inside_wording: Any) -> Posting | None:
        folio = None
        tested = 0
        elect = str(allusion.get('pick') or '').strip()
        count = None if not isinstance(allusion.get('count'), int) else allusion.get('count')
        if elect or (count or 0) > 1:
            span_b = min(len(discovered), 3 * self.collate_boundary, max(4, 2 * (count or 2)))
            for start in range(0, span_b, self.collate_boundary):
                if self.paces_leftover() < 3:
                    break
                to_do = [xx.id for xx in discovered[start:min(start + self.collate_boundary, span_b)] if not (jotpad.reservoir.get(xx.id) and jotpad.reservoir[xx.id].unfolded)]
                for pst in self.unfold_lot(None, to_do):
                    jotpad.reservoir[pst.id] = pst
            ledgers = [jotpad.reservoir[pst.id].ledger(1200) for pst in discovered if pst.id in jotpad.reservoir and jotpad.reservoir[pst.id].unfolded]
            if ledgers:
                client = 'DESCRIPTION: category ' + classification + '; seller ' + merchant + '; attributes ' + json.dumps(qualities, ensure_ascii=False) + ('' if not elect else f'\nPICK: {elect}') + ('' if not count else f'\nLISTINGS THAT FIT: {count}') + '\n\nCANDIDATES:\n' + json.dumps(ledgers, ensure_ascii=False)
                match = self.model.json([QUICK_ENGINE, STANDBY_ENGINE], CORRESPOND_INSTRUCTION, client, 600)
                if isinstance(match, dict) and match.get('id') in jotpad.reservoir:
                    folio = jotpad.reservoir[match['id']]
            discovered = [pst for pst in discovered if not (jotpad.reservoir.get(pst.id) and jotpad.reservoir[pst.id].unfolded)] if not folio is not None else []
        singular = not self.functions.get('compare') or getattr(self, 'collate_halted', False)
        dimension = self.collate_boundary if not singular else 1
        spotted: list[Posting] = []
        while discovered and tested < (3 if not singular else 5) and (folio is None) and (self.paces_leftover() >= 3):
            lot = discovered[:dimension]
            discovered = discovered[dimension:]
            tested += 1
            unfolded = self.unfold_lot(None, [pst.id for pst in lot])
            for pst in unfolded:
                jotpad.reservoir[pst.id] = pst
            presently = [jotpad.reservoir[pst.id] for pst in lot if pst.id in jotpad.reservoir and jotpad.reservoir[pst.id].unfolded]
            literal = [pst for pst in presently if qualities_correspond(pst)]
            if literal:
                folio = literal[0]
                break
            spotted += presently
            if singular and discovered and (len(spotted) < 3):
                continue
            exhibited = presently if not singular else spotted
            ledgers = [pst.ledger(1600) for pst in exhibited]
            spotted = []
            if not ledgers:
                continue
            client = 'DESCRIPTION: category ' + classification + '; seller ' + merchant + '; attributes ' + json.dumps(qualities, ensure_ascii=False) + '\n\nCANDIDATES:\n' + json.dumps(ledgers, ensure_ascii=False)
            match = self.model.json([QUICK_ENGINE, STANDBY_ENGINE], CORRESPOND_INSTRUCTION, client, 600)
            couched = [pst for pst in exhibited if qualities_inside_wording(pst)]
            if isinstance(match, dict) and match.get('id') in jotpad.reservoir:
                folio = jotpad.reservoir[match['id']]
            elif isinstance(match, dict) and 'id' in match:
                continue
            elif len(couched) == 1:
                folio = couched[0]
            elif len(ledgers) == 1 and (not discovered):
                folio = jotpad.reservoir[ledgers[0]['id']]
        return folio

    def work_out_citation(self, article: Article, demand: Stipulation, leap: int=0) -> None:
        allusion = demand.citation or {}
        if not getattr(demand, 'allusion_imprint0', None):
            demand.allusion_imprint0 = _Shape._allusion_imprint(allusion)
        minimum = max(4, getattr(self, 'allusion_minimum', 0))
        if 7 < leap or self.paces_leftover() < minimum or self.duration_leftover() < ENFOLD_SECS + 20:
            return
        merchant = str(allusion.get('seller') or '').strip()
        classification = str(allusion.get('category') or '').strip()
        qualities = {} if not isinstance(allusion.get('attributes'), dict) else allusion.get('attributes')
        destination_aspect = str(allusion.get('property') or demand.aspect).strip()
        clef = json.dumps([_Text._normalize(merchant), _Text._normalize(classification), sorted(((_Text._normalize(kk), _Text._normalize(vv)) for (kk, vv) in qualities.items()))])
        (jotpad, probed) = self.__dict__.setdefault('ref_search', {}).setdefault(clef, (Article({'name': classification or 'listing', 'category': classification, 'search_terms': []}, 99), set()))
        values = [str(vv).strip() for vv in qualities.values() if str(vv).strip() and len(str(vv).split()) <= 3]
        for vv in qualities.values():
            if len(str(vv).split()) > 3:
                values += [ww for ww in re.findall('[A-Za-z][A-Za-z-]{3,}', str(vv)) if _Text._normalize(ww) != _Text._normalize(merchant) and ww not in values][:4]
        values.sort(key=lambda vv: bool(re.search('\\d', vv)))
        quality_vocables = ' '.join(values[:3])
        substantive = '' if not classification else classification.split()[-1]
        substantives = list(dict.fromkeys((cnt for cnt in [substantive, '' if not classification else (re.split('\\s+and\\s+', classification)[0].split() or [''])[-1]] if cnt)))
        searches = []
        if merchant:
            searches.append((merchant + ' ' + quality_vocables).strip())
            searches += [f'{merchant} {cnt}' for cnt in substantives]
            searches.append(merchant)
        if classification and quality_vocables:
            searches.append(f'{quality_vocables} {substantive}'.strip())
        searches = list(dict.fromkeys((srch for srch in searches if srch)))

        def merchant_ok(pst: Posting) -> bool:
            if not merchant:
                return True
            return _Text._akin(pst.merchant, merchant) >= 0.5 or _Text._akin(pst.marque, merchant) >= 0.5 or _Text._normalize(merchant) in _Text._normalize(pst.title + ' ' + pst.wording + ' ' + pst.jottings)

        def coincidence(pst: Posting) -> int:
            vocables = _Text._counters(quality_vocables)
            if not vocables:
                return 0
            untreated = pst.title + ' ' + pst.wording + ' ' + pst.jottings + ' ' + json.dumps(pst.particulars)
            wording = _Text._counters(untreated)
            compressed = _Text._normalize(untreated).replace(' ', '')
            merged = sum((1 for vv in values if re.search('\\d', vv) and (not (_Text._counters(vv) and _Text._counters(vv) <= wording)) and (len(_Text._normalize(vv).replace(' ', '')) >= 3) and (_Text._normalize(vv).replace(' ', '') in compressed)))
            return len(vocables & wording) + merged

        def qualities_inside_wording(pst: Posting) -> bool:
            if not qualities:
                return False
            wording = _Text._vocables(' '.join([pst.title or '', pst.jottings or '', pst.wording or '', ' '.join((f'{kk} {vv}' for (kk, vv) in pst.particulars.items()))]))
            return all((_Text._vocables(vv) and _Text._vocables(vv) <= wording for vv in qualities.values() if str(vv).strip()))

        def qualities_correspond(pst: Posting) -> bool:
            if not qualities or not pst.unfolded:
                return False
            particulars = [(_Text._normalize(re.sub('(?<=[a-z0-9])(?=[A-Z])', ' ', str(kk))), _Text._normalize(vv)) for (kk, vv) in pst.particulars.items()]
            for (kk, vv) in qualities.items():
                craving = _Text._normalize(vv)
                if not craving or not any(((_Text._nigh(f_key, kk) or _Text._akin(f_key, kk) >= 0.5) and craving == f_val for (f_key, f_val) in particulars)):
                    return False
            return True
        folio = None

        def aspirants_presently() -> list[Posting]:
            my_own = [pst for pst in jotpad.reservoir.values() if pst.id not in probed and merchant_ok(pst)]
            gained = [pst for pst in my_own if not classification or self.classification_ok(jotpad, pst)]
            if not gained and merchant and (len(my_own) <= 2 * self.collate_boundary):
                gained = my_own
            purchase = {cap: idx for (idx, cap) in enumerate(jotpad.reservoir)}
            return sorted(gained, key=lambda pst: (-coincidence(pst), not _Text._akin(pst.merchant, merchant) >= 0.5, purchase.get(pst.id, 0)))
        for srch in searches:
            if folio is not None or self.paces_leftover() < minimum:
                break
            for _ in range(3):
                if self.paces_leftover() < minimum or self.search(jotpad, inquiry=srch, upper_folios=1, purse=3) == 0 or aspirants_presently():
                    break
            discovered = aspirants_presently()
            if not discovered:
                continue
            probed.update((pst.id for pst in discovered))
            folio = self._correspond_folio(jotpad, discovered, allusion, merchant, classification, qualities, qualities_correspond, qualities_inside_wording)
        if folio is None and (not probed):
            self.jot('reference_unfound', seller=merchant, category=classification)
            return
        if folio is None:
            self.jot('reference_unmatched', seller=merchant, category=classification)
            return
        coupled = 'counterpart of reference' == demand.operator_
        kinship = str(allusion.get('relation') or '').strip()

        def peruse_folio() -> Any:
            client = f'PROPERTY WANTED: {destination_aspect}\n' + ('' if not kinship else f'RELATION THE SHOPPER ASKED FOR: {kinship}\n') + '\nPAGE:\n' + json.dumps(folio.ledger(3000), ensure_ascii=False)
            return self.model.json([QUICK_ENGINE, STANDBY_ENGINE], ALLUDE_INSTRUCTION, client, 600)
        meagre = self.meagre_collate and (not folio.particulars)
        if meagre and (not folio.jottings) and (self.paces_leftover() >= 3):
            (folio.particulars, meagre) = (self.fetch_up_particulars(folio.id) or {}, False)
        peruse = peruse_folio()
        if meagre and self.paces_leftover() >= 3 and isinstance(peruse, dict) and (not peruse.get('redirect')) and (not peruse.get('value') or coupled):
            folio.particulars = self.fetch_up_particulars(folio.id) or {}
            if folio.particulars:
                peruse = peruse_folio()
        self.jot('reference', hop=leap, page=folio.id, read=_Text._dense(peruse, 200))
        reroute = None if not isinstance(peruse, dict) else peruse.get('redirect')
        if isinstance(reroute, dict) and reroute:
            concerning = str(reroute.get('property') or destination_aspect)
            if not (_Text._nigh(concerning, destination_aspect) or _Text._akin(concerning, destination_aspect) >= 0.5 or _Text._normalize(destination_aspect) in _Text._normalize(concerning) or (_Text._normalize(concerning) in _Text._normalize(destination_aspect))):
                reroute = None
                if not peruse.get('value') and meagre and (not folio.particulars) and (self.paces_leftover() >= 3):
                    folio.particulars = self.fetch_up_particulars(folio.id) or {}
                    once_more = None if not folio.particulars else peruse_folio()
                    if isinstance(once_more, dict):
                        peruse['value'] = once_more.get('value') or peruse.get('value')
                        peruse['paired'] = once_more.get('paired') or peruse.get('paired')
                if not peruse.get('value'):
                    native = next((vv for (kk, vv) in folio.particulars.items() if _Text._nigh(str(kk).replace('_', ' '), destination_aspect) or _Text._akin(str(kk).replace('_', ' '), destination_aspect) >= 0.5), None)
                    if native not in (None, ''):
                        peruse['value'] = str(native)
                self.jot('reference_other_redirect', about=concerning, wanted=destination_aspect, value=peruse.get('value'))
        if isinstance(reroute, dict) and reroute:
            demand.citation = reroute
            demand.citation['property'] = destination_aspect
            if kinship:
                demand.citation['relation'] = kinship
            if allusion.get('tolerance'):
                demand.citation['tolerance'] = allusion['tolerance']
            self.work_out_citation(article, demand, leap + 1)
            return
        worth = None if not isinstance(peruse, dict) else peruse.get('value' if not coupled else 'paired')
        if isinstance(peruse, dict) and 'same as reference' == demand.operator_ and peruse.get('paired') and peruse.get('wanted') and (_Text._normalize(peruse['wanted']) == _Text._normalize(peruse['paired'])) and (_Text._normalize(peruse['paired']) != _Text._normalize(peruse.get('value'))):
            worth = peruse['paired']
            self.jot('relation_paired', page=folio.id, relation=kinship[:80], value=worth)
        if worth and 'within reference' == demand.operator_:
            leeway = str(allusion.get('tolerance') or '').strip()
            (qty_n, magnitude) = (_Text._digit(worth), _Text._digit(leeway))
            module = re.sub('^[\\s\\d.,+-]+', '', str(worth)).strip()
            if not (qty_n is not None and magnitude is not None):
                demand.operator_ = 'same as reference'
            else:
                (bottom_i, top_i) = (qty_n - magnitude, qty_n + magnitude)
                demand.allusion_imprint = _Shape._allusion_imprint(allusion)
                (demand.worth, demand.operator_, demand.citation) = (f'{bottom_i:g} {module}'.strip(), 'at least', None)
                article.stipulations.append(Stipulation(demand.aspect, 'at most', f'{top_i:g} {module}'.strip(), provenance=demand.provenance))
                self.jot('reference_range', prop=demand.aspect, low=demand.worth, high=f'{top_i:g} {module}'.strip())
                worth = None
        if worth:
            demand.allusion_imprint = _Shape._allusion_imprint(allusion)
            demand.worth = str(worth)
            demand.operator_ = ALLUSION_OPERATORS.get(demand.operator_, 'is')
            demand.citation = None
            if 'is' == demand.operator_ and len(demand.worth) <= 40 and article.expressions and (not any((_Text._normalize(demand.worth) in _Text._normalize(tl) for tl in article.expressions))):
                article.inquiry_subsequent(f'{demand.worth} {article.substantive}')
        elif coupled and isinstance(peruse, dict) and peruse.get('value'):
            self.jot('pair_unstated', page=folio.id, value=peruse.get('value'))
            demand.allusion_imprint = _Shape._allusion_imprint(allusion)
            (demand.worth, demand.operator_, demand.citation) = (str(peruse['value']), 'is', None)

class _Questioning:
    """Planning, sending and reading questions about hidden wants."""

    def plural_design(self, count: int, seized: Any) -> list[tuple[Article, str]]:
        files: dict[int, list[str]] = {art.index: [] for art in self.items}
        initial: list[int] = []
        for (art, pricing) in self.mapped_out:
            if art.index in files and (not seized(art, pricing)) and (not any((_Text._nigh(pricing, srch) for srch in files[art.index]))):
                files[art.index].append(pricing)
                if art.index not in initial:
                    initial.append(art.index)
        for art in self.items:
            bill_of_fare = [err for err in self.characteristic_bill_of_fare(art) if err['listings'] >= 2 or err['of'] <= 2]
            srch = files[art.index]
            summit = next((err['property'] for err in bill_of_fare if err['listings'] >= 3), None)
            if summit and (not seized(art, summit)) and (not any((_Text._nigh(summit, xx) for xx in srch[:2]))):
                srch[:] = [xx for xx in srch if not _Text._nigh(xx, summit)]
                srch.insert(min(1, len(srch)), summit)
            for err in bill_of_fare:
                if not seized(art, err['property']) and (not any((_Text._nigh(err['property'], xx) for xx in srch))):
                    srch.append(err['property'])
            if art.index not in initial:
                initial.append(art.index)
        related_to = {art.index for art in self.items if any(('shopper' == stp.provenance for stp in art.stipulations))}
        by_index_b = {art.index: art for art in self.items}
        spin = sorted(initial, key=lambda idx: (idx in related_to, len(by_index_b[idx].petitioned), initial.index(idx)))
        design: list[tuple[Article, str]] = []
        while len(design) < count and any((files[idx] for idx in spin)):
            for idx in spin:
                if files[idx] and len(design) < count:
                    design.append((by_index_b[idx], files[idx].pop(0)))
        return design

    def design_queries(self, count: int, machine: str=BLUEPRINT_ENGINE) -> list[tuple[Article, str]]:
        if 0 >= count:
            return []
        design: list[tuple[Article, str]] = []

        def seized(article: Article, aspect: str) -> bool:
            if any((_Text._nigh(aspect, xa) for xa in article.petitioned)):
                return True
            if any((_Text._nigh(aspect, xd) or _Text._akin(aspect, xd) >= 0.5 for xd in self.resolved_characteristics(article))):
                return True
            if any((_Text._nigh(aspect, stp.aspect) for stp in article.stipulations)) or self.related_to_previously(article, aspect):
                return True
            return any((art is article and _Text._nigh(aspect, pp) for (art, pp) in design))
        if not getattr(self, 'mapped_out', None):
            segments = []
            for article in self.items:
                segments.append({'item': article.moniker, 'category': article.classification, 'known_requirements': article.demand_wordings(), 'decided_by_the_request': self.resolved_characteristics(article), 'hints': article.signals, 'already_asked': article.petitioned, 'property_menu': self.characteristic_bill_of_fare(article)})
            grand_total = count if not self.missive_purse is not None else self.missive_purse
            client = 'REQUEST:\n' + self.appeal + '\n\nSUBJECTS THE REQUEST TELLS THE ASSISTANT TO RAISE: ' + json.dumps(self.petition_concerning) + '\n\nITEMS:\n' + json.dumps(segments, ensure_ascii=False) + f'\n\nQUESTION_COUNT = {max(grand_total, count)}'
            rejoinder = self.model.json([machine, STANDBY_ENGINE if not QUICK_ENGINE != machine else QUICK_ENGINE], BLUEPRINT_INSTRUCTION, client, 700)
            purchased: list[tuple[Article, str]] = []
            if isinstance(rejoinder, dict) and isinstance(rejoinder.get('questions'), list):
                for srch in rejoinder['questions']:
                    if not isinstance(srch, dict) or not srch.get('property'):
                        continue
                    destination = max(self.items, key=lambda art: _Text._akin(art.moniker, srch.get('item') or art.moniker))
                    aspect = re.sub('\\s+', ' ', str(srch['property']).replace('_', ' ')).strip().lower()
                    if aspect and (not any((art is destination and _Text._nigh(aspect, pp) for (art, pp) in purchased))):
                        purchased.append((destination, aspect))
            self.mapped_out = purchased
            self.jot('plan_questions', questions=[(art.moniker, pp) for (art, pp) in purchased])
        if len(self.items) > 1:
            return self.plural_design(count, seized)
        for (article, aspect) in self.mapped_out:
            if len(design) >= count:
                break
            if not seized(article, aspect):
                design.append((article, aspect))
        if 3 <= count:
            summit_n = 1 if not len(self.items) == 1 else 2
            for article in self.items:
                summits = [err['property'] for err in self.characteristic_bill_of_fare(article) if err['listings'] >= 3][:summit_n]
                for (kk, aspect) in enumerate(summits):
                    if seized(article, aspect):
                        continue
                    pos = min(len(design), len(self.items) if not len(self.items) == 1 else 1 + 2 * kk)
                    design.insert(pos, (article, aspect))
            design = design[:count]
        if len(design) < count:
            bills = {art.index: [err for err in self.characteristic_bill_of_fare(art) if err['listings'] >= 2 or err['of'] <= 2] for art in self.items}
            purchase = sorted(self.items, key=lambda art: (sum((1 for (xh, _) in design if xh is art)) + len(art.petitioned), art.index))
            while len(design) < count:
                joined = False
                for article in purchase:
                    while bills[article.index]:
                        aspect = bills[article.index].pop(0)['property']
                        if seized(article, aspect):
                            continue
                        design.append((article, aspect))
                        joined = True
                        break
                    if len(design) >= count:
                        break
                if not joined:
                    break
        return design[:count]

    def resolved_characteristics(self, article: Article) -> list[str]:
        away = []
        for xc in getattr(self, 'clauses', []):
            if xc['src'] is article:
                away.append(xc['prop'])
            away += [str(xb['property']) for (_, xd, xb) in xc['branches'] if xd is article]
        if any((xa is article for (xa, _) in self.conform_couplings())):
            away.append('compatibility')
        return list(dict.fromkeys(away))

    def peruse_signals(self) -> None:
        segments = []
        for article in self.items:
            segments.append({'item': article.moniker, 'category': article.classification, 'known_requirements': article.demand_wordings(), 'hints': article.signals, 'property_menu': [err['property'] for err in self.characteristic_bill_of_fare(article)]})
        client = 'REQUEST:\n' + self.appeal + '\n\nSUBJECTS THE REQUEST TELLS THE ASSISTANT TO RAISE: ' + json.dumps(self.petition_concerning) + '\n\nITEMS:\n' + json.dumps(segments, ensure_ascii=False)
        rejoinder = self.model.json([BLUEPRINT_ENGINE, QUICK_ENGINE], BLUEPRINT_INSTRUCTION, client, 500)
        self.signalled: list[tuple[Article, str]] = []
        self.clutter: set[str] = set()
        if not isinstance(rejoinder, dict):
            return
        for srch in rejoinder.get('hinted') or []:
            if not isinstance(srch, dict) or not srch.get('property'):
                continue
            destination = max(self.items, key=lambda art: _Text._akin(art.moniker, srch.get('item') or art.moniker))
            aspect = re.sub('\\s+', ' ', str(srch['property']).replace('_', ' ')).strip().lower()
            if aspect and (not any((art is destination and _Text._nigh(pp, aspect) for (art, pp) in self.signalled))):
                self.signalled.append((destination, aspect))
        self.clutter = {re.sub('\\s+', ' ', str(jdx).replace('_', ' ')).strip().lower() for jdx in rejoinder.get('junk') or [] if jdx}
        self.jot('hints', hinted=[(art.moniker, pp) for (art, pp) in self.signalled], junk=sorted(self.clutter)[:12])

    def characteristic_bill_of_fare(self, article: Article) -> list[dict[str, Any]]:
        unfolded = [pst for pst in article.reservoir.values() if pst.unfolded and self.classification_ok(article, pst)]
        breathing = [pst for pst in unfolded if not (article.findings.get(pst.id) or {}).get('failed')]
        if len(breathing) >= 4:
            unfolded = breathing
        apiece: list[dict[str, str]] = []
        for pst in unfolded:
            xf: dict[str, str] = {}
            for (kk, vv) in pst.particulars.items():
                clef = re.sub('(?<=[a-z0-9])(?=[A-Z])', ' ', str(kk))
                clef = re.sub('\\s+', ' ', clef.replace('_', ' ').replace('.', ' ')).strip().lower()
                if clef and clef not in xf:
                    xf[clef] = str(vv).strip().lower()[:60]
            apiece.append(xf)
        quantities: dict[str, int] = {}
        for xf in apiece:
            for kk in xf:
                quantities[kk] = quantities.get(kk, 0) + 1
        keys = sorted(quantities, key=lambda kk: (-quantities[kk], len(kk), kk))
        nickname: dict[str, str] = {}
        for (idx, xa) in enumerate(keys):
            if xa in nickname:
                continue
            for xb in keys[idx + 1:]:
                if xb in nickname:
                    continue
                twin = [xf for xf in apiece if xa in xf and xb in xf]
                if len(twin) >= 2 and sum((1 for xf in twin if xf[xa] == xf[xb])) >= 0.8 * len(twin):
                    nickname[xb] = xa
        stable = {stp.clef() for stp in article.stipulations} | set(article.petitioned)
        bill_of_fare = []
        for clef in keys:
            if clef in nickname:
                continue
            monikers = [clef] + [xb for (xb, xa) in nickname.items() if clef == xa]
            if any((_Text._normalize(cnt) in stable or any((_Text._akin(cnt, xf) >= 0.67 for xf in stable)) for cnt in monikers)):
                continue
            if _Text._counters(clef) & _NON_PETITIONED:
                continue
            bearers = sum((1 for xf in apiece if any((cnt in xf for cnt in monikers))))
            values: list[str] = []
            for xf in apiece:
                vv = next((xf[cnt] for cnt in monikers if cnt in xf), None)
                if vv and vv not in values:
                    values.append(vv)
            mass = bearers * (1.0 if not (len(values) == 1 and 3 <= bearers) else 0.5)
            separate = len({next((xf[cnt] for cnt in monikers if cnt in xf), None) for xf in apiece} - {None})
            if 4 <= bearers and bearers <= separate:
                mass *= 0.5
            bill_of_fare.append({'property': clef, 'listings': bearers, 'of': len(unfolded), 'values': values[:3], **({} if not len(monikers) > 1 else {'same_as': monikers[1:4]}), '_w': mass})
        bill_of_fare.sort(key=lambda err: (-err['_w'], err['property']))
        for err in bill_of_fare:
            err.pop('_w', None)
        return bill_of_fare[:40]

    def carry_out_retort_held_on(self, article: Article, aspect: str, rejoinder: dict[str, Any] | None) -> None:
        leftover = 2 if not self.missive_purse is not None else self.missive_purse - self.messages_transmitted
        self.allusion_minimum = max(0, leftover) + self.terminus_fallback()
        try:
            self.carry_out_retort(article, aspect, rejoinder)
        finally:
            self.allusion_minimum = 0

    def carry_out_retort(self, article: Article, aspect: str, rejoinder: dict[str, Any] | None) -> None:
        if not rejoinder:
            return
        for pace in rejoinder.get('steps') or []:
            if isinstance(pace, dict):
                self.append_pace(pace)
        genus = str(rejoinder.get('kind') or '').lower()
        purse = _Text._digit(rejoinder.get('budget'))
        if purse is not None and 0 < purse:
            self.lid = purse
            self.jot('budget', cap=purse)
        if 'need' == genus and rejoinder.get('value') or ('need' == genus and rejoinder.get('reference')):
            pp = str(rejoinder.get('property') or aspect)
            demand = Stipulation(pp, rejoinder.get('op') or 'is', rejoinder.get('value') or '', rejoinder.get('reference'), provenance='shopper')
            if demand.citation:
                demand.citation['property'] = demand.aspect
                imprint = _Shape._allusion_imprint(demand.citation)
                if imprint and any((imprint in (getattr(stp, 'allusion_imprint', None), getattr(stp, 'allusion_imprint0', None)) or (stp.citation and _Shape._allusion_imprint(stp.citation) == imprint) for stp in article.stipulations)):
                    self.jot('reference_repeat', item=article.moniker, prop=demand.aspect)
                    return
            if demand.worth and any(('shopper' == stp.provenance and _Text._normalize(stp.worth) == _Text._normalize(demand.worth) and (demand.operator_ == stp.operator_) for stp in article.stipulations)):
                return
            article.stipulations = [stp for stp in article.stipulations if not (stp.clef() == demand.clef() and 'shopper' == stp.provenance)]
            article.stipulations.append(demand)
            if demand.operator_ in ('is', 'one of') and demand.worth and (not demand.citation) and (len(demand.worth) <= 40):
                article.inquiry_subsequent(f"{demand.worth.split('|')[0].strip()} {article.substantive}")
            if demand.citation and demand.operator_ in ALLUSION_OPERATORS:
                self.work_out_citation(article, demand)
        elif 'free' == genus:
            article.loosed.append(_Text._normalize(str(rejoinder.get('property') or aspect)))

    def characteristic_titled(self, article: Article, retort: str, petitioned: list[str], worth: str) -> str | None:
        wording = ' ' + _Text._normalize(retort) + ' '
        if any((xa and _Text._normalize(xa) and (' ' + _Text._normalize(xa) + ' ' in wording) for xa in petitioned)):
            return None
        keys: set[str] = set()
        for pst in article.reservoir.values():
            for kk in pst.particulars:
                clef = _Text._normalize(re.sub('(?<=[a-z0-9])(?=[A-Z])', ' ', str(kk)))
                if len(clef) >= 4 and (not _Text._counters(clef) & _Text._counters(worth)) and (not any((_Text._nigh(clef, xa) for xa in petitioned if xa))):
                    keys.add(clef)
        finds = [kk for kk in keys if ' ' + kk + ' ' in wording]
        finds = [kk for kk in finds if not any((xo != kk and ' ' + kk + ' ' in ' ' + xo + ' ' for xo in finds))]
        return None if not len(finds) == 1 else finds[0]

    @staticmethod
    def related_to_previously(article: Article, aspect: str) -> bool:
        moniker = _Text._normalize(aspect)
        return len(moniker.split()) >= 2 and any((f' {moniker} ' in strand for strand in getattr(article, 'related_to_strands', [])))

    def petition(self, article: Article, aspect: str, thin_down: bool=False, as_well: str | None=None) -> dict[str, Any] | None:
        details = self.functions['message']
        substantive = article.moniker
        wording = f'Which {aspect} would you like the {substantive} to have?'
        if as_well:
            wording = f'For the {substantive}, which {aspect} and which {as_well} would you like?'
        if thin_down:
            spotted = []
            for pst in article.reservoir.values():
                for (kk, vv) in pst.particulars.items():
                    if _Text._nigh(re.sub('(?<=[a-z0-9])(?=[A-Z])', ' ', str(kk)).replace('_', ' '), aspect) and str(vv) not in spotted:
                        spotted.append(str(vv)[:30])
            instances = ' or '.join(spotted[:2])
            wording = f'Just this one thing: for the {substantive}, what {aspect} do you need' + ('?' if not instances else f', for example {instances}?')
        wording = self._clip('message', details['text'], wording)
        result = self.dispatch(details['name'], {details['text']: wording})
        self.messages_transmitted += 1
        if not thin_down:
            article.petitioned.append(_Text._normalize(aspect))
            if as_well:
                article.petitioned.append(_Text._normalize(as_well))
        if not result['ok']:
            return None
        observation = result['obs']
        retort = None
        if isinstance(observation, str):
            retort = observation
        elif isinstance(observation, dict):
            threads = [vv for vv in observation.values() if isinstance(vv, str)]
            retort = _Text._dense(observation, 400) if not threads else max(threads, key=len)
        if not retort:
            return None
        noted = [f'{stp.aspect}: {stp.operator_} {stp.worth}' for stp in article.stipulations if 'shopper' == stp.provenance and stp.worth]
        client = f"ITEM: {substantive}\nPROPERTY ASKED ABOUT: {(aspect if not as_well else aspect + ' or ' + as_well)}\nQUESTION: {wording}\nKNOWN NEEDS: {json.dumps(noted)}\nREPLY: {retort}"
        rejoinder = self.model.json([QUICK_ENGINE, STANDBY_ENGINE, BLUEPRINT_ENGINE], RETORT_INSTRUCTION, client, 600)
        if isinstance(rejoinder, dict) and str(rejoinder.get('kind') or '').lower() == 'need':
            titled = self.characteristic_titled(article, retort, [aspect, as_well or '', str(rejoinder.get('property') or '')], str(rejoinder.get('value') or ''))
            if titled:
                rejoinder['property'] = titled
        self.jot('reply', item=substantive, prop=aspect, reply=retort[:200], read=_Text._dense(rejoinder, 300))
        if isinstance(rejoinder, dict) and str(rejoinder.get('kind') or '').lower() in ('need', 'repeat'):
            article.related_to_strands = list(getattr(article, 'related_to_strands', [])) + [' ' + _Text._normalize(retort) + ' ']
        return None if not isinstance(rejoinder, dict) else rejoinder

class _Messaging:
    """The message allowance, the process checkpoint, basket confirmation and shopper lines."""

    def append_pace(self, pace: dict[str, Any]) -> None:
        if not pace.get('goal') or len(self.protocol) >= 8:
            return
        pace = dict(pace)
        pace['moment'] = str(pace.get('moment') or 'before_finish').lower()
        pace['kind'] = 'tool' if not str(pace.get('kind') or '').lower() == 'message' else 'message'
        if any((_Text._akin(pace['goal'], state_s.get('goal')) >= 0.6 for state_s in self.protocol)):
            return
        self.protocol.append(pace)
        self.ratify_initial = self.ratify_initial or (pace['kind'] == 'message' and pace['moment'] in ('before_order', 'before_finish'))
        self.jot('step_added', step=pace)

    def knob_strands(self) -> bool:
        strands = [pst for pst in self.outstanding_strands if pst not in self.dealt_strands]
        self.outstanding_strands = []
        if not strands:
            return False
        for pst in strands:
            self.dealt_strands.add(pst)
        client = 'LINES:\n- ' + '\n- '.join(strands) + '\n\nREQUESTED ITEMS: ' + json.dumps([idx.moniker for idx in self.items])
        rejoinder = self.model.json([QUICK_ENGINE, STANDBY_ENGINE], INCIDENT_INSTRUCTION, client, 700)
        self.jot('event', lines=[pst[:160] for pst in strands], read=_Text._dense(rejoinder, 300))
        altered = False
        if not isinstance(rejoinder, dict):
            return False
        for chr_ in rejoinder.get('changes') or []:
            if not isinstance(chr_, dict):
                continue
            genus = str(chr_.get('kind') or '').lower()
            destination = None
            if chr_.get('item'):
                destination = max(self.items, key=lambda art: _Text._akin(art.moniker, chr_.get('item')))
                if _Text._akin(destination.moniker, chr_.get('item')) < 0.2 and len(self.items) > 1:
                    destination = None
            if 'budget' == genus and _Text._digit(chr_.get('budget')):
                purse = _Text._digit(chr_.get('budget'))
                inside_strands = [float(xx.replace(',', '')) for xx in re.findall('\\d[\\d,]*(?:\\.\\d+)?', ' '.join(strands))]
                if purse not in inside_strands:
                    for substitute in (purse * 100, purse / 100):
                        if substitute in inside_strands:
                            purse = substitute
                            break
                    else:
                        if inside_strands and self.lid is not None:
                            nigh = [xx for xx in inside_strands if self.lid / 20 <= xx <= self.lid * 20]
                            if nigh:
                                purse = nigh[0]
                self.lid = purse
                altered = True
            elif 'drop' == genus and chr_.get('property'):
                for article in self.items if not destination else [destination]:
                    prior_to = len(article.stipulations)
                    article.stipulations = [stp for stp in article.stipulations if _Text._akin(stp.aspect, chr_['property']) < 0.5]
                    if len(article.stipulations) != prior_to:
                        altered = True
                        for vv in article.findings.values():
                            vv['failed'] = {xf for xf in vv['failed'] if _Text._akin(xf.split(':')[0], chr_['property']) < 0.5}
            elif 'add' == genus and chr_.get('property') and chr_.get('value'):
                for article in self.items[:1] if not destination else [destination]:
                    article.stipulations.append(Stipulation(chr_['property'], chr_.get('op') or 'is', chr_['value'], None, provenance='shopper'))
                    altered = True
            elif genus in ('price', 'stock'):
                altered = True
                self.update_trolley_items()
            elif 'step' == genus and isinstance(chr_.get('step'), dict):
                self.append_pace(chr_['step'])
        return altered

    def ratify_pannier(self) -> bool:
        if not self.able_missive() or self.paces_leftover() < 2:
            return False
        details = self.functions['message']
        (components, grand_total) = ([], 0.0)
        for (cap, units_n) in self.pannier_imprint():
            article = self.trolley_article.get(cap)
            posting = None if not article else article.reservoir.get(cap)
            title = re.sub('\\s+', ' ', (cap if not posting else posting.title) or cap).strip()[:70]
            tariff = None if not (posting and posting.tariff is not None) else posting.tariff
            if tariff is not None:
                grand_total += tariff * units_n
            components.append(f'{units_n} x {title}' + ('' if not tariff is not None else f' ({tariff:g} each)'))
        wording = 'Your final basket: ' + '; '.join(components) + ('.' if not grand_total else f'. Total {grand_total:g}.') + ' Shall I check out this basket? Please confirm with a yes.'
        wording = self._clip('message', details['text'], wording)
        result = self.dispatch(details['name'], {details['text']: wording})
        self.messages_transmitted += 1
        self.ratified_imprint = self.pannier_imprint()
        self.jot('confirm_asked', basket=self.ratified_imprint, ok=result['ok'], obs=_Text._dense(result['obs'], 160))
        return True

    def pannier_imprint(self) -> tuple:
        return tuple(sorted(((cap, int(srch or 0)) for (cap, srch) in self.trolley.items() if srch)))

    def barrier_check(self, juncture: str, purpose: str) -> bool | None:
        if self.barrier_check_dispatches.get(juncture, 0) >= 3 or self.paces_leftover() < 2 or self.duration_leftover() < ENFOLD_SECS:
            return False
        self.barrier_check_dispatches[juncture] = self.barrier_check_dispatches.get(juncture, 0) + 1
        terminating = {(self.functions.get(stp) or {}).get('name') for stp in ('checkout', 'decline', 'cart_add', 'cart_set')} - {None}
        utensils = [{'name': cnt, 'description': self.utensil_portrayal(cnt), 'parameters': self.outlines[cnt]} for cnt in self.outlines if cnt not in terminating]
        pannier = []
        for (cap, units_n) in self.pannier_imprint():
            article = self.trolley_article.get(cap)
            pst = None if not article else article.reservoir.get(cap)
            pannier.append({'id': cap, 'quantity': units_n, 'title': ('' if not pst else pst.title)[:120], 'price': None if not pst else pst.tariff, 'for': None if not article else article.moniker})
        grand_total = sum(((xb['price'] or 0) * xb['quantity'] for xb in pannier))
        leftover_dispatch_notes = None if not self.missive_purse is not None else self.missive_purse - self.messages_transmitted
        client = 'REQUEST:\n' + self.appeal + '\n\nSTEPS THE REQUEST DEMANDS:\n' + json.dumps(self.protocol, ensure_ascii=False) + f'\n\nMOMENT: {juncture} (the assistant is about to {purpose})' + '\n\nBASKET NOW: ' + json.dumps({'lines': pannier, 'total': grand_total}, ensure_ascii=False) + f"\nMESSAGES LEFT: {('no limit' if not leftover_dispatch_notes is not None else leftover_dispatch_notes)}; STEPS LEFT: {self.paces_leftover()}" + '\n\nTOOLS:\n' + json.dumps(utensils, ensure_ascii=False) + '\n\nDONE SO FAR (tool calls and results, oldest first):\n' + json.dumps(self.annals[-30:], ensure_ascii=False)
        rejoinder = self.model.json([BLUEPRINT_ENGINE, STANDBY_ENGINE], BARRIER_INSTRUCTION, client, 900)
        if not isinstance(rejoinder, dict):
            self.jot('gate_failed', moment=juncture)
            return None
        went = False
        operations = [xa for xa in (rejoinder.get('actions') or [])[:2] if isinstance(xa, dict)]
        dispatch_note_utensil = (self.functions.get('message') or {}).get('name')
        for (kk, operate) in enumerate(operations):
            if operate.get('tool') not in self.outlines or operate.get('tool') in terminating:
                continue
            moniker = operate['tool']
            if dispatch_note_utensil != moniker:
                payable = max(sum((1 for xa in operations[kk + 1:] if xa.get('tool') == dispatch_note_utensil)), self.outstanding_messages())
                if self.paces_leftover() < 2 + payable:
                    self.jot('gate_step_skipped', moment=juncture, tool=moniker, steps=self.paces_leftover(), due=payable)
                    continue
            outline = self.outlines[moniker]
            characteristics = outline.get('properties') or {}
            operands = {} if not isinstance(operate.get('args'), dict) else {kk: vv for (kk, vv) in (operate.get('args') or {}).items() if kk in characteristics}
            for (kk, vv) in list(operands.items()):
                enum = (characteristics.get(kk) or {}).get('enum')
                if isinstance(enum, list) and enum and (vv not in enum):
                    operands.pop(kk)
                elif isinstance(vv, str):
                    operands[kk] = self._clip_operands(characteristics.get(kk) or {}, vv)
            if any((kk not in operands for kk in outline.get('required') or [])):
                continue
            missive = moniker == (self.functions.get('message') or {}).get('name')
            if missive and (not self.able_missive()):
                continue
            if self.paces_leftover() < 2:
                break
            result = self.dispatch(moniker, operands)
            if missive:
                self.messages_transmitted += 1
                self.ratified_imprint = self.pannier_imprint()
            self.jot('gate_step', moment=juncture, tool=moniker, why=str(operate.get('why') or '')[:160], ok=result['ok'])
            went = True
        if not went:
            self.jot('gate_clear', moment=juncture)
        return went

    def able_petition(self) -> bool:
        if not self.able_missive():
            return False
        return self.missive_purse is None or self.messages_transmitted < self.missive_purse - self.withheld_messages()

    def outstanding_messages(self) -> int:
        return self.withheld_messages() if not (self.trolley and self.ratified_imprint == self.pannier_imprint()) else 0

    def withheld_messages(self) -> int:
        if 'message' not in self.functions or not self.protocol:
            return 0

        def necessity(junctures: tuple[str, ...]) -> int:
            return sum((max(1, int(state_s.get('messages') or 1)) for state_s in self.protocol if state_s['kind'] == 'message' and state_s['moment'] in junctures))
        withheld = necessity(('before_order', 'before_finish'))
        if self.missive_purse is not None:
            withheld = min(withheld, max(0, self.missive_purse - self.messages_transmitted))
        return withheld

    def able_missive(self) -> bool:
        if 'message' not in self.functions:
            return False
        if self.missive_purse is not None and self.missive_purse <= self.messages_transmitted:
            return False
        return self.paces_leftover() > 0

class _Checks:
    """The last checks before buying: validation, ranking and a search audit."""

    def inspection(self, lacking: list[Article]) -> bool:
        performed = False
        self.fallback_overrule = len(self.items) + 3 + max(self.withheld_messages(), 0 if not self.ratify_initial else 1)
        try:
            for article in lacking:
                if self.duration_leftover() < ENFOLD_SECS + 20:
                    break
                dockets = []
                reservoir = [pst for pst in article.reservoir.values() if self.classification_ok(article, pst) and pst.inventory is not False and (not (self.lid is not None and pst.tariff is not None and (pst.tariff * article.headcount > self.lid + 1e-06)))]
                for pst in sorted(reservoir, key=lambda pst: (not pst.unfolded, 1e+18 if not pst.tariff is not None else pst.tariff))[:40]:
                    docket: dict[str, Any] = {'id': pst.id, 'title': pst.title[:110], 'price': pst.tariff, 'read': pst.unfolded}
                    if not pst.unfolded:
                        docket['text'] = (pst.wording or '')[:160]
                    else:
                        vv = article.findings.get(pst.id) or {}
                        docket['failed'] = sorted(vv.get('failed') or [])[:3]
                        docket['unsure'] = sorted(vv.get('unsure') or [])[:3]
                        docket['properties'] = dict(list(pst.particulars.items())[:16])
                        if pst.jottings:
                            docket['notes'] = pst.jottings[:200]
                    dockets.append(docket)
                if not dockets:
                    continue
                tested = sorted({kk.split(':', 1)[1] for kk in article.pointer_poss if kk.startswith('search_text:')})
                client = f"PRODUCT: {article.moniker} (category: {article.classification}; quantity {article.headcount})\nBUDGET, ALL LINES TOGETHER: {('none stated' if not self.lid is not None else self.lid)}\nREQUIREMENTS:\n- " + ('\n- '.join(article.demand_wordings()) or 'none') + f'\nQUERIES TRIED: {json.dumps(tested, ensure_ascii=False)}\n\nLISTINGS SEEN:\n' + _Text._dense(dockets, 14000)
                rejoinder = self.model.json([BLUEPRINT_ENGINE, QUICK_ENGINE], INSPECTION_INSTRUCTION, client, 1500)
                if not isinstance(rejoinder, dict):
                    continue
                self.jot('audit', item=article.moniker, answer=_Text._dense(rejoinder, 300))
                for cap in [str(xx) for xx in rejoinder.get('fits_after_all') or []][:2]:
                    pst = article.reservoir.get(cap)
                    if pst is None or not pst.unfolded or self.column_falls_short(article, pst):
                        continue
                    if self._validate(article, pst, rigid=True):
                        article.findings[pst.id] = {'failed': set(), 'unsure': set(), 'weak': set(), 'loose': set()}
                        article.evaluated_per[pst.id] = set(article.demand_wordings())
                        performed = True
                newfound = [str(xx) for xx in rejoinder.get('unopened') or [] if str(xx) in article.reservoir and (not article.reservoir[str(xx)].unfolded)][:4]
                if newfound and self.paces_leftover() - self.terminus_fallback() >= 1:
                    if self.unfold_lot(article, newfound, minimum=self.terminus_fallback()):
                        performed = True
                for srch in [str(xx).strip() for xx in rejoinder.get('queries') or [] if str(xx).strip()][:2]:
                    if self.paces_leftover() - self.terminus_fallback() < 3 or self.duration_leftover() < ENFOLD_SECS + 15:
                        break
                    if _Text._normalize(srch) in {_Text._normalize(tl) for tl in tested} or self.search(article, inquiry=srch[:60], upper_folios=1, purse=self.terminus_fallback()) <= 0:
                        continue
                    lines = [pst for pst in self.aspirants(article) if not pst.unfolded]
                    self.sieve(article, lines)
                    lines = [pst for pst in self.aspirants(article) if not pst.unfolded and 'no' != pst.sieve][:2]
                    if lines and self.unfold_lot(article, [pst.id for pst in lines], minimum=self.terminus_fallback()):
                        performed = True
                self.evaluate(article, [pst for pst in self.aspirants(article) if pst.unfolded])
        finally:
            self.fallback_overrule = None
        return performed

    def order_by_finest(self, article: Article, hopefuls: list[Posting]) -> list[Posting]:
        if not self.finest or len(hopefuls) < 2:
            return hopefuls
        hoarded = getattr(self, 'finest_purchase', {}).get(article.index)
        numbers = [pst.id for pst in hopefuls]
        if hoarded is None or hoarded[0] != numbers:
            lines = [{'id': pst.id, 'title': pst.title[:120], 'price': pst.tariff, 'properties': pst.particulars, 'description': str(pst.wording or '')[:300]} for pst in hopefuls[:12]]
            client = f'MEASURE: {self.finest}\nPRODUCT: {article.moniker}\nLISTINGS: ' + _Text._dense(lines, 9000)
            rejoinder = self.model.json([BLUEPRINT_ENGINE, STANDBY_ENGINE], ORDERBY_INSTRUCTION, client, 600)
            purchase = [] if not isinstance(rejoinder, dict) else [str(xx) for xx in rejoinder.get('order') or []]
            self.jot('best_rank', measure=self.finest, order=purchase[:12])
            hoarded = (numbers, purchase)
            if purchase:
                self.finest_purchase = dict(getattr(self, 'finest_purchase', {}))
                self.finest_purchase[article.index] = hoarded
        purchase = hoarded[1]
        pos = {cap: idx for (idx, cap) in enumerate(purchase)}
        return sorted(hopefuls, key=lambda pst: pos.get(pst.id, len(pos) + numbers.index(pst.id)))

    def _validate(self, article: Article, posting: Posting, rigid: bool=False) -> bool:
        asks = article.demand_wordings()
        client = f"REQUESTED PRODUCT: {article.classification} (quantity {article.headcount})\nBUDGET, ALL LINES TOGETHER: {('none stated' if not self.lid is not None else self.lid)}\nREQUIREMENTS:\n- " + ('(none beyond the product kind, budget and availability)' if not asks else '\n- '.join(asks)) + '\n\nLISTING:\n' + json.dumps(posting.ledger(3500), ensure_ascii=False)
        rejoinder = self.model.json([VALIDATE_ENGINE, QUICK_ENGINE], VALIDATE_INSTRUCTION, client, 800)
        if not isinstance(rejoinder, dict):
            return not rigid
        if rigid and rejoinder.get('fits') is not True:
            return False
        snags = [str(pp) for pp in rejoinder.get('problems') or []]
        self.jot('verify', id=posting.id, fits=rejoinder.get('fits'), problems=snags[:4])
        if rejoinder.get('fits') is False and snags:
            vv = article.findings.setdefault(posting.id, {'failed': set(), 'unsure': set()})
            for pp in snags:
                finest = None if not asks else max(asks, key=lambda stp: _Text._akin(stp, pp))
                vv['failed'].add(pp if not (finest and _Text._akin(finest, pp) > 0.2) else finest)
            return False
        return True

    def validate(self, article: Article, posting: Posting) -> bool:
        clef = (article.index, posting.id, tuple(article.demand_wordings()), json.dumps(posting.particulars, sort_keys=True, default=str))
        hoard = self.__dict__.setdefault('verified', {})
        if clef not in hoard:
            hoard[clef] = self._validate(article, posting)
        return hoard[clef]

class _Verdicts:
    """Field-by-field and model verdicts on postings."""

    def evaluate_whole(self) -> None:
        _Pool._concurrent([lambda art=art: self.evaluate(art, [pst for pst in self.aspirants(art) if pst.unfolded]) for art in self.items])

    def column_strands(self, matrix: list[dict[str, Any]]) -> str:
        strands = []
        for line in matrix:
            spotted = ' | '.join((f'{vv} ({cnt})' for (vv, cnt) in sorted(line['values'].items(), key=lambda xx: -xx[1])[:8]))
            strands.append(f"- {line['req']}: field {', '.join((repr(kk) for kk in line['keys'][:3]))} on {line['carriers']} of {line['read']} read listings; values written: {spotted}; {len(line['writers'])} listing(s) write the required value itself")
        return '\n'.join(strands)

    def column_falls_short(self, article: Article, posting: Posting, matrix: list[dict[str, Any]] | None=None) -> set[str]:
        away = set()
        for line in matrix if not matrix is None else self.column_matrix(article):
            native = [kk for kk in posting.particulars if _Text._nigh(kk, line['prop'])]
            if not native or not line['writers'] - {posting.id}:
                continue
            members = [frozenset(_Text._counters(err)) for kk in native for err in re.split('[,;/|]', str(posting.particulars[kk])) if _Text._counters(err)]
            trunk = lambda vocables: {ww[:4] for ww in vocables}
            if members and (not any((trunk(ww) <= trunk(err) for ww in line['wants'] for err in members))):
                away.add(line['req'])
        return away

    def column_matrix(self, article: Article) -> list[dict[str, Any]]:
        peruse = [pst for pst in article.reservoir.values() if pst.unfolded and pst.particulars and self.classification_ok(article, pst)]
        if len(peruse) < 3:
            return []
        away = []
        for stp in article.running():
            if stp.operator_ not in ('is', 'one of') or not stp.worth or re.search('\\d', stp.worth):
                continue
            cravings = [frozenset(_Text._counters(ww)) for ww in stp.worth.split('|') if _Text._counters(ww)]
            if not cravings:
                continue
            keys: set[str] = set()
            values: dict[str, int] = {}
            (bearers, scribes) = (0, set())
            for pst in peruse:
                catch = [kk for kk in pst.particulars if _Text._nigh(kk, stp.aspect)]
                if not catch:
                    continue
                bearers += 1
                keys.update(catch)
                for kk in catch:
                    for err in re.split('[,;/|]', str(pst.particulars[kk])):
                        err = ' '.join(err.lower().split())
                        if err:
                            values[err] = values.get(err, 0) + 1
                            if any((ww <= frozenset(_Text._counters(err)) for ww in cravings)):
                                scribes.add(pst.id)
            if bearers:
                away.append({'req': stp.wording(), 'prop': stp.aspect, 'keys': sorted(keys), 'carriers': bearers, 'read': len(peruse), 'values': values, 'wants': cravings, 'writers': scribes})
        return away

    @staticmethod
    def unspoken(article: Article, posting: Posting, demand: Stipulation) -> bool:
        if demand.operator_ not in ('is', 'one of') or 'fit' == demand.provenance or (not demand.worth):
            return False
        if any((_Text._nigh(str(kk).replace('-', ' '), demand.aspect) or _Text._akin(str(kk).replace('-', ' ').replace('_', ' '), demand.aspect) >= 0.5 for kk in posting.particulars)):
            return False
        usual = _Text._vocables(article.moniker) | _Text._vocables(article.classification)
        substitutes = [ww for ww in ({xx for xx in _Text._vocables(vv) - usual if not (xx.isalpha() and len(xx) <= 3)} for vv in demand.worth.split('|')) if ww]
        if not substitutes:
            return False
        wording = {ww[:4] for ww in _Text._vocables(' '.join([posting.title or '', posting.wording or '', posting.jottings or '', posting.classification or '', ' '.join((f'{kk} {vv}' for (kk, vv) in posting.particulars.items()))]))}
        return not any(({ww[:4] for ww in xa} & wording for xa in substitutes))

    def evaluate(self, article: Article, postings: list[Posting]) -> None:
        self.let_go_caption_repeats(article)
        self.square_characteristics(article)
        asks = article.demand_wordings()
        if not asks:
            for pst in postings:
                if pst.unfolded and pst.id not in article.findings:
                    article.findings[pst.id] = {'failed': set(), 'unsure': set()}
                    article.evaluated_per[pst.id] = set()
            return
        to_do = [pst for pst in postings if pst.unfolded and (not article.evaluated_per.get(pst.id, set()) >= set(asks))]
        for pst in list(to_do):
            earlier = article.findings.get(pst.id)
            if pst.id in article.evaluated_per and earlier and set(earlier.get('failed') or ()) & set(asks):
                article.evaluated_per[pst.id] = set(asks)
                to_do.remove(pst)
        if not to_do or self.duration_leftover() < ENFOLD_SECS:
            return
        matrix = self.column_matrix(article)
        pillars = '' if not matrix else "\n\nSTORE FIELDS (from this store's records read so far):\n" + self.column_strands(matrix)
        (lots, instructions) = ([], [])
        for start in range(0, len(to_do), 4):
            lot = to_do[start:start + 4]
            ledgers = []
            for pst in lot:
                cell = pst.ledger()
                kindred = self.kindred_folios(article, pst)
                if kindred:
                    cell['related_pages'] = kindred
                ledgers.append(cell)
            lots.append(lot)
            instructions.append(f'REQUESTED PRODUCT: {article.classification}\nREQUIREMENTS (judge each):\n' + '\n'.join((f'R{kk}: {stp}' for (kk, stp) in enumerate(asks, 1))) + pillars + '\n\nLISTINGS:\n' + json.dumps(ledgers, ensure_ascii=False))
        rejoinders = _Pool._concurrent([lambda uu=uu: self.model.json([QUICK_ENGINE, STANDBY_ENGINE], EVALUATE_INSTRUCTION, uu, 2000) for uu in instructions])
        for (lot, rejoinder) in zip(lots, rejoinders):
            gained = {}
            if isinstance(rejoinder, dict) and isinstance(rejoinder.get('verdicts'), list):
                for vv in rejoinder['verdicts']:
                    if isinstance(vv, dict) and vv.get('id') in article.reservoir:
                        gained[vv['id']] = vv
            for pst in lot:
                vv = gained.get(pst.id)
                if vv is None:
                    continue
                fell_short = {str(xx) for xx in vv.get('failed') or [] if xx}
                hesitant = {str(xx) for xx in vv.get('unsure') or [] if xx}
                lax = {str(xx) for xx in vv.get('loose') or [] if xx}
                feeble = {str(xx) for xx in vv.get('weak') or [] if xx} | lax

                def normal(monikers: set[str]) -> set[str]:
                    away = set()
                    for cnt in monikers:
                        mm = re.fullmatch('\\s*R?(\\d+)\\s*', cnt)
                        if mm and 1 <= int(mm.group(1)) <= len(asks):
                            away.add(asks[int(mm.group(1)) - 1])
                            continue
                        finest = None if not asks else max(asks, key=lambda stp: _Text._akin(stp, cnt))
                        if not (finest and _Text._akin(finest, cnt) > 0.2):
                            away.add(cnt)
                        else:
                            away.add(finest)
                    return away
                article.findings[pst.id] = {'failed': normal(fell_short) | self.column_falls_short(article, pst, matrix), 'unsure': normal(hesitant), 'weak': normal(feeble), 'loose': normal(lax)}
                article.evaluated_per[pst.id] = set(asks)
                vv = article.findings[pst.id]
                for stp in article.running():
                    tl = stp.wording()
                    if tl not in vv['failed'] and tl not in vv['unsure'] and (tl not in vv['loose']) and self.unspoken(article, pst, stp):
                        vv['unsure'].add(tl)

class _Opening:
    """Opening postings one by one or several at once."""

    def unfold_lot(self, article: Article | None, numbers: list[str], minimum: int=0, folio: bool=False) -> list[Posting]:
        numbers = [idx for idx in numbers if idx]
        unfolded: list[Posting] = []
        if not numbers:
            return unfolded
        collate = None if getattr(self, 'collate_halted', False) else self.functions.get('compare')
        particulars_utensil = self.functions.get('facts')
        unfold_utensil = self.functions.get('open')
        if collate and (not getattr(self, 'collate_probed', False)):
            self.collate_probed = True
            highest = ((self._outline('compare').get('properties') or {}).get(collate.get('ids') or '') or {}).get('maxItems')
            if isinstance(highest, int) and 2 <= highest < self.collate_boundary:
                self.collate_boundary = highest
        pos = 0
        relined: set[str] = set()
        pages = not folio and self.page_just(article)
        while pos < len(numbers) and self.paces_leftover() > 0 and (self.duration_leftover() > ENFOLD_SECS):
            if self.meagre_collate and particulars_utensil and (article is not None) or pages:
                if minimum and self.paces_leftover() - 1 < minimum:
                    break
                gained = self.peruse_particulars(article, numbers[pos])
                pos += 1
                if gained is not None:
                    unfolded.append(gained)
                continue
            if collate and len(numbers) - pos >= 2:
                if minimum and self.paces_leftover() - 1 < minimum:
                    break
                wedge = numbers[pos:pos + self.collate_boundary]
                result = self.dispatch(collate['name'], {collate['ids']: wedge})
                if result['ok'] and _Shape._discover_sequence_of_lookups(result['obs']):
                    lines = [pst for pst in (self.posting_of(cell, 'detail') for cell in _Shape._discover_sequence_of_lookups(result['obs']) or []) if pst is not None]
                    meagre = bool(particulars_utensil) and bool(lines) and (not any((pst.particulars for pst in lines)))
                    if meagre and (not self.meagre_collate):
                        self.meagre_collate = True
                        self.recollect()
                        self.jot('thin_compare', tool=collate['name'], facts_tool=particulars_utensil.get('name'))
                    for posting in lines:
                        posting.folio_peruse = True
                        self.unfolded_folios.add(posting.id)
                        if meagre and article is not None:
                            self.fuse(article, posting, False)
                            continue
                        posting.unfolded = True
                        unfolded.append(posting if not article else self.fuse(article, posting, True))
                    if meagre and article is not None:
                        for posting in lines:
                            if minimum and self.paces_leftover() - 1 < minimum or self.duration_leftover() < ENFOLD_SECS:
                                break
                            gained = self.peruse_particulars(article, posting.id)
                            if gained is not None:
                                unfolded.append(gained)
                    pos += len(wedge)
                    behind = {pst.id for pst in lines}
                    let_gone = [idx for idx in wedge if idx not in behind and idx not in relined]
                    if lines and let_gone and (len(behind) < len(wedge)):
                        if 2 <= len(behind) < self.collate_boundary:
                            self.collate_boundary = len(behind)
                            self.recollect()
                        relined.update(let_gone)
                        numbers = numbers[:pos] + let_gone + numbers[pos:]
                    continue
                if result['error'] or (result['ok'] and (not _Shape._discover_sequence_of_lookups(result['obs']))):
                    if 2 < self.collate_boundary and len(wedge) > 2:
                        self.collate_boundary = max(2, min(self.collate_boundary, len(wedge)) // 2)
                        self.recollect()
                        continue
                    self.collate_halted = True
                    collate = None
                    continue
            if not unfold_utensil:
                break
            expense = 1 if not particulars_utensil else 2
            if minimum and self.paces_leftover() - expense < minimum:
                break
            cap = numbers[pos]
            pos += 1
            result = self.dispatch(unfold_utensil['name'], {unfold_utensil['id']: cap})
            if not result['ok'] or not isinstance(result['obs'], dict):
                continue
            cell = result['obs']
            posting = self.posting_of(cell, 'detail')
            if posting is None:
                posting = Posting(cap)
                posting.untreated = cell
            posting.folio_peruse = True
            noted = None if not article is not None else article.reservoir.get(cap)
            if not posting.particulars and particulars_utensil and (self.paces_leftover() > 0) and (not (noted is not None and noted.particulars)):
                posting.particulars = self.fetch_up_particulars(cap) or {}
            if cap != posting.id:
                posting.id = cap
            posting.unfolded = True
            unfolded.append(posting if not article else self.fuse(article, posting, True))
            self.unfolded_folios.add(cap)
        return unfolded

class _ShelfWork:
    """Which shelves count, which rows are worth reading, and the overall exploration."""

    def heat_up(self, article: Article, ration: int | None=None) -> None:
        if self.paces_leftover() <= self.terminus_fallback() + 2 or 'message' not in self.functions:
            return
        cease_at = self.paces_leftover() - (self.paces_leftover() if not ration is not None else ration)
        minimum = max(cease_at, self.terminus_fallback() + 2)
        destination = 5 if not self.lot_unfold() else 12

        def reserve() -> list[Posting]:
            away = [pst for pst in article.reservoir.values() if not pst.unfolded and self.classification_ok(article, pst) and (pst.inventory is not False)]
            away.sort(key=lambda pst: ('no' == pst.sieve, 'yes' != pst.sieve, 1e+18 if not pst.tariff is not None else pst.tariff))
            return away
        while self.paces_leftover() > minimum and self.duration_leftover() > ENFOLD_SECS + 20 and (not self.expended()):
            unfolded = [pst for pst in article.reservoir.values() if pst.unfolded and self.classification_ok(article, pst)]
            craving = destination - len(unfolded)
            if 0 >= craving:
                return
            to_do = reserve()
            if len(to_do) < 2 and article.search_over_with < len(article.expressions) + 1:
                prior_to = len(article.reservoir)
                self.search(article, inquiry=article.classification, upper_folios=1, purse=minimum)
                to_do = reserve()
                if len(article.reservoir) == prior_to and (not to_do):
                    return
            if not to_do:
                return
            wedge = min(craving, (1 if not self.meagre_collate else 4) if not self.lot_unfold() else self.collate_boundary)
            gained = self.unfold_lot(article, [pst.id for pst in to_do[:wedge]], minimum=minimum)
            if not gained:
                return
            self.evaluate(article, [pst for pst in self.aspirants(article) if pst.unfolded])

    def survey(self, article: Article, paces: int, terminal: bool=False) -> None:
        cease_at = self.paces_leftover() - paces
        expressions = list(article.expressions)
        expression_index = min(article.search_over_with, len(expressions))
        craving_conforms = (4 if not self.thriftiest else 3) if self.thriftiest or self.finest else 2
        while self.paces_leftover() > max(cease_at, self.terminus_fallback()) and self.duration_leftover() > ENFOLD_SECS + 10 and (not self.expended()):
            self.knob_strands()
            hopefuls = self.aspirants(article)
            self.sieve(article, hopefuls)
            hopefuls = self.aspirants(article)
            sealed = [pst for pst in hopefuls if not pst.unfolded]
            self.evaluate(article, [pst for pst in hopefuls if pst.unfolded])
            conforms = article.conforms()
            if terminal and conforms and (not self.trolley) and getattr(self, 'tentative_payable', False):
                self.tentative_trolley()
            if conforms and (not self.thriftiest) and (len(conforms) >= craving_conforms):
                break
            groups = self.worth_bands(article)
            if repr(groups) != getattr(article, 'worth_imprint', None):
                article.worth_imprint = repr(groups)
                article.worth_numbers = set()
                article.worth_entire = False
            if groups:

                def finds(pst: Posting) -> int:
                    if pst.id in article.worth_numbers:
                        return len(groups)
                    cnt = self.line_finds(pst, groups)
                    if self.whole_vocables and cnt < len(groups):
                        transmitted = [_Text._vocables(ss.split(':', 1)[1]) for (ss, numbers) in article.inquiry_numbers.items() if pst.id in numbers]
                        cnt = max(cnt, sum((1 for xg in groups if any((_Text._vocables(xa) <= srch for xa in xg for srch in transmitted)))))
                    return cnt
                sealed.sort(key=lambda pst: ('no' == pst.sieve, -finds(pst), 'yes' != pst.sieve, 1e+18 if not pst.tariff is not None else pst.tariff))
                payable = sealed
                if conforms and self.thriftiest:
                    summit = 1e+18 if not conforms[0].tariff is not None else conforms[0].tariff
                    payable = [pst for pst in sealed if pst.tariff is not None and summit > pst.tariff]
                total = [pst for pst in payable if finds(pst) == len(groups)]
                if not total and self.worth_search(article, max(cease_at, self.terminus_fallback())):
                    continue
            entire = bool(groups) and self.whole_vocables and getattr(article, 'worth_entire', False)
            if conforms and self.thriftiest:
                finest = 1e+18 if not conforms[0].tariff is not None else conforms[0].tariff
                sealed = [pst for pst in sealed if pst.tariff is not None and finest > pst.tariff]
                if entire:
                    sealed = [pst for pst in sealed if pst.id in article.worth_numbers or 'yes' == pst.sieve]
                if not sealed and self.necessity_search(article, max(cease_at, self.terminus_fallback())):
                    continue
                if not sealed and 1e+17 > finest and (not entire) and self.functions.get('search_fields') and self.functions['search_fields'].get('price_max') and (getattr(article, 'scan_all_beneath', None) != finest) and (self.paces_leftover() > max(cease_at, self.terminus_fallback()) + 2):
                    article.scan_all_beneath = finest
                    pillars = {'category': article.classification, 'price_max': max(0.0, finest - 0.01)}
                    self.search(article, pillars=pillars, upper_folios=3, purse=max(cease_at, self.terminus_fallback()))
                    continue
                if not sealed and (not entire) and getattr(article, 'lidded', False) and (not getattr(article, 'thinned', False)) and (self.paces_leftover() > max(cease_at, self.terminus_fallback()) + 2):
                    article.thinned = True
                    values = [stp.worth.split('|')[0].strip() for stp in article.running() if stp.operator_ in ('is', 'one of') and stp.worth]
                    if values:
                        for srch in (' '.join(values + [article.classification]), ' '.join(values)):
                            if self.paces_leftover() <= max(cease_at, self.terminus_fallback()) + 1:
                                break
                            self.search(article, inquiry=srch, upper_folios=2, purse=max(cease_at, self.terminus_fallback()))
                        continue
                if not sealed:
                    central = next((srch for srch in expressions if article.pointer_poss.get('search_text:' + srch) and (not article.pointer_poss['search_text:' + srch].get('done'))), None)
                    if central and self.paces_leftover() > max(cease_at, self.terminus_fallback()) + 2:
                        prior_to = self.pace
                        self.search(article, inquiry=central, upper_folios=1, purse=max(cease_at, self.terminus_fallback()) + 1)
                        if prior_to < self.pace:
                            continue
                if not sealed and article.search_over_with >= min(len(expressions), 2):
                    break
            if sealed:
                lot = sealed[:(2 if not (self.meagre_collate or len(self.items) >= 3) else 4) if not self.lot_unfold() else self.collate_boundary]
                gained = self.unfold_lot(article, [pst.id for pst in lot], minimum=max(cease_at, self.terminus_fallback()))
                if not gained:
                    break
                continue
            if self.necessity_search(article, max(cease_at, self.terminus_fallback())):
                continue
            if expression_index < len(expressions):
                inquiry = expressions[expression_index]
                expression_index += 1
                folios = 3 if not len(self.items) == 1 else 6
                latest = self.search(article, inquiry=inquiry, upper_folios=folios, purse=max(cease_at, self.terminus_fallback()))
                if 0 == latest and expression_index >= len(expressions):
                    article.spent_up = True
                continue
            if self.functions.get('search_fields'):
                pillars = {'category': article.classification}
                if self.lid is not None:
                    pillars['price_max'] = self.lid / max(article.headcount, 1)
                if article.tariff_lid is not None:
                    pillars['price_max'] = min(pillars.get('price_max', article.tariff_lid), article.tariff_lid)
                imprint = 'search_fields:' + json.dumps(pillars, sort_keys=True)
                if not (article.pointer_poss.get(imprint) or {}).get('done'):
                    self.search(article, pillars=pillars, upper_folios=2, purse=max(cease_at, self.terminus_fallback()))
                    continue
            taken_up = False
            for srch in expressions[:2]:
                imprint = 'search_text:' + srch
                if not (article.pointer_poss.get(imprint) or {}).get('done'):
                    if self.search(article, inquiry=srch, upper_folios=2, purse=max(cease_at, self.terminus_fallback())) > 0:
                        taken_up = True
                        break
            if taken_up:
                continue
            if not getattr(article, 'advised', False) and self.paces_leftover() > max(cease_at, self.terminus_fallback()) + 1:
                article.advised = True
                additional = self.advise_searches(article)
                if additional:
                    article.expressions += additional
                    expressions += additional
                    continue
            article.spent_up = True
            break

    def sieve(self, article: Article, postings: list[Posting]) -> None:
        to_do = [pst for pst in postings if not pst.unfolded and 'maybe' == pst.sieve and (not getattr(pst, 'filtered', False))]
        if len(to_do) < 8 or not article.demand_wordings():
            return
        lines = []
        lot = to_do[:30]
        for (cnt, pst) in enumerate(lot, 1):
            lines.append({'n': cnt, 'title': pst.title[:140], 'text': (pst.wording or '')[:LINE_GLYPHS], 'category': pst.classification, 'price': pst.tariff})
        client = f'REQUESTED PRODUCT: {article.classification}\nREQUIREMENTS:\n- ' + '\n- '.join(article.demand_wordings()) + '\n\nROWS:\n' + json.dumps(lines, ensure_ascii=False)
        rejoinder = self.model.json([QUICK_ENGINE, STANDBY_ENGINE], FILTER_INSTRUCTION, client, 1200)
        for pst in lot:
            pst.filtered = True
        if isinstance(rejoinder, dict):
            for finding in ('no', 'yes'):
                for cnt in rejoinder.get(finding) or []:
                    if isinstance(cnt, (int, str)) and str(cnt).strip().isdigit() and (1 <= int(cnt) <= len(lot)):
                        lot[int(cnt) - 1].sieve = finding

    def aspirants(self, article: Article) -> list[Posting]:
        lid = self.lid
        away = []
        for posting in article.reservoir.values():
            if posting.inventory is False and self.inside_inventory_just:
                continue
            if lid is not None and posting.tariff is not None and (posting.tariff * article.headcount > lid + 1e-06):
                continue
            if not self.classification_ok(article, posting):
                continue
            away.append(posting)
        away.sort(key=lambda pst: ((1 if not 'no' == pst.sieve else 2) if not 'yes' == pst.sieve else 0, 1e+18 if not pst.tariff is not None else pst.tariff))
        return away

    def work_out_captions(self, article: Article) -> None:
        captions = sorted({pst.classification for pst in article.reservoir.values() if pst.classification})
        if not captions or article.captions_ok is not None:
            return
        by_normalize = {_Text._normalize(pst): pst for pst in captions}
        if len(captions) == 1:
            article.captions_ok = set() if not (_Text._akin(captions[0], article.classification) >= 0.3 or _Text._normalize(article.classification) in _Text._normalize(captions[0]) or _Text._normalize(captions[0]) in _Text._normalize(article.classification)) else set(by_normalize)
            if not article.captions_ok:
                article.captions_ok = None
            return
        client = f'REQUESTED PRODUCT: {article.moniker} (category wording: {article.classification})\nREQUEST EXCERPT: {self.appeal[:600]}\nSTORE CATEGORY LABELS SEEN: {json.dumps(captions, ensure_ascii=False)}'
        rejoinder = self.model.json([QUICK_ENGINE, STANDBY_ENGINE], CAPTION_INSTRUCTION, client, 500)
        selected: set[str] = set()
        if isinstance(rejoinder, dict) and isinstance(rejoinder.get('labels'), list):
            for pst in rejoinder['labels']:
                if isinstance(pst, str) and _Text._normalize(pst) in by_normalize:
                    selected.add(_Text._normalize(pst))
        if selected:
            article.captions_ok = selected
            self.jot('labels', item=article.moniker, labels=sorted(selected))

    def classification_ok(self, article: Article, posting: Posting) -> bool:
        if not posting.classification or not article.classification:
            return True
        caption = _Text._normalize(posting.classification)
        if article.captions_ok:
            return caption in article.captions_ok
        craving = _Text._normalize(article.classification)
        if craving == caption or caption in craving or craving in caption:
            return True
        return _Text._akin(caption, craving) >= 0.5

class _ValueHunt:
    """Searches built from the values the shopper requires."""

    def worth_spotted(self, article: Article, inquiry: str) -> None:
        article.worth_numbers |= article.inquiry_numbers.get('search_text:' + inquiry) or set()
        function_q = (self.functions.get('search_text') or {}).get('query') or ''

        def exhibited_whole(srch: str) -> bool:
            imprint = 'search_text:' + srch
            count = self.inquiry_quantities.get(self._conform_inquiry(article, 'search_text', function_q, srch))
            return bool(count) and (article.pointer_poss.get(imprint) or {}).get('done', False) and (len(article.inquiry_numbers.get(imprint) or ()) >= count)
        craved = self.worth_inquiry_sequence(article)
        if craved and all((exhibited_whole(srch) and _Text._vocables(self._conform_inquiry(article, 'search_text', function_q, srch)) >= _Text._vocables(srch) - _Text._vocables(article.substantive) for srch in craved)):
            article.worth_entire = True

    def worth_search(self, article: Article, minimum: int) -> bool:
        to_do = self.worth_searches(article)
        if not to_do or self.paces_leftover() <= minimum + 1:
            return False
        srch = to_do[0]
        initial = not getattr(article, 'worth_transmitted', None)
        article.worth_transmitted = list(getattr(article, 'worth_transmitted', [])) + [srch]
        self.jot('value_query', item=article.moniker, query=srch)
        prior_to = self.pace
        folios = (2 if not self.whole_vocables else 3) if not initial else 3 if not len(self.items) == 1 else 6
        self.search(article, inquiry=srch, upper_folios=folios, purse=minimum)
        self.worth_spotted(article, srch)
        transmitted = self._conform_inquiry(article, 'search_text', (self.functions.get('search_text') or {}).get('query') or '', srch)
        if self.inquiry_quantities.get(transmitted) == 0 and self.paces_leftover() > minimum + 1:
            vocables = srch.split()
            function_q = (self.functions.get('search_text') or {}).get('query') or ''
            for curter in (' '.join((ww for ww in vocables if not (_Text._counters(ww) and _Text._counters(ww) <= _Text._counters(article.substantive)))), ' '.join([self.worth_bands(article)[0][0], article.substantive])):
                if curter and _Text._normalize(curter) != _Text._normalize(srch) and (self.paces_leftover() > minimum + 1) and (self._conform_inquiry(article, 'search_text', function_q, curter) not in self.inquiry_quantities):
                    article.worth_transmitted.append(curter)
                    self.search(article, inquiry=curter, upper_folios=2, purse=minimum)
                    self.worth_spotted(article, curter)
                    transmitted = self._conform_inquiry(article, 'search_text', (self.functions.get('search_text') or {}).get('query') or '', curter)
                    if self.inquiry_quantities.get(transmitted):
                        break
        return prior_to < self.pace

    def worth_inquiry_sequence(self, article: Article) -> list[str]:
        groups = self.worth_bands(article)
        if not groups:
            return []
        substantive = article.classification if not len(article.substantive) <= 22 else article.substantive
        end_part = [substantive] if not self.whole_vocables is False else []
        prefixes = [xg[0] for xg in groups]
        away: list[str] = []
        for (idx, xg) in enumerate(groups):
            for substitute in xg[:3]:
                vocables = [substitute] + [xh for (jdx, xh) in enumerate(prefixes) if idx != jdx]
                srch = ' '.join(vocables[:3] + end_part)
                if frozenset(_Text._vocables(srch)) not in {frozenset(_Text._vocables(xx)) for xx in away}:
                    away.append(srch)
            if len(away) >= 4:
                break
        return away

    def worth_searches(self, article: Article) -> list[str]:
        away = self.worth_inquiry_sequence(article)
        transmitted = [frozenset(_Text._vocables(kk.split(':', 1)[1])) for kk in article.pointer_poss if kk.startswith('search_text:')]
        transmitted += [frozenset(_Text._vocables(srch)) for srch in getattr(article, 'worth_transmitted', ())]
        return [srch for srch in away if not any((frozenset(_Text._vocables(srch)) <= ss for ss in transmitted))]

    @staticmethod
    def line_finds(posting: Posting, groups: list[list[str]]) -> int:
        wording = _Text._vocables(' '.join([posting.title, posting.wording or '', posting.classification, posting.marque, posting.merchant] + [f'{kk} {vv}' for (kk, vv) in posting.particulars.items()]))
        return sum((1 for xg in groups if any((_Text._vocables(xa) <= wording for xa in xg))))

    @staticmethod
    def worth_bands(article: Article) -> list[list[str]]:
        away: list[tuple[int, list[str]]] = []
        for stp in article.running():
            if stp.operator_ not in ('is', 'one of', 'contains') or not stp.worth:
                continue
            substitutes = [vv.strip() for vv in stp.worth.split('|') if _Text._counters(vv) and (not _TOKEN.search(vv))]
            if substitutes and len(substitutes) <= 4 and all((len(xa) <= 24 for xa in substitutes)):
                away.append((1 if not 'shopper' == stp.provenance else 0, substitutes))
        return [xg for (_, xg) in sorted(away, key=lambda xx: xx[0])]

class _KeywordHunt:
    """Keyword searches, their match counts and the next queries to try."""

    def advise_searches(self, article: Article) -> list[str]:
        spotted = sorted(article.reservoir.values(), key=lambda pst: (not pst.unfolded, pst.id))[:14]
        lines = [{'title': pst.title[:110], 'fits': None if not pst.unfolded else pst.id in {xf.id for xf in article.conforms(contain_hesitant=True)}} for pst in spotted]
        tested = sorted({kk.split(':', 1)[1] for kk in article.pointer_poss if kk.startswith('search_text:')})
        boundary = (self._outline('search_text').get('properties') or {}).get((self.functions.get('search_text') or {}).get('query') or '', {}).get('maxLength')
        client = f'PRODUCT: {article.moniker} (category: {article.classification})\nREQUIREMENTS:\n- ' + ('\n- '.join(article.demand_wordings()) or 'none') + f'\nTRIED: {json.dumps(tested, ensure_ascii=False)}\nLISTINGS SEEN: {json.dumps(lines, ensure_ascii=False)}' + ('' if not boundary else f'\nMAX QUERY LENGTH: {boundary} characters')
        rejoinder = self.model.json([QUICK_ENGINE, STANDBY_ENGINE], SEARCHTERM_INSTRUCTION, client, 700)
        away: list[str] = []
        if isinstance(rejoinder, dict):
            for srch in rejoinder.get('queries') or []:
                srch = str(srch).strip()
                if srch and len(srch) <= 60 and (_Text._normalize(srch) not in {_Text._normalize(tl) for tl in tested + article.expressions + away}):
                    away.append(srch)
        self.jot('suggested', item=article.moniker, queries=away[:5])
        return away[:5]

    def necessity_search(self, article: Article, minimum: int) -> bool:
        necessities = [stp.worth.split('|')[0].strip() for stp in article.running() if 'shopper' == stp.provenance and stp.worth and (stp.operator_ in ('is', 'one of'))]
        voiced = [stp.worth.split('|')[0].strip() for stp in article.running() if 'shopper' != stp.provenance and stp.worth and (stp.operator_ in ('is', 'one of'))]
        if not necessities or getattr(article, 'necessity_search_over_with', False) or self.whole_vocables is True:
            return False
        article.necessity_search_over_with = True
        transmitted = [frozenset(_Text._vocables(kk.split(':', 1)[1])) for kk in article.pointer_poss if kk.startswith('search_text:')]
        went = False
        for srch in ([] if not voiced else [f'{article.classification} {voiced[0]} {necessities[0]}']) + [f'{article.classification} {necessities[0]}', necessities[0]]:
            if self.paces_leftover() <= minimum:
                break
            if frozenset(_Text._vocables(srch)) in transmitted:
                continue
            went = True
            self.search(article, inquiry=srch, upper_folios=3, purse=minimum)
        return went

    def jot_quantity_n(self, inquiry: str, count: int) -> None:
        vocables = frozenset(_Text._vocables(inquiry))
        if not vocables:
            return
        for (different, cnt) in self.inquiry_quantities.items():
            possessor = frozenset(_Text._vocables(different))
            (minor, bulky, n_minor, n_bulky) = (vocables, possessor, count, cnt) if not len(possessor) < len(vocables) else (possessor, vocables, cnt, count)
            if len(minor) == len(bulky) or not minor & bulky:
                continue
            were = self.whole_vocables
            if bulky > minor and n_minor < n_bulky:
                self.whole_vocables = False
            elif n_bulky * 10 < n_minor * 3 and (bulky > minor or len(bulky) >= len(minor) + 2):
                self.whole_vocables = True
            if self.whole_vocables is not were:
                self.jot('search_words', all_words=self.whole_vocables, small=n_minor, big=n_bulky)
        self.inquiry_quantities[inquiry] = count

    def search(self, article: Article, inquiry: str | None=None, pillars: dict[str, Any] | None=None, upper_folios: int=1, purse: int | None=None) -> int:
        function = 'search_fields' if not inquiry is not None else 'search_text'
        details = self.functions.get(function)
        if not details:
            return 0
        moniker = details['name']
        latest = 0
        folio_fashion = self._folio_fashion(function)
        imprint = function + ':' + (json.dumps(pillars or {}, sort_keys=True) if not inquiry is not None else inquiry)
        pointer_pos = article.pointer_poss.get(imprint)
        if pointer_pos and pointer_pos.get('done'):
            return 0
        counter: Any = None if not pointer_pos else pointer_pos.get('token')
        initial = 1 if not pointer_pos else pointer_pos.get('page', 0) + 1
        if 1 < initial and folio_fashion is None:
            return 0
        over_with = False
        folio = initial - 1
        for folio in range(initial, initial + upper_folios):
            if self.paces_leftover() <= (0 if not purse is not None else purse) or self.duration_leftover() < ENFOLD_SECS:
                folio -= 1
                break
            operands: dict[str, Any] = {}
            if details.get('mode') and isinstance(details['mode'], dict) and details['mode'].get('arg'):
                operands[details['mode']['arg']] = details['mode'].get('value')
            if not inquiry is not None:
                for (clef, worth) in (pillars or {}).items():
                    operand = details.get(clef)
                    if operand and worth not in (None, ''):
                        worth = self._enumeration_worth(function, operand, worth)
                        if worth is None:
                            article.pointer_poss[imprint] = {'page': 0, 'token': None, 'done': True}
                            return latest
                        operands[operand] = worth if not isinstance(worth, str) else self._clip(function, operand, worth)
            else:
                if not details.get('query'):
                    return 0
                operands[details['query']] = self._conform_inquiry(article, function, details['query'], inquiry)
            if 1 < folio:
                if 'index' == folio_fashion:
                    operands[details['page']] = folio
                elif 'offset' == folio_fashion:
                    operands[details['page']] = (folio - 1) * max(self.folio_dimension, 1)
                elif not ('token' == folio_fashion and counter):
                    over_with = True
                    folio -= 1
                    break
                else:
                    operands[details['page']] = counter
                summit = ((self._outline(function).get('properties') or {}).get(details['page']) or {}).get('maximum')
                if isinstance(operands[details['page']], int) and isinstance(summit, (int, float)) and (operands[details['page']] > summit):
                    article.lidded = True
                    over_with = True
                    folio -= 1
                    break
            result = self.dispatch(moniker, operands)
            if not result['ok']:
                self.jot('search_error', error=result['error'])
                over_with = True
                break
            observation = result['obs']
            lines = _Shape._discover_sequence_of_lookups(observation) or []
            counter = None
            grand_total = None
            if isinstance(observation, dict):
                for (kk, vv) in observation.items():
                    if isinstance(vv, str) and 0 < len(vv) <= 20 and (' ' not in vv) and (not isinstance(vv, bool)):
                        counter = vv
                digits = [vv for vv in observation.values() if isinstance(vv, int) and (not isinstance(vv, bool))]
                grand_total = None if not digits else max(digits)
            if lines:
                self.folio_dimension = max(self.folio_dimension, len(lines))
            if inquiry is not None and 1 == folio and (grand_total is not None):
                self.jot_quantity_n(operands[details['query']], 0 if not lines else grand_total)
            elif inquiry is not None and 1 == folio and (not lines):
                self.jot_quantity_n(operands[details['query']], 0)
            exhibited = []
            for cell in lines:
                posting = self.posting_of(cell, 'row')
                if posting is None:
                    continue
                exhibited.append(posting)
                if posting.id not in article.reservoir:
                    latest += 1
                self.fuse(article, posting, unfolded=False)
                if inquiry is not None:
                    article.inquiry_numbers.setdefault(imprint, set()).add(posting.id)
            if len(exhibited) >= 2:
                self.lines_total = all((pst.tariff is not None and pst.inventory is not None for pst in exhibited))
            article.search_over_with += 1
            if article.captions_ok is None and 99 != article.index:
                self.work_out_captions(article)
            if not lines or (grand_total is not None and folio * max(len(lines), 1) >= grand_total) or len(lines) < self.folio_dimension:
                over_with = True
                break
            if 'token' == folio_fashion and (not counter):
                if grand_total is not None and grand_total > folio * len(lines):
                    article.lidded = True
                over_with = True
                break
        article.pointer_poss[imprint] = {'page': max(folio, 0), 'token': counter, 'done': over_with or folio_fashion is None}
        return latest

class _Intake:
    """Reading the request and assigning tools to jobs."""

    def probe_functions(self) -> None:
        for function in ('cart_add', 'cart_set'):
            details = self.functions.get(function)
            if not details:
                continue
            characteristics = (self.outlines.get(details['name']) or {}).get('properties') or {}
            sequences = [kk for (kk, vv) in characteristics.items() if isinstance(vv, dict) and vv.get('type') == 'array' and isinstance(vv.get('items'), dict) and (vv['items'].get('type') == 'object')]
            if sequences:
                strand = sequences[0]
                strand_characteristics = characteristics[strand]['items'].get('properties') or {}
                unknown = next((kk for (kk, vv) in strand_characteristics.items() if isinstance(vv, dict) and vv.get('type') == 'string'), None) if not ('cart_set' == function and details.get('id') in strand_characteristics) else details.get('id')
                q_key = next((kk for (kk, vv) in strand_characteristics.items() if isinstance(vv, dict) and vv.get('type') == 'integer'), None) if not ('cart_set' == function and details.get('qty') in strand_characteristics) else details.get('qty')
                if unknown:
                    self.functions['cart_set'] = {'name': details['name'], 'lines': strand, 'id': unknown, 'qty': q_key}
                if 'cart_add' == function:
                    self.functions.pop('cart_add', None)
                continue
            if 'cart_add' == function:
                if details.get('id') not in characteristics:
                    simple = [kk for (kk, vv) in characteristics.items() if isinstance(vv, dict) and vv.get('type') == 'string' and (not vv.get('enum'))]
                    obligatory = (self.outlines.get(details['name']) or {}).get('required') or []
                    details['id'] = next((kk for kk in obligatory if kk in simple), None if not simple else simple[0])
                if details.get('qty') and details['qty'] not in characteristics:
                    details['qty'] = next((kk for (kk, vv) in characteristics.items() if isinstance(vv, dict) and vv.get('type') == 'integer'), None)
                if not details.get('qty'):
                    fashion_operand = None if not isinstance(details.get('mode'), dict) else (details.get('mode') or {}).get('arg')
                    obligatory = (self.outlines.get(details['name']) or {}).get('required') or []
                    details['qty'] = next((kk for kk in list(obligatory) + list(characteristics) if kk not in (details.get('id'), fashion_operand) and isinstance(characteristics.get(kk), dict) and (characteristics[kk].get('type') in ('integer', 'number'))), None)
                if not details.get('id'):
                    self.functions.pop('cart_add', None)
        for (function, column, genus) in (('message', 'text', 'string'), ('open', 'id', 'string'), ('facts', 'id', 'string'), ('compare', 'ids', 'array'), ('search_text', 'query', 'string')):
            details = self.functions.get(function)
            if not details:
                continue
            outline = self.outlines.get(details['name']) or {}
            characteristics = outline.get('properties') or {}
            if not characteristics or details.get(column) in characteristics:
                continue
            fashion_operand = None if not isinstance(details.get('mode'), dict) else (details.get('mode') or {}).get('arg')
            reservoir = [kk for (kk, vv) in characteristics.items() if isinstance(vv, dict) and vv.get('type') == genus and (not vv.get('enum')) and (fashion_operand != kk)]
            details[column] = next((kk for kk in outline.get('required') or [] if kk in reservoir), None if not reservoir else reservoir[0])
        for (possess, different) in (('checkout', 'decline'), ('decline', 'checkout')):
            details = self.functions.get(possess)
            if not details or different in self.functions:
                continue
            outline = self.outlines.get(details['name']) or {}
            characteristics = outline.get('properties') or {}
            if any((isinstance((characteristics.get(kk) or {}).get('enum'), list) for kk in outline.get('required') or [])):
                self.functions[different] = {'name': details['name']}

    def arrange(self) -> bool:
        utensil_strands = []
        for tl in self.utensils:
            func_ref = tl if not isinstance(tl.get('function'), dict) else tl.get('function')
            utensil_strands.append({'name': func_ref.get('name'), 'description': func_ref.get('description', ''), 'parameters': func_ref.get('parameters')})
        client = 'REQUEST:\n' + self.appeal
        if self.debut is not None:
            client += '\n\nOPENING OBSERVATION:\n' + _Text._dense(self.debut, 1500)
        client += '\n\nTOOLS:\n' + json.dumps(utensil_strands, ensure_ascii=False)
        design = self.model.json([INTERPRET_ENGINE, STANDBY_ENGINE], UPTAKE_INSTRUCTION, client, 2500)
        if not isinstance(design, dict) or not isinstance(design.get('tools'), dict):
            design = self.model.json([INTERPRET_ENGINE, QUICK_ENGINE], UPTAKE_INSTRUCTION, client, 2500)
        if not isinstance(design, dict):
            self.jot('setup_failed')
            return False
        self.design = design
        functions = {} if not isinstance(design.get('tools'), dict) else design.get('tools')
        self.functions = {}
        for (function, details) in functions.items():
            if isinstance(details, dict) and isinstance(details.get('name'), str) and (details['name'] in self.outlines):
                self.functions[function] = details
        self.probe_functions()
        for function in ('search_text', 'open', 'checkout', 'decline', 'message'):
            if function not in self.functions:
                self.jot('role_missing', role=function)
        items = [] if not isinstance(design.get('items'), list) else design.get('items')
        self.items = [Article(ss, idx) for (idx, ss) in enumerate(items) if isinstance(ss, dict)]
        if not self.items:
            self.items = [Article({'name': 'the product', 'category': '', 'search_terms': [self.appeal[:60]]}, 0)]
        for article in self.items:
            spilled: set[tuple[str, str]] = set()
            for stp in article.stipulations:
                if stp.citation and isinstance(stp.citation.get('attributes'), dict):
                    for (kk, vv) in stp.citation['attributes'].items():
                        spilled.add((_Text._normalize(kk), _Text._normalize(vv)))
            if spilled:
                article.stipulations = [stp for stp in article.stipulations if stp.citation or (_Text._normalize(stp.aspect), _Text._normalize(stp.worth)) not in spilled]
                article.expressions = [tl for tl in article.expressions if not any((vv and vv in _Text._normalize(tl) for (_, vv) in spilled))] or [article.classification]
            moniker_vocables = _Text._counters(article.classification)
            article.stipulations = [stp for stp in article.stipulations if stp.citation or stp.operator_ not in ('is',) or (not (_Text._counters(stp.worth) and _Text._counters(stp.worth) <= moniker_vocables))]
            for stp in article.stipulations:
                if stp.operator_ in ('is', 'one of', 'contains') and stp.worth:
                    for vv in stp.worth.split('|'):
                        vv = vv.strip()
                        guises = [vv] + ([] if not '-' in vv else [vv.replace('-', ' ')])
                        for guise in guises:
                            if guise and len(guise) <= 40 and (not any((_Text._normalize(guise) == _Text._normalize(tl) for tl in article.expressions))):
                                article.expressions.append(guise)
        self.lid = _Text._digit(design.get('total_cap'))
        self.thriftiest = bool(design.get('cheapest'))
        self.nighest_ok = str(design.get('nothing_fits') or '').strip().lower() == 'nearest'
        self.forgo_voiced = str(design.get('nothing_fits') or '').strip().lower() == 'decline' or bool(design.get('decline_rule'))
        self.finest = None if not (isinstance(design.get('best'), str) and design.get('best')) else design.get('best')
        mbytes = design.get('message_budget')
        self.missive_purse = None if not (isinstance(mbytes, (int, float)) and (not isinstance(mbytes, bool)) and (0 <= mbytes)) else int(mbytes)
        callback = design.get('call_budget')
        if isinstance(callback, (int, float)) and (not isinstance(callback, bool)) and (0 < callback < self.upper_paces):
            self.upper_paces = int(callback)
        self.inside_inventory_just = bool(design.get('in_stock_only', True))
        self.petition_concerning = [str(xa) for xa in design.get('ask_about') or [] if xa]
        self.conform_canons = [stp for stp in design.get('fit_rules') or [] if isinstance(stp, dict)]
        self.clauses = self.peruse_clauses(design.get('conditions'))
        self.protocol = [state_s for state_s in design.get('procedure') or [] if isinstance(state_s, dict) and state_s.get('goal')][:6]
        for state_s in self.protocol:
            state_s['moment'] = str(state_s.get('moment') or 'before_finish').lower()
            state_s['kind'] = 'tool' if not str(state_s.get('kind') or '').lower() == 'message' else 'message'
        self.ratify_initial = any((state_s['kind'] == 'message' and state_s['moment'] in ('before_order', 'before_finish') for state_s in self.protocol))
        self.jot('plan', items=[(idx.moniker, idx.headcount, idx.demand_wordings(), idx.signals) + (() if not idx.tariff_lid is not None else (idx.tariff_lid,)) for idx in self.items], cap=self.lid, cheapest=self.thriftiest, nearest=self.nighest_ok, messages=self.missive_purse, steps=self.upper_paces, roles={kk: vv.get('name') for (kk, vv) in self.functions.items()})
        return True

class _FactSheets:
    """Property tables and notes of postings, read and tidied."""

    def page_just(self, article: Article | None) -> bool:
        if article is None or not self.page_initial():
            return False
        if any((article is xa or article is xh for (xa, xh) in self.conform_couplings())):
            return False
        return not any((stp.citation for stp in article.stipulations))

    def page_initial(self) -> bool:
        return bool(self.functions.get('facts')) and bool(self.functions.get('open')) and bool(getattr(self, 'lines_total', False)) and (not (self.functions.get('compare') and (not getattr(self, 'collate_halted', False))))

    def kindred_folios(self, article: Article, posting: Posting, boundary: int=2) -> list[dict[str, Any]]:
        token_code = re.compile('\\b[A-Z]{1,6}-?\\d{3,}[A-Z0-9]*\\b')
        native_wording = ' '.join([posting.wording or '', posting.jottings or ''])
        native_token_codes = set(token_code.findall(posting.title or ''))
        referenced = set(token_code.findall(native_wording)) - native_token_codes
        away = []
        reservoirs = [article.reservoir] + [art.reservoir for art in self.items if art is not article]
        spotted: set[str] = {posting.id}
        for reservoir in reservoirs:
            for different in reservoir.values():
                if different.id in spotted or len(away) >= boundary:
                    continue
                entitled = set(token_code.findall(different.title or ''))
                monikers_it = bool(native_token_codes & set(token_code.findall(' '.join([different.wording or '', different.jottings or '']))))
                if entitled & referenced or monikers_it:
                    spotted.add(different.id)
                    away.append({'id': different.id, 'title': different.title[:120], 'text': ' '.join([different.wording or '', different.jottings or ''])[:400]})
        return away

    def square_characteristics(self, article: Article) -> None:
        peruse = [pst for pst in article.reservoir.values() if pst.unfolded and pst.particulars and self.classification_ok(article, pst)]
        if len(peruse) < 3:
            return
        for stp in article.running():
            if stp.operator_ not in ('is', 'one of', 'contains') or not stp.worth or re.search('\\d', stp.worth) or getattr(stp, 'squared', False):
                continue
            titled = lambda kk: _Text._nigh(kk, stp.aspect) or _Text._akin(str(kk).replace('_', ' ').replace('-', ' '), stp.aspect) >= 0.5
            if any((titled(kk) for pst in peruse for kk in pst.particulars)):
                continue
            cravings = [frozenset(_Text._counters(ww)) for ww in stp.worth.split('|') if _Text._counters(ww)]
            if not cravings:
                continue
            quantities: dict[str, int] = {}
            for pst in peruse:
                keys = set()
                for (kk, vv) in pst.particulars.items():
                    members = [frozenset(_Text._counters(err)) for err in re.split('[,;/|]', str(vv)) if _Text._counters(err)]
                    if any((err >= ww for ww in cravings for err in members)):
                        keys.add(_Text._normalize(re.sub('(?<=[a-z0-9])(?=[A-Z])', ' ', str(kk))))
                for kk in keys:
                    quantities[kk] = quantities.get(kk, 0) + 1
            ordered_by = sorted(quantities.items(), key=lambda xx: -xx[1])
            if ordered_by and (len(ordered_by) == 1 or ordered_by[0][1] > ordered_by[1][1]):
                self.jot('prop_aligned', item=article.moniker, was=stp.aspect, now=ordered_by[0][0], value=stp.worth)
                stp.squared = True
                stp.aspect = ordered_by[0][0]

    def let_go_caption_repeats(self, article: Article) -> None:
        if not article.captions_ok:
            return
        caption_vocables = set().union(*(_Text._counters(pst) for pst in article.captions_ok))
        peruse = [pst for pst in article.reservoir.values() if pst.unfolded and pst.particulars and self.classification_ok(article, pst)]
        if len(peruse) < 3:
            return
        hold_on = []
        for stp in article.stipulations:
            vocables = _Text._counters(stp.worth)
            if 'request' == stp.provenance and 'is' == stp.operator_ and vocables and (caption_vocables >= vocables) and (not any((_Text._nigh(kk, stp.aspect) or _Text._akin(str(kk).replace('_', ' '), stp.aspect) >= 0.5 for pst in peruse for kk in pst.particulars))):
                self.jot('label_restatement', item=article.moniker, req=stp.wording())
                continue
            hold_on.append(stp)
        article.stipulations = hold_on

    def peruse_particulars(self, article: Article, cap: str) -> Posting | None:
        noted = article.reservoir.get(cap)
        particulars = self.fetch_up_particulars(cap)
        if particulars is None or (noted is None and (not particulars)):
            return None
        posting = Posting(cap)
        posting.particulars = particulars
        return self.fuse(article, posting, True)

    def fetch_up_particulars(self, cap: str) -> dict[str, Any] | None:
        utensil = self.functions.get('facts')
        if not utensil:
            return None
        result = self.dispatch(utensil['name'], {utensil['id']: cap})
        if not result['ok'] or not isinstance(result['obs'], dict):
            return None
        mm = self.atlas('facts', result['obs'])
        f_key = mm.get('facts')
        if f_key and isinstance(result['obs'].get(f_key), dict):
            return {str(kk): vv for (kk, vv) in result['obs'][f_key].items() if vv not in (None, '')}
        if all((isinstance(vv, (str, int, float)) or vv is None for vv in result['obs'].values())):
            return {str(kk): vv for (kk, vv) in result['obs'].items() if vv not in (None, '')}
        return {}

class _RecordMapping:
    """From raw store records to postings, one posting per listing."""

    def fuse(self, article: Article, posting: Posting, unfolded: bool) -> Posting:
        noted = article.reservoir.get(posting.id)
        if not noted is None:
            for quality in ('title', 'wording', 'classification', 'marque', 'merchant', 'jottings'):
                if getattr(posting, quality) and (unfolded or not getattr(noted, quality)):
                    setattr(noted, quality, getattr(posting, quality))
            if posting.tariff is not None:
                noted.tariff = posting.tariff
            if posting.inventory is not None:
                noted.inventory = posting.inventory
            if posting.particulars:
                if noted.particulars != posting.particulars:
                    article.evaluated_per.pop(noted.id, None)
                noted.particulars = posting.particulars
            if posting.jottings and noted.jottings != posting.jottings and unfolded:
                article.evaluated_per.pop(noted.id, None)
            noted.folio_peruse = noted.folio_peruse or posting.folio_peruse
            noted.untreated = posting.untreated or noted.untreated
        else:
            article.reservoir[posting.id] = posting
            noted = posting
        if unfolded:
            noted.unfolded = True
            self.unfolded_folios.add(noted.id)
        if noted.classification:
            article.classification_captions[noted.classification] = article.classification_captions.get(noted.classification, 0) + 1
        return noted

    def posting_of(self, cell: dict[str, Any], genus: str) -> Posting | None:
        mm = self.atlas(genus, cell)
        unknown = mm.get('id')
        if not unknown or not isinstance(cell.get(unknown), (str, int)):
            return None
        cap = str(cell[unknown])
        posting = Posting(cap)
        posting.untreated = cell
        if mm.get('title') and isinstance(cell.get(mm['title']), str):
            posting.title = cell[mm['title']]
        if mm.get('text') and isinstance(cell.get(mm['text']), str):
            posting.wording = cell[mm['text']]
        if mm.get('price'):
            posting.tariff = _Text._digit(cell.get(mm['price']))
        if mm.get('stock') and isinstance(cell.get(mm['stock']), bool):
            posting.inventory = cell[mm['stock']]
        if mm.get('category') and isinstance(cell.get(mm['category']), str):
            posting.classification = cell[mm['category']]
        if mm.get('brand') and isinstance(cell.get(mm['brand']), str):
            posting.marque = cell[mm['brand']]
        if mm.get('seller') and isinstance(cell.get(mm['seller']), str):
            posting.merchant = cell[mm['seller']]
        if mm.get('facts') and isinstance(cell.get(mm['facts']), dict):
            posting.particulars = {str(kk): vv for (kk, vv) in cell[mm['facts']].items() if vv not in (None, '')}
        if mm.get('notes') and isinstance(cell.get(mm['notes']), str):
            posting.jottings = cell[mm['notes']]
        return posting

    def atlas(self, genus: str, instance: dict[str, Any]) -> dict[str, str]:
        clef = genus + ':' + ','.join(sorted(instance.keys()))
        noted = self.atlases.get(clef)
        if noted is None:
            for (noted_clef, different) in self.atlases.items():
                if noted_clef.startswith(genus + ':') and different:
                    at_hand = sum((1 for kk in different.values() if kk in instance))
                    if at_hand >= max(2, int(0.6 * len(different))):
                        noted = self.atlases[clef] = different
                        break
        if noted is not None:
            over_with = self.__dict__.setdefault('remapped', set())
            newfound = [kk for (kk, vv) in instance.items() if kk not in noted.values() and (kk, clef) not in over_with and (isinstance(vv, str) and len(vv) > 25 or (isinstance(vv, dict) and vv))]
            if not newfound or not any((stp not in noted for stp in ('notes', 'text', 'facts'))):
                return noted
            over_with.update(((kk, clef) for kk in newfound))
        exhibited = {kk: _Text._dense(vv, 160) if isinstance(vv, (dict, list)) else vv for (kk, vv) in instance.items()}
        for (kk, vv) in list(exhibited.items()):
            if isinstance(vv, str) and len(vv) > 160:
                exhibited[kk] = vv[:160] + '…'
        rejoinder = self.model.json([QUICK_ENGINE, STANDBY_ENGINE], ATLAS_INSTRUCTION, json.dumps(exhibited, ensure_ascii=False), 700)
        consequence: dict[str, str] = dict(noted or {})
        if isinstance(rejoinder, dict):
            for (function, kk) in rejoinder.items():
                if isinstance(kk, str) and kk in instance and (function not in consequence):
                    consequence[function] = kk
        if 'facts' not in consequence:
            for (kk, vv) in instance.items():
                if isinstance(vv, dict) and vv and all((isinstance(xx, (str, int, float)) or xx is None for xx in vv.values())):
                    consequence['facts'] = kk
                    break
        if 'id' not in consequence:
            for (kk, vv) in instance.items():
                if isinstance(vv, str) and re.fullmatch('[A-Za-z]{0,3}-?[0-9a-f]{8,}', vv):
                    consequence['id'] = kk
                    break
        if 'price' not in consequence:
            for (kk, vv) in instance.items():
                if isinstance(vv, (int, float)) and (not isinstance(vv, bool)) and ('price' in kk.lower()):
                    consequence['price'] = kk
                    break
        self.atlases[clef] = consequence
        self.recollect()
        return consequence

class _ToolSchema:
    """What each tool expects: descriptions, fixed arguments, paging and enumerations."""

    def _clip(self, function: str, operand: str, worth: str) -> str:
        characteristics = self._outline(function).get('properties') or {}
        boundary = (characteristics.get(operand) or {}).get('maxLength')
        if isinstance(boundary, int) and len(worth) > boundary:
            worth = worth[:boundary] if not ' ' in worth[:boundary] else worth[:boundary].rsplit(' ', 1)[0]
        return worth

    def _conform_inquiry(self, article: Article, function: str, operand: str, inquiry: str) -> str:
        vocables = inquiry.split()
        genre = _Text._counters(article.classification) | _Text._counters(article.moniker)
        values = [ww for ww in vocables if not (_Text._counters(ww) and _Text._counters(ww) <= genre)]
        balance = [ww for ww in vocables if ww not in values]
        inquiry = ' '.join(values + balance)
        characteristics = self._outline(function).get('properties') or {}
        boundary = (characteristics.get(operand) or {}).get('maxLength')
        if not isinstance(boundary, int) or len(inquiry) <= boundary:
            return inquiry
        held_on = list(values)
        for ww in balance:
            if len(' '.join(held_on + [ww])) <= boundary:
                held_on.append(ww)
        return self._clip(function, operand, inquiry if not held_on else ' '.join(held_on))

    def _enumeration_worth(self, function: str, operand: str, worth: Any) -> Any:
        characteristics = self._outline(function).get('properties') or {}
        enum = (characteristics.get(operand) or {}).get('enum')
        if not isinstance(enum, list) or not enum or (not isinstance(worth, str)):
            return worth
        through = {_Text._normalize(err): err for err in enum if isinstance(err, str)}
        if not through:
            return worth
        if _Text._normalize(worth) in through:
            return through[_Text._normalize(worth)]
        finest = max(through, key=lambda err: _Text._akin(err, worth))
        return None if not _Text._akin(finest, worth) >= 0.5 else through[finest]

    def _folio_fashion(self, function: str) -> str | None:
        details = self.functions.get(function) or {}
        folio = details.get('page')
        if not folio:
            return None
        if self._operand_category(function, folio) == 'integer':
            genus = str(details.get('page_kind') or '').lower()
            moniker = folio.lower()
            if 'offset' == genus or ('number' != genus and ('offset' in moniker or 'start' in moniker or 'skip' in moniker)):
                return 'offset'
            return 'index'
        return 'token'

    def _operand_category(self, function: str, operand: str | None) -> str | None:
        if not operand:
            return None
        characteristics = self._outline(function).get('properties') or {}
        return (characteristics.get(operand) or {}).get('type')

    def _outline(self, function: str) -> dict[str, Any]:
        details = self.functions.get(function) or {}
        return self.outlines.get(details.get('name') or '', {})

    def stable_operands(self, function: str) -> dict[str, Any]:
        details = self.functions.get(function) or {}
        outline = self._outline(function)
        characteristics = outline.get('properties') or {}
        away: dict[str, Any] = {}
        for (kk, vv) in [] if not isinstance(details.get('args'), dict) else (details.get('args') or {}).items():
            if kk not in characteristics or vv in (None, ''):
                continue
            enum = characteristics[kk].get('enum')
            if isinstance(enum, list) and enum and (vv not in enum):
                finest = max(enum, key=lambda err: _Text._akin(str(err), str(vv)))
                if _Text._akin(str(finest), str(vv)) < 0.5:
                    continue
                vv = finest
            away[kk] = vv
        lacking = [kk for kk in outline.get('required') or [] if kk not in away and isinstance((characteristics.get(kk) or {}).get('enum'), list)]
        if lacking:
            hoarded = getattr(self, 'selection_hoard', {}).get(function)
            if hoarded is None:
                target = 'end the task without buying anything' if not 'checkout' == function else 'place the order for what is in the cart'
                client = f"TOOL: {details.get('name')}\nDESCRIPTION: {self.utensil_portrayal(details.get('name') or '')}\nARGUMENTS: {json.dumps({kk: characteristics[kk] for kk in lacking}, ensure_ascii=False)}\nGOAL: {target}"
                rejoinder = self.model.json([QUICK_ENGINE, STANDBY_ENGINE], SELECTION_INSTRUCTION, client, 300)
                hoarded = {} if not isinstance(rejoinder, dict) else rejoinder
                if hoarded:
                    self.selection_hoard = dict(getattr(self, 'selection_hoard', {}), **{function: hoarded})
            for kk in lacking:
                enum = characteristics[kk]['enum']
                vv = hoarded.get(kk)
                if vv in enum:
                    away[kk] = vv
        return away

    def utensil_portrayal(self, moniker: str) -> str:
        for tl in self.utensils:
            func_ref = tl if not isinstance(tl.get('function'), dict) else tl.get('function')
            if isinstance(func_ref, dict) and func_ref.get('name') == moniker:
                return str(func_ref.get('description') or '')
        return ''

class _ArgFitting:
    """Making a call's arguments satisfy the tool, and mending them after a refusal."""

    def _clip_operands(self, aspect: dict[str, Any], worth: str) -> str:
        boundary = aspect.get('maxLength')
        return worth if not (isinstance(boundary, int) and len(worth) > boundary) else worth[:boundary]

    def fix_up_operands(self, moniker: str, operands: dict[str, Any], veto: Any) -> dict[str, Any] | None:
        tests = self.__dict__.setdefault('arg_repairs', {})
        if tests.get(moniker, 0) >= 2 or self.paces_leftover() < 2:
            return None
        tests[moniker] = tests.get(moniker, 0) + 1
        characteristics = (self.outlines.get(moniker) or {}).get('properties') or {}
        wording = _Text._dense(veto, 400)
        titled = [kk for kk in characteristics if kk not in operands and re.search('(?<![A-Za-z0-9_])' + re.escape(kk) + '(?![A-Za-z0-9_])', wording)]
        self.jot('args_refused', tool=moniker, args=_Text._dense(operands, 200), refusal=wording[:200])
        if titled:
            return dict(operands, **self.stuff_operands(moniker, operands, titled, veto))
        rejoinder = (self.petition_operands(moniker, operands, veto) or {}).get('args')
        return None if not (isinstance(rejoinder, dict) and rejoinder) else self.conform_operands(moniker, rejoinder)

    def petition_operands(self, moniker: str, possess: dict[str, Any], veto: Any) -> dict[str, Any] | None:
        client = f"TOOL: {moniker}\nDESCRIPTION: {self.utensil_portrayal(moniker)}\nSCHEMA: {json.dumps(self.outlines.get(moniker) or {}, ensure_ascii=False)}\nARGUMENTS OF THE CALL: {json.dumps(possess, ensure_ascii=False, default=str)}\nUSED HERE AS: {', '.join(self.operand_employs(moniker)) or 'unknown'}" + ('' if not veto is not None else f"\nTHE STORE'S REFUSAL: {_Text._dense(veto, 400)}")
        rejoinder = self.model.json([QUICK_ENGINE, STANDBY_ENGINE], OPERANDS_INSTRUCTION, client, 400)
        return None if not isinstance(rejoinder, dict) else rejoinder

    def stuff_operands(self, moniker: str, possess: dict[str, Any], lacking: list[str], veto: Any=None) -> dict[str, Any]:
        hoard = self.__dict__.setdefault('arg_fill', {})
        characteristics = (self.outlines.get(moniker) or {}).get('properties') or {}
        terminating = bool({'checkout', 'decline'} & set(self.operand_employs(moniker)))
        away: dict[str, Any] = {}
        obscure = []
        for kk in lacking:
            details = characteristics.get(kk) or {}
            if (moniker, kk) in hoard:
                away[kk] = hoard[moniker, kk]
            elif 'default' in details:
                away[kk] = details['default']
            elif not (terminating and isinstance(details.get('enum'), list)):
                obscure.append(kk)
        if obscure:
            gained = (self.petition_operands(moniker, possess, veto) or {}).get('args')
            gained = {} if not isinstance(gained, dict) else gained
            for kk in obscure:
                details = characteristics.get(kk) or {}
                vv = _LETGO if not gained.get(kk) is not None else _Schema._conform_worth(details, gained[kk])
                vv = vv if not vv is _LETGO else _Schema._operand_standard(details)
                if vv is not _LETGO:
                    away[kk] = hoard[moniker, kk] = vv
            self.jot('args_filled', tool=moniker, args=_Text._dense(away, 200))
        return away

    def conform_operands(self, moniker: str, operands: dict[str, Any]) -> dict[str, Any]:
        outline = self.outlines.get(moniker) or {}
        characteristics = outline.get('properties') or {}
        if not characteristics:
            return operands
        away: dict[str, Any] = {}
        wandering: dict[Any, Any] = {}
        for (kk, vv) in operands.items():
            if vv is None:
                continue
            if kk not in characteristics:
                wandering[kk] = vv
                continue
            vv = _Schema._conform_worth(characteristics[kk], vv)
            if vv is not _LETGO:
                away[kk] = vv
        lacking = [kk for kk in outline.get('required') or [] if kk not in away]
        for kk in list(lacking):
            genus = {'string': str, 'integer': (int, float), 'number': (int, float), 'array': list, 'boolean': bool}.get((characteristics.get(kk) or {}).get('type'), object)
            for different in sorted(wandering, key=lambda xx: (not isinstance(wandering[xx], genus), -_Text._akin(str(xx), kk))):
                vv = _Schema._conform_worth(characteristics.get(kk) or {}, wandering[different])
                if vv is not _LETGO:
                    away[kk] = vv
                    wandering.pop(different)
                    lacking.remove(kk)
                    break
        if lacking:
            away.update(self.stuff_operands(moniker, away, lacking))
        return away

    def operand_employs(self, moniker: str) -> list[str]:
        return [function for (function, details) in self.functions.items() if isinstance(details, dict) and details.get('name') == moniker]

    @staticmethod
    def operands_vetoed(away: dict[str, Any]) -> bool:
        return bool(re.search('invalid|missing|required|argument|parameter|schema|malformed|validation|unexpected|must be', f"{away['error']} {_Text._dense(away['obs'], 400)}", re.I))

class _Dispatch:
    """Sending calls to the store and reading what comes back."""

    def lot_unfold(self) -> bool:
        return bool(self.functions.get('compare')) and (not getattr(self, 'collate_halted', False)) and (not self.meagre_collate)

    def transmit(self, moniker: str, operands: dict[str, Any]) -> dict[str, Any]:
        away: dict[str, Any] = {'obs': None, 'lines': [], 'error': None, 'ok': False}
        if self.concluded or self.defunct or moniker not in self.outlines:
            away['error'] = 'unavailable'
            return away
        concluding = moniker in {(self.functions.get(stp) or {}).get('name') for stp in ('checkout', 'decline')}
        if self.paces_leftover() <= 0 or (self.paces_leftover() <= 1 and (not concluding)):
            away['error'] = 'no steps left'
            return away
        freight: dict[str, Any] = dict(self.fastening)
        freight['step'] = self.pace + 1
        freight['action'] = {'name': moniker, 'args': operands}
        for _ in range(3):
            leftover = self.duration_leftover()
            if 3 > leftover:
                away['error'] = 'no time'
                return away
            _habitat.timeout = int(max(4, min(90, leftover - 1)))
            consequence = _habitat.post_verbose('/environment/call', freight)
            data = consequence.data
            if isinstance(data, dict) and ('observation' in data or 'finished' in data):
                self.pace = self.pace + 1 if not isinstance(data.get('step'), int) else data['step']
                if data.get('finished'):
                    self.concluded = True
                (observation, strands) = _Shape._unbox(data.get('observation'))
                for strand in strands:
                    if strand not in self.dealt_strands:
                        self.outstanding_strands.append(strand)
                away['obs'] = observation
                away['lines'] = strands
                if not (isinstance(observation, dict) and isinstance(observation.get('error'), str) and (len(observation) <= 2)):
                    away['ok'] = True
                else:
                    away['error'] = observation['error']
                self.jot('call', tool=moniker, args=_Text._dense(operands, 160), result=_Text._dense(observation, 200), lines=strands[:3])
                item_rec = {'step': self.pace, 'tool': moniker, 'args': _Text._dense(operands, 120 if not moniker == (self.functions.get('message') or {}).get('name') else 300), 'result': _Text._dense(observation, 160)}
                if strands:
                    item_rec['shopper_lines'] = [pst[:200] for pst in strands[:3]]
                if moniker == (self.functions.get('message') or {}).get('name'):
                    item_rec['basket_then'] = [list(xx) for xx in self.pannier_imprint()]
                self.annals.append(item_rec)
                return away
            error = consequence.error or {}
            standing = None if not isinstance(error, dict) else error.get('status')
            main_text = str(error if not isinstance(error, dict) else error.get('body'))[:300]
            self.jot('env_failure', status=standing, body=main_text)
            if 402 == standing:
                self.defunct = True
                away['error'] = 'payment'
                return away
            if 409 == standing:
                if 'finished' in main_text:
                    self.concluded = True
                    away['error'] = 'finished'
                    return away
                mm = re.search('expected (\\d+)', main_text)
                if mm:
                    self.pace = int(mm.group(1)) - 1
                    freight['step'] = self.pace + 1
                    continue
                away['error'] = 'conflict'
                return away
            if standing in (403, 422):
                away['error'] = f'refused {standing}'
                return away
        away['error'] = 'env failure'
        return away

    def dispatch(self, moniker: str, operands: dict[str, Any]) -> dict[str, Any]:
        operands = self.conform_operands(moniker, operands)
        away = self.transmit(moniker, operands)
        if away['error'] and self.operands_vetoed(away):
            stable = self.fix_up_operands(moniker, operands, away['obs'])
            if stable is not None and operands != stable:
                away = self.transmit(moniker, stable)
            away['bad_args'] = bool(away['error']) and self.operands_vetoed(away)
        return away

class _Diary:
    """The run journal, and the facts about a store kept for its next task."""

    def recollect(self) -> None:
        _SPHERE_REMEMBRANCE[self.sphere_clef] = {'maps': self.atlases, 'compare_limit': self.collate_boundary, 'thin_compare': self.meagre_collate, 'all_words': self.whole_vocables}

    def jot(self, genus: str, **pillars: Any) -> None:
        item_rec = {'kind': genus, 'step': self.pace, 't': round(self.launched.elapsed, 1)}
        item_rec.update(pillars)
        self.track.append(item_rec)

class _Timekeeping:
    """How much time, how many steps and how much model spend the sortie has left."""

    def expended(self) -> bool:
        return OUTLAY_LID <= self.model.expense

    def paces_leftover(self) -> int:
        return self.upper_paces - self.pace

    def duration_leftover(self) -> float:
        return ASSIGNMENT_SECS - self.launched.elapsed

class Sortie(_Ordering, _Closing, _CartKeeping, _Conditions, _References, _Questioning, _Messaging, _Checks, _Verdicts, _Opening, _ShelfWork, _ValueHunt, _KeywordHunt, _Intake, _FactSheets, _RecordMapping, _ToolSchema, _ArgFitting, _Dispatch, _Diary, _Timekeeping):
    """One task from the shopper's request to the closing call, assembled from the parts above."""

    def __init__(self, problem_data: dict[str, Any]):
        self.launched = _Chronometer()
        habitat = problem_data['environment']
        self.fastening = dict(habitat['binding'])
        self.charter = habitat['policy_view']
        self.appeal = str(self.charter.get('query') or '')
        self.debut = self.charter.get('opening')
        self.utensils = [tl for tl in self.charter.get('tools') or [] if isinstance(tl, dict)]
        self.outlines: dict[str, dict[str, Any]] = {}
        for tl in self.utensils:
            func_ref = tl if not isinstance(tl.get('function'), dict) else tl.get('function')
            if isinstance(func_ref, dict) and func_ref.get('name'):
                self.outlines[func_ref['name']] = func_ref.get('parameters') or {'type': 'object', 'properties': {}}
        self.upper_paces = int(self.charter.get('max_steps') or 30)
        self.pace = 0
        self.concluded = bool(self.charter.get('finished'))
        self.defunct = False
        self.track: list[dict[str, Any]] = []
        self.model = Seer(self)
        self.functions: dict[str, Any] = {}
        self.items: list[Article] = []
        self.lid: float | None = None
        self.thriftiest = False
        self.finest: str | None = None
        self.ratify_initial = False
        self.protocol: list[dict[str, Any]] = []
        self.annals: list[dict[str, Any]] = []
        self.barrier_check_dispatches: dict[str, int] = {}
        self.ratified_imprint: tuple | None = None
        self.missive_purse: int | None = None
        self.messages_transmitted = 0
        self.inside_inventory_just = True
        self.petition_concerning: list[str] = []
        self.conform_canons: list[dict[str, Any]] = []
        self.outstanding_strands: list[str] = []
        self.dealt_strands: set[str] = set()
        self.trolley: dict[str, int] = {}
        self.trolley_article: dict[str, Article] = {}
        self.atlases: dict[str, dict[str, str]] = {}
        self.collate_boundary = 4
        self.folio_dimension = 0
        self.sphere_clef = '|'.join(sorted(self.outlines))
        note = _SPHERE_REMEMBRANCE.get(self.sphere_clef) or {}
        self.atlases = dict(note.get('maps') or {})
        self.collate_boundary = int(note.get('compare_limit') or 4)
        self.meagre_collate = bool(note.get('thin_compare'))
        self.unfolded_folios: set[str] = set()
        self.forgo_grounds = ''
        self.fashion = 'full'
        self.inquiry_quantities: dict[str, int] = {}
        self.whole_vocables: bool | None = note.get('all_words')

    def execute(self) -> list[dict[str, Any]]:
        try:
            if not self.arrange():
                self.oblige_conclude()
                return self.track
            self.toil()
        except Exception as exc:
            self.jot('exception', error=repr(failure)[:300])
        try:
            if not self.concluded:
                self.oblige_conclude()
        except Exception as exc:
            self.jot('exception', error=repr(failure)[:300])
        self.jot('done', cost=round(self.model.expense, 4), llm_calls=self.model.dispatches, steps=self.pace)
        return self.track

    def toil(self) -> None:
        items = self.items
        cnt = len(items)
        queries = (3 if not self.missive_purse is not None else self.missive_purse) - self.withheld_messages()
        if 'message' not in self.functions or 0 > queries:
            queries = 0
        self.allusion_minimum = queries + (0 if not queries else 1) + self.terminus_fallback() + 2 * cnt
        for article in items:
            for demand in list(article.stipulations):
                if demand.citation and demand.operator_ in ALLUSION_OPERATORS:
                    self.work_out_citation(article, demand)
        self.allusion_minimum = 0
        loose = self.paces_leftover() - queries - (0 if not queries else 1) - self.terminus_fallback()
        apiece_article = 0 if not 0 < loose else max(2, loose // max(cnt, 1))
        if any((state_s['moment'] == 'start' for state_s in self.protocol)):
            self.barrier_check('start', 'start the task (search the store)')
            self.knob_strands()
            queries = max(0, min(queries, queries if not self.missive_purse is not None else self.missive_purse - self.messages_transmitted - self.withheld_messages()))
        for (kk, article) in enumerate(items):
            part = max(apiece_article, self.paces_leftover() - queries - 1 - self.terminus_fallback()) if not kk < cnt - 1 else apiece_article
            if 0 >= part:
                break
            prior_to = self.paces_leftover()
            self.survey(article, max(2, part // 2))
            self.heat_up(article, max(0, part - (prior_to - self.paces_leftover())))
        if 0 < queries and self.able_petition():
            wideness = 1 if not 1 < cnt else 2
            design = self.design_queries(queries * wideness)
            sole: list[tuple[Article, str]] = []

            def companion(article: Article, aspect: str, balance: list[tuple[Article, str]]) -> str | None:
                if 2 > wideness:
                    return None
                return next((pp for (art, pp) in balance if art is article and (not _Text._nigh(pp, aspect)) and (_Text._akin(pp, aspect) < 0.5)), None)

            def lay(article: Article, aspect: str, as_well: str | None) -> bool:
                prior_to = len(article.stipulations)
                rejoinder = self.petition(article, aspect, as_well=as_well)
                if isinstance(rejoinder, dict) and str(rejoinder.get('kind') or '').lower() == 'split':
                    if as_well:
                        sole[:0] = [(article, aspect), (article, as_well)]
                    elif self.able_petition() and self.paces_leftover() > self.terminus_fallback():
                        rejoinder = self.petition(article, aspect, thin_down=True)
                self.carry_out_retort_held_on(article, aspect, rejoinder)
                self.knob_strands()
                return len(article.stipulations) != prior_to
            while design or sole:
                if not self.able_petition() or self.paces_leftover() <= self.terminus_fallback():
                    break
                singular = bool(sole)
                (article, aspect) = design.pop(0) if not singular else sole.pop(0)
                as_well = companion(article, aspect, design) if not singular else None
                if as_well:
                    design.remove((article, as_well))
                if as_well and (any((_Text._akin(stp.aspect, as_well) >= 0.67 for stp in article.stipulations)) or self.related_to_previously(article, as_well)):
                    as_well = None
                if any((_Text._akin(stp.aspect, aspect) >= 0.67 for stp in article.stipulations)) or self.related_to_previously(article, aspect):
                    if not as_well:
                        continue
                    (aspect, as_well) = (as_well, None)
                if lay(article, aspect, as_well) and design:
                    leftover = min(len(design), len(design) if not self.missive_purse is not None else self.missive_purse - self.messages_transmitted - self.withheld_messages())
                    design = self.design_queries(leftover * wideness)
            while self.able_petition() and self.paces_leftover() > self.terminus_fallback() + 1 and (self.duration_leftover() > ENFOLD_SECS + 30):
                extra = self.design_queries(1 if not 1 < wideness else wideness * cnt)
                if not extra:
                    break
                (article, aspect) = extra[0]
                lay(article, aspect, companion(article, aspect, extra[1:]))
        for article in items:
            for demand in list(article.stipulations):
                if demand.citation and demand.operator_ in ALLUSION_OPERATORS:
                    self.allusion_minimum = self.terminus_fallback() + 2
                    self.work_out_citation(article, demand)
                    self.allusion_minimum = 0
        if 'ask' == self.fashion:
            return
        self.tentative_payable = True
        self.tentative_trolley()
        self.evaluate_whole()
        purchase = sorted(items, key=lambda art: (bool(art.conforms()), -len(art.reservoir)))
        for (kk, article) in enumerate(purchase):
            part = max(2, (self.paces_leftover() - self.terminus_fallback()) // max(len(purchase) - kk, 1))
            self.survey(article, part, terminal=True)
        self.select_with_conclude()

def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    sortie = Sortie(problem_data)
    try:
        return sortie.execute()
    finally:
        sortie.launched.halt()
__all__ = ['agent_main']