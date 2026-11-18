"""Chidori, third rebuild: a shopping agent for ORO Bench worlds whose decisions are made in Python. A hub
(Venture) holds the task's state and wires nine components (ledger, brief, scout, bench, liaison, inquiry, tracer,
coupler, closer); the long walks are objects of their own (_SearchWalk, _Opening, _Hunt, _HopTarget, _PickBoard),
and every larger step is split into small named moves (intake, duty fitting, goal settling, verdict reading, the
order pass, the gate deed, the ask round). Models only read: the request and the tools once, then small calls that
judge listings, read the shopper's lines and plan the questions. Time is counted from the work done, never read off
a clock."""

from __future__ import annotations

import json
import re
import threading
from difflib import SequenceMatcher
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from src.agent.proxy_client import ProxyClient

READER_SEAT = "anthropic/claude-sonnet-5.5"
PLANNER_SEAT = "anthropic/claude-sonnet-5.5"
CHECKER_SEAT = "anthropic/claude-sonnet-5.5"
QUICK_SEAT = "anthropic/claude-haiku-5.5"
SPARE_SEAT = "openai/gpt-6-luna"

TIME_LIMIT_S = 225.0
CLOSING_S = 45.0
CARD_CHARS = 2200
ROW_TEXT_CHARS = 260
LIGHT_SPEND_USD = 0.03
SPEND_LIMIT_USD = 0.05
_PART_SERIES = re.compile(r"\b([A-Z]{1,5}\d{1,3})-\d{2,}\b")
POINTER_OPS = {"same as reference": "is", "different from reference": "is not", "counterpart of reference": "is",
           "at least reference": "at least", "at most reference": "at most"}

_model_line = ProxyClient(timeout=75, max_retries=1, retry_delay=1)
_store_line = ProxyClient(timeout=90, max_retries=2, retry_delay=1)

_STORE_MEMORY: dict[str, Any] = {}


_FOLD_SPACES = re.compile(r"[^a-z0-9 ]+")
_FIRST_FIGURE = re.compile(r"-?\d+(?:\.\d+)?")
_FENCED = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def _singular(word: str) -> str:
    return word[:-1] if len(word) > 3 and word[-1] == "s" else word


def _plain(text: Any) -> str:
    folded = str(text or "").lower()
    for sign, spoken in (("&", " and "), ("_", " "), ("-", " ")):
        folded = folded.replace(sign, spoken)
    return _FOLD_SPACES.sub(" ", folded).strip()


def _keywords(text: Any) -> set[str]:
    return {_singular(w) for w in _plain(text).split() if len(w) >= 2}


def _all_words(text: Any) -> set[str]:
    return set(map(_singular, _plain(text).split()))


def _prefixes(text: Any) -> set[str]:
    return {w[:4] for w in _keywords(text)}


def _same_name(a: Any, b: Any) -> bool:
    if _plain(a) == _plain(b):
        return True
    left = _keywords(a)
    return bool(left) and left == _keywords(b)


def _names_facet(key_: Any, facet: Any) -> bool:
    """A record key names the property: the same name, or the name with one qualifying word ("clothing material")."""
    if _same_name(key_, facet):
        return True
    wanted, named = _keywords(facet), _keywords(key_)
    return bool(wanted) and wanted <= named and len(named - wanted) <= 1


def _spelled_within(wanted: Any, piece: Any) -> bool:
    """Every wanted word is in the piece, a long word also in a variant spelling ("aluminum" for "aluminium")."""
    return all(word in piece or (len(word) >= 5 and any(len(have) >= 5 and SequenceMatcher(None, word, have).ratio() >= 0.85
                                                          for have in piece))
               for word in wanted)


def _shelf_words(text: Any) -> set[str]:
    """A category name's content words, plurals folded ("accessories" and "accessory" are one word); short joining
    words drop out when longer ones remain."""
    words = {w[:-2] + "y" if len(w) > 4 and w.endswith("ie") else w for w in _keywords(text)}
    longer = {w for w in words if len(w) > 3}
    return longer or words


def _shelf_overlap(a: Any, b: Any) -> float:
    left, right = _shelf_words(a), _shelf_words(b)
    if not (left and right):
        return 0.0
    return len(left & right) / len(left | right)


def _overlap(a: Any, b: Any) -> float:
    left, right = _keywords(a), _keywords(b)
    if not (left and right):
        return 0.0
    return len(left & right) / len(left | right)


def _balanced_end(text: str, start: int, opener: str, closer: str) -> int | None:
    level, quoted, escaped = 0, False, False
    for at in range(start, len(text)):
        sign = text[at]
        if quoted:
            if escaped:
                escaped = False
            elif sign == "\\":
                escaped = True
            elif sign == '"':
                quoted = False
        elif sign == '"':
            quoted = True
        elif sign == opener:
            level += 1
        elif sign == closer:
            level -= 1
            if level == 0:
                return at
    return None


def _first_json(text: str) -> Any:
    if not text:
        return None
    body = text.strip()
    fenced = _FENCED.search(body)
    if fenced:
        body = fenced.group(1).strip()
    for opener, closer in ("{}", "[]"):
        start = body.find(opener)
        if start < 0:
            continue
        end = _balanced_end(body, start, opener, closer)
        if end is None:
            continue
        try:
            return json.loads(body[start:end + 1])
        except Exception:
            continue
    try:
        return json.loads(body)
    except Exception:
        return None


def _short_json(value: Any, limit: int) -> str:
    dumped = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if len(dumped) > limit:
        return dumped[:limit] + "…"
    return dumped


def _amount(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None
    found = _FIRST_FIGURE.search(value.replace(",", ""))
    if found is None:
        return None
    try:
        return float(found.group(0))
    except Exception:
        return None


def _wordy(value: Any) -> bool:
    return isinstance(value, str) and len(value) > 24 and " " in value.strip()


def _row_list(obs: Any, depth: int = 0) -> list[dict[str, Any]] | None:
    if isinstance(obs, list):
        return obs if obs and all(isinstance(row, dict) for row in obs) else None
    if not isinstance(obs, dict) or depth >= 2:
        return None
    longest = None
    for part in obs.values():
        found = _row_list(part, depth + 1)
        if found and (longest is None or len(found) > len(longest)):
            longest = found
    return longest


def _peel(obs: Any) -> tuple[Any, list[str]]:
    if not isinstance(obs, dict):
        return obs, []
    said: list[str] = []
    kept: dict[str, Any] = {}
    records_beside = sum(isinstance(part, dict) for part in obs.values())
    own_fields = any(isinstance(part, str) and part.strip() and " " not in part.strip() for part in obs.values())
    for key, part in obs.items():
        if isinstance(part, list) and part and all(map(_wordy, part)):
            said += [str(line) for line in part]
        elif _wordy(part) and records_beside == 1 and len(obs) <= 3 and not own_fields:
            said.append(str(part))
        else:
            kept[key] = part
    if said and len(kept) == 1:
        only = next(iter(kept.values()))
        if isinstance(only, dict):
            return only, said
    return kept, said


_REPLY_RATES = (("haiku", 0.62, 209.0), ("sonnet", 2.35, 147.0))
_OTHER_RATE = (2.84, 78.0)
_RATE_MARGIN = 1.5
_UNCOUNTED_REPLY = 4.0
_STORE_CALL = 2.0
_COUNTING = [True]


class CountedClock:
    """Seconds counted from the replies, store calls and pauses of the task; side-by-side calls count as one lane."""

    def __init__(self) -> None:
        self.total = 0.0
        self._lanes = threading.local()
        self._lock = threading.Lock()

    def charge(self, secs: float) -> None:
        if not _COUNTING[0] or secs <= 0:
            return
        lane_fn = getattr(self._lanes, "used", None)
        if lane_fn is not None:
            lane_fn[0] += secs
            return
        with self._lock:
            self.total += secs

    def elapsed(self) -> float:
        lane_fn = getattr(self._lanes, "used", None)
        return self.total + (lane_fn[0] if lane_fn is not None else 0.0)

    def reply(self, seat: str, payload_in: Any) -> None:
        used_meter = payload_in.get("usage") if isinstance(payload_in, dict) else None
        token_count = used_meter.get("completion_tokens") if isinstance(used_meter, dict) else None
        if not isinstance(token_count, (int, float)) or isinstance(token_count, bool):
            self.charge(_UNCOUNTED_REPLY)
            return
        settled, speed = next(((fields_, want) for letters, fields_, want in _REPLY_RATES if letters in seat), _OTHER_RATE)
        self.charge(_RATE_MARGIN * (settled + float(token_count) / speed))

    def lane(self, task_fn: Any) -> tuple[Any, float]:
        self._lanes.used = [0.0]
        try:
            return _guarded(task_fn), self._lanes.used[0]
        finally:
            self._lanes.used = None


def _pause(meter: CountedClock, secs: float) -> None:
    meter.charge(secs)
    if _COUNTING[0]:
        threading.Event().wait(secs)


def _TEST_FREEZE_CLOCK() -> None:
    _COUNTING[0] = False


def _side_by_side(task_fns: list, meter: CountedClock, lanes_n: int = 3) -> list:
    if len(task_fns) <= 1:
        return [_guarded(job_) for job_ in task_fns]
    pool_width = min(lanes_n, len(task_fns))
    with ThreadPoolExecutor(max_workers=pool_width) as seen_pool:
        finished_walk = list(seen_pool.map(meter.lane, task_fns))
    lane_times = [lane_used for _, lane_used in finished_walk]
    meter.charge(max(max(lane_times), sum(lane_times) / pool_width))
    return [amount_ for amount_, _ in finished_walk]


def _guarded(job: Any) -> Any:
    try:
        return job()
    except Exception:
        return None


class Oracle:
    """Model calls: the fallback seats, the spend and the counted seconds of every reply."""

    def __init__(self, venture: "Venture"):
        self.venture = venture
        self.spent_usd = 0.0
        self.requests = 0
        self.misses = 0

    def ask_text(self, seat: str, system_text: str, user_msg: str, token_cap: int) -> str | None:
        seat = self._seat_for(seat)
        remaining = self.venture.ledger.seconds_left()
        if remaining < CLOSING_S - 12:
            return None
        _model_line.timeout = int(max(10, min(45, remaining - CLOSING_S + 20)))
        self.requests += 1
        response = self._post_with_retry(self._request_body(seat, system_text, user_msg, token_cap))
        answered = isinstance(response.data, dict) and bool(response.data.get("choices"))
        if answered:
            said = self._reply_text(seat, response.data)
            if said:
                return said
        broken = response.error if isinstance(response.error, dict) else {}
        if not answered:
            self.venture.meter.charge(float(_model_line.timeout) if broken.get("status") is None else 1.0)
        self.misses += 1
        self.venture.ledger.jot("llm_failure", model=seat, error=str(getattr(response, "error", ""))[:200])
        return None

    @staticmethod
    def _fallbacks(seats: list[str]) -> list[str]:
        """The seats in order, with the other Claude model tried before the spare one once a Claude seat has failed."""
        order: list[str] = []
        for seat in seats:
            if seat == SPARE_SEAT and any(tried.startswith("anthropic/") for tried in order):
                order += [other for other in (PLANNER_SEAT, QUICK_SEAT) if other not in order]
            if seat not in order:
                order.append(seat)
        return order

    def ask_json(self, seats: list[str], system_text: str, user_msg: str, token_cap: int) -> Any:
        for seat in self._fallbacks(seats):
            phrase = self.ask_text(seat, system_text, user_msg, token_cap)
            decoded = _first_json(phrase or "")
            if decoded is not None:
                return decoded
        return None

    def _seat_for(self, seat: str) -> str:
        if self.spent_usd >= LIGHT_SPEND_USD and seat in (PLANNER_SEAT, CHECKER_SEAT, READER_SEAT) and seat != QUICK_SEAT:
            return QUICK_SEAT
        return seat

    @staticmethod
    def _request_body(seat: str, system_text: str, user_msg: str, token_cap: int) -> dict[str, Any]:
        system_block: Any = system_text
        if seat.startswith("anthropic/"):
            system_block = [{"type": "text", "text": system_text, "cache_control": {"type": "ephemeral"}}]
        return {
            "model": seat,
            "messages": [{"role": "system", "content": system_block}, {"role": "user", "content": user_msg}],
            "temperature": 0,
            "max_tokens": token_cap,
        }

    def _post_with_retry(self, request_body: dict[str, Any]) -> Any:
        response = _model_line.post_verbose("/inference/chat/completions", request_body)
        for round_no in range(2):
            if isinstance(response.data, dict) and response.data.get("choices"):
                break
            failure = response.error if isinstance(response.error, dict) else {}
            if failure.get("status") not in (429, 500, 502, 503, 504) or self.venture.ledger.seconds_left() < CLOSING_S + 10:
                break
            _pause(self.venture.meter, 2.0 * (round_no + 1))
            response = _model_line.post_verbose("/inference/chat/completions", request_body)
        return response

    def _reply_text(self, seat: str, payload_in: dict[str, Any]) -> str | None:
        self.venture.meter.reply(seat, payload_in)
        used_meter = payload_in.get("usage") or {}
        if isinstance(used_meter.get("cost"), (int, float)):
            self.spent_usd += float(used_meter["cost"])
        said = (payload_in["choices"][0].get("message") or {}).get("content")
        if isinstance(said, list):
            said = "".join(str(issue.get("text") or "") for issue in said if isinstance(issue, dict))
        return str(said) if said else None


INTAKE_PROMPT = """Your job is to brief a shopping assistant for a single task inside a simulated store. The input is the shopper's request and the list of the store's tools (OpenAI function schemas). Answer with ONE JSON object and nothing more.

{
 "items": [  // one entry for each distinct product the shopper means to buy; most requests have one
   {"name": "a plain noun phrase naming the product the way the request does (for a list, that line's own words)",
    "category": "the store category or the kind of product asked for, in the request's wording",
    "quantity": 1,
    "requirements": [ // each firm condition the request puts on THIS product's properties; never price, stock, quantity or budget
      {"property": "the property name a listing would use", "op": "is|is not|at least|at most|one of|contains|same as reference|different from reference|counterpart of reference|at least reference|at most reference",
       "value": "the value, bound or list exactly as written in the request (for 'one of': the values joined by ' | ')",
       "reference": null  // only for the ops that end in 'reference': {"category": "the kind of product only (describing words such as who it is for, a material or a style go into attributes)", "seller": "...", "attributes": {"property": "value", ...}, "property": "the property to look up on that page", "count": null or the number of listings the description says match, "pick": null or the rule that chooses among several (e.g. "whichever is wider")}
      }],
    "hints": ["property names the story suggests the shopper secretly cares about (a remark about who will wear it suggests a gender or age group; one about what sits inside a computer suggests the processor; where it will stand or what it sits beside suggests a size such as its height); empty when there are none"],
    "search_terms": ["2-4 queries for the store's search box, the most specific first. A query only matches listings holding all of its words, so keep each one short: the product noun with one or two value words (a colour, a material, a value with its unit such as 5v or 16gb); never a bare property name, never words that only identify a referenced listing; the last query is the bare product noun"]
   }],
 "total_cap": 12345 or null,        // the highest total price as a plain number in the request's own unit (cents stay cents)
 "cheapest": false,                 // true only when the request asks for the cheapest / lowest price / cheapest total
 "best": null,                      // the measure, when the request wants the best by some measure; otherwise null
 "message_budget": null,            // the number of messages the shopper allows, as an integer, or null
 "call_budget": null,               // the number of tool calls / steps the request allows, as an integer, or null
 "in_stock_only": true,             // the request wants only what can actually be bought now
 "ask_about": ["topics the request tells the assistant to bring up with the shopper, in the request's words; empty when none"],
 "fit_rules": [{"accessory": "item name", "host": "item name"}],   // rules of the kind 'make sure X fits Y'; empty when none
 "conditions": [{"if": {"item": "item name", "property": "...", "values": ["..."]}, "then": {"item": "item name", "property": "...", "op": "is|is not|at least|at most|one of", "value": "..."} or null, "otherwise": the same shape or null}],   // a condition on one item that depends on how another item turns out; empty when none
 "decline_rule": null,              // what the request says to do when nothing fits (say so plainly, offer nothing else, ...), or null
 "procedure": [                     // steps the request demands in HOW the task is done (not product conditions, budgets, message or step limits, and not the no-fit case, which is decline_rule): an approval or confirmation to obtain, something to show or tell the shopper, a tool to use, an order of steps; empty when none
   {"moment": "start|before_order|before_decline|before_finish", "kind": "message|tool", "tool": "tool name or null", "goal": "what has to happen, in the request's terms", "messages": 1}  // messages: how many shopper messages the step uses (0 for a tool); a check made in the store itself (stock, availability, a value) is kind "tool" with 0 messages unless the request wants to hear the result
 ],
 "tools": {
   "search_text": {"name": "tool", "query": "the argument for the keywords", "page": "the argument that moves to further results (a page number, a row offset, or a token from the last result), or null",
                   "page_kind": "number|offset|token or null", "mode": {"arg": "the argument that selects keyword search", "value": "its value"} or null},
   "search_fields": {"name": "the tool that searches by category/brand/price fields, or the same tool in its field mode", "category": "arg or null", "brand": "arg or null", "price_max": "arg or null", "page": "arg or null", "page_kind": "number|offset|token or null", "mode": {"arg": "...", "value": "..."} or null} or null,
   "open": {"name": "the tool that returns one listing in full", "id": "its id argument"},
   "facts": {"name": "a separate tool that returns a listing's property table, when the store has one", "id": "arg"} or null,
   "compare": {"name": "the tool that returns several listings in full", "ids": "its list argument"} or null,
   "cart_add": {"name": "the tool that puts a listing into the cart", "id": "arg", "qty": "the quantity arg or null",
                "mode": {"arg": "the argument that chooses add or remove", "add": "the value that adds", "remove": "the value that removes"} or null},
   "cart_set": {"name": "the tool that replaces the whole cart with a list of lines, when there is one", "lines": "its list argument", "id": "the listing id key inside a line", "qty": "the quantity key inside a line"} or null,
   "cart_view": {"name": "the tool that shows the cart"} or null,
   "checkout": {"name": "the tool that places the order (final)", "args": {"argument": "value"} or null},
   "decline": {"name": "the tool that ends without buying (final)", "remark": "text argument or null", "args": {"argument": "value"} or null},
   "message": {"name": "the tool that sends the shopper a message", "text": "its text argument"},
   "status": {"name": "the tool that reports the allowances left"} or null
 }
}

How to read the request:
- A requirement is only what the request firmly demands. Wishes (ideally, it would be nice) and story details (who it is for, where it will go) are NOT requirements; whatever they suggest goes into "hints". Words that merely name the kind of product or its category (men's, women's, kids' in "men's rain capes"; "gaming" inside a category name) go into "name"/"category", never into "requirements". Yet a value the request gives to a named property ("in the <value> style", "with <value> as its <property>", "a <property> of <value>") IS a requirement on that property, even if the value sounds like a kind of product; the category stays the product the request names.
- "3 x gadgets where the colour is red and the size is not large" is one item with quantity 3 and two requirements (colour is red; size is not large).
- A size given for what a single listing holds (a pack, set or box of N pieces, a pack holding N) is a requirement on that pack or set size; "quantity" still counts the listings to buy.
- "quantity" is the count the request says to order now ("ordering one", "get me 2 x ..."); people in the story who might each use one do not raise it. Without a stated count it is 1.
- A remark that only restates what a requirement already implies ("zip front only, so nothing with buttons or toggles") is ONE requirement (closure: zip front), not several; never invent properties for what it rules out.
- "A or B" for a property is op "one of" with value "A | B". "not X" / "anything but X" is op "is not". "at least N units" is "at least" with value "N units". Keep each value's unit and spelling as written.
- A requirement that points at ANOTHER listing's page (one named by its category, its seller and some attribute values): wanting that page's own value is op "same as reference"; wanting anything but that value is op "different from reference"; wanting the counterpart of that value (the value that page or the store pairs with it, the one that goes with it; wording such as "compatible with", "works with", "plays nicely with" or "matches up with" what the page shows) is op "counterpart of reference"; wanting a value no lower (or no higher) than that page's is op "at least reference" (or "at most reference"). The page named may itself only point on to another listing for that property: the assistant follows such pointers, so describe the page the request names. Fill "reference" with the category, the seller, the attributes given and the property to read. When the request says several of that seller's listings match and gives a rule for choosing one (the larger, the cheaper, the one with more of something), put the rule in "pick" and the number in "count"; "attributes" hold only plain identifying values, never the comparison. The attribute values that identify the other listing belong to THAT listing alone: they are never requirements on the requested product and never its search terms. Made-up example: "for the colour, avoid whatever Fenwick's garden hose page (length 20 m, material rubber) states" gives one requirement {"property": "colour", "op": "different from reference", "value": "", "reference": {"category": "garden hoses", "seller": "Fenwick", "attributes": {"length": "20 m", "material": "rubber"}, "property": "colour"}} and nothing about length or material.
- A requirement on one item that only applies when another item turns out a certain way ("should X's P turn out to be V, I need Y's Q to be W"; "Y's Q must be W, except when X's P is V") belongs in "conditions" (in the first case "then" applies when X's P is one of the values; in the second, "otherwise" applies when it is not), never in "requirements" or "procedure".
- The category is the kind of product; when the request names a store category (say "the category is garden tools"), use that exact wording.
- Money: when the request gives the total in cents, "total_cap" is that number of cents.
- Use the tool and argument names exactly as the schemas spell them. When ONE tool ends the task either way, with an argument choosing between buying the cart and leaving without buying, "checkout" and "decline" both name that tool, each with "args" holding the argument value that selects it (exactly as the schema lists it). A tool described as final that closes without buying is "decline"; the one that checks out the cart is "checkout". When one search tool has an enum argument switching between keyword search and field search, "search_text" and "search_fields" both name it with the matching mode values."""

ROWS_PROMPT = """You sort search results for a shopping assistant. For each numbered row (id, title, short text, category, price) decide whether the listing COULD satisfy the requirements once its full property page is read. Use only what the row shows: a row silent about a requirement could still satisfy it ("maybe"); a row whose own words plainly contradict a required value is "no"; a row that itself shows the required values is "yes". The store's category label tells what kind of product a row is: a row in the requested category is the right kind whatever its title suggests, so never say "no" over the kind of product. Answer with ONE JSON object and nothing more: {"yes": [row numbers], "no": [row numbers]}; any row in neither list counts as "maybe"."""

ASSESS_PROMPT = """For a shopping assistant, test store listings against the shopper's requirements, relying on nothing but each listing's own records. Reading rules:
- A requirement holds once the listing's own property for it gives the value. Key spellings vary (spaces, underscores, prefixes, abbreviations), so any key that names the property counts, and one property may appear under several keys. Compare values by meaning: a single element of a comma-separated list suffices, singular and plural are equal, close synonyms and obvious abbreviations are accepted, case does not matter, and values with units are compared after converting the units (numbers as numbers; a bound includes its edge; a range satisfies a bound only when the whole range does).
- The property table comes first. If any property covers the requirement, its value decides, and a different value fails it even when the title claims otherwise. Only if NO key covers it may the seller's notes, the options or variants field, the title or the description supply the value; if nothing supplies it, the requirement is unaddressed ("unsure"). A value found under some other, related property does not count (a handle is no lock type; a lining material is not the material).
- An "is not" or "different from" requirement fails when the listing's property gives the excluded value; "one of" needs one of the listed values; "contains" needs the value among those given. A requirement that a feature be absent (a value of none, no or without) holds when no record mentions the feature.
- The category label settles what kind of product a listing is: carrying the requested category makes it the right kind whatever the title says. Only a listing with no label at all can fail for being an accessory, a spare part or a case instead of the product.
- Compatibility with another product (given by model names or codes) holds only if this listing's own text lists one of those names among what it fits or works with. If the other product's page names a series or family of compatible part codes, a listing whose own part code (often in brackets in the title) starts with that series satisfies it. A sentence claiming that some OTHER part fits a product applies to the listing whose title carries that part's code, not to the page where the sentence appears; "related_pages" attached to a listing show such pages, and what they say about this listing's own part code counts for it. A mere pointer to another page proves nothing.
- Seller notes may hand one of the listing's properties over to ANOTHER listing (for that property, use another listing's value, not this one's). The listing's own value for that property then counts for nothing: a requirement on it is "unsure" unless a related page shown with the listing gives it. Notes handing over a property that no requirement concerns change nothing.
- STORE FIELDS, when supplied, show per required property which keys and values this store's read listings of the category use. If some of those listings write the exact required value in that field, a listing whose field holds a different value fails, even a related or broader one; a value that no read listing writes may still be met by a clear synonym.
- Assess only the requirements given; add none of your own.
The requirements are numbered R1, R2, ... Reply with ONE JSON object only (no commentary): {"verdicts": [{"id": "...", "failed": ["R numbers of requirements the records clearly fail"], "unsure": ["R numbers no record addresses"], "weak": ["R numbers met only by the title, the description or an options field, not by a property or a seller note"], "loose": ["R numbers met by a synonym, a broader term or a conversion rather than the value's own words (for example 'female' for a required 'girls')"]}, ...]} covering every listing and leaving out empty lists. A listing without "failed" and without "unsure" satisfies everything shown."""

ASK_PLAN_PROMPT = """You decide which questions a shopping assistant puts to a shopper. The shopper keeps hidden requirements: each one is a value of ONE property of one requested product, and the shopper tells it only when a message asks about exactly that property (by its name or an obvious synonym). A message gets an answer only about the FIRST property it names; a question about anything the shopper does not care about earns "no preference", and the message is gone.
The input is the request and, for each requested item: its name, the requirements known so far, hints from the story, and a property menu: the properties its candidate listings really carry (how many listings carry each, with sample values).
Pick exactly QUESTION_COUNT questions, best first, as JSON: {"questions": [{"item": "the item name exactly as given", "property": "the property name exactly as the menu spells it (words with spaces)"}, ...]}.
How to pick:
1. First the properties the story or the hints point to (what the request says the shopper will tell when asked: who will use or wear it points to an age group or a gender; where it will stand or what it sits beside points to a size such as its height; "how it holds on" points to a closure; what is inside points to a processor; "how it looks" points to colour, pattern or style). Match each hint to the nearest property name in the menu.
2. Then the properties a buyer of that kind of product most often has a view on, preferring those many candidate listings carry (the "listings" count): what it is made of, how it looks, its size or fit, its type or style, its variant such as a flavour, who it is for, its main technical figure, with values that differ from listing to listing. A property with one value everywhere, a pseudo-property such as an option or variant picker, an identifier, a model number, a warranty or shipping note, makes a poor question.
3. Never ask about a property a known requirement already fixes, nor about price, stock or quantity, nor twice about one property (two spellings of a property are one property). Share the questions among the items by how uncertain each is; an item with strong hints gets the first question."""

STORY_PROMPT = """You help a shopping assistant pick what to ask a shopper. The shopper keeps hidden requirements: each is a value of ONE property of one requested product, told only in answer to a question about that very property. The input is the request and, for each requested item, its known requirements and a property menu (the properties its candidate listings carry, with counts and sample values).
Answer with ONE JSON object and nothing more:
{"hinted": [{"item": "the item name exactly as given", "property": "the menu property name exactly as spelled", "why": "the request's words that point to it"}],
 "junk": ["menu properties that are no real product characteristic a shopper would state: option or variant pickers, identifiers, model numbers, warranty, shipping or packaging notes, repeats of a known requirement"]}
"hinted" holds only properties that the request's story, or the topics it says to raise, point to (a remark about the wearer points to a gender or age group property; where it will stand or what it sits beside points to a size such as its height; "how it holds on" points to a closure; what sits inside a computer points to a processor; "how it looks" points to colour, pattern or style; a topic to raise points to its property). Best first, three at most, empty when the request hints at nothing. Never list a property a known requirement already fixes."""

ANSWER_PROMPT = """A shopping assistant asked the shopper about one property of one product; read the shopper's reply and say what it means. Reply with ONE JSON object only:
{"kind": "need|repeat|free|split|unclear|other",
 "property": "the property the shopper speaks about (their own words, or the property asked about when that is what they mean)",
 "op": "is|is not|at least|at most|one of|same as reference|different from reference|counterpart of reference|at least reference|at most reference",
 "value": "the value, bound or list the shopper gave (list values joined by ' | ')",
 "reference": null or {"category": "...", "seller": "...", "attributes": {"property": "value"}, "property": "the property to read on that page", "count": null or how many listings it says match, "pick": null or the rule for choosing one of several},
 "budget": null or a number when the shopper names a new total they can spend,
 "search": "" or, for a need, a short store search (two to four words) for listings that meet it, worded the way such listings are titled (a product line or model name that carries the value, or the value with the product's noun),
 "answer": "" or, when the shopper asked the assistant something, a short plain answer,
 "steps": [] or the steps the shopper now demands in how the task is done (an approval to get, something to show or tell them, a tool to use before ordering), each {"moment": "before_order|before_decline|before_finish", "kind": "message|tool", "tool": null, "goal": "what has to happen", "messages": 1}}
Kinds:
- "need": the shopper insists on a value, bound, exclusion or list, however it is phrased (settled, firm, nothing but, only X will do, anything bar X, X or Y and nothing else). "Anything bar / other than X" becomes op "is not" with value X. If the shopper refers to ANOTHER listing's page (given by category, seller and attribute values): wanting that page's own value is op "same as reference" with "reference" filled in; wanting anything but it is op "different from reference"; wanting the value the page pairs with its own (its counterpart, what goes with it, what is compatible with or plays nicely with it) is op "counterpart of reference"; wanting a value no lower or no higher than the page's is op "at least reference" or "at most reference".
- "repeat": instead of answering about the property asked, the shopper restates a requirement already shown under KNOWN NEEDS (same property and value).
- "free": they do not mind, leave it open, anything goes, never thought about it.
- "split": they want the message split up, its topics asked separately, or only its first part answered (it covered more than one topic).
- "unclear": they did not understand the message, or ask to have it repeated or simplified.
- "other": everything else (a greeting, a question back, a budget change).
"property" is the property named in the reply itself when it names one (in the shopper's words), which may differ from the one asked; only when the reply names none is it the property asked about.
Explicit bound words such as "at most N", "at least N", "no more than N" or "under N" decide the op and the value, even inside an oddly built sentence (for example "nothing that falls short of at most 34 inches" is op "at most", value "34 inches")."""

NEWS_PROMPT = """While a shopping assistant was working, a store tool's result carried a line written by the shopper. Work out what that line changes and reply with ONE JSON object:
{"changes": [{"kind": "budget|drop|add|price|stock|step|other",
  "budget": null or the new highest total as a number,
  "property": null or the property a dropped or added requirement concerns,
  "op": "is|is not|at least|at most|one of", "value": null or the new requirement's value,
  "item": null or the requested product it is about, "note": "one short sentence"}]}
List every change the lines make, one entry per change: several lines (or one line saying two things, such as the item selling out and its price moving) give several entries.
Kinds:
- "budget": the shopper can now spend a different amount, higher or lower. Copy the figure exactly as written and keep the line's own unit; cents are never converted.
- "drop": some property is no longer required ("let X go", "drop X", "never mind X").
- "add": a requirement that was not there before.
- "price": the chosen listing now costs something else.
- "stock": the chosen listing can no longer be bought.
- "step": a new demand about how the work is done (get an approval, show or tell them something, use a tool before ordering); describe it in "step": {"moment": "before_order|before_decline|before_finish", "kind": "message|tool", "tool": null, "goal": "what has to happen", "messages": 1}."""

PAGE_VALUE_PROMPT = """A shopping assistant is following a reference and hands you one listing page from the store. It needs to know what this page gives for one named property. The value may sit in the property table or in the seller's notes; alternatively the notes may tell the reader to take that property from a DIFFERENT listing (identified by category, seller and a few attribute values) rather than from this page. The notes may also state which value of the property goes with which, either a single pair for this page's value or the store's complete list of pairs.
Reply with ONE JSON object: {"value": "this page's own value as written, or null", "paired": "the value the page pairs with that value, or null", "redirect": null or {"category": "...", "seller": "...", "attributes": {"property": "value"}, "property": "..."}}
If the notes send the WANTED property elsewhere, return that redirect and leave the value out; notes sending some unrelated property elsewhere do not count as a redirect. In a redirect, a name written just before the other listing's product kind ("<Name> <product kind>") is that listing's seller, and the product kind is its category; never carry this page's own seller or category over to it. If the notes give the value plainly, return it. Failing that, read the value from the property table under any key that names the property, and when no key does, from the options field, the title or the description if they give it. A sentence that merely lists the store's pairs does not reveal which value belongs to this page. Fill "paired" only from a pairing the page states: in a list of pairs, take the pair containing this page's own value (a pair works in both directions); leave it null when the page states no pairing or this page's own value is unknown."""

KEYWORDS_PROMPT = """You write search-box queries for a shopping assistant. A query matches only listings whose title, brand, category or property values hold every one of its words (word stems match; tiny words are ignored), and results come in a fixed order with few rows per page, so a query must be narrow enough that most of its matches satisfy the requirements. The input is the product, its requirements, the queries tried already and some listings seen (with whether they met the requirements). Answer with ONE JSON object and nothing more: {"queries": ["...", ...]} holding up to 5 new queries of 1-3 words each, best first: the product's own noun as the store's titles word it plus a required value as listings would write it, other words the store uses for that product or value, never a query tried already."""

ENDING_ARG_PROMPT = """A store tool that ends a shopping task needs an argument saying how it ends. The input is the tool's description, the arguments it still needs (with the values each one allows) and the goal. Answer with ONE JSON object and nothing more: {"<argument name>": "<the allowed value that reaches the goal>", ...}."""

PROCESS_PROMPT = """At one point of a shopping task, compare what the assistant has done with what the request says the process must include. Given to you: the request; the process steps it demands; the point reached (what the assistant intends to do next); the current basket; the allowances still open; the store's tools (minus those that end the task or edit the cart, which the assistant handles itself); and the full list of calls made so far with their results. Work out which demanded steps are owed at THIS point, before the assistant continues, and have not been carried out for the situation as it is now. A yes or a confirmation only counts for the basket in its present form; one given for an earlier basket counts for nothing. A step already carried out for the present situation is not owed twice, and a step that belongs to a different point is not owed now. Reply with ONE JSON object only: {"actions": [{"tool": "exact tool name", "args": {"argument": "value", ...}, "why": "the demanded step it carries out"}]}, in the order to run them and no more than two, or {"actions": []} when nothing is owed. Each shopper message has to make sense by itself: when it concerns the basket, list every line with its quantity, title and price, give the total, and state clearly what you want from the shopper. Do not ask about product properties here, and keep the final message for a demanded step only. Checks inside the store (stock, availability, a value) use the store's reading tools rather than a message, unless the request wants to be told; since the assistant reads every listing before buying it, such a check is owed only when the basket's listings were never read."""

ORDERING_PROMPT = """Every listing below already satisfies the shopper's requirements; the shopper now wants the top one by a stated measure, i.e. the highest or lowest value of something. For each listing find its value for that measure in its properties, its title or its description, and convert units where they differ so that the values can be put side by side. Reply with ONE JSON object only: {"order": ["id", ...], "values": {"id": "the value as read"}}. The order must hold every listing id with the best first; listings whose value you cannot find go to the end, and between equal values the cheaper listing comes first."""

LAST_LOOK_PROMPT = """You look over a shopping assistant's search for one product before it gives up on it. The input is the product, its requirements, the budget, the queries tried already and every listing seen: READ ones (their properties, notes, and the requirements the assistant judged failed or unaddressed) and rows seen only in search results. Answer with ONE JSON object and nothing more: {"fits_after_all": ["ids of READ listings that do satisfy every requirement after all, judged from their own records: a property's value decides, a different value fails, a property no record states is not met"], "unopened": ["ids of rows not read yet that most likely satisfy the requirements, best first, at most 4"], "queries": ["at most 2 new search queries of 1-3 words likely to find fitting listings, never one tried already"]}. Any list may be empty."""

CONDITION_PROMPT = """The shopper made a requirement depend on another product, and a shopping assistant now needs to know how one listing stands. Using only that listing's own records, tell whether its value for the given property equals one of the given values (equal in meaning, with units converted). Reply with ONE JSON object only: {"holds": true or false, "value": "the listing's value as written, or null"}. If none of its records gives that property, set "holds" to false and "value" to null."""

PICK_PAGE_PROMPT = """A shopping assistant wants one specific listing, described by a category, a seller and a few attribute values; the candidates and their records are below. Reply with ONE JSON object only, no explanation: {"id": "the id of the listing that matches the description, or null if none does"}. To match, a candidate needs the same seller, a category label that names the described kind, and records agreeing with every described attribute (equal in meaning, with units converted). A described category that carries extra words around the kind (who it is for, a material, a sleeve or style word) only needs a label naming the kind; the extra words are attributes the records must agree with. Where a candidate's property table is missing, treat its seller notes, title and description as its records: notes stating the described values make it a match. If a PICK rule is given, more than one candidate may match; apply the rule among those that match and return the one it picks."""

FINAL_CHECK_PROMPT = """Before a shopping assistant orders, you give the last verdict on one listing. Test it against EVERY listed requirement using nothing but its own records. A requirement holds when the listing's own property for it gives the value: any key naming the property counts, and so do comma lists, synonyms, abbreviations, singular or plural forms and converted units. When no key covers a requirement, the seller's notes, the options field, the title or the description may give it; a requirement no record gives is not met; a value filed under another property does not count. Compatibility holds when the listing names the other product among what it fits, or when the other product's page names a series of part codes and this listing's own part code begins with it. A property that the notes send to another listing is not met by this listing's own value. Exclusions and closed lists are binding. A listing carrying the requested category label is the right kind of product, whatever its title says. Price times quantity must stay within the budget, in the unit of the prices shown, and a listing the records show as unavailable fails. Judge the listed requirements only; never invent one.
Answer with ONE JSON object: {"fits": true or false, "problems": ["each requirement that is clearly not met or that no record addresses"]}."""

SHELF_PROMPT = """A store files its listings under category labels, and a shopper asked for one kind of product. Among the labels seen, name every label that IS that kind of product (the label whose meaning matches the request's wording as a whole, a wider label that contains it included, such as "sweaters and cardigans" for a cardigan). Leave out neighbouring kinds (accessories, parts, cases for the product, a narrower or different type). Answer with ONE JSON object and nothing more: {"labels": ["label exactly as given", ...]} (empty when none matches)."""

KEYS_PROMPT = """Below is a JSON record a store tool sent back. For each role listed, give the exact name of the key in the record that fills it, or null if no key does. Reply with ONE JSON object:
{"id": "the listing's identifier", "title": "its name", "text": "its description text", "price": "its price", "stock": "whether it can be bought", "category": "its category label", "brand": "its brand", "seller": "its seller or shop", "facts": "the object that holds its property table", "notes": "the seller's free-text notes on it", "total": "a cart's total price", "lines": "a cart's list of lines", "next": "the token or marker for the next page of results", "count": "the total number of matching results"}"""


class Offer:
    """One listing as the store showed it, merged from its search rows and its own page."""

    def __init__(self, ident: str):
        self.oid = ident
        self.heading = ""
        self.blurb = ""
        self.unit_price: float | None = None
        self.can_buy: bool | None = None
        self.shelf = ""
        self.make = ""
        self.vendor = ""
        self.table: dict[str, Any] = {}
        self.remarks = ""
        self.is_open = False
        self.page_seen = False
        self.raw_record: dict[str, Any] = {}
        self.row_verdict = "maybe"

    def card(self, cap_chars: int = CARD_CHARS) -> dict[str, Any]:
        record_: dict[str, Any] = {"id": self.oid, "title": self.heading[:160]}
        for key_, val in (("category", self.shelf), ("brand", self.make), ("seller", self.vendor)):
            if val:
                record_[key_] = val
        for key_, val in (("price", self.unit_price), ("can_buy", self.can_buy)):
            if val is not None:
                record_[key_] = val
        if self.table:
            record_["properties"] = self.table
        if self.remarks:
            record_["seller_notes"] = self.remarks[:600]
        if self.blurb:
            record_["description"] = self.blurb[:400 if self.table else 500]
        return self._fit_card(record_, cap_chars)

    def _fit_card(self, record_: dict[str, Any], cap_chars: int) -> dict[str, Any]:
        size = len(json.dumps(record_, ensure_ascii=False))
        if size > cap_chars and self.table:
            record_["properties"] = dict(list(self.table.items())[: max(6, int(len(self.table) * cap_chars / size))])
        if len(json.dumps(record_, ensure_ascii=False)) > cap_chars and record_.get("description"):
            record_["description"] = record_["description"][:200]
        return record_


class Want:
    """One requirement: a property, how it is compared, a value, or a pointer to another listing's page."""

    def __init__(self, facet: str, relation: str, amount_: str, pointer_of: dict | None = None, source: str = "request"):
        self.facet = str(facet or "").strip()
        self.relation = str(relation or "is").strip().lower()
        self.amount = str(amount_ or "").strip()
        self.pointer = pointer_of if isinstance(pointer_of, dict) else None
        self.source = source
        self.search = ""

    def as_line(self) -> str:
        if self.relation in POINTER_OPS and self.pointer:
            return f"{self.facet}: {self.relation} (not yet resolved)"
        return f"{self.facet}: {self.relation} {self.amount}"

    def slot(self) -> str:
        return _plain(self.facet)


class Goal:
    """One product the shopper asked for: its requirements, the listings seen for it and how they were judged."""

    def __init__(self, role_spec: dict[str, Any], position: int):
        self.slot_no = position
        self.label = str(role_spec.get("name") or role_spec.get("category") or "the product").strip()
        self.shelf = str(role_spec.get("category") or self.label).strip()
        units = role_spec.get("quantity")
        self.units = int(units) if isinstance(units, (int, float)) and not isinstance(units, bool) and units >= 1 else 1
        self.wants: list[Want] = [Want(want.get("property"), want.get("op"), want.get("value"), want.get("reference"))
                                  for want in role_spec.get("requirements") or [] if isinstance(want, dict) and want.get("property")]
        self.story = [str(head) for head in (role_spec.get("hints") or []) if head]
        self.queries = [str(term_) for term_ in (role_spec.get("search_terms") or []) if term_] or [self.shelf]
        self.head_noun = self.queries[-1]
        self._blank_state()

    def live_wants(self) -> list[Want]:
        return [want for want in self.wants if not (want.pointer and want.relation in POINTER_OPS)]

    def want_lines(self) -> list[str]:
        return [want.as_line() for want in self.live_wants()]

    def exact_hits(self, offer: Offer) -> int:
        if not offer.table:
            return 0
        pieces: set[frozenset[str]] = set()
        for val in offer.table.values():
            for entry_ in re.split(r"[,;/|]", str(val)):
                term_ = frozenset(_keywords(entry_))
                if term_:
                    pieces.add(term_)
        points = 0
        for want in self.live_wants():
            if want.relation not in ("is", "one of") or not want.amount:
                continue
            want_sets = [frozenset(_keywords(word)) for word in want.amount.split("|") if _keywords(word)]
            if any(word in pieces for word in want_sets):
                points += 1
        return points

    def good_offers(self, with_unsure: bool = False, lowest: bool = False, tabled_only: bool = False) -> list[Offer]:
        certain, open_set = [], []
        for oid, offer in self.offers.items():
            if not offer.is_open:
                continue
            val = self.rulings.get(oid)
            if val is None or val["failed"]:
                continue
            if tabled_only and val.get("weak"):
                continue
            if self.judged_against.get(oid, set()) >= set(self.want_lines()):
                (open_set if val["unsure"] else certain).append(offer)
        slot = lambda o: (o.unit_price if o.unit_price is not None else 1e18)
        if lowest:
            # (under a cheapest rule every judged fit is a fit: price decides, exact wording only breaks ties)
            certain.sort(key=lambda o: (slot(o), -self.exact_hits(o)))
        else:
            certain.sort(key=lambda o: (-self.exact_hits(o), len(self.rulings[o.oid].get("weak") or ()), slot(o)))
        open_set.sort(key=lambda o: (len(self.rulings[o.oid]["unsure"]), slot(o)))
        if certain or not with_unsure:
            return certain
        return open_set

    def _blank_state(self) -> None:
        self.offers: dict[str, Offer] = {}
        self.rulings: dict[str, dict[str, set[str]]] = {}
        self.judged_against: dict[str, set[str]] = {}
        self.asked_names: list[str] = []
        self.free_names: list[str] = []
        self.searches_run = 0
        self.dry = False
        self.pick: Offer | None = None
        self.shelf_counts: dict[str, int] = {}
        self.home_shelves: set[str] | None = None
        self.places: dict[str, dict[str, Any]] = {}


def _first_typed(fields: dict[str, Any], kind: str) -> str | None:
    return next((key_ for key_, val in fields.items() if isinstance(val, dict) and val.get("type") == kind), None)


def _is_count(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _enum_required(shape: dict[str, Any], skip: Any = ()) -> list[str]:
    fields_of = shape.get("properties") or {}
    return [key_ for key_ in shape.get("required") or []
            if key_ not in skip and isinstance((fields_of.get(key_) or {}).get("enum"), list)]


def _slot_name(key_: Any) -> str:
    slot = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(key_))
    return re.sub(r"\s+", " ", slot.replace("_", " ").replace(".", " ")).strip().lower()


class Ledger:
    """Steps, the store's wire, and how its records map onto offers."""

    def __init__(self, hub: "Venture") -> None:
        self.hub = hub

    def seconds_left(self) -> float:
        return TIME_LIMIT_S - self.hub.meter.elapsed()

    def moves_left(self) -> int:
        return self.hub.step_cap - self.hub.steps_used

    def jot(self, sort: str, **filters: Any) -> None:
        booked = {"kind": sort, "step": self.hub.steps_used, "t": round(self.hub.meter.elapsed(), 1)}
        booked.update(filters)
        self.hub.log_.append(booked)

    def keep_store_memory(self) -> None:
        _STORE_MEMORY[self.hub.store_key] = {"maps": self.hub.key_maps, "compare_limit": self.hub.side_limit,
                                         "thin_compare": self.hub.side_lacks_table}

    def out_of_money(self) -> bool:
        return self.hub.oracle.spent_usd >= SPEND_LIMIT_USD

    def reads_many(self) -> bool:
        return bool(self.hub.duties.get("compare")) and not getattr(self.hub, "side_off", False) and not self.hub.side_lacks_table

    def act(self, tool_name: str, argvals: dict[str, Any]) -> dict[str, Any]:
        result_: dict[str, Any] = {"obs": None, "lines": [], "error": self._blocked(tool_name), "ok": False}
        if result_["error"]:
            return result_
        hub = self.hub
        envelope: dict[str, Any] = dict(hub.envelope_base)
        envelope["step"] = hub.steps_used + 1
        envelope["action"] = {"name": tool_name, "args": argvals}
        for _ in range(3):
            remaining = self.seconds_left()
            if remaining < 3:
                result_["error"] = "no time"
                return result_
            _store_line.timeout = int(max(4, min(90, remaining - 1)))
            response = _store_line.post_verbose("/environment/call", envelope)
            store_fault = response.error if isinstance(response.error, dict) else {}
            hub.meter.charge(float(_store_line.timeout) if response.data is None and store_fault.get("status") is None else _STORE_CALL)
            payload_in = response.data
            if isinstance(payload_in, dict) and ("observation" in payload_in or "finished" in payload_in):
                self._booked(tool_name, argvals, payload_in, result_)
                return result_
            verdict = self._refused(response, envelope)
            if verdict != "retry":
                result_["error"] = verdict
                return result_
        result_["error"] = "env failure"
        return result_

    def _booked(self, tool_name: str, argvals: dict[str, Any], payload_in: dict[str, Any], result_: dict[str, Any]) -> None:
        """The store answered: its step count, the shopper lines riding on the result, the episode's history."""
        hub = self.hub
        hub.steps_used = payload_in["step"] if isinstance(payload_in.get("step"), int) else hub.steps_used + 1
        if payload_in.get("finished"):
            hub.over = True
        seen, said_lines = _peel(payload_in.get("observation"))
        for line_key in said_lines:
            if line_key not in hub.read_lines:
                hub.unread_lines.append(line_key)
        result_["obs"] = seen
        result_["lines"] = said_lines
        if isinstance(seen, dict) and isinstance(seen.get("error"), str) and len(seen) <= 2:
            result_["error"] = seen["error"]
        else:
            result_["ok"] = True
        self.jot("call", tool=tool_name, args=_short_json(argvals, 160), result=_short_json(seen, 200), lines=said_lines[:3])
        to_shopper = tool_name == (hub.duties.get("message") or {}).get("name")
        booked = {"step": hub.steps_used, "tool": tool_name, "args": _short_json(argvals, 300 if to_shopper else 120),
                  "result": _short_json(seen, 160)}
        if said_lines:
            booked["shopper_lines"] = [o[:200] for o in said_lines[:3]]
        if to_shopper:
            booked["basket_then"] = [list(elem) for elem in hub.liaison.basket_key()]
        hub.past.append(booked)

    def _refused(self, response: Any, envelope: dict[str, Any]) -> str:
        """A call the store did not answer: "retry" (a step number to resend with, or a passing fault), else the
        error the call ends with."""
        hub = self.hub
        failure = response.error or {}
        code_ = failure.get("status") if isinstance(failure, dict) else None
        body = str(failure.get("body") if isinstance(failure, dict) else failure)[:300]
        self.jot("env_failure", status=code_, body=body)
        if code_ == 402:
            hub.broke = True
            return "payment"
        if code_ == 409:
            if "finished" in body:
                hub.over = True
                return "finished"
            expected = re.search(r"expected (\d+)", body)
            if not expected:
                return "conflict"
            hub.steps_used = int(expected.group(1)) - 1
            envelope["step"] = hub.steps_used + 1
            return "retry"
        if code_ in (403, 422):
            return f"refused {code_}"
        return "retry"

    def key_map(self, sort: str, record_in: dict[str, Any]) -> dict[str, str]:
        slot = sort + ":" + ",".join(sorted(record_in.keys()))
        if slot in self.hub.key_maps:
            return self.hub.key_maps[slot]
        kin = self._kin_map(sort, slot, record_in)
        if kin is not None:
            return kin
        verdict_json = self.hub.oracle.ask_json([QUICK_SEAT, SPARE_SEAT], KEYS_PROMPT,
                                                json.dumps(self._record_preview(record_in), ensure_ascii=False), 700)
        response: dict[str, str] = {}
        if isinstance(verdict_json, dict):
            response = {duty: key_ for duty, key_ in verdict_json.items() if isinstance(key_, str) and key_ in record_in}
        self._guess_keys(response, record_in)
        self.hub.key_maps[slot] = response
        self.keep_store_memory()
        return response

    def offer_from(self, record_: dict[str, Any], sort: str) -> Offer | None:
        mapped = self.key_map(sort, record_)
        id_key = mapped.get("id")
        if not id_key or not isinstance(record_.get(id_key), (str, int)):
            return None
        offer = Offer(str(record_[id_key]))
        offer.raw_record = record_

        def cell(role: str, kind: type) -> Any:
            return record_[mapped[role]] if mapped.get(role) and isinstance(record_.get(mapped[role]), kind) else None

        for role, attr in (("title", "heading"), ("text", "blurb"), ("category", "shelf"), ("brand", "make"),
                           ("seller", "vendor"), ("notes", "remarks")):
            if cell(role, str) is not None:
                setattr(offer, attr, cell(role, str))
        if mapped.get("price"):
            offer.unit_price = _amount(record_.get(mapped["price"]))
        if cell("stock", bool) is not None:
            offer.can_buy = cell("stock", bool)
        if cell("facts", dict) is not None:
            offer.table = {str(key_): val for key_, val in cell("facts", dict).items() if val not in (None, "")}
            self.hub.__dict__.setdefault("seen_tables", {}).setdefault(offer.shelf or "", {})[offer.oid] = offer.table
        if not offer.remarks and not mapped.get("notes"):
            # (a notes field the key map could not name, empty on the record it was drawn from: one unmapped prose field)
            spare = [val for key_, val in record_.items()
                     if key_ not in mapped.values() and isinstance(val, str) and len(val) >= 40 and " " in val]
            if len(spare) == 1:
                offer.remarks = spare[0]
        return offer

    def fold_in(self, goal: Goal, offer: Offer, is_open: bool) -> Offer:
        seen_offer = goal.offers.get(offer.oid)
        if seen_offer is None:
            seen_offer = goal.offers[offer.oid] = offer
        else:
            self._merge_offer(goal, seen_offer, offer, is_open)
        if is_open:
            seen_offer.is_open = True
            self.hub.pages_open.add(seen_offer.oid)
        if seen_offer.shelf:
            goal.shelf_counts[seen_offer.shelf] = goal.shelf_counts.get(seen_offer.shelf, 0) + 1
        return seen_offer

    def _blocked(self, tool_name: str) -> str | None:
        hub = self.hub
        if hub.over or hub.broke or tool_name not in hub.shapes:
            return "unavailable"
        is_ending = tool_name in {(hub.duties.get(want) or {}).get("name") for want in ("checkout", "decline")}
        if self.moves_left() <= 0 or (self.moves_left() <= 1 and not is_ending):
            return "no steps left"
        return None

    def _kin_map(self, sort: str, slot: str, record_in: dict[str, Any]) -> dict[str, str] | None:
        for seen_slot, seen_offer in self.hub.key_maps.items():
            if not seen_slot.startswith(sort + ":") or not seen_offer:
                continue
            if sum(1 for key_ in seen_offer.values() if key_ in record_in) >= max(2, int(0.6 * len(seen_offer))):
                self.hub.key_maps[slot] = seen_offer
                return seen_offer
        return None

    @staticmethod
    def _record_preview(record_in: dict[str, Any]) -> dict[str, Any]:
        shown_now = {}
        for key_, val in record_in.items():
            val = val if not isinstance(val, (dict, list)) else _short_json(val, 160)
            shown_now[key_] = val[:160] + "…" if isinstance(val, str) and len(val) > 160 else val
        return shown_now

    @staticmethod
    def _guess_keys(response: dict[str, str], record_in: dict[str, Any]) -> None:
        def flat_dict(val: Any) -> bool:
            return isinstance(val, dict) and bool(val) and all(isinstance(elem, (str, int, float)) or elem is None for elem in val.values())

        guesses = (
            ("facts", lambda key_, val: flat_dict(val)),
            ("id", lambda key_, val: isinstance(val, str) and re.fullmatch(r"[A-Za-z]{0,3}-?[0-9a-f]{8,}", val)),
            ("price", lambda key_, val: isinstance(val, (int, float)) and not isinstance(val, bool) and "price" in key_.lower()),
        )
        for duty, fits in guesses:
            if duty not in response:
                found = next((key_ for key_, val in record_in.items() if fits(key_, val)), None)
                if found is not None:
                    response[duty] = found

    @staticmethod
    def _merge_offer(goal: Goal, seen_offer: Offer, offer: Offer, is_open: bool) -> None:
        for field_name in ("heading", "blurb", "shelf", "make", "vendor", "remarks"):
            if getattr(offer, field_name) and (is_open or not getattr(seen_offer, field_name)):
                setattr(seen_offer, field_name, getattr(offer, field_name))
        if offer.unit_price is not None:
            seen_offer.unit_price = offer.unit_price
        if offer.can_buy is not None:
            seen_offer.can_buy = offer.can_buy
        if offer.table:
            if offer.table != seen_offer.table:
                goal.judged_against.pop(seen_offer.oid, None)
            seen_offer.table = offer.table
        if offer.remarks and offer.remarks != seen_offer.remarks and is_open:
            goal.judged_against.pop(seen_offer.oid, None)
        seen_offer.page_seen = seen_offer.page_seen or offer.page_seen
        seen_offer.raw_record = offer.raw_record or seen_offer.raw_record


class Brief:
    """The request read once, and arguments fitted to the tools' schemas."""

    def __init__(self, hub: "Venture") -> None:
        self.hub = hub

    def read_request(self) -> bool:
        intake = self._intake_reply(self._intake_message())
        if not isinstance(intake, dict):
            self.hub.ledger.jot("setup_failed")
            return False
        self.hub.intake = intake
        self._seat_duties(intake.get("tools"))
        self.hub.goals = self._goals_from(intake.get("items"))
        for goal in self.hub.goals:
            self._settle_goal(goal)
        self._take_limits(intake)
        self._take_procedure(intake.get("procedure"))
        self.hub.ledger.jot("plan", items=[(pos_.label, pos_.units, pos_.want_lines(), pos_.story) for pos_ in self.hub.goals], cap=self.hub.ceiling,
                  cheapest=self.hub.lowest, messages=self.hub.message_cap, steps=self.hub.step_cap,
                  roles={key_: val.get("name") for key_, val in self.hub.duties.items()})
        return True

    def fit_duties(self) -> None:
        for duty in ("cart_add", "cart_set"):
            role_spec = self.hub.duties.get(duty)
            if role_spec:
                self._fit_cart_duty(duty, role_spec)
        self._mirror_ending("checkout", "decline")
        self._mirror_ending("decline", "checkout")

    def tool_text(self, tool_name: str) -> str:
        for fun in self._functions():
            if isinstance(fun, dict) and fun.get("name") == tool_name:
                return str(fun.get("description") or "")
        return ""

    def ending_argvals(self, duty: str) -> dict[str, Any]:
        role_spec = self.hub.duties.get(duty) or {}
        shape = self._shape_of(duty)
        fields_of = shape.get("properties") or {}
        given = role_spec.get("args") if isinstance(role_spec.get("args"), dict) else {}
        result_: dict[str, Any] = {}
        for key_, val in given.items():
            if key_ in fields_of and val not in (None, ""):
                usable, val = self._enum_fit(fields_of[key_].get("enum"), val)
                if usable:
                    result_[key_] = val
        lacking = _enum_required(shape, result_)
        if lacking:
            remembered = self._ending_choice(duty, role_spec, {key_: fields_of[key_] for key_ in lacking})
            for key_ in lacking:
                if remembered.get(key_) in fields_of[key_]["enum"]:
                    result_[key_] = remembered.get(key_)
        return result_

    def _shape_of(self, duty: str) -> dict[str, Any]:
        role_spec = self.hub.duties.get(duty) or {}
        return self.hub.shapes.get(role_spec.get("name") or "", {})

    def _type_of(self, duty: str, argname: str | None) -> str | None:
        if not argname:
            return None
        fields_of = self._shape_of(duty).get("properties") or {}
        return (fields_of.get(argname) or {}).get("type")

    def _paging_of(self, duty: str) -> str | None:
        role_spec = self.hub.duties.get(duty) or {}
        page_no = role_spec.get("page")
        if not page_no:
            return None
        if self._type_of(duty, page_no) == "integer":
            sort = str(role_spec.get("page_kind") or "").lower()
            tool_name = page_no.lower()
            if sort == "offset" or (sort != "number" and ("offset" in tool_name or "start" in tool_name or "skip" in tool_name)):
                return "offset"
            return "index"
        return "token"

    def _listed_spelling(self, duty: str, argname: str, amount_: Any) -> Any:
        fields_of = self._shape_of(duty).get("properties") or {}
        allowed = (fields_of.get(argname) or {}).get("enum")
        if not isinstance(allowed, list) or not allowed or not isinstance(amount_, str):
            return amount_
        by_plain = {_plain(entry_): entry_ for entry_ in allowed if isinstance(entry_, str)}
        if not by_plain:
            return amount_
        if _plain(amount_) in by_plain:
            return by_plain[_plain(amount_)]
        top_pick = max(by_plain, key=lambda entry_: _overlap(entry_, amount_))
        return by_plain[top_pick] if _overlap(top_pick, amount_) >= 0.5 else None

    def _query_as_sent(self, goal: Goal, duty: str, argname: str, words: str) -> str:
        kind_words = _keywords(goal.shelf) | _keywords(goal.label)
        seen_values: list[str] = []
        remaining_words: list[str] = []
        for word in words.split():
            (remaining_words if _keywords(word) and _keywords(word) <= kind_words else seen_values).append(word)
        words = " ".join(seen_values + remaining_words)
        cap_chars = ((self._shape_of(duty).get("properties") or {}).get(argname) or {}).get("maxLength")
        if not isinstance(cap_chars, int) or len(words) <= cap_chars:
            return words
        kept_words = list(seen_values)
        for word in remaining_words:
            if len(" ".join(kept_words + [word])) <= cap_chars:
                kept_words.append(word)
        return self._cut(duty, argname, " ".join(kept_words) if kept_words else words)

    def _cut(self, duty: str, argname: str, amount_: str) -> str:
        fields_of = self._shape_of(duty).get("properties") or {}
        cap_chars = (fields_of.get(argname) or {}).get("maxLength")
        if isinstance(cap_chars, int) and len(amount_) > cap_chars:
            amount_ = amount_[:cap_chars].rsplit(" ", 1)[0] if " " in amount_[:cap_chars] else amount_[:cap_chars]
        return amount_

    def _functions(self):
        for term_ in self.hub.tool_list:
            yield term_.get("function") if isinstance(term_.get("function"), dict) else term_

    def _intake_message(self) -> str:
        cards = [{"name": fun.get("name"), "description": fun.get("description", ""), "parameters": fun.get("parameters")}
                 for fun in self._functions()]
        parts = ["REQUEST:\n" + self.hub.ask_text_in]
        if self.hub.first_seen is not None:
            parts.append("OPENING OBSERVATION:\n" + _short_json(self.hub.first_seen, 1500))
        parts.append("TOOLS:\n" + json.dumps(cards, ensure_ascii=False))
        return "\n\n".join(parts)

    def _intake_reply(self, user_msg: str) -> Any:
        intake = self.hub.oracle.ask_json([READER_SEAT, SPARE_SEAT], INTAKE_PROMPT, user_msg, 2500)
        if isinstance(intake, dict) and isinstance(intake.get("tools"), dict):
            return intake
        return self.hub.oracle.ask_json([READER_SEAT, QUICK_SEAT], INTAKE_PROMPT, user_msg, 2500)

    def _seat_duties(self, named: Any) -> None:
        offered = named if isinstance(named, dict) else {}
        self.hub.duties = {duty: role_spec for duty, role_spec in offered.items()
                           if isinstance(role_spec, dict) and isinstance(role_spec.get("name"), str) and role_spec["name"] in self.hub.shapes}
        self.fit_duties()
        for duty in ("search_text", "open", "checkout", "decline", "message"):
            if duty not in self.hub.duties:
                self.hub.ledger.jot("role_missing", role=duty)

    def _goals_from(self, listed: Any) -> list[Goal]:
        specs = listed if isinstance(listed, list) else []
        goals = [Goal(spec_, pos_) for pos_, spec_ in enumerate(specs) if isinstance(spec_, dict)]
        return goals or [Goal({"name": "the product", "category": "", "search_terms": [self.hub.ask_text_in[:60]]}, 0)]

    @staticmethod
    def _pointer_values(goal: Goal) -> set[tuple[str, str]]:
        borrowed: set[tuple[str, str]] = set()
        for want in goal.wants:
            if want.pointer and isinstance(want.pointer.get("attributes"), dict):
                borrowed.update((_plain(key_), _plain(val)) for key_, val in want.pointer["attributes"].items())
        return borrowed

    @staticmethod
    def _spellings(amount_: str):
        for val in amount_.split("|"):
            val = val.strip()
            if val:
                yield val
                if "-" in val:
                    yield val.replace("-", " ")

    def _settle_goal(self, goal: Goal) -> None:
        borrowed = self._pointer_values(goal)
        if borrowed:
            goal.wants = [want for want in goal.wants if want.pointer or (_plain(want.facet), _plain(want.amount)) not in borrowed]
            goal.queries = [term_ for term_ in goal.queries if not any(val and val in _plain(term_) for _, val in borrowed)] or [goal.shelf]
        shelf_words = _keywords(goal.shelf)

        def names_shelf(want: Want) -> bool:
            words = _keywords(want.amount)
            return bool(words) and words <= shelf_words

        goal.wants = [want for want in goal.wants if want.pointer or want.relation != "is" or not names_shelf(want)]
        for want in goal.wants:
            if want.relation not in ("is", "one of", "contains") or not want.amount:
                continue
            for spelling in self._spellings(want.amount):
                if len(spelling) <= 40 and not any(_plain(spelling) == _plain(term_) for term_ in goal.queries):
                    goal.queries.append(spelling)

    def _take_limits(self, intake: dict[str, Any]) -> None:
        self.hub.ceiling = _amount(intake.get("total_cap"))
        self.hub.lowest = bool(intake.get("cheapest"))
        best = intake.get("best")
        self.hub.measure = best if isinstance(best, str) and best else None
        messages_said = intake.get("message_budget")
        self.hub.message_cap = int(messages_said) if _is_count(messages_said) and messages_said >= 0 else None
        steps_said = intake.get("call_budget")
        if _is_count(steps_said) and 0 < steps_said < self.hub.step_cap:
            self.hub.step_cap = int(steps_said)
        self.hub.stock_only = bool(intake.get("in_stock_only", True))
        self.hub.raise_topics = [str(first_) for first_ in (intake.get("ask_about") or []) if first_]
        self.hub.fit_list = [want for want in (intake.get("fit_rules") or []) if isinstance(want, dict)]
        self.hub.rules_of = self.hub.coupler.take_rules(intake.get("conditions"))

    def _take_procedure(self, listed: Any) -> None:
        process = [step_spec for step_spec in (listed or []) if isinstance(step_spec, dict) and step_spec.get("goal")][:6]
        for step_spec in process:
            step_spec["moment"] = str(step_spec.get("moment") or "before_finish").lower()
            step_spec["kind"] = "message" if str(step_spec.get("kind") or "").lower() == "message" else "tool"
        self.hub.process = process
        self.hub.show_basket_first = any(step_spec["kind"] == "message" and step_spec["moment"] in ("before_order", "before_finish")
                                         for step_spec in process)

    def _fit_cart_duty(self, duty: str, role_spec: dict[str, Any]) -> None:
        fields_of = (self.hub.shapes.get(role_spec["name"]) or {}).get("properties") or {}
        line_key = next((key_ for key_, val in fields_of.items() if isinstance(val, dict) and val.get("type") == "array"
                         and isinstance(val.get("items"), dict) and val["items"].get("type") == "object"), None)
        if line_key is not None:
            self._fit_line_cart(duty, role_spec, line_key, fields_of[line_key]["items"].get("properties") or {})
        elif duty == "cart_add":
            self._fit_flat_add(role_spec, fields_of)

    def _fit_line_cart(self, duty: str, role_spec: dict[str, Any], line_key: str, line_fields: dict[str, Any]) -> None:
        def field(named: str, kind: str) -> str | None:
            if duty == "cart_set" and role_spec.get(named) in line_fields:
                return role_spec.get(named)
            return _first_typed(line_fields, kind)

        id_key = field("id", "string")
        qty_key = field("qty", "integer")
        if id_key:
            self.hub.duties["cart_set"] = {"name": role_spec["name"], "lines": line_key, "id": id_key, "qty": qty_key}
        if duty == "cart_add":
            self.hub.duties.pop("cart_add", None)

    def _fit_flat_add(self, role_spec: dict[str, Any], fields_of: dict[str, Any]) -> None:
        if role_spec.get("id") not in fields_of:
            role_spec["id"] = _first_typed(fields_of, "string")
        if role_spec.get("qty") and role_spec["qty"] not in fields_of:
            role_spec["qty"] = _first_typed(fields_of, "integer")
        if not role_spec.get("id"):
            self.hub.duties.pop("cart_add", None)

    def _mirror_ending(self, present_role: str, alt_offer: str) -> None:
        role_spec = self.hub.duties.get(present_role)
        if not role_spec or alt_offer in self.hub.duties:
            return
        if _enum_required(self.hub.shapes.get(role_spec["name"]) or {}):
            self.hub.duties[alt_offer] = {"name": role_spec["name"]}

    @staticmethod
    def _enum_fit(allowed: Any, val: Any) -> tuple[bool, Any]:
        if not isinstance(allowed, list) or not allowed or val in allowed:
            return True, val
        top_pick = max(allowed, key=lambda entry_: _overlap(str(entry_), str(val)))
        return _overlap(str(top_pick), str(val)) >= 0.5, top_pick

    def _ending_choice(self, duty: str, role_spec: dict[str, Any], lacking_fields: dict[str, Any]) -> dict[str, Any]:
        remembered = getattr(self.hub, "ending_args", {}).get(duty)
        if remembered is not None:
            return remembered
        aim = "place the order for what is in the cart" if duty == "checkout" else "end the task without buying anything"
        user_msg = (f"TOOL: {role_spec.get('name')}\nDESCRIPTION: {self.tool_text(role_spec.get('name') or '')}\n"
                    f"ARGUMENTS: {json.dumps(lacking_fields, ensure_ascii=False)}\nGOAL: {aim}")
        verdict_json = self.hub.oracle.ask_json([QUICK_SEAT, SPARE_SEAT], ENDING_ARG_PROMPT, user_msg, 300)
        remembered = verdict_json if isinstance(verdict_json, dict) else {}
        if remembered:
            self.hub.ending_args = dict(getattr(self.hub, "ending_args", {}), **{duty: remembered})
        return remembered


class _SearchWalk:
    """One search (a signature of query or filters) walked page by page: the arguments of a page, what a page
    showed, and where the walk stands for the goal."""

    def __init__(self, scout: "Scout", goal: Goal, duty: str, spec: dict[str, Any], words: str | None,
                 filters: dict[str, Any] | None) -> None:
        self.scout, self.hub, self.goal = scout, scout.hub, goal
        self.duty, self.spec, self.words, self.filters = duty, spec, words, filters
        self.paging = self.hub.brief._paging_of(duty)
        self.signature = duty + ":" + (words if words is not None else json.dumps(filters or {}, sort_keys=True))
        place = goal.places.get(self.signature)
        self.over = bool(place and place.get("done"))
        self.bookmark: Any = place.get("token") if place else None
        self.first = (place.get("page", 0) + 1) if place else 1
        self.added = 0
        self.finished = False

    def closed(self) -> bool:
        return self.over or (self.first > 1 and self.paging is None)

    def base_args(self) -> tuple[dict[str, Any], str]:
        """The page's fixed arguments, and "" or why the walk cannot run ("no_query", "unlisted")."""
        spec, brief = self.spec, self.hub.brief
        argvals: dict[str, Any] = {}
        if spec.get("mode") and isinstance(spec["mode"], dict) and spec["mode"].get("arg"):
            argvals[spec["mode"]["arg"]] = spec["mode"].get("value")
        if self.words is not None:
            if not spec.get("query"):
                return argvals, "no_query"
            argvals[spec["query"]] = brief._query_as_sent(self.goal, self.duty, spec["query"], self.words)
            return argvals, ""
        for slot, amount_ in (self.filters or {}).items():
            argname = spec.get(slot)
            if not argname or amount_ in (None, ""):
                continue
            amount_ = brief._listed_spelling(self.duty, argname, amount_)
            if amount_ is None:
                return argvals, "unlisted"
            argvals[argname] = brief._cut(self.duty, argname, amount_) if isinstance(amount_, str) else amount_
        return argvals, ""

    def turn_page(self, argvals: dict[str, Any], page_no: int) -> bool:
        """The paging argument of a later page; False when the store pages no further."""
        spec = self.spec
        if self.paging == "index":
            argvals[spec["page"]] = page_no
        elif self.paging == "offset":
            argvals[spec["page"]] = (page_no - 1) * max(self.hub.rows_per_page, 1)
        elif self.paging == "token" and self.bookmark:
            argvals[spec["page"]] = self.bookmark
        else:
            return False
        highest = ((self.hub.brief._shape_of(self.duty).get("properties") or {}).get(spec["page"]) or {}).get("maximum")
        if isinstance(argvals[spec["page"]], int) and isinstance(highest, (int, float)) and argvals[spec["page"]] > highest:
            self.goal.hidden_more = True
            return False
        return True

    def take(self, seen: Any) -> tuple[list[dict[str, Any]], Any]:
        """A page's rows folded into the goal; the page's next-page token kept; (rows, reported match count)."""
        hub, goal = self.hub, self.goal
        rows = _row_list(seen) or []
        self.bookmark = None
        total = None
        if isinstance(seen, dict):
            for val in seen.values():
                if isinstance(val, str) and 0 < len(val) <= 20 and " " not in val and not isinstance(val, bool):
                    self.bookmark = val
            figures = [val for val in seen.values() if isinstance(val, int) and not isinstance(val, bool)]
            total = max(figures) if figures else None
        if rows:
            hub.rows_per_page = max(hub.rows_per_page, len(rows))
        for record_ in rows:
            offer = hub.ledger.offer_from(record_, "row")
            if offer is None:
                continue
            if offer.oid not in goal.offers:
                self.added += 1
            hub.ledger.fold_in(goal, offer, is_open=False)
        goal.searches_run += 1
        if goal.home_shelves is None and goal.slot_no != 99:
            self.scout.settle_shelves(goal)
        return rows, total

    def ended(self, rows: list[dict[str, Any]], total: Any, page_no: int) -> bool:
        if not rows or (total is not None and page_no * max(len(rows), 1) >= total) or len(rows) < self.hub.rows_per_page:
            return True
        if self.paging == "token" and not self.bookmark:
            if total is not None and total > page_no * len(rows):
                self.goal.hidden_more = True
            return True
        return False

    def run(self, page_limit: int, keep_steps: int | None) -> int:
        hub = self.hub
        page_no = self.first - 1
        for page_no in range(self.first, self.first + page_limit):
            if hub.ledger.moves_left() <= (keep_steps if keep_steps is not None else 0) or hub.ledger.seconds_left() < CLOSING_S:
                page_no -= 1
                break
            argvals, stop = self.base_args()
            if stop == "no_query":
                return 0
            if stop == "unlisted":
                self.goal.places[self.signature] = {"page": 0, "token": None, "done": True}
                return self.added
            if page_no > 1 and not self.turn_page(argvals, page_no):
                self.finished = True
                page_no -= 1
                break
            outcome = hub.ledger.act(self.spec["name"], argvals)
            if not outcome["ok"]:
                hub.ledger.jot("search_error", error=outcome["error"])
                self.finished = True
                break
            rows, total = self.take(outcome["obs"])
            if self.ended(rows, total, page_no):
                self.finished = True
                break
        self.goal.places[self.signature] = {"page": max(page_no, 0), "token": self.bookmark,
                                            "done": self.finished or self.paging is None}
        return self.added


class _Opening:
    """Listings opened in full: through their property tables alone (a store whose side-by-side rows carry none),
    side by side in batches, or one page at a time."""

    def __init__(self, scout: "Scout", goal: Goal | None, idents: list[str], keep_free: int) -> None:
        self.scout, self.hub, self.goal, self.idents, self.keep_free = scout, scout.hub, goal, idents, keep_free
        hub = self.hub
        self.side = hub.duties.get("compare") if not getattr(hub, "side_off", False) else None
        self.table_tool = hub.duties.get("facts")
        self.page_tool = hub.duties.get("open")
        self.opened: list[Offer] = []
        self.at = 0

    def tight(self, cost: int = 1) -> bool:
        return bool(self.keep_free) and self.hub.ledger.moves_left() - cost < self.keep_free

    def run(self) -> list[Offer]:
        hub = self.hub
        if self.side and not getattr(hub, "side_checked", False):
            hub.side_checked = True
            most = ((hub.brief._shape_of("compare").get("properties") or {}).get(self.side.get("ids") or "") or {}).get("maxItems")
            if isinstance(most, int) and 2 <= most < hub.side_limit:
                hub.side_limit = most
        while self.at < len(self.idents) and hub.ledger.moves_left() > 0 and hub.ledger.seconds_left() > CLOSING_S:
            if hub.side_lacks_table and self.table_tool and self.goal is not None:
                if self.tight():
                    break
                self._by_table(self.idents[self.at])
                self.at += 1
                continue
            if self.side and len(self.idents) - self.at >= 2:
                if self.tight():
                    break
                if self._side_batch():
                    continue
            if not self.page_tool or self.tight(2 if self.table_tool else 1):
                break
            self._single(self.idents[self.at])
        return self.opened

    def _by_table(self, oid: str) -> None:
        got = self.scout.open_by_table(self.goal, oid)
        if got is not None:
            self.opened.append(got)

    def _side_batch(self) -> bool:
        """One side-by-side batch; True when the loop goes on (a batch read, or the tool's limit lowered or dropped)."""
        hub = self.hub
        batch = self.idents[self.at: self.at + hub.side_limit]
        outcome = hub.ledger.act(self.side["name"], {self.side["ids"]: batch})
        if outcome["ok"] and _row_list(outcome["obs"]):
            self._take_side_rows([o for o in (hub.ledger.offer_from(rec, "detail") for rec in _row_list(outcome["obs"]) or []) if o is not None])
            self.at += len(batch)
            return True
        if outcome["error"] or (outcome["ok"] and not _row_list(outcome["obs"])):
            self._narrow_side(len(batch))
            return True
        return False

    def _single(self, oid: str) -> None:
        self.at += 1
        self._read_page(oid)

    def _read_page(self, oid: str) -> None:
        hub = self.hub
        outcome = hub.ledger.act(self.page_tool["name"], {self.page_tool["id"]: oid})
        if not outcome["ok"] or not isinstance(outcome["obs"], dict):
            return
        record_ = outcome["obs"]
        offer = hub.ledger.offer_from(record_, "detail")
        if offer is None:
            offer = Offer(oid)
            offer.raw_record = record_
        offer.page_seen = True
        if not offer.table and self.table_tool and hub.ledger.moves_left() > 0:
            offer.table = self.scout.table_of(oid) or {}
        if offer.oid != oid:
            offer.oid = oid
        offer.is_open = True
        self.opened.append(hub.ledger.fold_in(self.goal, offer, True) if self.goal else offer)
        hub.pages_open.add(oid)

    def _take_side_rows(self, rows: list[Offer]) -> None:
        hub, goal = self.hub, self.goal
        thin = bool(self.table_tool) and bool(rows) and not any(o.table for o in rows)
        if thin and not hub.side_lacks_table:
            hub.side_lacks_table = True
            hub.ledger.keep_store_memory()
            hub.ledger.jot("thin_compare", tool=self.side["name"], facts_tool=self.table_tool.get("name"))
        by_table = thin and goal is not None
        for offer in rows:
            offer.page_seen = True
            hub.pages_open.add(offer.oid)
            if by_table:
                hub.ledger.fold_in(goal, offer, False)
                continue
            offer.is_open = True
            self.opened.append(hub.ledger.fold_in(goal, offer, True) if goal else offer)
        if not by_table:
            return
        for offer in rows:
            if self.tight() or hub.ledger.seconds_left() < CLOSING_S:
                break
            self._by_table(offer.oid)

    def _narrow_side(self, sent: int) -> None:
        hub = self.hub
        if hub.side_limit > 2 and sent > 2:
            hub.side_limit = 4 if hub.side_limit > 4 else 2
            hub.ledger.keep_store_memory()
            return
        hub.side_off = True
        self.side = None


class Scout:
    """Searches, row screening and opening listings."""

    def __init__(self, hub: "Venture") -> None:
        self.hub = hub

    def look_up(self, goal: Goal, words: str | None = None, filters: dict[str, Any] | None = None,
               page_limit: int = 1, keep_steps: int | None = None) -> int:
        duty = "search_text" if words is not None else "search_fields"
        spec = self.hub.duties.get(duty)
        if not spec:
            return 0
        walk = _SearchWalk(self, goal, duty, spec, words, filters)
        if walk.closed():
            return 0
        return walk.run(page_limit, keep_steps)

    def on_shelf(self, goal: Goal, offer: Offer) -> bool:
        if not offer.shelf or not goal.shelf:
            return True
        label_plain = _plain(offer.shelf)
        if goal.home_shelves:
            return label_plain in goal.home_shelves
        wanted_n = _plain(goal.shelf)
        if label_plain == wanted_n or label_plain in wanted_n or wanted_n in label_plain:
            return True
        return max(_overlap(label_plain, wanted_n), _shelf_overlap(label_plain, wanted_n)) >= 0.5

    def settle_shelves(self, goal: Goal) -> None:
        shelf_labels = sorted({o.shelf for o in goal.offers.values() if o.shelf})
        if not shelf_labels or goal.home_shelves is not None:
            return
        label_by_plain = {_plain(o): o for o in shelf_labels}
        if len(shelf_labels) == 1:
            goal.home_shelves = set(label_by_plain) if (max(_overlap(shelf_labels[0], goal.shelf), _shelf_overlap(shelf_labels[0], goal.shelf)) >= 0.3 or _plain(goal.shelf) in _plain(shelf_labels[0])
                                              or _plain(shelf_labels[0]) in _plain(goal.shelf)) else set()
            if not goal.home_shelves:
                goal.home_shelves = None
            return
        user_msg = (f"REQUESTED PRODUCT: {goal.label} (category wording: {goal.shelf})\nREQUEST EXCERPT: {self.hub.ask_text_in[:600]}\n"
                f"STORE CATEGORY LABELS SEEN: {json.dumps(shelf_labels, ensure_ascii=False)}")
        verdict_json = self.hub.oracle.ask_json([QUICK_SEAT, SPARE_SEAT], SHELF_PROMPT, user_msg, 500)
        picked: set[str] = set()
        if isinstance(verdict_json, dict) and isinstance(verdict_json.get("labels"), list):
            for o in verdict_json["labels"]:
                if isinstance(o, str) and _plain(o) in label_by_plain:
                    picked.add(_plain(o))
        if picked:
            goal.home_shelves = picked
            self.hub.ledger.jot("labels", item=goal.label, labels=sorted(picked))

    def store_label(self, goal: Goal) -> str:
        """The category label as this store writes it on the goal's listings (the commonest one), else the request's words."""
        counts: dict[str, int] = {}
        for o in goal.offers.values():
            if o.shelf and self.on_shelf(goal, o):
                counts[o.shelf] = counts.get(o.shelf, 0) + 1
        return max(counts, key=lambda label: (counts[label], label)) if counts else goal.shelf

    def contenders(self, goal: Goal) -> list[Offer]:
        ceiling = self.hub.ceiling
        result_ = []
        for offer in goal.offers.values():
            if offer.can_buy is False and self.hub.stock_only:
                continue
            if ceiling is not None and offer.unit_price is not None and offer.unit_price * goal.units > ceiling + 1e-6:
                continue
            if not self.on_shelf(goal, offer):
                continue
            result_.append(offer)
        result_.sort(key=lambda o: (0 if o.row_verdict == "yes" else (2 if o.row_verdict == "no" else 1),
                                o.unit_price if o.unit_price is not None else 1e18))
        return result_

    def screen_rows(self, goal: Goal, offers: list[Offer]) -> None:
        pending = [o for o in offers if not o.is_open and o.row_verdict == "maybe" and not getattr(o, "row_checked", False)]
        if len(pending) < 8 or not goal.want_lines():
            return
        row_list_ = []
        group = pending[:30]
        for number_, o in enumerate(group, 1):
            row_list_.append({"n": number_, "title": o.heading[:140], "text": (o.blurb or "")[:ROW_TEXT_CHARS], "category": o.shelf, "price": o.unit_price})
        user_msg = (f"REQUESTED PRODUCT: {goal.shelf}\nREQUIREMENTS:\n- " + "\n- ".join(goal.want_lines())
                + "\n\nROWS:\n" + json.dumps(row_list_, ensure_ascii=False))
        verdict_json = self.hub.oracle.ask_json([QUICK_SEAT, SPARE_SEAT], ROWS_PROMPT, user_msg, 1200)
        for o in group:
            o.row_checked = True
        if isinstance(verdict_json, dict):
            for label_kind in ("no", "yes"):
                for number_ in verdict_json.get(label_kind) or []:
                    if isinstance(number_, (int, str)) and str(number_).strip().isdigit() and 1 <= int(number_) <= len(group):
                        group[int(number_) - 1].row_verdict = label_kind

    def open_offers(self, goal: Goal | None, idents: list[str], keep_free: int = 0) -> list[Offer]:
        idents = [pos_ for pos_ in idents if pos_]
        if not idents:
            return []
        return _Opening(self, goal, idents, keep_free).run()

    def table_of(self, oid: str) -> dict[str, Any] | None:
        tool_spec = self.hub.duties.get("facts")
        if not tool_spec:
            return None
        outcome = self.hub.ledger.act(tool_spec["name"], {tool_spec["id"]: oid})
        if not outcome["ok"] or not isinstance(outcome["obs"], dict):
            return None
        mapped = self.hub.ledger.key_map("facts", outcome["obs"])
        table_key = mapped.get("facts")
        if table_key and isinstance(outcome["obs"].get(table_key), dict):
            return {str(key_): val for key_, val in outcome["obs"][table_key].items() if val not in (None, "")}
        if all(isinstance(val, (str, int, float)) or val is None for val in outcome["obs"].values()):
            return {str(key_): val for key_, val in outcome["obs"].items() if val not in (None, "")}
        return {}

    def open_by_table(self, goal: Goal, oid: str) -> Offer | None:
        seen_offer = goal.offers.get(oid)
        table = self.table_of(oid)
        if table is None or (seen_offer is None and not table):
            return None
        offer = Offer(oid)
        offer.table = table
        return self.hub.ledger.fold_in(goal, offer, True)


class Bench:
    """Listings judged against the wants; fits ordered by the request's rule."""

    def __init__(self, hub: "Venture") -> None:
        self.hub = hub

    def drop_shelf_echoes(self, goal: Goal) -> None:
        if not goal.home_shelves:
            return
        shelf_words = set().union(*(_keywords(o) for o in goal.home_shelves))
        read_out = [o for o in goal.offers.values() if o.is_open and o.table and self.hub.scout.on_shelf(goal, o)]
        if len(read_out) < 3:
            return
        kept_wants = []
        for want in goal.wants:
            word_set = _keywords(want.amount)
            if (want.source == "request" and want.relation == "is" and word_set and word_set <= shelf_words
                    and not any(_same_name(key_, want.facet) or _overlap(str(key_).replace("_", " "), want.facet) >= 0.5 for o in read_out for key_ in o.table)):
                self.hub.ledger.jot("label_restatement", item=goal.label, req=want.as_line())
                continue
            kept_wants.append(want)
        goal.wants = kept_wants

    def assess(self, goal: Goal, offers: list[Offer]) -> None:
        self.drop_shelf_echoes(goal)
        want_lines = goal.want_lines()
        if not want_lines:
            self._open_rulings(goal, offers)
            return
        pending = [o for o in offers if o.is_open and not (goal.judged_against.get(o.oid, set()) >= set(want_lines))]
        if not pending or self.hub.ledger.seconds_left() < CLOSING_S:
            return
        field_rows = self.store_fields(goal)
        filters = ("\n\nSTORE FIELDS (from this store's records read so far):\n" + self.field_text(field_rows)) if field_rows else ""
        groups = [pending[begin:begin + 4] for begin in range(0, len(pending), 4)]
        user_texts = [self._assess_message(goal, group, want_lines, filters) for group in groups]
        verdict_jsons = _side_by_side([lambda user_text=user_text: self.hub.oracle.ask_json([QUICK_SEAT, SPARE_SEAT], ASSESS_PROMPT, user_text, 2000)
                                       for user_text in user_texts], self.hub.meter)
        for group, verdict_json in zip(groups, verdict_jsons):
            by_id = self._verdicts_by_id(goal, verdict_json)
            for o in group:
                if o.oid in by_id:
                    self._take_verdict(goal, o, by_id[o.oid], want_lines, field_rows)

    @staticmethod
    def not_written(goal: Goal, offer: Offer, want_: Want) -> bool:
        if want_.relation not in ("is", "one of") or want_.source == "fit" or not want_.amount:
            return False
        name_words_all = _all_words(goal.label) | _all_words(goal.shelf)
        variants = [{elem for elem in _all_words(val) - name_words_all if not (elem.isalpha() and len(elem) <= 3)}
                    for val in want_.amount.split("|") if _all_words(val)]
        if not variants or not all(variants):
            # (an alternative made only of short or product-name words cannot be told apart in the text: no verdict)
            return False
        phrase = _all_words(" ".join([offer.heading or "", offer.blurb or "", offer.remarks or "", offer.shelf or "",
                                " ".join(f"{key_} {val}" for key_, val in offer.table.items())]))
        return not any(first_ & phrase for first_ in variants)

    @staticmethod
    def table_writes(o: Offer, want: Want) -> bool:
        """The listing's own property for the want writes one of its values word for word, and no key carrying the bare
        property name says otherwise."""
        if want.relation not in ("is", "one of", "contains") or not want.amount or re.search(r"\d", want.amount):
            return False
        want_sets = [_keywords(word) for word in want.amount.split("|") if _keywords(word)]
        if not want_sets:
            return False

        def writes(key_: str) -> bool:
            pieces = [_keywords(entry_) for entry_ in re.split(r"[,;/|]", str(o.table[key_])) if _keywords(entry_)]
            return any(word <= piece for word in want_sets for piece in pieces)

        named = [key_ for key_ in o.table if _names_facet(key_, want.facet)]
        bare = [key_ for key_ in named if _same_name(key_, want.facet)]
        return any(writes(key_) for key_ in named) and all(writes(key_) for key_ in bare)

    def store_fields(self, goal: Goal) -> list[dict[str, Any]]:
        read_out = [o for o in goal.offers.values() if o.is_open and o.table and self.hub.scout.on_shelf(goal, o)]
        if len(read_out) < 3:
            return []
        rows = (self._field_row(want, read_out) for want in goal.live_wants())
        return [row for row in rows if row]

    def field_misses(self, goal: Goal, offer: Offer, field_rows: list[dict[str, Any]] | None = None) -> set[str]:
        result_ = set()
        for row_ in (self.store_fields(goal) if field_rows is None else field_rows):
            own_values = [key_ for key_ in offer.table if _names_facet(key_, row_["prop"])]
            if not own_values or not (row_["writers"] - {offer.oid}):
                continue
            own_pieces = [frozenset(_keywords(entry_)) for key_ in own_values for entry_ in re.split(r"[,;/|]", str(offer.table[key_])) if _keywords(entry_)]
            if own_pieces and not any(_spelled_within(word, entry_) for word in row_["wants"] for entry_ in own_pieces):
                result_.add(row_["req"])
        return result_

    def field_text(self, field_rows: list[dict[str, Any]]) -> str:
        said_lines = []
        for row_ in field_rows:
            seen_list = " | ".join(f"{val} ({number_})" for val, number_ in sorted(row_["values"].items(), key=lambda elem: -elem[1])[:8])
            said_lines.append(f"- {row_['req']}: field {', '.join(repr(key_) for key_ in row_['keys'][:3])} on {row_['carriers']} of "
                         f"{row_['read']} read listings; values written: {seen_list}; {len(row_['writers'])} listing(s) write the "
                         "required value itself")
        return "\n".join(said_lines)

    def assess_all(self) -> None:
        _side_by_side([lambda g=g: self.assess(g, [o for o in self.hub.scout.contenders(g) if o.is_open]) for g in self.hub.goals], self.hub.meter)

    def linked_pages(self, goal: Goal, offer: Offer, cap_chars: int = 2) -> list[dict[str, Any]]:
        part_code = re.compile(r"\b[A-Z]{1,6}-?\d{3,}[A-Z0-9]*\b")
        my_text = " ".join([offer.blurb or "", offer.remarks or ""])
        my_codes = set(part_code.findall(offer.heading or ""))
        named_codes = set(part_code.findall(my_text)) - my_codes
        result_ = []
        seen_pools = [goal.offers] + [g.offers for g in self.hub.goals if g is not goal]
        seen_list: set[str] = {offer.oid}
        for seen_pool in seen_pools:
            for alt_offer in seen_pool.values():
                if alt_offer.oid in seen_list or len(result_) >= cap_chars:
                    continue
                title_codes = set(part_code.findall(alt_offer.heading or ""))
                names_mine = bool(my_codes & set(part_code.findall(" ".join([alt_offer.blurb or "", alt_offer.remarks or ""]))))
                if (title_codes & named_codes) or names_mine:
                    seen_list.add(alt_offer.oid)
                    result_.append({"id": alt_offer.oid, "title": alt_offer.heading[:120],
                                "text": " ".join([alt_offer.blurb or "", alt_offer.remarks or ""])[:400]})
        return result_

    def check(self, goal: Goal, offer: Offer) -> bool:
        slot = (goal.slot_no, offer.oid, tuple(goal.want_lines()), json.dumps(offer.table, sort_keys=True, default=str))
        memo_box = self.hub.__dict__.setdefault("checked_memo", {})
        if slot not in memo_box:
            memo_box[slot] = self._check(goal, offer)
        return memo_box[slot]

    def _check(self, goal: Goal, offer: Offer, must_be_sure: bool = False) -> bool:
        want_lines = goal.want_lines()
        user_msg = (f"REQUESTED PRODUCT: {goal.shelf} (quantity {goal.units})\n"
                f"BUDGET, ALL LINES TOGETHER: {self.hub.ceiling if self.hub.ceiling is not None else 'none stated'}\n"
                "REQUIREMENTS:\n- " + ("\n- ".join(want_lines) if want_lines else "(none beyond the product kind, budget and availability)")
                + "\n\nLISTING:\n" + json.dumps(offer.card(3500), ensure_ascii=False))
        verdict_json = self.hub.oracle.ask_json([CHECKER_SEAT, QUICK_SEAT], FINAL_CHECK_PROMPT, user_msg, 800)
        if not isinstance(verdict_json, dict):
            return not must_be_sure
        if must_be_sure and verdict_json.get("fits") is not True:
            return False
        issues = [str(issue) for issue in (verdict_json.get("problems") or [])]
        self.hub.ledger.jot("verify", id=offer.oid, fits=verdict_json.get("fits"), problems=issues[:4])
        if verdict_json.get("fits") is False and issues:
            val = goal.rulings.setdefault(offer.oid, {"failed": set(), "unsure": set()})
            for issue in issues:
                top_pick = max(want_lines, key=lambda want: _overlap(want, issue)) if want_lines else None
                val["failed"].add(top_pick if top_pick and _overlap(top_pick, issue) > 0.2 else issue)
            return False
        return True

    def by_measure(self, goal: Goal, contenders: list[Offer]) -> list[Offer]:
        if not self.hub.measure or len(contenders) < 2:
            return contenders
        remembered = getattr(self.hub, "measure_order", {}).get(goal.slot_no)
        idents = [o.oid for o in contenders]
        if remembered is None or remembered[0] != idents:
            row_list_ = [{"id": o.oid, "title": o.heading[:120], "price": o.unit_price, "properties": o.table,
                     "description": str(o.blurb or "")[:300]} for o in contenders[:12]]
            user_msg = f"MEASURE: {self.hub.measure}\nPRODUCT: {goal.label}\nLISTINGS: " + _short_json(row_list_, 9000)
            verdict_json = self.hub.oracle.ask_json([PLANNER_SEAT, SPARE_SEAT], ORDERING_PROMPT, user_msg, 600)
            sequence = [str(elem) for elem in (verdict_json.get("order") or [])] if isinstance(verdict_json, dict) else []
            self.hub.ledger.jot("best_rank", measure=self.hub.measure, order=sequence[:12])
            remembered = (idents, sequence)
            if sequence:
                self.hub.measure_order = dict(getattr(self.hub, "measure_order", {}))
                self.hub.measure_order[goal.slot_no] = remembered
        sequence = remembered[1]
        cursor_at = {oid: pos_ for pos_, oid in enumerate(sequence)}
        return sorted(contenders, key=lambda o: cursor_at.get(o.oid, len(cursor_at) + idents.index(o.oid)))

    def orderable(self, goal: Goal) -> list[Offer]:
        field_rows = self.store_fields(goal)
        return [fields_ for fields_ in goal.good_offers(lowest=self.hub.lowest) if fields_.can_buy is not False and not self.field_misses(goal, fields_, field_rows)]

    @staticmethod
    def _open_rulings(goal: Goal, offers: list[Offer]) -> None:
        for o in offers:
            if o.is_open and o.oid not in goal.rulings:
                goal.rulings[o.oid] = {"failed": set(), "unsure": set()}
                goal.judged_against[o.oid] = set()

    def _assess_message(self, goal: Goal, group: list[Offer], want_lines: list[str], filters: str) -> str:
        record_list = []
        for o in group:
            record_ = o.card()
            neighbours = self.linked_pages(goal, o)
            if neighbours:
                record_["related_pages"] = neighbours
            record_list.append(record_)
        numbered = "\n".join(f"R{key_}: {want}" for key_, want in enumerate(want_lines, 1))
        return (f"REQUESTED PRODUCT: {goal.shelf}\nREQUIREMENTS (judge each):\n" + numbered + filters
                + "\n\nLISTINGS:\n" + json.dumps(record_list, ensure_ascii=False))

    @staticmethod
    def _verdicts_by_id(goal: Goal, verdict_json: Any) -> dict[str, dict[str, Any]]:
        if not isinstance(verdict_json, dict) or not isinstance(verdict_json.get("verdicts"), list):
            return {}
        return {val["id"]: val for val in verdict_json["verdicts"] if isinstance(val, dict) and val.get("id") in goal.offers}

    @staticmethod
    def _as_want_lines(aliases: set[str], want_lines: list[str]) -> set[str]:
        result_ = set()
        for number_ in aliases:
            mapped = re.fullmatch(r"\s*R?(\d+)\s*", number_)
            if mapped and 1 <= int(mapped.group(1)) <= len(want_lines):
                result_.add(want_lines[int(mapped.group(1)) - 1])
                continue
            top_pick = max(want_lines, key=lambda want: _overlap(want, number_)) if want_lines else None
            result_.add(top_pick if top_pick and _overlap(top_pick, number_) > 0.2 else number_)
        return result_

    def _take_verdict(self, goal: Goal, o: Offer, verdict: dict[str, Any], want_lines: list[str], field_rows: list[dict[str, Any]]) -> None:
        def named(key_: str) -> set[str]:
            return {str(elem) for elem in (verdict.get(key_) or []) if elem}

        broken, open_set, loose_set = named("failed"), named("unsure"), named("loose")
        weak_set = named("weak") | loose_set
        ruling = {"failed": self._as_want_lines(broken, want_lines) | self.field_misses(goal, o, field_rows),
                  "unsure": self._as_want_lines(open_set, want_lines),
                  "weak": self._as_want_lines(weak_set, want_lines),
                  "loose": self._as_want_lines(loose_set, want_lines)}
        goal.rulings[o.oid] = ruling
        goal.judged_against[o.oid] = set(want_lines)
        for want in goal.live_wants():
            if self.table_writes(o, want):
                ruling["failed"].discard(want.as_line())
                ruling["unsure"].discard(want.as_line())
        for want in goal.live_wants():
            term_ = want.as_line()
            if all(term_ not in ruling[part] for part in ("failed", "unsure", "loose")) and self.not_written(goal, o, want):
                ruling["unsure"].add(term_)

    @staticmethod
    def _cell_values(o: Offer, named_keys: list[str]):
        for key_ in named_keys:
            for entry_ in re.split(r"[,;/|]", str(o.table[key_])):
                entry_ = " ".join(entry_.lower().split())
                if entry_:
                    yield entry_

    def _field_row(self, want: Want, read_out: list[Offer]) -> dict[str, Any] | None:
        if want.relation not in ("is", "one of") or not want.amount or re.search(r"\d", want.amount):
            return None
        want_sets = [frozenset(_keywords(word)) for word in want.amount.split("|") if _keywords(word)]
        if not want_sets:
            return None
        key_set: set[str] = set()
        seen_values: dict[str, int] = {}
        holders, writer_ids = 0, set()
        for o in read_out:
            named_keys = [key_ for key_ in o.table if _names_facet(key_, want.facet)]
            if not named_keys:
                continue
            holders += 1
            key_set.update(named_keys)
            for entry_ in self._cell_values(o, named_keys):
                seen_values[entry_] = seen_values.get(entry_, 0) + 1
                if any(word <= frozenset(_keywords(entry_)) for word in want_sets):
                    writer_ids.add(o.oid)
        if not holders:
            return None
        return {"req": want.as_line(), "prop": want.facet, "keys": sorted(key_set), "carriers": holders, "read": len(read_out),
                "values": seen_values, "wants": want_sets, "writers": writer_ids}


class Liaison:
    """What goes to the shopper and what comes back, including lines riding on tool results."""

    def __init__(self, hub: "Venture") -> None:
        self.hub = hub

    def may_message(self) -> bool:
        if "message" not in self.hub.duties:
            return False
        if self.hub.message_cap is not None and self.hub.messages_used >= self.hub.message_cap:
            return False
        return self.hub.ledger.moves_left() > 0

    def kept_messages(self) -> int:
        if "message" not in self.hub.duties or not self.hub.process:
            return 0
        def owed_for(whens: tuple[str, ...]) -> int:
            return sum(max(1, int(step_spec.get("messages") or 1)) for step_spec in self.hub.process
                       if step_spec["kind"] == "message" and step_spec["moment"] in whens)
        kept = owed_for(("before_order", "before_finish"))
        if self.hub.message_cap is not None:
            kept = min(kept, max(0, self.hub.message_cap - self.hub.messages_used))
        return kept

    def owed_messages(self) -> int:
        return 0 if self.hub.basket and self.hub.shown_basket == self.basket_key() else self.kept_messages()

    def may_ask(self) -> bool:
        if not self.may_message():
            return False
        return self.hub.message_cap is None or self.hub.messages_used < self.hub.message_cap - self.kept_messages()

    def process_check(self, when_: str, about_to: str) -> bool | None:
        hub = self.hub
        if hub.gate_tally.get(when_, 0) >= 3 or hub.ledger.moves_left() < 2 or hub.ledger.seconds_left() < CLOSING_S:
            return False
        hub.gate_tally[when_] = hub.gate_tally.get(when_, 0) + 1
        closing_tools = {(hub.duties.get(want) or {}).get("name") for want in ("checkout", "decline", "cart_add", "cart_set")} - {None}
        verdict_json = hub.oracle.ask_json([PLANNER_SEAT, SPARE_SEAT], PROCESS_PROMPT, self._gate_brief(when_, about_to, closing_tools), 900)
        if not isinstance(verdict_json, dict):
            hub.ledger.jot("gate_failed", moment=when_)
            return None
        acted = False
        deeds = [d for d in (verdict_json.get("actions") or [])[:2] if isinstance(d, dict)]
        for at, deed in enumerate(deeds):
            if deed.get("tool") not in hub.shapes or deed.get("tool") in closing_tools:
                continue
            outcome = self._gate_deed(when_, deed, deeds[at + 1:])
            if outcome == "stop":
                break
            acted = acted or outcome == "done"
        if not acted:
            hub.ledger.jot("gate_clear", moment=when_)
        return acted

    def _gate_brief(self, when_: str, about_to: str, closing_tools: set[str]) -> str:
        """What the process check reads: the request, its demanded steps, the moment, the basket, the allowances, the
        tools it may use and the calls so far."""
        hub = self.hub
        tool_specs = [{"name": n, "description": hub.brief.tool_text(n), "parameters": hub.shapes[n]}
                      for n in hub.shapes if n not in closing_tools]
        lines_now = []
        for oid, units in self.basket_key():
            goal = hub.basket_goal.get(oid)
            o = goal.offers.get(oid) if goal else None
            lines_now.append({"id": oid, "quantity": units, "title": (o.heading if o else "")[:120],
                              "price": o.unit_price if o else None, "for": goal.label if goal else None})
        total = sum((line["price"] or 0) * line["quantity"] for line in lines_now)
        messages_left = (hub.message_cap - hub.messages_used) if hub.message_cap is not None else None
        return ("REQUEST:\n" + hub.ask_text_in + "\n\nSTEPS THE REQUEST DEMANDS:\n" + json.dumps(hub.process, ensure_ascii=False)
                + f"\n\nMOMENT: {when_} (the assistant is about to {about_to})"
                + "\n\nBASKET NOW: " + json.dumps({"lines": lines_now, "total": total}, ensure_ascii=False)
                + f"\nMESSAGES LEFT: {messages_left if messages_left is not None else 'no limit'}; STEPS LEFT: {hub.ledger.moves_left()}"
                + "\n\nTOOLS:\n" + json.dumps(tool_specs, ensure_ascii=False)
                + "\n\nDONE SO FAR (tool calls and results, oldest first):\n" + json.dumps(hub.past[-30:], ensure_ascii=False))

    def _gate_args(self, tool_name: str, deed: dict[str, Any]) -> dict[str, Any] | None:
        """The arguments the check gave, kept to the schema's names and listed values and cut to its lengths; None
        when a required one is missing."""
        shape = self.hub.shapes[tool_name]
        fields_of = shape.get("properties") or {}
        argvals = {k: v for k, v in (deed.get("args") or {}).items() if k in fields_of} if isinstance(deed.get("args"), dict) else {}
        for k, v in list(argvals.items()):
            allowed = (fields_of.get(k) or {}).get("enum")
            if isinstance(allowed, list) and allowed and v not in allowed:
                argvals.pop(k)
            elif isinstance(v, str):
                argvals[k] = self._cut_text(fields_of.get(k) or {}, v)
        if any(k not in argvals for k in shape.get("required") or []):
            return None
        return argvals

    def _cut_text(self, facet: dict[str, Any], amount_: str) -> str:
        cap_chars = facet.get("maxLength")
        return amount_[:cap_chars] if isinstance(cap_chars, int) and len(amount_) > cap_chars else amount_

    def basket_key(self) -> tuple:
        return tuple(sorted((oid, int(query_ or 0)) for oid, query_ in self.hub.basket.items() if query_))

    def show_basket(self) -> bool:
        if not self.may_message() or self.hub.ledger.moves_left() < 2:
            return False
        role_spec = self.hub.duties["message"]
        pieces_said, sum_now = [], 0.0
        for oid, units in self.basket_key():
            goal = self.hub.basket_goal.get(oid)
            offer = goal.offers.get(oid) if goal else None
            heading = re.sub(r"\s+", " ", (offer.heading if offer else oid) or oid).strip()[:70]
            unit_price = offer.unit_price if offer and offer.unit_price is not None else None
            if unit_price is not None:
                sum_now += unit_price * units
            pieces_said.append(f"{units} x {heading}" + (f" ({unit_price:g} each)" if unit_price is not None else ""))
        phrase = ("Your final basket: " + "; ".join(pieces_said) + (f". Total {sum_now:g}." if sum_now else ".")
                + " Shall I check out this basket? Please confirm with a yes.")
        phrase = self.hub.brief._cut("message", role_spec["text"], phrase)
        outcome = self.hub.ledger.act(role_spec["name"], {role_spec["text"]: phrase})
        self.hub.messages_used += 1
        self.hub.shown_basket = self.basket_key()
        self.hub.ledger.jot("confirm_asked", basket=self.hub.shown_basket, ok=outcome["ok"], obs=_short_json(outcome["obs"], 160))
        return True

    def ask_shopper(self, goal: Goal, facet: str, narrowed_q: bool = False) -> dict[str, Any] | None:
        role_spec = self.hub.duties["message"]
        phrase = self.hub.brief._cut("message", role_spec["text"], self._question_text(goal, facet, narrowed_q))
        outcome = self.hub.ledger.act(role_spec["name"], {role_spec["text"]: phrase})
        self.hub.messages_used += 1
        if not narrowed_q:
            goal.asked_names.append(_plain(facet))
        reply_text = self._reply_of(outcome["obs"]) if outcome["ok"] else None
        if not reply_text:
            return None
        return self._read_reply(goal, facet, phrase, reply_text)

    def facet_named(self, goal: Goal, reply_text: str, asked_names: list[str], amount_: str) -> str | None:
        phrase = " " + _plain(reply_text) + " "
        if any(first_ and _plain(first_) and " " + _plain(first_) + " " in phrase for first_ in asked_names):
            return None
        name_words = _keywords(goal.label) | _keywords(goal.shelf)
        if any(word in _keywords(reply_text) for first_ in asked_names if first_ for word in _keywords(first_) - name_words):
            # (the reply names a word of the asked property: the reading stands)
            return None
        key_set: set[str] = set()
        for o in goal.offers.values():
            for key_ in o.table:
                slot = _plain(re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(key_)))
                if len(slot) >= 4 and not (_keywords(slot) & _keywords(amount_)) and not any(_same_name(slot, first_) for first_ in asked_names if first_):
                    key_set.add(slot)
        named_hits = [key_ for key_ in key_set if " " + key_ + " " in phrase]
        named_hits = [key_ for key_ in named_hits if not any(key_ != other_key and " " + key_ + " " in " " + other_key + " " for other_key in named_hits)]
        return named_hits[0] if len(named_hits) == 1 else None

    def take_reply(self, goal: Goal, facet: str, verdict_json: dict[str, Any] | None) -> None:
        if not verdict_json:
            return
        for step_in in verdict_json.get("steps") or []:
            if isinstance(step_in, dict):
                self.add_process_step(step_in)
        sort = str(verdict_json.get("kind") or "").lower()
        self._take_budget(verdict_json.get("budget"))
        if sort == "need" and (verdict_json.get("value") or verdict_json.get("reference")):
            self._take_need(goal, facet, verdict_json)
        elif sort == "free":
            goal.free_names.append(_plain(str(verdict_json.get("property") or facet)))

    def take_reply_guarded(self, goal: Goal, facet: str, verdict_json: dict[str, Any] | None) -> None:
        remaining = (self.hub.message_cap - self.hub.messages_used) if self.hub.message_cap is not None else 2
        self.hub.chain_floor = max(0, remaining) + self.hub.closing_steps()
        try:
            self.take_reply(goal, facet, verdict_json)
        finally:
            self.hub.chain_floor = 0

    def read_news(self) -> bool:
        hub = self.hub
        said_lines = [o for o in hub.unread_lines if o not in hub.read_lines]
        hub.unread_lines = []
        if not said_lines:
            return False
        for o in said_lines:
            hub.read_lines.add(o)
        user_msg = "LINES:\n- " + "\n- ".join(said_lines) + "\n\nREQUESTED ITEMS: " + json.dumps([g.label for g in hub.goals])
        verdict_json = hub.oracle.ask_json([QUICK_SEAT, SPARE_SEAT], NEWS_PROMPT, user_msg, 700)
        hub.ledger.jot("event", lines=[o[:160] for o in said_lines], read=_short_json(verdict_json, 300))
        if not isinstance(verdict_json, dict):
            return False
        handlers = {"budget": self._news_budget, "drop": self._news_drop, "add": self._news_add,
                    "price": self._news_cart, "stock": self._news_gone, "step": self._news_step}
        moved_on = False
        for change in verdict_json.get("changes") or []:
            if not isinstance(change, dict):
                continue
            handler = handlers.get(str(change.get("kind") or "").lower())
            if handler is not None and handler(change, self._news_goal(change), said_lines):
                moved_on = True
        return moved_on

    def _news_goal(self, change: dict[str, Any]) -> Goal | None:
        goals = self.hub.goals
        if not change.get("item"):
            return None
        hit = max(goals, key=lambda g: _overlap(g.label, change.get("item")))
        if _overlap(hit.label, change.get("item")) < 0.2 and len(goals) > 1:
            return None
        return hit

    def _news_budget(self, change: dict[str, Any], goal_hit: Goal | None, said_lines: list[str]) -> bool:
        """A new ceiling, taken as the line writes it (a figure a hundred times off is the line's own figure)."""
        hub = self.hub
        if not _amount(change.get("budget")):
            return False
        ceiling = _amount(change.get("budget"))
        figures = [float(elem.replace(",", "")) for elem in re.findall(r"\d[\d,]*(?:\.\d+)?", " ".join(said_lines))]
        if ceiling not in figures:
            scaled = next((x for x in (ceiling * 100, ceiling / 100) if x in figures), None)
            if scaled is not None:
                ceiling = scaled
            elif figures and hub.ceiling is not None:
                near = [elem for elem in figures if hub.ceiling / 20 <= elem <= hub.ceiling * 20]
                if near:
                    ceiling = near[0]
        hub.ceiling = ceiling
        return True

    def _news_drop(self, change: dict[str, Any], goal_hit: Goal | None, said_lines: list[str]) -> bool:
        if not change.get("property"):
            return False
        moved_on = False
        for goal in ([goal_hit] if goal_hit else self.hub.goals):
            before = len(goal.wants)
            goal.wants = [want for want in goal.wants if _overlap(want.facet, change["property"]) < 0.5]
            if len(goal.wants) != before:
                moved_on = True
                for ruling in goal.rulings.values():
                    ruling["failed"] = {f for f in ruling["failed"] if _overlap(f.split(":")[0], change["property"]) < 0.5}
        return moved_on

    def _news_add(self, change: dict[str, Any], goal_hit: Goal | None, said_lines: list[str]) -> bool:
        if not (change.get("property") and change.get("value")):
            return False
        moved_on = False
        for goal in ([goal_hit] if goal_hit else self.hub.goals[:1]):
            goal.wants.append(Want(change["property"], change.get("op") or "is", change["value"], None, source="shopper"))
            moved_on = True
        return moved_on

    def _news_cart(self, change: dict[str, Any], goal_hit: Goal | None, said_lines: list[str]) -> bool:
        self.hub.closer.reread_basket()
        return True

    def _news_gone(self, change: dict[str, Any], goal_hit: Goal | None, said_lines: list[str]) -> bool:
        """A sold-out notice: the pick it names (or the only pick there is) is marked unbuyable at once, no re-read
        needed; otherwise the basket is read again."""
        picked = [g for g in ([goal_hit] if goal_hit else self.hub.goals) if g.pick is not None]
        if len(picked) != 1:
            return self._news_cart(change, goal_hit, said_lines)
        picked[0].pick.can_buy = False
        self.hub.ledger.jot("sold_out", item=picked[0].label, id=picked[0].pick.oid)
        return True

    def _news_step(self, change: dict[str, Any], goal_hit: Goal | None, said_lines: list[str]) -> bool:
        if isinstance(change.get("step"), dict):
            self.add_process_step(change["step"])
        return False

    def add_process_step(self, step_in: dict[str, Any]) -> None:
        if not step_in.get("goal") or len(self.hub.process) >= 8:
            return
        step_in = dict(step_in)
        step_in["moment"] = str(step_in.get("moment") or "before_finish").lower()
        step_in["kind"] = "message" if str(step_in.get("kind") or "").lower() == "message" else "tool"
        if any(_overlap(step_in["goal"], step_spec.get("goal")) >= 0.6 for step_spec in self.hub.process):
            return
        self.hub.process.append(step_in)
        self.hub.show_basket_first = self.hub.show_basket_first or (step_in["kind"] == "message" and step_in["moment"] in ("before_order", "before_finish"))
        self.hub.ledger.jot("step_added", step=step_in)

    def _gate_deed(self, when_: str, deed: dict[str, Any], later: list[dict[str, Any]]) -> str:
        hub = self.hub
        tool_name = deed["tool"]
        message_tool = (hub.duties.get("message") or {}).get("name")
        is_message = tool_name == message_tool
        if not is_message:
            owed = max(sum(1 for d in later if d.get("tool") == message_tool), self.owed_messages())
            if hub.ledger.moves_left() < 2 + owed:
                hub.ledger.jot("gate_step_skipped", moment=when_, tool=tool_name, steps=hub.ledger.moves_left(), due=owed)
                return "skip"
        argvals = self._gate_args(tool_name, deed)
        if argvals is None or (is_message and not self.may_message()):
            return "skip"
        if hub.ledger.moves_left() < 2:
            return "stop"
        outcome = hub.ledger.act(tool_name, argvals)
        if is_message:
            hub.messages_used += 1
            hub.shown_basket = self.basket_key()
        hub.ledger.jot("gate_step", moment=when_, tool=tool_name, why=str(deed.get("why") or "")[:160], ok=outcome["ok"])
        return "done"

    def _question_text(self, goal: Goal, facet: str, narrowed_q: bool) -> str:
        if not narrowed_q:
            return f"Which {facet} would you like the {goal.label} to have?"
        seen_list: list[str] = []
        for o in goal.offers.values():
            for key_, val in o.table.items():
                if _same_name(re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(key_)).replace("_", " "), facet) and str(val) not in seen_list:
                    seen_list.append(str(val)[:30])
        samples = " or ".join(seen_list[:2])
        return f"Just this one thing: for the {goal.label}, what {facet} do you need" + (f", for example {samples}?" if samples else "?")

    @staticmethod
    def _reply_of(seen: Any) -> str | None:
        if isinstance(seen, str):
            return seen
        if isinstance(seen, dict):
            text_values = [val for val in seen.values() if isinstance(val, str)]
            return max(text_values, key=len) if text_values else _short_json(seen, 400)
        return None

    def _read_reply(self, goal: Goal, facet: str, phrase: str, reply_text: str) -> dict[str, Any] | None:
        seen_offer = [f"{want.facet}: {want.relation} {want.amount}" for want in goal.wants if want.source == "shopper" and want.amount]
        user_msg = (f"ITEM: {goal.label}\nPROPERTY ASKED ABOUT: {facet}\nQUESTION: {phrase}\nKNOWN NEEDS: {json.dumps(seen_offer)}\nREPLY: {reply_text}")
        verdict_json = self.hub.oracle.ask_json([QUICK_SEAT, SPARE_SEAT, PLANNER_SEAT], ANSWER_PROMPT, user_msg, 600)
        if isinstance(verdict_json, dict) and str(verdict_json.get("kind") or "").lower() == "need":
            named_prop = self.facet_named(goal, reply_text, [facet, str(verdict_json.get("property") or "")], str(verdict_json.get("value") or ""))
            if named_prop:
                verdict_json["property"] = named_prop
        self.hub.ledger.jot("reply", item=goal.label, prop=facet, reply=reply_text[:200], read=_short_json(verdict_json, 300))
        return verdict_json if isinstance(verdict_json, dict) else None

    def _take_budget(self, said: Any) -> None:
        keep_steps = _amount(said)
        if keep_steps is not None and keep_steps > 0:
            self.hub.ceiling = keep_steps
            self.hub.ledger.jot("budget", cap=keep_steps)

    def _take_need(self, goal: Goal, facet: str, verdict_json: dict[str, Any]) -> None:
        issue = str(verdict_json.get("property") or facet)
        want_ = Want(issue, verdict_json.get("op") or "is", verdict_json.get("value") or "", verdict_json.get("reference"), source="shopper")
        want_.search = str(verdict_json.get("search") or "").strip()[:60]
        if want_.pointer:
            want_.pointer["property"] = want_.facet
        if want_.amount and any(want.source == "shopper" and _plain(want.amount) == _plain(want_.amount) and want.relation == want_.relation
                                for want in goal.wants):
            return
        goal.wants = [want for want in goal.wants if not (want.slot() == want_.slot() and want.source == "shopper")]
        goal.wants.append(want_)
        if want_.pointer and want_.relation in POINTER_OPS:
            self.hub.tracer.follow_pointer(goal, want_)


class Inquiry:
    """The order of the hidden-need questions."""

    def __init__(self, hub: "Venture") -> None:
        self.hub = hub

    def facet_menu(self, goal: Goal) -> list[dict[str, Any]]:
        is_open = [o for o in goal.offers.values() if o.is_open and self.hub.scout.on_shelf(goal, o)]
        standing = [o for o in is_open if not (goal.rulings.get(o.oid) or {}).get("failed")]
        if len(standing) >= 4:
            is_open = standing
        per_offer = [self._slot_fields(o) for o in is_open]
        tallies: dict[str, int] = {}
        for fields_ in per_offer:
            for key_ in fields_:
                tallies[key_] = tallies.get(key_, 0) + 1
        key_set = sorted(tallies, key=lambda key_: (-tallies[key_], len(key_), key_))
        twin_of = self._twins(key_set, per_offer)
        settled = {want.slot() for want in goal.wants} | set(goal.asked_names)

        def known(number_: str) -> bool:
            return _plain(number_) in settled or any(_overlap(number_, fields_) >= 0.67 for fields_ in settled)

        carte = []
        for slot in key_set:
            if slot in twin_of:
                continue
            aliases = [slot] + [second_ for second_, first_ in twin_of.items() if first_ == slot]
            pickers = self.hub.__dict__.setdefault("picker_slots", set())
            if any(number_ in pickers for number_ in aliases):
                continue
            if not any(known(number_) for number_ in aliases):
                entry_ = self._menu_entry(slot, aliases, per_offer, len(is_open))
                if entry_ is not None:
                    carte.append(entry_)
                elif len(aliases) == 1:
                    # (a variant picker in one category is the store's picker field in every category)
                    pickers.add(slot)
        carte.sort(key=lambda entry_: (-entry_["_w"], entry_["property"]))
        for entry_ in carte:
            entry_.pop("_w", None)
        return carte[:40]

    def story_read(self) -> None:
        sections = []
        for goal in self.hub.goals:
            sections.append({"item": goal.label, "category": goal.shelf, "known_requirements": goal.want_lines(),
                           "hints": goal.story, "property_menu": [entry_["property"] for entry_ in self.facet_menu(goal)]})
        user_msg = ("REQUEST:\n" + self.hub.ask_text_in + "\n\nSUBJECTS THE REQUEST TELLS THE ASSISTANT TO RAISE: " + json.dumps(self.hub.raise_topics)
                + "\n\nITEMS:\n" + json.dumps(sections, ensure_ascii=False))
        verdict_json = self.hub.oracle.ask_json([PLANNER_SEAT, QUICK_SEAT], ASK_PLAN_PROMPT, user_msg, 500)
        self.hub.story_props: list[tuple[Goal, str]] = []
        self.hub.junk_props: set[str] = set()
        if not isinstance(verdict_json, dict):
            return
        for query_ in verdict_json.get("hinted") or []:
            if not isinstance(query_, dict) or not query_.get("property"):
                continue
            goal_hit = max(self.hub.goals, key=lambda g: _overlap(g.label, query_.get("item") or g.label))
            facet = re.sub(r"\s+", " ", str(query_["property"]).replace("_", " ")).strip().lower()
            if facet and not any(g is goal_hit and _same_name(issue, facet) for g, issue in self.hub.story_props):
                self.hub.story_props.append((goal_hit, facet))
        self.hub.junk_props = {re.sub(r"\s+", " ", str(job_).replace("_", " ")).strip().lower() for job_ in (verdict_json.get("junk") or []) if job_}
        self.hub.ledger.jot("hints", hinted=[(g.label, issue) for g, issue in self.hub.story_props], junk=sorted(self.hub.junk_props)[:12])

    def settled_facets(self, goal: Goal) -> list[str]:
        result_ = []
        for third_ in getattr(self.hub, "rules_of", []):
            if third_["src"] is goal:
                result_.append(third_["prop"])
            result_ += [str(second_["property"]) for _, fourth_, second_ in third_["branches"] if fourth_ is goal]
        if any(first_ is goal for first_, _ in self.hub.coupler.fit_couples()):
            result_.append("compatibility")
        return list(dict.fromkeys(result_))

    def next_questions(self, how_many: int, seat: str = PLANNER_SEAT) -> list[tuple[Goal, str]]:
        if how_many <= 0:
            return []
        picked: list[tuple[Goal, str]] = []

        def already_taken(goal: Goal, facet: str) -> bool:
            return self._known_facet(goal, facet) or any(g is goal and _same_name(facet, issue) for g, issue in picked)

        if not getattr(self.hub, "plan_order", None):
            self._draw_plan(how_many, seat)
        if len(self.hub.goals) > 1:
            return self.rota_plan(how_many, already_taken)
        for goal, facet in self.hub.plan_order:
            if len(picked) >= how_many:
                break
            if not already_taken(goal, facet):
                picked.append((goal, facet))
        if how_many >= 3:
            self._weave_leads(picked, already_taken)
            del picked[how_many:]
        if len(picked) < how_many:
            self._fill_from_menus(picked, how_many, already_taken)
        return picked[:how_many]

    def rota_plan(self, how_many: int, already_taken: Any) -> list[tuple[Goal, str]]:
        lanes_of, leader = self._lanes_from_plan(already_taken)
        for g in self.hub.goals:
            self._widen_lane(g, lanes_of[g.slot_no], already_taken)
            if g.slot_no not in leader:
                leader.append(g.slot_no)
        told_ids = {g.slot_no for g in self.hub.goals if any(want.source == "shopper" for want in g.wants)}
        goal_by_index = {g.slot_no: g for g in self.hub.goals}
        rota = sorted(leader, key=lambda pos_: (pos_ in told_ids, len(goal_by_index[pos_].asked_names), leader.index(pos_)))
        if any(lanes_of[pos_] for pos_ in rota if pos_ not in told_ids):
            # (an item whose need is already told waits while another item's need is still unknown)
            rota = [pos_ for pos_ in rota if pos_ not in told_ids]
        intake: list[tuple[Goal, str]] = []
        while len(intake) < how_many and any(lanes_of[pos_] for pos_ in rota):
            for pos_ in rota:
                if lanes_of[pos_] and len(intake) < how_many:
                    intake.append((goal_by_index[pos_], lanes_of[pos_].pop(0)))
        return intake

    @staticmethod
    def _slot_fields(o: Offer) -> dict[str, str]:
        fields_: dict[str, str] = {}
        for key_, val in o.table.items():
            slot = _slot_name(key_)
            if slot and slot not in fields_:
                fields_[slot] = str(val).strip().lower()[:60]
        return fields_

    @staticmethod
    def _twins(key_set: list[str], per_offer: list[dict[str, str]]) -> dict[str, str]:
        def mirrors(first_: str, second_: str) -> bool:
            pair_rows = [fields_ for fields_ in per_offer if first_ in fields_ and second_ in fields_]
            return len(pair_rows) >= 2 and sum(1 for fields_ in pair_rows if fields_[first_] == fields_[second_]) >= 0.8 * len(pair_rows)

        twin_of: dict[str, str] = {}
        for pos_, first_ in enumerate(key_set):
            if first_ in twin_of:
                continue
            for second_ in key_set[pos_ + 1:]:
                if second_ not in twin_of and mirrors(first_, second_):
                    twin_of[second_] = first_
        return twin_of

    @staticmethod
    def _menu_entry(slot: str, aliases: list[str], per_offer: list[dict[str, str]], shown: int) -> dict[str, Any] | None:
        found = [next((fields_[number_] for number_ in aliases if number_ in fields_), None) for fields_ in per_offer]
        holders = sum(1 for val in found if val is not None)
        seen_values: list[str] = []
        for val in found:
            if val and val not in seen_values:
                seen_values.append(val)
        combos = sum(1 for val in found if val and "/" in val)
        if holders >= 3 and len(set(found) - {None}) >= holders and combos * 2 >= holders:
            # (a variant picker: every listing writes its own "colour / size" combination)
            return None
        heft = holders * (0.5 if len(seen_values) == 1 and holders >= 3 else 1.0)
        if holders >= 4 and len(set(found) - {None}) >= holders:
            heft *= 0.5
        return {"property": slot, "listings": holders, "of": shown, "values": seen_values[:3],
                **({"same_as": aliases[1:4]} if len(aliases) > 1 else {}), "_w": heft}

    def _known_facet(self, goal: Goal, facet: str) -> bool:
        name_words = _keywords(goal.label) | _keywords(goal.shelf)
        own = _keywords(facet)

        def nested(first_: str) -> bool:
            # (one name inside the other, the smaller one carrying a word of its own: "fit" in "pants fit type")
            other = _keywords(first_)
            small, large = (own, other) if len(own) <= len(other) else (other, own)
            return bool(small - name_words) and small <= large

        if any(_same_name(facet, first_) or nested(first_) for first_ in goal.asked_names):
            return True
        if any(_same_name(facet, fourth_) or _overlap(facet, fourth_) >= 0.5 for fourth_ in self.settled_facets(goal)):
            return True
        return any(_same_name(facet, want.facet) for want in goal.wants)

    def _store_pickers(self) -> None:
        """Variant-picker fields found in any category this store showed (pointer pages included)."""
        pickers = self.hub.__dict__.setdefault("picker_slots", set())
        for tables in getattr(self.hub, "seen_tables", {}).values():
            per_offer = []
            for table in tables.values():
                fields_: dict[str, str] = {}
                for key_, val in table.items():
                    slot = _slot_name(key_)
                    if slot and slot not in fields_:
                        fields_[slot] = str(val).strip().lower()[:60]
                per_offer.append(fields_)
            for slot in {slot for fields_ in per_offer for slot in fields_}:
                if slot not in pickers and self._menu_entry(slot, [slot], per_offer, len(per_offer)) is None:
                    pickers.add(slot)

    def _draw_plan(self, how_many: int, seat: str) -> None:
        self._store_pickers()
        for goal in self.hub.goals:
            self.facet_menu(goal)
        sections = [{"item": goal.label, "category": goal.shelf, "known_requirements": goal.want_lines(),
                     "decided_by_the_request": self.settled_facets(goal),
                     "hints": goal.story, "already_asked": goal.asked_names, "property_menu": self.facet_menu(goal)}
                    for goal in self.hub.goals]
        sum_now = self.hub.message_cap if self.hub.message_cap is not None else how_many
        user_msg = ("REQUEST:\n" + self.hub.ask_text_in + "\n\nSUBJECTS THE REQUEST TELLS THE ASSISTANT TO RAISE: " + json.dumps(self.hub.raise_topics)
                    + "\n\nITEMS:\n" + json.dumps(sections, ensure_ascii=False)
                    + f"\n\nQUESTION_COUNT = {max(sum_now, how_many)}")
        verdict_json = self.hub.oracle.ask_json([seat, QUICK_SEAT if seat != QUICK_SEAT else SPARE_SEAT], ASK_PLAN_PROMPT, user_msg, 700)
        in_order = self._planned_pairs(verdict_json)
        self.hub.plan_order = in_order
        self.hub.ledger.jot("plan_questions", questions=[(g.label, issue) for g, issue in in_order])

    def _planned_pairs(self, verdict_json: Any) -> list[tuple[Goal, str]]:
        in_order: list[tuple[Goal, str]] = []
        if not isinstance(verdict_json, dict) or not isinstance(verdict_json.get("questions"), list):
            return in_order
        for query_ in verdict_json["questions"]:
            if not isinstance(query_, dict) or not query_.get("property"):
                continue
            goal_hit = max(self.hub.goals, key=lambda g: _overlap(g.label, query_.get("item") or g.label))
            facet = re.sub(r"\s+", " ", str(query_["property"]).replace("_", " ")).strip().lower()
            if facet and not any(g is goal_hit and _same_name(facet, issue) for g, issue in in_order):
                in_order.append((goal_hit, facet))
        return in_order

    def _weave_leads(self, picked: list[tuple[Goal, str]], already_taken: Any) -> None:
        lead_count = 2 if len(self.hub.goals) == 1 else 1
        for goal in self.hub.goals:
            leads = [entry_["property"] for entry_ in self.facet_menu(goal) if entry_["listings"] >= 3][:lead_count]
            for key_, facet in enumerate(leads):
                if already_taken(goal, facet):
                    continue
                cursor_at = min(len(picked), 1 + 2 * key_ if len(self.hub.goals) == 1 else len(self.hub.goals))
                picked.insert(cursor_at, (goal, facet))

    def _fill_from_menus(self, picked: list[tuple[Goal, str]], how_many: int, already_taken: Any) -> None:
        cartes = {g.slot_no: [entry_ for entry_ in self.facet_menu(g) if entry_["listings"] >= 2 or entry_["of"] <= 2] for g in self.hub.goals}
        sequence = sorted(self.hub.goals, key=lambda g: (sum(1 for head, _ in picked if head is g) + len(g.asked_names), g.slot_no))
        while len(picked) < how_many:
            grew = False
            for goal in sequence:
                while cartes[goal.slot_no]:
                    facet = cartes[goal.slot_no].pop(0)["property"]
                    if not already_taken(goal, facet):
                        picked.append((goal, facet))
                        grew = True
                        break
                if len(picked) >= how_many:
                    break
            if not grew:
                break

    def _lanes_from_plan(self, already_taken: Any) -> tuple[dict[int, list[str]], list[int]]:
        lanes_of: dict[int, list[str]] = {g.slot_no: [] for g in self.hub.goals}
        leader: list[int] = []
        for g, prop_of in self.hub.plan_order:
            if g.slot_no in lanes_of and not already_taken(g, prop_of) and not any(_same_name(prop_of, query_) for query_ in lanes_of[g.slot_no]):
                lanes_of[g.slot_no].append(prop_of)
                if g.slot_no not in leader:
                    leader.append(g.slot_no)
        return lanes_of, leader

    def _widen_lane(self, g: Goal, query_: list[str], already_taken: Any) -> None:
        carte = [entry_ for entry_ in self.facet_menu(g) if entry_["listings"] >= 2 or entry_["of"] <= 2]
        by_count = sorted([entry_ for entry_ in carte if entry_["listings"] >= 3 and not already_taken(g, entry_["property"])],
                          key=lambda entry_: -entry_["listings"])
        tops = [entry_["property"] for entry_ in by_count[:2]]
        lead = [elem for elem in query_ if any(_same_name(elem, hint) or _overlap(elem, hint) >= 0.5 for hint in g.story)][:2]
        if len(lead) < 2:
            # (story hints the planner left out but the menu carries)
            lead += [entry_["property"] for entry_ in carte
                     if not already_taken(g, entry_["property"]) and not any(_same_name(entry_["property"], elem) for elem in lead)
                     and any(_same_name(entry_["property"], hint) or _overlap(entry_["property"], hint) >= 0.5 for hint in g.story)][:2 - len(lead)]
        tops = [top for top in tops if not any(_same_name(top, elem) for elem in lead)]
        if tops or lead:
            # (story hints first, then the properties most listings carry, where hidden needs mostly sit)
            query_[:] = lead + tops + [elem for elem in query_ if not any(_same_name(elem, top) for top in lead + tops)]
        for entry_ in carte:
            if not already_taken(g, entry_["property"]) and not any(_same_name(entry_["property"], elem) for elem in query_):
                query_.append(entry_["property"])


class _HopTarget:
    """The listing one hop is after (its seller, shelf and described values) and the code's tests of a candidate."""

    def __init__(self, pointer: dict[str, Any], want_: Want) -> None:
        self.vendor = str(pointer.get("seller") or "").strip()
        self.shelf = str(pointer.get("category") or "").strip()
        self.values = pointer.get("attributes") if isinstance(pointer.get("attributes"), dict) else {}
        self.facet = str(pointer.get("property") or want_.facet).strip()
        self.words = " ".join(str(val) for val in list(self.values.values())[:3] if val)
        self.rule = str(pointer.get("pick") or "").strip()
        self.count = pointer.get("count") if isinstance(pointer.get("count"), int) else None

    def queries(self) -> list[str]:
        if self.vendor:
            out = [(self.vendor + " " + self.words).strip()]
            if self.shelf:
                out.append(self.vendor + " " + self.shelf)
            return out + [self.vendor]
        return [(self.shelf + " " + self.words).strip()] if self.shelf else []

    def sold_by(self, o: Offer) -> bool:
        if not self.vendor:
            return True
        if _overlap(o.vendor, self.vendor) >= 0.5 or _overlap(o.make, self.vendor) >= 0.5:
            return True
        return _plain(self.vendor) in _plain(o.heading + " " + o.blurb + " " + o.remarks)

    def shared(self, o: Offer) -> int:
        wanted = _keywords(self.words)
        if not wanted:
            return 0
        return len(wanted & _keywords(o.heading + " " + o.blurb + " " + o.remarks + " " + json.dumps(o.table)))

    def written(self, o: Offer) -> bool:
        if not self.values:
            return False
        rows = " ".join(f"{key_} {val}" for key_, val in o.table.items())
        text = _all_words(" ".join([o.heading or "", o.remarks or "", o.blurb or "", rows]))
        return all(_all_words(val) and _all_words(val) <= text for val in self.values.values() if str(val).strip())

    def tabled(self, o: Offer) -> bool:
        if not self.values or not o.is_open:
            return False
        rows = [(_plain(re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(key_))), _plain(val)) for key_, val in o.table.items()]
        for key_, val in self.values.items():
            wanted = _plain(val)
            if not wanted:
                return False
            if not any((_same_name(rk, key_) or _overlap(rk, key_) >= 0.5) and rv == wanted for rk, rv in rows):
                return False
        return True

    def described(self) -> str:
        return ("DESCRIPTION: category " + self.shelf + "; seller " + self.vendor + "; attributes "
                + json.dumps(self.values, ensure_ascii=False))


class Tracer:
    """Wants whose value sits on another listing's page, followed page to page."""

    def __init__(self, hub: "Venture") -> None:
        self.hub = hub

    def follow_pointer(self, goal: Goal, want_: Want, hop_no: int = 0) -> None:
        while self._hop(goal, want_, hop_no):
            hop_no += 1

    def _hop(self, goal: Goal, want_: Want, hop_no: int) -> bool:
        """One page of the chain; True when its notes sent the want on to another page (the pointer is updated)."""
        hub = self.hub
        pointer = want_.pointer or {}
        keep_free = max(4, getattr(hub, "chain_floor", 0))
        if hop_no > 7 or hub.ledger.moves_left() < keep_free or hub.ledger.seconds_left() < CLOSING_S + 20:
            return False
        target = _HopTarget(pointer, want_)
        scratch = Goal({"name": target.shelf or "listing", "category": target.shelf, "search_terms": []}, 99)
        matches = self._seek(target, scratch)
        if not matches:
            hub.ledger.jot("reference_unfound", seller=target.vendor, category=target.shelf)
            return False
        matches.sort(key=lambda o: (-target.shared(o), not (_overlap(o.vendor, target.vendor) >= 0.5), o.unit_price or 0))
        page = None
        if target.rule or (target.count or 0) > 1:
            page = self._by_rule(target, scratch, matches)
        if page is None:
            page = self._by_batches(target, scratch, matches)
        if page is None:
            hub.ledger.jot("reference_unmatched", seller=target.vendor, category=target.shelf)
            return False
        return self._read_page(goal, want_, target, page, hop_no)

    def _seek(self, target: _HopTarget, scratch: Goal) -> list[Offer]:
        """The seller's listings of the described kind, from the first query that finds them."""
        scout, ledger = self.hub.scout, self.hub.ledger
        matches: list[Offer] = []
        for query_ in target.queries():
            if ledger.moves_left() < 4:
                break
            for _ in range(2):
                if scout.look_up(scratch, words=query_, page_limit=1, keep_steps=3) == 0:
                    break
                matches = [o for o in scratch.offers.values()
                           if target.sold_by(o) and (not target.shelf or scout.on_shelf(scratch, o))]
                if matches:
                    break
            if len(matches) >= 2 or (matches and target.shared(matches[0]) > 0):
                break
        return matches

    def _by_rule(self, target: _HopTarget, scratch: Goal, matches: list[Offer]) -> Offer | None:
        """Several of the seller's listings fit: every candidate page is read, then the model applies the rule."""
        hub = self.hub
        width = hub.side_limit
        for begin in range(0, min(len(matches), 3 * width), width):
            if hub.ledger.moves_left() < 3:
                break
            for o in hub.scout.open_offers(None, [elem.oid for elem in matches[begin:begin + width]]):
                scratch.offers[o.oid] = o
        cards = [scratch.offers[o.oid].card(1200) for o in matches if o.oid in scratch.offers and scratch.offers[o.oid].is_open]
        if not cards:
            return None
        user_msg = (target.described() + (f"\nPICK: {target.rule}" if target.rule else "")
                    + (f"\nLISTINGS THAT FIT: {target.count}" if target.count else "")
                    + "\n\nCANDIDATES:\n" + json.dumps(cards, ensure_ascii=False))
        chosen = hub.oracle.ask_json([QUICK_SEAT, SPARE_SEAT], PICK_PAGE_PROMPT, user_msg, 600)
        if isinstance(chosen, dict) and chosen.get("id") in scratch.offers:
            return scratch.offers[chosen["id"]]
        return None

    def _by_batches(self, target: _HopTarget, scratch: Goal, matches: list[Offer]) -> Offer | None:
        """Candidates opened a batch at a time; a page whose table states every described value wins at once, else
        the model picks among the pages read (a lone page whose text carries every value, or a last lone page)."""
        hub = self.hub
        single = not hub.duties.get("compare") or getattr(hub, "side_off", False)
        size = 1 if single else hub.side_limit
        rounds, waiting = 0, []
        while matches and rounds < (5 if single else 3) and hub.ledger.moves_left() >= 3:
            group, matches = matches[:size], matches[size:]
            rounds += 1
            for o in hub.scout.open_offers(None, [o.oid for o in group]):
                scratch.offers[o.oid] = o
            now_open = [scratch.offers[o.oid] for o in group if o.oid in scratch.offers and scratch.offers[o.oid].is_open]
            exact = next((o for o in now_open if target.tabled(o)), None)
            if exact is not None:
                return exact
            waiting += now_open
            if single and matches and len(waiting) < 3:
                continue
            shown = waiting if single else now_open
            waiting = []
            if shown:
                found = self._pick_shown(target, scratch, shown, last=not matches)
                if found is not None:
                    return found
        return None

    def _read_page(self, goal: Goal, want_: Want, target: _HopTarget, page: Offer, hop_no: int) -> bool:
        """The page read for the wanted facet: its notes may send it on (True), else its value settles the want."""
        hub = self.hub
        facet = target.facet
        pair = want_.relation == "counterpart of reference"

        def ask() -> Any:
            user_msg = f"PROPERTY WANTED: {facet}\n\nPAGE:\n" + json.dumps(page.card(3000), ensure_ascii=False)
            return hub.oracle.ask_json([QUICK_SEAT, SPARE_SEAT], PAGE_VALUE_PROMPT, user_msg, 600)

        no_table = hub.side_lacks_table and not page.table
        if no_table and not page.remarks and hub.ledger.moves_left() >= 3:
            page.table, no_table = hub.scout.table_of(page.oid) or {}, False
        read_out = ask()
        if no_table and hub.ledger.moves_left() >= 3 and isinstance(read_out, dict) and not read_out.get("redirect") \
                and (not read_out.get("value") or pair):
            page.table = hub.scout.table_of(page.oid) or {}
            if page.table:
                read_out = ask()
        hub.ledger.jot("reference", hop=hop_no, page=page.oid, read=_short_json(read_out, 200))
        onward = read_out.get("redirect") if isinstance(read_out, dict) else None
        if isinstance(onward, dict) and onward and not self._same_facet(str(onward.get("property") or facet), facet):
            onward = self._own_value_instead(read_out, page, facet, no_table, str(onward.get("property") or facet))
        if isinstance(onward, dict) and onward:
            want_.pointer = onward
            want_.pointer["property"] = facet
            return True
        self._settle(goal, want_, read_out, page, pair)
        return False

    @staticmethod
    def _same_facet(subject: str, facet: str) -> bool:
        return (_same_name(subject, facet) or _overlap(subject, facet) >= 0.5
                or _plain(facet) in _plain(subject) or _plain(subject) in _plain(facet))

    def _own_value_instead(self, read_out: dict[str, Any], page: Offer, facet: str, no_table: bool, subject: str) -> None:
        """The notes send some OTHER property elsewhere: this page's own value of the wanted one stands."""
        hub = self.hub
        if not read_out.get("value") and no_table and not page.table and hub.ledger.moves_left() >= 3:
            page.table = hub.scout.table_of(page.oid) or {}
            again = None
            if page.table:
                user_msg = f"PROPERTY WANTED: {facet}\n\nPAGE:\n" + json.dumps(page.card(3000), ensure_ascii=False)
                again = hub.oracle.ask_json([QUICK_SEAT, SPARE_SEAT], PAGE_VALUE_PROMPT, user_msg, 600)
            if isinstance(again, dict):
                read_out["value"] = again.get("value") or read_out.get("value")
                read_out["paired"] = again.get("paired") or read_out.get("paired")
        if not read_out.get("value"):
            own = next((val for key_, val in page.table.items()
                        if _same_name(str(key_).replace("_", " "), facet) or _overlap(str(key_).replace("_", " "), facet) >= 0.5), None)
            if own not in (None, ""):
                read_out["value"] = str(own)
        hub.ledger.jot("reference_other_redirect", about=subject, wanted=facet, value=read_out.get("value"))
        return None

    def _settle(self, goal: Goal, want_: Want, read_out: Any, page: Offer, pair: bool) -> None:
        amount_ = read_out.get("paired" if pair else "value") if isinstance(read_out, dict) else None
        if amount_:
            want_.amount = str(amount_)
            want_.relation = POINTER_OPS.get(want_.relation, "is")
            want_.pointer = None
            if want_.relation == "is" and len(want_.amount) <= 40 and goal.queries \
                    and not any(_plain(want_.amount) in _plain(term_) for term_ in goal.queries):
                goal.queries.insert(0, f"{want_.amount} {goal.head_noun}"[:60])
        elif pair and isinstance(read_out, dict) and read_out.get("value"):
            self.hub.ledger.jot("pair_unstated", page=page.oid, value=read_out.get("value"))

    def _pick_shown(self, target: _HopTarget, scratch: Goal, shown: list[Offer], last: bool) -> Offer | None:
        cards = [o.card(1600) for o in shown]
        user_msg = target.described() + "\n\nCANDIDATES:\n" + json.dumps(cards, ensure_ascii=False)
        chosen = self.hub.oracle.ask_json([QUICK_SEAT, SPARE_SEAT], PICK_PAGE_PROMPT, user_msg, 600)
        written = [o for o in shown if target.written(o)]
        if isinstance(chosen, dict) and chosen.get("id") in scratch.offers:
            return scratch.offers[chosen["id"]]
        if len(written) == 1:
            return written[0]
        if len(cards) == 1 and last:
            return scratch.offers[cards[0]["id"]]
        return None


class Coupler:
    """Wants that tie one requested product to another (fit rules, conditions)."""

    def __init__(self, hub: "Venture") -> None:
        self.hub = hub

    def goal_named(self, tool_name: Any) -> Goal | None:
        tool_name = str(tool_name or "")
        if not tool_name or not self.hub.goals:
            return None
        top_pick = max(self.hub.goals, key=lambda g: max(_overlap(g.label, tool_name), _overlap(g.shelf, tool_name)))
        return top_pick if max(_overlap(top_pick.label, tool_name), _overlap(top_pick.shelf, tool_name)) >= 0.2 else None

    def take_rules(self, raw_in: Any) -> list[dict[str, Any]]:
        result_ = []
        for third_ in raw_in if isinstance(raw_in, list) else []:
            rule_ = self._read_rule(third_)
            if rule_ is not None:
                result_.append(rule_)
                self._seed_branch_queries(rule_["branches"])
        if result_:
            self.hub.ledger.jot("conditions", rules=[(third_["src"].label, third_["prop"], third_["values"],
                                                      [(word, fourth_.label, second_.get("property"), second_.get("value"))
                                                       for word, fourth_, second_ in third_["branches"]]) for third_ in result_])
        return result_

    def rule_holds(self, rule_of: dict[str, Any], offer: Offer) -> bool | None:
        memo_box = self.hub.__dict__.setdefault("holds_memo", {})
        slot = (id(rule_of), offer.oid, json.dumps(offer.table, sort_keys=True, default=str))
        if slot in memo_box:
            return memo_box[slot]
        it_holds = self._holds_by_table(rule_of, offer)
        if it_holds is None:
            it_holds = self._holds_by_model(rule_of, offer)
        memo_box[slot] = it_holds
        self.hub.ledger.jot("condition", src=rule_of["src"].label, listing=offer.oid, prop=rule_of["prop"], holds=it_holds)
        return it_holds

    def apply_rules(self, goal: Goal, picked: dict[int, Offer]) -> None:
        my_rules = [third_ for third_ in self.hub.rules_of if any(fourth_ is goal for _, fourth_, _ in third_["branches"])]
        if not my_rules:
            return
        wanted_list = []
        for rule_of in my_rules:
            pick_rule = picked.get(rule_of["src"].slot_no)
            it_holds = self.rule_holds(rule_of, pick_rule) if pick_rule is not None else None
            if it_holds is None:
                continue
            for branch_holds, receiver, second_ in rule_of["branches"]:
                if receiver is goal and branch_holds == it_holds:
                    wanted_list.append(Want(second_["property"], second_.get("op") or "is", second_["value"], source="condition"))
        earlier = sorted(want.as_line() for want in goal.wants if want.source == "condition")
        if earlier == sorted(want.as_line() for want in wanted_list):
            return
        goal.wants = [want for want in goal.wants if want.source != "condition"] + wanted_list
        self.hub.ledger.jot("condition_applied", item=goal.label, reqs=[want.as_line() for want in wanted_list])
        self.hub.bench.assess(goal, [o for o in self.hub.scout.contenders(goal) if o.is_open])
        if not self.hub.bench.orderable(goal) and self.hub.ledger.moves_left() - self.hub.closing_steps() >= 3 and self.hub.ledger.seconds_left() > CLOSING_S + 20:
            goal.dry = False
            self.hub.hunt(goal, self.hub.ledger.moves_left() - self.hub.closing_steps(), last_round=True)

    def fit_couples(self) -> list[tuple[Goal, Goal]]:
        result_ = []
        for fit_rule in self.hub.fit_list:
            acc_name, host_offer = str(fit_rule.get("accessory") or ""), str(fit_rule.get("host") or "")
            if not acc_name or not host_offer or len(self.hub.goals) < 2:
                continue
            first_ = max(self.hub.goals, key=lambda g: _overlap(g.label, acc_name))
            head = max(self.hub.goals, key=lambda g: _overlap(g.label, host_offer))
            if first_ is not head and _overlap(first_.label, acc_name) >= 0.2 and _overlap(head.label, host_offer) >= 0.2 and (first_, head) not in result_:
                result_.append((first_, head))
        return result_

    def tie_fit(self, goal: Goal, host_goal: Goal, host_offer: Offer) -> None:
        self._fit_want(goal, host_goal, host_offer)
        wanted_list = self._spoken_codes(goal)
        series_codes = self._series_codes(host_goal, host_offer)
        self._look_codes(goal, series_codes, wanted_list)
        goal.series_ok = series_codes[:2]
        self._open_series(goal, series_codes)
        self.hub.ledger.jot("fit_bound", item=goal.label, host=host_offer.oid, series=series_codes[:2], looked_up=wanted_list[:3])
        self.hub.bench.assess(goal, [o for o in self.hub.scout.contenders(goal) if o.is_open])
        if series_codes:
            self._clear_fit_doubts(goal, series_codes)

    @staticmethod
    def of_series(offer: Offer, series_codes: list[str]) -> bool:
        return any(head in series_codes for head in _PART_SERIES.findall(offer.heading or ""))

    @staticmethod
    def _fit_want(goal: Goal, host_goal: Goal, host_offer: Offer) -> None:
        page_no = " ".join(elem for elem in [host_offer.heading[:120], (host_offer.blurb or "")[:300], (host_offer.remarks or "")[:150]] if elem)
        goal.wants = [want for want in goal.wants if want.source != "fit"]
        goal.wants.append(Want("compatibility", "is", f"works with the {host_goal.label} chosen, whose page reads: {page_no}",
                               source="fit"))

    def _spoken_codes(self, goal: Goal) -> list[str]:
        part_code = re.compile(r"\b[A-Z]{1,6}-?\d{3,}[A-Z0-9]*\b")
        title_codes = {third_ for g in self.hub.goals for o in g.offers.values() for third_ in part_code.findall(o.heading or "")}
        code_letters = {re.match(r"[A-Z]+", third_).group(0) for o in goal.offers.values() for third_ in part_code.findall(o.heading or "")}
        wanted_list: list[str] = []
        for o in self.hub.scout.contenders(goal):
            if not o.is_open or (goal.rulings.get(o.oid) or {}).get("failed"):
                continue
            for third_ in part_code.findall(" ".join([o.blurb or "", o.remarks or ""])):
                if third_ not in title_codes and third_ not in wanted_list and re.match(r"[A-Z]+", third_).group(0) in code_letters:
                    wanted_list.append(third_)
        return wanted_list

    def _series_codes(self, host_goal: Goal, host_offer: Offer) -> list[str]:
        host_words = " ".join([host_offer.blurb or "", host_offer.remarks or "", json.dumps(host_offer.table, ensure_ascii=False)])
        series_heads = {head for g in self.hub.goals if g is not host_goal for o in g.offers.values() for head in _PART_SERIES.findall(o.heading or "")}
        head_letters = {re.match(r"[A-Z]+", head).group(0) for head in series_heads}
        own_heads = _PART_SERIES.findall(host_offer.heading or "")
        series_codes: list[str] = []
        for third_ in re.findall(r"\b[A-Z]{1,5}\d{1,3}\b", host_words):
            if third_ in series_codes or third_ in own_heads:
                continue
            if third_ in series_heads or re.match(r"[A-Z]+", third_).group(0) in head_letters:
                series_codes.append(third_)
        return series_codes

    def _look_codes(self, goal: Goal, series_codes: list[str], wanted_list: list[str]) -> None:
        for third_ in series_codes[:2] + [word for word in wanted_list if word not in series_codes][:3]:
            if self.hub.ledger.moves_left() <= self.hub.closing_steps() + 2 or self.hub.ledger.seconds_left() < CLOSING_S + 10:
                break
            words = f"{third_} {goal.shelf}" if third_ in series_codes else third_
            self.hub.scout.look_up(goal, words=words, page_limit=1, keep_steps=self.hub.closing_steps())

    def _open_series(self, goal: Goal, series_codes: list[str]) -> None:
        if not series_codes or self.hub.ledger.moves_left() <= self.hub.closing_steps() + 2:
            return
        my_rules = [o for o in self.hub.scout.contenders(goal) if not o.is_open and self.of_series(o, series_codes)]
        if my_rules:
            self.hub.scout.open_offers(goal, [o.oid for o in my_rules[: self.hub.side_limit if self.hub.ledger.reads_many() else 3]],
                                       keep_free=self.hub.closing_steps() + 1)

    def _clear_fit_doubts(self, goal: Goal, series_codes: list[str]) -> None:
        fit_line = goal.wants[-1].as_line()
        for o in self.hub.scout.contenders(goal):
            val = goal.rulings.get(o.oid)
            if val and o.is_open and self.of_series(o, series_codes):
                for slot in ("unsure", "weak", "failed"):
                    if slot in val:
                        val[slot].discard(fit_line)

    def _read_rule(self, third_: Any) -> dict[str, Any] | None:
        rule_of = third_.get("if") if isinstance(third_, dict) else None
        if not isinstance(rule_of, dict) or not rule_of.get("property"):
            return None
        source_goal = self.goal_named(rule_of.get("item"))
        seen_values = rule_of.get("values")
        if isinstance(seen_values, list):
            seen_values = [str(val) for val in seen_values if val]
        else:
            seen_values = [str(seen_values)] if seen_values else []
        arms = []
        for slot, branch_holds in (("then", True), ("otherwise", False)):
            second_ = third_.get(slot)
            if isinstance(second_, dict) and second_.get("property") and second_.get("value"):
                receiver = self.goal_named(second_.get("item"))
                if receiver is not None and receiver is not source_goal:
                    arms.append((branch_holds, receiver, second_))
        if source_goal is None or not seen_values or not arms:
            return None
        return {"src": source_goal, "prop": str(rule_of["property"]), "values": seen_values, "branches": arms}

    @staticmethod
    def _seed_branch_queries(arms: list[tuple[bool, Goal, dict[str, Any]]]) -> None:
        for _, receiver, second_ in arms:
            for val in str(second_["value"]).split("|"):
                val = val.strip()
                if val and len(val) <= 40 and not any(_plain(val) in _plain(term_) for term_ in receiver.queries):
                    receiver.queries.append(f"{val} {receiver.head_noun}"[:60])

    @staticmethod
    def _holds_by_table(rule_of: dict[str, Any], offer: Offer) -> bool | None:
        own_values = [str(val) for key_, val in offer.table.items()
                      if _same_name(str(key_).replace("_", " "), rule_of["prop"]) or _overlap(str(key_).replace("_", " "), rule_of["prop"]) >= 0.5]
        if not own_values:
            return None
        phrase = _all_words(" ".join(own_values))
        want_sets = [_all_words(val) for val in rule_of["values"] if _all_words(val)]
        if any(word <= phrase for word in want_sets):
            return True
        if not any(word & phrase for word in want_sets):
            return False
        return None

    def _holds_by_model(self, rule_of: dict[str, Any], offer: Offer) -> bool | None:
        user_msg = (f"PROPERTY: {rule_of['prop']}\nVALUES: {json.dumps(rule_of['values'], ensure_ascii=False)}\n\nLISTING:\n"
                    + json.dumps(offer.card(2500), ensure_ascii=False))
        verdict_json = self.hub.oracle.ask_json([QUICK_SEAT, SPARE_SEAT], CONDITION_PROMPT, user_msg, 300)
        if isinstance(verdict_json, dict) and isinstance(verdict_json.get("holds"), bool):
            return verdict_json["holds"]
        return None


class _PickBoard:
    """One round of choosing: the pick per goal, the fit couples and condition rules that order the goals, and the
    swaps tried when a goal tied to another finds nothing."""

    def __init__(self, hub: "Venture") -> None:
        self.hub = hub
        self.picked: dict[int, Offer] = {}
        self.couples = hub.coupler.fit_couples()
        self.hosts = {head.slot_no for _, head in self.couples}
        self.host_of = {first_.slot_no: head for first_, head in self.couples}
        self.rules = getattr(hub, "rules_of", [])
        self.sources = {rule["src"].slot_no for rule in self.rules}
        self.waiting = {dst.slot_no for rule in self.rules for _, dst, _ in rule["branches"]}

    def check_leads(self) -> None:
        """The first pick of every goal that waits for no other goal, checked side by side."""
        bench = self.hub.bench
        leads = [(g, (bench.by_measure(g, bench.orderable(g)) or [None])[0]) for g in self.hub.goals
                 if g.slot_no not in self.host_of and g.slot_no not in self.waiting]
        _side_by_side([lambda g=g, o=o: bench.check(g, o) for g, o in leads if o is not None], self.hub.meter)

    def order(self) -> list[Goal]:
        return sorted(self.hub.goals, key=lambda g: (g.slot_no in self.host_of or g.slot_no in self.waiting,
                                                     g.slot_no not in self.hosts and g.slot_no not in self.sources, g.slot_no))

    def choose(self, goal: Goal) -> None:
        hub, picked = self.hub, self.picked
        if goal.slot_no in self.host_of and self.host_of[goal.slot_no].slot_no in picked:
            hub.coupler.tie_fit(goal, self.host_of[goal.slot_no], picked[self.host_of[goal.slot_no].slot_no])
        if goal.slot_no in self.waiting:
            hub.coupler.apply_rules(goal, picked)
        attempts = 0
        for contender in hub.bench.by_measure(goal, hub.bench.orderable(goal)):
            if attempts >= 3 or hub.ledger.seconds_left() < CLOSING_S:
                break
            attempts += 1
            if hub.bench.check(goal, contender):
                picked[goal.slot_no] = contender
                break
        if goal.slot_no not in picked:
            rest = hub.bench.by_measure(goal, hub.bench.orderable(goal))
            if rest and (attempts >= 3 or hub.ledger.seconds_left() < CLOSING_S):
                picked[goal.slot_no] = rest[0]
        if goal.slot_no in self.host_of and goal.slot_no not in picked:
            self._other_host(goal)
        if goal.slot_no in self.waiting and goal.slot_no not in picked:
            self._other_outcome(goal)

    def _other_host(self, goal: Goal) -> None:
        """No accessory fits the host chosen: another host that some accessory fits."""
        hub, picked = self.hub, self.picked
        host = self.host_of[goal.slot_no]
        for alt in hub.bench.orderable(host)[:3]:
            if picked.get(host.slot_no) is alt or hub.ledger.seconds_left() < CLOSING_S:
                continue
            hub.coupler.tie_fit(goal, host, alt)
            got = hub.bench.orderable(goal)
            if got:
                picked[host.slot_no] = alt
                picked[goal.slot_no] = got[0]
                return

    def _other_outcome(self, goal: Goal) -> None:
        """Nothing fits what a condition asks for: another pick of its source with the other outcome, unless an
        accessory already chosen must fit that source."""
        hub, picked = self.hub, self.picked
        for rule in [r for r in self.rules if r["src"].slot_no in picked and any(dst is goal for _, dst, _ in r["branches"])]:
            src = rule["src"]
            if any(head is src and first_.slot_no in picked for first_, head in self.couples) or hub.ledger.seconds_left() < CLOSING_S:
                continue
            now = hub.coupler.rule_holds(rule, picked[src.slot_no])
            for alt in hub.bench.orderable(src)[:4]:
                if alt is picked[src.slot_no] or hub.coupler.rule_holds(rule, alt) == now:
                    continue
                before = picked[src.slot_no]
                picked[src.slot_no] = alt
                hub.coupler.apply_rules(goal, picked)
                got = hub.bench.orderable(goal)
                if got and hub.bench.check(src, alt):
                    picked[goal.slot_no] = got[0]
                    break
                picked[src.slot_no] = before
                hub.coupler.apply_rules(goal, picked)
            if goal.slot_no in picked:
                return


class Closer:
    """The choice, the basket, the order and the decline."""

    def __init__(self, hub: "Venture") -> None:
        self.hub = hub

    def decide_and_close(self) -> None:
        hub = self.hub
        hub.liaison.read_news()
        hub.bench.assess_all()
        board = _PickBoard(hub)
        board.check_leads()
        for goal in board.order():
            board.choose(goal)
        lacking = [g for g in hub.goals if g.slot_no not in board.picked]
        if lacking and self._second_chance(lacking):
            return
        if lacking:
            hub.decline_why = "no fitting listing for " + ", ".join(g.label for g in lacking)
            hub.ledger.jot("no_fit", items=[g.label for g in lacking], pools={g.label: len(g.offers) for g in lacking})
            self.send_decline(hub.decline_why)
            return
        if hub.ceiling is not None and not self._within_budget(board.picked):
            return
        for goal in hub.goals:
            goal.pick = board.picked[goal.slot_no]
        self.order_round()

    def _second_chance(self, lacking: list[Goal]) -> bool:
        """Goals with nothing to order: one more search with the ending's steps cut to the bone, then the last look;
        True when either led to a new decision (already made)."""
        hub = self.hub
        roomy = hub.ledger.seconds_left() > CLOSING_S + 20 and not hub.ledger.out_of_money()
        if not getattr(hub, "retried", False) and roomy:
            hub.retried = True
            hub.reserve_forced = len(hub.goals) + 3 + max(hub.liaison.kept_messages(), 1 if hub.show_basket_first else 0)
            try:
                for g in lacking:
                    if hub.ledger.moves_left() - hub.closing_steps() >= 3:
                        g.dry = False
                        hub.hunt(g, hub.ledger.moves_left() - hub.closing_steps(), last_round=True)
            finally:
                hub.reserve_forced = None
            if any(hub.bench.orderable(g) for g in lacking):
                self.decide_and_close()
                return True
        if not getattr(hub, "looked_again", False) and hub.ledger.seconds_left() > CLOSING_S + 25 and not hub.ledger.out_of_money():
            hub.looked_again = True
            if self.last_look(lacking) and all(hub.bench.orderable(g) for g in lacking):
                self.decide_and_close()
                return True
        return False

    def _within_budget(self, picked: dict[int, Offer]) -> bool:
        """Cheaper fits stepped in, dearest line first, until the basket fits the ceiling; False (after declining)
        when even that leaves it over."""
        hub = self.hub

        def total() -> float:
            return sum((picked[g.slot_no].unit_price or 0) * g.units for g in hub.goals)
        for _ in range(10):
            if total() <= hub.ceiling + 1e-6:
                break
            if not self._cheaper_step(picked):
                break
        if total() > hub.ceiling + 1e-6:
            hub.decline_why = "the cheapest fitting combination is over the budget"
            hub.ledger.jot("over_budget", total=total(), cap=hub.ceiling)
            self.send_decline(hub.decline_why)
            return False
        return True

    def _cheaper_step(self, picked: dict[int, Offer]) -> bool:
        """The dearest line that has a cheaper fit gives way: to the best-ranked fit that brings the basket under the
        ceiling, else to its cheapest cheaper fit."""
        hub = self.hub
        total = sum((picked[g.slot_no].unit_price or 0) * g.units for g in hub.goals)
        for g in sorted(hub.goals, key=lambda x: -((picked[x.slot_no].unit_price or 0) * x.units)):
            now = picked[g.slot_no].unit_price if picked[g.slot_no].unit_price is not None else 1e18
            others = total - (picked[g.slot_no].unit_price or 0) * g.units
            ranked = [o for o in hub.bench.by_measure(g, hub.bench.orderable(g)) if o.unit_price is not None and o.unit_price < now - 1e-9]
            if not ranked:
                doubtful = self._doubtful_fit(g, lambda o: others + o.unit_price * g.units <= hub.ceiling + 1e-6)
                if doubtful is None:
                    continue
                picked[g.slot_no] = doubtful
                return True
            fits = [o for o in ranked if others + o.unit_price * g.units <= hub.ceiling + 1e-6]
            picked[g.slot_no] = fits[0] if fits else min(ranked, key=lambda o: o.unit_price)
            return True
        return False

    def _doubtful_fit(self, g: Goal, affordable: Any) -> Offer | None:
        """A listing with only unsure requirements that the budget allows and the final check accepts (two tries)."""
        want_lines = set(g.want_lines())
        doubtful = [o for o in g.offers.values() if o.is_open and o.can_buy is not False and o.unit_price is not None
                    and g.rulings.get(o.oid) and not g.rulings[o.oid]["failed"] and g.judged_against.get(o.oid, set()) >= want_lines
                    and affordable(o)]
        doubtful.sort(key=lambda o: (len(g.rulings[o.oid]["unsure"]), o.unit_price))
        for o in self.hub.bench.by_measure(g, doubtful)[:2]:
            if self.hub.bench.check(g, o):
                self.hub.ledger.jot("doubtful_fit", item=g.label, id=o.oid)
                return o
        return None

    def last_look(self, lacking: list[Goal]) -> bool:
        learned = False
        self.hub.reserve_forced = len(self.hub.goals) + 3 + max(self.hub.liaison.kept_messages(), 1 if self.hub.show_basket_first else 0)
        try:
            for goal in lacking:
                if self.hub.ledger.seconds_left() < CLOSING_S + 20:
                    break
                learned = self._review_goal(goal) or learned
        finally:
            self.hub.reserve_forced = None
        return learned

    def reread_basket(self) -> None:
        for oid, goal in list(self.hub.basket_goal.items()):
            if self.hub.ledger.moves_left() <= 2:
                return
            opening = _Opening(self.hub.scout, goal, [oid], 0)
            if opening.page_tool:
                # (one page read straight away: the closing reads past the opening loop's time guard)
                opening._read_page(oid)
            else:
                self.hub.scout.open_offers(goal, [oid])
            self.hub.ledger.jot("refreshed", id=oid, price=goal.offers[oid].unit_price if oid in goal.offers else None,
                      stock=goal.offers[oid].can_buy if oid in goal.offers else None)

    def replace_basket(self, said_lines: dict[str, int]) -> bool:
        role_spec = self.hub.duties.get("cart_set") or {}
        fields_of = self.hub.brief._shape_of("cart_set").get("properties") or {}
        line_fields = ((fields_of.get(role_spec.get("lines") or "") or {}).get("items") or {}).get("properties") or {}
        top_qty = (line_fields.get(role_spec.get("qty") or "") or {}).get("maximum")
        row_list_ = []
        for oid, units in said_lines.items():
            row_ = {role_spec["id"]: oid}
            if role_spec.get("qty"):
                row_[role_spec["qty"]] = min(units, top_qty) if isinstance(top_qty, int) else units
            row_list_.append(row_)
        outcome = self.hub.ledger.act(role_spec["name"], {role_spec["lines"]: row_list_})
        if not outcome["ok"]:
            self.hub.ledger.jot("cart_set_failed", error=outcome["error"], obs=_short_json(outcome["obs"], 200))
        return outcome["ok"]

    def put_in(self, goal: Goal, offer: Offer) -> bool:
        if not self.hub.duties.get("cart_add") and self.hub.duties.get("cart_set"):
            return self._set_whole_basket(goal, offer)
        role_spec = self.hub.duties.get("cart_add")
        return self._add_line(goal, offer, role_spec) if role_spec else False

    def take_out(self, offer_id: str) -> bool:
        if not self.hub.duties.get("cart_add") and self.hub.duties.get("cart_set"):
            return self._drop_from_whole_basket(offer_id)
        role_spec = self.hub.duties.get("cart_add")
        return self._remove_line(offer_id, role_spec) if role_spec else False

    def send_order(self) -> bool:
        role_spec = self.hub.duties.get("checkout")
        if not role_spec:
            return False
        if self.hub.process and self.hub.basket:
            acted = self.hub.liaison.process_check("before_order", "check out the basket below as the order")
            if acted and self.hub.liaison.owed_messages() and self.hub.ledger.moves_left() >= 2:
                acted = self.hub.liaison.process_check("before_order", "check out the basket below as the order") or acted
            if acted is not False and self.hub.show_basket_first and self.hub.liaison.basket_key() != self.hub.shown_basket:
                acted = self.hub.liaison.show_basket() or acted
            if acted and (self.hub.liaison.read_news() or self.basket_broken()):
                self.hub.ledger.jot("procedure_changed")
                return False
        outcome = self.hub.ledger.act(role_spec["name"], self.hub.brief.ending_argvals("checkout"))
        if self.hub.over:
            self.hub.ledger.jot("ordered", cart=self.hub.basket)
            return True
        self.hub.ledger.jot("order_refused", obs=_short_json(outcome["obs"], 200), lines=outcome["lines"][:3])
        if outcome["lines"]:
            # (the shopper spoke up at checkout: the total told before may be stale, so it is told again)
            self.hub.shown_basket = None
        return False

    def send_decline(self, why: str = "") -> bool:
        role_spec = self.hub.duties.get("decline")
        if not role_spec:
            return False
        if self.hub.process and not getattr(self.hub, "decline_checked", False):
            self.hub.decline_checked = True
            self.hub.liaison.process_check("before_decline", "end the task without buying anything")
        argvals: dict[str, Any] = self.hub.brief.ending_argvals("decline")
        remark_arg = role_spec.get("remark")
        if remark_arg and remark_arg in (self.hub.brief._shape_of("decline").get("required") or []):
            argvals[remark_arg] = self.hub.brief._cut("decline", remark_arg, "Nothing available meets every requirement of the request.")
        outcome = self.hub.ledger.act(role_spec["name"], argvals)
        self.hub.ledger.jot("declined", reason=why or self.hub.decline_why)
        if not self.hub.over and not getattr(self.hub, "decline_bounced", False) and self.hub.ledger.moves_left() >= 1:
            self.hub.decline_bounced = True
            self.hub.ledger.jot("decline_refused", obs=_short_json(outcome["obs"], 200), lines=outcome["lines"][:3])
            if self.hub.liaison.read_news() and self.hub.ledger.moves_left() >= len(self.hub.goals) + 2:
                self.decide_and_close()
            if not self.hub.over:
                self.hub.ledger.act(role_spec["name"], argvals)
        return self.hub.over

    def order_round(self, round_no: int = 0) -> None:
        while self._order_pass(round_no):
            round_no += 1

    def basket_broken(self) -> bool:
        for goal in self.hub.goals:
            o = goal.pick
            if o is None:
                continue
            if o.can_buy is False:
                return True
            if self.hub.ceiling is not None and o.unit_price is not None and sum((g.pick.unit_price or 0) * g.units for g in self.hub.goals if g.pick) > self.hub.ceiling + 1e-6:
                return True
        return False

    def pick_again(self) -> None:
        for goal in self.hub.goals:
            self._refresh_hunt(goal)
        for goal in self.hub.goals:
            self._settle_pick(goal)
        self._couple_picks()
        if any(g.pick is None for g in self.hub.goals):
            self.decide_and_close()
            return
        if self.hub.ceiling is not None and sum((g.pick.unit_price or 0) * g.units for g in self.hub.goals) > self.hub.ceiling + 1e-6:
            self.decide_and_close()

    def close_now(self) -> None:
        if self.hub.over:
            return
        for goal in self.hub.goals:
            if goal.pick is None or goal.pick.can_buy is False:
                good = self.hub.bench.orderable(goal)
                goal.pick = good[0] if good else None
        can_order = all(g.pick for g in self.hub.goals) and bool(self.hub.goals)
        if can_order and self.hub.ceiling is not None:
            self._fit_picks_to_ceiling()
            can_order = sum((g.pick.unit_price or 0) * g.units for g in self.hub.goals) <= self.hub.ceiling + 1e-6
        if can_order:
            stale_ids = [oid for oid in self.hub.basket if not any(g.pick.oid == oid for g in self.hub.goals)]
            to_put = [g for g in self.hub.goals if g.pick.oid not in self.hub.basket]
            if self.hub.ledger.moves_left() >= len(stale_ids) + len(to_put) + 1:
                for oid in stale_ids:
                    self.take_out(oid)
                for goal in to_put:
                    self.put_in(goal, goal.pick)
                if self.send_order():
                    return
        if not self.hub.over:
            self.send_decline("forced")

    def _fit_picks_to_ceiling(self) -> None:
        """Picks over a ceiling (one lowered late) give way, the dearest line first, to the best orderable listing that fits."""
        for goal in sorted(self.hub.goals, key=lambda g: -((g.pick.unit_price or 0) * g.units)):
            total = sum((g.pick.unit_price or 0) * g.units for g in self.hub.goals)
            if total <= self.hub.ceiling + 1e-6:
                return
            others = total - (goal.pick.unit_price or 0) * goal.units
            fits = [o for o in self.hub.bench.orderable(goal)
                    if o.unit_price is not None and others + o.unit_price * goal.units <= self.hub.ceiling + 1e-6]
            if fits:
                self.hub.ledger.jot("ceiling_swap", item=goal.label, old=goal.pick.oid, new=fits[0].oid)
                goal.pick = fits[0]

    def _within_cap(self, goal: Goal, o: Offer) -> bool:
        return not (self.hub.ceiling is not None and o.unit_price is not None and o.unit_price * goal.units > self.hub.ceiling + 1e-6)

    @staticmethod
    def _offer_summary(goal: Goal, o: Offer) -> dict[str, Any]:
        summary: dict[str, Any] = {"id": o.oid, "title": o.heading[:110], "price": o.unit_price, "read": o.is_open}
        if not o.is_open:
            summary["text"] = (o.blurb or "")[:160]
            return summary
        val = goal.rulings.get(o.oid) or {}
        summary["failed"] = sorted(val.get("failed") or [])[:3]
        summary["unsure"] = sorted(val.get("unsure") or [])[:3]
        summary["properties"] = dict(list(o.table.items())[:16])
        if o.remarks:
            summary["notes"] = o.remarks[:200]
        return summary

    def _review_goal(self, goal: Goal) -> bool:
        seen_pool = [o for o in goal.offers.values() if self.hub.scout.on_shelf(goal, o) and o.can_buy is not False and self._within_cap(goal, o)]
        ordered_pool = sorted(seen_pool, key=lambda o: (not o.is_open, o.unit_price if o.unit_price is not None else 1e18))[:40]
        summaries = [self._offer_summary(goal, o) for o in ordered_pool]
        if not summaries:
            return False
        tried_queries = sorted({key_.split(":", 1)[1] for key_ in goal.places if key_.startswith("search_text:")})
        cap_text = self.hub.ceiling if self.hub.ceiling is not None else "none stated"
        user_msg = (f"PRODUCT: {goal.label} (category: {goal.shelf}; quantity {goal.units})\nBUDGET, ALL LINES TOGETHER: "
                    f"{cap_text}\nREQUIREMENTS:\n- " + ("\n- ".join(goal.want_lines()) or "none")
                    + f"\nQUERIES TRIED: {json.dumps(tried_queries, ensure_ascii=False)}\n\nLISTINGS SEEN:\n" + _short_json(summaries, 14000))
        verdict_json = self.hub.oracle.ask_json([PLANNER_SEAT, QUICK_SEAT], LAST_LOOK_PROMPT, user_msg, 1500)
        if not isinstance(verdict_json, dict):
            return False
        self.hub.ledger.jot("audit", item=goal.label, answer=_short_json(verdict_json, 300))
        learned = self._accept_late(goal, verdict_json.get("fits_after_all"))
        learned = self._open_named(goal, verdict_json.get("unopened")) or learned
        learned = self._search_more(goal, verdict_json.get("queries"), tried_queries) or learned
        self.hub.bench.assess(goal, [o for o in self.hub.scout.contenders(goal) if o.is_open])
        return learned

    def _accept_late(self, goal: Goal, named: Any) -> bool:
        learned = False
        for oid in [str(elem) for elem in named or []][:2]:
            o = goal.offers.get(oid)
            if o is None or not o.is_open or self.hub.bench.field_misses(goal, o):
                continue
            if self.hub.bench._check(goal, o, must_be_sure=True):
                goal.rulings[o.oid] = {"failed": set(), "unsure": set(), "weak": set(), "loose": set()}
                goal.judged_against[o.oid] = set(goal.want_lines())
                learned = True
        return learned

    def _open_named(self, goal: Goal, named: Any) -> bool:
        unread = [str(elem) for elem in named or [] if str(elem) in goal.offers and not goal.offers[str(elem)].is_open][:4]
        if not unread or self.hub.ledger.moves_left() - self.hub.closing_steps() < 1:
            return False
        return bool(self.hub.scout.open_offers(goal, unread, keep_free=self.hub.closing_steps()))

    def _search_more(self, goal: Goal, named: Any, tried_queries: list[str]) -> bool:
        learned = False
        tried = {_plain(term_) for term_ in tried_queries}
        for query_ in [str(elem).strip() for elem in named or [] if str(elem).strip()][:2]:
            if self.hub.ledger.moves_left() - self.hub.closing_steps() < 3 or self.hub.ledger.seconds_left() < CLOSING_S + 15:
                break
            if _plain(query_) in tried or self.hub.scout.look_up(goal, words=query_[:60], page_limit=1, keep_steps=self.hub.closing_steps()) <= 0:
                continue
            self.hub.scout.screen_rows(goal, [o for o in self.hub.scout.contenders(goal) if not o.is_open])
            row_list_ = [o for o in self.hub.scout.contenders(goal) if not o.is_open and o.row_verdict != "no"][:2]
            if row_list_ and self.hub.scout.open_offers(goal, [o.oid for o in row_list_], keep_free=self.hub.closing_steps()):
                learned = True
        return learned

    def _set_whole_basket(self, goal: Goal, offer: Offer) -> bool:
        said_lines = dict(self.hub.basket)
        said_lines[offer.oid] = goal.units
        if not self.replace_basket(said_lines):
            return False
        self.hub.basket = said_lines
        self.hub.basket_goal[offer.oid] = goal
        return True

    def _add_units(self, role_spec: dict[str, Any], units: int) -> int:
        fields_of = self.hub.brief._shape_of("cart_add").get("properties") or {}
        top_qty = (fields_of.get(role_spec["qty"]) or {}).get("maximum")
        return top_qty if isinstance(top_qty, int) and units > top_qty else units

    def _add_line(self, goal: Goal, offer: Offer, role_spec: dict[str, Any]) -> bool:
        argvals: dict[str, Any] = {role_spec["id"]: offer.oid}
        units = goal.units
        if role_spec.get("qty"):
            units = self._add_units(role_spec, units)
            argvals[role_spec["qty"]] = units
        switch = role_spec.get("mode") if isinstance(role_spec.get("mode"), dict) else None
        if switch and switch.get("arg") and switch.get("add") is not None:
            argvals[switch["arg"]] = switch["add"]
        outcome = self.hub.ledger.act(role_spec["name"], argvals)
        if not outcome["ok"]:
            self.hub.ledger.jot("cart_add_failed", error=outcome["error"], obs=_short_json(outcome["obs"], 200))
            return False
        self.hub.basket[offer.oid] = units
        self.hub.basket_goal[offer.oid] = goal
        return True

    def _sync_basket(self) -> None:
        for oid in list(self.hub.basket):
            if not any(g.pick and g.pick.oid == oid for g in self.hub.goals):
                self.take_out(oid)
        for goal in self.hub.goals:
            if goal.pick and goal.pick.oid not in self.hub.basket and not self.put_in(goal, goal.pick):
                goal.rulings.setdefault(goal.pick.oid, {"failed": set(), "unsure": set()})["failed"].add("cart refused")
                goal.pick = None

    def _recheck_picks(self) -> bool:
        moved_on = False
        for goal in self.hub.goals:
            if goal.pick is None or goal.pick.oid in goal.judged_against:
                continue
            self.hub.bench.assess(goal, [goal.pick])
            if (goal.rulings.get(goal.pick.oid) or {}).get("failed"):
                self.hub.ledger.jot("chosen_failed", id=goal.pick.oid, failed=sorted(goal.rulings[goal.pick.oid]["failed"])[:3])
                moved_on = True
        return moved_on

    def _order_pass(self, round_no: int) -> bool:
        self.hub.order_tries = getattr(self.hub, "order_tries", 0) + 1
        if round_no > 2 or self.hub.over or self.hub.order_tries > 8:
            return False
        self._sync_basket()
        if any(g.pick is None for g in self.hub.goals):
            self.decide_and_close()
            return False
        if self.hub.ledger.moves_left() > 2 + self.hub.liaison.owed_messages() and self.hub.goals and self.hub.goals[0].pick:
            self.hub.scout.open_offers(self.hub.goals[0], [self.hub.goals[0].pick.oid])
        moved_on = self.hub.liaison.read_news()
        moved_on = self._recheck_picks() or moved_on
        if moved_on or self.basket_broken():
            self.hub.ledger.jot("rechoose", cart=self.hub.basket)
            for goal in self.hub.goals:
                self.hub.bench.assess(goal, [o for o in self.hub.scout.contenders(goal) if o.is_open])
            self.pick_again()
            return not self.hub.over
        if self.send_order():
            return False
        if not self.hub.liaison.read_news():
            self.reread_basket()
            self.hub.liaison.read_news()
        self.pick_again()
        return not self.hub.over

    def _refresh_hunt(self, goal: Goal) -> None:
        self.hub.bench.assess(goal, [o for o in self.hub.scout.contenders(goal) if o.is_open])
        if [fields_ for fields_ in goal.good_offers() if fields_.can_buy is not False] and not self.hub.lowest:
            return
        goal.told_searched = False
        goal.swept_under = None
        step_share = max(2, self.hub.ledger.moves_left() - self.hub.closing_steps())
        if step_share >= 3 and self.hub.ledger.seconds_left() > CLOSING_S + 20:
            self.hub.hunt(goal, step_share, last_round=True)

    def _settle_pick(self, goal: Goal) -> None:
        good = [fields_ for fields_ in goal.good_offers(lowest=self.hub.lowest) if fields_.can_buy is not False]
        if not good:
            goal.pick = None
        elif self.hub.lowest or goal.pick is None or goal.pick.oid not in [fields_.oid for fields_ in good]:
            goal.pick = good[0]

    def _couple_picks(self) -> None:
        picked_now = {g.slot_no: g.pick for g in self.hub.goals if g.pick is not None}
        receivers = {fourth_.slot_no: fourth_ for third_ in getattr(self.hub, "rules_of", []) for _, fourth_, _ in third_["branches"]}
        for receiver in receivers.values():
            self.hub.coupler.apply_rules(receiver, picked_now)
            worked = [fields_.oid for fields_ in self.hub.bench.orderable(receiver)]
            if receiver.pick is not None and receiver.pick.oid not in worked:
                receiver.pick = receiver.offers[worked[0]] if worked else None

    def _drop_from_whole_basket(self, offer_id: str) -> bool:
        said_lines = {key_: val for key_, val in self.hub.basket.items() if key_ != offer_id}
        if not self.replace_basket(said_lines):
            return False
        self.hub.basket = said_lines
        self.hub.basket_goal.pop(offer_id, None)
        return True

    def _removal_args(self, offer_id: str, role_spec: dict[str, Any]) -> dict[str, Any]:
        argvals: dict[str, Any] = {role_spec["id"]: offer_id}
        switch = role_spec.get("mode") if isinstance(role_spec.get("mode"), dict) else None
        if switch and switch.get("arg") and switch.get("remove") is not None:
            argvals[switch["arg"]] = switch["remove"]
            if role_spec.get("qty"):
                argvals[role_spec["qty"]] = self.hub.basket.get(offer_id, 1)
        elif role_spec.get("qty"):
            fields_of = self.hub.brief._shape_of("cart_add").get("properties") or {}
            low_qty = (fields_of.get(role_spec["qty"]) or {}).get("minimum")
            argvals[role_spec["qty"]] = low_qty if isinstance(low_qty, int) and low_qty > 0 else 0
        return argvals

    def _remove_line(self, offer_id: str, role_spec: dict[str, Any]) -> bool:
        outcome = self.hub.ledger.act(role_spec["name"], self._removal_args(offer_id, role_spec))
        if not outcome["ok"]:
            return False
        self.hub.basket.pop(offer_id, None)
        self.hub.basket_goal.pop(offer_id, None)
        return True


class _Hunt:
    """One goal's search-and-open loop within a step allowance: each round tries the moves in a fixed order and
    the first that applies takes the round ("again"), or ends the hunt ("stop")."""

    def __init__(self, hub: "Venture", goal: Goal, step_allow: int) -> None:
        self.hub, self.goal = hub, goal
        self.stop_line = hub.ledger.moves_left() - step_allow
        self.terms = list(goal.queries)
        self.term_at = min(goal.searches_run, len(self.terms))
        self.fits_wanted = 2 if not (hub.lowest or hub.measure) else 3 if hub.lowest else 4
        self.closed: list[Offer] = []
        self.good: list[Offer] = []

    def floor(self) -> int:
        return max(self.stop_line, self.hub.closing_steps())

    def go(self) -> None:
        hub = self.hub
        moves = (self._enough, self._lowest_moves, self._told_search, self._open_rows, self._next_term,
                 self._field_search, self._page_on, self._ask_queries)
        while hub.ledger.moves_left() > self.floor() and hub.ledger.seconds_left() > CLOSING_S + 10 and not hub.ledger.out_of_money():
            self._look()
            outcome = next((o for o in (move() for move in moves) if o is not None), "dry")
            if outcome == "dry":
                self.goal.dry = True
                return
            if outcome == "stop":
                return

    def _look(self) -> None:
        hub, goal = self.hub, self.goal
        hub.liaison.read_news()
        rows = hub.scout.contenders(goal)
        hub.scout.screen_rows(goal, rows)
        rows = hub.scout.contenders(goal)
        self.closed = [o for o in rows if not o.is_open]
        hub.bench.assess(goal, [o for o in rows if o.is_open])
        self.good = goal.good_offers()

    def _enough(self) -> str | None:
        if self.good and not self.hub.lowest and len(self.good) >= self.fits_wanted:
            if self.hub.measure and self.closed:
                # (a ranking rule needs the whole screened pool: unopened rows are read while the steps last)
                return None
            return "stop"
        return None

    def _lowest_moves(self) -> str | None:
        """For the cheapest rule every unopened row cheaper than the best fit must be seen."""
        if not (self.good and self.hub.lowest):
            return None
        best = self.good[0].unit_price if self.good[0].unit_price is not None else 1e18
        self.closed = [o for o in self.closed if (o.unit_price is not None and o.unit_price < best)]
        if self.closed:
            return None
        roomy = self.hub.ledger.moves_left() > self.floor() + 2
        move = self._sweep_below(best, roomy) or self._values_pass(roomy) or self._lead_pass()
        if move:
            return move
        if self.goal.searches_run >= min(len(self.terms), 2):
            return "stop"
        return None

    def _open_rows(self) -> str | None:
        hub = self.hub
        if not self.closed:
            return None
        group = self.closed[: hub.side_limit if hub.ledger.reads_many() else 4 if hub.side_lacks_table else 2]
        if not hub.scout.open_offers(self.goal, [o.oid for o in group], keep_free=self.floor()):
            return "stop"
        return "again"

    def _told_search(self) -> str | None:
        """Needs the shopper told, searched with the stated values (they narrow the search well)."""
        hub, goal = self.hub, self.goal
        told_wants = [w for w in goal.live_wants() if w.source == "shopper" and w.amount and w.relation in ("is", "one of")]
        told = [w.amount.split("|")[0].strip() for w in told_wants]
        phrased = [w.search for w in told_wants if w.search]
        stated = [w.amount.split("|")[0].strip() for w in goal.live_wants() if w.source != "shopper" and w.amount and w.relation in ("is", "one of")]
        if not told or getattr(goal, "told_searched", None) == tuple(told):
            return None
        goal.told_searched = tuple(told)
        newest = told[-1]
        for query_ in ([f"{stated[0]} {newest}"] if stated else []) + phrased[-1:] + [f"{goal.shelf} {newest}", newest]:
            if hub.ledger.moves_left() <= self.floor():
                break
            if hub.scout.look_up(goal, words=query_, page_limit=2, keep_steps=self.floor()) > 0:
                break
        return "again"

    def _next_term(self) -> str | None:
        hub = self.hub
        if self.term_at >= len(self.terms):
            return None
        words = self.terms[self.term_at]
        self.term_at += 1
        pages = 6 if len(hub.goals) == 1 else 3
        added = hub.scout.look_up(self.goal, words=words, page_limit=pages, keep_steps=self.floor())
        if added == 0 and self.term_at >= len(self.terms):
            self.goal.dry = True
        return "again"

    def _field_search(self) -> str | None:
        hub, goal = self.hub, self.goal
        if not hub.duties.get("search_fields"):
            return None
        filters = {"category": hub.scout.store_label(goal)}
        if hub.ceiling is not None:
            filters["price_max"] = hub.ceiling / max(goal.units, 1)
        if (goal.places.get("search_fields:" + json.dumps(filters, sort_keys=True)) or {}).get("done"):
            return None
        hub.scout.look_up(goal, filters=filters, page_limit=2, keep_steps=self.floor())
        return "again"

    def _page_on(self) -> str | None:
        """The most specific keyword queries paged on while steps remain."""
        for query_ in self.terms[:2]:
            if not (self.goal.places.get("search_text:" + query_) or {}).get("done"):
                if self.hub.scout.look_up(self.goal, words=query_, page_limit=2, keep_steps=self.floor()) > 0:
                    return "again"
        return None

    def _ask_queries(self) -> str | None:
        hub, goal = self.hub, self.goal
        if getattr(goal, "asked_for_queries", False) or hub.ledger.moves_left() <= self.floor() + 1:
            return None
        goal.asked_for_queries = True
        more = hub.more_queries(goal)
        if not more:
            return None
        goal.queries += more
        self.terms += more
        return "again"

    def _sweep_below(self, best: float, roomy: bool) -> str | None:
        hub, goal = self.hub, self.goal
        price_field = hub.duties.get("search_fields") and hub.duties["search_fields"].get("price_max")
        if not (best < 1e17 and price_field and getattr(goal, "swept_under", None) != best and roomy):
            return None
        goal.swept_under = best
        hub.scout.look_up(goal, filters={"category": hub.scout.store_label(goal), "price_max": max(0.0, best - 0.01)}, page_limit=3, keep_steps=self.floor())
        return "again"

    def _values_pass(self, roomy: bool) -> str | None:
        hub, goal = self.hub, self.goal
        if not (getattr(goal, "hidden_more", False) and not getattr(goal, "values_tried", False) and roomy):
            return None
        goal.values_tried = True
        values = [w.amount.split("|")[0].strip() for w in goal.live_wants() if w.relation in ("is", "one of") and w.amount]
        if not values:
            return None
        for query_ in (" ".join(values + [goal.shelf]), " ".join(values)):
            if hub.ledger.moves_left() <= self.floor() + 1:
                break
            hub.scout.look_up(goal, words=query_, page_limit=2, keep_steps=self.floor())
        return "again"

    def _lead_pass(self) -> str | None:
        hub, goal = self.hub, self.goal
        lead = next((q for q in self.terms if goal.places.get("search_text:" + q)
                     and not goal.places["search_text:" + q].get("done")), None)
        if not lead or hub.ledger.moves_left() <= self.floor() + 2:
            return None
        was_at = hub.steps_used
        hub.scout.look_up(goal, words=lead, page_limit=1, keep_steps=self.floor() + 1)
        return "again" if hub.steps_used > was_at else None


class Venture:
    """One task from start to finish."""

    def __init__(self, task_data: dict[str, Any]):
        self._seat_components()
        self.meter = CountedClock()
        self._read_view(task_data["environment"])
        self.oracle = Oracle(self)
        self._blank_plan()
        self._recall_store()

    def go(self) -> list[dict[str, Any]]:
        try:
            if not self.brief.read_request():
                self.closer.close_now()
                return self.log_
            self.shop()
        except Exception as fault:
            self.ledger.jot("exception", error=repr(fault)[:300])
        try:
            if not self.over:
                self.closer.close_now()
        except Exception as fault:
            self.ledger.jot("exception", error=repr(fault)[:300])
        self.ledger.jot("done", cost=round(self.oracle.spent_usd, 4), llm_calls=self.oracle.requests, steps=self.steps_used)
        return self.log_

    def closing_steps(self) -> int:
        if getattr(self, "reserve_forced", None) is not None:
            return self.reserve_forced
        return (len(self.goals) + 6 + (2 if self.duties.get("facts") and not self.duties.get("compare") else 0)
                + min(4, sum(1 for step_spec in self.process if step_spec["moment"] != "start") + (1 if self.show_basket_first else 0)))

    def shop(self) -> None:
        slots = (self.message_cap if self.message_cap is not None else 3) - self.liaison.kept_messages()
        if "message" not in self.duties or slots < 0:
            slots = 0
        self._chains(slots + (1 if slots else 0) + self.closing_steps() + 2 * len(self.goals))
        spare = self.ledger.moves_left() - slots - (1 if slots else 0) - self.closing_steps()
        each = max(2, spare // max(len(self.goals), 1)) if spare > 0 else 0
        if any(step_spec["moment"] == "start" for step_spec in self.process):
            self.liaison.process_check("start", "start the task (search the store)")
            self.liaison.read_news()
            slots = max(0, min(slots, self._messages_open() if self.message_cap is not None else slots))
        self._first_look(slots, each)
        if slots > 0 and self.liaison.may_ask():
            self._ask_round(slots)
        self._follow_pointers()
        if self.run_mode == "ask":
            return
        self.bench.assess_all()
        self._final_hunts()
        self.closer.decide_and_close()

    def _chains(self, floor: int, pairs: list[tuple[Goal, Want]] | None = None) -> None:
        """Pointer wants followed with `floor` steps kept free (all the request's own when no pairs are given)."""
        self.chain_floor = floor
        if pairs is None:
            pairs = [(g, w) for g in self.goals for w in list(g.wants) if w.pointer and w.relation in POINTER_OPS]
        for goal, want_ in pairs:
            self.tracer.follow_pointer(goal, want_)
        self.chain_floor = 0

    def _first_look(self, slots: int, each: int) -> None:
        last = len(self.goals) - 1
        for at, goal in enumerate(self.goals):
            share = each if at < last else max(each, self.ledger.moves_left() - slots - 1 - self.closing_steps())
            if share <= 0:
                break
            was_at = self.ledger.moves_left()
            self.hunt(goal, max(2, share // 2))
            self.read_menu_offers(goal, max(0, share - (was_at - self.ledger.moves_left())))

    def _ask_round(self, slots: int) -> None:
        """The planned questions (the plan is rebuilt after each need learned), then any messages left on more."""
        liaison, ledger = self.liaison, self.ledger
        plan = self.inquiry.next_questions(slots)
        while plan:
            goal, facet = plan.pop(0)
            if not liaison.may_ask() or ledger.moves_left() <= self.closing_steps():
                break
            if any(_overlap(want.facet, facet) >= 0.67 for want in goal.wants) or self.inquiry._known_facet(goal, facet):
                continue
            before = len(goal.wants)
            self._ask_once(goal, facet)
            if len(goal.wants) != before and plan:
                left = min(len(plan), self._messages_open() if self.message_cap is not None else len(plan))
                plan = self.inquiry.next_questions(left)
        while liaison.may_ask() and ledger.moves_left() > self.closing_steps() + 1 and ledger.seconds_left() > CLOSING_S + 30:
            more = self.inquiry.next_questions(1)
            if not more:
                break
            goal, facet = more[0]
            liaison.take_reply_guarded(goal, facet, liaison.ask_shopper(goal, facet))
            liaison.read_news()

    def hunt(self, goal: Goal, step_allow: int, last_round: bool = False) -> None:
        _Hunt(self, goal, step_allow).go()

    def read_menu_offers(self, goal: Goal, spend_steps: int | None = None) -> None:
        if self.ledger.moves_left() <= self.closing_steps() + 2 or "message" not in self.duties:
            return
        stop_line = self.ledger.moves_left() - (spend_steps if spend_steps is not None else self.ledger.moves_left())
        keep_free = max(stop_line, self.closing_steps() + 2)
        goal_hit = 12 if self.ledger.reads_many() else 5
        while self.ledger.moves_left() > keep_free and self.ledger.seconds_left() > CLOSING_S + 20 and not self.ledger.out_of_money():
            if not self._menu_round(goal, goal_hit, keep_free):
                return


    def more_queries(self, goal: Goal) -> list[str]:
        seen_list = sorted(goal.offers.values(), key=lambda o: (not o.is_open, o.oid))[:14]
        row_list_ = [{"title": o.heading[:110], "fits": (o.oid in {fields_.oid for fields_ in goal.good_offers(with_unsure=True)}) if o.is_open else None}
                for o in seen_list]
        tried_queries = sorted({key_.split(":", 1)[1] for key_ in goal.places if key_.startswith("search_text:")})
        cap_chars = (self.brief._shape_of("search_text").get("properties") or {}).get((self.duties.get("search_text") or {}).get("query") or "", {}).get("maxLength")
        user_msg = (f"PRODUCT: {goal.label} (category: {goal.shelf})\nREQUIREMENTS:\n- " + ("\n- ".join(goal.want_lines()) or "none")
                + f"\nTRIED: {json.dumps(tried_queries, ensure_ascii=False)}\nLISTINGS SEEN: {json.dumps(row_list_, ensure_ascii=False)}"
                + (f"\nMAX QUERY LENGTH: {cap_chars} characters" if cap_chars else ""))
        verdict_json = self.oracle.ask_json([QUICK_SEAT, SPARE_SEAT], KEYWORDS_PROMPT, user_msg, 700)
        result_: list[str] = []
        if isinstance(verdict_json, dict):
            for query_ in verdict_json.get("queries") or []:
                query_ = str(query_).strip()
                if query_ and len(query_) <= 60 and _plain(query_) not in {_plain(term_) for term_ in tried_queries + goal.queries + result_}:
                    result_.append(query_)
        self.ledger.jot("suggested", item=goal.label, queries=result_[:5])
        return result_[:5]

    def _seat_components(self) -> None:
        self.ledger = Ledger(self)
        self.brief = Brief(self)
        self.scout = Scout(self)
        self.bench = Bench(self)
        self.liaison = Liaison(self)
        self.inquiry = Inquiry(self)
        self.tracer = Tracer(self)
        self.coupler = Coupler(self)
        self.closer = Closer(self)

    def _read_view(self, env_part: dict[str, Any]) -> None:
        self.envelope_base = dict(env_part["binding"])
        self.view = env_part["policy_view"]
        self.ask_text_in = str(self.view.get("query") or "")
        self.first_seen = self.view.get("opening")
        self.tool_list = [term_ for term_ in self.view.get("tools") or [] if isinstance(term_, dict)]
        self.shapes: dict[str, dict[str, Any]] = {}
        for term_ in self.tool_list:
            fun = term_.get("function") if isinstance(term_.get("function"), dict) else term_
            if isinstance(fun, dict) and fun.get("name"):
                self.shapes[fun["name"]] = fun.get("parameters") or {"type": "object", "properties": {}}
        self.step_cap = int(self.view.get("max_steps") or 30)
        self.steps_used = 0
        self.over = bool(self.view.get("finished"))
        self.broke = False
        self.log_: list[dict[str, Any]] = []

    def _blank_plan(self) -> None:
        self.duties: dict[str, Any] = {}
        self.goals: list[Goal] = []
        self.ceiling: float | None = None
        self.lowest = False
        self.measure: str | None = None
        self.show_basket_first = False
        self.process: list[dict[str, Any]] = []
        self.past: list[dict[str, Any]] = []
        self.gate_tally: dict[str, int] = {}
        self.shown_basket: tuple | None = None
        self.message_cap: int | None = None
        self.messages_used = 0
        self.stock_only = True
        self.raise_topics: list[str] = []
        self.fit_list: list[dict[str, Any]] = []
        self.unread_lines: list[str] = []
        self.read_lines: set[str] = set()
        self.basket: dict[str, int] = {}
        self.basket_goal: dict[str, Goal] = {}

    def _recall_store(self) -> None:
        self.rows_per_page = 0
        self.store_key = "|".join(sorted(self.shapes))
        stored = _STORE_MEMORY.get(self.store_key) or {}
        self.key_maps: dict[str, dict[str, str]] = dict(stored.get("maps") or {})
        self.side_limit = int(stored.get("compare_limit") or 4)
        self.side_lacks_table = bool(stored.get("thin_compare"))
        self.pages_open: set[str] = set()
        self.decline_why = ""
        self.run_mode = "full"

    def _unopened_kind(self, goal: Goal) -> list[Offer]:
        result_ = [o for o in goal.offers.values() if not o.is_open and self.scout.on_shelf(goal, o) and o.can_buy is not False]
        result_.sort(key=lambda o: (o.row_verdict == "no", o.row_verdict != "yes", o.unit_price if o.unit_price is not None else 1e18))
        return result_

    def _menu_round(self, goal: Goal, goal_hit: int, keep_free: int) -> bool:
        wanted_n = goal_hit - len([o for o in goal.offers.values() if o.is_open and self.scout.on_shelf(goal, o)])
        if wanted_n <= 0:
            return False
        pending = self._unopened_kind(goal)
        if len(pending) < 2 and goal.searches_run < len(goal.queries) + 1:
            was_at = len(goal.offers)
            self.scout.look_up(goal, words=goal.shelf, page_limit=1, keep_steps=keep_free)
            pending = self._unopened_kind(goal)
            if len(goal.offers) == was_at and not pending:
                return False
        if not pending:
            return False
        slice_ids = min(wanted_n, self.side_limit if self.ledger.reads_many() else 4 if self.side_lacks_table else 1)
        if not self.scout.open_offers(goal, [o.oid for o in pending[:slice_ids]], keep_free=keep_free):
            return False
        self.bench.assess(goal, [o for o in self.scout.contenders(goal) if o.is_open])
        return True

    def _messages_open(self) -> int:
        return self.message_cap - self.messages_used - self.liaison.kept_messages()

    def _follow_pointers(self) -> None:
        for goal in self.goals:
            for want_ in list(goal.wants):
                if want_.pointer and want_.relation in POINTER_OPS:
                    self._chains(self.closing_steps() + 2, [(goal, want_)])

    def _final_hunts(self) -> None:
        order = sorted(self.goals, key=lambda g: (bool(g.good_offers()), -len(g.offers)))
        for at, goal in enumerate(order):
            share = max(2, (self.ledger.moves_left() - self.closing_steps()) // max(len(order) - at, 1))
            self.hunt(goal, share, last_round=True)

    def _ask_once(self, goal: Goal, facet: str) -> None:
        reply = self.liaison.ask_shopper(goal, facet)
        if isinstance(reply, dict) and str(reply.get("kind") or "").lower() == "split" \
                and self.liaison.may_ask() and self.ledger.moves_left() > self.closing_steps():
            reply = self.liaison.ask_shopper(goal, facet, narrowed_q=True)
        self.liaison.take_reply_guarded(goal, facet, reply)
        self.liaison.read_news()


def agent_main(task_data: dict[str, Any]) -> list[dict[str, Any]]:
    return Venture(task_data).go()


__all__ = ["agent_main"]
