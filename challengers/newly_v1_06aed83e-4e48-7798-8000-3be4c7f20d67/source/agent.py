"""Sieve: a shopping agent where code decides every step and models only read.

Four parts work over one shared table of listings:
  Interview  - a belief per (item, property). Each message names several properties in belief order, and
               each reply updates every name by how likely it was to be heard at its place in the message.
  Hunt       - searches and opens listings in the order the item's pick rule needs, until the fitting set
               is good enough or its steps run out.
  Verdicts   - one cached yes / no / unknown per (listing, requirement), each yes backed by a quote that code
               checks against the record. A change in the task only adds requirements to judge.
  Finish     - estimates the chance that ordering or ending without a purchase is right, and takes the better.
"""
from __future__ import annotations

import json
import re
import time
from typing import Any

from src.agent.proxy_client import ProxyClient

LIGHT = "anthropic/claude-haiku-5.5"
HEAVY = "anthropic/claude-sonnet-5.5"

COST_SOFT = 0.045     # above this spend, late reads use the light model first
COST_STOP = 0.075     # no model call above this spend: code finishes alone
DEADLINE = 245.0      # seconds this agent plans to use of the 300 s a task has
HURRY = 75.0          # with less time left than this, stop hunting and head for the finish
LAST_MODEL = 25.0     # with less time left than this, no model call starts

# Rough settings measured on one race: how likely a held preference is to be answered at each place in a
# message. They only set the order of later questions; a different world costs at most some efficiency.
HEARD = (0.9, 0.7, 0.7, 0.55, 0.55, 0.4, 0.4, 0.35)
MAX_NAMES = 8
MIN_BELIEF = 0.02

FIT_SCORE = 0.85      # chance an order is right when every requirement of a line has a checked yes
# Rough settings. Across this agent's own practice runs, picks with an open requirement or read only as a row were
# never acceptable, so an order needs a verified fit: these stay below the chance of ending without a purchase.
UNKNOWN_SCORE = 0.05  # one requirement the records leave open
NO_SCORE = 0.01       # one requirement the records contradict
UNREAD_SCORE = 0.02   # a listing seen only as a search row
# Chances that ending without a purchase is right. Rough settings: in one race only about 5% of such endings were.
END_BASE = 0.02       # every line has a fit
END_OPEN = 0.05       # some line has no fit but the search was not exhausted
END_STATED = 0.1      # exhausted search, and only the request's own requirements rule listings out
END_TOLD = 0.2        # exhausted search, and a requirement the shopper revealed rules listings out

_llm = ProxyClient(timeout=45, max_retries=1, retry_delay=1)
_env = ProxyClient(timeout=50, max_retries=3, retry_delay=1)

READ_REQUEST = """TASK read_request. An agent serves one shopper in an online store and acts only through the store's tools. Read the request and the tools. Return ONE JSON object, nothing else:
{"items": [{"name": "the kind of product in the request's words", "head": "the one word of the name that says what the product is", "shelf": "the store category name the request uses for it, or empty", "quantity": integer, "pick": "any" | "cheapest" | "most" | "least", "measure": "property the pick compares, or empty", "needs": [{"property": "property name in the request's words", "want": "the value, limit or exclusion exactly as stated", "words": "at most 3 words a listing meeting it would show, or empty for a limit or an exclusion, which no keyword can express", "pointer": true when the value must be read from another listing the request describes, "about": "product" when it describes the product itself, "other" when it concerns price, budget, stock, delivery or how to deal with the shopper}]}],
"budget": number or null, "budget_words": "the budget word for word as the request writes it, or empty", "budget_scope": "total" | "each",
"messages": how many messages the request allows the agent to send, or null,
"steps": [{"step": "a procedure the request tells the agent to follow, such as telling or checking something before ordering", "when": "before_order" | "before_end" | "any", "uses_message": true only when the request explicitly asks for a message to be sent}],
"hints": [{"item": item number, "property": "a property the story points at without giving its value", "words": "the story's words"}],
"roles": {"facts": "tool that shows one listing's attributes", "facts_id": "its id argument", "offer": "tool that shows one listing's price and stock if different from facts, else empty", "offer_id": "its id argument", "ask": "tool that messages the shopper", "ask_text": "its text argument", "add": {"tool": "cart tool", "lines": "when the tool sets the whole cart at once from a list of lines, the argument holding that list, else empty", "id": "id argument, or the id key inside a line", "qty": "quantity argument, or the quantity key inside a line, or empty", "fixed": {other arguments that make it add}}, "remove": {"tool": "cart tool, or empty when the cart is only set whole", "id": "id argument", "qty": "quantity argument or empty", "fixed": {other arguments that make it remove}}, "cart": "tool that shows the cart, or empty", "buy": "tool that places the order", "buy_args": {arguments that make it place the order, when one tool both orders and ends}, "end": "tool that ends without a purchase, which may be the same tool", "end_args": {arguments that make it end without a purchase; only what is required}},
"search": {"words_tool": "tool for keyword search", "words_fixed": {arguments that select a keyword search}, "words_arg": "argument holding the keywords", "words_max": max characters of the keywords or null, "words_page_arg": "its argument asking for a later page, or empty", "shelf_tool": "tool for browsing a category, which may be the same tool, or empty if none", "shelf_fixed": {arguments that select browsing a category}, "shelf_arg": "argument naming the category, or empty", "shelf_page_arg": "its argument asking for a later page, or empty", "shelf_with_words": true when the keyword search also accepts the category argument, "price_arg": "the browsing tool's argument limiting the highest price, or empty"}}
Rules: copy the budget's number exactly as the request writes it, with no change of unit. One entry per distinct product wanted. "needs" hold only what the request states about the product's own properties, including limits and exclusions. "want" is the value alone, with no explanation added. When the request says what the product itself is for, that purpose is a need, under the property "purpose". Other plans, places, activities and people in the story are not needs. Never make a need of the price or budget, stock or availability, substitutes, delivery, the message or step limits, or how to report back. Words that are simply what the product is called are not needs: the whole name goes in "name" and "shelf", and no part of it is split off as a need. Ending without a purchase when nothing fits, or saying so, is not a step. Use exact tool and argument names from the tools below.

REQUEST:
"""

SHAPE = """TASK shape. A store's search returned the result below. Name its parts by exact key. Return ONE JSON object, nothing else: {"rows": "key of the list of listings", "id": "key of a listing's id", "title": "key of its title or empty", "price": "key of its price or empty", "stock": "key saying if it can be bought, or empty", "category": "key of its category label or empty", "seller": "key of its seller or brand or empty", "total": "key of how many matched in all, or empty", "next": "key of the value that asks for the next page, or empty"}

RESULT:
"""

RANK = """TASK rank. A shopper holds a few firm preferences they did not write down. Each can be learned only by asking about that exact property. For each item below, list the properties worth asking about and estimate, for each, the chance the shopper holds an unwritten firm preference on it.
Return ONE JSON object, nothing else: {"items": [{"item": item number, "properties": [{"name": "plain property name, using all the words of the key", "keys": ["every listed key that means this same property"], "p": number from 0 to 1}]}]}
Rules: use the listed keys. Add a property no key lists only when the request's story points at it, and leave its "keys" empty. Each name is one property, with no "or" and no "vs". Leave out a key whose values list several purchase options at once. Merge different spellings of one property into one entry. Leave out properties the item's stated needs already settle, identifiers, seller names, and selling terms such as warranty, shipping or returns. Raise "p" for a property the request's story points at. Keep at most 14 per item, most likely first.

REQUEST:
"""

READ_REPLY = """TASK read_reply. An agent asked a shopper about some properties of the products below and got this reply. Return ONE JSON object, nothing else:
{"kind": "need" when the reply states a firm preference, "no_view" when it says the shopper does not mind, "puzzled" when it says the message was not understood, "too_many" when it asks for fewer things at a time, "refused" when the store says the message could not be sent or no messages remain, "other",
"needs": [{"item": item number, "property": "property name as the reply says it", "want": "the value, limit or exclusion, word for word with its units", "words": "at most 3 words a listing meeting it would show, or empty for a limit or an exclusion", "pointer": true when the value must be read from another listing the reply describes}],
"drop": [{"item": item number, "property": "requirement the shopper gives up"}], "budget": a new total limit as a number, or null}

"""

READ_NOTE = """TASK read_note. While an agent shops, the store showed this text. Return ONE JSON object, nothing else:
{"kind": "change" when it changes the task or says the order cannot go through, "error" when it only reports a malformed call, "other",
"budget": new budget number or null, "needs": [{"item": item number, "property": "...", "want": "...", "words": "at most 3 words a listing meeting it would show, or empty for a limit or an exclusion", "pointer": false}], "drop": [{"item": item number, "property": "..."}],
"price_moved": true when it says in any words that the price of something the agent picked has changed, "gone": [listing ids it says are unavailable], "gone_unnamed": true when it says in any words that something the agent picked or suggested is gone, sold out, used up or cannot be ordered, without giving an id, "summary": "a few words"}

"""

JUDGE = """TASK judge. For each listing record below decide each requirement of the item from the record alone.
- The attribute whose key names the requirement's property decides. Key spelling does not matter, one element of a list is enough, case and singular or plural do not matter, units convert, limits include the edge, and a range must lie wholly inside a limit.
- A value that contains all the required words meets it. A different word does not, even a near meaning.
- Records often repeat a property under several keys. When any key naming the property states the required value, the verdict is "yes", even if another such key states something else.
- With no such attribute, the value stated in another attribute, the title or the description counts. If nothing states it, the verdict is "unknown".
- "quote" is the exact words of the record that decide the verdict, copied character for character.
- A requirement about availability or price is decided by the record's stock and price fields.
- "kind_ok" is false when the listing's category or title shows another kind of product than the item.
Return ONE JSON object, nothing else: {"listings": [{"id": "...", "kind_ok": boolean, "measure": number or null, "verdicts": [{"n": requirement number, "verdict": "yes" | "no" | "unknown", "quote": "..."}]}]}

"""

QUERIES = """TASK queries. Write short keyword searches that would find listings of this item meeting its requirements. The store's search needs every word to be on the listing, so each extra word can hide the right one: use at most 3 words, the wanted value first, and vary the words a listing might use for it. Leave out anything the listing must not be. Return ONE JSON object, nothing else: {"queries": ["..."]} with at most 3 entries.

"""

RESULT = """TASK result. A store's tool replied as below to a call. Did the call go through? It did not when the reply says it was refused, failed or was not allowed. Return ONE JSON object, nothing else: {"ok": true or false}

"""

STOCK = """TASK stock. A store's listing has this value in its stock field. Can the listing be bought now? Return ONE JSON object, nothing else: {"buyable": true or false}

VALUE: """

OFFER = """TASK offer. A store's page for one listing is below. Name its parts by exact key. Return ONE JSON object, nothing else: {"price": "key of its price, or empty", "stock": "key saying if it can be bought, or empty"}

PAGE:
"""

POINTER = """TASK pointer. A requirement takes its value from another listing. Using the requirement and the pages opened so far, say what to do next. Return ONE JSON object, nothing else:
{"value": "the final value when the last page gives it, else empty", "open": "id of a listing shown below to open next, or empty", "search": "keywords, at most 6 words, to find the listing still needed, or empty", "shelf": "its category if known, or empty"}

"""

CHECK = """TASK check. An agent is about to order this cart. Check it against every requirement from the records alone, reading values the way a careful person would: spelling, case and units do not matter, a different word does. Return ONE JSON object, nothing else: {"wrong": [{"id": listing id, "n": requirement number, "why": "a few words"}]} with an empty list when all is right. List only what a record clearly contradicts.

"""

GATE = """TASK gate. The request asks the agent to follow these procedures before finishing. From the record of what was done, name the next call needed, if any. Return ONE JSON object, nothing else: {"call": {"tool": "exact tool name", "args": {}} or null}
Never call a tool that orders or ends the task.

"""


def _dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _plain(text: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def _stem(word: str) -> str:
    """One spelling for a word's singular and plural, worked out from its ending."""
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        word = word[:-1]
    word = re.sub(r"(ch|sh|ss|x|z)e$", r"\1", word)
    if len(word) > 4:
        word = re.sub(r"ie$", "y", word)
    return re.sub(r"(fe|f|ve)$", "v", word)


def _words(text: Any) -> set[str]:
    return {_stem(w) for w in _plain(text).split()}


def _json_of(text: str) -> dict[str, Any] | None:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        parsed = json.loads(text[start:end + 1])
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        found = re.search(r"-?\d+(?:\.\d+)?", value.replace(",", ""))
        return float(found.group()) if found else None
    return None


def _dicts(value: Any) -> list[dict[str, Any]]:
    return [x for x in value if isinstance(x, dict)] if isinstance(value, list) else []


def belief_after_silence(p: float, heard: float) -> float:
    """The belief in a property after a message named it, with chance `heard` of an answer, and got none."""
    missed = p * (1.0 - heard)
    return missed / (missed + 1.0 - p) if missed + 1.0 - p > 0 else 0.0


def heard_at(place: int) -> float:
    return HEARD[min(place, len(HEARD) - 1)]


def update_beliefs(beliefs: dict[str, float], names: list[str], hit: int | None) -> None:
    """Update beliefs after a message naming `names` in order. `hit` is the place of the name that was answered.
    Names before it were passed over, so they weaken. Names after it were never reached, so they keep their
    belief. With no hit, every name was passed over."""
    last = len(names) if hit is None else hit
    for place, name in enumerate(names[:last]):
        if name in beliefs:
            beliefs[name] = belief_after_silence(beliefs[name], heard_at(place))
    if hit is not None and names[hit] in beliefs:
        del beliefs[names[hit]]


def line_gain(beliefs: dict[str, float], limit: int = MAX_NAMES) -> float:
    """The chance that a message of the top names of this line brings a preference to light."""
    miss = 1.0
    for place, p in enumerate(sorted(beliefs.values(), reverse=True)[:limit]):
        miss *= 1.0 - p * heard_at(place)
    return 1.0 - miss


def split_notes(observation: Any) -> tuple[Any, list[str]]:
    """Separate a tool result from notes carried beside it: a two-key wrapper holding the result and a list of
    texts, or a bare list of texts."""
    if isinstance(observation, list) and observation and all(isinstance(x, str) for x in observation):
        return None, list(observation)
    if isinstance(observation, dict) and len(observation) == 2:
        first, second = list(observation.values())
        for texts, rest in ((first, second), (second, first)):
            if isinstance(texts, list) and texts and all(isinstance(x, str) for x in texts) and not isinstance(rest, list):
                return rest, list(texts)
    return observation, []


def fit_args(args: dict[str, Any], schema: dict[str, Any] | None) -> dict[str, Any]:
    """Shape arguments to a tool's schema: unknown names go, numbers are coerced and clamped, long text is cut."""
    props = (schema or {}).get("properties") or {}
    out: dict[str, Any] = {}
    for name, value in (args or {}).items():
        spec = props.get(name)
        if spec is None or value is None:
            continue
        kind = spec.get("type")
        if kind in ("integer", "number"):
            number = _number(value)
            if number is None:
                continue
            value = int(number) if kind == "integer" else number
            if isinstance(spec.get("minimum"), (int, float)):
                value = max(value, spec["minimum"])
            if isinstance(spec.get("maximum"), (int, float)):
                value = min(value, spec["maximum"])
        elif kind == "string":
            value = str(value)
            if isinstance(spec.get("maxLength"), int):
                value = value[:spec["maxLength"]]
        if isinstance(spec.get("enum"), list) and value not in spec["enum"]:
            continue
        out[name] = value
    return out


def search_phrase(want: str) -> str:
    """The first wording of a wanted value."""
    return re.split(r"[,;(]|\s+or\s+|\s+and\s+", str(want).strip())[0].strip()


def phrase_of(need: dict[str, Any]) -> str:
    """What to search for to meet a need: a value read from another listing, else the reader's search words, which are
    empty for a limit or an exclusion. With no reading, the wanted value's own words."""
    if need["value"]:
        return search_phrase(need["value"])
    return search_phrase(need["want"] if need["words"] is None else need["words"])


def query_ladder(name: str, phrases: list[str], head: str = "") -> list[str]:
    """Searches built from the wanted values. An all-words search hides a listing for every extra word, so each value
    goes with the word that says what the product is, then alone; several values together come first, as the sharpest
    try. Without a reading of that word, the name's last word stands in."""
    words = _plain(name).split()
    noun = _plain(head) or (words[-1] if words else "")
    out: list[str] = []
    if len(phrases) > 1:
        out.append(" ".join(phrases + [noun]).strip())
    out += [f"{p} {noun}".strip() for p in phrases]
    out += phrases
    if not phrases and name:
        out.append(name)
    seen: list[str] = []
    for query in out:
        if query and query not in seen:
            seen.append(query)
    return seen[:6]


def question_text(parts: list[tuple[str, list[str]]]) -> str:
    """One sentence per item, each naming that item's properties in order."""
    out = []
    for item, names in parts:
        if len(names) == 1:
            out.append(f"What {names[0]} would you like for the {item}?")
        else:
            out.append(f"For the {item}, what {', '.join(names[:-1])} or {names[-1]} would you like?")
    return " ".join(out)


class Line:
    """One requested product and everything known about it."""

    def __init__(self, number: int, raw: dict[str, Any]):
        self.number = number
        self.name = str(raw.get("name") or "item")
        self.head = str(raw.get("head") or "")
        self.shelf = str(raw.get("shelf") or "")
        self.quantity = max(1, int(_number(raw.get("quantity")) or 1))
        self.pick = str(raw.get("pick") or "any")
        self.measure = str(raw.get("measure") or "")
        self.needs: list[dict[str, Any]] = []
        self.ids: list[str] = []
        self.beliefs: dict[str, float] = {}
        self.keys_of: dict[str, list[str]] = {}
        self.asked = 0
        self.queries: list[str] = []
        self.queries_made = False
        self.searches = 0
        self.more: list[dict[str, Any]] = []
        self.word_hits: dict[str, int] = {}
        self.exact = False
        self.related = False  # some row's label shares a word with the item
        self.wide = False     # the item's own labels gave no fit, so any label may be read
        self.capped_at: float | None = None
        self.last_names: list[str] = []
        self.last_silent = False
        self.browsed: set[str] = set()
        self.tried: list[str] = []
        self.rethought = False
        for need in _dicts(raw.get("needs")):
            if need.get("about") != "other":
                self.add_need(need, False)

    def add_need(self, raw: dict[str, Any], told: bool) -> dict[str, Any] | None:
        prop, want = str(raw.get("property") or "").strip(), str(raw.get("want") or "").strip()
        if not prop and not want:
            return None
        text = f"{prop}: {want}"
        for need in self.needs:
            if need["text"] == text:
                need["live"] = True
                return need
        words = raw.get("words")
        need = {"text": text, "prop": prop, "want": want, "words": None if words is None else str(words), "told": told, "pointer": bool(raw.get("pointer")), "value": "", "live": True}
        self.needs.append(need)
        return need

    def live(self) -> list[dict[str, Any]]:
        return [n for n in self.needs if n["live"]]

    def label(self) -> str:
        return self.shelf.lower() if self.shelf else self.name


class Sieve:
    def __init__(self, problem: dict[str, Any]):
        environment = problem["environment"]
        self.binding = environment["binding"]
        policy = environment["policy_view"]
        self.request = str(policy.get("query") or "")
        if policy.get("opening") is not None:
            self.request += "\n\n" + _dump(policy["opening"])
        self.tools = policy.get("tools") or []
        self.schemas = {t["function"]["name"]: t["function"].get("parameters") or {} for t in self.tools}
        self.max_steps = int(policy.get("max_steps") or 30)
        self.finished = bool(policy.get("finished"))
        self.t0 = time.monotonic()
        self.step = 0
        self.cost = 0.0
        self.calls = 0
        self.trace: list[dict[str, Any]] = []
        self.roles: dict[str, Any] = {}
        self.search: dict[str, Any] = {}
        self.shape: dict[str, Any] = {}
        self.lines: list[Line] = []
        self.brief: dict[str, Any] = {}
        self.budget: float | None = None
        self.budget_each = False
        self.msg_limit = 3
        self.msg_used = 0
        self.name_cap = MAX_NAMES
        self.rows: dict[str, dict[str, Any]] = {}
        self.pages: dict[str, dict[str, Any]] = {}
        self.owner: dict[str, set[int]] = {}
        self.verdicts: dict[tuple[str, str], tuple[str, str]] = {}
        self.kind_bad: set[str] = set()
        self.judged: set[tuple[str, int]] = set()
        self.measures: dict[str, float] = {}
        self.gone: set[str] = set()
        self.cart: dict[str, int] = {}
        self.pending_gone = False
        self.pending_price = False
        self.changed = False
        self.notes_seen = 0
        self.note_readings: dict[str, dict[str, Any]] = {}
        self.searched: set[str] = set()
        self.done_log: list[str] = []
        self.buy_tries = 0
        self.pressed = False
        self.stock_words: dict[str, bool] = {}
        self.result_readings: dict[Any, bool] = {}

    # ---- clock, money, models ----
    def time_left(self) -> float:
        return DEADLINE - (time.monotonic() - self.t0)

    def steps_left(self) -> int:
        return self.max_steps - self.step

    def tail(self) -> int:
        """Steps kept for the finish: the first order (cart lines, a look at the cart, the order) and one swap. An
        order goes out at the first verified fit, so until then steps do more good finding one."""
        return 2 * len(self.lines) + 3

    def read(self, prompt: str, models: list[str], max_tokens: int = 1500) -> dict[str, Any] | None:
        for model in models:
            if self.cost >= COST_STOP or self.time_left() < LAST_MODEL:
                return None
            _llm.timeout = max(5, min(45, int(self.time_left() - 10)))
            self.calls += 1
            result = _llm.post_verbose("/inference/chat/completions", {
                "model": model, "max_tokens": max_tokens, "temperature": 0,
                "messages": [{"role": "user", "content": prompt}]})
            data = result.data
            if not isinstance(data, dict) or not data.get("choices"):
                continue
            usage = data.get("usage") or {}
            if isinstance(usage.get("cost"), (int, float)):
                self.cost += float(usage["cost"])
            parsed = _json_of(str(data["choices"][0].get("message", {}).get("content") or ""))
            if parsed is not None:
                return parsed
            self.trace.append({"kind": "read_fail", "model": model, "task": prompt[:24]})
        return None

    def careful(self) -> list[str]:
        return [HEAVY, LIGHT] if self.cost < COST_SOFT else [LIGHT, HEAVY]

    # ---- the store ----
    def call(self, name: str, args: dict[str, Any]) -> Any:
        """One environment step. Returns the tool's own result; notes beside it are read at once."""
        if self.finished or self.steps_left() <= 0 or name not in self.schemas:
            return None
        self.step += 1
        reply = _env.post("/environment/call", {**self.binding, "step": self.step, "action": {"name": name, "args": fit_args(args, self.schemas[name])}})
        if reply is None:
            self.step -= 1
            self.trace.append({"kind": "env_fail", "tool": name})
            return None
        self.finished = bool(reply.get("finished"))
        result, notes = split_notes(reply.get("observation"))
        self.done_log.append(f"{self.step}. {name} {_dump(args)[:160]} -> {_dump(result)[:160]}")
        for note in notes:
            self.on_note(note)
        return result

    def refused(self, tool: str, reply: Any) -> bool:
        """Whether a tool's reply says the call did not go through. Stores word this differently, so a model reads it,
        once per shape of reply. With no reading the call counts as done, and the order step reports any trouble."""
        if not isinstance(reply, dict) or not reply:
            return False
        shape = (tool, tuple(sorted((key, value is False, not value) for key, value in reply.items())))
        if shape not in self.result_readings:
            got = self.read(RESULT + f"TOOL: {tool}\nREPLY: {_dump(reply)[:600]}", [LIGHT, HEAVY], 60) or {}
            self.result_readings[shape] = got.get("ok") is False
        return self.result_readings[shape]

    def on_note(self, text: str) -> None:
        self.notes_seen += 1
        self.trace.append({"kind": "note", "text": text[:200]})
        got = self.note_readings.get(text)
        if got is None:
            items = "\n".join(f"item {l.number}: {l.name}" for l in self.lines)
            got = self.read(READ_NOTE + f"ITEMS:\n{items}\nCART: {_dump(self.cart)}\n\nTEXT:\n{text[:1500]}", [LIGHT, HEAVY], 500) or {}
            self.note_readings[text] = got
        self.apply_change(got)

    def apply_change(self, got: dict[str, Any]) -> None:
        # the kind label alone is not trusted: any concrete change counts, and a note that could not be read is
        # treated as possibly saying the pick is gone, which costs a stock check rather than a wrong order
        concrete = [got.get(k) for k in ("budget", "needs", "drop", "gone", "gone_unnamed", "price_moved")]
        if got.get("kind") == "change" or any(concrete):
            self.changed = True
        if not got:
            self.changed = True
            self.pending_gone = True
        budget = _number(got.get("budget"))
        if budget is not None:
            self.budget = budget
        for raw in _dicts(got.get("needs")):
            self.line_of(raw.get("item")).add_need(raw, True)
        self.drop_needs(got.get("drop"))
        for ident in got.get("gone") or []:
            self.gone.add(str(ident))
        if got.get("gone_unnamed"):
            self.pending_gone = True
        if got.get("price_moved"):
            self.pending_price = True

    def drop_needs(self, raws: Any) -> None:
        """The shopper gave these up. The same property can be named with more or fewer words each time, so a name
        contained in the other either way is a match."""
        for raw in _dicts(raws):
            words = _words(raw.get("property"))
            for need in self.line_of(raw.get("item")).needs:
                prop = _words(need["prop"])
                if words and prop and (words <= prop or prop <= words):
                    need["live"] = False

    def line_of(self, number: Any) -> Line:
        index = int(_number(number) or 1)
        return self.lines[min(max(index, 1), len(self.lines)) - 1]

    def learn_shape(self, result: Any) -> None:
        # an empty result cannot show what a listing looks like, so wait for one that holds records
        if self.shape.get("rows") or not isinstance(result, dict) or not any(_dicts(v) for v in result.values()):
            return
        got = self.read(SHAPE + _dump(result)[:2500], [LIGHT, HEAVY], 300) or {}
        if got.get("rows"):
            self.shape = got
            return
        # no reading available: the first list of records, and its first key holding a different code on every row
        for key, value in result.items():
            rows = _dicts(value)
            if rows:
                codes = [k for k in rows[0] if all(isinstance(r.get(k), str) and " " not in r[k] and re.search(r"\d", r[k]) for r in rows) and len({r[k] for r in rows}) == len(rows)]
                self.shape = {"rows": key, "id": codes[0] if codes else ""}
                return

    def rows_in(self, result: Any) -> list[dict[str, Any]]:
        if not isinstance(result, dict):
            return []
        rows = _dicts(result.get(self.shape.get("rows")))
        return rows or next((_dicts(v) for v in result.values() if _dicts(v)), [])

    def ident(self, row: dict[str, Any]) -> str:
        return str(row.get(self.shape.get("id")) or "")

    def price(self, ident: str) -> float | None:
        for record in (self.pages.get(ident), self.rows.get(ident)):
            if record:
                value = _number(record.get(self.shape.get("price")))
                if value is not None:
                    return value
        return None

    def in_stock(self, ident: str) -> bool:
        if ident in self.gone:
            return False
        key = self.shape.get("stock")
        for record in (self.pages.get(ident), self.rows.get(ident)):
            if record and key in record:
                return self.stock_says(record[key])
        return True

    def stock_says(self, value: Any) -> bool:
        """A flag or count speaks for itself. Words differ between stores, so a model reads each one, once."""
        if isinstance(value, (bool, int, float)):
            return bool(value)
        text = str(value or "").strip()
        if not text:
            return True
        if text not in self.stock_words:
            got = self.read(STOCK + text[:80], [LIGHT, HEAVY], 60) or {}
            self.stock_words[text] = got.get("buyable") is not False
        return self.stock_words[text]

    def category(self, ident: str) -> str:
        return str((self.rows.get(ident) or self.pages.get(ident) or {}).get(self.shape.get("category")) or "")

    def affordable(self, ident: str, line: Line) -> bool:
        price = self.price(ident)
        if self.budget is None or price is None:
            return True
        return (price if self.budget_each else price * line.quantity) <= self.budget

    def over_budget(self, cart: dict[str, int]) -> bool:
        """Whether a cart breaks a total budget. A limit per unit is checked listing by listing instead."""
        if self.budget is None or self.budget_each:
            return False
        return sum((self.price(i) or 0) * count for i, count in cart.items()) > self.budget

    def run_search(self, line: Line, args: dict[str, Any]) -> Any:
        tool = self.tool_of_search(args)
        mark = tool + _dump(sorted(fit_args(args, self.schemas.get(tool)).items()))
        if mark in self.searched:
            return None  # the same search shows the same rows, so it would only spend a step
        self.searched.add(mark)
        result = self.call(tool, args)
        self.learn_shape(result)
        rows = self.rows_in(result)
        for row in rows:
            ident = self.ident(row)
            if ident:
                self.rows[ident] = {**self.rows.get(ident, {}), **row}
                self.owner.setdefault(ident, set()).add(line.number)
                if ident not in line.ids:
                    line.ids.append(ident)
                if self.search.get("words_arg") in args:
                    line.word_hits[ident] = line.word_hits.get(ident, 0) + 1
        line.searches += 1
        if rows and line.number:
            self.learn_shelf(line)
        self.trace.append({"kind": "search", "args": _dump(args)[:120], "rows": len(rows), "item": line.number, "ids": [self.ident(r) for r in rows]})
        self.queue_next_page(line, args, result, rows)
        return result

    def queue_next_page(self, line: Line, args: dict[str, Any], result: Any, rows: list[dict[str, Any]]) -> None:
        """Remember how to read on: the store's own next value, or an offset or page number from the schema."""
        tool = self.tool_of_search(args)
        arg = self.search.get("words_page_arg") if tool == self.search.get("words_tool") and self.search.get("words_arg") in args else self.search.get("shelf_page_arg")
        if not arg or not rows or not isinstance(result, dict):
            return
        following = result.get(self.shape.get("next")) if self.shape.get("next") else None
        total = _number(result.get(self.shape.get("total"))) if self.shape.get("total") else None
        spec = ((self.schemas.get(tool) or {}).get("properties") or {}).get(arg) or {}
        current = args.get(arg)
        if following is None:
            if self.shape.get("next") and self.shape.get("next") in result:
                return
            if total is None or spec.get("type") != "integer":
                return
            if spec.get("minimum") == 1:
                following = int(current or 1) + 1
            else:
                following = int(current or 0) + len(rows)
            limit = spec.get("maximum")
            if isinstance(limit, int) and following > limit:
                return
            if spec.get("minimum") != 1 and following >= total:
                return
        # later pages of a browse come first; of keyword searches only the first one's are worth a step
        page = {**args, arg: following}
        if self.search.get("words_arg") not in args:
            line.more.insert(0, page)
        elif not any(self.search.get("words_arg") in other for other in line.more):
            line.more.append(page)

    def open_listing(self, ident: str) -> None:
        tool, arg = self.roles.get("facts"), self.roles.get("facts_id")
        if not tool or not arg:
            return
        result = self.call(tool, {arg: ident})
        if isinstance(result, dict) and result:
            self.pages[ident] = {**self.pages.get(ident, {}), **result}
        else:
            self.pages.setdefault(ident, {})
        self.read_offer(ident)

    def read_offer(self, ident: str) -> None:
        """A store may show price and stock only on a second page. It is read only when the price is unknown and a
        budget or a cheapest pick depends on it."""
        tool, arg = self.roles.get("offer"), self.roles.get("offer_id")
        depends = self.budget is not None or any(l.pick == "cheapest" for l in self.lines)
        if not tool or not arg or not depends or self.price(ident) is not None or self.steps_left() <= self.tail():
            return
        extra = self.call(tool, {arg: ident})
        if not isinstance(extra, dict) or not extra:
            return
        self.pages[ident] = {**self.pages.get(ident, {}), **extra}
        self.rows[ident] = {**self.rows.get(ident, {}), **extra}
        if not self.shape.get("price"):
            got = self.read(OFFER + _dump(extra)[:1500], [LIGHT, HEAVY], 100) or {}
            if got.get("price"):
                self.shape["price"] = got["price"]
            if got.get("stock") and not self.shape.get("stock"):
                self.shape["stock"] = got["stock"]

    def record_text(self, ident: str) -> str:
        return _dump({**(self.rows.get(ident) or {}), **(self.pages.get(ident) or {})})[:2200]

    # ---- reading the task ----
    def read_task(self) -> None:
        tools = "\n".join(f"- {t['function']['name']}: {t['function'].get('description', '')} | {_dump(t['function'].get('parameters') or {})}" for t in self.tools)
        got = self.read(READ_REQUEST + self.request + "\n\nTOOLS:\n" + tools, [HEAVY, LIGHT], 2500) or {}
        self.brief = got
        self.roles = got.get("roles") if isinstance(got.get("roles"), dict) else {}
        self.search = self.normal_search(got.get("search"))
        self.lines = [Line(i, raw) for i, raw in enumerate(_dicts(got.get("items")), 1)] or [Line(1, {"name": "item"})]
        self.budget = self.as_written(_number(got.get("budget")), got.get("budget_words"))
        self.budget_each = got.get("budget_scope") == "each"
        limit = _number(got.get("messages"))
        self.msg_limit = int(limit) if limit is not None else 3
        self.trace.append({"kind": "brief", "items": [(l.name, l.quantity, l.pick, [n["text"] for n in l.needs]) for l in self.lines], "budget": self.budget, "messages": self.msg_limit})

    def normal_search(self, raw: Any) -> dict[str, Any]:
        """One view of the search tools: a store may have one tool for both ways of searching, or one for each."""
        search = dict(raw) if isinstance(raw, dict) else {}
        shared = search.get("tool") or ""
        search["words_tool"] = search.get("words_tool") or shared
        search["shelf_tool"] = search.get("shelf_tool") or shared
        for field in ("words_page_arg", "shelf_page_arg"):
            search[field] = search.get(field) or search.get("page_arg") or ""
        return search

    def tool_of_search(self, args: dict[str, Any]) -> str:
        return self.search.get("words_tool") if self.search.get("words_arg") in args else (self.search.get("shelf_tool") or self.search.get("words_tool") or "")

    def as_written(self, budget: float | None, words: Any = "") -> float | None:
        """A budget read smaller than the request writes it was converted to a larger unit: the written number stands.
        A larger reading is kept, since the request's words can multiply the number. The words must really be there."""
        text = str(words or "")
        if budget is None or not text or text.lower() not in self.request.lower():
            return budget
        found = re.findall(r"\d+(?:\.\d+)?", text.replace(",", ""))
        return float(found[0]) if len(found) == 1 and float(found[0]) > budget else budget

    def steps_of_procedure(self, stage: str) -> list[str]:
        return [str(s.get("step")) for s in _dicts(self.brief.get("steps")) if s.get("when") in (stage, "any")]

    def messages_free(self) -> int:
        kept = sum(1 for s in _dicts(self.brief.get("steps")) if s.get("uses_message") and s.get("when") != "before_end")
        return self.msg_limit - self.msg_used - kept

    # ---- scouting: learn each item's properties ----
    def shelf_args(self, line: Line, query: str | None = None) -> dict[str, Any]:
        s = self.search
        if query:
            args = dict(s.get("words_fixed") or {})
            args[s.get("words_arg", "")] = query[:int(_number(s.get("words_max")) or 200)]
            if s.get("shelf_with_words") and s.get("shelf_arg") and line.shelf and s.get("words_tool") == s.get("shelf_tool"):
                args[s["shelf_arg"]] = line.shelf
        else:
            args = dict(s.get("shelf_fixed") or {})
            if s.get("shelf_arg"):
                args[s["shelf_arg"]] = line.shelf or line.name
        return args

    def learn_shelf(self, line: Line) -> None:
        """Take the store's own label for this item from the rows seen. The label that matches the item as a whole
        becomes the only one accepted, so a neighbouring category sharing a word is not bought from."""
        mine = _words(line.shelf) or _words(line.name)
        scored: dict[str, float] = {}
        for ident in line.ids:
            label = self.category(ident)
            words = _words(label)
            if label and label not in scored and words & (mine | _words(line.name)):
                scored[label] = len(words & mine) / len(words | mine)
        if not scored:
            return
        line.related = True
        best = max(scored, key=lambda k: scored[k])
        if scored[best] >= 0.5:
            line.shelf, line.exact = best, True
        elif not line.shelf:
            line.shelf = best

    def browse_label(self, line: Line) -> None:
        """Browse the store's own label for the item, once it is known, if that has not been done yet."""
        if self.search.get("shelf_arg") and line.shelf and line.shelf not in line.browsed and line.ids:
            line.browsed.add(line.shelf)
            self.run_search(line, self.shelf_args(line))

    def scout(self) -> None:
        share = max(2, min(5, (self.max_steps // 4) // max(1, len(self.lines))))
        for line in self.lines:
            if self.search.get("shelf_arg"):
                line.browsed.add(line.shelf or line.name)
                self.run_search(line, self.shelf_args(line))
            for query in query_ladder(line.name, self.phrases(line), line.head)[:2]:
                if not line.ids:
                    self.run_search(line, self.shelf_args(line, query))
            if not line.ids:
                self.run_search(line, self.shelf_args(line, line.name))
            self.browse_label(line)
            for ident in self.unread_by_fit(line)[:share]:
                self.open_listing(ident)
            self.set_aside_unanswerable(line)

    def want_words(self, line: Line) -> set[str]:
        words: set[str] = set()
        for need in line.live():
            if not need["pointer"]:
                words |= _words(phrase_of(need))
        return words

    def set_aside_unanswerable(self, line: Line) -> None:
        """A stated requirement that no listing read can speak to cannot decide between listings: it came from the story,
        not from the shelf, so it stops blocking fits. That takes at least three records judged, all left open, and no
        related key or such value among them: a store can file a requirement under a key of another name, and any
        verdict other than open shows it does. What the shopper revealed when asked is never set aside."""
        opened = [i for i in line.ids if self.pages.get(i)]
        if len(opened) < 3:
            return
        keys: set[str] = set()
        values: set[str] = set()
        for ident in opened:
            for key, value in self.facts_of(ident).items():
                keys |= _words(key)
                values |= _words(value)
        if not keys:
            return  # pages whose attributes could not be found say nothing about any requirement
        for need in line.live():
            if need["told"] or need["pointer"] or need.get("checked"):
                continue
            said = [self.verdicts[(i, need["text"])][0] for i in opened if (i, need["text"]) in self.verdicts]
            if len(said) < 3:
                continue
            need["checked"] = True
            if all(v == "unknown" for v in said) and not (_words(need["prop"]) & keys) and not (_words(phrase_of(need)) & values):
                need["live"] = False
                self.trace.append({"kind": "set_aside", "need": need["text"][:100]})

    def phrases(self, line: Line) -> list[str]:
        """Wanted values to search for, what the shopper revealed first."""
        ordered = sorted((n for n in line.live() if not n["pointer"]), key=lambda n: not n["told"])
        found = [phrase_of(n) for n in ordered]
        return [p for p in found if p]

    def label_fits(self, ident: str, line: Line) -> bool:
        """Whether a listing's label allows it. A store may call an item something the request does not, so sharing no
        word rejects a label only while some other label does share one; otherwise the reading of kind decides."""
        label = _words(self.category(ident))
        if not label or line.wide:
            return True
        if line.exact:
            return label == _words(line.shelf)
        return bool(label & (_words(line.name) | _words(line.shelf))) or not line.related

    def unread_by_fit(self, line: Line) -> list[str]:
        """Unread listings of a line in the order worth reading for its pick rule."""
        wants = self.want_words(line)
        rows = [i for i in line.ids if i not in self.pages and i not in self.kind_bad and self.in_stock(i) and self.affordable(i, line) and self.label_fits(i, line)]

        def overlap(ident: str) -> int:
            return len(wants & _words(self.record_text(ident)))

        # a row that shows the wanted words is far likelier to fit than one that does not, whatever its price, and a
        # row a keyword search returned has matched some of them, so both are read before a row merely browsed
        if line.pick == "cheapest":
            cost = lambda i: self.price(i) if self.price(i) is not None else 1e12
            return sorted(rows, key=lambda i: (-overlap(i), 0 if line.word_hits.get(i) else 1, cost(i)))
        return sorted(rows, key=lambda i: (-overlap(i), -line.word_hits.get(i, 0)))

    def key_table(self, line: Line) -> str:
        seen: dict[str, dict[str, Any]] = {}
        read = [i for i in line.ids if self.pages.get(i)]
        for ident in read:
            for key, value in self.facts_of(ident).items():
                slot = seen.setdefault(key, {"n": 0, "values": []})
                slot["n"] += 1
                if str(value)[:30] not in slot["values"]:
                    slot["values"].append(str(value)[:30])
        ordered = sorted(seen.items(), key=lambda kv: -kv[1]["n"])
        return f"{len(read)} listings read\n" + "\n".join(f"  {k} (on {s['n']}, {len(s['values'])} values): {' | '.join(s['values'][:4])}" for k, s in ordered[:40])

    def facts_of(self, ident: str) -> dict[str, Any]:
        """A record's attributes: its largest dict, else a list of name and value pairs, else the page's own short plain
        values. A lone nested entry may be about something else, so it does not stand for the whole record."""
        page = self.pages.get(ident) or {}
        best: dict[str, Any] = {}
        for value in page.values():
            if isinstance(value, dict) and len(value) > len(best):
                best = value
        if len(best) >= 2:
            return best
        pairs = {}
        for value in page.values():
            for row in _dicts(value):
                parts = list(row.values())
                if len(parts) == 2 and isinstance(parts[0], str):
                    pairs[parts[0]] = parts[1]
        if len(pairs) >= 2:
            return pairs
        known = {self.shape.get(k) for k in ("id", "title", "price", "stock", "category", "seller")}
        plain = {k: v for k, v in page.items() if k not in known and isinstance(v, (str, int, float)) and len(str(v)) <= 80}
        return {**plain, **best}

    def rank_properties(self) -> None:
        body = "\n\n".join(f"ITEM {l.number}: {l.name}\nSTATED NEEDS: {'; '.join(n['text'] for n in l.live()) or 'none'}\nKEYS SEEN: {self.key_table(l)}" for l in self.lines)
        hints = _dump(self.brief.get("hints") or [])
        got = self.read(RANK + self.request + "\n\nHINTS: " + hints + "\n\n" + body, self.careful(), 1800) or {}
        for entry in _dicts(got.get("items")):
            line = self.line_of(entry.get("item"))
            for raw in _dicts(entry.get("properties")):
                name = re.split(r"\s+(?:or|vs)\s+", _plain(raw.get("name")))[0]
                p = _number(raw.get("p"))
                if name and p is not None:
                    line.beliefs[name] = self.weigh(line, list(raw.get("keys") or []), min(max(p, 0.03), 0.9))
                    line.keys_of[name] = [_plain(k) for k in raw.get("keys") or []]
        for line in self.lines:
            if not line.beliefs:
                # the model gave nothing: fall back on the keys read, most carried first
                table = {}
                for ident in line.ids:
                    for key in self.facts_of(ident):
                        table[_plain(key)] = table.get(_plain(key), 0) + 1
                settled = _words(" ".join(n["prop"] for n in line.live()))
                for key, count in sorted(table.items(), key=lambda kv: -kv[1])[:12]:
                    if not _words(key) <= settled:
                        line.beliefs[key] = 0.1 + 0.02 * count
            self.trace.append({"kind": "beliefs", "item": line.number, "top": sorted(((round(p, 2), n) for n, p in line.beliefs.items()), reverse=True)[:10]})

    def weigh(self, line: Line, keys: list[Any], p: float) -> float:
        """Adjust the model's estimate by the shelf's evidence: a property on few listings or with one value among
        those read is less likely to separate listings, so less likely to be held back."""
        wanted = {_plain(k) for k in keys}
        read = [i for i in line.ids if self.pages.get(i)]
        if not read:
            return p
        if not wanted:
            return p * 0.5  # no listing read carries it, so only the story speaks for it
        values: set[str] = set()
        carried = 0
        for ident in read:
            hits = [str(v) for k, v in self.facts_of(ident).items() if _plain(k) in wanted]
            carried += bool(hits)
            values.update(hits)
        factor = 0.5 + 0.5 * carried / len(read)
        if len(values) == 1 and carried >= 3:
            factor *= 0.6
        if carried >= 3 and len(values) >= carried and sum(map(len, values)) / len(values) > 20:
            factor *= 0.4  # long text differing on every listing reads as an identifier or an option list
        return p * factor

    # ---- the interview ----
    def next_message(self) -> list[tuple[Line, list[str]]]:
        free = self.messages_free()
        live = [l for l in self.lines if any(p >= MIN_BELIEF for p in l.beliefs.values())]
        if free <= 0 or not live:
            return []
        fresh = [l for l in live if l.asked == 0]
        if len(fresh) > free:
            # more items than messages: this one message covers a fair share of them
            group = fresh[:-(free - 1)] if free > 1 else fresh
        elif fresh:
            group = fresh[:1]
        else:
            group = [max(live, key=lambda l: line_gain(l.beliefs, self.name_cap))]
        cap = max(2, self.name_cap // len(group)) if self.name_cap > 1 else 1
        parts = []
        for line in group:
            names = self.names_in_order(line, cap)
            if names:
                parts.append((line, names))
        return self.trim_to_length(parts)

    def names_in_order(self, line: Line, cap: int) -> list[str]:
        """The likeliest names. A reply that answered nothing may mean a name near the front drew it, so what stood
        behind then goes in front now: untried names first, then the last message's names reversed."""
        names = [n for n, p in sorted(line.beliefs.items(), key=lambda kv: -kv[1]) if p >= MIN_BELIEF][:cap]
        if not line.last_silent:
            return names
        return [n for n in names if n not in line.last_names] + [n for n in reversed(line.last_names) if n in names]

    def trim_to_length(self, parts: list[tuple[Line, list[str]]]) -> list[tuple[Line, list[str]]]:
        spec = ((self.schemas.get(self.roles.get("ask", "")) or {}).get("properties") or {}).get(self.roles.get("ask_text", "")) or {}
        limit = spec.get("maxLength") if isinstance(spec.get("maxLength"), int) else 800
        while len(question_text([(l.label(), n) for l, n in parts])) > limit and any(len(n) > 1 for _, n in parts):
            longest = max(parts, key=lambda part: len(part[1]))
            longest[1].pop()
        return parts

    def interview(self) -> None:
        if not self.roles.get("ask") or not self.roles.get("ask_text"):
            return
        while not self.finished and self.steps_left() > self.tail() and self.time_left() > HURRY:
            parts = self.next_message()
            if not parts:
                return
            text = question_text([(l.label(), n) for l, n in parts])
            self.msg_used += 1
            reply = self.call(self.roles["ask"], {self.roles["ask_text"]: text})
            words = _dump(reply)
            self.trace.append({"kind": "ask", "q": text, "a": words[:300]})
            if self.take_reply(parts, text, words) == "refused":
                # the store refused the message: its real limit is what was used before this one
                self.msg_used -= 1
                self.msg_limit = self.msg_used
                return

    def take_reply(self, parts: list[tuple[Line, list[str]]], text: str, words: str) -> Any:
        """Apply a reply to the beliefs and needs. Returns how the reply was read."""
        items = "\n".join(f"item {l.number}: {l.name}" for l in self.lines)
        got = self.read(READ_REPLY + f"PRODUCTS:\n{items}\n\nMESSAGE: {text}\n\nREPLY: {words[:900]}", [LIGHT, HEAVY], 500) or {}
        kind = got.get("kind")
        if kind == "refused":
            return kind
        for line, _ in parts:
            line.asked += 1
        if kind == "too_many":
            self.name_cap = max(1, self.name_cap // 2)
            return kind
        needs = _dicts(got.get("needs"))
        found: dict[int, list[dict[str, Any]]] = {}
        for raw in needs:
            line = self.line_of(raw.get("item") or parts[0][0].number)
            added = line.add_need(raw, True)
            if added:
                found.setdefault(line.number, []).append(added)
        budget = _number(got.get("budget"))
        if budget is not None:
            self.budget = budget
        self.drop_needs(got.get("drop"))
        for line, names in parts:
            answered = found.get(line.number, [])
            hit = None
            for need in answered:
                place = self.match_name(line, names, need["prop"])
                hit = place if hit is None else (hit if place is None else min(hit, place))
                self.forget_property(line, need["prop"])
            # a preference on a property this message did not list says nothing about the names it did list
            if hit is not None or not answered:
                update_beliefs(line.beliefs, names, hit)
            line.last_names = list(names)
            line.last_silent = not answered
            # the front name of an unanswered message was almost surely heard and holds nothing, and listed again
            # it could only stand in front of a name that does
            if not answered and names and names[0] in line.beliefs:
                line.beliefs[names[0]] = 0.0
        return kind

    def match_name(self, line: Line, names: list[str], prop: str) -> int | None:
        wanted = _words(prop)
        for place, name in enumerate(names):
            keys = [_words(k) for k in line.keys_of.get(name, [])] + [_words(name)]
            if any(k and (k <= wanted or wanted <= k) for k in keys):
                return place
        return None

    def forget_property(self, line: Line, prop: str) -> None:
        """A property is settled: no spelling of it is asked again."""
        wanted = _words(prop)
        for name in list(line.beliefs):
            keys = [_words(k) for k in line.keys_of.get(name, [])] + [_words(name)]
            if any(k and (k <= wanted or wanted <= k) for k in keys):
                line.beliefs[name] = 0.0

    # ---- requirements that point at other listings ----
    def resolve_pointers(self) -> None:
        for line in self.lines:
            for need in line.live():
                if need["pointer"] and not need["value"]:
                    self.follow(line, need)

    def follow(self, line: Line, need: dict[str, Any]) -> None:
        opened: list[str] = []
        shown: list[str] = []
        for _ in range(6):
            if self.steps_left() <= self.tail() + 2 or self.time_left() < HURRY:
                return
            pages = "\n".join(self.record_text(i)[:1200] for i in opened[-3:]) or "(none)"
            rows = "\n".join(self.record_text(i)[:200] for i in shown[-12:]) or "(none)"
            got = self.read(POINTER + f"REQUIREMENT: {need['text']}\n\nPAGES OPENED:\n{pages}\n\nLISTINGS SHOWN:\n{rows}", [LIGHT, HEAVY], 400) or {}
            value = str(got.get("value") or "").strip()
            if value:
                need["value"] = value
                self.trace.append({"kind": "pointer", "need": need["text"][:120], "value": value[:80]})
                return
            target = str(got.get("open") or "")
            if target:
                if target in opened:
                    return
                self.open_listing(target)
                if self.roles.get("offer") and self.roles.get("offer_id") and self.steps_left() > self.tail() + 2:
                    # a store that splits a listing keeps its remarks on the other page
                    extra = self.call(self.roles["offer"], {self.roles["offer_id"]: target})
                    if isinstance(extra, dict):
                        self.pages[target] = {**self.pages.get(target, {}), **extra}
                opened.append(target)
                continue
            words = str(got.get("search") or "").strip()
            if not words:
                return
            # a search for the other listing must not add to this item's own pool or page queue
            aside = Line(line.number, {"name": line.name, "shelf": str(got.get("shelf") or "")})
            self.run_search(aside, self.shelf_args(aside, words))
            shown += aside.ids

    # ---- verdicts ----
    def requirement_text(self, need: dict[str, Any]) -> str:
        return need["text"] + (f" (the page it points at gives: {need['value']})" if need["value"] else "")

    def judge(self, line: Line) -> None:
        self.set_aside_unanswerable(line)
        needs = line.live()
        ids = [i for i in line.ids if self.pages.get(i) and ((i, line.number) not in self.judged or any((i, n["text"]) not in self.verdicts for n in needs))]
        for start in range(0, len(ids), 5):
            batch = ids[start:start + 5]
            listed = "\n".join(f"{n + 1}. {self.requirement_text(need)}" for n, need in enumerate(needs)) or "(none)"
            extra = f"\n\"measure\" is the listing's value of: {line.measure}" if line.measure else ""
            records = "\n".join(f"ID {i}: {self.record_text(i)}" for i in batch)
            got = self.read(JUDGE + f"ITEM: {line.name} {('(category ' + line.shelf + ')') if line.shelf else ''}{extra}\n\nREQUIREMENTS:\n{listed}\n\nRECORDS:\n{records}", [LIGHT, HEAVY], 1800) or {}
            for entry in _dicts(got.get("listings")):
                ident = str(entry.get("id"))
                if ident not in batch:
                    continue
                self.judged.add((ident, line.number))
                # once the store's own label for the item is known, that label settles the kind
                if entry.get("kind_ok") is False and not line.exact:
                    self.kind_bad.add(ident)
                measure = _number(entry.get("measure"))
                if measure is not None:
                    self.measures[ident] = measure
                text = _plain(self.record_text(ident))
                for raw in _dicts(entry.get("verdicts")):
                    place = int(_number(raw.get("n")) or 0) - 1
                    if 0 <= place < len(needs):
                        verdict = str(raw.get("verdict"))
                        quote = _plain(raw.get("quote"))
                        # a yes needs words the record really holds
                        if verdict == "yes" and quote and quote not in text:
                            verdict = "unknown"
                        self.verdicts[(ident, needs[place]["text"])] = (verdict, str(raw.get("quote"))[:80])
                self.trace.append({"kind": "verdict", "id": ident, "kind_ok": entry.get("kind_ok"), "v": [self.verdicts.get((ident, n["text"]), ("?", ""))[0] for n in needs], "q": [self.verdicts.get((ident, n["text"]), ("", ""))[1][:40] for n in needs]})

    def verdict_of(self, ident: str, need: dict[str, Any]) -> str:
        """The listing's own category settles a need whose whole value it states, whatever other attributes say."""
        wanted = _words(need["value"] or need["want"])
        if wanted and wanted <= _words(self.category(ident)):
            return "yes"
        return self.verdicts.get((ident, need["text"]), ("unknown", ""))[0]

    def verdicts_of(self, ident: str, line: Line) -> list[str]:
        return [self.verdict_of(ident, n) for n in line.live()]

    def fits(self, line: Line) -> list[str]:
        out = [i for i in line.ids if self.pages.get(i) and i not in self.kind_bad and self.label_fits(i, line) and self.in_stock(i) and self.affordable(i, line) and all(v == "yes" for v in self.verdicts_of(i, line))]
        return self.by_pick(out, line)

    def by_pick(self, ids: list[str], line: Line) -> list[str]:
        cost = lambda i: self.price(i) if self.price(i) is not None else 1e12
        if line.pick == "most":
            return sorted(ids, key=lambda i: (-self.measures.get(i, -1e12), cost(i)))
        if line.pick == "least":
            return sorted(ids, key=lambda i: (self.measures.get(i, 1e12), cost(i)))
        return sorted(ids, key=cost)

    def score(self, ident: str, line: Line) -> float:
        """Rough chance that this listing is right for the line."""
        if ident in self.kind_bad or not self.label_fits(ident, line) or not self.in_stock(ident) or not self.affordable(ident, line):
            return 0.0
        if not self.pages.get(ident):
            return UNREAD_SCORE
        value = FIT_SCORE
        for verdict in self.verdicts_of(ident, line):
            value *= {"yes": 1.0, "unknown": UNKNOWN_SCORE}.get(verdict, NO_SCORE)
        return value

    # ---- the hunt ----
    def line_done(self, line: Line) -> bool:
        fits = self.fits(line)
        if not fits:
            return False
        if line.pick == "any":
            # any verified fit will do. A replacement is worth steps only if this one is refused at checkout, so the
            # steps wait until then instead of being spent on spares that may never be needed.
            return True
        if line.pick == "cheapest":
            best = self.price(fits[0])
            unread = self.unread_by_fit(line)
            cheaper = [i for i in unread if best is not None and (self.price(i) or 0) < best]
            arg = self.search.get("price_arg")
            if arg:
                return line.capped_at == best and not cheaper and not any(arg in page for page in line.more)
            return not cheaper and not line.more and not line.queries
        return not self.unread_by_fit(line) and not line.more

    def make_queries(self, line: Line) -> None:
        if line.queries_made:
            return
        line.queries_made = True
        needs = "; ".join(self.requirement_text(n) for n in line.live()) or "none"
        tried = f"\nTHESE FOUND NOTHING THAT FITS, SO TRY OTHER WORDS: {'; '.join(line.tried)}" if line.tried else ""
        got = self.read(QUERIES + f"ITEM: {line.name}\nREQUIREMENTS: {needs}{tried}", [LIGHT, HEAVY], 400) or {}
        written = [str(q) for q in got.get("queries") or [] if str(q).strip()]
        # searches from code come first and do not depend on a model answering; on a second pass only new words help
        planned = (written if line.tried else query_ladder(line.name, self.phrases(line), line.head) + written)
        line.queries = [q for q in dict.fromkeys(planned) if q not in line.tried][:8] or ([line.name] if not line.tried else [])
        line.tried += line.queries

    def look_below(self, line: Line) -> bool:
        """On a cheapest line, list the shelf under the best price found, so only cheaper listings are left to read."""
        arg = self.search.get("price_arg")
        fits = self.fits(line)
        if line.pick != "cheapest" or not arg or not fits:
            return False
        best = self.price(fits[0])
        if best is None or line.capped_at == best:
            return False
        line.capped_at = best
        # A store shows only so many pages, so a limit matching more than that hides some of its cheapest rows. Look
        # for the highest limit whose matches can all be shown: that is the widest band that hides nothing.
        pages = self.page_reach()
        low, high, limit = 0.0, best, best
        kept: dict[str, Any] | None = None
        for _ in range(5):
            if self.steps_left() <= self.tail() + 2:
                break
            before = len(line.more)
            result = self.run_search(line, {**self.shelf_args(line), arg: limit})
            if not isinstance(result, dict):
                break
            new_pages = line.more[before:] if len(line.more) > before else [p for p in line.more if p.get(arg) == limit]
            line.more = [p for p in line.more if p.get(arg) is None]
            total = _number(result.get(self.shape.get("total"))) if self.shape.get("total") else None
            shown = len(self.rows_in(result))
            if total is None:
                line.more += new_pages
                return True
            if total > shown * pages:
                high = limit
            else:
                low = limit
                kept = new_pages[0] if new_pages else None
            if limit == best and total <= shown * pages:
                break
            limit = round((low + high) / 2, 2)
            if high - low < max(1.0, best * 0.05):
                break
        if kept:
            line.more.append(kept)
        return True

    def page_reach(self) -> int:
        """How many pages of one search the store's schema allows; three when it does not say."""
        spec = ((self.schemas.get(self.search.get("shelf_tool", "")) or {}).get("properties") or {}).get(self.search.get("shelf_page_arg", "")) or {}
        top = spec.get("maximum")
        return top if isinstance(top, int) and spec.get("minimum") == 1 and top > 0 else 3

    def hunt(self, line: Line, allowance: int) -> None:
        stop = self.step + allowance
        self.make_queries(line)
        while not self.finished and self.step < stop and self.steps_left() > self.tail() and self.time_left() > HURRY:
            self.judge(line)
            self.browse_label(line)
            if self.look_below(line):
                continue
            if self.line_done(line):
                return
            # after the first search of this hunt, read a few listings before searching again
            unread = self.unread_by_fit(line)
            # later pages under a price limit hold the cheapest candidates, ahead of any new keyword search
            limited = [page for page in line.more if self.search.get("price_arg") in page]
            if not unread and limited:
                line.more.remove(limited[0])
                self.run_search(line, limited[0])
                continue
            if line.queries and (not unread or len([i for i in line.ids if self.pages.get(i)]) % 4 == 3):
                self.run_search(line, self.shelf_args(line, line.queries.pop(0)))
                continue
            if unread:
                for ident in unread[:2]:
                    if self.step < stop and self.steps_left() > self.tail():
                        self.open_listing(ident)
                continue
            if line.more:
                self.run_search(line, line.more.pop(0))
                continue
            if line.queries:
                self.run_search(line, self.shelf_args(line, line.queries.pop(0)))
                continue
            if not self.fits(line) and not line.rethought:
                # steps remain and nothing fits: one fresh set of searches before giving up
                line.rethought = True
                line.queries_made = False
                self.make_queries(line)
                continue
            if not self.fits(line) and not line.wide and not line.exact:
                # the labels near the item's words gave no fit and none is its whole name, so any label may be read
                # before giving up, with the reading of kind deciding
                line.wide = True
                if self.unread_by_fit(line):
                    continue
            return
        self.judge(line)

    def hunt_all(self) -> None:
        rounds = 0
        while rounds < 2 and not self.finished and self.time_left() > HURRY:
            rounds += 1
            open_lines = [l for l in self.lines if not self.line_done(l)]
            if not open_lines:
                return
            room = self.steps_left() - self.tail()
            if room <= 0:
                return
            for index, line in enumerate(open_lines):
                self.hunt(line, max(2, room // max(1, len(open_lines) - index)))
                room = self.steps_left() - self.tail()

    # ---- the finish ----
    def choose(self) -> dict[str, int]:
        """The best listing per line, under the budget, each line's backups kept for later swaps."""
        picks: dict[int, str] = {}
        for line in self.lines:
            taken = set(picks.values())
            fits = [i for i in self.fits(line) if i not in taken]
            if fits:
                picks[line.number] = fits[0]
                continue
            pool = [i for i in line.ids if i not in taken]
            best = max(pool, key=lambda i: self.score(i, line), default=None)
            if best is not None and self.score(best, line) > 0:
                picks[line.number] = best
        if self.budget is not None and not self.budget_each:
            for _ in range(len(self.lines) * 4):
                total = sum((self.price(i) or 0) * self.line_of(n).quantity for n, i in picks.items())
                if total <= self.budget:
                    break
                swaps = []
                for number, ident in picks.items():
                    line = self.line_of(number)
                    for other in self.fits(line):
                        if other in picks.values():
                            continue
                        saved = ((self.price(ident) or 0) - (self.price(other) or 0)) * line.quantity
                        if saved > 0:
                            swaps.append((saved, number, other))
                if not swaps:
                    break
                _, number, other = max(swaps)
                picks[number] = other
        return {i: self.line_of(n).quantity for n, i in picks.items()}

    def press_budget(self) -> None:
        """The cart breaks the total budget. A line that takes any fit stopped at its first, so cheaper fits may be
        unread: those lines now look for their cheapest, dearest first, and spare steps go to them."""
        self.pressed = True
        lines = [l for l in self.lines if l.pick == "any"]
        lines.sort(key=lambda l: -max((self.price(i) or 0 for i in self.fits(l)[:1]), default=0))
        for index, line in enumerate(lines):
            line.pick = "cheapest"
            room = self.steps_left() - self.tail()
            if room > 0 and self.time_left() > HURRY:
                self.hunt(line, max(2, room // (len(lines) - index)))

    def order_chance(self, cart: dict[str, int]) -> float:
        if not cart or len(cart) < len(self.lines):
            return 0.0
        chance = NO_SCORE if self.over_budget(cart) else 1.0
        for line in self.lines:
            chance *= max((self.score(i, line) for i in cart if i in line.ids), default=0.0)
        return chance

    def end_chance(self) -> float:
        missing = [l for l in self.lines if not self.fits(l)]
        if not missing:
            return END_BASE
        value = END_OPEN
        for line in missing:
            exhausted = line.searches >= 3 and not line.queries and not line.more and not self.unread_by_fit(line)
            if exhausted:
                told = any(n["told"] for n in line.live())
                value = max(value, END_TOLD if told else END_STATED)
        return value

    def set_cart(self, wanted: dict[str, int]) -> None:
        add, remove = self.roles.get("add") or {}, self.roles.get("remove") or {}
        if add.get("lines"):
            # this store sets the whole cart in one call
            if add.get("tool") and self.cart != wanted:
                lines = [{add.get("id"): ident, **({add["qty"]: count} if add.get("qty") else {})} for ident, count in wanted.items()]
                reply = self.call(add["tool"], {**(add.get("fixed") or {}), add["lines"]: lines})
                if not self.refused(add["tool"], reply):
                    self.cart = dict(wanted)
            return
        for ident, count in list(self.cart.items()):
            if ident not in wanted and remove.get("tool"):
                args = {**(remove.get("fixed") or {}), remove.get("id"): ident}
                if remove.get("qty"):
                    args[remove["qty"]] = count
                self.call(remove["tool"], args)
                self.cart.pop(ident, None)
        for ident, count in wanted.items():
            have = self.cart.get(ident, 0)
            if have < count and add.get("tool"):
                args = {**(add.get("fixed") or {}), add.get("id"): ident}
                if add.get("qty"):
                    args[add["qty"]] = count - have
                reply = self.call(add["tool"], args)
                if not self.refused(add["tool"], reply):
                    self.cart[ident] = count

    def check_cart(self, cart: dict[str, int]) -> bool:
        """A last independent read. Returns False when it found a requirement a record contradicts."""
        if self.cost >= COST_SOFT + 0.015:
            return True
        lines = []
        for line in self.lines:
            listed = "; ".join(f"{n + 1}. {self.requirement_text(need)}" for n, need in enumerate(line.live())) or "none"
            for ident in cart:
                if ident in line.ids:
                    lines.append(f"ITEM {line.name}, requirements: {listed}\nRECORD {ident}: {self.record_text(ident)}")
        got = self.read(CHECK + "\n\n".join(lines), [HEAVY, LIGHT], 700) or {}
        wrong = False
        for raw in _dicts(got.get("wrong")):
            ident = str(raw.get("id"))
            for line in self.lines:
                needs = line.live()
                place = int(_number(raw.get("n")) or 0) - 1
                if ident in line.ids and 0 <= place < len(needs):
                    self.verdicts[(ident, needs[place]["text"])] = ("no", str(raw.get("why"))[:80])
                    wrong = True
        return not wrong

    def gate(self, stage: str) -> None:
        steps = self.steps_of_procedure(stage)
        if not steps:
            return
        ending = {self.roles.get("buy"), self.roles.get("end")}
        tools = "\n".join(f"- {t['function']['name']}: {t['function'].get('description', '')} | {_dump(t['function'].get('parameters') or {})}" for t in self.tools if t["function"]["name"] not in ending)
        for _ in range(3):
            if self.steps_left() <= 2 or self.time_left() < LAST_MODEL + 10:
                return
            got = self.read(GATE + "PROCEDURES:\n" + "\n".join(steps) + f"\n\nCART: {_dump(self.cart)}\nMESSAGES LEFT: {self.msg_limit - self.msg_used}\n\nRECORD:\n" + "\n".join(self.done_log[-30:]) + "\n\nTOOLS:\n" + tools, [LIGHT, HEAVY], 400) or {}
            call = got.get("call")
            if not isinstance(call, dict) or call.get("tool") not in self.schemas or call.get("tool") in ending:
                return
            if call["tool"] == self.roles.get("ask"):
                self.msg_used += 1
            self.call(call["tool"], call.get("args") if isinstance(call.get("args"), dict) else {})

    def find_gone(self) -> None:
        """Something picked changed and the note did not say what: read the cart's lines afresh."""
        gone_note = self.pending_gone
        self.pending_gone = self.pending_price = False
        tool = self.roles.get("offer") or self.roles.get("facts")
        arg = self.roles.get("offer_id") if self.roles.get("offer") else self.roles.get("facts_id")
        for ident in list(self.cart):
            if tool and arg and self.steps_left() > len(self.lines) + 3:
                result = self.call(tool, {arg: ident})
                if isinstance(result, dict) and result:
                    self.pages[ident] = {**self.pages.get(ident, {}), **result}
                    self.rows[ident] = {**self.rows.get(ident, {}), **result}
        # records may not show stock at all; with one line in the cart, that line is the one the note meant
        if gone_note and len(self.cart) == 1 and all(self.in_stock(i) for i in self.cart):
            self.gone.update(self.cart)

    def try_order(self, cart: dict[str, int]) -> bool:
        self.changed = False
        self.set_cart(cart)
        if self.finished:
            return True
        self.gate("before_order")
        if self.roles.get("cart") and self.steps_left() > 4:
            self.call(self.roles["cart"], {})
        if self.changed and self.steps_left() > len(self.lines) + 3:
            # the task changed while the cart was being set: solve it again before ordering, while steps allow
            if self.pending_gone or self.pending_price:
                self.find_gone()
            return False
        self.buy_tries += 1
        notes_before = self.notes_seen
        result = self.call(self.roles.get("buy", ""), dict(self.roles.get("buy_args") or {}))
        if self.finished:
            self.trace.append({"kind": "ordered", "cart": dict(self.cart), "step": self.step})
            return True
        # refused: unless a note already explained it, the reply's own words are read as one
        if result is not None and self.notes_seen == notes_before:
            self.on_note(_dump(result)[:1500])
        if self.pending_gone or self.pending_price:
            self.find_gone()
        return False

    def end_without_order(self) -> None:
        self.gate("before_end")
        notes_before = self.notes_seen
        result = self.call(self.roles.get("end", ""), dict(self.roles.get("end_args") or {}))
        if not self.finished and result is not None and self.notes_seen == notes_before:
            self.on_note(_dump(result)[:1500])
        self.trace.append({"kind": "ended", "step": self.step, "finished": self.finished})

    def finish(self) -> None:
        for _ in range(5):
            if self.finished or self.steps_left() <= 0:
                return
            for line in self.lines:
                self.judge(line)
            cart = self.choose()
            if self.over_budget(cart) and not self.pressed and self.steps_left() > self.tail() and self.time_left() > HURRY:
                self.press_budget()
                continue
            # the check costs money and time but no steps, so a short step count is no reason to skip it
            if self.time_left() > HURRY / 2 and not self.check_cart(cart):
                cart = self.choose()
            order, end = self.order_chance(cart), self.end_chance()
            self.trace.append({"kind": "finish", "order": round(order, 3), "end": round(end, 3), "cart": cart, "fits": {l.number: len(self.fits(l)) for l in self.lines}})
            if order >= end and cart and self.roles.get("buy"):
                if self.try_order(cart):
                    return
                # a refusal changes the task: spare steps go to a line that lost every fit, or whose best price moved
                if self.steps_left() > self.tail() and self.time_left() > HURRY:
                    for line in self.lines:
                        if not self.fits(line) or line.pick == "cheapest":
                            self.hunt(line, self.steps_left() - self.tail())
                continue
            if self.roles.get("end"):
                # the store can refuse an ending too: then its note is read like any other, and the task solved again
                self.end_without_order()
                if self.finished:
                    return
                continue
            if cart and self.try_order(cart):
                return
        # no finish was accepted in those rounds: never leave without one
        if not self.finished:
            self.call(self.roles.get("buy", "") if self.cart else self.roles.get("end", ""), dict((self.roles.get("buy_args") if self.cart else self.roles.get("end_args")) or {}))

    def run(self) -> list[dict[str, Any]]:
        if not self.finished:
            self.read_task()
            self.scout()
            self.rank_properties()
            self.interview()
            self.resolve_pointers()
            self.hunt_all()
            self.finish()
        self.trace.append({"kind": "end", "finished": self.finished, "steps": self.step, "cost": round(self.cost, 5), "calls": self.calls, "seconds": round(time.monotonic() - self.t0, 1)})
        return self.trace


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        episode = Sieve(problem_data)
    except Exception as exc:
        return [{"kind": "crash", "error": f"{type(exc).__name__}: {exc}"[:400]}]
    try:
        return episode.run()
    except Exception as exc:
        episode.trace.append({"kind": "crash", "error": f"{type(exc).__name__}: {exc}"[:400]})
        try:
            if not episode.finished:
                if episode.cart and episode.roles.get("buy"):
                    episode.call(episode.roles["buy"], dict(episode.roles.get("buy_args") or {}))
                elif episode.roles.get("end"):
                    episode.call(episode.roles["end"], dict(episode.roles.get("end_args") or {}))
        except Exception as again:
            episode.trace.append({"kind": "crash", "error": f"{type(again).__name__}: {again}"[:400]})
        return episode.trace


__all__ = ["agent_main"]
