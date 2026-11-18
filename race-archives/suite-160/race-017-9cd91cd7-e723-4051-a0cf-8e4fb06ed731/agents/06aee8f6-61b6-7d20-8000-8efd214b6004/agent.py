"""Meridian: a model-led shopping agent for generated worlds.

Nothing about a world is assumed. A language model reads the world's tool list, the shopper's request, every product page
and every shopper message; the code keeps the state, spends the step budget, checks that the evidence the model quotes is
really on the page, and makes sure the episode ends with an order or a decline.
"""

from __future__ import annotations

import json
import queue
import re
import threading
import time
from typing import Any

from src.agent.proxy_client import ProxyClient

_MODELS = ("openai/gpt-6-luna", "deepseek/deepseek-v3.2")
_SYSTEM = ("You are the reasoning core of an online shopping assistant. You read a shopper's request together with catalog "
           "evidence and answer with exactly one JSON object and nothing else.")
_OPEN_BY = 170.0  # seconds into the task after which no new pages are opened
_STOP_BY = 215.0  # seconds into the task after which the episode is brought to its end at once
_COMMIT_STEPS = 8  # steps kept back for ordering, and for finding another listing if the pick sells out at the last moment
_LINE_STEPS = 3  # further steps kept back for each product in the order
_LAST_STEPS = 4  # the fewest steps with which looking further still leaves room to order what is found
_PER_JUDGE = 6
_UNSETTLED_AFTER = 9  # pages read before a need no page speaks to is set aside
_FIRST_PAGES = 8  # pages read before the shopper is asked: their property names are what the questions are made of
_RESULT_PAGES = 12  # result pages read for a product at most, the first page of each search included
_PAGES_READ = 24  # product pages read for a product at most: every page read also costs a model's reading
_lock = threading.Lock()
_shared: dict[str, Any] = {}  # what was read about a world's tools holds for every task of that world


def _json_block(text: str) -> dict | None:
    text = (text or "").strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    chunks = [fenced.group(1)] if fenced else []
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        chunks.append(text[start:end + 1])
    for chunk in chunks:
        try:
            data = json.loads(chunk)
        except ValueError:
            continue
        if isinstance(data, dict):
            return data
    return None


def _squash(text: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").casefold()).strip()


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"-?\d+(?:\.\d+)?", str(value or "").replace(",", ""))
    return float(match.group()) if match else None


def _sentences(value: Any) -> bool:
    return (isinstance(value, list) and bool(value)
            and all(isinstance(x, str) and len(x.split()) >= 3 for x in value))


def _split_reply(observation: Any) -> tuple[Any, list[str]]:
    """A tool's own result, and whatever the shopper said alongside it. Worlds can wrap a result together with lists of
    sentences under keys of their own, more than once."""
    heard: list[str] = []
    while isinstance(observation, dict):
        rest = {k: v for k, v in observation.items() if not _sentences(v)}
        if len(rest) == len(observation):
            break
        for key, value in observation.items():
            if key not in rest:
                heard.extend(x for x in value if x not in heard)
        observation = next(iter(rest.values())) if len(rest) == 1 and isinstance(next(iter(rest.values())), (dict, str, list)) else rest
    return observation, heard


def _spoken(result: Any) -> list[str]:
    """The words of a reply, whether it comes as plain text or as text under a key of the world's own."""
    if isinstance(result, str):
        return [result]
    if isinstance(result, dict):
        return [v for v in result.values() if isinstance(v, str) and len(v.split()) >= 3]
    return []


class Brain:
    def __init__(self, began: float):
        self.began = began
        self.log: list[dict] = []
        self.refused: set[str] = set()
        # a task has limited time: one call waits little, and a request turned back for load is tried soon again
        self.client = ProxyClient(timeout=35, max_retries=2, rate_limit_retry_delay=3)

    def _attempt(self, model: str, prompt: str, max_tokens: int, kind: str) -> dict | None:
        body: dict[str, Any] = {"model": model, "temperature": 0, "max_tokens": max_tokens,
                                "messages": [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": prompt}]}
        body["reasoning"] = {"enabled": False} if "glm" in model.casefold() else {"effort": "low"}
        began, status = time.monotonic(), None
        try:
            result = self.client.post_verbose("/inference/chat/completions", json_data=body)
            reply = result.data
            status = (result.error or {}).get("status") if reply is None else None
            data = _json_block(reply["choices"][0]["message"]["content"] or "")
        except Exception:
            if status in (400, 401, 403, 404):
                self.refused.add(model)
            self.log.append({"kind": kind, "model": model, "ok": False, "status": status})
            return None
        use = reply.get("usage") or {}
        self.log.append({"kind": kind, "model": model, "ok": data is not None, "secs": round(time.monotonic() - began, 1),
                         "pt": use.get("prompt_tokens"), "ct": use.get("completion_tokens"), "cost": use.get("cost")})
        return data

    def ask(self, prompt: str, kind: str, max_tokens: int = 1200) -> dict:
        for model in [m for m in _MODELS if m not in self.refused] or list(_MODELS):
            if time.monotonic() - self.began > _STOP_BY:
                break
            data = self._attempt(model, prompt, max_tokens, kind)
            if data is not None:
                return data
        return {}

    def ask_many(self, prompts: list[str], kind: str, max_tokens: int = 1800) -> list[dict]:
        answers: list[dict] = [{} for _ in prompts]
        work: queue.Queue = queue.Queue()
        for i in range(len(prompts)):
            work.put(i)

        def worker() -> None:
            while True:
                try:
                    i = work.get_nowait()
                except queue.Empty:
                    return
                answers[i] = self.ask(prompts[i], kind, max_tokens)
        threads = [threading.Thread(target=worker, daemon=True) for _ in range(min(4, len(prompts)))]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        return answers


# --------------------------------------------------------------------------------------
# prompts
# --------------------------------------------------------------------------------------

def prompt_tools(tools: list) -> str:
    return f"""An online shop offers these tools (JSON, OpenAI function format):
{json.dumps(tools, ensure_ascii=False)}

Say which tool does each job. Use the exact tool and argument names, and only ones listed. "fixed" holds arguments that
select this job when one tool does several jobs (for example a mode or an action value).
- "keyword": searching the catalog by typed words.
- "browse": listing the catalog by structured fields such as a department and a price ceiling (null if there is no such way).
- "facts": the property list of a single item (null if none).
- "page": a single item's full listing (null if none).
- "many": showing the listings of several items in one call, side by side (null if none). "most" is how many one call takes
  when the tool says so, otherwise null.
- "message": writing to the shopper.
- "add" and "remove": putting a single item into, and taking it out of, the cart or order (remove null if none).
- "cart": showing the cart (null if none).
- "order": placing the final order.
- "decline": finishing without an order.
Return JSON: {{"keyword": {{"tool": "", "text": "argument for the words", "page": "page argument or null",
"ceiling": "price ceiling argument or null", "department": "department argument or null",
"page_kind": "number, offset, or token when the value comes from the previous reply", "fixed": {{}}}},
"browse": {{"tool": "", "department": "", "page": "or null", "ceiling": "or null", "page_kind": "number or offset", "fixed": {{}}}},
"facts": {{"tool": "", "id": ""}}, "page": {{"tool": "", "id": ""}}, "many": {{"tool": "", "ids": "", "most": null}}, "message": {{"tool": "", "text": ""}},
"add": {{"tool": "", "id": "", "quantity": "or null", "fixed": {{}}}}, "remove": {{"tool": "", "id": "", "fixed": {{}}}},
"cart": {{"tool": ""}}, "order": {{"tool": ""}}, "decline": {{"tool": "", "text": "or null"}}}}"""


def prompt_row(row: dict) -> str:
    return f"""One result row from an online shop's search (JSON):
{json.dumps(row, ensure_ascii=False)[:1500]}

Name the key that holds each thing, or null when the row has none.
Return JSON: {{"id": "", "price": "", "stock": "", "title": "", "category": "", "brand": "", "description": ""}}"""


def prompt_paging(reply: dict) -> str:
    return f"""An online shop's search replied with this, the result rows left out (JSON):
{json.dumps(reply, ensure_ascii=False)[:800]}

Name the key whose value asks for the next page of results, and the key holding how many results there are in all; null for
either when the reply has none.
Return JSON: {{"next": "", "total": ""}}"""


def prompt_request(query: str, opening: Any) -> str:
    seen = f"\nWhat the shop shows at the start: {json.dumps(opening, ensure_ascii=False)}" if opening else ""
    return f"""A shopper's request (verbatim, the sentences may come in any order and much of it is small talk):
<<<
{query}
>>>{seen}

Read what the shopper firmly asks for.
- "wants": one entry per distinct product to buy. A name joined with "&", "/" or "or" under one requirement is one product.
  Each entry: "kind" (the kind of goods, whole, as the shopper names it), "quantity" (1 unless stated), "needs" and "queries".
- "needs": each requirement the shopper states on a property of that product: "property" (a short name for it), "value" as
  stated, "relation" ("is", "at least", "at most", "one of" or "not"; "not" is for what the shopper wants left out). One entry
  for a property: a value said again in looser words ("a basic style, just a plain one") is the first value alone. When the
  shopper asks for an absence a listing would state as a value, give that value with "is" ("no sleeves" is a sleeve length of
  sleeveless). When the value must be read from another listing the request describes ("the same size as the lamp that shop
  sells"), give the whole description in "refers_to", with "relation" saying how the value relates to it. Only properties of
  the item count: the words of the kind, the price and being in stock are checked apart, and the story around the request
  (where the item goes, who it is for, when it should arrive) is not a need unless it is said to be a property of the item.
- "queries": for that product, five keyword searches from specific to broad, each two to four words: a required value with
  the kind of goods, a required value with the name of its property, then the kind of goods alone. No prices or stock words.
- "price_limit": the highest price allowed, as a number exactly as the shopper writes it; "price_scope": "total" when it covers
  the whole order, "each" when it is per item.
- "objective": "lowest price", "most" or "least" when the shopper asks for that among the items that qualify, otherwise "".
  "measure": the property measured for "most" or "least", otherwise "". Saying money is tight is not an objective.
- "message_budget": how many messages the shopper allows, 0 when the request offers none, null when it says nothing.
- "if_none": what the shopper wants done when nothing qualifies.
- "procedures": steps the request requires besides finding and buying, such as getting approval, telling or showing the shopper
  something, checking something with them, reading something first or using a particular tool. Each: "step" (in the shopper's
  words), "when" ("choose" before products are chosen, "order" right before ordering, "end" just before finishing with no order),
  "messages" (messages to the shopper it takes). Requirements on items, limits on messages and steps, and what to do when
  nothing qualifies are not procedures. Empty when none.
Return JSON: {{"wants": [{{"kind": "", "quantity": 1, "needs": [{{"property": "", "value": "", "relation": "is", "refers_to": ""}}],
"queries": []}}], "price_limit": 0, "price_scope": "total", "objective": "", "measure": "", "message_budget": 0, "if_none": "",
"procedures": [{{"step": "", "when": "order", "messages": 1}}]}}"""


def prompt_shelf(wanted: Any, counts: dict[str, int]) -> str:
    return f"""A shopper wants: {json.dumps(wanted, ensure_ascii=False)}
The shop's departments among the search results so far (with how many results): {json.dumps(counts, ensure_ascii=False)}

Which one department is where this kind of goods belongs? null if none of them is. A department for accessories, parts or
supplies of the item, or another kind of goods sharing a word with it, is not it.
Return JSON: {{"department": "exact name or null"}}"""


def prompt_shortlist(needs: dict, rows: list[str], limit: int) -> str:
    listed = "\n".join(rows)
    return f"""What the shopper needs: {json.dumps(needs, ensure_ascii=False)}

Search results (tag | price | department | title | description):
{listed}

The full property list can be opened for only {limit} of these. Results show only part of what a listing states, so a result
that does not mention a need can still meet it. Leave out only what its own words rule out or what is plainly another kind of
goods. Order the rest by how likely each is to meet every need.
Return JSON: {{"open": ["tag", "..."]}}"""


def prompt_heard(needs: Any, lines: list[str], situation: str) -> str:
    said = "\n".join(f"- {x}" for x in lines)
    return f"""What the shopper wants so far: {json.dumps(needs, ensure_ascii=False)}
Situation: {situation}
The shopper just said:
{said}

Apply only what these lines state. Lines saying the shopper has no preference add nothing.
- "add": a requirement the shopper now states, as {{"item": the kind of goods it is about, "property", "value", "relation",
  "refers_to"}} ("relation": "is", "at least", "at most", "one of" or "not"; "refers_to": the description of another listing when
  the value must be read from that listing, otherwise empty).
- "drop": names of requirements in the list that a line explicitly gives up, naming the property. "No preference" answers to a
  question give up nothing. A property given up that is not in the list needs no entry.
- "price_limit": a new highest price if stated, else null. Give the number exactly as the shopper writes it, unconverted.
- "item_gone": true when a line says an item picked or suggested can no longer be had.
- "price_moved": true when a line says the price of a picked item changed.
- "not_understood": true when the shopper says they did not follow the message and asks for it again or more simply.
- "granted": true when the shopper agrees to what they were asked, false when they refuse it, null when neither.
- "stop": true when the shopper calls the purchase off.
Return JSON: {{"add": [], "drop": [], "price_limit": null, "item_gone": false, "price_moved": false, "not_understood": false,
"granted": null, "stop": false}}"""


def prompt_questions(query: str, needs: dict, names: list[str], count: int) -> str:
    return f"""A shopper's request (verbatim):
<<<
{query}
>>>
What is already required: {json.dumps(needs, ensure_ascii=False)}
Properties the pages of this kind of goods state: {json.dumps(names, ensure_ascii=False)}

Ask the shopper exactly {count} short question(s). A shopper answers a question about one property well and tends to wave off a
long list. Choose what is most likely to be a requirement they have not stated: first what the story in the request points to
(what they will do with the item, what will be seen, what went wrong before), then what most separates these goods from each
other. Skip what is already required, and labels that say nothing about the item itself. Each question names one property, or
two closely related ones, in plain words.
Return JSON: {{"ask": [["property"], ["property", "property"]]}}"""


_READING = """Read each page as follows.
- A property's value is what the page states under that property's own label. Several labels may name one property; when they
  disagree, the plainest label, the one named just as the property is, states it.
- A fact giving a list of values lists what the product comes in or combines; "options" names what this listing sells. Where
  options state a value for the property, they decide for this listing.
- Blended or combined values are not the plain value unless the plain value is plainly the main one. A value counts in another
  spelling, another unit or a nearby wording of the same thing, but not as a merely similar thing.
- A property stated in the facts is settled by the facts: words in the title or description do not override them. A property the
  facts do not state may be read from the title or description.
- A value only implied by other wording (a count of units for a pack type, say) is "unclear", not "yes": the property is stated when a
  label names it.
- A need with relation "at most" or "at least" counts numerically in the unit of the need.
- A page may say a property must be taken from some other listing rather than from this one: then this page does not state it, and
  a value it says it is not is not its value."""


def prompt_judge(needs: dict, cards: list[str]) -> str:
    pages = "\n\n".join(cards)
    return f"""What the shopper needs: {json.dumps(needs, ensure_ascii=False)}

Product pages:
{pages}

{_READING}

Go through exactly the needs listed under "needs", not the price or anything else. For each page and each need: copy the words on the page that state it ("words", empty when the page does not state it) and
say "match": "yes" (they meet the need), "no" (they state something that fails it), or "unclear" (the page does not settle it).
A need with relation "not" is met when the page states something else, and also when it does not state that property.
"kind_ok": whether the listing really is the kind of goods wanted, not an accessory, part or something else, and made for the
person the request is about (a child's item is not what a grown-up asks for when wanting it for themselves).
"verdict": "yes" when kind_ok and every need matches, "no" when any need fails or kind_ok is false, otherwise "unclear".
"see_other": when the page sends you to another listing for a property that is a need, {{"property": "the need", "listing": "its
description of that listing"}}, otherwise null.
"measure": when "choose_by" names a measure, its value on this page as one number, otherwise null.
Return JSON: {{"items": [{{"tag": "P1", "kind_ok": true, "needs": {{"property": {{"words": "", "match": "yes"}}}},
"verdict": "yes", "see_other": null, "measure": null}}]}}"""


def prompt_more(needs: dict, tried: list[str], seen_values: dict[str, list[str]]) -> str:
    return f"""What the shopper needs: {json.dumps(needs, ensure_ascii=False)}
Keyword searches already tried: {json.dumps(tried, ensure_ascii=False)}
Values product pages use for related properties: {json.dumps(seen_values, ensure_ascii=False)[:1500]}

The catalog search matches the words of titles, descriptions and property values and returns little when a query is long.
Give three new searches of one to three words likely to reach listings that meet every need, using other wordings of the
wanted values as listings write them.
Return JSON: {{"queries": ["", "", ""]}}"""


def prompt_step(step: str, wants: Any, cart: list[str], tools: list[str]) -> str:
    return f"""The shopper's request requires this step: {step}
What the shopper wants: {json.dumps(wants, ensure_ascii=False)}
What is in the cart now, with prices: {json.dumps(cart, ensure_ascii=False)}
Tools of the shop: {json.dumps(tools)}

Carry out the step. When it takes a message to the shopper, write one short sentence in plain words about that step alone, naming
what the step needs (for an approval, what is in the cart and its total). When it takes a tool, name it and its arguments.
Return JSON: {{"message": "text or empty", "tool": "tool name or empty", "args": {{}}}}"""


def prompt_lookup(description: str, seen: list[str]) -> str:
    listed = "\n".join(seen) or "(nothing yet)"
    return f"""A value has to be read from this listing: {description}
Results so far (tag | price | department | seller | title):
{listed}

If one of the results is that listing, give its tag. Otherwise give a keyword search of two or three words likely to reach it.
Return JSON: {{"tag": "tag or empty", "query": "words or empty"}}"""


def prompt_read_value(need: dict, card: str) -> str:
    return f"""A shopper's requirement takes its value from another listing: {json.dumps(need, ensure_ascii=False)}
That listing's page:
{card}

{_READING}

Give the property's value as this page states it, and the requirement it makes: "relation" is "is", "at least", "at most" or "not",
and "value" is what the item must have, working out any stated difference. If this page sends you on to a further listing for it, give
that listing's description in "next" and leave the value empty.
Return JSON: {{"value": "", "relation": "is", "next": ""}}"""


# --------------------------------------------------------------------------------------
# the world
# --------------------------------------------------------------------------------------

class Finished(Exception):
    """The world reported the episode over."""


class ToolsUnread(Exception):
    """No reading of the world's tools produced a search result."""


class World:
    """One episode's tool calls: one action per step, in order, up to the step limit."""

    def __init__(self, problem_data: dict):
        env = problem_data["environment"]
        self.binding, self.policy = env["binding"], env["policy_view"]
        self.tools = self.policy.get("tools") or []
        self.limit = int(self.policy.get("max_steps") or 0)
        self.step = 0
        self.finished = bool(self.policy.get("finished"))
        self.client = ProxyClient(timeout=60, max_retries=3)
        self.trace: list[dict] = []

    def left(self) -> int:
        return self.limit - self.step

    def call(self, name: str, args: dict) -> tuple[Any, list[str]]:
        if self.finished or self.left() <= 0:
            raise Finished()
        self.step += 1
        reply = self.client.post("/environment/call", json_data={**self.binding, "step": self.step,
                                                                 "action": {"name": name, "args": args}})
        if not isinstance(reply, dict):
            self.trace.append({"step": self.step, "name": name, "args": args, "error": "no reply"})
            return None, []
        self.finished = bool(reply.get("finished"))
        result, heard = _split_reply(reply.get("observation"))
        self.trace.append({"step": self.step, "name": name, "args": args, "finished": self.finished,
                           "result": json.dumps(result, ensure_ascii=False)[:600], "heard": heard})
        return result, heard

    def arguments(self, tool: Any) -> dict:
        for entry in self.tools:
            fn = entry.get("function", entry)
            if fn.get("name") == tool:
                return (fn.get("parameters") or {}).get("properties") or {}
        return {}

    def schema(self, tool: str, arg: str) -> dict:
        return self.arguments(tool).get(arg) or {}

    def has(self, tool: Any) -> bool:
        return any(entry.get("function", entry).get("name") == tool for entry in self.tools)


class Item:
    __slots__ = ("key", "row", "price", "stock", "title", "category", "brand", "text", "facts", "verdict", "gone", "opened", "passed", "extra")

    def __init__(self, key: str):
        self.key = key
        self.row: dict = {}
        self.price: float | None = None
        self.stock: bool | None = None
        self.title = self.category = self.brand = self.text = ""
        self.facts: dict = {}
        self.verdict: dict | None = None
        self.gone = self.opened = self.passed = False
        self.extra: dict = {}

    def card(self, tag: str) -> str:
        facts = "; ".join(f"{k}: {v}" for k, v in self.facts.items() if str(v).strip())
        extra = "".join(f"\n{k}: {str(v)[:500]}" for k, v in self.extra.items() if str(v).strip())
        return (f"[{tag}] {self.title}\nbrand: {self.brand or '(none)'}\ndepartment: {self.category}\nprice: {self.price}\n"
                f"description: {self.text[:400]}\nfacts: {facts or '(none)'}{extra}")

    def words(self) -> str:
        return _squash(" ".join([self.title, self.text, " ".join(f"{k} {v}" for k, v in self.facts.items()),
                                 " ".join(str(v) for v in self.extra.values())]))


class Hunt:
    """One product the shopper wants, and all that is known in looking for it."""

    def __init__(self, want: dict):
        self.name = str(want.get("kind") or "")
        amount = _number(want.get("quantity"))
        self.quantity = int(amount) if amount and amount >= 1 else 1
        self.needs = []
        for need in want.get("needs") or []:  # a value said again in looser words is not a second requirement on the property
            if isinstance(need, dict) and need.get("property") and _squash(need["property"]) not in {_squash(n["property"]) for n in self.needs}:
                self.needs.append(need)
        self.queries = [str(q) for q in want.get("queries") or []]
        self.items: dict[str, Item] = {}
        self.shelf: str | None = None
        self.tried: list[str] = []
        self.browsed: set[str] = set()
        self.asked: set[str] = set()
        self.messages = 0
        self.followed = 0
        self.looked_again = False
        self.shelf_pages = 0
        self.cheaper_tried = False
        self.shelf_done = False
        self.result_pages = 0
        self.pages_read = 0
        self.further: dict[str, Any] = {}  # for each search, what asks for its next page
        self.line: Item | None = None  # what of this product is in the cart


class Meridian:
    def __init__(self, problem_data: dict):
        self.began = time.monotonic()
        self.world = World(problem_data)
        self.brain = Brain(self.began)
        self.query = str(self.world.policy.get("query") or "")
        self.hunts: list[Hunt] = [Hunt({})]
        self.cur = self.hunts[0]
        self.map: dict = {}
        self.keys: dict = {}
        self.ask: dict = {}
        self.cap: float | None = None
        self.messages_left = 0
        self.steps_done: set[tuple] = set()
        self.stages: list[dict] = []
        self.stop_buying = False
        self.process: list[dict] = []
        self.heard_before: set[str] = set()
        self.reserve = _COMMIT_STEPS
        self.page_size = 0
        self.paging: dict = {}
        self.many_most = 0

    # what is being looked for right now; the stages below work on one product at a time
    items = property(lambda self: self.cur.items)
    tried = property(lambda self: self.cur.tried)
    browsed = property(lambda self: self.cur.browsed)
    asked_names = property(lambda self: self.cur.asked)
    needs = property(lambda self: self.cur.needs, lambda self, value: setattr(self.cur, "needs", value))
    shelf = property(lambda self: self.cur.shelf, lambda self, value: setattr(self.cur, "shelf", value))

    def mark(self, stage: str, **info: Any) -> None:
        self.stages.append({"t": round(time.monotonic() - self.began, 1), "step": self.world.step, "stage": stage, **info})

    def elapsed(self) -> float:
        return time.monotonic() - self.began

    # ---- reading the world's tools ----------------------------------------------------
    def read_tools(self) -> bool:
        tag = json.dumps(self.world.tools, sort_keys=True)
        with _lock:
            known = _shared.get(tag)
        for _ in range(3):  # a call lost to load must not cost the episode its reading of the tools
            data = known or self.brain.ask(prompt_tools(self.world.tools), "tools", 900)
            usable = (self.world.has((data.get("order") or {}).get("tool")) and self.world.has((data.get("add") or {}).get("tool"))
                      and self.world.has((data.get("keyword") or {}).get("tool") or (data.get("browse") or {}).get("tool")))
            if usable:
                self.map = data
                return True
            if self.elapsed() > _OPEN_BY / 2:
                break
        return False

    def tools_work(self, hunt: Hunt) -> bool:
        """A reading of the tools counts only once a search has really returned listings; only then do other tasks share it."""
        tag = json.dumps(self.world.tools, sort_keys=True)
        if hunt.items:
            with _lock:
                _shared[tag] = self.map
            return True
        with _lock:
            _shared.pop(tag, None)
        data = self.brain.ask(prompt_tools(self.world.tools) + "\nSearches made by an earlier reading returned nothing: check which "
                              "arguments a search really needs, including any mode argument and its value.", "tools", 900)
        if self.world.has((data.get("keyword") or {}).get("tool") or (data.get("browse") or {}).get("tool")):
            self.map = data
            self.discover([hunt.name])
        return bool(hunt.items)

    def role(self, name: str) -> dict:
        entry = self.map.get(name)
        return entry if isinstance(entry, dict) and self.world.has(entry.get("tool")) else {}

    # ---- rows and pages ---------------------------------------------------------------
    @staticmethod
    def _rows(result: Any) -> list[dict]:
        if isinstance(result, list):
            return [x for x in result if isinstance(x, dict)]
        if isinstance(result, dict):
            for value in result.values():
                if isinstance(value, list) and value and all(isinstance(x, dict) for x in value):
                    return value
        return []

    def read_row(self, row: dict) -> Item | None:
        if not self.keys:
            tag = "row:" + ",".join(sorted(row))
            with _lock:
                known = _shared.get(tag)
            data = known or self.brain.ask(prompt_row(row), "row", 500)
            if data.get("id") in row:
                with _lock:
                    _shared[tag] = data
                self.keys = data
            else:
                return None
        value = lambda name: row.get(self.keys.get(name)) if self.keys.get(name) else None
        key = str(value("id") or "")
        if not key:
            return None
        it = self.items.setdefault(key, Item(key))
        it.row = row
        price = _number(value("price"))
        it.price = price if price is not None else it.price
        stock = value("stock")
        it.stock = stock if isinstance(stock, bool) else it.stock
        it.title = str(value("title") or it.title)
        it.category = str(value("category") or it.category)
        it.brand = str(value("brand") or it.brand)
        it.text = str(value("description") or it.text)
        return it

    def needs_view(self) -> dict:
        choose = self.ask.get("objective") or None
        if choose and self.ask.get("measure"):
            choose = f"{choose} {self.ask['measure']}"
        return {"category": self.cur.name, "needs": [{k: v for k, v in n.items() if k != "told"} for n in self.needs],
                "price_limit": self.cap, "choose_by": choose}

    def all_wants(self) -> list[dict]:
        return [{"kind": h.name, "quantity": h.quantity, "needs": h.needs} for h in self.hunts]

    def hunt_named(self, name: Any) -> Hunt:
        words = set(_squash(name).split())
        return max(self.hunts, key=lambda h: (len(words & set(_squash(h.name).split())), h is self.cur))

    def listen(self, heard: list[str], situation: str) -> dict:
        heard = [x for x in heard if x not in self.heard_before]
        self.heard_before.update(heard)
        if not heard:
            return {}
        data = self.brain.ask(prompt_heard(self.all_wants(), heard, situation), "heard", 800)
        touched: set[int] = set()
        # a requirement is given up only by a line that names its property; saying one has no preference gives up nothing
        spoken = set(_squash(" ".join(heard)).split())
        dropped = {_squash(x) for x in data.get("drop") or [] if spoken & set(_squash(x).split())}
        for i, hunt in enumerate(self.hunts):
            kept = [n for n in hunt.needs if _squash(n.get("property")) not in dropped]
            if len(kept) != len(hunt.needs):
                hunt.needs = kept
                touched.add(i)
        for entry in data.get("add") or []:
            if isinstance(entry, dict) and entry.get("property"):
                hunt = self.hunt_named(entry.get("item"))
                hunt.needs = ([n for n in hunt.needs if _squash(n.get("property")) != _squash(entry["property"])]
                              + [{**{k: v for k, v in entry.items() if k != "item"}, "told": True}])
                touched.add(self.hunts.index(hunt))
        limit = _number(data.get("price_limit"))
        written = [_number(x) for x in re.findall(r"\d[\d,]*(?:\.\d+)?", " ".join(heard))]
        if limit is not None and limit not in written and written:
            limit = max(written)  # a limit counts as the shopper wrote it, never converted
        if limit is not None and limit != self.cap:
            self.cap = limit
        if data.get("stop"):
            self.stop_buying = True
        for i in touched:
            for it in self.hunts[i].items.values():
                it.verdict = None
                it.passed = False
        self.mark("heard", lines=heard, changed=sorted(touched), needs=[len(h.needs) for h in self.hunts], cap=self.cap)
        return data

    def tool_args(self, role: dict, **values: Any) -> dict:
        """Only arguments the tool's own schema lists go out, whatever the reading of the tools said."""
        known = self.world.arguments(role.get("tool"))
        args = dict(role.get("fixed") or {}) if isinstance(role.get("fixed"), dict) else {}
        args = {k: v for k, v in args.items() if k in known}
        for name, value in values.items():
            arg = re.sub(r"\s+or null$", "", str(role.get(name) or ""))
            if arg in known and value is not None:
                args[arg] = value
        return args

    def search(self, query: str, page: int | None = None) -> list[Item]:
        role = self.role("keyword")
        if not role:
            return []
        limit = self.world.schema(role["tool"], role.get("text") or "").get("maxLength")
        words = query[:int(limit)] if limit else query
        result, heard = self.world.call(role["tool"], self.tool_args(role, text=words, page=page, department=self.shelf))
        self.listen(heard, "during a catalog search")
        rows = self._rows(result)
        self.page_size = max(self.page_size, len(rows))
        got = [self.read_row(r) for r in rows]
        self.cur.result_pages += 1
        if page is None:
            self.tried.append(query)
        self.note_further(query, result, page, len(rows))
        return [x for x in got if x]

    def note_further(self, query: str, result: Any, page: Any, count: int) -> None:
        """What asks for this search's next page: the reply says it, or else the page number or the offset goes on."""
        self.cur.further.pop(query, None)
        if not count or not self.role("keyword").get("page"):
            return
        kind = _squash(self.role("keyword").get("page_kind"))
        if "token" in kind and isinstance(result, dict):
            rest = {k: v for k, v in result.items() if not isinstance(v, list)}
            tag = "paging:" + ",".join(sorted(rest))
            with _lock:
                known = _shared.get(tag)
            if not self.paging:
                self.paging = known or self.brain.ask(prompt_paging(rest), "paging", 200)
                with _lock:
                    _shared[tag] = self.paging
            value = rest.get(self.paging.get("next"))
            if value not in (None, ""):
                self.cur.further[query] = value
        elif "offset" in kind:
            self.cur.further[query] = (page or 0) + count
        elif "number" in kind:
            self.cur.further[query] = (page or 1) + 1

    def result_limit(self) -> int:
        """The lowest price can sit on any page, so it is worth more of the steps than a search for any listing that qualifies."""
        return _RESULT_PAGES + 6 if self.by_price_first() else _RESULT_PAGES

    def read_further(self, steps: int) -> bool:
        """The next result pages, taking the searches in turn. True when they showed listings not seen before. When the shopper
        wants the lowest price, a search whose page is all dearer than what is already good is left there."""
        known, hunt = set(self.items), self.cur
        for _ in range(steps):
            if not hunt.further or hunt.result_pages >= self.result_limit() or self.world.left() <= self.reserve + 2 or self.world.finished:
                break
            query = next(iter(hunt.further))
            before = len(self.items)
            got = self.search(query, hunt.further[query])
            good = self.good()
            if self.by_price_first() and good and got and all(it.price is not None and good[0].price is not None
                                                              and it.price >= good[0].price for it in got):
                hunt.further.pop(query, None)
            elif len(self.items) == before:
                hunt.further.pop(query, None)
        return any(k not in known for k in self.items)

    def browse(self, department: str, ceiling: float | None, page: int | None = None) -> list[Item]:
        role = self.role("browse")
        if not role:
            return []
        result, heard = self.world.call(role["tool"], self.tool_args(role, department=department, ceiling=ceiling, page=page))
        self.listen(heard, "during a catalog listing")
        rows = self._rows(result)
        self.page_size = max(self.page_size, len(rows))
        got = [self.read_row(r) for r in rows]
        return [x for x in got if x]

    def open(self, it: Item) -> None:
        role = self.role("facts") or self.role("page")
        self.cur.pages_read += 1
        if not role:
            it.opened = True
            return
        result, heard = self.world.call(role["tool"], self.tool_args(role, id=it.key))
        self.listen(heard, "while reading a product page")
        it.opened = True
        self.absorb(it, result)

    def absorb(self, it: Item, result: Any) -> None:
        if isinstance(result, dict):
            inner = [v for v in result.values() if isinstance(v, dict)]
            it.facts = inner[0] if len(inner) == 1 else {k: v for k, v in result.items() if not isinstance(v, (list, dict))}
            it.extra.update({k: v for k, v in result.items() if isinstance(v, str) and inner and v not in (it.title, it.text, it.key)
                             and k not in (self.keys.get("category"), self.keys.get("brand"))})
            if self.keys.get("id") in result:
                self.read_row(result)

    def open_all(self, batch: list[Item]) -> None:
        """Read these pages, several in one call where the shop has a tool for it. A call turned back says how many it takes
        only by failing: it is tried with fewer, then the pages are read one at a time."""
        role = self.role("many")
        arg = str(role.get("ids") or "")
        if not self.many_most:
            self.many_most = int(_number(role.get("most")) or self.world.schema(role.get("tool"), arg).get("maxItems") or 4) if role else 1
        batch = list(batch)
        while batch and self.world.left() > self.reserve and not self.world.finished:
            part, batch = batch[:self.many_most], batch[self.many_most:]
            if len(part) == 1 or self.many_most < 2 or arg not in self.world.arguments(role.get("tool")):
                self.open(part[0])
                batch = part[1:] + batch
                continue
            result, heard = self.world.call(role["tool"], {arg: [it.key for it in part]})
            self.listen(heard, "while reading product pages")
            rows = {str(r.get(self.keys.get("id"))): r for r in self._rows(result)}
            if not rows:
                self.many_most = self.many_most // 2
                batch = part + batch
                continue
            for it in part:
                if it.key in rows:
                    it.opened = True
                    self.cur.pages_read += 1
                    self.absorb(it, rows[it.key])
                elif self.world.left() > self.reserve:
                    self.open(it)

    def refresh(self, it: Item) -> None:
        """What the listing says now about its price and stock."""
        role = self.role("page")
        if not role or self.world.left() <= 1:
            it.gone = True
            return
        result, heard = self.world.call(role["tool"], self.tool_args(role, id=it.key))
        self.listen(heard, "while reading a listing")
        if isinstance(result, dict) and it.key in json.dumps(result):
            self.read_row(self._rows(result)[0] if self._rows(result) else result)
        else:
            it.gone = True

    def buyable(self, it: Item) -> bool:
        if it.gone or it.stock is False:
            return False
        if self.cap is not None and it.price is not None and it.price * self.cur.quantity > self.cap:
            return False
        return not self.shelf or not it.category or _squash(it.category) == _squash(self.shelf)

    # ---- the stages of a task ---------------------------------------------------------
    def read_request(self) -> None:
        for _ in range(3):
            self.ask = self.brain.ask(prompt_request(self.query, self.world.policy.get("opening")), "request", 2500)
            wants = [w for w in self.ask.get("wants") or [] if isinstance(w, dict) and w.get("kind")]
            if wants or self.elapsed() > _OPEN_BY / 2:
                break
        if wants:
            self.hunts = [Hunt(w) for w in wants]
            self.cur = self.hunts[0]
        self.cap = _number(self.ask.get("price_limit"))
        self.process = [p for p in self.ask.get("procedures") or [] if isinstance(p, dict) and p.get("step")]
        budget = _number(self.ask.get("message_budget"))
        self.messages_left = int(budget) if budget is not None else 1
        kept = sum(int(_number(p.get("messages")) or 0) for p in self.process)
        free = max(0, self.messages_left - kept)
        for i, hunt in enumerate(self.hunts):
            hunt.messages = free // len(self.hunts) + (1 if i < free % len(self.hunts) else 0)
        self.mark("request", wants=[(h.name, h.quantity, len(h.needs)) for h in self.hunts], cap=self.cap,
                  messages=self.messages_left, objective=self.ask.get("objective"), process=self.process)

    def find_shelf(self) -> None:
        counts: dict[str, int] = {}
        for it in self.items.values():
            if it.category:
                counts[it.category] = counts.get(it.category, 0) + 1
        if not counts:
            return
        answer = self.brain.ask(prompt_shelf(self.cur.name, counts), "shelf", 400).get("department")
        self.shelf = answer if answer in counts else None
        self.mark("shelf", shelf=self.shelf, seen=len(counts))

    def discover(self, queries: list[str]) -> None:
        for query in queries:
            if self.world.left() <= self.reserve + 2:
                return
            q = str(query).strip()
            if q and q not in self.tried:
                self.search(q)

    def list_shelf(self, ceiling: float | None) -> list[Item]:
        tag = f"{self.shelf}|{ceiling}"
        role = self.role("browse")
        if not self.shelf or tag in self.browsed or self.world.left() <= self.reserve + 2 or role.get("department") not in self.world.arguments(role.get("tool")):
            return []
        self.browsed.add(tag)
        return self.browse(self.shelf, ceiling)

    def next_shelf_page(self) -> bool:
        """The shelf's next page, while it has one and steps allow. True when it showed listings not seen before."""
        role = self.role("browse")
        hunt = self.cur
        if not role or not self.shelf or not role.get("page") or hunt.shelf_done or self.world.left() <= self.reserve + 2:
            return False
        hunt.shelf_pages += 1
        offset = "offset" in _squash(role.get("page_kind"))
        value = hunt.shelf_pages * self.page_size if offset else hunt.shelf_pages + 1
        highest = self.world.schema(role["tool"], str(role.get("page"))).get("maximum")
        if not self.page_size or (highest is not None and value > highest):
            hunt.shelf_done = True
            return False
        known = set(self.items)
        got = self.browse(self.shelf, self.cap, value)
        hunt.shelf_done = not got
        return any(it.key not in known for it in got)

    def by_price_first(self) -> bool:
        objective = _squash(self.ask.get("objective"))
        return bool(objective) and not self.ask.get("measure") and any(w in objective for w in ("low", "cheap", "least", "price"))

    def unopened(self) -> list[Item]:
        return [it for it in self.items.values() if not it.opened and not it.passed and self.buyable(it)]

    def shortlist(self, limit: int) -> list[Item]:
        rows = self.unopened()
        if not rows or limit <= 0:
            return []
        rows = sorted(rows, key=lambda it: it.price if it.price is not None else 1e12)[:80]
        tagged = {f"R{i + 1}": it for i, it in enumerate(rows)}
        lines = [f"{tag} | {it.price} | {it.category[:30]} | {it.title[:110]} | {it.text[:110]}".replace("\n", " ") for tag, it in tagged.items()]
        wanted = self.brain.ask(prompt_shortlist(self.needs_view(), lines, limit), "shortlist", 700).get("open") or []
        chosen = list(dict.fromkeys(tagged[t] for t in wanted if t in tagged))
        if chosen:
            for it in rows:
                it.passed = it not in chosen
        else:
            chosen = rows
        if self.by_price_first():
            chosen.sort(key=lambda it: it.price if it.price is not None else 1e12)
        return chosen[:limit]

    def judge(self) -> None:
        pending = [it for it in self.items.values() if it.opened and it.verdict is None and not it.gone]
        if not pending or self.elapsed() > _STOP_BY:
            return
        batches = [pending[i:i + _PER_JUDGE] for i in range(0, len(pending), _PER_JUDGE)]
        tagging = [{f"P{i + 1}": it for i, it in enumerate(batch)} for batch in batches]
        prompts = [prompt_judge(self.needs_view(), [it.card(tag) for tag, it in tagged.items()]) for tagged in tagging]
        for tagged, data in zip(tagging, self.brain.ask_many(prompts, "judge")):
            for row in data.get("items") or []:
                if isinstance(row, dict) and row.get("tag") in tagged:
                    tagged[row["tag"]].verdict = self.checked(tagged[row["tag"]], row)
        self.mark("judge", pages=len(pending), good=len(self.good()))
        before = len(self.needs)
        self.set_aside_unsettled()
        if len(self.needs) != before:
            self.judge()

    def checked(self, it: Item, row: dict) -> dict:
        """A yes stands only when every need the model read as met is quoted from the page itself."""
        table = row.get("needs") if isinstance(row.get("needs"), dict) else {}
        page = it.words() + " " + _squash(it.category)
        if str(row.get("verdict")).casefold() == "yes":
            for need in self.needs:
                entry = next((x for k, x in table.items() if _squash(k) == _squash(need.get("property"))), None)
                relation = _squash(need.get("relation"))
                if not isinstance(entry, dict) or str(entry.get("match")).casefold() != "yes":
                    if relation != "not":
                        row["verdict"] = "unclear"
                elif relation != "not" and _squash(entry.get("words")) not in page:
                    row["verdict"] = "unclear"
        return row

    def set_aside_unsettled(self) -> None:
        """A need read from the request that no page speaks to at all, after many pages, is not a property these listings
        have: it was a reading of the story. It stops holding every page back. What the shopper said when asked stays."""
        read = [it for it in self.items.values() if it.verdict]
        if len(read) < _UNSETTLED_AFTER:
            return

        def spoken(need: dict) -> bool:
            for it in read:
                table = it.verdict.get("needs") if isinstance(it.verdict.get("needs"), dict) else {}
                entry = next((x for k, x in table.items() if _squash(k) == _squash(need.get("property"))), None)
                if not isinstance(entry, dict) or str(entry.get("match")).casefold() != "unclear" or _squash(entry.get("words")):
                    return True
            return False
        silent = [n for n in self.needs if not n.get("told") and _squash(n.get("relation")) != "not" and not spoken(n)]
        if silent:
            self.needs = [n for n in self.needs if n not in silent]
            for it in read:
                if self.verdict_of(it) == "unclear":
                    it.verdict = None
            self.mark("set aside", needs=[n.get("property") for n in silent])

    def verdict_of(self, it: Item) -> str:
        return str((it.verdict or {}).get("verdict")).casefold()

    def good(self) -> list[Item]:
        order = [it for it in self.items.values() if self.verdict_of(it) == "yes" and self.buyable(it)]
        price = lambda it: it.price if it.price is not None else 1e12
        if self.by_price_first():
            return sorted(order, key=price)
        if self.ask.get("measure"):
            sign = -1 if "most" in _squash(self.ask.get("objective")) else 1

            def measured(it: Item) -> float:
                value = _number((it.verdict or {}).get("measure"))
                return sign * value if value is not None else 1e12
            return sorted(order, key=lambda it: (measured(it), price(it)))

        # with no price wish, first the listings that also say the required values in their own title
        def said(it: Item) -> int:
            title = _squash(it.title).replace(" ", "")
            return sum(1 for n in self.needs if _squash(n.get("relation")) != "not" and _squash(n.get("value")).replace(" ", "") in title)
        return sorted(order, key=lambda it: (-said(it), price(it)))

    def possible(self) -> list[Item]:
        order = [it for it in self.items.values() if self.verdict_of(it) == "unclear" and self.buyable(it)
                 and (it.verdict or {}).get("kind_ok") is not False]

        def meets(it: Item) -> int:
            table = (it.verdict or {}).get("needs")
            names = {_squash(n.get("property")) for n in self.needs}
            return sum(1 for k, x in (table or {}).items() if _squash(k) in names and isinstance(x, dict)
                       and str(x.get("match")).casefold() == "yes")
        order = [it for it in order if meets(it) or not self.needs]
        return sorted(order, key=lambda it: (-meets(it), it.price if it.price is not None else 1e12))

    def question_names(self) -> list[str]:
        counts: dict[str, int] = {}
        for it in self.items.values():
            for label in it.facts:
                name = re.sub(r"[._]+", " ", str(label)).strip().casefold()
                if name and name not in self.asked_names:
                    counts[name] = counts.get(name, 0) + 1
        return sorted(counts, key=lambda n: -counts[n])

    def question_shopper(self) -> None:
        """Ask by name about what matters most on this shelf's pages: a shopper's unstated requirement surfaces that way."""
        role = self.role("message")
        count = min(self.cur.messages, self.messages_left)
        names = self.question_names()
        if not role or count <= 0 or not names:
            return
        self.asked_names.update(names)  # asked once for each product, whatever comes of it
        picked = self.brain.ask(prompt_questions(self.query, self.needs_view(), names[:40], count), "questions", 800).get("ask") or []
        picked = (list(picked) + [n for n in names if n not in str(picked)])[:count]  # the most common properties fill what is left
        order = [[str(x) for x in (entry if isinstance(entry, list) else [entry])][:2] for entry in picked][:count]
        before = json.dumps(self.needs)
        about = f" for the {self.cur.name}" if len(self.hunts) > 1 else ""
        for part in order:
            if not any(part) or self.world.left() <= self.reserve or self.world.finished:
                break
            text = f"Do you have a requirement on {' or '.join(part)}{about}?"
            result, heard = self.world.call(role["tool"], self.tool_args(role, text=text))
            self.cur.messages -= 1
            self.messages_left -= 1
            self.listen(_spoken(result) + heard, "the shopper answers the question: " + text)
        self.mark("asked", left=self.messages_left, questions=order)
        if json.dumps(self.needs) != before and self.world.left() > self.reserve + 3:
            self.search_more()

    def settled(self) -> bool:
        """Enough good listings are known, two because a pick can sell out; when the shopper wants the lowest price, none
        of the listings still unread is cheaper than those two."""
        good = self.good()
        if len(good) < 2:
            return False
        if not self.by_price_first():
            return True
        if self.cur.further and self.cur.result_pages < self.result_limit() and self.world.left() > self.reserve + 2:
            return False
        return not any(it.price is not None and good[1].price is not None and it.price < good[1].price for it in self.unopened())

    def scan_low_prices(self) -> None:
        """The shelf listed under rising price ceilings reaches its cheapest listings, which keyword searches miss."""
        if not self.cap or not self.role("browse"):
            return
        spent = 0
        for share in (0.0625, 0.125, 0.25, 0.5):
            if spent >= self.world.limit // 4 or sum(1 for it in self.unopened() if it.price is not None
                                                     and it.price <= self.cap * share) >= 24:
                return
            ceiling = round(self.cap * share, 2)
            got = self.list_shelf(ceiling)
            spent += 1
            if self.page_size and len(got) >= self.page_size and self.role("browse").get("page") and self.world.left() > self.reserve + 2:
                kind = _squash(self.role("browse").get("page_kind"))
                self.browse(self.shelf or "", ceiling, len(got) if "offset" in kind else 2)
                spent += 1

    def explore(self) -> None:
        """Spend the steps left on pages until enough good listings are known: a pick can sell out while ordering."""
        rounds = 0
        while (self.elapsed() < (_STOP_BY if self.reserve == _LAST_STEPS else _OPEN_BY) and self.world.left() > self.reserve
               and not self.world.finished and not self.stop_buying):
            self.resolve_references()
            self.judge()
            if self.follow_referrals():
                continue
            if self.settled():
                return
            room = min(_PAGES_READ - self.cur.pages_read, (self.world.left() - self.reserve) * max(self.many_most, 1))
            if self.cur.pages_read >= _PAGES_READ:
                return
            batch = self.shortlist(min(_FIRST_PAGES if not self.asked_names else 8, room))
            if not batch:
                if self.read_further(2):
                    continue
                # before searching further or giving up, read what was passed over on its title alone
                passed = [it for it in self.items.values() if it.passed and not it.opened and self.buyable(it)]
                if passed and not self.good():
                    for it in passed:
                        it.passed = False
                    continue
                if self.next_shelf_page():
                    continue
                rounds += 1
                if rounds > 2 or self.world.left() <= self.reserve + 2:
                    return
                self.search_more()
                continue
            self.open_all(batch)
            if not self.asked_names and self.messages_left > 0:
                self.question_shopper()
        self.judge()

    def search_more(self) -> None:
        values: dict[str, list[str]] = {}
        for it in self.items.values():
            for label, value in it.facts.items():
                bucket = values.setdefault(str(label), [])
                if len(bucket) < 6 and str(value) not in bucket:
                    bucket.append(str(value)[:40])
        fresh = self.brain.ask(prompt_more(self.needs_view(), self.tried[-12:], values), "more", 600).get("queries") or []
        self.discover(fresh[:3])
        self.list_shelf(self.cap)

    def read_elsewhere(self, need: dict, description: str) -> dict:
        """Find the listing a value must be read from, a few steps at most, and read it there."""
        for _ in range(4):
            if self.world.left() <= max(_LAST_STEPS, self.reserve - 2) or self.elapsed() > _STOP_BY:
                return {}
            seen = sorted(self.items.values(), key=lambda it: -len(set(_squash(description).split()) & set(it.words().split())))[:25]
            tagged = {f"R{i + 1}": it for i, it in enumerate(seen)}
            lines = [f"{t} | {it.price} | {it.category} | {it.row.get('merchant', it.brand) if it.row else it.brand} | {it.title[:90]}"
                     for t, it in tagged.items()]
            data = self.brain.ask(prompt_lookup(description, lines), "lookup", 300)
            target = tagged.get(str(data.get("tag") or ""))
            if target:
                if not target.opened:
                    self.open(target)
                read = self.brain.ask(prompt_read_value(need, target.card("L")), "lookup", 300)
                if read.get("next"):
                    description = str(read["next"])
                    continue
                return read
            query = str(data.get("query") or "").strip()
            if not query or query in self.tried:
                return {}
            saved, self.cur.shelf = self.cur.shelf, None  # the other listing may sit on any shelf
            self.search(query)
            self.cur.shelf = saved
        return {}

    def resolve_references(self) -> None:
        for need in self.needs:
            description = str(need.get("refers_to") or "").strip()
            if not description or need.get("resolved"):
                continue
            need["resolved"] = True
            read = self.read_elsewhere(need, description)
            if str(read.get("value") or "").strip():
                need["value"], need["relation"] = read["value"], read.get("relation") or need.get("relation") or "is"
                need["refers_to"] = ""
                for it in self.items.values():
                    it.verdict = None
            self.mark("reference", need=need.get("property"), value=need.get("value"))

    def follow_referrals(self) -> bool:
        """A page that sends you elsewhere for a needed property: read it there, for the listings closest to being chosen."""
        for it in self.possible()[:2]:
            other = (it.verdict or {}).get("see_other")
            if not isinstance(other, dict) or not other.get("listing") or self.cur.followed >= 2:
                continue
            self.cur.followed += 1
            need = next((n for n in self.needs if _squash(n.get("property")) == _squash(other.get("property"))), None)
            read = self.read_elsewhere(need or {"property": other.get("property")}, str(other["listing"]))
            value = str(read.get("value") or "").strip()
            (it.verdict or {})["see_other"] = None
            if value:
                it.facts[str(other.get("property"))] = value
                it.verdict = None
                return True
        return False

    def cart_lines(self) -> list[str]:
        return [f"{h.quantity} x {h.line.title[:80]} at {h.line.price}" for h in self.hunts if h.line]

    def do_steps(self, when: str) -> bool:
        """Carry out what the request requires at this point. True when something changed that calls for another look."""
        changed = False
        state = "|".join(h.line.key if h.line else "" for h in self.hunts) if when == "order" else ""
        for p in self.process:
            if str(p.get("when")) != when or (p["step"], state) in self.steps_done or self.world.finished:
                continue
            self.steps_done.add((p["step"], state))
            names = [e.get("function", e).get("name") for e in self.world.tools]
            data = self.brain.ask(prompt_step(p["step"], self.all_wants(), self.cart_lines(), names), "step", 500)
            tool, text = str(data.get("tool") or ""), str(data.get("message") or "").strip()
            role = self.role("message")
            args = data.get("args") if isinstance(data.get("args"), dict) else {}
            if not text and tool == role.get("tool"):  # the message may come back as the tool's own arguments
                text = str(args.get(role.get("text")) or next((v for v in args.values() if isinstance(v, str)), "")).strip()
            if not text and not self.world.has(tool) and _number(p.get("messages")):
                text = f"{p['step']}. In the cart: " + "; ".join(self.cart_lines()) + "."
            before = json.dumps(self.all_wants())
            cap = self.cap
            for _ in range(2):
                if text and role and self.messages_left > 0 and self.world.left() > 1:
                    self.messages_left -= 1
                    result, heard = self.world.call(role["tool"], self.tool_args(role, text=text[:700]))
                elif self.world.has(tool) and tool != role.get("tool") and self.world.left() > 1:
                    known = self.world.arguments(tool)
                    result, heard = self.world.call(tool, {k: v for k, v in args.items() if k in known})
                else:
                    break
                lines = _spoken(result) + heard
                said = self.listen(lines, f"the reply to this step: {p['step']}")
                self.mark("procedure", action=p["step"], text=text[:160], lines=lines, granted=said.get("granted"))
                if said.get("granted") is False:
                    self.steps_done.discard((p["step"], state))
                    changed = True
                if not said.get("not_understood"):
                    break
            changed = changed or json.dumps(self.all_wants()) != before or self.cap != cap
        return changed

    def remove(self, hunt: Hunt) -> None:
        role = self.role("remove")
        if hunt.line and role and self.world.left() > 1 and not self.world.finished:
            _, heard = self.world.call(role["tool"], self.tool_args(role, id=hunt.line.key))
            self.listen(heard, "while taking an item out of the cart")
        hunt.line = None

    def plan(self, hunt: Hunt) -> Item | None:
        """The listing to buy for this product. With no good listing left it looks further; short of steps, the least doubtful
        one will do, because running out of steps does not show that the shop has nothing."""
        self.cur = hunt
        self.judge()
        order = self.good()
        if not order and self.world.left() > _LAST_STEPS and self.elapsed() < _STOP_BY and not hunt.looked_again:
            hunt.looked_again = True
            self.reserve = _LAST_STEPS + _LINE_STEPS * len(self.hunts)
            self.search_more()
            self.explore()
            order = self.good()
        return (order or self.possible() or [None])[0]

    def within_price(self, picks: dict[Hunt, Item]) -> None:
        """The whole order has to fit the limit: swap in cheaper good listings, starting with the dearest line."""
        if self.cap is None or str(self.ask.get("price_scope")) == "each":
            return
        total = lambda: sum((it.price or 0) * h.quantity for h, it in picks.items())
        for hunt in sorted(picks, key=lambda h: -(picks[h].price or 0)):
            self.cur = hunt
            for it in sorted(self.good(), key=lambda x: x.price if x.price is not None else 1e12):
                if total() <= self.cap:
                    return
                if (it.price or 0) < (picks[hunt].price or 0):
                    picks[hunt] = it

    def buy(self) -> None:
        """Keep the cart at the best listing for each product and order it. A world that does not accept the order has said
        why: what it said is applied and the listings in the cart are read again."""
        tries: dict[str, int] = {}
        while not self.world.finished and self.world.left() > 0 and not self.stop_buying:
            picks = {h: self.plan(h) for h in self.hunts}
            if self.world.finished:
                return
            if not all(picks.values()):
                break
            self.within_price(picks)
            total = sum((it.price or 0) * h.quantity for h, it in picks.items())
            if self.cap is not None and total > self.cap and str(self.ask.get("price_scope")) != "each":
                # still too dear: look on the shelf of the dearest line for what the rest of the order leaves room for
                dear = max(picks, key=lambda h: (picks[h].price or 0) * h.quantity)
                if not dear.cheaper_tried and self.world.left() > _LAST_STEPS:
                    dear.cheaper_tried = True
                    self.cur = dear
                    room = self.cap - (total - (picks[dear].price or 0) * dear.quantity)
                    self.reserve = _LAST_STEPS + len(self.hunts)
                    self.list_shelf(round(room / dear.quantity, 2))
                    saved, self.cap = self.cap, room
                    self.explore()
                    self.cap = saved
                    continue
            changing = [h for h, it in picks.items() if h.line is not it]
            if changing:
                hunt = changing[0]
                self.cur = hunt
                if hunt.line:
                    self.remove(hunt)
                    continue
                role = self.role("add")
                if self.world.left() < 2:
                    break
                _, heard = self.world.call(role["tool"], self.tool_args(role, id=picks[hunt].key, quantity=hunt.quantity))
                hunt.line = picks[hunt]
                self.listen(heard, f"after adding {picks[hunt].title[:80]} to the cart")
                self.mark("added", item=picks[hunt].key, price=picks[hunt].price, want=hunt.name)
                continue
            if self.do_steps("order"):
                continue
            result, heard = self.world.call(self.role("order")["tool"], {})
            if self.world.finished:
                self.mark("ordered", items=[h.line.key for h in self.hunts if h.line])
                return
            self.listen(heard or _spoken(result), "after trying to order the cart")
            self.mark("order refused", lines=heard, result=str(result)[:200])
            # what each listing in the cart says now decides whether it can still be had
            for hunt in self.hunts:
                if hunt.line and self.world.left() > 1:
                    self.cur = hunt
                    tries[hunt.line.key] = tries.get(hunt.line.key, 0) + 1
                    self.refresh(hunt.line)
                    if tries[hunt.line.key] >= 3:
                        hunt.line.gone = True
        self.decline()

    def decline(self) -> None:
        self.do_steps("end")
        while not self.world.finished and self.world.left() > 0:
            role = self.role("decline")
            if not role:
                return
            self.mark("declining")
            _, heard = self.world.call(role["tool"], self.tool_args(role, text="Nothing in the shop meets every requirement."))
            if self.world.finished:
                return
            self.listen(heard, "after declining")
            if not self.stop_buying and self.world.left() > 2:
                for hunt in self.hunts:
                    self.cur = hunt
                    self.judge()
                if all(h.items and (self.pick_ready(h)) for h in self.hunts):
                    self.buy()
                    return

    def pick_ready(self, hunt: Hunt) -> bool:
        self.cur = hunt
        return bool(self.good() or self.possible())

    def free_play(self) -> None:
        """A world whose tools could not be read: the model plays the tools directly."""
        messages: list[dict] = [{"role": "system", "content": "Use the supplied tools to serve the shopper exactly as the request "
                                 "says. Every reply must be exactly one tool call. Check each item's property list before "
                                 "ordering. Continue until a result reports finished=true."},
                                {"role": "user", "content": self.query}]
        client = ProxyClient(timeout=35, max_retries=2, rate_limit_retry_delay=3)
        lost = 0
        while not self.world.finished and self.world.left() > 0 and self.elapsed() < _STOP_BY:
            reply = client.post("/inference/chat/completions", json_data={
                "model": _MODELS[lost % len(_MODELS)], "messages": messages, "tools": self.world.tools, "tool_choice": "auto",
                "temperature": 0})
            note = (reply or {}).get("choices", [{}])[0].get("message") or {}
            calls = (note.get("tool_calls") or [])[:1]
            if not calls:
                lost += 1
                if lost > 6:
                    return
                continue
            messages.append({"role": "assistant", "content": note.get("content") or "", "tool_calls": calls})
            fn = calls[0]["function"]
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except ValueError:
                args = {}
            result, heard = self.world.call(fn["name"], args if isinstance(args, dict) else {})
            messages.append({"role": "tool", "tool_call_id": calls[0]["id"],
                             "content": json.dumps({"result": result, "shopper": heard}, ensure_ascii=False)[:6000]})

    def look_for(self, hunt: Hunt, reserve: int) -> None:
        self.cur = hunt
        self.reserve = reserve
        self.discover(hunt.queries[:3] or [hunt.name])
        if not self.tools_work(hunt):
            raise ToolsUnread()
        self.find_shelf()
        self.list_shelf(self.cap)
        if self.by_price_first():
            self.scan_low_prices()
        self.read_further(3)
        self.explore()
        if self.world.left() > reserve + 2 and len(self.good()) < 2:
            self.discover(hunt.queries[3:])
            self.explore()

    def run(self) -> list[dict]:
        try:
            if not self.read_tools():
                self.mark("free play")
                self.free_play()
            else:
                self.read_request()
                if not self.hunts[0].name:
                    self.free_play()
                    return self.finish_trace()
                self.do_steps("choose")
                commit = _COMMIT_STEPS + _LINE_STEPS * (len(self.hunts) - 1)
                share = max(0, self.world.limit - commit) // len(self.hunts)
                for i, hunt in enumerate(self.hunts):
                    self.look_for(hunt, commit + share * (len(self.hunts) - 1 - i))
                self.reserve = commit
                self.buy()
        except Finished:
            pass
        except ToolsUnread:
            self.mark("free play")
            try:
                self.free_play()
            except Finished:
                pass
        except Exception as exc:
            self.world.trace.append({"error": type(exc).__name__, "detail": str(exc)[:300]})
            try:
                if not self.world.finished and self.map:
                    self.buy()
            except Exception as again:
                self.world.trace.append({"error": type(again).__name__, "detail": str(again)[:300]})
        return self.finish_trace()

    def finish_trace(self) -> list[dict]:
        self.world.trace.append({"model_calls": self.brain.log, "stages": self.stages, "needs": self.needs_view(),
                                 "verdicts": {it.key: {"title": it.title[:80], "price": it.price, **(it.verdict or {})}
                                              for h in self.hunts for it in h.items.values() if it.verdict},
                                 "tried": [q for h in self.hunts for q in h.tried], "wants": self.all_wants()})
        return self.world.trace


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    return Meridian(problem_data).run()


__all__ = ["agent_main"]
