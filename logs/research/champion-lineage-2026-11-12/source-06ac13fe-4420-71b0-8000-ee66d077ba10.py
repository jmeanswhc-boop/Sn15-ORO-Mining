"""Shopping agent for shopper situations: one shopper, one order.

The shopper states some requirements, keeps others until asked, and the market or the shopper may
change while the agent shops. The agent reads the request once, asks at most the one question the
request leaves open, searches the catalogue, reads the listings it might buy, puts its best match in
the cart, checks what the cart did to price and stock, follows any change, and orders.

A language model reads the request, the shopper's lines and the listing pages; the code keeps the
episode in order (turns, searches, the cart, stock checks, the final order).
"""

from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from os import getenv
from time import monotonic
from typing import Any

from src.agent.proxy_client import ProxyClient

# ----------------------------------------------------------------------------- settings
MODELS = {
    "chutes": ["zai-org/GLM-5.2-TEE", "moonshotai/Kimi-K2.6-TEE"],
    "openrouter": ["z-ai/glm-5.2", "moonshotai/kimi-k2.6"],
}
SPEND_CAP_USD = 0.05          # per episode; the order is still placed when the cap is reached
CALLS_PER_TURN = 16           # tool calls sent in one turn (at most what the task allows)
OPEN_BATCH = 5                # listings per compare call
OPEN_LIMIT = 250              # listings read in full over the episode
SEARCH_LIMIT = 80             # searches and shelf filters sent while looking (every one loads the store)
JUDGE_CARDS = 18              # listings shown to the reader in one call
JUDGE_LIMIT = 120             # listings the reader is asked about over the episode
EPISODE_SECONDS = 180.0       # after this the agent stops looking and orders what it has
TURN_RESERVE = 6              # turns kept for the cart, the stock checks and the order
QUERY_LENGTH = 200            # a search is cut to this many characters
THINK_TOKENS = 3000           # room for a thinking pass before the tool call is written
MIN_TOKENS = 900              # every call: a provider may think before it writes the tool call, even when asked not to
NOT_A_PROPERTY = re.compile(r"categor|budget|price|\bstock|availab")     # terms of the purchase, not properties of the item
PLAIN_QUESTION = "What will you use it for?"

_proxy = ProxyClient(timeout=90, max_retries=2)


# ----------------------------------------------------------------------------- small helpers
def lowered(text: Any) -> str:
    return " ".join(str(text or "").lower().split())


def word_list(text: Any) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", lowered(text))


def singular(word: str) -> str:
    """One form for a word and its plural (also for words written with -ie or -y)."""
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith(("ches", "shes", "sses", "xes")):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        word = word[:-1]
    if len(word) > 4 and word.endswith("ie"):
        return word[:-2] + "y"
    return word


def closed_up(text: Any) -> str:
    """Amounts written as listings write them: the number and its unit as one word (3 cm -> 3cm)."""
    return re.sub(r"(\d)\s+(?=[a-z]{1,4}\b)", r"\1", lowered(text))


def same_word(one: str, other: str) -> bool:
    """Two words are the same word when they differ only by an ending (dot / dotted)."""
    if one == other:
        return True
    short, long = sorted((one, other), key=len)
    return len(short) >= 4 and not short[0].isdigit() and long.startswith(short) and len(long) - len(short) <= 2


def stems(text: Any) -> set[str]:
    return {singular(w) for w in word_list(closed_up(text))}


def label_key(text: Any) -> str:
    """A category label reduced to its words in singular form, for comparing two spellings of it."""
    return " ".join(singular(w) for w in word_list(str(text).replace("&", " and ")) if w != "and")


def money(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def tool_spec(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {"type": "function", "function": {"name": name, "description": description,
                                             "parameters": {"type": "object", "properties": properties, "required": required}}}


def strings(description: str) -> dict:
    return {"type": "array", "items": {"type": "string"}, "description": description}


def act(name: str, **args: Any) -> dict:
    return {"name": name, "args": {k: v for k, v in args.items() if v is not None}}


# ----------------------------------------------------------------------------- what the reader fills in
PLAN_TOOL = tool_spec("plan", "The shopper's request, written down for shopping.", {
    "item": {"type": "string", "description": "The shopper's own words for the kind of thing, copied from the request exactly as written, without the article and without the requirements."},
    "category": {"type": "string", "description": "A store category, only when the shopper names one as a category; otherwise empty."},
    "requirements": {"type": "array", "description": "Each property the item itself must have, stated in the request.", "items": {
        "type": "object", "properties": {
            "property": {"type": "string", "description": "The name of the property in the shopper's words (colour, material, pattern, style, capacity, ...)."},
            "value": {"type": "string", "description": "The required value, as the shopper wrote it, with its unit."},
            "written": {"type": "array", "items": {"type": "string"}, "maxItems": 6, "description": "The same value written another way: another spelling of the word, or the same amount in another unit. Empty when there is none."},
            "alike": {"type": "array", "items": {"type": "string"}, "maxItems": 6, "description": "Other words sellers use for the same value (a synonym; not a part or a wider kind of it). Empty when there is none."}},
        "required": ["property", "value"]}},
    "budget": {"type": "number", "description": "The most the shopper will pay; 0 when none is named."},
    "brand": {"type": "string", "description": "A brand the shopper requires; empty when none."},
    "doubted_pair": {"type": "array", "items": {"type": "string"}, "maxItems": 2,
                     "description": "When the shopper wishes for two things and doubts one item can have both: the two wishes, each as its value with what it describes (for example 'stainless steel material', 'the PROBEROS brand'). Empty otherwise."},
    "purposes": {"type": "array", "items": {"type": "string"}, "maxItems": 3,
                 "description": "Each use or recipient the request gives for the item, one entry each, in the shopper's words. Empty when it gives none."},
    "priority": {"type": "string", "description": "What the shopper says makes one fitting item better than another (for example the lowest price); empty when they say nothing."},
    "queries": strings("Six short catalogue searches (two to five words) that could return the item: the item words with the required values, with a synonym or another spelling of a value where one exists."),
}, ["item", "requirements", "budget", "doubted_pair", "purposes", "queries"])

HEAR_TOOL = tool_spec("hear", "What the shopper's new line changes.", {
    "adds": {"type": "array", "description": "Each requirement of the item the line states that was not known before.", "items": {
        "type": "object", "properties": {
            "property": {"type": "string"}, "value": {"type": "string"},
            "written": {"type": "array", "items": {"type": "string"}, "maxItems": 6, "description": "The same value written another way: another spelling of the word, or the same amount in another unit. Empty when there is none."},
            "alike": {"type": "array", "items": {"type": "string"}, "maxItems": 6, "description": "Other words sellers use for the same value (a synonym; not a part or a wider kind of it). Empty when there is none."}},
        "required": ["property", "value"]}},
    "drops": strings("Properties whose earlier requirement the line withdraws or replaces."),
    "budget": {"type": "number", "description": "A new spending limit the line states; 0 when it states none."},
    "priority": {"type": "string", "description": "What the line says matters most between fitting items; empty when nothing."},
    "asks": {"type": "boolean", "description": "True when the line asks the assistant something or expects an answer."},
    "about_change": {"type": "boolean", "description": "True when the line only reacts to the stock or the price of the chosen item having changed."},
    "queries": strings("Up to six short catalogue searches that use what the line added (try a synonym or another spelling of the value too); empty when it added nothing."),
}, ["adds", "drops", "budget", "asks", "queries"])

JUDGE_TOOL = tool_spec("judge", "Which listings state every requirement.", {
    "marks": strings("One short entry for every listing shown, in the order shown: the listing's number, then K when it is the kind of thing asked for or X when it is not, "
                     "then one letter per requirement in order - Y when the listing's own words state that value, N when they do not. Example for two requirements: '3 K YN'."),
    "proof": {"type": "array", "description": "Only for listings marked K with every requirement Y: where each requirement is stated.", "items": {
        "type": "object", "properties": {
            "n": {"type": "integer", "description": "The listing's number."},
            "where": strings("For each requirement in order: title, options, or the fact's label exactly as written."),
            "words": strings("For each requirement in order: the listing's exact words that state it."),
            "rank_value": {"type": "number", "description": "The listing's number for the shopper's priority when one is given, else 0."}},
        "required": ["n", "where", "words"]}},
    "queries": strings("When no listing meets every requirement: up to five other short searches worth trying; otherwise empty."),
}, ["marks"])

NAMES_TOOL = tool_spec("names", "What each text names for one property.", {
    "marks": {"type": "array", "items": {"type": "string"}, "description": "One entry per text, in order: '<number> <only|among|other|none>'."}},
    ["marks"])

WORDS_TOOL = tool_spec("words", "Other words sellers use for a value.", {
    "words": {"type": "array", "items": {"type": "string"}, "description": "Up to six, the most usual first; empty when there is none."}},
    ["words"])

RANK_TOOL = tool_spec("rank", "Each listing's figure for the shopper's priority.", {
    "values": {"type": "array", "items": {"type": "string"}, "description": "One entry per listing, in order: '<number> <figure>' with the figure as a plain number in one unit for every listing, or '<number> none' when the listing's words state none."},
    "better": {"type": "string", "enum": ["higher", "lower"], "description": "Whether a higher or a lower figure is better for this priority."}},
    ["values", "better"])

SEARCH_TOOL = tool_spec("search_plan", "New catalogue searches.", {
    "queries": strings("Six new searches of two to five words each, different from those tried."),
}, ["queries"])

REPLY_TOOL = tool_spec("reply", "A short answer to the shopper.", {
    "text": {"type": "string", "description": "One or two plain sentences that answer the shopper's line with the facts given. A statement, never a question."}},
    ["text"])

PLAN_GUIDE = """You prepare a purchase for a shopping assistant. Read the shopper's request and fill the plan tool from its words only.
- item: the words the shopper uses for the kind of thing, copied completely and exactly as written, with a leading word such as 'other' and any '&' or '/' kept (the store files its listings under such names); do not shorten them or replace them with a broader or narrower kind.
- requirements: every detail the request gives about the item itself is a requirement: its material, its colour, an amount with its unit (how much it holds, how big, heavy or long it is, how many pieces, how strong), its pattern, shape, style, kind or type, an occasion or use named together with the item, the make or thing the item must fit, and the like. A request for one kind of thing nearly always carries such a detail: read the request again before leaving the list empty. Name each property the way the shopper names it; when the request gives only the value, use the plain name of that kind of detail. Give the value in the fewest words: the word or amount itself (an amount with its unit in letters), without the property's name and without the kind of thing it belongs to. One phrase of the request gives one requirement; do not split it into two. The budget, being in stock and the category are not requirements, and neither is why the shopper is buying, which comes in a sentence of its own.
- doubted_pair: a request sometimes wishes for two things and doubts one item can offer both, in any wording; the shopper will say which matters more when asked. Give the two wishes; leave it empty when the request voices no such doubt.
- purposes: every use or recipient the request gives (what the item is for, whom it is for); each can carry a need the shopper has not stated yet.
- written: the same value written another way - another spelling of the word (grey: gray) or the same amount in another unit or wording (1 kg: 1000g, 1 kilo). alike: other words sellers use for the same value (navy: dark blue; sofa: couch), not a part or a wider kind of it.
- queries: searches as a shopper types them, two to five words, built from the item words and the required values. Write an amount the way listings do (the number joined to its unit)."""

HEAR_GUIDE = """You follow a shopper's conversation with a shopping assistant. Given what is already known and the shopper's new line, fill the hear tool with what the line changes.
- adds: a requirement of the item the line states (a property with its value), the property named in the shopper's words and the value in the fewest words (the word or amount itself). A purpose alone is not a property; the property the purpose calls for is. The budget, the category and being in stock are not properties. written: the same value written another way (another spelling of the word; the same amount in another unit: 1 kg is 1000g or 1 kilo). alike: other words sellers use for the same value (navy: dark blue; sofa: couch), not a part or a wider kind of it.
- drops: a property whose earlier requirement the line replaces or withdraws.
- A new line can hold several sentences or lines: read them in order; about the same property, a later one overrides an earlier one.
- A line that puts one value in place of another, withdraws one, or adds one is a change of what the shopper wants, even when it is worded as a question or a confirmation (for example ending in 'right?'): drop the old property and add the new value.
- budget: only when the line names a new limit.
- asks: true when the line puts a question to the assistant or expects an answer; a remark that needs no answer is false.
- about_change: true when the line reacts to the chosen item's stock or price having just changed (the assistant answers that by choosing again). A line can do that and also change a requirement: then fill adds and drops too.
- queries: searches of two to five words that combine the item words with the value just added."""

JUDGE_GUIDE = """You check store listings against a shopper's requirements. For every listing shown, say whether it is the kind of thing asked for and, for each requirement in order, whether the listing's own words state that value.
- A requirement is met when the listing's own words state the value: the same word, a plain synonym or another spelling of it, or the same amount written in another unit.
- Where to look: a colour or a size is what the options say, because the options describe the exact variant sold (when the options name another colour or amount, the requirement is unmet whatever the title lists); a material or an amount is stated by the title, the options or a fact; any other property the store keeps as a fact of its own is stated by the fact whose label names that property.
- When a listing shows 'the fact named for requirement N', that fact decides requirement N: met when it states the value, unmet when it states something else - unless the options state the value.
- A fact that states the value settles it even when the title's words suggest otherwise. A value that is absent is unmet; do not assume.
- An amount counts when it is one of several measures written for the variant (a size written as 40*30*5cm states 5 cm; a range such as 4-6cm includes 5 cm). An amount filed under a fact for a different quantity does not state it.
- marks: one entry per listing, in order, as '<number> <K or X> <Y or N for each requirement>'.
- proof: only for the listings that are K with every requirement Y - the place (title, options, or the fact's label exactly as written) and the listing's exact words that state each value; rank_value is the listing's number for the shopper's priority when one is given.
Mark every listing, in order."""

NAMES_GUIDE = """You read what a listing's own words say about one property of the item (its colour, or the material it is made of). For each numbered text answer with one word:
- only: the words name the wanted value - the same word, or another word sellers use for that same value - and no other value of that property.
- among: the words name the wanted value together with another value of that property (two colours, two materials), so it is unclear which one the item is.
- other: the words name a value of that property, but not the wanted one.
A shade or tint of the wanted colour (dark, light, off-, or a describing word before the colour) is the wanted colour; a name that joins two different colours is another colour.
The variant's words describe the exact item sold: a material they name is the item's material, whatever the title lists.
- none: the words name no value of that property.
Judge only the property asked about; sizes, codes, counts and other details are not values of it. Mark every text, in order."""

WORDS_GUIDE = """A shopper wants a certain value for a property of an item. List the other words or spellings that sellers use in listing titles and variant names for that same value: synonyms and other spellings of a word, or the same amount in other units and notations. Only words that mean the same value; not a part, a shade or a wider kind of it."""

RANK_GUIDE = """A shopper said what makes one fitting item better than another. For each listing, read from its own words (the options first, then its facts, then its title) the figure that this priority is about, and write it as a plain number in the same unit for every listing (convert where needed). Write none when the listing states no such figure; do not guess."""

SEARCH_GUIDE = """A shopping assistant has not yet found a store listing that meets every requirement of the shopper. Given the request, the requirements, the searches already tried and the titles of the listings seen, write new short catalogue searches that are likely to reach a listing meeting every requirement: combine the required values with each other and with the words such listings carry in their titles, use other words sellers use for a value, and try the values without the item words. Do not repeat a search already tried."""

REPLY_GUIDE = """You are a shopping assistant answering the shopper's lines; each one changed what the shopper wants. Answer each line in turn, in plain sentences, as statements: confirm the change in the shopper's own terms - yes, now the new value in place of the old one (or the added or dropped requirement) - and that the rest stays as it was. Name no item and no price, and do not list the shopper's other requirements. Do not ask anything and do not end with a question."""


# ----------------------------------------------------------------------------- inference
class Reader:
    """Language-model calls with a fallback model and a spending guard."""

    def __init__(self) -> None:
        provider = getenv("INFERENCE_PROVIDER", "chutes")
        self.models = MODELS.get(provider, MODELS["chutes"])
        self.spent = 0.0
        self.calls = 0

    def out_of_budget(self) -> bool:
        return self.spent >= SPEND_CAP_USD

    def fill(self, guide: str, text: str, tool: dict, max_tokens: int, second: bool = False, think: bool = False) -> dict | None:
        name = tool["function"]["name"]
        for model in (self.models[1:] if second else self.models):
            body = {"model": model, "temperature": 0, "max_tokens": max(max_tokens, MIN_TOKENS) + (THINK_TOKENS if think else 0),
                    "messages": [{"role": "system", "content": guide}, {"role": "user", "content": text}],
                    "tools": [tool], "tool_choice": {"type": "function", "function": {"name": name}}}
            # a thinking pass spends the completion budget before the tool call is written
            body["reasoning"] = {"enabled": think} if re.search(r"glm|kimi", model, re.I) else {"effort": "low"}
            reply = _proxy.post("/inference/chat/completions", json_data=body)
            self.calls += 1
            if not isinstance(reply, dict):
                continue
            usage = reply.get("usage") or {}
            cost = usage.get("cost")
            if isinstance(cost, (int, float)):
                self.spent += float(cost)
            else:
                self.spent += (int(usage.get("prompt_tokens") or 0) + 4 * int(usage.get("completion_tokens") or 0)) * 1e-6
            parsed = self._arguments(reply, name)
            if parsed is not None:
                return parsed
        return None

    @staticmethod
    def _closed(raw: str) -> dict | None:
        """Arguments cut off at the length limit: the longest prefix that ends with a complete value, closed."""
        cuts = [i + 1 for i, ch in enumerate(raw) if ch in "]}\""]
        for cut in reversed(cuts[-400:]):
            head, stack, quoted, escaped = raw[:cut].rstrip().rstrip(","), [], False, False
            for ch in head:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    quoted = not quoted
                elif not quoted and ch in "[{":
                    stack.append("]" if ch == "[" else "}")
                elif not quoted and ch in "]}" and stack:
                    stack.pop()
            if quoted:
                continue
            try:
                parsed = json.loads(head + "".join(reversed(stack)))
            except ValueError:
                continue
            if isinstance(parsed, dict):
                return parsed
        return None

    @staticmethod
    def _arguments(reply: dict, name: str) -> dict | None:
        try:
            message = reply["choices"][0]["message"]
        except (KeyError, IndexError, TypeError):
            return None
        for call in message.get("tool_calls") or []:
            raw = (call.get("function") or {}).get("arguments")
            try:
                parsed = json.loads(raw) if isinstance(raw, str) else raw
            except (TypeError, ValueError):
                parsed = Reader._closed(raw) if isinstance(raw, str) else None
            if isinstance(parsed, dict):
                return parsed
        content = message.get("content")
        if isinstance(content, str) and "{" in content:
            try:
                parsed = json.loads(content[content.index("{"): content.rindex("}") + 1])
            except ValueError:
                return None
            if isinstance(parsed, dict):
                return parsed.get(name) if isinstance(parsed.get(name), dict) else parsed
        return None


# ----------------------------------------------------------------------------- the store session
class Counter:
    """One environment session: turns, tool calls and what the store has shown so far."""

    def __init__(self, problem: dict) -> None:
        environment = problem["environment"]
        self.binding = dict(environment["binding"])
        policy = environment["policy_view"]
        self.request = str(policy.get("query") or "")
        self.tools = {t["function"]["name"]: t["function"] for t in policy.get("tools") or [] if isinstance(t, dict) and "function" in t}
        self.turn_limit = int(policy.get("max_steps") or 20)
        self.group_size = max(1, min(CALLS_PER_TURN, int(policy.get("max_calls_per_turn") or 1)))
        self.task = str(problem.get("problem_id") or problem.get("id") or "task")
        self.turn = 0
        self.over = False
        self.dialogue: list[dict] = []
        self.rows: dict[tuple[str, str], dict] = {}      # (product_id, sku) -> search / filter row
        self.row_order: list[tuple[str, str]] = []
        self.shelf: dict[tuple[str, str], str] = {}      # rows a category filter returned -> the label that returned them
        self.pages: dict[str, dict] = {}                 # product_id -> page from view / compare
        self.stock: dict[tuple[str, str], dict] = {}     # (product_id, sku) -> latest price / stock seen
        self.cart: list[tuple[str, str]] = []
        self.heard: list[str] = []                       # shopper lines not yet handled

    def turns_left(self) -> int:
        return self.turn_limit - self.turn

    def has(self, tool: str) -> bool:
        return tool in self.tools

    def send(self, actions: list[dict]) -> list[dict]:
        """One turn: the actions in order. Returns one observation per action sent."""
        actions = [a for a in actions if a["name"] in self.tools][: self.group_size]
        if not actions or self.over or self.turns_left() <= 0:
            return []
        self.turn += 1
        group = f"{self.task}-turn-{self.turn}"
        envelope = {**self.binding, "call_id": group, "idempotency_key": group, "turn": self.turn,
                    "calls": [{"call_id": f"{group}-{i}", "action": {"name": a["name"], "args": a["args"]}} for i, a in enumerate(actions, start=1)]}
        result = _proxy.post("/environment/call", json_data=envelope)
        if not isinstance(result, dict) or not isinstance(result.get("calls"), list):
            self.over = True
            self.dialogue.append({"role": "assistant", "content": "", "tool_calls": actions, "environment_result": None})
            return []
        self.dialogue.append({"role": "assistant", "content": "", "tool_calls": actions, "environment_result": result})
        observations = []
        for action, call in zip(actions, result["calls"]):
            step = call.get("observation") if isinstance(call, dict) else None
            step = step if isinstance(step, dict) else {}
            observation = step.get("observation") if isinstance(step.get("observation"), dict) else {}
            if step.get("done") is True:
                self.over = True
            self._learn(action, observation)
            observations.append(observation)
        said = result.get("user_message")
        if isinstance(said, dict) and str(said.get("content") or "").strip():
            self.heard.append(str(said["content"]).strip())
        return observations

    def _note(self, product_id: Any, sku: Any, price: Any, in_stock: Any) -> None:
        if product_id is None or sku is None:
            return
        entry = self.stock.setdefault((str(product_id), str(sku)), {})
        entry["turn"] = self.turn
        if isinstance(price, (int, float)) and not isinstance(price, bool):
            entry["price"] = float(price)
        if isinstance(in_stock, bool):
            entry["in_stock"] = in_stock

    def _learn(self, action: dict, observation: dict) -> None:
        name = action["name"]
        if name in ("search", "filter"):
            for row in observation.get("results") or []:
                if isinstance(row, dict) and row.get("product_id") and row.get("sku"):
                    key = (str(row["product_id"]), str(row["sku"]))
                    if key not in self.rows:
                        self.rows[key] = row
                        self.row_order.append(key)
                    if name == "filter" and action["args"].get("category"):
                        self.shelf.setdefault(key, str(action["args"]["category"]))
                    self._note(key[0], key[1], row.get("price"), row.get("in_stock"))
        elif name in ("view", "compare"):
            for page in ([observation] if name == "view" else observation.get("compared") or []):
                if isinstance(page, dict) and page.get("product_id"):
                    self.pages[str(page["product_id"])] = page
                    for variant in page.get("variants") or []:
                        if isinstance(variant, dict):
                            self._note(page["product_id"], variant.get("sku"), variant.get("price"), variant.get("in_stock"))
        elif name == "inspect_stock":
            self._note(observation.get("product_id"), observation.get("sku"), observation.get("price"), observation.get("in_stock"))
        elif name == "add_to_cart":
            added = observation.get("added") if isinstance(observation.get("added"), dict) else None
            if added:
                key = (str(added.get("product_id")), str(added.get("sku")))
                if key not in self.cart:
                    self.cart.append(key)
                self._note(key[0], key[1], added.get("price"), added.get("in_stock"))


# ----------------------------------------------------------------------------- the shopper's order
class Errand:
    def __init__(self, problem: dict) -> None:
        self.counter = Counter(problem)
        self.reader = Reader()
        self.started = monotonic()
        self.item = ""
        self.category = ""
        self.requirements: list[dict] = []     # {"property", "value"}
        self.budget: float | None = None
        self.brand = ""
        self.listed: list[tuple[str, str]] = []
        self.priority = ""
        self.queries: list[str] = []
        self.sent: set[str] = set()            # searches and filters already sent
        self.shelf_words_used = 0
        self.questions_asked = 0
        self.verdicts: dict[tuple[str, str], dict] = {}
        self.judged_under = ""
        self.judged_count = 0
        self.search_room = SEARCH_LIMIT        # searches allowed so far; a change by the shopper opens more
        self.open_room = OPEN_LIMIT            # listings that may be read so far
        self.judge_room = JUDGE_LIMIT          # listings the reader may be asked about so far
        self.gone: set[tuple[str, str]] = set()
        self.facts_for_reply: list[str] = []
        self.pending_line = ""
        self.owed_reaction = False             # a market change was just seen: the shopper's next question is about it
        self.reader_asked = False
        self.own_cache: dict = {}
        self.widened = False
        self.named_as: dict[tuple, str] = {}   # (listing, property, value) -> what the reader found its words to name
        self.name_checks = 0
        self.awaiting_answer = False           # a question was just put to the shopper
        self.probed: set[str] = set()          # titles whose other variants were already searched for
        self.lenient = False                   # nothing fits by the firm rules: a value stated anywhere on the page counts
        self.shelf_cache: tuple[int, str] = (-1, "")
        self.kin_done: set[str] = set()        # titles whose other listings and variants were already searched for
        self.later_questions: list[str] = []   # a further purpose, asked once the first question is answered
        self.pair_fallback = ""                # asked when the answer to the pair question names nothing
        self.set_aside: list[dict] = []        # requirements read as the wishes of a doubted pair, until the answer
        self.pending_filters: list[dict] = []  # store filters to send with the next searches
        self.search_asks = 0                   # times the reader wrote new searches because nothing fitted
        self.rank_asks = 0
        self.better = ""                       # the reader's word on the priority: a higher or a lower figure is better
        self.lost_titles: set[str] = set()     # titles of chosen listings that went out of stock or over the budget
        self.unanswered: list[str] = []        # lines that changed what the shopper wants, not yet replied to

    def note(self, text: str) -> None:
        self.counter.dialogue.append({"role": "assistant", "content": f"note: {text}"})

    def late(self) -> bool:
        return monotonic() - self.started > EPISODE_SECONDS

    def can_think(self) -> bool:
        return not self.reader.out_of_budget()

    # ---- the request
    def read_request(self) -> str:
        request = self.counter.request
        plan = self.reader.fill(PLAN_GUIDE, f"The shopper's request:\n{request}", PLAN_TOOL, 700, think=True) or {}
        if not [r for r in plan.get("requirements") or [] if isinstance(r, dict) and str(r.get("value") or "").strip()]:
            # a request without any detail of the item is rare: have the other model read it too
            again = self.reader.fill(PLAN_GUIDE, f"The shopper's request:\n{request}", PLAN_TOOL, 700, second=True) or {}
            if again.get("requirements"):
                plan = {**plan, **{k: v for k, v in again.items() if v}} if plan else again
        self.item = str(plan.get("item") or "").strip()
        self.category = str(plan.get("category") or "").strip()
        self.requirements = [self.requirement(r) for r in plan.get("requirements") or [] if isinstance(r, dict) and str(r.get("value") or "").strip()]
        self.requirements = [dict(r, stated=True) for r in self.requirements if not NOT_A_PROPERTY.search(lowered(f"{r['property']} {r['value']}"))]
        self.budget = money(plan.get("budget"))
        if self.budget is None:
            amounts = [float(m.replace(",", "")) for m in re.findall(r"(\d[\d,]*\.\d{2})", request)]
            self.budget = max(amounts) if amounts else None
        self.brand = str(plan.get("brand") or "").strip()
        self.listed = [(m.group(1), m.group(2)) for m in re.finditer(r"(\d{6,})::([\w-]+)", request)]
        self.priority = str(plan.get("priority") or "").strip()
        self.add_queries(self.value_queries() + list(plan.get("queries") or []))
        pair = [" ".join(str(w).split()) for w in plan.get("doubted_pair") or [] if str(w).strip()][:2]
        purposes = [" ".join(str(w).split()) for w in plan.get("purposes") or [] if str(w).strip()][:3]
        self.note(f"plan item={self.item!r} requirements={self.requirements} budget={self.budget} pair={pair} purposes={purposes} priority={self.priority!r}")
        if not self.queries:
            self.add_queries([" ".join(word_list(request)[:5])])
        purposes = [p for i, p in enumerate(purposes) if not any(i != j and self.within(p, q) and (not self.within(q, p) or j < i)
                                                                for j, q in enumerate(purposes))]
        if len(pair) == 2 and len(" ".join(pair).split()) <= 16 and not self.mistaken_pair(pair, purposes):
            # the shopper wants one of two things and says which when asked: that is the one open point; until
            # then neither wish is a firm requirement
            wished = lowered(" ".join(pair))
            self.set_aside = [r for r in self.requirements if lowered(r["value"]) in wished]
            self.requirements = [r for r in self.requirements if lowered(r["value"]) not in wished]
            self.pair_fallback = PLAIN_QUESTION
            return f"Which matters more to you: {pair[0]} or {pair[1]}?"
        # each further purpose can carry a need of its own: asked once the first answer is in
        self.later_questions = [f"And for {p}, what does it need to have?" for p in purposes[1:2] if len(p.split()) <= 12]
        # a request rarely tells everything the item must have: the purpose is asked even when the request names none
        return PLAIN_QUESTION

    @staticmethod
    def content_words(text: str) -> set[str]:
        return {singular(w) for w in word_list(text) if len(w) >= 4} - {"that", "what", "with", "this", "from", "have", "them", "they", "their"}

    def within(self, inner: str, outer: str) -> bool:
        """Every content word of one phrase stands in the other."""
        words = self.content_words(inner)
        return bool(words) and words <= self.content_words(outer)

    def mistaken_pair(self, pair: list[str], purposes: list[str]) -> bool:
        """A pair entry that shares a word with a purpose, or two with the priority, is not a wish for the item."""
        said = [self.content_words(p) for p in purposes]
        priority = self.content_words(self.priority)
        return any(any(self.content_words(w) & s for s in said) or len(self.content_words(w) & priority) >= 2 for w in pair)

    @staticmethod
    def requirement(raw: dict) -> dict:
        def texts(name: str) -> list[str]:
            return [str(w).strip() for w in raw.get(name) or [] if str(w).strip()][:6] if isinstance(raw.get(name), list) else []
        value = re.sub(r"(\d)\s*(?:\"|\u201d|\u2033|'')", r"\1 inch", str(raw.get("value") or "").strip())      # 5" is 5 inch
        name = lowered(raw.get("property"))
        filed = not re.search(r"\d", value) and not re.search(r"colou?r|material", name)
        return {"filed": filed, "property": str(raw.get("property") or "").strip(), "value": value,
                "written": texts("written"), "alike": texts("alike")}

    def state_text(self) -> str:
        lines = [f"Request: {self.counter.request}", f"Item: {self.item}",
                 "Requirements: " + ("; ".join(f"{r['property']} = {r['value']}" for r in self.requirements) or "(none)"),
                 f"Budget: {self.budget if self.budget is not None else 'none'}"]
        if self.priority:
            lines.append(f"Priority between fitting items: {self.priority}")
        return "\n".join(lines)

    # ---- searches
    def add_queries(self, queries: list, first: bool = False) -> None:
        fresh = []
        for query in queries:
            text = lowered(" ".join(str(query or "").split())[:QUERY_LENGTH])
            if text and text not in self.sent and text not in self.queries and text not in fresh:
                fresh.append(text)
        self.queries = fresh + self.queries if first else self.queries + fresh

    def values(self) -> list[str]:
        """Each required value as a search word, in the shopper's own spelling (the request and the shopper's lines
        give the store's own values): an amount closed up, without a word that only repeats the property's name."""
        out = []
        for requirement in self.requirements:
            named = self.name_words(requirement["property"])
            words = word_list(closed_up(requirement["value"]))
            out.append(" ".join([w for w in words if singular(w) not in named] or words))
        return out

    def item_words(self) -> list[str]:
        return [w for w in word_list(self.category or self.item) if len(w) >= 3 and w not in ("and", "the", "for", "with", "other")]

    def value_queries(self) -> list[str]:
        """The required values with the item words: every value, the newest alone, and each item word in turn."""
        values, item = self.values(), self.category or self.item
        label = self.store_label()
        if label and values and lowered(label) != lowered(item):
            self.add_queries([f"{' '.join(values)} {label}", f"{values[-1]} {label}"])
        if not values or not item:
            return [item] if item else []
        every, newest = " ".join(values), values[-1]
        out = [f"{every} {item}", f"{newest} {item}", f"{values[0]} {item}"]
        for requirement in self.requirements:       # the same search in the value's other spelling
            for other in (requirement.get("written") or [])[:3]:
                if closed_up(other) != closed_up(requirement["value"]):
                    out.append(f"{' '.join(closed_up(other if r is requirement else r['value']) for r in self.requirements)} {item}")
        out += [f"{every} {w}" for w in self.item_words()[:3]]
        if len(values) >= 2:
            out.append(every)               # the values alone: listings that state them all, whatever the title calls the item
        out += [f"{newest} {w}" for w in self.item_words()[:2]]
        return out

    def shelf_queries(self, count: int) -> list[str]:
        """Words the store's own listings of this category carry most often, each with the required values:
        a search shows ten rows, so other words reach other listings of the same shelf."""
        titles = [self.counter.rows[k].get("title") for k in self.counter.shelf if k in self.counter.rows]
        titles += [p.get("title") for p in self.counter.pages.values() if self.in_category(p)]
        if len(titles) < 3 or not self.requirements:
            return []
        known = set(self.item_words()) | {w for v in self.values() for w in word_list(closed_up(v))}
        counts: dict[str, int] = {}
        for title in titles:
            for word in set(word_list(title)):
                if len(word) >= 4 and not word.isdigit() and word not in known:
                    counts[word] = counts.get(word, 0) + 1
        common = [w for w, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])) if n >= 2]
        chosen = common[self.shelf_words_used: self.shelf_words_used + count]
        self.shelf_words_used += len(chosen)
        every, newest = " ".join(self.values()), self.values()[-1]
        head = (self.item_words() or [""])[-1]
        # (with the item's own word too for half of them: a listing may name the kind of thing in other words - tab, pad)
        return ([f"{w} {every}" for w in chosen] + [f"{w} {every} {head}" for w in chosen[: count // 2]]
                + [f"{w} {newest}" for w in chosen[: count // 2]])

    def kin_titles(self) -> list[str]:
        return list(dict.fromkeys(lowered((self.counter.pages.get(k[0]) or {}).get("title")) for k in self.verdicts if self.fits_words(k)))

    def kin_pending(self) -> bool:
        return len(self.kin_done) < 4 and any(title not in self.kin_done for title in self.kin_titles())

    def kin_queries(self) -> list[str]:
        """The required values with the title of a fitting listing: so specific a search returns that article's
        other variants and its other listings."""
        every = " ".join(self.values())
        out = []
        for key in [k for k in self.verdicts if self.fits_words(k)]:
            page = self.counter.pages.get(key[0]) or {}
            title = lowered(page.get("title"))
            if title in self.kin_done or len(self.kin_done) >= 4:
                continue
            self.kin_done.add(title)
            words = word_list(title)
            if len(words) < 2:
                continue
            out += [f"{every} {' '.join(words[:10])}", f"{every} {' '.join(words[:5])}", f"{every} {self.item} {' '.join(words[:3])}"]
            if len(words) > 10:
                out.append(f"{every} {' '.join(words)}")
            options = " ".join(word_list(self.variant_of(key).get("options")))
            if options:
                out.append(f"{options} {' '.join(words[:10])}")     # the listing's own words for this variant
            brand = lowered(page.get("brand"))
            if brand and brand not in ("none", "no brand", "oem"):
                out.append(f"{every} {brand} {self.item}")
            out.append(f"{every} {' '.join(words[:2])}")
            elsewhere = {w for row in self.counter.rows.values() if lowered(row.get("title")) != title for w in word_list(row.get("title"))}
            own = sorted((w for w in dict.fromkeys(words) if len(w) >= 5 and not w[0].isdigit() and w not in elsewhere), key=len, reverse=True)
            if own:
                out.append(own[0])          # a word no other listing carries leads straight to this article's variants
        return out

    def variant_queries(self) -> list[str]:
        """A search shows one variant of a listing. When a listing meets everything except what differs between
        its variants (the options shown name another colour or amount), the wanted variant may stand behind the
        one shown: the listing's title with the wanted value brings it up."""
        out = []
        for key, verdict in list(self.verdicts.items()):
            if len(self.probed) >= 4:
                break
            if not verdict.get("kind_ok") or verdict.get("by") != "words" or not self.requirements:
                continue
            levels = [self.plain(key, r) for r in self.requirements]
            off = [r for r, level in zip(self.requirements, levels) if level < 1]
            rest = [level for r, level in zip(self.requirements, levels) if level >= 1]
            if not off or len(rest) != len(self.requirements) - len(off):
                continue
            if any(self.kind(r) not in ("colour", "amount") for r, level in zip(self.requirements, levels) if level < 1):
                continue
            title = lowered((self.counter.pages.get(key[0]) or {}).get("title"))
            if not title or title in self.probed:
                continue
            self.probed.add(title)
            words = " ".join(word_list(title)[:10])
            for requirement in off:
                for word in [closed_up(requirement["value"])] + list(requirement.get("alike") or [])[:2]:
                    out.append(f"{word} {words}")
        return out

    def label_guesses(self) -> list[str]:
        """The item words as a category name, with plural endings as stores write them; a describing word in
        front may not belong to the name, so the same is tried without the first word or two."""
        base = self.category or self.item
        if not base:
            return []

        def plural(word: str) -> str:
            if word.endswith("s"):
                return word
            if re.search(r"(x|ch|sh)$", word):
                return word + "es"
            if re.search(r"[^aeiou]y$", word):
                return word[:-1] + "ies"
            return word + "s"

        out = []
        parts = base.split()
        for skip in range(min(3, len(parts))):
            phrase = " ".join(parts[skip:])
            if skip and not re.match(r"[A-Za-z]{3}", phrase):
                continue
            last = re.sub(r"([A-Za-z]+)(\W*)$", lambda m: plural(m.group(1)) + m.group(2), phrase)
            every = re.sub(r"([A-Za-z]+)(?=\s*(?:&|/|,|\band\b)|\W*$)", lambda m: plural(m.group(1)), phrase)
            out += [phrase, last, every] if not skip else [last]
        return list(dict.fromkeys(out))

    def store_label(self) -> str:
        """The category label the store answers to: one a filter returned rows for, or the one read on a page."""
        named = self.named_shelf()
        if named:
            return named
        if self.counter.shelf:
            return next(iter(self.counter.shelf.values()))
        wanted = {label_key(self.category), label_key(self.item)} - {""}
        for page in self.counter.pages.values():
            for label in page.get("category_path") or []:
                if label_key(label) in wanted:
                    return str(label)
        return ""

    def shelf_filters(self, count: int = 4) -> list[dict]:
        """The category at several price ceilings: each ceiling shows other listings of the shelf."""
        label = self.store_label()
        if not label or not self.counter.has("filter") or self.budget is None:
            return []
        out = []
        for share in (0.7, 0.45, 0.25, 0.12, 0.85, 0.55, 0.35, 0.18, 0.08, 0.04):
            tag = f"filter|{lowered(label)}|{share}"
            if tag not in self.sent and len(out) < count and len(self.sent) < self.search_room:
                self.sent.add(tag)
                out.append(act("filter", category=label, max_price=round(self.budget * share, 2), in_stock=True, k=10))
        return out

    def queue_filters(self) -> None:
        """The shelf under the present budget, and under the brand the shopper requires, to send with the next searches."""
        brand = next((r["value"] for r in self.requirements if "brand" in lowered(r["property"])), None)
        label = self.store_label()
        for category in ([label] if label else self.label_guesses()[:2]):
            self.pending_filters.append(act("filter", category=category, brand=brand, max_price=self.budget, in_stock=True, k=10))
        if brand:
            self.pending_filters.append(act("filter", brand=brand, max_price=self.budget, in_stock=True, k=10))

    def next_searches(self, room: int) -> list[dict]:
        out = self.pending_filters[:room] if self.counter.has("filter") else []
        self.pending_filters = self.pending_filters[len(out):] if out else []
        while self.queries and len(out) < room and len(self.sent) < self.search_room:
            query = self.queries.pop(0)
            if query in self.sent:
                continue
            self.sent.add(query)
            out.append(act("search", query=query, k=10, in_stock=True))
        return out

    # ---- the shopper's lines
    @staticmethod
    def listed_fields(answer: dict | None) -> dict:
        """A reader's answer with its lists as lists (a model sometimes writes a list as text)."""
        answer = dict(answer or {})
        for name in ("adds", "drops", "queries"):
            if isinstance(answer.get(name), str):
                try:
                    answer[name] = json.loads(answer[name])
                except ValueError:
                    answer[name] = []
            if not isinstance(answer.get(name), list):
                answer[name] = []
        return answer

    def hear(self) -> tuple[bool, bool]:
        """Take in the shopper lines received. Returns (something changed, a line expects an answer)."""
        changed = asks = False
        while self.counter.heard:
            line = self.counter.heard.pop(0)
            parts = [part.strip() for part in line.split("\n") if part.strip()]
            if len(parts) > 1:
                # several lines in one turn: each is heard on its own, in order (only the first answers a question)
                self.counter.heard[:0] = parts
                continue
            before = [(r["property"], r["value"]) for r in self.requirements]
            said = f"{self.state_text()}\n\nThe shopper's new line:\n{line}"
            heard = self.listed_fields(self.reader.fill(HEAR_GUIDE, said, HEAR_TOOL, 1500) if self.can_think() else None)
            if self.awaiting_answer and not [a for a in heard.get("adds") or [] if isinstance(a, dict)] and self.can_think():
                # the line answers the question just asked, and the first reading found no requirement in it
                again = self.listed_fields(self.reader.fill(HEAR_GUIDE, said, HEAR_TOOL, 1500, second=True))
                if [a for a in again.get("adds") or [] if isinstance(a, dict)]:
                    heard = again
            answering, self.awaiting_answer = self.awaiting_answer, False
            if heard.get("about_change") is True:
                # a line reacting to the market change: its figure is a price, not a budget (a requirement it states still counts)
                heard["budget"] = 0
            for name in heard.get("drops") or []:
                kept = [r for r in self.requirements if lowered(r["property"]) != lowered(name)]
                changed = changed or len(kept) != len(self.requirements)
                self.requirements = kept
            for add in heard.get("adds") or []:
                if not isinstance(add, dict) or not str(add.get("value") or "").strip():
                    continue
                new = self.requirement(add)
                if NOT_A_PROPERTY.search(lowered(f"{new['property']} {new['value']}")):
                    continue
                same = next((r for r in self.requirements if lowered(r["property"]) == lowered(new["property"])), None)
                if same is not None and lowered(same["value"]) == lowered(new["value"]):
                    continue
                if same is not None:
                    self.requirements.remove(same)
                if answering and self.pair_fallback and self.questions_asked == 1:
                    new["chosen"] = True        # the side of the doubted pair the shopper said matters more
                self.requirements.append(new)
                changed = True
                if "brand" in lowered(new["property"]):
                    # a brand the shopper now requires: ask the store for it directly
                    self.queue_filters()
                    self.add_queries([f"{new['value']} {self.item}", new["value"]], first=True)
            budget = money(heard.get("budget"))
            if budget is not None and any(abs(budget - float((self.counter.stock.get(k) or {}).get("price") or -1.0)) < 0.005 for k in self.counter.cart):
                budget = None               # the price of an item in the cart, not a new limit
            if budget is not None and budget != self.budget:
                lowered_limit = self.budget is None or budget < self.budget
                self.budget, changed = budget, True
                if lowered_limit:
                    self.queue_filters()        # the shelf again, under the new limit
            if str(heard.get("priority") or "").strip():
                self.priority, changed = str(heard["priority"]).strip(), True
            if not answering and [(r["property"], r["value"]) for r in self.requirements] != before:
                self.unanswered.append(line)    # the shopper changed what they want: a plain reply before the order
            if changed:
                self.add_queries(self.value_queries() + list(heard.get("queries") or []), first=True)
                # what the shopper wants moved: room to look again, whatever was used before
                self.search_room = max(self.search_room, len(self.sent) + 16)
                self.open_room = max(self.open_room, len(self.counter.pages) + 40)
                self.judge_room = max(self.judge_room, self.judged_count + 36)
            self.note(f"heard {line[:80]!r} -> adds={heard.get('adds')} drops={heard.get('drops')} budget={heard.get('budget')} asks={heard.get('asks')}")
            if heard.get("asks") and not heard.get("about_change"):
                if self.owed_reaction:
                    self.owed_reaction = False
                else:
                    asks = True
                    self.pending_line = line
            elif heard.get("about_change"):
                self.owed_reaction = False
        return changed, asks

    # ---- reading listings
    def affordable(self, key: tuple[str, str]) -> bool:
        seen = self.counter.stock.get(key) or {}
        price = seen.get("price")
        if seen.get("in_stock") is False or key in self.gone:
            return False
        return self.budget is None or not isinstance(price, float) or price <= self.budget + 1e-9

    def named_shelf(self) -> str:
        """The store's own label for the shelf the request names: of the labels read on listings, the longest
        whose words stand together in the request."""
        pages = self.counter.pages
        if self.shelf_cache[0] == len(pages):
            return self.shelf_cache[1]
        request = f" {label_key(self.counter.request)} "
        seen: dict[str, int] = {}
        for page in pages.values():
            path = [str(label) for label in page.get("category_path") or []]
            if path and label_key(path[-1]) and f" {label_key(path[-1])} " in request:
                seen[path[-1]] = seen.get(path[-1], 0) + 1
        own = {label_key(self.category), label_key(self.item)} - {""}
        best = max(seen, key=lambda label: (len(label_key(label).split()), label_key(label) in own, seen[label]), default="")
        self.shelf_cache = (len(pages), best)
        return best

    def in_category(self, page: dict) -> bool:
        named = self.named_shelf()
        wanted = {label_key(named)} if named else {label_key(self.category), label_key(self.item)} - {""}
        labels = {label_key(label) for label in page.get("category_path") or []}
        return bool(wanted & labels)

    def variant_of(self, key: tuple[str, str]) -> dict:
        page = self.counter.pages.get(key[0]) or {}
        return next((v for v in page.get("variants") or [] if str(v.get("sku")) == key[1]), {})

    @staticmethod
    def name_words(text: Any) -> set[str]:
        words = {singular(w) for w in word_list(str(text).replace("_", " "))} - {"of"}
        return {"color" if w == "colour" else w for w in words}

    @staticmethod
    def forms(requirement: dict, loose: bool = False) -> set[str]:
        """The required value in each way it may be written, as plain words; loose: the words sellers use alike."""
        spellings = list(requirement.get("written") or [])
        if requirement.get("filed"):        # (a value filed as a fact is filed in the store's own word)
            written = spellings + list(requirement.get("alike") or []) if loose else [requirement["value"]]
        elif re.search(r"\d", requirement["value"]):
            if re.search(r"\d\s*[a-z]", lowered(requirement["value"])):      # a figure alone is no measurement of an amount with a unit
                spellings = [form for form in spellings if not re.fullmatch(r"[\d.,\s]+", str(form).strip())]
            written = list(requirement.get("alike") or []) if loose else [requirement["value"]] + spellings
        else:                               # a word: another spelling of it is the word, anything else is a seller's other word
            close = [form for form in spellings if SequenceMatcher(None, lowered(form), lowered(requirement["value"])).ratio() >= 0.75]
            written = [form for form in spellings if form not in close] + list(requirement.get("alike") or []) if loose else [requirement["value"]] + close
        return {" ".join(singular(w) for w in word_list(closed_up(form))) for form in written} - {""}

    def forms_under(self, requirement: dict, label: str) -> set[str]:
        """Under a fact's label the value may drop the words the label already says (a make is filed by its
        name alone, not by the name and the kind of thing)."""
        named = self.name_words(label) | {"at", "the", "of", "a", "an", "in", "on", "for", "with", "its"}
        out = set(self.forms(requirement))
        for form in list(out):
            words = form.split()
            kept = [w for w in words if not any(same_word(w, n) for n in named)]
            if kept and len(kept) < len(words):
                out.add(" ".join(kept))
        return out

    def states(self, text: Any, requirement: dict, label: str = "") -> bool:
        """One entry of a fact (facts often list several) is exactly the required value."""
        entries = {" ".join(singular(w) for w in word_list(reading(part))) for part in re.split(r"[,;/|]+", lowered(text)) for reading in (lowered, closed_up)}
        if not label:
            return bool(entries & self.forms(requirement))
        named = self.name_words(label)
        for entry in list(entries):         # an entry may repeat the label's own words
            kept = [w for w in entry.split() if not any(same_word(w, n) for n in named)]
            if kept:
                entries.add(" ".join(kept))
        return bool(entries & self.forms_under(requirement, label))

    def kind(self, requirement: dict) -> str:
        """Where a store states a property: an amount; a colour, which the variant's options name; a material,
        which the listing's name carries; anything else, which is kept as a fact of its own."""
        if re.search(r"\d", requirement["value"]):
            return "amount"
        name = lowered(requirement["property"])
        if re.search(r"colou?r", name):
            return "colour"
        if "material" in name:
            return "material"
        return "fact"

    @staticmethod
    def entries(text: Any) -> list[str]:
        """The separate entries of a fact (a fact may list several), as plain words."""
        parts = (" ".join(singular(w) for w in word_list(closed_up(part))) for part in re.split(r"[,;/|]+", lowered(text)))
        return [part for part in parts if part]

    def entry_says(self, entry: str, requirement: dict, label: str, loose: bool = False) -> bool:
        """An entry filed under a label states the value: it carries every word of one of the value's forms.
        Words the label itself says do not have to be repeated, and may be repeated."""
        forms = self.forms(requirement, loose=True) if loose else self.forms_under(requirement, label)
        have = entry.split()
        return any(all(any(same_word(n, h) for h in have) for n in form.split()) for form in forms)

    def entry_is(self, entry: str, requirement: dict, label: str) -> bool:
        """The entry is the value and nothing else."""
        named = self.name_words(label)
        kept = " ".join(w for w in entry.split() if not any(same_word(w, n) for n in named)) or entry
        forms = self.forms_under(requirement, label)
        return entry in forms or kept in forms

    def own_labels(self, requirement: dict) -> tuple[set[str], set[str], set[str]]:
        """The fact labels that are this property's own, and the values the shelf files under them:
        the labels named like the property (one name holding all words of the other); the labels under which
        the shelf's listings state exactly such a value, as their only entry, more than once; and the single
        words that stand as entries under those labels."""
        pages = self.counter.pages
        stamp = (requirement["property"], requirement["value"], len(requirement.get("written") or []), len(pages), self.named_shelf())
        cached = self.own_cache.get(requirement["property"])
        if cached and cached[0] == stamp:
            return cached[1]
        wanted = self.name_words(requirement["property"])
        kind = self.kind(requirement)
        shelf = [page for page in pages.values() if self.in_category(page)] or list(pages.values())
        written = " ".join(word_list(requirement["property"]))
        tiers: list[set[str]] = [set(), set(), set(), set()]     # named exactly so; the same words; one name within the other
        units = {m.group(2) for form in self.forms(requirement) for m in re.finditer(r"(\d+(?:\.\d+)?)\s*([a-z]+)", form)}
        stated: dict[str, int] = {}
        for page in shelf:
            for label, value in (page.get("facts") or {}).items():
                have = self.name_words(label)
                if wanted and have:
                    if " ".join(word_list(str(label).replace("_", " "))) == written:
                        tiers[0].add(str(label))
                    elif have == wanted:
                        tiers[1].add(str(label))
                    elif have <= wanted or wanted <= have:
                        # a name that adds only the required amount's unit (power_w for 1200 W) is the property's own;
                        # one that adds another word (suction_power) may name another measure
                        tiers[2 if (have ^ wanted) <= units else 3].add(str(label))
                entries = self.entries(value)
                if len(entries) == 1 and self.entry_is(entries[0], requirement, str(label)):
                    stated[str(label)] = stated.get(str(label), 0) + 1
        related = next((tier for tier in tiers if tier), set())
        most = max(stated.values(), default=0)
        inferred: set[str] = set()
        if kind != "colour" and not any(stated.get(label) for label in related):
            # no fact named like the property carries such a value anywhere on the shelf: the label that does
            inferred = {label for label, count in stated.items() if count >= 2 and 3 * count >= most}
        words: set[str] = set()
        for page in shelf:
            for label, value in (page.get("facts") or {}).items():
                if str(label) in related or str(label) in inferred:
                    words |= {entry for entry in self.entries(value) if " " not in entry}
            if kind == "colour":        # a word that stands alone as a variant's name is a value variants are told apart by
                for variant in page.get("variants") or []:
                    words |= {entry for entry in self.entries(variant.get("options")) if re.fullmatch(r"[a-z]{3,}", entry)}
        answer = (related, inferred, words)
        self.own_cache[requirement["property"]] = (stamp, answer)
        return answer

    def named_facts(self, page: dict, requirement: dict) -> list[tuple[str, str]]:
        related, inferred, _ = self.own_labels(requirement)
        return [(str(label), str(value)) for label, value in (page.get("facts") or {}).items() if str(label) in related or str(label) in inferred]

    def says(self, text: Any, requirement: dict, whole: bool = False, loose: bool = False) -> bool:
        """The text carries every word of the required value, in one of the ways it may be written.
        whole: only words standing on their own count (a word joined to another by a hyphen is a different word)."""
        if whole:
            have = {singular(w) for reading in (lowered(text), closed_up(text)) for w in re.split(r"[\s/,;()\[\]|+]+", reading) if w}
            return any(set(form.split()) <= have for form in self.forms(requirement, loose))
        have = {singular(w) for reading in (lowered(text), closed_up(text)) for w in word_list(reading)}
        return any(all(any(same_word(n, h) for h in have) for n in form.split()) for form in self.forms(requirement, loose))

    def amounts(self, text: Any, requirement: dict) -> tuple[set[tuple[float, str]], set[tuple[float, str]]]:
        """The amounts a text states in the units of the required value: (those that are the value, the others)."""
        wanted: set[tuple[float, str]] = set()
        for form in self.forms(requirement):
            wanted |= {(float(m.group(1)), singular(m.group(2))) for m in re.finditer(r"(\d+(?:\.\d+)?)\s*([a-z]+)", form)}
        units = {unit for _, unit in wanted}
        found = {(float(m.group(1)), singular(m.group(2))) for m in re.finditer(r"(\d+(?:\.\d+)?)\s*([a-z]+)", lowered(text))}
        found = {pair for pair in found if pair[1] in units}
        return found & wanted, found - wanted

    @staticmethod
    def other_words(tokens: set[str]) -> set[str]:
        """Plain words among option tokens: not a code with digits or signs, and not a size letter code (xl, xxl)."""
        return {w for w in tokens if re.fullmatch(r"[a-z]{3,}", w) and not re.fullmatch(r"x*[sml]|x+l", w)}

    def unit_in_label(self, label: str, requirement: dict) -> bool:
        """The fact's label names the unit of the required amount (capacity_l, display_size_inch)."""
        units = {m.group(2) for form in self.forms(requirement) for m in re.finditer(r"(\d+(?:\.\d+)?)\s*([a-z]+)", form)}
        return bool(units & self.name_words(label))

    def has_amount(self, text: Any, requirement: dict) -> bool:
        """The text states the required amount: the same number with the same unit, however it is spaced."""
        return bool(self.amounts(text, requirement)[0]) or self.says(text, requirement)

    def other_amount(self, text: Any, requirement: dict) -> bool:
        """The text states an amount in a unit of the required value, but not that amount."""
        same, others = self.amounts(text, requirement)
        return bool(others) and not same and not self.says(text, requirement)

    def plain(self, key: tuple[str, str], requirement: dict) -> int:
        """How plainly the page states one required value: 3 where the store states that kind of property (the
        variant's options, the property's own fact), 2 in the listing's name, 1 unclearly (beside other values, in
        another fact, in another word for it), 0 nowhere, -1 when the page states something else for it."""
        page = self.counter.pages.get(key[0]) or {}
        options = str(self.variant_of(key).get("options") or "")
        title = page.get("title")
        kind = self.kind(requirement)
        related, inferred, words = self.own_labels(requirement)
        own = [(label, self.entries(value)) for label, value in self.named_facts(page, requirement)]
        singles = [(label, entries[0]) for label, entries in own if len(entries) == 1]
        lists = [(label, entry) for label, entries in own if len(entries) > 1 for entry in entries]
        stated = any(self.entry_says(entry, requirement, label) for label, entry in singles)
        listed = any(self.entry_says(entry, requirement, label) for label, entry in lists)
        value_words = {w for form in self.forms(requirement) for w in form.split()}
        elsewhere = any(self.says(value, requirement) for value in (page.get("facts") or {}).values())

        if kind == "amount":
            if self.other_amount(options, requirement):
                return -1           # the variant's own options state another amount of that kind
            if self.has_amount(options, requirement):
                return 3
            number = re.search(r"\d+(?:\.\d+)?", requirement["value"])
            bare = [label for label, entry in singles if label in related and number and re.fullmatch(r"\d+(?:\.\d+)?", entry) and float(entry) == float(number.group(0))]
            if any(self.unit_in_label(label, requirement) for label in bare) or any(self.has_amount(entry, requirement) for _, entry in singles):
                return 3
            if bare:
                # the property's own fact gives the figure without its unit: the same number, but no measurement on its own
                return 2 if self.has_amount(title, requirement) else 1
            if any(re.search(r"\d", entry) for _, entry in singles):
                return -1           # the property's own fact gives another figure
            if self.has_amount(title, requirement):
                return 1 if self.amounts(title, requirement)[1] else 2      # (a title with several such amounts lists its variants)
            return 1 if listed or any(self.has_amount(value, requirement) for value in (page.get("facts") or {}).values()) else 0

        named_as = self.named_as.get((key, requirement["property"], requirement["value"]))
        if kind == "colour":
            wanted = self.forms(requirement)
            parts = [{singular(w) for reading in (lowered(part), closed_up(part)) for w in re.split(r"[\s()\[\]+]+", reading) if w}
                     for part in re.split(r"[/,;|]+", options)]
            named = [part for part in parts if any(set(form.split()) <= part for form in wanted)]
            if named:
                # the part of the options that names the colour: another colour word beside it makes it another colour
                alone = any(not ((part & words) - value_words) and not self.other_words(part - value_words) for part in named)
                return 3 if alone and named_as not in ("among", "other") else 1
            if any(form.replace(" ", "") in lowered(options).replace(" ", "") for form in wanted):
                return -1           # the word joined to other characters is a code or another option's name, not this colour
            if stated:
                return 1 if options.strip() else 3      # (options that name the variant otherwise leave it unclear)
            alike = any(set(form.split()) <= part for form in self.forms(requirement, loose=True) for part in parts)
            if alike or named_as in ("only", "among"):
                return 1            # another word for the colour: the reader's or the seller's
            if options.strip():
                return -1           # the options name the variant: when they say something else, the variant is something else
            return 1 if listed else 0

        if kind == "material":
            if stated:
                return 3
            if named_as in ("among", "other"):
                return 1 if named_as == "among" else -1
            contrary = bool(singles)        # its own fact names another material
            if named_as == "only":
                return 1 if contrary else 2
            if self.says(title, requirement):
                named_too = ({singular(w) for w in word_list(title)} & words) - value_words
                return 1 if named_too or contrary else 2    # (another material beside it: unclear what the item is made of)
            if listed or self.says(title, requirement, loose=True):
                return 1
            if contrary:
                return -1
            return 1 if elsewhere else 0

        # a property the store keeps as a fact of its own
        if stated:
            return 3
        if listed or any(self.entry_says(entry, requirement, label, loose=True) for label, entry in singles):
            return 1
        if own:
            return -1               # its own fact states something else
        if related:
            return -1               # the store files this property as a fact, and this listing has none
        if inferred:
            return 0
        return 1 if elsewhere or self.says(title, requirement) else 0

    def exact(self, key: tuple[str, str], requirement: dict) -> int:
        """How firmly the page states the value: each of the property's own facts whose only entry is exactly the
        value counts for the listing, each that states something else counts against it; options that are
        exactly the value count for it."""
        page = self.counter.pages.get(key[0]) or {}
        options = str(self.variant_of(key).get("options") or "")
        score = 0
        for label, value in self.named_facts(page, requirement):
            entries = self.entries(value)
            if len(entries) == 1:
                score += 1 if self.entry_is(entries[0], requirement, label) else (0 if self.entry_says(entries[0], requirement, label) else -1)
        return score + int(self.states(options, requirement))

    def level_needed(self, requirement: dict) -> int:
        """The level at which a value counts as stated where the store states that kind of property."""
        return 2 if self.kind(requirement) in ("amount", "material") else 3

    def settled(self, key: tuple[str, str], requirement: dict) -> bool:
        """The page's own words state the value, plainly or not."""
        return self.plain(key, requirement) >= 1

    def row_score(self, key: tuple[str, str]) -> float:
        row = self.counter.rows.get(key) or {}
        title = stems(row.get("title"))
        score = 4.0 if key in self.counter.shelf else 0.0
        for requirement in self.requirements:
            need = stems(requirement["value"])
            others = [stems(form) for form in list(requirement.get("written") or []) + list(requirement.get("alike") or [])]
            if need and need <= title or any(other and other <= title for other in others):
                score += 3
            elif need & title:
                score += 1
        score += sum(1 for w in {singular(w) for w in self.item_words()} if w in title)
        if lowered(row.get("title")) in self.kin_titles() or lowered(row.get("title")) in self.probed:
            score += 20
        elif "-" in key[1] and key[1].rsplit("-", 1)[0] in {k[1].rsplit("-", 1)[0] for k in self.verdicts if "-" in k[1] and self.fits_words(k)}:
            score += 20             # another variant of a fitting listing
        return score + (50 if self.listed and key in self.listed else 0)

    def to_open(self, limit: int, allowance: int | None = None) -> list[dict]:
        allowance = self.open_room if allowance is None else allowance
        unopened = [k for k in self.counter.row_order if k[0] not in self.counter.pages and self.affordable(k)]
        if self.listed:
            unopened = [k for k in self.listed if k[0] not in self.counter.pages] + [k for k in unopened if k in self.listed]
        ranked = sorted(dict.fromkeys(unopened), key=lambda k: -self.row_score(k))
        ids = list(dict.fromkeys(k[0] for k in ranked))[: max(0, min(limit, allowance - len(self.counter.pages)))]
        if self.counter.has("compare"):
            return [act("compare", product_ids=ids[i:i + OPEN_BATCH]) for i in range(0, len(ids), OPEN_BATCH)]
        return [act("view", product_id=i) for i in ids]

    def candidates(self) -> list[tuple[str, str]]:
        out = []
        any_shelved = any(self.in_category(p) for p in self.counter.pages.values())
        for product_id, page in self.counter.pages.items():
            if any_shelved and not self.in_category(page):
                continue           # the store files the wanted kind under its own label: only those listings count
            for variant in page.get("variants") or []:
                key = (product_id, str(variant.get("sku")))
                if self.affordable(key) and (not self.listed or key in self.listed):
                    out.append(key)
        return out

    def card(self, number: int, key: tuple[str, str]) -> str:
        page = self.counter.pages[key[0]]
        named_words = {w for r in self.requirements for w in self.name_words(r["property"])}
        value_words = {w for r in self.requirements for w in stems(r["value"])}
        close, other = [], []
        for label, value in (page.get("facts") or {}).items():
            if self.name_words(label) & named_words or stems(value) & value_words:
                close.append(f"{label} = {str(value)[:110]}")
            else:
                other.append(f"{label} = {str(value)[:36]}")
        facts = "; ".join(close + other[:5])
        for position, requirement in enumerate(self.requirements, start=1):
            named = self.named_facts(page, requirement)
            if named:
                facts += f" || the fact named for requirement {position}: " + "; ".join(f"{label} = {value[:110]}" for label, value in named)
        price = (self.counter.stock.get(key) or {}).get("price")
        return (f"[{number}] title: {str(page.get('title') or '')[:150]} | options: {self.variant_of(key).get('options') or '-'} | price: {price} | "
                f"category: {' > '.join(str(x) for x in (page.get('category_path') or [])[-2:])} | brand: {page.get('brand') or '-'} | facts: {facts or '-'}")

    def judge(self) -> None:
        """Settle every unjudged candidate by the page's own words."""
        shelf_words = set(label_key(self.named_shelf()).split())
        self.requirements = [r for r in self.requirements
                             if not (r.get("stated") and shelf_words and set(label_key(r["value"]).split()) <= shelf_words)]
        self.name_check()
        stamp = json.dumps([self.requirements, self.priority, self.lenient, len(self.named_as)], sort_keys=True)
        if stamp != self.judged_under:
            self.verdicts, self.judged_under = {}, stamp
        for key in self.candidates():
            if key not in self.verdicts:
                in_shelf = self.in_category(self.counter.pages[key[0]])
                met = [self.settled(key, r) or (self.lenient and self.plain(key, r) >= 1) for r in self.requirements]
                self.verdicts[key] = {"kind_ok": in_shelf, "met": met, "by": "words"}

    def ask_reader(self) -> None:
        """Have the reader look at the nearest misses: a synonym, another spelling or another unit may state what plain words did not find."""
        open_keys = [k for k in self.verdicts if self.affordable(k) and self.verdicts[k].get("by") == "words" and self.verdicts[k].get("kind_ok")
                     and not self.fits_words(k) and not any(self.plain(k, r) == -1 for r in self.requirements)]
        open_keys.sort(key=lambda k: -sum(self.plain(k, r) for r in self.requirements))
        batch = open_keys[:JUDGE_CARDS]
        if not batch or not self.can_think() or self.judged_count >= self.judge_room:
            return
        self.judged_count += len(batch)
        wanted = "\n".join(f"{i}. {r['property']}: {r['value']}" for i, r in enumerate(self.requirements, start=1)) or "(none beyond the kind of thing)"
        text = (f"The shopper wants: {self.item}\nRequirements, in order:\n{wanted}\n"
                + (f"Priority between fitting items: {self.priority}\n" if self.priority else "")
                + "\nListings:\n" + "\n".join(self.card(i, key) for i, key in enumerate(batch, start=1)))
        answer = self.reader.fill(JUDGE_GUIDE, text, JUDGE_TOOL, 1400) or {}
        marks = answer.get("marks")
        marks = re.split(r"[\n;,]+", marks) if isinstance(marks, str) else (marks or [])
        for mark in marks:
            found = re.match(r"\s*\[?(\d+)\]?\W+([KkXx])\W*([YyNn\s,]*)", str(mark))
            if not found or not 1 <= int(found.group(1)) <= len(batch):
                continue
            key = batch[int(found.group(1)) - 1]
            said = [x in "Yy" for x in re.findall(r"[YyNn]", found.group(3))]
            said += [False] * (len(self.requirements) - len(said))
            # the reader may see a word stated in other words; an amount has to be found written on the page
            met = [self.settled(key, r) or (said[i] and self.plain(key, r) != -1 and not re.search(r"\d", r["value"])) for i, r in enumerate(self.requirements)]
            self.verdicts[key] = {"kind_ok": found.group(2) in "Kk", "met": met, "by": "reader"}
        for key in batch:
            if self.verdicts[key].get("by") == "words":
                self.verdicts[key]["by"] = "reader"      # shown once: not shown again
        for proof in answer.get("proof") or []:
            if isinstance(proof, dict) and isinstance(proof.get("n"), int) and 1 <= proof["n"] <= len(batch):
                self.verdicts[batch[proof["n"] - 1]].update({k: proof.get(k) for k in ("where", "words", "rank_value")})
        self.note(f"reader judged {len(batch)}: " + " | ".join(str(m)[:12] for m in marks)[:260])
        self.add_queries(answer.get("queries") or [])

    def fits_words(self, key: tuple[str, str]) -> bool:
        """The listing's words meet every requirement, whatever its stock is now."""
        verdict = self.verdicts.get(key) or {}
        met = verdict.get("met") or []
        return bool(verdict.get("kind_ok")) and len(met) >= len(self.requirements) and all(met[: len(self.requirements)])

    def fits(self, key: tuple[str, str]) -> bool:
        if not self.fits_words(key) or not self.affordable(key):
            return False
        page = self.counter.pages.get(key[0]) or {}
        shelved = self.in_category(page) or not any(self.in_category(p) for p in self.counter.pages.values())
        brand = not self.brand or set(word_list(self.brand)) <= set(word_list(page.get("brand")))
        return shelved and brand

    def ranked(self) -> list[tuple[str, str]]:
        """Fitting listings, best first: the shopper's priority, then the page that states the values most plainly, then price."""
        if self.better in ("higher", "lower"):
            direction = -1.0 if self.better == "higher" else 1.0
        else:
            direction = -1.0 if re.search(r"\b(high|larg|big|long|most|max|best)", lowered(self.priority)) else 1.0
        by_price = self.priced_priority()

        def order(key: tuple[str, str]) -> tuple:
            verdict = self.verdicts[key]
            price = (self.counter.stock.get(key) or {}).get("price") or 0.0
            value = price if by_price else verdict.get("rank_value")
            known = isinstance(value, (int, float)) and not isinstance(value, bool)
            plainly = sum(self.plain(key, r) for r in self.requirements) + sum(self.exact(key, r) for r in self.requirements)
            twin = lowered((self.counter.pages.get(key[0]) or {}).get("title")) in self.lost_titles
            return (0 if known or not self.priority else 1, direction * float(value) if known and self.priority else 0.0,
                    0 if self.firm(key) else 1, 0 if self.solid(key) else 1, 0 if self.verdicts[key].get("by") == "words" else 1, -plainly,
                    0 if twin else 1, price)

        return sorted((k for k in self.verdicts if self.fits(k)), key=order)

    def keeps_request(self, key: tuple[str, str]) -> bool:
        """The listing is the kind asked for and states every requirement the request itself wrote, where the store
        states that kind of property: the first cart is the shopper's first sight of the choice and holds to those."""
        stated = [r for r in self.requirements if r.get("stated")]
        return bool((self.verdicts.get(key) or {}).get("kind_ok")) and all(self.plain(key, r) >= self.level_needed(r) for r in stated)

    def meets_chosen(self, key: tuple[str, str]) -> bool:
        """The listing meets what the shopper said matters more (the chosen side of a doubted pair), when there is one."""
        met = ((self.verdicts.get(key) or {}).get("met") or [])[: len(self.requirements)]
        marks = [x for x, r in zip(met, self.requirements) if r.get("chosen")]
        return bool(marks) and all(marks)

    def priced_priority(self) -> bool:
        """The shopper's priority is about the price itself."""
        return bool(re.search(r"\b(cheap|price|cost|expensive|afford)", lowered(self.priority)))

    def rank_by_priority(self) -> None:
        """When the shopper said what makes one fitting item better, read each fitting listing's figure for it."""
        if not self.priority or self.priced_priority() or not self.can_think() or self.rank_asks >= 3:
            return
        fitting = self.ranked()
        keys = [k for k in fitting if "rank_value" not in self.verdicts[k]][:JUDGE_CARDS]
        if len(fitting) < 2 or not keys:
            return
        self.rank_asks += 1
        text = (f"Item: {self.item}\nThe shopper's priority: {self.priority}\n\nListings:\n"
                + "\n".join(self.card(i, key) for i, key in enumerate(keys, start=1)))
        answer = self.reader.fill(RANK_GUIDE, text, RANK_TOOL, 400) or {}
        if str(answer.get("better") or "").lower() in ("higher", "lower"):
            self.better = str(answer["better"]).lower()
        values = answer.get("values")
        values = re.split(r"[\n;]+", values) if isinstance(values, str) else (values or [])
        for entry in values:
            found = re.match(r"\s*\[?(\d+)\]?\W+(-?\d+(?:\.\d+)?)", str(entry))
            if found and 1 <= int(found.group(1)) <= len(keys):
                self.verdicts[keys[int(found.group(1)) - 1]]["rank_value"] = float(found.group(2))
        for key in keys:
            self.verdicts[key].setdefault("rank_value", None)      # read once: not asked again
        self.note(f"ranked by {self.priority!r}: " + " ".join(str(v)[:14] for v in values)[:200])

    def rethink_searches(self) -> bool:
        """Nothing found meets every requirement: the reader writes new searches from the request, the requirements,
        the searches tried and the titles seen, and the agent looks again. True when it did."""
        if self.search_asks >= 2 or not self.can_think() or self.late() or self.counter.turns_left() <= TURN_RESERVE + 2:
            return False
        self.search_asks += 1
        seen = list(dict.fromkeys(str(p.get("title") or "")[:90] for p in self.counter.pages.values()))[-25:]
        text = (f"{self.state_text()}\n\nSearches tried:\n" + "\n".join(sorted(self.sent)[-25:])
                + "\n\nTitles seen (none meets every requirement):\n" + "\n".join(seen))
        answer = self.listed_fields(self.reader.fill(SEARCH_GUIDE, text, SEARCH_TOOL, 400))
        queries = [q for q in answer.get("queries") or [] if isinstance(q, str) and q.strip()][:6]
        self.note(f"new searches: {queries}")
        if not queries:
            return False
        self.add_queries(queries, first=True)
        self.look(rounds=3, enough=1)
        return True

    def recart(self) -> tuple[str, str] | None:
        """The next fitting listing into the cart, its stock and price read again."""
        counter = self.counter
        for key in [k for k in self.ranked() if k not in self.gone][:2]:
            if counter.over or counter.turns_left() < 3:
                break
            group = [act("add_to_cart", product_id=key[0], sku=key[1])]
            if counter.has("inspect_stock"):
                group.append(act("inspect_stock", product_id=key[0], sku=key[1]))
            counter.send(group)
            self.hear()
            seen = counter.stock.get(key) or {}
            if seen.get("in_stock") is not False and (self.budget is None or not isinstance(seen.get("price"), float) or seen["price"] <= self.budget + 1e-9):
                return key
            self.gone.add(key)
        return None

    def reply_before_paying(self, choice: tuple[str, str]) -> None:
        """The lines the shopper said on their own get one plain reply before the order: each line in turn, with the
        item, its price and stock."""
        seen = self.counter.stock.get(choice) or {}
        title = str((self.counter.pages.get(choice[0]) or {}).get("title") or "")[:80]
        stock = "out of stock" if seen.get("in_stock") is False else "in stock"
        self.facts_for_reply.append(f"The item being ordered: {title}, price {seen.get('price')}, {stock}.")
        lines, self.unanswered = self.unanswered[-4:], []
        self.answer("\n".join(f"- {line}" for line in lines))

    def last_resort(self) -> None:
        """After a failure: still order the listing in the cart, or the best one left."""
        counter = self.counter
        if counter.over or counter.turns_left() < 1 or not counter.has("place_test_order"):
            return
        alive = [k for k in counter.cart if k not in self.gone]
        key = alive[-1] if alive else next((k for k in self.ranked() if k not in self.gone), None)
        if key is None:
            return
        if key not in counter.cart:
            if counter.turns_left() < 2:
                return
            counter.send([act("add_to_cart", product_id=key[0], sku=key[1])])
        counter.send([act("place_test_order", product_id=key[0], sku=key[1], justification=self.claims(key))])

    def near_misses(self) -> list[tuple[str, str]]:
        def score(key: tuple[str, str]) -> tuple:
            verdict = self.verdicts[key]
            met = (verdict.get("met") or [])[: len(self.requirements)]
            chosen = sum(1 for x, r in zip(met, self.requirements) if x and r.get("chosen"))
            # what the shopper said matters more outranks the count of other requirements met
            return (-int(bool(verdict.get("kind_ok"))), -chosen, -sum(1 for x in met if x), (self.counter.stock.get(key) or {}).get("price") or 0.0)
        return sorted((k for k in self.verdicts if self.affordable(k)), key=score)

    # ---- looking
    def hear_quietly(self) -> None:
        if self.counter.heard:
            self.hear()

    def name_check(self) -> None:
        """For a colour or a material, have the reader say what the words of the likeliest listings name: the
        wanted value alone, the wanted value beside another, or another value. Sellers write a colour in other
        words, and a title that names two materials leaves open which one the item is made of."""
        for requirement in self.requirements:
            kind = self.kind(requirement)
            if kind not in ("colour", "material") or not self.can_think() or self.name_checks >= 10:
                continue
            others = [r for r in self.requirements if r is not requirement]
            waiting = []
            for key in self.candidates():
                tag = (key, requirement["property"], requirement["value"])
                if tag in self.named_as or any(self.plain(key, r) < 1 for r in others):
                    continue
                page = self.counter.pages.get(key[0]) or {}
                options = str(self.variant_of(key).get("options") or "").strip()
                if kind == "colour":
                    text = f"variant: {options}" if options else f"title: {str(page.get('title') or '')[:150]}"
                else:
                    text = f"title: {str(page.get('title') or '')[:150]}" + (f" | variant: {options}" if options else "")
                waiting.append((tag, text))
            waiting = waiting[:12]
            if not waiting:
                continue
            self.name_checks += 1
            asked = (f"Item: {self.item}\nProperty: {requirement['property']}\nWanted value: {requirement['value']}\n\nTexts:\n"
                     + "\n".join(f"[{i}] {text}" for i, (_, text) in enumerate(waiting, start=1)))
            answer = self.reader.fill(NAMES_GUIDE, asked, NAMES_TOOL, 500) or {}
            marks = answer.get("marks")
            marks = re.split(r"[\n;,]+", marks) if isinstance(marks, str) else (marks or [])
            numbered = all(re.match(r"\s*\[?\d+", str(mark)) for mark in marks)
            if numbered or len(marks) == len(waiting):      # (marks without numbers count only when there is one per text)
                for position, mark in enumerate(marks, start=1):
                    word = re.search(r"\b(only|among|other|none)\b", lowered(mark))
                    number = re.match(r"\s*\[?(\d+)", str(mark))
                    at = int(number.group(1)) if numbered else position
                    if word and 1 <= at <= len(waiting):
                        self.named_as[waiting[at - 1][0]] = word.group(1)
            for tag, _ in waiting:
                self.named_as.setdefault(tag, "unread")      # asked once, not asked again
            self.note(f"names {requirement['value']}: " + " ".join(f"{i}:{self.named_as[tag]}" for i, (tag, _) in enumerate(waiting, start=1))[:240])

    def widen(self) -> None:
        """Too few listings fit by the words known: ask what else sellers call each value, and search for those."""
        self.widened = True
        for requirement in self.requirements:
            if not self.can_think():
                break
            answer = self.reader.fill(WORDS_GUIDE, f"Item: {self.item}\nProperty: {requirement['property']}\nValue: {requirement['value']}", WORDS_TOOL, 200) or {}
            words = [str(w).strip() for w in answer.get("words") or [] if str(w).strip()][:6]
            name = "written" if re.search(r"\d", requirement["value"]) else "alike"
            requirement[name] = list(dict.fromkeys(list(requirement.get(name) or []) + words))
            others = " ".join(r["value"] for r in self.requirements if r is not requirement)
            self.add_queries([f"{word} {others} {self.item}" for word in words[:3]], first=True)
        self.note(f"widened: {[(r['value'], r.get('written'), r.get('alike')) for r in self.requirements]}")

    def refill(self, lost: tuple[str, str], wide: bool = True) -> None:
        """The chosen variant has gone out of stock. A search shows one variant of each listing - the best in
        stock - so the searches that led to the lost one now show that listing's next variant, and its own title
        alone shows the same article listed again. wide: also search the values with it and the shelf at its price."""
        counter = self.counter
        words = word_list((counter.pages.get(lost[0]) or {}).get("title"))
        if len(words) < 2 or not counter.has("search"):
            return
        every = " ".join(self.values())
        own = " ".join(w for w in word_list(self.variant_of(lost).get("options")) if not w.replace(".", "").isdigit())
        queries = [" ".join(words), " ".join(words[:10])]
        if own:
            queries.append(f"{own} {' '.join(words[:10])}")       # the lost variant's own words: its nearest siblings share them
        queries += [f"{every} {' '.join(words[:10])}", f"{every} {' '.join(words[:5])}", f"{every} {self.item} {' '.join(words[:3])}"]
        if wide:
            queries += [f"{every} {' '.join(words)}"] + self.shelf_queries(6)
        texts = list(dict.fromkeys(lowered(" ".join(q.split())[:QUERY_LENGTH]) for q in queries))
        calls = [act("search", query=text, k=10, in_stock=True) for text in texts]
        label, price = self.store_label(), (counter.stock.get(lost) or {}).get("price")
        if wide and label and isinstance(price, float) and counter.has("filter"):
            # the same shelf at the lost variant's price: variants of one listing share their price
            ceiling = round(min(self.budget or price * 1.1, price * 1.1), 2)
            brand = str((counter.pages.get(lost[0]) or {}).get("brand") or "").strip()
            calls.append(act("filter", category=label, max_price=ceiling, in_stock=True, k=10))
            if brand and lowered(brand) not in ("none", "no brand", "oem"):
                calls.append(act("filter", category=label, brand=brand, max_price=ceiling, in_stock=True, k=10))
        counter.send(calls[: counter.group_size])
        self.hear_quietly()
        opens = self.to_open(4 * OPEN_BATCH, allowance=len(counter.pages) + 4 * OPEN_BATCH)
        if opens and not counter.over:
            counter.send(opens)
            self.hear_quietly()
        self.judge()

    def solid(self, key: tuple[str, str]) -> bool:
        """Every value is stated as required - in the options, the property's own fact or the title - not only
        in another word for it or in some other fact."""
        return (self.verdicts.get(key) or {}).get("by") == "words" and all(self.plain(key, r) >= 2 for r in self.requirements)

    def firm(self, key: tuple[str, str]) -> bool:
        """Every value is stated where it counts most: the options or the property's own fact (an amount: the title too)."""
        if (self.verdicts.get(key) or {}).get("by") != "words":
            return False
        return all(self.plain(key, r) >= self.level_needed(r) for r in self.requirements)

    def look(self, rounds: int, enough: int = 2) -> None:
        """Search and read listings until enough of them fit, or the rounds run out. Every search loads the
        store, so none is sent once enough listings fit firmly."""
        counter = self.counter
        for round_number in range(rounds):
            if counter.over or counter.turns_left() <= TURN_RESERVE or self.late():
                break
            opens = self.to_open(counter.group_size * OPEN_BATCH)
            if opens:
                counter.send(opens[: counter.group_size])
                self.hear_quietly()
            self.judge()
            fitting = [k for k in self.ranked() if self.solid(k)]
            if len([k for k in fitting if self.firm(k)]) >= enough:
                return
            if len(fitting) >= enough and not self.kin_pending() and round_number >= 2:
                return
            self.add_queries(self.kin_queries(), first=True)
            if len(fitting) < enough:
                self.add_queries(self.variant_queries(), first=True)
            if len(self.queries) < counter.group_size // 2:
                self.add_queries(self.shelf_queries(6))
            searches = self.shelf_filters(counter.group_size // 2)
            searches += self.next_searches(counter.group_size - len(searches))
            if not searches and not opens:
                break
            if searches:
                counter.send(searches)
                self.hear_quietly()
        opens = self.to_open(counter.group_size * OPEN_BATCH)
        if opens and not counter.over and counter.turns_left() > TURN_RESERVE:
            counter.send(opens[: counter.group_size])
            self.hear_quietly()
        self.judge()
        if len(self.ranked()) < enough and not self.widened and not counter.over:
            self.widen()
            return self.look(rounds=3, enough=enough)
        for _ in range(2):                  # plain words left too few: the reader may see the value stated in other words
            if len(self.ranked()) >= enough:
                return
            self.ask_reader()
        if len(self.ranked()) < enough and not self.lenient:
            self.lenient = True
            self.judge()

    # ---- the cart and the order
    def claims(self, key: tuple[str, str]) -> dict | None:
        seen = self.counter.stock.get(key) or {}
        page = self.counter.pages.get(key[0]) or {}
        places = {"title": str(page.get("title") or ""), "options": str(self.variant_of(key).get("options") or "")}
        places.update({str(label): str(value) for label, value in (page.get("facts") or {}).items()})
        claims: list[dict] = []
        if isinstance(seen.get("price"), float):
            claims.append({"field": "price", "value": seen["price"]})
        claims.append({"field": "in_stock", "value": True})
        verdict = self.verdicts.get(key) or {}
        for where, words in zip(verdict.get("where") or [], verdict.get("words") or []):
            where = re.sub(r"^facts?\s*[:.]\s*", "", str(where or ""))
            field = next((label for label in places if lowered(label) == lowered(where)), None)
            said = " ".join(word_list(words))
            if field and said and f" {said} " in f" {' '.join(word_list(places[field]))} " and not any(c["field"] == field for c in claims):
                claims.append({"field": field, "value": str(words).strip()})
        for requirement in self.requirements:        # the plain places that state each value, when the reader named none
            for field, text in places.items():
                value = " ".join(word_list(closed_up(requirement["value"])))
                if value and f" {value} " in f" {' '.join(word_list(closed_up(text)))} " and not any(c["field"] == field for c in claims):
                    found = re.search(re.escape(value).replace(r"\ ", r"\W+"), lowered(text))
                    if found:
                        claims.append({"field": field, "value": found.group(0)})
                        break
        return {"claims": claims}

    def answer(self, line: str) -> None:
        """Answer a shopper line that expects one, as a statement."""
        counter = self.counter
        if counter.over or not counter.has("message") or counter.turns_left() <= 2:
            return
        if line in self.unanswered:
            self.unanswered.remove(line)
        facts = "\n".join(self.facts_for_reply[-6:]) or "(nothing new)"
        reply = self.reader.fill(REPLY_GUIDE, f"Facts:\n{facts}\n\nThe shopper's line or lines:\n{line}", REPLY_TOOL, 300) if self.can_think() else None
        text = str((reply or {}).get("text") or "").strip().replace("?", ".")
        if text:
            counter.send([act("message", content=text[:400])])

    def run(self) -> list[dict]:
        counter = self.counter
        question = self.read_request()
        # first turn: the one open question (when the request leaves something open) together with the first searches
        first: list[dict] = []
        if question and counter.has("message"):
            first.append(act("message", content=question[:300]))
            self.questions_asked += 1
        if self.listed and counter.has("compare"):
            first.append(act("compare", product_ids=list(dict.fromkeys(k[0] for k in self.listed))[:OPEN_BATCH]))
        filters = [act("filter", category=guess, max_price=self.budget, in_stock=True, k=10) for guess in self.label_guesses()[:5]] if counter.has("filter") else []
        first.extend(self.next_searches(counter.group_size - len(first) - len(filters)))
        self.awaiting_answer = bool(question)
        counter.send(first + filters)
        known = len(self.requirements)
        self.hear()
        if self.pair_fallback and len(self.requirements) == known:
            # the answer to the pair named nothing: the two were not the open point, so the purpose is asked,
            # and what was set aside as a wish is a requirement again
            self.later_questions = [self.pair_fallback] + self.later_questions
            self.requirements += self.set_aside
            self.add_queries(self.value_queries(), first=True)
        for later in self.later_questions:
            if self.questions_asked >= 2 or not counter.has("message") or counter.over:
                break
            # the first purpose is answered; the next purpose is an open point of its own
            self.awaiting_answer = True
            counter.send([act("message", content=later[:300])] + self.next_searches(counter.group_size - 1))
            self.questions_asked += 1
            self.hear()
        self.look(rounds=9)
        commits = 0
        choice: tuple[str, str] | None = None
        while not counter.over and counter.turns_left() >= 2 and commits < 5:
            self.rank_by_priority()
            open_keys = [k for k in self.ranked() if k not in self.gone]
            if (not open_keys or not counter.cart and not any(self.keeps_request(k) for k in open_keys)) and self.rethink_searches():
                # nothing fits, or nothing for the first cart holds the request as written: look with new searches first
                continue
            order = open_keys or [k for k in self.near_misses() if k not in self.gone]
            if not order:
                break
            if not counter.cart:
                # the first cart: what the shopper said matters more, then the request as written (stable sort)
                order.sort(key=lambda k: (not self.meets_chosen(k), not self.keeps_request(k)))
            choice = order[0]
            commits += 1
            title = str((counter.pages.get(choice[0]) or {}).get("title") or "")[:50]
            fitting = f"{len(self.ranked())}: " + "; ".join(f"{str((counter.pages.get(k[0]) or {}).get('title') or '')[:28]}|{self.variant_of(k).get('options')}|{[self.plain(k, r) for r in self.requirements]}|{(self.verdicts.get(k) or {}).get('by')}" for k in self.ranked()[:4])
            self.note(f"commit {commits}: {choice} {title!r} fits={self.fits(choice)} turn={counter.turn} pages={len(counter.pages)} spent={self.reader.spent:.4f} calls={self.reader.calls} fitting: {fitting}")
            group = [act("add_to_cart", product_id=choice[0], sku=choice[1])]
            if counter.has("inspect_stock"):
                group.append(act("inspect_stock", product_id=choice[0], sku=choice[1]))
            counter.send(group)
            seen = counter.stock.get(choice) or {}
            price = seen.get("price")
            over_budget = self.budget is not None and isinstance(price, float) and price > self.budget + 1e-9
            changed, asks = self.hear()
            if seen.get("in_stock") is False or over_budget:
                self.gone.add(choice)
                self.lost_titles.add(lowered((counter.pages.get(choice[0]) or {}).get("title")))
                self.owed_reaction = True
                self.pending_line = ""     # the shopper's reaction to the change is answered by choosing again
                if counter.turns_left() > TURN_RESERVE and not self.late():
                    # the same article is often listed again, and it shares every fact the lost one was chosen for
                    self.refill(choice, wide=not any(self.firm(k) for k in self.ranked() if k not in self.gone))
                left = [k for k in self.ranked() if k not in self.gone]
                if len([k for k in left if self.firm(k)]) < 2 and counter.turns_left() > TURN_RESERVE and not self.late():
                    # fewer than two firm listings left: another sell-out would leave only weak ones, so look further now
                    # (the same article is often listed again, and the shelf has more to show)
                    self.kin_done.clear()
                    self.add_queries(self.kin_queries() + self.shelf_queries(6), first=True)
                    self.look(rounds=3 if left else 5, enough=2 if left else 1)
                choice = None
                continue
            if changed:
                self.judge()
                if not self.fits(choice):
                    self.gone.add(choice)
                    if not [k for k in self.ranked() if k not in self.gone] and counter.turns_left() > TURN_RESERVE and not self.late():
                        self.look(rounds=4, enough=1)
                    choice = None
                    continue
            if asks and self.pending_line:
                self.facts_for_reply.append(f"The listing now in the cart: {title}, {price}, in stock.")
                self.answer(self.pending_line)
                self.pending_line = ""
                more_changed, _ = self.hear()
                if more_changed:
                    continue
            if counter.turn > int((counter.stock.get(choice) or {}).get("turn") or 0) and counter.has("inspect_stock") and counter.turns_left() >= 3:
                # a turn went by since the choice was last read: the market may have moved, so read it again before paying
                counter.send([act("inspect_stock", product_id=choice[0], sku=choice[1])])
                late_change, _ = self.hear()
                seen = counter.stock.get(choice) or {}
                if seen.get("in_stock") is False or (self.budget is not None and isinstance(seen.get("price"), float) and seen["price"] > self.budget + 1e-9):
                    self.gone.add(choice)
                    self.lost_titles.add(lowered((counter.pages.get(choice[0]) or {}).get("title")))
                    self.owed_reaction = True
                    choice = None
                    continue
                if late_change:
                    # the shopper changed what they want while the choice waited: it must still fit
                    self.judge()
                    if not self.fits(choice):
                        self.gone.add(choice)
                        choice = None
                        continue
            if self.unanswered and counter.has("message") and counter.turns_left() >= 3:
                self.reply_before_paying(choice)
                if self.hear()[0]:
                    self.judge()
                    if not self.fits(choice):
                        self.gone.add(choice)
                        choice = None
                        continue
            break
        if choice is None and counter.cart and not counter.over:
            alive = [k for k in counter.cart if k not in self.gone]
            choice = alive[-1] if alive else None
        for _ in range(3):
            if choice is None or counter.over or counter.turns_left() < 1:
                break
            if self.unanswered and counter.has("message") and counter.turns_left() >= 2:
                self.reply_before_paying(choice)
            observations = counter.send([act("place_test_order", product_id=choice[0], sku=choice[1], justification=self.claims(choice))])
            if counter.over:
                break
            # the order was refused (for instance the shopper has a line to read first): take the line in and try again
            changed, asks = self.hear()
            if asks and self.pending_line and counter.turns_left() >= 2:
                self.answer(self.pending_line)
                self.pending_line = ""
                changed = self.hear()[0] or changed
            if changed and counter.turns_left() >= 3:
                # what the shopper wants changed while the order waited: the choice must still fit
                self.judge()
                if not self.fits(choice):
                    self.gone.add(choice)
                    choice = self.recart()
            if not observations:
                break
        return counter.dialogue


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    errand = Errand(problem_data)
    try:
        return errand.run()
    except Exception as exc:  # the dialogue so far is still returned for diagnosis, after a last order attempt
        errand.counter.dialogue.append({"role": "assistant", "content": f"agent error: {type(exc).__name__}: {exc}"})
        try:
            errand.last_resort()
        except Exception as late:
            errand.counter.dialogue.append({"role": "assistant", "content": f"agent error at the last order: {type(late).__name__}: {late}"})
        return errand.counter.dialogue


__all__ = ["agent_main"]
