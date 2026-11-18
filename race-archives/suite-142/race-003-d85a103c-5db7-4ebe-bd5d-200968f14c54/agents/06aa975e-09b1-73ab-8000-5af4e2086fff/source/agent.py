from __future__ import annotations

import json
import re
import time
from os import getenv
from typing import Any

from src.agent.proxy_client import ProxyClient

_MODELS = {
    "chutes": ("deepseek-ai/DeepSeek-V3.2-TEE", "Qwen/Qwen3-32B-TEE"),
    "openrouter": ("deepseek/deepseek-v3.2", "qwen/qwen3-32b"),
}
_PROXY = ProxyClient(timeout=110, max_retries=2, retry_delay=1.0)
_THINKER = ProxyClient(timeout=40, max_retries=1, retry_delay=1.0)
_TIME_BUDGET_S = 250.0
_OBS_CHARS = 6500
_EPS = 1e-9

_DIRECTED = re.compile(
    r"\b(highest|larger|largest|biggest|greatest|most|maximum|max|fastest|longest|"
    r"lowest|smallest|least|minimum|min|cheapest|shortest|fewest|better|best|more|less)\b", re.I)

_MOST_QUESTIONS = 3

_CLAIMABLE = ("screen_size", "resolution", "refresh_rate", "ram", "capacity", "wattage",
              "warranty_duration", "pack_quantity", "price", "in_stock", "brand")

_VARIANT_REF = re.compile(r"(\d{6,})::([A-Za-z0-9._\-]+)")

_READ_REQUEST = (
    "You turn a shopper's request into a checklist for a buying agent. Read the request and "
    "reply with one JSON object and nothing else:\n"
    '{"item": "<what kind of product, in the shopper\'s own words, or null>",'
    ' "brand": "<a maker they insist on, or null>", "preferred_make": "<a maker they would like but do not insist on, or null>",'
    ' "budget": <the most they will pay as a number, or null>,'
    ' "must_have": [{"need": "<the requirement in a few words>",'
    ' "number": <the figure it turns on, or null>, "unit": "<the word printed beside that figure,'
    ' such as inch or hz or gb, or null>",'
    ' "accepts": ["<a phrase a product page might print that would satisfy it>", "<another>"], "printed": <true only if the shopper demanded those very words appear in the listing>}],'
    ' "prefers": ["<each soft preference, one short phrase each>"],'
    ' "wants_reason": <true if they asked why the pick is right>,'
    ' "needs_asking": <true if they say a requirement will only be given if you ask>,'
    ' "survey_first": <true if they ask you to surface several distinct matching listings>,'
    ' "distinct_needed": <how many distinct matching listings they ask you to surface, or 0>}'
    "\nitem is only the kind of thing being bought, in the plainest word for it. "
    "Never put a size, a resolution, a refresh rate or any other figure in item: every "
    "measurement or specification the shopper states about the product is a firm requirement "
    "and belongs in must_have, with its figure in number and the word beside it in unit.\n"
    "Copy figures exactly. For accepts, list the ways a shop might print the same thing, "
    "including the plain figure and any common shorthand for it. Do not invent requirements "
    "they did not state. When the shopper says a listing has to match particular words, those "
    "words are a firm requirement: put them in must_have with the words themselves in accepts, "
    "and put the shelf or aisle they named in item. When they name a shelf and separately "
    "describe what they want, the shelf is the item and the description is a requirement."
)
_WEIGH_PRODUCTS = (
    "You decide which products satisfy a shopper and which is best, and you must show your working "
    "by quoting the shop's own words. You are given the checklist and a numbered list of products "
    "with the facts printed on their pages. For each product reply with exactly one line:\n"
    "<number> | YES or NO | <score 0-100> | <quote> ; <quote> ; ...\n"
    "Each quote must be copied character for character from that product's own lines, and together "
    "the quotes must prove every firm requirement. If a requirement is not printed on the page, the "
    "answer is NO with no quotes: do not infer it from the product name, and do not guess. Answer NO "
    "for an accessory, a spare part, a case, a bundle, or any item that is not the kind of thing the "
    "shopper asked for. The score says how closely it matches what the shopper named and how well "
    "it serves the use they described: full marks only for a product matching every thing they "
    "named, fewer marks for each detail that differs, weighted by the order of importance they "
    "gave. Where the shopper described what the thing is for, a page showing nothing that serves "
    "that use scores below 30 even when it is the right kind of thing and answers YES; being "
    "inexpensive is not a quality, and a shopper who says they would rather pay more for something "
    "that lasts is telling you the cheap one is the wrong answer. Reply with nothing but those lines."
)
_WRITE_CLAIMS = (
    "You justify a purchase to a shopper using only the product facts given. Reply with one JSON "
    'object and nothing else: {"claims": [{"field": "<one of screen_size, resolution, refresh_rate, '
    'brand>", "value": "<the value, copied word for word from the facts>"}]}. Use no other field '
    "name. Include the attribute the shopper asked about whenever the facts state it plainly, and "
    "the brand when the facts give one. Leave out anything you would have to work out or assume."
)
_READ_FIGURES = (
    "You name the units a shopper's priorities are measured in; you never choose between "
    "products and you never read a product. You are given the things a shopper cares about, "
    "in their order of importance. Reply with one JSON object and nothing else:\n"
    '{"measures": [{"unit": "<the word or mark a listing prints beside the figure for this, '
    'such as inch, hz, gb, ml, w, months or pieces; a screen size is always inch, never '
    'pixels; write pixels only for a display resolution; write price when it is what the '
    'thing costs; write null when it is not something printed as a figure at all>", '
    '"better": "high" or "low"}]}'
    " — one entry for each thing, in the same order. Nothing else."
)
_READ_DISPLAY = (
    "You read the display resolution off product listings. You are given numbered listings "
    "with the facts printed on their pages. Reply with one JSON object and nothing else:\n"
    '{"rows": [{"n": <listing number>, "pixels": "<width>x<height>"}]} - one row per listing. '
    "Copy the width and height where the page prints them. Where the page prints only a name "
    "for the resolution, give the pixel width and height that name denotes. Where the page "
    "says nothing about resolution, give null for pixels. Never take a figure from a model "
    "name or a photo or video resolution."
)
_PUT_IN_ORDER = (
    "You put products in the order a shopper would choose them. You are given what the shopper "
    "said and a numbered list of products. Apply their stated priorities in the order they gave "
    "them, one priority at a time: settle the first priority using the figures printed on each "
    "product, and only where products tie on it move to the next priority, and so on. Where the "
    "shopper named a particular thing, the one matching that name most closely comes first. "
    "Reply with the numbers alone, best first, separated by spaces, and nothing else."
)
_ANSWER_SHOPPER = (
    "You answer a shopper in one short sentence using only the facts given. Name the product and "
    "the figures. Never promise anything the facts do not show."
)
_SHOP_RULES = (
    "You are a shopping agent working a catalog through tools. One tool call per turn, at most "
    "30 turns. A controller checks every call and takes over the cart and the order once a "
    "product has been judged suitable and its price and stock verified, so your job is to find "
    "and open promising products: search or filter for candidates, then view or compare them so "
    "their facts are on record. Never add anything to the cart as a placeholder; the first cart "
    "addition is graded. Prices and stock change while you shop. Answer any direct question from "
    "the shopper in one short message and never send progress updates. Do not repeat a call with "
    "the same arguments."
)

def _num(value: Any) -> float | None:
    try:
        return None if value is None else float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None

def _json_block(text: str) -> dict:
    if not text:
        return {}
    start = text.find("{")
    while start != -1:
        depth = 0
        for spot in range(start, len(text)):
            if text[spot] == "{":
                depth += 1
            elif text[spot] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        loaded = json.loads(text[start : spot + 1])
                    except ValueError:
                        break
                    return loaded if isinstance(loaded, dict) else {}
        start = text.find("{", start + 1)
    return {}

def _arguments(call: dict) -> dict:
    raw = (call.get("function") or {}).get("arguments", "{}")
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw or "{}")
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}

def _measure_family(unit: str) -> str:
    w = re.sub(r"[^a-z]", "", (unit or "").casefold())
    if not w:
        return ""
    if w.startswith("inch") or w == "in" or w == "ins":
        return "inch"
    if w.startswith("hz") or "hertz" in w:
        return "hz"
    if w in ("gb", "tb") or w.startswith("gig") or w.startswith("tera") or w.startswith("gb"):
        return "gb"
    if w in ("ml", "l") or w.startswith("lit") or w.startswith("millil"):
        return "ml"
    if w in ("w", "kw") or w.startswith("watt") or w.startswith("kilowatt"):
        return "w"
    if w in ("mo", "mos", "yr", "yrs") or w.startswith("month") or w.startswith("year"):
        return "month"
    if w in ("pc", "pcs", "ct", "cts", "count", "unit", "units", "pack", "packs") or w.startswith("piece"):
        return "count"
    if "pixel" in w or "resolution" in w or w == "px":
        return "pixel"
    if w in ("price", "cost", "php", "peso", "pesos"):
        return "price"
    return "word"

_READERS = {
    "inch": (re.compile(r"(\d+(?:\.\d+)?)\s*[-\u2013]?\s*(?:inch(?:es)?\b|in\b|''|[\"\u201c\u201d])", re.I), {}),
    "hz": (re.compile(r"(\d+(?:\.\d+)?)\s*[-\u2013]?\s*(hz)\b", re.I), {}),
    "gb": (re.compile(r"(\d+(?:\.\d+)?)\s*(tb|gb)\b", re.I), {"tb": 1024.0}),
    "ml": (re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*[-\u2013]?\s*(ml|millilit(?:er|re)s?|l|lit(?:er|re)s?)(?!\w)", re.I),
           {"l": 1000.0, "liter": 1000.0, "liters": 1000.0, "litre": 1000.0, "litres": 1000.0}),
    "w": (re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*[-\u2013]?\s*(kw|kilowatts?|w|watts?)(?!\w)", re.I),
          {"kw": 1000.0, "kilowatt": 1000.0, "kilowatts": 1000.0}),
    "month": (re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*[-\u2013]?\s*(years?|yrs?|months?|mos?)(?!\w)", re.I),
              {"year": 12.0, "years": 12.0, "yr": 12.0, "yrs": 12.0}),
    "count": (re.compile(r"(?<![\w.])(\d+)\s*(?:-\s*)?(cts?|pcs?|pieces?|count|units?|pack(?!s))(?!\w)", re.I), {}),
}

def _measure_values(text: str, unit: str) -> set:
    fam = _measure_family(unit)
    text = _flatten(text)
    if fam == "pixel":
        return {float(int(m.group(1)) * int(m.group(2)))
                for m in re.finditer(r"(?<!\d)(\d{3,5})\s*[x\u00d7]\s*(\d{3,5})(?!\d)", text)}
    if fam in _READERS:
        pattern, scale = _READERS[fam]
        out = set()
        for m in pattern.finditer(text):
            figure = float(m.group(1))
            spoken = (m.group(2) or "").casefold() if m.re.groups >= 2 else ""
            out.add(round(figure * scale.get(spoken, 1.0), 3))
        return out
    word = re.escape((unit or "").strip().casefold())
    if not word:
        return set()
    pattern = re.compile(r"(\d+(?:\.\d+)?)\s*[-\u2013]?\s*(?:" + word + r"(?:e?s)?\b)", re.I)
    return {round(float(m.group(1)), 3) for m in pattern.finditer(text)}

def _memory_values(text: str) -> set:
    text = _flatten(text)
    pair = re.findall(r"(?<!\d)(\d+(?:\.\d+)?)\s*(?:gb?)?\s*\+\s*(\d+(?:\.\d+)?)\s*gb?\b", text, re.I)
    if pair:
        return {round(float(a), 3) for a, _b in pair}
    beside = re.findall(r"(\d+(?:\.\d+)?)\s*gb\s*(?:of\s+)?(?:ram|memory)\b", text, re.I)
    if beside:
        return {round(float(n), 3) for n in beside}
    return set()

def _about_memory(about: str) -> bool:
    return bool(re.search(r"\b(ram|memory)\b", _flatten(about)))

def _flatten(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[:;,_|]+", " ", (text or "").casefold())).strip()

def _trim(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 30] + ' ...(trimmed)"}'

class Clock:
    def __init__(self, budget: float) -> None:
        self.start = time.monotonic()
        self.budget = budget

    def left(self) -> float:
        return self.budget - (time.monotonic() - self.start)

class Checklist:

    def __init__(self) -> None:
        self.item: str | None = None
        self.preferred_make: str | None = None
        self.brand: str | None = None
        self.budget: float | None = None
        self.must_have: list = []
        self.prefers: list = []
        self.wants_reason = False
        self.needs_asking = False
        self.goal_text = ""
        self.survey_first = False
        self.distinct_needed = 0
        self.named: list = []
        self.rule_text = ""

    def absorb(self, said: dict) -> None:
        if not isinstance(said, dict):
            return
        text = lambda key: (str(said[key]).strip() if said.get(key) not in (None, "", "null") else None)
        self.item = text("item") or self.item
        self.preferred_make = text("preferred_make") or self.preferred_make
        make = text("brand")
        if make and self.goal_text and re.search(
                r"\b(prefer|lean(?:ing)? toward|ideally|would be my pick|if it fits|nice to have|"
                r"if the firm requirements)\b[^.]{0,80}\b" + re.escape(make) + r"\b|\b" + re.escape(make)
                + r"\b[^.]{0,80}\b(would be my pick|if it fits|if the firm requirements|but the firm)",
                self.goal_text, re.I):

            self.preferred_make = self.preferred_make or make
            make = None
        self.brand = make or self.brand

        for field in ("brand", "preferred_make"):
            value = getattr(self, field)
            if value and any(ch.isdigit() for ch in str(value)):
                phrase = str(value).strip()
                if phrase and phrase not in self.prefers:
                    self.prefers.insert(0, phrase)
                setattr(self, field, None)
        money = _num(said.get("budget"))
        if money is not None and money > 0:
            self.budget = money
        wanted = said.get("must_have")
        if isinstance(wanted, list):
            for item in wanted:
                if isinstance(item, dict):
                    need = str(item.get("need") or "").strip()
                    if not need or any(r["need"] == need for r in self.must_have):
                        continue
                    accepts = [str(a).strip() for a in (item.get("accepts") or []) if str(a).strip()]
                    figure = _num(item.get("number"))
                    if figure is not None and self.budget is not None and abs(figure - self.budget) < 0.01:
                        continue
                    self.must_have.append({"need": need, "number": figure,
                                           "unit": str(item.get("unit") or "").strip().lower() or None,
                                           "accepts": accepts,
                                           "printed": bool(item.get("printed")) and bool(accepts)})
                elif str(item).strip():
                    self.must_have.append({"need": str(item).strip(), "number": None,
                                           "unit": None, "accepts": []})
        liked = said.get("prefers")
        if isinstance(liked, list):
            for line in liked:
                line = str(line).strip()
                if line and line not in self.prefers:
                    self.prefers.append(line)
        self.wants_reason = bool(said.get("wants_reason")) or self.wants_reason
        self.needs_asking = bool(said.get("needs_asking")) or self.needs_asking
        self.survey_first = bool(said.get("survey_first")) or self.survey_first
        count = _num(said.get("distinct_needed"))
        if count:
            self.distinct_needed = max(self.distinct_needed, int(count))

    def tighten(self, money: float) -> None:
        if self.budget is None or money < self.budget:
            self.budget = money

    def as_prompt(self) -> str:
        lines = []
        if self.item:
            lines.append("wants: " + self.item)
        if self.brand:
            lines.append("brand required: " + self.brand)
        if self.preferred_make:
            lines.append("would like the maker: " + self.preferred_make)
        if self.budget is not None:
            lines.append("will pay at most: %.2f" % self.budget)
        for rule in self.must_have:
            shown = "must have: " + rule["need"]
            if rule.get("number") is not None and rule.get("unit"):
                shown += " (%g %s)" % (rule["number"], rule["unit"])
            lines.append(shown)
        for line in self.prefers:
            lines.append("prefers: " + line)
        return "\n".join(lines) or "no requirements recorded"

class Notebook:

    def __init__(self) -> None:
        self.pages: dict = {}
        self.rows: dict = {}
        self.state: dict = {}
        self.checked_on: dict = {}
        self.options: dict = {}
        self.listings: set = set()
        self.titles: dict = {}
        self.cart: list = []
        self.seq = 0
        self.adds = 0
        self.staked: str | None = None
        self.staked_at = 0
        self.written_off: set = set()
        self.spent_products: set = set()
        self.calls_made: set = set()

    @staticmethod
    def ref(pid: Any, sku: Any) -> str:
        return str(pid) + "::" + str(sku or "")

    def record(self, name: str, seen: dict) -> None:
        self.seq += 1
        if not isinstance(seen, dict) or seen.get("error"):
            return
        if name in ("search", "filter"):
            for row in seen.get("results") or []:
                if isinstance(row, dict) and row.get("product_id") and row.get("sku"):
                    ref = self.ref(row["product_id"], row["sku"])
                    self.rows[ref] = row
                    self.titles[str(row["product_id"])] = str(row.get("title") or "")
                    self.state.setdefault(ref, {"price": _num(row.get("price")),
                                                "in_stock": bool(row.get("in_stock", True))})
        elif name == "view":
            self._page(seen)
        elif name == "compare":
            for page in seen.get("compared") or []:
                self._page(page)
        elif name == "inspect_stock":
            ref = self.ref(seen.get("product_id"), seen.get("sku"))
            self.state[ref] = {"price": _num(seen.get("price")), "in_stock": bool(seen.get("in_stock"))}
            self.checked_on[ref] = self.seq
        elif name in ("add_to_cart", "remove_from_cart", "inspect_cart"):
            self._cart(seen)

    def _page(self, page: dict) -> None:
        if not isinstance(page, dict) or not page.get("product_id"):
            return
        pid = str(page["product_id"])
        self.pages[pid] = page
        self.titles[pid] = str(page.get("title") or "")
        if page.get("listing_id"):
            self.listings.add(str(page["listing_id"]))
        for variant in page.get("variants") or []:
            if isinstance(variant, dict) and variant.get("sku"):
                ref = self.ref(pid, variant["sku"])
                self.options[ref] = str(variant.get("options") or "")
                self.state[ref] = {"price": _num(variant.get("price")),
                                   "in_stock": bool(variant.get("in_stock"))}

    def _cart(self, seen: dict) -> None:
        rows = seen.get("cart")
        if isinstance(rows, list):
            self.cart = [(str(r.get("product_id")), str(r.get("sku") or "")) for r in rows
                         if isinstance(r, dict) and r.get("product_id")]
            for r in rows:
                if isinstance(r, dict) and r.get("product_id"):
                    ref = self.ref(r.get("product_id"), r.get("sku"))
                    price, live = _num(r.get("current_price")), r.get("current_in_stock")
                    held = self.state.setdefault(ref, {"price": None, "in_stock": None})
                    if price is not None:
                        held["price"] = price
                    if live is not None:
                        held["in_stock"] = bool(live)
        added = seen.get("added") or {}
        if added.get("product_id"):
            ref = self.ref(added["product_id"], added.get("sku"))
            self.adds += 1
            self.staked, self.staked_at = ref, self.seq
            pair = (str(added["product_id"]), str(added.get("sku") or ""))
            if pair not in self.cart:
                self.cart.append(pair)
        gone = seen.get("removed") or {}
        if gone.get("product_id"):
            pair = (str(gone["product_id"]), str(gone.get("sku") or ""))
            if pair in self.cart:
                self.cart.remove(pair)

    def facts_of(self, pid: str) -> dict:
        facts = (self.pages.get(str(pid)) or {}).get("facts")
        return facts if isinstance(facts, dict) else {}

    def describe(self, ref: str) -> str:
        pid, _, sku = ref.partition("::")
        page = self.pages.get(pid) or {}
        held = self.state.get(ref) or {}
        parts = ["title: " + str(page.get("title") or self.titles.get(pid, ""))[:220]]
        if page.get("brand"):
            parts.append("brand: " + str(page["brand"]))
        shelves = page.get("category_path")
        if isinstance(shelves, list) and shelves:
            parts.append("shelf: " + " / ".join(str(x) for x in shelves))
        if self.options.get(ref):
            parts.append("selected option: " + self.options[ref][:80])
        for key, value in list(self.facts_of(pid).items())[:10]:
            parts.append(str(key) + ": " + str(value)[:90])
        if held.get("price") is not None:
            parts.append("price: %.2f" % held["price"])
        parts.append("in stock: " + ("yes" if held.get("in_stock") else "no"))
        return "; ".join(parts)

class Agent:
    def __init__(self, problem: dict) -> None:
        world = problem["environment"]
        self.binding = dict(world["binding"])
        self.policy = world["policy_view"]
        self.tools = list(self.policy.get("tools") or [])
        self.tool_names = {(t.get("function") or {}).get("name") for t in self.tools if isinstance(t, dict)}
        self.max_steps = int(self.policy.get("max_steps") or 30)
        self.max_calls = max(1, int(self.policy.get("max_calls_per_turn") or 1))
        self.problem_id = str(problem.get("problem_id") or problem.get("id") or "problem")
        self.query = str(self.policy.get("query") or "")
        self.book = Notebook()
        self.clock = Clock(_TIME_BUDGET_S)
        self.list = Checklist()
        self.list.named = [(pid, sku.rstrip(".,;:")) for pid, sku in _VARIANT_REF.findall(self.query)]
        self.verdicts: dict = {}
        self.dialogue: list = []
        self.messages: list = [{"role": "system", "content": _SHOP_RULES},
                               {"role": "user", "content": self.query}]
        self.turn = 0
        self.done = False
        self.pending: dict | None = None
        self.question: str | None = None
        self.asked_opening = False
        self.recheck: tuple | None = None
        self.upset = False
        self.chosen: str | None = None
        self.nothing_else_to_do = False
        self.relaxed_once = False
        self.restaled: set = set()
        self.lost: list = []
        self._said_words = None
        self.orderings = 0
        self.ordered_field: tuple = ()
        self.searches = 0
        self.weighings = 0
        self.thinking = 0.0
        self.slow_model = False
        self.rejected_run = 0
        self.model_at = 0
        provider = (getenv("INFERENCE_PROVIDER") or "").strip().lower()
        self.roster = _MODELS.get(provider) or _MODELS["chutes"]
        pinned = getenv("SANDBOX_MODEL")
        if pinned:
            self.roster = (pinned,) + tuple(m for m in self.roster if m != pinned)

    def _think(self, brief: str, prompt: str, tools: bool = False) -> dict | None:
        body = {"model": self.roster[min(self.model_at, len(self.roster) - 1)],
                "messages": [{"role": "system", "content": brief}, {"role": "user", "content": prompt}],
                "temperature": 0}
        if tools:
            body["tools"] = self.tools
            body["tool_choice"] = "required"
        tries = 1 if self.slow_model else 2
        for attempt in range(tries):
            started = time.monotonic()
            reply = _THINKER.post("/inference/chat/completions", json_data=body)
            spent = time.monotonic() - started
            self.thinking += spent
            if spent > 25:
                self.slow_model = True
            if isinstance(reply, dict) and reply.get("choices"):
                return reply["choices"][0].get("message") or {}
            if self.clock.left() < 40:
                break
            self.model_at = min(self.model_at + 1, len(self.roster) - 1)
            body["model"] = self.roster[min(self.model_at, len(self.roster) - 1)]
        return None

    def _ask_model_for_action(self) -> dict | None:
        body = {"model": self.roster[min(self.model_at, len(self.roster) - 1)],
                "messages": self.messages, "tools": self.tools,
                "tool_choice": "required", "temperature": 0}
        for attempt in range(3):
            reply = _PROXY.post("/inference/chat/completions", json_data=body)
            if isinstance(reply, dict) and reply.get("choices"):
                said = reply["choices"][0].get("message") or {}
                calls = said.get("tool_calls") or []
                if calls:
                    return {"content": said.get("content") or "", "tool_calls": calls[: self.max_calls]}
                self.messages.append({"role": "assistant", "content": said.get("content") or ""})
                self._note("Choose exactly one of the supplied tools now.")
                continue
            if self.clock.left() < 30:
                break
            time.sleep(min(2.0 * (attempt + 1), 5.0))
            self.model_at = min(self.model_at + 1, len(self.roster) - 1)
        return None

    def read_request(self) -> None:
        goal, _, rules = self.query.partition("\n\nTask rules:")
        self.list.rule_text = rules.strip()
        self.list.goal_text = goal.strip()
        said = self._think(_READ_REQUEST, "Request:\n" + goal.strip()[:1200])
        self.list.absorb(_json_block(str((said or {}).get("content") or "")))
        if not (self.list.item or self.list.must_have or self.list.budget):

            self.list.item = " ".join(goal.split()[:12])
            amounts = [float(x.replace(",", "")) for x in re.findall(r"\d[\d,]*\.?\d*", goal)
                       if len(x.replace(",", "").split(".")[0]) >= 2]
            if amounts:
                self.list.budget = max(amounts)
            self._note("No checklist came back, so the request itself is the search.")
        for rule in self.stated_rules():
            if not any(r.get("number") is not None
                       and _measure_family(str(r.get("unit") or "")) == _measure_family(rule["unit"])
                       for r in self.list.must_have):
                self.list.must_have.append(rule)
                self._note("The request states %s, so that is a firm requirement." % rule["need"])
        if self.list.named:
            self._note("The request pins the choice to %d named variants." % len(self.list.named))
        self._note("Checklist read from the request: " + self.list.as_prompt().replace("\n", "; ")[:260])

    def weigh(self, refs: list) -> None:
        fresh = [r for r in refs if r not in self.verdicts][:8]
        cap = 1 if self.slow_model else (6 if self.clock.left() > 150 else 4)
        if not fresh or self.clock.left() < 60 or self.weighings >= cap:
            for ref in fresh:
                self.verdicts.setdefault(ref, {"ok": False, "score": 0.0, "why": "not judged"})
            return
        self.weighings += 1
        lines = ["%d. %s" % (spot, self.book.describe(ref)) for spot, ref in enumerate(fresh, start=1)]

        house = self.list.rule_text
        said = self._think(_WEIGH_PRODUCTS,
                           ("How this shop wants its listings read:\n" + house[:1300] + "\n\n" if house else "")
                           + "Checklist:\n" + self.list.as_prompt() + "\n\nProducts:\n" + "\n".join(lines))
        text = str((said or {}).get("content") or "")
        for line in text.splitlines():
            hit = re.match(r"\s*(\d+)\s*[|.:\-]\s*(YES|NO)\b[^|]*\|?\s*(\d+)?\s*\|?(.*)$", line.strip(), re.I)
            if not hit:
                continue
            spot = int(hit.group(1))
            if not 1 <= spot <= len(fresh):
                continue
            ref = fresh[spot - 1]
            says_yes = hit.group(2).upper() == "YES"
            try:
                score = max(0.0, min(100.0, float(hit.group(3) or 0))) / 100.0
            except ValueError:
                score = 0.0
            quotes = [q.strip().strip('"').strip() for q in (hit.group(4) or "").split(";")]
            quotes = [q for q in quotes if len(q) >= 4]
            if says_yes and quotes and not self.quotes_hold(ref, quotes):
                score *= 0.8
            self.verdicts[ref] = {"ok": says_yes, "score": score, "why": line.strip()[:130]}
        for ref in fresh:
            self.verdicts.setdefault(ref, {"ok": False, "score": 0.0, "why": "not judged"})

    def page_text(self, ref: str, variant_only: bool = False) -> str:
        pid = ref.split("::", 1)[0]
        pieces = [self.book.options.get(ref, "")]
        if not variant_only:
            page = self.book.pages.get(pid) or {}
            pieces.append(str(page.get("title") or ""))
            pieces.append(str(page.get("brand") or ""))
            held = self.book.state.get(ref) or {}
            if held.get("price") is not None:
                pieces.append("price: %.2f" % held["price"])
            pieces.append("in stock: " + ("yes" if held.get("in_stock") else "no"))
            for key, value in self.book.facts_of(pid).items():
                pieces.append(str(key) + ": " + str(value))
                pieces.append(str(key) + " " + str(value))
        return _flatten(" ".join(pieces))

    def figures_in(self, ref: str, unit: str, about: str = "") -> list:
        memory = _measure_family(unit) == "gb" and _about_memory(about)
        read = (lambda text: _memory_values(text)) if memory else (lambda text: _measure_values(text, unit))
        pid = ref.split("::", 1)[0]
        page = self.book.pages.get(pid) or {}
        out = [("option", read(self.book.options.get(ref, "")))]
        asked = set(re.findall(r"[a-z]{3,}", _flatten(about))) - {
            "the", "and", "for", "with", "most", "highest", "largest", "lowest", "least",
            "smallest", "biggest", "best", "longest", "shortest"}
        named, others = set(), set()
        for key, value in self.book.facts_of(pid).items():
            label = set(re.findall(r"[a-z]{3,}", str(key).casefold()))
            if memory and label & {"ram", "memory"}:
                said = _measure_values(str(value), "gb")
            else:
                said = read(str(key) + " " + str(value))
            if len(said) != 1:
                continue
            if label & asked:
                named |= said
            else:
                others |= said
        out.append(("facts", named if named else others))
        out.append(("title", read(str(page.get("title") or self.book.titles.get(pid, "")))))
        return out

    def meets(self, ref: str, rule: dict) -> bool | None:
        number, unit = rule.get("number"), rule.get("unit")
        if number is not None and unit:

            if _measure_family(unit) == "pixel":

                token = "%g" % number
                hay = self.page_text(ref)
                if re.search(r"(?<!\d)\d{3,5}\s*[x\u00d7]\s*" + token + r"(?!\d)|(?<!\d)" + token
                             + r"\s*[x\u00d7]\s*\d{3,5}(?!\d)|(?<!\d)" + token + r"p\b", hay):
                    return True
            else:
                for _source, found in self.figures_in(ref, unit, str(rule.get("need") or "")):
                    if len(found) == 1:

                        return abs(found.pop() - number) < 0.005
                    if len(found) > 1:
                        return False
        accepts = [a for a in (rule.get("accepts") or []) if len(a) >= 2]
        if accepts and any(re.sub(r"\s+", " ", a).casefold() in self.page_text(ref) for a in accepts):
            return True
        return None

    def right_make(self, ref: str) -> bool:
        wanted = _flatten(str(self.list.brand or ""))
        if not wanted:
            return True
        pid = ref.split("::", 1)[0]
        page = self.book.pages.get(pid) or {}
        said = _flatten(str(page.get("brand") or ""))
        if said:
            return wanted in said or said in wanted
        title = _flatten(str(page.get("title") or self.book.titles.get(pid, "")))
        return bool(title) and wanted in title

    def echo_score(self, ref: str) -> int:
        said = self._said_words
        if said is None:
            text = _flatten(self.list.goal_text or "")
            drop = set()
            for part in (self.list.item, self.list.brand):
                drop |= set(re.findall(r"[a-z]{4,}", _flatten(str(part or ""))))
            said = [w for w in dict.fromkeys(re.findall(r"[a-z]{4,}", text)) if w not in drop]
            self._said_words = said
        if not said:
            return 0

        page = re.sub(r"[^a-z0-9]+", " ", self.page_text(ref))
        return sum(1 for w in said if w in page or w.rstrip("s") in page)

    def mentions(self, ref: str) -> int:
        pid = ref.split("::", 1)[0]
        printed = _flatten(" ".join(str(v) for v in self.book.facts_of(pid).values()))
        printed += " " + _flatten(self.book.options.get(ref, ""))
        if not printed.strip():
            return 0
        seen = 0
        for rule in self.list.must_have:
            figure = rule.get("number")
            if figure is None:
                continue
            token = ("%g" % float(figure))
            if re.search(r"(?<!\d)" + re.escape(token) + r"(?!\d)", printed):
                seen += 1
        return seen

    def proves(self, ref: str) -> int:
        return sum(1 for rule in self.list.must_have if self.meets(ref, rule) is True)

    def requirements_met(self, ref: str, strict: bool = True) -> bool:
        if not self.right_make(ref):
            return False
        for rule in self.list.must_have:
            verdict = self.meets(ref, rule)
            if verdict is False:
                return False

            if verdict is None and rule.get("printed") and strict:

                return False
        return True

    def quotes_hold(self, ref: str, quotes: list) -> bool:
        if not quotes:
            return False
        pid = ref.split("::", 1)[0]
        page = self.book.pages.get(pid) or {}
        flat = self.page_text(ref)
        for quote in quotes:
            if _flatten(quote) not in flat:
                return False
        return True

    def affordable(self, ref: str) -> bool:
        held = self.book.state.get(ref) or {}
        price = held.get("price")
        if price is None or held.get("in_stock") is not True:
            return False
        return self.list.budget is None or price <= self.list.budget + _EPS

    def proven(self, ref: str) -> bool:
        seen_at = self.book.checked_on.get(ref, 0)
        if not seen_at:
            return False
        return not self.book.adds or seen_at > self.book.staked_at

    def buyable(self, ref: str) -> bool:
        if ref in self.book.written_off:
            return False
        pid = ref.split("::", 1)[0]
        if pid in self.book.spent_products:
            return False
        if pid not in self.book.pages:
            return False
        if not self.affordable(ref):
            return False
        if self.list.named and tuple(ref.split("::", 1)) not in [tuple(x) for x in self.list.named]:
            return False
        if not self.requirements_met(ref, strict=not self.nothing_else_to_do):
            return False
        if self.list.named:
            return True
        verdict = self.verdicts.get(ref)
        return bool(verdict and verdict["ok"])

    def figure_on(self, ref: str, unit: str, about: str = "") -> float | None:
        for _source, found in self.figures_in(ref, unit, about):
            if len(found) == 1:
                return found.pop()
            if len(found) > 1:
                return None
        return None

    def priorities(self) -> list:
        wanted = list(self.list.prefers)
        for rule in self.list.must_have:
            if rule.get("number") is None and not rule.get("printed"):
                need = " ".join(str(rule.get("need") or "").split())
                if need and need not in wanted:
                    wanted.append(need)

        return [w for w in wanted if _DIRECTED.search(w)]

    def display_of(self, refs: list) -> dict:
        if not refs or self.clock.left() < 50:
            return {}
        lines = ["%d. %s" % (spot, self.book.describe(ref)) for spot, ref in enumerate(refs[:10], start=1)]
        said = self._think(_READ_DISPLAY, "Listings:\n" + "\n".join(lines))
        rows = _json_block(str((said or {}).get("content") or "")).get("rows")
        out = {}
        for row in rows if isinstance(rows, list) else []:
            if not isinstance(row, dict):
                continue
            spot = _num(row.get("n"))
            pixels = str(row.get("pixels") or "")
            hit = re.search(r"(\d{3,5})\s*[x\u00d7]\s*(\d{3,5})", pixels)
            if spot is None or not hit or not 1 <= int(spot) <= len(refs[:10]):
                continue
            out[refs[int(spot) - 1]] = float(int(hit.group(1)) * int(hit.group(2)))
        return out

    def rank_by_figures(self, refs: list) -> list | None:
        wants = [" ".join(str(p).split()) for p in self.priorities()]
        wants = [w for w in wants if 2 <= len(w) <= 60][:4]
        if len(refs) < 2 or not wants or self.clock.left() < 60:
            return None
        said = self._think(_READ_FIGURES, "In order of importance:\n"
                           + "\n".join("- " + w for w in wants))
        measures = _json_block(str((said or {}).get("content") or "")).get("measures")
        if not isinstance(measures, list) or len(measures) != len(wants):
            return None
        reading = []
        for spot, measure in enumerate(measures):
            if not isinstance(measure, dict):
                return None
            unit = str(measure.get("unit") or "").strip().lower()
            bigger = str(measure.get("better") or "").strip().lower().startswith("h")
            if unit and unit not in ("null", "none"):
                reading.append((unit, bigger, wants[spot]))
        if not reading:
            return None
        field = refs[:10]
        figures = {}
        for ref in field:
            row = []
            for unit, _bigger, said in reading:
                fam = _measure_family(unit)
                if fam == "price":
                    row.append((self.book.state.get(ref) or {}).get("price"))
                else:
                    row.append(self.figure_on(ref, unit, said))
            figures[ref] = row
        for col, (unit, _bigger, _said) in enumerate(reading):
            if _measure_family(unit) != "pixel":
                continue
            unread = [ref for ref in field if figures[ref][col] is None]
            if unread:
                for ref, pixels in self.display_of(unread).items():
                    figures[ref][col] = pixels
        useful = [col for col in range(len(reading))
                  if any(figures[ref][col] is not None for ref in field)]
        if not useful:
            return None

        def place(ref: str) -> list:
            marks = []
            for col in useful:
                value = figures[ref][col]
                if value is None:
                    marks.append((1, 0.0))
                else:
                    marks.append((0, -value if reading[col][1] else value))
            return marks

        order = sorted(field, key=place)
        self._note("Put in order on " + ", ".join(reading[col][2] for col in useful) + ": "
                   + "; ".join("%s=%s" % (r.split("::")[0][-6:],
                                          [figures[r][c] for c in useful]) for r in order)[:240])
        return order + [r for r in refs if r not in field]

    def put_in_order(self, refs: list) -> list:
        pinned = bool(self.list.named)
        said_how = pinned or bool(self.list.prefers) or any(
            (rule.get("accepts") or []) and rule.get("number") is None for rule in self.list.must_have)
        if len(refs) < 2 or not said_how or self.ordered_field == tuple(refs[:10]):
            return refs
        if self.clock.left() < 70 or self.orderings >= 4:
            return refs
        if self.slow_model and not pinned:
            return refs
        self.orderings += 1
        self.ordered_field = tuple(refs[:10])
        by_figures = self.rank_by_figures(refs)
        if by_figures:
            return by_figures
        lines = ["%d. %s" % (spot, self.book.describe(ref)) for spot, ref in enumerate(refs[:10], start=1)]

        said = self._think(_PUT_IN_ORDER, "The shopper said:\n" + (self.list.goal_text or "")[:700]
                           + "\n" + self.list.as_prompt()
                           + "\n\nProducts:\n" + "\n".join(lines))
        numbers = re.findall(r"\d+", str((said or {}).get("content") or ""))
        seen, ranked = set(), []
        for token in numbers:
            spot = int(token)
            if 1 <= spot <= len(refs[:10]) and spot not in seen:
                seen.add(spot)
                ranked.append(refs[spot - 1])
        return ranked + [r for r in refs if r not in ranked] if ranked else refs

    def shortlist(self) -> list:
        picked = [r for r in self.book.state if self.buyable(r)]
        if not picked and self.nothing_else_to_do:

            loose = [r for r in self.book.state
                     if self.affordable(r) and r not in self.book.written_off
                     and r.split("::", 1)[0] in self.book.pages
                     and r.split("::", 1)[0] not in self.book.spent_products
                     and self.requirements_met(r, strict=False)
                     and (not self.list.named or tuple(r.split("::", 1)) in [tuple(x) for x in self.list.named])]
            judged_no = {r for r in loose if (self.verdicts.get(r) or {}).get("ok") is False}
            picked = [r for r in loose if r not in judged_no] or loose

        on_shelf = [r for r in picked if self.same_aisle(r)]
        if on_shelf and len(on_shelf) < len(picked):

            picked = on_shelf
        cheapest_wins = not self.priorities()

        picked.sort(key=lambda r: (-self.proves(r),
                                   -self.want_score(r),
                                   -self.named_clean(r),
                                   0 if self.like_lost(r) else 1,
                                   -self.mentions(r),
                                   -(self.verdicts.get(r) or {}).get("score", 0.0),
                                   -self.echo_score(r),
                                   ((self.book.state.get(r) or {}).get("price") or 1e12)
                                   if cheapest_wins else 0.0, r))
        return picked

    def want_groups(self) -> list:
        groups = []
        shades = self.shade_words()
        for phrase in self.sharp_wants():
            words = [w.casefold() for w in re.findall(r"[A-Za-z0-9]+", phrase) if len(w) >= 3]
            if words:

                groups.append(words + [s for s in shades if s not in words])
        said = self.list.goal_text or ""
        for word in re.findall(r"[A-Za-z][A-Za-z0-9]{2,}", str(self.list.preferred_make or "")):
            if [word.casefold()] not in groups:
                groups.append([word.casefold()])
        for phrase in self.list.prefers:
            for word in re.findall(r"[A-Za-z][A-Za-z0-9]{2,}", str(phrase)):
                spoken = re.search(r"(?<![.!?]\s)(?<!^)\b" + re.escape(word) + r"\b", said)
                already = any(word.casefold() in grp for grp in groups)
                if spoken and word[0].isupper() and not already:
                    groups.append([word.casefold()])
        return groups

    @staticmethod
    def _says(hay: str, word: str) -> bool:
        return (" " + word + " ") in (" " + hay + " ") or (
            any(c.isdigit() for c in word) and word in hay.replace(" ", ""))

    def same_aisle(self, ref: str) -> bool:
        page = self.book.pages.get(ref.split("::", 1)[0]) or {}
        shelves = page.get("category_path")
        if not isinstance(shelves, list) or not shelves:
            return True
        said = _flatten(str(self.list.item or ""))
        words = [w for w in re.findall(r"[a-z]{4,}", said)]
        if not words:
            return True
        where = _flatten(" ".join(str(x) for x in shelves))
        return any(w in where or w.rstrip("s") in where for w in words)

    def want_score(self, ref: str) -> int:
        groups = self.want_groups()
        if not groups or not self.same_aisle(ref):
            return 0
        hay = re.sub(r"[^a-z0-9]+", " ", self.page_text(ref))
        best = 0
        for words in groups:
            mark = sum((3 if any(c.isdigit() for c in w) else 1)
                       for w in words if self._says(hay, w))
            best = max(best, mark)
        return best

    def named_clean(self, ref: str) -> int:
        codes = set()
        for words in self.want_groups():
            codes |= {w for w in words if any(ch.isdigit() for ch in w) and len(w) >= 3}
        if not codes:
            return 0
        pid = ref.split("::", 1)[0]
        stated = ""
        for label, value in self.book.facts_of(pid).items():
            if _flatten(label) == "model":
                stated = _flatten(value)
                break
        if not stated or ";" in stated or "/" in stated:
            return 0
        squashed = re.sub(r"[^a-z0-9]+", "", stated)
        said = re.sub(r"[^a-z0-9]+", "", _flatten(
            str((self.book.pages.get(pid) or {}).get("title") or "")
            + " " + str((self.book.pages.get(pid) or {}).get("brand") or "")))
        for code in codes:
            if squashed == code:
                return 2
            if squashed.endswith(code):
                lead = squashed[: -len(code)]
                return 2 if lead and lead in said else 1
            if code in squashed:
                return 1
        return 0

    def carries_want(self, ref: str) -> bool:
        return self.want_score(ref) > 0

    def like_lost(self, ref: str) -> bool:
        if not self.lost:
            return False
        pid = ref.split("::", 1)[0]
        mine = _flatten(str((self.book.pages.get(pid) or {}).get("title")
                            or self.book.titles.get(pid, "")))
        if len(mine) < 12:
            return False
        for gone in self.lost:
            if mine == gone:
                return True
            head = " ".join(gone.split()[:6])
            if len(head) >= 12 and mine.startswith(head):
                return True
        return False

    def unread(self) -> list:
        waiting = []
        for ref, row in self.book.rows.items():
            pid = ref.split("::", 1)[0]
            if pid in self.book.pages or pid in self.book.spent_products:
                continue
            held = self.book.state.get(ref) or {}
            if held.get("in_stock") is False or held.get("price") is None:
                continue
            if self.list.budget is not None and held["price"] > self.list.budget + _EPS:
                continue
            title = str(row.get("title") or "").casefold()
            speaks = 1
            named = [{"accepts": [w for w in self.list.prefers]}] + list(self.list.must_have)
            for rule in named:
                for phrase in ([str(rule.get("number") or "")] if rule.get("number") is not None else [])\
                        + list(rule.get("accepts") or []):
                    phrase = _flatten(phrase)
                    if phrase and phrase in _flatten(title):
                        speaks = 0
                        break
                if speaks == 0:
                    break
            waiting.append((0 if self.like_lost(ref) else 1, speaks, held["price"], pid))
        waiting.sort()

        demanded = []
        for rule in self.list.must_have:
            if not rule.get("printed"):
                continue
            for phrase in (rule.get("accepts") or []):
                words = [w for w in re.findall(r"[a-z0-9]+", _flatten(phrase)) if len(w) > 1]
                if words:
                    demanded.append(words)
        if demanded:

            matching = []
            for row in waiting:
                said = set(re.findall(r"[a-z0-9]+", _flatten(self.book.titles.get(row[3], ""))))
                if any(all(w in said for w in words) for words in demanded):
                    matching.append(row)
            if matching:
                waiting = matching
        out = []
        for _twin, _speaks, _price, pid in waiting:
            if pid not in out:
                out.append(pid)
        return out

    def sharp_wants(self) -> list:
        figures = {("%g" % float(r["number"])) for r in self.list.must_have
                   if r.get("number") is not None}
        out = []
        for phrase in self.list.prefers:
            phrase = " ".join(str(phrase).split())
            if not (2 <= len(phrase) <= 40) or not any(c.isdigit() for c in phrase):
                continue
            spoken = set(re.findall(r"\d+(?:\.\d+)?", phrase))
            if spoken and spoken <= figures:
                continue
            out.append(phrase)
        return out

    def named_still_open(self) -> bool:
        if self.book.adds or self.nothing_else_to_do or self.want_in_hand():
            return False
        if self.clock.left() < 90 or len(self.book.pages) > (24 if self.max_calls > 1 else 14):
            return False
        codes = set()
        for words in self.want_groups():
            codes |= {w for w in words if any(ch.isdigit() for ch in w) and len(w) >= 3}
        if not codes:
            return False
        for ref, row in self.book.rows.items():
            pid = ref.split("::", 1)[0]
            if pid in self.book.pages or pid in self.book.spent_products:
                continue
            held = self.book.state.get(ref) or {}
            if held.get("in_stock") is False or held.get("price") is None:
                continue
            if self.list.budget is not None and held["price"] > self.list.budget + _EPS:
                continue
            hay = re.sub(r"[^a-z0-9]+", " ", _flatten(str(row.get("title") or "")))
            if any(code in hay.replace(" ", "") for code in codes):
                return True
        return False

    def want_in_hand(self) -> bool:
        wants = self.sharp_wants()
        if not wants:
            return True
        groups = self.want_groups()
        if not groups:
            return True
        for ref in self.book.state:
            if ref in self.book.written_off or not self.same_aisle(ref):
                continue
            if not self.affordable(ref) and ref.split("::", 1)[0] in self.book.pages:
                continue
            hay = re.sub(r"[^a-z0-9]+", " ", self.page_text(ref))
            for words in groups:
                if all(self._says(hay, w) for w in words):
                    return True
        return False

    def sharp_queries(self) -> list:
        out = []
        words = [w for w in [self.list.brand, self.list.item] if w]
        shades = self.shade_words()
        plain = []
        for phrase in self.sharp_wants()[:2]:
            term = " ".join((" ".join(words + shades + [phrase])).split())
            if term and term not in out:
                out.append(term)
            bare = [w for w in re.findall(r"[A-Za-z0-9]+", phrase) if any(c.isdigit() for c in w)]
            if bare:
                term = " ".join((" ".join(words + bare)).split())
                if term and term not in plain:
                    plain.append(term)
        return out + [t for t in plain if t not in out]

    def shade_words(self) -> list:
        out = []
        for phrase in self.list.prefers:
            if any(ch.isdigit() for ch in str(phrase)):
                continue
            for word in re.findall(r"[A-Za-z]{3,12}", str(phrase)):
                if word.casefold() not in out:
                    out.append(word.casefold())
        return out[:2]

    def next_lost_query(self) -> tuple | None:
        if not self.upset or not self.lost or "search" not in self.tool_names:
            return None
        if self.clock.left() < 70:
            return None
        for gone in self.lost[-2:]:
            words = [w for w in re.findall(r"[A-Za-z0-9+]+", gone) if len(w) > 1][:9]
            term = " ".join(words)
            if len(term) < 12:
                continue
            args = {"query": term, "in_stock": True, "k": 10}
            if "search" + json.dumps(args, sort_keys=True) not in self.book.calls_made:
                return ("search", args)
        return None

    def next_sharp_query(self) -> tuple | None:
        if self.clock.left() < 90 or "search" not in self.tool_names:
            return None
        if self.want_in_hand() and not self.upset:
            return None
        for term in self.sharp_queries():
            args = {"query": term, "in_stock": True, "k": 10}
            if "search" + json.dumps(args, sort_keys=True) not in self.book.calls_made:
                return ("search", args)
        return None

    def next_planned_query(self) -> tuple | None:
        if self.book.adds or self.searches >= _MOST_QUESTIONS - 1 or self.clock.left() < 110:
            return None
        if "search" not in self.tool_names:
            return None
        asked = 0
        for name, args in self.search_terms():
            if name != "search":
                continue
            asked += 1
            if asked > 2:
                break
            if name + json.dumps(args, sort_keys=True) not in self.book.calls_made:
                return (name, args)
        return None

    def sharp_pending(self) -> bool:
        return self.next_sharp_query() is not None

    def still_looking(self) -> bool:
        if self.unread():
            return True
        for name, args in self.search_terms():
            if name + json.dumps(args, sort_keys=True) not in self.book.calls_made:
                return True
        return False

    def ready_to_commit(self) -> bool:
        if self.book.adds or self.nothing_else_to_do:
            return True
        if self.named_still_open():

            return False
        if self.sharp_pending() or self.next_planned_query() is not None:

            return False
        if self.list.needs_asking and not self.asked_opening:
            return False
        if self.turn < 4 and self.clock.left() > 150 and (self.list.must_have or self.list.brand):
            return False
        if self.unread() and len(self.book.pages) < (24 if self.max_calls > 1 else 14)\
                and self.clock.left() > 110:

            return False
        if self.list.survey_first and len(self.book.rows) < max(3, self.list.distinct_needed):
            return self.turn >= self.max_steps * 0.6
        if len(self.book.listings) < 2 and self.list.prefers and not self.list.must_have:
            return False
        return True

    def _note(self, line: str) -> None:
        self.messages.append({"role": "user", "content": "[note] " + line})
        self.dialogue.append({"role": "assistant", "content": "[note] " + line[:280]})

    def reply_to(self, said: str) -> str:
        ref = self.book.staked if self.book.staked and self.book.staked not in self.book.written_off else None
        if ref is None:
            short = self.shortlist()
            ref = short[0] if short else None
        facts = self.book.describe(ref) if ref else "nothing verified yet"
        if self.upset:
            facts += "; the first item chosen is no longer valid and is being replaced"
        answer = self._think(_ANSWER_SHOPPER, "Shopper said: " + said[:280] + "\nFacts: " + facts)
        wording = str((answer or {}).get("content") or "").strip().replace("\n", " ")
        return wording[:400] if wording else ("Going with " + facts[:200])

    def claims_for(self, ref: str) -> list:
        pid = ref.split("::", 1)[0]
        facts = self.book.facts_of(pid)
        if not facts:
            return []
        if self.clock.left() < 45 or self.slow_model:
            return self.complete_claims([], ref)
        page = self.book.pages.get(pid) or {}
        held = self.book.state.get(ref) or {}
        block = "\n".join(str(k) + ": " + str(v)[:100] for k, v in facts.items())
        if page.get("brand"):
            block += "\nbrand: " + str(page["brand"])
        if held.get("price") is not None:
            block += "\nprice: %.2f" % held["price"]
        block += "\nin_stock: true"
        said = self._think(_WRITE_CLAIMS, "Shopper asked for: " + self.list.as_prompt()
                           + "\n\nFacts of the product being bought:\n" + block)
        offered = _json_block(str((said or {}).get("content") or "")).get("claims")
        kept = []
        spoken = {str(k).strip().lower(): str(v).strip().lower() for k, v in facts.items()}
        for claim in offered if isinstance(offered, list) else []:
            if not isinstance(claim, dict):
                continue
            field = str(claim.get("field") or "").strip()
            value = str(claim.get("value") or "").strip()
            if not field or not value:
                continue
            low_field, low_value = field.lower(), value.lower()
            if low_field == "price":
                continue
            if low_field == "in_stock":
                kept.append({"field": "in_stock", "value": "true"})
                continue
            if low_field == "brand" and page.get("brand") and low_value == str(page["brand"]).strip().lower():
                kept.append({"field": field, "value": value})
                continue

            asked_about = any(_flatten(rule["need"]).split()[0] in low_field.replace("_", " ")
                              or low_field.split("_")[0] in _flatten(rule["need"])
                              for rule in self.list.must_have if rule.get("need"))
            printed = _flatten(" ".join(str(v) for v in facts.values()))
            if asked_about and _flatten(value) in printed:
                kept.append({"field": low_field, "value": value})
        return self.complete_claims(kept, ref)

    def complete_claims(self, kept: list, ref: str) -> list:
        pid = ref.split("::", 1)[0]
        page = self.book.pages.get(pid) or {}
        held = self.book.state.get(ref) or {}
        allowed = set(self.claim_fields()) | {"price", "in_stock", "brand"}
        allowed |= {str(k) for k in self.book.facts_of(pid).keys()}
        kept = [c for c in kept if str(c.get("field") or "") in allowed]
        have = {str(c["field"]) for c in kept}

        for claim in self.stated_figures():
            if claim["field"] not in have:
                kept.append(claim)
                have.add(claim["field"])
        for rule in self.list.must_have:
            field = self.claim_field_for(rule)
            if field and field not in have:
                value = self.claim_value_for(rule)
                if value:
                    kept.append({"field": field, "value": value})
                    have.add(field)
                break
        if "in_stock" not in have:
            kept.append({"field": "in_stock", "value": "true"})
            have.add("in_stock")
        if "price" not in have:

            if self.list.budget is not None:
                kept.append({"field": "price", "value": "at most %.2f" % self.list.budget})
            elif held.get("price") is not None:
                kept.append({"field": "price", "value": "%.2f" % held["price"]})
            have.add("price")
        if not any(str(c["field"]) not in ("price", "in_stock", "screen_size",
                                          "resolution", "refresh_rate") for c in kept):

            soft = self.soft_claim(ref)
            if soft and soft["field"] not in have:
                kept.append(soft)
        return kept

    def claim_fields(self) -> list:
        for tool in self.tools:
            spec = (tool or {}).get("function") or {}
            if spec.get("name") != "place_test_order":
                continue
            field = (((spec.get("parameters") or {}).get("properties") or {})
                     .get("justification", {}).get("properties", {})
                     .get("claims", {}).get("items", {}).get("properties", {}).get("field") or {})
            allowed = field.get("enum")
            if isinstance(allowed, list) and allowed:
                return [str(a) for a in allowed]
            break
        return list(_CLAIMABLE)

    def soft_claim(self, ref: str) -> dict | None:
        pid = ref.split("::", 1)[0]
        page = self.book.pages.get(pid) or {}
        if page.get("brand"):
            return {"field": "brand", "value": str(page["brand"])}
        skip = {"display", "storage", "size", "refresh_rate", "ram", "capacity", "wattage",
                "warranty_duration", "pack_quantity", "phone_type", "device_type"}
        for label, value in self.book.facts_of(pid).items():
            label, value = str(label), str(value)
            if label in skip or not re.fullmatch(r"[a-z][a-z0-9_]*", label):
                continue
            if not value.strip() or len(value) > 80 or "[truncated]" in value or ";" in value:
                continue
            return {"field": label, "value": value}
        return None

    def stated_measures(self) -> list:
        said = " ".join((self.list.goal_text or "").split())
        out = []
        for fam, need in (("inch", "screen size"), ("hz", "refresh rate"), ("ml", "capacity"),
                          ("w", "wattage"), ("month", "warranty"), ("count", "pack quantity")):
            pattern, scale = _READERS[fam]
            hits = list(pattern.finditer(said))
            if fam == "month":
                hits = [h for h in hits if re.search(r"warrant", said[max(0, h.start() - 30):h.end() + 30], re.I)]
            found = {}
            for h in hits:
                figure = float(h.group(1)) * scale.get((h.group(2) or "").casefold() if h.re.groups >= 2 else "", 1.0)
                found[round(figure, 3)] = h.group(0).strip()
            if len(found) != 1:
                continue
            figure, phrase = next(iter(found.items()))
            if self.list.budget is not None and abs(figure - self.list.budget) < 0.01:
                continue
            unit = {"inch": "inch", "hz": "hz", "ml": "ml", "w": "w", "month": "months", "count": "pieces"}[fam]
            out.append({"need": need, "number": figure, "unit": unit, "accepts": [phrase], "printed": False})
        memory = {m.group(1) for m in re.finditer(r"(\d+(?:\.\d+)?)\s*gb(?:\s+of)?\s+(?:ram|memory)\b", said, re.I)}
        if len(memory) == 1:
            figure = _num(memory.pop())
            if figure is not None:
                out.append({"need": "RAM", "number": figure, "unit": "gb", "accepts": ["%g gb ram" % figure], "printed": False})
        shown = {}
        for m in re.finditer(r"(?<![\w.])(\d{3,4})p\b|(?<![\w.])(\d{3,5})\s*[x\u00d7]\s*(\d{3,5})(?!\d)", said, re.I):
            token = " ".join(m.group(0).split()).lower()
            figure = float(m.group(1)) if m.group(1) else float(min(int(m.group(2)), int(m.group(3))))
            shown[figure] = token
        if len(shown) == 1:
            figure, token = next(iter(shown.items()))
            out.append({"need": "display resolution", "number": figure, "unit": "pixels",
                        "accepts": [token], "printed": False})
        return out

    def stated_rules(self) -> list:
        return self.stated_measures()

    def stated_figures(self) -> list:
        out = []
        for rule in self.stated_measures():
            field = self.claim_field_for(rule)
            value = self.claim_value_for(rule)
            if field and value and field not in {c["field"] for c in out}:
                out.append({"field": field, "value": value})
        return out

    def claim_field_for(self, rule: dict) -> str | None:
        said = _flatten(str(rule.get("need") or "") + " " + str(rule.get("unit") or ""))
        for field in _CLAIMABLE:
            if field in ("price", "in_stock", "brand"):
                continue
            if field.replace("_", " ") in said:
                return field
        fam = _measure_family(str(rule.get("unit") or ""))
        if fam == "inch" or "inch" in said:
            return "screen_size"
        if fam == "hz" or " hz" in " " + said:
            return "refresh_rate"
        if fam == "pixel":
            return "resolution"
        if fam == "gb" and _about_memory(said):
            return "ram"
        return {"ml": "capacity", "w": "wattage", "month": "warranty_duration", "count": "pack_quantity"}.get(fam)

    def claim_value_for(self, rule: dict) -> str | None:
        fam = _measure_family(str(rule.get("unit") or ""))
        if fam == "pixel":
            spoken = [a for a in (rule.get("accepts") or []) if a]
            return str(spoken[0])[:20] if spoken else None
        if rule.get("number") is not None:
            if fam in ("ml", "w", "month", "count"):
                return "%g %s" % (float(rule["number"]), {"ml": "ml", "w": "w", "month": "months", "count": "pieces"}[fam])
            return "%g" % float(rule["number"])
        said = " ".join(str(rule.get("need") or "").split())
        return said[:40] if said else None

    def reason_was_asked(self) -> bool:
        if not self.list.wants_reason:
            return False
        asked = self.query.partition("\n\nTask rules:")[0]
        return bool(re.search(r"\b(why|reason|explain|justif)", asked, re.I))

    def order_args(self, ref: str) -> dict:
        pid, _, sku = ref.partition("::")
        args = {"product_id": pid, "sku": sku}
        if self.reason_was_asked():
            claims = self.claims_for(ref)
            if claims:
                args["justification"] = {"claims": claims}
        return args

    def _send(self, envelope: dict) -> dict | None:
        for attempt in range(4):
            reply = _PROXY.post("/environment/call", json_data=envelope)
            if isinstance(reply, dict) and reply.get("calls"):
                return reply
            if self.clock.left() < 20:
                break
            time.sleep(min(2.0 * (attempt + 1), 6.0))
        return None

    def _act(self, name: str, args: dict, spoken: str = "", call_id: str | None = None) -> dict | None:
        if self.pending is not None:
            kept = self.pending
            again = self._send(kept["envelope"])
            if again is None:
                return None
            self.pending = None
            self.turn = kept["turn"]
            self._absorb(kept["name"], kept["args"], kept["spoken"], kept["call_id"], again)
            if self.done:
                return None
        turn = self.turn + 1
        stamp = self.problem_id + "-turn-" + str(turn)
        envelope = {**self.binding, "call_id": stamp, "idempotency_key": stamp, "turn": turn,
                    "calls": [{"call_id": stamp + "-1", "action": {"name": name, "args": args}}]}
        reply = self._send(envelope)
        if reply is None:
            self.pending = {"envelope": envelope, "turn": turn, "name": name, "args": args,
                            "spoken": spoken, "call_id": call_id or ("agent-" + str(turn))}
            return None
        self.turn = turn
        return self._absorb(name, args, spoken, call_id or ("agent-" + str(turn)), reply)

    def _act_many(self, steps: list) -> list:
        steps = [(n, a) for n, a in steps if n in self.tool_names][:max(1, self.max_calls)]
        if not steps or self.pending is not None:
            return []
        turn = self.turn + 1
        stamp = self.problem_id + "-turn-" + str(turn)
        envelope = {**self.binding, "call_id": stamp, "idempotency_key": stamp, "turn": turn,
                    "calls": [{"call_id": "%s-%d" % (stamp, spot + 1),
                               "action": {"name": name, "args": args}}
                              for spot, (name, args) in enumerate(steps)]}
        reply = self._send(envelope)
        if not isinstance(reply, dict) or not reply.get("calls"):
            return []
        self.turn = turn
        answers = reply.get("calls") or []
        out = []
        for spot, (name, args) in enumerate(steps):
            answer = answers[spot] if spot < len(answers) else {}
            shell = (answer.get("observation") if isinstance(answer, dict) else None) or {}
            seen = shell.get("observation") if isinstance(shell.get("observation"), dict) else shell
            seen = seen if isinstance(seen, dict) else {}
            if shell.get("error") and not seen.get("error"):
                seen = dict(seen)
                seen["error"] = shell.get("error")
            if shell.get("done") or seen.get("done"):
                self.done = True
            self.book.record(name, seen)
            self.book.calls_made.add(name + json.dumps(args, sort_keys=True))
            call_id = "agent-%d-%d" % (turn, spot + 1)
            calls = [{"id": call_id, "type": "function",
                      "function": {"name": name, "arguments": json.dumps(args)}}]
            self.messages.append({"role": "assistant", "content": "", "tool_calls": calls})
            self.dialogue.append({"role": "assistant", "content": "", "tool_calls": calls,
                                  "environment_result": reply})
            self.messages.append({"role": "tool", "tool_call_id": call_id,
                                  "content": _trim(json.dumps(seen), _OBS_CHARS)})
            out.append(seen)
        spoke = reply.get("user_message")
        if isinstance(spoke, dict) and spoke.get("content"):
            self._hear(str(spoke["content"]))
        return out

    def _absorb(self, name: str, args: dict, spoken: str, call_id: str, reply: dict | None) -> dict | None:
        calls = [{"id": call_id, "type": "function",
                  "function": {"name": name, "arguments": json.dumps(args)}}]
        self.messages.append({"role": "assistant", "content": spoken, "tool_calls": calls})
        self.dialogue.append({"role": "assistant", "content": spoken, "tool_calls": calls,
                              "environment_result": reply})
        if not isinstance(reply, dict):
            self.book.calls_made.discard(name + json.dumps(args, sort_keys=True))
            self.messages.append({"role": "tool", "tool_call_id": call_id,
                                  "content": json.dumps({"error": "call failed; it will be retried"})})
            return None
        answers = reply.get("calls") or []
        shell = (answers[0].get("observation") if answers and isinstance(answers[0], dict) else None) or {}
        seen = shell.get("observation") if isinstance(shell.get("observation"), dict) else shell
        seen = seen if isinstance(seen, dict) else {}
        if shell.get("error") and not seen.get("error"):
            seen = dict(seen)
            seen["error"] = shell.get("error")
        if shell.get("done") or seen.get("done"):
            self.done = True
        self.book.record(name, seen)
        self.messages.append({"role": "tool", "tool_call_id": call_id,
                              "content": _trim(json.dumps(seen), _OBS_CHARS)})
        spoke = reply.get("user_message")
        if isinstance(spoke, dict) and spoke.get("content"):
            self._hear(str(spoke["content"]))
        return seen

    def _hear(self, said: str) -> None:
        self._note("shopper: " + said[:180])
        if not re.search(r"\d", said) and self.list.must_have and self.clock.left() < 120:
            if "?" in said:
                self.question = said
            return
        told = self._think(_READ_REQUEST, "Earlier checklist:\n" + self.list.as_prompt()
                           + "\n\nThe shopper has just said:\n" + said[:400]
                           + "\n\nGive the updated checklist.")
        fresh = _json_block(str((told or {}).get("content") or ""))
        before = self.list.budget
        rules_before = len(self.list.must_have)
        self.list.absorb(fresh)
        if len(self.list.must_have) > rules_before:

            self._note("A new firm requirement arrived: " + self.list.must_have[-1]["need"][:90])
            self.verdicts = {}
            self.chosen = None
            self.ordered_field = ()

            self.searches = 0
        money = _num(fresh.get("budget"))
        if money is not None and money > 0:
            self.list.tighten(money)
        if self.list.budget is not None and (before is None or self.list.budget < before):
            self._note("The shopper will now pay at most %.2f; earlier choices may no longer qualify."
                       % self.list.budget)
            for pid, sku in list(self.book.cart):
                ref = Notebook.ref(pid, sku)
                price = (self.book.state.get(ref) or {}).get("price")
                if price is not None and price > self.list.budget + _EPS:
                    self.book.written_off.add(ref)
            self.verdicts = {r: v for r, v in self.verdicts.items() if self.affordable(r)}
        if "?" in said or re.search(r"\b(let me know|which one|should i|tell me)\b", said, re.I):
            self.question = said

    def after_commit(self) -> None:
        pid, sku = self.recheck
        self.recheck = None
        ref = Notebook.ref(pid, sku)
        seen = self._act("inspect_stock", {"product_id": pid, "sku": sku})
        if not isinstance(seen, dict) or seen.get("error"):
            return
        if self.affordable(ref):
            self._note("Still good after the cart addition: " + self.book.describe(ref)[:160])
            return
        self.upset = True
        self.chosen = None
        gone = _flatten(str((self.book.pages.get(pid) or {}).get("title")
                            or self.book.titles.get(pid, "")))
        if gone and gone not in self.lost:
            self.lost.append(gone)
        self.ordered_field = ()
        self.book.written_off.add(ref)
        self.book.spent_products.add(pid)
        self._note("The shop changed this item after it was carted, so a different product is needed now.")

    def plan(self) -> tuple | None:
        if self.done:
            return None
        if self.question and "message" in self.tool_names:
            said, self.question = self.question, None
            return ("message", {"content": self.reply_to(said)})
        if not self.asked_opening and self.list.needs_asking and "message" in self.tool_names:
            self.asked_opening = True
            return ("message", {"content": "Before I look: what exactly is the requirement you said "
                                           "you would tell me if I asked?"})
        if self.list.named and "compare" in self.tool_names:
            missing = [pid for pid, _sku in self.list.named if pid not in self.book.pages]
            if missing:
                return ("compare", {"product_ids": missing[:5]})
        chase = self.next_lost_query() or self.next_sharp_query() or self.next_planned_query()
        if chase is not None and (not self.book.staked or self.upset):
            self.book.calls_made.add(chase[0] + json.dumps(chase[1], sort_keys=True))
            return chase
        staked = self.book.staked
        if staked and staked not in self.book.written_off:
            pid, _, sku = staked.partition("::")
            if not self.proven(staked):
                return ("inspect_stock", {"product_id": pid, "sku": sku})
            if (pid, sku) in self.book.cart and self.buyable(staked):
                return ("place_test_order", self.order_args(staked))
        for pid, sku in self.book.cart:
            ref = Notebook.ref(pid, sku)
            if ref in self.book.written_off and "remove_from_cart" in self.tool_names:
                return ("remove_from_cart", {"product_id": pid, "sku": sku})
        if self.upset and self.clock.left() > 60 and self.turn < self.max_steps - 4:

            if self.list.named:
                worth = {Notebook.ref(pid, sku) for pid, sku in self.list.named}
            else:
                worth = {r for r in self.book.state
                         if r.split("::", 1)[0] in self.book.pages and self.carries_want(r)}
                if not worth:
                    worth = set(self.shortlist()[:4])
                worth |= {Notebook.ref(pid, sku) for pid, sku in self.book.cart}
            stale = [r for r in self.book.state
                     if r in worth
                     and r not in self.book.written_off
                     and (r, self.book.staked_at) not in self.restaled
                     and r.split("::", 1)[0] in self.book.pages
                     and r.split("::", 1)[0] not in self.book.spent_products
                     and self.book.checked_on.get(r, 0) <= self.book.staked_at]
            if stale:
                self.restaled.add((stale[0], self.book.staked_at))
                pid, _, sku = stale[0].partition("::")
                return ("inspect_stock", {"product_id": pid, "sku": sku})
        judged = self.shortlist()
        if judged:

            if self.chosen and self.buyable(self.chosen) and self.chosen in judged:
                best = self.chosen
            else:
                best = self.put_in_order(judged)[0]
                if best != self.chosen:
                    self.chosen = best
                    self._note("Decided on " + self.book.describe(best)[:140])
            pid, _, sku = best.partition("::")
            if not self.proven(best):
                return ("inspect_stock", {"product_id": pid, "sku": sku})
            if self.ready_to_commit() and (pid, sku) not in self.book.cart:
                return ("add_to_cart", {"product_id": pid, "sku": sku})
        return None

    def judge_new_pages(self) -> None:
        waiting = [r for r in self.book.state
                   if r not in self.verdicts and r.split("::", 1)[0] in self.book.pages
                   and self.affordable(r) and r not in self.book.written_off]
        if waiting:

            waiting.sort(key=lambda r: (-self.want_score(r),
                                        (self.book.state.get(r) or {}).get("price") or 1e12))
            self.weigh(waiting)

    def close_out(self) -> None:
        for pid, sku in reversed(self.book.cart):
            ref = Notebook.ref(pid, sku)
            if self.done or ref in self.book.written_off:
                continue
            if not self.proven(ref):

                self._act("inspect_stock", {"product_id": pid, "sku": sku})
            if self.affordable(ref) and self.proven(ref) and not self.done:
                self._act("place_test_order", self.order_args(ref))
                return
        if self.done or self.turn > self.max_steps - 3:
            return
        judged = self.shortlist() or [
            r for r in self.book.state
            if self.affordable(r) and r not in self.book.written_off
            and r.split("::", 1)[0] in self.book.pages
            and self.requirements_met(r, strict=False)
            and (not self.list.named or tuple(r.split("::", 1)) in [tuple(x) for x in self.list.named])]
        if not judged:
            return
        ref = judged[0]
        pid, _, sku = ref.partition("::")
        if (pid, sku) not in self.book.cart:
            seen = self._act("add_to_cart", {"product_id": pid, "sku": sku})
            if self.done or not isinstance(seen, dict) or seen.get("error"):
                return
        self._act("inspect_stock", {"product_id": pid, "sku": sku})
        if not self.done and self.affordable(ref) and self.proven(ref):
            self._act("place_test_order", self.order_args(ref))

    def open_the_shelf(self) -> None:
        if self.max_calls < 2 or self.list.named or self.clock.left() < 120:
            return
        rounds, searches = [], 0
        for name, args in self.search_terms():
            if name == "search":
                if searches >= _MOST_QUESTIONS:
                    continue
                searches += 1
            mark = name + json.dumps(args, sort_keys=True)
            if mark in self.book.calls_made:
                continue
            rounds.append((name, args))
            if len(rounds) >= min(self.max_calls, 5):
                break
        if len(rounds) > 1 and self._act_many(rounds):
            self.searches += searches
            self._note("Asked the shop %d questions in one round." % len(rounds))

    def run(self) -> list:
        self.read_request()
        self.open_the_shelf()
        while not self.done and self.turn < self.max_steps:
            if self.clock.left() < 40 or self.turn >= self.max_steps - 3:
                self.close_out()
                break
            if self.recheck and "inspect_stock" in self.tool_names:
                self.after_commit()
                continue
            self.judge_new_pages()
            step = self.plan()
            if step is not None:
                name, args = step
                seen = self._act(name, args)
                if name in ("search", "filter"):
                    self.searches += 1
                if name == "add_to_cart" and isinstance(seen, dict) and not seen.get("error"):
                    added = seen.get("added") or {}
                    if added.get("product_id"):
                        self.recheck = (str(added["product_id"]), str(added.get("sku") or ""))
                continue
            unread = self.unread()
            if unread and "compare" in self.tool_names and len(unread) > 1:
                lots = [unread[at:at + 5] for at in range(0, min(len(unread), 15), 5)]
                rounds = [("compare", {"product_ids": lot}) for lot in lots if len(lot) > 1]
                if len(rounds) > 1 and self.max_calls > 1 and self._act_many(rounds):
                    continue
                self._act("compare", {"product_ids": unread[:5]})
                continue
            if unread and "view" in self.tool_names:
                self._act("view", {"product_id": unread[0]})
                continue
            asked = self.ask_shop()
            if asked is None:
                if self.shortlist() and not self.nothing_else_to_do:

                    self.nothing_else_to_do = True
                    self._note("Nothing further to look at; going with the best candidate found.")
                    continue
                if not self.relaxed_once:

                    self.relaxed_once = True
                    self.nothing_else_to_do = True
                    self._note("No candidate proved every requirement; weighing what was found.")
                    continue
                self.close_out()
                break
            name, args = asked
            self._act(name, args)
            if name in ("search", "filter"):
                self.searches += 1
        return self.dialogue

    def ask_shop(self) -> tuple | None:
        wanted = self.search_terms()
        if self.searches >= _MOST_QUESTIONS and self.shortlist():
            wanted = [(n, a) for n, a in wanted if n != "search"]
        for name, args in wanted:
            mark = name + json.dumps(args, sort_keys=True)
            if mark in self.book.calls_made:
                continue
            self.book.calls_made.add(mark)
            return (name, args)
        return None

    def search_terms(self) -> list:
        if self.list.named:
            return []
        out = []
        words = [w for w in [self.list.brand, self.list.item] if w]

        needs = []
        for rule in self.list.must_have:
            if rule.get("number") is not None and rule.get("unit"):
                needs.append("%g %s" % (rule["number"], rule["unit"]))
            for phrase in (rule.get("accepts") or []):
                phrase = " ".join(str(phrase).split())
                if 2 <= len(phrase) <= 40 and any(c.isdigit() for c in phrase):
                    needs.append(phrase)
        for phrase in self.list.prefers:

            phrase = " ".join(str(phrase).split())
            if 2 <= len(phrase) <= 40:
                needs.append(phrase)
        for rule in self.list.must_have:
            phrase = " ".join(str(rule.get("need") or "").split())
            if 2 <= len(phrase) <= 40 and phrase not in needs:
                needs.append(phrase)
        seen, ordered = set(), []
        for phrase in needs:
            key = phrase.casefold()
            if key not in seen:
                seen.add(key)
                ordered.append(phrase)
        needs = ordered

        def put(term: str) -> str:
            kept, said = [], set()
            for word in term.split():
                key = re.sub(r"[^a-z0-9]+", "", word.casefold())
                if key and key not in said:
                    said.add(key)
                    kept.append(word)
            return " ".join(kept)
        if "search" in self.tool_names:

            for term in self.sharp_queries():
                if ("search", {"query": term, "in_stock": True, "k": 10}) not in out:
                    out.append(("search", {"query": term, "in_stock": True, "k": 10}))
        if "filter" in self.tool_names and self.list.item:
            base = {"category": self.list.item, "in_stock": True, "k": 10}
            if self.list.budget is not None:
                base["max_price"] = self.list.budget
            soft_make = self.list.preferred_make if self.list.preferred_make and not any(
                ch.isdigit() for ch in str(self.list.preferred_make)) else None
            if self.list.brand or soft_make:
                out.append(("filter", dict(base, brand=self.list.brand or soft_make)))
            out.append(("filter", base))
        if "search" in self.tool_names:
            phrases = [" ".join(words + [n]) for n in needs[:3]]
            phrases += [" ".join(words), " ".join(needs[:1]), self.list.item or ""]
            asked = []
            for extra in phrases:
                phrase = put(" ".join(extra.split()))
                if not phrase:
                    continue

                token = frozenset(re.findall(r"[a-z0-9]+", phrase.casefold()))
                if any(token <= before or before <= token for before in asked):
                    continue
                asked.append(token)
                out.append(("search", {"query": phrase, "in_stock": True, "k": 10}))
        return out

    def refuse(self, name: str, args: dict) -> str | None:
        if name not in self.tool_names:
            return "that tool does not exist here"
        if name in ("add_to_cart", "place_test_order"):
            ref = Notebook.ref(args.get("product_id"), args.get("sku"))
            if not args.get("product_id") or not args.get("sku"):
                return "name the exact product_id and sku"
            if ref in self.book.written_off:
                return "that variant is no longer valid; choose another product"
            if not self.affordable(ref):
                return "that variant is out of stock, over budget, or its price is unknown"
            if self.list.named and tuple(ref.split("::", 1)) not in [tuple(x) for x in self.list.named]:
                return "the shopper pinned the choice to the variants they listed; that is not one of them"
            if ref.split("::", 1)[0] not in self.book.pages:
                return "open that product with view or compare first"
            verdict = self.verdicts.get(ref)
            if verdict is not None and not verdict["ok"]:
                return "that product was judged not to meet the requirements: " + verdict["why"]
            if name == "place_test_order":
                if (args.get("product_id"), args.get("sku")) not in self.book.cart:
                    return "add it to the cart first"
                if not self.proven(ref):
                    return "check its live price and stock again before ordering"
            if name == "add_to_cart" and not self.ready_to_commit():
                return "wait: the shopper may still change what they are asking for"
        if name == "message":
            text = str(args.get("content") or "")
            if not text.strip():
                return "a message needs something in it"
            if not self.question and "?" not in text:
                return "do not send progress updates; use message to answer or ask a question"
        mark = name + json.dumps(args, sort_keys=True)
        if mark in self.book.calls_made and name in ("search", "filter", "view", "compare"):
            return "that exact call was already made; change the arguments"
        self.book.calls_made.add(mark)
        return None

def agent_main(problem_data: dict) -> list:
    try:
        return Agent(problem_data).run()
    except Exception as trouble:
        return [{"role": "assistant", "content": "agent error: " + type(trouble).__name__ + ": " + str(trouble)[:200]}]

__all__ = ["agent_main"]
