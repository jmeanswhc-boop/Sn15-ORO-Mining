"""Hotaru: a pipeline shopper for ORO Bench worlds.

Code chooses every store action, phase by phase. Models only return structured
judgements. There is no free-roaming tool-calling loop.

    read -> tools -> scout -> sweep -> chase -> select -> align -> confirm
         -> commit (repair rounds after a refused checkout)

The shopper holds a few subjects: properties they truly want, and properties they
are indifferent about. A message is answered about the FIRST of those subjects that
it names, however many other properties it also names, and an indifferent subject
named first hides a real one named later. The wants have to be drawn out before
ordering, so the interview comes first:
- every message names every property still open, across every product;
- after a reply without a want the order is turned around (reverse, then
  centre-out), so that a different subject comes first;
- after a want is drawn out, it leaves the list and its product goes last.
The pipeline orders its best cart unless a product has no listing at all.
"""
from __future__ import annotations

import json
import re
import time
from typing import Any

from src.agent.proxy_client import ProxyClient

FAST = "anthropic/claude-haiku-5.5"
SHARP = "anthropic/claude-sonnet-5.5"
BACKUP = "openai/gpt-6-luna"

EPISODE_SECONDS = 300.0
COMMIT_BY = 228.0          # after this the pipeline goes straight to the checkout
LAST_CALL = 268.0          # after this no model call starts
SPEND_LIMIT = 0.05         # USD per episode; past it only the fast model is used
NOTE_WORDS = 6             # a loose string of at least this many words in a result is read as a note

_models = ProxyClient(timeout=45, max_retries=1, retry_delay=1)
_world = ProxyClient(timeout=45, max_retries=2, retry_delay=1)


# ============================================================================ prompts
READ = """Turn a shopper's request into a JSON work order. Output only the JSON object, with exactly these keys:
"products": one entry per distinct thing to buy: {"name": short noun phrase from the request, "qty": integer (1 when unstated), "fits_with": number (from 1) of another product in this list that this one must fit, suit or work with, else null, "needs": ["<property>: <required value or limit>" for every firm requirement the request states on that thing], "points": ["<property>: <the request's own words, from those saying how the value relates to the other listing (same as, other than, goes with, close to, at least, at most) to the end of that listing's description>" for requirements whose value has to be read off another listing]}
"budget": the most the shopper will pay, as the number written in the request with no unit conversion ("390 cts" gives 390, "12 dollars" gives 12), or null; "budget_scope": "total" or "each"
"pick": "cheapest", "most", "least" or "any"; "measure": the quantity to maximise or minimise, else ""
"messages": the cap on messages to the shopper as an integer, or null; "steps": the cap on store actions as an integer, or null
"in_stock": true unless the request accepts items that are not available now
"before_order": [each explicit procedural instruction to contact the shopper right before placing the order, such as a request to approve the total or to hear the pick before checkout], [] when none. Remarks inviting the agent to ask questions or check details are not before_order steps
"ask_first": [properties the request explicitly says to ask the shopper about], [] when none
"hints": [{"product": number from 1, "property": a property the story implies matters to the shopper without giving its value}], [] when none. Look for sentences about what the shopper leaves to you or cannot judge, whose taste decides, how it looks or the setting it belongs in, who will use or wear it, or what is inside it
"quiet": true when the request says not to offer substitutes or alternatives if nothing fits
Copy each value word for word as the request writes it, in one form only, with no brackets and no change of singular or plural ("pleats" stays "pleats", never "pleat"; "no sleeves" stays "no sleeves"). Words that only name the store category the product belongs to are part of its name, not a need; a word narrowing the kind of product is a need ("a linen tablecloth" gives name "tablecloth" and need "material: linen").
Every property value the request states is a need, even when it is written next to the product ("a lamp, brass finish" gives name "lamp" and need "finish: brass"). A use the product must be made for counts as a need ("boots for hiking" gives "use: hiking"); the person or room it is for does not. Keep price, stock, quantity and the rest of the story's background out of "needs"."""

TOOLS = """These are the tools of an online store. Say which tool does each job and which of its arguments to use. Output only a JSON object:
{"search": {"tool": name, "query_arg": argument for free text or keywords or null, "category_arg": argument filtering by category/department or null, "price_cap_arg": argument for a maximum price or null, "page_arg": argument for page number/offset or null, "token_arg": argument taking a continuation token or null, "mode_arg": argument choosing the search mode or null, "keyword_mode": value of mode_arg for keyword search or null, "filter_mode": value of mode_arg for category/field filtering or null},
 "browse": {"tool": a SEPARATE tool that lists by category, brand or price cap, only when the search tool above cannot, "category_arg": ... or null, "price_cap_arg": ... or null, "page_arg": ... or null, "token_arg": ... or null} or null,
 "attributes": {"tool": the tool returning one listing's properties/specifications, "id_arg": its listing-id argument},
 "compare": {"tool": a tool that shows several listings side by side in one call, "ids_arg": its list-of-ids argument} or null,
 "page": {"tool": the tool returning one listing's full page (description, stock, price), "id_arg": ...} or null,
 "message": {"tool": the tool sending a message to the shopper, "text_arg": ...} or null,
 "cart": {"tool": the tool changing the cart, "id_arg": ..., "qty_arg": ... or null, "op_arg": argument choosing add/remove or null, "add": value of op_arg that adds or null, "remove": value of op_arg that removes or null, "lines_arg": the array argument when the tool sets the whole cart at once or null, "line_id": the id key inside one array entry or null, "line_qty": the count key inside one array entry or null},
 "cart_view": {"tool": ...} or null,
 "buy": {"tool": the tool placing the order, "args": {fixed arguments that make it place the order}},
 "decline": {"tool": the tool ending without a purchase, "args": {fixed arguments that make it end without buying}, "text_arg": its free-text argument or null}}
Use exact tool and argument names from the definitions."""

VOCAB = """A shopper wants the products below and keeps a few unstated wants, each tied to one property of one product. You list the properties a question could name. For each product give every property a buyer of that kind of product could care about, most likely to matter first: what the request's story hints at, then what decides the variant (type, form, material, size or capacity, fit, interface, clean numeric specs).
Use the keys seen on the store's listings below, merged when two keys mean the same thing, and add any other property a buyer of that kind of product usually chooses between that the keys miss. Leave out what the request already settles for that product, identifiers, seller, brand, price, stock, shipping and warranty.
Name each property in two or three plain words. Output only JSON: {"lists": [{"product": number, "properties": ["...", ...]}]} with up to 22 properties per product."""

HEAR = """A shopping assistant asked the shopper: QUESTION
The shopper replied: ANSWER
Products: PRODUCTS
Decide whether the reply states a firm want. Output only JSON:
{"want": true or false, "property": the property the reply itself names, "value": the wanted value or limit alone, exactly as a listing could print it, without the words framing it ("brass" from "matching brass, no arguing" or "settled on brass and nothing else"; "at least 40 cm" from "equal to at least 40 cm"), "product": number of the product it is about (the reply may name it; otherwise the product whose property it is), "points": true when the value is defined through some other listing, "split": true when the reply only asks for fewer things or one thing at a time}
Reading rules. A reply can state a want in many colourful ways. "value" holds the wanted value alone: a refusal of everything except one value states that value, so the value is what is left once the refusal is taken away. Words setting a limit or a range stay in the value. Comparisons with anything not sold in the store (a family story, a habit, a saying) are only colour: leave them out and set "points" to false. "points" is true only when the reply names another listing, page, shop or seller in the store to read the value from. Replies showing puzzlement or indifference state no want, whatever their wording.
When "points" is true, "value" must copy, word for word, everything from the words saying how the value relates to the other listing (same, different, belonging with it, close to it, above or below it) through the description of that listing to the end of the description, leaving out any refusal before those words.
Invented examples:
"Any width other than 30 cm is a no from me" gives value "30 cm", points false.
"I'm firm on a frame matching steel, the way my father always did it" gives value "steel", points false.
"Any finish but the one on the lamp on the Acme page, tall model, is a no from me" gives value "the one on the lamp on the Acme page, tall model", points true."""

FIELDS = """Below is one listing record from an online store's tool output. Say which key holds each thing. Output only JSON:
{"title": key of the product's name, "seller": key of the shop or seller offering it, "brand": key of the product's brand or maker, "category": key of the store section or category, "price": key of the price, "stock": key saying whether it can be bought}
Use null when no key holds that thing. A key holding free text about the seller is not the seller.
RECORD:
"""

NEAR = """A shopper just stated a want on the property PROPERTY. From the names below, list only those that are other names for that same property (a reader would take a question about them as a question about PROPERTY). Different properties that merely share a word do not count.
NAMES
Output only JSON: {"near": ["...", ...]}"""

NOTE = """While shopping for a customer, these messages appeared in the store's replies:
NOTES
Products: PRODUCTS
Current budget: BUDGET
Output only JSON: {"budget": new total budget, the number exactly as the message writes it with no unit conversion ("3000 cents" gives 3000), or null, "drop": [{"product": number, "property": "a requirement the shopper no longer needs: one taken off the list, scrapped, binned or to carry on without"}], "add": [{"product": number, "need": "<property>: <value>"}], "sold_out": true when any message says the shopper's pick or the chosen item is gone, cleared, greyed out, run out, dried up, down to none or cannot be bought (each message counts on its own; a requirement taken off the list is a drop, not a sell-out), "gone": [listing ids named as unavailable], "price_changed": true if a price changed, "approved": true/false/null for a reply to a request for approval}"""

RULE = """A property requirement holds when an attribute whose key names that property (any key spelling, plain synonyms) contains the wanted value: one element of a comma or slash list is enough, a value containing the wanted words counts, units are converted, limits include the edge. When several attributes name the property, one of them stating the value is enough. When the store category itself says the value, the category meets the requirement. If all the attributes naming the property state something else, the listing fails, whatever other keys say. When no attribute names the property, a related attribute, the title or the description may state it. A listing whose own title or description plainly gives another value of that property fails even when an attribute says the wanted value (a title selling canvas against a required leather, a description giving 40 cm against a required 60 cm). Unlabelled figures prove nothing. A requirement to differ from a value is met unless the listing shows that value."""

JUDGE = """Decide which listings fit a product.
Product: PRODUCT (store category: LABEL)
Requirements: NEEDS
Rules: RULE The listing's category must be this kind of product. When the listing's fine print sends the reader to another listing for a property, the listing's own value for that property does not count.
Listings:
RECORDS
Output only JSON: {"verdicts": [{"id": listing id, "fit": "yes" | "no" | "maybe", "why": "at most 10 words"}]}. Use "maybe" when the record lacks the information (for example its properties were not read)."""

TARGET = """A shopper's requirement defines a value through another listing in an online store. Pull it apart. A refusal of everything other than the relation is no part of it: the relation is what the shopper is left demanding. Output only JSON:
{"property": the property of the shopper's product this is about, "relation": "same" (the same value as on that page, also when the requirement only points at the page), "not" (any value but that one, never that one), "partner" (a value made to suit that page's value under a store rule, not the identical value, even after a word of agreement; also when the sentence asks for whatever belongs or goes with that value), "within" (no further than some distance N from it), "floor" (that value or higher), "ceiling" (that value or lower), "distance": the number N for "within", else null, "shop": the shop named, "kind": the kind of product of that listing, "traits": ["each 'property value' phrase describing it", ...], "plain": when the requirement names no other listing, shop or page, the wanted value alone as a listing could print it, else null}
"property" is what the sentence demands of the shopper's own product: it comes before the relation words, and the properties that describe the other listing never are it. "kind" and "traits" describe the other listing only. Invented example: "the width of my shelves must suit whatever Acme's floor lamp with a height of 40 cm and a matte finish has" gives property "width", relation "partner", shop "Acme", kind "floor lamp", traits ["height 40 cm", "finish matte"]."""

READ_PAGE = """A buying agent is tracing where the value of PROPERTY is stated. Below is one listing's record. Output only JSON:
{"redirect": null, or {"shop": the shop or maker named, "kind": the kind of product named, "traits": [every phrase of that sentence describing that other listing, copied from it, one phrase per trait, words before its kind included ("used", "gray", "organic 750 ml"), never PROPERTY itself]} when the fine print sends the reader to another listing for PROPERTY in particular (or a plain synonym of it),
 "value": the listing's own value for PROPERTY from its attributes, or from its text when no attribute names it, else "",
 "pairs": {"x": "y", ...} when the record prints a store rule matching each value of PROPERTY to a partner value, and also when it says this listing's own value goes with some other value (then x is its own value and y that other value), else {}}
A list of codes or models the product fits is no redirect. A remark about a different property is no redirect. A remark that only mentions other versions or listings, without sending the reader there for PROPERTY, is no redirect. A sentence saying a version exists elsewhere, or that this is not some other version, tells nothing about this listing.
RECORD:
"""

QUERIES = """A store search did not find a listing that fits. Product: PRODUCT. Requirements: NEEDS. Queries tried: TRIED. Store categories seen: LABELS.
Suggest up to 3 new short search queries (2-5 words each: the wanted value words plus the product noun, or other names for the product). Output only JSON: {"queries": ["..."]}"""

MEASURE = """For each listing, give its value of MEASURE as a number in one common unit, or null when the record does not state it.
RECORDS
Output only JSON: {"values": {"listing id": number or null}}"""

CONFIRM = """Final review before an order is placed. The order must contain each requested product at its quantity, every requirement met, inside the budget, and when the pick rule is "cheapest" no cheaper listing of the same product may fit.
Request: REQUEST
Requirements in force: SPEC
Planned lines:
LINES
Other listings seen for these products:
OTHERS
Reading rule: RULE
Output only JSON: {"ok": true or false, "issues": ["..."], "swap": [{"product": number, "id": a listing id from above that should replace that product's line}], "contradicted": [numbers of the products whose planned line's record states a value that breaks a requirement; a record that is merely silent does not count]}. Reject only for a concrete defect visible in the records."""

AUDIT = """A shopping assistant found no fitting listing for a product and is about to tell the shopper so. Check the listings.
Product: PRODUCT. Requirements: NEEDS. Budget: BUDGET.
Reading rule: RULE
RECORDS
Output only JSON: {"fits": [ids of listings that do meet every requirement], "promising": [ids whose unread details could still make them fit]}"""

ASK_OK = """You are a shop assistant buying for a customer. Write ONE short sentence to the customer that does this: STEP. The planned order: LINES (total TOTAL). Output only that sentence."""

JUDGE = JUDGE.replace("RULE", RULE)
CONFIRM = CONFIRM.replace("RULE", RULE)
AUDIT = AUDIT.replace("RULE", RULE)

SORRY = "I looked through the store and nothing in stock matches everything you asked for, so there is no order this time."


# ============================================================================ small helpers
def to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return ""


def read_json(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    pieces = [fenced.group(1)] if fenced else []
    start = text.find("{")
    if start >= 0:
        pieces.append(text[start: text.rfind("}") + 1])
    for piece in pieces:
        try:
            value = json.loads(piece)
        except ValueError:
            continue
        if isinstance(value, dict):
            return value
    return None


def plain(text: Any) -> str:
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(text))
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def roots(text: Any) -> set[str]:
    out = set()
    for w in plain(text).split():
        if len(w) > 3 and w.endswith("s"):
            w = w[:-1]
        if len(w) >= 3:
            out.add(w)
    return out


def sends_for(sentence: str, prop: str) -> bool:
    """Is this sentence about where to read this property? The property comes before the words naming the other listing."""
    head = re.split(r"\b(?:from|by|to the|at the)\b", sentence, maxsplit=1, flags=re.I)[0]
    want = roots(prop)
    return bool(want) and want <= roots(head)


def keep_limit(reply: str, value: str) -> str:
    """Words setting a limit stay with the value: "at most 30 mm" must not shrink to "30 mm"."""
    limit = r"at\s+least|at\s+most|no\s+more\s+than|no\s+less\s+than|not\s+more\s+than|not\s+less\s+than|up\s+to|under|over|above|below|within|minimum(?:\s+of)?|maximum(?:\s+of)?"
    if not value or re.search(rf"\b(?:{limit}|between|or\s+more|or\s+less)\b", value, re.I):
        return value
    found = re.search(rf"\b(?:{limit})\s+{re.escape(value)}(?!\w)", reply, re.I) or re.search(rf"(?<!\w){re.escape(value)}\s+or\s+(?:more|less|fewer|higher|lower|above|below|under|over)\b", reply, re.I)
    return re.sub(r"\s+", " ", found.group(0)) if found else value


def number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        m = re.fullmatch(r"\s*[^\d\-]{0,4}(-?\d[\d,]*(?:\.\d+)?)\s*[a-zA-Z%¢$]{0,10}\s*", value)
        if m:
            return float(m.group(1).replace(",", ""))
    return None


def whole(value: Any) -> int | None:
    n = number(value)
    return int(n) if n is not None and n == int(n) else None


def short(value: Any, n: int) -> str:
    s = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return s if len(s) <= n else s[: n - 1] + "…"


def items_of(value: Any) -> list[Any]:
    return value if isinstance(value, list) else ([] if value in (None, "", {}) else [value])


def idish(value: Any) -> bool:
    return isinstance(value, str) and 3 <= len(value) <= 64 and " " not in value and bool(re.search(r"\d", value))


LIMIT_WORDS = ("not", "about", "at", "least", "most", "between", "under", "over", "up", "to", "or", "and", "by", "than", "more", "less", "above", "below", "higher", "lower")


def query_of(need: str) -> str:
    """Words to search a need by. Every word must match, so a figure goes without the word right after it, and a short adjective goes with its property."""
    prop, _, value = need.partition(":")
    inner = re.findall(r"\(([^()]*)\)", value)
    value = re.sub(r"\(.*?\)", " ", value)
    if inner and len(value.split()) <= 1 and not re.search(r"\d", inner[0]):
        value = inner[0]
    tokens = [t.strip(",.;") for t in value.split()]
    figures, others, after_figure = [], [], False
    for t in tokens:
        if re.fullmatch(r"\d+(?:\.\d+)?", t):
            figures.append(t)
            after_figure = True
            continue
        if not after_figure and t.lower() not in LIMIT_WORDS:
            others.append(t)
        after_figure = False
    if figures and not others:
        return " ".join(figures[:2])
    words = [t for t in tokens if t.lower() not in ("not", "about", "at", "least", "most", "between")]
    if len(words) == 1 and len(words[0]) <= 3 and words[0].isalpha():
        words += prop.split()[:2]
    return " ".join(words[:4])


class Printed:
    """Reads records a store prints as text ("key: value | key: value", "a=b; c=d" inside a value) into objects."""

    FIELD = re.compile(r"\s*([A-Za-z][A-Za-z0-9_ \-]{0,40}?)\s*[:=]\s*(.*)", re.S)

    def __init__(self) -> None:
        self.names: set[str] = set()

    @staticmethod
    def value(text: str) -> Any:
        text = text.strip()
        low = text.lower()
        if low in ("", "none", "null"):
            return None
        if low in ("true", "false"):
            return low == "true"
        if re.fullmatch(r"-?\d+(?:\.\d+)?", text):
            return float(text) if "." in text else int(text)
        return text

    @classmethod
    def block(cls, text: str) -> dict[str, str] | None:
        """A value made of "a=b" pairs."""
        pairs: dict[str, str] = {}
        last = None
        for piece in text.split(";"):
            if "=" in piece:
                last, _, value = piece.partition("=")
                last = last.strip()
                pairs[last] = value.strip()
            elif last and piece.strip():
                pairs[last] += "; " + piece.strip()
            elif piece.strip():
                return None
        return pairs if len(pairs) >= 2 else None

    def record(self, text: str, names: set[str] | None = None) -> dict[str, Any] | None:
        for mark in ("|", "\n", "\t", ";"):
            pieces = text.split(mark)
            if len(pieces) < 3:
                continue
            found: dict[str, str] = {}
            last = None
            for piece in pieces:
                got = self.FIELD.fullmatch(piece)
                if got and got.group(1) not in found and (names is None or got.group(1) in names):
                    last = got.group(1)
                    found[last] = got.group(2)
                elif last is None:
                    found = {}
                    break
                else:
                    found[last] += mark + piece
            if len(found) >= 3 and any(idish(v.strip()) for v in found.values()):
                return {k: self.block(v) or self.value(v) for k, v in found.items()}
        return None

    def read(self, node: Any) -> Any:
        if isinstance(node, list):
            texts = [n for n in node if isinstance(n, str)]
            rows = [self.record(t) for t in texts]
            if texts and all(rows):
                counts: dict[str, int] = {}
                for row in rows:
                    for key in row:
                        counts[key] = counts.get(key, 0) + 1
                common = {k for k, n in counts.items() if n * 2 >= len(rows)}
                self.names |= common
                return [self.record(t, common) or r for t, r in zip(texts, rows)]
            return [self.read(n) for n in node]
        if isinstance(node, dict):
            small = len(node) <= 3
            return {k: (self.read(v) if isinstance(v, (list, dict)) and (small or isinstance(v, list)) else self.read(v) if isinstance(v, str) and small else v) for k, v in node.items()}
        if isinstance(node, str):
            text = node.strip()
            if text[:1] in ("{", "["):
                try:
                    return self.read(json.loads(text))
                except ValueError:
                    pass
            row = self.record(text, self.names or None) or self.record(text)
            return row if row else node
        return node


# ============================================================================ transport
class Link:
    """Steps in the world and model calls, with the clock, step and spend accounting."""

    def __init__(self, binding: dict[str, Any], max_steps: int):
        self.t0 = time.monotonic()
        self.binding = binding
        self.max_steps = max_steps
        self.step = 0
        self.done = False
        self.broke = False
        self.cost = 0.0
        self.calls = 0
        self.no_switch: set[str] = {SHARP}   # models whose provider will not turn thinking off
        self.log: list[dict[str, Any]] = []
        self.printed = Printed()

    def age(self) -> float:
        return time.monotonic() - self.t0

    def steps_left(self) -> int:
        return self.max_steps - self.step

    def act(self, tool: str, args: dict[str, Any]) -> dict[str, Any] | None:
        if self.done or self.steps_left() <= 0 or self.age() > EPISODE_SECONDS - 5:
            return None
        for attempt in range(3):
            payload = dict(self.binding)
            payload["step"] = self.step + 1
            payload["action"] = {"name": tool, "args": args}
            _world.timeout = max(5, min(45, int(EPISODE_SECONDS - 4 - self.age())))
            result = _world.post_verbose("/environment/call", payload)
            data = result.data
            if isinstance(data, dict):
                self.step = data["step"] if isinstance(data.get("step"), int) else self.step + 1
                if data.get("finished"):
                    self.done = True
                self.log.append({"kind": "act", "step": self.step, "tool": tool, "args": short(args, 200)})
                data["observation"] = self.printed.read(data.get("observation"))
                return data
            error = result.error or {}
            status, body = error.get("status"), str(error.get("body") or "")
            self.log.append({"kind": "act_error", "tool": tool, "status": status, "body": body[:200]})
            if status == 409:
                found = re.search(r"expected (\d+)", body)
                if found:
                    self.step = int(found.group(1)) - 1
                    continue
                if re.search(r"finish|final|quarantin", body, re.I):
                    self.done = True
                return None
            if status == 402:
                self.broke = True
                return None
            if status in (400, 401, 403, 404, 422):
                return None
            time.sleep(1.5 * (attempt + 1))
        return None

    def say(self, prompt: str, tier: str = FAST, max_tokens: int = 900, think: bool = False) -> str:
        chain = [tier, BACKUP] if tier == FAST else [tier, FAST]
        for model in chain:
            if self.broke or self.age() > LAST_CALL:
                return ""
            if model != FAST and self.cost > SPEND_LIMIT:
                continue
            text = self._one(model, prompt, max_tokens, think)
            if text:
                return text
        return ""

    def _one(self, model: str, prompt: str, max_tokens: int, think: bool) -> str:
        body: dict[str, Any] = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": 0}
        if not think and model not in self.no_switch:
            body["reasoning"] = {"enabled": False}
        for attempt in range(3):
            if self.age() > LAST_CALL:
                return ""
            _models.timeout = max(8, min(50, int(EPISODE_SECONDS - 6 - self.age())))
            self.calls += 1
            result = _models.post_verbose("/inference/chat/completions", body)
            data = result.data
            if isinstance(data, dict) and data.get("choices"):
                usage = data.get("usage") or {}
                if isinstance(usage.get("cost"), (int, float)):
                    self.cost += float(usage["cost"])
                choice = data["choices"][0]
                text = to_text((choice.get("message") or {}).get("content"))
                if text.strip():
                    return text
                if choice.get("finish_reason") == "length" and attempt < 2:
                    # the whole budget went to thinking: retry with thinking off and more room
                    body = dict(body, max_tokens=min(4000, int(body["max_tokens"]) * 3))
                    if model not in self.no_switch:
                        body["reasoning"] = {"enabled": False}
                    continue
                return ""
            error = result.error or {}
            status = error.get("status")
            self.log.append({"kind": "model_error", "model": model, "status": status, "body": str(error.get("body") or "")[:160]})
            if status == 400 and "reasoning" in body:
                self.no_switch.add(model)
                body = {k: v for k, v in body.items() if k != "reasoning"}
                continue
            if status == 402:
                self.broke = True
                return ""
            if status in (400, 401, 403, 404):
                return ""
            time.sleep(2.0 * (attempt + 1))
        return ""

    def ask(self, prompt: str, tier: str = FAST, max_tokens: int = 900, think: bool = False) -> dict[str, Any]:
        got = read_json(self.say(prompt, tier, max_tokens, think))
        if got is None and not self.broke and self.age() < LAST_CALL and self.cost < SPEND_LIMIT:
            got = read_json(self.say(prompt + "\nReply with the JSON object only.", SHARP if tier == FAST else FAST, max_tokens, False))
        return got or {}


# ============================================================================ what the store has shown
class Shelf:
    """Listings seen so far, merged across views, with their fields sorted by meaning."""

    PRICE = re.compile(r"price|cost|amount", re.I)
    STOCK = re.compile(r"stock|avail|orderable|buyable|purchas|sellable|in_store|on_hand", re.I)
    CATEGORY = re.compile(r"categ|group|depart|aisle|shelf|(^|_)path$", re.I)
    SELLER = re.compile(r"^(?!.*(?:comment|note|remark|review|blurb|desc|text|about)).*(?:shop|seller|sold_?by|vendor|store|merchant|retailer)", re.I)
    BRAND = re.compile(r"brand|maker|manufactur", re.I)
    TITLE = re.compile(r"name|title|heading", re.I)
    IDKEY = re.compile(r"(^|_)(id|pid|ref|sku|code)(_|$)|id$|_id|code$|ref$", re.I)

    def __init__(self) -> None:
        self.recs: dict[str, dict[str, Any]] = {}
        self.order: list[str] = []
        self.id_key: str | None = None
        self.read_attrs: set[str] = set()
        self.read_page: set[str] = set()
        self.gone: set[str] = set()
        self.roles: dict[str, str] = {}
        self.link_cache: dict[str, tuple[tuple[str, str], dict[str, Any]]] = {}
        self.part_notes: dict[str, set[str]] = {}

    # ---- reading results
    def _id_key_of(self, rows: list[dict[str, Any]]) -> str | None:
        best = None
        for key in rows[0]:
            values = [r.get(key) for r in rows]
            if not all(idish(v) for v in values) or len(set(values)) < len(values):
                continue
            score = (bool(self.IDKEY.search(str(key))), -abs(len(str(values[0])) - 14))
            if best is None or score > best[0]:
                best = (score, key)
        return best[1] if best else None

    def take(self, observation: Any, single_view: str | None = None) -> tuple[list[str], list[str]]:
        """Absorb listings from a result. single_view is "attributes" or "page" for a one-listing view.
        Returns (listing ids seen, loose note sentences)."""
        ids: list[str] = []
        notes: list[str] = []

        def record(node: dict[str, Any], top: bool) -> None:
            key = self.id_key
            ident = node.get(key) if key else None
            if not idish(ident):
                return
            if ident not in self.recs:
                self.recs[ident] = {}
                self.order.append(ident)
            merged = self.recs[ident]
            for k, v in node.items():
                if (self.STOCK.search(str(k)) or k == self.roles.get("stock")) and v in (False, 0, "false", "no", "out of stock"):
                    self.gone.add(ident)
                elif (self.STOCK.search(str(k)) or k == self.roles.get("stock")) and v in (True, "true", "yes", "in stock") and ident in self.gone:
                    self.gone.discard(ident)
                merged[k] = v
            ids.append(ident)
            if top and single_view == "attributes":
                self.read_attrs.add(ident)
            if top and single_view == "page":
                self.read_page.add(ident)
            if self.attributes(ident):
                self.read_attrs.add(ident)

        def walk(node: Any, depth: int, top: bool) -> None:
            if depth > 6:
                return
            if isinstance(node, list):
                rows = [n for n in node if isinstance(n, dict)]
                if rows and len(rows) * 2 >= len(node):
                    if not self.id_key or not all(self.id_key in r for r in rows):
                        found = self._id_key_of(rows) if all(len(r) >= 2 for r in rows) else None
                        if found:
                            self.id_key = found
                    if self.id_key and all(self.id_key in r for r in rows):
                        for r in rows:
                            record(r, False)
                        return
                for n in node:
                    if isinstance(n, str):
                        if len(n.split()) >= 3:
                            notes.append(n)
                    else:
                        walk(n, depth + 1, False)
                return
            if isinstance(node, dict):
                if self.id_key and self.id_key in node and idish(node.get(self.id_key)):
                    record(node, top)
                    return
                if not self.id_key and top and single_view:
                    found = self._id_key_of([node])
                    if found:
                        self.id_key = found
                        record(node, True)
                        return
                for value in node.values():
                    if isinstance(value, str):
                        if len(value.split()) >= NOTE_WORDS:
                            notes.append(value)
                    else:
                        walk(value, depth + 1, top and isinstance(value, dict))
            elif isinstance(node, str) and len(node.split()) >= NOTE_WORDS:
                notes.append(node)

        walk(observation, 0, True)
        return list(dict.fromkeys(ids)), notes

    # ---- reading fields
    def _field(self, ident: str, pattern: re.Pattern, kind: str) -> Any:
        rec = self.recs.get(ident) or {}
        role = next((r for r, p in (("price", self.PRICE), ("category", self.CATEGORY), ("seller", self.SELLER), ("brand", self.BRAND)) if p is pattern), None)
        named = rec.get(self.roles.get(role)) if role else None
        if kind == "num" and number(named) is not None:
            return number(named)
        if kind == "str" and isinstance(named, str) and named.strip():
            return named.strip()
        for key, value in rec.items():
            if not pattern.search(str(key)):
                continue
            if kind == "num" and number(value) is not None:
                return number(value)
            if kind == "str" and isinstance(value, str) and value.strip():
                return value.strip()
        return None

    def price(self, ident: str) -> float | None:
        return self._field(ident, self.PRICE, "num")

    def category(self, ident: str) -> str:
        return self._field(ident, self.CATEGORY, "str") or ""

    def seller(self, ident: str) -> str:
        return self._field(ident, self.SELLER, "str") or self._field(ident, self.BRAND, "str") or ""

    def title(self, ident: str) -> str:
        rec = self.recs.get(ident) or {}
        named = rec.get(self.roles.get("title"))
        if isinstance(named, str) and named.strip():
            return named
        for key, value in rec.items():
            if self.TITLE.search(str(key)) and not self.CATEGORY.search(str(key)) and not self.SELLER.search(str(key)) and not self.BRAND.search(str(key)) and isinstance(value, str):
                return value
        texts = [v for v in rec.values() if isinstance(v, str) and 8 <= len(v) <= 200]
        return texts[0] if texts else ident

    def attributes(self, ident: str) -> dict[str, Any]:
        rec = self.recs.get(ident) or {}
        best: dict[str, Any] = {}
        for value in rec.values():
            if isinstance(value, dict):
                flat = {k: v for k, v in value.items() if isinstance(v, (str, int, float)) and not isinstance(v, bool)}
                if len(flat) >= 2 and len(flat) > len(best):
                    best = flat
        return best

    def texts(self, ident: str) -> list[str]:
        rec = self.recs.get(ident) or {}
        title = self.title(ident)
        found = [v for k, v in rec.items() if isinstance(v, str) and len(v) >= 25 and v != title and not self.CATEGORY.search(str(k))]
        return sorted(found, key=len)   # fine print first: it is the short text

    def available(self, ident: str) -> bool:
        return ident not in self.gone

    def words_of(self, ident: str) -> str:
        parts = [self.title(ident), self.category(ident), self.seller(ident)] + self.texts(ident)
        parts += [f"{k} {v}" for k, v in self.attributes(ident).items()]
        return " ".join(str(p) for p in parts).lower()

    def links(self, ident: str) -> dict[str, Any]:
        """Fitment codes printed in a listing's text: its own model, what it fits, its own part code, continuation parts."""
        title = self.title(ident)
        words = " ".join(v for v in (self.recs.get(ident) or {}).values() if isinstance(v, str) and v != title)
        kept = self.link_cache.get(ident)
        if kept and kept[0] == (title, words):
            return kept[1]
        code = r"[A-Z]{1,5}-?\d{3,6}[A-Z]?"
        own = re.search(r"\bpart\s+(" + code + r")", title)
        model = re.search(r"\bmodel\s+(?:ref\.?|no\.?|number)\s*:?\s*(" + code + r")\b", words, re.I) or re.search(r"\bmodel\s*:?\s*(" + code + r")\b", words, re.I)
        fits: set[str] = set()
        families: list[tuple[str, int, set[str]]] = []
        for m in re.finditer(r"\b(with|for|fits|suits)\b\s*:?\s*([^.;\n]*)", words, re.I):
            family = re.search(r"\bevery\s+([A-Z]{1,5}-?\d{1,4})(x+)\s*models?\b(?:\s*,?\s*(?:except|but not|other than)\s+(.*))?", m.group(2), re.I)
            if family:
                stem = family.group(1).replace("-", "").upper()
                families.append((stem, len(stem) + len(family.group(2)), set(re.findall(code, family.group(3) or ""))))
            else:
                fits.update(re.findall(code, m.group(2)))
        for m in re.finditer(r"\b(?:fits|suits)\b\s*:?\s*([^.;\n]*)", words, re.I):
            fits.update(re.findall(r"[A-Z][a-z]+ \d{3,6}\b", m.group(1)))
        named = code + r"|[A-Z][a-z]+ \d{3,6}\b"
        for m in re.finditer(r"\bpart\s+(" + code + r")\s+also\s+fits\s*:?\s*([^.;\n]*)", words, re.I):
            self.part_notes.setdefault(m.group(1).upper(), set()).update(re.findall(named, m.group(2)))
            for c in re.findall(named, m.group(2)):
                fits.discard(c)
            if own and m.group(1) == own.group(1):
                fits.update(re.findall(named, m.group(2)))
        names = {model.group(1)} if model else set()
        if model:
            for m in re.finditer(r"\balso\b(?!\s+fits)([^.;\n]*)", words, re.I):
                names.update(re.findall(named, m.group(1)))
        more = []
        for m in re.finditer(r"\bpart\s+(" + code + r")\b(?!\s+also)", words, re.I):
            if (not own or m.group(1).upper() != own.group(1).upper()) and m.group(1) not in more:
                more.append(m.group(1))
        fits.difference_update(more)
        fits.difference_update(re.findall(code, title))
        found = {"own": own.group(1) if own else "", "model": model.group(1) if model else "", "names": names, "fits": fits, "families": families, "more": more}
        self.link_cache[ident] = ((title, words), found)
        return found

    def names_of(self, model: str) -> set[str]:
        memo = self.__dict__.setdefault("names_memo", {})
        stamp = (model, len(self.order), len(self.read_attrs))
        if stamp not in memo:
            memo[stamp] = self.names_of_now(model)
        return memo[stamp]

    def names_of_now(self, model: str) -> set[str]:
        """Every name a device goes by, as far as the listings seen print them."""
        names = {model}
        for ident in self.order:
            found = self.links(ident)
            if found["model"] == model:
                names |= found["names"]
        return names

    def names_model(self, codes: dict[str, Any], model: str) -> bool:
        """Does this listing's own fit statement name the device, singly or as a family, under any of its names?"""
        names = self.names_of(model)
        if names & codes["fits"] or names & self.part_notes.get(codes["own"].upper(), set()):
            return True
        for name in names:
            compact = name.replace("-", "").upper()
            if any(compact.startswith(stem) and len(compact) == width and not (names & others) for stem, width, others in codes["families"]):
                return True
        return False

    @staticmethod
    def states_fit(codes: dict[str, Any]) -> bool:
        return bool(codes["fits"] or codes["families"] or codes["more"])

    def card(self, ident: str, text_room: int = 420) -> str:
        """One listing as a compact line for a prompt."""
        bits = [f"id={ident}", f"title={short(self.title(ident), 120)}"]
        if self.category(ident):
            bits.append(f"category={self.category(ident)}")
        if self.price(ident) is not None:
            bits.append(f"price={self.price(ident):g}")
        bits.append("in stock" if self.available(ident) else "SOLD OUT")
        if self.seller(ident):
            bits.append(f"shop={self.seller(ident)}")
        attrs = self.attributes(ident)
        if attrs:
            bits.append("properties{" + "; ".join(f"{k}: {short(str(v), 60)}" for k, v in list(attrs.items())[:24]) + "}")
        else:
            bits.append("properties not read")
        for t in self.texts(ident)[:2]:
            bits.append("text: " + short(t, text_room))
        return " | ".join(bits)


# ============================================================================ the store's tools
class Kit:
    """Which tool does what, and how to call it."""

    def __init__(self, tools: list[dict[str, Any]]):
        self.schemas: dict[str, dict[str, Any]] = {}
        self.descriptions: dict[str, str] = {}
        for t in tools:
            fn = t.get("function") if isinstance(t, dict) else None
            if isinstance(fn, dict) and isinstance(fn.get("name"), str):
                params = fn.get("parameters") if isinstance(fn.get("parameters"), dict) else {}
                self.schemas[fn["name"]] = params
                self.descriptions[fn["name"]] = str(fn.get("description") or "")
        self.map: dict[str, dict[str, Any]] = {}

    def props(self, tool: str | None) -> dict[str, Any]:
        return (self.schemas.get(tool or "") or {}).get("properties") or {}

    def has(self, tool: Any, arg: Any = None) -> bool:
        if not isinstance(tool, str) or tool not in self.schemas:
            return False
        return arg is None or (isinstance(arg, str) and arg in self.props(tool))

    def learn(self, link: Link) -> None:
        listing = "\n".join(f"- {name}: {self.descriptions[name]} | arguments: {json.dumps(self.schemas[name])}" for name in self.schemas)
        got = link.ask(TOOLS + "\n\nTOOLS:\n" + listing, FAST, 1500)
        for job, spec in got.items():
            if isinstance(spec, dict) and self.has(spec.get("tool")):
                self.map[job] = {k: v for k, v in spec.items() if v not in (None, "")}
        self._fill_gaps()

    def _guess(self, pattern: str, exclude: set[str]) -> str | None:
        for name in self.schemas:
            if name in exclude:
                continue
            if re.search(pattern, name + " " + self.descriptions[name], re.I):
                return name
        return None

    def _free_string(self, tool: str, pattern: str = "") -> str | None:
        props = self.props(tool)
        names = [k for k, v in props.items() if isinstance(v, dict) and v.get("type") == "string" and "enum" not in v]
        named = [k for k in names if pattern and re.search(pattern, k, re.I)]
        return (named or names or [None])[0]

    def _fill_gaps(self) -> None:
        used = {m["tool"] for m in self.map.values()}
        guesses = {
            "search": r"search|find|seek|query|browse|lookup",
            "message": r"message|ask|chat|shopper|customer|buyer|user",
            "buy": r"checkout|purchase|order|buy|settle",
            "decline": r"decline|give.?up|refuse|abstain|no.?purchase|abandon",
            "cart": r"cart|bag|basket|stash|hold",
            "attributes": r"spec|propert|technical|datasheet|attribute|detail",
        }
        for job, pattern in guesses.items():
            if job not in self.map:
                tool = self._guess(pattern, used)
                if tool:
                    self.map[job] = {"tool": tool}
                    used.add(tool)
        search = self.map.get("search")
        if search and not self.has(search["tool"], search.get("query_arg")):
            search["query_arg"] = self._free_string(search["tool"], r"query|text|phrase|keyword|term|search")
        for job in ("attributes", "page"):
            spec = self.map.get(job)
            if spec and not self.has(spec["tool"], spec.get("id_arg")):
                spec["id_arg"] = self._free_string(spec["tool"], r"id|code|ref|item|product|listing")
        if "attributes" not in self.map:
            viewer = self.map.get("page") or ({"tool": self._guess(r"open|page|view|inspect|describe|listing|item", used)} if self._guess(r"open|page|view|inspect|describe|listing|item", used) else None)
            if viewer:
                self.map["attributes"] = dict(viewer)
                spec = self.map["attributes"]
                if not self.has(spec["tool"], spec.get("id_arg")):
                    spec["id_arg"] = self._free_string(spec["tool"], r"id|code|ref|item|product|listing")
        if "page" not in self.map and "attributes" in self.map:
            self.map["page"] = dict(self.map["attributes"])
        side = self.map.get("compare")
        if side is None:
            for name in self.schemas:
                arrays = [k for k, v in self.props(name).items() if isinstance(v, dict) and v.get("type") == "array"]
                if arrays and re.search(r"compar|contrast|weigh|side.by.side", name + " " + self.descriptions[name], re.I):
                    self.map["compare"] = {"tool": name, "ids_arg": arrays[0]}
                    break
        elif not self.has(side.get("tool"), side.get("ids_arg")):
            self.map.pop("compare", None)
        msg = self.map.get("message")
        if msg and not self.has(msg["tool"], msg.get("text_arg")):
            msg["text_arg"] = self._free_string(msg["tool"], r"text|message|body|note|content")
        cart = self.map.get("cart")
        if cart and not self.has(cart["tool"], cart.get("id_arg")):
            cart["id_arg"] = self._free_string(cart["tool"], r"id|code|item|product|listing")
        for job in ("buy", "decline"):
            spec = self.map.get(job)
            if spec is not None:
                spec["args"] = spec.get("args") if isinstance(spec.get("args"), dict) else {}

    def tool(self, job: str) -> str | None:
        spec = self.map.get(job)
        return spec["tool"] if spec else None

    def complete(self, tool: str, args: dict[str, Any]) -> dict[str, Any] | None:
        """Fill required arguments the call lacks (an enum's first value, a number's minimum); None if it cannot."""
        schema = self.schemas.get(tool) or {}
        props = schema.get("properties") or {}
        out = {k: v for k, v in args.items() if k in props or not props}
        for key in schema.get("required") or []:
            if key in out:
                continue
            spec = props.get(key) or {}
            if isinstance(spec.get("enum"), list) and spec["enum"]:
                out[key] = spec["enum"][0]
            elif spec.get("type") in ("integer", "number"):
                out[key] = spec.get("minimum", 1)
            else:
                return None
        for key, value in list(out.items()):
            if (self.props(tool).get(key) or {}).get("type") == "array" and not isinstance(value, list):
                out[key] = value = [value]
            spec = props.get(key) or {}
            if spec.get("type") == "string" and isinstance(value, str) and isinstance(spec.get("maxLength"), int):
                out[key] = value[: spec["maxLength"]]
            if spec.get("type") == "integer" and isinstance(value, float):
                out[key] = int(value)
            if spec.get("type") == "string" and isinstance(value, (int, float)) and not isinstance(value, bool):
                out[key] = str(value)
        return out


# ============================================================================ what the shopper wants
class Product:
    def __init__(self, number_: int, name: str, qty: int, needs: list[str], points: list[str], fits_with: int | None = None):
        self.number = number_
        self.name = name
        self.qty = max(1, qty)
        self.needs = list(dict.fromkeys(n for n in needs if n))
        self.points = list(dict.fromkeys(p for p in points if p))
        self.resolved: dict[str, str] = {}
        self.fits_with = fits_with
        self.names: list[str] = []
        self.asked: list[str] = []
        self.free: list[str] = []
        self.label = ""
        self.pool: list[str] = []
        self.tried: list[str] = []
        self.version = 0
        self.verdicts: dict[str, tuple[int, str, str]] = {}
        self.choice: str | None = None

    def requirements(self) -> list[str]:
        out = list(self.needs)
        for p in self.points:
            out.append(self.resolved.get(p) or p.split(":", 1)[0] + ": [unknown so far: its value is defined through another listing; the description of that listing is not a requirement on this product]")
        return out

    def settled(self, prop: str) -> bool:
        """True when a stated requirement, an earlier question or a no-preference answer already covers this property."""
        target = roots(prop)
        if not target:
            return False
        covered = [roots(k.split(":", 1)[0]) for k in self.needs + self.points + self.asked + self.free]
        return any(c and (len(c & target) == min(len(c), len(target))) for c in covered)

    def change(self) -> None:
        self.version += 1


# ============================================================================ the pipeline
class Hotaru:
    def __init__(self, problem: dict[str, Any]):
        env = problem.get("environment") or {}
        view = env.get("policy_view") or {}
        self.request = str(view.get("query") or "")
        self.link = Link(dict(env.get("binding") or {}), max(1, whole(view.get("max_steps")) or 30))
        self.link.done = bool(view.get("finished"))
        self.opening = view.get("opening")
        self.kit = Kit(view.get("tools") or [])
        self.shelf = Shelf()
        self.codes_unsearchable = False
        self.search_memo: dict[str, tuple[Any, list[str]]] = {}
        self.found_by: dict[str, set[str]] = {}
        self.singly_opened: set[str] = set()
        self.held = 0
        self.fit_settled = False
        self.refused = False
        self.turns: dict[int, int] = {}
        self.products: list[Product] = []
        self.budget: float | None = None
        self.budget_each = False
        self.pick = "any"
        self.measure = ""
        self.msg_cap: int | None = None
        self.msgs_sent = 0
        self.in_stock = True
        self.before_order: list[str] = []
        self.ask_first: list[str] = []
        self.hints: list[dict[str, Any]] = []
        self.quiet = True
        self.cart: dict[str, int] = {}
        self.notes_new: list[str] = []
        self.looked: set[str] = set()
        self.batch = 4
        self.approved = False
        self.trace = self.link.log

    # ---------------------------------------------------------------- utilities
    def note(self, **entry: Any) -> None:
        self.trace.append(entry)

    def steps_left(self) -> int:
        return self.link.steps_left()

    def reserve(self) -> int:
        """Steps kept back for the cart and the checkout, and until the first refusal also for a replacement search and its reading."""
        if self.refused:
            return len(self.products) + 2 + self.pending_parts()
        if len(self.products) > 1:
            return len(self.products) + 3 + (1 if self.before_order else 0) + self.pending_parts()
        return (9 if self.link.max_steps >= 24 else 7) + (1 if self.before_order else 0) + self.pending_parts()

    def spare(self) -> int:
        return self.steps_left() - self.reserve() - self.held

    def late(self) -> bool:
        return self.link.age() > COMMIT_BY or self.link.done or self.link.broke

    def products_text(self) -> str:
        return "; ".join(f"{p.number}: {p.name}" for p in self.products)

    def spec_text(self) -> str:
        lines = []
        for p in self.products:
            lines.append(f"product {p.number}: {p.name} x{p.qty}" + (f" (store category {p.label})" if p.label else "")
                         + " - " + ("; ".join(p.requirements()) or "no stated requirements"))
        if self.budget is not None:
            lines.append(f"budget: at most {self.budget:g} " + ("for each product" if self.budget_each else "in total"))
        lines.append(f"pick rule: {self.pick}" + (f" ({self.measure})" if self.measure else ""))
        if self.in_stock:
            lines.append("only items in stock")
        return "\n".join(lines)

    def call(self, job: str, args: dict[str, Any], view: str | None = None) -> Any:
        """Run one store call; absorb listings and notes. Returns the observation or None."""
        tool = self.kit.tool(job)
        if not tool or self.link.done:
            return None
        shaped = self.kit.complete(tool, args)
        if shaped is None:
            return None
        data = self.link.act(tool, shaped)
        if data is None:
            return None
        observation = data.get("observation")
        absorb = job in ("search", "browse", "attributes", "page", "compare")
        if absorb:
            _, notes = self.shelf.take(observation, view)
        else:
            _, notes = Shelf().take(observation)  # notes only; cart and checkout replies hold no new listings
        if notes:
            self.notes_new.extend(n for n in notes if n not in self.notes_new)
        return observation

    # ---------------------------------------------------------------- read the request
    def read(self) -> None:
        got = self.link.ask(READ + "\n\nREQUEST:\n" + self.request, SHARP, 2600, think=True)
        for i, entry in enumerate(items_of(got.get("products")), 1):
            if not isinstance(entry, dict) or not str(entry.get("name") or "").strip():
                continue
            self.products.append(Product(len(self.products) + 1, str(entry["name"]).strip(), whole(entry.get("qty")) or 1,
                                         [str(n).strip() for n in items_of(entry.get("needs")) if str(n).strip()],
                                         [str(n).strip() for n in items_of(entry.get("points")) if str(n).strip()], whole(entry.get("fits_with"))))
        if not self.products:
            self.products = [Product(1, " ".join(self.request.split()[:6]) or "item", 1, [], [])]
        self.budget = number(got.get("budget"))
        if self.budget is not None:
            # the number must be one the request states; a model that converted cents to dollars is undone here
            stated = {float(x.replace(",", "")) for x in re.findall(r"\d[\d,]*(?:\.\d+)?", self.request)}
            if self.budget not in stated:
                for factor in (100.0, 0.01):
                    if round(self.budget * factor, 2) in stated:
                        self.budget = round(self.budget * factor, 2)
                        break
        self.budget_each = str(got.get("budget_scope") or "").lower() == "each"
        self.pick = str(got.get("pick") or "any").lower() if str(got.get("pick") or "").lower() in ("cheapest", "most", "least") else "any"
        self.measure = str(got.get("measure") or "")
        self.msg_cap = whole(got.get("messages"))
        steps = whole(got.get("steps"))
        if steps and 5 <= steps < self.link.max_steps:
            self.link.max_steps = steps
        self.in_stock = got.get("in_stock") is not False
        self.before_order = [str(s) for s in items_of(got.get("before_order")) if str(s).strip()]
        self.ask_first = [str(s) for s in items_of(got.get("ask_first")) if str(s).strip()]
        self.hints = [h for h in items_of(got.get("hints")) if isinstance(h, dict)]
        self.quiet = got.get("quiet") is not False
        self.note(kind="read", products=[(p.name, p.qty, p.needs, p.points) for p in self.products], budget=self.budget, pick=self.pick, messages=self.msg_cap)

    # ---------------------------------------------------------------- search
    def search(self, query: str | None = None, category: str | None = None, page: Any = None, token: Any = None, cap: float | None = None, free: bool = False) -> list[str]:
        spec = self.kit.map.get("search") or {}
        job = "search"
        side = self.kit.map.get("browse") or {}
        if not query and side.get("tool") and (category or cap is not None) and not (spec.get("category_arg") if category else spec.get("price_cap_arg")):
            spec, job = side, "browse"
        args: dict[str, Any] = {}
        if spec.get("mode_arg"):
            mode = spec.get("filter_mode") if category and spec.get("filter_mode") else spec.get("keyword_mode")
            if mode is not None:
                args[spec["mode_arg"]] = mode
        if query and spec.get("query_arg"):
            args[spec["query_arg"]] = query
        if category and spec.get("category_arg"):
            args[spec["category_arg"]] = category
        limit = None if free else (cap if cap is not None else self.budget)
        if limit is not None and spec.get("price_cap_arg") and self.kit.has(spec["tool"], spec.get("price_cap_arg")):
            args[spec["price_cap_arg"]] = limit
        if page is not None and spec.get("page_arg"):
            args[spec["page_arg"]] = page
        if token is not None and spec.get("token_arg"):
            args[spec["token_arg"]] = token
        if not args:
            return []
        memo_key = json.dumps([job, args], sort_keys=True, default=str)
        if memo_key in self.search_memo:
            observation, ids = self.search_memo[memo_key]
            self.last_search = (args, observation, ids, job)
            return ids
        observation = self.call(job, args)
        if observation is None:
            return []
        ids, _ = self.shelf.take(observation)
        self.search_memo[memo_key] = (observation, ids)
        if query:
            for ident in ids:
                self.found_by.setdefault(ident, set()).add(query)
        self.last_search = (args, observation, ids, job)
        if ids and not self.shelf.roles and not getattr(self, "roles_read", False):
            self.roles_read = True
            record = self.shelf.recs.get(ids[0]) or {}
            shown = {k: short(v, 60) for k, v in record.items()}
            got = self.link.ask(FIELDS + json.dumps(shown, ensure_ascii=False), FAST, 200)
            self.shelf.roles = {r: k for r, k in got.items() if isinstance(k, str) and k in record}
            self.note(kind="fields", roles=self.shelf.roles)
        return ids

    def next_page(self) -> list[str]:
        """The following page of the last search, when the result says there is one."""
        args, observation, ids, job = getattr(self, "last_search", ({}, None, [], "search"))
        spec = self.kit.map.get(job) or {}
        if not ids or not isinstance(observation, dict):
            return []
        token = None
        for key, value in observation.items():
            if re.search(r"next|more|continue|cursor|token", str(key), re.I):
                if value in (None, "", False):
                    return []
                if not isinstance(value, (list, dict)):
                    token = value
        held_cap = args.get(spec.get("price_cap_arg", "")) if spec.get("price_cap_arg") else None
        if token is not None and spec.get("token_arg"):
            return self.search(args.get(spec.get("query_arg", "")), args.get(spec.get("category_arg", "")), token=token, cap=held_cap)
        if token is not None and spec.get("page_arg"):
            return self.search(args.get(spec.get("query_arg", "")), args.get(spec.get("category_arg", "")), page=token, cap=held_cap)
        page_arg = spec.get("page_arg")
        if page_arg and self.kit.has(spec["tool"], page_arg):
            pspec = self.kit.props(spec["tool"]).get(page_arg) or {}
            current = whole(args.get(page_arg)) or int(pspec.get("minimum", 1) or 1)
            if re.search(r"offset|skip|start", page_arg, re.I):
                nxt = current + len(ids)
            else:
                nxt = current + 1
            if isinstance(pspec.get("maximum"), (int, float)) and nxt > pspec["maximum"]:
                return []
            if token is None and not self._more_hint(observation, len(ids), current):
                return []
            return self.search(args.get(spec.get("query_arg", "")), args.get(spec.get("category_arg", "")), page=nxt, cap=held_cap)
        return []

    def _more_hint(self, observation: dict[str, Any], shown: int, current: int) -> bool:
        for key, value in observation.items():
            if re.search(r"count|total|matched|found|number", str(key), re.I) and whole(value) is not None:
                return whole(value) > shown * max(1, current)
        return shown >= 5

    # ---------------------------------------------------------------- scout
    def label_for(self, product: Product, ids: list[str]) -> str:
        counts: dict[str, int] = {}
        name = roots(product.name)
        for ident in ids:
            cat = self.shelf.category(ident)
            if not cat:
                continue
            common = roots(cat) & name
            weight = 3 * len(common) if common else (1 if roots(self.shelf.title(ident)) & name else 0)
            counts[cat] = counts.get(cat, 0) + weight
        def shared(cat: str) -> int:
            return sum(1 for b in name if any(a.startswith(b) or b.startswith(a) for a in roots(cat) if min(len(a), len(b)) >= 4))
        asked = roots(self.request)
        live = [c for c, n in counts.items() if n > 0]
        if not live:
            return ""
        most = max(shared(c) for c in live)
        closest = [c for c in live if shared(c) == most]
        own = name | {w for n in product.needs for w in roots(n.split(":", 1)[-1])}

        def beyond(cat: str) -> int:
            return sum(1 for a in roots(cat) if not any(a.startswith(b) or b.startswith(a) for b in own if min(len(a), len(b)) >= 4))
        fewest = min(beyond(c) for c in closest)
        closest = [c for c in closest if beyond(c) == fewest]
        spelled = [c for c in closest if len(roots(c)) > 1 and roots(c) <= asked]
        return max(spelled or closest, key=lambda c: counts[c])

    def relabel(self, product: Product) -> None:
        better = self.label_for(product, product.pool)
        if better and better != product.label:
            product.label = better
            product.change()
        if product.label:
            have = roots(product.label)
            keep = [n for n in product.needs if not (roots(n.split(":", 1)[-1]) and roots(n.split(":", 1)[-1]) <= have)]
            if len(keep) != len(product.needs):
                product.needs = keep
                product.change()

    def members(self, product: Product) -> list[str]:
        """Listings of this product's kind seen so far."""
        if product.label:
            return [i for i in self.shelf.order if self.shelf.category(i) == product.label]
        return [i for i in product.pool if i in self.shelf.recs]

    def scout(self) -> None:
        share = max(4, int((0.42 if len(self.products) == 1 else 0.36) * self.link.max_steps) // max(1, len(self.products)))
        for product in self.products:
            if self.late() or self.spare() < 4:
                break
            start = self.link.step
            query = " ".join([w for w in product.name.split() if w.lower() not in ("or", "and", "the", "a", "an", "of", "for", "with")][:5]) or product.name
            ids = self.search(query)
            product.tried.append(query)
            if not self.label_for(product, ids) and len(query.split()) > 1 and self.spare() > 6:
                query = query.split()[-1]
                ids = self.search(query) or ids
                product.tried.append(query)
            product.pool.extend(i for i in ids if i not in product.pool)
            product.label = self.label_for(product, ids)
            values = " ".join(n.split(":", 1)[-1] for n in product.needs[:2]).strip()
            if values and self.spare() > 6 and self.link.step - start < share - 2:
                head = (product.name.split() or [""])[-1].lower().rstrip("s")
                joined = values if head and head in values.lower() else values + " " + product.name
                q2 = " ".join(joined.split()[:6])
                if q2 not in product.tried:
                    product.tried.append(q2)
                    more = self.search(q2)
                    product.pool.extend(i for i in more if i not in product.pool)
            self.relabel(product)
            spec = self.kit.map.get("search") or {}
            if product.label and (len(self.products) < 3 or not product.pool) and (spec.get("category_arg") or (self.kit.map.get("browse") or {}).get("category_arg")) and self.spare() > 6:
                more = self.search(None, product.label)
                product.pool.extend(i for i in more if i not in product.pool)
            if self.pick == "cheapest" and self.spare() > 10:
                # the cheapest fit may sit on a later page of the widest search
                more = self.next_page()
                product.pool.extend(i for i in more if i not in product.pool)
            if not product.label:
                product.label = self.label_for(product, product.pool)
            room = max(2, min(9, share - (self.link.step - start)))
            self.read_some(product, room)
        self.note(kind="scout", labels=[p.label for p in self.products], steps=self.link.step)

    def ranked(self, product: Product, ids: list[str]) -> list[str]:
        """Candidates in reading order: in stock, inside the budget, closest to the requirements; cheapest first under a cheapest pick."""
        want = set()
        for need in product.requirements():
            want |= roots(need.split(":", 1)[-1])
        cap = self.budget if self.budget is not None else None
        models = self.fit_targets(product)

        def key(ident: str) -> tuple:
            price = self.shelf.price(ident)
            over = cap is not None and price is not None and price * (product.qty if not self.budget_each else 1) > cap
            score = len(want & roots(self.shelf.words_of(ident)))
            hit = any(roots(q) & want for q in self.found_by.get(ident, ()))
            misfit = bool(models) and not any(self.covers(ident, m) for m in models)
            if self.pick == "cheapest":
                return (over, not self.shelf.available(ident), misfit, not hit, price if price is not None else 1e12, -score)
            return (over, not self.shelf.available(ident), misfit, -score, not hit, price if price is not None else 1e12)

        return sorted(dict.fromkeys(ids), key=key)

    def fit_targets(self, product: Product) -> list[str]:
        """The device codes this product has to fit, as far as they are known."""
        found = re.findall(r"\b[A-Z]{2,}-?\d{3,}\b", " ".join(product.requirements()))
        if product.fits_with and 1 <= product.fits_with <= len(self.products):
            dev = self.products[product.fits_with - 1]
            ids = [dev.choice] if dev.choice else self.fits(dev)[:4]
            found += [self.shelf.links(i)["model"] for i in ids if i and self.shelf.links(i)["model"]]
        return list(dict.fromkeys(found))

    def read_attrs(self, ident: str) -> bool:
        if ident in self.shelf.read_attrs or not self.kit.tool("attributes"):
            return False
        spec = self.kit.map["attributes"]
        opened = self.call("attributes", {spec.get("id_arg"): ident}, "attributes") is not None
        if opened:
            self.singly_opened.add(ident)
        return opened

    def read_page(self, ident: str) -> bool:
        spec = self.kit.map.get("page") or {}
        if not spec or spec.get("tool") == self.kit.tool("attributes") or ident in self.shelf.read_page:
            return False
        return self.call("page", {spec.get("id_arg"): ident}, "page") is not None

    def reread(self, ident: str) -> bool:
        """Look at a listing again (stock and price can change), even when it was read before."""
        spec = self.kit.map.get("page") or self.kit.map.get("attributes") or {}
        if not spec.get("id_arg"):
            return False
        view = "attributes" if spec.get("tool") == self.kit.tool("attributes") else "page"
        return self.call("page" if "page" in self.kit.map else "attributes", {spec["id_arg"]: ident}, view) is not None

    def read_many(self, ids: list[str]) -> int:
        """Read listings' attributes: several per step when the store's side-by-side tool shows them, else one at a time."""
        todo = [i for i in dict.fromkeys(ids) if i not in self.shelf.read_attrs]
        done = 0
        spec = self.kit.map.get("compare") or {}
        least = 1 if self.refused else 3
        while len(todo) >= 2 and self.batch > 1 and spec.get("ids_arg") and self.spare() >= least and not self.late():
            chunk, rest = todo[:self.batch], todo[self.batch:]
            self.call("compare", {spec["ids_arg"]: chunk}, "attributes")
            got = [i for i in chunk if i in self.shelf.read_attrs]
            done += len(got)
            if got:
                todo = [i for i in chunk if i not in got] + rest
                if len(got) < len(chunk):
                    todo = rest + [i for i in chunk if i not in got]
                    break
            else:
                self.batch = 2 if self.batch > 2 else 0     # too many at once, or this tool shows no attributes
        for ident in todo:
            if self.spare() < least or self.late():
                break
            if self.read_attrs(ident):
                done += 1
        return done

    def read_some(self, product: Product, count: int) -> int:
        unread = [i for i in self.ranked(product, self.members(product) or product.pool) if i not in self.shelf.read_attrs and self.shelf.available(i)]
        return self.read_many(unread[:count])

    # ---------------------------------------------------------------- interview
    def key_stats(self, product: Product) -> list[tuple[str, int, list[str]]]:
        stats: dict[str, tuple[int, list[str]]] = {}
        for ident in self.members(product) or product.pool:
            for key, value in self.shelf.attributes(ident).items():
                name = " ".join([w for w in plain(key).split() if len(w) > 2] or plain(key).split())
                if not name or re.search(r"\b(id|sku|url|link|image|warranty|shipping|option|options|variant|barcode|price|seller|shop|brand|align|margin|indent|padding|font|text)\b", name):
                    continue
                n, vals = stats.get(name, (0, []))
                text = short(str(value).lower(), 40)
                stats[name] = (n + 1, vals + ([text] if text not in vals else []))
        ranked = sorted(stats.items(), key=lambda kv: -(2 * kv[1][0] + min(4, len(kv[1][1]))))
        return [(k, n, vals[:3]) for k, (n, vals) in ranked if not product.settled(k)][:40]

    def vocabulary(self) -> None:
        """One ordered property list per product: the model's list, then every other key the listings carry."""
        lines = []
        for p in self.products:
            lines.append(f"product {p.number}: {p.name} - settled: " + ("; ".join(p.requirements()) or "nothing"))
            for key, n, vals in self.key_stats(p):
                lines.append(f"   key '{key}' on {n} listings, e.g. {' | '.join(vals)}")
        extra = ""
        if self.ask_first:
            extra += "\nThe request says to ask about: " + "; ".join(self.ask_first)
        if self.hints:
            extra += "\nHints read from the story: " + "; ".join(f"product {h.get('product')}: {h.get('property')}" for h in self.hints)
        got = self.link.ask(VOCAB + "\n\nREQUEST:\n" + self.request + "\n\nPRODUCTS AND LISTING KEYS:\n" + "\n".join(lines) + extra, SHARP, 1500)
        offered: dict[int, list[str]] = {}
        for entry in items_of(got.get("lists")):
            if isinstance(entry, dict) and whole(entry.get("product")):
                offered[whole(entry["product"])] = [str(x) for x in items_of(entry.get("properties"))]
        for p in self.products:
            chosen: list[str] = []
            keys = [k for k, _, _ in self.key_stats(p)]
            carried = [x for x in offered.get(p.number, []) if roots(x) & {w for k in keys for w in roots(k)}]
            for raw in carried + keys + offered.get(p.number, []):
                name = re.split(r"\s+(?:and|or)\s+|[,/;(]", raw)[0].strip(" ?.").lower()
                if not name or len(name.split()) > 4 or p.settled(name):
                    continue
                self.place_name(chosen, name)
            p.names = chosen
        self.note(kind="vocabulary", names=[p.names for p in self.products])

    @staticmethod
    def place_name(chosen: list[str], name: str) -> None:
        """Keep one name per concept: a name whose words sit inside another's is that other name; the longer wording stays."""
        new = roots(name)
        for i, old in enumerate(chosen):
            have = roots(old)
            if new and have and (new <= have or have <= new):
                if have < new:
                    chosen[i] = name
                return
        chosen.append(name)

    @staticmethod
    def turned(names: list[str], stage: int) -> list[str]:
        """The stage's order of a list: as ranked, reversed, centre-out, then interleavings."""
        groups: dict[str, list[str]] = {}
        for name in names:
            groups.setdefault((name.split() or [""])[-1].lower(), []).append(name)
        blocks = [sorted(g, key=lambda n: -len(n.split())) for g in groups.values()]
        if stage == 1:
            blocks = blocks[::-1]
        elif stage == 2:
            # wants and other subjects cluster near the top of a ranked list, so the third order starts inside that cluster
            aim = 0.15 * (len(blocks) - 1)
            blocks = [b for _, b in sorted(enumerate(blocks), key=lambda x: (abs(x[0] - aim), x[0]))]
        elif stage > 2 and blocks:
            shift = stage % len(blocks)
            blocks = blocks[shift:] + blocks[:shift]
        return [n for b in blocks for n in b]

    def sweep_text(self, order: list[Product], stage: int, room: int, shrink: float) -> str:
        queue = [p for p in order if p.names]
        if stage == 1:
            queue = queue[::-1]
        elif stage >= 2 and len(queue) > 2:
            queue = queue[len(queue) // 2:] + queue[: len(queue) // 2]
        queue.sort(key=lambda p: self.gave.get(p.number, 0))
        if len(queue) == 2:
            queue = queue[:1]
            stage = self.turns.get(queue[0].number, 0)
            self.turns[queue[0].number] = stage + 1
        per = max(1, int((room - 45 * len(queue)) / max(1, len(queue)) / 16 * shrink))
        parts = []
        for p in queue:
            listed = self.turned(p.names[:max(per, 1)] if len(p.names) > per else p.names, stage)
            if not listed:
                continue
            words = listed[0] if len(listed) == 1 else ", ".join(listed[:-1]) + " or " + listed[-1]
            parts.append(f"What do you want for the {p.name} in terms of {words}?")
        return " ".join(parts)

    def sweep(self) -> None:
        held = 1 if self.before_order else 0
        if self.messages_left() <= held or not self.kit.tool("message") or self.spare() < 2:
            return
        self.vocabulary()
        spec = self.kit.map.get("message") or {}
        limit = (self.kit.props(spec.get("tool")).get(spec.get("text_arg")) or {}).get("maxLength")
        room = min(780, int(limit) - 10) if isinstance(limit, (int, float)) else 780
        order = list(self.products)
        stage = 0
        shrink = 1.0
        seen: set[str] = set()
        self.gave: dict[int, int] = {}
        while self.messages_left() > held and self.spare() > 1 and not self.late():
            question = self.sweep_text(order, stage, room, shrink)
            if not question:
                break
            answer = self.send(question)
            if answer is None:
                break
            got = self.link.ask(HEAR.replace("QUESTION", question).replace("ANSWER", answer).replace("PRODUCTS", self.products_text()), FAST, 700)
            wants = got.get("want") is True and str(got.get("value") or "").strip()
            self.note(kind="ask", q=short(question, 300), a=short(answer, 300), got=short(got, 300), stage=stage)
            if wants:
                prop = str(got.get("property") or "").strip()
                value = keep_limit(answer, str(got.get("value")).strip())
                shop_named = bool(re.search(r"[A-Z][\w-]+(?:'s\b| page\b)|\b(?:sold by|from|at|by)\s+[A-Z]", answer))
                elsewhere = shop_named and (got.get("points") is True or bool(re.search(r"\b(?:listing|page)\b", answer, re.I)))
                need = f"{prop}: {answer.strip()}" if elsewhere else f"{prop}: {value}"
                target = self.products[min(max(1, whole(got.get("product")) or order[0].number), len(self.products)) - 1]
                if elsewhere:
                    head = re.split(r"\b(?:listing|page)\b", answer, maxsplit=1, flags=re.I)[0]
                    closest = max(self.products, key=lambda p: len(roots(head) & (roots(p.name) | roots(p.label))))
                    if roots(head) & (roots(closest.name) | roots(closest.label)):
                        target = closest
                    spec = self.link.ask(TARGET + "\n\nREQUIREMENT:\n" + answer, FAST, 500)
                    prop = str(spec.get("property") or prop).strip()
                    need = f"{prop}: {answer.strip()}"
                (target.points if elsewhere else target.needs).append(need)
                target.change()
                gone = roots(prop) - {"type", "kind"}
                before = len(target.names)
                target.names = [n for n in target.names if not (gone and (roots(n) - {"type", "kind"}) & gone)]
                if target.names and self.messages_left() > held:
                    close = self.link.ask(NEAR.replace("PROPERTY", prop).replace("NAMES", ", ".join(target.names)), FAST, 300)
                    drop = {str(x).strip().lower() for x in items_of(close.get("near"))}
                    target.names = [n for n in target.names if n.lower() not in drop]
                if prop.lower() in seen or len(target.names) == before:
                    stage += 1      # the same want again, or one no listed name covers: a near name still answers for it
                seen.add(prop.lower())
                self.gave[target.number] = self.gave.get(target.number, 0) + 1
            elif got.get("split") is True and re.search(r"\b(?:one (?:at a time|thing|by one)|fewer|too many|at once|single)\b", answer, re.I):
                shrink *= 0.25
            else:
                stage += 1
            if self.notes_new:
                self.absorb_notes()

    def messages_left(self) -> int:
        cap = self.msg_cap if self.msg_cap is not None else 3
        return cap - self.msgs_sent

    def send(self, text: str) -> str | None:
        spec = self.kit.map.get("message") or {}
        if not spec or not spec.get("text_arg") or self.messages_left() <= 0:
            return None
        tool = spec["tool"]
        args = self.kit.complete(tool, {spec["text_arg"]: text})
        if args is None:
            return None
        data = self.link.act(tool, args)
        if data is None:
            return None
        self.msgs_sent += 1
        observation = data.get("observation")
        if isinstance(observation, str):
            return observation
        answer, notes = "", []
        if isinstance(observation, dict):
            direct = [v for v in observation.values() if isinstance(v, str)]
            answer = max(direct, key=len) if direct else ""
            for v in observation.values():
                if isinstance(v, list):
                    notes += [x for x in v if isinstance(x, str)]
                if isinstance(v, dict):
                    inner = [x for x in v.values() if isinstance(x, str)]
                    if inner and not answer:
                        answer = max(inner, key=len)
        if notes:
            self.notes_new.extend(n for n in notes if n not in self.notes_new)
        return answer or json.dumps(observation)[:400]

    # ---------------------------------------------------------------- notes inside results
    def absorb_notes(self) -> dict[str, Any]:
        notes, self.notes_new = self.notes_new, []
        if not notes:
            return {}
        wanted = "; ".join(f"{p.number}: {p.name} ({', '.join(p.requirements()) or 'no requirements'})" for p in self.products)
        got = self.link.ask(NOTE.replace("NOTES", "\n".join(f"- {short(n, 300)}" for n in notes[:8])).replace("PRODUCTS", wanted)
                            .replace("BUDGET", "none" if self.budget is None else f"{self.budget:g}"), FAST, 600)
        self.note(kind="notes", notes=[short(n, 160) for n in notes[:6]], got=short(got, 300))
        budget = number(got.get("budget"))
        if budget is not None and budget > 0:
            written = [float(x.replace(",", "")) for n in notes for x in re.findall(r"\d[\d,]*(?:\.\d+)?", n)]
            if written and budget not in written:
                near = [w for w in written if abs(w - budget * 100) < 1 or abs(w - budget / 100) < 0.01]
                budget = near[0] if near else budget
        if budget is not None and budget > 0 and budget != self.budget:
            self.budget = budget
            for p in self.products:
                p.change()
        for entry in items_of(got.get("drop")):
            if isinstance(entry, dict):
                p = self.products[min(max(1, whole(entry.get("product")) or 1), len(self.products)) - 1]
                gone = roots(str(entry.get("property") or "").split(":", 1)[0])

                def same(n: str) -> bool:
                    have = roots(n.split(":", 1)[0])
                    return bool(gone and have and (have <= gone or gone <= have))

                keep = [n for n in p.needs if not same(n)]
                keep_points = [n for n in p.points if not same(n)]
                if len(keep) != len(p.needs) or len(keep_points) != len(p.points):
                    p.needs, p.points = keep, keep_points
                    p.change()
        for entry in items_of(got.get("add")):
            if isinstance(entry, dict) and ":" in str(entry.get("need") or ""):
                p = self.products[min(max(1, whole(entry.get("product")) or 1), len(self.products)) - 1]
                if entry["need"] not in p.needs:
                    p.needs.append(str(entry["need"]))
                    p.change()
        for ident in items_of(got.get("gone")):
            if str(ident) in self.shelf.recs:
                self.shelf.gone.add(str(ident))
        if got.get("approved") is True:
            self.approved = True
        return got

    # ---------------------------------------------------------------- chase pointers
    def chase_all(self) -> None:
        for product in self.products:
            for pointer in list(product.points):
                if pointer in product.resolved or self.late() or self.spare() < 4:
                    continue
                self.chase(product, pointer)

    def locate(self, shop: str, kind: str, traits: list[str], used: set[str], start: int, allowed: int) -> str | None:
        """The listing a description picks out: rows of that shop, read until one carries the traits."""
        shop_key = plain(shop)

        def ours() -> list[str]:
            rows = [i for i in self.shelf.order if i not in used and shop_key and plain(self.shelf.seller(i)) == shop_key]
            kin = [i for i in rows if roots(kind) & roots(self.shelf.category(i))]
            return kin or rows

        def told(ident: str) -> str:
            """What the listing says of itself: mentions of other versions are not about it."""
            own = []
            for piece in [self.shelf.title(ident), self.shelf.category(ident), self.shelf.seller(ident)] + self.shelf.texts(ident):
                own += [s for s in re.split(r"(?<=[.!?])\s+", str(piece)) if not re.search(r"\belsewhere\b|\bis not\b|\bnot the\b|\bnot a\b|\bnot (?:to |from |by |on )?(?:this|that)\b|\brather than\b|\binstead of\b", s, re.I)]
            own += [f"{k} {v}" for k, v in self.shelf.attributes(ident).items()]
            return plain(" ".join(own))

        def against(ident: str, trait: str) -> bool:
            """An attribute named by the trait states something else."""
            words = roots(trait)
            for key, val in self.shelf.attributes(ident).items():
                named = roots(key)
                rest = words - named
                if named and named <= words and rest and not (rest & roots(val)):
                    return True
            return False

        def score(ident: str) -> int:
            text = set(told(ident).split())
            return sum(1 for t in traits if all(w in text for w in plain(t).split()) and not against(ident, t))

        def lean(ident: str) -> int:
            text = set(told(ident).split())
            return sum(1 for t in traits for w in plain(t).split() if w in text)

        wanted = max(1, sum(len(plain(t).split()) for t in traits))

        def kin_seen() -> bool:
            return any(roots(kind) & roots(self.shelf.category(i)) for i in ours())

        if not kin_seen():
            # searching needs every word to match, and a kind word may not be on the listing: the shop name alone is the safe query
            for query in (shop, f"{shop} {kind.split()[-1] if kind else ''}".strip()):
                if self.link.step - start >= allowed or self.spare() < 2:
                    break
                self.search(query)
                if kin_seen():
                    break
        def own_page(ident: str) -> str:
            # the side-by-side view shows the same record; a chain counts only pages opened on their own
            if ident not in self.singly_opened and self.spare() >= 1 and (score(ident) > 0 or not traits):
                self.reread(ident)
                self.singly_opened.add(ident)
            return ident

        pages = 0
        while pages < 3:
            pages += 1
            for _ in range(8):
                rows = sorted(ours(), key=lambda i: (-score(i), -lean(i), i in self.shelf.read_attrs))
                sure = [i for i in rows if i in self.shelf.read_attrs and score(i) >= max(1, len(traits))]
                if sure:
                    return own_page(sure[0])
                unread = [i for i in rows if i not in self.shelf.read_attrs]
                if not unread or self.link.step - start >= allowed or self.spare() < 2:
                    break
                self.read_many(unread[:max(1, self.batch)])
            if self.link.step - start >= allowed or self.spare() < 3 or not self.next_page():
                break
        best = sorted(ours(), key=lambda i: (-score(i), -lean(i)))
        return own_page(best[0]) if best and (score(best[0]) > 0 or lean(best[0]) > 0 or not traits) else None

    def chase(self, product: Product, pointer: str) -> None:
        start = self.link.step
        allowed = min(24, max(3, self.spare() - 2))
        spec = self.link.ask(TARGET + "\n\nREQUIREMENT:\n" + pointer, FAST, 500)
        prop = str(spec.get("property") or pointer.split(":", 1)[0]).strip()
        plain_value = str(spec.get("plain") or "").strip()
        if plain_value and not str(spec.get("shop") or "").strip() and not items_of(spec.get("traits")):
            if pointer in product.points:
                product.points.remove(pointer)
            product.needs.append(f"{prop}: {plain_value}")
            product.change()
            self.note(kind="plain_after_all", pointer=short(pointer, 120), need=f"{prop}: {plain_value}")
            return
        relation = str(spec.get("relation") or "same").lower()
        shop, kind = str(spec.get("shop") or ""), str(spec.get("kind") or "")
        traits = [str(t) for t in items_of(spec.get("traits")) if not (roots(prop) and roots(prop) <= roots(t))]
        used: set[str] = set()
        pairs: dict[str, str] = {}
        value, last = "", ""
        for _ in range(7):
            if self.late() or self.spare() < 2:
                break
            ident = self.locate(shop, kind, traits, used, start, allowed)
            if not ident:
                break
            used.add(ident)
            last = ident
            page = self.link.ask(READ_PAGE.replace("PROPERTY", prop) + self.shelf.card(ident, 700), FAST, 700)
            if isinstance(page.get("pairs"), dict) and page["pairs"]:
                pairs = {plain(k): str(v) for k, v in page["pairs"].items()}
            hop = page.get("redirect")
            self.note(kind="hop", at=ident, shop=shop, traits=short(traits, 160), read=short(page, 240))
            if isinstance(hop, dict) and hop.get("shop"):
                said = [s for t in self.shelf.texts(ident) for s in re.split(r"(?<=[.!?])\s+", t) if str(hop.get("shop")).lower() in s.lower()]
                if not any(sends_for(s, prop) for s in said):
                    hop = None
            if isinstance(hop, dict) and hop.get("shop"):
                shop, kind = str(hop.get("shop")), str(hop.get("kind") or kind)
                traits = [str(t) for t in items_of(hop.get("traits")) if not roots(t) <= roots(prop)]
                home = {w for i in self.shelf.order if plain(self.shelf.seller(i)) == plain(shop) for w in roots(self.shelf.category(i))}
                kept = []
                for t in traits:
                    left = [w for w in plain(t).split() if w not in plain(shop).split() and not (roots(w) & home) and w not in ("from", "with", "the", "and", "that", "which", "has", "have", "is", "in", "of", "for", "a", "an", "its")]
                    if left:
                        kept.append(" ".join(left))
                traits = kept or traits
                if not traits and kind:
                    sections = {w for i in self.shelf.order if plain(self.shelf.seller(i)) == plain(shop) for w in roots(self.shelf.category(i))}
                    traits = [w for w in plain(kind).split() if len(w) > 2 and not ({w} & sections) and not (roots(w) & sections)]
                if not traits:
                    # the model gave no description: the sentence that names the shop is the description
                    for text in self.shelf.texts(ident):
                        for sentence in re.split(r"(?<=[.!?])\s+", text):
                            if shop.lower() in sentence.lower():
                                tail = re.split(re.escape(shop), sentence, flags=re.I)[-1]
                                traits = [t.strip() for t in re.split(r",|\band\b|\bwith\b|\(|\)", tail) if len(t.strip()) > 2 and "not " not in t.lower()]
                continue
            value = str(page.get("value") or "").strip()
            said = roots(" ".join(self.shelf.texts(ident) + [self.shelf.title(ident)]))
            if value and not (any(roots(k) & roots(prop) for k in self.shelf.attributes(ident)) or roots(prop) <= said):
                value = ""
            if value or not traits:
                break
        if not value:
            self.note(kind="chase_lost", pointer=short(pointer, 160), hops=list(used), spec=short(spec, 200), steps=self.link.step - start, allowed=allowed)
            return
        if relation == "partner" and not pairs and last:
            home = plain(self.shelf.seller(last))
            others = [i for i in self.shelf.order if plain(self.shelf.seller(i)) == home and i not in used and i not in self.shelf.read_attrs][:4]
            if others and self.spare() >= 2:
                self.read_many(others)
            for other in [i for i in self.shelf.order if plain(self.shelf.seller(i)) == home and i not in used and self.shelf.texts(i)][:6]:
                extra = self.link.ask(READ_PAGE.replace("PROPERTY", prop) + self.shelf.card(other, 700), FAST, 700)
                if isinstance(extra.get("pairs"), dict) and extra["pairs"]:
                    found_rule = {plain(k): str(v) for k, v in extra["pairs"].items()}
                    own_there = str(extra.get("value") or "").strip()
                    if plain(value) not in found_rule and own_there and plain(own_there) in found_rule:
                        value = own_there
                        last = other
                    pairs = found_rule
                    used.add(other)
                    break
        if relation == "partner":
            final = pairs.get(plain(value)) or next((v for k, v in pairs.items() if k in plain(value) or plain(value) in k), "") or f"the partner of {value} in the shop's pairing"
        elif relation == "not":
            final = f"not {value}"
        elif relation == "floor":
            final = f"at least {value}"
        elif relation == "ceiling":
            final = f"at most {value}"
        elif relation == "within" and number(spec.get("distance")) is not None and re.search(r"\d", value):
            centre = float(re.search(r"\d+(?:\.\d+)?", value).group(0))
            gap = number(spec.get("distance"))
            final = f"between {centre - gap:g} and {centre + gap:g}"
        else:
            final = value
        product.resolved[pointer] = f"{prop}: {final}"
        product.change()
        self.note(kind="chased", pointer=short(pointer, 120), value=value, final=final, hops=list(used), last=last, steps=self.link.step - start)

    def redirect_check(self, product: Product, ident: str) -> None:
        """A chosen listing whose fine print sends a needed property to another listing: chase that listing's value."""
        for text in self.shelf.texts(ident):
            for sentence in re.split(r"(?<=[.!?])\s+", text):
                if not re.search(r"\b(rather than|instead|another|different|not (?:to |from |by |on )?(?:this|that)|taken from|comes? from|go by|look to|refer)\b", sentence, re.I):
                    continue
                props = {plain(n.split(":", 1)[0]) for n in product.requirements()}
                if not any(sends_for(sentence, prop) for prop in props if prop):
                    continue
                pointer = f"{sentence.strip()} (fine print of {ident})"
                if pointer not in product.points:
                    product.points.append(pointer)
                    self.chase(product, pointer)

    # ---------------------------------------------------------------- select
    def judge(self, product: Product, ids: list[str]) -> None:
        def stale(i: str) -> bool:
            old = product.verdicts.get(i)
            return old is None or old[0] != product.version or (old[1] == "maybe" and i in self.shelf.read_attrs and old[2] != "read")

        todo = [i for i in ids if stale(i)]
        for start in range(0, len(todo), 10):
            chunk = todo[start: start + 10]
            if not chunk or self.link.age() > LAST_CALL:
                return
            prompt = (JUDGE.replace("PRODUCT", product.name).replace("LABEL", product.label or "unknown")
                      .replace("NEEDS", "; ".join(product.requirements()) or "none beyond the kind of product")
                      .replace("RECORDS", "\n".join(self.shelf.card(i) for i in chunk)))
            got = self.link.ask(prompt, FAST, 2600, think=True)
            for v in items_of(got.get("verdicts")):
                if isinstance(v, dict) and str(v.get("id")) in chunk:
                    fit = str(v.get("fit") or "maybe").lower()
                    if str(v.get("id")) not in self.shelf.read_attrs and fit == "yes" and self.kit.tool("attributes") and product.requirements():
                        fit = "maybe"   # a fit is only trusted on read properties
                    product.verdicts[str(v["id"])] = (product.version, fit if fit in ("yes", "no", "maybe") else "maybe", "read" if str(v["id"]) in self.shelf.read_attrs else "row")

    def fits(self, product: Product) -> list[str]:
        out = []
        taken = {p.choice for p in self.products if p is not product and p.choice}
        for ident, (version, fit, _) in product.verdicts.items():
            if version == product.version and fit == "yes" and self.shelf.available(ident) and self.affordable(product, ident) and ident not in taken:
                out.append(ident)
        own = [i for i in out if product.label and self.shelf.category(i) == product.label]
        return self.by_evidence(product, own or out)

    def backing(self, product: Product, ident: str) -> int:
        """How many of the product's needs this listing states under a key naming their property."""
        attrs = self.shelf.attributes(ident)
        score = 0
        for need in product.requirements():
            prop, _, value = need.partition(":")
            words = roots(value) - {"least", "most", "between", "not"}
            for key, val in attrs.items():
                if roots(prop) & roots(key) and (not words or words & roots(str(val))):
                    score += 1
                    break
        return score

    def well_backed(self, product: Product, ids: list[str]) -> int:
        wanted = len(product.requirements())
        return sum(1 for i in ids if self.backing(product, i) >= wanted)

    def more_backing(self, product: Product) -> None:
        """Further rows of the value searches, read and judged, so that a second solid listing is at hand."""
        if getattr(product, "backed_version", None) == product.version:
            return
        product.backed_version = product.version
        least = 1 if self.refused else 6
        want = set()
        for need in product.requirements():
            want |= roots(need.split(":", 1)[-1])
        waiting = [i for i in self.members(product) if i not in self.shelf.read_attrs and self.shelf.available(i) and self.affordable(product, i)
                   and any(roots(q) & want for q in self.found_by.get(i, ()))][:4]
        if waiting and self.spare() >= least:
            self.read_many(waiting)
            self.judge(product, waiting)
            if self.well_backed(product, self.fits(product)) >= (1 if self.refused else 2):
                return
        for query in [q for q in reversed(product.tried) if q][:2]:
            if self.spare() < 9 or self.late():
                return
            more = self.search(query)
            product.pool.extend(i for i in more if i not in product.pool)
            for _ in range(2):
                if self.spare() < 9 or self.late():
                    break
                more = self.next_page()
                if not more:
                    break
                product.pool.extend(i for i in more if i not in product.pool)
            fresh = [i for i in self.ranked(product, [i for i in product.pool if not product.label or self.shelf.category(i) == product.label]) if i not in self.shelf.read_attrs][:4]
            if fresh and self.spare() >= 6:
                self.read_many(fresh)
                self.judge(product, fresh)
            if self.well_backed(product, self.fits(product)) >= 2:
                return

    def by_evidence(self, product: Product, ids: list[str]) -> list[str]:
        """Listings stating each need under a key that names its property come first, then those whose own words say it too."""
        def strength(ident: str) -> int:
            said = roots(self.shelf.title(ident) + " " + " ".join(str(t) for t in self.shelf.texts(ident)))
            score = 2 * self.backing(product, ident)
            for need in product.requirements():
                words = roots(need.partition(":")[2]) - {"least", "most", "between", "not"}
                if words and words <= said:
                    score += 1
            return score
        return sorted(ids, key=lambda i: -strength(i))

    def maybes(self, product: Product) -> list[str]:
        taken = {p.choice for p in self.products if p is not product and p.choice}
        return [i for i, (v, fit, _) in product.verdicts.items() if v == product.version and fit == "maybe" and self.shelf.available(i) and self.affordable(product, i) and i not in taken]

    def affordable(self, product: Product, ident: str) -> bool:
        price = self.shelf.price(ident)
        if self.budget is None or price is None:
            return True
        return price * (1 if self.budget_each else product.qty) <= self.budget

    def candidates(self, product: Product) -> list[str]:
        ids = [i for i in self.ranked(product, self.members(product) + product.pool) if self.shelf.available(i) and self.affordable(product, i)]
        if product.label:
            ids = [i for i in ids if self.shelf.category(i) == product.label] + [i for i in ids if self.shelf.category(i) != product.label]
        return ids[:60]

    def targeted(self, product: Product) -> None:
        """Search on each drawn-out or stated value once, before judging: browsing rarely reaches the narrow set."""
        plainvals = []
        for need in product.requirements():
            value = need.split(":", 1)[-1]
            if "[unknown" not in need and len(value) <= 30 and not re.search(r"\b(at least|at most|between|not)\b", value, re.I):
                plainvals.append(query_of(need))
        together = " ".join(list(dict.fromkeys(" ".join(plainvals).split()))[:6])
        if len(plainvals) >= 2 and together not in product.tried and self.spare() >= 5 and not self.late():
            product.tried.append(together)
            ids = self.search(together)
            product.pool.extend(i for i in ids if i not in product.pool)
            if ids and not product.label:
                product.label = self.label_for(product, ids)
        for need in product.requirements()[-3:]:
            query = query_of(need)
            if "[unknown" in need or not query or self.spare() < 5 or self.late():
                continue
            if re.fullmatch(r"[\d. ]+", query):
                head = next((w for w in reversed(product.name.split()) if w.lower() not in ("or", "and", "the", "a", "an", "of", "for", "with")), "")
                query = f"{head} {query}".strip()
            if query in product.tried:
                continue
            noun = next((w for w in reversed(product.name.split()) if w.lower() not in ("or", "and", "the", "a", "an", "of", "for", "with")), "")
            ids = []
            paired = f"{query} {noun}" if noun and noun.lower() not in query.lower() and not re.fullmatch(r"[\d. ]+", query) else ""
            named_as = need.split(":", 1)[0].strip()
            if paired and re.search(r"\d", query) and len(named_as.split()) <= 2 and not (roots(named_as) & roots(query)):
                paired = f"{query} {named_as} {noun}"
            if paired and paired not in product.tried:
                product.tried.append(paired)
                ids = self.search(paired)
                product.pool.extend(i for i in ids if i not in product.pool)
            if sum(1 for i in ids if self.shelf.category(i) == product.label) >= 3 or self.spare() < 5:
                continue
            product.tried.append(query)
            bare = self.search(query)
            if not bare and re.fullmatch(r"[A-Za-z]{4,}", query) and self.spare() >= 5:
                other = query[:-1] if query.lower().endswith("s") else query + "s"
                product.tried.append(other)
                bare = self.search(other)
                query = other if bare else query
            product.pool.extend(i for i in bare if i not in product.pool)
            ids = bare or ids
            if not [i for i in ids if self.shelf.category(i) == product.label] and len(query.split()) > 1 and noun and self.spare() >= 5:
                for word in query.split()[:2]:
                    shorter = f"{word} {noun}"
                    if shorter in product.tried or self.spare() < 5:
                        continue
                    product.tried.append(shorter)
                    some = self.search(shorter)
                    product.pool.extend(i for i in some if i not in product.pool)
                    if [i for i in some if self.shelf.category(i) == product.label]:
                        ids = some
                        break
            for _ in range(2):
                if not ids or not product.label or self.spare() < 9 or sum(1 for i in product.pool if self.shelf.category(i) == product.label and query.split()[0].lower() in self.shelf.words_of(i)) >= 4:
                    break
                ids = self.next_page()
                product.pool.extend(i for i in ids if i not in product.pool)
            if not product.label:
                product.label = self.label_for(product, ids)

    def page_narrow(self, product: Product) -> None:
        """Result rows come in no price order: under a cheapest pick, page the narrowest searches so that cheap rows get read."""
        if self.pick != "cheapest":
            return
        spec = self.kit.map.get("search") or {}
        capped = bool(spec.get("price_cap_arg") or (self.kit.map.get("browse") or {}).get("price_cap_arg"))
        done = product.__dict__.setdefault("paged_queries", set())
        tried = [q for q in product.tried if q != " ".join(product.name.split()[:6]) and q not in done][-3:] or [q for q in product.tried[-1:] if q not in done]
        for query in reversed(tried):
            done.add(query)
            if self.spare() < 8 or self.late():
                return
            ids = self.search(query)
            product.pool.extend(i for i in ids if i not in product.pool)
            shown = len(ids)
            ended = False
            for _ in range(4 if capped else 40):
                if self.spare() < 8 or self.late():
                    return
                more = self.next_page()
                if not more:
                    ended = True
                    break
                shown += len(more)
                product.pool.extend(i for i in more if i not in product.pool)
            total = next((whole(v) for k, v in (getattr(self, "last_search", ({}, None))[1] or {}).items() if re.search(r"count|total|matched|found", str(k), re.I) and whole(v) is not None), None) if isinstance(getattr(self, "last_search", ({}, None))[1], dict) else None
            if ended and (total is None or total <= shown) and product.requirements():
                product.__dict__["value_rows_all_seen"] = product.version

    def wide_cheap(self, product: Product) -> None:
        """A cheapest pick over more listings than one search can show: further searches on the rarer words of the cheapest rows seen,
        each followed for as long as its pages bring new rows."""
        if self.pick != "cheapest" or (self.kit.map.get("search") or {}).get("price_cap_arg"):
            return
        if product.__dict__.get("value_rows_all_seen") == product.version:
            return
        floor = 2 if self.refused else 8
        used = product.__dict__.setdefault("wide_used", set())
        values = [query_of(n) for n in product.requirements() if "[unknown" not in n and not re.search(r"\b(?:at least|at most|between|not)\b", n, re.I)]
        noun = next((v for v in values if v), "") or (product.name.split() or [""])[-1]
        named = roots(product.name) | roots(noun)
        for _ in range(12):
            if self.spare() < floor or self.late():
                return
            rows = [i for i in self.members(product) if self.shelf.available(i) and self.shelf.price(i) is not None]
            if not rows:
                return
            rows.sort(key=lambda i: self.shelf.price(i))
            counts: dict[str, int] = {}
            for ident in rows:
                for word in roots(self.shelf.title(ident)):
                    counts[word] = counts.get(word, 0) + 1
            limit = max(3, len(rows) // 4)
            word = ""
            for ident in rows[:15]:
                title_words = re.findall(r"[^\W\d_]{4,}", self.shelf.title(ident))
                usable = [w for w in title_words if w.lower() not in used and not (roots(w) & named) and counts.get(next(iter(roots(w)), ""), 0) <= limit]
                if usable:
                    word = usable[0]
                    break
            if not word:
                return
            used.add(word.lower())
            query = f"{word} {noun}".strip()
            before = set(product.pool)
            ids = self.search(query)
            product.pool.extend(i for i in ids if i not in product.pool)
            for _ in range(6):
                if not [i for i in ids if i not in before] or self.spare() < floor or self.late():
                    break
                before = set(product.pool)
                ids = self.next_page()
                product.pool.extend(i for i in ids if i not in product.pool)
        return

    def select(self, product: Product) -> str | None:
        self.relabel(product)
        self.targeted(product)
        self.relabel(product)
        self.page_narrow(product)
        self.wide_cheap(product)
        for round_ in range(3):
            if self.late():
                break
            cands = self.candidates(product)
            self.judge(product, [i for i in cands if i in self.shelf.read_attrs][:20] + [i for i in cands if i not in self.shelf.read_attrs][:10])
            good = self.fits(product)
            if good and product.label and self.spare() > 3 and all(self.shelf.category(i) != product.label for i in good):
                own = [i for i in cands if self.shelf.category(i) == product.label and i not in self.shelf.read_attrs][:4]
                if own:
                    self.read_many(own)
                    self.judge(product, own)
                    good = self.fits(product)
            if self.pick == "cheapest":
                # every cheaper candidate that might fit is read before the cheapest fit is trusted
                for _ in range(4):
                    cap = self.shelf.price(min(good, key=lambda i: self.shelf.price(i) or 1e12)) if good else None
                    cheaper = [i for i in self.maybes(product) if cap is None or (self.shelf.price(i) or 0) < cap]
                    cheaper = [i for i in self.ranked(product, cheaper) if i not in self.shelf.read_attrs][:3]
                    if not cheaper or self.spare() < 3 or self.late():
                        break
                    self.read_many(cheaper)
                    self.judge(product, cheaper)
                    good = self.fits(product)
                if good:
                    good = self.sweep_cheaper(product, good)
                    return min(good, key=lambda i: self.shelf.price(i) or 1e12)
            elif good:
                if self.pick in ("most", "least") and len(good) > 1:
                    return self.extreme(good)
                if len(good) < 2 and self.spare() > 6 and not self.late():
                    self.read_some(product, 3)
                    self.judge(product, [i for i in self.candidates(product) if i in self.shelf.read_attrs][:12])
                    good = self.fits(product) or good
                if self.well_backed(product, good) < (1 if self.refused else 2) and self.spare() > (0 if self.refused else 9) and not self.late():
                    self.more_backing(product)
                    good = self.fits(product) or good
                return good[0]
            # nothing confirmed: read the most promising unread candidates, then search wider
            unread = [i for i in self.maybes(product) if i not in self.shelf.read_attrs][:4]
            if unread and self.spare() > 2:
                self.read_many(unread)
                continue
            if self.spare() < 4:
                break
            labels = sorted({self.shelf.category(i) for i in self.shelf.order if self.shelf.category(i)})[:20]
            got = self.link.ask(QUERIES.replace("PRODUCT", product.name).replace("NEEDS", "; ".join(product.requirements()) or "none")
                                .replace("TRIED", "; ".join(product.tried) or "none").replace("LABELS", ", ".join(labels) or "unknown"), FAST, 300)
            fresh = 0
            for q in [str(x).strip() for x in items_of(got.get("queries")) if str(x).strip()][:2]:
                if q in product.tried or self.spare() < 3:
                    continue
                product.tried.append(q)
                ids = self.search(q)
                if not product.label:
                    product.label = self.label_for(product, ids)
                product.pool.extend(i for i in ids if i not in product.pool)
                fresh += len(ids)
            if not fresh:
                break
            self.read_some(product, 2)
        return self.keep_reading(product)

    def browse_next(self, product: Product) -> Any:
        """The next offset or page of the product's category listing, from how many of its rows were seen."""
        spec = self.kit.map.get("search") or {}
        page_arg = str(spec.get("page_arg") or "")
        schema = self.kit.props(spec.get("tool")).get(page_arg) or {}
        seen = len(self.members(product))
        if schema.get("minimum") == 0 or re.search(r"offset|skip|start|begin|from|row", page_arg, re.I):
            return seen
        return 1 + seen // max(1, len(getattr(self, "last_search", ({}, None, []))[2]) or 10)

    def keep_reading(self, product: Product) -> str | None:
        """Steps are there to be used: read the likeliest unread candidates, a few at a time, until one fits."""
        waiting = sum(1 for p in self.products if p is not product and not p.choice)
        floor = self.reserve() + (self.spare() * waiting) // (waiting + 1)
        for _ in range(6):
            if self.late() or self.steps_left() - floor < 2:
                break
            unread = [i for i in self.candidates(product) if i not in self.shelf.read_attrs and (product.verdicts.get(i) or (0, "maybe"))[1] != "no"][:4]
            if not unread:
                # the rows seen are used up: more rows from the narrowest search, then from the category
                more: list[str] = []
                for query in reversed(product.tried[-3:]):
                    if self.steps_left() - floor < 3:
                        break
                    if getattr(self, "last_search", ({},))[0].get((self.kit.map.get("search") or {}).get("query_arg", "")) != query:
                        more = self.search(query)
                    more = self.next_page() or more
                    if [i for i in more if i not in self.shelf.read_attrs]:
                        break
                if not more and product.label and self.steps_left() - floor >= 3:
                    more = self.search(None, product.label, page=self.browse_next(product))
                product.pool.extend(i for i in more if i not in product.pool)
                self.judge(product, [i for i in more if self.shelf.category(i) == product.label or not product.label][:10])
                unread = [i for i in self.candidates(product) if i not in self.shelf.read_attrs and (product.verdicts.get(i) or (0, "maybe"))[1] != "no"][:4]
                if not unread:
                    continue
            self.read_many(unread)
            self.judge(product, unread)
            good = self.fits(product)
            if good:
                return min(good, key=lambda i: self.shelf.price(i) or 1e12) if self.pick == "cheapest" else good[0]
        return None

    def sweep_cheaper(self, product: Product, good: list[str]) -> list[str]:
        """Under a cheapest pick, when the search takes a price cap: list the product's category below the best fit's price,
        judge what comes back, and repeat until nothing cheaper fits."""
        spec = self.kit.map.get("search") or {}
        if not spec.get("price_cap_arg") and (self.kit.map.get("browse") or {}).get("price_cap_arg"):
            spec = dict(self.kit.map["browse"], category_arg=(self.kit.map["browse"].get("category_arg")))
        if not spec.get("price_cap_arg") or not product.label:
            return good
        good = self.sweep_values(product, good)
        ladder = [0.5, 0.75, 1.0, 1.0]
        for rung in range(4):
            best = min(good, key=lambda i: self.shelf.price(i) or 1e12)
            price = self.shelf.price(best)
            if price is None or self.spare() < 4 or self.late():
                break
            below = round(price * ladder[rung], 2) if ladder[rung] < 1 else price - 0.01
            ids = self.search(None if spec.get("category_arg") else product.name, product.label if spec.get("category_arg") else None, cap=below)
            product.pool.extend(i for i in ids if i not in product.pool)
            fresh = [i for i in self.ranked(product, ids) if (self.shelf.price(i) or 1e12) < price and self.shelf.available(i) and self.shelf.category(i) == product.label]
            if not fresh:
                if ladder[rung] < 1:
                    continue
                break
            self.judge(product, fresh[:12])
            for i in [x for x in self.ranked(product, self.maybes(product)) if (self.shelf.price(x) or 1e12) < price and x not in self.shelf.read_attrs][:6]:
                if self.spare() >= 4:
                    self.read_attrs(i)
            self.judge(product, fresh[:12])
            cheaper = [i for i in self.fits(product) if (self.shelf.price(i) or 1e12) < price]
            if not cheaper:
                if ladder[rung] < 1:
                    continue
                break
            good = self.fits(product)
        self.note(kind="sweep", product=product.number, best=min(good, key=lambda i: self.shelf.price(i) or 1e12))
        return good

    def sweep_values(self, product: Product, good: list[str]) -> list[str]:
        """The value searches again, capped just under the best price: a short list, all of it worth reading."""
        queries = []
        for need in product.requirements():
            if "[unknown" in need or len(need.split(":", 1)[-1]) > 30 or re.search(r"\b(at least|at most|between|not)\b", need, re.I):
                continue
            query = query_of(need)
            if query and query not in queries:
                queries.append(query)
        for query in queries[:2]:
            for _ in range(3):
                best = min(good, key=lambda i: self.shelf.price(i) or 1e12)
                price = self.shelf.price(best)
                if price is None or self.spare() < 3 or self.late():
                    return good
                ids = self.search(query, cap=price - 0.01)
                if self.spare() >= 5 and len(ids) >= 5:
                    ids = ids + [i for i in self.next_page() if i not in ids]
                product.pool.extend(i for i in ids if i not in product.pool)
                fresh = [i for i in self.ranked(product, ids) if (self.shelf.price(i) or 1e12) < price and self.shelf.available(i) and self.shelf.category(i) == product.label]
                if not fresh:
                    break
                self.judge(product, fresh[:12])
                for i in [x for x in self.ranked(product, self.maybes(product)) if (self.shelf.price(x) or 1e12) < price and x not in self.shelf.read_attrs][:5]:
                    if self.spare() >= 3:
                        self.read_attrs(i)
                self.judge(product, fresh[:12])
                cheaper = [i for i in self.fits(product) if (self.shelf.price(i) or 1e12) < price]
                if not cheaper:
                    break
                good = self.fits(product)
        return good

    def extreme(self, ids: list[str]) -> str:
        got = self.link.ask(MEASURE.replace("MEASURE", self.measure or "the measure").replace("RECORDS", "\n".join(self.shelf.card(i) for i in ids)), FAST, 500)
        values = got.get("values") if isinstance(got.get("values"), dict) else {}
        scored = [(number(values.get(i)), i) for i in ids if number(values.get(i)) is not None]
        if not scored:
            return ids[0]
        return (max if self.pick == "most" else min)(scored)[1]

    def trace_parts(self) -> None:
        """A carted line whose text continues on another part: that part's listing is looked up and opened before the order."""
        for p in self.products:
            if not p.choice:
                continue
            for part in self.shelf.links(p.choice)["more"]:
                if part in self.looked or self.steps_left() <= len(self.products) + 1:
                    continue
                self.looked.add(part)
                found = self.search(part, free=True)
                self.note(kind="part_lookup", line=p.choice, part=part, rows=len(found))
                for ident in found:
                    if self.shelf.links(ident)["own"].upper() == part.upper() and self.steps_left() > len(self.products) + 1:
                        self.read_attrs(ident)
                        break

    def pending_parts(self) -> int:
        return sum(1 for p in self.products if p.choice for part in self.shelf.links(p.choice)["more"] if part not in self.looked)

    def covers(self, accessory: str, model: str) -> bool | None:
        """Does the accessory's list name the device model? None when its list continues on a part not seen."""
        seen: set[str] = set()
        queue = [accessory]
        unknown = False
        while queue:
            ident = queue.pop()
            if ident in seen or ident not in self.shelf.recs:
                continue
            seen.add(ident)
            codes = self.shelf.links(ident)
            if self.shelf.names_model(codes, model):
                return True
            for part in codes["more"]:
                found = [i for i in self.shelf.order if self.shelf.links(i)["own"] == part]
                if found:
                    queue.extend(found)
                else:
                    unknown = True
        return None if unknown else False

    def code_set(self, accessory: str) -> set[str]:
        """Every device code an accessory's text lists, continuation parts included."""
        listed = [self.shelf.links(accessory)]
        for part in listed[0]["more"]:
            for ident in self.shelf.order:
                if self.shelf.links(ident)["own"].upper() == part.upper():
                    listed.append(self.shelf.links(ident))
        codes: set[str] = set()
        for one in listed:
            codes |= one["fits"]
        known = {self.shelf.links(i)["model"] for i in self.shelf.order} - {""}
        return codes | {m for m in known if any(self.shelf.names_model(one, m) for one in listed)}

    def move_device(self) -> None:
        """When the chosen accessories share a code and the chosen device lacks it, look for a fitting device with that code."""
        devices = [p for p in self.products if p.choice and self.shelf.links(p.choice)["model"] and not self.shelf.states_fit(self.shelf.links(p.choice))]
        for dev in devices[:1]:
            parts = [p for p in self.products if p is not dev and p.choice and self.shelf.states_fit(self.shelf.links(p.choice))]
            if not parts or self.spare() < 4 or self.late():
                continue
            self.trace_parts()
            shared: set[str] | None = None
            for p in parts:
                shared = self.code_set(p.choice) if shared is None else shared & self.code_set(p.choice)
            if not shared or self.shelf.names_of(self.shelf.links(dev.choice)["model"]) & shared:
                continue
            for code in sorted(shared)[:3]:
                rows = [i for i in self.members(dev) + dev.pool if code in self.shelf.links(i)["names"] and self.shelf.available(i) and self.affordable(dev, i)]
                if not rows and self.spare() >= 4 and not self.codes_unsearchable:
                    found = self.search(code, free=True)
                    if not found:
                        self.codes_unsearchable = True
                    dev.pool.extend(i for i in found if i not in dev.pool)
                    rows = [i for i in found if code in self.shelf.links(i)["names"] and self.shelf.available(i) and self.affordable(dev, i)]
                for i in rows[:4]:
                    if i not in self.shelf.read_attrs and self.spare() >= 3:
                        self.read_attrs(i)
                self.judge(dev, rows[:6])
                good = [i for i in self.fits(dev) if self.shelf.links(i)["names"] & shared]
                if good:
                    self.note(kind="device_moved", code=self.shelf.links(good[0])["model"], id=good[0])
                    dev.choice = good[0]
                    break

    def scan_rows(self, product: Product, model: str) -> list[str]:
        """Rows print fit statements: read further pages of this product's searches until a few rows name the device."""
        def named() -> list[str]:
            return [i for i in product.pool if (not product.label or self.shelf.category(i) == product.label) and self.covers(i, model) and self.shelf.available(i) and self.affordable(product, i)]
        queries = [q for q in reversed(product.tried) if q] or [product.name]
        for query in queries[:3]:
            if len(named()) >= 3 or self.spare() < 9 or self.late():
                break
            more = self.search(query)
            product.pool.extend(i for i in more if i not in product.pool)
            for _ in range(5):
                if len(named()) >= 3 or self.spare() < 9 or self.late():
                    break
                more = self.next_page()
                if not more:
                    break
                product.pool.extend(i for i in more if i not in product.pool)
        if len(named()) < 2 and product.label and self.spare() >= 9 and not self.late():
            more = self.search(category=product.label)
            product.pool.extend(i for i in more if i not in product.pool)
            for _ in range(4):
                if len(named()) >= 3 or self.spare() < 9 or self.late():
                    break
                more = self.next_page()
                if not more:
                    break
                product.pool.extend(i for i in more if i not in product.pool)
        return named()

    def usable(self, product: Product) -> list[str]:
        """Listings that may satisfy this product, confirmed ones first, each group cheapest first."""
        def price(i: str) -> float:
            return self.shelf.price(i) or 1e12
        yes = sorted(self.fits(product), key=price)
        found = yes + sorted((i for i in self.maybes(product) if i not in yes), key=price)
        return [i for i in found if not product.label or self.shelf.category(i) == product.label]

    def code_plan(self, dev: Product, accs: list[Product]) -> tuple[int, float, str, dict[int, str]] | None:
        """The device listing whose code the most accessories can meet, and the accessory each would use."""
        best = None
        seen: set[str] = set()
        confirmed = set(self.fits(dev))
        for ident in self.usable(dev):
            model = self.shelf.links(ident)["model"]
            if not model or model in seen:
                continue
            seen.add(model)
            lines: dict[int, str] = {}
            for acc in accs:
                # a line already holding a code it names stays, so a good earlier pick is not churned
                ready = [i for i in self.usable(acc) if self.covers(i, model) is True and i not in lines.values()]
                if ready:
                    lines[acc.number] = ready[0]
            total = (self.shelf.price(ident) or 0) * dev.qty + sum((self.shelf.price(i) or 0) * next(a.qty for a in accs if a.number == n) for n, i in lines.items())
            within = self.budget is None or self.budget_each or total <= self.budget
            rank = (len(lines), within, ident in confirmed, sum(1 for n, i in lines.items() if i in self.fits(next(a for a in accs if a.number == n))), -total)
            if best is None or rank > best[0]:
                best = (rank, ident, lines)
        return best

    def checked(self, product: Product, rows: list[str]) -> str | None:
        """The first of these rows that proves to meet its needs, opening them a few at a time."""
        def live() -> list[str]:
            return [i for i in rows if self.shelf.available(i) and self.affordable(product, i) and (product.verdicts.get(i) or (0, "maybe"))[1] != "no"]
        for _ in range(5):
            good = [i for i in self.fits(product) if i in rows]
            if good:
                return (min(good, key=lambda i: self.shelf.price(i) or 1e12) if self.pick == "cheapest" else self.by_evidence(product, good)[0])
            unread = sorted([i for i in self.ranked(product, live()) if i not in self.shelf.read_attrs], key=lambda i: -self.looks_right(product, i))[:max(1, self.batch)]
            if not unread or self.spare() < 1 or self.late():
                self.judge(product, live()[:10])
                good = [i for i in self.fits(product) if i in rows]
                return good[0] if good else None
            if self.read_many(unread) == 0:
                self.judge(product, live()[:10])
                good = [i for i in self.fits(product) if i in rows]
                return good[0] if good else None
            self.judge(product, unread)
        good = [i for i in self.fits(product) if i in rows]
        return good[0] if good else None

    def looks_right(self, product: Product, ident: str) -> int:
        """How many of the product's needs the listing's own words already state, whole; a confirmed fit counts as all of them."""
        needs = [plain(n.split(":", 1)[-1]) for n in product.requirements() if "[unknown" not in n]
        needs = [v for v in needs if v and not re.search(r"\b(?:at least|at most|between|not)\b", v)]
        if (product.verdicts.get(ident) or (0, "maybe"))[1] == "yes":
            return len(needs)
        text = " " + plain(" ".join([self.shelf.title(ident)] + [str(t) for t in self.shelf.texts(ident)] + [f"{k} {v}" for k, v in self.shelf.attributes(ident).items()])) + " "
        return sum(1 for v in needs if f" {v} " in text)

    def model_plan(self, dev: Product, accs: list[Product], tried: set[str]) -> tuple[str, list[str], dict[int, list[str]]] | None:
        """The device code the most accessories can meet from the rows seen, the rows carrying it, and each accessory's rows that name it."""
        def open_rows(p: Product) -> list[str]:
            rows = [i for i in self.ranked(p, self.members(p) or p.pool) if self.shelf.available(i) and self.affordable(p, i) and (p.verdicts.get(i) or (0, "maybe"))[1] != "no"]
            return sorted(rows, key=lambda i: -self.looks_right(p, i))
        by_model: dict[str, list[str]] = {}
        for ident in open_rows(dev):
            model = self.shelf.links(ident)["model"]
            if model and model not in tried:
                by_model.setdefault(model, []).append(ident)
        rows = {a.number: open_rows(a) for a in accs}
        best = None
        for model, ids in by_model.items():
            lines = {n: [i for i in found if self.covers(i, model) is True] for n, found in rows.items()}
            covered = sum(1 for v in lines.values() if v)
            strong = sum(1 for a in accs if lines[a.number] and self.looks_right(a, lines[a.number][0]) > 0)
            total = (self.shelf.price(ids[0]) or 0) * dev.qty
            for a in accs:
                if lines[a.number]:
                    total += min(self.shelf.price(i) or 0 for i in lines[a.number]) * a.qty
            within = self.budget is None or self.budget_each or total <= self.budget
            sure = sum(1 for i in ids if i in self.fits(dev))
            rank = (covered, strong, self.looks_right(dev, ids[0]) > 0 or not dev.requirements(), within, sure, -total)
            if best is None or rank > best[0]:
                best = (rank, model, ids, lines)
        return None if best is None else (best[1], best[2], best[3])

    def harvest(self, product: Product, enough: int, limit: int | None = None) -> None:
        """Further pages of this product's narrowest searches until enough affordable rows of its kind are known."""
        def have() -> int:
            rows = [i for i in (self.members(product) or product.pool) if self.shelf.available(i) and self.affordable(product, i)]
            return sum(1 for i in rows if self.looks_right(product, i) > 0) * 3 if product.requirements() else len(rows)
        pages = limit if limit is not None else max(2, int(0.25 * self.link.max_steps / max(1, len(self.products))))
        for query in [q for q in reversed(product.tried) if q][:3]:
            if have() >= enough or self.spare() < 3 or self.late() or pages <= 0:
                return
            more = self.search(query)
            product.pool.extend(i for i in more if i not in product.pool)
            for _ in range(5):
                if have() >= enough or self.spare() < 3 or self.late() or pages <= 0:
                    break
                pages -= 1
                more = self.next_page()
                if not more:
                    break
                product.pool.extend(i for i in more if i not in product.pool)

    def fit_first(self) -> None:
        for dev in self.products:
            accs = [a for a in self.products if a is not dev and a.fits_with == dev.number]
            if not accs or self.late():
                continue
            group = [dev] + accs
            others = sum(1 for p in self.products if p not in group and not p.choice)
            self.held = (max(0, self.spare()) * others) // (others + len(group))
            for at, p in enumerate(group):
                self.relabel(p)
                self.targeted(p)
                self.relabel(p)
                self.harvest(p, 10 if p is dev else 28)
            for p in group:
                room = [i for i in (self.members(p) or p.pool) if self.shelf.available(i) and self.affordable(p, i)]
                self.judge(p, self.ranked(p, room)[:30])
            tried: set[str] = set()
            kept = None
            for _ in range(6):
                plan = self.model_plan(dev, accs, tried)
                if plan is None or self.late():
                    break
                model, ids, lines = plan
                tried.add(model)
                self.note(kind="fit_plan", model=model, devices=ids[:4], lines={k: v[:4] for k, v in lines.items()})
                device = self.checked(dev, ids)
                if not device and all(lines.get(a.number) for a in accs) and self.spare() >= 4:
                    self.harvest(dev, 10 ** 6, 3)
                    more = [i for i in (self.members(dev) or dev.pool) if self.shelf.links(i)["model"] == model and i not in ids]
                    self.judge(dev, more[:12])
                    device = self.checked(dev, more)
                if not device:
                    continue
                chosen: dict[int, str] = {}
                for acc in accs:
                    pick = self.checked(acc, [i for i in lines.get(acc.number) or [] if i not in chosen.values()])
                    if not pick and self.spare() >= 3:
                        found = self.scan_rows(acc, model)
                        self.judge(acc, found[:12])
                        pick = self.checked(acc, [i for i in found if i not in chosen.values()])
                    if pick:
                        chosen[acc.number] = pick
                if kept is None or len(chosen) > len(kept[1]):
                    kept = (device, chosen)
                if len(chosen) == len(accs) or self.spare() < 2:
                    break
            self.held = 0
            if kept:
                dev.choice = kept[0]
                for acc in accs:
                    acc.choice = kept[1].get(acc.number, acc.choice)
                self.fit_settled = len(kept[1]) == len(accs)
                self.note(kind="fit_first", device=kept[0], lines=kept[1], settled=self.fit_settled)

    def joint_fit(self) -> None:
        if self.fit_settled:
            return
        for dev in self.products:
            accs = [a for a in self.products if a is not dev and a.fits_with == dev.number]
            if not accs or self.late():
                continue
            for p in [dev] + accs:
                room = [i for i in (self.members(p) or p.pool) if self.shelf.available(i) and self.affordable(p, i)]
                self.judge(p, self.ranked(p, room)[:30])
            plan = self.code_plan(dev, accs)
            for _ in range(3):
                if plan is None or len(plan[2]) == len(accs) or self.spare() < 9 or self.late():
                    break
                model = self.shelf.links(plan[1])["model"]
                for acc in accs:
                    if acc.number not in plan[2] and self.spare() >= 9:
                        found = self.scan_rows(acc, model)
                        self.judge(acc, found[:12])
                plan = self.code_plan(dev, accs)
            if plan is None or not plan[2]:
                continue
            _, ident, lines = plan
            if dev.choice != ident or any(a.choice != lines.get(a.number, a.choice) for a in accs):
                self.note(kind="joint_fit", device=ident, model=self.shelf.links(ident)["model"], lines=lines, wanted=len(accs))
            dev.choice = ident
            for acc in accs:
                acc.choice = lines.get(acc.number, acc.choice)

    def align(self) -> None:
        """Devices and their accessories chosen together, on the codes the listings print."""
        self.joint_fit()
        self.move_device()
        for acc in self.products:
            dev = self.products[acc.fits_with - 1] if acc.fits_with and 1 <= acc.fits_with <= len(self.products) and acc.fits_with != acc.number else None
            if dev is None and acc.choice:
                own_codes = self.shelf.links(acc.choice)
                if self.shelf.states_fit(own_codes):
                    dev = next((p for p in self.products if p is not acc and p.choice and self.shelf.links(p.choice)["model"] and not self.shelf.states_fit(self.shelf.links(p.choice))), None)
            if dev is None:
                continue
            if not acc.choice or not dev.choice or self.spare() < 2 or self.late():
                continue
            model = self.shelf.links(dev.choice)["model"]
            if not model:
                continue
            verdict = self.covers(acc.choice, model)
            if verdict is None:
                self.trace_parts()
                verdict = self.covers(acc.choice, model)
            if verdict is not False:
                continue
            ready = [i for i in self.ranked(acc, self.fits(acc) + self.maybes(acc)) if self.covers(i, model)]
            if not ready and self.spare() >= 4:
                ids: list[str] = []
                for name in sorted(self.shelf.names_of(model), key=lambda n: n != model):
                    if self.spare() < 4 or self.codes_unsearchable:
                        break
                    found = self.search(name, free=True)
                    ids += found
                    if not found:
                        self.codes_unsearchable = True
                    if [i for i in ids if self.shelf.category(i) == acc.label and self.covers(i, model)]:
                        break
                acc.pool.extend(i for i in ids if i not in acc.pool)
                ids += self.scan_rows(acc, model)
                near = [i for i in ids if self.shelf.category(i) == acc.label and self.covers(i, model) and self.affordable(acc, i)]
                for i in near[:5]:
                    if self.spare() >= 3:
                        self.read_attrs(i)
                self.judge(acc, near[:8])
                ready = [i for i in self.ranked(acc, self.fits(acc) + self.maybes(acc)) if self.covers(i, model)]
            if ready:
                self.note(kind="aligned", product=acc.number, model=model, id=ready[0])
                acc.choice = ready[0]
                continue
            # no accessory for this device: another fitting device the accessory's list does name
            for alt in self.ranked(dev, self.fits(dev)):
                code = self.shelf.links(alt)["model"]
                if code and self.covers(acc.choice, code):
                    self.note(kind="aligned_device", product=dev.number, model=code, id=alt)
                    dev.choice = alt
                    break

    def fit_budget(self) -> None:
        """Keep the total inside the budget by moving the dearest lines to cheaper fits."""
        if self.budget is None or self.budget_each:
            return
        for _ in range(len(self.products)):
            total = sum((self.shelf.price(p.choice) or 0) * p.qty for p in self.products if p.choice)
            if total <= self.budget:
                return
            moves = []
            for p in self.products:
                if not p.choice:
                    continue
                now = self.shelf.price(p.choice) or 0
                dev = self.products[p.fits_with - 1] if p.fits_with and 1 <= p.fits_with <= len(self.products) and p.fits_with != p.number else None
                model = self.shelf.links(dev.choice)["model"] if dev and dev.choice else ""
                for alt in self.fits(p):
                    price = self.shelf.price(alt)
                    if price is not None and price < now and (not model or self.covers(alt, model)):
                        moves.append(((now - price) * p.qty, p, alt))
            if not moves:
                return
            _, p, alt = max(moves, key=lambda m: m[0])
            p.choice = alt

    # ---------------------------------------------------------------- confirm
    def confirm(self, last: bool = False) -> None:
        chosen = [p for p in self.products if p.choice]
        if not chosen or self.late():
            return
        for p in chosen:
            room = self.steps_left() - len(self.products) - 1
            if room > 0 and p.choice not in self.shelf.read_page and self.kit.tool("page") != self.kit.tool("attributes") and (p.choice not in self.shelf.read_attrs or self.spare() > 2):
                self.read_page(p.choice)
                room -= 1
            if room > 0:
                self.read_attrs(p.choice)
            self.redirect_check(p, p.choice)
        self.absorb_notes()
        for p in chosen:
            if p.choice and (not self.shelf.available(p.choice) or (p.verdicts.get(p.choice) or (0, ""))[0] != p.version):
                self.judge(p, [p.choice])
                if not self.shelf.available(p.choice) or p.choice not in self.fits(p):
                    p.choice = self.select(p)
        lines = "\n".join(f"product {p.number} x{p.qty}: " + self.shelf.card(p.choice, 500) for p in self.products if p.choice)
        others = []
        for p in self.products:
            for i in (self.fits(p) + self.maybes(p))[:5]:
                if i != p.choice:
                    others.append(f"(product {p.number}) " + self.shelf.card(i, 160))
        got = self.link.ask(CONFIRM.replace("REQUEST", self.request).replace("SPEC", self.spec_text()).replace("LINES", lines)
                            .replace("OTHERS", "\n".join(others[:12]) or "none"), SHARP if self.link.cost < 0.03 else FAST, 2200, think=True)
        self.note(kind="confirm", got=short(got, 400))
        if got.get("ok") is False:
            swapped = set()
            wrong = [whole(n) for n in items_of(got.get("contradicted"))]
            self.hopeless = bool(last and not items_of(got.get("swap")) and any(
                n and 1 <= n <= len(self.products) and self.products[n - 1].choice not in self.fits(self.products[n - 1]) for n in wrong)
                and (self.messages_left() <= 0 or not self.kit.tool("message")))
            for entry in items_of(got.get("swap")):
                if not isinstance(entry, dict):
                    continue
                n = whole(entry.get("product"))
                ident = str(entry.get("id") or "")
                if n and 1 <= n <= len(self.products) and ident in self.shelf.recs and self.shelf.available(ident):
                    target = self.products[n - 1]
                    dev = self.products[target.fits_with - 1] if target.fits_with and 1 <= target.fits_with <= len(self.products) else None
                    model = self.shelf.links(dev.choice)["model"] if dev and dev.choice else ""
                    if model and self.covers(ident, model) is False:
                        continue
                    if ident not in self.fits(target) and target.choice in self.fits(target):
                        continue
                    target.choice = ident
                    swapped.add(n)
            if not swapped and not last:
                # an objection without a better listing: lines the selection had not confirmed are dropped
                for p in self.products:
                    if p.choice and p.choice not in self.fits(p):
                        p.verdicts[p.choice] = (p.version, "no", "review")
                        p.choice = None

    # ---------------------------------------------------------------- commit
    def set_cart(self, lines: dict[str, int]) -> bool:
        spec = self.kit.map.get("cart") or {}
        props = self.kit.props(spec.get("tool"))
        listed = spec.get("lines_arg") if spec.get("lines_arg") in props else next((k for k, v in props.items() if isinstance(v, dict) and v.get("type") == "array"), None)
        if listed:
            entry = ((props[listed].get("items") or {}).get("properties") or {})
            id_key = spec.get("line_id") if spec.get("line_id") in entry else next((k for k in entry if re.search(r"id|sku|item|product|listing|code", k, re.I) and not re.search(r"qty|quant|count|unit|number", k, re.I)), None)
            qty_key = spec.get("line_qty") if spec.get("line_qty") in entry else next((k for k in entry if re.search(r"qty|quant|count|unit|number|amount", k, re.I)), None)
            if not id_key:
                return False
            body = [{id_key: ident, **({qty_key: qty} if qty_key else {})} for ident, qty in lines.items()]
            if self.call("cart", {listed: body}) is None:
                return False
            self.cart = dict(lines)
            return True
        if not spec or not spec.get("id_arg"):
            return False
        for ident in [i for i in self.cart if i not in lines]:
            args = {spec["id_arg"]: ident}
            if spec.get("op_arg") and spec.get("remove") is not None:
                args[spec["op_arg"]] = spec["remove"]
                if spec.get("qty_arg"):
                    args[spec["qty_arg"]] = self.cart[ident]
                if self.call("cart", args) is not None:
                    self.cart.pop(ident, None)
            elif spec.get("qty_arg"):
                args[spec["qty_arg"]] = 0
                if self.call("cart", args) is not None:
                    self.cart.pop(ident, None)
        for ident, qty in lines.items():
            if self.cart.get(ident) == qty:
                continue
            args = {spec["id_arg"]: ident}
            if spec.get("op_arg") and spec.get("add") is not None:
                args[spec["op_arg"]] = spec["add"]
            if spec.get("qty_arg"):
                args[spec["qty_arg"]] = qty - self.cart.get(ident, 0) if self.cart.get(ident) else qty
            if self.call("cart", args) is None:
                return False
            self.cart[ident] = qty
        return True

    def ask_approval(self, lines: dict[str, int]) -> None:
        if not self.before_order or self.approved or self.messages_left() <= 0:
            return
        total = sum((self.shelf.price(i) or 0) * q for i, q in lines.items())
        summary = "; ".join(f"{q} x {short(self.shelf.title(i), 60)} at {self.shelf.price(i) or '?'}" for i, q in lines.items())
        text = self.link.say(ASK_OK.replace("STEP", "; ".join(self.before_order)).replace("LINES", summary).replace("TOTAL", f"{total:g}"), FAST, 200).strip().strip('"')
        if not text or len(text) > 300 or "\n" in text.strip():
            text = f"I plan to order {summary}, total {total:g}. May I go ahead?"
        answer = self.send(text[:380])
        if answer:
            self.notes_new.append(answer)
            self.absorb_notes()
            self.approved = True

    def buy(self) -> Any:
        spec = self.kit.map.get("buy") or {}
        if not spec:
            return None
        return self.call("buy", dict(spec.get("args") or {}))

    def commit(self) -> bool:
        """Cart the chosen lines and check out; after a refusal apply the change and try again."""
        for round_ in range(4):
            if self.link.done:
                return True
            lines = {p.choice: p.qty for p in self.products if p.choice}
            if len(lines) < len(self.products) or self.steps_left() < 2:
                return False
            if round_ == 0:
                self.ask_approval(lines)
                if any(not p.choice for p in self.products):
                    continue
                lines = {p.choice: p.qty for p in self.products if p.choice}
            self.trace_parts()
            if not self.set_cart(lines):
                return False
            if self.steps_left() < 1:
                return False
            self.buy()
            if self.link.done:
                self.note(kind="ordered", lines=lines, round=round_)
                return True
            # refused: read what changed
            got = self.absorb_notes()
            self.note(kind="refused", round=round_, got=short(got, 300))
            self.refused = True
            if not got.get("sold_out") and not got.get("gone") and self.steps_left() > 3:
                # the page itself says whether a line can still be bought, whatever the note was read to mean
                for ident in list(lines):
                    self.reread(ident)
            if got.get("sold_out") or got.get("gone") or got.get("price_changed"):
                if got.get("sold_out") and len(lines) == 1 and not got.get("price_changed"):
                    self.shelf.gone.update(lines)
                else:
                    for ident in list(lines):
                        if ident in self.shelf.gone or self.steps_left() <= 3:
                            continue
                        self.reread(ident)
                    lost = [i for i in lines if i in self.shelf.gone]
                    if not lost and len(lines) == 1:
                        self.shelf.gone.update(lines)
                for p in self.products:
                    if p.choice and p.choice in self.shelf.gone:
                        self.twins(p, p.choice)
            for p in self.products:
                if p.choice and (not self.shelf.available(p.choice) or not self.affordable(p, p.choice) or p.verdicts.get(p.choice, (0,))[0] != p.version):
                    self.judge(p, [p.choice])
                    if not self.shelf.available(p.choice) or p.choice not in self.fits(p):
                        p.choice = self.select(p) if self.steps_left() > 3 else (next((i for i in self.fits(p) if i != p.choice), None))
            if self.pick == "cheapest" and (got.get("price_changed") or got.get("drop") or got.get("budget")):
                for p in self.products:
                    if self.steps_left() > 3:
                        p.choice = self.select(p) or p.choice
                    good = self.fits(p)
                    if good:
                        p.choice = min(good, key=lambda i: self.shelf.price(i) or 1e12)
            self.fit_budget()
        return self.link.done

    def twins(self, product: Product, lost: str) -> None:
        """The same goods are often listed more than once: a sold-out listing's twins come up on a search for its title."""
        title = self.shelf.title(lost)
        words = re.findall(r"[^\W_]{2,}", title)
        if len(words) < 2 or self.spare() < 1 or self.late():
            return
        mine = roots(title)
        price = self.shelf.price(lost)

        def closeness(ident: str) -> tuple:
            same_title = plain(self.shelf.title(ident)) == plain(title)
            same_price = price is not None and self.shelf.price(ident) == price
            return (not same_title, not same_price, -len(mine & roots(self.shelf.title(ident))))

        def alike(ids: list[str]) -> list[str]:
            good = [i for i in ids if i != lost and self.shelf.available(i) and self.affordable(product, i)
                    and len(mine & roots(self.shelf.title(i))) >= 0.7 * max(1, len(mine))]
            return sorted(good, key=closeness)

        ids = self.search(" ".join(words[:6]), cap=price)
        if not [i for i in ids if i != lost] and len(words) > 3 and self.spare() >= 1:
            ids = self.search(" ".join(words[:3]), cap=price)
        product.pool.extend(i for i in ids if i not in product.pool)
        for _ in range(2):
            if not ids or any(closeness(i)[0] is False for i in alike(ids)) or self.spare() < 2:
                break
            more = self.next_page()
            if not more:
                break
            ids = ids + [i for i in more if i not in ids]
            product.pool.extend(i for i in more if i not in product.pool)
        found = alike(ids)
        self.note(kind="twins", lost=lost, found=found[:6])
        fresh = [i for i in found if i not in self.shelf.read_attrs][:3]
        if fresh:
            self.read_many(fresh)
        if found:
            self.judge(product, found[:6])

    # ---------------------------------------------------------------- decline
    def audit(self, product: Product) -> str | None:
        ids = self.candidates(product)[:14]
        if not ids:
            return None
        got = self.link.ask(AUDIT.replace("PRODUCT", product.name).replace("NEEDS", "; ".join(product.requirements()) or "none")
                            .replace("BUDGET", "none" if self.budget is None else f"{self.budget:g}")
                            .replace("RECORDS", "\n".join(self.shelf.card(i, 300) for i in ids)), SHARP if self.link.cost < 0.02 else FAST, 2200, think=True)
        self.note(kind="audit", product=product.number, got=short(got, 300))
        for ident in items_of(got.get("fits")):
            if str(ident) in ids:
                product.verdicts[str(ident)] = (product.version, "yes", "audit")
                return str(ident)
        for ident in items_of(got.get("promising"))[:3]:
            if str(ident) in ids and self.spare() >= (1 if self.refused else 3) and self.read_attrs(str(ident)):
                self.judge(product, [str(ident)])
        good = self.fits(product)
        return good[0] if good else None

    def holds_nothing(self, product: Product) -> bool:
        """Does a search on all of this product's firm plain values together come back empty?"""
        values = []
        for need in product.requirements():
            value = re.sub(r"\(.*?\)", " ", need.split(":", 1)[-1]).strip()
            if value and "[unknown" not in need and not re.search(r"\d|\b(?:at least|at most|between|not|except|same|like|or)\b", value, re.I) and len(value.split()) <= 3:
                values.append(value)
        if len(values) < 2 or self.spare() < 1:
            return False
        return not self.search(" ".join(" ".join(values).split()[:6]))

    def decline(self) -> None:
        spec = self.kit.map.get("decline") or {}
        if not spec or self.link.done:
            return
        args = dict(spec.get("args") or {})
        if spec.get("text_arg"):
            args[spec["text_arg"]] = SORRY
        self.call("decline", args)
        self.note(kind="declined", finished=self.link.done)

    # ---------------------------------------------------------------- the episode
    def run(self) -> list[dict[str, Any]]:
        if self.link.done:
            return self.trace
        self.read()
        self.kit.learn(self.link)
        self.note(kind="tools", map=short(self.kit.map, 600))
        if self.opening is not None:
            _, notes = self.shelf.take(self.link.printed.read(self.opening))
            self.notes_new.extend(notes)
        self.scout()
        self.absorb_notes()
        self.sweep()
        self.chase_all()
        self.fit_first()
        for attempt in range(4):
            if self.link.done:
                break
            for at, p in enumerate(self.products):
                if not p.choice or not self.shelf.available(p.choice) or p.choice not in self.fits(p):
                    waiting = sum(1 for q in self.products[at + 1:] if not q.choice)
                    self.held = (max(0, self.spare()) * waiting) // (waiting + 1)
                    p.choice = self.select(p) if not self.late() else (self.fits(p) or [None])[0]
                    self.held = 0
            missing = [p for p in self.products if not p.choice]
            for p in missing:
                if not self.late():
                    p.choice = self.audit(p)
            missing = [p for p in self.products if not p.choice]
            if missing and attempt == 0 and len(self.products) == 1 and self.messages_left() <= 0 and any(self.holds_nothing(p) for p in missing):
                self.note(kind="nothing_holds_the_values", products=[p.number for p in missing])
                self.decline()
                if self.link.done:
                    break
                self.absorb_notes()
            if missing:
                # declines are rarely right: a product with a plausible (unconfirmed) listing gets that listing
                for p in missing:
                    room = [i for i in self.members(p) or p.pool if self.shelf.available(i) and self.affordable(p, i)]
                    p.choice = (self.by_evidence(p, self.ranked(p, self.maybes(p))) or self.by_evidence(p, self.ranked(p, room)) or [None])[0]
                    if p.choice:
                        self.note(kind="unconfirmed_line", product=p.number, id=p.choice)
                if any(not p.choice for p in self.products):
                    # nothing at all for a product: one last plain search before giving up on the order
                    for p in self.products:
                        if not p.choice and self.spare() >= 1:
                            ids = self.search(" ".join(p.name.split()[:3]))
                            p.pool.extend(i for i in ids if i not in p.pool)
                            p.choice = (self.ranked(p, [i for i in p.pool if self.shelf.available(i)]) or [None])[0]
                    if any(not p.choice for p in self.products):
                        self.decline()
                        if self.link.done:
                            break
                        got = self.absorb_notes()
                        self.note(kind="decline_refused", got=short(got, 300))
                    continue
            self.align()
            self.fit_budget()
            self.hopeless = False
            self.confirm(last=attempt >= 2 or self.spare() < 5)
            if self.hopeless and len(self.products) > 1:
                self.hopeless = False
            if self.hopeless and attempt < 3 and self.steps_left() > len(self.products) + 3:
                found = False
                for p in self.products:
                    if p.choice and p.choice not in self.fits(p):
                        p.verdicts[p.choice] = (p.version, "no", "review")
                        p.choice = self.keep_reading(p)
                        found = found or bool(p.choice)
                self.note(kind="kept_looking", found=found)
                if found:
                    self.hopeless = False
                    if all(p.choice for p in self.products) and self.commit():
                        break
                    continue
            if self.hopeless:
                self.note(kind="nothing_fits")
                self.decline()
                if self.link.done:
                    break
                self.absorb_notes()
                continue
            if any(not p.choice for p in self.products):
                continue
            if self.commit():
                break
        if not self.link.done:
            lines = {p.choice: p.qty for p in self.products if p.choice}
            if len(lines) == len(self.products) and self.steps_left() >= 1:
                if self.set_cart(lines):
                    self.buy()
            if not self.link.done:
                self.decline()
        self.note(kind="end", finished=self.link.done, steps=self.link.step, cost=round(self.link.cost, 5), calls=self.link.calls, seconds=round(self.link.age(), 1))
        return self.trace


def _salvage(episode: Hotaru) -> None:
    """After an unexpected error: order a complete selection if there is one, else end the episode cleanly."""
    if episode.link.done:
        return
    lines = {p.choice: p.qty for p in episode.products if p.choice}
    if lines and len(lines) == len(episode.products) and episode.set_cart(lines):
        episode.buy()
    if not episode.link.done:
        episode.decline()


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    episode: Hotaru | None = None
    try:
        episode = Hotaru(problem_data)
        return episode.run()
    except Exception as problem:  # never let an error leave the episode open
        log = episode.trace if episode is not None else []
        log.append({"kind": "error", "detail": f"{type(problem).__name__}: {problem}"[:300]})
        if episode is not None:
            try:
                _salvage(episode)
            except Exception as second:
                log.append({"kind": "error", "detail": f"salvage: {type(second).__name__}: {second}"[:300]})
        return log


__all__ = ["agent_main"]
