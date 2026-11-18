"""kohaku: a blackboard shopper for ORO Bench worlds.

Code keeps a board (work order, listings seen, needs, notes, cart). The opening is run by code: a look at every
product, a harvest of attribute keys from opened listings, then the interview, before the first planning round.
Each round after that, a model reads the whole board and returns a short batch of macro-actions; code runs them,
enforces the hard rules and renders the board again.

Interview: a message is answered about the first subject of the shopper's it names, however long the message is,
and a subject the shopper is indifferent to hides any named after it. The wants have to be drawn out before ordering,
so every product gets one long question of its own, ranked by a model, and the messages left go back to
the products that gave nothing, in a turned order (reversed, then centre-out). Entry point: agent_main(problem_data).
"""
from __future__ import annotations
import json
import re
import time
from typing import Any
from src.agent.proxy_client import ProxyClient

QUICK = "anthropic/claude-haiku-5.5"
DEEP = "anthropic/claude-sonnet-5.5"
OTHER = "openai/gpt-6-luna"

CLOCK_TOTAL = 300.0
CLOCK_SETTLE = 215.0      # after this the board loop stops and the order is placed with what is chosen
CLOCK_FINAL = 262.0       # after this no model call starts
SPEND_DEEP_UNTIL = 0.045  # USD; past it the deep model is not used any more
SPEND_STOP = 0.075        # USD; past it no model call is made
ROUNDS_MAX = 16
ACTIONS_PER_ROUND = 6
OPEN_PER_ROUND = 5
HOPS_MAX = 6
HOP_STEPS = 22
BOARD_ROWS = 14
FACT_CHARS = 230
TEXT_CHARS = 170

_store = ProxyClient(timeout=90, max_retries=3, retry_delay=1)
_models = ProxyClient(timeout=55, max_retries=1, retry_delay=1)

# ----------------------------------------------------------------------------------------------------------------
# prompts
# ----------------------------------------------------------------------------------------------------------------
WORK_ORDER = """Read a shopper's request to an online-store buying agent and write the work order as ONE JSON object (no prose).
Keys:
"products": one entry per distinct product to buy: {"name": the product in the request's words, "qty": units of one listing (1 unless stated; a pair/set sold as one listing is 1), "needs": firm conditions on this product as "property: value" strings with the exact value, limit, exclusion or list (never price, stock, quantity), "lookups": needs whose value must be read off another listing the request describes ("colour: the same as the rug a named shop sells in beige", "plug: the type that belongs with the one on a named shop's cable page") copied in full, from the words saying how the value relates to that listing to the end of its description, "fits_with": the number (1-based) of another requested product this one must work with, fit, or be compatible with, else null}.
A product named in one phrase with one quantity is ONE product. A condition sits in "needs" only when the request states it as a fact about the product (a use the product must be made for counts; the person or room it is for does not). Words that only say what kind of product it is (the store category's own name, such as "filter coffee, the ground kind" or "garden furniture") belong in "name", not in "needs". Words that describe the other listing in a lookup stay inside that lookup, never as separate needs. A word narrowing the kind of product is a need ("a linen tablecloth": name "tablecloth", need "material: linen").
"budget": {"amount": number or null, "unit": "cents"/"dollars"/..., "scope": "total" or "each"} (keep the request's own unit: 2230 cents stays 2230, never 22.30).
"pick": "cheapest" when the lowest price or total decides, "most"/"least" with "measure" when an extreme of a measure decides, else "any".
"messages": how many messages the request allows, or null. "steps": how many steps or moves it allows, or null.
"procedure": steps the request demands besides finding and buying: [{"what": at most 14 words, "stage": "before_choice" / "before_order" / "before_decline", "needs_message": true when it is done by messaging the shopper}]. Requirements, the budget, stock, and how to end when nothing fits are not procedure.
"ask_first": properties the request says to ask the shopper about before choosing, else [].
"decline_rule": how the request says to end when nothing fits, in at most 15 words, or "" when it says nothing.
"story_hints": for each product, the properties the story points at without giving their value (someone whose taste decides the look, a setting the product goes in, what it is used for, what matters to the person using it): [{"product": number, "property": plain property name, "from": up to 8 words of the request}], [] when none.
"""

TOOL_MAP = """Here are the tools of an online store, as JSON schemas. Map them to jobs. Answer with ONE JSON object:
{"search": {"tool": name of the keyword search tool, "query_arg": its free-text argument, "mode_arg": an argument that switches between keyword and filter search or null, "mode_keyword": its value for keyword search or null, "mode_filter": its value for filter search or null, "category_arg": argument that filters by category/department/section or null, "brand_arg": argument for brand/maker/manufacturer or null, "cap_arg": argument for a maximum price or null, "page_arg": argument for the page number, row offset or continuation token or null, "page_kind": "index" (1-based page numbers) / "offset" (row offset) / "token" (a value copied from the previous result) / null, "filter_tool": a different tool used for category/filter search, or null when the same tool does it},
 "open": {"tool": tool showing one listing's attributes/properties/specifications, "id_arg": its listing-id argument}, "page": {"tool": tool showing one listing's full page when it is a different tool, "id_arg": ...} or null,
 "cart": {"tool": the cart-changing tool, "lines_arg": argument taking a list of line objects or null, "line_id_key": key for the listing id inside a line object or null, "line_qty_key": key for the quantity inside a line object or null, "id_arg": single listing id argument or null, "qty_arg": quantity argument or null, "mode_arg": add/remove switch argument or null, "mode_add": value that adds, "mode_remove": value that removes},
 "cart_view": tool that shows the cart or null,
 "ask": {"tool": the tool that messages the shopper, "text_arg": its text argument},
 "buy": {"tool": the tool that places the order, "args": the fixed arguments that make it place the order (for a tool with a mode argument, that mode), else {}},
 "decline": {"tool": the tool that ends without a purchase, "args": the fixed arguments that make it do so, "text_arg": an argument taking a remark or reason, or null}}
Use exact tool and argument names. A tool may fill two jobs (buy and decline) with different arguments."""

ROUND = """You plan the next moves of a buying agent in an online store. The board below holds everything known: the shopper's request, the work order, the listings seen (rows from searches; opened listings show their attributes), what was asked and answered, notes the store or shopper sent, the cart, and the budget of steps and messages. Every store call costs one step. The job ends with ONE order holding every requested product at its quantity, each listing meeting every need in force, inside the budget; ending without an order earns nothing, so an imperfect order beats no order.

How to read listings: a need "property: value" is met when the listing's attribute for that property (any key spelling, synonyms) states the value or a plain synonym of it ("quick" for "fast", "second hand" for "pre-owned") - one element of a list is enough, a value containing the wanted words counts, units convert, limits include the edge; when a differently named key disagrees, the key that names the property wins, and when several keys name it, one stating the value is enough. A listing whose title or description plainly gives another value of that property does not meet the need, whatever a key says. With no attribute for the property, the title or description may state it; an unlabelled number proves nothing. The category label must be the requested kind of product, and a need that the category label itself states (a kind, a season, an audience in the label's words) is met by every listing of that category. "not X" is met unless the listing shows X. A listing marked sold out or over budget cannot be bought. A listing shown as "opened, judged no" is settled: do not open or choose it again.
The pick rule "cheapest" means the lowest-priced listing that meets every need, among ALL fitting listings of that product, so cheaper unopened rows must be opened or listed below the best price before ordering.
Fitment: when a product must work with another requested product, the chosen accessory's compatibility list (codes parsed from its text, shown as "fits:") must name the device's model code (shown as "model:"); a reference to another part means the rest of the list is on that part's listing. Choose them together: codes show on rows, so when no accessory that meets its needs fits the chosen device, switch the device to one that meets its needs and whose code the accessories list, instead of hunting further for that accessory.

Return ONE JSON object and nothing else:
{"verdicts": [{"id": listing id, "product": product number, "fit": "yes" / "no" / "maybe", "why": at most 12 words}] for opened listings whose verdict is new or changed ("yes" only when every need in force is seen met in its attributes),
 "actions": up to ACTIONS of these, in the order to run them:
   {"do": "list", "product": n, "page": p}  - a page of the product's category (page 1 first; use when few rows of that kind are known or the pick rule needs more candidates)
   {"do": "search", "product": n, "words": "..."}  - keyword search, short and value-first (the wanted value plus the product noun); the engine reads only the first few words
   {"do": "cheaper", "product": n, "below": price}  - list the product's category with a price cap below the best fitting price (only when the board says the search takes a cap)
   {"do": "open", "ids": [...]}  - open listings to read their attributes (unopened rows that could fit; at most OPEN per round; never a listing already opened)
   {"do": "interview"}  - spend the shopper messages on questions about unstated preferences (do this once, after a few listings of each product are opened, before choosing)
   {"do": "choose", "product": n, "id": "..."}  - name the listing to buy for a product (a "yes" listing; a "maybe" only when nothing better can be found with the steps left)
   {"do": "order"}  - place the order with the chosen listings (every product must be chosen; chains and procedure are handled for you)
 "note": at most 25 words on the plan}
Do not repeat a search or listing already in the search log. Stores often sell out the first listing ordered, so when steps allow have a second fitting listing judged before you order. When steps are short, choose and order."""

ASK_PLAN = """A buying agent will ask a shopper one long question per product. The shopper answers about the first property in the question that they have an opinion on, so the list must be complete and the likeliest properties should come first. The shopper keeps one or two firm, unstated wants on single properties.
For each product, list the properties a question should name, best first:
1. what the request's story points at without giving its value (whose taste decides, where it will be used, what matters to the user);
2. the properties that define the variant a buyer chooses between: type, style, cut, construction, material, form, intended user, and the numeric specs of that kind of product;
3. every other attribute key shown below that a buyer could care about, plus any property usual for that kind of product that the keys miss.
Leave out what the request already fixes for that product, identifiers, options, model, brand, seller, price, stock and warranty. Merge keys that mean the same. Name each property in two or three plain words.
Return ONE JSON object and nothing else: {"lists": [{"product": number, "properties": ["...", up to 22]}]}."""

ANSWER = """A buying agent asked a shopper about one property of one product. Read the answer and return ONE JSON object and nothing else:
{"kind": "value" when the answer states a firm value, limit or exclusion for a property; "pointer" only when the answer names a specific listing, page, shop or seller in the store to read the value from, whether it must match that listing, differ from it, belong with it or stay close to it; without one the kind is "value" whatever comparison words it uses, and a comparison with anything outside the store (a family story, a habit, a saying) is only colour that stays out of the value; "none" when the shopper has no preference or is puzzled by the question; "split" when the shopper only asks for fewer things or one thing at a time; "other" for anything else. A want can be put in colourful words: refusing everything except a value states that value, so the value is what is left once the refusal is taken away. Invented examples: "Any width other than 30 cm is a no from me" is kind "value", value "30 cm"; "I'm firm on a frame matching steel, the way my father always did it" is kind "value", value "steel"; "Any finish but the one on the lamp on the Acme page, tall model, is a no from me" is kind "pointer", value "the one on the lamp on the Acme page, tall model",
 "property": the property the answer itself names (it may differ from the ones asked),
 "product": number of the product it is about when the question listed several, else null,
 "value": for "value": the value alone with its unit or limit words, exactly as a listing could print it, without colour, refusal or framing words ("at least 30 cm", "a folding model", "not plastic", and "steel" from "matching steel, no arguing"); for "pointer": every word from those saying how the value relates to the other listing through that listing's whole description, including any instruction about following it further, copied exactly, leaving out any refusal before them,
 "budget": a new spending limit stated in the answer as a number, or null}"""

NOTE = """A store's reply to a buying agent carried these lines (from the shopper or the store). Decide what they change. Return ONE JSON object and nothing else:
{"budget": new total spending limit as a number in the request's unit, or null, "drop": [{"product": number, "property": requirement the shopper gives up}], "add": [{"product": number, "need": "property: value"}], "sold_out": [listing ids said to be unavailable], "chosen_gone": true when the lines say the agent's pick, suggestion or recommended item is gone, sold out, unavailable or cannot be ordered (even without naming it), "price_changed": true/false, "refused": true when the lines say an order or ending was not accepted, "summary": at most 15 words}
A requirement replaced is one drop plus one add. Report every change."""

POINTER = """A buying agent must read a property's value from another listing in the store. Parse the requirement below. A refusal of everything other than the relation is no part of it: the relation is what the shopper is left demanding. Return ONE JSON object and nothing else:
{"property": the property whose value is wanted, plain words, "relation": "same" (the same value as on that page, also when the requirement only points at the page), "unlike" (any value but that one, never that one), "pair" (a value made to suit that page's value under a store rule, not the identical value, even after a word of agreement; also when the sentence asks for whatever belongs or goes with that value), "near" (within a distance of it), "above" (that value or higher), "below" (that value or lower), "distance": the number for "near", else null, "seller": the seller or shop named, or "", "brand": the brand named, or "", "kind": the kind of product of that listing, "traits": ["property value" phrases that pick the listing out, e.g. "height 40 cm", "finish matte"], "plain": when the requirement names no other listing, shop or page, the wanted value alone as a listing could print it, else null}
"property" is what the sentence asks of the shopper's own product, named before the relation words; what describes the other listing goes in "kind" and "traits" only. Invented example: "my shelf length has to suit whatever Acme's lamp with a height of 40 cm and a matte finish has" is property "length", relation "pair", seller "Acme", kind "lamp", traits ["height 40 cm", "finish matte"]."""

PAGE = """A buying agent opened a listing while tracing where the value of PROPERTY must be read. Read the record and decide what the page says about that property. Return ONE JSON object and nothing else:
{"what": "redirect" when the page sends the reader to another listing for this property (the sentence names the property or a plain synonym); "value" when the page states the listing's own value of the property; "unknown" otherwise,
 "value": the listing's own value of PROPERTY as stated (attributes first, else the text), or "",
 "pairs": {"a": "b", ...} when the page states a store rule matching values of the property to partner values, each left value mapped to its right value, or says this listing's own value goes with another value (then a is its own value and b the other), else {},
 "next": for a redirect: {"seller": the seller/shop named, or "", "brand": "", "kind": kind of product, "traits": [every phrase of that sentence describing the other listing, words before its kind included ("used", "gray", "organic 750 ml")]}, else null}
A list of models or codes the product fits is NOT a redirect. A sentence that a version exists elsewhere, or that this is not some other version, says nothing about this listing. A remark that only mentions other versions, listings or parts, without sending the reader there for this property, is NOT a redirect. A redirect about a different property is not one either."""

REVIEW = """Final review before a buying agent places an order. Decide from the records whether the cart is right: one line per requested product at the requested quantity, each listing's attributes meeting every need in force (reading rules: the attribute naming the property decides, one of several such attributes stating the value is enough, a title or description plainly giving another value of the property breaks the need, any key spelling, one list element is enough, a value containing the wanted words counts, units convert, limits include the edge; with no such attribute the title or description counts; "not X" holds unless X is shown; a pointer need already resolved to a value is just that value), nothing sold out, the total inside the budget, and under the "cheapest" rule no listed cheaper listing of the same product that clearly fits.
Return ONE JSON object and nothing else: {"ok": true/false, "problems": [at most 4 short problems], "swap": [{"product": number, "id": a listing shown that should replace the line}], "contradicted": [numbers of the products whose listing's record states a value that breaks a need; a record that is merely silent does not count]}. Use ok=false only for a problem the records show."""

STEP_MESSAGE = """Write ONE short plain sentence (under 25 words) that a buying agent sends to the shopper to do this step: STEP. The planned order is: LINES. Output only the sentence."""

NOTHING_FITS = "Nothing in the store meets every one of your requirements, so I am not placing an order."


# ----------------------------------------------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------------------------------------------
def _dump(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        return str(value)


def _json(text: Any) -> dict[str, Any] | None:
    if not isinstance(text, str):
        return None
    body = text.strip()
    fence = re.match(r"^```[A-Za-z]*\s*(.*?)\s*```$", body, re.S)
    if fence:
        body = fence.group(1)
    for candidate in (body, body[body.find("{"): body.rfind("}") + 1] if "{" in body else ""):
        if not candidate:
            continue
        try:
            got = json.loads(candidate)
            if isinstance(got, dict):
                return got
        except ValueError:
            continue
    return None


def _norm(text: Any) -> str:
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(text or ""))
    return re.sub(r"[\s_\-./]+", " ", text).strip().lower()


def _tokens(text: Any) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", _norm(text)) if len(w) >= 2}


def _singular(word: str) -> str:
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("es") and word[-3] in "sxz":
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _stems(text: Any) -> set[str]:
    return {_singular(w) for w in _tokens(text) if w not in ("the", "and", "for", "with", "of", "a", "an")}


def _num(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        m = re.search(r"-?\d+(?:[.,]\d+)?", value.replace(",", ""))
        if m:
            try:
                return float(m.group(0))
            except ValueError:
                return None
    return None


def _count(value: Any) -> int | None:
    n = _num(value)
    return int(n) if n is not None and float(n).is_integer() else None


def _clip(text: Any, limit: int) -> str:
    s = str(text if text is not None else "")
    s = re.sub(r"\s+", " ", s)
    return s if len(s) <= limit else s[: limit - 1] + "…"


def _kinds(schema: Any) -> list[str]:
    if not isinstance(schema, dict):
        return []
    t = schema.get("type")
    return [t] if isinstance(t, str) else [k for k in (t or []) if isinstance(k, str)]


def _coerce(args: dict[str, Any], schema: Any) -> tuple[dict[str, Any], list[str]]:
    """Shape arguments to a JSON schema: cast types, drop unknown keys, report missing required ones."""
    faults: list[str] = []
    if not isinstance(schema, dict):
        return dict(args), faults
    props = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
    out: dict[str, Any] = {}
    for key, value in args.items():
        if key not in props:
            if schema.get("additionalProperties") is False:
                continue
            out[key] = value
            continue
        spec = props[key] if isinstance(props[key], dict) else {}
        kinds = _kinds(spec)
        if value is None:
            continue
        if "integer" in kinds and not isinstance(value, bool):
            n = _num(value)
            if n is not None:
                value = int(n)
        elif "number" in kinds and not isinstance(value, bool):
            n = _num(value)
            if n is not None:
                value = n
        elif "string" in kinds and not isinstance(value, str):
            value = str(value)
        elif "boolean" in kinds and isinstance(value, str):
            value = value.strip().lower() == "true"
        elif "array" in kinds and not isinstance(value, list):
            value = [value]
        options = spec.get("enum")
        if isinstance(options, list) and value not in options:
            match = next((o for o in options if isinstance(o, str) and isinstance(value, str) and o.lower() == value.lower()), None)
            if match is None:
                faults.append(f"{key}: not one of {options}")
                continue
            value = match
        if isinstance(value, str) and isinstance(spec.get("maxLength"), int):
            value = value[: spec["maxLength"]]
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if isinstance(spec.get("minimum"), (int, float)) and value < spec["minimum"]:
                value = spec["minimum"]
            if isinstance(spec.get("maximum"), (int, float)) and value > spec["maximum"]:
                value = spec["maximum"]
        if isinstance(value, list) and isinstance(spec.get("maxItems"), int):
            value = value[: spec["maxItems"]]
        out[key] = value
    for key in schema.get("required") or []:
        if key not in out:
            faults.append(f"{key} is required")
    return out, faults


CODE = re.compile(r"\b([A-Z]{1,5}-?\d{3,6}[A-Z]?)\b")


def _codes(text: str) -> list[str]:
    return [c.replace("-", "").upper() for c in CODE.findall(text or "")]


def _names(text: str) -> list[str]:
    """Device names in a piece of text: printed codes, and a maker's word plus a number."""
    return _codes(text) + [w.replace(" ", "").upper() for w in re.findall(r"\b[A-Z][a-z]+ \d{3,6}\b", text or "")]


def _fitment(title: str, text: str) -> dict[str, Any]:
    """Parse model and compatibility codes from a listing's text (generic patterns, no world-specific words)."""
    own = re.search(r"\[\s*([A-Z]{1,5}-?\d{3,6})\s*\]", title or "") or re.search(r"\bpart\s+(?:no\.?|number|#)?\s*([A-Z]{1,5}-?\d{3,6})\b", title or "", re.I)
    own_code = own.group(1).replace("-", "").upper() if own else ""
    models = _codes(" ".join(m.group(1) for m in re.finditer(r"\bmodel\s*(?:ref\.?|no\.?|number|#|:)?\s*[:.]?\s*([A-Z]{1,5}-?\d{3,6}[A-Z]?)", text or "", re.I)))
    fits: set[str] = set()
    families: list[tuple[str, set[str]]] = []
    more: list[str] = []
    for m in re.finditer(r"\b(?:with|for|(fits|suits))\b\s*:?\s*([^.;\n]*)", text or "", re.I):
        body = m.group(2)
        fam = re.search(r"every\s+([A-Z]{1,5}-?\d{1,4})x+\s*model(?:s)?(?:\s*,?\s*(?:except|but not|other than)\s+([^.;\n]*))?", body, re.I)
        if fam:
            families.append((fam.group(1).replace("-", "").upper(), set(_codes(fam.group(2) or ""))))
        else:
            fits.update(_codes(body))
    for m in re.finditer(r"\b(?:fits|suits)\b\s*:?\s*([^.;\n]*)", text or "", re.I):
        fits.update(w.replace(" ", "").upper() for w in re.findall(r"\b[A-Z][a-z]+ \d{3,6}\b", m.group(1)))
    for m in re.finditer(r"\bpart\s+[A-Z]{1,5}-?\d{3,6}\s+also\s+fits\s*:?\s*([^.;\n]*)", text or "", re.I):
        fits.difference_update(_names(m.group(1)))   # what another part fits says nothing about this one
    for m in re.finditer(r"\bpart\s+([A-Z]{1,5}-?\d{3,6})\s+also\s+fits\s*:?\s*([^.;\n]*)", text or "", re.I):
        if own_code and m.group(1).replace("-", "").upper() == own_code:
            fits.update(_names(m.group(2)))
    for m in re.finditer(r"\bpart\s+([A-Z]{1,5}-?\d{3,6})\b(?!\s+also)", text or "", re.I):
        code = m.group(1).replace("-", "").upper()
        if code != own_code:
            more.append(code)
    fits.difference_update(more)
    fits.difference_update(_codes(title or ""))
    aliases: set[str] = set()
    if models:
        for m in re.finditer(r"\balso\b(?!\s+fits)([^.;\n]*)", text or "", re.I):
            aliases.update(_names(m.group(1)))
    return {"own": own_code, "models": models, "aliases": aliases, "fits": fits, "families": families, "more": list(dict.fromkeys(more))}


# ----------------------------------------------------------------------------------------------------------------
# the store side: listings seen, fields learnt from the data
# ----------------------------------------------------------------------------------------------------------------
FIELD = {
    "price": re.compile(r"price|cost|amount|fee", re.I),
    "stock": re.compile(r"stock|avail|orderable|purchas|buyable|in_?stock|on_?hand", re.I),
    "category": re.compile(r"categor|section|group|depart|shelf|aisle|(^|_)path$|kind|type_of_product", re.I),
    "seller": re.compile(r"seller|shop|vendor|store|merchant|sold_by|retailer", re.I),
    "brand": re.compile(r"brand|maker|manufactur|label", re.I),
    "title": re.compile(r"title|name|heading|headline|caption", re.I),
    "remarks": re.compile(r"remark|note|fine|print|comment|footnote|small", re.I),
    "description": re.compile(r"desc|about|summary|blurb|overview|detail", re.I),
    "id": re.compile(r"id|sku|code|ref|key", re.I),
}


LIMIT_WORDS = ("not", "about", "at", "least", "most", "between", "under", "over", "up", "to", "or", "and", "by", "than", "more", "less", "above", "below", "higher", "lower")


def _need_words(need: str) -> str:
    """Words to search a need by. Every word must match, so a figure goes without the word right after it, and a short adjective goes with its property."""
    name, _, value = need.partition(":")
    pieces = [t.strip(",.;") for t in re.sub(r"\(.*?\)", " ", value).split()]
    numbers, others, after_number = [], [], False
    for t in pieces:
        if re.fullmatch(r"\d+(?:\.\d+)?", t):
            numbers.append(t)
            after_number = True
            continue
        if not after_number and t.lower() not in LIMIT_WORDS:
            others.append(t)
        after_number = False
    if numbers and not others:
        return " ".join(numbers[:2])
    kept = [t for t in pieces if t.lower() not in ("not", "about", "at", "least", "most", "between")]
    if len(kept) == 1 and len(kept[0]) <= 3 and kept[0].isalpha():
        kept += name.split()[:2]
    return " ".join(kept[:4])


class TextRecords:
    """Turns records a store prints as text into objects: "key: value | key: value" rows, with "a=b; c=d" inside a value."""

    NAME = re.compile(r"\s*([A-Za-z][A-Za-z0-9_ \-]{0,40}?)\s*[:=]\s*(.*)", re.S)

    def __init__(self) -> None:
        self.known: set[str] = set()

    @staticmethod
    def scalar(text: str) -> Any:
        text = text.strip()
        low = text.lower()
        if low in ("", "none", "null"):
            return None
        if low in ("true", "false"):
            return low == "true"
        if re.fullmatch(r"-?\d+(?:\.\d+)?", text):
            return float(text) if "." in text else int(text)
        return text

    @staticmethod
    def pairs(text: str) -> dict[str, str] | None:
        found: dict[str, str] = {}
        key = None
        for bit in text.split(";"):
            if "=" in bit:
                key, _, value = bit.partition("=")
                key = key.strip()
                found[key] = value.strip()
            elif key and bit.strip():
                found[key] += "; " + bit.strip()
            elif bit.strip():
                return None
        return found if len(found) >= 2 else None

    def parse(self, text: str, allowed: set[str] | None = None) -> dict[str, Any] | None:
        for mark in ("|", "\n", "\t", ";"):
            bits = text.split(mark)
            if len(bits) < 3:
                continue
            row: dict[str, str] = {}
            key = None
            for bit in bits:
                m = self.NAME.fullmatch(bit)
                if m and m.group(1) not in row and (allowed is None or m.group(1) in allowed or re.fullmatch(r"[a-z][a-z0-9_]*", m.group(1))):
                    key = m.group(1)
                    row[key] = m.group(2)
                elif key is None:
                    row = {}
                    break
                else:
                    row[key] += mark + bit
            if len(row) >= 3 and any(" " not in v.strip() and 3 <= len(v.strip()) <= 64 and re.search(r"\d", v) for v in row.values()):
                return {k: self.pairs(v) or self.scalar(v) for k, v in row.items()}
        return None

    def convert(self, node: Any) -> Any:
        if isinstance(node, str):
            text = node.strip()
            if text[:1] in ("{", "["):
                try:
                    return self.convert(json.loads(text))
                except ValueError:
                    pass
            return self.parse(text, self.known or None) or self.parse(text) or node
        if isinstance(node, list):
            lines = [x for x in node if isinstance(x, str)]
            rows = [self.parse(x) for x in lines]
            if lines and all(rows):
                tally: dict[str, int] = {}
                for r in rows:
                    for k in r:
                        tally[k] = tally.get(k, 0) + 1
                shared = {k for k, n in tally.items() if n * 2 >= len(rows)}
                self.known |= shared
                return [self.parse(x, shared) or r for x, r in zip(lines, rows)]
            return [self.convert(x) for x in node]
        if isinstance(node, dict):
            wrapper = len(node) <= 3
            out: dict[str, Any] = {}
            for k, v in node.items():
                out[k] = self.convert(v) if wrapper or isinstance(v, list) else v
            return out
        return node


class Listing:
    __slots__ = ("ident", "row", "opened", "facts", "fit", "codes", "seen_at")

    def __init__(self, ident: str):
        self.ident = ident
        self.row: dict[str, Any] = {}
        self.opened = False
        self.facts: dict[str, Any] = {}
        self.fit: dict[int, tuple[str, str]] = {}
        self.codes: dict[str, Any] = {}
        self.seen_at = 0


class Shelf:
    def __init__(self) -> None:
        self.items: dict[str, Listing] = {}
        self.order: list[str] = []
        self.id_key: str | None = None
        self.keys: dict[str, str | None] = {}
        self.sold_out: set[str] = set()

    # --- learning the row shape
    def learn_id(self, rows: list[dict[str, Any]]) -> None:
        if self.id_key:
            return
        best = None
        for key in rows[0]:
            values = [r.get(key) for r in rows]
            if not all(isinstance(v, str) and 2 < len(v) <= 64 and " " not in v.strip() for v in values):
                continue
            if len(rows) > 1 and len({v for v in values}) < len(values):
                continue
            score = (bool(FIELD["id"].search(key)), bool(re.search(r"\d", values[0])), -len(values[0]))
            if best is None or score > best[0]:
                best = (score, key)
        if best:
            self.id_key = best[1]

    def key_for(self, job: str, row: dict[str, Any]) -> str | None:
        if job in self.keys and self.keys[job] in row:
            return self.keys[job]
        pattern = FIELD[job]
        cands = [k for k, v in row.items() if k != self.id_key and pattern.search(str(k))]
        if job == "price":
            cands = [k for k in cands if _num(row[k]) is not None or isinstance(row[k], dict)]
        elif job == "stock":
            cands = [k for k in cands if isinstance(row[k], (bool, int, str))]
        elif job in ("category", "seller", "brand", "title", "remarks", "description"):
            cands = [k for k in cands if isinstance(row[k], str) or row[k] is None]
        if job == "brand":
            cands = [k for k in cands if not FIELD["seller"].search(k) or not re.search(r"seller|shop|vendor|merchant", k, re.I)]
        if job == "seller":
            cands = [k for k in cands if re.search(r"seller|shop|vendor|merchant|sold_by|retailer|store", k, re.I) and not FIELD["remarks"].search(k) and not FIELD["description"].search(k)]
        if job == "title":
            cands = [k for k in cands if not re.search(r"brand|maker|user|seller|shop|file|sold", k, re.I)]
        if job == "description":
            cands = [k for k in cands if not FIELD["remarks"].search(k)]
        if not cands:
            return None
        self.keys[job] = cands[0]
        return cands[0]

    @staticmethod
    def fact_block(row: dict[str, Any]) -> dict[str, Any]:
        best: dict[str, Any] = {}
        for key, value in row.items():
            if isinstance(value, dict):
                scalars = {k: v for k, v in value.items() if isinstance(v, (str, int, float)) and not isinstance(v, bool) or isinstance(v, list)}
                if len(scalars) >= 2 and len(scalars) > len(best):
                    best = scalars
        return best

    # --- absorbing observations
    def is_row(self, node: Any) -> bool:
        return isinstance(node, dict) and len(node) >= 3 and bool(self.id_key) and isinstance(node.get(self.id_key), str)

    def absorb(self, observation: Any, opened: bool = False, step: int = 0) -> list[str]:
        """Record every listing record found in an observation; returns their ids in order."""
        found: list[str] = []

        def take(node: dict[str, Any]) -> None:
            ident = str(node[self.id_key]).strip()  # type: ignore[index]
            item = self.items.get(ident)
            if item is None:
                item = Listing(ident)
                self.items[ident] = item
                self.order.append(ident)
            stock_key = self.key_for("stock", node)
            if stock_key and stock_key in item.row and item.row.get(stock_key) != node.get(stock_key):
                pass
            item.row.update(node)
            item.seen_at = step
            block = self.fact_block(node)
            if block:
                item.facts = block
                item.opened = True
            if opened:
                item.opened = True
            text = " ".join(str(v) for v in node.values() if isinstance(v, str))
            text += " " + " ".join(f"{k} {v}" for k, v in item.facts.items())
            item.codes = _fitment(self.title(ident), text)
            if stock_key and node.get(stock_key) is False:
                self.sold_out.add(ident)
            elif stock_key and node.get(stock_key) is True:
                self.sold_out.discard(ident)
            found.append(ident)

        def walk(node: Any, depth: int) -> None:
            if depth > 6:
                return
            if isinstance(node, list):
                rows = [n for n in node if isinstance(n, dict) and len(n) >= 3]
                if rows and not self.id_key:
                    self.learn_id(rows)
                for n in node:
                    walk(n, depth + 1)
            elif isinstance(node, dict):
                if not self.id_key and depth == 0 and len(node) >= 4 and not any(isinstance(v, list) and v and isinstance(v[0], dict) for v in node.values()):
                    self.learn_id([node])
                if self.is_row(node):
                    take(node)
                    return
                for v in node.values():
                    walk(v, depth + 1)

        walk(observation, 0)
        return found

    # --- fields
    def row(self, ident: str) -> dict[str, Any]:
        return self.items[ident].row if ident in self.items else {}

    def field(self, ident: str, job: str) -> Any:
        row = self.row(ident)
        key = self.key_for(job, row) if row else None
        return row.get(key) if key else None

    def price(self, ident: str) -> float | None:
        value = self.field(ident, "price")
        if isinstance(value, dict):
            value = next((v for v in value.values() if _num(v) is not None), None)
        return _num(value)

    def available(self, ident: str) -> bool:
        if ident in self.sold_out:
            return False
        value = self.field(ident, "stock")
        if value is False or (isinstance(value, (int, float)) and not isinstance(value, bool) and value <= 0):
            return False
        if isinstance(value, str) and re.search(r"\b(out of stock|sold out|unavailable|no)\b", value, re.I):
            return False
        return True

    def category(self, ident: str) -> str:
        return str(self.field(ident, "category") or "")

    def seller(self, ident: str) -> str:
        return str(self.field(ident, "seller") or "")

    def brand(self, ident: str) -> str:
        return str(self.field(ident, "brand") or "")

    def title(self, ident: str) -> str:
        return str(self.field(ident, "title") or "")

    def remarks(self, ident: str) -> str:
        named = self.field(ident, "remarks")
        if named:
            return str(named)
        # worlds name the fine-print field freely: any other free-text field of the row counts
        row = self.row(ident)
        taken = {self.key_for(job, row) for job in ("title", "description", "category", "seller", "brand")}
        return " ".join(v for k, v in row.items() if k not in taken and isinstance(v, str) and len(v) >= 15)

    def description(self, ident: str) -> str:
        return str(self.field(ident, "description") or "")

    def text(self, ident: str) -> str:
        item = self.items.get(ident)
        if not item:
            return ""
        parts = [self.title(ident), self.description(ident), self.remarks(ident), _dump(item.facts)]
        return " ".join(p for p in parts if p)

    def card(self, ident: str, needs: list[str], fit_text: str = "", model_wanted: str = "") -> str:
        item = self.items[ident]
        price = self.price(ident)
        bits = [ident, f"{price:g}" if price is not None else "price ?", "in stock" if self.available(ident) else "SOLD OUT"]
        if self.seller(ident):
            bits.append("seller " + _clip(self.seller(ident), 24))
        if self.brand(ident) and _norm(self.brand(ident)) != _norm(self.seller(ident)):
            bits.append("brand " + _clip(self.brand(ident), 24))
        bits.append(_clip(self.category(ident), 30))
        bits.append('"' + _clip(self.title(ident), 70) + '"')
        head = " | ".join(bits)
        lines = [head]
        if item.opened:
            facts = item.facts
            want = {w for n in needs for w in _stems(n.split(":")[0])}
            ordered = sorted(facts.items(), key=lambda kv: (not (_stems(kv[0]) & want), len(str(kv[1]))))
            shown = "; ".join(f"{k}={_clip(v, 60)}" for k, v in ordered)
            lines.append("   attributes: " + _clip(shown, FACT_CHARS * 2 if want else FACT_CHARS))
        else:
            lines.append("   (row only, attributes unread) " + _clip(self.description(ident), 120))
        rem = self.remarks(ident)
        if rem:
            lines.append("   fine print: " + _clip(rem, TEXT_CHARS))
        codes = item.codes
        if codes.get("models") or codes.get("fits") or codes.get("families") or codes.get("more"):
            parts = []
            if codes.get("models"):
                parts.append("model: " + ", ".join(codes["models"][:3]))
            if codes.get("fits"):
                parts.append("fits: " + ", ".join(sorted(codes["fits"])[:8]))
            for prefix, exc in codes.get("families") or []:
                parts.append(f"fits every {prefix}xx" + (f" except {', '.join(sorted(exc))}" if exc else ""))
            if codes.get("more"):
                parts.append("see part " + ", ".join(codes["more"]) + " for the rest")
            lines.append("   codes: " + "; ".join(parts))
        if fit_text:
            lines.append("   verdict: " + fit_text)
        return "\n".join(lines)


# ----------------------------------------------------------------------------------------------------------------
# the agent
# ----------------------------------------------------------------------------------------------------------------
def _limit_kept(reply: str, value: str) -> str:
    """The value as the reply bounds it: a figure that the reply limits keeps that limit."""
    words = r"(?:at least|at most|no more than|no less than|not more than|not less than|up to|under|over|above|below|within|maximum of|minimum of|maximum|minimum)"
    if not value or re.search(words, value, re.I) or re.search(r"\bor (?:more|less|fewer|higher|lower)\b", value, re.I):
        return value
    hit = re.search(words + r"\s+" + re.escape(value), reply, re.I)
    if hit:
        return " ".join(hit.group(0).split())
    hit = re.search(re.escape(value) + r"\s+or\s+(?:more|less|fewer|higher|lower|above|below|under|over)\b", reply, re.I)
    return " ".join(hit.group(0).split()) if hit else value


def _about(sentence: str, prop: str) -> bool:
    """The property a redirect is for comes before the words naming the other listing."""
    first = re.split(r"\b(?:from|by|to the|at the)\b", sentence, maxsplit=1, flags=re.I)[0]
    wanted = _stems(prop)
    return bool(wanted) and wanted <= _stems(first)


class Product:
    def __init__(self, number: int, name: str, qty: int, needs: list[str], lookups: list[str], fits_with: int | None):
        self.number = number
        self.name = name
        self.qty = qty
        self.stated = needs
        self.heard: list[str] = []
        self.dropped: list[str] = []
        self.lookups: list[str] = lookups
        self.resolved: dict[str, str] = {}
        self.lost: list[str] = []
        self.fits_with = fits_with
        self.label = ""
        self.asked: list[str] = []
        self.chosen: str | None = None
        self.hints: list[str] = []

    def needs(self) -> list[str]:
        live = []
        for n in self.stated + self.heard:
            if n in self.dropped:
                continue
            if n in self.lookups:
                if n in self.resolved:
                    live.append(self.resolved[n])
                continue
            live.append(n)
        return live

    def pending(self) -> list[str]:
        return [n for n in self.lookups if n not in self.resolved and n not in self.dropped and n not in self.lost]

    def fixed_props(self) -> set[str]:
        out: set[str] = set()
        for n in self.stated + self.heard:
            out |= _stems(n.split(":")[0])
        return out


class Kohaku:
    def __init__(self, problem: dict[str, Any]):
        env = problem["environment"]
        self.binding = dict(env["binding"])
        self.view = env.get("policy_view") or {}
        self.request = str(self.view.get("query") or "")
        self.came_from: dict[str, set[str]] = {}
        self.tools = [t for t in self.view.get("tools") or [] if isinstance(t, dict) and isinstance(t.get("function"), dict)]
        self.schemas = {t["function"]["name"]: (t["function"].get("parameters") if isinstance(t["function"].get("parameters"), dict) else {"type": "object", "properties": {}}) for t in self.tools}
        steps = _count(self.view.get("max_steps"))
        self.max_steps = steps if steps and steps > 0 else 30
        self.step = 0
        self.text_records = TextRecords()
        self.finished = bool(self.view.get("finished"))
        self.started = time.monotonic()
        self.cost = 0.0
        self.calls = 0
        self.trace: list[dict[str, Any]] = []
        self.shelf = Shelf()
        self.products: list[Product] = []
        self.budget: float | None = None
        self.budget_unit = ""
        self.budget_scope = "total"
        self.pick = "any"
        self.measure = ""
        self.msg_limit = 0
        self.msg_used = 0
        self.procedure: list[dict[str, Any]] = []
        self.done_steps: list[str] = []
        self.ask_first: list[str] = []
        self.decline_rule = ""
        self.story_hints: list[dict[str, Any]] = []
        self.map: dict[str, Any] = {}
        self.notes: list[str] = []
        self.note_lines: set[str] = set()
        self.qa: list[tuple[int, str, str]] = []
        self.interviewed = False
        self.looked: set[str] = set()
        self.log: list[str] = []
        self.searched: set[str] = set()
        self.verdicts: dict[tuple[str, int], tuple[str, str]] = {}
        self.vetoes: list[str] = []
        self.cart: dict[str, int] = {}
        self.cart_total: float | None = None
        self.order_tries = 0
        self.refusals = 0
        self.last_refusal_changed = True
        self.round = 0
        self.rows_per_page = 10
        self.page_tokens: dict[str, Any] = {}
        self.no_reasoning_rejected: set[str] = set()
        self.pairs_seen: dict[str, dict[str, str]] = {}
        self.cheaper_done: set[str] = set()
        self.hop_steps = 0

    # ------------------------------------------------------------------ bookkeeping
    def note(self, **entry: Any) -> None:
        self.trace.append(entry)

    def left(self) -> int:
        return self.max_steps - self.step

    def age(self) -> float:
        return time.monotonic() - self.started

    def reserve(self) -> int:
        """Steps kept for the finish: cart set, order, one repair order, a required message."""
        talk = 1 if any(p.get("needs_message") and p.get("stage") == "before_order" for p in self.procedure) and self.msg_used < self.msg_limit else 0
        each = 1 if (self.map.get("cart") or {}).get("lines_arg") or not self.map.get("cart") else len(self.products)
        room = each + 1 + talk + self.pending_parts()
        if self.refusals and not (self.map.get("cart") or {}).get("lines_arg"):
            room += 1   # taking the lost line out is a step of its own
        if len(self.products) == 1 and not self.refusals:
            room += 5 if self.max_steps >= 24 else 3
        elif len(self.products) > 1 and not self.refusals:
            room += 1
        return room

    def spare(self) -> int:
        return self.left() - self.reserve()

    # ------------------------------------------------------------------ store calls
    def act(self, name: str, args: dict[str, Any]) -> Any:
        """One store call. Returns the observation (notes already absorbed), or None on failure."""
        if self.finished or self.left() <= 0 or name not in self.schemas:
            return None
        args, faults = _coerce(args, self.schemas[name])
        if faults:
            self.note(kind="bad_args", tool=name, faults=faults)
            return None
        payload = dict(self.binding)
        payload["step"] = self.step + 1
        payload["action"] = {"name": name, "args": args}
        for _ in range(4):
            result = _store.post_verbose("/environment/call", payload)
            data = result.data
            if isinstance(data, dict):
                self.step = data["step"] if isinstance(data.get("step"), int) else self.step + 1
                if data.get("finished"):
                    self.finished = True
                observation = self.text_records.convert(data.get("observation"))
                real, lines = self.unwrap(observation)
                self.log.append(f"{name} {_dump(args)[:90]}")
                if lines:
                    self.take_notes(lines, name)
                return real
            error = result.error or {}
            status = error.get("status")
            body = str(error.get("body") or "")
            self.note(kind="store_error", status=status, body=body[:160])
            if status == 409:
                low = body.lower()
                if "finish" in low or "quarantin" in low or "final" in low:
                    self.finished = True
                    return None
                m = re.search(r"expected (\d+)", body)
                if m:
                    self.step = int(m.group(1)) - 1
                    payload["step"] = self.step + 1
                    continue
                self.step += 1
                payload["step"] = self.step + 1
                continue
            if status in (400, 401, 402, 403, 404, 422):
                return None
        return None

    def unwrap(self, observation: Any) -> tuple[Any, list[str]]:
        """Separate the real result from note lines the store wrapped around or into it."""
        lines: list[str] = []
        real = observation
        for _ in range(3):
            if not isinstance(real, dict) or self.shelf.is_row(real):
                break
            lists = {k: v for k, v in real.items() if isinstance(v, list) and v and all(isinstance(x, str) for x in v)}
            heavy = {k: v for k, v in real.items() if isinstance(v, (dict, list)) and k not in lists}
            strs = {k: v for k, v in real.items() if isinstance(v, str) and k != "error"}
            rest = len(real) - len(lists) - len(heavy) - len(strs) - (1 if "error" in real else 0)
            if len(heavy) == 1 and (lists or strs) and rest == 0:
                # a wrapper around one real result, with the notes beside it ({"dekigi": result, "kere": [notes]},
                # {"observation": result, "refusal": [notes]})
                for v in lists.values():
                    lines += [x for x in v if len(x.split()) >= 4]
                lines += [v for v in strs.values() if len(v.split()) >= 5]
                real = next(iter(heavy.values()))
                continue
            if not heavy and lists and len(strs) == 1 and rest == 0:
                # a wrapped reply: the shopper's words plus note lists
                for v in lists.values():
                    lines += [x for x in v if len(x.split()) >= 4]
                real = dict(strs)
                break
            if heavy:
                # a plain result; long loose strings beside it are notes
                lines += [v for v in strs.values() if len(v.split()) >= 6]
                for v in lists.values():
                    lines += [x for x in v if len(x.split()) >= 6]
            elif len(real) >= 2 and lists and not strs:
                for v in lists.values():
                    lines += [x for x in v if len(x.split()) >= 6]
            break
        return real, [l for l in lines if l not in self.note_lines]

    # ------------------------------------------------------------------ model calls
    def deep_ok(self) -> bool:
        return self.cost <= SPEND_DEEP_UNTIL and self.age() < CLOCK_FINAL - 20

    def ask_model(self, prompt: str, model: str = QUICK, max_tokens: int = 1500, stable: str = "") -> dict[str, Any]:
        """One JSON answer. `stable` is a prefix that stays the same across calls (cached)."""
        if self.cost > SPEND_STOP or self.age() > CLOCK_FINAL:
            return {}
        if model == DEEP and not self.deep_ok():
            model = QUICK
        order = [model, QUICK if model != QUICK else OTHER]
        for which in order:
            got = self._call(which, prompt, max_tokens, stable)
            parsed = _json(got)
            if parsed is not None:
                return parsed
            if self.cost > SPEND_STOP:
                break
        return {}

    def _call(self, model: str, prompt: str, max_tokens: int, stable: str) -> str:
        messages: list[dict[str, Any]] = []
        if stable:
            messages.append({"role": "system", "content": [{"type": "text", "text": stable, "cache_control": {"type": "ephemeral"}}] if model.startswith("anthropic/") else stable})
        messages.append({"role": "user", "content": prompt})
        body: dict[str, Any] = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0}
        for attempt in range(3):
            self.calls += 1
            result = _models.post_verbose("/inference/chat/completions", body)
            data = result.data
            if isinstance(data, dict) and data.get("choices"):
                usage = data.get("usage") or {}
                if isinstance(usage.get("cost"), (int, float)):
                    self.cost += float(usage["cost"])
                choice = data["choices"][0] or {}
                message = choice.get("message") or {}
                content = message.get("content")
                if isinstance(content, list):
                    content = " ".join(str(c.get("text") or "") for c in content if isinstance(c, dict))
                if isinstance(content, str) and content.strip():
                    return content
                # the whole budget went into thinking: once more without it, then with more room
                if choice.get("finish_reason") == "length" and model not in self.no_reasoning_rejected and "reasoning" not in body:
                    body["reasoning"] = {"enabled": False}
                    body["max_tokens"] = max_tokens * 2
                    continue
                body["max_tokens"] = max_tokens * 3
                continue
            error = result.error or {}
            status = error.get("status")
            text = str(error.get("body") or "")
            self.note(kind="model_error", model=model, status=status, body=text[:120])
            if status == 400 and "reasoning" in body:
                self.no_reasoning_rejected.add(model)
                body.pop("reasoning", None)
                continue
            if status in (402,):
                self.cost = SPEND_STOP + 1
                return ""
            if status in (408, 425, 429, 500, 502, 503, 504, None) and attempt < 2 and self.age() < CLOCK_FINAL:
                time.sleep(3 + 4 * attempt)
                continue
            return ""
        return ""

    # ------------------------------------------------------------------ setup
    def tool_lines(self) -> str:
        return "\n".join(f"- {t['function']['name']}: {t['function'].get('description', '')} | parameters: {_dump(self.schemas[t['function']['name']])}" for t in self.tools)

    def read_request(self) -> None:
        prompt = WORK_ORDER + "\n\nTOOLS (for category labels and what the store calls things):\n" + self.tool_lines() + "\n\nREQUEST:\n" + self.request
        if self.view.get("opening") is not None:
            prompt += "\n\nWHAT THE STORE SHOWS FIRST:\n" + _dump(self.view.get("opening"))[:1500]
        order = self.ask_model(prompt, DEEP, 2500)
        if not order.get("products"):
            order = self.ask_model(prompt, QUICK, 2500) or order
        self.note(kind="work_order", order=order)
        for i, entry in enumerate(order.get("products") or [], 1):
            if not isinstance(entry, dict):
                continue
            qty = _count(entry.get("qty")) or 1
            needs = [str(n).strip() for n in (entry.get("needs") or []) if str(n).strip()]
            lookups = [str(n).strip() for n in (entry.get("lookups") or []) if str(n).strip()]
            for lk in lookups:
                if lk not in needs:
                    needs.append(lk)
            fits = _count(entry.get("fits_with"))
            self.products.append(Product(i, str(entry.get("name") or "the product"), max(1, qty), needs, lookups, fits))
        if not self.products:
            self.products = [Product(1, "the requested product", 1, [], [], None)]
        for p in self.products:
            if p.fits_with is not None and not (1 <= p.fits_with <= len(self.products) and p.fits_with != p.number):
                p.fits_with = None
        budget = order.get("budget") if isinstance(order.get("budget"), dict) else {}
        amount = _num(budget.get("amount"))
        self.budget_unit = str(budget.get("unit") or "")
        self.budget_scope = "each" if budget.get("scope") == "each" else "total"
        if amount and amount > 0:
            text = self.request.replace(",", "")
            if not re.search(r"(?<![\d.])" + re.escape(f"{amount:g}") + r"(?![\d])", text):
                for alt in (amount * 100, amount / 100):
                    if re.search(r"(?<![\d.])" + re.escape(f"{alt:g}") + r"(?![\d])", text):
                        amount = alt
                        break
            self.budget = amount
        pick = order.get("pick")
        self.pick = pick if pick in ("cheapest", "most", "least") else "any"
        self.measure = str(order.get("measure") or "")
        limit = _count(order.get("messages"))
        self.msg_limit = limit if limit is not None and limit >= 0 else (3 if self.map.get("ask") else 0)
        steps = _count(order.get("steps"))
        if steps and 8 <= steps < self.max_steps:
            self.max_steps = steps
        for entry in order.get("procedure") or []:
            if isinstance(entry, dict) and str(entry.get("what") or "").strip():
                stage = str(entry.get("stage") or "before_order")
                self.procedure.append({"what": str(entry["what"]).strip(), "stage": stage if stage in ("before_choice", "before_order", "before_decline") else "before_order", "needs_message": bool(entry.get("needs_message"))})
        self.ask_first = [str(x) for x in order.get("ask_first") or [] if str(x).strip()]
        self.decline_rule = str(order.get("decline_rule") or "").strip()
        for hint in order.get("story_hints") or []:
            if isinstance(hint, dict) and str(hint.get("property") or "").strip():
                n = _count(hint.get("product")) or 1
                if 1 <= n <= len(self.products):
                    self.products[n - 1].hints.append(str(hint["property"]).strip())
                    self.story_hints.append(hint)

    def map_tools(self) -> None:
        got = self.ask_model(TOOL_MAP + "\n\nTOOLS:\n" + self.tool_lines(), QUICK, 1800)
        if not isinstance(got.get("search"), dict) or not isinstance(got.get("buy"), dict):
            got = self.ask_model(TOOL_MAP + "\n\nTOOLS:\n" + self.tool_lines(), DEEP, 1800) or got
        self.map = got if isinstance(got, dict) else {}
        names = set(self.schemas)

        def props(tool: str | None) -> dict[str, Any]:
            return (self.schemas.get(tool or "") or {}).get("properties") or {}

        def free_text(tool: str | None, pattern: str = "") -> str | None:
            for k, v in props(tool).items():
                if isinstance(v, dict) and "string" in _kinds(v) and "enum" not in v and (not pattern or re.search(pattern, k, re.I)):
                    return k
            return None

        def named(tool: str | None, words: str) -> bool:
            return bool(tool) and bool(set(re.findall(r"[a-z]+", _norm(tool))) & set(words.split()))

        # search: the keyword tool is the one with a free-text argument; a second search-like tool with a category
        # argument is the filter tool
        s = self.map.get("search") if isinstance(self.map.get("search"), dict) else {}
        searchy = [n for n in names if named(n, "search find query look browse list lookup filter listings products items")]
        if s.get("tool") not in names or not free_text(s.get("tool"), r"text|query|term|word|phrase|keyword|search|q$"):
            s["tool"] = next((n for n in searchy if free_text(n, r"text|query|term|word|phrase|keyword|search|q$")), s.get("tool") if s.get("tool") in names else (searchy[0] if searchy else None))
        if s.get("filter_tool") not in names:
            s["filter_tool"] = next((n for n in searchy if n != s.get("tool") and any(re.search(r"categ|section|depart|group|shelf|aisle", k, re.I) for k in props(n))), None)
        s["query_arg"] = s.get("query_arg") if s.get("query_arg") in props(s.get("tool")) else free_text(s.get("tool"), r"text|query|term|word|phrase|keyword|search|q$")
        for job, pattern in (("category_arg", r"categ|section|depart|group|shelf|aisle|label"), ("brand_arg", r"brand|maker|manufact"), ("cap_arg", r"max|cap|limit|under|below|price|budget"), ("page_arg", r"page|offset|start|from|token|cursor|next|continue")):
            tool = s.get("filter_tool") if job != "page_arg" and s.get("filter_tool") in names else s.get("tool")
            if s.get(job) not in props(tool) and s.get(job) not in props(s.get("tool")):
                s[job] = next((k for k in props(tool) if re.search(pattern, k, re.I)), None)
        if s.get("filter_tool") not in names:
            s["filter_tool"] = None
        if s.get("mode_arg") not in props(s.get("tool")):
            s["mode_arg"] = None
            for k, v in props(s.get("tool")).items():
                opts = v.get("enum") if isinstance(v, dict) else None
                if isinstance(opts, list) and any(re.search(r"typed|keyword|text|free|term|filter|field|categ", str(o), re.I) for o in opts):
                    s["mode_arg"] = k
                    s["mode_keyword"] = next((o for o in opts if re.search(r"typed|keyword|text|free|term", str(o), re.I)), None)
                    s["mode_filter"] = next((o for o in opts if re.search(r"filter|field|categ|attr", str(o), re.I)), None)
                    break
        if not s.get("page_kind"):
            k = str(s.get("page_arg") or "")
            s["page_kind"] = "token" if re.search(r"token|cursor|continue", k, re.I) else "offset" if re.search(r"offset|start|from|row", k, re.I) else "index" if k else None
        self.map["search"] = s
        # open / page
        for job in ("open", "page"):
            o = self.map.get(job) if isinstance(self.map.get(job), dict) else {}
            if o.get("tool") not in names:
                o = {}
            if o and o.get("id_arg") not in props(o.get("tool")):
                o["id_arg"] = free_text(o.get("tool"), r"id|sku|code|target|item|listing|product") or free_text(o.get("tool"))
            self.map[job] = o if o.get("tool") and o.get("id_arg") else {}
        if not self.map["open"]:
            ranked = sorted(names, key=lambda n: (not named(n, "datasheet spec specs specification attributes attribute properties property technical describe details detail facts"), not named(n, "view inspect page listing item show")))
            for n in ranked:
                arg = free_text(n, r"id|sku|code|target|item|listing")
                if arg and len(props(n)) == 1 and named(n, "datasheet spec specs specification attributes attribute properties property technical describe details detail facts view inspect page listing item show"):
                    self.map["open"] = {"tool": n, "id_arg": arg}
                    break
        if not self.map["page"]:
            for n in names:
                arg = free_text(n, r"id|sku|code|target|item|listing")
                if arg and n != (self.map["open"] or {}).get("tool") and len(props(n)) == 1 and named(n, "page inspect view listing visit"):
                    self.map["page"] = {"tool": n, "id_arg": arg}
                    break
        if self.map.get("page") and self.map["page"].get("tool") == (self.map["open"] or {}).get("tool"):
            self.map["page"] = {}
        # cart
        c = self.map.get("cart") if isinstance(self.map.get("cart"), dict) else {}
        if c.get("tool") not in names:
            def cartish(n: str) -> bool:
                if named(n, "view show review see content contents list"):
                    return False
                cp0 = props(n)
                has_id = bool(free_text(n, r"id|sku|code|product|item|listing|chosen|target"))
                has_qty = any(isinstance(v, dict) and ("integer" in _kinds(v) or "number" in _kinds(v)) for v in cp0.values())
                has_lines = any(isinstance(v, dict) and "array" in _kinds(v) for v in cp0.values())
                return named(n, "cart bag basket hold keep add put trolley") and (has_id and has_qty or has_lines)
            c["tool"] = next((n for n in names if cartish(n)), None)
        cp = props(c.get("tool"))
        lines_arg = next((k for k, v in cp.items() if isinstance(v, dict) and "array" in _kinds(v)), None)
        if lines_arg:
            inner = (cp[lines_arg].get("items") or {}).get("properties") or {}
            c["lines_arg"] = lines_arg
            c["line_id_key"] = c.get("line_id_key") if c.get("line_id_key") in inner else next((k for k, v in inner.items() if isinstance(v, dict) and "string" in _kinds(v)), None)
            c["line_qty_key"] = c.get("line_qty_key") if c.get("line_qty_key") in inner else next((k for k, v in inner.items() if isinstance(v, dict) and ("integer" in _kinds(v) or "number" in _kinds(v))), None)
        else:
            c["lines_arg"] = None
            c["id_arg"] = c.get("id_arg") if c.get("id_arg") in cp else free_text(c.get("tool"), r"id|sku|code|product|item|listing|chosen")
            c["qty_arg"] = c.get("qty_arg") if c.get("qty_arg") in cp else next((k for k, v in cp.items() if isinstance(v, dict) and ("integer" in _kinds(v) or "number" in _kinds(v))), None)
            mode = next(((k, v["enum"]) for k, v in cp.items() if isinstance(v, dict) and isinstance(v.get("enum"), list) and len(v["enum"]) >= 2), None)
            if mode:
                c["mode_arg"] = mode[0]
                opts = [str(o) for o in mode[1]]
                c["mode_remove"] = c.get("mode_remove") if str(c.get("mode_remove")) in opts else next((o for o in opts if re.search(r"remov|delet|drop|discard|withdraw|take|clear|minus", o, re.I)), opts[-1])
                c["mode_add"] = c.get("mode_add") if str(c.get("mode_add")) in opts else next((o for o in opts if o != c["mode_remove"]), opts[0])
            else:
                c["mode_arg"] = None
        self.map["cart"] = c if c.get("tool") else {}
        cv = self.map.get("cart_view")
        self.map["cart_view"] = cv if isinstance(cv, str) and cv in names else next((n for n in names if named(n, "cart bag basket") and named(n, "view show review see content contents look")), None)
        # ask
        a = self.map.get("ask") if isinstance(self.map.get("ask"), dict) else {}
        if a.get("tool") not in names:
            a["tool"] = next((n for n in names if named(n, "message ask say write tell chat customer shopper user note contact") and not named(n, "give up refuse decline abort quit")), None)
        a["text_arg"] = a.get("text_arg") if a.get("text_arg") in props(a.get("tool")) else free_text(a.get("tool"))
        self.map["ask"] = a if a.get("tool") and a.get("text_arg") else {}
        # buy / decline
        for job, pattern in (("buy", "buy checkout check out purchase order settle pay conclude place finish submit"), ("decline", "refuse decline give up abort quit pass cancel end conclude walk away")):
            b = self.map.get(job) if isinstance(self.map.get(job), dict) else {}
            if b.get("tool") not in names:
                b["tool"] = next((n for n in names if named(n, pattern)), None)
            if not isinstance(b.get("args"), dict):
                b["args"] = {}
            b["args"], _ = _coerce(b["args"], self.schemas.get(b.get("tool") or ""))
            if job == "decline":
                b["text_arg"] = b.get("text_arg") if b.get("text_arg") in props(b.get("tool")) else free_text(b.get("tool"), r"remark|reason|text|message|note|justif|comment|why")
            self.map[job] = b if b.get("tool") else {}
        buy, dec = self.map.get("buy") or {}, self.map.get("decline") or {}
        if buy.get("tool") and buy.get("tool") == dec.get("tool") and buy.get("args") == dec.get("args"):
            mode = next(((k, v["enum"]) for k, v in props(buy["tool"]).items() if isinstance(v, dict) and isinstance(v.get("enum"), list) and len(v["enum"]) >= 2), None)
            if mode:
                opts = [str(o) for o in mode[1]]
                no = next((o for o in opts if re.search(r"pass|refuse|decline|abort|no|cancel|give|quit|skip", o, re.I)), opts[-1])
                yes = next((o for o in opts if re.search(r"order|buy|purchase|check|confirm|yes|place", o, re.I) and o != no), next(o for o in opts if o != no))
                buy["args"] = {mode[0]: yes}
                dec["args"] = {mode[0]: no}
        self.note(kind="tool_map", map=self.map)

    # ------------------------------------------------------------------ notes
    def take_notes(self, lines: list[str], tool: str) -> None:
        fresh = [l for l in lines if l not in self.note_lines]
        if not fresh:
            return
        self.note_lines.update(fresh)
        prompt = NOTE + "\n\nPRODUCTS:\n" + "\n".join(f"{p.number}. {p.name}: " + "; ".join(p.needs()) for p in self.products) + f"\n\nBUDGET NOW: {self.budget} {self.budget_unit}\n\nLINES:\n" + "\n".join(fresh)
        got = self.ask_model(prompt, QUICK, 900)
        changed: list[str] = []
        budget = _num(got.get("budget"))
        if budget and budget > 0:
            printed = [float(x.replace(",", "")) for line in fresh for x in re.findall(r"\d[\d,]*(?:\.\d+)?", line)]
            if printed and budget not in printed:
                match = [w for w in printed if abs(w - budget * 100) < 1 or abs(w - budget / 100) < 0.01]
                budget = match[0] if match else budget
        if budget and budget > 0 and budget != self.budget:
            self.budget = budget
            changed.append(f"budget now {budget:g}")
        for entry in got.get("drop") or []:
            if not isinstance(entry, dict):
                continue
            n = _count(entry.get("product")) or 1
            prop = _stems(str(entry.get("property") or "").split(":")[0])
            if not prop:
                continue
            for p in self.products if n < 1 or n > len(self.products) else [self.products[n - 1]]:
                for need in p.stated + p.heard:
                    if need not in p.dropped and (_stems(need.split(":")[0]) & prop):
                        p.dropped.append(need)
                        changed.append(f"dropped '{need}' for {p.name}")
        for entry in got.get("add") or []:
            if isinstance(entry, dict) and str(entry.get("need") or "").strip():
                n = _count(entry.get("product")) or 1
                p = self.products[min(max(1, n), len(self.products)) - 1]
                need = str(entry["need"]).strip()
                if need not in p.heard and need not in p.stated:
                    p.heard.append(need)
                    changed.append(f"added '{need}' for {p.name}")
        for ident in got.get("sold_out") or []:
            ident = str(ident)
            if ident in self.shelf.items:
                self.shelf.sold_out.add(ident)
                changed.append(f"{ident} sold out")
        if got.get("chosen_gone") is True and not any(str(i) in self.shelf.items for i in got.get("sold_out") or []):
            held = [i for i in (self.cart or {p.chosen: p.qty for p in self.products if p.chosen}) if i]
            if len(held) == 1:
                self.shelf.sold_out.add(held[0])
                changed.append(f"{held[0]} sold out")
            elif held:
                gone = self.recheck_stock(held)
                for i in gone:
                    self.shelf.sold_out.add(i)
                    changed.append(f"{i} sold out")
        if got.get("price_changed"):
            changed.append("a price changed")
            tool = self.map.get("page") or self.map.get("open") or {}
            for ident in [i for i in (self.cart or {p.chosen: p.qty for p in self.products if p.chosen}) if i]:
                if tool and self.left() > 2:
                    seen = self.act(tool["tool"], {tool["id_arg"]: ident})
                    if seen is not None:
                        self.shelf.absorb(seen, step=self.step)
        if changed:
            self.cheaper_done.clear()
            for p in self.products:
                key_changed = any("dropped" in c or "added" in c for c in changed)
                if key_changed:
                    for ident in list(self.shelf.items):
                        self.verdicts.pop((ident, p.number), None)
        summary = str(got.get("summary") or "").strip()
        self.notes.append((summary + (" (" + "; ".join(changed) + ")" if changed else "")).strip() or " | ".join(fresh)[:200])
        self.last_refusal_changed = bool(changed)
        self.note(kind="note", tool=tool, lines=[l[:160] for l in fresh], changed=changed)

    def recheck_stock(self, held: list[str]) -> list[str]:
        """Several lines and a note that one is gone: re-read each line's page where it shows stock; otherwise
        treat every line as gone (a fresh choice beats a refused order)."""
        gone: list[str] = []
        tool = (self.map.get("page") or self.map.get("open") or {})
        for ident in held:
            if not tool or self.left() <= 2 or not self.shelf.key_for("stock", self.shelf.row(ident)):
                gone.append(ident)
                continue
            observation = self.act(tool["tool"], {tool["id_arg"]: ident})
            if observation is None:
                gone.append(ident)
                continue
            self.shelf.absorb(observation, step=self.step)
            if not self.shelf.available(ident):
                gone.append(ident)
        return gone or held

    # ------------------------------------------------------------------ searches
    def search(self, product: Product, words: str | None = None, page: int = 1, cap: float | None = None, category: str | None = None, brand: str | None = None) -> list[str]:
        s = self.map.get("search") or {}
        tool = s.get("tool")
        if not tool:
            return []
        use_filter = words is None and (category or brand or cap is not None)
        if use_filter and s.get("filter_tool"):
            tool = s["filter_tool"]
        args: dict[str, Any] = {}
        props = (self.schemas.get(tool) or {}).get("properties") or {}
        if words is not None and s.get("query_arg") in props:
            spec = props[s["query_arg"]]
            limit = spec.get("maxLength") if isinstance(spec, dict) and isinstance(spec.get("maxLength"), int) else 60
            args[s["query_arg"]] = words[:limit]
        if category and s.get("category_arg") in props:
            args[s["category_arg"]] = category
        if brand and s.get("brand_arg") in props:
            args[s["brand_arg"]] = brand
        if cap is not None and s.get("cap_arg") in props:
            args[s["cap_arg"]] = cap
        if s.get("mode_arg") in props:
            args[s["mode_arg"]] = s.get("mode_filter") if use_filter and s.get("mode_filter") else (s.get("mode_keyword") or s.get("mode_filter"))
            if use_filter and not s.get("mode_filter"):
                args.pop(s["mode_arg"], None)
        key = _dump({k: v for k, v in args.items()})
        if page > 1 and s.get("page_arg") in props:
            kind = s.get("page_kind")
            if kind == "offset":
                args[s["page_arg"]] = (page - 1) * self.rows_per_page
            elif kind == "token":
                token = self.page_tokens.get(key)
                if token is None:
                    return []
                args[s["page_arg"]] = token
            else:
                args[s["page_arg"]] = page
        mark = f"{tool}:{key}:{page}"
        if mark in self.searched:
            return []
        self.searched.add(mark)
        if not args and words is None:
            return []
        observation = self.act(tool, args)
        if observation is None:
            return []
        found = self.shelf.absorb(observation, step=self.step)
        if words:
            for ident in found:
                self.came_from.setdefault(ident, set()).add(words)
        if found and len(found) > 1:
            self.rows_per_page = max(self.rows_per_page, len(found)) if self.rows_per_page == 10 else self.rows_per_page
        if isinstance(observation, dict):
            for k, v in observation.items():
                if re.search(r"token|cursor|next|continue", str(k), re.I) and isinstance(v, (str, int)) and not isinstance(v, bool):
                    self.page_tokens[key] = v
        label = f"{'list' if use_filter else 'search'} " + (f"'{words}'" if words is not None else "") + (f" category '{category}'" if category else "") + (f" brand '{brand}'" if brand else "") + (f" under {cap:g}" if cap is not None else "") + (f" page {page}" if page > 1 else "")
        self.log[-1] = label + f" -> {len(found)} rows" + (": " + ", ".join(found[:12]) if found else " (nothing)")
        if cap is not None and found and all((self.shelf.price(i) or 0) > cap for i in found):
            self.log[-1] += " (the cap was ignored)"
        self.note(kind="search", tool=tool, args=args, rows=len(found))
        if not found and words and page == 1 and len(words.split()) >= 3 and self.spare() >= 2:
            # an all-words engine: try the two most telling words (value first, then the product noun)
            parts = [w for w in words.split() if w.lower() not in ("the", "a", "an", "with", "for", "of", "and", "under", "below")]
            shorter = " ".join(parts[:2]) if len(parts) >= 2 else parts[0] if parts else ""
            if shorter and shorter != words:
                return self.search(product, words=shorter, page=1, cap=cap, category=category, brand=brand)
        return found

    def open(self, ident: str) -> bool:
        o = self.map.get("open") or {}
        if not o or ident not in self.shelf.items:
            return False
        item = self.shelf.items[ident]
        if item.opened:
            return True
        observation = self.act(o["tool"], {o["id_arg"]: ident})
        if observation is None:
            return False
        found = self.shelf.absorb(observation, opened=True, step=self.step)
        if ident not in found and isinstance(observation, dict):
            item.row.update({k: v for k, v in observation.items() if not isinstance(v, (dict, list)) or k not in item.row})
            block = self.shelf.fact_block(observation)
            if block:
                item.facts = block
            item.opened = True
            item.codes = _fitment(self.shelf.title(ident), self.shelf.text(ident))
        self.log[-1] = f"open {ident}"
        return True

    def open_page(self, ident: str) -> None:
        p = self.map.get("page") or {}
        if p and ident in self.shelf.items:
            observation = self.act(p["tool"], {p["id_arg"]: ident})
            if observation is not None:
                self.shelf.absorb(observation, opened=True, step=self.step)
                if isinstance(observation, dict):
                    item = self.shelf.items[ident]
                    item.row.update({k: v for k, v in observation.items() if not isinstance(v, (dict, list))})
                    item.codes = _fitment(self.shelf.title(ident), self.shelf.text(ident))
                self.log[-1] = f"page {ident}"

    # ------------------------------------------------------------------ product <-> listing matching
    def owner(self, ident: str) -> Product | None:
        label = self.shelf.category(ident)
        best, top = None, 0.0
        for p in self.products:
            score = 0.0
            if label and p.label and _norm(label) == _norm(p.label):
                score = 3.0
            elif label:
                mine = _stems(p.name) | _stems(p.label)
                theirs = _stems(label)
                score = len(mine & theirs) / max(1, len(mine | theirs))
            if score == 0.0 and not label:
                score = len(_stems(p.name) & _stems(self.shelf.title(ident))) / max(1, len(_stems(p.name)))
            if score > top:
                best, top = p, score
        return best if top > 0.15 else None

    def members(self, product: Product) -> list[str]:
        ours = [i for i in self.shelf.order if self.owner(i) is product]
        if product.label:
            exact = [i for i in ours if _norm(self.shelf.category(i)) == _norm(product.label)]
            if exact:
                return exact
        return ours

    def learn_labels(self) -> None:
        asked = _stems(self.request)
        for p in self.products:
            counts: dict[str, int] = {}
            for ident in self.shelf.order:
                label = self.shelf.category(ident)
                if label and (_stems(label) & _stems(p.name)):
                    counts[label] = counts.get(label, 0) + len(_stems(label) & _stems(p.name))
            name_words = _stems(p.name)
            if counts:
                most = max(len(_stems(c) & name_words) for c in counts)
                closest = [c for c in counts if len(_stems(c) & name_words) == most]
                asked_of_it = name_words | {w for n in p.needs() for w in _stems(n.split(":", 1)[-1])}
                least = min(len(_stems(c) - asked_of_it) for c in closest)
                closest = [c for c in closest if len(_stems(c) - asked_of_it) == least]
                spelled = [c for c in closest if len(_stems(c)) > 1 and _stems(c) <= asked]
                p.label = max(spelled or closest, key=lambda c: counts[c])

    def verdict(self, ident: str, product: Product) -> tuple[str, str]:
        return self.verdicts.get((ident, product.number), ("", ""))

    def affordable(self, ident: str, product: Product) -> bool:
        price = self.shelf.price(ident)
        if self.budget is None or price is None:
            return True
        if self.budget_scope == "each":
            return price <= self.budget + 1e-9
        others = sum((self.shelf.price(q.chosen) or 0) * q.qty for q in self.products if q is not product and q.chosen)
        return price * product.qty + others <= self.budget + 1e-9

    def ranked(self, product: Product) -> list[str]:
        ids = self.members(product)
        rank = {"yes": 0, "maybe": 1, "": 2, "no": 3}

        wanted = {w for n in product.needs() for w in _stems(n.split(":", 1)[-1])}

        def key(i: str) -> tuple[Any, ...]:
            v = self.verdict(i, product)[0]
            brought = any(_stems(q) & wanted for q in self.came_from.get(i, ()))
            return (rank.get(v, 2), not self.shelf.available(i), not self.affordable(i, product), not self.shelf.items[i].opened, not brought, self.shelf.price(i) if self.pick == "cheapest" and self.shelf.price(i) is not None else 0, -self.shelf.items[i].seen_at)

        return sorted(ids, key=key)

    # ------------------------------------------------------------------ fitment
    def fits_device(self, accessory: str, device: str) -> bool | None:
        codes = self.shelf.items[device].codes
        models = codes.get("models") or []
        if not models:
            return None
        names = {models[0]} | (codes.get("aliases") or set())
        acc = self.shelf.items[accessory].codes
        if not (acc.get("fits") or acc.get("families") or acc.get("more")):
            return None
        seen: set[str] = set()
        queue = [accessory]
        complete = True
        while queue:
            ident = queue.pop(0)
            if ident in seen:
                continue
            seen.add(ident)
            c = self.shelf.items[ident].codes
            if names & (c.get("fits") or set()):
                return True
            for prefix, exc in c.get("families") or []:
                if any(n.startswith(prefix) for n in names) and not names & exc:
                    return True
            for part in c.get("more") or []:
                nxt = self.by_code(part)
                if nxt:
                    queue.append(nxt)
                else:
                    complete = False
        return False if complete else None

    def by_code(self, code: str) -> str | None:
        return next((i for i in self.shelf.order if self.shelf.items[i].codes.get("own") == code), None)

    def fetch_parts(self, accessory: str, device: str) -> None:
        """Follow 'see part X for the rest' continuations by searching the code, when it decides the fit."""
        for _ in range(3):
            if self.fits_device(accessory, device) is not None or self.spare() < 2:
                return
            missing = [part for i in [accessory] + [self.by_code(c) for c in self.shelf.items[accessory].codes.get("more") or []] if i for part in self.shelf.items[i].codes.get("more") or [] if not self.by_code(part)]
            if not missing:
                return
            owner = self.owner(accessory) or self.products[0]
            found = self.search(owner, words=missing[0])
            if not self.by_code(missing[0]):
                for ident in found[:3]:
                    if missing[0] in (self.shelf.title(ident) + self.shelf.description(ident)).replace("-", "").upper():
                        self.shelf.items[ident].codes["own"] = missing[0]
                        break
                else:
                    return

    def parts_trail(self) -> None:
        """A carted line whose text continues on another part: that part's listing is looked up and opened before the order."""
        for p in self.products:
            if not p.chosen or p.chosen not in self.shelf.items:
                continue
            for part in self.shelf.items[p.chosen].codes.get("more") or []:
                if part in self.looked or self.left() <= len(self.products) + 1:
                    continue
                self.looked.add(part)
                found = self.search(p, words=part)
                self.note(kind="part_lookup", line=p.chosen, part=part, rows=len(found))
                target = self.by_code(part)
                if target and not self.shelf.items[target].opened and self.left() > len(self.products) + 1:
                    self.open(target)

    def pending_parts(self) -> int:
        return sum(1 for p in self.products if p.chosen and p.chosen in self.shelf.items for part in self.shelf.items[p.chosen].codes.get("more") or [] if part not in self.looked)

    def fit_problems(self) -> list[str]:
        for p in self.products:
            if p.fits_with is None and p.chosen and p.chosen in self.shelf.items:
                c = self.shelf.items[p.chosen].codes
                if c.get("fits") or c.get("families") or c.get("more"):
                    dev = next((q for q in self.products if q is not p and q.chosen and q.chosen in self.shelf.items and self.shelf.items[q.chosen].codes.get("models") and not self.shelf.items[q.chosen].codes.get("fits")), None)
                    if dev:
                        p.fits_with = dev.number
        problems = []
        for p in self.products:
            if p.fits_with is None or not p.chosen:
                continue
            device = self.products[p.fits_with - 1]
            if not device.chosen:
                continue
            self.fetch_parts(p.chosen, device.chosen)
            fit = self.fits_device(p.chosen, device.chosen)
            if fit is False:
                model = (self.shelf.items[device.chosen].codes.get("models") or ["?"])[0]
                good = [i for i in self.members(p) if i != p.chosen and self.fits_device(i, device.chosen) and self.shelf.available(i)]
                problems.append(f"{p.chosen} ({p.name}) does not fit {device.chosen} (model {model})" + (f"; rows that do: {', '.join(good[:5])}" if good else "; no seen row of that kind fits it: open or search for one, or choose another " + device.name))
        return problems

    # ------------------------------------------------------------------ interview
    def key_table(self, product: Product) -> list[tuple[str, int, int, list[str]]]:
        opened = [i for i in self.members(product) if self.shelf.items[i].opened]
        stats: dict[str, dict[str, Any]] = {}
        for ident in opened:
            for k, v in self.shelf.items[ident].facts.items():
                name = " ".join([w for w in _norm(k).split() if len(w) > 2] or _norm(k).split())
                if not name:
                    continue
                slot = stats.setdefault(name, {"n": 0, "values": []})
                slot["n"] += 1
                text = _clip(v, 40).lower()
                if text and text not in slot["values"]:
                    slot["values"].append(text)
        table = []
        for name, slot in stats.items():
            if re.search(r"\b(id|sku|code|asin|upc|ean|url|link|image|photo|warrant|option|variant|model|align|margin|indent|padding|font|text)s?\b", name):
                continue
            table.append((name, slot["n"], len(slot["values"]), slot["values"][:3]))
        table.sort(key=lambda t: (-t[1], t[2]))
        return table[:40]

    def plan_lists(self) -> dict[int, list[str]]:
        lines = []
        for p in self.products:
            lines.append(f"product {p.number}: {p.name} - fixed by the request: " + ("; ".join(p.needs()) or "nothing") + (f"; story hints: {', '.join(p.hints)}" if p.hints else ""))
            lines.append("  attribute keys on opened listings: " + ("; ".join(f"{name} (on {n}: {' / '.join(vals)})" for name, n, _, vals in self.key_table(p)) or "(none opened yet)"))
        prompt = ASK_PLAN + "\n\nREQUEST:\n" + self.request + "\n\nPRODUCTS:\n" + "\n".join(lines)
        if self.ask_first:
            prompt += "\n\nTHE REQUEST SAYS TO ASK ABOUT: " + "; ".join(self.ask_first)
        got = self.ask_model(prompt, DEEP, 1600)
        offered: dict[int, list[str]] = {}
        for entry in got.get("lists") or []:
            if isinstance(entry, dict) and _count(entry.get("product")):
                offered[_count(entry["product"])] = [str(x) for x in entry.get("properties") or []]
        lists: dict[int, list[str]] = {}
        for p in self.products:
            names: list[str] = []
            stems: list[set[str]] = []
            fixed = p.fixed_props()
            keys = [t[0] for t in self.key_table(p)]
            known = {w for k in keys for w in _stems(k)}
            carried = [x for x in offered.get(p.number, []) if _stems(x) & known]
            for raw in [a for a in self.ask_first] + carried + keys + offered.get(p.number, []):
                prop = re.split(r"[/(,;]| and | or ", raw)[0].strip().rstrip("?.").lower()
                st = _stems(prop)
                if not prop or len(prop.split()) > 4 or not st or (st <= fixed and raw not in self.ask_first):
                    continue
                twin = next((i for i, old in enumerate(stems) if st <= old or old <= st), None)
                if twin is None:
                    names.append(prop)
                    stems.append(st)
                elif stems[twin] < st:
                    names[twin], stems[twin] = prop, st
            lists[p.number] = names
        self.note(kind="ask_lists", lists=lists)
        return lists

    @staticmethod
    def turn_order(names: list[str], misses: int) -> list[str]:
        families: dict[str, list[str]] = {}
        for name in names:
            families.setdefault((name.split() or [""])[-1].lower(), []).append(name)
        runs = [sorted(f, key=lambda n: -len(n.split())) for f in families.values()]
        if misses == 1:
            runs = runs[::-1]
        elif misses >= 2:
            # wants and other subjects cluster near the top of a ranked list, so this order starts inside that cluster
            aim = 0.15 * (len(runs) - 1)
            by_aim = sorted(range(len(runs)), key=lambda i: (abs(i - aim), i))
            runs = [runs[(i + misses - 2) % len(runs)] for i in by_aim]
        return [n for run in runs for n in run]

    def question_for(self, group: list[Product], lists: dict[int, list[str]], misses: dict[int, int], room: int, keep: float) -> str:
        share = max(1, int((room - 30 * len(group)) / max(1, len(group)) / 15 * keep))
        parts = []
        for p in group:
            names = self.turn_order(lists[p.number][:share], misses.get(p.number, 0))
            if names:
                words = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " or " + names[-1]
                parts.append(f"For the {p.name}: which {words} do you prefer?")
        return " ".join(parts)

    def messages_free(self) -> int:
        reserved = sum(1 for p in self.procedure if p.get("needs_message") and p.get("stage") in ("before_order",) and p["what"] not in self.done_steps)
        return self.msg_limit - self.msg_used - reserved

    def send(self, text: str) -> str | None:
        a = self.map.get("ask") or {}
        if not a or self.msg_used >= self.msg_limit or self.left() <= 1:
            return None
        observation = self.act(a["tool"], {a["text_arg"]: text})
        if observation is None:
            return None
        self.msg_used += 1
        reply = observation
        if isinstance(observation, dict):
            texts = [v for v in observation.values() if isinstance(v, str)]
            reply = " ".join(texts) if texts else _dump(observation)
        elif not isinstance(observation, str):
            reply = _dump(observation)
        self.log[-1] = f"message '{_clip(text, 60)}' -> '{_clip(reply, 80)}'"
        return str(reply)

    def interview(self) -> None:
        if self.interviewed:
            return
        self.interviewed = True
        if not self.map.get("ask") or self.messages_free() <= 0:
            return
        lists = self.plan_lists()
        schema = ((self.schemas.get(self.map["ask"]["tool"]) or {}).get("properties") or {}).get(self.map["ask"]["text_arg"]) or {}
        room = min(760, int(schema["maxLength"]) - 10) if isinstance(schema.get("maxLength"), (int, float)) else 760
        asked: dict[int, int] = {}
        misses: dict[int, int] = {}
        keep = 1.0
        heard_props: set[str] = set()
        while self.messages_free() > 0 and self.left() > self.reserve() + 1 and not self.finished:
            free = self.messages_free()
            fresh = [p for p in self.products if not asked.get(p.number) and lists.get(p.number)]
            listed = [p for p in self.products if lists.get(p.number)]
            if len(listed) >= 3:
                sent = len(self.qa)
                queue = listed[::-1] if sent == 1 else (listed[len(listed) // 2:] + listed[: len(listed) // 2] if sent >= 2 else listed)
                group = sorted(queue, key=lambda p: len(p.heard))
            elif fresh:
                # more products than messages: the last message carries the rest
                group = fresh[:1] if len(fresh) <= free or free > 1 else fresh
            else:
                pool = [p for p in self.products if lists.get(p.number)]
                if not pool:
                    break
                dry = [p for p in pool if misses.get(p.number)]
                group = [max(dry or pool, key=lambda p: (misses.get(p.number, 0) == min(misses.get(q.number, 0) for q in (dry or pool)), len(lists[p.number])))]
            question = self.question_for(group, lists, misses, room, keep)
            if not question:
                break
            reply = self.send(question)
            if reply is None:
                break
            for p in group:
                asked[p.number] = asked.get(p.number, 0) + 1
                p.asked.append(question[:60])
            self.qa.append((group[0].number, question, reply))
            got = self.hear(group[0], question, reply, several=len(self.products) > 1)
            kind = got.get("kind")
            if kind in ("value", "pointer"):
                n = _count(got.get("product"))
                target = self.products[n - 1] if n and 1 <= n <= len(self.products) else group[0]
                gone = _stems(str(got.get("property") or "")) - {"type", "kind"}
                if gone:
                    before = len(lists.get(target.number, []))
                    lists[target.number] = [x for x in lists.get(target.number, []) if not ((_stems(x) - {"type", "kind"}) & gone)]
                    if len(lists[target.number]) == before or _norm(str(got.get("property"))) in heard_props:
                        misses[target.number] = misses.get(target.number, 0) + 1   # a near name still answers for it: turn the order
                    heard_props.add(_norm(str(got.get("property"))))
                if len(group) > 1:
                    for p in group:
                        if p is not target:
                            misses[p.number] = misses.get(p.number, 0) + 1
            elif kind == "split" and re.search(r"\b(?:one (?:at a time|thing|by one)|fewer|too many|at once|single)\b", reply, re.I):
                keep *= 0.25
            else:
                for p in group:
                    misses[p.number] = misses.get(p.number, 0) + 1
        # a search on every value drawn out, so the board shows listings that can carry it
        for p in self.products:
            for need in p.heard[-2:] or p.needs()[-1:]:
                words = _need_words(need)
                if need not in p.lookups and words and self.spare() > 4:
                    self.search(p, words=words)

    def hear(self, product: Product, prop: str, reply: str, several: bool = False) -> dict[str, Any]:
        listed = ("\nPRODUCTS:\n" + "\n".join(f"{p.number}. {p.name}" for p in self.products)) if several else f"\nPRODUCT: {product.name}"
        got = self.ask_model(ANSWER + listed + f"\nQUESTION: {prop}\nANSWER:\n{reply}", QUICK, 800)
        kind = got.get("kind")
        n = _count(got.get("product"))
        target = self.products[n - 1] if n and 1 <= n <= len(self.products) else product
        name = str(got.get("property") or "").strip() or "preference"
        value = _limit_kept(reply, str(got.get("value") or "").strip())
        budget = _num(got.get("budget"))
        if budget and budget > 0:
            self.budget = budget
            self.notes.append(f"budget now {budget:g} (from the shopper)")
        names_other = bool(re.search(r"[A-Z][\w-]+(?:'s\b| page\b)|\b(?:sold by|from|at|by)\s+[A-Z]", reply)) and (kind == "pointer" or bool(re.search(r"\b(?:listing|page)\b", reply, re.I)))
        if names_other:
            before = re.split(r"\b(?:listing|page)\b", reply, maxsplit=1, flags=re.I)[0]
            near = max(self.products, key=lambda p: len(_stems(before) & (_stems(p.name) | _stems(p.label))))
            if _stems(before) & (_stems(near.name) | _stems(near.label)):
                target = near
            parsed = self.ask_model(POINTER + "\n\nPRODUCT: " + target.name + "\nREQUIREMENT:\n" + reply, QUICK, 600)
            name = str(parsed.get("property") or name).strip()
        if kind in ("value", "pointer") and value and not names_other:
            need = f"{name}: {value}"
            if need not in target.heard:
                target.heard.append(need)
        elif kind in ("value", "pointer") and value:
            need = f"{name}: {reply.strip()}"
            if need not in target.heard:
                target.heard.append(need)
                target.lookups.append(need)
        self.note(kind="heard", product=target.number, got=got)
        return got

    # ------------------------------------------------------------------ pointer chains
    def resolve_chains(self) -> None:
        for p in self.products:
            for need in p.pending():
                if self.spare() < 3 or self.finished or self.age() > CLOCK_SETTLE:
                    return
                self.trace_chain(p, need)

    def find_listing(self, product: Product, seller: str, brand: str, kind: str, traits: list[str], exclude: set[str]) -> str | None:
        """Find the listing a description picks out: by seller/brand among rows seen, else by searching."""

        def score(ident: str) -> float:
            if ident in exclude:
                return -1.0
            s = 0.0
            if seller and _norm(seller) in (_norm(self.shelf.seller(ident)), _norm(self.shelf.brand(ident))):
                s += 3.0
            elif seller and _norm(seller) in _norm(self.shelf.text(ident)):
                s += 1.5
            elif seller:
                return -1.0
            if brand and _norm(brand) in (_norm(self.shelf.brand(ident)), _norm(self.shelf.seller(ident))):
                s += 1.5
            if kind:
                s += 1.0 * len(_stems(kind) & _stems(self.shelf.category(ident))) / max(1, len(_stems(kind)))
            text = _norm(" ".join(x for x in re.split(r"(?<=[.!?])\s+", self.shelf.text(ident)) if not re.search(r"\belsewhere\b|\bis not\b|\bnot the\b|\bnot a\b", x, re.I)))
            facts = self.shelf.items[ident].facts if ident in self.shelf.items else {}
            for t in traits:
                words = [w for w in _tokens(t) if w not in ("the", "a", "an", "with", "of", "and")]
                wrong = False
                for fk, fv in facts.items():
                    key_words = _stems(fk)
                    rest = _stems(t) - key_words
                    if key_words and key_words <= _stems(t) and rest and not (rest & _stems(str(fv))):
                        wrong = True
                whole_words = _tokens(text)
                if wrong:
                    s -= 1.0
                elif words and all(w in whole_words for w in words):
                    s += 1.0
                elif words and any(w in whole_words for w in words if len(w) > 2):
                    s += 0.3
            return s

        def best() -> tuple[str | None, float]:
            ranked = sorted(self.shelf.order, key=lambda i: -score(i))
            return (ranked[0], score(ranked[0])) if ranked and score(ranked[0]) > 0 else (None, 0.0)

        ident, s = best()
        need = 3.0 + float(len(traits))
        if ident is None or s < need:
            for words in dict.fromkeys(w for w in [seller or brand, " ".join(x for x in [seller or brand, kind] if x)] if w):
                if self.spare() >= 2 and self.hop_steps < HOP_STEPS and (ident is None or s < need):
                    self.hop_steps += 1
                    self.search(product, words=words)
                    ident, s = best()
            if (ident is None or s < need) and seller and self.spare() >= 2 and self.hop_steps < HOP_STEPS and (self.map.get("search") or {}).get("brand_arg"):
                self.hop_steps += 1
                self.search(product, brand=seller)
                ident, s = best()
        if ident is None:
            return None
        # with traits given, open the top candidates of that seller until one carries the traits
        if traits:
            cands = [i for i in sorted(self.shelf.order, key=lambda i: -score(i)) if score(i) >= 3.0][:9]
            for c in cands:
                if self.shelf.items[c].opened and score(c) >= need:
                    return c
            for c in cands:
                if self.spare() < 2 or self.hop_steps >= HOP_STEPS:
                    break
                if not self.shelf.items[c].opened:
                    self.hop_steps += 1
                    self.open(c)
                    if score(c) >= need:
                        return c
            ident, s = best()
        return ident

    def trace_chain(self, product: Product, need: str) -> None:
        parsed = self.ask_model(POINTER + "\n\nPRODUCT: " + product.name + "\nREQUIREMENT:\n" + need, QUICK, 600)
        prop = str(parsed.get("property") or need.split(":")[0]).strip()
        relation = str(parsed.get("relation") or "same")
        seller = re.sub(r"['’]s$", "", str(parsed.get("seller") or "").strip())
        brand = re.sub(r"['’]s$", "", str(parsed.get("brand") or "").strip())
        kind = str(parsed.get("kind") or "").strip()
        traits = [str(t) for t in parsed.get("traits") or [] if str(t).strip()]
        plain_value = str(parsed.get("plain") or "").strip()
        if plain_value and not seller and not brand and not traits:
            product.resolved[need] = f"{prop}: {plain_value}"
            for ident in list(self.shelf.items):
                self.verdicts.pop((ident, product.number), None)
            self.note(kind="plain_after_all", need=need[:120], resolved=product.resolved[need])
            return
        self.hop_steps = 0
        visited: set[str] = set()
        pairs: dict[str, str] = {}
        value = ""
        hops: list[str] = []
        for _ in range(HOPS_MAX):
            if self.spare() < 2 or self.finished:
                break
            ident = self.find_listing(product, seller, brand, kind, traits, visited)
            if not ident:
                break
            visited.add(ident)
            hops.append(ident)
            if not self.shelf.items[ident].opened:
                if self.hop_steps >= HOP_STEPS:
                    break
                self.hop_steps += 1
                self.open(ident)
            if self.map.get("page") and not self.shelf.remarks(ident) and self.hop_steps < HOP_STEPS:
                self.hop_steps += 1
                self.open_page(ident)
            record = {"title": self.shelf.title(ident), "category": self.shelf.category(ident), "seller": self.shelf.seller(ident), "attributes": self.shelf.items[ident].facts, "description": _clip(self.shelf.description(ident), 600), "fine_print": self.shelf.remarks(ident)}
            read = self.ask_model(PAGE.replace("PROPERTY", prop) + "\n\nRECORD:\n" + _dump(record), QUICK, 800)
            if isinstance(read.get("pairs"), dict) and read["pairs"]:
                pairs = {_norm(k): str(v).strip() for k, v in read["pairs"].items()}
                self.pairs_seen[_norm(prop)] = pairs
            own = str(read.get("value") or "").strip()
            said_here = _stems(self.shelf.text(ident))
            if own and not (any(_stems(k) & _stems(prop) for k in self.shelf.items[ident].facts) or _stems(prop) <= said_here):
                own = ""
            target_shop = str((read.get("next") or {}).get("seller") or (read.get("next") or {}).get("brand") or "") if isinstance(read.get("next"), dict) else ""
            sentences = [x for x in re.split(r"(?<=[.!?])\s+", self.shelf.text(ident)) if target_shop and target_shop.lower() in x.lower()]
            if read.get("what") == "redirect" and isinstance(read.get("next"), dict) and (not target_shop or any(_about(x, prop) for x in sentences)):
                nxt = read["next"]
                seller = re.sub(r"['’]s$", "", str(nxt.get("seller") or "").strip())
                brand = re.sub(r"['’]s$", "", str(nxt.get("brand") or "").strip())
                kind = str(nxt.get("kind") or kind).strip()
                traits = [str(t) for t in nxt.get("traits") or [] if str(t).strip()]
                sections_here = {w for i in self.shelf.order if _norm(self.shelf.seller(i)) == _norm(seller) for w in _stems(self.shelf.category(i))}
                trimmed = []
                for t in traits:
                    rest = [w for w in _tokens(t) if w not in _tokens(seller) and not (_stems(w) & sections_here) and w not in ("from", "with", "the", "and", "that", "which", "has", "have", "is", "in", "of", "for", "an", "its")]
                    if rest:
                        trimmed.append(" ".join(rest))
                traits = trimmed or traits
                if not traits and kind:
                    sections = {w for i in self.shelf.order if _norm(self.shelf.seller(i)) == _norm(seller) for w in _stems(self.shelf.category(i))}
                    traits = [w for w in _tokens(kind) if len(w) > 2 and not (_stems(w) & sections)]
                continue
            if own:
                value = own
            elif not value:
                # read it from the attributes directly when the model gave nothing
                for k, v in self.shelf.items[ident].facts.items():
                    if _stems(k) & _stems(prop):
                        value = str(v)
                        break
            if not value and traits:
                continue
            break
        if value and relation == "pair" and not pairs and not self.pairs_seen.get(_norm(prop)) and hops:
            # the store's rule may be printed on another listing of the same shop
            shop = self.shelf.seller(hops[-1])
            for other in [i for i in self.shelf.order if _norm(self.shelf.seller(i)) == _norm(shop) and i not in visited and not self.shelf.items[i].opened][:3]:
                if self.spare() < 2:
                    break
                self.open(other)
                rec = {"title": self.shelf.title(other), "attributes": self.shelf.items[other].facts, "description": _clip(self.shelf.text(other), 900)}
                got = self.ask_model(PAGE.replace("PROPERTY", prop) + "\n\nRECORD:\n" + _dump(rec), QUICK, 600)
                if isinstance(got.get("pairs"), dict) and got["pairs"]:
                    found_rule = {_norm(k): str(v).strip() for k, v in got["pairs"].items()}
                    own_there = str(got.get("value") or "").strip()
                    if _norm(value) not in found_rule and own_there and _norm(own_there) in found_rule:
                        told = _norm(" ".join(x for x in re.split(r"(?<=[.!?])\s+", self.shelf.text(other)) if not re.search(r"\belsewhere\b|\bis not\b|\bnot the\b|\bnot a\b", x, re.I)))
                        if not traits or any(all(w in told for w in _tokens(t)) for t in traits):
                            value = own_there
                        else:
                            continue
                    pairs = found_rule
                    self.pairs_seen[_norm(prop)] = pairs
                    hops.append(other)
                    break
        if value:
            if relation == "pair":
                rule = pairs or self.pairs_seen.get(_norm(prop)) or {}
                partner = rule.get(_norm(value)) or next((v for k, v in rule.items() if k in _norm(value) or _norm(value) in k), "")
                resolved = f"{prop}: {partner}" if partner else f"{prop}: the value that pairs with {value} under the store's rule"
            elif relation == "unlike":
                resolved = f"{prop}: not {value}"
            elif relation == "near":
                gap, base = _num(parsed.get("distance")), _num(value)
                resolved = f"{prop}: between {base - gap:g} and {base + gap:g}" if gap is not None and base is not None else f"{prop}: close to {value}"
            elif relation == "above":
                resolved = f"{prop}: at least {value}"
            elif relation == "below":
                resolved = f"{prop}: at most {value}"
            else:
                resolved = f"{prop}: {value}"
            product.resolved[need] = resolved
            for ident in list(self.shelf.items):
                self.verdicts.pop((ident, product.number), None)
        else:
            product.lost.append(need)
        self.note(kind="chain", product=product.number, need=need[:120], hops=hops, value=value, resolved=product.resolved.get(need), pairs=pairs)

    def redirect_before_buy(self, product: Product) -> None:
        """A chosen listing whose fine print sends the reader elsewhere for a property the needs name: follow it once."""
        ident = product.chosen
        if not ident or self.spare() < 3:
            return
        rem = self.shelf.remarks(ident)
        if not rem or not re.search(r"\b(refer|look (?:to|at)|go by|taken from|take .{0,30} from|instead of this|not (?:to|from|by) this|rather than this|never (?:to|from|by) this)\b", rem, re.I):
            return
        for need in product.needs():
            prop = need.split(":")[0].strip()
            if not any(_about(x, prop) for x in re.split(r"(?<=[.!?])\s+", rem)):
                continue
            key = f"redirect:{ident}:{_norm(prop)}"
            if key in self.searched:
                continue
            self.searched.add(key)
            fake = f"{prop}: the value stated for this listing's {prop} on the page its fine print refers to: {rem}"
            product.lookups.append(fake)
            product.heard.append(fake)
            self.trace_chain(product, fake)
            resolved = product.resolved.get(fake)
            if resolved:
                self.verdicts.pop((ident, product.number), None)
            else:
                product.lookups.remove(fake)
                product.heard.remove(fake)

    # ------------------------------------------------------------------ the board
    def board(self) -> str:
        lines = ["WORK ORDER"]
        for p in self.products:
            line = f"product {p.number}: {p.name} ×{p.qty}" + (f" (category label: {p.label})" if p.label else "")
            needs = p.needs()
            line += " — needs in force: " + ("; ".join(needs) if needs else "none beyond the kind of product")
            if p.pending():
                line += " — VALUE STILL TO BE READ FROM ANOTHER LISTING (resolved for you before the order): " + "; ".join(_clip(n, 90) for n in p.pending())
            if p.lost:
                line += " — could not be traced (treat as no constraint): " + "; ".join(_clip(n, 60) for n in p.lost)
            if p.dropped:
                line += " — dropped by the shopper: " + "; ".join(p.dropped)
            if p.fits_with:
                line += f" — must work with product {p.fits_with} (code fitment)"
            if p.chosen:
                line += f" — CHOSEN: {p.chosen}"
            lines.append(line)
        if self.budget is not None:
            lines.append(f"budget: {self.budget:g} {self.budget_unit} {'for each item' if self.budget_scope == 'each' else 'in total (price × quantity summed)'}")
        lines.append("pick rule: " + {"cheapest": "the CHEAPEST listing that meets every need", "most": f"the most {self.measure}", "least": f"the least {self.measure}", "any": "any listing that meets every need (do not hunt for better)"}[self.pick])
        if self.procedure:
            lines.append("procedure demanded by the request (code does the before_order/before_decline ones at the order): " + "; ".join(f"{p['what']} [{p['stage']}]" for p in self.procedure))
        s = self.map.get("search") or {}
        caps = []
        if s.get("category_arg"):
            caps.append("category listing")
        if s.get("cap_arg"):
            caps.append("a maximum-price cap")
        if s.get("brand_arg"):
            caps.append("a brand filter")
        lines.append("search takes: keywords (first words count most)" + (", " + ", ".join(caps) if caps else "") + f"; {self.rows_per_page} rows a page")
        lines.append(f"steps left: {self.left()} (keep {self.reserve()} for the order); messages left: {max(0, self.msg_limit - self.msg_used)}" + ("; interview done" if self.interviewed else "; interview not done yet"))
        if self.notes:
            lines.append("notes from the store/shopper so far: " + " | ".join(self.notes[-5:]))
        if self.qa:
            lines.append("QUESTIONS AND ANSWERS:")
            for n, q, a in self.qa:
                lines.append(f"  (product {n}) Q: {q} -> A: {_clip(a, 140)}")
        lines.append("")
        lines.append("LISTINGS SEEN (per product; opened ones show attributes; 'row only' ones need opening to be judged)")
        shown: set[str] = set()
        for p in self.products:
            ids = self.ranked(p)
            lines.append(f"-- product {p.number} ({p.name}): {len(ids)} rows seen, {sum(1 for i in ids if self.shelf.items[i].opened)} opened")
            device_model = ""
            if p.fits_with and self.products[p.fits_with - 1].chosen:
                device_model = (self.shelf.items[self.products[p.fits_with - 1].chosen].codes.get("models") or [""])[0]
            opened_ids = [i for i in ids if self.shelf.items[i].opened]
            unopened = [i for i in ids if not self.shelf.items[i].opened]
            for ident in opened_ids + unopened[:BOARD_ROWS]:
                v, why = self.verdict(ident, p)
                fit_text = f"{v} ({why})" if v else ""
                if device_model:
                    f = self.fits_device(ident, self.products[p.fits_with - 1].chosen)  # type: ignore[arg-type]
                    fit_text += (" | " if fit_text else "") + ("fits the chosen device" if f else "DOES NOT fit the chosen device" if f is False else "fit unknown")
                if v == "no" and ident != p.chosen:
                    price = self.shelf.price(ident)
                    lines.append(f"{ident} | {price:g} | opened, judged no ({why})" if price is not None else f"{ident} | opened, judged no ({why})")
                else:
                    lines.append(self.shelf.card(ident, p.needs(), fit_text))
                shown.add(ident)
            if len(unopened) > BOARD_ROWS:
                lines.append(f"   … and {len(unopened) - BOARD_ROWS} more unopened rows not shown (search with the wanted value to surface better ones)")
        stray = [i for i in self.shelf.order if i not in shown and self.owner(i) is None][-6:]
        if stray:
            lines.append("-- rows of other kinds (chain hops or off-target): " + ", ".join(f"{i} [{_clip(self.shelf.category(i), 20)}]" for i in stray))
        lines.append("")
        lines.append("SEARCH LOG: " + (" || ".join(self.log[-18:]) or "(nothing yet)"))
        table = self.fit_table()
        if table:
            lines.append("WHICH DEVICE CODE THE ROWS CAN MEET (best first; open these rows before others):")
            lines += table
        if self.vetoes:
            lines.append("FROM THE LAST ROUND: " + " | ".join(self.vetoes[-4:]))
        if self.cart:
            lines.append("CART: " + _dump(self.cart) + (f" total {self.cart_total:g}" if self.cart_total is not None else ""))
        return "\n".join(lines)

    def fit_table(self) -> list[str]:
        """Per device code printed on rows: its device rows, and the rows of each accessory whose fit list names it."""
        out: list[str] = []
        for dev in self.products:
            accs = [a for a in self.products if a is not dev and a.fits_with == dev.number]
            if not accs:
                continue
            codes: dict[str, list[str]] = {}
            for ident in self.members(dev):
                if not self.shelf.available(ident) or self.verdict(ident, dev)[0] == "no" or not self.affordable(ident, dev):
                    continue
                for code in (self.shelf.items[ident].codes.get("models") or [])[:1]:
                    codes.setdefault(code, []).append(ident)
            scored = []
            for code, ids in codes.items():
                parts = {}
                for acc in accs:
                    parts[acc.number] = [i for i in self.members(acc) if self.shelf.available(i) and self.verdict(i, acc)[0] != "no" and self.fits_device(i, ids[0]) is True][:4]
                scored.append((sum(1 for v in parts.values() if v), code, ids, parts))
            scored.sort(key=lambda s: (-s[0], -len(s[2])))
            for covered, code, ids, parts in scored[:5]:
                out.append(f"  code {code}: product {dev.number} rows {', '.join(ids[:4])}; " + "; ".join(f"product {n} rows {', '.join(v) if v else 'none yet'}" for n, v in parts.items()))
        return out

    # ------------------------------------------------------------------ the rounds
    def plan_round(self, final: bool = False) -> dict[str, Any]:
        stable = ROUND.replace("ACTIONS", str(ACTIONS_PER_ROUND)).replace("OPEN", str(OPEN_PER_ROUND)) + "\n\nREQUEST:\n" + self.request
        prompt = "BOARD (round %d):\n" % self.round + self.board()
        if final:
            prompt += "\n\nFINAL ROUND: no store steps are left for browsing. Give verdicts for every opened listing not yet judged, then for each product without a confirmed choice choose the best opened, in-stock listing on the board (the closest fit to the needs, inside the budget) and order. Only choose and order actions run now."
        model = DEEP if (self.round in (1,) and len(self.products) > 1 or final) and self.deep_ok() else QUICK
        got = self.ask_model(prompt, model, 2400, stable=stable)
        if not isinstance(got.get("actions"), list):
            got = self.ask_model(prompt, QUICK if model != QUICK else OTHER, 2400, stable=stable) or got
        self.note(kind="round", n=self.round, model=model, final=final, note=str(got.get("note") or "")[:120], actions=got.get("actions"))
        return got

    def settle_choices(self) -> None:
        """One planning round that may only choose, so the fallback pick is a judged one rather than a guess."""
        if self.finished or self.cost > SPEND_STOP or self.age() > CLOCK_FINAL - 15:
            return
        if all(p.chosen and self.shelf.available(p.chosen) and self.verdict(p.chosen, p)[0] in ("yes", "maybe") for p in self.products):
            return
        self.round += 1
        got = self.plan_round(final=True)
        self.apply_verdicts(got)
        got["actions"] = [a for a in got.get("actions") or [] if isinstance(a, dict) and a.get("do") in ("choose", "order")]
        self.run_actions(got)

    def apply_verdicts(self, got: dict[str, Any]) -> None:
        for v in got.get("verdicts") or []:
            if not isinstance(v, dict):
                continue
            ident = str(v.get("id") or "").strip()
            n = _count(v.get("product")) or 0
            fit = str(v.get("fit") or "").lower()
            if ident in self.shelf.items and 1 <= n <= len(self.products) and fit in ("yes", "no", "maybe"):
                if fit == "yes" and not self.shelf.items[ident].opened:
                    fit = "maybe"
                self.verdicts[(ident, n)] = (fit, _clip(v.get("why"), 60))

    def run_actions(self, got: dict[str, Any]) -> str:
        """Execute a round's actions; returns 'order' when the planner wants to order."""
        self.vetoes = []
        opened = 0
        want_order = False
        for action in (got.get("actions") or [])[:ACTIONS_PER_ROUND]:
            if not isinstance(action, dict) or self.finished:
                continue
            do = str(action.get("do") or "")
            if self.spare() <= 0 and do not in ("choose", "order"):
                self.vetoes.append(f"'{do}' skipped: only {self.left()} steps left, choose and order")
                continue
            if self.age() > CLOCK_SETTLE and do not in ("choose", "order"):
                continue
            n = _count(action.get("product")) or 1
            product = self.products[min(max(1, n), len(self.products)) - 1]
            if do == "list":
                page = max(1, _count(action.get("page")) or 1)
                label = product.label or self.guess_label(product)
                if not label:
                    found = self.search(product, words=product.name, page=page)
                else:
                    found = self.search(product, category=label, page=page)
                    if not found and page == 1 and label == product.label and self.spare() > 2:
                        found = self.search(product, words=product.name)
                self.learn_labels()
            elif do == "search":
                words = str(action.get("words") or "").strip()
                if words:
                    self.search(product, words=words)
                    self.learn_labels()
            elif do == "cheaper":
                below = _num(action.get("below"))
                label = product.label or self.guess_label(product)
                s = self.map.get("search") or {}
                if below is not None and s.get("cap_arg"):
                    self.search(product, category=label, cap=max(0.0, below - 0.01))
                else:
                    self.vetoes.append("'cheaper' needs a price cap the search does not take; search with value words instead")
            elif do == "open":
                for ident in action.get("ids") or []:
                    ident = str(ident).strip()
                    if ident not in self.shelf.items:
                        self.vetoes.append(f"open {ident}: not a listing id seen")
                        continue
                    if self.shelf.items[ident].opened:
                        v, why = self.verdict(ident, product)
                        self.vetoes.append(f"{ident} was opened already" + (f" (judged {v}: {why})" if v else "; its attributes are on the board"))
                        continue
                    if opened >= OPEN_PER_ROUND or self.spare() <= 0:
                        break
                    self.open(ident)
                    opened += 1
            elif do == "interview":
                self.interview()
            elif do == "choose":
                ident = str(action.get("id") or "").strip()
                if ident not in self.shelf.items:
                    self.vetoes.append(f"choose {ident}: not a listing id seen")
                    continue
                owner = self.owner(ident)
                if owner is not None and owner is not product and self.shelf.category(ident):
                    self.vetoes.append(f"choose {ident} for product {product.number}: its category is '{self.shelf.category(ident)}', not the product's kind")
                    continue
                if not self.shelf.available(ident):
                    self.vetoes.append(f"choose {ident}: sold out")
                    continue
                product.chosen = ident
            elif do == "order":
                want_order = True
        return "order" if want_order else ""

    def guess_label(self, product: Product) -> str:
        self.learn_labels()
        return product.label

    def ready_to_order(self) -> list[str]:
        missing = []
        for p in self.products:
            if not p.chosen:
                missing.append(f"product {p.number} ({p.name}) has no chosen listing")
            elif not self.shelf.available(p.chosen):
                missing.append(f"{p.chosen} is sold out; choose another for product {p.number}")
            elif self.verdict(p.chosen, p)[0] != "yes" and not getattr(self, "_looked_more", False) and self.spare() > 8:
                opened = [i for i in self.members(p) if self.shelf.items[i].opened]
                unopened = [i for i in self.ranked(p) if not self.shelf.items[i].opened and self.shelf.available(i) and self.affordable(i, p)]
                if len(opened) < 3 and unopened:
                    self._looked_more = True
                    missing.append(f"{p.chosen} is not a confirmed fit for product {p.number} and only {len(opened)} listing(s) of that kind were opened: open a few more candidates ({', '.join(unopened[:4])}) before ordering")
        return missing

    def fill_missing(self) -> None:
        """Never end without a purchase: give every product its best candidate."""
        for p in self.products:
            if p.chosen and self.shelf.available(p.chosen):
                continue
            pool = [i for i in self.ranked(p) if self.shelf.available(i)]
            want = [w for n in p.needs() for w in _tokens(n.split(":", 1)[-1]) if len(w) > 2 and w not in ("not", "the", "and", "with", "least", "most")]

            def match(i: str) -> int:
                text = _norm(self.shelf.text(i))
                return sum(1 for w in want if w in text)

            ids = [i for i in pool if self.verdict(i, p)[0] != "no" and self.affordable(i, p)] or [i for i in pool if self.verdict(i, p)[0] != "no"] or pool
            rank = {"yes": 0, "maybe": 1, "": 2, "no": 3}
            ids.sort(key=lambda i: (rank.get(self.verdict(i, p)[0], 2), -match(i), not self.shelf.items[i].opened))
            if not ids and self.spare() >= 2:
                found = self.search(p, words=p.name)
                ids = [i for i in found if self.shelf.available(i)]
            if not ids:
                near = _stems(p.name)
                rows = [i for i in self.shelf.order if self.shelf.available(i) and self.affordable(i, p)]
                ids = sorted(rows, key=lambda i: -len(near & _stems(self.shelf.title(i) + " " + self.shelf.category(i))))
            if ids:
                p.chosen = ids[0]
                self.note(kind="filled", product=p.number, ident=ids[0], verdict=self.verdict(ids[0], p))

    def siblings(self, product: Product, lost: str) -> None:
        """Identical goods are listed several times: look for the sold-out listing's title again and open what matches it best."""
        title = self.shelf.title(lost)
        words = [w for w in re.split(r"[^0-9A-Za-z]+", title) if len(w) >= 2]
        if len(words) < 2 or self.spare() < 1:
            return
        s = self.map.get("search") or {}
        price = self.shelf.price(lost)
        cap = price if s.get("cap_arg") and price is not None else None
        found = self.search(product, words=" ".join(words[:6]), cap=cap)
        mine = _stems(title)
        same = [i for i in found if i != lost and self.shelf.available(i) and self.affordable(i, product) and len(mine & _stems(self.shelf.title(i))) >= 0.7 * max(1, len(mine))]
        same.sort(key=lambda i: (_norm(self.shelf.title(i)) != _norm(title), self.shelf.price(i) != price))
        self.note(kind="siblings", lost=lost, found=same[:6])
        for ident in [i for i in same if not self.shelf.items[i].opened][:3]:
            if self.spare() < 1:
                break
            self.open(ident)

    def cheap_words(self, product: Product) -> None:
        """More listings than one search can show, under a cheapest pick: search the rarer words of the cheapest rows seen, following pages while they bring new rows."""
        s = self.map.get("search") or {}
        if self.pick != "cheapest" or s.get("cap_arg"):
            return
        floor = 1 if self.refusals else 7
        tried = getattr(self, "_cheap_tried", set())
        self._cheap_tried = tried
        head = (product.name.split() or [""])[-1]
        for _ in range(10):
            if self.spare() < floor or self.age() > CLOCK_SETTLE:
                return
            rows = [i for i in self.members(product) if self.shelf.available(i) and self.shelf.price(i) is not None]
            if not rows:
                return
            rows.sort(key=lambda i: self.shelf.price(i))
            freq: dict[str, int] = {}
            for i in rows:
                for w in _stems(self.shelf.title(i)):
                    freq[w] = freq.get(w, 0) + 1
            cut = max(3, len(rows) // 4)
            pick = ""
            for i in rows[:15]:
                for w in re.findall(r"[A-Za-z]{4,}", self.shelf.title(i)):
                    st = _stems(w)
                    if w.lower() not in tried and not (st & _stems(product.name)) and all(freq.get(x, 0) <= cut for x in st):
                        pick = w
                        break
                if pick:
                    break
            if not pick:
                return
            tried.add(pick.lower())
            seen = set(self.shelf.order)
            for page in range(1, 8):
                if self.spare() < floor:
                    return
                found = self.search(product, words=f"{pick} {head}".strip(), page=page)
                if not found or not [i for i in found if i not in seen]:
                    break
                seen.update(found)

    def open_backups(self) -> None:
        """Stores often sell out the first line ordered: with steps to spare, make sure each product has a second
        judged candidate by opening up to two more affordable rows (cheapest first under a cheapest rule)."""
        if getattr(self, "_backups_done", False) or self.spare() < 6:
            return
        self._backups_done = True
        for p in self.products:
            if not p.chosen:
                continue
            yes = [i for i in self.members(p) if i != p.chosen and self.shelf.available(i) and self.verdict(i, p)[0] == "yes"]
            if yes:
                continue
            rows = [i for i in self.ranked(p) if i != p.chosen and not self.shelf.items[i].opened and self.shelf.available(i) and self.affordable(i, p) and self.verdict(i, p)[0] != "no"]
            if self.pick == "cheapest":
                rows.sort(key=lambda i: self.shelf.price(i) if self.shelf.price(i) is not None else 1e18)
            for ident in rows[:2]:
                if self.spare() < 5:
                    break
                self.open(ident)
                self.note(kind="backup_open", product=p.number, ident=ident)

    # ------------------------------------------------------------------ cheapest sweep
    def sweep_cheaper(self) -> None:
        if self.pick != "cheapest":
            return
        s = self.map.get("search") or {}
        for p in self.products:
            if not p.chosen or p.number in self.cheaper_done:
                continue
            price = self.shelf.price(p.chosen)
            if price is None:
                continue
            label = p.label or self.guess_label(p)
            if not s.get("cap_arg"):
                self.cheaper_done.add(p.number)
                query = next((_need_words(n) for n in reversed(p.needs()) if _need_words(n)), "") or p.name
                for page in range(1, 8):
                    if self.spare() < 8:
                        break
                    if not self.search(p, words=query, page=page) and page > 1:
                        break
                self.cheap_words(p)
                cheaper = [i for i in self.members(p) if self.shelf.available(i) and (self.shelf.price(i) or price) < price and self.verdict(i, p)[0] != "no" and not self.shelf.items[i].opened]
                for ident in sorted(cheaper, key=lambda i: self.shelf.price(i) or 0)[:5]:
                    if self.spare() < 4:
                        break
                    self.open(ident)
                if cheaper:
                    self.vetoes.append(f"cheapest rule: cheaper rows of {p.name} were listed and opened; judge them before ordering")
                continue
            if not label:
                continue
            self.cheaper_done.add(p.number)
            query = next((_need_words(n) for n in reversed(p.needs()) if _need_words(n)), "")
            if query and self.spare() >= 4:
                under = [i for i in self.search(p, words=query, cap=max(0.0, price - 0.01)) if self.shelf.available(i) and (self.shelf.price(i) or price) < price and self.verdict(i, p)[0] != "no"]
                if under:
                    for ident in sorted(under, key=lambda i: self.shelf.price(i) or 0)[:5]:
                        if self.spare() < 4:
                            break
                        self.open(ident)
                    self.vetoes.append(f"cheapest rule: cheaper rows of {label} were listed and opened; judge them before ordering")
                    continue
            for factor in (0.5, 0.75, 1.0):
                if self.spare() < 4:
                    break
                found = self.search(p, category=label, cap=max(0.0, round(price * factor, 2) if factor < 1 else price - 0.01))
                cheaper = [i for i in found if self.shelf.available(i) and (self.shelf.price(i) or price) < price and self.verdict(i, p)[0] != "no"]
                if not cheaper:
                    continue
                # let the planner judge them: open the cheapest few now
                for ident in sorted(cheaper, key=lambda i: self.shelf.price(i) or 0)[:5]:
                    if self.spare() < 4:
                        break
                    self.open(ident)
                self.vetoes.append(f"cheapest rule: cheaper rows of {label} were listed and opened; judge them before ordering")
                break

    # ------------------------------------------------------------------ finishing
    def set_cart(self, lines: dict[str, int]) -> bool:
        c = self.map.get("cart") or {}
        if not c:
            return True
        tool = c["tool"]
        if c.get("lines_arg"):
            entries = [{c["line_id_key"]: i, **({c["line_qty_key"]: q} if c.get("line_qty_key") else {})} for i, q in lines.items()]
            observation = self.act(tool, {c["lines_arg"]: entries})
            ok = observation is not None and not (isinstance(observation, dict) and observation.get("error"))
            if ok:
                self.cart = dict(lines)
                if isinstance(observation, dict):
                    total = next((_num(v) for k, v in observation.items() if re.search(r"total|sum|amount", str(k), re.I) and _num(v) is not None), None)
                    self.cart_total = total
            return ok
        if not c.get("id_arg"):
            return True
        ok = True
        for ident, qty in list(self.cart.items()):
            if ident in lines and lines[ident] == qty:
                continue
            if c.get("mode_arg"):
                args = {c["id_arg"]: ident, c["mode_arg"]: c.get("mode_remove")}
                if c.get("qty_arg"):
                    args[c["qty_arg"]] = qty
                if self.act(tool, args) is None:
                    ok = False
                self.cart.pop(ident, None)
        for ident, qty in lines.items():
            if self.cart.get(ident) == qty:
                continue
            args = {c["id_arg"]: ident}
            if c.get("mode_arg"):
                args[c["mode_arg"]] = c.get("mode_add")
            if c.get("qty_arg"):
                args[c["qty_arg"]] = qty
            if self.act(tool, args) is None:
                ok = False
            else:
                self.cart[ident] = qty
        return ok

    def do_procedure(self, stage: str, lines: dict[str, int]) -> None:
        for entry in self.procedure:
            if entry["stage"] != stage or entry["what"] in self.done_steps:
                continue
            if entry.get("needs_message") and self.map.get("ask") and self.msg_used < self.msg_limit and self.left() > 1:
                planned = "; ".join(f"{i} ×{q} at {self.shelf.price(i):g}" if self.shelf.price(i) is not None else f"{i} ×{q}" for i, q in lines.items())
                got = self._call(QUICK, STEP_MESSAGE.replace("STEP", entry["what"]).replace("LINES", planned or "nothing yet"), 200, "")
                text = got.strip().strip('"') if got else f"Before I finish: {entry['what']}. Shall I go ahead?"
                reply = self.send(_clip(text, 300))
                self.done_steps.append(entry["what"])
                if reply:
                    self.qa.append((0, text, reply))
                    got2 = self.ask_model(NOTE + "\n\nPRODUCTS:\n" + "\n".join(f"{p.number}. {p.name}" for p in self.products) + f"\n\nBUDGET NOW: {self.budget}\n\nLINES:\n" + reply, QUICK, 600)
                    if _num(got2.get("budget")):
                        self.budget = _num(got2.get("budget"))
            elif not entry.get("needs_message"):
                self.done_steps.append(entry["what"])

    def review(self, lines: dict[str, int]) -> dict[str, Any]:
        if not self.deep_ok() and self.cost > SPEND_STOP:
            return {}
        records = []
        for p in self.products:
            ident = p.chosen
            if not ident:
                continue
            records.append(f"product {p.number} ({p.name} ×{p.qty}) needs: " + ("; ".join(p.needs()) or "none") + "\n" + self.shelf.card(ident, p.needs()))
            if self.pick == "cheapest":
                others = [i for i in self.ranked(p) if i != ident and self.shelf.items[i].opened and self.shelf.available(i) and (self.shelf.price(i) or 1e18) < (self.shelf.price(ident) or 0)][:6]
                if others:
                    records.append("  cheaper opened listings of that kind:\n" + "\n".join(self.shelf.card(i, p.needs()) for i in others))
        prompt = REVIEW + "\n\nREQUEST:\n" + self.request + f"\n\nBUDGET: {self.budget} {self.budget_unit} ({self.budget_scope})" + ("\nNOTES APPLIED: " + " | ".join(self.notes[-4:]) if self.notes else "") + "\n\nCART AND RECORDS:\n" + "\n".join(records)
        got = self.ask_model(prompt, DEEP, 900)
        self.note(kind="review", got=got)
        return got

    def place_order(self) -> bool:
        """Cart set + order. Returns True when the episode ended."""
        b = self.map.get("buy") or {}
        if not b:
            return False
        lines = {p.chosen: p.qty for p in self.products if p.chosen}
        if not lines:
            return False
        if self.order_tries < 3 and self.deep_ok() and self.spare() >= 2:
            got = self.review(lines)
            if got and got.get("ok") is False:
                swapped = False
                for sw in got.get("swap") or []:
                    if not isinstance(sw, dict):
                        continue
                    n = _count(sw.get("product")) or 0
                    ident = str(sw.get("id") or "")
                    if 1 <= n <= len(self.products) and ident in self.shelf.items and self.shelf.available(ident) and self.shelf.items[ident].opened and self.verdict(ident, self.products[n - 1])[0] != "no" and (self.owner(ident) is self.products[n - 1] or not self.shelf.category(ident)):
                        self.products[n - 1].chosen = ident
                        swapped = True
                if swapped:
                    lines = {p.chosen: p.qty for p in self.products if p.chosen}
                self.vetoes.append("review: " + "; ".join(str(x) for x in got.get("problems") or [])[:200])
        self.do_procedure("before_order", lines)
        self.parts_trail()
        if not self.set_cart(lines):
            self.note(kind="cart_failed")
        if self.finished:
            return True
        self.order_tries += 1
        self.last_refusal_changed = False
        observation = self.act(b["tool"], dict(b.get("args") or {}))
        self.note(kind="order", tries=self.order_tries, lines=lines, reply=_clip(_dump(observation), 300), finished=self.finished)
        if self.finished:
            return True
        self.refusals += 1
        if observation is not None and isinstance(observation, dict):
            lines_text = [v for v in observation.values() if isinstance(v, str) and len(v.split()) >= 4]
            if lines_text:
                self.take_notes(lines_text, b["tool"])
        return False

    def decline(self) -> None:
        d = self.map.get("decline") or {}
        if not d or self.finished or self.left() < 1:
            return
        self.do_procedure("before_decline", {})
        args = dict(d.get("args") or {})
        if d.get("text_arg"):
            args[d["text_arg"]] = NOTHING_FITS
        self.act(d["tool"], args)
        self.note(kind="declined", finished=self.finished)

    # ------------------------------------------------------------------ main loop
    def run(self) -> list[dict[str, Any]]:
        if self.finished:
            return self.trace
        self.map_tools()
        self.read_request()
        if not self.msg_limit and self.map.get("ask"):
            self.msg_limit = 2
        opening = self.view.get("opening")
        if opening is not None:
            real, lines = self.unwrap(opening)
            self.shelf.absorb(real)
            if lines:
                self.take_notes(lines, "opening")
        # a first look at every product before the planner starts: one category page or a name search each
        for p in self.products:
            if self.spare() <= 1:
                break
            s = self.map.get("search") or {}
            if s.get("category_arg") and p.label:
                self.search(p, category=p.label)
            else:
                content = " ".join(w for w in p.name.split() if w.lower() not in ("or", "and", "the", "a", "an", "of", "for", "with")) or p.name
                first = self.search(p, words=content)
                self.learn_labels()
                if not p.label and len(content.split()) > 1:
                    first = self.search(p, words=content.split()[-1]) or first
                    self.learn_labels()
                if not p.label and first:
                    tally: dict[str, int] = {}
                    for ident in first:
                        cat = self.shelf.category(ident)
                        if cat:
                            tally[cat] = tally.get(cat, 0) + 1
                    if tally:
                        p.label = max(tally, key=lambda k: tally[k])
            self.learn_labels()
            if s.get("category_arg") and p.label and not any(self.owner(i) is p for i in self.shelf.order[-self.rows_per_page:]):
                self.search(p, category=p.label)
        # pointer needs the request itself states are resolved early, so the interview is not spent on them
        self.resolve_chains()
        # the attribute keys of opened listings feed the questions
        quota = max(3, min(9, int(0.4 * self.max_steps / max(1, len(self.products))) - 1))
        for p in self.products:
            for ident in [i for i in self.ranked(p) if self.shelf.available(i) and not self.shelf.items[i].opened][:quota]:
                if self.spare() <= 4:
                    break
                self.open(ident)
        self.interview()
        self.resolve_chains()
        ordered = False
        while not self.finished and self.round < ROUNDS_MAX and self.age() < CLOCK_SETTLE and self.cost < SPEND_STOP:
            self.round += 1
            got = self.plan_round()
            if not got:
                break
            self.apply_verdicts(got)
            signature = _dump(got.get("actions"))
            if signature == getattr(self, "_last_actions", None) and self.round > 2:
                self.note(kind="repeat_plan", round=self.round)
                self.finish(force=True)
                ordered = self.finished
                break
            self._last_actions = signature
            wants = self.run_actions(got)
            if not self.interviewed and (wants == "order" or self.round >= 3 or self.spare() <= 6) and self.messages_free() > 0:
                self.interview()
                self.resolve_chains()
                if wants == "order":
                    continue  # one more look with the answers in hand
            if wants == "order" or self.spare() <= 1:
                missing = self.ready_to_order()
                if missing and self.spare() > 3 and self.round < ROUNDS_MAX - 1:
                    self.vetoes += missing
                    continue
                if self.finish():
                    ordered = True
                    break
        if not self.finished and not ordered:
            self.finish(force=True)
        if not self.finished:
            self.decline()
        self.note(kind="end", finished=self.finished, steps=self.step, cost=round(self.cost, 5), calls=self.calls, seconds=round(self.age(), 1))
        return self.trace

    def finish(self, force: bool = False) -> bool:
        """Resolve what must be resolved, then order; after a refusal apply the notes and order again."""
        for attempt in range(4):
            if self.finished:
                return True
            self.resolve_chains()
            for p in self.products:
                self.redirect_before_buy(p)
            if any(not p.chosen or self.verdict(p.chosen, p)[0] not in ("yes", "maybe") for p in self.products):
                self.settle_choices()
            self.fill_missing()
            problems = self.fit_problems()
            if problems:
                self.repair_fit()
                problems = self.fit_problems()
            if problems and attempt == 0 and self.spare() > 3 and not force:
                self.vetoes += problems
                return False
            if attempt == 0 and not force:
                self.open_backups()
                before = len(self.log)
                self.sweep_cheaper()
                if len(self.log) > before and self.spare() > 3:
                    return False  # cheaper rows were opened: one more planner round judges them
            if not any(p.chosen for p in self.products):
                break
            over = self.over_budget()
            if over and self.spare() > 2 and attempt == 0 and not force:
                self.vetoes.append(over)
                return False
            if self.nothing_fits(attempt, force):
                self.decline()
                if self.finished:
                    return True
                continue
            if self.place_order():
                return True
            if self.left() <= 1:
                break
            # refused: the notes changed the task or not
            if self.last_refusal_changed and self.spare() > 0 and attempt < 2:
                for p in self.products:
                    if p.chosen and not self.shelf.available(p.chosen):
                        self.siblings(p, p.chosen)
                    self.cheap_words(p)
                for p in self.products:
                    if p.chosen and (self.verdict(p.chosen, p)[0] == "no" or not self.shelf.available(p.chosen) or not self.affordable(p.chosen, p)):
                        p.chosen = None
                        backup = [i for i in self.ranked(p) if self.shelf.available(i) and self.affordable(i, p) and self.verdict(i, p)[0] == "yes"]
                        if backup:
                            p.chosen = backup[0]
                            self.note(kind="backup", product=p.number, ident=backup[0])
                # let the planner judge and choose a replacement; one round to look, one more to settle
                for _ in range(2):
                    if not (self.round < ROUNDS_MAX and self.age() < CLOCK_SETTLE and self.spare() > 1):
                        break
                    if all(p.chosen and self.verdict(p.chosen, p)[0] in ("yes", "maybe") and self.shelf.available(p.chosen) for p in self.products):
                        break
                    self.round += 1
                    got = self.plan_round()
                    self.apply_verdicts(got)
                    self.run_actions(got)
                continue
            if not self.last_refusal_changed and self.decline_rule and attempt >= 1:
                break
        return self.finished

    def nothing_fits(self, attempt: int, force: bool) -> bool:
        """End without a purchase only on evidence: every question is spent, a line was never confirmed, and its record breaks a need."""
        if len(self.products) > 1 or (self.messages_free() > 0 and self.map.get("ask")):
            return False
        weak = [p for p in self.products if p.chosen and self.verdict(p.chosen, p)[0] != "yes"]
        if not weak or not self.deep_ok():
            return False
        got = self.review({p.chosen: p.qty for p in self.products if p.chosen})
        wrong = {_count(n) for n in got.get("contradicted") or []}
        if got.get("ok") is False and not got.get("swap") and any(p.number in wrong for p in weak):
            if not force and attempt == 0 and self.spare() > 4:
                self.vetoes.append("the chosen line breaks a need: find another listing")
                return False
            self.note(kind="nothing_fits", problems=got.get("problems"))
            return True
        return False

    def over_budget(self) -> str:
        if self.budget is None:
            return ""
        if self.budget_scope == "each":
            for p in self.products:
                if p.chosen and (self.shelf.price(p.chosen) or 0) > self.budget + 1e-9:
                    return f"{p.chosen} costs {self.shelf.price(p.chosen):g}, over the {self.budget:g} limit for each item"
            return ""
        total = sum((self.shelf.price(p.chosen) or 0) * p.qty for p in self.products if p.chosen)
        if total > self.budget + 1e-9 and total <= 50 * self.budget:
            return f"the planned total {total:g} is over the budget {self.budget:g}: choose cheaper listings"
        return ""

    def repair_fit(self) -> None:
        """Choose the device and every accessory that must fit it together: keep what fits, swap what does not,
        and when no accessory of some kind fits the device, switch to another device that a full set fits."""
        for device in self.products:
            deps = [p for p in self.products if p.fits_with == device.number and p.chosen]
            if not deps or not device.chosen:
                continue
            if all(self.fits_device(p.chosen, device.chosen) is not False for p in deps):
                continue

            def assign(dev: str) -> dict[int, str] | None:
                out: dict[int, str] = {}
                for p in deps:
                    keep = p.chosen if p.chosen and self.shelf.available(p.chosen) and self.fits_device(p.chosen, dev) else None
                    cands = [keep] if keep else [i for i in self.ranked(p) if self.shelf.available(i) and self.verdict(i, p)[0] != "no" and self.fits_device(i, dev)]
                    if not cands:
                        return None
                    out[p.number] = cands[0]
                return out

            firm = [p for p in deps if self.verdict(p.chosen, p)[0] == "yes"]
            if firm and not all(self.fits_device(p.chosen, device.chosen) is not False for p in firm):
                options = [i for i in self.ranked(device) if i != device.chosen and self.shelf.available(i) and self.verdict(i, device)[0] != "no" and self.shelf.items[i].codes.get("models") and all(self.fits_device(p.chosen, i) for p in firm)]
                for i in options[:3]:
                    if not self.shelf.items[i].opened and self.spare() >= 2:
                        self.open(i)
                if options:
                    device.chosen = options[0]
                    self.note(kind="device_moved", device=device.chosen)
            got = assign(device.chosen)
            if got is None:
                for dev in [i for i in self.ranked(device) if i != device.chosen and self.shelf.available(i) and self.verdict(i, device)[0] in ("yes", "maybe")]:
                    got = assign(dev)
                    if got:
                        device.chosen = dev
                        break
            if got:
                for p in deps:
                    p.chosen = got[p.number]
                self.note(kind="fit_repaired", device=device.chosen, lines={p.number: p.chosen for p in deps})


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        agent = Kohaku(problem_data)
    except Exception as exc:  # noqa: BLE001
        return [{"kind": "crash", "error": f"{type(exc).__name__}: {exc}"[:300]}]
    try:
        return agent.run()
    except Exception as exc:  # noqa: BLE001
        agent.trace.append({"kind": "crash", "error": f"{type(exc).__name__}: {exc}"[:300]})
        try:
            if not agent.finished:
                agent.fill_missing()
                if not agent.place_order() and not agent.finished:
                    agent.decline()
        except Exception as again:  # noqa: BLE001
            agent.trace.append({"kind": "crash", "error": f"{type(again).__name__}: {again}"[:300]})
        return agent.trace


__all__ = ["agent_main"]
