"""Ledger: a shopping agent for ORO Bench worlds.

The world's tools change from race to race, so nothing here assumes a tool name,
an argument name or a result shape. A language model drives the world's own tools;
the code around it keeps the books: steps and messages, every listing seen, the
cart, the requirements as they stand (stated, revealed by the shopper, changed
during the task), the questions still worth asking, and a check of every order
or decline before it is sent.
"""

from __future__ import annotations

import json
import math
import re
import time
from os import getenv
from typing import Any

from src.agent.proxy_client import ProxyClient

# ---------------------------------------------------------------------------
# models and budgets

_MODELS = {
    # live runs go through OpenRouter; model ids are OpenRouter ids
    "openrouter": {
        "main": "anthropic/claude-haiku-5.5",          # drives the tool loop
        "alt": "deepseek/deepseek-v3.2",               # last-resort fallback
        "reader": "openai/gpt-6-luna",                 # structured reading: result keys, verifier, decline check, planner, pushed lines
        "knowledge": "anthropic/claude-opus-5.5",      # one small call per task: which store label a described product belongs to
        "setup": "anthropic/claude-sonnet-5.5",        # request parse and answers to the agent's questions
        "setup_alt": "anthropic/claude-haiku-5.5",
    },
    # local testing with a Chutes key uses the same models where Chutes serves them
    "chutes": {
        "main": "deepseek-ai/DeepSeek-V3.2-TEE",
        "alt": "moonshotai/Kimi-K2.6-TEE",
        "reader": "deepseek-ai/DeepSeek-V3.2-TEE",
        "knowledge": "deepseek-ai/DeepSeek-V3.2-TEE",
        "setup": "deepseek-ai/DeepSeek-V3.2-TEE",
        "setup_alt": "moonshotai/Kimi-K2.6-TEE",
    },
}

TASK_SECONDS = 272.0      # the sandbox allows 300 s per task; keep a margin for the last calls
WRAP_UP_SECONDS = 60.0    # with less than this left, stop exploring and finish
FORCE_SECONDS = 22.0      # with less than this left, finish without the model
COST_SOFT_USD = 0.09      # per-task spend after which the extra checks are skipped
MAX_TURNS = 70
OBS_CHARS = 7000
HISTORY_CHARS = 70000
FINISH_RESERVE = 7        # steps kept for cart repair and a second order attempt


def _provider() -> str:
    provider = (getenv("INFERENCE_PROVIDER") or "").strip().lower()
    if provider in _MODELS:
        return provider
    return "openrouter"


def _models() -> dict[str, str]:
    return _MODELS[_provider()]


# ---------------------------------------------------------------------------
# prompts

JUDGE_RULES = """How a listing is judged against one requirement:
1. Find the listing's effective value for the requirement's property. Use the properties whose key names that property, in any spelling, word order or prefix; when no property names it, use the title, the description and the category label. A key that names a different property does not count even when its value contains the wanted word. When several keys name the property and disagree, the wanted value counts if any of them states it.
2. Compare by meaning, not by spelling: another spelling, inflection, word form or unit of the same value is the same value; one entry of a list is enough; a broader class, a narrower class, a blend or a merely related value is a different value. Numbers are compared after converting units; a limit includes its edge; a range qualifies only when all of it does; exclusions and closed lists are respected.
3. Verdict: met; not met, when the effective value is clearly different or an exclusion is violated; or unknown, when nothing in the record speaks to it. Unknown is not a failure.
Price, stock and category are judged from the id, price, in_stock and category_label fields at the top of each record, never from the property table. Only the listed requirements count; nothing is inferred from the story."""

SYSTEM_PROMPT = """You are a careful personal shopper working inside one online store, acting only through the tools provided. One shopper has sent the request shown below. A hidden grader scores the episode 1 only when the finish is fully right, otherwise 0:
- an ORDER is right when the cart holds exactly the requested item(s) at the requested quantity and nothing else; every item meets every stated requirement, every need the shopper revealed when asked, and every change announced during the task; price times quantity is within the current budget (equal is fine); the item is in stock when the request wants things on hand; it is the cheapest (or the best by the stated measure) of ALL fitting items when the request asks for that; and every procedure the request states has been followed;
- a DECLINE is right only when nothing in the store can meet the request as it now stands; then the request's rule for declining applies exactly (if it says not to offer substitutes, the remark must not name, describe or point at any listing).

HOW THE STORE WORKS
- Every tool call costs one step from the step budget, including calls that fail. Messages to the shopper also count against a separate message budget. Thinking is free; steps are not. Never repeat a call whose result you already have.
- Tool names, arguments and result shapes differ from store to store. Use exactly the argument names, types and allowed values the schemas give. Learn from the first results which tool shows which facts (price, stock, category, properties).
- A listing has an id, a price, a stock flag, a category or shelf label, a title, a short text, and a table of properties (facts). Search results usually show only the summary; a details tool shows the properties. Some stores have a compare tool that shows the properties of several listings in one step: use it when it exists.
- Search tools can be narrow: a capped page size, pages or tokens to continue, every word required, filters only in one mode. Compare the reported total with the rows you can see; narrow by category, price or words, and page on, until you have seen the relevant set. Short queries (the product noun plus one distinctive word) find more than long ones.
- The item must belong to the category the request names, read as a whole (a request for "A or B" fits a category called "A & B", not a narrower neighbour).

HOW TO JUDGE A LISTING
""" + JUDGE_RULES + """
- A seller note can redirect a property to another listing: then the listing's effective value for that property is the value shown on the referenced listing, which must be opened before judging.
- A shopper's answer can define a value through another listing (identified by its seller, category and some of its properties): find that listing, read the property there, and use that value, or exclude it when the shopper excludes it.
- Wishes are not requirements: use them only to choose among items that already fit, never over a cheapest rule.
- Never invent a requirement from the story. Such details are at most hints about what to ask the shopper.

CHEAPEST OR BEST
- When the request asks for the cheapest, the order must be the lowest-priced item among ALL items that meet every requirement, not the first fitting item you saw. Use price filters when they exist, page through the results, and check cheaper candidates' properties before settling. Keep the next fitting items as backups.
- When the request asks for the best by some measure (largest, longest, most, highest ...), compare all fitting items on that measure and take the extreme; an item whose properties do not state the measure cannot win.

THE SHOPPER
- The shopper has needs they did not write down and reveals one only when a message asks about that exact property. The harness plans and sends these questions for you at the right moments; the shopper's answers are then shown to you with their meaning, and every revealed need is a hard requirement from then on.
- Use the message tool yourself only when the request's procedure requires telling or asking the shopper something, or to answer a question the shopper asked you. Every message must raise exactly ONE point in one short plain sentence, never two topics, never a report of what you found.

CHANGES
- While you work, the shopper may change things: a new budget (higher or lower), a requirement dropped or added, a price going up or down, an item selling out. Such lines can arrive inside any tool result or attached to an order attempt. Each change is binding from that moment: re-check your pick and your backups against it.
- An order attempt can be REFUSED: the result then carries the shopper's line (often under a key such as refusal), even if the inner part looks successful. Nothing has been bought while the episode continues. Apply every change the line states, fix the cart (remove a sold-out or no longer acceptable line, add the next verified fit), re-read the new pick's price and stock, and order again.

FINISHING
- Before an order: the cart holds exactly the intended line(s) at the right quantity and nothing else; you have re-read the chosen item's price and stock after your last cart change; its record meets every current requirement; no cheaper fitting item exists when cheapest was asked; every procedural rule has been followed.
- An item whose record leaves a requirement unknown is not a verified fit: it is a candidate to read further or to replace with one that states the value. Order an item with an unknown requirement only when no listing of this kind states that property at all; when, after a thorough search, no item meets every requirement, decline.
- Get a first order attempt in while at least seven steps remain, so a refusal can still be handled. Do not swap the cart back and forth: every cart change costs a step.
- Decline only after searching several ways and finding that every candidate clearly fails a requirement, the budget or stock. A wrong decline scores 0 just like a wrong order; when a verified fit exists, order it.
- Every reply must contain at least one tool call. Calls that do not depend on each other's results (opening several listings, running several searches, carting several products) go in the same reply, since each model turn costs wall-clock time and the task has a time limit as well as a step limit. Keep the text short: what you now require, which candidates fit (id, price, stock) and your next move."""

SETUP_PROMPT = """You prepare a shopping agent for one task. Read the shopper's request and the store's tools, then return ONE JSON object and nothing else, with exactly these keys:

"items": a list with one object per DIFFERENT product requested (usually one): {"product": short name in the request's words, "category_words": the words of the product range or category as the request puts them, "quantity": integer units of one listing (1 unless the request asks for several of this product; a pair or a set is 1), "requirements": [strings "property: constraint" with the shopper's exact value, limit, exclusion or closed list], "wishes": [soft preferences phrased as ideally, preferably, would be nice, if possible]}.
  A requirement is any detail the request states as a condition on the product, whether phrased as a demand, as a fact about the item wanted, as an exclusion or closed list, or as something the shopper will check or verify themselves. A purpose, occasion or intended use is a requirement only when it is stated as such a condition; when it merely explains why the shopper is buying, it is story. Each stated condition is one requirement: a remark that the shopper will check it adds no second requirement. Do not include price, stock, quantity, delivery or timing (stores list no such facts), words that name the kind of product rather than a property (they belong to the category), or anything the story merely implies. Only soft wording (ideally, preferably, would be nice, if possible, a bonus) makes a wish.
"budget": the maximum total price as a number, or null when no limit is stated.
"budget_unit": the price unit the request uses (for example cents, dollars, PHP), or null.
"in_stock_only": true when the request wants only items on hand / in stock.
"objective": "cheapest" when the request asks for the cheapest or lowest-priced fitting option; "best" when it asks for the best by some measure; otherwise "any". A remark about how much the budget matters is not a cheapest rule.
"objective_measure": for "best", what is measured; otherwise "".
"message_budget": the number of messages the shopper allows, as an integer, or null.
"step_budget": the number of steps or moves the request allows, as an integer, or null.
"decline_rule": the request's own words on what to do when nothing fits, or "".
"no_substitutes": true when the decline rule forbids offering, naming or suggesting alternatives; false otherwise.
"rules": every procedural rule the request states for the assistant, quoted word for word: how many listings to look at or compare before choosing, what to check, what to tell or ask the shopper and when, what to put in a remark or message, how to decline, what not to do. Do not list here the budget, the stock rule, the message or step limits, the cheapest/best rule or a requirement (they have their own keys), nor remarks about what the shopper will do themselves, greetings or asides. When unsure whether a sentence is a rule for the assistant, include it.
"hints": [{"property": a product property the story points at without stating a value, named as a listing would name it, "why": the request's words}], most promising first; empty when the request is plain. Hints are what to ask the shopper about first; they are never requirements.
"searches": [{"product": the product's name as above, "queries": [up to 3 short phrases to type into the store's search, each the product noun or its category plus ONE distinctive requirement value, as a shopper would type; never a sentence]}], one entry per product.
"ask_first": [plain one-sentence questions the agent must ask the shopper before choosing: the maximum price when no budget is stated; which of two wishes matters more when the request says both may not be possible together (name both); anything the request says to check with the shopper first]. Empty when none.
"roles": {"search": name, "details": name, "compare": name or null, "message": name or null, "cart_add": name or null, "cart_remove": name or null, "cart_view": name or null, "status": name or null, "order": name, "decline": name or null} using exact tool names (null when no tool fits). "message" sends text to the shopper; "cart_add" puts a listing in the cart (it may also remove, through an argument); "order" buys the cart; "decline" ends without buying (it may be the same tool as order, chosen by an argument).
"finish_args": when one tool both orders and declines, {"order": {argument: value}, "decline": {argument: value}} naming the argument values that select each; otherwise null.
"confirm_before_order": true when a rule asks the assistant to show, present, run by, read back or confirm the chosen item(s) or the basket with the shopper, or to get their go-ahead, before ordering; false otherwise.
"messages_required": true when a rule requires the assistant to send the shopper a message during the task, before or apart from the final order or decline (a report, a confirmation, an answer, something to tell them first); false otherwise. What to say when declining or ordering is the finish itself (decline_rule and rules), not a message.
Every sentence of the request must land somewhere: a requirement, a wish, the budget, a limit, the objective, a rule, a hint, ask_first, or pure story."""

PLAN_PROMPT = """You choose what a shopping agent asks the shopper. The shopper has hidden needs: each is a value of one property of the product, revealed only when a message asks about that very property. A message is read for the FIRST property it raises; a message that raises two things, or none, annoys the shopper and wastes a message. Properties the shopper does not care about get a "no preference" answer, which costs nothing but the message.

Choose up to COUNT different properties to ask about, most promising first:
1. properties the request's story points at (listed as hints) that are not settled by a stated requirement;
2. properties that most of the listings seen carry and whose values differ between listings;
3. never: price, stock, quantity, brand, maker, seller, shop, condition, warranty, delivery, a property the request already settles, or a property already asked. Never two near-duplicate properties.
Name each property as the listings name it (spaces instead of underscores). Write each question exactly as "Which <property> would you like the <product> to have?" with the product named by its plain noun from the request (no adjectives or values), or, for a numeric property, "What <property> would you like the <product> to have?". No example values, no second clause.
When several different products are requested, spread the questions over the products and name the product in each question.
Return ONE JSON object and nothing else: {"questions": [{"product": "...", "property": "...", "question": "..."}]}"""

INTERPRET_PROMPT = """You read what a shopper just said to a shopping agent and report its meaning. Return ONE JSON object and nothing else, with these keys:
"kind": one of "need" (the shopper states a firm value, limit, exclusion or choice for a property), "no_preference" (they do not mind about the property asked), "unclear" (they did not understand or ask the agent to repeat or simplify, or to ask one thing at a time), "change" (a new budget, a dropped or replaced requirement, a changed quantity, an item sold out or re-priced), "question" (the shopper asks the agent something or asks for a confirmation), "other".
"needs": [{"product": the number of the product in the PRODUCTS list the need is about (0 when only one product is requested or the need applies to all of them), "need": "property: constraint" with the exact value, limit (keep units and words like at least / at most), exclusion ("not X") or closed list the shopper gave}]; empty otherwise.
"no_preference_about": [properties the shopper said they do not care about].
"dropped": [earlier requirements the shopper no longer wants, each copied word for word from the REQUIREMENTS SO FAR list].
"new_budget": the new maximum total price as a number (same unit as before), or null.
"new_quantity": {"product": name or "", "quantity": integer} or null.
"unavailable": true when the line says a picked item is gone, sold out or can no longer be bought.
"price_changed": true when the line says a picked item's price changed.
"cart_line": when unavailable or price_changed is true and several cart lines are listed, the number of the CART LINES entry the shopper refers to (by position, name or kind); null when unclear or only one line exists.
"reference": when the shopper defines the value indirectly through another listing, identified by its seller, category and some of its properties, either to copy that listing's value or to exclude it, {"property": ..., "mode": "same" or "exclude", "seller": ..., "category": ..., "facts": {property: value}} describing the listing to look up; otherwise null.
"reply": when kind is "question", a one-sentence plain answer to send back; otherwise "".
"meaning": one short sentence.
Read the shopper literally and charitably: folksy, figurative or roundabout phrasing that names a value means they firmly want that value. Decide from the logic of the sentence, not from the presence of a negative word, whether the shopper insists on a value (treats it as fixed, refuses to compromise on it, accepts nothing else: the need is that value) or excludes it (refuses the value itself: the need is "not" that value). A number with a comparison (at least, at most, under, over, no more than) is a limit; keep its unit. A new budget replaces the old one whether higher or lower, even when called good news. A "no preference" answer applies only to the property asked and never drops a stated requirement."""

VERIFY_PROMPT = """You are the final checker for a shopping agent that is about to ACTION. The task scores 1 only if the finish is fully right. Judge only from the records shown.
""" + JUDGE_RULES + """
Further rules: price times quantity must be within the budget (equal is fine); a listing shown out of stock fails when stock is required; the listing must sit in the requested category, read as a whole. When several products are requested, judge each listing only against its own product's requirements.
Return ONE JSON object and nothing else:
{"verdicts": [{"id": listing id, "fits": true | false | "unknown",
               "failed": [{"requirement": the requirement, "key": the property key, exactly as written in the record, whose value contradicts it, "value": that value}],
               "unknown": [requirements nothing in the record speaks to]}],
 "cheaper_fits": [ids of OTHER listings in the records that appear to fit and are cheaper than the chosen one],
 "worth_opening": [ids of unopened cheaper rows in the right category whose properties should be read before ordering, when the cheapest is asked for; at most 4],
 "rules": [{"rule": text, "satisfied": true | false | "unknown", "needs_message": true when following it requires sending the shopper a message, "why": short}],
 "advice": one sentence on what to do if anything is wrong}
A failed entry must quote a real key and value from the record; without one, the requirement goes under unknown. Keep every string under twelve words."""

RESULT_MAP_PROMPT = """A shopping agent received the result below from one tool of an online store. Name which keys play which roles, so the agent can read every later result of this tool. Keys are shown as paths with "." between levels and "[]" for list items; name a key by its last segment exactly as written. Return ONE JSON object and nothing else, with these keys (null or [] when the result has no such key; never guess):
"rows": the list whose items are listings (products), or null;
"id": the key, inside a listing, holding the listing's identifier;
"price": the key holding a listing's unit price;
"stock": the key saying whether a listing can be bought; "in_stock_values": [the values of that key meaning it can be bought];
"category": the key holding the store category, department or shelf label;
"title": the key holding the listing's name;
"brand": the key holding the maker or brand; "seller": the key holding the shop or seller;
"text": [keys holding descriptions, seller notes or remarks about a listing];
"facts": the key holding a listing's table of properties (specifications);
"quantity": the key holding how many units of a listing a cart line has;
"total": the key holding the cart's total price;
"count": the key holding how many listings matched a search; "more": the key used to fetch the next page;
"shopper_lines": [keys, at any level, whose values are messages from the shopper to the agent: sentences, notes, refusals];
"error": [keys that report a failure or refusal of the call];
"reply": the key holding the shopper's answer when this tool sends the shopper a message."""

PRODUCT_MAP_PROMPT = """A shopper asked a shopping agent for one or more products. The store sorts its listings under category labels. For each store category label below, say which requested product (by its number) a listing under that label would be, or 0 when listings of that label are none of the requested products. A label fits a product when the product, as the shopper describes it, is the kind of thing sold under that label; a label that merely shares a word does not fit, and a label for a different kind of thing does not fit even when the product could be described by its words. When the shopper names a range with several alternatives or words and one label covers that whole range, only that label fits; a label naming just part of the range does not.
Return ONE JSON object and nothing else: {"labels": {"<label exactly as written>": number}}"""

DECLINE_PROMPT = """A shopping agent wants to tell the shopper that nothing in the store fits. A wrong decline scores 0, so check carefully. Using the listing records and the rules below, name every listing that may meet every current requirement (within budget, not shown out of stock, in the requested category), and say whether the search was thorough.
""" + JUDGE_RULES + """
A listing whose requirements are met or unknown may fit; only a listing with a clearly different stated value, a price over budget, an out-of-stock flag or a wrong category label cannot.
Return ONE JSON object and nothing else: {"possible_fits": [ids, best first, at most 5], "worth_opening": [unopened ids worth reading first, at most 4], "search_more": "a concrete search still worth doing, or empty", "reason": "one sentence"}"""


# ---------------------------------------------------------------------------
# small helpers

_ID_SHAPE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-:.]{2,63}$")
_ID_KEY = re.compile(r"(^|[_\s])(id|ids|sku|code|ref|key|number|pid)([_\s]|$)|product_?id|listing_?id|item_?id|_id$|^id", re.I)
_PRICE_KEY = re.compile(r"price|cost|amount", re.I)
_TOTAL_KEY = re.compile(r"total|sum|running|due|subtotal", re.I)
_STOCK_KEY = re.compile(r"stock|avail|on_?hand|can_?buy|purchas|orderable|hand", re.I)
_CATEGORY_KEY = re.compile(r"categor|department|shelf|aisle|section|dept|range|family|group", re.I)
_TITLE_KEY = re.compile(r"title|name|heading|headline|caption|label", re.I)
_BRAND_KEY = re.compile(r"brand|make$|maker|manufacturer|marque|company", re.I)
_SELLER_KEY = re.compile(r"seller|shop|store|merchant|vendor|outlet|sold_?by", re.I)
_TEXT_KEY = re.compile(r"desc|summary|overview|blurb|about|text|body|detail", re.I)
_NOTE_KEY = re.compile(r"note|memo|good_to_know|remark|advice|comment|info", re.I)
_FACTS_KEY = re.compile(r"fact|propert|trait|spec|attribut|feature|detail|table|characteristic|param", re.I)
_QTY_KEY = re.compile(r"qty|quantity|count|units|number|how_many|wanted|needed|line_count|amount_total", re.I)
_ERROR_KEY = re.compile(r"error|issue|problem|fail|invalid|refus|denied|reject", re.I)
_REFUSAL_KEY = re.compile(r"refus|object|complain|reject|declin", re.I)
_ADD_WORD = re.compile(r"add|put|include|insert|hold|keep|reserve|plus|increase|set|take|stash", re.I)
_REMOVE_WORD = re.compile(r"remov|releas|drop|minus|delet|discard|clear|decreas|take_?out|exclude", re.I)
_ORDER_WORD = re.compile(r"order|buy|purchase|checkout|check_?out|settle|pay|commit|place|confirm|finali|submit|complete|wrap", re.I)
_DECLINE_WORD = re.compile(r"declin|refus|abandon|give_?up|stand_?down|withdraw|quit|stop|cancel|no_?purchase|nothing|without|walk|abort|skip", re.I)
_CART_WORD = re.compile(r"cart|bag|basket|trolley|stash|hold|items|lines|selection", re.I)
_FENCE = re.compile(r"^```[A-Za-z]*\s*|\s*```$")

# Result keys named by the reader model (per episode), consulted before the key regexes above.
_HINT_ROLES = ("id", "price", "stock", "category", "title", "brand", "seller", "text", "facts", "quantity", "total", "count", "more", "shopper_lines", "error", "reply", "rows")
_HINTS: dict[str, set[str]] = {role: set() for role in _HINT_ROLES}
_HINTS["in_stock_values"] = set()


def _reset_hints() -> None:
    for role in _HINTS:
        _HINTS[role] = set()


def _hinted(role: str, key: Any) -> bool:
    return str(key) in _HINTS[role]


def _hinted_any(key: Any) -> str | None:
    """The role the model gave this key, if any."""
    text = str(key)
    for role in _HINT_ROLES:
        if text in _HINTS[role]:
            return role
    return None


def _now() -> float:
    return time.monotonic()


def _compact(value: Any, limit: int = OBS_CHARS) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        text = str(value)
    if len(text) > limit:
        text = text[:limit] + "...(cut)"
    return text


def _short(text: Any, limit: int) -> str:
    text = str(text if text is not None else "")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _number(value: Any) -> float | None:
    """A number from a number or a numeric string such as '1901', '19.01', '$19.01', '1,901'."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, str):
        text = value.strip().replace(",", "")
        match = re.fullmatch(r"[^\d\-]{0,3}(-?\d+(?:\.\d+)?)\s*[A-Za-z%]{0,6}", text)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None
    return None


def _truthy(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value > 0
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("true", "yes", "y", "1", "in stock", "in_stock", "available", "on hand", "on_hand", "ok"):
            return True
        if text in ("false", "no", "n", "0", "out of stock", "out_of_stock", "unavailable", "sold out", "none", "backorder"):
            return False
    return None


def _parse_json(text: Any) -> dict[str, Any] | None:
    """The largest complete JSON object in a model reply (models sometimes add a smaller example after it)."""
    if not isinstance(text, str) or not text.strip():
        return None
    text = _FENCE.sub("", text.strip())
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except ValueError:
        pass
    decoder = json.JSONDecoder()
    found = None
    index = text.find("{")
    while 0 <= index < len(text):
        try:
            value, stop = decoder.raw_decode(text, index)
        except ValueError:
            index = text.find("{", index + 1)
            continue
        if isinstance(value, dict) and (found is None or len(json.dumps(value)) > len(json.dumps(found))):
            found = value
        index = text.find("{", stop)
    return found


def _sentences(value: Any, out: list[str], depth: int = 0) -> None:
    if depth > 4:
        return
    if isinstance(value, str):
        if len(value.split()) >= 4:
            out.append(value)
    elif isinstance(value, list):
        for item in value[:12]:
            _sentences(item, out, depth + 1)


def _words(name: str) -> set[str]:
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(name))
    return {w.lower() for w in re.split(r"[^A-Za-z0-9]+", spaced) if w}


# ---------------------------------------------------------------------------
# argument checking against the world's schema

def _type_ok(value: Any, kind: str) -> bool:
    if kind == "string":
        return isinstance(value, str)
    if kind == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if kind == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if kind == "boolean":
        return isinstance(value, bool)
    if kind == "object":
        return isinstance(value, dict)
    if kind == "array":
        return isinstance(value, list)
    if kind == "null":
        return value is None
    return True


def _coerce(value: Any, schema: dict[str, Any]) -> Any:
    kinds = schema.get("type")
    if isinstance(kinds, str):
        kinds = [kinds]
    if not kinds:
        return value
    if any(_type_ok(value, k) for k in kinds):
        if "integer" in kinds and isinstance(value, float) and value.is_integer():
            return int(value)
        return value
    for kind in kinds:
        try:
            if kind == "integer":
                if isinstance(value, float) and value.is_integer():
                    return int(value)
                if isinstance(value, str) and re.fullmatch(r"\s*-?\d+(\.0+)?\s*", value):
                    return int(float(value))
            if kind == "number" and isinstance(value, str) and re.fullmatch(r"\s*-?\d+(\.\d+)?\s*", value):
                number = float(value)
                return int(number) if number.is_integer() else number
            if kind == "string" and isinstance(value, (int, float)) and not isinstance(value, bool):
                return str(value)
            if kind == "boolean" and isinstance(value, str) and value.strip().lower() in ("true", "false"):
                return value.strip().lower() == "true"
            if kind == "array" and isinstance(value, str):
                stripped = value.strip()
                if stripped.startswith("["):
                    parsed = json.loads(stripped)
                    if isinstance(parsed, list):
                        return parsed
                return [part.strip() for part in stripped.split(",") if part.strip()]
            if kind == "object" and isinstance(value, str) and value.strip().startswith("{"):
                parsed = json.loads(value)
                if isinstance(parsed, dict):
                    return parsed
        except (TypeError, ValueError):
            continue
    return value


def _check(value: Any, schema: dict[str, Any], path: str) -> tuple[Any, list[str]]:
    """Coerce and validate a value against a JSON schema fragment; returns (value, problems)."""
    if not isinstance(schema, dict):
        return value, []
    problems: list[str] = []
    value = _coerce(value, schema)
    kinds = schema.get("type")
    if isinstance(kinds, str):
        kinds = [kinds]
    if kinds and not any(_type_ok(value, k) for k in kinds):
        return value, [f"{path}: expected {'/'.join(kinds)}"]
    if "enum" in schema and value not in schema["enum"]:
        if isinstance(value, str):
            for option in schema["enum"]:
                if isinstance(option, str) and option.strip().lower() == value.strip().lower():
                    value = option
                    break
        if value not in schema["enum"]:
            problems.append(f"{path}: must be one of {schema['enum']}")
    if "const" in schema and value != schema["const"]:
        problems.append(f"{path}: must equal {schema['const']!r}")
    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            problems.append(f"{path}: needs at least {schema['minLength']} characters")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            cut = value[: schema["maxLength"]]
            value = cut.rsplit(" ", 1)[0] if " " in cut.strip() and len(cut) > 40 else cut
        if "pattern" in schema:
            try:
                if not re.search(schema["pattern"], value):
                    problems.append(f"{path}: does not match the required pattern")
            except re.error:
                pass
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            problems.append(f"{path}: below the minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            problems.append(f"{path}: above the maximum {schema['maximum']}")
        if isinstance(schema.get("exclusiveMinimum"), (int, float)) and value <= schema["exclusiveMinimum"]:
            problems.append(f"{path}: must be above {schema['exclusiveMinimum']}")
        if isinstance(schema.get("exclusiveMaximum"), (int, float)) and value >= schema["exclusiveMaximum"]:
            problems.append(f"{path}: must be below {schema['exclusiveMaximum']}")
    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            problems.append(f"{path}: needs at least {schema['minItems']} items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            problems.append(f"{path}: allows at most {schema['maxItems']} items")
        if isinstance(schema.get("items"), dict):
            fixed = []
            for index, item in enumerate(value):
                item, sub = _check(item, schema["items"], f"{path}[{index}]")
                fixed.append(item)
                problems += sub
            value = fixed
    if isinstance(value, dict):
        properties = schema.get("properties") or {}
        for name in schema.get("required") or []:
            if name not in value or value.get(name) is None:
                problems.append(f"{path}.{name}: required")
        fixed = {}
        for name, item in value.items():
            if item is None and name not in (schema.get("required") or []):
                continue
            if name in properties:
                item, sub = _check(item, properties[name], f"{path}.{name}")
                problems += sub
            elif schema.get("additionalProperties") is False and properties:
                problems.append(f"{path}.{name}: not an argument of this tool (arguments: {', '.join(properties)})")
                continue
            fixed[name] = item
        value = fixed
    return value, problems


# ---------------------------------------------------------------------------
# reading observations

class Listing:
    """What the store has shown about one listing."""

    def __init__(self, ident: str):
        self.id = ident
        self.price: float | None = None
        self.in_stock: bool | None = None
        self.category: str = ""
        self.title: str = ""
        self.brand: str = ""
        self.seller: str = ""
        self.facts: dict[str, Any] = {}
        self.notes: list[str] = []
        self.opened = False
        self.record: dict[str, Any] = {}
        self.first_seen = 0
        self.last_seen = 0

    def summary(self) -> str:
        bits = [self.id, f"price {self.price:g}" if self.price is not None else "price ?"]
        bits.append("in stock" if self.in_stock else ("OUT OF STOCK" if self.in_stock is False else "stock ?"))
        if self.category:
            bits.append(f"[{_short(self.category, 40)}]")
        if self.title:
            bits.append(_short(self.title, 70))
        return " | ".join(bits)

    def full(self, limit: int = 1800) -> str:
        record: dict[str, Any] = {"id": self.id, "price": self.price if self.price is not None else "?", "in_stock": "yes" if self.in_stock else ("NO" if self.in_stock is False else "?")}
        if self.category:
            record["category_label"] = self.category
        if self.title:
            record["title"] = self.title
        record.update({k: v for k, v in self.record.items() if k not in record})
        if self.facts:
            record["facts"] = self.facts
        if self.notes:
            record["notes"] = self.notes[:3]
        return _compact(record, limit)


def _is_listing(obj: dict[str, Any]) -> str | None:
    """The id of a listing-like dict (an id plus a price, a title or a facts table), else None."""
    ident = None
    for key, value in obj.items():
        if isinstance(value, str) and value.strip() and _hinted("id", key):
            ident = value.strip()
            break
    if ident is None:
        for key, value in obj.items():
            if isinstance(value, str) and _ID_KEY.search(str(key)) and _ID_SHAPE.match(value) and any(c.isdigit() for c in value) and _hinted_any(key) in (None, "id"):
                ident = value
                break
    if ident is None:
        return None
    has_more = False
    for key, value in obj.items():
        if value == ident:
            continue
        role = _hinted_any(key)
        if role in ("price", "title", "facts", "quantity"):
            has_more = True
        elif role is None and _PRICE_KEY.search(str(key)) and _number(value) is not None:
            has_more = True
        elif role is None and _TITLE_KEY.search(str(key)) and isinstance(value, str) and value.strip():
            has_more = True
        elif isinstance(value, dict) and value and all(not isinstance(v, (dict, list)) for v in value.values()):
            has_more = True
    return ident if has_more else None


def _facts_of(obj: dict[str, Any]) -> dict[str, Any]:
    best: dict[str, Any] = {}
    for key, value in obj.items():
        if _hinted("facts", key) and isinstance(value, dict) and value:
            return {str(k): v for k, v in value.items()}
    for key, value in obj.items():
        if isinstance(value, dict) and value and all(not isinstance(v, (dict, list)) for v in value.values()):
            if len(value) > len(best) or _FACTS_KEY.search(str(key)):
                best = {str(k): v for k, v in value.items()}
        elif isinstance(value, list) and value and all(isinstance(v, dict) and len(v) == 2 for v in value):
            pairs: dict[str, Any] = {}
            for pair in value:
                items = list(pair.values())
                if isinstance(items[0], str):
                    pairs[items[0]] = items[1]
            if len(pairs) > len(best):
                best = pairs
    return best


def _walk_listings(obj: Any, depth: int = 0) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    if depth > 6:
        return found
    if isinstance(obj, dict):
        if _is_listing(obj):
            found.append(obj)
        for value in obj.values():
            if isinstance(value, (dict, list)):
                found.extend(_walk_listings(value, depth + 1))
    elif isinstance(obj, list):
        for value in obj:
            found.extend(_walk_listings(value, depth + 1))
    return found


def _loose_lines(obs: Any) -> list[str]:
    """Sentences riding on a result outside any listing: the shopper's lines and refusals."""
    out: list[str] = []

    def walk(value: Any, depth: int, inside_listing: bool) -> None:
        if depth > 5:
            return
        if isinstance(value, dict):
            listing = _is_listing(value) is not None
            for key, item in value.items():
                if _hinted("shopper_lines", key) and isinstance(item, str) and item.strip():
                    out.append(item.strip())
                elif isinstance(item, list) and item and all(isinstance(v, str) for v in item):
                    if (not listing and not inside_listing) or _hinted("shopper_lines", key):
                        _sentences(item, out)
                elif isinstance(item, str) and not listing and not inside_listing and _REFUSAL_KEY.search(str(key)) and len(item.split()) >= 4 and _hinted_any(key) in (None, "shopper_lines", "error"):
                    out.append(item)
                elif isinstance(item, str) and item.startswith('["') and not listing:
                    try:
                        parsed = json.loads(item)
                    except ValueError:
                        parsed = None
                    if isinstance(parsed, list):
                        _sentences(parsed, out)
                elif isinstance(item, (dict, list)):
                    walk(item, depth + 1, inside_listing or listing)
        elif isinstance(value, list):
            if value and all(isinstance(v, str) for v in value):
                if not inside_listing:
                    _sentences(value, out)
            else:
                for item in value:
                    walk(item, depth + 1, inside_listing)

    walk(obs, 0, False)
    seen: list[str] = []
    for line in out:
        if line not in seen:
            seen.append(line)
    return seen


def _result_failed(obs: Any) -> bool:
    if isinstance(obs, dict):
        for key, value in obs.items():
            if _hinted("error", key) and value not in (None, False, "", [], {}):
                return True
            if _hinted_any(key) not in (None, "error"):
                continue
            if _ERROR_KEY.search(str(key)) and value not in (None, False, "", [], {}):
                return True
            if isinstance(value, str) and re.fullmatch(r"[A-Z][A-Z0-9_]{3,}", value) and value not in ("OK", "SUCCESS", "DONE", "ADDED", "REMOVED", "UPDATED", "ACCEPTED", "TRUE", "PLACED"):
                return True
    return False


def _compact_obs(obs: Any) -> Any:
    """The result as the model sees it: long marketing text cut, facts and notes kept."""

    def shrink(value: Any, depth: int) -> Any:
        if depth > 7:
            return None
        if isinstance(value, dict):
            listing = _is_listing(value) is not None
            out: dict[str, Any] = {}
            for key, item in value.items():
                if isinstance(item, str):
                    limit = 400
                    if listing and _TEXT_KEY.search(str(key)) and not _NOTE_KEY.search(str(key)):
                        limit = 110
                    elif listing and _NOTE_KEY.search(str(key)):
                        limit = 260
                    out[key] = _short(item, limit)
                else:
                    out[key] = shrink(item, depth + 1)
            return out
        if isinstance(value, list):
            rows = value[:14] if all(isinstance(v, dict) for v in value) else value[:40]
            return [shrink(v, depth + 1) for v in rows]
        return value

    return shrink(obs, 0)


_STOP_WORDS = {"the", "a", "an", "of", "for", "in", "on", "and", "or", "be", "is", "must", "should", "has", "have", "it", "its", "with", "to", "as", "at", "only", "exactly", "that", "nothing", "else", "no"}


def _stem(word: str) -> str:
    """A plural reduced to its singular, enough to compare product nouns (shelves/shelf, watches/watch, dresses/dress)."""
    if len(word) > 3 and word.endswith("fe") and not word.endswith("ffe"):
        return word[:-2] + "ve"   # knife -> knive, matching knives -> knive
    if len(word) > 3 and word.endswith("f") and not word.endswith("ff"):
        return word[:-1] + "ve"   # shelf -> shelve, matching shelves -> shelve
    if len(word) <= 3 or not word.endswith("s") or word.endswith("ss"):
        return word
    if word.endswith("ves") and len(word) > 5:
        return word[:-1]          # sleeves -> sleeve, shelves -> shelve
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith(("ches", "shes", "sses", "xes", "zes")) and len(word) > 4:
        return word[:-2]
    return word[:-1]


def _plain_words(text: str) -> set[str]:
    out = set()
    for word in re.findall(r"[a-z0-9]+", str(text).lower().replace("_", " ").replace(".", " ")):
        if word in _STOP_WORDS or len(word) < 2:
            continue
        out.add(_stem(word))
    return out


def _label_states(listing: "Listing", requirement: str) -> bool:
    """Whether the listing's own category label states the requirement's value."""
    return bool(listing.category) and _evidence(listing, requirement)[1].startswith("category label: ")


def _evidence(listing: "Listing", requirement: str) -> tuple[str, str]:
    """How a listing's own facts relate to one requirement: ('stated' | 'other' | 'absent' | 'title', shown lines)."""
    prop, sep, value = requirement.partition(":")
    if not sep:
        prop, value = requirement, ""
    prop_words = _plain_words(prop) or {prop.lower()}
    # the wanted value: alternatives ("A or B", "A / B") without the exclusions ("not X", "nothing else")
    positive = re.split(r"\b(?:not|nothing|anything but|no|except|leave out|never)\b", value.lower(), maxsplit=1)[0]
    alternatives = [_plain_words(a) for a in re.split(r"\bor\b|/|,|;", positive) if _plain_words(a)]
    value_words = _plain_words(value)
    ordered = [w for w in re.findall(r"[a-z0-9]+", prop.lower()) if w not in _STOP_WORDS]
    head = _stem((ordered[0] if " of " in prop.lower() else ordered[-1]) if ordered else "")
    matched: list[tuple[str, str]] = []
    for key, fact in listing.facts.items():
        key_words = _plain_words(key)
        if (head and head in key_words) or (prop_words & key_words):
            matched.append((str(key), str(fact)))
    matched.sort(key=lambda kv: -len(_plain_words(kv[0]) & prop_words) * 10 + len(_plain_words(kv[0]) - prop_words))
    shown = "; ".join(f"{k}: {_short(v, 60)}" for k, v in matched[:4])

    def states(text: str) -> bool:
        words = _plain_words(text)
        return any(alt <= words or all(w in text.lower() for w in alt) for alt in alternatives)

    if alternatives and listing.category and states(listing.category):
        return "stated", "category label: " + _short(listing.category, 60)
    if not matched:
        text = listing.title + " " + " ".join(listing.notes)
        if alternatives and states(text):
            return "title", "no property named for it; the title says: " + _short(listing.title, 80)
        return "absent", "no property named for it"
    if alternatives and any(states(f"{k} {v}") for k, v in matched):
        return "stated", shown
    best_overlap = max(len(_plain_words(k) & prop_words) for k, _ in matched)
    if best_overlap >= len(prop_words) or len(prop_words) == 1:
        return "other", shown
    return "unclear", shown


# ---------------------------------------------------------------------------
# the episode

class Episode:
    def __init__(self, problem_data: dict[str, Any]):
        self.started = _now()
        _reset_hints()
        environment = problem_data["environment"]
        self.binding = dict(environment["binding"])
        self.policy = environment["policy_view"]
        self.request = str(self.policy.get("query") or "")
        self.opening = self.policy.get("opening")
        self.tools = [t for t in (self.policy.get("tools") or []) if isinstance(t, dict) and isinstance(t.get("function"), dict) and t["function"].get("name")]
        self.schemas = {t["function"]["name"]: (t["function"].get("parameters") or {"type": "object", "properties": {}}) for t in self.tools}
        self.descriptions = {t["function"]["name"]: str(t["function"].get("description") or "") for t in self.tools}
        self.max_steps = int(self.policy.get("max_steps") or 30)
        self.finished = bool(self.policy.get("finished"))
        self.dead = False
        self.step = 0
        self.trace: list[dict[str, Any]] = []
        self.cost = 0.0
        self.llm_calls = 0
        self.models = _models()
        self.llm = ProxyClient(timeout=75, max_retries=1, retry_delay=1)
        self.env = ProxyClient(timeout=90, max_retries=2, retry_delay=1)

        # the brief
        self.plan: dict[str, Any] = {}
        self.roles: dict[str, str] = {}
        self.items: list[dict[str, Any]] = []
        self.budget: float | None = None
        self.budget_unit = ""
        self.in_stock_only = True
        self.objective = "any"
        self.message_budget: int | None = None
        self.rules: list[str] = []
        self.hints: list[dict[str, str]] = []
        self.ask_first: list[str] = []
        self.finish_args: dict[str, dict[str, Any]] = {}

        # the ledger
        self.revealed: list[str] = []
        self.revealed_product: dict[str, int] = {}   # revealed need -> index of the product it is about
        self.product_map: dict[str, int] = {}        # store category label (lower case) -> product index, -1 for none
        self.product_map_failures = 0
        self.key_map: dict[str, dict[str, Any]] = {}   # tool name -> roles of the keys in its results, named by the reader model
        self.key_map_tried: set[str] = set()
        self.dropped: list[str] = []
        self.no_preference: list[str] = []
        self.references: list[dict[str, Any]] = []
        self.changes: list[str] = []
        self.ledger_version = 0

        # listings, searches, cart
        self.listings: dict[str, Listing] = {}
        self.order_seen: list[str] = []
        self.searches: list[str] = []
        self.search_stats: list[tuple[int, float | None, Any]] = []   # (rows shown, matches reported, more-marker) per search
        self.search_plan: dict[int, list[str]] = {}                      # product index -> short search phrases from the setup parse
        self.cart: dict[str, int] = {}
        self.quantity_words: set[str] = set()   # words of the cart tool's integer arguments (e.g. pieces), naming quantity keys in results
        self.cart_dirty = True
        self.cart_changed_at = -1
        self.reread_at: dict[str, int] = {}

        # shopper exchanges
        self.messages_used = 0
        self.asked: list[tuple[str, str, str]] = []   # (property, question, reply)
        self.asked_properties: list[str] = []
        self.heard: list[str] = []
        self.pending_reply = ""
        self.confirm_before_order = False
        self.messages_required = False
        self.basket_told_at = -1
        self.last_line = ""
        self.last_line_is_reply = False
        self.planned = False
        self.plan_rounds = 0
        self.message_refused = False

        # gates
        self.order_attempts = 0
        self.refusals = 0
        self.refused_at = -1
        self.holds = {"order": 0, "decline": 0, "search": 0, "message": 0}
        self.verdicts: dict[str, dict[str, Any]] = {}
        self.last_verify_key = ""
        self.last_verify_ok = False
        self.verify_advice = ""           # an ungrounded doubt of the checker, shown with the order result
        self.preferred_told: set[str] = set()   # cart compositions already told about better-stated candidates
        self.recent_calls: list[str] = []
        self.history: list[dict[str, Any]] = []
        self.idle = 0

    # -- time, steps, cost ---------------------------------------------------

    def time_left(self) -> float:
        return TASK_SECONDS - (_now() - self.started)

    def steps_left(self) -> int:
        return self.max_steps - self.step

    def messages_left(self) -> int | None:
        if self.message_budget is None:
            return None
        return max(0, self.message_budget - self.messages_used)

    def note(self, kind: str, **data: Any) -> None:
        entry = {"kind": kind, "t": round(_now() - self.started, 1), "step": self.step}
        entry.update(data)
        self.trace.append(entry)

    # -- inference -----------------------------------------------------------

    def infer(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, models: list[str], max_tokens: int) -> dict[str, Any] | None:
        for model in models:
            left = self.time_left()
            if left < FORCE_SECONDS - 6:
                return None
            self.llm.timeout = int(max(8, min(75, left - FORCE_SECONDS + 10)))
            sent = messages
            if model.startswith("anthropic/") and len(messages) >= 2:
                sent = [dict(m) for m in messages]
                for index in (0, len(sent) - 1):
                    content = sent[index].get("content")
                    if isinstance(content, str) and content:
                        sent[index]["content"] = [{"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}]
            body: dict[str, Any] = {"model": model, "messages": sent, "temperature": 0, "max_tokens": max_tokens}
            if tools:
                body["tools"] = tools
                body["tool_choice"] = "auto"
            result = self.llm.post_verbose("/inference/chat/completions", body)
            self.llm_calls += 1
            data = result.data
            if isinstance(data, dict) and data.get("choices"):
                usage = data.get("usage") or {}
                cost = usage.get("cost") if isinstance(usage, dict) else None
                if isinstance(cost, (int, float)) and not isinstance(cost, bool):
                    self.cost += float(cost)
                message = (data["choices"][0] or {}).get("message") or {}
                if message.get("content") or message.get("tool_calls"):
                    return message
            error = result.error or {}
            self.note("inference_failure", model=model, status=error.get("status"), body=str(error.get("body") or "")[:200])
            if error.get("status") == 402:
                self.dead = True
                return None
        return None

    def ask_json(self, prompt: str, max_tokens: int, models: list[str], require: tuple[str, ...] = ()) -> dict[str, Any] | None:
        message = self.infer([{"role": "user", "content": prompt}], None, models, max_tokens)
        if not message:
            return None
        parsed = _parse_json(message.get("content") or "")
        if parsed is not None and require and not any(key in parsed for key in require):
            parsed = None
        if parsed is None and self.time_left() > WRAP_UP_SECONDS:
            message = self.infer([{"role": "user", "content": prompt + "\n\nReturn only the complete JSON object, nothing else."}], None, models[:1], max_tokens)
            parsed = _parse_json((message or {}).get("content") or "")
            if parsed is not None and require and not any(key in parsed for key in require):
                parsed = None
        return parsed

    def finish_reserve(self) -> int:
        """Steps kept for the finish: cart repair and a second order attempt, plus the re-search a cheapest pick needs after a refusal."""
        return FINISH_RESERVE + (4 if self.objective == "cheapest" else 0)

    def cost_cap(self) -> float:
        """Spend after which the extra model checks are skipped; grows with the number of products requested."""
        return COST_SOFT_USD * (1 + 0.6 * (len(self.items) - 1))

    def checks_allowed(self) -> bool:
        return self.cost < self.cost_cap() and self.time_left() > WRAP_UP_SECONDS

    # -- the world -----------------------------------------------------------

    def call(self, name: str, args: dict[str, Any]) -> dict[str, Any] | None:
        """One environment step. Returns the response, or None when the call could not be made."""
        payload: dict[str, Any] = dict(self.binding)
        for attempt in range(3):
            left = self.time_left()
            if left < 2 or self.finished:
                return None
            payload["step"] = self.step + 1
            payload["action"] = {"name": name, "args": args}
            self.env.timeout = int(max(3, min(90, left - 1)))
            result = self.env.post_verbose("/environment/call", payload)
            data = result.data
            if isinstance(data, dict):
                if isinstance(data.get("step"), int):
                    self.step = data["step"]
                else:
                    self.step += 1
                if data.get("finished"):
                    self.finished = True
                self.note("env", name=name, args=args, observation=_compact(data.get("observation"), 2500), finished=bool(data.get("finished")))
                return data
            error = result.error or {}
            status = error.get("status")
            body = str(error.get("body") or "")
            self.note("env_failure", name=name, status=status, body=body[:200])
            if status == 409:
                if "finish" in body.lower() or "ended" in body.lower():
                    self.finished = True
                    return None
                match = re.search(r"expected\D{0,12}(\d+)", body)
                if match:
                    self.step = int(match.group(1)) - 1
                    continue
                return None
            if status == 429:
                time.sleep(1.5 + attempt)
                continue
            if status in (400, 403, 422):
                return None
        return None

    # -- setup -----------------------------------------------------------------

    def tool_lines(self) -> str:
        lines = []
        for tool in self.tools:
            function = tool["function"]
            lines.append(f"- {function['name']}: {_short(function.get('description') or '', 300)} | parameters: {_compact(function.get('parameters') or {}, 900)}")
        return "\n".join(lines)

    def setup(self) -> None:
        prompt = SETUP_PROMPT + "\n\nTHE STORE'S TOOLS:\n" + self.tool_lines() + "\n\nTHE SHOPPER'S REQUEST:\n" + self.request
        if self.opening is not None:
            prompt += "\n\nWHAT THE STORE SHOWED FIRST:\n" + _compact(self.opening, 3000)
        plan = self.ask_json(prompt, 2500, [self.models["setup"], self.models["setup_alt"]], require=("items", "roles")) or {}
        if not isinstance(plan.get("roles"), dict) and self.time_left() > 200:
            plan = self.ask_json(prompt, 2500, [self.models["setup_alt"], self.models["alt"]], require=("items", "roles")) or plan
        self.plan = plan
        self.note("plan", plan=plan)
        roles = plan.get("roles") if isinstance(plan.get("roles"), dict) else {}
        self.roles = {k: v for k, v in roles.items() if isinstance(v, str) and v in self.schemas}
        self.fill_roles()
        self.quantity_words = set()
        for role in ("cart_add", "cart_remove"):
            properties = (self.schemas.get(self.roles.get(role) or "") or {}).get("properties") or {}
            for key, spec in properties.items():
                if isinstance(spec, dict) and spec.get("type") in ("integer", "number"):
                    self.quantity_words |= _words(key)
        items = plan.get("items") if isinstance(plan.get("items"), list) else []
        self.items = []
        for item in items:
            if not isinstance(item, dict) or not str(item.get("product") or "").strip():
                continue
            quantity = item.get("quantity")
            if isinstance(quantity, str) and quantity.strip().isdigit():
                quantity = int(quantity)
            quantity = int(quantity) if isinstance(quantity, (int, float)) and not isinstance(quantity, bool) and quantity >= 1 else 1
            self.items.append({
                "product": str(item["product"]).strip(),
                "category_words": str(item.get("category_words") or "").strip(),
                "quantity": quantity,
                "requirements": [str(r).strip() for r in (item.get("requirements") or []) if str(r).strip()],
                "wishes": [str(w).strip() for w in (item.get("wishes") or []) if str(w).strip()],
            })
        if not self.items:
            self.items = [{"product": "the requested product", "category_words": "", "quantity": 1, "requirements": [], "wishes": []}]
        for item in self.items:
            seen_values: list[set[str]] = []
            kept = []
            for requirement in item["requirements"]:
                _, sep, value = requirement.partition(":")
                words = _plain_words(value if sep else requirement)
                if words and words in seen_values:
                    continue
                seen_values.append(words)
                kept.append(requirement)
            item["requirements"] = kept
        self.read_search_plan(plan)
        budget = _number(plan.get("budget"))
        self.budget = budget if budget is not None and budget > 0 else None
        self.budget_unit = str(plan.get("budget_unit") or "").strip()
        self.in_stock_only = bool(plan.get("in_stock_only", True))
        self.objective = plan.get("objective") if plan.get("objective") in ("cheapest", "best", "any") else "any"
        budget_messages = plan.get("message_budget")
        if isinstance(budget_messages, (int, float)) and not isinstance(budget_messages, bool) and 0 <= budget_messages <= 20:
            self.message_budget = int(budget_messages)
        steps = plan.get("step_budget")
        if isinstance(steps, (int, float)) and not isinstance(steps, bool) and 3 <= steps < self.max_steps:
            self.max_steps = int(steps)
        self.rules = [str(r).strip() for r in (plan.get("rules") or []) if str(r).strip()][:8]
        self.hints = [{"property": str(h.get("property") or "").strip().lower(), "why": str(h.get("why") or "")} for h in (plan.get("hints") or []) if isinstance(h, dict) and str(h.get("property") or "").strip()][:4]
        self.ask_first = [str(a).strip() for a in (plan.get("ask_first") or []) if str(a).strip()][:2]
        if self.budget is None and self.roles.get("message") and not any("spend" in a.lower() or "budget" in a.lower() or "price" in a.lower() for a in self.ask_first):
            self.ask_first.insert(0, "What is the most you would like to spend in total?")
        self.confirm_before_order = bool(plan.get("confirm_before_order"))
        self.messages_required = bool(plan.get("messages_required")) or self.confirm_before_order
        finish = plan.get("finish_args")
        if isinstance(finish, dict):
            self.finish_args = {k: v for k, v in finish.items() if k in ("order", "decline") and isinstance(v, dict)}

    def read_search_plan(self, plan: dict[str, Any]) -> None:
        """The short search phrases the setup model proposed, per product."""
        self.search_plan = {}
        entries = plan.get("searches") if isinstance(plan.get("searches"), list) else []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("product") or "").strip().lower()
            index = next((i for i, item in enumerate(self.items) if name and (name in item["product"].lower() or item["product"].lower() in name)), 0 if len(self.items) == 1 else None)
            if index is None:
                continue
            phrases = [str(q).strip() for q in (entry.get("queries") or []) if isinstance(q, (str, int, float)) and str(q).strip()][:3]
            if phrases and index not in self.search_plan:
                self.search_plan[index] = phrases

    def search_plan_lines(self) -> str:
        """Planned search phrases not yet typed into the store, per product (advice; each search costs a step)."""
        if not self.search_plan or not self.roles.get("search"):
            return ""
        done = [_plain_words(s) for s in self.searches]
        parts = []
        for index, phrases in self.search_plan.items():
            if index >= len(self.items):
                continue
            if any(i in self.listings and self.product_of(self.listings[i]) == index for i in self.intended_lines()):
                continue
            if self.plausible_candidates(index):
                continue
            pending = [p for p in phrases if not any(_plain_words(p) <= words for words in done)]
            if pending:
                parts.append(f"{self.items[index]['product']}: \"{pending[0]}\"")
        if not parts:
            return ""
        return "[search plan] no opened candidate of this product fits so far; a search phrase from the request not yet tried: " + " | ".join(parts)

    def fill_roles(self) -> None:
        """Roles the model did not name, guessed from tool names, descriptions and schemas."""
        decline_words = {"decline", "refuse", "reject", "abandon", "abort", "withdraw", "quit", "cancel", "skip", "standdown", "giveup", "nopurchase", "walkaway", "without", "forgo", "bail"}
        order_words = {"order", "buy", "purchase", "checkout", "settle", "pay", "commit", "place", "finalize", "finalise", "submit", "complete", "conclude", "wrap", "wrapup", "end", "finish", "done", "close"}
        message_words = {"message", "shopper", "customer", "buyer", "client", "ask", "chat", "say", "write", "note", "talk", "contact", "tell", "question", "speak", "reply", "converse"}
        search_words = {"search", "find", "hunt", "locate", "query", "browse", "look", "lookup", "seek", "sift", "filter", "shelf", "catalog", "catalogue", "explore", "discover", "scan", "listings"}
        details_words = {"detail", "details", "inspect", "open", "view", "read", "spec", "specs", "sheet", "datasheet", "info", "describe", "examine", "show", "fetch", "get", "peek", "listing", "item", "offer", "product"}
        compare_words = {"compare", "weigh", "match", "matchup", "versus", "contrast", "side"}
        cart_words = {"cart", "bag", "basket", "trolley", "stash", "hold", "keep", "selection", "reserve", "tray"}
        view_words = {"view", "show", "list", "review", "status", "summary", "contents", "see", "my", "current", "report", "check", "items"}
        status_words = {"status", "budget", "remaining", "left", "account", "counter", "quota", "allowance", "report"}
        add_words = {"add", "put", "include", "insert", "hold", "keep", "reserve", "set", "take", "stash", "pick"}
        remove_words = {"remove", "release", "drop", "delete", "discard", "clear", "exclude", "unhold"}
        taken = set(self.roles.values())

        def words(name: str) -> set[str]:
            spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name)
            parts = [w.lower() for w in re.split(r"[^A-Za-z0-9]+", spaced) if w]
            out = set(parts)
            for i in range(len(parts) - 1):
                out.add(parts[i] + parts[i + 1])
            return out

        def props(name: str) -> dict[str, Any]:
            return (self.schemas.get(name) or {}).get("properties") or {}

        def string_params(name: str) -> list[str]:
            return [k for k, v in props(name).items() if isinstance(v, dict) and v.get("type") == "string" and "enum" not in v]

        def one_text(name: str) -> bool:
            return len(string_params(name)) == 1 and len(props(name)) <= 2 and (props(name)[string_params(name)[0]].get("maxLength") or 999) >= 60

        def pick(role: str, test) -> None:
            if role in self.roles:
                return
            for tool in self.tools:
                name = tool["function"]["name"]
                if name in taken:
                    continue
                if test(name, words(name), words(self.descriptions.get(name, ""))):
                    self.roles[role] = name
                    taken.add(name)
                    return

        pick("decline", lambda n, w, d: bool(w & decline_words) and not (w & cart_words))
        pick("order", lambda n, w, d: bool(w & order_words) and not (w & cart_words) and not (w & message_words) and not (w & search_words) and not (w & decline_words))
        pick("order", lambda n, w, d: bool(d & {"checkout", "purchase", "buy", "order"}) and not (w & cart_words) and not (w & message_words) and not (w & search_words) and not props(n))
        pick("message", lambda n, w, d: bool(w & message_words) and one_text(n))
        pick("message", lambda n, w, d: bool(d & {"shopper", "customer", "message"}) and one_text(n))
        pick("compare", lambda n, w, d: bool(w & compare_words) or (any(isinstance(v, dict) and v.get("type") == "array" for v in props(n).values()) and bool((w | d) & details_words)))
        pick("cart_view", lambda n, w, d: bool(w & cart_words) and not props(n))
        pick("cart_view", lambda n, w, d: bool(d & cart_words) and bool(w & view_words) and not props(n))
        pick("status", lambda n, w, d: bool(w & status_words) and not props(n))
        pick("cart_remove", lambda n, w, d: bool(w & remove_words) and bool(props(n)) and not (w & search_words))
        pick("cart_add", lambda n, w, d: bool(w & (cart_words | add_words)) and bool(props(n)) and not (w & remove_words) and not (w & search_words) and not (w & details_words - {"item", "product", "listing"}))
        pick("cart_add", lambda n, w, d: bool(d & cart_words) and bool(props(n)) and not (w & search_words) and not (w & details_words - {"item", "product", "listing"}))
        pick("details", lambda n, w, d: bool(w & details_words) and len(string_params(n)) == 1 and len(props(n)) == 1 and not (w & search_words))
        pick("search", lambda n, w, d: bool(w & search_words) and bool(props(n)))
        pick("details", lambda n, w, d: bool((w | d) & details_words) and len(props(n)) == 1)
        pick("search", lambda n, w, d: bool(d & search_words) and bool(props(n)))
        if "decline" not in self.roles and "order" in self.roles:
            # one tool may both order and decline, through an enum argument
            for key, spec in props(self.roles["order"]).items():
                if isinstance(spec, dict) and spec.get("enum"):
                    self.roles["decline"] = self.roles["order"]
                    break
        if "order" not in self.roles and "decline" in self.roles:
            for key, spec in props(self.roles["decline"]).items():
                if isinstance(spec, dict) and spec.get("enum"):
                    self.roles["order"] = self.roles["decline"]
                    break

    def finish_mode(self, name: str, args: dict[str, Any]) -> str | None:
        """'order' or 'decline' for a call of a finishing tool, else None."""
        order, decline = self.roles.get("order"), self.roles.get("decline")
        if name not in (order, decline):
            return None
        if order != decline:
            return "order" if name == order else "decline"
        for mode in ("decline", "order"):
            want = self.finish_args.get(mode) or {}
            if want and all(str(args.get(k)).lower() == str(v).lower() for k, v in want.items()):
                return mode
        properties = (self.schemas.get(name) or {}).get("properties") or {}
        for key, spec in properties.items():
            if isinstance(spec, dict) and spec.get("enum") and args.get(key) is not None:
                value = str(args.get(key))
                if _DECLINE_WORD.search(value) and not _ORDER_WORD.search(value):
                    return "decline"
                if _ORDER_WORD.search(value):
                    return "order"
                # the first enum value usually buys
                return "order" if value == str(spec["enum"][0]) else "decline"
        return "order"

    def finish_call(self, mode: str, remark: str) -> tuple[str, dict[str, Any]] | None:
        """Arguments that finish the episode in the given mode, read from the schema."""
        name = self.roles.get(mode) or self.roles.get("order" if mode == "decline" else "decline")
        if not name:
            return None
        schema = self.schemas.get(name) or {}
        properties = schema.get("properties") or {}
        args: dict[str, Any] = {}
        want = self.finish_args.get(mode) or {}
        for key, spec in properties.items():
            if not isinstance(spec, dict):
                continue
            if key in want:
                args[key] = want[key]
            elif spec.get("enum"):
                options = [str(v) for v in spec["enum"]]
                if mode == "decline":
                    chosen = next((o for o in options if _DECLINE_WORD.search(o)), None) or next((o for o in options if not _ORDER_WORD.search(o)), options[-1])
                else:
                    chosen = next((o for o in options if _ORDER_WORD.search(o) and not _DECLINE_WORD.search(o)), None) or options[0]
                args[key] = chosen
            elif spec.get("type") == "string" or (isinstance(spec.get("type"), list) and "string" in spec["type"]):
                if key in (schema.get("required") or []) or mode == "decline":
                    args[key] = remark
            elif spec.get("type") == "boolean" and key in (schema.get("required") or []):
                args[key] = True
            elif spec.get("type") in ("integer", "number") and key in (schema.get("required") or []):
                args[key] = spec.get("minimum", 1)
        args, problems = _check(args, schema, "args")
        if problems:
            return None
        return name, args

    # -- the ledger ------------------------------------------------------------

    def tagged_requirements(self) -> list[tuple[int | None, str]]:
        """Every current requirement with the index of the product it is about (None when it applies to all)."""
        out: list[tuple[int | None, str]] = []
        many = len(self.items) > 1
        for index, item in enumerate(self.items):
            for requirement in item["requirements"]:
                text = f"{item['product']} {requirement}" if many and not requirement.lower().startswith(item["product"].lower()) else requirement
                if text not in self.dropped and requirement not in self.dropped:
                    out.append((index if many else None, text))
        for requirement in self.revealed:
            if requirement not in self.dropped and all(requirement != text for _, text in out):
                out.append((self.revealed_product.get(requirement) if many else None, requirement))
        return out

    def requirements(self) -> list[str]:
        return [text for _, text in self.tagged_requirements()]

    def ledger(self) -> str:
        lines = []
        many = len(self.items) > 1
        for item in self.items:
            head = f"- buy: {item['product']} x{item['quantity']}"
            if item["category_words"]:
                head += f" (category: {item['category_words']})"
            lines.append(head)
        for requirement in self.requirements():
            lines.append(f"- requirement: {requirement}")
        for reference in self.references:
            lines.append(f"- requirement defined through another listing: {reference.get('text')}" + (" [resolved: " + reference["resolved"] + "]" if reference.get("resolved") else " [NOT YET RESOLVED: find that listing and read the property]"))
        if self.budget is not None:
            lines.append(f"- total price at most {self.budget:g} {self.budget_unit}".rstrip() + " (price shown times quantity)")
        else:
            lines.append("- no price limit known" + (" (ask the shopper)" if self.roles.get("message") and self.messages_left() != 0 else ""))
        if self.in_stock_only:
            lines.append("- only items in stock")
        if self.objective == "cheapest":
            lines.append("- choose the CHEAPEST item that meets everything (check cheaper candidates before settling)")
        elif self.objective == "best":
            lines.append(f"- choose the BEST item by: {self.plan.get('objective_measure') or 'the stated measure'}")
        else:
            lines.append("- no cheapest or best rule: any item that meets everything within the budget is right (do not hunt for the lowest price)")
        wishes = [w for item in self.items for w in item["wishes"]]
        if wishes:
            lines.append("- nice to have, not required: " + "; ".join(wishes))
        if self.dropped:
            lines.append("- no longer required: " + "; ".join(self.dropped))
        if self.no_preference:
            lines.append("- the shopper does not care about: " + ", ".join(self.no_preference))
        for rule in self.rules:
            lines.append(f"- RULE to follow: {rule}")
        if self.plan.get("decline_rule"):
            lines.append(f"- if nothing fits: {self.plan.get('decline_rule')}")
        if many:
            lines.append("- several products: one cart line per product, each at its own quantity; judge each listing against its own product's requirements")
        return "\n".join(lines)

    # -- listings ----------------------------------------------------------------

    def absorb(self, name: str, args: dict[str, Any], obs: Any) -> list[str]:
        """Record what a result shows; returns the shopper's loose lines carried by it."""
        is_details = name in (self.roles.get("details"), self.roles.get("compare"))
        for row in _walk_listings(obs):
            ident = _is_listing(row)
            if not ident:
                continue
            listing = self.listings.get(ident)
            if listing is None:
                listing = Listing(ident)
                listing.first_seen = self.step
                self.listings[ident] = listing
                self.order_seen.append(ident)
            listing.last_seen = self.step
            facts = _facts_of(row)
            for key, value in row.items():
                if isinstance(value, (dict, list)):
                    continue
                key_text = str(key)
                role = _hinted_any(key)
                if role is None:
                    if _PRICE_KEY.search(key_text) and not _TOTAL_KEY.search(key_text) and _number(value) is not None and not _QTY_KEY.search(key_text):
                        role = "price"
                    elif _STOCK_KEY.search(key_text) and _truthy(value) is not None:
                        role = "stock"
                    elif _CATEGORY_KEY.search(key_text) and isinstance(value, str):
                        role = "category"
                    elif _BRAND_KEY.search(key_text) and isinstance(value, str):
                        role = "brand"
                    elif _SELLER_KEY.search(key_text) and isinstance(value, str):
                        role = "seller"
                    elif _TITLE_KEY.search(key_text) and isinstance(value, str):
                        role = "title"
                    elif isinstance(value, str) and _NOTE_KEY.search(key_text):
                        role = "text"
                if role == "price" and _number(value) is not None:
                    listing.price = _number(value)
                elif role == "stock":
                    flag = _truthy(value)
                    if _HINTS["in_stock_values"]:
                        flag = str(value).strip().lower() in _HINTS["in_stock_values"]
                    if flag is not None:
                        listing.in_stock = flag
                elif role == "category" and isinstance(value, str) and value.strip():
                    listing.category = value.strip()
                elif role == "brand" and isinstance(value, str) and value.strip() and value.strip().lower() != "null":
                    listing.brand = value.strip()
                elif role == "seller" and isinstance(value, str) and value.strip():
                    listing.seller = value.strip()
                elif role == "title" and isinstance(value, str) and value.strip():
                    if not listing.title or len(value) > len(listing.title):
                        listing.title = value.strip()
                elif role == "text" and isinstance(value, str) and len(value.split()) >= 4 and value.strip().lower() != "null":
                    if value not in listing.notes:
                        listing.notes.append(value.strip())
            if facts:
                listing.facts.update(facts)
                listing.opened = True
            elif is_details and name != self.roles.get("cart_view"):
                listing.opened = True
            keep = {k: v for k, v in row.items() if not isinstance(v, (dict, list)) and not (isinstance(v, str) and len(v) > 220 and not _NOTE_KEY.search(str(k)))}
            for k, v in keep.items():
                listing.record[k] = _short(v, 300) if isinstance(v, str) else v
        # search coverage
        if name == self.roles.get("search") or (name not in (self.roles.get("details"), self.roles.get("compare"), self.roles.get("cart_view"), self.roles.get("cart_add"), self.roles.get("cart_remove")) and _walk_listings(obs)):
            total = None
            more = None
            if isinstance(obs, dict):
                for key, value in obs.items():
                    key_text = str(key).lower()
                    if _hinted("count", key) and _number(value) is not None:
                        total = _number(value)
                    elif _hinted("more", key):
                        if value not in (None, "", "null", False, 0):
                            more = value
                    elif _hinted_any(key) is not None:
                        continue
                    elif _number(value) is not None and re.search(r"count|total|match|found|hits|result", key_text) and not re.search(r"page|next|cursor|token|price", key_text):
                        total = _number(value)
                    elif re.search(r"next|page|cursor|token|more|continue|onward|follow|bookmark|marker", key_text) and value not in (None, "", "null", False, 0):
                        more = value
            rows = len(_walk_listings(obs))
            self.searches.append(f"{name}({_compact(args, 160)}) -> {rows} rows" + (f" of {total:g} matching" if total is not None else "") + (f", more available ({_short(more, 20)})" if more is not None else ""))
            self.search_stats.append((rows, total, more))
        return _loose_lines(obs)

    def candidates(self) -> list[Listing]:
        rows = [self.listings[i] for i in self.order_seen]
        rows.sort(key=lambda l: (l.price if l.price is not None else 1e18))
        return rows

    def affordable(self, listing: Listing, quantity: int = 1) -> bool:
        if self.budget is not None and listing.price is not None and listing.price * quantity > self.budget + 1e-9:
            return False
        if self.in_stock_only and listing.in_stock is False:
            return False
        return True

    def product_of(self, listing: Listing) -> int:
        """Which requested product a listing belongs to (index into items, -1 for none); 0 when one product is requested."""
        if len(self.items) <= 1:
            return 0
        label = " ".join(str(listing.category or "").lower().split())
        if label:
            if label not in self.product_map:
                self.map_categories()
            if label in self.product_map:
                return self.product_map[label]
        return self.product_by_words(listing.category, listing.title)

    def category_fits(self, listing: Listing) -> bool | None:
        """Whether the listing's store category label belongs to the requested range, by the model's label map; None when unknown."""
        label = " ".join(str(listing.category or "").lower().split())
        if not label:
            return None
        if label not in self.product_map:
            self.map_categories()
        value = self.product_map.get(label)
        return None if value is None else value >= 0

    def product_by_words(self, category: str, title: str) -> int:
        """Fallback assignment by shared words, used only when the category map is unavailable."""
        category_words = _plain_words(category)
        title_words = _plain_words(title)
        best, score = -1, 0
        for index, item in enumerate(self.items):
            words = _plain_words(item["product"]) | _plain_words(item["category_words"])
            points = len(words & category_words) * 3 + len(words & title_words)
            if points > score:
                best, score = index, points
        return best

    def map_result_keys(self, name: str, obs: Any) -> None:
        """Ask the reader model once which keys of this tool's results play which roles, and register them as hints."""
        if name in self.key_map_tried or not isinstance(obs, (dict, list)) or self.dead or self.time_left() < WRAP_UP_SECONDS:
            return
        paths: dict[str, Any] = {}

        def walk(value: Any, path: str, depth: int) -> None:
            if depth > 4 or len(paths) > 70:
                return
            if isinstance(value, dict):
                for key, item in value.items():
                    sub = f"{path}.{key}" if path else str(key)
                    if isinstance(item, (dict, list)):
                        if sub not in paths:
                            paths[sub] = "(list)" if isinstance(item, list) else "(object)"
                        walk(item, sub, depth + 1)
                    elif sub not in paths:
                        paths[sub] = _short(json.dumps(item, ensure_ascii=False), 80)
            elif isinstance(value, list):
                for item in value[:3]:
                    walk(item, path + "[]", depth + 1)

        walk(obs, "", 0)
        if len(paths) < 2:
            return
        self.key_map_tried.add(name)
        prompt = RESULT_MAP_PROMPT + f"\n\nTOOL: {name}\nDESCRIPTION: {_short(self.descriptions.get(name, ''), 300)}\n\nKEYS OF THE RESULT (path: example value):\n" + "\n".join(f"{p}: {v}" for p, v in paths.items())
        result = self.ask_json(prompt, 1200, [self.models["reader"], self.models["main"], self.models["alt"]])
        if not isinstance(result, dict):
            self.note("key_map_failed", tool=name)
            return
        known = {p.split(".")[-1].replace("[]", "") for p in paths}
        mapping: dict[str, Any] = {}
        for role in _HINT_ROLES:
            value = result.get(role)
            keys = value if isinstance(value, list) else ([value] if isinstance(value, str) else [])
            cleaned = []
            for key in keys:
                leaf = str(key).split(".")[-1].replace("[]", "").strip()
                if leaf and leaf in known:
                    cleaned.append(leaf)
            if cleaned:
                mapping[role] = cleaned
        values = result.get("in_stock_values")
        if isinstance(values, list) and values:
            mapping["in_stock_values"] = [str(v).strip().lower() for v in values if str(v).strip()]
        self.key_map[name] = mapping
        for role, keys in mapping.items():
            if role == "in_stock_values":
                _HINTS["in_stock_values"].update(keys)
            elif role in _HINTS:
                _HINTS[role].update(keys)
        self.note("key_map", tool=name, mapping=mapping)

    def map_categories(self) -> None:
        """Ask the reader model once which requested product each new store category label belongs to."""
        pending: dict[str, str] = {}
        for listing in self.listings.values():
            label = " ".join(str(listing.category or "").lower().split())
            if label and label not in self.product_map and label not in pending:
                pending[label] = str(listing.category).strip()
        if not pending:
            return
        result = None
        if self.product_map_failures < 2 and not self.dead and self.time_left() > FORCE_SECONDS + 10:
            prompt = PRODUCT_MAP_PROMPT + "\n\nPRODUCTS REQUESTED:\n" + "\n".join(
                f"{n}. {i['product']}" + (f" (the request calls the range: {i['category_words']})" if i["category_words"] else "") + (f"; requirements: {'; '.join(i['requirements'][:4])}" if i["requirements"] else "")
                for n, i in enumerate(self.items, 1))
            prompt += "\n\nSTORE CATEGORY LABELS:\n" + "\n".join(f"- {label}" for label in pending.values())
            result = self.ask_json(prompt, 800, [self.models["knowledge"], self.models["reader"], self.models["main"]])
            mapping = result.get("labels") if isinstance(result, dict) else None
            if isinstance(mapping, dict):
                for key, value in mapping.items():
                    label = " ".join(str(key).lower().split())
                    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
                        value = int(value)
                    if label in pending and isinstance(value, (int, float)) and not isinstance(value, bool):
                        number = int(value)
                        self.product_map[label] = number - 1 if 1 <= number <= len(self.items) else -1
            else:
                result = None
        if result is None:
            self.product_map_failures += 1
        if result is None and self.product_map_failures >= 2:
            # the reader is unavailable: fall back to shared words so the labels stop being asked about
            for label, shown in pending.items():
                self.product_map.setdefault(label, self.product_by_words(shown, ""))
        self.note("category_map", asked=list(pending.values()), mapping={k: v for k, v in self.product_map.items() if k in pending})

    def requirements_for(self, index: int) -> list[str]:
        """The requirements one product must meet: its own and the ones that apply to every product."""
        return [text for product, text in self.tagged_requirements() if product is None or product == index]

    def candidate_table(self, limit: int = 18) -> str:
        if len(self.items) > 1:
            return self.candidate_table_by_product()
        opened = [l for l in self.candidates() if l.opened]
        unopened = [l for l in self.candidates() if not l.opened and self.affordable(l)]
        lines = []
        for listing in opened[:limit]:
            verdict = self.verdicts.get(f"{listing.id}|{self.ledger_version}")
            tag = ""
            if verdict:
                if verdict.get("fits") is True:
                    tag = " => fits every checked requirement"
                elif verdict.get("fits") is False:
                    tag = " => FAILS: " + "; ".join(str(x) for x in (verdict.get("failed") or [])[:3])
                elif verdict.get("unknown"):
                    tag = " => unknown: " + "; ".join(str(x) for x in verdict["unknown"][:3])
            money = "" if self.affordable(listing) else " (over budget or out of stock)"
            reading = self.literal_reading(listing)
            lines.append("  " + listing.summary() + money + tag + (" | literal check: " + reading if reading else ""))
        text = ""
        if lines:
            text += "Opened listings (properties known), cheapest first:\n" + "\n".join(lines)
        if unopened:
            text += ("\n" if text else "") + "Cheapest rows seen but not opened: " + "; ".join(f"{l.id} {l.price:g}" if l.price is not None else l.id for l in unopened[:8])
        return text

    def candidate_table_by_product(self) -> str:
        blocks = []
        by_product: dict[int, list[Listing]] = {}
        for listing in self.candidates():
            by_product.setdefault(self.product_of(listing), []).append(listing)
        for index, item in enumerate(self.items):
            rows = by_product.get(index, [])
            opened = [l for l in rows if l.opened]
            unopened = [l for l in rows if not l.opened and self.affordable(l, item["quantity"])]
            in_cart = [i for i in self.intended_lines() if i in self.listings and self.product_of(self.listings[i]) == index]
            head = f"PRODUCT {index + 1}: {item['product']} (need {item['quantity']} unit(s))" + (f"; in cart: {', '.join(f'{i} x{self.intended_lines()[i]}' for i in in_cart)}" if in_cart else "; NOT in the cart yet")
            lines = [head]
            for listing in opened[:8]:
                verdict = self.verdicts.get(f"{listing.id}|{self.ledger_version}")
                tag = ""
                if verdict:
                    if verdict.get("fits") is True:
                        tag = " => fits every checked requirement"
                    elif verdict.get("fits") is False:
                        tag = " => FAILS: " + "; ".join(str(x) for x in (verdict.get("failed") or [])[:3])
                money = "" if self.affordable(listing, item["quantity"]) else " (over budget or out of stock)"
                reading = self.literal_reading(listing, index)
                lines.append("  " + listing.summary() + money + tag + (" | literal check: " + reading if reading else ""))
            if unopened:
                lines.append("  not opened yet, cheapest first: " + "; ".join(f"{l.id} {l.price:g}" if l.price is not None else l.id for l in unopened[:5]))
            if not rows:
                lines.append("  no listing of this product seen yet: search for it")
            blocks.append("\n".join(lines))
        stray = by_product.get(-1, [])
        if stray:
            blocks.append("Listings not matched to any requested product: " + ", ".join(l.id for l in stray[:6]))
        return "\n".join(blocks)

    def literal_reading(self, listing: Listing, product: int | None = None) -> str:
        """A short literal reading of the listing's facts against each requirement (presentation only)."""
        parts = []
        requirements = self.requirements_for(product) if product is not None else self.requirements()
        for requirement in requirements[:6]:
            kind, shown = _evidence(listing, requirement)
            label = {"stated": "states", "other": "DIFFERS", "absent": "no such property", "title": "title only", "unclear": "no key names it fully"}[kind]
            parts.append(f"{requirement.split(':')[0].strip()} -> {label}" + (f" ({_short(shown, 70)})" if kind in ("other", "stated", "unclear") else ""))
        return "; ".join(parts)

    def category_labels(self) -> str:
        counts: dict[str, int] = {}
        for listing in self.listings.values():
            if listing.category:
                counts[listing.category] = counts.get(listing.category, 0) + 1
        ranked = sorted(counts.items(), key=lambda kv: -kv[1])[:8]
        return ", ".join(f"{k} (x{v})" for k, v in ranked)

    def status(self) -> str:
        left = self.steps_left()
        parts = [f"[status] step {self.step} of {self.max_steps} ({left} left)"]
        if self.message_budget is not None:
            parts.append(f"messages {self.messages_used} of {self.message_budget} used")
        else:
            parts.append(f"messages sent {self.messages_used}")
        parts.append(f"about {int(max(0, self.time_left() - FORCE_SECONDS))} s left")
        text = "; ".join(parts) + "."
        if self.cart:
            text += " Cart as tracked: " + ", ".join(f"{k} x{v}" for k, v in self.cart.items() if v) + "."
        text += "\n[requirements now]\n" + self.ledger()
        labels = self.category_labels()
        if labels:
            text += "\n[category labels seen on listings, use them exactly in filters] " + labels
        table = self.candidate_table()
        if table:
            text += "\n[candidates]\n" + table
        if self.searches:
            text += "\n[searches so far] " + " | ".join(self.searches[-6:])
        coverage = self.coverage_hint()
        if coverage:
            text += "\n[coverage] " + coverage
        planned = self.search_plan_lines()
        if planned:
            text += "\n" + planned
        if self.changes:
            text += "\n[changes announced] " + " | ".join(self.changes[-4:])
        if len(self.items) > 1:
            open_products = [i for i, item in enumerate(self.items) if not any(x in self.listings and self.product_of(self.listings[x]) == i for x in self.intended_lines())]
            if open_products:
                share = max(1, (left - self.finish_reserve()) // max(1, len(open_products)))
                text += f"\n[pacing] {len(open_products)} of {len(self.items)} products still need a cart line; about {share} steps each remain for them. Search, open two candidates and cart each product in turn; do not spend more on one product than that."
        if self.confirm_before_order:
            text += "\n[rule] the shopper must be shown the basket before the order; the harness sends that summary when you call the order tool."
        if self.refusals and not self.finished:
            text += "\nNOTHING HAS BEEN BOUGHT YET: the last order attempt was refused; fix the cart and order again (or decline only if nothing fits any more)."
        if left <= 2:
            text += "\nLAST STEPS: order now if the cart holds a fitting item; otherwise decline."
        elif left <= 5 or self.time_left() < WRAP_UP_SECONDS:
            text += "\nFINISH NOW: make the cart hold exactly the chosen line(s), then order (or decline if nothing fits)."
        elif self.order_attempts == 0 and left <= self.finish_reserve() + 5:
            text += "\nSteps are running low: put the best verified fit in the cart and order within the next few steps."
        return text

    # -- shopper exchanges -------------------------------------------------------

    def coverage_hint(self) -> str:
        """For a cheapest objective: when searches report more matches than they show, say how to reach the unseen cheaper rows."""
        if self.objective != "cheapest" or not self.search_stats:
            return ""
        unseen = [(rows, total) for rows, total, more in self.search_stats[-4:] if total is not None and rows and total > rows]
        if not unseen:
            return ""
        rows, total = unseen[-1]
        schema = self.schemas.get(self.roles.get("search") or "") or {}
        numeric = [k for k, v in (schema.get("properties") or {}).items() if isinstance(v, dict) and v.get("type") in ("integer", "number")]
        best = min((l.price for l in self.listings.values() if l.price is not None and l.opened and l.in_stock is not False and self.affordable(l)), default=None)
        hint = f"the last such search reported {total:g} matches but showed {rows}; results are not sorted by price, so the cheapest fitting item may be among the rows you cannot page to."
        if numeric:
            hint += f" The search tool's numeric arguments are {', '.join(numeric)}: tighten the price limit (for example to about half of {best:g}) until the match count fits in the rows you can see, then open the cheapest rows." if best is not None else f" The search tool's numeric arguments are {', '.join(numeric)}: use a price limit low enough that the match count fits in the rows you can see."
        else:
            hint += " Narrow the words or the category until the count fits in the rows you can see."
        return hint

    def message_args(self, text: str) -> dict[str, Any] | None:
        name = self.roles.get("message")
        if not name:
            return None
        schema = self.schemas.get(name) or {}
        properties = schema.get("properties") or {}
        required = list(schema.get("required") or [])
        strings = [k for k, v in properties.items() if isinstance(v, dict) and (v.get("type") == "string" or (isinstance(v.get("type"), list) and "string" in v["type"])) and "enum" not in v]
        target = next((k for k in required if k in strings), None) or (strings[0] if strings else None)
        if not target or any(k != target for k in required):
            return None
        args, problems = _check({target: text}, schema, "args")
        return None if problems else args

    def send_message(self, text: str, about: str) -> str | None:
        """Send one message to the shopper; returns the reply text (or None when not sent)."""
        name = self.roles.get("message")
        left = self.messages_left()
        if not name or (left is not None and left <= 0) or self.steps_left() <= 2:
            return None
        args = self.message_args(text)
        if args is None:
            return None
        reply = self.call(name, args)
        if reply is None:
            return None
        obs = reply.get("observation")
        if _result_failed(obs) and not (isinstance(obs, dict) and any(isinstance(v, str) and len(v.split()) >= 4 and not _ERROR_KEY.search(str(k)) for k, v in obs.items())):
            self.message_refused = True
            if self.message_budget is None or self.message_budget > self.messages_used:
                self.message_budget = self.messages_used
            self.note("message_refused", observation=_compact(obs, 300))
            return None
        self.messages_used += 1
        self.map_result_keys(name, obs)
        self.absorb(name, args, obs)
        text_reply = ""
        if isinstance(obs, str):
            text_reply = obs
        elif isinstance(obs, dict):
            strings = [v for k, v in obs.items() if isinstance(v, str) and v.strip() and _hinted("reply", k)]
            if not strings:
                strings = [v for k, v in obs.items() if isinstance(v, str) and v.strip() and not _ERROR_KEY.search(str(k)) and _hinted_any(k) in (None, "reply")]
            text_reply = strings[0] if len(strings) == 1 else (" ".join(strings) if strings else _compact(obs, 400))
            lines = _loose_lines(obs)
            extra = [l for l in lines if l != text_reply]
            if extra:
                text_reply = (text_reply + " " if text_reply else "") + " ".join(extra)
        else:
            text_reply = _compact(obs, 400)
        self.asked.append((about, text, text_reply))
        if about:
            self.asked_properties.append(about.lower())
        self.heard.append(text_reply)
        return text_reply

    def interpret(self, line: str, question: str | None, about: str | None) -> dict[str, Any]:
        """What a shopper line means, applied to the ledger. Returns the parsed reading."""
        reading: dict[str, Any] = {}
        if self.time_left() < FORCE_SECONDS + 10:
            return reading
        prompt = INTERPRET_PROMPT + "\n\nTHE SHOPPER'S ORIGINAL REQUEST:\n" + self.request
        prompt += "\n\nREQUIREMENTS SO FAR:\n" + "\n".join("- " + r for r in self.requirements())
        if self.budget is not None:
            prompt += f"\n- budget: {self.budget:g} {self.budget_unit}".rstrip()
        if len(self.items) > 1:
            prompt += "\n\nPRODUCTS:\n" + "\n".join(f"{n}. {i['product']}" for n, i in enumerate(self.items, 1))
        if self.asked:
            prompt += "\n\nEARLIER EXCHANGES:\n" + "\n".join(f"- asked: {q} -> shopper: {_short(a, 200)}" for _, q, a in self.asked[-6:])
        if question:
            prompt += "\n\nTHE AGENT'S MESSAGE:\n" + question
            if about:
                prompt += f"\n(it asked about the property: {about})"
        else:
            prompt += "\n\nTHIS LINE ARRIVED INSIDE A TOOL RESULT (not an answer to a question)."
        cart_lines = [i for i in self.intended_lines() if i in self.listings]
        if len(cart_lines) > 1:
            prompt += "\n\nCART LINES (in the order they were added):\n" + "\n".join(f"{n}. {i}: {_short(self.listings[i].title, 60)} [{_short(self.listings[i].category, 30)}]" for n, i in enumerate(cart_lines, 1))
        prompt += "\n\nTHE SHOPPER NOW SAID:\n" + line
        self.last_line = line
        self.last_line_is_reply = question is not None
        models = [self.models["setup"], self.models["reader"], self.models["alt"]] if question is not None else [self.models["reader"], self.models["main"], self.models["alt"]]
        reading = self.ask_json(prompt, 900, models) or {}
        # cheap fallbacks when the reader is unavailable
        if not reading:
            reading = {"kind": "other", "meaning": line}
        self.apply_reading(reading, about)
        self.note("heard", line=line[:300], reading=reading)
        return reading

    def apply_reading(self, reading: dict[str, Any], about: str | None) -> None:
        changed = False
        many = len(self.items) > 1
        needs: list[str] = []
        for need in reading.get("needs") or []:
            product: int | None = None
            if isinstance(need, dict):
                number = need.get("product")
                if isinstance(number, str) and number.strip().isdigit():
                    number = int(number)
                if isinstance(number, (int, float)) and not isinstance(number, bool) and 1 <= int(number) <= len(self.items) and many:
                    product = int(number) - 1
                need = need.get("need")
            if not isinstance(need, str) or not need.strip():
                continue
            text = need.strip()
            if product is not None and not text.lower().startswith(self.items[product]["product"].lower()):
                text = f"{self.items[product]['product']} {text}"
            needs.append(text)
            if text not in self.revealed:
                self.revealed.append(text)
                if product is not None:
                    self.revealed_product[text] = product
                changed = True
        reading["needs"] = needs
        for prop in reading.get("no_preference_about") or []:
            if isinstance(prop, str) and prop.strip() and prop.strip().lower() not in self.no_preference:
                self.no_preference.append(prop.strip().lower())
        if reading.get("kind") == "no_preference" and about and about.lower() not in self.no_preference:
            self.no_preference.append(about.lower())
        for drop in reading.get("dropped") or []:
            if isinstance(drop, str) and drop.strip():
                target = self.match_requirement(drop)
                if target and target not in self.dropped:
                    self.dropped.append(target)
                    changed = True
                elif not target and drop.strip() not in self.dropped:
                    self.dropped.append(drop.strip())
                    changed = True
        budget = _number(reading.get("new_budget"))
        if budget is not None and budget > 0 and budget != self.budget:
            self.budget = budget
            self.changes.append(f"new budget {budget:g}")
            changed = True
        quantity = reading.get("new_quantity")
        if isinstance(quantity, dict):
            count = quantity.get("quantity")
            if isinstance(count, (int, float)) and not isinstance(count, bool) and count >= 1:
                target = None
                wanted = str(quantity.get("product") or "").lower()
                for item in self.items:
                    if wanted and wanted in item["product"].lower():
                        target = item
                target = target or (self.items[0] if len(self.items) == 1 else None)
                if target and target["quantity"] != int(count):
                    target["quantity"] = int(count)
                    self.changes.append(f"quantity of {target['product']} now {int(count)}")
                    changed = True
        if reading.get("unavailable") and reading.get("new_budget") is not None:
            reading["unavailable"] = False      # a budget line is about money, not stock
        if reading.get("unavailable"):
            cart_lines = [i for i in self.intended_lines() if i in self.listings]
            number = reading.get("cart_line")
            if isinstance(number, str) and number.strip().isdigit():
                number = int(number)
            affected = [cart_lines[int(number) - 1]] if isinstance(number, (int, float)) and not isinstance(number, bool) and 1 <= int(number) <= len(cart_lines) else cart_lines
            for ident in affected:
                self.listings[ident].in_stock = False
            which = ", ".join(affected) if affected else "the picked item"
            self.changes.append(f"{which} can no longer be bought (remove it from the cart and take the next verified fit)")
            changed = True
        if reading.get("price_changed"):
            self.changes.append("the picked item's price changed: re-read it before ordering")
            for ident in list(self.cart):
                self.reread_at.pop(ident, None)
        reference = reading.get("reference")
        if isinstance(reference, dict) and reference.get("property"):
            text = f"{reference.get('property')}: must be {'the same as' if reference.get('mode') != 'exclude' else 'anything but'} the value shown on the listing of {reference.get('category') or 'the same kind'} sold by {reference.get('seller') or '?'} with " + ", ".join(f"{k} {v}" for k, v in (reference.get("facts") or {}).items())
            reference["text"] = text
            if not any(r.get("text") == text for r in self.references):
                self.references.append(reference)
                changed = True
        if reading.get("kind") == "question" and str(reading.get("reply") or "").strip():
            self.pending_reply = str(reading["reply"]).strip()
        if changed:
            self.ledger_version += 1

    def match_requirement(self, text: str) -> str | None:
        """The current requirement a shopper line names: by exact words first, by containment next, by word overlap last."""
        wanted = " ".join(text.lower().split())
        current = self.requirements()
        for requirement in current:
            if " ".join(requirement.lower().split()) == wanted:
                return requirement
        for requirement in current:
            folded = " ".join(requirement.lower().split())
            if len(wanted) >= 6 and (wanted in folded or folded in wanted):
                return requirement
        words = _words(text)
        best, score = None, 0
        for requirement in current:
            common = len(words & _words(requirement))
            if common > score:
                best, score = requirement, common
        return best if score >= 1 else None

    def property_menu(self, product: int | None = None) -> str:
        """Properties of the opened listings with their values, most widespread and varied first."""
        values: dict[str, dict[str, int]] = {}
        counts: dict[str, int] = {}
        pool = [l for l in self.candidates() if l.facts and (product is None or self.product_of(l) == product)]
        opened = [l for l in pool if self.affordable(l)] or pool
        for listing in opened[:24]:
            for key, value in listing.facts.items():
                name = re.sub(r"[_.\-]+", " ", str(key)).strip().lower()
                if _PRICE_KEY.search(name) or _STOCK_KEY.search(name) or re.search(r"brand|seller|shop|maker|option|variant|sku|model|url|image|id$", name):
                    continue
                counts[name] = counts.get(name, 0) + 1
                bucket = values.setdefault(name, {})
                text = _short(str(value).strip().lower(), 40)
                bucket[text] = bucket.get(text, 0) + 1
        ranked = sorted(counts, key=lambda k: (-counts[k], -min(len(values[k]), 4), k))
        lines = []
        for name in ranked[:26]:
            sample = sorted(values[name], key=lambda v: -values[name][v])[:5]
            lines.append(f"- {name} (in {counts[name]} of {len(opened)} listings): " + " | ".join(sample))
        return "\n".join(lines)

    def question_phase(self, reason: str) -> bool:
        """Ask the shopper about the likeliest hidden needs. Returns True when any message was sent."""
        if not self.roles.get("message") or self.message_refused or self.finished or self.dead:
            self.planned = True
            return False
        left = self.messages_left()
        if left is None:
            left = 3
        reserve = 1 if (self.confirm_before_order or self.messages_required) and left >= 2 else 0
        count = left - reserve
        if count <= 0 or self.steps_left() <= self.finish_reserve() + 1 or self.time_left() < WRAP_UP_SECONDS + 30:
            self.planned = True
            return False
        sent_any = False
        # 1. things the request itself leaves to the shopper
        while self.ask_first and count > 0:
            question = self.ask_first.pop(0)
            reply = self.send_message(question, "")
            if reply is None:
                break
            sent_any = True
            count -= 1
            self.interpret(reply, question, None)
        if count <= 0:
            self.planned = True
            return sent_any
        # 2. hidden needs, chosen from the story and the listings seen
        opened = [l for l in self.candidates() if l.facts]
        if len(opened) < 2 and reason != "forced" and self.steps_left() > self.finish_reserve() + 8:
            return sent_any
        self.plan_rounds += 1
        prompt = PLAN_PROMPT.replace("COUNT", str(count)) + "\n\nTHE SHOPPER'S REQUEST:\n" + self.request
        prompt += "\n\nPRODUCT(S): " + "; ".join(f"{i['product']} (requirements: {', '.join(i['requirements']) or 'none stated'})" for i in self.items)
        if self.hints:
            prompt += "\n\nSTORY HINTS (ask these first unless settled): " + "; ".join(f"{h['property']} ({_short(h['why'], 80)})" for h in self.hints)
        if self.asked_properties or self.no_preference:
            prompt += "\n\nALREADY ASKED OR SETTLED: " + ", ".join(sorted(set(self.asked_properties) | set(self.no_preference)))
        if len(self.items) > 1:
            menus = []
            for index, item in enumerate(self.items):
                asked_here = sum(1 for a in self.asked if a[0] and a[0].startswith(item["product"].lower() + " "))
                menus.append(f"PRODUCT: {item['product']} (questions asked about it so far: {asked_here})\n" + (self.property_menu(index) or "(no listing of this product opened yet)"))
            menu = "\n\n".join(menus)
        else:
            menu = self.property_menu()
        prompt += "\n\nPROPERTIES OF THE LISTINGS SEEN SO FAR (name, how many listings, sample values):\n" + (menu or "(no listing opened yet: use the hints and common properties of this kind of product)")
        result = self.ask_json(prompt, 1400, [self.models["reader"], self.models["main"], self.models["alt"]], require=("questions",)) or {}
        questions = [q for q in (result.get("questions") or []) if isinstance(q, dict) and str(q.get("question") or "").strip()]
        if not questions:
            settled = set(self.asked_properties) | set(self.no_preference)
            product = self.items[0]["product"] if len(self.items) == 1 else ""
            for hint in self.hints:
                if hint["property"] and hint["property"] not in settled:
                    questions.append({"product": product, "property": hint["property"], "question": f"Which {hint['property']} would you like the {product or 'item'} to have?"})
        self.note("question_plan", questions=questions)
        for item in questions:
            if count <= 0 or self.steps_left() <= self.finish_reserve() + 1 or self.time_left() < WRAP_UP_SECONDS + 15:
                break
            prop = str(item.get("property") or "").strip().lower()
            if len(self.items) > 1 and str(item.get("product") or "").strip():
                prop = str(item.get("product")).strip().lower() + " " + prop
            if prop and (prop in self.asked_properties or prop in self.no_preference):
                continue
            question = str(item["question"]).strip()
            reply = self.send_message(question, prop)
            if reply is None:
                break
            sent_any = True
            count -= 1
            reading = self.interpret(reply, question, prop)
            if reading.get("kind") == "unclear":
                # the shopper did not follow: move to the next property, never repeat
                continue
        # messages left over (and questions still possible): plan another round on a later turn, at most three rounds
        self.planned = not (count > 0 and questions and sent_any and self.plan_rounds < 3 and self.steps_left() > self.finish_reserve() + 2)
        return sent_any

    # -- cart tracking ---------------------------------------------------------------

    def cart_effect(self, name: str, args: dict[str, Any], obs: Any) -> None:
        if name not in (self.roles.get("cart_add"), self.roles.get("cart_remove")):
            return
        if _result_failed(obs) and not _walk_listings(obs):
            return
        properties = (self.schemas.get(name) or {}).get("properties") or {}
        remove = name == self.roles.get("cart_remove")
        quantity = None
        ids: list[str] = []
        lines: list[tuple[str, int]] = []

        def collect(value: Any, depth: int) -> None:
            nonlocal quantity
            if depth > 3:
                return
            if isinstance(value, dict):
                local_id, local_qty = None, None
                for key, item in value.items():
                    if isinstance(item, str) and _ID_SHAPE.match(item) and item in self.listings:
                        local_id = item
                    elif isinstance(item, int) and not isinstance(item, bool) and self.is_quantity_key(key):
                        local_qty = item
                    elif isinstance(item, (dict, list)):
                        collect(item, depth + 1)
                if local_id:
                    lines.append((local_id, local_qty if local_qty is not None else 1))
            elif isinstance(value, list):
                for item in value:
                    collect(item, depth + 1)

        for key, value in args.items():
            spec = properties.get(key) if isinstance(properties.get(key), dict) else {}
            if isinstance(value, str) and spec.get("enum"):
                if _REMOVE_WORD.search(value) and not _ADD_WORD.search(value):
                    remove = True
                elif _ADD_WORD.search(value):
                    remove = False
            elif isinstance(value, str) and _ID_SHAPE.match(value) and (value in self.listings or any(c.isdigit() for c in value)) and not spec.get("enum"):
                ids.append(value)
            elif isinstance(value, int) and not isinstance(value, bool) and (self.is_quantity_key(key) or spec.get("type") == "integer"):
                quantity = value
            elif isinstance(value, (dict, list)):
                collect(value, 0)
        if lines and not ids:
            # a call that passes whole cart lines replaces the cart
            self.cart = {ident: qty for ident, qty in lines}
        else:
            for ident in ids:
                if remove:
                    if quantity is None:
                        self.cart[ident] = 0
                    else:
                        self.cart[ident] = max(0, self.cart.get(ident, 0) - quantity)
                else:
                    self.cart[ident] = self.cart.get(ident, 0) + (quantity if quantity is not None else 1)
        # a result that lists the cart wins
        rows = _walk_listings(obs)
        if rows and isinstance(obs, dict) and any(_TOTAL_KEY.search(str(k)) or _CART_WORD.search(str(k)) for k in obs):
            self.cart_from_rows(rows)
        self.cart = {k: v for k, v in self.cart.items() if v > 0}
        self.cart_changed_at = self.step

    def is_quantity_key(self, key: Any) -> bool:
        """A result key that holds a line quantity: named like the cart tool's own count argument, or a generic count word."""
        text = str(key)
        if _hinted("quantity", key):
            return True
        if _hinted_any(key) is not None or _PRICE_KEY.search(text) or _ID_KEY.search(text):
            return False
        return bool(_QTY_KEY.search(text)) or bool(_words(text) & self.quantity_words)

    def cart_from_rows(self, rows: list[dict[str, Any]]) -> None:
        fresh: dict[str, int] = {}
        for row in rows:
            ident = _is_listing(row)
            if not ident:
                continue
            qty = 1
            for key, value in row.items():
                if isinstance(value, int) and not isinstance(value, bool) and self.is_quantity_key(key):
                    qty = value
            fresh[ident] = qty
        self.cart = fresh

    def intended_lines(self) -> dict[str, int]:
        return {k: v for k, v in self.cart.items() if v > 0}

    # -- gates before finishing ----------------------------------------------------------

    def records_for(self, idents: list[str], limit: int) -> str:
        blocks = []
        for ident in idents:
            listing = self.listings.get(ident)
            if listing:
                blocks.append(listing.full(limit))
        return "\n".join(blocks)

    def verify_order(self) -> str | None:
        """A hold message when the order should not go out yet, else None."""
        lines = self.intended_lines()
        if not lines:
            if self.roles.get("cart_add"):
                return "Not sent (no step used): the cart is empty as far as the tracked cart actions show. Put the chosen listing in the cart first, then order."
            return None
        problems: list[str] = []
        total = 0.0
        many = len(self.items) > 1
        for ident, qty in lines.items():
            listing = self.listings.get(ident)
            if listing is None:
                problems.append(f"{ident} was never shown by the store")
                continue
            if not listing.opened and self.roles.get("details") and self.holds["order"] < 2:
                problems.append(f"the properties of {ident} were never opened")
            if listing.price is not None:
                total += listing.price * qty
            if listing.in_stock is False and self.in_stock_only:
                problems.append(f"{ident} is shown out of stock")
            if self.refusals and listing.last_seen <= self.refused_at and self.roles.get("details") and self.steps_left() > 3 and self.holds["order"] < 2:
                problems.append(f"re-read {ident} (price and stock) after the refusal before ordering again")
        got = sum(lines.values())
        if not many and got != self.items[0]["quantity"]:
            problems.append(f"the cart holds {got} unit(s) but the request wants {self.items[0]['quantity']} of one listing")
        if not many and len(lines) > 1:
            problems.append("the cart holds more than one listing; remove the other line(s)")
        if many:
            per_product: dict[int, list[str]] = {}
            for ident in lines:
                if ident in self.listings:
                    per_product.setdefault(self.product_of(self.listings[ident]), []).append(ident)
            for index, item in enumerate(self.items):
                held = per_product.get(index, [])
                if not held:
                    problems.append(f"no cart line for product {index + 1} ({item['product']})")
                elif len(held) > 1:
                    problems.append(f"product {index + 1} ({item['product']}) has {len(held)} cart lines ({', '.join(held)}); keep one")
                elif lines[held[0]] != item["quantity"]:
                    problems.append(f"{held[0]} ({item['product']}) is in the cart x{lines[held[0]]} but the request wants x{item['quantity']}")
            if per_product.get(-1):
                problems.append("cart lines that match none of the requested products: " + ", ".join(per_product[-1]))
        if not many:
            for ident in lines:
                listing = self.listings.get(ident)
                if listing and listing.category and self.category_fits(listing) is False:
                    wanted = self.items[0]["category_words"] or self.items[0]["product"]
                    problems.append(f"{ident} carries the store category label '{listing.category}', which is not the requested range ('{wanted}'); labels seen: {self.category_labels() or 'none'}; pick a listing from the right category")
        if self.budget is not None and total > self.budget + 1e-9:
            problems.append(f"the cart total {total:g} is over the budget {self.budget:g}")
        if problems and self.holds["order"] < 2:
            self.holds["order"] += 1
            return "Not sent (no step used). Before ordering: " + "; ".join(problems) + ". Fix this, then order again."
        if self.confirm_before_order and self.basket_told_at < self.cart_changed_at and self.roles.get("message") and (self.messages_left() is None or self.messages_left() > 0) and self.steps_left() >= 2:
            told = self.tell_basket()
            if told:
                return "Not sent yet (no step used). The request asks that the shopper be told the basket before ordering, so the basket summary was sent for you; the shopper said: \"" + _short(told, 200) + "\". Apply any change it carries, then call the order tool again."
        if not self.checks_allowed() or self.holds["order"] >= 3:
            return None
        key = json.dumps(sorted(lines.items())) + f"|{self.ledger_version}|{len(self.listings)}"
        if key == self.last_verify_key:
            return None
        self.last_verify_key = key
        chosen = list(lines)
        others = [l for l in self.candidates() if l.id not in lines and l.opened and self.affordable(l)]
        cheaper_unopened = []
        if self.objective == "cheapest":
            top = max((self.listings[i].price or 0) for i in chosen)
            cheaper_unopened = [l for l in self.candidates() if not l.opened and self.affordable(l) and l.price is not None and l.price < top]
        prompt = VERIFY_PROMPT.replace("ACTION", "place an order") + "\n\nTHE SHOPPER'S REQUEST:\n" + self.request
        prompt += "\n\nCURRENT REQUIREMENTS:\n" + self.ledger()
        prompt += "\n\nTHE CART (listing id x quantity): " + ", ".join(f"{k} x{v}" for k, v in lines.items())
        prompt += "\n\nRECORDS OF THE LISTINGS IN THE CART:\n" + self.records_for(chosen, 2200)
        prompt += "\n\nLITERAL READING OF THE CART LISTINGS (facts whose key shares a word with each requirement):\n" + "\n".join(f"- {i}: {self.literal_reading(self.listings[i])}" for i in chosen if i in self.listings)
        if others:
            prompt += "\n\nOTHER OPENED LISTINGS WITHIN BUDGET (cheapest first):\n" + self.records_for([l.id for l in others[:8]], 900)
        if cheaper_unopened:
            prompt += "\n\nCHEAPER ROWS NOT OPENED YET (summary only):\n" + "\n".join(l.summary() for l in cheaper_unopened[:10])
        if self.rules:
            prompt += "\n\nPROCEDURAL RULES FROM THE REQUEST, AND WHAT THE AGENT DID:\n" + "\n".join("- " + r for r in self.rules)
            prompt += "\nMessages sent to the shopper: " + (" || ".join(f"[{q}] -> [{_short(a, 120)}]" for _, q, a in self.asked) or "none")
            prompt += f"\nListings whose properties were opened: {len([l for l in self.listings.values() if l.opened])}; searches made: {len(self.searches)}; order attempts so far: {self.order_attempts}"
        result = self.ask_json(prompt, 2200, [self.models["reader"], self.models["main"], self.models["alt"]], require=("verdicts",)) or {}
        self.note("verify", result=result)
        hold: list[str] = []
        advice_only: list[str] = []
        for verdict in result.get("verdicts") or []:
            if not isinstance(verdict, dict):
                continue
            ident = str(verdict.get("id") or "")
            self.verdicts[f"{ident}|{self.ledger_version}"] = verdict
            if ident in lines and verdict.get("fits") is False:
                grounded = self.grounded_failures(self.listings.get(ident), verdict)
                if grounded:
                    hold.append(f"{ident} does not meet: " + "; ".join(grounded[:4]))
                else:
                    advice_only.append(f"{ident}: " + "; ".join(_short(str(x), 80) for x in (verdict.get("failed") or verdict.get("unknown") or [])[:3]))
        cheaper = [str(c) for c in (result.get("cheaper_fits") or []) if str(c) in self.listings and str(c) not in lines]
        if cheaper and self.objective == "cheapest":
            hold.append("cheaper opened listings appear to fit: " + ", ".join(cheaper[:4]))
        worth = [str(w) for w in (result.get("worth_opening") or []) if str(w) in self.listings and not self.listings[str(w)].opened]
        if worth and self.objective == "cheapest" and self.steps_left() > self.finish_reserve() + len(worth[:3]) and self.holds["order"] < 2:
            hold.append("cheaper rows in the right category should be opened first: " + ", ".join(worth[:3]))
        for rule in result.get("rules") or []:
            if not isinstance(rule, dict) or rule.get("satisfied") is not False or self.steps_left() <= 3:
                continue
            if rule.get("needs_message") and (not self.roles.get("message") or self.message_refused or self.messages_left() == 0):
                continue      # nothing the agent can still do about it
            hold.append(f"rule not yet followed: {rule.get('rule')} ({rule.get('why')})")
        if hold:
            self.holds["order"] += 1
            return "Not sent (no step used). An independent check found: " + " | ".join(hold) + f". {result.get('advice') or ''} Fix this, then order again; if you are certain the check is wrong, call the order tool once more."
        preferred = self.better_stated(lines)
        if preferred:
            self.holds["order"] += 1
            return "Not sent (no step used). " + preferred + " Switch the cart to one of them, or order again if you have a reason to keep the current line."
        if advice_only:
            self.verify_advice = "The independent check could not confirm: " + " | ".join(advice_only) + ". It cited no property of the listing, so the order went ahead."
        self.last_verify_ok = True
        return None

    def better_stated(self, lines: dict[str, int]) -> str:
        """Once per cart composition: when a cart line leaves a requirement unstated, name opened candidates whose records state every requirement."""
        key = json.dumps(sorted(lines))
        if key in self.preferred_told:
            return ""
        messages = []
        for ident in lines:
            listing = self.listings.get(ident)
            if listing is None:
                continue
            product = self.product_of(listing) if len(self.items) > 1 else 0
            if product < 0:
                continue
            requirements = self.requirements_for(product)
            unstated = [r for r in requirements if _evidence(listing, r)[0] not in ("stated", "title")]
            if not unstated:
                continue
            better = [l for l in self.plausible_candidates(product) if l.id not in lines and all(_evidence(l, r)[0] == "stated" for r in requirements)]
            if not better:
                continue
            shown = "; ".join(f"{l.id} ({l.price:g})" if l.price is not None else l.id for l in better[:4])
            messages.append(f"{ident} does not state {', '.join(_short(r, 50) for r in unstated[:3])} in its own record, while {shown} state every requirement.")
        if messages:
            self.preferred_told.add(key)
        return " ".join(messages)

    def grounded_failures(self, listing: Listing | None, verdict: dict[str, Any]) -> list[str]:
        """The checker's failed entries that quote a property the listing really has (the only kind that can hold an order)."""
        if listing is None:
            return []
        keys = {" ".join(str(k).lower().replace("_", " ").replace(".", " ").split()): str(v) for k, v in listing.facts.items()}
        out = []
        for entry in verdict.get("failed") or []:
            if isinstance(entry, dict) and _label_states(listing, str(entry.get("requirement") or "")):
                continue      # the store's own category label states this value: no property can contradict it
            if isinstance(entry, dict):
                key = " ".join(str(entry.get("key") or "").lower().replace("_", " ").replace(".", " ").split())
                if key and key in keys:
                    out.append(f"{entry.get('requirement')} (the record's {entry.get('key')} says {_short(keys[key], 40)})")
            elif isinstance(entry, str):
                text = entry.lower()
                if any(k and k in text for k in keys):
                    out.append(entry)
        return out

    def verdict_contradicts(self, listing: Listing) -> bool:
        verdict = self.verdicts.get(f"{listing.id}|{self.ledger_version}") or {}
        return verdict.get("fits") is False and bool(self.grounded_failures(listing, verdict))

    def basket_summary(self) -> str:
        lines = self.intended_lines()
        parts = []
        total = 0.0
        for ident, qty in lines.items():
            listing = self.listings.get(ident)
            title = _short(listing.title, 60) if listing and listing.title else ident
            price = f" at {listing.price:g} each" if listing and listing.price is not None else ""
            if listing and listing.price is not None:
                total += listing.price * qty
            parts.append(f"{qty} x {title} ({ident}){price}")
        text = "Here is the final basket: " + "; ".join(parts) + f". Total {total:g} {self.budget_unit}".rstrip() + ". Do you approve this basket so I can place the order?"
        return text

    def tell_basket(self) -> str | None:
        """Send the basket summary the request asks for; returns the shopper's reply or None."""
        if not self.intended_lines() or not self.roles.get("message"):
            return None
        reply = self.send_message(self.basket_summary(), "")
        if reply is None:
            return None
        self.basket_told_at = self.step
        self.interpret(reply, self.basket_summary(), None)
        return reply

    def plausible_candidates(self, product: int = 0) -> list[Listing]:
        """Opened, affordable, in-stock listings of the requested kind whose facts contradict no requirement."""
        out = []
        for listing in self.candidates():
            if not listing.opened or listing.in_stock is False or not self.affordable(listing, self.items[product]["quantity"] if product < len(self.items) else 1):
                continue
            if len(self.items) > 1 and self.product_of(listing) != product:
                continue
            if len(self.items) == 1 and self.category_fits(listing) is False:
                continue
            if self.verdict_contradicts(listing):
                continue
            if any(_evidence(listing, r)[0] == "other" for r in self.requirements_for(product)):
                continue
            out.append(listing)
        return out

    def verify_decline(self) -> str | None:
        if self.holds["decline"] >= 2 or self.steps_left() <= 4:
            return None
        plausible = [l for i in range(len(self.items)) for l in self.plausible_candidates(i)]
        if plausible and self.roles.get("cart_add") and self.roles.get("order"):
            self.holds["decline"] += 1
            shown = "; ".join(f"{l.id} ({l.price:g}, {_short(l.title, 40)})" for l in plausible[:5] if l.price is not None) or ", ".join(l.id for l in plausible[:5])
            return ("Not sent (no step used). A wrong decline scores 0, and these opened, affordable, in-stock listings contradict no current requirement: " + shown +
                    ". Put the best of them in the cart and order it; decline only if every one of them clearly fails a requirement.")
        if not self.checks_allowed():
            return None
        self.holds["decline"] += 1
        opened = [l for l in self.candidates() if l.opened]
        unopened = [l for l in self.candidates() if not l.opened and self.affordable(l)]
        prompt = DECLINE_PROMPT + "\n\nTHE SHOPPER'S REQUEST:\n" + self.request
        prompt += "\n\nCURRENT REQUIREMENTS:\n" + self.ledger()
        prompt += "\n\nOPENED LISTINGS:\n" + (self.records_for([l.id for l in opened[:14]], 1100) or "(none)")
        prompt += "\n\nROWS SEEN BUT NOT OPENED (cheapest first):\n" + ("\n".join(l.summary() for l in unopened[:16]) or "(none)")
        prompt += "\n\nSEARCHES MADE:\n" + ("\n".join(self.searches[-12:]) or "(none)")
        result = self.ask_json(prompt, 1400, [self.models["reader"], self.models["main"], self.models["alt"]], require=("possible_fits", "worth_opening", "reason")) or {}
        self.note("decline_check", result=result)
        hold: list[str] = []
        fits = [str(x) for x in (result.get("possible_fits") or []) if str(x) in self.listings]
        if fits:
            hold.append("these listings may meet every requirement: " + ", ".join(fits[:5]))
        worth = [str(x) for x in (result.get("worth_opening") or []) if str(x) in self.listings and not self.listings[str(x)].opened]
        if worth and self.steps_left() > self.finish_reserve():
            hold.append("open these before giving up: " + ", ".join(worth[:4]))
        more = str(result.get("search_more") or "").strip()
        if more and more.lower() not in ("null", "none", "no") and self.steps_left() > self.finish_reserve() + 2 and len(self.searches) < 6:
            hold.append("a search still worth doing: " + more)
        if hold:
            return "Not sent (no step used). A wrong decline scores 0. " + " | ".join(hold) + ". Do that first; decline again only if nothing fits."
        return None

    def clean_remark(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if self.plan.get("no_substitutes") is False:
            return args
        properties = (self.schemas.get(name) or {}).get("properties") or {}
        cleaned = dict(args)
        for key, value in args.items():
            spec = properties.get(key) if isinstance(properties.get(key), dict) else {}
            if not isinstance(value, str) or spec.get("enum"):
                continue
            dirty = any(token in self.listings for token in re.split(r"[\s.,;:!?()\[\]{}<>\"'`]+", value))
            dirty = dirty or any(l.title and len(l.title) > 12 and l.title.lower()[:30] in value.lower() for l in self.listings.values())
            if dirty or self.plan.get("no_substitutes"):
                cleaned[key] = _short("Sorry, nothing in the store meets all of your requirements, so I am not placing an order.", spec.get("maxLength", 400))
        return cleaned

    # -- running tool calls ------------------------------------------------------------------

    def allowed_tools(self) -> list[dict[str, Any]]:
        left = self.steps_left()
        order, decline = self.roles.get("order"), self.roles.get("decline")
        keep: set[str] | None = None
        if left <= 1:
            keep = {order, decline}
        elif left <= 3:
            keep = {order, decline, self.roles.get("cart_add"), self.roles.get("cart_remove")}
        elif left <= self.finish_reserve() and (self.order_attempts == 0 or self.refusals):
            keep = {order, decline, self.roles.get("cart_add"), self.roles.get("cart_remove"), self.roles.get("cart_view")}
        tools = self.tools
        if keep is not None and order:
            narrowed = [t for t in self.tools if t["function"]["name"] in keep]
            if narrowed:
                tools = narrowed
        return tools

    def run_calls(self, calls: list[dict[str, Any]]) -> list[tuple[str, str]]:
        replies: list[tuple[str, str]] = []
        allowed = {t["function"]["name"] for t in self.allowed_tools()}
        finishing = {self.roles.get("order"), self.roles.get("decline")} - {None}
        message_tool = self.roles.get("message")
        held = False
        for call in calls:
            call_id = call["id"]
            function = call.get("function") or {}
            name = function.get("name")
            raw = function.get("arguments")
            if held:
                replies.append((call_id, "Not run (no step used): an earlier call in this reply was held; decide again."))
                continue
            if name not in self.schemas:
                replies.append((call_id, f"Not run (no step used): '{name}' is not a tool of this store. Tools: {', '.join(sorted(self.schemas))}."))
                continue
            if name not in allowed:
                replies.append((call_id, "Not run (no step used): too few steps remain for this; only these tools can be used now: " + ", ".join(sorted(allowed)) + "."))
                held = True
                continue
            if isinstance(raw, dict):
                args: Any = raw
            else:
                args = _parse_json(raw or "{}") if (raw or "").strip() else {}
                if args is None:
                    replies.append((call_id, "Not run (no step used): the arguments were not a valid JSON object."))
                    continue
            args, problems = _check(args, self.schemas[name], "args")
            if problems:
                replies.append((call_id, "Not run (no step used): invalid arguments: " + "; ".join(problems[:6]) + ". Schema: " + _compact(self.schemas[name], 600)))
                held = True
                continue
            mode = self.finish_mode(name, args)
            if mode and len(calls) > 1:
                replies.append((call_id, "Not run (no step used): an order or a decline must be sent alone, after you have read your other results."))
                held = True
                continue
            signature = name + _compact(args, 400)
            if mode is None and name not in (message_tool, self.roles.get("cart_add"), self.roles.get("cart_remove")) and signature in self.recent_calls:
                replies.append((call_id, "Not run (no step used): you made this exact call already and nothing has changed since; its result is above. Nothing new will arrive without a different action: open or search for another candidate, change the cart, or finish with the order or decline tool."))
                continue
            if name == message_tool:
                if not self.planned and not self.pending_reply and not self.message_refused and self.question_phase("forced"):
                    replies.append((call_id, "Not run (no step used): the questions to the shopper were planned and sent for you instead; their answers are in the status below. Apply every revealed need, then continue."))
                    held = True
                    continue
                hold = self.gate_message(args)
                if hold:
                    replies.append((call_id, hold))
                    held = True
                    continue
            if name in (self.roles.get("cart_add"), self.roles.get("order")) and not self.planned and self.roles.get("message") and not self.message_refused:
                # ask the shopper about hidden needs before committing to a listing
                if self.question_phase("forced"):
                    replies.append((call_id, "Not run (no step used): before committing, the shopper was asked about hidden needs; their answers are in the status below. Re-check your choice against the requirements now, then continue."))
                    held = True
                    continue
            if mode == "order":
                hold = self.verify_order()
                if hold:
                    replies.append((call_id, hold))
                    held = True
                    continue
            if mode == "decline":
                hold = self.verify_decline()
                if hold:
                    replies.append((call_id, hold))
                    held = True
                    continue
                args = self.clean_remark(name, args)
            replies.append((call_id, self.perform(name, args)))
            if self.finished or self.dead:
                for later in calls[calls.index(call) + 1:]:
                    replies.append((later["id"], "Not run: the episode has ended."))
                break
        return replies

    def gate_message(self, args: dict[str, Any]) -> str | None:
        left = self.messages_left()
        if left is not None and left <= 0:
            return "Not run (no step used): the message budget is used up; do not message the shopper again."
        text = " ".join(str(v) for v in args.values() if isinstance(v, str))
        if self.pending_reply:
            return None
        if not self.messages_required and self.planned and self.holds["message"] < 2:
            self.holds["message"] += 1
            return "Not run (no step used): questions about hidden needs are planned and sent for you; the remaining messages are kept for what the request's procedure requires. If a rule requires this message, send it again."
        if text.count("?") > 1 or re.search(r"\bor\b.*\?", text) and len(text) > 140:
            if self.holds["message"] < 3:
                self.holds["message"] += 1
                return "Not run (no step used): a message must raise exactly one point in one short sentence. Rewrite it."
        return None

    def perform(self, name: str, args: dict[str, Any]) -> str:
        """Run one checked tool call and return the text the model sees."""
        mode = self.finish_mode(name, args)
        reply = self.call(name, args)
        if reply is None:
            if self.finished:
                return "The episode has ended."
            return "The call could not be completed (network or server problem). Try once more or choose another action."
        obs = reply.get("observation")
        if mode is None and name not in (self.roles.get("cart_add"), self.roles.get("cart_remove"), self.roles.get("message")):
            self.recent_calls = (self.recent_calls + [name + _compact(args, 400)])[-3:]
        else:
            self.recent_calls = []
        if mode == "order":
            self.order_attempts += 1
        notes: list[str] = []
        if name == self.roles.get("message"):
            lines_before = len(self.heard)
            if self.confirm_before_order and self.cart and not _result_failed(obs):
                self.basket_told_at = self.step    # the model told the shopper itself; the harness must not repeat it
            if _result_failed(obs):
                self.message_refused = True
                self.message_budget = self.messages_used
                notes.append("The store refused this message: the message budget is used up.")
            else:
                self.messages_used += 1
                text_reply = obs if isinstance(obs, str) else " ".join(v for v in obs.values() if isinstance(v, str)) if isinstance(obs, dict) else _compact(obs, 300)
                question = " ".join(str(v) for v in args.values() if isinstance(v, str))
                self.asked.append(("", question, text_reply))
                self.heard.append(text_reply)
                if text_reply.strip():
                    reading = self.interpret(text_reply, question, None)
                    if reading.get("meaning"):
                        notes.append("Meaning: " + str(reading.get("meaning")))
                    if self.confirm_before_order and self.cart and not any(reading.get(k) for k in ("needs", "dropped", "new_budget", "new_quantity", "unavailable", "price_changed", "reference")) and reading.get("kind") not in ("question", "unclear"):
                        notes.append("The shopper raised no objection and asked for no change: the rule to run the basket by them is satisfied. Do not wait for further words; order now if everything else is in place.")
                    if reading.get("needs"):
                        notes.append("New requirement(s): " + "; ".join(str(n) for n in reading["needs"]))
            del lines_before
        else:
            self.map_result_keys(name, obs)
            lines = self.absorb(name, args, obs)
            self.cart_effect(name, args, obs)
            if name in (self.roles.get("details"), self.roles.get("compare")):
                for row in _walk_listings(obs):
                    ident = _is_listing(row)
                    if ident:
                        self.reread_at[ident] = self.step
            if name == self.roles.get("cart_view") and _walk_listings(obs):
                self.cart_from_rows(_walk_listings(obs))
            fresh = [l for l in lines if l not in self.heard]
            refused = isinstance(obs, dict) and any(_REFUSAL_KEY.search(str(k)) for k in obs) and mode is not None and not reply.get("finished")
            if mode is not None and not reply.get("finished") and not refused:
                refused = bool(fresh) or (not self.finished)
            if refused and mode == "order":
                self.refusals += 1
                self.refused_at = self.step
                self.last_verify_key = ""
                notes.append("THE ORDER WAS REFUSED and the episode continues: nothing has been bought. Apply the shopper's line below, fix the cart, re-read the pick, and order again.")
            if refused and mode == "decline":
                notes.append("THE DECLINE WAS NOT ACCEPTED and the episode continues. Apply the shopper's line below and decide again.")
            if fresh:
                self.heard.extend(fresh)
                reading = self.interpret("\n".join(fresh), None, None)
                meaning = str(reading.get("meaning") or "")
                notes.append("Shopper line(s) in this result: " + " / ".join("\"" + _short(line, 160) + "\"" for line in fresh) + (f" => {meaning}" if meaning else ""))
                if reading.get("needs"):
                    notes.append("New requirement(s): " + "; ".join(str(n) for n in reading["needs"]))
                if reading.get("dropped"):
                    notes.append("Dropped: " + "; ".join(str(n) for n in reading["dropped"]))
                if reading.get("new_budget") is not None:
                    notes.append(f"New budget: {reading.get('new_budget')}")
                if reading.get("unavailable"):
                    notes.append("The picked item can no longer be bought: remove it from the cart and take the next verified fit.")
            if isinstance(obs, dict) and obs.get("error") == "budget_exceeded" or (isinstance(obs, dict) and any(isinstance(v, str) and "budget_exceeded" in v for v in obs.values())):
                notes.append("That tool's allowance is used up; do not call it again.")
        text = _compact(_compact_obs(obs), OBS_CHARS)
        if mode == "order" and self.verify_advice:
            notes.append(self.verify_advice)
            self.verify_advice = ""
        if reply.get("finished"):
            text += " [the episode is finished]"
        if notes:
            text += "\n[note] " + " ".join(notes)
        return text

    def force_finish(self) -> None:
        if self.finished or self.steps_left() <= 0 or self.time_left() < 2:
            return
        lines = self.intended_lines()
        gone = [i for i in lines if i in self.listings and self.listings[i].in_stock is False]
        remover = self.roles.get("cart_remove") or self.roles.get("cart_add")
        if gone and remover and self.time_left() > 8 and self.steps_left() >= len(gone) + 2:
            for ident in gone:
                args = self.remove_args(ident, lines[ident])
                if args is None:
                    continue
                reply = self.call(remover, args)
                self.note("forced_release", listing=ident, reply=_compact((reply or {}).get("observation"), 160))
                if reply is not None and not _result_failed(reply.get("observation")):
                    self.cart.pop(ident, None)
            lines = self.intended_lines()
        fits = [i for i in lines if i in self.listings and (self.verdicts.get(f"{i}|{self.ledger_version}") or {}).get("fits") is not False and self.affordable(self.listings[i])]
        if self.roles.get("cart_add") and self.roles.get("order") and self.time_left() > 6:
            missing = [index for index in range(len(self.items)) if not any(i in self.listings and self.product_of(self.listings[i]) == index for i in lines)]
            if len(self.items) == 1 and lines:
                missing = []
            if missing and self.steps_left() >= len(missing) + 1:
                for index in missing:
                    best = self.best_plausible(index)
                    if not best:
                        continue
                    args = self.cart_args(best, self.items[index]["quantity"])
                    if args is None:
                        continue
                    reply = self.call(self.roles["cart_add"], args)
                    self.note("forced_cart", listing=best, product=index, reply=_compact((reply or {}).get("observation"), 200))
                    if reply is not None and not _result_failed(reply.get("observation")):
                        self.cart[best] = self.items[index]["quantity"]
                lines = self.intended_lines()
                fits = [i for i in lines if i in self.listings and (self.verdicts.get(f"{i}|{self.ledger_version}") or {}).get("fits") is not False and self.affordable(self.listings[i])]
        if lines and self.confirm_before_order and self.basket_told_at < self.cart_changed_at and self.steps_left() >= 2:
            self.tell_basket()
        if lines and self.roles.get("order"):
            call = self.finish_call("order", "Order the cart.")
            if call:
                reply = self.call(call[0], call[1])
                self.note("forced_finish", action=call, finished=self.finished)
                if self.finished or reply is None or self.steps_left() <= 0:
                    return
        call = self.finish_call("decline", "Sorry, nothing in the store meets all of your requirements, so I am not placing an order.")
        if call:
            self.call(call[0], call[1])
            self.note("forced_finish", action=call, finished=self.finished)

    def best_plausible(self, product: int = 0) -> str | None:
        """The opened, affordable listing of one product whose facts state the most requirements (and contradict none), cheapest first."""
        ranked = []
        quantity = self.items[product]["quantity"] if product < len(self.items) else 1
        for listing in self.candidates():
            if not listing.opened or not self.affordable(listing, quantity):
                continue
            if len(self.items) > 1 and self.product_of(listing) != product:
                continue
            verdict = self.verdicts.get(f"{listing.id}|{self.ledger_version}") or {}
            if self.verdict_contradicts(listing):
                continue
            stated = other = 0
            for requirement in self.requirements_for(product):
                kind, _ = _evidence(listing, requirement)
                stated += kind == "stated"
                other += kind == "other"
            if other:
                continue
            ranked.append((-(stated + (2 if verdict.get("fits") is True else 0)), listing.price if listing.price is not None else 1e18, listing.id))
        ranked.sort()
        return ranked[0][2] if ranked else None

    def remove_args(self, ident: str, quantity: int) -> dict[str, Any] | None:
        """Arguments that take one listing out of the cart, read from the remove tool's schema (or the add tool's removing mode)."""
        name = self.roles.get("cart_remove") or self.roles.get("cart_add")
        schema = self.schemas.get(name or "") or {}
        properties = schema.get("properties") or {}
        args: dict[str, Any] = {}
        placed = False
        for key, spec in properties.items():
            if not isinstance(spec, dict):
                continue
            kind = spec.get("type")
            if spec.get("enum"):
                options = [str(v) for v in spec["enum"]]
                args[key] = next((v for v in options if _REMOVE_WORD.search(v)), None) or next((v for v in options if not _ADD_WORD.search(v)), options[-1])
            elif kind in ("integer", "number"):
                args[key] = max(quantity, spec.get("minimum", 1) if isinstance(spec.get("minimum"), (int, float)) else 1)
            elif kind == "string" and not placed:
                args[key] = ident
                placed = True
            elif kind == "array" and not placed and isinstance(spec.get("items"), dict):
                args[key] = [ident] if spec["items"].get("type") != "object" else [{k: (ident if v.get("type") == "string" else quantity) for k, v in (spec["items"].get("properties") or {}).items() if isinstance(v, dict)}]
                placed = True
        if not placed or name == self.roles.get("cart_add") and not any(isinstance(s, dict) and s.get("enum") for s in properties.values()):
            return None if not placed or name != self.roles.get("cart_remove") else args
        args, problems = _check(args, schema, "args")
        return None if problems else args

    def cart_args(self, ident: str, quantity: int | None = None) -> dict[str, Any] | None:
        """Arguments that put one listing in the cart at the required quantity, read from the cart tool's schema."""
        name = self.roles.get("cart_add")
        schema = self.schemas.get(name or "") or {}
        properties = schema.get("properties") or {}
        if quantity is None:
            quantity = self.items[0]["quantity"] if self.items else 1
        args: dict[str, Any] = {}
        placed = False
        for key, spec in properties.items():
            if not isinstance(spec, dict):
                continue
            kind = spec.get("type")
            if spec.get("enum"):
                options = [str(v) for v in spec["enum"]]
                args[key] = next((v for v in options if _ADD_WORD.search(v)), None) or next((v for v in options if not _REMOVE_WORD.search(v)), options[0])
            elif kind in ("integer", "number"):
                args[key] = max(quantity, spec.get("minimum", 1) if isinstance(spec.get("minimum"), (int, float)) else 1)
            elif kind == "string" and not placed:
                args[key] = ident
                placed = True
            elif kind == "array" and not placed and isinstance(spec.get("items"), dict):
                item_schema = spec["items"]
                if item_schema.get("type") == "object":
                    line: dict[str, Any] = {}
                    for sub_key, sub_spec in (item_schema.get("properties") or {}).items():
                        if isinstance(sub_spec, dict) and sub_spec.get("type") == "string":
                            line[sub_key] = ident
                        elif isinstance(sub_spec, dict) and sub_spec.get("type") in ("integer", "number"):
                            line[sub_key] = quantity
                    args[key] = [line]
                else:
                    args[key] = [ident]
                placed = True
        if not placed:
            return None
        args, problems = _check(args, schema, "args")
        return None if problems else args

    # -- the loop -------------------------------------------------------------------------

    def trim_history(self) -> None:
        total = sum(len(str(m.get("content") or "")) for m in self.history)
        index = 2
        while total > HISTORY_CHARS and index < len(self.history) - 6:
            message = self.history[index]
            if message.get("role") == "tool" and len(str(message.get("content") or "")) > 400:
                stub = "(older result trimmed; everything learned from it is in the status card)"
                total -= len(str(message["content"])) - len(stub)
                message["content"] = stub
            index += 1

    def run(self) -> list[dict[str, Any]]:
        if self.finished:
            return self.trace
        self.setup()
        self.note("roles", roles=self.roles, items=self.items, budget=self.budget, objective=self.objective, message_budget=self.message_budget, max_steps=self.max_steps)
        if self.opening is not None:
            self.absorb("", {}, self.opening)
        user = "THE SHOPPER'S REQUEST:\n" + self.request
        if self.opening is not None:
            user += "\n\nWHAT THE STORE SHOWED FIRST:\n" + _compact(_compact_obs(self.opening), 4000)
        user += "\n\nA first reading of the request (check it against the request itself):\n" + self.ledger()
        tool_roles = ", ".join(f"{role}: {name}" for role, name in self.roles.items())
        if tool_roles:
            user += "\n\nTool roles as read from the schemas: " + tool_roles + "."
        user += "\n\nStart by finding the right category and listings; open the properties of two or three fitting candidates early."
        self.history = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]
        failures = 0
        counter = 0
        # questions the request itself leaves to the shopper come first
        if self.ask_first and self.roles.get("message"):
            if self.question_phase("first"):
                self.history.append({"role": "user", "content": "The shopper answered the first question(s); see the status card."})
        for _ in range(MAX_TURNS):
            if self.finished or self.steps_left() <= 0:
                break
            if self.time_left() < FORCE_SECONDS or self.dead:
                self.force_finish()
                break
            if self.pending_reply and self.roles.get("message"):
                text, self.pending_reply = self.pending_reply, ""
                reply = self.send_message(text, "")
                if reply:
                    self.interpret(reply, text, None)
                    self.history.append({"role": "user", "content": f"(The shopper's question was answered: \"{text}\" -> \"{_short(reply, 200)}\")"})
                    continue
            if not self.planned and self.roles.get("message") and len([l for l in self.listings.values() if l.facts]) >= 2:
                if self.question_phase("ready"):
                    self.history.append({"role": "user", "content": "The shopper has been asked about hidden needs; their answers are in the status card. Apply every revealed need as a hard requirement."})
            self.trim_history()
            message = self.infer(self.history + [{"role": "user", "content": self.status()}], self.allowed_tools(), [self.models["main"], self.models["alt"], self.models["main"]], 1800)
            if message is None:
                failures += 1
                if failures >= 2 or self.dead:
                    self.force_finish()
                    break
                continue
            failures = 0
            calls = []
            for call in message.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                counter += 1
                function = call.get("function") or {}
                arguments = function.get("arguments")
                calls.append({"id": call.get("id") or f"call_{counter}", "type": "function", "function": {"name": function.get("name"), "arguments": arguments if isinstance(arguments, str) else json.dumps(arguments or {})}})
            content = message.get("content") or ""
            if not isinstance(content, str):
                content = _compact(content, 2000)
            if not calls:
                self.idle += 1
                self.history.append({"role": "assistant", "content": content or "(no tool call)"})
                self.history.append({"role": "user", "content": "Every reply must call one of the tools. Continue."})
                if self.idle >= 3:
                    self.force_finish()
                    break
                continue
            self.idle = 0
            self.note("model", content=content[:1500], calls=[c["function"] for c in calls])
            self.history.append({"role": "assistant", "content": content or None, "tool_calls": calls})
            replies = self.run_calls(calls)
            for call_id, text in replies:
                self.history.append({"role": "tool", "tool_call_id": call_id, "content": text})
            answered = {r[0] for r in replies}
            for call in calls:
                if call["id"] not in answered:
                    self.history.append({"role": "tool", "tool_call_id": call["id"], "content": "Not run: the episode has ended."})
        if not self.finished:
            self.force_finish()
        self.note("end", finished=self.finished, steps=self.step, cost=round(self.cost, 5), llm_calls=self.llm_calls, seconds=round(_now() - self.started, 1))
        return self.trace


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    episode = Episode(problem_data)
    try:
        return episode.run()
    except Exception as exc:  # keep the trajectory and still finish when the world allows it
        episode.note("crash", error=f"{type(exc).__name__}: {exc}"[:500])
        try:
            if not episode.finished:
                episode.force_finish()
        except Exception as again:
            episode.note("crash", error=f"{type(again).__name__}: {again}"[:500])
        return episode.trace


__all__ = ["agent_main"]
