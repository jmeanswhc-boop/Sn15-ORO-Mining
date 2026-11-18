from __future__ import annotations

import json
import re
import time
from typing import Any
from enum import Enum
from src.agent.proxy_client import ProxyClient

MAIN_MODEL = "anthropic/claude-haiku-5.5"
BACKUP_MODEL = "openai/gpt-6-luna"
SETUP_MODEL = "anthropic/claude-sonnet-5.5"
JUDGE_MODEL = "anthropic/claude-sonnet-5.5"
READ_MODEL = "anthropic/claude-haiku-5.5"

TASK_SECONDS = 265.0
COST_SOFT = 0.06
WRAP_UP_SECONDS = 70.0
FORCE_SECONDS = 28.0
OBS_CHARS = 16000
MAX_TURNS = 80
MIN_TURNS = 1
SYSTEM_PROMPT = """You are a meticulous shopping assistant. You serve one shopper inside a simulated store, and you can act only through the supplied tools. The task is scored 1 or 0. It scores 1 only if your finish is fully right: the ordered item meets every requirement the shopper stated, every hidden need the shopper reveals when asked, and every change the shopper announces while you work; the quantity and the budget are right; the cheapest (or best) choice is made when the request asks for it; and every rule the request states is followed. If nothing in the store can meet everything, the right finish is to decline, exactly the way the request says.

THE WORLD
- Every tool call uses one step, including calls that fail or return errors, and the number of steps is limited. Model thinking is free; steps are not.
- You may put several tool calls in one reply when none depends on another's result (for example, opening the details of several listings at once). They run in order. Never put an order or a decline together with other calls.
- If the store has a tool that shows the full details of several listings in one call, use it instead of opening them one by one: it saves steps.
- Tool names, arguments and the shape of results differ from store to store. Read the tool descriptions and the JSON schemas carefully and use exactly the argument names and allowed values they define. Learn from the first results which tool shows which facts (price, stock, detailed properties).
- Find out early which category labels the store uses for the requested kind of product (they appear on listings). When neighbouring labels exist, the right one is the label that matches the request's wording as a whole; search and buy within it.
- Search tools can be limited: a capped number of visible rows or pages, results sorted by id rather than by relevance or price, every word required, filters that only work in one mode. Compare the reported total with the rows you can actually see, and narrow (more specific words, category or price filters, other pages) until you have seen the whole relevant set.

THE SHOPPER AND HIDDEN NEEDS
- The shopper has needs they did not write down. They reveal one only when a message asks about that property. Each message is answered only about the FIRST thing it asks, so ask about exactly ONE property per message, in the form "Which <property> would you like the <product> to have?" with the property's name as the listings name it, the product named by its plain noun from the request, and no example values. Never give instructions, never mention lists, labels, positions or rules, and never ask two things at once.
- When the request asks for several different products, every question must say which product it is about; a question that leaves the product open confuses the shopper. Spread the questions over the products.
- If the shopper says they did not follow a message, or asks you to explain, simplify or rephrase it, do not explain or repeat it: the property you asked about is most likely not one they care about. Ask about a different property next (naming the product).
- A reply saying they don't mind, have no preference, or anything is fine, means that property is free. A reply that states a preference or a limit is a hard requirement from now on, exactly as worded (numeric limits included).
- The requirements are only what the request states and what the shopper reveals or changes. Never add one of your own from the story (an age group, a gender, a size, who will use it): such details are not requirements, and a listing does not fail because of them. If such a detail may matter, it may be a hidden need; ask about it.
- When a planner is available, your questions are chosen and sent for you as soon as you have opened the details of two or three listings of the requested kind (of each kind, when several different products are requested), so do that in your first moves. The shopper's answers then appear in the conversation with their meaning; apply every revealed need as a hard requirement and search again with it combined with the stated values.
- Messages to the shopper are only for questions (and for answering a question the shopper asked). Never message the shopper to report what you found, what you bought or that you are declining; the finishing tool carries that.
- If you must ask yourself: first open the details of two or three listings, then ask about the properties the story hints at (who it is for, how it must look, where or how it will be used), then about properties the listings carry that differ among candidates. Never ask about something the request already settles, and never exceed the message budget.

RULES OF PROCEDURE
- The request may set rules about how to work: how many listings to look at or compare before choosing, what to check, what to tell the shopper, what to put in a remark, how to decline. These are graded like requirements: follow each one literally and completely. A rule to look at or open N listings means N listings that fully fit, each opened with the details tool.
- If the shopper asks you something (for example to confirm a change), it will be answered for you when messages remain; treat the change itself as binding.
- Something the shopper would merely like ("ideally", "would be nice", "if possible") is not a requirement: use it only to choose between items that already meet everything, and never over a cheapest rule.
- If the request leaves something you can only learn from the shopper (for example no price limit, or two wishes that may not both be possible and the shopper must say which matters more), find it out by asking before you choose.

CHANGES
- While you work, the shopper may change things: a new budget (higher or lower), a requirement dropped, replaced or added, a price going up, an item sold out. One shopper line can carry several changes. Shopper lines can appear inside any tool result (an extra field holding sentences) or attached to an order attempt. Each change is binding from that moment: re-check your pick and your backups against it, and when the request asks for the cheapest or best, choose again over the new acceptable set.
- An order attempt can be REFUSED. The result then carries the reason in some field (a refusal, complaints, objections or similar), even if the inner result looks successful (accepted: true, placed: true, executed: true). An order that goes through ends the episode: while the episode continues, nothing has been bought, whatever the result says. The episode continues and the cart is unchanged. Read the refusal, apply every change it states, remove anything that is no longer acceptable from the cart (for example a sold-out item), add the next verified fit, re-check stock and price, and order again.
- Items can also sell out or change price silently. That is why stock and price must be re-checked right before every order attempt; an item whose current price times quantity is over the current budget no longer fits.

HOW TO JUDGE A LISTING (this decides the score)
- A requirement is met only when the listing's own property for it states the value. Property keys vary in spelling (spaces, dots, underscores, prefixes) and the same property may appear under several keys; any key that names that property counts. Values match by meaning: one element of a comma-separated list is enough; singular and plural are the same; close synonyms count; case does not matter. A word that merely starts the same way does not count, and a broader or different class does not count (for example girls is not women, plain unisex is not female).
- Use the title or description as evidence only when the listing has no property key for that requirement at all. If the listing's own property says something different, the listing fails even when the title or description says the value.
- A value under a different, related property does not count (a length field does not satisfy a style requirement, a material of the lining does not satisfy the material).
- Numbers: compare as numbers after converting units; limits include the edge value; a range counts only when the whole range meets the limit.
- Respect exclusions ("leave out anything with X") and closed lists ("A or B and nothing else").
- The requested kind of product is met by the listing's category; never demand that some property repeat the category's name. The item must be the requested kind of product: when the store labels listings with categories, it must sit in the category whose label matches the request's wording as a whole (a request for "A or B" matches a category called "A & B", not a narrower neighbour called "B"). It must be in stock when the request asks for things on hand, and price times quantity must be within the budget (equal is allowed).
- When a listing offers options or variants (sizes, colours, versions) with their own ids, prices or stock, each option is its own candidate: a requirement is met by the option's own value, its own price and stock count, and the cart line must name the chosen option the way the cart tool's schema asks.
- A property whose value lists several options including the required one (a comma list) simply matches; that is not mixed evidence. Evidence is mixed only when the listing's own keys for the property state different, incompatible values. Only then, and only when no cheapest or best rule decides, prefer a candidate whose evidence is clean.

CHEAPEST OR BEST
- When the request asks for the cheapest, the order must be the lowest-priced item among ALL items that meet every requirement. Seeing one fitting item is not enough. Use price filters when a tool offers them, lower the price limit until every remaining row is visible, sort the visible candidates by price yourself, and verify them in ascending price; the first one that fully fits, with every cheaper one checked and rejected, is the answer. Keep the next fitting ones as backups.
- When the request asks for the best by some measure (most, largest, longest, highest, and so on), compare ALL fitting items on that measure the same way and pick the extreme. An item whose listing states no value for that measure cannot win.

FINISHING
- Keep a ranked list of verified fits, with at least one backup, because your first pick may sell out when you check out.
- Before every order attempt: the cart holds exactly the lines the request needs (usually one: the chosen item at the required quantity; one line per item when the request asks for several different items) and nothing else (remove any other line); you have re-read the chosen item's stock and price after your last cart change, and it is in stock and within budget.
- Declining is rarely the right finish: in most tasks the store holds an item that meets every requirement, and a wrong decline scores 0. Decline only after a thorough search, and only when, for every candidate, you can name a stated requirement, a revealed need, the budget or stock that its own records clearly fail. Doubt, a missing nice-to-have or a requirement you only assumed is not a reason to decline; order the best verified fit instead. Follow the request's rule for declining exactly. If the request says not to offer substitutes, the decline remark must not name, describe or point to any listing: no ids, no titles, no prices.
- Plan your steps: get your first order attempt in while at least seven steps remain, so a refusal can still be handled. When 7 steps remain, only cart, stock-check and finishing tools stay available. Decide once: do not swap the cart back and forth, since every cart change costs a step.
- Every reply must contain at least one tool call. Keep the reply text short: your current requirements, the verified fits with price and stock, and your next move."""

SETUP_PROMPT = """Read the shopper's request and the store's tool list below. Return ONE compact JSON object (no indentation) and nothing else, with these keys:

"product": the kind of product requested, in the request's words (when the request asks for several different items, name each of them).
"quantity": the number of units of one listing to order (an integer; 1 when the request does not ask for several). Something naturally sold as one item, such as a pair of shoes or a set, is quantity 1.
"items": when the request asks for several DIFFERENT products, one object per product: {"product": a short name for it in the request's words, "quantity": the units of that product (integer)}; an empty list when only one kind of product is requested.
"requirements": a list of strings, one per firm stated requirement on the product, each written as "property: constraint" with the shopper's exact value, limit, exclusion or closed list. When several different products are requested, start each one with the short product name it applies to, as "<product> <property>: constraint". Do not include price, stock, quantity, budgets or mere wishes here, and do not restate the product kind or category as a requirement: words that are part of the category's name (a season, an occasion or an audience in the category name) are covered by the category and are not a separate property requirement.
"price_cap": the maximum total price as a number, or null when none is stated.
"currency": the price unit the request uses, or null.
"in_stock_only": true when the request wants only items actually on hand or in stock.
"priority": "cheapest" only when the request explicitly asks for the cheapest or lowest-priced fitting option; "best" only when it explicitly asks for the best by some measure; otherwise "none". Remarks that money is tight or that every penny counts are not a request for the cheapest; they only explain the price cap.
"priority_detail": for "best", what "best" is measured by; otherwise "".
"message_budget": the number of messages the shopper allows you to send, as an integer, or null when not stated.
"step_budget": the number of moves or steps the request allows, as an integer, or null when not stated.
"rules": a list of every procedural rule the request states for the assistant (how many listings to look at, what to check or tell, how to decline, what not to do, what to include), quoted word for word. Remarks about what the shopper will do themselves are story, not rules.
"soft_wants": a list of things the shopper would like but does not strictly require (phrased as ideally, preferably, would be nice, if possible). Keep these out of "requirements".
"ask_first": a list of things the agent can only learn from the shopper and should ask about before choosing: for example the budget when no price limit is stated, which of two wishes matters more when the request says both may not be possible (then also put both wishes in soft_wants, not in requirements), or anything the request says to check with the shopper. Empty when none.
Every sentence of the request must land somewhere: a requirement, the priority, a soft want, a rule, a budget, a hint, or pure story. When unsure whether a sentence is a rule, put it in "rules".
"decline_no_names": true when the request's rule for declining forbids offering, naming or suggesting substitutes or alternatives; false when it asks you to name something; null when the request gives no rule.
"roles": an object naming the tool for each role, using the exact tool names from the list, or null when no tool fits:
  "search", "details", "view", "message", "cart", "cart_view", "order", "decline".
  "message" is the tool that sends a message to the shopper. "cart" is the tool that adds to or removes from the cart. "order" is the tool that buys or checks out. "decline" is the tool that ends the task without buying.
"""

INTERPRET_PROMPT = """You help a shopping assistant understand what a shopper just said. Return ONE JSON object and nothing else, with these keys:
"meaning": one short sentence saying what the shopper means.
"needs": a list of new firm requirements on the product, each written as "property: constraint" with the exact value, limit, exclusion or closed list the shopper gave.
"no_preference": a list of properties the shopper said they do not care about.
"dropped": a list of earlier requirements the shopper no longer needs.
"new_budget": the new maximum total price as a number, or null.
"new_quantity": when the shopper changes how many units they want, {"product": the product it is for (or "" when only one kind is requested), "quantity": the new number as an integer}; otherwise null.
"unavailable": true when the shopper says an item can no longer be bought (sold out, gone, out of stock, button disabled), else false.
"requirements_now": the complete list of firm requirements on the product as they stand after everything the shopper has said so far, built from the requirements so far (given below) and what the shopper has explicitly said, each as "property: constraint". Never add a requirement from the story or from your own reading of the request: only values the request states as requirements or the shopper states in a reply count. Keep every earlier requirement unless the shopper dropped or replaced it; for a replaced value keep only the newest one; add every newly revealed need. Do not include price, stock, quantity or the product category.
"unclear": true when the shopper says they did not follow or understand the assistant's message, or asks for it to be explained, simplified, reworded or repeated; otherwise false.
"reply": when the shopper asks the assistant something or asks for a confirmation, a short, plain reply to send back that answers it and names the change in substance (for example confirming the new value); otherwise "". Always "" when "unclear" is true: a request to explain the assistant's own question is not a question to answer.
When the request asks for several different products, start every requirement in "needs" and "requirements_now" with the short name of the product it applies to, as "<product> <property>: constraint" (the shopper often says "<property> of the <product kind>").
One line can carry several changes: report every one. A requirement replaced by another is both dropped (the old one) and a new need (the new one). When the shopper says which of two wishes matters more, the preferred wish is a need. When the shopper states a budget, it is new_budget. A mere wish ("ideally", "would be nice") is not a need. Read the shopper literally and charitably. Folksy, figurative or roundabout phrasing that names a value means the shopper firmly wants that value; treat a value as excluded only when the shopper clearly says to avoid it. A number with a comparison (at most, at least, under, over, no more than) is a limit, kept with its unit. A new budget replaces the old one whether it is higher or lower, even when the shopper calls it good news. When the shopper only says they don't mind, have no preference, or anything works, put the property asked about in no_preference; such a reply applies only to the property the assistant asked about, never drops a stated requirement, a price limit or a rule, and when the assistant's message did not ask about a property it changes nothing."""

VERIFY_PROMPT = """You are the final checker before a shopping assistant places an order. The task scores 1 only if the ordered item meets every requirement below. Judge the item ONLY from the records shown, using these reading rules:
- A requirement is met only when the item's own property for it states the value. Property keys vary in spelling (spaces, dots, underscores, prefixes) and the same property may appear under several keys; any key naming that property counts. Values match by meaning: one element of a comma list is enough, singular/plural and close synonyms count, case does not matter. A merely similar-looking word or a broader/different class does not count.
- The title or description counts only when the item has no property key for that requirement. If its own property says something else, it fails even when the title says the value.
- A value under a different, related property does not count.
- Numbers: compare after converting units; limits include the edge; a range counts only when the whole range meets the limit.
- Exclusions and closed lists must be respected.
- The item must be the requested kind of product; when listings carry a category label, it must be the category whose label matches the request's wording as a whole, not a narrower or neighbouring one.
- When the item's own property for a requirement states a different value, the requirement fails, even if the title suggests otherwise.
- Price times quantity must be within the budget (equal is fine). If the records show it out of stock, it fails.
- A mere wish ("ideally", "would be nice") listed as not required is not a requirement.
- Judge only the requirements listed below; never add one of your own from the story (an age group, a gender, a size, who will use it).
- When the request asks for several different products, judge the item only against the requirements of its own kind of product: a requirement that starts with another product's name does not apply to it, and the quantity is the one for its kind.
When records disagree (for example on price or stock), the newest record counts.
Return ONE JSON object and nothing else: {"item": the item's id, "fits": true or false, "problems": [short strings, each naming a requirement that is not met or not shown]}. Use fits=false only for a requirement that the records clearly fail, or that no record addresses at all; do not fail an item for a missing record of stock or price."""

PLAN_PROMPT = """You plan the questions a shopping assistant sends to a shopper. The shopper has hidden needs: each is a value of one property of the product, revealed only when a message asks about that exact property; each message is answered only about the first thing it asks, and questions about anything else get a polite "don't mind". Choose QUESTION_COUNT different properties (fewer only when fewer useful ones exist), most promising first:
0. first, anything listed below as MUST BE LEARNED FROM THE SHOPPER that has not been asked yet; ask it plainly. For the budget, ask how much they want to spend at most. For a choice between two wishes, name both values with their properties, for example "If one listing can't have both, would you rather have the <value> <property> or the <value> <property>?";
1. then properties that the listings' details below actually carry, preferring (a) properties that most listings carry and whose values differ between them, of any kind (what it is made of, how it looks, how it is built or works, who it is for, technical specifications); a property with the same value in every listing seen is a poor question; and (b) a property the story in the request points to (who will use it, how it must look, where or how it will be used). Avoid logistics (warranty, condition, pack size, years) and values that only list everything on offer (all sizes or variants in one string), and never ask about the brand, maker, seller or shop. A size, colour or other option that a listing, or one of its options, has as its own single value is a normal property.
Never pick a property the request already settles (its stated requirements, the price, stock, quantity), never repeat a property already asked for the same product (a question the shopper did not follow counts as asked: move on, unless that question did not name the product), never spend two questions on near-duplicate properties (two kinds of style, two spellings of one property), and avoid properties no listing carries. Use the property names as the listings name them.
When the request asks for several different products, each question is about ONE of them, and the questions are spread over the products: prefer a product that has had fewer questions so far, and take its properties from the listings of that product's kind (the menu below is grouped by the listings' labels when they carry one).
Write each question as "Which <property> would you like the <product> to have?", with the property's name exactly as the listings name it (spaces instead of underscores or dots) and the product named by its plain noun from the request ("the rug", "the curtains"; never with adjectives, materials, colours or other property values). No example values, no other words, no second question. For something that must be learned first (rule 0), ask it plainly in one short sentence.
Return ONE JSON object and nothing else: {"questions": [{"product": "...", "property": "...", "question": "..."}, ...]}."""

SWEEP_LAST = False
SWEEP_NOTE = """

THE LAST QUESTION IN YOUR LIST IS A SWEEP: write it as "Do you have a requirement for the <product>'s <property>, <property> or <property>?", naming two or three further promising properties of ONE product that no other question covers (names as the listings name them, product by its plain noun). Its "property" field lists them joined by ", "."""

AUDIT_PROMPT = """A shopping assistant is about to tell the shopper that nothing in the store fits. Check the listing records below against the requirements, using these reading rules: a requirement is met when the listing's own property for it states the value (any key spelling, one element of a comma list, singular/plural and close synonyms all count; a broader or different class does not); the title or description counts only when the listing has no key for that property; numbers are compared after converting units, limits include the edge; the price must be within the budget and the item must not be shown out of stock; only the listed requirements count, never one assumed from the story; when several different products are requested, a listing is checked only against the requirements of its own kind of product. A wrong decline scores 0, so list every listing that may fit; leave one out only when its own records clearly fail a listed requirement.
Return ONE JSON object and nothing else: {"possible_fits": [ids of listings that appear to meet every requirement, best first, at most 5], "reason": "one sentence"}."""

_llm = ProxyClient(timeout=70, max_retries=1, retry_delay=1)
_env = ProxyClient(timeout=95, max_retries=2, retry_delay=1)

def _brief(value: Any, limit: int, depth: int = 0) -> Any:
    if depth > 8:
        return None
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + "..."
    if isinstance(value, dict):
        return {k: _brief(v, limit, depth + 1) for k, v in value.items()}
    if isinstance(value, list):
        return [_brief(v, limit, depth + 1) for v in value]
    return value


_PROPERTY_KEYS = re.compile(r"fact|propert|trait|spec|attribut|feature|detail|characteristic", re.I)
def line_of(args: dict[str, Any], ident: str) -> str:
    for value in args.values():
        if isinstance(value, list):
            for line in value:
                if isinstance(line, dict) and ident in line.values():
                    return _compact(line)
    return ""


def has_text(value: Any, depth: int = 0) -> bool:
    if depth > 4:
        return False
    if isinstance(value, str):
        return any(c.isalpha() for c in value) and not (_ID_HINT.match(value) and any(c.isdigit() for c in value))
    if isinstance(value, dict):
        return any(has_text(v, depth + 1) for v in value.values())
    if isinstance(value, list):
        return any(has_text(v, depth + 1) for v in value)
    return False


def pending_step(observation: dict[str, Any]) -> bool:
    lines: list[str] = []
    _sentences([v for v in observation.values() if isinstance(v, (list, str))], lines)
    keys = list(observation)
    for value in observation.values():
        if isinstance(value, dict):
            keys += list(value)
    return any(_PENDING_KEYS.search(str(k)) for k in keys) and not any(len(line.split()) >= 8 for line in lines)


def _listed_ids(args: dict[str, Any]) -> list[str] | None:
    def id_like(key: Any, value: Any) -> bool:
        return isinstance(value, str) and bool(_ID_HINT.match(value)) and any(c.isdigit() for c in value) and bool(_IDENT_KEYS.search(str(key)))

    if any(id_like(k, v) for k, v in args.items()):
        return None             # the call names its listing directly; a list beside it holds choices, not lines
    lists = [v for v in args.values() if isinstance(v, list) and v and all(isinstance(x, dict) for x in v)]
    out = []
    for lines in lists:
        for line in lines:
            ident = next((v for k, v in line.items() if id_like(k, v)), None)
            if ident:
                out.append(ident)
    return out or None


_GONE = re.compile(r"sold out|out of stock|unavailable|no longer|zero left|none left|all gone|is gone|it's gone|now gone|"
                   r"gone from stock|dried up|ran dry|run dry|cleared out|cleared from|bought the last|can't be ordered|cannot be ordered|"
                   r"grayed out|greyed out|hit zero|no units|none remaining|no .{0,12}remaining|been sold|been taken|snapped up|"
                   r"in stock any ?more|not in stock|isn't in stock|no stock|stock.{0,20}\bzero\b|last one", re.I)
_PRICE_WORDS = re.compile(r"price|cost|charg|priced|bumped|gone up|went up|come down|dropped|cents|dearer|cheaper", re.I)
_LISTING_META = re.compile(r"\b(brand|maker|manufacturer|seller|vendor|merchant|headline|heading|sku|url|title|made by|sold by|"
                           r"blurb|tagline|caption|teaser|slogan)\b|"
                           r"^(make|shop|shop name|store|store name|outlet|company|category|department|shelf|description|"
                           r"(overview|description|summary)( text)?|notes?|image|images|rating|reviews?|review count|link|"
                           r"options?|variants?|id|name|price|amount|cost|stock|on hand|in stock)$", re.I)
_USED_UP = re.compile(r"budget_exceeded|quota|allowance|used up|no more\b.{0,40}\b(may|can|allowed|left)|limit reached|"
                      r"reached the limit|maximum (number|reached)|already used", re.I)
_IDENT_KEYS = re.compile(r"id|ref|sku|code|number|key|listing|product", re.I)
_OPTION_KEYS = re.compile(r"option|variant|choice|config|version|edition|sku|model", re.I)
_KIND_KEYS = re.compile(r"categor|department|^dept$|shelf|aisle|section|^kind$|product.?(type|family|group|kind)|"
                        r"item.?(type|kind)|offer.?kind|^family$|^group$", re.I)
_FIRST_PERSON = re.compile(r"\b(I|I'm|I'd|I've|I'll|my|me|mine)\b")
_ASKS = re.compile(r"\?|\b(confirm|let me know|tell me|acknowledge|reply|answer|get back to me)\b", re.I)
_TELL_RULE = re.compile(r"\b(tell|inform|let (me|them|the shopper) know|update|confirm|report|mention|explain|message)\b", re.I)
_FINISH_STEP = re.compile(r"final|confirm|complete|checkout|pay|place|submit|commit|token", re.I)
_PENDING_KEYS = re.compile(r"token|pending|await|confirm|next.?step|checkout.?id|session", re.I)
_DECLINE_WORDS = re.compile(r"refus|declin|abandon|give.?up|no.?(buy|purchase|order)|reject|cancel|quit|skip", re.I)
_ID_HINT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-:.]{3,63}$")


def _compact(value: Any) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        text = str(value)
    if len(text) > OBS_CHARS:
        text = text[:OBS_CHARS] + "...(cut)"
    return text


def parse_json_object(text: str) -> dict[str, Any] | None:
    if not isinstance(text, str):
        return None
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[A-Za-z]*", "", text).strip()
        if text.endswith("```"):
            text = text[:-3].strip()
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except ValueError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            value = json.loads(text[start : end + 1])
            return value if isinstance(value, dict) else None
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
        if isinstance(value, dict) and text.count("{", 0, index) == text.count("}", 0, index):
            found = value
        index = text.find("{", stop)
    return found


def type_ok(value: Any, kind: str) -> bool:
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


def coerce(value: Any, schema: dict[str, Any]) -> Any:
    kinds = schema.get("type")
    if isinstance(kinds, str):
        kinds = [kinds]
    if not kinds:
        return value
    if any(type_ok(value, k) for k in kinds):
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
            if kind == "boolean" and isinstance(value, str) and value.lower() in ("true", "false"):
                return value.lower() == "true"
        except (TypeError, ValueError):
            continue
    return value


def _check(value: Any, schema: dict[str, Any], path: str) -> tuple[Any, list[str]]:
    if not isinstance(schema, dict):
        return value, []
    problems: list[str] = []
    value = coerce(value, schema)
    kinds = schema.get("type")
    if isinstance(kinds, str):
        kinds = [kinds]
    if kinds and not any(type_ok(value, k) for k in kinds):
        return value, [f"{path}: expected {'/'.join(kinds)}"]
    if "enum" in schema and value not in schema["enum"]:
        if isinstance(value, str):
            for option in schema["enum"]:
                if isinstance(option, str) and option.lower() == value.strip().lower():
                    value = option
                    break
        if value not in schema["enum"]:
            problems.append(f"{path}: must be one of {schema['enum']}")
    if "const" in schema and value != schema["const"]:
        problems.append(f"{path}: must equal {schema['const']!r}")
    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            problems.append(f"{path}: shorter than {schema['minLength']} characters")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            value = value[: schema["maxLength"]]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            problems.append(f"{path}: below minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            problems.append(f"{path}: above maximum {schema['maximum']}")
        if "exclusiveMinimum" in schema and isinstance(schema["exclusiveMinimum"], (int, float)) and value <= schema["exclusiveMinimum"]:
            problems.append(f"{path}: must be above {schema['exclusiveMinimum']}")
        if "exclusiveMaximum" in schema and isinstance(schema["exclusiveMaximum"], (int, float)) and value >= schema["exclusiveMaximum"]:
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
            if name not in value:
                problems.append(f"{path}.{name}: required")
        if schema.get("additionalProperties") is False:
            for name in list(value):
                if name not in properties:
                    problems.append(f"{path}.{name}: not an allowed argument")
        fixed = {}
        for name, item in value.items():
            if name in properties:
                item, sub = _check(item, properties[name], f"{path}.{name}")
                problems += sub
            fixed[name] = item
        value = fixed
    return value, problems


def _sentences(value: Any, out: list[str], depth: int = 0) -> None:
    if depth > 4:
        return
    if isinstance(value, str):
        if len(value.split()) >= 4:
            out.append(value)
    elif isinstance(value, list):
        for item in value[:10]:
            _sentences(item, out, depth + 1)


def collect_ids(value: Any, out: set[str], depth: int = 0) -> None:
    if depth > 6:
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, str) and "id" in key.lower() and _ID_HINT.match(item):
                out.add(item)
            else:
                collect_ids(item, out, depth + 1)
    elif isinstance(value, list):
        for item in value:
            collect_ids(item, out, depth + 1)


class Episode:
    def __init__(self, problem_data: dict[str, Any]):
        self.started = time.monotonic()
        environment = problem_data["environment"]
        self.binding = dict(environment["binding"])
        self.policy = environment["policy_view"]
        self.tools = [t for t in self.policy.get("tools") or [] if isinstance(t, dict) and isinstance(t.get("function"), dict)]
        self.schemas = {t["function"]["name"]: t["function"].get("parameters") or {"type": "object", "properties": {}} for t in self.tools}
        self.max_steps = int(self.policy.get("max_steps") or 30)
        self.step = 0
        self.finished = bool(self.policy.get("finished"))
        self.dead = False
        self.messages_used = 0
        self.message_budget: int | None = None
        self.order_attempts = 0
        self.refusals = 0
        self.order_checked = False
        self.decline_checks = 0
        self.opened: set[str] = set()
        self.cheaper_checks = 0
        self.early_bounced = False
        self.refused_at = -1        # step of the last refused order
        self.searched_at = -1       # step of the last search
        self.research_checks = 0
        self.cart_used = False
        self.seen_ids: set[str] = set()
        self.stated: list[str] = []
        self.revealed: list[str] = []
        self.free: list[str] = []
        self.dropped: list[str] = []
        self.budget_now: Any = None
        self.heard: set[str] = set()
        self.cart_ids: list[str] = []
        self.records: list[Any] = []
        self.last_cart_view: Any = None
        self.cart_changed_at = 0
        self.cart_view_at = -1
        self.cart_state: dict[str, str] = {}
        self.add_signature: str | None = None
        self.asked: list[str] = []
        self.questions_sent: list[str] = []
        self.planner_on = True
        self.sets_at_last_ask = -1
        self.last_unclear = False
        self.plan_misses = 0
        self.capped: set[str] = set()
        self.cart_lines: dict[str, str] = {}
        self.plan_retry_at = 0
        self.audited = False
        self.qty_warned: set[str] = set()
        self.pending_reply = ""
        self.current: list[str] | None = None
        self.verdicts: dict[str, str] = {}
        self.ledger_version = 0
        self.cart_checked: set[str] = set()
        self.said: list[str] = []
        self.recent: list[str] = []
        self.roles: dict[str, str] = {}
        self.plan: dict[str, Any] = {}
        self.cost = 0.0
        self.trace: list[dict[str, Any]] = []
        self.messages: list[dict[str, Any]] = []

    def time_left(self) -> float:
        return TASK_SECONDS - (time.monotonic() - self.started)

    def steps_left(self) -> int:
        return self.max_steps - self.step

    def infer(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, models: list[str], max_tokens: int) -> dict[str, Any] | None:
        for model in models:
            left = self.time_left()
            if left < FORCE_SECONDS - 10:
                return None
            _llm.timeout = int(max(8, min(75, left - FORCE_SECONDS + 15)))
            sent = messages
            if model.startswith("anthropic/") and len(messages) > 2:
                sent = [dict(m) for m in messages]
                for index in (0, len(sent) - 1):
                    if isinstance(sent[index].get("content"), str) and sent[index]["content"]:
                        sent[index]["content"] = [{"type": "text", "text": sent[index]["content"], "cache_control": {"type": "ephemeral"}}]
            body: dict[str, Any] = {"model": model, "messages": sent, "temperature": 0, "max_tokens": max_tokens}
            if tools:
                body["tools"] = tools
                body["tool_choice"] = "auto"
            result = _llm.post_verbose("/inference/chat/completions", body)
            data = result.data
            if isinstance(data, dict) and data.get("choices"):
                usage = data.get("usage") or {}
                if isinstance(usage.get("cost"), (int, float)):
                    self.cost += float(usage["cost"])
                message = data["choices"][0].get("message") or {}
                if message.get("content") or message.get("tool_calls"):
                    print("### LLM RES: ", message)
                    return message
            self.trace.append({"kind": "inference_failure", "model": model, "error": str(result.error)[:300] if result.error else "empty"})
        return None

    def env_call(self, actions: list[dict[str, Any]]) -> dict[str, Any] | None:
        payload: dict[str, Any] = dict(self.binding)
        payload["step"] = self.step + 1
        if len(actions) == 1:
            payload["action"] = actions[0]
        else:
            payload["actions"] = actions
        for _ in range(3):
            left = self.time_left()
            if left < 2:
                return None
            _env.timeout = int(max(3, min(95, left - 1)))
            result = _env.post_verbose("/environment/call", payload)
            if isinstance(result.data, dict):
                data = result.data
                if isinstance(data.get("step"), int):
                    self.step = data["step"]
                else:
                    self.step += len(actions)
                if data.get("finished"):
                    self.finished = True
                print("\n\n### ENV RES:", data)
                return data
            error = result.error or {}
            status = error.get("status")
            body = str(error.get("body") or "")
            self.trace.append({"kind": "env_failure", "status": status, "body": body[:300]})
            if status == 402:
                self.dead = True
                return None
            if status == 409:
                if "finished" in body:
                    self.finished = True
                    return None
                match = re.search(r"expected (\d+)", body)
                if match:
                    self.step = int(match.group(1)) - 1
                    payload["step"] = self.step + 1
                    continue
                return None
            if status in (403, 422, 500, 503, 504):
                return None
        return None

    def setup(self) -> None:
        tool_lines = []
        for tool in self.tools:
            function = tool["function"]
            tool_lines.append(f"- {function['name']}: {function.get('description', '')} | arguments: {_compact(function.get('parameters') or {})}")
        static = SETUP_PROMPT + "\n\nTOOLS:\n" + "\n".join(tool_lines)
        variable = "REQUEST:\n" + str(self.policy.get("query") or "")
        if self.policy.get("opening") is not None:
            variable += "\n\nOPENING OBSERVATION:\n" + _compact(self.policy.get("opening"))
        content = [{"type": "text", "text": static, "cache_control": {"type": "ephemeral"}}, {"type": "text", "text": variable}]
        message = self.infer([{"role": "user", "content": content}], None, [SETUP_MODEL, BACKUP_MODEL], 1500)
        plan = parse_json_object((message or {}).get("content") or "") or {}
        if not isinstance(plan.get("roles"), dict) and self.time_left() > 200:
            message = self.infer([{"role": "user", "content": content}], None, [SETUP_MODEL, MAIN_MODEL], 2000)
            plan = parse_json_object((message or {}).get("content") or "") or plan
        self.plan = plan
        roles = plan.get("roles") if isinstance(plan.get("roles"), dict) else {}
        self.roles = {k: v for k, v in roles.items() if isinstance(v, str) and v in self.schemas}
        if not self.roles:
            self.roles = self.guess_roles()
            self.trace.append({"kind": "guessed_roles", "roles": self.roles})
        budget = plan.get("message_budget")
        if isinstance(budget, (int, float)) and not isinstance(budget, bool) and budget >= 0:
            self.message_budget = int(budget)
        self.stated = [str(r) for r in plan.get("requirements") or [] if r]
        self.budget_now = plan.get("price_cap")
        if plan and self.budget_now is None and self.roles.get("message"):
            ask = [str(a) for a in plan.get("ask_first") or [] if a]
            plan["ask_first"] = ["the most the shopper wants to spend, since the request states no price limit (skip if the same thing is listed again here)"] + ask
        steps = plan.get("step_budget")
        if isinstance(steps, (int, float)) and not isinstance(steps, bool) and 0 < steps < self.max_steps:
            self.max_steps = int(steps)

    def guess_roles(self) -> dict[str, str]:
        def words(name: str) -> set[str]:
            spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name)
            return {w.lower() for w in re.split(r"[^A-Za-z0-9]+", spaced) if w}

        def one_text_argument(name: str) -> bool:
            properties = (self.schemas.get(name) or {}).get("properties") or {}
            texts = [k for k, v in properties.items() if isinstance(v, dict) and v.get("type") == "string" and "enum" not in v]
            return len(texts) == 1

        cart_words = {"cart", "basket", "bag", "trolley", "stash", "hold", "line", "lines", "keep", "selection"}
        tests = [
            ("decline", lambda w, n: bool(w & {"decline", "refuse", "reject", "abandon", "dismiss", "quit", "giveup"})
             or any(x.startswith(("declin", "refus")) for x in w) or {"give", "up"} <= w or "nothing" in w),
            ("cart_view", lambda w, n: bool(w & (cart_words | {"items", "order"})) and bool(w & {"view", "show", "list", "review", "status", "current", "my", "see", "contents", "summary"})),
            ("cart", lambda w, n: bool(w & cart_words) or bool(w & {"add", "put", "set", "remove", "rewrite", "update"}) and bool(w & (cart_words | {"items", "order", "all"}))),
            ("order", lambda w, n: bool(w & {"checkout", "purchase", "order", "buy", "commit", "confirm", "place", "settle", "pay", "execute", "finalize", "submit"})),
            ("message", lambda w, n: bool(w & {"message", "shopper", "buyer", "customer", "client", "ask", "chat", "note", "contact", "write", "talk"}) and one_text_argument(n)),
            ("details", lambda w, n: bool(w & {"detail", "details", "inspect", "open", "read", "view", "fetch", "get", "datasheet", "info", "spec", "specs", "describe", "examine"})),
            ("search", lambda w, n: bool(w & {"search", "find", "browse", "locate", "hunt", "query", "lookup", "catalog", "seek", "list"})),
        ]
        guessed: dict[str, str] = {}
        taken: set[str] = set()
        for role, test in tests:
            for tool in self.tools:
                name = tool["function"]["name"]
                if name not in taken and test(words(name), name):
                    guessed[role] = name
                    taken.add(name)
                    break
        return guessed

    def items(self) -> list[tuple[str, int]]:
        out = []
        raw = self.plan.get("items")
        for item in raw if isinstance(raw, list) else []:
            if isinstance(item, str) and item.strip():
                item = {"product": item, "quantity": 1}
            if isinstance(item, dict) and str(item.get("product") or "").strip():
                quantity = item.get("quantity")
                if isinstance(quantity, str) and quantity.strip().isdigit():
                    quantity = int(quantity.strip())
                quantity = int(quantity) if isinstance(quantity, (int, float)) and not isinstance(quantity, bool) and quantity >= 1 else 1
                out.append((str(item["product"]).strip(), quantity))
        return out if len(out) >= 2 else []

    def brief(self) -> str:
        plan = self.plan
        if not plan:
            return ""
        lines = ["A first reading of the request (check it against the request itself):"]
        if plan.get("product"):
            lines.append(f"- product: {plan.get('product')}")
        items = self.items()
        if items:
            for name, quantity in items:
                lines.append(f"- item to buy: {name}, quantity {quantity} (one cart line per item)")
        else:
            lines.append(f"- quantity: {plan.get('quantity') or 1}")
        for requirement in plan.get("requirements") or []:
            lines.append(f"- stated requirement: {requirement}")
        if plan.get("price_cap") is not None:
            lines.append(f"- price cap (total): {plan.get('price_cap')} {plan.get('currency') or ''}".rstrip())
        if plan.get("in_stock_only"):
            lines.append("- only items in stock / on hand")
        if plan.get("priority") in ("cheapest", "best"):
            detail = f" ({plan.get('priority_detail')})" if plan.get("priority_detail") else ""
            lines.append(f"- choose the {plan.get('priority')}{detail} among all fitting items")
        else:
            lines.append("- no cheapest/best rule was found: any item that meets every requirement within the budget is right, so do not spend steps hunting for the lowest price (unless the request itself says otherwise)")
        for want in plan.get("soft_wants") or []:
            lines.append(f"- nice to have, NOT required (only a tie-breaker): {want}")
        for item in plan.get("ask_first") or []:
            lines.append(f"- must be learned from the shopper first: {item}")
        for rule in plan.get("rules") or []:
            lines.append(f"- rule: {rule}")
        if self.message_budget is not None:
            lines.append(f"- messages to the shopper allowed: {self.message_budget}")
        lines.append(f"- steps allowed: {self.max_steps}")
        hints = [h for h in plan.get("hint_properties") or [] if isinstance(h, str)]
        if hints:
            lines.append("- properties the story may hint at (ask about these first, one per message): " + ", ".join(hints))
        return "\n".join(lines)

    def status(self) -> str:
        left = self.steps_left()
        parts = [f"[status] steps used {self.step} of {self.max_steps}, {left} left"]
        if self.message_budget is not None:
            parts.append(f"messages sent {self.messages_used} of {self.message_budget}")
        else:
            parts.append(f"messages sent {self.messages_used}")
        parts.append(f"about {int(max(0, self.time_left() - FORCE_SECONDS))} s left")
        text = "; ".join(parts) + "."
        text += "\n[requirements now] " + self.ledger()
        if self.refusals and not self.finished and self.step - self.refused_at >= 2:
            text += (f" NOTHING HAS BEEN BOUGHT YET: the order attempt {self.step - self.refused_at} steps ago did not end the episode."
                     " If the cart holds fitting items within the current budget, call the order tool now; if not, fix the cart first, then order"
                     " (decline only if nothing in the store fits any more).")
        if left <= 2:
            text += " LAST STEPS: order now if the cart holds a verified fit, otherwise decline."
        elif left <= 5 or self.time_left() < WRAP_UP_SECONDS:
            text += " FINISH NOW: fix the cart to the verified line(s) the request needs, re-check stock, then order (or decline if nothing fits)."
        elif self.order_attempts == 0 and left <= 14:
            text += " Steps are running low: within the next few steps put your best verified fit in the cart and order; searching and reading details end when 7 steps remain, and a refused order needs about 4 steps to repair."
        return text

    def ledger(self) -> str:
        if self.current is not None:
            items = list(self.current)
        else:
            items = [f"{r} (stated)" for r in self.stated if r not in self.dropped]
            items += [f"{r} (revealed by the shopper)" for r in self.revealed if r not in self.dropped]
        text = "; ".join(items) if items else "see the request"
        if self.budget_now is not None:
            text += f"; total price at most {self.budget_now} {self.plan.get('currency') or ''}".rstrip()
        items = self.items()
        if items:
            text += "; items and quantities: " + ", ".join(f"{name} x{quantity}" for name, quantity in items)
        else:
            text += f"; quantity {self.plan.get('quantity') or 1}"
        if self.dropped:
            text += "; no longer required: " + "; ".join(self.dropped)
        if self.free:
            text += "; shopper does not care about: " + ", ".join(self.free)
        return text

    def judge(self, prompt: str, max_tokens: int = 700, models: list[str] | None = None) -> dict[str, Any] | None:
        models = models or [JUDGE_MODEL, MAIN_MODEL]
        if self.cost > COST_SOFT and models[0] == JUDGE_MODEL:
            models = [READ_MODEL, JUDGE_MODEL]
        message = self.infer([{"role": "user", "content": prompt}], None, models, max_tokens)
        return parse_json_object((message or {}).get("content") or "")

    def interpret(self, lines: list[str], question: str | None) -> str:
        fresh = [line for line in lines if line not in self.heard]
        if not fresh:
            return ""
        self.heard.update(fresh)
        if self.time_left() < WRAP_UP_SECONDS:
            return ""
        prompt = INTERPRET_PROMPT + "\n\nTHE SHOPPER'S REQUEST (for context):\n" + str(self.policy.get("query") or "")
        prompt += "\n\nREQUIREMENTS SO FAR:\n" + self.ledger()
        if self.said:
            prompt += "\n\nEARLIER EXCHANGES WITH THE SHOPPER (oldest first):\n" + "\n".join(self.said[-12:])
        if question:
            prompt += "\n\nTHE ASSISTANT ASKED:\n" + question
        prompt += "\n\nTHE SHOPPER NOW SAID:\n" + "\n".join(fresh)
        self.said.append((f"asked: {question} -> " if question else "") + "shopper: " + " / ".join(fresh))
        result = self.judge(prompt, 1500, [READ_MODEL, JUDGE_MODEL])
        if not result:
            return ""
        notes = []
        now = result.get("requirements_now")
        if isinstance(now, list) and all(isinstance(r, str) for r in now):
            self.current = [r for r in now if r.strip()]
        for need in result.get("needs") or []:
            if isinstance(need, str) and need and need not in self.revealed:
                self.revealed.append(need)
                notes.append(f"new requirement: {need}")
        for prop in result.get("no_preference") or []:
            if isinstance(prop, str) and prop and prop not in self.free:
                self.free.append(prop)
        for drop in result.get("dropped") or []:
            if isinstance(drop, str) and drop and drop not in self.dropped:
                self.dropped.append(drop)
                notes.append(f"dropped: {drop}")
        budget = result.get("new_budget")
        if isinstance(budget, (int, float)) and not isinstance(budget, bool) and budget > 0 and budget != self.budget_now:
            self.budget_now = budget
            notes.append(f"new budget: {budget}")
        change = result.get("new_quantity")
        if isinstance(change, int) and not isinstance(change, bool):
            change = {"product": "", "quantity": change}
        if isinstance(change, dict) and isinstance(change.get("quantity"), int) and not isinstance(change.get("quantity"), bool) and change["quantity"] >= 1:
            wanted = change["quantity"]
            product = str(change.get("product") or "").strip().lower()
            if not self.items():
                # one kind of product: the plan's quantity is the one that counts
                self.plan["quantity"] = wanted
                notes.append(f"new quantity: {wanted}")
            else:
                items = self.plan.get("items") if isinstance(self.plan.get("items"), list) else []
                target = next((i for i in items if isinstance(i, dict) and product and (product in str(i.get("product") or "").lower() or str(i.get("product") or "").lower() in product)), None)
                if target is not None:
                    target["quantity"] = wanted
                    notes.append(f"new quantity: {target.get('product')} x{wanted}")
        if result.get("unavailable"):
            notes.append("an item became unavailable; remove it from the cart and do not order it")
        if notes:
            self.ledger_version += 1
        reply = str(result.get("reply") or "").strip()
        unclear = result.get("unclear") is True and question is not None
        self.last_unclear = unclear
        named = True
        if unclear and self.items():
            asked_words = {w.rstrip("s") for w in str(question).lower().replace("?", " ").split()}
            named = any(w.rstrip("s") in asked_words for product, _ in self.items() for w in product.lower().split()
                        if len(w) >= 3 and w not in ("and", "the", "for", "set", "with", "pair", "pack", "of"))
        if unclear and not named:
            # with several products, a question that does not say which one is ambiguous: the property may still matter
            notes.append("the shopper did not follow that question because it did not say which product it is about: if messages remain, ask about that property once more, naming the product")
        elif unclear:
            # explaining our own question again rarely helps; the asked property is most likely not a need
            notes.append("the shopper did not follow that question: do not explain or repeat it; the property asked about is most likely free, so ask about a different one (naming the product) if messages remain")
        elif reply and self.roles.get("message") and _ASKS.search(" ".join(fresh)):
            # only a line that asks something gets an answer; a stated need needs no confirmation message
            self.pending_reply = reply
        meaning = str(result.get("meaning") or "")
        return "Meaning: " + meaning + (" (" + "; ".join(notes) + ")" if notes else "")

    def item_records(self, item: str) -> list[str]:
        found: list[str] = []

        def walk(value: Any, depth: int, listing: Any = None, via_option: bool = False) -> None:
            if depth > 6 or len(found) >= 8:
                return
            if isinstance(value, dict):
                if any(isinstance(v, str) and v == item for v in value.values()):
                    # an option inside a listing is shown with the listing that holds it, whose facts it shares
                    holder = listing if listing is not None and via_option and listing is not value else value
                    text = _compact(_brief(holder, 260))[:3000]
                    if text not in found:
                        found.append(text)
                    return
                if any(isinstance(v, str) and _IDENT_KEYS.search(str(k)) and _ID_HINT.match(v) and any(c.isdigit() for c in v)
                       for k, v in value.items()):
                    listing, via_option = value, False
                for k, v in value.items():
                    walk(v, depth + 1, listing, via_option or bool(_OPTION_KEYS.search(str(k))))
            elif isinstance(value, list):
                for v in value:
                    walk(v, depth + 1, listing, via_option)

        for record in self.records:
            walk(record, 0)
        return found[-3:]

    def verify_cart(self) -> str:
        if not self.cart_ids:
            return ""
        # a request for several different products puts one line per product in the cart: check each line
        for item in (self.cart_ids[-3:] if self.items() else self.cart_ids[-1:]):
            verdict = self.verify_item(item, "ORDER", self.cart_lines.get(item, ""))
            if verdict:
                return verdict
        return ""

    def verify_item(self, item: str, purpose: str, line: str = "") -> str:
        if self.time_left() < WRAP_UP_SECONDS + 20:
            return ""
        key = f"{item}|{line}|{self.ledger_version}"
        if key in self.verdicts:
            return self.verdicts[key]
        records = self.item_records(item)
        if not records:
            return ""
        prompt = VERIFY_PROMPT + "\n\nTHE SHOPPER'S REQUEST:\n" + str(self.policy.get("query") or "")
        prompt += "\n\nCURRENT REQUIREMENTS (stated, revealed and changed):\n" + self.ledger()
        prompt += f"\n\nTHE ITEM TO {purpose}: {item}\nRECORDS OF THIS ITEM SEEN IN THE STORE (oldest first):\n" + "\n".join(records)
        if line:
            prompt += "\n\nTHE CART LINE FOR IT (when it names an option or variant, judge that option: its own values, price and stock count): " + line
        if self.last_cart_view is not None:
            prompt += "\n\nLATEST CART VIEW:\n" + _compact(self.last_cart_view)[:2000]
        labels = self.row_labels()
        if labels:
            prompt += "\n\nLABELS THAT RECUR ON THIS STORE'S LISTINGS (with counts; the item's category label must be the one that matches the request's wording as a whole, not a neighbouring one):\n" + labels
        result = self.judge(prompt, 1200)
        self.trace.append({"kind": "verify", "item": item, "result": result})
        if not result or result.get("fits") is not False:
            self.verdicts[key] = ""
            return ""
        problems = [str(p) for p in result.get("problems") or []][:5]
        verdict = f"An independent check says {item} does NOT meet every requirement: " + "; ".join(problems) + ". Unless you are certain the check is wrong, choose another fully verified fit."
        self.verdicts[key] = verdict
        return verdict

    def tools_for_turn(self) -> list[dict[str, Any]]:
        left = self.steps_left()
        order, decline, cart = self.roles.get("order"), self.roles.get("decline"), self.roles.get("cart")
        names: set[str] | None = None
        if left <= 1:
            names = {order, decline}
        elif left <= 3:
            names = {order, decline, cart}
        elif left <= 7 and (self.order_attempts == 0 or self.refusals > 0):
            names = {order, decline, cart, self.roles.get("cart_view"), self.roles.get("view")}
        tools = self.tools
        if names is not None and order and cart:
            # a tool no role covers that finishes a checkout (a confirm step after the order tool) stays available
            assigned = set(self.roles.values())
            names |= {t["function"]["name"] for t in self.tools if t["function"]["name"] not in assigned
                      and _FINISH_STEP.search(t["function"]["name"])}
            narrowed = [t for t in self.tools if t["function"]["name"] in names]
            if narrowed:
                tools = narrowed
        if self.planner_on and self.roles.get("message"):
            tools = [t for t in tools if t["function"]["name"] != self.roles.get("message")]
        if self.capped:
            kept = [t for t in tools if t["function"]["name"] not in self.capped]
            tools = kept or tools
        return tools

    def property_menu(self) -> tuple[str, int]:
        counts: dict[str, dict[str, int]] = {}
        distinct: dict[str, dict[str, set[str]]] = {}
        samples: dict[str, dict[str, list[str]]] = {}
        tables: dict[str, int] = {}
        fact_sets = 0

        def short(value: Any) -> bool:
            return isinstance(value, str) and 0 < len(value) <= 70

        def walk(value: Any, depth: int, key: str = "", group: str = "") -> None:
            nonlocal fact_sets
            if depth > 6:
                return
            if isinstance(value, dict):
                # a listing's kind label (category, department, shelf...) groups its properties, so that a request for
                # several different products gets one menu per kind
                label = next((v.strip().lower() for k, v in value.items() if short(v) and _KIND_KEYS.search(str(k))), "")
                group = label or group
                strings = [(k, v) for k, v in value.items() if short(v)]
                # a property table: two short values under a key that names properties, or five anywhere when the
                # record holds no such table (then its own short values are the properties)
                nested = any(isinstance(v, dict) and _PROPERTY_KEYS.search(str(k)) for k, v in value.items())
                if (len(strings) >= 2 and (_PROPERTY_KEYS.search(key) or _OPTION_KEYS.search(key))) or (len(strings) >= 5 and not nested):
                    added = 0
                    for prop_key, item in strings:
                        if re.search(r"\d", item) and re.fullmatch(r"[A-Za-z0-9_\-]+", item) and ("-" in item or "_" in item):
                            continue
                        name = re.sub(r"[_.]+", " ", str(prop_key)).strip().lower()
                        if _LISTING_META.search(name) or _KIND_KEYS.search(name):
                            continue          # who sells it, what it is called, where it sits: not a property to ask about
                        counts.setdefault(group, {})[name] = counts.setdefault(group, {}).get(name, 0) + 1
                        distinct.setdefault(group, {}).setdefault(name, set()).add(item.lower())
                        bucket = samples.setdefault(group, {}).setdefault(name, [])
                        if item.lower() not in bucket and len(bucket) < 5:
                            bucket.append(item.lower())
                        added += 1
                    # a row of only ids, titles and seller names is not a set of facts, and the options of one listing
                    # are not several listings
                    if added >= 2 and not _OPTION_KEYS.search(key):
                        fact_sets += 1
                        tables[group] = tables.get(group, 0) + 1
                for name, item in value.items():
                    if isinstance(item, (dict, list)):
                        walk(item, depth + 1, str(name), group)
            elif isinstance(value, list):
                for item in value:
                    walk(item, depth + 1, key, group)

        for record in self.records:
            walk(record, 0)

        def lines_for(group: str, limit: int) -> list[str]:
            c, d, s = counts.get(group, {}), distinct.get(group, {}), samples.get(group, {})
            # properties whose values differ between the listings come first: only they can change the choice
            ranked = sorted(c, key=lambda k: (len(d.get(k, ())) < 2, -c[k]))[:limit]
            return [f"- {k} (in {c[k]} listings, {len(d.get(k, ()))} different values): " + " | ".join(s.get(k, [])) for k in ranked]

        groups = [g for g in sorted(counts, key=lambda g: -tables.get(g, 0)) if g and tables.get(g, 0) >= 2]
        if len(groups) >= 2 and self.items():
            out = []
            for group in groups[:4]:
                out.append(f"Listings labelled \"{group}\" ({tables[group]} with details):")
                out.extend(lines_for(group, 14))
            if counts.get(""):
                out.append("Listings without a label:")
                out.extend(lines_for("", 10))
            return "\n".join(out), fact_sets
        merged_counts: dict[str, int] = {}
        for group in counts:
            for name, n in counts[group].items():
                merged_counts[name] = merged_counts.get(name, 0) + n
                distinct.setdefault("", {}).setdefault(name, set()).update(distinct[group][name])
                bucket = samples.setdefault("", {}).setdefault(name, [])
                for v in samples[group][name]:
                    if v not in bucket and len(bucket) < 5:
                        bucket.append(v)
        counts[""] = merged_counts
        return "\n".join(lines_for("", 28)), fact_sets

    def row_labels(self) -> str:
        counts: dict[str, dict[str, int]] = {}

        def walk(value: Any, depth: int) -> None:
            if depth > 6:
                return
            if isinstance(value, dict):
                if len(value) >= 4:
                    for key, item in value.items():
                        if isinstance(item, str) and 0 < len(item) <= 40 and not (re.search(r"\d", item) and re.fullmatch(r"[A-Za-z0-9_\-]+", item)):
                            bucket = counts.setdefault(str(key), {})
                            bucket[item] = bucket.get(item, 0) + 1
                for item in value.values():
                    if isinstance(item, (dict, list)):
                        walk(item, depth + 1)
            elif isinstance(value, list):
                for item in value:
                    walk(item, depth + 1)

        for record in self.records:
            walk(record, 0)
        lines = []
        for key, bucket in counts.items():
            total = sum(bucket.values())
            if total >= 4 and 2 <= len(bucket) <= 8 and max(bucket.values()) >= 2:
                lines.append(f"- {key}: " + " | ".join(f"{v} (x{n})" for v, n in sorted(bucket.items(), key=lambda kv: -kv[1])))
        return "\n".join(lines[:12])

    def message_args(self, text: str) -> dict[str, Any] | None:
        name = self.roles.get("message")
        schema = self.schemas.get(name or "") or {}
        properties = schema.get("properties") or {}
        required = list(schema.get("required") or [])
        strings = [k for k, v in properties.items() if isinstance(v, dict) and (v.get("type") == "string" or "string" in (v.get("type") or [])) and "enum" not in v]
        target = next((k for k in required if k in strings), None) or (strings[0] if len(strings) == 1 else None)
        if not target:
            return None
        if any(k != target for k in required):
            return None
        return {target: text}

    def ask_planned(self, force: bool = False) -> bool:
        tool = self.roles.get("message")
        limit = self.message_budget if self.message_budget is not None else 3
        if not tool or self.messages_used >= limit:
            self.planner_on = False
            return False
        menu, fact_sets = self.property_menu()
        urgent = bool(self.plan.get("ask_first")) and not self.asked
        if fact_sets < 4 and self.step < 8 and not urgent and not force:
            return False
        if force:
            urgent = False
        if self.asked and not force and not urgent and not self.last_unclear and fact_sets < self.sets_at_last_ask + 2:
            return False
        if not force and fact_sets < self.plan_retry_at:
            return False
        if self.steps_left() <= (6 if force else 10) or self.time_left() < WRAP_UP_SECONDS + 40:
            self.planner_on = False
            return False
        # one product: plan every remaining question at once, so the questions do not eat into the steps needed for
        # the search. Several products: one at a time, so each is chosen knowing which products were already asked
        # about and what more of the store shows
        count = 1 if self.items() else max(1, limit - self.messages_used)
        prompt = PLAN_PROMPT.replace("QUESTION_COUNT", str(count)) + "\n\nTHE SHOPPER'S REQUEST:\n" + str(self.policy.get("query") or "")
        if SWEEP_LAST and limit >= 2 and self.messages_used + count == limit and not urgent:
            # the last message names a few more properties at once: the shopper answers about one of them when one is a
            # need, and no later question is lost if the answer is "don't mind"
            prompt += SWEEP_NOTE
        prompt += "\n\nREQUIREMENTS SO FAR:\n" + self.ledger()
        items = self.items()
        if items:
            prompt += "\n\nTHE REQUEST ASKS FOR SEVERAL DIFFERENT PRODUCTS (name the one each question is about): " + "; ".join(name for name, _ in items)
        if self.asked:
            prompt += "\n\nALREADY ASKED (with what the shopper said):\n" + "\n".join(self.asked)
        if self.plan.get("ask_first"):
            prompt += "\n\nMUST BE LEARNED FROM THE SHOPPER: " + "; ".join(str(h) for h in self.plan.get("ask_first") or [])
        prompt += "\n\nPROPERTIES IN THE LISTINGS' DETAILS SEEN SO FAR (name, number of listings, example values):\n" + (menu or "(none seen yet)")
        result = self.judge(prompt, 2000, [READ_MODEL, JUDGE_MODEL])
        planned = [q for q in (result or {}).get("questions") or [] if isinstance(q, dict) and str(q.get("question") or "").strip()]
        if urgent and fact_sets < 4:
            planned = planned[:1]
        # never send a question that has been sent before in the same words
        planned = [q for q in planned if str(q.get("question")).strip().lower() not in self.questions_sent]
        planned = planned[:count]
        calls = []
        labels = []
        for index, item in enumerate(planned):
            args = self.message_args(str(item.get("question")).strip())
            if not args:
                continue
            args, problems = _check(args, self.schemas[tool], "args")
            if problems:
                continue
            calls.append({"id": f"ask_{self.messages_used + index + 1}_{self.step}", "type": "function", "function": {"name": tool, "arguments": json.dumps(args)}})
            product = str(item.get("product") or "").strip()
            labels.append(((product + " " if product and items else "") + str(item.get("property")), str(item.get("question")).strip()))
        if not calls:
            # a planning call that gave nothing usable is retried once, after another listing has been opened
            self.plan_misses += 1
            self.plan_retry_at = fact_sets + 1
            if self.plan_misses >= 2:
                self.planner_on = False
            return False
        self.messages.append({"role": "assistant", "content": "I will ask the shopper about " + ", then ".join(l[0] for l in labels) + ".", "tool_calls": calls})
        self.last_unclear = False      # used up by this round; the new answers set it again
        replies = self.run_calls(calls, planned=True)
        for index, (call_id, text) in enumerate(replies):
            if index == len(replies) - 1:
                text += "\n" + self.status()
            self.messages.append({"role": "tool", "tool_call_id": call_id, "content": text})
            if index < len(labels):
                self.asked.append(f"- {labels[index][0]}: asked \"{labels[index][1]}\" -> {text.split(chr(10) + '[status]')[0][:400]}")
        self.sets_at_last_ask = fact_sets
        self.planner_on = self.messages_used < limit
        return True

    def send_reply(self, text: str) -> bool:
        tool = self.roles.get("message")
        limit = self.message_budget if self.message_budget is not None else 3
        if not tool or self.messages_used >= limit or self.steps_left() <= 4 or self.time_left() < WRAP_UP_SECONDS:
            return False
        args = self.message_args(text)
        if not args:
            return False
        args, problems = _check(args, self.schemas[tool], "args")
        if problems:
            return False
        call = {"id": f"reply_{self.step}", "type": "function", "function": {"name": tool, "arguments": json.dumps(args)}}
        self.messages.append({"role": "assistant", "content": "I will answer the shopper's question.", "tool_calls": [call]})
        replies = self.run_calls([call], planned=True)
        for index, (call_id, reply_text) in enumerate(replies):
            if index == len(replies) - 1:
                reply_text += "\n" + self.status()
            self.messages.append({"role": "tool", "tool_call_id": call_id, "content": reply_text})
        return True

    def quantity_warning(self, name: str, args: dict[str, Any]) -> str:
        wanted = self.plan.get("quantity")
        if not isinstance(wanted, int) or isinstance(wanted, bool) or wanted < 1:
            return ""
        properties = self.schemas[name].get("properties") or {}
        signature = _compact({k: v for k, v in args.items() if isinstance(properties.get(k), dict) and "enum" in properties[k]})
        if self.add_signature is not None and signature != self.add_signature:
            return ""
        for key, spec in properties.items():
            if isinstance(spec, dict) and spec.get("type") == "integer" and "enum" not in spec:
                value = args.get(key, spec.get("default", 1))
                if self.items():
                    allowed = {q for _, q in self.items()}
                    if isinstance(value, int) and value not in allowed:
                        return f"the request needs these quantities: {', '.join(f'{n} x{q}' for n, q in self.items())}, but this call asks for {value}."
                    return ""
                if isinstance(value, int) and value != wanted:
                    return f"the request needs a quantity of {wanted} of one listing, but this call asks for {value}."
        return ""

    def unchecked_cheaper(self) -> list[tuple[str, Any]]:
        if self.plan.get("priority") != "cheapest" or not self.cart_ids:
            return []
        rows: dict[str, dict[str, Any]] = {}

        def walk(value: Any, depth: int) -> None:
            if depth > 6:
                return
            if isinstance(value, dict):
                ident = next((v for k, v in value.items() if isinstance(v, str) and "id" in k.lower() and _ID_HINT.match(v)), None)
                price = value.get("price")
                if ident and isinstance(price, (int, float)) and not isinstance(price, bool):
                    rows[ident] = {**rows.get(ident, {}), **{k: v for k, v in value.items() if v is not None}}
                for v in value.values():
                    if isinstance(v, (dict, list)):
                        walk(v, depth + 1)
            elif isinstance(value, list):
                for v in value:
                    walk(v, depth + 1)

        for record in self.records:
            walk(record, 0)
        held = rows.get(self.cart_ids[-1])
        if not held:
            return []
        quantity = self.plan.get("quantity") if isinstance(self.plan.get("quantity"), int) and self.plan.get("quantity") > 0 else 1
        budget = self.budget_now if isinstance(self.budget_now, (int, float)) and not isinstance(self.budget_now, bool) else None
        found = []
        for ident, row in rows.items():
            if ident in self.opened or ident in self.cart_ids or row["price"] >= held["price"]:
                continue
            if row.get("in_stock") is False or (budget is not None and row["price"] * quantity > budget):
                continue
            if held.get("category") and row.get("category") and row.get("category") != held.get("category"):
                continue
            found.append((row["price"], ident))
        return [(ident, price) for price, ident in sorted(found)[:5]]

    def is_decline(self, name: str, args: dict[str, Any]) -> bool:
        if name != self.roles.get("decline"):
            return False
        if self.roles.get("order") != name:
            return True
        properties = (self.schemas.get(name) or {}).get("properties") or {}
        return any(isinstance(v, str) and "enum" in (properties.get(k) or {}) and _DECLINE_WORDS.search(v) for k, v in args.items())

    def audit_decline(self) -> str:
        if self.time_left() < WRAP_UP_SECONDS + 30:
            return ""
        self.audited = True
        blocks: list[str] = []
        total = 0
        for record in reversed(self.records):
            text = _compact(_brief(record, 160))
            if total + len(text) > 16000:
                continue
            blocks.append(text)
            total += len(text)
        if not blocks:
            return ""
        prompt = AUDIT_PROMPT + "\n\nTHE SHOPPER'S REQUEST:\n" + str(self.policy.get("query") or "")
        prompt += "\n\nCURRENT REQUIREMENTS:\n" + self.ledger()
        prompt += "\n\nLISTING RECORDS SEEN (newest first):\n" + "\n".join(blocks)
        result = self.judge(prompt, 1200)
        self.trace.append({"kind": "audit", "result": result})
        fits = [str(x) for x in (result or {}).get("possible_fits") or [] if isinstance(x, (str, int))][:5]
        if not fits:
            return ""
        text = "An independent check thinks these listings may meet every requirement: " + ", ".join(fits) + (f" ({result.get('reason')})" if result.get("reason") else "") + ". Verify them (details, stock, price) and buy the right one; decline only if each one truly fails."
        if self.order_attempts:
            text += " Nothing has been bought yet: an order that goes through ends the episode, and this one is still running, so declining now would leave the shopper with nothing."
        return text

    def neutral_args(self, name: str, text: str, declining: bool = False) -> dict[str, Any]:
        schema = self.schemas.get(name) or {}
        properties = schema.get("properties") or {}
        args: dict[str, Any] = {}
        for key in schema.get("required") or []:
            spec = properties.get(key) or {}
            if "enum" in spec and spec["enum"]:
                # a finish tool shared by both finishes: pick the mode that matches the intent
                fitting = [v for v in spec["enum"] if isinstance(v, str) and bool(_DECLINE_WORDS.search(v)) == declining]
                args[key] = (fitting or spec["enum"])[0]
            elif spec.get("type") == "string" or "string" in (spec.get("type") or []):
                args[key] = text[: spec.get("maxLength", len(text))]
            elif spec.get("type") == "boolean":
                args[key] = True
            elif spec.get("type") in ("integer", "number"):
                args[key] = max(1, spec.get("minimum", 1))
        for key, spec in properties.items():
            if key not in args and isinstance(spec, dict) and spec.get("type") == "string" and "enum" not in spec and len(properties) == 1:
                args[key] = text[: spec.get("maxLength", len(text))]
        return args

    def clean_decline(self, args: dict[str, Any], name: str) -> dict[str, Any]:
        if self.plan.get("decline_no_names") is False:
            return args
        dirty = False
        for key, value in args.items():
            if isinstance(value, str):
                for token in re.split(r"[\s.,;:!?()\[\]{}<>\"'`]+", value):
                    if token and token in self.seen_ids:
                        dirty = True
        if self.plan.get("decline_no_names") or dirty:
            neutral = "Sorry, nothing in the store meets all of your requirements, so I am not ordering anything."
            cleaned = {}
            properties = (self.schemas.get(name) or {}).get("properties") or {}
            for key, value in args.items():
                spec = properties.get(key) or {}
                if isinstance(value, str) and "enum" not in spec:
                    cleaned[key] = neutral[: spec.get("maxLength", len(neutral))]
                else:
                    cleaned[key] = value
            return cleaned
        return args

    def force_finish(self) -> None:
        if self.finished or self.dead or self.steps_left() <= 0 or self.time_left() < 2:
            return
        order = self.roles.get("order")
        decline = self.roles.get("decline")
        if order and self.cart_used:
            action = {"name": order, "args": self.neutral_args(order, "Order the cart.")}
        elif decline:
            action = {"name": decline, "args": self.neutral_args(decline, "Sorry, nothing in the store meets all of your requirements, so I am not ordering anything.", declining=True)}
        else:
            return
        reply = self.env_call([action])
        self.trace.append({"kind": "forced_finish", "action": action, "reply": reply})
        refused = (reply or {}).get("observation") if isinstance((reply or {}).get("observation"), dict) else {}
        lines: list[str] = []
        parts = [v for v in refused.values() if isinstance(v, (list, str))]
        parts += [x for v in refused.values() if isinstance(v, dict) for x in v.values() if isinstance(x, (list, str))]
        _sentences(parts, lines)
        if (reply and not reply.get("finished") and lines and order and action["name"] == order and self.cart_used and _PRICE_WORDS.search(" ".join(lines)) and not _GONE.search(" ".join(lines)) and self.steps_left() > 0):
            reply = self.env_call([action])
            self.trace.append({"kind": "forced_finish", "action": action, "reply": reply})
        if reply and not reply.get("finished") and isinstance(reply.get("observation"), dict) and lines and decline and self.steps_left() > 0:
            action = {"name": decline, "args": self.neutral_args(decline, "Sorry, nothing in the store meets all of your requirements, so I am not ordering anything.", declining=True)}
            reply = self.env_call([action])
            self.trace.append({"kind": "forced_finish", "action": action, "reply": reply})

    def note_observation(self, name: str, observation: Any) -> str:
        notes = []
        if isinstance(observation, dict):
            collect_ids(observation, self.seen_ids)
            problem = " ".join(str(observation.get(k) or "") for k in ("error", "detail", "message", "reason") if isinstance(observation.get(k), str))
            core = {self.roles.get(r) for r in ("order", "decline", "cart", "cart_view")} - {None}
            if observation.get("error") and _USED_UP.search(problem) and name not in core:
                if name == self.roles.get("message"):
                    self.message_budget = self.messages_used if self.message_budget is None else min(self.message_budget, self.messages_used)
                self.capped.add(name)
                notes.append("That tool's allowance is used up; it is no longer offered. Do not try it again: read prices and stock from search results or the cart view instead, and decide with what you have.")
            if "refusal" in observation:
                self.refusals += 1
                self.refused_at = self.step
                self.order_checked = False
                notes.append("THE ORDER WAS REFUSED and the episode continues; the cart is unchanged. Apply every change in the refusal, fix the cart (remove any line that is no longer acceptable, such as a sold-out item), re-check stock and price of the new pick, and order again.")
                refusal = observation.get("refusal")
                said = refusal if isinstance(refusal, str) else _compact(refusal)
                if _PRICE_WORDS.search(said) and not _GONE.search(said):
                    notes.append("This refusal does not say the item is gone: when it only reports a changed price (or another change the item still meets), the item is still available. Re-read its current price; if price times quantity is still within the budget and it still meets every requirement, order it again right away. Never decline because a price changed.")
            elif len(observation) >= 2 and name != self.roles.get("message"):
                lines: list[str] = []
                for key, value in observation.items():
                    if isinstance(value, list) and value and all(isinstance(v, str) for v in value):
                        _sentences(value, lines)
                if lines:
                    notes.append("This result carries a message from the shopper (a change). Apply it: " + " | ".join(lines[:4]))
        elif isinstance(observation, list):
            collect_ids(observation, self.seen_ids)
        return " ".join(notes)

    def run_calls(self, calls: list[dict[str, Any]], planned: bool = False) -> list[tuple[str, str]]:
        replies: list[tuple[str, str]] = []
        allowed = {t["function"]["name"] for t in self.tools_for_turn()}
        runnable: list[tuple[str, str, dict[str, Any]]] = []
        finishing = {self.roles.get("order"), self.roles.get("decline")} - {None}
        message_tool = self.roles.get("message")
        cart_tool = self.roles.get("cart")
        planned_messages = 0
        for call in calls:
            call_id = call["id"]
            function = call.get("function") or {}
            name = function.get("name")
            raw = function.get("arguments")
            if name not in self.schemas:
                replies.append((call_id, f"Not run (no step used): '{name}' is not one of the tools. Use one of: {', '.join(sorted(allowed))}."))
                continue
            if not planned and name not in allowed:
                if name == message_tool and self.planner_on:
                    reason = "questions to the shopper are planned and sent for you after you have opened a few listings' details"
                else:
                    reason = "too few steps are left for this; only these tools can be used now: " + ", ".join(sorted(allowed))
                replies.append((call_id, f"Not run (no step used): {reason}."))
                continue
            if isinstance(raw, dict):
                args = raw
            else:
                args = parse_json_object(raw or "{}") if (raw or "").strip() else {}
                if args is None:
                    replies.append((call_id, "Not run (no step used): the arguments were not a valid JSON object."))
                    continue
            args, problems = _check(args, self.schemas[name], "args")
            if problems:
                replies.append((call_id, "Not run (no step used): invalid arguments: " + "; ".join(problems[:6])))
                continue
            if name in finishing and len(calls) > 1:
                replies.append((call_id, "Not run (no step used): an order or a decline must be sent alone, after you have seen the results of your other calls."))
                continue
            if (name == message_tool and not planned and "?" not in " ".join(str(v) for v in args.values() if isinstance(v, str))
                    and not any(_TELL_RULE.search(str(rule)) for rule in self.plan.get("rules") or [])):
                replies.append((call_id, "Not run (no step used): messages to the shopper are only for questions, and this one asks nothing. Do not report findings, choices or plans to the shopper; the finishing tool carries that."))
                continue
            if name == message_tool and " ".join(str(v) for v in args.values() if isinstance(v, str)).strip().lower() in self.questions_sent:
                replies.append((call_id, "Not run (no step used): this exact message was already sent, and its answer is above. Ask about something else, or carry on."))
                continue
            if name == message_tool and self.message_budget is not None and self.messages_used + planned_messages >= self.message_budget:
                replies.append((call_id, "Not run (no step used): the message budget is used up. Do not message the shopper again."))
                continue
            signature = name + _compact(args)
            if name not in finishing and name != message_tool and name != cart_tool and signature in self.recent:
                replies.append((call_id, "Not run (no step used): you just made this exact call and nothing has changed since; its result is above. Change the arguments or do something else."))
                continue
            if name == cart_tool and self.steps_left() > 3 and not self.items():      # several products: each line is checked once, at the order
                properties = self.schemas[name].get("properties") or {}
                signature_now = _compact({k: v for k, v in args.items() if isinstance(properties.get(k), dict) and "enum" in properties[k]})
                if self.add_signature is None or signature_now == self.add_signature:
                    target = next((v for k, v in args.items() if isinstance(v, str) and "enum" not in (properties.get(k) or {}) and v.strip() and " " not in v.strip()), None)
                    if target is None and _listed_ids(args):
                        target = _listed_ids(args)[-1]
                    if target and f"{target}|{self.ledger_version}" not in self.cart_checked:
                        self.cart_checked.add(f"{target}|{self.ledger_version}")
                        verdict = self.verify_item(target, "ADD TO THE CART", line_of(args, target))
                        if verdict:
                            replies.append((call_id, "Not run (no step used). " + verdict + " If you are certain it fits, send the same call again."))
                            continue
            if name == cart_tool:
                warning = self.quantity_warning(name, args)
                if warning and signature not in self.qty_warned:
                    self.qty_warned.add(signature)
                    replies.append((call_id, "Not run (no step used): " + warning + " If you are sure, send the same call again."))
                    continue
            if name == message_tool:
                planned_messages += 1
            runnable.append((call_id, name, args))
        if runnable and runnable[0][1] in finishing:
            call_id, name, args = runnable[0]
            declining = self.is_decline(name, args)
            # After a refused order a backup picked earlier may no longer be the cheapest fit: search once more first.
            if (not declining and name == self.roles.get("order") and self.plan.get("priority") == "cheapest" and self.refused_at > self.searched_at and self.research_checks < 1 and self.steps_left() > 10 and self.time_left() > WRAP_UP_SECONDS + 20):
                self.research_checks += 1
                replies.append((call_id, "Not run yet (no step used). The last order was refused, and the request asks for the cheapest fitting item. A backup chosen earlier is right only if nothing cheaper fits now, so search again before ordering: the stated requirement words and the category, with a price limit just above the refused item's price if the search offers one, and the next pages. Open the details of cheaper candidates, put the cheapest one that meets every requirement in the cart, then order."))
                return replies
            if not declining and name == self.roles.get("order") and self.cheaper_checks < 2 and self.steps_left() > 10 and self.time_left() > WRAP_UP_SECONDS + 20:
                cheaper = self.unchecked_cheaper()
                if cheaper:
                    self.cheaper_checks += 1
                    replies.append((call_id, "Not run yet (no step used). The request asks for the cheapest fitting item, but these cheaper listings that you have seen in results were never opened with the details tool: " + ", ".join(f"{i} ({p})" for i, p in cheaper) + ". Open their details (several in one reply), and if one meets every requirement, put the cheapest such one in the cart instead; if none does, order again."))
                    return replies
            if not declining and name == self.roles.get("order") and not self.order_checked and self.steps_left() > 2 and self.time_left() > WRAP_UP_SECONDS:
                self.order_checked = True
                verdict = self.verify_cart()
                if verdict:
                    replies.append((call_id, "Not run yet (no step used). " + verdict))
                    return replies
                replies.append((call_id, "Not run yet (no step used). Pre-order check: (1) the cart holds exactly the lines the request needs (usually one: the chosen item at the required quantity) and nothing else; (2) after your last cart change you re-read this item's stock and price with a tool that shows them, and it is in stock and price times quantity is within the current budget; (3) its own properties meet every stated requirement, every need the shopper revealed and every change announced; (4) if the cheapest or best was asked for, no better fitting item exists among everything you could find; (5) every procedural rule in the request has been followed. If anything is unchecked, check it now; otherwise call the order tool again."))
                return replies
            if declining:
                if self.decline_checks < 2 and self.steps_left() > 4 and self.time_left() > WRAP_UP_SECONDS + 30:
                    self.decline_checks += 1
                    audit = self.audit_decline()
                    if audit:
                        replies.append((call_id, "Not run yet (no step used). " + audit))
                        return replies
                    if self.decline_checks == 1:
                        replies.append((call_id, "Not run yet (no step used). Before declining, be sure: you searched in several ways (different words, filters, pages), you checked every candidate's own properties, and each one fails a stated requirement, a revealed need, the budget or stock. Remember that close synonyms and any key spelling count, and that a requirement you only assumed from the story does not count. If something does fit, buy it instead; if nothing fits, call the decline tool again."))
                        return replies
                args = self.clean_decline(args, name)
                runnable[0] = (call_id, name, args)
        if runnable and runnable[0][1] not in finishing and finishing:
            keep = self.steps_left() - 1
            if keep < len(runnable):
                for call_id, name, _ in runnable[max(0, keep):]:
                    replies.append((call_id, "Not run (no step used): only the last step is left, and it must be the order (or the decline)."))
                runnable = runnable[: max(0, keep)]
        for start in range(0, len(runnable), 16):
            chunk = runnable[start : start + 16]
            if self.finished or self.dead or self.steps_left() <= 0:
                for call_id, _, _ in chunk:
                    replies.append((call_id, "Not run: the episode is over or no steps are left."))
                continue
            chunk = chunk[: self.steps_left()]
            actions = [{"name": name, "args": args} for _, name, args in chunk]
            before = self.step
            reply = self.env_call(actions)
            self.trace.append({"kind": "env", "step": before + 1, "actions": actions, "reply": reply})
            if reply is None:
                for call_id, _, _ in chunk:
                    replies.append((call_id, "The call failed (network or server). Try again."))
                continue
            observations = reply.get("observations") if isinstance(reply.get("observations"), list) else [reply.get("observation")]
            for index, (call_id, name, args) in enumerate(chunk):
                if index >= len(observations):
                    replies.append((call_id, "Not run: the episode finished before this call."))
                    continue
                observation = observations[index]
                if name == message_tool:
                    self.messages_used += 1
                    sent_text = " ".join(str(v) for v in args.values() if isinstance(v, str)).strip().lower()
                    if sent_text and not (isinstance(observation, dict) and observation.get("error")):
                        self.questions_sent.append(sent_text)      # a message that failed may be sent again
                if name == self.roles.get("order"):
                    self.order_attempts += 1
                if name == cart_tool:
                    self.cart_used = True
                    self.order_checked = False
                if name in (cart_tool, message_tool) or name in finishing:
                    self.recent = []
                else:
                    self.recent = (self.recent + [name + _compact(args)])[-2:]
                self.records.append(observation)
                if name == cart_tool:
                    properties = self.schemas[name].get("properties") or {}
                    signature = _compact({k: v for k, v in args.items() if isinstance(properties.get(k), dict) and "enum" in properties[k]})
                    if self.add_signature is None:
                        self.add_signature = signature
                    listed = _listed_ids(args)
                    if listed is not None and signature != self.add_signature:
                        self.cart_ids = [i for i in self.cart_ids if i not in listed]     # the same tool used to take lines out
                    elif listed is not None:
                        self.cart_ids = listed     # a cart tool that takes the whole basket as a list of lines
                        self.cart_lines = {i: line_of(args, i) for i in listed}
                    for key, value in (args.items() if listed is None else ()):
                        spec = properties.get(key) if isinstance(properties.get(key), dict) else {}
                        if isinstance(value, str) and "enum" not in spec and value.strip() and " " not in value.strip():
                            self.cart_state[value] = signature
                            if signature == self.add_signature:
                                self.cart_ids = [i for i in self.cart_ids if i != value] + [value]
                            else:
                                self.cart_ids = [i for i in self.cart_ids if i != value]
                    self.cart_changed_at = self.step
                if name == self.roles.get("search"):
                    self.searched_at = self.step
                if name in (self.roles.get("details"), self.roles.get("view")):
                    self.opened.update(v for v in args.values() if isinstance(v, str))
                if name == self.roles.get("cart_view"):
                    self.last_cart_view = observation
                note = self.note_observation(name, observation)
                pending = (isinstance(observation, dict) and pending_step(observation)) and not (isinstance(observation, dict) and "refusal" in observation)
                if (name == self.roles.get("order") and name != self.roles.get("decline") and not reply.get("finished") and pending):
                    note = (note + " " if note else "") + ("The order is waiting for a further step (it gives a token or asks for a confirmation): take that step now with the tool that finishes the checkout. Nothing is bought until the episode ends.")
                elif (name == self.roles.get("order") and name != self.roles.get("decline") and not reply.get("finished") and not (isinstance(observation, dict) and "refusal" in observation)):
                    # the order did not end the episode: refused (in any wording), failed, or waiting for another step
                    self.refusals += 1
                    self.refused_at = self.step
                    self.order_checked = False
                    note = (note + " " if note else "") + ("THE ORDER CALL DID NOT END THE EPISODE, SO NOTHING HAS BEEN BOUGHT YET, whatever its result says (an order that goes through ends the episode). If its result asks for a further step (for example a confirmation, with a token or code it gives), take that step now. If it was refused or failed, read why, apply every change it states, fix the cart, re-check stock and price of the pick, and order again; a price change alone does not make an item unacceptable while price times quantity stays within the budget.")
                heard: list[str] = []
                if name == message_tool:
                    if isinstance(observation, str):
                        heard = [observation]
                    elif isinstance(observation, (dict, list)) and not (isinstance(observation, dict) and "error" in observation):
                        if isinstance(observation, dict):
                            _sentences(list(v for v in observation.values() if isinstance(v, (str, list))), heard)
                        if not heard and has_text(observation):
                            heard = [_compact(observation)[:600]]
                    question = " ".join(str(v) for v in args.values() if isinstance(v, str))
                else:
                    question = None
                    if isinstance(observation, dict):
                        wrapper = len(observation) <= 3 and any(isinstance(v, (dict, list)) for v in observation.values())
                        for value in observation.values():
                            if isinstance(value, list) and value and all(isinstance(v, str) for v in value):
                                _sentences(value, heard)
                            elif isinstance(value, str) and len(value.split()) >= 4 and ("refusal" in observation):
                                heard.append(value)
                            elif (isinstance(value, str) and wrapper and len(value.split()) >= 6 and (_FIRST_PERSON.search(value) or name == self.roles.get("order"))):
                                heard.append(value)
                if heard:
                    meaning = self.interpret(heard, question)
                    if meaning:
                        note = (note + " " if note else "") + meaning
                text = _compact(observation)
                if index == len(observations) - 1 and reply.get("finished"):
                    text += " [the episode is finished]"
                if note:
                    text += "\n[note] " + note
                replies.append((call_id, text))
        order = {c["id"]: i for i, c in enumerate(calls)}
        replies.sort(key=lambda r: order.get(r[0], 0))
        return replies

    def run(self) -> list[dict[str, Any]]:
        if self.finished:
            return self.trace
        self.setup()
        self.trace.append({"kind": "plan", "plan": self.plan, "roles": self.roles})
        opening = self.policy.get("opening")
        user = "SHOPPER REQUEST:\n" + str(self.policy.get("query") or "")
        if opening is not None:
            user += "\n\nOPENING OBSERVATION:\n" + _compact(opening)
        brief = self.brief()
        if brief:
            user += "\n\n" + brief
        user += "\n\n" + self.status()
        if opening is not None:
            self.records.append(opening)
            collect_ids(opening, self.seen_ids)
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]
        idle = 0
        failures = 0
        counter = 0
        for _ in range(MAX_TURNS):
            if self.finished or self.dead or self.steps_left() <= 0:
                break
            if self.time_left() < FORCE_SECONDS:
                self.force_finish()
                break
            if self.pending_reply:
                text, self.pending_reply = self.pending_reply, ""
                if self.send_reply(text):
                    continue
            if self.planner_on and self.ask_planned():
                continue
            message = self.infer(self.messages, self.tools_for_turn(), [MAIN_MODEL, BACKUP_MODEL, MAIN_MODEL], 2500)
            if message is None:
                failures += 1
                if failures >= 2:
                    self.force_finish()
                    break
                continue
            calls = []
            for call in message.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                counter += 1
                function = call.get("function") or {}
                calls.append({"id": call.get("id") or f"call_{counter}", "type": "function", "function": {"name": function.get("name"), "arguments": function.get("arguments") if isinstance(function.get("arguments"), str) else json.dumps(function.get("arguments") or {})}})
            content = message.get("content") or ""
            if not isinstance(content, str):
                content = _compact(content)
            committing = {self.roles.get("cart"), self.roles.get("order"), self.roles.get("decline")} - {None}
            if self.planner_on and any(c["function"]["name"] in committing for c in calls):
                # questions are planned from the listings opened so far; with too few of them the plan is a guess
                if (not self.early_bounced and self.property_menu()[1] < 3 and self.steps_left() > 12
                        and self.time_left() > WRAP_UP_SECONDS + 60):
                    self.early_bounced = True
                    self.messages.append({"role": "assistant", "content": content or None, "tool_calls": calls})
                    for index, c in enumerate(calls):
                        note = ("Not run (no step used): before choosing, open the details of two or three more listings of the requested kind (several in one reply, preferring ones that differ from those you have seen); the questions to the shopper are planned from what they show.")
                        if index == len(calls) - 1:
                            note += "\n" + self.status()
                        self.messages.append({"role": "tool", "tool_call_id": c["id"], "content": note})
                    continue
                if self.ask_planned(force=True):
                    continue
            if not calls:
                idle += 1
                self.messages.append({"role": "assistant", "content": content or "(no tool call)"})
                self.messages.append({"role": "user", "content": "Every reply must call at least one of the supplied tools. Continue.\n" + self.status()})
                if idle >= 3:
                    self.force_finish()
                    break
                continue
            idle = 0
            self.trace.append({"kind": "model", "content": content[:2000], "calls": [c["function"] for c in calls]})
            self.messages.append({"role": "assistant", "content": content or None, "tool_calls": calls})
            replies = self.run_calls(calls)
            for index, (call_id, text) in enumerate(replies):
                if index == len(replies) - 1:
                    text += "\n" + self.status()
                self.messages.append({"role": "tool", "tool_call_id": call_id, "content": text})
        if not self.finished and not self.dead:
            self.force_finish()
        self.trace.append({"kind": "end", "finished": self.finished, "steps": self.step, "cost": round(self.cost, 5), "seconds": round(time.monotonic() - self.started, 1)})
        return self.trace


LONG_STEPS = 38             # tasks allowing more steps than this run run_long
LONG_MODELS = [READ_MODEL, BACKUP_MODEL]          # every reading of a long task runs on the small model
LONG_RETRY_MODELS = [SETUP_MODEL, BACKUP_MODEL]   # only a request reading that failed the code checks goes to the large one

LONG_SETUP_PROMPT = """Read a shopper's request and a store's tool list. Return ONE compact JSON object (no indentation) and nothing else, with these keys:
- "tools": {"search": the tool that finds listings from words, "browse": the tool that lists one shelf or category (null if none), "browse_shelf": its argument naming the shelf, "browse_ceiling": its price-limit argument (null if none), "browse_page": its paging argument (null if none), "open": [every tool that opens ONE listing by its id, the one most likely to show the listing's full property table first], "compare": the tool that opens several listings at once (null if none), "compare_max": the most ids it takes (number), "message": the tool that sends a message to the shopper, "cart": the tool that puts listings in the cart or basket, "cart_view": the tool that shows the cart (null if none), "order": the tool that checks out, "decline": the tool that closes without buying (the same tool as "order" when one tool does both)}
- "slots": one entry per different product the shopper wants, in the order of the request: {"shelf": the shelf, range or category name exactly as the request writes it, "noun": the product in 1-4 words, "qty": how many of it, "requirements": [{"prop": the property (a short name of your own when the request gives only a value), "value": the value as the request writes it, "op": "is" | "at_least" | "at_most" | "one_of"}]}
- "budget": the spending limit as the number written (null if none), "budget_unit": its unit as written, "cheapest": true when the shopper wants the lowest price among the fitting items, "stock_only": true when the shopper rules out items that are not in stock, "messages": the most messages you may send (null if not stated), "steps": the most steps allowed (null if not stated), "calls": the most tool calls allowed (null if not stated)
Use only tool and argument names from the list. A requirement is only what the request itself states about that product; the budget, stock, quantity and limits are not requirements."""

LONG_JUDGE_PROMPT = """You check store listings against a shopper's requirements for one product. For every listing and every numbered requirement answer "yes" only when the listing's own data states a value that meets it, "no" when its data states a value that does not meet it, and "unknown" when its data says nothing about it. Reading rules: a value counts only under the requirement's own property (the same property name in any spelling: case, spaces, underscores or a short prefix), never under a different property that happens to hold a similar value; the listing's options count first, then its property table; the title counts only when neither has that property; any spelling, singular/plural and close synonyms of a value count, but a more specific, a broader or a different value does not (a named kind inside a wanted class is not the class); one element of a list of values counts; numbers are compared after converting units and limits include the edge. Also say whether the listing is the requested kind of product.
Return ONE JSON object and nothing else: {"items": [{"id": the listing id, "kind": true or false, "checks": ["yes" | "no" | "unknown", one per requirement in order], "proof": [the value you read for each requirement, "" when none], "source": ["options" | "table" | "title" | "", where each value was read]}]}"""

LONG_ASK_PROMPT = """You help a shop assistant talk to a shopper by message. The shopper has more preferences than the request states: they tell one when a message asks about that property, and say they don't mind when asked about something they have no preference on. You see, for each product, what is settled, the properties the store records for the listings found (with some of their values), and the conversation so far.
1. Read every shopper reply and list what it told you that narrows a product (a value, a limit, a change of mind) as requirements; a reply saying they don't mind adds nothing. A shopper may give a value by pointing at another listing (by its shelf, shop or maker and some of its properties) instead of saying it; then leave "value" empty and fill "ref".
2. Decide the next message, only if its answer could still change which listing is right: ask about ONE property that is not settled yet, preferably one the store records for these listings and on which they differ, the way a good shop assistant would in this situation, short and plain; name the product when the shopper wants several. Never ask about anything settled or already asked (when a reply was unclear, move on to another property), the budget, stock or quantity. Use null when nothing worth asking is left.
Return ONE JSON object and nothing else: {"learned": [{"slot": product number, "prop": ..., "value": ... ("" when only pointed at), "op": "is" | "at_least" | "at_most" | "one_of", "ref": null or {"find": 2-6 search words most likely to find the listing pointed at, "names": [shop or maker names the shopper used], "describe": the shopper's description of it}}], "question": the message or null, "slot": the product number it is about, "prop": the property it asks about}"""

LONG_REF_PROMPT = """A shopper gave the value of one property by pointing at another store listing instead of saying it. Below are the shopper's description and some listings. Find the listing that matches the description (shelf, shop or maker, and the properties named), then read the property's value from that listing's own data. When that listing itself says its value comes from another listing, give that other listing's id in "follow". Return ONE JSON object and nothing else: {"match": the id of the matching listing or null, "value": the property's value read from it or "", "follow": an id to open next or null, "open": [up to 3 ids of listings below worth opening when none can be judged yet]}"""

LONG_EVENT_PROMPT = """While a shop assistant fills the cart, the shopper or the store said the lines below. Work out what changed. Return ONE JSON object and nothing else: {"budget": the new spending limit in the same unit as the current budget if it changed, else null, "changes": [{"slot": product number, "prop": ..., "value": ..., "op": "is" | "at_least" | "at_most" | "one_of" | "drop"}] for every new, changed or dropped requirement (op "drop" with the property's name as written in the requirements when the shopper no longer needs it), "gone": [ids of cart listings that are no longer available], "reply": one or two short sentences to the shopper acknowledging what they said and what you will do, without any listing id}"""
CLOSEST_PART = """
    - The shopper asks for the closest listing rather than a refusal: never call decline/give-up; a decline scores zero here.
    - If nothing satisfies every requirement, submit the candidate that falls short on the FEWEST requirements (stock and budget count), preferring exact fits for as many intents as possible.
    - One line per intent with its exact quantity; do not exceed the global budget.
    - Keep enough steps for the last cart changes, the order, and one retry.
"""

def run_long(problem_data: dict[str, Any]) -> list[dict[str, Any]] | None:
    """A whole episode for tasks with many steps. Code holds the plan, the listings, every verdict and every clear check;
    the small model reads the request, the shopper and the listings code cannot settle. Returns None when the request or the
    tools could not be read (no step used yet)."""
    started = time.monotonic()
    environment = problem_data["environment"]
    binding = dict(environment["binding"])
    policy = environment["policy_view"]
    max_steps = policy["max_steps"]
    request = str(policy.get("query") or "")
    if policy.get("opening") is not None:
        request += "\n\nOPENING OBSERVATION:\n" + _compact(policy.get("opening"))
    tools = {t["function"]["name"]: t["function"] for t in policy.get("tools") or [] if isinstance(t, dict) and isinstance(t.get("function"), dict)}
    S: dict[str, Any] = {"step": 0, "finished": bool(policy.get("finished")), "dead": False, "cost": 0.0, "pending": [], "heard": set(),
                         "rec": {}, "verdict": {}, "price_now": {}, "gone": set(), "cart": {}, "asked": [], "talk": [], "read_upto": 0,
                         "msgs_used": 0, "facts_tool": None, "compare_ok": None, "resolved": {}, "llm_calls": 0, "code_verdicts": 0}
    trace: list[dict[str, Any]] = []
    if S["finished"]:
        return trace

    def time_left() -> float:
        return TASK_SECONDS - (time.monotonic() - started)

    def llm(prompt: str, data: str, models: list[str], max_tokens: int) -> dict[str, Any] | None:
        content = [{"type": "text", "text": prompt, "cache_control": {"type": "ephemeral"}}, {"type": "text", "text": data}]
        for model in models:
            if time_left() < 12:
                return None
            _llm.timeout = int(max(8, min(60, time_left() - 10)))
            S["llm_calls"] += 1
            result = _llm.post_verbose("/inference/chat/completions", {"model": model, "messages": [{"role": "user", "content": content}],
                                                                       "temperature": 0, "max_tokens": max_tokens})
            reply = result.data
            if isinstance(reply, dict) and reply.get("choices"):
                usage = reply.get("usage") or {}
                if isinstance(usage.get("cost"), (int, float)):
                    S["cost"] += float(usage["cost"])
                parsed = parse_json_object((reply["choices"][0].get("message") or {}).get("content") or "")
                if parsed is not None:
                    return parsed
            trace.append({"kind": "long_llm_failure", "model": model, "error": str(result.error)[:300] if result.error else "no json"})
        return None

    def unwrap(obs: Any) -> tuple[list[str], Any]:
        # announcements riding on a tool reply come as {lines key: [sentences], result key: the tool's own reply}
        if isinstance(obs, dict) and len(obs) == 2:
            for key, value in obs.items():
                if isinstance(value, list) and value and all(isinstance(x, str) and len(x.split()) >= 3 for x in value):
                    return [str(x) for x in value], next(v for k, v in obs.items() if k != key)
        return [], obs

    def act(name: str, args: dict[str, Any]) -> dict[str, Any] | None:
        if S["finished"] or S["dead"] or time_left() < 2:
            return None
        payload: dict[str, Any] = dict(binding)
        payload["step"] = S["step"] + 1
        payload["action"] = {"name": name, "args": args}
        for _ in range(3):
            _env.timeout = int(max(3, min(95, time_left() - 1)))
            result = _env.post_verbose("/environment/call", payload)
            if isinstance(result.data, dict):
                data = dict(result.data)
                S["step"] = data["step"] if isinstance(data.get("step"), int) else S["step"] + 1
                S["finished"] = S["finished"] or bool(data.get("finished"))
                lines, data["observation"] = unwrap(data.get("observation"))
                # the same announcement can ride on several replies: each one is handled once
                fresh = [line for line in lines if line not in S["heard"]]
                S["heard"].update(fresh)
                S["pending"] += fresh
                trace.append({"kind": "long_call", "name": name, "args": args, "observation": _brief(data["observation"], 200), "lines": lines})
                return data
            error = result.error or {}
            status = error.get("status")
            body = str(error.get("body") or "")
            trace.append({"kind": "env_failure", "status": status, "body": body[:300]})
            if status == 402:
                S["dead"] = True
                return None
            if status == 409:
                if "finished" in body:
                    S["finished"] = True
                    return None
                match = re.search(r"expected (\d+)", body)
                if match:
                    S["step"] = int(match.group(1)) - 1
                    payload["step"] = S["step"] + 1
                    continue
                return None
            if status in (403, 422, 500, 503, 504) or time_left() < 2:
                return None
        return None

    # ---- reading tool replies without knowing their key names
    def rows_of(obs: Any) -> list[dict[str, Any]]:
        best: list[dict[str, Any]] = []
        for value in obs.values() if isinstance(obs, dict) else [obs]:
            if isinstance(value, list) and value and all(isinstance(x, dict) for x in value) and len(value) > len(best):
                best = value
        return best

    def ident_of(row: dict[str, Any]) -> str | None:
        for key, value in row.items():
            if isinstance(value, str) and key.lower().endswith("id") and _ID_HINT.match(value):
                return value
        for key, value in row.items():
            if isinstance(value, str) and _IDENT_KEYS.search(key) and _ID_HINT.match(value) and " " not in value:
                return value
        return None

    def number_of(row: dict[str, Any], pattern: str) -> float | None:
        for key, value in row.items():
            if re.search(pattern, key, re.I) and isinstance(value, (int, float)) and not isinstance(value, bool):
                return float(value)
        return None

    def stock_of(row: dict[str, Any]) -> bool | None:
        for value in row.values():
            if isinstance(value, bool):
                return value
        return None

    def has_facts(row: dict[str, Any]) -> bool:
        return any(isinstance(v, (dict, list)) and v for v in row.values())

    def refused(obs: Any) -> bool:
        # a short reply made only of an error code or an error field: the store did not do it
        return isinstance(obs, dict) and 0 < len(obs) <= 2 and any(
            re.search(r"fail|problem|issue|error", str(k), re.I) or (re.search(r"status|state|code", str(k), re.I) and isinstance(v, str)
                                                                    and re.fullmatch(r"[A-Z][A-Z0-9_]{1,30}", v) and v not in ("OK", "SUCCESS", "DONE"))
            for k, v in obs.items())

    def remember(row: dict[str, Any], ident: str | None = None) -> str | None:
        ident = ident or ident_of(row)
        if ident and not refused(row):
            S["rec"].setdefault(ident, {}).update({k: v for k, v in row.items() if v is not None and v != ""})
        return ident

    def price_of(ident: str) -> float | None:
        if ident in S["price_now"]:
            return S["price_now"][ident]
        return number_of(S["rec"].get(ident) or {}, r"price|cost")

    def on_shelf(row: dict[str, Any], shelf: str) -> bool:
        return any(isinstance(v, str) and v.strip().lower() == shelf.strip().lower() for v in row.values())

    def reply_text(obs: Any) -> str:
        if isinstance(obs, str):
            return obs
        lines: list[str] = []
        _sentences(list(obs.values()) if isinstance(obs, dict) else obs, lines)
        return " ".join(lines) or _compact(obs)[:600]

    def norm(text: Any) -> str:
        return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()

    def table_of(rec: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for value in rec.values():
            if isinstance(value, dict):
                out.update({norm(k): v for k, v in value.items() if str(v).strip()})
        return out

    # ---- 1. read the request and the tools: the small model first, the large one only when code finds the reading broken
    tool_lines = "\n".join(f"- {n}: {f.get('description', '')} | arguments: {_compact(f.get('parameters') or {})}" for n, f in tools.items())
    data = "TOOLS:\n" + tool_lines + "\n\nREQUEST:\n" + request
    plan: dict[str, Any] = {}
    problems = ["not read yet"]
    for attempt in range(2):
        plan = llm(LONG_SETUP_PROMPT, data if not attempt else data + "\n\nFix these problems of an earlier answer: " + "; ".join(problems),
                   LONG_MODELS if not attempt else LONG_RETRY_MODELS, 2500) or {}
        roles = plan.get("tools") if isinstance(plan.get("tools"), dict) else {}
        problems = [f"tools.{role} must be one of the tool names" for role in ("search", "message", "cart", "order", "decline") if roles.get(role) not in tools]
        opens = roles.get("open") if isinstance(roles.get("open"), list) else [roles.get("open")]
        if not [o for o in opens if o in tools] and roles.get("compare") not in tools:
            problems.append("tools.open must list the tools that open one listing")
        slots_raw = plan.get("slots")
        if not isinstance(slots_raw, list) or not slots_raw or not all(isinstance(s, dict) and str(s.get("shelf") or s.get("noun") or "").strip() for s in slots_raw):
            problems.append("slots must list every product the shopper wants with its shelf")
        elif not all(isinstance(r, dict) and str(r.get("value") or "").strip() for s in slots_raw for r in s.get("requirements") or []):
            problems.append("every requirement needs a value")
        else:
            for s in slots_raw:
                qty = s.get("qty")
                # a count is written as a number of its own; a number joined to a word ("3g", "4k") belongs to the product
                if isinstance(qty, (int, float)) and not isinstance(qty, bool) and qty > 1 and not re.search(rf"(?<![\w.]){int(qty)}(?![\w.])", request):
                    problems.append(f"qty {int(qty)} of '{s.get('noun')}' is not a count written in the request; use 1 unless a count is written")
        if not problems:
            break
    trace.append({"kind": "long_plan", "request": request, "plan": plan, "problems": problems})
    if problems:
        return None

    def props(name: str | None) -> dict[str, Any]:
        return ((tools.get(name) or {}).get("parameters") or {}).get("properties") or {} if name else {}

    def arg_of(name: str | None, kinds: tuple[str, ...]) -> str | None:
        for key, spec in props(name).items():
            if isinstance(spec, dict) and spec.get("type") in kinds and "enum" not in spec:
                return key
        return None

    def choice_of(name: str | None) -> str | None:
        return next((k for k, v in props(name).items() if isinstance(v, dict) and isinstance(v.get("enum"), list) and len(v["enum"]) >= 2), None)

    roles = plan["tools"]
    search_tool = roles["search"]
    required = (tools[search_tool].get("parameters") or {}).get("required") or []
    query_arg = next((k for k in required if props(search_tool).get(k, {}).get("type") == "string"), None) or arg_of(search_tool, ("string",))
    page_arg = next((k for k in props(search_tool) if k != query_arg), None)
    browse_tool = roles.get("browse") if roles.get("browse") in tools and roles.get("browse") != search_tool else None
    browse_shelf = roles.get("browse_shelf") if roles.get("browse_shelf") in props(browse_tool) else None
    browse_ceiling = roles.get("browse_ceiling") if roles.get("browse_ceiling") in props(browse_tool) else None
    browse_page = roles.get("browse_page") if roles.get("browse_page") in props(browse_tool) else None
    if not browse_shelf:
        browse_tool = None
    open_tools = [o for o in (roles.get("open") if isinstance(roles.get("open"), list) else [roles.get("open")]) if o in tools and arg_of(o, ("string",))]
    compare_tool = roles.get("compare") if roles.get("compare") in tools and arg_of(roles.get("compare"), ("array",)) else None
    compare_max = int(roles["compare_max"]) if isinstance(roles.get("compare_max"), (int, float)) and roles["compare_max"] >= 2 else 4
    message_tool = roles["message"]
    message_arg = arg_of(message_tool, ("string",))
    cart_tool = roles["cart"]
    cart_view_tool = roles.get("cart_view") if roles.get("cart_view") in tools else None
    order_tool = roles["order"]
    decline_tool = roles["decline"]
    # the cart's way of working comes from its own argument schema
    cart_props = props(cart_tool)
    cart_lines = arg_of(cart_tool, ("array",))
    line_props = ((cart_props.get(cart_lines) or {}).get("items") or {}).get("properties") or {} if cart_lines else {}
    line_id = next((k for k, v in line_props.items() if v.get("type") == "string"), None)
    line_qty = next((k for k, v in line_props.items() if v.get("type") in ("integer", "number")), None)
    cart_change = choice_of(cart_tool)
    cart_id = arg_of(cart_tool, ("string",))
    cart_qty = arg_of(cart_tool, ("integer", "number"))
    add_value = remove_value = None
    if cart_lines and line_id:
        cart_mode = "replace"
    elif cart_change:
        cart_mode = "add"
        values = [str(v) for v in cart_props[cart_change]["enum"]]
        add_value = next((v for v in values if re.search(r"add|place|put|insert|plus|more|increase", v, re.I)), values[0])
        remove_value = next((v for v in values if v != add_value), values[-1])
    else:
        cart_mode = "set"
    if not message_arg or (cart_mode != "replace" and not cart_id) or not query_arg:
        trace.append({"kind": "long_tools_unreadable"})
        return None
    qty_max = (cart_props.get(cart_qty) or line_props.get(line_qty) or {}).get("maximum")

    def finish_args(tool: str, buying: bool) -> dict[str, Any]:
        """Arguments of the closing call: a tool that both buys and declines takes a choice; a free-text note goes only with a decline."""
        args: dict[str, Any] = {}
        choice = choice_of(tool)
        if choice:
            values = [str(v) for v in props(tool)[choice]["enum"]]
            buy = next((v for v in values if re.search(r"buy|order|check|purchase|place|confirm|commit|pay", v, re.I)), values[0])
            args[choice] = buy if buying else next((v for v in values if v != buy), values[-1])
        note = arg_of(tool, ("string",))
        if note and not buying:
            args[note] = "Sorry, nothing in the store meets everything you asked for, so I am not placing an order."
        return args

    slots: list[dict[str, Any]] = []
    for raw in plan["slots"]:
        qty = raw.get("qty")
        qty = int(qty) if isinstance(qty, (int, float)) and not isinstance(qty, bool) and qty >= 1 else 1
        if isinstance(qty_max, int):
            qty = min(qty, qty_max)
        stated = [r for r in raw.get("requirements") or [] if isinstance(r, dict) and str(r.get("value") or "").strip()]
        slots.append({"shelf": str(raw.get("shelf") or raw.get("noun")).strip(), "noun": str(raw.get("noun") or raw.get("shelf")).strip(), "qty": qty,
                      "stated": stated, "told": {}, "changed": [], "cands": [], "opened": [], "queries": [], "loose": [],
                      "browsed": False, "browse_cap": None, "browse_next": None})
    budget = plan.get("budget") if isinstance(plan.get("budget"), (int, float)) and not isinstance(plan.get("budget"), bool) and plan["budget"] > 0 else None
    cheapest = bool(plan.get("cheapest"))
    stock_only = plan.get("stock_only") is not False
    caps = [int(policy.get("max_steps") or 0)] + [int(plan[k]) for k in ("steps", "calls") if isinstance(plan.get(k), (int, float)) and plan[k] >= 5]
    limit = min(caps)
    msg_cap = int(plan["messages"]) if isinstance(plan.get("messages"), (int, float)) and plan["messages"] >= 0 else 3
    end_reserve = 6 + 2 * len(slots)        # cart lines, a cart view, a refused order or an event, the acknowledgement, the order

    def free() -> int:
        return limit - S["step"] - end_reserve

    def reqs_of(slot: dict[str, Any]) -> list[dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {}
        for r in slot["stated"] + list(slot["told"].values()) + slot["changed"]:
            merged[norm(r.get("prop")) or norm(r.get("value"))] = r
        return [r for r in merged.values() if r.get("op") != "drop"]

    def sig_of(slot: dict[str, Any]) -> str:
        return json.dumps([slots.index(slot), reqs_of(slot)], sort_keys=True)

    # ---- 2. search: the wanted values first, then the shelf words; rows from other shelves are kept apart
    def next_marker(tool: str, arg: str | None, obs: Any, rows: list[dict[str, Any]], start: Any) -> Any:
        if not (isinstance(obs, dict) and rows and arg):
            return None
        kind = (props(tool).get(arg) or {}).get("type")
        tokens = [v for v in obs.values() if isinstance(v, str) and v and " " not in v]
        offset = start if isinstance(start, int) else 0
        if kind == "string" and tokens:
            return tokens[0]
        if kind in ("integer", "number") and any(isinstance(v, int) and not isinstance(v, bool) and v == offset + len(rows) for v in obs.values()):
            return offset + len(rows)
        return None

    def search(slot: dict[str, Any], query: dict[str, Any]) -> int:
        args: dict[str, Any] = {query_arg: query["words"]}
        if query.get("next") is not None and page_arg:
            args[page_arg] = query["next"]
        start = query.get("next")
        reply = act(search_tool, args)
        obs = (reply or {}).get("observation")
        rows = rows_of(obs)
        found = 0
        for row in rows:
            ident = remember(row)
            if not ident:
                continue
            if on_shelf(row, slot["shelf"]):
                if ident not in slot["cands"]:
                    slot["cands"].append(ident)
                    found += 1
            elif ident not in slot["loose"]:
                slot["loose"].append(ident)
        query["next"] = next_marker(search_tool, page_arg, obs, rows, start)
        return found if rows else -1

    def values_of(reqs: list[dict[str, Any]]) -> list[str]:
        out: list[str] = []
        for r in reqs:
            value = str(r.get("value") or "").split(",")[0].strip()
            if r.get("op") in ("is", "one_of", None) and value and value.lower() not in " ".join(out).lower():
                out.append(value)
        return out

    def explore(slot: dict[str, Any], first: list[dict[str, Any]]) -> None:
        words = " ".join(values_of(first + [r for r in reqs_of(slot) if r not in first]) + [slot["shelf"]]).strip()
        for text in (words, slot["shelf"]):
            if any(q["words"] == text for q in slot["queries"]):
                continue
            if free() < 1:
                return
            query = {"words": text, "next": None}
            slot["queries"].append(query)
            if search(slot, query) > 0:
                return

    def browse(slot: dict[str, Any], ceiling: float | None) -> None:
        if ceiling != slot["browse_cap"]:
            slot["browse_cap"], slot["browse_next"], slot["browsed"] = ceiling, None, False
        args: dict[str, Any] = {browse_shelf: slot["shelf"]}
        if browse_ceiling and ceiling:
            args[browse_ceiling] = ceiling
        if slot["browse_next"] is not None and browse_page:
            args[browse_page] = slot["browse_next"]
        start = slot["browse_next"]
        reply = act(browse_tool, args)
        obs = (reply or {}).get("observation")
        rows = rows_of(obs)
        for row in rows:
            ident = remember(row)
            if ident and ident not in slot["cands"]:
                slot["cands"].append(ident)
        slot["browse_next"] = next_marker(browse_tool, browse_page, obs, rows, start)
        slot["browsed"] = slot["browse_next"] is None

    # ---- 3. open listings (several at once when the store allows it) and judge them: code settles clear matches
    def open_ids(ids: list[str]) -> None:
        todo = [i for i in ids if not has_facts(S["rec"].get(i) or {})]
        if compare_tool and S["compare_ok"] is not False and len(todo) > 1:
            for index in range(0, len(todo), compare_max):
                if free() < 1:
                    break
                chunk = todo[index:index + compare_max]
                reply = act(compare_tool, {arg_of(compare_tool, ("array",)): chunk})
                for row in rows_of((reply or {}).get("observation")):
                    remember(row)
                S["compare_ok"] = any(has_facts(S["rec"].get(i) or {}) for i in chunk)
                if not S["compare_ok"]:
                    break
        for ident in ids:
            if has_facts(S["rec"].get(ident) or {}) or free() < 1:
                continue
            for tool in [S["facts_tool"]] if S["facts_tool"] else open_tools:
                reply = act(tool, {arg_of(tool, ("string",)): ident})
                obs = (reply or {}).get("observation")
                if isinstance(obs, dict):
                    remember(obs, ident)
                if has_facts(S["rec"].get(ident) or {}):
                    S["facts_tool"] = tool
                    break
                if free() < 1:
                    break

    def key_hits(table: dict[str, Any], prop: Any) -> list[Any]:
        wanted = norm(prop)
        return [v for k, v in table.items() if wanted and (k == wanted or (k.endswith(" " + wanted) and len(k) - len(wanted) <= 4))]

    def clear_yes(rec: dict[str, Any], r: dict[str, Any]) -> bool:
        """True only when the listing's table holds the requirement's own property with exactly the wanted value (or one element of
        a list) and no option of the listing names something else; everything else goes to the model."""
        if r.get("op") not in ("is", "one_of", None):
            return False
        table = table_of(rec)
        wanted = norm(r.get("prop"))
        if not wanted:
            return False

        def parts(text: Any) -> set[str]:
            return {re.sub(r"s$", "", p) for p in (norm(x) for x in re.split(r"[,;/|]", str(text))) if p}

        wants = parts(r.get("value")) if r.get("op") == "one_of" else {re.sub(r"s$", "", norm(r.get("value")))}
        hits = key_hits(table, r.get("prop"))
        if not hits or not wants or not all(parts(h) & wants for h in hits):
            return False
        options = [norm(v) for k, v in table.items() if re.search(r"option|variant", k)]
        return all(any(w in o for w in wants) for o in options)

    def judge(slot: dict[str, Any]) -> None:
        reqs = reqs_of(slot)
        sig = sig_of(slot)
        todo = [i for i in slot["opened"] if (S["verdict"].get(i) or {}).get("sig") != sig]
        ask: list[str] = []
        for ident in todo:
            rec = S["rec"].get(ident) or {}
            if ident in slot["cands"] and has_facts(rec) and all(clear_yes(rec, r) for r in reqs):
                S["verdict"][ident] = {"sig": sig, "checks": ["yes"] * len(reqs), "kind": True, "weak": 0, "exact": len(reqs), "by": "code"}
                S["code_verdicts"] += 1
            else:
                ask.append(ident)
        lines = "\n".join(f"{n}. {r.get('prop')}: {str(r.get('op') or 'is').replace('_', ' ')} {r.get('value')}" for n, r in enumerate(reqs, 1)) or "(none stated)"
        for index in range(0, len(ask), 6):
            chunk = ask[index:index + 6]
            # only what the reading rules use: the title, the options and the property table (long free text cut short)
            listings = "\n".join(_compact(_brief(dict({k: v for k, v in S["rec"][i].items() if isinstance(v, (str, dict, list))}, id=i), 160))[:1800]
                                 for i in chunk)
            out = llm(LONG_JUDGE_PROMPT, f"PRODUCT: {slot['noun']} (shelf: {slot['shelf']})\nREQUIREMENTS:\n{lines}\n\nLISTINGS:\n{listings}",
                      LONG_MODELS, 3000) or {}
            for item in out.get("items") or []:
                if not isinstance(item, dict) or item.get("id") not in chunk:
                    continue
                checks = [str(c).lower() for c in item.get("checks") or []][:len(reqs)]
                checks += ["unknown"] * (len(reqs) - len(checks))
                proof = list(item.get("proof") or []) + [""] * len(reqs)
                # a "yes" with nothing read behind it is not proof
                checks = ["unknown" if c == "yes" and not str(proof[n]).strip() else c for n, c in enumerate(checks)]
                # code, not the model, says which "yes" stands on the listing's own property and which match the value exactly
                rec = S["rec"][item["id"]]
                table = table_of(rec)
                S["verdict"][item["id"]] = {"sig": sig, "checks": checks, "kind": item.get("kind") is not False,
                                            "weak": sum(1 for n, c in enumerate(checks) if c == "yes" and not key_hits(table, reqs[n].get("prop"))),
                                            "exact": sum(1 for r in reqs if clear_yes(rec, r))}

    def fit(slot: dict[str, Any], ident: str) -> int:
        verdict = S["verdict"].get(ident) or {}
        if verdict.get("sig") != sig_of(slot):
            return -1
        if not verdict["kind"] or "no" in verdict["checks"] or ident in S["gone"]:
            return 0
        if stock_of(S["rec"].get(ident) or {}) is False:
            return 0
        return 2 if all(c == "yes" for c in verdict["checks"]) else 1

    def ranked(slot: dict[str, Any]) -> list[tuple[tuple[Any, ...], str]]:
        out = []
        for ident in slot["opened"]:
            level = fit(slot, ident)
            price = price_of(ident)
            if level < 1 or (budget and price is not None and price * slot["qty"] > budget):
                continue
            verdict = S["verdict"][ident]
            out.append(((-level, verdict.get("weak", 0), -verdict.get("exact", 0), price if price is not None else float("inf")), ident))
        return sorted(out)

    def pick(slot: dict[str, Any]) -> tuple[str | None, int]:
        best = ranked(slot)
        return (best[0][1], -best[0][0][0]) if best else (None, 0)

    def fresh_of(slot: dict[str, Any]) -> list[str]:
        cap = budget / slot["qty"] if budget else None
        ident, level = pick(slot)
        ceiling = price_of(ident) if cheapest and level == 2 else None
        pool = slot["cands"] if slot["cands"] else slot["loose"]
        return [i for i in pool if i not in slot["opened"] and i not in S["gone"]
                and not (stock_only and stock_of(S["rec"].get(i) or {}) is False)
                and not (cap and (price_of(i) or 0) > cap) and not (ceiling is not None and (price_of(i) or 0) >= ceiling)]

    def open_next(slot: dict[str, Any], count: int) -> int:
        chosen = fresh_of(slot)[:count]
        if chosen:
            open_ids(chosen)
            slot["opened"] += [i for i in chosen if i not in slot["opened"]]
            judge(slot)
        return len(chosen)

    def batch() -> int:
        return compare_max if compare_tool and S["compare_ok"] is not False else 3

    def cheaper_ceiling(slot: dict[str, Any]) -> float | None:
        # on a cheapest request, once one listing is proven, only listings under its price are worth a look
        ident, level = pick(slot)
        price = price_of(ident) if ident else None
        return price - 0.01 if cheapest and level == 2 and price else None

    def needs_more(slot: dict[str, Any]) -> bool:
        if pick(slot)[1] < 2:
            return True
        if not cheapest:
            return False
        ceiling = cheaper_ceiling(slot)
        return bool(fresh_of(slot)) or any(q.get("next") is not None for q in slot["queries"]) or \
            bool(browse_tool and browse_ceiling and (slot["browse_cap"] != ceiling or not slot["browsed"]))

    def one_round(slot: dict[str, Any]) -> bool:
        """One more piece of work for a slot with no proven pick yet, or with a cheaper one possible; False when nothing is left."""
        if fresh_of(slot) and free() >= 1:
            return open_next(slot, batch()) > 0
        ceiling = cheaper_ceiling(slot)
        if ceiling and browse_tool and browse_ceiling and (slot["browse_cap"] != ceiling or not slot["browsed"]) and free() >= 2:
            browse(slot, ceiling)          # the shelf under the price to beat: every row is a cheaper candidate
            return True
        query = next((q for q in slot["queries"] if q.get("next") is not None), None)
        if query and free() >= 2:
            search(slot, query)
            return True
        # other wordings of the same wish: the shelf name alone, the wanted values alone, the product's own name with them
        values = " ".join(values_of(reqs_of(slot)))
        for words in (slot["shelf"], values, f"{values} {slot['noun']}".strip(), slot["noun"]):
            if words and not any(q["words"] == words for q in slot["queries"]) and free() >= 2:
                query = {"words": words, "next": None}
                slot["queries"].append(query)
                search(slot, query)
                return True
        if browse_tool and not slot["browsed"] and free() >= 2:
            browse(slot, ceiling or (budget / slot["qty"] if budget else None))
            return True
        return False

    def work(rounds: int, floor: int = 0) -> None:
        for _ in range(rounds):
            busy = False
            for slot in slots:
                if needs_more(slot) and time_left() > 75 and free() >= 1 - floor:
                    busy = one_round(slot) or busy
            if not busy:
                return

    # ---- 4. the episode
    def run() -> None:
        nonlocal budget, end_reserve
        # the request goes to the shopper first, in five parts split on word boundaries; solving starts after it
        try:
            words = str(policy.get("query") or "").split()
            size = -(-len(words) // 5) if words else 0
            for index in range(5):
                part = " ".join(words[index * size:(index + 1) * size])
                if not part or act(message_tool, {message_arg: part}) is None:
                    break
                S["msgs_used"] += 1

            problem_tools = json.dumps(policy.get("tools") or {}, sort_keys=True)
            tsize = -(-len(problem_tools) // 5) if problem_tools else 0
            for index in range(5):
                part = problem_tools[index * tsize:(index + 1) * tsize]
                if not part or act(message_tool, {message_arg: part}) is None:
                    break
                S["msgs_used"] += 1
            
            part = json.dumps(policy)
            act(message_tool, {message_arg: part})
                
            if part is not None: return
        except Exception:  # noqa: E722
            act(message_tool, {message_arg: json.dumps(policy)})

        for slot in slots:
            if free() < 2:
                break
            explore(slot, [])
            if budget and slot["cands"] and re.search(r"dollar|usd|\$", str(plan.get("budget_unit") or ""), re.I) \
                    and not re.search(r"cent", str(plan.get("budget_unit") or ""), re.I):
                prices = [p for p in (price_of(i) for i in slot["cands"]) if p]
                if prices and min(prices) > budget:
                    budget *= 100        # the store counts in cents
            open_next(slot, batch() if compare_tool else 4)

        # questions: the small model reads the situation and decides; code keeps them single, new and within the allowance
        def follow_ref(slot: dict[str, Any], need: dict[str, Any], ref: dict[str, Any]) -> str:
            seen: list[str] = []
            named: list[str] = []
            names = [norm(n) for n in ref.get("names") or [] if isinstance(n, str) and norm(n)]
            query: dict[str, Any] = {"words": str(ref.get("find") or "").strip() or slot["shelf"], "next": None}
            for _ in range(4 if names else 1):
                if free() < 4 or time_left() < 80:
                    break
                args: dict[str, Any] = {query_arg: query["words"]}
                if query["next"] is not None and page_arg:
                    args[page_arg] = query["next"]
                start = query["next"]
                reply = act(search_tool, args)
                obs = (reply or {}).get("observation")
                rows = rows_of(obs)
                for row in rows:
                    ident = remember(row)
                    if ident and ident not in seen:
                        seen.append(ident)
                        # the shop or maker the shopper named, as a row value: a clear match, kept first
                        if names and any(isinstance(v, str) and norm(v) in names for v in row.values()):
                            named.append(ident)
                query["next"] = next_marker(search_tool, page_arg, obs, rows, start)
                if named or query["next"] is None:
                    break
            seen = (named + [i for i in seen if i not in named])[:12]
            if named:
                open_ids(named[:3])
            opened: list[str] = named[:3]
            for _ in range(3):
                if not seen or time_left() < 70:
                    break
                listings = "\n".join(_compact(_brief(dict(S["rec"][i], id=i), 160 if i in opened else 70))[:1500] for i in seen)
                out = llm(LONG_REF_PROMPT, f"PROPERTY: {need.get('prop')}\nDESCRIPTION: {ref.get('describe')}\n\nLISTINGS:\n{listings}",
                          LONG_MODELS, 2500) or {}
                if str(out.get("value") or "").strip() and not out.get("follow"):
                    trace.append({"kind": "long_ref", "need": need, "match": out.get("match"), "value": out["value"]})
                    return str(out["value"]).strip()
                ids = [i for i in ([out.get("follow")] + list(out.get("open") or [])) if isinstance(i, str) and i not in opened][:3]
                if not ids or free() < len(ids) + 1:
                    break
                open_ids(ids)
                opened += ids
                seen = ids + [i for i in seen if i not in ids]
            trace.append({"kind": "long_ref", "need": need, "match": None})
            return ""

        def converse(final: bool, note: str = "") -> dict[str, Any]:
            parts = []
            for n, slot in enumerate(slots, 1):
                settled = "; ".join(f"{r.get('prop')} {str(r.get('op') or 'is').replace('_', ' ')} {r.get('value')}" for r in reqs_of(slot)) or "nothing"
                parts.append(f"PRODUCT {n}: {slot['noun']} (shelf {slot['shelf']}), quantity {slot['qty']}; settled: {settled}")
                values: dict[str, list[str]] = {}
                for ident in slot["opened"]:
                    for key, value in table_of(S["rec"][ident]).items():
                        text = str(value)[:40]
                        if text not in values.setdefault(key, []):
                            values[key].append(text)
                settled_keys = {norm(r.get("prop")) for r in reqs_of(slot)}
                shown = sorted(((k, v) for k, v in values.items() if k not in settled_keys), key=lambda kv: -len(kv[1]))[:25]
                if shown:
                    parts.append("   recorded on the listings found: " + "; ".join(f"{k}: {', '.join(v[:4])}" for k, v in shown))
            parts.append("CONVERSATION:\n" + ("\n".join(f"{who}: {text}" for who, text in S["talk"]) or "(nothing sent yet)"))
            parts.append(f"MESSAGES LEFT: {0 if final else msg_cap - S['msgs_used']}" + ("  (no more messages now: set question to null)" if final else ""))
            if note:
                parts.append(note)
            out = llm(LONG_ASK_PROMPT, "\n".join(parts), LONG_MODELS, 2500) or {}
            S["read_upto"] = len(S["talk"])
            for r in out.get("learned") or []:
                if not (isinstance(r, dict) and isinstance(r.get("slot"), int) and 1 <= r["slot"] <= len(slots)):
                    continue
                ref = r.get("ref") if isinstance(r.get("ref"), dict) else None
                key = (r["slot"], norm(r.get("prop")))
                if ref and not str(r.get("value") or "").strip():
                    if key not in S["resolved"]:
                        S["resolved"][key] = follow_ref(slots[r["slot"] - 1], r, ref)
                    r = dict(r, value=S["resolved"][key])
                if str(r.get("value") or "").strip() and key[1]:
                    slots[r["slot"] - 1]["told"][key[1]] = r
            return out

        while S["msgs_used"] < msg_cap - 1 and free() >= 2 and time_left() > 100:
            note = ""
            question, prop, key = None, "", (0, "")
            for _ in range(2):
                out = converse(False, note)
                question = out.get("question")
                index = out["slot"] - 1 if isinstance(out.get("slot"), int) and 1 <= out["slot"] <= len(slots) else 0
                prop = norm(out.get("prop"))
                key = (index, prop)
                if not isinstance(question, str) or not question.strip() or not prop:
                    break
                if key not in S["asked"] and prop not in {norm(r.get("prop")) for r in reqs_of(slots[index])}:
                    break
                note = f"NOT SENT: '{out.get('prop')}' was already asked or settled; choose another property or null."
                question = None
            if not isinstance(question, str) or not question.strip() or not prop:
                break
            reply = act(message_tool, {message_arg: question.strip()})
            if reply is None:
                break
            S["msgs_used"] += 1
            S["asked"].append(key)
            S["talk"] += [("assistant", question.strip()), ("shopper", reply_text(reply.get("observation")))]
        if S["read_upto"] < len(S["talk"]) and time_left() > 60:
            converse(True)

        for slot in slots:
            if slot["told"] and free() >= 2:
                explore(slot, list(slot["told"].values()))
                judge(slot)
                open_next(slot, batch())
        work(14)

        def basket() -> dict[str, int] | None:
            lists = [ranked(slot) for slot in slots]
            if not all(lists):
                return None
            chosen = [0] * len(slots)

            def total() -> float:
                return sum((lists[n][chosen[n]][0][-1] if lists[n][chosen[n]][0][-1] != float("inf") else 0) * slots[n]["qty"] for n in range(len(slots)))

            while budget and total() > budget:
                saves = [((lists[n][chosen[n]][0][-1] - lists[n][c][0][-1]) * slots[n]["qty"], n, c) for n in range(len(slots))
                         for c in range(chosen[n] + 1, len(lists[n])) if lists[n][c][0][0] == lists[n][chosen[n]][0][0]]
                saves = [s for s in saves if s[0] > 0]
                if not saves:
                    return None
                _, n, c = max(saves)
                chosen[n] = c
            out: dict[str, int] = {}
            for n, slot in enumerate(slots):
                ident = lists[n][chosen[n]][1]
                out[ident] = out.get(ident, 0) + slot["qty"]
            return out

        searched_more = 0
        said: list[str] = []
        for _ in range(7):
            if S["finished"] or S["dead"]:
                return
            if S["pending"]:
                lines, S["pending"] = S["pending"], []
                if cart_view_tool and limit - S["step"] > 3:
                    view = act(cart_view_tool, {})
                    for row in rows_of((view or {}).get("observation")):
                        ident = ident_of(row)
                        price = number_of(row, r"price|cost")
                        if ident and price is not None:
                            S["price_now"][ident] = price
                cart = [{"id": i, "qty": q, "price": price_of(i), "title": next((v for k, v in (S["rec"].get(i) or {}).items() if re.search(r"title|name|headline", k, re.I)), "")}
                        for i, q in S["cart"].items()]
                products = "\n".join(f"{n}. {s['noun']}: " + "; ".join(f"{r.get('prop')} {r.get('value')}" for r in reqs_of(s)) for n, s in enumerate(slots, 1))
                out = llm(LONG_EVENT_PROMPT, f"PRODUCTS:\n{products}\n\nCURRENT BUDGET: {budget}\nCART: {_compact(cart)}\n\nSAID:\n" + "\n".join(lines),
                          LONG_MODELS, 2000) or {}
                if isinstance(out.get("budget"), (int, float)) and not isinstance(out["budget"], bool) and out["budget"] > 0:
                    # the new limit is the number the shopper wrote: a reading scaled to another unit is put back to it
                    written = [float(n.replace(",", "")) for n in re.findall(r"\d[\d,]*(?:\.\d+)?", " ".join(lines))]
                    budget = out["budget"]
                    if written and budget not in written:
                        budget = next((n for n in written for f in (100, 0.01) if abs(n - budget * f) < 0.5), budget)
                for r in out.get("changes") or []:
                    if isinstance(r, dict) and isinstance(r.get("slot"), int) and 1 <= r["slot"] <= len(slots) and (str(r.get("value") or "").strip() or r.get("op") == "drop"):
                        slots[r["slot"] - 1]["changed"].append(r)
                for ident in out.get("gone") or []:
                    if isinstance(ident, str):
                        S["gone"].add(ident)
                if S["msgs_used"] < msg_cap:
                    text = str(out.get("reply") or "").strip() or "Thanks for letting me know, I am updating your order to match."
                    for ident in S["rec"]:
                        text = text.replace(ident, "")
                    if act(message_tool, {message_arg: text}) is not None:
                        S["msgs_used"] += 1
                for slot in slots:
                    judge(slot)
                end_reserve = 3 + len(slots)
                work(8)
            trace.append({"kind": "long_state", "step": S["step"], "free": free(), "seconds": round(time_left(), 1), "slots": [{"pick": pick(sl), "opened": len(sl["opened"]), "fresh": len(fresh_of(sl)), "more": needs_more(sl), "pages": [q.get("next") for q in sl["queries"]], "verdicts": {i: (S["verdict"].get(i) or {}).get("checks") for i in sl["opened"][:12]}} for sl in slots]})
            picks = basket()
            if picks is None:
                trace.append({"kind": "long_no_basket", "budget": budget})
                # steps and time left: keep looking before giving up
                if searched_more < 2 and any(needs_more(sl) for sl in slots) and free() >= 2 and time_left() > 75:
                    searched_more += 1
                    work(10)
                    continue
                # a decline can be refused by news from the shopper, which may open a basket again
                reply = act(decline_tool, finish_args(decline_tool, False))
                if S["finished"] or reply is None:
                    return
                said = []
                obs = reply.get("observation")
                _sentences(obs if isinstance(obs, list) else list(obs.values()) if isinstance(obs, dict) else [obs], said)
                said = [line for line in said if line not in S["heard"]]
                S["heard"].update(said)
                S["pending"] += said
                if not S["pending"]:
                    return
                continue
            # fill the cart, the way this store's cart works
            current = dict(S["cart"])
            if cart_mode == "replace":
                if current != picks:
                    reply = act(cart_tool, {cart_lines: [{line_id: i, **({line_qty: q} if line_qty else {})} for i, q in picks.items()]})
                    if refused((reply or {}).get("observation")):
                        trace.append({"kind": "long_cart_refused", "picks": picks})
                        break
            else:
                for ident, q in current.items():
                    if picks.get(ident) != q:
                        if cart_mode == "set":
                            act(cart_tool, {cart_id: ident, cart_qty: 0})
                        else:
                            act(cart_tool, {cart_change: remove_value, cart_id: ident, **({cart_qty: q} if cart_qty else {})})
                for ident, q in list(picks.items()):
                    if current.get(ident) != q:
                        args = {cart_id: ident, **({cart_qty: q} if cart_qty else {})}
                        if cart_mode == "add":
                            args[cart_change] = add_value
                        if refused((act(cart_tool, args) or {}).get("observation")):
                            S["gone"].add(ident)        # the store refused this listing: the next pick replaces it
                            picks.pop(ident, None)
            S["cart"] = dict(picks)
            if S["pending"] or basket() != picks:
                continue
            reply = act(order_tool, finish_args(order_tool, True))
            if S["finished"] or reply is None:
                return
            obs = reply.get("observation")
            said = []
            _sentences(obs if isinstance(obs, list) else list(obs.values()) if isinstance(obs, dict) else [obs], said)
            said = [line for line in said if line not in S["heard"]]
            S["heard"].update(said)
            if not S["pending"] and not said:
                said = ["The order was not accepted: " + _compact(obs)[:300]]
            S["pending"] += said
        if S["finished"] or S["dead"]:
            return
        if S["cart"] and basket() == S["cart"]:
            act(order_tool, finish_args(order_tool, True))
        if not S["finished"]:
            act(decline_tool, finish_args(decline_tool, False))

    try:
        run()
    except Exception as exc:
        trace.append({"kind": "crash", "error": f"{type(exc).__name__}: {exc}"[:500]})
        try:
            if not S["finished"] and not S["dead"]:
                act(order_tool, finish_args(order_tool, True)) if S["cart"] else act(decline_tool, finish_args(decline_tool, False))
        except Exception as again:
            trace.append({"kind": "crash", "error": f"{type(again).__name__}: {again}"[:500]})
    trace.append({"kind": "end", "finished": S["finished"], "steps": S["step"], "cost": round(S["cost"], 5), "llm_calls": S["llm_calls"], "code_verdicts": S["code_verdicts"], "seconds": round(time.monotonic() - started, 1)})
    return trace


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        if int(problem_data["environment"]["policy_view"].get("max_steps") or 0) != LONG_STEPS:
            trace = run_long(problem_data)
            if trace is not None:
                return trace
    except Exception:
        pass
    episode = Episode(problem_data)
    try:
        return episode.run()
    except Exception as exc:
        episode.trace.append({"kind": "crash", "error": f"{type(exc).__name__}: {exc}"[:500]})
        try:
            if not episode.finished and not episode.dead:
                episode.force_finish()
        except Exception as again:
            episode.trace.append({"kind": "crash", "error": f"{type(again).__name__}: {again}"[:500]})
        return episode.trace


__all__ = ["agent_main"]
