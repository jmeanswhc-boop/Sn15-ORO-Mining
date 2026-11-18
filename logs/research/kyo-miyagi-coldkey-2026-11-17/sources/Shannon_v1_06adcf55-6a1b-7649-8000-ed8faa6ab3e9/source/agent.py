"""Shopping agent for runtime-4 worlds: a model works the store's tools turn by turn, while the code reads the
request once, keeps the requirement ledger, plans the questions, checks every pick before it is carted or ordered,
audits a decline, and closes the episode when steps or time run short. No clock is read: waiting is counted in
fixed slices."""
from __future__ import annotations

import json
import re
import threading
from contextlib import contextmanager
from typing import Any, Iterator

from src.agent.proxy_client import ProxyClient

# ---------------------------------------------------------------------------------------------------------- settings
TURN_MODEL = "anthropic/claude-haiku-5.5"# drives the tool turns
FALLBACK_MODEL = "openai/gpt-6-luna"# takes over when another model gives nothing
INTAKE_MODEL = "anthropic/claude-sonnet-5.5"# reads the request and the tool list once
JUDGE_MODEL = "anthropic/claude-sonnet-5.5"# verifies picks, audits declines
EAR_MODEL = "anthropic/claude-haiku-5.5"# reads the shopper's lines
HEAVY_MODEL = "anthropic/claude-sonnet-5.5"# plans the questions and plays the turns that decide the search or the pick
# rough USD per token (fresh input, cached input, output), used only when a reply carries no cost figure
TOKEN_PRICES = {"anthropic/claude-haiku-5.5": (1.5e-7, 1.5e-8, 8e-7), "anthropic/claude-sonnet-5.5": (2e-6, 2e-7, 1e-5),
         "openai/gpt-6-luna": (1.5e-7, 1.5e-8, 8e-7)}
SPEND_CEILING = 0.05              # USD ceiling for one episode, all model calls counted
SPEND_WIND_DOWN = 0.065           # above this spend the progress line asks to finish
SPEND_STOP = 0.085                # above this spend the episode is finished from code
SEARCH_FLOOD_AT = 6               # searches with no listing opened before a new search is held once
JUDGE_RESERVE = 0.012             # kept free under the ceiling for the checks still to come
CACHE_HIT_SHARE = 0.65            # share of a long prompt expected to be served from the model's cache
HEAVY_OUTPUT_GUESS = 700          # output tokens assumed for a strong call
HEAVY_TURNS = 0                   # strong turns at most per episode (0: every turn stays on TURN_MODEL; measured runs showed no gain for the cost)

TIME_BUDGET = 265.0               # the share of the network's 300 s this agent plans to use
WIND_DOWN = 70.0                  # below this, only finishing work is done
HARD_STOP = 28.0                  # below this, the episode is closed at once
TICK = 0.25                       # waiting on a call is counted in slices of this length
OBSERVATION_LIMIT = 16000         # longest observation text handed to the model
MAX_TURNS = 80                    # model turns at most
STATUS_MARK = "[progress]"        # opens the status line appended to tool results

# ---------------------------------------------------------------------------------------------------------- briefs
TURN_GUIDE = """You act as a careful shopping assistant for one shopper inside a simulated store, and you can act only through the tools you are given. Every task is scored 1 or 0, and it earns 1 only if your finish is correct in all respects: the item you order satisfies every requirement the shopper stated, every hidden need the shopper reveals when asked, and every change the shopper makes during your work; the quantity and the budget are right; when the request asks for the cheapest (or the best), that is what you picked; and you followed every rule the request sets out. One request may want several different products in a single order; in that case each line has to be correct. If no item in the store satisfies all of it, the correct finish is a decline carried out exactly as the request describes.

ABOUT THE STORE
- Each tool call costs one step, including failed calls and calls that return errors, and the steps are limited. Thinking costs nothing; steps do.
- Calls whose results do not depend on one another may go in the same reply (opening the details of several listings at once, for example); they run in the order given. An order or a decline is never sent together with other calls.
- Each store gives its tools, arguments and result fields its own names. Read the tool descriptions and JSON schemas closely and use the argument names and allowed values exactly as they are defined. From the first results, figure out which tool shows which facts (price, stock, full properties).
- Early on, find the category labels this store uses for the requested kind of product (the listings show them). When similar labels sit side by side, the correct label is the one that matches the request's wording taken as a whole; search and buy within it.
- Search tools can be restricted: capped rows or pages, sorting by id rather than relevance or price, every word required, filters that work in only one mode. Put the required values into the search words and drop the product kind when a category filter already handles it; keep the words few, because long phrases may match nothing. Compare the reported total with the rows you can actually see, and keep narrowing (sharper words, category or price filters, later pages) until the whole relevant set has gone past you. Results can also contain listings from other categories, even with a category filter on; keep only the rows of the requested kind.
- To reach a later page, send the exact token or value the previous result supplied for the next page (a cursor, bookmark or offset field in that result), with the other arguments unchanged. Never make one up: an invented page token returns nothing, and that tells you nothing about whether further listings exist.
- When one tool shows several items in full in a single call, prefer it to opening details one by one: it costs a single step for all of them. Tracing values from one listing to another can take several hops, so spare steps wherever possible.
- Read the description of the cart tool: some cart tools add or remove one item per call, while others overwrite the whole cart with the lines you send, so every line you wish to keep must be included in each such call.

THE SHOPPER AND HIDDEN NEEDS
- The shopper has needs that the request does not spell out. A need is revealed only when a message asks about that specific property, and only the FIRST thing a message asks is answered. So every message should ask about ONE property only, as a short, plain, friendly question to the shopper that names the property (one or two example values are fine). Give no instructions, mention no lists, labels, positions or rules, and never combine two questions.
- When the shopper replies that they do not mind, have no preference, or that anything is fine, the property stays open. When the reply states a preference or a limit, it becomes a hard requirement from that point, exactly as phrased (numeric limits included).
- If a planner is active, your questions are chosen and sent for you once the details of two or three listings of the requested kind have been opened, so open those in your first moves. The shopper's replies then show up in the conversation along with their meaning; treat every revealed need as binding from then on and search again with it added to the stated values.
- If you must ask by yourself: first open the details of two or three listings, then ask about what the story hints at (who the item is meant for, how it should look, where or how it will be used), and after that about properties the listings carry whose values vary between candidates. Never ask about something the request already settles, and never exceed the message budget.
- Ask about what the product itself is like. Questions about who makes or sells it (brand, maker, shop) are almost never answered. If the shopper says they did not understand a question, do not repeat it in other words; turn to a different property.

RULES OF PROCEDURE
- A request can lay down rules for how you work: how many listings to view or compare before deciding, what to check, what to tell the shopper, what a remark has to say, how to decline. These are graded just like requirements, so carry out each one literally and completely. A rule to view or open N listings means N listings that satisfy everything, each opened through the details tool.
- If the shopper asks you a question (to confirm a change, for example), it gets answered for you while messages are left; the change itself binds you either way.
- Things the shopper would merely like ("ideally", "would be nice", "if possible") are not requirements: they can only settle a tie between items that already satisfy everything, and they never override a cheapest rule.
- If the request leaves open something only the shopper can answer (no price limit at all, for instance, or two wishes that might not both be achievable so the shopper has to rank them), ask about it before you choose.

CHANGES
- During your work the shopper may change things: a new budget (higher or lower), a requirement removed, replaced or added, a price rising, an item selling out. One shopper line can hold several changes. Such lines can appear inside any tool result (an extra field containing sentences) or arrive with an order attempt. Each change binds from the moment it shows up: check your pick and your backups against it once more, and when the request wants the cheapest or best, choose again from what is now acceptable.
- An order attempt can be REFUSED. Its result then contains a "refusal" field, even when the inner result appears successful. The episode goes on and the cart stays as it was. Read the refusal, apply every change it mentions, remove from the cart anything that is no longer acceptable (a sold-out item, for example), add the next verified fit, check its stock and price again, and order again.
- Items can also sell out or change price without warning. So check stock and price again right before every order attempt; as soon as an item's current price times the quantity exceeds the current budget, it no longer fits.

JUDGING A LISTING (the score depends on this)
- A requirement holds only if the listing's own property for it shows that value. Property keys are written in many ways (spaces, dots, underscores, prefixes) and one property may appear under more than one key; any key that names that property counts. Values are compared by meaning: one item of a comma-separated list is enough; singular and plural are the same; close synonyms count; case is irrelevant. A word that merely begins the same way does not count, nor does a broader or different class (girls does not mean women, plain unisex does not mean female).
- Use the title or description only when the listing has no property key at all for that requirement. When the listing's own property shows something different, the listing fails even if the title or description shows the value.
- A value kept under a different, related property does not suffice (a length field cannot meet a style requirement; the lining's material cannot meet the material).
- Numbers: convert the units, then compare them as numbers; the limit itself is allowed; a range counts only if every part of it satisfies the limit.
- Respect exclusions ("leave out anything with X") and closed lists ("A or B, nothing else").
- A value can be defined through another listing. The request may link a property to the value shown on another product's page (described by its seller or brand, its category and some properties; if that page now points to another listing, follow it), and a listing's own notes may state that one of its properties comes from another listing rather than from itself. In both cases locate that other listing, open it, read the value, and use that value. The one who sells a listing (its shop or seller) may be different from its brand.
- The listing's category covers the requested kind of product; never demand that some property repeat the category name. But when the request links a value to a named property ("where the style is dresser"), the listing's own key for that property has to show it, even if the value repeats the category name; the category alone does not meet it, and a title naming another variant (a chest of drawers is not a dresser) fails. Only an item of the requested kind is acceptable: where the store tags listings with categories, it has to be in the category whose label matches the request's wording taken as a whole (asking for "A or B" points to a category named "A & B", not to a narrower neighbour named "B"). Stock is needed whenever the request wants things available, and price times quantity has to stay within the budget (equal is allowed).
- A property that lists several options with the required one among them (a comma list) is simply a match, not mixed evidence. Evidence is mixed only when the listing's own keys for that property give values that cannot both be true. Only in that case, and only when no cheapest or best rule decides, prefer a candidate with clean evidence.

CHEAPEST OR BEST
- When the request wants the cheapest, the order must be the lowest-priced item among ALL items satisfying every requirement; having found one fitting item is not enough. Use price filters when a tool offers them, keep reducing the price limit until every remaining row is visible, sort the visible candidates by price yourself, and verify them from the lowest price up; the first one that fits completely, with every cheaper one verified and rejected, is the answer. Keep the next fitting ones as backups.
- When the request wants the best by some measure (most, largest, longest, highest and so forth), compare ALL fitting items on that measure the same way and choose the extreme. A listing that gives no value for that measure cannot win.

SEVERAL PRODUCTS
- When the request wants several different products, each is a separate line with its own quantity and its own requirements, and the budget applies to the whole order: the sum of price times quantity across all lines. A cheapest rule then applies to that total, and the total is lowest when each line takes its own cheapest fully fitting item (price times that line's quantity), so search for the cheapest fit for each line on its own. Find and verify a fit for each line, keep backups for each, and place all lines in the cart before ordering.

FINISHING UP
- Maintain a ranked list of verified fits with at least one backup, since your first pick might sell out at checkout.
- Before every order attempt: the cart contains exactly the lines the request asks for (usually one: the chosen item at the required quantity) and nothing else (remove any other line); after your final cart change you have read each chosen item's stock and price again, every one is in stock and the total is within budget.
- Decline only when a thorough search turns up nothing that satisfies every requirement. Follow the request's rule for declining exactly. If the request says no substitutes may be offered, the decline remark must not name, describe or refer to any listing: no ids, no titles, no prices.
- Plan your steps: make the first order attempt while seven or more steps are left, so that a refusal can still be handled. Once 7 steps are left, only cart, stock-check and finishing tools remain available. Decide once: do not swap the cart back and forth, because every cart change costs a step.
- Every reply must include at least one tool call. Keep the reply text brief: your requirements as they currently stand, the verified fits with price and stock, and your next move."""

INTAKE_GUIDE = """Read the shopper's request and the store's tool list shown below. Reply with one compact JSON object (no indentation) and nothing more, containing these keys:

"product": the kind of product requested, in the request's own words.
"quantity": how many units of a single listing to order (an integer; 1 if the request does not want several). Something normally sold as one item, such as a pair of shoes or a set, is quantity 1.
"lines": when the request wants more than one kind of product, a list with one object for each kind: {"product": ..., "quantity": integer, "requirements": ["property: constraint", ...]}; otherwise an empty list.
"requirements": a list of strings, one per firm requirement the request places on the product (when there are several products, all of them, each beginning with its product, e.g. "<product> — property: constraint"), each written "property: constraint" using the shopper's own value, limit, exclusion or closed list. A requirement is a property of the product that a listing can itself state (colour, size, material, a rating and so on); delivery or arrival dates, how the item will suit the shopper's own devices, rooms or plans, and other circumstances of the shopper are story, not requirements. Omit price, stock, quantity, budgets and mere wishes, and do not repeat the product kind or category as a requirement: words that are part of the category's name (a season, an occasion or an audience named in the category) are handled by the category and are not a separate property requirement.
"references": every value the request defines through another listing ("the colour shown on the page for ..."), each as "property: same as <that listing as the request describes it>"; these also belong in requirements. Empty if there are none.
"price_cap": the highest total price (of the whole order) as a number, exactly as the request writes it and in the request's own unit (never convert: a limit of "250 points" is 250, never a converted figure), or null if the request gives none.
"currency": the price unit the request uses, word for word, or null.
"in_stock_only": true if the request wants only items actually available or in stock.
"priority": "cheapest" only if the request explicitly asks for the cheapest or lowest-priced fitting option; "best" only if it explicitly asks for the best by some measure; otherwise "none". Saying that money is tight or that every penny counts is not a request for the cheapest; it only explains the price cap.
"priority_detail": for "best", the measure that decides what is best; otherwise "".
"message_budget": how many messages the shopper allows you to send, as an integer, or null if the request does not say.
"step_budget": how many moves or steps the request permits, as an integer, or null if it does not say.
"rules": every procedural rule the request sets for the assistant (how many listings to look at, what to check or tell, how to decline, what not to do, what to include), quoted word for word. What the shopper says they will do on their own is story, not a rule.
"soft_wants": what the shopper would like without strictly needing it (phrased as ideally, preferably, would be nice, if possible). Keep these out of "requirements".
"ask_first": what the agent can learn only from the shopper and should ask before choosing: for example the budget if no price limit is given, which of two wishes matters more when the request says both may not be possible (then also put both wishes into soft_wants, not into requirements), or anything the request says to check with the shopper. Empty if there is nothing.
Every sentence of the request must end up somewhere: a requirement, the priority, a soft want, a rule, a budget, a hint, or plain story. If you are unsure whether a sentence is a rule, put it under "rules".
"decline_no_names": true if the request's rule for declining forbids offering, naming or suggesting substitutes or alternatives; false if it asks you to name something; null if the request has no such rule.
"cart_mode": "replace" if the cart tool sets the whole cart to the lines it receives, otherwise "add".
"roles": an object naming the tool for each role, using the exact tool names from the list, or null where no tool fits:
  "search", "details", "view", "message", "cart", "cart_view", "order", "decline".
  "message" is the tool that writes to the shopper. "cart" is the tool that adds items to or removes them from the cart. "order" is the tool that buys or checks out. "decline" is the tool that ends the task without a purchase.
"""

EAR_GUIDE = """You help a shopping assistant make sense of something a shopper has just said. Reply with one JSON object and nothing more, containing these keys:
"meaning": one short sentence saying what the shopper means.
"needs": a list of new firm requirements on the product, each written "property: constraint" with the exact value, limit, exclusion or closed list the shopper gave.
"no_preference": a list of properties the shopper said they do not mind about.
"dropped": a list of earlier requirements the shopper no longer needs.
"new_budget": the new highest total price as a number, or null.
"unavailable": true if the shopper says an item can no longer be bought (sold out, gone, out of stock, button disabled), otherwise false.
"requirements_now": the complete list of firm requirements on the product as they stand after everything the shopper has said, rebuilt from the request, the earlier shopper lines and this line, each as "property: constraint". Keep every earlier requirement unless the shopper dropped or replaced it; for a replaced value keep only the newest; add every newly revealed need. Omit price, stock, quantity and the product category, and anything that is not a property a listing can state (delivery dates, the shopper's own devices, rooms or plans). When the request wants several products, begin each with its product as the earlier requirements do.
"not_understood": true if the shopper only says they did not understand the assistant's message, otherwise false.
"reply": if the shopper asks the assistant a question, a short plain answer to send back that addresses it and names the change itself (for example confirming the new value); otherwise "".
One line may contain several changes: report every one of them. A requirement replaced by another is both dropped (the old one) and a new need (the new one). When the shopper says which of two wishes matters more, the preferred wish is a need. Any budget the shopper gives is new_budget. A mere wish ("ideally", "would be nice") is not a need. Read the shopper literally and in good faith: homely, figurative or indirect wording that names a value means they firmly want that value; treat a value as excluded only when the shopper clearly says to avoid it. A number paired with a comparison (at most, at least, under, over, no more than) is a limit and keeps its unit. A new budget replaces the old one whether it is higher or lower, even if the shopper calls it good news. When the shopper only says they do not mind, have no preference, or that anything is fine, put the property that was asked about into no_preference. When the shopper only says they did not understand the question, nothing has changed: leave every list empty and say so in meaning."""

JUDGE_GUIDE = """You are the final check before a shopping assistant orders. The task earns 1 only if the ordered item satisfies every requirement below. Judge the item from the records shown and from nothing else, reading them as follows:
- A requirement holds only if the item's own property for it shows the value. Property keys are written in many ways (spaces, dots, underscores, prefixes) and one property may appear under more than one key; any key that names that property counts. Values are compared by meaning: one item of a comma list is enough, singular/plural and close synonyms count, case is irrelevant. A word that only looks alike, or a broader or different class, does not count.
- The title or description counts only if the item has no property key for that requirement. When its own property shows something different, it fails even if the title shows the value.
- A value kept under a different, related property does not suffice.
- Numbers: convert the units, then compare; the limit itself is allowed; a range counts only if every part of it satisfies the limit.
- Exclusions and closed lists must be respected.
- Only an item of the requested kind passes; where listings have a category label, it has to be the category whose label matches the request's wording taken as a whole, not a narrower or neighbouring one.
- The category decides only the kind of product. A requirement written as "property: value" has to be shown by the item's own key for that property, even when the value repeats words of the category name; the category label never meets it. If the item lacks such a key, only a title or description that clearly states that exact value counts.
- If the item's own property for a requirement shows a different value, the requirement fails, whatever the title may hint.
- Price times quantity has to stay within the budget (equal is allowed). If the records show the item out of stock, it fails.
- When the request wants several products, judge the item only against the requirements of its own product; the budget then covers the sum across all cart lines, so judge the price only when the cart view shows the whole order.
- When a requirement takes its value from another listing, or the item's own notes say a property comes from another listing, use that other listing's value from the records of other listings shown below; if no record shows it, that requirement is not shown.
- A mere wish ("ideally", "would be nice") marked as not required is not a requirement.
Prices in the records are in the store's own unit, the same unit as the budget figure in the requirements: compare the numbers directly, never rescale them. If records conflict (on price or stock, for instance), the most recent record wins.
Reply with one JSON object and nothing more: {"item": the item's id, "fits": true or false, "problems": [short strings, each naming a requirement that is not met or not shown]}. Give fits=false only for a requirement the records clearly fail, or one that no record addresses at all; do not fail an item because no record shows its stock or price."""

PLAN_GUIDE = """You plan the questions a shopping assistant puts to a shopper. The shopper has hidden needs: each is the value of one property of the product, disclosed only when a message asks about exactly that property; only the first thing a message asks gets answered, and questions about anything else get a polite "don't mind". Pick {count} different properties (fewer only when there are not that many suitable ones), the most promising first; an unused message is a hidden need never found:
0. first, anything listed below as MUST BE LEARNED FROM THE SHOPPER that has not been asked yet; ask it plainly. For the budget, ask the most they would like to spend. For a choice between two wishes, name both values together with their properties, for example "If one listing can't offer both, would you rather have the <value> <property> or the <value> <property>?";
1. after that, properties that the listings' details below actually carry, favouring (a) a property the request's story hints at indirectly (who the item is for, or who will wear or use it, how it should look in photos or to others, where or how it will be used), and (b) properties describing what the product is like whose values vary across the listings (who it is made for, pattern, shape, type or style variant, closure, length, function) over logistics (warranty, condition, pack size, variant or size lists, years).
Never pick a property the request already settles (its stated requirements, the price, stock, quantity), never ask again about a property already asked, never spend two questions on near-duplicate properties (two kinds of style, two spellings of the same property), and skip properties no listing carries. Never ask who makes or sells the product (brand, maker, shop): such questions are almost never answered. If the shopper did not understand an earlier question, do not ask about that property again. When the request wants several products, each may have hidden needs of its own: spread the questions across the products. Name properties the way the listings name them. Write each question as one short, plain, friendly sentence to the shopper that names that single property in everyday words and may offer one or two example values from the listings (never brand or shop names). No instructions, no lists, no second question.
Reply with one JSON object and nothing more: {{"questions": [{{"property": "...", "question": "..."}}, ...]}}."""

REVIEW_GUIDE = """A shopping assistant is about to tell the shopper that nothing in the store fits. Work through the listing records below and test them against the requirements, reading them as follows: a requirement holds when the listing's own property for it shows the value (any spelling of the key, one item of a comma list, singular/plural and close synonyms all count; a broader or different class does not); the title or description counts only if the listing has no key for that property; numbers are compared after converting units, and the limit itself is allowed; the price has to be within the budget (for several products, the whole order), and an item shown out of stock does not fit; a value that the request or a listing's notes take from another listing is that other listing's value.
Reply with one JSON object and nothing more: {"possible_fits": [ids of listings that appear to satisfy every requirement, best first, 5 at most], "reason": "one sentence"}."""

NOTHING_FITS = "Sorry, nothing in the store meets all of your requirements, so I am not ordering anything."

_MODELS = ProxyClient(timeout=70, max_retries=1, retry_delay=1)
_SHOP = ProxyClient(timeout=95, max_retries=2, retry_delay=1)# ---------------------------------------------------------------------------------------------------------- json and schemas
ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-:.]{3,63}$")


def clip(value: Any, limit: int, depth: int = 0) -> Any:
    """Shorten overlong text anywhere inside nested lists and dicts; anything nested past level 8 becomes None."""
    if depth > 8:
        return None
    below = depth + 1
    if isinstance(value, dict):
        return dict((name, clip(inner, limit, below)) for name, inner in value.items())
    if isinstance(value, list):
        return list(map(lambda inner: clip(inner, limit, below), value))
    if isinstance(value, str) and len(value) > limit:
        return f"{value[:limit]}..."
    return value


def flat_json(value: Any) -> str:
    """Compact single-line JSON text (str() as the fallback), trimmed once it passes the observation limit."""
    try:
        rendered = json.dumps(value, separators=(",", ":"), ensure_ascii=False)
    except (ValueError, TypeError):
        rendered = str(value)
    if len(rendered) > OBSERVATION_LIMIT:
        rendered = rendered[:OBSERVATION_LIMIT] + "...(cut)"
    return rendered


def _pa_unfence(raw: str) -> str:
    """Drop a surrounding markdown code fence (with its language tag) from model output."""
    body = raw.strip()
    if not body.startswith("```"):
        return body
    body = re.sub(r"^```[A-Za-z]*", "", body).strip()
    return body[:-3].strip() if body.endswith("```") else body


def _pa_decode(blob: str) -> tuple[bool, Any]:
    """(parsed, value) for a JSON text; parsed is False when the text is not valid JSON."""
    try:
        return True, json.loads(blob)
    except ValueError:
        return False, None


def object_in(text: Any) -> dict[str, Any] | None:
    """Pull a JSON object out of model output: tried as a whole first, then the span from the first '{' to the last '}'."""
    if not isinstance(text, str):
        return None
    body = _pa_unfence(text)
    parsed_ok, parsed = _pa_decode(body)
    if not parsed_ok:
        span = _between_braces(body)
        parsed = None if span is None else _pa_decode(span)[1]
    return parsed if isinstance(parsed, dict) else None


def _between_braces(text: str) -> str | None:
    opening = text.find("{")
    closing = text.rfind("}")
    if opening < 0 or closing <= opening:
        return None
    return text[opening:closing + 1]


_TYPE_CHECKS = {
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "null": lambda v: v is None,
}


def type_matches(value: Any, kind: str) -> bool:
    checker = _TYPE_CHECKS.get(kind)
    return checker is None or checker(value)


def declared_types(schema: dict[str, Any]) -> list[str]:
    raw = schema.get("type")
    if isinstance(raw, str):
        return [raw]
    return list(raw) if raw else []


def _pa_matches_any(value: Any, kinds: list[str]) -> bool:
    return any(type_matches(value, kind) for kind in kinds)


def _pa_convert(value: Any, kind: str) -> tuple[bool, Any]:
    """(changed, new value) for one lossless conversion of value into the given JSON type."""
    plain_number = isinstance(value, (int, float)) and not isinstance(value, bool)
    if kind == "integer":
        if isinstance(value, float) and value.is_integer():
            return True, int(value)
        if isinstance(value, str) and re.fullmatch(r"\s*-?\d+(\.0+)?\s*", value):
            return True, int(float(value))
    elif kind == "number":
        if isinstance(value, str) and re.fullmatch(r"\s*-?\d+(\.\d+)?\s*", value):
            amount = float(value)
            return True, (int(amount) if amount.is_integer() else amount)
    elif kind == "string":
        if plain_number:
            return True, str(value)
    elif kind == "boolean":
        if isinstance(value, str) and value.lower() in ("true", "false"):
            return True, value.lower() == "true"
    return False, value


def nudge_type(value: Any, schema: dict[str, Any]) -> Any:
    """Move a value onto the schema's declared type only when nothing is lost (e.g. 5.0 to 5, "7" to 7, 3 to "3", "true" to True)."""
    kinds = declared_types(schema)
    if not kinds:
        return value
    if _pa_matches_any(value, kinds):
        whole_float = "integer" in kinds and isinstance(value, float) and value.is_integer()
        return int(value) if whole_float else value
    for kind in kinds:
        try:
            changed, converted = _pa_convert(value, kind)
        except (TypeError, ValueError):
            continue
        if changed:
            return converted
    return value


def _pa_enum_fix(value: Any, schema: dict[str, Any], where: str, problems: list[str]) -> Any:
    """Snap a string onto an enum option that differs only in letter case; note a problem when none matches."""
    options = schema["enum"]
    if value in options:
        return value
    if isinstance(value, str):
        wanted = value.strip().lower()
        for option in options:
            if isinstance(option, str) and option.lower() == wanted:
                value = option
                break
    if value not in options:
        problems.append(f"{where}: must be one of {options}")
    return value


def _pa_text_faults(value: str, schema: dict[str, Any], where: str) -> list[str]:
    found = []
    size = len(value)
    if "minLength" in schema and size < schema["minLength"]:
        found.append(f"{where}: too short, the minimum is {schema['minLength']} characters")
    if "maxLength" in schema and size > schema["maxLength"]:
        found.append(f"{where}: too long ({size} characters, the limit is {schema['maxLength']}); shorten it")
    return found


def _pa_fit_list(items: list[Any], schema: dict[str, Any], where: str, problems: list[str]) -> list[Any]:
    count = len(items)
    if "minItems" in schema and count < schema["minItems"]:
        problems.append(f"{where}: give at least {schema['minItems']} entries")
    if "maxItems" in schema and count > schema["maxItems"]:
        problems.append(f"{where}: no more than {schema['maxItems']} entries are accepted")
    inner = schema.get("items")
    if not isinstance(inner, dict):
        return items
    rebuilt = []
    for slot, entry in enumerate(items):
        fixed, extra = fit_schema(entry, inner, f"{where}[{slot}]")
        rebuilt.append(fixed)
        problems.extend(extra)
    return rebuilt


def _pa_fit_object(obj: dict[str, Any], schema: dict[str, Any], where: str, problems: list[str]) -> dict[str, Any]:
    props = schema.get("properties") or {}
    problems.extend(f"{where}.{key}: required but absent" for key in (schema.get("required") or []) if key not in obj)
    if schema.get("additionalProperties") is False:
        problems.extend(f"{where}.{key}: this tool has no such argument" for key in obj if key not in props)
    rebuilt = {}
    for key, entry in obj.items():
        if key in props:
            entry, extra = fit_schema(entry, props[key], f"{where}.{key}")
            problems.extend(extra)
        rebuilt[key] = entry
    return rebuilt


def fit_schema(value: Any, schema: dict[str, Any], where: str) -> tuple[Any, list[str]]:
    """Coerce a value toward a JSON schema (lossless type fixes, enum letter case, recursion into lists and
    objects) and return it with the problems that remain."""
    if not isinstance(schema, dict):
        return value, []
    value = nudge_type(value, schema)
    kinds = declared_types(schema)
    if kinds and not _pa_matches_any(value, kinds):
        return value, [f"{where}: expected type {'/'.join(kinds)}"]
    problems: list[str] = []
    if "enum" in schema:
        value = _pa_enum_fix(value, schema, where, problems)
    if "const" in schema and value != schema["const"]:
        problems.append(f"{where}: must equal {schema['const']!r}")
    if isinstance(value, str):
        problems.extend(_pa_text_faults(value, schema, where))
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        problems.extend(_range_issues(value, schema, where))
    if isinstance(value, list):
        value = _pa_fit_list(value, schema, where, problems)
    if isinstance(value, dict):
        value = _pa_fit_object(value, schema, where, problems)
    return value, problems


def _range_issues(value: float, schema: dict[str, Any], where: str) -> list[str]:
    found = []
    if "minimum" in schema and value < schema["minimum"]:
        found.append(f"{where}: below the lowest allowed value {schema['minimum']}")
    if "maximum" in schema and value > schema["maximum"]:
        found.append(f"{where}: above the highest allowed value {schema['maximum']}")
    floor = schema.get("exclusiveMinimum")
    if isinstance(floor, (int, float)) and value <= floor:
        found.append(f"{where}: must be greater than {floor}")
    ceiling = schema.get("exclusiveMaximum")
    if isinstance(ceiling, (int, float)) and value >= ceiling:
        found.append(f"{where}: must be less than {ceiling}")
    return found


def long_strings(value: Any, into: list[str], depth: int = 0) -> None:
    """Gather texts of at least four words; lists are searched (ten items per level) down to depth 4."""
    if depth > 4:
        return
    if isinstance(value, list):
        for item in value[:10]:
            long_strings(item, into, depth + 1)
    elif isinstance(value, str) and len(value.split()) >= 4:
        into.append(value)


def walk_dicts(value: Any, depth: int = 0) -> Iterator[dict[str, Any]]:
    """Yield each dict found in a nested value, outer ones before inner ones, to a depth of six."""
    if depth > 6:
        return
    if isinstance(value, dict):
        yield value
        children: Any = value.values()
    elif isinstance(value, list):
        children = value
    else:
        return
    for child in children:
        yield from walk_dicts(child, depth + 1)


def word_set(text: Any) -> set[str]:
    """Lower-cased tokens of the text, keeping digit runs and words longer than two letters."""
    tokens = re.findall(r"[a-z0-9]+", str(text or "").lower())
    return set(filter(lambda tok: len(tok) > 2 or tok.isdigit(), tokens))


def _pa_id_hit(key: Any, entry: Any) -> bool:
    return bool(isinstance(entry, str) and "id" in key.lower() and ID_PATTERN.match(entry))


def collect_ids(value: Any, into: set[str], depth: int = 0) -> None:
    """Add identifier-looking strings found under keys containing "id" (searched to depth six)."""
    if depth > 6:
        return
    if isinstance(value, list):
        for child in value:
            collect_ids(child, into, depth + 1)
        return
    if not isinstance(value, dict):
        return
    for key, entry in value.items():
        if _pa_id_hit(key, entry):
            into.add(entry)
            continue
        collect_ids(entry, into, depth + 1)


def is_text_slot(spec: Any) -> bool:
    if not isinstance(spec, dict):
        return False
    declared = spec.get("type")
    return declared == "string" or "string" in (declared or [])


def _pa_bare_id(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    core = value.strip()
    return bool(core) and " " not in core


def _pa_first_int(values: Any) -> Any:
    for candidate in values:
        if isinstance(candidate, int) and not isinstance(candidate, bool):
            return candidate
    return None


def _pa_list_lines(entries: list[Any]) -> list[tuple[str, Any]]:
    out: list[tuple[str, Any]] = []
    for entry in entries:
        if isinstance(entry, dict):
            ident = next(filter(_pa_bare_id, entry.values()), None)
            if ident:
                out.append((ident, _pa_first_int(entry.values())))
        elif _pa_bare_id(entry):
            out.append((entry, None))
    return out


def lines_named(args: dict[str, Any], fields: dict[str, Any]) -> list[tuple[str, Any]]:
    """The (id, quantity) lines a cart call refers to. Top-level single-token ids share the call's own quantity;
    each entry of a list argument gives its own line (quantity None when the entry has none)."""
    shared_qty = None
    for key, val in args.items():
        if "enum" in (fields.get(key) or {}):
            continue
        if isinstance(val, int) and not isinstance(val, bool):
            shared_qty = val
            break
    found: list[tuple[str, Any]] = []
    for key, val in args.items():
        slot = fields.get(key)
        if not isinstance(slot, dict):
            slot = {}
        if _pa_bare_id(val) and "enum" not in slot:
            found.append((val, shared_qty))
        elif isinstance(val, list):
            found.extend(_pa_list_lines(val))
    return found


def mode_key(args: dict[str, Any], fields: dict[str, Any]) -> str:
    """Text of the arguments whose schema has an enum; this separates e.g. adding from removing on one cart tool."""
    chosen = {}
    for key, val in args.items():
        slot = fields.get(key)
        if isinstance(slot, dict) and "enum" in slot:
            chosen[key] = val
    return flat_json(chosen)


def _pe_crash_entry(error: BaseException) -> dict[str, Any]:
    return {"kind": "crash", "error": f"{type(error).__name__}: {error}"[:500]}






# ---------------------------------------------------------------------------------------------------------- gatekeeper
class Gatekeeper:
    """Checks every call the model proposes before the store sees it (tool on offer, argument schema, repeated
    calls, invented page values, runs of searches, cart checks) and decides when an order or decline waits."""

    _PB_SKIP = "Not run (no step used)"

    def tools_now(self) -> list[dict[str, Any]]:
        roles = self.role_map
        buy, quit_, basket = roles.get("order"), roles.get("decline"), roles.get("cart")
        remaining = self.steps_remaining()
        allowed = self._pb_allowed_names(remaining, buy, quit_, basket)
        menu = self.tool_specs
        if allowed is not None and buy and basket:
            menu = [spec for spec in self.tool_specs if spec["function"]["name"] in allowed] or menu
        chat = roles.get("message")
        if not (self.planning and chat):
            return menu
        return [spec for spec in menu if spec["function"]["name"] != chat]

    def _pb_allowed_names(self, remaining: int, buy: Any, quit_: Any, basket: Any) -> set[Any] | None:
        """Which tool names stay on offer as the step budget shrinks (None: all of them)."""
        if remaining > 7:
            return None
        if remaining > 3:
            undecided = self.orders_tried == 0 or self.refused_count > 0
            return {buy, quit_, basket, self.role_map.get("cart_view"), self.role_map.get("view")} if undecided else None
        if remaining > 1:
            return {buy, quit_, basket}
        return {buy, quit_}

    def terminal_tools(self) -> set[str]:
        finishers = {self.role_map.get(role) for role in ("order", "decline")}
        finishers.discard(None)
        return finishers

    def _pb_parse_args(self, raw: Any) -> dict[str, Any] | None:
        if isinstance(raw, dict):
            return raw
        source = raw or ""
        return object_in(source or "{}") if source.strip() else {}

    def _pb_offer_refusal(self, name: Any, offered: set[str]) -> str:
        if name == self.role_map.get("message") and self.planning:
            reason = "the planner writes and sends your questions to the shopper after you have looked at the details of a few listings"
        else:
            reason = "not enough steps are left for it; the tools still usable now are " + ", ".join(sorted(offered))
        return f"{self._PB_SKIP}: {reason}."

    def _vet_call(self, call: dict[str, Any], calls: list[dict[str, Any]], offered: set[str], planned: bool, queued_messages: int) -> Any:
        """Gives back (name, args) when the call may go out; otherwise the note, costing no step, that says why not."""
        fn = call.get("function") or {}
        tool, payload = fn.get("name"), fn.get("arguments")
        skip = self._PB_SKIP
        finishers = self.terminal_tools()
        chat, basket = self.role_map.get("message"), self.role_map.get("cart")
        if tool not in self.arg_schemas:
            return f"{skip}: no tool named '{tool}' exists. Pick one of: {', '.join(sorted(offered))}."
        if tool not in offered and not planned:
            return self._pb_offer_refusal(tool, offered)
        parsed = self._pb_parse_args(payload)
        if parsed is None:
            return skip + ": the arguments could not be read as a JSON object."
        parsed, errors = fit_schema(parsed, self.arg_schemas[tool], "args")
        if errors:
            return skip + ": the arguments are wrong: " + "; ".join(errors[:6])
        if len(calls) > 1 and tool in finishers:
            return skip + ": send an order or a decline alone, once you have read the results of your other calls."
        cap = self.message_allowance
        if tool == chat and cap is not None and self.notes_sent + queued_messages >= cap:
            return skip + ": the message allowance is used up. Write nothing more to the shopper."
        invented = self.made_up_page(tool, parsed)
        problem = invented or self.flood_check(tool, parsed, planned)
        if problem:
            return skip + ": " + problem
        fingerprint = tool + flat_json(parsed)
        repeatable = tool in finishers or tool in (chat, basket)
        if not repeatable and fingerprint in self.last_calls:
            return skip + ": this exact call was just made and nothing has changed since; its result is shown above. Use other arguments or take another action."
        if tool != basket:
            return tool, parsed
        if self.steps_remaining() > 3:
            objection = self._pb_cart_objection(tool, parsed)
            if objection:
                return "Not run (no step used). " + objection + " If you are certain it fits, repeat the same call."
        concern = self.count_problem(tool, parsed)
        if concern and fingerprint not in self.count_warned:
            self.count_warned.add(fingerprint)
            return skip + ": " + concern + " If you are certain, repeat the same call."
        return tool, parsed

    def _pb_cart_objection(self, tool: str, parsed: dict[str, Any]) -> str:
        """Vets up to four not-yet-checked items a cart call names; the first objection found, or ''."""
        spec = self.arg_schemas[tool].get("properties") or {}
        same_mode = self.cart_replaces() or self.add_mode is None or mode_key(parsed, spec) == self.add_mode
        if not same_mode:
            return ""
        for item, _qty in lines_named(parsed, spec)[:4]:
            tag = f"{item}|{self.ledger_rev}"
            if tag in self.cart_vetted:
                continue
            self.cart_vetted.add(tag)
            opinion = self.vet_item(item, "ADD TO THE CART")
            if opinion:
                self.heavy_due = True
                return opinion
        return ""

    @staticmethod
    def _pb_scalars(mapping: dict[str, Any]) -> list[Any]:
        return [val for val in mapping.values() if isinstance(val, (str, int)) and not isinstance(val, bool)]

    def made_up_page(self, name: str, args: dict[str, Any]) -> str:
        """When a call copies an earlier call of the same tool plus one extra argument, it asks for a later page;
        that extra value has to appear in a result of the tool rather than be invented."""
        earlier_results = [result for tool, _a, result in self.call_log if tool == name]
        index = len(self.call_log)
        while index:
            index -= 1
            prior_tool, prior_args, result = self.call_log[index]
            if prior_tool != name or not isinstance(result, dict):
                continue
            extra = [key for key in args if key not in prior_args]
            if len(extra) != 1:
                continue
            if not all(args.get(key) == val for key, val in prior_args.items()):
                continue
            field = extra[0]
            candidate = args[field]
            as_text = str(candidate)
            if any(as_text in flat_json(r) for r in earlier_results):
                return ""
            pool = self._pb_scalars(result)
            for nested in result.values():
                if isinstance(nested, dict):
                    pool.extend(self._pb_scalars(nested))
            if candidate in pool or as_text in flat_json(result):
                return ""
            tokens = [v for v in pool if isinstance(v, str) and v.strip() and " " not in v and len(v) <= 40]
            tail = (f"That result returned these values: {', '.join(map(str, tokens[:6]))}. " if tokens
                    else "That result did not offer another page. ")
            return (f"The {field} value '{candidate}' is not taken from what that same call returned, so it cannot reach a later page. "
                    + tail + "Use the next-page value exactly as the result returned it.")
        return ""

    def flood_check(self, name: str, args: dict[str, Any], planned: bool) -> str:
        """A long run of searches with no listing ever opened holds the next fresh search once and names listings to open."""
        finder = self.role_map.get("search")
        if planned or name != finder:
            return ""
        if "flood" in self.holds or self.steps_remaining() <= 8:
            return ""
        earlier = [a for tool, a, _r in self.call_log if tool == finder]
        readers = (self.role_map.get("details"), self.role_map.get("view"))
        opened_any = any(tool in readers for tool, _a, _r in self.call_log)
        if opened_any or len(earlier) < SEARCH_FLOOD_AT:
            return ""
        for prev in earlier:
            if set(prev) < set(args) and all(args.get(k) == v for k, v in prev.items()):
                return ""   # only paging on through a search already made
        picks = self.lead_listings(4)
        if not picks:
            return ""
        self.holds.add("flood")
        return (f"There have been {len(earlier)} searches and not a single listing opened. Going through what the searches already "
                "showed tends to work better than more searching: open these listings first (several together when a tool shows "
                "several items): " + ", ".join(picks) + ". After that, search only for what they leave unanswered; to search "
                "regardless, send this call once more.")

    def _pb_hold(self, answers: list[tuple[str, str]], call_id: str, text: str) -> bool:
        answers.append((call_id, "Not run yet (no step used). " + text))
        return True

    def _hold_finish(self, queue: list[tuple[str, str, dict[str, Any]]], answers: list[tuple[str, str]]) -> bool:
        """First screening of an order or decline; returns True when the call is kept back on this pass."""
        call_id, name, args = queue[0]
        buying = name == self.role_map.get("order")
        quitting = name == self.role_map.get("decline")
        roomy = self.steps_remaining() > 6
        if buying and roomy:
            unopened = self.cheaper_unread()
            tag = f"cheap|{self.ledger_rev}|{len(self.seen_records)}|{self.cart_items}"
            if unopened and tag not in self.holds:
                self.holds.add(tag)
                return self._pb_hold(answers, call_id, "The request wants the cheapest fit, and these listings in the same category are priced below your pick yet were never opened: " + ", ".join(unopened) + ". Open them, cheapest first (several together when a tool shows several items), and order the first that meets everything; if none of them does, send this order once more.")
        if quitting and roomy:
            unopened = self.lead_listings(min(4, self.steps_remaining() - 4))
            tag = f"reach|{self.ledger_rev}"
            if unopened and tag not in self.holds:
                self.holds.add(tag)
                return self._pb_hold(answers, call_id, "Your searches for the required values turned up these listings and none has been opened yet: " + ", ".join(unopened) + ". Open them (several together when a tool shows several items) and buy one that meets everything; if none of them does, send the decline once more.")
        if buying:
            return self._pb_hold_order(call_id, answers)
        if quitting:
            return self._pb_hold_decline(queue, answers)
        return False

    def _pb_hold_order(self, call_id: str, answers: list[tuple[str, str]]) -> bool:
        if self.order_vetted or self.steps_remaining() <= 2 or self.seconds_left() <= WIND_DOWN:
            return False
        self.order_vetted = True
        finding = self.vet_cart()
        if finding:
            return self._pb_hold(answers, call_id, finding)
        return self._pb_hold(answers, call_id, "Run through this before you order: (1) the cart contains precisely the lines the request calls for (normally just one: the chosen item in the required quantity) and nothing more; (2) after your latest cart change you checked each item's stock and price again with a tool that shows them, and all are in stock with the order total inside the current budget; (3) every item's own properties satisfy each stated requirement, each need the shopper revealed and each announced change; (4) if the cheapest or best was requested, nothing better fits among everything you could find; (5) each procedural rule in the request has been done. Settle anything not yet checked now; otherwise call the order tool once more.")

    def _pb_hold_decline(self, queue: list[tuple[str, str, dict[str, Any]]], answers: list[tuple[str, str]]) -> bool:
        call_id, name, args = queue[0]
        first_look = not self.decline_vetted and self.steps_remaining() > 6 and self.seconds_left() > WIND_DOWN + 30
        if not first_look:
            queue[0] = (call_id, name, self.clean_remark(args, name))
            return False
        self.decline_vetted = True
        review = self.review_before_decline()
        if review:
            return self._pb_hold(answers, call_id, review)
        return self._pb_hold(answers, call_id, "Before you decline, be certain that you searched in more than one way (different words, filters, pages), that you looked at each candidate's own properties, and that every one fails a stated requirement, a revealed need, the budget or stock. Close synonyms and any spelling of a key count. If something fits, buy it; if nothing does, call the decline tool once more.")


# ---------------------------------------------------------------------------------------------------------- courier
class Courier:
    """Takes the calls that passed screening, ships them to the store in groups and turns each store reply into model-readable text."""

    def route_calls(self, calls: list[dict[str, Any]], planned: bool = False) -> list[tuple[str, str]]:
        replies: list[tuple[str, str]] = []
        menu = {spec["function"]["name"] for spec in self.tools_now()}
        finals = self.terminal_tools()
        pending: list[tuple[str, str, dict[str, Any]]] = []
        for item in calls:
            talk_so_far = sum(1 for entry in pending if entry[1] == self.role_map.get("message"))
            outcome = self._vet_call(item, calls, menu, planned, talk_so_far)
            if isinstance(outcome, str):
                replies.append((item["id"], outcome))
            else:
                pending.append((item["id"], outcome[0], outcome[1]))
        if pending:
            if pending[0][1] in finals:
                if self._hold_finish(pending, replies):
                    self.heavy_due = True
                    return replies
            elif finals:
                pending = self._pc_keep_last_step(pending, replies)
        cursor = 0
        while cursor < len(pending):
            self._send_part(pending[cursor:cursor + 16], replies)
            cursor += 16
        position_of = {}
        for index, item in enumerate(calls):
            position_of[item["id"]] = index
        return self._pc_in_call_order(replies, position_of)

    @staticmethod
    def _pc_in_call_order(replies: list[tuple[str, str]], position_of: dict[str, int]) -> list[tuple[str, str]]:
        replies.sort(key=lambda pair: position_of.get(pair[0], 0))
        return replies

    def _pc_keep_last_step(self, pending: list[tuple[str, str, dict[str, Any]]], replies: list[tuple[str, str]]) -> list[tuple[str, str, dict[str, Any]]]:
        """With a finishing tool on offer, the final step stays free for it: calls past that point are answered without running."""
        usable = max(0, self.steps_remaining() - 1)
        if usable >= len(pending):
            return pending
        replies.extend((entry[0], "Not run (no step used): only one step remains, and it is kept for an order or a decline.") for entry in pending[usable:])
        return pending[:usable]

    def _send_part(self, batch: list[tuple[str, str, dict[str, Any]]], answers: list[tuple[str, str]]) -> None:
        blocked = self.done or self.out_of_funds or self.steps_remaining() <= 0
        if blocked:
            answers.extend((entry[0], "Not run: no steps are left or the episode has already closed.") for entry in batch)
            return
        batch = batch[: self.steps_remaining()]
        actions = []
        for _, tool, payload in batch:
            actions.append({"name": tool, "args": payload})
        step_no = self.steps_used + 1
        reply = self.call_store(actions)
        self.log.append({"kind": "env", "step": step_no, "actions": actions, "reply": reply})
        if reply is None:
            answers.extend((entry[0], "The store could not be reached (network or server trouble). Send the call again.") for entry in batch)
            return
        results = reply.get("observations")
        if not isinstance(results, list):
            results = [reply.get("observation")]
        final_index = len(results) - 1
        for index, (call_id, tool, payload) in enumerate(batch):
            if index > final_index:
                answers.append((call_id, "Not run: the episode closed before this call was reached."))
            else:
                closing = index == final_index and bool(reply.get("finished"))
                answers.append((call_id, self._record_result(tool, payload, results[index], closing)))

    def _pc_tally(self, tool: str, payload: dict[str, Any], result: Any) -> None:
        """Update counters, the repeat memory and the stored records for one finished call."""
        roles = self.role_map
        speak, cart = roles.get("message"), roles.get("cart")
        finals = self.terminal_tools()
        if tool == speak:
            self.notes_sent += 1
        if tool == roles.get("order"):
            self.orders_tried += 1
        if tool == cart:
            self.cart_touched = True
            self.order_vetted = False
        resets_memory = tool in finals or tool in (cart, speak)
        self.last_calls = [] if resets_memory else (self.last_calls + [tool + flat_json(payload)])[-2:]
        self.seen_records.append(result)
        self.call_log.append((tool, dict(payload), result))
        if tool in (roles.get("details"), roles.get("view")):
            self.sheet_records.append(result)
        if tool == cart:
            self._track_cart(tool, payload)
        if tool == roles.get("cart_view"):
            self.cart_view_seen = result

    def _pc_shopper_lines(self, tool: str, payload: dict[str, Any], result: Any) -> tuple[list[str], str | None]:
        """Pull out what the shopper said in a result, plus the question that drew it (only for messages we sent)."""
        found: list[str] = []
        if tool != self.role_map.get("message"):
            if isinstance(result, dict):
                for field in result.values():
                    if isinstance(field, str):
                        if "refusal" in result and len(field.split()) >= 4:
                            found.append(field)
                    elif isinstance(field, list) and field and all(isinstance(piece, str) for piece in field):
                        long_strings(field, found)
            return found, None
        asked = " ".join(str(part) for part in payload.values() if isinstance(part, str))
        if "?" not in asked:
            return [], asked     # a statement drew only a reaction, which is no answer
        if isinstance(result, str):
            found.append(result)
        elif isinstance(result, dict) and "error" not in result:
            long_strings([field for field in result.values() if isinstance(field, (str, list))], found)
        return found, asked

    def _record_result(self, name: str, args: dict[str, Any], observation: Any, ended: bool = False) -> str:
        """Log one store result, interpret any shopper speech inside it and build the text the model will see."""
        self._pc_tally(name, args, observation)
        comment = self.annotate(name, observation)
        heard, asked = self._pc_shopper_lines(name, args, observation)
        if heard:
            gist = self.listen(heard, asked)
            if gist:
                comment = f"{comment} {gist}" if comment else gist
        body = flat_json(observation)
        if ended:
            body += " [the episode has ended]"
        return f"{body}\n[remark] {comment}" if comment else body

    def _track_cart(self, name: str, args: dict[str, Any]) -> None:
        props = self.arg_schemas[name].get("properties") or {}
        style = mode_key(args, props)
        if self.add_mode is None:
            self.add_mode = style
        entries = lines_named(args, props)
        if self.cart_replaces():
            self.cart_items = [pid for pid, _ in entries]
            self.cart_units = dict(entries)
        else:
            for pid, qty in entries:
                self.cart_modes[pid] = style
                kept = [other for other in self.cart_items if other != pid]
                if style == self.add_mode:
                    kept.append(pid)
                    self.cart_units[pid] = qty
                else:
                    self.cart_units.pop(pid, None)
                self.cart_items = kept
        self.cart_touched_at = self.steps_used

    def annotate(self, name: str, observation: Any) -> str:
        if isinstance(observation, list):
            collect_ids(observation, self.known_ids)
            return ""
        if not isinstance(observation, dict):
            return ""
        collect_ids(observation, self.known_ids)
        remarks: list[str] = []
        if observation.get("error") == "budget_exceeded":
            ceiling = self.message_allowance
            if ceiling is not None:
                self.message_allowance = min(ceiling, self.notes_sent)
            elif name == self.role_map.get("message"):
                self.message_allowance = self.notes_sent
            remarks.append("Nothing of that allowance is left; stop calling this tool.")
        if "refusal" in observation:
            self.refused_count += 1
            self.order_vetted = False
            self.heavy_due = True
            remarks.append("THE ORDER WAS REFUSED, yet the episode continues and the cart is unchanged. Make every change the refusal asks for, fix the cart (remove any line that can no longer be accepted, a sold-out item for example), check the stock and price of the new pick again, and then place the order once more.")
        elif name != self.role_map.get("message") and len(observation) >= 2:
            said: list[str] = []
            for field in observation.values():
                if isinstance(field, list) and field and all(isinstance(piece, str) for piece in field):
                    long_strings(field, said)
            if said:
                remarks.append("The shopper sent a message (a change) inside this result. Act on it: " + " | ".join(said[:4]))
        return " ".join(remarks)


# ---------------------------------------------------------------------------------------------------------- referee
class Referee:
    """Independent reviews: of a listing before it is carted or bought, of the cart's line counts, of a planned
    decline, plus scrubbing listing names out of a decline note."""

    def vet_item(self, item: str, purpose: str) -> str:
        # Too close to the end, or nothing on record about this listing: stay silent.
        if self.seconds_left() < WIND_DOWN + 20:
            return ""
        memo = f"{item}|{self.ledger_rev}"
        if memo in self.verdict_cache:
            return self.verdict_cache[memo]
        evidence = self.records_for(item)
        if not evidence:
            return ""
        reply = self.ask_json(self._pd_judge_prompt(item, purpose, evidence), 1200)
        self.log.append({"kind": "verify", "item": item, "result": reply})
        self.verdict_cache[memo] = outcome = self._pd_verdict_from(item, reply)
        return outcome

    def _pd_judge_prompt(self, item: str, purpose: str, evidence: list[str]) -> str:
        """Assemble the checker's prompt: guide, request, live needs, the listing's records, then optional extras."""
        parts = [JUDGE_GUIDE,
                 "\n\nWHAT THE SHOPPER ASKED FOR:\n", self.query_text(),
                 "\n\nNEEDS IN FORCE (stated, disclosed or changed):\n", self.needs_summary(),
                 f"\n\nLISTING TO {purpose}: {item}\nWHAT THE STORE HAS SHOWN ABOUT IT (earliest first):\n", "\n".join(evidence)]
        if self.cart_view_seen is not None:
            parts += ["\n\nLATEST LOOK AT THE CART:\n", flat_json(self.cart_view_seen)[:3000]]
        borrowed = self.foreign_records(item) if self.reading.get("references") else ""
        if borrowed:
            parts += ["\n\nOTHER LISTINGS' RECORDS (latest first; use only for values borrowed from another listing):\n", borrowed]
        recurring = self.repeated_labels()
        if recurring:
            parts += ["\n\nLABELS THAT KEEP COMING BACK ON THIS STORE'S LISTINGS (with counts; the item's category label has to be the one that matches the request's whole wording, not a near one):\n", recurring]
        return "".join(parts)

    @staticmethod
    def _pd_verdict_from(item: str, reply: dict[str, Any] | None) -> str:
        """Empty unless the checker said plainly that the listing does not fit."""
        if reply and reply.get("fits") is False:
            faults = [str(f) for f in reply.get("problems") or []][:5]
            return (f"A separate check finds that {item} does NOT meet every requirement: " + "; ".join(faults)
                    + ". Unless you are certain the check got it wrong, choose another fit that is fully verified.")
        return ""

    def vet_cart(self) -> str:
        notes = []
        for held in self.cart_items[-4:]:
            notes.append(self.vet_item(held, "ORDER"))
        return " ".join(filter(None, notes))

    def count_problem(self, name: str, args: dict[str, Any]) -> str:
        spec_map = self.arg_schemas[name].get("properties") or {}
        replacing = self.cart_replaces()
        if not replacing and self.add_mode is not None and mode_key(args, spec_map) != self.add_mode:
            return ""
        set_lines = lines_named(args, spec_map)
        products = self.product_lines()
        if products:
            return self._pd_basket_mismatch(products, set_lines) if self.cart_replaces() else ""
        target = self.reading.get("quantity")
        if isinstance(target, bool) or not isinstance(target, int) or target < 1:
            return ""
        if self.cart_replaces() and len(set_lines) > 1:
            return f"the request is for a single product, yet this call fills {len(set_lines)} cart lines."
        explicit = [args.get(field, spec.get("default", 1)) for field, spec in spec_map.items()
                    if isinstance(spec, dict) and spec.get("type") == "integer" and "enum" not in spec]
        per_line = [qty for _, qty in set_lines]
        wrong = next((qty for qty in explicit + per_line if isinstance(qty, int) and qty != target), None)
        if wrong is None:
            return ""
        return f"the request is for {target} of one listing, yet this call asks for {wrong}."

    @staticmethod
    def _pd_basket_mismatch(products: list[tuple[Any, Any]], set_lines: list[tuple[str, Any]]) -> str:
        """Several products wanted: the call must set exactly that many lines with exactly those counts."""
        expected = sorted(qty for _, qty in products)
        actual = sorted(qty if isinstance(qty, int) else 1 for _, qty in set_lines)
        if len(set_lines) == len(products) and actual == expected:
            return ""
        wish = ", ".join(f"{qty} x {what}" for what, qty in products)
        return f"the request is for {wish}, yet this call fills {len(set_lines)} line(s) with counts {actual}."

    def review_before_decline(self) -> str:
        if self.reviewed or self.seconds_left() < WIND_DOWN + 30 or self.spent > SPEND_CEILING:
            return ""
        self.reviewed = True
        packed = self._pd_pack_records(14000)
        if not packed:
            return ""
        prompt = "".join([REVIEW_GUIDE, "\n\nWHAT THE SHOPPER ASKED FOR:\n", self.query_text(),
                          "\n\nNEEDS IN FORCE:\n", self.needs_summary(),
                          "\n\nLISTING RECORDS SHOWN (latest first):\n", "\n".join(packed)])
        reply = self.ask_json(prompt, 1200)
        self.log.append({"kind": "audit", "result": reply})
        maybe = [str(x) for x in (reply or {}).get("possible_fits") or [] if isinstance(x, (str, int))][:5]
        if not maybe:
            return ""
        why = reply.get("reason")
        return ("A separate check thinks these listings could meet every requirement: " + ", ".join(maybe)
                + (f" ({why})" if why else "")
                + ". Check them (details, stock, price) and buy the right one; decline only if each of them truly fails.")

    def _pd_pack_records(self, room: int) -> list[str]:
        """Latest records first, each trimmed; any record that would overflow the room is skipped, not the rest."""
        packed, used = [], 0
        for record in reversed(self.seen_records):
            snippet = flat_json(clip(record, 160))
            if used + len(snippet) <= room:
                packed.append(snippet)
                used += len(snippet)
        return packed

    def clean_remark(self, args: dict[str, Any], name: str) -> dict[str, Any]:
        """Swap every free-text argument for the stock apology when the request bars naming listings, or when the
        remark happens to name a listing the store showed."""
        banned = self.reading.get("decline_no_names")
        if banned is False:
            return args
        if not banned and not self._pd_mentions_listing(args):
            return args
        specs = (self.arg_schemas.get(name) or {}).get("properties") or {}
        return {field: self._pd_plain_value(value, specs.get(field) or {}) for field, value in args.items()}

    def _pd_mentions_listing(self, args: dict[str, Any]) -> bool:
        for value in args.values():
            if not isinstance(value, str):
                continue
            if any(word and word in self.known_ids for word in re.split(r"[\s.,;:!?()\[\]{}<>\"'`]+", value)):
                return True
        return False

    @staticmethod
    def _pd_plain_value(value: Any, spec: dict[str, Any]) -> Any:
        if not isinstance(value, str) or "enum" in spec:
            return value
        return NOTHING_FITS[: spec.get("maxLength", len(NOTHING_FITS))]


# ---------------------------------------------------------------------------------------------------------- archive
class Archive:
    """Lookups over everything the store has returned: a listing's own records, a table of all listings, unopened
    leads, cheaper alternatives, property names and category-like labels."""

    @staticmethod
    def _pd_visit(root: Any, on_dict: Any) -> None:
        """Depth-first, pre-order walk down to nesting level 6; on_dict(node) returning True stops the descent there."""
        pending = [(root, 0)]
        while pending:
            node, level = pending.pop()
            if level > 6:
                continue
            if isinstance(node, dict):
                if on_dict(node):
                    continue
                below = list(node.values())
            elif isinstance(node, list):
                below = node
            else:
                continue
            pending.extend((child, level + 1) for child in reversed(below))

    @staticmethod
    def _pd_listing_id(node: dict[str, Any]) -> str | None:
        """First string value under an id-ish key that looks like a listing identifier."""
        for key, value in node.items():
            if isinstance(value, str) and "id" in str(key).lower() and ID_PATTERN.match(value):
                return value
        return None

    @staticmethod
    def _pd_code_like(text: str, needs_joiner: bool) -> bool:
        """Digits plus only word characters and dashes: an identifier or code (optionally it must contain - or _)."""
        if not (re.search(r"\d", text) and re.fullmatch(r"[A-Za-z0-9_\-]+", text)):
            return False
        return not needs_joiner or "-" in text or "_" in text

    def listing_table(self) -> dict[str, dict[str, Any]]:
        """id -> {price, stock, label, opened} for each listing seen. Price is the number-valued key seen most often,
        stock the most frequent true/false key, and label the repeating short-text key with the fewest distinct values."""
        tagged: list[tuple[str, dict[str, Any]]] = []
        for record in self.seen_records:
            for node in walk_dicts(record):
                ident = self._pd_listing_id(node)
                if ident:
                    tagged.append((ident, node))
        price_key, stock_key, label_key = self._pd_pick_columns(tagged)
        opened: set[str] = set()
        for sheet in self.sheet_records:
            collect_ids(sheet, opened)
        table: dict[str, dict[str, Any]] = {}
        for ident, node in tagged:
            if ident not in table:
                table[ident] = {"price": None, "stock": None, "label": None, "opened": ident in opened}
            entry = table[ident]
            cost = node.get(price_key) if price_key else None
            if isinstance(cost, (int, float)) and not isinstance(cost, bool):
                entry["price"] = cost
            flag = node.get(stock_key) if stock_key else None
            if isinstance(flag, bool):
                entry["stock"] = flag
            tag = node.get(label_key) if label_key else None
            if isinstance(tag, str):
                entry["label"] = tag
        return table

    @staticmethod
    def _pd_pick_columns(tagged: list[tuple[str, dict[str, Any]]]) -> tuple[Any, Any, Any]:
        num_hits: dict[str, int] = {}
        bool_hits: dict[str, int] = {}
        strings: dict[str, list[str]] = {}
        for ident, node in tagged:
            for key, value in node.items():
                if isinstance(value, bool):
                    bool_hits[key] = bool_hits.get(key, 0) + 1
                elif isinstance(value, (int, float)):
                    num_hits[key] = num_hits.get(key, 0) + 1
                elif isinstance(value, str) and value != ident and 0 < len(value) <= 40:
                    strings.setdefault(key, []).append(value)
        top_number = max(num_hits, key=num_hits.get, default=None)
        top_flag = max(bool_hits, key=bool_hits.get, default=None)
        recurring = [key for key, seen in strings.items() if len(seen) >= 2 * len(set(seen))]
        category = min(recurring, key=lambda key: (len(set(strings[key])), -len(strings[key])), default=None)
        return top_number, top_flag, category

    def within_budget(self, row: dict[str, Any]) -> bool:
        """False for a listing shown out of stock or priced over a known budget; True otherwise."""
        if row.get("stock") is False:
            return False
        cap, cost = self.budget, row.get("price")
        usable_cap = isinstance(cap, (int, float)) and not isinstance(cap, bool)
        return cost is None or not usable_cap or cost <= cap

    def wanted_words(self) -> list[set[str]]:
        """For each requirement in force, the word set of its value (text after the colon, before any note or !=)."""
        source = list(self.ledger_now) if self.ledger_now is not None else [
            need for need in self.given + self.told if need not in self.given_up]
        bags = (word_set(re.split(r"[(]|!=", str(need).partition(":")[2], maxsplit=1)[0]) for need in source)
        return [bag for bag in bags if bag]

    def lead_listings(self, limit: int) -> list[str]:
        """Affordable listings not yet opened that came up in searches typing required values; listings from searches
        that typed more of those values rank first, then the cheaper one."""
        search = self.role_map.get("search")
        needs = self.wanted_words()
        if not (search and needs):
            return []
        slots = (self.arg_schemas.get(search) or {}).get("properties") or {}
        table = self.listing_table()
        best: dict[str, int] = {}
        for tool, sent, result in self.call_log:
            if tool != search:
                continue
            typed: set[str] = set()
            for slot, text in sent.items():
                if isinstance(text, str) and "enum" not in (slots.get(slot) or {}):
                    typed.update(word_set(text))
            matched = len([need for need in needs if need <= typed])
            if matched == 0:
                continue
            surfaced: set[str] = set()
            collect_ids(result, surfaced)
            for ident in surfaced:
                if best.get(ident, 0) < matched:
                    best[ident] = matched
        opened_kinds = {row["label"] for row in table.values() if row["opened"] and row["label"] is not None}

        def eligible(ident: str) -> bool:
            row = table.get(ident)
            if row is None or row["opened"] or not self.within_budget(row) or ident in self.cart_items:
                return False
            return not opened_kinds or row["label"] in opened_kinds

        def rank(ident: str) -> tuple[int, float, str]:
            cost = table[ident]["price"]
            return -best[ident], (float("inf") if cost is None else cost), ident
        return sorted(filter(eligible, best), key=rank)[:limit]

    def cheaper_unread(self, limit: int = 4) -> list[str]:
        """Cheapest-first requests only: unopened, affordable listings of a carted item's category that undercut it."""
        if not self.cart_items or self.reading.get("priority") != "cheapest":
            return []
        table = self.listing_table()
        undercut: set[tuple[Any, str]] = set()
        for held in self.cart_items:
            anchor = table.get(held) or {}
            ceiling = anchor.get("price")
            if ceiling is None:
                continue
            kind = anchor.get("label")
            for ident, row in table.items():
                cost = row["price"]
                if ident in self.cart_items or row["opened"] or cost is None or cost >= ceiling:
                    continue
                if not self.within_budget(row) or (kind is not None and row["label"] != kind):
                    continue
                undercut.add((cost, ident))
        return [f"{ident} ({cost})" for cost, ident in sorted(undercut)][:limit]

    def records_for(self, item: str) -> list[str]:
        hits: list[str] = []

        def grab(node: dict[str, Any]) -> bool:
            if not any(isinstance(value, str) and value == item for value in node.values()):
                return False
            if len(hits) < 8:
                snippet = flat_json(clip(node, 400))[:2500]
                if snippet not in hits:
                    hits.append(snippet)
            return True
        for record in self.seen_records:
            if len(hits) >= 8:
                break
            self._pd_visit(record, grab)
        return hits[-4:]

    def foreign_records(self, item: str, room: int = 6000) -> str:
        """Records of listings besides this one, latest first, for values that the request or a note borrows."""
        gathered: list[str] = []
        named: set[str] = set()

        def grab(node: dict[str, Any]) -> bool:
            ident = self._pd_listing_id(node)
            if not ident or len(node) < 5:
                return False
            if ident != item and ident not in named:
                named.add(ident)
                gathered.append(flat_json(clip(node, 200))[:1500])
            return True
        for record in reversed(self.seen_records):
            self._pd_visit(record, grab)
        chosen: list[str] = []
        used = 0
        for snippet in gathered:
            used += len(snippet)
            if used > room:
                break
            chosen.append(snippet)
        return "\n".join(chosen)

    def property_menu(self) -> tuple[str, int]:
        """Property names found on fact sheets (dicts holding at least five short strings) with how often each
        appears and a few sample values, plus the number of such sheets."""
        tally: dict[str, int] = {}
        examples: dict[str, list[str]] = {}
        sheet_total = 0

        def take(node: dict[str, Any]) -> bool:
            nonlocal sheet_total
            short = [(key, text) for key, text in node.items() if isinstance(text, str) and 0 < len(text) <= 70]
            if len(short) < 5:
                return False
            sheet_total += 1
            for key, text in short:
                if self._pd_code_like(text, True):
                    continue        # identifiers and codes are not property values
                prop = re.sub(r"[_.]+", " ", str(key)).strip().lower()
                tally[prop] = tally.get(prop, 0) + 1
                bucket = examples.setdefault(prop, [])
                lowered = text.lower()
                if len(bucket) < 5 and lowered not in bucket:
                    bucket.append(lowered)
            return False
        has_detail = self.role_map.get("details") or self.role_map.get("view")
        for record in self.sheet_records if has_detail else self.seen_records:
            self._pd_visit(record, take)
        leaders = sorted(tally, key=tally.__getitem__, reverse=True)[:28]
        menu = "\n".join(f"- {prop} (in {tally[prop]} listings): " + " | ".join(examples.get(prop, [])) for prop in leaders)
        return menu, sheet_total

    def repeated_labels(self) -> str:
        """Keys whose short text values recur across records with 2 to 8 distinct values: likely category labels."""
        per_key: dict[str, dict[str, int]] = {}

        def take(node: dict[str, Any]) -> bool:
            if len(node) < 4:
                return False
            for key, text in node.items():
                if isinstance(text, str) and 0 < len(text) <= 40 and not self._pd_code_like(text, False):
                    counter = per_key.setdefault(str(key), {})
                    counter[text] = counter.get(text, 0) + 1
            return False
        for record in self.seen_records:
            self._pd_visit(record, take)
        rows = []
        for key, counter in per_key.items():
            if len(rows) == 12:
                break
            spread = len(counter)
            if 2 <= spread <= 8 and sum(counter.values()) >= 4 and max(counter.values()) >= 2:
                ordered = sorted(counter.items(), key=lambda pair: pair[1], reverse=True)
                rows.append(f"- {key}: " + " | ".join(f"{text} (x{n})" for text, n in ordered))
        return "\n".join(rows)


# ---------------------------------------------------------------------------------------------------------- interview
class Interview:
    """Talking to the shopper: questions drawn up from the listing properties, and answers to the shopper's questions."""

    def note_cap(self) -> int:
        allowance = self.message_allowance
        return 3 if allowance is None else allowance

    @staticmethod
    def _pe_text_slot(must: list[str], free: list[str]) -> str | None:
        """The free-text argument that should carry a note: a required one first, else the only one there is."""
        for name in must:
            if name in free:
                if name:
                    return name
                break
        return free[0] if len(free) == 1 else None

    def note_args(self, text: str) -> dict[str, Any] | None:
        spec = self.arg_schemas.get(self.role_map.get("message") or "") or {}
        must = list(spec.get("required") or [])
        free = [name for name, shape in (spec.get("properties") or {}).items() if is_text_slot(shape) and "enum" not in shape]
        target = self._pe_text_slot(must, free)
        if not target:
            return None
        if [name for name in must if name != target]:
            return None
        return {target: text}

    def note_call(self, text: str, call_id: str) -> dict[str, Any] | None:
        sender = self.role_map.get("message")
        packed = self.note_args(text)
        if packed:
            fitted, problems = fit_schema(packed, self.arg_schemas[sender], "args")
            if not problems:
                return {"id": call_id, "type": "function", "function": {"name": sender, "arguments": json.dumps(fitted)}}
        return None

    def _pe_file_results(self, answers: list[tuple[str, str]]) -> list[str]:
        """Put tool results into the conversation, the newest one carrying the status line; gives back the texts stored."""
        stored: list[str] = []
        final = len(answers) - 1
        for index, (call_id, body) in enumerate(answers):
            if index == final:
                body = body + "\n" + self.status_line()
            self.convo.append({"role": "tool", "tool_call_id": call_id, "content": body})
            stored.append(body)
        return stored

    def _pe_question_prompt(self, room: int, survey: str) -> str:
        sections = [PLAN_GUIDE.format(count=room),
                    "REQUEST FROM THE SHOPPER:\n" + self.query_text(),
                    "NEEDED UP TO NOW:\n" + self.needs_summary()]
        if self.asked_log:
            sections.append("QUESTIONS ALREADY SENT (and how the shopper replied):\n" + "\n".join(self.asked_log))
        must_learn = self.reading.get("ask_first")
        if must_learn:
            sections.append("MUST BE LEARNED FROM THE SHOPPER: " + "; ".join(str(item) for item in self.reading.get("ask_first") or []))
        sections.append("PROPERTIES FOUND IN LISTING DETAILS UP TO NOW (name, listing count, example values):\n" + (survey or "(nothing seen so far)"))
        return "\n\n".join(sections)

    def ask_planned(self, force: bool = False) -> bool:
        cap = self.note_cap()
        if not self.role_map.get("message") or self.notes_sent >= cap:
            self.planning = False
            return False
        survey, sheet_count = self.property_menu()
        pressing = not self.asked_log and bool(self.reading.get("ask_first"))
        early = sheet_count < 4 and self.steps_used < 8
        if early and not (pressing or force):
            return False
        pressing = pressing and not force
        if self.steps_remaining() <= 10 or self.seconds_left() < WIND_DOWN + 40:
            self.planning = False
            return False
        room = cap - self.notes_sent
        prompt = self._pe_question_prompt(room, survey)
        order = [HEAVY_MODEL, EAR_MODEL] if self.heavy_fits(prompt) else [EAR_MODEL, HEAVY_MODEL]
        drafted = (self.ask_json(prompt, 2000, order) or {}).get("questions") or []
        picked = [entry for entry in drafted if isinstance(entry, dict) and str(entry.get("question") or "").strip()]
        single = pressing and sheet_count < 4
        picked = picked[:1] if single else picked
        picked = picked[:room]
        outgoing: list[dict[str, Any]] = []
        topics: list[tuple[str, str]] = []
        base = self.notes_sent
        for offset, entry in enumerate(picked, start=1):
            wording = str(entry.get("question")).strip()
            built = self.note_call(wording, f"ask_{base + offset}_{self.steps_used}")
            if built is not None:
                outgoing.append(built)
                topics.append((str(entry.get("property")), wording))
        if not outgoing:
            self.planning = False
            return False
        self.convo.append({"role": "assistant", "content": "Putting questions to the shopper on " + ", then ".join(topic for topic, _ in topics) + ".",
                           "tool_calls": outgoing})
        stored = self._pe_file_results(self.route_calls(outgoing, planned=True))
        for (topic, wording), body in zip(topics, stored):
            heard = body.split("\n" + STATUS_MARK)[0][:400]
            self.asked_log.append(f"- {topic}: question \"{wording}\" -> {heard}")
        if self.notes_sent >= cap or not single:
            self.planning = False
        return True

    def respond(self, text: str) -> bool:
        blocked = (not self.role_map.get("message") or self.notes_sent >= self.note_cap()
                   or self.steps_remaining() <= 4 or self.seconds_left() < WIND_DOWN)
        if blocked:
            return False
        outgoing = self.note_call(text, f"reply_{self.steps_used}")
        if outgoing is None:
            return False
        self.convo.append({"role": "assistant", "content": "Replying to what the shopper asked.", "tool_calls": [outgoing]})
        self._pe_file_results(self.route_calls([outgoing], planned=True))
        return True


# ---------------------------------------------------------------------------------------------------------- needbook
class NeedBook:
    """Keeps the current requirements, writes the progress line, and interprets what the shopper says."""

    def _pe_need_entries(self) -> list[str]:
        if self.ledger_now is not None:
            return list(self.ledger_now)
        kept = [(need, "stated in the request") for need in self.given if need not in self.given_up]
        kept += [(need, "disclosed by the shopper") for need in self.told if need not in self.given_up]
        return [f"{need} ({origin})" for need, origin in kept]

    def needs_summary(self) -> str:
        entries = self._pe_need_entries()
        pieces = ["; ".join(entries) if entries else "as in the request"]
        if self.budget is not None:
            pieces.append(f"whole price no more than {self.budget} {self.reading.get('currency') or ''}".rstrip())
        kinds = self.product_lines()
        pieces.append("lines to order: " + ", ".join(f"{count} x {product}" for product, count in kinds) if kinds
                      else f"units wanted {self.reading.get('quantity') or 1}")
        if self.given_up:
            pieces.append("not required any more: " + "; ".join(self.given_up))
        if self.open_props:
            pieces.append("the shopper has no preference on: " + ", ".join(self.open_props))
        return "; ".join(pieces)

    def _pe_pressure_hint(self, left: int) -> str:
        if left <= 2:
            return " LAST STEPS: order now when the cart holds a verified fit; if it does not, decline."
        if left <= 5 or self.seconds_left() < WIND_DOWN or self.spent > SPEND_WIND_DOWN:
            return (" WRAP UP NOW: make the cart hold exactly the verified lines the request wants, check their stock again,"
                    " then order (or decline when nothing fits).")
        if self.orders_tried == 0 and left <= 14:
            return (" Few steps remain: in the next few steps put your best verified fit in the cart and order; search and detail"
                    " reading end once 7 steps remain, and fixing a refused order costs about 4 steps.")
        return ""

    def status_line(self) -> str:
        left = self.steps_remaining()
        sent = (f"{self.notes_sent} of {self.message_allowance} messages used" if self.message_allowance is not None
                else f"{self.notes_sent} messages used")
        head = "; ".join([f"{STATUS_MARK} steps: {self.steps_used} spent of {self.step_limit}, {left} remaining", sent,
                          f"about {int(max(0, self.seconds_left() - HARD_STOP))} s remaining"])
        return head + ".\n[needed now] " + self.needs_summary() + self._pe_pressure_hint(left)

    def ask_json(self, prompt: str, max_tokens: int = 700, route: list[str] | None = None) -> dict[str, Any] | None:
        answer = self.call_models([{"role": "user", "content": prompt}], None, route or [JUDGE_MODEL, TURN_MODEL], max_tokens)
        return object_in((answer or {}).get("content") or "")

    def _pe_ear_prompt(self, fresh: list[str], question: str | None) -> str:
        sections = [EAR_GUIDE, "REQUEST FROM THE SHOPPER (background):\n" + self.query_text(), "NEEDED UP TO NOW:\n" + self.needs_summary()]
        if self.exchanges:
            sections.append("PREVIOUS TURNS WITH THE SHOPPER (oldest first):\n" + "\n".join(self.exchanges[-12:]))
        if question:
            sections.append("THE ASSISTANT'S QUESTION:\n" + question)
        sections.append("THE SHOPPER'S NEW WORDS:\n" + "\n".join(fresh))
        return "\n\n".join(sections)

    def _pe_confused(self) -> str:
        self.confused_in_a_row += 1
        stop = self.confused_in_a_row >= 2
        if stop:
            self.planning = False
            self.message_allowance = self.notes_sent
        tail = " Do not message the shopper again." if stop else ""
        return "That message was not understood by the shopper; do not rephrase and resend the same question." + tail

    def listen(self, lines: list[str], question: str | None) -> str:
        """Turn shopper lines not heard before into ledger changes; gives a short note of what they mean ("" if nothing new)."""
        fresh = list(filter(lambda said: said not in self.heard_lines, lines))
        if not fresh:
            return ""
        self.heard_lines.update(fresh)
        if self.seconds_left() < WIND_DOWN:
            return ""
        prompt = self._pe_ear_prompt(fresh, question)
        lead = f"question: {question} -> " if question else ""
        self.exchanges.append(lead + "shopper: " + " / ".join(fresh))
        reading = self.ask_json(prompt, 1500, [EAR_MODEL, JUDGE_MODEL])
        if not reading:
            return ""
        if reading.get("not_understood") is True:
            return self._pe_confused()
        self.confused_in_a_row = 0
        if question:
            # a reply to our own question may add needs; it never removes or relaxes one already in force
            reading["dropped"] = []
            listed = reading.get("requirements_now")
            if isinstance(listed, list) and self.ledger_now:
                reading["requirements_now"] = listed + [need for need in self.ledger_now if need not in listed]
        changes = self._absorb(reading)
        suggested = str(reading.get("reply") or "").strip()
        asked_back = any("?" in said for said in fresh)
        if suggested and self.role_map.get("message") and asked_back:
            self.reply_waiting = suggested
        gist = "Shopper means: " + str(reading.get("meaning") or "")
        return gist + (" (" + "; ".join(changes) + ")" if changes else "")

    @staticmethod
    def _pe_new_texts(values: Any, have: list[str]) -> list[str]:
        """Non-empty strings from a reading list that are not yet on the given list (in their order of arrival)."""
        added: list[str] = []
        for value in values or []:
            if isinstance(value, str) and value and value not in have:
                have.append(value)
                added.append(value)
        return added

    def _absorb(self, reading: dict[str, Any]) -> list[str]:
        listed = reading.get("requirements_now")
        if isinstance(listed, list) and all(isinstance(need, str) for need in listed):
            self.ledger_now = [need for need in listed if need.strip()]
        changes = [f"requirement added: {need}" for need in self._pe_new_texts(reading.get("needs"), self.told)]
        self._pe_new_texts(reading.get("no_preference"), self.open_props)
        changes += [f"requirement removed: {gone}" for gone in self._pe_new_texts(reading.get("dropped"), self.given_up)]
        limit = reading.get("new_budget")
        usable = isinstance(limit, (int, float)) and not isinstance(limit, bool) and limit > 0
        if usable and limit != self.budget:
            self.budget = limit
            changes.append(f"budget changed to: {limit}")
        if reading.get("unavailable"):
            changes.append("an item has become unavailable; remove it from the cart and leave it out of the order")
        if changes:
            self.ledger_rev += 1
            self.heavy_due = True
        return changes


# ---------------------------------------------------------------------------------------------------------- intake
class Intake:
    """A single model pass over the request and the tool catalogue, plus the summary note handed to the turn model."""

    def _pc_count(self, raw: Any) -> bool:
        return isinstance(raw, (int, float)) and not isinstance(raw, bool)

    def read_request_once(self) -> None:
        tool_rows = []
        for spec in self.tool_specs:
            fn = spec["function"]
            tool_rows.append("- %s: %s | arguments: %s" % (fn["name"], fn.get("description", ""), flat_json(fn.get("parameters") or {})))
        stable_block = "\n".join([INTAKE_GUIDE + "\n\nTOOLS:"] + tool_rows)
        opening = self.view.get("opening")
        task_block = "REQUEST:\n" + self.query_text() + ("" if opening is None else "\n\nFIRST OBSERVATION:\n" + flat_json(self.view.get("opening")))
        content = [{"type": "text", "text": stable_block, "cache_control": {"type": "ephemeral"}},
                   {"type": "text", "text": task_block}]
        answer = self.call_models([{"role": "user", "content": content}], None, [INTAKE_MODEL, FALLBACK_MODEL], 1500)
        self.reading = parsed = object_in((answer or {}).get("content") or "") or {}
        roles = parsed.get("roles")
        if not isinstance(roles, dict):
            roles = {}
        self.role_map = {}
        for role, tool in roles.items():
            if isinstance(tool, str) and tool in self.arg_schemas:
                self.role_map[role] = tool
        talk_cap = parsed.get("message_budget")
        if self._pc_count(talk_cap) and talk_cap >= 0:
            self.message_allowance = int(talk_cap)
        self.given = [str(need) for need in parsed.get("requirements") or [] if need]
        self.budget = parsed.get("price_cap")
        if parsed and self.budget is None and self.role_map.get("message"):
            prior = [str(topic) for topic in parsed.get("ask_first") or [] if topic]
            prior.insert(0, "the highest amount the shopper will pay, since the request sets no price limit (leave it out if it is listed again below)")
            parsed["ask_first"] = prior
        step_cap = parsed.get("step_budget")
        if self._pc_count(step_cap) and 0 < step_cap < self.step_limit:
            self.step_limit = int(step_cap)

    def query_text(self) -> str:
        text = self.view.get("query")
        return str(text) if text else ""

    def product_lines(self) -> list[tuple[str, int]]:
        """(product, quantity) pairs, but only when the request names two or more kinds of product."""
        def units(raw: Any) -> int:
            valid = isinstance(raw, int) and not isinstance(raw, bool) and raw > 0
            return raw if valid else 1
        pairs = [(str(row["product"]), units(row.get("quantity")))
                 for row in self.reading.get("lines") or [] if isinstance(row, dict) and row.get("product")]
        return pairs if len(pairs) >= 2 else []

    def cart_replaces(self) -> bool:
        return "replace" == self.reading.get("cart_mode")

    def _pc_choice_rule(self, parsed: dict[str, Any]) -> str:
        goal = parsed.get("priority")
        if goal not in ("cheapest", "best"):
            return "- no cheapest/best rule was found: any item that meets every requirement within the budget is correct, so do not spend steps chasing the lowest price (unless the request asks for it)"
        detail = parsed.get("priority_detail")
        return f"- choose the {goal}{f' ({detail})' if detail else ''} among all items that fit"

    def intake_note(self) -> str:
        """Turn the request reading into a checklist for the acting model."""
        parsed = self.reading
        if not parsed:
            return ""
        out = ["Reading of the request (compare it with the request itself):"]
        if parsed.get("product"):
            out.append("- product: %s" % (parsed.get("product"),))
        groups = self.product_lines()
        if not groups:
            out.append("- quantity: %s" % (parsed.get("quantity") or 1,))
        else:
            out.extend("- order line: %s x %s" % (qty, item) for item, qty in groups)
            out.append("- more than one product: every line carries its own requirements, and the budget applies to the whole order")
        out.extend("- stated requirement: %s" % (need,) for need in parsed.get("requirements") or [])
        out.extend("- value copied from another listing (locate and open that listing first): %s" % (ref,) for ref in parsed.get("references") or [])
        if self.cart_replaces():
            out.append("- the cart tool overwrites the whole cart with whatever lines it gets: every line you want kept must be sent each time")
        cap = parsed.get("price_cap")
        if cap is not None:
            out.append(("- highest total price: %s %s" % (cap, parsed.get("currency") or "")).rstrip())
        if parsed.get("in_stock_only"):
            out.append("- only items that are in stock / on hand")
        out.append(self._pc_choice_rule(parsed))
        sections = (("soft_wants", "- nice to have, NOT required (only to break ties): "),
                    ("ask_first", "- must be learned from the shopper first: "),
                    ("rules", "- rule: "))
        for key, lead in sections:
            out.extend(lead + str(entry) for entry in parsed.get(key) or [])
        if self.message_allowance is not None:
            out.append("- messages the shopper permits: %s" % (self.message_allowance,))
        out.append("- steps permitted: %s" % (self.step_limit,))
        pointers = [prop for prop in parsed.get("hint_properties") or [] if isinstance(prop, str)]
        if pointers:
            out.append("- properties the story may hint at (ask about these first, one per message): " + ", ".join(pointers))
        return "\n".join(out)


# ---------------------------------------------------------------------------------------------------------- wrapup
class Wrapup:
    """The code-driven ending: send an order when the cart was used, otherwise a decline."""

    @staticmethod
    def _pa_slot_default(spec: Any, text: str) -> tuple[bool, Any]:
        """(filled, value) for one required argument: the text for free-text slots, else a neutral default."""
        if is_text_slot(spec):
            return True, text[: spec.get("maxLength", len(text))]
        if "enum" in spec and spec["enum"]:
            return True, spec["enum"][0]
        kind = spec.get("type")
        if kind == "boolean":
            return True, True
        if kind in ("integer", "number"):
            return True, max(1, spec.get("minimum", 1))
        return False, None

    def default_args(self, name: str, text: str) -> dict[str, Any]:
        schema = self.arg_schemas.get(name) or {}
        props = schema.get("properties") or {}
        filled: dict[str, Any] = {}
        for key in schema.get("required") or []:
            ok, guess = self._pa_slot_default(props.get(key) or {}, text)
            if ok:
                filled[key] = guess
        unfilled = [(key, spec) for key, spec in props.items() if key not in filled]
        if len(props) == 1:
            for key, spec in unfilled:
                if isinstance(spec, dict) and spec.get("type") == "string" and "enum" not in spec:
                    filled[key] = text[: spec.get("maxLength", len(text))]
        return filled

    def _pa_send_final(self, action: dict[str, Any]) -> dict[str, Any] | None:
        answer = self.call_store([action])
        self.log.append({"kind": "forced_finish", "action": action, "reply": answer})
        return answer

    @staticmethod
    def _pa_was_refused(answer: Any) -> bool:
        if not answer or answer.get("finished"):
            return False
        seen = answer.get("observation")
        return isinstance(seen, dict) and "refusal" in seen

    def wrap_up(self) -> None:
        """Last resort once the turns are over: order (cart used) or decline, and turn a refused order into a decline."""
        if self.done or self.out_of_funds:
            return
        if self.steps_remaining() <= 0 or self.seconds_left() < 2:
            return
        order_tool = self.role_map.get("order")
        decline_tool = self.role_map.get("decline")
        if order_tool and self.cart_touched:
            opening = {"name": order_tool, "args": self.default_args(order_tool, "Please order the cart.")}
        elif decline_tool:
            opening = {"name": decline_tool, "args": self.default_args(decline_tool, NOTHING_FITS)}
        else:
            return
        if not self._pa_was_refused(self._pa_send_final(opening)):
            return
        if decline_tool and self.steps_remaining() > 0:
            self._pa_send_final({"name": decline_tool, "args": self.default_args(decline_tool, NOTHING_FITS)})


# ---------------------------------------------------------------------------------------------------------- channel
class Channel:
    """Outbound traffic: model requests tried along a route (with cost booking) and numbered store steps."""

    def steps_remaining(self) -> int:
        used, cap = self.steps_used, self.step_limit
        return cap - used

    def _pa_take_message(self, model: str, data: Any) -> dict[str, Any] | None:
        """Book the cost of a reply and hand back its message when it carries text or tool calls."""
        if not (isinstance(data, dict) and data.get("choices")):
            return None
        self.spent += self._priced(model, data.get("usage") or {})
        message = data["choices"][0].get("message") or {}
        return message if (message.get("content") or message.get("tool_calls")) else None

    def call_models(self, convo: list[dict[str, Any]], tool_specs: list[dict[str, Any]] | None, route: list[str], max_tokens: int) -> dict[str, Any] | None:
        for model in route:
            remaining = self.seconds_left()
            if remaining < HARD_STOP - 10:
                return None
            _MODELS.timeout = int(max(8, min(75, remaining - HARD_STOP + 15)))
            request: dict[str, Any] = {
                "model": model,
                "messages": self._with_cache_marks(model, convo),
                "temperature": 0,
                "max_tokens": max_tokens,
            }
            if tool_specs:
                request.update(tools=tool_specs, tool_choice="auto")
            with self.counting():
                outcome = _MODELS.post_verbose("/inference/chat/completions", request)
            message = self._pa_take_message(model, outcome.data)
            if message is not None:
                return message
            problem = str(outcome.error)[:300] if outcome.error else "empty"
            self.log.append({"kind": "inference_failure", "model": model, "error": problem})
        return None

    @staticmethod
    def _priced(model: str, usage: dict[str, Any]) -> float:
        reported = usage.get("cost")
        if isinstance(reported, (int, float)):
            return float(reported)
        rate_in, rate_cached, rate_out = TOKEN_PRICES.get(model, TOKEN_PRICES[HEAVY_MODEL])
        sent = usage.get("prompt_tokens") or 0
        reused = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
        made = usage.get("completion_tokens") or 0
        return max(0, sent - reused) * rate_in + reused * rate_cached + made * rate_out

    def heavy_fits(self, payload: Any) -> bool:
        """Estimate (from the payload's size) whether one strong-model call still leaves the check reserve
        under the episode's spend ceiling."""
        rate_in, rate_cached, rate_out = TOKEN_PRICES[HEAVY_MODEL]
        approx_tokens = len(flat_json(payload)) / 4
        per_token = (1 - CACHE_HIT_SHARE) * rate_in + CACHE_HIT_SHARE * rate_cached
        estimate = approx_tokens * per_token + HEAVY_OUTPUT_GUESS * rate_out
        return self.spent + estimate + JUDGE_RESERVE <= SPEND_CEILING

    def _pa_heavy_allowed(self) -> bool:
        if not HEAVY_TURNS or not self.heavy_due or self.heavy_used >= HEAVY_TURNS:
            return False
        return self.heavy_fits([self.convo, self.tools_now()])

    def turn_models(self) -> list[str]:
        """Route for the next turn: led by the strong model when a search/pick turn is due and the budget allows."""
        if not self._pa_heavy_allowed():
            return [TURN_MODEL, FALLBACK_MODEL, TURN_MODEL]
        self.heavy_due = False
        self.heavy_used += 1
        return [HEAVY_MODEL, TURN_MODEL, FALLBACK_MODEL]

    @staticmethod
    def _with_cache_marks(model: str, convo: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Anthropic models with more than two messages: copy the list and flag the first and last text for caching."""
        if len(convo) <= 2 or not model.startswith("anthropic/"):
            return convo
        copies = list(map(dict, convo))
        for idx in (0, -1):
            body = copies[idx].get("content")
            if isinstance(body, str) and body:
                copies[idx]["content"] = [{"type": "text", "text": body, "cache_control": {"type": "ephemeral"}}]
        return copies

    def _pa_book_reply(self, data: dict[str, Any], count: int) -> dict[str, Any]:
        reported = data.get("step")
        self.steps_used = reported if isinstance(reported, int) else self.steps_used + count
        if data.get("finished"):
            self.done = True
        return data

    def _pa_retry_after(self, error: Any, request: dict[str, Any]) -> bool:
        """Log a failed store call; True when one more attempt makes sense (step resync or an unknown status)."""
        code = error.get("status")
        text = str(error.get("body") or "")
        self.log.append({"kind": "env_failure", "status": code, "body": text[:300]})
        if code == 402:
            self.out_of_funds = True
            return False
        if code in (403, 422, 500, 503, 504):
            return False
        if code != 409:
            return True
        if "finished" in text:
            self.done = True
            return False
        hit = re.search(r"expected (\d+)", text)
        if hit is None:
            return False
        self.steps_used = int(hit.group(1)) - 1
        request["step"] = self.steps_used + 1
        return True

    def call_store(self, actions: list[dict[str, Any]]) -> dict[str, Any] | None:
        request: dict[str, Any] = dict(self.link)
        single = len(actions) == 1
        request.update({"step": self.steps_used + 1, ("action" if single else "actions"): actions[0] if single else actions})
        for _attempt in range(3):
            remaining = self.seconds_left()
            if remaining < 2:
                return None
            _SHOP.timeout = int(max(3, min(95, remaining - 1)))
            with self.counting():
                outcome = _SHOP.post_verbose("/environment/call", request)
            if isinstance(outcome.data, dict):
                return self._pa_book_reply(outcome.data, len(actions))
            if not self._pa_retry_after(outcome.error or {}, request):
                return None
        return None


# ---------------------------------------------------------------------------------------------------------- timekeeping
class TimeKeeping:
    """Clock-free timing: while a request is pending, a helper thread adds one tick per fixed slice."""

    def _start_clock(self) -> None:
        self.ticks = 0

    def seconds_left(self) -> float:
        elapsed = self.ticks * TICK
        return TIME_BUDGET - elapsed

    @contextmanager
    def counting(self) -> Iterator[None]:
        """Tick once per TICK on a helper thread during the wrapped call, plus a final tick for the partial slice."""
        stop = threading.Event()

        def _pa_tally() -> None:
            while True:
                if stop.wait(TICK):
                    return
                self.ticks += 1
        worker = threading.Thread(target=_pa_tally, daemon=True)
        worker.start()
        try:
            yield
        finally:
            stop.set()
            worker.join()
            self.ticks += 1



# ---------------------------------------------------------------------------------------------------------- the session
class Agent(TimeKeeping, Channel, Intake, NeedBook, Archive, Referee, Interview, Wrapup, Gatekeeper, Courier):
    def __init__(self, problem_data: dict[str, Any]):
        self._start_clock()
        environment = problem_data["environment"]
        self.link = dict(environment["binding"])
        self.view = environment["policy_view"]
        self.tool_specs = [t for t in self.view.get("tools") or [] if isinstance(t, dict) and isinstance(t.get("function"), dict)]
        self.arg_schemas = {t["function"]["name"]: t["function"].get("parameters") or {"type": "object", "properties": {}} for t in self.tool_specs}
        self.step_limit = int(self.view.get("max_steps") or 30)
        self.steps_used = 0
        self.done = bool(self.view.get("finished"))
        self.out_of_funds = False
        # shopper and request
        self.reading: dict[str, Any] = {}
        self.role_map: dict[str, str] = {}
        self.message_allowance: int | None = None
        self.notes_sent = 0
        self.given: list[str] = []
        self.told: list[str] = []
        self.open_props: list[str] = []
        self.given_up: list[str] = []
        self.ledger_now: list[str] | None = None
        self.budget: Any = None
        self.ledger_rev = 0
        self.heard_lines: set[str] = set()
        self.exchanges: list[str] = []
        self.asked_log: list[str] = []
        self.planning = True
        self.reply_waiting = ""
        # store and cart
        self.seen_records: list[Any] = []
        self.known_ids: set[str] = set()
        self.cart_items: list[str] = []
        self.cart_modes: dict[str, str] = {}
        self.cart_units: dict[str, Any] = {}
        self.sheet_records: list[Any] = []
        self.call_log: list[tuple[str, dict[str, Any], Any]] = []
        self.holds: set[str] = set()
        self.confused_in_a_row = 0
        self.cart_touched = False
        self.cart_touched_at = 0
        self.cart_view_seen: Any = None
        self.add_mode: str | None = None
        self.last_calls: list[str] = []
        # checks
        self.orders_tried = 0
        self.refused_count = 0
        self.order_vetted = False
        self.decline_vetted = False
        self.reviewed = False
        self.verdict_cache: dict[str, str] = {}
        self.cart_vetted: set[str] = set()
        self.count_warned: set[str] = set()
        # the conversation with the acting model
        self.convo: list[dict[str, Any]] = []
        self.spent = 0.0
        self.heavy_due = True
        self.heavy_used = 0
        self.log: list[dict[str, Any]] = []

    def first_turn_text(self) -> str:
        sections = ["REQUEST FROM THE SHOPPER:\n" + self.query_text()]
        start = self.view.get("opening")
        if start is not None:
            sections.append("OPENING OBSERVATION:\n" + flat_json(start))
        reading_note = self.intake_note()
        if reading_note:
            sections.append(reading_note)
        sections.append(self.status_line())
        return "\n\n".join(sections)

    def _pe_record_opening(self) -> None:
        start = self.view.get("opening")
        if start is None:
            return
        self.seen_records.append(start)
        collect_ids(start, self.known_ids)

    @staticmethod
    def _pe_normalise_calls(raw_calls: Any, tally: dict[str, int]) -> list[dict[str, Any]]:
        """Model tool calls in one standard shape: string arguments and an id for every call."""
        tidy: list[dict[str, Any]] = []
        for raw in raw_calls:
            if isinstance(raw, dict):
                tally["numbered"] += 1
                fn = raw.get("function") or {}
                packed = fn.get("arguments")
                if not isinstance(packed, str):
                    packed = json.dumps(fn.get("arguments") or {})
                tidy.append({"id": raw.get("id") or f"call_{tally['numbered']}", "type": "function",
                             "function": {"name": fn.get("name"), "arguments": packed}})
        return tidy

    def _pe_turn(self, tally: dict[str, int]) -> bool:
        """One pass of the main loop; True when the loop should stop."""
        if self.done or self.out_of_funds or self.steps_remaining() <= 0:
            return True
        if self.seconds_left() < HARD_STOP or self.spent > SPEND_STOP:
            self.wrap_up()
            return True
        if self.reply_waiting:
            pending = self.reply_waiting
            self.reply_waiting = ""
            if self.respond(pending):
                return False
        if self.planning and self.ask_planned():
            return False
        offered = self.tools_now()
        reply = self.call_models(self.convo, offered, self.turn_models(), 2500)
        if reply is None:
            tally["failed"] += 1
            if tally["failed"] < 2:
                return False
            self.wrap_up()
            return True
        calls = self._pe_normalise_calls(reply.get("tool_calls") or [], tally)
        said = reply.get("content") or ""
        said = said if isinstance(said, str) else flat_json(said)
        final_roles = {self.role_map.get(role) for role in ("cart", "order", "decline")}
        final_roles.discard(None)
        commits = any(call["function"]["name"] in final_roles for call in calls)
        if self.planning and commits and self.ask_planned(force=True):
            return False
        if calls:
            tally["silent"] = 0
            self.log.append({"kind": "model", "content": said[:2000], "calls": [call["function"] for call in calls]})
            self.convo.append({"role": "assistant", "content": said or None, "tool_calls": calls})
            self._pe_file_results(self.route_calls(calls))
            return False
        tally["silent"] += 1
        self.convo.append({"role": "assistant", "content": said or "(no tool used)"})
        self.convo.append({"role": "user", "content": "Every reply must use at least one of the tools. Carry on.\n" + self.status_line()})
        if tally["silent"] < 3:
            return False
        self.wrap_up()
        return True

    def play(self) -> list[dict[str, Any]]:
        if not self.done:
            self.read_request_once()
            self.log.append({"kind": "plan", "plan": self.reading, "roles": self.role_map})
            opener = self.first_turn_text()
            self._pe_record_opening()
            self.convo = [{"role": "system", "content": TURN_GUIDE}, {"role": "user", "content": opener}]
            tally = {"silent": 0, "failed": 0, "numbered": 0}
            turns = 0
            while turns < MAX_TURNS and not self._pe_turn(tally):
                turns += 1
            if not (self.done or self.out_of_funds):
                self.wrap_up()
            self.log.append({"kind": "end", "finished": self.done, "steps": self.steps_used, "cost": round(self.spent, 5),
                             "seconds": round(self.ticks * TICK, 1)})
        return self.log


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Run one episode; any crash is logged and the episode is still closed from code where possible."""
    runner = Agent(problem_data)
    try:
        trace = runner.play()
    except Exception as failure:
        runner.log.append(_pe_crash_entry(failure))
        try:
            if not (runner.done or runner.out_of_funds):
                runner.wrap_up()
        except Exception as second:
            runner.log.append(_pe_crash_entry(second))
        trace = runner.log
    return trace


__all__ = ["agent_main"]
