from __future__ import annotations

import collections
import itertools
import json
import re
import time
from types import SimpleNamespace
from typing import Any, Callable

from src.agent.proxy_client import ProxyClient

MAIN_MODEL = "anthropic/claude-haiku-5.5"
BACKUP_MODEL = "openai/gpt-6-luna"
SETUP_MODEL = "anthropic/claude-haiku-5.5"
SETUP_FALLBACKS = ["openai/gpt-oss-120b", "anthropic/claude-sonnet-5.5"]
LAST_RESORT = "anthropic/claude-sonnet-5.5"
READ_MODEL = "anthropic/claude-haiku-5.5"

TASK_SECONDS = 265.0
WRAP_UP_SECONDS = 70.0
FORCE_SECONDS = 28.0
OBS_CHARS = 16000
MAX_TURNS = 80

SYSTEM_PROMPT = """You are a careful buying assistant working for a single shopper in a simulated shop; the supplied tools are your only way to act. Scoring is all or nothing (1 or 0), and a 1 needs a finish that is right in every respect: the bought item satisfies each requirement the shopper wrote, each hidden need the shopper discloses when asked and each change the shopper makes along the way; quantity and budget are correct; the request's cheapest (or best) demand, if any, is honoured; and every rule the request lays down is obeyed. When no item in the shop can satisfy all of that, the correct finish is a decline carried out precisely as the request describes.

Check: you are about to call a tool. Then: remember that each call costs one step, failed or erroring calls included, and steps are capped; thinking costs nothing, steps do.
Check: several calls are independent of each other's results (say, the details of a few listings). Then: send them together in one reply; they execute in sequence. An order or a decline always goes out by itself, never alongside other calls.
Check: the shop offers a tool that returns full details for many listings at once. Then: prefer it to opening listings singly, because it spares steps.
Check: you meet a shop's tools. Then: study each description and JSON schema closely, since names, arguments and result layouts change between shops; pass exactly the argument names and permitted values they define, and work out from the first results which tool reveals price, stock and detailed properties.
Check: you begin searching. Then: learn soon which category labels the shop puts on listings of the requested product type; where neighbouring labels exist, take the one whose label fits the request's wording taken whole, and both search and buy inside it.
Check: a search tool looks restricted (only so many rows or pages shown, ordering by id instead of relevance or price, all words mandatory, filters valid in one mode only). Then: weigh the total it reports against the rows actually visible, and keep narrowing (sharper words, category or price filters, further pages) until the entire relevant set has been seen.

Check: you want to uncover the shopper's unwritten needs. Then: know that each one comes out only when a message asks about that very property, and only the FIRST thing a message asks gets an answer; so ask about ONE property per message, phrased "Which <property> would you like the <product> to have?", using the property name the listings use and the product's plain noun from the request, with no sample values. Do not instruct the shopper, do not mention lists, labels, positions or rules, and do not ask two things in one go.
Check: the request names several distinct products. Then: let every question state which product it concerns, since an open-ended one confuses the shopper, and distribute the questions across the products.
Check: the shopper says they did not follow, or wants your message explained, simplified or reworded. Then: neither explain nor repeat it; the property you raised is probably not one they care about, so move on to another property next (naming the product).
Check: the shopper answers that they don't mind, have no preference or anything will do. Then: treat that property as open. Check: the answer states a preference or a limit. Then: hold it as a hard requirement from that point on, worded exactly as given (numeric limits included).
Check: you are tempted to add a requirement drawn from the story (an age group, a gender, a size, who will use the item). Then: don't; requirements are only what the request states plus what the shopper reveals or changes, and a listing never fails over such details. If one might matter, it may be a hidden need, so ask about it.
Check: a planner is available. Then: open the details of two or three listings of the requested type early (of each type when several distinct products are wanted), because the questions are picked and sent for you once you have; the answers then show up in the conversation with their meaning, so treat every revealed need as a hard requirement and search again combining it with the stated values.
Check: you feel like messaging the shopper. Then: do so only to ask a question (or to answer one the shopper put to you); never message to report findings, a purchase or a decline, because the finishing tool carries that.
Check: you have to ask questions yourself. Then: open two or three listings' details first, then ask about properties the story hints at (who it is for, how it should look, where or how it will be used), then about properties the listings carry that vary between candidates; never ask about anything the request already settles, and never go beyond the message budget.

Check: the request lays down rules for how to work (how many listings to view or compare before choosing, what to verify, what to tell the shopper, what a remark must contain, how to decline). Then: treat them as graded just like requirements and follow each one literally and in full; a rule to view or open N listings means N fully fitting listings, each opened with the details tool.
Check: the shopper asks you something (for instance to confirm a change). Then: expect it to be answered for you while messages remain, and treat the change itself as binding.
Check: the shopper only would like something ("ideally", "would be nice", "if possible"). Then: it is not a requirement; use it solely to decide between items that already satisfy everything, and never let it outrank a cheapest rule.
Check: something can only be learned from the shopper (no price limit given, say, or two wishes that may not both be achievable so the shopper must say which counts more). Then: ask about it before you choose.

Check: the shopper changes things while you work (budget raised or lowered, a requirement dropped, swapped or added, a price rise, an item sold out; one shopper line may carry several). Then: know such lines can turn up inside any tool result (an extra field of sentences) or with an order attempt; each change binds from that moment, so re-test your pick and backups against it and, when the request wants the cheapest or best, choose again over the new acceptable set.
Check: an order attempt comes back. Then: look for a refusal; its reason sits in some field (refusal, complaints, objections or alike) even when the inner result looks successful (accepted: true, placed: true, executed: true). An order that goes through ends the episode, so while the episode runs on nothing has been bought, whatever the result claims; the cart stays as it was. Read the refusal, apply each change it names, take anything no longer acceptable (a sold-out item, for example) out of the cart, add the next verified fit, re-check stock and price, and order again.
Check: you are near an order. Then: remember items can also sell out or reprice silently, so re-check stock and price just before every order attempt; an item whose current price times quantity exceeds the current budget no longer fits.

Check: you are judging whether a listing meets a requirement (this decides the score). Then: count it as met only when the listing's own property for it gives the value. Key spellings vary (spaces, dots, underscores, prefixes) and one property may sit under several keys; any key naming the property counts. Values must hold the whole required value: a single element of a comma-separated list suffices, singular equals plural, spelling variants and case are ignored, and a longer value containing every required word passes. A different word of similar meaning, part of a combined value, a word that merely begins the same way, or a broader or different class does not count (girls is not women; plain unisex is not female).
Check: the listing has no property key at all for a requirement. Then: the requirement is not met; a title or description never stands in for a missing property.
Check: the value sits under a different but related property (a length field for a style requirement, the lining's material for the material). Then: it does not count.
Check: the requirement is numeric. Then: compare as numbers after unit conversion; limits include the boundary value; a range qualifies only if all of it meets the limit.
Check: the request excludes something ("leave out anything with X") or gives a closed list ("A or B and nothing else"). Then: respect it.
Check: you assess the product type. Then: the listing's category settles it, and no property needs to repeat the category name. The item must be the requested type: where the shop labels categories, it must sit in the one whose label matches the request's wording as a whole (a request for "A or B" fits a category named "A & B", not a narrower neighbour named "B"). It must be in stock when the request wants items on hand, and price times quantity must stay within the budget (equal is allowed).
Check: a listing offers options or variants (sizes, colours, versions) with their own ids, prices or stock. Then: treat each option as a candidate of its own, met by its own value, price and stock count, and make the cart line name the chosen option the way the cart tool's schema asks.
Check: a property's value lists several options including the required one (a comma list). Then: it simply matches and is not mixed evidence; evidence is mixed only when the listing's own keys for the property give different, incompatible values, and only then, with no cheapest or best rule deciding, favour a candidate with clean evidence.

Check: the request asks for the cheapest. Then: order the lowest-priced item among ALL items meeting every requirement; one fitting item seen is not enough. Use price filters where a tool has them, lower the price limit until every remaining row shows, sort the visible candidates by price yourself and verify them from cheapest up; the first that fully fits, with every cheaper one checked and rejected, is the answer, and the next fitting ones are your backups.
Check: the request asks for the best by some measure (most, largest, longest, highest and the like). Then: compare ALL fitting items on that measure the same way and take the extreme; an item whose listing gives no value for the measure cannot win.

Check: you are heading for the finish. Then: keep a ranked list of verified fits with at least one backup, since the first pick may sell out at checkout.
Check: you are about to attempt an order. Then: make sure the cart holds exactly the lines the request needs (normally one: the chosen item at the required quantity; one line per item when several different items are requested) and nothing else (remove any other line), and that you re-read the chosen item's stock and price after your last cart change and it is in stock and within budget.
Check: you are considering a decline. Then: recall that it is seldom right, since most tasks have an item meeting every requirement and a wrong decline scores 0; decline only after a thorough search and only when, for every candidate, you can point to a stated requirement, a revealed need, the budget or stock that its own records plainly fail. Doubt, a missing nice-to-have or a merely assumed requirement is no reason to decline; order the best verified fit instead. Follow the request's decline rule exactly, and if it says to offer no substitutes, the decline remark must not name, describe or point to any listing: no ids, no titles, no prices.
Check: you are planning your steps. Then: make your first order attempt while at least seven steps remain, so a refusal can still be dealt with; when 7 steps remain, only the cart, stock-check and finishing tools stay available. Settle once and do not shuffle the cart back and forth, since every cart change costs a step.
Reply: every reply carries at least one tool call, with short text listing your current requirements, the verified fits with price and stock, and your next move."""


SETUP_PROMPT = """Study the shopper's wish and the shop's tool catalogue that follow, and condense them.
Check: you answer. Then: produce a single JSON object written on one line, without indentation or any surrounding words, carrying precisely the keys described here.
Check: key "product". Then: what sort of goods are wanted, phrased as the request phrases it (list each when the shopper wants several different things).
Check: key "quantity". Then: an integer, the count of units to buy from one listing (1 unless more are wanted); goods that are normally sold as a single unit, like a pair of shoes or a set, count as 1.
Check: key "items". Then: only if the shopper wants several DIFFERENT products, one entry each as {"product": its brief name taken from the request, "noun": its plain noun with no property values, "quantity": how many of it (integer)}; otherwise an empty list.
Check: key "requirements". Then: list of strings, a string for each firm condition the request puts on the goods, shaped "property: constraint" and keeping the shopper's precise value, limit, exclusion or closed list; for several different products, open every string with the brief name of the product concerned, as "<product> <property>: constraint". Price, stock, quantity, budgets and simple wishes stay out of it, and so does the product kind or category itself: words that belong to the category's own name (a season, an occasion or an audience named in it) are already handled by the category, not an extra property condition.
Check: key "price_cap". Then: the ceiling on the total price as a number; null if the request names no ceiling.
Check: key "currency". Then: which money unit the request prices in; null if unknown.
Check: key "in_stock_only". Then: true if the shopper only accepts goods that are really available or on hand.
Check: key "priority". Then: "cheapest" solely when the request plainly asks for the lowest-priced option that fits; "best" solely when it plainly asks for the top item on some measure; in all other cases "none". Comments that funds are short or each penny matters do not ask for the cheapest; they only justify the price ceiling.
Check: key "priority_detail". Then: with "best", the yardstick that defines best; with anything else, "".
Check: key "message_budget". Then: the integer number of messages you are permitted to send the shopper; null if the request is silent.
Check: key "step_budget". Then: the integer number of moves or steps permitted by the request; null if it is silent.
Check: key "tool_call_cap". Then: the integer cap the request puts on tool calls, actions or uses of the shop's tools (however it words it); null if it is silent.
Check: key "rules". Then: every working rule the request gives the assistant (how many listings must be looked at, what must be checked or said, the manner of declining, what is forbidden, what must be included), each copied verbatim; things the shopper says they will do themselves are background, not rules.
Check: key "soft_wants". Then: the shopper's preferences that are not strict (signalled by ideally, preferably, would be nice, if possible); never also place them in "requirements".
Check: key "ask_first". Then: points that only the shopper can settle and that should be raised before picking: which of two wishes wins if the request warns both might not be achievable (put both of those wishes in soft_wants and leave them out of requirements), or whatever the request says to check with the shopper; an empty list if nothing applies.
Check: you have placed the keys above. Then: confirm each sentence of the request is accounted for as a requirement, the priority, a soft want, a rule, a budget, a hint, or mere background; if a sentence might be a rule, put it in "rules".
Check: key "pointers". Then: for each condition saying a property must match (or be the same as) the details of ANOTHER listing, {"property": the property that must match, "category": that other listing's kind of goods, "seller": the seller or maker named for it, "traits": the other words that describe it}; an empty list if none. The condition itself still appears among the requirements as "<property>: same as the <category> sold by <seller>", and the describing words are never requirements of their own.
Check: key "decline_no_names". Then: true if the request's rule on declining bans offering, naming or proposing substitutes or alternatives; false if it wants something named; null if it sets no decline rule.
Check: key "roles". Then: an object giving, for each role below, the matching tool's exact name from the catalogue, or null if none suits:
  "search", "details", "view", "message", "cart", "cart_view", "order", "decline".
  "message" means the tool that writes to the shopper; "cart" the tool that puts items into or takes them out of the cart; "order" the tool that purchases or completes checkout; "decline" the tool that closes the task with no purchase.
Check: key "finish_modes". Then: when a single tool both buys and closes without buying, told apart by one argument with fixed allowed values, give {"argument": that argument's name, "order": the allowed value that buys, "decline": the allowed value that ends with no purchase}, each copied exactly from the schema; null when buying and declining are separate tools.
Check: key "noun". Then: the plain noun for the wanted goods, the one a shopper would use for it in a short question ("lamp", "boots"), with no article and without colours, materials or other property values; with several different products, the noun for the first.
Reply: that JSON object and nothing more.
"""

INTERPRET_PROMPT = """A shopping assistant has just received words from the shopper; work out their sense and record it as a single JSON object.
Check: key "meaning". Then: a brief sentence stating the shopper's point.
Check: key "needs". Then: list of firm conditions on the goods that are new in this line, each shaped "property: constraint" with the precise value, limit, exclusion or closed list the shopper used.
Check: key "no_preference". Then: list of properties the shopper declared they are indifferent to.
Check: key "dropped". Then: list of former conditions the shopper has given up.
Check: key "new_budget". Then: the fresh ceiling on the total price as a number; null otherwise.
Check: key "new_quantity". Then: if the shopper alters the number of units, {"product": which product it is for ("" if just one kind is wanted), "quantity": the fresh count as an integer}; null otherwise.
Check: key "unavailable". Then: true if the shopper reports that an item cannot be bought any more (sold out, gone, out of stock, button disabled); false otherwise.
Check: key "requirements_now". Then: every firm condition on the goods as it stands once all the shopper's words are counted, assembled from the conditions so far (shown below) plus what the shopper said outright, each shaped "property: constraint". Do not invent conditions from the background story or from your own take on the request; the only valid sources are values the request gives as requirements and values the shopper gives in a reply. Carry over each former condition unless it was given up or superseded, retain only the latest value of a superseded one, and include each need just revealed. Exclude price, stock, quantity and the product category.
Check: key "pointers". Then: for each condition saying a property must match (or be the same as) the details of ANOTHER listing, {"property": the property that must match, "category": that other listing's kind of goods, "seller": the seller or maker named for it, "traits": the other words that describe it}; an empty list if none. The condition itself still appears among the requirements as "<property>: same as the <category> sold by <seller>", and the describing words are never requirements of their own.
Check: key "unclear". Then: true if the shopper says the assistant's message lost them or was not understood, or wants it explained, made simpler, put another way or said again; false otherwise.
Check: key "reply". Then: if the shopper puts a question to the assistant or wants something confirmed, a brief plain response to send that settles it and states the substance of the change (for example by confirming the new value); "" otherwise. Whenever "unclear" is true it must be "", since asking the assistant to clarify its own question is not a question needing an answer.
Check: the shopper wants several different products. Then: every string in "needs" and "requirements_now" opens with the brief name of the product it concerns, as "<product> <property>: constraint" (shoppers tend to say "<property> of the <product kind>").
Check: a single line holds more than one change. Then: report all of them; a condition superseded by another counts as dropped (the former) and as a new need (the latter).
Check: the shopper picks one of two wishes as more important. Then: record the favoured wish as a need.
Check: the shopper names a budget. Then: it goes in new_budget and supersedes the former one, up or down, even if the shopper frames it as good news.
Check: the shopper merely hopes for something ("ideally", "would be nice"). Then: it is not a need.
Check: the shopper names a value in homely, figurative or indirect language. Then: take the words at face value and in good faith: that value is firmly wanted; regard a value as ruled out only if the shopper plainly says to steer clear of it.
Check: a number arrives with a comparison word (at most, at least, under, over, no more than). Then: it is a limit; keep its unit.
Check: the shopper says nothing more than that they don't mind, have no preference or anything works. Then: put the property the assistant asked about into no_preference; such an answer affects only that property, never removes a stated requirement, a price limit or a rule, and alters nothing if the assistant's message named no property.
Reply: ONE JSON object with these keys and no other text."""

PICK_PROMPT = """A shopping assistant may send the shopper one more question, about ONE property of the product named below; choose which.
Check: the candidates. Then: choose only from the numbered list; each is a property the store's listings carry, with how many listings show it, how many distinct values it takes and some sample values.
Check: the request's story points at one of them (who will use it, the look wanted, where or how it will be used). Then: choose that one.
Check: the earlier questions and answers. Then: a revealed need may point to a related property that is still open; never choose something they settled.
Check: a candidate is about logistics (warranty, condition, package contents, shipping) or merely lists every option on offer. Then: pass it by.
Check: nothing points anywhere. Then: choose number 1, since the list is ranked by how well the property tells the listings apart.
Reply: ONE JSON object and no other text: {"pick": the candidate's number}."""

POINTER_PROMPT = """A shopper needs the PROPERTY of the listing below; read it from the listing's records.
Check: the listing plainly contradicts the describing words given below (another kind of goods, or a stated property with a different value). Then: it is the wrong listing; reply {"wrong_listing": true}.
Check: the listing's own records give that property (any key spelling, or its notes). Then: return that value as written.
Check: its notes say this property is taken from, or should be judged by, another listing instead of this one. Then: return no value and describe that other listing, since the value lives there.
Reply: ONE JSON object and no other text: {"value": the value or null, "go_to": null or {"category": its kind of goods, "seller": the seller or maker named for it, "traits": the other words that describe it}}."""

FIRST_PROMPT = """A shopping assistant has to settle one point with the shopper before choosing an item.
Check: you write the message. Then: one short, plain question in everyday words that settles exactly the point below; when it is a choice between two wishes, name both values with their properties ("If one listing can't have both, would you rather have the <value> <property> or the <value> <property>?"); no greeting, no second question, no list.
Reply: ONE JSON object and no other text: {"question": "..."}."""

AUDIT_PROMPT = """A shopping assistant intends to report that the shop holds no fitting item; before that, test the listing records below against the requirements.
Check: you test a requirement. Then: it holds when the listing's own property shows the value (any key spelling, a single comma-list element, singular/plural and near synonyms all pass; a wider or other class fails), and title or description text matters only when the listing lacks a key for that property.
Check: figures, cost and availability. Then: convert units before comparing, with a limit's edge value passing; the price has to fit the budget and the item may not be shown out of stock.
Check: which requirements to use. Then: just the ones listed, never one guessed from the story; when the shopper wants several different products, test each listing only against the requirements of its own kind.
Check: a listing might fit. Then: include it, since a mistaken decline earns 0; omit a listing only if its own records plainly fail a listed requirement.
Reply: ONE JSON object and no other text: {"possible_fits": [ids of listings that appear to meet every requirement, best first, at most 5], "reason": "one sentence"}."""

_llm = ProxyClient(timeout=70, max_retries=1, retry_delay=1)
_env = ProxyClient(timeout=95, max_retries=2, retry_delay=1)

NEUTRAL_DECLINE = "Sorry, nothing in the store meets all of your requirements, so I am not ordering anything."
STATUS_TAG = "[status]"

PROPERTY_KEY = re.compile(r"fact|propert|trait|spec|attribut|feature|detail|characteristic", re.I)
# names that describe the listing itself (seller, title, shelf...) and never make a planner question
LISTING_META = re.compile(r"\b(brand|maker|manufacturer|seller|vendor|merchant|headline|heading|sku|url|title|made by|sold by|"
                          r"blurb|tagline|caption|teaser|slogan)\b|"
                          r"^(make|shop|shop name|store|store name|outlet|company|category|department|shelf|description|"
                          r"(overview|description|summary)( text)?|notes?|image|images|rating|reviews?|review count|link|"
                          r"options?|variants?|id|name|price|amount|cost|stock|on hand|in stock)$", re.I)
ALLOWANCE_SPENT = re.compile(r"budget_exceeded|quota|allowance|used up|no more\b.{0,40}\b(may|can|allowed|left)|limit reached|"
                             r"reached the limit|maximum (number|reached)|already used", re.I)
IDENT_KEY = re.compile(r"id|ref|sku|code|number|key|listing|product", re.I)
OPTION_KEY = re.compile(r"option|variant|choice|config|version|edition|sku|model", re.I)
KIND_KEY = re.compile(r"categor|department|^dept$|shelf|aisle|section|^kind$|product.?(type|family|group|kind)|"
                      r"item.?(type|kind)|offer.?kind|^family$|^group$", re.I)
TELL_RULE = re.compile(r"\b(tell|inform|let (me|them|the shopper) know|update|confirm|report|mention|explain|message)\b", re.I)
CHECKOUT_STEP = re.compile(r"final|confirm|complete|checkout|pay|place|submit|commit|token", re.I)
PENDING_KEY = re.compile(r"token|pending|await|confirm|next.?step|checkout.?id|session", re.I)
DECLINE_WORDS = re.compile(r"refus|declin|abandon|give.?up|no.?(buy|purchase|order)|reject|cancel|quit|skip", re.I)
ID_SHAPE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-:.]{3,63}$")
CODE_LIKE = re.compile(r"[A-Za-z0-9_\-]+")


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def looks_like_id(text: str) -> bool:
    return bool(ID_SHAPE.match(text)) and any(ch.isdigit() for ch in text)


def is_code(text: str) -> bool:
    return bool(re.search(r"\d", text) and CODE_LIKE.fullmatch(text))


def is_text_type(spec: dict[str, Any]) -> bool:
    return spec.get("type") == "string" or "string" in (spec.get("type") or [])


def string_values(args: dict[str, Any]) -> str:
    return " ".join(str(v) for v in args.values() if isinstance(v, str))


def one_word(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip()) and " " not in value.strip()


def enum_signature(args: dict[str, Any], properties: dict[str, Any]) -> str:
    return dump({k: v for k, v in args.items() if isinstance(properties.get(k), dict) and "enum" in properties[k]})


def shorten(value: Any, limit: int, depth: int = 0) -> Any:
    if depth > 8:
        return None
    if isinstance(value, list):
        return [shorten(v, limit, depth + 1) for v in value]
    if isinstance(value, dict):
        return {k: shorten(v, limit, depth + 1) for k, v in value.items()}
    if isinstance(value, str) and len(value) > limit:
        return value[:limit] + "..."
    return value


def dump(value: Any) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        text = str(value)
    return text if len(text) <= OBS_CHARS else text[:OBS_CHARS] + "...(cut)"


def long_lines(value: Any, depth: int = 0) -> list[str]:
    if depth > 4:
        return []
    if isinstance(value, str):
        return [value] if len(value.split()) >= 4 else []
    found: list[str] = []
    if isinstance(value, list):
        for item in value[:10]:
            found += long_lines(item, depth + 1)
    return found


def dict_nodes(value: Any, deepest: int = 6):
    pending = [(value, 0)]
    while pending:
        node, depth = pending.pop()
        if depth > deepest:
            continue
        if isinstance(node, dict):
            yield node
            children = list(node.values())
        elif isinstance(node, list):
            children = node
        else:
            continue
        pending.extend((child, depth + 1) for child in reversed(children))


def gather_ids(value: Any, into: set[str]) -> None:
    for node in dict_nodes(value):
        into.update(item for key, item in node.items() if isinstance(item, str) and "id" in key.lower() and ID_SHAPE.match(item))


def has_words(value: Any, depth: int = 0) -> bool:
    if depth > 4:
        return False
    if isinstance(value, str):
        return any(ch.isalpha() for ch in value) and not looks_like_id(value)
    if isinstance(value, dict):
        value = list(value.values())
    return isinstance(value, list) and any(has_words(v, depth + 1) for v in value)


def awaits_further_step(observation: dict[str, Any]) -> bool:
    lines = long_lines([v for v in observation.values() if isinstance(v, (list, str))])
    keys = list(observation)
    for inner in observation.values():
        if isinstance(inner, dict):
            keys.extend(inner)
    if not any(PENDING_KEY.search(str(k)) for k in keys):
        return False
    return all(len(line.split()) < 8 for line in lines)


def cart_line_text(args: dict[str, Any], ident: str) -> str:
    for value in args.values():
        if not isinstance(value, list):
            continue
        for line in value:
            if isinstance(line, dict) and ident in line.values():
                return dump(line)
    return ""


def _names_listing(key: Any, value: Any) -> bool:
    return isinstance(value, str) and looks_like_id(value) and bool(IDENT_KEY.search(str(key)))


def cart_line_ids(args: dict[str, Any]) -> list[str] | None:
    if any(_names_listing(k, v) for k, v in args.items()):
        return None             # a direct listing argument: any list next to it holds choices
    line_lists = [v for v in args.values() if isinstance(v, list) and v and all(isinstance(x, dict) for x in v)]
    firsts = (next((v for k, v in line.items() if _names_listing(k, v)), None) for lines in line_lists for line in lines)
    found = [ident for ident in firsts if ident]
    return found or None


def _top_level_last(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    best = None
    position = text.find("{")
    while 0 <= position < len(text):
        try:
            value, after = decoder.raw_decode(text, position)
        except ValueError:
            position = text.find("{", position + 1)
            continue
        balanced = text.count("{", 0, position) == text.count("}", 0, position)
        if balanced and isinstance(value, dict):
            best = value
        position = text.find("{", after)
    return best


def read_json_object(text: Any) -> dict[str, Any] | None:
    if not isinstance(text, str):
        return None
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[A-Za-z]*", "", text).strip()
        if text.endswith("```"):
            text = text[:-3].strip()
    attempts = [text]
    first, last = text.find("{"), text.rfind("}")
    if first >= 0 and last > first:
        attempts.append(text[first : last + 1])
    for attempt in attempts:
        try:
            value = json.loads(attempt)
        except ValueError:
            continue
        return value if isinstance(value, dict) else None
    return _top_level_last(text)


def reply_object(message: dict[str, Any] | None) -> dict[str, Any] | None:
    return read_json_object((message or {}).get("content") or "")


TYPE_TESTS: dict[str, Callable[[Any], bool]] = {
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": is_number,
    "boolean": lambda v: isinstance(v, bool),
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "null": lambda v: v is None,
}
UNCHANGED = object()


def fits_type(value: Any, kind: Any) -> bool:
    test = TYPE_TESTS.get(kind) if isinstance(kind, str) else None
    return test(value) if test else True


def declared_types(schema: dict[str, Any]) -> Any:
    kinds = schema.get("type")
    return [kinds] if isinstance(kinds, str) else kinds


def _to_integer(value: Any) -> Any:
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and re.fullmatch(r"\s*-?\d+(\.0+)?\s*", value):
        return int(float(value))
    return UNCHANGED


def _to_number(value: Any) -> Any:
    if not (isinstance(value, str) and re.fullmatch(r"\s*-?\d+(\.\d+)?\s*", value)):
        return UNCHANGED
    number = float(value)
    return int(number) if number.is_integer() else number


def _to_string(value: Any) -> Any:
    return str(value) if is_number(value) else UNCHANGED


def _to_boolean(value: Any) -> Any:
    if isinstance(value, str) and value.lower() in ("true", "false"):
        return value.lower() == "true"
    return UNCHANGED


CONVERTERS: dict[str, Callable[[Any], Any]] = {"integer": _to_integer, "number": _to_number, "string": _to_string, "boolean": _to_boolean}


def _convert(value: Any, kind: Any) -> Any:
    converter = CONVERTERS.get(kind) if isinstance(kind, str) else None
    if converter is None:
        return UNCHANGED
    try:
        return converter(value)
    except (TypeError, ValueError):
        return UNCHANGED


def coerce(value: Any, schema: dict[str, Any]) -> Any:
    kinds = declared_types(schema)
    if not kinds:
        return value
    if not any(fits_type(value, k) for k in kinds):
        for kind in kinds:
            converted = _convert(value, kind)
            if converted is not UNCHANGED:
                return converted
        return value
    if "integer" in kinds and isinstance(value, float) and value.is_integer():
        return int(value)
    return value


BOUND_RULES: list[tuple[str, Callable[[Any, Any], bool], str]] = [
    ("minimum", lambda v, edge: v < edge, "below minimum {}"),
    ("maximum", lambda v, edge: v > edge, "above maximum {}"),
    ("exclusiveMinimum", lambda v, edge: isinstance(edge, (int, float)) and v <= edge, "must be above {}"),
    ("exclusiveMaximum", lambda v, edge: isinstance(edge, (int, float)) and v >= edge, "must be below {}"),
]


def _enum_value(value: Any, schema: dict[str, Any], path: str, problems: list[str]) -> Any:
    options = schema["enum"]
    if value in options:
        return value
    if isinstance(value, str):
        wanted = value.strip().lower()
        value = next((o for o in options if isinstance(o, str) and o.lower() == wanted), value)
    if value not in options:
        problems.append(f"{path}: must be one of {options}")
    return value


def _scalar_problems(value: Any, schema: dict[str, Any], path: str, problems: list[str]) -> Any:
    if "enum" in schema:
        value = _enum_value(value, schema, path, problems)
    if "const" in schema and value != schema["const"]:
        problems.append(f"{path}: must equal {schema['const']!r}")
    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            problems.append(f"{path}: shorter than {schema['minLength']} characters")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            value = value[: schema["maxLength"]]
        return value
    if is_number(value):
        problems.extend(f"{path}: " + message.format(schema[key]) for key, broken, message in BOUND_RULES
                        if key in schema and broken(value, schema[key]))
    return value


def _list_problems(value: list[Any], schema: dict[str, Any], path: str, problems: list[str]) -> list[Any]:
    if "minItems" in schema and len(value) < schema["minItems"]:
        problems.append(f"{path}: needs at least {schema['minItems']} items")
    if "maxItems" in schema and len(value) > schema["maxItems"]:
        problems.append(f"{path}: allows at most {schema['maxItems']} items")
    inner = schema.get("items")
    if not isinstance(inner, dict):
        return value
    checked = []
    for position, item in enumerate(value):
        item, found = check_against(item, inner, f"{path}[{position}]")
        checked.append(item)
        problems.extend(found)
    return checked


def _object_problems(value: dict[str, Any], schema: dict[str, Any], path: str, problems: list[str]) -> dict[str, Any]:
    properties = schema.get("properties") or {}
    problems.extend(f"{path}.{name}: required" for name in schema.get("required") or [] if name not in value)
    if schema.get("additionalProperties") is False:
        problems.extend(f"{path}.{name}: not an allowed argument" for name in list(value) if name not in properties)
    checked = {}
    for name, item in value.items():
        if name in properties:
            item, found = check_against(item, properties[name], f"{path}.{name}")
            problems.extend(found)
        checked[name] = item
    return checked


def check_against(value: Any, schema: Any, path: str) -> tuple[Any, list[str]]:
    if not isinstance(schema, dict):
        return value, []
    value = coerce(value, schema)
    kinds = declared_types(schema)
    if kinds and not any(fits_type(value, k) for k in kinds):
        return value, [f"{path}: expected {'/'.join(kinds)}"]
    problems: list[str] = []
    value = _scalar_problems(value, schema, path, problems)
    if isinstance(value, list):
        value = _list_problems(value, schema, path, problems)
    if isinstance(value, dict):
        value = _object_problems(value, schema, path, problems)
    return value, problems


class Books(SimpleNamespace):
    def steps_left(self) -> int:
        return self.max_steps - self.step

    def query(self) -> str:
        return str(self.policy.get("query") or "")

    def role(self, name: str) -> str | None:
        return self.roles.get(name)

    def message_limit(self) -> int:
        return 5 if self.message_budget is None else self.message_budget


def open_books(problem_data: dict[str, Any]) -> Books:
    environment = problem_data["environment"]
    policy = environment["policy_view"]
    binding = dict(environment["binding"])
    tools = [t for t in policy.get("tools") or [] if isinstance(t, dict) and isinstance(t.get("function"), dict)]
    blank = {"type": "object", "properties": {}}
    schemas = {tool_name(t): t["function"].get("parameters") or blank for t in tools}
    return Books(
        policy=policy, binding=binding, tools=tools, schemas=schemas,
        max_steps=int(policy.get("max_steps") or 30), finished=bool(policy.get("finished")),
        step=0, dead=False, cost=0.0, plan={}, roles={}, trace=[], messages=[], records=[], seen_ids=set(),
        # what the shopper wants
        stated=[], revealed=[], free=[], dropped=[], current=None, budget_now=None, ledger_version=0,
        # talking to the shopper
        message_budget=None, messages_used=0, heard=set(), said=[], asked=[], questions_sent=[], pending_reply="",
        planner_on=True, sets_at_last_ask=-1, last_unclear=False, plan_misses=0, plan_retry_at=0,
        # the cart and the finish
        cart_used=False, cart_ids=[], cart_lines={}, add_signature=None, last_cart_view=None, cart_checked=set(),
        qty_warned=set(), verdicts={}, order_attempts=0, refusals=0, refused_at=-1, searched_at=-1,
        order_checked=False, decline_checks=0, research_checks=0, early_bounced=False,
        opened=set(), capped=set(), recent=[],
        mode_key=None, order_mode=None, decline_mode=None, passed=set(), swapped=set(),
        asked_props={}, asked_at=-1, first_done=set(), cart_counts={}, pointers=[], order_refused=False,
        binding_fails={}, viewer_tried=False, pointer_needs=[], pointer_checked=set(), budget_held=False, handoff=None, picked={}, basket_holds={}, bounds={},
    )


def tool_name(tool: dict[str, Any]) -> str:
    return tool["function"]["name"]


def name_words(name: str) -> set[str]:
    split_camel = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name)
    return {piece.lower() for piece in re.split(r"[^A-Za-z0-9]+", split_camel) if piece}


CART_WORDS = {"cart", "basket", "bag", "trolley", "stash", "hold", "line", "lines", "keep", "selection"}


LOOK_WORDS = {"view", "show", "list", "review", "status", "current", "my", "see", "contents", "summary"}
EDIT_WORDS = {"add", "put", "set", "remove", "rewrite", "update"}
# each role: alternatives, each a list of word groups that must all be present
ROLE_HINTS: list[tuple[str, list[list[set[str]]]]] = [
    ("cart_view", [[CART_WORDS | {"items", "order"}, LOOK_WORDS]]),
    ("cart", [[CART_WORDS], [EDIT_WORDS, CART_WORDS | {"items", "order", "all"}]]),
    ("order", [[{"checkout", "purchase", "order", "buy", "commit", "confirm", "place", "settle", "pay", "execute", "finalize", "submit"}]]),
    ("message", [[{"message", "shopper", "buyer", "customer", "client", "ask", "chat", "note", "contact", "write", "talk"}]]),
    ("details", [[{"detail", "details", "inspect", "open", "read", "view", "fetch", "get", "datasheet", "info", "spec", "specs", "describe", "examine"}]]),
    ("search", [[{"search", "find", "browse", "locate", "hunt", "query", "lookup", "catalog", "seek", "list"}]]),
]


def _sounds_like_decline(words: set[str]) -> bool:
    if words & {"decline", "refuse", "reject", "abandon", "dismiss", "quit", "giveup"}:
        return True
    return any(w.startswith(("declin", "refus")) for w in words) or {"give", "up"} <= words or "nothing" in words


def _role_fits(role: str, words: set[str], schema: Any) -> bool:
    if role == "decline":
        return _sounds_like_decline(words)
    alternatives = dict(ROLE_HINTS)[role]
    if not any(all(words & group for group in option) for option in alternatives):
        return False
    if role != "message":
        return True
    properties = (schema or {}).get("properties") or {}
    return sum(1 for spec in properties.values() if isinstance(spec, dict) and spec.get("type") == "string" and "enum" not in spec) == 1


def guess_roles(tools: list[dict[str, Any]], schemas: dict[str, Any]) -> dict[str, str]:
    guessed: dict[str, str] = {}
    for role in ["decline"] + [r for r, _ in ROLE_HINTS]:
        for name in (tool_name(t) for t in tools):
            if name not in guessed.values() and _role_fits(role, name_words(name), schemas.get(name)):
                guessed[role] = name
                break
    return guessed


def wanted_products(plan: dict[str, Any]) -> list[tuple[str, int]]:
    raw = plan.get("items")
    products = []
    for entry in raw if isinstance(raw, list) else []:
        if isinstance(entry, str) and entry.strip():
            entry = {"product": entry, "quantity": 1}
        if not (isinstance(entry, dict) and str(entry.get("product") or "").strip()):
            continue
        count = entry.get("quantity")
        if isinstance(count, str) and count.strip().isdigit():
            count = int(count.strip())
        products.append((str(entry["product"]).strip(), int(count) if is_number(count) and count >= 1 else 1))
    return products if len(products) >= 2 else []


NO_PRIORITY_LINE = ("- Check: no cheapest/best rule was read. Then: any item meeting every requirement within budget is correct;"
                    " skip the hunt for the lowest price unless the request itself asks for it")


def _priority_line(plan: dict[str, Any]) -> str:
    priority = plan.get("priority")
    if priority not in ("cheapest", "best"):
        return NO_PRIORITY_LINE
    measure = f" ({plan.get('priority_detail')})" if plan.get("priority_detail") else ""
    return f"- pick the {priority}{measure} of all fitting items"


def opening_brief(books: Books) -> str:
    plan = books.plan
    if not plan:
        return ""
    products = wanted_products(plan)
    hints = [h for h in plan.get("hint_properties") or [] if isinstance(h, str)]
    cap = f"- total price cap: {plan.get('price_cap')} {plan.get('currency') or ''}".rstrip()
    rows: list[str] = [
        "Draft reading of the request (Check: this reading. Then: compare it against the request itself):",
        plan.get("product") and f"- product: {plan.get('product')}",
        "" if products else f"- quantity: {plan.get('quantity') or 1}",
        *(f"- buy: {name}, quantity {count} (its own cart line)" for name, count in products),
        *(f"- stated requirement: {r}" for r in plan.get("requirements") or []),
        "" if plan.get("price_cap") is None else cap,
        plan.get("in_stock_only") and "- in-stock / on-hand items only",
        _priority_line(plan),
        *(f"- optional wish, NOT a requirement (tie-break only): {w}" for w in plan.get("soft_wants") or []),
        *(f"- learn from the shopper before choosing: {a}" for a in plan.get("ask_first") or []),
        *(f"- rule: {r}" for r in plan.get("rules") or []),
        "" if books.message_budget is None else f"- shopper messages allowed: {books.message_budget}",
        f"- step allowance: {books.max_steps}",
        hints and "- properties the story may point to (ask these first, one per message): " + ", ".join(hints),
        *("- " + pointer_note(p) for p in books.pointers),
    ]
    return "\n".join(row for row in rows if row)


def requirements_line(books: Books) -> str:
    if books.current is not None:
        entries = list(books.current)
    else:
        tagged = [(r, "stated") for r in books.stated] + [(r, "shopper revealed") for r in books.revealed]
        entries = [f"{r} ({origin})" for r, origin in tagged if r not in books.dropped]
    products = wanted_products(books.plan)
    budget = "" if books.budget_now is None else f"total at most {books.budget_now} {books.plan.get('currency') or ''}".rstrip()
    amounts = ("products and quantities: " + ", ".join(f"{name} x{count}" for name, count in products)
               if products else f"quantity {books.plan.get('quantity') or 1}")
    parts = ["; ".join(entries) or "as in the request", budget, amounts,
             books.dropped and "dropped: " + "; ".join(books.dropped),
             books.free and "shopper has no preference on: " + ", ".join(books.free)]
    return "; ".join(part for part in parts if part)


def _pace_warning(books: Books, left: int, seconds_left: Callable[[], float]) -> str:
    if left <= 2:
        return " Check: these are the final steps. Then: order if the cart holds a verified fit, else decline."
    if left <= 5 or seconds_left() < WRAP_UP_SECONDS:
        return " Check: time to finish. Then: set the cart to the verified line(s) the request needs, re-check stock, and order (or decline when nothing fits)."
    if books.order_attempts == 0 and left <= 14:
        return (" Check: steps are getting scarce. Then: within a few steps put your best verified fit in the cart and order; search and detail"
                " tools close at 7 remaining steps, and repairing a refused order takes about 4.")
    return ""


def status_line(books: Books, seconds_left: Callable[[], float]) -> str:
    left = books.steps_left()
    budget = "" if books.message_budget is None else f" of {books.message_budget}"
    seconds = int(max(0, seconds_left() - FORCE_SECONDS))
    head = (f"{STATUS_TAG} step {books.step} of {books.max_steps} used, {left} remaining; messages sent {books.messages_used}{budget};"
            f" roughly {seconds} s remaining.\n[requirements now] " + requirements_line(books))
    since = books.step - books.refused_at
    unbought = ""
    if books.refusals and not books.finished and since >= 2:
        # a refused order is easy to misread as a purchase
        unbought = (f" Check: you think something was bought. Then: know it was not; the order try {since} steps back did not end the episode."
                    " Check: the cart holds fitting items within the current budget. Then: call the order tool now; otherwise repair the cart and order"
                    " (decline only when nothing in the store fits any longer).")
    return head + unbought + basket_pace(books) + _pace_warning(books, left, seconds_left)


def basket_pace(books: Books) -> str:
    targets = product_targets(books)
    if len(targets) < 2:
        return ""
    owners = {product_of(books, item) for item in books.cart_ids}
    missing = [noun or product for product, noun in targets if product not in owners]
    # each product gets an even share of the steps left once cart, cart view, order and one repair are set aside
    share = max(2, (books.max_steps - len(targets) - 5) // len(targets))
    due = min(len(targets), books.step // share)
    text = (f" [basket] {len(targets) - len(missing)} of {len(targets)} products have a cart line"
            + (f" (still missing: {', '.join(missing)})" if missing else "") + f"; about {share} steps per product.")
    if len(targets) - len(missing) < due and missing:
        text += f" Check: you are behind. Then: cart the best verified fit for the {missing[0]} now and move to the next product."
    return text


def _closing_roles(books: Books) -> list[str] | None:
    left = books.steps_left()
    if left <= 1:
        return ["order", "decline"]
    if left <= 3:
        return ["order", "decline", "cart"]
    if left <= 7 and (books.order_attempts == 0 or books.refusals > 0):
        return ["order", "decline", "cart", "cart_view", "view"]
    return None


def turn_tools(books: Books) -> list[dict[str, Any]]:
    offered = books.tools
    closing = _closing_roles(books)
    if closing is not None and books.role("order") and books.role("cart"):
        keep = {books.role(r) for r in closing}
        # an unassigned checkout-finishing tool (a confirm after ordering) stays on offer
        assigned = set(books.roles.values())
        keep.update(n for n in map(tool_name, books.tools) if n not in assigned and CHECKOUT_STEP.search(n))
        offered = [t for t in books.tools if tool_name(t) in keep] or offered
    hidden = {books.role("message")} if books.planner_on and books.role("message") else set()
    offered = [t for t in offered if tool_name(t) not in hidden]
    if books.capped:
        offered = [t for t in offered if tool_name(t) not in books.capped] or offered
    return offered


def _short_text(value: Any) -> bool:
    return isinstance(value, str) and 0 < len(value) <= 70


class PropertyTally:
    def __init__(self) -> None:
        self.counts: dict[str, dict[str, int]] = {}
        self.values: dict[str, dict[str, set[str]]] = {}
        self.samples: dict[str, dict[str, list[str]]] = {}
        self.tables: dict[str, int] = {}
        self.fact_sets = 0

    def note(self, group: str, name: str, value: str) -> None:
        per_group = self.counts.setdefault(group, {})
        per_group[name] = per_group.get(name, 0) + 1
        self.values.setdefault(group, {}).setdefault(name, set()).add(value.lower())
        examples = self.samples.setdefault(group, {}).setdefault(name, [])
        if value.lower() not in examples and len(examples) < 5:
            examples.append(value.lower())

    def visit(self, value: Any, depth: int, key: str, group: str) -> None:
        if depth > 6:
            return
        if isinstance(value, list):
            for item in value:
                self.visit(item, depth + 1, key, group)
            return
        if not isinstance(value, dict):
            return
        # a kind label (category, shelf...) splits the menu per kind of product
        group = next((v.strip().lower() for k, v in value.items() if _short_text(v) and KIND_KEY.search(str(k))), "") or group
        pairs = [(k, v) for k, v in value.items() if _short_text(v)]
        holds_table = any(isinstance(v, dict) and PROPERTY_KEY.search(str(k)) for k, v in value.items())
        named_table = len(pairs) >= 2 and bool(PROPERTY_KEY.search(key) or OPTION_KEY.search(key))
        if named_table or (len(pairs) >= 5 and not holds_table):
            kept = 0
            for raw_key, text in pairs:
                if is_code(text) and ("-" in text or "_" in text):
                    continue
                name = re.sub(r"[_.]+", " ", str(raw_key)).strip().lower()
                if LISTING_META.search(name) or KIND_KEY.search(name):
                    continue
                self.note(group, name, text)
                kept += 1
            # ids and titles alone are no fact set, and options of one listing are not separate listings
            if kept >= 2 and not OPTION_KEY.search(key):
                self.fact_sets += 1
                self.tables[group] = self.tables.get(group, 0) + 1
        for name, item in value.items():
            if isinstance(item, (dict, list)):
                self.visit(item, depth + 1, str(name), group)

    def lines(self, group: str, limit: int) -> list[str]:
        counts, values, samples = self.counts.get(group, {}), self.values.get(group, {}), self.samples.get(group, {})

        def rank(name: str) -> tuple[bool, int]:
            # varying properties first, since only they can decide between listings
            return len(values.get(name, ())) < 2, -counts[name]

        out = []
        for name in sorted(counts, key=rank)[:limit]:
            spread = len(values.get(name, ()))
            out.append(f"- {name} (seen in {counts[name]} listings, {spread} distinct values): " + " | ".join(samples.get(name, [])))
        return out

    def fold_into_unlabelled(self) -> None:
        total: dict[str, int] = {}
        for group, per_group in self.counts.items():
            for name, n in per_group.items():
                total[name] = total.get(name, 0) + n
                self.values.setdefault("", {}).setdefault(name, set()).update(self.values[group][name])
                examples = self.samples.setdefault("", {}).setdefault(name, [])
                for sample in self.samples[group][name]:
                    if sample not in examples and len(examples) < 5:
                        examples.append(sample)
        self.counts[""] = total


def property_menu(records: list[Any], several: bool) -> tuple[str, int]:
    tally = PropertyTally()
    for record in records:
        tally.visit(record, 0, "", "")
    by_size = sorted(tally.counts, key=lambda g: -tally.tables.get(g, 0))
    groups = [g for g in by_size if g and tally.tables.get(g, 0) >= 2]
    if len(groups) < 2 or not several:
        tally.fold_into_unlabelled()
        return "\n".join(tally.lines("", 28)), tally.fact_sets
    out = []
    for group in groups[:4]:
        out.append(f"Listings carrying the label \"{group}\" ({tally.tables[group]} with details):")
        out += tally.lines(group, 14)
    if tally.counts.get(""):
        out.append("Listings with no label:")
        out += tally.lines("", 10)
    return "\n".join(out), tally.fact_sets


PRICE_KEY = re.compile(r"price|cost|amount|fee|charge", re.I)
COUNT_WORD = re.compile(r"count|qty|quant|units|stock|number|size|pack", re.I)
STOCK_KEY = re.compile(r"stock|avail|buy|on.?hand", re.I)


def row_price(node: dict[str, Any]) -> Any:
    if is_number(node.get("price")):
        return node["price"]
    # a store may name its price "amount" or "cost": a number under any such key is the price, a plain price key first
    keys = sorted((k for k, v in node.items() if is_number(v) and PRICE_KEY.search(str(k)) and not COUNT_WORD.search(str(k))),
                  key=lambda k: ("price" not in str(k).lower(), "total" in str(k).lower(), len(str(k))))
    return node[keys[0]] if keys else None


def row_stock(node: dict[str, Any]) -> Any:
    flags = [k for k, v in node.items() if isinstance(v, bool)]
    key = flags[0] if len(flags) == 1 else next((k for k in flags if STOCK_KEY.search(str(k))), None)
    return node[key] if key is not None else None


def priced_rows(records: list[Any]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for record in records:
        for node in dict_nodes(record):
            ident = _row_ident(node)
            price = row_price(node)
            if not ident or not is_number(price):
                continue
            merged = dict(rows.get(ident, {}))
            merged.update((k, v) for k, v in node.items() if v is not None)
            merged["price"] = price
            stock = row_stock(node)
            if stock is not None:
                merged["in_stock"] = stock
            kind = next((v for k, v in node.items() if _short_text(v) and KIND_KEY.search(str(k))), None)
            if kind is not None:
                merged["category"] = kind
            rows[ident] = merged
    return rows


def cart_total(books: Books) -> Any:
    rows = priced_rows(books.records)
    wanted = books.plan.get("quantity") if isinstance(books.plan.get("quantity"), int) else 1
    total = 0
    for item in books.cart_ids:
        price = (rows.get(item) or {}).get("price")
        if not is_number(price):
            return None
        total += price * books.cart_counts.get(item, wanted)
    return total if books.cart_ids else None


def setup_content(books: Books) -> list[dict[str, Any]]:
    described = []
    for tool in books.tools:
        function = tool["function"]
        described.append(f"- {function['name']}: {function.get('description', '')} | arguments: {dump(function.get('parameters') or {})}")
    fixed_part = SETUP_PROMPT + "\n\nTOOLS:\n" + "\n".join(described)
    task_part = "REQUEST:\n" + books.query()
    if books.policy.get("opening") is not None:
        task_part += "\n\nOPENING OBSERVATION:\n" + dump(books.policy.get("opening"))
    return [{"type": "text", "text": fixed_part, "cache_control": {"type": "ephemeral"}}, {"type": "text", "text": task_part}]


def _plain_number(plan: dict[str, Any], key: str) -> Any:
    value = plan.get(key)
    return value if is_number(value) else None


def _mode_choices(books: Books) -> list[tuple[str, list[str]]]:
    name = books.role("order")
    if not name or name != books.role("decline"):
        return []
    properties = (books.schemas.get(name) or {}).get("properties") or {}
    choices = []
    for key, spec in properties.items():
        values = [v for v in (spec.get("enum") or []) if isinstance(v, str)] if isinstance(spec, dict) else []
        if len(values) >= 2:
            choices.append((key, values))
    return choices


def learn_finish_modes(books: Books, told: Any) -> None:
    # one tool both buys and declines: the schema's own values decide which call is which
    choices = _mode_choices(books)
    if not choices:
        return
    told = told if isinstance(told, dict) else {}
    key, values = next(((k, v) for k, v in choices if k == told.get("argument")), choices[0])
    order = told.get("order") if told.get("order") in values else None
    decline = told.get("decline") if told.get("decline") in values and told.get("decline") != order else None
    if order is None and decline is None:
        buys = [v for v in values if CHECKOUT_STEP.search(v) and not DECLINE_WORDS.search(v)]
        closes = [v for v in values if DECLINE_WORDS.search(v)]
        order = buys[0] if len(buys) == 1 else None
        decline = closes[0] if len(closes) == 1 else None
    others = [v for v in values if v not in (order, decline)]
    if order is None and decline is not None:
        order = next((v for v in others if CHECKOUT_STEP.search(v)), others[0] if others else None)
    elif decline is None and order is not None and others:
        decline = others[0] if len(others) == 1 else next((v for v in others if DECLINE_WORDS.search(v)), None)
    books.mode_key, books.order_mode, books.decline_mode = key, order, decline
    books.trace.append({"kind": "finish_modes", "argument": key, "order": order, "decline": decline})


def adopt_reading(books: Books, plan: dict[str, Any]) -> None:
    books.plan = plan
    named = plan.get("roles")
    books.roles = {k: v for k, v in named.items() if isinstance(v, str) and v in books.schemas} if isinstance(named, dict) else {}
    if not books.roles:
        # with no roles every role-based guard is off: fall back to the tool names
        books.roles = guess_roles(books.tools, books.schemas)
        books.trace.append({"kind": "guessed_roles", "roles": books.roles})
    messages_allowed = _plain_number(plan, "message_budget")
    if messages_allowed is not None and messages_allowed >= 0:
        books.message_budget = int(messages_allowed)
    books.stated = [str(r) for r in plan.get("requirements") or [] if r]
    books.budget_now = plan.get("price_cap")
    learn_finish_modes(books, plan.get("finish_modes"))
    adopt_pointers(books, plan.get("pointers"))
    # the request's own cap binds when it is tighter than the world's
    for key in ("step_budget", "tool_call_cap"):
        steps_allowed = _plain_number(plan, key)
        if steps_allowed is not None and 0 < steps_allowed < books.max_steps:
            books.max_steps = int(steps_allowed)


def opening_message(books: Books, seconds_left: Callable[[], float]) -> str:
    opening = books.policy.get("opening")
    text = "SHOPPER REQUEST:\n" + books.query()
    if opening is not None:
        text += "\n\nOPENING OBSERVATION:\n" + dump(opening)
    brief = opening_brief(books)
    if brief:
        text += "\n\n" + brief
    done = handoff_note(books)
    if done:
        text += "\n\n" + done
    return text + "\n\n" + status_line(books, seconds_left)


def compose(*parts: str) -> str:
    return "\n\n".join(part for part in parts if part)


def interpret_prompt(books: Books, fresh: list[str], question: str | None) -> str:
    history = "\n".join(books.said[-12:])
    return compose(
        INTERPRET_PROMPT,
        "THE SHOPPER'S REQUEST (context only):\n" + books.query(),
        "REQUIREMENTS UP TO NOW:\n" + requirements_line(books),
        history and "EARLIER TALK WITH THE SHOPPER (oldest first):\n" + history,
        question and "THE ASSISTANT'S QUESTION:\n" + question,
        "WHAT THE SHOPPER SAYS NOW:\n" + "\n".join(fresh),
    )


def audit_prompt(books: Books, blocks: list[str]) -> str:
    return compose(AUDIT_PROMPT, "THE SHOPPER'S REQUEST:\n" + books.query(),
                   "REQUIREMENTS IN FORCE:\n" + requirements_line(books),
                   "LISTING RECORDS SEEN (newest first):\n" + "\n".join(blocks))


def audit_blocks(records: list[Any]) -> list[str]:
    blocks: list[str] = []
    size = 0
    for record in reversed(records):
        text = dump(shorten(record, 160))
        if size + len(text) <= 16000:
            blocks.append(text)
            size += len(text)
    return blocks


def _merge_new(target: list[str], values: Any) -> list[str]:
    added = []
    for value in values or []:
        if isinstance(value, str) and value and value not in target:
            target.append(value)
            added.append(value)
    return added


def _question_named_product(books: Books, question: str | None) -> bool:
    asked_words = {w.rstrip("s") for w in str(question).lower().replace("?", " ").split()}
    filler = ("and", "the", "for", "set", "with", "pair", "pack", "of")
    return any(w.rstrip("s") in asked_words for product, _ in wanted_products(books.plan) for w in product.lower().split()
               if len(w) >= 3 and w not in filler)


def _apply_quantity(books: Books, change: Any, notes: list[str]) -> None:
    if isinstance(change, int) and not isinstance(change, bool):
        change = {"product": "", "quantity": change}
    if not isinstance(change, dict):
        return
    wanted = change.get("quantity")
    if not (isinstance(wanted, int) and not isinstance(wanted, bool) and wanted >= 1):
        return
    if not wanted_products(books.plan):
        books.plan["quantity"] = wanted
        notes.append(f"quantity now {wanted}")
        return
    product = str(change.get("product") or "").strip().lower()
    entries = books.plan.get("items") if isinstance(books.plan.get("items"), list) else []
    for entry in entries:
        if not (isinstance(entry, dict) and product):
            continue
        label = str(entry.get("product") or "").lower()
        if product in label or label in product:
            entry["quantity"] = wanted
            notes.append(f"quantity now {entry.get('product')} x{wanted}")
            return


def pointer_note(p: dict[str, Any]) -> str:
    return (f"the {p['property']} need points to another listing: find the {p['category']} sold by {p['seller']} ({p['traits']}) by searching its kind"
            f" and describing words and paging on, then open the row whose seller is {p['seller']}; code reads its {p['property']} and turns it into a"
            " plain requirement. The words that identify that listing are not requirements for the item you buy")


def adopt_pointers(books: Books, raw: Any) -> list[str]:
    notes = []
    for entry in raw if isinstance(raw, list) else []:
        if not isinstance(entry, dict):
            continue
        prop, seller = str(entry.get("property") or "").strip(), str(entry.get("seller") or "").strip()
        if not prop or not seller or any(p["property"].lower() == prop.lower() and p["origin"].lower() == seller.lower() for p in books.pointers):
            continue
        pointer = {"property": prop, "seller": seller, "origin": seller, "category": str(entry.get("category") or "").strip(),
                   "traits": str(entry.get("traits") or "").strip(), "hops": 0, "done": False, "tried": set()}
        books.pointers.append(pointer)
        notes.append(pointer_note(pointer))
    return notes


def _row_ident(node: dict[str, Any]) -> str | None:
    idents = [(k, v) for k, v in node.items() if isinstance(v, str) and IDENT_KEY.search(k) and looks_like_id(v)]
    return min(idents, key=lambda kv: "id" not in kv[0].lower())[1] if idents else None


def seller_rows(books: Books, pointer: dict[str, Any]) -> list[str]:
    wanted, kind = pointer["seller"].lower(), _stems(pointer["category"])
    found: list[str] = []
    for record in books.records:
        for node in dict_nodes(record):
            ident = _row_ident(node)
            if not ident or ident in found or not any(isinstance(v, str) and v.strip().lower() == wanted for v in node.values()):
                continue
            label = next((v for k, v in node.items() if _short_text(v) and KIND_KEY.search(str(k))), "")
            if kind and label and not _shared_stems(kind, _stems(label)):
                continue
            found.append(ident)
    return found


def item_node(books: Books, item: str) -> dict[str, Any]:
    best: dict[str, Any] = {}
    for record in books.records:
        for node in dict_nodes(record):
            if item in [v for v in node.values() if isinstance(v, str)] and len(node) >= len(best):
                best = node
    return best


def full_record(books: Books, item: str) -> str:
    best = item_node(books, item)
    return dump(best)[:5000] if best else ""


def named_listings(books: Books, item: str) -> list[str]:
    # a listing's own note may send a property to another listing; give the gate that listing's record
    own = item_node(books, item)
    text = dump(own).lower()
    mine = {v.strip().lower() for node in dict_nodes(own) for v in node.values() if isinstance(v, str)}
    found = [r for p in books.pointers if not p["done"] for r in seller_rows(books, p) if r in books.opened and r != item]
    hits: dict[str, int] = {}
    for record in reversed(books.records):
        for node in dict_nodes(record):
            ident = _row_ident(node)
            if not ident or ident == item or ident in found or ident not in books.opened:
                continue
            marks = {v.strip().lower() for v in node.values() if _short_text(v) and len(v.strip()) > 3}
            hits[ident] = max(hits.get(ident, 0), len([m for m in marks if m not in mine and m in text]))
    # the listing a note names shares the most words with it (its seller and its kind, say)
    top = max(hits.values(), default=0)
    return (found + [i for i, n in hits.items() if top and n == top])[:3]


# ---- judge core: the same text runs alone on the arena judge bench ----
JUDGE_CHAIN = ["openai/gpt-oss-120b", "deepseek/deepseek-v4.1-flash", "anthropic/claude-haiku-5.5"]
CHECK_BATCH = 8

CHECK_PROMPT = """You check shop listings against what a shopper needs. JSON only.
Return {"verdicts": {"<listing id>": {"fits": true or false, "failed_need": number of the first need it fails or null, "quote": the listing's own fact text showing the failure, copied exactly, or ""}}} with every listing id given.
A listing fits only if every need is stated by one of its own facts: any key that names the property counts (spelling, spaces, dots, underscores,
prefixes), and when several keys describe the property, one passing is enough. The fact's value must hold the whole required value: spelling variants,
singular/plural and a longer value that contains every required word all pass; a value holding only part of a combined value, a different word of
similar meaning, or a neighbouring class fails. A need may read "at least N unit", "at most N unit", "X or Y" (any one) or "not X"; convert units before
comparing, and a limit's edge passes. Title and description never stand in for a missing fact: a need no fact states fails, with an empty quote.
A note saying a property is taken from another listing sends you to that listing's record under OTHER LISTINGS when one is given."""

PICK_ONE_PROMPT = """One of these shop listings is right for the shopper's needs; the others each miss something. JSON only: {"choice": "<listing id>"}.
Judge on the listings' own properties ("facts"); values match by sense; convert units; a limit's edge passes."""


def squash(text: Any) -> str:
    return "".join(re.findall(r"[a-z0-9]+", str(text).lower()))


def same_key(a: Any, b: Any) -> bool:
    a, b = squash(a), squash(b)
    return a == b or (len(a) > 3 and len(b) > 3 and (a in b or b in a))


def plain_match(have: Any, want: Any) -> bool:
    # code accepts only a plain match; anything looser goes to the LLM
    return squash(have) == squash(want) or any(squash(part) == squash(want) for part in re.split(r"[,/;|]", str(have)) if part.strip())


BOUND_PROMPT = """Some shopper needs below set an amount. For each such need give its limit, JSON only:
{"needs": [{"n": need number, "bound": "at_least" | "at_most" | "equal", "value": number, "unit": the unit as written or "", "forms": [{"value": number, "unit": text}]}]}
"forms" gives the same amount in each other unit or spelling a listing might print. Leave out every need that sets no amount."""
PLAIN_AMOUNT = re.compile(r"\s*(-?\d+(?:\.\d+)?)\s*([^\d\s,;/()]*)\s*")


def add_bounds(ask: Callable[[str, str], Any], needs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # one small reading turns stated amounts into limits code can compare; units come from the reading, never from a table
    if not any(re.search(r"\d", str(n.get("value"))) and not n.get("bound") for n in needs):
        return needs
    reply = ask(BOUND_PROMPT, need_lines(needs)) or {}
    out = [dict(n) for n in needs]
    for entry in reply.get("needs") or []:
        number = entry.get("n") if isinstance(entry, dict) else None
        if isinstance(number, int) and 1 <= number <= len(out) and not out[number - 1].get("bound"):
            out[number - 1].update(bound=entry.get("bound"), amount=entry.get("value"), unit=entry.get("unit"), forms=entry.get("forms") or [])
    return out


def number_cell(listing: dict[str, Any], need: dict[str, Any]) -> tuple[str | None, str]:
    # code decides only a plain "<number> <unit>" fact whose unit text matches one of the need's forms; anything else stays with the LLM
    bound = need.get("bound")
    forms = [(need.get("amount"), need.get("unit"))] + [(f.get("value"), f.get("unit")) for f in need.get("forms") or [] if isinstance(f, dict)]
    forms = [(float(v), squash(u)) for v, u in forms if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if bound not in ("at_least", "at_most", "equal") or not forms:
        return None, ""
    verdict, quote = None, ""
    for key, value in (listing.get("facts") or {}).items():
        found = PLAIN_AMOUNT.fullmatch(str(value)) if same_key(key, need.get("name")) else None
        if not found:
            continue
        have, unit = float(found.group(1)), squash(found.group(2))
        for want, want_unit in forms:
            if want_unit != unit:
                continue
            ok = have >= want if bound == "at_least" else have <= want if bound == "at_most" else abs(have - want) < 1e-9
            if ok:
                return "pass", str(value)
            verdict, quote = "fail", str(value)
    return verdict, quote


def code_cells(listing: dict[str, Any], needs: list[dict[str, Any]]) -> list[bool]:
    facts = listing.get("facts") or {}
    return [any(same_key(key, need.get("name")) and plain_match(value, need.get("value")) for key, value in facts.items()) for need in needs]


def listing_lines(listing: dict[str, Any]) -> dict[str, Any]:
    shown = {"id": listing.get("id"), "title": listing.get("title"), "facts": listing.get("facts")}
    if listing.get("note"):
        shown["note"] = listing["note"]
    if not listing.get("facts"):
        shown["description"] = str(listing.get("description") or "")[:600]
    return shown


def quoted_in(listing: dict[str, Any], quote: str) -> bool:
    wanted = squash(quote)
    facts = listing.get("facts") or {}
    return len(wanted) >= 2 and any(wanted in squash(value) or wanted in squash(f"{key} {value}") for key, value in facts.items())


UNSEEN_NOTE = " (no listing in this shop has a fact for this property: it fails only when the listing's own words contradict it, quoted)"


UNSEEN_SHEETS = 3


def need_lines(needs: list[dict[str, Any]]) -> str:
    return "\n".join(f"{n}. {need.get('name')}: {need.get('value')}{UNSEEN_NOTE if need.get('unseen') else ''}" for n, need in enumerate(needs, 1))


def listing_cost(listing: dict[str, Any], quantity: int) -> float:
    price = listing.get("price")
    return price * quantity if isinstance(price, (int, float)) and not isinstance(price, bool) else float("inf")


def judge_listings(ask: Callable[[str, str], Any], needs: list[dict[str, Any]], listings: list[dict[str, Any]], quantity: int = 1,
                   budget: Any = None, in_stock_only: bool = False, context: str = "") -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    unsure = []
    for listing in listings:
        cost = listing_cost(listing, quantity)
        if in_stock_only and listing.get("in_stock") is False:
            out[listing["id"]] = {"fits": False, "binding": True, "why": "it is out of stock", "passed": 0}
            continue
        if isinstance(budget, (int, float)) and cost != float("inf") and cost > budget:
            out[listing["id"]] = {"fits": False, "binding": True, "why": f"price x quantity {cost:g} is over the budget {budget:g}", "passed": 0}
            continue
        cells = code_cells(listing, needs)
        numbers = [number_cell(listing, need) for need in needs]
        cells = [c or numbers[i][0] == "pass" for i, c in enumerate(cells)]
        failed = next((i for i, c in enumerate(cells) if not c and numbers[i][0] == "fail"), None)
        if failed is not None:
            need = needs[failed]
            out[listing["id"]] = {"fits": False, "binding": True, "passed": sum(cells),
                                  "why": f"{need.get('name')}: {need.get('value')} (its own fact reads \"{numbers[failed][1]}\")"}
            continue
        out[listing["id"]] = {"fits": True if all(cells) else None, "binding": False, "why": "", "passed": sum(cells)}
        if not all(cells):
            unsure.append(listing)
    for start in range(0, len(unsure), CHECK_BATCH):
        chunk = unsure[start:start + CHECK_BATCH]
        user = ("NEEDS:\n" + need_lines(needs) + ("\n\nOTHER LISTINGS:\n" + context if context else "")
                + "\n\nLISTINGS:\n" + json.dumps([listing_lines(c) for c in chunk]))
        reply = ask(CHECK_PROMPT, user) or {}
        verdicts = reply.get("verdicts") if isinstance(reply.get("verdicts"), dict) else reply.get("fits") if isinstance(reply.get("fits"), dict) else {}
        for listing in chunk:
            verdict = verdicts.get(listing["id"])
            verdict = {"fits": verdict} if isinstance(verdict, bool) else verdict
            if not (isinstance(verdict, dict) and isinstance(verdict.get("fits"), bool)):
                continue
            entry = out[listing["id"]]
            if verdict["fits"]:
                entry["fits"] = True
                continue
            number = verdict.get("failed_need")
            need = needs[number - 1] if isinstance(number, int) and 1 <= number <= len(needs) else None
            quote = str(verdict.get("quote") or "").strip()
            if need and need.get("unseen") and not (quote and quoted_in(listing, quote)):
                # the shop shows this property nowhere, so only a quoted contradiction can fail it
                entry["fits"] = True
                continue
            # a failure the listing's own fact shows is binding; one without such a quote stays advisory
            entry.update(fits=False, binding=bool(quote) and quoted_in(listing, quote),
                         why=(f"{need.get('name')}: {need.get('value')}" if need else "a need")
                         + (f" (its own fact reads \"{quote}\")" if quote else " (no fact states it)"))
    return out


def choose_listing(verdicts: dict[str, dict[str, Any]], listings: list[dict[str, Any]], quantity: int, cheapest: bool) -> str | None:
    fits = [c for c in listings if (verdicts.get(c["id"]) or {}).get("fits") is True]
    if not fits:
        return None
    # the cheapest fit when the request asks for it; otherwise the surest fit, then the cheaper one
    if cheapest:
        return min(fits, key=lambda c: listing_cost(c, quantity))["id"]
    return min(fits, key=lambda c: (-(verdicts[c["id"]].get("passed") or 0), listing_cost(c, quantity)))["id"]


def pick_one(ask: Callable[[str, str], Any], needs: list[dict[str, Any]], listings: list[dict[str, Any]]) -> str | None:
    user = "NEEDS:\n" + need_lines(needs) + "\n\nLISTINGS:\n" + json.dumps([listing_lines(c) for c in listings])
    choice = (ask(PICK_ONE_PROMPT, user) or {}).get("choice")
    return choice if choice in {c["id"] for c in listings} else None
FACT_KEY = re.compile(r"fact|propert|trait|spec|attribut|feature|detail|characteristic", re.I)
NAME_KEY = re.compile(r"title|name|heading|caption|headline", re.I)
NOTE_KEY = re.compile(r"note|know|remark|caveat", re.I)
BUSY = (429, 500, 502, 503, 504)


def first_object(text: Any) -> dict[str, Any] | None:
    text = str(text or "")
    start = text.find("{")
    while start >= 0:
        try:
            found = json.JSONDecoder().raw_decode(text[start:])[0]
            if isinstance(found, dict):
                return found
        except ValueError:
            pass
        start = text.find("{", start + 1)
    return None


def chain_ask(post: Callable[[dict[str, Any]], tuple[Any, Any]], chain: list[str] | None = None, max_tokens: int = 4000) -> Callable[[str, str], Any]:
    # post(body) -> (chat completion dict or None, HTTP status or None); a busy provider gets one retry, then the next LLM
    def ask(system: str, user: str) -> Any:
        for model in chain or JUDGE_CHAIN:
            for attempt in range(2):
                body = {"model": model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                        "temperature": 0, "max_tokens": max_tokens}
                if "gpt-oss" in model:
                    body["reasoning"] = {"effort": "low"}
                data, status = post(body)
                message = ((data.get("choices") or [{}])[0].get("message") or {}) if isinstance(data, dict) else {}
                found = first_object(message.get("content"))
                if found is not None:
                    return found
                if attempt or status not in BUSY:
                    break
                time.sleep(1.5)
        return None
    return ask


def listing_from_record(item: str, node: dict[str, Any], price: Any = None, in_stock: Any = None, category: Any = None) -> dict[str, Any]:
    tables = [v for k, v in node.items() if isinstance(v, dict) and FACT_KEY.search(str(k))]
    tables = tables or [v for v in node.values() if isinstance(v, dict) and sum(1 for x in v.values() if isinstance(x, str) and 0 < len(x) <= 70) >= 3]
    best = max(tables, key=len) if tables else {}
    facts = {str(k): str(v) for k, v in best.items() if isinstance(v, (str, int, float)) and not isinstance(v, bool)}
    title = next((v for k, v in node.items() if isinstance(v, str) and NAME_KEY.search(str(k))), "")
    note = " ".join(v for k, v in node.items() if isinstance(v, str) and NOTE_KEY.search(str(k)) and v.strip())
    listing = {"id": item, "title": title, "facts": facts, "price": price, "in_stock": in_stock, "category": category}
    if note:
        listing["note"] = note
    if not facts:
        listing["description"] = max((v for v in node.values() if isinstance(v, str)), key=len, default="")
    return listing


def needs_from_ledger(entries: list[str], product: str = "", products: list[str] | None = None, skip: list[str] | None = None) -> list[dict[str, str]]:
    # "property: constraint" strings; with several products each opens with its product's name
    products, skip = products or [], [s.lower() for s in skip or [] if s]
    needs = []
    for entry in entries:
        head, sep, value = str(entry).partition(":")
        if not sep or not value.strip() or any(mark in str(entry).lower() for mark in skip):
            continue
        name = head.strip()
        owners = [p for p in products if name.lower().startswith(p.lower())]
        if owners:
            owner = max(owners, key=len)
            if owner != product:
                continue
            name = name[len(owner):].strip() or name
        needs.append({"name": name, "value": value.strip()})
    return needs


def basket_check(lines: list[dict[str, Any]], products: list[dict[str, Any]], budget: Any = None,
                 cheaper: dict[str, dict[str, Any]] | None = None) -> list[str]:
    # lines: {id, product, quantity, price, verified, binding, in_stock_problem}; products: {name, noun, quantity}
    problems: list[str] = []
    total, priced = 0.0, True
    cheaper = cheaper or {}
    for product in products:
        label = product.get("noun") or product.get("name") or "item"
        mine = [line for line in lines if line.get("product") == product.get("name")]
        if not mine:
            problems.append(f"no cart line yet for the {label}")
            continue
        if len(mine) > 1:
            problems.append(f"{len(mine)} lines for the {label} ({', '.join(str(m['id']) for m in mine)}); keep only one")
        line = mine[-1]
        want, have = product.get("quantity") or 1, line.get("quantity")
        if isinstance(have, (int, float)) and have != want:
            problems.append(f"{line['id']} is set to {have} but the {label} is wanted {want} times")
        price = line.get("price")
        if isinstance(price, (int, float)) and not isinstance(price, bool):
            total += price * (have if isinstance(have, (int, float)) else want)
        else:
            priced = False
        if line.get("binding"):
            problems.append(str(line["binding"]))
            continue
        if line.get("verified") is False:
            problems.append(f"{line['id']} is not confirmed to meet every need of the {label}; check its own facts or choose a verified fit")
        if line.get("in_stock_problem"):
            problems.append(f"{line['id']} shows as not in stock; re-check it or choose an in-stock fit")
        swap = cheaper.get(product.get("name")) or {}
        if swap.get("fit"):
            problems.append(f"for the {label}, {swap['fit']} also meets every need and costs less than {line['id']}: swap to it")
        elif swap.get("unopened"):
            problems.append(f"for the {label}, cheaper in-stock rows {', '.join(swap['unopened'])} were never opened: open them first")
    if len(products) > 1:
        names = {p.get("name") for p in products}
        problems += [f"{line['id']} matches none of the wanted products; take it out of the cart" for line in lines if line.get("product") not in names]
    if isinstance(budget, (int, float)) and not isinstance(budget, bool) and priced and lines and total > budget:
        problems.append(f"the lines add up to {total:g} against the budget {budget:g}: search for cheaper fits product by product (a lower"
                        " total is the fix, never a decline)")
    return problems
# ---- end judge core ----


def listing_view(books: Books, item: str) -> dict[str, Any] | None:
    # a store can split one listing over several tools (offer, spec sheet): facts come from whichever record carries them
    nodes = [node for record in books.records for node in dict_nodes(record) if item in [v for v in node.values() if isinstance(v, str)]]
    if not nodes:
        return None
    row = priced_rows(books.records).get(item) or {}
    views = [listing_from_record(item, node, row.get("price"), row.get("in_stock"), row.get("category")) for node in nodes]
    listing = max(views, key=lambda v: (len(v["facts"]), len(str(v.get("title") or ""))))
    listing["title"] = listing["title"] or next((v["title"] for v in views if v["title"]), "") or str(row.get("title") or "")
    notes = list(dict.fromkeys(v["note"] for v in views if v.get("note")))
    if notes:
        listing["note"] = " ".join(notes)
    return listing


def ledger_entries(books: Books) -> list[str]:
    if books.current is not None:
        return [r for r in books.current if isinstance(r, str)]
    return [r for r in books.stated + books.revealed if isinstance(r, str) and r not in books.dropped]


def product_quantity(books: Books, product: str) -> int:
    wanted = dict(wanted_products(books.plan)).get(product)
    single = books.plan.get("quantity")
    return wanted or (single if isinstance(single, int) and not isinstance(single, bool) and single >= 1 else 1)


def needs_for(books: Books, product: str) -> list[dict[str, str]]:
    waiting = [p["origin"] for p in books.pointers if not p["done"]]
    # a need that only points at another listing names no value a listing can show: it stays unknown rather than failing every candidate
    entries = [e for e in ledger_entries(books) if e not in books.pointer_needs]
    return needs_from_ledger(entries, product, [name for name, _ in wanted_products(books.plan)], waiting)


POINTER_NEEDS_PROMPT = """Each line below is one condition a shopper has set. Most state a value a listing can show ("color: red", "width: at most 30 in").
Some instead point at a different listing to take the value from or to differ from, such as "material: same as the lamp Zeno sells" or "size: not what Zeno's boots say, the plain black ones"; such a line names no value of its own.
Answer in JSON only: {"pointers": [numbers of the lines that point at another listing]}, or {"pointers": []}."""


def mark_pointer_needs(shell: Shell, books: Books) -> None:
    # needs revealed in talk can point at another listing too: each ledger line is read once, and pointing lines stay out of the judge
    fresh = [e for e in ledger_entries(books) if e not in books.pointer_checked]
    if not fresh or shell.seconds_left() < WRAP_UP_SECONDS + 20:
        return
    books.pointer_checked.update(fresh)
    reply = judge_ask(shell)(POINTER_NEEDS_PROMPT, "\n".join(f"{n}. {e}" for n, e in enumerate(fresh, 1)))
    marked = [fresh[n - 1] for n in (reply or {}).get("pointers") or [] if isinstance(n, int) and 1 <= n <= len(fresh)]
    if len(marked) < len(fresh) or len(fresh) == 1:
        _merge_new(books.pointer_needs, marked)


def bounded_needs(shell: Shell, books: Books, product: str) -> list[dict[str, Any]]:
    mark_pointer_needs(shell, books)
    key = f"{product}|{books.ledger_version}|{len(books.pointer_needs)}"
    if key not in books.bounds:
        books.bounds[key] = add_bounds(judge_ask(shell), needs_for(books, product))
    return [dict(need, unseen=True) if unseen_property(books, product, need) else need for need in books.bounds[key]]


def unseen_property(books: Books, product: str, need: dict[str, Any]) -> bool:
    # a need can block only on a property the shop's listings carry: with several listings opened and none naming it, it is unseen
    sheets = [view.get("facts") or {} for view in (listing_view(books, item) or {} for item in books.opened)
              if view.get("facts") and (not product or product_of(books, view["id"]) == product)]
    asked = _stems(str(need.get("name")).replace("_", " "))
    # spellings of one property share their start ("colour", "color"): such a key counts as carrying it
    near = lambda key: same_key(key, need.get("name")) or any(a[:4] == b[:4] and len(a) >= 4 for a in asked for b in _stems(str(key).replace("_", " ")))
    return len(sheets) >= UNSEEN_SHEETS and not any(near(key) for facts in sheets for key in facts)


def product_of(books: Books, item: str, rows: dict[str, dict[str, Any]] | None = None) -> str:
    targets = product_targets(books)
    if len(targets) < 2:
        return targets[0][0] if targets else ""
    row = (rows if rows is not None else priced_rows(books.records)).get(item) or {}
    label = " ".join(str(x) for x in (row.get("category"), row.get("title"), item_node(books, item).get("title")) if x)
    scored = sorted(((_shared_stems(_stems(name + " " + noun), _stems(label)), -index, name) for index, (name, noun) in enumerate(targets)), reverse=True)
    return scored[0][2] if scored and scored[0][0] > 0 else ""


def judge_ask(shell: Shell) -> Callable[[str, str], Any]:
    def ask(system: str, user: str) -> Any:
        return reply_object(shell.infer([{"role": "system", "content": system}, {"role": "user", "content": user}], None, JUDGE_CHAIN, 4000))
    return ask


def settle_pointer(books: Books, pointer: dict[str, Any], value: str) -> str:
    origin = pointer["origin"].lower()
    texts = (books.current or []) + books.revealed + books.stated
    stale = [r for r in texts if origin in r.lower()]
    prefix = ""
    if wanted_products(books.plan) and stale and pointer["property"].lower() in stale[0].lower():
        prefix = stale[0][: stale[0].lower().index(pointer["property"].lower())]
    new = f"{prefix}{pointer['property']}: {value}"
    if books.current is not None:
        books.current = [r for r in books.current if r not in stale] + [new]
    books.revealed = [r for r in books.revealed if r not in stale] + [new]
    _merge_new(books.dropped, stale)
    books.ledger_version += 1
    return new


def resolve_pointers(shell: Shell, books: Books) -> str:
    notes = []
    for pointer in books.pointers:
        if pointer["done"] or shell.seconds_left() < WRAP_UP_SECONDS:
            continue
        for item in seller_rows(books, pointer):
            if item in pointer["tried"] or item not in books.opened:
                continue
            pointer["tried"].add(item)
            prompt = compose(POINTER_PROMPT.replace("PROPERTY", pointer["property"]),
                             f"DESCRIBING WORDS: {pointer['category']} sold by {pointer['seller']}, {pointer['traits']}", f"LISTING {item}:\n" + full_record(books, item))
            result = shell.judge(prompt, 1500, [READ_MODEL, "openai/gpt-oss-120b", LAST_RESORT]) or {}
            if result.get("wrong_listing") is True:
                continue
            value, onward = result.get("value"), result.get("go_to")
            if isinstance(value, (str, int, float)) and not isinstance(value, bool) and str(value).strip():
                pointer["done"] = True
                new = settle_pointer(books, pointer, str(value).strip())
                notes.append(f"Pointer need settled from {item}: the requirement is now \"{new}\"; the words that named {item} are no requirement.")
                break
            if isinstance(onward, dict) and str(onward.get("seller") or "").strip() and pointer["hops"] < 2:
                pointer.update(seller=str(onward["seller"]).strip(), category=str(onward.get("category") or pointer["category"]).strip(),
                               traits=str(onward.get("traits") or "").strip(), hops=pointer["hops"] + 1)
                notes.append(f"{item} takes its {pointer['property']} from another listing, so the value lives there: " + pointer_note(pointer) + ".")
                break
    return " ".join(notes)


def pointer_rows_note(books: Books) -> str:
    notes = []
    for pointer in books.pointers:
        rows = [r for r in seller_rows(books, pointer) if r not in books.opened][:3]
        if not pointer["done"] and rows:
            notes.append(f"Rows {', '.join(rows)} are sold by {pointer['seller']}: open them to read the {pointer['property']} the pointer need asks for.")
    return " ".join(notes)


def absorb_interpretation(books: Books, result: dict[str, Any], fresh: list[str], question: str | None) -> str:
    notes: list[str] = []
    now = result.get("requirements_now")
    if isinstance(now, list) and all(isinstance(r, str) for r in now):
        books.current = [r for r in now if r.strip()]
    notes += [f"new requirement: {n}" for n in _merge_new(books.revealed, result.get("needs"))]
    _merge_new(books.free, result.get("no_preference"))
    notes += adopt_pointers(books, result.get("pointers"))
    notes += [f"no longer needed: {d}" for d in _merge_new(books.dropped, result.get("dropped"))]
    budget = result.get("new_budget")
    if is_number(budget) and budget > 0 and budget != books.budget_now:
        books.budget_now = budget
        notes.append(f"budget now {budget}")
        total = cart_total(books)
        if total is not None:
            notes.append(f"the cart lines' latest store prices add up to {total:g} (store units) against the new budget {budget:g};"
                         " Check: that is over the budget. Then: swap lines for fits that keep the total within it: search again with the requirement"
                         " words, page on and open the cheaper rows; a lower budget is never a reason to decline")
    _apply_quantity(books, result.get("new_quantity"), notes)
    if result.get("unavailable"):
        held = ", ".join(books.cart_ids[-4:])
        notes.append("an item can no longer be bought; take it out of the cart and never order it. Then: look for a twin: search again with the"
                     " requirement words and the category" + (f" and the key property values of the cart line(s) {held}" if held else "")
                     + ", open the top rows, and cart the cheapest one that meets every requirement")
    if notes:
        books.ledger_version += 1
    reply = str(result.get("reply") or "").strip()
    unclear = result.get("unclear") is True and question is not None
    books.last_unclear = unclear
    if unclear and wanted_products(books.plan) and not _question_named_product(books, question):
        # with several products an unnamed question is ambiguous, so the property may still count
        notes.append("Check: the shopper lost track because the question named no product. Then: if messages remain, ask that property once more with the product named")
    elif unclear:
        notes.append("Check: the shopper did not follow the question. Then: neither explain nor repeat it; treat that property as most likely free and, if messages remain, ask a different one with the product named")
    elif reply and books.roles.get("message"):
        # the reader drafts a reply only when the shopper asked for something
        books.pending_reply = reply
    meaning = str(result.get("meaning") or "")
    return "Shopper meant: " + meaning + (" (" + "; ".join(notes) + ")" if notes else "")


def message_arguments(books: Books, text: str) -> dict[str, Any] | None:
    schema = books.schemas.get(books.role("message") or "") or {}
    properties = schema.get("properties") or {}
    required = list(schema.get("required") or [])
    texts = [k for k, v in properties.items() if isinstance(v, dict) and is_text_type(v) and "enum" not in v]
    slot = next((k for k in required if k in texts), None) or (texts[0] if len(texts) == 1 else None)
    if not slot or any(k != slot for k in required):
        return None
    return {slot: text}


def quantity_problem(books: Books, name: str, args: dict[str, Any]) -> str:
    wanted = books.plan.get("quantity")
    if not isinstance(wanted, int) or isinstance(wanted, bool) or wanted < 1:
        return ""
    properties = books.schemas[name].get("properties") or {}
    if books.add_signature is not None and enum_signature(args, properties) != books.add_signature:
        return ""
    for key, spec in properties.items():
        if not (isinstance(spec, dict) and spec.get("type") == "integer" and "enum" not in spec):
            continue
        value = args.get(key, spec.get("default", 1))
        products = wanted_products(books.plan)
        if products:
            target = _cart_target(args, properties)
            product = product_of(books, target) if target else ""
            want = dict(products).get(product)
            if want and isinstance(value, int) and value != want:
                return f"this line is for the {product}, which the request wants {want} of, yet this call sets {value}."
            return ""
        if isinstance(value, int) and value != wanted:
            return f"the request calls for {wanted} units of a single listing, yet this call sets {value}."
    return ""


def is_decline(books: Books, name: str, args: dict[str, Any]) -> bool:
    if name != books.role("decline"):
        return False
    if books.role("order") != name:
        return True
    if books.mode_key and books.decline_mode is not None:
        return args.get(books.mode_key) == books.decline_mode
    properties = (books.schemas.get(name) or {}).get("properties") or {}
    return any(isinstance(v, str) and "enum" in (properties.get(k) or {}) and DECLINE_WORDS.search(v) for k, v in args.items())


def with_mode(books: Books, name: str, args: dict[str, Any], declining: bool) -> dict[str, Any]:
    if not books.mode_key or name != books.role("order") or name != books.role("decline"):
        return args
    mode = books.decline_mode if declining else books.order_mode
    return {**args, books.mode_key: mode} if mode is not None else args


def _filler(spec: dict[str, Any], text: str, declining: bool) -> Any:
    if "enum" in spec and spec["enum"]:
        # one tool for both finishes: choose the mode that matches
        matching = [v for v in spec["enum"] if isinstance(v, str) and bool(DECLINE_WORDS.search(v)) == declining]
        return (matching or spec["enum"])[0]
    if is_text_type(spec):
        return text[: spec.get("maxLength", len(text))]
    kind = spec.get("type")
    if kind == "boolean":
        return True
    if kind in ("integer", "number"):
        return max(1, spec.get("minimum", 1))
    return UNCHANGED


def finish_arguments(books: Books, name: str, text: str, declining: bool = False) -> dict[str, Any]:
    schema = books.schemas.get(name) or {}
    properties = schema.get("properties") or {}
    args: dict[str, Any] = {}
    for key in schema.get("required") or []:
        value = _filler(properties.get(key) or {}, text, declining)
        if value is not UNCHANGED:
            args[key] = value
    args = with_mode(books, name, args, declining)
    if len(properties) != 1:
        return args
    for key, spec in properties.items():
        free_text = isinstance(spec, dict) and spec.get("type") == "string" and "enum" not in spec
        if free_text and key not in args:
            args[key] = text[: spec.get("maxLength", len(text))]
    return args


def scrub_decline(books: Books, args: dict[str, Any], name: str) -> dict[str, Any]:
    rule = books.plan.get("decline_no_names")
    if rule is False:
        return args
    mentions_listing = any(token and token in books.seen_ids
                           for value in args.values() if isinstance(value, str)
                           for token in re.split(r"[\s.,;:!?()\[\]{}<>\"'`]+", value))
    if not (rule or mentions_listing):
        return args
    properties = (books.schemas.get(name) or {}).get("properties") or {}
    cleaned = {}
    for key, value in args.items():
        spec = properties.get(key) or {}
        cleaned[key] = NEUTRAL_DECLINE[: spec.get("maxLength", len(NEUTRAL_DECLINE))] if isinstance(value, str) and "enum" not in spec else value
    return cleaned


def _allowance_note(books: Books, name: str, observation: dict[str, Any]) -> str:
    problem = " ".join(str(observation.get(k) or "") for k in ("error", "detail", "message", "reason") if isinstance(observation.get(k), str))
    core = {books.role(r) for r in ("order", "decline", "cart", "cart_view")} - {None}
    if not (observation.get("error") and ALLOWANCE_SPENT.search(problem)) or name in core:
        return ""
    if name == books.role("message"):
        books.message_budget = books.messages_used if books.message_budget is None else min(books.message_budget, books.messages_used)
    books.capped.add(name)
    return "Check: this tool's allowance is spent. Then: it is withdrawn, so never call it again; take prices and stock from search results or the cart view and decide with what you already know."


def _refusal_notes(books: Books, refusal: Any) -> list[str]:
    books.refusals += 1
    books.refused_at = books.step
    books.order_checked = False
    return ["Check: THE ORDER WAS REFUSED. Then: know the episode goes on with the cart as it was; apply each change the refusal names, repair"
            " the cart (drop any line no longer acceptable, such as a sold-out item), re-check the new pick's stock and price, and order again."
            " A new price alone never makes an item unacceptable: while price times quantity stays within the budget and nothing is reported"
            " gone, order the same cart again."]


def _change_note(observation: dict[str, Any]) -> str:
    lines: list[str] = []
    for value in observation.values():
        if isinstance(value, list) and value and all(isinstance(v, str) for v in value):
            lines += long_lines(value)
    return "Check: this result holds a shopper message (a change). Then: apply it: " + " | ".join(lines[:4]) if lines else ""


def event_parts(observation: Any) -> tuple[Any, list[str]]:
    # an event rides on any result as {<payload key>: result, <lines key>: [sentences]}, under changing key names
    if not isinstance(observation, dict) or len(observation) < 2:
        return observation, []
    lists = [k for k, v in observation.items() if isinstance(v, list) and v and all(isinstance(x, str) for x in v)
             and any(len(x.split()) >= 4 for x in v)]
    if len(lists) != 1:
        return observation, []
    rest = {k: v for k, v in observation.items() if k != lists[0]}
    return (next(iter(rest.values())) if len(rest) == 1 else rest), [x for x in observation[lists[0]] if x.strip()]


def observation_notes(books: Books, name: str, observation: Any) -> str:
    if isinstance(observation, (dict, list)):
        gather_ids(observation, books.seen_ids)
    if not isinstance(observation, dict):
        return ""
    notes = [_allowance_note(books, name, observation)]
    finishing = name in {books.role("order"), books.role("decline")} - {None}
    lines = event_parts(observation)[1]
    if "refusal" in observation or (finishing and lines):
        notes += _refusal_notes(books, observation.get("refusal") if "refusal" in observation else lines)
    elif len(observation) >= 2 and name != books.role("message"):
        notes.append(_change_note(observation))
    return " ".join(n for n in notes if n)


def shopper_lines(books: Books, name: str, args: dict[str, Any], observation: Any) -> tuple[list[str], str | None]:
    heard: list[str] = []
    if name == books.role("message"):
        if isinstance(observation, str):
            heard = [observation]
        elif isinstance(observation, list) or (isinstance(observation, dict) and "error" not in observation):
            if isinstance(observation, dict):
                heard = long_lines([v for v in observation.values() if isinstance(v, (str, list))])
            if not heard and has_words(observation):
                # a terse or structured answer is still the shopper speaking
                heard = [dump(observation)[:600]]
        return heard, string_values(args)
    if isinstance(observation, dict):
        # a listing's own record (it carries an id) is never the shopper talking, however short it is
        listing = any(_names_listing(k, v) for k, v in observation.items())
        wrapped = len(observation) <= 3 and any(isinstance(v, (dict, list)) for v in observation.values()) and not listing
        for key, value in observation.items():
            if isinstance(value, list) and value and all(isinstance(v, str) for v in value):
                heard += long_lines(value)
            elif isinstance(value, str) and len(value.split()) >= 4 and "refusal" in observation:
                heard.append(value)
            elif isinstance(value, str) and wrapped and len(value.split()) >= 6 and not NAME_KEY.search(str(key)):
                heard.append(value)     # prose framing a structured payload
    return heard, None


def request_body(model: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, max_tokens: int) -> dict[str, Any]:
    sent = messages
    if model.startswith("anthropic/") and len(messages) > 2:
        sent = [dict(m) for m in messages]
        for index in (0, len(sent) - 1):
            text = sent[index].get("content")
            if isinstance(text, str) and text:
                sent[index]["content"] = [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]
    body: dict[str, Any] = {"model": model, "messages": sent, "temperature": 0, "max_tokens": max_tokens}
    if "gpt-oss" in model:
        # left at its default it reasons past the token limit and returns nothing
        body["reasoning"] = {"effort": "low"}
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"
    return body


def usable_message(books: Books, data: Any) -> dict[str, Any] | None:
    if not (isinstance(data, dict) and data.get("choices")):
        return None
    usage = data.get("usage") or {}
    if isinstance(usage.get("cost"), (int, float)):
        books.cost += float(usage["cost"])
    message = data["choices"][0].get("message") or {}
    return message if message.get("content") or message.get("tool_calls") else None


def world_failure(books: Books, status: Any, body: str) -> str:
    if status == 402:
        books.dead = True
        return "stop"
    if status == 409:
        if "finished" in body:
            books.finished = True
            return "stop"
        expected = re.search(r"expected (\d+)", body)
        if not expected:
            return "stop"
        books.step = int(expected.group(1)) - 1
        return "resync"
    return "stop" if status in (403, 422, 500, 503, 504) else "again"


class Shell:
    def __init__(self) -> None:
        self.started = time.monotonic()
        self.books: Books | None = None

    def seconds_left(self) -> float:
        return TASK_SECONDS - (time.monotonic() - self.started)

    def elapsed(self) -> float:
        return round(time.monotonic() - self.started, 1)

    def _attempt(self, model: str, left: float, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, max_tokens: int) -> Any:
        _llm.timeout = int(max(8, min(75, left - FORCE_SECONDS + 15)))
        return _llm.post_verbose("/inference/chat/completions", request_body(model, messages, tools, max_tokens))

    def infer(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, models: list[str], max_tokens: int) -> dict[str, Any] | None:
        for model in models:
            for attempt in range(2):
                left = self.seconds_left()
                if left < FORCE_SECONDS - 10:
                    return None
                result = self._attempt(model, left, messages, tools, max_tokens)
                message = usable_message(self.books, result.data)
                if message is not None:
                    return message
                failure = str(result.error)[:300] if result.error else "empty"
                self.books.trace.append({"kind": "inference_failure", "model": model, "error": failure})
                status = result.error.get("status") if isinstance(result.error, dict) else None
                # a busy provider often answers a moment later; anything else moves to the next LLM
                if attempt or status not in (429, 500, 502, 503, 504) or self.seconds_left() < FORCE_SECONDS + 20:
                    break
                time.sleep(1.5)
        return None

    def judge(self, prompt: str, max_tokens: int, models: list[str] | None = None) -> dict[str, Any] | None:
        models = models or JUDGE_CHAIN + [LAST_RESORT]
        return reply_object(self.infer([{"role": "user", "content": prompt}], None, models, max_tokens))

    def call_world(self, actions: list[dict[str, Any]]) -> dict[str, Any] | None:
        books = self.books
        payload: dict[str, Any] = dict(books.binding)
        payload["step"] = books.step + 1
        payload["action" if len(actions) == 1 else "actions"] = actions[0] if len(actions) == 1 else actions
        for _ in range(3):
            left = self.seconds_left()
            if left < 2:
                return None
            _env.timeout = int(max(3, min(95, left - 1)))
            result = _env.post_verbose("/environment/call", payload)
            data = result.data
            if isinstance(data, dict):
                books.step = data["step"] if isinstance(data.get("step"), int) else books.step + len(actions)
                if data.get("finished"):
                    books.finished = True
                return data
            error = result.error or {}
            status, body = error.get("status"), str(error.get("body") or "")
            books.trace.append({"kind": "env_failure", "status": status, "body": body[:300]})
            outcome = world_failure(books, status, body)
            if outcome == "stop":
                return None
            if outcome == "resync":
                payload["step"] = books.step + 1
        return None


SKIP = "Not run (no step used): "
HOLD = "Not run yet (no step used). "


def post_replies(shell: Shell, books: Books, replies: list[tuple[str, str]]) -> list[str]:
    texts = []
    for index, (call_id, text) in enumerate(replies):
        if index == len(replies) - 1:
            text += "\n" + status_line(books, shell.seconds_left)
        books.messages.append({"role": "tool", "tool_call_id": call_id, "content": text})
        texts.append(text)
    return texts


def joined(note: str, extra: str) -> str:
    return (note + " " if note else "") + extra if extra else note


def interpret(shell: Shell, books: Books, lines: list[str], question: str | None) -> str:
    # the same plain answer to a new question is new information
    fresh = [line for line in lines if f"{question}|{line}" not in books.heard]
    if not fresh:
        return ""
    if shell.seconds_left() < FORCE_SECONDS + 12:
        return ""
    books.heard.update(f"{question}|{line}" for line in fresh)
    prompt = interpret_prompt(books, fresh, question)
    books.said.append((f"we asked: {question} -> " if question else "") + "shopper said: " + " / ".join(fresh))
    result = shell.judge(prompt, 3000, [READ_MODEL, "openai/gpt-oss-120b", LAST_RESORT])
    return absorb_interpretation(books, result, fresh, question) if result else ""


def verify_item(shell: Shell, books: Books, item: str, purpose: str, line: str = "") -> str:
    if shell.seconds_left() < WRAP_UP_SECONDS + 20:
        return ""
    named = named_listings(books, item)
    key = f"{item}|{line}|{books.ledger_version}|{','.join(named)}"
    if key in books.verdicts:
        return books.verdicts[key]
    view = listing_view(books, item)
    if not view:
        return ""
    product = product_of(books, item)
    context = "\n".join(full_record(books, n)[:2500] for n in named)
    # budget and stock stay with the basket gate: a misread unit or flag must never block a listing for good
    verdict = judge_listings(judge_ask(shell), bounded_needs(shell, books, product), [view], product_quantity(books, product), None, False,
                             context).get(item) or {}
    books.trace.append({"kind": "verify", "item": item, "purpose": purpose, "verdict": verdict})
    text = ""
    stamp = f"{item}|{books.ledger_version}"
    if verdict.get("fits") is True:
        books.passed.add(stamp)
    elif verdict.get("fits") is False and verdict.get("binding"):
        text = (f"Check: {item} fails {verdict.get('why')} by its own record. Then: this is final; never cart or order it, and choose another"
                " listing that meets every need.")
        books.binding_fails[stamp] = text
    elif verdict.get("fits") is False:
        text = (f"Check: an independent review finds {item} short: {verdict.get('why')}. Then: pick another fully verified fit, unless you are"
                " sure the review is mistaken.")
    books.verdicts[key] = text
    return text


def audit_fits(shell: Shell, books: Books) -> tuple[list[str], str]:
    if shell.seconds_left() < WRAP_UP_SECONDS + 15:
        return [], ""
    blocks = audit_blocks(books.records)
    if not blocks:
        return [], ""
    result = shell.judge(audit_prompt(books, blocks), 3000)
    books.trace.append({"kind": "audit", "result": result})
    fits = [str(x) for x in (result or {}).get("possible_fits") or [] if isinstance(x, (str, int))][:5]
    return fits, str((result or {}).get("reason") or "")


def audit_text(books: Books, fits: list[str], why: str) -> str:
    if not fits:
        return ""
    reason = f" ({why})" if why else ""
    text = ("Check: an independent review believes these listings may meet every requirement: " + ", ".join(fits) + reason
            + ". Then: verify each (details, stock, price) and buy the right one; decline only when every one of them truly fails.")
    if books.order_attempts:
        text += " Nothing is bought yet: a successful order ends the episode and this one is still going, so a decline now leaves the shopper empty-handed."
    return text


class Pending:
    def __init__(self, books: Books, name: str, args: dict[str, Any], finishing: set[str], planned: bool, batch_size: int, queued: int) -> None:
        self.books = books
        self.name = name
        self.args = args
        self.finishing = finishing
        self.planned = planned
        self.batch_size = batch_size
        self.queued = queued
        self.signature = name + dump(args)
        self.is_message = name == books.role("message")
        self.is_cart = name == books.role("cart")


def _guard_alone(p: Pending) -> str:
    if p.name in p.finishing and p.batch_size > 1:
        return "Check: this is an order or a decline. Then: send it by itself, once the results of your other calls are in."
    return ""


def _guard_message(p: Pending) -> str:
    if not p.is_message:
        return ""
    books, text = p.books, string_values(p.args)
    tell_rule = any(TELL_RULE.search(str(rule)) for rule in books.plan.get("rules") or [])
    if not (p.planned or "?" in text or tell_rule):
        return "Check: the message asks nothing. Then: keep shopper messages for questions only; findings, picks and plans travel with the finishing tool."
    if text.strip().lower() in books.questions_sent:
        return "Check: this very message went out already and its answer is above. Then: ask something else or move on."
    if books.message_budget is not None and books.messages_used + p.queued >= books.message_budget:
        return "Check: the message budget is spent. Then: send the shopper nothing more."
    return ""


def _guard_repeat(p: Pending) -> str:
    exempt = p.name in p.finishing or p.is_message or p.is_cart
    if exempt or p.signature not in p.books.recent:
        return ""
    return "Check: you made this identical call a moment ago and nothing changed; its result is above. Then: alter the arguments or do something different."


CALL_GUARDS = (_guard_alone, _guard_message, _guard_repeat)


def _cart_target(args: dict[str, Any], properties: dict[str, Any]) -> str | None:
    target = next((v for k, v in args.items() if one_word(v) and "enum" not in (properties.get(k) or {})), None)
    if target is None and cart_line_ids(args):
        target = cart_line_ids(args)[-1]
    return target


def reference_shelf(books: Books, item: str) -> str:
    # a shelf the shopper named only to take a value from holds no product to buy: an item there that names no wanted product is refused
    shelves = [p["category"] for p in books.pointers if p.get("category")]
    if not shelves or not item:
        return ""
    row = priced_rows(books.records).get(item) or {}
    own = _stems(" ".join(str(x) for x in (row.get("category"), row.get("title"), item_node(books, item).get("title")) if x))
    if not own or any(_shared_stems(_stems(name + " " + noun), own) for name, noun in product_targets(books)):
        return ""
    return next((shelf for shelf in shelves if _shared_stems(_stems(shelf), own)), "")


def _guard_cart(shell: Shell, p: Pending) -> str:
    books, name, args = p.books, p.name, p.args
    properties = books.schemas[name].get("properties") or {}
    for key, value in list(args.items()):
        # a listing named by its title instead of its id: the store cannot resolve it, so send the one opened listing that title belongs to
        if isinstance(value, str) and not one_word(value) and "enum" not in (properties.get(key) or {}) and len(value.strip()) >= 12:
            owners = [i for i in books.opened if str((listing_view(books, i) or {}).get("title") or "").lower().startswith(value.strip().lower())]
            if len(owners) == 1:
                args[key] = owners[0]
    count = cart_count(books, name, args)
    if is_number(count) and count <= 0:
        target = _cart_target(args, properties)
        stamp = f"{target}|{books.ledger_version}"
        if target and stamp in books.passed and stamp not in books.binding_fails and p.signature not in books.qty_warned:
            # the main loop can read the matching rule more strictly than the check did and swap a passed, cheaper line for a dearer one
            books.qty_warned.add(p.signature)
            return (SKIP + f"{target} passed the requirement check against every current need (any key that names the property counts, a"
                    " shorter or longer one too). Check: one of its own facts shows it failing a need. Then: repeat the same call.")
        return ""
    shelf = reference_shelf(books, _cart_target(args, properties) or "")
    if shelf:
        return (SKIP + f"{_cart_target(args, properties)} sits with the {shelf}, which the shopper named only as the place to read a value from;"
                " it is none of the products asked for. Check: a listing of the wanted product. Then: add that one instead.")
    checks_line = books.steps_left() > 3
    if checks_line and (books.add_signature is None or enum_signature(args, properties) == books.add_signature):
        target = _cart_target(args, properties)
        stamp = f"{target}|{books.ledger_version}"
        if target and stamp in books.binding_fails:
            return SKIP + books.binding_fails[stamp]
        if target and stamp not in books.cart_checked:
            books.cart_checked.add(stamp)
            verdict = verify_item(shell, books, target, "ADD TO THE CART", cart_line_text(args, target))
            if verdict and stamp in books.binding_fails:
                return SKIP + verdict
            if verdict:
                return "Not run (no step used). " + verdict + " Check: you are sure it fits. Then: repeat the same call."
    warning = quantity_problem(books, name, args)
    if not warning or p.signature in books.qty_warned:
        return ""
    books.qty_warned.add(p.signature)
    return SKIP + warning + " Check: you are sure. Then: repeat the same call."


def _parsed_arguments(raw: Any) -> dict[str, Any] | None:
    if isinstance(raw, dict):
        return raw
    if not (raw or "").strip():
        return {}
    return read_json_object(raw or "{}")


def screen_call(shell: Shell, books: Books, call: dict[str, Any], allowed: set[str], finishing: set[str],
                planned: bool, batch_size: int, queued_messages: int) -> str | tuple[str, dict[str, Any]]:
    function = call.get("function") or {}
    name = function.get("name")
    offered = ", ".join(sorted(allowed))
    if name not in books.schemas:
        return SKIP + f"'{name}' is no tool here. Pick from: {offered}."
    if not (planned or name in allowed):
        if name == books.role("message") and books.planner_on:
            return SKIP + "shopper questions are planned and sent on your behalf once a few listings' details are open."
        return SKIP + "too few steps remain for that; the only usable tools now are: " + offered + "."
    args = _parsed_arguments(function.get("arguments"))
    if args is None:
        return SKIP + "the arguments did not form a valid JSON object."
    args, problems = check_against(args, books.schemas[name], "args")
    if problems:
        return SKIP + "bad arguments: " + "; ".join(problems[:6])
    pending = Pending(books, name, args, finishing, planned, batch_size, queued_messages)
    for guard in CALL_GUARDS:
        refusal = guard(pending)
        if refusal:
            return SKIP + refusal
    refusal = _guard_cart(shell, pending) if pending.is_cart else ""
    return refusal or (name, args)


def proven_lines(books: Books) -> list[str]:
    return [item for item in books.cart_ids if f"{item}|{books.ledger_version}" in books.passed]


def cart_arguments(books: Books, tool: str, item: str, count: int) -> dict[str, Any] | None:
    schema = books.schemas.get(tool) or {}
    properties = schema.get("properties") or {}
    known = read_json_object(books.add_signature or "") or {}
    args: dict[str, Any] = {}
    for key, spec in properties.items():
        if not isinstance(spec, dict):
            continue
        kinds = declared_types(spec) or []
        if "enum" in spec:
            choice = known.get(key, next((v for v in spec["enum"] if isinstance(v, str) and name_words(v) & EDIT_WORDS), None))
            if choice is not None:
                args[key] = choice
        elif "string" in kinds and IDENT_KEY.search(key):
            args[key] = item
        elif "integer" in kinds or "number" in kinds:
            args[key] = count
    if item not in args.values():
        # the item slot can carry any name ("TargetItem"): it is then the one free-text slot the tool has
        texts = [k for k, spec in properties.items() if isinstance(spec, dict) and "enum" not in spec and "string" in (declared_types(spec) or [])]
        if len(texts) == 1:
            args[texts[0]] = item
    filled = [k for k in args if not (isinstance(properties[k], dict) and "enum" in properties[k])]
    if len(filled) < 2 or any(k not in args for k in schema.get("required") or []):
        return None
    return args


def code_cart(shell: Shell, books: Books, item: str, count: int | None = None) -> str:
    tool = books.role("cart")
    wanted = books.plan.get("quantity")
    count = count or (wanted if isinstance(wanted, int) and not isinstance(wanted, bool) and wanted >= 1 else 1)
    args = cart_arguments(books, tool, item, count) if tool else None
    if not args:
        return ""
    args, problems = check_against(args, books.schemas[tool], "args")
    if problems:
        return ""
    books.trace.append({"kind": "code_cart", "item": item, "args": args})
    answered = _send_chunk(shell, books, [("code_cart", tool, args)])
    return answered[0][1].split("\n[note]")[0][:600] if answered else ""


def review_cart(shell: Shell, books: Books) -> list[str]:
    for item in books.cart_ids:
        if f"{item}|{books.ledger_version}" not in books.passed:
            verify_item(shell, books, item, "ORDER", books.cart_lines.get(item, ""))
    return proven_lines(books)


DECLINE_CHECKLIST = ("Check: you are about to decline. Then: confirm you searched several ways (other words, filters, pages), read each candidate's own"
                     " properties, and saw each fail a stated requirement, a revealed need, the budget or stock. Close synonyms and any key spelling"
                     " count, and a requirement you only inferred from the story does not. Check: something fits. Then: buy it. Check: nothing fits."
                     " Then: call the decline tool again.")


def gate_decline(shell: Shell, books: Books, name: str, args: dict[str, Any]) -> tuple[str | None, str, dict[str, Any]]:
    order = books.role("order")
    if books.order_refused and books.cart_ids and order:
        # after a refused order the cart is usually one repair from a pass; a decline there has not won
        books.order_refused = False
        books.trace.append({"kind": "refused_decline_to_order", "lines": list(books.cart_ids)})
        return None, order, finish_arguments(books, order, "Order the cart again.")
    roomy = books.steps_left() > 3 and shell.seconds_left() > WRAP_UP_SECONDS + 15
    proven = review_cart(shell, books) if roomy else proven_lines(books)
    if proven and order:
        # a reviewed fit sits in the cart: buying it is right far more often than a decline
        books.trace.append({"kind": "decline_to_order", "lines": proven})
        return None, order, finish_arguments(books, order, "Order the cart.")
    if roomy and books.decline_checks < 3:
        books.decline_checks += 1
        fits, why = audit_fits(shell, books)
        for item in [f for f in fits if f not in books.swapped][:2] if not wanted_products(books.plan) else []:
            if verify_item(shell, books, item, "BUY INSTEAD OF DECLINING") or f"{item}|{books.ledger_version}" not in books.passed:
                continue
            books.swapped.add(item)
            carted = code_cart(shell, books, item)
            if carted:
                return (HOLD + f"A separate review checked {item} against every current requirement and found it fits, so code put it in the cart"
                        f" (result: {carted}). Then: call the order tool; declining now would leave the shopper with nothing although a fit exists."), name, args
        text = audit_text(books, fits, why)
        if text:
            return HOLD + text, name, args
        if books.decline_checks == 1:
            return HOLD + DECLINE_CHECKLIST, name, args
    # a decline is right far less often than an order of the closest opened listing; the decline stands only when every candidate
    # within stock and budget fails a need by its own quoted fact
    if order and not books.cart_ids and not wanted_products(books.plan) and books.steps_left() >= 2:
        for item in closest_listings(books)[:3]:
            if shell.seconds_left() > WRAP_UP_SECONDS:
                verify_item(shell, books, item, "BUY INSTEAD OF DECLINING")
            if f"{item}|{books.ledger_version}" in books.binding_fails:
                continue
            if code_cart(shell, books, item) and books.cart_ids:
                books.trace.append({"kind": "decline_to_closest", "item": item})
                return None, order, finish_arguments(books, order, "Order the cart.")
            break
    return None, name, scrub_decline(books, args, name)


def closest_listings(books: Books, product: str = "") -> list[str]:
    needs = needs_for(books, product)
    quantity = product_quantity(books, product)
    rows = priced_rows(books.records)
    # a basket's budget covers every line, so one product's candidates are not held to it alone
    budget = store_budget(books, rows) if not product else None
    ranked = []
    for item in books.opened:
        if f"{item}|{books.ledger_version}" in books.binding_fails or (product and product_of(books, item, rows) != product):
            continue
        view = listing_view(books, item)
        if not view or not view.get("facts") or (books.plan.get("in_stock_only") and view.get("in_stock") is False):
            continue
        if is_number(budget) and listing_cost(view, quantity) != float("inf") and listing_cost(view, quantity) > budget:
            continue
        ranked.append((-sum(code_cells(view, needs)), listing_cost(view, quantity), item))
    return [item for _, _, item in sorted(ranked)]


def cheaper_fits(shell: Shell, books: Books, item: str, product: str, rows: dict[str, dict[str, Any]]) -> tuple[list[str], list[str]]:
    held = (rows.get(item) or {}).get("price")
    if not is_number(held):
        return [], []
    label = (rows.get(item) or {}).get("category")
    cheaper = sorted((row["price"], ident) for ident, row in rows.items()
                     if ident != item and row["price"] < held and row.get("in_stock") is not False
                     and (not label or not row.get("category") or row.get("category") == label) and product_of(books, ident, rows) == product)
    opened = [ident for _, ident in cheaper if ident in books.opened and f"{ident}|{books.ledger_version}" not in books.binding_fails][:CHECK_BATCH]
    views = [v for v in (listing_view(books, ident) for ident in opened) if v]
    found = []
    if views and shell.seconds_left() > WRAP_UP_SECONDS + 25:
        verdicts = judge_listings(judge_ask(shell), bounded_needs(shell, books, product), views, product_quantity(books, product), None,
                                  bool(books.plan.get("in_stock_only")))
        for view in views:
            if (verdicts.get(view["id"]) or {}).get("fits") is True:
                books.passed.add(f"{view['id']}|{books.ledger_version}")
                found.append(view["id"])
    unopened = [ident for _, ident in cheaper if ident not in books.opened][:2]
    return found, unopened


def store_budget(books: Books, rows: dict[str, dict[str, Any]]) -> Any:
    # the ask stage converts the shopper's figure into store units; otherwise a figure far below the shelf prices is read as whole money
    for line in (books.handoff or {}).get("lines") or []:
        budget = (line.get("record") or {}).get("budget")
        if is_number(budget) and budget > 0:
            return budget
    if not is_number(books.budget_now):
        return None
    prices = sorted(row["price"] for row in rows.values() if is_number(row.get("price")))
    if prices and prices[len(prices) // 2] > books.budget_now * 5:
        return books.budget_now * 100
    return books.budget_now


def gate_basket(shell: Shell, books: Books) -> tuple[str, bool]:
    targets = product_targets(books)
    rows = priced_rows(books.records)
    checking = shell.seconds_left() > WRAP_UP_SECONDS + 20
    lines = []
    for item in books.cart_ids:
        verify_item(shell, books, item, "ORDER", books.cart_lines.get(item, ""))
        stamp = f"{item}|{books.ledger_version}"
        row = rows.get(item) or {}
        lines.append({"id": item, "product": product_of(books, item, rows), "quantity": books.cart_counts.get(item), "price": row.get("price"),
                      "verified": True if stamp in books.passed else (False if checking else None),
                      "binding": books.binding_fails.get(stamp, ""),
                      "in_stock_problem": bool(books.plan.get("in_stock_only")) and row.get("in_stock") is False})
    cheaper = {}
    if books.plan.get("priority") == "cheapest":
        for product, _ in targets:
            mine = [line for line in lines if line["product"] == product]
            if len(mine) == 1 and mine[0]["verified"] and not mine[0]["binding"]:
                found, unopened = cheaper_fits(shell, books, mine[0]["id"], product, rows)
                cheaper[product] = {"fit": found[0] if found else None, "unopened": unopened}
    # the budget note is given once; after that the shopper's total is left to the order tool
    budget = None if books.budget_held else store_budget(books, rows)
    wanted = [{"name": p, "noun": n, "quantity": product_quantity(books, p)} for p, n in targets]
    problems = basket_check(lines, wanted, budget, cheaper)
    if problems and problems[-1].startswith("the lines add up to"):
        books.budget_held = True
    # lines, quantities and extras alone: these the cart can always fix
    plain = [dict(line, verified=None, binding="", in_stock_problem=False) for line in lines]
    return "; ".join(problems), bool(basket_check(plain, wanted))


def gate_finish(shell: Shell, books: Books, name: str, args: dict[str, Any]) -> tuple[str | None, str, dict[str, Any]]:
    if name == books.role("order") and books.mode_key and books.mode_key not in args:
        args = with_mode(books, name, args, False)
    if is_decline(books, name, args):
        return gate_decline(shell, books, name, args)
    if name != books.role("order"):
        return None, name, args
    # after a refusal an older backup may have lost its place as the cheapest fit
    if (books.plan.get("priority") == "cheapest" and books.refused_at > books.searched_at and books.research_checks < 1
            and books.steps_left() > 10 and shell.seconds_left() > WRAP_UP_SECONDS + 20):
        books.research_checks += 1
        return ("Not run yet (no step used). Check: the previous order was refused and the request wants the cheapest fitting item. Then: search again before"
                " ordering, because an earlier backup is right only when nothing cheaper fits now: use the stated requirement words and the category, a"
                " price limit just over the refused item's price when the search allows one, and further pages. Open cheaper candidates' details, put the"
                " cheapest one meeting every requirement in the cart, and then order."), name, args
    if not books.cart_ids and books.cart_used is False and books.steps_left() > 1:
        return HOLD + "The cart is empty, so an order cannot go through. Then: put the verified fit(s) in the cart first.", name, args
    state = f"{sorted(books.cart_ids)}|{sorted(books.cart_counts.items())}|{books.ledger_version}"
    held = books.basket_holds.get(state, 0)
    if books.steps_left() > 3 and shell.seconds_left() > WRAP_UP_SECONDS and (not books.order_checked or 0 < held < 3):
        # a wrong set of lines is held up to three times per cart state, anything else once
        books.order_checked = True
        problems, structural = gate_basket(shell, books)
        if problems and (held == 0 or structural):
            books.basket_holds[state] = held + 1
            return (HOLD + "The cart is not ready to order: " + problems + ". Then: fix each point (every cart change costs a step), then"
                    " call the order tool again."), name, args
    if wanted_products(books.plan):
        fill_missing_products(shell, books)
    return None, name, args


def fill_missing_products(shell: Shell, books: Books) -> None:
    # an order without a line for one of the wanted products fails for certain: the closest unproven candidate is the better bet
    rows = priced_rows(books.records)
    held = {product_of(books, item, rows) for item in books.cart_ids}
    for product, _ in product_targets(books):
        if product in held or books.steps_left() < 2:
            continue
        for item in closest_listings(books, product)[:3]:
            if shell.seconds_left() > WRAP_UP_SECONDS:
                verify_item(shell, books, item, "FILL THE MISSING LINE")
            if f"{item}|{books.ledger_version}" in books.binding_fails:
                continue
            if code_cart(shell, books, item, product_quantity(books, product)):
                books.trace.append({"kind": "fill_missing", "product": product, "item": item})
            break


def cart_count(books: Books, name: str, args: dict[str, Any]) -> Any:
    properties = (books.schemas.get(name) or {}).get("properties") or {}
    key = next((k for k, spec in properties.items() if isinstance(spec, dict) and "enum" not in spec
                and set(declared_types(spec) or []) & {"integer", "number"}), None)
    return args.get(key) if key else None


def record_cart_change(books: Books, name: str, args: dict[str, Any]) -> None:
    properties = books.schemas[name].get("properties") or {}
    signature = enum_signature(args, properties)
    if books.add_signature is None:
        books.add_signature = signature
    listed = cart_line_ids(args)
    if listed is not None:
        if signature != books.add_signature:
            books.cart_ids = [i for i in books.cart_ids if i not in listed]     # the same tool taking lines out
        else:
            books.cart_ids = listed     # the whole basket passed as a list of lines
            books.cart_lines = {i: cart_line_text(args, i) for i in listed}
        return
    count = cart_count(books, name, args)
    removing = is_number(count) and count <= 0
    for key, value in args.items():
        spec = properties.get(key) if isinstance(properties.get(key), dict) else {}
        if "enum" in spec or not one_word(value):
            continue
        rest = [i for i in books.cart_ids if i != value]
        books.cart_ids = rest + [value] if signature == books.add_signature and not removing else rest
        if is_number(count) and not removing:
            books.cart_counts[value] = count


def order_followup(books: Books, name: str, args: dict[str, Any], observation: Any, reply: dict[str, Any]) -> str:
    if name != books.role("order") or is_decline(books, name, args) or reply.get("finished"):
        return ""
    if isinstance(observation, dict) and ("refusal" in observation or event_parts(observation)[1]):
        return ""
    if isinstance(observation, dict) and awaits_further_step(observation):
        return ("Check: the order waits for one more step (it hands over a token or asks for confirmation). Then: take that step now with the tool"
                " that completes checkout; nothing is bought until the episode ends.")
    # no end yet: a refusal, a failure or a checkout still pending
    books.refusals += 1
    books.refused_at = books.step
    books.order_checked = False
    return ("Check: THE ORDER CALL DID NOT END THE EPISODE. Then: know that NOTHING IS BOUGHT YET, whatever the result claims (a completed order ends"
            " the episode). Check: the result asks for another step (a confirmation with a token or code it supplies). Then: take that step now."
            " Check: it was refused or failed. Then: read the reason, apply each change it names, repair the cart, re-check the pick's stock and"
            " price, and order again; a new price alone leaves an item acceptable while price times quantity stays within budget.")


def _count_call(books: Books, name: str, args: dict[str, Any], observation: Any) -> None:
    message_tool, cart_tool = books.role("message"), books.role("cart")
    if name == message_tool:
        books.messages_used += 1
        sent = string_values(args).strip().lower()
        failed = isinstance(observation, dict) and observation.get("error")
        if sent and not failed:
            books.questions_sent.append(sent)      # a failed message may go out again
    if name == books.role("order"):
        books.order_attempts += 1
    if name == cart_tool:
        books.cart_used = True
        books.order_checked = False
    resets = name in (cart_tool, message_tool) or name in {books.role("order"), books.role("decline")} - {None}
    books.recent = [] if resets else (books.recent + [name + dump(args)])[-2:]


def _remember(books: Books, name: str, args: dict[str, Any], observation: Any) -> None:
    books.records.append(observation)
    if name == books.role("cart"):
        record_cart_change(books, name, args)
    if name == books.role("search"):
        books.searched_at = books.step
    if name in (books.role("details"), books.role("view")):
        books.opened.update(v for v in args.values() if isinstance(v, str))
    if name == books.role("cart_view"):
        books.last_cart_view = observation


def absorb_result(shell: Shell, books: Books, name: str, args: dict[str, Any], observation: Any, reply: dict[str, Any], last: bool) -> str:
    _count_call(books, name, args, observation)
    _remember(books, name, args, observation)
    if name == books.role("order") and not is_decline(books, name, args) and isinstance(observation, dict) and (
            "refusal" in observation or event_parts(observation)[1]):
        books.order_refused = True
    note = joined(observation_notes(books, name, observation), order_followup(books, name, args, observation, reply))
    if books.pointers and name in (books.role("details"), books.role("view")):
        note = joined(note, resolve_pointers(shell, books))
    elif books.pointers and name == books.role("search"):
        note = joined(note, pointer_rows_note(books))
    heard, question = shopper_lines(books, name, args, observation)
    if heard:
        note = joined(note, interpret(shell, books, heard, question))
    text = dump(observation)
    if last and reply.get("finished"):
        text += " [episode over]"
    return text + ("\n[note] " + note if note else "")


def _send_chunk(shell: Shell, books: Books, chunk: list[tuple[str, str, dict[str, Any]]]) -> list[tuple[str, str]]:
    actions = [{"name": name, "args": args} for _, name, args in chunk]
    before = books.step
    reply = shell.call_world(actions)
    books.trace.append({"kind": "env", "step": before + 1, "actions": actions, "reply": reply})
    if reply is None:
        return [(call_id, "The call failed (network or server); try it again.") for call_id, _, _ in chunk]
    observations = reply.get("observations") if isinstance(reply.get("observations"), list) else [reply.get("observation")]
    answered = []
    for index, (call_id, name, args) in enumerate(chunk[: len(observations)]):
        is_last = index == len(observations) - 1
        answered.append((call_id, absorb_result(shell, books, name, args, observations[index], reply, is_last)))
    answered += [(call_id, "Not run: the episode ended before this call.") for call_id, _, _ in chunk[len(observations):]]
    return answered


def dispatch(shell: Shell, books: Books, runnable: list[tuple[str, str, dict[str, Any]]], replies: list[tuple[str, str]]) -> None:
    chunks = [runnable[i : i + 16] for i in range(0, len(runnable), 16)]
    for chunk in chunks:
        if books.finished or books.dead or books.steps_left() <= 0:
            replies.extend((call_id, "Not run: the episode has ended or no steps remain.") for call_id, _, _ in chunk)
        else:
            replies.extend(_send_chunk(shell, books, chunk[: books.steps_left()]))


def run_calls(shell: Shell, books: Books, calls: list[dict[str, Any]], planned: bool = False) -> list[tuple[str, str]]:
    allowed = {tool_name(t) for t in turn_tools(books)}
    finishing = {books.role("order"), books.role("decline")} - {None}
    replies: list[tuple[str, str]] = []
    runnable: list[tuple[str, str, dict[str, Any]]] = []
    queued_messages = 0
    for call in calls:
        outcome = screen_call(shell, books, call, allowed, finishing, planned, len(calls), queued_messages)
        if isinstance(outcome, str):
            replies.append((call["id"], outcome))
            continue
        name, args = outcome
        if name == books.role("message"):
            queued_messages += 1
        runnable.append((call["id"], name, args))
    if runnable and runnable[0][1] in finishing:
        call_id, name, args = runnable[0]
        held, name, args = gate_finish(shell, books, name, args)
        if held is not None:
            replies.append((call_id, held))
            return replies
        runnable[0] = (call_id, name, args)
    elif runnable and finishing:
        room = max(0, books.steps_left() - 1)
        if books.steps_left() - 1 < len(runnable):
            replies.extend((call_id, SKIP + "only the final step remains, and it belongs to the order (or the decline).")
                           for call_id, _, _ in runnable[room:])
            runnable = runnable[:room]
    dispatch(shell, books, runnable, replies)
    position = {c["id"]: i for i, c in enumerate(calls)}
    replies.sort(key=lambda r: position.get(r[0], 0))
    return replies


def _stems(text: str) -> set[str]:
    return {w.lower().rstrip("s") for w in re.findall(r"[A-Za-z]{3,}", text)}


def product_targets(books: Books) -> list[tuple[str, str]]:
    products = wanted_products(books.plan)
    if products:
        nouns = {str(e.get("product") or "").strip(): str(e.get("noun") or "").strip()
                 for e in books.plan.get("items") or [] if isinstance(e, dict)}
        return [(name, nouns.get(name) or name) for name, _ in products]
    noun = str(books.plan.get("noun") or "").strip() or str(books.plan.get("product") or "").strip() or "item"
    return [("", noun)]


def _settled_names(books: Books) -> list[str]:
    texts = list(books.current) if books.current is not None else books.stated + books.revealed
    names = [re.sub(r"[_.]+", " ", t.split(":")[0]).lower() for t in texts if isinstance(t, str)]
    return names + [re.sub(r"[_.]+", " ", str(f)).lower() for f in books.free]


def _shared_stems(left: set[str], right: set[str]) -> int:
    return sum(1 for a in left if any(a == b or (min(len(a), len(b)) >= 4 and (a.startswith(b) or b.startswith(a))) for b in right))


def candidate_properties(books: Books, product: str, noun: str) -> list[str]:
    tally = PropertyTally()
    for record in books.records:
        tally.visit(record, 0, "", "")
    group = ""
    if product:
        scored = sorted(((_shared_stems(_stems(product + " " + noun), _stems(g)), tally.tables.get(g, 0), g) for g in tally.counts if g), reverse=True)
        group = scored[0][2] if scored and scored[0][0] > 0 else ""
        if not group and scored:
            return []       # its listings are not open yet; other products' keys would only confuse the shopper
    if not group:
        tally.fold_into_unlabelled()
    counts, values, samples = tally.counts.get(group, {}), tally.values.get(group, {}), tally.samples.get(group, {})
    asked = books.asked_props.get(product, set())
    settled = _settled_names(books) + sorted(asked)
    out = []
    for name in sorted(counts, key=lambda n: (len(values.get(n, ())) < 2, -counts[n])):
        words = set(name.split())
        if name in asked or any(s and (words <= set(s.split()) or set(s.split()) <= words) for s in settled):
            continue
        out.append(f"{name} (seen in {counts[name]} listings, {len(values.get(name, ()))} distinct values): " + " | ".join(samples.get(name, [])[:4]))
    return out[:8]


def pick_property(shell: Shell, books: Books, noun: str, shortlist: list[str]) -> str:
    names = [line.split(" (seen in ")[0] for line in shortlist]
    if len(shortlist) == 1:
        return names[0]
    prompt = compose(PICK_PROMPT, "THE SHOPPER'S REQUEST:\n" + books.query(), f"PRODUCT: the {noun}",
                     "REQUIREMENTS UP TO NOW:\n" + requirements_line(books),
                     books.asked and "ASKED BEFORE (and the shopper's answer):\n" + "\n".join(books.asked),
                     "CANDIDATES:\n" + "\n".join(f"{i}. {line}" for i, line in enumerate(shortlist, 1)))
    result = shell.judge(prompt, 1500, [READ_MODEL, "openai/gpt-oss-120b", LAST_RESORT]) or {}
    pick = _to_integer(result.get("pick")) if not is_number(result.get("pick")) else int(result.get("pick"))
    return names[pick - 1] if isinstance(pick, int) and 1 <= pick <= len(names) else names[0]


def next_question(shell: Shell, books: Books) -> tuple[str, str]:
    first = next((str(a) for a in books.plan.get("ask_first") or [] if a and str(a) not in books.first_done), "")
    if first:
        books.first_done.add(first)
        result = shell.judge(compose(FIRST_PROMPT, "THE SHOPPER'S REQUEST:\n" + books.query(), "POINT TO SETTLE: " + first), 1500,
                             [READ_MODEL, "openai/gpt-oss-120b", LAST_RESORT]) or {}
        question = str(result.get("question") or "").strip()
        if question:
            return question, first
    targets = product_targets(books)
    # every product gets a question before any gets a second one
    turn_order = sorted(range(len(targets)), key=lambda i: (len(books.asked_props.get(targets[i][0], set())), i))
    for index in turn_order:
        product, noun = targets[index]
        shortlist = candidate_properties(books, product, noun)
        if not shortlist:
            continue
        name = pick_property(shell, books, noun, shortlist)
        books.asked_props.setdefault(product, set()).add(name)
        return f"Which {name} would you like the {noun} to have?", (product + " " if product else "") + name
    return "", ""


def ask_planned(shell: Shell, books: Books, force: bool = False) -> bool:
    tool = books.role("message")
    limit = books.message_limit()
    if not tool or books.messages_used >= limit:
        books.planner_on = False
        return False
    tally = PropertyTally()
    for record in books.records:
        tally.visit(record, 0, "", "")
    fact_sets = tally.fact_sets
    urgent = any(str(a) not in books.first_done for a in books.plan.get("ask_first") or [] if a)
    if not force:
        if fact_sets < 4 and books.step < 8 and not urgent:
            return False
        # one question per round: the assistant acts on each answer before the next one goes out
        if books.asked and books.step < books.asked_at + 2:
            return False
        if fact_sets < books.plan_retry_at:
            return False
    if books.steps_left() <= (6 if force else 10) or shell.seconds_left() < WRAP_UP_SECONDS + 40:
        books.planner_on = False
        return False
    question, label = next_question(shell, books)
    args = message_arguments(books, question) if question else None
    if args:
        args, problems = check_against(args, books.schemas[tool], "args")
        args = None if problems or question.strip().lower() in books.questions_sent else args
    if not args:
        # nothing to ask yet: one more try once another listing is opened
        books.plan_misses += 1
        books.plan_retry_at = fact_sets + 1
        if books.plan_misses >= 3:
            books.planner_on = False
        return False
    call = {"id": f"ask_{books.messages_used + 1}_{books.step}", "type": "function", "function": {"name": tool, "arguments": json.dumps(args)}}
    books.messages.append({"role": "assistant", "content": f"Next I put this to the shopper: {label}.", "tool_calls": [call]})
    books.last_unclear = False
    texts = post_replies(shell, books, run_calls(shell, books, [call], planned=True))
    if texts:
        books.asked.append(f"- {label}: we asked \"{question}\" -> {texts[0].split(chr(10) + STATUS_TAG)[0][:400]}")
    books.asked_at = books.step
    books.sets_at_last_ask = fact_sets
    books.planner_on = books.messages_used < limit
    return True


def send_reply(shell: Shell, books: Books, text: str) -> bool:
    tool = books.role("message")
    if not tool or books.messages_used >= books.message_limit() or books.steps_left() <= 4 or shell.seconds_left() < WRAP_UP_SECONDS:
        return False
    args = message_arguments(books, text)
    if not args:
        return False
    args, problems = check_against(args, books.schemas[tool], "args")
    if problems:
        return False
    call = {"id": f"reply_{books.step}", "type": "function", "function": {"name": tool, "arguments": json.dumps(args)}}
    books.messages.append({"role": "assistant", "content": "Answering what the shopper asked.", "tool_calls": [call]})
    post_replies(shell, books, run_calls(shell, books, [call], planned=True))
    return True


def forced_call(shell: Shell, books: Books, action: dict[str, Any]) -> dict[str, Any] | None:
    reply = shell.call_world([action])
    books.trace.append({"kind": "forced_finish", "action": action, "reply": reply})
    return reply


def decline_action(books: Books, name: str) -> dict[str, Any]:
    return {"name": name, "args": finish_arguments(books, name, NEUTRAL_DECLINE, declining=True)}


def force_finish(shell: Shell, books: Books) -> None:
    if books.finished or books.dead or books.steps_left() <= 0 or shell.seconds_left() < 2:
        return
    order, decline = books.role("order"), books.role("decline")
    if order and books.cart_used:
        action = {"name": order, "args": finish_arguments(books, order, "Order the cart.")}
    elif decline:
        action = decline_action(books, decline)
    else:
        return
    reply = forced_call(shell, books, action)
    observation = (reply or {}).get("observation")
    refused = observation if isinstance(observation, dict) else {}
    parts = [v for v in refused.values() if isinstance(v, (list, str))]
    parts += [x for v in refused.values() if isinstance(v, dict) for x in v.values() if isinstance(x, (list, str))]
    lines = long_lines(parts)
    if reply and not reply.get("finished") and lines and order and action["name"] == order and books.cart_used and books.steps_left() > 0:
        # a refusal is often only a price change, so the cart gets one more try before the decline
        reply = forced_call(shell, books, action)
    if reply and not reply.get("finished") and isinstance(reply.get("observation"), dict) and lines and decline and books.steps_left() > 0:
        forced_call(shell, books, decline_action(books, decline))


# ---- ask engine: V1 frozen a1728300 with paired basket questions, its two model names carry an ASK_ prefix and its backup is Haiku ----
ASK_READ_MODEL = "google/gemini-3.8-flash"
ASK_BACKUP_MODEL = "anthropic/claude-haiku-5.5"
SEAT_PARAMS: dict[str, dict[str, Any]] = {
    ASK_READ_MODEL: {"reasoning": {"effort": "minimal"}},
    ASK_BACKUP_MODEL: {},
}
# the reading LLM reasons before it answers, so each seat leaves room for both
OUTPUT_TOKENS = {"request": 3000, "hint": 1500, "reply": 1500}
LOOK_LISTINGS = 8
MENU_SIZE = 24
SAMPLE_VALUES = 3
RECORD_LIMIT = 24
STEP_SPARE = 1
SEARCH_STEPS = 4
COMMIT_STEPS = 2
EVENT_STEPS = 3
CHEAPEST_STEPS = 2
LOOK_MINIMUM = 2
LOOKUP_STEPS = 3
FIND_LISTINGS = 3
BATCH_LISTINGS = 12
FIRST_PICKS = 2
LLM_ROUNDS = 2
LLM_WAIT = 4

REQUEST_GUIDE = """# Shopping assistant - this step: read the customer's opening request
- You read one request a customer sent to an online shop and return its plain facts as JSON.
# Fields
- "products": one entry per distinct product the customer wants, in the order named. Shop category names often join two words with "and" or "&"; keep such a name whole as one product, and split only where the request gives each part its own quantity or conditions:
  - "name": a short noun phrase for the product, the way a shop would label it (two to four words).
  - "search": the words to type into the shop's search box to find it: the product noun, plus at most two of the values the customer fixed.
  - "stated": every condition the request fixes for that product, each written "property: value" (for example "material: wool" or "price: at most 1500 cents"). Leave out stock, delivery and how to choose among fits. When the request ties a value to another listing (one a named seller sells), the attributes that single out that listing describe it, not this product: they are never stated conditions.
  - "quantity": how many units of it the customer wants, as an integer (1 when not said).
- "messages": how many messages the customer lets you send, as an integer, or null when the request does not say.
- "cheapest": true when the customer wants the lowest price or lowest total among the fits, else false.
- "budget": the most the whole order may cost, as a number in the money units the request uses (for example 2950 for "2950¢", 150 for "$150"), or null when there is no cap.
- "budget_unit": "cents" when that cap is given in cents or pence, "whole" when it is given in whole currency (dollars, pounds, euros), or null when there is no cap.
- "in_stock": true when the customer wants only items that are in stock and ready to ship, else false.
- "story": the background the customer gives (who uses it, where, when, why, what they will notice) in one short sentence, or "" when there is none.
# Answer
- ONE JSON object and nothing else."""

HINT_GUIDE = """# Shopping assistant - this step: choose which properties to ask the customer about
- The customer keeps back a wish or two about a product and states it only when a question names that exact property.
- For each product (P1, P2 ...) you get the properties seen on the shop's listings for it, with how many listings carry each and some sample values.
# Rules
- Rank up to eight properties per product, the one most likely to hold an unstated wish first.
- A property the request already settles holds no unstated wish: leave it out. Leave out identifiers, titles and prices too.
- Lean on what the background implies without saying it: who uses the item, where or how it is used, how it should look or feel, what went wrong with the old one.
- A property such a detail points at goes in "story" as well as in "ranked". With no such detail, "story" stays empty.
- Use property names exactly as listed; two spellings of one property count once.
- Keys that name one property in different words (a short and a long name, two words for one thing) go together in "same", one list per property, each key written as listed.
# Answer
- ONE JSON object and nothing else: {"P1": {"ranked": ["property", ...], "story": ["property", ...], "same": [["property", "property"], ...]}, "P2": {...}}"""

REPLY_GUIDE = """# Shopping assistant - this step: read the customer's reply
- You get the question that was sent, the customer's reply, and the property names the shop's listings carry for each product (P1, P2 ...).
# Rules
- Write down each wish the reply states. A reply can state none, one or several.
- "product": the label of the product the wish is about.
- "kind": "value" when the reply sets a property to a value; "listing" when the value has to be read off another listing the reply names.
- For "value": "property" is the property the reply names (the listed name that matches it, when one does); "value" is the value exactly as written; "relation" is "is", "not", "at_most" or "at_least". When the value carries its own comparison, that comparison is the relation, whatever words frame the sentence. A comparison word before a value that is no number makes it "is". Figures of speech that only stress how firm the wish is are no value.
- For "listing": "property" as above; "value" is the words that name that other listing; "relation" is "is", or "not" when the reply wants a value different from that listing's; "search" is two or three words to find that listing with the shop's search: the seller or brand name it gives, then the product noun.
- Indifference, leaving it to the shop, a plea to ask again more simply or one point at a time, or anything else without a wish gives an empty list.
- "split" is true when the reply asks for fewer things per question (one question at a time, pick one, too many asks); a reply that only says it did not understand is not that.
# Answer
- ONE JSON object and nothing else: {"wishes": [{"product": "P1", "kind": "value", "property": "", "value": "", "relation": "", "search": ""}], "split": false}"""

RELATION_WORDS = {"is": "", "not": "not ", "at_most": "at most ", "at_least": "at least "}


def words(text: Any) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(text).lower())


def plain(name: Any) -> str:
    return " ".join(words(str(name).replace("_", " ")))


def content_words(name: Any) -> set[str]:
    return {word for word in words(plain(name)) if len(word) > 2}


def decode_object(text: str) -> dict[str, Any] | None:
    found = re.search(r"\{.*\}", text or "", re.S)
    if not found:
        return None
    try:
        value = json.loads(found.group(0))
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def ask_llm(shop: Any, prompt: str, seat: str) -> dict[str, Any] | None:
    for round_number in range(LLM_ROUNDS):
        failed = False
        for model in (ASK_READ_MODEL, ASK_BACKUP_MODEL):
            try:
                text = shop.chat([{"role": "user", "content": prompt}], model=model, temperature=0, max_tokens=OUTPUT_TOKENS[seat], **SEAT_PARAMS[model])
            except Exception:
                failed = True
                continue
            answer = decode_object(text if isinstance(text, str) else "")
            if answer is not None:
                return answer
        if not failed or round_number == LLM_ROUNDS - 1:
            break
        time.sleep(LLM_WAIT)
    return None


def packed_pairs(text: str) -> list[tuple[str, str]]:
    return re.findall(r'([A-Za-z][\w \-]*?)\s*=\s*"([^"]*)"', text)


def leaf_facts(listing: Any, into: dict[str, str] | None = None, depth: int = 0) -> dict[str, str]:
    facts = {} if into is None else into
    if isinstance(listing, dict):
        for key, value in listing.items():
            if isinstance(value, str) and len(packed_pairs(value)) >= 2:
                for name, inner in packed_pairs(value):
                    if inner and len(inner) <= 60:
                        facts.setdefault(name.strip(), inner)
            elif isinstance(value, (dict, list)):
                if depth < 3:
                    leaf_facts(value, facts, depth + 1)
            elif value not in (None, "") and depth > 0 and len(str(value)) <= 60 and len(str(value).split()) <= 5:
                facts.setdefault(str(key), str(value))
        if depth == 0 and not facts:
            for key, value in listing.items():
                if not isinstance(value, (dict, list, bool)) and value not in (None, "") and len(str(value)) <= 40:
                    facts[str(key)] = str(value)
    elif isinstance(listing, list):
        for item in listing:
            if isinstance(item, dict) and len(item) == 2 and {"name", "value"} <= set(item):
                if item["value"] not in (None, "") and len(str(item["value"])) <= 60:
                    facts.setdefault(str(item["name"]), str(item["value"]))
            elif depth < 3:
                leaf_facts(item, facts, depth + 1)
    return facts


def full_text(listing: Any) -> str:
    if isinstance(listing, dict):
        return " ".join(full_text(value) for value in listing.values())
    if isinstance(listing, list):
        return " ".join(full_text(value) for value in listing)
    return "" if listing is None else str(listing)


def row_ids(result: Any) -> list[str]:
    rows = result.get("rows") if isinstance(result, dict) else result
    ids = []
    for row in rows or []:
        ident = (row.get("id") or row.get("listing_id") or row.get("product_id")) if isinstance(row, dict) else row
        if ident and str(ident) not in ids:
            ids.append(str(ident))
    return ids


def menu_order(listings: list[dict[str, str]], settled: set[str]) -> list[tuple[str, int, list[str]]]:
    seen: dict[str, dict[str, int]] = {}
    for facts in listings:
        for key, value in facts.items():
            seen.setdefault(key, {})
            seen[key][value] = seen[key].get(value, 0) + 1
    ranked = []
    for key, values in seen.items():
        name = plain(key)
        if not name or name in settled:
            continue
        carriers = sum(values.values())
        distinct = len(values)
        if all(re.fullmatch(r"[a-z]{0,3}[-_]?[0-9a-f]{8,}", value.lower()) for value in values):
            continue
        if distinct == carriers and sum(len(value.split()) for value in values) >= 4 * distinct:
            continue
        weight = carriers * min(distinct, 4)
        samples = [value for value, _ in sorted(values.items(), key=lambda pair: -pair[1])][:SAMPLE_VALUES]
        ranked.append((weight, key, carriers, samples))
    ranked.sort(key=lambda entry: -entry[0])
    unique: list[tuple[str, int, list[str]]] = []
    names: set[str] = set()
    for _, key, carriers, samples in ranked:
        if plain(key).replace(" ", "") not in names:
            names.add(plain(key).replace(" ", ""))
            unique.append((key, carriers, samples))
    return unique


def merged_order(first: list[str], story: list[str], spread: list[str]) -> list[str]:
    known = {plain(key): key for key in spread}
    order = [known[plain(key)] for key in first + story if plain(key) in known]
    return list(dict.fromkeys(order + spread))


QUESTION_CHARS = 70


def example_text(samples: list[str], count: int = 2) -> str:
    shown = [value for value in samples if len(value) <= 24][:count]
    return f", such as {' or '.join(shown)}" if shown else ""


def question_text(product: str, key: str, samples: list[str], basket: bool) -> str:
    # short questions draw more answers: examples stay only while the question stays short
    bare = f"What {plain(key)} do you need for the {product}?"
    if basket:
        return bare
    for count in (2, 1):
        worded = f"What {plain(key)} do you need for the {product}{example_text(samples, count)}?"
        if len(worded) <= QUESTION_CHARS:
            return worded
    return bare


class Product:
    def __init__(self, label: str, name: str, search: str, stated: list[str], quantity: int = 1) -> None:
        self.label = label
        self.quantity = quantity
        self.name = name
        self.search = search
        self.stated = stated
        self.listings: dict[str, dict[str, str]] = {}
        self.menu: list[tuple[str, int, list[str]]] = []
        self.queue: list[str] = []
        self.asked: list[str] = []
        self.bare: list[str] = []
        self.same: list[set[str]] = []
        self.told = 0
        self.searches: list[dict[str, Any]] = []
        self.rows: dict[str, dict[str, Any]] = {}
        self.shelf: str | None = None
        self.strength = 0.0

    def called(self, basket: bool) -> str:
        return (self.shelf or self.name) if basket else self.name

    def samples(self, key: str) -> list[str]:
        return next((samples for name, _, samples in self.menu if name == key), [])


class AskEngine:
    def __init__(self, task: dict[str, Any], shop: Any) -> None:
        self.task = task
        self.shop = shop
        self.request = str(task.get("request") or "")
        self.products: list[Product] = []
        self.messages = 0
        self.story = ""
        self.record: list[str] = []
        self.sent = 0
        self.split = False
        self.trace: list[dict[str, Any]] = []
        self.texts: dict[str, str] = {}
        self.facts: dict[str, dict[str, str]] = {}
        self.labels: dict[str, set[str]] = {}
        self.view_steps = 1.0
        self.cheapest = False
        self.budget: float | None = None
        self.in_stock = False
        self.budget_unit: str | None = None
        self.told_lines: list[tuple[Product, str, str, str]] = []
        self.said: list[str] = []
        self.pages: dict[str, dict[str, Any]] = {}
        self.refused: list[dict[str, Any]] = []

    def search(self, product: Product, query: str, page: Any = None) -> Any:
        result = self.shop.search(query, page=page) if page is not None else self.shop.search(query)
        rows = (result.get("rows") if isinstance(result, dict) else result) or []
        count = result.get("count") if isinstance(result, dict) else None
        product.searches.append({"query": query, "count": count, "next": result.get("next") if isinstance(result, dict) else None, "ids": row_ids(result)})
        for row in rows:
            if isinstance(row, dict):
                ident = row.get("id") or row.get("listing_id") or row.get("product_id")
                if ident:
                    product.rows.setdefault(str(ident), dict(row))
        if not rows and count is None:
            self.refused.append({"query": query, "length": len(query)})
        return result

    def open_listing(self, ident: str) -> dict[str, str]:
        if ident not in self.facts:
            before = self.steps_left()
            listing = self.shop.view(ident)
            self.view_steps = max(self.view_steps, before - self.steps_left())
            self.keep_listing(ident, listing)
        return self.facts[ident]

    def open_listings(self, ids: list[str]) -> None:
        fresh = [ident for ident in ids if ident not in self.facts]
        if not fresh:
            return
        if not callable(getattr(self.shop, "view_many", None)):
            for ident in fresh:
                self.open_listing(ident)
            return
        for start in range(0, len(fresh), BATCH_LISTINGS):
            chunk = fresh[start:start + BATCH_LISTINGS]
            before = self.steps_left()
            result = self.shop.view_many(chunk)
            self.view_steps = max(1, before - self.steps_left()) / len(chunk)
            listings = result.items() if isinstance(result, dict) else zip(chunk, result or [])
            for ident, listing in listings:
                self.keep_listing(str(ident), listing)

    def keep_listing(self, ident: str, listing: Any) -> None:
        self.facts[ident] = {key: value for key, value in leaf_facts(listing).items() if value != ident}
        self.texts[ident] = full_text(listing)
        self.labels[ident] = {" ".join(words(value)) for value in (listing.values() if isinstance(listing, dict) else []) if isinstance(value, str) and len(value) <= 40}
        self.pages[ident] = {key: value for key, value in listing.items() if not isinstance(value, (dict, list))} if isinstance(listing, dict) else {}

    def steps_left(self) -> int:
        try:
            return int(self.shop.steps_left)
        except Exception:
            return 40

    def steps_free(self) -> int:
        later = (SEARCH_STEPS + 1) * len(self.products) + COMMIT_STEPS + EVENT_STEPS + (CHEAPEST_STEPS if self.cheapest else 0)
        return self.steps_left() - later

    def read_request(self) -> None:
        reading = ask_llm(self.shop, REQUEST_GUIDE + "\n\n# Request\n" + self.request, "request") or {}
        for spot, item in enumerate(reading.get("products") or []):
            if isinstance(item, dict) and item.get("name"):
                stated = [str(line) for line in item.get("stated") or [] if line]
                amount = item.get("quantity")
                self.products.append(Product(f"P{spot + 1}", str(item["name"]), str(item.get("search") or item["name"]), stated, int(amount) if isinstance(amount, (int, float)) and amount > 0 else 1))
        if not self.products:
            self.products.append(Product("P1", "item", " ".join(words(self.request)[:6]), []))
        allowance = reading.get("messages")
        self.messages = int(allowance) if isinstance(allowance, (int, float)) and allowance > 0 else 2
        self.story = str(reading.get("story") or "")
        self.cheapest = reading.get("cheapest") is True
        self.budget = reading.get("budget") if isinstance(reading.get("budget"), (int, float)) else None
        self.in_stock = reading.get("in_stock") is True
        self.budget_unit = reading.get("budget_unit") if reading.get("budget_unit") in ("cents", "whole") else None
        self.trace.append({"kind": "request", "products": [p.name for p in self.products], "messages": self.messages, "story": bool(self.story)})

    def look(self) -> None:
        for spot, product in enumerate(self.products):
            room = (self.steps_free() - self.messages - STEP_SPARE - LOOKUP_STEPS) // (len(self.products) - spot)
            result = self.search(product, product.search)
            ids = row_ids(result)
            if len(ids) < LOOK_LISTINGS and room > LOOK_LISTINGS * self.view_steps and isinstance(result, dict) and result.get("next") is not None:
                more = self.search(product, product.search, result.get("next"))
                ids += [ident for ident in row_ids(more) if ident not in ids]
            if len(ids) < LOOK_MINIMUM * 2 and plain(product.name) != plain(product.search):
                more = self.search(product, product.name)
                ids += [ident for ident in row_ids(more) if ident not in ids]
            if callable(getattr(self.shop, "view_many", None)):
                self.open_listings(ids[:LOOK_LISTINGS])
                product.listings.update({ident: self.facts[ident] for ident in ids[:LOOK_LISTINGS] if ident in self.facts})
            for ident in ids:
                share = min(LOOK_LISTINGS, max(LOOK_MINIMUM, int((room - 1) // self.view_steps)))
                if len(product.listings) >= share or self.steps_left() - self.view_steps < self.messages + STEP_SPARE:
                    break
                product.listings[ident] = self.open_listing(ident)
            settled = {plain(line.split(":")[0]) for line in product.stated if ":" in line}
            product.menu = menu_order(list(product.listings.values()), settled)[:MENU_SIZE]
            product.shelf = self.shelf_of(product, product.rows)
            self.trace.append({"kind": "look", "product": product.label, "opened": len(product.listings), "menu": [key for key, _, _ in product.menu]})

    def rank(self) -> None:
        blocks = []
        for product in self.products:
            lines = [f"- {key} ({carriers} listings): {', '.join(samples)}" for key, carriers, samples in product.menu]
            stated = "; ".join(product.stated) or "nothing"
            blocks.append(f"## {product.label}: {product.name}\n- already stated: {stated}\n" + "\n".join(lines))
        answer = None
        basket = len(self.products) > 1
        if (basket or self.story) and any(product.menu for product in self.products):
            answer = ask_llm(self.shop, HINT_GUIDE + "\n\n# Request\n" + self.request + "\n\n# Products\n" + "\n\n".join(blocks), "hint")
        for product in self.products:
            pick = (answer or {}).get(product.label) or {}
            story = [str(key) for key in pick.get("story") or []] if isinstance(pick, dict) else []
            ranked = [str(key) for key in pick.get("ranked") or []] if isinstance(pick, dict) else []
            groups = pick.get("same") if isinstance(pick, dict) and isinstance(pick.get("same"), list) else []
            product.same = [{plain(key) for key in group} for group in groups if isinstance(group, list) and len(group) > 1]
            product.queue = merged_order(ranked[:FIRST_PICKS] if basket else [], story, [key for key, _, _ in product.menu])
            top = next((carriers for key, carriers, _ in product.menu if product.queue and key == product.queue[0]), 0)
            product.strength = (1.0 if story else 0.0) + top / max(1, len(product.listings))
            self.trace.append({"kind": "rank", "product": product.label, "ranked": ranked, "story": story, "queue": list(product.queue)})

    def next_asks(self) -> list[tuple[Product, str]]:
        waiting = [product for product in self.products if product.queue and not product.told] or [product for product in self.products if product.queue]
        if not waiting:
            return []
        waiting.sort(key=lambda product: (product.told > 0, len(product.asked), -product.strength))
        return [(waiting[0], waiting[0].queue[0])]

    def converse(self) -> None:
        while self.sent < self.messages and self.steps_left() > STEP_SPARE and (self.steps_free() > 0 or self.sent < len(self.products)):
            asks = self.next_asks()
            if not asks:
                break
            for product, key in asks:
                product.queue.remove(key)
                product.asked.append(key)
            basket = len(self.products) > 1
            product, key = asks[0]
            pair = None
            if basket and product.queue and key not in product.bare and self.messages - self.sent < sum(len(other.queue) for other in self.products) + 1:
                # messages are scarcer than keys: two keys of one product share a question, never more and never two products
                pair = product.queue.pop(0)
                product.asked.append(pair)
            if pair:
                message = f"Which {plain(key)} or {plain(pair)} would you like for the {product.called(basket)}?"
            elif key in product.bare:
                shared = any(plain(key) in {plain(other_key) for other_key, _, _ in other.menu} for other in self.products if other is not product)
                message = f"What {plain(key)} do you need{' for the ' + product.called(basket) if shared else ''}?"
            else:
                message = question_text(product.called(basket), key, product.samples(key), basket)
            reply = self.shop.ask(message)
            self.sent += 1
            told = self.read_reply(message, str(reply or ""), [product for product, _ in asks])
            # a property the shop spells several ways is asked once: its other spellings leave the queue
            product.queue = [other for other in product.queue if not any(plain(key) in group and plain(other) in group for group in product.same)]
            if not told and not self.split:
                # an open answer covers the keys whose words hold the asked key's words or sit inside them
                for open_key in [key] + ([pair] if pair else []):
                    asked = content_words(open_key)
                    product.queue = [other for other in product.queue if not (asked and content_words(other) and (content_words(other) <= asked or asked <= content_words(other)))]
            if not told and self.split and key not in product.bare:
                product.bare.append(key)
                product.queue.insert(0, key)
            self.trace.append({"kind": "ask", "message": message, "reply": str(reply or "")[:300], "told": told})

    def read_reply(self, message: str, reply: str, involved: list[Product]) -> list[str]:
        catalogue = "\n".join(f"- {product.label} ({product.name}): {', '.join(key for key, _, _ in product.menu)}" for product in self.products)
        answer = ask_llm(self.shop, REPLY_GUIDE + f"\n\n# Question\n{message}\n\n# Reply\n{reply}\n\n# Listing properties\n{catalogue}", "reply") or {}
        self.split = answer.get("split") is True
        told = []
        for wish in answer.get("wishes") or []:
            if not isinstance(wish, dict) or not wish.get("property") or not wish.get("value"):
                continue
            product = next((item for item in self.products if item.label == wish.get("product")), involved[0])
            prop = str(wish["property"])
            value = str(wish["value"])
            if wish.get("kind") == "listing":
                value = self.value_from_listing(product, prop, value, str(wish.get("search") or "")) or value
                entry = f"{prop}: {RELATION_WORDS.get(str(wish.get('relation')), '')}{value}"
            else:
                entry = f"{prop}: {RELATION_WORDS.get(str(wish.get('relation')), '')}{value}"
            if len(self.products) > 1:
                entry = f"{product.name} - {entry}"
            product.told += 1
            self.told_lines.append((product, prop, entry.split(": ", 1)[-1] if ": " in entry else value, reply))
            words_entry = f"{product.name} - {prop}, in the shopper's words: {reply}" if len(self.products) > 1 else f"{prop}, in the shopper's words: {reply}"
            if words_entry not in self.said:
                self.said.append(words_entry)
            if entry not in self.record:
                self.record.append(entry)
                told.append(entry)
            said = set(words(plain(prop)))
            product.queue = [key for key in product.queue if not (set(words(plain(key))) <= said or said <= set(words(plain(key))))]
        return told

    def value_from_listing(self, product: Product, prop: str, named: str, search: str) -> str | None:
        target = set(words(named))
        if not target:
            return None
        noun = set(words(product.name)) | set(words(prop))
        names = [token for token in words(search) if token in target and token not in noun] or [token for token in words(named) if len(token) > 3 and token not in set(words(self.request)) and token not in noun][:1]
        best = self.closest_listing(target, names)
        if (best is None or not self.owns(best, names)) and self.steps_free() > STEP_SPARE + self.messages - self.sent:
            for query in dict.fromkeys(filter(None, [search.strip(), " ".join(names)])):
                ids = [ident for ident in row_ids(self.search(product, query)) if ident not in self.facts]
                for ident in ids[:FIND_LISTINGS]:
                    if self.steps_left() <= STEP_SPARE:
                        break
                    self.open_listing(ident)
                best = self.closest_listing(target, names)
                if (best is not None and self.owns(best, names)) or self.steps_left() <= STEP_SPARE + 1:
                    break
        if best is None:
            return None
        return next((value for key, value in self.facts[best].items() if plain(key) == plain(prop)), None)

    def owns(self, ident: str, names: list[str]) -> bool:
        return not names or " ".join(names) in self.labels.get(ident, set())

    def closest_listing(self, target: set[str], names: list[str]) -> str | None:
        spread = {token: sum(1 for text in self.texts.values() if token in set(words(text))) for token in target}
        def weight(token: str) -> float:
            return 1.0 / spread[token] if spread[token] else 0.0
        whole = sum(weight(token) for token in target)
        best, rank = None, (False, 0.0)
        for ident, text in self.texts.items():
            have = set(words(text))
            if names and not all(name in have for name in names):
                continue
            mark = (self.owns(ident, names), sum(weight(token) for token in target & have))
            if mark > rank:
                best, rank = ident, mark
        if best is None or (not names and rank[1] < 0.5 * whole):
            return None
        return best

    def store_budget(self) -> float | None:
        if self.budget is None or self.budget_unit is None:
            return None
        if self.budget_unit == "cents":
            return self.budget
        prices = sorted(float(row["price"]) for product in self.products for row in product.rows.values() if isinstance(row.get("price"), (int, float)))
        if not prices:
            return None
        middle = prices[len(prices) // 2]
        return self.budget * 100 if middle > self.budget * 5 else self.budget

    def shelf_of(self, product: Product, listings: dict[str, dict[str, Any]]) -> str | None:
        if product.shelf:
            return product.shelf
        try:
            names = [str(name) for name in (self.shop.categories or [])]
        except Exception:
            names = []
        wanted = set(words(product.name))
        if names and wanted:
            best = max(names, key=lambda name: len(wanted & set(words(name))) / max(1, len(set(words(name)) | wanted)))
            if wanted & set(words(best)):
                return best
        seen: dict[str, int] = {}
        for entry in listings.values():
            label = entry.get("category") or entry.get("department")
            if isinstance(label, str) and label:
                seen[label] = seen.get(label, 0) + 1
        if not seen:
            return None
        label, count = max(seen.items(), key=lambda pair: pair[1])
        return label if count * 2 > sum(seen.values()) else None

    def finish(self) -> dict[str, Any]:
        entries = list(self.record) + self.said
        for product in self.products:
            for line in product.stated:
                entry = f"{product.name} - {line}" if len(self.products) > 1 else line
                if entry not in entries:
                    entries.append(entry)
        lines = []
        for spot, product in enumerate(self.products):
            listings = {}
            for ident in list(product.rows) + [ident for ident in self.facts if ident not in product.rows and any(ident in search["ids"] for search in product.searches)]:
                entry = dict(product.rows.get(ident) or {})
                entry.update(self.pages.get(ident) or {})
                entry.update({"id": ident, "opened": ident in self.facts, "facts": dict(self.facts.get(ident) or {})})
                if ident in self.texts:
                    entry["description"] = self.texts[ident]
                listings[ident] = entry
            requirements = [{"name": line.split(":", 1)[0].strip(), "value": line.split(":", 1)[1].strip(), "source": "request"} for line in product.stated if ":" in line and not plain(line.split(":", 1)[0]).startswith(("price", "quantity", "budget"))]
            requirements += [{"name": prop, "value": value, "source": "hidden_need", "said": reply} for owner, prop, value, reply in self.told_lines if owner is product]
            shelf = self.shelf_of(product, listings)
            spec = {"category": shelf or product.name, "quantity": product.quantity, "budget": self.store_budget(), "in_stock": self.in_stock,
                    "requirements": requirements, "cheapest": self.cheapest, "products_in_order": len(self.products)}
            lines.append({"line": spot, "category": product.name, "shelf": shelf, "record": spec, "searches": product.searches, "listings": listings})
        return {"record": entries[:RECORD_LIMIT], "lines": lines, "refused": self.refused}


def ask_engine(task: dict, shop) -> dict:
    engine = AskEngine(task, shop)
    closed = globals().get("ShopClosed") or RuntimeError
    try:
        engine.read_request()
        engine.look()
        engine.rank()
        engine.converse()
    except closed:
        engine.trace.append({"kind": "closed"})
    except Exception as problem:
        engine.trace.append({"kind": "error", "error": str(problem)[:200]})
    return engine.finish()
# ---- end ask engine ----


# ---- search engine: V2 frozen 9f65c818, unchanged except its words() is named search_words() ----
READER_LLM = "anthropic/claude-haiku-5.5"
BACKUP_LLM = "openai/gpt-6-luna"
MAX_OPTION_SETS = 4
MAX_QUERIES = 14
MAX_PROBES = 6
MAX_PAGES = 6
FULL_PAGES = 9
BATCH = 8
RETRIES = 3
SPARE = 15
# the longest query the world took or refused, kept across the items of one run
SEEN = {"limit": 200}
# a number and the unit text written straight after it, read from store fact values
AMOUNT = re.compile(r"(\d+(?:\.\d+)?)\s*([^\d\s,;/()]*)")

READER_PROMPT = """You prepare a shopper's needs for an online shop's keyword search and for checking listing facts.
The search returns listings whose text holds every query word, so query words must be ones a listing itself would carry.
The shopper's own words find the right listings most often, so keep them; never swap a value for a different one.
For the record below, reply with the product's plain noun as listing titles write it, and with one entry per requirement, in the record's order:
- "query": search phrases for the wanted value, without the noun: each allowed value as the record words it first, then at most one other spelling of it; [] only when the requirement rules values out or sets an upper or lower limit
- When the record allows several values, give each its own query and its own option, all inside that requirement's one entry: exactly one entry per requirement.
- "options": each value that satisfies the requirement as the record words it, plus other spellings of that same value (plural, hyphen, abbreviation); never a neighbouring or similar value
- "excluded": true only when the requirement's value itself says those values are unwanted
- "pointer": true when the value does not state the wanted value but refers to another listing to take it from (a seller's listing to match or differ from); its query and options are then []
- "bound": "at_least" or "at_most" when the value sets a lower or upper limit, "equal" when it names an exact amount, otherwise null
- "value", "unit": the amount and its unit as stated, otherwise null
- "forms": the same amount in each other unit or spelling a listing might use, as [{"value": number, "unit": text}]
Reply with JSON only: {"noun": "...", "needs": [{"query": [...], "options": [...], "excluded": false, "pointer": false, "bound": null, "value": null, "unit": null, "forms": []}]}

Record:
"""


def search_words(text):
    return re.findall(r"[a-z0-9]+", str(text).lower())


def key_words(key):
    return search_words(re.sub(r"([a-z])([A-Z])", r"\1 \2", str(key)))


def stem(word):
    return word[:-1] if len(word) > 3 and word.endswith("s") else word


def present(term, bag):
    return any(w == term or w.startswith(term) for w in bag)


def unit_text(unit):
    # units are compared as written: case, dots and a plural "s" aside
    text = str(unit or "").strip().lower().replace(".", "")
    if text[:1] in ('"', "'"):
        return text[:1]
    return text[:-1] if len(text) > 2 and text.endswith("s") else text


def amount(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def row_text(row):
    # every text the row carries, nested tables as "key value"; the row's own id is no listing text
    parts = []
    for key, value in row.items():
        if value == row.get("id") or isinstance(value, bool):
            continue
        if isinstance(value, dict):
            parts += [f"{k} {v}" for k, v in value.items() if not isinstance(v, (dict, list))]
        elif isinstance(value, list):
            parts += [str(v) for v in value if isinstance(v, (str, int, float)) and not isinstance(v, bool)]
        elif isinstance(value, str):
            parts.append(value)
    return " ".join(parts)


def strong(value):
    # a bare number or a single letter matches too much listing text to prove a value
    return all(len(w) > 1 and not w.isdigit() for w in search_words(value))


def facts_of(row):
    facts = row.get("facts")
    return facts if isinstance(facts, dict) else {}


def json_in(text):
    start, end = str(text or "").find("{"), str(text or "").rfind("}")
    try:
        return json.loads(text[start:end + 1]) if start >= 0 < end else None
    except ValueError:
        return None


class Need:
    def __init__(self, name, value, read=None):
        self.name = str(name or "").lower()
        self.value = str(value or "")
        self.read = isinstance(read, dict)
        read = read if self.read else {}
        self.excluded = read.get("excluded") is True
        # a value that only names another listing holds no words a fitting listing carries: it neither searches nor rules out
        self.pointer = read.get("pointer") is True
        bound = read.get("bound")
        forms = [(amount(read.get("value")), unit_text(read.get("unit")))]
        forms += [(amount(f.get("value")), unit_text(f.get("unit"))) for f in read.get("forms") or [] if isinstance(f, dict)]
        self.forms = [(v, u) for v, u in forms if v is not None]
        self.bound = bound if bound in ("at_least", "at_most", "equal") and self.forms else None
        # the record's own wording stays an option: a reading adds spellings, it never replaces the shopper's words
        own = " ".join(search_words(value))
        options = [" ".join(search_words(o)) for o in read.get("options") or [] if search_words(o)]
        self.options = list(dict.fromkeys([own] + options if own else options))
        query = [" ".join(search_words(q)) for q in read.get("query") or [] if search_words(q)]
        self.query = list(dict.fromkeys([own] + query if own else query))
        if self.pointer:
            self.options, self.query, self.bound = [], [], None

    def query_options(self):
        if self.excluded or self.bound in ("at_least", "at_most"):
            return []
        return [q for q in self.query if q]

    def own_key(self, row):
        # the fact whose name holds every longer word of the need's name, with the fewest other words
        name = {w for w in search_words(self.name) if len(w) > 2}
        keys = [k for k in facts_of(row) if name and name <= set(key_words(k))]
        return min(keys, key=lambda k: len(set(key_words(k)) - name)) if keys else None

    def judge(self, row):
        # 1 shown, 0 unknown, -1 contradicted
        if self.pointer:
            return 0
        if self.bound in ("at_least", "at_most"):
            return self.judge_amount(row)
        bag = {stem(w) for w in search_words(row_text(row))}
        shown = any(all(present(stem(w), bag) for w in search_words(option)) for option in self.options)
        if self.excluded:
            return -1 if shown else 0
        if shown:
            return 1
        if self.bound == "equal":
            return self.judge_amount(row)
        # without a reading nothing is ruled out: the words may be the shopper's, not the listing's
        return -1 if self.read and self.own_key(row) else 0

    def judge_amount(self, row):
        key = self.own_key(row)
        found = AMOUNT.search(str(facts_of(row).get(key, "")).lower()) if key else None
        if not found:
            return 0
        have, unit = float(found.group(1)), unit_text(found.group(2))
        wanted = next((v for v, u in self.forms if u == unit), None)
        if wanted is None:
            return 0
        if self.bound == "equal":
            return 1 if abs(have - wanted) <= 0.02 * max(abs(wanted), 1e-9) else -1
        return 1 if (have >= wanted if self.bound == "at_least" else have <= wanted) else -1


def reader_messages(record):
    requirements = [{"name": r.get("name"), "value": r.get("value")} for r in record.get("requirements") or [] if isinstance(r, dict)]
    return [{"role": "user", "content": READER_PROMPT + json.dumps({"category": record.get("category"), "requirements": requirements})}]


def reading_from(reply, record):
    # a reading that does not cover every requirement in order is no reading
    read = json_in(reply)
    needs = read.get("needs") if isinstance(read, dict) else None
    count = len([r for r in record.get("requirements") or [] if isinstance(r, dict)])
    return read if isinstance(needs, list) and len(needs) == count else None


class Engine:
    def __init__(self, task, shop):
        self.record = task.get("record") or {}
        self.shop = shop
        self.category = str(self.record.get("category") or "").strip()
        self.requirements = [r for r in self.record.get("requirements") or [] if isinstance(r, dict)]
        self.needs = [Need(r.get("name"), r.get("value")) for r in self.requirements]
        self.noun_words = search_words(self.category)
        self.cheapest = bool(self.record.get("cheapest"))
        self.rows = {}
        self.order = []
        self.viewed = set()
        self.credit = {}
        self.counts = {}
        self.sent = set()
        self.and_search = True
        self.sent_text = ""
        self.capped = False
        self.best = (-1, frozenset(), frozenset(), [])
        self.reading = task.get("reading")
        self.seed(task.get("seed"))

    def seed(self, line):
        # what an earlier stage already searched and opened: kept as rows, never asked for again
        if not isinstance(line, dict):
            return
        for search in line.get("searches") or []:
            if isinstance(search, dict) and search_words(search.get("query")):
                self.sent.add(" ".join(search_words(search.get("query"))))
        for ident, listing in (line.get("listings") or {}).items():
            if not isinstance(listing, dict):
                continue
            row = {**listing, "id": listing.get("id") or ident}
            self.take({"rows": [row]})
            if listing.get("opened"):
                self.viewed.add(row["id"])

    def adopt(self, read):
        self.needs = [Need(r.get("name"), r.get("value"), n) for r, n in zip(self.requirements, read["needs"])]
        if search_words(read.get("noun")):
            self.noun_words = search_words(read.get("noun"))

    def steps(self):
        try:
            return int(self.shop.steps_left)
        except Exception:
            return 0

    def call(self, tool, *args, **kwargs):
        # a busy store answers 429 now and then; wait and ask again, but never past the item's steps
        for attempt in range(RETRIES):
            try:
                return getattr(self.shop, tool)(*args, **kwargs)
            except TypeError:
                raise
            except Exception as error:
                if type(error).__name__ == "ShopClosed" or attempt == RETRIES - 1:
                    raise
                time.sleep(2 * (attempt + 1))

    def read_needs(self):
        if not self.requirements:
            return
        given = reading_from(json.dumps(self.reading), self.record) if self.reading else None
        if given:
            self.adopt(given)
            return
        messages = reader_messages(self.record)
        for llm in (READER_LLM, READER_LLM, BACKUP_LLM):
            try:
                reply = self.shop.chat(messages, model=llm, max_tokens=900, temperature=0, reasoning={"enabled": False})
            except Exception as error:
                if type(error).__name__ == "ShopClosed":
                    raise
                continue
            read = reading_from(reply, self.record)
            if read:
                self.reading = read
                self.adopt(read)
                return

    def noun(self):
        return " ".join(self.noun_words)

    def take(self, found, covers=()):
        if not isinstance(found, dict):
            return []
        got = []
        for row in found.get("rows") or []:
            ident = row.get("id")
            if not ident:
                continue
            if ident not in self.rows:
                self.order.append(ident)
            merged = dict(self.rows.get(ident) or {})
            merged.update({k: v for k, v in row.items() if v is not None})
            self.rows[ident] = merged
            self.credit.setdefault(ident, set()).update(covers)
            got.append(ident)
        return got

    def note_count(self, covers, count):
        # an all-words search never finds more by adding words; if it does, query words are no proof
        self.counts[covers] = count
        for other, seen in self.counts.items():
            if (other < covers and seen < count) or (covers < other and count < seen):
                self.and_search = False

    def plan(self):
        noun = self.noun()
        wanted = [i for i, need in enumerate(self.needs) if need.query_options()]
        plans = []
        for size in range(len(wanted), 0, -1):
            for subset in itertools.combinations(wanted, size):
                choices = itertools.product(*[self.needs[i].query_options() for i in subset])
                for values in itertools.islice(choices, MAX_OPTION_SETS):
                    credit = frozenset(i for i, value in zip(subset, values) if strong(value))
                    plans.append((frozenset(subset), credit, " ".join(list(values) + [noun])))
        return plans[:MAX_QUERIES - 1] + [(frozenset(), frozenset(), noun)]

    def shorten(self, terms):
        # the noun's leading words go first (the shelf check still holds them), then trailing words
        terms = list(terms)
        while len(" ".join(terms)) > SEEN["limit"] and len(terms) > 1:
            cut = next((w for w in self.noun_words[:-1] if w in terms), None)
            terms.remove(cut) if cut else terms.pop()
        return terms

    def ask(self, text):
        # an answer without rows and without a count is a refusal, not an empty result
        try:
            found = self.call("search", text)
        except Exception as error:
            if type(error).__name__ == "ShopClosed":
                raise
            return {}
        if not isinstance(found, dict) or (not found.get("rows") and not isinstance(found.get("count"), int)):
            return None
        return found

    def search(self, covers, credit, query, reserve, probe=False):
        terms = self.shorten(dict.fromkeys(search_words(query)))
        if not terms or self.steps() <= reserve or " ".join(terms) in self.sent:
            return None
        self.sent.add(" ".join(terms))
        found = self.ask(" ".join(terms))
        if found is None:
            # the world refused the query (its length, mostly): remember the limit and ask again shorter
            SEEN["limit"] = min(SEEN["limit"], len(" ".join(terms)) - 1)
            terms = self.shorten(terms)
            if self.steps() <= reserve or len(" ".join(terms)) > SEEN["limit"]:
                return None
            self.sent.add(" ".join(terms))
            found = self.ask(" ".join(terms))
            if found is None:
                return None
        if not found:
            return None
        self.sent_text = " ".join(terms)
        got = self.take(found, credit)
        count = found.get("count") if isinstance(found, dict) else None
        if isinstance(count, int) and count > len(found.get("rows") or []) and not found.get("next"):
            self.capped = True
        if probe:
            return found if got else None
        if got and len(credit) > self.best[0]:
            noun = set(self.noun_words)
            self.best = (len(credit), credit, covers, [w for w in terms if w not in noun])
        self.note_count(covers, count if isinstance(count, int) else len(got))
        return found if got else None

    def page_on(self, found, query, credit, reserve):
        # a loose search can match hundreds of rows: a few pages, then the rest of the share goes elsewhere
        page, pages = found.get("next"), 1
        count, shown = found.get("count"), len(found.get("rows") or []) or 1
        # a small result is read whole; the allowance is in pages, since a store may show only a few rows a page
        most = float("inf") if isinstance(count, int) and -(-count // shown) <= FULL_PAGES else MAX_PAGES
        while page and self.steps() > reserve and pages < most:
            pages += 1
            found = self.call("search", query, page=page)
            self.take(found, credit)
            page = found.get("next") if isinstance(found, dict) else None

    def probe(self, reserve):
        # a world that shows one page only: open other windows of the shelf with words its titles use
        if not self.capped:
            return
        _, credit, covers, values = self.best
        used = {stem(w) for w in values + self.noun_words}
        counts = collections.Counter(stem(w) for i in self.rows if self.on_shelf(i)
                                     for w in set(search_words(self.rows[i].get("title")))
                                     if len(w) > 3 and not w.isdigit())
        tried = 0
        for word, _ in counts.most_common():
            if tried >= MAX_PROBES or self.steps() <= reserve:
                return
            if word in used:
                continue
            if not self.cheapest and sum(1 for i in self.pool() if self.full(i)) >= 10:
                return
            tried += 1
            self.search(covers, credit, " ".join(values + [word] + self.noun_words[-1:]), reserve, probe=True)

    def browse(self, reserve):
        if not self.category or self.steps() <= reserve:
            return
        budget = self.record.get("budget")
        quantity = self.record.get("quantity") or 1
        cap = budget / quantity if isinstance(budget, (int, float)) and budget > 0 else None
        try:
            found = self.call("browse", self.category, max_price=cap)
        except TypeError:
            found = self.call("browse", self.category)
        self.take(found)
        page = found.get("next") if isinstance(found, dict) else None
        while self.cheapest and page and self.steps() > reserve:
            try:
                found = self.call("browse", self.category, page=page, max_price=cap)
            except TypeError:
                found = self.call("browse", self.category, page=page)
            self.take(found)
            page = found.get("next") if isinstance(found, dict) else None

    def fits_money(self, ident):
        row = self.rows[ident]
        if row.get("in_stock") is False:
            return False
        budget, price = self.record.get("budget"), row.get("price")
        quantity = self.record.get("quantity") or 1
        return not (isinstance(budget, (int, float)) and isinstance(price, (int, float)) and price * quantity > budget)

    def on_shelf(self, ident):
        label = str(self.rows[ident].get("category") or "").lower()
        return not label or not self.category or label == self.category.lower()

    def marks(self, ident):
        row = self.rows[ident]
        credited = self.credit.get(ident, set()) if self.and_search else set()
        out = []
        for i, need in enumerate(self.needs):
            mark = need.judge(row)
            if mark == 0 and i in credited:
                mark = 1
            out.append(mark)
        return out

    def score(self, ident):
        marks = self.marks(ident)
        price = self.rows[ident].get("price")
        price = price if isinstance(price, (int, float)) else float("inf")
        return (marks.count(-1), -marks.count(1), price if self.cheapest else 0, marks.count(0), self.order.index(ident))

    def pool(self):
        return sorted((i for i in self.rows if self.fits_money(i) and self.on_shelf(i)), key=self.score)

    def full(self, ident):
        return all(m == 1 for m in self.marks(ident))

    def absorb(self, ident, listing):
        self.viewed.add(ident)
        if isinstance(listing, dict):
            self.rows[ident] = {**self.rows[ident], **{k: v for k, v in listing.items() if v is not None}}
            # a listing that lacks a value its search words claimed: this world's search words are no proof
            if any(self.needs[i].judge(self.rows[ident]) == -1 for i in self.credit.get(ident, ())):
                self.and_search = False

    def view_leaders(self):
        many = callable(getattr(self.shop, "view_many", None))
        while self.steps() > 0:
            fresh = [i for i in self.pool()[:10] if i not in self.viewed]
            if not fresh:
                return
            if many:
                chosen = sorted(fresh, key=lambda i: 0 in self.marks(i), reverse=True)[:BATCH]
                listings = self.call("view_many", chosen)
                if isinstance(listings, list):
                    listings = {row.get("id"): row for row in listings if isinstance(row, dict)}
                for ident in chosen:
                    self.absorb(ident, (listings or {}).get(ident))
                continue
            ident = next((i for i in fresh if 0 in self.marks(i)), fresh[0])
            self.absorb(ident, self.call("view", ident))

    def run(self):
        start = self.steps()
        # with steps to spare, the plain noun goes first, so the item holds rows even if the reading below fails
        if start >= SPARE and not self.reading:
            self.search(frozenset(), frozenset(), self.noun(), 0)
        self.read_needs()
        numeric = any(need.bound in ("at_least", "at_most") for need in self.needs)
        reserve = start // 3 if numeric else min(2, start // 5)
        done = set()
        every = frozenset(i for i, need in enumerate(self.needs) if need.query_options())
        for covers, credit, query in self.plan():
            if any(covers < d for d in done) and len(self.pool()) >= 10:
                continue
            found = self.search(covers, credit, query, reserve)
            if found is None:
                continue
            if self.cheapest or (isinstance(found.get("count"), int) and found["count"] <= 3 * len(found.get("rows") or [])):
                self.page_on(found, self.sent_text, credit, reserve)
            if covers:
                done.add(covers)
            if not self.cheapest and sum(1 for i in self.pool() if self.full(i)) >= 10:
                break
        self.probe(reserve)
        # a cheapest pick needs the cheap end of the shelf unless a search already held every wanted value
        if len(self.pool()) < 10 or (self.cheapest and not (every and every in done)):
            self.browse(reserve)
        self.view_leaders()
        return self.pool()[:10]


def search_engine(task, shop):
    engine = Engine(task, shop)
    try:
        candidates = engine.run()
    except Exception:
        # ShopClosed past the item's steps, or a store tool that fails: keep what the item found
        candidates = engine.pool()[:10]
    return {"candidates": candidates}
# ---- end search engine ----


def list_of_ids_tool(schema: dict[str, Any]) -> bool:
    required = schema.get("required") or []
    spec = (schema.get("properties") or {}).get(required[0]) if len(required) == 1 else None
    return isinstance(spec, dict) and spec.get("type") == "array" and ((spec.get("items") or {}).get("type") == "string")


class ShopClosed(Exception):
    pass


SEARCH_SLOT = re.compile(r"^q$|query|text|wanted|keyword|term|phrase|search|find|words|look", re.I)
PAGE_SLOT = re.compile(r"start|page|offset|from|first|row|skip|cursor|token|after|marker", re.I)
FREE_MODE = re.compile(r"phrase|text|free|keyword|query|search|words", re.I)
COUNT_KEY = re.compile(r"count|total|found|matches", re.I)
NEXT_KEY = re.compile(r"next|after|more|cursor", re.I)
CAP_SLOT = re.compile(r"price|cost|ceiling|max|budget|cap", re.I)
BRAND_SLOT = re.compile(r"brand|marque|maker|make|label", re.I)


class WorldShop:
    # the ask engine's shop, served through this agent's own tool layer so steps, records and the ledger stay in step
    def __init__(self, shell: Shell, books: Books) -> None:
        self.shell, self.books = shell, books
        self.answers: dict[str, Any] = {}
        browse = self.browse_tool()
        spec = ((books.schemas.get(browse[0]) or {}).get("properties") or {}).get(browse[1]) or {} if browse else {}
        self.categories: list[str] = [v for v in spec.get("enum") or [] if isinstance(v, str)]
        if not books.role("view"):
            # setup left the batch view unmapped: a tool whose one required input is a list of ids does that job
            listed = [name for name, schema in books.schemas.items() if name not in books.roles.values() and list_of_ids_tool(schema)]
            if len(listed) == 1:
                books.roles["view"] = listed[0]
        if books.role("view") and self.multi_view_slot():
            self.view_many = self.view_many_listings

    @property
    def steps_left(self) -> int:
        return self.books.steps_left()

    def _call(self, role: str, args: dict[str, Any]) -> Any:
        return self._call_named(self.books.role(role), args)

    def _call_named(self, name: str | None, args: dict[str, Any]) -> Any:
        books = self.books
        if not name or books.finished or books.dead or books.steps_left() <= 0 or self.shell.seconds_left() < WRAP_UP_SECONDS:
            raise ShopClosed()
        args, problems = check_against(args, books.schemas[name], "args")
        if problems:
            return None
        # the engines often repeat a search or a view the other one already made: the shop's answer is reused, no step spent,
        # while nothing in the store has moved; any other call or any event wipes what was kept
        looks = name in {books.role("search"), books.role("details"), books.role("view"), (self.browse_tool() or [None])[0]}
        asked = name + json.dumps(args, sort_keys=True)
        if looks and asked in self.answers:
            return self.answers[asked]
        before = len(books.records)
        _send_chunk(self.shell, books, [(f"ask_engine_{books.step}", name, args)])
        answer = books.records[-1] if len(books.records) > before else None
        if not looks or event_parts(answer)[1]:
            self.answers.clear()
        elif answer is not None:
            self.answers[asked] = answer
        return answer

    def _slots(self, role: str) -> tuple[dict[str, Any], list[str]]:
        schema = self.books.schemas.get(self.books.role(role) or "") or {}
        props = schema.get("properties") or {}
        texts = [k for k, v in props.items() if isinstance(v, dict) and is_text_type(v) and "enum" not in v]
        return props, texts

    def chat(self, messages: list[dict[str, Any]], model: str | None = None, temperature: float = 0, max_tokens: int = 1500, **params: Any) -> str:
        body = request_body(model or ASK_READ_MODEL, messages, None, max_tokens)
        body.update(params)
        _llm.timeout = int(max(8, min(60, self.shell.seconds_left() - FORCE_SECONDS)))
        result = _llm.post_verbose("/inference/chat/completions", body)
        message = usable_message(self.books, result.data)
        if message is None:
            raise RuntimeError(str(result.error)[:200])
        return str(message.get("content") or "")

    def search(self, query: str, page: Any = None) -> dict[str, Any]:
        props, texts = self._slots("search")
        slot = next((k for k in texts if SEARCH_SLOT.search(k)), texts[0] if texts else None)
        if not slot:
            return {"rows": [], "count": None, "next": None}
        args: dict[str, Any] = {slot: query}
        required = set((self.books.schemas.get(self.books.role("search") or "") or {}).get("required") or [])
        for key, spec in props.items():
            if not isinstance(spec, dict) or key == slot:
                continue
            if spec.get("enum"):
                fitting = [v for v in spec["enum"] if isinstance(v, str) and FREE_MODE.search(v)]
                if fitting or key in required:
                    args[key] = (fitting or spec["enum"])[0]
            elif page is not None and PAGE_SLOT.search(key) and (set(declared_types(spec) or []) & {"integer", "number"} or isinstance(page, str)):
                args[key] = page
        return self.rows_of(self._call("search", args))

    def rows_of(self, observation: Any) -> dict[str, Any]:
        rows, order = priced_rows([observation]) if observation is not None else {}, []
        for node in dict_nodes(observation):
            ident = _row_ident(node)
            if ident and ident not in rows and any(_short_text(v) for v in node.values()):
                # a row whose price sits under a key this code does not read is still a row the engines can open
                rows[ident] = dict(node)
            if ident and ident in rows and ident not in order:
                order.append(ident)
        listed = [{"id": i, "title": next((v for k, v in rows[i].items() if isinstance(v, str) and NAME_KEY.search(str(k))), ""),
                   "price": rows[i].get("price"), "in_stock": rows[i].get("in_stock"), "category": rows[i].get("category"),
                   "description": max((v for v in rows[i].values() if isinstance(v, str)), key=len, default="")[:400]} for i in order]
        top = observation if isinstance(observation, dict) else {}
        count = next((v for k, v in top.items() if is_number(v) and COUNT_KEY.search(str(k))), None)
        after = next((v for k, v in top.items() if NEXT_KEY.search(str(k)) and v not in (None, False)), None)
        return {"rows": listed, "count": count, "next": after}

    def view(self, ident: str) -> dict[str, Any]:
        props, texts = self._slots("details")
        slot = next((k for k in texts if IDENT_KEY.search(k)), texts[0] if len(texts) == 1 else None)
        if not slot:
            return {}
        self._call("details", {slot: ident})
        view = listing_view(self.books, ident) or {}
        if not view.get("facts") and not self.books.viewer_tried:
            # the mapped details tool can show the offer only; a spare tool taking just one id may show the properties
            self.books.viewer_tried = True
            for name in self.spare_viewers():
                spare = next(iter((self.books.schemas.get(name) or {}).get("required") or []), None)
                self._call_named(name, {spare: ident})
                view = listing_view(self.books, ident) or {}
                if view.get("facts"):
                    self.books.roles["details"] = name
                    break
        return view

    def spare_viewers(self) -> list[str]:
        # a tool setup gave another reading role can still be the one that shows properties; only tools that act are left out
        used = {self.books.role(r) for r in ("details", "search", "message", "cart", "order", "decline")} - {None}
        out = []
        for name, schema in self.books.schemas.items():
            required = schema.get("required") or []
            spec = (schema.get("properties") or {}).get(required[0]) if len(required) == 1 else None
            if name not in used and isinstance(spec, dict) and is_text_type(spec) and "enum" not in spec:
                out.append(name)
        return out[:2]

    def browse_tool(self) -> tuple[str, str, str | None, str | None, str | None, dict[str, Any]] | None:
        searcher = self.books.role("search")
        used = set(self.books.roles.values()) - {searcher}
        for tool in self.books.tools:
            name = tool_name(tool)
            schema = self.books.schemas.get(name) or {}
            props = schema.get("properties") or {}
            if name in used:
                continue
            numeric = lambda k: isinstance(props[k], dict) and bool(set(declared_types(props[k]) or []) & {"integer", "number"})
            cap = next((k for k in props if numeric(k) and CAP_SLOT.search(k)), None)
            # a mode switch holds a few fixed words; a shelf list holds many
            modes = {k: v["enum"] for k, v in props.items() if isinstance(v, dict) and k != cap and 1 < len(v.get("enum") or []) <= 4}
            text = [k for k, v in props.items() if isinstance(v, dict) and k not in modes and k != cap and ("string" in (declared_types(v) or []) or v.get("enum"))]
            if name == searcher:
                # the search tool browses a shelf only through a mode switch, and its free query slot is no shelf
                text = [k for k in text if not SEARCH_SLOT.search(k)]
                if not modes:
                    continue
            shelf = next((k for k in text if not BRAND_SLOT.search(k) and not PAGE_SLOT.search(k)), None)
            if cap and shelf:
                brand = next((k for k in text if BRAND_SLOT.search(k)), None)
                page = next((k for k in props if k not in (cap, shelf, brand) and PAGE_SLOT.search(k)), None)
                fixed = {k: next((v for v in values if isinstance(v, str) and not FREE_MODE.search(v)), values[0])
                         for k, values in modes.items() if name == searcher or k in (schema.get("required") or [])}
                return name, shelf, cap, brand, page, fixed
        return None

    def browse(self, category: str, page: Any = None, max_price: Any = None, brand: Any = None) -> dict[str, Any]:
        found = self.browse_tool()
        if not found:
            return self.search(category, page)
        name, shelf, cap, brand_slot, page_slot, fixed = found
        spec = (self.books.schemas[name].get("properties") or {}).get(shelf) or {}
        choices = [v for v in spec.get("enum") or [] if isinstance(v, str)]
        label = max(choices, key=lambda v: _shared_stems(_stems(category), _stems(v))) if choices else category
        args: dict[str, Any] = {**fixed, shelf: label}
        if is_number(max_price):
            args[cap] = max_price
        if brand and brand_slot:
            args[brand_slot] = brand
        if page is not None and page_slot:
            args[page_slot] = page
        observation = self._call_named(name, args)
        return self.rows_of(observation)

    def multi_view_slot(self) -> str | None:
        props = (self.books.schemas.get(self.books.role("view") or "") or {}).get("properties") or {}
        return next((k for k, v in props.items() if isinstance(v, dict) and v.get("type") == "array"), None)

    def view_many_listings(self, ids: list[str]) -> dict[str, dict[str, Any]]:
        slot, out, pending = self.multi_view_slot(), {}, list(ids)
        spec = ((self.books.schemas.get(self.books.role("view") or "") or {}).get("properties") or {}).get(slot) or {}
        # a store may answer a long list with nothing at all: stay at its stated maximum, else six
        size = spec["maxItems"] if isinstance(spec.get("maxItems"), int) and 0 < spec["maxItems"] < 6 else 6
        while pending and slot:
            chunk = pending[:size]
            self._call("view", {slot: chunk})
            got = {i: listing_view(self.books, i) for i in chunk}
            got = {i: v for i, v in got.items() if v and v.get("facts")}
            if not got and size > 1:
                size //= 2     # a store can refuse a long list; a shorter one usually goes through
                continue
            out.update(got)
            pending = pending[len(chunk):]
        return out

    def ask(self, message: str) -> str:
        args = message_arguments(self.books, message)
        if not args:
            raise ShopClosed()
        observation = self._call("message", args)
        heard, _ = shopper_lines(self.books, self.books.role("message") or "", args, observation)
        return " ".join(heard)


DESCRIBING_PROMPT = """These needs were read from a shopper's request. A request can fix one property by pointing at a different listing, e.g. "the same X as the Y that Z sells, the one with A and B" or "any X except what the page for Z's Y, with A and B, says". Inside that sentence, A and B only tell which listing is meant. A reader often copies them into the needs list as if they were wishes ("material: A", "color: B"); they are not wishes for the product being bought.
Find the request's sentence that names the other listing. Each need, other than the pointer itself, whose words come from that sentence is describing.
Answer in JSON only: {"pointer": number of the need that points at another listing, or null, "describing": [numbers of the describing needs]}. With no such pointer, answer {"pointer": null, "describing": []}."""


def drop_describing_needs(shell: Shell, books: Books, entries: list[str]) -> list[str]:
    if len(entries) < 2:
        return entries
    numbered = "\n".join(f"{n}. {e}" for n, e in enumerate(entries, 1))
    reply = judge_ask(shell)(DESCRIBING_PROMPT, f"SHOPPER'S REQUEST:\n{books.query()}\n\nNEEDS:\n{numbered}")
    if not isinstance(reply, dict) or not isinstance(reply.get("pointer"), int):
        return entries
    pointer = reply["pointer"]
    cut = {n for n in reply.get("describing") or [] if isinstance(n, int) and n != pointer and 1 <= n <= len(entries)}
    if not 1 <= pointer <= len(entries) or len(cut) >= len(entries) - 1:
        return entries
    _merge_new(books.dropped, [entries[n - 1] for n in sorted(cut)])
    return [e for n, e in enumerate(entries, 1) if n not in cut]


def same_property(left: str, right: str, owner: str) -> bool:
    # one key's words all inside the other's ("height" and "sneaker_height"): two readings of one property
    a, b = (_stems(text.split(":")[0]) - _stems(owner) for text in (left, right))
    return bool(a and b) and (a <= b or b <= a)


def run_ask_engine(shell: Shell, books: Books) -> None:
    if not (books.role("message") and books.role("search")) or shell.seconds_left() < 150:
        return
    asked_before = books.messages_used
    handoff = ask_engine({"task_id": "", "request": books.query()}, WorldShop(shell, books))
    books.handoff = handoff
    targets = product_targets(books)
    stated: list[str] = []
    for line in handoff.get("lines") or []:
        label = f"{line.get('category') or ''} {line.get('shelf') or ''}"
        owner = max(targets, key=lambda t: _shared_stems(_stems(t[0] + " " + t[1]), _stems(label)))[0] if len(targets) > 1 else ""
        for need in (line.get("record") or {}).get("requirements") or []:
            if not (need.get("name") and need.get("value")):
                continue
            entry = (f"{owner} " if owner else "") + f"{need['name']}: {need['value']}"
            if need.get("source") == "hidden_need":
                # the engine's answer for a property replaces any other settling of it, e.g. a pointer the loop resolved from a different listing
                books.revealed = [r for r in books.revealed if r == entry or not same_property(r, entry, owner)]
                if books.current is not None:
                    books.current = [r for r in books.current if r == entry or not same_property(r, entry, owner)]
                _merge_new(books.revealed, [entry])
            else:
                stated.append(entry)
    if stated:
        # the engine's own structured reading of the request replaces the setup reader's, which can turn story details into conditions
        books.stated = drop_describing_needs(shell, books, stated)
        books.current = None
    books.ledger_version += 1
    # the engine owns the questions once it has asked: the in-loop planner stays off so no question is spent twice
    if books.messages_used > asked_before:
        books.planner_on = False
    books.trace.append({"kind": "ask_engine", "record": handoff.get("record"), "messages": books.messages_used, "steps": books.step,
                        "opened": [i for line in handoff.get("lines") or [] for i, x in (line.get("listings") or {}).items() if x.get("opened")]})


SEARCH_STEPS_CAP = 8


class ProductShop:
    # one product's share of the store steps, so the search for one product cannot starve the next
    def __init__(self, shop: WorldShop, share: int) -> None:
        self.shop, self.share, self.start = shop, share, shop.books.step
        self.categories = shop.categories
        if hasattr(shop, "view_many"):
            self.view_many = shop.view_many

    @property
    def steps_left(self) -> int:
        return max(0, min(self.share - (self.shop.books.step - self.start), self.shop.books.steps_left() - len(product_targets(self.shop.books)) - 5))

    def __getattr__(self, name: str) -> Any:
        return getattr(self.shop, name)


def run_search_and_judge(shell: Shell, books: Books) -> None:
    targets = product_targets(books)
    lines = (books.handoff or {}).get("lines") or []
    shop = WorldShop(shell, books)
    books.picked = {}
    for index, (product, noun) in enumerate(targets):
        if books.finished or shell.seconds_left() < WRAP_UP_SECONDS + 60:
            break
        label = product + " " + noun
        scored = [(_shared_stems(_stems(label), _stems(f"{line.get('category')} {line.get('shelf')}")), line) for line in lines]
        best = max(scored, key=lambda pair: pair[0], default=(0, None))
        # a hand-off line seeds only the product it is about
        seed = best[1] if best[0] > 0 or len(targets) == 1 else None
        quantity = product_quantity(books, product)
        needs = needs_for(books, product)
        record = dict((seed or {}).get("record") or {})
        record.update(category=record.get("category") or noun or product, quantity=quantity, cheapest=books.plan.get("priority") == "cheapest",
                      requirements=[dict(n, source="request") for n in needs])
        # search gets a fair share but never more than SEARCH_STEPS, so the judge and the cart always have steps left
        share = max(4, min(SEARCH_STEPS_CAP, (books.steps_left() - len(targets) - 5) // max(1, len(targets) - index)))
        try:
            found = search_engine({"record": record, "seed": seed}, ProductShop(shop, share)).get("candidates") or []
        except ShopClosed:
            found = []
        mine = [i for i in found if len(targets) < 2 or product_of(books, i) == product]
        listings = [v for v in (listing_view(books, i) for i in mine[:4]) if v and v.get("facts")]
        if not listings or shell.seconds_left() < WRAP_UP_SECONDS + 30:
            continue
        verdicts = judge_listings(judge_ask(shell), bounded_needs(shell, books, product), listings, quantity)
        for listing in listings:
            stamp = f"{listing['id']}|{books.ledger_version}"
            verdict = verdicts.get(listing["id"]) or {}
            if verdict.get("fits") is True:
                books.passed.add(stamp)
            elif verdict.get("fits") is False and verdict.get("binding"):
                books.binding_fails[stamp] = f"Check: {listing['id']} fails {verdict.get('why')} by its own record. Then: this is final; never cart or order it."
        choice = choose_listing(verdicts, listings, quantity, books.plan.get("priority") == "cheapest")
        books.trace.append({"kind": "search_judge", "product": product, "candidates": found[:6], "choice": choice,
                            "verdicts": {k: (v.get("fits"), v.get("why")) for k, v in verdicts.items()}})
        if choice and books.steps_left() > len(targets) + 4:
            if code_cart(shell, books, choice, quantity):
                books.picked[product] = choice


def handoff_note(books: Books) -> str:
    handoff = getattr(books, "handoff", None) or {}
    lines = handoff.get("lines") or []
    if not lines:
        return ""
    out = [f"Already done before your first turn (steps used {books.step}, shopper messages used {books.messages_used}; the questions are finished):"]
    for line in lines:
        opened = [i for i, x in (line.get("listings") or {}).items() if x.get("opened")]
        seen = [i for i, x in (line.get("listings") or {}).items() if not x.get("opened")]
        out.append(f"- {line.get('category')}: seen in results but not opened: {', '.join(seen[:12]) or 'none'}; opened:")
        for ident in opened[:10]:
            view = listing_view(books, ident) or {}
            facts = "; ".join(f"{k}: {str(v)[:40]}" for k, v in list((view.get("facts") or {}).items())[:12])
            out.append(f"  {ident} | {str(view.get('title') or '')[:70]} | price {view.get('price')} | in stock {view.get('in_stock')} | {facts}")
    picked = getattr(books, "picked", None) or {}
    if picked:
        out.append("Code already searched each product and put these checked fits in the cart: "
                   + "; ".join(f"{(p or 'the item')}: {i}" for p, i in picked.items()) + ". Then: unless a line is missing or a need changed, order now.")
    out.append("Search further and open more only where these do not settle the pick; every pick still passes the separate check.")
    return "\n".join(out)


def setup_phase(shell: Shell, books: Books) -> None:
    content = setup_content(books)
    plan: dict[str, Any] = {}
    # the cheap reader first; a dearer one only when the reading misses the tool roles or the product
    for model in [SETUP_MODEL] + SETUP_FALLBACKS:
        if shell.seconds_left() < 200 and plan:
            break
        reading = reply_object(shell.infer([{"role": "user", "content": content}], None, [model], 4000)) or {}
        if reading:
            plan = reading
        if isinstance(plan.get("roles"), dict) and plan.get("roles") and str(plan.get("product") or "").strip():
            break
    adopt_reading(books, plan)
    books.trace.append({"kind": "plan", "plan": books.plan, "roles": books.roles})


def opening_phase(shell: Shell, books: Books) -> None:
    first_user = opening_message(books, shell.seconds_left)
    opening = books.policy.get("opening")
    if opening is not None:
        books.records.append(opening)
        gather_ids(opening, books.seen_ids)
    books.messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": first_user}]


NEXT_TURN = "next"
STOP = "stop"


class Turn:
    def __init__(self) -> None:
        self.idle = 0
        self.failures = 0
        self.counter = 0
        self.content = ""
        self.calls: list[dict[str, Any]] = []


def step_guard(shell: Shell, books: Books, turn: Turn) -> str | None:
    if books.finished or books.dead or books.steps_left() <= 0:
        return STOP
    if shell.seconds_left() < FORCE_SECONDS:
        force_finish(shell, books)
        return STOP
    return None


def step_answer(shell: Shell, books: Books, turn: Turn) -> str | None:
    if not books.pending_reply:
        return None
    text, books.pending_reply = books.pending_reply, ""
    return NEXT_TURN if send_reply(shell, books, text) else None


def step_plan(shell: Shell, books: Books, turn: Turn) -> str | None:
    return NEXT_TURN if books.planner_on and ask_planned(shell, books) else None


KEEP_RESULTS = 6
OLD_RESULT_CHARS = 1500


def trim_history(books: Books) -> None:
    # older tool results shrink so a long basket episode stays fast; every record is still kept for the checks
    results = [m for m in books.messages if m.get("role") == "tool"]
    for message in results[:-KEEP_RESULTS]:
        text = message.get("content")
        if isinstance(text, str) and len(text) > OLD_RESULT_CHARS:
            message["content"] = text[:OLD_RESULT_CHARS] + " ...(older result shortened; open the listing again if a detail is needed)"


def step_think(shell: Shell, books: Books, turn: Turn) -> str | None:
    trim_history(books)
    message = shell.infer(books.messages, turn_tools(books), [MAIN_MODEL, BACKUP_MODEL, MAIN_MODEL], 4000)
    if message is None:
        turn.failures += 1
        if turn.failures < 2:
            return NEXT_TURN
        force_finish(shell, books)
        return STOP
    turn.failures = 0
    turn.calls = []
    for call in message.get("tool_calls") or []:
        if not isinstance(call, dict):
            continue
        turn.counter += 1
        function = call.get("function") or {}
        arguments = function.get("arguments")
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments or {})
        turn.calls.append({"id": call.get("id") or f"call_{turn.counter}", "type": "function",
                           "function": {"name": function.get("name"), "arguments": arguments}})
    content = message.get("content") or ""
    turn.content = content if isinstance(content, str) else dump(content)
    return None


def step_hold_commit(shell: Shell, books: Books, turn: Turn) -> str | None:
    committing = {books.role("cart"), books.role("order"), books.role("decline")} - {None}
    if not (books.planner_on and any(c["function"]["name"] in committing for c in turn.calls)):
        return None
    # too few opened listings make the question plan a guess
    if (not books.early_bounced and property_menu(books.records, bool(wanted_products(books.plan)))[1] < 3
            and books.steps_left() > 12 and shell.seconds_left() > WRAP_UP_SECONDS + 60):
        books.early_bounced = True
        books.messages.append({"role": "assistant", "content": turn.content or None, "tool_calls": turn.calls})
        note = (SKIP + "Check: you want to choose already. Then: first open the details of two or three more listings of the requested kind"
                " (several per reply, favouring ones unlike those seen); the shopper questions are planned from what they show.")
        post_replies(shell, books, [(c["id"], note) for c in turn.calls])
        return NEXT_TURN
    asked = False
    # before the first commit every remaining question goes out, each chosen knowing the answers before it
    while books.planner_on and ask_planned(shell, books, force=True):
        asked = True
    return NEXT_TURN if asked else None


def step_nudge(shell: Shell, books: Books, turn: Turn) -> str | None:
    if turn.calls:
        turn.idle = 0
        return None
    turn.idle += 1
    books.messages.append({"role": "assistant", "content": turn.content or "(no tool call)"})
    books.messages.append({"role": "user", "content": "Check: your reply called no tool. Then: call at least one supplied tool in every reply and carry on.\n"
                           + status_line(books, shell.seconds_left)})
    if turn.idle < 3:
        return NEXT_TURN
    force_finish(shell, books)
    return STOP


def step_act(shell: Shell, books: Books, turn: Turn) -> str | None:
    books.trace.append({"kind": "model", "content": turn.content[:2000], "calls": [c["function"] for c in turn.calls]})
    books.messages.append({"role": "assistant", "content": turn.content or None, "tool_calls": turn.calls})
    post_replies(shell, books, run_calls(shell, books, turn.calls))
    return NEXT_TURN


TURN_STEPS = (step_guard, step_answer, step_plan, step_think, step_hold_commit, step_nudge, step_act)


def turn_loop(shell: Shell, books: Books) -> None:
    turn = Turn()
    for _ in range(MAX_TURNS):
        outcome = None
        for step in TURN_STEPS:
            outcome = step(shell, books, turn)
            if outcome is not None:
                break
        if outcome == STOP:
            break


def closing_phase(shell: Shell, books: Books) -> None:
    if not books.finished and not books.dead:
        force_finish(shell, books)
    books.trace.append({"kind": "end", "finished": books.finished, "steps": books.step, "cost": round(books.cost, 5), "seconds": shell.elapsed()})


def run_episode(shell: Shell, books: Books) -> list[dict[str, Any]]:
    if books.finished:
        return books.trace
    setup_phase(shell, books)
    run_ask_engine(shell, books)
    run_search_and_judge(shell, books)
    opening_phase(shell, books)
    turn_loop(shell, books)
    closing_phase(shell, books)
    return books.trace


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    shell = Shell()
    books = open_books(problem_data)
    shell.books = books
    try:
        return run_episode(shell, books)
    except Exception as exc:
        books.trace.append({"kind": "crash", "error": f"{type(exc).__name__}: {exc}"[:500]})
        try:
            if not books.finished and not books.dead:
                force_finish(shell, books)
        except Exception as again:
            books.trace.append({"kind": "crash", "error": f"{type(again).__name__}: {again}"[:500]})
        return books.trace


__all__ = ["agent_main"]
