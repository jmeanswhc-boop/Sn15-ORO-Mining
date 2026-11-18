from __future__ import annotations

import json
import re
import time
from types import SimpleNamespace
from typing import Any, Callable

from src.agent.proxy_client import ProxyClient

MAIN_MODEL = "anthropic/claude-haiku-5.5"
BACKUP_MODEL = "openai/gpt-6-luna"
SETUP_MODEL = "anthropic/claude-sonnet-5.5"
JUDGE_MODEL = "anthropic/claude-sonnet-5.5"
READ_MODEL = "anthropic/claude-haiku-5.5"

TASK_SECONDS = 265.0
COST_SOFT = 0.06            # spend in dollars past which the checks move to the cheaper LLM
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

Check: you are judging whether a listing meets a requirement (this decides the score). Then: count it as met only when the listing's own property for it gives the value. Key spellings vary (spaces, dots, underscores, prefixes) and one property may sit under several keys; any key naming the property counts. Values match on meaning: a single element of a comma-separated list suffices, singular equals plural, close synonyms count, case is ignored. A word that merely begins the same way does not count, nor does a broader or different class (girls is not women; plain unisex is not female).
Check: the listing has no property key at all for a requirement. Then: only then use the title or description as evidence; if the listing's own property says otherwise, it fails even though the title or description shows the value.
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

VERIFY_PROMPT = """Your job is the final gate ahead of an order by a shopping assistant: the task earns a 1 only when the item satisfies all the requirements given below, and your verdict must rest ONLY on the records provided.
Check: you test a requirement. Then: it holds only if the item's own property for it shows the value; keys may be spelled in many ways (spaces, dots, underscores, prefixes) and one property may turn up under several keys, any of which counts; values are compared by sense, so a single comma-list element is enough, singular/plural and near synonyms are fine, and letter case is irrelevant, while a look-alike word or a wider or other class fails.
Check: support comes only from the title or description. Then: accept it only if the item has no property key for that requirement at all; where its own property disagrees, the requirement fails despite the title.
Check: the value appears under some other, related property. Then: it does not satisfy the requirement.
Check: figures are involved. Then: convert units first; a limit's edge value passes; a range passes only if all of it is inside the limit.
Check: there is an exclusion or a closed list. Then: enforce it.
Check: the kind of product. Then: the item has to be the kind requested; when listings show a category label, it must be the label that suits the request's wording taken as a whole, not a narrower or adjacent label.
Check: the item's own property for a requirement shows some other value. Then: the requirement is failed, whatever the title implies.
Check: cost and availability. Then: price times quantity may not exceed the budget (being equal passes); if the records show it out of stock, the item fails.
Check: a simple wish ("ideally", "would be nice") appears marked as not required. Then: do not treat it as a requirement.
Check: a detail from the story tempts you (an age group, a gender, a size, who will use it). Then: ignore it; test only the requirements given below.
Check: the shopper wants several different products. Then: test the item only against the requirements for its own kind; one opening with another product's name is not its concern, and its quantity is that of its kind.
Check: records contradict one another (price or stock, say). Then: trust the latest one.
Check: the item's own record says a property comes from another listing, or a requirement asks for another listing's value. Then: judge that property by the matching record under OTHER LISTINGS NAMED; when no matching record is given, that property alone is no reason to set fits=false.
Check: you are about to set fits=false. Then: do so only for a requirement the records plainly fail or that no record speaks to at all; a lacking stock or price record is never grounds for failure.
Reply: ONE JSON object and no other text: {"item": the item's id, "fits": true or false, "problems": [short strings, each naming a requirement that is not met or not shown]}."""

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
# wording of a refusal that means the item cannot be bought any more
GONE_WORDS = re.compile(r"sold out|out of stock|unavailable|no longer|zero left|none left|all gone|is gone|it's gone|now gone|"
                        r"gone from stock|dried up|ran dry|run dry|cleared out|cleared from|bought the last|can't be ordered|cannot be ordered|"
                        r"grayed out|greyed out|hit zero|no units|none remaining|no .{0,12}remaining|been sold|been taken|snapped up|"
                        r"in stock any ?more|not in stock|isn't in stock|no stock|stock.{0,20}\bzero\b|last one", re.I)
PRICE_WORDS = re.compile(r"price|cost|charg|priced|bumped|gone up|went up|come down|dropped|cents|dearer|cheaper", re.I)
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
FIRST_PERSON = re.compile(r"\b(I|I'm|I'd|I've|I'll|my|me|mine)\b")
ASKS_BACK = re.compile(r"\?|\b(confirm|let me know|tell me|acknowledge|reply|answer|get back to me)\b", re.I)
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
        order_checked=False, decline_checks=0, cheaper_checks=0, research_checks=0, early_bounced=False,
        opened=set(), capped=set(), recent=[],
        mode_key=None, order_mode=None, decline_mode=None, passed=set(), swapped=set(),
        asked_props={}, asked_at=-1, first_done=set(), cart_counts={}, pointers=[], order_refused=False,
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
    return head + unbought + _pace_warning(books, left, seconds_left)


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


def recurring_labels(records: list[Any]) -> str:
    counts: dict[str, dict[str, int]] = {}
    for record in records:
        for node in dict_nodes(record):
            if len(node) < 4:
                continue
            for key, item in node.items():
                if isinstance(item, str) and 0 < len(item) <= 40 and not is_code(item):
                    bucket = counts.setdefault(str(key), {})
                    bucket[item] = bucket.get(item, 0) + 1
    lines = []
    for key, bucket in counts.items():
        if sum(bucket.values()) < 4 or not 2 <= len(bucket) <= 8 or max(bucket.values()) < 2:
            continue
        ordered = sorted(bucket.items(), key=lambda pair: -pair[1])
        lines.append(f"- {key}: " + " | ".join(f"{v} (x{n})" for v, n in ordered))
    return "\n".join(lines[:12])


def records_of_item(records: list[Any], item: str) -> list[str]:
    found: list[str] = []
    for record in records:
        pending: list[tuple[Any, int, Any, bool]] = [(record, 0, None, False)]
        while pending and len(found) < 8:
            node, depth, listing, in_option = pending.pop()
            if depth > 6:
                continue
            if isinstance(node, list):
                pending.extend((child, depth + 1, listing, in_option) for child in reversed(node))
                continue
            if not isinstance(node, dict):
                continue
            if item in [v for v in node.values() if isinstance(v, str)]:
                # an option is reported through its parent listing
                shown = listing if (listing is not None and in_option and listing is not node) else node
                text = dump(shorten(shown, 260))[:3000]
                if text not in found:
                    found.append(text)
                continue
            if any(isinstance(v, str) and IDENT_KEY.search(str(k)) and looks_like_id(v) for k, v in node.items()):
                listing, in_option = node, False
            branches = [(child, depth + 1, listing, in_option or bool(OPTION_KEY.search(str(k)))) for k, child in node.items()]
            pending.extend(reversed(branches))
    return found[-3:]


PRICE_KEY = re.compile(r"price|cost", re.I)
STOCK_KEY = re.compile(r"stock|avail|buy|on.?hand", re.I)


def row_price(node: dict[str, Any]) -> Any:
    if is_number(node.get("price")):
        return node["price"]
    keys = sorted((k for k, v in node.items() if is_number(v) and PRICE_KEY.search(str(k))), key=lambda k: ("total" in str(k).lower(), len(str(k))))
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


def cheaper_unopened(books: Books) -> list[tuple[str, Any]]:
    if books.plan.get("priority") != "cheapest" or not books.cart_ids:
        return []
    rows = priced_rows(books.records)
    held = rows.get(books.cart_ids[-1])
    if not held:
        return []
    wanted = books.plan.get("quantity")
    quantity = wanted if isinstance(wanted, int) and wanted > 0 else 1
    budget = books.budget_now if is_number(books.budget_now) else None
    candidates = []
    for ident, row in rows.items():
        if ident in books.opened or ident in books.cart_ids or row["price"] >= held["price"]:
            continue
        if row.get("in_stock") is False:
            continue
        if budget is not None and row["price"] * quantity > budget:
            continue
        if held.get("category") and row.get("category") and row.get("category") != held.get("category"):
            continue
        candidates.append((row["price"], ident))
    candidates.sort()
    return [(ident, price) for price, ident in candidates[:5]]


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
    steps_allowed = _plain_number(plan, "step_budget")
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


def verify_prompt(books: Books, item: str, purpose: str, line: str, records: list[str], named: list[str]) -> str:
    labels = recurring_labels(books.records)
    cart_view = "" if books.last_cart_view is None else "NEWEST CART VIEW:\n" + dump(books.last_cart_view)[:2000]
    return compose(
        VERIFY_PROMPT,
        "THE SHOPPER'S REQUEST:\n" + books.query(),
        "REQUIREMENTS IN FORCE (stated, revealed, changed):\n" + requirements_line(books),
        f"ITEM TO {purpose}: {item}\nSTORE RECORDS OF THIS ITEM (oldest first):\n" + "\n".join(records),
        named and "OTHER LISTINGS NAMED (by this item's notes or by a requirement):\n" + "\n".join(full_record(books, n)[:2500] for n in named),
        line and "ITS CART LINE (Check: it names an option or variant. Then: judge that option by its own values, price and stock): " + line,
        cart_view,
        labels and "LABELS REPEATED ACROSS THIS STORE'S LISTINGS (with counts; Check: the item's category label. Then: it must match the request's wording taken whole, not a neighbour):\n" + labels,
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
            result = shell.judge(prompt, 400, [READ_MODEL, JUDGE_MODEL]) or {}
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
    elif reply and books.roles.get("message") and ASKS_BACK.search(" ".join(fresh)):
        # only a line that asks for something earns an answer
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
            if isinstance(value, int) and value not in {q for _, q in products}:
                return f"the request calls for these quantities: {', '.join(f'{n} x{q}' for n, q in products)}, yet this call sets {value}."
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
    notes = ["Check: THE ORDER WAS REFUSED. Then: know the episode goes on with the cart as it was; apply each change the refusal names, repair the cart (drop any line no longer acceptable, such as a sold-out item), re-check the new pick's stock and price, and order again."]
    said = refusal if isinstance(refusal, str) else dump(refusal)
    if PRICE_WORDS.search(said) and not GONE_WORDS.search(said):
        notes.append("Check: the refusal never says the item is gone and only reports a new price (or a change the item still satisfies). Then: treat the item as still available; read its current price, and if price times quantity stays within budget and every requirement still holds, order it again at once. A price change is never a reason to decline.")
    return notes


def _change_note(observation: dict[str, Any]) -> str:
    lines: list[str] = []
    for value in observation.values():
        if isinstance(value, list) and value and all(isinstance(v, str) for v in value):
            lines += long_lines(value)
    return "Check: this result holds a shopper message (a change). Then: apply it: " + " | ".join(lines[:4]) if lines else ""


def event_parts(observation: Any) -> tuple[Any, list[str]]:
    # an event rides on any result as {<payload key>: result, <lines key>: [sentences]}, under changing key names
    if not isinstance(observation, dict) or len(observation) != 2:
        return observation, []
    lists = [k for k, v in observation.items() if isinstance(v, list) and v and all(isinstance(x, str) for x in v)]
    if len(lists) != 1:
        return observation, []
    payload = next(k for k in observation if k != lists[0])
    return observation[payload], [x for x in observation[lists[0]] if x.strip()]


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
        wrapped = len(observation) <= 3 and any(isinstance(v, (dict, list)) for v in observation.values())
        for value in observation.values():
            if isinstance(value, list) and value and all(isinstance(v, str) for v in value):
                heard += long_lines(value)
            elif isinstance(value, str) and len(value.split()) >= 4 and "refusal" in observation:
                heard.append(value)
            elif (isinstance(value, str) and wrapped and len(value.split()) >= 6
                  and (FIRST_PERSON.search(value) or name == books.role("order"))):
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
            left = self.seconds_left()
            if left < FORCE_SECONDS - 10:
                break
            result = self._attempt(model, left, messages, tools, max_tokens)
            message = usable_message(self.books, result.data)
            if message is not None:
                return message
            failure = str(result.error)[:300] if result.error else "empty"
            self.books.trace.append({"kind": "inference_failure", "model": model, "error": failure})
        return None

    def judge(self, prompt: str, max_tokens: int, models: list[str] | None = None) -> dict[str, Any] | None:
        models = models or [JUDGE_MODEL, MAIN_MODEL]
        if self.books.cost > COST_SOFT and models[0] == JUDGE_MODEL:
            # a costly episode still checks, only on the cheaper LLM
            models = [READ_MODEL, JUDGE_MODEL]
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
    result = shell.judge(prompt, 1500, [READ_MODEL, JUDGE_MODEL])
    return absorb_interpretation(books, result, fresh, question) if result else ""


def verify_item(shell: Shell, books: Books, item: str, purpose: str, line: str = "") -> str:
    if shell.seconds_left() < WRAP_UP_SECONDS + 20:
        return ""
    named = named_listings(books, item)
    key = f"{item}|{line}|{books.ledger_version}|{','.join(named)}"
    if key in books.verdicts:
        return books.verdicts[key]
    records = records_of_item(books.records, item)
    if not records:
        return ""
    result = shell.judge(verify_prompt(books, item, purpose, line, records, named), 1200)
    books.trace.append({"kind": "verify", "item": item, "result": result})
    verdict = ""
    if result and result.get("fits") is True:
        books.passed.add(f"{item}|{books.ledger_version}")
    if result and result.get("fits") is False:
        problems = [str(p) for p in result.get("problems") or []][:5]
        verdict = (f"Check: an independent review finds {item} short of the requirements: " + "; ".join(problems)
                   + ". Then: pick another fully verified fit, unless you are sure the review is mistaken.")
    books.verdicts[key] = verdict
    return verdict


def verify_cart(shell: Shell, books: Books) -> str:
    if not books.cart_ids:
        return ""
    # several products mean one cart line each, and each line is checked
    lines = books.cart_ids[-4:] if wanted_products(books.plan) else books.cart_ids[-1:]
    for item in lines:
        verdict = verify_item(shell, books, item, "ORDER", books.cart_lines.get(item, ""))
        if verdict:
            return verdict
    return ""


def audit_fits(shell: Shell, books: Books) -> tuple[list[str], str]:
    if shell.seconds_left() < WRAP_UP_SECONDS + 15:
        return [], ""
    blocks = audit_blocks(books.records)
    if not blocks:
        return [], ""
    result = shell.judge(audit_prompt(books, blocks), 1200)
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


def _guard_cart(shell: Shell, p: Pending) -> str:
    books, name, args = p.books, p.name, p.args
    properties = books.schemas[name].get("properties") or {}
    count = cart_count(books, name, args)
    if is_number(count) and count <= 0:
        return ""
    checks_line = books.steps_left() > 3
    if checks_line and (books.add_signature is None or enum_signature(args, properties) == books.add_signature):
        target = _cart_target(args, properties)
        stamp = f"{target}|{books.ledger_version}"
        if target and stamp not in books.cart_checked:
            books.cart_checked.add(stamp)
            verdict = verify_item(shell, books, target, "ADD TO THE CART", cart_line_text(args, target))
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
    filled = [k for k in args if not (isinstance(properties[k], dict) and "enum" in properties[k])]
    if len(filled) < 2 or any(k not in args for k in schema.get("required") or []):
        return None
    return args


def code_cart(shell: Shell, books: Books, item: str) -> str:
    tool = books.role("cart")
    wanted = books.plan.get("quantity")
    count = wanted if isinstance(wanted, int) and not isinstance(wanted, bool) and wanted >= 1 else 1
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
    return None, name, scrub_decline(books, args, name)


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
    if books.cheaper_checks < 2 and books.steps_left() > 10 and shell.seconds_left() > WRAP_UP_SECONDS + 20:
        cheaper = cheaper_unopened(books)
        if cheaper:
            books.cheaper_checks += 1
            return ("Not run yet (no step used). Check: the request wants the cheapest fitting item, yet these cheaper listings from your results were never"
                    " opened with the details tool: " + ", ".join(f"{i} ({p})" for i, p in cheaper) + ". Then: open their details (several per reply);"
                    " if any meets every requirement, put the cheapest such one in the cart instead, and if none does, order again."), name, args
    if not books.order_checked and books.steps_left() > 2 and shell.seconds_left() > WRAP_UP_SECONDS:
        books.order_checked = True
        verdict = verify_cart(shell, books)
        if verdict:
            return HOLD + verdict, name, args
        return (HOLD + "Final review before ordering. Check: (1) the cart has exactly the lines the request needs (normally one: the chosen item at the"
                " required quantity) and nothing more; (2) since your last cart change you re-read this item's stock and price with a tool that shows"
                " them, and it is in stock with price times quantity inside the current budget; (3) its own properties satisfy every stated"
                " requirement, every need the shopper revealed and every announced change; (4) when the cheapest or best was requested, nothing"
                " better that fits exists among all you could find; (5) every procedural rule of the request is met. Then: check whatever is"
                " still open now, or else call the order tool again."), name, args
    return None, name, args


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
        if any(sample.count(",") >= 3 for sample in samples.get(name, [])):
            continue        # a value that enumerates the options on offer
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
    result = shell.judge(prompt, 300, [READ_MODEL, JUDGE_MODEL]) or {}
    pick = _to_integer(result.get("pick")) if not is_number(result.get("pick")) else int(result.get("pick"))
    return names[pick - 1] if isinstance(pick, int) and 1 <= pick <= len(names) else names[0]


def next_question(shell: Shell, books: Books) -> tuple[str, str]:
    first = next((str(a) for a in books.plan.get("ask_first") or [] if a and str(a) not in books.first_done), "")
    if first:
        books.first_done.add(first)
        result = shell.judge(compose(FIRST_PROMPT, "THE SHOPPER'S REQUEST:\n" + books.query(), "POINT TO SETTLE: " + first), 300,
                             [READ_MODEL, JUDGE_MODEL]) or {}
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
    said = " ".join(lines)
    price_only = bool(PRICE_WORDS.search(said)) and not GONE_WORDS.search(said)
    if (reply and not reply.get("finished") and lines and order and action["name"] == order and books.cart_used
            and price_only and books.steps_left() > 0):
        # turned away only over a price change: the cart is still good, so order once more
        reply = forced_call(shell, books, action)
    if reply and not reply.get("finished") and isinstance(reply.get("observation"), dict) and lines and decline and books.steps_left() > 0:
        forced_call(shell, books, decline_action(books, decline))


def setup_phase(shell: Shell, books: Books) -> None:
    content = setup_content(books)
    plan = reply_object(shell.infer([{"role": "user", "content": content}], None, [SETUP_MODEL, BACKUP_MODEL], 1500)) or {}
    if not isinstance(plan.get("roles"), dict) and shell.seconds_left() > 200:
        # every later step leans on this reading, so it gets one more try
        plan = reply_object(shell.infer([{"role": "user", "content": content}], None, [SETUP_MODEL, MAIN_MODEL], 2000)) or plan
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


def step_think(shell: Shell, books: Books, turn: Turn) -> str | None:
    message = shell.infer(books.messages, turn_tools(books), [MAIN_MODEL, BACKUP_MODEL, MAIN_MODEL], 2500)
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
