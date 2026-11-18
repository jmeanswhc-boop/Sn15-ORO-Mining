"""Polaris v11: a shopping agent for ORO world sessions (one tool call per environment step).

The world's tools, their argument names and the keys of their results change from world to world, so the agent first
has a model read the tool list into roles (search, listing, datasheet, cart, checkout, decline, message), then works
a fixed plan: read the request into requirements and budgets, search, settle the store category, ask the shopper
about the properties a held-back need most likely concerns, read the datasheets of the cheapest listings that could
fit, judge them by their own facts against every requirement, cart the cheapest fitting one in the quantity asked
for and check out - or decline, naming no listing, when none fits. Shopper lines can arrive with any result and
above all with the first checkout (a changed budget, a dropped requirement, the carted listing gone out of stock);
each is read as it comes and the plan is redone from what is known.
"""

from __future__ import annotations

import json
import math
import re
import time
from typing import Any

from src.agent.proxy_client import ProxyClient

PROXY = ProxyClient(timeout=60, max_retries=2)   # the client retries a rate limit or a fault itself (5 s); Gate paces the requests
STRONG = ["anthropic/claude-sonnet-5.5", "openai/gpt-6-sol", "z-ai/glm-5.3", "moonshotai/Kimi-K2.6-TEE", "deepseek-ai/DeepSeek-V3.2-TEE"]
QUICK = ["anthropic/claude-haiku-5.5", "openai/gpt-6-luna", "z-ai/glm-5.3", "deepseek-ai/DeepSeek-V3.2-TEE", "zai-org/GLM-5.2-TEE"]
TIME_LIMIT = 265.0            # seconds of wall time the agent allows itself per task (the network allows 300)
FINISH_ROOM = 5               # steps kept for carting, tidying the cart, checking out and one retry
FIRST_ROOM = 14               # steps kept before the first finish attempt once a fit is known: the store's changes arrive there, and the re-plan needs its own carting, re-reading and ordering
JUDGE_BATCH = 5               # listings judged in one model call
STRONG_JUDGES = 5             # judge calls on the strong model; later ones use the quick model (cost)
SCREEN_BATCH = 24             # listings screened by their search rows in one model call
SEARCH_PAGES = 3              # pages of a broad search
DEEP_PAGES = 6                # pages of the narrowest search for a required value, when its matches fit in them
ASK_MOST = 6                  # questions at most, however many the shopper allows
QUESTION_PROPS = 8            # properties one question names: the shopper answers about the held-back one when it is named
SHEETS_FIRST = 5              # datasheets read before the questions, so they can draw on the shelf's facts
PACE = 1.2                    # seconds between one task's requests: the proxy admits 30 a second per sandbox, shared by up to 30 tasks
PACE_MOST = 4.0               # the spacing after repeated rate limits
COMPARE_MOST = 4              # listings read in one step through a side-by-side tool, when the world has one
SPEND_QUICK = 0.040           # USD of inference in one task after which only the quick models are used
SPEND_STOP = 0.050            # USD after which no model is asked at all: 90 such tasks stay under the run key's 5 USD
CALL_COST = {"strong": 0.007, "quick": 0.0006}   # a call's cost assumed when the reply carries no usage cost



def now() -> float:
    return time.monotonic()


REFERENCE_WORDS = re.compile(r"written on the page|figure from the page|listing it defers|naming some other listing|the page (?:of|for) |pointed to|counterpart of the one|falls away from|given on the listing|on the listing for|listing for the .{2,60} sold by|sold by .{2,40}, with|whatever'?s on the page|wherever that page|mates up with|'s listing of the|land on from the page|you'?ll find at|following each listing|whatever you land on", re.I)
CONFIRM_WORDS = re.compile(r"confirm|check with me|run (?:it|that|the \w+) (?:by|past) me|before (?:you |placing|ordering|buying|paying|checking out)|my (?:ok|okay|go-?ahead|say-?so|nod|approval|blessing)|approv|sign off|show me (?:the|your) (?:basket|cart|pick|choice)|let me (?:see|know|check) (?:the|what|before)|tell me (?:what|before)|green light", re.I)
AVOID_ASKING = re.compile(r"\b(?:brand|make|model|warranty|condition|sku|options?|variants?|price|cost|stock|availability|seller|shop|vendor|dealer)\b", re.I)
EMPTY_WORDS = re.compile(r"basket is empty|cart is empty|bag is empty|nothing in (?:your|the) (?:basket|cart|bag)|empty right now|no items in", re.I)
DEFER_WORDS = re.compile(r"\bfor (?:its |the |a )?[a-z][a-z /&-]{2,40}, (?:look|refer|go|take|use|check|see|rely|turn)\b|\btake (?:its|the) [a-z][a-z /&-]{2,40} from\b|\b(?:should|is to|are to) be taken from\b|\bisn't taken from\b|\bnot (?:from |by |to )?this (?:item|listing|one|product|rug|phone|cable|charger|piece|jacket|powder|lamp|dresser|cabinet|organiser|organizer|handle|lock|headset|adaptor|adapter|sneaker|shoe|bar)\b|\brather than (?:to |from |by )?this\b|\binstead of this\b|\bgo by (?:the|another|a) |\brefer to (?:the|a|another) |\blook (?:to|at) (?:the|a|another) |\bfrom a different (?:product|listing|item)\b", re.I)
POINTER_HOPS = 6              # pages followed along a chain of "for <property>, look to <listing>" pointers
RANKED_HITS = 400             # a search answering with this many matches or more is ranked: its first pages hold the best, deeper ones noise


def words(text: Any) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(text or "").lower())


def first_json(text: str) -> Any:
    """The first JSON object or array in a model's reply."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fenced:
        text = fenced.group(1)
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        while start >= 0:
            depth, quoted, escaped = 0, False, False
            for at in range(start, len(text)):
                ch = text[at]
                if quoted:
                    if escaped:
                        escaped = False
                    elif ch == "\\":
                        escaped = True
                    elif ch == '"':
                        quoted = False
                    continue
                if ch == '"':
                    quoted = True
                elif ch == opener:
                    depth += 1
                elif ch == closer:
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[start:at + 1])
                        except ValueError:
                            break
            start = text.find(opener, start + 1)
    return None


class Gate:
    """Spaces one task's requests to the proxy and paces its retries. The proxy admits 30 requests a second from the
    whole sandbox, where up to 30 tasks run at once, so a task keeps to about one a second and slows down after a 429."""

    def __init__(self, seed: str) -> None:
        self.gap = PACE
        self.last = 0.0
        self.spread = (sum(map(ord, seed)) % 97) / 97.0     # a fixed offset per task, so tasks do not retry in step

    def wait(self) -> None:
        delay = self.last + self.gap - now()
        if delay > 0:
            time.sleep(delay)
        self.last = now()

    def limited(self, attempt: int) -> None:
        """After a 429: a wider spacing, and a pause that grows with the attempt."""
        self.gap = min(PACE_MOST, self.gap * 1.5)
        time.sleep(min(8.0, 2.0 ** attempt) * (0.7 + 0.6 * self.spread))

    def eased(self) -> None:
        self.gap = max(PACE, self.gap * 0.95)


class Models:
    """Chat completions through the proxy: the same model again after a rate limit or a fault, the next model when one
    is refused or answers with something that is not JSON; replies are parsed as JSON."""

    def __init__(self, deadline: float, gate: Gate) -> None:
        self.deadline = deadline
        self.gate = gate
        self.calls = 0
        self.spent = 0.0                # USD of inference this task, from the replies' usage.cost

    def ask(self, system: str, user: str, strong: bool = False, max_tokens: int = 1500) -> Any:
        if self.spent >= SPEND_STOP:
            return None                 # the task's share of the run key is used up: the rest goes on what is known
        if self.spent >= SPEND_QUICK:
            strong = False
        for model in (STRONG if strong else QUICK):
            for attempt in range(3):
                if self.deadline - now() < 10:
                    return None
                body: dict[str, Any] = {"model": model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                                        "max_tokens": max_tokens, "temperature": 0}
                if model.startswith(("openai/", "z-ai/")):
                    body["reasoning"] = {"effort": "low"}
                    body["max_tokens"] = max_tokens + 800
                self.gate.wait()
                self.calls += 1
                try:
                    outcome = PROXY.post_verbose("/inference/chat/completions", json_data=body)
                except Exception:
                    outcome = None
                reply = outcome.data if outcome is not None else None
                if isinstance(reply, dict):
                    self.gate.eased()
                    usage = reply.get("usage") if isinstance(reply.get("usage"), dict) else {}
                    cost = usage.get("cost")
                    self.spent += float(cost) if isinstance(cost, (int, float)) and not isinstance(cost, bool) else CALL_COST["strong" if strong else "quick"]
                    try:
                        choice = reply["choices"][0]
                        content = choice["message"].get("content") or ""
                        cut = choice.get("finish_reason") == "length"
                    except (KeyError, IndexError, TypeError, AttributeError):
                        content, cut = "", False
                    parsed = None if cut else first_json(content)   # a cut-off answer would parse to a fragment
                    if parsed is not None:
                        return parsed
                    if cut and attempt == 0:
                        max_tokens *= 2                     # cut off: once more with room for the whole answer
                        continue
                    break                                   # an answer that is not JSON: the next model
                status = ((outcome.error if outcome is not None else None) or {}).get("status")
                if status == 429:
                    self.gate.limited(attempt)              # rate limited: the same model after a pause
                elif status is None or status >= 500:
                    time.sleep(1.0 + attempt)               # no answer or a fault upstream: the same model again
                else:
                    break                                   # refused outright (a bad request, a model not allowed): the next model
        return None


TOOLS_GUIDE = """You map the tools of an online store's assistant interface to roles. Tool names, argument names and
descriptions are arbitrary in every store; read each tool's description and argument schema. Reply with one JSON
object only:
{
 "search": {"tool": "<name>", "text_arg": "<argument for keywords or null>", "text_args_fixed": {<other arguments a keyword search must send, e.g. a mode enum set to its keyword value>},
            "category_arg": "<argument for a category/department or null>", "brand_arg": "<argument for a brand/maker or null>",
            "price_arg": "<argument for a maximum price or null>", "fields_args_fixed": {<arguments a category/brand/price search must send, e.g. the mode enum set to its field value>},
            "page_arg": "<argument for the page number, offset or next-page token, or null>"},
 "browse": {"tool": "<a different tool that lists one category/department/shelf without keywords, if the keyword search tool cannot filter by category>", "category_arg": "<name>", "page_arg": "<name or null>", "price_arg": "<argument capping the price, or null>", "brand_arg": "<argument for a brand, maker, manufacturer or seller, or null>", "fixed": {}} or null,
 "listing": {"tool": "<tool returning one product in full>", "id_arg": "<name>", "fixed": {<a mode enum set to its value, if any>}} or null,
 "datasheet": {"tool": "<tool returning a product's properties/specifications/facts>", "id_arg": "<name>", "fixed": {}} or null,
 "compare": {"tool": "<tool returning several products in full at once, given a list of product ids>", "ids_arg": "<name of the list argument>"} or null,
 "cart_add": {"tool": "<name>", "id_arg": "<the product-id argument, or the list argument when the tool takes a list of lines>", "quantity_arg": "<name or null>", "fixed": {<e.g. an operation enum set to its add value>},
              "line_id_field": "<when the tool takes a list of line objects: the field of a line holding the product id, else null>", "line_quantity_field": "<its quantity field, else null>"},
 "cart_remove": {"tool": "<name>", "id_arg": "<name>", "quantity_arg": "<name or null>", "fixed": {<e.g. the operation enum set to its remove value>}} or null,
 "cart_view": {"tool": "<name>"} or null,
 "checkout": {"tool": "<tool that places the order / buys what is in the cart>", "fixed": {<e.g. an outcome enum set to its buy/commit value>}},
 "decline": {"tool": "<tool that ends the task without buying>", "remark_arg": "<name or null>", "fixed": {<e.g. the outcome enum set to its decline/refuse value>}},
 "message": {"tool": "<tool that sends the shopper a message or question>", "text_arg": "<name>", "fixed": {}} or null
}
Use only tool and argument names from the list and only enum values the schema allows. Leave out arguments a role
does not need. If one tool does several roles (one finishing tool with an enum for buying or declining, one cart tool
with an enum for adding or removing), name it in each role with the enum value that selects that role in "fixed"."""

REQUEST_GUIDE = """You read a shopper's request to an online store assistant and write down exactly what it demands.
The shopper writes casually and mixes in stories; extract only what decides the purchase. A request may ask for one
item or for several different items at once (each with its own quantity): list every item to buy, in the order the request
names them. Reply with one JSON object:
{
 "items": [{
   "item": "<what to buy, in 1-4 plain words, singular>",
   "category": "<the store category or department the request names for it, in its words, or ''>",
   "requirements": [{"property": "<what about the item, e.g. 'blade material'>", "value": "<the required value, e.g. 'carbon steel'>", "quote": "<the request's exact words demanding it>"}],
   "exclusions": [{"property": "<property>", "value": "<value the item must NOT have>", "quote": "<the request's exact words>"}],
   "quantity": <how many units of this item to order: the number the request states outright for this item, default 1; people, rooms or occasions in the story do not multiply it unless the request says to buy one for each>,
   "search_queries": ["<3 to 4 short keyword searches, 1-3 words each: the item word with one key value first, then the item word alone>"],
   "references": [{"property": "<a property whose required value the request takes from ANOTHER listing>", "shop": "<the shop, seller or brand that listing is from>", "category": "<the kind of item that other listing is, e.g. 'sardines', 'chilled food', 'desk fan'>", "attributes": ["<details given to tell that listing apart>", ...], "negated": <true when the value must be anything OTHER than that listing's>, "relation": "<'same' (the same value), 'differ' (anything but it), 'within' (a figure inside a tolerance of it: 'inside 2 inches of'), or 'pairs' (a value made to go with / matching / compatible with it, as a rule on that page says)>", "tolerance": "<for 'within': the figure and unit, e.g. '2 inches'; else ''>"}],
   "linked": [{"property": "<a property whose value is set by ANOTHER ITEM OF THIS SAME REQUEST, e.g. 'compatible devices' when a charger must suit the phone being bought>", "to_item": "<that other item, in the words used above>"}]
 }],
 "budget": <the highest total price allowed for the whole order, a number exactly as written, or null>,
 "currency": "<the unit word or sign the prices use, as written in the request>",
 "priority": "cheapest" (only when the request asks outright for the cheapest or least expensive fitting item; a price cap, or money being tight, is not that) | "none" | "<another ranking the shopper asks for, e.g. 'longest battery life'>",
 "stocked_only": <true when the items must be in stock / on hand>,
 "message_budget": <how many messages the assistant may send the shopper, or null>,
 "step_budget": <how many moves/actions/steps the assistant may make, or null>,
 "hidden_needs_hint": "<who it is for or why, and anything suggesting unstated preferences, or ''>",
 "confirm_first": <true only when the request says to confirm with or notify the shopper before placing the order>,
 "say_if_none": <true when the request says to tell the shopper when nothing fits rather than offer a substitute>,
 "procedures": ["<any other explicit instruction about how to proceed, other than the above>"]
}
A requirement is a property the item itself must have that the shopper demands outright (it has to, must, should be,
only, nothing but, can be one of several values and nothing else): material, style, colour family, a size, a feature, a
country of origin, a maker... Any detail the shopper says they will check or verify themselves is a requirement. A
statement of the item's kind within its category is a requirement too ('the kind of lamp is a floor lamp' -> lamp type:
floor lamp), and so is a use or occasion the item must suit when the shopper says they will check it ('it is for a
hiking trip, I will verify' -> use: hiking). A choice between values is one requirement with both ('finish: matte or
satin'). Being in stock, the price and the category are not requirements, and neither are words that only repeat the
name of the store category the item is in. Details of the shopper's story (what broke, what they own, who they shop
with) are never requirements. Never invent a requirement. When the request takes a property's value from another listing ("the same colour as
the Anmoanno sardines in cans", "whatever packaging the Dorvosa chilled item comes in", "the shape given on the listing for
the X sold by Y", "matching whatever's on the page for Risabel's listing of the wardrobe organisers - the one with material
plastic and style contemporary - or wherever that page ends up sending you", "a type that mates up with whatever's on the page
for Vexvexsaqi's listing of the protein powders - the one with protein per serving 50 g", and the negated "definitely not
whatever's on the page for Voshibel's listing of the denim jackets - the one with pattern stripe"), put it in "references"
(not in requirements) with that listing's shop, kind and the details given ("the one with ..."). A property one item must
share with or suit another item of the same request (a cable compatible with the phone) goes in "linked". When the item
must fit a device the shopper already has and names by model ('a case for my ROT3110', 'it has to suit my MOZ-7460'), that
is a requirement: {"property": "compatible model", "value": "<the model code exactly as written>"}."""

CATEGORY_GUIDE = """An online store files every listing under one category. Given what the shopper wants and the
categories of listings a search turned up (with how many), say which category the wanted item belongs to: the one a
store would file exactly that kind of item under. Name a second only if the item truly fits both. Reply with JSON:
{"categories": ["<exact category text from the list>"]}"""

SCREEN_GUIDE = """You screen store listings from their search rows (title, category, brand, description) before their
datasheets are read. For each listing say how likely it is to have every requirement: "likely" (the row shows it is the
wanted kind of item and nothing in it contradicts a requirement), "possible" (the kind fits but the row does not show
the requirements either way), "unlikely" (another kind of item, or the row states a value that contradicts a requirement).
When a store category is given, a listing filed under it is the wanted kind, whatever its title calls the item.
Reply with JSON: {"rows": [{"id": "<id>", "fit": "likely|possible|unlikely"}]}"""

QUESTION_GUIDE = """A shopper asked an online store assistant for an item and holds back a firm wish about one property of it, which
they reveal only when asked about that very property; asked about anything else they say they do not mind, or that they did
not follow, so each question is a guess at that property. From the properties listed (the facts the store's listings of
this kind carry, with how many carry each and sample values) choose up to 5 to ask about, likeliest first. The held-back
wish is about a property a buyer of this kind of item decides on and that the request leaves open: the item's type or kind
within its category, its material, colour, style, pattern or construction, how it closes or is worn, a connector, plug or
battery type, a display or screen size, a dimension limit (width, depth, height), a flavour or main ingredient, who it is
for (gender, age), a weight or volume, a pack or packaging type, a resolution, a processor or memory type, an operating
system. Across many such shoppers it was most often the material, the colour, the style or type, a construction or
pattern, a connector, plug or battery type, a display size or resolution, a width, depth, height, weight or volume limit, a
fruit type, flavour or dietary need, a packaging or pack type; it was never a brand, make, model, part type, options list,
condition, warranty, stock or price. The request's story and purpose point at it when they mention a part, a use or a
person. Order EVERY listed property, likeliest first (the assistant asks about them in groups of several); leave out only
a property the request already fixes or mentions, the price, stock, condition, brand, warranty, a measurement the request
gives, and a field that only lists a listing's variants. Reply with JSON: {"ask": ["<property exactly as listed>", ...]}"""

KEYS_GUIDE = """A store returned the result below from a product search. Name the key of each field of a product row, or
null when there is none: {"id": "<key holding the product id>", "price": "<key holding the price number>",
"stock": "<key holding whether it is in stock>", "category": "<key holding the category/department>",
"next_page": "<top-level key holding the next page number or token, or null>"}. Reply with that JSON only."""

LINE_GUIDE = """You read lines a shopper sent to an online store assistant during a purchase and say what changed.
Reply with one JSON object:
{
 "kind": "<disclosure | indifferent | confused | one_at_a_time | change | info | other>",
 "item": "<which of the items the lines concern, when the request names several and the lines say which; else ''>",
 "add": [{"property": "<property>", "value": "<value the item must have now>"}],
 "drop": ["<property of a requirement or preference the shopper no longer needs>"],
 "exclude": [{"property": "<property>", "value": "<value the item must not have>"}],
 "unexclude": ["<property of an exclusion the shopper lifted>"],
 "budget": <the new highest total price, exactly the number the shopper wrote in their own unit, if they changed it; else null>,
 "quantity": <new quantity if changed, else null>,
 "chosen_unavailable": <true when a line says the item the assistant picked, suggested or carted is gone, sold out, unavailable or can't be ordered>,
 "reference": <null, or when a line defines a property's value by another listing's page ("<property> ... the same as whatever you land on from the page of the <category> you'll find at <shop> with <property> <value>, <property> <value> and <property> <value>, following each listing it names", "<property> ... made to go with whatever you land on from the page of the <category> you'll find at <shop> with ...", "<property> ... never whatever you land on from the page of ...", "<property> ... inside 2 inches of whatever you land on from the page of ...", "the same as what's written on the page for the <category> that <shop> has listed, with <property> <value> and <property> <value>", "the same as the figure from the page of <category> from <shop> with ...", "the counterpart of the one pointed to (or stated) on the page of <shop>'s <category> with <property> <value> and <property> <value>", "<property> falls away from the counterpart of ..." (that is: must equal it), "nowt but what's given on the listing for the <category> sold by <shop>, with <property> <value>, ..." (that is: must equal it), "<property> ... matching whatever's on the page for <shop>'s listing of the <category> - the one with <property> <value> and <property> <value> - or wherever that page ends up sending you", "<property> ... a type that mates up with whatever's on the page for <shop>'s listing of the <category> - the one with ...", or the negated forms "anything bar what's written on the page ..." / "matching not the one pointed to ... on the page of ..." / "anything other than what's given on the listing for the <category> sold by <shop>, with ..." / "definitely not whatever's on the page for <shop>'s listing of the <category> - the one with ... - ever"): {"property": "<the property whose value is defined so>", "negated": <true for 'anything bar' / 'anything besides' / 'not the one' / 'anything other than' / 'definitely not' / 'never whatever'>, "category": "<the category named, as written>", "shop": "<the shop or seller named (the name before 's, or after 'from' / 'you'll find at' / 'that ... has listed')>", "attributes": {"<property>": "<value>", ...}, "relation": "<'same' | 'differ' | 'within' (a figure inside a tolerance of that page's: 'inside 133.27 watts of') | 'pairs' ('made to go with', 'a type that mates up with', 'compatible with': the rule on that page decides)>", "tolerance": "<for 'within': the figure and unit as written, e.g. '133.27 watts'; else ''>"}>
}
A value defined by another listing's page goes in "reference" only, never in add or exclude.
An answer of no preference to the assistant's question is 'indifferent' and drops nothing.
'disclosure': the shopper states what they need for a property ('the finish has to be brushed steel' -> add finish:
brushed steel; 'blade length of no more than 20 cm' -> value 'at most 20 cm'; 'material of the area rugs stays locked to
jute', 'pencil it in permanently - color equals white', 'I'll only look at things where display size comes out as at least
2 inch', 'protein source has got to match whey', 'style stands firm at traditional', 'I'm holding firm on closure sitting at
slip-on', 'I'll take nothing short of beige when it comes to color', 'width landing on at most 101 cm', 'I'm sticking
with fruit type of the canned fruit coming in as pineapple', 'me heart's set on packaging type being pouch', 'count me firm
on material matching iron', 'I want resolution landing on wqxga', 'with size class of the jumpsuits, I'll only look at
regular', 'no wriggle room from me on pants fit type, so I'm holding out for regular', 'pin style down to casual for me',
'keep depth locked to at least 35.56 cm', 'I need pattern sorted as plain', 'for me, display size means at least 15.6
inches', 'make sure height comes out as at most 123 cm' are all disclosures, and so is 'anything on pattern other than
floral gets a polite no thanks from me' (pattern must be floral: an add, not an exclusion); '<property> of the <items>' names
which item). When the assistant's question named several properties, the shopper answers about ONE of them: the property
the lines name is the one disclosed; the others stay unknown. 'indifferent': no preference ('any option will do', 'surprise
me', 'your call', 'not bothered', 'not fussed', 'don't mind a jot'). 'confused': they did not follow the question ('too much
jargon', 'say it again in plain talk', 'that's fog to me', 'all Greek to me'). 'one_at_a_time': they ask for the questions
one by one, the top one first, single file, not stacked. A line letting a requirement or preference go (it no longer matters, it was only a wish, ignore
it) is a drop of it. A new value for a property replaces the old one (put it in add). Keep values in the shopper's
words. Never convert a price into another unit."""

DEFER_GUIDE = """A store listing's page text may say that one of its properties is not its own but is to be taken from another
listing ("For pack type, go by the Anshidor IP Security Cameras listing with 12v input voltage and us plug type, not by
this item"; "For input voltage, refer to the Shisano IP security camera (turret style, us plug type)"; "Its material isn't
taken from this jacket; see the Katador Binders & Binder Accessories (lever arch binder, plain pattern) for that"; "For
storage, go by Noanlen's used mini PC with 16gb system memory"; "For color, look to the Sneakers item from Notaan, with
occasion casual and graphic print pattern, rather than to this rug"; "Take its color from the Wardrobe Organiser by Belratasa
made of metal with open storage"; "Brand compatibility should be taken from the Torqikaan power adaptor and charger (output
voltage: 5), not from this item"; "For the color, use the Wardrobe Organisers from Anlenkozen (waterproof storage feature,
contemporary style)"). List every such deferral, and also every value the text states for THIS listing itself ("Shibelzen's
area rug is beige and measures 5' x 8'" -> color: beige, size: 5' x 8'): {"deferrals": [{"property": "<the property
deferred>", "shop": "<the shop or seller named: a proper name such as Anshidor, Shisano, Notaan, Belratasa>", "category":
"<the kind of listing to look at, e.g. 'IP security cameras', 'sneakers', 'wardrobe organisers'>", "attributes": ["<each
attribute value given to tell that listing apart, e.g. '12v input voltage', 'occasion casual', 'material metal'>", ...]}],
"stated": {"<property>": "<value the text gives for this listing itself>", ...}}
Reply with that JSON only; {"deferrals": [], "stated": {}} when the text defers and states nothing."""

ROWS_GUIDE = """A store's search tool returned the product rows below as plain text, one row per line, numbered. Write each row as a
record: {"rows": [{"n": <row number>, "id": "<the product id or code exactly as written>", "title": "<the product's name>",
"price": <the price as a number, or null>, "in_stock": <true, false or null>, "category": "<its category, or null>"}, ...]}.
Copy ids exactly. Reply with that JSON only."""

PICK_GUIDE = """A shopper described one store listing by its seller and some of its details ("the desktop computer you'll find at
Volenko with storage 8000 gb"). Below are the listings of that seller that were read, with their facts. Say which one the
description means: every detail given must hold for it, read with a shop assistant's sense - a figure may be written another
way (16 x 500 gb is 8000 gb, 1 tb is 1000 gb), a value may sit in another field or in the title, a word may be a synonym.
Reply with JSON: {"id": "<the listing id>", "sure": <true when the details single it out, false when it is only the closest>}
or {"id": null, "sure": false} when none fits."""

RULE_GUIDE = """A shopper wants one property of the item they are buying to go with (be compatible with, be paired with, mate up
with) the value another listing's page gives for that property. That page's facts and text are below. When the page states a
pairing or compatibility rule for the property ("our memory type pairings are ddr3 with ddr5, ddr4 with ddr4", "a 12 inch
woofer goes with 300 watts"), the wanted value is the partner the rule gives for this page's own value; when the page states no
rule, the wanted value is the page's own value. Reply with JSON: {"value": "<the wanted value, in the page's words>", "rule": "<the
rule's words, or ''>"}; {"value": null} when the page gives neither."""

JUDGE_GUIDE = """You check store listings against a shopper's requirements the way a strict grader does: from each listing's
facts and title, literally. For each listing and each requirement answer:
- "y": a fact states the required value, or an obvious form of the same word ('hand tufted' for hand-tufted, 'flat-woven'
  for flatweave, 'solid oak' for oak, '80% wool' for wool, 'grass-fed whey' for whey; for a limit such as 'at most 29 g', a
  figure inside it in the same unit). The fact may sit under another label naming the same property ('weave' or 'weave type'
  for construction, 'furniture type' for the style of furniture, 'display size inch' for display size); facts often repeat a
  property under several labels, and one stating the value is enough. When no fact speaks of the property at all, the title
  stating the value is "y".
- "n": a fact of the property states a different value ('pile height: 0.6 inches' is not 'high', 'pile height: medium' is
  not '0.3 inches', 'construction: handcrafted' is not 'flatweave', 'style: modern plush' is not 'shag', 'neckline: crew
  neck' is not 'hooded'), or the title names a different value when no fact speaks. Another word for the very same thing is
  "y" ('full sleeve' for long sleeves, 'elasticated waist' for elastic waist), but a level is never a measure, a part of a
  range is not the range, and a related value is not the value ('mid heel' is not 'high heel').
- "?": nothing in the facts or the title speaks of the property. Descriptions, seller texts and the kind of item alone never
  settle a property; a value such items usually have is not a stated value. The one exception is which devices an accessory
  fits: a description saying which models it fits decides that ('Fits: every ROT31xx model, except ROT3130' fits ROT3110 but
  not ROT3130; 'Fits: MOZ-2150, Drane 746' fits a device sold as Drane 746).
An exclusion is "y" when the listing does NOT have that value and "n" when a fact or the title names it. The kind check: when
a store category is given, "y" for a listing filed under it, whatever its title calls the item; without one, "y" when the
listing is the kind of item wanted.
Reply with JSON: {"listings": [{"id": "<id>", "kind": "y|n|?", "checks": ["y|n|?", ...one per requirement, in order]}]}"""


class Polaris:
    def __init__(self, problem: dict[str, Any]) -> None:
        self.started = now()
        environment = problem["environment"]
        self.binding = environment["binding"]
        policy = environment["policy_view"]
        self.request = str(policy.get("query") or "")
        self.opening = policy.get("opening")
        self.tools = policy.get("tools") or []
        self.max_steps = int(policy.get("max_steps") or 30)
        self.finished = bool(policy.get("finished"))
        self.gate = Gate(str(self.binding.get("session_id") or self.request))
        self.models = Models(self.started + TIME_LIMIT, self.gate)
        self.step = 0
        self.dialogue: list[dict[str, Any]] = []
        self.roles: dict[str, Any] = {}
        self.items: dict[str, dict[str, Any]] = {}       # id -> merged fields seen for the listing
        self.sheets: dict[str, dict[str, Any]] = {}      # id -> facts read from its datasheet
        self.verdicts: dict[str, dict[str, Any]] = {}    # id -> {"stamp", "ok", "checks", "kind"}
        self.screen: dict[str, str] = {}                 # id -> likely / possible / unlikely (per stamp)
        self.screen_stamp = ""
        self.searched: set[str] = set()
        self.page_state: dict[str, dict[str, Any]] = {}  # search tag -> pages read, next page token, total
        self.rows_of: dict[str, list[str]] = {}          # search tag -> the listings its pages returned
        self.valued: set[str] = set()                    # listings a search on a required value returned
        self.lines_seen: list[str] = []
        self.asked: list[str] = []
        self.messages_sent = 0
        self.single_questions = False                    # True once the shopper asked for one property at a time
        self.rows_calls = 0                              # model readings of text rows no rule could parse
        self.last_reply_kind = ""                        # what the shopper's last lines were (disclosure, confused, ...)
        self.gone: set[str] = set()
        self.cart: dict[str, int] = {}
        self.plan: dict[str, Any] = {}
        self.requirements: list[dict[str, str]] = []
        self.exclusions: list[dict[str, str]] = []
        self.budget: float | None = None
        self.currency = ""
        self.quantity = 1
        self.message_budget = 0
        self.categories: list[str] = []
        self.keys: dict[str, str] = {}                   # id / price / stock / category -> the key this world uses
        self.keys_named = False
        self.order_tried: list[str] = []
        self.refusals = 0
        self.wants: list[dict[str, Any]] = []            # one entry per item the request asks for (its plan, needs, choice)
        self.want_at = 0
        self.attempted = False                           # whether a finish has been attempted (the store's changes arrive there)
        self.label_values: dict[str, list[str]] = {}     # fact label -> sample values, for the questions
        self.compare_cap = 0                             # ids the side-by-side tool is known to take (0: as the schema says)
        self.finish_faults: set[str] = set()             # finishing calls the store faulted (never repeated as they were)
        self.checkout_faults = 0
        self.category_values: dict[str, str] = {}        # category label -> the value the search's category argument takes
        self.combined_search: bool | None = None         # whether keywords and a category go in one search call
        self.judge_calls = 0
        self.confirmed: set[str] = set()                 # baskets already put to the shopper for approval
        self.viewed: set[str] = set()                    # listings whose full page was opened
        self.detailed: set[str] = set()                  # listings read through the datasheet tool itself (not side by side)
        self.found_now: set[str] = set()                 # listings shown while the current want is gathered
        self.rechecked: set[str] = set()                 # carted listings read again after carting
        self.confirm_needed = False                      # whether the request asks for the basket to be put to the shopper
        self.say_if_none = False                         # whether the request asks to be told when nothing fits
        self.carted: set[str] = set()                    # every listing ever put in the cart by this agent
        self.dropped: set[str] = set()                   # carted listings since removed
        self.defer_cache: dict[str, dict[str, Any]] = {}  # listing -> the deferrals and stated values its page text carries
        self.deferred_done: set[str] = set()             # listings whose page pointers were followed for the current needs
        self.query_seen: dict[str, str] = {}             # a query's word signature -> the search tag first used for it
        roomy = self.max_steps >= 70                     # a long step budget: wider searches, more candidates
        self.pages = 3
        self.deep = 6 if roomy else DEEP_PAGES
        self.sheets_first = 6 if roomy else SHEETS_FIRST
        self.fits_most = 14 if roomy else 12
        self.enough = 20 if roomy else 14

    # ---- bookkeeping -----------------------------------------------------------------------------------------
    def note(self, text: str) -> None:
        self.dialogue.append({"role": "assistant", "content": f"note: {text}"[:2500]})

    def time_left(self) -> float:
        return self.started + TIME_LIMIT - now()

    def steps_left(self) -> int:
        return self.max_steps - self.step

    def spec(self, name: str) -> dict[str, Any] | None:
        for entry in self.tools:
            function = entry.get("function") if isinstance(entry, dict) and "function" in entry else entry
            if isinstance(function, dict) and function.get("name") == name:
                return function
        return None

    def params(self, name: str) -> dict[str, Any]:
        return (((self.spec(name) or {}).get("parameters") or {}).get("properties") or {})

    # ---- environment -----------------------------------------------------------------------------------------
    def complete(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """The arguments with every required one the schema names filled in (a world may require more than a role needs)."""
        schema = (self.spec(name) or {}).get("parameters") or {}
        props = schema.get("properties") or {}
        args = {k: v for k, v in args.items() if k in props or not props}
        for arg in schema.get("required") or []:
            if arg in args:
                continue
            rule = props.get(arg) or {}
            if rule.get("enum"):
                args[arg] = rule["enum"][0]
            elif rule.get("type") in ("integer", "number"):
                args[arg] = rule.get("minimum", 1)
            elif rule.get("type") == "boolean":
                args[arg] = True
            elif rule.get("type") == "string":
                args[arg] = "ok"[: max(1, int(rule.get("maxLength") or 2))] if not rule.get("minLength") or int(rule["minLength"]) <= 2 else "x" * int(rule["minLength"])
        return args

    def call(self, name: str, args: dict[str, Any]) -> Any:
        """One environment step; the observation (None when the episode is over or the call failed)."""
        if self.finished or self.step >= self.max_steps:
            return None
        args = self.complete(name, args)
        self.step += 1
        result = None
        for attempt in range(7):
            if self.time_left() < 2:
                break
            self.gate.wait()
            envelope = {**self.binding, "step": self.step, "action": {"name": name, "args": args}}
            try:
                outcome = PROXY.post_verbose("/environment/call", json_data=envelope)
            except Exception:
                outcome = None
            result = outcome.data if outcome is not None else None
            if isinstance(result, dict):
                self.gate.eased()
                break
            error = outcome.error if outcome is not None else None
            said = json.dumps(error, default=str)
            expected = re.search(r"expected (\d+)", said)
            if expected:
                self.step = int(expected.group(1))        # the session's own count of steps decides
            elif re.search(r"finished|quarantin|unknown session", said):
                self.finished = True
                return None
            elif (error or {}).get("status") == 429:
                self.gate.limited(attempt)                # rate limited: a pause that grows, and wider spacing after
            else:
                time.sleep(0.5 * (attempt + 1))
        if not isinstance(result, dict):
            self.note(f"environment call failed at step {self.step}: {name}")
            self.step -= 1                      # the session did not count a call that never got through
            return None
        self.finished = bool(result.get("finished"))
        if isinstance(result.get("step"), int) and not isinstance(result.get("step"), bool):
            self.step = int(result["step"])     # the session's own count is the truth
        self.dialogue.append({"role": "assistant", "content": "", "tool_call": {"name": name, "args": args}, "environment_result": result})
        observation = self.normalize(result.get("observation"))
        self.absorb(observation)
        return observation

    # ---- reading results -------------------------------------------------------------------------------------
    @staticmethod
    def text_value(key: str, value: str) -> Any:
        """A value written in a text row read as the store means it: None, a bool, a number, or 'a=b; c=d' facts as a dict."""
        plain = value.strip()
        if plain in ("", "None", "none", "null", "-"):
            return None
        if plain in ("True", "true", "False", "false"):
            return plain.lower() == "true"
        if re.fullmatch(r"-?\d+", plain):
            return int(plain)
        if re.fullmatch(r"-?\d+\.\d+", plain):
            return float(plain)
        parts = [part for part in re.split(r";\s*", plain) if part.strip()]
        pairs = [part.split("=", 1) for part in parts if "=" in part]
        if pairs and (len(pairs) >= 2 or re.search(r"fact|attribute|spec|propert|detail|feature", key)) and len(pairs) >= len(parts) - 1:
            return {k.strip().lower(): v.strip() for k, v in pairs if k.strip()}
        return plain

    @staticmethod
    def text_record(text: str) -> dict[str, Any] | None:
        """A product row or page written as one text ('ref_ref: p-1 | sold_by_name: X | title: ... | price: 1595 | on_hand:
        True | facts: a=b; c=d | item_blurb: ...') read as a record; None for any text that is not three such parts or more
        (a shopper's line, a title). Parts are split on ' | ' or line breaks; each gives its key before the first ': '."""
        if not isinstance(text, str) or len(text) < 12 or ": " not in text:
            return None
        parts = [part for part in re.split(r"\s\|\s|\n+|\t+", text.strip()) if part.strip()]
        record: dict[str, Any] = {}
        for part in parts:
            found = re.match(r"\s*([A-Za-z][\w .\-/]{0,40}?)\s*:\s(.*)$", part, re.S)
            if not found:
                continue
            key = found.group(1).strip().lower()
            if len(key.split()) > 4:
                continue
            record[key] = Polaris.text_value(key, found.group(2))
        return record if len(record) >= 3 else None

    def normalize(self, value: Any, under: str = "") -> Any:
        """A result with its text-written rows and pages turned into records, so every reader sees the same shapes; a
        list of row texts no rule reads is given to a model (never the shopper's lines, which sit under other keys)."""
        if isinstance(value, str):
            plain = value.strip()
            if plain[:1] in "{[" and plain[-1:] in "}]":
                try:
                    return self.normalize(json.loads(plain), under)     # JSON sent as text
                except ValueError:
                    pass
            lines = [line for line in re.split(r"\n+", plain) if line.strip()]
            if len(lines) >= 2:
                records = [self.text_record(line) for line in lines]
                if sum(1 for r in records if r) >= 2 and sum(1 for r in records if r) >= len(lines) // 2:
                    return [r for r in records if r]                    # one row per line
            record = self.text_record(value)
            return record if record else value
        if isinstance(value, list):
            out = [self.normalize(v, under) for v in value]
            rows = [v for v in out if isinstance(v, str) and len(v) > 25 and re.search(r"\b[a-z]{1,3}-[0-9a-z]{6,}\b|\b(?:id|sku|code|ref)\b\s*[:#=]?\s*\w", v, re.I)]
            if len(rows) >= 2 and not any(isinstance(v, dict) for v in out) and not re.search(r"refus|line|message|note|said|reply|hufa|kere|jezudo|event|remark", under, re.I) and self.rows_calls < 20:
                parsed = self.rows_from_text(rows)
                if parsed:
                    out = [parsed.get(v, v) if isinstance(v, str) else v for v in out]
            return out
        if isinstance(value, dict):
            return {k: self.normalize(v, str(k)) for k, v in value.items()}
        return value

    def rows_from_text(self, rows: list[str]) -> dict[str, dict[str, Any]]:
        """Text rows read into records by a model: {row text: record}; empty when the model gives nothing usable."""
        self.rows_calls += 1
        text = "\n".join(f"{i + 1}. {row[:400]}" for i, row in enumerate(rows[:30]))
        answer = self.models.ask(ROWS_GUIDE, text, strong=False, max_tokens=120 * min(30, len(rows)) + 100)
        out: dict[str, dict[str, Any]] = {}
        for rec in ((answer or {}).get("rows") or []) if isinstance(answer, dict) else []:
            if not isinstance(rec, dict):
                continue
            n = rec.get("n")
            ident = str(rec.get("id") or "").strip()
            if not isinstance(n, int) or not 1 <= n <= len(rows) or not ident or ident not in rows[n - 1]:
                continue
            record = {"id": ident, "title": str(rec.get("title") or "")[:120], "text": rows[n - 1][:600]}
            if isinstance(rec.get("price"), (int, float)) and not isinstance(rec.get("price"), bool):
                record["price"] = rec["price"]
            if isinstance(rec.get("in_stock"), bool):
                record["in_stock"] = rec["in_stock"]
            if isinstance(rec.get("category"), str) and rec["category"].strip():
                record["category"] = rec["category"].strip()
            out[rows[n - 1]] = record
        if out:
            self.note(f"{len(out)} text rows read by a model")
        return out

    def flat_facts(self, key: str) -> dict[str, Any]:
        """The plain fields of a listing's record that read as its properties, for a world whose page has no fact block."""
        roles = {self.keys.get(r) for r in ("id", "price", "stock", "category") if self.keys.get(r)}
        skip = re.compile(r"title|name|head|desc|blurb|comment|remark|note|\bid\b|_id|ref|url|image|price|cost|stock|hand|avail|seller|sold|merchant|brand|categ|shelf|section|sku|code", re.I)
        return {k: v for k, v in self.items.get(key, {}).items() if k not in roles and not skip.search(str(k)) and isinstance(v, (str, int, float)) and not isinstance(v, bool) and str(v).strip()}

    @staticmethod
    def dicts_in(value: Any) -> list[dict[str, Any]]:
        found: list[dict[str, Any]] = []
        if isinstance(value, dict):
            found.append(value)
            for child in value.values():
                found += Polaris.dicts_in(child)
        elif isinstance(value, list):
            for child in value:
                found += Polaris.dicts_in(child)
        return found

    def name_keys(self, observation: Any) -> None:
        """A model names the keys of a search result's rows once; the heuristic reading fills what it leaves out."""
        if self.keys_named:
            return
        self.keys_named = True
        names: set[str] = set()
        for record in self.dicts_in(observation):
            names.update(str(k) for k in record)
        answer = self.models.ask(KEYS_GUIDE, json.dumps(observation, ensure_ascii=False)[:3000], strong=False, max_tokens=400)
        for role in ("id", "price", "stock", "category", "next_page"):
            value = (answer or {}).get(role) if isinstance(answer, dict) else None
            if isinstance(value, str) and value in names and role not in self.keys:
                self.keys[role] = value
        self.note(f"keys: {self.keys}")

    def learn_keys(self, rows: list[dict[str, Any]]) -> None:
        """Which keys of a product record hold its id, price, stock flag and category (the names change per world)."""
        keys = set(rows[0])
        for row in rows[1:]:
            keys &= set(row)

        def pick(role: str, test, hint: str) -> None:
            if role in self.keys:
                return
            options = [k for k in keys if all(test(r.get(k)) for r in rows)]
            options.sort(key=lambda k: (0 if re.search(hint, k.lower()) else 1, len(k)))
            if role == "id":
                options = [k for k in options if len({r[k] for r in rows}) == len(rows)]
                if len(rows) < 2:
                    options = [k for k in options if re.search(hint, k.lower())]
            if role == "category":
                options = [k for k in options if re.search(hint, k.lower())]
            if options:
                self.keys[role] = options[0]
        pick("id", lambda v: isinstance(v, str) and 0 < len(v) <= 64 and " " not in v and bool(re.search(r"\d", v)), r"id|pid|sku|code|ref")
        if "price" not in self.keys:
            numeric = [k for k in keys if all(isinstance(r.get(k), (int, float)) and not isinstance(r.get(k), bool) for r in rows)]
            named = [k for k in numeric if re.search(r"price|cost|amount|fee|charge|total|value|pay", k.lower())]
            if not named and len(numeric) > 1 and self.budget:
                # no key says it holds the price: the one whose figures sit around the budget
                def spread(k: str) -> float:
                    figures = [float(r[k]) for r in rows]
                    return sum(abs(math.log10(max(f, 1e-6) / self.budget)) for f in figures) / len(figures)
                numeric.sort(key=spread)
                self.keys["price"] = numeric[0]
        pick("price", lambda v: isinstance(v, (int, float)) and not isinstance(v, bool), r"price|cost|amount|fee|charge|total|value|pay")
        pick("stock", lambda v: isinstance(v, bool), r"stock|avail|hand|left|sold")
        pick("category", lambda v: isinstance(v, str) or v is None, r"categ|depart|aisle|section|shelf|group|class")

    def product_rows(self, observation: Any) -> list[dict[str, Any]]:
        """Records in a result that describe one product each."""
        lists: list[list[dict[str, Any]]] = []

        def walk(value: Any) -> None:
            if isinstance(value, list):
                group = [v for v in value if isinstance(v, dict)]
                if group:
                    lists.append(group)
                for v in value:
                    walk(v)
            elif isinstance(value, dict):
                for v in value.values():
                    walk(v)
        walk(observation)
        if lists and not self.keys_named:
            self.name_keys(observation)
        for group in lists:
            if len(group) >= 2:
                self.learn_keys(group)
        key = self.keys.get("id")
        return [record for record in self.dicts_in(observation) if key and isinstance(record.get(key), str)]

    def absorb(self, observation: Any) -> None:
        """Keep every product record a result shows; a nested record of plain values is its facts."""
        for row in self.product_rows(observation):
            key = row[self.keys["id"]]
            self.found_now.add(key)
            merged = self.items.setdefault(key, {})
            for name, value in row.items():
                if isinstance(value, dict) and value and all(not isinstance(v, (dict, list)) for v in value.values()):
                    self.sheets.setdefault(key, {}).update({str(k): v for k, v in value.items()})
                elif not isinstance(value, (dict, list)):
                    merged[name] = value

    def value(self, key: str, role: str) -> Any:
        name = self.keys.get(role)
        return self.items.get(key, {}).get(name) if name else None

    def price(self, key: str) -> float | None:
        value = self.value(key, "price")
        return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None

    def in_stock(self, key: str) -> bool | None:
        value = self.value(key, "stock")
        return value if isinstance(value, bool) else None

    def category_of(self, key: str) -> str:
        return str(self.value(key, "category") or "")

    @staticmethod
    def sentences(observation: Any, skip: set[int]) -> list[str]:
        """Strings in a result that read as something a person said (not product data)."""
        found: list[str] = []

        def walk(value: Any) -> None:
            if id(value) in skip:
                return
            if isinstance(value, str):
                text = value.strip()
                if len(text.split()) >= 4 and re.search(r"[a-zA-Z]", text):
                    found.append(text)
            elif isinstance(value, dict):
                for v in value.values():
                    walk(v)
            elif isinstance(value, list):
                for v in value:
                    walk(v)
        walk(observation)
        return found

    def shopper_lines(self, observation: Any) -> list[str]:
        if isinstance(observation, str) and observation.strip():
            lines = [observation.strip()]           # a bare string result is what the shopper said
        else:
            skip = {id(row) for row in self.product_rows(observation)}
            lines = self.sentences(observation, skip)
        lines = [line for line in lines if line not in self.lines_seen]
        self.lines_seen += lines
        return lines

    # ---- planning ----------------------------------------------------------------------------------------------
    def map_tools(self) -> None:
        listing = json.dumps([{"name": (t.get("function") or t).get("name"), "description": (t.get("function") or t).get("description"),
                               "parameters": (t.get("function") or t).get("parameters")} for t in self.tools], ensure_ascii=False)
        for strong in (False, True):
            answer = self.models.ask(TOOLS_GUIDE, "Tools:\n" + listing, strong=strong, max_tokens=1600)
            self.roles = self.checked_roles(answer if isinstance(answer, dict) else {})
            if all(role in self.roles for role in ("search", "cart_add", "checkout")) and (self.roles["search"].get("text_arg") or self.roles["search"].get("category_arg")):
                break
        self.fill_roles()
        self.fix_cart_roles()
        self.note("tool roles: " + json.dumps(self.roles)[:1500])

    def checked_roles(self, roles: dict[str, Any]) -> dict[str, Any]:
        checked: dict[str, Any] = {}
        for role, spec in roles.items():
            if not isinstance(spec, dict) or not self.spec(str(spec.get("tool"))):
                continue
            names = set(self.params(str(spec["tool"])))
            clean: dict[str, Any] = {"tool": spec["tool"]}
            for field, value in spec.items():
                if field.endswith("_arg"):
                    clean[field] = value if isinstance(value, str) and value in names else None
                elif field in ("text_args_fixed", "fields_args_fixed", "fixed"):
                    clean[field] = {k: v for k, v in value.items() if k in names and self.allowed(spec["tool"], k, v)} if isinstance(value, dict) else {}
            if role in ("listing", "datasheet", "cart_add", "cart_remove") and not clean.get("id_arg"):
                continue
            if role == "compare" and (not clean.get("ids_arg") or (self.params(spec["tool"]).get(clean["ids_arg"]) or {}).get("type") != "array"):
                continue
            if role == "browse" and not clean.get("category_arg"):
                continue
            if role == "message" and not clean.get("text_arg"):
                continue
            checked[role] = clean
        checkout, decline = checked.get("checkout"), checked.get("decline")
        if checkout and decline and checkout["tool"] == decline["tool"] and (checkout.get("fixed") or {}) == (decline.get("fixed") or {}):
            # one finishing tool for both: the decline takes the other value of the enum that selects the outcome
            props = self.params(decline["tool"])
            for arg, rule in props.items():
                values = [v for v in (rule.get("enum") or []) if isinstance(v, str)]
                if len(values) < 2:
                    continue
                buying = (checkout.get("fixed") or {}).get(arg) or values[0]
                others = [v for v in values if v != buying]
                liked = [v for v in others if re.search(r"declin|refus|abandon|abstain|cancel|quit|give|no|without|stop|end|leave|drop|walk|skip|exit|close|bail|forgo|pass", v.lower())]
                decline["fixed"] = {**(decline.get("fixed") or {}), arg: (liked or others)[0]}
                checkout["fixed"] = {**(checkout.get("fixed") or {}), arg: buying}
                break
        return checked

    def allowed(self, tool: str, arg: str, value: Any) -> bool:
        schema = self.params(tool).get(arg) or {}
        return value in schema["enum"] if "enum" in schema else True

    def fix_cart_roles(self) -> None:
        """A cart tool's count argument is never pinned to a fixed value (a model may read 'lineNumber' as an index and
        fix it at 0, which empties the cart); when the model named no count argument, the tool's integer argument is it."""
        for role in ("cart_add", "cart_remove"):
            spec = self.roles.get(role)
            if not isinstance(spec, dict) or not spec.get("tool"):
                continue
            props = self.params(spec["tool"])
            ints = [n for n, r in props.items() if (r or {}).get("type") in ("integer", "number") and n != spec.get("id_arg")]
            fixed = spec.get("fixed") or {}
            spec["fixed"] = {k: v for k, v in fixed.items() if k not in ints}
            if not spec.get("quantity_arg") and ints:
                named = [n for n in ints if re.search(r"count|qty|quant|unit|amount|how|number|pieces|line|want|need", n.lower())]
                if len(named) == 1 or len(ints) == 1:
                    spec["quantity_arg"] = (named or ints)[0]

    def fill_roles(self) -> None:
        """Fill roles the model left out from the tools' names and descriptions."""
        specs = [(t.get("function") or t) for t in self.tools if isinstance(t, dict)]
        patterns = {"search": r"search|find|look ?up|browse|query", "checkout": r"check ?out|purchase|place .*order|buy|pay",
                    "decline": r"decline|no purchase|without (?:buying|purchase)|abandon|refuse",
                    "message": r"shopper|customer|question|message|ask|chat|contact", "cart_view": r"view .*(?:cart|basket)|(?:cart|basket) contents"}
        search = self.roles.get("search")
        if search and not search.get("text_arg"):
            for spec in specs:
                props = ((spec.get("parameters") or {}).get("properties") or {})
                text_arg = next((n for n, r in props.items() if (r or {}).get("type") == "string" and not (r or {}).get("enum")
                                 and re.search(r"keyword|query|words|text|wanted|needle|search|term|phrase|look", n.lower())), None)
                if spec.get("name") != search["tool"] and text_arg:
                    if search.get("category_arg") and "browse" not in self.roles:
                        self.roles["browse"] = {"tool": search["tool"], "category_arg": search["category_arg"], "page_arg": search.get("page_arg"), "fixed": dict(search.get("fields_args_fixed") or {})}
                    self.roles["search"] = {"tool": spec["name"], "text_arg": text_arg, "text_args_fixed": {}, "fields_args_fixed": {},
                                            "category_arg": None, "page_arg": next((n for n, r in props.items() if re.search(r"page|offset|start|cursor|token|from", n.lower())), None)}
                    break
        for role in ("browse", "search"):
            spec = self.roles.get(role)
            if isinstance(spec, dict) and spec.get("tool") and not spec.get("brand_arg"):
                props = self.params(spec["tool"])
                named = [n for n, r in props.items() if (r or {}).get("type", "string") == "string" and not (r or {}).get("enum")
                         and re.search(r"brand|maker|manufactur|seller|vendor|shop|merchant|producer|dealer|marque", n.lower()) and n not in spec.values()]
                if len(named) == 1:
                    spec["brand_arg"] = named[0]
        taken = {spec["tool"] for spec in self.roles.values() if isinstance(spec, dict)}
        for role, pattern in patterns.items():
            if role in self.roles:
                continue
            for spec in specs:
                text = f"{spec.get('name', '')} {spec.get('description', '')}".lower().replace("_", " ")
                if spec.get("name") in taken or not re.search(pattern, text):
                    continue
                props = ((spec.get("parameters") or {}).get("properties") or {})
                entry: dict[str, Any] = {"tool": spec["name"]}
                if role == "message":
                    strings = [n for n, s in props.items() if (s or {}).get("type") == "string"]
                    if not strings:
                        continue
                    entry["text_arg"] = strings[0]
                if role == "decline":
                    entry["remark_arg"] = next((n for n, s in props.items() if (s or {}).get("type") == "string"), None)
                    entry["fixed"] = {}
                if role == "search":
                    entry.update({"text_arg": next((n for n, s in props.items() if (s or {}).get("type") == "string"), None),
                                  "text_args_fixed": {}, "fields_args_fixed": {}})
                self.roles[role] = entry
                taken.add(spec["name"])
                break

    def read_request(self) -> None:
        plan = self.models.ask(REQUEST_GUIDE, f"Request:\n{self.request}", strong=True, max_tokens=1800)
        plan = plan if isinstance(plan, dict) else {}
        if not plan:
            plan = self.models.ask(REQUEST_GUIDE, f"Request:\n{self.request}", strong=False, max_tokens=1800)
            plan = plan if isinstance(plan, dict) else {}
        items = [i for i in (plan.get("items") or []) if isinstance(i, dict) and str(i.get("item") or "").strip()]
        if not items:
            items = [plan]                              # the older shape: one item at the top level
        shared = {k: v for k, v in plan.items() if k not in ("items", "item", "category", "requirements", "exclusions", "quantity", "search_queries")}
        self.wants = []
        for entry in items[:4]:
            want_plan = {**shared, **{k: entry.get(k) for k in ("item", "category", "search_queries") if k in entry}}
            self.wants.append({
                "plan": want_plan,
                "requirements": [{"property": str(r.get("property") or "").strip(), "value": str(r.get("value") or "").strip()}
                                 for r in entry.get("requirements") or [] if isinstance(r, dict) and str(r.get("value") or "").strip() and self.demanded(r)],
                "exclusions": [{"property": str(r.get("property") or "").strip(), "value": str(r.get("value") or "").strip()}
                               for r in entry.get("exclusions") or [] if isinstance(r, dict) and str(r.get("value") or "").strip() and self.demanded(r)],
                "quantity": max(1, int(self.number(entry.get("quantity")) or 1)), "categories": [], "valued": set(), "asked": [], "tried": [], "choice": None,
                "found": set(), "references": [r for r in entry.get("references") or [] if isinstance(r, dict) and str(r.get("property") or "").strip() and (r.get("shop") or r.get("category"))][:3],
                "linked": [l for l in entry.get("linked") or [] if isinstance(l, dict) and str(l.get("property") or "").strip() and str(l.get("to_item") or "").strip()][:2]})
        self.load_want(0)
        self.confirm_needed = bool(plan.get("confirm_first")) or bool(CONFIRM_WORDS.search(self.request))
        self.say_if_none = bool(plan.get("say_if_none"))
        self.currency = str(plan.get("currency") or "").strip()
        self.budget = self.number(plan.get("budget"))
        written = self.amounts_in(self.request)
        if written and (self.budget is None or self.budget not in written):
            self.budget = max(written) if self.budget is None else min(written, key=lambda w: abs(w - self.budget))
        budget = self.number(plan.get("message_budget"))
        said = re.search(r"(\d+)\s+(?:messages|questions|notes)", self.request, re.I)
        if said:
            budget = float(said.group(1))
        self.message_budget = int(budget) if budget is not None else (0 if not self.roles.get("message") else 3)
        steps = self.number(plan.get("step_budget"))
        said = re.search(r"(\d+)\s+(?:moves|steps|actions|tool calls)", self.request, re.I)
        if said:
            steps = float(said.group(1))
        if steps:
            self.max_steps = min(self.max_steps, int(steps))
        for want in self.wants:
            self.note(f"request: item={want['plan'].get('item')!r} category={want['plan'].get('category')!r} needs={want['requirements']} not={want['exclusions']} qty={want['quantity']}")
        self.note(f"budget={self.budget} {self.currency} messages={self.message_budget} steps={self.max_steps} priority={plan.get('priority')!r} items={len(self.wants)}")

    def load_want(self, at: int) -> None:
        """Make one of the wanted items the current one: its plan, needs, quantity and category become the working ones."""
        want = self.wants[at]
        self.want_at = at
        self.plan, self.requirements, self.exclusions = want["plan"], want["requirements"], want["exclusions"]
        self.quantity, self.categories, self.valued, self.asked, self.order_tried = want["quantity"], want["categories"], want["valued"], want["asked"], want["tried"]
        self.found_now = want.setdefault("found", set())

    def save_want(self) -> None:
        want = self.wants[self.want_at]
        want.update(plan=self.plan, requirements=self.requirements, exclusions=self.exclusions, quantity=self.quantity,
                    categories=self.categories, valued=self.valued, asked=self.asked, tried=self.order_tried, found=self.found_now)

    def want_for(self, text: Any) -> int | None:
        """Which want a shopper's words name (by the words of its item or category), when there are several."""
        if len(self.wants) < 2 or not isinstance(text, str) or not words(text):
            return None
        said = set(words(text))
        scored = []
        for at, want in enumerate(self.wants):
            own = set(words(str(want["plan"].get("item") or ""))) | set(w for c in want["categories"] for w in words(c))
            scored.append((len(said & own), at))
        best = max(scored)
        return best[1] if best[0] > 0 and [sc for sc, _ in scored].count(best[0]) == 1 else None

    def demanded(self, requirement: dict[str, Any]) -> bool:
        """True when the request's own words name the value (a requirement the reader made up from the story is dropped)."""
        said = set(words(self.request))
        value = [w for w in words(requirement.get("value")) if len(w) > 2]
        return not value or all(w in said or w.rstrip("s") in said or w + "s" in said for w in value)

    def drop_kind_words(self) -> None:
        """Requirements whose value only repeats the category's name are not separate requirements: the category already
        decides the kind ('reading or desk lamp' under 'Reading and Desk Lamps')."""
        named = set().union(*[self.label_words(c) for c in self.categories]) if self.categories else set()
        kept = []
        for requirement in self.requirements:
            value = self.label_words(requirement["value"])
            if value and value <= named:
                self.note(f"dropped {requirement} (it names the category or item)")
                continue
            kept.append(requirement)
        self.requirements = kept

    def amounts_in(self, text: str) -> list[float]:
        """Prices a text writes with the request's own currency word or sign, after or before the figure."""
        if not self.currency:
            return []
        unit = re.escape(self.currency)
        figure = r"(\d[\d,]*(?:\.\d+)?)"
        found = [m.group(1) for m in re.finditer(figure + r"\s*" + unit + r"(?![a-z])", text, re.I)]
        found += [m.group(1) for m in re.finditer(r"(?<![a-z])" + unit + r"\s*" + figure, text, re.I)]
        return [float(f.replace(",", "")) for f in found]

    @staticmethod
    def number(value: Any) -> float | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return float(value)
        found = re.search(r"\d+(?:\.\d+)?", str(value or "").replace(",", ""))
        return float(found.group(0)) if found else None

    # ---- searching ---------------------------------------------------------------------------------------------
    def row_keys(self, observation: Any) -> list[str]:
        key = self.keys.get("id")
        return [str(r[key]) for r in self.product_rows(observation) if r.get(key) is not None] if key else []

    def value_hits(self, observation: Any) -> None:
        """Mark the listings a search on a required value returned: their indexed text holds the value somewhere."""
        for row in self.product_rows(observation):
            self.valued.add(row[self.keys["id"]])

    def category_value(self, label: str) -> str | None:
        """What the search's category argument takes for a category label: the label itself, or the allowed value that
        names it (an enum may write categories in its own case or form); None when the argument allows nothing like it."""
        spec = self.roles.get("search") or {}
        if not spec.get("category_arg"):
            spec = self.roles.get("browse") or {}
        rule = self.params(str(spec.get("tool") or "")).get(str(spec.get("category_arg") or "")) or {}
        allowed = [v for v in (rule.get("enum") or []) if isinstance(v, str)]
        if not spec.get("category_arg"):
            return None
        if not allowed:
            return label
        if label in self.category_values:
            return self.category_values[label] or None
        exact = [v for v in allowed if v.lower() == label.lower()]
        close = exact or [v for v in allowed if self.label_words(v) == self.label_words(label)]
        self.category_values[label] = close[0] if close else ""
        return close[0] if close else None

    def search(self, text: str | None = None, category: str | None = None, cap: float | None = None, page: Any = None) -> Any:
        spec = self.roles.get("search") or {}
        value = self.category_value(category) if category else None
        if text is not None:
            if not spec.get("text_arg"):
                return None
            args = dict(spec.get("text_args_fixed") or {})
            schema = self.params(spec["tool"]).get(spec["text_arg"]) or {}
            args[spec["text_arg"]] = text[: int(schema.get("maxLength") or 60)]
            if value and spec.get("category_arg") and self.combined_search is not False:
                args[spec["category_arg"]] = value       # keywords within the category, where the tool takes both
        else:
            if not value:
                return None
            if not spec.get("category_arg") and self.roles.get("browse"):
                spec = self.roles["browse"]              # a tool of its own lists a category
                args = dict(spec.get("fixed") or {})
            else:
                args = dict(spec.get("fields_args_fixed") or {})
            args[spec["category_arg"]] = value
            if cap is not None and spec.get("price_arg"):
                schema = self.params(spec["tool"]).get(spec["price_arg"]) or {}
                args[spec["price_arg"]] = int(cap) if schema.get("type") == "integer" else round(cap, 2)
        if page is not None and spec.get("page_arg"):
            kind = (self.params(spec["tool"]).get(spec["page_arg"]) or {}).get("type")
            if kind == "string":
                page = str(page)
            elif kind in ("integer", "number") and isinstance(page, str) and page.isdigit():
                page = int(page)
            args[spec["page_arg"]] = page
        observation = self.call(spec["tool"], args)
        if text is not None and value and spec.get("category_arg") and self.combined_search is None:
            # the first search that sent keywords and a category together tells whether the store takes both at once
            if self.product_rows(observation) or not self.refused(observation):
                self.combined_search = True
            else:
                self.combined_search = False
                self.note("keywords and a category do not go together here: keywords alone from now on")
                args.pop(spec["category_arg"], None)
                observation = self.call(spec["tool"], args)
        if text is not None and any(set(words(r["value"])) and set(words(r["value"])) <= set(words(text)) for r in self.requirements):
            self.value_hits(observation)
        self.events(observation)
        return observation

    def counts(self, observation: Any) -> tuple[int | None, Any]:
        """(how many matched, the next page token) from a search result's top-level values: the token under the key a
        model named for it or one whose name says so, else the one plain value left beside the count."""
        if isinstance(observation, dict) and len(observation) <= 3 and any(isinstance(v, dict) for v in observation.values()):
            inner = [v for v in observation.values() if isinstance(v, dict)]
            if inner and not any(isinstance(v, list) and v and isinstance(v[0], dict) for v in observation.values()):
                observation = inner[0]            # a result wrapped together with shopper lines
        if not isinstance(observation, dict):
            return None, None
        total, more, named, loose = None, None, False, []
        rows = any(isinstance(v, list) and v and isinstance(v[0], dict) for v in observation.values())
        for key, value in observation.items():
            if isinstance(value, bool) or isinstance(value, (list, dict)):
                continue
            name = key.lower()
            if key == self.keys.get("next_page"):
                more, named = value, True
            elif isinstance(value, int) and re.search(r"match|total|count|number|found|hits|results", name) and total is None:
                total = value
            elif re.search(r"next|continue|more|cursor|page|token", name):
                more, named = value, True
            else:
                loose.append(value)
        if not named and rows and len(loose) == 1 and isinstance(loose[0], (str, int)):
            more = loose[0]
        return total, more

    @staticmethod
    def query_terms(text: str) -> list[str]:
        """Words of a value as search terms: figures kept whole ('0.39', '29.3'), the words of a limit or comparison
        ('at least', 'no more than', 'up to') left out, since a store indexes what listings say, not how a need is put."""
        text = re.sub(r"\b(?:at|no)\s+(?:least|most|more|less|fewer|greater|smaller)(?:\s+than)?\b|\b(?:up|close)\s+to\b|\b(?:under|over|below|above|minimum|maximum|min|max|around|about|exactly|than|or)\b",
                      " ", text.lower())
        return [t for t in re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text) if t not in ("a", "an", "the", "of", "in", "with", "for", "and", "it", "its")]

    def filtered(self) -> str | None:
        """The settled category when the search can filter by it (then its words need not go in the keywords)."""
        return self.categories[0] if self.categories and self.category_value(self.categories[0]) else None

    def value_query(self, value: str, extra: str = "") -> str:
        """Keywords for a required value: its own words first, then the item's, so a store that reads only the first
        words of a query still narrows by the value; the category's words follow when no category filter joins the
        search. A value naming alternatives ('double breasted or zipper') is searched by its first one."""
        first = re.split(r"\bor\b|/|,", value)[0]
        item = [w for w in self.query_terms(str(self.plan.get("item") or "")) if len(w) > 1][:2]
        base = self.query_terms(first) + self.query_terms(extra) + item
        have = self.stems(" ".join(base))
        category = [] if self.filtered() else [w for w in words(" ".join(self.categories[:1])) if len(w) > 1 and not self.stems(w) <= have]
        terms = list(dict.fromkeys(base + category))
        return " ".join(terms[:5])

    def alternatives(self, value: str) -> list[str]:
        return [part.strip() for part in re.split(r"\bor\b|/|,", value) if words(part)]

    def split_words(self, query: str) -> list[str]:
        """Title words common among the listings a value search returned but not in every one: each narrows that search
        to a part small enough to see in full."""
        titles = [set(words(" ".join(str(v) for k, v in self.items.get(key, {}).items() if isinstance(v, str) and re.search(r"title|name|head", k.lower()))))
                  for key in self.valued]
        titles = [t for t in titles if t]
        if len(titles) < 6:
            return []
        used = set(words(query))
        tally: dict[str, int] = {}
        for title in titles:
            for w in title - used:
                if len(w) > 2 and not w.isdigit():
                    tally[w] = tally.get(w, 0) + 1
        picks = [w for w, n in sorted(tally.items(), key=lambda kv: -kv[1]) if 0.12 <= n / len(titles) <= 0.6]
        return picks[:4]

    def query_signature(self, text: str) -> str:
        """The words of a query as a set, plurals folded: 'sneakers' and 'sneaker', 'denim jacket retro' and 'retro denim
        jacket' are one search (figures kept whole)."""
        return " ".join(sorted({w[:-1] if len(w) > 3 and w.endswith("s") and not re.search(r"\d", w) else w for w in self.query_terms(text)}))

    def search_all(self, text: str | None = None, category: str | None = None, cap: float | None = None, pages: int | None = None) -> int:
        """A search and its following pages: all of them when every match fits in the pages shown, else `pages` (default
        one); a ranked search with hundreds of matches is read two pages deep at most, its best rows come first. A search
        run before (the same words in any order or form) continues from where it stopped. Returns how many matched (-1
        when nothing was run)."""
        if text is not None and category and self.combined_search is False:
            category = None
        if text is not None:
            signature = json.dumps([self.query_signature(text), category, cap]).lower()
            if signature in self.query_seen:
                tag = self.query_seen[signature]
            else:
                tag = json.dumps([text, category, cap]).lower()
                self.query_seen[signature] = tag
        else:
            tag = json.dumps([text, category, cap]).lower()
        if text is not None and category and tag not in self.page_state:
            plain = json.dumps([text, None, cap]).lower()
            rows = self.rows_of.get(plain) or []
            if rows and sum(1 for k in rows if self.category_of(k) in self.categories) >= 0.8 * len(rows):
                tag, category = plain, None         # the same words, searched without the filter, found this category: its pages continue
        state = self.page_state.get(tag)
        if state is None:
            if self.steps_left() <= FINISH_ROOM + 3:
                return -1
            self.searched.add(tag)
            observation = self.search(text, category, cap)
            if text is not None and category and self.combined_search is False:
                tag = json.dumps([text, None, cap]).lower()     # the store took the keywords alone
            total, more = self.counts(observation)
            shown = len(self.product_rows(observation))
            state = self.page_state[tag] = {"page": 1, "more": more, "total": total if total is not None else shown, "shown": shown, "known": total is not None}
            self.rows_of.setdefault(tag, []).extend(self.row_keys(observation))
        if pages is None:
            pages = self.pages if state["shown"] and state["total"] <= state["shown"] * self.pages else 1
        elif pages >= self.deep and state["shown"] and state["total"] > state["shown"] * self.deep:
            pages = self.pages                    # more matches than the deep read covers: the usual pages, then narrower queries
        if text is not None and state["known"] and state["total"] >= RANKED_HITS:
            pages = min(pages, 2)                 # a ranked search: the best matches are on the first pages
        if 0 < state["shown"] <= 5 and pages > 1 and state["total"] > state["shown"] * pages:
            pages = min(pages * 2, 12)            # a store showing four or five rows a page is read twice as deep
        pager = (self.roles.get("search") or {}) if text is not None or (self.roles.get("search") or {}).get("category_arg") else (self.roles.get("browse") or {})
        while state["more"] and pager.get("page_arg") and state["page"] < pages and self.steps_left() > FINISH_ROOM + 5:
            state["page"] += 1
            more = state["more"]
            observation = self.search(text, category, cap, page=more if isinstance(more, (int, str)) else state["page"])
            _, state["more"] = self.counts(observation)
            self.rows_of.setdefault(tag, []).extend(self.row_keys(observation))
        return state["total"] if state["total"] is not None else -1

    def crowded(self, category: str, cap: float | None) -> bool:
        """True when a category search under `cap` matched more listings than its pages show (or gave no count yet has
        more pages): the cheapest end is not all in view."""
        state = self.page_state.get(json.dumps([None, category, cap]).lower()) or {}
        if not state:
            return False
        if state.get("known") is False:
            return bool(state.get("more"))
        return int(state.get("total") or 0) > int(state.get("shown") or 0) * self.pages

    def open_listings(self) -> list[str]:
        """Listings that may still fit: within the budget, in stock, in the category, not ruled out; cheapest first."""
        out = []
        for key in self.items:
            if key in self.gone:
                continue
            price = self.price(key)
            if self.budget is not None and price is not None and price * self.quantity > self.budget + 1e-9:
                continue
            if self.plan.get("stocked_only", True) is not False and self.in_stock(key) is False:
                continue
            if self.categories and self.keys.get("category") and self.category_of(key) not in self.categories:
                continue
            verdict = self.verdicts.get(key)
            if verdict and verdict.get("stamp") == self.stamp() and verdict.get("ok") is False:
                continue
            if self.screen_stamp == self.stamp() and self.screen.get(key) == "unlikely" and not self.wants_cheapest() and not self.mentions(key):
                continue                                # a row writing a required value stays whatever the screen made of it
            out.append(key)
        return sorted(out, key=lambda k: (self.price(k) if self.price(k) is not None else 1e18, k))

    def settle_category(self) -> None:
        """The store category of the wanted item, from the categories the searches turned up."""
        tally: dict[str, int] = {}
        for key in self.items:
            label = self.category_of(key)
            if label:
                tally[label] = tally.get(label, 0) + 1
        if not tally:
            return
        named = str(self.plan.get("category") or "").strip().lower()
        exact = [label for label in tally if label.lower() == named] or self.close_category(named, tally)
        if exact:
            self.categories = exact[:1]
            self.note(f"category: {self.categories} (named in the request) from {tally}")
            self.drop_kind_words()
            return
        shown = "\n".join(f"- {label} ({count})" for label, count in sorted(tally.items(), key=lambda kv: -kv[1]))
        answer = self.models.ask(CATEGORY_GUIDE, f"Shopper wants: {self.plan.get('item')} (request says category: {self.plan.get('category') or '-'})\n"
                                 f"Request: {self.request[:700]}\n\nCategories seen:\n{shown}", strong=False, max_tokens=400)
        picked = [str(c) for c in ((answer or {}).get("categories") or []) if str(c) in tally] if isinstance(answer, dict) else []
        self.categories = picked[:2]
        self.note(f"category: {self.categories} from {tally}")
        self.drop_kind_words()

    @staticmethod
    def carrying_words(text: str) -> list[str]:
        """The words of a category name that carry it, in order: the longer ones, unless that leaves fewer than two."""
        found = words(text)
        long = [w for w in found if len(w) > 3]
        return long if len(long) >= 2 else found

    @classmethod
    def label_words(cls, text: str) -> set[str]:
        return {w[:-1] if w.endswith("s") and len(w) > 3 else w for w in cls.carrying_words(text)}

    def close_category(self, named: str, tally: dict[str, int]) -> list[str]:
        """The category whose name says what the request's category says, in other words or another form."""
        want = self.label_words(named)
        if not want:
            return []
        scored = [(len(want & self.label_words(label)) / len(want | self.label_words(label)), tally[label], label) for label in tally]
        scored = [entry for entry in scored if entry[0] >= 0.6]
        return [max(scored)[2]] if scored else []

    def gather(self) -> None:
        """Keyword searches from the plan, the category settled from them, then the category within the budget; when
        the cheapest is wanted, the cheapest end of the category in full (the cap halved until every match shows)."""
        queries = [q for q in self.plan.get("search_queries") or [] if isinstance(q, str) and q.strip()]
        several = len(self.wants) > 1
        named = " ".join(self.carrying_words(str(self.plan.get("category") or "")))
        if named and named not in [q.lower() for q in queries] and (several or not (self.roles.get("search") or {}).get("category_arg")):
            queries.insert(0, named)                # the named category's own words reach its listings, unless the category search below covers it
        item = str(self.plan.get("item") or "").strip()
        if item and item.lower() not in [q.lower() for q in queries]:
            queries.append(item)
        mine = len([k for k in self.items if self.categories and self.category_of(k) in self.categories]) if self.categories else 0
        for number, query in enumerate(queries[: 2 if several else 4]):
            if self.steps_left() <= FINISH_ROOM + 10 * (len(self.wants) - self.want_at):
                break
            total = self.search_all(text=query, pages=self.pages if number == 0 and self.requirements and not several else None)
            if len(self.items) - mine >= self.enough and total > 0:
                break
        self.settle_category()
        within = self.filtered()
        deep = self.deep if not several else self.pages
        for number, requirement in enumerate(self.requirements[: 2 if several else 3]):
            query = self.value_query(requirement["value"])
            if query and self.steps_left() > FINISH_ROOM + 10 * (len(self.wants) - self.want_at):
                self.search_all(text=query, category=within, pages=deep if number == 0 else self.pages if number == 1 and not several else None)
        if not self.requirements and within and item and self.steps_left() > FINISH_ROOM + 10 * (len(self.wants) - self.want_at):
            self.search_all(text=item, category=within, pages=self.pages if not several else None)
        search = self.roles.get("search") or {}
        if not (search.get("category_arg") or self.roles.get("browse")) or not self.categories or several:
            return
        label = self.categories[0]
        total = self.search_all(category=label, cap=self.budget)
        capped = search.get("price_arg") or (self.roles.get("browse") or {}).get("price_arg")
        if not (self.wants_cheapest() and capped and self.budget and self.crowded(label, self.budget)):
            return
        # the highest cap under which the category's cheapest listings all show in the pages, found by halving (six
        # steps at most), then read in full: the cheapest fit is among a store's cheapest few dozen of the kind
        low, high, best = 0.0, self.budget, None
        for _ in range(6):
            if self.steps_left() <= FINISH_ROOM + 14:
                break
            cap = round((low + high) / 2, 2)
            total = self.search_all(category=label, cap=cap, pages=1)
            if self.crowded(label, cap):
                high = cap
            else:
                low = cap
                if total > 0:
                    best = cap
                    state = self.page_state.get(json.dumps([None, label, cap]).lower()) or {}
                    if total > int(state.get("shown") or 0) * max(1, self.pages - 1):
                        break                       # the pages are nearly full at this cap: the cheapest end is in view
        if best is None and high < self.budget:
            best = high                             # no cap showed everything: the pages under the lowest crowded one still hold the cheapest end
        if best is not None:
            self.search_all(category=label, cap=best, pages=self.pages)

    # ---- the shopper ---------------------------------------------------------------------------------------------
    def events(self, observation: Any, asked: str = "") -> list[str]:
        """Read shopper lines that came with a result; apply what they change."""
        lines = self.shopper_lines(observation)
        if not lines:
            return []
        state = (f"Item: {self.plan.get('item')}\nRequirements now: {json.dumps(self.requirements)}\nExclusions now: {json.dumps(self.exclusions)}\n"
                 f"Budget now: {self.budget} {self.currency}\nQuantity now: {self.quantity}\n")
        if asked:
            state += f"The assistant had asked: {asked}\n"
        answer = self.models.ask(LINE_GUIDE, state + "Shopper lines:\n" + "\n".join(lines), strong=False, max_tokens=800)
        answer = answer if isinstance(answer, dict) else {}
        self.note(f"shopper said {lines!r} -> {json.dumps(answer)[:600]}")
        self.last_reply_kind = str(answer.get("kind") or "")
        if asked and len(lines) == 1 and self.last_reply_kind in ("indifferent", "confused", "one_at_a_time"):
            return lines                     # 'I don't mind' to our question changes nothing
        target = self.want_for(answer.get("item")) if answer.get("item") else self.want_for(" ".join(lines))
        current = self.want_at
        if target is not None and target != current:
            self.save_want()
            self.load_want(target)           # the lines speak of another of the wanted items
        for prop in answer.get("drop") or []:
            self.requirements = [r for r in self.requirements if not self.same_property(r["property"], str(prop))]
            self.exclusions = [r for r in self.exclusions if not self.same_property(r["property"], str(prop))]
        for prop in answer.get("unexclude") or []:
            self.exclusions = [r for r in self.exclusions if not self.same_property(r["property"], str(prop))]
        for add in answer.get("add") or []:
            if isinstance(add, dict) and str(add.get("value") or "").strip():
                prop = str(add.get("property") or "").strip()
                self.requirements = [r for r in self.requirements if not self.same_property(r["property"], prop)]
                self.requirements.append({"property": prop, "value": str(add["value"]).strip()})
        for ex in answer.get("exclude") or []:
            if isinstance(ex, dict) and str(ex.get("value") or "").strip():
                self.exclusions.append({"property": str(ex.get("property") or "").strip(), "value": str(ex["value"]).strip()})
        budget = self.number(answer.get("budget"))
        if budget:
            written = [a for line in lines for a in self.amounts_in(line)] or [self.number(m) for line in lines for m in re.findall(r"\d[\d,]*(?:\.\d+)?", line)]
            written = [w for w in written if w]
            if written and budget not in written:
                budget = min(written, key=lambda w: abs(w - budget))       # the shopper's own figure, in their unit
            self.budget = budget
        quantity = self.number(answer.get("quantity"))
        if quantity and quantity >= 1:
            self.quantity = int(quantity)
        reference = answer.get("reference")
        if isinstance(reference, dict) and str(reference.get("property") or "").strip() and (reference.get("shop") or reference.get("category")):
            prop = str(reference["property"]).strip()
            value = self.resolve_reference(reference)
            if value:
                if reference.get("negated") or str(reference.get("relation") or "").lower().startswith("differ"):
                    self.exclusions.append({"property": prop, "value": value})
                else:
                    self.requirements = [r for r in self.requirements if not self.same_property(r["property"], prop)]
                    self.requirements.append({"property": prop, "value": value})
            else:
                self.note(f"the page {prop!r} refers to was not found: the property stays open")
        # a value that only points at another page cannot be checked against a listing
        self.requirements = [r for r in self.requirements if not REFERENCE_WORDS.search(r["value"])]
        self.exclusions = [r for r in self.exclusions if not REFERENCE_WORDS.search(r["value"])]
        if answer.get("chosen_unavailable"):
            own = [self.wants[self.want_at].get("choice")] if self.wants[self.want_at].get("choice") else []
            for key in (own if len(self.wants) > 1 and (target is not None or len(self.cart) <= 1) else list(self.cart)) or self.order_tried[-1:]:
                self.gone.add(key)
        if target is not None and target != current:
            self.save_want()
            self.load_want(current)
        return lines

    def resolve_reference(self, reference: dict[str, Any], hop: int = 0, trail: list[str] | None = None) -> str | None:
        """The value a shopper (or a listing's page) defines by another listing's page: "the same as what the <category>
        from <shop> with <attributes> says about <property>". That listing is found through the shop's name and its
        attributes (a ranked search lists a shop's own rows first for its name), read, and when its page text sends the
        property on to yet another listing ("For color, look to the Sneakers item from Notaan ..."), that one is read in
        turn, up to POINTER_HOPS pages along the chain. The value is the first page's own fact for the property that no
        pointer overrides, or the value its text states."""
        prop = str(reference.get("property") or "").strip()
        shop = str(reference.get("shop") or "").strip()
        category = str(reference.get("category") or "").strip()
        given = reference.get("attributes")
        if isinstance(given, dict):
            values = [f"{k} {v}".strip() for k, v in given.items() if str(v).strip()]
        else:
            values = [str(v).strip() for v in (given if isinstance(given, list) else []) if str(v).strip()]
        if not prop or not (shop or category) or hop >= POINTER_HOPS or self.steps_left() <= FINISH_ROOM + 5 or self.time_left() < 45:
            return None
        tag = json.dumps([prop.lower(), shop.lower(), category.lower(), sorted(v.lower() for v in values)])
        trail = list(trail or [])
        if tag in trail:
            return None                              # a chain that comes back on itself
        trail.append(tag)
        self.note(f"looking up {prop!r} on the page of {category!r} from {shop!r} with {values} (hop {hop})")
        shop_words = words(shop)
        filler = {"the", "and", "with", "kept", "in", "on", "of", "for", "a", "an", "type", "one", "that", "its", "it", "is", "are", "from", "by", "as", "comes", "come", "style", "made"}

        def content(text: str) -> list[str]:
            return [w for w in words(text) if w not in filler and len(w) > 1]

        def of_shop(key: str) -> bool:
            if not shop_words:
                return False
            record = self.items.get(key, {})
            return any(isinstance(v, str) and words(v) == shop_words for v in record.values())

        def in_category(key: str) -> bool:
            return not category or not self.keys.get("category") or self.same_property(self.category_of(key), category) or bool(set(self.label_words(category)) & set(self.label_words(self.category_of(key))))

        def fold_words(text: Any) -> set[str]:
            return {w[:-1] if len(w) > 3 and w.endswith("s") else w for w in words(text)}

        def matched(key: str) -> int:
            """How many of the attributes the listing shows, each within one fact (its label and value together) or the title."""
            sheet = self.sheets.get(key) or {}
            facts = [fold_words(f"{label} {value}") for label, value in sheet.items() if isinstance(value, (str, int, float))]
            facts.append(fold_words(self.title_of(key)))
            hits = 0
            for v in values:
                need = {w[:-1] if len(w) > 3 and w.endswith("s") else w for w in content(v)}
                if need and any(need <= fact for fact in facts):
                    hits += 1
            return hits

        needed = max(1, (len(values) + 1) // 2) if values else 0

        def candidates() -> list[str]:
            pool = [k for k in self.items if of_shop(k) and in_category(k)] or [k for k in self.items if of_shop(k)]
            if not pool and not shop_words:
                pool = [k for k in self.items if k in found_now and in_category(k)]
            return sorted(pool, key=lambda k: (-matched(k), 0 if k in self.sheets else 1, self.price(k) if self.price(k) is not None else 1e18))

        attribute_words = [w for v in values for w in content(v) if not w.isdigit()][:4]
        queries: list[str] = []
        if shop:
            queries.append(" ".join([shop] + attribute_words[:3]))
            queries.append(" ".join([shop] + self.query_terms(category)[:2]))
            queries.append(shop)
        else:
            queries.append(" ".join(self.query_terms(category)[:2] + attribute_words[:3]))
        found_now: set[str] = set()
        before = set(self.items)
        if shop and self.search_shop(shop, category):
            found_now |= set(self.items) - before
        for query in queries:
            query = " ".join(query.split())[:60]
            if not query or self.steps_left() <= FINISH_ROOM + 6:
                break
            pool = candidates()
            if pool and (not values or matched(pool[0]) >= len(values)):
                break                               # the listing pointed to is on hand
            if len(pool) >= 2 and query != queries[0]:
                break                               # the shop's listings are on hand: their datasheets tell them apart
            self.search_all(text=query, pages=1)
            found_now |= set(self.items) - before
        chosen: list[str] = []
        for key in candidates()[:4]:
            if self.steps_left() <= FINISH_ROOM + 4 or self.time_left() < 40:
                break
            self.read_sheet(key)
            self.read_page(key, room=FINISH_ROOM + 2)
            chosen.append(key)
            if not values or matched(key) >= len(values):
                break                               # every attribute given shows on this page: it is the one
        chosen.sort(key=lambda k: -matched(k))
        if chosen and values and matched(chosen[0]) < (len(values) if len(values) >= 2 else needed):
            # no read page shows every detail literally: a model reads them the way a shop assistant would (16 x 500 gb is
            # 8000 gb, a value may sit in another field); the shop's other unread pages are read first while steps allow
            for key in [k for k in candidates() if k not in chosen][:4]:
                if self.steps_left() <= FINISH_ROOM + 8 or self.time_left() < 60:
                    break
                self.read_sheet(key)
                self.read_page(key, room=FINISH_ROOM + 2)
                chosen.append(key)
                if matched(key) >= len(values):
                    break
            chosen.sort(key=lambda k: -matched(k))
            if matched(chosen[0]) < (len(values) if len(values) >= 2 else needed):
                picked = self.pick_pointed(chosen[:8], shop, category, values)
                if picked:
                    chosen = [picked] + [k for k in chosen if k != picked]
                else:
                    self.note(f"no page of {shop or category!r} shows the details given ({values}): the pointer stays unresolved")
                    chosen = []
        for key in chosen[:1]:
            page = self.page_pointers(key)
            onward = [d for d in page.get("deferrals") or [] if self.same_property(str(d.get("property") or ""), prop)]
            for deferral in onward:
                self.note(f"the page of {key} sends {prop!r} on to {deferral}")
                value = self.resolve_reference({"property": prop, "shop": deferral.get("shop"), "category": deferral.get("category"), "attributes": deferral.get("attributes") or []}, hop + 1, trail)
                if value:
                    return value
            if onward:
                continue                            # the pointer could not be followed: this page's own words do not count
            own = None
            for label, value in self.property_facts(key, prop):
                self.note(f"the page of {key} says {label}: {value}")
                own = str(value).strip()
                break
            if own is None:
                for stated, value in (page.get("stated") or {}).items():
                    if self.same_property(str(stated), prop) and str(value).strip():
                        self.note(f"the page of {key} states {stated}: {value}")
                        own = str(value).strip()
                        break
            if own is not None and hop == 0:
                return self.related_value(key, prop, own, reference)
            if own is not None:
                return own
        return None

    def pick_pointed(self, keys: list[str], shop: str, category: str, values: list[str]) -> str | None:
        """The read listing a pointer's description means, by a model, when none shows every detail word for word."""
        if not keys or not values:
            return None
        text = (f"Description: the {category or 'listing'} from {shop or 'the shop'} with {', '.join(values)}\n\nListings read:\n"
                + "\n".join(json.dumps({"id": k, **self.card(k), "page": self.page_text(k)[:600]}, ensure_ascii=False)[:1800] for k in keys))
        answer = self.models.ask(PICK_GUIDE, text, strong=True, max_tokens=200)
        picked = str((answer or {}).get("id") or "") if isinstance(answer, dict) else ""
        if picked in keys:
            self.note(f"the page meant by {values} is read as {picked} (sure: {(answer or {}).get('sure')})")
            return picked
        return None

    def related_value(self, key: str, prop: str, own: str, reference: dict[str, Any]) -> str:
        """The value a pointer's relation makes of the target page's own value: 'within' a tolerance gives a between-limit,
        'pairs' gives the partner a rule on that page names (or the page's value when it states none), else the value."""
        relation = str(reference.get("relation") or "").lower()
        if relation.startswith("within"):
            figures = self.numbers(str(reference.get("tolerance") or ""))
            found = self.numbers(own)
            if figures and found:
                span, unit = figures[0][0], figures[0][1] or found[0][1]
                centre = found[0][0]
                value = f"between {centre - span:g} and {centre + span:g} {unit}".strip()
                self.note(f"{prop!r} must lie inside {reference.get('tolerance')} of {own!r}: {value}")
                return value
        if relation.startswith("pair") or relation.startswith("compat") or relation.startswith("mate"):
            text = f"Property: {prop}\nThis page's own value: {own}\nPage facts: {json.dumps(self.sheets.get(key) or {}, ensure_ascii=False)[:1500]}\nPage text: {self.page_text(key)[:1200]}"
            answer = self.models.ask(RULE_GUIDE, text, strong=True, max_tokens=200)
            value = str((answer or {}).get("value") or "").strip() if isinstance(answer, dict) else ""
            if value:
                self.note(f"{prop!r} must go with {own!r}: the page's rule {str((answer or {}).get('rule') or '')[:80]!r} gives {value!r}")
                return value
        return own


    def search_shop(self, shop: str, category: str = "") -> bool:
        """One call of the store's brand or manufacturer filter for a shop's listings, when a tool takes one (the browse
        tool's brand argument, else the search tool's); False when no tool does or the call was made before."""
        for role in ("browse", "search"):
            spec = self.roles.get(role) or {}
            brand_arg = spec.get("brand_arg")
            if not spec.get("tool") or not brand_arg:
                continue
            tag = json.dumps(["shop", shop.lower(), category.lower()])
            if tag in self.searched or self.steps_left() <= FINISH_ROOM + 5:
                return False
            self.searched.add(tag)
            args = dict(spec.get("fields_args_fixed") or spec.get("fixed") or {})
            args[brand_arg] = shop
            category_arg = spec.get("category_arg")
            if category_arg and category:
                rule = self.params(spec["tool"]).get(category_arg) or {}
                if rule.get("enum"):
                    close = [v for v in rule["enum"] if isinstance(v, str) and self.label_words(v) & self.label_words(category)]
                    if close:
                        args[category_arg] = close[0]
                    elif category_arg in (self.params(spec["tool"]).get("required") or []):
                        return False
                else:
                    known = [c for c in self.categories if self.label_words(c) & self.label_words(category)]
                    args[category_arg] = known[0] if known else category
            elif category_arg and category_arg in ((self.spec(spec["tool"]) or {}).get("parameters") or {}).get("required", []):
                return False
            observation = self.call(spec["tool"], args)
            self.events(observation)
            return bool(self.product_rows(observation))
        return False

    def page_pointers(self, key: str) -> dict[str, Any]:
        """What a listing's page text defers to other listings and what it states outright, parsed once by a model when the
        text reads like a pointer: {"deferrals": [{"property", "shop", "category", "attributes"}], "stated": {property: value}}."""
        if key in self.defer_cache:
            return self.defer_cache[key]
        text = self.page_text(key)
        found: dict[str, Any] = {"deferrals": [], "stated": {}}
        if DEFER_WORDS.search(text):
            answer = self.models.ask(DEFER_GUIDE, text[:3000], strong=False, max_tokens=600)
            if isinstance(answer, dict):
                found["deferrals"] = [d for d in (answer.get("deferrals") or []) if isinstance(d, dict) and str(d.get("property") or "").strip() and (d.get("shop") or d.get("category"))]
                stated = answer.get("stated")
                found["stated"] = {str(k): str(v) for k, v in stated.items() if str(v).strip()} if isinstance(stated, dict) else {}
        self.defer_cache[key] = found
        return found

    def deferrals(self, key: str, prop: str | None = None) -> list[dict[str, Any]]:
        """Where a listing's page sends a property to another listing (all of them, or those of one property)."""
        found = self.page_pointers(key).get("deferrals") or []
        return [d for d in found if prop is None or self.same_property(str(d.get("property") or ""), prop)]

    def apply_deferrals(self, key: str) -> None:
        """A listing's page may send a required property to another listing ("For color, look to the Sneakers item from
        Notaan ..., rather than to this rug"): then the value written there is the one that counts, whatever the listing's
        own facts say. Followed, within the step room, for a listing that is otherwise a fit; a pointer that cannot be
        followed leaves the check unsettled."""
        verdict = self.verdicts.get(key) or {}
        if verdict.get("stamp") != self.stamp() or verdict.get("ok") is False or key in self.deferred_done or not self.requirements:
            return
        if not DEFER_WORDS.search(self.page_text(key)):
            return
        self.deferred_done.add(key)
        pointers = self.deferrals(key)
        if not pointers:
            return
        checks, grades = list(verdict.get("checks") or []), list(verdict.get("grades") or [])
        while len(grades) < len(checks):
            grades.append(0)
        for at, requirement in enumerate(self.requirements[:len(checks)]):
            onward = [d for d in pointers if self.same_property(str(d.get("property") or ""), requirement["property"])]
            if not onward:
                continue
            if self.steps_left() <= FINISH_ROOM + 8 or self.time_left() < 60:
                checks[at], grades[at] = "?", 0
                self.note(f"{key}: {requirement['property']!r} is defined by another page, which there is no room to open")
                continue
            value = self.resolve_reference({"property": requirement["property"], "shop": onward[0].get("shop"), "category": onward[0].get("category"), "attributes": onward[0].get("attributes") or []}, hop=1)
            if value is None:
                mark, grade = "?", 0
            else:
                mark, grade = self.compare_value(value, requirement)
            self.note(f"{key}: {requirement['property']!r} is defined by another page -> {value!r}: {mark}")
            checks[at], grades[at] = mark, (grade if mark == "y" else 0)
        ok: bool | None = verdict.get("kind") == "y" and all(m == "y" for m in checks)
        if not ok and verdict.get("kind") != "n" and "n" not in checks:
            ok = None
        self.verdicts[key] = {**verdict, "ok": ok, "checks": checks, "grades": grades}

    @staticmethod
    def same_property(a: str, b: str) -> bool:
        def stems(text: str) -> set[str]:
            return {w[:-1] if len(w) > 3 and w.endswith("s") else w for w in words(text) if len(w) > 2}
        x, y = stems(a), stems(b)
        return bool(x and y and (x <= y or y <= x))

    @staticmethod
    def core_label(label: str) -> str:
        """A fact label's plain name: separators as spaces, camel case split ('clothingMaterial' -> 'clothing material')."""
        spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(label))
        return re.sub(r"[._\-]+", " ", spaced).strip().lower()

    @staticmethod
    def variant_label(label: str, values: set[str]) -> bool:
        """A fact that lists a listing's purchase variants or stock choices rather than a property of the item."""
        if re.search(r"\b(?:option|variant|sku|stock|availab|condition|warranty|shipping|delivery)", label):
            return True
        return sum(1 for v in values if "/" in v or re.search(r"\bint:|\bx\d|\d\s*(?:pcs|pc|pack)\b", v)) > len(values) / 2

    @staticmethod
    def plain_property(label: str) -> str:
        """A fact label as a shopper would say it: short code-like prefixes and suffixes dropped ('fa general styles' ->
        'general styles', 'm sleeve length' -> 'sleeve length', 'storage type gr' -> 'storage type')."""
        parts = label.split()
        while len(parts) > 1 and len(parts[0]) <= 3 and not parts[0].isdigit():
            parts = parts[1:]
        while len(parts) > 1 and len(parts[-1]) <= 2 and not parts[-1].isdigit():
            parts = parts[:-1]
        return " ".join(parts)

    def question_labels(self) -> list[str]:
        """Fact labels to ask about, likeliest first: by how many of the category's read listings carry the label and
        how many values it takes, then re-ranked by a model that knows the request; never one already required."""
        read = 0
        stats: dict[str, dict[str, Any]] = {}
        for key, sheet in self.sheets.items():
            if not sheet or (self.categories and self.keys.get("category") and self.category_of(key) not in self.categories):
                continue
            if len(self.wants) > 1 and not self.categories and key not in self.found_now:
                continue                        # another wanted item's listing: its properties are not this item's
            read += 1
            seen_here: set[str] = set()
            for label, value in sheet.items():
                if not isinstance(value, (str, int, float)) or isinstance(value, bool):
                    continue
                core = self.core_label(label)
                if not core:
                    continue
                entry = stats.setdefault(core, {"carriers": 0, "values": set()})
                if core not in seen_here:
                    entry["carriers"] += 1
                    seen_here.add(core)
                entry["values"].add(str(value).strip().lower()[:40])
        taken = self.requirements + self.exclusions
        scored = sorted(((entry["carriers"] / max(1, read) * min(len(entry["values"]), 6) * (0.15 if AVOID_ASKING.search(core) else 1.0) * (1.0 if len(entry["values"]) >= 2 else 0.3), core, entry) for core, entry in stats.items()
                        if (len(entry["values"]) >= 2 or entry["carriers"] >= 2) and not self.variant_label(core, entry["values"]) and not any(self.same_property(core, r["property"]) for r in taken)), key=lambda sc: -sc[0])
        order = [core for _, core, _ in scored]
        shown = "\n".join(f"- {core} ({entry['carriers']} of {read} listings): {', '.join(sorted(entry['values'])[:6])}" for _, core, entry in scored[:16])
        text = (f"Request:\n{self.request}\n\nThe item: {self.plan.get('item')} (store category: {', '.join(self.categories) or '-'})\n"
                f"Already required: {json.dumps(taken)}\nPurpose and hints: {self.plan.get('hidden_needs_hint')}\n\nProperties the listings carry:\n{shown or '(none read)'}")
        answer = self.models.ask(QUESTION_GUIDE, text, strong=True, max_tokens=500)
        picked = [str(p).strip().lower() for p in ((answer or {}).get("ask") or []) if str(p).strip()] if isinstance(answer, dict) else []
        picked = [p for p in picked if p in stats and not any(self.same_property(p, r["property"]) for r in taken)]
        order = picked + [core for core in order if core not in picked]
        order = [p for p in order if not any(self.same_property(p, done) for done in self.asked)]
        self.label_values = {core: sorted(entry["values"])[:3] for core, entry in stats.items()}
        self.note(f"question labels: {order[:6]} (scores {[(round(sc, 2), core) for sc, core, _ in scored[:6]]}) from {read} datasheets")
        return order

    def question_for(self, props: list[str]) -> str:
        """One question naming one item and one or several of its properties: the shopper answers about the held-back one
        when it is among those named (in race 14 a question naming eight drew a disclosure in 36% of messages, a question
        naming one in 17%), and a list of properties reads as one question. No value examples, which draw puzzled answers."""
        item = " ".join(words(str(self.plan.get("item") or ""))[:4])
        names = [self.plain_property(p) for p in props]
        if len(names) == 1:
            about = f" for the {item}" if item else ""
            return f"What {names[0]} would you like{about}?"
        listed = ", ".join(names[:-1]) + f" or {names[-1]}"
        return f"For the {item or 'item'}, what {listed} would you like?"

    def ask_shopper(self) -> None:
        """The messages the budget allows, spread over the wanted items (in race 14 the held-back wish sat on any of the
        items equally often), each one question naming up to QUESTION_PROPS properties of the item. The properties are
        dealt round-robin into as many groups as messages remain, so every message LEADS with one of the likeliest
        (the practice shopper answered only about the first property named) while all of them get named (the race-14
        shopper answered about any named one). A disclosure removes its property; the others named with it are asked
        again only once every other one has been. A 'one at a time' answer switches to single-property questions."""
        spec = self.roles.get("message")
        if not spec or self.message_budget <= 0:
            return
        limit = min(self.message_budget, ASK_MOST)
        if self.roles.get("checkout") and self.confirm_needed:
            limit = max(0, limit - 1)         # one message is kept for putting the basket to the shopper before ordering
        count = max(1, len(self.wants))
        reserved = sum(1 for j in range(limit) if j % count > self.want_at)      # messages kept for the items after this one
        allowed = max(0, limit - reserved)
        if self.messages_sent >= allowed:
            return
        cap = (self.params(spec["tool"]).get(spec["text_arg"]) or {}).get("maxLength")
        cap = int(cap) if isinstance(cap, int) and cap >= 60 else 1000
        names: dict[str, list[str]] = {}          # plain name -> the labels it stands for ('pattern' and 'fa pattern')
        ordered: list[str] = []
        for label in self.question_labels():
            if any(self.same_property(label, done) for done in self.asked):
                continue
            name = self.plain_property(label)
            if name in names:
                names[name].append(label)
                continue
            names[name] = [label]
            ordered.append(label)
        rounds = max(1, allowed - self.messages_sent)
        if self.single_questions:
            groups = [[p] for p in ordered]
        else:
            groups = [g for g in (ordered[j::rounds][:QUESTION_PROPS] for j in range(rounds)) if g]
        leftovers: list[str] = []
        while self.messages_sent < allowed:
            if self.steps_left() <= FINISH_ROOM + 6 or self.time_left() < 40:
                break
            if not groups:
                if not leftovers:
                    break
                groups, leftovers = [leftovers[:QUESTION_PROPS]], leftovers[QUESTION_PROPS:]
            group = groups.pop(0)
            if self.single_questions and len(group) > 1:
                groups = [[p] for p in group[1:]] + groups
                group = group[:1]
            question = self.question_for(group)
            while len(group) > 1 and len(question) > cap:
                leftovers.append(group[-1])
                group = group[:-1]
                question = self.question_for(group)
            covered = [label for g in group for label in names.get(self.plain_property(g), [g])]
            before = [json.dumps(r, sort_keys=True) for r in self.requirements + self.exclusions]
            self.messages_sent += 1
            observation = self.call(spec["tool"], {spec["text_arg"]: question})
            if isinstance(observation, dict) and "error" in json.dumps(observation).lower() and len(json.dumps(observation)) < 200:
                self.note(f"message refused: {observation}")
                break
            self.events(observation, asked=question)
            if self.last_reply_kind == "one_at_a_time" and len(group) > 1:
                self.single_questions = True
                self.note("the shopper wants one property at a time: single questions from here")
                groups = [[p] for p in group] + [[p] for g in groups for p in g]       # nothing was answered: all still open
                continue
            self.asked.extend(covered)
            new = [r for r in self.requirements + self.exclusions if json.dumps(r, sort_keys=True) not in before]
            if new:
                told = [str(r.get("property") or "") for r in new]
                leftovers.extend(p for p in group if not any(self.same_property(p, t) for t in told))

    # ---- judging ---------------------------------------------------------------------------------------------------
    def stamp(self) -> str:
        return json.dumps([self.requirements, self.exclusions, self.plan.get("item"), self.categories], sort_keys=True)

    def read_sheet(self, key: str) -> None:
        spec = self.roles.get("datasheet") or self.roles.get("listing")
        if not spec or key in self.sheets or self.steps_left() <= FINISH_ROOM:
            return
        observation = self.call(spec["tool"], {spec["id_arg"]: key})
        self.sheets.setdefault(key, {})
        if not self.sheets[key] and len(self.flat_facts(key)) >= 2:
            self.sheets[key].update(self.flat_facts(key))    # a page without a fact block: its plain fields are the facts
        self.detailed.add(key)
        self.events(observation)
        if spec is self.roles.get("listing") or spec.get("tool") == (self.roles.get("listing") or {}).get("tool"):
            self.viewed.add(key)

    def read_details(self, key: str, room: int = 2) -> None:
        """The datasheet tool called for a listing seen only side by side: a shopper may require that each item ordered
        was looked at on its own."""
        spec = self.roles.get("datasheet") or self.roles.get("listing")
        if not spec or key in self.detailed or self.steps_left() <= room:
            return
        self.detailed.add(key)
        observation = self.call(spec["tool"], {spec["id_arg"]: key})
        self.sheets.setdefault(key, {})
        if not self.sheets[key] and len(self.flat_facts(key)) >= 2:
            self.sheets[key].update(self.flat_facts(key))
        self.events(observation)

    def recheck_cart(self) -> None:
        """Each carted listing looked at once more AFTER carting: its full page where the world has a page tool, and
        its datasheet too. Shoppers require that what is ordered was looked at as it now stands; in race 11 an order
        whose page was opened after carting failed that rule a quarter of the time, one opened only before it half."""
        page = self.roles.get("listing")
        sheet = self.roles.get("datasheet")
        specs = [s for s in (page, sheet) if s and s.get("tool") and s.get("id_arg")]
        if page and sheet and page.get("tool") == sheet.get("tool"):
            specs = [page]
        if not specs:
            return
        for key in list(self.cart):
            if key in self.rechecked:
                continue
            if self.steps_left() <= 4 + 2 * len(self.cart) + len(specs) * len(self.cart):
                specs = specs[:1]                   # short of steps: the page alone (the order and a retry keep theirs)
                if self.steps_left() <= 4 + 3 * len(self.cart):
                    continue
            self.rechecked.add(key)
            for spec in specs:
                observation = self.call(spec["tool"], {spec["id_arg"]: key})
                self.events(observation)

    def read_page(self, key: str, room: int = FINISH_ROOM) -> None:
        """Open a listing's full page when the world has one apart from the datasheet: a shopper may require it of
        anything ordered, and the page's text is where a property may be deferred to another listing."""
        spec = self.roles.get("listing")
        if not spec or key in self.viewed or spec.get("tool") == (self.roles.get("datasheet") or {}).get("tool") or self.steps_left() <= room:
            return
        self.viewed.add(key)
        observation = self.call(spec["tool"], {spec["id_arg"]: key})
        self.events(observation)

    def page_text(self, key: str) -> str:
        return " ".join(str(v) for v in self.items.get(key, {}).values() if isinstance(v, str))

    def compare_size(self) -> int:
        spec = self.roles.get("compare") or {}
        most = (self.params(str(spec.get("tool") or "")).get(str(spec.get("ids_arg") or "")) or {}).get("maxItems")
        size = max(2, min(COMPARE_MOST, most)) if isinstance(most, int) and most >= 2 else COMPARE_MOST
        return min(size, self.compare_cap) if self.compare_cap else size

    def read_sheets(self, keys: list[str]) -> None:
        """Datasheets of listings not read yet: several in one step through a tool that shows listings side by side when
        the world has one (given up if its answer carries no facts), else one step each."""
        todo = [k for k in keys if k not in self.sheets]
        compare = self.roles.get("compare")
        if compare and len(todo) > 1:
            size = self.compare_size()
            least = (self.params(compare["tool"]).get(compare["ids_arg"]) or {}).get("minItems")
            least = least if isinstance(least, int) else 1
            for at in range(0, len(todo), size):
                part = todo[at:at + size]
                if len(part) < max(2, least) or self.steps_left() <= FINISH_ROOM:
                    break
                observation = self.call(compare["tool"], {compare["ids_arg"]: part})
                self.events(observation)
                if observation is None:
                    break
                shown = [k for k in part if self.sheets.get(k)]
                if not shown and len(part) > 2 and not self.compare_cap and self.steps_left() > FINISH_ROOM:
                    # the store may take fewer ids at once than the schema let on: two of them, once
                    observation = self.call(compare["tool"], {compare["ids_arg"]: part[:2]})
                    self.events(observation)
                    shown = [k for k in part[:2] if self.sheets.get(k)]
                    if shown:
                        self.compare_cap = 2
                        size = 2
                        self.note("the side-by-side tool takes two listings at a time here")
                if not shown:
                    self.note("the side-by-side tool showed no facts: one listing a step from here")
                    self.roles.pop("compare", None)
                    break
                if size == 2 and len(part) > 2:
                    todo = [k for k in todo if k not in self.sheets]
                    for at2 in range(0, len(todo), 2):
                        pair = todo[at2:at2 + 2]
                        if len(pair) < 2 or self.steps_left() <= FINISH_ROOM:
                            break
                        self.events(self.call(compare["tool"], {compare["ids_arg"]: pair}))
                    break
            todo = [k for k in todo if k not in self.sheets]
        for key in todo:
            self.read_sheet(key)

    def row_card(self, key: str) -> dict[str, Any]:
        record = {k: v for k, v in self.items.get(key, {}).items()}
        for name, value in list(record.items()):
            if isinstance(value, str) and len(value) > 220:
                record[name] = value[:220]
        return record

    def card(self, key: str) -> dict[str, Any]:
        record = self.row_card(key)
        record["facts"] = self.sheets.get(key) or {}
        return record

    def checks(self) -> list[dict[str, str]]:
        return self.requirements + [{"property": f"NOT {r['property']}", "value": r["value"]} for r in self.exclusions]

    def wanted(self) -> str:
        category = ", ".join(self.categories) or str(self.plan.get("category") or "").strip()
        return (f"Item wanted: {self.plan.get('item')}\n" + (f"Store category: {category}\n" if category else "Store category: not given\n")
                + "Requirements, in order:\n" + "\n".join(f"{i}. {r['property']}: {r['value']}" + (" (must NOT have this value)" if r["property"].startswith("NOT ") else "")
                                                        for i, r in enumerate(self.checks(), start=1)))

    def screen_rows(self, keys: list[str]) -> None:
        """Rule out, from their search rows alone, listings plainly of another kind or stating another value."""
        if self.screen_stamp != self.stamp():
            self.screen, self.screen_stamp = {}, self.stamp()
        todo = [k for k in keys if k not in self.screen]
        for at in range(0, len(todo), SCREEN_BATCH):
            batch = todo[at:at + SCREEN_BATCH]
            text = self.wanted() + "\n\nRows:\n" + "\n".join(json.dumps({"id": k, **self.row_card(k)}, ensure_ascii=False)[:700] for k in batch)
            answer = self.models.ask(SCREEN_GUIDE, text, strong=False, max_tokens=400 + 50 * len(batch))
            for row in ((answer or {}).get("rows") or []) if isinstance(answer, dict) else []:
                if isinstance(row, dict) and str(row.get("id")) in batch:
                    self.screen[str(row["id"])] = str(row.get("fit") or "possible").lower()
            for key in batch:
                self.screen.setdefault(key, "possible")
            unlikely = [k for k in batch if self.screen.get(k) == "unlikely"]
            if unlikely:
                self.note("screened unlikely: " + ", ".join(unlikely[:12]))

    def judge(self, keys: list[str]) -> None:
        """Every requirement checked against each listing's own facts first (literal_check); the model is asked only about
        listings the letter leaves open (a check '?' or an unsettled kind) and never overrides a literal 'y' or 'n'."""
        todo = [k for k in keys if (self.verdicts.get(k) or {}).get("stamp") != self.stamp()]
        if not todo:
            return
        checks = self.checks()
        results: dict[str, dict[str, Any]] = {}
        for key in todo:
            marks: list[str] = []
            grades: list[int] = []
            for requirement in self.requirements:
                mark, grade = self.literal_check(key, requirement)
                marks.append(mark)
                grades.append(grade)
            for exclusion in self.exclusions:
                mark, grade = self.literal_check(key, exclusion, exclude=True)
                marks.append(mark)
                grades.append(grade)
            if self.categories and self.keys.get("category") and self.category_of(key):
                kind = "y" if self.category_of(key) in self.categories else "n"
            else:
                kind = "?"
            results[key] = {"marks": marks, "grades": grades, "kind": kind}
        open_keys = [k for k in todo if "n" not in results[k]["marks"] and results[k]["kind"] != "n" and (any(m in ("?", "-") for m in results[k]["marks"]) or results[k]["kind"] == "?")]
        for at in range(0, len(open_keys), JUDGE_BATCH):
            batch = open_keys[at:at + JUDGE_BATCH]
            text = self.wanted() + "\n\nListings:\n" + "\n".join(json.dumps({"id": k, **self.card(k)}, ensure_ascii=False)[:2200] for k in batch)
            self.judge_calls += 1
            answer = self.models.ask(JUDGE_GUIDE, text, strong=self.judge_calls <= STRONG_JUDGES, max_tokens=500 + 150 * len(batch))
            for row in ((answer or {}).get("listings") or []) if isinstance(answer, dict) else []:
                if not isinstance(row, dict) or str(row.get("id")) not in batch:
                    continue
                key = str(row["id"])
                said = [str(m).strip().lower()[:1] for m in (row.get("checks") or [])][:len(checks)]
                result = results[key]
                for i, mark in enumerate(result["marks"]):
                    if mark not in ("?", "-") or i >= len(said):
                        continue
                    if said[i] == "y":
                        requirement = checks[i]
                        is_exclusion = i >= len(self.requirements)
                        result["marks"][i] = "y"
                        result["grades"][i] = 1 if is_exclusion or mark == "-" or not self.unmentioned(key, requirement) else 0
                    elif said[i] == "n" or mark == "-":
                        result["marks"][i] = "n"           # the model confirms, or did not answer, another value
                        result["grades"][i] = 0
                if result["kind"] == "?":
                    result["kind"] = str(row.get("kind") or "?").strip().lower()[:1] or "?"
        for key in todo:
            result = results[key]
            result["marks"] = ["n" if m == "-" else m for m in result["marks"]]
            ok: bool | None = result["kind"] == "y" and all(m == "y" for m in result["marks"])
            if not ok and result["kind"] != "n" and "n" not in result["marks"]:
                ok = None                                   # unsettled: not ordered while a settled fit exists
            self.verdicts[key] = {"stamp": self.stamp(), "ok": ok, "checks": result["marks"], "grades": result["grades"], "kind": result["kind"]}
        self.note("judged " + ", ".join(f"{k}:{(self.verdicts.get(k) or {}).get('kind')}{''.join((self.verdicts.get(k) or {}).get('checks') or [])}/{''.join(str(g) for g in (self.verdicts.get(k) or {}).get('grades') or [])}" for k in todo))

    @staticmethod
    def stems(text: str) -> set[str]:
        return {w[:-1] if len(w) > 3 and w.endswith("s") else w for w in words(str(text).lower().replace("colour", "color")) if len(w) > 2}

    def fact_entries(self, key: str, prop: str) -> list[tuple[str, list[str]]]:
        """The listing's facts for a property: the labels carrying every word of the property ('material filter' for
        material), each value split into its entries. A label sharing only part of the name ('muslim top style' for
        'type of top', 'dial size' for 'dial glass') is no fact of it: on 771 practice orders the grader read none as one."""
        sheet = self.sheets.get(key) or {}
        want = self.stems(prop)
        avoided = not AVOID_ASKING.search(prop)
        rows = []
        for label, value in sheet.items():
            if not isinstance(value, (str, int, float)) or isinstance(value, bool):
                continue
            have = self.stems(self.core_label(label))
            rows.append((label, have, [e.strip().lower() for e in re.split(r"[,;/|]+", str(value)) if e.strip()]))
        return [(l, e) for l, h, e in rows if want and want <= h and not (avoided and AVOID_ASKING.search(self.core_label(l)))]

    @staticmethod
    def entry_has(entry: str, option: str) -> bool:
        have = set(words(entry))
        need = [w for w in words(option) if len(w) > 1]
        return bool(need) and all(w in have or w.rstrip("s") in have or w + "s" in have for w in need)

    @staticmethod
    def plain(text: Any) -> str:
        """A value or label as plain words: lower case, separators as spaces, figures kept whole ('1.77', '100%')."""
        text = str(text or "").lower().replace("&", " and ").replace("colour", "color")
        text = re.sub(r"[_\-/|;:,()\[\]{}]+", " ", text)
        return " ".join(re.findall(r"[a-z0-9]+(?:\.[0-9]+)?%?|\"", text))

    @staticmethod
    def numbers(text: str) -> list[tuple[float, str]]:
        """Each figure in a text with the unit word right after it ('' when none): '44.958 mm' -> [(44.958, 'mm')]."""
        out: list[tuple[float, str]] = []
        for found in re.finditer(r"(-?\d+(?:[.,]\d+)?)\s*([a-z\"%]{0,12})", str(text).lower()):
            raw = found.group(1)
            try:
                figure = float(raw.replace(",", "")) if re.fullmatch(r"-?\d{1,3}(?:,\d{3})+", raw) else float(raw.replace(",", "."))
            except ValueError:
                continue
            out.append((figure, Polaris.unit_word(found.group(2))))
        return out

    @staticmethod
    def unit_word(word: str) -> str:
        word = str(word or "").lower().strip(".")
        if re.fullmatch(r'"|in|inch|inches', word):
            return "inch"
        if len(word) > 4 and word.endswith("es"):
            word = word[:-2]
        elif len(word) > 3 and word.endswith("s"):
            word = word[:-1]
        return word

    def limit_of(self, value: str) -> tuple[str, Any, str] | None:
        """('>=' | '<=' | '==', figure, unit) for a value that is one measure or limit ('at least 3.9 inch', 'at most 101
        cm', '0.3 inches', '19 v'); ('[]', (low, high), unit) for 'between 10 and 14 inches' / 'from 10 to 14 cm'; None
        for words or for other several-figure values ('240 x 320 pixels')."""
        figures = self.numbers(value)
        text = value.lower()
        if len(figures) == 2 and re.search(r"\bbetween\b.*\band\b|\bfrom\b.*\bto\b", text) and not re.search(r"\bx\b|\*", text):
            low, high = sorted((figures[0][0], figures[1][0]))
            return "[]", (low, high), figures[1][1] or figures[0][1]
        if len(figures) != 1:
            return None
        if re.search(r"\b(?:at least|no less than|not less than|minimum|min|over|above|more than|from|or more|and up)\b", text):
            op = ">="
        elif re.search(r"\b(?:at most|no more than|not more than|up to|under|below|maximum|max|less than|within|or less)\b", text):
            op = "<="
        else:
            op = "=="
        return op, figures[0][0], figures[0][1]

    def property_facts(self, key: str, prop: str) -> list[tuple[str, str]]:
        """The listing's facts for a property, (label, value): the labels whose words are exactly the property's first,
        then those carrying every word of it ('material filter', 'display size inch', 'clock shape'). A label sharing
        only part of the name ('dial size' for 'dial glass') is no fact of it."""
        sheet = self.sheets.get(key) or {}
        want = self.stems(self.plain(prop))
        if not want:
            return []
        exact, wider = [], []
        avoided = not AVOID_ASKING.search(prop)
        for label, value in sheet.items():
            if not isinstance(value, (str, int, float)) or isinstance(value, bool) or not str(value).strip():
                continue
            core = self.core_label(label)
            have = self.stems(self.plain(core))
            if not want <= have or (avoided and AVOID_ASKING.search(core)):
                continue
            (exact if have == want else wider).append((label, str(value)))
        return exact + wider                      # one fact naming the value is enough, under whichever of its labels

    def value_marks(self, entries: list[str], value: str) -> tuple[str, int]:
        """How fact entries read against a required value: ('y', 3) when one is the value, ('y', 2) when one contains all
        its words (a list of variants, '80% wool' for wool), ('n', 0) when entries exist and none does, ('?', 0) without
        entries. A value naming alternatives is met by any of them."""
        options = [self.fold(o) for o in (self.alternatives(value) or [value]) if self.plain(o)]
        if not options:
            return ("?", 0)
        best = ("n", 0) if entries else ("?", 0)
        for entry in entries:
            plain = self.fold(entry)
            if not plain:
                continue
            tight = plain.replace(" ", "")
            have = set(plain.split())
            for option in options:
                option_tight = option.replace(" ", "")
                if plain == option or tight == option_tight:
                    return ("y", 3)
                if all(w in have for w in option.split()) or (len(option_tight) >= 6 and option_tight in tight):
                    best = ("y", 2)
        return best

    @classmethod
    def fold(cls, text: Any) -> str:
        """Plain words with plurals folded: 'long sleeves' and 'long sleeve' read the same."""
        return " ".join(w[:-1] if len(w) > 3 and w.endswith("s") and not re.search(r"\d", w) and not w.endswith("ss") else w for w in cls.plain(text).split())

    def limit_marks(self, facts: list[tuple[str, str]], limit: tuple[str, float, str]) -> tuple[str, int]:
        """Facts against a measure or limit: a figure in the same unit (the unit word after it, or in the label: 'display
        size inch') inside the limit is 'y'; a figure outside it 'n'; figures in another unit, or none, leave it open."""
        op, figure, unit = limit
        verdict = None
        for label, raw in facts:
            label_units = {self.unit_word(w) for w in self.plain(self.core_label(label)).split()}
            for number, found_unit in self.numbers(raw):
                if not found_unit and unit and unit in label_units:
                    found_unit = unit
                if unit and found_unit and found_unit != unit:
                    continue                          # another unit: not comparable here
                if op == "[]":
                    fits = figure[0] - 1e-9 <= number <= figure[1] + 1e-9
                elif op == ">=":
                    fits = number >= figure - 1e-9
                elif op == "<=":
                    fits = number <= figure + 1e-9
                else:
                    fits = abs(number - figure) <= 0.011 * max(1.0, abs(figure))
                if fits:
                    return ("y", 3 if (found_unit == unit or not unit) else 2)
                verdict = "n"
        return ("n", 0) if verdict == "n" else ("?", 0)

    def compare_value(self, value: str, requirement: dict[str, str]) -> tuple[str, int]:
        """One value (from a page a pointer led to) against a requirement."""
        limit = self.limit_of(requirement["value"])
        if limit:
            return self.limit_marks([("", value)], limit)
        return self.value_marks([e for e in re.split(r"[,;/|]+", value) if e.strip()], requirement["value"])

    def literal_check(self, key: str, requirement: dict[str, Any], exclude: bool = False) -> tuple[str, int]:
        """What the listing's own words settle about a requirement, read as the grader reads them: ('y' | 'n' | '?', grade).
        Grade 3: a fact of the property states the value; 2: a fact of the property contains it; 1: the title states it and
        no fact of the property speaks; 0: nothing settled. A fact of the property naming another value is '-' (open: the
        model says whether the words mean the same, as 'full sleeve' does for 'long sleeves'), except a bare measure
        against a word, which is 'n' (in race 13 'pile height: 0.6 inches' failed 'high'); a figure outside a limit is 'n';
        a listing silent on the property is '?' and left to the model. A value marked "anywhere" (a model code an
        accessory must list) is met wherever the listing writes it."""
        if not self.sheets.get(key):
            return ("?", 0)
        prop, value = requirement["property"], requirement["value"]
        options = self.alternatives(value) or [value]
        if all(self.codes_in(o) and len(self.plain(o).split()) <= 3 for o in options):
            fits = self.fits_codes(key, options)
            if fits is not None:
                return fits                          # the models the listing says it fits decide, wildcards and exceptions read
        if requirement.get("anywhere"):
            text = self.plain(json.dumps([self.items.get(key, {}), self.sheets.get(key, {})], ensure_ascii=False)).replace(" ", "")
            hit = any(len(self.plain(o).replace(" ", "")) >= 3 and self.plain(o).replace(" ", "") in text for o in options)
            return ("y", 2) if hit else ("?", 0)
        facts = self.property_facts(key, prop)
        limit = self.limit_of(value)
        if limit and not exclude:
            return self.limit_marks(facts, limit) if facts else ("?", 0)
        entries = [e for _, v in facts for e in re.split(r"[,;/|]+", v) if e.strip()]
        mark, grade = self.value_marks(entries, value)
        if exclude:
            if facts:
                return ("n", 0) if mark == "y" else ("y", 3)
            return ("n", 0) if self.value_marks([self.title_of(key)], value)[0] == "y" else ("?", 0)
        if facts:
            if mark == "n" and all(self.numbers(e) and not re.search(r"[a-z]{4,}", self.plain(e).replace("inch", "")) for e in entries):
                return ("n", 0)                       # a bare measure ('0.6 inches') is never the word required ('high')
            if mark == "n" and self.peers_write(key, prop, value):
                return ("n", 0)                       # other listings write the value itself in this field: another word is another value
            return (mark, grade) if mark == "y" else ("-", 0)   # another value: open, since the words may mean the same thing
        titled, _ = self.value_marks([self.title_of(key)], value)
        if titled == "y" and self.peers_carry(key, prop) >= 2:
            return ("?", 0)                           # the kind's listings carry a field for it and this one does not: a title is no fact of it
        return ("y", 1) if titled == "y" else ("?", 0)

    def fits_codes(self, key: str, options: list[str]) -> tuple[str, int] | None:
        """Whether a listing's text says it fits one of the model codes wanted: ('y', 2) when it names one, or a wildcard
        form covers one ('ROT31xx' covers ROT3110) and no exception names it; ('n', 0) when an exception names it, or the
        text names codes of the same family and none of them is one wanted; None when the text names no such codes."""
        def tight(code: str) -> str:
            return re.sub(r"[^a-z0-9]", "", code.lower())
        wanted = [tight(o) for o in options if tight(o)]
        record = self.items.get(key, {})
        text = " ".join(str(v) for v in list(record.values()) + list((self.sheets.get(key) or {}).values()) if isinstance(v, (str, int, float)))
        named = False
        family = False
        for sentence in re.split(r"(?<=[.!?])\s+|\n+| \| ", text):
            codes = self.codes_in(sentence)
            if not codes:
                continue
            cut = re.search(r"\b(?:except|excluding|exclude|but not|other than|not for|not compatible|does not fit|doesn't fit|won't fit)\b", sentence, re.I)
            barred = {tight(c) for c in self.codes_in(sentence[cut.end():])} if cut else set()
            for code in codes:
                c = tight(code)
                if c in barred:
                    if c in wanted:
                        return ("n", 0)
                    continue
                if any(c[:3] == w[:3] or re.sub(r"\d", "", c)[:3] == re.sub(r"\d", "", w)[:3] for w in wanted):
                    family = True
                if c in wanted:
                    named = True
                elif "x" in c[2:]:
                    pattern = re.sub(r"x", r"[0-9a-z]", c[:2]) if False else c[:2] + re.sub(r"x", "[0-9a-z]", c[2:])
                    if any(re.fullmatch(pattern, w) and w not in barred for w in wanted):
                        named = True
        if named:
            return ("y", 2)
        return ("n", 0) if family else None

    def peer_keys(self, key: str) -> list[str]:
        """The other read listings of the same category (or of the wanted kind when categories are unknown)."""
        own = self.category_of(key) if self.keys.get("category") else None
        return [k for k in self.sheets if k != key and self.sheets.get(k) and (own is None or self.category_of(k) == own)]

    def peers_carry(self, key: str, prop: str) -> int:
        """How many other read listings of the kind carry a fact of the property."""
        return sum(1 for k in self.peer_keys(key) if self.property_facts(k, prop))

    def peers_write(self, key: str, prop: str, value: str) -> bool:
        """Whether another read listing of the kind writes the required value itself in its fact of the property: then the
        grader takes the field's words as they are, and a synonym on this listing is another value."""
        for k in self.peer_keys(key):
            entries = [e for _, v in self.property_facts(k, prop) for e in re.split(r"[,;/|]+", v) if e.strip()]
            if entries and self.value_marks(entries, value) == ("y", 3):
                return True
        return False

    def literal_mark(self, key: str, requirement: dict[str, str], exclude: bool = False) -> str | None:
        """The old name: 'y' / 'n' when the letter settles it, '?' when the listing is silent, None when no datasheet is read."""
        if not self.sheets.get(key):
            return None
        return self.literal_check(key, requirement, exclude)[0]

    def unmentioned(self, key: str, requirement: dict[str, str]) -> bool:
        """True when no word of the required value, nor of any alternative it names, appears anywhere in the listing (a
        value with figures is left to the judge: limits and ranges need reading)."""
        if re.search(r"\d", requirement["value"]):
            return False
        text = set(words(json.dumps([self.items.get(key, {}), self.sheets.get(key, {})], ensure_ascii=False)))
        for option in self.alternatives(requirement["value"]) or [requirement["value"]]:
            wanted = [w for w in words(option) if len(w) > 2]
            if not wanted or any(w in text or w.rstrip("s") in text or w + "s" in text for w in wanted):
                return False
        return True

    def fitting(self, settled: bool = True) -> list[str]:
        want = (True,) if settled else (True, None)
        return [k for k in self.open_listings() if (self.verdicts.get(k) or {}).get("stamp") == self.stamp() and (self.verdicts.get(k) or {}).get("ok") in want]

    def examine(self) -> None:
        """Read and judge open listings, the likeliest and cheapest first, until a fitting one is known - two before the
        first order when steps allow, since the first pick often sells out at the checkout (26 of 30 practice tasks) and
        the re-plan then needs a fit at hand. Before the first finish attempt a known fit keeps FIRST_ROOM steps
        unspent, so the re-plan after a refusal has room to read."""
        reader = self.roles.get("datasheet") or self.roles.get("listing") or self.roles.get("compare")
        rounds = 0
        while self.time_left() > 40 and rounds < 5:
            rounds += 1
            first = not self.attempted and not self.refusals and bool(self.fitting())
            per_step = self.compare_size() if self.roles.get("compare") else 1
            later = len(self.wants) - 1 - self.want_at            # wants still to be examined after this one
            spare = (self.steps_left() - (FIRST_ROOM if first else FINISH_ROOM + 5) - 4 * later) * per_step
            pool = self.open_listings()
            self.screen_rows(pool[:60])
            cheapest = self.wants_cheapest()

            def rank(k: str) -> tuple:
                price = self.price(k) if self.price(k) is not None else 1e18
                named = self.mentions(k) > 0            # the row itself writes a required value: read before the screen's guesses
                if cheapest:
                    # the cheapest fit is wanted: read in price order, so nothing cheaper is left unjudged; the screen's dismissals last
                    return (1 if self.screen.get(k) == "unlikely" else 0, price)
                return (0 if named else 1, 1 if self.screen.get(k) == "unlikely" else 0, -self.mentions(k), 0 if k in self.valued else 1,
                        0 if self.screen.get(k) == "likely" else 1, price)
            order = sorted((k for k in self.open_listings() if (self.verdicts.get(k) or {}).get("stamp") != self.stamp()), key=rank)
            batch: list[str] = []
            limit = 3 if first else JUDGE_BATCH             # the insurance round reads a few more, not another five
            for key in order:
                if len(batch) >= limit:
                    break
                if reader and key not in self.sheets:
                    if spare <= 0:
                        continue            # an unread sheet costs a step (a share of one side by side); read ones are judged again for free
                    spare -= 1
                batch.append(key)
            if not batch:
                return
            self.read_sheets(batch)
            self.judge(batch)
            for key in sorted(batch, key=lambda k: (0 if (self.verdicts.get(k) or {}).get("ok") else 1, self.price(k) if self.price(k) is not None else 1e18))[:2]:
                if (self.verdicts.get(key) or {}).get("ok") is not False:
                    self.apply_deferrals(key)       # a page that sends a required property elsewhere decides by that page
            fits = self.fitting()
            strong = [k for k in fits if self.strength(k) >= 2]
            if fits and not strong and spare > 0 and rounds < 4:
                continue                            # only weak fits (title or inference): read on for one whose facts state the values
            if fits and self.wants_ranking() and spare > 0 and rounds < 4 and len(self.wants) == 1 and not self.attempted:
                continue                            # the best by a measure: the shelf is read while steps allow, then ranked
            if fits and (not self.wants_cheapest() or self.cheapest_known(fits[0])):
                if len(fits) >= 2 or self.attempted or self.refusals or len(self.wants) > 1:
                    return
                # one fit known before the first order: another round (3 reads at most) while steps beyond FIRST_ROOM remain

    def mentions(self, key: str) -> int:
        """How many requirement values the listing's row writes (a cheap sign of a fit before its datasheet is read)."""
        text = " ".join(str(v) for v in self.items.get(key, {}).values() if isinstance(v, str)).lower()
        return sum(1 for r in self.requirements if all(w in words(text) for w in words(r["value"])))

    def wants_ranking(self) -> bool:
        """Whether the request ranks the fitting listings by something other than price ('the most powerful')."""
        priority = str(self.plan.get("priority") or "").lower().strip()
        return bool(priority) and not priority.startswith(("cheap", "none", "least expensive", "lowest price"))

    def wants_cheapest(self) -> bool:
        return str(self.plan.get("priority") or "").lower().startswith("cheap")

    def cheapest_known(self, best: str) -> bool:
        """True when no open listing cheaper than `best` is left unjudged."""
        price = self.price(best) or 0.0
        return not any((self.verdicts.get(k) or {}).get("stamp") != self.stamp() and (self.price(k) or 0.0) < price for k in self.open_listings())

    # ---- finishing -------------------------------------------------------------------------------------------------
    def evidence(self, key: str, requirement: dict[str, str]) -> int:
        """The grade of a listing's evidence for a requirement: 3 a fact of the property states it, 2 a fact of the property
        contains it, 1 the title (or a model's reading of words the listing carries), 0 a model's inference alone."""
        verdict = self.verdicts.get(key) or {}
        if verdict.get("stamp") == self.stamp():
            for at, known in enumerate(self.requirements):
                if known is requirement or known == requirement:
                    grades = verdict.get("grades") or []
                    return int(grades[at]) if at < len(grades) else 0
        mark, grade = self.literal_check(key, requirement)
        return grade if mark == "y" else 0

    def strength(self, key: str) -> int:
        return min((self.evidence(key, r) for r in self.requirements), default=3)

    def choose(self) -> str | None:
        fits = self.fitting()
        if not fits:
            return None
        if self.wants_cheapest():
            strong = [k for k in fits if self.strength(k) >= 2]
            return (strong or fits)[0]        # the cheapest whose facts state every value; a weaker reading only when none does
        fits = sorted(fits, key=lambda k: (-self.strength(k), self.price(k) if self.price(k) is not None else 1e18))
        priority = str(self.plan.get("priority") or "").lower()
        if priority and not priority.startswith(("cheap", "none")) and len(fits) > 1:
            text = (f"Shopper's ranking: {priority}\nPick the best listing by it from its facts. Reply JSON {{\"id\": \"<id>\"}}.\n"
                    + "\n".join(json.dumps({"id": k, **self.card(k)}, ensure_ascii=False)[:1500] for k in fits[:self.fits_most]))
            answer = self.models.ask("You rank store listings by what the shopper asked for.", text, strong=True, max_tokens=400)
            pick = str((answer or {}).get("id") or "") if isinstance(answer, dict) else ""
            if pick in fits:
                return pick
        return fits[0]          # the plainest fitting listing, the cheapest among equals

    def likeliest_unsettled(self) -> str | None:
        """Among open listings the judge left unsettled (no requirement contradicted, some not spoken of), the cheapest
        whose text still carries every required value somewhere (the judge was unsure, not the listing silent); None
        when there is no such listing, and then a decline stands."""
        unsettled = [k for k in self.fitting(settled=False) if (self.verdicts.get(k) or {}).get("ok") is None and k not in self.order_tried
                     and not any(self.unmentioned(k, r) for r in self.requirements)]
        if not unsettled:
            return None
        return min(unsettled, key=lambda k: (-self.mentions(k), self.price(k) if self.price(k) is not None else 1e18))

    def cart_lines(self, observation: Any) -> dict[str, int] | None:
        if not isinstance(observation, (dict, list)):
            return None
        lines: dict[str, int] = {}
        for record in self.dicts_in(observation):
            key = record.get(self.keys.get("id", "")) if self.keys.get("id") else None
            if not isinstance(key, str):
                continue
            # the line's count: an integer field named for a count, else the one integer field that is not the price
            counts = {k: v for k, v in record.items() if isinstance(v, int) and not isinstance(v, bool) and k != self.keys.get("price") and v != self.price(key)}
            named = [v for k, v in counts.items() if re.search(r"qty|quant|count|units|number|how|total|pieces|^n$|num", k.lower())]
            count = named[0] if named else (next(iter(counts.values())) if len(counts) == 1 else 1)
            lines[key] = int(count) if 0 < int(count) <= 99 else 1
        return lines

    def line_fields(self, role: str) -> tuple[str, str | None] | None:
        """When the cart tool's id argument takes a list of line objects: (id field, quantity field) of a line, from the
        schema of its items (the model's naming as a fallback); None when the argument takes one product id."""
        spec = self.roles.get(role) or {}
        rule = self.params(str(spec.get("tool") or "")).get(str(spec.get("id_arg") or "")) or {}
        if rule.get("type") != "array":
            return None
        props = ((rule.get("items") or {}).get("properties") or {})
        id_field = next((n for n, r in props.items() if (r or {}).get("type", "string") == "string" and re.search(r"id|product|item|listing|sku|ref|number|which", n.lower())), None)
        qty_field = next((n for n, r in props.items() if (r or {}).get("type") in ("integer", "number") and re.search(r"count|qty|quant|unit|amount|how|number|pieces", n.lower())), None)
        id_field = id_field or (spec.get("line_id_field") if spec.get("line_id_field") in props or not props else None) or next(iter(props), None)
        qty_field = qty_field or (spec.get("line_quantity_field") if spec.get("line_quantity_field") in props else None)
        return (id_field, qty_field) if id_field else ("id", "quantity")

    def change_cart(self, role: str, key: str, count: int) -> Any:
        spec = self.roles.get(role)
        if role == "cart_remove" and not spec:
            # no removing tool: a cart tool that takes a quantity may set a line's count, so the line is set to nothing
            add = self.roles.get("cart_add") or {}
            if add.get("quantity_arg") and not self.line_fields("cart_add"):
                args = dict(add.get("fixed") or {})
                args[add["id_arg"]] = key
                args[add["quantity_arg"]] = 0
                observation = self.call(add["tool"], args)
                self.events(observation)
                return observation
            return None
        if not spec or count <= 0:
            return None
        if self.line_fields(role):
            # a list-taking cart tool is given the whole basket wanted (it may set the basket rather than add to it)
            wanted = {k: v for k, v in self.cart.items()}
            if role == "cart_remove":
                left = wanted.get(key, 0) - count
                wanted.pop(key, None)
                if left > 0:
                    wanted[key] = left
            else:
                wanted[key] = wanted.get(key, 0) + count
            return self.set_cart(wanted)
        args = dict(spec.get("fixed") or {})
        args[spec["id_arg"]] = key
        if spec.get("quantity_arg"):
            args[spec["quantity_arg"]] = count
        observation = self.call(spec["tool"], args)
        self.events(observation)
        if role == "cart_remove":
            self.dropped.add(key)
        else:
            self.carted.add(key)
            self.dropped.discard(key)
        return observation

    def set_cart(self, wanted: dict[str, int]) -> Any:
        """The whole basket in one call of the list-taking cart tool."""
        spec = self.roles.get("cart_add") or {}
        fields = self.line_fields("cart_add")
        if not spec or not fields:
            return None
        id_field, qty_field = fields
        lines = []
        for key, count in wanted.items():
            line: dict[str, Any] = {id_field: key}
            if qty_field:
                line[qty_field] = int(count)
            lines.append(line)
        args = dict(spec.get("fixed") or {})
        args[spec["id_arg"]] = lines
        observation = self.call(spec["tool"], args)
        self.events(observation)
        if not self.refused(observation):
            self.cart = dict(wanted)
        return observation

    def basket_wanted(self) -> dict[str, int]:
        return {w["choice"]: w["quantity"] for w in self.wants if w.get("choice")}

    def basket_total(self) -> float | None:
        prices = [self.price(k) for k in self.basket_wanted()]
        return None if any(p is None for p in prices) else sum(p * n for p, n in zip(prices, self.basket_wanted().values()))

    def fill_cart(self) -> bool:
        """Put every want's chosen listing in the cart in its quantity, then tidy the cart against what the store shows:
        extra lines out, short lines topped up. False when a line could not be carted (it is marked gone)."""
        wanted = self.basket_wanted()
        if self.line_fields("cart_add"):
            if self.cart != wanted:
                observation = self.set_cart(wanted)
                if self.refused(observation):
                    self.note(f"cart refused: {json.dumps(observation)[:300]}")
                    for key in wanted:
                        self.gone.add(key)
                    return False
        else:
            for key, count in list(self.cart.items()):
                if key not in wanted and self.steps_left() > 2:
                    self.change_cart("cart_remove", key, count)      # a line no longer wanted goes first: it must not be ordered
                    self.cart.pop(key, None)
            for key, count in wanted.items():
                have = self.cart.get(key, 0)
                if have >= count:
                    continue
                observation = self.change_cart("cart_add", key, count - have)
                if self.refused(observation):
                    self.note(f"cart add refused for {key}: {json.dumps(observation)[:300]}")
                    self.gone.add(key)
                    return False
                self.cart[key] = count
        view = self.roles.get("cart_view")
        if view and self.steps_left() > 4 + 2 * len(wanted):
            observation = self.call(view["tool"], dict(view.get("fixed") or {}))
            self.events(observation)
            lines = self.cart_lines(observation)
            if lines is not None:
                self.cart = lines
        wanted = self.basket_wanted()
        if self.cart != wanted and self.steps_left() > 2:
            if self.line_fields("cart_add"):
                self.set_cart(wanted)
            else:
                for key, count in list(self.cart.items()):
                    if key not in wanted and self.steps_left() > 1:
                        self.change_cart("cart_remove", key, count)
                        self.cart.pop(key, None)
                for key, count in wanted.items():
                    have = self.cart.get(key, 0)
                    if have > count and self.steps_left() > 1:
                        self.change_cart("cart_remove", key, have - count)
                        self.cart[key] = count
                    elif have < count and self.steps_left() > 1:
                        self.change_cart("cart_add", key, count - have)
                        self.cart[key] = count
        total = self.basket_total()
        if self.budget is not None and total is not None and total > self.budget + 1e-9:
            self.note(f"the basket now costs {total}, over the budget {self.budget}: chosen again")
            return False
        return all(key not in self.gone for key in wanted)

    @staticmethod
    def refused(observation: Any) -> bool:
        if observation is None:
            return True
        text = json.dumps(observation).lower()
        return bool(re.search(r'"(?:error|issue|problem|fault)"', text)) or bool(re.search(r"not found|invalid|unknown|refus|denied|exceed", text))

    def shopper_lines_in(self, observation: Any) -> bool:
        """Whether a result carries shopper lines (a refusal with reasons rather than a store fault)."""
        if isinstance(observation, str):
            return len(observation.split()) >= 4
        skip = {id(row) for row in self.product_rows(observation)}
        return bool(self.sentences(observation, skip))

    def checkout(self) -> bool:
        spec = self.roles.get("checkout")
        if not spec:
            return False
        args = dict(spec.get("fixed") or {})
        tag = json.dumps(args, sort_keys=True)
        if self.checkout_faults >= 2:
            return False
        observation = self.call(spec["tool"], args)
        if self.finished:
            return True
        self.refusals += 1
        if not self.shopper_lines_in(observation) or EMPTY_WORDS.search(json.dumps(observation)):
            self.checkout_faults += 1                   # a fault (an empty cart, a bad argument), not the shopper's refusal
            self.finish_faults.add(tag)
            self.cart = {}                              # whatever the store holds, it is not what was meant: carted afresh
            if EMPTY_WORDS.search(json.dumps(observation)):
                self.note("the store says the basket is empty: carted again with explicit counts")
                return False
        lines = self.events(observation)
        self.note(f"checkout refused: {json.dumps(observation)[:400]} lines={lines}")
        if lines and self.steps_left() > FINISH_ROOM + len(self.cart):
            self.rechecked = set()                      # after the shopper's change the carted listings are read again
        view = self.roles.get("cart_view")
        if lines and view and self.steps_left() > 3 + 2 * max(1, len(self.cart)):
            # the refusal mostly concerns the carted listing's price or stock: the cart shows what they are now
            shown = self.call(view["tool"], dict(view.get("fixed") or {}))
            self.events(shown)
            held = self.cart_lines(shown)
            if held is not None:
                self.cart = held
        return False

    def decline(self) -> bool:
        """End without buying; False when the store refused (shopper lines came with the refusal and are applied)."""
        spec = self.roles.get("decline")
        if not spec:
            return False
        args = dict(spec.get("fixed") or {})
        if spec.get("remark_arg"):
            args[spec["remark_arg"]] = "Sorry, nothing available right now meets everything you asked for, so I am not buying anything."
        tag = json.dumps(args, sort_keys=True)
        if tag in self.finish_faults:
            # the store faulted this very call before: the outcome enum's other values are tried once each, then nothing
            props = self.params(spec["tool"])
            tried = {json.loads(t).get(a) for t in self.finish_faults for a in props}
            for arg, rule in props.items():
                for value in (rule.get("enum") or []):
                    if value not in tried and json.dumps({**args, arg: value}, sort_keys=True) not in self.finish_faults:
                        args = {**args, arg: value}
                        tag = json.dumps(args, sort_keys=True)
                        break
                else:
                    continue
                break
            else:
                return False
        observation = self.call(spec["tool"], args)
        if not self.finished:
            self.refusals += 1
            if not self.shopper_lines_in(observation):
                self.finish_faults.add(tag)
            self.note(f"decline refused: {json.dumps(observation)[:300]}")
        return self.finished

    def want_index(self, text: Any, exclude: int | None = None) -> int | None:
        """Which want some words name: the one whose item is exactly those words, else by the words of its item or
        category; never `exclude` (the item asking); None when no single one does."""
        if not isinstance(text, str) or not words(text):
            return None
        said = set(words(text))
        exact = [at for at, want in enumerate(self.wants) if at != exclude and set(words(str(want["plan"].get("item") or ""))) == said]
        if len(exact) == 1:
            return exact[0]
        scored = []
        for at, want in enumerate(self.wants):
            if at == exclude:
                continue
            own = set(words(str(want["plan"].get("item") or ""))) | set(words(str(want["plan"].get("category") or ""))) | {w for c in want["categories"] for w in words(c)}
            scored.append((len(said & own), -len(own - said), at))
        if not scored:
            return None
        best = max(scored)
        return best[2] if best[0] > 0 and [sc[:2] for sc in scored].count(best[:2]) == 1 else None

    def identity_tokens(self, key: str) -> list[str]:
        """What another listing would name to say it suits this one: the code-like names in its description (its model
        number and any other name it is sold under, 'MOZ-7460', 'Drane 746'), its model facts and the code-like words of
        its title ('VIR3110', 'C3-00', 'G710N')."""
        tokens: list[str] = []
        record = self.items.get(key, {})
        texts = [str(v) for k, v in record.items() if isinstance(v, str) and re.search(r"desc|blurb|about|summary|overview|detail|text", str(k), re.I)]
        for text in texts:
            tokens += self.codes_in(text[:300])
        for label, value in (self.sheets.get(key) or {}).items():
            if re.search(r"model|part number|mpn|series", self.core_label(label)) and isinstance(value, str):
                tokens += [p.strip() for p in re.split(r"[,;/|]+", value) if 2 <= len(p.strip()) <= 30]
        tokens += re.findall(r"\b(?=[A-Za-z0-9-]*\d)[A-Z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)*\b", self.title_of(key))
        seen: list[str] = []
        for token in tokens:
            if token.lower() not in [s.lower() for s in seen] and not re.fullmatch(r"\d+[a-zA-Z]{0,2}", token):
                seen.append(token)
        return seen[:4]

    @staticmethod
    def codes_in(text: str) -> list[str]:
        """Model-code-like names in a text: letters joined to three or more figures ('ROT3110', 'MOZ-7460', 'Drane 746'),
        wildcard forms included ('ROT31xx')."""
        found = re.findall(r"\b[A-Za-z]{2,12}[- ]?\d{2,6}(?:[xX]{1,3}|[A-Za-z]?)\b", str(text))
        return [f for f in found if len(re.sub(r"\D", "", f)) >= 2 and len(re.sub(r"[^A-Za-z]", "", f)) >= 2 and sum(ch.isdigit() for ch in f) + f.lower().count("x") >= 3]

    def apply_links(self, at: int) -> None:
        """A want whose property must suit another item of the request (a charger for the laptop chosen) gets that item's
        identity tokens as a requirement checked against the models a listing says it fits, and a search for them."""
        want = self.wants[at]
        for link in want.get("linked") or []:
            other = self.want_index(str(link.get("to_item") or ""), exclude=at)
            if other is None or other == at or not self.wants[other].get("choice"):
                continue
            tokens = self.identity_tokens(self.wants[other]["choice"])
            prop = str(link.get("property") or "").strip()
            if not tokens or not prop:
                continue
            value = " or ".join(tokens)
            want["requirements"] = [r for r in want["requirements"] if not self.same_property(r["property"], prop)] + [{"property": prop, "value": value, "anywhere": True}]
            if want.get("choice") and self.literal_check(want["choice"], want["requirements"][-1])[0] != "y":
                want["choice"] = None                 # the pick made before the device was known does not fit it
            self.note(f"{want['plan'].get('item')!r} must suit {self.wants[other]['plan'].get('item')!r}: {prop} -> {tokens}")
            current = self.want_at
            self.save_want()
            self.load_want(at)
            item_word = " ".join(self.query_terms(str(self.plan.get("item") or ""))[:1])
            for token in tokens[:2]:
                if self.steps_left() > FINISH_ROOM + 8:
                    self.search_all(text=f"{token} {item_word}".strip(), pages=1)
            self.save_want()
            self.load_want(current)

    def choose_want(self, at: int) -> str | None:
        """The listing to order for one want: the settled fit, else after widening, else the likeliest unsettled one."""
        self.apply_links(at)
        self.load_want(at)
        try:
            for _ in range(2):
                self.examine()
                choice = self.choose()
                if choice is not None:
                    return choice
                if not (self.steps_left() > FINISH_ROOM + 7 * len(self.wants) and self.time_left() > 60 and self.widen()):
                    break
            choice = self.likeliest_unsettled()
            if choice is not None:
                # a listing silent on a property is a better bet than a decline, which is seldom the right answer
                self.note(f"no settled fit for {self.plan.get('item')!r}: the likeliest unsettled listing {choice}")
                return choice
            absent = [r for r in self.requirements if self.value_absent(r)]
            if absent and self.say_if_none:
                self.note(f"no listing seen for {self.plan.get('item')!r} carries {absent}, and the shopper asked to be told: declining")
                return None
            choice = self.best_available()
            if choice is not None:
                self.note(f"no fit for {self.plan.get('item')!r}: the closest listing {choice} is ordered (a decline is almost never right)")
            return choice
        finally:
            self.save_want()

    def value_absent(self, requirement: dict[str, str]) -> bool:
        """True when, among at least twelve listings of the wanted kind read or shown, none carries the required value
        anywhere (a figure or limit is never judged absent)."""
        value = requirement["value"]
        if re.search(r"\d", value) or re.search(r"\b(?:at least|at most|up to|under|over|no more|no less|between|than)\b", value, re.I):
            return False
        options = self.alternatives(value) or [value]
        seen = 0
        for key in self.items:
            if self.categories and self.keys.get("category") and self.category_of(key) not in self.categories:
                continue
            seen += 1
            text = json.dumps([self.items.get(key, {}), self.sheets.get(key, {})], ensure_ascii=False)
            if any(self.entry_has(text, o) for o in options):
                return False
        return seen >= 12

    def best_available(self) -> str | None:
        """The closest listing of the wanted kind within the budget and in stock: fewest requirements contradicted, then
        fewest unknown, then the most of the required values in its text, the cheapest among equals. None when no
        listing of the kind is on hand (then a decline stands)."""
        pool = [k for k in self.open_listings() if k not in self.order_tried and k not in self.gone]
        judged = [k for k in pool if (self.verdicts.get(k) or {}).get("kind") in ("y", "?")]
        if not judged:
            judged = [k for k in pool if (self.verdicts.get(k) or {}).get("kind") != "n" and (self.screen.get(k) != "unlikely")
                      and (not self.categories or not self.keys.get("category") or self.category_of(k) in self.categories)]
        if not judged:
            # every open listing was ruled out: the least wrong of the kind still beats a decline, which is almost never right
            judged = [k for k in self.items if k not in self.gone and k not in self.order_tried and self.in_stock(k) is not False
                      and (self.budget is None or self.price(k) is None or self.price(k) * self.quantity <= self.budget + 1e-9)
                      and (not self.categories or not self.keys.get("category") or self.category_of(k) in self.categories)
                      and (self.verdicts.get(k) or {}).get("kind") != "n"]
        if not judged:
            return None

        def score(k: str) -> tuple:
            verdict = self.verdicts.get(k) or {}
            marks = verdict.get("checks") or []
            price = self.price(k) if self.price(k) is not None else 1e18
            return (0 if verdict.get("kind") == "y" else 1, marks.count("n"), marks.count("?"), -self.mentions(k), price)
        return min(judged, key=score)

    def within_budget(self) -> bool:
        """Swap the dearest want's choice for its next cheaper fit until the basket is within the budget; False when
        it cannot be."""
        for _ in range(6):
            total = self.basket_total()
            if self.budget is None or total is None or total <= self.budget + 1e-9:
                return True
            swapped = False
            for at in sorted(range(len(self.wants)), key=lambda i: -(self.price(self.wants[i]["choice"] or "") or 0) * self.wants[i]["quantity"]):
                self.load_want(at)
                current = self.wants[at]["choice"]
                cheaper = [k for k in self.fitting() if k != current and (self.price(k) or 0) < (self.price(current) or 0)]
                self.save_want()
                if cheaper:
                    self.wants[at]["choice"] = cheaper[-1] if self.wants_cheapest() else cheaper[0]
                    swapped = True
                    break
            if not swapped:
                self.note(f"the basket costs {total} against a budget of {self.budget}: no cheaper fits")
                return False
        return False

    def finish(self) -> None:
        """Order a fitting listing for every wanted item, the cheapest where asked, re-planning after any refusal;
        decline when nothing fits one of them."""
        for _ in range(6):
            if self.finished:
                return
            self.drop_gone_lines()
            missing = []
            for at in sorted(range(len(self.wants)), key=lambda i: 1 if self.wants[i].get("linked") else 0):
                choice = self.wants[at].get("choice")
                if choice is None or choice in self.gone or (self.verdicts.get(choice) or {}).get("ok") is False:
                    choice = self.choose_want(at)
                    self.wants[at]["choice"] = choice
                if choice is None:
                    missing.append(at)
                else:
                    self.wants[at]["tried"].append(choice)
            if missing or not self.within_budget():
                self.note("nothing fits " + ", ".join(repr(self.wants[at]["plan"].get("item")) for at in missing) + ": declining" if missing else "declining")
                if self.decline() or self.steps_left() <= 1:
                    return
                continue                # the refusal brought changes (a dropped requirement, a new budget): plan again
            self.attempted = True
            lines = max(1, len([w for w in self.wants if w.get("choice")]))
            for want in self.wants:
                if want.get("choice") and self.steps_left() > 6 + 3 * lines:
                    if want["choice"] not in self.sheets or not self.sheets.get(want["choice"]):
                        self.read_details(want["choice"], room=6 + 3 * lines)  # anything ordered is read on its own first
                    if (self.roles.get("listing") or {}).get("tool") != (self.roles.get("datasheet") or {}).get("tool"):
                        self.read_page(want["choice"], room=6 + 3 * lines)     # and its full page opened where the world has one
            if not self.fill_cart():
                continue
            if any(w.get("choice") in self.gone for w in self.wants):
                continue
            self.recheck_cart()
            if any(w.get("choice") in self.gone for w in self.wants):
                continue
            if not self.confirm_basket():
                continue                # the shopper's answer changed the plan
            if self.checkout():
                return
        if not self.finished and not self.decline() and self.steps_left() > 1:
            self.decline()

    def drop_gone_lines(self) -> None:
        """A carted line the shopper reported gone leaves the cart before anything else is spent on the re-plan: an
        order that still carries it is wrong whatever else is added."""
        if self.line_fields("cart_add"):
            return                                      # a list-taking cart tool is set whole when the basket is filled
        # the store's cart view may no longer list a sold-out line, so the agent's own record of what it carted decides
        for key in [k for k in (self.carted | set(self.cart)) if k in self.gone and k not in self.dropped]:
            if self.steps_left() <= 2:
                break
            self.change_cart("cart_remove", key, self.cart.get(key, 1))
            self.cart.pop(key, None)

    def plan_signature(self) -> str:
        self.save_want()
        return json.dumps([(w["requirements"], w["exclusions"], w.get("quantity")) for w in self.wants], sort_keys=True) + f"|{self.budget}|{sorted(self.gone)}"

    def facts_brief(self, key: str) -> str:
        """Up to three facts of a listing that bear on what was required, for the shopper's check."""
        sheet = self.sheets.get(key) or {}
        said = []
        for requirement in self.requirements:
            for label, value in sheet.items():
                if isinstance(value, (str, int, float)) and not isinstance(value, bool) and self.same_property(self.core_label(label), requirement["property"]):
                    said.append(f"{self.plain_property(self.core_label(label))} {str(value)[:30]}")
                    break
            if len(said) >= 3:
                break
        return f" ({', '.join(said)})" if said else ""

    def confirm_basket(self) -> bool:
        """Put the filled basket to the shopper for a yes before ordering: a request may demand it, and it costs one
        message. True when the answer changed nothing; the same basket is put once."""
        spec = self.roles.get("message")
        tag = json.dumps(sorted(self.cart.items()))
        if not spec or not self.cart or self.messages_sent >= self.message_budget or tag in self.confirmed or self.steps_left() <= 2 or not self.confirm_needed:
            return True
        self.confirmed.add(tag)
        self.messages_sent += 1
        unit = self.currency or ""
        total = self.basket_total()
        cap = (self.params(spec["tool"]).get(spec["text_arg"]) or {}).get("maxLength")
        cap = min(int(cap), 700) if isinstance(cap, int) and cap >= 80 else 700
        text = ""
        for width in (45, 30, 20):
            lines = []
            for want in self.wants:
                key = want.get("choice")
                if key and key in self.cart:
                    price = self.price(key)
                    line = f"{self.cart[key]} x {self.short_title(key, width)}"
                    lines.append(f"{line} at {price:g} {unit}".rstrip() if price is not None else line)
            if not lines:
                return True
            text = "I am ready to order: " + "; ".join(lines)
            if total and len(lines) > 1:
                text += f". Total {total:g} {unit}".rstrip()
            text += ". Shall I place the order?"
            if len(text) <= cap:
                break
        if len(text) > cap:
            text = text[:cap - 30].rsplit(" ", 1)[0] + ". Shall I place the order?"
        before = self.plan_signature()
        self.note(f"basket put to the shopper: {text[:300]}")
        observation = self.call(spec["tool"], {spec["text_arg"]: text})
        self.events(observation, asked=text)
        changed = self.plan_signature() != before
        if changed:
            self.note("the shopper's answer changed the plan")
        return not changed

    def short_title(self, key: str, width: int) -> str:
        """The first words of a listing's title, without shop codes or bracketed tags: plain enough for a busy shopper."""
        title = re.sub(r"\[[^\]]*\]|\([^)]*\)|【[^】]*】|[|/]+", " ", self.title_of(key))
        title = " ".join(title.split())
        if len(title) <= width:
            return title
        cut = title[:width].rsplit(" ", 1)[0]
        return cut if len(cut) >= 8 else title[:width]

    def title_of(self, key: str) -> str:
        """The listing's title: a 'title' field first, then a heading, then a name that is not the seller's or brand's."""
        record = self.items.get(key, {})
        for test in (lambda k: "title" in k, lambda k: "head" in k, lambda k: "name" in k and not re.search(r"sold|seller|brand|shop|vendor|merchant|maker|store|user|buyer", k)):
            name = next((k for k in record if test(k.lower()) and isinstance(record.get(k), str) and record[k].strip()), None)
            if name:
                return str(record[name])[:80]
        return key

    def widen(self) -> bool:
        """More searches when nothing read so far fits: each requirement's value within the category (continuing its
        pages), with the item and alone, the plan's unused queries, and the category in full."""
        item = str(self.plan.get("item") or "").strip()
        before = len(self.items)
        queries = []
        within = self.filtered()
        for requirement in self.requirements:
            for value in self.alternatives(requirement["value"]) or [requirement["value"]]:
                queries += [self.value_query(value), " ".join(self.query_terms(value)[:2] + self.query_terms(item)[:1]), " ".join(self.query_terms(value)[:2])]
        for requirement in self.requirements[:1]:
            base = self.value_query(requirement["value"])
            state = self.page_state.get(json.dumps([base, None, None]).lower())
            if state and state["total"] > state["shown"] * self.pages:
                queries += [self.value_query(requirement["value"], extra=w) for w in self.split_words(base)]
        queries += [q for q in self.plan.get("search_queries") or [] if isinstance(q, str)]
        if self.requirements:
            queries += [" ".join([w] + self.query_terms(item)[:1]) for w in self.split_words(self.value_query(self.requirements[0]["value"]))]
        tried = 0
        for query in queries:
            query = " ".join(query.split())[:60]
            tag = json.dumps([query, within if self.combined_search is not False else None, None]).lower()
            state = self.page_state.get(tag)
            if not query or len(words(query)) > 6 or (state and (not state["more"] or state["page"] >= self.pages)):
                continue
            self.search_all(text=query, category=within, pages=self.pages)
            tried += 1
            if tried >= 4 or self.steps_left() <= FINISH_ROOM + 7:
                break
        if self.categories and (self.roles.get("search") or {}).get("category_arg") and self.steps_left() > FINISH_ROOM + 8:
            self.search_all(category=self.categories[0], cap=self.budget, pages=self.pages)
        return len(self.items) > before and bool([k for k in self.open_listings() if (self.verdicts.get(k) or {}).get("stamp") != self.stamp()])

    # ---- the whole task -------------------------------------------------------------------------------------------
    def run(self) -> list[dict[str, Any]]:
        if self.finished:
            return self.dialogue
        self.map_tools()
        self.read_request()
        if self.opening is not None:
            self.events({"opening": self.opening})
        for at in range(len(self.wants)):
            self.load_want(at)
            self.gather()
            pool = self.open_listings()
            self.screen_rows(pool[:48])
            if self.wants_cheapest():
                first = sorted(self.open_listings(), key=lambda k: (1 if self.screen.get(k) == "unlikely" else 0, self.price(k) if self.price(k) is not None else 1e18))
            else:
                first = sorted(self.open_listings(), key=lambda k: (0 if self.mentions(k) else 1, 0 if self.screen.get(k) == "likely" else 1,
                                                                  0 if k in self.valued else 1, self.price(k) if self.price(k) is not None else 1e18))
            sheets = self.sheets_first * (2 if self.roles.get("compare") else 1)
            if self.steps_left() > FINISH_ROOM + 10 * (len(self.wants) - at):
                self.read_sheets(first[: sheets if at == 0 else max(3, sheets // 2)])    # every item gets questions: each shelf is read enough for a menu
            stated = [json.dumps(r, sort_keys=True) for r in self.requirements]
            self.ask_shopper()
            self.save_want()
            for reference in (self.wants[at].get("references") or [])[:1]:
                if self.steps_left() <= FINISH_ROOM + 12 * (len(self.wants) - at) or self.time_left() < 120:
                    break
                prop = str(reference.get("property") or "").strip()
                value = self.resolve_reference(reference)
                self.note(f"the request's reference for {prop!r} -> {value!r}")
                if value:
                    if reference.get("negated") or str(reference.get("relation") or "").lower().startswith("differ"):
                        self.exclusions.append({"property": prop, "value": value})
                    else:
                        self.requirements = [r for r in self.requirements if not self.same_property(r["property"], prop)]
                        self.requirements.append({"property": prop, "value": value})
            self.save_want()
            for requirement in self.requirements:
                if json.dumps(requirement, sort_keys=True) not in stated and self.steps_left() > FINISH_ROOM + 12:
                    self.search_all(text=self.value_query(requirement["value"]), category=self.filtered(),
                                    pages=self.deep if (self.wants_cheapest() or self.wants_ranking()) and len(self.wants) == 1 and self.steps_left() > 40 else self.pages)
            if [json.dumps(r, sort_keys=True) for r in self.requirements] != stated:
                self.note(f"requirements after the questions: {self.requirements}")
            self.save_want()
        self.finish()
        return self.dialogue


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    agent = Polaris(problem_data)
    try:
        return agent.run()
    except Exception as exc:  # still end the episode if anything breaks
        agent.note(f"agent error: {type(exc).__name__}: {exc}")
        try:
            if not agent.finished and agent.cart and agent.roles.get("checkout") and agent.cart == agent.basket_wanted():
                agent.checkout()
            if not agent.finished:
                agent.decline()
        except Exception as late:
            agent.note(f"agent error at the end: {type(late).__name__}: {late}")
        return agent.dialogue


__all__ = ["agent_main"]
