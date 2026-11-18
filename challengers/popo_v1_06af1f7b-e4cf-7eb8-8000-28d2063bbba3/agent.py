from __future__ import annotations

import difflib
import json
import re
import time
from typing import Any

from src.agent.proxy_client import ProxyClient

LIGHT = "anthropic/claude-haiku-5.5"
HEAVY = "anthropic/claude-sonnet-5.5"
SPARE = "openai/gpt-6-luna"

TASK_SECONDS = 300.0
SAFE_END = 30.0          # seconds kept for the forced finish
NO_HEAVY_AFTER = 225.0   # no strong-model judging after this many seconds
INTERVIEW_BY = 170.0     # no new interview message after this many seconds
SPEND_SOFT = 0.06        # above this, judging calls use the light model
SPEND_HARD = 0.085       # above this, no optional model call
LIST_MAX = 8             # properties named in one message
SETTLED_BY_PLAIN = 1     # a "don't mind" reply settles only the first property (practice: place 1 revealed 77/77, place 2 19/34, place 3 16/44)
OPEN_BEFORE_ASK = 2      # opened listings of an item before its interview starts
ASK_AFTER_SHARE = 4      # or once this share (1/n) of the steps is used
LOOP_TURNS = 40
OBS_CHARS = 7000
DESC_CHARS = 260
HISTORY_KEEP = 14        # tool results kept in full in the walk's context
HOLDS = 2                # times a checkout or a decline can be sent back
AUDIT_OPEN = 3           # listings the decline audit may open
# search and trail limits, Cairn's values
HUNT_BY = 200.0          # no decline search or trail after this many seconds
HUNT_GRACE = 25.0
HUNT_STEPS = 24          # searches a decline hunt may spend
HUNT_KEEP = 10           # steps the hunt leaves untouched
HUNT_FIT_WORDS = 4       # further words searched per item
HUNT_FIT_OPENS = 4       # rows opened per item after the hunt
HUNT_COMMON = 0.6        # a word on more than this share of an item's rows finds nothing new
HUNT_VALUE_CHARS = 30
HUNT_ROUNDS = 2          # value hunts under a cheapest rule
HUNT_IDLE = 4            # searches in a row without a new row end a hunt
HUNT_PAGE_NEW = 5        # a page showing fewer new rows ends the walk of a long search
HUNT_WHOLE = 49          # a search matching more than this is not walked to its end
FIT_PAGES = 2            # further pages of earlier searches read before a decline
TRAIL_ROOM = 15          # steps needed to follow a pointing requirement
EARLY_TRAILS = 3         # chain rounds the early follow may spend
NO_NAME = ('none', 'null', 'n/a', 'unknown')

_llm = ProxyClient(timeout=60, max_retries=1, retry_delay=1)
_env = ProxyClient(timeout=60, max_retries=1, retry_delay=1)

ID_VALUE = re.compile(r"^[A-Za-z]{1,4}[-_][A-Za-z0-9]{5,40}$|^[A-Za-z0-9]{8,40}$")
ID_KEY = re.compile(r"(?:^|_)(?:id|ids|pid|sku|code|number|ref|item|product|listing)s?(?:_|$)", re.I)
SKIP_WORDS = {"id", "ids", "pid", "sku", "ref", "title", "name", "headline", "description", "details", "overview",
              "blurb", "comments", "pointers", "text", "summary", "price", "cost", "amount", "stock", "hand", "buy",
              "available", "availability", "image", "url", "link", "rating", "reviews", "review", "score",
              "category", "section", "department", "group", "shelf", "seller", "vendor", "shop", "store", "company",
              "brand", "maker", "sold", "from", "qty", "quantity", "count", "line", "total"}
NOT_ASK = re.compile(r"\b(?:id|sku|model|series|code|title|name|description|price|stock|image|url|rating|review|"
                     r"seller|vendor|shop|store|barcode|ean|upc|asin|options?|variants?)\b")
STOCK_WORDS = re.compile(r"stock|on_hand|available|can_buy|buyable|in_store", re.I)
PRICE_WORDS = re.compile(r"price|cost|amount|owed|total", re.I)
CROWD_WORDS = re.compile(r"\b(?:one (?:thing|question)s? at a time|one at a time|too many|single question|"
                         r"one by one|separately|several things|so many|a lot at once|all at once)\b", re.I)
CONFUSED_WORDS = re.compile(r"\b(?:didn'?t follow|don'?t follow|not sure what you mean|lost me|put it differently|"
                            r"rephrase|didn'?t catch|confus|over my head|say that again|"
                            r"what do you mean|simpler words)\b", re.I)
NOPREF_WORDS = re.compile(r"\b(?:don'?t mind|no preference|whatever|anything|indifferent|"
                          r"no opinion|either way|doesn'?t matter)\b", re.I)

READ_RULES = (
    "Reading rules. A requirement on a property is met when the listing's own attribute for that property states the "
    "value: key spelling does not matter (spaces, underscores, camel case, prefixes), one element of a comma or slash "
    "list is enough, case, singular or plural and plain synonyms count, and a value that holds the wanted value's "
    "words counts (\"navy blue\" meets blue, \"leather boots\" meets leather). A different value does not count, nor does "
    "an abbreviation or a broader or neighbouring class (kids is not adults). Keys that name the same property decide, "
    "and a key that adds the product's own name to the property (a lamp's \"lamp shape\") names it too: when one of "
    "them states the wanted value the requirement is met, even if another of them says something else; when they "
    "state only other values the listing fails, whatever the title says. A key for a different property decides "
    "nothing. A listing with no attribute for the property does not meet it: words in its title or description are "
    "not enough. A value that is exactly the wanted one beats one that only holds its words (\"matte\" beats \"matte and glossy\"). Numbers are compared after converting units; limits include the edge; a "
    "range must lie wholly inside a limit. The listing's category label must be the requested kind of product, and "
    "a requirement that the category label itself states (\"organic\" for a category named Organic "
    "Tea) is met by that label. Price "
    "times quantity must be within the budget, and a listing shown out of stock does not fit when stock matters. A "
    "listing the store shows in stock is on hand: that is all a request for something on the shelf, not just listed "
    "online, asks."
)

SHEET = """You prepare a shopping task for an agent. Read the shopper's request and the store's tools. Return ONE compact JSON object and nothing else, with these keys:
"items": one entry per distinct product to buy: {"name": the product in the request's words, "label": the store category the request names for it, or "", "quantity": units of that one listing (1 when not stated; a pair or a set sold as one listing is 1), "musts": firm requirements on that product, each "property: constraint" with the exact value, limit, exclusion or closed list, named by the request's own word for the property}. Price, budget, stock and quantity are never musts. Words that are part of the product's own name (a season or audience named with the product) are not musts. A requirement whose value is given by pointing at another listing (its value is whatever another shop's listing shows) is ONE must holding the whole description; the words describing that other listing are not musts of this product. A product named in one phrase with one quantity is one item.
"budget": the most the shopper pays in total, as a number in the request's unit, or null. "budget_each": true when the limit is per unit.
"in_stock": true unless the request accepts items that are not available now.
"pick": "cheapest", "most" or "least" when the request asks for the lowest price or for the extreme of a measure, else "any". "measure": what is maximised or minimised, else "".
"messages": how many messages the agent may send the shopper (integer) or null. "steps": how many steps or moves the request allows (integer) or null.
"wishes": things the shopper would merely like (ideally, if possible).
"ask_first": things the request says the agent must check with the shopper before choosing.
"steps_required": every procedural step the request asks for besides finding and buying (getting approval, telling or showing the shopper something, checking or reading something, using a certain tool, how many listings to look at), each {"step": what must be done in at most 18 words, "when": "choose", "order" (just before checking out) or "end" (just before ending without a purchase), "messages": messages it takes (0 when none)}. Requirements, budgets, limits and how to decline are not steps, and neither is a remark that the shopper will verify something themselves after the purchase.
"decline_rule": the request's words about how to end when nothing fits, or "". "quiet_decline": true when it says not to suggest or name alternatives, else false.
"roles": the exact tool name for each job, or null: "search" (finds listings), "open" (shows one listing's attributes or properties; when two single-listing tools exist, the one that promises attributes, properties or specifications), "offer" (another single-listing tool, showing price and stock), "compare" (shows several listings at once), "ask" (messages the shopper), "cart" (changes the cart), "cart_view" (shows the cart), "buy" (checks out), "refuse" (ends without buying).
"buy_args": fixed arguments that make the buy tool check out (for a tool with a mode argument, that mode), else {}. "refuse_args": the same for ending without a purchase, else {}.
"lookups": the musts (copied exactly) whose value is given by pointing at another listing, else [].
"search_field": the search tool's parameter that takes free words, or "". "search_args": fixed arguments that make it search by words (a mode), else {}."""

WALK = """You are Compass, a careful buyer working for one shopper in an online store. You act only through the store's tools. The episode scores 1 only when the finish is exactly right: every requested item at its quantity, each meeting every requirement in force at checkout (stated, revealed by the shopper, or changed while you work), within the budget, the cheapest or the extreme one when the request asks for it, and every procedural step the request asks for done. When nothing in the store fits, the right finish is ending without a purchase, in the way the request asks.

THE STORE
- Every tool call is one step; steps are limited. Thinking is free. Read each tool's description and schema and use the exact argument names and allowed values.
- Search: short queries work best (the wanted value plus the product noun). Some stores need every word to match, cap the rows per page, or page by a token. Compare the reported total with the rows you saw and page on when the answer depends on seeing them all. A filter that returns nothing usually has the wrong value or parameter: retry with a label exactly as rows show it.
- Search rows are summaries. Read a listing's attributes with the single-listing or side-by-side tool before you rely on it, and never open the same listing twice without a reason.
- You never message the shopper yourself: Compass's interviewer asks the questions, and the answers appear in the conversation and the ledger. Procedural steps the request asks for just before checking out or ending (an approval, telling the shopper something) are carried out by Compass when you call the checkout or ending tool.
- Shopper or store notes can appear inside any tool result (a new budget, a dropped or new requirement, a sold-out listing). They are binding from then on; the ledger shows them.

WHAT FITS
- """ + READ_RULES + """
- A requirement can point at another listing ("the same X as the listing that shop Y sells with Z"). Find that listing, open its page, and read the value there; when that page sends you to a further listing, follow it. Only then judge candidates.
- When one requested item must fit or suit another, choose them together from the listings' own words.

CHEAPEST, MOST, LEAST
- The order must be the best of ALL fitting listings on that rule. Use price filters or sorting when a tool offers them, page through the fitting set, open the leading candidates in order, and buy the first that fully fits after every better one was checked and rejected.

FINISHING
- Put exactly the requested items at their quantities in the cart and nothing else, then check out. Keep a backup in mind: a pick can sell out at checkout. If a checkout is refused, read why, fix the cart and check out again.
- End without a purchase only after a thorough search finds nothing that fits. If the request says not to suggest alternatives, the ending remark names no listing.
- Keep each reply short: what you know, what you do next. Every reply must call at least one tool. You may call several tools in one reply when none depends on another."""

PLAN = """You choose the properties a shopping agent asks the shopper about. The shopper holds a few firm preferences they did not write down, each about one property of one product. A question reveals one only when it names that exact property; the shopper mostly answers the FIRST property in a message that holds a preference, so order matters most.

You get the request, the items with what the request already settles, and for each item the attribute keys seen on its opened listings, with how many listings carry each key and example values.

Rank, for each item:
1. What the request says must be checked with the shopper (ASK FIRST), if any.
2. Properties the request's story hints at without giving a value: who will use or wear it, how it must look, where or how it will be used, what it must go with, something the shopper says they leave to the agent or cannot judge. Name each hinted property by the item's listed key that fits it.
3. Then the listed keys a buyer of that product chooses between (its type or style, form, fit, size or capacity, material, colour, pattern, features), favouring keys most listings carry and whose values differ between listings.
Skip properties the request already settles (a must names them), second names of a property already chosen, identifiers, model or series names, warranty and other logistics. A key seen on only one listing goes last.
"label" is the property's plain name with all of the key's words ("lid colour" for lidColour), and nothing else.
Return ONE JSON object and nothing else: {"ask": [{"item": item number, "label": "..."}]} with up to COUNT labels per item, each item's labels best first."""

HEAR = """A shopping agent sent the shopper a message naming some properties of a product, and the shopper replied. Return ONE JSON object and nothing else:
{"needs": ["property: constraint", ...], "points": [the needs (copied exactly) whose value is given by pointing at another listing], "free": ["property", ...], "kind": "firm" | "nopref" | "confused" | "crowded" | "other", "budget": a new total limit as a number or null, "drop": ["a requirement the shopper gives up", ...]}
- "needs": every firm preference the reply states, with the exact value or limit and unit, named by the property the reply names. Folksy or roundabout wording that names a value still states it. A value given by pointing at another listing ("like the one shop X sells") is one need holding those words exactly.
- "free": properties the reply says the shopper does not mind about.
- "kind": "firm" when the reply states at least one preference; "nopref" when it only says they don't mind; "confused" when it says they did not follow or asks to put it differently; "crowded" when it complains about too many questions or wants one at a time; else "other"."""

NOTE = """While a shopping agent works, a tool result carried the note below (from the shopper or the store). Return ONE JSON object and nothing else:
{"change": true when the note changes the task (budget, requirements, price, availability, a refused order), "budget": the new total limit as a number in the request's unit or null, "add": ["property: constraint", ...], "drop": ["the requirement given up", ...], "gone": [listing ids said to be unavailable], "sold_out": true when it says the chosen listing is gone or sold out, "summary": "one short sentence"}
A replaced requirement is a drop of the old one and an add of the new one. Report every change."""

CHECK = """You are the last check before a shopping agent checks out. Decide from the records shown whether this cart is exactly right.
""" + READ_RULES + """
Each requested item must be one cart line at its quantity, nothing else may be in the cart, and the total must be within the budget. The newest record of a listing wins. When the pick rule is cheapest, most or least, also look at the OTHER LISTINGS: if one of them clearly fits every requirement and beats the cart's choice on that rule, the cart is wrong.
Return ONE JSON object and nothing else: {"ok": true or false, "problems": ["short concrete problem", ...], "better": [listing ids that should replace a cart line, best first]}. Use ok=false only for a problem the records clearly show, or a requirement no record addresses."""

AUDIT = """A shopping agent wants to end without buying because it thinks nothing in the store fits. Check every listing record shown against the requirements.
""" + READ_RULES + """
Records marked ROW were seen only as search rows: their attributes are unread, so they cannot be failed on a property the row does not show.
Return ONE JSON object and nothing else: {"fits": [ids of listings that appear to meet every requirement, best first, at most 4], "open": [ids of ROW listings of the requested kind, within the budget, whose page could show they fit, most promising first, at most 6], "search": [up to 2 short search queries that could find a fitting listing not seen yet], "reason": "one sentence"}"""

GATE = """A shopping agent is about to STAGE. The request asks for the procedural steps below. From the numbered RECORD of what the agent has done, decide for each step whether it is done for the cart as it is now (a step about the order counts only when its record line comes after the cart last changed). Questions about product properties do not do a step.
If a step is still missing, give the one store call that does it next, using a tool from TOOLS with its exact argument names. A message to the shopper is one short plain sentence about that step only.
Return ONE JSON object and nothing else: {"missing": ["step", ...], "call": {"tool": "...", "args": {...}} or null}. Give no call when everything is done or the messages or steps left cannot do it."""


SWAP = """A listing in a shopping agent's cart has sold out, or the agent needs the next choice. From the listing records shown, find the listings that meet every requirement now in force.
""" + READ_RULES + """
Records marked ROW were seen only as search rows (attributes unread); list them under "open" when they could fit.
Return ONE JSON object and nothing else: {"fits": [ids that meet every requirement, in the order to buy them: listings whose attributes state each value exactly first, then (cheapest first under a cheapest rule, the extreme first under a most or least rule)], "open": [ROW ids worth opening, most promising first, at most 4]}"""

SWEEP = """A shopping agent must buy the CHEAPEST listing that meets every requirement. Its cart holds a listing priced PRICE. Give up to 2 store calls that list the listings of the requested kind priced below PRICE, using the store's price filter, category filter or sorting where the tools offer them (copy the argument names and allowed values exactly from the schemas; the earlier calls show what worked in this store). Prefer one call that lists the whole cheaper set of the requested kind; keep any keyword part short (the product noun and a required value).
Return ONE JSON object and nothing else: {"calls": [{"tool": "...", "args": {...}}]}"""


# Cairn's chain-following prompt (public code, race-17 and race-18 winner), unchanged.
TRAIL = 'A shopping agent is about to check out. Some requirements take their value from another listing instead of stating it, and a listing\'s page can send the reader on to a further listing for that property. The order counts only when the agent has opened in full every listing along each such chain. Follow each requirement\'s chain from the requirement\'s own description, link by link, through the pages\' own words, using the records below (OPENED: the agent has opened that page; ROW: its page not read, seen only as a search result row or by its attributes).\nReturn ONE JSON object and nothing else: {"chains": [{"need": the requirement copied exactly, "links": [{"id": the listing id when a record below is that link, else "", "about": how the requirement or the previous page describes that listing, "find": "for a link not found: search words (its shop and the values the description gives)"}], "done": true when the last link is an OPENED page that states the property\'s value itself (also when the requirement asks for a value other than or unlike that one: the page\'s value is what it is measured against), else false, "value": the property\'s value as stated on that last page; when the requirement asks for what goes with that value and the page names what its value pairs with, that other value; or ""}]}\nA record is a link only when it matches the whole description of that link: the shop or brand, the kind of product AND every property value the description gives (for example "a 10 pieces multipack" rules out the same shop\'s listing with 10 capsules and no multipack); when no record matches all of it, that link is not found yet, even if the same shop sells other listings of that kind. Give "find" for such a link: the shop\'s name and the description\'s values, as a few search words.\nList each chain\'s links in order and stop at the first link whose page is not OPENED or not found, since its words are still unknown.'


def _clip(value: Any, limit: int) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _json_obj(text: Any) -> dict[str, Any] | None:
    if not isinstance(text, str):
        return None
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*", "", text).rstrip("`").strip()
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except ValueError:
        pass
    start = text.find("{")
    while start >= 0:
        depth = 0
        quoted = False
        escaped = False
        for index in range(start, len(text)):
            char = text[index]
            if quoted:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    quoted = False
                continue
            if char == '"':
                quoted = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    try:
                        value = json.loads(text[start: index + 1])
                        if isinstance(value, dict):
                            return value
                    except ValueError:
                        break
                    break
        start = text.find("{", start + 1)
    return None


def _num(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        match = re.search(r"-?\d+(?:[.,]\d+)?", value.replace(",", ""))
        if match:
            try:
                return float(match.group(0))
            except ValueError:
                return None
    return None


def _words(text: str) -> list[str]:
    spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(text))
    return [w for w in re.split(r"[^a-z0-9]+", spaced.lower()) if w]


def _plain_key(key: str) -> str:
    return " ".join(_words(key))


def _stem(word: str) -> str:
    for end in ("ies", "es", "s"):
        if len(word) > 4 and word.endswith(end):
            return word[: -len(end)] + ("y" if end == "ies" else "")
    return word


def _same_words(a: str, b: str) -> bool:
    wa = {_stem(w) for w in _words(a)}
    wb = {_stem(w) for w in _words(b)}
    return bool(wa) and wa == wb


def _overlap(a: str, b: str) -> float:
    wa = {_stem(w) for w in _words(a)}
    wb = {_stem(w) for w in _words(b)}
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def _pairs_text(text: str) -> dict[str, str] | None:
    """Parse "key: value | key: value" or "key=value; key=value" record strings."""
    if not isinstance(text, str) or len(text) < 6:
        return None
    parts = None
    if " | " in text and re.search(r"(?:^|\| )[\w .\-/]{1,40}: ", text):
        parts = [p for p in re.split(r"\s\|\s(?=[\w .\-/]{1,40}:\s)", text)]
        sep = ":"
    elif re.search(r"[\w .\-/]{1,40}=", text) and (";" in text or "\n" in text):
        parts = re.split(r";\s*|\n", text)
        sep = "="
    elif "\n" in text and re.search(r"^[\w .\-/]{1,40}:\s", text, re.M):
        parts = text.split("\n")
        sep = ":"
    if not parts:
        return None
    out: dict[str, str] = {}
    for part in parts:
        if sep not in part:
            continue
        key, _, value = part.partition(sep)
        key = key.strip(" -*\t")
        if 0 < len(key) <= 40 and not re.search(r"[{}\[\]]", key):
            out[key] = value.strip()
    return out if len(out) >= 2 else None


def _flatten(node: Any, out: dict[str, Any], prefix: str = "", depth: int = 0) -> None:
    if depth > 4:
        return
    if isinstance(node, dict):
        for key, value in node.items():
            name = str(key)
            if isinstance(value, dict):
                _flatten(value, out, name if not prefix else prefix, depth + 1)
            elif isinstance(value, str):
                inner = _pairs_text(value) if ("=" in value and (";" in value or "\n" in value)) else None
                if inner and not ID_VALUE.match(value):
                    for inner_key, inner_value in inner.items():
                        out.setdefault(inner_key, inner_value)
                else:
                    out[name] = value
            elif isinstance(value, list) and all(not isinstance(v, (dict, list)) for v in value):
                out[name] = ", ".join(str(v) for v in value)
            elif not isinstance(value, (list, dict)):
                out[name] = value


def _glued(text: str) -> str:
    """A number and its short unit as one word ("65 W" and "65W"). From Cairn."""
    return re.sub(r"(\d)\s+([a-z]{1,3})(?![a-z0-9])", r"\1\2", text.lower())


def _text_words(text: str) -> set[str]:
    low = text.lower()
    return set(re.findall(r"[a-z0-9]+", low)) | set(re.findall(r"[a-z0-9]+", _glued(low)))


def _value_tokens(value: str) -> list[str]:
    """A value's words a search or a row can hold (from Cairn)."""
    low = _glued(value)
    return re.findall(r"[a-z0-9]{3,}", low) or re.findall(r"[0-9]+[a-z]|[0-9]*[a-z][a-z0-9]+", low) or re.findall(r"[0-9]+", low)


def _roots(text: str) -> set[str]:
    out = set()
    for w in _words(text):
        w = w[:-2] if re.search(r"(?:ch|sh|ss|x|z)es$", w) else w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w
        if len(w) >= 3 and w not in ("and", "for", "the", "with"):
            out.add(w)
    return out


class Compass:
    def __init__(self, problem: dict[str, Any]):
        self.started = time.monotonic()
        environment = problem["environment"]
        self.binding = dict(environment["binding"])
        view = environment["policy_view"]
        self.request = str(view.get("query") or "")
        self.opening = view.get("opening")
        self.tools = [t for t in view.get("tools") or [] if isinstance(t, dict) and isinstance(t.get("function"), dict)]
        self.schemas = {t["function"]["name"]: t["function"].get("parameters") or {} for t in self.tools}
        self.descs = {t["function"]["name"]: str(t["function"].get("description") or "") for t in self.tools}
        try:
            self.max_steps = int(view.get("max_steps") or 30)
        except (TypeError, ValueError):
            self.max_steps = 30
        self.step = 0
        self.finished = bool(view.get("finished"))
        self.dead = False
        self.cost = 0.0
        self.trace: list[dict[str, Any]] = []
        # task sheet
        self.sheet: dict[str, Any] = {}
        self.items: list[dict[str, Any]] = []
        self.roles: dict[str, str] = {}
        self.budget: float | None = None
        self.msg_limit = 0
        self.msg_used = 0
        self.reserve = 0
        # store memory
        self.records: dict[str, dict[str, Any]] = {}   # id -> merged fields
        self.opened: set[str] = set()                   # ids whose attributes were read
        self.order_seen: list[str] = []
        self.gone: set[str] = set()
        self.notes: list[str] = []
        self.log: list[str] = []                        # numbered record of calls and messages
        self.cart_at = 0                                # log length when the cart last changed
        self.cart_view_at = -1
        self.cart_lines: list[dict[str, Any]] = []
        # interview
        self.queue: dict[int, list[str]] = {}
        self.asked: dict[int, list[str]] = {}
        self.free: dict[int, list[str]] = {}
        self.dropped_labels: set[str] = set()
        self.singles = False
        self.interviewed = False
        self.planned = False
        self.heard_lines: list[str] = []
        self.buy_holds = 0
        self.refuse_holds = 0
        self.audited = False
        self.swept = 0
        self.convo: list[dict[str, Any]] = []
        self.inbox: list[str] = []
        # search memory and chains (Cairn's parts)
        self.searches: list[tuple[str, dict[str, Any], Any, list[str]]] = []
        self.read_marks: dict[str, Any] = {}
        self.fit_hunted = False
        self.refs: list[tuple[int, str]] = []
        self.ref_values: dict[str, str] = {}
        self.trails = 0
        self.trail_blocks = 0
        self.trail_room: int | None = None
        self.refuse_trailed = False
        self.link_searches = 0
        self.link_seen: dict[str, list[str]] = {}
        self.hunts = 0
        self.audit_fits: list[str] = []
        self.hunt_mark = ""
        self.hunt_tried: dict[int, set[str]] = {}
        self.hunt_stem: dict[tuple[int, tuple[str, ...]], str] = {}
        self.value_rows: set[str] = set()

    # ------------------------------------------------------------------ clock, spend, steps
    def elapsed(self) -> float:
        return time.monotonic() - self.started

    def time_left(self) -> float:
        return TASK_SECONDS - self.elapsed()

    def steps_left(self) -> int:
        return self.max_steps - self.step

    def msgs_left(self) -> int:
        return max(0, self.msg_limit - self.msg_used)

    def judge_model(self) -> str | None:
        if self.cost >= SPEND_HARD or self.elapsed() > NO_HEAVY_AFTER + 30:
            return None
        if self.cost >= SPEND_SOFT or self.elapsed() > NO_HEAVY_AFTER:
            return LIGHT
        return HEAVY

    # ------------------------------------------------------------------ model and store calls
    def infer(self, model: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
              max_tokens: int = 1500, effort: str | None = None) -> dict[str, Any] | None:
        for name in (model, SPARE if model != SPARE else LIGHT):
            left = self.time_left() - SAFE_END + 10
            if left < 6:
                return None
            _llm.timeout = int(max(6, min(55, left)))
            sent = messages
            if name.startswith("anthropic/"):
                sent = [dict(m) for m in messages]
                marks = [0] + ([len(sent) - 1] if len(sent) > 2 else [])
                for index in marks:
                    content = sent[index].get("content")
                    if isinstance(content, str) and content:
                        sent[index]["content"] = [{"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}]
            body: dict[str, Any] = {"model": name, "messages": sent, "max_tokens": max_tokens, "temperature": 0}
            if tools:
                body["tools"] = tools
                body["tool_choice"] = "auto"
            if effort:
                # judging calls think briefly: a long hidden deliberation used to run out of tokens before the answer
                body["reasoning"] = {"effort": effort}
            result = _llm.post_verbose("/inference/chat/completions", body)
            data = result.data
            if isinstance(data, dict) and data.get("choices"):
                usage = data.get("usage") or {}
                if isinstance(usage.get("cost"), (int, float)):
                    self.cost += float(usage["cost"])
                choice = data["choices"][0] or {}
                message = choice.get("message") or {}
                cut = choice.get("finish_reason") == "length" and not message.get("tool_calls")
                if (message.get("content") or message.get("tool_calls")) and not (cut and not tools):
                    return message
                if cut:
                    self.trace.append({"kind": "model_cut", "model": name})
            self.trace.append({"kind": "model_failure", "model": name, "error": _clip(result.error, 300)})
        return None

    def ask_json(self, prompt: str, payload: str, model: str | None = None, max_tokens: int = 1500) -> dict[str, Any]:
        model = model or self.judge_model()
        if not model:
            return {}
        content = [{"type": "text", "text": prompt, "cache_control": {"type": "ephemeral"}}, {"type": "text", "text": payload}]
        cuts = sum(1 for t in self.trace if t.get("kind") == "model_cut")
        message = self.infer(model, [{"role": "user", "content": content}], None, max(max_tokens, 2400))
        got = _json_obj((message or {}).get("content")) or {}
        if not got and sum(1 for t in self.trace if t.get("kind") == "model_cut") > cuts and self.time_left() > SAFE_END + 40:
            # the answer ran out of tokens while thinking: ask once more with brief thinking
            message = self.infer(model, [{"role": "user", "content": content}], None, max(max_tokens, 2400), "low")
            got = _json_obj((message or {}).get("content")) or {}
        return got

    def call(self, name: str, args: dict[str, Any]) -> Any:
        """One store step. Returns the observation, or None when the call failed."""
        if self.finished or self.dead:
            return None
        payload = dict(self.binding)
        payload["step"] = self.step + 1
        payload["action"] = {"name": name, "args": args}
        for _ in range(3):
            left = self.time_left()
            if left < 2:
                return None
            _env.timeout = int(max(3, min(60, left - 1)))
            result = _env.post_verbose("/environment/call", payload)
            data = result.data
            if isinstance(data, dict):
                self.step = data["step"] if isinstance(data.get("step"), int) else self.step + 1
                if data.get("finished"):
                    self.finished = True
                observation = data.get("observation")
                self.after_call(name, args, observation)
                return observation
            error = result.error or {}
            status = error.get("status") if isinstance(error, dict) else None
            body = str(error.get("body") if isinstance(error, dict) else error)
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
            if status in (400, 403, 404, 422):
                return None
        return None

    # ------------------------------------------------------------------ reading the request
    def read_request(self) -> None:
        lines = []
        for tool in self.tools:
            fn = tool["function"]
            lines.append(f"- {fn['name']}: {fn.get('description', '')} | parameters: {_clip(fn.get('parameters') or {}, 900)}")
        payload = "REQUEST:\n" + self.request + "\n\nTOOLS:\n" + "\n".join(lines)
        if self.opening is not None:
            payload += "\n\nOPENING OBSERVATION:\n" + _clip(self.opening, 2500)
        sheet = self.ask_json(SHEET, payload, HEAVY, 2200)
        if not sheet:
            sheet = self.ask_json(SHEET, payload, LIGHT, 2200)
        self.sheet = sheet
        self.trace.append({"kind": "sheet", "sheet": _clip(sheet, 2500)})
        for raw in sheet.get("items") or []:
            if not isinstance(raw, dict):
                continue
            qty = _num(raw.get("quantity"))
            self.items.append({
                "name": str(raw.get("name") or "the product"),
                "label": str(raw.get("label") or ""),
                "qty": int(qty) if qty and qty > 0 else 1,
                "musts": [str(m) for m in raw.get("musts") or [] if m and not re.search(
                    r"stock|availab|on.hand|in.store|on the shelf|price|budget|cost|quantity", str(m).split(":")[0], re.I)],
                "needs": [],
            })
        if not self.items:
            self.items.append({"name": "the product", "label": "", "qty": 1, "musts": [], "needs": []})
        for lookup in sheet.get("lookups") or []:
            text = str(lookup).strip()
            index = next((n for n, item in enumerate(self.items) if text in item["musts"]), None)
            if index is None:
                key = text.split(":")[0].strip().lower()
                index, text = next(((n, m) for n, item in enumerate(self.items) for m in item["musts"]
                                    if m.split(":")[0].strip().lower() == key), (0, text))
                if text not in self.items[index]["musts"]:
                    self.items[index]["musts"].append(text)
            if text and (index, text) not in self.refs:
                self.refs.append((index, text))
        self.budget = _num(sheet.get("budget"))
        roles = sheet.get("roles") if isinstance(sheet.get("roles"), dict) else {}
        self.roles = {k: v for k, v in roles.items() if isinstance(v, str) and v in self.schemas}
        self.guess_roles()
        limit = sheet.get("messages")
        if isinstance(limit, (int, float)) and not isinstance(limit, bool) and limit >= 0:
            self.msg_limit = int(limit)
        else:
            match = re.search(r"(\d+)\s+(?:messages?|questions?)", self.request, re.I)
            self.msg_limit = int(match.group(1)) if match else (3 if self.roles.get("ask") else 0)
        if not self.roles.get("ask"):
            self.msg_limit = 0
        steps = _num(sheet.get("steps"))
        if steps and 8 <= steps < self.max_steps:
            self.max_steps = int(steps)
        talk = [s for s in self.process() if (_num(s.get("messages")) or 0) > 0]
        self.reserve = min(self.msg_limit, int(sum(_num(s.get("messages")) or 0 for s in talk)))
        for index in range(len(self.items)):
            self.queue[index] = []
            self.asked[index] = []
            self.free[index] = []

    def guess_roles(self) -> None:
        """Fill roles the sheet left out: tool names first, then descriptions."""
        hints = [
            ("refuse", r"decline|give.?up|abandon|refuse|no.?purchase|pass|without|close.?out|cancel"),
            ("buy", r"checkout|check.?out|purchase|order|buy|settle|conclude|finish|pay"),
            ("ask", r"ask|message|question|shopper|customer|buyer|chat|write|contact|note"),
            ("cart_view", r"cart|bag|basket|selection"),
            ("cart", r"cart|bag|basket|hold|stash|keep|add"),
            ("compare", r"compare|weigh|side|several|batch|multiple"),
            ("open", r"detail|spec|attribute|propert|facts|datasheet|full|read|inspect|page|show|view"),
            ("search", r"search|find|locate|seek|lookup|query|browse|list"),
        ]
        for field in ("name", "desc"):
            for role, pattern in hints:
                if self.roles.get(role):
                    continue
                for name, desc in self.descs.items():
                    text = name if field == "name" else desc
                    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text).replace("_", " ").lower()
                    if name in self.roles.values() or not re.search(pattern, text):
                        continue
                    props = (self.schemas.get(name) or {}).get("properties") or {}
                    if role == "cart_view" and props:
                        continue
                    if role in ("cart", "ask", "search", "open", "compare") and not props:
                        continue
                    self.roles[role] = name
                    break

    def process(self, when: str | None = None) -> list[dict[str, Any]]:
        out = []
        for step in self.sheet.get("steps_required") or []:
            if isinstance(step, dict) and step.get("step") and (when is None or step.get("when") == when):
                out.append(step)
        return out

    # ------------------------------------------------------------------ store memory
    def learn(self, node: Any, source: str, depth: int = 0) -> list[str]:
        """Merge every listing record found in an observation; returns the ids found."""
        found: list[str] = []
        if depth > 6:
            return found
        if isinstance(node, str):
            pairs = _pairs_text(node)
            if pairs:
                found += self.merge(pairs, source)
            return found
        if isinstance(node, list):
            for value in node:
                found += self.learn(value, source, depth + 1)
            return found
        if not isinstance(node, dict):
            return found
        ident = self.ident(node)
        if ident:
            flat: dict[str, Any] = {}
            _flatten(node, flat)
            found += self.merge(flat, source, ident)
            for value in node.values():
                if isinstance(value, list):
                    found += self.learn(value, source, depth + 1)
            return found
        for value in node.values():
            found += self.learn(value, source, depth + 1)
        return found

    def ident(self, node: dict[str, Any]) -> str | None:
        for key, value in node.items():
            if isinstance(value, str) and ID_KEY.search(str(key)) and ID_VALUE.match(value.strip()):
                if not re.fullmatch(r"\d{1,6}", value):
                    return value.strip()
        return None

    def merge(self, fields: dict[str, Any], source: str, ident: str | None = None) -> list[str]:
        if ident is None:
            for key, value in fields.items():
                if isinstance(value, str) and ID_KEY.search(str(key)) and ID_VALUE.match(value.strip()):
                    ident = value.strip()
                    break
        if not ident:
            return []
        record = self.records.setdefault(ident, {})
        for key, value in fields.items():
            if value is None or value == "" or value == "None":
                continue
            if isinstance(value, str) and "=" in value and ";" in value and not set(_words(key)) & SKIP_WORDS:
                inner = _pairs_text(value)
                if inner:
                    for inner_key, inner_value in inner.items():
                        record[inner_key] = inner_value
                    continue
            record[str(key)] = value
        if ident not in self.order_seen:
            self.order_seen.append(ident)
        if source != "search":
            attrs = self.attributes(ident)
            if len(attrs) >= 3:
                self.opened.add(ident)
        return [ident]

    def attributes(self, ident: str) -> dict[str, Any]:
        record = self.records.get(ident) or {}
        out = {}
        for key, value in record.items():
            words = set(_words(key))
            if not words or words & SKIP_WORDS or (ID_KEY.search(key) and ID_VALUE.match(str(value))):
                continue
            if isinstance(value, str) and len(value) > 160:
                continue
            out[key] = value
        return out

    def price_of(self, ident: str) -> float | None:
        record = self.records.get(ident) or {}
        for key, value in record.items():
            if PRICE_WORDS.search(key) and not re.search(r"total|owed", key, re.I):
                number = _num(value)
                if number is not None:
                    return number
        return None

    def in_stock(self, ident: str) -> bool | None:
        if ident in self.gone:
            return False
        record = self.records.get(ident) or {}
        for key, value in record.items():
            if STOCK_WORDS.search(key):
                if isinstance(value, bool):
                    return value
                text = str(value).lower()
                if re.search(r"^(?:no|none|false|0|n)$|out.?of.?stock|sold.?out|unavailable", text):
                    return False
                return True
        return None

    def category_of(self, ident: str) -> str:
        record = self.records.get(ident) or {}
        for key, value in record.items():
            if re.search(r"categor|section|department|group|shelf|label", key, re.I) and isinstance(value, str):
                return value
        return ""

    def title_of(self, ident: str) -> str:
        record = self.records.get(ident) or {}
        for key, value in record.items():
            if re.search(r"title|name|headline", key, re.I) and isinstance(value, str) and not ID_VALUE.match(value):
                return value
        return ""

    def owner(self, ident: str) -> int:
        if len(self.items) == 1:
            return 0
        text = self.category_of(ident) + " " + self.title_of(ident)
        scores = [(_overlap(text, item["label"] or item["name"]) + 0.5 * _overlap(text, item["name"]), i)
                  for i, item in enumerate(self.items)]
        return max(scores)[1]

    def record_text(self, ident: str, full: bool = True) -> str:
        record = self.records.get(ident) or {}
        parts = []
        for key, value in record.items():
            text = str(value)
            if re.search(r"descr|details|overview|text|summary", key, re.I):
                text = text[: (500 if full else DESC_CHARS)]
            parts.append(f"{key}: {text}")
        mark = "OPENED" if ident in self.opened else "ROW"
        if ident in self.gone:
            mark += ", SOLD OUT"
        return f"[{ident}] ({mark}) " + " | ".join(parts)

    def after_call(self, name: str, args: dict[str, Any], observation: Any) -> None:
        role = "search" if name == self.roles.get("search") else "other"
        ids = self.learn(observation, role)
        if role == "search" and isinstance(observation, (dict, list)) and not (
                isinstance(observation, dict) and observation.get("error")):
            self.searches.append((name, dict(args), observation, list(dict.fromkeys(ids))))
            self.read_marks[self.call_mark(name, args)] = observation
        self.log.append(f"{name} {_clip(args, 200)} -> {_clip(observation, 300)}")
        if name == self.roles.get("cart") or (name == self.roles.get("buy") and name != self.roles.get("cart_view")):
            self.cart_at = len(self.log)
        if name == self.roles.get("cart_view"):
            self.cart_view_at = len(self.log)
            self.cart_lines = self.lines_in(observation)
        self.find_notes(observation, name)

    def lines_in(self, observation: Any) -> list[dict[str, Any]]:
        lines: list[dict[str, Any]] = []

        def walk(node: Any, depth: int) -> None:
            if depth > 5:
                return
            if isinstance(node, dict):
                ident = self.ident(node)
                if ident:
                    qty = None
                    for key, value in node.items():
                        if re.search(r"qty|quantit|count|number|units|how_many|amount", key, re.I) and \
                                not re.search(r"price|cost|owed", key, re.I):
                            qty = _num(value)
                    lines.append({"id": ident, "qty": int(qty) if qty else 1})
                    return
                for value in node.values():
                    walk(value, depth + 1)
            elif isinstance(node, list):
                for value in node:
                    walk(value, depth + 1)
            elif isinstance(node, str):
                pairs = _pairs_text(node)
                if pairs:
                    walk(pairs, depth + 1)

        walk(observation, 0)
        return lines

    # ------------------------------------------------------------------ notes inside tool results
    def find_notes(self, observation: Any, name: str) -> None:
        sentences: list[str] = []

        def walk(node: Any, depth: int, key: str) -> None:
            if depth > 5:
                return
            if isinstance(node, dict):
                for k, v in node.items():
                    walk(v, depth + 1, str(k))
            elif isinstance(node, list):
                for v in node:
                    walk(v, depth + 1, key)
            elif isinstance(node, str):
                text = node.strip()
                if len(text) > 25 and " " in text and not _pairs_text(text) and \
                        not re.search(r"descr|details|overview|title|headline|name|summary", key, re.I) and \
                        re.search(r"[.!?]$|\b(?:I|my|you|your|sorry|now|anymore|budget|sold|gone)\b", text):
                    sentences.append(text)

        if name == self.roles.get("ask"):
            return
        walk(observation, 0, "")
        fresh = [s for s in sentences if s not in self.notes]
        if not fresh:
            return
        self.notes += fresh
        note = self.ask_json(NOTE, "REQUEST:\n" + self.request + "\n\nITEMS:\n" + self.items_text() +
                             "\n\nNOTE:\n" + "\n".join(fresh), LIGHT, 600)
        if not note:
            self.inbox.append("Store note: " + " ".join(fresh))
            return
        if note.get("change"):
            budget = _num(note.get("budget"))
            if budget:
                self.budget = budget
            for add in note.get("add") or []:
                self.items[0 if len(self.items) == 1 else self.best_item(str(add))]["needs"].append(str(add))
            for drop in note.get("drop") or []:
                self.drop_need(str(drop))
            for ident in note.get("gone") or []:
                if isinstance(ident, str):
                    self.gone.add(ident)
            if note.get("sold_out"):
                for line in self.cart_lines:
                    self.gone.add(line["id"])
        self.inbox.append("Note in the last result: " + " ".join(fresh) + " -> " + str(note.get("summary") or ""))

    def best_item(self, text: str) -> int:
        scores = [(_overlap(text, " ".join(item["musts"] + [item["name"]])), i) for i, item in enumerate(self.items)]
        return max(scores)[1]

    def drop_need(self, text: str) -> None:
        prop = text.split(":")[0]
        for item in self.items:
            for bucket in ("musts", "needs"):
                keep = []
                for need in item[bucket]:
                    if _overlap(need.split(":")[0], prop) >= 0.5 or _overlap(need, text) >= 0.6:
                        item.setdefault("dropped", []).append(need)
                    else:
                        keep.append(need)
                item[bucket] = keep

    # ------------------------------------------------------------------ ledger
    def items_text(self) -> str:
        rows = []
        for index, item in enumerate(self.items, 1):
            line = f"{index}. {item['name']} x{item['qty']}"
            if item["label"]:
                line += f" (category: {item['label']})"
            reqs = item["musts"] + item["needs"]
            line += "; requirements: " + ("; ".join(reqs) if reqs else "none stated")
            if item.get("dropped"):
                line += "; no longer required: " + "; ".join(item["dropped"])
            kind = self.store_kind(index - 1)
            if kind:
                line += f"; the store's category for it: {kind} (listings of other categories are not this product)"
            pointed = [f"{need} -> {self.ref_values.get(need, 'value still to read from that listing')}"
                       for n, need in self.refs if n == index - 1 and need in reqs]
            if pointed:
                line += "; takes its value from another listing: " + " | ".join(pointed)
            rows.append(line)
        return "\n".join(rows)

    def store_kind(self, index: int) -> str:
        """The seen category label whose name covers more of the item's product words than any other ("Tables & Chairs"
        for tables or chairs, where a "Chairs" category covers one); "" when no label stands out."""
        item = self.items[index]
        stop = {"and", "or", "for", "the", "with", "pair", "set", "new", "one"}
        nouns = _roots(" ".join(w for w in _words(item["name"]) if w not in stop))
        if not nouns:
            return ""
        labels = {self.category_of(i) for i in self.order_seen} - {""}
        cover = sorted(((len(nouns & _roots(l)), l) for l in labels), reverse=True)
        if not cover or cover[0][0] == 0 or (len(cover) > 1 and cover[1][0] == cover[0][0]):
            return ""
        return cover[0][1]

    def ledger(self) -> str:
        sheet = self.sheet
        parts = ["LEDGER (Compass)", self.items_text()]
        if self.budget is not None:
            parts.append(f"Budget: {self.budget:g}{' per unit' if sheet.get('budget_each') else ' in total'}")
        pick = sheet.get("pick") or "any"
        if pick != "any":
            parts.append(f"Pick rule: {pick} {sheet.get('measure') or ''}".strip())
        if sheet.get("in_stock", True):
            parts.append("Only listings in stock count.")
        if sheet.get("wishes"):
            parts.append("Wishes (not required): " + "; ".join(map(str, sheet["wishes"])))
        steps = self.process()
        if steps:
            parts.append("Procedural steps the request asks for: " + "; ".join(f"{s['step']} ({s.get('when')})" for s in steps))
        free = [f"{self.items[i]['name']}: {', '.join(v)}" for i, v in self.free.items() if v]
        if free:
            parts.append("Shopper does not mind: " + "; ".join(free))
        if self.gone:
            parts.append("Sold out or unavailable: " + ", ".join(sorted(self.gone)))
        opened = [i for i in self.order_seen if i in self.opened][-12:]
        if opened:
            parts.append("Opened listings: " + "; ".join(
                f"{i} {self.price_of(i) if self.price_of(i) is not None else '?'}" for i in opened))
        if pick == "cheapest":
            parts.append(self.cheap_view())
        parts.append(f"Steps left: {self.steps_left()} of {self.max_steps}. Messages left: {self.msgs_left()}. "
                     f"Time left: {int(self.time_left())} s.")
        return "\n".join(parts)

    def cheap_view(self) -> str:
        """Under a cheapest rule: the lowest-priced rows seen, so the walk opens them in price order."""
        rows = []
        for ident in self.order_seen:
            price = self.price_of(ident)
            if price is None or ident in self.gone or self.in_stock(ident) is False:
                continue
            if self.budget is not None and price > self.budget:
                continue
            rows.append((price, ident))
        rows.sort()
        unread = [f"{i} {p:g} {self.title_of(i)[:50]}" for p, i in rows if i not in self.opened][:6]
        read = [f"{i} {p:g}" for p, i in rows if i in self.opened][:6]
        out = "Cheapest rule helper: lowest-priced rows in budget not opened yet: " + ("; ".join(unread) or "none")
        out += ". Lowest-priced opened: " + ("; ".join(read) or "none") + "."
        return out

    # ------------------------------------------------------------------ interview
    def key_table(self, index: int) -> list[tuple[str, int, list[str]]]:
        counts: dict[str, list] = {}
        mine = [i for i in self.order_seen if i in self.opened and self.owner(i) == index]
        for ident in mine:
            for key, value in self.attributes(ident).items():
                plain = _plain_key(key)
                if not plain or NOT_ASK.search(plain):
                    continue
                entry = counts.setdefault(plain, [0, []])
                entry[0] += 1
                text = str(value)[:40]
                if text not in entry[1]:
                    entry[1].append(text)
        rows = [(k, v[0], v[1][:4]) for k, v in counts.items()]
        rows.sort(key=lambda r: (-r[1], -len(r[2])))
        return rows[:30]

    def settled(self, index: int, label: str) -> bool:
        item = self.items[index]
        own = {_stem(w) for w in _words(item["name"] + " " + item["label"])}
        bare = " ".join(w for w in _words(label) if _stem(w) not in own) or label
        for need in item["musts"] + item["needs"]:
            prop = need.split(":")[0]
            if _same_words(prop, label) or _same_words(prop, bare) or _overlap(prop, label) >= 0.67:
                return True
        if any(_same_words(label, f) for f in self.free[index]):
            return True
        return label in self.dropped_labels

    def plan_questions(self) -> None:
        self.planned = True
        blocks = []
        for index, item in enumerate(self.items):
            table = self.key_table(index)
            rows = "\n".join(f"  - {key} (on {count} listing{'s' if count != 1 else ''}; e.g. {', '.join(values)})"
                             for key, count, values in table)
            blocks.append(f"ITEM {index + 1}: {item['name']}\n  settled by the request: "
                          f"{'; '.join(item['musts']) or 'nothing'}\n  keys on opened listings:\n{rows or '  (none read)'}")
        ask_first = self.sheet.get("ask_first") or []
        payload = ("REQUEST:\n" + self.request + "\n\n" + "\n\n".join(blocks) +
                   ("\n\nASK FIRST:\n" + "\n".join(map(str, ask_first)) if ask_first else ""))
        count = LIST_MAX * max(2, self.msg_limit)
        plan = self.ask_json(PLAN.replace("COUNT", str(count)), payload, self.judge_model() or LIGHT, 1500)
        for entry in plan.get("ask") or []:
            if not isinstance(entry, dict) or not entry.get("label"):
                continue
            number = _num(entry.get("item"))
            index = int(number) - 1 if number else 0
            if 0 <= index < len(self.items):
                label = str(entry["label"]).strip().lower()
                if label not in self.queue[index] and not self.settled(index, label):
                    self.queue[index].append(label)
        for index in range(len(self.items)):
            for key, _, _ in self.key_table(index):
                if key not in self.queue[index] and not self.settled(index, key) and len(self.queue[index]) < count:
                    self.queue[index].append(key)

    def interview_due(self) -> bool:
        if self.interviewed or self.msgs_left() - self.reserve <= 0 or not self.roles.get("ask"):
            return False
        if self.elapsed() > INTERVIEW_BY or self.steps_left() < 6:
            return False
        ready = all(sum(1 for i in self.opened if self.owner(i) == idx) >= OPEN_BEFORE_ASK
                    for idx in range(len(self.items)))
        return ready or self.step >= max(6, self.max_steps // ASK_AFTER_SHARE) or self.elapsed() > 90

    def interview(self, spend_all: bool = True) -> str:
        """Send planned questions while messages remain. Returns a summary for the walk."""
        if not self.planned:
            self.plan_questions()
        self.interviewed = True
        heard = []
        turn = 0
        while self.msgs_left() - self.reserve > 0 and self.steps_left() > 4 and self.elapsed() < INTERVIEW_BY + 20:
            index = self.next_item(turn)
            if index is None:
                break
            labels = [l for l in self.queue[index] if not self.settled(index, l)]
            if not labels:
                break
            size = 1 if self.singles else LIST_MAX
            batch = labels[:size]
            text = self.question(index, batch)
            observation = self.call(self.roles["ask"], self.message_args(text))
            if observation is None:
                break
            self.msg_used += 1
            reply = _clip(observation, 600)
            self.log.append(f"message to shopper: {text} -> {reply}")
            heard.append(f"Q: {text}\nA: {reply}")
            self.hear(index, batch, reply)
            self.trace.append({"kind": "asked", "labels": batch, "reply": reply[:200],
                               "needs": list(self.items[index]["needs"])})
            turn += 1
            if not spend_all:
                break
        if not heard:
            return ""
        chain = self.early_trail()
        if self.sheet.get("pick") == "cheapest":
            # while steps remain: by checkout the walk has often used them up (practice: the hunt never ran there)
            hunted = self.hunt_cheaper()
            chain = (chain + "\n" + hunted).strip() if hunted else chain
        pointed = self.pending_refs()
        return ("Compass asked the shopper:\n" + "\n".join(heard) +
                "\n\nRequirements now:\n" + self.items_text() +
                "\nSearch again with every requirement in mind and open the candidates you may buy." +
                (" These take their value from another listing: " + " | ".join(pointed) + ". Find that listing, open "
                 "its page in full, and open every listing its page refers to; the last page gives the value, and the "
                 "order counts only after all of them are opened." if pointed else "") +
                ("\n" + chain if chain else ""))

    def next_item(self, turn: int) -> int | None:
        open_items = [i for i in range(len(self.items)) if any(not self.settled(i, l) for l in self.queue[i])]
        if not open_items:
            return None
        # items without a revealed need first, then round robin
        open_items.sort(key=lambda i: (len(self.items[i]["needs"]) > 0, len(self.asked[i])))
        return open_items[0] if len(open_items) == 1 else open_items[turn % len(open_items)]

    def question(self, index: int, labels: list[str]) -> str:
        name = self.items[index]["name"]
        if len(labels) == 1:
            return f"What {labels[0]} would you like for the {name}?"
        return f"What {', '.join(labels[:-1])} or {labels[-1]} would you like for the {name}?"

    def message_args(self, text: str) -> dict[str, Any]:
        schema = self.schemas.get(self.roles.get("ask", ""), {})
        props = schema.get("properties") or {}
        args: dict[str, Any] = {}
        target = None
        for key, spec in props.items():
            if isinstance(spec, dict) and spec.get("type") in ("string", None) and not spec.get("enum"):
                if re.search(r"text|body|message|note|content|question|msg|say|reply|chat", key, re.I):
                    target = key
                    break
        if target is None:
            target = next((k for k, s in props.items() if isinstance(s, dict) and s.get("type") == "string"
                           and not s.get("enum")), None)
        for key in schema.get("required") or []:
            spec = props.get(key) or {}
            if key == target:
                continue
            if spec.get("enum"):
                args[key] = spec["enum"][0]
        if target:
            limit = (props.get(target) or {}).get("maxLength")
            args[target] = text[: int(limit)] if isinstance(limit, int) else text
        return args

    def hear(self, index: int, batch: list[str], reply: str) -> None:
        verdict = self.ask_json(HEAR, f"PRODUCT: {self.items[index]['name']}\nPROPERTIES NAMED: {', '.join(batch)}"
                                f"\nREPLY: {reply}", self.judge_model() or LIGHT, 600)
        kind = str(verdict.get("kind") or "")
        if not kind:
            kind = ("crowded" if CROWD_WORDS.search(reply) else "confused" if CONFUSED_WORDS.search(reply)
                    else "nopref" if NOPREF_WORDS.search(reply) else "other")
        needs = [str(n) for n in verdict.get("needs") or [] if n and ":" in str(n)]
        budget = _num(verdict.get("budget"))
        if budget:
            self.budget = budget
        for drop in verdict.get("drop") or []:
            self.drop_need(str(drop))
        for label in batch:
            if label not in self.asked[index]:
                self.asked[index].append(label)
        if kind == "crowded":
            self.singles = True
            return
        if kind == "confused" or (kind == "other" and not needs):
            # a confused list is usually confused by its first property; that property is almost never a hidden one
            self.dropped_labels.add(batch[0])
            self.free[index].append(batch[0])
            return
        if needs:
            for need in needs:
                if need not in self.items[index]["needs"]:
                    self.items[index]["needs"].append(need)
            for point in verdict.get("points") or []:
                point = str(point).strip()
                if point in needs and (index, point) not in self.refs:
                    self.refs.append((index, point))
            # properties named before the answered one count as not wanted
            hit = None
            for pos, label in enumerate(batch):
                if any(_overlap(n.split(":")[0], label) >= 0.5 for n in needs):
                    hit = pos
                    break
            before = batch[:hit] if hit is not None else batch[:1]
            for label in before:
                self.free[index].append(label)
            self.requeue(index, batch[(hit if hit is not None else 0) + 1:])
            return
        if len(batch) == 1:
            self.free[index].append(batch[0])
            return
        # a "don't mind" reply answers about the first properties; the rest are asked again first
        for label in batch[:SETTLED_BY_PLAIN]:
            if label not in self.free[index]:
                self.free[index].append(label)
        self.requeue(index, batch[SETTLED_BY_PLAIN:])

    def requeue(self, index: int, labels: list[str]) -> None:
        """Put labels named but not answered back at the front of what is still to ask, in their order."""
        rest = [l for l in self.queue[index] if l not in labels]
        done = [l for l in rest if self.settled(index, l)]
        pending = [l for l in rest if not self.settled(index, l)]
        self.queue[index] = done + list(labels) + pending

    # ------------------------------------------------------------------ the walk
    def walk_tools(self) -> list[dict[str, Any]]:
        hidden = {self.roles.get("ask")}
        return [t for t in self.tools if t["function"]["name"] not in hidden]

    def show(self, observation: Any) -> str:
        def trim(node: Any, depth: int = 0) -> Any:
            if depth > 6:
                return node
            if isinstance(node, dict):
                return {k: (v[:DESC_CHARS] + "..." if isinstance(v, str) and len(v) > DESC_CHARS and
                            re.search(r"descr|details|overview|text|summary", k, re.I) else trim(v, depth + 1))
                        for k, v in node.items()}
            if isinstance(node, list):
                return [trim(v, depth + 1) for v in node]
            if isinstance(node, str) and len(node) > 900:
                return re.sub(r"(description|details|overview): (.{%d}).*?(?= \| |$)" % DESC_CHARS, r"\1: \2...", node)
            return node
        return _clip(trim(observation), OBS_CHARS)

    def is_buy(self, name: str, args: dict[str, Any]) -> bool:
        if name != self.roles.get("buy"):
            return False
        if self.roles.get("refuse") == name:
            want = self.sheet.get("buy_args") or {}
            if isinstance(want, dict) and want:
                return all(str(args.get(k, "")).lower() == str(v).lower() for k, v in want.items())
            return not re.search(r"cancel|decline|none|abandon|refuse|give|pass|no", _clip(args, 200), re.I)
        return True

    def is_refuse(self, name: str, args: dict[str, Any]) -> bool:
        if name != self.roles.get("refuse"):
            return False
        if self.roles.get("buy") == name:
            return not self.is_buy(name, args)
        return True

    def run(self) -> None:
        self.read_request()
        if self.opening is not None:
            self.learn(self.opening, "other")
        tools = self.walk_tools()
        self.convo = [
            {"role": "system", "content": WALK},
            {"role": "user", "content": "REQUEST:\n" + self.request + "\n\n" + self.ledger() +
             ("\n\nOPENING OBSERVATION:\n" + self.show(self.opening) if self.opening is not None else "") +
             "\n\nStart: search for the requested items and open the most promising listings."},
        ]
        idle = 0
        for _ in range(LOOP_TURNS):
            if self.finished or self.dead:
                return
            if self.time_left() < SAFE_END + 8 or self.steps_left() <= 2:
                break
            if self.interview_due():
                summary = self.interview()
                if summary:
                    self.push_user(summary + "\n\n" + self.ledger())
            self.trim_history()
            message = self.infer(LIGHT, self.convo, tools, 1400)
            if message is None:
                idle += 1
                if idle >= 2:
                    break
                continue
            calls = message.get("tool_calls") or []
            content = message.get("content") or ""
            if not calls:
                idle += 1
                self.convo.append({"role": "assistant", "content": content or "(no call)"})
                self.push_user("Every reply must call a tool. Continue.\n" + self.ledger())
                if idle >= 3:
                    break
                continue
            idle = 0
            self.convo.append({"role": "assistant", "content": content or None, "tool_calls": calls})
            for position, tool_call in enumerate(calls):
                text = self.run_call(tool_call)
                if position == len(calls) - 1:
                    extra = "\n".join(self.inbox)
                    self.inbox = []
                    text += ("\n" + extra if extra else "") + "\n\n" + self.ledger()
                self.convo.append({"role": "tool", "tool_call_id": tool_call.get("id") or "x", "content": text})
                if self.finished:
                    return
        if not self.finished and not self.dead:
            self.force_finish()

    def push_user(self, text: str) -> None:
        if self.convo and self.convo[-1]["role"] == "user":
            self.convo[-1]["content"] += "\n\n" + text
        else:
            self.convo.append({"role": "user", "content": text})

    def trim_history(self) -> None:
        tool_turns = [i for i, m in enumerate(self.convo) if m["role"] == "tool"]
        for i in tool_turns[:-HISTORY_KEEP]:
            content = self.convo[i]["content"]
            if len(content) > 400:
                self.convo[i]["content"] = content[:400] + " ...(older result shortened; records are kept by Compass)"

    def run_call(self, tool_call: dict[str, Any]) -> str:
        fn = tool_call.get("function") or {}
        name = str(fn.get("name") or "")
        raw = fn.get("arguments") or "{}"
        try:
            args = json.loads(raw) if isinstance(raw, str) else dict(raw)
        except (ValueError, TypeError):
            return "Error: the arguments were not valid JSON. No step was used."
        if not isinstance(args, dict):
            args = {}
        if name not in self.schemas:
            return f"Error: there is no tool named {name}. No step was used."
        if name == self.roles.get("ask"):
            return "Compass's interviewer handles messages to the shopper. No step was used."
        if self.is_buy(name, args):
            held = self.before_buy()
            if held:
                return held
        elif self.is_refuse(name, args):
            held = self.before_refuse()
            if held:
                return held
            args = self.clean_refuse(name, args)
        if self.finished:
            return "The episode has finished."
        observation = self.call(name, args)
        if observation is None:
            return "The call failed (no result). " + ("The episode has finished." if self.finished else "")
        text = self.show(observation)
        if self.is_buy(name, args) and not self.finished:
            self.trace.append({"kind": "refused", "obs": _clip(observation, 400)})
            text = ("THE CHECKOUT DID NOT GO THROUGH: the episode is still running and nothing was bought. Read the "
                    "reasons below, apply every change, fix the cart (remove a sold-out line, add the next verified "
                    "fit), and check out again.\n" + text)
            if re.search(r"sold|gone|left|out of stock|unavailable|no longer|none left|ran out|bought|empty",
                         _clip(observation, 1500), re.I):
                if len(self.cart_lines) == 1:
                    self.gone.add(self.cart_lines[0]["id"])
                hint = self.replacements()
                if hint:
                    text += "\n" + hint
        return text

    # ------------------------------------------------------------------ checkout gate
    def refresh_cart(self) -> None:
        view = self.roles.get("cart_view")
        if view and self.cart_view_at < self.cart_at and self.steps_left() > 2:
            self.call(view, {})

    def before_buy(self) -> str | None:
        if self.time_left() < SAFE_END or self.steps_left() <= 1:
            return None
        # a spare message is one more question before buying
        if self.roles.get("ask") and self.msgs_left() - self.reserve > 0 and self.steps_left() > 4 and \
                self.elapsed() < INTERVIEW_BY + 30:
            summary = self.interview(spend_all=True)
            if summary and any(self.items[i]["needs"] for i in range(len(self.items))):
                return "Checkout held: " + summary + "\nCheck that the cart meets these, then check out."
        gate = self.run_gate("order")
        if gate:
            return gate
        if self.pending_refs() and self.elapsed() < HUNT_BY + HUNT_GRACE:
            chain, hold = self.trail()
            if hold:
                return "Checkout held. " + chain
            if chain:
                self.inbox.append(chain)
        # always look at the cart after its last change before ordering (races 16-18, orders on tasks with a
        # procedural step: viewed 0.18-0.35 passed, not viewed 0.08-0.11)
        self.refresh_cart()
        if self.buy_holds >= HOLDS or not self.judge_model():
            return None
        if not self.cart_lines:
            if self.roles.get("cart_view") and self.cart_view_at >= 0:
                self.trace.append({"kind": "gate", "what": "empty cart"})
                return "Checkout held: the cart is empty. Add the chosen listing at the requested quantity first."
            return None
        if self.sheet.get("pick") == "cheapest":
            hunted = self.hunt_cheaper()
            if hunted:
                self.inbox.append(hunted)
        if (self.sheet.get("pick") == "cheapest") and self.swept < 2 and len(self.items) == 1:
            self.cheaper_sweep()
        verdict = self.check_cart()
        self.trace.append({"kind": "gate", "what": "cart check", "verdict": _clip(verdict, 600)})
        if verdict and verdict.get("ok") is False and verdict.get("problems"):
            self.buy_holds += 1
            better = [b for b in verdict.get("better") or [] if isinstance(b, str)]
            text = ("Checkout held by Compass's check (a stronger model reading the records): " +
                    "; ".join(map(str, verdict["problems"]))[:900])
            if better:
                text += (f"\nIt names these as better fits: {', '.join(better[:4])}. Open the first one if its "
                         "attributes are not read yet; unless its own record clearly fails a requirement, put it in the "
                         "cart instead of the current line and check out.")
            else:
                text += "\nFix the cart, then check out. If the records clearly mean the same, check out as it is."
            return text
        return None

    def cheaper_sweep(self) -> None:
        """Under a cheapest rule: list the cheaper part of the requested kind and open its unread rows."""
        self.swept += 1
        prices = [self.price_of(l["id"]) for l in self.cart_lines]
        prices = [p for p in prices if p is not None]
        if not prices or self.steps_left() < 8 or self.time_left() < SAFE_END + 50:
            return
        price = min(prices)
        search_tools = [n for n in self.schemas if n in (self.roles.get("search"),) or
                        re.search(r"search|find|locate|seek|lookup|browse|list", n + " " + self.descs.get(n, ""), re.I)]
        tools = "\n".join(f"- {n}: {self.descs.get(n, '')} | {_clip(self.schemas.get(n), 700)}" for n in search_tools[:3])
        tried = "\n".join(line for line in self.log if any(line.startswith(n + " ") for n in search_tools))[-2500:]
        cat = self.category_of(self.cart_lines[0]["id"])
        payload = (SWEEP.replace("PRICE", f"{price:g}") + "\n\nREQUEST:\n" + self.request + "\n\nREQUIREMENTS:\n" +
                   self.items_text() + (f"\nThe cart line's category: {cat}" if cat else "") +
                   "\n\nSEARCH TOOLS:\n" + tools + "\n\nEARLIER SEARCH CALLS:\n" + (tried or "(none)"))
        plan = self.ask_json("Answer the task below.", payload, LIGHT, 500)
        found: list[str] = []
        for call in (plan.get("calls") or [])[:2]:
            if not isinstance(call, dict) or call.get("tool") not in search_tools or self.steps_left() < 7:
                continue
            args = call.get("args") if isinstance(call.get("args"), dict) else {}
            observation = self.call(call["tool"], args)
            if observation is not None:
                found += self.learn(observation, "search")
        rows = []
        for ident in dict.fromkeys(found):
            p = self.price_of(ident)
            if p is None or p >= price or ident in self.opened or ident in self.gone or self.in_stock(ident) is False:
                continue
            if cat and self.category_of(ident) and self.category_of(ident) != cat:
                continue
            rows.append((p, ident))
        rows.sort()
        opener = self.roles.get("open") or self.roles.get("offer")
        opened = 0
        for p, ident in rows:
            if opened >= 4 or self.steps_left() < 6 or not opener:
                break
            args = self.id_args(opener, ident)
            if args is None:
                break
            if self.call(opener, args) is not None:
                opened += 1
        self.trace.append({"kind": "sweep", "price": price, "plan": _clip(plan, 300), "cheaper_rows": len(rows),
                           "opened": opened})

    def replacements(self) -> str:
        """After a sold-out refusal: the next listings to buy, from every record seen."""
        if not self.judge_model():
            return ""
        ids = [i for i in self.order_seen if i not in self.gone and self.in_stock(i) is not False]
        ids = [i for i in ids if self.budget is None or (self.price_of(i) or 0) <= self.budget][-40:]
        if not ids:
            return ""
        payload = ("REQUEST:\n" + self.request + "\n\nREQUIREMENTS NOW:\n" + self.items_text() +
                   (f"\nBudget: {self.budget:g}" if self.budget is not None else "") +
                   f"\nPick rule: {self.sheet.get('pick') or 'any'}\nSold out: {', '.join(sorted(self.gone)) or 'see notes'}"
                   "\n\nRECORDS:\n" + "\n".join(self.record_text(i, False) for i in ids))
        verdict = self.ask_json(SWAP, _clip(payload, 30000), None, 700)
        self.trace.append({"kind": "swap", "verdict": _clip(verdict, 400)})
        fits = [f for f in verdict.get("fits") or [] if isinstance(f, str) and f in self.records and f not in self.gone]
        fits.sort(key=lambda f: -self.exact_hits(f))
        to_open = [o for o in verdict.get("open") or [] if isinstance(o, str) and o in self.records]
        if fits:
            return ("Compass's check of every record seen gives these replacements, in the order to buy: " +
                    "; ".join(f"{f} ({self.price_of(f) if self.price_of(f) is not None else '?'})" for f in fits[:4]) +
                    ". Remove the sold-out line from the cart, add the first one (open it first if its attributes are "
                    "not read), and check out. Do not end without a purchase while one of these fits.")
        if to_open:
            return ("No record seen so far is a sure fit. Open these listings next: " + ", ".join(to_open[:4]) +
                    ", then search on if none fits.")
        return ""

    def exact_hits(self, ident: str) -> int:
        """How many plain requirement values of the listing's item one of its own attributes states exactly."""
        index = self.owner(ident)
        attrs = self.attributes(ident)
        hits = 0
        for need in self.items[index]["musts"] + self.items[index]["needs"]:
            prop, _, value = need.partition(":")
            want = " ".join(_words(value))
            if not want:
                continue
            for key, got in attrs.items():
                parts = [" ".join(_words(p)) for p in re.split(r"[,/;]", str(got))]
                bare = lambda t: {w[:-1] if len(w) > 3 and w.endswith("s") else w for w in _words(t)}
                named = bare(prop) <= bare(key) or _overlap(key, prop) >= 0.5
                if named and any(p == want or _stem(p) == _stem(want) for p in parts):
                    hits += 1
                    break
        return hits

    def check_cart(self) -> dict[str, Any]:
        cart_ids = [line["id"] for line in self.cart_lines]
        lines = "\n".join(f"- {l['id']} x{l['qty']}: " + self.record_text(l["id"]) for l in self.cart_lines)
        others = ""
        pick = self.sheet.get("pick") or "any"
        if pick != "any":
            rest = [i for i in self.order_seen if i not in cart_ids and i not in self.gone]
            rest = [i for i in rest if i in self.opened][:14]
            others = "\n\nOTHER LISTINGS:\n" + "\n".join(self.record_text(i, False) for i in rest)
        payload = ("REQUEST:\n" + self.request + "\n\nREQUIREMENTS NOW:\n" + self.items_text() +
                   (f"\nBudget: {self.budget:g}" if self.budget is not None else "") +
                   f"\nPick rule: {pick} {self.sheet.get('measure') or ''}" +
                   ("\nShopper notes so far:\n" + "\n".join(self.notes[-4:]) if self.notes else "") +
                   "\n\nCART:\n" + lines + others)
        return self.ask_json(CHECK, payload, None, 900)

    def run_gate(self, stage: str) -> str | None:
        steps = self.process(stage) + [s for s in self.process("choose")]
        if not steps or not self.judge_model() or self.steps_left() <= 2:
            return None
        tools = [f"- {n}: {self.descs.get(n, '')} | {_clip(self.schemas.get(n), 500)}" for n in self.schemas]
        record = "\n".join(f"{i}. {line}" for i, line in enumerate(self.log, 1))
        payload = (GATE.replace("STAGE", "check out" if stage == "order" else "end without a purchase") +
                   "\n\nREQUEST:\n" + self.request + "\n\nSTEPS:\n" + "\n".join(f"- {s['step']}" for s in steps) +
                   f"\n\nCART LAST CHANGED AFTER RECORD LINE: {self.cart_at}" +
                   "\n\nRECORD:\n" + _clip(record, 9000) +
                   f"\n\nMESSAGES LEFT: {self.msgs_left()}  STEPS LEFT: {self.steps_left()}\n\nTOOLS:\n" + "\n".join(tools))
        for _ in range(2):
            verdict = self.ask_json("Answer the task below.", payload, None, 700)
            self.trace.append({"kind": "gate", "what": "steps " + stage, "verdict": _clip(verdict, 500)})
            call = verdict.get("call") if isinstance(verdict.get("call"), dict) else None
            if not call or call.get("tool") not in self.schemas:
                return None
            name = call["tool"]
            args = call.get("args") if isinstance(call.get("args"), dict) else {}
            if self.is_buy(name, args) or self.is_refuse(name, args):
                return None
            if name == self.roles.get("ask"):
                if self.msgs_left() <= 0:
                    return None
                self.msg_used += 1
                self.reserve = max(0, self.reserve - 1)
            observation = self.call(name, args)
            if observation is None:
                return None
            if name == self.roles.get("ask"):
                self.log.append(f"message to shopper (procedural step): {_clip(args, 300)} -> {_clip(observation, 400)}")
                text = _clip(observation, 500)
                verdict2 = self.ask_json(HEAR, f"PRODUCT: {self.items[0]['name']}\nPROPERTIES NAMED: (a procedural "
                                         f"step)\nREPLY: {text}", LIGHT, 400)
                new = [str(n) for n in verdict2.get("needs") or [] if ":" in str(n)]
                for need in new:
                    self.items[0]["needs"].append(need)
                if new:
                    return (f"Compass did a step the request asks for ({name}); the shopper replied: {text}\n"
                            "Requirements now:\n" + self.items_text() + "\nCheck the cart against them, then check out.")
            payload = payload.replace("\nMESSAGES LEFT", f"\n{len(self.log)}. (just done) {self.log[-1]}\nMESSAGES LEFT")
        return None

    # ------------------------------------------------------------------ decline gate
    def before_refuse(self) -> str | None:
        if self.time_left() < SAFE_END or self.steps_left() <= 1:
            return None
        if self.roles.get("ask") and self.msgs_left() - self.reserve > 0 and self.steps_left() > 6 and \
                self.elapsed() < INTERVIEW_BY + 30:
            summary = self.interview(spend_all=True)
            if summary:
                return "Ending held: " + summary
        pointed = self.refuse_trail()
        if pointed:
            return "Ending held. " + pointed
        if not self.fit_hunted and self.steps_left() > 6:
            self.hunt_fits()
        if self.audit_fits and self.refuse_holds < HOLDS and self.steps_left() > 4:
            leads, self.audit_fits = self.audit_fits, []
            self.refuse_holds += 1
            self.trace.append({"kind": "gate", "what": "decline held", "leads": leads})
            return ("Ending held: the audit found listings that appear to fit: " + ", ".join(leads) + ". Put the best "
                    "fitting one in the cart (respecting the pick rule) and check out. End without a purchase only if "
                    "each of them clearly fails a requirement in its own record.")
        if not self.audited and self.judge_model() and self.refuse_holds < HOLDS and self.steps_left() > 4:
            self.audited = True
            hint = self.audit()
            if hint:
                self.refuse_holds += 1
                return hint
        gate = self.run_gate("end")
        return gate

    def audit(self) -> str | None:
        ids = [i for i in self.order_seen if i not in self.gone][-40:]
        if not ids:
            return None
        payload = ("REQUEST:\n" + self.request + "\n\nREQUIREMENTS NOW:\n" + self.items_text() +
                   (f"\nBudget: {self.budget:g}" if self.budget is not None else "") +
                   "\n\nRECORDS:\n" + "\n".join(self.record_text(i, False) for i in ids))
        verdict = self.ask_json(AUDIT, _clip(payload, 30000), None, 900)
        self.trace.append({"kind": "gate", "what": "audit", "verdict": _clip(verdict, 600)})
        fits = [f for f in verdict.get("fits") or [] if isinstance(f, str) and f in self.records]
        self.audit_fits = [f for f in fits if f not in self.gone][:4]
        to_open = [o for o in verdict.get("open") or [] if isinstance(o, str) and o in self.records and o not in self.opened]
        queries = [q for q in verdict.get("search") or [] if isinstance(q, str) and q.strip()]
        found = []
        opener = self.roles.get("open") or self.roles.get("offer")
        if opener and self.steps_left() > AUDIT_OPEN + 3:
            for ident in to_open[:AUDIT_OPEN]:
                args = self.id_args(opener, ident)
                if args is None:
                    break
                observation = self.call(opener, args)
                if observation is not None:
                    found.append(f"{ident}: {self.show(observation)[:1500]}")
        searched = []
        search = self.roles.get("search")
        field = str(self.sheet.get("search_field") or "")
        if search and field and queries and self.steps_left() > 5:
            base = self.sheet.get("search_args") if isinstance(self.sheet.get("search_args"), dict) else {}
            args = dict(base)
            limit = ((self.schemas.get(search) or {}).get("properties") or {}).get(field, {}).get("maxLength")
            query = queries[0][: int(limit)] if isinstance(limit, int) else queries[0]
            args[field] = query
            observation = self.call(search, args)
            if observation is not None:
                searched.append(f"search '{query}': {self.show(observation)[:2500]}")
        if not fits and not found and not searched:
            return None
        return ("Ending held by Compass's audit. " + str(verdict.get("reason") or "") +
                (f"\nListings that may fit: {', '.join(fits)}" if fits else "") +
                ("\nOpened for you:\n" + "\n".join(found) if found else "") +
                ("\nSearched for you:\n" + "\n".join(searched) if searched else "") +
                "\nCheck these against every requirement. Buy one that fits; end without a purchase only if none does.")

    def id_args(self, name: str, ident: str) -> dict[str, Any] | None:
        schema = self.schemas.get(name) or {}
        props = schema.get("properties") or {}
        required = schema.get("required") or list(props)
        args: dict[str, Any] = {}
        for key in required:
            spec = props.get(key) or {}
            if spec.get("type") == "array":
                args[key] = [ident]
            elif spec.get("enum"):
                args[key] = spec["enum"][0]
            elif spec.get("type") in ("string", None):
                args[key] = ident
            else:
                return None
        return args or None

    def clean_refuse(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if not self.sheet.get("quiet_decline"):
            return args
        out = {}
        for key, value in args.items():
            if isinstance(value, str) and len(value) > 20:
                text = re.sub(r"\b[A-Za-z]{1,4}[-_][A-Za-z0-9]{5,}\b", "", value)
                if any(self.title_of(i) and self.title_of(i)[:20].lower() in text.lower() for i in self.records):
                    text = "I searched the store carefully and found nothing that meets every requirement."
                out[key] = text
            else:
                out[key] = value
        return out

# ------------------------------------------------------------------ search and trail parts ported from Cairn
    # The methods below adapt Cairn's public code (race-17 and race-18 winner): hunt_words, value_words, unread_pages,
    # hunt_fits, trail, early_trail, refuse_trail, find_link, shop_rows. The logic follows Cairn; the data access is
    # rewritten for Compass's record store.
    def tail(self) -> int:
        """Steps kept for the finish: a cart view and the final call."""
        return 2

    def search_plan(self) -> tuple[str, dict[str, Any], str] | None:
        """(search tool, fixed arguments, free-words field) for a search by words, or None when the schema leaves a doubt."""
        tool = self.roles.get("search")
        if not tool:
            return None
        props = (self.schemas.get(tool) or {}).get("properties") or {}
        field = str(self.sheet.get("search_field") or "")
        spec = props.get(field) if isinstance(props.get(field), dict) else None
        if not spec or spec.get("enum") or spec.get("type") not in ("string", None):
            names = [k for k, v in props.items() if isinstance(v, dict) and v.get("type") in ("string", None) and
                     not v.get("enum") and re.search(r"query|text|keyword|words|term|search|phrase|^q$", k, re.I)]
            if len(names) != 1:
                return None
            field = names[0]
        fixed = self.sheet.get("search_args") if isinstance(self.sheet.get("search_args"), dict) else {}
        base = {k: v for k, v in fixed.items() if k in props and k != field}
        for key, spec in props.items():
            if key in base or key == field or not isinstance(spec, dict) or not isinstance(spec.get("enum"), list):
                continue
            reads = [o for o in spec["enum"] if isinstance(o, str) and
                     re.search(r"text|typed|word|query|keyword|search|free", o, re.I)]
            if len(reads) == 1:
                base[key] = reads[0]
        return tool, base, field

    def shape(self, name: str, args: dict[str, Any]) -> dict[str, Any] | None:
        """Arguments cut to the schema's lengths and limits; None when a required one is missing or a value breaks it."""
        schema = self.schemas.get(name) or {}
        props = schema.get("properties") or {}
        out: dict[str, Any] = {}
        for key, value in args.items():
            spec = props.get(key)
            if not isinstance(spec, dict):
                if props:
                    continue
                out[key] = value
                continue
            if isinstance(value, str):
                limit = spec.get("maxLength")
                if isinstance(limit, int) and len(value) > limit:
                    value = value[:limit].rsplit(" ", 1)[0] if " " in value[:limit] else value[:limit]
                if spec.get("type") == "integer" and re.fullmatch(r"\d+", value):
                    value = int(value)
            if isinstance(spec.get("enum"), list) and value not in spec["enum"]:
                return None
            if isinstance(value, int) and not isinstance(value, bool):
                if isinstance(spec.get("maximum"), (int, float)) and value > spec["maximum"]:
                    return None
            out[key] = value
        if any(k not in out for k in schema.get("required") or []):
            return None
        return out

    def call_mark(self, name: str, args: dict[str, Any]) -> str:
        """One spelling per call: words in any case and spacing, a first page left out."""
        props = (self.schemas.get(name) or {}).get("properties") or {}
        kept: dict[str, Any] = {}
        for key, value in sorted(args.items()):
            spec = props.get(key) if isinstance(props.get(key), dict) else {}
            if isinstance(value, int) and not isinstance(value, bool) and \
                    value in (spec.get("minimum"), spec.get("default"), 1 if re.search(r"page", key, re.I) else None):
                continue
            kept[key] = " ".join(value.lower().split()) if isinstance(value, str) else value
        return name + json.dumps(kept, sort_keys=True, default=str)

    def paging(self, name: str, observation: Any) -> tuple[int, int, str, Any] | None:
        """(listings matched, rows shown, the argument that turns the page, its next value or None) of a search result."""
        if not isinstance(observation, dict):
            return None
        rows = next((v for v in observation.values() if isinstance(v, list) and v and
                     all(isinstance(x, (dict, str)) for x in v)), None)
        props = (self.schemas.get(name) or {}).get("properties") or {}
        arg = next((k for k, v in props.items() if isinstance(v, dict) and not v.get("enum") and
                    re.search(r"page|cursor|offset|continue|next|token|start", k, re.I)), None)
        if rows is None or arg is None:
            return None
        next_key = next((k for k, v in observation.items() if not isinstance(v, (bool, list, dict)) and
                         re.search(r"next|continue|cursor|more", k, re.I)), None)
        count = next((v for k, v in observation.items() if isinstance(v, int) and not isinstance(v, bool) and
                      re.search(r"total|matched|count|hits|found", k, re.I)), len(rows))
        if next_key is None:
            return None
        value = observation.get(next_key)
        spec = props[arg]
        if isinstance(value, str) and spec.get("type") == "integer":
            value = int(value) if re.fullmatch(r"\d+", value) else None
        if isinstance(value, int) and isinstance(spec.get("maximum"), (int, float)) and value > spec["maximum"]:
            value = None
        return count, len(rows), arg, value

    def item_ids(self, index: int) -> list[str]:
        return [i for i in self.order_seen if len(self.items) == 1 or self.owner(i) == index]

    def value_words(self, index: int) -> list[str]:
        """Requirement values of the item a words search can match: plain values, not limits, exclusions or references."""
        refs = {need for _, need in self.refs}
        item = self.items[index]
        out = []
        for need in item["musts"] + item["needs"]:
            if need in refs or ":" not in need:
                continue
            value = need.split(":", 1)[1].strip()
            if not value or len(value) > 40 or re.search(
                    r"\b(?:not|no|other than|unlike|alongside|listing|page|shown|at most|at least|under|over|below|"
                    r"above|less|more|between|within|up to|or|same|like|matching|[<>])\b|[<>]", value, re.I):
                continue
            out.append(value)
        return out

    def hunt_words(self, index: int) -> tuple[str, list[str]]:
        """The item's noun (a word of its store label no other item's label holds) and the words a further search can
        add, most promising first: a word on most rows finds nothing new and a word on one row finds little else."""
        item = self.items[index]
        mine = self.item_ids(index)
        labels = [self.category_of(i).lower() for i in mine if self.category_of(i)]
        names = [self.title_of(i) for i in mine if self.title_of(i)]
        for ident in mine:
            if ident in self.opened:
                names.append(" ".join(str(v) for v in self.attributes(ident).values()
                                      if isinstance(v, str) and len(v) <= HUNT_VALUE_CHARS))
        others = {w for n, other in enumerate(self.items) if n != index
                  for w in re.findall(r"[a-z]{4,}", (other["label"] + " " + other["name"]).lower())}
        label = max(labels, key=labels.count) if labels else (item["label"] or item["name"]).lower()
        nouns = [w for w in re.findall(r"[a-z]{4,}", label) if w not in others]
        noun = max(nouns, key=len) if nouns else ""
        if not names:
            return noun, []
        counts: dict[str, int] = {}
        for name in names:
            for word in {w.lower() for w in re.findall(r"[A-Za-z][A-Za-z'-]{2,}", name)}:
                counts[word] = counts.get(word, 0) + 1
        own = _roots(item["label"]) | _roots(item["name"])
        ceiling = max(2, int(len(names) * HUNT_COMMON))
        words = [w for w, n in counts.items() if 2 <= n <= ceiling and w != noun and _roots(w) and not _roots(w) & own]
        words.sort(key=lambda w: (-counts[w], w))
        return noun, words

    def unread_pages(self, tool: str, wanted: dict[int, set[str]], room: int) -> int:
        """Up to `room` further pages of the earlier searches that hold a required value, each item its share first.
        Cairn's data: accepted rows sat on unread pages of the episode's own searches in nearly every wrong practice
        decline; in races 13-15 about 23% of passing carts were first seen on page 2 or later."""
        spent = 0
        share = max(1, room // max(1, len(wanted)))
        for limit in (share, room):
            for index, words in wanted.items():
                stop = min(room, spent + limit)
                for name, args, seen, ids in list(self.searches):
                    text = " ".join(v for v in args.values() if isinstance(v, str)).lower()
                    if name != tool or not words & _text_words(text):
                        continue
                    if len(self.items) > 1 and ids and index not in {self.owner(i) for i in ids}:
                        continue
                    call, walked = args, {self.call_mark(name, args)}
                    while spent < stop and not self.finished and self.steps_left() > self.tail() + 2:
                        page = self.paging(name, seen)
                        if not page or page[3] is None:
                            break
                        nxt = self.shape(name, {**call, page[2]: page[3]})
                        mark = self.call_mark(name, nxt) if nxt else ""
                        if not nxt or mark in walked:
                            break
                        walked.add(mark)
                        call = nxt
                        if mark in self.read_marks:
                            seen = self.read_marks[mark]
                            continue
                        observation = self.call(name, nxt)
                        spent += 1
                        if observation is None:
                            return spent
                        seen = observation
        return spent

    def open_many(self, ids: list[str], keep: int) -> list[str]:
        """Open listings' attribute pages, one step each, while `keep` steps remain for the finish."""
        opener = self.roles.get("open") or self.roles.get("offer")
        read = []
        for ident in ids:
            if not opener or self.finished or self.steps_left() <= keep or self.elapsed() > HUNT_BY + HUNT_GRACE:
                break
            args = self.id_args(opener, ident)
            if args is None:
                break
            if self.call(opener, args) is not None:
                read.append(ident)
        return read

    def hunt_fits(self) -> list[str]:
        """Search on, without the model, before ending without a purchase (Cairn: in race 15, on 433 of 495 wrong
        declines of 13 leaders another agent held an accepted cart). Requirement values and their words find listings
        the first searches did not show; the best rows are opened so the audit judges their attributes."""
        if self.fit_hunted or self.elapsed() > HUNT_BY:
            return []
        plan = self.search_plan()
        if not plan:
            return []
        tool, base, field = plan
        keep = self.tail() + len(self.items) + 1
        spare = self.steps_left() - keep
        if spare < 3:
            return []
        self.fit_hunted = True
        held = {self.owner(l["id"]) for l in self.cart_lines if l.get("id") in self.records}
        indexes = [i for i in range(len(self.items)) if i not in held] or list(range(len(self.items)))
        wanted = {i: {w for v in self.value_words(i) for w in _value_tokens(v)} for i in indexes}
        paged = self.unread_pages(tool, wanted, min(FIT_PAGES, spare // 2))
        room = min(HUNT_STEPS, self.steps_left() - self.tail() - HUNT_KEEP)
        spent = 0
        for index in indexes if room >= 3 else []:
            values = self.value_words(index)
            noun, words = self.hunt_words(index)
            tries = [f"{v} {noun}".strip() for v in values] + [v for v in values if noun]
            tries += [f"{w} {noun}".strip() for w in words[:HUNT_FIT_WORDS]] + ([noun] if noun else [])
            share = max(2, room // max(1, len(indexes)))
            used = 0
            for query in dict.fromkeys(tries):
                if spent >= room or used >= share or self.finished or self.elapsed() > HUNT_BY + HUNT_GRACE:
                    break
                args = self.shape(tool, {**base, field: query})
                if args is None or self.call_mark(tool, args) in self.read_marks:
                    continue
                observation = self.call(tool, args)
                spent += 1
                used += 1
                if observation is None:
                    return []
        picks: list[str] = []
        for index in indexes:
            pool = [i for i in self.item_ids(index) if i not in self.opened and i not in self.gone and
                    self.in_stock(i) is not False]
            pool = [i for i in pool if self.budget is None or (self.price_of(i) or 0) <= self.budget]
            words = wanted.get(index) or set()
            pool.sort(key=lambda i: (-len(words & _text_words(self.record_text(i, False))),
                                     self.price_of(i) if self.price_of(i) is not None else float("inf")))
            picks += pool[:HUNT_FIT_OPENS]
        read = self.open_many(picks, keep)
        self.trace.append({"kind": "hunt_fits", "pages": paged, "searches": spent, "opened": len(read)})
        return read

    def rows_of_item(self, index: int) -> int:
        return len(self.item_ids(index))

    def hunt_item(self, index: int, tool: str, base: dict[str, Any], field: str, room: int) -> tuple[int, int]:
        """Up to `room` searches for one item: all its required values together (with the noun, then without), then split
        by further words; each search walked page by page while pages keep showing new rows (Cairn's hunt_item)."""
        tried = self.hunt_tried.setdefault(index, set())
        noun, _ = self.hunt_words(index)
        values = self.value_words(index)
        state = {"spent": 0, "gained": 0, "idle": 0, "issued": False}

        def run(query: str) -> bool:
            tried.add(query.lower())
            args = self.shape(tool, {**base, field: query})
            if args is None or state["spent"] >= room or self.finished:
                return False
            state["issued"] = True
            holds_values = bool(values) and all(v.lower() in query.lower() for v in values)
            found = False
            walked: set[str] = set()
            while not self.finished and self.elapsed() <= HUNT_BY + HUNT_GRACE and self.steps_left() > self.tail() + 2:
                mark = self.call_mark(tool, args)
                if mark in walked:
                    return found
                walked.add(mark)
                if mark in self.read_marks:
                    seen = self.read_marks[mark]
                    new = HUNT_PAGE_NEW
                elif state["spent"] >= room:
                    return found
                else:
                    before = self.rows_of_item(index)
                    seen = self.call(tool, args)
                    state["spent"] += 1
                    if seen is None:
                        return found
                    new = self.rows_of_item(index) - before
                    state["gained"] += new
                    state["idle"] = 0 if new else state["idle"] + 1
                ids = next((s[3] for s in reversed(self.searches) if self.call_mark(s[0], s[1]) == mark), [])
                found = found or bool(ids)
                if holds_values:
                    self.value_rows.update(ids)
                page = self.paging(tool, seen)
                if not page or page[3] is None or (new < HUNT_PAGE_NEW and page[0] > HUNT_WHOLE):
                    return found
                nxt = self.shape(tool, {**args, page[2]: page[3]})
                if nxt is None:
                    return found
                args = nxt
            return found

        key = (index, tuple(values))
        stem = self.hunt_stem.get(key)
        if stem is None:
            tries = ([" ".join(values + [noun]).strip(), " ".join(values)] if values else [])
            tries += [" ".join(values[:-1] + [noun]).strip()] if len(values) >= 2 else []
            tries += [noun] if noun else []
            stem = ""
            for query in dict.fromkeys(t for t in tries if t):
                state["issued"] = False
                if run(query):
                    stem = query
                    break
                if not state["issued"]:
                    return state["spent"], state["gained"]
            self.hunt_stem[key] = stem
        while state["spent"] < room and state["idle"] < HUNT_IDLE and not self.finished and \
                self.elapsed() <= HUNT_BY + HUNT_GRACE:
            _, words = self.hunt_words(index)
            words = [w for w in words if f"{w} {stem}".strip().lower() not in tried and w not in stem.lower().split()]
            if not words:
                break
            run(f"{words[0]} {stem}".strip())
        return state["spent"], state["gained"]

    def hunt_cheaper(self) -> str:
        """Under a cheapest rule: search on without the model while the store shows only part of what it holds (Cairn:
        searches stop at a page cap, so the lowest price of a kind can sit where no first search reaches), then open the
        cheapest unread rows below the cart's price. Returns a note for the walk."""
        mark = json.dumps([sorted(self.live_needs()), self.budget, sorted(self.gone)], default=str)
        if self.sheet.get("pick") != "cheapest" or self.hunts >= HUNT_ROUNDS or self.hunt_mark == mark:
            return ""
        plan = self.search_plan()
        if not plan or self.elapsed() > HUNT_BY:
            return ""
        tool, base, field = plan
        room = min(HUNT_STEPS, self.steps_left() - self.tail() - HUNT_KEEP - 2 * len(self.items))
        if room < 3:
            return ""
        self.hunts += 1
        self.hunt_mark = mark
        # first the unread pages of the searches already made: under a cheapest rule the lowest price often sits on
        # the last page of the first search (practice: every passing run of one task read page 3 of it)
        wanted = {i: {w for v in self.value_words(i) for w in _value_tokens(v)} or set(_words(self.items[i]["name"]))
                  for i in range(len(self.items))}
        before = self.rows_of_item(0) if len(self.items) == 1 else len(self.order_seen)
        spent = self.unread_pages(tool, wanted, min(4, room // 2))
        fresh = (self.rows_of_item(0) if len(self.items) == 1 else len(self.order_seen)) - before
        room -= spent
        share = max(3, room // max(1, len(self.items)))
        for index in range(len(self.items)):
            if spent >= room or self.finished:
                break
            used, gained = self.hunt_item(index, tool, base, field, min(share, room - spent))
            spent += used
            fresh += gained
        opened = self.open_cheaper()
        self.trace.append({"kind": "hunt", "steps": spent, "new_rows": fresh, "opened": opened})
        if not spent and not opened:
            return ""
        return (f"Compass ran {spent} more searches for the lowest price and they showed {fresh} more rows; it opened "
                f"{', '.join(opened) or 'none'} (rows priced below the cart's line). Compare them with the cart.")

    def open_cheaper(self) -> list[str]:
        """The unread buyable rows of a cart line's label priced below it, rows found by all required values first."""
        picks: list[str] = []
        if not self.cart_lines:
            # no cart yet: the cheapest unread buyable rows of each item, rows found by all required values first
            for index in range(len(self.items)):
                pool = [i for i in self.item_ids(index) if i not in self.opened and i not in self.gone and
                        self.in_stock(i) is not False and self.price_of(i) is not None and
                        (self.budget is None or (self.price_of(i) or 0) <= self.budget)]
                pool.sort(key=lambda i: self.price_of(i) or 0.0)
                lowest = [i for i in pool if i in self.value_rows][:2]
                picks += [i for i in lowest + pool[:1] if i not in picks]
        for line in self.cart_lines:
            price = self.price_of(line.get("id", ""))
            kind = self.category_of(line.get("id", "")).strip().lower()
            if price is None:
                continue
            pool = [i for i in self.order_seen if i not in self.opened and i not in self.gone and
                    self.in_stock(i) is not False and (not kind or self.category_of(i).strip().lower() == kind) and
                    (self.price_of(i) if self.price_of(i) is not None else float("inf")) < price]
            pool.sort(key=lambda i: self.price_of(i) or 0.0)
            lowest = [i for i in pool if i in self.value_rows][:1]
            pool = lowest + [i for i in pool if i not in lowest]
            picks += [i for i in pool[:2] if i not in picks]
        return self.open_many(picks[:6], self.tail() + 6)

    def live_needs(self) -> set[str]:
        return {need for item in self.items for need in item["musts"] + item["needs"]}

    def pending_refs(self) -> list[str]:
        live = self.live_needs()
        return [need for _, need in self.refs if need in live and need not in self.ref_values]

    def refuse_trail(self) -> str:
        """Once, before a decline with room left: follow what a requirement points at."""
        if self.refuse_trailed or self.steps_left() < TRAIL_ROOM or self.elapsed() > HUNT_BY or not self.pending_refs():
            return ""
        self.refuse_trailed = True
        text, _ = self.trail(HUNT_BY)
        return text + "Search for those values now and buy what fits; end without a purchase only if nothing does." if text else ""

    def early_trail(self) -> str:
        """Follow the chains of requirements that point at another listing, once, while there is room to act on them."""
        if self.steps_left() < TRAIL_ROOM or self.elapsed() > HUNT_BY or not self.pending_refs():
            return ""
        kept = (self.trails, self.trail_blocks)
        self.trails, self.trail_room = 0, EARLY_TRAILS
        try:
            text, _ = self.trail(HUNT_BY)
        finally:
            (self.trails, self.trail_blocks), self.trail_room = kept, None
        self.trace.append({"kind": "early_trail", "left": self.steps_left(), "values": list(self.ref_values.values())[:3]})
        return text

    def trail(self, stop: float | None = None) -> tuple[str, bool]:
        """Follow each pointing requirement's chain of listings with a model reading the records; open the next link,
        search for a link not seen yet, and hold the checkout while a link is still unread. Returns (text, hold)."""
        pending = self.pending_refs()
        opener = self.roles.get("offer") or self.roles.get("open")
        seen: list[str] = []
        lost: list[str] = []
        while pending and self.trails < min(6 if self.steps_left() > 15 else 4, self.trail_room or 6) and \
                self.cost < SPEND_HARD and (stop is None or self.elapsed() <= stop) and self.time_left() > SAFE_END + 30:
            self.trails += 1
            records = [self.record_text(i)[:1400] for i in reversed(self.order_seen) if i in self.opened][:45]
            for ident in reversed(self.order_seen):
                if len(records) >= 60:
                    break
                if ident not in self.opened:
                    records.append(self.record_text(ident, False)[:500])
            payload = ("REQUIREMENTS WITH A VALUE ON ANOTHER LISTING:\n" + "\n".join(pending) +
                       "\n\nRECORDS (newest first):\n" + "\n".join(records))[:45000]
            got = self.ask_json(TRAIL, payload, LIGHT, 3000)
            if not got and self.cost < SPEND_SOFT:
                got = self.ask_json(TRAIL, payload, HEAVY, 3000)
            self.trace.append({"kind": "trail", "pending": pending, "got": _clip(got, 600)})
            reach: list[str] = []
            lost = []
            gaps: list[tuple[str, str]] = []
            for chain in got.get("chains") or []:
                if not isinstance(chain, dict):
                    continue
                need = next((p for p in pending if p == str(chain.get("need") or "").strip()), None) or \
                    (pending[0] if len(pending) == 1 else None)
                if not need:
                    continue
                links = [l for l in chain.get("links") or [] if isinstance(l, dict)]
                ids = [str(l.get("id") or "").strip() for l in links]
                value = str(chain.get("value") or "").strip()
                if chain.get("done") is True and links and all(i in self.opened for i in ids) and value:
                    self.ref_values[need] = value
                    continue
                if links and all(i in self.opened for i in ids) and not str(links[-1].get("find") or "").strip():
                    self.ref_values[need] = value or f"as stated on {ids[-1]}"
                    continue
                closed = [i for i in ids if i in self.records and i not in self.opened]
                if closed:
                    reach += [i for i in closed if i not in reach]
                else:
                    gap = next((l for l, i in zip(links, ids) if i not in self.records), {})
                    about = str(gap.get("about") or "") or "the listing it points at"
                    words = str(gap.get("find") or "").strip()
                    lost.append(f"{need} -> find {about}" + (f" (search: {words})" if words else ""))
                    gaps.append((about, words))
            pending = [p for p in pending if p not in self.ref_values]
            if not reach and gaps and opener and self.steps_left() > 12 and self.link_searches < 3:
                reach = self.find_link(*gaps[0])
                if self.trace and self.trace[-1].get("kind") == "link_known":
                    seen.append(f"{self.trace[-1]['id']}: an opened page of the shop this chain names that shows the "
                                "values its description gives; read the value from it")
                    lost = lost[1:]
            if not reach or not opener or self.steps_left() < len(reach) + 4 or self.finished:
                break
            for ident in reach[:max(0, min(3, 6 - len(seen)))]:
                args = self.id_args(opener, ident)
                if args is None or self.time_left() < SAFE_END + 15:
                    break
                observation = self.call(opener, args)
                if observation is None or self.finished:
                    break
                self.opened.add(ident)
                seen.append(f"{ident}: " + self.show(observation)[:1500])
        text = ""
        if seen:
            text += "[Compass opened the pages a requirement points at]\n" + "\n".join(seen) + "\n"
        if self.ref_values:
            text += "[values read from the pages requirements point at] " + \
                "; ".join(f"{k} -> {v}" for k, v in self.ref_values.items()) + "\n"
        hold = bool(lost) and self.trail_blocks < (3 if self.steps_left() > 15 else 2) and self.steps_left() > 5
        if hold:
            self.trail_blocks += 1
            text += ("The order counts only after every page a requirement points at has been opened in full (a search "
                     "row is not enough), following each page that refers to a further listing. Still to find and open: "
                     + " | ".join(lost[:3]) + ". Search for it by its shop's name together with the values its "
                     "description gives (the shop's other listings of that kind are not it), open the row of that shop "
                     "whose page shows those values, apply the value its page gives, then check out.")
        return text, hold

    def shop_names(self, ident: str) -> list[str]:
        """Short text values of a record that can name who sells or makes it (not its id, label, title or text)."""
        out = []
        for key, value in (self.records.get(ident) or {}).items():
            if not isinstance(value, str) or not 2 < len(value.strip()) <= 40 or ID_VALUE.match(value.strip()):
                continue
            if re.search(r"categor|title|headline|^name$|product_name|descr|details|overview", key, re.I):
                continue
            if value.strip().lower() in NO_NAME or re.fullmatch(r"[\d.,\s$%]+", value.strip()):
                continue
            out.append(value.strip())
        return out

    def shop_rows(self, about: str, ids: list[str], opened: bool = False) -> list[str]:
        low = about.lower()
        out = []
        for ident in ids:
            if (ident in self.opened) != opened:
                continue
            if any(re.search(r"(?<![a-z0-9])" + re.escape(n.lower()) + r"(?![a-z0-9])", low) for n in self.shop_names(ident)):
                out.append(ident)
        return out

    def find_link(self, about: str, words: str) -> list[str]:
        """Rows of the shop a chain names, searched by the words the model gave; [] when an opened page of that shop
        already shows the description's values (traced as link_known)."""
        shop = self.shop_rows(about, list(self.records)) + self.shop_rows(about, list(self.opened), opened=True)
        named = {w for ident in shop for n in self.shop_names(ident) for w in re.findall(r"[a-z0-9]{3,}", n.lower())}
        stop = {"the", "and", "with", "from", "that", "comes", "listing", "page", "its", "for", "of"}
        values = {w for w in re.findall(r"[a-z0-9]+", (words or about).lower()) if len(w) >= 3 or w.isdigit()} - named - stop
        for ident in self.shop_rows(about, list(self.opened), opened=True):
            text = " ".join(str(v) for v in (self.records.get(ident) or {}).values()).lower()
            if values and all(re.search(r"(?<![a-z0-9])" + re.escape(v) + r"(?![a-z0-9])", text) for v in values):
                self.trace.append({"kind": "link_known", "id": ident})
                return []
        known: list[str] = []
        memo = " ".join(words.lower().split())
        plan = self.search_plan()
        if memo and memo in self.link_seen:
            known = self.shop_rows(about, self.link_seen[memo])
        elif plan and words:
            tool, base, field = plan
            for query in dict.fromkeys((words, " ".join(words.split()[:2]))):
                args = self.shape(tool, {**base, field: query})
                if args is None:
                    break
                self.link_searches += 1
                before = len(self.searches)
                observation = self.call(tool, args)
                if observation is None:
                    break
                ids = self.searches[-1][3] if len(self.searches) > before else []
                self.link_seen[memo] = ids
                known = self.shop_rows(about, ids)
                self.trace.append({"kind": "link_search", "words": query[:80], "rows": known[:4]})
                if ids:
                    break
        kind = set(re.findall(r"[a-z0-9]{3,}", (words or about).lower())) - named - stop

        def of_kind(ids: list[str]) -> list[str]:
            return [i for i in ids if not kind or kind & set(re.findall(r"[a-z0-9]{3,}", self.record_text(i).lower()))]
        return (of_kind(known) or of_kind(self.shop_rows(about, list(reversed(self.order_seen)))))[:2]

    # ------------------------------------------------------------------ forced finish
    def force_finish(self) -> None:
        if self.finished or self.dead or self.steps_left() <= 0 or self.time_left() < 3:
            return
        buy = self.roles.get("buy")
        if self.cart_view_at < self.cart_at and self.roles.get("cart_view") and self.steps_left() > 2 and \
                self.time_left() > 8:
            self.call(self.roles["cart_view"], {})
        if buy and self.cart_lines:
            args = dict(self.sheet.get("buy_args") or {}) if isinstance(self.sheet.get("buy_args"), dict) else {}
            self.call(buy, self.fill_required(buy, args))
            if self.finished:
                return
        refuse = self.roles.get("refuse")
        if refuse and not self.finished:
            args = dict(self.sheet.get("refuse_args") or {}) if isinstance(self.sheet.get("refuse_args"), dict) else {}
            args = self.fill_required(refuse, args)
            self.call(refuse, self.clean_refuse(refuse, args))

    def fill_required(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        schema = self.schemas.get(name) or {}
        props = schema.get("properties") or {}
        for key in schema.get("required") or []:
            if key in args:
                continue
            spec = props.get(key) or {}
            if spec.get("enum"):
                args[key] = spec["enum"][0]
            elif spec.get("type") in ("string", None):
                args[key] = "I searched the store and found nothing that meets every requirement."
        return args


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    agent = Compass(problem_data)
    try:
        agent.run()
    except Exception as exc:  # keep the episode alive: finish it as well as possible
        agent.trace.append({"kind": "crash", "error": f"{type(exc).__name__}: {exc}"[:500]})
        try:
            agent.force_finish()
        except Exception as again:
            agent.trace.append({"kind": "crash", "error": f"{type(again).__name__}: {again}"[:500]})
    agent.trace.append({"kind": "end", "finished": agent.finished, "steps": agent.step, "cost": round(agent.cost, 5),
                        "seconds": round(agent.elapsed(), 1), "messages": agent.msg_used})
    return agent.trace


__all__ = ["agent_main"]
