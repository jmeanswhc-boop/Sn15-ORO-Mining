"""A shopping agent for ORO Bench store worlds.

Python owns the whole flow: searching and paging, screening candidates, planning and reading questions, following
references, the cart, store events, checkout and declining. Models only read: the request and tool list once at the
start, then short calls that judge listings, read the shopper and plan questions. When the routine is stuck (nothing
buyable, an order or exit turned down, an unknown store line or tool) a model pilot drives a few moves through the
same engines and within the same limits.
"""

from __future__ import annotations

import json
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from src.agent.proxy_client import ProxyClient

QUICK_MODEL = "anthropic/claude-haiku-5.5"
READER_MODEL = "anthropic/claude-sonnet-5.5"
CHECKER_MODEL = "anthropic/claude-sonnet-5.5"
PLANNER_MODEL = "anthropic/claude-sonnet-5.5"
SPARE_MODEL = "openai/gpt-6-luna"

# wall clock: tasks get 300 s, we plan within TIME_BUDGET and wind down inside CLOSE_MARGIN
CLOSE_MARGIN = 45.0
TIME_BUDGET = 255.0
# text sizes handed to judges and screeners
ROW_TEXT = 260
CARD_CHARS = 2200
# spend lines in USD per task: Sonnet roles fall back to the fast model, exploring stops, the pilot stops
HELM_MOVES = 6
HELM_LINE = 0.065
HUNT_LINE = 0.05
SONNET_LINE = 0.03

# wording that points the reader at some other listing's page
SEE_OTHER = re.compile(r"\b(?:more|details?|info(?:rmation)?|see|read|look)\b[^.;:!?]{0,24}?\b(?:part|model|item|listing|"
                      r"product|page|code)\s*(?:no\.?|number|#)?\s*([A-Za-z0-9][A-Za-z0-9-]{2,})", re.I)
# the series prefix of a dashed code, and part / model codes in general
SERIES_HEAD = re.compile(r"\b([A-Z]{1,5}\d{1,3})-\d{2,}\b")
PART_CODE = re.compile(r"\b[A-Za-z]{2,}-?\d{3,}[A-Za-z]?\b|\b[A-Za-z]{2,}\d*-\d{2,}\b")
# relation against a referenced page -> the relation it turns into once that page has been read
POINTER_RELS = dict([
    ("same as reference", "is"),
    ("different from reference", "is not"),
    ("counterpart of reference", "is"),
    ("at least reference", "at least"),
    ("at most reference", "at most"),
    ("within reference", "at least"),
])

STORE_LINK = ProxyClient(timeout=90, max_retries=2, retry_delay=1)
MODEL_LINK = ProxyClient(timeout=75, max_retries=1, retry_delay=1)

# selector-style fields bundle several choices into one value; we never ask the shopper about them
NEVER_ASK = set(["variant", "option"])

# per-world memory kept across tasks run in the same process
WORLD_NOTES = {}

_QH_RETRY_CODES = (500, 502, 503, 504, 429)
_QH_CART_OK = r"\b(ok|okay|success\w*|added|done|accepted|applied|updated|in (the |your )?(cart|basket))\b"


# ----------------------------------------------------------------------------- small text and value tools

def _qh_numeric(x):
    return isinstance(x, (int, float))


def flat_text(text):
    s = str(text or "").lower()
    for old, new in (("&", " and "), ("_", " "), ("-", " ")):
        s = s.replace(old, new)
    return re.sub(r"[^a-z0-9 ]+", " ", s).strip()


def _qh_one(w):
    # crude singular: drop a trailing s from words longer than three letters
    if w.endswith("s") and len(w) > 3:
        return w[:-1]
    return w


def word_bag(text):
    bag = set()
    for w in flat_text(text).split():
        if len(w) >= 2:
            bag.add(_qh_one(w))
    return bag


def plain_words(text):
    """word_bag that also keeps single characters."""
    return set(map(_qh_one, flat_text(text).split()))


def stem_bag(text):
    stems = set()
    for w in word_bag(text):
        stems.add(w[:4])
    return stems


def close_names(a, b):
    """Same property name under another spelling."""
    if flat_text(a) != flat_text(b):
        left = word_bag(a)
        return bool(left) and left == word_bag(b)
    return True


def likeness(a, b):
    left = word_bag(a)
    right = word_bag(b)
    if left and right:
        return len(left & right) / len(left | right)
    return 0.0


def _qh_span_end(s, at, opener, closer):
    """Index just past the bracket closing the one at `at` (strings respected), or -1."""
    level = 0
    quoted = escaped = False
    pos, size = at, len(s)
    while pos < size:
        c = s[pos]
        pos += 1
        if quoted:
            if escaped:
                escaped = False
            elif c == "\\":
                escaped = True
            elif c == '"':
                quoted = False
        elif c == '"':
            quoted = True
        elif c == opener:
            level += 1
        elif c == closer:
            level -= 1
            if level == 0:
                return pos
    return -1


def _qh_loads(s):
    try:
        return True, json.loads(s)
    except Exception:
        return False, None


def pull_json(text):
    """First JSON object (else array) found in a reply; None when nothing parses."""
    if not text:
        return None
    body = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", body, re.S)
    if fenced:
        body = fenced.group(1).strip()
    for pair in ("{}", "[]"):
        at = body.find(pair[0])
        end = -1 if at == -1 else _qh_span_end(body, at, pair[0], pair[1])
        if end > 0:
            ok, parsed = _qh_loads(body[at:end])
            if ok:
                return parsed
    return _qh_loads(body)[1]


def brief(value, limit):
    s = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if len(s) > limit:
        s = s[:limit] + "..."
    return s


def as_amount(value):
    v = value
    if isinstance(v, str):
        hit = re.search(r"-?\d+(?:\.\d+)?", v.replace(",", ""))
        if hit is None:
            return None
        try:
            return float(hit.group(0))
        except Exception:
            return None
    if _qh_numeric(v) and v is not True and v is not False:
        return float(v)
    return None


NO_VALUE = object()


def _qh_kind(spec):
    t = spec.get("type")
    if isinstance(t, list):
        t = next((x for x in t if x != "null"), None)
    return t


def _qh_enum_fit(choices, v):
    if v in choices:
        return v
    spelled = {}
    for c in choices:
        if isinstance(c, str):
            spelled[flat_text(c)] = c
    if not spelled or not isinstance(v, str):
        return NO_VALUE
    flat = flat_text(v)
    if flat in spelled:
        return spelled[flat]
    top = max(spelled, key=lambda c: likeness(c, v))
    return spelled[top] if likeness(top, v) >= 0.5 else NO_VALUE


def _qh_text_fit(spec, v):
    if not isinstance(v, str):
        v = str(v[0]) if (isinstance(v, list) and v) else str(v)
    cap = spec.get("maxLength")
    if isinstance(cap, int) and len(v) > cap:
        head = v[:cap]
        v = head.rsplit(" ", 1)[0] if " " in head else head
    shortest = spec.get("minLength")
    too_short = isinstance(shortest, int) and len(v) < shortest
    return NO_VALUE if too_short else v


def coerce_arg(spec, value):
    """Fit a value to its argument schema without changing what it means (convert, wrap, clip, use the schema's
    spelling of an enum value); NO_VALUE when that is not possible."""
    t = _qh_kind(spec)
    v = value
    choices = spec.get("enum")
    if isinstance(choices, list) and choices:
        return _qh_enum_fit(choices, v)
    if t in ("number", "integer"):
        num = as_amount(v)
        if num is None:
            return NO_VALUE
        lowest, highest = spec.get("minimum"), spec.get("maximum")
        if _qh_numeric(lowest) and num < lowest:
            num = lowest
        if _qh_numeric(highest) and highest < num:
            num = highest
        return int(round(num)) if t == "integer" else num
    if t == "string":
        return _qh_text_fit(spec, v)
    if t == "boolean":
        if v is True or v is False:
            return v
        return {"false": False, "true": True}.get(str(v).strip().lower(), NO_VALUE)
    if t == "array":
        many = v if isinstance(v, list) else [v]
        most = spec.get("maxItems")
        if isinstance(most, int):
            many = many[:most]
        return many
    return v


def plain_arg(spec):
    """Simplest schema-valid filler for a required argument nothing else supplies."""
    t = _qh_kind(spec)
    choices = spec.get("enum")
    if isinstance(choices, list) and choices:
        return choices[0]
    if t == "boolean":
        return False
    if t == "string":
        return coerce_arg(spec, "n/a")
    if t in ("number", "integer"):
        return coerce_arg(spec, 1)
    if t != "array" or spec.get("minItems"):
        return NO_VALUE
    return []


def cart_was_refused(obs):
    """True for a one-field cart answer whose text explains a refusal rather than confirming."""
    if not isinstance(obs, dict) or len(obs) != 1:
        return False
    sole = list(obs.values())[0]
    if not isinstance(sole, str):
        return False
    return re.search(_QH_CART_OK, sole, re.I) is None


def pointer_key(ref):
    """Identity of a referenced listing: seller, category, the sorted attribute values."""
    if not isinstance(ref, dict):
        return ()
    named = ref.get("attributes")
    vals = []
    if isinstance(named, dict):
        vals = sorted(flat_text(x) for x in named.values() if x)
    seller = flat_text(ref.get("seller"))
    return seller, flat_text(ref.get("category")), tuple(vals)


def reads_as_line(value):
    if not isinstance(value, str):
        return False
    return len(value) > 24 and " " in value.strip()


def record_rows(obs, depth=0):
    if isinstance(obs, list):
        every = bool(obs)
        for x in obs:
            if not isinstance(x, dict):
                every = False
                break
        return obs if every else None
    if depth >= 2 or not isinstance(obs, dict):
        return None
    longest, size = None, 0
    for inner in obs.values():
        rows = record_rows(inner, depth + 1)
        if rows and (longest is None or len(rows) > size):
            longest, size = rows, len(rows)
    return longest


def split_result(obs):
    """(result, shopper lines carried alongside it)."""
    said = []
    if not isinstance(obs, dict):
        return obs, said
    nested = sum(isinstance(x, dict) for x in obs.values())
    # a record with an id-like field is a record, not an envelope around one
    bare_word = False
    for x in obs.values():
        if isinstance(x, str):
            trimmed = x.strip()
            if trimmed and " " not in trimmed:
                bare_word = True
    envelope = nested == 1 and len(obs) <= 3 and not bare_word
    kept = {}
    for key in obs:
        x = obs[key]
        if isinstance(x, list) and x and all(map(reads_as_line, x)):
            said += [str(y) for y in x]
        elif envelope and reads_as_line(x):
            said.append(str(x))
        else:
            kept[key] = x
    if said and len(kept) == 1:
        only = list(kept.values())[0]
        if isinstance(only, dict):
            kept = only
    return kept, said


# ----------------------------------------------------------------------------- model calls

def guarded(job):
    try:
        out = job()
    except Exception:
        out = None
    return out


def run_together(jobs, workers=3):
    """Call independent jobs in parallel; results keep job order, None where a job raised."""
    count = len(jobs)
    if count < 2:
        return list(map(guarded, jobs))
    with ThreadPoolExecutor(max_workers=min(count, workers)) as pool:
        outs = pool.map(guarded, jobs)
        return list(outs)


def _qh_has_choices(d):
    return isinstance(d, dict) and bool(d.get("choices"))


def _qh_reply_text(d):
    said = (d["choices"][0].get("message") or {}).get("content")
    if isinstance(said, list):
        pieces = []
        for part in said:
            if isinstance(part, dict):
                pieces.append(str(part.get("text") or ""))
        said = "".join(pieces)
    return said


class ModelDesk:
    def __init__(self, episode):
        self.misses = 0
        self.n_calls = 0
        self.dollars = 0.0
        self.errand = episode

    def talk(self, model, system, user, max_tokens):
        name = model
        strong = (READER_MODEL, PLANNER_MODEL, CHECKER_MODEL)
        if name != QUICK_MODEL and name in strong and SONNET_LINE <= self.dollars:
            name = QUICK_MODEL
        clock = self.errand.seconds_left
        remaining = clock()
        if CLOSE_MARGIN - 12 > remaining:
            return None
        wait = min(45, remaining - CLOSE_MARGIN + 20)
        MODEL_LINK.timeout = int(max(10, wait))
        head = system
        if name.startswith("anthropic/"):
            head = [dict(type="text", text=system, cache_control=dict(type="ephemeral"))]
        chat = [dict(role="system", content=head), dict(role="user", content=user)]
        payload = dict(model=name, messages=chat, temperature=0, max_tokens=max_tokens)
        send = lambda: MODEL_LINK.post_verbose("/inference/chat/completions", payload)
        self.n_calls = self.n_calls + 1
        reply = send()
        tries = 0
        # retry only refusals (rate limit, gateway errors), never timeouts, and only while time allows
        while tries < 2 and not _qh_has_choices(reply.data):
            err = reply.error
            code = err.get("status") if isinstance(err, dict) else None
            if code not in _QH_RETRY_CODES or clock() < CLOSE_MARGIN + 10:
                break
            tries += 1
            time.sleep(2.0 * tries)
            reply = send()
        got = reply.data
        if _qh_has_choices(got):
            spent = (got.get("usage") or {}).get("cost")
            if _qh_numeric(spent):
                self.dollars = self.dollars + float(spent)
            said = _qh_reply_text(got)
            if said:
                return str(said)
        self.misses = self.misses + 1
        why = str(getattr(reply, "error", ""))
        self.errand.log("llm_failure", model=name, error=why[:200])
        return None

    def ask_json(self, models, system, user, max_tokens):
        for name in models:
            got = pull_json(self.talk(name, system, user, max_tokens) or "")
            if got is not None:
                return got
        return None
# ----------------------------------------------------------------------------- model instructions

P_INTAKE = """You get a shopper's request and the store's tool list (OpenAI function schemas), and you set up a shopping assistant for this one task in a simulated store. Reply with ONE JSON object and no other text.

{
 "items": [  // an entry for each distinct product the shopper means to buy; most often just one
   {"name": "a plain noun phrase for that product, worded as in the request (for a list, that entry's own words)",
    "category": "the kind of product or store category being asked for, in the wording of the request",
    "quantity": 1,
    "requirements": [ // each firm demand the request places on a property of THIS product; never price, stock, quantity or budget
      {"property": "the property's name as a listing would label it", "op": "is | is not | at least | at most | one of | contains | same as reference | different from reference | counterpart of reference | at least reference | at most reference | within reference",
       "value": "the value, bound or list just as the request writes it (for 'one of', the values joined by ' | ')",
       "reference": null  // only for the relations ending in 'reference': {"category": "...", "seller": "...", "attributes": {"property": "value", ...}, "property": "which property to read on that page", "count": null or the number of listings the description says qualify, "pick": null or the rule that selects one of several (say "whichever is wider"), "relation": "the request's own words on how the wanted value relates to the value on that page", "tolerance": null or, for 'within reference', the allowed amount with its unit}
      }],
    "price_cap": null,  // the highest price the request allows for ONE unit of this product (a limit on its own price, not on the total), a bare number in the request's money unit; otherwise null
    "hints": ["property names the story suggests the shopper quietly cares about (a remark on who will use the product, or on what goes inside it, points at the property recording exactly that); empty if there are none"],
    "search_terms": ["2-4 queries for the store's search box, most specific first. A listing has to contain every word of a query, so each one stays short: the product noun with one or two required value words (a figure keeps its unit), never a bare property name, never values that only describe a referenced listing; the final query is the product noun by itself"]
   }],
 "total_cap": 12345 or null,        // the highest total price, a bare number in the unit the request counts in (cents if it says cents)
 "cheapest": false,                 // true only if the request wants the cheapest / lowest priced / cheapest total among what fits (a way of ranking near misses for a nothing-fits case does not count)
 "best": null,                      // if the request wants the best by some measure, name the measure; otherwise null
 "message_budget": null,            // how many messages the shopper permits, an integer, or null
 "call_budget": null,               // how many tool calls / steps the request permits, an integer, or null
 "in_stock_only": true,             // the request wants only listings that can really be bought
 "ask_about": ["topics the request tells the assistant to bring up with the shopper, in its own words; empty if none"],
 "fit_rules": [{"accessory": "item name", "host": "item name"}],   // rules shaped like 'make sure X fits Y'; empty if none
 "conditions": [{"if": {"item": "item name", "property": "...", "values": ["..."]}, "then": {"item": "item name", "property": "...", "op": "is | is not | at least | at most | one of", "value": "..."} or null, "otherwise": the same shape or null}],   // a demand on one product that depends on how another product turns out; empty if none
 "decline_rule": null,              // what the request says to do if nothing fits (for instance: just tell me, no substitutes), or null
 "nothing_fits": null,              // "nearest" if, for the case that nothing meets every requirement, the request asks to buy, offer or recommend the closest listing; "decline" if it says to end without buying then; null if it says neither
 "procedure": [                     // steps the request demands about HOW the task gets done (not product requirements, budgets, message or step limits, and never the rule for a nothing-fits case such as saying so or offering a near miss: that belongs in decline_rule / nothing_fits): an approval or confirmation to obtain, something to show or tell the shopper, a tool to use, an order of steps; empty if none
   {"moment": "start | before_order | before_decline | before_finish", "kind": "message | tool", "tool": "tool name or null", "goal": "what has to happen, in the request's terms", "messages": 1}  // messages: how many messages to the shopper the step uses (0 for a tool); a check the assistant runs in the store itself (stock, availability, a value) is kind "tool" with 0 messages unless the request wants to be told the outcome
 ],
 "tools": {
   "search_text": {"name": "tool", "query": "the argument for keywords", "page": "the argument that moves on to further results (a page number, a row offset, or a token from the previous result), or null",
                   "page_kind": "number | offset | token or null", "mode": {"arg": "the argument that selects keyword search", "value": "its value"} or null},
   "search_fields": {"name": "the tool that searches by category/brand/price fields, or the same tool in its field mode", "category": "arg or null", "brand": "arg or null", "price_max": "arg or null", "page": "arg or null", "page_kind": "number | offset | token or null", "mode": {"arg": "...", "value": "..."} or null} or null,
   "open": {"name": "the tool returning one whole listing", "id": "its id argument"},
   "facts": {"name": "a separate tool returning a listing's property table, if the store has it", "id": "arg"} or null,
   "compare": {"name": "a tool returning several whole listings", "ids": "its list argument"} or null,
   "cart_add": {"name": "the tool that puts a listing into the cart", "id": "arg", "qty": "quantity arg or null",
                "mode": {"arg": "the argument that chooses adding or removing", "add": "the value that adds", "remove": "the value that removes"} or null},
   "cart_set": {"name": "a tool that sets the whole cart from a list of lines, if the store has one", "lines": "its list argument", "id": "the key holding the listing id inside a line", "qty": "the key holding the quantity inside a line"} or null,
   "cart_view": {"name": "the tool that displays the cart"} or null,
   "checkout": {"name": "the tool that places the order (final)", "args": {"argument": "value"} or null},
   "decline": {"name": "the tool that ends with no purchase (final)", "remark": "a text argument or null", "args": {"argument": "value"} or null},
   "message": {"name": "the tool that sends the shopper a message", "text": "its text argument"},
   "status": {"name": "a tool that reports the allowances left"} or null
 }
}

How to read the request:
- Only firm demands count as requirements. Wishes (ideally, would be nice) and story details (who it is for, where it will stand) are NOT requirements; note what they point at under "hints". Words that merely name the product kind or its category (words belonging to the category's own name) go in "name"/"category", never in "requirements". But a value the request gives for a named property ("in the <value> style", "with <value> as its <property>", "a <property> of <value>") IS a requirement on that property, even if the value sounds like a product kind; the category then remains the product the request names.
- A list such as "3 x gadgets where the colour is red and the size is not large" is one item of quantity 3 with two requirements (colour is red; size is not large).
- A number the request gives for what a single listing holds (a pack, set or box of N pieces, a pack holding N) is a requirement on that pack or set size, while "quantity" counts how many listings to buy.
- A restatement that only spells out what a requirement already means ("zip front only, so nothing with buttons or toggles") is ONE requirement (closure: zip front), not several; invent no properties for the things it excludes.
- An invitation to ask the shopper things (an offer to reply to any question the assistant may have, a wish to be questioned first) is no procedure step: the assistant's own questions answer it. Put any subject it names into "ask_about".
- "A or B" for one property is op "one of" with value "A | B". "not X" / "anything but X" is op "is not". "at least N units" is "at least" with value "N units". Keep a value's unit and spelling as written.
- A requirement pointing at ANOTHER listing's page (one named by its category, its seller and some attribute values) that wants the same value that page states is op "same as reference"; one wanting anything except that page's value is op "different from reference"; one wanting a value that goes with that page's value rather than the value itself (its counterpart: what the page or the store pairs with it, what suits it, works with it or is compatible with it) is op "counterpart of reference"; one wanting a value no lower (or no higher) than that page's is op "at least reference" (or "at most reference"); one allowing a margin up or down from that page's figure is op "within reference", with that amount in "tolerance". The named page may itself point the reader to a further listing for that property: the assistant follows such pointers itself, so describe the page the request names. Fill "reference" with the category, the seller, the named attributes and the property to read. If the request says several of that seller's listings qualify and gives a rule for choosing one (the larger, the cheaper, the one with more of something), put the rule in "pick" and the number in "count"; "attributes" keep only plain identifying values, never the comparison. The attribute values identifying that other listing describe THAT listing alone: they are never requirements on the requested item and never search terms for it. Made-up example: "for the colour, avoid whatever Fenwick's garden hose page (length 20 m, material rubber) states" yields one requirement {"property": "colour", "op": "different from reference", "value": "", "reference": {"category": "garden hoses", "seller": "Fenwick", "attributes": {"length": "20 m", "material": "rubber"}, "property": "colour"}} and nothing on length or material.
- A requirement on one item that applies only when another item turns out a certain way ("should X's P turn out to be V, I need Y's Q to be W"; "Y's Q must be W, except when X's P is V") goes in "conditions" (in the first, "then" applies when X's P is one of the values; in the second, "otherwise" applies when it is not), never in "requirements" or "procedure".
- The category is the product kind; when the request names a store category (for instance "the category is garden tools"), use exactly its wording.
- Money: if the request gives the total in cents, "total_cap" is that number of cents. Every amount (total_cap, price_cap) uses the one unit the request counts money in: if that is cents, convert an amount written in dollars or in words ("twelve dollars forty") to cents (1240).
- Copy tool and argument names exactly from the schemas. When ONE tool ends the task either way, through an argument choosing between buying the cart and leaving with no purchase, "checkout" and "decline" both name that tool, each with "args" holding the argument value that selects it (spelled as the schema lists it). A tool described as final that closes with no purchase is "decline"; the one that checks the cart out is "checkout". If one search tool has an enum argument switching between keyword and field search, "search_text" and "search_fields" both name it with the matching mode values."""

P_TRIAGE = """Sort a store's search rows for a shopping assistant. Each row shows an id, a title, a little text, a category and a price. Decide from the row alone whether the listing MIGHT satisfy the requirements once its full property page is opened: if the row is silent on a requirement it still might ("maybe"); if the row's own words plainly contradict a required value, it cannot ("no"); if the row itself shows the required values, it does ("yes"). The category label settles what kind of product a row is: a row filed under the requested category is the right kind whatever its title suggests, so the product kind is never a reason for "no". Rows are numbered. Answer with a single JSON object and no other text: {"yes": [row numbers], "no": [row numbers]}; rows in neither list are "maybe"."""

P_ASSESS = """Decide for a shopping assistant whether store listings satisfy a shopper's requirements, reading each listing's own records.
- Met: the listing's own property for the requirement states the value. Keys are spelled many ways (spaces, underscores, prefixes, short forms) and one property may sit under several keys; any key naming the property counts. Values match by meaning: a single entry of a comma list suffices, singular equals plural, near synonyms and plain abbreviations count, letter case is irrelevant, a store value built from the required value's own words with qualifiers added satisfies it, and a value with a unit is compared after unit conversion (numbers as numbers; a bound includes its edge; a range satisfies a bound only if all of it does). When the requirement names one figure and the property gives a span or a list of figures the product works with (a supported range, several sizes), the figure inside that span or list meets it.
- The property table comes first: if some property covers the requirement, its value decides, and a different value fails even when the title claims otherwise. Only if NO key covers the requirement may the seller's notes, an options or variants field, the title or the description supply the value. If nothing states it, the requirement is unaddressed ("unsure"). A value held by a different, related property does not count (a property of one part is not the property of the whole product). When the listing carries the same property under several keys that disagree, the requirement is met if any one of those keys states the required value. When a property lists several alternatives and the listing's options field names the variant this particular listing is, that variant decides: a required value found only among the other alternatives is not met.
- "is not" and "different from" fail when the property states the excluded value; "one of" needs one of the values listed; "contains" needs the value among those stated. A demand that a feature be absent (value none, no, without) is met when no record mentions the feature.
- The category label settles the product kind: a listing labelled with the requested category is the right kind whatever its title says. Only an unlabelled listing can fail for being an accessory, a spare part or a case instead of the product.
- A compatibility demand (another product named by models or codes) is met only when this listing's own text names one of them among what it fits or works with. If the other product's page lists a series or family of compatible part codes, a listing whose own part code (often bracketed in its title) begins with that series meets it. A sentence saying some OTHER part fits a product speaks for the listing whose title carries that part's code, not for the page it appears on; "related_pages" show such pages, and what they say about this listing's own part code counts for this listing. A bare pointer to another page proves nothing.
- If the seller's notes send one property off to a DIFFERENT listing (use or follow that other listing's value for it), this listing's own value for that property does not count: a requirement on it is "unsure" unless a related page shown here states it. Notes that send away a property no requirement mentions change nothing.
- STORE FIELDS, when present, list for each required property the keys and values this store's read listings of the category use. If some of them carry exactly the required value in that field, a listing whose field shows a different value fails, even a broader or related one (the required words plus qualifiers is not a different value); a value no read listing carries may still be met by a clear synonym.
- Judge only the requirements listed; add none.
Requirements are numbered R1, R2, ... Answer with a single JSON object and no commentary: {"verdicts": [{"id": "...", "failed": ["R numbers the records clearly fail"], "unsure": ["R numbers no record addresses"], "weak": ["R numbers met only by the title, the description or an options field, not by a property or a seller note"], "loose": ["R numbers met through a synonym or a broader term instead of the value's own words; a unit conversion or a code covered by a series is not loose"]}, ...]} with an entry for every listing, omitting empty lists. A listing with nothing under "failed" or "unsure" meets all of them."""

P_QUESTIONS = """Plan which questions a shopping assistant sends a shopper. The shopper keeps some requirements to themselves: each one is a value of ONE property of one requested product, and it is revealed only to a message asking about exactly that property (its name or an obvious synonym). Only the FIRST property a message names gets an answer; asking about something the shopper does not care about earns "no preference" and wastes the message.
Input: the request, and per requested item its name, the requirements known so far, hints from the story, and a property menu listing the properties its candidate listings really have (how many listings carry each, and sample values).
Pick exactly QUESTION_COUNT questions, best first, as JSON: {"questions": [{"item": "item name exactly as given", "property": "property name spelled as in the menu (words separated by spaces)"}, ...]}.
Choosing:
1. Start with properties the story or hints point at: a remark about who uses the product, how it mounts or fits, what goes inside it, or how it should look points to the menu property recording exactly that; match each hint to the nearest menu name.
2. Next, the properties buyers of such a product usually hold a personal view on, favouring those many candidates carry (the "listings" count) and whose values vary between listings. A property with one value everywhere, a stand-in such as an option or variant selector, an identifier, a model number, a warranty or shipping note makes a poor question.
3. Skip properties a known requirement already fixes, price, stock and quantity, and any property already chosen (two spellings of one property are the same). Share questions among the items according to how uncertain each one is; the item with the strongest hints is asked first."""

P_REPLY = """Interpret a shopper's reply to a shopping assistant's question about one property of one product (sometimes two). Answer with a single JSON object and no other text:
{"kind": "need|repeat|free|split|unclear|other",
 "property": "the property the reply is about (the shopper's words, or the property asked about if that is what they mean)",
 "op": "is|is not|at least|at most|one of|same as reference|different from reference|counterpart of reference|at least reference|at most reference|within reference",
 "value": "the value, bound or list given (list values joined with ' | ')",
 "reference": null or {"category": "...", "seller": "...", "attributes": {"property": "value"}, "property": "which property to read on that page", "count": null or how many listings qualify by their account, "pick": null or the rule selecting one of several, "relation": "how they say the wanted value relates to that page's value, in their words", "tolerance": null or the margin allowed up or down from that page's figure},
 "budget": null or a number when they state a new total they can spend,
 "answer": "" or, when they asked the assistant something, a short plain reply,
 "steps": [] or steps they now demand in how the task is run (an approval to get, something to show or tell them, a tool to use before ordering), each {"moment": "before_order|before_decline|before_finish", "kind": "message|tool", "tool": null, "goal": "what has to happen", "messages": 1}}
- "need": they insist on a value, bound, exclusion or list, however phrased (settled, firm, nothing but, only X will do, anything bar X, X or Y and nothing else). "anything bar / other than X" is op "is not", value X. If they point at a DIFFERENT listing's page (named by category, seller and attribute values) and want the value it states, op "same as reference" with "reference" filled; anything except that value, "different from reference"; the value that page pairs with its own (its counterpart, what goes with it), "counterpart of reference"; no lower or no higher than that page's value, "at least reference" or "at most reference".
- "repeat": instead of answering, they restate a requirement already under KNOWN NEEDS (same property and value).
- "free": they do not mind, leave it open, anything will do, have no view on it.
- "split": they want the message broken up, its topics asked one by one, or only its first part answered (it touched more than one topic).
- "unclear": they did not follow it, or ask for it to be repeated or simplified.
- "other": anything else (a greeting, a counter-question, a budget change).
"property" is whatever property the reply itself names (their words), even if it differs from the one asked; only when the reply names none is it the property asked (of two asked, the one the value belongs to).
Explicit bound words in the reply ("at most N", "at least N", "no more than N", "under N") fix the op and the value even inside an oddly built sentence (so "the length I need sits at an at most 12 cm" is op "at most", value "12 cm").
Without bound words an insisted value is op "is", whatever turn of phrase carries it: refusing to look past a value, or to accept anything other than it, wants that very value, figures included (op "is", value "50 cm")."""

P_NEWS = """A line from the shopper came back inside a store tool's result while a shopping assistant was at work. Report what it changes as a single JSON object:
{"changes": [{"kind": "budget|drop|add|price|stock|step|other",
  "budget": null or the new highest total as a number,
  "property": null or the property of a requirement that is dropped or added,
  "op": "is|is not|at least|at most|one of", "value": null or the added requirement's value,
  "item": null or the requested product concerned, "note": "a short sentence"}]}
- "budget": a different amount the shopper can spend (up or down); copy the number exactly as the line has it, in the line's unit (cents remain cents, no conversion).
- "drop": the shopper gives up a requirement ("let X go", "drop X", "never mind X").
- "add": a new requirement appears.
- "price": the shopper reports that the chosen listing's price moved.
- "stock": the shopper reports that the chosen listing can no longer be bought.
- "step": the shopper demands a step in how the task is run (an approval, something to show or tell them, a tool to use before ordering); give it as "step": {"moment": "before_order|before_decline|before_finish", "kind": "message|tool", "tool": null, "goal": "what has to happen", "messages": 1}."""

P_POINTER = """A shopping assistant is tracing a reference and has opened one listing page; it needs that page's value for one property. Three things can happen on such a page: the property table or the seller's notes give the value; the notes tell the reader to look the property up on ANOTHER listing (named by category, seller and a few attribute values); or the notes say which values of the property belong together (one pair for this page's value, or the store's complete list of pairs).
Reply with one JSON object: {"value": "the value this page itself gives, as written, or null", "paired": "the value this page pairs with its own value, or null", "wanted": "what the RELATION from the shopper asks for: the paired value if it asks for something that goes with, suits or is compatible with the page's value and a pairing is stated; the page's own value if it asks for the same value", "redirect": null or {"category": "null unless the notes name one", "seller": "...", "attributes": {"property": "value"}, "property": "..."}}
Rules: a note that hands the WANTED property over to a different listing gives a redirect and no value (a note handing some other property elsewhere does not count as a redirect here). A note stating the value outright gives that value. Failing both, use the property table under any key naming the property, and only when no key names it the options field, title or description. A sentence that just enumerates the store's pairs does not tell you which value this page holds. Fill "paired" only from a pairing the page states, taking from a list of pairs the one containing this page's own value (a pair works in both directions); leave it null when the page states no pairing or its own value is unknown."""

P_QUERIES = """Write search-box queries for a shopping assistant. A listing matches a query only if each of its words occurs in the listing's title, brand, category or property values (stems match; very short words are ignored), and results arrive in a fixed order a few rows per page, so a query must be narrow enough that most matches satisfy the requirements. You are given the product, its requirements, the queries tried so far and some listings seen (marked as meeting the requirements or not). Answer with a single JSON object and no other text: {"queries": ["...", ...]} holding at most 5 new queries of 1-3 words, best first: the product noun as the store's titles write it with a required value as the listings would spell it, or other words the store uses for the product or the value; never repeat a query already tried."""

P_ARGS = """A shopping assistant's call to a store tool does not fit what the tool accepts. You are given the tool's name and description, its schema, the call's current arguments, what the tool is being used for, and the store's refusal if there was one. Answer with a single JSON object and no other text: {"args": {the full argument set for the call}}. Keep the current arguments unless the refusal says they are wrong; fill in what is missing with values that serve the purpose (a count of 1 unless the arguments say otherwise, the first page of results, the choice that serves the purpose). Use only argument names and values the schema permits."""

P_CHOICE = """A store tool that closes a shopping task needs an argument choosing how it closes. You are given the tool's description, the arguments still needed (each with its allowed values) and the aim. Answer with a single JSON object and no other text: {"<argument name>": "<the allowed value that achieves the aim>", ...}."""

P_PROCESS = """Check, at one moment of a shopping task, whether the assistant's process obeys the shopper's request. You are given the request, the steps it demands in how the task is run, the moment (what the assistant is about to do), the current basket, the allowances left, the store's tools (tools that end the task or change the cart are omitted; the assistant handles those) and every call so far with its result. Work out which demanded steps are due at this point, ahead of the assistant's next move, and not yet done for the present state: an approval or confirmation only counts for the basket exactly as it stands now (one given for an earlier basket does not); a step already done for this state is not due again; a step due at another moment is not due. Answer with a single JSON object and no other text: {"actions": [{"tool": "exact tool name", "args": {"argument": "value", ...}, "why": "the demanded step it fulfils"}]} in order, at most 2, or {"actions": []} if nothing is due. A message to the shopper must make sense on its own: about a basket, list every line (quantity, title, price) and the total, and state plainly what is asked. Never include a question about product properties, and never use the final allowed message for anything except a demanded step. Checks inside the store (stock, availability, a value) use the store's reading tools, not a message, unless the shopper asked to be told; the assistant reads every listing it buys anyway, so such a check is due only if the basket's listings have not been read."""

P_RANK = """Every listing below meets the shopper's requirements; the shopper wants the one that does best on a given measure (a value to push as high or as low as possible). Read each listing's value for the measure from its properties, title or description, converting units where needed so the values compare. Answer with a single JSON object and no other text: {"order": ["id", ...], "values": {"id": "the value as read"}} listing every id, best first; ids whose value cannot be read go last, and on equal values the cheaper listing comes first."""

P_REVIEW = """A shopping assistant is about to give up on one product. Take an independent look at what it searched and saw. Input: the product, its requirements, the budget, the queries it ran, and every listing it met: READ listings (properties, notes, and which requirements it marked failed or unaddressed) and rows it only saw in results. Reply with one JSON object and nothing more: {"fits_after_all": ["ids of READ listings whose own records do satisfy every requirement after all (a property value decides, a different value fails, a property no record gives is unmet)"], "unopened": ["up to 4 ids of rows not yet read that most probably satisfy the requirements, most likely first"], "queries": ["up to 2 fresh keyword queries of 1-3 words that would likely list fitting listings, none of them already run"]}. Any list may be left empty."""

P_RULE = """For a shopping assistant, test one listing against a condition the shopper set: is the listing's value for the given property one of the given values (same meaning, units converted), going by its own records? Answer with a single JSON object and no other text: {"holds": true or false, "value": "the listing's value as written, or null"}; if no record states the property, "holds" is false and "value" null."""

P_MATCH = """A shopping assistant is trying to find one specific listing, described by a category, a seller and some attribute values, among the candidates below (with their records). Answer with a single JSON object and no explanation: {"id": "the id of the listing fitting the description, or null if none does"}. The seller must agree, the category label must name the described kind, and every described attribute must agree with that listing's records (same meaning, units converted). If a listing's property table is missing, its seller notes, title and description serve as its records, and a listing whose notes state the described values fits. A listing whose records leave a described value unstated, or give a different one, does not fit: answer null instead of the nearest listing (the assistant keeps looking). If a PICK rule is given, several listings may fit; apply it to those and return the one it selects."""

P_CHECK = """This is the final test before a shopping assistant buys. Go through EVERY requirement listed and decide it from the listing's own records alone.
- Met when the listing's own property for it gives the value. Any key naming the property counts (when several keys name it and disagree, one stating the value is enough), and so do an entry of a comma list, synonyms, short forms, singular or plural, converted units, a span or list of supported figures that contains the required figure, and a store value made of the required value's own words with extra qualifiers. The store's wording decides; outside ideas about what such a product really is do not.
- When a property lists several alternatives and the options field names the variant this listing is, that variant decides; a value found only among the other alternatives does not meet it.
- With no key covering it, the seller's notes, the options field, the title or the description may give it. If nothing gives it, it is not met. A value held by some other property does not count.
- Compatibility is met when the listing names the other product among what it fits, or when the other product's page lists a series of compatible part codes and this listing's own part code begins with it.
- If the notes send a property off to a different listing's value, this listing's own value for it does not meet it.
- Exclusions and closed lists are binding. A listing labelled with the requested category is the right kind of product whatever its title says.
- Price times quantity has to fit the budget (same unit as the prices). Records showing it cannot be bought mean it fails.
Test only the listed requirements; add none of your own.
Reply with one JSON object: {"fits": true or false, "problems": ["each requirement that is clearly unmet or that no record addresses"]}."""

P_KINDS = """A store files listings under category labels, and a shopper asked for one kind of product. Among the labels seen, list each label that IS that kind of product (whose meaning matches the request's wording taken whole, including a broader label whose name combines it with a neighbouring kind). Leave out neighbouring kinds (accessories, parts, cases for it, a narrower or different type). Answer with a single JSON object and no other text: {"labels": ["label exactly as given", ...]} (empty if none matches)."""

P_KEYS = """A store tool sent back the JSON record shown below. For each role here, give the exact name of the key that plays it, or null if none does. Reply with one JSON object:
{"id": "key holding the listing's identifier", "title": "key with its name", "text": "key with its descriptive text", "price": "key with its price", "stock": "key telling whether it is buyable", "category": "key with its category label", "brand": "key with its brand", "seller": "key naming who sells it", "facts": "key whose object is its property table", "notes": "key with the seller's free-text remarks", "total": "key with a cart's total", "lines": "key with a cart's lines", "next": "key with the marker or token for the following results page", "count": "key with the overall number of matches"}"""


P_HELM = """You take over from a shopping assistant's fixed routine at a moment it cannot settle alone: it has no listing it may buy for some product, the store turned down its order or its exit, or a store line or tool showed up that the routine could not place. The routine searched, read and judged listings until now; its record of the task is below. Choose ONE move; you will see what it brought and choose again.
Return ONE JSON object and nothing else: {"think": "one or two short sentences", "move": {...}} with one of these moves:
- {"do": "search", "item": "item name", "query": "1-3 keywords"}: a store search for that product; the best new rows are read in full and judged. Every keyword must appear in a listing's own words; a query already tried brings nothing new.
- {"do": "read", "item": "item name", "ids": ["listing id", ...]}: up to 4 listings read in full and judged against that product's requirements.
- {"do": "ask", "item": "item name", "property": "one property, spelled as the listings spell it"}: the routine sends one message asking which value of that property the buyer wants (only while questions_left is above 0).
- {"do": "tool", "name": "tool name", "args": {...}}: one call of a store tool listed under other_tools (or, after a refusal, the message tool), as its description says.
- {"do": "accept", "item": "item name", "id": "listing id", "why": "which records meet each requirement"}: the listing was read and does meet every requirement after all. An independent checker reads it again, and only its agreement makes the listing a fit.
- {"do": "news", "line": "the store line, copied word for word", "kind": "budget, drop or gone", "item": "item name or null", "property": "for drop: the requirement the line lets go", "budget": "for budget: the new total as a number", "id": "for gone: the listing the line says can no longer be bought"}: a change a store line announces that the record does not show yet.
- {"do": "finish"}: hand back. The routine orders the fits now known; with none for some product it ends without buying.
How to steer:
- Read the store lines and the latest results first. A line can raise or cut the budget, let a requirement go, or say why the order or the exit was refused; act on what it says.
- Then look for the fit the routine missed: unread rows whose title and price suit; a near miss whose own records meet the requirement it was failed on; a query in the store's own words for the product or the required value.
- A requirement is met only by the listing's own records; never accept a listing that leaves a requirement unstated or states another value.
- Every search, read, ask or tool move costs store steps: stay within steps_you_may_use, never repeat a move already made, and choose "finish" when nothing promising is left."""


# ----------------------------------------------------------------------------- task state

_QA_SPLIT = r"[,;/|]"
_QA_CAMEL = r"(?<=[a-z0-9])(?=[A-Z])"
_QA_KINDS = {"string": str, "integer": (int, float), "number": (int, float), "array": list, "boolean": bool}


def _qa_count(x):
    # a real number, booleans excluded
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _qa_spaced(key):
    # property key with camelCase split, then flattened
    return flat_text(re.sub(_QA_CAMEL, " ", str(key)))


def _qa_elements(values):
    # every comma / slash / semicolon / bar piece of the given values, as word sets
    found = set()
    for val in values:
        for piece in re.split(_QA_SPLIT, str(val)):
            bag = word_bag(piece)
            if bag:
                found.add(frozenset(bag))
    return found


def _qa_want_sets(want):
    # the alternatives of a required value (split at |), as word sets
    acc = []
    for variant in want.split("|"):
        bag = word_bag(variant)
        if bag:
            acc.append(frozenset(bag))
    return acc


def _qa_names_field(key, field):
    # a store key that names the required property (same name in other spelling, or mostly the same words)
    return close_names(str(key), field) or likeness(str(key), field) >= 0.5


def _qa_any_names(offers, field):
    for one in offers:
        for kname in one.props:
            if _qa_names_field(kname, field):
                return True
    return False


def _qa_stems(words):
    return set(wd[:4] for wd in words)


def _qa_fn(tool):
    inner = tool.get("function")
    return inner if isinstance(inner, dict) else tool


def _qa_clean_table(table):
    clean = {}
    for kname, val in table.items():
        if val not in (None, ""):
            clean[str(kname)] = val
    return clean


def _qa_first_typed(fields, typ):
    for kname, val in fields.items():
        if isinstance(val, dict) and val.get("type") == typ:
            return kname
    return None


def _qa_object_array(val):
    if not isinstance(val, dict) or val.get("type") != "array":
        return False
    inner = val.get("items")
    return isinstance(inner, dict) and inner.get("type") == "object"


def _qa_mode_arg(spec):
    mode = spec.get("mode")
    if isinstance(mode, dict):
        return mode.get("arg")
    return None


def _qa_arg_spec(errand, role, arg):
    arg_props = errand.schema_of(role).get("properties") or {}
    return arg_props.get(arg) or {}


def _qa_read_listings(errand, item):
    acc = []
    for one in item.seen.values():
        if one.read_full and one.props and errand.right_kind(item, one):
            acc.append(one)
    return acc


def _qa_names_of(verdict, key):
    return set(str(elt) for elt in (verdict.get(key) or []) if elt)


def _qa_cost(offer):
    return 1e18 if offer.cost_each is None else offer.cost_each


def _qa_props_of(errand, tool):
    return (errand.arg_specs.get(tool) or {}).get("properties") or {}


def _qa_required_of(errand, tool):
    return (errand.arg_specs.get(tool) or {}).get("required") or []


def _qa_plain_typed(val, typ):
    # an argument of this type that is not a fixed choice list
    return isinstance(val, dict) and val.get("type") == typ and not val.get("enum")


def _qa_pick_named(order, pool):
    # the first required name found in the pool, else the pool's first, else None
    for name in order:
        if name in pool:
            return name
    return pool[0] if pool else None


class Offer:
    def __init__(self, id_):
        self.ref = id_
        self.heading = self.blurb = self.remarks = ""
        self.kind = self.maker = self.vendor = ""
        self.cost_each = None
        self.in_stock = None
        self.props = {}
        self.source_rec = {}
        self.read_full = False
        # True once a compare row or the listing's own page was read, so its notes are known
        self.own_page = False
        self.triage_mark = "maybe"

    def card(self, limit=CARD_CHARS):
        record = dict(id=self.ref, title=self.heading[:160])
        for kname, val in (("category", self.kind), ("brand", self.maker), ("seller", self.vendor)):
            if val:
                record[kname] = val
        for kname, val in (("price", self.cost_each), ("can_buy", self.in_stock)):
            if val is not None:
                record[kname] = val
        if self.props:
            # stores repeat one value under several spellings of the key: show each pair once
            field_rows, marks = {}, set()
            for label_name, val in self.props.items():
                mark = (_qa_spaced(label_name), json.dumps(val, sort_keys=True, default=str))
                if mark in marks:
                    continue
                marks.add(mark)
                field_rows[label_name] = val
            record["properties"] = field_rows
        if self.remarks:
            record["seller_notes"] = self.remarks[:600]
        if self.blurb:
            record["description"] = self.blurb[:400] if self.props else self.blurb[:500]
        size = len(json.dumps(record, ensure_ascii=False))
        if self.props and size > limit:
            field_rows = record["properties"]
            admit = max(6, int(len(field_rows) * limit / size))
            record["properties"] = dict(list(field_rows.items())[:admit])
        if record.get("description") and len(json.dumps(record, ensure_ascii=False)) > limit:
            record["description"] = record["description"][:200]
        return record


class Need:
    def __init__(self, prop, op, value, reference=None, origin="request"):
        self.source = origin
        self.pointer = reference if isinstance(reference, dict) else None
        self.want = str(value or "").strip()
        self.rel = str(op or "is").strip().lower()
        self.field = str(prop or "").strip()

    def line(self):
        pending = self.pointer and self.rel in POINTER_RELS
        suffix = "(not yet resolved)" if pending else self.want
        return f"{self.field}: {self.rel} {suffix}"

    def slug(self):
        label_name = self.field
        return flat_text(label_name)


class Wanted:
    def __init__(self, spec, index):
        pick = spec.get
        self.slot = index
        self.label = str(pick("name") or pick("category") or "the product").strip()
        self.kind = str(pick("category") or self.label).strip()
        amount = pick("quantity")
        self.count = int(amount) if (_qa_count(amount) and amount >= 1) else 1
        self.needs = []
        for frow in pick("requirements") or []:
            if not isinstance(frow, dict) or not frow.get("property"):
                continue
            self.needs.append(Need(frow.get("property"), frow.get("op"), frow.get("value"), frow.get("reference")))
        self.clues = [str(host) for host in (pick("hints") or []) if host]
        # most one unit may cost; a zero or negative cap means none
        ceiling = as_amount(pick("price_cap"))
        self.unit_limit = None if (ceiling is not None and ceiling <= 0) else ceiling
        terms = [str(term) for term in (pick("search_terms") or []) if term]
        self.queries = terms or [self.kind]
        # the reader puts the bare product noun last
        self.head_noun = self.queries[-1]
        # fresh per-item search and judging state
        for attr, start in (("seen", {}), ("rulings", {}), ("assessed_on", {}), ("questioned", []), ("released", []),
                            ("seeks", 0), ("dry", False), ("pick", None), ("kind_counts", {}), ("kinds_ok", None),
                            ("pages", {}), ("query_rows", {}), ("value_rows", set()), ("values_all_seen", False)):
            setattr(self, attr, start)
        # rulings: id -> {"failed", "unsure", ...}; assessed_on: id -> requirement lines judged; pages: search
        # signature -> paging state; query_rows: keyword search -> ids listed; value_rows: ids listed by value queries

    def queue_query(self, query):
        """Schedule a query for the next keyword search of this item, unless it is already planned."""
        if not query:
            return
        flat = flat_text(query)
        for planned in self.queries:
            if flat_text(planned) == flat:
                return
        self.queries.insert(min(self.seeks, len(self.queries)), query[:60])

    def too_dear(self, listing):
        price = listing.cost_each
        if self.unit_limit is None or price is None:
            return False
        return price > self.unit_limit + 1e-6

    def live_needs(self):
        acc = []
        for need in self.needs:
            if need.pointer and need.rel in POINTER_RELS:
                continue
            acc.append(need)
        return acc

    def need_lines(self):
        return list(number.line() for number in self.live_needs())

    def buried_count(self, listing):
        """Plain-value requirements whose matching field holds the value only inside a longer phrase."""
        matches = 0
        for need in self.live_needs():
            if need.rel not in ("is", "one of") or not need.want:
                continue
            want_bags = _qa_want_sets(need.want)
            own_keys = [val for kname, val in listing.props.items() if _qa_names_field(kname, need.field)]
            if want_bags and own_keys:
                pieces = _qa_elements(own_keys)
                if not any(wd in pieces for wd in want_bags):
                    matches += 1
        return matches

    def exact_hits(self, listing):
        """Plain-value requirements that one listed piece of some property states exactly."""
        if not listing.props:
            return 0
        pieces = _qa_elements(listing.props.values())
        hits = 0
        for need in self.live_needs():
            if need.rel in ("is", "one of") and need.want:
                if any(wd in pieces for wd in _qa_want_sets(need.want)):
                    hits += 1
        return hits

    def good_offers(self, include_unsure=False, cheapest=False, strong_only=False):
        lines_now = set(self.need_lines())
        certain, doubtful = [], []
        for offer_id, offer in self.seen.items():
            if not offer.read_full:
                continue
            ruling = self.rulings.get(offer_id)
            if ruling is None or ruling["failed"] or self.too_dear(offer):
                continue
            if strong_only and ruling.get("weak"):
                continue
            if not (self.assessed_on.get(offer_id, set()) >= lines_now):
                continue
            if ruling["unsure"] or self.loose_words(offer_id):
                doubtful.append(offer)
            else:
                certain.append(offer)
        if cheapest:
            # by price; at equal price a listing stating the value as written goes before one burying it in a phrase
            def order(o):
                return _qa_cost(o), self.buried_count(o) > 0
        else:
            # table evidence first, then fewer weak readings, then price
            def order(o):
                return -self.exact_hits(o), len(self.rulings[o.ref].get("weak") or ()), _qa_cost(o)
        certain.sort(key=order)

        def doubt(o):
            return len(self.rulings[o.ref]["unsure"]) + len(self.loose_words(o.ref)), _qa_cost(o)
        doubtful.sort(key=doubt)
        if include_unsure and not certain:
            return doubtful
        return certain

    def loose_words(self, lid):
        """Word-only requirement lines this listing meets only through a synonym or broader word (not its own words)."""
        ruling = self.rulings.get(lid) or {}
        line_set = set(ruling.get("loose") or ())
        if not line_set:
            return []
        offer = self.seen.get(lid)
        acc = []
        for need in self.live_needs():
            line_text = need.line()
            if line_text not in line_set or need.rel not in ("is", "one of"):
                continue
            if not need.want or re.search(r"\d", need.want) or need.source == "fit":
                continue
            if offer is not None and self.carries_words(offer, need):
                continue
            acc.append(line_text)
        return acc

    @staticmethod
    def carries_words(listing, req):
        """The listing's own field holds all words of the value, or one listed piece uses only the value's words."""
        own_keys = [str(val) for kname, val in listing.props.items() if _qa_names_field(kname, req.field)]
        wordset = set()
        for line_text in own_keys:
            wordset |= word_bag(line_text)
        pieces = [word_bag(issue) for line_text in own_keys for issue in re.split(_QA_SPLIT, line_text) if word_bag(issue)]
        for option in req.want.split("|"):
            bag = word_bag(option)
            if not bag:
                continue
            if bag <= wordset:
                return True
            for piece in pieces:
                if piece <= bag:
                    return True
        return False


class Errand:
    def __init__(self, problem_data):
        self.t0 = time.monotonic()
        env = problem_data["environment"]
        self.bind = dict(env["binding"])
        self.view = env["policy_view"]
        look = self.view.get
        self.ask_text = str(look("query") or "")
        self.first_view = look("opening")
        self.tool_list = list(filter(lambda term: isinstance(term, dict), look("tools") or []))
        self.arg_specs = {}
        for tool_entry in self.tool_list:
            func = _qa_fn(tool_entry)
            if isinstance(func, dict) and func.get("name"):
                self.arg_specs[func["name"]] = func.get("parameters") or {"type": "object", "properties": {}}
        self.move_cap = int(look("max_steps") or 30)
        self.done = bool(look("finished"))
        self.desk = ModelDesk(self)
        self.world_sig = "|".join(sorted(self.arg_specs))
        notes = WORLD_NOTES.get(self.world_sig) or {}
        self.key_maps = dict(notes.get("maps") or {})
        # compare batches start at 4: stores refuse more without naming a limit, and each refusal costs a step
        self.side_max = int(notes.get("compare_limit") or 4)
        # compare rows lacked the property table that a separate facts tool provides
        self.side_thin = bool(notes.get("thin_compare"))
        # does the store need every query word (None = not known yet)
        self.strict_words = notes.get("all_words")
        blank = dict(
            moves=0, halted=False, events=[], role_map={}, wants=[], limit=None, best_by=None, lowest_price=False,
            approval_due=False,     # the procedure wants a shopper message before ordering
            demands=[], journal=[],  # journal: calls so far, as the process gate reads them
            gate_count={}, approved_key=None, msgs_used=0, msg_cap=None, stock_only=True, raise_topics=[],
            fit_links=[], seen_news=set(), news_queue=[], basket_owner={}, basket={}, rows_per_page=0,
            pages_read=set(), exit_reason="",
            style="full",           # "ask" ends after the questions (engine test)
            query_tally={},         # sent query -> reported match count
            news_log=[],            # store lines in arrival order
            odd_news=[],            # lines where the event reader saw no change
            latest_refusal=None,
            review_due=True,        # pilot may review the record before a decline
            tools_called=set())
        for attr, start in blank.items():
            setattr(self, attr, start)

    # -- clock, counters, log
    def seconds_left(self):
        spent = time.monotonic() - self.t0
        return TIME_BUDGET - spent

    def moves_left(self):
        ceiling = self.move_cap
        return ceiling - self.moves

    def log(self, kind, **fields):
        stamp = round(time.monotonic() - self.t0, 1)
        tool_entry = dict(kind=kind, step=self.moves, t=stamp)
        tool_entry.update(fields)
        self.events.append(tool_entry)

    def keep_world(self):
        WORLD_NOTES[self.world_sig] = dict(maps=self.key_maps, compare_limit=self.side_max,
                                          thin_compare=self.side_thin, all_words=self.strict_words)

    def over_budget_spend(self):
        """Spend reached the exploring cap."""
        return HUNT_LINE <= self.desk.dollars

    def opens_many(self):
        """One compare step reads several listings with their tables."""
        if self.side_thin or getattr(self, "side_off", False):
            return False
        return bool(self.role_map.get("compare"))

    # -- store calls
    def use_tool(self, name, args):
        """Call a tool with schema-shaped arguments; on an argument refusal try one repaired call. The result's
        "bad_args" says whether the refusal stands."""
        call = self.shape_args(name, args)
        outcome = self.step_env(name, call)
        if not outcome["error"] or not self.bad_args_reply(outcome):
            return outcome
        repaired = self.fix_args(name, call, outcome["obs"])
        if repaired is not None and repaired != call:
            outcome = self.step_env(name, repaired)
        outcome["bad_args"] = bool(outcome["error"]) and self.bad_args_reply(outcome)
        return outcome

    @staticmethod
    def bad_args_reply(out):
        blob = " ".join([f"{out['error']}", brief(out["obs"], 400)])
        match = re.search(r"invalid|missing|required|argument|parameter|schema|malformed|validation|unexpected|must be",
                        blob, re.I)
        return match is not None

    def roles_of(self, name):
        found = []
        for job, conf in self.role_map.items():
            if isinstance(conf, dict) and conf.get("name") == name:
                found.append(job)
        return found

    def shape_args(self, name, args):
        """Fit arguments to the schema: convert or clip values, move an unknown-named one onto a missing required
        argument (else drop it), and fill required arguments still missing."""
        sch = self.arg_specs.get(name) or {}
        arg_props = sch.get("properties") or {}
        if not arg_props:
            return args
        shaped, loose_set = {}, {}
        for kname, val in args.items():
            if val is None:
                continue
            if kname in arg_props:
                fitted = coerce_arg(arg_props[kname], val)
                if fitted is not NO_VALUE:
                    shaped[kname] = fitted
            else:
                loose_set[kname] = val
        lacking = [kk for kk in (sch.get("required") or []) if kk not in shaped]
        for target in list(lacking):
            typ = _QA_KINDS.get((arg_props.get(target) or {}).get("type"), object)

            # same-typed values first (a count never stands in for an id), then the closest name
            def rank(rival, typ=typ, target=target):
                return not isinstance(loose_set[rival], typ), -likeness(str(rival), target)
            for rival in sorted(loose_set, key=rank):
                fitted = coerce_arg(arg_props.get(target) or {}, loose_set[rival])
                if fitted is NO_VALUE:
                    continue
                shaped[target] = fitted
                loose_set.pop(rival)
                lacking.remove(target)
                break
        if lacking:
            shaped.update(self.supply_args(name, shaped, lacking))
        return shaped

    def supply_args(self, name, have, missing, refusal=None):
        """Fill required arguments: schema default, else a cheap model call, else the plainest allowed value; the
        result is remembered for this tool. Choice arguments of an ending tool stay for fixed_args to decide."""
        stash = self.__dict__.setdefault("arg_fill", {})
        arg_props = (self.arg_specs.get(name) or {}).get("properties") or {}
        is_ending = bool({"checkout", "decline"} & set(self.roles_of(name)))
        filled, ask = {}, []
        for kname in missing:
            conf = arg_props.get(kname) or {}
            if (name, kname) in stash:
                filled[kname] = stash[(name, kname)]
            elif "default" in conf:
                filled[kname] = conf["default"]
            elif is_ending and isinstance(conf.get("enum"), list):
                pass
            else:
                ask.append(kname)
        if not ask:
            return filled
        reply = (self.model_args(name, have, refusal) or {}).get("args")
        if not isinstance(reply, dict):
            reply = {}
        for kname in ask:
            conf = arg_props.get(kname) or {}
            val = NO_VALUE
            if reply.get(kname) is not None:
                val = coerce_arg(conf, reply[kname])
            if val is NO_VALUE:
                val = plain_arg(conf)
            if val is not NO_VALUE:
                filled[kname] = val
                stash[(name, kname)] = val
        self.log("args_filled", tool=name, args=brief(filled, 200))
        return filled

    def model_args(self, name, have, refusal):
        parts = [
            f"TOOL: {name}",
            f"DESCRIPTION: {self.tool_blurb(name)}",
            "SCHEMA: " + json.dumps(self.arg_specs.get(name) or {}, ensure_ascii=False),
            "ARGUMENTS OF THE CALL: " + json.dumps(have, ensure_ascii=False, default=str),
            "USED HERE AS: " + (", ".join(self.roles_of(name)) or "unknown"),
        ]
        if refusal is not None:
            parts.append("THE STORE'S REFUSAL: " + brief(refusal, 400))
        reply = self.desk.ask_json([QUICK_MODEL, SPARE_MODEL], P_ARGS, "\n".join(parts), 400)
        if isinstance(reply, dict):
            return reply
        return None

    def fix_args(self, name, args, refusal):
        """Corrected arguments after an argument refusal (two repairs per tool per task); None means no retry."""
        numbers = self.__dict__.setdefault("arg_repairs", {})
        used = numbers.get(name, 0)
        if used >= 2 or self.moves_left() < 2:
            return None
        numbers[name] = used + 1
        arg_props = (self.arg_specs.get(name) or {}).get("properties") or {}
        said = brief(refusal, 400)
        cited_keys = []
        for kname in arg_props:
            if kname in args:
                continue
            if re.search(r"(?<![A-Za-z0-9_])" + re.escape(kname) + r"(?![A-Za-z0-9_])", said):
                cited_keys.append(kname)
        self.log("args_refused", tool=name, args=brief(args, 200), refusal=said[:200])
        if cited_keys:
            extra = self.supply_args(name, args, cited_keys, refusal)
            return dict(args, **extra)
        reply = (self.model_args(name, args, refusal) or {}).get("args")
        if isinstance(reply, dict) and reply:
            return self.shape_args(name, reply)
        return None

    def step_env(self, name, args):
        """Send one action to the store; returns {"obs", "lines", "error", "ok"}."""
        res = dict(obs=None, lines=[], error=None, ok=False)

        def fail(why):
            res["error"] = why
            return res
        if self.done or self.halted or name not in self.arg_specs:
            return fail("unavailable")
        enders = set((self.role_map.get(r) or {}).get("name") for r in ("checkout", "decline"))
        steps_left = self.moves_left()
        if steps_left < 1 or (steps_left < 2 and name not in enders):
            return fail("no steps left")
        packet = dict(self.bind, step=self.moves + 1, action={"name": name, "args": args})
        talk_tool = (self.role_map.get("message") or {}).get("name")

        def landed(data):
            self.tools_called.add(name)
            step_no = data.get("step")
            self.moves = step_no if isinstance(step_no, int) else self.moves + 1
            if data.get("finished"):
                self.done = True
            body_obs, said_lines = split_result(data.get("observation"))
            for elt in said_lines:
                if elt not in self.seen_news:
                    self.news_queue.append(elt)
            res.update(obs=body_obs, lines=said_lines)
            problem = body_obs.get("error") if isinstance(body_obs, dict) else None
            if isinstance(problem, str) and len(body_obs) <= 2:
                res["error"] = problem
            else:
                res["ok"] = True
            self.log("call", tool=name, args=brief(args, 160), result=brief(body_obs, 200), lines=said_lines[:3])
            spoke = name == talk_tool
            note = dict(step=self.moves, tool=name, args=brief(args, 300 if spoke else 120), result=brief(body_obs, 160))
            if said_lines:
                note["shopper_lines"] = [elt[:200] for elt in said_lines[:3]]
            if spoke:
                note["basket_then"] = list(map(list, self.basket_key()))
            self.journal.append(note)
            return res
        tries = 0
        while tries < 3:
            tries += 1
            remaining = self.seconds_left()
            if remaining < 3:
                return fail("no time")
            STORE_LINK.timeout = int(max(4, min(90, remaining - 1)))
            reply = STORE_LINK.post_verbose("/environment/call", packet)
            data = reply.data
            if isinstance(data, dict) and ("observation" in data or "finished" in data):
                return landed(data)
            err = reply.error or {}
            code = err.get("status") if isinstance(err, dict) else None
            detail = str(err.get("body") if isinstance(err, dict) else err)[:300]
            self.log("env_failure", status=code, body=detail)
            if code == 409 and "finished" not in detail:
                hit = re.search(r"expected (\d+)", detail)
                if not hit:
                    return fail("conflict")
                self.moves = int(hit.group(1)) - 1
                packet["step"] = self.moves + 1
                continue
            if code == 409:
                self.done = True
                return fail("finished")
            if code == 402:
                self.halted = True
                return fail("payment")
            if code in (403, 422):
                return fail(f"refused {code}")
        return fail("env failure")

    # -- which record keys play which role
    def key_roles(self, kind, sample):
        signature = kind + ":" + ",".join(sorted(sample.keys()))
        mapping = self.key_maps.get(signature)
        if mapping is None:
            prefix = kind + ":"
            for other_sig, rival in self.key_maps.items():
                if not rival or not other_sig.startswith(prefix):
                    continue
                hits = len([kk for kk in rival.values() if kk in sample])
                if hits >= max(2, int(0.6 * len(rival))):
                    mapping = rival
                    self.key_maps[signature] = rival
                    break
        if mapping is not None:
            # a map learned while text fields were empty lacks them: remap once per newly filled key
            asked = self.__dict__.setdefault("remapped", set())
            newer = []
            for kname, val in sample.items():
                if kname in mapping.values() or (kname, signature) in asked:
                    continue
                if (isinstance(val, str) and len(val) > 25) or (isinstance(val, dict) and val):
                    newer.append(kname)
            if not newer or all(cond in mapping for cond in ("notes", "text", "facts")):
                return mapping
            asked.update((kname, signature) for kname in newer)
        display = {}
        for kname, val in sample.items():
            if isinstance(val, (dict, list)):
                val = brief(val, 160)
            if isinstance(val, str) and len(val) > 160:
                val = val[:160] + "..."
            display[kname] = val
        reply = self.desk.ask_json([QUICK_MODEL, SPARE_MODEL], P_KEYS, json.dumps(display, ensure_ascii=False), 700)
        found = dict(mapping or {})
        if isinstance(reply, dict):
            for job, kname in reply.items():
                if job in found or not isinstance(kname, str) or kname not in sample:
                    continue
                found[job] = kname

        def first(test):
            hits = [kk for kk, value_v in sample.items() if test(kk, value_v)]
            return hits[:1]
        # fall back on the record's shape
        if "facts" not in found:
            match = first(lambda kk, value_v: isinstance(value_v, dict) and value_v
                        and all(isinstance(elt, (str, int, float)) or elt is None for elt in value_v.values()))
            if match:
                found["facts"] = match[0]
        if "id" not in found:
            match = first(lambda kk, value_v: isinstance(value_v, str) and re.fullmatch(r"[A-Za-z]{0,3}-?[0-9a-f]{8,}", value_v))
            if match:
                found["id"] = match[0]
        if "price" not in found:
            match = first(lambda kk, value_v: _qa_count(value_v) and "price" in kk.lower())
            if match:
                found["price"] = match[0]
        self.key_maps[signature] = found
        self.keep_world()
        return found

    def offer_from(self, rec, kind):
        role_names = self.key_roles(kind, rec)
        id_key = role_names.get("id")
        if not id_key or not isinstance(rec.get(id_key), (str, int)):
            return None
        offer = Offer(str(rec[id_key]))
        offer.source_rec = rec
        for job, field_name in (("title", "heading"), ("text", "blurb")):
            kname = role_names.get(job)
            if kname and isinstance(rec.get(kname), str):
                setattr(offer, field_name, rec[kname])
        price_key = role_names.get("price")
        if price_key:
            offer.cost_each = as_amount(rec.get(price_key))
        stock_key = role_names.get("stock")
        if stock_key and isinstance(rec.get(stock_key), bool):
            offer.in_stock = rec[stock_key]
        for job, field_name in (("category", "kind"), ("brand", "maker"), ("seller", "vendor")):
            kname = role_names.get(job)
            if kname and isinstance(rec.get(kname), str):
                setattr(offer, field_name, rec[kname])
        facts_key = role_names.get("facts")
        if facts_key and isinstance(rec.get(facts_key), dict):
            offer.props = _qa_clean_table(rec[facts_key])
        notes_key = role_names.get("notes")
        if notes_key and isinstance(rec.get(notes_key), str):
            offer.remarks = rec[notes_key]
        return offer

    def absorb(self, item, listing, opened):
        prior = item.seen.get(listing.ref)
        if prior is None:
            prior = listing
            item.seen[listing.ref] = prior
        else:
            for field_name in ("heading", "blurb", "kind", "maker", "vendor", "remarks"):
                newer = getattr(listing, field_name)
                if newer and (opened or not getattr(prior, field_name)):
                    setattr(prior, field_name, newer)
            if listing.cost_each is not None:
                prior.cost_each = listing.cost_each
            if listing.in_stock is not None:
                prior.in_stock = listing.in_stock
            if listing.props:
                if prior.props != listing.props:
                    # changed records: judge again
                    item.assessed_on.pop(prior.ref, None)
                prior.props = listing.props
            if opened and listing.remarks and listing.remarks != prior.remarks:
                item.assessed_on.pop(prior.ref, None)
            prior.own_page = prior.own_page or listing.own_page
            prior.source_rec = listing.source_rec or prior.source_rec
        if opened:
            prior.read_full = True
            self.pages_read.add(prior.ref)
        cat_label = prior.kind
        if cat_label:
            item.kind_counts[cat_label] = item.kind_counts.get(cat_label, 0) + 1
        return prior

    # -- setup from the request and the tool list
    def read_request(self):
        ask = self.desk.ask_json
        prompt = self._qa_intake_prompt()
        intake = ask([READER_MODEL, SPARE_MODEL], P_INTAKE, prompt, 2500)
        usable = isinstance(intake, dict) and isinstance(intake.get("tools"), dict)
        if not usable:
            intake = ask([READER_MODEL, QUICK_MODEL], P_INTAKE, prompt, 2500)
        if not isinstance(intake, dict):
            self.log("setup_failed")
            return False
        self.brief_plan = intake
        P = intake.get
        offered = P("tools") if isinstance(P("tools"), dict) else {}
        known_tools = self.arg_specs
        self.role_map = {job: conf for job, conf in offered.items()
                         if isinstance(conf, dict) and isinstance(conf.get("name"), str) and conf["name"] in known_tools}
        self.settle_roles()
        absent = [job for job in ("search_text", "open", "checkout", "decline", "message") if job not in self.role_map]
        for job in absent:
            self.log("role_missing", role=job)
        listed = P("items") if isinstance(P("items"), list) else []
        self.wants = [Wanted(entry, slot_no) for slot_no, entry in enumerate(listed) if isinstance(entry, dict)]
        if len(self.wants) == 0:
            fallback = {"name": "the product", "category": "", "search_terms": [self.ask_text[:60]]}
            self.wants = [Wanted(fallback, 0)]
        for product in self.wants:
            self._qa_tidy_item(product)

        def number(key):
            raw = P(key)
            return raw if _qa_count(raw) else None
        self.limit = as_amount(P("total_cap"))
        self.lowest_price = bool(P("cheapest"))
        when_none = str(P("nothing_fits") or "").strip().lower()
        # "nearest": with no full fit, order the closest listing
        self.closest_ok = when_none == "nearest"
        # the request says how to end when nothing fits, so its decline is stated
        self.exit_stated = when_none == "decline" or bool(P("decline_rule"))
        measure = P("best")
        self.best_by = None
        if isinstance(measure, str) and measure:
            self.best_by = measure
        msgs = number("message_budget")
        self.msg_cap = int(msgs) if (msgs is not None and msgs >= 0) else None
        calls = number("call_budget")
        if calls is not None and 0 < calls < self.move_cap:
            self.move_cap = int(calls)
        self.stock_only = bool(P("in_stock_only", True))
        self.raise_topics = list(map(str, filter(None, P("ask_about") or [])))
        self.fit_links = list(filter(lambda d: isinstance(d, dict), P("fit_rules") or []))
        self.rules = self.parse_conditions(P("conditions"))
        proc = [d for d in (P("procedure") or []) if isinstance(d, dict) and d.get("goal")]
        self.demands = proc[:6]
        for d in self.demands:
            d["moment"] = str(d.get("moment") or "before_finish").lower()
            is_msg = str(d.get("kind") or "").lower() == "message"
            d["kind"] = "message" if is_msg else "tool"
        self.approval_due = any(d["moment"] in ("before_order", "before_finish") and d["kind"] == "message"
                                for d in self.demands)
        summary = []
        for each in self.wants:
            frow = [each.label, each.count, each.need_lines(), each.clues]
            if each.unit_limit is not None:
                frow.append(each.unit_limit)
            summary.append(tuple(frow))
        tool_names = {}
        for job, conf in self.role_map.items():
            tool_names[job] = conf.get("name")
        self.log("plan", items=summary, cap=self.limit, cheapest=self.lowest_price, nearest=self.closest_ok,
                 messages=self.msg_cap, steps=self.move_cap, roles=tool_names)
        return True

    def _qa_intake_prompt(self):
        described = []
        for tool_entry in self.tool_list:
            func = _qa_fn(tool_entry)
            described.append(dict(name=func.get("name"), description=func.get("description", ""),
                                  parameters=func.get("parameters")))
        text = "REQUEST:\n" + self.ask_text
        if self.first_view is not None:
            text = text + "\n\nOPENING OBSERVATION:\n" + brief(self.first_view, 1500)
        return text + "\n\nTOOLS:\n" + json.dumps(described, ensure_ascii=False)

    def _qa_tidy_item(self, item):
        # values the request gives for a referenced page are that page's, not this item's
        ref_pairs = set()
        for need in item.needs:
            ptr = need.pointer
            if ptr and isinstance(ptr.get("attributes"), dict):
                for kk, value_v in ptr["attributes"].items():
                    ref_pairs.add((flat_text(kk), flat_text(value_v)))
        if ref_pairs:
            item.needs = [number for number in item.needs if number.pointer or (flat_text(number.field), flat_text(number.want)) not in ref_pairs]
            kept_queries = []
            for qtext in item.queries:
                fq = flat_text(qtext)
                if not any(value_v and value_v in fq for _, value_v in ref_pairs):
                    kept_queries.append(qtext)
            item.queries = kept_queries or [item.kind]
        # a plain "is" value made only of the category's words restates the kind; a word beyond it narrows the kind
        noun_words = word_bag(item.kind)
        retained = []
        for need in item.needs:
            bag = word_bag(need.want)
            if need.pointer or need.rel != "is" or not (bag and bag <= noun_words):
                retained.append(need)
        item.needs = retained
        # every stated value is also a query on its own (fewer words match more listings)
        for need in item.needs:
            if need.rel not in ("is", "one of", "contains") or not need.want:
                continue
            for variant in need.want.split("|"):
                variant = variant.strip()
                variants = [variant, variant.replace("-", " ")] if "-" in variant else [variant]
                for phrase in variants:
                    if not phrase or len(phrase) > 40:
                        continue
                    flat = flat_text(phrase)
                    if any(flat == flat_text(term) for term in item.queries):
                        continue
                    item.queries.append(phrase)

    def settle_roles(self):
        """Align cart and ending roles with the actual schemas: a list-of-lines cart tool becomes cart_set, unknown
        argument names give way to the schema's own, a choice-needing ending tool covers both endings."""
        rm = self.role_map
        for job in ("cart_add", "cart_set"):
            conf = rm.get(job)
            if conf:
                self._qa_settle_cart(job, conf)
        # other roles: an argument name the schema lacks becomes its argument of the right type
        plain_roles = (("message", "text", "string"), ("open", "id", "string"), ("facts", "id", "string"),
                       ("compare", "ids", "array"), ("search_text", "query", "string"))
        for job, slot_name, typ_name in plain_roles:
            conf = rm.get(job)
            if not conf:
                continue
            sch = self.arg_specs.get(conf["name"]) or {}
            arg_props = sch.get("properties") or {}
            if arg_props and conf.get(slot_name) not in arg_props:
                skip_arg = _qa_mode_arg(conf)
                fitting = [kk for kk, vv in arg_props.items() if _qa_plain_typed(vv, typ_name) and kk != skip_arg]
                conf[slot_name] = _qa_pick_named(sch.get("required") or [], fitting)
        # an ending tool with a required choice ends the task either way
        for held_role, rival in (("checkout", "decline"), ("decline", "checkout")):
            conf = rm.get(held_role)
            if not conf or rival in rm:
                continue
            sch = self.arg_specs.get(conf["name"]) or {}
            arg_props = sch.get("properties") or {}
            choice = [kk for kk in sch.get("required") or [] if isinstance((arg_props.get(kk) or {}).get("enum"), list)]
            if choice:
                rm[rival] = {"name": conf["name"]}

    def _qa_settle_cart(self, job, conf):
        tool_name = conf["name"]
        arg_props = _qa_props_of(self, tool_name)
        lists = [kk for kk in arg_props if _qa_object_array(arg_props[kk])]
        if lists:
            entry_props = arg_props[lists[0]]["items"].get("properties") or {}
            keep_named = job == "cart_set"
            if keep_named and conf.get("id") in entry_props:
                id_key = conf.get("id")
            else:
                id_key = _qa_first_typed(entry_props, "string")
            if keep_named and conf.get("qty") in entry_props:
                qty_key = conf.get("qty")
            else:
                qty_key = _qa_first_typed(entry_props, "integer")
            if id_key:
                self.role_map["cart_set"] = dict(name=tool_name, lines=lists[0], id=id_key, qty=qty_key)
            if job == "cart_add":
                self.role_map.pop("cart_add", None)
            return
        if job == "cart_set":
            return
        if conf.get("id") not in arg_props:
            texts = [kk for kk, vv in arg_props.items() if _qa_plain_typed(vv, "string")]
            conf["id"] = _qa_pick_named(_qa_required_of(self, tool_name), texts)
        if conf.get("qty") and conf["qty"] not in arg_props:
            conf["qty"] = _qa_first_typed(arg_props, "integer")
        if not conf.get("qty"):
            # an unclaimed whole-number argument is the quantity, whatever it is called
            skip = (conf.get("id"), _qa_mode_arg(conf))
            order = list(_qa_required_of(self, tool_name)) + list(arg_props)
            found = None
            for kk in order:
                if kk in skip:
                    continue
                kspec = arg_props.get(kk)
                if isinstance(kspec, dict) and kspec.get("type") in ("integer", "number"):
                    found = kk
                    break
            conf["qty"] = found
        if not conf.get("id"):
            self.role_map.pop("cart_add", None)

    def tool_blurb(self, name):
        for tool_entry in self.tool_list:
            func = _qa_fn(tool_entry)
            if not isinstance(func, dict) or func.get("name") != name:
                continue
            return str(func.get("description") or "")
        return ""

    def preset_args(self, role):
        """Arguments choosing this ending on its tool: the reader's values snapped to the schema's listed values, and
        any required choice still open picked by one cheap model call."""
        conf = self.role_map.get(role) or {}
        sch = self.schema_of(role)
        arg_props = sch.get("properties") or {}
        selected = {}
        given = conf.get("args")
        for kname, val in (given.items() if isinstance(given, dict) else []):
            if kname not in arg_props or val in (None, ""):
                continue
            allowed = arg_props[kname].get("enum")
            if isinstance(allowed, list) and allowed and val not in allowed:
                closest = max(allowed, key=lambda elem: likeness(str(elem), str(val)))
                if likeness(str(closest), str(val)) < 0.5:
                    # nothing listed is meant; the choice call below decides
                    continue
                val = closest
            selected[kname] = val
        open_keys = []
        for kname in sch.get("required") or []:
            if kname not in selected and isinstance((arg_props.get(kname) or {}).get("enum"), list):
                open_keys.append(kname)
        if not open_keys:
            return selected
        picked = getattr(self, "choice_memo", {}).get(role)
        if picked is None:
            if role == "checkout":
                aim = "place the order for what is in the cart"
            else:
                aim = "end the task without buying anything"
            tool_name = conf.get("name")
            display = {}
            for kname in open_keys:
                display[kname] = arg_props[kname]
            prompt = "\n".join([f"TOOL: {tool_name}", f"DESCRIPTION: {self.tool_blurb(tool_name or '')}",
                              "ARGUMENTS: " + json.dumps(display, ensure_ascii=False), "GOAL: " + aim])
            reply = self.desk.ask_json([QUICK_MODEL, SPARE_MODEL], P_CHOICE, prompt, 300)
            picked = reply if isinstance(reply, dict) else {}
            if picked:
                # only a useful answer is remembered; a failed call is retried next time
                self.choice_memo = dict(getattr(self, "choice_memo", {}), **{role: picked})
        for kname in open_keys:
            val = picked.get(kname)
            if val in arg_props[kname]["enum"]:
                selected[kname] = val
        return selected

    # -- schema lookups
    def schema_of(self, role):
        tool_name = (self.role_map.get(role) or {}).get("name") or ""
        return self.arg_specs.get(tool_name, {})

    def arg_kind(self, role, arg):
        if arg:
            return _qa_arg_spec(self, role, arg).get("type")
        return None

    def paging_kind(self, role):
        """"index" (page number), "offset" (row offset), "token", or None when the search does not page."""
        conf = self.role_map.get(role) or {}
        pg = conf.get("page")
        if not pg:
            return None
        if self.arg_kind(role, pg) != "integer":
            return "token"
        hint = str(conf.get("page_kind") or "").lower()
        if hint == "offset":
            return "offset"
        if hint != "number":
            low = pg.lower()
            if "offset" in low or "start" in low or "skip" in low:
                return "offset"
        return "index"

    def enum_pick(self, role, arg, value):
        """The schema's own spelling of a listed value; None when nothing listed is close."""
        allowed = _qa_arg_spec(self, role, arg).get("enum")
        if not (isinstance(allowed, list) and allowed and isinstance(value, str)):
            return value
        spelled = {}
        for elem in allowed:
            if isinstance(elem, str):
                spelled[flat_text(elem)] = elem
        if not spelled:
            return value
        flat = flat_text(value)
        if flat in spelled:
            return spelled[flat]
        peak = max(spelled, key=lambda elem: likeness(elem, value))
        if likeness(peak, value) >= 0.5:
            return spelled[peak]
        return None

    def query_for(self, item, role, arg, query):
        """The query as sent: value words before category words (values decide matches; some stores read only the
        first words); over the length limit category words go first, then the query is cut at a word."""
        cat_bag = word_bag(item.kind) | word_bag(item.label)

        def is_cat(word):
            bag = word_bag(word)
            return bool(bag) and bag <= cat_bag
        written_vals, tail_words = [], []
        for word in query.split():
            (tail_words if is_cat(word) else written_vals).append(word)
        joined = " ".join(written_vals + tail_words)
        ceiling = _qa_arg_spec(self, role, arg).get("maxLength")
        if not isinstance(ceiling, int) or len(joined) <= ceiling:
            return joined
        retained = written_vals[:]
        for word in tail_words:
            # category words come back in order while they still fit
            if len(" ".join(retained + [word])) <= ceiling:
                retained.append(word)
        return self.cut_to_fit(role, arg, " ".join(retained) if retained else joined)

    def cut_to_fit(self, role, arg, value):
        ceiling = _qa_arg_spec(self, role, arg).get("maxLength")
        if not isinstance(ceiling, int) or len(value) <= ceiling:
            return value
        head = value[:ceiling]
        if " " in head:
            return head.rsplit(" ", 1)[0]
        return head

    # -- search
    def seek(self, item, query=None, fields=None, max_pages=1, budget=None):
        """Search and page on; returns how many listings were new to the item's pool."""
        by_text = query is not None
        job = ("search_fields", "search_text")[by_text]
        conf = self.role_map.get(job)
        if not conf:
            return 0
        mode = self.paging_kind(job)
        if by_text:
            signature = job + ":" + query
        else:
            signature = job + ":" + json.dumps(fields or {}, sort_keys=True)
        state = item.pages.get(signature) or {}
        if state.get("done"):
            return 0
        cursor = state.get("token")
        begin = 1 + state.get("page", 0)
        if begin > 1 and mode is None:
            return 0
        reserve = budget if budget is not None else 0

        def page_value(n):
            # the paging argument for page n; None when the store cannot be paged that far this way
            if mode == "index":
                return n
            if mode == "offset":
                return (n - 1) * max(self.rows_per_page, 1)
            if mode == "token" and cursor:
                return cursor
            return None

        def read_page(obs):
            # (next-page cursor, reported match count) of one result
            nxt = most = None
            if not isinstance(obs, dict):
                return nxt, most
            found = list(obs.values())
            for elt in found:
                if isinstance(elt, str) and " " not in elt and 0 < len(elt) <= 20:
                    nxt = elt
            # the match count is the largest integer beside the rows (an offset is smaller, and may come first)
            ints = [elt for elt in found if isinstance(elt, int) and _qa_count(elt)]
            if ints:
                most = max(ints)
            return nxt, most
        added = 0
        finished = False
        pg = begin - 1
        last = begin + max_pages - 1
        while pg < last:
            pg = pg + 1
            if self.moves_left() <= reserve or CLOSE_MARGIN > self.seconds_left():
                pg = pg - 1
                break
            call = {}
            how = conf.get("mode")
            if how and isinstance(how, dict) and how.get("arg"):
                call[how["arg"]] = how.get("value")
            if not by_text:
                for kname, val in (fields or {}).items():
                    param = conf.get(kname)
                    if not param or val in (None, ""):
                        continue
                    picked = self.enum_pick(job, param, val)
                    if picked is None:
                        # the store lists no such category: this field search cannot run
                        item.pages[signature] = dict(page=0, token=None, done=True)
                        return added
                    if isinstance(picked, str):
                        picked = self.cut_to_fit(job, param, picked)
                    call[param] = picked
            elif conf.get("query"):
                call[conf["query"]] = self.query_for(item, job, conf["query"], query)
            else:
                return 0
            if pg > 1:
                step_val = page_value(pg)
                if step_val is None:
                    finished = True
                    pg = pg - 1
                    break
                call[conf["page"]] = step_val
                peak = _qa_arg_spec(self, job, conf["page"]).get("maximum")
                beyond = isinstance(step_val, int) and isinstance(peak, (int, float)) and step_val > peak
                if beyond:
                    # the store will not page this far: what lies beyond stays hidden
                    item.hidden_more = True
                    finished = True
                    pg = pg - 1
                    break
            outcome = self.use_tool(conf["name"], call)
            if not outcome["ok"]:
                finished = True
                self.log("search_error", error=outcome["error"])
                break
            records = record_rows(outcome["obs"]) or []
            cursor, matches = read_page(outcome["obs"])
            if records:
                self.rows_per_page = max(self.rows_per_page, len(records))
            if pg == 1 and by_text and (matches is not None or not records):
                self.tally_query(call[conf["query"]], 0 if (matches is None or not records) else matches)
            display = []
            for record in records:
                offer = self.offer_from(record, "row")
                if offer is not None:
                    display.append(offer)
                    is_new = offer.ref not in item.seen
                    added = added + (1 if is_new else 0)
                    self.absorb(item, offer, opened=False)
                    if by_text:
                        item.query_rows.setdefault(signature, set()).add(offer.ref)
            if len(display) > 1:
                # rows with price and stock leave little for the page to add when a table tool exists
                self.rows_rich = not any(o.cost_each is None or o.in_stock is None for o in display)
            item.seeks += 1
            if item.slot != 99 and item.kinds_ok is None:
                self.settle_labels(item)
            per_page = len(records)
            ended = not records or per_page < self.rows_per_page
            ended = ended or (matches is not None and pg * max(per_page, 1) >= matches)
            if not ended and mode == "token" and not cursor:
                if matches is not None and pg * per_page < matches:
                    # more matches than the store would list
                    item.hidden_more = True
                ended = True
            if ended:
                finished = True
                break
        item.pages[signature] = {"page": max(pg, 0), "token": cursor, "done": finished or mode is None}
        return added

    def tally_query(self, query, count):
        """Record a query's match count. Two queries where one's words contain the other's tell how the store
        matches: more words with fewer matches means every word must match."""
        mine = frozenset(plain_words(query))
        if not mine:
            return
        for rival, seen_count in self.query_tally.items():
            theirs = frozenset(plain_words(rival))
            if len(theirs) < len(mine):
                smaller, larger, small_n, big_n = theirs, mine, seen_count, count
            else:
                smaller, larger, small_n, big_n = mine, theirs, count, seen_count
            if len(smaller) == len(larger) or not (smaller & larger):
                continue
            prior_moves = self.strict_words
            if smaller < larger and big_n > small_n:
                self.strict_words = False
            elif big_n * 10 < small_n * 3 and (smaller < larger or len(larger) >= len(smaller) + 2):
                # far fewer matches for more words (a small dip can be the store's count cap)
                self.strict_words = True
            if prior_moves is not self.strict_words:
                self.log("search_words", all_words=self.strict_words, small=small_n, big=big_n)
        self.query_tally[query] = count

    @staticmethod
    def value_sets(item):
        """Plain required values as keyword groups (one per requirement, alternatives inside), shopper answers first."""
        ordered = []
        for need in item.live_needs():
            if need.rel not in ("is", "one of", "contains") or not need.want:
                continue
            # part codes are matched through fitment wording, never typed into a keyword query
            options = []
            for variant in need.want.split("|"):
                if word_bag(variant) and not PART_CODE.search(variant):
                    options.append(variant.strip())
            if not options or len(options) > 4:
                continue
            if any(len(o) > 24 for o in options):
                continue
            ordered.append((0 if need.source == "shopper" else 1, options))
        ordered.sort(key=lambda pair: pair[0])
        return [opts for _, opts in ordered]

    @staticmethod
    def row_matches(listing, groups):
        """Value groups the listing's own words carry."""
        pieces = [listing.heading, listing.blurb or "", listing.kind, listing.maker, listing.vendor]
        for kname, val in listing.props.items():
            pieces.append(f"{kname} {val}")
        bag = plain_words(" ".join(pieces))
        hits = 0
        for bunch in groups:
            for variant in bunch:
                if plain_words(variant) <= bag:
                    hits += 1
                    break
        return hits

    def value_terms(self, item):
        """Value queries (see value_term_order) not yet covered by a query already sent."""
        order = self.value_term_order(item)
        covered = []
        for kname in item.pages:
            if kname.startswith("search_text:"):
                covered.append(frozenset(plain_words(kname.split(":", 1)[1])))
        for sent_q in getattr(item, "values_sent", ()):
            covered.append(frozenset(plain_words(sent_q)))
        acc = []
        for qtext in order:
            wordset = frozenset(plain_words(qtext))
            if not any(wordset <= sset for sset in covered):
                acc.append(qtext)
        return acc

    def value_term_order(self, item):
        bunches = self.value_sets(item)
        if not bunches:
            return []
        noun_word = item.kind if len(item.head_noun) > 22 else item.head_noun
        # a store ranking by any word credits every listing with the category words: then values go alone
        suffix = [] if self.strict_words is False else [noun_word]
        leaders = [grp[0] for grp in bunches]
        acc = []
        # word values first, figures left out: a keyword store matches a figure only in its own spelling
        worded = []
        for grp in bunches:
            if not all(re.search(r"\d", lhs) for lhs in grp):
                worded.append(grp[0])
        if worded and len(worded) < len(bunches):
            acc.append(" ".join(worded[:3] + suffix))
        for idx, bunch in enumerate(bunches):
            others = [host for jdx, host in enumerate(leaders) if jdx != idx]
            for variant in bunch[:3]:
                qtext = " ".join(([variant] + others)[:3] + suffix)
                held_role = set(frozenset(plain_words(elt)) for elt in acc)
                if frozenset(plain_words(qtext)) not in held_role:
                    acc.append(qtext)
            if len(acc) >= 4:
                break
        return acc

    def seek_by_values(self, item, floor):
        """Send the next value query (and shorter ones when it matches nothing); True when a search ran."""
        queue_l = self.value_terms(item)
        if not queue_l or self.moves_left() <= floor + 1:
            return False
        qtext = queue_l[0]
        opening = not getattr(item, "values_sent", None)
        item.values_sent = list(getattr(item, "values_sent", [])) + [qtext]
        self.log("value_query", item=item.label, query=qtext)
        prior_moves = self.moves

        def qarg():
            return (self.role_map.get("search_text") or {}).get("query") or ""
        # the first value query pages as deep as the item's own queries; later ones less, unless every word counts
        if opening:
            depth = 6 if len(self.wants) == 1 else 3
        else:
            depth = 3 if self.strict_words else 2
        self.seek(item, query=qtext, max_pages=depth, budget=floor)
        self.mark_value_rows(item, qtext)
        sent_q = self.query_for(item, "search_text", qarg(), qtext)
        if self.query_tally.get(sent_q) != 0 or self.moves_left() <= floor + 1:
            return self.moves > prior_moves
        # no listing has every word: try the values without the noun, then the first value with the noun
        qparam = qarg()
        noun_bag = word_bag(item.head_noun)
        bare = []
        for wd in qtext.split():
            bag = word_bag(wd)
            if not (bag and bag <= noun_bag):
                bare.append(wd)
        fallbacks = (" ".join(bare), " ".join([self.value_sets(item)[0][0], item.head_noun]))
        for reduced in fallbacks:
            if not reduced or flat_text(reduced) == flat_text(qtext) or self.moves_left() <= floor + 1:
                continue
            # skip one the store would receive exactly as an earlier query
            if self.query_for(item, "search_text", qparam, reduced) in self.query_tally:
                continue
            item.values_sent.append(reduced)
            self.seek(item, query=reduced, max_pages=2, budget=floor)
            self.mark_value_rows(item, reduced)
            if self.query_tally.get(self.query_for(item, "search_text", qarg(), reduced)):
                break
        return self.moves > prior_moves

    def mark_value_rows(self, item, query):
        """Add a value query's rows to the item's value ids; when every value query was sent whole and listed all
        its matches, the value ids are complete."""
        item.value_rows |= item.query_rows.get("search_text:" + query) or set()
        qparam = (self.role_map.get("search_text") or {}).get("query") or ""

        def shown_all(qtext):
            signature = "search_text:" + qtext
            tally_n = self.query_tally.get(self.query_for(item, "search_text", qparam, qtext))
            if not tally_n:
                return False
            if not (item.pages.get(signature) or {}).get("done", False):
                return False
            return len(item.query_rows.get(signature) or ()) >= tally_n
        targets = self.value_term_order(item)
        if not targets:
            return
        for qtext in targets:
            if not shown_all(qtext):
                return
            whole = plain_words(self.query_for(item, "search_text", qparam, qtext))
            if not whole >= plain_words(qtext) - plain_words(item.head_noun):
                return
        item.values_all_seen = True

    def right_kind(self, item, listing):
        if not (listing.kind and item.kind):
            return True
        cat_label = flat_text(listing.kind)
        if item.kinds_ok:
            return cat_label in item.kinds_ok
        target = flat_text(item.kind)
        if cat_label in target or target in cat_label:
            return True
        return likeness(cat_label, target) >= 0.5

    def settle_labels(self, item):
        """Decide which category labels seen so far are the requested kind (one cheap model call)."""
        if item.kinds_ok is not None:
            return
        cat_labels = sorted(set(o.kind for o in item.seen.values() if o.kind))
        if not cat_labels:
            return
        by_flat = {}
        for label_name in cat_labels:
            by_flat[flat_text(label_name)] = label_name
        if len(cat_labels) == 1:
            only = flat_text(cat_labels[0])
            target = flat_text(item.kind)
            fit_flag = likeness(cat_labels[0], item.kind) >= 0.3 or target in only or only in target
            # a single foreign label leaves the word rule in force; more labels may still come
            item.kinds_ok = set(by_flat) if fit_flag else None
            return
        prompt = "\n".join([f"REQUESTED PRODUCT: {item.label} (category wording: {item.kind})",
                          f"REQUEST EXCERPT: {self.ask_text[:600]}",
                          "STORE CATEGORY LABELS SEEN: " + json.dumps(cat_labels, ensure_ascii=False)])
        reply = self.desk.ask_json([QUICK_MODEL, SPARE_MODEL], P_KINDS, prompt, 500)
        selected = set()
        given_names = reply.get("labels") if isinstance(reply, dict) else None
        if isinstance(given_names, list):
            for label_name in given_names:
                if isinstance(label_name, str) and flat_text(label_name) in by_flat:
                    selected.add(flat_text(label_name))
        if selected:
            item.kinds_ok = selected
            self.log("labels", item=item.label, labels=sorted(selected))

    def pool_rows(self, item):
        ceiling = self.limit
        rank = {"yes": 0, "no": 2}

        def keep(o):
            if self.stock_only and o.in_stock is False:
                return False
            if ceiling is not None and o.cost_each is not None and o.cost_each * item.count > ceiling + 1e-6:
                return False
            return self.right_kind(item, o)
        picked = [o for o in item.seen.values() if keep(o)]
        picked.sort(key=lambda o: (rank.get(o.triage_mark, 1), _qa_cost(o)))
        return picked

    def triage(self, item, listings):
        queue_l = []
        for o in listings:
            if o.read_full or o.triage_mark != "maybe" or getattr(o, "triaged", False):
                continue
            queue_l.append(o)
        if len(queue_l) < 8 or not item.need_lines():
            return
        lot = queue_l[:30]
        records = []
        for number, o in enumerate(lot, 1):
            records.append(dict(n=number, title=o.heading[:140], text=(o.blurb or "")[:ROW_TEXT], category=o.kind,
                             price=o.cost_each))
        prompt = ("REQUESTED PRODUCT: " + f"{item.kind}" + "\nREQUIREMENTS:\n- " + "\n- ".join(item.need_lines())
                + "\n\nROWS:\n" + json.dumps(records, ensure_ascii=False))
        # numbers in, numbers out: a short answer comes back fast
        reply = self.desk.ask_json([QUICK_MODEL, SPARE_MODEL], P_TRIAGE, prompt, 1200)
        for o in lot:
            o.triaged = True
        if not isinstance(reply, dict):
            return
        for ruling in ("no", "yes"):
            for number in reply.get(ruling) or []:
                if not isinstance(number, (int, str)) or not str(number).strip().isdigit():
                    continue
                if 1 <= int(number) <= len(lot):
                    lot[int(number) - 1].triage_mark = ruling

    # -- reading listings
    def table_mode(self):
        """Store with a table tool beside its page tool, no usable compare tool, and rows that show price and stock:
        a listing is judged from row plus table (one step); pages are read only where a page matters."""
        rm = self.role_map
        if not (rm.get("facts") and rm.get("open") and getattr(self, "rows_rich", False)):
            return False
        return not (rm.get("compare") and not getattr(self, "side_off", False))

    def table_only(self, item):
        if item is None or not self.table_mode():
            return False
        for lhs, host in self.pairings():
            # a fit is read off both products' pages
            if item is lhs or item is host:
                return False
        for need in item.needs:
            if need.pointer:
                return False
        return True

    def open_offers(self, item, ids, floor=0, page=False):
        """Read listings in full (compare tool when there is one); returns the read listings. page=True reads the
        listing's own page even where its table would do."""
        results = []
        queue = list(filter(None, ids))
        if len(queue) == 0:
            return results
        roles_now = self.role_map
        side = roles_now.get("compare") if not getattr(self, "side_off", False) else None
        table_tool, page_tool = roles_now.get("facts"), roles_now.get("open")
        if side and not getattr(self, "side_checked", False):
            # honour a limit stated on the list argument
            self.side_checked = True
            declared = _qa_arg_spec(self, "compare", side.get("ids") or "").get("maxItems")
            if isinstance(declared, int) and declared >= 2 and declared < self.side_max:
                self.side_max = declared
        tables_only = False if page else self.table_only(item)

        def keep(found):
            if found is not None:
                results.append(found)

        def short(step_cost):
            # true when this read would eat into the steps kept for later
            return bool(floor) and self.moves_left() - step_cost < floor

        def wrap(o):
            return self.absorb(item, o, True) if item else o

        def from_page(offer_id, record):
            one = self.offer_from(record, "detail")
            if one is None:
                one = Offer(offer_id)
                one.source_rec = record
            one.own_page = True
            prior_offer = item.seen.get(offer_id) if item is not None else None
            if not one.props and table_tool and self.moves_left() > 0:
                if prior_offer is None or not prior_offer.props:
                    one.props = self.pull_table(offer_id) or {}
            one.ref = offer_id
            one.read_full = True
            results.append(wrap(one))
            self.pages_read.add(offer_id)
        at = 0
        resent = set()
        while at < len(queue):
            if self.moves_left() <= 0 or self.seconds_left() <= CLOSE_MARGIN:
                break
            if tables_only or (self.side_thin and table_tool and item is not None):
                # each table on its own: compare rows carry none in this store
                if short(1):
                    break
                at += 1
                keep(self.open_by_table(item, queue[at - 1]))
                continue
            if side and len(queue) - at > 1:
                if short(1):
                    break
                id_slice = queue[at:at + self.side_max]
                outcome = self.use_tool(side["name"], {side["ids"]: id_slice})
                recs = record_rows(outcome["obs"])
                if outcome["ok"] and recs:
                    parsed = [o for o in map(lambda r: self.offer_from(r, "detail"), recs) if o is not None]
                    bare_rows = bool(table_tool) and len(parsed) > 0 and all(not o.props for o in parsed)
                    if bare_rows and not self.side_thin:
                        self.side_thin = True
                        self.keep_world()
                        self.log("thin_compare", tool=side["name"], facts_tool=table_tool.get("name"))
                    split = bare_rows and item is not None
                    for o in parsed:
                        o.own_page = True
                        self.pages_read.add(o.ref)
                        if split:
                            # keep the row's notes now; its table is read below
                            self.absorb(item, o, False)
                        else:
                            o.read_full = True
                            results.append(wrap(o))
                    if split:
                        for o in parsed:
                            if short(1) or CLOSE_MARGIN > self.seconds_left():
                                break
                            keep(self.open_by_table(item, o.ref))
                    at = at + len(id_slice)
                    returned = {o.ref for o in parsed}
                    missed = [elt for elt in id_slice if not (elt in returned or elt in resent)]
                    if parsed and missed and len(returned) < len(id_slice):
                        # fewer rows than ids: the tool caps the batch, the others go round again
                        if 2 <= len(returned) < self.side_max:
                            self.side_max = len(returned)
                            self.keep_world()
                        resent.update(missed)
                        queue[at:at] = missed
                    continue
                if outcome["error"] or outcome["ok"]:
                    # whatever the error says, the tool may take fewer ids: halve, and below two stop using it
                    if self.side_max > 2 and len(id_slice) > 2:
                        self.side_max = max(2, min(self.side_max, len(id_slice)) // 2)
                        self.keep_world()
                    else:
                        self.side_off = True
                        side = None
                    continue
            if not page_tool or short(2 if table_tool else 1):
                break
            offer_id = queue[at]
            at += 1
            outcome = self.use_tool(page_tool["name"], {page_tool["id"]: offer_id})
            if outcome["ok"] and isinstance(outcome["obs"], dict):
                from_page(offer_id, outcome["obs"])
        return results

    def pull_table(self, lid):
        """A listing's property table via the facts tool (one step): {} when it has none, None on a failed call."""
        tool_name = self.role_map.get("facts")
        if not tool_name:
            return None
        outcome = self.use_tool(tool_name["name"], {tool_name["id"]: lid})
        record = outcome["obs"]
        if not outcome["ok"] or not isinstance(record, dict):
            return None
        table_key = self.key_roles("facts", record).get("facts")
        if table_key and isinstance(record.get(table_key), dict):
            return _qa_clean_table(record[table_key])
        flat = all(isinstance(value_v, (str, int, float)) or value_v is None for value_v in record.values())
        return _qa_clean_table(record) if flat else {}

    def open_by_table(self, item, lid):
        """Read a listing through its table only; its seen row supplies title, price and text."""
        prior_offer = item.seen.get(lid)
        field_rows = self.pull_table(lid)
        if field_rows is None:
            return None
        if not field_rows and prior_offer is None:
            return None
        newer = Offer(lid)
        newer.props = field_rows
        return self.absorb(item, newer, True)

    # -- judging
    def drop_kind_needs(self, item):
        """Drop a request value that only repeats the store's category label for the item, on a property no read
        listing has: it names the category, not a requirement."""
        if not item.kinds_ok:
            return
        label_bag = set()
        for label_name in item.kinds_ok:
            label_bag |= word_bag(label_name)
        opened_ok = _qa_read_listings(self, item)
        if len(opened_ok) < 3:
            return
        retained = []
        for need in item.needs:
            bag = word_bag(need.want)
            restates = need.source == "request" and need.rel == "is" and bag and bag <= label_bag
            if restates and not _qa_any_names(opened_ok, need.field):
                self.log("label_restatement", item=item.label, req=need.line())
                continue
            retained.append(need)
        item.needs = retained

    def match_fields(self, item):
        """When no read listing names a required property but they carry the required value under one other key,
        the requirement takes that key's name (the store files it differently). Word values only."""
        opened_ok = _qa_read_listings(self, item)
        if len(opened_ok) < 3:
            return
        for need in item.live_needs():
            if need.rel not in ("is", "one of", "contains") or not need.want:
                continue
            if re.search(r"\d", need.want) or getattr(need, "fields_matched", False):
                continue
            if _qa_any_names(opened_ok, need.field):
                continue
            want_bags = _qa_want_sets(need.want)
            if not want_bags:
                continue
            tally = {}
            for o in opened_ok:
                knames = set()
                for kname, val in o.props.items():
                    pieces = _qa_elements([val])
                    if any(wd <= issue for wd in want_bags for issue in pieces):
                        knames.add(_qa_spaced(kname))
                for kname in knames:
                    tally[kname] = tally.get(kname, 0) + 1
            ordered = sorted(tally.items(), key=lambda kv: -kv[1])
            if not ordered:
                continue
            if len(ordered) > 1 and ordered[0][1] <= ordered[1][1]:
                continue
            peak = ordered[0][0]
            self.log("prop_aligned", item=item.label, was=need.field, now=peak, value=need.want)
            need.fields_matched = True
            need.field = peak

    def assess(self, item, listings):
        self.drop_kind_needs(item)
        self.match_fields(item)
        req_lines = item.need_lines()
        rulings, judged = item.rulings, item.assessed_on
        if len(req_lines) == 0:
            # only the product kind to judge: each read listing stands
            for o in listings:
                if not o.read_full or o.ref in rulings:
                    continue
                rulings[o.ref] = {"failed": set(), "unsure": set()}
                judged[o.ref] = set()
            return
        req_set = set(req_lines)
        queue_l = []
        for o in listings:
            if o.read_full and not judged.get(o.ref, set()) >= req_set:
                queue_l.append(o)
        for o in list(queue_l):
            # failing a requirement still in force decides it: no call needed
            prior = rulings.get(o.ref)
            if not (o.ref in judged and prior):
                continue
            if set(prior.get("failed") or ()) & req_set:
                judged[o.ref] = set(req_lines)
                queue_l.remove(o)
        if not queue_l or CLOSE_MARGIN > self.seconds_left():
            return
        field_rows = self.store_fields(item)
        notes_part = ""
        if field_rows:
            notes_part = "\n\nSTORE FIELDS (from this store's records read so far):\n" + self.field_notes(field_rows)
        numbered = []
        for pos, line_text in enumerate(req_lines):
            numbered.append("R%d: %s" % (pos + 1, line_text))
        head = "REQUESTED PRODUCT: " + f"{item.kind}" + "\nREQUIREMENTS (judge each):\n" + "\n".join(numbered) \
            + notes_part + "\n\nLISTINGS:\n"

        def card_of(o):
            shown = o.card()
            links = self.linked_pages(item, o)
            if links:
                shown["related_pages"] = links
            return shown
        bunches = [queue_l[cut:cut + 4] for cut in range(0, len(queue_l), 4)]
        prompts = [head + json.dumps([card_of(o) for o in b], ensure_ascii=False) for b in bunches]
        # independent batches run in parallel: model wait is the cost here, not steps
        replies = run_together([(lambda u=u: self.desk.ask_json([QUICK_MODEL, SPARE_MODEL], P_ASSESS, u, 2000))
                                for u in prompts])

        def canon(given):
            # requirement references (R2, a paraphrase) back to the requirement lines
            back = set()
            for label in given:
                mt = re.fullmatch(r"\s*R?(\d+)\s*", label)
                idx = int(mt.group(1)) if mt else 0
                if 0 < idx <= len(req_lines):
                    back.add(req_lines[idx - 1])
                    continue
                closest = max(req_lines, key=lambda cond: likeness(cond, label)) if req_lines else None
                back.add(closest if (closest and likeness(closest, label) > 0.2) else label)
            return back
        for bunch, reply in zip(bunches, replies):
            verdicts = reply.get("verdicts") if isinstance(reply, dict) else None
            by_id = {}
            for one in (verdicts if isinstance(verdicts, list) else []):
                if isinstance(one, dict) and one.get("id") in item.seen:
                    by_id[one["id"]] = one
            for o in bunch:
                if o.ref not in by_id:
                    continue
                said = by_id[o.ref]
                loose_set = _qa_names_of(said, "loose")
                ruling = dict(failed=canon(_qa_names_of(said, "failed")) | self.field_misses(item, o, field_rows),
                              unsure=canon(_qa_names_of(said, "unsure")),
                              weak=canon(_qa_names_of(said, "weak") | loose_set),
                              loose=canon(loose_set))
                rulings[o.ref] = ruling
                judged[o.ref] = set(req_lines)
                for need in item.live_needs():
                    # a value met as written must have its words somewhere in the records
                    line_text = need.line()
                    if any(line_text in ruling[part] for part in ("failed", "unsure", "loose")):
                        continue
                    if self.never_stated(item, o, need):
                        ruling["unsure"].add(line_text)

    @staticmethod
    def never_stated(item, listing, req):
        """None of the listing's records has a word of the value beyond the product's own name words."""
        if req.rel not in ("is", "one of") or req.source == "fit" or not req.want:
            return False
        # a key naming the property is for the judge to read (other spellings, ranges, ages)
        for kname in listing.props:
            if _qa_names_field(kname, req.field):
                return False
        name_words = plain_words(item.label) | plain_words(item.kind)
        # short alphabetic words (articles, joiners) prove nothing; figures do
        variants = []
        for option in req.want.split("|"):
            wordset = set(elt for elt in plain_words(option) - name_words if not (elt.isalpha() and len(elt) <= 3))
            if wordset:
                variants.append(wordset)
        if not variants:
            return False
        table_text = " ".join(f"{kk} {value_v}" for kk, value_v in listing.props.items())
        blob = " ".join([listing.heading or "", listing.blurb or "", listing.remarks or "", listing.kind or "", table_text])
        stems = _qa_stems(plain_words(blob))
        for wordset in variants:
            if _qa_stems(wordset) & stems:
                return False
        return True

    def store_fields(self, item):
        """How this store writes each plain required value, from the listings read so far: keys naming the
        property, how many listings carry one, the values written there, and which listings write the value."""
        opened_ok = _qa_read_listings(self, item)
        if len(opened_ok) < 3:
            return []
        acc = []
        for need in item.live_needs():
            if need.rel not in ("is", "one of") or not need.want or re.search(r"\d", need.want):
                continue
            want_bags = _qa_want_sets(need.want)
            if not want_bags:
                continue
            knames, written_vals, stating = set(), {}, set()
            holders = 0
            for o in opened_ok:
                match = [kk for kk in o.props if close_names(kk, need.field)]
                if not match:
                    continue
                holders += 1
                knames.update(match)
                for kname in match:
                    for piece in re.split(_QA_SPLIT, str(o.props[kname])):
                        piece = " ".join(piece.lower().split())
                        if not piece:
                            continue
                        written_vals[piece] = written_vals.get(piece, 0) + 1
                        bag = frozenset(word_bag(piece))
                        if any(wd <= bag for wd in want_bags):
                            stating.add(o.ref)
            if holders:
                acc.append(dict(req=need.line(), prop=need.field, keys=sorted(knames), carriers=holders,
                                read=len(opened_ok), values=written_vals, wants=want_bags, writers=stating))
        return acc

    def field_misses(self, item, listing, table=None):
        """Requirements the listing fails by the store's own wording: its field lacks the value while other read
        listings write it exactly there."""
        acc = set()
        records = self.store_fields(item) if table is None else table
        for frow in records:
            own_keys = [kk for kk in listing.props if close_names(kk, frow["prop"])]
            if not own_keys or not (frow["writers"] - {listing.ref}):
                continue
            pieces = _qa_elements([listing.props[kk] for kk in own_keys])
            if not pieces:
                continue
            # compare by stems so another spelling of the value is not a plain failure
            if not any(_qa_stems(wd) <= _qa_stems(issue) for wd in frow["wants"] for issue in pieces):
                acc.add(frow["req"])
        return acc

    def field_notes(self, table):
        acc = []
        for frow in table:
            peak = sorted(frow["values"].items(), key=lambda kv: -kv[1])[:8]
            written = " | ".join(f"{value_v} ({number})" for value_v, number in peak)
            knames = ", ".join(repr(kk) for kk in frow["keys"][:3])
            acc.append(f"- {frow['req']}: field {knames} on {frow['carriers']} of {frow['read']} read listings; values "
                       f"written: {written}; {len(frow['writers'])} listing(s) write the required value itself")
        return "\n".join(acc)

    def assess_everything(self):
        """Judge read candidates of all items at once."""
        jobs = []
        for each in self.wants:
            jobs.append(lambda each=each: self.assess(each, [o for o in self.pool_rows(each) if o.read_full]))
        run_together(jobs)

    def linked_pages(self, item, listing, limit=2):
        """Known listings tied to this one by a part code: named in its text by their title code, or naming its title
        code in theirs (a compatibility list may sit on the other page)."""
        pattern = re.compile(r"\b[A-Z]{1,6}-?\d{3,}[A-Z0-9]*\b")

        def notes(o):
            return " ".join([o.blurb or "", o.remarks or ""])
        title_codes = set(pattern.findall(listing.heading or ""))
        cited = set(pattern.findall(notes(listing))) - title_codes
        found = []
        taken = {listing.ref}
        stores = [item.seen]
        for each in self.wants:
            if each is not item:
                stores.append(each.seen)
        for candidates in stores:
            for rival in candidates.values():
                if rival.ref in taken or len(found) >= limit:
                    continue
                their_codes = set(pattern.findall(rival.heading or ""))
                if (their_codes & cited) or (title_codes & set(pattern.findall(notes(rival)))):
                    taken.add(rival.ref)
                    found.append({"id": rival.ref, "title": rival.heading[:120], "text": notes(rival)[:400]})
        return found

    def confirm_offer(self, item, listing):
        signature = (item.slot, listing.ref, tuple(item.need_lines()), json.dumps(listing.props, sort_keys=True, default=str))
        stash = self.__dict__.setdefault("verified", {})
        if signature in stash:
            return stash[signature]
        ruling = self.check_offer(item, listing)
        stash[signature] = ruling
        return ruling

    def check_offer(self, item, listing, strict=False):
        req_lines = item.need_lines()
        ceiling = "none stated" if self.limit is None else self.limit
        listed = "\n- ".join(req_lines) if req_lines else "(none beyond the product kind, budget and availability)"
        prompt = "".join([f"REQUESTED PRODUCT: {item.kind} (quantity {item.count})\n",
                        f"BUDGET, SUM OF EVERY LINE: {ceiling}\n", "REQUIREMENTS:\n- ", listed,
                        "\n\nLISTING:\n", json.dumps(listing.card(3500), ensure_ascii=False)])
        reply = self.desk.ask_json([CHECKER_MODEL, QUICK_MODEL, SPARE_MODEL], P_CHECK, prompt, 800)
        if not isinstance(reply, dict):
            return not strict
        fit_flag = reply.get("fits")
        if strict and fit_flag is not True:
            return False
        issues = [str(issue) for issue in (reply.get("problems") or [])]
        self.log("verify", id=listing.ref, fits=fit_flag, problems=issues[:4])
        if fit_flag is not False or not issues:
            return True
        mark = item.rulings.setdefault(listing.ref, {"failed": set(), "unsure": set()})
        for issue in issues:
            nearest = max(req_lines, key=lambda cond: likeness(cond, issue)) if req_lines else None
            mark["failed"].add(nearest if nearest and likeness(nearest, issue) > 0.2 else issue)
        return False

    # -- the shopper
    def may_message(self):
        cap = self.msg_cap
        if "message" in self.role_map and (cap is None or self.msgs_used < cap):
            return self.moves_left() > 0
        return False

    def kept_messages(self):
        """How many messages stay reserved for the closing steps (a confirmation, a notice)."""
        if not self.demands or "message" not in self.role_map:
            return 0
        held_back = 0
        for dem in self.demands:
            # a decline keeps nothing: the ending tool carries its own note
            if dem["kind"] != "message" or dem["moment"] not in ("before_order", "before_finish"):
                continue
            held_back += max(1, int(dem.get("messages") or 1))
        cap = self.msg_cap
        if cap is None:
            return held_back
        return min(held_back, max(0, cap - self.msgs_used))

    def owed_messages(self):
        """Reserved messages still due for the current basket (zero once that basket was shown)."""
        if self.basket and self.approved_key == self.basket_key():
            return 0
        return self.kept_messages()

    def may_ask(self):
        """True while a question fits besides the reserved closing messages."""
        if self.may_message():
            cap = self.msg_cap
            return cap is None or self.msgs_used < cap - self.kept_messages()
        return False

    def process_check(self, moment, intent):
        """Ask the planner which demanded steps are due now and run them as validated tool calls. True when one ran,
        False when none was due, None when no answer came back."""
        attempts = self.gate_count.get(moment, 0)
        if attempts >= 3 or self.moves_left() < 2 or _kb_short(self, 0):
            return False
        self.gate_count[moment] = attempts + 1
        closing = set()
        for role in ("checkout", "decline", "cart_add", "cart_set"):
            tname = (self.role_map.get(role) or {}).get("name")
            if tname is not None:
                closing.add(tname)
        offered = []
        for tname in self.arg_specs:
            if tname in closing:
                continue
            offered.append({"name": tname, "description": self.tool_blurb(tname), "parameters": self.arg_specs[tname]})
        recs = []
        for refd, cnt in self.basket_key():
            owner_w = self.basket_owner.get(refd)
            offer = owner_w.seen.get(refd) if owner_w else None
            recd = {"id": refd, "quantity": cnt, "title": "", "price": None, "for": None}
            if offer:
                recd["title"] = offer.heading[:120]
                recd["price"] = offer.cost_each
            if owner_w:
                recd["for"] = owner_w.label
            recs.append(recd)
        tot = 0
        for recd in recs:
            tot += (recd["price"] or 0) * recd["quantity"]
        spare_rows = "no limit" if self.msg_cap is None else self.msg_cap - self.msgs_used
        pieces = [
            "REQUEST:\n", self.ask_text,
            "\n\nSTEPS THE REQUEST DEMANDS:\n", _kb_js(self.demands),
            f"\n\nMOMENT: {moment} (the assistant is about to {intent})",
            "\n\nBASKET NOW: ", _kb_js({"lines": recs, "total": tot}),
            f"\nMESSAGES LEFT: {spare_rows}; STEPS LEFT: {self.moves_left()}",
            "\n\nTOOLS:\n", _kb_js(offered),
            "\n\nDONE SO FAR (tool calls and results, oldest first):\n", _kb_js(self.journal[-30:]),
        ]
        rsp = self.desk.ask_json([PLANNER_MODEL, QUICK_MODEL, SPARE_MODEL], P_PROCESS, "".join(pieces), 900)
        if not isinstance(rsp, dict):
            self.log("gate_failed", moment=moment)
            return None
        queue_ids = []
        for action in (rsp.get("actions") or [])[:2]:
            if isinstance(action, dict):
                queue_ids.append(action)
        speak = (self.role_map.get("message") or {}).get("name")
        acted = False
        for index_, action in enumerate(queue_ids):
            tname = action.get("tool")
            if tname not in self.arg_specs or tname in closing:
                continue
            if tname != speak:
                # keep steps for messages still owed (later in this answer or reserved) and the checkout
                later = 0
                for alt_o in queue_ids[index_ + 1:]:
                    if alt_o.get("tool") == speak:
                        later += 1
                candidates = max(later, self.owed_messages())
                if self.moves_left() < candidates + 2:
                    self.log("gate_step_skipped", moment=moment, tool=tname, steps=self.moves_left(), due=candidates)
                    continue
            cfg = self.arg_specs[tname]
            flds = cfg.get("properties") or {}
            preset = action.get("args")
            call_args = {}
            if isinstance(preset, dict):
                for kname, val in preset.items():
                    if kname in flds:
                        call_args[kname] = val
            for kname in list(call_args):
                val = call_args[kname]
                fspec = flds.get(kname) or {}
                allowed = fspec.get("enum")
                if isinstance(allowed, list) and allowed and val not in allowed:
                    del call_args[kname]
                elif isinstance(val, str):
                    call_args[kname] = self.cut_arg(fspec, val)
            if not all(rq in call_args for rq in cfg.get("required") or []):
                continue
            talking = (self.role_map.get("message") or {}).get("name") == tname
            if talking and not self.may_message():
                continue
            if self.moves_left() < 2:
                break
            resp = self.use_tool(tname, call_args)
            if talking:
                self.msgs_used += 1
                self.approved_key = self.basket_key()
            self.log("gate_step", moment=moment, tool=tname, why=str(action.get("why") or "")[:160], ok=resp["ok"])
            acted = True
        if not acted:
            self.log("gate_clear", moment=moment)
        return acted

    def cut_arg(self, prop, value):
        cap = prop.get("maxLength")
        if isinstance(cap, int) and len(value) > cap:
            return value[:cap]
        return value

    def basket_key(self):
        fit_pairs = []
        for refd, cnt in self.basket.items():
            if cnt:
                fit_pairs.append((refd, int(cnt or 0)))
        fit_pairs.sort()
        return tuple(fit_pairs)

    def show_basket(self):
        """Send the shopper the basket and ask them to confirm; True once the message is sent."""
        can_close = self.may_message() and self.moves_left() >= 2
        if not can_close:
            return False
        cfg = self.role_map["message"]
        bits = []
        tot = 0.0
        for refd, cnt in self.basket_key():
            owner_w = self.basket_owner.get(refd)
            offer = owner_w.seen.get(refd) if owner_w else None
            nm = _kb_squash((offer.heading if offer else refd) or refd)[:70]
            each = None
            if offer and offer.cost_each is not None:
                each = offer.cost_each
            ln_ = f"{cnt} x {nm}"
            if each is not None:
                tot += each * cnt
                ln_ += f" ({each:g} each)"
            bits.append(ln_)
        tail = f". Total {tot:g}." if tot else "."
        body = "Your final basket: " + "; ".join(bits) + tail + " Shall I check out this basket? A yes from you lets me place it."
        body = self.cut_to_fit("message", cfg["text"], body)
        resp = self.use_tool(cfg["name"], {cfg["text"]: body})
        self.msgs_used += 1
        self.approved_key = self.basket_key()
        self.log("confirm_asked", basket=self.approved_key, ok=resp["ok"], obs=brief(resp["obs"], 160))
        return True

    def put_question(self, item, prop, narrow=False, also=None):
        cfg = self.role_map["message"]
        asked_txt = self.cut_to_fit("message", cfg["text"], _kb_question_text(item, prop, narrow, also))
        sent_r = self.use_tool(cfg["name"], {cfg["text"]: asked_txt})
        self.msgs_used += 1
        if not narrow:
            item.questioned.extend(flat_text(p_) for p_ in ([prop, also] if also else [prop]))
        heard = _kb_reply_text(sent_r["obs"]) if sent_r["ok"] else None
        if not heard:
            return None
        told = [f"{rq.field}: {rq.rel} {rq.want}" for rq in item.needs if rq.source == "shopper" and rq.want]
        about_p = prop + " or " + also if also else prop
        prompt_txt = f"ITEM: {item.label}\nPROPERTY ASKED ABOUT: {about_p}\nQUESTION: {asked_txt}\nKNOWN NEEDS: {json.dumps(told)}\nREPLY: {heard}"
        parsed = self.desk.ask_json([QUICK_MODEL, SPARE_MODEL, PLANNER_MODEL], P_REPLY, prompt_txt, 600)
        sort = str(parsed.get("kind") or "").lower() if isinstance(parsed, dict) else None
        if sort == "need":
            named_p = self.field_in_reply(item, heard, [prop, also or "", str(parsed.get("property") or "")],
                                          str(parsed.get("value") or ""))
            if named_p:
                # the reply names a different listing property: the need sits on that one
                parsed["property"] = named_p
        self.log("reply", item=item.label, prop=prop, reply=heard[:200], read=brief(parsed, 300))
        if sort is None:
            return None
        if sort in ("need", "repeat"):
            # remember the shopper's wording so a later question on the same subject counts as a repeat
            item.said_lines = list(getattr(item, "said_lines", [])) + [" " + flat_text(heard) + " "]
        return parsed

    @staticmethod
    def said_before(item, prop):
        """True when a multi-word property name already appeared in a reply that stated a need."""
        wbag = flat_text(prop)
        if len(wbag.split()) < 2:
            return False
        probe = f" {wbag} "
        for ln_ in getattr(item, "said_lines", []):
            if probe in ln_:
                return True
        return False

    def field_in_reply(self, item, reply, asked, value):
        """When a reply mentions none of the asked names, the single listing property it does mention (else None)."""
        padded = " " + flat_text(reply) + " "
        for al in asked:
            if al and flat_text(al) and " " + flat_text(al) + " " in padded:
                return None
        aliases = set()
        for offer in item.seen.values():
            for kname in offer.props:
                nm = flat_text(_kb_spaced(kname))
                if len(nm) < 4 or word_bag(nm) & word_bag(value):
                    continue
                if any(close_names(nm, al) for al in asked if al):
                    continue
                aliases.add(nm)
        hit_count = [nm for nm in aliases if " " + nm + " " in padded]
        kept = []
        for hh in hit_count:
            # drop a name that is only part of a longer hit
            inner = False
            for oo in hit_count:
                if hh != oo and " " + hh + " " in " " + oo + " ":
                    inner = True
                    break
            if not inner:
                kept.append(hh)
        if len(kept) == 1:
            return kept[0]
        return None

    def take_reply(self, item, prop, answer):
        if not answer:
            return
        for step_d in answer.get("steps") or []:
            if not isinstance(step_d, dict):
                continue
            self.add_demand(step_d)
        sort = str(answer.get("kind") or "").lower()
        cap = as_amount(answer.get("budget"))
        if cap is not None and cap > 0:
            self.limit = cap
            self.log("budget", cap=cap)
        if sort == "free":
            item.released.append(flat_text(str(answer.get("property") or prop)))
            return
        if sort != "need" or not (answer.get("value") or answer.get("reference")):
            return
        newer = Need(str(answer.get("property") or prop), answer.get("op") or "is", answer.get("value") or "",
                     answer.get("reference"), origin="shopper")
        if newer.pointer:
            # the page gets read for the need's own property
            newer.pointer["property"] = newer.field
            signature = pointer_key(newer.pointer)
            if signature:
                for prev in item.needs:
                    match_o = signature in (getattr(prev, "chain_key", None), getattr(prev, "chain_key0", None))
                    if match_o or (prev.pointer and pointer_key(prev.pointer) == signature):
                        # the same page reference again: the value already taken from it stands
                        self.log("reference_repeat", item=item.label, prop=newer.field)
                        return
        if newer.want:
            for prev in item.needs:
                if prev.source == "shopper" and flat_text(prev.want) == flat_text(newer.want) and prev.rel == newer.rel:
                    return
        # only a shopper-given requirement on the same property gets replaced
        keep = []
        for prev in item.needs:
            if prev.slug() != newer.slug() or prev.source != "shopper":
                keep.append(prev)
        keep.append(newer)
        item.needs = keep
        if newer.rel in ("is", "one of") and newer.want and not newer.pointer and len(newer.want) <= 40:
            lead = newer.want.split("|")[0].strip()
            item.queue_query(f"{lead} {item.head_noun}")
        if newer.pointer and newer.rel in POINTER_RELS:
            self.follow_pointer(item, newer)

    def take_reply_once(self, item, prop, answer):
        """take_reply while questions are still pending: a reference chain must leave their steps untouched."""
        if self.msg_cap is None:
            steps_now = 2
        else:
            steps_now = self.msg_cap - self.msgs_used
        self.chain_floor = max(0, steps_now) + self.ending_steps()
        try:
            self.take_reply(item, prop, answer)
        finally:
            self.chain_floor = 0

    def field_menu(self, item, carried=False):
        """Properties found on the item's opened listings, ranked by how much their values vary (or, with carried, by
        how many listings have them); keys holding equal values are folded together."""
        bucket = list(filter(lambda cand_r: cand_r.read_full and self.right_kind(item, cand_r), item.seen.values()))
        live = [cand_r for cand_r in bucket if not _kb_verdict(item, cand_r).get("failed")]
        if len(live) >= 4:
            bucket = live
        tables = _kb_prop_tables(bucket)
        tally = {}
        for tbl_row in tables:
            tally.update({nm: tally.get(nm, 0) + 1 for nm in tbl_row})
        # on a tie the plainer (shorter) name speaks for its aliases
        key_names = sorted(tally, key=lambda nm: (-tally[nm], len(nm), nm))
        folded = _kb_fold_aliases(key_names, tables)
        decided = {rq.slug() for rq in item.needs} | set(item.questioned)

        def is_fixed(nm):
            return flat_text(nm) in decided or any(likeness(nm, fx) >= 0.67 for fx in decided)
        options = []
        for kname in (nm for nm in key_names if nm not in folded):
            group = [kname] + [bb for bb, al in folded.items() if al == kname]
            if any(is_fixed(nm) for nm in group):
                continue
            if word_bag(kname) & NEVER_ASK:
                continue            # variant pickers confuse the shopper
            held_vals = [vv for vv in _kb_first_values(tables, group) if vv is not None]
            options.append(_kb_menu_entry(kname, group, held_vals, len(bucket), carried))
        options.sort(key=lambda ent: (-ent["_w"], ent["property"]))
        for ent in options:
            ent.pop("_w")
        return options[:40]

    def settled_fields(self, item):
        """Properties the request itself already fixes (rule sources, rule targets, pairings): not to be asked."""
        hit_any = []
        for link in getattr(self, "rules", []):
            if link["src"] is item:
                hit_any.append(link["prop"])
            for _, dest, branch in link["branches"]:
                if dest is item:
                    hit_any.append(str(branch["property"]))
        for host_s, _ in self.pairings():
            if host_s is item:
                hit_any.append("compatibility")
                break
        return list(dict.fromkeys(hit_any))

    def _kq_carried_first(self, picked):
        """The planner's questions with those on properties that fewer than half of the product's read listings carry
        moved behind the rest (order kept otherwise), unless a story hint names them: across races 5-10 a question on a
        property most of the agent's own reads carry drew a requirement .16-.23 of the time, a rarer one .10-.16."""
        menus = {}
        for wt in self.wants:
            menus[wt.slot] = [(e["property"], e["listings"] / max(e["of"], 1)) for e in self.field_menu(wt)]

        def rare(pair):
            wt, prop_name = pair
            if any(close_names(prop_name, c) or likeness(prop_name, c) >= 0.6 for c in wt.clues):
                return False
            share = next((sh for name, sh in menus.get(wt.slot, []) if close_names(prop_name, name)), None)
            return share is not None and share < 0.5
        return [pr for pr in picked if not rare(pr)] + [pr for pr in picked if rare(pr)]

    def question_plan(self, count, model=PLANNER_MODEL):
        """Question order: the planner proposes it once, code drops repeats and settled properties and fills the gaps
        from the property menus."""
        if count <= 0:
            return []
        agenda = []

        def is_taken(itm, prop_name):
            for al in itm.questioned:
                if close_names(prop_name, al):
                    return True
            for dd in self.settled_fields(itm):
                if _kb_near(prop_name, dd):
                    return True
            for rq in itm.needs:
                if close_names(prop_name, rq.field):
                    return True
            if self.said_before(itm, prop_name):
                return True
            for wt, pp in agenda:
                if wt is itm and close_names(prop_name, pp):
                    return True
            return False

        if not getattr(self, "q_order", None):
            item_blocks = []
            for itm in self.wants:
                block = {"item": itm.label, "category": itm.kind}
                block["known_requirements"] = itm.need_lines()
                block["decided_by_the_request"] = self.settled_fields(itm)
                block["hints"] = itm.clues
                block["already_asked"] = itm.questioned
                block["property_menu"] = self.field_menu(itm)
                item_blocks.append(block)
            asked_for = count if self.msg_cap is None else self.msg_cap
            body_text = "".join(["REQUEST:\n", self.ask_text, "\n\nSUBJECTS THE REQUEST TELLS THE ASSISTANT TO RAISE: ",
                            json.dumps(self.raise_topics), "\n\nITEMS:\n", _kb_js(item_blocks),
                            f"\n\nQUESTION_COUNT = {max(asked_for, count)}"])
            backup = SPARE_MODEL if model == QUICK_MODEL else QUICK_MODEL
            rsp = self.desk.ask_json([model, backup] + ([SPARE_MODEL] if backup != SPARE_MODEL else [QUICK_MODEL]), P_QUESTIONS, body_text, 700)
            picked = []
            if isinstance(rsp, dict) and isinstance(rsp.get("questions"), list):
                for qtext in rsp["questions"]:
                    if not isinstance(qtext, dict) or not qtext.get("property"):
                        continue

                    def fit(wt, qtext=qtext):
                        return likeness(wt.label, qtext.get("item") or wt.label)
                    tgt = max(self.wants, key=fit)
                    prop_name = _kb_squash(str(qtext["property"]).replace("_", " ")).lower()
                    if not prop_name:
                        continue
                    if any(wt is tgt and close_names(prop_name, pp) for wt, pp in picked):
                        continue
                    picked.append((tgt, prop_name))
            # (K4 reordering by coverage was measured harmful against b08: off)
            self.q_order = picked
            self.log("plan_questions", questions=[(wt.label, pp) for wt, pp in picked])
        if len(self.wants) > 1:
            return self.turn_plan(count, is_taken)
        for itm, prop_name in self.q_order:
            if len(agenda) < count:
                if not is_taken(itm, prop_name):
                    agenda.append((itm, prop_name))
            else:
                break
        # one item from here on: its two most carried properties are likely hidden needs, put them early
        if count >= 3:
            for itm in self.wants:
                top_props = []
                for ent in self.field_menu(itm, carried=True):
                    if ent["listings"] >= 3:
                        top_props.append(ent["property"])
                for ki, prop_name in enumerate(top_props[:2]):
                    if not is_taken(itm, prop_name):
                        agenda.insert(min(len(agenda), 2 * ki + 1), (itm, prop_name))
            del agenda[count:]
        if len(agenda) < count:
            menu_of = {}
            for wt in self.wants:
                menu_of[wt.slot] = [ent for ent in self.field_menu(wt) if ent["listings"] >= 2 or ent["of"] <= 2]

            def load(wt):
                return (sum(1 for hh, _ in agenda if hh is wt) + len(wt.questioned), wt.slot)
            sequence = sorted(self.wants, key=load)
            while len(agenda) < count:
                grew = False
                for itm in sequence:
                    queue = menu_of[itm.slot]
                    while queue:
                        prop_name = queue.pop(0)["property"]
                        if not is_taken(itm, prop_name):
                            agenda.append((itm, prop_name))
                            grew = True
                            break
                    if len(agenda) >= count:
                        break
                if not grew:
                    break
        return agenda[:count]

    def turn_plan(self, count, taken):
        """Several items: one queue per item (planner picks, its most carried property in second place, then its
        menu), served round robin so every item gets asked before any gets a third question."""
        per_item_q = {}
        for wt in self.wants:
            per_item_q[wt.slot] = []
        order_first = []
        for wt, pname in self.q_order:
            if wt.slot not in per_item_q or taken(wt, pname):
                continue
            ln_ = per_item_q[wt.slot]
            if any(close_names(pname, xv) for xv in ln_):
                continue
            ln_.append(pname)
            if wt.slot not in order_first:
                order_first.append(wt.slot)
        for wt in self.wants:
            options = [ent for ent in self.field_menu(wt) if ent["listings"] >= 2 or ent["of"] <= 2]
            ln_ = per_item_q[wt.slot]
            head_e = None
            for ent in self.field_menu(wt, carried=True):
                if ent["listings"] >= 3:
                    head_e = ent["property"]
                    break
            if head_e and not taken(wt, head_e) and not any(close_names(head_e, xv) for xv in ln_[:2]):
                ln_[:] = [xv for xv in ln_ if not close_names(xv, head_e)]
                ln_.insert(min(1, len(ln_)), head_e)
            for ent in options:
                nm = ent["property"]
                if taken(wt, nm) or any(close_names(nm, xv) for xv in ln_):
                    continue
                ln_.append(nm)
            if wt.slot not in order_first:
                order_first.append(wt.slot)
        heard_slots = set()
        for wt in self.wants:
            if any(rq.source == "shopper" for rq in wt.needs):
                heard_slots.add(wt.slot)
        by_slot = {wt.slot: wt for wt in self.wants}

        def turn_key(ix):
            # least-asked items first; the plan is rebuilt after every told need
            return (ix in heard_slots, len(by_slot[ix].questioned), order_first.index(ix))
        turn_no = sorted(order_first, key=turn_key)
        agenda = []
        while len(agenda) < count and any(per_item_q[ix] for ix in turn_no):
            for ix in turn_no:
                if len(agenda) >= count:
                    continue
                if per_item_q[ix]:
                    agenda.append((by_slot[ix], per_item_q[ix].pop(0)))
        return agenda

    # -- page references (values quoted from another page)
    def follow_pointer(self, item, req, hop=0):
        refd = req.pointer or {}
        if not getattr(req, "chain_key0", None):
            # remember the first page named: a later repeat points at this one
            req.chain_key0 = pointer_key(refd)
        need_left = max(4, getattr(self, "chain_floor", 0))
        if hop > 7 or self.moves_left() < need_left or _kb_short(self, 20):
            return
        vendor_s = str(refd.get("seller") or "").strip()
        cat_name = str(refd.get("category") or "").strip()
        attr_map = refd.get("attributes")
        if not isinstance(attr_map, dict):
            attr_map = {}
        wanted_prop = str(refd.get("property") or req.field).strip()
        # one search state per described page, so a retry pages further and skips listings already tried
        probe, tried_ids = _kb_ref_state(self, vendor_s, cat_name, attr_map)
        vals = _kb_ref_words(attr_map, vendor_s)
        attr_text = " ".join(vals[:3])

        def matches_attrs(cand_r):
            return _kb_attrs_match(attr_map, cand_r)

        def text_has_attrs(cand_r):
            return _kb_attrs_in_text(attr_map, cand_r)

        def current_candidates():
            pool_o = [cand_r for cand_r in probe.seen.values() if cand_r.ref not in tried_ids and _kb_vendor_ok(vendor_s, cand_r)]
            right = [cand_r for cand_r in pool_o if not cat_name or self.right_kind(probe, cand_r)]
            if not right and vendor_s and len(pool_o) <= 2 * self.side_max:
                right = pool_o          # loosely described category: trust the seller's own listings
            place = {pid: n_at for n_at, pid in enumerate(probe.seen)}
            # equal scores keep the store's order (best keyword matches first)
            return sorted(right, key=lambda cand_r: (-_kb_overlap(attr_text, vals, cand_r), not (likeness(cand_r.vendor, vendor_s) >= 0.5),
                                                place.get(cand_r.ref, 0)))
        pg = None
        for qtext in _kb_ref_queries(vendor_s, cat_name, attr_text):
            if pg is not None or self.moves_left() < need_left:
                break
            # the store shows few rows per page: up to three pages before moving on
            rounds = 3
            while rounds:
                rounds -= 1
                if self.moves_left() < need_left:
                    break
                if self.seek(probe, query=qtext, max_pages=1, budget=3) == 0:
                    break
                if current_candidates():
                    break
            hit_any = current_candidates()
            if hit_any:
                tried_ids.update(cand_r.ref for cand_r in hit_any)
                pg = self.find_page(probe, hit_any, refd, vendor_s, cat_name, attr_map, matches_attrs, text_has_attrs,
                                    need_left)
        if pg is None:
            self.log("reference_unmatched" if tried_ids else "reference_unfound", seller=vendor_s, category=cat_name)
            return
        pairing = req.rel == "counterpart of reference"
        relation_txt = str(refd.get("relation") or "").strip()

        def ask_page():
            head = f"PROPERTY WANTED: {wanted_prop}\n"
            if relation_txt:
                head += f"RELATION THE SHOPPER ASKED FOR: {relation_txt}\n"
            return _kb_fast(self, P_POINTER, head + "\nPAGE:\n" + _kb_js(pg.card(3000)), 600)

        def load_table():
            pg.props = self.pull_table(pg.ref) or {}
            return pg.props

        # stores whose rows lack a property table: read the notes first, so a page that only points on costs no table
        no_table = self.side_thin and not pg.props
        if no_table and not pg.remarks and self.moves_left() >= 3:
            load_table()
            no_table = False
        reading = ask_page()
        as_dict = isinstance(reading, dict)
        if no_table and self.moves_left() >= 3 and as_dict and not reading.get("redirect") \
                and (pairing or not reading.get("value")):
            # a pairing depends on the page's own value, which only its table states
            if load_table():
                reading = ask_page()
        self.log("reference", hop=hop, page=pg.ref, read=brief(reading, 200))
        onward = reading.get("redirect") if isinstance(reading, dict) else None
        if not (isinstance(onward, dict) and onward):
            onward = None
        if onward is not None:
            about_prop = str(onward.get("property") or wanted_prop)
            if not _kb_same_prop(about_prop, wanted_prop):
                # the notes redirect some other property: the wanted one is read off this page
                onward = None
                if no_table and not pg.props and not reading.get("value") and self.moves_left() >= 3:
                    reread = ask_page() if load_table() else None
                    if isinstance(reread, dict):
                        for fld in ("value", "paired"):
                            reading[fld] = reread.get(fld) or reading.get(fld)
                if not reading.get("value"):
                    own_v = next((pv for pk, pv in pg.props.items() if _kb_near(str(pk).replace("_", " "), wanted_prop)),
                                 None)
                    if own_v not in (None, ""):
                        reading["value"] = str(own_v)
                self.log("reference_other_redirect", about=about_prop, wanted=wanted_prop, value=reading.get("value"))
        if onward is None:
            _kb_apply_page_value(self, item, req, refd, reading, pairing, relation_txt, pg)
            return
        req.pointer = onward
        onward["property"] = wanted_prop
        if relation_txt:
            onward["relation"] = relation_txt
        if refd.get("tolerance"):
            onward["tolerance"] = refd["tolerance"]
        self.follow_pointer(item, req, hop + 1)

    def find_page(self, scratch, found, ref, seller, category, attrs,
                    attrs_match, attrs_in_text, floor=0):
        """Pick the listing a reference describes from the candidates, reading them a batch at a time and never below
        the caller's step floor (the ending keeps its steps)."""
        floor = max(3, floor)
        pg = None
        attempted = 0
        choice = str(ref.get("pick") or "").strip()
        qty_n = ref.get("count")
        if not isinstance(qty_n, int):
            qty_n = None

        def is_read(ofr):
            fetched = scratch.seen.get(ofr.ref)
            return bool(fetched and fetched.read_full)

        def describe():
            return ("DESCRIPTION: category " + category + "; seller " + seller + "; attributes "
                    + _kb_js(attrs))
        if choice or (qty_n or 0) > 1:
            # choosing among several seller listings: read them all first (a few more than the stated count)
            upto = min(len(found), 3 * self.side_max, max(4, 2 * (qty_n or 2)))
            for t_start in range(0, upto, self.side_max):
                if self.moves_left() < floor:
                    break
                queue_ids = []
                for xv in found[t_start:min(t_start + self.side_max, upto)]:
                    if not is_read(xv):
                        queue_ids.append(xv.ref)
                for ofr in self.open_offers(None, queue_ids, floor=floor):
                    scratch.seen[ofr.ref] = ofr
            cardlist = []
            for ofr in found:
                if ofr.ref in scratch.seen and scratch.seen[ofr.ref].read_full:
                    cardlist.append(scratch.seen[ofr.ref].card(1200))
            if cardlist:
                body_text = describe()
                if choice:
                    body_text += f"\nPICK: {choice}"
                if qty_n:
                    body_text += f"\nLISTINGS THAT FIT: {qty_n}"
                body_text += "\n\nCANDIDATES:\n" + _kb_js(cardlist)
                verdict = _kb_fast(self, P_MATCH, body_text, 600)
                if isinstance(verdict, dict) and verdict.get("id") in scratch.seen:
                    pg = scratch.seen[verdict["id"]]
            # pages already read and rejected are not opened again
            if pg is not None:
                found = []
            else:
                found = [ofr for ofr in found if not is_read(ofr)]
        one_by_one = not self.role_map.get("compare") or getattr(self, "side_off", False)
        step_size = self.side_max
        rounds = 3
        if one_by_one:
            step_size = 1
            rounds = 5
        pending = []
        while found and attempted < rounds and pg is None and self.moves_left() >= floor:
            chunk_l = found[:step_size]
            found = found[step_size:]
            attempted += 1
            for ofr in self.open_offers(None, [ofr.ref for ofr in chunk_l], floor=floor):
                scratch.seen[ofr.ref] = ofr
            batch_read = [scratch.seen[ofr.ref] for ofr in chunk_l if ofr.ref in scratch.seen and scratch.seen[ofr.ref].read_full]
            exact_hit = None
            for ofr in batch_read:
                if attrs_match(ofr):
                    exact_hit = ofr
                    break
            if exact_hit is not None:
                pg = exact_hit
                break
            pending += batch_read
            if one_by_one and found and len(pending) < 3:
                continue          # gather a few single pages before one model comparison
            displayed = pending if one_by_one else batch_read
            pending = []
            cardlist = [ofr.card(1600) for ofr in displayed]
            if not cardlist:
                continue
            body_text = describe() + "\n\nCANDIDATES:\n" + _kb_js(cardlist)
            verdict = _kb_fast(self, P_MATCH, body_text, 600)
            text_hits = [ofr for ofr in displayed if attrs_in_text(ofr)]
            if isinstance(verdict, dict):
                if verdict.get("id") in scratch.seen:
                    pg = scratch.seen[verdict["id"]]
                    continue
                if "id" in verdict:
                    continue                  # the matcher saw them all and none fits: keep looking
            if len(text_hits) == 1:
                pg = text_hits[0]          # the one whose own text carries every described value
            elif len(cardlist) == 1 and not found:
                pg = scratch.seen[cardlist[0]["id"]]
        return pg

    # -- shopper lines that arrive inside tool results
    def read_news(self):
        """Read the queued shopper lines once; True when the task changed."""
        pending_lines = [ln for ln in self.news_queue if ln not in self.seen_news]
        self.news_queue = []
        if not pending_lines:
            return False
        self.seen_news.update(pending_lines)
        self.news_log.extend(pending_lines)
        prompt_txt = "LINES:\n- " + "\n- ".join(pending_lines)
        prompt_txt += "\n\nREQUESTED ITEMS: " + json.dumps([wq.label for wq in self.wants])
        ans = _kb_fast(self, P_NEWS, prompt_txt, 700)
        self.log("event", lines=[ln[:160] for ln in pending_lines], read=brief(ans, 300))
        if not isinstance(ans, dict):
            self.odd_news.extend(pending_lines)
            return False
        entries = [chg_e for chg_e in (ans.get("changes") or []) if isinstance(chg_e, dict)]
        sorts = {str(chg_e.get("kind") or "").lower() for chg_e in entries}
        if not sorts - {"other", ""}:
            # nothing to act on here: the pilot sees these lines on its next look
            self.odd_news.extend(pending_lines)

        def aimed_at(chg_e):
            if not chg_e.get("item"):
                return None
            closest = max(self.wants, key=lambda wq: likeness(wq.label, chg_e.get("item")))
            if len(self.wants) > 1 and likeness(closest.label, chg_e.get("item")) < 0.2:
                return None
            return closest
        touched = False
        for chg_e in entries:
            sort = str(chg_e.get("kind") or "").lower()
            aim = aimed_at(chg_e)
            scope = [aim] if aim else self.wants
            if sort == "budget" and as_amount(chg_e.get("budget")):
                self.limit = _kb_fix_budget(as_amount(chg_e.get("budget")), pending_lines, self.limit)
                touched = True
            elif sort == "drop" and chg_e.get("property"):
                if self.let_go(scope, str(chg_e["property"])):
                    touched = True
            elif sort == "add" and chg_e.get("property") and chg_e.get("value"):
                for wq in ([aim] if aim else self.wants[:1]):
                    wq.needs.append(Need(chg_e["property"], chg_e.get("op") or "is", chg_e["value"], None, origin="shopper"))
                    touched = True
            elif sort in ("price", "stock"):
                touched = True
                self.recheck_cart()
            elif sort == "step" and isinstance(chg_e.get("step"), dict):
                self.add_demand(chg_e["step"])
        if touched:
            self.moved_at = self.moves       # last step at which the request changed
        return touched

    def let_go(self, targets, prop):
        """Remove a need the shopper gave up, from the items and from every stored verdict; True if any went."""
        dropped = False
        for itm in targets:
            snap = len(itm.needs)
            itm.needs = [rq for rq in itm.needs if likeness(rq.field, prop) < 0.5]
            if len(itm.needs) == snap:
                continue
            dropped = True
            for verdict in itm.rulings.values():
                for kname in ("failed", "unsure", "weak", "loose"):
                    if not verdict.get(kname):
                        continue
                    verdict[kname] = {fx for fx in verdict[kname] if likeness(str(fx).split(":")[0], prop) < 0.5}
        return dropped

    def add_demand(self, step):
        """Add a step the shopper asked for mid-task to the list the gate checks."""
        if not step.get("goal") or len(self.demands) >= 8:
            return
        entry = dict(step)
        entry["moment"] = str(entry.get("moment") or "before_finish").lower()
        is_msg = str(entry.get("kind") or "").lower() == "message"
        entry["kind"] = "message" if is_msg else "tool"
        for step_d in self.demands:
            if likeness(entry["goal"], step_d.get("goal")) >= 0.6:
                return
        self.demands.append(entry)
        candidates = entry["kind"] == "message" and entry["moment"] in ("before_order", "before_finish")
        self.approval_due = self.approval_due or candidates
        self.log("step_added", step=entry)

    def recheck_cart(self):
        for pid, itm in list(self.basket_owner.items()):
            if self.moves_left() <= 2:
                return
            self.open_offers(itm, [pid], page=True)
            fetched = itm.seen[pid] if pid in itm.seen else None
            self.log("refreshed", id=pid, price=None if fetched is None else fetched.cost_each,
                     stock=None if fetched is None else fetched.in_stock)

    # -- cart and ending
    def replace_cart(self, lines):
        """Overwrite the cart with exactly these lines (id -> quantity) through the replacing tool."""
        cart_tool = self.role_map.get("cart_set") or {}
        line_arg = cart_tool.get("lines") or ""
        qty_arg = cart_tool.get("qty")
        line_schema = (self.schema_of("cart_set").get("properties") or {}).get(line_arg) or {}
        cap_n = (((line_schema.get("items") or {}).get("properties") or {}).get(qty_arg or "") or {}).get("maximum")
        payload_rows = []
        for line_ref, qty_n in lines.items():
            entry = {cart_tool["id"]: line_ref}
            if qty_arg:
                entry[qty_arg] = qty_n if not isinstance(cap_n, int) else min(qty_n, cap_n)
            payload_rows.append(entry)
        reply_d = self.use_tool(cart_tool["name"], {cart_tool["lines"]: payload_rows})
        if reply_d["ok"]:
            return reply_d["ok"]
        self.log("cart_set_failed", error=reply_d["error"], obs=brief(reply_d["obs"], 200))
        return reply_d["ok"]

    def put_in_cart(self, item, listing):
        return _kb_cart_change(self, listing.ref, item)

    def take_out(self, listing_id):
        return _kb_cart_change(self, listing_id, None)

    def check_out(self):
        cart_tool = self.role_map.get("checkout")
        if not cart_tool:
            return False
        if self.basket and self.demands:
            intent = "check out the basket below as the order"
            step_ran = self.process_check("before_order", intent)
            if step_ran and self.owed_messages() and self.moves_left() >= 2:
                # steps ran yet a demanded message for this basket is still owed
                step_ran = self.process_check("before_order", intent) or step_ran
            if step_ran is not False and self.approval_due and self.approved_key != self.basket_key():
                step_ran = self.show_basket() or step_ran
            if step_ran and (self.read_news() or self.pick_moved()):
                # the reply carried a change: let the caller choose again
                self.log("procedure_changed")
                return False
        self.show_parts()
        reply_d = self.use_tool(cart_tool["name"], self.preset_args("checkout"))
        if self.done:
            self.log("ordered", cart=self.basket)
            return True
        self.log("order_refused", obs=brief(reply_d["obs"], 200), lines=reply_d["lines"][:3])
        self.latest_refusal = _kb_refusal_note(cart_tool["name"], reply_d)
        return False

    def show_parts(self):
        """Before ordering, search once for part codes that a cart line's text points to and no seen title carries."""
        cfg = self.role_map.get("search_text")
        if not cfg or not cfg.get("query") or not self.basket:
            return
        displayed = self.__dict__.setdefault("parts_shown", set())
        series_heads = []
        for wt in self.wants:
            for ofr in wt.seen.values():
                series_heads.append(ofr.heading or "")
        all_titles = " ".join(series_heads)
        part_codes = []
        for pid in self.basket:
            owner_w = None
            for wt in self.wants:
                if pid in wt.seen:
                    owner_w = wt.seen[pid]
                    break
            if owner_w is None:
                continue
            for mt in SEE_OTHER.finditer(" ".join([owner_w.blurb or "", owner_w.remarks or ""])):
                code_re = mt.group(1).strip("-")
                if len(code_re) < 4 or not re.search(r"\d", code_re):
                    continue
                if code_re.lower() in displayed or code_re in part_codes:
                    continue
                pat = r"(?<![A-Za-z0-9])" + re.escape(code_re) + r"(?![A-Za-z0-9])"
                if re.search(pat, owner_w.heading or "", re.I):
                    continue          # the listing's own code
                if re.search(pat, all_titles, re.I):
                    displayed.add(code_re.lower())
                    continue
                part_codes.append(code_re)
        if not part_codes:
            return
        if self.moves_left() < self.owed_messages() + 2:
            return
        call_args = {}
        mo = cfg.get("mode")
        if isinstance(mo, dict) and mo.get("arg"):
            call_args[mo["arg"]] = mo.get("value")
        head = part_codes[:5]
        call_args[cfg["query"]] = self.cut_to_fit("search_text", cfg["query"], " ".join(head))
        resp = self.use_tool(cfg["name"], call_args)
        for cnd in head:
            displayed.add(cnd.lower())
        self.log("pointed_parts", codes=head, rows=len(record_rows(resp["obs"]) or []))

    def closest_offer(self, item):
        """Nearest miss for a request that accepts one: the read listing of the right kind with the fewest failed
        requirements (sold out, over the item's cap, store-field failures each count), then fewest unsure, then
        cheapest."""
        ftable = self.store_fields(item)
        top_price = None
        best_rank_key = None
        for ofr in item.seen.values():
            if not ofr.read_full or ofr.ref not in item.rulings:
                continue
            if not self.right_kind(item, ofr):
                continue
            verdict = item.rulings[ofr.ref]
            fails = set(verdict.get("failed") or ())
            if "cart refused" in fails:
                continue
            misses = len(fails | self.field_misses(item, ofr, ftable))
            if ofr.in_stock is False:
                misses += 1
            if item.too_dear(ofr):
                misses += 1
            unit_price = ofr.cost_each
            if self.limit is not None and unit_price is not None and unit_price * item.count > self.limit + 1e-6:
                misses += 1
            rank = (misses, len(verdict.get("unsure") or ()), 1e18 if unit_price is None else unit_price)
            if best_rank_key is None or rank < best_rank_key:
                top_price = ofr
                best_rank_key = rank
        return top_price

    def buy_closest(self):
        """Request allows the nearest listing when nothing fits: cart each item's pick (chosen, best fit, or nearest
        miss) and order."""
        if getattr(self, "closest_tried", False) or not self.wants or not self.role_map.get("checkout"):
            return False
        self.closest_tried = True
        preset = getattr(self, "close_picks", None) or {}
        lineup = {}
        for wq in self.wants:
            cand_o = preset.get(wq.slot)
            if cand_o is None:
                ranking = _kb_ranked(self, wq)
                cand_o = ranking[0] if ranking else self.closest_offer(wq)
            if cand_o is not None:
                lineup[wq.slot] = cand_o
        if not lineup:
            return False
        covered_w = [wq for wq in self.wants if wq.slot in lineup]
        lineup_refs = {cand_r.ref for cand_r in lineup.values()}
        to_drop = [pid2 for pid2 in self.basket if pid2 not in lineup_refs]
        to_put = [wq for wq in covered_w if lineup[wq.slot].ref not in self.basket]
        self.log("nearest_order", picks={wq.label: lineup[wq.slot].ref for wq in covered_w})
        for pid2 in to_drop:
            if self.moves_left() > len(to_put) + 1:
                self.take_out(pid2)
        for wq in to_put:
            if self.moves_left() < 2:
                break
            wq.pick = lineup[wq.slot]
            self.put_in_cart(wq, wq.pick)
        if self.moves_left() < 1 or not self.basket:
            return False
        moves_at = self.moves
        if self.check_out() or self.done:
            return True
        # refused: what the checkout brought (a dropped need, a larger budget) may let something fit now
        self.closest_rounds = getattr(self, "closest_rounds", 0) + 1
        news_or_move = self.read_news() or getattr(self, "moved_at", -1) >= moves_at
        if news_or_move and self.closest_rounds <= 2 and self.moves_left() >= len(self.wants) + 1:
            # unsure fits may be ordered again: the change can leave them the best known
            self.closest_tried = False
            self.unsure_done = False
            self.close_picks = None
            self.pick_and_close()
            return self.done
        return False

    def buy_unsure(self):
        """No rule for the nothing-fits case: order listings no requirement plainly fails (some left unconfirmed)
        instead of declining, provided each item has one and the total fits the budget."""
        if getattr(self, "unsure_done", False) or getattr(self, "closest_tried", False):
            return False
        if not self.role_map.get("checkout"):
            return False
        self.unsure_done = True
        choices = {}
        for itm in self.wants:
            ordered_ = _kb_ranked(self, itm)
            if not ordered_:
                ftable = self.store_fields(itm)
                ordered_ = []
                for ofr in itm.good_offers(include_unsure=True, cheapest=self.lowest_price):
                    if ofr.in_stock is False or self.field_misses(itm, ofr, ftable) or itm.loose_words(ofr.ref):
                        continue
                    if "cart refused" in _kb_verdict(itm, ofr).get("failed", ()):
                        continue
                    ordered_.append(ofr)
            if not ordered_:
                return False
            choices[itm.slot] = ordered_[0]
        tot = 0
        for wt in self.wants:
            tot += (choices[wt.slot].cost_each or 0) * wt.count
        if self.limit is not None and tot > self.limit + 1e-6:
            return False
        self.log("unsure_order", picks={wt.label: choices[wt.slot].ref for wt in self.wants})
        self.close_picks = choices
        return self.buy_closest()

    def walk_away(self, reason=""):
        decline = self.role_map.get("decline")
        if self.last_review(reason):
            self.pick_and_close()
            if self.done:
                return True
        if self.closest_ok:
            if self.buy_closest():
                return True
        elif not getattr(self, "exit_stated", False) and self.buy_unsure():
            return True
        if self.done:
            return True
        if not decline:
            return False
        if self.demands and not getattr(self, "exit_gated", False):
            self.exit_gated = True
            self.process_check("before_decline", "end the task without buying anything")
        payload = self.preset_args("decline")
        schema_props = self.schema_of("decline").get("properties") or {}
        note_slot = decline.get("remark")
        if not note_slot or note_slot not in schema_props:
            # unnamed free-text argument: the single open string field left after the fixed ones
            open_text = [pk for pk, pv in schema_props.items()
                         if isinstance(pv, dict) and pv.get("type") == "string" and not pv.get("enum") and pk not in payload]
            note_slot = None if len(open_text) != 1 else open_text[0]
        if note_slot and note_slot in schema_props:
            # the decline carries its reason when it can (what was sought, the budget, why nothing fit; no ids,
            # since a note citing one is graded as a missed step)
            payload[note_slot] = self.cut_to_fit("decline", note_slot, self.walk_note(reason or self.exit_reason))
        reply_d = self.use_tool(decline["name"], payload)
        self.log("declined", reason=reason or self.exit_reason)
        if self.done or getattr(self, "exit_refused", False) or self.moves_left() < 1:
            return self.done
        # the store refused: a change announced at the close (budget, price) may make something fit
        self.exit_refused = True
        self.log("decline_refused", obs=brief(reply_d["obs"], 200), lines=reply_d["lines"][:3])
        self.latest_refusal = _kb_refusal_note(decline["name"], reply_d)
        n_items = len(self.wants)
        # lines the closing call carried count as a change even if already read
        news_seen = self.read_news() or reply_d["lines"] or getattr(self, "moved_at", -1) >= self.moves - 1
        if news_seen and self.moves_left() >= n_items + 2:
            # listings ruled out before (old budget, a dropped need) may fit now: look once more
            self.review_due = True
            again_now = True
        else:
            again_now = self.moves_left() >= n_items + 3 and self.steer(
                "The store refused to close the task with no purchase.", self.latest_refusal, turns=4, refusal=True)
        if again_now:
            self.final_hunt = False
            self.reviewed = False
            self.pick_and_close()
        if self.done:
            return self.done
        # some stores refuse to close with items in the cart: empty it first
        for pid2 in list(self.basket):
            if self.moves_left() > 1:
                self.take_out(pid2)
        self.use_tool(decline["name"], payload)
        return self.done

    def walk_note(self, reason):
        """Plain closing note: what was sought, the budget, and why nothing was bought."""
        name_parts = []
        for wt in self.wants:
            name_parts.append(f"{wt.count} x {wt.label}" if wt.count > 1 else wt.label)
        sought = ", ".join(name_parts) or "the request"
        note = f"I looked through the store for {sought}"
        if self.limit is not None:
            note += f" with at most {self.limit:g} to spend"
        note += ", and no order will be placed: "
        lead = "no fitting listing for "
        reason_txt = ""
        if reason.startswith(lead):
            reason_txt = "no listing I turned up for " + reason[len(lead):] + " meets every requirement"
        elif reason.startswith("the cheapest fitting combination"):
            reason_txt = "the listings that meet every requirement cost more together than that"
        reason_txt = re.sub(r"\bp-[0-9a-f]{6,}\b", "", reason_txt).strip()
        return note + (reason_txt or "none of what the store shows meets every requirement") + "."

    # ------------------------------------------------------------------ pilot
    def helm_reserve(self):
        """Steps kept back from the pilot: one cart add per item, the closing call, a retry, and owed messages."""
        return self.owed_messages() + len(self.wants) + 2

    def spare_tools(self):
        """Store tools that no routine role covers."""
        used_names = set()
        for cfg in self.role_map.values():
            if isinstance(cfg, dict):
                used_names.add(cfg.get("name"))
        added_q = []
        for tt in self.tool_list:
            fdef = tt.get("function")
            if not isinstance(fdef, dict):
                fdef = tt
            if not isinstance(fdef, dict) or not fdef.get("name"):
                continue
            nm = fdef["name"]
            if nm in used_names:
                continue
            added_q.append({"name": nm, "description": str(fdef.get("description") or "")[:300],
                          "parameters": brief(fdef.get("parameters") or {}, 300), "called": nm in self.tools_called})
        return added_q

    def look_due(self):
        if getattr(self, "looked", False):
            return False
        if _kb_short(self, 40) or self.desk.dollars >= HELM_LINE:
            return False
        if self.moves_left() < self.helm_reserve() + 1:
            return False
        if len(self.odd_news) > getattr(self, "news_shown", 0):
            return True
        for tt in self.spare_tools():
            if not tt["called"]:
                return True
        return False

    def brief_state(self):
        """What the pilot reads: allowances, each item's needs and findings, the cart, store lines, recent calls."""
        held_back = self.helm_reserve()
        asks = 0
        if self.may_ask():
            asks = 1 if self.msg_cap is None else self.msg_cap - self.msgs_used - self.kept_messages()

        def cd(ofr):
            return {"id": ofr.ref, "title": ofr.heading[:90], "price": ofr.cost_each}

        def miss_key(rq):
            return (len(rq["failed"]), 1e18 if rq["price"] is None else rq["price"])
        item_blocks = []
        for wt in self.wants:
            fitting = self.can_buy(wt)
            fit_refs = set()
            for ofr in fitting:
                fit_refs.add(ofr.ref)
            close_misses = []
            doubtful = []
            for ofr in wt.seen.values():
                if not ofr.read_full or ofr.ref in fit_refs:
                    continue
                if not self.right_kind(wt, ofr):
                    continue
                verdict = _kb_verdict(wt, ofr)
                recd = cd(ofr)
                if ofr.in_stock is False:
                    recd["can_buy"] = False
                if verdict.get("failed"):
                    recd["failed"] = sorted(verdict["failed"])[:3]
                    close_misses.append(recd)
                elif verdict.get("unsure") or wt.loose_words(ofr.ref):
                    gaps = set(verdict.get("unsure") or ()) | set(wt.loose_words(ofr.ref))
                    recd["not_shown"] = sorted(gaps)[:3]
                    doubtful.append(recd)
            close_misses.sort(key=miss_key)
            not_read = []
            for ofr in self.pool_rows(wt):
                if not ofr.read_full and ofr.triage_mark != "no":
                    not_read.append(cd(ofr))
            block = {"name": wt.label, "category": wt.kind, "quantity": wt.count}
            if wt.unit_limit is not None:
                block["price_cap_each"] = wt.unit_limit
            block["requirements"] = wt.need_lines()
            block["asked_shopper_about"] = wt.questioned[:8]
            block["fits"] = [cd(ofr) for ofr in fitting[:4]]
            block["unsure"] = doubtful[:3]
            block["near_misses"] = close_misses[:5]
            block["rows_not_read"] = not_read[:6]
            block["queries_tried"] = sorted(_kb_texts_tried(wt))[:10]
            block["search_complete"] = bool(wt.dry or wt.values_all_seen)
            item_blocks.append(block)
        steps_now = self.moves_left()
        allow = {"steps_left": steps_now, "steps_you_may_use": max(0, steps_now - held_back), "questions_left": max(0, asks),
                 "seconds_left": max(0, int(self.seconds_left() - CLOSE_MARGIN))}
        pg_state = {"allowances": allow, "budget_total": self.limit, "cheapest_wanted": self.lowest_price,
                                 "best_by": self.best_by, "items": item_blocks}
        pg_state["cart"] = [{"id": pid, "quantity": qtext} for pid, qtext in self.basket.items()]
        pg_state["store_lines"] = [ofr[:240] for ofr in self.news_log[-8:]]
        pg_state["lines_the_routine_found_no_change_in"] = [ofr[:240] for ofr in self.odd_news[-4:]]
        pg_state["latest_calls"] = self.journal[-6:]
        pg_state["other_tools"] = self.spare_tools()
        return pg_state

    def steer(self, point, detail=None, turns=HELM_MOVES, refusal=False):
        """Let a model drive a few moves where the routine is stuck: one move per turn, run by the routine's engines
        inside the steps the ending needs. True when the moves changed what can be bought."""
        if self.done or self.halted or self.style != "full":
            return False
        self.helm = {"accepts": 0, "pushed": False, "refusal": refusal, "news": False}
        self.news_shown = len(self.odd_news)
        history = []
        gained = False
        for idx in range(turns):
            if self.done or _kb_short(self, 30) or self.desk.dollars >= HELM_LINE:
                break
            ans = _kb_fast(self, P_HELM, _kb_helm_text(self, point, detail, history), 2000)
            step_d = ans.get("move") if isinstance(ans, dict) else None
            if not isinstance(step_d, dict):
                break
            verb = str(step_d.get("do") or "").lower()
            if verb and verb != "finish":
                said, did_move = self.steer_move(verb, step_d)
                if did_move:
                    gained = True
                self.log("pilot", point=point[:40], turn=idx, move=brief(step_d, 200), report=said[:300],
                         think=str(ans.get("think") or "")[:200])
                history.append(f"{idx + 1}. {brief(step_d, 240)} -> {said[:700]}")
                continue
            holdmsg = self.hold_back()
            self.log("pilot", point=point[:40], turn=idx, move="finish", held=holdmsg)
            if not holdmsg:
                break
            history.append(f"{idx + 1}. finish -> not yet: {holdmsg}")
        if gained:
            self.news_shown = len(self.odd_news)
        return gained

    def hold_back(self):
        """Once per session, refuse a finish while some item has no fit, an incomplete search and unread promising
        rows that the steps could still cover."""
        pg_state = self.helm
        if pg_state["pushed"]:
            return ""
        if self.moves_left() - self.helm_reserve() < 1:
            return ""
        for wt in self.wants:
            if wt.dry or wt.values_all_seen or self.can_buy(wt):
                continue
            recs = []
            for ofr in self.pool_rows(wt):
                if not ofr.read_full and ofr.triage_mark != "no":
                    recs.append(ofr.ref)
            recs = recs[:4]
            if not recs:
                continue
            pg_state["pushed"] = True
            return f"no fit for {wt.label} and its search is not complete; rows not read: {recs}"
        return ""

    def steer_move(self, do, move):
        """Run one pilot move; returns (its outcome, whether the buyable set changed)."""
        itm = self.wanted_named(move.get("item"))
        if not itm:
            itm = self.wants[0] if len(self.wants) == 1 else None
        keep_back = self.helm_reserve()
        if do in ("search", "read", "ask", "tool") and self.moves_left() - keep_back < 1:
            return _kb_no("not done: the steps left are kept for the ending")
        if do in ("search", "read", "ask", "accept") and itm is None:
            return _kb_no("not done: name one of the items exactly")

        def fits_now():
            return {cand_r.ref for cand_r in self.can_buy(itm)}

        def open_and_judge(id_list):
            got_back = self.open_offers(itm, id_list, floor=keep_back) if id_list else []
            self.assess(itm, [cand_r for cand_r in got_back if cand_r.read_full])
            return got_back

        def do_search():
            text_q = _kb_squash(str(move.get("query") or ""))[:60]
            if not text_q or flat_text(text_q) in {flat_text(tq) for tq in _kb_texts_tried(itm)}:
                return _kb_no("not done: no query, or one tried already")
            if not self.role_map.get("search_text"):
                return _kb_no("not done: the store has no keyword search")
            seen_before = set(itm.seen)
            base_fits = fits_now()
            self.seek(itm, query=text_q, max_pages=1, budget=keep_back)
            unseen = [cand_r for cand_r in self.pool_rows(itm) if not cand_r.read_full and cand_r.ref not in seen_before]
            self.triage(itm, unseen)
            hopeful = sorted((cand_r for cand_r in unseen if cand_r.triage_mark != "no"),
                             key=lambda cand_r: (cand_r.triage_mark != "yes", _kb_price(cand_r)))
            n_open = self.side_max if self.opens_many() else 2
            got_back = open_and_judge([cand_r.ref for cand_r in hopeful[:n_open]])
            return self.move_report(itm, len(set(itm.seen) - seen_before), got_back, base_fits)

        def do_read():
            wanted_ids = []
            for raw_id in move.get("ids") or []:
                if raw_id and len(wanted_ids) < 4:
                    wanted_ids.append(str(raw_id))
            unread_ids = [ii for ii in wanted_ids if not getattr(itm.seen.get(ii), "read_full", False)]
            if not unread_ids:
                return _kb_no("not done: no unread listing named")
            base_fits = fits_now()
            return self.move_report(itm, 0, open_and_judge(unread_ids), base_fits)

        def do_ask():
            field_q = _kb_squash(str(move.get("property") or "").replace("_", " ")).lower()[:40]
            if not field_q or not self.may_ask():
                return _kb_no("not done: no question left, or no property named")
            asked_before = any(close_names(field_q, q_) for q_ in itm.questioned)
            if asked_before or any(close_names(field_q, nd_.field) for nd_ in itm.needs):
                return _kb_no("not done: asked or known already")
            known_n = len(itm.needs)
            heard = self.put_question(itm, field_q)
            self.take_reply_once(itm, field_q, heard)
            self.read_news()
            new_lines = [nd_.line() for nd_ in itm.needs[known_n:]]
            if new_lines:
                self.assess(itm, [cand_r for cand_r in itm.seen.values() if cand_r.read_full])
            return f"reply read as {brief(heard, 200)}; requirements now {itm.need_lines()}", bool(new_lines)

        def do_tool():
            tool_name = str(move.get("name") or "")
            tool_args = move.get("args") if isinstance(move.get("args"), dict) else {}
            my_roles = set(self.roles_of(tool_name))
            if tool_name not in self.arg_specs:
                return _kb_no("not done: no such tool")
            if my_roles & {"checkout", "decline", "cart_add", "cart_set"}:
                return _kb_no("not done: the routine itself fills the cart and ends the task")
            if "message" in my_roles:
                if not self.may_ask() and not (self.helm["refusal"] and self.may_message()):
                    return _kb_no("not done: no message left to spend")
                self.msgs_used += 1
            elif my_roles & {"search_text", "search_fields", "open", "facts", "compare"}:
                return _kb_no("not done: use the search or read move for this tool")
            reply_d = self.use_tool(tool_name, tool_args)
            changed_flag = self.read_news()
            tail = f" lines {reply_d['lines'][:3]}" if reply_d["lines"] else ""
            return f"result {brief(reply_d['obs'], 1500)}{tail}", changed_flag

        def do_accept():
            offer = itm.seen.get(str(move.get("id") or ""))
            if offer is None or not offer.read_full:
                return _kb_no("not done: read the listing first")
            if offer.ref in fits_now():
                return _kb_no("not done: it is a fit already")
            if self.helm["accepts"] >= 2:
                return _kb_no("not done: two claims were checked already")
            if (offer.in_stock is False or itm.too_dear(offer) or not self.right_kind(itm, offer)
                    or self.field_misses(itm, offer) or _kb_over_cap(self.limit, offer, itm.count)):
                return _kb_no("not done: it cannot be bought within the allowances or a store field fails it")
            self.helm["accepts"] += 1
            if not self.check_offer(itm, offer, strict=True):
                return _kb_no("the checker does not agree: it stays out")
            itm.rulings[offer.ref] = _kb_clean_ruling()
            itm.assessed_on[offer.ref] = set(itm.need_lines())
            self.log("pilot_accept", item=itm.label, id=offer.ref)
            return _kb_ok("the checker agrees: it is a fit now")
        handlers = {"search": do_search, "read": do_read, "ask": do_ask, "tool": do_tool, "accept": do_accept}
        if do in handlers:
            return handlers[do]()
        if do == "news":
            return self.apply_news(move)
        return _kb_no("not done: unknown move")

    def move_report(self, item, rows, opened, fits_before):
        verdict_map = {}
        for ofr in opened:
            val = _kb_verdict(item, ofr)
            if val.get("failed"):
                said = "fails " + "; ".join(sorted(val["failed"])[:2])
            elif val.get("unsure"):
                said = "not shown: " + "; ".join(sorted(val["unsure"])[:2])
            else:
                said = "meets every requirement"
            verdict_map[ofr.ref] = said
        fitting = [ofr.ref for ofr in self.can_buy(item)]
        gained = False
        for ix in fitting:
            if ix not in fits_before:
                gained = True
                break
        return f"{rows} new rows; read {verdict_map or 'none'}; fits now {fitting[:4]}", gained

    def apply_news(self, move):
        """A change the pilot claims from a store line: the line must exist and must state the change itself (the new
        amount, or the dropped requirement)."""
        claim = str(move.get("line") or "").strip()
        quoted = None
        if claim:
            quoted = next((ln for ln in reversed(self.news_log) if claim[:60] in ln or likeness(claim, ln) >= 0.6), None)
        if quoted is None:
            return _kb_no("not done: no store line says that")
        sort = str(move.get("kind") or "").lower()

        def mark(with_news):
            self.moved_at = self.moves
            if with_news:
                self.helm["news"] = True
        if sort == "budget":
            amount_v = as_amount(move.get("budget"))
            stated_l = [float(tq.replace(",", "")) for tq in re.findall(r"\d[\d,]*(?:\.\d+)?", quoted)]
            if amount_v is None or amount_v not in stated_l:
                return _kb_no("not done: the line does not state that amount")
            self.limit = amount_v
            mark(True)
            self.log("pilot_news", change="budget", budget=amount_v)
            return _kb_ok(f"budget now {amount_v:g}")
        if sort == "drop":
            field_q = str(move.get("property") or "")
            who = self.wanted_named(move.get("item"))
            scope = [who] if who else self.wants
            bag_line = word_bag(quoted)
            matching = [nd_ for wq in scope for nd_ in wq.needs if likeness(nd_.field, field_q) >= 0.5]
            if not any(word_bag(nd_.field) & bag_line or word_bag(nd_.want) & bag_line for nd_ in matching):
                return _kb_no("not done: the line does not name that requirement")
            if not self.let_go(scope, field_q):
                return _kb_no("not done: no such requirement")
            mark(True)
            self.log("pilot_news", change="drop", property=field_q)
            return _kb_ok(f"requirement on {field_q} dropped")
        if sort == "gone":
            lid_q = str(move.get("id") or "")
            owner_hit = next((wq.seen[lid_q] for wq in self.wants if lid_q in wq.seen), None)
            in_play = owner_hit is not None and (lid_q in self.basket or any(wq.pick is owner_hit for wq in self.wants))
            if not in_play:
                return _kb_no("not done: name a listing in the cart or picked")
            owner_hit.in_stock = False
            mark(False)
            self.log("pilot_news", change="gone", id=lid_q)
            return _kb_ok(f"{lid_q} marked as not buyable")
        return _kb_no("not done: kind must be budget, drop or gone")

    def second_reading(self):
        """Before declining: for each item with no fit, the independent checker rereads at most two listings which fail
        nothing plainly (only unconfirmed); one it passes becomes a fit."""
        hit_any = False
        for itm in self.wants:
            if self.can_buy(itm) or _kb_short(self, 25):
                continue
            ftable = self.store_fields(itm)
            done_ids = getattr(self, "rechecked", set())
            bucket = []
            for ofr in itm.seen.values():
                val = _kb_verdict(itm, ofr)
                if not ofr.read_full or val.get("failed"):
                    continue
                if not (val.get("unsure") or itm.loose_words(ofr.ref)):
                    continue
                if ofr.in_stock is False or itm.too_dear(ofr) or not self.right_kind(itm, ofr):
                    continue
                if self.field_misses(itm, ofr, ftable) or _kb_over_cap(self.limit, ofr, itm.count):
                    continue
                if ofr.ref in done_ids:
                    continue
                bucket.append(ofr)

            def doubt(ofr, itm=itm):
                return (len(_kb_verdict(itm, ofr).get("unsure") or ()), _kb_price(ofr))
            bucket.sort(key=doubt)
            for ofr in bucket[:2]:
                self.__dict__.setdefault("rechecked", set()).add(ofr.ref)
                if not self.check_offer(itm, ofr, strict=True):
                    continue
                itm.rulings[ofr.ref] = _kb_clean_ruling()
                itm.assessed_on[ofr.ref] = set(itm.need_lines())
                self.log("recheck_fit", item=itm.label, id=ofr.ref)
                hit_any = True
                break
        return hit_any

    def last_review(self, reason):
        """Last look before closing without a purchase: recheck unsure listings, then let the pilot review the record
        (once, and again after a refused exit). True when the caller should choose again."""
        if self.done or not self.review_due or reason in ("forced", "the cart tool refuses its arguments"):
            return False
        if not self.wants or _kb_short(self, 30):
            return False
        self.review_due = False
        hit_any = self.second_reading()
        if not hit_any and self.moves_left() >= self.helm_reserve() and self.desk.dollars < HELM_LINE:
            reason_txt = "Nothing it may buy was found for some product: " + (reason or "no fit")
            hit_any = self.steer(reason_txt, self.latest_refusal, turns=HELM_MOVES)
            if hit_any and self.helm.get("news"):
                # budget raised or need dropped: earlier rejects deserve another look
                self.final_hunt = False
                self.reviewed = False
        if hit_any:
            self.log("decline_review", found=True)
        return hit_any

    # ------------------------------------------------------------------ main flow
    def go(self):
        try:
            if self.read_request():
                self.shop()
            else:
                self.close_anyway()
                return self.events
        except Exception as err:   # an episode must always end
            self.log("exception", error=repr(err)[:300])
        try:
            if not self.done:
                self.close_anyway()
        except Exception as err:
            self.log("exception", error=repr(err)[:300])
        self.log("done", cost=round(self.desk.dollars, 4), llm_calls=self.desk.n_calls, steps=self.moves)
        return self.events

    def ending_steps(self):
        """Steps kept in reserve for the close: cart adds, a look at carted listings, a swap, the checkout and a retry."""
        decided = getattr(self, "reserve_fixed", None)
        if decided is not None:
            return decided
        look_steps = min(len(self.wants), 3) if self.table_mode() else 2
        tot = len(self.wants) + 6
        if self.role_map.get("facts") and not self.role_map.get("compare"):
            tot += look_steps
        later = 0
        for step_d in self.demands:
            if step_d["moment"] != "start":
                later += 1
        if self.approval_due:
            later += 1
        tot += min(4, later)
        if self.lowest_price and not getattr(self, "spares_read", False):
            tot += 1
        return tot

    def shop(self):
        all_items = self.wants
        cnt = len(all_items)
        # no stated cap: plan on 4 messages (caps seen were 4 or 5); the leftover loop keeps asking while allowed
        cap = 4 if self.msg_cap is None else self.msg_cap
        qcount = cap - self.kept_messages()
        if qcount < 0 or "message" not in self.role_map:
            qcount = 0
        extra_q = 1 if qcount else 0

        # references in the request itself, resolved early but without eating the question and ending steps
        self.chain_floor = 2 * cnt + qcount + extra_q + self.ending_steps()
        for itm in all_items:
            for nd in list(itm.needs):
                if not nd.pointer or nd.rel not in POINTER_RELS:
                    continue
                self.follow_pointer(itm, nd)
        self.chain_floor = 0

        # steps for the first search pass: questions (plus one re-ask) and the ending are held back
        free_steps = self.moves_left() - qcount - extra_q - self.ending_steps()
        per_share = 0
        if free_steps > 0:
            per_share = max(2, free_steps // max(cnt, 1))

        # a demanded opening step other than a message runs now; an opening message becomes a regular question
        for step_d in self.demands:
            if step_d["moment"] == "start" and step_d.get("kind") != "message":
                self.process_check("start", "start the task (search the store)")
                self.read_news()
                if self.msg_cap is not None:
                    qcount = min(qcount, self.msg_cap - self.msgs_used - self.kept_messages())
                qcount = max(0, qcount)
                break

        # first search and look per item
        for ki, itm in enumerate(all_items):
            if ki == cnt - 1:
                alloc = max(per_share, self.moves_left() - qcount - 1 - self.ending_steps())
            else:
                alloc = per_share
            if alloc <= 0:
                break
            t_start = self.moves_left()
            self.hunt(itm, max(2, alloc // 2))
            spent = t_start - self.moves_left()
            self.sample(itm, max(0, alloc - spent))

        # questions: one property of one product per message (two-property messages got half answered, races 9-12)
        if qcount > 0 and self.may_ask():
            agenda = self.question_plan(qcount)

            def ask_one(itm, prop_name):
                count0 = len(itm.needs)
                rsp = self.put_question(itm, prop_name, also=None)
                # after a reply asking for one question at a time a re-ask rarely tells a need: move on instead
                self.take_reply_once(itm, prop_name, rsp)
                self.read_news()
                return count0 != len(itm.needs)

            def known_ids(itm, prop_name):
                return any(likeness(rq.field, prop_name) >= 0.67 for rq in itm.needs) or self.said_before(itm, prop_name)

            while agenda:
                if not self.may_ask() or self.moves_left() <= self.ending_steps():
                    break
                itm, prop_name = agenda.pop(0)
                if known_ids(itm, prop_name):
                    continue
                if ask_one(itm, prop_name) and agenda:
                    # a need was heard: rebuild the rest so its property (and aliases) is skipped
                    spare_steps = len(agenda)
                    if self.msg_cap is not None:
                        spare_steps = min(spare_steps, self.msg_cap - self.msgs_used - self.kept_messages())
                    agenda = self.question_plan(spare_steps)
            # leftover messages go to further properties, since all hidden needs must come out
            while self.may_ask() and self.moves_left() > self.ending_steps() + 1 and _kb_ample(self, 30):
                more_plan = self.question_plan(1)
                if not more_plan:
                    break
                ask_one(more_plan[0][0], more_plan[0][1])

        # reference chains still unresolved (from the request or from a reply)
        for itm in all_items:
            for nd in list(itm.needs):
                if not nd.pointer or nd.rel not in POINTER_RELS:
                    continue
                self.chain_floor = self.ending_steps() + 2
                self.follow_pointer(itm, nd)
                self.chain_floor = 0

        if self.style == "ask":
            return

        self.early_due = True
        self.early_cart()

        # judge everything open with what is known now, then search more where no fit exists (fitless items first)
        self.assess_everything()

        def need_order(wt):
            return (bool(wt.good_offers()), -len(wt.seen))
        sequence = sorted(all_items, key=need_order)
        for ki, itm in enumerate(sequence):
            remaining = max(len(sequence) - ki, 1)
            self.hunt(itm, max(2, (_kb_room(self)) // remaining), final=True)

        self.pick_and_close()

    def early_cart(self):
        """Cart the best known pick before the final search: stores announce changes (sold out, new price, dropped need,
        new budget) once something is in the cart, so they come while steps remain. The cart is fixed to the final
        picks at checkout, at no cost when the pick holds."""
        if getattr(self, "early_done", False) or self.basket:
            return
        if not (self.role_map.get("cart_add") or self.role_map.get("cart_set")):
            return
        if _kb_room(self) < 3 or _kb_short(self, 30):
            self.early_done = True
            return
        # no fit yet: leave the cart empty; a later fit gets carted when the hunt calls again
        self.assess_everything()
        for itm in self.wants:
            ordered_ = _kb_ranked(self, itm)
            if not ordered_:
                continue
            choice = ordered_[0]
            if _kb_over_cap(self.limit, choice, itm.count):
                continue
            self.early_done = True
            if self.put_in_cart(itm, choice):
                self.log("provisional_cart", item=itm.label, id=choice.ref)
            return

    def sample(self, item, allowance=None):
        """Read more listings of the item's kind so the property menu is representative (about twelve with a multi-read
        tool, else five), inside the steps it may spend."""
        if "message" not in self.role_map or self.moves_left() <= self.ending_steps() + 2:
            return
        stop_line = 0 if allowance is None else self.moves_left() - allowance
        min_left = max(stop_line, self.ending_steps() + 2)
        goal = 12 if self.opens_many() else 5

        def rank(ofr):
            return (ofr.triage_mark == "no", ofr.triage_mark != "yes", _kb_price(ofr))

        def not_read():
            recs = []
            for ofr in item.seen.values():
                if ofr.read_full or ofr.in_stock is False:
                    continue
                if self.right_kind(item, ofr):
                    recs.append(ofr)
            recs.sort(key=rank)
            return recs

        while self.moves_left() > min_left:
            if self.seconds_left() <= CLOSE_MARGIN + 20 or self.over_budget_spend():
                break
            have = sum(1 for ofr in item.seen.values() if ofr.read_full and self.right_kind(item, ofr))
            misses_n = goal - have
            if misses_n <= 0:
                return
            queue_ids = not_read()
            if len(queue_ids) < 2 and item.seeks < len(item.queries) + 1:
                size0 = len(item.seen)
                self.seek(item, query=item.kind, max_pages=1, budget=min_left)
                queue_ids = not_read()
                if not queue_ids and len(item.seen) == size0:
                    return
            if not queue_ids:
                return
            if self.opens_many():
                span = self.side_max
            elif self.side_thin:
                span = 4
            else:
                span = 1
            fetched = self.open_offers(item, [ofr.ref for ofr in queue_ids[:min(misses_n, span)]], floor=min_left)
            if not fetched:
                return
            self.assess(item, _kb_read_rows(self, item))

    def hunt(self, item, steps, final=False):
        """Search and read listings for one item until enough fits are known or the step allowance runs out."""
        stop_line = self.moves_left() - steps
        qlist = list(item.queries)
        cursor = [min(item.seeks, len(qlist))]
        enough = 3 if self.lowest_price else (4 if self.best_by else 2)

        def lim():
            return max(stop_line, self.ending_steps())
        while self.moves_left() > lim() and _kb_ample(self, 10) and not self.over_budget_spend():
            self.read_news()
            self.triage(item, self.pool_rows(item))
            rows_now = self.pool_rows(item)
            unread_rows = [cand_r for cand_r in rows_now if not cand_r.read_full]
            self.assess(item, [cand_r for cand_r in rows_now if cand_r.read_full])
            fit_list = item.good_offers()
            if final and fit_list and not self.basket and getattr(self, "early_due", False):
                self.early_cart()
            # with the cheapest rule every unread row under the best fit's price still has to be seen
            if fit_list and not self.lowest_price and len(fit_list) >= enough:
                break
            grp_list = self.value_sets(item)
            if getattr(item, "values_key", None) != repr(grp_list):
                # values changed (a need was heard): earlier value-query coverage no longer applies
                item.values_key = repr(grp_list)
                item.value_rows = set()
                item.values_all_seen = False
            cheap_mode = bool(fit_list and self.lowest_price)
            price_top = _kb_price(fit_list[0]) if cheap_mode else None
            if grp_list:
                # rows carrying the required values open first; if none left carries them all, query the values
                unread_rows.sort(key=lambda cand_r: (cand_r.triage_mark == "no", -_kb_value_hits(self, item, grp_list, cand_r),
                                                cand_r.triage_mark != "yes", _kb_price(cand_r)))
                pool_c = unread_rows
                if cheap_mode:
                    pool_c = [cand_r for cand_r in unread_rows if cand_r.cost_each is not None and cand_r.cost_each < price_top]
                if not any(_kb_value_hits(self, item, grp_list, cand_r) == len(grp_list) for cand_r in pool_c)\
                        and self.seek_by_values(item, lim()):
                    continue
            all_seen = bool(grp_list) and self.strict_words and getattr(item, "values_all_seen", False)
            if cheap_mode:
                unread_rows = [cand_r for cand_r in unread_rows if cand_r.cost_each is not None and cand_r.cost_each < price_top]
                if all_seen:
                    # the value query returned every listing with the values: the rest cannot fit
                    unread_rows = [cand_r for cand_r in unread_rows if cand_r.triage_mark == "yes" or cand_r.ref in item.value_rows]
                if not unread_rows:
                    verdict_s = _kb_cheap_dry(self, item, price_top, all_seen, qlist, lim)
                    if verdict_s == "again":
                        continue
                    if verdict_s == "stop":
                        break
            if unread_rows:
                # many products mean many steps and little time: more pages per judging call
                span_n = 2
                if self.opens_many():
                    span_n = self.side_max
                elif self.side_thin or len(self.wants) >= 3:
                    span_n = 4
                if not self.open_offers(item, [cand_r.ref for cand_r in unread_rows[:span_n]], floor=lim()):
                    break
                continue
            # no unread rows remain: search further
            if _kb_search_more(self, item, qlist, cursor, lim) == "stop":
                break

    def seek_needs(self, item, floor):
        """Once per item, search the heard value as keywords: kind + a stated value + it, kind + it, then it alone (an
        any-word store ranks the bare value best; an all-words store was covered by the value queries). True when a
        search ran."""
        heard = []
        preset = []
        for rq in item.live_needs():
            if not rq.want or rq.rel not in ("is", "one of"):
                continue
            head = rq.want.split("|")[0].strip()
            if rq.source == "shopper":
                heard.append(head)
            else:
                preset.append(head)
        if not heard or getattr(item, "need_hunt_done", False) or self.strict_words is True:
            return False
        item.need_hunt_done = True
        done = {frozenset(plain_words(tq)) for tq in _kb_texts_tried(item)}
        qset = []
        if preset:
            qset.append(f"{item.kind} {preset[0]} {heard[0]}")
        qset += [f"{item.kind} {heard[0]}", heard[0]]
        did_run = False
        for qtext in qset:
            if self.moves_left() <= floor:
                break
            if frozenset(plain_words(qtext)) not in done:
                did_run = True
                self.seek(item, query=qtext, max_pages=3, budget=floor)
        return did_run

    def more_queries(self, item):
        """Ask a cheap model for fresh keyword queries when the item's own found too few fits."""
        def read_first(ofr):
            return (not ofr.read_full, ofr.ref)
        displayed = sorted(item.seen.values(), key=read_first)[:14]
        ok_ids = None
        recs = []
        for ofr in displayed:
            fit = None
            if ofr.read_full:
                if ok_ids is None:
                    ok_ids = {fx.ref for fx in item.good_offers(include_unsure=True)}
                fit = ofr.ref in ok_ids
            recs.append({"title": ofr.heading[:110], "fits": fit})
        attempted = sorted(_kb_texts_tried(item))
        qname = (self.role_map.get("search_text") or {}).get("query") or ""
        longest = (self.schema_of("search_text").get("properties") or {}).get(qname, {}).get("maxLength")
        heard_vals = "\n- ".join(item.need_lines()) or "none"
        body_text = f"PRODUCT: {item.label} (category: {item.kind})\nREQUIREMENTS:\n- " + heard_vals
        body_text += f"\nTRIED: {_kb_js(attempted)}\nLISTINGS SEEN: {_kb_js(recs)}"
        if longest:
            body_text += f"\nMAX QUERY LENGTH: {longest} characters"
        rsp = _kb_fast(self, P_QUERIES, body_text, 700)
        newer = []
        if isinstance(rsp, dict):
            for qtext in rsp.get("queries") or []:
                qtext = str(qtext).strip()
                if not qtext or len(qtext) > 60:
                    continue
                if flat_text(qtext) in {flat_text(tt) for tt in attempted + item.queries + newer}:
                    continue
                newer.append(qtext)
        self.log("suggested", item=item.label, queries=newer[:5])
        return newer[:5]

    def order_by_best(self, item, cands):
        """Order fits by the request's own measure of best with one model call; untouched when there is none."""
        if len(cands) < 2 or not self.best_by:
            return cands
        idlist = [ofr.ref for ofr in cands]
        memo = getattr(self, "best_rank", {}).get(item.slot)
        if memo is None or memo[0] != idlist:
            recs = []
            for ofr in cands[:12]:
                recs.append({"id": ofr.ref, "title": ofr.heading[:120], "price": ofr.cost_each, "properties": ofr.props,
                             "description": str(ofr.blurb or "")[:300]})
            body_text = f"MEASURE: {self.best_by}\nPRODUCT: {item.label}\nLISTINGS: " + brief(recs, 9000)
            rsp = self.desk.ask_json([PLANNER_MODEL, QUICK_MODEL, SPARE_MODEL], P_RANK, body_text, 600)
            ordered_ = []
            if isinstance(rsp, dict):
                ordered_ = [str(xv) for xv in (rsp.get("order") or [])]
            self.log("best_rank", measure=self.best_by, order=ordered_[:12])
            memo = (idlist, ordered_)
            if ordered_:
                ftable = dict(getattr(self, "best_rank", {}))
                ftable[item.slot] = memo
                self.best_rank = ftable
        place = {}
        for ix, pid in enumerate(memo[1]):
            place[pid] = ix

        def where(ofr):
            return place.get(ofr.ref, len(place) + idlist.index(ofr.ref))
        return sorted(cands, key=where)

    def read_spares(self):
        """Cheapest-rule tasks, before the first cart add: read the cheapest unread rows priced up to the second known
        fit, one batch per item, so a pick that sells out or reprices at carting has a checked stand-in."""
        self.spares_read = True        # its reserved step is released from now on
        for wt in self.wants:
            if not self.can_buy(wt):
                return                      # an item still without a fit needs the steps more

        def unit_price(ofr):
            return ofr.cost_each
        for itm in self.wants:
            if _kb_room(self) < 1 or _kb_short(self, 20):
                break
            fitting = self.can_buy(itm)
            if len(fitting) > 1:
                lim = fitting[1].cost_each
            elif fitting:
                lim = fitting[0].cost_each
            else:
                lim = None
            # rows a title screen rejected are read as well: the table decides and a title may contradict it
            recs = []
            for ofr in self.pool_rows(itm):
                if ofr.read_full or ofr.in_stock is False or ofr.cost_each is None:
                    continue
                if lim is not None and not (ofr.cost_each <= lim):
                    continue
                if _kb_over_cap(self.limit, ofr, itm.count):
                    continue
                recs.append(ofr)
            recs.sort(key=unit_price)
            if not recs:
                continue
            fetched = self.open_offers(itm, [ofr.ref for ofr in recs[:self.side_max]], floor=self.ending_steps())
            self.log("backups", item=itm.label, read=[ofr.ref for ofr in fetched], bound=lim)
            if fetched:
                self.assess(itm, fetched)

    def pick_and_close(self):
        self.read_news()
        if self.lowest_price and not getattr(self, "spares_read", False):
            self.read_spares()
        self.assess_everything()
        picked = {}
        fit_pairs = self.pairings()
        host_slots = set()
        acc_of = {}
        for acc_s, host_s in fit_pairs:
            host_slots.add(host_s.slot)
            acc_of[acc_s.slot] = host_s
        cond_list = getattr(self, "rules", [])
        source_slots = set()
        waiting = set()
        for cnd in cond_list:
            source_slots.add(cnd["src"].slot)
            for _, dd, _ in cnd["branches"]:
                waiting.add(dd.slot)
        # items whose choice waits on nobody get their first pick confirmed in parallel
        jobs = []
        for wt in self.wants:
            if wt.slot in acc_of or wt.slot in waiting:
                continue
            ordered_ = _kb_ranked(self, wt)
            if ordered_:
                jobs.append(lambda wt=wt, cnd=ordered_[0]: self.confirm_offer(wt, cnd))
        run_together(jobs)

        def turn_key(wt):
            # hosts before their accessories, condition sources before the items depending on them
            waits = wt.slot in acc_of or wt.slot in waiting
            leads = wt.slot in host_slots or wt.slot in source_slots
            return (waits, not leads, wt.slot)
        for itm in sorted(self.wants, key=turn_key):
            host_item = acc_of.get(itm.slot)
            if host_item is not None and host_item.slot in picked:
                self.tie_to_host(itm, host_item, picked[host_item.slot])
            if itm.slot in waiting:
                self.use_conditions(itm, picked)
            _kb_choose_one(self, itm, picked)
            if host_item is not None and itm.slot not in picked:
                # no accessory suits this host: try another host that one suits
                _kb_rehost(self, itm, host_item, picked)
            if itm.slot in waiting and itm.slot not in picked:
                # nothing meets what the condition demands for that pick: try a source pick with the opposite outcome
                _kb_flip_source(self, itm, picked, cond_list, fit_pairs)
        unfit = [wt for wt in self.wants if wt.slot not in picked]
        busy = self.over_budget_spend
        if unfit and not getattr(self, "final_hunt", False) and _kb_ample(self, 20) and not busy():
            # nothing to order for these yet: spend the reserved ending steps on one more search
            self.final_hunt = True
            if _kb_last_hunt(self, unfit):
                self.pick_and_close()
                return
        # (re-verifying a listing the audit names costs no step; its reads and searches keep the ending's steps)
        if unfit and not getattr(self, "reviewed", False) and _kb_ample(self, 25) and not busy():
            self.reviewed = True
            if self.review_search(unfit) and all(self.can_buy(wt) for wt in unfit):
                self.pick_and_close()
                return
        self.close_picks = dict(picked)
        if unfit:
            labels = [wt.label for wt in unfit]
            self.exit_reason = "no fitting listing for " + ", ".join(labels)
            self.log("no_fit", items=labels, pools={wt.label: len(wt.seen) for wt in unfit})
            self.walk_away(self.exit_reason)
            return
        if self.limit is not None:
            spend = _kb_squeeze_budget(self, picked)
            if spend > self.limit + 1e-6:
                self.exit_reason = "the cheapest fitting combination is over the budget"
                self.log("over_budget", total=spend, cap=self.limit)
                self.walk_away(self.exit_reason)
                return
        for itm in self.wants:
            itm.pick = picked[itm.slot]
        if not self.look_due():
            self.order_now()
            return
        # unplaced store lines or never-used store tools: the pilot looks once before the order (new world rules
        # surface there first)
        self.looked = True
        note = "Every product has a pick; before the order, check the store lines and tools the routine could not place."
        if self.steer(note, {"picks": {wt.label: picked[wt.slot].ref for wt in self.wants}}, turns=3):
            self.pick_and_close()
            return
        self.order_now()

    def review_search(self, missing):
        """Before declining: an independent look over all that was seen for each still-unfit item (possible fits, unread
        rows, untried searches), acted on by code. True when something new came of it."""
        gained_lines = False
        self.reserve_fixed = _kb_close_reserve(self)

        def sequence(ofr):
            return (not ofr.read_full, _kb_price(ofr))
        try:
            for itm in missing:
                if _kb_short(self, 20):
                    break
                bucket = []
                for ofr in itm.seen.values():
                    if not self.right_kind(itm, ofr) or ofr.in_stock is False:
                        continue
                    if _kb_over_cap(self.limit, ofr, itm.count):
                        continue
                    bucket.append(ofr)
                cardlist = []
                for ofr in sorted(bucket, key=sequence)[:40]:
                    cd = {"id": ofr.ref, "title": ofr.heading[:110], "price": ofr.cost_each, "read": ofr.read_full}
                    if not ofr.read_full:
                        cd["text"] = (ofr.blurb or "")[:160]
                    else:
                        val = _kb_verdict(itm, ofr)
                        cd["failed"] = sorted(val.get("failed") or [])[:3]
                        cd["unsure"] = sorted(val.get("unsure") or [])[:3]
                        cd["properties"] = dict(list(ofr.props.items())[:16])
                        if ofr.remarks:
                            cd["notes"] = ofr.remarks[:200]
                    cardlist.append(cd)
                if not cardlist:
                    continue
                attempted = sorted(_kb_texts_tried(itm))
                money = "none stated" if self.limit is None else self.limit
                body_text = (f"PRODUCT: {itm.label} (category: {itm.kind}; quantity {itm.count})\nBUDGET, SUM OF EVERY LINE: "
                        f"{money}\nREQUIREMENTS:\n- ")
                body_text += "\n- ".join(itm.need_lines()) or "none"
                body_text += f"\nQUERIES TRIED: {_kb_js(attempted)}\n\nLISTINGS SEEN:\n" + brief(cardlist, 14000)
                rsp = self.desk.ask_json([PLANNER_MODEL, QUICK_MODEL, SPARE_MODEL], P_REVIEW, body_text, 1500)
                if not isinstance(rsp, dict):
                    continue
                self.log("audit", item=itm.label, answer=brief(rsp, 300))
                mentioned_l = [str(xv) for xv in rsp.get("fits_after_all") or []]
                for pid in mentioned_l[:2]:
                    ofr = itm.seen.get(pid)
                    if ofr is None or not ofr.read_full:
                        continue
                    if self.field_misses(itm, ofr):
                        continue
                    if self.check_offer(itm, ofr, strict=True):
                        # a second independent reading agrees: set the old verdict aside
                        itm.rulings[ofr.ref] = _kb_clean_ruling()
                        itm.assessed_on[ofr.ref] = set(itm.need_lines())
                        gained_lines = True
                not_read = []
                for xv in rsp.get("unopened") or []:
                    if str(xv) in itm.seen and not itm.seen[str(xv)].read_full:
                        not_read.append(str(xv))
                not_read = not_read[:4]
                if not_read and _kb_room(self) >= 1:
                    if self.open_offers(itm, not_read, floor=self.ending_steps()):
                        gained_lines = True
                asked = []
                for xv in rsp.get("queries") or []:
                    if str(xv).strip():
                        asked.append(str(xv).strip())
                prev = {flat_text(tt) for tt in attempted}
                for qtext in asked[:2]:
                    if _kb_room(self) < 3 or _kb_short(self, 15):
                        break
                    if flat_text(qtext) in prev:
                        continue
                    if self.seek(itm, query=qtext[:60], max_pages=1, budget=self.ending_steps()) <= 0:
                        continue
                    self.triage(itm, [ofr for ofr in self.pool_rows(itm) if not ofr.read_full])
                    recs = [ofr for ofr in self.pool_rows(itm) if not ofr.read_full and ofr.triage_mark != "no"][:2]
                    if recs and self.open_offers(itm, [ofr.ref for ofr in recs], floor=self.ending_steps()):
                        gained_lines = True
                self.assess(itm, _kb_read_rows(self, itm))
        finally:
            self.reserve_fixed = None
        return gained_lines

    def wanted_named(self, name):
        body_text = str(name or "")
        if not body_text or not self.wants:
            return None

        def score(wt):
            return max(likeness(wt.label, body_text), likeness(wt.kind, body_text))
        top_price = max(self.wants, key=score)
        if score(top_price) >= 0.2:
            return top_price
        return None

    def parse_conditions(self, raw):
        """Requirements on one item that depend on how another item turns out, as {"src", "prop", "values",
        "branches": [(holds?, item, requirement spec)]}."""
        rules = []
        for cnd in (raw if isinstance(raw, list) else []):
            test = cnd.get("if") if isinstance(cnd, dict) else None
            if not isinstance(test, dict) or not test.get("property"):
                continue
            source_item = self.wanted_named(test.get("item"))
            raw_values = test.get("values")
            if isinstance(raw_values, list):
                vals = [str(val) for val in raw_values if val]
            elif raw_values:
                vals = [str(raw_values)]
            else:
                vals = []
            arm_list = []
            for kname, branch_val in (("then", True), ("otherwise", False)):
                cfg = cnd.get(kname)
                if not isinstance(cfg, dict) or not cfg.get("property") or not cfg.get("value"):
                    continue
                dest = self.wanted_named(cfg.get("item"))
                if dest is None or dest is source_item:
                    continue
                arm_list.append((branch_val, dest, cfg))
            if source_item is None or not vals or not arm_list:
                continue
            rules.append({"src": source_item, "prop": str(test["property"]), "values": vals, "branches": arm_list})
            for _, dest, cfg in arm_list:
                # search for the values a branch may demand, so listings carrying them are at hand
                for val in str(cfg["value"]).split("|"):
                    val = val.strip()
                    if not val or len(val) > 40:
                        continue
                    if any(flat_text(val) in flat_text(tt) for tt in dest.queries):
                        continue
                    dest.queries.append(f"{val} {dest.head_noun}"[:60])
        if rules:
            displayed = []
            for cnd in rules:
                arms = [(ww, dd.label, bb.get("property"), bb.get("value")) for ww, dd, bb in cnd["branches"]]
                displayed.append((cnd["src"].label, cnd["prop"], cnd["values"], arms))
            self.log("conditions", rules=displayed)
        return rules

    def rule_holds(self, cond, listing):
        """Is the listing's value for the condition's property one of the condition's values? Decided by the property
        table when it is plain either way, otherwise by one fast model call."""
        store = self.__dict__.setdefault("held_cache", {})
        token = (id(cond), listing.ref, json.dumps(listing.props, sort_keys=True, default=str))
        if token in store:
            return store[token]
        texts = [str(pv) for pk, pv in listing.props.items() if _kb_near(str(pk).replace("_", " "), cond["prop"])]
        verdict_b = None
        if texts:
            have = plain_words(" ".join(texts))
            bags = [bg for bg in (plain_words(xx) for xx in cond["values"]) if bg]
            if any(bg <= have for bg in bags):
                verdict_b = True
            elif all(not (bg & have) for bg in bags):
                verdict_b = False
        if verdict_b is None:
            ask = (f"PROPERTY: {cond['prop']}\nVALUES: {_kb_js(cond['values'])}\n\nLISTING:\n"
                   + _kb_js(listing.card(2500)))
            ans = _kb_fast(self, P_RULE, ask, 300)
            if isinstance(ans, dict) and isinstance(ans.get("holds"), bool):
                verdict_b = ans["holds"]
        store[token] = verdict_b
        self.log("condition", src=cond["src"].label, listing=listing.ref, prop=cond["prop"], holds=verdict_b)
        return verdict_b

    def use_conditions(self, item, chosen):
        """Apply the conditions whose source item has a pick to this item's needs, then judge its listings again."""
        relevant = [cn for cn in self.rules if any(dest is item for _, dest, _ in cn["branches"])]
        if not relevant:
            return
        derived = []
        for cn in relevant:
            src_pick = chosen.get(cn["src"].slot)
            outcome = None if src_pick is None else self.rule_holds(cn, src_pick)
            if outcome is None:
                continue
            derived += [Need(sp["property"], sp.get("op") or "is", sp["value"], origin="condition")
                        for flag, dest, sp in cn["branches"] if dest is item and flag == outcome]
        current = sorted(nd_.line() for nd_ in item.needs if nd_.source == "condition")
        if current == sorted(nd_.line() for nd_ in derived):
            return
        item.needs = [nd_ for nd_ in item.needs if nd_.source != "condition"] + derived
        self.log("condition_applied", item=item.label, reqs=[nd_.line() for nd_ in derived])
        self.assess(item, _kb_read_rows(self, item))
        if self.can_buy(item) or _kb_room(self) < 3 or not _kb_ample(self, 20):
            return
        item.dry = False
        self.hunt(item, _kb_room(self), final=True)

    def pairings(self):
        """The request's "X must fit Y" rules as (accessory item, host item) pairs."""
        hit_any = []
        multi = len(self.wants) >= 2
        for link in self.fit_links:
            acc_s = str(link.get("accessory") or "")
            host_s = str(link.get("host") or "")
            if not acc_s or not host_s or not multi:
                continue

            def best_for(body_text):
                return max(self.wants, key=lambda wt: likeness(wt.label, body_text))
            al = best_for(acc_s)
            hh = best_for(host_s)
            if al is hh or likeness(al.label, acc_s) < 0.2 or likeness(hh.label, host_s) < 0.2:
                continue
            if (al, hh) not in hit_any:
                hit_any.append((al, hh))
        return hit_any

    def tie_to_host(self, item, host_item, host):
        """Bind the accessory to the chosen host listing: a need quoting the host's page, a lookup of part codes its
        candidates point to, and a fresh judgement of the candidates."""
        pieces = []
        for part in (host.heading[:120], (host.blurb or "")[:300], (host.remarks or "")[:150]):
            if part:
                pieces.append(part)
        kept = list(filter(lambda nd_: nd_.source != "fit", item.needs))
        kept.append(Need("compatibility", "is",
                         f"works with the {host_item.label} chosen, whose page reads: {' '.join(pieces)}", origin="fit"))
        item.needs = kept
        sought = _kb_part_codes(self, item)
        series_codes = _kb_series_named(self, host_item, host)
        _kb_lookup_series(self, item, sought, series_codes)
        self.log("fit_bound", item=item.label, host=host.ref, series=series_codes[:2], looked_up=sought[:3])
        self.assess(item, _kb_read_rows(self, item))
        if series_codes:
            _kb_clear_fit_doubt(self, item, series_codes)

    @staticmethod
    def of_series(listing, series):
        for hh in SERIES_HEAD.findall(listing.heading or ""):
            if hh in series:
                return True
        return False

    def order_now(self, attempt=0):
        # every re-choice opens a new round; rounds are capped as well
        self.buy_rounds = getattr(self, "buy_rounds", 0) + 1
        if self.done or attempt > 2 or self.buy_rounds > 8:
            return
        all_wants = self.wants
        # make the cart hold exactly the picks
        pick_refs = {wq.pick.ref for wq in all_wants if wq.pick}
        for pid2 in [pid2 for pid2 in self.basket if pid2 not in pick_refs]:
            self.take_out(pid2)
        for wq in all_wants:
            sel = wq.pick
            if not sel or sel.ref in self.basket or self.put_in_cart(wq, sel):
                continue
            if getattr(self, "cart_broken", False):
                # the tool rejects the call itself: another listing would fail the same way
                self.log("cart_faulty")
                self.walk_away("the cart tool refuses its arguments")
                return
            book = wq.rulings.setdefault(sel.ref, {"failed": set(), "unsure": set()})
            book["failed"].add("cart refused")
            wq.pick = None
        if not all(wq.pick is not None for wq in all_wants):
            self.pick_and_close()
            return
        # look at the carted listing after the add: changes show up in the next result
        if self.table_mode():
            # lines judged from tables only get their own page read (stock, price, seller notes)
            for wq in all_wants:
                sel = wq.pick
                if sel is None or sel.own_page:
                    continue
                if self.moves_left() - self.owed_messages() > 2:
                    self.open_offers(wq, [sel.ref], page=True)
        elif self.moves_left() > 2 + self.owed_messages() and all_wants and all_wants[0].pick:
            self.open_offers(all_wants[0], [all_wants[0].pick.ref])
        news_flag = self.read_news()
        for wq in all_wants:
            # a pick read after the add is judged again; a contradicted one is swapped before checkout
            sel = wq.pick
            if sel is None or sel.ref in wq.assessed_on:
                continue
            self.assess(wq, [sel])
            fails = _kb_verdict(wq, wq.pick).get("failed")
            if fails:
                self.log("chosen_failed", id=wq.pick.ref, failed=sorted(fails)[:3])
                news_flag = True

        def retry():
            self.pick_again()
            if not self.done:
                self.order_now(attempt + 1)
        if news_flag or self.pick_moved():
            self.log("rechoose", cart=self.basket)
            for wq in all_wants:
                self.assess(wq, _kb_read_rows(self, wq))
            retry()
            return
        if self.check_out():
            return
        # refused: read why, then try again
        news_flag = self.read_news()
        if not news_flag:
            self.recheck_cart()
            news_flag = self.read_news() or self.pick_moved()
        if not (news_flag or self.done) and self.moves_left() >= len(all_wants) + 3:
            # nothing known explains the refusal: let the pilot read it with the record
            self.steer("The store turned down the checkout.", self.latest_refusal, turns=4, refusal=True)
        retry()

    def pick_moved(self):
        spend = None
        for itm in self.wants:
            choice = itm.pick
            if choice is None:
                continue
            if choice.in_stock is False:
                return True
            if self.limit is None or choice.cost_each is None:
                continue
            if spend is None:
                spend = 0
                for wt in self.wants:
                    if wt.pick:
                        spend += (wt.pick.cost_each or 0) * wt.count
            if spend > self.limit + 1e-6:
                return True
        return False

    def pick_again(self):
        """After a change, search again where needed and re-pick each item from its fits (cheapest rule kept)."""
        for itm in self.wants:
            self.assess(itm, _kb_read_rows(self, itm))
            if self.lowest_price or not any(fx.in_stock is not False for fx in itm.good_offers()):
                itm.need_hunt_done = False
                itm.swept_under = None
                alloc = max(2, _kb_room(self))
                if alloc >= 3 and _kb_ample(self, 20):
                    self.hunt(itm, alloc, final=True)
        for itm in self.wants:
            fitting = [fx for fx in itm.good_offers(cheapest=self.lowest_price) if fx.in_stock is not False]
            if not fitting:
                itm.pick = None
            elif self.lowest_price or itm.pick is None or all(fx.ref != itm.pick.ref for fx in fitting):
                itm.pick = fitting[0]
        choices = {}
        for wt in self.wants:
            if wt.pick is not None:
                choices[wt.slot] = wt.pick
        targets = {}
        for cnd in getattr(self, "rules", []):
            for _, dd, _ in cnd["branches"]:
                targets[dd.slot] = dd
        for dest in targets.values():
            # a new pick of a condition source may change what the condition demands
            self.use_conditions(dest, choices)
            ok_ids = [fx.ref for fx in self.can_buy(dest)]
            if dest.pick is None or dest.pick.ref in ok_ids:
                continue
            dest.pick = dest.seen[ok_ids[0]] if ok_ids else None
        if any(wt.pick is None for wt in self.wants):
            self.pick_and_close()
            return
        if self.limit is None:
            return
        spend = 0
        for wt in self.wants:
            spend += (wt.pick.cost_each or 0) * wt.count
        if spend > self.limit + 1e-6:
            self.pick_and_close()

    def close_anyway(self):
        """Out of time, steps or ideas: finish with what is known (a buyable fitting cart inside the budget)."""
        if self.done:
            return
        for itm in self.wants:
            if itm.pick is not None and itm.pick.in_stock is not False:
                continue
            fitting = self.can_buy(itm)
            itm.pick = fitting[0] if fitting else None
        can_close = bool(self.wants) and all(wt.pick for wt in self.wants)
        if can_close and self.limit is not None:
            spend = 0
            for wt in self.wants:
                spend += (wt.pick.cost_each or 0) * wt.count
            can_close = spend <= self.limit + 1e-6
        if can_close:
            picked = {wt.pick.ref for wt in self.wants}
            obsolete = [pid for pid in self.basket if pid not in picked]
            to_add = [wt for wt in self.wants if wt.pick.ref not in self.basket]
            if self.moves_left() >= len(obsolete) + len(to_add) + 1:
                for pid in obsolete:
                    self.take_out(pid)
                for itm in to_add:
                    self.put_in_cart(itm, itm.pick)
                if self.check_out():
                    return
        if not self.done:
            self.walk_away("forced")

    def can_buy(self, item):
        """Fitting listings in choosing order, minus those known sold out or failed by a store field."""
        ftable = self.store_fields(item)
        resp = []
        for fx in item.good_offers(cheapest=self.lowest_price):
            if fx.in_stock is False or self.field_misses(item, fx, ftable):
                continue
            resp.append(fx)
        return resp


def _kb_room(errand):
    """Steps left beyond those held for the ending."""
    return errand.moves_left() - errand.ending_steps()


def _kb_short(errand, more_q):
    """Less time left than the closing margin plus extra seconds."""
    return errand.seconds_left() < CLOSE_MARGIN + more_q


def _kb_ample(errand, more_q):
    """More time left than the closing margin plus extra seconds."""
    return errand.seconds_left() > CLOSE_MARGIN + more_q


def _kb_read_rows(errand, itm):
    """The item's pool rows that were read in full."""
    return [cand_r for cand_r in errand.pool_rows(itm) if cand_r.read_full]


def _kb_ranked(errand, itm):
    """The item's buyable fits in the request's order of best."""
    return errand.order_by_best(itm, errand.can_buy(itm))


def _kb_price(offer):
    """Unit price for sorting, unknown prices last."""
    return 1e18 if offer.cost_each is None else offer.cost_each


def _kb_verdict(itm, offer):
    """The stored verdict on a listing, empty when there is none."""
    return itm.rulings.get(offer.ref) or {}


def _kb_js(val_s):
    """JSON text for a prompt, non-ASCII kept as is."""
    return json.dumps(val_s, ensure_ascii=False)


def _kb_squash(txt):
    """Runs of whitespace collapsed to one space, ends trimmed."""
    return re.sub(r"\s+", " ", txt).strip()


def _kb_fast(errand, system, user_t, max_tokens):
    """One JSON answer from the fast model, the spare model as fallback."""
    return errand.desk.ask_json([QUICK_MODEL, SPARE_MODEL], system, user_t, max_tokens)


def _kb_near(aa, bb):
    """Two property names that read as the same one."""
    return close_names(aa, bb) or likeness(aa, bb) >= 0.5


def _kb_texts_tried(itm):
    """The keyword query texts already sent for the item."""
    return {kk.split(":", 1)[1] for kk in itm.pages if kk.startswith("search_text:")}


def _kb_fix_budget(cap, line_list, old_cap):
    """A budget read from shopper lines, checked against the figures the lines hold (the reader may slip by a
    factor of 100; otherwise the first figure near the old budget)."""
    nums = [float(xx.replace(",", "")) for xx in re.findall(r"\d[\d,]*(?:\.\d+)?", " ".join(line_list))]
    if cap in nums:
        return cap
    if cap * 100 in nums:
        return cap * 100
    if cap / 100 in nums:
        return cap / 100
    if nums and old_cap is not None:
        near_l = [xx for xx in nums if old_cap / 20 <= xx <= old_cap * 20]
        if near_l:
            return near_l[0]
    return cap


def _kb_set_mode(errand):
    """The store only offers a tool that replaces the whole cart."""
    return bool(not errand.role_map.get("cart_add") and errand.role_map.get("cart_set"))


def _kb_arg_spec(errand, role, arg):
    """The schema entry of one argument of a role's tool (empty when unknown)."""
    return (errand.schema_of(role).get("properties") or {}).get(arg) or {}


def _kb_mode_of(cart_tool):
    """A role's mode switch (argument and values), or None."""
    how = cart_tool.get("mode")
    return how if isinstance(how, dict) else None


def _kb_refusal_note(name_s, reply_d):
    """What the pilot is shown about a refused closing call."""
    return {"tool": name_s, "result": brief(reply_d["obs"], 600), "lines": reply_d["lines"][:3]}


def _kb_helm_text(errand, point, detail, history):
    """The pilot's prompt: the request, why it is called, the record and its moves so far."""
    txt = "REQUEST:\n" + errand.ask_text + "\n\nWHY YOU ARE CALLED: " + point
    if detail:
        txt += "\nDETAIL: " + brief(detail, 800)
    txt += "\n\nRECORD:\n" + brief(errand.brief_state(), 9000)
    if history:
        txt += "\n\nYOUR MOVES SO FAR:\n" + "\n".join(history)
    return txt


def _kb_ref_state(errand, vendor, cat, attr_map):
    """The search state kept for one described page: a scratch item and the ids already tried."""
    pair_list = sorted((flat_text(kk), flat_text(vv)) for kk, vv in attr_map.items())
    kkey = json.dumps([flat_text(vendor), flat_text(cat), pair_list])
    book = errand.__dict__.setdefault("ref_search", {})
    if kkey not in book:
        book[kkey] = (Wanted({"name": cat or "listing", "category": cat, "search_terms": []}, 99), set())
    return book[kkey]


def _kb_ref_words(attr_map, vendor):
    """Search words from a page description: short values whole, a few words of each long one; figures last (the
    request may give them in other units)."""
    plain_v = []
    long_v = []
    for av in attr_map.values():
        txt = str(av)
        if len(txt.split()) > 3:
            long_v.append(txt)
        elif txt.strip():
            plain_v.append(txt.strip())
    for txt in long_v:
        picked_w = []
        for wd in re.findall(r"[A-Za-z][A-Za-z-]{3,}", txt):
            if wd not in plain_v and flat_text(wd) != flat_text(vendor):
                picked_w.append(wd)
        plain_v += picked_w[:4]
    plain_v.sort(key=lambda wd: re.search(r"\d", wd) is not None)
    return plain_v


def _kb_ref_queries(vendor, cat, attr_text):
    """Keyword queries for a described page, most specific first ("kettles and tea makers" names its kind twice)."""
    noun_w = cat.split()[-1] if cat else ""
    first_part = re.split(r"\s+and\s+", cat)[0].split() if cat else []
    noun_list = []
    for nn in (noun_w, first_part[-1] if first_part else ""):
        if nn and nn not in noun_list:
            noun_list.append(nn)
    hits_n = []
    if vendor:
        hits_n.append(f"{vendor} {attr_text}".strip())
        # the described figures without their units: a store whose titles or fields write "170ml" or "170 ML" still
        # lists the page for the seller's name and the bare number (race 9: the paid runs looked the page up so)
        figures = re.findall(r"\d+(?:[.,]\d+)?", attr_text)
        if figures:
            hits_n.append(" ".join([vendor] + figures[:2]))
        hits_n += [vendor + " " + nn for nn in noun_list] + [vendor]
    if cat and attr_text:
        hits_n.append(f"{attr_text} {noun_w}".strip())
    return list(dict.fromkeys(qq for qq in hits_n if qq))


def _kb_vendor_ok(vendor, offer):
    """No seller named, or the listing belongs to the named seller."""
    if vendor:
        named_b = likeness(offer.vendor, vendor) >= 0.5 or likeness(offer.maker, vendor) >= 0.5
        return named_b or flat_text(vendor) in flat_text(" ".join([offer.heading, offer.blurb, offer.remarks]))
    return True


def _kb_overlap(attr_text, wlist, offer):
    """How many described words the listing's text carries; a figure counts written with or without its unit gap."""
    sought = word_bag(attr_text)
    if not sought:
        return 0
    blob = " ".join([offer.heading, offer.blurb, offer.remarks, json.dumps(offer.props)])
    have = word_bag(blob)
    tight = flat_text(blob).replace(" ", "")
    joined_n = 0
    for wq in wlist:
        if not re.search(r"\d", wq) or (word_bag(wq) and word_bag(wq) <= have):
            continue
        packed = flat_text(wq).replace(" ", "")
        if len(packed) >= 3 and packed in tight:
            joined_n += 1
    return len(sought & have) + joined_n


def _kb_attrs_in_text(attr_map, offer):
    """Every described value's words appear in the listing's own text (title, notes, description, table)."""
    if not attr_map:
        return False
    table_s = " ".join(f"{kk} {vv}" for kk, vv in offer.props.items())
    have = plain_words(" ".join([offer.heading or "", offer.remarks or "", offer.blurb or "", table_s]))
    for vv in attr_map.values():
        if str(vv).strip() and not (plain_words(vv) and plain_words(vv) <= have):
            return False
    return True


def _kb_attrs_match(attr_map, offer):
    """The listing's property table states every described value."""
    if not attr_map or not offer.read_full:
        return False
    fact_list = [(flat_text(_kb_spaced(kk)), flat_text(vv)) for kk, vv in offer.props.items()]
    for kk, vv in attr_map.items():
        goal = flat_text(vv)
        if not goal or not any(fval2 == goal and _kb_near(fkey2, kk) for fkey2, fval2 in fact_list):
            return False
    return True


def _kb_same_prop(about_p, wanted_p):
    """A redirect about the wanted property (by name, or one name inside the other)."""
    return _kb_near(about_p, wanted_p) or flat_text(wanted_p) in flat_text(about_p) or flat_text(about_p) in flat_text(wanted_p)


def _kb_pin_need(nd_q, refd, want_s, rel):
    """Settle a reference need on a value read from its page."""
    nd_q.chain_key = pointer_key(refd)
    nd_q.want = want_s
    nd_q.rel = rel
    nd_q.pointer = None


def _kb_apply_page_value(errand, itm, nd_q, refd, reading, pairing, relation_txt, page_o):
    """Turn what the page reader returned into the need's value, or into two bounds around it."""
    got_n = isinstance(reading, dict)
    val_s = None
    if got_n:
        val_s = reading.get("paired") if pairing else reading.get("value")
        twin = reading.get("paired")
        if nd_q.rel == "same as reference" and twin and reading.get("wanted"):
            if flat_text(reading["wanted"]) == flat_text(twin) and flat_text(twin) != flat_text(reading.get("value")):
                # the shopper wants what goes with the page's value, and the page names that
                val_s = reading["paired"]
                errand.log("relation_paired", page=page_o.ref, relation=relation_txt[:80], value=val_s)
    if val_s and nd_q.rel == "within reference":
        # a figure plus or minus an amount, in the page's unit
        centre = as_amount(val_s)
        spread_v = as_amount(str(refd.get("tolerance") or "").strip())
        unit_s = re.sub(r"^[\s\d.,+-]+", "", str(val_s)).strip()
        if centre is None or spread_v is None:
            nd_q.rel = "same as reference"
        else:
            upper = f"{centre + spread_v:g} {unit_s}".strip()
            _kb_pin_need(nd_q, refd, f"{centre - spread_v:g} {unit_s}".strip(), "at least")
            itm.needs.append(Need(nd_q.field, "at most", upper, origin=nd_q.source))
            errand.log("reference_range", prop=nd_q.field, low=nd_q.want, high=upper)
            val_s = None
    if val_s:
        _kb_pin_need(nd_q, refd, str(val_s), POINTER_RELS.get(nd_q.rel, "is"))
        if nd_q.rel == "is" and len(nd_q.want) <= 40 and itm.queries:
            # the page's value makes the item's sharpest query
            if not any(flat_text(nd_q.want) in flat_text(tq) for tq in itm.queries):
                itm.queue_query(f"{nd_q.want} {itm.head_noun}")
    elif pairing and got_n and reading.get("value"):
        # no pairing on the page: its own value is the nearest reading of what was pointed at
        errand.log("pair_unstated", page=page_o.ref, value=reading.get("value"))
        _kb_pin_need(nd_q, refd, str(reading["value"]), "is")


def _kb_value_hits(errand, itm, grp_list, offer):
    """Value groups a row carries in its own words (or that a value query showed it for, in an all-words store)."""
    if offer.ref in itm.value_rows:
        return len(grp_list)
    hits_n = errand.row_matches(offer, grp_list)
    if not errand.strict_words or hits_n >= len(grp_list):
        return hits_n
    sent_l = [plain_words(sig_k.split(":", 1)[1]) for sig_k, id_set in itm.query_rows.items() if offer.ref in id_set]
    covered = sum(1 for grp in grp_list if any(plain_words(aa) <= qq for aa in grp for qq in sent_l))
    return max(hits_n, covered)


def _kb_cheap_dry(errand, itm, best_p, all_seen, qlist, lim):
    """Cheapest rule with nothing cheaper left unread: heard values first, then a price-capped field search, then a
    query of every required value when the store hid matches, then paging the main query on. Returns "again" to loop,
    "stop" to end the hunt, None to fall through."""
    if errand.seek_needs(itm, lim()):
        return "again"
    by_fields = errand.role_map.get("search_fields")
    if best_p < 1e17 and not all_seen and by_fields and by_fields.get("price_max") \
            and getattr(itm, "swept_under", None) != best_p and errand.moves_left() > lim() + 2:
        # only listings cheaper than the best fit come back
        itm.swept_under = best_p
        errand.seek(itm, fields={"category": itm.kind, "price_max": max(0.0, best_p - 0.01)}, max_pages=3, budget=lim())
        return "again"
    if not all_seen and getattr(itm, "hidden_more", False) and not getattr(itm, "tightened", False) \
            and errand.moves_left() > lim() + 2:
        itm.tightened = True
        vals = [rr.want.split("|")[0].strip() for rr in itm.live_needs() if rr.rel in ("is", "one of") and rr.want]
        if vals:
            for qq in (" ".join(vals + [itm.kind]), " ".join(vals)):
                if errand.moves_left() <= lim() + 1:
                    break
                errand.seek(itm, query=qq, max_pages=2, budget=lim())
            return "again"
    # rows show prices, so paging on costs a step a page and only cheaper rows get read
    lead = next((qq for qq in qlist if itm.pages.get("search_text:" + qq)
                 and not itm.pages["search_text:" + qq].get("done")), None)
    if lead and errand.moves_left() > lim() + 2:
        before_n = errand.moves
        errand.seek(itm, query=lead, max_pages=1, budget=lim() + 1)
        if errand.moves > before_n:
            return "again"
    if itm.seeks >= min(len(qlist), 2):
        return "stop"
    return None


def _kb_field_query(errand, itm):
    """Field-search arguments for an item: its category, capped by the budget share and the item's own cap."""
    flds = {"category": itm.kind}
    if errand.limit is not None:
        flds["price_max"] = errand.limit / max(itm.count, 1)
    if itm.unit_limit is not None:
        flds["price_max"] = min(flds.get("price_max", itm.unit_limit), itm.unit_limit)
    return flds


def _kb_search_more(errand, itm, qlist, cursor, lim):
    """One more search for an item whose rows are all read: heard needs as keywords, the next query, a field search,
    paging the first queries, then model-suggested queries. "stop" when nothing is left to try."""
    if errand.seek_needs(itm, lim()):
        return "again"
    if cursor[0] < len(qlist):
        qq = qlist[cursor[0]]
        cursor[0] += 1
        got_n = errand.seek(itm, query=qq, max_pages=6 if len(errand.wants) == 1 else 3, budget=lim())
        if got_n == 0 and cursor[0] >= len(qlist):
            itm.dry = True
        return "again"
    if errand.role_map.get("search_fields"):
        flds = _kb_field_query(errand, itm)
        if not (itm.pages.get("search_fields:" + json.dumps(flds, sort_keys=True)) or {}).get("done"):
            errand.seek(itm, fields=flds, max_pages=2, budget=lim())
            return "again"
    for qq in qlist[:2]:
        if (itm.pages.get("search_text:" + qq) or {}).get("done"):
            continue
        if errand.seek(itm, query=qq, max_pages=2, budget=lim()) > 0:
            return "again"
    if not getattr(itm, "extra_terms", False) and errand.moves_left() > lim() + 1:
        itm.extra_terms = True
        more_q = errand.more_queries(itm)
        if more_q:
            itm.queries.extend(more_q)
            qlist.extend(more_q)
            return "again"
    itm.dry = True
    return "stop"


def _kb_choose_one(errand, itm, picked):
    """Confirm up to three of the item's ranked fits; after three tries (or out of time) the best one unconfirmed."""
    tries_n = 0
    for offer in _kb_ranked(errand, itm):
        if tries_n >= 3 or _kb_short(errand, 0):
            break
        tries_n += 1
        if errand.confirm_offer(itm, offer):
            picked[itm.slot] = offer
            return
    if itm.slot in picked:
        return
    fallback = _kb_ranked(errand, itm)
    if fallback and (tries_n >= 3 or _kb_short(errand, 0)):
        picked[itm.slot] = fallback[0]


def _kb_rehost(errand, itm, host_item, picked):
    """Another host listing that some accessory fits, when none fits the chosen one."""
    for alt_l in errand.can_buy(host_item)[:3]:
        if picked.get(host_item.slot) is alt_l or _kb_short(errand, 0):
            continue
        errand.tie_to_host(itm, host_item, alt_l)
        fit_list = errand.can_buy(itm)
        if fit_list:
            picked[host_item.slot] = alt_l
            picked[itm.slot] = fit_list[0]
            return


def _kb_flip_source(errand, itm, picked, cond_list, fit_pairs):
    """Nothing fits what a condition asks given its source's pick: try a source pick with the other outcome, unless a
    chosen accessory must fit that source."""
    linked = [cc for cc in cond_list if cc["src"].slot in picked and any(dd2 is itm for _, dd2, _ in cc["branches"])]
    for cond_d in linked:
        src_w = cond_d["src"]
        if _kb_short(errand, 0) or any(hh2 is src_w and aa.slot in picked for aa, hh2 in fit_pairs):
            continue
        current = picked[src_w.slot]
        outcome = errand.rule_holds(cond_d, current)
        for alt_l in errand.can_buy(src_w)[:4]:
            if alt_l is picked[src_w.slot] or errand.rule_holds(cond_d, alt_l) == outcome:
                continue
            picked[src_w.slot] = alt_l
            errand.use_conditions(itm, picked)
            fit_list = errand.can_buy(itm)
            if fit_list and errand.confirm_offer(src_w, alt_l):
                picked[itm.slot] = fit_list[0]
                break
            picked[src_w.slot] = current
            errand.use_conditions(itm, picked)
        if itm.slot in picked:
            break


def _kb_last_hunt(errand, unfit):
    """One more search for the items without a pick, keeping only the steps the close needs; True when one fits now.
    (The decline keeps steps too: a new budget or a dropped need may come with the closing call, race 12.)"""
    errand.reserve_fixed = _kb_close_reserve(errand)
    try:
        for wq in unfit:
            if _kb_room(errand) >= 3:
                wq.dry = False
                errand.hunt(wq, _kb_room(errand), final=True)
    finally:
        errand.reserve_fixed = None
    return any(errand.can_buy(wq) for wq in unfit)


def _kb_squeeze_budget(errand, picked):
    """Step the priciest line to its next cheaper fit until the total fits the budget (ten rounds at most); returns
    the total."""
    def spend():
        return sum((picked[wq.slot].cost_each or 0) * wq.count for wq in errand.wants)

    def line_cost(wq):
        return -((picked[wq.slot].cost_each or 0) * wq.count)
    for _ in range(10):
        if spend() <= errand.limit + 1e-6:
            break
        stepped = False
        for wq in sorted(errand.wants, key=line_cost):
            ladder = [ff for ff in wq.good_offers(cheapest=True) if ff.in_stock is not False]
            refs = [ff.ref for ff in ladder]
            at = refs.index(picked[wq.slot].ref) if picked[wq.slot].ref in refs else -1
            if 0 <= at < len(ladder) - 1:
                picked[wq.slot] = ladder[at + 1]
                stepped = True
                break
        if not stepped:
            break
    return spend()


def _kb_no(note):
    """A pilot move's outcome that changed nothing buyable."""
    return note, False


def _kb_ok(note):
    """A pilot move's outcome that changed what can be bought."""
    return note, True


def _kb_cart_change(errand, line_ref, owner_w):
    """Add one cart line for owner's item (owner given) or remove it (owner None), through the replacing tool when
    that is the store's only cart tool. True when the store took the change."""
    adding = owner_w is not None
    if _kb_set_mode(errand):
        new_lines = dict(errand.basket)
        if adding:
            new_lines[line_ref] = owner_w.count
        else:
            new_lines.pop(line_ref, None)
        if not errand.replace_cart(new_lines):
            return False
        errand.basket = new_lines
        if adding:
            errand.basket_owner[line_ref] = owner_w
        else:
            errand.basket_owner.pop(line_ref, None)
        return True
    cart_tool = errand.role_map.get("cart_add")
    if not cart_tool:
        return False
    call_args = {cart_tool["id"]: line_ref}
    how = _kb_mode_of(cart_tool)
    qty_key = cart_tool.get("qty")
    qty_n = owner_w.count if adding else None
    if adding:
        if qty_key:
            cap_q = _kb_arg_spec(errand, "cart_add", cart_tool["qty"]).get("maximum")
            if isinstance(cap_q, int) and qty_n > cap_q:
                qty_n = cap_q
            call_args[cart_tool["qty"]] = qty_n
        if how and how.get("arg") and how.get("add") is not None:
            call_args[how["arg"]] = how["add"]
    elif how and how.get("arg") and how.get("remove") is not None:
        call_args[how["arg"]] = how["remove"]
        if qty_key:
            call_args[cart_tool["qty"]] = errand.basket.get(line_ref, 1)
    elif qty_key:
        min_q = _kb_arg_spec(errand, "cart_add", cart_tool["qty"]).get("minimum")
        call_args[cart_tool["qty"]] = min_q if isinstance(min_q, int) and min_q > 0 else 0
    out_r = errand.use_tool(cart_tool["name"], call_args)
    if not adding:
        if not out_r["ok"]:
            return False
        errand.basket.pop(line_ref, None)
        errand.basket_owner.pop(line_ref, None)
        return True
    if out_r["ok"] and not cart_was_refused(out_r["obs"]):
        errand.basket[line_ref] = qty_n
        errand.basket_owner[line_ref] = owner_w
        return True
    errand.cart_broken = bool(out_r.get("bad_args"))
    errand.log("cart_add_failed", error=out_r["error"], obs=brief(out_r["obs"], 200))
    return False


def _kb_prop_tables(offers):
    """Each listing's property table with keys spelled plainly (camelCase split, separators as spaces, lower case)."""
    tables = []
    for cand_r in offers:
        tbl_row = {}
        for key_x, val_x in cand_r.props.items():
            nm = _kb_squash(_kb_spaced(key_x).replace("_", " ").replace(".", " ")).lower()
            if nm and nm not in tbl_row:
                tbl_row[nm] = str(val_x).strip().lower()[:60]
        tables.append(tbl_row)
    return tables


def _kb_fold_aliases(key_names, tables):
    """Keys that hold the same value in at least 80% of the (two or more) listings carrying both: later name -> the
    one that speaks for it."""
    folded = {}
    for pos_i, lead in enumerate(key_names):
        if lead in folded:
            continue
        for other_k in key_names[pos_i + 1:]:
            if other_k in folded:
                continue
            both_rows = [tbl_row for tbl_row in tables if lead in tbl_row and other_k in tbl_row]
            if len(both_rows) >= 2 and sum(tbl_row[lead] == tbl_row[other_k] for tbl_row in both_rows) >= 0.8 * len(both_rows):
                folded[other_k] = lead
    return folded


def _kb_first_values(tables, group):
    """Per listing, the value under the first of the group's names it carries (None when it carries none)."""
    return [next((tbl_row[nm_x] for nm_x in group if nm_x in tbl_row), None) for tbl_row in tables]


def _kb_entropy(held_vals):
    """Spread of the values seen (natural-log entropy)."""
    spread_v = 0.0
    for val_x in set(held_vals):
        frac_v = held_vals.count(val_x) / len(held_vals)
        spread_v -= frac_v * math.log(frac_v)
    return spread_v


def _kb_menu_entry(kname, group, held_vals, n_total, carried):
    """One property-menu line, with its ranking weight under "_w": value spread times the share of listings carrying
    it (a need sets listings apart; one value everywhere says nothing), or with carried the carrier count."""
    n_carry = len(held_vals)
    shown_vals = list(dict.fromkeys(val_x for val_x in held_vals if val_x))
    w_score = _kb_entropy(held_vals) * n_carry / max(n_total, 1)
    n_distinct = len(set(held_vals))
    w_count = n_carry * (0.5 if n_distinct == 1 and n_carry >= 3 else 1.0)
    if n_carry >= 4 and n_distinct >= n_carry:
        # a distinct value on every listing reads like an id (a year, a code)
        w_score *= 0.5
        w_count *= 0.5
    entry = {"property": kname, "listings": n_carry, "of": n_total, "values": shown_vals[:3]}
    if len(group) > 1:
        entry["same_as"] = group[1:4]
    entry["_w"] = w_count if carried else w_score
    return entry


def _kb_question_text(item, prop, narrow, also):
    """The question for one property (or two); narrow asks for that one thing with up to two example values."""
    noun_s = item.label
    if not narrow:
        if also:
            return f"For the {noun_s}, which {prop} and which {also} would you like?"
        return f"Which {prop} would you like the {noun_s} to have?"
    samples = []
    for cand_o in item.seen.values():
        for pkey, pval in cand_o.props.items():
            if close_names(_kb_spaced(pkey).replace("_", " "), prop) and str(pval) not in samples:
                samples.append(str(pval)[:30])
    examples_s = " or ".join(samples[:2])
    return f"Just this one thing: for the {noun_s}, what {prop} do you need" + (f", for example {examples_s}?" if examples_s else "?")


def _kb_reply_text(obs):
    """The shopper's words in a message result: the text itself, or the longest text field of a record."""
    if isinstance(obs, str):
        return obs
    if not isinstance(obs, dict):
        return None
    texts = [fv_ for fv_ in obs.values() if isinstance(fv_, str)]
    return max(texts, key=len) if texts else brief(obs, 400)


def _kb_part_codes(errand, item):
    """Part codes the item's live candidates point to in their texts that no seen title carries, of a family some of
    the item's titles use."""
    pattern = re.compile(r"\b[A-Z]{1,6}-?\d{3,}[A-Z0-9]*\b")
    in_titles = set()
    for w_ in errand.wants:
        for ob_ in w_.seen.values():
            in_titles.update(pattern.findall(ob_.heading or ""))
    heads = set()
    for ob_ in item.seen.values():
        heads.update(_kb_letters(cd_) for cd_ in pattern.findall(ob_.heading or ""))
    found_c = []
    for ob_ in errand.pool_rows(item):
        if ob_.read_full and not _kb_verdict(item, ob_).get("failed"):
            for cd_ in pattern.findall(" ".join([ob_.blurb or "", ob_.remarks or ""])):
                fresh_c = cd_ not in in_titles and cd_ not in found_c
                if fresh_c and _kb_letters(cd_) in heads:
                    found_c.append(cd_)
    return found_c


def _kb_letters(code):
    """The leading capital letters of a part code."""
    return re.match(r"[A-Z]+", code).group(0)


def _kb_lookup_series(errand, item, sought, series):
    """Search the host's series codes (with the item's kind) and the pointed-to part codes, then read the series'
    accessories, all within the steps the ending leaves."""
    rest = []
    for cd_ in sought:
        if cd_ not in series:
            rest.append(cd_)
    for cd_ in series[:2] + rest[:3]:
        if errand.moves_left() <= errand.ending_steps() + 2 or _kb_short(errand, 10):
            break
        errand.seek(item, query=f"{cd_} {item.kind}" if cd_ in series else cd_, max_pages=1, budget=errand.ending_steps())
    item.series_ok = series[:2]
    if not series or errand.moves_left() <= errand.ending_steps() + 2:
        return
    members = []
    for ob_ in errand.pool_rows(item):
        if not ob_.read_full and errand.of_series(ob_, series):
            members.append(ob_.ref)
    if members:
        width_n = errand.side_max if errand.opens_many() else 3
        errand.open_offers(item, members[:width_n], floor=errand.ending_steps() + 1)


def _kb_series_named(errand, host_item, host):
    """Short codes on the host page ("AB3") that head the part codes of the other items' titles ("AB3-1234"): the
    series that fits the host."""
    blob_s = " ".join([host.blurb or "", host.remarks or "", json.dumps(host.props, ensure_ascii=False)])
    heads = {hd_ for w_ in errand.wants if w_ is not host_item for ob_ in w_.seen.values()
             for hd_ in SERIES_HEAD.findall(ob_.heading or "")}
    own_heads = SERIES_HEAD.findall(host.heading or "")
    found_c = []
    for cd_ in re.findall(r"\b[A-Z]{1,5}\d{1,3}\b", blob_s):
        if cd_ in found_c or cd_ in own_heads:
            continue
        fam_s = re.match(r"[A-Z]+", cd_).group(0)
        if cd_ in heads or any(re.match(r"[A-Z]+", hd_).group(0) == fam_s for hd_ in heads):
            found_c.append(cd_)
    return found_c


def _kb_clear_fit_doubt(errand, item, series):
    """A read candidate whose own part code is of the host's series states the fit: drop that need from its doubts."""
    fit_txt = item.needs[-1].line()
    for ob_ in errand.pool_rows(item):
        book = item.rulings.get(ob_.ref)
        if book and ob_.read_full and errand.of_series(ob_, series):
            for slot_k in ("unsure", "weak", "failed"):
                if slot_k in book:
                    book[slot_k].discard(fit_txt)


def _kb_spaced(kkey):
    """A camelCase key with its words split apart."""
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(kkey))


def _kb_over_cap(cap, offer_x, count_n):
    """True when count units of the listing cost more than the cap allows."""
    if cap is None or offer_x.cost_each is None:
        return False
    return offer_x.cost_each * count_n > cap + 1e-6


def _kb_clean_ruling():
    """A verdict with nothing failed or doubtful."""
    return {"failed": set(), "unsure": set(), "weak": set(), "loose": set()}


def _kb_close_reserve(errand):
    """Steps kept for the close while a last search runs: cart adds, the shopper's yes, checkout and retries."""
    return len(errand.wants) + 5 + max(errand.kept_messages(), 1 if errand.approval_due else 0)


def agent_main(problem_data):
    return Errand(problem_data).go()


__all__ = ["agent_main"]
