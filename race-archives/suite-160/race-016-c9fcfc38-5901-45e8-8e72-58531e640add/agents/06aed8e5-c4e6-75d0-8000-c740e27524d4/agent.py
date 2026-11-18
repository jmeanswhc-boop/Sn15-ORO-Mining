# Sirius v8: a shopping agent for ORO world packs (oro-env-runtime 4.x, runtime contract 8).
#
# Each stage is decided the way the race-7 study found best (model, code or both); see README.md for the stage map.
#
# One shopper, one request, one finish (an order or a decline). Every tool call is one environment step, the world
# states its own budgets in the request (how many messages the shopper takes, how many moves the agent may make), and
# an episode scores 1 only when the finish is fully right: every stated and held-back requirement met, the cheapest
# where the request asks for it, the order within the budget, and a decline when nothing fits.

from __future__ import annotations

import html
import json
import re
import threading
from time import monotonic
from typing import Any

from src.agent.proxy_client import ProxyClient

__version__ = "8.24.0"

# ----------------------------------------------------------------------------- settings
# ---- models, chosen per stage (README: "Models per stage")
# Race #7: 30 of the 36 agents studied ran Claude Haiku 5.5 for the frequent calls and Sonnet 5.5 for the few that decide
# (request and tool reading, verifying a listing, the audit before a decline); that pair had the best mean race score
# (0.163; the one DeepSeek/GLM agent: 0.006). Since Nov 2026 ORO runs every agent on the miner's own OpenRouter account
# (Chutes is retired), so the last fallback of each chain is an open-weights OpenRouter model.
LIGHT = "anthropic/claude-haiku-5.5"
HEAVY = "anthropic/claude-sonnet-5.5"
SPARE = "openai/gpt-6-luna"
TEE_HEAVY = "z-ai/glm-5.2"              # (the names keep their old role: the last fallback, heavy and light)
TEE_LIGHT = "deepseek/deepseek-v3.2"
STAGE_MODELS = {               # the reader's tool name -> the models tried in order
    "map_tools": [HEAVY, LIGHT, SPARE, TEE_HEAVY],       # 1 tool mapping (only when the schemas leave it open)
    "plan": [HEAVY, SPARE, LIGHT, TEE_HEAVY],            # 2-3 request reading, required steps, closest-fit rule
    "steer": [LIGHT, SPARE, TEE_LIGHT],                  # 4 next store moves (frequent, a menu to pick from)
    "search_plan": [LIGHT, SPARE, TEE_LIGHT],            # 5 new searches
    "shelf_and_searches": [HEAVY, LIGHT, SPARE, TEE_HEAVY],  # 4-5 the product's store shelf and its first searches (once per product)
    "judge": [LIGHT, SPARE, HEAVY, TEE_LIGHT],           # 6 listing marks (batches; code overrules)
    "follow": [LIGHT, HEAVY, SPARE, TEE_HEAVY],          # 6 a value read from another listing
    "ask_about": [HEAVY, LIGHT, SPARE, TEE_HEAVY],       # 7 which properties to ask (race #13: decided the task)
    "hear": [LIGHT, HEAVY, SPARE, TEE_LIGHT],            # 8-9 the shopper's answers and notices
    "check": [HEAVY, LIGHT, SPARE, TEE_HEAVY],           # 11 last check of a listing before the cart
    "audit": [HEAVY, LIGHT, SPARE, TEE_HEAVY],           # 12 the closest listings before a decline
    "pair": [HEAVY, LIGHT, SPARE, TEE_HEAVY],            # 11 which products fit each other when no page lists models
}
DEFAULT_MODELS = [LIGHT, SPARE, TEE_LIGHT]
# $ per million tokens (input, output) on OpenRouter, 4 Nov 2026 - only for a reply that carries no usage.cost (the ORO
# proxy asks OpenRouter for it, so on the network the cost is read, not estimated); thinking tokens bill as output.
PRICES = {LIGHT: (0.10, 0.50), HEAVY: (2.00, 10.00), SPARE: (0.10, 0.50), TEE_HEAVY: (0.60, 2.20), TEE_LIGHT: (0.27, 0.40)}
# The miner's OpenRouter account pays, through a key ORO mints for each evaluation run with a cap: $5 a race run (90
# tasks), $2 a qualifying run. Past the cap every model call fails for the rest of the run, so each task keeps to its share:
# past the soft cap Haiku leads every stage but the last check of the choice; at the hard cap code finishes alone.
SPEND_SOFT_USD = 0.035
COST_CEILING_USD = 0.05          # (90 x 0.05 = $4.50, under the $5 race cap; a typical task spends about $0.04)
SOFT_KEEP = ("check",)
MODEL_STRIKES = 2             # a model that failed twice in an episode (refused name, outage) is skipped after
TASK_TIME_S = 270.0      # each task has 300 s on the network
MODEL_WAIT_S = 70.0           # the longest one model call is waited for
THINK_UNTIL_S = 150.0         # no thinking pass once fewer seconds than this are left (the slowest calls)
HURRY_KEEP = ("check", "hear")  # the only reader calls made in a hurry: the last check of the choice and the shopper's notices
CLOSING_MARGIN = 10           # steps kept for the finish until the request is read (then finish_reserve() sizes it)
DEFAULT_QUESTIONS = 2           # questions when neither the request nor the store says the message budget
REF_STEPS = 38                # the step limit the step plan was tuned on; its step counts scale to each task's own limit
# A request's own limits, as races word them: "wrap up inside 38 steps", "38 moves is the most you can make", "tool calls
# are capped at 50"; "you can message me 2 times at most", "total of 3 messages from your side". A bare number beside
# such a word ("3 calls to the landlord") is no limit.
STEP_WORDS = r"(?:moves?|steps?|tool calls?|actions?|calls?|turns?)"
STEP_LIMIT = [re.compile(p, re.I) for p in (
    rf"(?:inside|within|in|under|at most|no more than|max(?:imum)? of|up to|only)\s+(\d+)\s+{STEP_WORDS}\b",
    rf"(\d+)\s+{STEP_WORDS}\s+(?:is|are)\s+(?:the\s+)?(?:most|max|limit|cap|all)",
    rf"(\d+)\s+{STEP_WORDS}\s+(?:at most|max(?:imum)?|tops|or (?:fewer|less))",
    rf"{STEP_WORDS}\s+(?:are\s+|is\s+)?(?:capped|limited|kept|held)\s+(?:at|to)\s+(\d+)",
    rf"(?:limit|cap|budget|ceiling)\s+(?:of|is)\s+(\d+)\s+{STEP_WORDS}\b")]
GROUP_SIZE = 1                     # one store action per call, in order, as ORO's rules ask ("send environment steps one at a
                              # time"); the envelope code still handles a list
GRADE_GROUP = 12              # listings shown to the reader in one call
PAGE_GUESSES = 7                # rows a search page shows when the world does not say
TOKEN_FLOOR = 900
REASONING_TOKENS = 4000           # room for a thinking pass before the tool call is written
SEVERAL_CUE = re.compile(r"\b(?:two|three|four|2|3|4)\s+(?:different\s+)?(?:things|items|products|kinds)\b|\bfirst\b[^.]{0,80}[.,;].{0,400}\bsecond(?:ly)?\b"
                        r"|\bas well as\b|\balso (?:need|want|after|looking|get|grab|pick)\b|\bplus\b|\balong with\b|\btogether with\b"
                        r"|\balongside\b|\bon top of that\b|\band also\b|\banother (?:thing|item|product)\b|\bthe other (?:thing|item)\b"
                        r"|\b(?:each|both) of (?:them|these|those)\b|\bone (?:order|basket|cart) for\b", re.I | re.S)
# A cart tool's argument for how many units (race #6: "Pieces" went unread, so 149 lines went in at one unit each)
COUNT_TERMS = r"qty|quant|how many|count|units?\b|amount|number|pieces?\b|pcs|copies|multiple|\blots?\b|volume of units"
# A step on the way to the order is done only when the shopper asks for it: "show me the basket before you order", "tell
# me the total before you pay", "get my go-ahead before checking out", "run it by me first". A hint is no step: "tell me
# what's worth asking before I commit", "I like knowing how a thing came together before paying" (race #13: a message
# sent after the cart was filled went with 7 of 156 passes - an unasked one only spends a question).
_STEP_OBJECT = r"(?:basket|cart|trolley|bag|total|sum|subtotal|bill|order|items?|lines?|what(?:'s| is) in|selection|pick|choice)"
CONSENT_CUE = re.compile(
    rf"(?:\b(?:show|tell|read|send|list|run|give|walk|text|message)\s+(?:me|it|them)?\s*(?:through\s+)?(?:the|my|your|what's in the)?\s*{_STEP_OBJECT}\b[^.?!]{{0,60}}"
    rf"\b(?:before|prior to|first|ahead of)\b"
    rf"|\b(?:before|prior to)\b[^.?!]{{0,40}}\b(?:order|buy|pay|checkout|check out|purchase|place|commit|finali[sz]e)\w*[^.?!]{{0,40}}"
    rf"\b(?:show|tell|read|send|run|confirm|check with|clear (?:it )?with|ok it|okay it|sign off|go-?ahead|approv\w*|green light)\b[^.?!]{{0,30}}\b(?:me|my|with me|{_STEP_OBJECT})"
    rf"|\b(?:get|wait for|need|want)\s+(?:my|an?)\s+(?:go-?ahead|approval|ok|okay|sign-?off|green light|say-?so)\b"
    rf"|\brun it (?:by|past) me\b|\bcheck with me (?:first|before)\b|\bconfirm with me\b)", re.I)
# A request that wants the nearest listing offered when nothing fits exactly (race #13: 67 of 90 tasks). With no clean fit
# the right finish is a decline that says so and names the nearest listing: in the two such tasks all 14 declines were
# graded right but for the procedure, and all 4 near-miss orders wrong. (The leader's 27 failed declines were in tasks
# that had a clean fit.)
NEAR_HINT = re.compile(r"no exact (?:fit|match)|closest (?:match|listing|fit|thing)|nearest (?:match|fit|thing)|next best", re.I)
OVER_PCT = re.compile(r"up to (\d+(?:\.\d+)?)\s*%\s*more", re.I)
# A model or part code ("VIT-8030", "VIR3110") and the store's compatibility line ("Compatible with: every VIT-80xx model,
# except VIT-8030.") - race #13: 40 of 90 tasks turned on what fits what.
MODEL_CODE = re.compile(r"\b([A-Z]{2,5}-?\d[\dxX]{2,5})\b")
COMPAT_REQ = re.compile(r"compat|\bfits?\b|works? with|\bmodel\b|goes with|mates? up", re.I)
# "Whatever feature phones you choose, the wireless chargers must be compatible with it": the first names the anchor,
# the second the product bound by the rule (race #13: other products in the order are not bound).
# race #13: "Pick the combo with the smallest total" (26 of 90), "Multiple fits - cheapest one wins"
CHEAPEST_WORDS = (r"cost(?:s)? the least|cheapest|lowest price|least expensive|smallest (?:total|price|bill)|lowest (?:total|bill)|least in total|"
                  r"smaller (?:total|bill)|"
                  # value for money with no measure named: the best option is the cheapest that fits (the grader's "best
                  # option" is the lowest acceptable total)
                  r"value for (?:my |the |your )?money|best value|good value|great value|bang for (?:my|the|your) bucks?|most for (?:my|the|your) money|"
                  r"money's worth|best deal|good deal|bargain|(?:not|don't|won't|never) (?:want to |wanna )?overpay|budget[- ]friendly|easy on the (?:wallet|purse|pocket)")
# "The most battery capacity you can get", "the lightest one under 2000 cents", "as much storage as possible", "the
# longest battery life within budget": the best by a measure other than price (price stays a ceiling).
# (generic amount words only: a superlative tied to one measure, "the lightest", is the reader's best_by - ORO bans
# tables of attribute words)
_MOST = r"highest|largest|biggest|greatest|most|maximum|max|best"
_LEAST = r"lowest|smallest|fewest|least|minimum|min"
OBJECTIVE = [re.compile(p, re.I) for p in (
    rf"\b(?:the\s+|with\s+the\s+|has\s+the\s+)(?P<adj>{_MOST}|{_LEAST})\s+(?P<p>[a-z][a-z ]{{1,30}}?)(?=\s+(?:you\s+can|possible|within|under|for|in|on|that|i\s+can|available|going|there\s+is)\b|\s*[.,;!?)\u2013\u2014]|\s-\s|$)",
    rf"\bas\s+(?P<adj>much|many|big|high|large)\s+(?:a\s+)?(?P<p>[a-z][a-z ]{{0,30}}?)\s*as\s+(?:possible|you\s+can|the\s+budget)",
    rf"\bthe\s+(?P<adj>more|higher|bigger|larger)\s+(?:the\s+)?(?P<p>[a-z][a-z ]{{0,30}}?),?\s+the\s+better",
)]
_OBJ_STOP = re.compile(r"^(?:of|days?|evenings?|mornings?|nights?|weeks?|times?|people|folks|of the|i can|you can|part|bit|thing|things|"
                       r"one i|choice|price|cost|total|budget|spend|money|cheap|expensive|value|bang|deal|bargain|messages?|steps?|moves?|calls?)\b", re.I)


# Value for money by a measure: "the best price per ml", "cheapest per serving", "the most servings for my money",
# "best value per piece" - the lowest price for each unit of the measure (price / the listing's amount of it).
_PER_UNIT = r"[a-z]{1,12}"      # (whatever unit the shopper names: no list of units)
PER_UNIT = [re.compile(p, re.I) for p in (
    rf"\b(?:price|cost|cents?|money|value|pay(?:ing)?)\s+(?:per|for each|for every|each)\s+(?P<u>{_PER_UNIT})\b",
    rf"\b(?:cheapest|lowest|least|best)\s+(?:price\s+)?(?:per|for each)\s+(?P<u>{_PER_UNIT})\b",
    rf"\b(?:the\s+)?most\s+(?P<u>{_PER_UNIT})\s+(?:for|per)\s+(?:my|the|your|each)\s+(?:money|cash|penny|pennies|cent|cents|dollar)",
    rf"\b(?:most|more)\s+(?P<u>{_PER_UNIT})\s+for\s+(?:the|my|your)\s+(?:price|spend)")]


def objective_by_code(text: str) -> dict | None:
    # {"kind": "most" | "least", "property": ...} when the request asks for the best by a measure other than price;
    # {"kind": "least", "property": "price per <unit>", "per": <unit>} for value for money by a measure.
    t = " ".join(str(text or "").split())
    for rx in PER_UNIT:
        m = rx.search(t)
        if m:
            u = low(m.group("u"))
            u = u[:-1] if len(u) > 4 and u.endswith("s") and not u.endswith("ss") else u     # ('servings' -> 'serving')
            return {"kind": "least", "property": f"price per {u}", "per": u, "adj": "per"}
    for rx in OBJECTIVE:
        for m in rx.finditer(t):
            adj, prop = low(m.group("adj")), low(m.group("p")).strip()
            if prop in ("one", "of them", "option", "pick", ""):
                continue        # ("the lightest one": which measure is the reader's call)
            if not prop or _OBJ_STOP.match(prop) or re.search(r"\b(?:cheap|price|cost|budget|cents?)\b", prop):
                continue
            kind = "least" if re.match(rf"^(?:{_LEAST})$", adj) else "most"
            return {"kind": kind, "property": prop, "adj": adj}
    return None


MESSAGE_CAP = 380             # characters in one message (race #13: a world cut messages near 800, and a long one confuses)
ASK_TEMPLATE = "Which {} would you like?"
ASK_TEMPLATE_EACH = "For your {0}, which {1} would you like?"
ASK_TEMPLATE_FREE = "Could you let me know {}?"


_proxy = ProxyClient(timeout=90, max_retries=2)


# ----------------------------------------------------------------------------- text helpers
def low(text: Any) -> str:
    return " ".join(str(text or "").lower().replace("’", "'").split())


def label_tokens(name: Any) -> str:
    # A tool or argument name as words: 'ListRowStart' -> 'list row start', 'item_shelf' -> 'item shelf'.
    return low(re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(name or "")).replace("_", " ").replace(".", " "))


def words(text: Any) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", low(text))


def stem(word: str) -> str:
    w = word.lower()
    for suffix, keep in (("ies", "y"), ("sses", "ss"), ("ches", "ch"), ("shes", "sh"), ("xes", "x")):
        if len(w) > len(suffix) + 2 and w.endswith(suffix):
            return w[: -len(suffix)] + keep
    if len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
        w = w[:-1]
    for suffix in ("ing", "ed"):
        if len(w) > len(suffix) + 3 and w.endswith(suffix):
            w = w[: -len(suffix)]
            break
    return w


# ordinary English function words (articles, prepositions, conjunctions, pronouns), skipped when words are compared
FILLER = {"a", "an", "and", "the", "of", "for", "to", "in", "on", "with", "or", "is", "it", "be", "by", "at", "as", "that",
        "this", "its", "your", "my", "any", "some"}


def roots(text: Any, hold_filler: bool = False) -> list[str]:
    return [stem(w) for w in words(text) if hold_filler or w not in FILLER]


def holds(corpus: Any, probe_text: Any) -> bool:
    # Every content word of needle appears in haystack (stemmed, whole words).
    want = roots(probe_text)
    have = set(roots(corpus, hold_filler=True))
    return bool(want) and all(w in have for w in want)


def spread(text: Any) -> str:
    # Figures parted from their units ('3440x1440' -> '3440 x 1440', '30mm' -> '30 mm').
    return re.sub(r"(?<=\d)(?=[a-z])|(?<=[a-z])(?=\d)", " ", low(text))


PLAIN_LABEL = {"of", "the", "for", "a", "an"}     # function words in a label


def label_terms(key: Any) -> list[str]:
    return [w for w in dict.fromkeys(roots(label_tokens(key), hold_filler=True)) if w not in PLAIN_LABEL]


def attr_terms(prop: Any) -> set[str]:
    return set(roots(prop, hold_filler=True))


def label_agrees(key: Any, prop: Any) -> bool:
    # a fact label names the property: a shared word, or one word the start of the other
    k, p = set(label_terms(key)), {w for w in attr_terms(prop) if w not in PLAIN_LABEL}
    if not p:   # (a property named only by a grammar word is matched by that word)
        k, p = set(roots(label_tokens(key), hold_filler=True)), attr_terms(prop)
    if not k or not p:
        return False
    if k & p:
        return True
    return any(len(a) >= 4 and len(b) >= 4 and (a.startswith(b[:4]) or b.startswith(a[:4])) for a in k for b in p)


# ----------------------------------------------------------------------------- amounts with units (code decides)
# A figure and the unit written after it, compared only with the same unit ("16 gb" with "16 gb", "15.6 inches" with
# "15.6 inch"). No conversion table: a size written in another unit is left to the model (code marks it not stated).
_NUM = r"\d+(?:[.,]\d+)?"
# a figure: '1 3/8', '1-3/8', '3/4', '2.5', '1,250' (a thousands comma), then an optional unit word or symbol
QUANTITY = re.compile(rf"(?<![\w.])((?:\d+[ -])?\d+/\d+|{_NUM})\s*([a-z%\"'\u00b0]{{1,8}})?(?![a-z])", re.I)


def unit_key(unit: Any) -> str:
    # one spelling of a written unit: case and plural folded ("Inches" ~ "inch", "watts" ~ "watt")
    u = low(unit).strip(" .")
    return stem(u) if len(u) > 3 else u


def _figure(text: str) -> float:
    text = text.strip()
    m = re.fullmatch(r"(?:(\d+)[ -])?(\d+)/(\d+)", text)
    if m:
        return float(m.group(1) or 0) + float(m.group(2)) / max(1.0, float(m.group(3)))
    if re.fullmatch(r"\d{1,3}(?:,\d{3})+", text):
        return float(text.replace(",", ""))
    return float(text.replace(",", "."))


def quantities(text: Any) -> list[tuple[float, str]]:
    return [(v, d) for v, d, _ in quantities_u(text)]


def quantities_u(text: Any) -> list[tuple[float, str, str]]:
    # Every figure a text states with its unit as written ("" when none): "18 x 36 x 18 cm" -> 18, 36, 18 cm (a unit
    # after the last figure of an 'a x b' size applies to each). The second slot (the 'dimension') is the unit itself:
    # figures compare only with figures in the same unit.
    t = low(text).replace("\u2033", '"').replace("\u2032", "'").replace("\u201d", '"').replace("\u2019", "'")
    t = re.sub(r"(\d|\"|')\s*[x\u00d7*]\s*(?=\d)", r"\1 x ", t)        # ('3440x1440' -> '3440 x 1440')
    out: list[tuple[float, str, str]] = []
    found = [(m.group(1), unit_key(m.group(2) or ""), m.end()) for m in QUANTITY.finditer(t)]
    found = [(n, "" if u == "x" else u, e) for n, u, e in found]
    for i, (num, unit, end) in enumerate(found):
        if not unit:
            j = i
            while j + 1 < len(found) and re.fullmatch(r"\s*[x\u00d7*]\s*", t[found[j][2]: t.find(found[j + 1][0], found[j][2])] or "-"):
                j += 1
                if found[j][1]:
                    unit = found[j][1]
                    break
        try:
            v = _figure(num)
        except ValueError:
            continue
        out.append((v, unit, unit))
    return out


def units_differ(text: Any, value: Any) -> bool:
    # The text gives figures, but only in units other than the value's: the same size may be written another way, which
    # code does not convert (the model reads it).
    want = {d for _, d, _ in quantities_u(value) if d}
    have = {d for _, d, _ in quantities_u(text) if d}
    return bool(want) and bool(have) and not (want & have)


LIMIT_RE = re.compile(r"(at most|at least|no more than|no less than|not more than|not less than|up to|not over|no over|under|over|below|"
                   r"above|minimum(?: of)?|maximum(?: of)?|max\.?|min\.?|less than|more than|greater than|fewer than|"
                   r"no (?:higher|bigger|larger)(?: than)?|"
                   r"no (?:smaller|lower)(?: than)?|within|<=|>=|=<|=>|<|>|\u2264|\u2265)\s*"
                   r"((?:\d+[ -])?\d+/\d+|\d+(?:[.,]\d+)?)\s*([a-z%\"']*)"
                   r"|((?:\d+[ -])?\d+/\d+|\d+(?:[.,]\d+)?)\s*([a-z%\"']*)\s*(or (?:more|above|higher|greater|over|larger|bigger|longer|less|"
                   r"under|below|fewer|lower|smaller|shorter)|\+|and (?:up|above|over)|and (?:under|below|less)|at (?:the )?(?:most|least)|tops|max(?:imum)?|min(?:imum)?)", re.I)


def limit_of(value: Any) -> tuple[str, float, str] | None:
    # A limit a value states: (kind '<=' or '>=', the figure in its base unit, its dimension).
    m = LIMIT_RE.search(low(value))
    if not m:
        return None
    if m.group(1):
        op, num, unit = m.group(1), m.group(2), m.group(3)
    else:
        num, unit, op = m.group(4), m.group(5), m.group(6)
    lower = re.search(r"most|no more|not more|up to|not over|no over|under|below|max|less|fewer|lower|smaller|within|"
                      r"no (?:higher|bigger|larger)|<|\u2264|tops", op)
    # (inclusive unless the words exclude the figure itself: "less than 3", "more than 2", "under 80", "over 5", "<")
    strict = bool(re.search(r"^(?:less than|more than|greater than|fewer than|under|over|below|above|<|>)$", op.strip())) and "not" not in op
    dim, f = unit_key(unit), 1.0
    try:
        figure = _figure(num) * f
    except ValueError:
        return None
    return ("<" if strict else "<=") if lower else (">" if strict else ">="), figure, dim


RANGE = re.compile(r"\b(?:between|from)\s+((?:\d+[ -])?\d+/\d+|\d+(?:[.,]\d+)?)\s*([a-z%\"']*)\s+(?:and|to|-|\u2013)\s+"
                   r"((?:\d+[ -])?\d+/\d+|\d+(?:[.,]\d+)?)\s*([a-z%\"']*)", re.I)


def range_of(value: Any) -> tuple[float, float, str] | None:
    # "between 2 and 3 inch" / "from 30 to 40 cm": (low, high, dimension), both ends included.
    m = RANGE.search(low(value))
    if not m:
        return None
    dim = unit_key(m.group(4) or m.group(2) or "")
    try:
        a, b = _figure(m.group(1)), _figure(m.group(3))
    except ValueError:
        return None
    return min(a, b), max(a, b), dim


def same_amounts(text: Any, value: Any) -> bool | None:
    # An exact size or amount ("1 3/8 in", "2'6\" x 8'", "3440 x 1440", "35 mm") against a text: True when the text states
    # the same figures (after converting units, within 2%), False when it states other figures of that dimension, None
    # when either side has no figures to compare (the words decide then).
    want = quantities_u(value)
    if not want:
        return None
    have = quantities_u(text)
    dims = {d for _, d, _ in want}
    have = [h for h in have if h[1] in dims or not h[1] or not dims - {""}]
    if not have:
        return None

    def close(a: tuple, b: tuple) -> bool:
        # (an exact value is not rounded: "1.5 inch" is not "1.52 inch")
        tol = 0.001
        return abs(a[0] - b[0]) <= max(tol * max(abs(a[0]), abs(b[0])), 1e-9)
    left = list(have)
    for v, d, u in want:
        match_found = next((h for h in left if close(h, (v, d, u)) and (h[1] == d or not h[1] or not d)), None)
        if match_found is None:
            return False
        left.remove(match_found)
    return True


_BEFORE = [("at least", r"(?:at least|no less than|not less than|minimum(?: of)?|min\.?|upwards of|a minimum of|nothing (?:under|below|less than))\s*$"),
           ("more than", r"(?:more than|over|above|greater than|beyond|exceeding)\s*$"),
           ("at most", r"(?:at most|no more than|not more than|up to|maximum(?: of)?|max\.?|a maximum of|within|no (?:bigger|"
                       r"larger|higher)(?: than)?|nothing (?:over|above|more than))\s*$"),
           ("less than", r"(?:less than|under|below|fewer than|smaller than|lower than)\s*$")]
_AFTER = [("at least", r"^\s*(?:[a-z\"']{1,8}\s*)?(?:or more|or above|or higher|or greater|or bigger|\+|and up(?:wards)?|at (?:the )?least|minimum)\b"),
          ("at most", r"^\s*(?:[a-z\"']{1,8}\s*)?(?:or less|or under|or below|or fewer|or lower|or smaller|at (?:the )?most|max(?:imum)?|tops)\b")]


def bound_in_text(text: str, value: Any) -> str:
    # How the shopper's own words bind the value's figure where they write it: "exact", "at least", "at most", "more
    # than", "less than", a "between ... and ..." phrase, or "" when the figure is not in the text.
    v = str(value or "")
    numerals = re.findall(r"\d+(?:[.,]\d+)?(?:\s*/\s*\d+)?", v)
    if not numerals:
        return ""
    t = " ".join(str(text or "").split())
    for m in re.finditer(rf"(?<![\d.]){re.escape(numerals[0])}(?![\d])", t):
        before, after = low(t[max(0, m.start() - 40): m.start()]), low(t[m.end(): m.end() + 30])
        span = re.search(rf"(?:between|from)\s+{re.escape(numerals[0])}\s*[a-z\"']*\s+(?:and|to)\s+\d+(?:[.,]\d+)?\s*[a-z\"']*", low(t[max(0, m.start() - 10):]))
        if span and span.start() <= 10:
            return span.group(0)
        for kind, rx in _BEFORE:
            if re.search(rx, before):
                return kind
        for kind, rx in _AFTER:
            if re.search(rx, after):
                return kind
        return "exact"
    return ""


def bind_value(r: dict, said: str) -> tuple[str, str]:
    # The value a requirement holds, its figure bound as the shopper said it: the reader's bound when its words are not in
    # the text, the text's own words when they are (they win). Returns (value, note).
    value = str(r.get("value") or "").strip()
    if not re.search(r"\d", value):
        return value, ""
    kind = low(r.get("bound") or "")
    lo, upper_val = number(r.get("low")), number(r.get("high"))
    unit = str(r.get("unit") or "").strip()
    figure = re.sub(r"^(?:at least|at most|no more than|no less than|up to|more than|less than|under|over|between|from|minimum|maximum)\s+", "", low(value))
    if kind == "between" and lo is not None and upper_val is not None and not range_of(value):
        value = f"between {lo:g} and {upper_val:g} {unit}".strip()
    elif kind in ("at least", "at most", "more than", "less than") and not limit_of(value):
        value = f"{kind} {figure}"
    told = bound_in_text(said, figure)
    note = ""
    if told and told != "exact" and told.startswith(("between", "from")):
        if not range_of(value):
            value, note = told, f"bound from the shopper's words: {told}"
    elif told in ("at least", "at most", "more than", "less than"):
        has = limit_of(value)
        want = {"at least": ">=", "at most": "<=", "more than": ">", "less than": "<"}[told]
        if not has or has[0] != want:
            value, note = f"{told} {figure}", f"bound from the shopper's words: {told} {figure}"
    elif told == "exact" and (limit_of(value) or range_of(value)):
        value, note = figure, f"exact as the shopper wrote it: {figure}"
    return value, note


def list_items(text: Any) -> list[str]:
    # the items of a fact that lists several values ("trans fat free, sugar free, caffeine free")
    return [x for x in re.split(r"\s*[,;|]\s*|\s*\n\s*", str(text or "")) if x.strip()]


def value_matches(text: Any, value: Any, skip: set[str] = frozenset()) -> bool:
    # the text states the value: a figure within its bound (units converted), the same sizes, or every content word of it
    # (words of the property's own name aside); other ways of writing a value come from the reader (same_meaning)
    items = list_items(text)
    if len(items) > 1 and len(roots(spread(value))) > 1 and not re.search(r"\d", str(value)):
        # (a fact listing several values: a value of several words is met by one item, not by words spread over the
        # list - "sugar free, low fat" does not say "low sugar")
        need = roots(spread(value))

        def leads(item: str) -> bool:
            # (the item opens with the value's words, figures aside: "trans fat free" is a narrower thing than "fat free")
            got = [w for w in roots(spread(item)) if not re.fullmatch(r"[\d.%]+", w)]
            return got[: len(need)] == need
        return any(leads(item) for item in items)
    span = range_of(value)
    if span:
        lo, upper_val, dim = span
        numbers = [v for v, d in quantities(text) if not dim or not d or d == dim]
        return bool(numbers) and lo * 0.9999999 - 1e-9 <= numbers[0] <= upper_val * 1.0000001 + 1e-9
    bound = limit_of(value)
    if bound:
        kind, limit, dim = bound
        numbers = [v for v, d in quantities(text) if not dim or not d or d == dim]
        if not numbers:
            return False
        figure = numbers[0]
        eps = abs(limit) * 1e-7 + 1e-9
        return {"<=": figure <= limit + eps, "<": figure < limit - eps, ">=": figure >= limit - eps, ">": figure > limit + eps}[kind]
    if re.search(r"\d", str(value)) and re.fullmatch(r"(?:[\d\s.,/x\u00d7*'\"-]|(?<=[\d\s])[a-z%\u00b0]{1,8}\b)+", low(value)):
        same = same_amounts(text, value)       # (a value that is only figures and units)
        if same is not None:
            return same
    text = re.sub(r"(?<=\w)'s\b", "s", str(text))
    if quantities(value) and same_amounts(text, value) is False:
        return False        # ("cable length 2.7432 m" is not met by a page that only says "cable length")
    need = [w for w in roots(spread(value)) if w not in skip] or roots(spread(value))
    have = set(roots(spread(text), hold_filler=True))
    if bool(need) and all(w in have for w in need):
        return True
    # (the same word joined or split: "non-woven" / "nonwoven", "slip on" / "slip-on", "t shirt" / "tshirt")
    flattened = re.sub(r"[^a-z0-9]", "", low(value))
    return len(flattened) >= 5 and len(need) <= 3 and flattened in re.sub(r"[^a-z0-9]", "", low(text))


def number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    m = re.search(r"-?\d+(?:\.\d+)?", str(value or "").replace(",", ""))
    return float(m.group(0)) if m else None


def listify(value: Any) -> list:
    # A list a reader wrote as a list, or as a JSON string of one.
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value.strip().startswith("["):
        try:
            decoded = json.loads(value)
        except ValueError:
            decoded = LlmGateway._closed_json_list(value) if hasattr(LlmGateway, "_closed_json_list") else None
        return decoded if isinstance(decoded, list) else []
    return []


def entry_pointer(unparsed: Any) -> dict | None:
    # Another listing a value is to be read from, as the reader named it (its kind, its seller, details), and how the
    # value relates to it: "same" (matching it), "pairs" (a type that mates up with it, by the store's pairing rule) or
    # "not" (definitely not it).
    if not isinstance(unparsed, dict) or not str(unparsed.get("seller") or "").strip():
        return None
    relation = low(unparsed.get("relation") or "same")
    out = {"item": str(unparsed.get("item") or "").strip(), "seller": str(unparsed["seller"]).strip(),
           "details": [str(d).strip() for d in listify(unparsed.get("details")) if str(d).strip()][:4],
           "relation": ("pairs" if relation.startswith("pair") or "mate" in relation else "not" if relation.startswith("not")
                        else "within" if relation.startswith("within") and number(unparsed.get("tolerance")) else "same")}
    if out["relation"] == "within":
        out["tolerance"], out["unit"] = number(unparsed.get("tolerance")), str(unparsed.get("unit") or "").strip()
    return out


def split_pointers(unparsed: list, said: str, stated: bool) -> tuple[list, list]:
    # Requirements whose value is another listing's (its seller named in the shopper's own words), apart from the rest.
    kept, refs = [], []
    for r in unparsed:
        ref = entry_pointer(r.get("from_listing")) if isinstance(r, dict) else None
        if ref and holds(said, ref["seller"]):
            # (the relation the shopper's own words give wins over the reader's: "a type that mates up with whatever's on
            # the page for X" pairs, "definitely not whatever's on the page for X" excludes)
            told = relation_in(said, ref["seller"])
            if told:
                ref["relation"] = told
            coded = next((c["ref"] for c in refs_by_code(said, stated) if low(c["ref"]["seller"]) == low(ref["seller"])), None)
            if coded and coded["relation"] in ("within", "not", "pairs") and ref["relation"] == "same":
                ref.update({k: v for k, v in coded.items() if k in ("relation", "tolerance", "unit")})
            refs.append({"property": str(r.get("property") or "").strip(), "ref": ref, "stated": stated})
        else:
            kept.append(r)
    return kept, refs


# ----------------------------------------------------------------------------- other listings' pages (race #13)
# A shopper ties a value to another listing: "the color is a type that mates up with whatever's on the page for
# Lenshimi's listing of the usb and lightning cables - the one with data transfer rate 480 mbps and cable length 1.00584 m
# - or wherever that page ends up sending you" (18 of 90 requests, 28 tasks' answers). That page may send you on ("For
# color, use the Dorkodorri USB and Lightning cable, which is 1.37 m long, ..."), until a page states the value itself.
# "matching ... exactly" wants that value; "a type that mates up with" wants its partner by the store's pairing rule
# ("Our color rule pairs black with wenge, espresso with beige"); "definitely not" rules it out. (Checked against the
# leaders' own order notes: "Colour chain opened in full: ... charger (gold) -> ... whose page pairs gold with bronze.
# Handle colour bronze.")
def refs_by_code(text: str, stated: bool) -> list[dict]:
    # (references to another listing are read by the model - the plan's and the answers' from_listing - not by phrase
    # patterns in code, which would be fitted to one problem suite's wording)
    return []


def relation_in(text: str, seller: str) -> str:
    return ""


def pointer_for(text: Any, prop: str) -> str:
    return ""       # (a page that sends you on to another is read by the model: TRACE_FORM's "next")


def ref_from_text(text: str) -> dict | None:
    # A pointer's listing as seller, kind and details: "the Dorkodorri USB and Lightning cable, which is 1.37 m long,
    # with a usb c to usb c connector type" / "an Area Rugs product from Shibelzen that is hand-tufted, made of polyester"
    # / "Qilumo's power adaptor and charger (asus brand compatibility, 2.37 output current)".
    t = str(text or "")
    seller = (re.search(r"\b(?:from|by|sold by)\s+([A-Z][\w-]+)", t) or re.search(r"\b([A-Z][\w-]+)'s\b", t)
              or re.search(r"\b(?:the|another [a-z ]+?, the)\s+([A-Z][\w-]+)\b", t))
    if not seller:
        return None
    details = [d.strip() for text_piece in re.findall(r"\(([^)]*)\)", t) for d in re.split(r",|\band\b", text_piece) if d.strip()]
    details += [d.strip() for m in re.finditer(r"\b(?:which is|that is|with an?|with|that has|made of|sized|in)\s+([^,()]+)", t)
                for d in re.split(r",|\band\b", m.group(1)) if d.strip()]
    lead_part = re.split(r",|\bwhich\b|\bthat\b|\bwith\b", re.sub(r"\(.*?\)", " ", t))[0]
    lead_part = re.sub(rf"\b{re.escape(seller.group(1))}(?:'s)?\b", " ", lead_part)
    kind = " ".join(w for w in lead_part.split()
                    if low(w) not in ("the", "a", "an", "from", "by", "product", "listing", "item", "one", "another", "sold", "this"))
    return {"item": kind.strip(), "seller": seller.group(1), "details": details[:5], "relation": "same"}


def rules_in(text: Any, facts: dict | None = None) -> dict[str, dict[str, str]]:
    return {}       # (a store's pairing rule is read by the model: PARTNER_TOOL)


# ----------------------------------------------------------------------------- the request, read by code
# The model reads the request (Sonnet, with thinking); this reader checks it. Races word the same few things in a few
# templates each (practice, qualifying and race-13 requests): a price ceiling, a per-item ceiling, "cheapest", the
# products with their counts, and each required value tied to its property.
_MONEY = r"(\d[\d,]*(?:\.\d+)?)\s*(?:usd[- ]?cents?|us\s*cents?|cts|cents?|\$|dollars?|usd)"      # (race 14: "2250 usd-cents")
TOTAL_CAP = [re.compile(p, re.I) for p in (
    rf"(?:nothing|not a cent|no(?:thing)? (?:item )?(?:priced )?)\s*(?:over|above|past|more than)\s*(?:a mere\s*)?{_MONEY}",
    rf"(?:can't|cannot|can not|won't|shouldn't|mustn't|must not|not to)\s+(?:go\s+)?(?:over|above|past|exceed|top)\s*(?:a mere\s*)?{_MONEY}",
    rf"(?:price|spend(?:ing)?|budget|cost|total|whole (?:lot|order|thing)|all up it|the lot|all of it|ceiling|cap)[^.;]{{0,40}}?"
    rf"(?:tops out at|ceiling is|cap(?:ped)? (?:is|at)|limit is|is|comes? (?:in )?(?:at|to)|stays? (?:at or )?under|at or under|at most|max(?:imum)?(?: is)?|up to|"
    rf"under|within|of)\s*(?:a mere\s*|just\s*|only\s*)?{_MONEY}",
    rf"(?:can|could|will|i'll|willing to)\s+(?:only\s+)?(?:spend|pay|afford|put up|throw)[^.;]{{0,25}}?{_MONEY}",
    rf"{_MONEY}\s*(?:or (?:under|less|below)|at (?:the very )?most|tops|max(?:imum)?)",
    rf"(?:keep(?:ing)? (?:it|things|everything) )?(?:at or )?under\s*(?:a mere\s*)?{_MONEY}",
    # race 14: "I can part with 380 usd-cents", "Top whack is 365", "here's my ceiling: 305", "I've got 270 usd-cents to
    # spend", "Happy to spend up to 455", "The final total can run up to 53500"
    rf"\bpart\s+with\s+{_MONEY}",
    rf"\btop\s+whack\s+(?:i\s+can\s+spend\s+)?is\s+{_MONEY}",
    rf"\bceiling\W+\s*{_MONEY}",
    rf"\bgot\s+{_MONEY}\s+to\s+spend",
    rf"\b(?:spend|run|go)\s+up\s+to\s+{_MONEY}")]
# race 14: "No more than 2650 usd-cents, that is on each streaming media players" - a ceiling for one product of several
EACH_CAP = re.compile(rf"(?:keep|each|every|per)\b[^.;]{{0,50}}?(?:at or under|under|at most|max(?:imum)?|no more than|up to|tops out at|over)"
                      rf"\s*(?:a mere\s*)?{_MONEY}", re.I)
_VALUE_END = r"(?=\s*(?:[;!?(\u2013\u2014]|\.(?!\d)|,(?!\d)|\s-\s|\s(?:and|but|so|nothing|none|not|because|since|though|which|that's|for me|please)\b|$))"
FEATURE_TEMPLATES = [re.compile(p, re.I) for p in (
    # where the lock type is unkeyed / where the color is grey or titanium
    rf"where\s+the\s+(?P<prop>[a-z][a-z ]{{1,30}}?)\s+is\s+(?:a\s+type\s+that\s+)?(?P<value>(?:[^.;,()]|(?<=\d)[.,](?=\d)){{1,60}}?){_VALUE_END}",
    # ... where the X is Y and the color is grey or blue (the where-clause going on)
    rf"(?<=[a-z0-9])\s+and\s+the\s+(?P<prop>[a-z][a-z ]{{1,30}}?)\s+is\s+(?P<value>(?:[^.;,()?]|(?<=\d)[.,](?=\d)){{1,60}}?){_VALUE_END}",
    # the heel has to be a wedge / the material's got to say cotton / the style's got to be wrap / the input voltage has to read 5V
    rf"\bthe\s+(?P<prop>[a-z][a-z ]{{1,30}}?)(?:'s|\s+has|\s+must|\s+should|\s+needs?|\s+is|\s+ought)\s+(?:got\s+|gotta\s+)?(?:to\s+)?"
    rf"(?:be|say|read|come in|show|match)\s+(?:a\s+|an\s+)?(?P<value>(?:[^.;,()]|(?<=\d)[.,](?=\d)){{1,60}}?){_VALUE_END}",
    # with a pack size of at most 1 piece / with an output current of at most 3.33 a / with data transfer rate 480 mbps
    rf"\bwith\s+(?:a|an|the)?\s*(?P<prop>[a-z][a-z ]{{1,30}}?)\s+(?:of\s+)?(?P<value>(?:at (?:most|least)|no more than|up to|under|over)?\s*\d(?:[^.;,()]|(?<=\d)[.,](?=\d)){{0,40}}?){_VALUE_END}",
    # the size of which is 2'6" x 8'
    rf"\bthe\s+(?P<prop>[a-z][a-z ]{{1,30}}?)\s+of\s+which\s+is\s+(?P<value>(?:[^.;,()]|(?<=\d)[.,](?=\d)){{1,60}}?){_VALUE_END}",
    # running Android as the operating system
    rf"\brunning\s+(?P<value>[a-z0-9](?:[^.;,()]|(?<=\d)[.,](?=\d)){{0,30}}?)\s+as\s+the\s+(?P<prop>[a-z][a-z ]{{1,30}}?){_VALUE_END}",
    # the resolution has to land on 3440 x 1440
    rf"\bthe\s+(?P<prop>[a-z][a-z ]{{1,30}}?)(?:'s|\s+has|\s+must|\s+should|\s+needs?)\s+(?:got\s+|gotta\s+)?(?:to\s+)?(?:land on|sit at|be set at|hit|reach)\s+(?P<value>(?:[^.;,()]|(?<=\d)[.,](?=\d)){{1,40}}?){_VALUE_END}",
    # a top with long sleeves / a bomber jacket with a zipper closure / fall/winter clothing with a logo pattern on it
    rf"\b(?:want|wanting|after|hunting(?: for)?|need|looking for|find|grab|buy|get)\s+(?:me\s+)?(?:an?\s+|some\s+)?(?:[a-z/&'-]+\s+){{0,4}}?with\s+(?:an?\s+|the\s+)?(?P<value>[a-z0-9](?:[^.;,()]|(?<=\d)[.,](?=\d)){{1,40}}?){_VALUE_END}",
    # it's got to be sleeveless / they've got to be waterproof
    rf"\b(?:it|they)(?:'s|'ve|\s+has|\s+have)?\s+(?:got\s+|gotta\s+)?(?:to\s+)?be\s+(?!an?\s)(?P<value>[a-z0-9](?:[^.;,()]|(?<=\d)[.,](?=\d)){{1,30}}?){_VALUE_END}",
    # made of chiffon
    rf"\bmade\s+(?:of|from)\s+(?P<value>[a-z0-9%](?:[^.;,()?]|(?<=\d)[.,](?=\d)){{0,30}}?){_VALUE_END}",
    # jeans in a slim fit / it's got to come in cans
    rf"\b(?:come|comes)\s+in\s+(?P<value>[a-z][a-z -]{{1,30}}?){_VALUE_END}",
    # ... and a woofer size of at least 12 inches
    rf"\band\s+an?\s+(?P<prop>[a-z][a-z ]{{1,25}}?)\s+of\s+(?P<value>(?:at (?:most|least)|no more than|no less than|up to)\s*\d(?:[^.;,()?]|(?<=\d)[.,](?=\d)){{0,30}}?){_VALUE_END}")]
# what a value never is: the shopper's own asides
_VALUE_STOP = re.compile(r"^(?:something|anything|one|ones|here|there|it|in stock|available|on hand|sure|able|honest|fine|good|"
                         r"careful|clear|a pair|the one|my|your|our|this|that|and|or|but|so|the|then|also|"
                         r"compatible with (?:it|that|them|this|whatever))\b", re.I)
_BOUND_LEAD = re.compile(r"^(?:a?t? ?least|at most|no more than|no less than|up to|under|over|more than|less than|minimum|maximum)$", re.I)
_VALUE_TAIL = re.compile(r"\s+(?:on (?:the )?(?:tag|label|listing|it|them|front|box)|at all|either|too|as well|please|though|"
                         r"for (?:sure|me|her|him|them))$", re.I)
EXCLUDE_TEMPLATE = re.compile(r"(?:leave out|cross off|rule out|skip|drop|no|not|nothing|don't (?:want|show|hand) me)\s+(?:anything|whatever|any|the ones?)?\s*"
                              r"(?:with|that has|has|having|in|made of|made from)\s+(?:an?\s+|the\s+)?(?P<value>[a-z0-9](?:[^.;,()]|(?<=\d)[.,](?=\d)){1,40}?)"
                              r"(?=\s*(?:[.;!?,(]|\s-\s|$|\s(?:that's|is|are|and)\b))", re.I)
_PROP_STOP = re.compile(r"^(?:thing|one|item|price|cost|total|budget|whole lot|lot|rest|other|shopper|store|listing|page|request|"
                        r"order|bill|tag|label|catch|point|plan|idea|deal)$")


# The item and the shelf, as practice, qualifying and race-13 requests name them: "I'm after some UHT milk", "the thing
# is a hoodie or sweatshirt", "Mules are the thing", "Y'all got any tops made of chiffon?", "It's a clock I want", "I'll
# start with luxury & formal watches", "a watch in casual & fashion watches", "Fall/winter clothing is the category",
# "Got to find a monitor", "1 x door handles and locks where ...", "Find me area rugs - ...".
_ITEM_END = r"(?=\s*(?:[,.;:?!()\u2013\u2014]|\s-\s|\s+(?:and|that|which|where|with|in|for|made|from|whose|ones?|but|so|to|is|are|i|i'm|i'll)\b|$))"
ITEM_TEMPLATES = [re.compile(p, re.I) for p in (
    rf"(?P<shelf>[a-z][a-z &/'-]{{2,40}}?)\s+is\s+the\s+(?:category|shelf|department|section)\b",
    rf"\b(?:start(?:ing)? with|show me|find me|buy(?: me)?|go and buy|get me)\s+(?:some\s+|an?\s+|\d+\s*x?\s+)?(?P<shelf>[a-z][a-z &/'-]{{2,40}}?)(?:\s+item)?{_ITEM_END}",
    r"\b(?P<n>\d+)\s*x?\s+(?P<shelf>[a-z][a-z &/'-]{2,40}?)(?=\s+(?:where|whose|with|in|for|item|that|made|of|ones)\b|\s*[;,.?\u2013\u2014]|\s-\s|$)",
    rf"\bit's an?\s+(?P<item2>[a-z][a-z &/'-]{{2,30}}?)\s+i\s+(?:want|need|'m after)\b",
    rf"\b(?:the thing is|here's the item:|here's what i want:|it's gotta be|it's got to be|got any|i'm (?:after|wanting|hunting(?: for)?|looking for)|"
    rf"i (?:want|need)|hunting(?: for)?|after|got to find|need)\s+(?:some\s+|an?\s+|any\s+|one\s+|a new\s+)?(?P<item>[a-z][a-z &/'-]{{2,40}}?){_ITEM_END}",
    rf"\b(?P<item>[a-z][a-z &/'-]{{2,30}}?)\s+(?:are|is)\s+the\s+(?:thing|item)\b")]
_ITEM_STOP = re.compile(r"^(?:something|anything|one|ones|it|that|this|more|another|new|the|a|an|them|you|me|thing|things|fair|"
                        r"honest|first|class|whatever|pick|stuff|pair|pairs|piece|what|to|go|whichever|got|two|three|four|long|"
                        r"sure|careful|for|these|those|messages?|moves?|steps?|times?|calls?|minutes?|hours?|days?|weeks?|months?|"
                        r"years?|tool calls?|actions?|turns?)\b", re.I)


def item_by_code(text: str) -> tuple[str, str]:
    # (item, shelf) the request names by the races' own wordings; "" for each it does not.
    t = " ".join(str(text).replace("\u2019", "'").split())
    found: list[tuple[int, str, str]] = []
    for rx in ITEM_TEMPLATES:
        for m in rx.finditer(t):
            for key, val_text in m.groupdict().items():
                if not val_text or key == "n":
                    continue
                val_text = " ".join(val_text.split()).strip(" -'")
                if val_text and not _ITEM_STOP.match(val_text) and len(val_text.split()) <= 5:
                    found.append((m.start(key), "item" if key.startswith("item") else "shelf", val_text))
    found.sort(key=lambda x: (x[0], -len(x[2])))        # (the first the request says; the fuller of two at one place)
    shelf = next((v for _, k, v in found if k == "shelf"), "")
    item = next((v for _, k, v in found if k == "item"), "")
    item = re.sub(r"^(?:military|new)[- ]style\s+", "", item)
    return item or shelf, shelf


# ----------------------------------------------------------------------------- what the shopper wants bought
# Race 14 (90 requests, 54 of them for several products) names the products to buy amid a story full of other things:
# "Dot from the sailing club asked me to bring a pudding ... Could you fetch me 2 tins of canned fruit, each pack holding
# at least 4 pieces, and 1 jumpsuit or romper in a sexy style"; pages a value is read from ("the style is the same as
# whatever you land on from the page of the bath towels you'll find at Lenkata ..."); things the shopper owns ("has to fit
# the TV I already have, a ROR3320"); rules over the set ("The streaming media players has to work with whichever portable
# monitors you pick"). Only what the shopper asks to have fetched is bought: code masks the pages, the owned things and
# the set rules, then reads the products from the sentences that ask for them ("fetch me", "First, 1 x ...", "Then 2 ...",
# "; 3 x ...", ", plus 1 ..."), each with its count and its own words.
_FETCH_STRONG = r"(?:could|can|would)\s+you\s+(?:fetch|find|get|buy|grab|order|pick up)\s+(?:me|us)\b|\b(?:fetch|find|get|buy|grab|order)\s+(?:me|us)\b"
_FETCH_COUNTED = r"(?:i'?ll\s+have|i\s+want|i'?d\s+love|i'?ll\s+want|i\s+need|find\s+me|get\s+me)\s+(?=(?:\d+|one|two|three|four|five)\s+(?:x\s+)?[a-z])"
_FETCH_WEAK = r"\bfind\s+me\b|\bget\s+me\b|\bbuy\s+me\b|i'?d\s+love\s+an?\b|i\s+want\s+an?\b"
_WANTED = re.compile(r"^\W*(?:so\s+)?(?P<s>[A-Za-z][a-z][a-z&,' -]*?)\s+(?:with\b[^.]*?\s+)?(?:are|is)\s+what\s+i(?:'m\s+(?:after|wanting)|\s+want|\s+need)", re.I)
_HEAD = r"(?:first(?:ly)?|second(?:ly)?|third(?:ly)?|then|next|(?:the\s+)?last\s+one\s+is|last(?:ly)?|and\s+last|finally|and|plus|also|add)"
_NUM = r"(\d+|one|two|three|four|five)"
_NUMW = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
_STOP = (r"where|whose|with|in|that|that's|which|standing|item|items|at|for|from|each|running|rated|made|to|so|is|are|wants?|"
         r"please|then|plus|and\s+(?:\d|one|two|three)|or\s+\d|being|having|can|could|"
         r"the\s+(?:very\s+)?same|like|as|by|on|of\s+(?:at|the)|i|i'd|i'll|i'm|you|they|it")
_NOT_ITEM = re.compile(r"^(?:things?|bits?(?: of kit)?|items?|few|lot|help|your|whatever|something|anything|one|ones|it|them|those|these|that|"
                       r"this|me|us|more|new one)\b", re.I)
_CONTAINER = re.compile(r"^[a-z]+\s+of\s+(?=[a-z]+)", re.I)     # ('2 tins of canned fruit': the kind after 'of')


def _shelf_after(rest: str) -> str:
    m = re.match(rf"\s*(?P<s>[a-z][a-z/&'-]*(?:(?:\s+|,\s*)(?!(?:{_STOP})\b)[a-z&/'-][a-z/&'-]*){{0,7}})", rest, re.I)
    if not m:
        return ""
    s = m.group("s")
    if "," in s:
        lead_part, _, tail = s.partition(",")
        if not re.search(r"(?:&|\bor\b|\band\b)", tail):
            s = lead_part
    s = re.sub(r"\s+(?:&|or|and)$", "", s.strip(" ,"))
    return _CONTAINER.sub("", " ".join(s.split()))


def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.?!])\s+(?=[A-Z(])", text) if s.strip()]


def products_by_code(text: str) -> list[dict]:
    t = " ".join(str(text).replace("’", "'").split())
    masked = t
    sents = _sentences(masked)
    strong = any(re.search(_FETCH_STRONG, x, re.I) or re.search(_FETCH_COUNTED, x, re.I) for x in sents)
    out: list[dict] = []

    def add(n: int, rest: str, counted: bool = False) -> bool:
        shelf = _shelf_after(rest)
        if shelf and len(shelf) >= 3 and not _NOT_ITEM.match(shelf) and shelf.lower() not in {obj_item["item"].lower() for obj_item in out}:
            if out and out[-1]["clause"].endswith(rest) and len(out[-1]["clause"]) > len(rest):
                out[-1]["clause"] = out[-1]["clause"][: len(out[-1]["clause"]) - len(rest)]
            out.append({"item": shelf, "quantity": n, "clause": rest, "counted": counted})
            return True
        return False

    def count(c: str | None) -> int:
        return (int(c) if c.isdigit() else _NUMW[c.lower()]) if c else 1

    in_list = False
    for s in sents:
        fetch = re.search(_FETCH_STRONG, s, re.I) or re.search(_FETCH_COUNTED, s, re.I) or (None if strong else re.search(_FETCH_WEAK, s, re.I))
        lead_part = re.match(rf"\s*(?P<h>{_HEAD})[,]?\s+(?:{_NUM}(?:\s*x)?\s+|an?\s+)(?=[a-z])", s, re.I)
        if lead_part and (re.match(r"first", lead_part.group("h"), re.I) or in_list):
            in_list = True
            add(count(lead_part.group(2)), s[lead_part.end():], lead_part.group(2) is not None)
            body, base = s[lead_part.end():], lead_part.end()
        elif fetch:
            in_list = True
            body = re.sub(r"^\s*(?:is\s+|me\s+|us\s+)?", "", s[fetch.end():])
            lead = re.match(rf"\s*(?:{_NUM}(?:\s*x)?\s+|(?:an?|some|any)\s+)?(?=[a-z])", body, re.I)
            if re.match(r"\s*(?:those|them|these|it|one|that)\b", body, re.I):
                # ("Desktop computers with a fast processor are what I need, so fetch me those": the thing the sentence or
                # an earlier one names as wanted, else the subject that opens this sentence)
                prior = next((w for w in (_WANTED.match(x) for x in sents) if w), None) \
                    or re.match(r"^\W*(?P<s>[A-Za-z][a-z][a-z&,' -]*?)\s+(?:with\b[^.,]*?\s+)?(?:are|is)\b", s[: fetch.start()], re.I)
                if prior and not out:
                    add(1, prior.group("s"))
                continue
            if lead and not re.match(rf"\s*{_NUM}\s+(?:things|bits|items)\b", body, re.I):
                add(count(lead.group(1)), body[lead.end():], lead.group(1) is not None)
        else:
            w = _WANTED.match(s)
            if w and not out and not re.search(r"\b(?:it|this|that)\b", w.group("s"), re.I):
                add(1, w.group("s"))
            continue
        # further products in the same sentence: '; 3 x ...', ', and 1 ...', ', plus 1 ...', ' plus 2 ...'
        for m in re.finditer(rf"(?:;\s*(?:and\s+)?|,\s*(?:and\s+|plus\s+|then\s+)?|\bplus\s+)(?:{_NUM}(?:\s*x)?\s+|(?:an?)\s+)(?=[a-z])", body, re.I):
            if m.group(1) and re.match(r"\d+[-.]\d", body[m.start(1):]):
                continue
            if not m.group(1) and not re.search(r"plus|;", m.group(0), re.I):
                continue    # (', a ...' inside a clause is a detail; 'plus a ...' / '; a ...' start a product)
            add(count(m.group(1)), body[m.end():], m.group(1) is not None)
    for obj_item in out:
        obj_item["clause"] = re.sub(r"[\s,;]*(?:and|plus|then)?[\s,;]*$", "", obj_item["clause"].split("?")[0]).strip()
    return out


def request_by_code(text: str) -> dict:
    # The parts of a request code can read for certain: {"budget", "each_cap", "cheapest", "features": [(prop, value)],
    # "products": [(count, words)]}.
    t = " ".join(str(text).replace("\u2019", "'").split())
    # (a reference's own details - "the one with door thickness 1 3/8 in and color black" - describe the other listing,
    # not the item: they are read by refs_by_code, never as the item's features)
    whole = t
    t = t.replace("\u2014", "\u2013")
    out: dict[str, Any] = {"budget": None, "each_cap": None, "cheapest": bool(re.search(CHEAPEST_WORDS, low(t))), "features": [], "products": [],
                           "exclusions": [], "exclusion_props": {}, "wanted": products_by_code(whole)}
    out["item"], out["shelf"] = item_by_code(t)
    m = EACH_CAP.search(t)
    if m:
        out["each_cap"] = number(m.group(1).replace(",", ""))
    for pattern in TOTAL_CAP:
        for m in pattern.finditer(t):
            if m.group(0) == (EACH_CAP.search(t) or [None])[0]:
                continue
            if EACH_CAP.search(m.group(0)):
                continue
            if re.match(r"[^.;!?]{0,40}?\b(?:each|per|apiece|a piece|for every)\b", t[m.end():], re.I):
                continue        # ("no more than 2650 cents, that is on each player": a ceiling per item, not the total)
            out["budget"] = number(m.group(1).replace(",", ""))
            break
        if out["budget"] is not None:
            break
    for i, pattern in enumerate(FEATURE_TEMPLATES):
        for m in pattern.finditer(t):
            prop = m.groupdict().get("prop") or ""
            value = " ".join(m.group("value").split()).strip(" -")
            if re.match(r"(?:not|never|no)\s", low(value)):
                continue        # (a value ruled out: read as an exclusion below)
            if re.search(r"\bas\s+the\b", low(value)):
                continue        # ('with low sodium as the dietary need': read whole by its own template)
            prop = re.sub(r"^(?:the|its|their)\s+", "", " ".join(low(prop).split()))
            if prop and (len(prop.split()) > 4 or re.search(r"\b(?:ends?|goes|go|comes?|lives?|sits?|up|was|were|will|would|did|does|i|my|we)\b", prop)):
                continue        # (a clause, not a property: "where the charger ends up is still unsquared")
            if not value or (prop and _PROP_STOP.match(prop)) or len(value.split()) > 7:
                continue
            value = re.sub(r"^(?:a|an|the)\s+", "", value, flags=re.I)
            value = _VALUE_TAIL.sub("", value).strip()
            if not value or _VALUE_STOP.match(value) or (prop and _PROP_STOP.match(prop)):
                continue
            if _BOUND_LEAD.match(prop) or re.fullmatch(r"t least|t most", prop):
                # ('with at least 4 drawers': the bound was read as the property; the counted thing is the property)
                upper = re.search(r"most|no more|up to|under|less than|maximum", prop)
                value = f"{'at most' if upper else 'at least'} {value}"
                unit = re.search(r"\d\s*([a-z][a-z ]*)$", low(value))
                prop = unit.group(1).strip() if unit and len(unit.group(1).strip()) > 3 else ""
            if any(low(value) in low(v) or low(v) in low(value) for p, v in out["features"]):
                continue        # (the same value read by two templates)
            out["features"].append((prop, value))
    for m in EXCLUDE_TEMPLATE.finditer(t):
        value = _VALUE_TAIL.sub("", " ".join(m.group("value").split())).strip()
        if value and not _VALUE_STOP.match(value) and not re.search(r"\d+\s*(?:cts|cents)|stock|website|substitute", low(value)):
            out["exclusions"].append(value)
            out["features"] = [(p, v) for p, v in out["features"] if low(v) not in low(value)]
    for m in re.finditer(r"\b(\d+|one|two|three|four|a|an)\s*(?:x\s+)?([a-z][a-z &'-]{2,40}?)\s+(?:item\s+)?(?:where|in a|with|for a|that|made)\b", t, re.I):
        n = {"one": 1, "two": 2, "three": 3, "four": 4, "a": 1, "an": 1}.get(m.group(1).lower(), None) or int(m.group(1)) if m.group(1).isdigit() or m.group(1).lower() in ("one", "two", "three", "four") else None
        if n:
            out["products"].append((n, m.group(2).strip()))
    return out


def product_words(text: Any) -> set[str]:
    # a product's or shelf's words as stems, plurals folded, no joiners ("jumpsuits & rompers" ~ "jumpsuit or romper")
    out = set()
    for w in roots(spread(re.sub(r"[()&/,]", " ", str(text or "")))):
        if w in ("or", "and", "the", "kind", "item", "thing", "x"):
            continue
        out.add(re.sub(r"(?:ies|ie|y|es|s)$", "", w) if len(w) > 3 else w)
    return out


def fetch_text(request: str) -> str:
    # the sentences that ask for something to be fetched or bought (where the products to buy are named)
    t = " ".join(str(request).replace("\u2019", "'").split())
    sents = [x for x in re.split(r"(?<=[.?!])\s+(?=[A-Z(])", t) if x.strip()]
    return " ".join(x for x in sents if re.search(_FETCH_STRONG, x, re.I) or re.search(_FETCH_COUNTED, x, re.I)
                    or re.search(_FETCH_WEAK, x, re.I) or re.match(rf"\s*{_HEAD}[,]?\s+(?:\d+|an?\b)", x, re.I))


def code_product(w: dict) -> dict:
    # a product the request asks for that the reader left out, with what code reads in its own words
    own = request_by_code(w.get("clause") or "")
    need_items = []
    for prop, value in own.get("features") or []:
        either = [v.strip() for v in re.split(r"\s+or\s+", value) if v.strip()] if re.search(r"\sor\s", value) else []
        need_items.append({"property": prop or "", "value": either[0] if either else value, "either": either, "same_meaning": [],
                     "must": True, "by_code": True})
    return {"item": w["item"], "store_category": w["item"], "quantity": w["quantity"], "requirements": need_items, "by_code": True,
            "exclusions": [{"property": "", "value": v, "same_meaning": []} for v in own.get("exclusions") or []], "budget": 0}


def check_products(plan: dict, wanted: list[dict], request: str) -> tuple[dict, list[str]]:
    # The products to buy are those the request asks to have fetched (code reads them for sure from "fetch me 2 x ...;
    # 3 x ...", "First, 1 ... Then 2 ..."): one the reader left out is added, one it took from a page, a thing the shopper
    # owns or the story is dropped, and a count the request writes as a figure is the count.
    notes: list[str] = []
    plan = dict(plan)
    if not wanted:
        return plan, notes
    model = [dict(p) for p in listify(plan.get("products")) if isinstance(p, dict) and str(p.get("item") or "").strip()]
    if len(model) < 2:
        model = [{"item": plan.get("item") or "", "store_category": plan.get("store_category") or "", "quantity": plan.get("quantity") or 1,
                  "requirements": listify(plan.get("requirements")), "exclusions": listify(plan.get("exclusions")), "budget": 0, "top": True}]
    asked = product_words(fetch_text(request))

    def overlap(p: dict, w: dict) -> int:
        return len((product_words(p.get("item")) | product_words(p.get("store_category"))) & product_words(w["item"]))
    if len(wanted) >= 2:
        out, used = [], set()
        for w in wanted:
            best = max((i for i in range(len(model)) if i not in used), key=lambda i: overlap(model[i], w), default=None)
            if best is not None and overlap(model[best], w) > 0:
                used.add(best)
                p = dict(model[best])
                p.pop("top", None)
                if w.get("counted") and int(number(p.get("quantity")) or 1) != w["quantity"]:
                    notes.append(f"{w['item']}: {w['quantity']} as the request counts it (the reader had {p.get('quantity')})")
                    p["quantity"] = w["quantity"]
                if not str(p.get("store_category") or "").strip():
                    p["store_category"] = w["item"]
                out.append(p)
            else:
                notes.append(f"added {w['quantity']} x {w['item']} (named in the request, missing from the reading)")
                out.append(code_product(w))
        for i, p in enumerate(model):
            if i not in used and str(p.get("item") or "").strip():
                notes.append(f"dropped {p.get('item')!r}: not among the products the request asks to have fetched")
        plan["products"] = out
        first = out[0]
        plan.update({"item": first.get("item"), "store_category": first.get("store_category") or "",
                     "requirements": listify(first.get("requirements")), "exclusions": listify(first.get("exclusions")),
                     "quantity": first.get("quantity") or 1})
        return plan, notes
    w = wanted[0]
    if len(model) >= 2:
        keep = [p for p in model if overlap(p, w) > 0 or product_words(p.get("item")) <= asked]
        if len(keep) < len(model):
            notes.append(f"dropped {[p.get('item') for p in model if p not in keep]}: a page, an owned thing or the story, not a product to buy")
            plan["products"] = keep if len(keep) >= 2 else []
            if len(keep) == 1:
                p = keep[0]
                plan.update({"item": p.get("item"), "store_category": p.get("store_category") or "",
                             "requirements": listify(p.get("requirements")), "exclusions": listify(p.get("exclusions")),
                             "quantity": p.get("quantity") or 1})
        return plan, notes
    p = model[0]
    if not str(p.get("item") or "").strip():
        plan["item"] = w["item"]        # (one product: code fills only an item the reader left empty)
    if overlap(p, w) == 0 and str(p.get("item") or "").strip():
        return plan, notes
    if not str(plan.get("store_category") or "").strip():
        plan["store_category"] = w["item"]
    if w.get("counted") and int(number(plan.get("quantity")) or 1) != w["quantity"]:
        notes.append(f"quantity {w['quantity']} as the request counts it (the reader had {plan.get('quantity')})")
        plan["quantity"] = w["quantity"]
    return plan, notes


def merge_plan(plan: dict, code: dict, request: str) -> dict:
    # The model's reading, checked by code (race #7: request reading by model + code scored 0.178, model alone 0.138):
    # - the budget: the code's when the request words it as a limit, else the model's when that figure is in the request;
    # - "cheapest": either reader;
    # - a value the model read that no word of the request carries becomes a wish, not a requirement;
    # - a value code read from a fixed template that the model left out is added (one product only: with several, which
    #   product a value belongs to is the model's call).
    plan = dict(plan or {})
    plan, notes = check_products(plan, code.get("wanted") or [], request)
    if notes:
        plan["_notes"] = notes
    words_in = set(roots(spread(request), hold_filler=True))
    figure = number(plan.get("budget"))
    if code.get("budget"):
        plan["budget"] = code["budget"]
    elif figure and not re.search(rf"\b{re.escape(f'{figure:g}')}\b", request.replace(",", "")):
        plan["budget"] = 0
    plan["cheapest"] = bool(plan.get("cheapest")) or bool(code.get("cheapest"))
    if not str(plan.get("item") or "").strip() and code.get("item"):
        plan["item"] = code["item"]
    if not str(plan.get("store_category") or "").strip() and code.get("shelf"):
        plan["store_category"] = code["shelf"]
    need_items = [dict(r) for r in listify(plan.get("requirements")) if isinstance(r, dict)]
    for r in need_items:
        said = [w for w in roots(spread(r.get("value") or "")) if len(w) > 2]
        alts = [w for a in listify(r.get("same_meaning")) + listify(r.get("either")) for w in roots(spread(a))]
        if said and not r.get("from_listing") and not any(w in words_in for w in said + alts) and not re.search(r"\d", str(r.get("value"))):
            r["must"] = False
            r["unsaid"] = True
    # (the details the reader gives for a listing a value is read from describe that listing, not this item)
    ref_text = " ".join(" ".join(str(d) for d in listify((r.get("from_listing") or {}).get("details"))) for r in need_items
                        if isinstance(r, dict) and isinstance(r.get("from_listing"), dict))
    if len(listify(plan.get("products"))) < 2:
        for prop, value in code.get("features") or []:
            if ref_text and value_matches(ref_text, value):
                continue
            if any(value_matches(r.get("value"), value) or value_matches(value, r.get("value")) or
                   any(value_matches(a, value) for a in listify(r.get("same_meaning")) + listify(r.get("either"))) for r in need_items):
                continue
            if any(value_matches(x.get("value") if isinstance(x, dict) else x, value) for x in listify(plan.get("exclusions"))):
                continue
            either = [v.strip() for v in re.split(r"\s+or\s+", value) if v.strip()] if re.search(r"\sor\s", value) else []
            need_items.append({"property": prop or "", "value": either[0] if either else value, "either": either, "same_meaning": [],
                         "must": True, "by_code": True})
        ban = [x for x in listify(plan.get("exclusions")) if isinstance(x, dict)]
        for value in code.get("exclusions") or []:
            if not any(value_matches(x.get("value"), value) or value_matches(value, x.get("value")) for x in ban):
                ban.append({"property": (code.get("exclusion_props") or {}).get(value, ""), "value": value, "same_meaning": []})
        plan["exclusions"] = ban
    plan["requirements"] = need_items
    if code.get("each_cap") and not any(number(p.get("budget")) for p in listify(plan.get("products")) if isinstance(p, dict)):
        for p in listify(plan.get("products")):
            if isinstance(p, dict):
                p["budget"] = code["each_cap"]
    return plan


# ----------------------------------------------------------------------------- what fits what (code decides)
def code_matches(pattern: Any, code: Any) -> bool:
    # A family code covers its members: 'VIT-80xx' covers 'VIT-8030'; dashes and case do not count.
    p, c = re.sub(r"[^a-z0-9]", "", low(pattern)), re.sub(r"[^a-z0-9]", "", low(code))
    return bool(p) and len(p) == len(c) and all(a == b or (a == "x" and b.isdigit()) for a, b in zip(p, c))


def fits_model(text: Any, model: str) -> bool | None:
    # What a listing's own compatibility line says about one model: True, False, or None when it has no such line.
    # Race #13's grader also counts the listing's own "Part RO31209 also fits: VIR5020" sentence: 171 of 201 correct
    # compatible orders are explained with it, 164 without it.
    text = str(text or "")
    m = re.search(r"(?<!also )\b(?:compatible with|fits|works with|suits)\s*:\s*([^.]+)", text, re.I)
    if not m:
        return None
    parts = re.split(r"\bexcept\b|\bbut not\b|\bexcluding\b|\bother than\b", m.group(1), flags=re.I)
    allowed = MODEL_CODE.findall(parts[0]) + [c for surplus in re.findall(r"\bpart\s+[A-Z0-9-]+\s+also fits\s*:\s*([^.]+)", text, re.I)
                                              for c in MODEL_CODE.findall(surplus)]
    barred = [c for segment in parts[1:] for c in MODEL_CODE.findall(segment)]
    if any(code_matches(b, model) for b in barred):
        return False
    if any(code_matches(a, model) for a in allowed):
        return True
    return False if allowed else None


def model_of_text(text: Any) -> str:
    # The model code a listing names for itself ('Model: VIT-4530').
    m = re.search(r"\bmodel(?:\s*(?:no\.?|number|code))?\s*[:#]?\s*([A-Za-z]{2,5}-?\d{3,5})\b", str(text or ""), re.I)
    return m.group(1).upper() if m else ""


def compat_model(need_item: dict) -> str:
    # The model a requirement asks the item to fit ('compatible with: VIT-8030'); '' when it is no such requirement.
    m = re.search(r"\b([A-Za-z]{2,5}-?\d{3,5})\b", str(need_item.get("value") or ""))
    if m and (COMPAT_REQ.search(str(need_item.get("property") or "")) or COMPAT_REQ.search(str(need_item.get("value") or ""))):
        return m.group(1).upper()
    return ""


def form_schema(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {"type": "function", "function": {"name": name, "description": description,
                                             "parameters": {"type": "object", "properties": properties, "required": required}}}


def text_list(description: str) -> dict:
    return {"type": "array", "items": {"type": "string"}, "description": description}


# ----------------------------------------------------------------------------- model calls
# The Claude 5.5 models: thinking is always on (adaptive - it cannot be turned off, nor given a fixed budget), a
# non-default temperature is not taken, and Sonnet 5.5 refuses a request that forces one tool (a 400); prompts from 512
# tokens on can be cached (reads ~0.1x the input price, 5-minute writes 1.25x).
CLAUDE_55 = ("anthropic/claude-sonnet-5.5", "anthropic/claude-haiku-5.5")
NO_FORCED_TOOL = ("anthropic/claude-sonnet-5.5",)


def request_body(model: str, guide_text: str, text: str, tool: dict, max_tokens: int, think: bool) -> dict:
    # One reader call. For the Claude 5.5 models: room for the thinking that always runs (an answer cut short is a paid
    # call with no answer), the stage's fixed part - its tool and its guide, the same in every task - marked for the cache,
    # and on Sonnet 5.5 the tool asked for in words instead of forced. Other models get the plain request.
    name = tool["function"]["name"]
    if model not in CLAUDE_55:
        body = {"model": model, "temperature": 0, "max_tokens": max(max_tokens, TOKEN_FLOOR) + (REASONING_TOKENS if think else 0),
                "messages": [{"role": "system", "content": guide_text}, {"role": "user", "content": text}],
                "tools": [tool], "tool_choice": {"type": "function", "function": {"name": name}}}
        if think:
            body["reasoning"] = {"enabled": True}
        return body
    forced = model not in NO_FORCED_TOOL
    system = guide_text + ("" if forced else f"\n\nAnswer only by calling the {name} tool, once, with every required field filled.")
    return {"model": model, "max_tokens": max(max_tokens, TOKEN_FLOOR) + REASONING_TOKENS,
            "messages": [{"role": "system", "content": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]},
                         {"role": "user", "content": text}],
            "tools": [tool],
            "tool_choice": {"type": "function", "function": {"name": name}} if forced else "auto"}


class LlmGateway:
    # Language-model calls with a fallback model, a spending guard and a deadline per call.

    def __init__(self, started: float) -> None:
        self.started = started
        self.spent = 0.0
        self.calls = 0
        self.given_up = 0
        self.strikes: dict[str, int] = {}          # model -> failed calls in a row this episode
        self.secs: list[float] = []                # how long each model call took
        self.skipped = 0                           # calls not made in a hurry
        self.hurry = lambda: False                 # (set by the episode: little time left)
        self.used: dict[str, int] = {}             # model -> calls this episode
        self.last_refused = False

    def time_remaining(self) -> float:
        return TASK_TIME_S - (monotonic() - self.started)

    def _send(self, body: dict, limit: float = MODEL_WAIT_S) -> Any:
        bound = min(limit, self.time_remaining() - 25.0)
        if bound < 4.0:
            return None
        box: dict[str, Any] = {}

        def call() -> None:
            try:
                verbose = getattr(_proxy, "post_verbose", None)
                if verbose is None:
                    box["reply"] = _proxy.post("/inference/chat/completions", json_data=body)
                    return
                result = verbose("/inference/chat/completions", json_data=body)
                box["reply"] = getattr(result, "data", None)
                # (the proxy names a model this validator's provider does not serve: 403 "... is not allowed for provider")
                box["refused"] = box["reply"] is None and "not allowed" in json.dumps(getattr(result, "error", None), default=str).lower()
            except Exception as exc:  # a failed call is a missing answer, like a timeout
                box["error"] = type(exc).__name__
        worker = threading.Thread(target=call, daemon=True)
        worker.start()
        worker.join(bound)
        self.last_refused = bool(box.get("refused"))
        if worker.is_alive():
            self.given_up += 1
            return None
        return box.get("reply")

    def complete(self, guide_text: str, text: str, tool: dict, max_tokens: int = 1200, think: bool = False,
             limit: float = MODEL_WAIT_S, models: list[str] | None = None) -> dict | None:
        if self.spent >= COST_CEILING_USD:
            return None         # (the task's share of the run's capped key is spent: code finishes)
        name = tool["function"]["name"]
        if self.hurry() and name not in HURRY_KEEP:
            self.skipped += 1
            return None         # (the time left goes to finishing)
        if think and self.time_remaining() < THINK_UNTIL_S:
            think = False       # (a thinking pass is the slowest call: not with little time left)
        chain = list(models or STAGE_MODELS.get(name, DEFAULT_MODELS))
        if self.spent >= SPEND_SOFT_USD and LIGHT in chain and name not in SOFT_KEEP:
            chain = [LIGHT] + [m for m in chain if m not in (LIGHT, HEAVY)]
        for model in [m for m in chain if self.strikes.get(m, 0) < MODEL_STRIKES] or chain[-1:]:
            body = request_body(model, guide_text, text, tool, max_tokens, think)
            t0 = monotonic()
            reply = self._send(body, limit)
            self.secs.append(monotonic() - t0)
            self.calls += 1
            self.used[model] = self.used.get(model, 0) + 1
            if not isinstance(reply, dict):
                # (a name the provider refuses is dropped for the episode at once; any other failure counts one strike)
                self.strikes[model] = MODEL_STRIKES if self.last_refused else self.strikes.get(model, 0) + 1
                continue
            usage = reply.get("usage") or {}
            cost = usage.get("cost")
            pin, pout = PRICES.get(model, (2.0, 10.0))      # (unknown model: priced as the dearest)
            self.spent += float(cost) if isinstance(cost, (int, float)) else (
                int(usage.get("prompt_tokens") or 0) * pin + int(usage.get("completion_tokens") or 0) * pout) * 1e-6
            decoded = self._params_of(reply, name)
            if decoded is not None:
                self.strikes[model] = 0
                return decoded
        return None

    @staticmethod
    def _decode_entities(value: Any) -> Any:
        if isinstance(value, str):
            return html.unescape(value) if "&" in value else value
        if isinstance(value, list):
            return [LlmGateway._decode_entities(v) for v in value]
        if isinstance(value, dict):
            return {k: LlmGateway._decode_entities(v) for k, v in value.items()}
        return value

    @staticmethod
    def _params_of(reply: dict, name: str) -> dict | None:
        try:
            message = reply["choices"][0]["message"]
        except (KeyError, IndexError, TypeError):
            return None
        for call in message.get("tool_calls") or []:
            unparsed = (call.get("function") or {}).get("arguments")
            try:
                decoded = json.loads(unparsed) if isinstance(unparsed, str) else unparsed
            except (TypeError, ValueError):
                decoded = None
            if isinstance(decoded, dict):
                return LlmGateway._decode_entities(decoded)
        content = message.get("content")
        if isinstance(content, str) and "{" in content:
            try:
                decoded = json.loads(content[content.index("{"): content.rindex("}") + 1])
            except ValueError:
                return None
            if isinstance(decoded, dict):
                return LlmGateway._decode_entities(decoded.get(name) if isinstance(decoded.get(name), dict) else decoded)
        return None


# ----------------------------------------------------------------------------- the world's tools
ROLE_LIST = ("search", "filter", "facts", "view", "cart_add", "cart_remove", "cart_view", "order", "decline", "message")

ROLE_TERMS = {
    "search": r"\b(?:search\w*|find\w*|look ?up|lookup|query|browse|seek|hunt)\b",
    "facts": r"\b(?:propert\w*|traits?|facts?|attributes?|spec\w*|datasheet|details?|features?|sheet)\b",
    "view": r"\b(?:in full|view\w*|show\w*|open|listing|article|product page|read)\b",
    "cart_view": r"\b(?:cart|basket|bag|trolley)\b",
    "order": r"\b(?:purchase|check ?out|checkout|buy|pay|settle|place.*order|order)\b",
    "decline": r"\b(?:no purchase|without (?:a )?(?:purchas|buy)\w*|buys? nothing|give up|decline|abstain|nothing fits|close out|cannot|no order|withdraw|refuse|walk away|leave|stand ?down|stands? down)\b",
    "message": r"\b(?:shopper|customer|client|buyer|ask|question|message|write|note|chat|reply|tell)\b",
}
APPEND_TERMS = r"\b(?:add|append|attach|include|put|insert|plus|keep|place|increase|hold|reserve|in)\b"     # (race 14: SetToCart ActType append|delete)
DELETE_TERMS = r"\b(?:remove|discard|drop|delete|take|minus|decrease|out|clear|release|unhold|unreserve)\b"
# a finishing tool that both orders and declines tells which by an enum value
BUY_SETTING = r"\b(?:commit|buy|purchase|order|checkout|check ?out|confirm|accept|place|pay|complete|submit|proceed|yes)\b"
WALKAWAY_SETTING = r"\b(?:refuse|decline|abstain|cancel|none|nothing|give ?up|reject|skip|walk|no|abandon|without|pass)\b"
CLOSING_TERMS = r"final|no more actions|accepts no more|end(?:s)? the (?:task|episode|session|chat)|finish(?:es)? the|close(?:s)? (?:out|the)"
FINISH_TEXT = r"remark|reason|note|text|message|comment|why|explan|justif|rational|summar|detail|memo|because|account"
PAGE_PARAM = r"\b(?:page\w*|pg|p|offset|first\w*|start\w*|from|skip|token\w*|cursor|next|after|continu\w*|key|further|position|bookmark|resume\w*|onward)\b"


class Toolset:
    # Which world tool does what, and which argument carries what (read from names, descriptions and schemas).

    def __init__(self, tools: list[dict]) -> None:
        self.schemas: dict[str, dict] = {}
        for entry in tools or []:
            callback = entry.get("function") if isinstance(entry, dict) and "function" in entry else entry
            if isinstance(callback, dict) and callback.get("name"):
                self.schemas[callback["name"]] = callback
        self.role: dict[str, str] = {}             # role -> tool name
        self.arg: dict[tuple[str, str], str] = {}  # (role, slot) -> argument name
        self.value: dict[tuple[str, str], Any] = {}  # (role, slot) -> fixed value (an enum choice)
        self.count_drop = False                      # the cart tool takes an item out by setting its count to 0
        self._assign()

    # -- reading the schemas
    @staticmethod
    def _attrs(schema: dict) -> dict:
        return ((schema.get("parameters") or {}).get("properties") or {})

    @staticmethod
    def _prose(schema: dict) -> str:
        return low(f"{label_tokens(schema.get('name', ''))} {schema.get('description', '')}")

    def _locate_param(self, schema: dict, pattern: str, kind: str | None = None, forbid: set[str] = frozenset()) -> str:
        for name, prop in self._attrs(schema).items():
            if name in forbid:
                continue
            types = prop.get("type")
            types = types if isinstance(types, list) else [types]
            if kind == "untyped":
                if any(t for t in types):
                    continue
            elif kind and kind not in types and not (kind == "number" and "integer" in types):
                continue
            if re.search(pattern, label_tokens(name) + " " + low(prop.get("description"))):
                return name
        return ""

    @staticmethod
    def _switch_param(schema: dict) -> str:
        for name, prop in ((schema.get("parameters") or {}).get("properties") or {}).items():
            if len(prop.get("enum") or []) >= 2:
                return name
        return ""

    def _id_param(self, schema: dict) -> str:
        attrs = self._attrs(schema)
        found = self._locate_param(schema, r"\bid\b|ident|pid|sku|product|item|article|listing|target|chosen|which", "string")
        if found:
            return found
        text_items = [n for n, p in attrs.items() if p.get("type") == "string" and "enum" not in p]
        return text_items[0] if len(text_items) == 1 else ""

    def _assign(self) -> None:
        ratings: dict[str, dict[str, float]] = {}
        for name, schema in self.schemas.items():
            text = self._prose(schema)
            attrs = self._attrs(schema)
            required = (schema.get("parameters") or {}).get("required") or []
            s: dict[str, float] = {}
            for role, pattern in ROLE_TERMS.items():
                if re.search(pattern, text):
                    s[role] = 1.0
            choice_params = {n: [low(v) for v in p.get("enum") or []] for n, p in attrs.items() if p.get("enum")}
            holds_text = any(p.get("type") == "string" and "enum" not in p for p in attrs.values())
            text_params = [n for n, p in attrs.items() if p.get("type") == "string" and "enum" not in p
                         and re.search(r"text|query|keyword|term|phrase|free|words|needle|wanted|look|seek|\bq\b|search|for\b", label_tokens(n))]
            shelf_params = [n for n, p in attrs.items() if p.get("type") == "string"
                          and re.search(r"categ|depart|dept|shelf|section|aisle|class|group|collection|label|range", label_tokens(n))]
            if attrs and shelf_params and not text_params and not s.get("message"):
                s["filter"] = 3.5            # (lists one shelf: no words to search)
                s.pop("search", None)
            elif (s.get("search") or (text_params and not s.get("message"))) and attrs:
                s["search"] = 3.0
            final = bool(re.search(CLOSING_TERMS, text))
            mutual = [n for n, val_texts in choice_params.items()
                         if any(re.search(BUY_SETTING, v) for v in val_texts) and any(re.search(WALKAWAY_SETTING, v) for v in val_texts)]
            if mutual and (final or re.search(ROLE_TERMS["order"], text) or re.search(r"wrap|end|finish|close|conclude", text)):
                s["finish_both"] = 5.0      # (one tool orders or declines by an enum value)
            elif final:
                if re.search(ROLE_TERMS["decline"], text):
                    s["decline"] = 4.0
                elif re.search(ROLE_TERMS["order"], text):
                    s["order"] = 4.0
            if s.get("message") and holds_text and not s.get("search") and not final:
                s["message"] = 3.0
            buy_label = bool(re.search(ROLE_TERMS["order"], label_tokens(name))) and not re.search(ROLE_TERMS["decline"], label_tokens(name))
            if s.get("cart_view") and not attrs:
                s["cart_view"] = 0.0 if (final or buy_label) else 3.0     # (a "check out" tool that names the cart still orders)
            if buy_label and not attrs and not final:
                s["order"] = max(s.get("order", 0.0), 3.5)
            ids_param = [n for n, p in attrs.items() if p.get("type") == "array" and (p.get("items") or {}).get("type") == "string"
                       and re.search(r"\bids?\b|refs?|listing|product|item|article|code", label_tokens(n))]
            if ids_param and ids_param[0] in required and not final:
                s["facts_many"] = 4.0     # (several listings read whole in one call)
            if s.get("cart_view") and attrs and not final:
                both_sides = [v for val_texts in choice_params.values() for v in val_texts]
                if any(re.search(APPEND_TERMS, v) for v in both_sides) and any(re.search(DELETE_TERMS, v) for v in both_sides):
                    s["cart_add"] = s["cart_remove"] = 3.5
                elif re.search(APPEND_TERMS, text) and not re.search(DELETE_TERMS, text):
                    s["cart_add"] = 3.0
                elif re.search(DELETE_TERMS, text):
                    s["cart_remove"] = 3.0
                s["cart_view"] = 0.0
            count_param = self._locate_param(schema, r"qty|quant|how many|count|units|amount|number", "number")
            zero_allowed = count_param and (attrs.get(count_param) or {}).get("minimum") == 0
            if (s.get("cart_view") or re.search(r"\bcart\b|basket|bag|trolley|hold", text)) and (re.search(r"\bsets?\b", text) or zero_allowed) and count_param \
                    and self._id_param(schema) and not choice_params and not final:
                s["cart_add"] = max(s.get("cart_add", 0), 3.5)      # (a quantity setter: count 0 takes the item out)
                minimum = (attrs.get(count_param) or {}).get("minimum")
                if minimum is None or minimum <= 0:
                    s["cart_remove"] = max(s.get("cart_remove", 0), 3.5)
                    self.count_drop = True
                s["cart_view"] = 0.0
            if re.search(r"modify|change|update|edit", text) and choice_params and self._id_param(schema):
                both_sides = [v for val_texts in choice_params.values() for v in val_texts]
                if any(re.search(APPEND_TERMS, v) for v in both_sides):
                    s["cart_add"] = max(s.get("cart_add", 0), 3.0)
                    s["cart_remove"] = max(s.get("cart_remove", 0), 3.0)
            lines_param = [n for n, p in attrs.items() if p.get("type") == "array" and isinstance(p.get("items"), dict)
                         and p["items"].get("type") == "object" and any(re.search(r"id|listing|product|item|article", low(k)) for k in (p["items"].get("properties") or {}))]
            if lines_param and not final and not s.get("search"):
                s["cart_set"] = 4.0     # (the whole cart in one call: a list of lines, each a listing and a count)
            bare_id = len(attrs) >= 1 and len(required) <= 1 and self._id_param(schema) and len(attrs) <= 2
            if bare_id and not s.get("cart_add") and not final:
                if s.get("facts"):
                    s["facts"] = 3.0
                if s.get("view"):
                    s["view"] = 2.5
                if not s.get("facts") and not s.get("view"):
                    s["view"] = 1.5
            if re.search(ROLE_TERMS["decline"], label_tokens(name)):
                s.pop("order", None)        # (a tool named for ending without a purchase never orders: "stand down order")
                s["decline"] = max(s.get("decline", 0.0), 4.5 if not attrs or final else 1.0)
            ratings[name] = s
        taken: set[str] = set()
        both_sides = sorted(((rating.get("finish_both", 0.0), n) for n, rating in ratings.items()), reverse=True)
        if both_sides and both_sides[0][0] > 0:
            name = both_sides[0][1]
            self.role["order"] = self.role["decline"] = name
            taken.add(name)
            for arg, prop in self._attrs(self.schemas[name]).items():
                val_texts = [str(v) for v in prop.get("enum") or []]
                order = [v for v in val_texts if re.search(BUY_SETTING, low(v))]
                decline = [v for v in val_texts if re.search(WALKAWAY_SETTING, low(v)) and v not in order]
                if order and decline:
                    self.arg[("order", "mode")] = self.arg[("decline", "mode")] = arg
                    self.value[("order", "mode")], self.value[("decline", "mode")] = order[0], decline[0]
                    break
        for role in ("order", "decline", "search", "filter", "cart_set", "cart_add", "cart_remove", "cart_view", "message", "facts_many", "facts", "view"):
            best = sorted(((rating.get(role, 0.0), n) for n, rating in ratings.items() if n not in taken or role in ("cart_remove",)),
                          reverse=True)
            if role in self.role:
                continue
            if best and best[0][0] >= 1.0:
                self.role[role] = best[0][1]
                if role not in ("cart_add",):
                    taken.add(best[0][1])
        # (a tool that reports the account - messages left, steps used: no required argument, no other role)
        for name, schema in self.schemas.items():
            if name in self.role.values() or ((schema.get("parameters") or {}).get("required") or []):
                continue
            if re.search(r"\b(?:account|allowance|report|quota|usage|remaining|balance|budget|limits?|status)\b", self._prose(schema)) \
                    and not re.search(ROLE_TERMS["cart_view"] + "|" + ROLE_TERMS["order"], self._prose(schema)):
                self.role.setdefault("report", name)
        if "view" not in self.role and "facts" in self.role:
            self.role["view"] = self.role["facts"]
        if "facts" not in self.role and "view" in self.role:
            self.role["facts"] = self.role["view"]
        self._assign_params()

    def _assign_params(self) -> None:
        search = self.schemas.get(self.role.get("search", ""))
        if search:
            text = self._locate_param(search, r"text|query|keyword|term|phrase|free|words|needle|wanted|look|seek|\bq\b|search|for\b", "string")
            category = self._locate_param(search, r"categ|depart|dept|shelf|section|aisle|class|group|collection|label|range", "string", {text})
            if not text:
                # (the one free string left that is not the shelf, the brand or a page bookmark is the search text)
                loose = [n for n, p in self._attrs(search).items() if p.get("type") == "string" and "enum" not in p and n != category
                         and not re.search(r"brand|maker|page|bookmark|resume|token|cursor|after", low(n))]
                text = loose[0] if len(loose) == 1 else ""
            brand = self._locate_param(search, r"brand|maker|manufact|company|vendor|marque|make\b", "string", {text, category})
            cap = self._locate_param(search, r"price|cap|budget|max|spend|ceiling|limit|cost|upto|under", "number")
            page = self._locate_param(search, PAGE_PARAM, None, {text, category, brand, cap, self._switch_param(search)})
            for field_slot, arg in (("text", text), ("category", category), ("brand", brand), ("cap", cap), ("page", page)):
                if arg:
                    self.arg[("search", field_slot)] = arg
            for name, prop in self._attrs(search).items():
                enum = [str(v) for v in prop.get("enum") or []]
                if len(enum) >= 2:
                    typed = [v for v in enum if re.search(r"typ|key|text|word|query|free|full|search|term|phrase", low(v))]
                    labelled = [v for v in enum if re.search(r"field|filter|facet|attr|struct|param|catal|categ|browse|shelf|aisle|dept|depart", low(v))]
                    if typed or labelled:
                        self.arg[("search", "mode")] = name
                        self.value[("search", "typed")] = typed[0] if typed else [v for v in enum if v not in labelled][0]
                        self.value[("search", "fields")] = labelled[0] if labelled else [v for v in enum if v not in typed][0]
        for role in ("facts", "view", "cart_add", "cart_remove", "order"):
            schema = self.schemas.get(self.role.get(role, ""))
            if schema:
                arg = self._id_param(schema) if role != "order" else self._locate_param(
                    schema, r"\bid\b|ident|pid|sku|product|item|article|listing|target|chosen|which", "string")
                if arg:
                    self.arg[(role, "id")] = arg
                qty = (self._locate_param(schema, COUNT_TERMS, "number")
                       or self._locate_param(schema, COUNT_TERMS, "untyped", forbid={arg} if arg else frozenset()))
                if qty:
                    self.arg[(role, "qty")] = qty
        for role in ("cart_add", "cart_remove"):
            schema = self.schemas.get(self.role.get(role, ""))
            if not schema:
                continue
            for name, prop in self._attrs(schema).items():
                enum = [str(v) for v in prop.get("enum") or []]
                if not enum:
                    continue
                pattern = APPEND_TERMS if role == "cart_add" else DELETE_TERMS
                match_found = [v for v in enum if re.search(pattern, low(v))]
                if match_found:
                    self.arg[(role, "op")] = name
                    self.value[(role, "op")] = match_found[0]
        as_float = self.schemas.get(self.role.get("filter", ""))
        if as_float:
            shelf = self._locate_param(as_float, r"categ|depart|dept|shelf|section|aisle|class|group|collection|label|range", "string")
            if shelf:
                self.arg[("filter", "category")] = shelf
            cap = self._locate_param(as_float, r"price|cap|budget|max|spend|ceiling|limit|cost|upto|under", "number")
            if cap:
                self.arg[("filter", "cap")] = cap
            page = self._locate_param(as_float, PAGE_PARAM, None, {shelf, cap})
            if page:
                self.arg[("filter", "page")] = page
            maker = self._locate_param(as_float, r"brand|maker|manufact|company|vendor|marque|make\b", "string", {shelf})
            if maker:
                self.arg[("filter", "brand")] = maker
        shelf_list = self.schemas.get(self.role.get("cart_set", ""))
        if shelf_list:
            for name, prop in self._attrs(shelf_list).items():
                items = (prop.get("items") or {}).get("properties") or {} if prop.get("type") == "array" else {}
                unsure = next((k for k in items if re.search(r"id|listing|product|item|article", low(k)) and items[k].get("type") in ("string", None)), "")
                count_key = next((k for k in items if items[k].get("type") in ("integer", "number")), "")
                if unsure:
                    self.arg[("cart_set", "lines")], self.arg[("cart_set", "line_id")] = name, unsure
                    if count_key:
                        self.arg[("cart_set", "line_qty")] = count_key
                    break
        many = self.schemas.get(self.role.get("facts_many", ""))
        if many:
            arg = next((n for n, p in self._attrs(many).items() if p.get("type") == "array"), "")
            if arg:
                self.arg[("facts_many", "ids")] = arg
        note_text = self.schemas.get(self.role.get("message", ""))
        if note_text:
            arg = self._locate_param(note_text, r"text|body|message|content|note|question|say|words", "string")
            attrs = [n for n, p in self._attrs(note_text).items() if p.get("type") == "string"]
            self.arg[("message", "text")] = arg or (attrs[0] if attrs else "")
        # (a finishing tool's free text: race #13's conclude_request REQUIRED a 'justification' - unread, no order went out)
        for role in ("decline", "order"):
            schema = self.schemas.get(self.role.get(role, ""))
            if schema:
                arg = self._locate_param(schema, FINISH_TEXT, "string")
                if not arg:
                    loose = [n for n, p in self._attrs(schema).items() if p.get("type") == "string" and not p.get("enum")
                             and n != self.arg.get((role, "id")) and n != self.arg.get((role, "mode"))]
                    arg = loose[0] if len(loose) == 1 else ""
                if arg:
                    self.arg[(role, "text")] = arg

    def adopt_role_map(self, role_map: dict) -> None:
        # A reader's mapping fills what the schemas left open.
        for role, key in (("order", "order_value"), ("decline", "decline_value"), ("cart_add", "add_value"), ("cart_remove", "remove_value")):
            value, arg = role_map.get(key), role_map.get(key.replace("value", "arg"))
            name = role_map.get(role) or self.role.get(role)
            if isinstance(value, str) and value and isinstance(arg, str) and name in self.schemas and arg in self._attrs(self.schemas[name]):
                self.role.setdefault(role, name)
                field_slot = "mode" if role in ("order", "decline") else "op"
                self.arg.setdefault((role, field_slot), arg)
                self.value.setdefault((role, field_slot), value)
        # (the schemas left the finishing or search roles in doubt, so the reader's reading of them wins: a tool named
        # "StandDownOrder" is no order tool, a "check out" tool that mentions the cart is no cart view)
        changed = False
        for role in ("order", "decline", "search"):
            name = role_map.get(role)
            if isinstance(name, str) and name in self.schemas and self.role.get(role) != name and not \
                    (self.role.get("order") == self.role.get("decline") and ("order", "mode") in self.arg and role != "search"):
                for other in [r for r, n in self.role.items() if n == name and r in ("order", "decline", "cart_view", "search", "filter")]:
                    if other != role and not {role, other} <= {"order", "decline"}:
                        del self.role[other]
                if role in ("order", "decline") and self.role.get("order" if role == "decline" else "decline") == name:
                    del self.role["order" if role == "decline" else "decline"]
                self.role[role] = name
                changed = True
        if changed:
            for key in [k for k in self.arg if k[0] in ("order", "decline", "search", "filter", "cart_view")]:
                del self.arg[key]
            self._assign_params()
        for role in ROLE_LIST:
            name = role_map.get(role)
            if isinstance(name, str) and name in self.schemas and role not in self.role:
                self.role[role] = name
        for key, slot_kind in (("search_text_arg", ("search", "text")), ("search_category_arg", ("search", "category")),
                               ("search_price_arg", ("search", "cap")), ("search_page_arg", ("search", "page")),
                               ("id_arg", ("cart_add", "id")), ("message_text_arg", ("message", "text"))):
            value = role_map.get(key)
            if isinstance(value, str) and value and slot_kind not in self.arg:
                self.arg[slot_kind] = value

    def wants_llm(self) -> bool:
        if "cart_set" in self.role and ("cart_set", "lines") in self.arg:
            return not ("search" in self.role and "order" in self.role and "decline" in self.role and ("search", "text") in self.arg)
        same = self.role.get("order") and self.role.get("order") == self.role.get("decline")
        return not ("search" in self.role and "order" in self.role and "decline" in self.role
                    and (not same or ("order", "mode") in self.arg)
                    and ("cart_add" in self.role or ("order", "id") in self.arg) and ("search", "text") in self.arg
                    and ("cart_add" not in self.role or "cart_remove" in self.role))

    # -- building calls
    def has(self, role: str) -> bool:
        return role in self.role

    def call(self, role: str, **slots: Any) -> dict | None:
        name = self.role.get(role)
        if not name:
            return None
        schema = self.schemas[name]
        attrs = self._attrs(schema)
        args: dict[str, Any] = {}
        if role == "search":
            mode = slots.pop("mode", "typed")
            if ("search", "mode") in self.arg:
                args[self.arg[("search", "mode")]] = self.value[("search", mode)]
        if role == "cart_remove" and self.count_drop and self.role.get("cart_remove") == self.role.get("cart_add"):
            arg = self.arg.get(("cart_add", "qty"))
            slots = {k: v for k, v in slots.items() if k != "qty"}
            if arg:
                args[arg] = 0
        if role in ("order", "decline") and (role, "mode") in self.arg:
            args[self.arg[(role, "mode")]] = self.value[(role, "mode")]
        if role in ("cart_add", "cart_remove") and (role, "op") in self.arg:
            args[self.arg[(role, "op")]] = self.value[(role, "op")]
        for field_slot, value in slots.items():
            if value is None:
                continue
            arg = self.arg.get((role, field_slot))
            if not arg and field_slot in ("id", "qty") and role == "cart_remove" and self.role.get("cart_remove") == self.role.get("cart_add"):
                arg = self.arg.get(("cart_add", field_slot), "")
            if not arg:
                continue
            prop = attrs.get(arg) or {}
            types = prop.get("type")
            types = types if isinstance(types, list) else [types]
            if "integer" in types:
                value = int(round(float(value)))
            elif "number" in types:
                value = float(value)
                value = int(value) if value.is_integer() else value
            elif "string" in types:
                value = str(value)[: int(prop.get("maxLength") or 400)]
            if "maximum" in prop and isinstance(value, (int, float)):
                value = min(value, prop["maximum"])
            if "minimum" in prop and isinstance(value, (int, float)):
                value = max(value, prop["minimum"])
            args[arg] = value
        count_param = self.arg.get((role, "qty")) or (self.arg.get(("cart_add", "qty")) if role in ("cart_add", "order") else None)
        if count_param and count_param not in args and count_param in ((schema.get("parameters") or {}).get("required") or []):
            args[count_param] = 1        # (a count the tool requires: one unless asked otherwise)
        for need_item in (schema.get("parameters") or {}).get("required") or []:
            if need_item not in args:
                prop = attrs.get(need_item) or {}
                types = prop.get("type") if isinstance(prop.get("type"), list) else [prop.get("type")]
                if prop.get("enum"):
                    args[need_item] = prop["enum"][0]
                elif slots.get("text") and ("string" in types or None in types):
                    # (a required free text the map did not place: the call's own text goes there rather than no call)
                    args[need_item] = str(slots["text"])[: int(prop.get("maxLength") or 600)]
                else:
                    return None
        return {"name": name, "args": args}

    def put_basket_call(self, lines: list[tuple[str, int]]) -> dict | None:
        # The call that makes the cart hold exactly these lines (a world whose cart is set as a whole).
        name = self.role.get("cart_set")
        arg, unsure = self.arg.get(("cart_set", "lines")), self.arg.get(("cart_set", "line_id"))
        if not (name and arg and unsure):
            return None
        count_key = self.arg.get(("cart_set", "line_qty"))
        return {"name": name, "args": {arg: [{unsure: eid, **({count_key: qty} if count_key else {})} for eid, qty in lines]}}

    def text_cap(self) -> int:
        schema = self.schemas.get(self.role.get("search", ""))
        arg = self.arg.get(("search", "text"))
        if schema and arg:
            return int((self._attrs(schema).get(arg) or {}).get("maxLength") or 120)
        return 120

    def group_cap(self) -> int:
        # How many listings one call of the several-listings tool reads (in race #5 a call naming 7 or more came back
        # empty: 5 unless the schema says fewer).
        schema = self.schemas.get(self.role.get("facts_many", ""))
        arg = self.arg.get(("facts_many", "ids"))
        if schema and arg:
            return max(1, min(5, int((self._attrs(schema).get(arg) or {}).get("maxItems") or 5)))
        return 0

    def page_cap(self) -> int:
        schema = self.schemas.get(self.role.get("search", ""))
        arg = self.arg.get(("search", "page"))
        if schema and arg:
            return int((self._attrs(schema).get(arg) or {}).get("maximum") or 6)
        return 1

    def summarize(self) -> dict:
        return {"roles": dict(self.role), "args": {f"{r}.{s}": a for (r, s), a in self.arg.items()},
                "values": {f"{r}.{s}": v for (r, s), v in self.value.items()}}


ROLE_FORM = form_schema("map_tools", "Role of every shop tool.", {
    **{role: {"type": "string", "description": desc} for role, desc in (
        ("search", "looks up catalogue entries from typed words"),
        ("filter", "browses entries by shelf or price ceiling (leave empty when the word lookup covers this)"),
        ("facts", "gives a single entry's attributes / fact sheet"),
        ("view", "opens a single entry completely (price, stock)"),
        ("cart_add", "adds an entry to the basket"),
        ("cart_remove", "removes an entry from the basket"),
        ("cart_view", "displays the basket"),
        ("order", "buys what is in the basket / checkout (ends the task)"),
        ("decline", "closes with nothing bought (ends the task)"),
        ("message", "writes to the buyer (a note or a question)"))},
    "search_text_arg": {"type": "string"}, "search_category_arg": {"type": "string"},
    "search_price_arg": {"type": "string"}, "search_page_arg": {"type": "string"},
    "id_arg": {"type": "string", "description": "basket-tool parameter carrying the entry's id"},
    "message_text_arg": {"type": "string"},
    "order_arg": {"type": "string", "description": "if a single tool can both buy and walk away: the parameter choosing between them"},
    "order_value": {"type": "string", "description": "its setting for buying"},
    "decline_arg": {"type": "string", "description": "that same parameter (walking away)"},
    "decline_value": {"type": "string", "description": "its setting for walking away"},
    "add_arg": {"type": "string", "description": "if a single basket tool can both add and remove: the parameter choosing between them"},
    "add_value": {"type": "string"}, "remove_arg": {"type": "string"}, "remove_value": {"type": "string"},
}, ["search", "order", "decline"])


# ----------------------------------------------------------------------------- reading observations
PROSE_RE = re.compile(r"[a-z].*\s.*\s")


def is_prose_list(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(isinstance(v, str) and PROSE_RE.search(low(v)) for v in value)


def split_alerts(obs_data: Any) -> tuple[Any, list[str], list[str]]:
    # The tool's own result, the shopper's notices that came with it, and a refusal of a finish (runtime level).
    notices: list[str] = []
    refusal: list[str] = []
    for _ in range(4):
        if not isinstance(obs_data, dict):
            break
        if "refusal" in obs_data and set(obs_data) <= {"refusal", "observation"}:
            r = obs_data.get("refusal")
            refusal += [str(x) for x in (r if isinstance(r, list) else [r]) if x]
            obs_data = obs_data.get("observation")
            continue
        if len(obs_data) == 2:
            keys = list(obs_data)
            a, b = obs_data[keys[0]], obs_data[keys[1]]
            if is_prose_list(b) and not is_prose_list(a):
                notices += b
                obs_data = a
                continue
            if is_prose_list(a) and not is_prose_list(b):
                notices += a
                obs_data = b
                continue
        break
    return obs_data, notices, refusal


ID_LABELS = r"^(?:id|pid|item_?pid|product_?id|sku|listing_?id|article_?id|item_?id|uid|code)$|_id$|\bid\b|pid"


def entry_of(d: dict) -> dict | None:
    # A product row in any world's field names: id, title, price, in_stock, category, brand, description, facts.
    if not isinstance(d, dict):
        return None
    out: dict[str, Any] = {}
    for k, v in d.items():
        label_low = low(k.replace("_", " ").replace(".", " "))
        if "id" not in out and isinstance(v, str) and (re.search(ID_LABELS, low(k)) or re.search(r"\b(?:id|pid|sku|code)\b", label_tokens(k)) and re.search(r"\bid\b|pid|sku|product|item|listing", label_tokens(k))) and len(v) <= 64:
            out["id"] = v
        elif "title" not in out and isinstance(v, str) and re.search(r"title|name|headline|heading|label|caption", label_low) and not re.search(r"categ|brand|aisle|shelf|depart|shop|seller|store|vendor", label_low):
            out["title"] = v
        elif "price" not in out and isinstance(v, (int, float)) and not isinstance(v, bool) and re.search(r"price|cost|amount|cts|cents", label_low):
            out["price"] = float(v)
        elif "in_stock" not in out and isinstance(v, bool) and re.search(r"stock|avail|on hand|sold|have|can buy|buyable|purchasable|orderable|in store", label_low):
            out["in_stock"] = (not v) if re.search(r"sold|out of", label_low) else v
        elif "category" not in out and isinstance(v, str) and re.search(r"categ|depart|shelf|section|aisle|class", label_low) \
                and not re.search(r"remark|note|desc|about|detail", label_low):
            out["category"] = v
        elif "seller" not in out and isinstance(v, str) and re.search(r"seller|merchant|vendor|sold by|\bshop\b|\bstore\b", label_low) \
                and not re.search(r"note|text|desc|comment|about", label_low):
            out["seller"] = v
        elif "brand" not in out and (isinstance(v, str) or v is None) and re.search(r"brand|maker|\bmake\b|company|manufact", label_low):
            out["brand"] = v
        elif isinstance(v, str) and re.search(r"desc|about|detail|summary|overview|good to know|notes?\b|remark", label_low):
            out["description"] = (out.get("description", "") + " " + v).strip()      # (several text fields: one description)
        elif "facts" not in out and isinstance(v, dict) and v and all(isinstance(x, (str, int, float, bool, type(None))) for x in v.values()):
            out["facts"] = v
    if "id" not in out:
        return None
    if "title" not in out:
        lengthiest = sorted((v for k, v in d.items() if isinstance(v, str) and v != out["id"]), key=len)
        if lengthiest:
            out["title"] = lengthiest[-1][:200]
    return out


def entries_in(obs_data: Any) -> list[dict]:
    # Every product row an observation holds (a search page, one listing, a cart).
    found: list[dict] = []
    if isinstance(obs_data, dict):
        ent_row = entry_of(obs_data)
        if ent_row and ("title" in ent_row or "facts" in ent_row or "price" in ent_row):
            found.append(ent_row)
        for v in obs_data.values():
            if isinstance(v, list):
                for item in v:
                    if isinstance(item, dict):
                        r = entry_of(item)
                        if r:
                            found.append(r)
    elif isinstance(obs_data, list):
        for item in obs_data:
            r = entry_of(item) if isinstance(item, dict) else None
            if r:
                found.append(r)
    return found


def sum_of(obs_data: Any) -> int | None:
    if isinstance(obs_data, dict):
        for k, v in obs_data.items():
            if isinstance(v, int) and not isinstance(v, bool) and re.search(r"match|total|count|hits|found|results|number|tally", low(k)):
                return v
    return None


def following_page_of(obs_data: Any) -> Any:
    if isinstance(obs_data, dict):
        for k, v in obs_data.items():
            if re.search(r"next|continu|more|cursor|further|token|after|onward|bookmark|resume", label_tokens(k)) and not isinstance(v, (dict, list)):
                return v if v not in (False, "", 0) or isinstance(v, int) and not isinstance(v, bool) and v else None
    return None


# ----------------------------------------------------------------------------- the reader's tools and guides
ALT_FORMS = ("Alternative wordings a shop entry could print for precisely this value: a synonym, a variant spelling, a hyphenated, "
                "spaced or fused form, and the identical quantity converted exactly into every other unit shops typically print for it "
                "(a bound carries over: 'at least ...' remains 'at least ...'). Nothing wider, tighter or only loosely connected. "
                "Leave empty if no such wording exists.")

ENTRY_POINTER = {
    "type": "object", "description": "Use solely when the request gives no value itself and points instead to a different shop entry, identified by type and vendor: "
                                     "describe that entry here (and keep value blank). Omit in every other case.",
    "properties": {"item": {"type": "string", "description": "What type of goods that entry is."},
                   "seller": {"type": "string", "description": "Its vendor, verbatim."},
                   "details": text_list("Distinguishing details the request gives for it, each written 'property value'."),
                   "relation": {"type": "string", "enum": ["same", "pairs", "not", "within"],
                                "description": "same: the value must equal that listing's; pairs: the value that listing's value is paired with by a pairing rule the store states; not: anything but that listing's value; within: that listing's figure, give or take the tolerance."},
                   "tolerance": {"type": "number", "description": "With within: the give-or-take figure (0.51 for 'within 0.51 inches of')."},
                   "unit": {"type": "string", "description": "With within: the tolerance's unit."}},
    "required": ["item", "seller"]}

NEED_SCHEMA = {
    "type": "object", "properties": {
        "property": {"type": "string", "description": "Everyday name of the attribute, in a buyer's words."},
        "value": {"type": "string", "description": "Shortest wording of the needed value, unit and bound included ('no more than 2 kg' becomes '<= 2 kg', written at most)."},
        "same_meaning": text_list(ALT_FORMS),
        "either": text_list("If any one of a set of values is acceptable: list them all; otherwise leave empty."),
        "must": {"type": "boolean", "description": "True if the buyer demands it of the item; false if it is merely backstory, a hope or scene-setting."},
        "bound": {"type": "string", "enum": ["exact", "at least", "at most", "more than", "less than", "between", "none"],
                  "description": "For a numeric value: exact (precisely that number, never rounded), at least ('or more', 'no less than', 'minimum'), at most ('up to', 'no more than', 'no taller than'), more than / less than (the figure itself excluded: 'more than', 'under'), between (two figures, both included); none for a value without a figure."},
        "low": {"type": "number", "description": "The figure (the lower one for between)."},
        "high": {"type": "number", "description": "The upper figure, for between."},
        "unit": {"type": "string", "description": "Unit of that number exactly as the buyer typed it; blank if absent."},
        "from_listing": ENTRY_POINTER},
    "required": ["property", "value", "same_meaning", "must"]}

BAN_SCHEMA = {"type": "object", "properties": {"property": {"type": "string"}, "value": {"type": "string"}, "same_meaning": text_list(ALT_FORMS)},
             "required": ["property", "value"]}

BRIEF_FORM = form_schema("plan", "A buyer's request, turned into a shopping brief.", {
    "item": {"type": "string", "description": "Broad type of goods wanted, worded as the request words it, with no conditions attached."},
    "store_category": {"type": "string", "description": "Shop category named by the request, copied verbatim; blank if none is named."},
    "requirements": {"type": "array", "description": "Each attribute the request demands of the goods themselves (budget, stock and category are separate).",
                     "items": NEED_SCHEMA},
    "exclusions": {"type": "array", "description": "Values the buyer forbids.", "items": BAN_SCHEMA},
    "products": {"type": "array", "description": "Fill solely if a single order must contain at least two DISTINCT kinds of goods: one entry each, following the request's sequence. Leave empty for a single kind.", "items": {
        "type": "object", "properties": {
            "item": {"type": "string", "description": "Type of these goods, worded as in the request."},
            "store_category": {"type": "string", "description": "Shop category given for these goods; blank if none."},
            "quantity": {"type": "integer", "description": "Unit count for these goods (default 1)."},
            "budget": {"type": "number", "description": "Spending limit applying to these goods only (per piece by default); 0 if absent."},
            "requirements": {"type": "array", "items": NEED_SCHEMA},
            "exclusions": {"type": "array", "items": BAN_SCHEMA}},
        "required": ["item", "quantity", "requirements"]}},
    "budget": {"type": "number", "description": "Overall spending ceiling as a bare number, in the currency unit the request uses; 0 if absent."},
    "cheapest": {"type": "boolean", "description": "True if the buyer wants the lowest-priced qualifying entry."},
    "best_by": {"type": "string", "description": "If the buyer ranks by a yardstick other than price (largest amount, lightest, longest, highest within budget): that yardstick, named as shop entries name the attribute; for value for money by a yardstick ('price per <unit>'), that phrase; otherwise blank."},
    "best_direction": {"type": "string", "enum": ["most", "least", ""], "description": "With best_by: most (the highest / largest / longest / most) or least (the lowest / lightest / smallest / thinnest)."},
    "in_stock": {"type": "boolean", "description": "True if only goods currently available / in stock are acceptable."},
    "quantity": {"type": "integer", "description": "Unit count to buy (default 1)."},
    "message_budget": {"type": "integer", "description": "Total number of messages the buyer allows, per the request; -1 if unstated."},
    "step_budget": {"type": "integer", "description": "Allowed count of moves / tool calls / actions; -1 if unstated."},
    "decline_rule": {"type": "string", "description": "Instruction for the case where no entry qualifies; blank if not given."},
    "queries": text_list("Four brief lookup phrases (2-4 words apiece) combining the goods with their demanded values."),
    "ask_properties": text_list("Attributes of such goods that the request leaves open and the buyer might be keeping back, most probable first, each a plain attribute name."),
    "closest_fit": {"type": "boolean", "description": "True if, failing an exact match, the buyer wants to be told so and offered (or sold) the nearest entry, the one missing the fewest points, rather than leaving empty-handed. False if the buyer forbids substitutes."},
    "over_pct": {"type": "number", "description": "Alongside closest_fit: the percentage above the lowest-priced nearest entry the buyer tolerates; 0 if unstated."},
    "before_order": text_list("Extra actions the buyer wants performed before checkout, apart from locating and purchasing: presenting or describing the basket to them, obtaining their consent, reviewing the basket contents, confirming a point with them prior to payment. A few words each; blank list if none are requested."),
    "fit_rules": {"type": "array", "description": "With several products: each product that must fit, work with or suit another product of this same order, and the product it must fit (both named as in products). Empty when none.",
                  "items": {"type": "object", "properties": {"product": {"type": "string"}, "fits": {"type": "string"}}, "required": ["product", "fits"]}},
    "set_rules": {"type": "array", "description": "With several products: each rule that ties one product's property to another product's pick. kind unless: product's property must be one of values, except when the other product's other_property is one of other_values. kind if: only when the other product's other_property is one of other_values must product's property be one of values. Empty when none.",
                  "items": {"type": "object", "properties": {"kind": {"type": "string", "enum": ["unless", "if"]}, "product": {"type": "string"}, "property": {"type": "string"},
                                                             "values": text_list("Accepted values."), "other": {"type": "string"}, "other_property": {"type": "string"},
                                                             "other_values": text_list("The other product's values that trigger or lift the rule.")},
                            "required": ["kind", "product", "property", "values", "other", "other_property", "other_values"]}},
}, ["item", "requirements", "budget", "cheapest", "in_stock", "message_budget", "step_budget", "queries", "ask_properties"])

READING_BRIEF = """Your job: turn a buyer's message into a purchase brief that a shopping helper will follow. Complete the plan tool using nothing but what the message says.
- store_category: the shelf name the message gives, copied verbatim; blank if it gives none.
- item: the broad type of goods, worded as the message words it ('an office chair in grey leather' gives item 'office chair').
- requirements: each attribute demanded of the goods themselves (material, colour, dimensions or amount, form, look, subtype, a feature, origin, what it has to operate with, events it should suit). Anything the buyer insists upon, or promises to verify, counts. If the message names a subtype, item stays the broad type and the subtype becomes a requirement (type: ...). Each phrase yields exactly one requirement. Budget, availability and category are handled elsewhere. Story details do not count: the recipient, an event, the place or manner of use, earlier experiences, hesitations or fears - unless the buyer demands that attribute of the goods. Each value has to be something an entry could print as a word or a quantity.
- same_meaning: per value, every alternative an entry could print for precisely that value (synonyms, variant spellings, the identical quantity in a different unit or notation). Matching relies on this list, so include all usual forms, and never some other value.
- either: if one attribute accepts several values ("A or B", "only A and B"), list every one, with the first one going in value.
- from_listing: if a value has to come from another shop entry (identified by vendor and type, possibly with details, possibly reached through entries that page refers to) rather than stated, describe that entry there (seller, item, each detail given) plus its relation: same, pairs (its counterpart under a pairing rule the shop states), not, or within (that entry's number, plus or minus a stated margin); keep value blank.
- exclusions: values the message forbids ("not X", "anything except X", "no X", "avoid X", "stay away from X"), each with its attribute plus alternative forms (same_meaning).
- bound: for each numeric value state its kind - exact (precisely that number), at least, at most, more than / less than (number itself not allowed), or between - plus low / high and unit, as the buyer's wording dictates.
- Numeric values: preserve number, unit and bound precisely ("at least 12 cm", "at most 3 pieces"); "no taller than 80 cm" becomes "at most 80 cm"; "1,250 or more" becomes "at least 1250".
- When a value carries its attribute inside it ("a striped pattern", "a button closure"), split it: property: pattern, value: striped; property: closure, value: button.
- "X or Y" on one attribute: value X, either [X, Y].
- budget: the spending ceiling as a bare number in the currency unit the message uses. A ceiling per piece of one kind of goods belongs to that kind's own budget inside products, not to the overall figure.
- cheapest: true if the buyer wants the lowest-priced qualifying entry, or asks for value for money without naming a yardstick (a good deal, a bargain, not paying too much).
- best_by: value judged by a yardstick ("lowest cost per serving", "most servings for the money") gives best_by "price per serving", best_direction least.
- message_budget / step_budget: limits the buyer sets on messages, and on moves, actions or tool calls.
- products: if two or more distinct kinds of goods are to be purchased together, give each its own item and quantity, plus requirements, exclusions and, when stated, its own spending ceiling. The first kind is then also what the outer item and requirements describe. A single kind (any number of pieces) is no list: keep products empty.
- What counts as goods to buy: ONLY what the buyer asks to have fetched or purchased, each with its stated count. NOT goods to buy: objects in the backstory, something the buyer already has (goods that must suit it get the requirement "compatible with: <its model>"), and an entry a value is taken from (that goes in from_listing). Name the type as the message does, dropping containers ("3 jars of honey" means 3 x honey).
- Food: anything a food has to be free from or low in is a requirement whose value is the label shops print ("gluten free", "low sodium"), under the attribute holding such labels - never an exclusion of that ingredient (which would remove every entry labelled free of it). Keep labels intact: a narrower label ("<kind> fat free") differs from the wider one ("fat free"), and "low X" differs from "X free".
- fit_rules: goods in this order that have to suit each other ("the case must fit whichever phone you pick") - the goods that must suit, and the goods they must suit. Goods that must suit something the buyer has use a "compatible with" requirement instead.
- set_rules: a rule linking an attribute of one kind of goods to what is chosen for another ("the cable's plug must be type C, except when the charger's port is USB-A"; "if the monitor's resolution turns out to be 4K, the stand's arm must be heavy duty").
- queries: brief lookups the way a buyer would type them, combining the goods word with demanded values (no budget terms, no padding).
- closest_fit: true solely if the buyer asks, should nothing match exactly, to be informed and offered or sold the nearest entry; a buyer rejecting substitutes means false.
- compatibility: "brand compatibility" or "compatible devices" given as a feature of the goods is a normal requirement.
- before_order: solely actions the buyer requests (a consent, a view of the basket, a check-in with them prior to checkout), in whatever wording. Availability, spending limit, allowances for messages and moves, plus the no-match instruction are not such actions."""

QUESTION_FORM = form_schema("ask_about", "Attributes to question the buyer on, in sequence.", {
    "properties": text_list("Attributes worth a question, most probable to hide a condition first; each a single attribute of 1-3 plain words, worded like the shop's fact labels."),
}, ["properties"])

QUESTION_BRIEF = """A buyer keeps some conditions on the goods to themselves and reveals each one only when questioned about that exact attribute by name. Decide which attributes to question, in sequence, using the request, conditions known so far, plus fact labels that the shop prints on entries of this type (with a count of entries carrying each label and its usual values).
- Start with an attribute the request's story suggests but does not spell out: the recipient, the place or manner of use, the required look, what it has to operate with.
- Next, attributes of the goods carried by most entries whose values vary from entry to entry (material, appearance, construction or operation, intended user, dimensions and technical specs). An attribute that is identical across entries makes a weak question.
- Skip anything the request has already fixed, plus budget, stock, brand, vendor and category, and anything about the deal rather than the goods (guarantee, condition, shipping).
- Word each attribute as the shop's fact labels word it, plainly.
Give at most six."""

REPLY_FORM = form_schema("hear", "Effect of the buyer's replies.", {
    "adds": {"type": "array", "description": "Fresh conditions on the goods, previously unknown, supplied in a reply to a question.", "items": {
        "type": "object", "properties": {"property": {"type": "string"},
                                         "value": {"type": "string", "description": "Just the value, keeping its bound."},
                                         "bound": {"type": "string", "enum": ["exact", "at least", "at most", "more than", "less than", "between", "none"],
                                                   "description": "How a figure binds: exact (no rounding), at least, at most, more than, less than, between; none without a figure."},
                                         "low": {"type": "number"}, "high": {"type": "number"}, "unit": {"type": "string"},
                                         "same_meaning": text_list(ALT_FORMS),
                                         "product": {"type": "string", "description": "When the line says which product of the order this is for: that product as the order names it; else empty."},
                                         "from_listing": ENTRY_POINTER},
        "required": ["property", "value", "same_meaning"]}},
    "excludes": {"type": "array", "description": "Values a line rules out for the item (not ones it wants), each with its property and, when named, the product.",
                 "items": {"type": "object", "properties": {"property": {"type": "string"}, "value": {"type": "string"}, "product": {"type": "string"},
                                                            "same_meaning": text_list(ALT_FORMS)}, "required": ["property", "value"]}},
    "drops": text_list("Attributes (or exclusions) a reply takes back, worded as listed among known conditions."),
    "budget": {"type": "number", "description": "Fresh spending ceiling from a reply; 0 if absent."},
    "gone": {"type": "boolean", "description": "True if a reply reports the chosen / proposed goods no longer available (a price change alone does not count)."},
    "price_changed": {"type": "boolean", "description": "True if a reply reports a price rise or drop on the chosen goods."},
    "message_limit": {"type": "integer", "description": "If a reply alters the overall message allowance: the revised total (sent messages included; 'two more' after one sent is 3); -1 otherwise."},
    "step_limit": {"type": "integer", "description": "If a reply alters the overall allowance of moves / tool calls: the revised total; -1 otherwise."},
}, ["adds", "drops", "budget", "gone"])

REPLY_BRIEF = """You read what a buyer replies to their shopping helper. Complete the hear tool from the conditions known so far together with the fresh replies.
- excludes: a reply forbidding a value ("anything but X", "not X", "never X") - give attribute and value here, never under adds.
- adds: a reply responding to a question with a value the goods need, whatever the phrasing (direct, idiomatic, by rejecting all alternatives): attribute and value, bound preserved ('it can't be heavier than 3 kg' -> weight: at most 3 kg), and every alternative printed form of the value listed under same_meaning. Use the attribute the reply names, which may not be the one questioned. A reply rejecting all values except one names that one. If the reply names one kind of goods in the order, put it in product: the value belongs to it, even if a different one was questioned. If the reply offers no value, saying instead it has to agree with a different shop entry (identified by type and vendor), describe that entry plus the relation in from_listing, and keep value blank. A reply saying the buyer has no preference, is indifferent, or was confused adds nothing.
- Food that has to be free from or low in an ingredient is the printed label ('<ingredient> free', 'low <ingredient>'), kept intact.
- drops: a reply taking back a condition - name the known condition it takes back.
- budget: a reply stating a fresh spending amount, in the request's currency unit.
- gone: a reply saying the chosen or proposed goods have sold out, are unavailable, or can't be ordered any more."""

GRADE_FORM = form_schema("judge", "Strength with which every entry gives every condition.", {
    "marks": text_list("A line for each entry presented, in sequence: its number, next K when it belongs to the wanted goods type or X otherwise, next a letter for each condition in sequence: "
                     "S = title, or the attribute's matching fact, gives the value (any wording exactly equivalent); W = given faintly (description only, or under a different label); "
                     "N = never mentioned; C = title or facts give another value for the attribute. Sample with a pair of conditions: '3 K SW'."),
}, ["marks"])

GRADE_BRIEF = """Compare shop entries with a buyer's conditions. For every entry (its title and category, price, facts, a description excerpt) plus every numbered condition, rate how the entry's own text gives the value:
S - stated by the title, or by the fact bearing that attribute's name, using any wording or notation equivalent to the value;
W - stated faintly: in the description alone, or in a fact labelled differently;
N - not mentioned by the entry;
C - the entry gives another value for that attribute. For a bounded value, C if the printed number lies outside the bound.
Units: do the conversion yourself when a number appears in a different unit (S if it converts to the value, C if it converts to something else).
Label lists (a fact naming several values): the value has to equal one label; a narrower label differs from the wider one, and "low X", "X free" and "no added X" are separate labels.
Materials: a blend naming the material contains it; under 'NOT: x', a blend naming x counts as x.
A condition written 'NOT: property x' applies to that attribute alone: the same word under a different attribute is not x.
A condition written 'NOT: x' gets S if the entry clearly isn't x, N if silent, C if the entry is x.
Type mark K / X: does the entry belong to the goods type wanted (merely a component, an add-on or a spare for different goods is X).
Text about a different component or entry (identified by number, vendor or model) - saying it also suits something, or carries some value - concerns that other one rather than the entry being rated."""

GATE_FORM = form_schema("check", "Verdict on the entry against all conditions.", {
    "fits": {"type": "boolean", "description": "False solely for a plain breach of a condition."},
    "problems": text_list("Every condition plainly breached, quoting the text proving it; blank list if none."),
}, ["fits", "problems"])

GATE_BRIEF = """A shopping assistant is about to add an entry to the basket, and you are the final gate. The purchase counts only if every condition holds. Go through everything the entry shows: title, category, facts, description.
- Attribute names in the conditions are the assistant's wording and may differ from the shop's: decide on the value, accepting any equivalent wording or notation.
- A condition holds if the attribute's matching fact gives the value (key spelled any way; one item of a comma-separated list suffices). Title and description matter only if the attribute has no fact of its own.
- A condition breaks if the attribute's fact gives another value, or the title plainly offers alternative values for it (where a title offers a range of sizes or editions, it is no single entry of the demanded one).
- Do unit conversions yourself: a number in a different unit holds if it converts to the value or stays inside its bound.
- In a fact naming several labels, the value has to be a complete label (a narrower label differs from the wider one).
- Type: an entry on the shelf the request named is the correct type of goods, however its title describes it, except where it's merely a component, add-on or spare for other goods. Elsewhere, it has to be the type wanted.
- 'NOT: x' breaks if the entry is x; any limit breaks if the printed number lies beyond it.
- Rate the entry by the facts it carries; a note saying an attribute should come from another entry is not a fact about this one: disregard it.
- What an entry suits or operates with is what it states about itself. Text saying another component (by number) also suits something concerns that component, not this entry.
- Absent information is not a breach. Give fits=false only on a plain breach, and name it."""

TRACE_FORM = form_schema("follow", "Value the intended entry gives for the attribute.", {
    "listing": {"type": "integer", "description": "Number of the intended entry (vendor, type and details all agree); 0 if no entry qualifies."},
    "value": {"type": "string", "description": "Value printed by that entry for the attribute, as briefly as possible; blank if it defers to a different entry or is silent."},
    "same_meaning": text_list(ALT_FORMS),
    "next": {**ENTRY_POINTER, "description": "If that entry defers on this attribute to a different entry: the latter; otherwise omit."},
}, ["listing", "value"])

TRACE_BRIEF = """A buyer needs one attribute of the goods to agree with a different shop entry. You receive the attribute, the relation (same, pairs, not, within), the entry intended (type, vendor, some distinguishing details) and entries located in the shop.
- listing: choose the one matching the intended vendor and type, with agreeing details (a size might appear in a different unit). On a tie, choose the one whose notes address this attribute.
- Next, read what it states on that attribute. Notes take priority: a note may give the value directly, or say this attribute should follow yet another entry (other than the present one) - then put that entry in next and keep value blank. Only if the notes are silent, fall back on the entry's fact for the attribute.
- Relation pairs: the answer is the counterpart of that entry's own value under a pairing rule stated by the entry or its shop, not the value itself. Give the counterpart; without a stated pairing rule, keep value blank.
- value: just the value, worded as a shop entry would print it."""

LOOKUP_FORM = form_schema("search_plan", "New catalogue searches.", {
    "queries": text_list("At most four fresh brief lookups (2-4 words) apt to surface a fully qualifying entry; no repeats of past lookups."),
}, ["queries"])

LOOKUP_BRIEF = """No shop entry satisfying all conditions has turned up yet. Using the request, the conditions, past lookups (each with its match count) and titles already seen, write fresh brief catalogue lookups (2-4 words; a matching entry must contain every word, so shorter lookups match more): the goods word plus one demanded value, alternative vendor wording for some value, a lone value beside the goods' main word. If an entry says some other entry or component (identified by number or name) carries the demanded value, look up that identifier."""


SHELF_QUERY_TOOL = form_schema("shelf_and_searches", "The store shelf for one product and the searches to find it.", {
    "store_category": {"type": "string", "description": "Category (spelled exactly like one listed) where these goods are shelved; blank if no listed one is."},
    "queries": text_list("At most five lookups for these goods, strongest first: 2-4 words apiece, each word one a qualifying entry would contain."),
}, ["store_category", "queries"])

SHELF_QUERY_GUIDE = """Plan the catalogue lookups for one kind of goods in a buyer's request. Input: the goods as the buyer named them (plus any shelf they named), demanded values, forbidden values, the shop categories returned by the first lookup (each with its row count), a few entry titles, plus fact labels found on entries of the most probable shelf.
- store_category: the listed category holding these goods, copied exactly as listed (the shelf matching the buyer's words for the goods, in the shop's own wording and plural). A category for a component or add-on of the goods is not their shelf. Blank if nothing matches.
- queries: lookups worded as the shop's titles word things. Each combines the goods' main word (or the shelf's) with one or two demanded values, written as entries write them: the titles' usual unit and notation, the value's customary words. Add one lookup pairing the most distinctive value by itself beside the main word. Each word has to occur in a qualifying entry (a lookup keeps only entries containing all its words), so leave out budget, stock, quantity, guessed brands and padding. Never look up a forbidden value."""


PAIR_TOOL = form_schema("pair", "Which candidates fit the chosen listing.", {
    "fit": {"type": "array", "items": {"type": "integer"}, "description": "Numbers of the candidates that fit, work with or are compatible with the chosen listing; empty when none plainly does."},
}, ["fit"])

PAIR_GUIDE = """A buyer is purchasing several kinds of goods that have to suit one another (a charger for the selected phone, handles for the selected cabinet, a case for the selected device). Given the selected entry and candidates for the other kind, list the candidates whose own text (connector, size, model, brand compatibility, compatible devices) shows they suit or operate with the selected one. Absent information is not a match."""


STEER_TOOL = form_schema("steer", "The next store moves.", {
    "read": {"type": "array", "items": {"type": "integer"}, "description": "Numbers of unread listings worth reading in full (up to 4)."},
    "pages": {"type": "array", "items": {"type": "integer"}, "description": "Numbers of searches whose next page is worth reading (up to 2)."},
    "queries": text_list("At most three fresh brief lookups (2-4 words), none used before."),
}, ["read", "pages", "queries"])

STEER_GUIDE = """Direct a shopping assistant's hunt through a shop. Nothing located so far satisfies all conditions. Using the conditions, entries spotted but not opened (number, title, category, price) and past lookups (number, words, match count, whether another page exists), pick the actions most likely to surface an entry satisfying all of them: entries to open, further pages to fetch, fresh lookups. An entry whose title or category points to the goods type deserves opening even if its title omits some values; a lookup with many matches and another page may hold a match later on; a fresh lookup needs words a qualifying entry would contain (shorter lookups match more)."""

AUDIT_TOOL = form_schema("audit", "Listings that do meet every requirement after all.", {
    "fits": {"type": "array", "items": {"type": "integer"}, "description": "Numbers of the listings that meet every requirement; empty when none does."},
    "why": text_list("For each listing named in fits: the words of the listing that show each requirement met."),
}, ["fits"])

AUDIT_GUIDE = """A shopping assistant is about to report to a buyer that nothing matches. First, re-examine the entries that came nearest. An entry matches if, taken as a whole (title, facts, description), it satisfies all conditions, even where a value is phrased differently, in a different unit or a different notation. A value the entry omits is unmet. What an entry suits or operates with is what it states about itself, not about some other component. List only entries you are certain of."""


# ----------------------------------------------------------------------------- the episode
class Session:
    def __init__(self, problem: dict) -> None:
        environment = problem["environment"]
        self.binding = dict(environment.get("binding") or {})
        rulebook = environment.get("policy_view") or {}
        self.request = str(rulebook.get("query") or "")
        self.opening = rulebook.get("opening")
        self.tools = Toolset(rulebook.get("tools") or [])
        self.max_steps = int(rulebook.get("max_steps") or 30)
        self.finished = bool(rulebook.get("finished"))
        self.task = str(problem.get("problem_id") or problem.get("id") or "task")
        self.started = monotonic()
        self.reader = LlmGateway(self.started)
        self.env_secs: list[float] = []            # how long each store call took (per action)
        self.reader.hurry = self.hurry
        self.step = 0
        self.exchanges: list[dict] = []
        self.notes: list[str] = []
        # what the shopper wants
        self.item = ""
        self.category_named = ""
        self.requirements: list[dict] = []
        self.exclusions: list[dict] = []
        self.budget: float | None = None
        self.cheapest = False
        self.objective: dict | None = None         # the best by a measure other than price: {"kind": most/least, "property"}
        self.in_stock = True
        self.quantity = 1
        self.message_budget = DEFAULT_QUESTIONS
        self.message_known = False                 # the request or the store said the message budget
        self.policy_steps = self.max_steps         # the runner's own limit (a notice never raises the limit past it)
        self.resized = False
        self.msgs_out = 0
        self.messages_closed = False
        self.queries: list[str] = []
        self.ask_first: list[str] = []
        self.asked: list[str] = []
        self.refs: list[dict] = []                 # requirements whose value another listing holds (see trace_pointers)
        # what the store has shown
        self.rows: dict[str, dict] = {}
        self.facts: dict[str, dict] = {}
        self.issued_lookups: dict[str, int | None] = {}
        self.category = ""
        self.gone: set[str] = set()
        self.cart: dict[str, int] = {}
        self.verdicts: dict[str, dict] = {}
        self.judged_under = ""
        self._lexicon: dict[str, dict[str, int]] = {}
        self._lexicon_for = -1
        self._several: set[str] = set()
        self._tally_labels: dict[str, int] = {}
        self.margin = CLOSING_MARGIN
        self.products: list[dict] = []             # a request for several products: one state each (see load_products)
        self.spotted_by: dict[str, set[int]] = {}     # listing -> the products whose searches showed it
        self.stock_doubt = False                 # a sell-out notice that names no listing (several in the cart)
        self.current = 0
        self.item_budget: float | None = None      # this product's own price ceiling (several products)
        self.page_len = PAGE_GUESSES
        self.reply_lines: set[str] = set()
        self.lookup_yield: dict[str, int] = {}
        self.value_matches_all: dict[str, set[str]] = {}
        self.following_of: dict[str, Any] = {}
        self.lookup_role: dict[str, str] = {}      # search (as sent) -> the role of the tool it went to (search or filter)
        self.price_moved: set[str] = set()
        self.rejected: set[str] = set()            # listings the final check found to fall short
        self.checked: dict[str, bool] = {}         # (listing, requirements) -> the final check's verdict
        self.hunted_for_backed = False            # listings whose price a notice said changed          # search (as sent) -> the value the store gave for its next page
        self.extra_queries: list[str] = []
        self.queued_lines: list[str] = []
        self.rejections = 0
        self.before_order: list[str] = []          # steps the request asks for on the way to the order
        self.approved = False                     # a basket message has gone to the shopper
        self.approved_basket = ""                   # the cart that message showed
        self.closest_fit = False                   # nothing fits exactly: say so and order the nearest listing
        self.over_pct = 0.0
        self.cross_compat = False                  # several products, the others must fit the first one chosen
        self.near_note = ""                        # what the nearest listing falls short on, told to the shopper
        self.guarding = False
        self.msg_plan: dict | None = None
        self.steered = 0
        self.no_substitute = False                 # "don't offer a substitute": a decline names no listing
        self.set_rules: list[dict] = []            # "X's P must be V, except when Y's Q is W" (race 14)
        self.fit_rules: list[dict] = []            # products that must fit another product of the order (the reader's)
        self.proposed: list[str] = []               # the closest listings named when declining
        self.rules: dict[tuple[str, str], dict[str, str]] = {}    # (shelf, property) -> the store's pairing rule
        self.rule_pages: dict[str, dict[str, dict[str, str]]] = {}  # listing -> the rules its page states

    # ---- bookkeeping
    def note(self, text: str) -> None:
        self.notes.append(text)
        self.exchanges.append({"role": "assistant", "content": f"[note] {text}"})

    def time_remaining(self) -> float:
        return TASK_TIME_S - (monotonic() - self.started)

    def moves_remaining(self) -> int:
        return self.max_steps - self.step

    def spare_moves(self) -> int:
        # Steps that may go to looking, with the finishing reserve kept.
        return self.moves_remaining() - self.margin

    def time_reserve(self) -> float:
        # The seconds the finish needs, from this task's own pace: the finishing steps at the store's average call time,
        # one last check by the reader, and a margin.
        env = sum(self.env_secs[-12:]) / len(self.env_secs[-12:]) if self.env_secs else 2.0
        model = sum(self.reader.secs[-6:]) / len(self.reader.secs[-6:]) if self.reader.secs else 8.0
        return 15.0 + env * min(self.margin, max(1, self.moves_remaining())) + min(35.0, model)

    def hurry(self) -> bool:
        # Little time left: no more looking, asking or following - the order (or the decline) comes next.
        return self.time_remaining() < self.time_reserve() + 20.0

    def scaled(self, n: int, lo: int = 1) -> int:
        # A step count tuned on 38-step tasks, sized to this task's own limit (20 steps: about half; 50: a third more).
        return max(lo, round(n * self.max_steps / REF_STEPS))

    def finish_reserve(self) -> int:
        # The steps the finish needs, counted from this task: the cart calls (one, or one per product, or one per unit
        # when the cart tool takes no count), a stock re-read per line, the basket or "no clean fit" message the request
        # asks for, the order, and room to look again and re-cart after one refused order.
        n = max(1, len(self.products))
        qtys = [p["quantity"] for p in self.products] or [self.quantity]
        if self.tools.has("cart_set"):
            cart = 1
        elif self.tools.arg.get(("cart_add", "qty")):
            cart = n
        else:
            cart = sum(qtys)
        need = cart + (n if self.tools.has("view") else 0) + 1
        if self.tools.has("message"):
            need += 1 if (self.before_order or (self.closest_fit and not self.no_substitute)) else 0   # (one or the other goes out)
        if self.before_order and self.tools.has("cart_view"):
            need += 1
        need += n + min(self.scaled(6, 2), 8)       # (38-step single product: 10, as tuned; the room for a refused order
        #                                               no longer grows past 8 steps with a large limit - the rest goes to looking)
        return max(4, min(need, self.max_steps // 2))

    def set_limits(self, messages: int | None = None, steps: int | None = None, why: str = "") -> None:
        # A limit the shopper or the store changed mid-task; the finish reserve is sized again.
        if messages is not None and messages >= 0 and messages != self.message_budget:
            self.note(f"message limit {self.message_budget} -> {messages} {why}")
            self.message_budget, self.message_known = messages, True
            self.messages_closed = self.messages_closed and messages <= self.msgs_out
        if steps is not None and steps >= 3:
            steps = min(steps, self.policy_steps)
            if steps != self.max_steps:
                self.note(f"step limit {self.max_steps} -> {steps} {why}")
                self.max_steps = steps
        self.resized = True
        self.margin = self.finish_reserve()
        if getattr(self, "msg_plan", None) is not None:
            self.plan_messages(why or "a limit changed")

    def read_allowance(self) -> None:
        # The request left the message budget unsaid: a store tool that reports it (race #13's account_report:
        # {"allowance_unspent": 2, "taken": 0}) is asked once, rather than guessing.
        if self.message_known or not self.tools.has("report") or self.moves_remaining() < self.margin + 3:
            return
        call_result = self.send([self.tools.call("report")])
        numerals: dict[str, float] = {}

        def walk(obj_item: Any) -> None:
            if isinstance(obj_item, dict):
                for k, v in obj_item.items():
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        numerals[label_tokens(k)] = float(v)
                    walk(v)
        walk(call_result[0]["core"] if call_result else None)
        left = next((v for k, v in numerals.items() if re.search(r"unspent|left|remain|availab|allow", k)), None)
        used = next((v for k, v in numerals.items() if re.search(r"taken|used|spent|sent|asked", k)), 0.0)
        if left is not None:
            self.msgs_out = max(self.msgs_out, int(used))
            self.set_limits(messages=int(used + left), why="(the store's report)")

    # ---- the environment
    def send(self, actions: list[dict]) -> list[Any]:
        # Run the actions in order, each one step. Returns one observation per action run (fewer when the episode
        # ends inside the group).
        actions = [a for a in actions if a]
        out: list[Any] = []
        msg_tool = self.tools.role.get("message")
        while actions and not self.finished and self.moves_remaining() > 0:
            group = actions[: min(GROUP_SIZE, self.moves_remaining())]
            # (a message to the shopper goes alone: its answer is read by the world's own models, which can take long)
            if any(a["name"] == msg_tool for a in group):
                first = next(i for i, a in enumerate(group) if a["name"] == msg_tool)
                group = group[: max(1, first)] if first > 0 else group[:1]
            actions = actions[len(group):]
            wrapper = {**self.binding, "step": self.step + 1}
            if len(group) == 1:
                wrapper["action"] = group[0]
            else:
                wrapper["actions"] = group
            result = None
            t0 = monotonic()
            for _ in range(3):
                result = _proxy.post("/environment/call", json_data=wrapper)
                if isinstance(result, dict):
                    break
            if not isinstance(result, dict):
                self.note(f"store call #{self.step + 1} errored")
                self.finished = True if self.time_remaining() < 5 else self.finished
                return out
            self.env_secs.append((monotonic() - t0) / max(1, len(group)))
            observations = result.get("observations") if len(group) > 1 else None
            if not isinstance(observations, list):
                observations = [result.get("observation")]
            executed = observations[: len(group)]
            self.step = int(result.get("step") or (self.step + len(executed)))
            self.finished = bool(result.get("finished"))
            self.exchanges.append({"role": "assistant", "content": "", "tool_calls": group[: len(executed)], "environment_result": result})
            for action, obs_data in zip(group, executed):
                out.append(self._take_in(action, obs_data))
            if self.finished:
                break
        return out

    def _take_in(self, action: dict, obs_data: Any) -> dict:
        # Record what an observation shows; returns {"core", "notices", "refusal", "error"}.
        core, notices, refusal = split_alerts(obs_data)
        error = None
        if isinstance(core, dict) and len(core) <= 2:
            # (a world names its failures its own way: error, call_fault, issue, problem)
            for k, v in core.items():
                if re.search(r"error|fault|issue|problem|fail", low(k)) and isinstance(v, (str, int)) and v:
                    error = str(v)
        role = next((r for r, n in self.tools.role.items() if n == action["name"]), "")
        if role in ("search", "filter", "view", "facts", "facts_many", "cart_view") and not error:
            for ent_row in entries_in(core):
                found = rules_in(ent_row.get("description"), ent_row.get("facts") or self.facts.get(ent_row["id"]))
                if found:
                    self.rule_pages[ent_row["id"]] = found
                    for prop, pairs in found.items():
                        self.rules.setdefault((low(ent_row.get("category") or ""), prop), {}).update(pairs)
                eid = ent_row["id"]
                known = self.rows.setdefault(eid, {"id": eid})
                for k, v in ent_row.items():
                    if k == "facts":
                        self.facts[eid] = v
                    elif k == "description" and isinstance(v, str) and len(str(known.get(k) or "")) > len(v) and v[:60] in str(known.get(k)):
                        continue        # (a search row's shorter text never replaces the page read in full, notes and all)
                    elif v is not None:
                        known[k] = v
                if role == "facts" and "facts" not in ent_row and isinstance(core, dict):
                    flattened = {k: v for k, v in core.items() if isinstance(v, (str, int, float)) and k not in ("title",) and v != eid}
                    if flattened and len(flattened) >= 3:
                        self.facts[eid] = flattened
            if role in ("search", "filter"):
                key = json.dumps(action["args"], sort_keys=True)
                self.lookup_role[key] = role
                total, shown = sum_of(core), len(entries_in(core))
                self.issued_lookups[key] = total
                self.following_of[key] = following_page_of(core)
                page_param = self.tools.arg.get(("search", "page"))
                first = json.dumps({k: v for k, v in action["args"].items() if k != page_param}, sort_keys=True)
                borne = self.borne_by(action["args"].get(self.tools.arg.get(("search", "text"), ""), ""))
                for r in entries_in(core):
                    self.spotted_by.setdefault(r["id"], set()).add(self.current)
                    self.value_matches_all.setdefault(r["id"], set()).update(borne)
                worthwhile = sum(1 for r in entries_in(core) if self.on_shelf_row(r["id"]) and self.proof(r["id"]) > 0)
                self.lookup_yield[key] = worthwhile
                self.lookup_yield[first] = self.lookup_yield.get(first, 0) + worthwhile
                if total is not None and total > shown > 0:
                    self.page_len = max(self.page_len if self.page_len != PAGE_GUESSES else 0, shown)
        if role == "cart_add" and not error and not (isinstance(core, dict) and any(re.search(r"issue|error|fail", low(k)) for k in core)):
            pid = action["args"].get(self.tools.arg.get(("cart_add", "id"), ""), "")
            if pid:
                self.cart[pid] = self.cart.get(pid, 0) + int(action["args"].get(self.tools.arg.get(("cart_add", "qty"), ""), 1) or 1)
        if role == "cart_remove" and not error:
            pid = action["args"].get(self.tools.arg.get(("cart_remove", "id"), "") or self.tools.arg.get(("cart_add", "id"), ""), "")
            self.cart.pop(pid, None)
        if role == "message":
            if error:
                self.messages_closed = True
                if re.search(r"budget|limit|exceed|quota|allowance|too many", low(error)):
                    self.message_budget = self.msgs_out      # (the store's count is the one that holds)
            else:
                self.msgs_out += 1
        lines = notices + refusal
        if lines:
            self.queued_lines += lines
            self.note(f"shopper notices: {lines}")
        return {"core": core, "notices": notices, "refusal": refusal, "error": error, "role": role}

    # ---- reading the request
    def parse_request(self) -> None:
        text = f"The shopper's request:\n{self.request}"
        if self.opening is not None:
            text += f"\n\nOpening observation: {json.dumps(self.opening)[:1500]}"
        plan = self.reader.complete(READING_BRIEF, text, BRIEF_FORM, 2400, think=True, limit=35.0, models=[HEAVY, SPARE]) or {}
        if not str(plan.get("item") or "").strip():
            # (the thinking pass did not come back in time: the request is read again without it)
            self.note("plan: reasoning-mode reading errored, retrying plain")
            plan = self.reader.complete(READING_BRIEF, text, BRIEF_FORM, 2400) or {}
        self.code_read = request_by_code(self.request)
        wanted = self.code_read.get("wanted") or []
        listed = [p for p in listify(plan.get("products")) if isinstance(p, dict)]
        missing = [w for w in wanted if not any(product_words(w["item"]) & (product_words(p.get("item")) | product_words(p.get("store_category")))
                                                for p in listed)]
        if len(wanted) >= 2 and (len(listed) < 2 or missing):
            # (the request asks to have several products fetched and the reading kept one: read again, told which)
            names = "; ".join(f"{w['quantity']} x {w['item']}" for w in wanted)
            again = self.reader.complete(READING_BRIEF, text + f"\n\nNote: SEVERAL products are wanted here: {names}. Give all of them separate entries in "
                                     "products, carrying their own quantity, requirements and exclusions.", BRIEF_FORM, 2400, limit=40.0) or {}
            if len(listify(again.get("products"))) >= 2:
                plan = {**plan, "products": again["products"]}
        plan = merge_plan(plan, self.code_read, self.request)
        for line in plan.pop("_notes", []):
            self.note(f"products: {line}")
        self.set_rules = [{"kind": "if" if low(r.get("kind")) == "if" else "unless", "product": str(r.get("product") or ""),
                           "property": str(r.get("property") or ""), "values": [str(v) for v in listify(r.get("values")) if str(v).strip()],
                           "other": str(r.get("other") or ""), "other_property": str(r.get("other_property") or ""),
                           "other_values": [str(v) for v in listify(r.get("other_values")) if str(v).strip()]}
                          for r in listify(plan.get("set_rules")) if isinstance(r, dict) and r.get("product") and r.get("values")]
        coded_refs = refs_by_code(self.request, True)
        self.item = str(plan.get("item") or "").strip() or " ".join(words(self.request)[:2])
        self.category_named = str(plan.get("store_category") or "").strip()
        unparsed, self.refs = split_pointers(plan.get("requirements") or [], self.request, True)
        self.refs += [c for c in coded_refs if not any(low(c["ref"]["seller"]) == low(x["ref"]["seller"]) for x in self.refs)
                      and not listify(plan.get("products"))]
        if self.refs:
            # (a value tied to another page is no stated value: the reader may have copied the page's name as one)
            tied = {low(x["property"]) for x in self.refs}
            unparsed = [x for x in unparsed if not (isinstance(x, dict) and low(x.get("property")) in tied)]
        for r in unparsed:
            if isinstance(r, dict) and str(r.get("value") or "").strip():
                r["value"], why = bind_value(r, self.request)
                if why:
                    self.note(f"{r.get('property')}: {why}")
                self.requirements.append({"property": str(r.get("property") or "").strip(), "value": str(r["value"]).strip(),
                                          "same": [str(a) for a in listify(r.get("same_meaning")) if str(a).strip()][:8],
                                          "either": [str(a) for a in listify(r.get("either")) if str(a).strip()][:5], "stated": True,
                                          "must": r.get("must") is not False})
        for r in plan.get("exclusions") or []:
            if isinstance(r, dict) and str(r.get("value") or "").strip():
                self.exclusions.append({"property": str(r.get("property") or "").strip(), "value": str(r["value"]).strip(),
                                        "same": [str(a) for a in listify(r.get("same_meaning")) if str(a).strip()][:8]})
        self.tidy_needs()
        self.budget = number(plan.get("budget")) or self.cap_from_prose(self.request)
        self.cheapest = bool(plan.get("cheapest")) or bool(re.search(CHEAPEST_WORDS, low(self.request)))
        self.objective = self.objective_from(plan, self.request)
        self.in_stock = plan.get("in_stock") is not False
        self.quantity = max(1, int(number(plan.get("quantity")) or 1))
        stated = self.msg_cap_from_prose(self.request)
        msg_quota = number(plan.get("message_budget"))
        self.message_known = stated is not None or (msg_quota is not None and msg_quota >= 0)
        self.message_budget = stated if stated is not None else (int(msg_quota) if msg_quota is not None and msg_quota >= 0 else DEFAULT_QUESTIONS)
        # (the runner's max_steps is the task's own limit; a smaller one the request states counts only when it is
        # phrased as a limit and the reader agrees, or the reader's number is written in the request)
        steps = self.step_cap_from_prose(self.request)
        step_allow = number(plan.get("step_budget"))
        step_allow = int(step_allow) if step_allow is not None and step_allow > 0 else None
        limit = steps if steps and (step_allow is None or step_allow == steps) else step_allow if step_allow and steps is None and re.search(rf"\b{step_allow}\b", self.request) else None
        if limit and 3 <= limit < self.max_steps:
            self.note(f"step limit {self.max_steps} -> {limit} (the request's own)")
            self.max_steps = limit
        self.queries = [cond_item for cond_item in (plan.get("queries") or []) if isinstance(cond_item, str) and cond_item.strip()]
        self.ask_first = [p for p in (plan.get("ask_properties") or []) if isinstance(p, str) and p.strip()]
        self.before_order = [str(x).strip() for x in listify(plan.get("before_order")) if str(x).strip()][:4]
        if self.before_order and not CONSENT_CUE.search(self.request):
            # (the reader listed a step the request's own words never ask for: not done - it would spend a question)
            self.note(f"unrequested pre-checkout actions removed: {self.before_order}")
            self.before_order = []
        no_sub = re.search(r"(?:don't|do not|never|no)\s+(?:offer|suggest|push|bring)?\s*(?:a |any )?substitut", low(self.request))
        self.closest_fit = bool(NEAR_HINT.search(self.request)) or (bool(plan.get("closest_fit")) and not no_sub)
        self.no_substitute = bool(no_sub) or bool(re.search(r"\b(?:don't|do not|never|no)\s+(?:\w+\s+){0,2}(?:some |a |any )?substitut", low(self.request)))
        if self.no_substitute and not NEAR_HINT.search(self.request):
            self.closest_fit = False
        pct = OVER_PCT.search(self.request)
        self.over_pct = float(pct.group(1)) if pct else max(0.0, number(plan.get("over_pct")) or 0.0)
        self.fit_rules = [r for r in listify(plan.get("fit_rules")) if isinstance(r, dict) and r.get("product") and r.get("fits")]
        self.cross_compat = bool(self.fit_rules)
        if not self.before_order and CONSENT_CUE.search(self.request):
            self.before_order = ["let the buyer review the basket and approve it"]
        if len(listify(plan.get("products"))) < 2 and SEVERAL_CUE.search(self.request):
            # (the request lists several things and the reading kept one: read again, told so)
            cue = text + "\n\nNote: SEVERAL products of different kinds are wanted here: give all of them separate entries in products, carrying their own requirements."
            for think in (True, False):
                again = self.reader.complete(READING_BRIEF, cue, BRIEF_FORM, 2400, think=think, limit=40.0) or {}
                if len(listify(again.get("products"))) >= 2:
                    plan = {**plan, "products": again["products"]}
                    break
        self.load_products(plan)
        self.bind_owned()
        self.margin = self.finish_reserve()
        self.note(f"plan goods={self.item!r} shelf={self.category_named!r} wants={[(r['property'], r['value']) for r in self.requirements]} "
                  f"bans={self.exclusions} cap={self.budget} lowest_price={self.cheapest} units={self.quantity} "
                  f"msg_allow={self.message_budget} move_allow={self.max_steps} pre_checkout={self.before_order} nearest_ok={self.closest_fit}"
                  f"({self.over_pct:g}%) cross_compat={self.cross_compat} tools={self.tools.summarize()}")

    PRODUCT_STATE = ("item", "category_named", "category", "requirements", "exclusions", "quantity", "queries", "ask_first", "objective",
                   "verdicts", "judged_under", "hunted_for_backed", "extra_queries", "asked", "item_budget", "refs")

    def load_products(self, plan: dict) -> None:
        # A request for several different products: one state per product (its item, requirements, quantity, own
        # price ceiling); the episode works on one at a time and orders them together.
        unparsed = listify(plan.get("products"))
        unparsed = [p for p in unparsed if isinstance(p, dict) and str(p.get("item") or "").strip()]
        if len(unparsed) < 2:
            return
        snapshots = []
        for p in unparsed:
            plain, refs = split_pointers(p.get("requirements") or [], self.request, True)
            for r in plain:
                if isinstance(r, dict) and str(r.get("value") or "").strip():
                    r["value"] = bind_value(r, self.request)[0]
            need_items = [{"property": str(r.get("property") or "").strip(), "value": str(r["value"]).strip(),
                     "same": [str(a) for a in listify(r.get("same_meaning")) if str(a).strip()][:8],
                     "either": [str(a) for a in listify(r.get("either")) if str(a).strip()][:5],
                     "stated": True, "must": r.get("must") is not False}
                    for r in plain if isinstance(r, dict) and str(r.get("value") or "").strip()]
            ban = [{"property": str(x.get("property") or "").strip(), "value": str(x["value"]).strip(),
                     "same": [str(a) for a in listify(x.get("same_meaning")) if str(a).strip()][:8]}
                    for x in p.get("exclusions") or [] if isinstance(x, dict) and str(x.get("value") or "").strip()]
            snapshots.append({"item": str(p["item"]).strip(), "category_named": str(p.get("store_category") or "").strip(), "category": "",
                           "requirements": need_items, "exclusions": ban, "quantity": max(1, int(number(p.get("quantity")) or 1)),
                           "queries": [], "ask_first": [], "verdicts": {}, "judged_under": "", "hunted_for_backed": False,
                           "extra_queries": [], "asked": [], "item_budget": number(p.get("budget")) or None, "refs": refs,
                           "objective": self.objective_from(p, " ".join(s_ for s_ in re.split(r"(?<=[.;])\s+", self.request)
                                                                    if self.shelf_words(s_) & self.shelf_words(p.get("item") or "")))})
        # (a page reference the reader did not put on any product - as with a product code added: the product named last
        # before it in the request)
        flattened = " ".join(str(self.request or "").replace("\u2014", "\u2013").split())
        for c in refs_by_code(self.request, True):
            if c.get("at") is None or any(low(x["ref"]["seller"]) == low(c["ref"]["seller"]) for p in snapshots for x in p["refs"]):
                continue
            before = low(flattened[: c["at"]])
            where = [(max((before.rfind(w) for w in low(p["item"]).replace("&", " ").split() if len(w) > 3), default=-1), i)
                     for i, p in enumerate(snapshots)]
            pos, i = max(where)
            if pos >= 0:
                c = {k: v for k, v in c.items() if k != "at"}
                snapshots[i]["refs"].append(c)
                snapshots[i]["requirements"] = [r for r in snapshots[i]["requirements"] if attr_terms(r["property"]) != attr_terms(c["property"])]
        prose_lines = [x for x in re.split(r"(?<=[.!?;])\s+", self.request) if x.strip()]
        for p in snapshots:
            if not p["item_budget"]:
                # (a ceiling the reader did not put on the product: the one the product's own sentence states)
                lead_part = (words(p["item"]) or [""])[-1]
                for x in prose_lines:
                    if lead_part and holds(x, lead_part) and not re.search(r"whole lot|the lot|total|all told|altogether|together|bill", low(x)):
                        cap = self.cap_from_prose(x)
                        if cap:
                            p["item_budget"] = cap
                            break
        self.products = snapshots
        self.current = 0
        if any(p["item_budget"] for p in snapshots) and not re.search(r"\b(?:in total|altogether|all together|combined|overall|for everything|all of it|the lot|whole lot|whole order|"
                                                                     r"total (?:stays|comes|is|of)|the total|order total|grand total)\b", low(self.request)):
            self.budget = None      # (each product has its own ceiling; no limit on the order as a whole was stated)
        self._restore(0)
        for i in range(len(snapshots)):
            self.switch_to(i)
            self.tidy_needs()
        self.switch_to(0)
        self.note("products: " + " | ".join(f"{p['item']} x{p['quantity']} {[(r['property'], r['value']) for r in p['requirements']]}"
                                            f"{' <=' + str(p['item_budget']) if p['item_budget'] else ''}" for p in self.products))

    def _restore(self, i: int) -> None:
        for k in self.PRODUCT_STATE:
            setattr(self, k, self.products[i][k])
        self.current = i

    def switch_to(self, i: int) -> None:
        # Work on product i: the episode's item, requirements, verdicts... are that product's.
        if not self.products:
            return
        for k in self.PRODUCT_STATE:
            self.products[self.current][k] = getattr(self, k)
        self._restore(i)

    DEMAND_RE = re.compile(r"\b(?:must|has to|have to|got to|gotta|got ta|need(?:s|ed)?|should|only|nothing else|no other|required|"
                        r"i want|i'd like|i would like|i'm after|i am after|after|looking for|hunting|searching|i'll take|i will take|"
                        r"it's got|has got|can't be|cannot be|has to be|wants?|insist\w*|check|leave out|cross off|rule out|"
                        r"skip|avoid|exclude|not a|no\b|without|is the category|are what|"
                        r"fetch|find me|get me|i fancy|i'd love|i'll have|pick the|where the|whose)\b", re.I)     # (race 14: "Could you fetch me ...")

    def tidy_needs(self) -> None:
        # A value written as an absence ('no laces', 'without buckles') rules that value out; a requirement that only
        # a background sentence carries (no word in it insists: 'for her school play next month') is a wish.
        prose_lines = [low(x) for x in re.split(r"(?<=[.!?])\s+|\n", self.request) if x.strip()]
        kept = []
        for r in self.requirements:
            m = re.match(r"\s*(?:no|not|without|non)[\s-]+(.+)", low(r["value"]))
            if m and len(m.group(1).split()) == 1:
                self.exclusions.append({"property": r["property"], "value": m.group(1).strip()})
                continue
            word_items = [w for w in roots(spread(r["value"])) if len(w) > 2]
            where = [x for x in prose_lines if word_items and all(w in set(roots(spread(x), hold_filler=True)) for w in word_items)]
            if r.get("must", True) and where and not any(self.DEMAND_RE.search(x) for x in where):
                r["must"] = False
                self.note(f"requirement {r['property']}: {r['value']} is context, not a demand")
            kept.append(r)
        seen = set()
        # (a value the request rules out is no requirement as well: 'nothing blended' is not 'material: blended')
        banned = {" ".join(roots(spread(x["value"]))) for x in self.exclusions}
        kept = [r for r in kept if " ".join(roots(spread(r["value"]))) not in banned]
        self.requirements = kept
        self.exclusions = [x for x in self.exclusions if not ((key := (low(x["property"]), low(x["value"]))) in seen or seen.add(key))]

    @staticmethod
    def cap_from_prose(text: str) -> float | None:
        t = low(text)
        m = re.search(r"(?:over|above|past|more than|up to|max(?:imum)?|ceiling(?: is)?|cap(?: is)?|tops out at|budget(?: is)?|under|within"
                      r"|spend|comes? (?:in )?(?:at|to))\s*(?:a mere |just |only |of |up to )?\$?\s*(\d[\d,]*(?:\.\d+)?)(?![\d.]*\s*%)", t)
        if not m:
            # ('3950 cents or under', '1900 cents at the very most')
            m = re.search(r"(\d[\d,]*(?:\.\d+)?)\s*(?:us\s*)?(?:cents?|cts|dollars?|\$)?\s*(?:or (?:under|less|below)|at (?:the very )?most|tops)", t)
        return number(m.group(1)) if m else None

    @staticmethod
    def msg_cap_from_prose(text: str) -> int | None:
        t = low(text)
        numerals = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "a single": 1, "no": 0, "zero": 0}
        m = re.search(r"(\d+|one|two|three|four|five|six|a single|zero|no)\s+(?:more\s+)?(?:messages?|questions?|texts?)\b", t) \
            or re.search(r"(?:message|text|ask|write|ping|contact)\s+me\s+(\d+|one|two|three|four|five|six)\s+times?", t)
        if m:
            v = m.group(1)
            return int(v) if v.isdigit() else numerals[v]
        return None

    @staticmethod
    def step_cap_from_prose(text: str) -> int | None:
        for pattern in STEP_LIMIT:
            m = pattern.search(low(text))
            if m:
                return int(m.group(1))
        return None

    # ---- searching
    @staticmethod
    def clean_query(cond_item: str) -> str:
        # A search as titles word things: no joining words (a store that keeps listings holding every word finds none
        # with "or"), no quote marks, each word once.
        word_items = [w for w in re.sub(r"[\"()]", " ", str(cond_item)).split() if low(w) not in ("or", "and", "&", "the", "a", "an", "of", "for", "with", "in")]
        return " ".join(dict.fromkeys(word_items))

    def search(self, queries: list[str], pages: int = 1) -> list[dict]:
        limit = self.tools.text_cap()
        calls = []
        for cond_item in queries:
            cond_item = self.clean_query(cond_item)[:limit].strip()
            if not cond_item:
                continue
            # (on the item's shelf once it is known: a store that searches words within a shelf then shows only it)
            c = self.tools.call("search", mode="typed", text=cond_item, category=self.category or None)
            if c and json.dumps(c["args"], sort_keys=True) not in self.issued_lookups:
                calls.append(c)
        return self.send(calls)

    def shelf_lookup(self, cap: float | None, pages: int = 1, category: str | None = None) -> list[dict]:
        category = category or self.category
        if not category:
            return []
        if ("search", "category") in self.tools.arg:
            c = self.tools.call("search", mode="fields", category=category, cap=cap)
        elif self.tools.has("filter"):
            c = self.tools.call("filter", category=category, cap=cap)      # (a separate tool lists a shelf)
        else:
            return []
        if not c:
            return []
        key = json.dumps(c["args"], sort_keys=True)
        results = self.send([c]) if key not in self.issued_lookups else []
        for _ in range(pages - 1):
            following = self.following_params(key, unseen=False)
            if following is None or self.spare_moves() <= 0:
                break
            key = json.dumps(following, sort_keys=True)
            if key not in self.issued_lookups:
                results += self.send([{"name": c["name"], "args": following}])
        return results

    @staticmethod
    def shelf_words(text: Any) -> set[str]:
        # A shelf's or a product's words, as stems, plurals folded, without joiners: "Hoodies & Sweatshirts" ~ "hoodie or
        # sweatshirt", "Wardrobe Organisers" ~ "wardrobe organiser".
        out = set()
        for w in roots(spread(re.sub(r"[()&/,]", " ", str(text or "")))):
            if w in ("or", "and", "the", "kind", "item", "thing", "x"):
                continue
            out.add(re.sub(r"(?:ies|ie|y|es|s)$", "", w) if len(w) > 3 else w)      # ("hoody" / "hoodie" -> "hood")
        return out

    def shelf_score(self, shelf_name: str) -> float:
        # How well a store category names this product: its words found among the shelf the shopper named and the item,
        # its main noun first ("Watch Straps" is not the shelf for a watch). What a bracket adds ("UHT Milk(Ready to
        # Drink)") is a qualifier, not needed.
        said = self.shelf_words(f"{self.category_named} {self.item}")
        core = re.sub(r"\(.*?\)", " ", str(shelf_name or ""))
        cw = self.shelf_words(core)
        if not cw or not said:
            return 0.0
        lead_part = (sorted(self.shelf_words(core.split()[-1]) if core.split() else []) or [""])[0]
        cover = len(cw & said) / len(cw)
        return (cover + 0.5 * len(cw & said) / len(said)) if lead_part in said else 0.5 * cover

    def probe_shelf(self) -> None:
        # The shelf the shopper named, not among the categories the searches showed: listed once by that name (a filter
        # tool, or the search tool's shelf field) - a store that has it shows its rows.
        named = self.category_named
        if not named or any(low(ent_row.get("category") or "") == low(named) for ent_row in self.rows.values()) or self.spare_moves() < 6:
            return
        before = set(self.rows)
        self.shelf_lookup(None, category=named)
        shown = [eid for eid in self.rows if eid not in before]
        if shown:
            self.note(f"shelf {named!r} listed by name: {len(shown)} rows")

    def plan_searches(self) -> None:
        # Once per product, after the first search shows the store: the model names the store shelf the product is filed
        # under (code keeps it only when the store showed it) and writes the first searches from the shelf's words and the
        # product's features; code keeps searches that carry the item, the shelf or a required value.
        if getattr(self, "_planned", None) is None:
            self._planned = set()
        if self.current in self._planned or self.time_remaining() < 60:
            return
        self._planned.add(self.current)
        counts: dict[str, int] = {}
        for eid, ent_row in self.rows.items():
            if self.products and self.current not in self.spotted_by.get(eid, set()):
                continue
            if isinstance(ent_row.get("category"), str):
                counts[ent_row["category"]] = counts.get(ent_row["category"], 0) + 1
        cats = sorted(counts, key=lambda c: (-self.shelf_score(c), -counts[c]))[:12]
        titles = [str(ent_row.get("title"))[:80] for eid, ent_row in self.rows.items()
                  if not self.category or low(ent_row.get("category")) == low(self.category)][:10]
        text = (f"Product: {self.item}" + (f" (the shopper names the shelf: {self.category_named})" if self.category_named else "")
                + f"\nRequired values:\n" + ("\n".join(f"- {r['property']}: {r['value']}" + (f" (or {', '.join(r.get('either', [])[1:])})" if len(r.get('either', [])) > 1 else "")
                                                     for r in self.requirements if r.get("must", True)) or "(none stated yet)")
                + ("\nRuled out: " + "; ".join(f"{x['property']} {x['value']}" for x in self.exclusions) if self.exclusions else "")
                + "\n\nStore categories the first search showed (rows):\n" + ("\n".join(f"- {c} ({counts[c]})" for c in cats) or "(none)")
                + "\n\nListing titles seen:\n" + ("\n".join(f"- {t}" for t in titles) or "(none)")
                + f"\n\nFact labels on that shelf:\n{self.label_digest()}\n\nSearch box limit: {self.tools.text_cap()} characters.")
        answer = self.reader.complete(SHELF_QUERY_GUIDE, text, SHELF_QUERY_TOOL, 500) or {}
        shelf = str(answer.get("store_category") or "").strip()
        if shelf and any(low(shelf) == low(c) for c in counts):
            shelf = next(c for c in counts if low(c) == low(shelf))
            if shelf != self.category:
                self.note(f"shelf for {self.item!r}: {self.category!r} -> {shelf!r} (the planner)")
                self.category = shelf
        keep_words = self.shelf_words(f"{self.item} {self.category} {self.category_named}") | {
            w for r in self.requirements for v in self.literal_values(r) for w in roots(spread(v))}
        queries = []
        for cond_item in listify(answer.get("queries")):
            cond_item = " ".join(str(cond_item).replace('"', " ").split())[: self.tools.text_cap()]
            if cond_item and set(roots(spread(cond_item))) & keep_words and not re.search(r"\b(?:cheap|budget|stock|cents?|price|under|buy)\b", low(cond_item)) \
                    and not self.issued_lookup(cond_item) and cond_item not in queries:
                queries.append(cond_item)
        if queries:
            self.note(f"planned searches for {self.item!r}: {queries[:5]}")
            self.queries = queries[:5] + [cond_item for cond_item in self.queries if cond_item not in queries]

    def fix_shelf(self) -> None:
        self._fix_shelf()
        if self.category:
            shelf = set(roots(spread(self.category))) | set(roots(self.item))
            kept = []
            for r in self.requirements:
                if set(roots(spread(r["value"]))) <= shelf and r.get("stated"):
                    self.note(f"requirement {r['property']}: {r['value']} merely repeats shelf {self.category!r}; removed")
                    continue
                kept.append(r)
            self.requirements = kept

    def _fix_shelf(self) -> None:
        # The store category the item is filed under: the named one when the rows show it, else the one most rows
        # whose title carries the item word are filed under.
        counts: dict[str, int] = {}
        named = low(self.category_named)
        for eid, ent_row in self.rows.items():
            if self.products and self.current not in self.spotted_by.get(eid, set()):
                continue        # (several products: a shelf is read from this product's own searches)
            shelf_name = ent_row.get("category")
            if not isinstance(shelf_name, str):
                continue
            if named and low(shelf_name) == named:
                self.category = shelf_name
                return
            weight = 2 if holds(ent_row.get("title"), self.item) or holds(shelf_name, self.item) else 1
            counts[shelf_name] = counts.get(shelf_name, 0) + weight
        # (the store's name for the shelf the shopper named: 'Hoodies & Sweatshirts' for 'a hoodie or sweatshirt')
        named_like = sorted((c for c in counts if self.shelf_score(c) >= 1.0), key=lambda c: (-self.shelf_score(c), -counts[c]))
        if named_like:
            self.category = named_like[0]
            return
        # (a shelf whose whole name the request says - 'a luxury & formal watch' - is that shelf, however few rows show it)
        said = set(roots(spread(self.request)))
        voiced = [c for c in counts if set(roots(spread(c))) <= said and holds(c, self.item.split()[-1] if self.item else "")]
        if not named and voiced:
            self.category = max(voiced, key=lambda c: (len(roots(c)), counts[c]))
            return
        if named and "/" not in named and not counts:
            self.category = self.category_named
            return
        if counts:
            best = max(counts.items(), key=lambda pair_kv: pair_kv[1])[0]
            # (a category whose own name carries the item word wins over one that only files many such titles)
            for shelf_name in counts:
                if holds(shelf_name, self.item) and counts[shelf_name] * 2 >= counts[best]:
                    best = shelf_name
                    break
            self.category = best
        elif named:
            self.category = self.category_named

    # ---- what a listing is worth before its facts are read
    def within_cap(self, eid: str) -> bool:
        price = (self.rows.get(eid) or {}).get("price")
        if price is None:
            return True
        if self.item_budget and price > self.item_budget + 1e-9:
            return False        # (this product's own ceiling, when several products are bought)
        return self.budget is None or price * self.quantity <= self.budget + 1e-9

    def obtainable(self, eid: str) -> bool:
        if eid in self.gone:
            return False
        stock = (self.rows.get(eid) or {}).get("in_stock")
        return not self.in_stock or stock is not False

    def on_shelf_row(self, eid: str) -> bool:
        shelf_name = (self.rows.get(eid) or {}).get("category")
        return not self.category or not isinstance(shelf_name, str) or low(shelf_name) == low(self.category)

    def entry_rating(self, eid: str) -> float:
        ent_row = self.rows.get(eid) or {}
        title, text = ent_row.get("title") or "", f"{ent_row.get('title') or ''} {ent_row.get('description') or ''}"
        facts = " ".join(f"{k} {v}" for k, v in (self.facts.get(eid) or {}).items())
        score = 0.0
        for r in self.requirements:
            values = self.literal_values(r)
            if any(holds(title, v) for v in values):
                score += 3
            elif any(holds(facts, v) for v in values):
                score += 2.5
            elif any(holds(text, v) for v in values):
                score += 1
            else:
                score -= 1.5
        for x in self.exclusions:
            if holds(title, x["value"]) or holds(facts, f"{x['value']}"):
                score -= 4
        if self.on_shelf_row(eid):
            score += 2
        if holds(title, self.item):
            score += 1
        return score

    def candidates(self) -> list[str]:
        cand_pool = [eid for eid in self.rows if self.obtainable(eid) and self.within_cap(eid) and self.on_shelf_row(eid)]
        return sorted(cand_pool, key=lambda eid: (-self.entry_rating(eid), (self.rows[eid].get("price") or 0)))

    def load_facts(self, eids: list[str], margin: int | None = None) -> None:
        headroom = self.spare_moves() if margin is None else self.moves_remaining() - margin
        n = self.tools.group_cap()
        if n > 1:
            # (a tool that reads several listings at once: one step for up to n of them; a listing it leaves out is
            # read alone)
            eids = [r for r in eids if r not in self.facts][: max(0, headroom) * n]
            self.send([self.tools.call("facts_many", ids=eids[i:i + n]) for i in range(0, len(eids), n)])
            headroom = (self.spare_moves() if margin is None else self.moves_remaining() - margin)
            if eids and all(r not in self.facts for r in eids):
                self.tools.role.pop("facts_many", None)       # (it read none of them: listings are read one by one now)
            eids = [r for r in eids if r not in self.facts]
        eids = [r for r in eids if r not in self.facts][: max(0, headroom)]
        if eids and self.tools.has("facts"):
            self.send([self.tools.call("facts", id=r) for r in eids])

    # ---- judging
    def need_lines(self) -> list[str]:
        lines = []
        for r in self.requirements:
            either = [v for v in r.get("either", []) if low(v) != low(r["value"])]
            value = r["value"] + (f" (or {', '.join(either)})" if either else "")
            lines.append(f"{r['property']}: {value}")
        lines += [f"NOT: {x['property']} {x['value']}" for x in self.exclusions]
        return lines

    def entry_card(self, n: int, eid: str) -> str:
        ent_row = self.rows.get(eid) or {}
        facts = {k: str(v)[:80] for k, v in list((self.facts.get(eid) or {}).items())[:30] if len(str(v)) <= 300}
        desc = re.sub(r"<[^>]+>", " ", str(ent_row.get("description") or ""))
        return (f"{n}. {ent_row.get('title')!s} | category: {ent_row.get('category')} | price: {ent_row.get('price')} | facts: "
                f"{json.dumps(facts, ensure_ascii=False)} | description: {' '.join(desc.split())[:260]}")

    # ---- how a listing states each value, by the store's facts (what the grader reads)
    def lexicon(self) -> dict[str, dict[str, int]]:
        # Every fact label read so far, with the values listings give under it.
        if self._lexicon_for == len(self.facts):
            return self._lexicon
        out: dict[str, dict[str, int]] = {}
        listed: dict[str, list[int]] = {}
        for facts in self.facts.values():
            for k, v in facts.items():
                if len(str(v)) > 120:
                    continue        # (free text, not one value)
                seen = out.setdefault(str(k), {})
                parts = [low(p) for p in str(v).split(",") if low(p)]
                for segment in parts:
                    seen[segment] = seen.get(segment, 0) + 1
                listed.setdefault(str(k), []).append(len(parts))
        # (a label whose listings usually give several values lists features, not one value of a property)
        self._several = {k for k, counts in listed.items() if counts and sum(counts) / len(counts) > 1.3}
        self._tally_labels = {k: len(counts) for k, counts in listed.items()}
        self._lexicon, self._lexicon_for = out, len(self.facts)
        return out

    @staticmethod
    def literal_values(need_item: dict) -> list[str]:
        # The values a fact must state: the value and any accepted alternative (a seller's synonym is not the value).
        return list(dict.fromkeys([need_item["value"]] + [v for v in need_item.get("either", []) + need_item.get("same", []) if low(v) != low(need_item["value"])]))

    def labels_present(self, need_item: dict) -> set[str]:
        # The fact labels that file this property on this shelf: named after it, and giving the value (or one of its
        # alternatives) about as often as the label that gives it most. A label named after the property that seldom
        # gives such a value files something else ('style: casual' beside 'skirt style: wrap skirts').
        prop = attr_terms(need_item["property"])
        values = self.literal_values(need_item)
        share: dict[str, float] = {}
        for k, seen in self.lexicon().items():
            if not label_agrees(k, need_item["property"]):
                continue
            hits = sum(n for v, n in seen.items() if any(value_matches(v, val_text, prop) for val_text in values))
            if hits:
                share[k] = hits / max(1, self._tally_labels.get(k, 1))
        top = max(share.values(), default=0.0)
        return {k for k, s in share.items() if s >= top / 3}

    def grade(self, eid: str, need_item: dict, ban_item: bool = False) -> str:
        # How the listing states one requirement:
        # S every fact that files the property states the value (none says another);
        # T the facts that file it state the value, but another fact named after the property says something else;
        # W only the title, the description or an unrelated fact states it;
        # N nothing states it;  C a fact that files the property, or the title, states another value.
        ent_row = self.rows.get(eid) or {}
        facts = self.facts.get(eid)
        title = str(ent_row.get("title") or "")
        values = self.literal_values(need_item)
        prop = attr_terms(need_item["property"])
        if ban_item:
            # (a value ruled out is ruled out under any label: 'collar crew neck' is 'neckline: crew neck' too - but when
            # the shopper names the property, only under it: "Give regular a wide berth for the pants fit type" is no ban
            # on 'size class: regular')
            texts = [str(v) for k, v in (facts or {}).items() if len(str(v)) <= 80]
            own = [str(v) for k, v in (facts or {}).items() if need_item.get("property") and label_agrees(k, need_item["property"]) and len(str(v)) <= 80]
            if own:
                texts, title = own, ""
            # (a listing that says the thing is absent - "gluten free", "gluten-free", "no gluten", "without gluten", "low
            # gluten" - does not have it)
            absent = lambda t: re.sub(r"(?i)\b(?:%s)[\s-]+free\b|\b(?:no|without|low|zero|non)[\s-]+(?:added[\s-]+)?(?:%s)\b"  # noqa: E731
                                      % ("|".join(re.escape(low(x)) for x in values), "|".join(re.escape(low(x)) for x in values)), " ", str(t))
            texts, title = [absent(t) for t in texts], absent(title)
            match_found = any(value_matches(t, x, prop) for t in texts for x in values)
            return "C" if match_found or any(value_matches(title, x, prop) for x in values) else "S"
        model = compat_model(need_item)
        if model:
            # (what the item fits is read by code from the listing's own compatibility line, never from a sentence about
            # another part; no such line is no evidence)
            own = f"{ent_row.get('description') or ''} " + " ".join(f"{k}: {v}" for k, v in (facts or {}).items() if len(str(v)) <= 300)
            verdict = fits_model(own, model)
            return "N" if verdict is None else ("S" if verdict else "C")
        labels = self.labels_present(need_item)
        title_has = any(value_matches(title, val_text, prop) for val_text in values)
        # a title offering several numeric alternatives ('19/22/24/27') none of which is the required figure is a listing
        # of other versions (a structural reading of the title, not a list of words)
        wanted = {n for val_text in values for n in re.findall(r"\d+(?:\.\d+)?", val_text)}
        proposed = re.findall(r"\d+(?:\.\d+)?(?:\s*/\s*\d+(?:\.\d+)?)+", title)
        if wanted and proposed and not title_has and not wanted & set(re.findall(r"\d+(?:\.\d+)?", title)) \
                and not any(units_differ(title, val_text) for val_text in values):
            return "C"
        if facts is None:
            if title_has:
                return "W"
            return "W" if any(value_matches(ent_row.get("description"), val_text, prop) for val_text in values) else "N"
        named = [(k, v) for k, v in facts.items() if label_agrees(k, need_item["property"]) and len(str(v)) <= 80]
        shelved = [(k, v) for k, v in named if k in labels]
        if shelved:
            hits = [(k, v) for k, v in shelved if any(value_matches(v, val_text, prop) for val_text in values)]
            if not hits:
                # (a size written in another unit than the value's: code does not convert units - the reader judges it)
                return "N" if any(units_differ(v, val_text) for k, v in shelved for val_text in values) else "C"
            listed = {low(p) for k, v in hits for p in str(v).split(",") if low(p)}
            for k, v in shelved:
                if (k, v) in hits:
                    continue
                # ('dietary needs: organic' beside 'dietary_needs: organic, healthier choice, halal' says less, not otherwise)
                if not {low(p) for p in str(v).split(",") if low(p)} <= listed:
                    return "C"
            other_items = [(k, v) for k, v in named if k not in labels and not any(value_matches(v, val_text, prop) for val_text in values)]
            return "T" if other_items else "S"
        if named and any(value_matches(v, val_text, prop) for k, v in named for val_text in values):
            return "S"
        other = " ".join((f"{k} {v}" if not need_item.get("property") else str(v)) for k, v in facts.items() if len(str(v)) <= 80)
        if title_has or any(value_matches(other, val_text, prop) for val_text in values):
            return "W"
        return "N"

    def marks(self, eid: str) -> str:
        out = ""
        for r in self.requirements:
            m = self.grade(eid, r)
            said = (self.verdicts.get(eid) or {}).get("said", {}).get(self.need_key(r))
            if m == "N" and said == "S":
                m = "W"     # (the reader read the value in other words)
            elif said == "C" and m == "W":
                m = "C"     # (a value read only from the title or description, and the reader reads another value)
            out += m
        for x in self.exclusions:
            out += self.grade(eid, x, ban_item=True)
        return out

    @staticmethod
    def need_key(r: dict) -> str:
        return f"{low(r['property'])}={low(r['value'])}"

    def judge(self, eids: list[str]) -> None:
        # The reader says whether each listing is the kind of thing asked for, and where it reads a value the facts
        # do not state in the same words.
        fingerprint = json.dumps([self.need_lines(), self.item])
        if fingerprint != self.judged_under:
            self.verdicts, self.judged_under = {}, fingerprint
        pending_work = [r for r in eids if r not in self.verdicts]
        lines = self.need_lines()
        if not lines:
            for eid in pending_work:
                self.verdicts[eid] = {"kind": True, "marks": ""}
            return
        for start in range(0, len(pending_work), GRADE_GROUP):
            text_piece = pending_work[start: start + GRADE_GROUP]
            if self.time_remaining() < 40:
                break
            text = (f"The shopper wants: {self.item}" + (f" (store category: {self.category})" if self.category else "")
                    + "\nRequirements:\n" + "\n".join(f"{i}. {line}" for i, line in enumerate(lines, 1))
                    + "\n\nListings:\n" + "\n".join(self.entry_card(i, eid) for i, eid in enumerate(text_piece, 1)))
            answer = self.reader.complete(GRADE_BRIEF, text, GRADE_FORM, 900) or {}
            for entry in answer.get("marks") or []:
                m = re.match(r"\s*(\d+)\s*[.:)]?\s*([KX])\s*([SWNC ]*)", str(entry).upper())
                if not m:
                    continue
                index = int(m.group(1)) - 1
                if 0 <= index < len(text_piece):
                    marks = m.group(3).replace(" ", "")[: len(lines)]
                    said = {self.need_key(r): marks[i] for i, r in enumerate(self.requirements) if i < len(marks)}
                    self.verdicts[text_piece[index]] = {"kind": m.group(2) == "K", "said": said}
        for eid in pending_work:
            if eid not in self.verdicts:
                self.verdicts[eid] = {"kind": self.on_shelf_row(eid), "said": {}, "by": "code"}

    def rank_of(self, eid: str) -> int:
        # 1 every requirement filed plainly (S); 2 filed, with another label of the property saying otherwise (T);
        # 3 some requirement only in the title or another fact (W); 9 not a fit.
        v = self.verdicts.get(eid)
        # (the kind of product is met by the store's category; whether a listing on it is only a part or an accessory for
        # something else is the final check's - the reader's - judgement)
        on_shelf = bool(v and self.category and low((self.rows.get(eid) or {}).get("category")) == low(self.category))
        type_fits = bool(v and v.get("kind")) or on_shelf
        if not v or not type_fits or eid in self.gone or eid in self.rejected or not self.within_cap(eid) or not self.obtainable(eid):
            return 9
        if not self.on_shelf_row(eid):
            return 9
        marks = self.marks(eid)
        # (a requirement the request does not insist on ranks listings but rules none out)
        need = "".join(m for m, r in zip(marks, self.requirements) if r.get("must", True))
        if "C" in need or "N" in need or "C" in marks[len(self.requirements):]:
            return 9
        if eid not in self.facts or "W" in need:
            return 3
        return 2 if "T" in need else 1

    def fits(self) -> list[str]:
        # Judged fits, best first: plainly stated before weakly; then the lower price.
        good = [eid for eid in self.verdicts if self.rank_of(eid) < 9]
        price = lambda eid: self.rows[eid].get("price") or 0  # noqa: E731
        soft = [i for i, r in enumerate(self.requirements) if not r.get("must", True)]

        def hoped(eid: str) -> int:
            marks = self.marks(eid)
            return -sum(1 for i in soft if i < len(marks) and marks[i] in "ST")
        if self.objective:
            # (the best by the measure among the fits, within the budget; a listing that does not state it last; the
            # cheaper of equals)
            sign = -1 if self.objective["kind"] == "most" else 1

            def measure(eid: str) -> tuple:
                v = self.objective_value(eid)
                return (v is None, sign * (v or 0.0))
            return sorted(good, key=lambda eid: (self.rank_of(eid) >= 3, measure(eid), price(eid), self.rank_of(eid)))
        if self.cheapest:
            # (the cheapest that is filed as asked; a title-only fit only when none is)
            return sorted(good, key=lambda eid: (self.rank_of(eid) >= 3, price(eid), self.rank_of(eid)))
        return sorted(good, key=lambda eid: (self.rank_of(eid), hoped(eid), price(eid)))

    # ---- questions
    def label_digest(self) -> str:
        labels: dict[str, dict[str, int]] = {}
        for eid, facts in self.facts.items():
            if not self.on_shelf_row(eid):
                continue
            for k, v in facts.items():
                name = label_tokens(k)
                if not name or len(str(v)) > 120:
                    continue
                seen = labels.setdefault(name, {})
                for value in str(v).split(",")[:3]:
                    value = low(value)[:30]
                    if value:
                        seen[value] = seen.get(value, 0) + 1
        rank_list = sorted(labels.items(), key=lambda pair_kv: (-sum(pair_kv[1].values()), pair_kv[0]))[:18]
        return "\n".join(f"- {name} ({sum(val_texts.values())}): " + ", ".join(v for v, _ in sorted(val_texts.items(), key=lambda pair_kv: -pair_kv[1])[:5])
                         for name, val_texts in rank_list) or "(none read)"

    def pick_questions(self, count: int) -> list[str]:
        known = [r["property"] for r in self.requirements] + [x["property"] for x in self.exclusions]
        text = (f"Buyer's message:\n{self.request}\n\nGoods: {self.item}; shelf: {self.category or '?'}\n"
                f"Conditions so far: {self.need_lines() or '(none)'}\nQuestioned already: {self.asked or '(nothing)'}\n"
                f"Labels across {len([r for r in self.facts if self.on_shelf_row(r)])} entries of this type:\n{self.label_digest()}\n"
                f"Initial guesses: {self.ask_first}")
        answer = self.reader.complete(QUESTION_BRIEF, text, QUESTION_FORM, 500) or {}
        picked_set = [str(n).strip() for n in answer.get("properties") or [] if str(n).strip()]
        rank_list = self.diverse_labels()
        # (the reader's first pick - what the story hints at - then the labels most listings of this shelf carry with
        # values that differ between them, then the reader's other picks)
        names = picked_set[:1] + rank_list[:3] + picked_set[1:] + rank_list[3:] + self.ask_first
        out: list[str] = []
        for name in names:
            name = re.sub(r"[._]+", " ", name).strip().lower()
            if not name or len(name.split()) > 4:
                continue
            def same(a: str) -> set[str]:
                return {"color" if w == "colour" else w for w in roots(a)}
            if any(same(name) <= same(k) or same(k) <= same(name) for k in known + self.asked + out if roots(k)):
                continue
            if re.search(r"price|budget|stock|brand|category|quantity", name):
                continue        # (terms of the purchase, not properties of the item)
            out.append(name)
            if len(out) >= count:
                break
        return out

    def diverse_labels(self) -> list[str]:
        # the shelf's fact labels, those that split the listings read into real choices first (from the listings
        # themselves: a label every listing gives the same value cannot hold a choice); a label whose words contain
        # another's is that one ('item pattern' and 'pattern')
        borne: dict[frozenset, int] = {}
        values: dict[frozenset, set[str]] = {}
        label_of: dict[frozenset, str] = {}
        for eid, facts in self.facts.items():
            if not self.on_shelf_row(eid):
                continue
            spotted_here: set[frozenset] = set()
            for k, v in facts.items():
                if len(str(v)) > 60 or not str(v).strip():
                    continue
                key = frozenset(label_terms(k))
                if not key:
                    continue
                if key not in spotted_here:
                    borne[key] = borne.get(key, 0) + 1
                    spotted_here.add(key)
                values.setdefault(key, set()).add(low(v))
                name = " ".join(label_terms(k))
                if key not in label_of or len(name) < len(label_of[key]):
                    label_of[key] = name
        keys = list(borne)
        for a in keys:          # (fold a longer label into the shorter one it contains)
            for b in keys:
                if a is not b and a in borne and b in borne and a < b:
                    borne[a] += borne.pop(b)
                    values[a] |= values.pop(b, set())
        # (a label that splits the shelf into several real choices first: carried by many listings, with several values
        # each shared by several listings - on the race #4 shelves this held the shopper's need in its first five
        # three times in four, against two in three by count alone)
        n = max(1, sum(1 for eid in self.facts if self.on_shelf_row(eid)))
        rank_list = sorted(borne, key=lambda k: (-(borne[k] / n) * min(len(values.get(k, ())), borne[k] / max(1, len(values.get(k, ())))), label_of[k]))
        return [label_of[k] for k in rank_list]

    def question_text(self, prop: str) -> str:
        # One short plain question about one property: its name only, no example values, no second question.
        prop = " ".join(prop.split()).strip(" ?.").lower()
        if re.match(r"(?:who|what|where|when|how|which)\b", prop):
            return ASK_TEMPLATE_FREE.format(prop)       # ('who it is for' -> 'Could you let me know who it is for?')
        if self.products:
            return ASK_TEMPLATE_EACH.format(self.item, prop)      # (several products: the question names the one it is about)
        return ASK_TEMPLATE.format(prop)

    def ask(self, cap: int | None = None) -> None:
        # Put the questions the message budget allows, one property each; then read the answers.
        if not self.tools.has("message") or self.messages_closed or self.hurry():
            return      # (a shopper's reply is slow: no question in a hurry)
        headroom = min(self.questions_remaining(), max(0, self.spare_moves() - (self.scaled(4, 2) if self.products else self.scaled(8, 3))))
        if cap is not None:
            headroom = min(headroom, cap)
        if headroom <= 0:
            return
        properties = self.pick_questions(headroom)
        if not properties:
            return
        lines: list[str] = []
        backlog = list(properties)
        while backlog and headroom > 0 and not self.finished:
            prop = backlog.pop(0)
            self.asked.append(prop)
            text = self.question_text(prop)
            call_result = self.send([self.say(text)])
            if not call_result:
                break
            headroom -= 1
            reply = self.answer_text(call_result[0])
            lines.append(f"Asked: {text}\nShopper: {reply}")
            # (a reply such as "you've lost me" or "one thing at a time" is how these shoppers answer a property they
            # hold nothing on: in race #6 the same property asked again never drew a requirement, so none is re-asked)
            if self.messages_closed:
                break
        self.note("answers: " + " || ".join(l.replace("\n", " ") for l in lines))
        self.hear(lines, answers=True)

    @staticmethod
    def answer_text(call_result: dict) -> str:
        core = call_result.get("core")
        if isinstance(core, dict) and len(core) == 1 and isinstance(next(iter(core.values())), str):
            core = next(iter(core.values()))      # (a reply wrapped as {"value": "..."})
        return core if isinstance(core, str) else json.dumps(core)[:400] if core is not None else ""

    def hear(self, lines: list[str], answers: bool = False) -> bool:
        # Take in the shopper's lines (answers, notices, refusals). True when what the shopper wants changed.
        # (a world repeats a notice with the next observation: a line already taken in changes nothing again)
        lines = [l for l in lines if l.startswith("Asked:") or low(l) not in self.reply_lines]
        self.reply_lines |= {low(l) for l in lines}
        if not lines:
            return False
        before = json.dumps([self.need_lines(), self.budget, sorted(self.gone)])
        known = self.need_lines()
        text = (f"Goods: {self.item}\nConditions so far: {known or '(none)'}\nSpending ceiling: {self.budget}\n"
                f"Basket holds: {[(self.rows.get(r) or {}).get('title') for r in self.cart] or '(none)'}\n\nFresh replies:\n" + "\n".join(lines))
        text += f"\n\nMessages sent so far: {self.msgs_out} of {self.message_budget}; steps used: {self.step} of {self.max_steps}"
        answer = self.reader.complete(REPLY_BRIEF, text, REPLY_FORM, 600)
        coded_hear = self.parse_reply(lines)
        if answer is None:
            answer = coded_hear
        else:
            # (a value an answer gives in one of the shopper's set forms that the reader missed is still taken)
            answer = dict(answer)
            answer["adds"] = list(listify(answer.get("adds"))) + [
                a for a in coded_hear["adds"] if not any(isinstance(b, dict) and attr_terms(b.get("property")) == attr_terms(a["property"])
                                                         for b in listify(answer.get("adds")))]
        adds, refs = split_pointers(answer.get("adds") or [], "\n".join(lines), False)
        for line in lines:
            asked = re.search(r"Asked:.*?(?:what|which)\s+(.+?)\s+(?:do you want|would you like)", line, re.I)
            for c in refs_by_code(line.split("Shopper:", 1)[-1], False):
                if not c["property"] and asked:
                    c["property"] = asked.group(1).strip()      # (the property we asked about)
                if c["property"] and not any(low(c["ref"]["seller"]) == low(x["ref"]["seller"]) for x in refs):
                    refs.append(c)
        self.refs += refs
        for add in adds:
            if isinstance(add, dict) and str(add.get("value") or "").strip():
                bound_value, why = bind_value(add, "\n".join(l.split("Shopper:", 1)[-1] for l in lines))
                if why:
                    self.note(f"{add.get('property')}: {why}")
                prop, value = str(add.get("property") or "").strip(), bound_value.strip()
                if re.search(r"no preference|any|whatever|not picky|don't mind", low(value)):
                    continue
                new = {"property": prop, "value": value, "either": [], "stated": False, "must": True,
                       "same": [str(a) for a in listify(add.get("same_meaning")) if str(a).strip()][:8]}
                # (the product the answer names, as the reader gives it - not always the one asked about)
                whose = str(add.get("product") or "")
                idx = self.product_index(whose) if whose and self.products else None
                if idx is not None and idx != self.current:
                    other = self.products[idx]
                    other["requirements"] = [r for r in other["requirements"] if attr_terms(r["property"]) != attr_terms(prop)] + [new]
                    self.note(f"answer for {other['item']}: {prop} {value}")
                    continue
                self.requirements = [r for r in self.requirements if attr_terms(r["property"]) != attr_terms(prop)]
                self.requirements.append(new)
        for x in listify(answer.get("excludes")):
            if not isinstance(x, dict) or not str(x.get("value") or "").strip():
                continue
            ex = {"property": str(x.get("property") or "").strip(), "value": str(x["value"]).strip(),
                  "same": [str(a) for a in listify(x.get("same_meaning")) if str(a).strip()][:8]}
            idx = self.product_index(str(x.get("product") or "")) if x.get("product") and self.products else None
            if idx is not None and idx != self.current:
                self.products[idx]["exclusions"].append(ex)
            elif not any(low(e["value"]) == low(ex["value"]) for e in self.exclusions):
                self.exclusions.append(ex)
        # (an answer to our question withdraws nothing: "pick whichever" about the neckline is no leave to break the
        # request's own "not crew neck"; withdrawals come in the shopper's notices)
        for toss in [] if answers else answer.get("drops") or []:
            d = attr_terms(toss) - PLAIN_LABEL or attr_terms(toss)
            if not d:
                continue
            for i, p in enumerate(self.products):
                if i != self.current:      # (a dropped requirement may be another product's)
                    p["requirements"] = [r for r in p["requirements"] if not (d & attr_terms(r["property"] + " " + r["value"]))]
                    p["exclusions"] = [x for x in p["exclusions"] if not (d & attr_terms(x["property"] + " " + x["value"]))]
            self.requirements = [r for r in self.requirements if not (d & attr_terms(r["property"] + " " + r["value"]))]
            self.exclusions = [x for x in self.exclusions if not (d & attr_terms(x["property"] + " " + x["value"]))]
        asked_step = [said for said in (l.split("Shopper:", 1)[-1] for l in lines) if CONSENT_CUE.search(said)]
        if asked_step and not self.before_order:
            self.before_order = [f"as the shopper asked: {asked_step[0][:80]}"]
            self.note(f"a step the shopper asked for mid-task: {asked_step[0][:120]}")
            self.margin = self.finish_reserve()
            self.plan_messages("a step asked mid-task")
        coded = self.limits_by_code(lines)
        ml, sl = number(answer.get("message_limit")), number(answer.get("step_limit"))
        said = " ".join(low(l) for l in lines if not l.startswith("Asked:"))
        ml = coded[0] if coded[0] is not None else (int(ml) if ml is not None and ml >= 0 and re.search(r"messag|question|texts?\b|ask me", said) else None)
        sl = coded[1] if coded[1] is not None else (int(sl) if sl is not None and sl > 0 and re.search(rf"\b{int(sl)}\b", " ".join(lines)) else None)
        if ml is not None or sl is not None:
            self.set_limits(ml, sl, "(the shopper's line)")
        budget = number(answer.get("budget"))
        # (the amount as the shopper wrote it, in the request's unit: a reader may convert cents to dollars)
        printed = re.findall(r"(\d[\d,]*(?:\.\d+)?)\s*(?:cts|cents?|\$|dollars?|usd|eur|€|£)", low(" ".join(lines)))
        if budget and printed and not any(abs(budget - (number(w) or 0)) < 1e-6 for w in printed):
            budget = number(printed[-1])
        if budget and budget > 0:
            self.budget = budget
        if answer.get("gone") and not answers:
            if self.products and len(self.cart) > 1:
                self.stock_doubt = True          # (which line sold out is read from the listings)
            else:
                for eid in list(self.cart):
                    self.gone.add(eid)
        after = json.dumps([self.need_lines(), self.budget, sorted(self.gone)])
        if after != before:
            self.note(f"replies read: wants={self.need_lines()} cap={self.budget} sold_out={sorted(self.gone)}")
        return after != before

    def limits_by_code(self, lines: list[str]) -> tuple[int | None, int | None]:
        # A line that changes the message or step limit, read by code: "you can ask me 2 more questions" (more: added to
        # those sent), "only 3 messages in all now", "you've got 20 more moves", "steps are capped at 30 now".
        # (Answers to our own questions are skipped: "I've asked two questions myself" is no limit.)
        text = " ".join(low(l.split("Shopper:", 1)[-1]) for l in lines if not l.startswith("Asked:"))
        numerals = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "a single": 1, "no": 0, "zero": 0}
        val_text = lambda v: int(v) if v.isdigit() else numerals.get(v)  # noqa: E731
        msgs = steps = None
        m = re.search(r"(\d+|one|two|three|four|five|six|a single|no)\s+more\s+(?:messages?|questions?|texts?)", text)
        if m and val_text(m.group(1)) is not None:
            msgs = self.msgs_out + val_text(m.group(1))
        else:
            m = self.msg_cap_from_prose(text) if re.search(r"\b(?:now|instead|change|only|at most|in all|total)\b", text) else None
            msgs = m
        m = re.search(rf"(\d+)\s+more\s+{STEP_WORDS}", text)
        if m:
            steps = self.step + int(m.group(1))
        elif re.search(r"\b(?:now|instead|change|only)\b", text):
            steps = self.step_cap_from_prose(text)
        return msgs, steps

    def parse_reply(self, lines: list[str]) -> dict:
        text = low(" ".join(lines))
        out: dict[str, Any] = {"adds": [], "drops": [], "budget": 0, "gone": False}
        if re.search(r"(?<!price )\bgone\b(?! up| down)|sold out|zero left|nothing left|none left|unavailable|can't be ordered|cannot be ordered|ran dry|dried up|out of stock|grayed out|got bought|beat me to it|no longer|cleared out|shelves are bare|last of those", text):
            out["gone"] = True
        m = re.search(r"(\d[\d,]*(?:\.\d+)?)\s*(?:cts|cents|\$|dollars|usd)", text)
        if m and re.search(r"spend|budget|working with|afford|money|limit|can pay|capping|cap it|call it|most i can|throw at|in the jar|ceiling", text):
            out["budget"] = number(m.group(1))
        if re.search(r"never mind|forget|skip|drop|not worth|isn't worth|don't care|doesn't matter|nice-to-have|nice to have|no longer matters", text):
            for r in self.requirements + self.exclusions:
                if holds(text, r["property"]) or holds(text, r["value"]):
                    out["drops"].append(r["property"])
        return out

    # ---- looking
    def inspect(self) -> None:
        # Questions first: one search and one read of facts show the shelf's labels, the questions go out, and the
        # store is searched with what the answers told.
        # (race #5: asking once one search was in passed 0.65, against 0.39 asking after the store was searched)
        self.search(self.lookup_ideas()[:1] or [self.item])
        self.fix_shelf()
        self.probe_shelf()
        self.fix_shelf()
        self.plan_searches()
        # (race #6: the leaders asked after 5 to 8 listings' facts, and named the shopper's held-back property twice as
        # often; with steps to spare, seven are read so the shelf's labels are fuller when the questions are chosen)
        wide = max(3, min(9, self.scaled(7)))
        top = [r for r in self.candidates() if self.hopeful(r)][: min(wide, max(0, self.spare_moves() - self.scaled(12, 4)))]
        self.load_facts(top)
        self.ask()
        if self.queued_lines:
            lines, self.queued_lines = self.queued_lines, []
            self.hear(lines)
        self.trace_pointers()
        self.search([cond_item for cond_item in self.lookup_ideas() if not self.issued_lookup(cond_item)][: min(3, max(0, self.spare_moves() - self.scaled(8, 3)))])
        self.fix_shelf()
        shelf_member = [eid for eid in self.rows if self.on_shelf_row(eid)]
        if self.category and len(shelf_member) < 5 and self.spare_moves() > self.scaled(14, 6):
            self.shelf_lookup(self.budget)
        self.note(f"shelf={self.category!r} seen={len(self.rows)} on_shelf={len([r for r in self.rows if self.on_shelf_row(r)])} "
                  f"lookups={list(self.issued_lookups.values())}")
        self.judge([r for r in top if r in self.facts])
        self.dig(want=1 if self.cheapest else 3)

    def hopeful(self, eid: str) -> bool:
        # A row worth its facts: on the shelf, on sale, affordable, no value contradicted, some value named.
        if not (self.on_shelf_row(eid) and self.obtainable(eid) and self.within_cap(eid)) or eid in self.facts:
            return False
        marks = [self.grade(eid, r) for r in self.requirements]
        if "C" in marks:
            return False
        return not self.requirements or self.proof(eid) > 0

    def proof(self, eid: str) -> int:
        # How many requirements a row shows signs of before its facts are read: named in its title or description,
        # or matched by a search that carried the value (a search keeps only listings holding every word, facts too).
        hits = self.value_matches_all.get(eid, set())
        return sum(1 for r in self.requirements if self.need_key(r) in hits or self.grade(eid, r) in ("S", "T", "W"))

    def lookup_ideas(self) -> list[str]:
        # Narrow searches first: results come in id order, so a search matching fewer listings shows more fits.
        need_items = sorted(self.requirements, key=lambda r: (not r.get("must", True), r.get("stated", True), len(r["value"])))
        # (the item's main word: "hoodie or sweatshirt" -> hoodie, sweatshirt; "fall/winter clothing" names a shelf, not
        # what titles say - its other words stay out of searches)
        alts = [a.strip() for a in re.split(r"\s+or\s+|,", low(self.item)) if a.strip()] or [""]
        heads = [a.split()[-1] for a in alts if a.split()]
        lead_part = heads[0] if heads else ""
        suggestions: list[str] = []

        def lookup_ready(r: dict) -> str:
            # (a limit is no word a listing carries: its property is searched instead; a model to fit is written as a
            # family by the store - 'every VIT-80xx model' - so it is not searched)
            if compat_model(r):
                return ""
            return r["property"] if limit_of(r["value"]) else r["value"]
        titles = set(roots(" ".join(str(ent_row.get("title") or "") for ent_row in self.rows.values()), hold_filler=True))
        for r in need_items:
            v = lookup_ready(r)
            if not v:
                continue
            # (a property word the store's own titles use beside values: 'sleeve' is in titles, 'length' rarely)
            attr_tokens = [w for w in words(r["property"]) if w not in PLAIN_LABEL and stem(w) in titles]
            if len(words(v)) == 1 and attr_tokens and not limit_of(r["value"]):
                suggestions.append(f"{v} {attr_tokens[0]} {lead_part}")     # (one word alone matches too much: 'long' -> 'long sleeve top')
            for value in [r["value"]] + r.get("either", [])[1:] + r.get("same", [])[:3]:
                if limit_of(value):
                    value = r["property"]
                # (titles write the singular of a plural value: "long sleeve" for "long sleeves")
                one = re.sub(r"(?<=[a-z]{3})s$", "", value) if not re.search(r"(?:ss|us|is)$", value) else value
                suggestions += [f"{v} {h}" for v in dict.fromkeys([value, one]) for h in heads[:2]] + [f"{value} {self.item}", value]
        plain = [r for r in need_items if lookup_ready(r)]
        if len(plain) >= 2:
            suggestions.insert(0, f"{lookup_ready(plain[0])} {lookup_ready(plain[1])} {lead_part}")
        elif plain and len(plain) < len(need_items):
            suggestions.insert(0, f"{lookup_ready(plain[0])} {lead_part}")
        planned = list(self.queries) if self.current in (getattr(self, "_planned", None) or set()) else []
        suggestions = self.hint_queries() + planned + suggestions + self.queries + self.extra_queries
        out, seen = [], set()
        for cond_item in suggestions:
            cond_item = " ".join(dict.fromkeys(cond_item.split()))[: self.tools.text_cap()]     # (a word once: 'hoodie cotton hoodie')
            key = " ".join(sorted(set(roots(cond_item))))
            if key and key not in seen:
                seen.add(key)
                out.append(cond_item)
        return out

    ELSEWHERE = re.compile(r"(?:an?|versions?)\s+(.{2,120}?)\s+(?:version\s+)?exists?\s+elsewhere", re.I)

    def hint_queries(self) -> list[str]:
        # Race #13 listings point at their other versions ("A 1.77 inch display version from Zenridorqi exists elsewhere",
        # 220 of the remarks read): one that names a required value is searched by its seller, else by that value.
        out: list[str] = []
        lead_part = (words(self.item) or [""])[-1]
        for eid in list(self.rows)[-80:]:
            for m in self.ELSEWHERE.finditer(str((self.rows.get(eid) or {}).get("description") or "")):
                cue = m.group(1)
                if not any(value_matches(cue, v) for r in self.requirements for v in self.literal_values(r)):
                    continue
                seller = re.search(r"\bfrom\s+([A-Z][\w'-]+)", cue) or re.match(r"([A-Z][\w'-]+)\s+version", cue)
                plain = re.sub(r"(?i)\b(?:with|in|and|a|an)\b", " ", cue)     # (outside the f-string: Python < 3.12)
                out.append(f"{seller.group(1)} {lead_part}" if seller else f"{plain} {lead_part}")
        return out[:3]

    def following_page_call(self) -> dict | None:
        # The next page of the search whose shown rows were most often promising (narrow searches first).
        page_param = self.tools.arg.get(("search", "page"))
        if not page_param:
            return None
        best, top_rating = None, -1e9
        for key, total in self.issued_lookups.items():
            following = self.following_params(key)
            if following is None or total is None:
                continue
            worthwhile = self.lookup_yield.get(key, 0)
            score = worthwhile * 3 - total / 50.0
            if score > top_rating:
                best, top_rating = {"name": self.tools.role[self.lookup_role.get(key, "search")], "args": following}, score
        return best if top_rating > -6 else None

    def following_params(self, key: str, unseen: bool = True) -> dict | None:
        # The arguments for the page after a search sent: the store's own next-page value (a page number, an
        # offset or a cursor) put in the paging argument; None when the store gave none or it was sent already.
        role = self.lookup_role.get(key, "search")
        page_param = self.tools.arg.get((role, "page"))
        following = self.following_of.get(key)
        if not page_param or following is None or isinstance(following, bool):
            return None
        schema = self.tools.schemas.get(self.tools.role.get(role, ""), {})
        prop = (Toolset._attrs(schema).get(page_param) or {})
        kind = prop.get("type")
        try:
            value = int(following) if kind in ("integer", "number") else str(following)
        except (TypeError, ValueError):
            return None
        if isinstance(value, int) and "maximum" in prop and value > prop["maximum"]:
            return None
        args = {**json.loads(key), page_param: value}
        return None if unseen and json.dumps(args, sort_keys=True) in self.issued_lookups else args

    def trace_pointers(self) -> None:
        # A value tied to another listing's page: code finds that page (seller, kind, details) and follows its "For
        # <property>, use / refer to / look to ..." pointers until a page states the value itself; the reader steps in
        # only where a page's words are not in a known form. "pairs" takes the partner by the store's pairing rule,
        # "not" rules the value out.
        # (the chains share a budget: about 75% of the steps free for looking, at least 8 - a wrong value fails the task,
        # but the item itself still has
        # to be found and bought)
        self.ref_until = self.step + max(14, int(self.spare_moves() * 0.9))     # (pages are read whole to be weighed: a few more)
        while self.refs and not self.finished and not self.hurry():
            r = self.refs.pop(0)
            prop, rel = r["property"], r["ref"].get("relation", "same")
            value, same, path = self.chain_value(r["ref"], prop)
            if value and rel == "pairs":
                self.note(f"pairing rule for {prop}: the reader gives the partner {value}")
            self.note(f"listing reference for {prop} ({r['ref']['seller']}, {rel}): {value or 'not found'} via {path}")
            if not value:
                continue
            if rel == "within":
                # ("within 0.51 inches of whatever you land on ... on either side": that page's figure, give or take)
                fig = re.search(r"(\d+(?:\.\d+)?)\s*([a-z\"']*)", low(value))
                tol = number(r["ref"].get("tolerance")) or 0.0
                if not fig:
                    self.note(f"listing reference for {prop}: {value!r} has no figure to take {tol:g} either side of")
                    continue
                v, unit, tu = float(fig.group(1)), fig.group(2), low(r["ref"].get("unit") or "")
                if tu and unit and unit_key(tu) != unit_key(unit):
                    self.note(f"listing reference for {prop}: the page says {value!r}, the tolerance is in {tu}: left to the reader")
                    continue        # (no conversion table in code: a tolerance in another unit is not applied)
                value, same = f"between {v - tol:g} and {v + tol:g} {unit or tu}".strip(), []
            if rel == "not":
                self.exclusions.append({"property": prop, "value": value, "same": same})
                continue
            self.requirements = [x for x in self.requirements if attr_terms(x["property"]) != attr_terms(prop)]
            self.requirements.append({"property": prop, "value": value, "same": same, "either": [], "stated": r["stated"], "must": True})

    def chain_value(self, ref: dict, prop: str) -> tuple[str, list[str], list[str]]:
        # (the value, other ways of writing it, the pages read) at the end of a page's pointers for this property
        path: list[str] = []
        while ref and len(path) < 6:
            if self.spare_moves() < 3 or self.time_remaining() < 50 or self.hurry() or self.step >= getattr(self, "ref_until", 10 ** 9):
                break
            eids = [x for x in self.locate_entry(ref, prop) if x not in path]
            tied = self.best_matches(eids, ref)
            eid = tied[0] if tied else None
            if not eid:
                break
            # (the reader reads each page: whether it states the value, sends you on to another listing, or - for a pairing -
            # states the partner; code reads no page wording)
            rel = ref.get("relation", "same")
            meant = f"{ref.get('item')} sold by {ref.get('seller')}" + (f", with {', '.join(ref.get('details') or [])}" if ref.get("details") else "")
            text = (f"Property: {prop}\nRelation: {rel}\nListing meant: {meant}\nListings found:\n"
                    + "\n".join(self.whole_card(i, r) for i, r in enumerate(tied, 1)) + "\n")
            answer = self.reader.complete(TRACE_BRIEF, text, TRACE_FORM, 500)
            pick = int(number((answer or {}).get("listing")) or 1)
            eid = tied[pick - 1] if 1 <= pick <= len(tied) else eid       # (the page meant, by the reader's reading of its details)
            path.append(eid)
            if answer is None:
                # (no reader: the page's own fact for the property, for a plain match only)
                own = [str(v) for k, v in (self.facts.get(eid) or {}).items() if label_agrees(k, prop) and len(str(v)) <= 80]
                return (own[0], [], path) if own and rel != "pairs" else ("", [], path)
            value = str(answer.get("value") or "").strip()
            if value:
                return value, [str(a) for a in listify(answer.get("same_meaning")) if str(a).strip()][:8], path
            following = entry_pointer(answer.get("next"))
            if following:
                following["relation"] = rel
            ref = following
        return "", [], path

    def best_matches(self, eids: list[str], ref: dict) -> list[str]:
        # The pages that could be the one meant: the seller's, with the most details code can confirm, best first (up to
        # eight). Details code cannot confirm - a size in another unit - are the reader's to weigh.
        def score(eid: str) -> tuple:
            ent_row = self.rows.get(eid) or {}
            return (low(ent_row.get("seller")) == low(ref["seller"]), self.detail_hits(eid, ref))
        rank_list = sorted(eids, key=score, reverse=True)
        if not rank_list:
            return []
        top = score(rank_list[0])
        return [r for r in rank_list if score(r) == top][:8]

    def best_match(self, eids: list[str], ref: dict, prop: str = "") -> str | None:
        # The page meant: its seller and kind, then the most details its facts and notes state (units converted, joined
        # and split words alike: "1.00584 m" is "3.3 ft", "non-woven" is "nonwoven").
        def score(eid: str) -> tuple:
            ent_row = self.rows.get(eid) or {}
            says = bool(prop) and (bool(pointer_for(ent_row.get("description"), prop))
                                   or any(label_agrees(k, prop) for k in (self.facts.get(eid) or {})))
            return (low(ent_row.get("seller")) == low(ref["seller"]), self.detail_hits(eid, ref), says, eid in self.facts)
        rank_list = sorted(eids, key=score, reverse=True)
        if not rank_list:
            return None
        if prop:
            # (a chain ends on a page that states the property or sends you on for it: among pages within one detail of
            # the best, such a page wins over one that lists every option - "Aged Bronze, Black, ..., Satin Nickel")
            top = score(rank_list[0])[1]
            near = [r for r in rank_list if score(r)[0] == score(rank_list[0])[0] and score(r)[1] >= max(1, top - 1) and score(r)[2]]
            if near:
                return near[0]
        return rank_list[0]

    def partner_of(self, value: str, prop: str, path: list[str]) -> str:
        # The value's partner by the store's pairing rule: one stated on a page of the chain first, then one stated on that
        # shelf, then any for the property; a few more of the shelf's listings are read when none has been seen.
        def inspect(pairs: dict[str, str]) -> str:
            for a, b in pairs.items():
                if value_matches(value, a) and value_matches(a, value):
                    return b
            return ""
        for eid in reversed(path):
            for p, pairs in (self.rule_pages.get(eid) or {}).items():
                if p == "*" or label_agrees(p, prop) or attr_terms(p) & attr_terms(prop):
                    match_found = inspect(pairs)
                    if match_found:
                        return match_found
        shelves = [low((self.rows.get(eid) or {}).get("category") or "") for eid in reversed(path)]
        for tries in range(2):
            for (shelf, p), pairs in self.rules.items():
                if (shelf in shelves or tries) and (p == "*" or label_agrees(p, prop) or attr_terms(p) & attr_terms(prop)):
                    match_found = inspect(pairs)
                    if match_found:
                        return match_found
            if tries == 0 and shelves and self.spare_moves() >= 4 and self.step < getattr(self, "ref_until", 10 ** 9):
                unread = [eid for eid, ent_row in self.rows.items() if low(ent_row.get("category") or "") == shelves[0] and eid not in self.facts][:3]
                self.load_facts(unread)
        return ""

    def locate_entry(self, ref: dict, prop: str = "") -> list[str]:
        # The listings of that seller and kind: searched by the seller with the kind, then by the seller alone, then by
        # the kind with its details (a store that does not search sellers); the likeliest three read whole.
        seller = low(ref["seller"])
        kind = set(roots(ref["item"]))
        known = [eid for eid, ent_row in self.rows.items() if low(ent_row.get("seller") or "") == seller
                 and (not kind or kind & set(roots(f"{ent_row.get('category')} {ent_row.get('title')}")))]
        found: list[str] = known
        # (one search for the seller even when some of its listings were seen: those seen came up for other words)
        for cond_item in (f"{ref['seller']} {ref['item']}", ref["seller"], " ".join([ref["item"]] + [d.split()[-1] for d in ref["details"][:2]])):
            c = self.tools.call("search", mode="typed", text=" ".join(cond_item.split())[: self.tools.text_cap()])
            if not c or self.spare_moves() < 3:
                break
            if json.dumps(c["args"], sort_keys=True) in self.issued_lookups:
                continue
            call_result = self.send([c])
            shown = [ent_row["id"] for x in call_result for ent_row in entries_in(x["core"])]
            found = list(dict.fromkeys(known + [eid for eid in shown if low((self.rows.get(eid) or {}).get("seller") or "") == seller]))
            if not found and not any((self.rows.get(eid) or {}).get("seller") for eid in shown):
                found = [eid for eid in shown if kind & set(roots(str(self.rows[eid].get("category")) + " " + str(self.rows[eid].get("title"))))]
            if found:
                break

        def fit(eid: str) -> tuple:
            # (before a page is read, its search row: a detail's figure in the title or description - "3.3 ft" for
            # "cable length 1.00584 m" - or all its words)
            ent_row = self.rows[eid]
            text = f"{ent_row.get('title')} {ent_row.get('description')}"
            shown = sum(1 for d in ref["details"] if value_matches(text, d) or (quantities(d) and same_amounts(text, re.sub(r"^[^\d]*", "", d))))
            return (-len(kind & set(roots(f"{ent_row.get('category')} {ent_row.get('title')}"))), -shown, -self.detail_hits(eid, ref), eid not in self.facts)
        # (the details that tell the page apart - "480 mbps", "1.00584 m" - show only on the page itself: candidates are
        # read three at a time until one states them all, at most nine)
        rank_list = sorted(found, key=fit)
        read: list[str] = []
        pages = 0
        while ref["details"] and len(rank_list) < 12 and pages < 2 and self.spare_moves() > 4 and self.step < getattr(self, "ref_until", 10 ** 9) - 4:
            # (a seller with more listings than a page shows: the next pages of the seller's search too)
            following = [self.following_params(k) for k in list(self.issued_lookups) if ref["seller"].lower() in k.lower()]
            following = [a for a in following if a]
            if not following:
                break
            call_result = self.send([{"name": self.tools.role["search"], "args": following[0]}])
            pages += 1
            more = [ent_row["id"] for x in call_result for ent_row in entries_in(x["core"]) if low(ent_row.get("seller") or "") == seller]
            if not more:
                break
            rank_list = sorted(dict.fromkeys(rank_list + more), key=fit)
        for start in range(0, min(12, len(rank_list)), 3):
            if self.step >= getattr(self, "ref_until", 10 ** 9) - 2:
                break
            batch = rank_list[start: start + 3]
            self.load_facts([x for x in batch if x not in self.facts], margin=max(2, self.margin - 2))
            read += batch
            best = self.best_match(read, ref, prop) or ""
            says = not prop or bool(pointer_for((self.rows.get(best) or {}).get("description"), prop)) \
                or any(label_agrees(k, prop) for k in (self.facts.get(best) or {}))
            if (not ref["details"] or self.detail_hits(best, ref) >= len(ref["details"])) and says:
                break       # (the page meant: every detail, and it says something about the property)
            if self.spare_moves() < 4:
                break
        if ref["details"] and self.detail_hits(self.best_match(read, ref, prop) or "", ref) < len(ref["details"]):
            # (none states every detail: the seller with a detail's own words - "Korarira nonwoven fabric")
            for d in ref["details"][:2]:
                if self.step >= getattr(self, "ref_until", 10 ** 9) - 3:
                    break
                word_items = " ".join(low(d).split()[-3:])
                c = self.tools.call("search", mode="typed", text=f"{ref['seller']} {word_items}"[: self.tools.text_cap()])
                if not c or self.spare_moves() < 4 or json.dumps(c["args"], sort_keys=True) in self.issued_lookups:
                    continue
                call_result = self.send([c])
                more = [ent_row["id"] for x in call_result for ent_row in entries_in(x["core"]) if low(ent_row.get("seller") or "") == seller and ent_row["id"] not in read][:3]
                self.load_facts(more, margin=max(2, self.margin - 2))
                read += more
                if self.detail_hits(self.best_match(read, ref, prop) or "", ref) >= len(ref["details"]):
                    break
        return [eid for eid in read if eid in self.facts] or read

    def detail_hits(self, eid: str, ref: dict) -> int:
        # How many of the details named ("material non-woven fabric", "cable length 1.00584 m", "2 drawers") the page
        # states: against the fact of that name first ("material" -> "nonwoven fabric"), else anywhere on the page.
        ent_row = self.rows.get(eid) or {}
        facts = self.facts.get(eid) or {}
        text = f"{ent_row.get('title')} {ent_row.get('description')} " + " | ".join(f"{k} {v}" for k, v in facts.items())

        def match_found(d: str) -> bool:
            dw = low(d).split()
            lead = next((i for i, w in enumerate(dw) if re.match(r"[\d.]", w)), len(dw))     # (a property name holds no figure)
            for k, v in facts.items():
                kw = label_terms(k)
                for n in range(min(4, len(dw) - 1, max(1, lead)), 0, -1):
                    if kw and label_agrees(k, " ".join(dw[:n])):
                        rest = " ".join(dw[n:])
                        if value_matches(v, rest) or (not quantities(rest) and value_matches(rest, v)):
                            return True
            return value_matches(text, d)
        return sum(1 for d in ref.get("details") or [] if match_found(d))

    def whole_card(self, n: int, eid: str) -> str:
        ent_row = self.rows.get(eid) or {}
        desc = " ".join(re.sub(r"<[^>]+>", " ", str(ent_row.get("description") or "")).split())
        return (f"{n}. {ent_row.get('title')!s} | seller: {ent_row.get('seller')} | category: {ent_row.get('category')} | facts: "
                f"{json.dumps(self.facts.get(eid) or {}, ensure_ascii=False)[:900]} | notes and description: {desc[-900:]}")

    def absorb_lines(self) -> bool:
        # The shopper's notices that came with the last observations, taken in.
        if not self.queued_lines:
            return False
        lines, self.queued_lines = self.queued_lines, []
        return self.hear(lines)

    def dig(self, want: int = 2, margin: int | None = None) -> None:
        # Spend the steps above the reserve finding fits: facts of promising rows, then the next page of a useful
        # search, then a new narrow search - until enough plain fits are known or the steps run out.
        past_lookups = 0
        stashed = self.margin
        self.margin = stashed if margin is None else margin
        self.resized = False
        try:
            self._dig(want, past_lookups)
        finally:
            # (a limit changed while looking: the finish is sized again rather than the old reserve put back)
            self.margin = self.finish_reserve() if self.resized else stashed

    def _dig(self, want: int, past_lookups: int) -> None:
        while self.spare_moves() > 0 and self.time_remaining() > 45 and not self.finished and not self.hurry():
            self.absorb_lines()
            is_firm = [r for r in self.fits() if self.rank_of(r) <= 2]
            if len(is_firm) >= want and not (self.cheapest and want > 1):
                return
            if self.cheapest and is_firm and self.spare_moves() <= self.scaled(12, 4):
                return      # (the rest goes to listing the cheapest band)
            if self.cheapest:   # (the cheapest rows first: the first fits found are then the cheapest known)
                pending_work = sorted((r for r in self.rows if self.hopeful(r)), key=lambda r: (-self.proof(r), self.rows[r].get("price") or 0))
            else:
                pending_work = sorted((r for r in self.rows if self.hopeful(r)),
                              key=lambda r: (-self.proof(r), -self.entry_rating(r), self.rows[r].get("price") or 0))
            if pending_work:
                take = pending_work[: min(6, self.spare_moves())]
                self.load_facts(take)
                self.judge([r for r in take if r in self.facts])
                continue
            # (code has no promising listing left to read: the model chooses the next moves from a menu - race #7: agents
            # whose search moves were steered by a model checked by code scored 0.246, code alone 0.109)
            if self.steered < 2 and self.spare_moves() >= 3 and self.time_remaining() > 70:
                self.steered += 1
                if self.steer():
                    continue
            page = self.following_page_call()
            if page:
                self.send([page])
                continue
            unseen = [cond_item for cond_item in self.lookup_ideas() if not self.issued_lookup(cond_item)]
            if not unseen and past_lookups < 2 and self.time_remaining() > 60:
                past_lookups += 1
                self.request_lookups()
                unseen = [cond_item for cond_item in self.lookup_ideas() if not self.issued_lookup(cond_item)]
            if not unseen:
                return
            self.search(unseen[: min(2, self.spare_moves())])

    def issued_lookup(self, cond_item: str) -> bool:
        arg = self.tools.arg.get(("search", "text"))
        want = " ".join(sorted(set(roots(cond_item))))
        for key in self.issued_lookups:
            args = json.loads(key)
            if arg in args and " ".join(sorted(set(roots(args[arg])))) == want:
                return True
        return False

    def request_lookups(self) -> None:
        tried = [f"{json.loads(k).get(self.tools.arg.get(('search', 'text'), ''), '')} -> {n}" for k, n in self.issued_lookups.items()]
        titles = [str((self.rows.get(r) or {}).get("title"))[:70] for r in list(self.rows)[-15:]]
        # (listings whose words name a required value without being fits: their notes may point at the listing that is)
        referring = [f"{(self.rows[r].get('title') or '')[:60]}: {' '.join(str(self.rows[r].get('description') or '').split())[:220]}"
                    for r in list(self.facts)[-25:] if r in self.rows and r not in self.fits()
                    and any(holds(self.rows[r].get("description"), cond_item["value"]) for cond_item in self.requirements)][:4]
        text = (f"Buyer's message: {self.request}\nGoods: {self.item}\nConditions: {self.need_lines()}\n"
                f"Past lookups (match count): {tried}\nTitles already met: {titles}"
                + ("\nEntries naming a demanded value that they lack themselves:\n" + "\n".join(referring) if referring else ""))
        answer = self.reader.complete(LOOKUP_BRIEF, text, LOOKUP_FORM, 400) or {}
        self.extra_queries += [cond_item for cond_item in answer.get("queries") or [] if isinstance(cond_item, str) and cond_item.strip()][:4]

    def steer(self) -> bool:
        # The model picks listings to read, pages to read and searches to try; code keeps only moves that exist, have
        # not been made, and fit the steps left. True when a move was made.
        unread = sorted((r for r in self.rows if r not in self.facts and r not in self.gone and self.obtainable(r) and self.within_cap(r)),
                        key=lambda r: -self.entry_rating(r))[:20]
        paged = [(k, n) for k, n in self.issued_lookups.items() if self.following_params(k) is not None][:8]
        tried = [f"{i}. {json.loads(k).get(self.tools.arg.get(('search', 'text'), ''), '') or json.loads(k)} -> {n} matches, next page"
                 for i, (k, n) in enumerate(paged, 1)]
        done = [json.loads(k).get(self.tools.arg.get(("search", "text"), ""), "") for k in self.issued_lookups]
        if not unread and not paged and not self.tools.has("search"):
            return False
        text = (f"Request: {self.request[:1500]}\nItem: {self.item}" + (f" (store category: {self.category})" if self.category else "")
                + f"\nRequirements: {self.need_lines()}\nSteps left for looking: {self.spare_moves()}\n\nListings seen, not read:\n"
                + ("\n".join(f"{i}. {str(self.rows[r].get('title'))[:90]} | {self.rows[r].get('category')} | {self.rows[r].get('price')}"
                             for i, r in enumerate(unread, 1)) or "(none)")
                + "\n\nSearches with a next page:\n" + ("\n".join(tried) or "(none)")
                + f"\n\nAll searches tried: {[d for d in done if d][:20]}")
        answer = self.reader.complete(STEER_GUIDE, text, STEER_TOOL, 500) or {}
        pick = lambda key, cand_pool, n: [cand_pool[int(x) - 1] for x in listify(answer.get(key))  # noqa: E731
                                     if isinstance(x, (int, float)) and 1 <= int(x) <= len(cand_pool)][:n]
        reads = list(dict.fromkeys(pick("read", unread, 4)))[: max(0, self.spare_moves())]
        pages = [{"name": self.tools.role[self.lookup_role.get(k, "search")], "args": self.following_params(k)}
                 for k, _ in pick("pages", paged, 2)]
        queries = [cond_item for cond_item in listify(answer.get("queries")) if isinstance(cond_item, str) and cond_item.strip() and not self.issued_lookup(cond_item)][:3]
        self.note(f"steer: read={reads} pages={len(pages)} queries={queries}")
        moved = False
        if reads:
            self.load_facts(reads)
            self.judge([r for r in reads if r in self.facts])
            moved = True
        pages = [c for c in pages if c and c["args"]][: max(0, self.spare_moves())]
        if pages:
            self.send(pages)
            moved = True
        if queries and self.spare_moves() > 0:
            self.extra_queries += queries
            self.search(queries[: min(2, self.spare_moves())])
            moved = True
        return moved

    def appraise(self, passes: int = 2) -> None:
        # Read the facts of the likeliest listings and judge them, until fits are known or the steps run short.
        for _ in range(passes):
            cand_pool = [eid for eid in self.candidates() if eid not in self.verdicts]
            take = cand_pool[: min(8, max(0, self.spare_moves()))]
            if not take:
                break
            self.load_facts(take)
            self.judge(take + [eid for eid in self.candidates()[:20] if eid in self.facts and eid not in self.verdicts])
            if len([r for r in self.fits() if self.rank_of(r) == 1]) >= 2:
                break

    def borne_by(self, query: Any) -> set[str]:
        # The requirements whose value a search carried (every row it shows holds those words somewhere).
        said = set(roots(spread(query)))
        return {self.need_key(r) for r in self.requirements
                if any(set(roots(spread(v))) and set(roots(spread(v))) <= said
                       for v in self.literal_values(r))}

    def underbid(self) -> None:
        # Steps left over on a request for the cheapest: read the rows that could fit and cost less than the cheapest
        # fit known, cheapest first, and the next pages of the searches that showed them.
        while self.spare_moves() > 0 and self.time_remaining() > 45:
            fits = [r for r in self.fits() if self.rank_of(r) <= 2]
            ceiling = self.rows[fits[1]].get("price") if len(fits) >= 2 else (self.budget or 1e12)
            cand_pool = sorted((r for r in self.rows if self.hopeful(r) and (self.rows[r].get("price") or 0) < ceiling),
                          key=lambda r: self.rows[r].get("price") or 0)
            if cand_pool:
                take = cand_pool[: min(4, self.spare_moves())]
                self.load_facts(take)
                self.judge([r for r in take if r in self.facts])
                continue
            page = self.following_page_call()
            if not page:
                return
            self.send([page])

    def further_pages(self, surplus: int, upto: int | None = None) -> None:
        # Searches whose every match fits on the pages a search may show: read the rest of their pages, so that
        # every listing they match is known (results come in id order, not by fit or price).
        limit = upto or self.page_len * max(1, self.tools.page_cap())
        for _ in range(max(0, surplus)):
            if self.spare_moves() <= 0:
                return
            calls = []
            for key, total in list(self.issued_lookups.items()):
                if total is None or total <= self.page_len or total > limit:
                    continue
                following = self.following_params(key)
                if following is not None:
                    calls.append({"name": self.tools.role[self.lookup_role.get(key, "search")], "args": following})
            if not calls:
                return
            self.send(calls[: max(0, min(surplus, self.spare_moves()))])
            surplus -= len(calls)

    def lower_priced(self) -> None:
        # The request asks for the cheapest fit. Results come in id order, so a shelf with many listings shows only
        # some of them; listing the shelf under a price low enough that all of it shows lets every cheaper listing be
        # read. The cheapest fit is in that band when fits are many; when they are few, the searches for the values
        # already showed them all.
        if ("search", "cap") not in self.tools.arg or not self.category:
            return
        # (a search for the values that matches few listings holds every fit: its pages are read instead)
        text_param = self.tools.arg.get(("search", "text"))
        def bare_value_lookup(query: str) -> bool:
            # (the value with at most the item's words: a search with more words matches only some of the fits)
            said = set(roots(spread(query)))
            return any(said <= set(roots(spread(v))) | set(roots(self.item)) | set(roots(self.category))
                       and self.need_key(r) in self.borne_by(query)
                       for r in self.requirements if r.get("must", True) for v in self.literal_values(r))
        narrow = [n for k, n in self.issued_lookups.items() if n and text_param in json.loads(k)
                  and bare_value_lookup(json.loads(k)[text_param])]
        if narrow and min(narrow) <= 30:
            self.note(f"lowest price: value lookups are narrow ({min(narrow)}), reading those hits rather than the low-price slice")
            self.further_pages(self.spare_moves(), upto=30)
            return
        fits = self.fits()
        ceiling = self.rows[fits[0]].get("price") if fits else self.budget
        if not isinstance(ceiling, (int, float)):
            return
        peak_cap = float(ceiling) - (1 if float(ceiling).is_integer() else 0.01)
        shown = self.page_len * max(1, self.tools.page_cap())
        target = min(shown, 18)
        sample_pts: list[tuple[float, int]] = []      # (cap, listings under it)
        cap = peak_cap
        for _ in range(5):
            if self.spare_moves() < 7 or cap <= 0:
                break
            results = self.shelf_lookup(round(cap, 2))
            total = sum_of(results[0]["core"]) if results else None
            if total is None:
                break
            sample_pts.append((cap, total))
            if total <= shown and (total >= 5 or cap >= peak_cap):
                break
            # (the next cap between the nearest probes above and below the target, as prices run between them)
            above = min((p for p in sample_pts if p[1] > shown), default=None)
            beneath = max((p for p in sample_pts if p[1] <= shown), default=(0.0, 0))
            if above is None:
                cap = (cap + peak_cap) / 2 if cap < peak_cap else peak_cap
                continue
            lower_cap, lower_count = beneath
            upper_cap, upper_count = above
            if lower_cap == 0 and len(sample_pts) == 1:
                # (listings crowd toward the upper prices of a shelf: the first step down follows the square root)
                cap = upper_cap * (target / max(1, upper_count)) ** 0.5
                continue
            cap = lower_cap + (upper_cap - lower_cap) * max(0.1, min(0.9, (target - lower_count) / max(1, upper_count - lower_count)))
        fitting = [p for p in sample_pts if p[1] <= shown]
        if not fitting:
            return
        cap, total = max(fitting)
        pages = min(self.tools.page_cap(), -(-total // max(1, self.page_len)))
        if pages > 1 and self.spare_moves() >= 2:
            self.shelf_lookup(round(cap, 2), pages=pages)
        cand_pool = [eid for eid in self.rows if self.on_shelf_row(eid) and self.obtainable(eid) and self.within_cap(eid)
                and (self.rows[eid].get("price") or 1e9) <= cap + 1e-9 and eid not in self.facts]
        cand_pool = [eid for eid in sorted(cand_pool, key=lambda r: self.rows[r].get("price") or 0)
                if not any(self.grade(eid, r) == "C" for r in self.requirements)]
        self.note(f"low-price slice: samples={[(round(c), n) for c, n in sample_pts]} ceiling={cap:.0f}, {len(cand_pool)} queued")
        # (cheapest first, a few at a time: the first fits found are the cheapest of the band - two of them, as the
        # first chosen is often sold out by the time it is ordered)
        found = 0
        while self.spare_moves() > 0:
            while cand_pool and self.spare_moves() > 0:
                take, cand_pool = cand_pool[: min(4, self.spare_moves())], cand_pool[4:]
                self.load_facts(take)
                self.judge([eid for eid in take if eid in self.facts])
                found += sum(1 for r in take if self.rank_of(r) <= 2)
                if found >= 2:
                    return
            # (the band held fewer than two fits: the next band up, to the dearer fit known, is read too - a listing
            # whose title does not say the value is found only by its facts, and the first choice may sell out)
            pricier = [self.rows[r].get("price") or 0 for r in self.fits() if self.rank_of(r) <= 2]
            top = (sorted(pricier)[1] if len(pricier) >= 2 else self.budget or peak_cap) - 0.01
            # (a cap no higher than every listing under it shows: results come in id order, so a band larger than the
            # pages a search shows hides some of its cheapest)
            floor_cap, upper_val, best, total = cap, top, None, None
            for _ in range(3):
                if self.spare_moves() < 6 or upper_val <= floor_cap:
                    break
                try_no = round(floor_cap + (upper_val - floor_cap) * (0.15 if best is None else 0.5), 2)
                results = self.shelf_lookup(try_no)
                n = sum_of(results[0]["core"]) if results else None
                if n is None:
                    break
                if n <= shown:
                    best, total = try_no, n
                    if n >= shown * 0.6:
                        break
                    floor_cap, upper_val = try_no, upper_val
                else:
                    upper_val = try_no
            floor_cap = cap
            if best is None or best <= cap:
                return
            cap = best
            if total and total > self.page_len and self.spare_moves() >= 6:
                self.shelf_lookup(round(cap, 2), pages=min(self.tools.page_cap(), -(-total // max(1, self.page_len))))
            cand_pool = sorted((eid for eid in self.rows if self.on_shelf_row(eid) and self.obtainable(eid) and self.within_cap(eid)
                           and floor_cap < (self.rows[eid].get("price") or 1e9) <= cap + 1e-9 and eid not in self.facts
                           and not any(self.grade(eid, r) == "C" for r in self.requirements)),
                          key=lambda r: self.rows[r].get("price") or 0)
            self.note(f"low-price slice widened to {floor_cap:.0f}-{cap:.0f}, {len(cand_pool)} queued")

    # ---- finishing
    def adder_calls(self, eid: str, qty: int) -> list[dict]:
        # The cart calls that put `qty` units of one listing in: one call with the count when the cart tool takes one,
        # else one call per unit (a tool without a count adds one unit a call).
        # (race #6: 18 several-product orders went in at one unit a line where the shopper asked for 2 or 3)
        qty = max(1, int(qty or 1))
        if self.tools.arg.get(("cart_add", "qty")) or qty == 1:
            call = self.tools.call("cart_add", id=eid, qty=qty if qty > 1 else None)
            return [call] if call else []
        call = self.tools.call("cart_add", id=eid)
        return [call] * qty if call else []

    def put_basket(self, eid: str) -> bool:
        # Only the choice in the cart, in the quantity asked; its stock read again. True when it is there and on sale.
        calls = []
        if self.tools.has("cart_set"):
            calls.append(self.tools.put_basket_call([(eid, self.quantity)]))
            self.cart = {eid: self.quantity}
        for other in list(self.cart):
            if other != eid and self.tools.has("cart_remove"):
                calls.append(self.tools.call("cart_remove", id=other, qty=self.cart[other]))
        if self.cart.get(eid) not in (None, self.quantity) and self.tools.has("cart_remove") and not self.tools.has("cart_set"):
            calls.append(self.tools.call("cart_remove", id=eid, qty=self.cart[eid]))       # (a line in the wrong count)
            self.cart.pop(eid, None)
        if eid not in self.cart and self.tools.has("cart_add") and not self.tools.has("cart_set"):
            calls += self.adder_calls(eid, self.quantity)
        if self.tools.has("view"):
            calls.append(self.tools.call("view", id=eid))
        calls = [c for c in calls if c]
        if len(calls) >= self.moves_remaining() and calls and calls[-1]["name"] == self.tools.role.get("view"):
            calls = calls[:-1]      # (the order needs the last step: no second look at the stock)
        if len(calls) >= self.moves_remaining():
            return False
        results = self.send(calls)
        for other in list(self.cart):
            if other != eid:
                self.cart.pop(other, None)
        if self.queued_lines:
            lines, self.queued_lines = self.queued_lines, []
            self.hear(lines)
        failed = any(r["role"] == "cart_add" and (r["error"] or (isinstance(r["core"], dict) and any(re.search(r"issue|error|fail", low(k)) for k in r["core"])))
                     for r in results)
        if failed:
            self.note(f"basket refused {eid}: {[r['core'] for r in results if r['role'] == 'cart_add']}")
            self.gone.add(eid)
            self.cart.pop(eid, None)
            return False
        ent_row = self.rows.get(eid) or {}
        if ent_row.get("in_stock") is False or not self.within_cap(eid) or eid in self.gone:
            self.note(f"{eid} unavailable or too dear once in basket: in_stock={ent_row.get('in_stock')} price={ent_row.get('price')}")
            self.gone.add(eid)
            return False
        return True

    def gatekeep(self, eid: str) -> bool:
        # The reader reads the whole listing once more before it goes in the cart, with a thinking pass: a title
        # that names other values than the facts ('19/22/24/27 Inch ... 2K/4K' beside '3440 x 1440'), a kind that is
        # not the item, a fact that says otherwise. Only a clear failure counts; a reader that does not answer passes.
        fingerprint = json.dumps([eid, self.need_lines()])
        if fingerprint in self.checked:
            return self.checked[fingerprint]
        if self.time_remaining() < 50 or not self.requirements:
            return True
        on_shelf = bool(self.category) and self.on_shelf_row(eid)
        text = (f"The shopper wants: {self.item}" + (f" (store category: {self.category})" if self.category else "")
                + ("\nThis entry sits on the named shelf, so its goods type counts as satisfied; skip that judgement."
                   if on_shelf else "")
                + "\nRequirements:\n" + "\n".join(f"- {line}" for line in self.need_lines())
                + f"\n\nListing:\n{self.entry_card(1, eid)}")
        answer = self.reader.complete(GATE_BRIEF, text, GATE_FORM, 500, think=True)
        ok = not (isinstance(answer, dict) and answer.get("fits") is False and [p for p in answer.get("problems") or [] if str(p).strip()])
        self.checked[fingerprint] = ok
        if not ok:
            self.note(f"final check rejects {eid} {str((self.rows.get(eid) or {}).get('title'))[:50]!r}: {answer.get('problems')}")
        return ok

    # ---- the steps the request asks for before the order
    def objective_from(self, plan: dict, said: str) -> dict | None:
        # The reader's best_by, checked by code: the shopper's own words win ("the lightest one" is least weight); a
        # measure the words never name is not taken (price stays the ceiling, "cheapest" its own rule).
        coded = objective_by_code(said)
        by = str(plan.get("best_by") or "").strip()
        way = low(plan.get("best_direction") or "")
        item = str(plan.get("item") or self.item or "")
        if coded and self.shelf_words(coded["property"]) & self.shelf_words(item):
            # ("the thinnest laptop": the word after the adjective is the item - the adjective names the measure)
            coded = {**coded, "property": by} if by else None
        if coded:
            return coded
        per = re.match(rf"^(?:price|cost)\s+(?:per|a|for each)\s+({_PER_UNIT})$", low(by))
        if per and re.search(rf"\b{re.escape(per.group(1))}\b", low(said)):
            return {"kind": "least", "property": f"price per {per.group(1)}", "per": per.group(1)}
        named = set(roots(by)) & set(roots(said)) or re.search(r"\b(?:\w{3,}est|most|least)\b", low(said))     # ("the lightest one")
        if by and way in ("most", "least") and named and not re.search(r"price|cost|cheap", low(by)):
            return {"kind": way, "property": by}
        return None

    def objective_value(self, eid: str) -> float | None:
        # The listing's figure for the measure, from its own facts (units converted), else its title; for value by a
        # measure, its price for each unit of it.
        if not self.objective:
            return None
        if self.objective.get("per"):
            amount = self.unit_amount(eid, self.objective["per"])
            price = number((self.rows.get(eid) or {}).get("price"))
            return price / amount if amount and price is not None else None
        prop = self.objective["property"]

        def figure(r: str) -> tuple[float, str] | None:
            facts = self.facts.get(r) or {}
            val_texts = [str(v) for k, v in facts.items() if label_agrees(k, prop) or (attr_terms(prop) - PLAIN_LABEL) & attr_terms(k)]
            for v in val_texts + ([str((self.rows.get(r) or {}).get("title") or "")] if not val_texts else []):
                cond_item = quantities(v)
                if cond_item:
                    return cond_item[0]
            return None
        got = figure(eid)
        if got is None:
            return None
        # (no conversion table: figures rank only in the unit most listings use for this measure; another unit is unranked)
        units: dict[str, int] = {}
        for r in self.facts:
            f = figure(r)
            if f is not None:
                units[f[1]] = units.get(f[1], 0) + 1
        common = max(units.items(), key=lambda pair_kv: pair_kv[1])[0] if units else got[1]
        return got[0] if got[1] == common or not got[1] or not common else None

    def unit_amount(self, eid: str, unit: str) -> float | None:
        # How much of the unit one listing holds, written in that same unit: a figure in it (times a pack count such as
        # '6 x 250 ml' or 'pack of 6'), from a fact (else the title); or a count from a fact named after the unit
        # ('servings: 30'). A listing that writes it in another unit gives None (no conversion table in code).
        ent_row, facts = self.rows.get(eid) or {}, self.facts.get(eid) or {}
        title = str(ent_row.get("title") or "")
        want = unit_key(unit)
        for k, v in facts.items():
            if want and want[:4] in low(k) and re.search(r"\d", str(v)) and len(str(v)) <= 40:
                cond_item = number(re.sub(r"[^\d.]", " ", str(v)).split()[0])
                if cond_item:
                    return cond_item        # ('servings: 30' for 'per serving')
        for text in [str(v) for v in facts.values() if len(str(v)) <= 80] + [title]:
            pack = re.search(r"\b(\d+)\s*[x\u00d7]\s*(\d+(?:\.\d+)?\s*[a-z]+)", low(text))
            each = [v for v, d in quantities(pack.group(2)) if d == want] if pack else []
            if each:
                return each[0] * float(pack.group(1))      # ('6 x 250 ml': six bottles of 250 ml)
            found = [v for v, d in quantities(text) if d == want]
            if found:
                packs = re.search(r"\bpack\s+of\s+(\d+)|\b(\d+)[- ]pack\b", low(text))
                n = number(next((g for g in packs.groups() if g), None)) if packs else None
                return found[0] * (n or 1)
        return None

    def plan_messages(self, why: str = "the request read") -> None:
        # The message plan, made when the request is read and remade when a limit or a required step changes: the
        # shopper's message limit, less the messages the request itself requires (a basket / total / go-ahead step is one
        # message), less those sent - every other message is a question about a held-back feature, shared among the
        # products, a product whose request states fewer features (more held back) first.
        if not self.tools.has("message"):
            self.msg_plan = {"limit": 0, "required": 0, "questions": 0, "per_product": []}
            return
        required = self.held_messages()
        questions = max(0, self.message_budget - self.msgs_out - required)
        n = max(1, len(self.products))
        stated = ([len([r for r in p["requirements"] if r.get("must", True)]) for p in self.products]
                  if self.products else [len([r for r in self.requirements if r.get("must", True)])])
        share = [questions // n] * n
        for i in sorted(range(n), key=lambda i: (stated[i], i))[: questions - sum(share)]:
            share[i] += 1
        self.msg_plan = {"limit": self.message_budget, "required": required, "questions": questions, "per_product": share}
        self.note(f"message plan ({why}): limit {self.message_budget}, sent {self.msgs_out}, kept for asked steps "
                  f"{required}{' (' + self.held_for() + ')' if required else ''}, questions {questions}" + (f" per product {share}" if self.products else ""))

    def ask_leftover(self) -> bool:
        # Questions the plan left unasked (fewer properties were worth asking when the shelf's labels were few): asked
        # now, with the labels the looking has read since. True when an answer changed what fits.
        if self.questions_remaining() <= 0 or self.spare_moves() < self.scaled(4, 2) or self.time_remaining() < 70 or self.messages_closed:
            return False
        before = json.dumps([self.need_lines(), self.budget])
        self.note(f"leftover questions: {self.questions_remaining()}")
        self.ask(cap=self.questions_remaining())
        self.trace_pointers()
        return json.dumps([self.need_lines(), self.budget]) != before

    def questions_remaining(self) -> int:
        # Messages that may go to questions: one stays back for the message the request requires (see held_messages).
        return max(0, self.message_budget - self.msgs_out - self.held_messages())

    def offer_pending(self) -> bool:
        # Under "when there's no clean fit - say so - and offer the listing short on the fewest counts", the offer is a
        # message the task may need: until every product has a fitting listing, and while it has not been sent.
        if not self.closest_fit or self.no_substitute or self.proposed:
            return False
        if self.products:
            return not all(self._qualifiers_of(i) for i in range(len(self.products)))
        return not self.fits()

    def held_messages(self) -> int:
        # The one message kept back from the questions: the basket / total / go-ahead step the request asks for (sent when
        # a candidate is in the cart, right before the order), or the "no clean fit" offer while no fit has been found.
        # The two never both go out (a fit is ordered, no fit is declined), so one message covers either; once a fit is
        # found under the closest-fit rule (and no step was asked) the message is freed for a question about a feature
        # the request does not state.
        if not self.tools.has("message"):
            return 0
        return 1 if (self.before_order and not self.approved) or self.offer_pending() else 0

    def held_for(self) -> str:
        if self.before_order and not self.approved:
            return "the asked basket step"
        return "the no-clean-fit offer, freed once a fit is found"

    def say(self, text: str) -> dict | None:
        # One message to the shopper: plain, short, one point (race #13: a pasted-back request ran past the world's cut and
        # every reply was "you've lost me").
        text = " ".join(str(text).split())
        if len(text) > MESSAGE_CAP:
            text = text[: MESSAGE_CAP].rsplit(" ", 1)[0].rstrip(",;:") + "."
        return self.tools.call("message", text=text)

    def basket_summary(self, width: int = 60) -> str:
        lines, total = [], 0.0
        for eid, qty in self.cart.items():
            ent_row = self.rows.get(eid) or {}
            price = number(ent_row.get("price"))
            title = " ".join(str(ent_row.get("title") or eid).split())[:width]
            lines.append(f"{qty} x {title}" + (f" at {price:g} each" if price is not None else ""))
            total += (price or 0.0) * qty
        text = "; ".join(lines) or "nothing yet"
        text += f", {total:g} in total" if total else ""
        if len(text) > MESSAGE_CAP - 120 and width > 25:
            return self.basket_summary(width - 15)
        return text

    def finish_text(self, eids: list[str] | None = None) -> str:
        # The order's own note (a finishing tool may require one): the real lines and total, the requirements met, and
        # what the nearest listing falls short on when nothing fit exactly.
        met = "; ".join(self.need_lines()[:6])
        budget = f", within the {self.budget:g} budget" if self.budget else ""
        text = f"Ordering {self.basket_summary(50)}{budget}, in stock." + (f" Meets: {met}." if met and not self.near_note else "")
        return (text + (f" {self.near_note}" if self.near_note else ""))[:600]

    def seek_basket_ok(self) -> str:
        # Right before the order: the cart looked at once (it can carry the shopper's notes), then the basket told to
        # the shopper for their go-ahead. "" when the order may go on; else why it may not ("changed": the reply or a
        # note changed what fits, so the choice is weighed again).
        # (race #6: 12 of the 31 tasks lost to the leaders missed such a step, 9 of them with the same listing in the cart)
        fingerprint = json.dumps(sorted(self.cart.items()))
        if not self.before_order or self.approved_basket == fingerprint or self.finished:
            return ""
        self.approved_basket = fingerprint
        changed = False
        if self.tools.has("cart_view") and self.moves_remaining() >= 3:
            self.send([self.tools.call("cart_view")])
            changed = self.absorb_lines() or changed
        if self.tools.has("message") and not self.messages_closed and self.message_budget - self.msgs_out > 0 and self.moves_remaining() >= 2:
            text = (f"Your basket so far: {self.basket_summary()}. "
                    "Every item matches your request. Would you like me to check out now?")
            call_result = self.send([self.say(text)])
            core = call_result[0]["core"] if call_result else None
            if isinstance(core, dict) and len(core) == 1 and isinstance(next(iter(core.values())), str):
                core = next(iter(core.values()))
            reply = core if isinstance(core, str) else json.dumps(core)[:400] if core is not None else ""
            self.note(f"basket message -> {reply[:200]!r}")
            self.approved = True
            # (a go-ahead adds nothing; a reply that names a value or withdraws one changes what fits)
            changed = self.hear([f"Asked: {text}\nShopper: {reply}"]) or changed
            changed = self.absorb_lines() or changed
        else:
            self.approved = True
        return "changed" if changed else ""

    # ---- before a decline (race #7: an audit before declining was the leaders' habit; race #13: all 27 of the leader's
    # declines scored 0)
    def shortfalls(self, eid: str) -> list[str]:
        # What a listing falls short on: must-requirements it contradicts or does not state, values ruled out, stock, budget.
        out, marks = [], self.marks(eid)
        for m, r in zip(marks, self.requirements):
            if r.get("must", True) and m in "CN":
                out.append(f"{r['property']} {r['value']}" + (" (not stated)" if m == "N" else ""))
        for m, x in zip(marks[len(self.requirements):], self.exclusions):
            if m == "C":
                out.append(f"not {x['value']}")
        if not self.obtainable(eid):
            out.append("in stock")
        if not self.within_cap(eid):
            out.append("within budget")
        return out

    def right_kind(self, eid: str) -> bool:
        ent_row = self.rows.get(eid) or {}
        if self.category:
            return self.on_shelf_row(eid)
        v = self.verdicts.get(eid)
        if v is not None and "said" in v:
            return bool(v.get("kind"))
        lead_part = (words(self.item) or [""])[-1]
        return bool(lead_part) and holds(f"{ent_row.get('title')} {ent_row.get('category')}", lead_part)

    def near_pool(self, margin: int = 5) -> list[str]:
        # The listings of the right kind, their facts read (a few more read when steps allow), fewest shortfalls first.
        cand_pool = [r for r in self.rows if r not in self.gone and self.right_kind(r)]
        if len(cand_pool) < 3 and self.moves_remaining() - margin > 1 and self.time_remaining() > 60:
            # (every search carried a required value no listing has: the item alone shows what the shelf does hold)
            lead_part = (words(self.item) or [""])[-1]
            plain = [cond_item for cond_item in dict.fromkeys([self.item, self.category_named, lead_part]) if cond_item and not self.issued_lookup(cond_item)][:1]
            if plain:
                self.search(plain)
                self.fix_shelf()
            cand_pool = [r for r in self.rows if r not in self.gone and self.right_kind(r)]
        unread = sorted((r for r in cand_pool if r not in self.facts), key=lambda r: -self.entry_rating(r))[:4]
        if unread and self.moves_remaining() - margin > 0 and self.time_remaining() > 60:
            self.load_facts(unread, margin=margin)
            self.judge([r for r in unread if r in self.facts])
        return sorted(cand_pool, key=lambda r: (len(self.shortfalls(r)), not self.obtainable(r), self.rows[r].get("price") or 0))

    def audit_pick(self) -> str | None:
        # The model reads the closest listings once more; code lets it overrule only "not stated", never a stated
        # contradiction, stock or budget.
        cand_pool = [r for r in self.near_pool() if r in self.facts and self.obtainable(r) and self.within_cap(r)
                and not any(m == "C" for m in self.marks(r))][:8]
        if not cand_pool or not self.requirements or self.time_remaining() < 45:
            return None
        text = (f"The shopper wants: {self.item}" + (f" (store category: {self.category})" if self.category else "")
                + "\nRequirements:\n" + "\n".join(f"- {line}" for line in self.need_lines())
                + "\n\nListings:\n" + "\n".join(self.whole_card(i, r) for i, r in enumerate(cand_pool, 1)))
        answer = self.reader.complete(AUDIT_GUIDE, text, AUDIT_TOOL, 600, think=True) or {}
        for x in listify(answer.get("fits")):
            if isinstance(x, (int, float)) and 1 <= int(x) <= len(cand_pool) and self.gatekeep(cand_pool[int(x) - 1]):
                self.note(f"audit before decline finds {cand_pool[int(x) - 1]}: {listify(answer.get('why'))[:2]}")
                return cand_pool[int(x) - 1]
        return None

    def nearest(self) -> str | None:
        # The listing short on the fewest points (stock and budget counted), the cheapest of those; one on sale first.
        cand_pool = self.near_pool()
        if not cand_pool:
            return None
        fewest = len(self.shortfalls(cand_pool[0]))
        tied = [r for r in cand_pool if len(self.shortfalls(r)) == fewest]
        buyable = [r for r in tied if self.obtainable(r)] or tied
        return min(buyable, key=lambda r: self.rows[r].get("price") or 0)

    def fill_cart(self, want: dict[str, int]) -> bool:
        # The cart made to hold exactly these lines (no stock or budget test: the caller chose them knowingly).
        calls: list = []
        if self.tools.has("cart_set"):
            calls.append(self.tools.put_basket_call(list(want.items())))
        else:
            for eid in list(self.cart):
                if (eid not in want or self.cart[eid] != want[eid]) and self.tools.has("cart_remove"):
                    calls.append(self.tools.call("cart_remove", id=eid, qty=self.cart[eid]))
                    self.cart.pop(eid, None)
            for eid, qty in want.items():
                if eid not in self.cart:
                    calls += self.adder_calls(eid, qty)
        calls = [c for c in calls if c]
        if len(calls) >= self.moves_remaining():
            return False
        results = self.send(calls)
        self.cart = dict(want)
        self.absorb_lines()
        return not any(r["role"] in ("cart_add", "cart_set") and r["error"] for r in results)

    def offer_text(self, picked_set: list[str]) -> str:
        # "No clean fit" said, and the closest listing offered by its listing id, title, price and what it falls short on.
        price = lambda r: f"{number((self.rows.get(r) or {}).get('price')) or 0:g}"  # noqa: E731
        parts = [f"{r} ({str((self.rows.get(r) or {}).get('title'))[:50]}, {price(r)})"
                 + (f", short on {', '.join(self.shortfalls(r)[:3])}" if self.shortfalls(r) else "") for r in picked_set]
        return ("Nothing on offer meets every requirement, so I'm not buying anything. The closest I can offer is "
                + "; ".join(parts) + ".")

    def guard_decline(self) -> dict | None:
        # Before any decline: an audit of the closest listings (the model reads, code vetoes) - a fit the looking missed is
        # still ordered. Otherwise the decline stands, and when the request asks for the nearest listing to be offered
        # ("When there's no clean fit - say so - and offer the listing that comes up short on the fewest counts"), that
        # listing is named, never bought: in race #13's two tasks with no clean fit, all 14 declines were graded right but
        # for the procedure, and all 4 orders of a near miss were wrong.
        if self.finished or self.moves_remaining() < 3 or not self.tools.has("order") or not (self.tools.has("cart_add") or self.tools.has("cart_set")):
            self.prepare_offer()
            return None
        n = max(1, len(self.products))
        picked_set: dict[int, str] = {}
        for i in range(n):
            if self.products:
                self.switch_to(i)
            eid = next((r for r in self.fits() if r not in picked_set.values()), None) or self.audit_pick()
            if not eid or eid in picked_set.values():
                if self.products:
                    self.switch_to(0)
                self.prepare_offer()
                return None
            picked_set[i] = eid
        if self.products:
            self.switch_to(0)
        want = {r: (self.products[i]["quantity"] if self.products else self.quantity) for i, r in picked_set.items()}
        total = sum(float((self.rows.get(r) or {}).get("price") or 0) * cond_item for r, cond_item in want.items())
        if self.budget and total > self.budget + 1e-9:
            self.prepare_offer()
            return None         # (over the limit: the decline stands)
        self.note(f"the audit before declining found fits: {want}")
        if not self.fill_cart(want):
            self.prepare_offer()
            return None
        return self.order(picked_set[0])

    def prepare_offer(self) -> None:
        # The closest listing (for each product that has no fit) to offer when declining, and the "no clean fit" message
        # when one is left. A request that refuses substitutes gets no listing named at all.
        if not self.closest_fit or self.no_substitute or self.proposed:
            return
        here, picked_set = self.current, []
        for i in range(max(1, len(self.products))):
            if self.products:
                self.switch_to(i)
            if self.fits():
                continue
            eid = self.nearest()
            if eid and eid not in picked_set:
                picked_set.append(eid)
        if self.products:
            self.switch_to(here)
        self.proposed = picked_set
        if not picked_set:
            return
        self.near_note = self.offer_text(picked_set)
        self.note(f"offer instead of an order: {picked_set}")
        if self.tools.has("message") and not self.messages_closed and self.msgs_out < self.message_budget and self.moves_remaining() >= 2:
            self.send([self.say(self.near_note)])
            self.absorb_lines()

    def decline(self, reason: str) -> dict | None:
        if not self.guarding:
            self.guarding = True
            try:
                result = self.guard_decline()
            finally:
                self.guarding = False
            if result is not None or self.finished:
                return result
        if self.no_substitute or not self.near_note:
            # ("Nothing found? Just say so, and don't offer a substitute": a listing named in the note breaks that rule -
            # the practice world's grader marks any listing id in the decline note)
            remark = "I found no in-stock option that satisfies all your conditions inside your budget; I'm therefore stopping without a purchase or a stand-in."
        else:
            remark = self.near_note
        self.note(f"decline: {reason} | {remark[:160]}")
        call = self.tools.call("decline", text=remark)
        return self.send([call])[0] if call else None

    def order(self, eid: str) -> dict | None:
        if self.seek_basket_ok() == "changed":
            # (the reply or a note changed what fits: the choice goes back to the check before it is ordered)
            self.checked.pop(json.dumps([eid, self.need_lines()]), None)
            if self.requirements and not self.gatekeep(eid):
                self.note(f"{eid} stopped qualifying after the basket review")
                return {"core": None, "notices": [], "refusal": [], "error": "a buyer reply altered which entries qualify", "role": "order"}
        call = self.tools.call("order", id=eid, qty=self.quantity, text=self.finish_text())
        self.note(f"order {eid} {str((self.rows.get(eid) or {}).get('title'))[:60]!r} price={(self.rows.get(eid) or {}).get('price')} "
                  f"grades={self.marks(eid)} rank={self.rank_of(eid)} at_move={self.step}")
        return self.send([call])[0] if call else None

    def finish(self) -> None:
        for _ in range(4):
            if self.finished or self.moves_remaining() <= 0:
                return
            self.absorb_lines()
            fits = self.fits()
            if not fits and self.moves_remaining() > 4 and self.time_remaining() > 50:
                # (no fit left - the choice sold out, or the shopper changed what fits: look again, keeping only the
                # steps an order needs)
                self.dig(want=1, margin=4)
                fits = self.fits()
            choice = None
            for eid in fits[:5]:
                if self.moves_remaining() < 3:
                    break
                if eid not in self.facts and self.tools.has("facts") and self.moves_remaining() >= 6:
                    # (a fit read only from its row: its facts are read before it goes in the cart)
                    self.load_facts([eid], margin=5)
                    self.judge([eid])
                    if self.rank_of(eid) >= 9:
                        continue
                    superior = [r for r in self.fits() if r != eid and self.rank_of(r) < self.rank_of(eid)]
                    if superior and superior[0] in self.facts:
                        eid = superior[0]
                if self.rank_of(eid) >= 9:
                    continue
                if self.rank_of(eid) == 3 and not self.hunted_for_backed and self.moves_remaining() > 8 and self.time_remaining() > 60:
                    # (only the title says it: the steps left go to finding one the facts back first)
                    self.hunted_for_backed = True
                    self.dig(want=1, margin=8)
                    superior = [r for r in self.fits() if self.rank_of(r) < 3]
                    if superior:
                        eid = superior[0]
                if not self.gatekeep(eid):
                    self.rejected.add(eid)
                    continue
                if self.put_basket(eid):
                    choice = eid
                    break
            if choice is None:
                basketed = [r for r in self.cart if self.rank_of(r) < 9]
                fits = self.fits()
                if not fits and self.tools.has("decline"):
                    result = self.decline("nothing in the shop satisfies all conditions")
                elif basketed and self.tools.has("order"):
                    result = self.order(basketed[0])
                elif fits and self.moves_remaining() >= 2 and (self.tools.has("cart_add") or self.tools.has("cart_set")):
                    # (too few steps to read the stock again: the best fit goes in and is ordered)
                    eid = fits[0]
                    if self.tools.has("cart_set"):
                        self.send([self.tools.put_basket_call([(eid, self.quantity)])])
                    else:
                        calls = [self.tools.call("cart_remove", id=obj_item, qty=self.cart[obj_item]) for obj_item in list(self.cart) if self.tools.has("cart_remove")]
                        if len(calls) + 2 > self.moves_remaining():
                            calls = []
                        adds = self.adder_calls(eid, self.quantity)
                        if len(calls) + len(adds) + 1 > self.moves_remaining():
                            adds = adds[:1]        # (no steps for one call per unit: one unit rather than no order)
                        self.send([c for c in calls if c] + adds)
                    self.cart = {eid: self.quantity}
                    result = self.order(eid)
                elif self.tools.has("decline"):
                    result = self.decline("qualifying entries were refused by the basket")
                else:
                    return
            else:
                if self.moves_remaining() < 1:
                    return
                # (a question still free - the limit rose, or fewer were worth asking before: asked before the order; an
                # answer that rules the choice out sends it back)
                if self.questions_remaining() > 0 and self.ask_leftover():
                    if self.rank_of(choice) >= 9 or not self.gatekeep(choice):
                        self.rejected.add(choice)
                        continue
                result = self.order(choice)
            if result is None or self.finished:
                return
            # a refused finish: the shopper changed something, or the choice sold out
            lines = list(result.get("refusal") or []) + self.queued_lines
            self.queued_lines = []
            self.rejections += 1
            self.note(f"closing step rejected: {lines} err={result.get('error')} detail={str(result.get('core'))[:200]}")
            if not lines and result.get("error"):
                if choice is not None:
                    self.gone.add(choice)
                continue
            changed = self.hear(lines)
            if choice is not None and choice not in self.gone and self.rank_of(choice) < 9 and not changed:
                continue    # (refused for a line that changed nothing: try the same order again)
            if choice is not None and re.search(r"(?<!price )\bgone\b(?! up| down)|sold out|zero left|nothing left|none left|unavailable|can't be ordered|cannot be ordered|ran dry|dried up|out of stock|grayed out|got bought|beat me to it|no longer|cleared out|shelves are bare|last of those", low(" ".join(lines))):
                self.gone.add(choice)
            if choice is not None and re.search(r"price|priced|charg|cost|gone up|come down|went up|went down", low(" ".join(lines))):
                self.price_moved.add(choice)   # (its price is read again before the next order)
            if changed and self.spare_moves() > 2:
                if self.cheapest:
                    self.lower_priced()
                self.appraise(passes=1)

    # ---- several products in one order
    def drive_several(self) -> None:
        # Questions first: each product's search and a few facts, then its share of the questions, then its searches
        # with what the answers told and its share of the steps to find fits; one cart with a line for each, ordered once.
        n = len(self.products)
        self.margin = self.finish_reserve()
        base = self.margin
        for i in range(n):
            self.switch_to(i)
            self.search(self.lookup_ideas()[:1] or [self.item])
            self.fix_shelf()
            self.probe_shelf()
            self.fix_shelf()
            self.plan_searches()
            # (each product's planned searches: the first one now, on its own shelf)
            if self.spare_moves() > 2 * (n - i):
                self.search([cond_item for cond_item in self.queries if not self.issued_lookup(cond_item)][:1])
                self.fix_shelf()
        for i in range(n):
            self.switch_to(i)
            top = [r for r in self.candidates() if self.hopeful(r)][: max(0, min(3, self.spare_moves() // (n - i) - 2))]
            self.load_facts(top)
        for i in range(n):
            self.switch_to(i)
            share = (self.msg_plan.get("per_product") or [0] * n)[i] if getattr(self, "msg_plan", None) else -(-self.questions_remaining() // (n - i))
            self.ask(cap=min(share, self.questions_remaining()))
            self.absorb_lines()
            self.trace_pointers()
        for i in range(n):
            self.switch_to(i)
            if self.spare_moves() > 2:
                self.search([cond_item for cond_item in self.lookup_ideas() if not self.issued_lookup(cond_item)][:1])
                self.fix_shelf()
            self.judge([r for r in self.candidates() if r in self.facts][:3])
        for i in range(n):
            self.switch_to(i)
            share = max(2, self.spare_moves() // (n - i))
            self.dig(want=2, margin=max(base, self.moves_remaining() - share))
        self.margin = base
        for i in range(n):
            if self.questions_remaining() <= 0:
                break
            self.switch_to(i)
            if self.ask_leftover():
                self.dig(want=1, margin=max(base, self.moves_remaining() - max(2, self.spare_moves() // (n - i))))
        self.note("fits: " + " | ".join(
            f"{p['item']}: {[(r, self.rows[r].get('price')) for r in self._qualifiers_of(i)[:3]]}" for i, p in enumerate(self.products))
                  + f" step={self.step}/{self.max_steps} spent={self.reader.spent:.4f}")
        self.close_several()

    def _qualifiers_of(self, i: int) -> list[str]:
        here = self.current
        self.switch_to(i)
        fits = self.fits()
        self.switch_to(here)
        return fits

    def line_count(self, picked_set: dict[int, str]) -> float:
        return sum(float(self.rows[r].get("price") or 0) * self.products[i]["quantity"] for i, r in picked_set.items())

    def close_several(self) -> None:
        n = len(self.products)
        for _ in range(4):
            if self.finished or self.moves_remaining() <= 0:
                return
            self.absorb_lines()
            if self.stock_doubt and self.cart and self.tools.has("view") and self.moves_remaining() > len(self.cart) + 1:
                # a sell-out that named no listing: each line's stock is read again
                self.stock_doubt = False
                self.send([self.tools.call("view", id=r) for r in list(self.cart)])
                for r in list(self.cart):
                    if (self.rows.get(r) or {}).get("in_stock") is False:
                        self.gone.add(r)
            picked_set: dict[int, str] = {}
            for i in range(n):
                self.switch_to(i)
                fits = [r for r in self.fits() if r not in picked_set.values()]
                if not fits and self.moves_remaining() > 2 * n + 4 and self.time_remaining() > 50:
                    self.dig(want=1, margin=2 * n + 3)
                    fits = [r for r in self.fits() if r not in picked_set.values()]
                for eid in fits[:3]:
                    if self.gatekeep(eid):
                        picked_set[i] = eid
                        break
                if i not in picked_set:
                    break
            if len(picked_set) == n and self.cross_compat:
                picked_set = self.compat_picks(picked_set) or picked_set
            if len(picked_set) == n and self.set_rules:
                picked_set = self.rule_picks(picked_set)
            if len(picked_set) < n:
                missing = [p["item"] for i, p in enumerate(self.products) if i not in picked_set]
                if self.tools.has("decline"):
                    self.decline(f"nothing qualifies for {missing}")
                return
            if self.budget and self.line_count(picked_set) > self.budget + 1e-9:
                for i in range(n):          # (over the limit in all: the cheapest fit of each)
                    self.switch_to(i)
                    fits = sorted(self.fits(), key=lambda r: self.rows[r].get("price") or 0)
                    if fits:
                        picked_set[i] = fits[0]
                if self.line_count(picked_set) > self.budget + 1e-9:
                    if self.tools.has("decline"):
                        self.decline("combined price of the goods is above the ceiling")
                    return
            if not self.put_basket_several(picked_set):
                continue
            result = self.order(next(iter(picked_set.values())))
            if result is None or self.finished:
                return
            lines = list(result.get("refusal") or []) + self.queued_lines
            self.queued_lines = []
            self.note(f"closing step rejected: {lines} err={result.get('error')}")
            self.hear(lines)
            if re.search(r"price|priced|charg|cost|gone up|come down", low(" ".join(lines))) and self.tools.has("view"):
                self.stock_doubt = True       # (prices are read again with the stock)

    def listing_model(self, eid: str) -> str:
        ent_row = self.rows.get(eid) or {}
        facts = self.facts.get(eid) or {}
        return model_of_text(f"{ent_row.get('description') or ''} " + " ".join(f"{k}: {v}" for k, v in facts.items())) \
            or next((str(v).upper() for k, v in facts.items() if re.search(r"\bmodel\b", low(k)) and MODEL_CODE.search(str(v).upper())), "")

    def cross_roles(self) -> tuple[int | None, list[int]]:
        # Which product the others must fit and which products are bound to it, as the reader names them (fit_rules:
        # [{"product": the one that must fit, "fits": the one it must fit}]); (None, []) when the request sets none.
        anchor, bound = None, []
        for r in getattr(self, "fit_rules", []):
            a, b = self.product_index(str(r.get("fits") or "")), self.product_index(str(r.get("product") or ""))
            if a is None or b is None or a == b:
                continue
            if anchor is None:
                anchor = a
            if a == anchor and b not in bound:
                bound.append(b)
        return (anchor, bound) if anchor is not None else (None, [])

    def bind_owned(self) -> None:
        # (a product that must fit something the shopper owns carries the reader's requirement "compatible with <model>";
        # code no longer finds the owned thing by phrase patterns)
        return

    def pair_fits(self, base: str, cands: list[str]) -> list[str]:
        # Neither page lists models: the reader says which candidates fit or work with the chosen listing (code decides
        # whenever a "Compatible with:" line exists).
        if not cands or self.time_remaining() < 60:
            return []
        text = (f"Chosen listing:\n{self.whole_card(0, base)}\n\nCandidates:\n" + "\n".join(self.whole_card(i, r) for i, r in enumerate(cands, 1)))
        answer = self.reader.complete(PAIR_GUIDE, text, PAIR_TOOL, 400) or {}
        return [cands[int(x) - 1] for x in listify(answer.get("fit")) if isinstance(x, (int, float)) and 1 <= int(x) <= len(cands)]

    def listing_says(self, eid: str, prop: str, values: list[str]) -> bool | None:
        # Whether the listing's own fact for this property (else its title) holds one of the values; None when it does not
        # say.
        facts = self.facts.get(eid) or {}
        val_texts = [str(v) for k, v in facts.items() if label_agrees(k, prop)]
        if not val_texts:
            title = str((self.rows.get(eid) or {}).get("title") or "")
            return True if any(value_matches(title, v) for v in values) else None
        return any(value_matches(v, w) or value_matches(w, v) for v in val_texts for w in values)

    def product_index(self, name: str) -> int | None:
        # the product a name means: the most words shared, then the fewest words the product's name has besides ("laptop"
        # is the laptop, not the laptop sleeve)
        word_items = self.shelf_words(name)
        scored = [(len(self.shelf_words(p["item"]) & word_items) + len(self.shelf_words(p.get("category_named")) & word_items),
                   -len(self.shelf_words(p["item"]) - word_items), -i, i) for i, p in enumerate(self.products)]
        best = max(scored, default=(0, 0, 0, None))
        return best[3] if best[0] else None

    def rule_picks(self, picked_set: dict[int, str]) -> dict[int, str]:
        # "The streaming media players's plug type must be eu, except when the portable monitors's resolution is 1920 x
        # 1200" / "Should the computer monitors's resolution turn out to be 3840 x 2160, I need the streaming media
        # players's processor to be allwinner h313": when a rule binds on the chosen set, the bound product's pick must
        # meet it - another fit of it that does, else another pick of the other product that lifts the rule.
        here = self.current
        try:
            for rule in self.set_rules:
                xi, yi = self.product_index(rule["product"]), self.product_index(rule["other"])
                if xi is None or yi is None or xi == yi or xi not in picked_set or yi not in picked_set:
                    continue
                cond = self.listing_says(picked_set[yi], rule["other_property"], rule["other_values"]) is True
                binds = cond if rule["kind"] == "if" else not cond
                if not binds or self.listing_says(picked_set[xi], rule["property"], rule["values"]) is True:
                    continue
                self.switch_to(xi)
                alt = [r for r in self.fits() if r not in picked_set.values() and self.listing_says(r, rule["property"], rule["values"]) is True]
                if alt:
                    self.note(f"set rule ({rule['product']} {rule['property']} {rule['values']}): {picked_set[xi]} -> {alt[0]}")
                    picked_set[xi] = alt[0]
                    continue
                self.switch_to(yi)
                lift = [r for r in self.fits() if r not in picked_set.values()
                        and (self.listing_says(r, rule["other_property"], rule["other_values"]) is True) == (rule["kind"] == "unless")]
                if lift:
                    self.note(f"set rule ({rule['product']} {rule['property']} {rule['values']}): lifted by {rule['other']} {lift[0]}")
                    picked_set[yi] = lift[0]
                else:
                    self.note(f"set rule not met and no fit lifts it: {rule}")
            return picked_set
        finally:
            self.switch_to(here)

    def compat_picks(self, picked_set: dict[int, str]) -> dict[int, str] | None:
        # "Whatever cabinet you choose, the handles must fit it": the product whose listings name their own model is the
        # anchor; each other product's pick must list that model in its own compatibility line (code decides).
        here = self.current
        try:
            n = len(self.products)
            anchor, bound = self.cross_roles()
            if anchor is None:
                anchor = next((i for i in range(n) if self.listing_model(picked_set[i])), None)
            if anchor is None:
                return None
            bound = bound or [i for i in range(n) if i != anchor]
            self.switch_to(anchor)
            sets: list[dict[int, str]] = []
            for base in [picked_set[anchor]] + [r for r in self.fits() if r != picked_set[anchor]][:4]:
                model = self.listing_model(base)
                new = {anchor: base}
                for i in range(n):
                    if i == anchor:
                        continue
                    self.switch_to(i)
                    if i not in bound:
                        new[i] = picked_set[i]
                        continue
                    cands = [r for r in self.fits() if r not in new.values()]
                    if model:
                        ok = [r for r in cands if fits_model(f"{(self.rows.get(r) or {}).get('description') or ''}", model)]
                        if self.cheapest:
                            ok.sort(key=lambda r: self.rows[r].get("price") or 0)
                    else:
                        # (no model on the chosen page: the reader judges the pairs whose pages carry no model lines)
                        ok = self.pair_fits(base, [r for r in cands if fits_model(f"{(self.rows.get(r) or {}).get('description') or ''}", "ZZ-0000") is None][:6])
                    if not ok:
                        break
                    new[i] = ok[0]
                if len(new) == n:
                    sets.append(new)
                    if not self.cheapest:
                        break
            if not sets:
                return None
            # ("Pick the combo with the smallest total": of the compatible sets, the one with the smallest total)
            best = min(sets, key=self.line_count) if self.cheapest else sets[0]
            if best != picked_set:
                self.note(f"compatible set: {best} total {self.line_count(best):g} (of {len(sets)} sets)")
            return best
        finally:
            self.switch_to(here)

    def put_basket_several(self, picked_set: dict[int, str]) -> bool:
        # The cart holds exactly one line per product, each in its quantity; the new lines' stock read again.
        want = {eid: self.products[i]["quantity"] for i, eid in picked_set.items()}
        calls = []
        if self.tools.has("cart_set"):
            calls.append(self.tools.put_basket_call(list(want.items())))
        else:
            for eid in list(self.cart):
                mismatched = eid in want and self.cart[eid] != want[eid]      # (a line in another count goes out and back in)
                if (eid not in want or mismatched) and self.tools.has("cart_remove"):
                    calls.append(self.tools.call("cart_remove", id=eid, qty=self.cart[eid]))
                    if mismatched:
                        self.cart.pop(eid, None)
            for eid, qty in want.items():
                if eid not in self.cart:
                    calls += self.adder_calls(eid, qty)
        new = [eid for eid in want if eid not in self.cart]
        openings = [self.tools.call("view", id=eid) for eid in new] if self.tools.has("view") else []
        calls = [c for c in calls if c]
        if len(calls) >= self.moves_remaining():
            return False
        openings = [v for v in openings if v][: max(0, self.moves_remaining() - len(calls) - 1)]
        results = self.send(calls + openings)
        self.cart = dict(want)
        faulty = False
        for r in results:
            if r["role"] in ("cart_add", "cart_set") and (r["error"] or (isinstance(r["core"], dict) and any(
                    re.search(r"issue|error|fail|fault", low(k)) for k in r["core"]))):
                faulty = True
        for eid in new:
            ent_row = self.rows.get(eid) or {}
            if ent_row.get("in_stock") is False:
                self.gone.add(eid)
                faulty = True
        self.absorb_lines()
        return not faulty

    def fallback(self) -> None:
        if self.finished or self.moves_remaining() <= 0:
            return
        fits = [r for r in self.fits() if r not in self.gone]
        if fits and self.moves_remaining() >= 3 and self.put_basket(fits[0]):
            self.order(fits[0])
        elif self.cart and self.tools.has("order"):
            self.order(next(iter(self.cart)))
        elif self.tools.has("decline"):
            self.decline("last resort")

    # ---- the whole episode
    def run(self) -> list[dict]:
        if self.finished:
            return self.exchanges
        if self.tools.wants_llm():
            role_map = self.reader.complete("Assign every shop tool its role. Copy tool and parameter names verbatim.",
                                       json.dumps([self.tools.schemas[n] for n in self.tools.schemas])[:12000], ROLE_FORM, 700) or {}
            self.tools.adopt_role_map(role_map)
        self.parse_request()
        self.read_allowance()
        self.plan_messages()
        if self.products:
            self.drive_several()
            return self.exchanges
        self.inspect()
        if self.ask_leftover():
            self.appraise(passes=1)
            self.dig(want=1)
        if self.objective:
            # (the best by a measure: more fits are read before choosing - the first found is seldom the best)
            self.dig(want=5)
            self.appraise(passes=2)
            best = self.fits()[:3]
            self.note(f"best by {self.objective['kind']} {self.objective['property']}: {[(r, self.objective_value(r)) for r in best]}")
        elif self.cheapest:
            self.lower_priced()
            self.dig(want=1)
            self.underbid()
        self.note(f"fits: {[(r, self.rows[r].get('price'), self.rank_of(r)) for r in self.fits()[:6]]} step={self.step}/{self.max_steps} "
                  f"cost={self.reader.spent:.4f} model_calls={self.reader.calls} used={self.reader.used} t={monotonic() - self.started:.0f}s "
                  f"store {sum(self.env_secs) / max(1, len(self.env_secs)):.1f}s/call model {sum(self.reader.secs) / max(1, len(self.reader.secs)):.1f}s/call "
                  f"skipped in a hurry {self.reader.skipped}")
        self.finish()
        return self.exchanges


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    session = Session(problem_data)
    try:
        return session.run()
    except Exception as exc:  # the dialogue so far is still returned, after a last finish attempt
        session.note(f"agent error: {type(exc).__name__}: {exc}")
        try:
            session.fallback()
        except Exception as tardy:
            session.note(f"crash during final close: {type(tardy).__name__}: {tardy}")
        return session.exchanges


__all__ = ["agent_main"]
